"""
BTC/USD spot price feed via Coinbase public API (no auth, no geo-restrictions).

- Historical 1-minute candles fetched from Coinbase REST on startup
- Real-time price updates streamed via Coinbase WebSocket ticker channel
- 1-minute candles are built in-memory from the ticker stream
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass

import httpx
import numpy as np
import websockets

from src.monitoring.logging import get_logger

logger = get_logger("btc_spot")

COINBASE_REST_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"


@dataclass
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


class BinanceDataStore:
    """In-memory ring buffer of BTC/USD 1-minute candles."""

    def __init__(self, max_candles: int = 500):
        self._candles: deque[Candle] = deque(maxlen=max_candles)
        self._latest_price: float | None = None
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._current_candle: Candle | None = None

    @property
    def latest_price(self) -> float | None:
        return self._latest_price

    @property
    def candle_count(self) -> int:
        return len(self._candles)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    async def on_price(self, price: float, volume: float = 0.0) -> None:
        """Called on every real-time price tick from WebSocket."""
        async with self._lock:
            self._latest_price = price
            now_ms = int(time.time() * 1000)
            minute_ms = (now_ms // 60_000) * 60_000

            if self._current_candle is None or self._current_candle.timestamp_ms != minute_ms:
                if self._current_candle is not None:
                    self._current_candle.is_closed = True
                    self._candles.append(self._current_candle)
                self._current_candle = Candle(
                    timestamp_ms=minute_ms,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                    is_closed=False,
                )
            else:
                c = self._current_candle
                c.close = price
                c.high = max(c.high, price)
                c.low = min(c.low, price)
                c.volume += volume

            if not self._ready.is_set() and len(self._candles) >= 30:
                self._ready.set()

    async def update_candle(self, candle: Candle) -> None:
        async with self._lock:
            if self._candles and self._candles[-1].timestamp_ms == candle.timestamp_ms:
                self._candles[-1] = candle
            else:
                self._candles.append(candle)
            self._latest_price = candle.close
            if not self._ready.is_set() and len(self._candles) >= 30:
                self._ready.set()

    async def load_history(self, candles: list[Candle]) -> None:
        async with self._lock:
            for c in candles:
                self._candles.append(c)
            if candles:
                self._latest_price = candles[-1].close
            if len(self._candles) >= 30:
                self._ready.set()
        logger.info("btc_history_loaded", candle_count=len(candles))

    async def wait_ready(self, timeout: float = 60.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_ohlcv(self, n: int | None = None) -> np.ndarray:
        candles = [c for c in self._candles if c.is_closed]
        if n is not None:
            candles = candles[-n:]
        if not candles:
            return np.empty((0, 5), dtype=float)
        return np.array(
            [[c.open, c.high, c.low, c.close, c.volume] for c in candles],
            dtype=float,
        )

    def get_closes(self, n: int | None = None) -> np.ndarray:
        candles = [c for c in self._candles if c.is_closed]
        if n is not None:
            candles = candles[-n:]
        return np.array([c.close for c in candles], dtype=float)

    def get_volumes(self, n: int | None = None) -> np.ndarray:
        candles = [c for c in self._candles if c.is_closed]
        if n is not None:
            candles = candles[-n:]
        return np.array([c.volume for c in candles], dtype=float)

    def closed_candle_dicts(self) -> list[dict]:
        return [
            {
                "timestamp_ms": c.timestamp_ms,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in self._candles
            if c.is_closed
        ]


# ---------------------------------------------------------------------------
# Coinbase REST – fetch historical 1m candles (public, no auth)
# ---------------------------------------------------------------------------

async def fetch_historical_klines(
    symbol: str = "BTC-USD",
    interval: str = "1m",
    limit: int = 300,
) -> list[Candle]:
    """Fetch recent 1-minute candles from Coinbase Exchange REST API."""
    granularity = 60
    url = f"{COINBASE_REST_CANDLES}?granularity={granularity}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    candles: list[Candle] = []
    for row in data:
        candles.append(
            Candle(
                timestamp_ms=int(row[0]) * 1000,
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                is_closed=True,
            )
        )

    candles.sort(key=lambda c: c.timestamp_ms)
    if limit:
        candles = candles[-limit:]
    return candles


# ---------------------------------------------------------------------------
# Coinbase WebSocket – real-time ticker
# ---------------------------------------------------------------------------

async def run_binance_ws(
    store: BinanceDataStore,
    ws_url: str = COINBASE_WS_URL,
) -> None:
    """Long-running task: streams BTC/USD ticker from Coinbase into *store*."""
    subscribe_msg = json.dumps(
        {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        }
    )

    while True:
        try:
            logger.info("btc_ws_connecting", url=ws_url)
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(subscribe_msg)
                logger.info("btc_ws_connected")
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    try:
                        price = float(msg["price"])
                        volume = float(msg.get("last_size", 0))
                        await store.on_price(price, volume)
                    except (KeyError, ValueError):
                        continue
        except Exception as exc:
            logger.warning("btc_ws_reconnect", error=str(exc))
            await asyncio.sleep(3)
