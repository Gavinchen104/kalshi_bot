"""Coinbase BTC-USD ticker WS feed + REST history bootstrap + 1-min candle ring buffer."""
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


logger = get_logger("coinbase_ws")


@dataclass
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


class CoinbaseStore:
    def __init__(self, max_candles: int = 500):
        self._candles: deque[Candle] = deque(maxlen=max_candles)
        self._current: Candle | None = None
        self._latest_price: float | None = None
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def latest_price(self) -> float | None:
        return self._latest_price

    @property
    def candle_count(self) -> int:
        return len(self._candles)

    async def wait_ready(self, timeout: float = 60.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def load_history(self, candles: list[Candle]) -> None:
        async with self._lock:
            for c in candles:
                self._candles.append(c)
            if candles:
                self._latest_price = candles[-1].close
            if len(self._candles) >= 30:
                self._ready.set()

    async def on_tick(self, price: float, volume: float = 0.0) -> None:
        async with self._lock:
            self._latest_price = price
            now_ms = int(time.time() * 1000)
            minute_ms = (now_ms // 60_000) * 60_000
            if self._current is None or self._current.timestamp_ms != minute_ms:
                if self._current is not None:
                    self._current.is_closed = True
                    self._candles.append(self._current)
                self._current = Candle(minute_ms, price, price, price, price, volume, is_closed=False)
            else:
                c = self._current
                c.close = price
                c.high = max(c.high, price)
                c.low = min(c.low, price)
                c.volume += volume
            if not self._ready.is_set() and len(self._candles) >= 30:
                self._ready.set()

    def closes(self, n: int | None = None) -> np.ndarray:
        rows = [c.close for c in self._candles if c.is_closed]
        if n is not None:
            rows = rows[-n:]
        return np.array(rows, dtype=float)

    def closed_candle_dicts(self) -> list[dict]:
        return [
            {
                "timestamp_ms": c.timestamp_ms,
                "open": c.open, "high": c.high, "low": c.low, "close": c.close,
                "volume": c.volume,
            }
            for c in self._candles
            if c.is_closed
        ]


async def fetch_history(rest_url: str, product_id: str, limit: int = 300) -> list[Candle]:
    """Pull recent 1-minute candles from Coinbase Exchange REST."""
    url = f"{rest_url}?granularity=60"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
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
    return candles[-limit:] if limit else candles


async def run_ws(store: CoinbaseStore, ws_url: str, product_id: str = "BTC-USD") -> None:
    subscribe = json.dumps({"type": "subscribe", "product_ids": [product_id], "channels": ["ticker"]})
    while True:
        try:
            logger.info("coinbase_connecting", url=ws_url)
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(subscribe)
                logger.info("coinbase_connected")
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "ticker":
                        continue
                    try:
                        await store.on_tick(float(msg["price"]), float(msg.get("last_size", 0)))
                    except (KeyError, ValueError):
                        continue
        except Exception as exc:
            logger.warning("coinbase_reconnect", error=str(exc))
            await asyncio.sleep(3)
