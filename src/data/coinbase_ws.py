"""Coinbase BTC-USD ticker WS feed + REST history bootstrap + 1-min candle ring buffer."""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

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


_COINBASE_MAX_CANDLES = 300  # Exchange API hard cap of candles per request


def _rows_to_candles(data) -> list[Candle]:
    """Parse Coinbase Exchange candle rows [time, low, high, open, close, vol]."""
    out: list[Candle] = []
    for row in data:
        out.append(Candle(
            timestamp_ms=int(row[0]) * 1000,
            low=float(row[1]), high=float(row[2]),
            open=float(row[3]), close=float(row[4]),
            volume=float(row[5]), is_closed=True,
        ))
    return out


def _http_window_fetcher(rest_url: str, product_id: str):
    async def _fetch(start_iso: str, end_iso: str):
        url = f"{rest_url}?granularity=60&start={start_iso}&end={end_iso}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    return _fetch


async def backfill_range(
    rest_url: str,
    product_id: str,
    start_ms: int,
    end_ms: int,
    *,
    pause_s: float = 0.2,
    _window_fetcher=None,
) -> list[Candle]:
    """Backfill 1-minute candles over [start_ms, end_ms) via paginated Coinbase
    REST.

    The Exchange candles endpoint returns at most 300 candles per request, so
    the range is walked in 300-minute windows. Returns candles sorted ascending
    and deduped by timestamp (windows are half-open but a 1-candle overlap from
    Coinbase rounding is harmless).

    `_window_fetcher(start_iso, end_iso) -> list[row]` is injectable so the
    pagination/dedupe logic is unit-testable without HTTP. `pause_s` throttles
    real requests to the public endpoint (tests pass 0.0).
    """
    if end_ms <= start_ms:
        return []

    fetcher = _window_fetcher or _http_window_fetcher(rest_url, product_id)
    window_ms = _COINBASE_MAX_CANDLES * 60_000
    by_ts: dict[int, Candle] = {}
    cur = start_ms
    while cur < end_ms:
        win_end = min(cur + window_ms, end_ms)
        start_iso = datetime.fromtimestamp(cur / 1000, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(win_end / 1000, tz=timezone.utc).isoformat()
        rows = await fetcher(start_iso, end_iso)
        for c in _rows_to_candles(rows):
            by_ts[c.timestamp_ms] = c
        cur = win_end
        if pause_s:
            await asyncio.sleep(pause_s)
    return sorted(by_ts.values(), key=lambda c: c.timestamp_ms)


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
