"""Kalshi WS v2 client.

Subscribes to the `orderbook_delta` channel (which Kalshi uses to deliver both
`orderbook_snapshot` and `orderbook_delta` frames). For each subscribed market
we maintain an in-memory book and emit a MarketState whenever it changes.

Why not `ticker`? Empirically the v2 ticker channel acks the subscribe but
sends no frames (at least for BTC markets at the time of writing).
`orderbook_delta` is the canonical price feed.

Book semantics on Kalshi binary markets:
  - `yes` = YES bids (people willing to buy YES at price p, in dollars 0..1)
  - `no`  = NO bids (people willing to buy NO at price p, in dollars 0..1)
  - YES ask = 1 - max(no_bids); the synthetic offer to sell YES is to buy NO.
  - YES bid = max(yes_bids).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import websockets

from src.data.kalshi_auth import KalshiAuth
from src.monitoring.logging import get_logger
from src.types import MarketState


logger = get_logger("kalshi_ws")


class _Book:
    """Per-market book. Stores YES-bid and NO-bid sides keyed by integer cents.

    Prices on the wire are dollar strings like "0.0100" (1 cent). We store as
    int cents to keep dict keys exact.
    """
    __slots__ = ("yes", "no")

    def __init__(self) -> None:
        self.yes: dict[int, float] = {}
        self.no: dict[int, float] = {}

    def reset_from_snapshot(self, yes_levels: list, no_levels: list) -> None:
        self.yes = self._levels_to_dict(yes_levels)
        self.no = self._levels_to_dict(no_levels)

    @staticmethod
    def _levels_to_dict(levels: list | None) -> dict[int, float]:
        out: dict[int, float] = {}
        if not levels:
            return out
        for entry in levels:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                price_cents = int(round(float(entry[0]) * 100))
                size = float(entry[1])
            except (TypeError, ValueError):
                continue
            if size > 0:
                out[price_cents] = size
        return out

    def apply_delta(self, side: str, price_cents: int, size: float) -> None:
        target = self.yes if side == "yes" else self.no
        if size <= 0:
            target.pop(price_cents, None)
        else:
            target[price_cents] = size

    def top(self) -> tuple[int | None, int | None, int, int]:
        """Return (yes_bid_cents, yes_ask_cents, yes_bid_size, yes_ask_size)."""
        yes_bid = max(self.yes) if self.yes else None
        best_no_bid = max(self.no) if self.no else None
        yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
        yes_bid_size = int(self.yes.get(yes_bid, 0.0)) if yes_bid is not None else 0
        yes_ask_size = int(self.no.get(best_no_bid, 0.0)) if best_no_bid is not None else 0
        return yes_bid, yes_ask, yes_bid_size, yes_ask_size


class KalshiWSClient:
    def __init__(
        self,
        ws_url: str,
        api_key: str,
        api_secret: str,
        ticker_regex: str = "^KXBTC.*",
        market_tickers: list[str] | None = None,
    ):
        self.ws_url = ws_url
        self.auth = KalshiAuth(api_key_id=api_key, private_key_raw=api_secret)
        self.ticker_re = re.compile(ticker_regex)
        self.market_tickers = market_tickers or []
        self._books: dict[str, _Book] = {}

    async def stream(self) -> AsyncIterator[MarketState]:
        while True:
            try:
                logger.info("ws_connecting", url=self.ws_url)
                headers = self.auth.build_headers("GET", self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    additional_headers=headers,
                ) as ws:
                    await self._subscribe(ws)
                    logger.info("ws_connected")
                    snapshots_seen = 0
                    async for raw in ws:
                        state = self._handle(raw)
                        if state is None:
                            continue
                        if self.ticker_re.match(state.market_id):
                            if snapshots_seen == 0:
                                logger.info("ws_first_state", market_id=state.market_id)
                            snapshots_seen += 1
                            yield state
            except Exception as exc:
                logger.warning("ws_reconnect", error=str(exc))
                await asyncio.sleep(2)

    async def preflight(self, timeout: float = 8.0) -> None:
        headers = self.auth.build_headers("GET", self.ws_url)
        async with websockets.connect(
            self.ws_url, additional_headers=headers, open_timeout=timeout
        ) as ws:
            await ws.close()

    async def _subscribe(self, ws) -> None:
        params: dict = {"channels": ["orderbook_delta"]}
        if self.market_tickers:
            params["market_tickers"] = self.market_tickers
        payload = {"id": 1, "cmd": "subscribe", "params": params}
        await ws.send(json.dumps(payload))
        logger.info("ws_subscribe", n_markets=len(self.market_tickers))

    def _handle(self, raw: str) -> MarketState | None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return None
        msg_type = msg.get("type")
        body = msg.get("msg")
        if not isinstance(body, dict):
            return None

        if msg_type == "orderbook_snapshot":
            return self._on_snapshot(body)
        if msg_type == "orderbook_delta":
            return self._on_delta(body)
        return None

    def _on_snapshot(self, body: dict) -> MarketState | None:
        market_id = str(body.get("market_ticker", ""))
        if not market_id:
            return None
        book = self._books.setdefault(market_id, _Book())
        yes_levels = body.get("yes_dollars_fp") or body.get("yes") or []
        no_levels = body.get("no_dollars_fp") or body.get("no") or []
        book.reset_from_snapshot(yes_levels, no_levels)
        return self._top_to_state(market_id, book)

    def _on_delta(self, body: dict) -> MarketState | None:
        market_id = str(body.get("market_ticker", ""))
        if not market_id:
            return None
        book = self._books.get(market_id)
        if book is None:
            # Late delta with no prior snapshot — wait for the next snapshot.
            return None
        try:
            price = body.get("price_dollars")
            if price is None:
                price = body.get("price")
            price_cents = int(round(float(price) * 100)) if price is not None else None
            side = body.get("side", "")
            delta = body.get("delta")
            if delta is None:
                # Some deltas report absolute size in `size`.
                size_abs = body.get("size")
                if size_abs is None or price_cents is None or side not in ("yes", "no"):
                    return None
                book.apply_delta(side, price_cents, float(size_abs))
            else:
                if price_cents is None or side not in ("yes", "no"):
                    return None
                current = (book.yes if side == "yes" else book.no).get(price_cents, 0.0)
                book.apply_delta(side, price_cents, current + float(delta))
        except (TypeError, ValueError):
            return None
        return self._top_to_state(market_id, book)

    def _top_to_state(self, market_id: str, book: _Book) -> MarketState:
        yb, ya, yb_sz, ya_sz = book.top()
        return MarketState(
            market_id=market_id,
            bid_cents=yb,
            ask_cents=ya,
            bid_size=yb_sz,
            ask_size=ya_sz,
            last_trade_cents=None,
            updated_at=datetime.now(tz=timezone.utc),
        )
