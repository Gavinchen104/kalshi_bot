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


class KalshiWSClient:
    def __init__(
        self,
        ws_url: str,
        api_key: str,
        api_secret: str,
        ticker_regex: str = "^KXBTC.*",
    ):
        self.ws_url = ws_url
        self.auth = KalshiAuth(api_key_id=api_key, private_key_raw=api_secret)
        self.ticker_re = re.compile(ticker_regex)

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
                    async for raw in ws:
                        state = self._parse(raw)
                        if state and self.ticker_re.match(state.market_id):
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
        payload = {"id": 1, "cmd": "subscribe", "params": {"channels": ["ticker"]}}
        await ws.send(json.dumps(payload))

    def _parse(self, raw: str) -> MarketState | None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if msg.get("type") != "ticker":
            return None
        body = msg.get("msg")
        if not isinstance(body, dict):
            return None

        market_id = str(body.get("market_ticker", ""))
        if not market_id:
            return None

        bid = _dollars_to_cents(body.get("yes_bid_dollars"))
        ask = _dollars_to_cents(body.get("yes_ask_dollars"))
        bid_sz = _fp_to_int(body.get("yes_bid_size_fp"))
        ask_sz = _fp_to_int(body.get("yes_ask_size_fp"))
        last = _dollars_to_cents(body.get("price_dollars"))

        return MarketState(
            market_id=market_id,
            bid_cents=bid,
            ask_cents=ask,
            bid_size=bid_sz,
            ask_size=ask_sz,
            last_trade_cents=last,
            updated_at=datetime.now(tz=timezone.utc),
        )


def _dollars_to_cents(value) -> int | None:
    if value is None or value == "":
        return None
    return int(round(float(value) * 100))


def _fp_to_int(value) -> int:
    if value is None or value == "":
        return 0
    return int(round(float(value)))
