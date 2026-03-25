from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import websockets

from src.api.auth import KalshiAuth
from src.monitoring.logging import get_logger
from src.types import MarketState


logger = get_logger("ws_client")


class KalshiWSClient:
    """Websocket client skeleton with reconnect loop."""

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        api_secret: str,
        market_ids: list[str] | None = None,
    ):
        self.ws_url = ws_url
        self.auth = KalshiAuth(api_key_id=api_key, private_key_raw=api_secret)
        self.market_ids = market_ids or []

    async def stream(self) -> AsyncIterator[MarketState]:
        while True:
            try:
                logger.info("ws_connecting", ws_url=self.ws_url, markets=self.market_ids)
                ws_headers = self.auth.build_headers("GET", self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    additional_headers=ws_headers,
                ) as ws:
                    logger.info("ws_connected")
                    await self._subscribe(ws)
                    async for message in ws:
                        state = self._parse_message(message)
                        if state:
                            yield state
            except Exception as exc:
                logger.warning("ws_reconnect", error=str(exc))
                await asyncio.sleep(2)

    async def preflight_auth(self, timeout_seconds: float = 8.0) -> None:
        ws_headers = self.auth.build_headers("GET", self.ws_url)
        async with websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=20,
            additional_headers=ws_headers,
            open_timeout=timeout_seconds,
        ) as ws:
            await ws.close()

    async def stream_from_jsonl(self, path: str, loop_forever: bool = True) -> AsyncIterator[MarketState]:
        stream_path = Path(path)
        if not stream_path.exists():
            raise FileNotFoundError(f"Mock stream not found: {path}")

        logger.info("mock_stream_start", path=path, loop_forever=loop_forever)
        while True:
            lines = stream_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                state = self._parse_message(line)
                if state:
                    yield state
                    await asyncio.sleep(0.2)
            if not loop_forever:
                break

    async def _subscribe(self, ws) -> None:
        params: dict = {"channels": ["ticker"]}
        if self.market_ids:
            params["market_tickers"] = self.market_ids
        payload = {
            "id": 1,
            "cmd": "subscribe",
            "params": params,
        }
        await ws.send(json.dumps(payload))

    def _parse_message(self, raw: str) -> MarketState | None:
        msg = json.loads(raw)
        data = msg.get("data")
        if not isinstance(data, dict):
            msg_type = msg.get("type")
            if msg_type != "ticker":
                return None
            payload = msg.get("msg")
            if not isinstance(payload, dict):
                return None
            data = {
                "market_ticker": payload.get("market_ticker"),
                "yes_bid": self._dollars_to_cents(payload.get("yes_bid_dollars")),
                "yes_ask": self._dollars_to_cents(payload.get("yes_ask_dollars")),
                "yes_bid_size": self._fp_to_int(payload.get("yes_bid_size_fp")),
                "yes_ask_size": self._fp_to_int(payload.get("yes_ask_size_fp")),
                "last_price": self._dollars_to_cents(payload.get("price_dollars")),
            }
        if not isinstance(data, dict):
            return None

        market_id = str(data.get("market_ticker", ""))
        if not market_id:
            return None

        bid = self._to_int_or_none(data.get("yes_bid"))
        ask = self._to_int_or_none(data.get("yes_ask"))
        bid_size = int(data.get("yes_bid_size", 0) or 0)
        ask_size = int(data.get("yes_ask_size", 0) or 0)
        last = self._to_int_or_none(data.get("last_price"))

        return MarketState(
            market_id=market_id,
            bid_cents=bid,
            ask_cents=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_trade_cents=last,
            updated_at=datetime.now(tz=timezone.utc),
        )

    @staticmethod
    def _to_int_or_none(value) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _dollars_to_cents(value) -> int | None:
        if value is None or value == "":
            return None
        return int(round(float(value) * 100))

    @staticmethod
    def _fp_to_int(value) -> int:
        if value is None or value == "":
            return 0
        return int(round(float(value)))

