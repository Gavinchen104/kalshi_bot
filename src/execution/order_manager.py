from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from src.api.kalshi_client import KalshiClient
from src.monitoring.logging import get_logger
from src.storage.repository import Repository
from src.types import MarketState, ProposedOrder


logger = get_logger("order_manager")


class PaperFillSimulator:
    """
    Simulates realistic fill behavior for paper trading.

    Fill probability model based on order aggressiveness vs the live quote:
      - At or through touch (aggressive):      ~97%
      - 1-2 cents inside touch (semi-passive): ~75%
      - 3-5 cents inside touch (passive):      ~40%
      - 6+ cents inside (deep resting):        ~10%

    Slippage: small adverse price impact added to the touch price.
    Fees: configurable bps of notional (applied on fill price × qty).
    """

    def __init__(
        self,
        fee_bps: int = 50,
        slippage_bps: int = 20,
        rng: random.Random | None = None,
    ) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._rng = rng or random.Random()

    def simulate(self, order: ProposedOrder, state: MarketState | None = None) -> dict:
        bid = state.bid_cents if state and state.bid_cents is not None else None
        ask = state.ask_cents if state and state.ask_cents is not None else None

        if self._rng.random() > self._fill_probability(order, bid, ask):
            return {
                "status": "paper_unfilled",
                "fill_price_cents": None,
                "fee_cents": 0,
                "slippage_cents": 0,
            }

        # Use touch price for fill realism (buyer pays ask, seller receives bid)
        if order.side == "yes" and ask is not None:
            base_price = ask
        elif order.side == "no" and bid is not None:
            base_price = 100 - bid
        else:
            base_price = order.price_cents

        slippage_per_unit = max(0, int(base_price * self.slippage_bps / 10000))
        fill_price = min(99, max(1, base_price + slippage_per_unit))
        fee_cents = max(0, int(fill_price * order.quantity * self.fee_bps / 10000))
        slippage_cents = slippage_per_unit * order.quantity

        return {
            "status": "paper_filled",
            "fill_price_cents": fill_price,
            "fee_cents": fee_cents,
            "slippage_cents": slippage_cents,
        }

    def _fill_probability(
        self, order: ProposedOrder, bid: int | None, ask: int | None
    ) -> float:
        if order.side == "yes":
            if ask is None:
                return 0.8
            distance = order.price_cents - ask  # positive = paying above ask
        else:
            if bid is None:
                return 0.8
            distance = bid - (100 - order.price_cents)

        if distance >= 0:
            return 0.97
        elif distance >= -2:
            return 0.75
        elif distance >= -5:
            return 0.40
        else:
            return 0.10


class OrderManager:
    def __init__(
        self,
        client: KalshiClient,
        paper_mode: bool = True,
        repository: Repository | None = None,
        min_order_interval_ms: int = 0,
        fee_bps: int = 0,
        slippage_bps: int = 0,
        fill_simulator: PaperFillSimulator | None = None,
    ):
        self.client = client
        self.paper_mode = paper_mode
        self.repository = repository
        self.min_order_interval_ms = min_order_interval_ms
        self._last_order_at: datetime | None = None
        self._fill_sim = fill_simulator or PaperFillSimulator(
            fee_bps=fee_bps, slippage_bps=slippage_bps
        )

    async def execute(
        self, order: ProposedOrder, state: MarketState | None = None
    ) -> dict:
        if self._is_rate_limited():
            response = {"status": "skipped_rate_limited", "order": order.__dict__}
            if self.repository:
                self.repository.save_order_event(
                    market_id=order.market_id,
                    side=order.side,
                    price_cents=order.price_cents,
                    quantity=order.quantity,
                    status="skipped_rate_limited",
                    response=response,
                )
            return response

        payload = {
            "market_ticker": order.market_id,
            "side": order.side,
            "price": order.price_cents,
            "count": order.quantity,
            "time_in_force": order.tif,
            "type": "limit",
        }

        if self.paper_mode:
            fill_result = self._fill_sim.simulate(order, state)
            response = {**fill_result, "order": payload}
            if self.repository:
                self.repository.save_order_event(
                    market_id=order.market_id,
                    side=order.side,
                    price_cents=order.price_cents,
                    quantity=order.quantity,
                    status=fill_result["status"],
                    response=response,
                )
            logger.info(
                "paper_order",
                status=fill_result["status"],
                fill_price=fill_result.get("fill_price_cents"),
                fee_cents=fill_result.get("fee_cents"),
                market_id=order.market_id,
            )
            self._last_order_at = datetime.now(tz=timezone.utc)
            return response

        logger.info("live_order_send", payload=payload)
        response = await self.client.place_order(payload)
        kalshi_order = response.get("order", {})
        order_id = str(kalshi_order.get("order_id") or kalshi_order.get("id") or "")
        if self.repository:
            self.repository.save_order_event(
                market_id=order.market_id,
                side=order.side,
                price_cents=order.price_cents,
                quantity=order.quantity,
                status="live_sent",
                response=response,
                order_id=order_id or None,
            )
        self._last_order_at = datetime.now(tz=timezone.utc)
        return response

    def _is_rate_limited(self) -> bool:
        if self.min_order_interval_ms <= 0 or self._last_order_at is None:
            return False
        return datetime.now(tz=timezone.utc) - self._last_order_at < timedelta(
            milliseconds=self.min_order_interval_ms
        )
