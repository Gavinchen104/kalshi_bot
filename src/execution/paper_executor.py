"""Simulates fills against the visible top-of-book. No live order placement here — research bench."""
from __future__ import annotations

import time

from src.config import ExecutionConfig
from src.execution.slippage import apply_slippage, fee_for
from src.monitoring.logging import get_logger
from src.types import MarketState, ProposedOrder


logger = get_logger("paper_exec")


class PaperExecutor:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self._last_order_ts_ms: float = 0.0

    def execute(self, order: ProposedOrder, state: MarketState) -> dict:
        now_ms = time.time() * 1000
        if now_ms - self._last_order_ts_ms < self.config.min_order_interval_ms:
            return {"status": "skipped_rate_limited"}
        self._last_order_ts_ms = now_ms

        # Fill if the resting book crosses our limit.
        if order.side == "yes":
            if state.ask_cents is None or state.ask_cents > order.price_cents:
                return {"status": "paper_unfilled"}
            intended = state.ask_cents
        else:
            if state.bid_cents is None or state.bid_cents < (100 - order.price_cents):
                return {"status": "paper_unfilled"}
            intended = 100 - state.bid_cents

        fill_price = apply_slippage(intended, order.side, self.config.slippage_bps)
        fee = fee_for(order.quantity, fill_price, self.config.fee_bps)
        logger.info(
            "paper_fill",
            market_id=order.market_id, side=order.side,
            qty=order.quantity, fill_price=fill_price, fee=fee,
        )
        return {
            "status": "paper_filled",
            "fill_price_cents": fill_price,
            "fee_cents": fee,
        }
