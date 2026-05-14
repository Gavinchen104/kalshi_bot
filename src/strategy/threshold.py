"""
Threshold strategy: enter long YES when the Kalshi YES ask is at or above
`entry_prob_threshold` (in cents, default 95) within the last
`last_minutes_window` minutes of a 15-minute window. Exit when the YES bid
drops to or below `exit_prob_threshold` (default 85).

Intended for paper-mode test runs only. Sizing aims for "all in" but is
ultimately clamped by `risk.max_order_size` and `risk.max_position_per_market`.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from src.api.binance_ws import BinanceDataStore
from src.config import StrategyConfig
from src.portfolio.pnl import PnLTracker
from src.strategy.filters import has_acceptable_spread, has_min_depth
from src.types import MarketState, ProposedOrder, Signal


class ThresholdStrategy:
    def __init__(
        self,
        config: StrategyConfig,
        btc_store: BinanceDataStore | None = None,
        bankroll_cents: int = 50_000,
    ):
        self.config = config
        self._market_re = re.compile(config.target_market_regex, re.IGNORECASE)
        self._btc_store = btc_store
        self._bankroll_cents = bankroll_cents

    def set_btc_store(self, store: BinanceDataStore) -> None:
        self._btc_store = store

    def set_model_quality(self, beats_baseline: bool) -> None:
        return

    def reload_model(self) -> bool:
        return False

    def compute_signal(self, state: MarketState) -> Signal | None:
        if not self._market_re.search(state.market_id):
            return None
        if not self._in_last_window(state.updated_at):
            return None
        if state.ask_cents is None or state.ask_cents < self.config.entry_prob_threshold:
            return None
        if not has_acceptable_spread(state, self.config.max_spread_cents):
            return None
        if not has_min_depth(state, self.config.min_top_book_depth):
            return None
        if self.config.require_coinbase_ready:
            if self._btc_store is None or not self._btc_store.is_ready:
                return None

        coinbase_px = self._btc_store.latest_price if self._btc_store else None
        return Signal(
            market_id=state.market_id,
            side="yes",
            edge_bps=max(1, (100 - state.ask_cents) * 100),
            fair_value_cents=state.ask_cents,
            reason=(
                f"threshold:ask>={self.config.entry_prob_threshold}c,"
                f"last_{self.config.last_minutes_window}m,"
                f"btc_spot={coinbase_px}"
            ),
            predicted_prob=state.ask_cents / 100.0,
            model_name="threshold",
        )

    def kelly_size(self, signal: Signal, market_price_cents: int) -> int:
        if market_price_cents <= 0:
            return 1
        afford = self._bankroll_cents // market_price_cents
        cap = max(1, getattr(self.config, "max_kelly_contracts", 1))
        return max(1, min(afford, cap))

    def _in_last_window(self, when: datetime) -> bool:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        window = max(1, self.config.window_minutes)
        last = max(1, self.config.last_minutes_window)
        return (when.minute % window) >= (window - last)


class ThresholdExitManager:
    """Exit by selling YES (i.e. crossing into the NO side) when bid drops to threshold."""

    def __init__(self, exit_prob_threshold: int = 85) -> None:
        self.exit_prob_threshold = exit_prob_threshold

    def check_exits(
        self,
        state: MarketState,
        pnl_tracker: PnLTracker,
    ) -> list[ProposedOrder]:
        pos = pnl_tracker.ledger.position_for(state.market_id)
        if pos is None or pos.is_flat:
            return []
        if pos.net_quantity <= 0:
            return []
        if state.bid_cents is None or state.bid_cents > self.exit_prob_threshold:
            return []

        qty = pos.net_quantity
        price_cents = max(1, min(99, 100 - state.bid_cents))
        return [
            ProposedOrder(
                market_id=state.market_id,
                side="no",
                price_cents=price_cents,
                quantity=qty,
                tif="GTC",
            )
        ]
