"""
Edge strategy: compare our pricer's probability against Kalshi's market price.
Signal only when the disagreement is large enough to overcome fees+slippage+safety margin.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import StrategyConfig
from src.types import MarketState, ProbEstimate, Signal


class EdgeStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def evaluate(self, est: ProbEstimate, state: MarketState) -> Signal | None:
        if state.bid_cents is None or state.ask_cents is None:
            return None
        if (state.ask_cents - state.bid_cents) > self.config.max_spread_cents:
            return None
        if state.bid_size < self.config.min_top_book_depth or state.ask_size < self.config.min_top_book_depth:
            return None
        if est.horizon_seconds < self.config.min_horizon_seconds:
            return None
        if est.horizon_seconds > self.config.max_horizon_seconds:
            return None

        our_prob = est.prob
        market_mid_prob = (state.bid_cents + state.ask_cents) / 200.0  # midpoint as Kalshi's prob

        # Buy YES when our_prob >> ask/100 (Kalshi is selling too cheaply)
        yes_edge = our_prob - state.ask_cents / 100.0
        # Buy NO when (1 - our_prob) >> (100 - bid)/100, i.e. our_prob << bid/100
        no_edge = state.bid_cents / 100.0 - our_prob

        if yes_edge >= self.config.edge_threshold:
            return Signal(
                market_id=est.market_id, side="yes",
                our_prob=our_prob, market_prob=market_mid_prob,
                edge=yes_edge, fair_price_cents=int(round(our_prob * 100)),
                reason=f"yes:our={our_prob:.3f}>ask={state.ask_cents}c",
            )
        if no_edge >= self.config.edge_threshold:
            return Signal(
                market_id=est.market_id, side="no",
                our_prob=our_prob, market_prob=market_mid_prob,
                edge=no_edge, fair_price_cents=int(round((1 - our_prob) * 100)),
                reason=f"no:our={our_prob:.3f}<bid={state.bid_cents}c",
            )
        return None
