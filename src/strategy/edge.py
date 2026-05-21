"""
Edge strategy: compare our pricer's probability against Kalshi's market price.
Signal only when the disagreement is large enough to overcome fees+slippage+safety margin.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import StrategyConfig
from src.pricing.ticker import parse_ticker
from src.types import MarketState, ProbEstimate, Signal


class EdgeStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _near_strike(self, market_id: str, spot_usd: float) -> bool:
        """A1 guard: True when |spot − strike| (or |spot − bracket mid|)
        is within ``near_strike_guard_usd``. 0 disables the guard."""
        band = self.config.near_strike_guard_usd
        if band <= 0:
            return False
        terms = parse_ticker(market_id, settlement_mode=True)
        if terms is None:
            return False  # unparsable ticker → don't block trading
        if terms.direction == "above":
            ref = terms.strike_usd
        else:
            if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
                return False
            ref = (terms.bracket_low_usd + terms.bracket_high_usd) / 2.0
        if ref is None:
            return False
        return abs(spot_usd - ref) < band

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
        # A1 near-strike guard: drop signals where |spot − strike| is within
        # the measured Coinbase-vs-Kalshi-source risk band.
        if self._near_strike(est.market_id, est.spot_usd):
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
