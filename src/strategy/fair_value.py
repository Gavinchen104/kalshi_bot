from __future__ import annotations

from src.types import MarketState


def estimate_fair_value_cents(state: MarketState) -> int | None:
    if state.bid_cents is None and state.ask_cents is None:
        return state.last_trade_cents
    if state.bid_cents is None:
        return state.ask_cents
    if state.ask_cents is None:
        return state.bid_cents
    return (state.bid_cents + state.ask_cents) // 2

