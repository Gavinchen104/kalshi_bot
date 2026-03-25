from src.types import MarketState


def has_acceptable_spread(state: MarketState, max_spread_cents: int) -> bool:
    if state.bid_cents is None or state.ask_cents is None:
        return False
    return (state.ask_cents - state.bid_cents) <= max_spread_cents


def has_min_depth(state: MarketState, min_depth: int) -> bool:
    return state.bid_size >= min_depth and state.ask_size >= min_depth

