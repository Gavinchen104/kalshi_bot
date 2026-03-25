from src.types import MarketState, ProposedOrder


def estimate_slippage_cents(order: ProposedOrder, state: MarketState) -> int:
    _ = order
    if state.bid_cents is None or state.ask_cents is None:
        return 0
    return max(0, state.ask_cents - state.bid_cents)

