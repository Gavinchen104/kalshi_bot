from __future__ import annotations


def apply_slippage(intended_price_cents: int, side: str, slippage_bps: int) -> int:
    """Worsen the fill price by `slippage_bps` basis points of $1 (i.e. cents)."""
    cents_delta = max(1, int(round(slippage_bps / 100)))
    if side == "yes":
        return min(99, intended_price_cents + cents_delta)
    return min(99, intended_price_cents + cents_delta)


def fee_for(quantity: int, price_cents: int, fee_bps: int) -> int:
    """Simple maker/taker fee model: fee_bps applied to notional in cents."""
    notional = quantity * price_cents
    return max(0, int(round(notional * fee_bps / 10_000)))
