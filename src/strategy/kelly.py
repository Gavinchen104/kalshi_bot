"""Fractional Kelly sizing for binary contracts priced in cents."""
from __future__ import annotations


def kelly_contracts(
    our_prob: float,
    price_cents: int,
    bankroll_cents: int,
    kelly_fraction: float = 0.25,
    max_contracts: int = 5,
) -> int:
    """
    For a YES contract bought at p (cents), payoff is (100 - p) on win, -p on loss.
    Kelly fraction = (our_prob * (100 - p) - (1 - our_prob) * p) / ((100 - p) * p / 100)

    We apply a `kelly_fraction` multiplier (default quarter Kelly) for safety.
    """
    if not (0.0 < our_prob < 1.0):
        return 0
    if price_cents <= 0 or price_cents >= 100:
        return 0
    p = price_cents / 100.0
    edge = our_prob - p
    if edge <= 0:
        return 0
    b = (1 - p) / p  # odds offered
    f_full = edge / (b * p)
    f = max(0.0, kelly_fraction * f_full)
    qty = int((bankroll_cents * f) // price_cents)
    return max(0, min(qty, max_contracts))
