"""
Fractional Kelly criterion for binary-option position sizing.

For a YES contract at market price *p* (implied probability):
  - win payout per contract = (100 - p) cents
  - loss per contract       = p cents
  - full Kelly fraction     = (model_prob - implied) / (1 - implied)

We default to **half-Kelly** to reduce variance at the cost of slightly
lower expected growth.
"""
from __future__ import annotations


def kelly_fraction(
    model_prob: float,
    market_implied_prob: float,
    kelly_scale: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    if model_prob <= market_implied_prob or market_implied_prob >= 0.99:
        return 0.0
    full = (model_prob - market_implied_prob) / (1.0 - market_implied_prob)
    return min(max(full * kelly_scale, 0.0), max_fraction)


def kelly_contracts(
    model_prob: float,
    market_price_cents: int,
    bankroll_cents: int,
    min_contracts: int = 1,
    max_contracts: int = 10,
    kelly_scale: float = 0.5,
) -> int:
    """Convert a Kelly fraction into a concrete number of contracts to buy."""
    implied = market_price_cents / 100.0
    frac = kelly_fraction(model_prob, implied, kelly_scale=kelly_scale)
    if frac <= 0:
        return 0
    bet_cents = frac * bankroll_cents
    contracts = int(bet_cents / max(market_price_cents, 1))
    if contracts < min_contracts:
        return 0
    return min(contracts, max_contracts)
