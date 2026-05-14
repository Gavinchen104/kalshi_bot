from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["yes", "no"]


@dataclass(frozen=True)
class MarketState:
    """Top-of-book snapshot of a Kalshi market at one instant."""
    market_id: str
    bid_cents: int | None
    ask_cents: int | None
    bid_size: int
    ask_size: int
    last_trade_cents: int | None
    updated_at: datetime


@dataclass(frozen=True)
class ContractTerms:
    """Parsed Kalshi BTC contract metadata.

    direction:
      "above"   — pays $1 if BTC_T > strike_usd at close_time. (KXBTCD-*-T<strike>)
      "bracket" — pays $1 if bracket_low_usd <= BTC_T < bracket_high_usd. (KXBTC-*-B<low>)
    """
    market_id: str
    close_time: datetime
    direction: Literal["above", "bracket"]
    strike_usd: float | None = None
    bracket_low_usd: float | None = None
    bracket_high_usd: float | None = None


@dataclass(frozen=True)
class ProbEstimate:
    """A pricer's view of the YES probability for a market."""
    market_id: str
    prob: float
    horizon_seconds: float
    spot_usd: float
    vol_annualized: float
    source: str
    computed_at: datetime


@dataclass(frozen=True)
class Signal:
    market_id: str
    side: Side
    our_prob: float
    market_prob: float
    edge: float
    fair_price_cents: int
    reason: str


@dataclass(frozen=True)
class ProposedOrder:
    market_id: str
    side: Side
    price_cents: int
    quantity: int
    tif: str = "GTC"
