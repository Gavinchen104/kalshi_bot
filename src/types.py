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
    """Parsed Kalshi BTC 15m contract metadata."""
    market_id: str
    strike_usd: float
    close_time: datetime
    direction: Literal["above", "below"]


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
