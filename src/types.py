from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Side = Literal["yes", "no"]


@dataclass
class MarketState:
    market_id: str
    bid_cents: int | None
    ask_cents: int | None
    bid_size: int
    ask_size: int
    last_trade_cents: int | None
    updated_at: datetime


@dataclass
class Signal:
    market_id: str
    side: Side
    edge_bps: int
    fair_value_cents: int
    reason: str
    predicted_prob: float | None = None
    model_name: str | None = None


@dataclass
class ProposedOrder:
    market_id: str
    side: Side
    price_cents: int
    quantity: int
    tif: str = "GTC"

