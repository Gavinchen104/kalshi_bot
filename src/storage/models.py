from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderEvent:
    order_id: str
    market_id: str
    status: str
    created_at: datetime

