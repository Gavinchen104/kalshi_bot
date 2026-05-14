from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.config import RiskConfig
from src.risk.kill_switch import KillSwitch
from src.types import MarketState, ProposedOrder


@dataclass
class RiskResult:
    allowed: bool
    reason: str = ""


class RiskEngine:
    def __init__(self, config: RiskConfig, kill_switch: KillSwitch):
        self.config = config
        self.kill_switch = kill_switch
        self.session_pnl_cents = 0
        self.peak_pnl_cents = 0
        self._order_times: list[datetime] = []
        self._positions: dict[str, int] = {}
        self._gross: int = 0

    def update_session_pnl(self, pnl_cents: int) -> None:
        self.session_pnl_cents = pnl_cents
        if pnl_cents > self.peak_pnl_cents:
            self.peak_pnl_cents = pnl_cents

    def apply_fill(self, market_id: str, signed_qty: int) -> None:
        self._positions[market_id] = self._positions.get(market_id, 0) + signed_qty
        self._gross = sum(abs(q) for q in self._positions.values())

    def validate(self, order: ProposedOrder, state: MarketState) -> RiskResult:
        if self.kill_switch.engaged:
            return RiskResult(False, f"kill_switch:{self.kill_switch.reason}")
        if order.price_cents < 1 or order.price_cents > 99:
            return RiskResult(False, "invalid_price")
        if order.quantity <= 0:
            return RiskResult(False, "invalid_qty")
        if self._stale(state):
            return RiskResult(False, "stale_data")
        if self._rate_limited():
            return RiskResult(False, "rate_limit")
        if self._breach_market_position(order):
            return RiskResult(False, "max_position_per_market")
        if self._breach_gross(order):
            return RiskResult(False, "max_gross_exposure")
        if self.session_pnl_cents <= -self.config.max_daily_loss_cents:
            self.kill_switch.engage("daily_loss_limit")
            return RiskResult(False, "daily_loss_limit")
        if (self.peak_pnl_cents - self.session_pnl_cents) >= self.config.max_drawdown_cents:
            self.kill_switch.engage("drawdown_limit")
            return RiskResult(False, "drawdown_limit")
        self._record_time()
        return RiskResult(True)

    def _breach_market_position(self, order: ProposedOrder) -> bool:
        signed = order.quantity if order.side == "yes" else -order.quantity
        projected = self._positions.get(order.market_id, 0) + signed
        return abs(projected) > self.config.max_position_per_market

    def _breach_gross(self, order: ProposedOrder) -> bool:
        return (self._gross + order.quantity) > self.config.max_gross_exposure

    def _stale(self, state: MarketState) -> bool:
        age = datetime.now(tz=timezone.utc) - state.updated_at
        return age.total_seconds() > self.config.max_data_age_seconds

    def _rate_limited(self) -> bool:
        if self.config.max_orders_per_minute <= 0:
            return False
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        self._order_times = [t for t in self._order_times if t > cutoff]
        return len(self._order_times) >= self.config.max_orders_per_minute

    def _record_time(self) -> None:
        self._order_times.append(datetime.now(tz=timezone.utc))
