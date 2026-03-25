from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.config import RiskConfig
from src.risk.exposure import ExposureBook
from src.risk.kill_switch import KillSwitch
from src.types import MarketState, ProposedOrder


@dataclass
class RiskResult:
    allowed: bool
    reason: str = ""


class RiskEngine:
    def __init__(self, config: RiskConfig, exposure: ExposureBook, kill_switch: KillSwitch):
        self.config = config
        self.exposure = exposure
        self.kill_switch = kill_switch
        self.session_pnl_cents = 0
        self.peak_pnl_cents = 0
        self._order_timestamps: list[datetime] = field(default_factory=list) if False else []

    def validate(self, order: ProposedOrder, state: MarketState) -> RiskResult:
        if self.kill_switch.engaged:
            return RiskResult(False, f"kill_switch:{self.kill_switch.reason}")
        if order.price_cents < 1 or order.price_cents > 99:
            return RiskResult(False, "invalid_price")
        if order.quantity > self.config.max_order_size:
            return RiskResult(False, "max_order_size")
        if self._would_breach_market_position(order):
            return RiskResult(False, "max_position_per_market")
        if self._would_breach_gross_exposure(order):
            return RiskResult(False, "max_gross_exposure")
        if self._is_stale(state):
            return RiskResult(False, "stale_market_data")
        if self._is_per_minute_rate_limited():
            return RiskResult(False, "max_orders_per_minute")
        if self.session_pnl_cents <= -self.config.max_daily_loss_cents:
            self.kill_switch.engage("daily_loss_limit")
            return RiskResult(False, "daily_loss_limit")
        if (self.peak_pnl_cents - self.session_pnl_cents) >= self.config.max_drawdown_cents:
            self.kill_switch.engage("drawdown_limit")
            return RiskResult(False, "drawdown_limit")

        self._record_order_timestamp()
        return RiskResult(True)

    def update_session_pnl(self, pnl_cents: int) -> None:
        self.session_pnl_cents = pnl_cents
        if pnl_cents > self.peak_pnl_cents:
            self.peak_pnl_cents = pnl_cents

    def _would_breach_market_position(self, order: ProposedOrder) -> bool:
        signed_qty = order.quantity if order.side == "yes" else -order.quantity
        projected = self.exposure.get(order.market_id) + signed_qty
        return abs(projected) > self.config.max_position_per_market

    def _would_breach_gross_exposure(self, order: ProposedOrder) -> bool:
        max_gross = getattr(self.config, "max_gross_exposure", 0)
        if max_gross <= 0:
            return False
        return (self.exposure.gross + order.quantity) > max_gross

    def _is_stale(self, state: MarketState) -> bool:
        age = datetime.now(tz=timezone.utc) - state.updated_at
        return age.total_seconds() > self.config.max_data_age_seconds

    def _is_per_minute_rate_limited(self) -> bool:
        max_per_min = getattr(self.config, "max_orders_per_minute", 0)
        if max_per_min <= 0:
            return False
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(minutes=1)
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        return len(self._order_timestamps) >= max_per_min

    def _record_order_timestamp(self) -> None:
        self._order_timestamps.append(datetime.now(tz=timezone.utc))
