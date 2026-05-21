from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

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
        self._market_probs: dict[str, float] = {}
        self._tail_short_by_market: dict[str, int] = {}
        self._vol_history: list[float] = []

    def update_session_pnl(self, pnl_cents: int) -> None:
        self.session_pnl_cents = pnl_cents
        if pnl_cents > self.peak_pnl_cents:
            self.peak_pnl_cents = pnl_cents

    def update_market_probability(self, market_id: str, prob: float) -> None:
        self._market_probs[market_id] = min(1.0, max(0.0, float(prob)))

    def update_realized_vol(self, vol_annualized: float) -> None:
        """Engage the kill switch when live vol jumps far above recent regime."""
        vol = float(vol_annualized)
        if vol <= 0:
            return
        min_n = self.config.vol_regime_min_samples
        if len(self._vol_history) >= min_n and self.config.vol_regime_zscore > 0:
            avg = mean(self._vol_history)
            sigma = pstdev(self._vol_history)
            if sigma > 0 and vol > avg + self.config.vol_regime_zscore * sigma:
                self.kill_switch.engage("vol_regime_jump")
        self._vol_history.append(vol)
        keep = max(min_n * 4, min_n, 1)
        self._vol_history = self._vol_history[-keep:]

    def apply_fill(self, market_id: str, signed_qty: int, prob: float | None = None) -> None:
        self._positions[market_id] = self._positions.get(market_id, 0) + signed_qty
        self._gross = sum(abs(q) for q in self._positions.values())
        p = self._market_probs.get(market_id) if prob is None else min(1.0, max(0.0, float(prob)))
        if p is not None and self._is_tail_short(signed_qty, p):
            self._tail_short_by_market[market_id] = (
                self._tail_short_by_market.get(market_id, 0) + abs(signed_qty)
            )

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
        if self._breach_tail_short(order):
            return RiskResult(False, "max_tail_short_exposure")
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

    def _breach_tail_short(self, order: ProposedOrder) -> bool:
        cap = self.config.max_tail_short_exposure
        if cap <= 0:
            return False
        prob = self._market_probs.get(order.market_id)
        if prob is None:
            return False
        signed = order.quantity if order.side == "yes" else -order.quantity
        if not self._is_tail_short(signed, prob):
            return False
        return (sum(self._tail_short_by_market.values()) + order.quantity) > cap

    def _is_tail_short(self, signed_qty: int, prob: float) -> bool:
        if signed_qty < 0 and prob <= self.config.tail_low_prob:
            return True
        if signed_qty > 0 and prob >= self.config.tail_high_prob:
            return True
        return False

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
