from __future__ import annotations

from datetime import datetime, timezone

from src.config import RiskConfig
from src.risk.engine import RiskEngine
from src.risk.kill_switch import KillSwitch
from src.types import MarketState, ProposedOrder


def _state() -> MarketState:
    return MarketState(
        market_id="KXBTCD-26MAY1417-T80000",
        bid_cents=40,
        ask_cents=42,
        bid_size=10,
        ask_size=10,
        last_trade_cents=41,
        updated_at=datetime.now(tz=timezone.utc),
    )


def _order(side="yes", qty=1) -> ProposedOrder:
    return ProposedOrder(
        market_id="KXBTCD-26MAY1417-T80000",
        side=side,
        price_cents=42,
        quantity=qty,
    )


def _risk(**kwargs) -> RiskEngine:
    cfg = RiskConfig(max_orders_per_minute=0, **kwargs)
    return RiskEngine(cfg, KillSwitch())


def test_tail_short_cap_blocks_low_probability_no_exposure():
    risk = _risk(max_tail_short_exposure=2)
    risk.update_market_probability("KXBTCD-26MAY1417-T80000", 0.05)
    risk.apply_fill("KXBTCD-26MAY1417-T80000", -2, prob=0.05)

    verdict = risk.validate(_order(side="no", qty=1), _state())

    assert not verdict.allowed
    assert verdict.reason == "max_tail_short_exposure"


def test_tail_short_cap_blocks_high_probability_yes_exposure():
    risk = _risk(max_tail_short_exposure=2)
    risk.update_market_probability("KXBTCD-26MAY1417-T80000", 0.95)
    risk.apply_fill("KXBTCD-26MAY1417-T80000", 2, prob=0.95)

    verdict = risk.validate(_order(side="yes", qty=1), _state())

    assert not verdict.allowed
    assert verdict.reason == "max_tail_short_exposure"


def test_vol_regime_jump_engages_kill_switch():
    kill = KillSwitch()
    cfg = RiskConfig(max_orders_per_minute=0, vol_regime_min_samples=3, vol_regime_zscore=2.0)
    risk = RiskEngine(cfg, kill)
    for vol in (0.40, 0.41, 0.39):
        risk.update_realized_vol(vol)

    risk.update_realized_vol(0.80)
    verdict = risk.validate(_order(), _state())

    assert kill.engaged
    assert kill.reason == "vol_regime_jump"
    assert not verdict.allowed
    assert verdict.reason == "kill_switch:vol_regime_jump"
