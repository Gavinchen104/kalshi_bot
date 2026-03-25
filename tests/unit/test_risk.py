from datetime import datetime, timedelta, timezone

from src.config import RiskConfig
from src.risk.exposure import ExposureBook
from src.risk.kill_switch import KillSwitch
from src.risk.limits import RiskEngine
from src.types import MarketState, ProposedOrder


def _risk_engine(max_gross: int = 100, max_per_min: int = 60) -> RiskEngine:
    cfg = RiskConfig(
        max_position_per_market=5,
        max_order_size=2,
        max_daily_loss_cents=1000,
        max_drawdown_cents=500,
        max_data_age_seconds=5,
        max_gross_exposure=max_gross,
        max_orders_per_minute=max_per_min,
    )
    return RiskEngine(cfg, exposure=ExposureBook(), kill_switch=KillSwitch())


def test_risk_blocks_large_order() -> None:
    risk = _risk_engine()
    order = ProposedOrder(market_id="MKT", side="yes", price_cents=50, quantity=3)
    state = MarketState(
        market_id="MKT",
        bid_cents=49,
        ask_cents=51,
        bid_size=10,
        ask_size=10,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc),
    )
    result = risk.validate(order, state)
    assert not result.allowed
    assert result.reason == "max_order_size"


def test_risk_blocks_stale_data() -> None:
    risk = _risk_engine()
    order = ProposedOrder(market_id="MKT", side="yes", price_cents=50, quantity=1)
    state = MarketState(
        market_id="MKT",
        bid_cents=49,
        ask_cents=51,
        bid_size=10,
        ask_size=10,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc) - timedelta(seconds=30),
    )
    result = risk.validate(order, state)
    assert not result.allowed
    assert result.reason == "stale_market_data"


def test_risk_blocks_invalid_price() -> None:
    risk = _risk_engine()
    order = ProposedOrder(market_id="MKT", side="yes", price_cents=0, quantity=1)
    state = MarketState(
        market_id="MKT",
        bid_cents=49,
        ask_cents=51,
        bid_size=10,
        ask_size=10,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc),
    )
    result = risk.validate(order, state)
    assert not result.allowed
    assert result.reason == "invalid_price"


def test_risk_blocks_gross_exposure_breach() -> None:
    risk = _risk_engine(max_gross=3)
    state = MarketState(
        market_id="MKT",
        bid_cents=49,
        ask_cents=51,
        bid_size=10,
        ask_size=10,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc),
    )
    # First fill exposure so gross = 2
    risk.exposure.apply_fill("MKT", 2)
    # Now trying to add 2 more would make gross = 4 > 3
    order = ProposedOrder(market_id="MKT2", side="yes", price_cents=50, quantity=2)
    result = risk.validate(order, state)
    assert not result.allowed
    assert result.reason == "max_gross_exposure"


def test_risk_blocks_per_minute_rate_limit() -> None:
    risk = _risk_engine(max_per_min=2)
    state = MarketState(
        market_id="MKT",
        bid_cents=49,
        ask_cents=51,
        bid_size=10,
        ask_size=10,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc),
    )
    order = ProposedOrder(market_id="MKT", side="yes", price_cents=50, quantity=1)
    first = risk.validate(order, state)
    second = risk.validate(order, state)
    third = risk.validate(order, state)
    assert first.allowed
    assert second.allowed
    assert not third.allowed
    assert third.reason == "max_orders_per_minute"

