from datetime import datetime, timezone

from src.portfolio.pnl import PnLTracker
from src.strategy.exit_manager import ExitManager
from src.types import MarketState


def _state(market_id: str = "MKT-1", bid: int = 50, ask: int = 52, last: int = 51) -> MarketState:
    return MarketState(
        market_id=market_id,
        bid_cents=bid,
        ask_cents=ask,
        bid_size=20,
        ask_size=20,
        last_trade_cents=last,
        updated_at=datetime.now(tz=timezone.utc),
    )


def test_take_profit_closes_long_yes() -> None:
    # 25% take profit on entry of 40c → threshold = 10c gain
    em = ExitManager(take_profit_pct=0.25, stop_loss_pct=0.50)
    pnl = PnLTracker()
    pnl.on_fill("MKT-1", "yes", quantity=3, fill_price_cents=40)

    # Mark at 51 → unrealized = +11/contract → exceeds 25% of 40 = 10
    state = _state("MKT-1", bid=50, ask=52, last=51)
    orders = em.check_exits(state, pnl)
    assert len(orders) == 1
    assert orders[0].side == "no"
    assert orders[0].quantity == 3


def test_stop_loss_closes_long_yes() -> None:
    # 15% stop loss on entry of 60c → threshold = 9c loss
    em = ExitManager(take_profit_pct=0.50, stop_loss_pct=0.15)
    pnl = PnLTracker()
    pnl.on_fill("MKT-1", "yes", quantity=2, fill_price_cents=60)

    # Mark at 50 → unrealized = -10/contract → exceeds 15% of 60 = 9
    state = _state("MKT-1", bid=49, ask=51, last=50)
    orders = em.check_exits(state, pnl)
    assert len(orders) == 1
    assert orders[0].side == "no"
    assert orders[0].quantity == 2


def test_take_profit_closes_short_yes() -> None:
    # 20% take profit on effective entry ~70c → threshold = 14c gain
    em = ExitManager(take_profit_pct=0.20, stop_loss_pct=0.50)
    pnl = PnLTracker()
    pnl.on_fill("MKT-1", "no", quantity=2, fill_price_cents=30)

    # Mark at 55 → unrealized per contract = 70 - 55 = 15 → exceeds 14
    state = _state("MKT-1", bid=54, ask=56, last=55)
    orders = em.check_exits(state, pnl)
    assert len(orders) == 1
    assert orders[0].side == "yes"
    assert orders[0].quantity == 2


def test_no_exit_within_thresholds() -> None:
    # 25% TP / 25% SL on entry of 50c → thresholds at ±12.5c
    em = ExitManager(take_profit_pct=0.25, stop_loss_pct=0.25)
    pnl = PnLTracker()
    pnl.on_fill("MKT-1", "yes", quantity=1, fill_price_cents=50)

    # Mark at 51 → unrealized = +1/contract → well within both thresholds
    state = _state("MKT-1", bid=50, ask=52, last=51)
    orders = em.check_exits(state, pnl)
    assert len(orders) == 0


def test_no_exit_for_flat_position() -> None:
    em = ExitManager(take_profit_pct=0.01, stop_loss_pct=0.01)
    pnl = PnLTracker()

    state = _state("MKT-1", bid=50, ask=52, last=51)
    orders = em.check_exits(state, pnl)
    assert len(orders) == 0
