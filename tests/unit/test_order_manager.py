import random
from datetime import datetime, timezone

import pytest

from src.execution.order_manager import OrderManager, PaperFillSimulator
from src.types import MarketState, ProposedOrder


class _DummyClient:
    async def place_order(self, payload):
        return {"status": "ok", "payload": payload}


def _state(bid: int = 48, ask: int = 52) -> MarketState:
    return MarketState(
        market_id="MKT-1",
        bid_cents=bid,
        ask_cents=ask,
        bid_size=20,
        ask_size=20,
        last_trade_cents=50,
        updated_at=datetime.now(tz=timezone.utc),
    )


@pytest.mark.asyncio
async def test_order_manager_rate_limits() -> None:
    manager = OrderManager(_DummyClient(), paper_mode=True, min_order_interval_ms=10)
    order = ProposedOrder(market_id="MKT-1", side="yes", price_cents=50, quantity=1)

    first = await manager.execute(order)
    second = await manager.execute(order)

    assert first["status"] in ("paper_filled", "paper_unfilled")
    assert second["status"] == "skipped_rate_limited"


@pytest.mark.asyncio
async def test_order_manager_paper_fill_returns_price() -> None:
    always_fill = PaperFillSimulator(fee_bps=50, slippage_bps=0, rng=random.Random(0))
    # Seed 0 should fill at least some orders; we use a sim that always fills
    class _AlwaysFillSim:
        fee_bps = 50
        slippage_bps = 0
        def simulate(self, order, state=None):
            return {"status": "paper_filled", "fill_price_cents": order.price_cents, "fee_cents": 1, "slippage_cents": 0}

    manager = OrderManager(_DummyClient(), paper_mode=True, fill_simulator=_AlwaysFillSim())
    order = ProposedOrder(market_id="MKT-1", side="yes", price_cents=52, quantity=1)
    result = await manager.execute(order, state=_state())

    assert result["status"] == "paper_filled"
    assert result["fill_price_cents"] == 52
    assert result["fee_cents"] == 1


def test_fill_simulator_aggressive_order_high_fill_prob() -> None:
    sim = PaperFillSimulator(fee_bps=0, slippage_bps=0, rng=random.Random(42))
    state = _state(bid=48, ask=52)
    order = ProposedOrder(market_id="MKT-1", side="yes", price_cents=55, quantity=1)

    results = [sim.simulate(order, state)["status"] for _ in range(100)]
    fill_rate = sum(1 for r in results if r == "paper_filled") / 100
    assert fill_rate >= 0.90, f"Expected ≥90% fill rate for aggressive order, got {fill_rate:.0%}"


def test_fill_simulator_passive_order_low_fill_prob() -> None:
    sim = PaperFillSimulator(fee_bps=0, slippage_bps=0, rng=random.Random(7))
    state = _state(bid=48, ask=52)
    order = ProposedOrder(market_id="MKT-1", side="yes", price_cents=44, quantity=1)

    results = [sim.simulate(order, state)["status"] for _ in range(100)]
    fill_rate = sum(1 for r in results if r == "paper_filled") / 100
    assert fill_rate <= 0.20, f"Expected ≤20% fill rate for passive order, got {fill_rate:.0%}"


def test_fill_simulator_applies_fees() -> None:
    sim = PaperFillSimulator(fee_bps=100, slippage_bps=0, rng=random.Random(0))
    state = _state(bid=48, ask=52)
    order = ProposedOrder(market_id="MKT-1", side="yes", price_cents=55, quantity=10)

    for _ in range(20):
        result = sim.simulate(order, state)
        if result["status"] == "paper_filled":
            assert result["fee_cents"] > 0
            break
