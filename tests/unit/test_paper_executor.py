from __future__ import annotations

from datetime import datetime, timezone

from src.config import ExecutionConfig
from src.execution.paper_executor import PaperExecutor
from src.types import MarketState, ProposedOrder


def _state(bid_size=3, ask_size=2) -> MarketState:
    return MarketState(
        market_id="KXBTCD-26MAY1417-T80000",
        bid_cents=40,
        ask_cents=42,
        bid_size=bid_size,
        ask_size=ask_size,
        last_trade_cents=41,
        updated_at=datetime.now(tz=timezone.utc),
    )


def _executor() -> PaperExecutor:
    return PaperExecutor(ExecutionConfig(min_order_interval_ms=0, slippage_bps=0))


def test_paper_executor_partially_fills_against_visible_ask_depth():
    order = ProposedOrder(
        market_id="KXBTCD-26MAY1417-T80000",
        side="yes",
        price_cents=42,
        quantity=5,
    )

    result = _executor().execute(order, _state(ask_size=2))

    assert result["status"] == "paper_partially_filled"
    assert result["filled_quantity"] == 2
    assert result["fill_price_cents"] == 43


def test_paper_executor_unfilled_when_no_usable_depth():
    order = ProposedOrder(
        market_id="KXBTCD-26MAY1417-T80000",
        side="no",
        price_cents=60,
        quantity=5,
    )

    result = _executor().execute(order, _state(bid_size=0))

    assert result["status"] == "paper_unfilled"


def test_paper_executor_full_fill_with_sufficient_depth():
    order = ProposedOrder(
        market_id="KXBTCD-26MAY1417-T80000",
        side="yes",
        price_cents=42,
        quantity=2,
    )

    result = _executor().execute(order, _state(ask_size=5))

    assert result["status"] == "paper_filled"
    assert result["filled_quantity"] == 2
