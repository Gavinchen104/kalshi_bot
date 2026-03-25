from src.backtest.engine import BacktestResult, BacktestTrade
from datetime import datetime, timezone


def test_backtest_result_empty():
    r = BacktestResult()
    assert r.win_rate == 0.0
    assert r.total_pnl_cents == 0


def test_backtest_result_summary():
    t = BacktestTrade(
        market_id="MKT-1",
        side="yes",
        entry_price_cents=50,
        exit_price_cents=60,
        quantity=1,
        fee_cents=2,
        pnl_cents=8,
        entry_time=datetime.now(tz=timezone.utc),
    )
    r = BacktestResult(
        total_pnl_cents=8,
        n_trades=1,
        n_wins=1,
        n_losses=0,
        trades=[t],
        equity_curve=[8],
    )
    summary = r.summary()
    assert summary["win_rate"] == 1.0
    assert summary["n_trades"] == 1
    assert summary["total_pnl_cents"] == 8


def test_backtest_result_win_rate_calculation():
    t1 = BacktestTrade(
        market_id="M", side="yes", entry_price_cents=50,
        exit_price_cents=60, quantity=1, fee_cents=0, pnl_cents=10,
        entry_time=datetime.now(tz=timezone.utc),
    )
    t2 = BacktestTrade(
        market_id="M", side="yes", entry_price_cents=50,
        exit_price_cents=40, quantity=1, fee_cents=0, pnl_cents=-10,
        entry_time=datetime.now(tz=timezone.utc),
    )
    r = BacktestResult(n_trades=2, n_wins=1, n_losses=1, trades=[t1, t2])
    assert r.win_rate == 0.5
