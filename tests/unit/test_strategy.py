from datetime import datetime, timezone

from src.config import StrategyConfig
from src.strategy.signals import Strategy
from src.types import MarketState


def test_strategy_no_signal_when_spread_too_wide() -> None:
    strategy = Strategy(
        StrategyConfig(
            min_edge_bps=50,
            min_top_book_depth=10,
            max_spread_cents=2,
            target_market_regex=".*BTC.*",
            use_ml_model=False,
        )
    )
    state = MarketState(
        market_id="TEST-1",
        bid_cents=40,
        ask_cents=50,
        bid_size=100,
        ask_size=100,
        last_trade_cents=45,
        updated_at=datetime.now(tz=timezone.utc),
    )
    assert strategy.compute_signal(state) is None


def test_strategy_respects_target_market_regex() -> None:
    strategy = Strategy(
        StrategyConfig(
            min_edge_bps=50,
            min_top_book_depth=10,
            max_spread_cents=2,
            target_market_regex=".*BTC.*15.*",
            use_ml_model=False,
        )
    )
    state = MarketState(
        market_id="KXETHD-TEST",
        bid_cents=40,
        ask_cents=41,
        bid_size=100,
        ask_size=100,
        last_trade_cents=40,
        updated_at=datetime.now(tz=timezone.utc),
    )
    assert strategy.compute_signal(state) is None

