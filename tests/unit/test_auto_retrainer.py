import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from src.config import AutoTrainConfig, StrategyConfig
from src.storage.repository import Repository
from src.training.auto_retrainer import AutoRetrainer


class _DummyStrategy:
    def __init__(self):
        self.reload_count = 0
        self._edge = False

    def reload_model(self) -> bool:
        self.reload_count += 1
        return True

    def set_model_quality(self, beats: bool) -> None:
        self._edge = beats


def _make_cfg():
    return AutoTrainConfig(
        enabled=True,
        interval_minutes=1,
        min_market_states=1,
        market_like="BTC",
        horizon=1,
        limit=100,
        epochs=1,
        lr=0.1,
    )


def _make_strat_cfg():
    return StrategyConfig(
        min_edge_bps=50,
        min_top_book_depth=10,
        max_spread_cents=8,
        target_market_regex=".*BTC.*15.*",
        use_ml_model=True,
        model_path="data/models/unused.json",
        min_model_confidence=0.01,
        feature_lookback=5,
    )


def _seed_market_states(repo, n=10):
    for _ in range(n):
        repo.log_event(
            "market_state",
            market_id="KXBTC15M-TEST",
            payload={
                "bid_cents": 50, "ask_cents": 51, "last_trade_cents": 50,
                "bid_size": 10, "ask_size": 10,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )


def test_auto_retrainer_runs_and_records(tmp_path) -> None:
    db_path = str(tmp_path / "bot.db")
    repo = Repository(db_path)
    _seed_market_states(repo)

    dummy = _DummyStrategy()
    fake_metrics = {
        "rows": 500.0, "val_accuracy": 0.65, "val_logloss": 0.55,
        "wf_mean_accuracy": 0.60, "wf_mean_baseline": 0.55,
        "beats_baseline": 1.0, "training_mode": "kalshi_tick",
        "recipe": "test_recipe",
    }

    with patch(
        "src.training.auto_retrainer.train_from_repository",
        return_value=fake_metrics,
    ):
        retrainer = AutoRetrainer(
            config=_make_cfg(),
            strategy_config=_make_strat_cfg(),
            repository=repo,
            strategy=dummy,
        )
        asyncio.run(retrainer.maybe_retrain())

    assert dummy._edge is True
    events = repo.recent_model_events(limit=5)
    assert any(e["event_type"] == "model_retrained" for e in events)


def test_auto_retrainer_no_trade_when_no_edge(tmp_path) -> None:
    db_path = str(tmp_path / "bot.db")
    repo = Repository(db_path)
    _seed_market_states(repo)

    dummy = _DummyStrategy()
    fake_metrics = {
        "rows": 500.0, "val_accuracy": 0.55, "val_logloss": 0.65,
        "wf_mean_accuracy": 0.50, "wf_mean_baseline": 0.55,
        "beats_baseline": 0.0, "training_mode": "kalshi_tick",
        "recipe": "test_recipe",
    }

    with patch(
        "src.training.auto_retrainer.train_from_repository",
        return_value=fake_metrics,
    ):
        retrainer = AutoRetrainer(
            config=_make_cfg(),
            strategy_config=_make_strat_cfg(),
            repository=repo,
            strategy=dummy,
        )
        asyncio.run(retrainer.maybe_retrain())

    assert dummy._edge is False
    assert retrainer.model_beats_baseline is False
