from datetime import datetime, timedelta, timezone

from src.strategy.ml_model import FEATURE_NAMES, LinearProbModel, build_feature_map
from src.types import MarketState


def test_build_feature_map_returns_expected_keys() -> None:
    base_time = datetime.now(tz=timezone.utc)
    window = []
    for i in range(8):
        window.append(
            MarketState(
                market_id="KXBTC15M-TEST",
                bid_cents=49 + (i % 2),
                ask_cents=51 + (i % 2),
                bid_size=100 + i,
                ask_size=90 + i,
                last_trade_cents=50 + (i % 3),
                updated_at=base_time + timedelta(seconds=i),
            )
        )
    feature_map = build_feature_map(window)
    assert feature_map is not None
    assert all(name in feature_map for name in FEATURE_NAMES)


def test_linear_prob_model_predicts_probability_range() -> None:
    model = LinearProbModel(
        feature_names=list(FEATURE_NAMES),
        weights=[0.1 for _ in FEATURE_NAMES],
        bias=0.0,
        means=[0.0 for _ in FEATURE_NAMES],
        stds=[1.0 for _ in FEATURE_NAMES],
    )
    p = model.predict_proba({name: 0.5 for name in FEATURE_NAMES})
    assert 0.0 <= p <= 1.0

