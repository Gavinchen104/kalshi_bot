import numpy as np

from src.strategy.features import (
    ENHANCED_FEATURE_NAMES,
    build_enhanced_features,
    compute_bollinger_pct,
    compute_macd_histogram,
    compute_momentum,
    compute_realized_volatility,
    compute_rsi,
    compute_volume_ratio,
)
from src.types import MarketState
from datetime import datetime, timezone


def _state(bid=50, ask=52, last=51, bid_size=20, ask_size=20):
    return MarketState(
        market_id="MKT-1",
        bid_cents=bid,
        ask_cents=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        last_trade_cents=last,
        updated_at=datetime.now(tz=timezone.utc),
    )


def test_rsi_within_bounds():
    closes = np.array([100 + i * 0.5 for i in range(30)], dtype=float)
    rsi = compute_rsi(closes, 14)
    assert 0 <= rsi <= 100


def test_rsi_neutral_on_flat():
    closes = np.array([100.0] * 30, dtype=float)
    rsi = compute_rsi(closes, 14)
    assert rsi == 50.0


def test_macd_returns_float():
    closes = np.array([100 + np.sin(i * 0.3) for i in range(50)], dtype=float)
    val = compute_macd_histogram(closes)
    assert isinstance(val, float)


def test_bollinger_mid_on_flat():
    closes = np.array([50.0] * 25, dtype=float)
    pct = compute_bollinger_pct(closes)
    assert 0.49 <= pct <= 0.51


def test_realized_vol_non_negative():
    closes = np.array([100 + i for i in range(20)], dtype=float)
    vol = compute_realized_volatility(closes, 10)
    assert vol >= 0.0


def test_momentum_positive_on_uptrend():
    closes = np.array([100 + i for i in range(20)], dtype=float)
    m = compute_momentum(closes, 5)
    assert m > 0.0


def test_volume_ratio_one_on_uniform():
    volumes = np.array([100.0] * 30, dtype=float)
    r = compute_volume_ratio(volumes)
    assert abs(r - 1.0) < 0.01


def test_build_enhanced_features_with_spot():
    window = [_state(bid=48 + i, ask=50 + i, last=49 + i) for i in range(25)]
    closes = np.array([60000 + i * 10 for i in range(50)], dtype=float)
    volumes = np.array([100 + i for i in range(50)], dtype=float)
    ohlcv = np.column_stack([closes - 5, closes + 5, closes - 10, closes, volumes])

    features = build_enhanced_features(window, closes, volumes, ohlcv)
    assert features is not None
    for name in ENHANCED_FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"


def test_build_enhanced_features_without_spot():
    window = [_state(bid=48 + i, ask=50 + i, last=49 + i) for i in range(25)]
    features = build_enhanced_features(window)
    assert features is not None
    assert features["rsi_14"] == 50.0
    assert features["bb_pct"] == 0.5
