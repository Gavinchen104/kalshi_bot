"""Unit tests for vol estimators (T09)."""
from __future__ import annotations

import numpy as np

from src.pricing.volatility import close_to_close_vol, horizon_matched_vol


def _calm_closes(n: int, base: float = 80_000.0) -> np.ndarray:
    """Smooth ramp — low realized vol."""
    return np.linspace(base, base + n, n + 1)


def _noisy_closes(n: int, base: float = 80_000.0, scale: float = 50.0) -> np.ndarray:
    """Higher-vol synthetic series with reproducible noise."""
    rng = np.random.default_rng(42)
    return base + np.cumsum(rng.normal(0.0, scale, n + 1))


def test_horizon_matched_uses_window_proportional_to_horizon():
    # 1500 candles available; floor=60, cap=1440.
    closes = _noisy_closes(1500)
    short = horizon_matched_vol(closes, horizon_seconds=15 * 60, floor_min=60, cap_min=1440)
    long = horizon_matched_vol(closes, horizon_seconds=24 * 3600, floor_min=60, cap_min=1440)
    assert short is not None and long is not None
    # 15min horizon → window=60; 24h horizon → window=1440. Different windows
    # over the same series produce different vol estimates.
    assert short != long
    # Cross-check with explicit close_to_close_vol calls.
    assert short == close_to_close_vol(closes, window=60)
    assert long == close_to_close_vol(closes, window=1440)


def test_horizon_matched_clamps_to_floor_and_cap():
    closes = _noisy_closes(2000)
    # Horizon below floor → window = floor.
    sub_floor = horizon_matched_vol(closes, horizon_seconds=5 * 60, floor_min=60, cap_min=1440)
    assert sub_floor == close_to_close_vol(closes, window=60)
    # Horizon above cap → window = cap.
    over_cap = horizon_matched_vol(closes, horizon_seconds=72 * 3600, floor_min=60, cap_min=1440)
    assert over_cap == close_to_close_vol(closes, window=1440)


def test_horizon_matched_returns_none_when_insufficient_history():
    # Only 50 candles but the horizon implies a window of 1440 → None.
    closes = _noisy_closes(50)
    assert horizon_matched_vol(closes, horizon_seconds=24 * 3600,
                                floor_min=60, cap_min=1440) is None


def test_horizon_matched_zero_or_negative_horizon_returns_none():
    closes = _noisy_closes(200)
    assert horizon_matched_vol(closes, horizon_seconds=0) is None
    assert horizon_matched_vol(closes, horizon_seconds=-10) is None
