"""Unit tests for vol estimators (T09)."""
from __future__ import annotations

import numpy as np

from src.pricing.volatility import (
    blend_vol,
    close_to_close_vol,
    ewma_vol,
    horizon_matched_vol,
    long_window_floor_vol,
)


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


# ── blend ───────────────────────────────────────────────────────────────────

def test_blend_is_mean_of_available_window_vols():
    closes = _noisy_closes(2000)
    expected = np.mean([
        close_to_close_vol(closes, window=w) for w in (60, 360, 1440)
    ])
    got = blend_vol(closes, horizon_seconds=3600,
                     windows_min=(60, 360, 1440))
    assert got is not None
    assert abs(got - expected) < 1e-12


def test_blend_falls_back_to_available_windows():
    # 500 candles → 1440 window is unavailable; blend uses just {60, 360}.
    closes = _noisy_closes(500)
    got = blend_vol(closes, windows_min=(60, 360, 1440))
    expected = np.mean([
        close_to_close_vol(closes, window=60),
        close_to_close_vol(closes, window=360),
    ])
    assert got is not None and abs(got - expected) < 1e-12


def test_blend_returns_none_when_no_window_available():
    closes = _noisy_closes(20)
    assert blend_vol(closes, windows_min=(60, 360, 1440)) is None


# ── long-window floor ──────────────────────────────────────────────────────

def test_long_window_floor_equals_direct_realized_vol():
    closes = _noisy_closes(8 * 24 * 60)  # 8 days of 1-min candles
    direct = close_to_close_vol(closes, window=7 * 24 * 60)
    got = long_window_floor_vol(closes, lookback_days=7)
    assert got is not None and direct is not None
    assert abs(got - direct) < 1e-12


def test_long_window_floor_disabled_at_zero_days():
    closes = _noisy_closes(1000)
    assert long_window_floor_vol(closes, lookback_days=0) is None
    assert long_window_floor_vol(closes, lookback_days=-1) is None


def test_long_window_floor_returns_none_when_insufficient_history():
    closes = _noisy_closes(100)
    assert long_window_floor_vol(closes, lookback_days=7) is None


# ── EWMA ───────────────────────────────────────────────────────────────────

def test_ewma_reacts_faster_to_recent_burst_than_simple_window():
    # First 1000 minutes calm; last 50 minutes high-vol burst.
    rng = np.random.default_rng(3)
    calm = rng.normal(0.0, 1.0, 1000)
    burst = rng.normal(0.0, 100.0, 50)
    closes = 80_000 + np.cumsum(np.concatenate([calm, burst]))
    simple = close_to_close_vol(closes, window=1000)
    # Short half-life — should weight the recent burst heavily.
    ew = ewma_vol(closes, horizon_seconds=3600, half_life_min=30)
    assert simple is not None and ew is not None
    assert ew > simple  # EWMA picks up the regime change the long window dilutes


def test_ewma_default_half_life_scales_with_horizon():
    # Same series, different horizons → different half-life defaults → different vol.
    rng = np.random.default_rng(5)
    closes = 80_000 + np.cumsum(rng.normal(0.0, 20.0, 1500))
    short_h = ewma_vol(closes, horizon_seconds=900)        # 15 min
    long_h = ewma_vol(closes, horizon_seconds=24 * 3600)   # 24 h
    assert short_h is not None and long_h is not None
    assert short_h != long_h


def test_ewma_zero_or_one_close_returns_none():
    assert ewma_vol(np.array([]), horizon_seconds=600) is None
    assert ewma_vol(np.array([80_000.0]), horizon_seconds=600) is None
