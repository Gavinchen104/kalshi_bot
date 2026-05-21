"""Unit tests for the isotonic calibration layer (Phase 3 / B1)."""
from __future__ import annotations

import numpy as np

from src.strategy.calibrator import (
    IsotonicCalibrator,
    time_series_expanding_folds,
    time_series_split,
)


def _brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def test_isotonic_output_is_monotone_non_decreasing():
    rng = np.random.default_rng(0)
    # Synthetic: raw probs are noisy, but truth is monotone in them.
    x = rng.uniform(0.0, 1.0, 500)
    y = (rng.uniform(0.0, 1.0, 500) < x).astype(float)
    cal = IsotonicCalibrator().fit(x, y)
    grid = np.linspace(0.0, 1.0, 100)
    yhat = cal.predict(grid)
    assert np.all(np.diff(yhat) >= -1e-12), "isotonic must be non-decreasing"
    assert yhat.min() >= 0.0 and yhat.max() <= 1.0


def test_isotonic_recovers_calibration_on_a_biased_pricer():
    """Synthetic Phase-1-style miscalibration: raw pricer compresses truth
    toward extremes (calls 0.001 when reality is 0.10). Isotonic should
    *reduce* Brier on a held-out fold."""
    rng = np.random.default_rng(7)
    n = 4000
    # True latent probability roughly uniform.
    p_true = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p_true).astype(float)
    # Biased raw: cube the truth → pushes mass to extremes (overconfident).
    p_raw = p_true ** 3

    # Time-series 1-fold split (no shuffle): first 80% train, last 20% OOS.
    tr, te = time_series_split(n, train_frac=0.8)
    cal = IsotonicCalibrator().fit(p_raw[tr], y[tr])
    p_cal_oos = cal.predict(p_raw[te])

    brier_raw = _brier(p_raw[te], y[te])
    brier_cal = _brier(p_cal_oos, y[te])
    # On a deliberately miscalibrated pricer, isotonic should improve OOS.
    assert brier_cal < brier_raw - 0.01


def test_isotonic_handles_constant_input():
    cal = IsotonicCalibrator().fit(np.full(100, 0.5), np.r_[np.ones(40), np.zeros(60)])
    # Single pooled block → empirical mean 0.4 → predict returns 0.4 for any query.
    out = cal.predict(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out, 0.4)


def test_time_series_split_does_not_overlap_and_preserves_order():
    tr, te = time_series_split(100, train_frac=0.8)
    assert tr.start == 0 and tr.stop == 80
    assert te.start == 80 and te.stop == 100
    # Strict: train ends exactly where test begins.
    assert tr.stop == te.start


def test_time_series_expanding_folds_are_strictly_ordered():
    folds = list(time_series_expanding_folds(100, n_folds=4))
    assert len(folds) == 4
    prev_test_end = 0
    for tr, te in folds:
        assert tr.start == 0                # expanding from the start
        assert tr.stop == te.start          # no overlap
        assert te.stop > te.start           # non-empty test
        assert te.stop >= prev_test_end     # walks forward in time
        prev_test_end = te.stop
    assert folds[-1][1].stop == 100         # last fold reaches the end


def test_predict_before_fit_errors():
    import pytest
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict(np.array([0.5]))


def test_calibrator_round_trips_json_artifact(tmp_path):
    cal = IsotonicCalibrator().fit(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    path = tmp_path / "b1_isotonic.json"

    cal.save(path, metadata={"phase": "3", "workstream": "B1"})
    loaded = IsotonicCalibrator.load(path)

    grid = np.array([0.05, 0.15, 0.85, 0.95])
    assert np.allclose(loaded.predict(grid), cal.predict(grid))
    assert loaded.predict_one(0.85) == cal.predict_one(0.85)
