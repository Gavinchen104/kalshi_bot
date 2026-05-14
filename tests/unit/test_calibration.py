from __future__ import annotations

import math

from src.measurement.calibration import compute


def test_empty():
    rep = compute([])
    assert rep.n_samples == 0 and rep.brier is None


def test_perfect_predictor_gets_brier_zero():
    pairs = [(1.0, 1)] * 50 + [(0.0, 0)] * 50
    rep = compute(pairs)
    assert rep.brier == 0.0


def test_random_predictor_brier_quarter():
    pairs = [(0.5, i % 2) for i in range(100)]
    rep = compute(pairs)
    assert math.isclose(rep.brier, 0.25, abs_tol=1e-9)


def test_bins_count():
    pairs = [(0.05, 0), (0.15, 0), (0.25, 1), (0.85, 1), (0.95, 1)]
    rep = compute(pairs, n_bins=10)
    assert len(rep.bins) == 10
    assert sum(b["n"] for b in rep.bins) == len(pairs)
