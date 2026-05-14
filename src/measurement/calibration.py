"""
Calibration metrics for binary probability forecasts.

Given a list of (predicted_prob, outcome ∈ {0, 1}) pairs:
  - Brier score: mean squared error of probability vs outcome (lower = better).
  - Log loss: cross-entropy (lower = better).
  - Reliability bins: for each predicted-prob bucket, the empirical frequency
    of YES outcomes. A well-calibrated forecaster has empirical ≈ predicted in every bin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalibrationReport:
    n_samples: int
    brier: float | None
    log_loss: float | None
    bins: list[dict]


def compute(pairs: list[tuple[float, int]], n_bins: int = 10) -> CalibrationReport:
    if not pairs:
        return CalibrationReport(0, None, None, [])
    n = len(pairs)
    brier = sum((p - o) ** 2 for p, o in pairs) / n
    log_loss_sum = 0.0
    eps = 1e-9
    for p, o in pairs:
        p_c = min(1 - eps, max(eps, p))
        log_loss_sum += -(o * math.log(p_c) + (1 - o) * math.log(1 - p_c))
    log_loss = log_loss_sum / n

    bins: list[dict] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bucket = [
            (p, o) for p, o in pairs
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not bucket:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mean_pred": None, "emp_freq": None})
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        emp_freq = sum(o for _, o in bucket) / len(bucket)
        bins.append({
            "lo": lo, "hi": hi, "n": len(bucket),
            "mean_pred": mean_pred, "emp_freq": emp_freq,
        })
    return CalibrationReport(n, brier, log_loss, bins)
