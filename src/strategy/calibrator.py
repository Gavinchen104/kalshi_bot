"""
Empirical calibration layer (Phase 3 / Track B / B1).

Phase 2 GATE A showed the raw BS-from-realized-vol probabilities are
systematically miscalibrated on the Kalshi BTC daily contracts (log loss 1.25
across every vol mode, tail-bin empirical 9.6% when predicted ~0.1%). The
B1 hypothesis: BS is *monotonically* informative about truth even when its
calibration is wrong, so a learned monotone mapping (isotonic regression) can
recover a calibrated probability without changing the input feature.

Implementation: pool-adjacent-violators (PAV) — the textbook O(n log n)
isotonic regressor with binary targets — implemented from scratch on numpy
so we don't pull sklearn into deps for one function. Cross-validation is
**strictly time-series** (fit on past, score on strictly later folds): the
calibration mapping must generalize forward, not be measured against the
data it was fit on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IsotonicCalibrator:
    """Monotone non-decreasing prob → prob mapping fitted by PAV.

    After ``fit(raw_probs, outcomes)``, ``predict(raw_probs)`` returns the
    calibrated probabilities. The fitted curve is stored as two parallel
    arrays of breakpoints + values; ``predict`` does a binary search.
    """
    # Sorted (ascending) raw_probs marking the boundaries of merged PAV blocks.
    # `_block_x[i]` is the *upper edge* of the i'th block. `_block_y[i]` is the
    # block's calibrated probability (= empirical YES rate within the block).
    _block_x: np.ndarray | None = None
    _block_y: np.ndarray | None = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit PAV isotonic regression on (raw_prob, outcome ∈ {0,1}) pairs.

        Outcomes need not be {0,1} strictly — any real targets work, but
        binary is the calibration use case. Equal raw_probs are pooled.
        """
        x = np.asarray(raw_probs, dtype=float).ravel()
        y = np.asarray(outcomes, dtype=float).ravel()
        if x.shape[0] == 0 or x.shape[0] != y.shape[0]:
            self._block_x = np.empty(0)
            self._block_y = np.empty(0)
            return self
        # Sort by raw_prob (stable so ties keep original outcome order).
        order = np.argsort(x, kind="mergesort")
        x_s, y_s = x[order], y[order]

        # PAV: walk left→right, maintain a stack of (sum, weight, max_x).
        # When a new point violates monotonicity (mean < previous block's
        # mean), merge backward until restored. O(n) amortized.
        sums: list[float] = []
        wts: list[float] = []
        ups: list[float] = []  # max raw_prob in this block
        for xi, yi in zip(x_s, y_s):
            sums.append(float(yi))
            wts.append(1.0)
            ups.append(float(xi))
            # Merge while previous block's mean > current block's mean.
            while len(sums) >= 2 and (sums[-2] / wts[-2]) > (sums[-1] / wts[-1]):
                sums[-2] += sums[-1]; wts[-2] += wts[-1]
                ups[-2] = ups[-1]
                sums.pop(); wts.pop(); ups.pop()
        self._block_x = np.asarray(ups, dtype=float)
        self._block_y = np.asarray([s / w for s, w in zip(sums, wts)], dtype=float)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map each raw_prob through the fitted isotonic step function.

        Queries below the first block's upper edge use the first block's
        value; queries above the last block use the last block's value.
        """
        if self._block_x is None or self._block_x.size == 0:
            raise RuntimeError("IsotonicCalibrator.fit() has not been called")
        q = np.asarray(raw_probs, dtype=float).ravel()
        # searchsorted gives the index of the first block whose upper edge
        # is >= q (i.e. the block q falls into).
        idx = np.searchsorted(self._block_x, q, side="left")
        idx = np.clip(idx, 0, self._block_x.size - 1)
        return self._block_y[idx]


def time_series_split(n: int, train_frac: float = 0.8) -> tuple[slice, slice]:
    """Strict time-series 1-fold split. Caller passes time-ordered data;
    returned slices preserve order with no overlap and no look-ahead."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    cut = max(1, int(n * train_frac))
    return slice(0, cut), slice(cut, n)


def time_series_expanding_folds(n: int, n_folds: int = 4):
    """Expanding-window time-series CV.

    Yields (train_slice, test_slice) for each fold. Fold k trains on the
    earliest k/(n_folds+1) of the data and scores on the next 1/(n_folds+1).
    All windows are contiguous, in time order, no overlap.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    chunk = max(1, n // (n_folds + 1))
    for k in range(1, n_folds + 1):
        train_end = k * chunk
        test_end = (k + 1) * chunk if k < n_folds else n
        yield slice(0, train_end), slice(train_end, test_end)
