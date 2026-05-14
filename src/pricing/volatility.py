"""
Realized-volatility estimators from 1-minute Coinbase BTC-USD closes.

Annualized using 365 * 24 * 60 minutes per year (crypto trades 24/7).
"""
from __future__ import annotations

import math

import numpy as np


MINUTES_PER_YEAR = 365 * 24 * 60


def close_to_close_vol(closes: np.ndarray, window: int = 60) -> float | None:
    """Standard close-to-close realized vol over the last `window` 1-min closes."""
    if closes is None or len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    log_returns = np.diff(np.log(tail))
    if log_returns.size == 0:
        return None
    sigma_min = float(np.std(log_returns, ddof=1))
    if not math.isfinite(sigma_min) or sigma_min <= 0:
        return None
    return sigma_min * math.sqrt(MINUTES_PER_YEAR)


def clamp_vol(sigma: float | None, floor: float, ceiling: float) -> float | None:
    if sigma is None:
        return None
    if not math.isfinite(sigma):
        return None
    return max(floor, min(ceiling, sigma))
