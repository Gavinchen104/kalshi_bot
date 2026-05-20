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


def horizon_matched_vol(
    closes: np.ndarray,
    horizon_seconds: float,
    floor_min: int = 60,
    cap_min: int = 1440,
) -> float | None:
    """Realized vol with a lookback window scaled to the contract horizon.

    Window in minutes = clamp(round(horizon_seconds / 60), floor_min, cap_min).
    For a daily (24h) contract this uses ~1440 1-min closes; for a 15-min
    contract it falls back to floor_min (typically 60). This is Phase 2 WS1's
    direct fix for "60-min vol applied to 10-34h horizons" mispricing.
    Returns None if there isn't enough candle history for the chosen window.
    """
    if horizon_seconds is None or horizon_seconds <= 0:
        return None
    horizon_min = max(1, int(round(horizon_seconds / 60.0)))
    window = max(floor_min, min(cap_min, horizon_min))
    return close_to_close_vol(closes, window=window)
