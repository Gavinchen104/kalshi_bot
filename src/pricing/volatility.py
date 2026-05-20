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


# Default blend windows: 1h captures intraday regime, 6h dampens 1h calm
# pockets, 24h anchors the daily scale. Phase 2 W1.2.
_DEFAULT_BLEND_WINDOWS_MIN = (60, 360, 1440)


def blend_vol(
    closes: np.ndarray,
    horizon_seconds: float | None = None,
    windows_min: tuple[int, ...] = _DEFAULT_BLEND_WINDOWS_MIN,
) -> float | None:
    """Simple-average realized vol across multiple lookback windows.

    Robust to any single window landing in a calm pocket — the failure mode
    we observed in Phase 1 (a calm 60-min window understates true daily vol).
    `horizon_seconds` is accepted for API parity with horizon_matched_vol but
    is not used in this v1 implementation; a future revision may horizon-weight
    the blend. Returns None if no window has enough history.
    """
    vols = [v for w in windows_min if (v := close_to_close_vol(closes, window=w)) is not None]
    if not vols:
        return None
    return float(sum(vols) / len(vols))


def long_window_floor_vol(closes: np.ndarray, lookback_days: int) -> float | None:
    """Trailing realized vol over `lookback_days` of 1-min closes.

    Intended as a *floor* on whichever short-window estimator the pricer uses:
    short windows can land in calm pockets and understate, but the trailing
    multi-day vol cannot. Phase 2 W1.4. Returns None if `lookback_days <= 0`
    or there isn't enough history.
    """
    if lookback_days is None or lookback_days <= 0:
        return None
    window = lookback_days * 24 * 60
    return close_to_close_vol(closes, window=window)


def ewma_vol(
    closes: np.ndarray,
    horizon_seconds: float | None = None,
    half_life_min: int | None = None,
) -> float | None:
    """Exponentially-weighted realized vol over 1-min log returns.

    A single calm hour drags an unweighted window down; EWMA gives the latest
    minute the highest weight and decays smoothly, so regime shifts surface
    sooner. ``half_life_min`` defaults to ``max(30, min(1440, horizon_min // 4))``
    when not supplied — long-horizon contracts get longer memory. Phase 2 W1.3.
    Returns None if there isn't at least two closes or returns are degenerate.
    """
    if closes is None or len(closes) < 2:
        return None
    if half_life_min is None:
        h = max(1, int((horizon_seconds or 0) / 60.0))
        # /4 means ~94% of weight inside one horizon window.
        half_life_min = max(30, min(1440, max(60, h // 4)))
    log_returns = np.diff(np.log(closes))
    if log_returns.size == 0:
        return None
    lam = 0.5 ** (1.0 / float(half_life_min))
    n = log_returns.size
    weights = (1.0 - lam) * np.power(lam, np.arange(n)[::-1])
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return None
    weights = weights / w_sum  # renormalize for finite samples
    var_min = float(np.sum(weights * log_returns**2))
    if not math.isfinite(var_min) or var_min <= 0:
        return None
    return math.sqrt(var_min) * math.sqrt(MINUTES_PER_YEAR)
