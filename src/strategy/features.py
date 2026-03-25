"""
Enhanced feature engineering combining BTC spot market data with Kalshi order book data.

Spot features  (from Binance 1m candles): RSI, MACD, Bollinger %B, realised
volatility, momentum, VWAP deviation, relative volume.

Book features  (from Kalshi ticker): spread, imbalance, depth, mid-returns,
short-horizon volatility.
"""
from __future__ import annotations

import numpy as np

from src.types import MarketState

ENHANCED_FEATURE_NAMES: list[str] = [
    # BTC spot
    "rsi_14",
    "macd_histogram",
    "bb_pct",
    "realized_vol_5m",
    "realized_vol_15m",
    "momentum_5m",
    "momentum_15m",
    "momentum_60m",
    "vwap_deviation",
    "volume_ratio",
    # Kalshi book
    "spread",
    "imbalance",
    "depth",
    "trade_minus_mid",
    "ret_1",
    "ret_3",
    "ret_5",
    "vol_10",
]


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    if len(data) < period:
        return data.copy()
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_macd_histogram(
    closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> float:
    if len(closes) < slow + signal:
        return 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    if len(macd_line) < signal:
        return 0.0
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return float(histogram[-1]) if len(histogram) > 0 else 0.0


def compute_bollinger_pct(
    closes: np.ndarray, period: int = 20, num_std: float = 2.0
) -> float:
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    sma = float(np.mean(window))
    std = float(np.std(window))
    if std == 0:
        return 0.5
    lower = sma - num_std * std
    band_width = 2 * num_std * std
    if band_width == 0:
        return 0.5
    return float((closes[-1] - lower) / band_width)


def compute_realized_volatility(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 0.0
    window = closes[-(period + 1) :]
    log_rets = np.diff(np.log(np.maximum(window, 1e-10)))
    return float(np.std(log_rets))


def compute_momentum(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 0.0
    base = closes[-(period + 1)]
    if base == 0:
        return 0.0
    return float((closes[-1] - base) / base)


def compute_vwap_deviation(ohlcv: np.ndarray, period: int = 20) -> float:
    if len(ohlcv) < period:
        return 0.0
    w = ohlcv[-period:]
    typical = (w[:, 1] + w[:, 2] + w[:, 3]) / 3.0
    vol = w[:, 4]
    total_vol = float(np.sum(vol))
    if total_vol == 0:
        return 0.0
    vwap = float(np.sum(typical * vol) / total_vol)
    if vwap == 0:
        return 0.0
    return float((ohlcv[-1, 3] - vwap) / vwap)


def compute_volume_ratio(volumes: np.ndarray, short: int = 5, long: int = 20) -> float:
    if len(volumes) < long:
        return 1.0
    short_avg = float(np.mean(volumes[-short:]))
    long_avg = float(np.mean(volumes[-long:]))
    if long_avg == 0:
        return 1.0
    return float(short_avg / long_avg)


# ---------------------------------------------------------------------------
# Kalshi book helpers
# ---------------------------------------------------------------------------

def _mid_price_cents(state: MarketState) -> int | None:
    if state.bid_cents is not None and state.ask_cents is not None:
        return (state.bid_cents + state.ask_cents) // 2
    return state.last_trade_cents


def _safe_ret(mids: np.ndarray, lookback: int) -> float:
    if len(mids) <= lookback:
        return 0.0
    base = mids[-1 - lookback]
    if base == 0:
        return 0.0
    return float((mids[-1] - base) / base)


def _kalshi_features(kalshi_window: list[MarketState]) -> dict[str, float] | None:
    if len(kalshi_window) < 6:
        return None
    latest = kalshi_window[-1]
    if latest.bid_cents is None or latest.ask_cents is None:
        return None

    mids: list[float] = []
    for s in kalshi_window:
        m = _mid_price_cents(s)
        if m is None:
            return None
        mids.append(float(m))
    mid_arr = np.array(mids, dtype=float)

    spread = float(latest.ask_cents - latest.bid_cents)
    depth_total = max(1, latest.bid_size + latest.ask_size)
    imbalance = float(latest.bid_size - latest.ask_size) / float(depth_total)

    rets: list[float] = []
    for i in range(max(1, len(mid_arr) - 10), len(mid_arr)):
        prev = mid_arr[i - 1]
        if prev != 0:
            rets.append(float((mid_arr[i] - prev) / prev))
    vol_10 = float(np.std(np.array(rets, dtype=float))) if rets else 0.0

    last_trade = (
        latest.last_trade_cents
        if latest.last_trade_cents is not None
        else int(mid_arr[-1])
    )
    trade_minus_mid = float(last_trade - mid_arr[-1]) / 100.0
    depth = float(np.log1p(depth_total))

    return {
        "spread": spread,
        "imbalance": imbalance,
        "depth": depth,
        "trade_minus_mid": trade_minus_mid,
        "ret_1": _safe_ret(mid_arr, 1),
        "ret_3": _safe_ret(mid_arr, 3),
        "ret_5": _safe_ret(mid_arr, 5),
        "vol_10": vol_10,
    }


# ---------------------------------------------------------------------------
# BTC spot feature block — works for both live and training
# ---------------------------------------------------------------------------

_SPOT_NEUTRAL: dict[str, float] = {
    "rsi_14": 50.0,
    "macd_histogram": 0.0,
    "bb_pct": 0.5,
    "realized_vol_5m": 0.0,
    "realized_vol_15m": 0.0,
    "momentum_5m": 0.0,
    "momentum_15m": 0.0,
    "momentum_60m": 0.0,
    "vwap_deviation": 0.0,
    "volume_ratio": 1.0,
}


def _spot_features(
    closes: np.ndarray,
    volumes: np.ndarray | None = None,
    ohlcv: np.ndarray | None = None,
) -> dict[str, float]:
    if len(closes) < 30:
        return dict(_SPOT_NEUTRAL)
    feats: dict[str, float] = {}
    feats["rsi_14"] = compute_rsi(closes, 14)
    feats["macd_histogram"] = compute_macd_histogram(closes)
    feats["bb_pct"] = compute_bollinger_pct(closes)
    feats["realized_vol_5m"] = compute_realized_volatility(closes, 5)
    feats["realized_vol_15m"] = compute_realized_volatility(closes, 15)
    feats["momentum_5m"] = compute_momentum(closes, 5)
    feats["momentum_15m"] = compute_momentum(closes, 15)
    feats["momentum_60m"] = compute_momentum(closes, 60)
    feats["vwap_deviation"] = (
        compute_vwap_deviation(ohlcv) if ohlcv is not None and len(ohlcv) >= 20 else 0.0
    )
    feats["volume_ratio"] = (
        compute_volume_ratio(volumes) if volumes is not None and len(volumes) >= 20 else 1.0
    )
    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_enhanced_features(
    kalshi_window: list[MarketState],
    btc_closes: np.ndarray | None = None,
    btc_volumes: np.ndarray | None = None,
    btc_ohlcv: np.ndarray | None = None,
) -> dict[str, float] | None:
    """
    Build the full 18-feature vector combining BTC spot indicators with
    Kalshi order-book features.

    Accepts raw numpy arrays so it works identically for live inference
    (fed from BinanceDataStore) and historical training (fed from DB).
    """
    book = _kalshi_features(kalshi_window)
    if book is None:
        return None

    if btc_closes is not None and len(btc_closes) >= 30:
        spot = _spot_features(btc_closes, btc_volumes, btc_ohlcv)
    else:
        spot = dict(_SPOT_NEUTRAL)

    return {**spot, **book}
