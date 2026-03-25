"""
Training pipeline for BTC prediction model.

Accepts a Recipe from the strategy explorer to control hyperparameters,
horizons, feature subsets, and price filters.  Falls through gracefully:

  1. **Spot-direction** — predicts BTC spot return direction using Coinbase
     1m candles.  Only used when sufficient balanced data exists.
  2. **Kalshi-tick** — predicts mid-price direction on uncertain (near-
     the-money) Kalshi contracts.

Walk-forward cross-validation determines whether the trained model beats
the naive majority-class baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from src.storage.repository import Repository
from src.strategy.features import ENHANCED_FEATURE_NAMES, build_enhanced_features
from src.strategy.ml_model import GBMModel
from src.training.strategy_explorer import Recipe
from src.types import MarketState

SPOT_FEATURE_NAMES = [n for n in ENHANCED_FEATURE_NAMES if n in {
    "rsi_14", "macd_histogram", "bb_pct", "realized_vol_5m", "realized_vol_15m",
    "momentum_5m", "momentum_15m", "momentum_60m", "vwap_deviation", "volume_ratio",
}]

BOOK_FEATURE_NAMES = [n for n in ENHANCED_FEATURE_NAMES if n in {
    "spread", "imbalance", "depth", "trade_minus_mid",
    "ret_1", "ret_3", "ret_5", "vol_10",
}]


def _feature_names_for(subset: str) -> list[str]:
    if subset == "spot_only":
        return list(SPOT_FEATURE_NAMES)
    if subset == "book_only":
        return list(BOOK_FEATURE_NAMES)
    return list(ENHANCED_FEATURE_NAMES)


# ── public entry point ────────────────────────────────────────────────────

def train_from_repository(
    db_path: str,
    market_like: str,
    target_market_regex: str,
    model_output_path: str,
    lookback: int = 20,
    horizon: int = 15,
    limit: int = 100_000,
    epochs: int = 200,
    lr: float = 0.05,
    n_ensemble_models: int = 3,
    recipe: Recipe | None = None,
) -> dict[str, float]:
    if recipe is None:
        recipe = Recipe(name="default", horizon=horizon, lookback=lookback)

    repo = Repository(db_path)
    btc_candles = repo.btc_candle_series(limit=100_000)

    # Try spot-direction first when we have enough candle history
    if len(btc_candles) >= 120:
        try:
            return _train_spot_direction(
                repo, btc_candles, market_like, target_market_regex,
                model_output_path, recipe,
            )
        except RuntimeError:
            pass

    return _train_kalshi_tick(
        repo, btc_candles, market_like, target_market_regex,
        model_output_path, recipe,
    )


# ── MODE 1: Spot-direction ────────────────────────────────────────────────

def _train_spot_direction(
    repo: Repository,
    btc_candles: list[dict],
    market_like: str,
    target_market_regex: str,
    model_output_path: str,
    recipe: Recipe,
) -> dict[str, float]:
    """
    Iterate over BTC candles (1 per minute), build features from the nearest
    Kalshi state + spot context, predict BTC return direction.  One sample per
    candle minute avoids label duplication.
    """
    import bisect
    from datetime import datetime, timezone

    # Load states covering the BTC candle time window (not just the latest N)
    candle_start = datetime.fromtimestamp(btc_candles[0]["timestamp_ms"] / 1000, tz=timezone.utc)
    candle_end = datetime.fromtimestamp(btc_candles[-1]["timestamp_ms"] / 1000, tz=timezone.utc)
    all_states = repo.market_state_series_by_time(
        market_like=market_like,
        after_iso=candle_start.isoformat(),
        before_iso=candle_end.isoformat(),
        sample_every_n=50,  # ~1 per 500ms instead of every 10ms
    )
    if not all_states:
        raise RuntimeError("No market states in candle time range")

    # Time-sorted index for fast nearest-state lookup
    state_ts_pairs = sorted((s.updated_at.timestamp(), i) for i, s in enumerate(all_states))
    state_ts_arr = [t for t, _ in state_ts_pairs]
    state_idx_arr = [i for _, i in state_ts_pairs]

    closes = np.array([c["close"] for c in btc_candles], dtype=float)
    timestamps_ms = np.array([c["timestamp_ms"] for c in btc_candles], dtype=np.int64)

    feature_names = _feature_names_for(recipe.feature_subset)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    hz = max(recipe.horizon, 5)

    for candle_idx in range(max(30, recipe.lookback), len(closes) - hz):
        current_close = closes[candle_idx]
        future_close = closes[candle_idx + hz]
        if current_close == 0:
            continue
        ret = (future_close - current_close) / current_close
        if abs(ret) < 0.00005:
            continue
        y = 1 if ret > 0 else 0

        candle_ts = timestamps_ms[candle_idx] / 1000.0
        pos = bisect.bisect_left(state_ts_arr, candle_ts)
        best_si = None
        best_diff = 120.0
        for cand in (pos - 1, pos, pos + 1):
            if 0 <= cand < len(state_ts_arr):
                diff = abs(state_ts_arr[cand] - candle_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_si = state_idx_arr[cand]
        if best_si is None:
            continue

        window_start = max(0, best_si - recipe.lookback)
        window = all_states[window_start: best_si + 1]
        if len(window) < 6:
            continue

        sc, sv, so = _spot_arrays(btc_candles, candle_idx)
        fmap = build_enhanced_features(window, sc, sv, so)
        if fmap is None:
            continue
        x_rows.append([fmap.get(n, 0.0) for n in feature_names])
        y_rows.append(y)

    if len(x_rows) < 80:
        raise RuntimeError(f"Only {len(x_rows)} spot-direction rows")

    x = _clean(np.array(x_rows, dtype=float))
    y = np.array(y_rows, dtype=float)

    pos_rate = float(np.mean(y))
    if pos_rate < 0.15 or pos_rate > 0.85:
        raise RuntimeError(f"Spot target imbalanced ({pos_rate:.1%})")

    metrics = _train_lgbm(x, y, feature_names, model_output_path, recipe)
    wf = _walk_forward(x, y, feature_names, recipe)
    edge = wf.get("wf_mean_accuracy", 0) - wf.get("wf_mean_baseline", 1)

    return {
        "rows": float(len(x)),
        "training_mode": "spot_direction",
        "recipe": recipe.name,
        "class_balance": round(pos_rate, 3),
        "beats_baseline": 1.0 if edge > 0 else 0.0,
        **metrics, **wf,
    }


# ── MODE 2: Kalshi-tick ───────────────────────────────────────────────────

def _train_kalshi_tick(
    repo: Repository,
    btc_candles: list[dict],
    market_like: str,
    target_market_regex: str,
    model_output_path: str,
    recipe: Recipe,
) -> dict[str, float]:
    # Load enough data to cover the horizon (time-based, in minutes)
    # With ~500 states/sec across markets, 500K covers ~15-20 minutes
    data_limit = max(100_000, recipe.horizon * 40_000)
    states = repo.market_state_series(market_like=market_like, limit=data_limit)
    grouped: dict[str, list[MarketState]] = defaultdict(list)
    for s in states:
        grouped[s.market_id].append(s)

    has_spot = len(btc_candles) >= 30
    feature_names = _feature_names_for(recipe.feature_subset)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    horizon_secs = recipe.horizon * 60  # horizon is now in MINUTES

    for market_id, series in grouped.items():
        if target_market_regex and not _matches_regex(market_id, target_market_regex):
            continue

        # Build timestamp index for fast time-based lookups
        ts_list = [s.updated_at.timestamp() for s in series]

        # Sample at ~1 per minute to avoid duplicate near-identical samples
        last_sample_ts = 0.0
        for i in range(recipe.lookback, len(series)):
            state = series[i]
            state_ts = ts_list[i]
            if state_ts - last_sample_ts < 30:  # at most 1 sample per 30s
                continue

            mid = _price_for_label(state)
            if mid is None or mid < recipe.mid_price_lo or mid > recipe.mid_price_hi:
                continue

            # Find the state closest to state_ts + horizon_minutes
            target_ts = state_ts + horizon_secs
            fut_idx = _find_nearest_idx(ts_list, target_ts, i + 1, tolerance=60)
            if fut_idx is None:
                continue
            fut = _price_for_label(series[fut_idx])
            if fut is None or fut == mid:
                continue

            window = series[max(0, i - recipe.lookback): i + 1]
            spot_ctx = _spot_context_for(state.updated_at, btc_candles) if has_spot else None
            fmap = build_enhanced_features(window, *(spot_ctx or (None, None, None)))
            if fmap is None:
                continue

            y = 1 if fut > mid else 0
            x_rows.append([fmap.get(n, 0.0) for n in feature_names])
            y_rows.append(y)
            last_sample_ts = state_ts

    if len(x_rows) < 100:
        raise RuntimeError(f"Only {len(x_rows)} rows")

    x = _clean(np.array(x_rows, dtype=float))
    y = np.array(y_rows, dtype=float)

    try:
        metrics = _train_lgbm(x, y, feature_names, model_output_path, recipe)
    except ImportError:
        metrics = _train_linear_fallback(x, y, feature_names, model_output_path)

    wf = _walk_forward(x, y, feature_names, recipe)
    edge = wf.get("wf_mean_accuracy", 0) - wf.get("wf_mean_baseline", 1)

    return {
        "rows": float(len(x)),
        "training_mode": "kalshi_tick",
        "recipe": recipe.name,
        "beats_baseline": 1.0 if edge > 0 else 0.0,
        **metrics, **wf,
    }


# ── LightGBM core ────────────────────────────────────────────────────────

def _train_lgbm(
    x: np.ndarray, y: np.ndarray, feature_names: list[str],
    output_path: str, recipe: Recipe,
) -> dict[str, float]:
    import lightgbm as lgb

    split = int(0.8 * len(x))
    x_tr, x_vl = x[:split], x[split:]
    y_tr, y_vl = y[:split], y[split:]

    ds_tr = lgb.Dataset(x_tr, label=y_tr, feature_name=feature_names)
    ds_vl = lgb.Dataset(x_vl, label=y_vl, feature_name=feature_names, reference=ds_tr)

    params = recipe.to_lgbm_params()
    cbs = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
    booster = lgb.train(params, ds_tr, num_boost_round=800, valid_sets=[ds_vl], callbacks=cbs)

    vp = np.clip(booster.predict(x_vl), 1e-6, 1 - 1e-6)
    val_acc = float(np.mean((vp >= 0.5) == y_vl))
    val_ll = float(-np.mean(y_vl * np.log(vp) + (1 - y_vl) * np.log(1 - vp)))
    baseline = float(max(np.mean(y_vl), 1 - np.mean(y_vl)))

    model = GBMModel(
        booster=booster, feature_names=feature_names,
        metadata={
            "model_name": f"btc_lgbm_{recipe.name}",
            "best_iteration": booster.best_iteration,
            "val_accuracy": val_acc, "val_logloss": val_ll,
            "recipe": recipe.name,
        },
    )
    model.save(output_path)

    imp = model.feature_importance
    return {
        "model_type": 1.0,
        "val_accuracy": val_acc, "val_logloss": val_ll,
        "val_baseline": baseline,
        "best_iteration": float(booster.best_iteration),
        **{f"imp_{k}": v for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:5]},
    }


# ── Walk-forward ──────────────────────────────────────────────────────────

def _walk_forward(
    x: np.ndarray, y: np.ndarray, feature_names: list[str],
    recipe: Recipe, n_folds: int = 5,
) -> dict[str, float]:
    fold_size = len(x) // (n_folds + 1)
    if fold_size < 40:
        return {}
    try:
        import lightgbm as lgb
    except ImportError:
        return _wf_linear(x, y, n_folds)

    params = recipe.to_lgbm_params()
    accs, lls, baselines = [], [], []
    for fold in range(n_folds):
        te = (fold + 1) * fold_size
        vs, ve = te, te + fold_size
        tr = lgb.Dataset(x[:te], label=y[:te], feature_name=feature_names)
        vl = lgb.Dataset(x[vs:ve], label=y[vs:ve], feature_name=feature_names, reference=tr)
        cb = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        bst = lgb.train(params, tr, num_boost_round=400, valid_sets=[vl], callbacks=cb)
        vp = np.clip(bst.predict(x[vs:ve]), 1e-6, 1 - 1e-6)
        yv = y[vs:ve]
        accs.append(float(np.mean((vp >= 0.5) == yv)))
        lls.append(float(-np.mean(yv * np.log(vp) + (1 - yv) * np.log(1 - vp))))
        baselines.append(float(max(np.mean(yv), 1 - np.mean(yv))))

    return {
        "wf_folds": float(n_folds),
        "wf_mean_accuracy": float(np.mean(accs)),
        "wf_mean_logloss": float(np.mean(lls)),
        "wf_mean_baseline": float(np.mean(baselines)),
        "wf_beat_baseline_count": float(sum(a > b for a, b in zip(accs, baselines))),
    }


# ── Linear fallbacks ─────────────────────────────────────────────────────

def _train_linear_fallback(x, y, feature_names, output_path) -> dict[str, float]:
    from src.strategy.ml_model import LinearProbModel
    split = int(0.8 * len(x))
    x_tr, x_vl = x[:split], x[split:]
    y_tr, y_vl = y[:split], y[split:]
    mu, sd = x_tr.mean(0), np.where(x_tr.std(0) == 0, 1, x_tr.std(0))
    xtn = np.clip(np.nan_to_num((x_tr - mu) / sd), -10, 10)
    xvn = np.clip(np.nan_to_num((x_vl - mu) / sd), -10, 10)
    w, b = np.zeros(xtn.shape[1]), 0.0
    for _ in range(200):
        p = _sigmoid(xtn @ w + b)
        err = p - y_tr
        w -= 0.05 * np.clip(np.nan_to_num((xtn.T @ err) / len(xtn)), -1, 1)
        b -= 0.05 * float(np.clip(np.mean(err), -1, 1))
    LinearProbModel(
        feature_names=list(feature_names), weights=w.tolist(),
        bias=float(b), means=mu.tolist(), stds=sd.tolist(),
    ).save(output_path)
    vp = np.clip(_sigmoid(xvn @ w + b), 1e-6, 1 - 1e-6)
    return {
        "model_type": 0.0,
        "val_accuracy": float(np.mean((vp >= 0.5) == y_vl)),
        "val_logloss": float(-np.mean(y_vl * np.log(vp) + (1 - y_vl) * np.log(1 - vp))),
    }


def _wf_linear(x, y, n_folds=5):
    fold_size = len(x) // (n_folds + 1)
    if fold_size < 40:
        return {}
    accs, lls, baselines = [], [], []
    for fold in range(n_folds):
        te = (fold + 1) * fold_size
        xtr, ytr = x[:te], y[:te]
        xvl, yvl = x[te:te + fold_size], y[te:te + fold_size]
        mu, sd = xtr.mean(0), np.where(xtr.std(0) == 0, 1, xtr.std(0))
        xtn = np.clip(np.nan_to_num((xtr - mu) / sd), -10, 10)
        xvn = np.clip(np.nan_to_num((xvl - mu) / sd), -10, 10)
        w, b = np.zeros(xtn.shape[1]), 0.0
        for _ in range(100):
            p = _sigmoid(xtn @ w + b)
            err = p - ytr
            w -= 0.05 * np.clip(np.nan_to_num((xtn.T @ err) / len(xtn)), -1, 1)
            b -= 0.05 * float(np.clip(np.mean(err), -1, 1))
        vp = np.clip(_sigmoid(xvn @ w + b), 1e-6, 1 - 1e-6)
        accs.append(float(np.mean((vp >= 0.5) == yvl)))
        lls.append(float(-np.mean(yvl * np.log(vp) + (1 - yvl) * np.log(1 - vp))))
        baselines.append(float(max(np.mean(yvl), 1 - np.mean(yvl))))
    return {
        "wf_folds": float(n_folds),
        "wf_mean_accuracy": float(np.mean(accs)),
        "wf_mean_logloss": float(np.mean(lls)),
        "wf_mean_baseline": float(np.mean(baselines)),
        "wf_beat_baseline_count": float(sum(a > b for a, b in zip(accs, baselines))),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _spot_arrays(candles, idx, window=100):
    start = max(0, idx - window + 1)
    slc = candles[start: idx + 1]
    c = np.array([r["close"] for r in slc], dtype=float)
    v = np.array([r["volume"] for r in slc], dtype=float)
    o = np.array([[r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in slc], dtype=float)
    return c, v, o


def _find_nearest_idx(ts_list: list[float], target_ts: float,
                      start: int, tolerance: float = 60) -> int | None:
    """Binary search for the index in ts_list closest to target_ts (seconds)."""
    import bisect
    idx = bisect.bisect_left(ts_list, target_ts, lo=start)
    best = None
    best_diff = tolerance + 1
    for candidate in (idx - 1, idx, idx + 1):
        if start <= candidate < len(ts_list):
            diff = abs(ts_list[candidate] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = candidate
    return best if best_diff <= tolerance else None


def _find_candle_idx(ts_array: np.ndarray, target_ms: int):
    idx = int(np.searchsorted(ts_array, target_ms, side="right")) - 1
    if idx < 0 or idx >= len(ts_array):
        return None
    if abs(ts_array[idx] - target_ms) > 120_000:
        return None
    return idx


def _spot_context_for(ts, candles, window=100):
    ts_ms = int(ts.timestamp() * 1000)
    relevant = [c for c in candles if c["timestamp_ms"] <= ts_ms]
    if len(relevant) < 30:
        return None
    relevant = relevant[-window:]
    closes = np.array([c["close"] for c in relevant], dtype=float)
    volumes = np.array([c["volume"] for c in relevant], dtype=float)
    ohlcv = np.array(
        [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in relevant], dtype=float)
    return closes, volumes, ohlcv


def _price_for_label(state):
    if state.last_trade_cents is not None:
        return state.last_trade_cents
    if state.bid_cents is not None and state.ask_cents is not None:
        return (state.bid_cents + state.ask_cents) // 2
    return None


def _matches_regex(market_id, pattern):
    import re
    return re.search(pattern, market_id, flags=re.IGNORECASE) is not None


def _sigmoid(x):
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def _clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/bot.db")
    parser.add_argument("--market-like", default="BTC")
    parser.add_argument("--target-market-regex", default=".*BTC.*")
    parser.add_argument("--model-output-path", default="data/models/btc_15m_model.json")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    m = train_from_repository(
        db_path=args.db_path, market_like=args.market_like,
        target_market_regex=args.target_market_regex,
        model_output_path=args.model_output_path,
        lookback=args.lookback, horizon=args.horizon, limit=args.limit,
    )
    print("Training complete")
    for k, v in sorted(m.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
