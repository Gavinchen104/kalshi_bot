"""
Model abstraction layer.

Supports three model types:
  1. GBMModel        – LightGBM classifier (new default)
  2. LinearProbModel – single logistic regression (legacy)
  3. LinearProbEnsemble – bootstrap ensemble of (2) (legacy)

``load_model(path)`` auto-detects the format from the JSON metadata file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.strategy.features import ENHANCED_FEATURE_NAMES
from src.types import MarketState

# Legacy 8-feature list kept for old model files
FEATURE_NAMES = [
    "spread",
    "imbalance",
    "ret_1",
    "ret_3",
    "ret_5",
    "vol_10",
    "depth",
    "trade_minus_mid",
]


# ---------------------------------------------------------------------------
# GBM (LightGBM) model – the new default
# ---------------------------------------------------------------------------

class GBMModel:
    """LightGBM booster wrapped with feature-name metadata and save/load."""

    def __init__(
        self,
        booster: Any,
        feature_names: list[str],
        metadata: dict[str, Any] | None = None,
    ):
        self._booster = booster
        self.feature_names = feature_names
        self._metadata = metadata or {}
        self.model_name: str = self._metadata.get("model_name", "btc_lgbm_v1")

    def predict_proba(self, feature_map: dict[str, float]) -> float:
        x = np.array(
            [[feature_map.get(n, 0.0) for n in self.feature_names]], dtype=float
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        prob = self._booster.predict(x)[0]
        return float(np.clip(prob, 0.01, 0.99))

    def predict_proba_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(self._booster.predict(x), 0.01, 0.99)

    @property
    def feature_importance(self) -> dict[str, float]:
        imp = self._booster.feature_importance(importance_type="gain")
        total = float(np.sum(imp)) or 1.0
        return {n: round(float(v) / total, 4) for n, v in zip(self.feature_names, imp)}

    def save(self, path: str) -> None:
        import lightgbm as lgb  # noqa: F811

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        booster_path = str(out.with_suffix(".lgb"))
        self._booster.save_model(booster_path)
        meta = {
            "type": "gbm",
            "feature_names": self.feature_names,
            "booster_path": Path(booster_path).name,
            **self._metadata,
        }
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "GBMModel":
        import lightgbm as lgb  # noqa: F811

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        booster_file = raw.get("booster_path", Path(path).with_suffix(".lgb").name)
        booster_path = str(Path(path).parent / booster_file)
        booster = lgb.Booster(model_file=booster_path)
        return cls(
            booster=booster,
            feature_names=raw.get("feature_names", list(ENHANCED_FEATURE_NAMES)),
            metadata=raw,
        )


# ---------------------------------------------------------------------------
# Legacy linear models (kept for backward-compatible loading)
# ---------------------------------------------------------------------------

@dataclass
class LinearProbModel:
    feature_names: list[str]
    weights: list[float]
    bias: float
    means: list[float]
    stds: list[float]
    model_name: str = "btc_15m_linear_prob_v1"

    def predict_proba(self, feature_map: dict[str, float]) -> float:
        x = np.array(
            [feature_map.get(name, 0.0) for name in self.feature_names], dtype=float
        )
        means = np.array(self.means, dtype=float)
        stds = np.array(self.stds, dtype=float)
        x_norm = (x - means) / stds
        z = float(np.dot(np.array(self.weights, dtype=float), x_norm) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "stds": self.stds,
            "model_name": self.model_name,
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LinearProbModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=list(raw["feature_names"]),
            weights=list(raw["weights"]),
            bias=float(raw["bias"]),
            means=list(raw["means"]),
            stds=list(raw["stds"]),
            model_name=str(raw.get("model_name", "btc_15m_linear_prob_v1")),
        )


@dataclass
class LinearProbEnsemble:
    models: list[LinearProbModel]

    @property
    def model_name(self) -> str:
        return f"ensemble_{len(self.models)}"

    def predict_proba(self, feature_map: dict[str, float]) -> float:
        if not self.models:
            return 0.5
        return float(np.mean([m.predict_proba(feature_map) for m in self.models]))

    def to_dict(self) -> dict[str, Any]:
        return {"ensemble": [m.to_dict() for m in self.models]}

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LinearProbEnsemble":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        models = [
            LinearProbModel(
                feature_names=list(m["feature_names"]),
                weights=list(m["weights"]),
                bias=float(m["bias"]),
                means=list(m["means"]),
                stds=list(m["stds"]),
                model_name=str(m.get("model_name", "btc_15m_linear_prob_v1")),
            )
            for m in raw["ensemble"]
        ]
        return cls(models=models)


# ---------------------------------------------------------------------------
# Legacy feature builder (used only for old-format models)
# ---------------------------------------------------------------------------

def build_feature_map(window: list[MarketState]) -> dict[str, float] | None:
    if len(window) < 6:
        return None
    mids = [_mid_price_cents(s) for s in window]
    if any(m is None for m in mids):
        return None
    mid_arr = np.array([float(m) for m in mids if m is not None], dtype=float)
    if len(mid_arr) < 6:
        return None

    latest = window[-1]
    if latest.bid_cents is None or latest.ask_cents is None:
        return None

    spread = float(latest.ask_cents - latest.bid_cents)
    depth_total = max(1, latest.bid_size + latest.ask_size)
    imbalance = float(latest.bid_size - latest.ask_size) / float(depth_total)
    ret_1 = _safe_ret(mid_arr, 1)
    ret_3 = _safe_ret(mid_arr, 3)
    ret_5 = _safe_ret(mid_arr, 5)

    returns: list[float] = []
    for i in range(max(1, len(mid_arr) - 10), len(mid_arr)):
        prev = mid_arr[i - 1]
        curr = mid_arr[i]
        if prev != 0:
            returns.append((curr - prev) / prev)
    vol_10 = float(np.std(np.array(returns, dtype=float))) if returns else 0.0

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
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "vol_10": vol_10,
        "depth": depth,
        "trade_minus_mid": trade_minus_mid,
    }


def _mid_price_cents(state: MarketState) -> int | None:
    if state.bid_cents is None or state.ask_cents is None:
        return state.last_trade_cents
    return (state.bid_cents + state.ask_cents) // 2


def _safe_ret(mids: np.ndarray, lookback: int) -> float:
    if len(mids) <= lookback:
        return 0.0
    base = mids[-1 - lookback]
    if base == 0:
        return 0.0
    return float((mids[-1] - base) / base)


# ---------------------------------------------------------------------------
# Universal loader
# ---------------------------------------------------------------------------

from typing import Union

ModelType = Union[GBMModel, LinearProbModel, LinearProbEnsemble]


def load_model(path: str) -> ModelType:
    """Auto-detect the stored format and return the right model class."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("type") == "gbm":
        return GBMModel.load(path)
    if "ensemble" in raw:
        return LinearProbEnsemble.load(path)
    return LinearProbModel.load(path)
