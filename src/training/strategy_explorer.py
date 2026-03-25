"""
Adaptive strategy explorer.

Maintains a pool of training "recipes" — different combinations of
hyperparameters, horizons, feature subsets, and price filters.  Each retrain
cycle tries a recipe, records its walk-forward score, and promotes the best
one as champion.  When the champion decays (PnL goes negative), exploration
resumes.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.monitoring.logging import get_logger

logger = get_logger("strategy_explorer")


@dataclass
class Recipe:
    name: str
    horizon: int = 15
    num_leaves: int = 31
    learning_rate: float = 0.03
    max_depth: int = 5
    min_child_samples: int = 30
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    lambda_l1: float = 0.5
    lambda_l2: float = 2.0
    mid_price_lo: int = 15
    mid_price_hi: int = 85
    feature_subset: str = "all"  # "all", "spot_only", "book_only", "top10"
    lookback: int = 20

    @property
    def config_hash(self) -> str:
        d = asdict(self)
        d.pop("name", None)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_lgbm_params(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 5,
            "min_child_samples": self.min_child_samples,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
            "verbose": -1,
        }


# Pre-defined diverse recipe pool.  Most-promising configs first so the
# explorer finds edge quickly.  Tighter mid-price filters are tried early
# because they produce balanced targets where the model can learn real
# patterns instead of just predicting the majority class.
RECIPE_POOL: list[Recipe] = [
    # --- Coin-flip zone first (35–65c = most uncertainty) ---
    Recipe(name="h10_coinflip", horizon=10, mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h15_coinflip", horizon=15, mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h20_coinflip", horizon=20, mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h5_coinflip", horizon=5, mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h30_coinflip", horizon=30, mid_price_lo=35, mid_price_hi=65),

    # --- Coinflip + complexity variations ---
    Recipe(name="h15_cf_complex", horizon=15, mid_price_lo=35, mid_price_hi=65,
           num_leaves=63, max_depth=7, min_child_samples=15),
    Recipe(name="h10_cf_simple", horizon=10, mid_price_lo=35, mid_price_hi=65,
           num_leaves=15, max_depth=3, lambda_l1=2.0, lambda_l2=5.0),
    Recipe(name="h15_cf_fast", horizon=15, mid_price_lo=35, mid_price_hi=65,
           learning_rate=0.1, num_leaves=15, max_depth=4),
    Recipe(name="h20_cf_slow", horizon=20, mid_price_lo=35, mid_price_hi=65,
           learning_rate=0.008, num_leaves=31, max_depth=6),

    # --- Tight filter (25–75c) ---
    Recipe(name="h10_tight", horizon=10, mid_price_lo=25, mid_price_hi=75),
    Recipe(name="h15_tight", horizon=15, mid_price_lo=25, mid_price_hi=75),
    Recipe(name="h20_tight", horizon=20, mid_price_lo=25, mid_price_hi=75),
    Recipe(name="h15_tight_complex", horizon=15, mid_price_lo=25, mid_price_hi=75,
           num_leaves=63, max_depth=7),

    # --- Feature subsets (coinflip zone for balanced data) ---
    Recipe(name="h15_cf_book", horizon=15, mid_price_lo=35, mid_price_hi=65,
           feature_subset="book_only"),
    Recipe(name="h15_cf_spot", horizon=15, mid_price_lo=35, mid_price_hi=65,
           feature_subset="spot_only"),
    Recipe(name="h20_cf_book", horizon=20, mid_price_lo=35, mid_price_hi=65,
           feature_subset="book_only"),

    # --- Tight (25-75) with varied complexity ---
    Recipe(name="h15_tight_simple", horizon=15, mid_price_lo=25, mid_price_hi=75,
           num_leaves=15, max_depth=3, lambda_l1=2.0, lambda_l2=5.0),
    Recipe(name="h20_tight_simple", horizon=20, mid_price_lo=25, mid_price_hi=75,
           num_leaves=15, max_depth=3, lambda_l1=2.0, lambda_l2=5.0),
    Recipe(name="h15_tight_fast", horizon=15, mid_price_lo=25, mid_price_hi=75,
           learning_rate=0.1, num_leaves=15, max_depth=4),
    Recipe(name="h15_tight_slow", horizon=15, mid_price_lo=25, mid_price_hi=75,
           learning_rate=0.008, num_leaves=31, max_depth=6),

    # --- Longer lookback (coinflip zone) ---
    Recipe(name="h15_lb40_cf", horizon=15, lookback=40, mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h20_lb40_cf", horizon=20, lookback=40, mid_price_lo=35, mid_price_hi=65),

    # --- Low feature fraction (more generalisation, coinflip zone) ---
    Recipe(name="h15_low_ff_cf", horizon=15, feature_fraction=0.4, bagging_fraction=0.5,
           mid_price_lo=35, mid_price_hi=65),
    Recipe(name="h20_low_ff_tight", horizon=20, feature_fraction=0.4, bagging_fraction=0.5,
           mid_price_lo=25, mid_price_hi=75),
]


@dataclass
class ExplorationResult:
    recipe_name: str
    config_hash: str
    wf_accuracy: float
    wf_baseline: float
    edge: float  # wf_accuracy - wf_baseline
    val_accuracy: float
    rows: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StrategyExplorer:
    """
    Manages the exploration-exploitation cycle.

    Exploration: try untried recipes round-robin.
    Exploitation: when a champion exists, keep using it.
    Re-explore: when champion decays (performance drops or PnL negative).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._champion: Recipe | None = None
        self._champion_edge: float = 0.0
        self._results: list[ExplorationResult] = []
        self._tried_hashes: set[str] = set()
        self._round_robin_idx = 0
        self._consecutive_no_edge = 0
        self._init_db()
        self._load_history()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exploration_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recipe_name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    wf_accuracy REAL,
                    wf_baseline REAL,
                    edge REAL,
                    val_accuracy REAL,
                    rows INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

    def _load_history(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT recipe_name, config_hash, wf_accuracy, wf_baseline, edge, val_accuracy, rows "
                "FROM exploration_runs ORDER BY id"
            ).fetchall()
        for r in rows:
            res = ExplorationResult(
                recipe_name=r[0], config_hash=r[1],
                wf_accuracy=r[2] or 0, wf_baseline=r[3] or 0,
                edge=r[4] or 0, val_accuracy=r[5] or 0, rows=r[6] or 0,
            )
            self._results.append(res)
            self._tried_hashes.add(r[1])

        # Restore champion = best result that had positive edge
        positive = [r for r in self._results if r.edge > 0]
        if positive:
            best = max(positive, key=lambda r: r.edge)
            for recipe in RECIPE_POOL:
                if recipe.config_hash == best.config_hash:
                    self._champion = recipe
                    self._champion_edge = best.edge
                    logger.info(
                        "champion_restored",
                        name=recipe.name, edge=best.edge,
                    )
                    break

    def pick_recipe(self) -> Recipe:
        """Pick the next recipe to try."""
        # Exploitation: if champion exists and still performing, use it
        if self._champion is not None and self._consecutive_no_edge < 3:
            return self._champion

        # Exploration: pick next untried recipe
        untried = [r for r in RECIPE_POOL if r.config_hash not in self._tried_hashes]
        if untried:
            recipe = untried[0]
            logger.info("exploring_new_recipe", name=recipe.name, remaining=len(untried) - 1)
            return recipe

        # All tried — mutate the best-performing one
        if self._results:
            best = max(self._results, key=lambda r: r.edge)
            base = None
            for r in RECIPE_POOL:
                if r.config_hash == best.config_hash:
                    base = r
                    break
            if base is not None:
                return self._mutate(base)

        # Fallback: round-robin
        recipe = RECIPE_POOL[self._round_robin_idx % len(RECIPE_POOL)]
        self._round_robin_idx += 1
        return recipe

    def record_result(self, recipe: Recipe, metrics: dict[str, float]) -> None:
        wf_acc = metrics.get("wf_mean_accuracy", 0)
        wf_base = metrics.get("wf_mean_baseline", 1)
        edge = wf_acc - wf_base

        result = ExplorationResult(
            recipe_name=recipe.name,
            config_hash=recipe.config_hash,
            wf_accuracy=wf_acc,
            wf_baseline=wf_base,
            edge=edge,
            val_accuracy=metrics.get("val_accuracy", 0),
            rows=int(metrics.get("rows", 0)),
        )
        self._results.append(result)
        self._tried_hashes.add(recipe.config_hash)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO exploration_runs "
                "(recipe_name, config_hash, config_json, wf_accuracy, wf_baseline, edge, val_accuracy, rows) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (recipe.name, recipe.config_hash, json.dumps(asdict(recipe)),
                 wf_acc, wf_base, edge, result.val_accuracy, result.rows),
            )

        if edge > 0:
            self._consecutive_no_edge = 0
            if edge > self._champion_edge:
                self._champion = recipe
                self._champion_edge = edge
                logger.info(
                    "new_champion",
                    name=recipe.name,
                    edge=round(edge * 100, 2),
                    wf_acc=round(wf_acc, 4),
                )
        else:
            self._consecutive_no_edge += 1
            if self._consecutive_no_edge >= 3 and self._champion is not None:
                logger.info(
                    "champion_invalidated",
                    old_champion=self._champion.name,
                    consecutive_misses=self._consecutive_no_edge,
                )
                self._champion = None
                self._champion_edge = 0.0

    def has_champion(self) -> bool:
        return self._champion is not None and self._champion_edge > 0

    def signal_pnl_decay(self) -> None:
        """Called when live PnL turns negative. Forces re-exploration."""
        if self._champion is not None:
            logger.info("pnl_decay_detected", champion=self._champion.name)
            self._champion = None
            self._champion_edge = 0.0
            self._consecutive_no_edge = 0

    @property
    def champion_name(self) -> str | None:
        return self._champion.name if self._champion else None

    @property
    def total_explored(self) -> int:
        return len(self._tried_hashes)

    @property
    def best_edge_pct(self) -> float:
        return round(self._champion_edge * 100, 3) if self._champion else 0.0

    def _mutate(self, base: Recipe) -> Recipe:
        """Create a small random variation of a recipe."""
        rng = random.Random()
        mutations = {
            "horizon": rng.choice([max(3, base.horizon - 5), base.horizon, base.horizon + 5]),
            "num_leaves": rng.choice([max(7, base.num_leaves - 8), base.num_leaves, base.num_leaves + 8]),
            "learning_rate": round(base.learning_rate * rng.choice([0.5, 1.0, 2.0]), 4),
            "max_depth": rng.choice([max(2, base.max_depth - 1), base.max_depth, base.max_depth + 1]),
            "min_child_samples": rng.choice([max(5, base.min_child_samples - 10), base.min_child_samples, base.min_child_samples + 10]),
            "lambda_l1": round(base.lambda_l1 * rng.choice([0.5, 1.0, 2.0]), 3),
            "lambda_l2": round(base.lambda_l2 * rng.choice([0.5, 1.0, 2.0]), 3),
            "mid_price_lo": rng.choice([15, 20, 25, 30]),
            "mid_price_hi": rng.choice([70, 75, 80, 85]),
        }
        return Recipe(
            name=f"mutant_{base.name}_{rng.randint(0, 999)}",
            horizon=mutations["horizon"],
            num_leaves=mutations["num_leaves"],
            learning_rate=max(0.001, min(0.3, mutations["learning_rate"])),
            max_depth=mutations["max_depth"],
            min_child_samples=mutations["min_child_samples"],
            feature_fraction=base.feature_fraction,
            bagging_fraction=base.bagging_fraction,
            lambda_l1=mutations["lambda_l1"],
            lambda_l2=mutations["lambda_l2"],
            mid_price_lo=mutations["mid_price_lo"],
            mid_price_hi=mutations["mid_price_hi"],
            feature_subset=base.feature_subset,
            lookback=base.lookback,
        )
