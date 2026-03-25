"""
Adaptive auto-retrainer.

Each cycle:
  1.  Ask the StrategyExplorer for the next Recipe to try.
  2.  Train a model using that Recipe's hyperparameters.
  3.  Record the walk-forward result back into the explorer.
  4.  If the model beats baseline → deploy it and enable trading.
  5.  If not → keep exploring (no trading until edge is found).
  6.  Monitor live PnL; if it declines, force re-exploration.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.config import AutoTrainConfig, StrategyConfig
from src.monitoring.logging import get_logger
from src.storage.repository import Repository
from src.strategy.signals import Strategy
from src.training.strategy_explorer import StrategyExplorer
from src.training.train_ml_strategy import train_from_repository

logger = get_logger("auto_retrainer")


class AutoRetrainer:
    def __init__(
        self,
        config: AutoTrainConfig,
        strategy_config: StrategyConfig,
        repository: Repository,
        strategy: Strategy,
    ):
        self.config = config
        self.strategy_config = strategy_config
        self.repository = repository
        self.strategy = strategy
        self._explorer = StrategyExplorer(repository.db_path)

        self._next_run_at = datetime.now(tz=timezone.utc)
        self._running = False
        self.model_beats_baseline: bool = False

        # PnL decay tracking — only set once per champion, never reset on re-validation
        self._last_pnl_check_at = datetime.now(tz=timezone.utc)
        self._pnl_at_deploy: float | None = None
        self._pnl_high_water: float | None = None
        self._deployed_at: datetime | None = None
        self._deployed_champion: str | None = None

    async def maybe_retrain(self) -> None:
        if not self.config.enabled or self._running:
            return
        now = datetime.now(tz=timezone.utc)
        if now < self._next_run_at:
            return

        self._check_pnl_decay()

        interval = self.config.interval_minutes
        if not self._explorer.has_champion():
            interval = max(2, interval // 2)
        self._next_run_at = now + timedelta(minutes=interval)

        rows = self.repository.market_state_count(self.config.market_like)
        if rows < self.config.min_market_states:
            return

        self._running = True
        try:
            recipe = self._explorer.pick_recipe()
            logger.info(
                "training_recipe",
                recipe=recipe.name,
                horizon=recipe.horizon,
                num_leaves=recipe.num_leaves,
                lr=recipe.learning_rate,
                mid_range=f"{recipe.mid_price_lo}-{recipe.mid_price_hi}",
                feature_subset=recipe.feature_subset,
                explored=self._explorer.total_explored,
                champion=self._explorer.champion_name,
            )

            metrics = await asyncio.to_thread(self._train_with_recipe, recipe)
            self._explorer.record_result(recipe, metrics)

            beats = metrics.get("beats_baseline", 0.0) == 1.0
            self.model_beats_baseline = beats

            if beats:
                loaded = self.strategy.reload_model()
                self.strategy.set_model_quality(True)

                # Only reset PnL tracking when champion CHANGES
                new_champion = self._explorer.champion_name
                if new_champion != self._deployed_champion:
                    current_pnl = self._current_pnl_total() or 0
                    self._pnl_at_deploy = current_pnl
                    self._pnl_high_water = current_pnl
                    self._deployed_at = now
                    self._deployed_champion = new_champion
                    logger.info(
                        "new_champion_deployed",
                        recipe=recipe.name,
                        pnl_at_deploy=current_pnl,
                    )
                else:
                    # Same champion re-validated — update high water mark only upward
                    current_pnl = self._current_pnl_total()
                    if current_pnl is not None and self._pnl_high_water is not None:
                        self._pnl_high_water = max(self._pnl_high_water, current_pnl)

                logger.info(
                    "model_deployed",
                    recipe=recipe.name,
                    edge_pct=self._explorer.best_edge_pct,
                    wf_acc=metrics.get("wf_mean_accuracy"),
                    wf_base=metrics.get("wf_mean_baseline"),
                )
            else:
                self.strategy.set_model_quality(False)
                logger.info(
                    "model_no_edge",
                    recipe=recipe.name,
                    wf_acc=round(metrics.get("wf_mean_accuracy", 0), 4),
                    wf_base=round(metrics.get("wf_mean_baseline", 0), 4),
                    explored=self._explorer.total_explored,
                    champion=self._explorer.champion_name,
                )

            self.repository.log_event("model_retrained", payload={
                "market_state_rows": rows,
                "recipe": recipe.name,
                "explored_total": self._explorer.total_explored,
                "champion": self._explorer.champion_name,
                "best_edge_pct": self._explorer.best_edge_pct,
                **metrics,
            })

        except Exception as exc:
            self.repository.log_event("model_retrain_failed", payload={
                "error": str(exc), "rows": rows,
            })
            logger.warning("retrain_failed", error=str(exc))
        finally:
            self._running = False

    def _train_with_recipe(self, recipe) -> dict[str, float]:
        return train_from_repository(
            db_path=self.repository.db_path,
            market_like=self.config.market_like,
            target_market_regex=self.strategy_config.target_market_regex,
            model_output_path=self.strategy_config.model_path,
            lookback=recipe.lookback,
            horizon=recipe.horizon,
            limit=self.config.limit,
            recipe=recipe,
        )

    # ── PnL decay detection ──────────────────────────────────────────

    def _check_pnl_decay(self) -> None:
        if not self.model_beats_baseline:
            return
        if self._deployed_at is None or self._pnl_at_deploy is None:
            return

        now = datetime.now(tz=timezone.utc)
        if (now - self._last_pnl_check_at).total_seconds() < 60:
            return
        self._last_pnl_check_at = now

        current = self._current_pnl_total()
        if current is None:
            return

        if self._pnl_high_water is not None:
            self._pnl_high_water = max(self._pnl_high_water, current)

        # Trigger 1: PnL dropped 300c ($3) from deploy point
        decline_from_deploy = current - self._pnl_at_deploy
        # Trigger 2: PnL dropped 300c ($3) from high water mark (drawdown)
        hwm = self._pnl_high_water or self._pnl_at_deploy
        drawdown = current - hwm

        should_trigger = decline_from_deploy < -300 or drawdown < -300

        if should_trigger:
            logger.warning(
                "pnl_decay_triggered",
                decline_from_deploy=decline_from_deploy,
                drawdown=drawdown,
                pnl_at_deploy=self._pnl_at_deploy,
                high_water=hwm,
                current_pnl=current,
                champion=self._deployed_champion,
            )
            self._explorer.signal_pnl_decay()
            self.strategy.set_model_quality(False)
            self.model_beats_baseline = False
            self._pnl_at_deploy = None
            self._pnl_high_water = None
            self._deployed_at = None
            self._deployed_champion = None

    def _current_pnl_total(self) -> float | None:
        try:
            series = self.repository.pnl_series(limit=1)
            if series:
                return float(series[-1].get("total_cents", 0))
        except Exception:
            pass
        return None
