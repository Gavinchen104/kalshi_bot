from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    kalshi_api_key: str = ""
    kalshi_api_secret: str = ""
    bot_log_level: str = "INFO"
    bot_paper_mode: bool = True
    bot_allow_live_trading: bool = False
    bot_alert_webhook_url: str = ""


class AppConfig(BaseModel):
    env: str = "dev"
    log_level: str = "INFO"


class KalshiConfig(BaseModel):
    base_url: str
    ws_url: str
    ticker_regex: str = "^KXBTC.*"


class CoinbaseConfig(BaseModel):
    ws_url: str
    rest_candles_url: str
    product_id: str = "BTC-USD"
    history_candles: int = 300
    # On startup, backfill missing candle minutes going back at most this far.
    backfill_max_hours: int = 48
    # How often the runtime re-checks for and repairs candle gaps.
    gap_repair_interval_minutes: int = 15
    # Periodic repair only scans this recent a window (lighter than startup).
    gap_repair_lookback_hours: int = 3


class PricerConfig(BaseModel):
    vol_window_minutes: int = 60
    vol_floor_annualized: float = 0.20
    vol_ceiling_annualized: float = 3.00
    min_horizon_seconds: int = 5
    # Default native bracket width for KXBTC-*-B<low> 15m markets. Series-specific
    # widths (e.g. $500 instead of $250) should be inferred in Phase 2 from
    # observed adjacent strikes in the same series.
    bracket_width_usd_default: float = 250.0
    # Vol estimator selection. Default "fixed" preserves Phase 1 behavior
    # (uses vol_window_minutes). Other modes ("horizon_scaled", "blend",
    # "ewma") are introduced through Phase 2 Commit B; "fixed" remains the
    # default so each mode change is A/B-comparable on the same data.
    vol_mode: str = "fixed"
    # Horizon-scaled mode clamps the lookback window to [floor, cap] minutes.
    vol_window_floor_min: int = 60
    vol_window_cap_min: int = 1440
    # Long-window vol floor (Phase 2 W1.4): if > 0, the final sigma is
    # max(estimator_sigma, trailing N-day realized vol). 0 disables (default,
    # preserves Phase 1 behavior). 7 = one week of 1-min closes.
    vol_long_floor_days: int = 0
    # Optional EWMA half-life override in minutes. None → horizon-tuned default.
    ewma_half_life_min: int | None = None


class StrategyConfig(BaseModel):
    edge_threshold: float = 0.04
    min_horizon_seconds: int = 30
    max_horizon_seconds: int = 900
    max_spread_cents: int = 10
    min_top_book_depth: int = 5
    # Phase 3 / B1: persisted isotonic calibration artifact. When present, the
    # live strategy maps raw BS probabilities through it before edge decisions.
    calibration_model_path: str | None = None
    # Phase 3 / A1: suppress signals when |spot − strike| (or |spot − bracket
    # midpoint|) is below this band. Bounds Kalshi-vs-Coinbase resolution-
    # source basis risk at the moneyness boundary. 0 = guard disabled (default
    # until A1 measurement informs a value).
    near_strike_guard_usd: float = 0.0


class SizingConfig(BaseModel):
    bankroll_cents: int = 50_000
    kelly_fraction: float = 0.25
    max_contracts_per_trade: int = 5


class RiskConfig(BaseModel):
    max_position_per_market: int = 25
    max_gross_exposure: int = 50
    max_tail_short_exposure: int = 10
    tail_low_prob: float = 0.10
    tail_high_prob: float = 0.90
    vol_regime_zscore: float = 3.0
    vol_regime_min_samples: int = 30
    max_daily_loss_cents: int = 5_000
    max_drawdown_cents: int = 3_000
    max_data_age_seconds: int = 10
    max_orders_per_minute: int = 60


class ExecutionConfig(BaseModel):
    fee_bps: int = 50
    slippage_bps: int = 20
    min_order_interval_ms: int = 500
    top_book_fill_fraction: float = 1.0


class StorageConfig(BaseModel):
    db_path: str = "data/bot.db"


class MeasurementConfig(BaseModel):
    calibration_bins: int = 10
    calibration_window: int = 500


class MonitoringConfig(BaseModel):
    alert_cooldown_seconds: int = 300
    stalled_feed_seconds: int = 60
    data_quality_window_hours: int = 48
    data_quality_min_coverage: float = 0.95
    calibration_drift_brier_threshold: float = 0.20
    calibration_drift_min_samples: int = 100


class Settings(BaseModel):
    app: AppConfig
    kalshi: KalshiConfig
    coinbase: CoinbaseConfig
    pricer: PricerConfig
    strategy: StrategyConfig
    sizing: SizingConfig
    risk: RiskConfig
    execution: ExecutionConfig
    storage: StorageConfig
    measurement: MeasurementConfig
    monitoring: MonitoringConfig
    env: EnvSettings = Field(default_factory=EnvSettings)


def load_settings(path: str = "config/settings.yaml") -> Settings:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return Settings(
        app=AppConfig(**raw.get("app", {})),
        kalshi=KalshiConfig(**raw["kalshi"]),
        coinbase=CoinbaseConfig(**raw["coinbase"]),
        pricer=PricerConfig(**raw.get("pricer", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        sizing=SizingConfig(**raw.get("sizing", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        storage=StorageConfig(**raw.get("storage", {})),
        measurement=MeasurementConfig(**raw.get("measurement", {})),
        monitoring=MonitoringConfig(**raw.get("monitoring", {})),
        env=EnvSettings(),
    )
