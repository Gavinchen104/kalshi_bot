from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    kalshi_api_key: str = Field(default="")
    kalshi_api_secret: str = Field(default="")
    kalshi_base_url: str = Field(default="https://api.elections.kalshi.com")
    kalshi_ws_url: str = Field(default="wss://api.elections.kalshi.com/trade-api/ws/v2")

    bot_env: str = Field(default="dev")
    bot_paper_mode: bool = Field(default=True)
    bot_allow_live_trading: bool = Field(default=False)
    bot_log_level: str = Field(default="INFO")


class AppConfig(BaseModel):
    name: str
    env: str
    paper_mode: bool
    log_level: str


class RiskConfig(BaseModel):
    max_position_per_market: int
    max_order_size: int
    max_daily_loss_cents: int
    max_drawdown_cents: int
    max_data_age_seconds: int
    max_gross_exposure: int = 500
    max_orders_per_minute: int = 30


class StrategyConfig(BaseModel):
    min_edge_bps: int
    min_top_book_depth: int
    max_spread_cents: int
    target_market_regex: str = ".*BTC.*"
    use_ml_model: bool = True
    model_path: str = "data/models/btc_15m_model.json"
    min_model_confidence: float = 0.03
    feature_lookback: int = 20
    cost_floor_bps: int = 100
    n_ensemble_models: int = 3
    max_kelly_contracts: int = 10
    default_order_size: int = 1
    bankroll_cents: int = 50_000


class ExecutionConfig(BaseModel):
    default_order_size: int
    min_order_interval_ms: int
    min_price_cents: int
    max_price_cents: int
    fee_bps: int = 50
    slippage_bps: int = 20
    take_profit_pct: float = 0.25
    stop_loss_pct: float = 0.15


class MarketsConfig(BaseModel):
    allowlist: list[str]
    symbol_filters: list[str] = Field(default_factory=list)


class StorageConfig(BaseModel):
    db_path: str


class TestingConfig(BaseModel):
    mock_data_path: str
    use_mock_data_stream: bool


class AutoTrainConfig(BaseModel):
    enabled: bool = True
    interval_minutes: int = 15
    min_market_states: int = 200
    market_like: str = "BTC"
    horizon: int = 5
    limit: int = 100000
    epochs: int = 200
    lr: float = 0.05
    min_retrains_before_pause: int = 8
    plateau_window: int = 6
    plateau_accuracy_span_threshold: float = 0.01
    plateau_logloss_span_threshold: float = 0.05
    pause_minutes_on_plateau: int = 180


class BinanceConfig(BaseModel):
    enabled: bool = True
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    history_candles: int = 500
    save_interval_seconds: int = 60


class Settings(BaseModel):
    app: AppConfig
    risk: RiskConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    markets: MarketsConfig
    storage: StorageConfig
    testing: TestingConfig
    autotrain: AutoTrainConfig
    binance: BinanceConfig = BinanceConfig()
    env: EnvSettings


def load_settings(path: str = "config/settings.yaml") -> Settings:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    env = EnvSettings()
    return Settings(
        app=AppConfig(**raw.get("app", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        markets=MarketsConfig(**raw.get("markets", {})),
        storage=StorageConfig(**raw.get("storage", {})),
        testing=TestingConfig(**raw.get("testing", {})),
        autotrain=AutoTrainConfig(**raw.get("autotrain", {})),
        binance=BinanceConfig(**raw.get("binance", {})),
        env=env,
    )
