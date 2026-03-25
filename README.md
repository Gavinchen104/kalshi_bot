# Kalshi Bot Starter

This repository is a starter scaffold for building an automated Kalshi trading bot.

## Quickstart

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

```bash
pip install -e ".[dev]"
```

3. Copy environment file and edit values:

```bash
cp .env.example .env
```

By default, the bot is paper-only. Real orders are locked unless you explicitly set:

```bash
BOT_ALLOW_LIVE_TRADING=true
```

4. Run the bot:

```bash
python -m src.main
```

5. Run dashboard (in another terminal):

```bash
streamlit run dashboard/app.py
```

By default, dashboard reads `data/bot.db`.

## Testing Modes

- **Live data + paper orders (default)**:
  - `testing.use_mock_data_stream: false`
  - `app.paper_mode: true`
- **Offline mock stream + paper orders**:
  - set `testing.use_mock_data_stream: true`
  - default sample file: `data/mock_market_stream.jsonl`

The bot now has extra safety/testing behavior:

- Refuses to run if no market filters are configured
- Order rate-limiting via `execution.min_order_interval_ms`
- Price bounds enforcement (`1..99` cents)

## Bitcoin-only setup

To keep trading scope strictly on Bitcoin markets:

- Set `markets.symbol_filters` to `["BTC"]`
- Optionally set exact tickers in `markets.allowlist` for tighter control

Current default config already includes a BTC symbol filter.

## BTC 15m ML Strategy

The strategy now supports an ML-driven fair value model for BTC 15-minute contracts.

- Config keys under `strategy`:
  - `target_market_regex` (default: `.*BTC.*`)
  - `use_ml_model`
  - `model_path`
  - `min_model_confidence`
  - `feature_lookback`

### Train model from collected data

First run the bot in paper mode to collect `market_state` rows in `data/bot.db`, then train:

```bash
python -m src.training.train_ml_strategy \
  --db-path data/bot.db \
  --market-like BTC \
  --target-market-regex ".*BTC.*15.*" \
  --model-output-path data/models/btc_15m_model.json
```

After training, restart the bot so it loads the saved model.

Note: this is an experimental research model and does not guarantee profitability.

### Automatic retraining

The bot can retrain itself while running using `autotrain` settings in `config/settings.yaml`.

- `enabled`: turn auto retraining on/off
- `interval_minutes`: retrain cadence
- `min_market_states`: minimum BTC samples before retraining
- `market_like`: symbol filter (for BTC keep `BTC`)
- `horizon`, `limit`, `epochs`, `lr`: trainer parameters

When enabled, the bot logs:

- `model_retrain_skipped` (not enough data yet)
- `model_retrained` (metrics + loaded flag)
- `model_retrain_failed` (error details)

## Structure

- `src/api`: API and websocket clients
- `src/strategy`: signal logic
- `src/execution`: order lifecycle and routing
- `src/risk`: risk limits and kill switch
- `src/portfolio`: position and pnl state
- `src/storage`: persistence abstractions
- `src/monitoring`: logging and metrics stubs
- `tests`: unit/integration/replay test folders

