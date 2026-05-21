# Phase 3 Operations Runbook

This runbook is the Phase 3 X2 operating checklist. It is for paper trading
through GATE B and the later GATE C tiny-live decision. Phase 4 scaling remains
out of scope until GATE C has held for two green realized-PnL weeks.

## Start And Restart

1. Confirm `.env` has `BOT_PAPER_MODE=true` and `BOT_ALLOW_LIVE_TRADING=false`.
2. Start the supervisor:

   ```bash
   ./scripts/run_bot.sh
   ```

3. Watch `logs/bot-*.log` for `bot_start`, `heartbeat`,
   `calibration_snapshot`, `paper_fill`, and `risk_block` events.
4. To restart cleanly, stop the supervisor with `Ctrl-C`, wait for the child
   process to exit, then run `./scripts/run_bot.sh` again.

## Alerts

Set `BOT_ALERT_WEBHOOK_URL` to a JSON webhook for out-of-band alerts. If it is
blank, alerts are still emitted as structured `alert` log events.

Alert keys:

- `stalled_kalshi_feed`: no Kalshi market-state update for longer than
  `monitoring.stalled_feed_seconds`. Restart if it persists after the next
  supervisor heartbeat.
- `coinbase_data_quality`: 48-hour candle coverage fell below
  `monitoring.data_quality_min_coverage`. Check Coinbase WS health and let gap
  repair backfill before trusting GATE B data.
- `calibration_drift`: rolling Brier exceeded
  `monitoring.calibration_drift_brier_threshold`. Do not proceed toward GATE C
  until the forward window is diagnosed.
- `risk_daily_loss_limit`, `risk_drawdown_limit`,
  `risk_max_tail_short_exposure`, or `risk_kill_switch:*`: trading was blocked
  by A4 controls. Treat any kill-switch alert as a stop condition.

## Kill Switch Procedure

1. Leave `BOT_ALLOW_LIVE_TRADING=false` unless GATE C has a written human
   sign-off.
2. If a kill-switch alert appears, stop the supervisor and preserve the latest
   log file.
3. Inspect the matching `risk_block` event and the latest PnL/calibration
   snapshots in the dashboard.
4. Fix the cause, run unit tests, and restart in paper mode only.

## GATE B Checklist

GATE B requires all of the following over at least five continuous trading days:

- No backfilled gaps inside the validation window.
- Live `calibration_snapshot` metrics are consistent with the B1 backtest within
  sampling noise.
- Edge-decile realized paper PnL is monotonic increasing.
- Net paper PnL is greater than estimated fees.
- No unresolved stalled-feed, data-quality, risk, or calibration-drift alerts.

Failing any item means diagnose paper/live divergence before live trading.

## GATE C Go/No-Go

GATE C may be considered only after GATE B passes.

- Set `sizing.max_contracts_per_trade: 1`.
- Confirm A4 controls are enabled: per-market cap, gross cap, tail-short cap,
  vol-regime kill switch, daily loss limit, and max drawdown halt.
- Keep `BOT_ALLOW_LIVE_TRADING=false` until a human writes an explicit go
  decision with date, reviewer, bankroll, and rollback trigger.
- If approved, run tiny live only. Two consecutive green realized-PnL weeks are
  required before Phase 4 can be discussed.

## A6 Scaling Ladder Definition

This ladder is defined now but must not be executed until GATE C has held.

- Step 0: paper only, current Phase 3 default.
- Step 1: tiny live, `max_contracts_per_trade: 1`, only after GATE C sign-off.
- Step 2: `max_contracts_per_trade: 3`, only after two green tiny-live weeks and
  no calibration-drift or risk kill-switch alerts.
- Step 3: `max_contracts_per_trade: 10`, only after another sustained-green
  window at Step 2.
- Roll back one step immediately on drawdown breach, calibration drift, or any
  unresolved data-quality alert.

## Backup And Restore

The local SQLite DB is `data/bot.db`.

Backup:

```bash
sqlite3 data/bot.db ".backup 'data/bot-backup-$(date +%Y%m%d-%H%M%S).db'"
```

Restore:

```bash
cp data/bot.db data/bot-before-restore.db
cp data/bot-backup-YYYYMMDD-HHMMSS.db data/bot.db
```
