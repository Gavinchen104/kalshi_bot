# Building an Automated Trading Bot for Kalshi

## 1) Scope and Safety First

This guide explains how to design and build an automated trading bot for Kalshi event contracts, from MVP to production.

Before writing code:

- Read Kalshi's API docs, terms, and exchange rules.
- Confirm whether your account type and region permit automated trading.
- Start in sandbox/paper mode first, then small notional in production.
- Treat this as high-risk software: enforce strict risk limits and kill switches.

This is not investment advice.

---

## 2) What You Are Building

A reliable bot usually has these components:

1. **Market data ingestion**: order book, trades, tickers, market metadata.
2. **Signal engine**: computes edge/fair value and decides if there is a trade.
3. **Execution engine**: places/cancels orders, handles partial fills/retries.
4. **Risk engine**: enforces position limits, drawdown limits, and rate controls.
5. **State + persistence**: stores fills, orders, positions, PnL, model features.
6. **Monitoring + alerts**: logs, metrics, health checks, and on-call notifications.
7. **Ops layer**: config, secrets, deployment, and failover controls.

---

## 3) Recommended Tech Stack (Python MVP)

You can build this in many languages; Python is fastest for iteration.

- **Language**: Python 3.11+
- **API/http**: `httpx`
- **Websocket**: `websockets` (or any asyncio client)
- **Validation**: `pydantic`
- **Dataframes**: `pandas` (optional)
- **DB**: `sqlite` for MVP, Postgres for production
- **Queue/scheduler**: `asyncio` tasks (MVP), Redis/Celery (later)
- **Observability**: `structlog` + Prometheus/OpenTelemetry + Grafana
- **Packaging**: `uv` or `poetry`
- **Deploy**: Docker + systemd/VM or Kubernetes

---

## 4) Project Structure

Use a structure like this:

```text
kalshi-bot/
  README.md
  pyproject.toml
  .env.example
  config/
    settings.yaml
  src/
    main.py
    api/
      kalshi_client.py
      ws_client.py
      auth.py
    strategy/
      signals.py
      fair_value.py
      filters.py
    execution/
      order_manager.py
      router.py
      slippage.py
    risk/
      limits.py
      kill_switch.py
      exposure.py
    portfolio/
      positions.py
      pnl.py
    storage/
      models.py
      repository.py
    monitoring/
      logging.py
      metrics.py
      alerts.py
    backtest/
      simulator.py
      replay.py
  tests/
    unit/
    integration/
    replay/
```

---

## 5) Kalshi API Integration Plan

### 5.1 Authentication and Connectivity

Build a typed API client with:

- Session/auth setup (per official docs).
- Retries with exponential backoff for transient failures.
- Idempotency keys for order placement where supported.
- Clock sync checks (important for signed requests/timestamps).

### 5.2 REST Endpoints You Usually Need

- List markets and metadata (status, settlement rules, tick size, expiry).
- Get order book snapshots.
- Place/cancel/modify orders.
- Fetch open orders, fills, and account balances.

### 5.3 Websocket Streams

Subscribe to:

- Top-of-book/order book deltas.
- Trade prints.
- Market status changes (halt, close, settle).

Design websocket handlers to:

- Reconnect automatically.
- Detect stale feed with heartbeat timeout.
- Rebuild order book from snapshot + incremental updates.

---

## 6) Trading Strategy Design (MVP to Advanced)

Start simple, measurable, and risk-limited.

### 6.1 MVP Strategies

1. **Spread capture / passive quoting**
   - Post bids/offers around estimated fair value.
   - Cancel/replace as fair value moves.

2. **Mean reversion on mispricing**
   - Define fair value from short-window microprice/flow.
   - Trade only when observed price deviates beyond threshold.

3. **Event-probability model**
   - Map external features/news/weather/economic releases to probability.
   - Trade when model probability differs from market-implied probability by edge > fees + slippage buffer.

### 6.2 Decision Rule Template

Use a strict rule:

`expected_edge = model_prob - market_prob - costs`

Trade only if:

- `abs(expected_edge) >= min_edge_threshold`
- Position after trade stays under all risk limits
- Market is liquid enough (spread, depth, recent fills)

### 6.3 Market Filters (Important)

Avoid low-quality markets:

- Too wide spread.
- Too little depth at best levels.
- Too close to settlement unless strategy explicitly handles it.
- Recent feed instability.

---

## 7) Risk Management (Non-Negotiable)

Implement these hard limits in code (cannot be bypassed by strategy):

1. **Per-market max position**
2. **Global gross and net exposure caps**
3. **Per-order max size**
4. **Daily loss limit**
5. **Max drawdown from session peak PnL**
6. **Max order rate / cancel rate**
7. **Stale data lockout** (no fresh data -> no new orders)
8. **Kill switch** (manual and automatic)

If any limit is breached:

- Cancel all open orders.
- Block new orders.
- Alert immediately.

---

## 8) Execution Engine Details

### 8.1 Order Lifecycle State Machine

Track every order through explicit states:

- `created -> sent -> acknowledged -> partially_filled -> filled`
- `created/sent -> rejected`
- `acknowledged/partial -> cancel_pending -> canceled`

Persist state transitions with timestamps for audits and debugging.

### 8.2 Execution Logic Checklist

- Price validation against tick size and allowed range.
- Size rounding and minimum size checks.
- Timeout + retry policy for uncertain responses.
- Reconciliation loop:
  - Poll exchange open orders/fills periodically.
  - Compare with local state.
  - Repair discrepancies.

### 8.3 Slippage and Queue Position Controls

- Prefer passive orders for maker behavior when edge allows.
- Use IOC/marketable logic only when urgency justifies cost.
- Track realized slippage and fill ratio per market/time-of-day.

---

## 9) Data, Storage, and Analytics

Store at least:

- Market snapshots and updates (or compacted form).
- Signals/features used at decision time.
- Order intents and actual exchange responses.
- Fill events and fees.
- Position and PnL timeline.

This lets you answer:

- "Why did we place this order?"
- "What did we know at that moment?"
- "Did realized performance match expected edge?"

Schema tip:

- Keep immutable append-only event tables for core trading events.
- Build materialized views for fast dashboard queries.

---

## 10) Backtesting and Simulation

Backtesting is mandatory before production.

### 10.1 Levels of Testing

1. **Unit tests**: feature calculations, risk checks, rounding rules.
2. **Replay tests**: run strategy on historical tick/order book streams.
3. **Execution simulation**: model partial fills, queue priority, latency.
4. **Paper trading**: live data, simulated orders.
5. **Canary trading**: tiny real size with strict limits.

### 10.2 Metrics to Track

- Win rate, average edge captured.
- Sharpe-like return/risk measures.
- Max drawdown.
- Fill ratio, adverse selection.
- Latency (signal->order, order->ack, ack->fill).
- PnL by market category and time bucket.

---

## 11) Monitoring and Operations

### 11.1 Logging

Use structured logs with keys:

- `market_id`, `order_id`, `signal_id`, `position`, `edge`, `latency_ms`, `decision_reason`.

### 11.2 Metrics + Alerts

Track and alert on:

- Process heartbeat.
- Websocket disconnect/staleness.
- Error rate spikes.
- Rejected order rate.
- Risk-limit breach attempts.
- Drawdown or daily loss threshold events.

### 11.3 Runbooks

Write simple docs for:

- "Bot not placing orders"
- "Orders rejected"
- "Data feed stale"
- "Kill switch triggered"

---

## 12) Security and Secrets

- Never hardcode API secrets.
- Use environment variables or secret manager.
- Restrict key permissions if possible.
- Rotate keys regularly.
- Redact secrets from logs and error traces.

`.env.example` should contain only placeholder names, never real values.

---

## 13) Deployment Path

### Phase 1: Local MVP

- One strategy.
- One/few liquid markets.
- SQLite.
- Console logs + basic alerts.

### Phase 2: Stable Paper Trader

- Full risk framework.
- Persistent DB (Postgres).
- Basic dashboard + alerts.
- Replay harness integrated in CI.

### Phase 3: Production

- Canary notional first.
- Automated restarts and health checks.
- Strict release process:
  - run tests
  - replay test pack
  - config diff review
  - staged rollout

---

## 14) Minimal Build Checklist

Use this checklist in order:

1. Set up project skeleton and config loader.
2. Implement Kalshi REST client with auth.
3. Implement websocket market stream with reconnect/heartbeat.
4. Build market state cache (book + last trade + status).
5. Implement first strategy signal function.
6. Implement risk checks as independent gate.
7. Implement order manager and lifecycle tracking.
8. Add persistence for orders/fills/positions.
9. Add paper-trading mode.
10. Add monitoring, alerts, and kill switch.
11. Run historical replays and tune thresholds.
12. Deploy tiny-size canary and monitor closely.

---

## 15) Example Bot Loop (Pseudocode)

```python
async def main():
    cfg = load_config()
    api = KalshiClient(cfg)
    ws = KalshiWSClient(cfg)
    risk = RiskEngine(cfg)
    strategy = Strategy(cfg)
    om = OrderManager(api, risk, cfg)

    async for event in ws.stream():
        market_state.update(event)
        signal = strategy.compute_signal(market_state)
        if not signal:
            continue

        proposed_order = strategy.to_order(signal, market_state)
        allowed, reason = risk.validate(proposed_order, portfolio_state, market_state)
        if not allowed:
            log.info("risk_block", reason=reason)
            continue

        await om.execute(proposed_order)
        await reconcile_positions_and_pnl(api, portfolio_state)
```

---

## 16) Common Failure Modes and Defenses

1. **Double-ordering from retries**
   - Use idempotency keys and client order IDs.

2. **Trading on stale data**
   - Enforce max data age guard.

3. **Unbounded exposure in fast markets**
   - Hard cap plus pre-trade position projection.

4. **Silent bot crash**
   - Supervisor restarts + heartbeat alerts.

5. **Model drift**
   - Continuous performance monitoring and scheduled retraining/retuning.

---

## 17) Suggested Next Step

Build an MVP with:

- One market universe,
- One simple edge model,
- Full risk gates,
- Paper trading only.

Then iterate with replay-driven improvements before increasing capital.

