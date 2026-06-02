# GATE C Go/No-Go Checklist

_Phase 3 / A5 — Tiny-size live trading sign-off._

**Do not flip `BOT_ALLOW_LIVE_TRADING=true` until every item below is ✅ AND a human has dated and signed the bottom of this document.** No exceptions.
This is the only sign-off that admits real money to the bot, and the only
human override the plan permits.

## How to use this checklist

1. Make a working copy: `cp docs/GATE_C_CHECKLIST.md docs/gate_c_signoffs/<YYYY-MM-DD>-tinylive.md`
2. Work through every section in order. Each line is a hard requirement.
3. For each `[ ]`, either:
   - Run the verification command, paste the output (or the pass/fail) under the line, and tick `[x]`, OR
   - Write `[N/A — <reason>]` if a line truly does not apply (rare).
4. If any line cannot be ticked, **stop**. Do not partial-deploy. Fix the cause, then redo the failed section.
5. The "Final live-flag enablement" section is irreversible-ish: only flip the env var after the rest of the checklist is fully ✅ and the sign-off block is filled.

---

## 1. Prerequisites — gates already cleared

- [ ] **GATE A** recorded in `PHASE2_PLAN.md §2.1`.
  - Expected: FAIL (Phase 2 outcome) → Track B pivot chosen.
- [ ] **B1 mini-gate** recorded in `PHASE3_PLAN.md §2.1` with current artifact.
  - `python -m src.backtest.engine --mini-gate-b1` → exit code **0**, paste tail of output.
- [ ] **A1 basis study** recorded, guard band set in `config/settings.yaml` (`strategy.near_strike_guard_usd`).
  - `python -c "from src.config import load_settings; print(load_settings().strategy.near_strike_guard_usd)"` → non-zero value matches the doc.
- [ ] **GATE B** PASS recorded in `PHASE3_PLAN.md` with date + the `--gate-b` exit-0 transcript.
  - Re-run on the day of sign-off: `python -m src.backtest.engine --gate-b` → exit **0**.
  - All five sub-criteria PASS (days, log loss, tail emp, monotonicity, net>fees, coverage).

## 2. Pricer & calibrator state

- [ ] **Production calibrator artifact exists** at `settings.strategy.calibration_model_path`.
  - `ls -la data/models/b1_isotonic.json` shows non-empty file.
- [ ] **Artifact metadata is current** (not stale).
  - `python -c "import json; m=json.load(open('data/models/b1_isotonic.json'))['metadata']; print(m['fit_at'], m.get('n_settled_pairs'))"`
  - `fit_at` within the last 14 days OR explicit rationale for using older artifact.
- [ ] **Live strategy loads the artifact on init** (no silent fallback to raw probs).
  - `python -c "from src.config import load_settings; from src.strategy.edge import EdgeStrategy; s=load_settings(); es=EdgeStrategy(s.strategy); assert es.calibrator is not None, 'CALIBRATOR DID NOT LOAD'; print('calibrator OK')"`
- [ ] **Vol mode locked** to the one used during GATE B (do NOT switch vol modes between GATE B and live).
  - `settings.pricer.vol_mode` value: ___________
- [ ] **No uncommitted changes to pricer/edge/calibrator code** since GATE B run.
  - `git status --short src/pricing/ src/strategy/edge.py src/strategy/calibrator.py` → empty.

## 3. Risk controls — verified ARMED (not just configured)

For each, both check the config AND verify the corresponding test passes.

- [ ] `risk.max_position_per_market` set to **the agreed live-step value** (Step 1 = matches `sizing.max_contracts_per_trade`).
- [ ] `risk.max_gross_exposure` set, ≥ per-market cap and ≤ daily-loss limit / contract worst case.
- [ ] `risk.max_daily_loss_cents` set; verify the kill-switch trips in test:
  - `python -m pytest tests/unit/ -q -k risk` → all pass.
- [ ] `risk.max_drawdown_cents` set; verify halt logic.
- [ ] **Tail-short exposure cap** active (A4). Net short on `our_prob < 0.10` or > 0.90 is bounded.
- [ ] **Vol-regime kill switch** active (A4). Confirm trip condition documented.
- [ ] **Manual kill switch** path documented in `docs/RUNBOOK.md` ("Kill Switch Procedure") and tested in the last 30 days.
  - Date last tested: ___________

## 4. Configuration sanity (live vs. GATE-B parity)

Drift between what GATE B measured and what runs live is the most common cause of live-vs-paper divergence.

- [ ] `config/settings.yaml` byte-identical to GATE B run (or only knobs in §3 changed).
  - `git log --oneline -5 -- config/settings.yaml` — last edit reviewed.
- [ ] `kalshi.base_url` and `kalshi.ws_url` point to the **intended** environment (production, not sandbox). Double-check.
  - production base_url: `https://api.elections.kalshi.com`
- [ ] `app.paper_mode: true` is **disabled** (live mode) — verify with `python -c "from src.config import load_settings; print(load_settings().app)"`.
  - And conversely, **`BOT_ALLOW_LIVE_TRADING=true`** in `.env` (set in §8, not now).
- [ ] `strategy.edge_threshold`, `strategy.max_spread_cents`, `strategy.min_top_book_depth`, `strategy.near_strike_guard_usd` all match GATE B values.
- [ ] `sizing.max_contracts_per_trade: 1` (Step 1 hard cap).
- [ ] `sizing.bankroll_cents` set to the **tiny-live agreed amount** (see §7).
- [ ] No dev-mode flags (`testing.use_mock_data_stream`, etc.) accidentally on.

## 5. Observability & alerting — verified DELIVERING

Logs aren't alerts. Each item below must reach a human out-of-band.

- [ ] **Heartbeat alarm** wired: missing `heartbeat` event for > N minutes → pages someone. Pager: ___________
- [ ] **Data-quality alarm** wired: 48h candle coverage < 95% → alerts.
- [ ] **Calibration-drift alarm** wired (X1): rolling Brier crosses `monitoring.calibration_drift_brier_threshold` → alerts.
  - `python -c "from src.config import load_settings; m=load_settings().monitoring; print(m.calibration_drift_brier_threshold, m.calibration_drift_min_samples)"`
- [ ] **PnL/drawdown alarm** wired: hitting `risk.max_daily_loss_cents` or `risk.max_drawdown_cents` → alerts + flatten.
- [ ] **Dashboard reachable** from outside the bot host. URL/method to access: ___________
- [ ] **At least one test-fire** of each alarm in the last 7 days. Logs of test-fires preserved.

## 6. Operations readiness

- [ ] Bot runs under `./scripts/run_bot.sh` (or systemd/equivalent), not bare `python -m`.
  - Auto-restart confirmed: kill the process once, observe restart within the supervisor's backoff window.
- [ ] **Latest log file** has no unresolved `ERROR`/`warning` lines in the last 24h.
  - `tail -200 logs/$(ls -t logs | head -1)` skimmed for red flags.
- [ ] DB backup taken in the last 24h, restore drill done in the last 30 days.
  - Per `docs/RUNBOOK.md` "Backup And Restore".
- [ ] Clock sync verified (settlement is timestamp-sensitive).
  - `chronyc tracking` or equivalent within ±100ms of NTP.
- [ ] **Ops runbook** (`docs/RUNBOOK.md`) reviewed end-to-end on the day of sign-off.

## 7. Bankroll & sizing for tiny-live

- [ ] Real funded Kalshi account balance verified.
  - `python -c "import asyncio, src.data.kalshi_rest as k; ..."` (or via Kalshi UI). Balance: $___________
- [ ] `sizing.bankroll_cents` ≤ the *tiny-live agreed amount* (per Phase 3 A5 / A6 Step 1):
  - Step 1 sized so one full kill-switch trip is recoverable within the operator's stated risk budget.
  - Agreed tiny-live bankroll for this sign-off: $___________
- [ ] **Rollback trigger** defined explicitly: what realized-PnL / drawdown / alert pattern flips this back to paper.
  - Trigger criterion (one sentence): _______________________________
- [ ] Step 2 (next ladder rung) is **NOT** preconfigured. Sizing changes require a fresh GATE D sign-off after 2 green weeks at Step 1.

## 8. Final live-flag enablement (do last)

Only proceed if every item above is ✅ and the sign-off block below is filled.

- [ ] `.env` has `BOT_ALLOW_LIVE_TRADING=true` (set NOW, not earlier).
- [ ] Bot restarted (supervisor picks up the new env on next restart cycle).
- [ ] First live signal observed in logs: timestamp ___________, market_id ___________.
- [ ] First live fill observed (if any in the first hour): paste fill_price + qty.
- [ ] Dashboard's CALIBRATION panel transitions away from PENDING within one resolution cycle.
- [ ] Subscribe to alerts on the operator's phone (verify a test alert arrives).

---

## Sign-off

| Field | Value |
|---|---|
| Date (UTC) | ___________ |
| Operator | ___________ |
| Reviewer (a second human) | ___________ |
| GATE B exit-0 transcript path | `logs/gate-b-<YYYYMMDD>.log` |
| Artifact path + `fit_at` | `data/models/b1_isotonic.json` · ___________ |
| Bankroll (USD) | $___________ |
| Per-market cap (contracts) | 1 |
| Rollback trigger | _______________________________ |
| Two-week review date | ___________ (today + 14d) |

> **Signed under the operating principle that gates are binding.** A failed
> rollback trigger flips back to paper immediately, no discretion. No size
> increase until two consecutive green realized-PnL weeks AND a new GATE D
> sign-off per `docs/RUNBOOK.md §A6 Scaling Ladder`.

**Operator signature:** ___________

**Reviewer signature:** ___________

---

## Post-sign-off

- Move this signed copy into `docs/gate_c_signoffs/` (never delete).
- Set a calendar reminder for the two-week review.
- Add a one-line entry to `PHASE3_PLAN.md` referencing this signed file.
