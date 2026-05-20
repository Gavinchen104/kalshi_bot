# Phase 2 Plan — Strategy 2 Remediation & Validation

_Status: ✅ Complete · Outcome: **GATE A FAIL** (2026-05-20) · Next: Phase 3 Track B (pivot)_

---

## 1. Context — What Phase 1 Established

Phase 1 built and ran Strategy 2 (Black–Scholes binary pricer using Coinbase
realized volatility, traded against Kalshi BTC market prices), plus a dashboard
and an offline backtest harness (`src/backtest/engine.py`).

**The decisive Phase 1 result** (backtest over 43,635 settled predictions):

| Metric | Value | Interpretation |
|---|---|---|
| Brier score | 0.0788 | Superficially good — but misleading (97% of predictions are trivial 0/1 extremes) |
| Log loss | 1.2801 | **Worse than the 0.693 coin-flip baseline** |
| `0.00–0.10` bin | predicted 0.001, **empirical 0.096** | Pricer is ~96× overconfident on tails |
| `0.50–0.70` bins | predicted 0.55/0.65, empirical 0.44/0.49 | Overconfident in the body too |

**Root cause:** `vol_window_minutes: 60` — a 60-minute realized-vol window is
used to price contracts that resolve 10–34 hours out. Calm short windows
systematically understate the volatility BTC actually realizes over a day, so
the pricer calls tail outcomes "near-impossible" when they happen ~10% of the
time. The "edge" and S-curve seen on the dashboard were **measurement
artifacts, not alpha**.

**Secondary findings (must also be fixed):**

1. **`src/measurement/reporter.py`** only scans the 2,000 most-recent
   `prob_estimate` rows — all still-open markets — so live calibration never
   populates.
2. **`src/pricing/ticker.py::parse_ticker`** computes a `now`-relative
   `close_time`, so historical markets settle against the wrong timestamp. The
   backtest works around this with an epoch anchor; the live path does not.
3. **Data continuity:** Coinbase candle history has 134 gaps >5 min; the bot
   collected candles only ~32% of wall-clock time (2,324 candles / 7,165 min)
   due to crashes/restarts. ~50% of estimates are unsettleable.

---

## 2. Phase 2 Objective & Success Criteria

**Objective:** Determine whether a realized-volatility BS pricer can be made
well-calibrated at the tails. If yes, advance Strategy 2 toward live. If no,
fail fast and pivot.

**Hard success gate (GATE A — pricer calibration):**
Re-running `python -m src.backtest.engine` after the vol fix must show:

- `0.00–0.10` predicted bin: empirical YES frequency **< 0.02** (currently 0.096)
- Overall **log loss < 0.69** (beats coin flip; currently 1.28)
- No reliability bin off by **> 0.07** absolute between mean_pred and emp_freq
- Settleable sample size **≥ 20,000** (requires WS3 data fix)

If GATE A fails after WS1 + reasonable tuning → invoke the pivot options in §5.

---

## 2.1 GATE A Results _(2026-05-20)_

Run via `python -m src.backtest.engine --gate-a` on `data/bot.db`
(43,635 settleable estimates, the same population as the Phase 1 baseline).

| Mode               | settle n | Brier | Log loss | Tail emp (0.0–0.1) | Max bin gap | Monotone | GATE A |
|--------------------|---------:|------:|---------:|-------------------:|------------:|---------:|:------:|
| replay (Phase 1)   |  43,635  | 0.079 | **1.280**| **0.096**          | 0.158       | FAIL     | ❌ FAIL |
| reprice/fixed      |  43,635  | 0.078 | **1.251**| **0.096**          | 0.124       | FAIL     | ❌ FAIL |
| reprice/horizon    |  43,635  | 0.078 | **1.251**| **0.096**          | 0.124       | FAIL     | ❌ FAIL |
| reprice/blend      |  43,635  | 0.079 | **1.282**| **0.096**          | 0.215       | FAIL     | ❌ FAIL |
| reprice/ewma       |  43,635  | 0.079 | **1.270**| **0.096**          | 0.190       | FAIL     | ❌ FAIL |

Bar (for reference): log loss < 0.69, tail emp < 0.02, max bin gap ≤ 0.07,
settleable n ≥ 20,000. The settleable-n criterion passes; **every other
criterion fails by a wide margin in every mode.**

**Why the four vol modes are near-identical on this data:** Phase 1's stored
`horizon_seconds` were all in the minute-scale (a side effect of the
now-relative `parse_ticker` bug fixed in T01). With short stored horizons,
`horizon_scaled` clamps to the floor window (≈ `fixed`), and `blend`/`ewma`
also fall back toward short-window behavior. So this sweep does **not**
discriminate vol modes on horizon — it tells us that even with the floor,
blend, and EWMA refinements, the realized-vol-into-BS premise misses the
calibration bar at the same place the Phase 1 baseline did.

**Decision:** **GATE A FAIL.** Proceed to **Phase 3 Track B (pivot).** Plan
prescribes B1 (empirical calibration layer — cheapest) first, then B2
(Deribit IV) if insufficient, then B3 (reframe/abandon) as the hard stop.

The unanimous failure across realized-vol-only modes is itself a finding: the
edge isn't in *which* realized window you average, it's that realized vol
alone is the wrong input. The B1 calibration layer keeps BS as a feature; B2
swaps the input for option-implied vol. Both are anticipated by the plan and
do not require revisiting Phase 2.

---

## 3. Workstreams

### WS1 — Horizon-Matched Volatility _(core fix)_

**Problem:** one fixed 60-min vol window for all horizons.

**Approach (incremental, cheapest first):**

1. **W1.1 — Horizon-scaled window.** Replace the fixed window with a lookback
   proportional to the contract horizon, floored and capped:
   `window_min = clamp(horizon_minutes, 60, 1440)`.
   File: `src/pricing/volatility.py` (new function `horizon_matched_vol`),
   `src/pricing/pricer.py` (call site), `src/config.py` (`PricerConfig`).
2. **W1.2 — Multi-timescale blend.** Compute realized vol at {1h, 6h, 24h}
   windows; select/blend by horizon bucket. Robust when one window is in a calm
   regime.
3. **W1.3 — EWMA vol with horizon-tuned half-life** (refinement if W1.1/W1.2
   under-deliver). Exponentially weighted, less sensitive to a single calm hour.
4. **W1.4 — Long-window vol floor.** Never price below trailing 7-day realized
   vol — a structural guard against the exact failure we observed.

**Config additions (`PricerConfig`):**
`vol_mode: "fixed" | "horizon_scaled" | "blend" | "ewma"`,
`vol_window_floor_min`, `vol_window_cap_min`, `vol_long_floor_days`.
Keep `fixed` as the default so behavior is opt-in and A/B-comparable.

**Acceptance:** backtest log loss drops materially and the `0.00–0.10` bin
empirical frequency moves toward its predicted value (toward GATE A).

---

### WS2 — Settlement & Calibration Bug Fixes

1. **W2.1 — `parse_ticker` close-time.** Add an explicit
   `settlement_mode: bool` (or a dedicated `resolve_close_time(market_id)`)
   that returns the contract's true anchored close time regardless of `now`.
   Live pricing keeps the next-boundary fallback only when the ticker lacks a
   resolvable date. File: `src/pricing/ticker.py`. Add unit tests for a
   historical ticker resolving days later.
2. **W2.2 — `reporter.py` settlement window.** Scan estimates whose
   `close_time` has **passed** (not just the most recent rows). Page through
   all unsettled estimates, not a recency-limited slice.
   File: `src/measurement/reporter.py`.
3. **W2.3 — Regression tests.** `tests/unit/test_reporter.py`,
   extend `tests/unit/test_ticker.py` for the historical-settlement case.

**Acceptance:** live `calibration_snapshot` populates within one resolution
cycle; backtest and live calibration agree within sampling noise.

---

### WS3 — Data Continuity / Infrastructure Reliability

Without continuous BTC candles, no backtest or live calibration is
trustworthy. This is a **prerequisite for trusting GATE A at scale**.

1. **W3.1 — Coinbase REST gap backfill on startup.** On boot, detect missing
   minute ranges in `coinbase_candle` and backfill via paginated Coinbase REST
   (the endpoint caps ~300 candles/call → loop with time windows).
   File: `src/data/coinbase_ws.py` (`fetch_history` → add range backfill).
2. **W3.2 — Periodic gap repair.** A low-frequency task that re-backfills any
   gap created by a transient WS disconnect.
3. **W3.3 — Process resilience.** Supervisor/auto-restart (systemd, `pm2`, or a
   shell `until` loop) plus the existing decoupled housekeeping task so a
   stalled Kalshi feed never blocks Coinbase persistence (already partially
   done in Phase 1).
4. **W3.4 — Data-quality panel.** Dashboard tile: candle coverage %, largest
   gap, settleable-estimate %. Makes silent data loss visible.

**Acceptance:** ≥ 95% minute-candle coverage over any trailing 48h window;
backtest settleable rate ≥ 80%.

---

### WS4 — Backtest Hardening & Re-validation

1. **W4.1 — A/B vol modes.** `--vol-mode` flag on `src/backtest/engine.py` to
   compare `fixed` vs `horizon_scaled` vs `blend` on the same data in one run.
2. **W4.2 — Re-price mode (optional).** Currently the backtest replays *stored*
   probabilities. Add a mode that **re-runs the pricer** from candle history so
   new vol modes can be evaluated on historical data **without waiting for new
   live data**. (Requires WS3 data quality to be meaningful.)
3. **W4.3 — Realistic execution replay.** Honor rate-limit + position caps so
   trade-level PnL is faithful (Phase 1 used "one entry/contract"; keep that as
   a mode, add a faithful mode).
4. **W4.4 — Edge-decile monotonicity test** as an automated pass/fail, not just
   a printed table.

**Acceptance:** one command produces a side-by-side vol-mode comparison with
calibration + PnL + decile monotonicity for each.

---

### WS5 — Decision Gate & Pivot Options

Run after WS1–WS4. See §5 for the explicit gate logic.

**Pivot options if realized-vol cannot be calibrated:**

- **P1 — Deribit option-implied volatility.** Replace realized vol with
  market-implied vol from the BTC option chain (call-spread ≈ digital). The
  "correct" input; new infra (Deribit client, IV interpolation).
- **P2 — Empirical calibration layer.** Keep BS as a feature; learn an isotonic
  / Platt mapping from raw BS prob → calibrated prob using settled history.
  Cheaper, but only works if BS is monotonic w.r.t. truth (the reliability
  table suggests it roughly is).
- **P3 — Abandon Strategy 2**, revisit the ranked alternatives from the
  original research (near-expiry convergence as a filter; ensemble).

---

## 4. Sequencing & Dependencies

```
WS2 (bug fixes)         ──┐  small, unblock correctness first
WS3 (data continuity)   ──┼──► WS1 (horizon vol) ──► WS4 (re-validate) ──► WS5 (gate)
                          │        ▲
WS4.2 re-price mode  ─────┘        └─ needs WS3 data quality to be meaningful
```

Recommended order:

1. **WS2** — fast, pure correctness; do first so all later numbers are trustworthy.
2. **WS3.1 + WS3.3** — backfill + resilience; without this, every later metric is noisy.
3. **WS1.1** — simplest horizon-scaled window; re-run backtest immediately.
4. **WS4.1 / WS4.2** — A/B + re-price mode to iterate vol modes fast.
5. **WS1.2 → W1.4** — escalate only if W1.1 misses GATE A.
6. **WS5** — decision gate.

---

## 5. Decision Gates (Go / No-Go)

**GATE A — Pricer calibration** (after WS1 + WS3):
- PASS → proceed to GATE B.
- FAIL after W1.1–W1.4 → pivot P1 (Deribit IV) or P2 (calibration layer).

**GATE B — Forward paper validation** (continuous data, ≥ 5 days):
- Live `calibration_snapshot` Brier and per-bin reliability match the backtest
  within sampling noise.
- Edge-decile PnL is **monotonic increasing** (higher predicted edge →
  higher realized PnL/contract).
- PASS → GATE C.
- FAIL → diagnose live/backtest divergence (execution, latency, data) before
  any sizing discussion.

**GATE C — Tiny-size live** (only after A + B):
- `max_contracts_per_trade: 1`, smallest meaningful bankroll.
- Two weeks green realized PnL before any scale-up discussion.

No live trading (`BOT_ALLOW_LIVE_TRADING=true`) before GATE C is explicitly
cleared.

---

## 6. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| BTC fat tails make *any* lognormal/BS model structurally miscalibrated | Medium | W1.4 vol floor; P2 empirical calibration layer; P1 IV pivot |
| Horizon-matched vol fixes tails but kills the (apparent) edge entirely | High | Expected — the edge was an artifact. Real edge, if any, is small; size accordingly |
| Resolution-source basis (Kalshi settles off an index, we price off Coinbase) | Medium | Quantify basis vs CF Benchmarks; widen edge threshold near strike |
| Data backfill from Coinbase REST has its own gaps/limits | Low | Pagination + gap-repair task (W3.2); surface coverage in dashboard (W3.4) |
| Overfitting vol-mode params to the existing ~4 days of data | Medium | Hold out a time slice; require GATE B forward validation, not just backtest |

---

## 7. Out of Scope for Phase 2

- New strategies (near-expiry filter, ensemble) — revisit only if Strategy 2 is abandoned.
- Live trading at any size beyond GATE C's tiny-size test.
- Multi-venue / cross-exchange arbitrage.
- Dashboard cosmetic work beyond the WS3.4 data-quality panel.

---

## 8. Concrete Task Checklist

- [ ] **W2.1** `parse_ticker` settlement-mode close time + unit tests
- [ ] **W2.2** `reporter.py` scans all passed-close estimates, not recent slice
- [ ] **W2.3** regression tests for historical settlement & calibration
- [ ] **W3.1** Coinbase REST paginated gap backfill on startup
- [ ] **W3.2** periodic gap-repair task
- [ ] **W3.3** process supervisor / auto-restart
- [ ] **W3.4** dashboard data-quality tile (coverage %, max gap, settleable %)
- [ ] **W1.1** horizon-scaled vol window + `PricerConfig` knobs + `vol_mode`
- [ ] **W4.1** backtest `--vol-mode` A/B flag
- [ ] **W4.2** backtest re-price mode (recompute pricer from candles)
- [ ] **W4.3** faithful execution replay (rate-limit + position caps)
- [ ] **W4.4** automated edge-decile monotonicity pass/fail
- [ ] Re-run backtest → evaluate **GATE A**
- [ ] **W1.2–W1.4** escalate vol modeling only if GATE A misses
- [ ] **GATE A** decision: proceed or pivot (P1/P2/P3)
- [ ] **GATE B** forward paper validation (≥5 days, continuous data)
- [ ] **GATE C** tiny-size live (only if A + B pass)

---

## 9. Commit-by-Commit Task Breakdown

18 atomic tasks in dependency order. Each is sized for **one commit that leaves
the test suite green** (`python -m pytest tests/unit/ -q` must pass after every
one). Bundle the implementation and its tests in the same commit. Suggested
commit messages match the repo's existing terse style — adjust freely.

> **Rule for every commit:** run `python -m pytest tests/unit/ -q` before
> committing. A task is not done if it reds the suite.

### Block 1 — Correctness bug fixes (do first; unblocks trustworthy metrics)

**T1 — parse_ticker: true settlement close-time**
- Scope: add `settlement: bool = False` (or `resolve_close_time(market_id)`) to
  `parse_ticker` that returns the contract's real anchored close time
  regardless of `now`. Keep the next-boundary fallback only for the live path
  when no date is resolvable.
- Files: `src/pricing/ticker.py`, `tests/unit/test_ticker.py`
- Acceptance: new test — a ticker that closed days before `now` resolves to its
  true historical close, not a now-relative boundary. Existing ticker tests
  still pass.
- Commit: `ticker: true settlement close-time (fix now-relative bug)`

**T2 — reporter: settle all passed-close estimates**
- Scope: `settle_and_snapshot` must scan estimates whose `close_time` has
  passed, paging through history — not just the 2,000 most-recent rows.
  Use T1's settlement close-time.
- Files: `src/measurement/reporter.py`, `tests/unit/test_reporter.py` (new)
- Acceptance: new test seeds estimates for a closed market + candles → a
  calibration snapshot is produced with the correct outcomes.
- Commit: `reporter: settle on close-time, not recency window`

**T3 — backtest uses shared settlement helper**
- Scope: replace the backtest's local epoch-anchor hack with the T1 helper so
  backtest and live settlement share one code path.
- Files: `src/backtest/engine.py`
- Acceptance: `python -m src.backtest.engine` reproduces the same coverage /
  Brier as before (regression check, numbers unchanged).
- Commit: `backtest: use shared settlement close-time helper`

### Block 2 — Data continuity (metrics are noise until this lands)

**T4 — Coinbase REST paginated gap-backfill (pure function)**
- Scope: function that, given a start/end minute range, fetches all Coinbase
  1-min candles via paginated REST (≤300/call, loop windows). No wiring yet.
- Files: `src/data/coinbase_ws.py`, `tests/unit/test_coinbase_backfill.py` (new,
  mock the HTTP client)
- Acceptance: unit test with a mocked multi-page response returns a contiguous,
  de-duplicated, time-sorted candle list.
- Commit: `coinbase: paginated REST gap-backfill function`

**T5 — Backfill missing candle ranges on startup**
- Scope: on boot, detect gaps in `coinbase_candle` vs wall clock and call T4 to
  fill them before the strategy loop starts.
- Files: `src/runtime/main.py`, `src/storage/repository.py` (gap query helper)
- Acceptance: start against a DB with a known gap → gap is filled; log line
  reports candles backfilled.
- Commit: `runtime: backfill coinbase gaps on startup`

**T6 — Periodic gap-repair task**
- Scope: low-frequency background task that re-backfills any gap from a
  transient WS disconnect while running.
- Files: `src/runtime/main.py`
- Acceptance: simulated WS drop → gap is repaired within one repair interval
  (manual/log verification acceptable; note in commit).
- Commit: `runtime: periodic coinbase gap-repair task`

**T7 — Process supervisor / auto-restart**
- Scope: a supervised run script (`scripts/run_bot.sh` with an `until` loop, or
  a systemd/pm2 unit) that restarts the bot on crash and logs restarts.
- Files: `scripts/run_bot.sh` (new), short README note in the script header
- Acceptance: kill the bot PID → it comes back within N seconds; restart logged.
- Commit: `ops: supervised auto-restart run script`

**T8 — Dashboard data-quality panel**
- Scope: a tile showing candle coverage % (trailing 48h), largest gap, and
  settleable-estimate %.
- Files: `dashboard/app.py`, repository helper if needed
- Acceptance: panel renders real numbers; coverage matches a manual SQL check.
- Commit: `dashboard: data-quality panel (coverage, gaps, settleable%)`

### Block 3 — Horizon-matched volatility (the core fix)

**T9 — horizon_matched_vol pure function**
- Scope: new vol estimator that takes a horizon and returns annualized vol from
  a horizon-scaled lookback window `clamp(horizon_min, floor, cap)`. Pure, no
  pricer wiring.
- Files: `src/pricing/volatility.py`, `tests/unit/test_volatility.py` (extend)
- Acceptance: unit tests — longer horizon selects a longer window; clamps at
  floor/cap; matches `close_to_close_vol` when horizon == window.
- Commit: `volatility: horizon-matched realized-vol estimator`

**T10 — PricerConfig: vol_mode + window knobs (no behavior change)**
- Scope: add `vol_mode: "fixed"|"horizon_scaled"|"blend"|"ewma"`,
  `vol_window_floor_min`, `vol_window_cap_min`, `vol_long_floor_days` to
  `PricerConfig`, default `vol_mode: "fixed"`. Plumb into `settings.yaml`.
- Files: `src/config.py`, `config/settings.yaml`
- Acceptance: settings load; default keeps `fixed` → existing tests unchanged.
- Commit: `config: vol_mode + horizon-vol knobs (default fixed)`

**T11 — Wire horizon_scaled mode into CoinbasePricer**
- Scope: `CoinbasePricer` selects the vol estimator by `vol_mode`. `fixed`
  unchanged; `horizon_scaled` uses T9 with the contract horizon.
- Files: `src/pricing/pricer.py`, `tests/unit/test_pricer.py` (extend)
- Acceptance: test — same inputs, `horizon_scaled` yields a different (longer-
  window) vol than `fixed` for a long-horizon contract; `fixed` path identical
  to before.
- Commit: `pricer: select vol estimator by vol_mode`

**T12 — Multi-timescale blend vol mode**
- Scope: implement `blend` — combine {1h, 6h, 24h} realized vols by horizon
  bucket.
- Files: `src/pricing/volatility.py`, `src/pricing/pricer.py`, tests
- Acceptance: unit test for blend weighting; pricer `blend` path covered.
- Commit: `volatility: multi-timescale blend vol mode`

**T13 — Long-window vol floor guard**
- Scope: never price below trailing `vol_long_floor_days` realized vol;
  applies across modes.
- Files: `src/pricing/volatility.py` / `pricer.py`, tests
- Acceptance: test — a calm short window cannot push vol below the long floor.
- Commit: `volatility: trailing long-window vol floor`

**T14 — EWMA vol mode (refinement)**
- Scope: implement `ewma` with a horizon-tuned half-life.
- Files: `src/pricing/volatility.py`, `src/pricing/pricer.py`, tests
- Acceptance: unit test — EWMA reacts faster than SMA to a vol jump; pricer
  `ewma` path covered.
- Commit: `volatility: EWMA vol mode (horizon-tuned half-life)`

### Block 4 — Backtest hardening & re-validation

**T15 — Backtest --vol-mode A/B flag**
- Scope: `--vol-mode` on `src/backtest/engine.py` to run/report a mode.
- Files: `src/backtest/engine.py`
- Acceptance: `--vol-mode fixed` reproduces current numbers; flag documented in
  `--help`.
- Commit: `backtest: --vol-mode flag`

**T16 — Backtest re-price mode (recompute pricer from candles)**
- Scope: a mode that re-runs `CoinbasePricer` from candle history instead of
  replaying stored probs, so new vol modes can be evaluated on past data.
- Files: `src/backtest/engine.py`
- Acceptance: re-price with `fixed` approximates stored-prob calibration within
  tolerance (sanity); `horizon_scaled` produces a distinct result.
- Commit: `backtest: re-price-from-candles mode`

**T17 — Faithful execution replay mode**
- Scope: optional mode honoring rate-limit + position caps so trade PnL is
  realistic; keep "one entry/contract" as the other mode.
- Files: `src/backtest/engine.py`
- Acceptance: replay trade count is bounded by rate-limit math; both modes
  selectable via flag.
- Commit: `backtest: faithful rate-limited execution replay`

**T18 — Automated edge-decile monotonicity check + GATE A record**
- Scope: turn the decile table into an automated PASS/FAIL (monotone increasing
  avg PnL/contract); re-run the full backtest under the best vol mode and
  record the **GATE A** outcome (numbers + decision) in §2 of this file.
- Files: `src/backtest/engine.py`, `PHASE2_PLAN.md`
- Acceptance: backtest prints a clear GATE A PASS/FAIL; this doc updated with
  the measured `0.00–0.10` bin freq, log loss, and the go/pivot decision.
- Commit: `backtest: GATE A monotonicity check + record decision`

### Sequencing notes

- **T1 → T2 → T3** must be in order (T2 and T3 depend on T1's helper).
- **T4 → T5 → T6** in order; **T7, T8** independent (can interleave anywhere
  after Block 1).
- **T9 → T10 → T11** in order; **T12/T13/T14** only if GATE A misses after T11.
- **T15 → T16 → T17** in order; **T18** last (it's the gate).
- Re-run `python -m src.backtest.engine` after T11 and after each of
  T12–T14 — that running comparison is the whole point of Phase 2.
