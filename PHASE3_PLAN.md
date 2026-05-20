# Phase 3 Plan — Post-GATE-A: Path to Live, or Pivot

_Status: 🔄 Track B active (Phase 2 GATE A FAIL) · B1 mini-gate PASS (2026-05-20, brittle on tail) · Next: A1 basis study_

---

## 1. Entry Condition

Phase 3 **must not start** until Phase 2 is complete and **GATE A has been run
and recorded in `PHASE2_PLAN.md`** (T19). Phase 2 delivered: WS2 bug fixes,
WS3 data continuity, the WS1 horizon-matched-vol ladder, the hardened backtest
(WS4), and a GATE A pass/fail result.

Phase 3 is **conditional and branches on that result.** Do not pick a track
before the number exists — that's the discipline the gate exists to enforce.

---

## 2. The Branch Point

```
                 ┌─ GATE A PASS ──────────────► TRACK A  (Validation → Live)
Phase 2 GATE A ──┤
                 └─ GATE A FAIL (after the ─────► TRACK B  (Pivot)
                    W1.1→W1.4 vol ladder)

CROSS-CUTTING (X1–X3) runs regardless of branch.
```

GATE A pass bar (from Phase 2 §2): `0.00–0.10` bin empirical < 0.02, overall
log loss < 0.69, no reliability bin off by > 0.07, settleable n ≥ 20,000.

**Phase 2 GATE A result (2026-05-20): ❌ FAIL.** All five vol modes (replay,
fixed, horizon_scaled, blend, ewma) missed by wide margins (log loss ≥ 1.25
vs 0.69 bar; tail bin emp ≈ 0.096 vs 0.02 bar). Full per-mode breakdown in
[PHASE2_PLAN.md §2.1](PHASE2_PLAN.md#21-gate-a-results-2026-05-20).
**Branch taken: TRACK B (pivot).**

---

## 2.1 Track B Mini-Gate Results

### B1 — Empirical calibration layer · _2026-05-20_

`python -m src.backtest.engine --mini-gate-b1` (43,635 settled pairs,
time-series 80/20 split, no shuffle, no look-ahead).

| Metric | Raw (OOS) | Calibrated (OOS) | Bar |
|---|---:|---:|:--|
| Log loss | 1.6942 | **0.2658** | < 0.69 ✅ |
| Tail-bin emp (0.0–0.1) | 0.1241 | **0.0192** | < 0.02 ✅ |
| Brier | 0.1030 | 0.0777 | lower better |

**Mini-gate per plan letter: ✅ PASS.**

**Robustness audit (4-fold expanding-window CV):**

| fold | train n | test n | cal log loss | cal tail emp | pass? |
|---:|---:|---:|---:|---:|:-:|
| 1 | 8,727 | 8,727 | 0.27 | 0.072 | ❌ |
| 2 | 17,454 | 8,727 | 0.24 | 0.016 | ✅ |
| 3 | 26,181 | 8,727 | 0.41 | 0.056 | ❌ |
| 4 | 34,908 | 8,727 | 0.27 | 0.019 | ✅ |

- **Log loss criterion is robust** (< 0.69 across all 4 folds; 0.24–0.41).
- **Tail-bin criterion is brittle** (passes 2/4 folds). The dataset isn't
  homogeneous; calibration on Phase-1 data has time-varying tightness.

**Decision:** record B1 as PASS per the plan as written; do **not** treat it
as a green light. The forward GATE B (live calibration over ≥5 continuous
days) is the binding test that resolves the multi-fold concern. The B1
mapping is genuinely informative (log loss dropped from 1.69 → 0.27 OOS) but
the tail criterion needs forward confirmation, not backward overfit.

**Next:** rejoin Track A at **A1 (resolution-source basis study)**. Implementation
will produce + persist the calibrator artifact and wire it into the live
strategy so GATE B exercises the same calibrated path.

---

## 3. TRACK A — Validation to Live _(only if GATE A passed)_

Calibration being good on a *backtest* is necessary, not sufficient. Track A
closes the gap between "the pricer is calibrated on history" and "we can risk
real money."

### A1 — Resolution-source basis study
Kalshi settles off its own index/source; we price off Coinbase spot. Quantify
the basis distribution between Coinbase and the actual Kalshi settlement source
over collected history. Parameterize a near-strike guard: suppress signals when
`|spot − strike|` is within the measured p95 basis.
**Files:** new `src/measurement/basis.py`, `src/strategy/edge.py`.
**Acceptance:** basis p50/p95/p99 documented; near-strike guard configurable
and on by default.

### A2 — Execution-quality measurement & paper-fill calibration
Compare the paper executor's fills against realistic Kalshi book behavior
(queue position, partial fills, slippage vs. observed depth). Tighten
`PaperExecutor` so paper PnL is a faithful lower bound, not optimistic.
**Files:** `src/execution/paper_executor.py`, `src/execution/slippage.py`.
**Acceptance:** paper vs. modeled-realistic fill divergence < 1¢ median.

### A3 — GATE B: forward paper validation
With continuous data (post-WS3) and the bot under the supervisor:
- ≥ 5 trading days of forward paper data, no backfilled gaps in the window.
- Live `calibration_snapshot` Brier/per-bin matches the backtest within
  sampling noise.
- Edge-decile realized PnL is **monotonic increasing** (the W4.4 check, live).
- Net paper PnL **> fees** over the window.
**PASS → A4. FAIL → diagnose live/backtest divergence (A2, latency, data)
before anything else. Do not proceed to live on a failed GATE B.**

### A4 — Risk & sizing hardening
Before any real order: per-market position cap, gross-exposure cap,
**tail-short exposure cap** (limit net short on < 0.10 / > 0.90 contracts),
**vol-regime kill switch** (halt when realized vol jumps > Nσ), daily-loss
limit, max-drawdown halt. All enforced in `RiskEngine`, unit-tested.
**Files:** `src/risk/engine.py`, `src/risk/kill_switch.py`, tests.

### A5 — GATE C: tiny-size live
- `max_contracts_per_trade: 1`, smallest meaningful bankroll.
- `BOT_ALLOW_LIVE_TRADING=true` only behind an explicit written go/no-go
  checklist and human sign-off (checklist lives in the ops runbook, X2).
- Two weeks of **green realized PnL** at this size before any scale-up talk.

### A6 — Capital scaling ladder
Defined step-ups (e.g., 1 → 3 → 10 contracts) each requiring a sustained-green
window and each with an automatic rollback trigger (drawdown / calibration
drift). No discretionary size jumps.

---

## 4. TRACK B — Pivot _(only if GATE A failed after the W1 ladder)_

A GATE A failure means realized-vol-into-BS is structurally miscalibrated for
these contracts. Try the cheapest rescue first; escalate; then stop.

### B1 — Empirical calibration layer _(cheapest — try first)_
Learn an isotonic / Platt mapping `raw_BS_prob → calibrated_prob` from settled
history. Keep BS as a feature, not the final probability.
**Files:** new `src/strategy/calibrator.py`, wired in `src/strategy/edge.py`.
**Method discipline:** time-series cross-validation, **no look-ahead** (fit on
past, score on strictly-later folds).
**Mini-gate B1:** out-of-sample log loss < 0.69 **and** `0.00–0.10` bin
empirical < 0.02 on the held-out fold. PASS → rejoin Track A at A1.

### B2 — Deribit option-implied volatility _(if B1 insufficient)_
Replace the realized-vol input with market-implied vol from the BTC option
chain (a digital ≈ a tight call-spread). New infrastructure.
**Files:** new `src/data/deribit.py`, IV-surface interpolation, pricer hook.
**Mini-gate B2:** same calibration bar as GATE A on backtest **and** a forward
window. PASS → rejoin Track A at A1.

### B3 — Reframe or abandon _(if B1 and B2 both fail)_
The "vol-vs-Kalshi" premise is not a real edge. Document the negative result in
this file. Options: reduce Strategy 2 to a **near-expiry-convergence filter
only** (no standalone alpha claim), or stop. **No further capital or engineering
time without a brand-new, written hypothesis.** This is a hard stop, not a
"keep tweaking."

---

## 5. Cross-Cutting Workstreams _(run regardless of branch)_

### X1 — Live observability & alerting
Promote the existing `heartbeat` log into real alerts: stalled-feed alarm,
data-quality alarm (48h coverage < 95%), PnL/drawdown alert, **calibration-drift
alarm** (rolling Brier worsens > threshold). Out-of-band channel (not just logs).

### X2 — Operational runbook
Restart/recovery steps, kill-switch procedure, the GATE C go/no-go checklist,
what each X1 alert means, DB backup/restore. Lives in `docs/RUNBOOK.md`.

### X3 — Settlement-source fidelity
Where possible, settle calibration against the **actual Kalshi resolution
source**, not the Coinbase proxy — for truth measurement, distinct from the
pricing input. Reduces the risk that "good calibration" is an artifact of
scoring against the same feed we price from.

---

## 6. Decision Gates

| Gate | Track | Bar | On fail |
|---|---|---|---|
| **GATE B** | A | forward paper: live≈backtest calibration, monotonic decile PnL, net > fees, ≥5d continuous | diagnose divergence; no live |
| **GATE C** | A | tiny-size live, written checklist + human sign-off, 2wk green realized | revert to paper |
| **GATE D** | A | each scaling step: sustained green + auto-rollback armed | roll back a step |
| **Mini-B1** | B | OOS log loss < 0.69 & tail bin < 0.02 | escalate to B2 |
| **Mini-B2** | B | GATE-A bar on backtest + forward | escalate to B3 |
| **Hard stop** | B3 | — | no live, no spend without new hypothesis |

No `BOT_ALLOW_LIVE_TRADING=true` before GATE C, ever.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Calibration layer overfits history | time-series CV, OOS fold, then GATE B forward validation |
| Resolution-source basis underestimated | A1 quantifies before any live; near-strike guard |
| Live ≠ paper (latency, queue position) | A2 fill calibration + GATE B divergence check |
| Process dies mid-live | supervisor (Phase 2) + X1 alerting + A4 kill switches |
| Regime shift after validation | A4 vol-regime kill switch + X1 calibration-drift alarm |
| "Keep tweaking" past a real failure | hard stop in B3; gates are binding |

---

## 8. Out of Scope for Phase 3

- Multi-venue / cross-exchange arbitrage.
- New asset classes or contract families beyond BTC.
- ML beyond the B1 calibration layer.
- Any Track work before Phase 2 GATE A is recorded.

---

## 9. Task Checklist

- [ ] Confirm Phase 2 GATE A result is recorded in `PHASE2_PLAN.md`; select Track A or B
- **Track A:** [ ] A1 basis study · [ ] A2 fill calibration · [ ] **GATE B** · [ ] A4 risk hardening · [ ] **GATE C** (sign-off) · [ ] A6 scaling ladder
- **Track B:** [ ] B1 calibration layer + mini-gate · [ ] B2 Deribit IV + mini-gate · [ ] B3 reframe/abandon decision (documented)
- **Cross-cutting:** [ ] X1 alerting · [ ] X2 runbook · [ ] X3 settlement-source fidelity
