# Kalshi BTC Bot — Roadmap

_Index of all phase plans. Read this first._

**The through-line question:** Can a volatility-based binary pricer
systematically beat Kalshi's BTC market prices — and if so, can we run it
safely with real money?

Every phase exists to answer one slice of that, behind a **binding decision
gate**. Gates are not suggestions: a failed gate stops or redirects work. No
live trading before Phase 3 GATE C.

---

## Phase map

| Phase | Objective | Binding gate | Status | Plan |
|---|---|---|---|---|
| **1 — Build & Discover** | Build Strategy 2 (BS pricer vs Kalshi) end-to-end + dashboard + backtest; find out if it works | Post-mortem: is there *any* signal? | ✅ Complete (negative result) | [PHASE1_PLAN.md](PHASE1_PLAN.md) |
| **2 — Remediate & Validate** | Fix the bugs Phase 1 exposed; horizon-matched vol; prove or disprove the pricer can be calibrated | **GATE A** — pricer calibration on backtest | ✅ Complete (GATE A FAIL, 2026-05-20) | [PHASE2_PLAN.md](PHASE2_PLAN.md) |
| **3 — Live or Pivot** | If GATE A passes: validate forward → tiny-size live. If it fails: escalating pivots, then hard stop | **GATE B/C** (live) or **mini-gates** (pivot) | 🔄 GATE B forward paper pending | [PHASE3_PLAN.md](PHASE3_PLAN.md) |
| **4 — Operate & Scale** | Run as a production system; scale capital responsibly; detect alpha decay | **GATE D** (each scale step) | ⏸ Gated on GATE C | [PHASE4_PLAN.md](PHASE4_PLAN.md) |

---

## Where we are now

**Phase 2 complete · GATE A FAIL (2026-05-20).** All five vol modes
(replay baseline, reprice/fixed, reprice/horizon_scaled, reprice/blend,
reprice/ewma) miss the bar by a wide margin: log loss 1.25–1.28 vs the 0.69
threshold, tail-bin empirical 0.096 vs the 0.02 threshold. Full per-mode
breakdown in [PHASE2_PLAN.md §2.1](PHASE2_PLAN.md). Realized-vol-only Black–
Scholes does not pass calibration for these contracts.

**Active work: Phase 3 Track B rejoined Track A validation.** **B1 — empirical calibration
layer**: mini-gate PASS (2026-05-20). Isotonic regression on raw BS-probs
drops OOS log loss from 1.69 → 0.27 and tail-bin emp from 0.124 → 0.019
(80/20 time-series split, no look-ahead). Multi-fold CV shows the tail
criterion is brittle (2/4 folds pass) — log loss criterion is solid — so
B1 is a *tentative* pass; the binding test is GATE B forward validation.

**Built since B1:** A1 near-strike guard, live calibrator artifact wiring, A2
top-book-aware paper fills, A4 risk hardening, X1 alerting, and X2 runbook.

**Next:** run **GATE B** (forward paper, ≥5 days continuous). GATE C remains
blocked until GATE B passes and a human signs the tiny-live checklist.

---

## Operating principles (apply to every phase)

1. **Gates are binding.** A failed gate redirects or halts work — it is never a
   cue to "keep tweaking until the number looks good."
2. **Fail fast, cheaply.** Cheapest viable approach first; escalate only on a
   measured miss; stop when the escalation ladder is exhausted.
3. **No look-ahead, no self-grading.** Validate forward and against the actual
   settlement source, not the feed we price from.
4. **No real money before Phase 3 GATE C**, and only behind a written go/no-go
   checklist with human sign-off.
5. **Every claim is a number.** "Looks good" is not a result; a Brier score on
   out-of-sample data is.
