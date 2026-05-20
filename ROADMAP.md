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
| **3 — Live or Pivot** | If GATE A passes: validate forward → tiny-size live. If it fails: escalating pivots, then hard stop | **GATE B/C** (live) or **mini-gates** (pivot) | 🔄 Track B (pivot) active | [PHASE3_PLAN.md](PHASE3_PLAN.md) |
| **4 — Operate & Scale** | Run as a production system; scale capital responsibly; detect alpha decay | **GATE D** (each scale step) | ⏸ Gated on GATE C | [PHASE4_PLAN.md](PHASE4_PLAN.md) |

---

## Where we are now

**Phase 2 complete · GATE A FAIL (2026-05-20).** All five vol modes
(replay baseline, reprice/fixed, reprice/horizon_scaled, reprice/blend,
reprice/ewma) miss the bar by a wide margin: log loss 1.25–1.28 vs the 0.69
threshold, tail-bin empirical 0.096 vs the 0.02 threshold. Full per-mode
breakdown in [PHASE2_PLAN.md §2.1](PHASE2_PLAN.md). Realized-vol-only Black–
Scholes does not pass calibration for these contracts.

**Active work: Phase 3 Track B (pivot).** Per the plan, the cheapest viable
pivot is run first: **B1 — empirical calibration layer** (isotonic / Platt
mapping from raw BS prob → calibrated prob, fit on settled history with
time-series cross-validation). If B1 misses its mini-gate, **B2 — Deribit
option-implied volatility**. If both miss, **B3 — reframe or abandon** is a
hard stop, not a "keep tweaking."

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
