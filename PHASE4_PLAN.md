# Phase 4 Plan — Operate & Scale

_Status: draft · Entry-gated on Phase 3 GATE C · Track-A only_

---

## 1. Entry Condition

Phase 4 **must not start** unless Phase 3 Track A reached **GATE C** and held it:
tiny-size live trading with **two consecutive weeks of green realized PnL**, all
risk kill-switches (Phase 3 A4) live-tested, and the ops runbook (Phase 3 X2) in
place.

If Phase 3 ended on Track B (pivot) or the B3 hard stop, Phase 4 does not exist
until a *new* validated strategy re-enters at Phase 3 GATE C.

This is the only phase that touches meaningful capital. It is an **operations**
phase, not a research phase: the edge is assumed proven; the work is running it
without blowing up and knowing when it stops working.

---

## 2. Objective & Success Criteria

**Objective:** Run the bot as a production system and scale capital responsibly
while continuously verifying the edge still exists.

**Phase success =** capital scaled along the ladder with realized risk-adjusted
return within modeled bounds, **and** an alpha-decay monitor that would have
caught a dead edge before it cost more than one ladder step.

---

## 3. Workstreams

### P4-1 — Capital scaling ladder (execution of Phase 3 A6)
Fixed, pre-committed steps (e.g. 1 → 3 → 10 → 25 contracts / corresponding
bankroll). Each step:
- Entry requires a sustained-green window at the prior step.
- Has an **armed auto-rollback**: drawdown breach or calibration drift
  demotes one step automatically, no discretion.
- No step-skipping, no discretionary size changes. **GATE D** per step.

### P4-2 — Production SRE
- 24/7 supervised process (Phase 2 supervisor) on a stable host, not a laptop.
- Out-of-band alerting (Phase 3 X1) with an on-call owner and ack discipline.
- DB backup + restore drill; log retention; clock-sync monitoring (settlement
  is timestamp-sensitive).
- Deploy procedure: code changes go through tests + a backtest GATE-A re-check
  before touching the live process.

### P4-3 — Automated risk governance
Promote Phase 3 A4 from "exists" to "authoritative":
- Hard daily-loss and max-drawdown halts that flatten and disable trading.
- Tail-short exposure cap and vol-regime kill switch enforced server-side of
  the strategy (cannot be bypassed by a signal).
- A single documented manual kill switch; tested monthly.

### P4-4 — Performance attribution & alpha-decay monitor
- Decompose realized PnL: edge vs. fees vs. slippage vs. basis.
- Rolling live Brier / log loss vs. the Phase 2 backtest baseline; **alarm and
  auto-derisk** when rolling calibration degrades past a threshold for N days.
- Edge-decile monotonicity recomputed weekly on live fills (the W4.4 check, in
  production).

### P4-5 — Periodic re-validation
- Monthly: re-run the full backtest GATE-A on the latest data; the edge must
  still clear the bar. A miss freezes scaling (no new step) pending review.
- Quarterly: re-examine the resolution-source basis study (Phase 3 A1) — basis
  regimes change.

### P4-6 — Records & compliance hygiene
- Immutable trade/PnL ledger export.
- Tax-lot / realized-PnL accounting export.
- Document the trading entity, account, and limits; keep within Kalshi
  position/rate limits with margin.

---

## 4. Decision Gates

| Gate | Bar | On fail |
|---|---|---|
| **GATE D** (per scale step) | sustained-green window at current step + rollback armed + monthly re-validation passing | hold or roll back one step |
| **Alpha-decay halt** | rolling live calibration within threshold of backtest baseline | auto-derisk to step 1; freeze ladder; review |
| **Drawdown halt** | intraday/daily loss < hard limit | flatten, disable trading, page on-call |
| **Re-validation freeze** | monthly backtest GATE-A still passes | no new step until resolved |

Scaling is **monotone-with-rollback**: you can only move up one tested step at a
time, and any halt can move you down immediately.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Alpha decays silently (market gets efficient) | P4-4 rolling calibration alarm + P4-5 monthly re-validation freeze |
| Regime shift (vol explosion, BTC structural break) | P4-3 vol-regime kill switch + drawdown halt |
| Operational outage during a live position | P4-2 supervised host + alerting + manual kill switch |
| Over-scaling on a lucky window | pre-committed ladder, GATE D, no discretionary jumps |
| Resolution-source basis drift | P4-5 quarterly basis re-study |
| Capacity / market-impact at larger size | ladder caps; monitor fill slippage vs. size in P4-4 |

---

## 6. Out of Scope

- New strategies or asset classes (those re-enter at Phase 3 GATE C).
- Discretionary/manual trading overrides.
- Leverage or external capital.

---

## 7. Task Checklist

- [ ] Confirm Phase 3 GATE C held (2 weeks green realized) before any P4 work
- [ ] P4-2 production host + supervised deploy + backup drill
- [ ] P4-3 authoritative risk halts, manual kill switch tested
- [ ] P4-4 attribution + alpha-decay alarm live
- [ ] P4-1 ladder step 1→next behind **GATE D**
- [ ] P4-5 monthly backtest re-validation + quarterly basis re-study scheduled
- [ ] P4-6 ledger / tax / limits records in place
