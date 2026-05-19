# Phase 1 Plan — Build & Discover _(retrospective)_

_Status: ✅ Complete · Outcome: negative result · Superseded by Phase 2_

This is the historical record of Phase 1, written retrospectively so the phase
set is complete. Phase 1 was executed before the plan-doc discipline existed;
its post-mortem is what motivated Phase 2.

---

## 1. Objective (as it stood)

Build **Strategy 2** end-to-end and find out, empirically, whether it has edge:

> Price Kalshi BTC binary markets with a Black–Scholes digital model driven by
> Coinbase **realized volatility**; trade when our probability disagrees with
> Kalshi's price by more than fees + a threshold.

Success criterion for the phase was deliberately weak — *not* "is it
profitable" but "**is there any measurable signal worth refining?**"

---

## 2. What was built

- **Pricing:** `src/pricing/` — BS digital pricer (`pricer.py`), realized-vol
  estimator (`volatility.py`), Kalshi ticker parser (`ticker.py`).
- **Strategy:** `src/strategy/edge.py` — edge = our_prob − market price,
  gated by threshold/horizon/spread; `kelly.py` sizing.
- **Data:** `src/data/` — Coinbase WS + REST feed, Kalshi WS feed.
- **Execution/risk:** `src/execution/paper_executor.py`, `src/risk/`.
- **Measurement:** `src/measurement/` — calibration metrics + settlement
  reporter.
- **Runtime:** `src/runtime/main.py`; **dashboard:** `dashboard/app.py`
  (Bloomberg-style terminal); **backtest:** `src/backtest/engine.py`.

Paper-trading only. Several integration bugs were found and fixed live during
Phase 1 (decimal-strike ticker regex, candle-persistence gated on signals,
`max_horizon_seconds` rejecting daily markets, Kalshi WS silent without an
explicit `market_tickers` subscription, Kalshi-quiet stalling the whole loop).

---

## 3. Outcome — the decisive post-mortem

Backtest over **43,635 settled predictions**:

| Metric | Value | Reading |
|---|---|---|
| Brier | 0.0788 | Misleading — 97% of predictions are trivial 0/1 extremes |
| Log loss | 1.2801 | **Worse than the 0.693 coin-flip baseline** |
| `0.00–0.10` bin | predicted 0.001 → empirical **0.096** | Pricer ~96× overconfident on tails |
| `0.50–0.70` bins | predicted 0.55/0.65 → empirical 0.44/0.49 | Overconfident in the body too |

**Root cause:** a fixed 60-minute realized-vol window was used to price
contracts resolving **10–34 hours** out. Calm short windows understate the vol
BTC actually realizes over a day, so the model called tail outcomes
"impossible" when they occurred ~10% of the time. The dashboard's apparent
"edge" and S-curve were **measurement artifacts of vol underestimation, not
alpha.**

**Secondary findings (latent, blocking trust in any metric):**

1. `reporter.py` only scanned the most-recent estimates (all still-open) →
   live calibration silently never populated.
2. `parse_ticker` used a `now`-relative close time → historical markets
   settled at the wrong timestamp.
3. Data continuity: ~32% wall-clock candle coverage (134 gaps > 5 min) from
   crashes/restarts → ~50% of estimates unsettleable.

---

## 4. Lessons → why Phase 2 exists

- The negative result is **informative, not wasted**: it falsified the naive
  realized-vol premise and produced a precise, testable root cause.
- We could not trust *any* number until the settlement/calibration bugs and the
  data-continuity problem were fixed — hence Phase 2 WS2/WS3 before WS1.
- The fix hypothesis (horizon-matched volatility) and a hard pass/fail gate
  (GATE A) became Phase 2's core. Phase 1 deliberately had no such gate; that
  gap is corrected from Phase 2 onward.

---

## 5. Carried forward into Phase 2

- Backtest harness (`src/backtest/engine.py`) — reused as the GATE A instrument.
- The 43,635-sample calibration result — the **baseline** every Phase 2 vol
  mode is measured against.
- The bug list above — became Phase 2 WS2 (T01–T03) and WS3 (T04–T08).
