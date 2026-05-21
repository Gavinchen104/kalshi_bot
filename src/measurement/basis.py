"""
Phase 3 / Track A / A1 — Resolution-Source Basis Study.

Kalshi BTC contracts settle off Kalshi's own price source (not Coinbase). We
price off Coinbase spot. The risk is that near the strike, the disagreement
between our feed and Kalshi's settlement source can flip the outcome — so a
near-the-money "edge" can be an artifact of feed mismatch, not real alpha.

We do not have direct access to Kalshi's settlement source, so this module
does not measure that basis directly. Instead it measures the two things
we *can* measure that bound the risk:

  1. **Settlement-proximity density** — for each settled contract, the
     distance |settle_price − strike|. Reveals how many contracts settled in
     a tight band around the strike, where a small basis would flip them.

  2. **Intra-Coinbase tick jitter at close time** — variance of 1-min closes
     in the ±N-minute window around each settlement candle. A lower bound
     on what plausible Coinbase-vs-Kalshi-source basis could be: if even
     Coinbase itself jitters by $X within 5 minutes of settle, the cross-
     feed basis is at least that large in expectation.

From these we derive a **recommended near-strike guard band** that the live
strategy uses to suppress edge signals when |spot − strike| is below the
band. Default is conservative: guard = max(intra-feed jitter p95, 2× the
p10 of settlement-proximity distances).
"""
from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from src.config import load_settings
from src.pricing.ticker import parse_ticker


_EPOCH_NOW = None  # parse_ticker(settlement_mode=True) ignores `now`
_SETTLE_TOL_MS = 2 * 60_000
_JITTER_WINDOW_MIN = 5  # ±5 1-min closes around the settlement candle


@dataclass
class BasisReport:
    n_settled: int
    proximity_usd: np.ndarray      # |settle_price − strike| per settled contract
    jitter_std_usd: np.ndarray     # std of 1-min closes in ±W window around each settle candle
    proximity_pct: dict            # {"p10","p25","p50","p75","p90"} of proximity
    jitter_pct: dict               # {"p50","p90","p95","p99"} of jitter std
    recommended_guard_usd: float


def _percentiles(arr: np.ndarray, qs: list[float]) -> dict:
    if arr.size == 0:
        return {f"p{int(q*100)}": None for q in qs}
    out = np.percentile(arr, [q * 100 for q in qs])
    return {f"p{int(q*100)}": float(v) for q, v in zip(qs, out)}


def _nearest_candle_idx(close_ms: int, cand_ts: list[int]) -> int | None:
    if not cand_ts:
        return None
    i = bisect_left(cand_ts, close_ms)
    best, diff = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(cand_ts):
            d = abs(cand_ts[j] - close_ms)
            if diff is None or d < diff:
                best, diff = j, d
    if best is None or diff > _SETTLE_TOL_MS:
        return None
    return best


def compute_basis_report(db_path: str, settings) -> BasisReport:
    """Walk all settleable contracts; for each, record
    (|settle_price - strike|, intra-Coinbase jitter near close)."""
    bw = settings.pricer.bracket_width_usd_default
    conn = sqlite3.connect(db_path)
    try:
        # Distinct settled contracts (one row per market_id is enough — we're
        # measuring contract-level outcomes, not per-estimate calibration).
        markets = [r[0] for r in conn.execute(
            "SELECT DISTINCT market_id FROM prob_estimate"
        ).fetchall()]
        cand_rows = conn.execute(
            "SELECT timestamp_ms, close FROM coinbase_candle ORDER BY timestamp_ms ASC"
        ).fetchall()
    finally:
        conn.close()

    cand_ts = [int(r[0]) for r in cand_rows]
    cand_close = np.asarray([float(r[1]) for r in cand_rows], dtype=float)

    prox: list[float] = []
    jit: list[float] = []
    for mid in markets:
        terms = parse_ticker(mid, settlement_mode=True, bracket_width_usd=bw)
        if terms is None:
            continue
        # Reference price the contract is decided against on the Coinbase side.
        if terms.direction == "above":
            if terms.strike_usd is None:
                continue
            ref = terms.strike_usd
        else:
            if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
                continue
            ref = (terms.bracket_low_usd + terms.bracket_high_usd) / 2.0

        close_ms = int(terms.close_time.timestamp() * 1000)
        idx = _nearest_candle_idx(close_ms, cand_ts)
        if idx is None:
            continue
        settle_px = cand_close[idx]
        prox.append(abs(settle_px - ref))

        lo = max(0, idx - _JITTER_WINDOW_MIN)
        hi = min(len(cand_ts), idx + _JITTER_WINDOW_MIN + 1)
        window = cand_close[lo:hi]
        if window.size >= 3:
            jit.append(float(np.std(window, ddof=1)))

    proximity = np.asarray(prox, dtype=float)
    jitter = np.asarray(jit, dtype=float)
    prox_pct = _percentiles(proximity, [0.10, 0.25, 0.50, 0.75, 0.90])
    jit_pct = _percentiles(jitter, [0.50, 0.90, 0.95, 0.99])

    # Band = jitter_p95 * safety_multiple, capped at proximity_p50.
    #
    # The basis magnitude we're guarding against is approximated by intra-feed
    # jitter (~tens of USD); proximity measures *exposure* (how rarely contracts
    # are near-the-money), NOT basis size, so it acts only as a cap — never set
    # the band so wide it would suppress more than ~half of contracts.
    SAFETY_MULTIPLE = 2.0
    band = 0.0
    if jit_pct["p95"] is not None:
        band = jit_pct["p95"] * SAFETY_MULTIPLE
    if prox_pct["p50"] is not None:
        band = min(band, prox_pct["p50"])  # cap: don't suppress > half of contracts
    return BasisReport(
        n_settled=len(prox), proximity_usd=proximity, jitter_std_usd=jitter,
        proximity_pct=prox_pct, jitter_pct=jit_pct,
        recommended_guard_usd=float(band),
    )


def print_basis_report(r: BasisReport) -> None:
    print("=" * 72)
    print("PHASE 3 / A1 — RESOLUTION-SOURCE BASIS STUDY")
    print("=" * 72)
    print(f"\n  settled contracts measured: {r.n_settled:,}")
    if r.n_settled == 0:
        print("  (no settled contracts; cannot estimate basis or guard band)")
        print("=" * 72)
        return

    print("\n── SETTLEMENT-PROXIMITY |settle − strike| (USD) ────────────────")
    print(f"  How tightly contracts settle around the strike. Tight = more")
    print(f"  contracts a small Kalshi-vs-Coinbase basis could flip.")
    p = r.proximity_pct
    print(f"    p10 ${p['p10']:>8.2f}   p25 ${p['p25']:>8.2f}   p50 ${p['p50']:>8.2f}"
          f"   p75 ${p['p75']:>8.2f}   p90 ${p['p90']:>8.2f}")

    print("\n── INTRA-COINBASE CLOSE-TIME JITTER (USD, 1-min closes, ±5min) ─")
    print(f"  Lower bound on plausible cross-feed basis: even within Coinbase,")
    print(f"  the close price varies by this much in the 11-min window.")
    j = r.jitter_pct
    j50 = f"${j['p50']:>7.2f}" if j['p50'] is not None else "      —"
    j90 = f"${j['p90']:>7.2f}" if j['p90'] is not None else "      —"
    j95 = f"${j['p95']:>7.2f}" if j['p95'] is not None else "      —"
    j99 = f"${j['p99']:>7.2f}" if j['p99'] is not None else "      —"
    print(f"    p50 {j50}   p90 {j90}   p95 {j95}   p99 {j99}")

    print("\n── RECOMMENDED NEAR-STRIKE GUARD BAND ──────────────────────────")
    print(f"  band = jitter_p95 × 2, capped at proximity_p50")
    print(f"       = ${r.recommended_guard_usd:.2f}")
    print(f"\n  Wire this into settings.yaml as:")
    print(f"     strategy:")
    print(f"       near_strike_guard_usd: {r.recommended_guard_usd:.2f}")
    print(f"\n  EdgeStrategy will then suppress signals where |spot − strike|")
    print(f"  is within this band, removing the contracts most exposed to")
    print(f"  cross-feed basis risk at the moneyness boundary.")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 A1 — basis-proxy study")
    ap.add_argument("--db", default=None, help="SQLite DB path (default: settings)")
    args = ap.parse_args()
    settings = load_settings()
    db_path = args.db or settings.storage.db_path
    r = compute_basis_report(db_path, settings)
    print_basis_report(r)


if __name__ == "__main__":
    main()
