"""
Offline backtest: replay stored prob_estimates against realized BTC outcomes.

This does NOT re-run the pricer — it replays exactly the probabilities the live
bot computed and stored, so it answers the real question: "did the signals the
bot actually generated make money, and is the predicted edge correlated with
realized PnL?"

Pipeline per stored estimate (chronological):
  1. Re-derive the EdgeStrategy decision (edge_threshold + horizon filters).
     The top-of-book depth filter is skipped (size isn't persisted in
     prob_estimate); it only affects execution realism, not edge existence.
  2. Settle the contract against the Coinbase candle nearest its true close
     time (ticker is parsed with an early `now` so close_time == the real
     historical anchor, working around the now-relative bug in parse_ticker).
  3. Model the fill price + fee identically to PaperExecutor (slippage + fee_bps).
  4. Hold to expiry; realize PnL at settlement.

Trading assumption: at most ONE entry per unique contract (first time it
signals), held to settlement. The live bot is rate-limited and position-capped,
so replaying every repeated estimate as a new trade would massively over-count.

Calibration (Brier / log loss / reliability bins) is computed over ALL
settleable estimates, independent of whether we traded them — that's the clean
measure of pricer quality.
"""
from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import load_settings
from src.execution.slippage import apply_slippage, fee_for
from src.measurement.calibration import compute
from src.pricing.pricer import CoinbasePricer
from src.pricing.ticker import parse_ticker


# An anchor far in the past so parse_ticker's `anchor >= now` is always true,
# yielding the contract's real historical close_time instead of a now-relative
# fallback.
_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
_SETTLE_TOLERANCE_MS = 2 * 60_000


@dataclass
class Trade:
    market_id: str
    side: str            # "yes" | "no"
    predicted_prob: float
    predicted_edge: float
    fill_price_cents: int
    quantity: int
    fee_cents: int
    outcome_yes: int     # 1 if BTC settled YES, else 0
    pnl_cents: int
    horizon_seconds: float


def _load_estimates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT market_id, prob, horizon_seconds, spot_usd, vol_annualized,
               computed_at, market_yes_ask_cents, market_yes_bid_cents,
               market_mid_cents
        FROM prob_estimate
        ORDER BY id ASC
        """
    ).fetchall()
    keys = ("market_id", "prob", "horizon_seconds", "spot_usd", "vol_annualized",
            "computed_at", "ask", "bid", "mid")
    return [dict(zip(keys, r)) for r in rows]


def _load_candles(conn: sqlite3.Connection) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT timestamp_ms, close FROM coinbase_candle ORDER BY timestamp_ms ASC"
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    closes = [float(r[1]) for r in rows]
    return ts, closes


def _settle_price(close_ms: int, cand_ts: list[int], cand_close: list[float]) -> float | None:
    """Return the Coinbase close nearest to close_ms, or None if no candle is
    within the tolerance window."""
    if not cand_ts:
        return None
    i = bisect_left(cand_ts, close_ms)
    best_idx, best_diff = None, None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(cand_ts):
            d = abs(cand_ts[j] - close_ms)
            if best_diff is None or d < best_diff:
                best_idx, best_diff = j, d
    if best_idx is None or best_diff > _SETTLE_TOLERANCE_MS:
        return None
    return cand_close[best_idx]


def _parse_iso_ms(s: str) -> int | None:
    """Parse a stored ISO-8601 timestamp to unix-ms. SQLite stores either
    'YYYY-MM-DDTHH:MM:SS.ffffff+00:00' (from .isoformat()) or '… ' with a
    space separator. Returns None on parse failure."""
    try:
        dt = datetime.fromisoformat(s.replace(" ", "T"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_pricer(settings, vol_mode_override: str | None = None) -> CoinbasePricer:
    """Construct a CoinbasePricer from settings, optionally overriding vol_mode."""
    return CoinbasePricer(
        vol_window_minutes=settings.pricer.vol_window_minutes,
        vol_floor=settings.pricer.vol_floor_annualized,
        vol_ceiling=settings.pricer.vol_ceiling_annualized,
        min_horizon_seconds=settings.pricer.min_horizon_seconds,
        bracket_width_usd_default=settings.pricer.bracket_width_usd_default,
        vol_mode=vol_mode_override or settings.pricer.vol_mode,
        vol_window_floor_min=settings.pricer.vol_window_floor_min,
        vol_window_cap_min=settings.pricer.vol_window_cap_min,
        vol_long_floor_days=settings.pricer.vol_long_floor_days,
        ewma_half_life_min=settings.pricer.ewma_half_life_min,
    )


def _outcome(terms, settle_price: float) -> int | None:
    if terms.direction == "above":
        if terms.strike_usd is None:
            return None
        return 1 if settle_price >= terms.strike_usd else 0
    # bracket
    if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
        return None
    return 1 if (terms.bracket_low_usd <= settle_price < terms.bracket_high_usd) else 0


def run_backtest(
    db_path: str,
    settings,
    *,
    repricer_mode: str = "replay",
    vol_mode: str | None = None,
    execution_mode: str = "one-entry",
    calibrator_artifact_path: str | None = None,
    since_iso: str | None = None,
) -> dict:
    """Replay the strategy over historical estimates.

    Parameters:
      repricer_mode: "replay" uses stored probs (Phase-1 behavior); "reprice"
        recomputes each prob from the candle history available at the
        estimate's `computed_at`, using `vol_mode` (defaults to
        settings.pricer.vol_mode). This is how we A/B the vol modes
        introduced in B1/B2 against the same data.
      execution_mode: "one-entry" caps to one trade per unique market (the
        Phase-1 simplification); "faithful" honors the live rate-limit,
        per-market position cap, and gross-exposure cap, so trade-level PnL
        is comparable to a live run.
      calibrator_artifact_path: if set, load an IsotonicCalibrator JSON and
        apply `cal.predict_one()` to every raw prob before calibration metrics
        and the strategy decision. Mirrors the live EdgeStrategy path. Used
        by GATE B forward validation.
      since_iso: skip estimates with computed_at < since_iso. Used by GATE B
        to evaluate ONLY the forward window after the calibrator was deployed.
    """
    if repricer_mode not in ("replay", "reprice"):
        raise ValueError(f"unknown repricer_mode={repricer_mode!r}")
    if execution_mode not in ("one-entry", "faithful"):
        raise ValueError(f"unknown execution_mode={execution_mode!r}")

    # Optional calibrator: applied to every raw prob just like live EdgeStrategy.
    calibrator = None
    if calibrator_artifact_path and Path(calibrator_artifact_path).exists():
        from src.strategy.calibrator import IsotonicCalibrator as _IsoCal
        calibrator = _IsoCal.load(calibrator_artifact_path)
    since_ms = _parse_iso_ms(since_iso) if since_iso else None

    edge_threshold = settings.strategy.edge_threshold
    min_h = settings.strategy.min_horizon_seconds
    max_h = settings.strategy.max_horizon_seconds
    max_spread = settings.strategy.max_spread_cents
    bracket_w = settings.pricer.bracket_width_usd_default
    fee_bps = settings.execution.fee_bps
    slip_bps = settings.execution.slippage_bps
    bankroll = settings.sizing.bankroll_cents
    kelly_frac = settings.sizing.kelly_fraction
    max_ctr = settings.sizing.max_contracts_per_trade

    # Faithful-execution caps (only consulted when execution_mode == "faithful").
    rate_limit_ms = settings.execution.min_order_interval_ms
    pos_cap = settings.risk.max_position_per_market
    gross_cap = settings.risk.max_gross_exposure

    from src.strategy.kelly import kelly_contracts

    conn = sqlite3.connect(db_path)
    try:
        estimates = _load_estimates(conn)
        cand_ts, cand_close = _load_candles(conn)
    finally:
        conn.close()

    # Pre-convert closes to a numpy array so re-price can take cheap views.
    closes_arr = np.asarray(cand_close, dtype=float) if cand_close else np.empty(0)

    # Re-price prerequisites
    pricer = _build_pricer(settings, vol_mode) if repricer_mode == "reprice" else None
    vol_window_floor = settings.pricer.vol_window_floor_min
    n_reprice_skipped = 0  # estimates we couldn't re-price (insufficient history etc.)

    calib_pairs: list[tuple[float, int]] = []
    trades: list[Trade] = []
    traded_markets: set[str] = set()
    # Faithful-execution state
    positions_signed: dict[str, int] = defaultdict(int)
    gross_abs: int = 0
    last_order_ms: int = 0
    n_rate_limited = 0
    n_cap_blocked = 0

    n_total = len(estimates)
    n_unparsable = 0
    n_unsettleable = 0

    n_before_since = 0
    for e in estimates:
        # GATE B forward window: skip estimates from before the calibrator went live.
        if since_ms is not None:
            ts = _parse_iso_ms(e["computed_at"])
            if ts is not None and ts < since_ms:
                n_before_since += 1
                continue
        terms = parse_ticker(e["market_id"], now=_EPOCH, bracket_width_usd=bracket_w)
        if terms is None:
            n_unparsable += 1
            continue
        close_ms = int(terms.close_time.timestamp() * 1000)
        settle_px = _settle_price(close_ms, cand_ts, cand_close)
        if settle_px is None:
            n_unsettleable += 1
            continue
        outcome = _outcome(terms, settle_px)
        if outcome is None:
            n_unsettleable += 1
            continue

        # ── Re-price (B3 / T16): recompute prob with the candle history
        # available at this estimate's computed_at, under the chosen vol_mode.
        # Replay mode just uses the stored prob (Phase-1 behavior).
        #
        # Critical: Phase-1 stored estimates were priced against parse_ticker's
        # now-relative close_time, so settlement_mode `terms.close_time` is often
        # in the past relative to `computed_at`. To reprice faithfully we use the
        # bot's live perspective: close_time_eff = computed_at + stored
        # horizon_seconds, with the ticker's strike/bracket. Outcome settlement
        # still uses the ticker anchor — that's what actually happened.
        if repricer_mode == "reprice":
            now_ms = _parse_iso_ms(e["computed_at"])
            if now_ms is None:
                n_reprice_skipped += 1
                continue
            # Closes strictly before computed_at (no peeking at future candles).
            i = bisect_right(cand_ts, now_ms)
            if i < vol_window_floor + 1:
                n_reprice_skipped += 1
                continue
            closes_view = closes_arr[:i]
            now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
            from dataclasses import replace as _dc_replace
            from datetime import timedelta as _td
            effective_close = now_dt + _td(seconds=float(e["horizon_seconds"]))
            terms_for_reprice = _dc_replace(terms, close_time=effective_close)
            est_re = pricer.price(
                market_id=e["market_id"],
                spot_usd=float(e["spot_usd"]),
                closes_1m=closes_view,
                now=now_dt,
                terms_override=terms_for_reprice,
            )
            if est_re is None:
                n_reprice_skipped += 1
                continue
            prob = float(est_re.prob)
        else:
            prob = float(e["prob"])
        # Optional calibration layer: apply BEFORE both calibration metrics and
        # the strategy decision so the report reflects the deployed pipeline.
        if calibrator is not None:
            prob = float(calibrator.predict_one(prob))
        calib_pairs.append((prob, outcome))

        # ── Strategy decision (mirror EdgeStrategy, minus depth filter) ──
        bid, ask = e["bid"], e["ask"]
        if bid is None or ask is None:
            continue
        if (ask - bid) > max_spread:
            continue
        h = e["horizon_seconds"]
        if h < min_h or h > max_h:
            continue

        yes_edge = prob - ask / 100.0
        no_edge = bid / 100.0 - prob
        if yes_edge >= edge_threshold:
            side = "yes"
            edge = yes_edge
            intended = ask
        elif no_edge >= edge_threshold:
            side = "no"
            edge = no_edge
            intended = 100 - bid
        else:
            continue

        kelly_prob = prob if side == "yes" else (1.0 - prob)
        qty = kelly_contracts(
            our_prob=kelly_prob,
            price_cents=intended,
            bankroll_cents=bankroll,
            kelly_fraction=kelly_frac,
            max_contracts=max_ctr,
        )
        if qty <= 0:
            continue

        if execution_mode == "one-entry":
            # Phase-1 simplification: one trade per unique market, held to expiry.
            if e["market_id"] in traded_markets:
                continue
            traded_markets.add(e["market_id"])
        else:  # "faithful": honor rate-limit + per-market + gross caps
            now_ms = _parse_iso_ms(e["computed_at"]) or 0
            if now_ms - last_order_ms < rate_limit_ms:
                n_rate_limited += 1
                continue
            cur = positions_signed[e["market_id"]]
            signed_delta = qty if side == "yes" else -qty
            new_pos = cur + signed_delta
            new_gross = gross_abs - abs(cur) + abs(new_pos)
            if abs(new_pos) > pos_cap or new_gross > gross_cap:
                n_cap_blocked += 1
                continue
            positions_signed[e["market_id"]] = new_pos
            gross_abs = new_gross
            last_order_ms = now_ms

        fill_price = apply_slippage(intended, side, slip_bps)
        fee = fee_for(qty, fill_price, fee_bps)

        # Settlement PnL. A "yes" position pays 100 on outcome_yes=1.
        # A "no" position pays 100 on outcome_yes=0.
        won = (outcome == 1) if side == "yes" else (outcome == 0)
        if won:
            pnl = qty * (100 - fill_price) - fee
        else:
            pnl = -qty * fill_price - fee

        trades.append(Trade(
            market_id=e["market_id"], side=side,
            predicted_prob=prob, predicted_edge=edge,
            fill_price_cents=fill_price, quantity=qty, fee_cents=fee,
            outcome_yes=outcome, pnl_cents=pnl, horizon_seconds=h,
        ))

    report = compute(calib_pairs, n_bins=10)

    return {
        "n_total_estimates": n_total,
        "n_unparsable": n_unparsable,
        "n_unsettleable": n_unsettleable,
        "n_settleable": len(calib_pairs),
        "calibration": report,
        "trades": trades,
        "repricer_mode": repricer_mode,
        "vol_mode": (vol_mode or settings.pricer.vol_mode) if repricer_mode == "reprice" else "replay",
        "execution_mode": execution_mode,
        "n_reprice_skipped": n_reprice_skipped,
        "n_rate_limited": n_rate_limited,
        "n_before_since": n_before_since,
        "calibrator_loaded": calibrator is not None,
        "since_iso": since_iso,
        "n_cap_blocked": n_cap_blocked,
    }


def _decile_table(trades: list[Trade]) -> list[dict]:
    """Bucket trades by predicted edge into deciles; show realized avg PnL/contract.
    If the edge is real, higher edge deciles should show higher realized PnL."""
    if not trades:
        return []
    ordered = sorted(trades, key=lambda t: t.predicted_edge)
    n = len(ordered)
    out = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10 if d < 9 else n
        bucket = ordered[lo:hi]
        if not bucket:
            continue
        tot_pnl = sum(t.pnl_cents for t in bucket)
        tot_ctr = sum(t.quantity for t in bucket)
        wins = sum(1 for t in bucket
                   if (t.outcome_yes == 1) == (t.side == "yes"))
        out.append({
            "decile": d + 1,
            "edge_lo": bucket[0].predicted_edge,
            "edge_hi": bucket[-1].predicted_edge,
            "n": len(bucket),
            "win_rate": wins / len(bucket),
            "pnl_cents": tot_pnl,
            "avg_pnl_per_contract": (tot_pnl / tot_ctr) if tot_ctr else 0.0,
        })
    return out


def _monotonicity_check(deciles: list[dict]) -> dict:
    """Test whether avg PnL/contract rises across edge deciles — the
    discriminating sign that the predicted edge is *real* and not noise.

    Returns: {"rising_steps": int (out of 9), "spread_cents": float,
              "passes": bool, "note": str}.
    Pass criteria: rising_steps >= 7 (out of 9 adjacent comparisons)
                   AND spread_cents (top decile vs bottom) >= 2.0.
    Requires at least 5 populated deciles to be meaningful.
    """
    pops = [d for d in deciles if d["n"] > 0]
    if len(pops) < 5:
        return {"rising_steps": 0, "spread_cents": 0.0, "passes": False,
                "note": f"insufficient populated deciles ({len(pops)} < 5)"}
    pnl = [d["avg_pnl_per_contract"] for d in pops]
    rising = sum(1 for a, b in zip(pnl, pnl[1:]) if b > a)
    spread = pnl[-1] - pnl[0]
    passes = rising >= 7 and spread >= 2.0
    return {
        "rising_steps": rising, "spread_cents": float(spread),
        "passes": passes,
        "note": f"rising {rising}/{len(pops)-1} · spread {spread:+.2f}c",
    }


def _gate_a_check(res: dict) -> dict:
    """Evaluate Phase 2 GATE A criteria against a backtest result dict.

    Bar (from PHASE2_PLAN.md §2):
      - 0.00-0.10 predicted bin: empirical YES < 0.02 (vs 0.096 baseline)
      - log loss < 0.69 (beats coin flip; vs 1.28 baseline)
      - no reliability bin off by > 0.07 abs (mean_pred vs emp_freq)
      - settleable n >= 20,000
    """
    rep = res.get("calibration")
    settleable = res.get("n_settleable", 0)
    crit: dict = {
        "settleable_n_ok": settleable >= 20_000,
        "settleable_n": settleable,
    }
    if rep is None or rep.n_samples == 0:
        crit.update({
            "log_loss_ok": False, "log_loss": None,
            "tail_bin_ok": False, "tail_emp": None,
            "max_bin_gap_ok": False, "max_bin_gap": None,
            "passes": False,
        })
        return crit
    crit["log_loss"] = rep.log_loss
    crit["log_loss_ok"] = rep.log_loss is not None and rep.log_loss < 0.69
    tail_bin = next((b for b in rep.bins if b["lo"] == 0.0 and b["hi"] == 0.1), None)
    crit["tail_emp"] = tail_bin["emp_freq"] if tail_bin else None
    crit["tail_bin_ok"] = (
        tail_bin is not None and tail_bin["emp_freq"] is not None
        and tail_bin["emp_freq"] < 0.02
    )
    gaps = [
        abs(b["emp_freq"] - b["mean_pred"])
        for b in rep.bins
        if b["n"] > 0 and b["mean_pred"] is not None and b["emp_freq"] is not None
    ]
    crit["max_bin_gap"] = max(gaps) if gaps else None
    crit["max_bin_gap_ok"] = bool(gaps) and max(gaps) <= 0.07
    crit["passes"] = (
        crit["settleable_n_ok"] and crit["log_loss_ok"]
        and crit["tail_bin_ok"] and crit["max_bin_gap_ok"]
    )
    return crit


def _group_pnl(trades: list[Trade], keyfn) -> dict:
    agg: dict = defaultdict(lambda: {"n": 0, "pnl": 0, "wins": 0})
    for t in trades:
        k = keyfn(t)
        agg[k]["n"] += 1
        agg[k]["pnl"] += t.pnl_cents
        if (t.outcome_yes == 1) == (t.side == "yes"):
            agg[k]["wins"] += 1
    return agg


def print_report(res: dict) -> None:
    rep = res["calibration"]
    trades: list[Trade] = res["trades"]

    print("=" * 72)
    print("BACKTEST REPORT — Strategy 2 (BS-from-realized-vol vs Kalshi)")
    print(
        f"  mode: repricer={res.get('repricer_mode','replay')} "
        f"vol_mode={res.get('vol_mode','replay')} "
        f"execution={res.get('execution_mode','one-entry')}"
    )
    print("=" * 72)

    print("\n── DATA COVERAGE ───────────────────────────────────────────────")
    print(f"  total prob_estimates : {res['n_total_estimates']:,}")
    print(f"  unparsable tickers   : {res['n_unparsable']:,}")
    print(f"  unsettleable         : {res['n_unsettleable']:,} "
          f"(no candle within ±2min of close)")
    print(f"  settleable           : {res['n_settleable']:,} "
          f"({100*res['n_settleable']/max(1,res['n_total_estimates']):.1f}%)")
    if res.get("repricer_mode") == "reprice":
        print(f"  reprice skipped      : {res.get('n_reprice_skipped',0):,} "
              f"(insufficient candle history at estimate time)")
    if res.get("execution_mode") == "faithful":
        print(f"  rate-limited         : {res.get('n_rate_limited',0):,}")
        print(f"  position/exposure cap blocks: {res.get('n_cap_blocked',0):,}")

    print("\n── PRICER CALIBRATION (all settleable estimates) ───────────────")
    if rep.n_samples == 0:
        print("  no settleable estimates — cannot compute calibration")
    else:
        print(f"  samples   : {rep.n_samples:,}")
        print(f"  Brier     : {rep.brier:.4f}   "
              f"(0.25 = coin flip · <0.20 good · >0.25 worse than random)")
        print(f"  Log loss  : {rep.log_loss:.4f}   (0.693 = coin flip)")
        print("\n  Reliability — predicted vs empirical YES frequency:")
        print(f"  {'bin':>10} {'n':>7} {'mean_pred':>10} {'emp_freq':>10}  flag")
        for b in rep.bins:
            if b["n"] == 0:
                continue
            mp, ef = b["mean_pred"], b["emp_freq"]
            flag = ""
            if mp is not None and ef is not None:
                gap = ef - mp
                if abs(gap) > 0.10:
                    flag = "<< miscalibrated" if gap > 0 else ">> overconfident"
            print(f"  {b['lo']:.2f}-{b['hi']:.2f} {b['n']:>7,} "
                  f"{(mp if mp is not None else 0):>10.3f} "
                  f"{(ef if ef is not None else 0):>10.3f}  {flag}")

    print("\n── STRATEGY BACKTEST (1 entry/contract, held to expiry) ────────")
    if not trades:
        print("  no trades taken")
        return
    n = len(trades)
    gross = sum(t.pnl_cents + t.fee_cents for t in trades)
    fees = sum(t.fee_cents for t in trades)
    net = sum(t.pnl_cents for t in trades)
    wins = sum(1 for t in trades if (t.outcome_yes == 1) == (t.side == "yes"))
    invested = sum(t.quantity * t.fill_price_cents for t in trades)
    print(f"  trades        : {n:,}")
    print(f"  win rate      : {100*wins/n:.1f}%  ({wins}/{n})")
    print(f"  capital used  : ${invested/100:,.2f} (sum of entry notional)")
    print(f"  gross PnL     : ${gross/100:,.2f}")
    print(f"  fees          : ${fees/100:,.2f}")
    print(f"  NET PnL       : ${net/100:,.2f}")
    print(f"  ROI on cap    : {100*net/max(1,invested):.2f}%")
    print(f"  avg PnL/trade : {net/n:.1f}c")

    print("\n── EDGE-DECILE ANALYSIS (the 'is the edge real?' test) ─────────")
    print("  If the edge is real, avg PnL/contract should RISE with edge decile.")
    print(f"  {'dec':>3} {'edge_range':>16} {'n':>5} {'win%':>6} "
          f"{'net_pnl$':>10} {'pnl/ctr(c)':>11}")
    for r in _decile_table(trades):
        print(f"  {r['decile']:>3} "
              f"{r['edge_lo']:>7.3f}-{r['edge_hi']:<7.3f} "
              f"{r['n']:>5} {100*r['win_rate']:>5.0f}% "
              f"{r['pnl_cents']/100:>10.2f} {r['avg_pnl_per_contract']:>11.2f}")

    print("\n── PnL BY SIDE ─────────────────────────────────────────────────")
    for side, a in sorted(_group_pnl(trades, lambda t: t.side).items()):
        print(f"  {side.upper():>3}  n={a['n']:>4}  win={100*a['wins']/max(1,a['n']):>5.1f}%  "
              f"net=${a['pnl']/100:>9.2f}")

    print("\n── PnL BY HORIZON ──────────────────────────────────────────────")
    def hbucket(t: Trade) -> str:
        hh = t.horizon_seconds / 3600
        if hh < 1:
            return "0-1h"
        if hh < 6:
            return "1-6h"
        if hh < 18:
            return "6-18h"
        return "18h+"
    order = {"0-1h": 0, "1-6h": 1, "6-18h": 2, "18h+": 3}
    for k, a in sorted(_group_pnl(trades, hbucket).items(), key=lambda kv: order.get(kv[0], 9)):
        print(f"  {k:>6}  n={a['n']:>4}  win={100*a['wins']/max(1,a['n']):>5.1f}%  "
              f"net=${a['pnl']/100:>9.2f}")

    # ── Monotonicity check (T18): is realized PnL monotone in predicted edge? ─
    mono = _monotonicity_check(_decile_table(trades))
    print("\n── EDGE-DECILE MONOTONICITY ────────────────────────────────────")
    print(f"  rising_steps={mono['rising_steps']}  spread={mono['spread_cents']:+.2f}c"
          f"  →  {'PASS' if mono['passes'] else 'FAIL'}  ({mono['note']})")

    # ── GATE A check (T18): Phase 2 binding decision ─────────────────────────
    gate = _gate_a_check(res)
    print("\n── GATE A (Phase 2 binding decision) ───────────────────────────")
    print(f"  settleable n     : {gate['settleable_n']:,}     "
          f"{'PASS' if gate['settleable_n_ok'] else 'FAIL'}  (≥20,000)")
    ll = f"{gate['log_loss']:.4f}" if gate['log_loss'] is not None else "—"
    print(f"  log loss         : {ll}    "
          f"{'PASS' if gate['log_loss_ok'] else 'FAIL'}  (<0.69)")
    te = f"{gate['tail_emp']:.4f}" if gate['tail_emp'] is not None else "—"
    print(f"  tail bin emp     : {te}    "
          f"{'PASS' if gate['tail_bin_ok'] else 'FAIL'}  (<0.02)")
    mg = f"{gate['max_bin_gap']:.3f}" if gate['max_bin_gap'] is not None else "—"
    print(f"  max bin gap      : {mg}     "
          f"{'PASS' if gate['max_bin_gap_ok'] else 'FAIL'}  (≤0.07)")
    print(f"  ────────────────────────  GATE A: "
          f"{'✅ PASS' if gate['passes'] else '❌ FAIL'}")
    print("=" * 72)


_GATE_A_MODES = (
    ("replay", "replay", None),
    ("reprice/fixed", "reprice", "fixed"),
    ("reprice/horizon", "reprice", "horizon_scaled"),
    ("reprice/blend", "reprice", "blend"),
    ("reprice/ewma", "reprice", "ewma"),
)


def run_gate_a(db_path: str, settings) -> int:
    """Run the backtest across the replay baseline + each reprice vol_mode and
    print a side-by-side GATE A summary. Returns 0 if ANY mode passes, else 1
    (suitable for CI gating)."""
    print("=" * 78)
    print("GATE A SWEEP — Phase 2 binding decision across vol modes")
    print("=" * 78)

    rows: list[dict] = []
    for label, rep_mode, vol_mode in _GATE_A_MODES:
        res = run_backtest(
            db_path, settings,
            repricer_mode=rep_mode, vol_mode=vol_mode,
            execution_mode="one-entry",
        )
        mono = _monotonicity_check(_decile_table(res["trades"]))
        gate = _gate_a_check(res)
        rows.append({"label": label, "res": res, "mono": mono, "gate": gate})

    # Header
    print(f"\n  {'mode':<18} {'settle_n':>9} {'brier':>7} {'logloss':>8} "
          f"{'tail_emp':>9} {'maxgap':>7} {'mono':>6} {'GATE A':>9}")
    print("  " + "-" * 76)
    for r in rows:
        rep = r["res"]["calibration"]
        ll = f"{rep.log_loss:.3f}" if rep.n_samples else "—"
        br = f"{rep.brier:.3f}" if rep.n_samples else "—"
        te = f"{r['gate']['tail_emp']:.3f}" if r['gate']['tail_emp'] is not None else "—"
        mg = f"{r['gate']['max_bin_gap']:.3f}" if r['gate']['max_bin_gap'] is not None else "—"
        print(f"  {r['label']:<18} {r['res']['n_settleable']:>9,} "
              f"{br:>7} {ll:>8} {te:>9} {mg:>7} "
              f"{'PASS' if r['mono']['passes'] else 'FAIL':>6} "
              f"{'✅ PASS' if r['gate']['passes'] else '❌ FAIL':>9}")

    any_pass = any(r["gate"]["passes"] for r in rows)
    print("\n  ──────────────────────────────────────────────────────────────")
    if any_pass:
        winners = [r["label"] for r in rows if r["gate"]["passes"]]
        print(f"  GATE A: ✅ PASS  ({', '.join(winners)})")
        print(f"  → Proceed to Phase 3 Track A (forward paper validation → GATE B)")
        rc = 0
    else:
        print(f"  GATE A: ❌ FAIL  (no vol mode meets the bar)")
        print(f"  → Proceed to Phase 3 Track B (pivot: P1 Deribit IV / P2 calibration layer)")
        rc = 1
    print("=" * 78)
    return rc


def _collect_settled_pairs(db_path: str, settings) -> tuple[np.ndarray, np.ndarray]:
    """Run the settlement loop once with replay (stored probs) and return
    (raw_probs, outcomes) for every settleable estimate, in time order.
    Reuses the engine's settlement logic verbatim — no look-ahead."""
    res = run_backtest(
        db_path, settings,
        repricer_mode="replay", vol_mode=None, execution_mode="one-entry",
    )
    rep = res["calibration"]
    # `compute()` doesn't return the raw pair list; rebuild it cheaply by
    # walking the loop again. Cheap vs. the fitting we'll do next.
    import sqlite3
    from src.strategy.calibrator import IsotonicCalibrator  # noqa: F401 (sentinel)
    bracket_w = settings.pricer.bracket_width_usd_default
    conn = sqlite3.connect(db_path)
    try:
        estimates = _load_estimates(conn)
        cand_ts, cand_close = _load_candles(conn)
    finally:
        conn.close()
    pairs: list[tuple[float, int]] = []
    for e in estimates:
        terms = parse_ticker(e["market_id"], now=_EPOCH, bracket_width_usd=bracket_w)
        if terms is None:
            continue
        close_ms = int(terms.close_time.timestamp() * 1000)
        sp = _settle_price(close_ms, cand_ts, cand_close)
        if sp is None:
            continue
        out = _outcome(terms, sp)
        if out is None:
            continue
        pairs.append((float(e["prob"]), out))
    raw = np.fromiter((p for p, _ in pairs), dtype=float, count=len(pairs))
    y = np.fromiter((o for _, o in pairs), dtype=float, count=len(pairs))
    _ = rep  # for symmetry; not used here
    return raw, y


def run_fit_calibrator(db_path: str, settings, out_path: str | None = None) -> int:
    """Fit the production isotonic calibrator on ALL settled history and persist
    it to the configured path (settings.strategy.calibration_model_path).

    Deployment fit uses every available settled pair — no held-out split. That
    is correct for production: the artifact is applied to *future* estimates
    (genuinely out-of-sample), and the B1 mini-gate already proved the mapping
    generalizes forward. Without this artifact the live EdgeStrategy silently
    falls back to raw (uncalibrated) probabilities."""
    from datetime import datetime as _dt
    from src.measurement.calibration import compute as _compute
    from src.strategy.calibrator import IsotonicCalibrator

    target = out_path or settings.strategy.calibration_model_path
    if not target:
        print("No calibration_model_path configured (settings.pricer) and no --out given.")
        return 1

    print("=" * 78)
    print("FIT PRODUCTION CALIBRATOR — Phase 3 B1 artifact")
    print("=" * 78)

    raw, y = _collect_settled_pairs(db_path, settings)
    n = raw.size
    print(f"\n  settled pairs (fit set): {n:,}")
    if n < 100:
        print("  too few settled pairs to fit a production calibrator")
        print("=" * 78)
        return 1

    cal = IsotonicCalibrator().fit(raw, y)

    # In-sample calibration improvement (sanity, not validation — the B1
    # mini-gate is the out-of-sample test).
    rep_raw = _compute(list(zip(raw.tolist(), y.astype(int).tolist())), n_bins=10)
    p_cal = cal.predict(raw)
    rep_cal = _compute(list(zip(p_cal.tolist(), y.astype(int).tolist())), n_bins=10)
    print(f"  in-sample log loss : raw {rep_raw.log_loss:.4f}  →  cal {rep_cal.log_loss:.4f}")
    print(f"  in-sample brier    : raw {rep_raw.brier:.4f}  →  cal {rep_cal.brier:.4f}")

    cal.save(target, metadata={
        "phase": "3", "workstream": "B1",
        "fit_at": _dt.now(tz=timezone.utc).isoformat(),
        "n_settled_pairs": int(n),
        "db_path": db_path,
        "note": "production isotonic calibrator; fit on all settled history",
    })
    print(f"\n  saved → {target}")
    print("  live EdgeStrategy will load + apply this on next start.")
    print("=" * 78)
    return 0


def run_mini_gate_b1(db_path: str, settings) -> int:
    """Phase 3 / Track B / B1 mini-gate.

    Fit an isotonic calibrator on the earlier 80% of settled (raw_prob,
    outcome) pairs in time order; score on the last 20%. Strict time-series
    CV — no shuffle, no look-ahead. PASS iff OOS log loss < 0.69 AND the
    calibrated-prob 0.0–0.1 bin's empirical YES freq < 0.02. Returns 0 on
    PASS (suitable for CI gating)."""
    from src.measurement.calibration import compute as _compute_calib
    from src.strategy.calibrator import IsotonicCalibrator, time_series_split

    print("=" * 78)
    print("MINI-GATE B1 — empirical calibration layer (Phase 3 Track B)")
    print("=" * 78)

    raw, y = _collect_settled_pairs(db_path, settings)
    n = raw.size
    print(f"\n  settled pairs: {n:,}  (time-ordered)")
    if n < 100:
        print("  too few settled pairs to meaningfully fit/score")
        print("=" * 78)
        return 1

    tr, te = time_series_split(n, train_frac=0.8)
    cal = IsotonicCalibrator().fit(raw[tr], y[tr])
    p_cal_oos = cal.predict(raw[te])
    y_oos = y[te]

    rep_raw = _compute_calib(list(zip(raw[te].tolist(), y_oos.astype(int).tolist())), n_bins=10)
    rep_cal = _compute_calib(list(zip(p_cal_oos.tolist(), y_oos.astype(int).tolist())), n_bins=10)

    def _tail(rep) -> float | None:
        for b in rep.bins:
            if b["lo"] == 0.0 and b["hi"] == 0.1 and b["emp_freq"] is not None:
                return b["emp_freq"]
        return None

    raw_ll, cal_ll = rep_raw.log_loss, rep_cal.log_loss
    raw_tail, cal_tail = _tail(rep_raw), _tail(rep_cal)

    print(f"\n  {'metric':<22}{'raw (OOS)':>14}{'calibrated (OOS)':>22}{'bar':>10}")
    print("  " + "-" * 66)
    print(f"  {'log loss':<22}{raw_ll:>14.4f}{cal_ll:>22.4f}{'< 0.69':>10}")
    rt = f"{raw_tail:.4f}" if raw_tail is not None else "—"
    ct = f"{cal_tail:.4f}" if cal_tail is not None else "—"
    print(f"  {'tail bin emp (0-0.1)':<22}{rt:>14}{ct:>22}{'< 0.02':>10}")
    print(f"  {'brier':<22}{rep_raw.brier:>14.4f}{rep_cal.brier:>22.4f}{'(lower better)':>10}")

    ll_pass = cal_ll is not None and cal_ll < 0.69
    tail_pass = cal_tail is not None and cal_tail < 0.02
    passes = ll_pass and tail_pass
    print("\n  ──────────────────────────────────────────────────────────────")
    print(f"  log loss < 0.69  : {'PASS' if ll_pass else 'FAIL'}")
    print(f"  tail emp < 0.02  : {'PASS' if tail_pass else 'FAIL'}")
    if passes:
        print(f"\n  MINI-GATE B1: ✅ PASS  → calibration layer is viable")
        print(f"  → Rejoin Phase 3 Track A at A1 (basis study) with calibrated probs")
        artifact_path = settings.strategy.calibration_model_path
        if artifact_path:
            final_cal = IsotonicCalibrator().fit(raw, y)
            final_cal.save(
                artifact_path,
                metadata={
                    "phase": "3",
                    "workstream": "B1",
                    "fit_samples": int(n),
                    "mini_gate_train_samples": int(tr.stop - tr.start),
                    "mini_gate_test_samples": int(te.stop - te.start),
                    "oos_log_loss": cal_ll,
                    "oos_tail_emp_0_0p1": cal_tail,
                },
            )
            print(f"  wrote live calibrator artifact: {artifact_path}")
        rc = 0
    else:
        print(f"\n  MINI-GATE B1: ❌ FAIL  → escalate to B2 (Deribit IV)")
        print(f"  Per PHASE3_PLAN.md §4: if B1 misses, attempt B2; if B2 misses, B3 hard stop.")
        rc = 1
    print("=" * 78)
    return rc


def _read_calibrator_fit_at(artifact_path: str | None) -> str | None:
    """Read the production calibrator's `metadata.fit_at` ISO timestamp.
    That's the earliest moment the live pipeline could have been emitting
    calibrated probs — the natural default GATE B 'since' boundary."""
    if not artifact_path or not Path(artifact_path).exists():
        return None
    try:
        import json as _json
        payload = _json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        meta = payload.get("metadata") or {}
        v = meta.get("fit_at")
        return str(v) if v else None
    except Exception:
        return None


def _coverage_pct(db_path: str, since_iso: str, now_iso: str | None = None) -> tuple[float, int]:
    """Coinbase candle coverage % over the GATE B forward window + largest gap (minutes).
    Reuses find_candle_gaps so the metric matches the dashboard / WS3 acceptance bar."""
    from src.data.coinbase_ws import find_candle_gaps
    import sqlite3
    s_ms = _parse_iso_ms(since_iso)
    if s_ms is None:
        return 0.0, 0
    e_ms = _parse_iso_ms(now_iso) if now_iso else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    total_min = max(1, (e_ms - s_ms) // 60_000)
    conn = sqlite3.connect(db_path)
    try:
        ts = [int(r[0]) for r in conn.execute(
            "SELECT timestamp_ms FROM coinbase_candle WHERE timestamp_ms >= ? AND timestamp_ms < ? "
            "ORDER BY timestamp_ms ASC",
            (s_ms, e_ms),
        ).fetchall()]
    finally:
        conn.close()
    have = len(ts)
    coverage_pct = 100.0 * have / total_min
    gaps = find_candle_gaps(ts, s_ms, e_ms)
    max_gap_min = (max((b - a) // 60_000 for a, b in gaps)) if gaps else 0
    return float(coverage_pct), int(max_gap_min)


def run_gate_b(
    db_path: str,
    settings,
    since_iso: str | None = None,
    min_days: int = 5,
) -> int:
    """Phase 3 / A3 GATE B — forward paper validation.

    Evaluates the *deployed calibrated* pipeline over the post-deploy window.
    PASS iff every criterion holds:
      1) ≥ min_days of forward data
      2) Calibrated log loss < 0.69  AND  tail-bin emp < 0.02   (same bar as B1)
      3) Edge-decile realized PnL monotone-rising (≥ 7/9 + spread ≥ 2¢)
      4) Net paper PnL > total fees over the window
      5) Coinbase candle coverage ≥ 95% in the window (no big gaps)
    Returns exit code 0 on PASS, 1 on FAIL — suitable for CI gating."""
    artifact = settings.strategy.calibration_model_path

    # Resolve forward window start.
    eff_since = since_iso or _read_calibrator_fit_at(artifact)
    if eff_since is None:
        print("GATE B requires either --since or a calibrator artifact with metadata.fit_at.")
        return 1
    if not artifact or not Path(artifact).exists():
        print(f"GATE B requires a deployed calibrator artifact "
              f"(settings.strategy.calibration_model_path = {artifact!r}).")
        print("Run `python -m src.backtest.engine --fit-calibrator` first.")
        return 1

    print("=" * 78)
    print("GATE B — Phase 3 / A3 forward paper validation")
    print("=" * 78)
    print(f"  artifact : {artifact}")
    print(f"  since    : {eff_since}")

    res = run_backtest(
        db_path, settings,
        repricer_mode="replay",
        vol_mode=None,
        execution_mode="faithful",
        calibrator_artifact_path=artifact,
        since_iso=eff_since,
    )
    rep = res["calibration"]
    trades = res["trades"]
    n_settle = res["n_settleable"]

    # --- 1) Days covered ---
    since_ms = _parse_iso_ms(eff_since) or 0
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    days_covered = (now_ms - since_ms) / (24 * 3600 * 1000)
    days_ok = days_covered >= min_days

    # --- 2) Calibration bar (mirrors B1 mini-gate) ---
    ll_ok = (rep.n_samples > 0 and rep.log_loss is not None and rep.log_loss < 0.69)
    tail_emp = next(
        (b["emp_freq"] for b in rep.bins
         if b["lo"] == 0.0 and b["hi"] == 0.1 and b["emp_freq"] is not None),
        None,
    )
    tail_ok = tail_emp is not None and tail_emp < 0.02

    # --- 3) Edge-decile monotonicity ---
    mono = _monotonicity_check(_decile_table(trades))

    # --- 4) Net paper PnL > fees ---
    fees = sum(t.fee_cents for t in trades)
    net = sum(t.pnl_cents for t in trades)
    pnl_ok = net > fees

    # --- 5) Coverage / no big gaps ---
    cov_pct, max_gap_min = _coverage_pct(db_path, eff_since)
    cov_ok = cov_pct >= 95.0

    # ── Report ──────────────────────────────────────────────
    print(f"\n  forward days covered : {days_covered:>6.2f}     {'PASS' if days_ok else 'FAIL'}  "
          f"(≥ {min_days})")
    print(f"  settleable n (window): {n_settle:>6,}")
    ll = f"{rep.log_loss:.4f}" if rep.log_loss is not None else "—"
    print(f"  cal log loss         : {ll:>10}   {'PASS' if ll_ok else 'FAIL'}  (< 0.69)")
    te = f"{tail_emp:.4f}" if tail_emp is not None else "—"
    print(f"  cal tail bin emp     : {te:>10}   {'PASS' if tail_ok else 'FAIL'}  (< 0.02)")
    print(f"  decile monotonicity  : {mono['note']:>26}  {'PASS' if mono['passes'] else 'FAIL'}")
    print(f"  net PnL              : {net/100:>+8.2f} USD vs fees {fees/100:.2f}   "
          f"{'PASS' if pnl_ok else 'FAIL'}  (net > fees)")
    print(f"  candle coverage      : {cov_pct:>5.1f}% (max gap {max_gap_min}m)   "
          f"{'PASS' if cov_ok else 'FAIL'}  (≥ 95%)")

    passes = all([days_ok, ll_ok, tail_ok, mono["passes"], pnl_ok, cov_ok])
    print("\n  ──────────────────────────────────────────────────────────────")
    if passes:
        print(f"  GATE B: ✅ PASS  → proceed to A4 risk hardening + GATE C tiny-size live")
        rc = 0
    else:
        print(f"  GATE B: ❌ FAIL  → diagnose live/backtest divergence; do NOT proceed to GATE C")
        rc = 1
    print("=" * 78)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay-backtest Strategy 2.")
    ap.add_argument("--db", default=None, help="SQLite DB path (default: from settings)")
    ap.add_argument(
        "--repricer-mode", choices=("replay", "reprice"), default="replay",
        help="replay = use stored probs (Phase 1 default); "
             "reprice = recompute prob from candles at each estimate's computed_at",
    )
    ap.add_argument(
        "--vol-mode", default=None,
        choices=("fixed", "horizon_scaled", "blend", "ewma"),
        help="vol estimator for --repricer-mode=reprice (default: settings.pricer.vol_mode)",
    )
    ap.add_argument(
        "--execution", choices=("one-entry", "faithful"), default="one-entry",
        help="one-entry = one trade per unique market (Phase 1 default); "
             "faithful = honor live rate-limit + per-market + gross-exposure caps",
    )
    ap.add_argument(
        "--gate-a", action="store_true",
        help="Run the full Phase-2 GATE A sweep: replay baseline + each reprice "
             "vol_mode, side-by-side summary, exit code 1 if no mode passes.",
    )
    ap.add_argument(
        "--mini-gate-b1", action="store_true",
        help="Run Phase-3 B1 mini-gate: fit isotonic calibrator on the earlier "
             "80%% of settled pairs, score on the last 20%%, exit code 1 on FAIL.",
    )
    ap.add_argument(
        "--fit-calibrator", action="store_true",
        help="Fit the production isotonic calibrator on all settled history and "
             "save it to settings.strategy.calibration_model_path (or --out).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output path for --fit-calibrator (default: settings.strategy.calibration_model_path).",
    )
    ap.add_argument(
        "--gate-b", action="store_true",
        help="Phase-3 A3 GATE B forward paper validation over the post-deploy "
             "window. Exit 1 unless calibration + monotonicity + PnL + coverage "
             "+ ≥min-days all pass.",
    )
    ap.add_argument(
        "--since", default=None,
        help="GATE B forward-window start (ISO8601). Default: calibrator artifact's metadata.fit_at.",
    )
    ap.add_argument(
        "--min-days", type=int, default=5,
        help="GATE B minimum forward-window length in days (default 5).",
    )
    args = ap.parse_args()

    settings = load_settings()
    db_path = args.db or settings.storage.db_path
    if not Path(db_path).exists():
        print(f"Missing SQLite DB: {db_path}")
        print("Run the bot long enough to collect data, or pass --db /path/to/bot.db.")
        import sys as _sys
        _sys.exit(1)

    if args.gate_a:
        import sys as _sys
        _sys.exit(run_gate_a(db_path, settings))
    if args.mini_gate_b1:
        import sys as _sys
        _sys.exit(run_mini_gate_b1(db_path, settings))
    if args.fit_calibrator:
        import sys as _sys
        _sys.exit(run_fit_calibrator(db_path, settings, out_path=args.out))
    if args.gate_b:
        import sys as _sys
        _sys.exit(run_gate_b(db_path, settings, since_iso=args.since, min_days=args.min_days))

    res = run_backtest(
        db_path, settings,
        repricer_mode=args.repricer_mode,
        vol_mode=args.vol_mode,
        execution_mode=args.execution,
    )
    print_report(res)


if __name__ == "__main__":
    main()
