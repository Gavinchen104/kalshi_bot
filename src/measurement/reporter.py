"""
Pairs prob_estimates with realized outcomes after settlement, then writes a
calibration snapshot.

Outcome inference: resolve the ticker → (strike/bracket, true close_time). If a
Coinbase candle exists within ±2 min of the close, decide YES/NO. Calibration
is computed over the most recent `window` *settled* predictions.

Two correctness requirements (both were bugs in the original implementation):
  1. Scan ALL estimates, not just the most-recent slice — the recent rows are
     all still-open markets, so a recency-limited scan never settles anything.
  2. Use the contract's TRUE anchored close time (settlement_mode), not the
     now-relative fallback, or historical markets settle at the wrong time.
"""
from __future__ import annotations

from bisect import bisect_left

from src.measurement.calibration import compute
from src.monitoring.logging import get_logger
from src.pricing.ticker import parse_ticker
from src.storage.repository import Repository


logger = get_logger("measurement.reporter")

_SETTLE_TOLERANCE_MS = 2 * 60_000


def _nearest_close(close_ms: int, cand_ts: list[int], cand_close: list[float]) -> float | None:
    """Coinbase close nearest to close_ms, or None if none within tolerance."""
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


def settle_and_snapshot(repo: Repository, window: int = 500, n_bins: int = 10) -> dict | None:
    """Walk all prob_estimates, settle the ones whose markets have closed,
    write a calibration snapshot over the most recent `window` settled pairs.

    Returns the report dict on success, or None if nothing is settleable yet.
    """
    estimates = repo.prob_estimates_for_settlement()
    candles = repo.recent_candles(limit=10_000)
    if not estimates or not candles:
        return None

    candles_by_ts = sorted(candles, key=lambda c: c["timestamp_ms"])
    cand_ts = [int(c["timestamp_ms"]) for c in candles_by_ts]
    cand_close = [float(c["close"]) for c in candles_by_ts]

    pairs: list[tuple[float, int]] = []
    for e in estimates:
        # settlement_mode=True → contract's real anchored close, not now-relative.
        terms = parse_ticker(e["market_id"], settlement_mode=True)
        if terms is None:
            continue
        close_ms = int(terms.close_time.timestamp() * 1000)
        settle_price = _nearest_close(close_ms, cand_ts, cand_close)
        if settle_price is None:
            continue
        if terms.direction == "above":
            if terms.strike_usd is None:
                continue
            outcome = 1 if settle_price >= terms.strike_usd else 0
        else:  # bracket
            if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
                continue
            outcome = 1 if (terms.bracket_low_usd <= settle_price < terms.bracket_high_usd) else 0
        pairs.append((float(e["prob"]), outcome))

    if not pairs:
        return None

    # Most recent `window` settled predictions (estimates were oldest-first).
    pairs = pairs[-window:]

    report = compute(pairs, n_bins=n_bins)
    repo.save_calibration(
        window_size=len(pairs),
        brier=report.brier,
        log_loss=report.log_loss,
        n_samples=report.n_samples,
        bins=report.bins,
    )
    logger.info(
        "calibration_snapshot",
        n=report.n_samples, brier=report.brier, log_loss=report.log_loss,
    )
    return {
        "n_samples": report.n_samples,
        "brier": report.brier,
        "log_loss": report.log_loss,
        "bins": report.bins,
    }
