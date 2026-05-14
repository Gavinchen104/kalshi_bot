"""
Pairs prob_estimates with realized outcomes after settlement, then writes a
calibration snapshot.

Outcome inference for v1: parse the ticker → (strike, close_time). If close_time
has passed, find the closest Coinbase candle to close_time and decide YES (close >= strike)
or NO. Calibration windows over the most recent N settled predictions.
"""
from __future__ import annotations

from src.measurement.calibration import compute
from src.monitoring.logging import get_logger
from src.pricing.ticker import parse_ticker
from src.storage.repository import Repository


logger = get_logger("measurement.reporter")


def settle_and_snapshot(repo: Repository, window: int = 500, n_bins: int = 10) -> dict | None:
    """Walks recent prob_estimates, infers outcomes, writes a calibration snapshot.

    Returns the report dict on success, or None if there's nothing to score yet.
    """
    estimates = repo.recent_prob_estimates(limit=window * 4)
    candles = repo.recent_candles(limit=10_000)
    if not estimates or not candles:
        return None

    candles_by_ts = sorted(candles, key=lambda c: c["timestamp_ms"])

    pairs: list[tuple[float, int]] = []
    for e in estimates:
        terms = parse_ticker(e["market_id"])
        if terms is None:
            continue
        close_ms = int(terms.close_time.timestamp() * 1000)
        # Has the market actually closed yet (relative to our candle history)?
        if not candles_by_ts or candles_by_ts[-1]["timestamp_ms"] < close_ms:
            continue
        # Find the candle closest to close_time (within ±2 minutes).
        nearest = min(candles_by_ts, key=lambda c: abs(c["timestamp_ms"] - close_ms))
        if abs(nearest["timestamp_ms"] - close_ms) > 2 * 60_000:
            continue
        settle_price = float(nearest["close"])
        if terms.direction == "above":
            if terms.strike_usd is None:
                continue
            outcome = 1 if settle_price >= terms.strike_usd else 0
        else:  # bracket
            if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
                continue
            outcome = 1 if (terms.bracket_low_usd <= settle_price < terms.bracket_high_usd) else 0
        pairs.append((float(e["prob"]), outcome))
        if len(pairs) >= window:
            break

    if not pairs:
        return None

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
