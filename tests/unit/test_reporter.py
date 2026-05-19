"""Regression tests for settle_and_snapshot (T03).

Locks in the two bugs fixed in T01/T02:
  - calibration must populate from a synthetic fixture DB (not return None);
  - a market that closed days before "now" must still settle correctly
    (settlement_mode close time, not now-relative).
No live-DB dependency: everything runs against a tmp_path SQLite file.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.measurement.reporter import settle_and_snapshot
from src.storage.repository import Repository
from src.types import ProbEstimate


# A contract that closed long before any plausible test run time.
_CLOSE = datetime(2026, 5, 14, 17, 0, 0, tzinfo=timezone.utc)
_CLOSE_MS = int(_CLOSE.timestamp() * 1000)


def _est(market_id: str, prob: float) -> ProbEstimate:
    return ProbEstimate(
        market_id=market_id,
        prob=prob,
        horizon_seconds=3600.0,
        spot_usd=80_000.0,
        vol_annualized=0.5,
        source="test",
        computed_at=datetime(2026, 5, 14, 16, 0, 0, tzinfo=timezone.utc),
    )


def _seed(repo: Repository, settle_close: float) -> None:
    # One candle exactly at the contracts' close time.
    repo.save_candles([{
        "timestamp_ms": _CLOSE_MS,
        "open": settle_close, "high": settle_close,
        "low": settle_close, "close": settle_close,
        "volume": 1.0,
    }])


def test_calibration_populates_from_fixture(tmp_path):
    repo = Repository(str(tmp_path / "t.db"))
    _seed(repo, settle_close=81_000.0)

    # T80000 → 81000 >= 80000 → YES (outcome 1). Predicted 0.9 (well-calibrated-ish).
    repo.save_prob_estimate(_est("KXBTCD-26MAY1417-T80000", 0.90))
    # T90000 → 81000 < 90000 → NO (outcome 0). Predicted 0.05.
    repo.save_prob_estimate(_est("KXBTCD-26MAY1417-T90000", 0.05))

    res = settle_and_snapshot(repo, window=500, n_bins=10)
    assert res is not None, "calibration must populate, not return None"
    assert res["n_samples"] == 2
    # Both predictions were roughly right → Brier should be small.
    assert res["brier"] < 0.05
    # And it must be persisted.
    assert repo.latest_calibration() is not None


def test_past_market_still_settles(tmp_path):
    # Regression for the now-relative close bug: the contract closed days ago
    # relative to real wall-clock; it must still settle (settlement_mode).
    repo = Repository(str(tmp_path / "t.db"))
    _seed(repo, settle_close=70_000.0)
    repo.save_prob_estimate(_est("KXBTCD-26MAY1417-T80000", 0.30))  # 70k<80k → NO

    res = settle_and_snapshot(repo, window=500, n_bins=10)
    assert res is not None
    assert res["n_samples"] == 1


def test_returns_none_when_no_candle_near_close(tmp_path):
    repo = Repository(str(tmp_path / "t.db"))
    # Candle is 10 minutes from close → outside the ±2min tolerance.
    repo.save_candles([{
        "timestamp_ms": _CLOSE_MS + 10 * 60_000,
        "open": 81_000.0, "high": 81_000.0, "low": 81_000.0,
        "close": 81_000.0, "volume": 1.0,
    }])
    repo.save_prob_estimate(_est("KXBTCD-26MAY1417-T80000", 0.9))

    assert settle_and_snapshot(repo) is None


def test_bracket_outcome(tmp_path):
    repo = Repository(str(tmp_path / "t.db"))
    _seed(repo, settle_close=71_500.0)
    # B71375 with default $250 width → [71375, 71625). 71500 is inside → YES.
    repo.save_prob_estimate(_est("KXBTC-26MAY1417-B71375", 0.80))
    res = settle_and_snapshot(repo)
    assert res is not None and res["n_samples"] == 1
    # Single correct-ish prediction (0.8 vs outcome 1) → Brier = 0.04.
    assert abs(res["brier"] - 0.04) < 1e-9
