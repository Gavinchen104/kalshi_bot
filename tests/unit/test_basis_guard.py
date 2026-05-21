"""Unit tests for the A1 near-strike guard in EdgeStrategy."""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import StrategyConfig
from src.strategy.calibrator import IsotonicCalibrator
from src.strategy.edge import EdgeStrategy
from src.types import MarketState, ProbEstimate


def _state(bid=40, ask=42, size=10) -> MarketState:
    return MarketState(
        market_id="KXBTCD-26MAY1417-T80000",
        bid_cents=bid, ask_cents=ask,
        bid_size=size, ask_size=size,
        last_trade_cents=(bid + ask) // 2,
        updated_at=datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


def _est(market_id="KXBTCD-26MAY1417-T80000", prob=0.95, spot=80_010.0,
         horizon=3600) -> ProbEstimate:
    return ProbEstimate(
        market_id=market_id, prob=prob, horizon_seconds=horizon,
        spot_usd=spot, vol_annualized=0.5, source="test",
        computed_at=datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc),
    )


def _cfg(guard_usd=0.0) -> StrategyConfig:
    return StrategyConfig(
        edge_threshold=0.04, min_horizon_seconds=30, max_horizon_seconds=86_400,
        max_spread_cents=10, min_top_book_depth=5,
        near_strike_guard_usd=guard_usd,
    )


def test_guard_disabled_at_zero_lets_signal_through():
    s = EdgeStrategy(_cfg(guard_usd=0.0))
    sig = s.evaluate(_est(spot=80_005.0), _state())
    assert sig is not None and sig.side == "yes"


def test_guard_suppresses_signal_when_spot_near_strike():
    # spot 80,005 vs strike 80,000 → |Δ|=5; guard 50 → blocked.
    s = EdgeStrategy(_cfg(guard_usd=50.0))
    sig = s.evaluate(_est(spot=80_005.0), _state())
    assert sig is None


def test_guard_does_not_suppress_when_spot_far_from_strike():
    # spot 80,100 vs strike 80,000 → |Δ|=100; guard 50 → not blocked.
    s = EdgeStrategy(_cfg(guard_usd=50.0))
    sig = s.evaluate(_est(spot=80_100.0), _state())
    assert sig is not None


def test_guard_uses_bracket_midpoint_for_B_markets():
    # B79875 with default $250 width → bracket [79875, 80125), mid 80000.
    # spot 80,020 → |Δ|=20 from mid → blocked at guard 50.
    s = EdgeStrategy(_cfg(guard_usd=50.0))
    blocked = s.evaluate(_est(market_id="KXBTC-26MAY1417-B79875", spot=80_020.0),
                          _state())
    assert blocked is None
    far = s.evaluate(_est(market_id="KXBTC-26MAY1417-B79875", spot=80_500.0),
                     _state())
    # spot far outside bracket → not blocked by guard (other gates may still
    # reject; here our_prob=0.95 and ask=42 → YES edge huge → passes)
    assert far is not None


def test_unparsable_ticker_does_not_block():
    s = EdgeStrategy(_cfg(guard_usd=50.0))
    sig = s.evaluate(_est(market_id="garbage-id", spot=80_005.0), _state())
    # Guard returns False for unparsable tickers → edge filter still applies normally.
    assert sig is not None


def test_strategy_uses_configured_calibration_model(tmp_path):
    cal = IsotonicCalibrator().fit(
        raw_probs=[0.20, 0.90],
        outcomes=[0.80, 0.95],
    )
    path = tmp_path / "b1_isotonic.json"
    cal.save(path)

    cfg = _cfg(guard_usd=0.0).model_copy(update={"calibration_model_path": str(path)})
    strategy = EdgeStrategy(cfg)
    sig = strategy.evaluate(_est(prob=0.20), _state(bid=60, ask=62))

    assert sig is not None
    assert sig.side == "yes"
    assert sig.our_prob == 0.8
