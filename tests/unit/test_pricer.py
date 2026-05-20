from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from src.pricing.pricer import CoinbasePricer, price_bracket_prob, price_yes_prob
from src.types import ContractTerms


def test_at_the_money_is_close_to_half():
    # spot == strike → P(spot > strike at expiry) ≈ 0.5 (slightly less due to -0.5σ²T)
    p = price_yes_prob(spot_usd=100_000, strike_usd=100_000, horizon_seconds=900, sigma_annualized=0.8)
    assert 0.49 < p < 0.51


def test_deep_in_the_money_high_prob():
    p = price_yes_prob(spot_usd=110_000, strike_usd=100_000, horizon_seconds=900, sigma_annualized=0.8)
    assert p > 0.95


def test_deep_out_of_money_low_prob():
    p = price_yes_prob(spot_usd=90_000, strike_usd=100_000, horizon_seconds=900, sigma_annualized=0.8)
    assert p < 0.05


def test_shorter_horizon_pulls_probability_toward_indicator():
    p_long = price_yes_prob(101_000, 100_000, horizon_seconds=900, sigma_annualized=0.8)
    p_short = price_yes_prob(101_000, 100_000, horizon_seconds=60, sigma_annualized=0.8)
    assert p_short > p_long  # less time for spot to wander below strike


def test_pricer_returns_none_when_horizon_too_short():
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    terms = ContractTerms(
        market_id="KXBTCD-26MAY1412-T100000",
        close_time=now + timedelta(seconds=1),
        direction="above",
        strike_usd=100_000.0,
    )
    pricer = CoinbasePricer(vol_window_minutes=10, min_horizon_seconds=30)
    closes = np.linspace(100_000, 101_000, 50)
    est = pricer.price(
        market_id=terms.market_id, spot_usd=100_500, closes_1m=closes,
        now=now, terms_override=terms,
    )
    assert est is None


def test_bracket_high_prob_when_spot_inside_narrow_bracket_near_expiry():
    # Spot near the bracket center, short horizon, low vol → very likely settles in-bracket.
    p = price_bracket_prob(
        spot_usd=80_000, bracket_low_usd=79_875, bracket_high_usd=80_125,
        horizon_seconds=30, sigma_annualized=0.5,
    )
    assert p > 0.8


def test_bracket_low_prob_when_spot_far_below():
    p = price_bracket_prob(
        spot_usd=70_000, bracket_low_usd=80_000, bracket_high_usd=80_250,
        horizon_seconds=900, sigma_annualized=0.6,
    )
    assert p < 0.01


def test_bracket_probs_sum_close_to_one_across_contiguous_brackets():
    # A grid of $250 brackets covering ±$5,000 around spot should sum ≈ probability mass.
    spot, sigma, horizon = 80_000.0, 0.8, 900.0
    width = 250.0
    grid_lows = [spot - 5_000 + i * width for i in range(40)]
    total = sum(
        price_bracket_prob(spot, low, low + width, horizon, sigma)
        for low in grid_lows
    )
    # Most of the probability mass should be inside ±$5k at 15-min horizon.
    assert total > 0.99
    assert total <= 1.0 + 1e-9


def test_pricer_rejects_unknown_vol_mode():
    import pytest
    with pytest.raises(ValueError, match="unknown vol_mode"):
        CoinbasePricer(vol_mode="nonsense")


def test_pricer_fixed_mode_bit_identical_to_phase_1_default():
    """B1 must not change behavior at default config. Compute the prob both
    via the explicit Phase-1 path (close_to_close_vol over vol_window_minutes)
    and via the new dispatch with vol_mode='fixed' — they must match exactly."""
    from src.pricing.volatility import clamp_vol, close_to_close_vol
    from src.pricing.pricer import price_yes_prob

    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    terms = ContractTerms(
        market_id="KXBTCD-26MAY1417-T80000",
        close_time=now + timedelta(hours=5),
        direction="above",
        strike_usd=80_000.0,
    )
    rng = np.random.default_rng(7)
    closes = 80_000 + np.cumsum(rng.normal(0.0, 30.0, 200))

    # Manual Phase-1 calculation.
    raw = close_to_close_vol(closes, window=60)
    sigma = clamp_vol(raw, 0.20, 3.00)
    horizon = (terms.close_time - now).total_seconds()
    expected = price_yes_prob(80_000, 80_000.0, horizon, sigma)

    # Via dispatch with default vol_mode='fixed'.
    pricer = CoinbasePricer(
        vol_window_minutes=60, vol_floor=0.20, vol_ceiling=3.00,
        min_horizon_seconds=5, vol_mode="fixed",
    )
    est = pricer.price(
        market_id=terms.market_id, spot_usd=80_000,
        closes_1m=closes, now=now, terms_override=terms,
    )
    assert est is not None
    assert est.prob == expected
    assert est.vol_annualized == sigma


def test_pricer_horizon_scaled_uses_longer_window_for_long_horizon():
    """At a 24h horizon, horizon_scaled picks a 1440-min window; over the same
    closes that should produce a DIFFERENT sigma than the fixed 60-min path,
    and (for daily contracts on real BTC) typically a larger one."""
    from src.pricing.volatility import close_to_close_vol

    rng = np.random.default_rng(11)
    # Mildly heteroskedastic series: calm tail (last 60), noisier earlier.
    base = 80_000.0
    noisy = rng.normal(0.0, 80.0, 1500)
    calm_tail = rng.normal(0.0, 5.0, 60)
    closes = base + np.cumsum(np.concatenate([noisy, calm_tail]))

    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    terms = ContractTerms(
        market_id="KXBTCD-26MAY1512-T80000",
        close_time=now + timedelta(hours=24),
        direction="above",
        strike_usd=80_000.0,
    )

    fixed_pricer = CoinbasePricer(vol_window_minutes=60, vol_mode="fixed")
    scaled_pricer = CoinbasePricer(vol_mode="horizon_scaled",
                                    vol_window_floor_min=60, vol_window_cap_min=1440)

    fixed_est = fixed_pricer.price(terms.market_id, 80_000, closes,
                                   now=now, terms_override=terms)
    scaled_est = scaled_pricer.price(terms.market_id, 80_000, closes,
                                     now=now, terms_override=terms)
    assert fixed_est is not None and scaled_est is not None
    # Fixed sees only the calm tail → low vol. Scaled sees the full 1440 →
    # higher vol. (This is the entire Phase-1 mispricing the fix targets.)
    assert scaled_est.vol_annualized > fixed_est.vol_annualized
    # And the calibrated probabilities should differ accordingly.
    assert scaled_est.prob != fixed_est.prob


def test_pricer_handles_bracket_market_via_terms_override():
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    terms = ContractTerms(
        market_id="KXBTC-26MAY1412-B79875",
        close_time=now + timedelta(minutes=5),
        direction="bracket",
        bracket_low_usd=79_875.0,
        bracket_high_usd=80_125.0,
    )
    pricer = CoinbasePricer(vol_window_minutes=10, min_horizon_seconds=30)
    closes = np.linspace(79_500, 80_000, 50)
    est = pricer.price(
        market_id=terms.market_id, spot_usd=80_000, closes_1m=closes,
        now=now, terms_override=terms,
    )
    assert est is not None
    assert est.source == "coinbase_phi_bracket"
    assert 0.0 < est.prob < 1.0
