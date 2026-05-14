from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from src.pricing.pricer import CoinbasePricer, price_yes_prob
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
        market_id="KXBTC-26MAY1412-T100000",
        strike_usd=100_000.0,
        close_time=now + timedelta(seconds=1),
        direction="above",
    )
    pricer = CoinbasePricer(vol_window_minutes=10, min_horizon_seconds=30)
    closes = np.linspace(100_000, 101_000, 50)
    est = pricer.price(
        market_id=terms.market_id, spot_usd=100_500, closes_1m=closes,
        now=now, terms_override=terms,
    )
    assert est is None
