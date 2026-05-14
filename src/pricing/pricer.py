"""
Closed-form binary pricer for Kalshi BTC 15m contracts.

Model: BTC spot follows a driftless geometric Brownian motion under the
short-horizon, risk-neutral assumption. For a contract paying $1 if
BTC_T > strike at expiry,

    P(BTC_T > K) = N(d2)
    d2 = (ln(S/K) - 0.5 * σ^2 * T) / (σ * sqrt(T))

where T is time-to-expiry in years and σ is annualized vol.

At 15-minute horizons the 0.5σ²T term is tiny but included for correctness.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from scipy.stats import norm

from src.pricing.ticker import fallback_close_time, parse_ticker
from src.pricing.volatility import MINUTES_PER_YEAR, clamp_vol, close_to_close_vol
from src.types import ContractTerms, ProbEstimate


def price_yes_prob(
    spot_usd: float,
    strike_usd: float,
    horizon_seconds: float,
    sigma_annualized: float,
) -> float:
    """P(spot > strike at expiry) under driftless lognormal."""
    if spot_usd <= 0 or strike_usd <= 0 or horizon_seconds <= 0 or sigma_annualized <= 0:
        # Degenerate inputs: collapse to indicator on current spot.
        return 1.0 if spot_usd > strike_usd else 0.0

    T = horizon_seconds / (MINUTES_PER_YEAR * 60.0)
    vol_t = sigma_annualized * math.sqrt(T)
    if vol_t <= 0:
        return 1.0 if spot_usd > strike_usd else 0.0

    d2 = (math.log(spot_usd / strike_usd) - 0.5 * sigma_annualized**2 * T) / vol_t
    return float(norm.cdf(d2))


class CoinbasePricer:
    """Builds a ProbEstimate for a Kalshi BTC 15m market using Coinbase data."""

    def __init__(
        self,
        vol_window_minutes: int = 60,
        vol_floor: float = 0.20,
        vol_ceiling: float = 3.00,
        min_horizon_seconds: int = 5,
    ) -> None:
        self.vol_window_minutes = vol_window_minutes
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
        self.min_horizon_seconds = min_horizon_seconds

    def price(
        self,
        market_id: str,
        spot_usd: float,
        closes_1m,
        now: datetime | None = None,
        terms_override: ContractTerms | None = None,
    ) -> ProbEstimate | None:
        if now is None:
            now = datetime.now(tz=timezone.utc)

        terms = terms_override or parse_ticker(market_id, now=now)
        if terms is None:
            # Unparseable ticker — still try with fallback close_time and a
            # missing strike will short-circuit below.
            return None

        horizon = (terms.close_time - now).total_seconds()
        if horizon < self.min_horizon_seconds:
            return None

        sigma_raw = close_to_close_vol(closes_1m, window=self.vol_window_minutes)
        sigma = clamp_vol(sigma_raw, self.vol_floor, self.vol_ceiling)
        if sigma is None:
            return None

        prob = price_yes_prob(
            spot_usd=spot_usd,
            strike_usd=terms.strike_usd,
            horizon_seconds=horizon,
            sigma_annualized=sigma,
        )
        return ProbEstimate(
            market_id=market_id,
            prob=prob,
            horizon_seconds=horizon,
            spot_usd=spot_usd,
            vol_annualized=sigma,
            source="coinbase_phi",
            computed_at=now,
        )
