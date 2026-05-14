"""
Closed-form binary pricer for Kalshi BTC contracts.

Model: BTC spot follows a driftless geometric Brownian motion under the
short-horizon, risk-neutral assumption. Two payoff shapes are supported.

Above-strike (KXBTCD-*-T<strike>):
    P(BTC_T > K) = N(d2(K))
    d2(K) = (ln(S/K) - 0.5 * σ^2 * T) / (σ * sqrt(T))

Range bracket (KXBTC-*-B<low>):
    P(K_lo <= BTC_T < K_hi) = N(d2(K_lo)) - N(d2(K_hi))

T is time-to-expiry in years; σ is annualized vol.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from scipy.stats import norm

from src.pricing.ticker import parse_ticker
from src.pricing.volatility import MINUTES_PER_YEAR, clamp_vol, close_to_close_vol
from src.types import ContractTerms, ProbEstimate


def _d2(spot_usd: float, strike_usd: float, sigma_annualized: float, T_years: float) -> float:
    vol_t = sigma_annualized * math.sqrt(T_years)
    return (math.log(spot_usd / strike_usd) - 0.5 * sigma_annualized**2 * T_years) / vol_t


def price_yes_prob(
    spot_usd: float,
    strike_usd: float,
    horizon_seconds: float,
    sigma_annualized: float,
) -> float:
    """P(spot > strike at expiry) under driftless lognormal."""
    if spot_usd <= 0 or strike_usd <= 0 or horizon_seconds <= 0 or sigma_annualized <= 0:
        return 1.0 if spot_usd > strike_usd else 0.0
    T = horizon_seconds / (MINUTES_PER_YEAR * 60.0)
    if sigma_annualized * math.sqrt(T) <= 0:
        return 1.0 if spot_usd > strike_usd else 0.0
    return float(norm.cdf(_d2(spot_usd, strike_usd, sigma_annualized, T)))


def price_bracket_prob(
    spot_usd: float,
    bracket_low_usd: float,
    bracket_high_usd: float,
    horizon_seconds: float,
    sigma_annualized: float,
) -> float:
    """P(low <= spot_T < high) under driftless lognormal."""
    if bracket_high_usd <= bracket_low_usd:
        return 0.0
    if spot_usd <= 0 or horizon_seconds <= 0 or sigma_annualized <= 0:
        return 1.0 if (bracket_low_usd <= spot_usd < bracket_high_usd) else 0.0
    T = horizon_seconds / (MINUTES_PER_YEAR * 60.0)
    if sigma_annualized * math.sqrt(T) <= 0:
        return 1.0 if (bracket_low_usd <= spot_usd < bracket_high_usd) else 0.0
    # P(spot > low) - P(spot > high) = P(low <= spot < high)
    p_above_low = float(norm.cdf(_d2(spot_usd, bracket_low_usd, sigma_annualized, T)))
    p_above_high = float(norm.cdf(_d2(spot_usd, bracket_high_usd, sigma_annualized, T)))
    return max(0.0, p_above_low - p_above_high)


class CoinbasePricer:
    def __init__(
        self,
        vol_window_minutes: int = 60,
        vol_floor: float = 0.20,
        vol_ceiling: float = 3.00,
        min_horizon_seconds: int = 5,
        bracket_width_usd_default: float = 250.0,
    ) -> None:
        self.vol_window_minutes = vol_window_minutes
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
        self.min_horizon_seconds = min_horizon_seconds
        self.bracket_width_usd_default = bracket_width_usd_default

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

        terms = terms_override or parse_ticker(
            market_id, now=now, bracket_width_usd=self.bracket_width_usd_default
        )
        if terms is None:
            return None

        horizon = (terms.close_time - now).total_seconds()
        if horizon < self.min_horizon_seconds:
            return None

        sigma_raw = close_to_close_vol(closes_1m, window=self.vol_window_minutes)
        sigma = clamp_vol(sigma_raw, self.vol_floor, self.vol_ceiling)
        if sigma is None:
            return None

        if terms.direction == "above":
            if terms.strike_usd is None:
                return None
            prob = price_yes_prob(
                spot_usd=spot_usd,
                strike_usd=terms.strike_usd,
                horizon_seconds=horizon,
                sigma_annualized=sigma,
            )
            source = "coinbase_phi_above"
        else:
            if terms.bracket_low_usd is None or terms.bracket_high_usd is None:
                return None
            prob = price_bracket_prob(
                spot_usd=spot_usd,
                bracket_low_usd=terms.bracket_low_usd,
                bracket_high_usd=terms.bracket_high_usd,
                horizon_seconds=horizon,
                sigma_annualized=sigma,
            )
            source = "coinbase_phi_bracket"

        return ProbEstimate(
            market_id=market_id,
            prob=prob,
            horizon_seconds=horizon,
            spot_usd=spot_usd,
            vol_annualized=sigma,
            source=source,
            computed_at=now,
        )
