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
from src.pricing.volatility import (
    MINUTES_PER_YEAR,
    clamp_vol,
    close_to_close_vol,
    horizon_matched_vol,
)
from src.types import ContractTerms, ProbEstimate


# Vol-mode dispatch table; B2 adds "blend" and "ewma". "fixed" is the Phase 1
# baseline and remains the default so each mode change is A/B-comparable.
SUPPORTED_VOL_MODES = ("fixed", "horizon_scaled")


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
        vol_mode: str = "fixed",
        vol_window_floor_min: int = 60,
        vol_window_cap_min: int = 1440,
    ) -> None:
        if vol_mode not in SUPPORTED_VOL_MODES:
            raise ValueError(
                f"unknown vol_mode={vol_mode!r}; supported: {SUPPORTED_VOL_MODES}"
            )
        self.vol_window_minutes = vol_window_minutes
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
        self.min_horizon_seconds = min_horizon_seconds
        self.bracket_width_usd_default = bracket_width_usd_default
        self.vol_mode = vol_mode
        self.vol_window_floor_min = vol_window_floor_min
        self.vol_window_cap_min = vol_window_cap_min

    def _estimate_vol(self, closes_1m, horizon_seconds: float) -> float | None:
        """Dispatch to the selected vol estimator. 'fixed' is bit-identical to
        the Phase 1 path (close_to_close_vol over self.vol_window_minutes)."""
        if self.vol_mode == "horizon_scaled":
            return horizon_matched_vol(
                closes_1m, horizon_seconds=horizon_seconds,
                floor_min=self.vol_window_floor_min,
                cap_min=self.vol_window_cap_min,
            )
        # default / "fixed"
        return close_to_close_vol(closes_1m, window=self.vol_window_minutes)

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

        sigma_raw = self._estimate_vol(closes_1m, horizon_seconds=horizon)
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
