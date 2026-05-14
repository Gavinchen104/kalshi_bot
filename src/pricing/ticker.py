"""
Parser for Kalshi BTC market tickers.

Two contract families currently handled:
    KXBTCD-<YY><MMM><DD><HH>-T<strike>   — daily above-strike, pays $1 if BTC_T > strike.
    KXBTC-<YY><MMM><DD><HH>-B<low>       — 15-min bracket, pays $1 if low <= BTC_T < low + width.

The intraday bracket width is series-dependent ($250 or $500 are common). We
default to `bracket_width_usd_default` from config and infer per-series later
when a BracketRegistry observes adjacent strikes (Phase 2).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from src.types import ContractTerms


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# KXBTC<optional letters/digits>-<YY><MMM><DD><HH>-<T|B><strike-or-low>
_TICKER_RE = re.compile(
    r"^KXBTC[A-Z0-9]*-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hh>\d{2})"
    r"-(?P<kind>[TB])(?P<strike>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def next_quarter_boundary(now: datetime) -> datetime:
    """Return the next :00 / :15 / :30 / :45 UTC boundary strictly after now."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    minute = (now.minute // 15 + 1) * 15
    base = now.replace(second=0, microsecond=0, minute=0)
    return base + timedelta(minutes=minute)


def parse_ticker(
    market_id: str,
    now: datetime | None = None,
    bracket_width_usd: float = 250.0,
) -> ContractTerms | None:
    """Parse a Kalshi BTC ticker. Returns None if the ticker isn't recognizable."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    m = _TICKER_RE.match(market_id)
    if not m:
        return None

    mon_idx = _MONTHS.get(m.group("mon").upper())
    if mon_idx is None:
        return None

    try:
        year = 2000 + int(m.group("yy"))
        day = int(m.group("dd"))
        hour = int(m.group("hh"))
        strike_field = float(m.group("strike"))
    except ValueError:
        return None

    try:
        anchor = datetime(year, mon_idx, day, hour, 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None

    close_time = anchor if anchor >= now else next_quarter_boundary(now)

    kind = m.group("kind").upper()
    if kind == "T":
        return ContractTerms(
            market_id=market_id,
            close_time=close_time,
            direction="above",
            strike_usd=strike_field,
        )
    # kind == "B": range bracket
    return ContractTerms(
        market_id=market_id,
        close_time=close_time,
        direction="bracket",
        bracket_low_usd=strike_field,
        bracket_high_usd=strike_field + bracket_width_usd,
    )


def fallback_close_time(now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(tz=timezone.utc)
    return next_quarter_boundary(now)
