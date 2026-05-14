"""
Best-effort parser for Kalshi BTC 15-minute market tickers.

Observed ticker patterns include forms like:
    KXBTC-26MAY1314-T108500     -> BTC ≥ $108,500 at 14:00 UTC on 2026-05-13
    KXBTCD-26MAY1314-T108500    -> same with KXBTCD prefix
    KXBTC15M-26MAY1314-T108500

The parser is intentionally lenient: if it cannot pin down the close time
from the ticker, the caller can fall back to the next 15-minute wall-clock
boundary, which is the right answer for every active 15m market.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from src.types import ContractTerms


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# KXBTC<optional suffix>-<YY><MMM><DD><HH>-T<strike>
# The hour code in the middle field is a 2-digit hour-of-day in UTC.
_TICKER_RE = re.compile(
    r"^KXBTC[A-Z0-9]*-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hh>\d{2})-T(?P<strike>\d+)$",
    re.IGNORECASE,
)


def next_quarter_boundary(now: datetime) -> datetime:
    """Return the next :00 / :15 / :30 / :45 UTC boundary strictly after now."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    minute = (now.minute // 15 + 1) * 15
    base = now.replace(second=0, microsecond=0, minute=0)
    return base + timedelta(minutes=minute)


def parse_ticker(market_id: str, now: datetime | None = None) -> ContractTerms | None:
    """Parse a Kalshi BTC 15m ticker. Returns None if the ticker isn't recognizable."""
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
        strike = float(m.group("strike"))
    except ValueError:
        return None

    # The hour in the ticker corresponds to the *closing* hour. For BTC 15m
    # markets the close minute is determined by which window within the hour.
    # Without more info we anchor to the next :00/:15/:30/:45 boundary at or
    # after that wall-clock hour.
    try:
        anchor = datetime(year, mon_idx, day, hour, 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None

    close_time = anchor if anchor >= now else next_quarter_boundary(now)

    return ContractTerms(
        market_id=market_id,
        strike_usd=strike,
        close_time=close_time,
        direction="above",
    )


def fallback_close_time(now: datetime | None = None) -> datetime:
    """If the ticker is unparseable, use the next 15-min boundary as a best guess."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    return next_quarter_boundary(now)
