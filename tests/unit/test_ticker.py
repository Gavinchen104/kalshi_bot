from __future__ import annotations

from datetime import datetime, timezone

from src.pricing.ticker import next_quarter_boundary, parse_ticker


def test_parse_basic():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    terms = parse_ticker("KXBTC-26MAY1413-T108500", now=now)
    assert terms is not None
    assert terms.strike_usd == 108_500.0
    assert terms.close_time.year == 2026 and terms.close_time.month == 5
    assert terms.close_time.day == 14
    assert terms.close_time.hour == 13


def test_parse_with_suffix():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    terms = parse_ticker("KXBTCD-26MAY1413-T108500", now=now)
    assert terms is not None
    assert terms.strike_usd == 108_500.0

    terms2 = parse_ticker("KXBTC15M-26MAY1413-T108500", now=now)
    assert terms2 is not None
    assert terms2.strike_usd == 108_500.0


def test_parse_rejects_non_btc():
    assert parse_ticker("KXETH-26MAY1413-T3000", now=datetime(2026, 5, 14, tzinfo=timezone.utc)) is None
    assert parse_ticker("garbage", now=datetime(2026, 5, 14, tzinfo=timezone.utc)) is None


def test_next_quarter_boundary():
    now = datetime(2026, 5, 14, 11, 7, 30, tzinfo=timezone.utc)
    nb = next_quarter_boundary(now)
    assert nb == datetime(2026, 5, 14, 11, 15, 0, tzinfo=timezone.utc)

    nb2 = next_quarter_boundary(datetime(2026, 5, 14, 11, 15, 0, tzinfo=timezone.utc))
    assert nb2 == datetime(2026, 5, 14, 11, 30, 0, tzinfo=timezone.utc)
