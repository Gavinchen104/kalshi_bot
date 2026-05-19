from __future__ import annotations

from datetime import datetime, timezone

from src.pricing.ticker import next_quarter_boundary, parse_ticker, resolve_close_time


def test_parse_above_strike():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    t = parse_ticker("KXBTCD-26MAY1413-T108500", now=now)
    assert t is not None
    assert t.direction == "above"
    assert t.strike_usd == 108_500.0
    assert t.bracket_low_usd is None and t.bracket_high_usd is None
    assert t.close_time.year == 2026 and t.close_time.month == 5
    assert t.close_time.day == 14
    assert t.close_time.hour == 13


def test_parse_above_decimal_strike():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    t = parse_ticker("KXBTCD-26MAY1413-T108500.5", now=now)
    assert t is not None and t.strike_usd == 108_500.5


def test_parse_bracket():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    t = parse_ticker("KXBTC-26MAY1413-B71375", now=now, bracket_width_usd=250.0)
    assert t is not None
    assert t.direction == "bracket"
    assert t.bracket_low_usd == 71_375.0
    assert t.bracket_high_usd == 71_625.0
    assert t.strike_usd is None


def test_parse_bracket_custom_width():
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    t = parse_ticker("KXBTC-26MAY1517-B67750", now=now, bracket_width_usd=500.0)
    assert t is not None and t.bracket_high_usd == 68_250.0


def test_parse_rejects_non_btc_and_garbage():
    now = datetime(2026, 5, 14, tzinfo=timezone.utc)
    assert parse_ticker("KXETH-26MAY1413-T3000", now=now) is None
    assert parse_ticker("garbage", now=now) is None
    # Skips wide-format yearly markets that don't fit YY MMM DD HH pattern.
    assert parse_ticker("KXBTC2026200-27JAN01-200000", now=now) is None


def test_settlement_mode_uses_true_anchor_for_past_market():
    # A contract that closed 4 days before "now". Default mode wrongly rolls
    # close_time to the next 15-min boundary from now; settlement_mode must
    # return the real historical close (2026-05-14 17:00).
    now = datetime(2026, 5, 18, 9, 7, 30, tzinfo=timezone.utc)
    default = parse_ticker("KXBTCD-26MAY1417-T80000", now=now)
    assert default is not None
    assert default.close_time != datetime(2026, 5, 14, 17, 0, 0, tzinfo=timezone.utc)

    settled = parse_ticker("KXBTCD-26MAY1417-T80000", now=now, settlement_mode=True)
    assert settled is not None
    assert settled.close_time == datetime(2026, 5, 14, 17, 0, 0, tzinfo=timezone.utc)


def test_resolve_close_time_helper():
    assert resolve_close_time("KXBTCD-26MAY1417-T80000") == datetime(
        2026, 5, 14, 17, 0, 0, tzinfo=timezone.utc
    )
    assert resolve_close_time("garbage") is None


def test_settlement_mode_does_not_change_future_market():
    # If the contract is still in the future, both modes agree on the anchor.
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    default = parse_ticker("KXBTCD-26MAY1413-T108500", now=now)
    settled = parse_ticker("KXBTCD-26MAY1413-T108500", now=now, settlement_mode=True)
    assert default is not None and settled is not None
    assert default.close_time == settled.close_time


def test_next_quarter_boundary():
    now = datetime(2026, 5, 14, 11, 7, 30, tzinfo=timezone.utc)
    assert next_quarter_boundary(now) == datetime(2026, 5, 14, 11, 15, 0, tzinfo=timezone.utc)
    on_boundary = datetime(2026, 5, 14, 11, 15, 0, tzinfo=timezone.utc)
    assert next_quarter_boundary(on_boundary) == datetime(2026, 5, 14, 11, 30, 0, tzinfo=timezone.utc)
