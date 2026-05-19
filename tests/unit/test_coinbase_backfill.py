"""Unit tests for backfill_range pagination/dedupe (T04).

Uses an injected window fetcher — no HTTP — so we can assert the range is
walked in <=300-candle windows and that overlapping windows dedupe cleanly.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.coinbase_ws import backfill_range, find_candle_gaps


M = 60_000  # one minute in ms


def test_find_gaps_no_existing_returns_full_range():
    assert find_candle_gaps([], 0, 100 * M) == [(0, 100 * M)]


def test_find_gaps_tiny_hole_ignored():
    # 2-minute hole (< default min_gap_candles=5) is not worth a request.
    ts = [0, M, 2 * M, 6 * M, 7 * M]  # missing 3,4,5
    assert find_candle_gaps(ts, 0, 8 * M) == []


def test_find_gaps_internal_leading_trailing():
    # existing: minutes 10..12 only, window 0..30
    ts = [10 * M, 11 * M, 12 * M]
    gaps = find_candle_gaps(ts, 0, 30 * M, min_gap_candles=5)
    # leading [0,10m), trailing [13m,30m); no internal gap
    assert (0, 10 * M) in gaps
    assert (13 * M, 30 * M) in gaps


def test_find_gaps_internal_only():
    ts = [0, M, 2 * M, 20 * M, 21 * M]  # big hole between 2m and 20m
    gaps = find_candle_gaps(ts, 0, 22 * M, min_gap_candles=5)
    assert (3 * M, 20 * M) in gaps


def _ms(y, mo, d, h, mi) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


def _make_fetcher(calls: list):
    """Returns one synthetic candle row per minute in [start, end), and records
    each (start_iso, end_iso) call for pagination assertions."""
    async def _fetch(start_iso: str, end_iso: str):
        calls.append((start_iso, end_iso))
        s = int(datetime.fromisoformat(start_iso).timestamp())
        e = int(datetime.fromisoformat(end_iso).timestamp())
        rows = []
        t = s - (s % 60)
        while t < e:
            # Coinbase row shape: [time, low, high, open, close, volume]
            rows.append([t, 100.0, 110.0, 105.0, 108.0, 1.5])
            t += 60
        return rows
    return _fetch


@pytest.mark.asyncio
async def test_empty_or_inverted_range_returns_empty():
    assert await backfill_range("u", "BTC-USD", 1000, 1000, _window_fetcher=_make_fetcher([])) == []
    assert await backfill_range("u", "BTC-USD", 2000, 1000, _window_fetcher=_make_fetcher([])) == []


@pytest.mark.asyncio
async def test_single_window_under_300():
    start = _ms(2026, 5, 14, 0, 0)
    end = _ms(2026, 5, 14, 1, 0)  # 60 minutes
    calls: list = []
    out = await backfill_range("u", "BTC-USD", start, end,
                               pause_s=0.0, _window_fetcher=_make_fetcher(calls))
    assert len(calls) == 1                 # one window, well under 300
    assert len(out) == 60
    assert out == sorted(out, key=lambda c: c.timestamp_ms)
    assert out[0].timestamp_ms == start
    assert all(c.is_closed for c in out)


@pytest.mark.asyncio
async def test_paginates_in_300min_windows():
    start = _ms(2026, 5, 14, 0, 0)
    end = _ms(2026, 5, 14, 12, 0)  # 720 minutes → ceil(720/300) = 3 windows
    calls: list = []
    out = await backfill_range("u", "BTC-USD", start, end,
                               pause_s=0.0, _window_fetcher=_make_fetcher(calls))
    assert len(calls) == 3
    assert len(out) == 720                 # contiguous, deduped
    ts = [c.timestamp_ms for c in out]
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)         # no duplicates across window seams


@pytest.mark.asyncio
async def test_dedupes_overlapping_rows():
    """A fetcher that returns the SAME row in every window must not duplicate."""
    async def dup_fetch(start_iso, end_iso):
        return [[_ms(2026, 5, 14, 0, 0) // 1000, 1.0, 2.0, 1.5, 1.8, 1.0]]
    out = await backfill_range("u", "BTC-USD",
                               _ms(2026, 5, 14, 0, 0), _ms(2026, 5, 14, 12, 0),
                               pause_s=0.0, _window_fetcher=dup_fetch)
    assert len(out) == 1
