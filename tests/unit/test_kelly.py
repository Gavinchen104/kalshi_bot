from __future__ import annotations

from src.strategy.kelly import kelly_contracts


def test_zero_when_no_edge():
    assert kelly_contracts(our_prob=0.5, price_cents=50, bankroll_cents=10_000) == 0
    assert kelly_contracts(our_prob=0.4, price_cents=50, bankroll_cents=10_000) == 0


def test_positive_when_edge():
    qty = kelly_contracts(our_prob=0.7, price_cents=50, bankroll_cents=10_000, kelly_fraction=1.0, max_contracts=999)
    assert qty > 0


def test_max_contracts_caps():
    qty = kelly_contracts(our_prob=0.9, price_cents=20, bankroll_cents=1_000_000, kelly_fraction=1.0, max_contracts=5)
    assert qty == 5


def test_edge_cases_at_zero_or_one():
    assert kelly_contracts(0.0, 50, 1_000) == 0
    assert kelly_contracts(1.0, 50, 1_000) == 0
    assert kelly_contracts(0.7, 0, 1_000) == 0
    assert kelly_contracts(0.7, 100, 1_000) == 0
