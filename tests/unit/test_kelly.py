from src.strategy.kelly import kelly_contracts, kelly_fraction


def test_kelly_no_edge_returns_zero():
    assert kelly_fraction(0.50, 0.50) == 0.0


def test_kelly_positive_edge():
    f = kelly_fraction(0.70, 0.50, kelly_scale=1.0)
    assert f > 0.0
    assert f <= 1.0


def test_kelly_half_is_smaller():
    full = kelly_fraction(0.70, 0.50, kelly_scale=1.0)
    half = kelly_fraction(0.70, 0.50, kelly_scale=0.5)
    assert half < full


def test_kelly_contracts_basic():
    n = kelly_contracts(
        model_prob=0.70,
        market_price_cents=50,
        bankroll_cents=50_000,
        min_contracts=1,
        max_contracts=10,
        kelly_scale=0.5,
    )
    assert 1 <= n <= 10


def test_kelly_contracts_no_edge():
    n = kelly_contracts(
        model_prob=0.50,
        market_price_cents=50,
        bankroll_cents=50_000,
    )
    assert n == 0


def test_kelly_contracts_respects_max():
    n = kelly_contracts(
        model_prob=0.95,
        market_price_cents=10,
        bankroll_cents=1_000_000,
        max_contracts=5,
    )
    assert n <= 5
