from src.portfolio.ledger import PositionLedger


def test_ledger_opens_long_yes_position() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=2, fill_price_cents=50)
    pos = ledger.position_for("MKT-1")
    assert pos is not None
    assert pos.net_quantity == 2
    assert pos.avg_entry_cents == 50.0
    assert pos.realized_pnl_cents == 0


def test_ledger_closes_long_and_books_profit() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=2, fill_price_cents=40)
    realized = ledger.on_fill("MKT-1", "no", quantity=2, fill_price_cents=40)
    # Closing YES long via NO buy: effective close price = 100 - 40 = 60
    # PnL = (60 - 40) * 2 = 40 cents
    assert realized == 40
    assert ledger.position_for("MKT-1").net_quantity == 0


def test_ledger_deducts_fees() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=1, fill_price_cents=50)
    realized = ledger.on_fill("MKT-1", "no", quantity=1, fill_price_cents=50, fee_cents=3)
    # Close at 100-50=50, same as entry → realized before fees = 0, after fees = -3
    assert realized == -3
    assert ledger.total_realized_cents == -3


def test_ledger_mark_to_market_unrealized() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=5, fill_price_cents=40)
    unrealized = ledger.mark_to_market("MKT-1", mark_price_cents=50)
    # (50 - 40) * 5 = 50 cents
    assert unrealized == 50


def test_ledger_snapshot_excludes_flat_zero_pnl() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=1, fill_price_cents=50)
    ledger.on_fill("MKT-1", "no", quantity=1, fill_price_cents=50)
    # Flat with zero realized PnL → excluded from snapshot
    assert ledger.snapshot() == []


def test_ledger_snapshot_includes_closed_with_pnl() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=1, fill_price_cents=40)
    ledger.on_fill("MKT-1", "no", quantity=1, fill_price_cents=40)
    snap = ledger.snapshot()
    assert len(snap) == 1
    assert snap[0]["realized_pnl_cents"] == 20


def test_ledger_no_buy_creates_short_yes_position() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-2", "no", quantity=3, fill_price_cents=30)
    pos = ledger.position_for("MKT-2")
    assert pos is not None
    assert pos.net_quantity == -3
    assert pos.avg_entry_cents == 70.0  # 100 - 30


def test_ledger_settle_yes_win() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=4, fill_price_cents=60)
    realized = ledger.settle_position("MKT-1", settlement_cents=100)
    # (100 - 60) * 4 = 160 cents
    assert realized == 160
    assert ledger.position_for("MKT-1").is_flat


def test_ledger_settle_yes_loss() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-1", "yes", quantity=2, fill_price_cents=70)
    realized = ledger.settle_position("MKT-1", settlement_cents=0)
    # (0 - 70) * 2 = -140 cents
    assert realized == -140
    assert ledger.position_for("MKT-1").is_flat


def test_ledger_restore_from_snapshot() -> None:
    ledger = PositionLedger()
    ledger.on_fill("MKT-A", "yes", quantity=3, fill_price_cents=45)

    snap = ledger.snapshot()
    ledger2 = PositionLedger()
    ledger2.restore_from_snapshot(snap)

    pos = ledger2.position_for("MKT-A")
    assert pos is not None
    assert pos.net_quantity == 3
    assert pos.avg_entry_cents == 45.0
