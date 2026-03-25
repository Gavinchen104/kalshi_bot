from src.storage.repository import Repository


def test_repository_writes_and_reads(tmp_path) -> None:
    db_file = tmp_path / "bot.db"
    repo = Repository(str(db_file))

    repo.log_event("bot_start", payload={"paper_mode": True})
    repo.save_order_event(
        market_id="MKT-1",
        side="yes",
        price_cents=51,
        quantity=1,
        status="paper_accepted",
        response={"status": "paper_accepted"},
    )
    repo.save_pnl_snapshot(total_cents=125, realized_cents=100, unrealized_cents=25)

    events = repo.recent_events()
    orders = repo.recent_orders()
    pnl = repo.pnl_series()

    assert len(events) == 1
    assert events[0]["event_type"] == "bot_start"
    assert len(orders) == 1
    assert orders[0]["market_id"] == "MKT-1"
    assert len(pnl) == 1
    assert pnl[0]["total_cents"] == 125


def test_repository_latest_btc_market_state(tmp_path) -> None:
    db_file = tmp_path / "bot.db"
    repo = Repository(str(db_file))

    repo.log_event("market_state", market_id="KXETHD-TEST", payload={"last_trade_cents": 40})
    repo.log_event("market_state", market_id="KXBTCD-TEST", payload={"last_trade_cents": 51})

    latest = repo.latest_market_state("BTC")
    assert latest is not None
    assert latest["market_id"] == "KXBTCD-TEST"


def test_repository_model_event_views(tmp_path) -> None:
    db_file = tmp_path / "bot.db"
    repo = Repository(str(db_file))

    repo.log_event("model_retrain_skipped", payload={"reason": "insufficient_data", "rows": 50})
    repo.log_event(
        "model_retrained",
        payload={"rows": 500, "val_accuracy": 0.61, "val_logloss": 0.67, "loaded": True},
    )
    repo.log_event("model_retrain_failed", payload={"error": "boom"})

    recent = repo.recent_model_events(limit=10)
    metrics = repo.model_metrics_series(limit=10)

    assert len(recent) == 3
    assert recent[0]["event_type"] == "model_retrain_failed"
    assert len(metrics) == 1
    assert metrics[0]["rows"] == 500

