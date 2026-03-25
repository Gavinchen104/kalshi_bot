from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from src.api.binance_ws import BinanceDataStore, fetch_historical_klines, run_binance_ws
from src.api.kalshi_client import KalshiClient
from src.api.ws_client import KalshiWSClient
from src.config import load_settings
from src.execution.order_manager import OrderManager
from src.execution.order_poller import OrderPoller
from src.execution.paper_settler import PaperSettler
from src.execution.router import OrderRouter
from src.monitoring.alerts import send_alert
from src.monitoring.logging import configure_logging, get_logger
from src.portfolio.pnl import PnLTracker
from src.risk.exposure import ExposureBook
from src.risk.kill_switch import KillSwitch
from src.risk.limits import RiskEngine
from src.storage.repository import Repository
from src.strategy.exit_manager import ExitManager
from src.strategy.signals import Strategy
from src.training.auto_retrainer import AutoRetrainer


logger = get_logger("main")


async def run() -> None:
    settings = load_settings()
    configure_logging(settings.app.log_level)
    paper_mode = settings.app.paper_mode or settings.env.bot_paper_mode
    live_allowed = settings.env.bot_allow_live_trading
    if not paper_mode and not live_allowed:
        raise RuntimeError(
            "Live trading is locked. Set BOT_ALLOW_LIVE_TRADING=true to enable real orders."
        )

    client = KalshiClient(
        base_url=settings.env.kalshi_base_url,
        api_key=settings.env.kalshi_api_key,
        api_secret=settings.env.kalshi_api_secret,
    )
    ws = KalshiWSClient(
        ws_url=settings.env.kalshi_ws_url,
        api_key=settings.env.kalshi_api_key,
        api_secret=settings.env.kalshi_api_secret,
        market_ids=settings.markets.allowlist,
    )

    # ── Binance BTC spot feed ─────────────────────────────────────────────
    btc_store = BinanceDataStore(max_candles=settings.binance.history_candles)
    if settings.binance.enabled and not settings.testing.use_mock_data_stream:
        try:
            history = await fetch_historical_klines(limit=settings.binance.history_candles)
            await btc_store.load_history(history)
            logger.info("binance_history_loaded", candles=btc_store.candle_count)
        except Exception as exc:
            logger.warning("binance_history_failed", error=str(exc))

    repository = Repository(settings.storage.db_path)

    strategy = Strategy(
        settings.strategy,
        btc_store=btc_store,
        bankroll_cents=settings.strategy.bankroll_cents,
    )
    router = OrderRouter(settings.execution, strategy=strategy)
    kill_switch = KillSwitch()
    exposure = ExposureBook()
    risk = RiskEngine(settings.risk, exposure=exposure, kill_switch=kill_switch)
    order_manager = OrderManager(
        client,
        paper_mode=paper_mode,
        repository=repository,
        min_order_interval_ms=settings.execution.min_order_interval_ms,
        fee_bps=settings.execution.fee_bps,
        slippage_bps=settings.execution.slippage_bps,
    )
    auto_retrainer = AutoRetrainer(
        config=settings.autotrain,
        strategy_config=settings.strategy,
        repository=repository,
        strategy=strategy,
    )
    exit_manager = ExitManager(
        take_profit_pct=settings.execution.take_profit_pct,
        stop_loss_pct=settings.execution.stop_loss_pct,
    )
    pnl_tracker = PnLTracker()

    last_positions = repository.latest_positions()
    if last_positions:
        pnl_tracker.restore_from_snapshot(last_positions)
        logger.info("positions_restored", count=len(last_positions))
        repository.log_event("positions_restored", payload={"count": len(last_positions)})

    paper_settler = PaperSettler(
        client=client,
        pnl_tracker=pnl_tracker,
        repository=repository,
        check_interval_seconds=300,
    )
    order_poller = OrderPoller(
        client=client,
        repository=repository,
        pnl_tracker=pnl_tracker,
        exposure=exposure,
        poll_interval_seconds=30,
    )
    _snapshot_counter = 0
    _last_candle_save = 0.0

    logger.info(
        "bot_start",
        env=settings.app.env,
        paper_mode=paper_mode,
        live_allowed=live_allowed,
        binance_enabled=settings.binance.enabled,
    )
    repository.log_event(
        "bot_start",
        payload={
            "env": settings.app.env,
            "paper_mode": paper_mode,
            "live_allowed": live_allowed,
            "binance_enabled": settings.binance.enabled,
        },
    )
    if not settings.markets.allowlist and not settings.markets.symbol_filters:
        raise RuntimeError(
            "No market filters configured. Set markets.allowlist or markets.symbol_filters in config/settings.yaml"
        )
    if not settings.testing.use_mock_data_stream:
        await _run_live_preflight(client, ws, repository)

    # Start Binance WS as a background task
    binance_task = None
    if settings.binance.enabled and not settings.testing.use_mock_data_stream:
        binance_task = asyncio.create_task(run_binance_ws(btc_store, settings.binance.ws_url))
        ready = await btc_store.wait_ready(timeout=15.0)
        if ready:
            logger.info("binance_ws_ready", candles=btc_store.candle_count)
        else:
            logger.warning("binance_ws_timeout", candles=btc_store.candle_count)

    try:
        async for state in _select_stream(settings.testing.use_mock_data_stream, ws, settings.testing.mock_data_path):
            if not _is_allowed_market(
                state.market_id,
                allowlist=settings.markets.allowlist,
                symbol_filters=settings.markets.symbol_filters,
            ):
                continue

            mark = state.last_trade_cents
            if mark is None and state.bid_cents is not None and state.ask_cents is not None:
                mark = (state.bid_cents + state.ask_cents) // 2
            if mark is not None:
                pnl_tracker.update_mark(state.market_id, mark)

            repository.log_event(
                "market_state",
                market_id=state.market_id,
                payload={
                    "bid_cents": state.bid_cents,
                    "ask_cents": state.ask_cents,
                    "last_trade_cents": state.last_trade_cents,
                    "bid_size": state.bid_size,
                    "ask_size": state.ask_size,
                    "updated_at": state.updated_at.isoformat(),
                },
            )

            # Periodically persist BTC spot candles to DB for training
            now = time.monotonic()
            if (
                settings.binance.enabled
                and btc_store.candle_count > 0
                and now - _last_candle_save > settings.binance.save_interval_seconds
            ):
                candle_dicts = btc_store.closed_candle_dicts()
                if candle_dicts:
                    repository.save_btc_candles(candle_dicts)
                    _last_candle_save = now

            await auto_retrainer.maybe_retrain()
            if paper_mode:
                await paper_settler.maybe_settle()
            else:
                await order_poller.maybe_poll()

            exit_orders = exit_manager.check_exits(state, pnl_tracker)
            for exit_order in exit_orders:
                exit_resp = await order_manager.execute(exit_order, state=state)
                exit_status = exit_resp.get("status", "")
                repository.log_event(
                    "exit_order",
                    market_id=exit_order.market_id,
                    payload={
                        "side": exit_order.side,
                        "price_cents": exit_order.price_cents,
                        "quantity": exit_order.quantity,
                        "status": exit_status,
                    },
                )
                if exit_status == "paper_filled" and exit_resp.get("fill_price_cents") is not None:
                    fill_price = int(exit_resp["fill_price_cents"])
                    fee_cents = int(exit_resp.get("fee_cents") or 0)
                    pnl_tracker.on_fill(
                        market_id=exit_order.market_id,
                        side=exit_order.side,
                        quantity=exit_order.quantity,
                        fill_price_cents=fill_price,
                        fee_cents=fee_cents,
                    )
                    signed_qty = exit_order.quantity if exit_order.side == "yes" else -exit_order.quantity
                    exposure.apply_fill(exit_order.market_id, signed_qty)
                    logger.info(
                        "exit_filled",
                        market_id=exit_order.market_id,
                        side=exit_order.side,
                        fill_price=fill_price,
                        realized_pnl=pnl_tracker.realized_cents,
                    )
                    repository.save_position_snapshot(
                        positions=pnl_tracker.position_snapshot(),
                        realized_cents=pnl_tracker.realized_cents,
                        unrealized_cents=pnl_tracker.unrealized_cents,
                    )
                    repository.save_pnl_snapshot(
                        total_cents=pnl_tracker.total_cents,
                        realized_cents=pnl_tracker.realized_cents,
                        unrealized_cents=pnl_tracker.unrealized_cents,
                    )

            signal = strategy.compute_signal(state)
            if signal is None:
                continue
            repository.log_event(
                "signal",
                market_id=state.market_id,
                payload={
                    "last_trade": state.last_trade_cents,
                    "reason": signal.reason,
                    "predicted_prob": signal.predicted_prob,
                    "model_name": signal.model_name,
                },
            )

            order = router.signal_to_order(signal, state=state)
            risk.update_session_pnl(pnl_tracker.total_cents)
            result = risk.validate(order, state)
            if not result.allowed:
                logger.info("risk_block", reason=result.reason, market_id=state.market_id)
                repository.log_event(
                    "risk_block",
                    market_id=state.market_id,
                    payload={"reason": result.reason},
                )
                if kill_switch.engaged:
                    send_alert(f"Kill switch engaged: {kill_switch.reason}")
                continue

            response = await order_manager.execute(order, state=state)
            logger.info("order_result", response=response, market_id=state.market_id)
            repository.log_event("order_result", market_id=state.market_id, payload={"response": response})

            if response.get("status") == "paper_filled" and response.get("fill_price_cents") is not None:
                fill_price = int(response["fill_price_cents"])
                fee_cents = int(response.get("fee_cents") or 0)
                pnl_tracker.on_fill(
                    market_id=order.market_id,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price_cents=fill_price,
                    fee_cents=fee_cents,
                )
                signed_qty = order.quantity if order.side == "yes" else -order.quantity
                exposure.apply_fill(order.market_id, signed_qty)
                logger.info(
                    "fill_booked",
                    market_id=order.market_id,
                    side=order.side,
                    fill_price=fill_price,
                    fee_cents=fee_cents,
                    realized_pnl=pnl_tracker.realized_cents,
                    unrealized_pnl=pnl_tracker.unrealized_cents,
                )
                repository.save_position_snapshot(
                    positions=pnl_tracker.position_snapshot(),
                    realized_cents=pnl_tracker.realized_cents,
                    unrealized_cents=pnl_tracker.unrealized_cents,
                )

            repository.save_pnl_snapshot(
                total_cents=pnl_tracker.total_cents,
                realized_cents=pnl_tracker.realized_cents,
                unrealized_cents=pnl_tracker.unrealized_cents,
            )
            _snapshot_counter += 1
            if _snapshot_counter % 10 == 0:
                repository.save_position_snapshot(
                    positions=pnl_tracker.position_snapshot(),
                    realized_cents=pnl_tracker.realized_cents,
                    unrealized_cents=pnl_tracker.unrealized_cents,
                )
    finally:
        if binance_task is not None:
            binance_task.cancel()
        await client.close()


async def _select_stream(
    use_mock_data_stream: bool, ws: KalshiWSClient, mock_data_path: str
) -> AsyncIterator:
    if use_mock_data_stream:
        async for state in ws.stream_from_jsonl(mock_data_path):
            yield state
    else:
        async for state in ws.stream():
            yield state


def _is_allowed_market(market_id: str, allowlist: list[str], symbol_filters: list[str]) -> bool:
    market_id_upper = market_id.upper()
    if allowlist and market_id in allowlist:
        return True
    if symbol_filters:
        return any(token.upper() in market_id_upper for token in symbol_filters)
    return not allowlist


async def _run_live_preflight(
    client: KalshiClient,
    ws: KalshiWSClient,
    repository: Repository,
) -> None:
    logger.info("preflight_start", mode="live_data")
    try:
        await client.get_balance()
        await ws.preflight_auth()
    except Exception as exc:
        repository.log_event("preflight_failed", payload={"error": str(exc)})
        raise RuntimeError(
            "Live data preflight failed. Check KALSHI_API_KEY/KALSHI_API_SECRET format and account API access."
        ) from exc
    repository.log_event("preflight_ok", payload={"mode": "live_data"})
    logger.info("preflight_ok", mode="live_data")


if __name__ == "__main__":
    asyncio.run(run())
