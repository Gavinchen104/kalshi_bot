from __future__ import annotations

import asyncio
import time

from src.config import load_settings
from src.data.coinbase_ws import (
    CoinbaseStore,
    backfill_range,
    fetch_history,
    find_candle_gaps,
    run_ws as run_coinbase_ws,
)
from src.data.kalshi_rest import KalshiClient
from src.data.kalshi_ws import KalshiWSClient
from src.execution.paper_executor import PaperExecutor
from src.measurement.reporter import settle_and_snapshot
from src.monitoring.alerts import AlertManager
from src.monitoring.logging import configure_logging, get_logger
from src.portfolio.pnl import PnLTracker
from src.pricing.pricer import CoinbasePricer
from src.risk.engine import RiskEngine
from src.risk.kill_switch import KillSwitch
from src.storage.repository import Repository
from src.strategy.edge import EdgeStrategy
from src.strategy.kelly import kelly_contracts
from src.types import ProposedOrder


logger = get_logger("runtime.main")


async def _fast_spot_writer(repo, coinbase_store) -> None:
    """Upsert the latest Coinbase spot price every 500ms so the dashboard
    sees sub-second updates without waiting for minute candle closes."""
    while True:
        await asyncio.sleep(0.5)
        try:
            price = coinbase_store.latest_price
            if price is not None:
                repo.upsert_live_spot(price)
        except Exception as exc:
            logger.warning("fast_spot_writer_failed", error=str(exc))


async def _periodic_housekeeping(
    repo,
    coinbase_store,
    settings,
    state_marker: dict,
    alerts: AlertManager,
) -> None:
    """Persist Coinbase candles + run calibration on a fixed cadence,
    independent of whether Kalshi WS is producing market state updates.

    Also emits a heartbeat log so a stalled main loop is visible in logs."""
    while True:
        await asyncio.sleep(10)
        try:
            candles = coinbase_store.closed_candle_dicts()
            if candles:
                n = repo.save_candles(candles)
                logger.debug("housekeeping_candles_saved", n=n)
        except Exception as exc:
            logger.warning("housekeeping_candles_failed", error=str(exc))

        # Heartbeat: surfaces "alive but Kalshi silent" state.
        now = time.monotonic()
        secs_since_kalshi = now - state_marker.get("last_kalshi_ts", now)
        logger.info(
            "heartbeat",
            secs_since_kalshi_state=int(secs_since_kalshi),
            coinbase_candles=coinbase_store.candle_count,
            spot=coinbase_store.latest_price,
        )
        if secs_since_kalshi > settings.monitoring.stalled_feed_seconds:
            await alerts.emit(
                "stalled_kalshi_feed",
                "Kalshi market-state feed has stalled",
                {"seconds_since_kalshi_state": int(secs_since_kalshi)},
            )

        window_ms = settings.monitoring.data_quality_window_hours * 3600 * 1000
        end_ms = int(time.time() * 1000)
        coverage = repo.candle_coverage(end_ms - window_ms, end_ms)
        if coverage["coverage"] < settings.monitoring.data_quality_min_coverage:
            await alerts.emit(
                "coinbase_data_quality",
                "Coinbase candle coverage is below the Phase 3 X1 bar",
                coverage,
            )

        # Calibration runs every ~2 minutes.
        if (now - state_marker.get("last_calibration_ts", 0)) > 120.0:
            try:
                report = settle_and_snapshot(
                    repo,
                    window=settings.measurement.calibration_window,
                    n_bins=settings.measurement.calibration_bins,
                )
                state_marker["last_calibration_ts"] = now
                if (
                    report is not None
                    and report["n_samples"] >= settings.monitoring.calibration_drift_min_samples
                    and report["brier"] is not None
                    and report["brier"] > settings.monitoring.calibration_drift_brier_threshold
                ):
                    await alerts.emit(
                        "calibration_drift",
                        "Rolling calibration Brier exceeded the Phase 3 X1 threshold",
                        {"brier": report["brier"], "n_samples": report["n_samples"]},
                    )
            except Exception as exc:
                logger.warning("calibration_failed", error=str(exc))


async def _periodic_gap_repair(repo, settings) -> None:
    """Re-run gap detection + backfill on a fixed cadence so candle continuity
    self-heals after a transient Coinbase WS disconnect, without waiting for a
    process restart. Cheap when there are no gaps (one timestamps query, no
    REST calls). Best-effort."""
    interval = max(60, settings.coinbase.gap_repair_interval_minutes * 60)
    while True:
        await asyncio.sleep(interval)
        try:
            n = await _backfill_candle_gaps(
                repo, settings,
                lookback_hours=settings.coinbase.gap_repair_lookback_hours,
            )
            if n:
                logger.info("gap_repair_filled", n=n)
        except Exception as exc:
            logger.warning("gap_repair_failed", error=str(exc))


async def _backfill_candle_gaps(repo, settings, lookback_hours: int | None = None) -> int:
    """Detect missing minute ranges in coinbase_candle over the last
    `lookback_hours` (defaults to settings.coinbase.backfill_max_hours for the
    startup full sweep; periodic repair passes a lighter recent window) and
    backfill them from Coinbase REST. Best-effort: never blocks startup."""
    hours = lookback_hours or settings.coinbase.backfill_max_hours
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - hours * 3600 * 1000
    existing = repo.candle_timestamps(start_ms, now_ms)
    gaps = find_candle_gaps(existing, start_ms, now_ms)
    total = 0
    for g_start, g_end in gaps:
        try:
            candles = await backfill_range(
                settings.coinbase.rest_candles_url,
                settings.coinbase.product_id,
                g_start, g_end,
            )
        except Exception as exc:
            logger.warning(
                "candle_backfill_window_failed",
                start=g_start, end=g_end, error=str(exc),
            )
            continue
        if candles:
            total += repo.save_candles([
                {"timestamp_ms": c.timestamp_ms, "open": c.open, "high": c.high,
                 "low": c.low, "close": c.close, "volume": c.volume}
                for c in candles
            ])
    return total


async def run() -> None:
    settings = load_settings()
    configure_logging(settings.app.log_level)
    logger.info("bot_start", env=settings.app.env)

    repo = Repository(settings.storage.db_path)
    coinbase_store = CoinbaseStore(max_candles=settings.coinbase.history_candles + 200)

    try:
        history = await fetch_history(
            settings.coinbase.rest_candles_url,
            settings.coinbase.product_id,
            limit=settings.coinbase.history_candles,
        )
        await coinbase_store.load_history(history)
        logger.info("coinbase_history_loaded", n=coinbase_store.candle_count)
    except Exception as exc:
        logger.warning("coinbase_history_failed", error=str(exc))

    # Repair candle continuity for settlement/backtest (best-effort).
    try:
        n_filled = await _backfill_candle_gaps(repo, settings)
        logger.info("candle_gaps_backfilled", n=n_filled)
    except Exception as exc:
        logger.warning("candle_backfill_failed", error=str(exc))

    coinbase_task = asyncio.create_task(
        run_coinbase_ws(coinbase_store, settings.coinbase.ws_url, settings.coinbase.product_id)
    )
    ready = await coinbase_store.wait_ready(timeout=20.0)
    if not ready:
        logger.warning("coinbase_not_ready_yet", candles=coinbase_store.candle_count)

    pricer = CoinbasePricer(
        vol_window_minutes=settings.pricer.vol_window_minutes,
        vol_floor=settings.pricer.vol_floor_annualized,
        vol_ceiling=settings.pricer.vol_ceiling_annualized,
        min_horizon_seconds=settings.pricer.min_horizon_seconds,
        bracket_width_usd_default=settings.pricer.bracket_width_usd_default,
        vol_mode=settings.pricer.vol_mode,
        vol_window_floor_min=settings.pricer.vol_window_floor_min,
        vol_window_cap_min=settings.pricer.vol_window_cap_min,
        vol_long_floor_days=settings.pricer.vol_long_floor_days,
        ewma_half_life_min=settings.pricer.ewma_half_life_min,
    )
    strategy = EdgeStrategy(settings.strategy)
    kill = KillSwitch()
    risk = RiskEngine(settings.risk, kill_switch=kill)
    executor = PaperExecutor(settings.execution)
    pnl = PnLTracker()
    alerts = AlertManager(settings.monitoring, webhook_url=settings.env.bot_alert_webhook_url)

    # Fetch the currently-open BTC market tickers so we can explicitly subscribe.
    # Kalshi's WS v2 ticker channel sends nothing without an explicit market list.
    kalshi_rest = KalshiClient(
        base_url=settings.kalshi.base_url,
        api_key=settings.env.kalshi_api_key,
        api_secret=settings.env.kalshi_api_secret,
    )
    try:
        btc_tickers = await kalshi_rest.list_open_btc_markets()
        logger.info("kalshi_btc_markets_loaded", n=len(btc_tickers))
    except Exception as exc:
        logger.warning("kalshi_btc_markets_failed", error=str(exc))
        btc_tickers = []
    finally:
        await kalshi_rest.close()

    kalshi_ws = KalshiWSClient(
        ws_url=settings.kalshi.ws_url,
        api_key=settings.env.kalshi_api_key,
        api_secret=settings.env.kalshi_api_secret,
        ticker_regex=settings.kalshi.ticker_regex,
        market_tickers=btc_tickers,
    )

    # Shared marker so the housekeeping task can detect a stalled Kalshi feed.
    state_marker: dict = {"last_kalshi_ts": time.monotonic(), "last_calibration_ts": 0.0}
    housekeeping_task = asyncio.create_task(
        _periodic_housekeeping(repo, coinbase_store, settings, state_marker, alerts)
    )
    fast_spot_task = asyncio.create_task(_fast_spot_writer(repo, coinbase_store))
    gap_repair_task = asyncio.create_task(_periodic_gap_repair(repo, settings))

    try:
        async for state in kalshi_ws.stream():
            state_marker["last_kalshi_ts"] = time.monotonic()
            repo.save_market_state(state)
            mark = state.last_trade_cents or (
                (state.bid_cents + state.ask_cents) // 2
                if state.bid_cents is not None and state.ask_cents is not None
                else None
            )
            if mark is not None:
                pnl.update_mark(state.market_id, mark)

            spot = coinbase_store.latest_price
            closes = coinbase_store.closes()
            if spot is None or closes.size < settings.pricer.vol_window_minutes + 1:
                continue

            est = pricer.price(state.market_id, spot, closes)
            if est is None:
                continue
            risk.update_realized_vol(est.vol_annualized)
            repo.save_prob_estimate(est, state)

            signal = strategy.evaluate(est, state)
            if signal is None:
                continue
            repo.save_signal(signal)
            risk.update_market_probability(signal.market_id, signal.our_prob)

            price = state.ask_cents if signal.side == "yes" else (100 - (state.bid_cents or 0))
            qty = kelly_contracts(
                our_prob=signal.our_prob if signal.side == "yes" else (1 - signal.our_prob),
                price_cents=price,
                bankroll_cents=settings.sizing.bankroll_cents,
                kelly_fraction=settings.sizing.kelly_fraction,
                max_contracts=settings.sizing.max_contracts_per_trade,
            )
            if qty <= 0:
                continue
            order = ProposedOrder(
                market_id=signal.market_id, side=signal.side,
                price_cents=price, quantity=qty,
            )
            risk.update_session_pnl(pnl.total_cents)
            verdict = risk.validate(order, state)
            if not verdict.allowed:
                repo.log_event("risk_block", {"market_id": order.market_id, "reason": verdict.reason})
                if (
                    verdict.reason.startswith("kill_switch:")
                    or verdict.reason in {
                        "daily_loss_limit", "drawdown_limit", "max_tail_short_exposure",
                    }
                ):
                    await alerts.emit(
                        f"risk_{verdict.reason}",
                        "Risk engine blocked an order",
                        {"market_id": order.market_id, "reason": verdict.reason},
                    )
                continue

            result = executor.execute(order, state)
            repo.save_paper_order(
                order, status=result.get("status", "unknown"),
                fill_price_cents=result.get("fill_price_cents"),
                fee_cents=result.get("fee_cents"),
                filled_quantity=result.get("filled_quantity"),
            )
            if result.get("status") in {"paper_filled", "paper_partially_filled"}:
                fill_price = int(result["fill_price_cents"])
                fee = int(result.get("fee_cents") or 0)
                filled_qty = int(result.get("filled_quantity") or order.quantity)
                pnl.on_fill(order.market_id, order.side, filled_qty, fill_price, fee)
                signed = filled_qty if order.side == "yes" else -filled_qty
                risk.apply_fill(order.market_id, signed, prob=signal.our_prob)
                for pos in pnl.position_snapshot():
                    repo.save_position(
                        pos["market_id"], pos["net_quantity"],
                        pos["avg_entry_cents"], pos["realized_pnl_cents"],
                    )

            repo.save_pnl(pnl.total_cents, pnl.realized_cents, pnl.unrealized_cents)
    finally:
        for task in (coinbase_task, housekeeping_task, fast_spot_task, gap_repair_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
