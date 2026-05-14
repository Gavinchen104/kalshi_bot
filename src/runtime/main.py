from __future__ import annotations

import asyncio
import time

from src.config import load_settings
from src.data.coinbase_ws import CoinbaseStore, fetch_history, run_ws as run_coinbase_ws
from src.data.kalshi_rest import KalshiClient
from src.data.kalshi_ws import KalshiWSClient
from src.execution.paper_executor import PaperExecutor
from src.measurement.reporter import settle_and_snapshot
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
    )
    strategy = EdgeStrategy(settings.strategy)
    kill = KillSwitch()
    risk = RiskEngine(settings.risk, kill_switch=kill)
    executor = PaperExecutor(settings.execution)
    pnl = PnLTracker()

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

    last_candle_persist = 0.0
    last_calibration = 0.0

    try:
        async for state in kalshi_ws.stream():
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
            repo.save_prob_estimate(est, state)

            # Periodic housekeeping that must run regardless of whether a
            # tradeable signal fires this iteration.
            now = time.monotonic()
            if now - last_candle_persist > 30.0:
                repo.save_candles(coinbase_store.closed_candle_dicts())
                last_candle_persist = now
            if now - last_calibration > 120.0:
                try:
                    settle_and_snapshot(
                        repo,
                        window=settings.measurement.calibration_window,
                        n_bins=settings.measurement.calibration_bins,
                    )
                except Exception as exc:
                    logger.warning("calibration_failed", error=str(exc))
                last_calibration = now

            signal = strategy.evaluate(est, state)
            if signal is None:
                continue
            repo.save_signal(signal)

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
                continue

            result = executor.execute(order, state)
            repo.save_paper_order(
                order, status=result.get("status", "unknown"),
                fill_price_cents=result.get("fill_price_cents"),
                fee_cents=result.get("fee_cents"),
            )
            if result.get("status") == "paper_filled":
                fill_price = int(result["fill_price_cents"])
                fee = int(result.get("fee_cents") or 0)
                pnl.on_fill(order.market_id, order.side, order.quantity, fill_price, fee)
                signed = order.quantity if order.side == "yes" else -order.quantity
                risk.apply_fill(order.market_id, signed)
                for pos in pnl.position_snapshot():
                    repo.save_position(
                        pos["market_id"], pos["net_quantity"],
                        pos["avg_entry_cents"], pos["realized_pnl_cents"],
                    )

            repo.save_pnl(pnl.total_cents, pnl.realized_cents, pnl.unrealized_cents)
    finally:
        coinbase_task.cancel()
        try:
            await coinbase_task
        except (asyncio.CancelledError, Exception):
            pass


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
