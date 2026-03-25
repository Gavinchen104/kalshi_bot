"""
Historical backtest engine.

Replays stored Kalshi market states and (optionally) BTC spot candles through
the prediction model with realistic transaction costs.  Produces a performance
report including PnL, Sharpe ratio, max drawdown, win rate, and per-trade log.

Usage (CLI):
    python -m src.backtest.engine --db-path data/bot.db --model-path data/models/btc_15m_model.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.storage.repository import Repository
from src.strategy.features import ENHANCED_FEATURE_NAMES, build_enhanced_features
from src.strategy.ml_model import load_model
from src.types import MarketState


@dataclass
class BacktestTrade:
    market_id: str
    side: str
    entry_price_cents: int
    exit_price_cents: int
    quantity: int
    fee_cents: int
    pnl_cents: int
    entry_time: datetime
    exit_time: datetime | None = None


@dataclass
class BacktestResult:
    total_pnl_cents: int = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    max_drawdown_cents: int = 0
    sharpe_ratio: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_win_cents(self) -> float:
        wins = [t.pnl_cents for t in self.trades if t.pnl_cents > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss_cents(self) -> float:
        losses = [t.pnl_cents for t in self.trades if t.pnl_cents <= 0]
        return float(np.mean(losses)) if losses else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "total_pnl_cents": self.total_pnl_cents,
            "total_pnl_dollars": self.total_pnl_cents / 100,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win_cents": round(self.avg_win_cents, 1),
            "avg_loss_cents": round(self.avg_loss_cents, 1),
            "max_drawdown_cents": self.max_drawdown_cents,
            "sharpe_ratio": round(self.sharpe_ratio, 3),
        }


def run_backtest(
    db_path: str,
    model_path: str,
    market_like: str = "BTC",
    target_regex: str = ".*BTC.*",
    lookback: int = 20,
    horizon: int = 5,
    fee_bps: int = 50,
    slippage_bps: int = 20,
    min_confidence: float = 0.03,
    take_profit_pct: float = 0.25,
    stop_loss_pct: float = 0.15,
    limit: int = 100_000,
) -> BacktestResult:
    import re

    repo = Repository(db_path)
    model = load_model(model_path)
    market_re = re.compile(target_regex, re.IGNORECASE)

    states = repo.market_state_series(market_like=market_like, limit=limit)
    btc_candles = repo.btc_candle_series(limit=100_000)
    has_spot = len(btc_candles) >= 30

    grouped: dict[str, list[MarketState]] = defaultdict(list)
    for s in states:
        if market_re.search(s.market_id):
            grouped[s.market_id].append(s)

    result = BacktestResult()
    cumulative_pnl = 0
    peak_pnl = 0
    pnl_per_step: list[float] = []

    for market_id, series in grouped.items():
        windows: deque[MarketState] = deque(maxlen=lookback + 15)
        open_pos: dict[str, dict] | None = None

        for i, state in enumerate(series):
            windows.append(state)

            if open_pos is not None:
                mark = _mark(state)
                if mark is not None:
                    entry = open_pos["entry_price"]
                    if open_pos["side"] == "yes":
                        unreal_per = mark - entry
                    else:
                        unreal_per = entry - mark

                    tp_thresh = entry * take_profit_pct
                    sl_thresh = entry * stop_loss_pct

                    if unreal_per >= tp_thresh or unreal_per <= -sl_thresh or i == len(series) - 1:
                        qty = open_pos["quantity"]
                        slip = max(1, int(mark * slippage_bps / 10_000))
                        if open_pos["side"] == "yes":
                            exit_price = mark - slip
                        else:
                            exit_price = mark + slip
                        fee = max(1, int(exit_price * qty * fee_bps / 10_000))

                        if open_pos["side"] == "yes":
                            gross = (exit_price - entry) * qty
                        else:
                            gross = (entry - exit_price) * qty
                        pnl = gross - open_pos["entry_fee"] - fee

                        trade = BacktestTrade(
                            market_id=market_id,
                            side=open_pos["side"],
                            entry_price_cents=entry,
                            exit_price_cents=exit_price,
                            quantity=qty,
                            fee_cents=open_pos["entry_fee"] + fee,
                            pnl_cents=pnl,
                            entry_time=open_pos["entry_time"],
                            exit_time=state.updated_at,
                        )
                        result.trades.append(trade)
                        result.n_trades += 1
                        if pnl > 0:
                            result.n_wins += 1
                        else:
                            result.n_losses += 1
                        cumulative_pnl += pnl
                        pnl_per_step.append(float(pnl))
                        result.equity_curve.append(cumulative_pnl)
                        if cumulative_pnl > peak_pnl:
                            peak_pnl = cumulative_pnl
                        dd = peak_pnl - cumulative_pnl
                        if dd > result.max_drawdown_cents:
                            result.max_drawdown_cents = dd
                        open_pos = None
                continue

            if len(windows) < lookback:
                continue

            spot_ctx = _spot_context(state.updated_at, btc_candles) if has_spot else None
            if spot_ctx:
                fmap = build_enhanced_features(list(windows), *spot_ctx)
            else:
                fmap = build_enhanced_features(list(windows))
            if fmap is None:
                continue

            prob = model.predict_proba(fmap)
            confidence = abs(prob - 0.5)
            if confidence < min_confidence:
                continue

            fair = int(round(prob * 100))
            if state.ask_cents is None or state.bid_cents is None:
                continue

            if fair > state.ask_cents:
                side = "yes"
                entry_price = state.ask_cents
            elif fair < state.bid_cents:
                side = "no"
                entry_price = 100 - state.bid_cents
            else:
                continue

            slip = max(1, int(entry_price * slippage_bps / 10_000))
            entry_price += slip
            entry_fee = max(1, int(entry_price * fee_bps / 10_000))

            open_pos = {
                "side": side,
                "entry_price": entry_price,
                "entry_fee": entry_fee,
                "quantity": 1,
                "entry_time": state.updated_at,
            }

    result.total_pnl_cents = cumulative_pnl

    if len(pnl_per_step) >= 2:
        arr = np.array(pnl_per_step, dtype=float)
        std = float(np.std(arr))
        if std > 0:
            result.sharpe_ratio = float(np.mean(arr) / std) * np.sqrt(len(arr))

    return result


def _mark(state: MarketState) -> int | None:
    if state.last_trade_cents is not None:
        return state.last_trade_cents
    if state.bid_cents is not None and state.ask_cents is not None:
        return (state.bid_cents + state.ask_cents) // 2
    return None


def _spot_context(
    ts: datetime,
    candles: list[dict],
    window: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ts_ms = int(ts.timestamp() * 1000)
    relevant = [c for c in candles if c["timestamp_ms"] <= ts_ms]
    if len(relevant) < 30:
        return None
    relevant = relevant[-window:]
    closes = np.array([c["close"] for c in relevant], dtype=float)
    volumes = np.array([c["volume"] for c in relevant], dtype=float)
    ohlcv = np.array(
        [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in relevant],
        dtype=float,
    )
    return closes, volumes, ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical backtest.")
    parser.add_argument("--db-path", default="data/bot.db")
    parser.add_argument("--model-path", default="data/models/btc_15m_model.json")
    parser.add_argument("--market-like", default="BTC")
    parser.add_argument("--target-regex", default=".*BTC.*")
    parser.add_argument("--fee-bps", type=int, default=50)
    parser.add_argument("--slippage-bps", type=int, default=20)
    parser.add_argument("--take-profit-pct", type=float, default=0.25)
    parser.add_argument("--stop-loss-pct", type=float, default=0.15)
    args = parser.parse_args()

    result = run_backtest(
        db_path=args.db_path,
        model_path=args.model_path,
        market_like=args.market_like,
        target_regex=args.target_regex,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
    )
    print("\n=== Backtest Results ===")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")
    if result.trades:
        print(f"\nLast 10 trades:")
        for t in result.trades[-10:]:
            print(
                f"  {t.market_id} {t.side} entry={t.entry_price_cents}c "
                f"exit={t.exit_price_cents}c pnl={t.pnl_cents:+d}c"
            )


if __name__ == "__main__":
    main()
