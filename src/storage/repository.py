from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.types import MarketState, ProbEstimate, ProposedOrder, Signal


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Repository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── writes ────────────────────────────────────────────────────────────
    def save_market_state(self, state: MarketState) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO market_state(market_id, bid_cents, ask_cents, bid_size, ask_size,
                                         last_trade_cents, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.market_id, state.bid_cents, state.ask_cents,
                    state.bid_size, state.ask_size, state.last_trade_cents,
                    state.updated_at.isoformat(),
                ),
            )

    def save_prob_estimate(
        self,
        est: ProbEstimate,
        market_state: MarketState | None = None,
    ) -> None:
        bid = market_state.bid_cents if market_state else None
        ask = market_state.ask_cents if market_state else None
        mid = (bid + ask) // 2 if (isinstance(bid, int) and isinstance(ask, int)) else None
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO prob_estimate(market_id, prob, horizon_seconds, spot_usd,
                                          vol_annualized, source, computed_at,
                                          market_yes_ask_cents, market_yes_bid_cents,
                                          market_mid_cents)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    est.market_id, est.prob, est.horizon_seconds, est.spot_usd,
                    est.vol_annualized, est.source, est.computed_at.isoformat(),
                    ask, bid, mid,
                ),
            )

    def save_signal(self, sig: Signal) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO signal(market_id, side, our_prob, market_prob, edge,
                                   fair_price_cents, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sig.market_id, sig.side, sig.our_prob, sig.market_prob, sig.edge,
                 sig.fair_price_cents, sig.reason),
            )

    def save_paper_order(
        self,
        order: ProposedOrder,
        status: str,
        fill_price_cents: int | None = None,
        fee_cents: int | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO paper_order(market_id, side, price_cents, quantity, status,
                                       fill_price_cents, fee_cents)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (order.market_id, order.side, order.price_cents, order.quantity,
                 status, fill_price_cents, fee_cents),
            )
            return cur.lastrowid

    def save_candles(self, candles: list[dict]) -> int:
        if not candles:
            return 0
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO coinbase_candle VALUES (?, ?, ?, ?, ?, ?)",
                [(c_["timestamp_ms"], c_["open"], c_["high"], c_["low"], c_["close"], c_["volume"])
                 for c_ in candles],
            )
        return len(candles)

    def upsert_live_spot(self, price: float) -> None:
        """Write the current Coinbase spot price to a single-row table.
        Called frequently (~500ms) so the dashboard can render sub-second updates."""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO live_spot(id, price, updated_at) "
                "VALUES (1, ?, datetime('now'))",
                (price,),
            )

    def get_live_spot(self) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT price, updated_at FROM live_spot WHERE id = 1"
            ).fetchone()
        if not row:
            return None
        return {"price": float(row[0]), "updated_at": row[1]}

    def save_pnl(self, total: int, realized: int, unrealized: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO pnl_snapshot(total_cents, realized_cents, unrealized_cents) VALUES (?, ?, ?)",
                (total, realized, unrealized),
            )

    def save_position(self, market_id: str, net_qty: int, avg_entry: float, realized: int) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO position_snapshot(market_id, net_quantity, avg_entry_cents, realized_pnl_cents)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    net_quantity = excluded.net_quantity,
                    avg_entry_cents = excluded.avg_entry_cents,
                    realized_pnl_cents = excluded.realized_pnl_cents,
                    updated_at = datetime('now')
                """,
                (market_id, net_qty, avg_entry, realized),
            )

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO event(event_type, payload_json) VALUES (?, ?)",
                (event_type, json.dumps(payload or {})),
            )

    def save_calibration(
        self,
        window_size: int,
        brier: float | None,
        log_loss: float | None,
        n_samples: int,
        bins: list[dict],
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO calibration_snapshot(window_size, brier_score, log_loss, n_samples, bin_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (window_size, brier, log_loss, n_samples, json.dumps(bins)),
            )

    # ── reads (used by dashboard / measurement) ───────────────────────────
    def latest_market_state(self, market_like: str = "BTC") -> dict | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT market_id, bid_cents, ask_cents, last_trade_cents, updated_at
                FROM market_state
                WHERE UPPER(market_id) LIKE ?
                ORDER BY id DESC LIMIT 1
                """,
                (f"%{market_like.upper()}%",),
            ).fetchone()
        if not row:
            return None
        keys = ("market_id", "bid_cents", "ask_cents", "last_trade_cents", "updated_at")
        return dict(zip(keys, row))

    def recent_candles(self, limit: int = 300) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT timestamp_ms, open, high, low, close, volume FROM coinbase_candle "
                "ORDER BY timestamp_ms DESC LIMIT ?",
                (limit,),
            ).fetchall()
        rows.reverse()
        return [
            {"timestamp_ms": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
            for r in rows
        ]

    def candle_timestamps(self, start_ms: int, end_ms: int) -> list[int]:
        """Existing candle timestamps within [start_ms, end_ms), ascending.
        Used for gap detection (cheap: timestamps only, not full OHLCV)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT timestamp_ms FROM coinbase_candle "
                "WHERE timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms ASC",
                (start_ms, end_ms),
            ).fetchall()
        return [int(r[0]) for r in rows]

    def recent_prob_estimates(self, limit: int = 500) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT market_id, prob, horizon_seconds, spot_usd, vol_annualized,
                       source, computed_at, market_mid_cents, market_yes_ask_cents,
                       market_yes_bid_cents
                FROM prob_estimate ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = ("market_id", "prob", "horizon_seconds", "spot_usd", "vol_annualized",
                "source", "computed_at", "market_mid_cents", "market_yes_ask_cents",
                "market_yes_bid_cents")
        return [dict(zip(keys, r)) for r in rows]

    def prob_estimates_for_settlement(self) -> list[dict]:
        """All (market_id, prob) rows oldest-first, for outcome settlement.

        Unlike recent_prob_estimates, this is NOT recency-limited: settlement
        must look back at older estimates whose markets have since closed."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT market_id, prob FROM prob_estimate ORDER BY id ASC"
            ).fetchall()
        return [{"market_id": r[0], "prob": float(r[1])} for r in rows]

    def recent_signals(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT market_id, side, our_prob, market_prob, edge, fair_price_cents, "
                "reason, created_at FROM signal ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("market_id", "side", "our_prob", "market_prob", "edge", "fair_price_cents",
                "reason", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def recent_paper_orders(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT market_id, side, price_cents, quantity, status, fill_price_cents, "
                "fee_cents, created_at FROM paper_order ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("market_id", "side", "price_cents", "quantity", "status", "fill_price_cents",
                "fee_cents", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def pnl_series(self, limit: int = 1000) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT total_cents, realized_cents, unrealized_cents, created_at "
                "FROM pnl_snapshot ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("total_cents", "realized_cents", "unrealized_cents", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def latest_positions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT market_id, net_quantity, avg_entry_cents, realized_pnl_cents, updated_at "
                "FROM position_snapshot WHERE net_quantity != 0",
            ).fetchall()
        keys = ("market_id", "net_quantity", "avg_entry_cents", "realized_pnl_cents", "updated_at")
        return [dict(zip(keys, r)) for r in rows]

    def latest_calibration(self) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT window_size, brier_score, log_loss, n_samples, bin_json, created_at "
                "FROM calibration_snapshot ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "window_size": row[0], "brier_score": row[1], "log_loss": row[2],
            "n_samples": row[3], "bin_json": row[4], "created_at": row[5],
        }
