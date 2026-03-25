from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.types import MarketState


class Repository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        db_file = Path(self.db_path)
        if db_file.parent:
            db_file.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    market_id TEXT,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price_cents INTEGER NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT,
                    order_id TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Migrate existing tables that predate the order_id column
            try:
                conn.execute("ALTER TABLE order_events ADD COLUMN order_id TEXT")
            except Exception:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pnl_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_cents INTEGER NOT NULL,
                    realized_cents INTEGER NOT NULL,
                    unrealized_cents INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_json TEXT NOT NULL,
                    realized_cents INTEGER NOT NULL,
                    unrealized_cents INTEGER NOT NULL,
                    total_cents INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS btc_spot_candles (
                    timestamp_ms INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )

    def log_event(
        self,
        event_type: str,
        market_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_events (event_type, market_id, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_type,
                    market_id,
                    json.dumps(payload or {}),
                    _utc_now_iso(),
                ),
            )

    def save_order_event(
        self,
        market_id: str,
        side: str,
        price_cents: int,
        quantity: int,
        status: str,
        response: dict[str, Any] | None = None,
        order_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO order_events
                (market_id, side, price_cents, quantity, status, response_json, order_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    side,
                    price_cents,
                    quantity,
                    status,
                    json.dumps(response or {}),
                    order_id,
                    _utc_now_iso(),
                ),
            )

    def pending_live_orders(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, market_id, side, price_cents, quantity, status, order_id, created_at
                FROM order_events
                WHERE status = 'live_sent' AND order_id IS NOT NULL
                ORDER BY id ASC
                LIMIT 200
                """
            ).fetchall()
        return [
            {
                "id": r[0],
                "market_id": r[1],
                "side": r[2],
                "price_cents": r[3],
                "quantity": r[4],
                "status": r[5],
                "order_id": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    def update_order_status(
        self,
        db_id: int,
        new_status: str,
        fill_price_cents: int | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE order_events SET status = ?, response_json = ? WHERE id = ?",
                (new_status, json.dumps(response or {}), db_id),
            )

    def save_position_snapshot(
        self,
        positions: list[dict[str, Any]],
        realized_cents: int,
        unrealized_cents: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO position_snapshots
                (snapshot_json, realized_cents, unrealized_cents, total_cents, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    json.dumps(positions),
                    realized_cents,
                    unrealized_cents,
                    realized_cents + unrealized_cents,
                    _utc_now_iso(),
                ),
            )

    def save_pnl_snapshot(self, total_cents: int, realized_cents: int, unrealized_cents: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pnl_snapshots (total_cents, realized_cents, unrealized_cents, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (total_cents, realized_cents, unrealized_cents, _utc_now_iso()),
            )

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, market_id, payload_json, created_at
                FROM bot_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "event_type": row[0],
                "market_id": row[1],
                "payload_json": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def recent_orders(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, side, price_cents, quantity, status, response_json, created_at
                FROM order_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "market_id": row[0],
                "side": row[1],
                "price_cents": row[2],
                "quantity": row[3],
                "status": row[4],
                "response_json": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    def pnl_series(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT total_cents, realized_cents, unrealized_cents, created_at
                FROM (
                    SELECT total_cents, realized_cents, unrealized_cents, created_at, id
                    FROM pnl_snapshots
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "total_cents": row[0],
                "realized_cents": row[1],
                "unrealized_cents": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def order_status_counts(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as cnt
                FROM order_events
                GROUP BY status
                ORDER BY cnt DESC
                """
            ).fetchall()
        return [{"status": row[0], "count": row[1]} for row in rows]

    def market_order_counts(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, COUNT(*) as cnt
                FROM order_events
                GROUP BY market_id
                ORDER BY cnt DESC
                """
            ).fetchall()
        return [{"market_id": row[0], "count": row[1]} for row in rows]

    def latest_market_state(self, symbol_token: str = "BTC") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT market_id, payload_json, created_at
                FROM bot_events
                WHERE event_type = 'market_state' AND UPPER(COALESCE(market_id, '')) LIKE ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (f"%{symbol_token.upper()}%",),
            ).fetchone()
        if not row:
            return None
        return {
            "market_id": row[0],
            "payload_json": row[1],
            "created_at": row[2],
        }

    def market_state_series(self, market_like: str = "BTC", limit: int = 50000) -> list[MarketState]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, payload_json, created_at FROM (
                    SELECT market_id, payload_json, created_at, id
                    FROM bot_events
                    WHERE event_type = 'market_state' AND UPPER(COALESCE(market_id, '')) LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (f"%{market_like.upper()}%", limit),
            ).fetchall()
        result: list[MarketState] = []
        for market_id, payload_json, created_at in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            updated_raw = str(payload.get("updated_at") or created_at)
            updated_at = _parse_iso_or_now(updated_raw)
            result.append(
                MarketState(
                    market_id=str(market_id or ""),
                    bid_cents=_to_int_or_none(payload.get("bid_cents")),
                    ask_cents=_to_int_or_none(payload.get("ask_cents")),
                    bid_size=int(payload.get("bid_size", 0) or 0),
                    ask_size=int(payload.get("ask_size", 0) or 0),
                    last_trade_cents=_to_int_or_none(payload.get("last_trade_cents")),
                    updated_at=updated_at,
                )
            )
        return result

    def market_state_series_by_time(
        self, market_like: str = "BTC",
        after_iso: str | None = None, before_iso: str | None = None,
        sample_every_n: int = 1,
    ) -> list[MarketState]:
        """Load states within a time window, optionally sub-sampling every Nth row."""
        clauses = ["event_type = 'market_state'", "UPPER(COALESCE(market_id, '')) LIKE ?"]
        params: list = [f"%{market_like.upper()}%"]
        if after_iso:
            clauses.append("created_at >= ?")
            params.append(after_iso)
        if before_iso:
            clauses.append("created_at <= ?")
            params.append(before_iso)
        where = " AND ".join(clauses)
        sql = f"""
            SELECT market_id, payload_json, created_at
            FROM bot_events
            WHERE {where}
            ORDER BY id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        result: list[MarketState] = []
        for idx, (market_id, payload_json, created_at) in enumerate(rows):
            if sample_every_n > 1 and idx % sample_every_n != 0:
                continue
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            updated_raw = str(payload.get("updated_at") or created_at)
            updated_at = _parse_iso_or_now(updated_raw)
            result.append(
                MarketState(
                    market_id=str(market_id or ""),
                    bid_cents=_to_int_or_none(payload.get("bid_cents")),
                    ask_cents=_to_int_or_none(payload.get("ask_cents")),
                    bid_size=int(payload.get("bid_size", 0) or 0),
                    ask_size=int(payload.get("ask_size", 0) or 0),
                    last_trade_cents=_to_int_or_none(payload.get("last_trade_cents")),
                    updated_at=updated_at,
                )
            )
        return result

    def market_state_count(self, market_like: str = "BTC") -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM bot_events
                WHERE event_type = 'market_state' AND UPPER(COALESCE(market_id, '')) LIKE ?
                """,
                (f"%{market_like.upper()}%",),
            ).fetchone()
        return int(row[0] if row else 0)

    def position_series(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT snapshot_json, realized_cents, unrealized_cents, total_cents, created_at
                FROM position_snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for snap_json, realized, unrealized, total, created_at in rows:
            try:
                positions = json.loads(snap_json or "[]")
            except Exception:
                positions = []
            result.append(
                {
                    "positions": positions,
                    "realized_cents": realized,
                    "unrealized_cents": unrealized,
                    "total_cents": total,
                    "created_at": created_at,
                }
            )
        return result

    def latest_positions(self) -> list[dict[str, Any]]:
        rows = self.position_series(limit=1)
        if not rows:
            return []
        return rows[0]["positions"]

    def fill_rate_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as cnt
                FROM order_events
                GROUP BY status
                """
            ).fetchall()
        stats: dict[str, int] = {}
        for status, cnt in rows:
            stats[status] = cnt
        return stats

    def recent_filled_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return filled orders with fill details extracted from response_json."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, side, price_cents, quantity, status, response_json, created_at
                FROM order_events
                WHERE status IN ('paper_filled', 'live_filled')
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for market_id, side, price_cents, quantity, status, response_json, created_at in rows:
            try:
                resp = json.loads(response_json or "{}")
            except Exception:
                resp = {}
            fill_price = resp.get("fill_price_cents", price_cents)
            fee_cents = resp.get("fee_cents", 0) or 0
            slippage_cents = resp.get("slippage_cents", 0) or 0
            result.append({
                "time": created_at,
                "market": market_id,
                "action": f"BUY {side.upper()}",
                "contracts": quantity,
                "order_price_cents": price_cents,
                "fill_price_cents": fill_price,
                "fee_cents": fee_cents,
                "slippage_cents": slippage_cents,
                "net_cost_cents": (fill_price or price_cents) * quantity + fee_cents,
                "status": status,
            })
        return result

    def save_btc_candles(self, candles: list[dict]) -> int:
        """Upsert closed BTC spot candles.  Returns number of rows inserted."""
        if not candles:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO btc_spot_candles
                (timestamp_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(c["timestamp_ms"]),
                        float(c["open"]),
                        float(c["high"]),
                        float(c["low"]),
                        float(c["close"]),
                        float(c["volume"]),
                    )
                    for c in candles
                ],
            )
        return len(candles)

    def btc_candle_series(self, limit: int = 100_000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_ms, open, high, low, close, volume
                FROM btc_spot_candles
                ORDER BY timestamp_ms ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "timestamp_ms": r[0],
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in rows
        ]

    def btc_candle_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM btc_spot_candles").fetchone()
        return int(row[0] if row else 0)

    def recent_model_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, payload_json, created_at
                FROM bot_events
                WHERE event_type IN (
                    'model_retrained',
                    'model_retrain_skipped',
                    'model_retrain_failed',
                    'model_retrain_paused'
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for event_type, payload_json, created_at in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            result.append(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "created_at": created_at,
                }
            )
        return result

    def model_metrics_series(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json, created_at
                FROM bot_events
                WHERE event_type = 'model_retrained'
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for payload_json, created_at in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except Exception:
                payload = {}
            result.append(
                {
                    "created_at": created_at,
                    "val_accuracy": float(payload.get("val_accuracy", 0.0) or 0.0),
                    "val_logloss": float(payload.get("val_logloss", 0.0) or 0.0),
                    "rows": int(payload.get("rows", 0) or 0),
                    "loaded": bool(payload.get("loaded", False)),
                    "wf_mean_accuracy": payload.get("wf_mean_accuracy"),
                    "wf_mean_logloss": payload.get("wf_mean_logloss"),
                    "wf_beat_baseline_count": payload.get("wf_beat_baseline_count"),
                }
            )
        return result


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso_or_now(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(tz=timezone.utc)


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None

