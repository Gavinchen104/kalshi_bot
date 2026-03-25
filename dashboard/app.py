from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.repository import Repository


st.set_page_config(page_title="Kalshi Bot Dashboard", layout="wide")
st.title("Kalshi Bot Dashboard")

default_db_path = "data/bot.db"
db_path = st.sidebar.text_input("SQLite DB path", value=default_db_path)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (sec)", min_value=1, max_value=30, value=5)

db_exists = Path(db_path).exists()
if not db_exists:
    st.warning(f"Database not found at `{db_path}`. Start the bot first to create it.")
    st.stop()

repo = Repository(db_path)


def _load_model_baseline() -> dict:
    model_path = PROJECT_ROOT / "data" / "models" / "btc_15m_model.json"
    if not model_path.exists():
        return {}
    try:
        raw = json.loads(model_path.read_text(encoding="utf-8"))
        return raw
    except Exception:
        return {}


def render_panels() -> None:
    orders = repo.recent_orders(limit=300)
    events = repo.recent_events(limit=300)
    pnl = repo.pnl_series(limit=1000)
    status_counts = repo.order_status_counts()
    market_counts = repo.market_order_counts()
    latest_btc_market = repo.latest_market_state(symbol_token="BTC")
    model_events = repo.recent_model_events(limit=200)
    model_metrics = repo.model_metrics_series(limit=500)
    latest_positions = repo.latest_positions()
    fill_stats = repo.fill_rate_stats()
    latest_event_ts = events[0]["created_at"] if events else "N/A"

    total_orders = len(orders)
    paper_filled = fill_stats.get("paper_filled", 0)
    paper_unfilled = fill_stats.get("paper_unfilled", 0)
    skipped = fill_stats.get("skipped_rate_limited", 0)
    latest_pnl_row = pnl[-1] if pnl else {}
    latest_realized = latest_pnl_row.get("realized_cents", 0) / 100
    latest_unrealized = latest_pnl_row.get("unrealized_cents", 0) / 100
    latest_total_pnl = latest_pnl_row.get("total_cents", 0) / 100

    # ── Heartbeat ────────────────────────────────────────────────────────
    hb1, hb2, hb3 = st.columns(3)
    hb1.metric("Dashboard Refresh (UTC)", datetime.now(tz=timezone.utc).strftime("%H:%M:%S"))
    hb2.metric("Latest Bot Event", latest_event_ts)
    btc_candle_count = repo.btc_candle_count()
    hb3.metric("BTC Spot Candles (DB)", btc_candle_count)

    # ── BTC Spot Price ────────────────────────────────────────────────────
    btc_candles = repo.btc_candle_series(limit=5)
    if btc_candles:
        latest_candle = btc_candles[-1]
        st.subheader("BTC Spot Price (Binance)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Price", f"${latest_candle['close']:,.2f}")
        s2.metric("High", f"${latest_candle['high']:,.2f}")
        s3.metric("Low", f"${latest_candle['low']:,.2f}")
        s4.metric("Volume", f"{latest_candle['volume']:,.1f} BTC")

    # ── Portfolio Equity Curve ────────────────────────────────────────────
    STARTING_BALANCE = 500.00
    current_balance = STARTING_BALANCE + latest_total_pnl
    balance_return_pct = (latest_total_pnl / STARTING_BALANCE) * 100 if STARTING_BALANCE else 0.0

    st.subheader("Portfolio Value")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Starting Balance", f"${STARTING_BALANCE:,.2f}")
    b2.metric("Current Balance", f"${current_balance:,.2f}")
    b3.metric("Total PnL", f"${latest_total_pnl:+.2f}")
    b4.metric("Return", f"{balance_return_pct:+.2f}%")
    b5.metric("Realized", f"${latest_realized:+.2f}")
    b6.metric("Unrealized", f"${latest_unrealized:+.2f}")

    if pnl:
        equity_chart = [
            {
                "timestamp": row["created_at"],
                "balance": STARTING_BALANCE + row["total_cents"] / 100,
                "realized_equity": STARTING_BALANCE + row["realized_cents"] / 100,
            }
            for row in pnl
        ]
        st.line_chart(equity_chart, x="timestamp", y=["balance", "realized_equity"])

        peak = STARTING_BALANCE
        drawdown_chart = []
        for row in pnl:
            bal = STARTING_BALANCE + row["total_cents"] / 100
            if bal > peak:
                peak = bal
            dd = ((bal - peak) / peak) * 100 if peak > 0 else 0.0
            drawdown_chart.append({"timestamp": row["created_at"], "drawdown_pct": dd})
        st.subheader("Drawdown (%)")
        st.area_chart(drawdown_chart, x="timestamp", y="drawdown_pct")
    else:
        st.info("No PnL snapshots yet.")

    st.subheader("PnL Breakdown")
    p1, p2 = st.columns(2)
    p1.metric("Total Orders", total_orders)
    p2.metric("Paper Filled", paper_filled)

    # ── Trade Log ─────────────────────────────────────────────────────────
    st.subheader("Trade Log")
    filled_trades = repo.recent_filled_trades(limit=200)
    if filled_trades:
        trade_rows = []
        for t in filled_trades:
            trade_rows.append({
                "Time (UTC)": t["time"],
                "Market": t["market"],
                "Action": t["action"],
                "Contracts": t["contracts"],
                "Order Price": f'{t["order_price_cents"]}c',
                "Fill Price": f'{t["fill_price_cents"]}c' if t["fill_price_cents"] else "-",
                "Fee": f'{t["fee_cents"]}c',
                "Slippage": f'{t["slippage_cents"]}c',
                "Net Cost": f'${t["net_cost_cents"] / 100:.2f}',
                "Status": t["status"],
            })
        st.dataframe(trade_rows, width="stretch")
    else:
        st.info("No filled trades yet.")

    # ── Open Positions ────────────────────────────────────────────────────
    st.subheader("Open Positions")
    if latest_positions:
        pos_rows = []
        for p in latest_positions:
            qty = p.get("net_quantity", 0)
            entry = p.get("avg_entry_cents", 0)
            real_pnl = p.get("realized_pnl_cents", 0)
            pos_rows.append({
                "Market": p.get("market_id", ""),
                "Side": "LONG YES" if qty > 0 else ("SHORT YES" if qty < 0 else "FLAT"),
                "Contracts": abs(qty),
                "Avg Entry": f'{entry:.1f}c',
                "Realized PnL": f'${real_pnl / 100:+.2f}',
            })
        st.dataframe(pos_rows, width="stretch")
    else:
        st.info("No open positions.")

    # ── Current BTC Market (Kalshi) ───────────────────────────────────────
    st.subheader("Current BTC Market (Kalshi)")
    if latest_btc_market:
        payload = {}
        try:
            payload = json.loads(latest_btc_market["payload_json"])
        except Exception:
            pass
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Market", latest_btc_market["market_id"] or "N/A")
        c2.metric("Bid", payload.get("bid_cents", "N/A"))
        c3.metric("Ask", payload.get("ask_cents", "N/A"))
        c4.metric("Last", payload.get("last_trade_cents", "N/A"))
        c5.metric("Updated", latest_btc_market["created_at"])
    else:
        st.info("No BTC market snapshot yet.")

    # ── Fill Quality ──────────────────────────────────────────────────────
    st.subheader("Fill Quality")
    total_attempted = paper_filled + paper_unfilled
    fill_rate = (paper_filled / total_attempted * 100) if total_attempted > 0 else 0.0
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Fill Rate", f"{fill_rate:.1f}%")
    f2.metric("Filled", paper_filled)
    f3.metric("Unfilled", paper_unfilled)
    f4.metric("Rate-Limited Skips", skipped)

    # ── Model Feature Importance ──────────────────────────────────────────
    model_baseline = _load_model_baseline()
    if model_baseline.get("type") == "gbm":
        st.subheader("Model Feature Importance (LightGBM)")
        try:
            import lightgbm as lgb
            booster_file = model_baseline.get("booster_path", "btc_15m_model.lgb")
            booster_path = str(PROJECT_ROOT / "data" / "models" / booster_file)
            booster = lgb.Booster(model_file=booster_path)
            feat_names = model_baseline.get("feature_names", [])
            imp = booster.feature_importance(importance_type="gain")
            total = float(sum(imp)) or 1.0
            imp_data = sorted(
                [{"feature": n, "importance": round(float(v) / total * 100, 1)}
                 for n, v in zip(feat_names, imp)],
                key=lambda x: x["importance"],
                reverse=True,
            )
            st.bar_chart(imp_data, x="feature", y="importance")
        except Exception:
            st.info("Could not load feature importance.")
    else:
        st.subheader("Feature Drift vs Training Baseline")
        if model_baseline.get("feature_names") and latest_btc_market:
            recent_states = repo.market_state_series(market_like="BTC", limit=500)
            if len(recent_states) >= 10:
                from src.strategy.ml_model import FEATURE_NAMES, build_feature_map
                import numpy as np

                feature_rows: list[list[float]] = []
                for i in range(20, len(recent_states)):
                    window = recent_states[max(0, i - 20): i + 1]
                    fm = build_feature_map(window)
                    if fm:
                        feature_rows.append([fm.get(n, 0.0) for n in FEATURE_NAMES])

                if feature_rows:
                    arr = np.array(feature_rows, dtype=float)
                    current_means = arr.mean(axis=0).tolist()
                    training_means = model_baseline.get("means", [])
                    training_stds = model_baseline.get("stds", [])
                    drift_rows = []
                    for i, name in enumerate(FEATURE_NAMES):
                        t_mean = training_means[i] if i < len(training_means) else 0.0
                        t_std = training_stds[i] if i < len(training_stds) else 1.0
                        c_mean = current_means[i] if i < len(current_means) else 0.0
                        z_score = (c_mean - t_mean) / (t_std if t_std != 0 else 1.0)
                        drift_rows.append({
                            "feature": name,
                            "training_mean": round(t_mean, 4),
                            "current_mean": round(c_mean, 4),
                            "z_score": round(z_score, 2),
                        })
                    st.dataframe(drift_rows, width="stretch")

    # ── Recent Orders ─────────────────────────────────────────────────────
    st.subheader("Recent Orders (all)")
    if orders:
        preview = []
        for r in orders[:100]:
            try:
                resp = json.loads(r.get("response_json") or "{}")
            except Exception:
                resp = {}
            preview.append({
                "Time": r["created_at"],
                "Market": r["market_id"],
                "Side": r["side"].upper(),
                "Order Price": f'{r["price_cents"]}c',
                "Qty": r["quantity"],
                "Fill Price": f'{resp.get("fill_price_cents", "-")}c' if resp.get("fill_price_cents") else "-",
                "Fee": f'{resp.get("fee_cents", 0)}c' if resp.get("fee_cents") else "-",
                "Status": r["status"],
            })
        st.dataframe(preview, width="stretch")
    else:
        st.info("No orders yet.")

    # ── Recent Events ─────────────────────────────────────────────────────
    st.subheader("Recent Events")
    if events:
        preview_events = []
        for row in events[:100]:
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                payload = row["payload_json"]
            preview_events.append({
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "market_id": row["market_id"],
                "payload": payload,
            })
        st.dataframe(preview_events, width="stretch")
    else:
        st.info("No events yet.")

    # ── Order Breakdown ───────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.subheader("Order Status Breakdown")
        if status_counts:
            st.bar_chart(status_counts, x="status", y="count")
        else:
            st.info("No order status data yet.")
    with right:
        st.subheader("Orders by Market")
        if market_counts:
            st.bar_chart(market_counts, x="market_id", y="count")
        else:
            st.info("No market activity yet.")

    # ── Adaptive Learning (Strategy Explorer) ─────────────────────────────
    st.subheader("Adaptive Strategy Explorer")
    import sqlite3
    try:
        _exp_conn = sqlite3.connect(db_path)
        _exp_rows = _exp_conn.execute(
            "SELECT recipe_name, wf_accuracy, wf_baseline, edge, rows, created_at "
            "FROM exploration_runs ORDER BY id DESC LIMIT 50"
        ).fetchall()
        _exp_conn.close()
    except Exception:
        _exp_rows = []

    if _exp_rows:
        champion_row = max(_exp_rows, key=lambda r: (r[3] or -999))
        has_champion = (champion_row[3] or 0) > 0
        explored = len(_exp_rows)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Recipes Explored", explored)
        e2.metric("Champion", champion_row[0] if has_champion else "None yet")
        e3.metric("Best Edge", f"{(champion_row[3] or 0)*100:+.2f}%")
        e4.metric("Status", "TRADING" if has_champion else "EXPLORING")

        exp_table = []
        for name, wf, base, edge, rows_count, ts in _exp_rows:
            edge_val = edge or 0
            exp_table.append({
                "Time": ts,
                "Recipe": name,
                "WF Accuracy": f"{(wf or 0):.3f}",
                "Baseline": f"{(base or 0):.3f}",
                "Edge": f"{edge_val*100:+.2f}%",
                "Rows": rows_count or 0,
                "Beat?": "YES" if edge_val > 0 else "no",
            })
        st.dataframe(exp_table, width="stretch")

        edges = [{"recipe": r[0], "edge_pct": (r[3] or 0) * 100} for r in reversed(_exp_rows)]
        st.line_chart(edges, x="recipe", y="edge_pct")
    else:
        st.info("Strategy explorer has not run yet. Waiting for first training cycle...")

    # ── Model Training ────────────────────────────────────────────────────
    st.subheader("Model Training")
    latest_model_event = model_events[0] if model_events else None
    total_retrained = sum(1 for e in model_events if e["event_type"] == "model_retrained")
    total_skipped = sum(1 for e in model_events if e["event_type"] == "model_retrain_skipped")
    total_failed = sum(1 for e in model_events if e["event_type"] == "model_retrain_failed")
    total_paused = sum(1 for e in model_events if e["event_type"] == "model_retrain_paused")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Retrains", total_retrained)
    m2.metric("Skipped", total_skipped)
    m3.metric("Failed", total_failed)
    m4.metric("Paused", total_paused)
    m5.metric("Latest Status", latest_model_event["event_type"] if latest_model_event else "N/A")

    if model_metrics:
        training_chart = [
            {
                "timestamp": row["created_at"],
                "val_accuracy": row["val_accuracy"],
                "val_logloss": row["val_logloss"],
                "wf_mean_accuracy": row.get("wf_mean_accuracy", None),
            }
            for row in model_metrics
        ]
        chart_cols = ["val_accuracy", "val_logloss"]
        if any(r["wf_mean_accuracy"] for r in training_chart):
            chart_cols.append("wf_mean_accuracy")
        st.line_chart(training_chart, x="timestamp", y=chart_cols)
    else:
        st.info("No completed retraining runs yet.")

    if model_events:
        training_table = []
        for item in model_events[:50]:
            p = item["payload"]
            training_table.append({
                "created_at": item["created_at"],
                "event_type": item["event_type"],
                "recipe": p.get("recipe", "-"),
                "rows": p.get("rows"),
                "val_accuracy": p.get("val_accuracy"),
                "wf_accuracy": p.get("wf_mean_accuracy"),
                "wf_baseline": p.get("wf_mean_baseline"),
                "beats_baseline": "YES" if p.get("beats_baseline") == 1.0 else "no",
                "champion": p.get("champion", "-"),
                "explored": p.get("explored_total"),
                "best_edge": f'{p.get("best_edge_pct", 0)}%',
                "error": p.get("error"),
            })
        st.dataframe(training_table, width="stretch")
    else:
        st.info("No model training events yet.")

    # ── Backtest (on-demand) ──────────────────────────────────────────────
    st.subheader("Backtest (on-demand)")
    model_path = PROJECT_ROOT / "data" / "models" / "btc_15m_model.json"
    if model_path.exists():
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    from src.backtest.engine import run_backtest
                    bt = run_backtest(
                        db_path=db_path,
                        model_path=str(model_path),
                    )
                    summary = bt.summary()
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Total PnL", f"${summary['total_pnl_dollars']:+.2f}")
                    r2.metric("Trades", int(summary["n_trades"]))
                    r3.metric("Win Rate", f"{summary['win_rate']:.1%}")
                    r4.metric("Sharpe", f"{summary['sharpe_ratio']:.2f}")
                    if bt.equity_curve:
                        st.line_chart(
                            [{"step": i, "pnl_cents": v} for i, v in enumerate(bt.equity_curve)],
                            x="step",
                            y="pnl_cents",
                        )
                    st.metric("Max Drawdown", f"${summary['max_drawdown_cents'] / 100:.2f}")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
    else:
        st.info("No trained model found. Train the model first to run a backtest.")


if hasattr(st, "fragment"):
    run_every = f"{refresh_seconds}s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def live_fragment() -> None:
        render_panels()

    live_fragment()
else:
    render_panels()
    if auto_refresh:
        st.caption(f"Auto refresh every {refresh_seconds}s.")
        if st_autorefresh is not None:
            st_autorefresh(interval=refresh_seconds * 1000, key="backend_autorefresh")
        else:
            st.markdown(
                f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
                unsafe_allow_html=True,
            )
