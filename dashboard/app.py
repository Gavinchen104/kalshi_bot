from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.repository import Repository


st.set_page_config(page_title="Kalshi BTC Research Bench", layout="wide")
st.title("Kalshi BTC 15m — Research Bench")

db_path = st.sidebar.text_input("SQLite DB path", value="data/bot.db")
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (sec)", 1, 30, 5)

if not Path(db_path).exists():
    st.warning(
        f"Database not found at `{db_path}`. Start the bot in another terminal "
        "(`python -m src.runtime.main`) — this page will auto-refresh once it appears."
    )
    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
        def _wait() -> None:
            if Path(db_path).exists():
                st.rerun()
            else:
                st.caption(f"Polling for `{db_path}`…")
        _wait()
    st.stop()

repo = Repository(db_path)


def render() -> None:
    latest_kalshi = repo.latest_market_state(market_like="BTC")
    candles = repo.recent_candles(limit=300)
    estimates = repo.recent_prob_estimates(limit=200)
    signals = repo.recent_signals(limit=200)
    orders = repo.recent_paper_orders(limit=200)
    pnl = repo.pnl_series(limit=1000)
    positions = repo.latest_positions()
    cal = repo.latest_calibration()

    hb1, hb2, hb3 = st.columns(3)
    hb1.metric("Refresh (UTC)", datetime.now(tz=timezone.utc).strftime("%H:%M:%S"))
    hb2.metric("BTC Candles in DB", len(candles))
    hb3.metric("Prob Estimates Logged", len(estimates))

    # ── Live View: Kalshi vs Coinbase ───────────────────────────────────
    st.subheader("Live View — Kalshi (left) vs Coinbase BTC (right)")
    left, right = st.columns(2)

    with left:
        st.markdown("### Kalshi BTC Market")
        if latest_kalshi:
            ts = latest_kalshi.get("updated_at", "—")
            st.caption(f"`{latest_kalshi.get('market_id')}` · updated {ts}")
            bid = latest_kalshi.get("bid_cents")
            ask = latest_kalshi.get("ask_cents")
            last = latest_kalshi.get("last_trade_cents")
            spread = (ask - bid) if (isinstance(ask, int) and isinstance(bid, int)) else None
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Bid", f"{bid}c" if bid is not None else "—")
            k2.metric("Ask", f"{ask}c" if ask is not None else "—")
            k3.metric("Last", f"{last}c" if last is not None else "—")
            k4.metric("Spread", f"{spread}c" if spread is not None else "—")
        else:
            st.info("No Kalshi market data yet.")

        st.markdown("**Recent prob estimates vs market**")
        if estimates:
            pe_rows = [
                {
                    "computed_at": e["computed_at"],
                    "market": e["market_id"],
                    "our_prob": f'{e["prob"]:.3f}',
                    "mkt_mid_c": e["market_mid_cents"],
                    "vol_ann": f'{e["vol_annualized"]:.2f}',
                    "horizon_s": int(e["horizon_seconds"]),
                }
                for e in estimates[:50]
            ]
            st.dataframe(pe_rows, width="stretch", height=260)
        else:
            st.info("No prob estimates yet — strategy needs Coinbase vol + a Kalshi market.")

    with right:
        st.markdown("### Coinbase BTC-USD")
        if candles:
            latest = candles[-1]
            prev = candles[-2]["close"] if len(candles) > 1 else latest["open"]
            delta = latest["close"] - prev
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${latest['close']:,.2f}", f"{delta:+,.2f}")
            c2.metric("High", f"${latest['high']:,.2f}")
            c3.metric("Low", f"${latest['low']:,.2f}")
            c4.metric("Volume", f"{latest['volume']:,.2f} BTC")

            chart = [
                {
                    "time": datetime.fromtimestamp(c["timestamp_ms"] / 1000, tz=timezone.utc).isoformat(),
                    "close": c["close"], "high": c["high"], "low": c["low"],
                }
                for c in candles
            ]
            st.line_chart(chart, x="time", y=["close", "high", "low"], height=260)
        else:
            st.info("No Coinbase candles yet.")

    st.divider()

    # ── Edge & Signals ─────────────────────────────────────────────────
    st.subheader("Edge: Our Prob vs Market")
    if estimates:
        scatter = []
        for e in estimates[:200]:
            mid_c = e.get("market_mid_cents")
            if mid_c is None:
                continue
            scatter.append({"market_prob": mid_c / 100.0, "our_prob": e["prob"]})
        if scatter:
            st.scatter_chart(scatter, x="market_prob", y="our_prob", height=260)
            st.caption("Points above the y=x diagonal → we think YES is underpriced; below → overpriced.")

    st.subheader("Recent Signals")
    if signals:
        st.dataframe(
            [{
                "created_at": s["created_at"], "market": s["market_id"], "side": s["side"].upper(),
                "our": f'{s["our_prob"]:.3f}', "mkt": f'{s["market_prob"]:.3f}',
                "edge": f'{s["edge"]:+.3f}', "fair_c": s["fair_price_cents"],
            } for s in signals[:100]],
            width="stretch", height=240,
        )
    else:
        st.info("No signals yet.")

    # ── Calibration ─────────────────────────────────────────────────────
    st.subheader("Calibration (settled predictions)")
    if cal:
        m1, m2, m3 = st.columns(3)
        m1.metric("Samples", cal["n_samples"])
        m2.metric("Brier", f'{cal["brier"]:.4f}' if cal["brier"] is not None else "—")
        m3.metric("Log Loss", f'{cal["log_loss"]:.4f}' if cal["log_loss"] is not None else "—")
        st.caption("Brier baseline for 50/50 random = 0.25; perfect = 0.0. Lower is better.")
        try:
            bins = json.loads(cal["bin_json"])
            rows = [{
                "bin": f'{b["lo"]:.2f}-{b["hi"]:.2f}',
                "n": b["n"],
                "mean_pred": b["mean_pred"],
                "emp_freq": b["emp_freq"],
            } for b in bins]
            st.dataframe(rows, width="stretch")
        except Exception:
            pass
    else:
        st.info("Not enough settled predictions yet (need at least one full 15-min window after startup).")

    # ── Portfolio & Orders ──────────────────────────────────────────────
    st.subheader("Portfolio")
    if pnl:
        latest = pnl[-1]
        p1, p2, p3 = st.columns(3)
        p1.metric("Total PnL", f'${latest["total_cents"]/100:+.2f}')
        p2.metric("Realized", f'${latest["realized_cents"]/100:+.2f}')
        p3.metric("Unrealized", f'${latest["unrealized_cents"]/100:+.2f}')
        st.line_chart(
            [{"t": r["created_at"], "total": r["total_cents"]/100} for r in pnl],
            x="t", y="total", height=200,
        )
    if positions:
        st.markdown("**Open positions**")
        st.dataframe(positions, width="stretch")

    st.subheader("Recent Paper Orders")
    if orders:
        st.dataframe(orders[:100], width="stretch", height=240)
    else:
        st.info("No paper orders yet.")


if hasattr(st, "fragment"):
    run_every = f"{refresh_seconds}s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def live_fragment() -> None:
        render()

    live_fragment()
else:
    render()
