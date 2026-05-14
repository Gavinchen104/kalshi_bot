from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.repository import Repository


st.set_page_config(
    page_title="KALSHI BTC TERMINAL",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Bloomberg-style theme ────────────────────────────────────────────────────
AMBER = "#ff9500"
GREEN = "#00ff66"
RED = "#ff3b30"
BLUE = "#3399ff"
DIM = "#888888"
BG = "#0a0a0a"
PANEL = "#111111"
GRID = "#1f1f1f"

st.markdown(
    f"""
<style>
.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}

/* Typography */
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
div[data-testid="stMetricDelta"], .stDataFrame, .stDataFrame * {{
    font-family: 'Menlo','Monaco',monospace !important;
}}
.stDataFrame {{ font-size: 0.78rem !important; }}
h1, h2, h3, h4 {{
    font-family: 'Menlo','Monaco',monospace !important;
    color: {AMBER} !important;
    letter-spacing: 0.08em;
}}
h1 {{ font-size: 1.4rem !important; }}
h2 {{ font-size: 1.0rem !important; }}
h3 {{ font-size: 0.85rem !important; }}

.bb-header {{
    color: {AMBER};
    font-family: 'Menlo',monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    border-bottom: 1px solid {AMBER};
    margin: 0.4rem 0 0.3rem 0;
    padding-bottom: 0.15rem;
    text-transform: uppercase;
}}
.bb-tick {{ font-family: 'Menlo',monospace; font-size: 1.3rem; font-weight: 600; }}
.bb-sub {{ font-family: 'Menlo',monospace; font-size: 0.75rem; color: {DIM}; }}
.bb-pos {{ color: {GREEN}; }}
.bb-neg {{ color: {RED}; }}
.bb-neu {{ color: {DIM}; }}
.bb-amber {{ color: {AMBER}; }}
hr {{ border-color: {GRID} !important; margin: 0.4rem 0 !important; }}
.block-container {{ padding-top: 0.6rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }}

/* ─── Liveness: pulse indicator ─── */
.pulse-dot {{
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: {GREEN};
    box-shadow: 0 0 8px {GREEN};
    animation: pulse 1.2s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}}
@keyframes pulse {{
    0%   {{ opacity: 0.4; transform: scale(0.85); }}
    50%  {{ opacity: 1.0; transform: scale(1.10); }}
    100% {{ opacity: 0.4; transform: scale(0.85); }}
}}

/* ─── Liveness: tick flash on price change ─── */
@keyframes flash-up {{
    0%   {{ background-color: rgba(0,255,102,0.35); }}
    100% {{ background-color: transparent; }}
}}
@keyframes flash-down {{
    0%   {{ background-color: rgba(255,59,48,0.35); }}
    100% {{ background-color: transparent; }}
}}
.tick-up   {{ animation: flash-up 1.0s ease-out; }}
.tick-down {{ animation: flash-down 1.0s ease-out; }}

/* ─── Ticker tape (scrolling marquee) ─── */
.ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background: {PANEL};
    border-top: 1px solid {GRID};
    border-bottom: 1px solid {GRID};
    padding: 6px 0;
    margin-bottom: 0.6rem;
}}
.ticker-content {{
    display: inline-block;
    white-space: nowrap;
    padding-left: 100%;
    animation: ticker-scroll 70s linear infinite;
    font-family: 'Menlo',monospace;
    font-size: 0.85rem;
    color: #cccccc;
}}
.ticker-content:hover {{ animation-play-state: paused; }}
@keyframes ticker-scroll {{
    0%   {{ transform: translate3d(0, 0, 0); }}
    100% {{ transform: translate3d(-100%, 0, 0); }}
}}
.ticker-item {{ margin-right: 2.2rem; }}
.ticker-sep {{ color: {AMBER}; margin: 0 0.6rem; }}

/* ─── New-row highlight in signal feed ─── */
.fresh-row-banner {{
    background: linear-gradient(90deg, rgba(255,149,0,0.18), transparent 60%);
    border-left: 3px solid {AMBER};
    padding: 4px 8px;
    margin: 2px 0;
    font-family: 'Menlo',monospace;
    font-size: 0.8rem;
    animation: flash-up 1.2s ease-out;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar (compact controls) ──────────────────────────────────────────────
db_path = st.sidebar.text_input("DB path", value="data/bot.db")
refresh_seconds = st.sidebar.slider("Refresh (s)", 1, 10, 1)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
ticker_tape = st.sidebar.checkbox("Ticker tape", value=True)

if not Path(db_path).exists():
    st.error(f"DB not found at `{db_path}`. Start the bot: `python -m src.runtime.main`")
    st.stop()

repo = Repository(db_path)


# ── Helpers ─────────────────────────────────────────────────────────────────
def color_for(x: float) -> str:
    if x is None or pd.isna(x):
        return DIM
    return GREEN if x > 0 else (RED if x < 0 else DIM)


def fmt_money(cents: int | float | None) -> str:
    if cents is None:
        return "—"
    return f"${cents/100:+,.2f}"


def realized_vol(closes: np.ndarray, window: int) -> float | None:
    if closes is None or len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    r = np.diff(np.log(tail))
    if r.size == 0:
        return None
    s = float(np.std(r, ddof=1))
    return s * np.sqrt(365 * 24 * 60)


def plotly_layout(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Menlo, Monaco, monospace", color="#cccccc", size=11),
        margin=dict(l=40, r=10, t=20, b=30),
        height=height, showlegend=True,
        legend=dict(font=dict(size=10, color="#bbbbbb"), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=1.08, x=0),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color="#888888"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color="#888888"),
    )
    return fig


def sparkline(values: list[float], color: str = AMBER, height: int = 36) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=1.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=height, showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def market_watch_rows(limit: int = 25) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT pe.market_id, pe.prob, pe.horizon_seconds, pe.vol_annualized,
                   pe.market_mid_cents, pe.market_yes_ask_cents, pe.market_yes_bid_cents,
                   pe.spot_usd, pe.computed_at
            FROM prob_estimate pe
            INNER JOIN (
                SELECT market_id, MAX(id) AS max_id FROM prob_estimate GROUP BY market_id
            ) latest ON pe.id = latest.max_id
            ORDER BY ABS(pe.prob - pe.market_mid_cents/100.0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        keys = ("market_id", "prob", "horizon_seconds", "vol_annualized",
                "market_mid_cents", "ask_cents", "bid_cents", "spot_usd", "computed_at")
        return [dict(zip(keys, r)) for r in rows]
    finally:
        conn.close()


def signal_rate_per_min(signals: list[dict]) -> float:
    """Compute signals/min over the actual recent window. Caps span at 60s minimum
    to avoid divide-by-tiny when bursts of signals share the same second."""
    if len(signals) < 2:
        return 0.0
    try:
        t_last = datetime.fromisoformat(signals[0]["created_at"].replace(" ", "T"))
        t_first = datetime.fromisoformat(signals[-1]["created_at"].replace(" ", "T"))
        span_sec = max(60.0, (t_last - t_first).total_seconds())
        return len(signals) / (span_sec / 60.0)
    except Exception:
        return 0.0


# ── Liveness state (persists across fragment re-renders) ────────────────────
if "prev_btc" not in st.session_state:
    st.session_state.prev_btc = None
if "prev_signal_count" not in st.session_state:
    st.session_state.prev_signal_count = 0
if "prev_fill_count" not in st.session_state:
    st.session_state.prev_fill_count = 0


# ── Render ──────────────────────────────────────────────────────────────────
def render() -> None:
    candles = repo.recent_candles(limit=2000)
    estimates = repo.recent_prob_estimates(limit=2000)
    signals = repo.recent_signals(limit=500)
    orders = repo.recent_paper_orders(limit=100)
    pnl = repo.pnl_series(limit=2000)
    positions = repo.latest_positions()
    cal = repo.latest_calibration()
    watch = market_watch_rows(limit=25)

    now_utc = datetime.now(tz=timezone.utc)
    closes = np.array([c["close"] for c in candles], dtype=float) if candles else np.array([])
    # Live spot updates faster than minute candles — read it for the BTC SPOT panel.
    live_spot = repo.get_live_spot()

    # ═══ TICKER TAPE (top scrolling marquee) ═══════════════════════════════
    if ticker_tape and signals:
        items = []
        for s in signals[:25]:
            ts = s["created_at"].split(" ")[-1].split(".")[0] if " " in s["created_at"] else s["created_at"][-12:-7]
            short_id = s["market_id"].replace("KXBTC", "")
            edge_cents = s["edge"] * 100
            side_col = GREEN if s["side"] == "yes" else RED
            items.append(
                f"<span class='ticker-item'>"
                f"<span class='bb-amber'>{ts}</span> "
                f"<span style='color:#ddd'>{short_id}</span> "
                f"<span style='color:{side_col}'>{s['side'].upper()}</span> "
                f"<span style='color:{AMBER}'>{edge_cents:+.1f}¢</span>"
                f"</span><span class='ticker-sep'>•</span>"
            )
        st.markdown(
            f"<div class='ticker-wrap'><div class='ticker-content'>{''.join(items)*2}</div></div>",
            unsafe_allow_html=True,
        )

    # ═══ HEADER ═════════════════════════════════════════════════════════════
    st.markdown(
        f"<h1><span class='pulse-dot'></span>KALSHI BTC TERMINAL "
        f"<span class='bb-sub'>· {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"· refresh {refresh_seconds}s · {len(estimates)} estimates · {len(signals)} signals</span></h1>",
        unsafe_allow_html=True,
    )

    # ═══ TOP STATUS STRIP ═══════════════════════════════════════════════════
    btc_now = float(closes[-1]) if closes.size else None
    btc_1m = float(closes[-2]) if closes.size >= 2 else btc_now
    btc_15m = float(closes[-15]) if closes.size >= 15 else btc_now
    btc_1h = float(closes[-60]) if closes.size >= 60 else btc_now
    d1m = (btc_now - btc_1m) if (btc_now and btc_1m) else 0
    d15 = (btc_now - btc_15m) if (btc_now and btc_15m) else 0
    d1h = (btc_now - btc_1h) if (btc_now and btc_1h) else 0

    # Tick highlight class
    prev = st.session_state.prev_btc
    tick_class = ""
    if btc_now and prev is not None and btc_now != prev:
        tick_class = "tick-up" if btc_now > prev else "tick-down"
    st.session_state.prev_btc = btc_now

    vol_15m = realized_vol(closes, 15)
    vol_1h = realized_vol(closes, 60)
    vol_6h = realized_vol(closes, 360)
    vol_24h = realized_vol(closes, 1440)

    sig_rate = signal_rate_per_min(signals)
    yes_n = sum(1 for s in signals if s["side"] == "yes")
    no_n = sum(1 for s in signals if s["side"] == "no")

    # Detect new signal arrival
    new_signals = len(signals) - st.session_state.prev_signal_count
    st.session_state.prev_signal_count = len(signals)
    sig_flash_cls = "tick-up" if new_signals > 0 else ""

    filled = [o for o in orders if o["status"] == "paper_filled"]
    rejected = [o for o in orders if o["status"] != "paper_filled"]
    new_fills = len(filled) - st.session_state.prev_fill_count
    st.session_state.prev_fill_count = len(filled)
    fill_flash_cls = "tick-up" if new_fills > 0 else ""

    latest_pnl = pnl[-1] if pnl else {"total_cents": 0, "realized_cents": 0, "unrealized_cents": 0}

    sc = st.columns([1.4, 1.0, 1.0, 1.0, 1.0, 1.2])

    with sc[0]:
        st.markdown("<div class='bb-header'>BTC SPOT</div>", unsafe_allow_html=True)
        col = color_for(d1m)
        st.markdown(
            f"<div class='bb-tick {tick_class}' style='color:{col}; padding: 2px 4px; border-radius: 3px;'>"
            f"${btc_now:,.2f}</div>"
            f"<div class='bb-sub'>1m {d1m:+,.2f} · 15m {d15:+,.2f} · 1h {d1h:+,.2f}</div>"
            if btc_now is not None else "<div class='bb-tick bb-neu'>—</div>",
            unsafe_allow_html=True,
        )
        # Inline mini sparkline of last 60 minutes
        if closes.size >= 30:
            spark_vals = closes[-min(60, closes.size):].tolist()
            spark_color = GREEN if spark_vals[-1] >= spark_vals[0] else RED
            st.plotly_chart(sparkline(spark_vals, color=spark_color, height=32),
                            use_container_width=True, config={"displayModeBar": False})

    with sc[1]:
        st.markdown("<div class='bb-header'>VOL (ANN)</div>", unsafe_allow_html=True)
        v15 = f"{vol_15m*100:.0f}%" if vol_15m else "—"
        v1 = f"{vol_1h*100:.0f}%" if vol_1h else "—"
        v6 = f"{vol_6h*100:.0f}%" if vol_6h else "—"
        v24 = f"{vol_24h*100:.0f}%" if vol_24h else "—"
        st.markdown(
            f"<div class='bb-tick bb-amber'>{v1}</div>"
            f"<div class='bb-sub'>15m {v15} · 6h {v6} · 24h {v24}</div>",
            unsafe_allow_html=True,
        )

    with sc[2]:
        st.markdown("<div class='bb-header'>SIGNALS</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='bb-tick bb-amber {sig_flash_cls}' style='padding:2px 4px;border-radius:3px;'>"
            f"{sig_rate:.0f}/min</div>"
            f"<div class='bb-sub'><span class='bb-pos'>YES {yes_n}</span> · "
            f"<span class='bb-neg'>NO {no_n}</span>"
            f"{f' · +{new_signals} new' if new_signals > 0 else ''}</div>",
            unsafe_allow_html=True,
        )

    with sc[3]:
        st.markdown("<div class='bb-header'>FILLS</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='bb-tick bb-amber {fill_flash_cls}' style='padding:2px 4px;border-radius:3px;'>"
            f"{len(filled)}</div>"
            f"<div class='bb-sub'>blocked {len(rejected)}"
            f"{f' · +{new_fills} new' if new_fills > 0 else ''}</div>",
            unsafe_allow_html=True,
        )

    with sc[4]:
        st.markdown("<div class='bb-header'>P&amp;L (TOTAL)</div>", unsafe_allow_html=True)
        col = color_for(latest_pnl["total_cents"])
        st.markdown(
            f"<div class='bb-tick' style='color:{col}'>{fmt_money(latest_pnl['total_cents'])}</div>"
            f"<div class='bb-sub'>R <span style='color:{color_for(latest_pnl['realized_cents'])}'>"
            f"{fmt_money(latest_pnl['realized_cents'])}</span> · "
            f"U <span style='color:{color_for(latest_pnl['unrealized_cents'])}'>"
            f"{fmt_money(latest_pnl['unrealized_cents'])}</span></div>",
            unsafe_allow_html=True,
        )

    with sc[5]:
        st.markdown("<div class='bb-header'>CALIBRATION</div>", unsafe_allow_html=True)
        if cal:
            brier = cal.get("brier")
            ll = cal.get("log_loss")
            brier_col = GREEN if (brier and brier < 0.20) else (AMBER if brier and brier < 0.25 else RED)
            st.markdown(
                f"<div class='bb-tick' style='color:{brier_col}'>Brier {brier:.4f}</div>"
                f"<div class='bb-sub'>LogLoss {ll:.4f} · n={cal['n_samples']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='bb-tick bb-neu'>PENDING</div>"
                "<div class='bb-sub'>awaiting first resolution</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ═══ ROW 1: MARKET WATCH · EDGE SCATTER ═════════════════════════════════
    r1c1, r1c2 = st.columns([1.1, 1.0])

    with r1c1:
        st.markdown("<div class='bb-header'>MARKET WATCH · TOP |EDGE|</div>", unsafe_allow_html=True)
        if watch:
            mw = []
            for r in watch:
                mid_c = r.get("market_mid_cents") or 0
                our = r["prob"]
                edge = our - mid_c / 100.0
                side = "YES" if edge > 0 else "NO"
                strike = None
                try:
                    strike = float(r["market_id"].split("-T")[-1])
                except Exception:
                    pass
                mw.append({
                    "MARKET": r["market_id"].replace("KXBTC", ""),
                    "STRIKE": f"${strike:,.0f}" if strike else "—",
                    "DIST": f"{((r['spot_usd']-strike)/strike*100):+.2f}%" if (strike and r.get("spot_usd")) else "—",
                    "B/A": f"{r.get('bid_cents','—')}/{r.get('ask_cents','—')}",
                    "MID": int(mid_c) if mid_c else "—",
                    "OUR": f"{our:.3f}",
                    "EDGE": f"{edge*100:+.1f}¢",
                    "SIDE": side,
                    "HRS": f"{r['horizon_seconds']/3600:.1f}h",
                    "VOL": f"{r['vol_annualized']*100:.0f}%",
                })
            st.dataframe(pd.DataFrame(mw), width="stretch", height=420, hide_index=True)
        else:
            st.info("No prob estimates yet.")

    with r1c2:
        st.markdown("<div class='bb-header'>EDGE SCATTER · OUR vs MARKET (color = horizon)</div>",
                    unsafe_allow_html=True)
        if estimates:
            df_e = pd.DataFrame([
                {
                    "market_prob": (e.get("market_mid_cents") or 0) / 100.0,
                    "our_prob": e["prob"],
                    "horizon_h": e["horizon_seconds"] / 3600,
                    "vol_ann": e["vol_annualized"],
                    "market_id": e["market_id"],
                }
                for e in estimates if e.get("market_mid_cents") is not None
            ])
            if not df_e.empty:
                fig = px.scatter(
                    df_e, x="market_prob", y="our_prob",
                    color="horizon_h", color_continuous_scale="Plasma",
                    opacity=0.75,
                    hover_data={"market_prob": ":.3f", "our_prob": ":.3f",
                                "horizon_h": ":.1f", "vol_ann": ":.2f", "market_id": True},
                )
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color=DIM, dash="dot", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.update_xaxes(title="market_prob (Kalshi mid)", range=[0, 1])
                fig.update_yaxes(title="our_prob (BS)", range=[0, 1])
                fig.update_coloraxes(colorbar=dict(title="hrs", thickness=10, len=0.8))
                st.plotly_chart(plotly_layout(fig, height=420), use_container_width=True)
        else:
            st.info("No estimates.")

    # ═══ ROW 2: VOL TERM STRUCTURE · EDGE HISTOGRAM ══════════════════════════
    r2c1, r2c2 = st.columns([1.0, 1.0])

    with r2c1:
        st.markdown("<div class='bb-header'>VOL TERM STRUCTURE · REALIZED (ANN)</div>",
                    unsafe_allow_html=True)
        vol_terms = []
        for label, w in [("15m", 15), ("30m", 30), ("1h", 60), ("3h", 180),
                          ("6h", 360), ("12h", 720), ("24h", 1440)]:
            v = realized_vol(closes, w)
            vol_terms.append({"horizon": label, "vol": (v * 100) if v else None})
        vt = pd.DataFrame(vol_terms)
        vt_known = vt[vt["vol"].notna()]
        if not vt_known.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vt_known["horizon"], y=vt_known["vol"],
                mode="lines+markers+text",
                line=dict(color=AMBER, width=2),
                marker=dict(size=8, color=AMBER),
                text=[f"{v:.0f}%" for v in vt_known["vol"]],
                textposition="top center",
                textfont=dict(color="#cccccc", size=10),
                name="realized",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            if estimates:
                pricer_vol = estimates[0]["vol_annualized"] * 100
                fig.add_hline(
                    y=pricer_vol, line=dict(color=GREEN, dash="dash", width=1),
                    annotation_text=f"pricer: {pricer_vol:.0f}%",
                    annotation_position="bottom right",
                    annotation_font_color=GREEN,
                )
            ymax = max(120, vt_known["vol"].max() * 1.2)
            fig.update_yaxes(title="annualized %", range=[0, ymax])
            fig.update_xaxes(title="lookback window")
            st.plotly_chart(plotly_layout(fig, height=280), use_container_width=True)
            st.markdown(
                "<div class='bb-sub'>If 24h vol &gt;&gt; 1h vol, the pricer is "
                "underestimating vol for long-horizon markets → tail mispricings are likely "
                "artifacts, not edge.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Not enough candles yet for term structure.")

    with r2c2:
        st.markdown("<div class='bb-header'>EDGE DISTRIBUTION · BY SIDE</div>", unsafe_allow_html=True)
        if signals:
            df_s = pd.DataFrame(signals)
            df_s["edge_cents"] = df_s["edge"] * 100
            fig = go.Figure()
            for side, color in [("yes", GREEN), ("no", RED)]:
                sub = df_s[df_s["side"] == side]
                if not sub.empty:
                    fig.add_trace(go.Histogram(
                        x=sub["edge_cents"], name=side.upper(),
                        marker=dict(color=color, line=dict(color="#000000", width=0.5)),
                        opacity=0.75, nbinsx=30,
                    ))
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title="edge (¢)")
            fig.update_yaxes(title="count")
            st.plotly_chart(plotly_layout(fig, height=280), use_container_width=True)
            stats_cols = st.columns(4)
            for i, (label, val) in enumerate([
                ("min", df_s["edge_cents"].min()),
                ("median", df_s["edge_cents"].median()),
                ("mean", df_s["edge_cents"].mean()),
                ("max", df_s["edge_cents"].max()),
            ]):
                stats_cols[i].markdown(
                    f"<div class='bb-sub'>{label.upper()}</div>"
                    f"<div style='font-family:monospace;color:{AMBER};font-size:0.95rem'>{val:+.1f}¢</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No signals yet.")

    # ═══ ROW 3: SIGNAL FEED · PNL / POSITIONS ════════════════════════════════
    r3c1, r3c2 = st.columns([1.1, 1.0])

    with r3c1:
        st.markdown("<div class='bb-header'>SIGNAL FEED · LIVE</div>", unsafe_allow_html=True)
        if signals:
            # Highlight signals fired in the last 3 seconds
            fresh_cutoff = now_utc.timestamp() - 3.0
            sf = []
            for s in signals[:40]:
                try:
                    ts_obj = datetime.fromisoformat(s["created_at"].replace(" ", "T"))
                    if ts_obj.tzinfo is None:
                        ts_obj = ts_obj.replace(tzinfo=timezone.utc)
                    is_fresh = ts_obj.timestamp() > fresh_cutoff
                except Exception:
                    is_fresh = False
                ts = s["created_at"].split(" ")[-1].split(".")[0] if " " in s["created_at"] else s["created_at"][-12:-7]
                sf.append({
                    "•": "●" if is_fresh else "",
                    "TIME": ts,
                    "MARKET": s["market_id"].replace("KXBTC", ""),
                    "SIDE": s["side"].upper(),
                    "OUR": f"{s['our_prob']:.3f}",
                    "MKT": f"{s['market_prob']:.3f}",
                    "EDGE": f"{s['edge']*100:+.1f}¢",
                    "FAIR": f"{s['fair_price_cents']}c",
                })
            st.dataframe(pd.DataFrame(sf), width="stretch", height=420, hide_index=True)
        else:
            st.info("No signals.")

    with r3c2:
        st.markdown("<div class='bb-header'>P&amp;L TIMELINE</div>", unsafe_allow_html=True)
        if pnl and len(pnl) >= 2:
            df_p = pd.DataFrame(pnl)
            df_p["t"] = pd.to_datetime(df_p["created_at"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_p["t"], y=df_p["total_cents"] / 100,
                mode="lines", name="total",
                line=dict(color=AMBER, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=df_p["t"], y=df_p["realized_cents"] / 100,
                mode="lines", name="realized",
                line=dict(color=GREEN, width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=df_p["t"], y=df_p["unrealized_cents"] / 100,
                mode="lines", name="unrealized",
                line=dict(color=BLUE, width=1.2, dash="dot"),
            ))
            fig.add_hline(y=0, line=dict(color=DIM, dash="dot", width=1))
            fig.update_yaxes(title="USD")
            fig.update_xaxes(title=None)
            st.plotly_chart(plotly_layout(fig, height=240), use_container_width=True)
        elif pnl:
            st.markdown(
                f"<div class='bb-sub'>Single snapshot: {fmt_money(pnl[0]['total_cents'])}. "
                f"Waiting for more PnL events to build a timeline…</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No PnL snapshots.")

        st.markdown("<div class='bb-header'>POSITIONS · OPEN</div>", unsafe_allow_html=True)
        if positions:
            pos_df = pd.DataFrame([
                {
                    "MARKET": p["market_id"].replace("KXBTC", ""),
                    "QTY": p["net_quantity"],
                    "AVG": f"{p['avg_entry_cents']:.1f}c",
                    "REAL": fmt_money(p["realized_pnl_cents"]),
                }
                for p in positions
            ])
            st.dataframe(pos_df, width="stretch", height=160, hide_index=True)
        else:
            st.markdown("<div class='bb-sub'>flat — no open positions</div>", unsafe_allow_html=True)

    # ═══ ROW 4: BTC SPOT · RECENT FILLS ═════════════════════════════════════
    r4c1, r4c2 = st.columns([1.5, 1.0])

    with r4c1:
        st.markdown("<div class='bb-header'>BTC SPOT · LAST 5 HOURS</div>", unsafe_allow_html=True)
        if candles:
            df_c = pd.DataFrame(candles[-300:])
            df_c["t"] = pd.to_datetime(df_c["timestamp_ms"], unit="ms", utc=True)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_c["t"],
                open=df_c["open"], high=df_c["high"],
                low=df_c["low"], close=df_c["close"],
                increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
                decreasing=dict(line=dict(color=RED), fillcolor=RED),
                showlegend=False,
            ))
            fig.update_yaxes(title="USD")
            fig.update_xaxes(title=None, rangeslider_visible=False)
            st.plotly_chart(plotly_layout(fig, height=260), use_container_width=True)
        else:
            st.info("No candles.")

    with r4c2:
        st.markdown("<div class='bb-header'>FILLS · RECENT 20</div>", unsafe_allow_html=True)
        if orders:
            of = []
            for o in orders[:20]:
                ts = o["created_at"].split(" ")[-1].split(".")[0] if " " in o["created_at"] else o["created_at"][-12:-7]
                of.append({
                    "TIME": ts,
                    "MARKET": o["market_id"].replace("KXBTC", ""),
                    "SIDE": o["side"].upper(),
                    "PX": f"{o['price_cents']}c",
                    "QTY": o["quantity"],
                    "STATUS": o["status"].replace("paper_", "").upper(),
                    "FILL": f"{o['fill_price_cents']}c" if o.get("fill_price_cents") else "—",
                })
            st.dataframe(pd.DataFrame(of), width="stretch", height=260, hide_index=True)
        else:
            st.info("No orders.")

    # ═══ ROW 5: CALIBRATION (full width, when populated) ═════════════════════
    if cal:
        st.markdown("<div class='bb-header'>CALIBRATION · RELIABILITY BY PROB BIN</div>",
                    unsafe_allow_html=True)
        try:
            bins = json.loads(cal["bin_json"])
            cb = pd.DataFrame(bins)
            if not cb.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color=DIM, dash="dot", width=1),
                    name="perfect", hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=cb["mean_pred"], y=cb["emp_freq"],
                    mode="markers+lines",
                    marker=dict(size=cb["n"].clip(upper=40), color=AMBER,
                                line=dict(color="#000000", width=1)),
                    line=dict(color=AMBER, width=1.5),
                    name="observed",
                    hovertemplate="pred %{x:.2f} → emp %{y:.2f}<br>n=%{marker.size}<extra></extra>",
                ))
                fig.update_xaxes(title="predicted prob", range=[0, 1])
                fig.update_yaxes(title="empirical freq", range=[0, 1])
                st.plotly_chart(plotly_layout(fig, height=320), use_container_width=True)
        except Exception as exc:
            st.warning(f"Calibration parse failed: {exc}")


# ── Auto-refresh ────────────────────────────────────────────────────────────
if hasattr(st, "fragment"):
    run_every = f"{refresh_seconds}s" if auto_refresh else None

    @st.fragment(run_every=run_every)
    def live_fragment() -> None:
        render()

    live_fragment()
else:
    render()
