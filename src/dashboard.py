"""
Polymarket Intelligence Bot — Streamlit Analytics Dashboard.

Run with:
    streamlit run src/dashboard.py

Connects read-only to data/bot_state.db and displays three tabs:
  1. Capital Growth Curve (P&L)
  2. AI vs Market Spread
  3. Network / Errors Log
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "data" / "bot_state.db"

st.set_page_config(
    page_title="Polymarket Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database helpers ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=30)
def load_evaluations() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM evaluations ORDER BY created_at ASC", conn
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_orders() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM orders ORDER BY created_at DESC", conn
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_positions() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = _connect()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM positions ORDER BY created_at DESC", conn
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Polymarket Intel")
st.sidebar.caption(f"DB: `{DB_PATH.name}`")

if not DB_PATH.exists():
    st.sidebar.warning("Database not found.  Run the bot first to generate data.")

refresh_secs = st.sidebar.slider(
    "Auto-refresh (seconds)", min_value=10, max_value=120, value=30, step=5
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**View-only dashboard** — no private keys, no order placement."
)

# ── Load data ─────────────────────────────────────────────────────────────────

evals  = load_evaluations()
orders = load_orders()
positions = load_positions()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📊 Capital Growth (P&L)",
    "🔬 AI vs Market Spread",
    "🌐 Network / Errors",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — P&L Curve
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Capital Growth Curve")

    if evals.empty:
        st.info("No evaluation data yet.  Start the bot to generate entries.")
    else:
        buys = evals[evals["action"] == "BUY"].copy()

        if buys.empty:
            st.info("No BUY actions recorded yet.")
        else:
            buys["simulated_pnl"] = buys["position_size_usd"] * buys["expected_value"]
            buys["cumulative_pnl"] = buys["simulated_pnl"].cumsum()
            buys["cumulative_capital"] = buys["position_size_usd"].cumsum()

            # KPIs
            total_pnl      = buys["simulated_pnl"].sum()
            total_capital   = buys["position_size_usd"].sum()
            avg_ev          = buys["expected_value"].mean()
            trade_count     = len(buys)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Simulated P&L", f"${total_pnl:,.2f}",
                       delta=f"{'▲' if total_pnl >= 0 else '▼'}")
            c2.metric("Capital Deployed", f"${total_capital:,.2f}")
            c3.metric("Avg Expected Value", f"{avg_ev:+.4f}")
            c4.metric("BUY Evaluations", f"{trade_count:,}")

            # Cumulative P&L line chart
            fig = px.line(
                buys,
                x="created_at",
                y="cumulative_pnl",
                title="Cumulative Simulated P&L Over Time",
                labels={"created_at": "Time", "cumulative_pnl": "Cumulative P&L ($)"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Per-trade P&L bar chart
            fig2 = px.bar(
                buys.tail(50),
                x="created_at",
                y="simulated_pnl",
                color="confidence",
                title="Per-Trade Simulated P&L (last 50 BUYs)",
                labels={"simulated_pnl": "P&L ($)", "created_at": "Time"},
                color_discrete_map={
                    "HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"
                },
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            # Raw data expander
            with st.expander("Raw BUY evaluations"):
                st.dataframe(
                    buys[[
                        "created_at", "question", "market_price",
                        "estimated_prob", "expected_value", "kelly_fraction",
                        "position_size_usd", "confidence", "simulated_pnl",
                    ]].tail(100),
                    use_container_width=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — AI vs Market Spread
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("AI Probability vs Market Price")

    if evals.empty:
        st.info("No evaluation data yet.")
    else:
        evals_c = evals.copy()
        evals_c["spread"] = evals_c["estimated_prob"] - evals_c["market_price"]
        evals_c["positive_ev"] = evals_c["spread"] > 0

        # KPIs
        mean_spread = evals_c["spread"].mean()
        med_spread  = evals_c["spread"].median()
        pct_pos_ev  = evals_c["positive_ev"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Spread (AI − Market)", f"{mean_spread:+.4f}")
        c2.metric("Median Spread", f"{med_spread:+.4f}")
        c3.metric("% Positive EV", f"{pct_pos_ev:.1f}%")

        col_left, col_right = st.columns(2)

        with col_left:
            # Scatter: market_price vs estimated_prob
            fig_scatter = px.scatter(
                evals_c,
                x="market_price",
                y="estimated_prob",
                color="confidence",
                title="Market Price vs AI Estimate",
                labels={
                    "market_price": "Market YES Price",
                    "estimated_prob": "AI Probability Estimate",
                },
                color_discrete_map={
                    "HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"
                },
                opacity=0.7,
            )
            # 45° line = perfect agreement
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="y = x",
                    showlegend=True,
                )
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            # Histogram: spread distribution
            fig_hist = px.histogram(
                evals_c,
                x="spread",
                nbins=40,
                title="Spread Distribution (AI − Market)",
                labels={"spread": "Spread"},
                color_discrete_sequence=["#3498db"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)
            fig_hist.update_layout(height=420)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Spread over time
        fig_time = px.scatter(
            evals_c,
            x="created_at",
            y="spread",
            color="action",
            title="Spread Over Time",
            labels={"created_at": "Time", "spread": "AI − Market"},
            color_discrete_map={
                "BUY": "#2ecc71", "SKIP": "#95a5a6", "REJECT": "#e74c3c"
            },
            opacity=0.6,
        )
        fig_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_time.update_layout(height=350)
        st.plotly_chart(fig_time, use_container_width=True)

        with st.expander("Evaluation details"):
            st.dataframe(
                evals_c[[
                    "created_at", "question", "market_price",
                    "estimated_prob", "spread", "confidence", "action",
                ]].tail(200),
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Network / Errors
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Network Health & Error Log")

    if orders.empty:
        st.info("No order data yet.")
    else:
        total_orders = len(orders)
        filled       = len(orders[orders["status"].isin(["FILLED", "MATCHED"])])
        rejected     = len(orders[orders["status"] == "REJECTED"])
        dry_run      = len(orders[orders["status"] == "DRY_RUN"])
        errored      = len(orders[orders["error_message"].notna() & (orders["error_message"] != "")])
        fill_rate    = filled / max(total_orders - dry_run, 1) * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Orders", f"{total_orders:,}")
        c2.metric("Filled", f"{filled:,}")
        c3.metric("Rejected", f"{rejected:,}")
        c4.metric("Dry-Run", f"{dry_run:,}")
        c5.metric("Fill Rate", f"{fill_rate:.1f}%")

        # Order status distribution
        status_counts = orders["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]

        fig_bar = px.bar(
            status_counts,
            x="status",
            y="count",
            title="Order Status Distribution",
            color="status",
            color_discrete_map={
                "FILLED": "#2ecc71",
                "MATCHED": "#27ae60",
                "PENDING": "#f39c12",
                "REJECTED": "#e74c3c",
                "CANCELLED": "#95a5a6",
                "DRY_RUN": "#3498db",
            },
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Error log table
        error_df = orders[
            (orders["error_message"].notna() & (orders["error_message"] != ""))
            | (orders["status"] == "REJECTED")
        ].copy()

        if error_df.empty:
            st.success("No errors recorded.")
        else:
            st.subheader(f"Error Log ({len(error_df)} entries)")
            st.dataframe(
                error_df[[
                    "created_at", "question", "side", "price",
                    "size", "status", "error_message",
                ]],
                use_container_width=True,
            )

    # ── Positions summary ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Open Positions")

    if positions.empty:
        st.info("No position data.")
    else:
        open_pos = positions[positions["status"] == "OPEN"]
        if open_pos.empty:
            st.info("No open positions.")
        else:
            st.dataframe(
                open_pos[[
                    "question", "outcome", "shares", "avg_entry_price",
                    "current_price", "cost_usd", "current_value_usd",
                    "unrealized_pnl", "created_at",
                ]],
                use_container_width=True,
            )

# ── Auto-refresh ──────────────────────────────────────────────────────────────

import time
time.sleep(refresh_secs)
st.rerun()
