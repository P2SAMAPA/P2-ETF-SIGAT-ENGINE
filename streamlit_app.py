"""
Streamlit UI for P2-ETF-SIGAT-ENGINE
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import config
from push_results import load_latest_result
from us_calendar import next_trading_day

st.set_page_config(page_title="SiGAT Engine", layout="wide")

# ---------- Professional CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        border: 1px solid #F0F2F5;
        transition: box-shadow 0.2s;
    }
    .card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
    }

    .ticker-large {
        font-size: 4.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
        color: #111827;
    }
    .pred-return {
        font-size: 1.4rem;
        color: #059669;
        font-weight: 500;
        margin: 0.3rem 0 0.5rem 0;
    }
    .meta-text {
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .source-badge {
        background-color: #F3F4F6;
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #374151;
        margin-top: 0.5rem;
    }

    .section-divider {
        margin: 1.5rem 0;
        border-top: 1px solid #E5E7EB;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #111827;
    }

    .stSelectbox > div > div {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown('<div class="main-header">SIGAT — Signed Graph Attention ETF Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">SGCN · SiGAT · Adversarial Relationships · Cross‑Universe Signals</div>',
    unsafe_allow_html=True,
)

# Three tabs
tab_fi, tab_eq, tab_comb = st.tabs([
    "Option A — Fixed Income / Commodities",
    "Option B — Equity Sectors",
    "Option C — Combined Universe"
])

# Load latest results
results = load_latest_result()


# ---------- Helper Functions ----------
def format_pct(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value*100:.1f}%"


def format_number(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{decimals}f}"


def display_metrics_card(metrics: dict):
    if not metrics:
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-label">ANN RETURN</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("ann_return"))}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">ANN VOL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("ann_vol"))}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-label">SHARPE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(metrics.get("sharpe"), 2)}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-label">MAX DD</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("max_dd"))}</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-label">HIT RATE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("hit_rate"))}</div>', unsafe_allow_html=True)


def display_shrinking_weights(windows: list, selected_ticker: str):
    if not windows:
        return

    scores = {}
    for w in windows:
        ticker = w["ticker"]
        ret = w["metrics"].get("ann_return", 0.0)
        sharpe = w["metrics"].get("sharpe", 0.0)
        max_dd = w["metrics"].get("max_dd", -1.0)
        hit_rate = w["metrics"].get("hit_rate", 0.0)

        if ret <= 0:
            weight = 0.0
        else:
            dd_score = 1.0 / (1.0 + abs(max_dd))
            weight = (
                config.WEIGHT_RETURN * ret
                + config.WEIGHT_SHARPE * sharpe
                + config.WEIGHT_HITRATE * hit_rate
                + config.WEIGHT_MAXDD * dd_score
            )
        scores[ticker] = scores.get(ticker, 0.0) + weight

    if not scores:
        return

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    df_scores = pd.DataFrame(sorted_scores, columns=["ETF", "Weighted Score"])
    df_scores["Weighted Score"] = df_scores["Weighted Score"].round(4)
    df_scores.index = df_scores.index + 1

    st.markdown("#### ETF Weighted Scores (Shrinking Window Aggregation)")
    st.dataframe(
        df_scores.style.highlight_between(
            subset=["Weighted Score"],
            left=df_scores["Weighted Score"].max(),
            right=df_scores["Weighted Score"].max(),
            color="#E6F7E6"
        ).format({"Weighted Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=False,
    )
    st.caption("Weighting: 60% Return · 10% Sharpe · 10% Hit Rate · 20% Max DD (inverted). Negative return years receive zero weight.")


def display_global_card(universe_data: dict):
    global_data = universe_data.get("global", {})
    if not global_data or not global_data.get("ticker"):
        st.info("⏳ Waiting for training output...")
        return

    ticker = global_data["ticker"]
    pred_return = global_data.get("pred_return")
    metrics = global_data.get("metrics", {})
    test_start = global_data.get("test_start", "")
    test_end = global_data.get("test_end", "")

    next_day = next_trading_day(datetime.utcnow())
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f'<div class="ticker-large">{ticker}</div>', unsafe_allow_html=True)
        if pred_return is not None:
            pred_str = f"{pred_return*100:.2f}%"
            st.markdown(f'<div class="pred-return">Predicted Return: {pred_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-text">Signal for {next_day.strftime("%Y-%m-%d")} · Generated {gen_time}</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-badge">Source: Global Training</div>', unsafe_allow_html=True)
    with col2:
        # No 2nd/3rd placeholders
        pass

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('**FIXED SPLIT (80/10/10)**')
    st.markdown(f'<div class="meta-text">Test: {test_start} → {test_end}</div>', unsafe_allow_html=True)

    display_metrics_card(metrics)

    st.markdown('</div>', unsafe_allow_html=True)


def display_shrinking_card(universe_data: dict, universe_name: str):
    shrinking_data = universe_data.get("shrinking", {})
    if not shrinking_data or not shrinking_data.get("ticker"):
        st.info("⏳ Waiting for training output...")
        return

    ticker = shrinking_data["ticker"]
    pred_return = shrinking_data.get("pred_return")
    windows = shrinking_data.get("windows", [])

    next_day = next_trading_day(datetime.utcnow())
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f'<div class="ticker-large">{ticker}</div>', unsafe_allow_html=True)
        if pred_return is not None:
            pred_str = f"{pred_return*100:.2f}%"
            st.markdown(f'<div class="pred-return">Predicted Return: {pred_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-text">Signal for {next_day.strftime("%Y-%m-%d")} · Generated {gen_time}</div>', unsafe_allow_html=True)
        st.markdown('<div class="source-badge">Source: Shrinking Window</div>', unsafe_allow_html=True)
    with col2:
        # No 2nd/3rd placeholders
        pass

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if windows:
        window_labels = [
            f"Window {i+1}: {w['window_start']} → {w['val_end']} (OOS: {w['test_start']} → {w['test_end']})"
            for i, w in enumerate(windows)
        ]
        selected_idx = st.selectbox(
            "Select a training window to view its out‑of‑sample metrics:",
            range(len(window_labels)),
            format_func=lambda i: window_labels[i],
            key=f"shrinking_select_{universe_name}"
        )
        selected_window = windows[selected_idx]
        st.markdown(f'<div class="meta-text">OOS Period: {selected_window["test_start"]} → {selected_window["test_end"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-text">Selected ETF for this window: <strong>{selected_window["ticker"]}</strong></div>', unsafe_allow_html=True)
        display_metrics_card(selected_window["metrics"])
    else:
        st.info("No window data available.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    display_shrinking_weights(windows, ticker)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Render Tabs ----------
with tab_fi:
    st.subheader("FI / Commodities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Training")
        display_global_card(results.get("fi", {}))
    with col2:
        st.markdown("### Shrinking Window")
        display_shrinking_card(results.get("fi", {}), "fi")

with tab_eq:
    st.subheader("Equity Sectors")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Training")
        display_global_card(results.get("equity", {}))
    with col2:
        st.markdown("### Shrinking Window")
        display_shrinking_card(results.get("equity", {}), "equity")

with tab_comb:
    st.subheader("Combined Universe")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Training")
        display_global_card(results.get("combined", {}))
    with col2:
        st.markdown("### Shrinking Window")
        display_shrinking_card(results.get("combined", {}), "combined")
