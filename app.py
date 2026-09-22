"""
Streamlit dashboard for the Earnings Call Tone Analyzer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import DEFAULT_TICKERS, FORWARD_DAYS, TONE_DIMENSIONS
from src.backtester import ToneBacktester
from src.price_fetcher import PriceFetcher
from src.tone_analyzer import ToneAnalyzer
from src.transcript_loader import TranscriptLoader
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Earnings Call Tone Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Controls")
st.sidebar.markdown("### Universe")
tickers = st.sidebar.multiselect(
    "Tickers",
    options=DEFAULT_TICKERS,
    default=DEFAULT_TICKERS[:6],
)

use_sample = st.sidebar.checkbox("Use sample transcripts (offline)", value=True)
prefer_llm = st.sidebar.checkbox(
    "Prefer OpenAI LLM scoring (needs OPENAI_API_KEY)", value=False
)
forward_days = st.sidebar.slider("Forward return window (days)", 20, 90, FORWARD_DAYS)

run_button = st.sidebar.button("Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Investment Thesis**  
Management tone that combines high optimism with high evasiveness and low specificity  
is often an early warning that subsequent price performance will be weak.  
This tool quantifies those gaps and backtests them.
"""
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("AI-Powered Earnings Call Tone Analyzer")
st.markdown(
    "Detect shifts in management language (evasiveness, over-optimism, vagueness) "
    "and measure their relationship to subsequent stock returns."
)

@st.cache_data(show_spinner="Loading transcripts & scoring tone...")
def run_pipeline(
    selected_tickers: list,
    use_sample_data: bool,
    use_llm: bool,
    fwd_days: int,
):
    loader = TranscriptLoader(use_sample=use_sample_data)
    transcripts = loader.get_transcripts(tickers=selected_tickers)
    if not transcripts:
        return None, None, None, None, None

    analyzer = ToneAnalyzer(prefer_llm=use_llm)
    scored = analyzer.score_batch(transcripts)

    fetcher = PriceFetcher()
    with_returns = fetcher.attach_returns(scored, days=fwd_days)

    bt = ToneBacktester(forward_days=fwd_days)
    df, stats, corr, reg = bt.run(with_returns)
    return df, stats, corr, reg, with_returns


if run_button or "df" not in st.session_state:
    with st.spinner("Running full pipeline..."):
        df, stats, corr, reg, raw = run_pipeline(
            tickers, use_sample, prefer_llm, forward_days
        )
        st.session_state["df"] = df
        st.session_state["stats"] = stats
        st.session_state["corr"] = corr
        st.session_state["reg"] = reg
        st.session_state["raw"] = raw

df = st.session_state.get("df")
stats = st.session_state.get("stats")
corr = st.session_state.get("corr")
reg = st.session_state.get("reg")

if df is None or df.empty:
    st.warning("No data available for the selected tickers. Try enabling sample data.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Events", stats.get("n_events", 0))
col2.metric("Warning Signals", stats.get("n_warning", 0))
col3.metric(
    "Avg Return (All)",
    f"{stats.get('mean_return_all', 0)*100:.1f}%",
)
col4.metric(
    "Avg Return (Warning)",
    f"{stats.get('mean_return_warning', float('nan'))*100:.1f}%",
    delta=f"{stats.get('return_diff_warning_vs_rest', 0)*100:.1f}% vs rest",
)
col5.metric(
    "Warning Hit Rate (neg. return)",
    f"{stats.get('warning_hit_rate', float('nan'))*100:.0f}%",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Signal vs Returns", "Tone Distributions", "Correlations", "Event Detail"]
)

with tab1:
    st.subheader("Warning Score vs Forward Return")
    fig = px.scatter(
        df,
        x="warning_score",
        y=f"forward_return_{forward_days}d",
        color="is_warning",
        hover_data=["ticker", "call_date", "fiscal_year", "fiscal_quarter"],
        labels={
            "warning_score": "Warning Score (higher = more concerning)",
            f"forward_return_{forward_days}d": f"{forward_days}-Day Return",
        },
        title="Does higher warning score predict weaker subsequent returns?",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # Box plot
    fig2 = px.box(
        df,
        x="is_warning",
        y=f"forward_return_{forward_days}d",
        color="is_warning",
        points="all",
        title="Return Distribution: Warning vs Non-Warning Events",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Tone Dimension Distributions")
    melt = df.melt(
        id_vars=["ticker", "call_date"],
        value_vars=TONE_DIMENSIONS,
        var_name="dimension",
        value_name="score",
    )
    fig = px.violin(
        melt,
        x="dimension",
        y="score",
        box=True,
        points="all",
        title="Distribution of Tone Scores",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation with Forward Returns")
    if corr is not None and not corr.empty:
        fig = px.bar(
            corr.reset_index(),
            x="index",
            y="correlation_with_forward_return",
            labels={"index": "Tone Dimension", "correlation_with_forward_return": "Pearson r"},
            title="Which tone features relate most to subsequent returns?",
            color="correlation_with_forward_return",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

    if reg:
        st.markdown("### Simple OLS Coefficients")
        st.json(reg)

with tab4:
    st.subheader("Individual Events")
    display_cols = [
        "ticker",
        "call_date",
        "fiscal_year",
        "fiscal_quarter",
        "evasiveness",
        "optimism",
        "specificity",
        "warning_score",
        "is_warning",
        f"forward_return_{forward_days}d",
    ]
    show = df[display_cols].sort_values("call_date", ascending=False)
    show[f"forward_return_{forward_days}d"] = show[f"forward_return_{forward_days}d"].map(
        lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
    )
    st.dataframe(show, use_container_width=True, height=400)

    # Expandable transcript view
    st.markdown("### Inspect a Transcript")
    options = [
        f"{r['ticker']} Q{r['fiscal_quarter']} {r['fiscal_year']} ({r['call_date'].date() if hasattr(r['call_date'], 'date') else r['call_date']})"
        for _, r in df.iterrows()
    ]
    choice = st.selectbox("Select event", options)
    if choice:
        idx = options.index(choice)
        row = df.iloc[idx]
        st.markdown(f"**Warning Score:** {row['warning_score']:.2f}")
        st.markdown(
            f"**Scores** — Evasiveness: {row['evasiveness']:.1f} | "
            f"Optimism: {row['optimism']:.1f} | Specificity: {row['specificity']:.1f}"
        )
        # Original transcript text is in the raw list
        raw = st.session_state.get("raw") or []
        matching = [r for r in raw if r["ticker"] == row["ticker"] and r.get("fiscal_year") == row["fiscal_year"] and r.get("fiscal_quarter") == row["fiscal_quarter"]]
        if matching:
            st.text_area("Transcript excerpt", matching[0].get("transcript", "")[:3000], height=250)

st.markdown("---")
st.caption(
    "Demo built for quant / hedge-fund applications. "
    "Replace sample transcripts with a real API key for production use. "
    "Past patterns are not guarantees of future results."
)
