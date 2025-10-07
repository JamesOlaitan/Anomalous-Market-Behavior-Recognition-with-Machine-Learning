"""Interactive Streamlit dashboard for anomaly detection."""
import json
import os

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.utils.config import load_config

# Page config
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_db_connection():
    """Load database connection."""
    config = load_config()
    db_path = config["sql"]["database"]
    if not os.path.exists(db_path):
        st.error(f"Database not found: {db_path}")
        st.info("Please run the data pipeline first: `make all`")
        st.stop()
    return duckdb.connect(db_path, read_only=True)


@st.cache_data
def load_data(_conn):
    """Load data from database."""
    query = """
        SELECT
            p.date,
            p.symbol,
            rp.close,
            f.returns,
            f.rolling_corr,
            f.vix,
            p.p_anom,
            p.post_normal,
            p.post_anomalous,
            p.state,
            l.label
        FROM predictions p
        JOIN features f ON p.date = f.date AND p.symbol = f.symbol
        JOIN labels l ON p.date = l.date AND p.symbol = l.symbol
        JOIN raw_prices rp ON p.date = rp.date AND p.symbol = rp.symbol
        ORDER BY p.date, p.symbol
    """
    return _conn.execute(query).df()


@st.cache_data
def load_metrics():
    """Load metrics from file."""
    try:
        with open("artifacts/metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def plot_time_series(df: pd.DataFrame, symbol: str):
    """Plot time series with anomaly highlights."""
    df_sym = df[df["symbol"] == symbol].copy()

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Price",
            "LSTM Anomaly Probability",
            "Markov Posterior P(A)",
            "Ground Truth Labels",
        ),
    )

    # Price
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["close"],
            name="Close Price",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # LSTM scores
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["p_anom"],
            name="LSTM P(Anomaly)",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

    # Markov posteriors
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["post_anomalous"],
            name="Markov P(A)",
            line=dict(color="green"),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=3, col=1)

    # True labels
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["label"],
            name="True Anomaly",
            fill="tozeroy",
            line=dict(color="red"),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(height=900, showlegend=True, title_text=f"Anomaly Detection: {symbol}")
    fig.update_xaxes(title_text="Date", row=4, col=1)

    return fig


def plot_correlation_analysis(df: pd.DataFrame):
    """Plot correlation analysis."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Rolling Correlation with SPY", "VIX Index"),
    )

    symbols = df["symbol"].unique()
    for symbol in symbols:
        df_sym = df[df["symbol"] == symbol]
        fig.add_trace(
            go.Scatter(
                x=df_sym["date"],
                y=df_sym["rolling_corr"],
                name=f"{symbol} Corr",
                mode="lines",
            ),
            row=1,
            col=1,
        )

    # VIX (just one copy)
    df_first = df[df["symbol"] == symbols[0]]
    fig.add_trace(
        go.Scatter(
            x=df_first["date"],
            y=df_first["vix"],
            name="VIX",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=600, showlegend=True, title_text="Correlation & VIX Analysis")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)

    return fig


def main():
    """Main dashboard function."""
    st.title("📊 Anomalous Market Behavior Detection Dashboard")
    st.markdown("---")

    # Load data
    conn = load_db_connection()
    df = load_data(conn)
    metrics = load_metrics()

    # Sidebar
    st.sidebar.header("Configuration")

    # Symbol selector
    symbols = df["symbol"].unique()
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

    # Date range
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df["date"].min(), df["date"].max()),
        min_value=df["date"].min(),
        max_value=df["date"].max(),
    )

    # Threshold controls
    st.sidebar.header("Detection Parameters")
    lstm_threshold = st.sidebar.slider("LSTM Threshold", 0.0, 1.0, 0.5, 0.05)
    markov_threshold = st.sidebar.slider("Markov Threshold", 0.0, 1.0, 0.7, 0.05)

    # Filter data by date
    df_filtered = df[
        (df["date"] >= pd.Timestamp(date_range[0]))
        & (df["date"] <= pd.Timestamp(date_range[1]))
    ]

    # Metrics summary
    if metrics:
        st.header("📈 Model Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("LSTM Metrics")
            lstm_metrics = metrics["lstm"]
            st.metric("F1 Score", f"{lstm_metrics['f1']:.4f}")
            st.metric("Precision", f"{lstm_metrics['precision']:.4f}")
            st.metric("Recall", f"{lstm_metrics['recall']:.4f}")
            st.metric("ROC-AUC", f"{lstm_metrics['roc_auc']:.4f}")
            st.metric("PR-AUC", f"{lstm_metrics['pr_auc']:.4f}")

        with col2:
            st.subheader("Markov Smoothed Metrics")
            markov_metrics = metrics["markov"]
            st.metric("F1 Score", f"{markov_metrics['f1']:.4f}")
            st.metric("Precision", f"{markov_metrics['precision']:.4f}")
            st.metric("Recall", f"{markov_metrics['recall']:.4f}")
            st.metric("ROC-AUC", f"{markov_metrics['roc_auc']:.4f}")
            st.metric("PR-AUC", f"{markov_metrics['pr_auc']:.4f}")

        st.markdown("---")

    # Time series plot
    st.header(f"🔍 Detailed Analysis: {selected_symbol}")
    fig_ts = plot_time_series(df_filtered, selected_symbol)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Correlation analysis
    st.header("📉 Correlation & VIX Analysis")
    fig_corr = plot_correlation_analysis(df_filtered)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Anomaly windows
    st.header("⚠️ Detected Anomaly Windows")
    df_sym = df_filtered[df_filtered["symbol"] == selected_symbol]
    anomalies_lstm = df_sym[df_sym["p_anom"] > lstm_threshold]
    anomalies_markov = df_sym[df_sym["post_anomalous"] > markov_threshold]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df_sym))
    col2.metric("LSTM Anomalies", len(anomalies_lstm))
    col3.metric("Markov Anomalies", len(anomalies_markov))

    # Show anomaly table
    if len(anomalies_markov) > 0:
        st.subheader("Anomaly Events (Markov)")
        display_cols = ["date", "close", "rolling_corr", "vix", "p_anom", "post_anomalous", "label"]
        st.dataframe(
            anomalies_markov[display_cols].sort_values("date", ascending=False).head(20)
        )

    # Data dictionary
    with st.expander("📖 Data Dictionary"):
        st.markdown("""
        **Columns:**
        - **date**: Trading date
        - **symbol**: Stock/ETF symbol
        - **close**: Closing price
        - **returns**: Daily returns
        - **rolling_corr**: 60-day rolling correlation with SPY
        - **vix**: VIX index (volatility measure)
        - **p_anom**: LSTM anomaly probability [0, 1]
        - **post_normal**: Markov posterior P(Normal)
        - **post_anomalous**: Markov posterior P(Anomalous)
        - **state**: Markov state (N=Normal, A=Anomalous)
        - **label**: Ground truth anomaly label (0=Normal, 1=Anomaly)

        **Markov Smoother:**
        The Markov smoother applies temporal smoothing to LSTM scores using a simple
        2-state Hidden Markov Model (HMM-lite) with states Normal (N) and Anomalous (A).
        The transition matrix encodes persistence in each state, reducing false positives
        from noisy LSTM predictions.
        """)

    # Footer
    st.markdown("---")
    caption_text = (
        "Anomalous Market Behavior Recognition with Machine Learning | "
        "PyTorch + DuckDB + Streamlit"
    )
    st.caption(caption_text)


if __name__ == "__main__":
    main()
