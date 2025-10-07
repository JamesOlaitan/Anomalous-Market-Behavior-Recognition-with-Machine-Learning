"""Feature engineering for anomaly detection."""
import duckdb
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns and log returns.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with returns
    """
    df = df.sort_values("date")
    df["returns"] = df.groupby("symbol")["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df.groupby("symbol")["close"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility.

    Args:
        df: DataFrame with returns
        window: Rolling window size

    Returns:
        DataFrame with volatility
    """
    df["volatility"] = (
        df.groupby("symbol")["returns"]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )
    return df


def compute_rolling_correlation(
    df: pd.DataFrame, reference_symbol: str = "SPY", window: int = 60
) -> pd.DataFrame:
    """
    Compute rolling correlation with reference symbol.

    Args:
        df: DataFrame with returns
        reference_symbol: Reference symbol for correlation (e.g., SPY)
        window: Rolling window size

    Returns:
        DataFrame with rolling correlation
    """
    # Pivot to have symbols as columns
    returns_pivot = df.pivot(index="date", columns="symbol", values="returns")

    # Ensure reference symbol exists
    if reference_symbol not in returns_pivot.columns:
        logger.warning(
            f"Reference symbol {reference_symbol} not found. Using first symbol."
        )
        reference_symbol = returns_pivot.columns[0]

    # Compute rolling correlation with reference
    correlations = {}
    for symbol in returns_pivot.columns:
        if symbol != reference_symbol:
            corr = (
                returns_pivot[symbol]
                .rolling(window, min_periods=int(window * 0.5))
                .corr(returns_pivot[reference_symbol])
            )
            correlations[symbol] = corr

    # Convert back to long format
    corr_df = pd.DataFrame(correlations).reset_index()
    corr_df = corr_df.melt(id_vars="date", var_name="symbol", value_name="rolling_corr")

    # Merge back to original df
    df = df.merge(corr_df, on=["date", "symbol"], how="left")

    return df


def compute_z_scores(df: pd.DataFrame, column: str, window: int = 120) -> pd.DataFrame:
    """
    Compute rolling z-scores for a column.

    Args:
        df: DataFrame
        column: Column name to compute z-scores for
        window: Rolling window size

    Returns:
        DataFrame with z-scores
    """
    zscore_col = f"{column.replace('rolling_', '')}_zscore"

    df[zscore_col] = df.groupby("symbol")[column].transform(
        lambda x: (x - x.rolling(window, min_periods=1).mean())
        / x.rolling(window, min_periods=1).std()
    )

    return df


def merge_vix(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VIX data and compute VIX delta.

    Args:
        df: Main features DataFrame
        vix_df: VIX DataFrame

    Returns:
        DataFrame with VIX features
    """
    # Merge VIX
    df = df.merge(vix_df, on="date", how="left")

    # Forward fill VIX (in case of date misalignment)
    df["vix"] = df["vix"].ffill()

    # Compute VIX delta
    df["vix_delta"] = df["vix"].diff()

    return df


def generate_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate anomaly labels based on correlation breakdown.

    Args:
        df: Features DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with labels
    """
    label_config = config["labels"]
    corr_threshold = label_config["corr_threshold"]
    delta_threshold = label_config["delta_threshold"]
    delta_window = label_config["delta_window"]
    dwell_window = label_config["dwell_window"]

    logger.info(f"Generating labels with correlation threshold: {corr_threshold}")

    # Compute correlation delta
    df["corr_delta"] = df.groupby("symbol")["rolling_corr"].diff(delta_window)

    # Label as anomaly if:
    # 1. Correlation drops below threshold
    # 2. AND correlation change is negative and large
    df["label"] = (
        (df["rolling_corr"] < corr_threshold) & (df["corr_delta"] < delta_threshold)
    ).astype(int)

    # Extend labels for dwell window (persistence)
    if dwell_window > 1:
        df["label"] = (
            df.groupby("symbol")["label"]
            .transform(lambda x: x.rolling(dwell_window, min_periods=1).max())
            .astype(int)
        )

    # Drop temporary column
    df = df.drop(columns=["corr_delta"])

    anomaly_ratio = df["label"].mean()
    logger.info(f"Generated labels: {anomaly_ratio:.2%} anomalies")

    return df


def engineer_features(conn: duckdb.DuckDBPyConnection, config: dict) -> None:
    """
    Main feature engineering pipeline.

    Args:
        conn: DuckDB connection
        config: Configuration dictionary
    """
    logger.info("Starting feature engineering")

    # Load data from database
    logger.info("Loading price data from database")
    prices = conn.execute("SELECT * FROM raw_prices ORDER BY date, symbol").df()

    logger.info("Loading VIX data from database")
    vix = conn.execute("SELECT * FROM raw_vix ORDER BY date").df()

    # Compute features
    logger.info("Computing returns")
    features = compute_returns(prices)

    logger.info("Computing volatility")
    features = compute_volatility(
        features, window=config["features"]["volatility_window"]
    )

    logger.info("Computing rolling correlation")
    features = compute_rolling_correlation(
        features,
        reference_symbol="SPY",
        window=config["features"]["rolling_window"],
    )

    logger.info("Computing z-scores")
    features = compute_z_scores(features, "rolling_corr")
    features = compute_z_scores(features, "volatility")

    logger.info("Merging VIX data")
    features = merge_vix(features, vix)

    # Generate labels
    logger.info("Generating labels")
    features = generate_labels(features, config)

    # Drop rows with NaN in critical columns
    before_count = len(features)
    features = features.dropna(
        subset=[
            "returns",
            "log_returns",
            "volatility",
            "rolling_corr",
            "corr_zscore",
            "vol_zscore",
            "vix",
        ]
    )
    after_count = len(features)
    logger.info(f"Dropped {before_count - after_count} rows with missing values")

    # Select and order columns for features table
    features_cols = [
        "date",
        "symbol",
        "returns",
        "log_returns",
        "volatility",
        "rolling_corr",
        "corr_zscore",
        "vol_zscore",
        "vix",
        "vix_delta",
    ]
    features_for_db = features[features_cols]

    # Select columns for labels table
    labels_for_db = features[["date", "symbol", "label"]]

    # Save to database
    logger.info("Saving features to database")
    conn.execute("DELETE FROM features")
    conn.register("features_for_db", features_for_db)
    conn.execute("INSERT INTO features SELECT * FROM features_for_db")

    logger.info("Saving labels to database")
    conn.register("labels_for_db", labels_for_db)
    conn.execute("DELETE FROM labels")
    conn.execute("INSERT INTO labels SELECT * FROM labels_for_db")

    feature_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    label_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    logger.info(f"✅ Feature engineering completed: {feature_count} feature records")
    logger.info(f"✅ Labeling completed: {label_count} label records")


def main():
    """Main function for feature engineering."""
    config = load_config()

    # Connect to database
    db_path = config["sql"]["database"]
    logger.info(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)

    try:
        engineer_features(conn, config)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
