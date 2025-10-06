"""Ingest raw data into DuckDB database."""
import logging

import duckdb
import pandas as pd

from src.utils.config import load_config
from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


def create_database_schema(conn: duckdb.DuckDBPyConnection, config: dict) -> None:
    """
    Create database schema and tables.

    Args:
        conn: DuckDB connection
        config: Configuration dictionary
    """
    logger.info("Creating database schema")

    # Raw prices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            date DATE,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (date, symbol)
        )
    """)

    # Raw VIX table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_vix (
            date DATE PRIMARY KEY,
            vix DOUBLE
        )
    """)

    # Features table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            date DATE,
            symbol VARCHAR,
            returns DOUBLE,
            log_returns DOUBLE,
            volatility DOUBLE,
            rolling_corr DOUBLE,
            corr_zscore DOUBLE,
            vol_zscore DOUBLE,
            vix DOUBLE,
            vix_delta DOUBLE,
            PRIMARY KEY (date, symbol)
        )
    """)

    # Labels table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            date DATE,
            symbol VARCHAR,
            label INTEGER,
            PRIMARY KEY (date, symbol)
        )
    """)

    # Predictions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            date DATE,
            symbol VARCHAR,
            p_anom DOUBLE,
            post_normal DOUBLE,
            post_anomalous DOUBLE,
            state VARCHAR,
            PRIMARY KEY (date, symbol)
        )
    """)

    logger.info("✅ Schema created successfully")


def ingest_prices(conn: duckdb.DuckDBPyConnection, file_path: str) -> None:
    """
    Ingest price data into database.

    Args:
        conn: DuckDB connection
        file_path: Path to prices CSV file
    """
    logger.info(f"Ingesting prices from {file_path}")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # Delete existing data
    conn.execute("DELETE FROM raw_prices")

    # Insert new data
    conn.execute("INSERT INTO raw_prices SELECT * FROM df")

    count = conn.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
    logger.info(f"Ingested {count} price records")


def ingest_vix(conn: duckdb.DuckDBPyConnection, file_path: str) -> None:
    """
    Ingest VIX data into database.

    Args:
        conn: DuckDB connection
        file_path: Path to VIX CSV file
    """
    logger.info(f"Ingesting VIX from {file_path}")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # Delete existing data
    conn.execute("DELETE FROM raw_vix")

    # Insert new data
    conn.execute("INSERT INTO raw_vix SELECT * FROM df")

    count = conn.execute("SELECT COUNT(*) FROM raw_vix").fetchone()[0]
    logger.info(f"Ingested {count} VIX records")


def main():
    """Main function to ingest data into SQL."""
    config = load_config()

    # Connect to DuckDB
    db_path = config["sql"]["database"]
    logger.info(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)

    try:
        # Create schema
        create_database_schema(conn, config)

        # Ingest data
        raw_dir = config["paths"]["raw_dir"]
        ingest_prices(conn, f"{raw_dir}/prices.csv")
        ingest_vix(conn, f"{raw_dir}/vix.csv")

        logger.info("✅ Data ingestion completed successfully")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

