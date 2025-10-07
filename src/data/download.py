"""Download financial data from Yahoo Finance."""
from typing import List

import pandas as pd
import yfinance as yf

from src.utils.config import load_config
from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


def download_price_data(
    symbols: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Download OHLCV data for given symbols.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading data for {len(symbols)} symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")

    data = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        threads=True,
        progress=True,
    )

    if len(symbols) == 1:
        # yfinance returns different structure for single symbol
        data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])

    # Reshape to long format: Date, Symbol, Open, High, Low, Close, Volume
    result_list = []
    for symbol in symbols:
        try:
            symbol_data = data.xs(symbol, axis=1, level=1)
            symbol_data["symbol"] = symbol
            symbol_data = symbol_data.reset_index()
            symbol_data.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "symbol",
            ]
            result_list.append(symbol_data)
        except KeyError:
            logger.warning(f"No data found for symbol {symbol}")

    result = pd.concat(result_list, ignore_index=True)
    logger.info(f"Downloaded {len(result)} rows of data")

    return result


def download_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download VIX index data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with VIX data
    """
    logger.info("Downloading VIX data")

    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    vix = vix[["Close"]].reset_index()
    vix.columns = ["date", "vix"]

    logger.info(f"Downloaded {len(vix)} rows of VIX data")

    return vix


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in financial data.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing values handled
    """
    initial_count = df.isnull().sum().sum()

    if initial_count > 0:
        logger.info(f"Handling {initial_count} missing values")

        # Forward fill then backward fill
        df = df.ffill().bfill()

        # Interpolate any remaining
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

        final_count = df.isnull().sum().sum()
        logger.info(f"Missing values after handling: {final_count}")
    else:
        logger.info("No missing values detected")

    return df


def main():
    """Main function to download and save data."""
    config = load_config()

    # Extract config
    symbols = config["data"]["symbols"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_dir = config["paths"]["raw_dir"]

    # Download price data
    prices = download_price_data(symbols, start_date, end_date)
    prices = handle_missing_data(prices)

    # Save raw prices
    prices_file = f"{raw_dir}/prices.csv"
    prices.to_csv(prices_file, index=False)
    logger.info(f"Saved prices to {prices_file}")

    # Download VIX data
    vix = download_vix_data(start_date, end_date)
    vix = handle_missing_data(vix)

    # Save VIX
    vix_file = f"{raw_dir}/vix.csv"
    vix.to_csv(vix_file, index=False)
    logger.info(f"Saved VIX data to {vix_file}")

    logger.info("✅ Data download completed successfully")


if __name__ == "__main__":
    main()
