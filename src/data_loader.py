# src/data_loader.py

"""
Data Loader Module
==================

This module is for functions to download historical financial data for
specified tickers and save it to CSV files in the data directory.
"""

import os
from typing import List
import pandas as pd
import yfinance as yf


def download_data(tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """
    Downloads historical data for given tickers using yfinance.

    :param tickers: List of ticker symbols to download.
    :param start_date: Start date for the data in 'YYYY-MM-DD' format.
    :param end_date: End date for the data in 'YYYY-MM-DD' format.
    :param interval: Data interval (e.g., '1d' for daily data).
    :return: DataFrame containing the adjusted closing prices.
    """
    # Downloads data
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        threads=True
    )

    # Keeps only the 'Adj Close' column
    adj_close = data['Adj Close']
    return adj_close


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the DataFrame to a CSV file.

    :param df: DataFrame to save.
    :param file_path: Destination file path.
    """
    # Creates a directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")


def main():
    """
    Main function to download and save data.
    """
    # Tickers and date range
    tickers = ['VOO', 'VNQ', '^VIX']  # VOO: S&P 500 ETF, VNQ: REITs ETF, ^VIX: VIX Index
    start_date = '2010-09-09'
    end_date = '2024-09-09'

    closing_prices = download_data(tickers, start_date, end_date)

    # Displays the first few rows
    print(closing_prices.head())

    # Saves data to CSV
    save_data(closing_prices, 'data/raw/closing_prices.csv')


if __name__ == '__main__':
    main()