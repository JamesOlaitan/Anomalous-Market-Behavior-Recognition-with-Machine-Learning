# src/preprocess.py

"""
Data Preprocessing Module
=========================

This module includes functions to preprocess financial data for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns for each asset.

    :param df: DataFrame with adjusted closing prices.
    :return: DataFrame with daily returns.
    """
    returns = df.pct_change().dropna()
    return returns


def compute_rolling_correlation(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Computes rolling correlation between two assets.

    :param df: DataFrame with returns of two assets.
    :param window: Window size for rolling correlation.
    :return: Series containing rolling correlation values.
    """
    asset1 = df.iloc[:, 0]
    asset2 = df.iloc[:, 1]
    rolling_corr = asset1.rolling(window).corr(asset2)
    return rolling_corr


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame by forward filling, backward filling, and interpolation.

    :param df: DataFrame with potential missing values.
    :return: DataFrame with missing values handled.
    """
    # Checks for missing values
    if df.isnull().values.any():
        print("Missing values detected in returns. Handling missing data...")
        # Forward fill
        df = df.ffill()
        # Backward fill
        df = df.bfill()
        # Interpolates remaining missing values
        df = df.interpolate(method='linear')
    else:
        print("No missing values detected in returns.")
    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses data for modeling.

    :param file_path: Path to the CSV file with raw data.
    :return: DataFrame with preprocessed features.
    """
    # Loads data
    df = load_data(file_path)
    print("Data loaded successfully.")

    # Calculates returns
    returns = calculate_returns(df[['VOO', 'VNQ']])
    print("Daily returns calculated.")

    # Handles missing values in returns
    returns = handle_missing_values(returns)

    # Computes rolling correlation
    rolling_corr = compute_rolling_correlation(returns)
    print("Rolling correlation computed.")

    # Aligns VIX data with returns index
    vix = df['^VIX'].reindex(returns.index)
    vix = vix.fillna(method='ffill')  # Forward fill VIX data if necessary
    print("VIX data aligned with returns.")

    # Combines features into a single DataFrame
    features = pd.DataFrame({
        'VOO_Returns': returns['VOO'],
        'VNQ_Returns': returns['VNQ'],
        'Rolling_Corr': rolling_corr,
        'VIX': vix
    })

    # Handles any remaining missing values
    features = handle_missing_values(features)

    # Drops any rows with NaN values (if any remain)
    features = features.dropna()

    print("Preprocessing complete.")
    return features


def main():
    """
    Main function to preprocess data.
    """
    # File path to raw data
    raw_data_path = 'data/raw/closing_prices.csv'

    # Preprocesses data
    features = preprocess_data(raw_data_path)

    # Saves processed data
    processed_data_path = 'data/processed/features.csv'
    features.to_csv(processed_data_path)
    print(f"Preprocessed data saved to {processed_data_path}")


if __name__ == '__main__':
    main()