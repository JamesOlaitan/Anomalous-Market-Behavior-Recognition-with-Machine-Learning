"""Tests for feature engineering."""

import numpy as np
import pandas as pd

from src.data.features import (
    compute_returns,
    compute_rolling_correlation,
    compute_volatility,
    compute_z_scores,
)


def test_compute_returns():
    """Test returns computation."""
    # Create sample data
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "symbol": ["A"] * 5,
            "close": [100, 102, 101, 103, 102],
        }
    )

    result = compute_returns(df)

    # Check returns are computed
    assert "returns" in result.columns
    assert "log_returns" in result.columns

    # First return should be NaN
    assert pd.isna(result.iloc[0]["returns"])

    # Check second return (2% increase)
    assert np.isclose(result.iloc[1]["returns"], 0.02, atol=1e-6)


def test_compute_volatility():
    """Test volatility computation."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 100,
            "returns": np.random.randn(100) * 0.01,
        }
    )

    result = compute_volatility(df, window=20)

    assert "volatility" in result.columns
    assert not result["volatility"].isna().all()
    # Check only non-NaN values are non-negative
    assert (result["volatility"].dropna() >= 0).all()


def test_compute_rolling_correlation():
    """Test rolling correlation computation."""
    # Create sample data with two symbols
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100)
    spy_returns = np.random.randn(100) * 0.01
    xlf_returns = np.random.randn(100) * 0.01
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["SPY"] * 100 + ["XLF"] * 100,
            "returns": list(spy_returns) + list(xlf_returns),
        }
    )

    result = compute_rolling_correlation(df, reference_symbol="SPY", window=30)

    # Check rolling_corr is computed
    assert "rolling_corr" in result.columns

    # SPY should have NaN correlation (self-correlation not computed)
    spy_corr = result[result["symbol"] == "SPY"]["rolling_corr"]
    assert spy_corr.isna().all()

    # XLF should have correlation values
    xlf_corr = result[result["symbol"] == "XLF"]["rolling_corr"]
    assert not xlf_corr.isna().all()

    # Correlation should be between -1 and 1
    valid_corr = xlf_corr.dropna()
    assert (valid_corr >= -1).all() and (valid_corr <= 1).all()


def test_compute_z_scores():
    """Test z-score computation."""
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 100,
            "rolling_corr": np.random.randn(100),
        }
    )

    result = compute_z_scores(df, "rolling_corr", window=50)

    assert "corr_zscore" in result.columns

    # Z-scores should have reasonable values (mostly within -3, 3)
    zscore_valid = result["corr_zscore"].dropna()
    assert len(zscore_valid) > 0
    assert zscore_valid.abs().quantile(0.95) < 5  # 95% within 5 std devs


def test_feature_pipeline_integration():
    """Test that features can be computed in sequence."""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["SPY"] * 200 + ["XLF"] * 200,
            "close": np.cumsum(np.random.randn(400) * 0.01) + 100,
        }
    )

    # Apply transformations
    df = compute_returns(df)
    df = compute_volatility(df, window=20)
    df = compute_rolling_correlation(df, reference_symbol="SPY", window=60)
    df = compute_z_scores(df, "rolling_corr", window=120)
    df = compute_z_scores(df, "volatility", window=120)

    # Check all features exist
    expected_cols = [
        "returns",
        "log_returns",
        "volatility",
        "rolling_corr",
        "corr_zscore",
        "vol_zscore",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check no infinite values
    numeric_df = df[expected_cols].select_dtypes(include=[np.number])
    assert not np.isinf(numeric_df).any().any()
