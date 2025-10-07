"""End-to-end integration test."""
import os
import tempfile

import duckdb
import numpy as np
import pandas as pd
import torch

from src.data.features import (
    compute_returns,
    compute_rolling_correlation,
    compute_volatility,
    compute_z_scores,
    merge_vix,
)
from src.models.lstm import AnomalyLSTM, create_dataloader, train_epoch
from src.models.markov_smoother import MarkovSmoother
from src.utils.seed import set_seeds


def test_end_to_end_pipeline():
    """Test the entire pipeline from features to evaluation."""
    set_seeds(42)

    # 1. Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500)
    symbols = ["SPY", "XLF"]

    # Create price data
    prices_data = []
    for symbol in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        for i, date in enumerate(dates):
            prices_data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": prices[i],
                    "open": prices[i] * 0.99,
                    "high": prices[i] * 1.01,
                    "low": prices[i] * 0.98,
                    "volume": int(1e6),
                }
            )

    prices_df = pd.DataFrame(prices_data)

    # Create VIX data
    vix_df = pd.DataFrame(
        {"date": dates, "vix": 20 + np.random.randn(len(dates)) * 5}
    )

    # 2. Feature engineering
    features = compute_returns(prices_df)
    features = compute_volatility(features, window=20)
    features = compute_rolling_correlation(features, reference_symbol="SPY", window=60)
    features = compute_z_scores(features, "rolling_corr", window=120)
    features = compute_z_scores(features, "volatility", window=120)
    features = merge_vix(features, vix_df)

    # Drop NaNs
    features = features.dropna(
        subset=[
            "returns",
            "volatility",
            "rolling_corr",
            "corr_zscore",
            "vol_zscore",
            "vix",
        ]
    )

    assert len(features) > 0, "No features after preprocessing"

    # 3. Create synthetic labels (correlation breakdown)
    vol_threshold = features["volatility"].quantile(0.8)
    features["label"] = (
        (features["rolling_corr"] < 0.2) & (features["volatility"] > vol_threshold)
    ).astype(int)

    # 4. Prepare data for training
    feature_cols = [
        "returns",
        "log_returns",
        "volatility",
        "rolling_corr",
        "corr_zscore",
        "vol_zscore",
        "vix",
        "vix_delta",
    ]

    X = features[feature_cols].values
    y = features["label"].values

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Split
    train_size = int(0.7 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # 5. Train LSTM (few epochs for speed)
    model = AnomalyLSTM(input_size=X_train.shape[1], hidden_size=16, num_layers=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)

    for epoch in range(3):  # Just 3 epochs for testing
        loss = train_epoch(model, train_loader, criterion, optimizer, "cpu")
        assert not np.isnan(loss), f"Loss is NaN at epoch {epoch}"

    # 6. Generate predictions
    model.eval()
    with torch.no_grad():
        X_val_seq, _ = create_dataloader(
            X_val, y_val, batch_size=len(X_val), shuffle=False
        )
        X_batch, _ = next(iter(X_val_seq))
        outputs = model(X_batch)
        p_anom = torch.sigmoid(outputs).cpu().numpy().flatten()

    assert len(p_anom) == len(y_val)
    assert (p_anom >= 0).all() and (p_anom <= 1).all()

    # 7. Apply Markov smoother
    smoother = MarkovSmoother(states=("N", "A"))
    smoother.fit(labels=y_val, p_anom=p_anom)

    posteriors, state_seq, flags = smoother.forward(p_anom, tau=0.7, k=3, d=0)

    assert posteriors.shape == (len(y_val), 2)
    assert state_seq.shape == (len(y_val),)
    assert flags.shape == (len(y_val),)

    # 8. Compute metrics
    from sklearn.metrics import f1_score, roc_auc_score

    f1 = f1_score(y_val, flags)
    auc_score = roc_auc_score(y_val, p_anom)

    # Metrics should exist (values may vary)
    assert 0 <= f1 <= 1
    assert 0 <= auc_score <= 1

    print(f"End-to-end test passed! F1: {f1:.4f}, AUC: {auc_score:.4f}")


def test_database_operations():
    """Test database creation and operations."""
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        conn = duckdb.connect(db_path)

        # Create schema
        conn.execute(
            """
            CREATE TABLE test_table (
                date DATE,
                value DOUBLE
            )
        """
        )

        # Insert data
        test_data = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=10), "value": range(10)}
        )
        conn.register("test_data", test_data)
        conn.execute("INSERT INTO test_table SELECT * FROM test_data")

        # Query data
        result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        assert result == 10

        conn.close()
