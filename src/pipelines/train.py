"""Training pipeline for LSTM anomaly detection model."""

import os

import duckdb
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.lstm import (
    AnomalyLSTM,
    create_dataloader,
    evaluate_model,
    train_epoch,
)
from src.models.markov_smoother import MarkovSmoother
from src.utils.config import get_device, load_config
from src.utils.io import save_json, save_model, save_pickle
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seeds

logger = setup_logging(name=__name__)


def load_data_from_db(conn: duckdb.DuckDBPyConnection, config: dict) -> tuple:
    """
    Load features and labels from database and split into train/val/test.

    Args:
        conn: DuckDB connection
        config: Configuration dictionary

    Returns:
        Tuple of (train_data, val_data, test_data, scaler)
    """
    logger.info("Loading data from database")

    # Load features and labels
    query = """
        SELECT f.*, l.label
        FROM features f
        JOIN labels l ON f.date = l.date AND f.symbol = l.symbol
        ORDER BY f.date, f.symbol
    """
    df = conn.execute(query).df()

    logger.info(f"Loaded {len(df)} records")

    # Feature columns (exclude date, symbol, label)
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

    # Split by date (chronological)
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]

    dates = sorted(df["date"].unique())

    train_end_idx = int(len(dates) * train_ratio)
    val_end_idx = int(len(dates) * (train_ratio + val_ratio))

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    logger.info(f"Train: {len(train_df)} samples ({len(train_dates)} dates)")
    logger.info(f"Val: {len(val_df)} samples ({len(val_dates)} dates)")
    logger.info(f"Test: {len(test_df)} samples ({len(test_dates)} dates)")

    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Check class balance
    train_pos_ratio = y_train.mean()
    val_pos_ratio = y_val.mean()
    test_pos_ratio = y_test.mean()

    logger.info(f"Train positive ratio: {train_pos_ratio:.2%}")
    logger.info(f"Val positive ratio: {val_pos_ratio:.2%}")
    logger.info(f"Test positive ratio: {test_pos_ratio:.2%}")

    train_data = (X_train, y_train, train_df)
    val_data = (X_val, y_val, val_df)
    test_data = (X_test, y_test, test_df)

    return train_data, val_data, test_data, scaler


def train_model(config: dict) -> None:
    """
    Main training pipeline.

    Args:
        config: Configuration dictionary
    """
    # Set seeds
    set_seeds(config["seed"])

    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Connect to database
    db_path = config["sql"]["database"]
    conn = duckdb.connect(db_path)

    try:
        # Load data
        train_data, val_data, _test_data, scaler = load_data_from_db(conn, config)
        X_train, y_train, _ = train_data
        X_val, y_val, _ = val_data

        # Create dataloaders
        batch_size = config["model"]["batch_size"]
        train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # Initialize model
        input_size = X_train.shape[1]
        model = AnomalyLSTM(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
        ).to(device)

        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss function with class imbalance handling
        pos_weight = torch.tensor([config["model"]["pos_weight"]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

        # Training loop
        epochs = config["model"]["epochs"]
        patience = config["model"]["early_stopping_patience"]
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = f"{config['paths']['models_dir']}/best_model.pt"

        os.makedirs(config["paths"]["models_dir"], exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)

            # Compute metrics
            val_auc = roc_auc_score(val_labels.flatten(), val_preds.flatten())
            val_pr_auc = average_precision_score(val_labels.flatten(), val_preds.flatten())

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val ROC-AUC: {val_auc:.4f}, "
                f"Val PR-AUC: {val_pr_auc:.4f}"
            )

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(model, best_model_path)
                logger.info(f"✅ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model")

        # Save scaler
        scaler_path = f"{config['paths']['models_dir']}/scaler.pkl"
        save_pickle(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Save model config
        model_config = {
            "input_size": input_size,
            "hidden_size": config["model"]["hidden_size"],
            "num_layers": config["model"]["num_layers"],
            "dropout": config["model"]["dropout"],
            "bidirectional": config["model"]["bidirectional"],
        }
        model_config_path = f"{config['paths']['models_dir']}/model_config.json"
        save_json(model_config, model_config_path)
        logger.info(f"Saved model config to {model_config_path}")

        # Fit Markov smoother on validation set
        logger.info("Fitting Markov smoother on validation set")
        val_preds_flat = val_preds.flatten()
        val_labels_flat = val_labels.flatten()

        markov_config = config["markov"]
        smoother = MarkovSmoother(
            states=tuple(["N", "A"][: markov_config["num_states"]]),
            T=markov_config.get("transition_matrix"),
            alpha=markov_config["alpha"],
        )

        # Fit transition matrix
        smoother.fit(labels=val_labels_flat, p_anom=val_preds_flat)

        # Save smoother
        smoother_path = f"{config['paths']['models_dir']}/markov_smoother.pkl"
        save_pickle(smoother, smoother_path)
        logger.info(f"Saved Markov smoother to {smoother_path}")

        logger.info("✅ Training pipeline completed successfully")

    finally:
        conn.close()


def main():
    """Main function."""
    config = load_config()
    train_model(config)


if __name__ == "__main__":
    main()
