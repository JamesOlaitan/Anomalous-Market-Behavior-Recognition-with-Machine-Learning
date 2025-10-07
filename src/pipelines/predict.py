"""Prediction pipeline to generate and save anomaly scores."""
import duckdb
import pandas as pd
import torch

from src.models.lstm import AnomalyLSTM, predict
from src.utils.config import get_device, load_config
from src.utils.io import load_json, load_pickle
from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


def load_trained_model(config: dict, device: str) -> tuple:
    """
    Load trained model, scaler, and Markov smoother.

    Args:
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Tuple of (model, scaler, smoother, model_config)
    """
    models_dir = config["paths"]["models_dir"]

    # Load model config
    model_config_path = f"{models_dir}/model_config.json"
    model_config = load_json(model_config_path)
    logger.info(f"Loaded model config from {model_config_path}")

    # Initialize and load model
    model = AnomalyLSTM(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        bidirectional=model_config["bidirectional"],
    ).to(device)

    model_path = f"{models_dir}/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Load scaler
    scaler_path = f"{models_dir}/scaler.pkl"
    scaler = load_pickle(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")

    # Load Markov smoother
    smoother_path = f"{models_dir}/markov_smoother.pkl"
    smoother = load_pickle(smoother_path)
    logger.info(f"Loaded Markov smoother from {smoother_path}")

    return model, scaler, smoother, model_config


def generate_predictions(config: dict) -> None:
    """
    Generate predictions and save to database.

    Args:
        config: Configuration dictionary
    """
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Load model and smoother
    model, scaler, smoother, _model_config = load_trained_model(config, device)

    # Connect to database
    db_path = config["sql"]["database"]
    conn = duckdb.connect(db_path)

    try:
        # Load features
        logger.info("Loading features from database")
        query = """
            SELECT *
            FROM features
            ORDER BY date, symbol
        """
        df = conn.execute(query).df()
        logger.info(f"Loaded {len(df)} feature records")

        # Feature columns
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

        # Scale features
        X = scaler.transform(df[feature_cols].values)

        # Generate LSTM predictions
        logger.info("Generating LSTM predictions")
        p_anom = predict(model, X, device, seq_length=1)

        # Apply Markov smoother
        logger.info("Applying Markov smoother")
        markov_config = config["markov"]
        posteriors, state_seq, _anomaly_flags = smoother.forward(
            p_anom,
            tau=markov_config["decision_threshold"],
            k=markov_config["consecutive_steps"],
            d=markov_config["dwell_steps"],
        )

        # Prepare results
        predictions_df = pd.DataFrame(
            {
                "date": df["date"],
                "symbol": df["symbol"],
                "p_anom": p_anom,
                "post_normal": posteriors[:, 0],
                "post_anomalous": (
                    posteriors[:, 1] if posteriors.shape[1] > 1
                    else posteriors[:, 0]
                ),
                "state": [smoother.get_state_name(s) for s in state_seq],
            }
        )

        # Save to database
        logger.info("Saving predictions to database")
        conn.execute("DELETE FROM predictions")
        conn.register("predictions_df", predictions_df)
        conn.execute("INSERT INTO predictions SELECT * FROM predictions_df")

        pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        logger.info(f"✅ Saved {pred_count} prediction records to database")

        # Summary statistics
        anomaly_ratio_lstm = (p_anom > 0.5).mean()

        logger.info(f"LSTM anomaly ratio (p > 0.5): {anomaly_ratio_lstm:.2%}")

    finally:
        conn.close()


def main():
    """Main function."""
    config = load_config()
    generate_predictions(config)


if __name__ == "__main__":
    main()
