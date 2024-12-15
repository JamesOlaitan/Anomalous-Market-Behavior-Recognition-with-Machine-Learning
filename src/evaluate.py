"""
Evaluation and Validation Module
===============================

This module refines and validates the anomaly detection model by:
- Calculating reconstruction errors from the trained autoencoder.
- Determining a threshold to classify anomalies.
- Identifying anomalies based on reconstruction errors.
- Suggesting validation by comparing detected anomalies with known historical events.
- Providing guidance for hyperparameter tuning to improve model performance.

Assumptions:
------------
- The preprocessed features have already been scaled.
- Ground-truth annotations or historical event data (e.g., known market crashes) can be used externally
  to validate detected anomalies.
- Hyperparameter tuning is demonstrated conceptually.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def load_features(file_path: str) -> pd.DataFrame:
    """
    Loads preprocessed feature data from a CSV file.

    :param file_path: Path to the CSV file with preprocessed features.
    :return: DataFrame containing features indexed by Date.
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df


def compute_reconstruction_errors(model, features: np.ndarray) -> np.ndarray:
    """
    Computes reconstruction errors for each sample using the trained autoencoder.

    :param model: Trained autoencoder model.
    :param features: Numpy array of scaled feature data.
    :return: Numpy array of reconstruction errors, one for each sample.
    """
    reconstructed = model.predict(features, verbose=0)
    errors = np.mean(np.square(features - reconstructed), axis=1)
    return errors


def determine_anomaly_threshold(errors: np.ndarray, quantile: float = 0.99) -> float:
    """
    Determines an anomaly threshold using a given quantile of the reconstruction errors.

    :param errors: Numpy array of reconstruction errors.
    :param quantile: Quantile for threshold determination (e.g., 0.99 for 99th percentile).
    :return: Threshold value for anomaly classification.
    """
    threshold = np.quantile(errors, quantile)
    return threshold


def identify_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Identifies anomalies by comparing reconstruction errors to the threshold.

    :param errors: Numpy array of reconstruction errors.
    :param threshold: Threshold value for anomaly detection.
    :return: Binary array indicating anomalies (1) and normal points (0).
    """
    anomalies = (errors > threshold).astype(int)
    return anomalies


def compare_with_known_events(anomalies: np.ndarray, dates: pd.DatetimeIndex, events: Dict[str, pd.Timestamp]) -> None:
    """
    Compares detected anomalies with known historical events (e.g., financial crises).

    :param anomalies: Binary array indicating anomalies.
    :param dates: Datetime index associated with the feature data.
    :param events: Dictionary of event descriptions to known event dates.
                   Example: {"2008 Financial Crisis Start": pd.Timestamp("2008-09-15")}
    :return: None. Prints out overlaps between anomalies and known events.
    """
    for event_desc, event_date in events.items():
        # Find anomalies near the event date
        close_dates = (dates >= (event_date - pd.Timedelta(days=10))) & (dates <= (event_date + pd.Timedelta(days=10)))
        if np.any(anomalies[close_dates] == 1):
            print(f"Anomalies detected near {event_desc} ({event_date.date()}).")
        else:
            print(f"No anomalies detected near {event_desc} ({event_date.date()}).")


def hyperparameter_tuning_guidance() -> None:
    """
    Provides guidance on hyperparameter tuning steps for improving model performance.
    (conceptual; not a full implementation).

    Steps:
    - Modify the autoencoder architecture (layers, neurons, activation functions) in model.py.
    - Adjust learning rate in the optimizer (e.g., Adam) when compiling the model.
    - Retrain the model and evaluate reconstruction errors again.
    - Use a validation set or time-series split to compare different configurations.
    - Repeat the process until metrics (anomaly detection accuracy) improve.
    """
    print("Hyperparameter Tuning Guidance:")
    print("- Experiment with different encoding_dims in the autoencoder.")
    print("- Try adding more layers or changing activation functions (e.g., relu, elu).")
    print("- Adjust the optimizer's learning rate (e.g., learning_rate=0.001, 0.0001).")
    print("- Evaluate each configuration using a validation set and track precision, recall, and F1-score.")
    print("- Incorporate domain knowledge to refine features (e.g., additional financial indicators).")


def main():
    """
    Main function for model refinement and validation.
    """
    # Loads preprocessed features
    features_path = 'data/processed/features.csv'
    features_df = load_features(features_path)
    print("Features loaded.")

    # Converts features to numpy array
    scaled_features = features_df.values
    dates = features_df.index

    # Loads trained model
    model_path = 'models/autoencoder_model'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Please run training first.")

    model = load_model(model_path)
    print("Model loaded successfully.")

    # Compute reconstruction errors
    errors = compute_reconstruction_errors(model, scaled_features)
    print("Reconstruction errors computed.")

    # Determines anomaly threshold
    threshold = determine_anomaly_threshold(errors, quantile=0.99)
    print(f"Anomaly threshold determined: {threshold}")

    # Identifies anomalies
    anomalies = identify_anomalies(errors, threshold)
    anomaly_ratio = np.mean(anomalies) * 100
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} samples ({anomaly_ratio:.2f}% of data).")

    # Compare with known events (example)
    known_events = {
        "COVID-19 Market Crash": pd.Timestamp("2020-03-15"),
        "2008 Financial Crisis Start": pd.Timestamp("2008-09-15")
    }
    compare_with_known_events(anomalies, dates, known_events)

    # Hyperparameter tuning guidance
    hyperparameter_tuning_guidance()


if __name__ == '__main__':
    main()