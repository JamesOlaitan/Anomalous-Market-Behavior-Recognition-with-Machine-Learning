"""
Training Module
===============

This module handles data loading, feature scaling, model creation, and training of the autoencoder.
It prepares the input features, normalizes them, and trains the model to detect anomalies in market data.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from model import build_autoencoder
import tensorflow as tf


def load_features(file_path: str) -> pd.DataFrame:
    """
    Loads preprocessed features from a CSV file.

    :param file_path: Path to the CSV file containing processed features.
    :return: DataFrame containing preprocessed features.
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df


def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scales the features using MinMaxScaler.

    :param df: DataFrame containing features to scale.
    :return: Tuple of (scaled feature array, scaler object).
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    return scaled_data, scaler


def train_autoencoder(features: np.ndarray, epochs: int = 50, batch_size: int = 32) -> tf.keras.Model:
    """
    Trains the autoencoder model on the provided features.

    :param features: Numpy array of scaled features.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :return: Trained autoencoder model.
    """
    input_dim = features.shape[1]

    # Builds the autoencoder
    autoencoder = build_autoencoder(input_dim=input_dim)

    # Trains the model on the entire dataset
    history = autoencoder.fit(
        features, features,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    return autoencoder


def save_model(model: tf.keras.Model, model_path: str) -> None:
    """
    Saves the trained model to a specified path.

    :param model: Trained Keras model.
    :param model_path: File path to save the model.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")


def main():
    """
    Main function to orchestrate the model training process.
    """
    # File path to preprocessed data
    processed_data_path = 'data/processed/features.csv'

    # Loads features
    features_df = load_features(processed_data_path)
    print("Features loaded successfully.")

    # Scales features
    scaled_features, scaler = scale_features(features_df)
    print("Features scaled successfully.")

    # Trains the autoencoder
    autoencoder = train_autoencoder(scaled_features, epochs=50, batch_size=32)
    print("Autoencoder trained successfully.")

    # Saves the trained model
    model_path = 'models/autoencoder_model'
    save_model(autoencoder, model_path)


if __name__ == '__main__':
    main()