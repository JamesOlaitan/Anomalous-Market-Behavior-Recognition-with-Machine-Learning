"""
Model Definition Module
=======================

This module defines the autoencoder architecture for anomaly detection in financial data.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int, encoding_dim: int = 8) -> keras.Model:
    """
    Builds an autoencoder model using the Keras API in TensorFlow.

    :param input_dim: Number of input features.
    :param encoding_dim: Dimension of the bottleneck encoding layer.
    :return: Compiled Keras model representing the autoencoder.
    """
    # Encoder
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    # Autoencoder Model
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name="autoencoder")

    # Compiles the model with Mean Squared Error loss and Adam optimizer
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder