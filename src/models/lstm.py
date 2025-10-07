"""LSTM model for anomaly detection."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AnomalyLSTM(nn.Module):
    """LSTM model for binary anomaly classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AnomalyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output logits of shape (batch_size, seq_len, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Fully connected layer
        out = self.fc(lstm_out)

        return out


def create_sequences(
    features: np.ndarray, labels: np.ndarray, seq_length: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        features: Feature array of shape (T, F)
        labels: Label array of shape (T,)
        seq_length: Sequence length (1 for point-wise)

    Returns:
        Tuple of (sequences, sequence_labels)
        - sequences: shape (N, seq_length, F)
        - sequence_labels: shape (N, seq_length, 1)
    """
    if seq_length == 1:
        # Point-wise: each sample is independent
        return features.reshape(-1, 1, features.shape[1]), labels.reshape(-1, 1, 1)

    X, y = [], []
    for i in range(len(features) - seq_length + 1):
        X.append(features[i : i + seq_length])
        y.append(labels[i : i + seq_length])

    # Reshape labels to (N, seq_length, 1) to match model output
    return np.array(X), np.array(y).reshape(-1, seq_length, 1)


def create_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    seq_length: int = 1,
) -> DataLoader:
    """
    Create PyTorch DataLoader.

    Args:
        features: Feature array
        labels: Label array
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seq_length: Sequence length for LSTM

    Returns:
        DataLoader
    """
    # Create sequences
    X, y = create_sequences(features, labels, seq_length)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def train_epoch(
    model: AnomalyLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """
    Train for one epoch.

    Args:
        model: LSTM model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Compute loss
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(
    model: AnomalyLSTM, dataloader: DataLoader, criterion: nn.Module, device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model.

    Args:
        model: LSTM model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (loss, predictions, labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)

            # Compute loss
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # Get predictions
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, all_preds, all_labels


def predict(
    model: AnomalyLSTM, features: np.ndarray, device: str, seq_length: int = 1
) -> np.ndarray:
    """
    Generate predictions.

    Args:
        model: Trained LSTM model
        features: Feature array of shape (T, F)
        device: Device to run on
        seq_length: Sequence length

    Returns:
        Anomaly probabilities of shape (T,)
    """
    model.eval()

    # Create sequences
    X, _ = create_sequences(features, np.zeros(len(features)), seq_length)
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs)

    # Reshape back to (T,)
    probs = probs.cpu().numpy().reshape(-1)

    return probs
