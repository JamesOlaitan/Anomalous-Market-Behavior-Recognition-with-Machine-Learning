"""Tests for LSTM model."""
import numpy as np
import pytest
import torch

from src.models.lstm import AnomalyLSTM, create_sequences, predict


def test_lstm_initialization():
    """Test LSTM model initialization."""
    model = AnomalyLSTM(input_size=8, hidden_size=32, num_layers=1)

    assert model.hidden_size == 32
    assert model.num_layers == 1
    assert not model.bidirectional

    # Check model can be created
    assert isinstance(model, torch.nn.Module)


def test_lstm_forward_pass():
    """Test LSTM forward pass."""
    model = AnomalyLSTM(input_size=8, hidden_size=32, num_layers=1)
    model.eval()

    # Create dummy input (batch_size=4, seq_len=10, input_size=8)
    x = torch.randn(4, 10, 8)

    with torch.no_grad():
        output = model(x)

    # Output should have shape (batch_size, seq_len, 1)
    assert output.shape == (4, 10, 1)


def test_lstm_bidirectional():
    """Test bidirectional LSTM."""
    model = AnomalyLSTM(
        input_size=8, hidden_size=32, num_layers=1, bidirectional=True
    )

    x = torch.randn(2, 5, 8)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 5, 1)


def test_create_sequences():
    """Test sequence creation."""
    features = np.random.randn(100, 8)
    labels = np.random.randint(0, 2, size=100)

    # Test with seq_length=1 (point-wise)
    X, y = create_sequences(features, labels, seq_length=1)
    assert X.shape == (100, 1, 8)
    assert y.shape == (100, 1)

    # Test with seq_length=10
    X, y = create_sequences(features, labels, seq_length=10)
    assert X.shape == (91, 10, 8)  # 100 - 10 + 1
    assert y.shape == (91, 10)


def test_predict():
    """Test prediction function."""
    model = AnomalyLSTM(input_size=8, hidden_size=16, num_layers=1)
    model.eval()

    features = np.random.randn(50, 8)

    probs = predict(model, features, device="cpu", seq_length=1)

    # Check output shape
    assert probs.shape == (50,)

    # Check probabilities are in [0, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


def test_lstm_overfit_tiny_dataset():
    """Test that LSTM can overfit a tiny synthetic dataset."""
    # Create a tiny synthetic dataset
    np.random.seed(42)
    torch.manual_seed(42)

    # 50 samples, 4 features
    X = np.random.randn(50, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # Simple rule

    # Create model
    model = AnomalyLSTM(input_size=4, hidden_size=16, num_layers=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train for a few epochs
    X_seq, y_seq = create_sequences(X, y, seq_length=1)
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # Check that loss decreased significantly
    final_loss = loss.item()
    assert final_loss < 0.5  # Should be able to overfit


def test_lstm_output_shapes_batch():
    """Test LSTM with different batch sizes."""
    model = AnomalyLSTM(input_size=5, hidden_size=20, num_layers=2)
    model.eval()

    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 1, 5)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 1, 1)

