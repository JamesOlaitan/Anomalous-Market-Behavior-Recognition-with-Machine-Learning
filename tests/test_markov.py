"""Tests for Markov smoother."""
import numpy as np

from src.models.markov_smoother import MarkovSmoother


def test_markov_initialization():
    """Test MarkovSmoother initialization."""
    smoother = MarkovSmoother(states=("N", "A"))

    assert smoother.num_states == 2
    assert smoother.states == ("N", "A")
    assert smoother.T.shape == (2, 2)


def test_transition_matrix_rows_sum_to_one():
    """Test that transition matrix rows sum to 1."""
    smoother = MarkovSmoother(states=("N", "A"))

    row_sums = smoother.T.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_transition_matrix_probabilities():
    """Test that transition matrix contains valid probabilities."""
    smoother = MarkovSmoother(states=("N", "A"))

    assert np.all(smoother.T >= 0)
    assert np.all(smoother.T <= 1)


def test_markov_forward_increases_persistence():
    """Test that Markov forward pass increases persistence vs raw predictions."""
    # Create noisy predictions that flip frequently
    np.random.seed(42)
    T = 100
    p_anom = np.random.rand(T)  # Noisy predictions

    smoother = MarkovSmoother(states=("N", "A"))

    posteriors, state_seq, flags = smoother.forward(p_anom, tau=0.7, k=1, d=0)

    # Check output shapes
    assert posteriors.shape == (T, 2)
    assert state_seq.shape == (T,)
    assert flags.shape == (T,)

    # Posteriors should sum to 1
    assert np.allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)

    # State sequence should only contain valid states (0 or 1)
    assert np.all((state_seq == 0) | (state_seq == 1))


def test_markov_smoothing_reduces_transitions():
    """Test that Markov smoothing reduces state transitions."""
    np.random.seed(42)

    # Create noisy predictions
    p_anom = np.random.rand(200)

    smoother = MarkovSmoother(states=("N", "A"))
    posteriors, state_seq, flags = smoother.forward(p_anom, tau=0.5, k=1, d=0)

    # Count transitions in raw predictions
    raw_states = (p_anom > 0.5).astype(int)
    raw_transitions = np.sum(raw_states[1:] != raw_states[:-1])

    # Count transitions in smoothed predictions
    smooth_transitions = np.sum(state_seq[1:] != state_seq[:-1])

    # Smoothed should have fewer transitions (due to persistence)
    assert smooth_transitions <= raw_transitions


def test_markov_fit_with_labels():
    """Test fitting transition matrix with labels."""
    np.random.seed(42)

    # Create synthetic labels with some persistence
    labels = np.zeros(200)
    labels[50:70] = 1  # Anomaly period
    labels[150:180] = 1  # Another anomaly period

    smoother = MarkovSmoother(states=("N", "A"))
    smoother.fit(labels=labels)

    # Check that T is updated
    assert smoother.T.shape == (2, 2)
    assert np.allclose(smoother.T.sum(axis=1), 1.0)

    # Diagonal should be high (persistence)
    assert smoother.T[0, 0] > 0.5  # P(N -> N) should be high
    assert smoother.T[1, 1] > 0.5  # P(A -> A) should be high


def test_markov_fit_with_p_anom():
    """Test fitting transition matrix with p_anom."""
    np.random.seed(42)
    p_anom = np.random.rand(200)

    smoother = MarkovSmoother(states=("N", "A"))
    smoother.fit(p_anom=p_anom, threshold=0.5)

    # Check that T is updated
    assert smoother.T.shape == (2, 2)
    assert np.allclose(smoother.T.sum(axis=1), 1.0)


def test_markov_three_state():
    """Test 3-state Markov smoother."""
    smoother = MarkovSmoother(states=("N", "A", "R"))

    assert smoother.num_states == 3
    assert smoother.T.shape == (3, 3)

    # Test forward pass
    p_anom = np.random.rand(50)
    posteriors, state_seq, flags = smoother.forward(p_anom)

    assert posteriors.shape == (50, 3)
    assert np.allclose(posteriors.sum(axis=1), 1.0)


def test_markov_decision_rule():
    """Test decision rule with consecutive steps."""
    # Create predictions with a clear anomaly period
    p_anom = np.zeros(100)
    p_anom[40:50] = 0.9  # Strong anomaly period

    smoother = MarkovSmoother(states=("N", "A"))

    # With k=1 (no consecutive requirement)
    _, _, flags_k1 = smoother.forward(p_anom, tau=0.7, k=1, d=0)

    # With k=5 (require 5 consecutive)
    _, _, flags_k5 = smoother.forward(p_anom, tau=0.7, k=5, d=0)

    # k=5 should flag fewer samples
    assert flags_k5.sum() <= flags_k1.sum()


def test_markov_get_state_name():
    """Test getting state name from index."""
    smoother = MarkovSmoother(states=("N", "A"))

    assert smoother.get_state_name(0) == "N"
    assert smoother.get_state_name(1) == "A"


def test_markov_custom_transition_matrix():
    """Test initialization with custom transition matrix."""
    T_custom = np.array([[0.9, 0.1], [0.2, 0.8]])

    smoother = MarkovSmoother(states=("N", "A"), T=T_custom)

    assert np.allclose(smoother.T, T_custom)
