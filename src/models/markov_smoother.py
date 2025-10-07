"""Simple Markov (HMM-lite) temporal smoother for LSTM anomaly scores."""

from typing import Optional, Tuple

import numpy as np

from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


class MarkovSmoother:
    """
    Simple 2-3 state Markov smoother for temporal smoothing of anomaly probabilities.

    States: Normal (N), Anomalous (A), optionally Recovery (R)
    """

    def __init__(
        self,
        states: Tuple[str, ...] = ("N", "A"),
        T: Optional[np.ndarray] = None,
        alpha: float = 50.0,
    ):
        """
        Initialize Markov smoother.

        Args:
            states: State names (default: ("N", "A") for 2-state)
            T: Transition matrix (if None, uses default or learns from data)
            alpha: Dirichlet prior concentration (for learning T)
        """
        self.states = states
        self.num_states = len(states)
        self.alpha = alpha

        if T is not None:
            self.T = np.array(T)
            self._validate_transition_matrix()
        else:
            self.T = self._default_transition_matrix()

        logger.info(f"Initialized MarkovSmoother with {self.num_states} states")
        logger.info(f"States: {self.states}")
        logger.info(f"Transition matrix:\n{self.T}")

    def _default_transition_matrix(self) -> np.ndarray:
        """
        Create default transition matrix with high self-transition probabilities.

        Returns:
            Transition matrix
        """
        if self.num_states == 2:
            # 2-state: N, A
            # High persistence in each state
            T = np.array([[0.97, 0.03], [0.15, 0.85]])
        elif self.num_states == 3:
            # 3-state: N, A, R (Recovery)
            T = np.array(
                [
                    [0.95, 0.03, 0.02],  # from N
                    [0.05, 0.80, 0.15],  # from A
                    [0.70, 0.10, 0.20],  # from R
                ]
            )
        else:
            # General: uniform with higher diagonal
            T = np.ones((self.num_states, self.num_states)) * 0.1
            np.fill_diagonal(T, 0.7)
            # Normalize
            T = T / T.sum(axis=1, keepdims=True)

        return T

    def _validate_transition_matrix(self) -> None:
        """Validate that transition matrix is properly formed."""
        assert self.T.shape == (
            self.num_states,
            self.num_states,
        ), f"T shape mismatch: {self.T.shape}"

        row_sums = self.T.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"T rows don't sum to 1: {row_sums}"

        assert np.all(self.T >= 0) and np.all(self.T <= 1), "T contains invalid probabilities"

    def fit(
        self,
        labels: Optional[np.ndarray] = None,
        p_anom: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Estimate transition matrix from labels or thresholded p_anom.

        Args:
            labels: Ground truth labels (0/1) if available
            p_anom: Anomaly probabilities from LSTM
            threshold: Threshold for converting p_anom to binary states
        """
        if labels is not None:
            state_seq = labels
        elif p_anom is not None:
            state_seq = (p_anom > threshold).astype(int)
        else:
            logger.warning("No data provided for fitting, using default T")
            return

        # Count transitions with Dirichlet prior
        counts = np.zeros((self.num_states, self.num_states))

        # Add prior (concentrated on diagonal for persistence)
        prior = np.ones((self.num_states, self.num_states))
        for i in range(self.num_states):
            prior[i, i] = self.alpha * 0.8 / self.num_states  # 80% to diagonal
            prior[i, :] += self.alpha * 0.2 / (self.num_states**2)  # 20% spread uniformly

        counts += prior

        # Count observed transitions
        for t in range(len(state_seq) - 1):
            from_state = int(state_seq[t])
            to_state = int(state_seq[t + 1])

            if from_state < self.num_states and to_state < self.num_states:
                counts[from_state, to_state] += 1

        # Normalize to get probabilities
        self.T = counts / counts.sum(axis=1, keepdims=True)

        logger.info(f"Fitted transition matrix from {len(state_seq)} observations")
        logger.info(f"Transition matrix:\n{self.T}")

    def forward(
        self,
        p_anom: np.ndarray,
        tau: float = 0.7,
        k: int = 3,
        d: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through Markov chain to smooth anomaly probabilities.

        Args:
            p_anom: LSTM anomaly probabilities of shape (T,)
            tau: Decision threshold for P(A)
            k: Consecutive steps above threshold to flag
            d: Dwell steps for confirmation

        Returns:
            Tuple of:
                - posteriors: State posteriors of shape (T, S)
                - state_seq: Most likely state sequence of shape (T,)
                - anomaly_flags: Binary anomaly flags of shape (T,)
        """
        T_len = len(p_anom)
        posteriors = np.zeros((T_len, self.num_states))

        # Initial prior (uniform)
        prior = np.ones(self.num_states) / self.num_states

        for t in range(T_len):
            # Compute emission probabilities
            if self.num_states == 2:
                # 2-state: N, A
                lik = np.array([1 - p_anom[t], p_anom[t]])
            elif self.num_states == 3:
                # 3-state: N, A, R
                # R emission is mixture
                lik = np.array(
                    [
                        1 - p_anom[t],
                        p_anom[t],
                        0.5 * p_anom[t] + 0.5 * (1 - p_anom[t]),
                    ]
                )
            else:
                # General: uniform
                lik = np.ones(self.num_states)

            # Add small epsilon to avoid zero probabilities
            lik = np.clip(lik, 1e-10, 1.0)

            # Forward update: prior_t = posterior_{t-1} @ T
            prior_t = prior @ self.T

            # Posterior: prior * likelihood, normalized
            post_t = prior_t * lik
            post_t = post_t / (post_t.sum() + 1e-10)

            posteriors[t] = post_t

            # Update prior for next step
            prior = post_t

        # Determine most likely state sequence (greedy)
        state_seq = np.argmax(posteriors, axis=1)

        # Apply decision rule for anomaly flags
        # Flag anomaly if P(A) > tau for k consecutive steps
        if self.num_states >= 2:
            p_anomalous = posteriors[:, 1]  # P(A) is second state
        else:
            p_anomalous = posteriors[:, 0]

        anomaly_flags = self._apply_decision_rule(p_anomalous, tau, k, d)

        return posteriors, state_seq, anomaly_flags

    def _apply_decision_rule(
        self, p_anomalous: np.ndarray, tau: float, k: int, d: int
    ) -> np.ndarray:
        """
        Apply decision rule to flag anomalies.

        Args:
            p_anomalous: P(A) over time
            tau: Threshold
            k: Consecutive steps
            d: Dwell steps for confirmation

        Returns:
            Binary anomaly flags
        """
        flags = np.zeros(len(p_anomalous), dtype=int)

        # Simple rule: flag if above threshold for k consecutive steps
        above_threshold = (p_anomalous > tau).astype(int)

        for i in range(len(above_threshold) - k + 1):
            if np.sum(above_threshold[i : i + k]) == k:
                # Flag this window
                flags[i : i + k] = 1

        # Optional dwell confirmation
        if d > 0:
            confirmed_flags = np.zeros_like(flags)
            for i in range(len(flags) - d + 1):
                if np.sum(flags[i : i + d]) >= d:
                    confirmed_flags[i : i + d] = 1
            flags = confirmed_flags

        return flags

    def get_state_name(self, state_idx: int) -> str:
        """Get state name from index."""
        return self.states[state_idx]

    def get_transition_matrix(self) -> np.ndarray:
        """Get transition matrix."""
        return self.T.copy()
