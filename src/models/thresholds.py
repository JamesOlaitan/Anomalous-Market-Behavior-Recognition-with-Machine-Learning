"""Threshold utilities for anomaly detection."""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)


def find_optimal_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal threshold for classification based on validation set.

    Args:
        y_true: True labels
        y_scores: Predicted scores/probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    if metric == "f1":
        best_idx = np.argmax(f1_scores)
        best_score = f1_scores[best_idx]
    elif metric == "precision":
        best_idx = np.argmax(precision[:-1])
        best_score = precision[best_idx]
    elif metric == "recall":
        best_idx = np.argmax(recall[:-1])
        best_score = recall[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_threshold = thresholds[best_idx]

    logger.info(
        f"Optimal threshold for {metric}: {optimal_threshold:.4f} " f"(score: {best_score:.4f})"
    )

    return optimal_threshold, best_score


def tune_markov_params(
    y_true: np.ndarray,
    p_anom: np.ndarray,
    posteriors: np.ndarray,
    tau_range: np.ndarray = np.linspace(0.5, 0.9, 9),
    k_range: range = range(1, 6),
) -> Dict[str, float]:
    """
    Tune Markov decision parameters (tau, k) on validation set.

    Args:
        y_true: True labels
        p_anom: LSTM probabilities
        posteriors: Markov posteriors
        tau_range: Range of tau values to try
        k_range: Range of k values to try

    Returns:
        Dictionary with optimal parameters
    """
    from src.models.markov_smoother import MarkovSmoother

    best_f1 = 0.0
    best_params = {"tau": 0.7, "k": 3, "d": 0}

    smoother = MarkovSmoother()

    for tau in tau_range:
        for k in k_range:
            # Apply decision rule
            _, _, flags = smoother.forward(p_anom, tau=tau, k=k, d=0)

            # Compute F1
            f1 = f1_score(y_true.flatten(), flags.flatten())

            if f1 > best_f1:
                best_f1 = f1
                best_params = {"tau": tau, "k": k, "d": 0}

    logger.info(f"Best Markov params: {best_params} (F1: {best_f1:.4f})")

    return best_params
