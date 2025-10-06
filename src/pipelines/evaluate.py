"""Evaluation pipeline to compute metrics and generate plots."""
import logging
import os
from datetime import datetime

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.logging_config import setup_logging

logger = setup_logging(name=__name__)
sns.set_style("whitegrid")


def load_predictions_and_labels(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load predictions and labels from database.

    Args:
        conn: DuckDB connection

    Returns:
        DataFrame with predictions and labels
    """
    query = """
        SELECT
            p.*,
            l.label
        FROM predictions p
        JOIN labels l ON p.date = l.date AND p.symbol = l.symbol
        ORDER BY p.date, p.symbol
    """
    df = conn.execute(query).df()
    logger.info(f"Loaded {len(df)} records with predictions and labels")
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted scores/probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
    }

    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics["pr_auc"] = auc(recall, precision)

    return metrics


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_scores: Predicted scores
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray, y_scores: np.ndarray, save_path: str
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_scores: Predicted scores
        save_path: Path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR curve to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_time_series(df: pd.DataFrame, save_path: str) -> None:
    """
    Plot time series of anomaly scores.

    Args:
        df: DataFrame with predictions
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot LSTM scores
    axes[0].plot(df["date"], df["p_anom"], alpha=0.7, label="LSTM P(Anomaly)")
    axes[0].axhline(0.5, color="r", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    axes[0].set_ylabel("LSTM Score")
    axes[0].legend()
    axes[0].set_title("LSTM Anomaly Scores")

    # Plot Markov posteriors
    axes[1].plot(df["date"], df["post_anomalous"], alpha=0.7, label="Markov P(A)", color="orange")
    axes[1].axhline(0.7, color="r", linestyle="--", alpha=0.5, label="Threshold (0.7)")
    axes[1].set_ylabel("Markov P(A)")
    axes[1].legend()
    axes[1].set_title("Markov Smoothed Posteriors")

    # Plot true labels
    axes[2].fill_between(
        df["date"],
        0,
        df["label"],
        alpha=0.3,
        color="red",
        label="True Anomalies",
    )
    axes[2].set_ylabel("True Label")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].set_title("Ground Truth Anomalies")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved time series plot to {save_path}")


def evaluate_model(config: dict) -> None:
    """
    Main evaluation pipeline.

    Args:
        config: Configuration dictionary
    """
    # Connect to database
    db_path = config["sql"]["database"]
    conn = duckdb.connect(db_path)

    try:
        # Load data
        df = load_predictions_and_labels(conn)

        # Prepare data
        y_true = df["label"].values
        y_scores_lstm = df["p_anom"].values
        y_scores_markov = df["post_anomalous"].values

        # Binary predictions (threshold at 0.5 for LSTM, 0.7 for Markov)
        y_pred_lstm = (y_scores_lstm > 0.5).astype(int)
        y_pred_markov = (y_scores_markov > 0.7).astype(int)

        # Compute metrics
        logger.info("Computing LSTM metrics")
        lstm_metrics = compute_metrics(y_true, y_pred_lstm, y_scores_lstm)

        logger.info("Computing Markov metrics")
        markov_metrics = compute_metrics(y_true, y_pred_markov, y_scores_markov)

        # Print metrics
        logger.info("\n" + "=" * 50)
        logger.info("LSTM Metrics:")
        logger.info(f"  Precision: {lstm_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {lstm_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {lstm_metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {lstm_metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:    {lstm_metrics['pr_auc']:.4f}")
        logger.info("=" * 50)

        logger.info("\n" + "=" * 50)
        logger.info("Markov Smoothed Metrics:")
        logger.info(f"  Precision: {markov_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {markov_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {markov_metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {markov_metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:    {markov_metrics['pr_auc']:.4f}")
        logger.info("=" * 50 + "\n")

        # Create artifacts directory
        artifacts_dir = config["paths"]["artifacts_dir"]
        plots_dir = config["paths"]["plots_dir"]
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots
        logger.info("Generating plots")

        # ROC curves
        plot_roc_curve(y_true, y_scores_lstm, f"{plots_dir}/roc_curve_lstm.png")
        plot_roc_curve(y_true, y_scores_markov, f"{plots_dir}/roc_curve_markov.png")

        # PR curves
        plot_precision_recall_curve(
            y_true, y_scores_lstm, f"{plots_dir}/pr_curve_lstm.png"
        )
        plot_precision_recall_curve(
            y_true, y_scores_markov, f"{plots_dir}/pr_curve_markov.png"
        )

        # Confusion matrices
        plot_confusion_matrix(y_true, y_pred_lstm, f"{plots_dir}/confusion_matrix_lstm.png")
        plot_confusion_matrix(y_true, y_pred_markov, f"{plots_dir}/confusion_matrix_markov.png")

        # Time series (use a subset for clarity)
        df_subset = df.head(1000)  # First 1000 samples
        plot_time_series(df_subset, f"{plots_dir}/time_series.png")

        # Save metrics to JSON
        metrics_output = {
            "timestamp": datetime.now().isoformat(),
            "lstm": lstm_metrics,
            "markov": markov_metrics,
            "dataset_size": len(df),
            "anomaly_ratio": float(y_true.mean()),
        }

        metrics_file = f"{artifacts_dir}/metrics.json"
        save_json(metrics_output, metrics_file)
        logger.info(f"Saved metrics to {metrics_file}")

        # Print classification report
        logger.info("\nLSTM Classification Report:")
        print(classification_report(y_true, y_pred_lstm, target_names=["Normal", "Anomaly"]))

        logger.info("\nMarkov Classification Report:")
        print(classification_report(y_true, y_pred_markov, target_names=["Normal", "Anomaly"]))

        logger.info("✅ Evaluation completed successfully")

    finally:
        conn.close()


def main():
    """Main function."""
    config = load_config()
    evaluate_model(config)


if __name__ == "__main__":
    main()

