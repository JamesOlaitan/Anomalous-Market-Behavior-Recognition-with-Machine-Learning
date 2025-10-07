#!/usr/bin/env python3
"""Update README.md metrics table from artifacts/metrics.json."""

import json
import re
from datetime import datetime
from pathlib import Path


def load_metrics(metrics_path: str = "artifacts/metrics.json") -> dict:
    """Load metrics from JSON file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def format_metric(value: float) -> str:
    """Format metric as percentage or decimal."""
    return f"{value:.3f}"


def update_readme(metrics: dict, readme_path: str = "README.md") -> None:
    """Update README.md with computed metrics."""
    with open(readme_path, "r") as f:
        content = f.read()

    lstm = metrics["lstm"]
    markov = metrics["markov"]
    timestamp = metrics.get("timestamp", datetime.now().isoformat())

    # Update the table
    table_pattern = r"(\|\s*Model\s*\|.*?\n\|.*?\n)(\|\s*LSTM\s*\|.*?\n\|\s*Markov\s*\|.*?\n)"

    new_table_rows = (
        f"| LSTM  | {format_metric(lstm['precision'])}       | "
        f"{format_metric(lstm['recall'])}    | **{format_metric(lstm['f1'])}**      | "
        f"{format_metric(lstm['roc_auc'])}     | {format_metric(lstm['pr_auc'])}    |\n"
        f"| Markov| {format_metric(markov['precision'])}       | "
        f"{format_metric(markov['recall'])}    | **{format_metric(markov['f1'])}**      | "
        f"{format_metric(markov['roc_auc'])}     | {format_metric(markov['pr_auc'])}    |"
    )

    content = re.sub(table_pattern, r"\1" + new_table_rows + "\n", content)

    # Update timestamp
    date_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M UTC")
    content = re.sub(
        r"\*\*Last Updated\*\*:.*",
        f"**Last Updated**: {date_str} *(commit: `{get_commit_hash()}`)*",
        content,
    )

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"✅ README.md updated with metrics from {timestamp}")
    print("\nMetrics Summary:")
    print(f"  LSTM   - F1: {lstm['f1']:.3f}, ROC-AUC: {lstm['roc_auc']:.3f}")
    print(f"  Markov - F1: {markov['f1']:.3f}, ROC-AUC: {markov['roc_auc']:.3f}")


def get_commit_hash() -> str:
    """Get short commit hash."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "local"


def main():
    """Main function."""
    metrics_file = Path("artifacts/metrics.json")

    if not metrics_file.exists():
        print("❌ artifacts/metrics.json not found!")
        print("\nPlease run the pipeline first:")
        print("  make all")
        return 1

    try:
        metrics = load_metrics(str(metrics_file))
        update_readme(metrics)
        print("\n💡 Tip: Don't forget to commit the updated README.md!")
        return 0
    except Exception as e:
        print(f"❌ Error updating README: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
