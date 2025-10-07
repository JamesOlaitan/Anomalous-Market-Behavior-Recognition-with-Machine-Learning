# 🚀 Quick Start: Running the Pipeline & Getting Metrics

This guide shows you how to run the full pipeline and automatically fill the metrics table in README.md.

## Prerequisites

- Python 3.9+ installed
- ~500MB disk space for data and models
- 5-15 minutes for initial run

## Step 1: Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make setup
# or: pip install -r requirements.txt && pre-commit install
```

## Step 2: Run Full Pipeline

```bash
# Run everything: download data → train model → evaluate → update README
make all
```

This will:
1. **Download data** (SPY, XLF, XLK, VNQ, VIX from 2010-2024)
2. **Engineer features** (rolling correlation, volatility, VIX)
3. **Train LSTM model** (~3-5 min on CPU)
4. **Fit Markov smoother**
5. **Generate predictions** on test set
6. **Compute metrics** (Precision, Recall, F1, ROC-AUC, PR-AUC)
7. **Update README.md** table automatically ✨

## Step 3: View Results

### Check Metrics File
```bash
cat artifacts/metrics.json
```

Example output:
```json
{
  "timestamp": "2024-10-07T15:23:45.123456",
  "lstm": {
    "precision": 0.7234,
    "recall": 0.6891,
    "f1": 0.7058,
    "roc_auc": 0.8421,
    "pr_auc": 0.7856
  },
  "markov": {
    "precision": 0.7654,
    "recall": 0.7123,
    "f1": 0.7378,
    "roc_auc": 0.8567,
    "pr_auc": 0.8102
  }
}
```

### View Plots
```bash
open artifacts/plots/
# Or: ls -la artifacts/plots/
```

Generated plots:
- `roc_curve_lstm.png` / `roc_curve_markov.png`
- `pr_curve_lstm.png` / `pr_curve_markov.png`
- `confusion_matrix_lstm.png` / `confusion_matrix_markov.png`
- `time_series.png` (predictions over time)

### Launch Interactive Dashboard
```bash
make dashboard
# Opens at http://localhost:8501
```

## Step 4: Commit Updated Metrics

```bash
# Check what changed
git diff README.md

# Commit the updated metrics
git add README.md artifacts/metrics.json artifacts/plots/
git commit -m "docs: update metrics after running full pipeline

Results from $(date):
- LSTM F1: <your_value>
- Markov F1: <your_value>"
```

## 🔧 Individual Pipeline Steps

If you want to run steps individually:

```bash
# 1. Download and ingest data
make data

# 2. Engineer features
make features

# 3. Train model
make train

# 4. Generate predictions
make predict

# 5. Evaluate and compute metrics
make eval

# 6. Update README with metrics
make update-readme

# 7. Launch dashboard
make dashboard
```

## 🔄 Re-running with Different Parameters

Edit `config.yaml` to customize:
- Symbols (default: SPY, XLF, XLK, VNQ)
- Date range (default: 2010-2024)
- LSTM architecture (hidden size, layers, etc.)
- Training hyperparameters (epochs, learning rate, etc.)
- Markov smoother parameters (states, transition matrix)

Then re-run:
```bash
make clean  # Remove old artifacts
make all    # Run full pipeline with new config
```

## ⚡ Quick Commands Cheat Sheet

```bash
make help           # Show all available commands
make test           # Run test suite
make lint           # Check code quality
make format         # Auto-format code
make clean          # Remove generated files
make all            # Run full pipeline + update README
make dashboard      # Launch Streamlit dashboard
```

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Ensure package is installed in editable mode
pip install -e .
```

### "Database file not found"
```bash
# Re-run data ingestion
make data
```

### Metrics not updating in README
```bash
# Manually run the update script
python update_metrics.py

# Or check if artifacts/metrics.json exists
ls -la artifacts/metrics.json
```

### CUDA/GPU issues
The code automatically falls back to CPU if CUDA is unavailable. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
make all
```

## 📊 Understanding the Metrics

- **Precision**: Of predicted anomalies, what % were actually anomalies?
- **Recall**: Of actual anomalies, what % did we detect?
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC**: Area under the ROC curve (true positive vs false positive rate)
- **PR-AUC**: Area under the Precision-Recall curve (better for imbalanced data)

**Markov vs LSTM**: The Markov smoother typically has higher precision (fewer false alarms) by enforcing temporal consistency.

## 🎯 Next Steps

1. ✅ Run `make all` to fill the metrics table
2. 📊 Explore the dashboard: `make dashboard`
3. 📈 Adjust parameters in `config.yaml` and re-run
4. 🔍 Analyze results in `artifacts/` directory
5. 📝 Document your findings in `CHANGELOG.md`

---

**Need help?** Check the main [README.md](README.md) or open an issue on GitHub.
