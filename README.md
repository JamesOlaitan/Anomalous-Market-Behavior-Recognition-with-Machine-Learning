# Anomalous Market Behavior Recognition with Machine Learning

[![CI](https://github.com/jamesolaitan/Anomalous-Market-Behavior-Recognition-with-Machine-Learning/workflows/CI/badge.svg)](https://github.com/jamesolaitan/Anomalous-Market-Behavior-Recognition-with-Machine-Learning/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect correlation breakdown anomalies in financial markets using LSTM neural networks with Markov temporal smoothing**

A production-ready machine learning pipeline for detecting anomalous market behavior through correlation breakdown analysis. This project combines PyTorch LSTM models with a simple Markov (HMM-lite) temporal smoother to identify periods when asset correlations deviate significantly from normal patterns.

## 🎯 Key Features

- **LSTM Anomaly Detection**: PyTorch-based LSTM model trained on ~14 years of financial data
- **VIX Integration**: Incorporates VIX (volatility index) as a systemic risk feature
- **Markov Temporal Smoother**: Simple 2-3 state HMM-lite reduces false positives via temporal smoothing
- **SQL Data Pipeline**: DuckDB-based pipeline for efficient data processing and storage
- **Interactive Dashboard**: Streamlit dashboard for real-time exploration and visualization
- **Production Ready**: Complete testing, CI/CD, Docker support, and comprehensive documentation

## 📊 Current Performance Metrics

| Model | Precision | Recall | **F1 Score** | ROC-AUC | PR-AUC |
|-------|-----------|--------|--------------|---------|--------|
| LSTM  | 0.750       | 0.600    | **0.667**      | 1.000     | 0.787    |
| Markov| 0.000       | 0.000    | **0.000**      | 1.000     | 0.787    |

*Note: Metrics will be computed after running `make all`. See [Running the Pipeline](#-running-the-pipeline) below.*

**Last Updated**: 2025-10-07 10:13 UTC *(commit: `cf28c22`)*
**Dataset**: 2010-2024 (SPY, XLF, XLK, VNQ + VIX)

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Data      │────▶│   Features   │────▶│   Labels    │
│ (yfinance)  │     │ (rolling corr│     │(breakdown)  │
│ SPY,XLF,    │     │  vol, VIX)   │     │   rules     │
│ XLK,VNQ,VIX │     └──────────────┘     └─────────────┘
└─────────────┘              │                    │
                             ▼                    ▼
                      ┌──────────────────────────────┐
                      │       DuckDB Database        │
                      │  (raw_prices, features,      │
                      │   labels, predictions)       │
                      └──────────────────────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
              ┌──────────────┐          ┌──────────────┐
              │ LSTM Model   │          │   Markov     │
              │  (PyTorch)   │────────▶ │  Smoother    │
              │              │ p_anom   │  (HMM-lite)  │
              └──────────────┘          └──────────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                      ┌──────────────────────────┐
                      │    Evaluation Pipeline   │
                      │  (Metrics, Plots, JSON)  │
                      └──────────────────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────┐
                      │  Streamlit Dashboard     │
                      │  (Interactive Viz)       │
                      └──────────────────────────┘
```

## 📁 Project Structure

```
repo/
├── src/
│   ├── data/
│   │   ├── download.py         # Fetch OHLCV + VIX from Yahoo Finance
│   │   ├── ingest_sql.py       # Load data into DuckDB
│   │   └── features.py         # Feature engineering (corr, vol, VIX)
│   ├── models/
│   │   ├── lstm.py             # PyTorch LSTM model
│   │   ├── markov_smoother.py  # HMM-lite temporal smoother
│   │   └── thresholds.py       # Threshold tuning utilities
│   ├── pipelines/
│   │   ├── train.py            # Training orchestration
│   │   ├── predict.py          # Prediction generation
│   │   └── evaluate.py         # Metrics & plots
│   ├── viz/
│   │   └── dashboard.py        # Streamlit dashboard
│   └── utils/
│       ├── config.py           # Configuration management
│       ├── seed.py             # Random seed utilities
│       ├── logging_config.py   # Logging setup
│       └── io.py               # I/O utilities
├── sql/
│   └── schema.sql              # Database schema
├── tests/
│   ├── test_features.py        # Feature engineering tests
│   ├── test_lstm.py            # LSTM model tests
│   ├── test_markov.py          # Markov smoother tests
│   └── test_end_to_end.py      # Integration tests
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── Makefile                    # Build automation
├── Dockerfile                  # Container definition
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/jamesolaitan/Anomalous-Market-Behavior-Recognition-with-Machine-Learning.git
cd Anomalous-Market-Behavior-Recognition-with-Machine-Learning

# Install dependencies
make setup

# Or manually:
pip install -r requirements.txt
pre-commit install
```

### 🔧 Running the Pipeline

Run the complete pipeline with a single command:

```bash
make all
```

This executes:
1. **Data download** (`make data`): Fetches ~14 years of data from Yahoo Finance
2. **Feature engineering** (`make features`): Computes rolling correlations, volatility, z-scores, VIX features
3. **Training** (`make train`): Trains LSTM model with early stopping (~100 epochs)
4. **Prediction** (`make predict`): Generates predictions and applies Markov smoothing
5. **Evaluation** (`make eval`): Computes metrics (F1, ROC-AUC, PR-AUC) and generates plots

Or run steps individually:

```bash
make data         # Download and ingest data
make features     # Engineer features
make train        # Train LSTM model
make predict      # Generate predictions
make eval         # Evaluate and compute metrics
make dashboard    # Launch Streamlit dashboard
```

### 📊 View Results

After running the pipeline, launch the interactive dashboard:

```bash
make dashboard
# or
streamlit run src/viz/dashboard.py
```

Open your browser to `http://localhost:8501` to explore:
- Time series with anomaly highlights
- LSTM anomaly probabilities
- Markov smoothed posteriors
- Feature correlations and VIX
- Performance metrics

### 🧪 Testing

```bash
make test         # Run all tests with coverage
make lint         # Run linters (flake8, mypy)
make format       # Format code (black, isort)
```

## 📖 How It Works

### 1. Data & Features

**Symbols**: SPY (S&P 500), XLF (Financials), XLK (Technology), VNQ (Real Estate)  
**Period**: 2010-01-01 to 2024-12-31  
**Features**:
- Daily returns and log returns
- Rolling volatility (20-day window)
- Rolling correlation with SPY (60-day window)
- Z-scores of correlation and volatility
- VIX level and delta

### 2. Labeling (Correlation Breakdown)

Anomalies are labeled when:
1. Rolling correlation drops below threshold (< 0.1)
2. **AND** correlation change is steep (Δcorr < -0.3 within 10 days)

Labels persist for 5 days to mark "breakdown episodes."

### 3. LSTM Model

**Architecture**:
- Input: 8 features
- LSTM: 1 layer, 64 hidden units
- Output: sigmoid(logit) → P(anomaly)

**Training**:
- Loss: BCEWithLogitsLoss with pos_weight=5.0 (class imbalance)
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=10 epochs on validation PR-AUC

### 4. Markov Smoother (HMM-lite)

**States**: Normal (N), Anomalous (A)

**Transition Matrix** (default):
```
        N      A
N    0.97   0.03
A    0.15   0.85
```

**How it works**:
1. Takes LSTM probabilities `p_anom_t` as observations
2. Performs forward update: `prior_t = posterior_{t-1} @ T`
3. Computes posterior: `post_t ∝ prior_t * P(obs_t | state)`
4. Flags anomaly if `P(A) > 0.7` for 3+ consecutive steps

**Benefits**: Reduces false positives by enforcing temporal persistence.

### 5. Evaluation

**Metrics** (both point-level and event-level):
- Precision, Recall, F1 Score
- ROC-AUC
- PR-AUC (Precision-Recall Area Under Curve)

**Outputs**:
- `artifacts/metrics.json`: Computed metrics
- `artifacts/plots/`: ROC curves, PR curves, confusion matrices, time series

## ⚙️ Configuration

All parameters are in `config.yaml`:

```yaml
data:
  symbols: [SPY, XLF, XLK, VNQ]
  start_date: "2010-01-01"
  end_date: "2024-12-31"

features:
  rolling_window: 60  # correlation window

labels:
  corr_threshold: 0.1
  delta_threshold: -0.3

model:
  hidden_size: 64
  learning_rate: 0.001
  epochs: 100

markov:
  num_states: 2
  decision_threshold: 0.7
  consecutive_steps: 3
```

## 🐳 Docker

Run the entire pipeline in a reproducible Docker container:

```bash
# Build image
docker build -t anomaly-detection .

# Run pipeline
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/artifacts:/app/artifacts \
           anomaly-detection \
           make all

# Launch dashboard
docker run -p 8501:8501 \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/artifacts:/app/artifacts \
           anomaly-detection \
           streamlit run src/viz/dashboard.py
```

## 🧩 Extending the Project

### Add New Symbols

Edit `config.yaml`:
```yaml
data:
  symbols: [SPY, XLF, XLK, VNQ, QQQ, IWM]
```

### Use 3-State Markov Model

Edit `config.yaml`:
```yaml
markov:
  num_states: 3  # N, A, R (Recovery)
```

### Tune Hyperparameters

Modify `config.yaml` or override via CLI (future enhancement).

## 📈 Results Interpretation

### What is a "Correlation Breakdown"?

Financial assets often move together (correlation). A **correlation breakdown** occurs when this relationship suddenly weakens, often signaling:
- Market stress or regime change
- Sector rotation
- Flight to safety (VIX spike)
- Structural market shifts

### When to Use This Model?

- **Risk monitoring**: Flag periods of unusual market behavior
- **Portfolio rebalancing**: Adjust allocations when correlations break
- **Event detection**: Identify crisis periods (COVID-19, 2008 crash, etc.)

## 🛠️ Development

### Adding a New Feature

1. Add computation in `src/data/features.py`
2. Update SQL schema in `sql/schema.sql`
3. Retrain model: `make train`
4. Add tests in `tests/test_features.py`

### Adding a New Model

1. Create module in `src/models/`
2. Update training pipeline in `src/pipelines/train.py`
3. Add tests in `tests/`

## 📋 Project Checklist

- ✅ Audit repo: list files, note issues, TODOs
- ✅ Set up tooling: .gitignore, requirements.txt, pre-commit, Makefile, CI
- ✅ Restructure project with proper src/ layout
- ✅ Create utility modules (config, logging, seed, io)
- ✅ Data ingestion (prices + VIX) → SQL with DuckDB
- ✅ Feature engineering (rolling corr, vol, VIX merges, z-scores)
- ✅ Labeling (correlation breakdown rules)
- ✅ LSTM model (PyTorch) - train & save
- ✅ Markov smoother (T estimation + forward update + decision)
- ✅ Prediction pipeline → SQL predictions
- ✅ Evaluation (metrics: P/R/F1, ROC-AUC, PR-AUC; write F1)
- ✅ Dashboard (Streamlit) with visualizations
- ✅ SQL schema and feature views
- ✅ Tests (unit + e2e) and fix failures
- ✅ Docker setup for reproducibility
- ✅ Docs (README with usage, diagrams)
- ✅ GitHub Actions CI

## 📝 Changelog

### Version 2.0.0 (2024)

**Major Upgrade - Complete Rewrite**

- ✨ **LSTM Model**: Replaced autoencoder with PyTorch LSTM
- ✨ **Markov Smoother**: Added HMM-lite temporal smoothing (2-state)
- ✨ **DuckDB Pipeline**: Complete SQL pipeline for data management
- ✨ **Streamlit Dashboard**: Interactive visualization dashboard
- ✨ **Comprehensive Testing**: Unit tests + e2e tests + CI/CD
- ✨ **Production Ready**: Docker, pre-commit hooks, linting, type hints
- 🔧 **Improved Features**: Added VIX delta, z-scores, rolling windows
- 🔧 **Better Labeling**: Correlation breakdown with persistence
- 🔧 **Reproducibility**: Fixed seeds, config-driven, documented
- 📚 **Documentation**: Complete README with architecture diagrams

### Version 1.0.0 (Original)

- Basic autoencoder for anomaly detection
- Simple data loading with yfinance
- TensorFlow/Keras implementation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Before submitting**:
- Run tests: `make test`
- Format code: `make format`
- Check linting: `make lint`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: James Olaitan  
**GitHub**: [@jamesolaitan](https://github.com/jamesolaitan)

## 🙏 Acknowledgments

- Data source: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- Frameworks: PyTorch, DuckDB, Streamlit, scikit-learn
- Inspiration: Financial time series analysis and anomaly detection research

---

**⭐ If you find this project useful, please consider giving it a star!**
