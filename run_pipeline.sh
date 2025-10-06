#!/bin/bash

# Anomalous Market Behavior Recognition - Pipeline Runner
# This script runs the complete ML pipeline with error handling and progress tracking

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Anomalous Market Behavior Recognition Pipeline              ║"
echo "║   PyTorch + DuckDB + Streamlit                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Consider using one."
    echo "   To create: python3 -m venv venv && source venv/bin/activate"
    echo ""
fi

# Step 1: Data Download
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Step 1/5: Downloading financial data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.data.download; then
    echo "✓ Data download completed"
else
    echo "✗ Data download failed"
    exit 1
fi

# Step 2: SQL Ingestion
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗄️  Step 2/5: Ingesting data into DuckDB..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.data.ingest_sql; then
    echo "✓ SQL ingestion completed"
else
    echo "✗ SQL ingestion failed"
    exit 1
fi

# Step 3: Feature Engineering
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Step 3/5: Engineering features..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.data.features; then
    echo "✓ Feature engineering completed"
else
    echo "✗ Feature engineering failed"
    exit 1
fi

# Step 4: Model Training
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 Step 4/5: Training LSTM model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.pipelines.train; then
    echo "✓ Model training completed"
else
    echo "✗ Model training failed"
    exit 1
fi

# Step 5: Prediction
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔮 Step 5/5: Generating predictions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.pipelines.predict; then
    echo "✓ Prediction completed"
else
    echo "✗ Prediction failed"
    exit 1
fi

# Step 6: Evaluation
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 6/6: Evaluating model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if python3 -m src.pipelines.evaluate; then
    echo "✓ Evaluation completed"
else
    echo "✗ Evaluation failed"
    exit 1
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Completed! ✨                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results:"
echo "   - Metrics: artifacts/metrics.json"
echo "   - Plots:   artifacts/plots/"
echo "   - Models:  models/"
echo ""
echo "🚀 Next steps:"
echo "   1. View metrics: cat artifacts/metrics.json | python -m json.tool"
echo "   2. Launch dashboard: streamlit run src/viz/dashboard.py"
echo "   3. Run tests: pytest tests/ -v"
echo ""

