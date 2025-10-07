#!/usr/bin/env python3
"""Verify imports work as expected with the new pytest configuration."""
import sys
from pathlib import Path

# This mimics what conftest.py does
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Now test the imports
try:
    from src.models.lstm import AnomalyLSTM, create_sequences
    from src.models.markov_smoother import MarkovSmoother
    from src.data.features import compute_returns, compute_volatility
    from src.pipelines.train import main as train_main
    print("✅ All imports successful!")
    print("   - src.models.lstm")
    print("   - src.models.markov_smoother")
    print("   - src.data.features")
    print("   - src.pipelines.train")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
