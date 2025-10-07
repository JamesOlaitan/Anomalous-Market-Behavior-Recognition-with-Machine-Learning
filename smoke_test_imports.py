#!/usr/bin/env python3
"""Smoke test: verify src.models imports work."""
import sys
from pathlib import Path

# Add repo root to path (mimics conftest.py)
repo = Path(__file__).parent.resolve()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

# Try imports
try:
    from src.models.lstm import AnomalyLSTM
    from src.models.markov_smoother import MarkovSmoother
    from src.data.features import compute_returns
    print("✅ All imports successful!")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
