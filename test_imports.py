#!/usr/bin/env python3
"""Quick test to verify imports work."""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

try:
    from src.models.lstm import AnomalyLSTM
    from src.models.markov_smoother import MarkovSmoother
    from src.data.features import compute_returns
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
