"""Root conftest.py - ensures src package is importable during test collection."""

import sys
from pathlib import Path

# Add repository root to sys.path so 'src' can be imported
# This runs BEFORE pytest collects test modules
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
