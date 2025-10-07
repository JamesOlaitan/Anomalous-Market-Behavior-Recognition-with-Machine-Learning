"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add project root to sys.path to ensure src/ is importable
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

