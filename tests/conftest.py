"""Pytest configuration."""

import sys
from pathlib import Path

repo = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo))
