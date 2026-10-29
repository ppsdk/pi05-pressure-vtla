"""Paths for integration tests without installing the main project."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "packages/openpi-client/src"))
