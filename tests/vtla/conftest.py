"""Run component checks independently of src/openpi/conftest.py's JAX fixtures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
