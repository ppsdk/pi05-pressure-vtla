"""Fit independent resistive-array signal stats using only NumPy.

Pass the TRAIN split manifest explicitly. This does not calibrate ADC to physical
pressure, fit state/action statistics, or read validation/test manifests.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# Make direct ``python scripts/compute_pressure_stats.py`` work before installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.training.pressure_dataset import fit_pressure_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Explicit TRAIN split manifest JSON")
    parser.add_argument("--output", required=True, type=Path, help="Destination pressure stats NPZ")
    parser.add_argument("--min-scale", type=float, default=1e-6)
    args = parser.parse_args()
    stats = fit_pressure_stats(args.manifest, min_scale=args.min_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A file handle prevents NumPy silently appending '.npz' to the supplied path.
    with args.output.open("wb") as stream:
        np.savez(
            stream,
            **stats,
            signal_unit=np.asarray("signal"),
            training_manifest=np.asarray(str(args.manifest.resolve())),
        )
    missing = int(np.count_nonzero(stats["count"] == 0))
    print(f"Saved signal normalization statistics to {args.output.resolve()} ({missing} taxels without valid samples).")


if __name__ == "__main__":
    main()
