#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy an existing Colab/Drive checkpoint into the repository data directory."
    )
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--source", required=True, help="Existing .npz checkpoint.")
    parser.add_argument(
        "--destination-root",
        default=None,
        help="Repository destination root.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional destination filename. Defaults to the source filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)

    if str(source).startswith("/content/drive/") and not source.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            pass

    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    if args.destination_root is None:
        destination_dir = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / args.regime
    else:
        destination_dir = Path(args.destination_root) / args.regime
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (args.name or source.name)
    shutil.copy2(source, destination)
    print(f"Copied checkpoint to: {destination}")


if __name__ == "__main__":
    main()
