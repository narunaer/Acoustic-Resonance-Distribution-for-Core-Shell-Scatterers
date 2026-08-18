#!/usr/bin/env python
"""Generate all standard figures/analyses for one dataset variant."""
import argparse
import subprocess
import sys

from core_shell.checkpoints import repository_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--variant",
        choices=("raw", "manual_corrected", "final"),
        default=None,
        help=(
            "Dataset stage. Default: density=final; mixed/velocity=manual_corrected. "
            "Use --variant raw to reproduce the automatic detector output separately."
        ),
    )
    args = parser.parse_args()

    variant = repository_variant(args.regime, args.variant)
    common = ["--regime", args.regime, "--variant", variant]
    if args.checkpoint is not None:
        common.extend(["--checkpoint", args.checkpoint])

    for script in (
        "scripts/plot_selected_cases.py",
        "scripts/plot_delta_eta_map.py",
        "scripts/plot_alpha_beta_map.py",
        "scripts/fit_existence_classifier.py",
    ):
        subprocess.run([sys.executable, script, *common], check=True)

    if args.regime in ("velocity", "mixed"):
        recurrence_args = ["--regime", args.regime, "--variant", variant]
        if args.checkpoint is not None:
            recurrence_args.extend(["--checkpoint", args.checkpoint])
        subprocess.run(
            [sys.executable, "scripts/fit_recurrence.py", *recurrence_args],
            check=True,
        )

    print(f"Completed figures/analyses for {args.regime} [{variant}].")


if __name__ == "__main__":
    main()
