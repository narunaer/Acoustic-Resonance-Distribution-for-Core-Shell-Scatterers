#!/usr/bin/env python
import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reproducible local pipeline for one or all regimes."
    )
    parser.add_argument(
        "--regime",
        choices=("density", "mixed", "velocity", "all"),
        default="all",
    )
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--x-min", type=float, default=1e-4)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--mesh-points", type=int, default=3000)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip dataset generation and use checkpoints already stored in data/checkpoints/.",
    )
    parser.add_argument(
        "--variant",
        choices=("raw", "manual_corrected", "final"),
        default=None,
        help="Figure/analysis stage. Default is the article-final dataset.",
    )
    return parser.parse_args()


def run(command):
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    regimes = ("density", "mixed", "velocity") if args.regime == "all" else (args.regime,)

    if not args.figures_only and args.variant == "raw":
        raise SystemExit(
            "--variant raw is intended for figures-only comparisons. "
            "Generation always preserves raw and creates the final manual-corrected dataset automatically."
        )

    for regime in regimes:
        if not args.figures_only:
            run([
                sys.executable,
                "scripts/generate_dataset.py",
                "--regime", regime,
                "--lmax", str(args.lmax),
                "--x-min", str(args.x_min),
                "--x-max", str(args.x_max),
                "--mesh-points", str(args.mesh_points),
            ])

        figure_command = [
            sys.executable,
            "scripts/make_all_figures.py",
            "--regime", regime,
        ]
        if args.variant is not None:
            figure_command.extend(["--variant", args.variant])
        run(figure_command)

    run([sys.executable, "scripts/check_repository.py"])


if __name__ == "__main__":
    main()
