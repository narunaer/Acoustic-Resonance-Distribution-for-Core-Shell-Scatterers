#!/usr/bin/env python
import argparse
import os
import time

import numpy as np

from core_shell.checkpoints import (
    DELTA_VALUES,
    build_eta_grid,
    checkpoint_paths,
    default_output_root,
    repository_generation_paths,
    export_checkpoint_csv,
    load_resume_state,
    save_checkpoint,
)
from core_shell.detection import detect_resonances
from core_shell.physics import eta_from_alpha_beta
from core_shell.manual_corrections import apply_manual_corrections, corrected_checkpoint_path
from core_shell.regimes import get_regime


def parse_args():
    parser = argparse.ArgumentParser(description="Generate density or velocity core-shell resonance data.")
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output root. By default, data are written inside the repository.",
    )
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--x-min", type=float, default=1e-4)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--mesh-points", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    regime = get_regime(args.regime)
    output_root = args.output_root

    alpha_new, beta_new = build_eta_grid(seed=args.seed)
    n_pairs = len(alpha_new)
    total = n_pairs * len(DELTA_VALUES)
    paths = (
        repository_generation_paths(regime, args.lmax)
        if output_root is None
        else checkpoint_paths(output_root, regime, args.lmax)
    )

    curves, done, alpha_saved, beta_saved = load_resume_state(
        paths["checkpoint"], total, args.lmax
    )
    if len(done) != total:
        curves, done, alpha_saved, beta_saved = {}, np.zeros(total, dtype=bool), None, None

    if alpha_saved is not None and len(alpha_saved) == n_pairs:
        alpha_grid, beta_grid = alpha_saved, beta_saved
    else:
        alpha_grid, beta_grid = alpha_new, beta_new

    start = time.time()
    new_curves = 0

    for delta_index, delta in enumerate(DELTA_VALUES):
        for pair_index in range(n_pairs):
            flat_index = delta_index * n_pairs + pair_index
            if done[flat_index]:
                continue

            alpha = float(alpha_grid[pair_index])
            beta = float(beta_grid[pair_index])
            peaks = detect_resonances(
                delta,
                alpha,
                beta,
                regime,
                mix_factor=regime.mix_factor,
                x_min=args.x_min,
                x_max=args.x_max,
                n_global_mesh=args.mesh_points,
                lmax=args.lmax,
            )

            key = f"{delta_index}_{pair_index}"
            curves[key] = {
                "delta": float(delta),
                "alpha": alpha,
                "beta": beta,
                "eta": float(eta_from_alpha_beta(alpha, beta)),
                "picos": peaks,
            }
            done[flat_index] = True
            new_curves += 1

            if new_curves % args.checkpoint_every == 0:
                save_checkpoint(
                    paths["checkpoint"],
                    curves,
                    done,
                    alpha_grid,
                    beta_grid,
                    args.lmax,
                )
                print(
                    f"[{int(np.sum(done))}/{total}] "
                    f"regime={args.regime} delta={delta:.2f} "
                    f"eta={curves[key]['eta']:+.3f} peaks={len(peaks)} "
                    f"time={time.time() - start:.1f}s"
                )

    save_checkpoint(
        paths["checkpoint"],
        curves,
        done,
        alpha_grid,
        beta_grid,
        args.lmax,
    )
    export_checkpoint_csv(paths, curves, args.lmax)
    print(f"Completed {int(np.sum(done))}/{total} curves.")
    print(f"Raw checkpoint: {paths['checkpoint']}")

    if args.regime in ("mixed", "velocity"):
        corrected_curves, report = apply_manual_corrections(
            paths["checkpoint"],
            args.regime,
            lmax=args.lmax,
        )
        corrected_paths = repository_generation_paths(
            regime, args.lmax, variant="manual_corrected"
        )
        corrected_paths["checkpoint"] = report["corrected_checkpoint"]
        export_checkpoint_csv(corrected_paths, corrected_curves, args.lmax)
        print(f"Manual corrections applied: {args.regime}")
        print(f"Final corrected checkpoint: {report['corrected_checkpoint']}")
        print(f"Manual correction record: {report['corrections_json']}")


if __name__ == "__main__":
    main()
