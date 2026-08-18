#!/usr/bin/env python
import argparse
from core_shell.checkpoints import export_checkpoint_csv, repository_generation_paths
from core_shell.manual_corrections import apply_manual_corrections
from core_shell.regimes import get_regime


def main():
    parser = argparse.ArgumentParser(
        description="Apply the exact hand-curated correction list to an existing raw checkpoint."
    )
    parser.add_argument("--regime", choices=("mixed", "velocity"), required=True)
    parser.add_argument("--lmax", type=int, default=10)
    args = parser.parse_args()

    regime = get_regime(args.regime)
    raw_paths = repository_generation_paths(regime, args.lmax, variant="raw")
    raw_checkpoint = raw_paths["checkpoint"]

    corrected_curves, report = apply_manual_corrections(
        raw_checkpoint, args.regime, lmax=args.lmax
    )

    corrected_paths = repository_generation_paths(
        regime, args.lmax, variant="manual_corrected"
    )
    corrected_paths["checkpoint"] = report["corrected_checkpoint"]
    export_checkpoint_csv(corrected_paths, corrected_curves, args.lmax)

    print(f"Manual corrections applied: {args.regime}")
    print(f"Raw checkpoint preserved: {raw_checkpoint}")
    print(f"Final corrected checkpoint: {report['corrected_checkpoint']}")
    print(f"Correction record: {report['corrections_json']}")
    print(
        "Applied records: "
        f"D0={report['removed_d0_markers']}, "
        f"individual peaks={report['removed_peak_entries']}, "
        f"clear-all curves={report['cleared_curves']}, "
        f"insertions={report['inserted_peak_records']}"
    )


if __name__ == "__main__":
    main()
