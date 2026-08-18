#!/usr/bin/env python
from pathlib import Path

MIB = 1024**2
GITHUB_REGULAR_FILE_LIMIT_MIB = 100
ROOT = Path(__file__).resolve().parents[1]


def describe_checkpoint(path):
    size_mib = path.stat().st_size / MIB
    suffix = (
        "  -> use Git LFS or external data hosting"
        if size_mib > GITHUB_REGULAR_FILE_LIMIT_MIB
        else ""
    )
    return f"{path.relative_to(ROOT)} ({size_mib:.1f} MiB){suffix}"


def files_with_suffix(directory, suffixes):
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def main():
    print(f"Repository: {ROOT}\n")
    ok = True

    for regime in ("density", "mixed", "velocity"):
        print(f"[{regime}]")
        if regime == "density":
            stages = ("final",)
            required_stage = "final"
        else:
            stages = ("raw", "manual_corrected")
            required_stage = "manual_corrected"

        for stage in stages:
            checkpoint_dir = ROOT / "data" / "checkpoints" / regime / stage
            checkpoints = sorted(checkpoint_dir.glob("*.npz")) if checkpoint_dir.exists() else []
            required = stage == required_stage or stage == "raw"
            if not checkpoints:
                print(f"  {stage} checkpoint: MISSING")
                if required:
                    ok = False
            else:
                for checkpoint in checkpoints:
                    print(f"  {stage} checkpoint: {describe_checkpoint(checkpoint)}")

            figure_dir = ROOT / "figures" / regime / stage
            figures = files_with_suffix(figure_dir, {".png", ".pdf", ".svg", ".eps"})
            print(f"  {stage} figures: {len(figures)}")
            if stage == required_stage and not figures:
                print(f"  {stage} figures status: NOT GENERATED")
                ok = False

            results_dir = ROOT / "results" / regime / stage
            existence = sorted(results_dir.glob("*_existence_classifier_formula.txt")) if results_dir.exists() else []
            recurrence = sorted(results_dir.glob("*_anchored_recurrence_formula.txt")) if results_dir.exists() else []
            if stage == required_stage:
                print(f"  {stage} existence formula: {'OK' if existence else 'MISSING'}")
                if not existence:
                    ok = False
                if regime in ("mixed", "velocity"):
                    print(f"  {stage} recurrence formula: {'OK' if recurrence else 'MISSING'}")
                    if not recurrence:
                        ok = False

        legacy = ROOT / "figures" / regime / "legacy_unclassified"
        legacy_files = files_with_suffix(legacy, {".png", ".pdf", ".svg", ".eps"})
        if legacy_files:
            print(f"  legacy_unclassified figures preserved: {len(legacy_files)}")
        print()

    if ok:
        print("Repository contains the staged checkpoints, article-final figures, and final formula files.")
    else:
        print("Repository is not yet publication-complete. See missing items above.")


if __name__ == "__main__":
    main()
