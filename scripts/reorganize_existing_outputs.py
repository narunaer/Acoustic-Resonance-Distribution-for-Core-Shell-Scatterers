#!/usr/bin/env python
"""One-time safe migration from the old flat output layout to staged folders.

This script never deletes a destination file and never recomputes the 6800 curves.
It only moves already-generated files into raw/final/manual_corrected folders.
For mixed/velocity figures generated before staging was introduced, their provenance
cannot be inferred safely, so they are preserved under legacy_unclassified/.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def move_safe(source: Path, destination: Path):
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"SKIP (destination exists): {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
        return False
    shutil.move(str(source), str(destination))
    print(f"MOVE: {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
    return True


def move_direct_children(source_dir: Path, destination_dir: Path, excluded_names=()):
    if not source_dir.exists():
        return
    excluded = set(excluded_names)
    for child in list(source_dir.iterdir()):
        if child.name in excluded or child.name == ".gitkeep":
            continue
        move_safe(child, destination_dir / child.name)


def migrate_checkpoints(regime):
    base = ROOT / "data" / "checkpoints" / regime
    base.mkdir(parents=True, exist_ok=True)
    variants = ["final"] if regime == "density" else ["raw", "manual_corrected"]
    for variant in variants:
        (base / variant).mkdir(parents=True, exist_ok=True)

    for path in list(base.iterdir()):
        if path.is_dir() or path.name == ".gitkeep":
            continue
        low = path.name.lower()
        if regime == "density":
            destination = base / "final" / path.name
        elif "corrigido_manual" in low or "correcoes_manuais" in low:
            destination = base / "manual_corrected" / path.name
        else:
            destination = base / "raw" / path.name
        move_safe(path, destination)


def migrate_exports(regime):
    base = ROOT / "data" / "exports" / regime
    base.mkdir(parents=True, exist_ok=True)
    variants = ["final"] if regime == "density" else ["raw", "manual_corrected"]
    for variant in variants:
        (base / variant).mkdir(parents=True, exist_ok=True)

    for path in list(base.iterdir()):
        if path.is_dir() or path.name == ".gitkeep":
            continue
        low = path.name.lower()
        if regime == "density":
            destination = base / "final" / path.name
        elif "corrigido_manual" in low or "manual_corrected" in low:
            destination = base / "manual_corrected" / path.name
        else:
            destination = base / "raw" / path.name
        move_safe(path, destination)


def migrate_figures(regime):
    base = ROOT / "figures" / regime
    base.mkdir(parents=True, exist_ok=True)
    if regime == "density":
        (base / "final").mkdir(parents=True, exist_ok=True)
        move_direct_children(base, base / "final", excluded_names=("final",))
    else:
        (base / "raw").mkdir(parents=True, exist_ok=True)
        (base / "manual_corrected").mkdir(parents=True, exist_ok=True)
        legacy = base / "legacy_unclassified"
        legacy.mkdir(parents=True, exist_ok=True)
        move_direct_children(
            base,
            legacy,
            excluded_names=("raw", "manual_corrected", "legacy_unclassified"),
        )


def ensure_results(regime):
    base = ROOT / "results" / regime
    variants = ["final"] if regime == "density" else ["raw", "manual_corrected"]
    for variant in variants:
        (base / variant).mkdir(parents=True, exist_ok=True)


def main():
    print(f"Repository: {ROOT}\n")
    for regime in ("density", "mixed", "velocity"):
        print(f"--- {regime} ---")
        migrate_checkpoints(regime)
        migrate_exports(regime)
        migrate_figures(regime)
        ensure_results(regime)
        print()

    print("Migration finished. No dataset was recomputed.")
    print("Mixed/velocity old figures, when present, were preserved in legacy_unclassified/ because their stage could not be identified safely.")
    print("Next: apply manual corrections if needed, then regenerate raw/final figures into their separate folders.")


if __name__ == "__main__":
    main()
