import os
import time
import uuid

import numpy as np
import pandas as pd

from .physics import eta_from_alpha_beta

DELTA_VALUES = np.array([
    0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
    0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50,
], dtype=float)

ETA_BINS = np.linspace(-1.0, 1.0, 21)
TARGET_PER_BIN = 20
CHECKPOINT_VERSION = 2


def build_eta_grid(target_per_bin=TARGET_PER_BIN, seed=42):
    values = np.unique(np.concatenate([
        np.linspace(0.001, 0.009, 10),
        np.linspace(0.01, 0.09, 10),
        np.linspace(0.1, 0.9, 10),
        np.linspace(1.0, 9.0, 10),
        np.linspace(10.0, 90.0, 10),
        np.linspace(100.0, 900.0, 10),
        [1000.0, 2000.0, 3000.0],
    ]))

    alpha_mesh, beta_mesh = np.meshgrid(values, values, indexing="ij")
    alpha_all = alpha_mesh.ravel()
    beta_all = beta_mesh.ravel()
    eta_all = (alpha_all - beta_all) / (alpha_all + beta_all)

    bin_index = np.searchsorted(ETA_BINS, eta_all, side="left") - 1
    bin_index[eta_all == ETA_BINS[0]] = 0
    bin_index = np.clip(bin_index, 0, len(ETA_BINS) - 2)

    rng = np.random.default_rng(seed)
    selected_alpha, selected_beta = [], []

    for i in range(len(ETA_BINS) - 1):
        mask = bin_index == i
        alpha_bin = alpha_all[mask]
        beta_bin = beta_all[mask]
        available = alpha_bin.size

        if available > target_per_bin:
            indices = np.linspace(0, available - 1, target_per_bin, dtype=int)
            alpha_bin = alpha_bin[indices]
            beta_bin = beta_bin[indices]
        else:
            missing = target_per_bin - available
            if missing > 0:
                eta_mid = 0.5 * (ETA_BINS[i] + ETA_BINS[i + 1])
                scales = rng.choice(values, size=missing, replace=True)
                alpha_bin = np.concatenate([
                    alpha_bin,
                    scales * (1.0 + eta_mid) / 2.0,
                ])
                beta_bin = np.concatenate([
                    beta_bin,
                    scales * (1.0 - eta_mid) / 2.0,
                ])

        selected_alpha.append(alpha_bin)
        selected_beta.append(beta_bin)

    return np.concatenate(selected_alpha), np.concatenate(selected_beta)


def safe_savez(path, **arrays):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(
        directory,
        "." + os.path.basename(path) + f".tmp_{uuid.uuid4().hex}.npz",
    )
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, path)


def checkpoint_paths(output_root, regime, lmax):
    contrast = regime.contrast_in
    save_dir = os.path.join(
        output_root,
        f"resultados_coreshell_total_lmax{int(lmax)}_{contrast}",
    )
    export_dir = os.path.join(save_dir, "exports")
    run_name = f"coreshell_total_lmax{int(lmax)}_{contrast}_final"
    return {
        "save_dir": save_dir,
        "checkpoint": os.path.join(save_dir, f"checkpoint_{run_name}.npz"),
        "peaks_csv": os.path.join(export_dir, f"peaks_{run_name}.csv"),
        "summary_csv": os.path.join(export_dir, f"summary_{run_name}.csv"),
    }


def load_resume_state(checkpoint_path, total, lmax):
    if not os.path.exists(checkpoint_path):
        return {}, np.zeros(total, dtype=bool), None, None

    with np.load(checkpoint_path, allow_pickle=True) as data:
        version = int(data.get("checkpoint_version", -1))
        saved_lmax = int(data.get("lmax", -1))

        if version != CHECKPOINT_VERSION or saved_lmax != int(lmax):
            return {}, np.zeros(total, dtype=bool), None, None

        curves = data["picos_por_curva"].item()
        done = np.asarray(data["done_flat"], dtype=bool)
        alpha_grid = (
            np.asarray(data["alpha_grid"], dtype=float)
            if "alpha_grid" in data.files
            else None
        )
        beta_grid = (
            np.asarray(data["beta_grid"], dtype=float)
            if "beta_grid" in data.files
            else None
        )

    return curves, done, alpha_grid, beta_grid


def save_checkpoint(checkpoint_path, curves, done, alpha_grid, beta_grid, lmax):
    safe_savez(
        checkpoint_path,
        checkpoint_version=CHECKPOINT_VERSION,
        lmax=int(lmax),
        picos_por_curva=np.array(curves, dtype=object),
        done_flat=np.asarray(done, dtype=bool),
        alpha_grid=np.asarray(alpha_grid, dtype=float),
        beta_grid=np.asarray(beta_grid, dtype=float),
        delta_vals=DELTA_VALUES,
    )


def export_checkpoint_csv(paths, curves, lmax):
    peak_rows = []
    summary_rows = []

    for key, item in curves.items():
        idd, ip = key.split("_")
        peaks = item["picos"]
        q_values = [
            peak["Q"]
            for peak in peaks
            if peak.get("Q") is not None and np.isfinite(peak["Q"])
        ]

        summary_rows.append({
            "delta_index": int(idd),
            "pair_index": int(ip),
            "delta": item["delta"],
            "alpha": item["alpha"],
            "beta": item["beta"],
            "eta": item["eta"],
            "lmax": int(lmax),
            "n_peaks": len(peaks),
            "n_d0_confirmed": sum(bool(p.get("D0_confirmado", False)) for p in peaks),
            "n_d0_recovered": sum(p.get("origem") == "zoom_D0" for p in peaks),
            "first_xM": peaks[0]["xM_peak"] if peaks else np.nan,
            "max_Q": max(q_values) if q_values else np.nan,
        })

        for peak_number, peak in enumerate(peaks, start=1):
            peak_rows.append({
                "delta_index": int(idd),
                "pair_index": int(ip),
                "delta": item["delta"],
                "alpha": item["alpha"],
                "beta": item["beta"],
                "eta": item["eta"],
                "peak_number": peak_number,
                "xM_peak": peak.get("xM_peak", np.nan),
                "FWHM": peak.get("FWHM", np.nan),
                "Q": peak.get("Q", np.nan),
                "WT_peak": peak.get("WT_peak", np.nan),
                "origin": peak.get("origem", ""),
                "D0_confirmed": peak.get("D0_confirmado", False),
                "xM_D0": peak.get("xM_D0", np.nan),
                "above_qmin": peak.get("acima_do_qmin", np.nan),
                "thin_width": peak.get("largura_fina", np.nan),
            })

    os.makedirs(os.path.dirname(paths["peaks_csv"]), exist_ok=True)
    pd.DataFrame(peak_rows).to_csv(paths["peaks_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)


def load_curves_dataframe(checkpoint_path, include_wt=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    with np.load(checkpoint_path, allow_pickle=True) as data:
        if "picos_por_curva" not in data.files:
            raise KeyError("Missing 'picos_por_curva' in checkpoint.")
        curves = data["picos_por_curva"].item()

    rows = []
    for key, item in curves.items():
        alpha = float(item.get("alpha", np.nan))
        beta = float(item.get("beta", np.nan))
        delta = float(item.get("delta", np.nan))
        if not (
            np.isfinite(alpha) and alpha > 0.0
            and np.isfinite(beta) and beta > 0.0
            and np.isfinite(delta)
        ):
            continue

        peaks = []
        for peak in item.get("picos", []):
            xM = peak.get("xM_peak", np.nan)
            if not (np.isfinite(xM) and xM > 0.0):
                continue

            fwhm = peak.get("FWHM", np.nan)
            q = peak.get("Q", np.nan)
            wt = peak.get("WT_peak", np.nan)
            values = [
                float(xM),
                float(fwhm) if np.isfinite(fwhm) else np.nan,
                float(q) if np.isfinite(q) else np.nan,
            ]
            if include_wt:
                values.append(float(wt) if np.isfinite(wt) else np.nan)
            peaks.append(tuple(values))

        peaks.sort(key=lambda peak: peak[0])
        rows.append({
            "curve_key": key,
            "alpha": alpha,
            "beta": beta,
            "eta": float(eta_from_alpha_beta(alpha, beta)),
            "delta": delta,
            "n_peaks": len(peaks),
            "x_peaks": [peak[0] for peak in peaks],
            "peaks": peaks,
        })

    return pd.DataFrame(rows)


def default_output_root():
    return "/content/drive/MyDrive" if os.path.exists("/content/drive/MyDrive") else "results"


def mount_colab_drive_if_available():
    try:
        from google.colab import drive
    except ImportError:
        return False

    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    return True


def repository_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def repository_variant(regime_name, variant=None):
    """Normalize the public repository variant name.

    Density has a single final dataset. Mixed and velocity preserve both the
    automatic detector output (raw) and the hand-curated article dataset
    (manual_corrected).
    """
    if regime_name == "density":
        if variant not in (None, "final"):
            raise ValueError("density supports only variant='final'.")
        return "final"

    if regime_name in ("mixed", "velocity"):
        if variant in (None, "final", "manual", "manual_corrected"):
            return "manual_corrected"
        if variant == "raw":
            return "raw"
        raise ValueError(
            f"{regime_name} variant must be 'raw' or 'manual_corrected'."
        )

    raise ValueError(f"Unknown regime: {regime_name}")


def repository_checkpoint_dir(regime_name, variant=None):
    variant = repository_variant(regime_name, variant)
    return os.path.join(
        repository_root(),
        "data",
        "checkpoints",
        regime_name,
        variant,
    )


def repository_export_dir(regime_name, variant=None):
    variant = repository_variant(regime_name, variant)
    return os.path.join(
        repository_root(),
        "data",
        "exports",
        regime_name,
        variant,
    )


def repository_figure_dir(regime_name, variant=None):
    variant = repository_variant(regime_name, variant)
    return os.path.join(
        repository_root(),
        "figures",
        regime_name,
        variant,
    )


def repository_results_dir(regime_name, variant=None):
    variant = repository_variant(regime_name, variant)
    return os.path.join(
        repository_root(),
        "results",
        regime_name,
        variant,
    )


def repository_checkpoint_name(regime, lmax, variant=None):
    variant = repository_variant(regime.name, variant)
    base = (
        f"checkpoint_coreshell_total_lmax{int(lmax)}_"
        f"{regime.contrast_in}_final"
    )
    if variant == "manual_corrected":
        base += "_corrigido_manual"
    return base + ".npz"


def find_repository_checkpoint(regime_name, variant=None):
    from .regimes import get_regime

    variant = repository_variant(regime_name, variant)
    regime = get_regime(regime_name)
    directory = repository_checkpoint_dir(regime_name, variant)
    expected = os.path.join(
        directory,
        repository_checkpoint_name(regime, 10, variant),
    )
    if os.path.exists(expected):
        return expected

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {directory}"
        )

    candidates = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".npz")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if variant == "manual_corrected":
            raise FileNotFoundError(
                f"No manually corrected checkpoint found in: {directory}\n"
                f"Run: python scripts/apply_manual_corrections.py --regime {regime_name}"
            )
        raise FileNotFoundError(f"No .npz checkpoint found in: {directory}")

    raise RuntimeError(
        "Multiple checkpoints were found. Pass --checkpoint explicitly:\n"
        + "\n".join(candidates)
    )


def repository_generation_paths(regime, lmax, variant=None):
    """Paths for generated datasets.

    Generation writes density to final and mixed/velocity to raw. Pass an
    explicit variant only for exporting/reading a different stage.
    """
    if variant is None:
        variant = "final" if regime.name == "density" else "raw"
    variant = repository_variant(regime.name, variant)

    checkpoint_dir = repository_checkpoint_dir(regime.name, variant)
    export_dir = repository_export_dir(regime.name, variant)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    checkpoint_name = repository_checkpoint_name(regime, lmax, variant)
    suffix = "manual_corrected" if variant == "manual_corrected" else variant
    return {
        "save_dir": checkpoint_dir,
        "checkpoint": os.path.join(checkpoint_dir, checkpoint_name),
        "peaks_csv": os.path.join(
            export_dir,
            f"peaks_{regime.name}_lmax{int(lmax)}_{suffix}.csv",
        ),
        "summary_csv": os.path.join(
            export_dir,
            f"summary_{regime.name}_lmax{int(lmax)}_{suffix}.csv",
        ),
    }
