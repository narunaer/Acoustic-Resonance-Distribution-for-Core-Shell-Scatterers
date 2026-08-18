#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

from core_shell.checkpoints import load_curves_dataframe, find_repository_checkpoint, repository_figure_dir
from core_shell.detection import global_nonuniform_mesh
from core_shell.physics import make_materials_from_alpha_beta, total_internal_energy
from core_shell.regimes import get_regime

REPRESENTATIVE_PAIRS = {
    "eta_positive": {"label": r"$\eta\approx+0.7$", "alpha": 0.006333333333333333, "beta": 0.001},
    "eta_zero": {"label": r"$\eta\approx0$", "alpha": 0.008111111111111111, "beta": 0.008111111111111111},
    "eta_negative": {"label": r"$\eta\approx-0.7$", "alpha": 0.001, "beta": 0.006333333333333333},
}
TARGET_DELTAS = [0.10, 0.50, 1.35]
COLORS = ["#0f223d", "#810000", "black"]
LINESTYLES = ["-", "--", "-."]
Y_TICKS = np.array([1e-7, 1e-3, 1e1, 1e5, 1e9])
Y_LIM = (1e-7, 1e10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=("raw", "manual_corrected", "final"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lmax", type=int, default=10)
    return parser.parse_args()


def nearest_pair(df, alpha, beta):
    pairs = df[["alpha", "beta"]].drop_duplicates().reset_index(drop=True)
    target = np.log([alpha, beta])
    available = np.log(pairs.to_numpy(dtype=float))
    index = int(np.argmin(np.sum((available - target) ** 2, axis=1)))
    return float(pairs.loc[index, "alpha"]), float(pairs.loc[index, "beta"])


def nearest_delta_rows(df, alpha, beta):
    pair_df = df[
        np.isclose(df["alpha"], alpha, rtol=1e-9)
        & np.isclose(df["beta"], beta, rtol=1e-9)
    ]
    rows = []
    for target in TARGET_DELTAS:
        if pair_df.empty:
            rows.append(None)
            continue
        index = int(np.argmin(np.abs(pair_df["delta"].to_numpy(dtype=float) - target)))
        rows.append(pair_df.iloc[index])
    return rows


def summarize(row, mode):
    if row is None or not row["peaks"]:
        return None
    peaks = row["peaks"]
    x = np.array([p[0] for p in peaks])
    fwhm = np.array([p[1] for p in peaks])
    q = np.array([p[2] for p in peaks])

    if mode == "two_peaks":
        second = peaks[1] if len(peaks) >= 2 else (np.nan, np.nan, np.nan, np.nan)
        return [
            x[0], fwhm[0], q[0],
            second[0], second[1], second[2],
            second[0] - x[0] if len(peaks) >= 2 else np.nan,
        ]

    gaps = np.diff(x)
    return [
        x[0],
        fwhm[0],
        q[0],
        np.median(gaps) if len(gaps) else np.nan,
        np.mean(fwhm[np.isfinite(fwhm)]) if np.any(np.isfinite(fwhm)) else np.nan,
        np.mean(q[np.isfinite(q)]) if np.any(np.isfinite(q)) else np.nan,
    ]


def save_table(path, summaries, mode):
    if mode == "two_peaks":
        row_labels = [
            r"1st Peak ($x_M$)", "1st Peak FWHM", r"1st Peak $Q$-factor",
            r"2nd Peak ($x_M$)", "2nd Peak FWHM", r"2nd Peak $Q$-factor",
            r"Peak Separation ($\Delta x_M$)",
        ]
    else:
        row_labels = [
            r"1st Peak ($x_M$)", "1st Peak FWHM", r"1st Peak $Q$-factor",
            r"Macro-FSR ($\Delta x_M$)", "Mean FWHM", r"Mean $Q$-factor",
        ]

    data = []
    for i in range(len(row_labels)):
        row = []
        for summary in summaries:
            if summary is None or not np.isfinite(summary[i]):
                row.append("--")
            else:
                row.append(f"{summary[i]:.4g}")
        data.append(row)

    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.axis("off")
        table = ax.table(
            cellText=[[row_labels[i], *data[i]] for i in range(len(row_labels))],
            colLabels=["Parameter", *[rf"$\delta={d:.2f}$" for d in TARGET_DELTAS]],
            cellLoc="center",
            colWidths=[0.38, 0.206, 0.206, 0.208],
            bbox=[0.02, 0.02, 0.96, 0.96],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.visible_edges = ""
            if col == 0:
                cell.get_text().set_ha("left")
        for col in range(4):
            table[(0, col)].visible_edges = "TB"
            table[(0, col)].set_linewidth(1.4)
            table[(len(row_labels), col)].visible_edges = "B"
            table[(len(row_labels), col)].set_linewidth(1.4)
        fig.savefig(path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(path + ".svg", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


def plot_case(path, rows, label, regime, lmax):
    x_mesh = global_nonuniform_mesh(1e-4, 1.0, n_low=1500, n_high=1500)

    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix"}):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))

        for i, row in enumerate(rows):
            if row is None:
                continue
            materials = make_materials_from_alpha_beta(
                row["alpha"],
                row["beta"],
                contrast_in=regime.contrast_in,
                mix_factor=regime.mix_factor,
            )
            W = total_internal_energy(x_mesh, row["delta"], *materials, lmax=lmax)
            color = COLORS[i]
            ax.plot(
                x_mesh, W,
                color=color,
                linestyle=LINESTYLES[i],
                linewidth=2.0,
                label=rf"$\delta={row['delta']:.2f}$",
            )

            visible_peaks = [p for p in row["peaks"] if p[0] <= regime.selected_x_max]
            if visible_peaks:
                ax.scatter(
                    [p[0] for p in visible_peaks],
                    [p[3] for p in visible_peaks],
                    s=24,
                    color=color,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=5,
                )

        ax.set_yscale("log")
        ax.set_xlim(1e-4, regime.selected_x_max)
        ax.set_ylim(*Y_LIM)
        ax.xaxis.set_major_locator(FixedLocator(regime.selected_x_ticks))
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.minorticks_off()
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$x_M=k_Ma$", fontsize=20)
        ax.tick_params(direction="in", length=5, width=1.2, labelsize=16)
        ax.grid(False)
        ax.legend(loc="lower right", fontsize=16, frameon=True)
        ax.text(
            0.965, 0.94, label,
            transform=ax.transAxes, ha="right", va="top", fontsize=16,
            bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": "black"},
        )
        fig.savefig(path + ".png", dpi=600, bbox_inches="tight")
        fig.savefig(path + ".pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    regime = get_regime(args.regime)
    checkpoint = args.checkpoint or find_repository_checkpoint(args.regime, args.variant)
    df = load_curves_dataframe(checkpoint, include_wt=True)
    output_dir = args.output_dir or os.path.join(repository_figure_dir(args.regime, args.variant), "selected_cases")
    os.makedirs(output_dir, exist_ok=True)

    for case_name, info in REPRESENTATIVE_PAIRS.items():
        alpha, beta = nearest_pair(df, info["alpha"], info["beta"])
        rows = nearest_delta_rows(df, alpha, beta)
        summaries = [summarize(row, regime.selected_summary) for row in rows]
        save_table(os.path.join(output_dir, f"{args.regime}_{case_name}_table"), summaries, regime.selected_summary)
        plot_case(
            os.path.join(output_dir, f"{args.regime}_{case_name}_energy"),
            rows,
            info["label"],
            regime,
            args.lmax,
        )

    print(f"Saved selected-case figures to: {output_dir}")


if __name__ == "__main__":
    main()
