#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core_shell.analysis import eta_bin_indices, nearest_index, project_peaks_to_delta_eta
from core_shell.checkpoints import DELTA_VALUES, ETA_BINS, load_curves_dataframe, find_repository_checkpoint, repository_figure_dir
from core_shell.regimes import get_regime

CMAP = LinearSegmentedColormap.from_list(
    "mean_resonances", [(0.0, "white"), (0.5, "#526F92"), (1.0, "black")], N=256
)
CMAP.set_bad("#D9D9D9")
BOX_COLOR = "#600B00"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=("raw", "manual_corrected", "final"), default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def build_scale(mean_map, regime):
    values = mean_map[np.isfinite(mean_map)]
    if values.size == 0:
        raise RuntimeError("No finite cells are available.")

    if regime.mean_map_fixed_max is not None:
        scale_max = regime.mean_map_fixed_max
    else:
        scale_max = max(regime.mean_map_ticks - 1, int(np.ceil(np.max(values))))

    ticks = np.rint(np.linspace(0.0, scale_max, regime.mean_map_ticks)).astype(int)
    if len(np.unique(ticks)) < regime.mean_map_ticks:
        scale_max = regime.mean_map_ticks - 1
        ticks = np.arange(regime.mean_map_ticks, dtype=int)

    return Normalize(vmin=0.0, vmax=scale_max, clip=True), ticks, float(np.max(values)), scale_max


def main():
    args = parse_args()
    regime = get_regime(args.regime)
    checkpoint = args.checkpoint or find_repository_checkpoint(args.regime, args.variant)
    df = load_curves_dataframe(checkpoint)
    eta_centers, mean_map, samples_map = project_peaks_to_delta_eta(df, DELTA_VALUES, ETA_BINS)
    norm, ticks, data_max, scale_max = build_scale(mean_map, regime)

    output_dir = args.output_dir or repository_figure_dir(args.regime, args.variant)
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, f"{args.regime}_delta_eta_mean_resonances")

    x_edges = np.arange(len(DELTA_VALUES) + 1) - 0.5
    y_edges = np.arange(len(eta_centers) + 1) - 0.5

    fig = plt.figure(figsize=(11, 13))
    ax = fig.add_axes([0.105, 0.115, 0.725, 0.725])
    mesh = ax.pcolormesh(
        x_edges, y_edges, mean_map.T,
        cmap=CMAP, norm=norm, shading="flat",
        edgecolors="white", linewidths=0.42,
    )

    eta_highlights = [(0.7, BOX_COLOR), (0.0, "#B0B0B0"), (-0.7, "black")]
    delta_highlights = [0.1, 0.5, 1.4]
    eta_indices = [
        int(eta_bin_indices([eta], ETA_BINS)[0])
        for eta, _ in eta_highlights
    ]
    delta_indices = [nearest_index(delta, DELTA_VALUES) for delta in delta_highlights]

    for i_delta in delta_indices:
        for i_eta, (_, color) in zip(eta_indices, eta_highlights):
            ax.add_patch(Rectangle(
                (i_delta - 0.5, i_eta - 0.5), 1, 1,
                fill=False, edgecolor=color, linewidth=5, zorder=10,
            ))

    delta_ticks = [0.01, 0.2, 0.5, 0.8, 1.1, 1.5]
    delta_indices = [nearest_index(value, DELTA_VALUES) for value in delta_ticks]
    ax.set_xticks(delta_indices)
    ax.set_xticklabels([f"{DELTA_VALUES[i]:g}" for i in delta_indices])

    eta_ticks = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    eta_step = ETA_BINS[1] - ETA_BINS[0]
    eta_positions = (eta_ticks - ETA_BINS[0]) / eta_step - 0.5
    ax.set_yticks(eta_positions)
    ax.set_yticklabels([f"{value:.1f}" for value in eta_ticks])
    ax.axhline((0.0 - ETA_BINS[0]) / eta_step - 0.5, color="#C7D0DB", linestyle="--", linewidth=1.35)

    ax.set_xlabel(r"$\delta$", fontsize=30)
    ax.set_ylabel(r"$\eta=(Z_A-Z_B)/(Z_A+Z_B)$", fontsize=30)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=24)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.25)
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cbar.set_label("Resonances", fontsize=30, labelpad=26, rotation=270)
    cbar.ax.tick_params(labelsize=24)

    ax.text(
        0.98, 0.975, r"$\left\langle N_{\mathrm{res}}\right\rangle$",
        transform=ax.transAxes, fontsize=30, color="white",
        va="top", ha="right",
        bbox={"facecolor": BOX_COLOR, "edgecolor": "none"},
    )

    for ext in ("png", "pdf", "eps"):
        fig.savefig(f"{output_base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        f"{output_base}_data.npz",
        eta_bins=ETA_BINS,
        eta_centers=eta_centers,
        delta_values=DELTA_VALUES,
        mean_map=mean_map,
        samples_map=samples_map,
        data_max=data_max,
        scale_max=scale_max,
        ticks=ticks,
    )
    print(f"Saved: {output_base}.png/.pdf/.eps")


if __name__ == "__main__":
    main()
