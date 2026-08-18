#!/usr/bin/env python
import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from core_shell.analysis import bin_delta_in_alpha_beta_plane, boundary_segments
from core_shell.checkpoints import DELTA_VALUES, load_curves_dataframe, find_repository_checkpoint, repository_figure_dir
from core_shell.regimes import get_regime

FIXED_DELTA = 1.50
OVERLAY_DELTAS = [0.01, 0.50]
MIN_Z, MAX_Z, N_LOG_BINS = 1e-3, 1000.0, 20
BOX_COLOR = "#600B00"
CMAP = LinearSegmentedColormap.from_list(
    "resonance_count", ["white", "#D9E1EA", "#526F92", "black"], N=256
)
CMAP.set_bad("#D9D9D9")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=("raw", "manual_corrected", "final"), default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def fill_empty_cells(max_map, count_map, curves_df, real_delta, log_edges):
    occupied = count_map > 0
    empty = ~occupied
    if not (np.any(empty) and np.any(occupied)):
        return max_map

    delta_values = np.asarray(DELTA_VALUES, dtype=float)
    target_index = int(np.argmin(np.abs(delta_values - real_delta)))
    curve_delta_indices = np.array([
        int(np.argmin(np.abs(delta_values - value)))
        for value in curves_df["delta"].to_numpy(dtype=float)
    ])
    mask = curve_delta_indices == target_index
    alpha = curves_df.loc[mask, "alpha"].to_numpy(dtype=float)
    beta = curves_df.loc[mask, "beta"].to_numpy(dtype=float)
    peaks = curves_df.loc[mask, "n_peaks"].to_numpy(dtype=float)

    valid = (
        np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(peaks)
        & (alpha >= MIN_Z) & (alpha <= MAX_Z)
        & (beta >= MIN_Z) & (beta <= MAX_Z)
    )
    log_alpha = np.log10(alpha[valid])
    log_beta = np.log10(beta[valid])
    peaks = peaks[valid]

    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    grid_alpha, grid_beta = np.meshgrid(centers, centers, indexing="ij")
    empty_coordinates = np.column_stack([grid_alpha[empty], grid_beta[empty]])
    valid_points = np.column_stack([log_alpha, log_beta])
    _, nearest = cKDTree(valid_points).query(empty_coordinates)

    filled = max_map.copy()
    filled[empty] = peaks[nearest]
    return filled


def coordinate(value, log_edges):
    n_bins = len(log_edges) - 1
    return (np.log10(value) - log_edges[0]) / (log_edges[-1] - log_edges[0]) * n_bins - 0.5


def draw_segments(ax, segments, color, linestyle, linewidth, alpha, zorder):
    for start, end in segments:
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=color, linestyle=linestyle, linewidth=linewidth,
            alpha=alpha, solid_capstyle="butt", zorder=zorder,
        )


def main():
    args = parse_args()
    regime = get_regime(args.regime)
    checkpoint = args.checkpoint or find_repository_checkpoint(args.regime, args.variant)
    df = load_curves_dataframe(checkpoint)
    log_edges = np.linspace(np.log10(MIN_Z), np.log10(MAX_Z), N_LOG_BINS + 1)

    base_delta, base_map, base_counts = bin_delta_in_alpha_beta_plane(
        df, DELTA_VALUES, FIXED_DELTA, log_edges
    )
    base_map = fill_empty_cells(base_map, base_counts, df, base_delta, log_edges)

    overlay = []
    for delta in OVERLAY_DELTAS:
        real_delta, max_map, counts = bin_delta_in_alpha_beta_plane(
            df, DELTA_VALUES, delta, log_edges
        )
        max_map = fill_empty_cells(max_map, counts, df, real_delta, log_edges)
        overlay.append((real_delta, max_map))

    resonance_map = np.where(np.isfinite(base_map), np.maximum(base_map, 0.0), np.nan)
    finite = resonance_map[np.isfinite(resonance_map)]
    max_resonances = int(np.ceil(np.max(finite))) if finite.size else 0
    norm = Normalize(vmin=0.0, vmax=max(max_resonances, 1))
    ticks = np.unique(np.rint(np.linspace(0, max_resonances, min(5, max_resonances + 1))).astype(int)) if max_resonances else np.array([0])

    output_dir = args.output_dir or repository_figure_dir(args.regime, args.variant)
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, f"{args.regime}_alpha_beta_resonance_map")

    fig = plt.figure(figsize=(10.5, 10.5))
    ax = fig.add_axes([0.12, 0.13, 0.70, 0.70])
    cax = fig.add_axes([0.84, 0.13, 0.03, 0.70])

    bad_color = CMAP.get_bad()
    for i_alpha in range(N_LOG_BINS):
        for i_beta in range(N_LOG_BINS):
            value = resonance_map[i_alpha, i_beta]
            color = CMAP(norm(value)) if np.isfinite(value) else bad_color
            ax.add_patch(Rectangle(
                (i_alpha - 0.5, i_beta - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.42,
            ))

    base_style = ("#7FA6D9", (0, (5.0, 2.2)), 2.0, 0.95)
    overlay_styles = [
        ("#600B00", (0, (4.5, 2.5)), 1.9, 0.92),
        ("black", (0, (5.0, 2.2)), 2.0, 0.82),
    ]
    draw_segments(ax, boundary_segments(base_map > 0), *base_style, 17)
    for i, (_, max_map) in enumerate(overlay):
        draw_segments(ax, boundary_segments(max_map > 0), *overlay_styles[i], 18 + i)

    n_bins = N_LOG_BINS
    line_style = dict(color="#AEB9C5", linestyle=":", linewidth=2.43, alpha=0.35, zorder=6)
    ax.plot([-0.5, n_bins - 0.5], [-0.5, n_bins - 0.5], **line_style)
    one = coordinate(1.0, log_edges)
    ax.axvline(one, **line_style)
    ax.axhline(one, **line_style)

    tick_values = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], dtype=float)
    tick_positions = [coordinate(value, log_edges) for value in tick_values]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{value:g}" for value in tick_values])
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{value:g}" for value in tick_values])
    ax.set_xlabel(r"$\alpha=Z_A/Z_M$", fontsize=23)
    ax.set_ylabel(r"$\beta=Z_B/Z_M$", fontsize=23)
    ax.tick_params(labelsize=15)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(-0.5, n_bins - 0.5)
    ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D([0], [0], color=overlay_styles[i][0], linestyle=overlay_styles[i][1],
               linewidth=overlay_styles[i][2], label=rf"Reference frontier: $\delta={overlay[i][0]:g}$")
        for i in range(2)
    ]
    handles.append(Line2D(
        [0], [0], color=base_style[0], linestyle=base_style[1],
        linewidth=base_style[2], label=rf"Reference frontier: $\delta={base_delta:g}$",
    ))
    y_anchor = one - 3.0 if regime.alpha_beta_legend_side == "below" else one + 1.0
    ax.legend(
        handles=handles, loc="lower right",
        bbox_to_anchor=(19.15, y_anchor), bbox_transform=ax.transData,
        fontsize=16, frameon=True, facecolor="white", edgecolor="none",
    )

    ax.text(
        0.50, 0.965, rf"$\delta={base_delta:g}$",
        transform=ax.transAxes, fontsize=24, color="white",
        va="top", ha="center", bbox={"facecolor": BOX_COLOR, "edgecolor": "none"},
    )
    ax.text(
        0.955, 0.965, r"$N_{\mathrm{res}}$",
        transform=ax.transAxes, fontsize=24, color="white",
        va="top", ha="right", bbox={"facecolor": BOX_COLOR, "edgecolor": "none"},
    )

    mapper = cm.ScalarMappable(norm=norm, cmap=CMAP)
    mapper.set_array([])
    cbar = fig.colorbar(mapper, cax=cax, ticks=ticks)
    cbar.set_label("Resonances", fontsize=20, labelpad=18, rotation=270)
    cbar.ax.tick_params(labelsize=15)

    for ext in ("png", "pdf", "eps"):
        fig.savefig(f"{output_base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        f"{output_base}_data.npz",
        fixed_delta=base_delta,
        overlay_deltas=np.asarray([item[0] for item in overlay]),
        log_edges=log_edges,
        resonance_map=resonance_map,
        samples_per_bin=base_counts,
        resonance_scale_max=max_resonances,
        resonance_ticks=ticks,
    )
    print(f"Saved: {output_base}.png/.pdf/.eps")


if __name__ == "__main__":
    main()
