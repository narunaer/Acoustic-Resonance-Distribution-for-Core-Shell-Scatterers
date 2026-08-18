import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_widths

from .physics import (
    core_shell_denominator,
    make_materials_from_alpha_beta,
    total_internal_energy,
)
from .regimes import RegimeConfig


def global_nonuniform_mesh(x_min=1e-4, x_max=1.0, x_split=0.1, n_low=1500, n_high=1500):
    if not (0.0 < x_min < x_max):
        raise ValueError("Require 0 < x_min < x_max.")

    x_split = float(np.clip(x_split, x_min, x_max))
    n_low = max(int(n_low), 2)
    n_high = max(int(n_high), 2)

    if x_split <= x_min:
        return np.linspace(x_min, x_max, n_low + n_high)
    if x_split >= x_max:
        return np.geomspace(x_min, x_max, n_low + n_high)

    low = np.geomspace(x_min, x_split, n_low, endpoint=False)
    high = np.linspace(x_split, x_max, n_high)
    return np.unique(np.concatenate([low, high]))


def d0_magnitude_squared(xM, delta, rho0, rho1, rho2, c0, c1, c2):
    xM = np.asarray(xM, dtype=float)
    xA = (c0 / c1) * xM
    xB = (c0 / c2) * xM
    yM = (1.0 + delta) * xM
    yB = (1.0 + delta) * xB

    with np.errstate(all="ignore"):
        D = core_shell_denominator(
            0, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2
        )
    return np.real(D) ** 2 + np.imag(D) ** 2


def d0_candidates(
    delta,
    rho0,
    rho1,
    rho2,
    c0,
    c1,
    c2,
    x_min=1e-4,
    x_max=1.0,
    n_coarse=6000,
):
    x = np.linspace(x_min, x_max, int(n_coarse))
    values = d0_magnitude_squared(x, delta, rho0, rho1, rho2, c0, c1, c2)

    valid = np.isfinite(values[:-2]) & np.isfinite(values[1:-1]) & np.isfinite(values[2:])
    local_min = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    indices = np.flatnonzero(valid & local_min) + 1

    candidates = []
    for i in indices:
        lo = x[max(i - 2, 0)]
        hi = x[min(i + 2, len(x) - 1)]
        result = minimize_scalar(
            lambda xx: d0_magnitude_squared(
                xx, delta, rho0, rho1, rho2, c0, c1, c2
            ),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-13},
        )
        if result.success and np.isfinite(result.x):
            candidates.append(float(result.x))

    if not candidates:
        return np.array([], dtype=float)
    return np.unique(np.asarray(candidates, dtype=float))


def d0_background_ratio(xc, delta, rho0, rho1, rho2, c0, c1, c2, x_min, x_max):
    value = d0_magnitude_squared(xc, delta, rho0, rho1, rho2, c0, c1, c2)
    x_background = np.concatenate([
        np.linspace(xc * 0.5, xc * 0.9, 25),
        np.linspace(xc * 1.1, xc * 1.5, 25),
    ])
    x_background = x_background[(x_background > x_min) & (x_background < x_max)]

    if len(x_background) == 0 or not np.isfinite(value):
        return np.inf

    background = d0_magnitude_squared(
        x_background, delta, rho0, rho1, rho2, c0, c1, c2
    )
    background = background[np.isfinite(background) & (background > 0.0)]
    if len(background) == 0:
        return np.inf

    median = np.median(background)
    return np.inf if median <= 0.0 else float(value / median)


def prepare_log_signal(W):
    W = np.asarray(W, dtype=float)
    valid = np.isfinite(W) & (W > 0.0)
    if np.count_nonzero(valid) < 5:
        return None

    floor = max(np.min(W[valid]) * 1e-14, np.finfo(float).tiny)
    clipped = np.where(valid, W, floor)
    return np.log10(np.maximum(clipped, floor))


def peaks_on_mesh(x_mesh, W, config):
    x_mesh = np.asarray(x_mesh, dtype=float)
    W = np.asarray(W, dtype=float)
    y_log = prepare_log_signal(W)
    if y_log is None:
        return []

    kwargs = {"prominence": config.prominence_min}
    if config.min_peak_distance_samples is not None:
        kwargs["distance"] = max(
            1,
            min(int(config.min_peak_distance_samples), len(y_log) - 1),
        )

    indices, _ = find_peaks(y_log, **kwargs)
    indices = indices[(indices >= 3) & (indices <= len(x_mesh) - 4)]
    if len(indices) == 0:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, left_ips, right_ips = peak_widths(W, indices, rel_height=0.5)

    axis = np.arange(len(x_mesh), dtype=float)
    x_left = np.interp(left_ips, axis, x_mesh)
    x_right = np.interp(right_ips, axis, x_mesh)

    peaks = []
    for k, i in enumerate(indices):
        x0 = float(x_mesh[i])
        fwhm = float(x_right[k] - x_left[k])
        w_peak = float(W[i])

        if not np.isfinite(fwhm) or fwhm <= 0.0:
            continue
        if not np.isfinite(w_peak) or w_peak <= 0.0:
            continue
        if config.max_candidate_fwhm is not None and fwhm > config.max_candidate_fwhm:
            continue

        peaks.append({
            "xM_peak": x0,
            "FWHM": fwhm,
            "Q": x0 / fwhm,
            "WT_peak": w_peak,
        })

    return peaks


def peak_background_ratio(xc, w_peak, fwhm, x_mesh, W):
    margin = max(30.0 * fwhm, 0.1 * xc)
    x_background = np.concatenate([
        np.linspace(xc - 2.0 * margin, xc - margin, 25),
        np.linspace(xc + margin, xc + 2.0 * margin, 25),
    ])
    x_background = x_background[
        (x_background >= x_mesh[0]) & (x_background <= x_mesh[-1])
    ]

    if len(x_background) == 0 or not np.isfinite(w_peak) or w_peak <= 0.0:
        return 0.0

    background = np.interp(x_background, x_mesh, W)
    background = background[np.isfinite(background) & (background > 0.0)]
    if len(background) == 0:
        return np.inf

    median = np.median(background)
    return np.inf if median <= 0.0 else float(w_peak / median)


def candidate_neighbor_distances(candidates, x_min, x_max):
    candidates = np.asarray(candidates, dtype=float)
    if len(candidates) == 0:
        return np.array([], dtype=float)
    if len(candidates) == 1:
        return np.array([x_max - x_min], dtype=float)

    left = np.empty_like(candidates)
    right = np.empty_like(candidates)
    left[0] = candidates[0] - x_min
    left[1:] = np.diff(candidates)
    right[:-1] = np.diff(candidates)
    right[-1] = x_max - candidates[-1]
    return np.minimum(left, right)


def measure_d0_candidate_with_zoom(
    xc,
    delta,
    materials,
    x_min,
    x_max,
    max_window,
    lmax,
    config,
):
    rho0, rho1, rho2, c0, c1, c2 = materials

    for base_window in config.zoom_windows:
        window = min(float(base_window), float(max_window))
        if window <= 0.0:
            continue

        lo = max(x_min, xc - window)
        hi = min(x_max, xc + window)
        if hi <= lo:
            continue

        x_local = np.linspace(lo, hi, config.zoom_points)
        W_local = total_internal_energy(
            x_local, delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
        )
        peaks = peaks_on_mesh(x_local, W_local, config)
        if not peaks:
            continue

        nearest = dict(min(peaks, key=lambda peak: abs(peak["xM_peak"] - xc)))
        dx = float(np.median(np.diff(x_local)))
        tolerance = max(config.min_match_tolerance, config.match_factor * dx)

        if abs(nearest["xM_peak"] - xc) <= max(tolerance, 0.1 * window):
            return nearest

    return None


def confirm_d0_candidate(
    regime,
    xc,
    recovered,
    x_mesh,
    W_total,
    delta,
    materials,
    x_min,
    x_max,
    lmax,
):
    config = regime.detector

    if regime.name == "density":
        return (
            peak_background_ratio(
                xc,
                recovered["WT_peak"],
                recovered["FWHM"],
                x_mesh,
                W_total,
            )
            >= config.min_peak_background_ratio
        )

    rho0, rho1, rho2, c0, c1, c2 = materials
    x_background = np.concatenate([
        np.linspace(xc * 0.5, xc * 0.9, 25),
        np.linspace(xc * 1.1, xc * 1.5, 25),
    ])
    x_background = x_background[(x_background > x_min) & (x_background < x_max)]
    if len(x_background) == 0:
        return False

    w_xc = total_internal_energy(
        np.array([xc]), delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )[0]
    background = np.interp(x_background, x_mesh, W_total)
    background = background[np.isfinite(background) & (background > 0.0)]

    if len(background) == 0 or not np.isfinite(w_xc) or w_xc <= 0.0:
        return False

    median = np.median(background)
    return bool(
        median > 0.0
        and w_xc > config.d0_energy_ratio_min * median
    )


def detect_resonances(
    delta,
    alpha,
    beta,
    regime,
    mix_factor=None,
    x_min=1e-4,
    x_max=1.0,
    n_global_mesh=3000,
    lmax=10,
    max_match_tolerance=2e-3,
):
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative.")

    config = regime.detector
    materials = make_materials_from_alpha_beta(
        alpha,
        beta,
        contrast_in=regime.contrast_in,
        mix_factor=regime.mix_factor if mix_factor is None else mix_factor,
    )
    rho0, rho1, rho2, c0, c1, c2 = materials

    candidates = np.sort(
        d0_candidates(
            delta,
            rho0,
            rho1,
            rho2,
            c0,
            c1,
            c2,
            x_min=x_min,
            x_max=x_max,
        )
    )

    n_global_mesh = max(int(n_global_mesh), 20)
    n_low = n_global_mesh // 2
    n_high = n_global_mesh - n_low
    x_mesh = global_nonuniform_mesh(
        x_min=x_min,
        x_max=x_max,
        n_low=n_low,
        n_high=n_high,
    )
    local_spacing = np.gradient(x_mesh)

    W_total = total_internal_energy(
        x_mesh, delta, rho0, rho1, rho2, c0, c1, c2, lmax=lmax
    )
    global_peaks = peaks_on_mesh(x_mesh, W_total, config)
    result = []

    for peak in global_peaks:
        if config.min_peak_background_ratio is not None:
            ratio = peak_background_ratio(
                peak["xM_peak"],
                peak["WT_peak"],
                peak["FWHM"],
                x_mesh,
                W_total,
            )
            if ratio < config.min_peak_background_ratio:
                continue

        item = dict(peak)
        item["origem"] = "malha_global"
        item["D0_confirmado"] = False
        item["xM_D0"] = np.nan
        result.append(item)

    neighbor_distance = candidate_neighbor_distances(candidates, x_min, x_max)

    for xc, distance_to_neighbor in zip(candidates, neighbor_distance):
        ratio_d0 = d0_background_ratio(
            xc,
            delta,
            rho0,
            rho1,
            rho2,
            c0,
            c1,
            c2,
            x_min,
            x_max,
        )
        if ratio_d0 > config.d0_background_ratio_max:
            continue

        spacing_here = float(np.interp(xc, x_mesh, local_spacing))
        tolerance = max(
            config.min_match_tolerance,
            config.match_factor * spacing_here,
        )
        tolerance = min(tolerance, float(max_match_tolerance))

        if result:
            distances = np.array(
                [abs(peak["xM_peak"] - xc) for peak in result],
                dtype=float,
            )
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] <= tolerance:
                result[nearest_index]["D0_confirmado"] = True
                result[nearest_index]["xM_D0"] = float(xc)
                continue

        max_window = max(float(distance_to_neighbor) / 2.0, 1e-6)
        recovered = measure_d0_candidate_with_zoom(
            xc,
            delta,
            materials,
            x_min,
            x_max,
            max_window,
            lmax,
            config,
        )
        if recovered is None:
            continue

        if not confirm_d0_candidate(
            regime,
            xc,
            recovered,
            x_mesh,
            W_total,
            delta,
            materials,
            x_min,
            x_max,
            lmax,
        ):
            continue

        if result:
            distances = np.array(
                [abs(peak["xM_peak"] - recovered["xM_peak"]) for peak in result],
                dtype=float,
            )
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] <= tolerance:
                result[nearest_index]["D0_confirmado"] = True
                result[nearest_index]["xM_D0"] = float(xc)
                continue

        recovered = dict(recovered)
        recovered["origem"] = "zoom_D0"
        recovered["D0_confirmado"] = True
        recovered["xM_D0"] = float(xc)
        result.append(recovered)

    result.sort(key=lambda peak: peak["xM_peak"])

    if config.q_min_base is not None:
        q_min = (
            config.q_min_many_peaks
            if len(result) > config.many_peaks_threshold
            else config.q_min_base
        )
        for peak in result:
            peak["acima_do_qmin"] = bool(peak["Q"] >= q_min)
            peak["largura_fina"] = bool(
                peak["FWHM"] <= config.thin_fwhm_threshold
            )
            peak["q_min_usado"] = float(q_min)
            peak["lmax_usado"] = int(lmax)

    return result
