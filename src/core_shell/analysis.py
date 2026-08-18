import numpy as np


def nearest_index(value, array):
    return int(np.argmin(np.abs(np.asarray(array, dtype=float) - float(value))))


def eta_bin_indices(eta_values, eta_bins):
    eta_values = np.asarray(eta_values, dtype=float)
    eta_values = np.clip(eta_values, eta_bins[0], eta_bins[-1])
    indices = np.searchsorted(eta_bins, eta_values, side="left") - 1
    return np.clip(indices, 0, len(eta_bins) - 2)


def project_peaks_to_delta_eta(curves_df, delta_values, eta_bins):
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    mean_map = np.full((len(delta_values), len(eta_centers)), np.nan, dtype=float)
    samples_map = np.zeros_like(mean_map, dtype=int)

    eta_indices = eta_bin_indices(curves_df["eta"].to_numpy(dtype=float), eta_bins)
    delta_indices = np.array([
        nearest_index(value, delta_values)
        for value in curves_df["delta"].to_numpy(dtype=float)
    ])
    n_peaks = curves_df["n_peaks"].to_numpy(dtype=float)

    for i_delta in range(len(delta_values)):
        delta_mask = delta_indices == i_delta
        for i_eta in range(len(eta_centers)):
            mask = delta_mask & (eta_indices == i_eta)
            if not np.any(mask):
                continue
            values = n_peaks[mask]
            samples_map[i_delta, i_eta] = len(values)
            mean_map[i_delta, i_eta] = np.mean(values)

    return eta_centers, mean_map, samples_map


def bin_delta_in_alpha_beta_plane(curves_df, delta_values, delta_value, log_edges):
    delta_index = nearest_index(delta_value, delta_values)
    real_delta = float(delta_values[delta_index])

    point_delta_index = np.array([
        nearest_index(value, delta_values)
        for value in curves_df["delta"].to_numpy(dtype=float)
    ])
    mask = point_delta_index == delta_index

    alpha = curves_df.loc[mask, "alpha"].to_numpy(dtype=float)
    beta = curves_df.loc[mask, "beta"].to_numpy(dtype=float)
    peaks = curves_df.loc[mask, "n_peaks"].to_numpy(dtype=float)

    min_z, max_z = 10**log_edges[0], 10**log_edges[-1]
    valid = (
        np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(peaks)
        & (alpha >= min_z) & (alpha <= max_z)
        & (beta >= min_z) & (beta <= max_z)
    )

    n_bins = len(log_edges) - 1
    count_map = np.zeros((n_bins, n_bins), dtype=int)
    max_map = np.full((n_bins, n_bins), -np.inf, dtype=float)

    if not np.any(valid):
        return real_delta, np.full((n_bins, n_bins), np.nan), count_map

    alpha, beta, peaks = alpha[valid], beta[valid], peaks[valid]
    log_alpha, log_beta = np.log10(alpha), np.log10(beta)
    i_alpha = np.clip(np.searchsorted(log_edges, log_alpha, side="right") - 1, 0, n_bins - 1)
    i_beta = np.clip(np.searchsorted(log_edges, log_beta, side="right") - 1, 0, n_bins - 1)

    np.add.at(count_map, (i_alpha, i_beta), 1)
    np.maximum.at(max_map, (i_alpha, i_beta), peaks)

    return real_delta, max_map, count_map


def boundary_segments(mask):
    n_alpha, n_beta = mask.shape
    segments = []
    for i_alpha in range(n_alpha):
        for i_beta in range(n_beta):
            if not mask[i_alpha, i_beta]:
                continue
            xl, xr = i_alpha - 0.5, i_alpha + 0.5
            yb, yt = i_beta - 0.5, i_beta + 0.5
            if i_alpha == 0 or not mask[i_alpha - 1, i_beta]:
                segments.append(((xl, yb), (xl, yt)))
            if i_alpha == n_alpha - 1 or not mask[i_alpha + 1, i_beta]:
                segments.append(((xr, yb), (xr, yt)))
            if i_beta == 0 or not mask[i_alpha, i_beta - 1]:
                segments.append(((xl, yb), (xr, yb)))
            if i_beta == n_beta - 1 or not mask[i_alpha, i_beta + 1]:
                segments.append(((xl, yt), (xr, yt)))
    return segments


def build_transitions(curves_df):
    rows = []
    for _, row in curves_df.iterrows():
        x_peaks = row["x_peaks"]
        for i in range(max(0, len(x_peaks) - 1)):
            rows.append({
                "curve_key": row["curve_key"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "eta": row["eta"],
                "delta": row["delta"],
                "xM_n": x_peaks[i],
                "xM_np1": x_peaks[i + 1],
            })
    import pandas as pd
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        return {
            "N": 0, "r": np.nan, "R2": np.nan,
            "RMSE": np.nan, "MAE": np.nan, "MAPE_percent": np.nan,
        }

    residuals = y_true - y_pred
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else np.nan
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res / ss_tot if not np.isclose(ss_tot, 0.0) else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    nonzero = np.abs(y_true) > 1e-15
    mape = (
        float(100.0 * np.mean(np.abs(residuals[nonzero] / y_true[nonzero])))
        if np.any(nonzero)
        else np.nan
    )
    return {
        "N": int(len(y_true)),
        "r": r,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
    }
