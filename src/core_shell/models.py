import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .analysis import build_transitions, regression_metrics


def split_curves(curves_df, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(curves_df))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(indices) * test_fraction)))
    return curves_df.iloc[indices[n_test:]].copy(), curves_df.iloc[indices[:n_test]].copy()


def fit_existence_classifier(curves_df, test_fraction=0.2, seed=42):
    df = curves_df.copy()
    df["log_alpha"] = np.log(df["alpha"].to_numpy(dtype=float))
    df["log_beta"] = np.log(df["beta"].to_numpy(dtype=float))
    df["has_resonance"] = (df["n_peaks"] > 0).astype(int)
    train_df, test_df = split_curves(df, test_fraction, seed)

    features = ["log_alpha", "log_beta", "delta"]
    x_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["has_resonance"].to_numpy(dtype=int)
    x_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df["has_resonance"].to_numpy(dtype=int)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(x_train, y_train)

    def evaluate(x, y):
        probability = model.predict_proba(x)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = {
            "N": int(len(y)),
            "Positive_rate": float(np.mean(y)),
            "Accuracy": float(accuracy_score(y, prediction)),
            "Precision": float(precision_score(y, prediction, zero_division=0)),
            "Recall": float(recall_score(y, prediction, zero_division=0)),
            "F1": float(f1_score(y, prediction, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
        }
        return metrics, probability, prediction

    train_metrics, train_probability, train_prediction = evaluate(x_train, y_train)
    test_metrics, test_probability, test_prediction = evaluate(x_test, y_test)

    train_df["resonance_probability"] = train_probability
    train_df["predicted_resonance"] = train_prediction
    test_df["resonance_probability"] = test_probability
    test_df["predicted_resonance"] = test_prediction

    scaler = model.named_steps["scale"]
    logistic = model.named_steps["clf"]
    scaled = logistic.coef_[0].astype(float)
    coefficients = scaled / scaler.scale_
    intercept = float(logistic.intercept_[0] - np.sum(scaled * scaler.mean_ / scaler.scale_))
    effective_coefficients = np.concatenate([[intercept], coefficients])

    return {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "formula_coefficients": effective_coefficients,
    }


def sphere_reference(alpha, regime_name, mix_factor=0.5):
    alpha = np.asarray(alpha, dtype=float)
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha must be finite and positive.")

    if regime_name == "velocity":
        return np.sqrt(3.0) * alpha

    if regime_name == "mixed":
        m = alpha ** (mix_factor - 1.0)
        m_t = 1.0 / alpha
        return np.sqrt(3.0 / (m * m_t))

    raise ValueError("Anchored recurrence is defined here only for velocity and mixed.")


def recurrence_design_matrix(alpha, eta, delta, x_n, regime_name, mix_factor=0.5):
    alpha = np.asarray(alpha, dtype=float)
    eta = np.asarray(eta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    x_n = np.asarray(x_n, dtype=float)
    x_ref = sphere_reference(alpha, regime_name, mix_factor=mix_factor)

    intercept_part = np.column_stack([
        x_ref,
        eta * x_ref,
        delta * x_ref,
        eta**2 * x_ref,
        eta * delta * x_ref,
        delta**2 * x_ref,
    ])
    slope_part = np.column_stack([
        x_n,
        eta * x_n,
        delta * x_n,
        eta**2 * x_n,
        eta * delta * x_n,
        delta**2 * x_n,
    ])
    return np.column_stack([intercept_part, slope_part])


RECURRENCE_COEFFICIENT_NAMES = [
    "a0", "a_eta", "a_delta", "a_eta2", "a_eta_delta", "a_delta2",
    "b0", "b_eta", "b_delta", "b_eta2", "b_eta_delta", "b_delta2",
]


def fit_anchored_recurrence(
    curves_df,
    regime_name,
    mix_factor=0.5,
    test_fraction=0.2,
    seed=42,
):
    transitions = build_transitions(curves_df)
    if len(transitions) < 20:
        raise RuntimeError(f"Only {len(transitions)} transitions are available.")

    rng = np.random.default_rng(seed)
    keys = np.array(transitions["curve_key"].unique().tolist(), dtype=object)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_fraction)))
    test_keys = set(keys[:n_test])
    train_keys = set(keys[n_test:])

    train_df = transitions[transitions["curve_key"].isin(train_keys)].copy()
    test_df = transitions[transitions["curve_key"].isin(test_keys)].copy()

    def matrix(df):
        return recurrence_design_matrix(
            df["alpha"].to_numpy(dtype=float),
            df["eta"].to_numpy(dtype=float),
            df["delta"].to_numpy(dtype=float),
            df["xM_n"].to_numpy(dtype=float),
            regime_name=regime_name,
            mix_factor=mix_factor,
        )

    x_train = matrix(train_df)
    y_train = train_df["xM_np1"].to_numpy(dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        x_train, y_train, rcond=None
    )

    train_prediction = matrix(train_df) @ coefficients
    test_prediction = matrix(test_df) @ coefficients
    train_df["xM_np1_pred"] = train_prediction
    test_df["xM_np1_pred"] = test_prediction

    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) > 1 and singular_values[-1] > 0.0
        else np.nan
    )

    return {
        "transitions_df": transitions,
        "train_df": train_df,
        "test_df": test_df,
        "coefficients": coefficients,
        "coefficient_names": RECURRENCE_COEFFICIENT_NAMES,
        "rank": int(rank),
        "condition_number": condition_number,
        "train_metrics": regression_metrics(
            train_df["xM_np1"], train_prediction
        ),
        "test_metrics": regression_metrics(
            test_df["xM_np1"], test_prediction
        ),
    }
