# ============================================================
# RELAÇÕES DE EXISTÊNCIA E DE RECORRÊNCIA DAS RESSONÂNCIAS (CORE-SHELL)
# Ajusta e avalia as duas relações empíricas do artigo, por regime de contraste.
#
# Os checkpoints do gerador são achados pelo nome dentro de PASTA (e subpastas):
#   *mix0.00*.npz -> contraste na velocidade (f = 0)
#   *mix0.50*.npz -> contraste misto        (f = 0,5)
#   *mix1.00*.npz -> contraste na densidade (f = 1)
#
# 1. existência (Eq. 12 do artigo, Tabela I), nos três regimes:
#      P_res = 1/(1 + exp(-g)),
#      g = c0 + c1 ln a + c2 ln b + c3 d + c4 (ln a)^2 + c5 ln a ln b + c6 d ln a + c7 (ln b)^2 + c8 d ln b + c9 d^2
# 2. recorrência ancorada (Eq. 13 do artigo, Tabela II), velocidade e misto:
#      x_{n+1} = x_ref A(eta,d) + B(eta,d) x_n,  x_ref = sqrt(3/(m m_t)) = sqrt(3 a^(2-f))
#
# DADOS
#   ressonâncias: ressonancias_malha (máximos de W_T, proeminência >= 0,1 em log10), 1e-4 <= x_M <= 1;
#   existe ressonância quando a curva tem pelo menos um pico.
#
# TREINO / TESTE: separação por curva inteira, 20 % de teste (semente 42).
#
# SAÍDAS: PASTA/relacoes_existencia_recorrencia_final/
#   <regime>/  tabelas de métricas, gráficos e fórmula da existência (formula_existencia.txt)
#   tabela_coeficientes_existencia, tabela_coeficientes_recorrencia_ancorada (Tabelas I e II; PNG, PDF, CSV e .tex)
# ============================================================

import os
import re
import sys
import glob
import shutil
import subprocess
import importlib.util

# ============================================================
# PASTA (a única coisa a ajustar)
# ============================================================
PASTA = next((a for a in sys.argv[1:] if os.path.isdir(a)), os.environ.get("CORESHELL_DADOS", "/content/drive/MyDrive/coreshell_final/resultados_mix"))   # no PC: python script.py <pasta_com_os_checkpoints>

if PASTA.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

# dependências: scikit-learn e, no Colab, LaTeX para os rótulos
if importlib.util.find_spec("sklearn") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
if shutil.which("latex") is None and os.path.exists("/content"):
    print("Instalando LaTeX (só na primeira vez, alguns minutos)...")
    subprocess.run("apt-get update -qq && apt-get install -y -qq texlive texlive-latex-extra "
                   "texlive-fonts-recommended dvipng cm-super", shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.special import spherical_jn, spherical_yn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

USA_LATEX = shutil.which("latex") is not None
plt.rcParams.update({"text.usetex": USA_LATEX, "font.family": "sans-serif"})
VIRG = "{,}" if USA_LATEX else ","     # vírgula decimal ("{,}" evita o espaço extra do LaTeX)
PCT = r"\%" if USA_LATEX else "%"

OUTPUT_ROOT = os.path.join(PASTA, "relacoes_existencia_recorrencia_final")
DIR_REL = OUTPUT_ROOT
os.makedirs(DIR_REL, exist_ok=True)

X_MIN = 1e-4
X_MAX = 1.0
TEST_FRACTION = 0.20
RANDOM_SEED = 42
SEMENTE_PHI = 2026
N_BOOT = 300
LMAX_FAMILIAS = 10

CASES = {
    "velocity": {"f": 0.00, "recorrencia": True, "nome": "Contraste na velocidade"},
    "mixed": {"f": 0.50, "recorrencia": True, "nome": "Contraste misto"},
    "density": {"f": 1.00, "recorrencia": False, "nome": "Contraste na densidade"},
}


# ============================================================
# CHECKPOINTS PELO NOME
# ============================================================
def achar_checkpoints(pasta):
    """{f: arquivo} para todo *mixX.XX*.npz dentro da pasta (e subpastas)."""
    achados = {}
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "*.npz"), recursive=True)):
        nome = os.path.basename(arq)
        if "_dados" in nome or "aleatorio" in nome or not nome.startswith("checkpoint"):
            continue
        m = re.search(r"mix([0-9]+\.[0-9]+)", nome)
        if m:
            f0 = round(float(m.group(1)), 2)
            achados.setdefault(f0, []).append(arq)
    # se houver mais de um arquivo pro mesmo f, prefere o "reparado" e depois o nome mais curto
    return {f0: sorted(v, key=lambda a: ("reparado" not in a, len(a), a))[0] for f0, v in achados.items()}


GRADES = achar_checkpoints(PASTA)
if not GRADES:
    raise FileNotFoundError(f"Nenhum checkpoint *mixX.XX*.npz em {PASTA}")
print("Checkpoints encontrados:")
for f0, arq in sorted(GRADES.items()):
    print(f"  f = {f0:.2f}: {os.path.relpath(arq, PASTA)}")

for regime, caso in CASES.items():
    caso["checkpoint"] = GRADES.get(round(caso["f"], 2))
    if caso["checkpoint"] is None:
        print(f"AVISO: sem checkpoint para {caso['nome']} (f = {caso['f']:.2f}); esse regime fica de fora.")


# ############################################################################
# FUNÇÕES: LEITURA, TRANSIÇÕES E AJUSTES
# ############################################################################
# ============================================================
# FUNÇÕES DE APOIO
# ============================================================
def eta_from_alpha_beta(alpha, beta):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return (alpha - beta) / (alpha + beta)


def sphere_reference_velocity(alpha, mix_factor):
    """x_ref = sqrt(3/(m*mt)) da esfera homogênea (sem casca).

    m  = c_M/c_A = alpha**(f - 1)
    mt = Z_M/Z_A = 1/alpha
    =>  x_ref = sqrt(3 * alpha**(2 - f))   (f = 0: sqrt(3)*alpha, como no caso velocity)
    """
    alpha = np.asarray(alpha, dtype=float)
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("Alpha precisa ser finito e positivo.")
    m = alpha ** (mix_factor - 1.0)
    mt = 1.0 / alpha
    return np.sqrt(3.0 / (m * mt))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        return {"N": 0, "r": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE_percent": np.nan}

    residuals = y_true - y_pred
    r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else np.nan

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if not np.isclose(ss_tot, 0.0) else np.nan

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    positive_y = np.abs(y_true) > 1e-15
    mape = (
        float(100.0 * np.mean(np.abs(residuals[positive_y] / y_true[positive_y])))
        if np.any(positive_y) else np.nan
    )

    return {"N": int(len(y_true)), "r": r_pearson, "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE_percent": mape}


# ============================================================
# LEITURA DO CHECKPOINT (formato coreshell_final) + TRANSIÇÕES
# ============================================================
def curva_valida(item):
    if item.get("status") != "ok":
        return False
    reparo = item.get("reparo_velocity")
    return not (isinstance(reparo, dict) and not reparo.get("ok", True))


def load_curves_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado:\n{checkpoint_path}")

    with np.load(checkpoint_path, allow_pickle=True) as data:
        curves = data["curves"].item()

    rows = []
    for chave, item in curves.items():
        alpha = float(item.get("alpha", np.nan))
        beta = float(item.get("beta", np.nan))
        delta = float(item.get("delta", np.nan))

        if not (
            np.isfinite(alpha) and alpha > 0.0
            and np.isfinite(beta) and beta > 0.0
            and np.isfinite(delta)
            and curva_valida(item)
        ):
            continue

        peak_tuples = []
        for pico in item.get("ressonancias_malha", []) or []:
            xM = float(pico.get("xM_peak", np.nan))
            if not (np.isfinite(xM) and X_MIN <= xM <= X_MAX):
                continue
            fwhm = float(pico.get("FWHM_energy", np.nan))
            q = float(pico.get("Q_energy", np.nan))
            peak_tuples.append((xM, fwhm, q))

        peak_tuples.sort(key=lambda p: p[0])
        x_peaks = [p[0] for p in peak_tuples]

        rows.append({
            "curve_key": chave, "alpha": alpha, "beta": beta,
            "eta": float(eta_from_alpha_beta(alpha, beta)), "delta": delta,
            "n_peaks": len(x_peaks), "x_peaks": x_peaks, "peaks": peak_tuples,
        })

    return pd.DataFrame(rows), len(curves)


def build_transitions(curves_df):
    rows = []
    for _, row in curves_df.iterrows():
        x_peaks = row["x_peaks"]
        if len(x_peaks) < 2:
            continue
        for index in range(len(x_peaks) - 1):
            rows.append({
                "curve_key": row["curve_key"], "alpha": row["alpha"],
                "beta": row["beta"], "eta": row["eta"], "delta": row["delta"],
                "xM_n": x_peaks[index], "xM_np1": x_peaks[index + 1],
            })
    return pd.DataFrame(rows)


def split_train_test_by_curve(transitions_df, test_fraction, seed):
    rng = np.random.default_rng(seed)
    unique_keys = np.array(transitions_df["curve_key"].unique().tolist(), dtype=object)
    rng.shuffle(unique_keys)

    n_test = max(1, int(round(len(unique_keys) * test_fraction)))
    test_keys = set(unique_keys[:n_test])
    train_keys = set(unique_keys[n_test:])

    train_df = transitions_df.loc[transitions_df["curve_key"].isin(train_keys)].copy()
    test_df = transitions_df.loc[transitions_df["curve_key"].isin(test_keys)].copy()
    return train_df, test_df, len(train_keys), len(test_keys)


def split_curves_train_test(curves_df, test_fraction, seed):
    rng = np.random.default_rng(seed)
    index = np.arange(len(curves_df))
    rng.shuffle(index)
    n_test = max(1, int(round(len(index) * test_fraction)))
    return curves_df.iloc[index[n_test:]].copy(), curves_df.iloc[index[:n_test]].copy()


# ============================================================
# RECORRÊNCIA ANCORADA: x_{n+1} = x_ref A(eta,delta) + B(eta,delta) x_n
#   x_ref = sqrt(3 alpha^(2-f))  (esfera homogênea, limite de Minnaert)
# ============================================================
COEF_NAMES_ANCORADA = [
    "a0", "a_eta", "a_delta", "a_eta2", "a_eta_delta", "a_delta2",
    "b0", "b_eta", "b_delta", "b_eta2", "b_eta_delta", "b_delta2",
]


def build_design_matrix_ancorada(alpha, eta, delta, x_n, mix_factor):
    alpha = np.asarray(alpha, dtype=float)
    eta = np.asarray(eta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    x_n = np.asarray(x_n, dtype=float)

    x_ref = sphere_reference_velocity(alpha, mix_factor)

    XA = np.column_stack([x_ref, eta * x_ref, delta * x_ref, eta**2 * x_ref, eta * delta * x_ref, delta**2 * x_ref])
    XB = np.column_stack([x_n, eta * x_n, delta * x_n, eta**2 * x_n, eta * delta * x_n, delta**2 * x_n])
    return np.column_stack([XA, XB])


def ols_standard_errors(X, y, coefficients):
    residuals = y - X @ coefficients
    n, p = X.shape
    s2 = float(residuals @ residuals) / max(n - p, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def fit_recurrence(curves_df, label, test_fraction, seed, tipo, mix_factor):
    """Recorrência ancorada, Eq. 13 do artigo (tipo = "ancorada")."""
    anchored = True
    transitions_df = build_transitions(curves_df)

    if len(transitions_df) < 20:
        raise RuntimeError(f"[{label}] poucas transições ({len(transitions_df)}) pra ajustar/testar.")

    train_df, test_df, n_train_curves, n_test_curves = split_train_test_by_curve(
        transitions_df, test_fraction, seed
    )

    def build_X(df):
        return build_design_matrix_ancorada(
            df["alpha"].to_numpy(dtype=float), df["eta"].to_numpy(dtype=float),
            df["delta"].to_numpy(dtype=float), df["xM_n"].to_numpy(dtype=float), mix_factor,
        )

    X_train = build_X(train_df)
    y_train = train_df["xM_np1"].to_numpy(dtype=float)
    coefficients, _, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)

    def predict(df):
        return build_X(df) @ coefficients

    coef_names = COEF_NAMES_ANCORADA

    standard_errors = ols_standard_errors(X_train, y_train, coefficients)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["xM_np1_pred"] = predict(train_df)
    test_df["xM_np1_pred"] = predict(test_df)

    metrics_train = calculate_metrics(train_df["xM_np1"], train_df["xM_np1_pred"])
    metrics_test = calculate_metrics(test_df["xM_np1"], test_df["xM_np1_pred"])
    condition_number = float(sv[0] / sv[-1]) if len(sv) > 1 and sv[-1] > 0.0 else np.nan

    return {
        "label": label, "anchored": anchored, "tipo": tipo,
        "transitions_df": transitions_df,
        "train_df": train_df, "test_df": test_df,
        "coefficients": coefficients, "standard_errors": standard_errors, "coef_names": coef_names,
        "rank": int(rank), "condition_number": condition_number,
        "n_curves_total": int(len(curves_df)),
        "n_curves_with_peaks": int((curves_df["n_peaks"] > 0).sum()),
        "total_peaks": int(curves_df["n_peaks"].sum()),
        "n_train_curves": n_train_curves, "n_test_curves": n_test_curves,
        "metrics_train": metrics_train, "metrics_test": metrics_test,
    }


# ============================================================
# CLASSIFICADOR DE EXISTÊNCIA DE RESSONÂNCIA (Eq. 12 do artigo)
# ============================================================
CLF_METRIC_ORDER = ["N", "Positive_rate", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

CLF_TERM_NAMES = [
    "1", "ln_alpha", "ln_beta", "delta", "ln_alpha2",
    "ln_alpha_ln_beta", "ln_alpha_delta", "ln_beta2",
    "ln_beta_delta", "delta2",
]


def classifier_terms(alpha, beta, delta):
    ln_alpha = np.log(np.asarray(alpha, dtype=float))
    ln_beta = np.log(np.asarray(beta, dtype=float))
    delta = np.asarray(delta, dtype=float)
    return np.array([
        np.ones_like(ln_alpha),
        ln_alpha,
        ln_beta,
        delta,
        ln_alpha**2,
        ln_alpha * ln_beta,
        ln_alpha * delta,
        ln_beta**2,
        ln_beta * delta,
        delta**2,
    ])


def extract_classifier_formula(model):
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["clf"]

    scaled_coefficients = logistic.coef_[0].astype(float)
    effective_coefficients = scaled_coefficients / scaler.scale_
    effective_intercept = float(
        logistic.intercept_[0]
        - np.sum(scaled_coefficients * scaler.mean_ / scaler.scale_)
    )

    return np.concatenate([[effective_intercept], effective_coefficients])


def classifier_score(alpha, beta, delta, coefficients):
    return np.tensordot(np.asarray(coefficients, dtype=float), classifier_terms(alpha, beta, delta), axes=(0, 0))


def classifier_probability(alpha, beta, delta, coefficients):
    score = classifier_score(alpha, beta, delta, coefficients)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -700.0, 700.0)))


def logistic_standard_errors(train_df, coefficients):
    X = classifier_terms(train_df["alpha"], train_df["beta"], train_df["delta"]).T
    p = classifier_probability(train_df["alpha"], train_df["beta"], train_df["delta"], coefficients)
    w = p * (1.0 - p)
    fisher = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(fisher)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def fit_existence_classifier(curves_df, test_fraction, seed):
    df = curves_df.copy()
    df["log_alpha"] = np.log(df["alpha"].to_numpy(dtype=float))
    df["log_beta"] = np.log(df["beta"].to_numpy(dtype=float))
    df["has_resonance"] = (df["n_peaks"] > 0).astype(int)

    train_df, test_df = split_curves_train_test(df, test_fraction, seed)

    feature_cols = ["log_alpha", "log_beta", "delta"]
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["has_resonance"].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["has_resonance"].to_numpy(dtype=int)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)

    def evaluate(X, y):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "N": int(len(y)), "Positive_rate": float(np.mean(y)),
            "Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "F1": float(f1_score(y, pred, zero_division=0)),
        }
        metrics["ROC_AUC"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else np.nan
        return metrics, proba, pred

    metrics_train, proba_train, pred_train = evaluate(X_train, y_train)
    metrics_test, proba_test, pred_test = evaluate(X_test, y_test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["probability_resonance"] = proba_train
    train_df["predicted_resonance"] = pred_train
    test_df["probability_resonance"] = proba_test
    test_df["predicted_resonance"] = pred_test

    effective_coefficients = extract_classifier_formula(model)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "formula_coefficients": effective_coefficients,
        "standard_errors": logistic_standard_errors(train_df, effective_coefficients),
        "formula_term_names": CLF_TERM_NAMES,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }


# ############################################################################
# FUNÇÕES: TABELAS E GRÁFICOS
# ############################################################################
# ============================================================
# FORMATAÇÃO (vírgula decimal e porcentagem conforme LaTeX disponível)
# ============================================================
def format_decimal_comma(value, decimals=None):
    value = float(value)
    text = f"{value:g}" if decimals is None else f"{value:.{decimals}f}"
    return text.replace(".", VIRG)


def latex_sci(value, sig=3):
    if not np.isfinite(value):
        return "--"
    value = float(value)
    if np.isclose(value, 0.0):
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    decimals = max(sig - 1, 0)
    mantissa_text = format_decimal_comma(mantissa, decimals)
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def fmt_int(value):
    return "--" if not np.isfinite(value) else f"{int(round(value))}"


def fmt_r(value):
    return "--" if not np.isfinite(value) else format_decimal_comma(value, 4)


def fmt_mape(value):
    return "--" if not np.isfinite(value) else f"{format_decimal_comma(value, 2)}{PCT}"


def fmt_pct01(value):
    return "--" if not np.isfinite(value) else f"{format_decimal_comma(100.0 * float(value), 1)}{PCT}"


def fmt_pm(value, error):
    """Valor ± incerteza, com a incerteza em 2 algarismos significativos (como nas Tabelas I e II do artigo)."""
    if not np.isfinite(value):
        return "--"
    if not (np.isfinite(error) and error > 0.0):
        return f"${format_decimal_comma(value, 4)}$"
    decimals = max(0, -int(np.floor(np.log10(error))) + 1)
    valor = f"{value:+.{decimals}f}".replace(".", VIRG)
    incerteza = f"{error:.{decimals}f}".replace(".", VIRG)
    return rf"${valor}\pm{incerteza}$"


FORMATTERS_BY_METRIC = {"N": fmt_int, "r": fmt_r, "R2": fmt_r, "RMSE": latex_sci, "MAE": latex_sci, "MAPE_percent": fmt_mape}

CLF_FORMATTERS = {
    "N": fmt_int, "Positive_rate": fmt_pct01, "Accuracy": fmt_r, "Precision": fmt_r,
    "Recall": fmt_r, "F1": fmt_r, "ROC_AUC": fmt_r,
}

METRIC_ORDER = ["N", "r", "R2", "RMSE", "MAE", "MAPE_percent"]

row_labels_recorrencia = ["N\n(transições)", "Pearson\n" + r"$r$", r"$R^2$", "RMSE", "MAE", "MAPE\n(" + PCT + ")"]

row_labels_classifier = [
    "N\n(curvas)", "Taxa de\npositivos", "Acurácia", "Precisão", "Sensibilidade", "F1", "ROC AUC",
]


def build_table_data_2col(metrics_train, metrics_test, metric_order, formatters):
    return [[formatters[m](metrics_train.get(m, np.nan)), formatters[m](metrics_test.get(m, np.nan))] for m in metric_order]


# ============================================================
# TABELAS COMPACTAS
# ============================================================
TABLE_FIGSIZE = (5.4, 4.5)
TABLE_DPI = 600


def save_compact_table(filename, table_data, row_labels, columns=("Treino", "Teste"),
                       figsize=TABLE_FIGSIZE, first_col_width=0.42, header_label="Parâmetro",
                       fontsize=19):
    with plt.rc_context({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 30,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.axis("off")
        ax.set_facecolor("white")

        n_cols = len(columns)
        cell_text = [[row_labels[i], *table_data[i]] for i in range(len(row_labels))]

        table = ax.table(
            cellText=cell_text, colLabels=[header_label, *columns], cellLoc="center",
            colWidths=[first_col_width] + [(1.0 - first_col_width) / n_cols] * n_cols,
            bbox=[0.02, 0.02, 0.96, 0.96],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)

        n_rows = len(row_labels)
        for (row, col), cell in table.get_celld().items():
            cell.visible_edges = ""
            cell.set_edgecolor("black")
            cell.set_linewidth(0.0)
            cell.set_facecolor("white")
            cell.PAD = 0.01
            text = cell.get_text()
            text.set_color("black")
            text.set_fontweight("normal")
            text.set_va("center")
            text.set_linespacing(0.90)
            text.set_ha("left" if col == 0 else "center")
            text.set_fontsize(fontsize)

        for col in range(n_cols + 1):
            cell = table[(0, col)]
            cell.visible_edges = "TB"
            cell.set_linewidth(1.5)
            cell.get_text().set_fontsize(fontsize + 1)
        table[(0, 0)].get_text().set_fontweight("bold")

        for col in range(n_cols + 1):
            cell = table[(n_rows, col)]
            cell.visible_edges = "B"
            cell.set_linewidth(1.5)

        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        fig.savefig(f"{filename}.png", dpi=TABLE_DPI, bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        fig.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        fig.savefig(f"{filename}.svg", bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        plt.show()
        plt.close(fig)


# ============================================================
# GRÁFICOS
# ============================================================
def plot_single(result, output_dir, filename_slug):
    with plt.rc_context({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 16,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        test_df = result["test_df"]

        ax.scatter(test_df["xM_n"], test_df["xM_np1"], s=36, facecolors="#D9D9D9",
                   edgecolors="#8A8A8A", linewidths=0.45, alpha=0.75, zorder=1, label="Valor observado")
        ax.scatter(test_df["xM_n"], test_df["xM_np1_pred"], s=28, facecolors="#0f223d",
                   edgecolors="black", linewidths=0.45, alpha=0.9, zorder=3, label="Valor previsto")

        lims = [min(test_df["xM_n"].min(), test_df["xM_np1"].min()), max(test_df["xM_n"].max(), test_df["xM_np1"].max())]
        ax.plot(lims, lims, "--", color="0.35", linewidth=1.15, zorder=0, label="1:1")

        m = result["metrics_test"]
        text = (
            f"$N={m['N']}$\n"
            f"$r={format_decimal_comma(m['r'], 4)}$\n"
            f"$R^2={format_decimal_comma(m['R2'], 4)}$\n"
            f"RMSE = {latex_sci(m['RMSE'])}\n"
            f"MAPE = {format_decimal_comma(m['MAPE_percent'], 2)}{PCT}"
        )
        ax.text(0.035, 0.965, text, transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                linespacing=1.05, bbox=dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="black", linewidth=1.0))

        ax.set_xlabel(r"$x_M(n)$", fontsize=18, labelpad=7)
        ax.set_ylabel(r"$x_M(n+1)$", fontsize=18, labelpad=7)
        formatter_abnt = FuncFormatter(lambda value, pos: format_decimal_comma(value, 2))
        ax.xaxis.set_major_formatter(formatter_abnt)
        ax.yaxis.set_major_formatter(formatter_abnt)
        ax.grid(False)
        ax.tick_params(direction="in", length=5.0, width=1.15, labelsize=15, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.15)
            spine.set_edgecolor("black")

        legend = ax.legend(loc="lower right", fontsize=10.5, frameon=True, fancybox=False,
                           framealpha=1.0, handlelength=2.0, borderpad=0.35,
                           labelspacing=0.28, handletextpad=0.45, borderaxespad=0.55)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor("white")

        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
        base = os.path.join(output_dir, f"grafico_{filename_slug}")
        fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        plt.show()
        plt.close(fig)

    return base


def plot_existence_classifier(clf_result, output_dir, filename_slug):
    with plt.rc_context({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 16,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
    }):
        test_df = clf_result["test_df"].copy()
        y_true = test_df["has_resonance"].to_numpy(dtype=float)
        probabilities = test_df["probability_resonance"].to_numpy(dtype=float)

        rng = np.random.default_rng(RANDOM_SEED)
        x_jitter = y_true + rng.uniform(-0.045, 0.045, size=len(y_true))

        fig, ax = plt.subplots(figsize=(6.5, 5.0))

        no_resonance = y_true == 0
        resonance = y_true == 1

        ax.scatter(x_jitter[no_resonance], probabilities[no_resonance], s=28,
                   facecolors="#D9D9D9", edgecolors="#8A8A8A", linewidths=0.45,
                   alpha=0.75, zorder=2, label="Sem ressonância")
        ax.scatter(x_jitter[resonance], probabilities[resonance], s=28,
                   facecolors="#0f223d", edgecolors="black", linewidths=0.45,
                   alpha=0.85, zorder=3, label="Com ressonância")

        ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.35", linewidth=1.15, zorder=0, label="1:1")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1.15, zorder=1,
                   label=r"Limiar $P_{\mathrm{res}}=0{,}5$")

        metrics = clf_result["metrics_test"]
        text = (
            f"$N={metrics['N']}$\n"
            f"Acurácia = {format_decimal_comma(metrics['Accuracy'], 3)}\n"
            f"Precisão = {format_decimal_comma(metrics['Precision'], 3)}\n"
            f"Sensibilidade = {format_decimal_comma(metrics['Recall'], 3)}\n"
            f"$F_1$ = {format_decimal_comma(metrics['F1'], 3)}\n"
            f"ROC AUC = {format_decimal_comma(metrics['ROC_AUC'], 3)}"
        )
        ax.text(0.035, 0.965, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, linespacing=1.05,
                bbox=dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="black", linewidth=1.0))

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "1"], fontsize=15)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0", f"0{VIRG}25", f"0{VIRG}50", f"0{VIRG}75", f"1{VIRG}00"], fontsize=15)
        ax.set_xlabel("Existência observada de ressonância", fontsize=18, labelpad=7)
        ax.set_ylabel(r"Probabilidade prevista $P_{\mathrm{res}}$", fontsize=18, labelpad=7)
        ax.grid(False)
        ax.tick_params(direction="in", length=5.0, width=1.15, labelsize=15, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.15)
            spine.set_edgecolor("black")
        legend = ax.legend(loc="lower right", fontsize=10.5, frameon=True, fancybox=False,
                           framealpha=1.0, handlelength=2.0, borderpad=0.35,
                           labelspacing=0.28, handletextpad=0.45, borderaxespad=0.55)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor("white")

        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)

        base = os.path.join(output_dir, f"grafico_{filename_slug}")
        fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03, facecolor="white", edgecolor="white")
        plt.show()
        plt.close(fig)

    return base


# ============================================================
# FÓRMULA DO CLASSIFICADOR EM TEXTO (pra copiar em código ou no LaTeX)
# ============================================================
def build_classifier_formula_text(coefficients, errors):
    c = np.asarray(coefficients, dtype=float)
    s = np.asarray(errors, dtype=float)

    symbolic = (
        "P_res(alpha,beta,delta) = 1 / (1 + exp[-g(alpha,beta,delta)])\n"
        "g = c0 + c1 ln(alpha) + c2 ln(beta) + c3 delta "
        "+ c4 [ln(alpha)]^2 + c5 ln(alpha)ln(beta) "
        "+ c6 delta ln(alpha) + c7 [ln(beta)]^2 "
        "+ c8 delta ln(beta) + c9 delta^2\n"
        "Existe ressonancia quando P_res >= 0.5, equivalentemente g >= 0.\n"
    )

    numeric = "\nCOEFICIENTES (valor, erro padrao)\n" + "".join(
        f"c{i} = {c[i]:+.12e}  +- {s[i]:.3e}\n" for i in range(len(c))
    )

    latex = (
        "\nLATEX\n"
        r"\begin{aligned}" "\n"
        r"P_{\mathrm{res}}(\alpha,\beta,\delta)"
        r"&=\frac{1}{1+\exp[-g(\alpha,\beta,\delta)]},\\" "\n"
        r"g(\alpha,\beta,\delta)"
        f"&={c[0]:+.8e}"
        rf"{c[1]:+.8e}\ln\alpha"
        rf"{c[2]:+.8e}\ln\beta"
        rf"{c[3]:+.8e}\delta"
        rf"{c[4]:+.8e}(\ln\alpha)^2"
        rf"{c[5]:+.8e}\ln\alpha\ln\beta"
        rf"{c[6]:+.8e}\delta\ln\alpha"
        rf"{c[7]:+.8e}(\ln\beta)^2"
        rf"{c[8]:+.8e}\delta\ln\beta"
        rf"{c[9]:+.8e}\delta^2."
        "\n" r"\end{aligned}" "\n"
    )

    return symbolic + numeric + latex


# ############################################################################
# AJUSTES POR REGIME (existência e recorrência ancorada)
# ############################################################################
RESULTADOS = {}

for regime, caso in CASES.items():
    if caso["checkpoint"] is None:
        continue
    pasta = os.path.join(DIR_REL, regime)
    os.makedirs(pasta, exist_ok=True)

    print("=" * 96)
    print(f"{caso['nome'].upper()} | f={caso['f']:.2f}")
    print("=" * 96)
    print(f"Checkpoint: {caso['checkpoint']}")

    curves_df, n_total = load_curves_from_checkpoint(caso["checkpoint"])
    print(f"Curvas no checkpoint: {n_total} | usadas: {len(curves_df)} | com ressonância: {int((curves_df['n_peaks'] > 0).sum())}")

    RESULTADOS[regime] = {"curves_df": curves_df}

    # ---------------- recorrência ----------------
    if caso["recorrencia"]:
        tipos = [("ancorada", "Ancorada (esfera, x_ref)", pasta)]
        for kind_slug, kind_label, pasta_saida in tipos:
            os.makedirs(pasta_saida, exist_ok=True)
            result = fit_recurrence(curves_df, kind_label, TEST_FRACTION, RANDOM_SEED, kind_slug, caso["f"])
            RESULTADOS[regime][kind_slug] = result

            print(f"\n--- Recorrência {kind_label} ---")
            print(f"Transições: {len(result['transitions_df'])} | curvas treino/teste: {result['n_train_curves']}/{result['n_test_curves']}")
            print(f"Rank: {result['rank']}/{len(result['coef_names'])} | número de condição: {result['condition_number']:.3e}")
            for name, value, error in zip(result["coef_names"], result["coefficients"], result["standard_errors"]):
                print(f"  {name:>14s} = {value:+.6e}  ± {error:.2e}")
            print(f"TREINO: {result['metrics_train']}")
            print(f"TESTE : {result['metrics_test']}")

            table_data = build_table_data_2col(result["metrics_train"], result["metrics_test"], METRIC_ORDER, FORMATTERS_BY_METRIC)
            save_compact_table(os.path.join(pasta_saida, f"tabela_recorrencia_{kind_slug}"), table_data, row_labels_recorrencia)
            plot_single(result, pasta_saida, f"recorrencia_{kind_slug}")

    # ---------------- existência ----------------
    clf = fit_existence_classifier(curves_df, TEST_FRACTION, RANDOM_SEED)
    RESULTADOS[regime]["existencia"] = clf

    print("\n--- Existência ---")
    for name, value, error in zip(clf["formula_term_names"], clf["formula_coefficients"], clf["standard_errors"]):
        print(f"  {name:>18s} = {value:+.6e}  ± {error:.2e}")
    print(f"TREINO: {clf['metrics_train']}")
    print(f"TESTE : {clf['metrics_test']}")

    with open(os.path.join(pasta, "formula_existencia.txt"), "w", encoding="utf-8") as fh:
        fh.write(build_classifier_formula_text(clf["formula_coefficients"], clf["standard_errors"]))

    clf_table_data = build_table_data_2col(clf["metrics_train"], clf["metrics_test"], CLF_METRIC_ORDER, CLF_FORMATTERS)
    save_compact_table(os.path.join(pasta, "tabela_existencia"), clf_table_data, row_labels_classifier)
    plot_existence_classifier(clf, pasta, "existencia")

    print(f"\nArquivos em: {pasta}\n")


# ############################################################################
# TABELAS DE COEFICIENTES (Tabelas I e II do artigo)
# ############################################################################
def salvar_tabela_coeficientes(base, linhas, colunas, valores, erros, figsize, first_col_width=0.16, fontsize=17):
    """linhas: rótulos das linhas; colunas: rótulos das colunas; valores/erros: matrizes (linhas x colunas)."""
    dados = [[fmt_pm(valores[i][j], erros[i][j]) for j in range(len(colunas))] for i in range(len(linhas))]
    save_compact_table(base, dados, linhas, columns=colunas, figsize=figsize,
                       first_col_width=first_col_width, header_label="Coef.", fontsize=fontsize)

    registros = []
    for i, linha in enumerate(linhas):
        for j, coluna in enumerate(colunas):
            registros.append({"coeficiente": linha.replace("$", ""), "coluna": coluna,
                              "valor": valores[i][j], "erro_padrao": erros[i][j]})
    pd.DataFrame(registros).to_csv(base + ".csv", index=False)

    with open(base + ".tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{l" + "c" * len(colunas) + "}\n\\hline\n")
        fh.write("Coef. & " + " & ".join(c.replace("\n", " ") for c in colunas) + " \\\\\n\\hline\n")
        for i, linha in enumerate(linhas):
            fh.write(linha + " & " + " & ".join(dados[i]) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")


# ---------------- existência: c0..c9 por regime (ordem da Tabela I) ----------------
ORDEM_EXISTENCIA = ["density", "velocity", "mixed"]
regimes_exist = [r for r in ORDEM_EXISTENCIA if "existencia" in RESULTADOS.get(r, {})]

salvar_tabela_coeficientes(
    os.path.join(DIR_REL, "tabela_coeficientes_existencia"),
    [f"$c_{{{i}}}$" for i in range(10)],
    [CASES[r]["nome"].replace("Contraste ", "Contraste\n") for r in regimes_exist],
    [[RESULTADOS[r]["existencia"]["formula_coefficients"][i] for r in regimes_exist] for i in range(10)],
    [[RESULTADOS[r]["existencia"]["standard_errors"][i] for r in regimes_exist] for i in range(10)],
    figsize=(7.6, 6.2),
)

# ---------------- recorrência ancorada: A e B por regime (arranjo da Tabela II) ----------------
regimes_rec = [r for r in ("velocity", "mixed") if "ancorada" in RESULTADOS.get(r, {})]


def tabela_coeficientes_ancorada(chave, base):
    colunas, valores, erros = [], [[] for _ in range(6)], [[] for _ in range(6)]
    for r in regimes_rec:
        res = RESULTADOS[r][chave]
        curto = "Velocidade" if r == "velocity" else "Misto"
        for parte, desloc in (("A", 0), ("B", 6)):
            colunas.append(f"{curto}\n${parte}$")
            for i in range(6):
                valores[i].append(res["coefficients"][desloc + i])
                erros[i].append(res["standard_errors"][desloc + i])
    salvar_tabela_coeficientes(base, [f"${i}$" for i in range(6)], colunas, valores, erros,
                               figsize=(12.5, 4.8), first_col_width=0.08, fontsize=16)


if regimes_rec:
    tabela_coeficientes_ancorada("ancorada", os.path.join(DIR_REL, "tabela_coeficientes_recorrencia_ancorada"))

print(f"Tabelas de coeficientes em: {DIR_REL}")
