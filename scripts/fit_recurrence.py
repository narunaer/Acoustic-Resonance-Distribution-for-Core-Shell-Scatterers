#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from core_shell.analysis import build_transitions, regression_metrics
from core_shell.checkpoints import load_curves_dataframe, find_repository_checkpoint, repository_figure_dir
from core_shell.models import recurrence_design_matrix
from core_shell.regimes import get_regime

TABLE_FIGSIZE = (5.4, 4.5)
TABLE_DPI = 600
COMPARE_COLUMNS = ["Treino", "Teste"]
ROW_LABELS = ["N\n(transições)", "Pearson\n" + r"$r$", r"$R^2$", "RMSE", "MAE", "MAPE\n(%)"]
METRIC_ORDER = ["N", "r", "R2", "RMSE", "MAE", "MAPE_percent"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit anchored and non-anchored recurrence models and save the article tables/figures.")
    parser.add_argument("--regime", choices=("velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=("raw", "manual_corrected", "final"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latex_sci(value, sig=3):
    if not np.isfinite(value):
        return "--"
    value = float(value)
    if np.isclose(value, 0.0):
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    decimals = max(sig - 1, 0)
    return rf"${mantissa:.{decimals}f}\times10^{{{exponent}}}$"


def fmt_int(value):
    return "--" if not np.isfinite(value) else f"{int(round(value))}"


def fmt_r(value):
    return "--" if not np.isfinite(value) else f"{float(value):.4f}"


def fmt_mape(value):
    return "--" if not np.isfinite(value) else f"{float(value):.2f}%"


FORMATTERS = {
    "N": fmt_int,
    "r": fmt_r,
    "R2": fmt_r,
    "RMSE": latex_sci,
    "MAE": latex_sci,
    "MAPE_percent": fmt_mape,
}


def build_table_data(metrics_train, metrics_test):
    return [[FORMATTERS[m](metrics_train.get(m, np.nan)), FORMATTERS[m](metrics_test.get(m, np.nan))] for m in METRIC_ORDER]


def save_compact_table(filename, table_data):
    with plt.rc_context({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 30,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=TABLE_FIGSIZE, facecolor="white")
        ax.axis("off")
        ax.set_facecolor("white")
        cell_text = [[ROW_LABELS[i], *table_data[i]] for i in range(len(ROW_LABELS))]
        table = ax.table(
            cellText=cell_text,
            colLabels=["Parameter", *COMPARE_COLUMNS],
            cellLoc="center",
            colWidths=[0.42, 0.29, 0.29],
            bbox=[0.02, 0.02, 0.96, 0.96],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(19)
        n_rows = len(ROW_LABELS)
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
            text.set_fontsize(19)
        for col in range(3):
            cell = table[(0, col)]
            cell.visible_edges = "TB"
            cell.set_linewidth(1.5)
            cell.get_text().set_fontsize(20)
        table[(0, 0)].get_text().set_fontweight("bold")
        for col in range(3):
            cell = table[(n_rows, col)]
            cell.visible_edges = "B"
            cell.set_linewidth(1.5)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        fig.savefig(f"{filename}.png", dpi=TABLE_DPI, bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        fig.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        fig.savefig(f"{filename}.svg", bbox_inches="tight", pad_inches=0.01, facecolor="white", edgecolor="white")
        plt.close(fig)


def split_by_curve(transitions, test_fraction, seed):
    rng = np.random.default_rng(seed)
    keys = np.asarray(transitions["curve_key"].unique().tolist(), dtype=object)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_fraction)))
    test_keys = set(keys[:n_test])
    train_keys = set(keys[n_test:])
    train = transitions[transitions["curve_key"].isin(train_keys)].copy()
    test = transitions[transitions["curve_key"].isin(test_keys)].copy()
    return train, test, len(train_keys), len(test_keys)


def nonanchored_matrix(df):
    eta = df["eta"].to_numpy(dtype=float)
    delta = df["delta"].to_numpy(dtype=float)
    log_xn = np.log(df["xM_n"].to_numpy(dtype=float))
    return np.column_stack([
        np.ones_like(log_xn),
        log_xn,
        eta,
        delta,
        log_xn**2,
        eta**2,
        delta**2,
        log_xn * eta,
        log_xn * delta,
        eta * delta,
    ])


def fit_model(curves, regime_name, mix_factor, test_fraction, seed, anchored):
    transitions = build_transitions(curves)
    if len(transitions) < 20:
        raise RuntimeError(f"Only {len(transitions)} transitions are available.")
    train, test, n_train_curves, n_test_curves = split_by_curve(transitions, test_fraction, seed)

    if anchored:
        def matrix(df):
            return recurrence_design_matrix(
                df["alpha"].to_numpy(dtype=float),
                df["eta"].to_numpy(dtype=float),
                df["delta"].to_numpy(dtype=float),
                df["xM_n"].to_numpy(dtype=float),
                regime_name=regime_name,
                mix_factor=mix_factor,
            )
        X_train = matrix(train)
        y_train = train["xM_np1"].to_numpy(dtype=float)
        coefficients, _, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)
        train_pred = matrix(train) @ coefficients
        test_pred = matrix(test) @ coefficients
    else:
        X_train = nonanchored_matrix(train)
        y_train = np.log(train["xM_np1"].to_numpy(dtype=float))
        coefficients, _, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)
        train_pred = np.exp(nonanchored_matrix(train) @ coefficients)
        test_pred = np.exp(nonanchored_matrix(test) @ coefficients)

    train["xM_np1_pred"] = train_pred
    test["xM_np1_pred"] = test_pred
    condition_number = float(sv[0] / sv[-1]) if len(sv) > 1 and sv[-1] > 0.0 else np.nan
    return {
        "train_df": train,
        "test_df": test,
        "metrics_train": regression_metrics(train["xM_np1"], train_pred),
        "metrics_test": regression_metrics(test["xM_np1"], test_pred),
        "coefficients": np.asarray(coefficients, dtype=float),
        "rank": int(rank),
        "condition_number": condition_number,
        "n_train_curves": n_train_curves,
        "n_test_curves": n_test_curves,
    }


def recurrence_formula_text(coefficients, anchored, regime_name, mix_factor):
    c = np.asarray(coefficients, dtype=float)
    if anchored:
        names = [
            "a0", "a_eta", "a_delta", "a_eta2", "a_eta_delta", "a_delta2",
            "b0", "b_eta", "b_delta", "b_eta2", "b_eta_delta", "b_delta2",
        ]
        lines = [
            "Recorrência ancorada",
            "",
            "x_M(n+1) = x_ref * A(eta,delta) + x_M(n) * B(eta,delta)",
            "A(eta,delta) = a0 + a_eta*eta + a_delta*delta + a_eta2*eta^2 + a_eta_delta*eta*delta + a_delta2*delta^2",
            "B(eta,delta) = b0 + b_eta*eta + b_delta*delta + b_eta2*eta^2 + b_eta_delta*eta*delta + b_delta2*delta^2",
            "",
        ]
        if regime_name == "velocity":
            lines += ["x_ref = sqrt(3) * alpha", ""]
        else:
            lines += [
                f"mix_factor = {mix_factor}",
                "m = alpha^(mix_factor - 1)",
                "m_t = 1/alpha",
                "x_ref = sqrt(3/(m*m_t))",
                "",
            ]
        lines += [f"{name} = {value:+.12e}" for name, value in zip(names, c)]
        return "\n".join(lines) + "\n"

    names = [
        "c0", "c_logx", "c_eta", "c_delta", "c_logx2",
        "c_eta2", "c_delta2", "c_logx_eta", "c_logx_delta", "c_eta_delta",
    ]
    lines = [
        "Recorrência não ancorada",
        "",
        "ln[x_M(n+1)] = c0 + c_logx*ln[x_M(n)] + c_eta*eta + c_delta*delta",
        "               + c_logx2*ln[x_M(n)]^2 + c_eta2*eta^2 + c_delta2*delta^2",
        "               + c_logx_eta*ln[x_M(n)]*eta + c_logx_delta*ln[x_M(n)]*delta + c_eta_delta*eta*delta",
        "",
        "x_M(n+1) = exp(expressao_acima)",
        "",
    ]
    lines += [f"{name} = {value:+.12e}" for name, value in zip(names, c)]
    return "\n".join(lines) + "\n"


def save_recurrence_formula(filename, coefficients, anchored, regime_name, mix_factor):
    text = recurrence_formula_text(coefficients, anchored, regime_name, mix_factor)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    return text


def corrected_run(checkpoint, output_dir):
    text = f"{checkpoint} {output_dir}".lower()
    return "corrig" in text or "manual_correct" in text


def plot_single(result, model_kind_label, output_dir, filename_slug, corrected):
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 14}):
        fig, ax = plt.subplots(figsize=(7.5, 7))
        test = result["test_df"]
        ax.scatter(test["xM_n"], test["xM_np1"], s=60, facecolors="#D9D9D9", edgecolors="#C9C9C9", alpha=0.75, zorder=1, label="Dado real (teste)")
        ax.scatter(test["xM_n"], test["xM_np1_pred"], s=34, facecolors="#0f223d", edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3, label="Recorrência (previsto)")
        lims = [min(test["xM_n"].min(), test["xM_np1"].min()), max(test["xM_n"].max(), test["xM_np1"].max())]
        ax.plot(lims, lims, "--", color="black", linewidth=1.5, zorder=0, label="1:1")
        m = result["metrics_test"]
        text = f"N={m['N']}\nr={m['r']:.4f}\nR²={m['R2']:.4f}\nRMSE={m['RMSE']:.3e}\nMAPE={m['MAPE_percent']:.2f}%"
        ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9))
        ax.set_xlabel(r"$X_M(n)$")
        ax.set_ylabel(r"$X_M(n+1)$")
        state = "corrigido manualmente" if corrected else "sem correção"
        ax.set_title(f"Recorrência {model_kind_label} — {state} (teste, {result['n_test_curves']} curvas)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=10, frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"grafico_{filename_slug}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"grafico_{filename_slug}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"grafico_{filename_slug}.svg"), bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    regime = get_regime(args.regime)
    checkpoint = args.checkpoint or find_repository_checkpoint(args.regime, args.variant)
    import inspect
    load_params = inspect.signature(load_curves_dataframe).parameters
    load_kwargs = {}
    if "accepted_only" in load_params:
        load_kwargs["accepted_only"] = args.accepted_only
    if "include_wt" in load_params:
        load_kwargs["include_wt"] = False
    curves = load_curves_dataframe(checkpoint, **load_kwargs)
    output_dir = args.output_dir or os.path.join(repository_figure_dir(args.regime, args.variant), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    corrected = corrected_run(checkpoint, output_dir)
    suffix = "_corrigido" if corrected else ""

    for anchored, kind_slug, kind_label in ((True, "ancorada", "Ancorada (esfera)"), (False, "nao_ancorada", "Não ancorada")):
        result = fit_model(curves, args.regime, regime.mix_factor, args.test_fraction, args.seed, anchored)
        print(f"\n--- {kind_label} ---")
        print("TREINO:", result["metrics_train"])
        print("TESTE :", result["metrics_test"])
        table_data = build_table_data(result["metrics_train"], result["metrics_test"])
        table_prefix = os.path.join(output_dir, f"tabela_recorrencia_{kind_slug}{suffix}")
        save_compact_table(table_prefix, table_data)
        formula_filename = os.path.join(output_dir, f"formula_recorrencia_{kind_slug}{suffix}.txt")
        save_recurrence_formula(
            formula_filename,
            result["coefficients"],
            anchored,
            args.regime,
            regime.mix_factor,
        )
        plot_single(result, kind_label, output_dir, f"{kind_slug}{suffix}", corrected)
        print(f"Tabela: {table_prefix}.png/.pdf/.svg")
        print(f"Fórmula: {formula_filename}")

    print(f"\nSaved recurrence tables/formulas/figures to: {output_dir}")


if __name__ == "__main__":
    main()
