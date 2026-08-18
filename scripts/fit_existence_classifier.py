#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from core_shell.checkpoints import load_curves_dataframe, find_repository_checkpoint, repository_figure_dir
from core_shell.models import fit_existence_classifier

TABLE_FIGSIZE = (5.4, 4.5)
TABLE_DPI = 600
COMPARE_COLUMNS = ["Treino", "Teste"]
METRIC_ORDER = ["N", "Positive_rate", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
ROW_LABELS = ["N\n(curvas)", "Taxa de\npositivos", "Acurácia", "Precisão", "Recall", "F1", "ROC AUC"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the resonance-existence classifier and save the compact article table.")
    parser.add_argument("--regime", choices=("density", "velocity", "mixed"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", choices=("raw", "manual_corrected", "final"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fmt_int(value):
    return "--" if not np.isfinite(value) else f"{int(round(value))}"


def fmt_r(value):
    return "--" if not np.isfinite(value) else f"{float(value):.4f}"


def fmt_pct01(value):
    return "--" if not np.isfinite(value) else f"{100.0 * float(value):.1f}%"


FORMATTERS = {
    "N": fmt_int,
    "Positive_rate": fmt_pct01,
    "Accuracy": fmt_r,
    "Precision": fmt_r,
    "Recall": fmt_r,
    "F1": fmt_r,
    "ROC_AUC": fmt_r,
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


def classifier_formula_text(coefficients):
    c = np.asarray(coefficients, dtype=float)
    lines = [
        "Classificador de existência de ressonância",
        "",
        "P_res(alpha,beta,delta) = 1 / (1 + exp[-g(alpha,beta,delta)])",
        "g = c0 + c1*ln(alpha) + c2*ln(beta) + c3*delta",
        "    + c4*[ln(alpha)]^2 + c5*ln(alpha)*ln(beta) + c6*delta*ln(alpha)",
        "    + c7*[ln(beta)]^2 + c8*delta*ln(beta) + c9*delta^2",
        "",
    ]
    lines += [f"c{i} = {value:+.12e}" for i, value in enumerate(c)]
    return "\n".join(lines) + "\n"


def save_classifier_formula(filename, coefficients):
    text = classifier_formula_text(coefficients)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    return text


def corrected_run(checkpoint, output_dir):
    text = f"{checkpoint} {output_dir}".lower()
    return "corrig" in text or "manual_correct" in text


def main():
    args = parse_args()
    checkpoint = args.checkpoint or find_repository_checkpoint(args.regime, args.variant)
    import inspect
    load_params = inspect.signature(load_curves_dataframe).parameters
    load_kwargs = {}
    if "accepted_only" in load_params:
        load_kwargs["accepted_only"] = args.accepted_only
    if "include_wt" in load_params:
        load_kwargs["include_wt"] = False
    curves = load_curves_dataframe(checkpoint, **load_kwargs)
    result = fit_existence_classifier(curves, args.test_fraction, args.seed)
    output_dir = args.output_dir or os.path.join(repository_figure_dir(args.regime, args.variant), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_corrigido" if corrected_run(checkpoint, output_dir) else ""
    prefix = os.path.join(output_dir, f"tabela_classificador_existencia{suffix}")
    save_compact_table(prefix, build_table_data(result["train_metrics"], result["test_metrics"]))
    formula_filename = os.path.join(output_dir, f"formula_classificador_existencia{suffix}.txt")
    save_classifier_formula(formula_filename, result["formula_coefficients"])
    print("TREINO:", result["train_metrics"])
    print("TESTE :", result["test_metrics"])
    print(f"Tabela classificador: {prefix}.png/.pdf/.svg")
    print(f"Fórmula classificador: {formula_filename}")


if __name__ == "__main__":
    main()
