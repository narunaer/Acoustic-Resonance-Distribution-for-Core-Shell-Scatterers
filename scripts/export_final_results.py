#!/usr/bin/env python
"""Regenerate results/<regime>/<variant>/*.txt+json from the checkpoints
already stored in data/checkpoints/, using the installed core_shell
package (the same fitting code used everywhere else in this repo).

For mixed and velocity, both the raw and the manual_corrected checkpoint
are fitted, so results/<regime>/raw/ and results/<regime>/manual_corrected/
are always kept in sync and clearly distinguished. Density has no manual
correction stage, so only results/density/final/ is written.

Run after data/checkpoints/ is populated (generate_dataset.py +
apply_manual_corrections.py, or checkpoints already committed):

    python scripts/export_final_results.py
"""
import argparse
import json
import os

from core_shell.checkpoints import (
    find_repository_checkpoint,
    load_curves_dataframe,
    repository_results_dir,
)
from core_shell.models import fit_anchored_recurrence, fit_existence_classifier

CLASSIFIER_FORMULA_TEMPLATE = """RESONANCE-EXISTENCE CLASSIFIER
==============================

P_res(alpha,beta,delta) = 1 / (1 + exp[-g(alpha,beta,delta)])

g(alpha,beta,delta) = c0 + c1*ln(alpha) + c2*ln(beta) + c3*delta + c4*[ln(alpha)]^2 + c5*ln(alpha)*ln(beta) + c6*delta*ln(alpha) + c7*[ln(beta)]^2 + c8*delta*ln(beta) + c9*delta^2

Decision used by the classifier:
  resonance     if P_res >= 0.5
  no resonance  if P_res < 0.5

COEFFICIENTS
{coef_lines}
"""

RECURRENCE_FORMULA_TEMPLATE = """ANCHORED RESONANCE RECURRENCE
================================

x_M(n+1) = x_ref * A(eta,delta) + B(eta,delta) * x_M(n)

A(eta,delta) = a0 + a_eta*eta + a_delta*delta + a_eta2*eta^2 + a_eta_delta*eta*delta + a_delta2*delta^2
B(eta,delta) = b0 + b_eta*eta + b_delta*delta + b_eta2*eta^2 + b_eta_delta*eta*delta + b_delta2*delta^2

{xref_block}

NUMERICAL FORM FOR THIS FIT
A(eta,delta) = ({a0:+.12e}) ({a_eta:+.12e})*eta ({a_delta:+.12e})*delta ({a_eta2:+.12e})*eta^2 ({a_eta_delta:+.12e})*eta*delta ({a_delta2:+.12e})*delta^2
B(eta,delta) = ({b0:+.12e}) ({b_eta:+.12e})*eta ({b_delta:+.12e})*delta ({b_eta2:+.12e})*eta^2 ({b_eta_delta:+.12e})*eta*delta ({b_delta2:+.12e})*delta^2

COEFFICIENTS
{coef_lines}
"""

XREF_BLOCK = {
    "velocity": "x_ref = sqrt(3) * alpha",
    "mixed": (
        "x_ref = sqrt(3/(m*m_t)), m=alpha^(mix_factor-1), m_t=1/alpha\n"
        "For mix_factor=0.5: x_ref = sqrt(3) * alpha^0.75"
    ),
}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def export_classifier(regime, variant, curves, checkpoint_path, output_dir):
    result = fit_existence_classifier(curves)
    names = [f"c{i}" for i in range(10)]
    coefficients = {name: float(value) for name, value in zip(names, result["formula_coefficients"])}

    coef_lines = "\n".join(f"{name} = {value:+.12e}" for name, value in coefficients.items())
    formula_text = CLASSIFIER_FORMULA_TEMPLATE.format(coef_lines=coef_lines)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{regime}_existence_classifier_formula.txt"), "w", encoding="utf-8") as file:
        file.write(formula_text)
    write_json(os.path.join(output_dir, f"{regime}_existence_classifier_coefficients.json"), coefficients)
    write_json(os.path.join(output_dir, f"{regime}_existence_classifier_metrics.json"), {
        "training": result["train_metrics"],
        "test": result["test_metrics"],
        "regime": regime,
        "variant": variant,
        "checkpoint": checkpoint_path,
    })


def export_recurrence(regime, variant, curves, checkpoint_path, output_dir, mix_factor=0.5):
    result = fit_anchored_recurrence(curves, regime_name=regime, mix_factor=mix_factor)
    coefficients = {
        name: float(value)
        for name, value in zip(result["coefficient_names"], result["coefficients"])
    }
    coef_lines = "\n".join(f"{name} = {value:+.12e}" for name, value in coefficients.items())
    formula_text = RECURRENCE_FORMULA_TEMPLATE.format(
        xref_block=XREF_BLOCK[regime], coef_lines=coef_lines, **coefficients,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{regime}_anchored_recurrence_formula.txt"), "w", encoding="utf-8") as file:
        file.write(formula_text)
    write_json(os.path.join(output_dir, f"{regime}_anchored_recurrence_coefficients.json"), coefficients)
    write_json(os.path.join(output_dir, f"{regime}_anchored_recurrence_metrics.json"), {
        "training": result["train_metrics"],
        "test": result["test_metrics"],
        "rank": result["rank"],
        "condition_number": result["condition_number"],
        "regime": regime,
        "variant": variant,
        "mix_factor": mix_factor,
        "checkpoint": checkpoint_path,
    })


def process(regime, variant):
    checkpoint = find_repository_checkpoint(regime, variant)
    curves = load_curves_dataframe(checkpoint)
    output_dir = repository_results_dir(regime, variant)

    print(f"[{regime}/{variant}] checkpoint: {checkpoint}")
    print(f"[{regime}/{variant}] curves: {len(curves)} | with peaks: {int((curves['n_peaks'] > 0).sum())}")

    export_classifier(regime, variant, curves, checkpoint, output_dir)
    if regime in ("mixed", "velocity"):
        export_recurrence(regime, variant, curves, checkpoint, output_dir)

    print(f"[{regime}/{variant}] wrote results to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime", choices=("density", "mixed", "velocity", "all"), default="all"
    )
    args = parser.parse_args()

    jobs = []
    regimes = ("density", "mixed", "velocity") if args.regime == "all" else (args.regime,)
    for regime in regimes:
        if regime == "density":
            jobs.append(("density", "final"))
        else:
            jobs.append((regime, "raw"))
            jobs.append((regime, "manual_corrected"))

    for regime, variant in jobs:
        process(regime, variant)


if __name__ == "__main__":
    main()
