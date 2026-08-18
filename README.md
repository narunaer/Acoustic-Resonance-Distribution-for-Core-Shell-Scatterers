# From Energy to Design: Resonance State Maps of Fluid Core-Shell Scatterers

Reproducible code, raw detector outputs, explicit manual curation, final datasets, figures, recurrence fits, and resonance-existence classifiers for density, mixed, and sound-speed/velocity contrast regimes.

The repository runs locally from VS Code or any terminal. Google Colab is not required.

## Repository structure

```text
data/
  checkpoints/
    density/
      final/
    mixed/
      raw/
      manual_corrected/
    velocity/
      raw/
      manual_corrected/
  exports/
    density/
      final/
    mixed/
      raw/
      manual_corrected/
    velocity/
      raw/
      manual_corrected/

figures/
  density/
    final/
  mixed/
    raw/
    manual_corrected/
  velocity/
    raw/
    manual_corrected/

results/
  density/
    final/
  mixed/
    raw/
    manual_corrected/
  velocity/
    raw/
    manual_corrected/

src/core_shell/
  physics.py
  regimes.py
  detection.py
  checkpoints.py
  manual_corrections.py
  analysis.py
  models.py

scripts/
  generate_dataset.py
  apply_manual_corrections.py
  reorganize_existing_outputs.py
  make_all_figures.py
  fit_recurrence.py
  fit_existence_classifier.py
  check_repository.py
```

For mixed and velocity:

- `raw/` is the untouched automatic detector output;
- `manual_corrected/` is the hand-curated dataset used for the article.

The two stages never share the same output directory.

## 1. Install locally

From the repository root:

```bash
python -m pip install -e .
```

Optional validation:

```bash
python scripts/validate_installation.py
```

## 2. If you already generated data with the older flat layout

Run once:

```bash
python scripts/reorganize_existing_outputs.py
```

This does **not** recompute the 6800 curves. It only moves existing files into the staged folder structure.

Because old mixed/velocity figures could have been produced from either the raw or corrected checkpoint and previously shared filenames, the migration preserves them under:

```text
figures/mixed/legacy_unclassified/
figures/velocity/legacy_unclassified/
```

They can then be regenerated into clearly identified `raw/` and `manual_corrected/` folders.

## 3. Generate datasets from scratch

```bash
python scripts/generate_dataset.py --regime density
python scripts/generate_dataset.py --regime mixed
python scripts/generate_dataset.py --regime velocity
```

Density has one final stage.

Mixed and velocity are deterministic two-stage pipelines:

1. automatic detector -> `raw/`;
2. explicit hand-curated correction list -> `manual_corrected/`.

The manual lists are stored in:

```text
src/core_shell/manual_corrections.py
```

The raw checkpoint is preserved.

### Apply manual corrections to already-generated raw checkpoints

```bash
python scripts/apply_manual_corrections.py --regime mixed
python scripts/apply_manual_corrections.py --regime velocity
```

No 6800-curve regeneration is needed.

## 4. Generate article-final figures and fits

The default for mixed and velocity is always the manually corrected dataset:

```bash
python scripts/make_all_figures.py --regime density
python scripts/make_all_figures.py --regime mixed
python scripts/make_all_figures.py --regime velocity
```

Final outputs go to:

```text
figures/density/final/
figures/mixed/manual_corrected/
figures/velocity/manual_corrected/
```

To generate the raw detector version separately for comparison:

```bash
python scripts/make_all_figures.py --regime mixed --variant raw
python scripts/make_all_figures.py --regime velocity --variant raw
```

These go to:

```text
figures/mixed/raw/
figures/velocity/raw/
```

and never overwrite the article-final figures.

## 5. Where the recurrence relation is saved

For mixed and velocity, `make_all_figures.py` also runs the anchored recurrence fit.

Final formulas are written to:

```text
results/mixed/manual_corrected/mixed_anchored_recurrence_formula.txt
results/velocity/manual_corrected/velocity_anchored_recurrence_formula.txt
```

The file contains the complete fitted form with the numerical coefficients:

```text
x_M(n+1) = x_ref*A(eta,delta) + B(eta,delta)*x_M(n)
```

with quadratic `A(eta,delta)` and `B(eta,delta)`.

Reference factors:

Velocity:

```text
x_ref = sqrt(3)*alpha
```

Mixed with `mix_factor=0.5`:

```text
x_ref = sqrt(3)*alpha^(3/4)
```

The same results folder also contains coefficient JSON and train/test metrics JSON. The corresponding fit figure is stored under `figures/<regime>/<stage>/analysis/`.

## 6. Where the resonance-existence formula is saved

For every regime, the logistic existence classifier writes its complete fitted formula to:

```text
results/density/final/density_existence_classifier_formula.txt
results/mixed/manual_corrected/mixed_existence_classifier_formula.txt
results/velocity/manual_corrected/velocity_existence_classifier_formula.txt
```

The model is

```text
P_res(alpha,beta,delta) = 1/(1 + exp[-g(alpha,beta,delta)])
```

where `g` is quadratic in `ln(alpha)`, `ln(beta)`, and `delta`. The text file contains all fitted `c0...c9` values and the decision rule `P_res >= 0.5`.

Coefficient JSON and metrics JSON are stored beside the formula. The classifier figure is stored under `figures/<regime>/<stage>/analysis/`.

## 7. Standard numerical setup

```text
x_M range       : 1e-4 to 1
global mesh     : 3000 points
lmax            : 10
shell thickness : 17 delta values
alpha-beta grid : eta-balanced
```

## 8. Validate before publishing

```bash
python scripts/check_repository.py
```

The checker reports raw/final checkpoints separately and verifies that the article-final figures and formula files exist.
