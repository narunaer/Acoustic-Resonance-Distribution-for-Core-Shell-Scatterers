# Acoustic resonances of fluid core-shell scatterers

Code and results for the paper *Acoustic Resonance Distribution for Core-Shell Scatterers* (N. E. Rodrigues, G. Nakamura, O. M. Bruno, A. S. Martinez, Journal of Applied Physics).

A fluid core A (radius a) is covered by a fluid shell B (radius b = (1 + δ) a) and immersed in a host medium M. The contrasts are α = Z_A/Z_M and β = Z_B/Z_M, split between density and sound speed by the mixing factor f (ρ_i = ρ_M r^f, c_i = c_M r^(1-f)): f = 0 is the sound-speed-driven regime, f = 0.5 the mixed regime and f = 1 the density-driven regime. The dimensionless frequency is x_M = k_M a, in 10⁻⁴ ≤ x_M ≤ 1, with ℓ = 0 to 10.

## Contents

| Folder | File | What it does |
|---|---|---|
| `gerador/` | `gerador_coreshell.py` | Generates the resonance dataset. For every (α, β, δ) of the grid it computes the total internal energy W_T = α W̃_A + β W̃_B and counts the resonances. |
| `validacao/` | `fem_autocontido.py` | Independent checks: the closed-form coefficients against `np.linalg.solve` of the boundary system, the poles against the argument principle, and the integral of the coefficients on a pole-adapted finite-element mesh against the residue sum. |
| `relacoes/` | `relacoes_existencia_recorrencia.py` | Fits and evaluates the two empirical relations of the paper: existence of resonance (Eq. 12, Table I) and the anchored recurrence relation (Eq. 13, Table II). |
| `relacoes/` | `custo_autocontido.py` | Computational cost per configuration: full calculation x existence and recurrence relations. |
| `sistemas/` | `sistemas_reais_liu_marston.ipynb` | Figs. 4 and 5: the systems of Liu et al. (2000) and Marston (2025), with the resonance positions predicted by the recurrence relation. |
| `figuras/` | `figuras_artigo.ipynb` | Figs. 3 and 6 to 8: resonance maps in the (α, β) plane, δ x η maps, W_T spectra of the selected cases and their tables. |
| `resultados/` | | The results used in the paper (see below). |

## How the resonances are counted (generator)

* Base grid with 5000 points in x_M: 1000 log-spaced in [10⁻⁴, 0.1], 3000 linear in [0.1, 0.9] and 1000 linear in [0.9, 1].
* α ≥ 1 and β ≥ 1: every maximum of W_T with prominence ≥ 0.1 in log10 is a resonance.
* α < 1 or β < 1: the zeros of z⁴ D_ℓ are found in the complex plane and checked with the argument principle. The grid is refined around each pole with a step on the scale of Δ_W = π/Φ, Φ = c_M/c_A + δ c_M/c_B. A pole counts as a resonance when Q ≥ 2 and its area is at least 0.1% of the energy in a window of width Δ_W.

`LEIAME.txt`, written in each output folder, explains every file and column.

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

The figures use LaTeX for the labels. Install a LaTeX distribution (TeX Live or MiKTeX, with `dvipng` and `cm-super`). Without LaTeX, the relation scripts fall back to matplotlib text; the notebooks need LaTeX.

## Running

1. Dataset: 6800 curves per regime (17 values of δ and 400 pairs (α, β) balanced in η = (α − β)/(α + β)). Use `--regime mixed` for the three regimes, so the files are named `*_mix0.00`, `*_mix0.50` and `*_mix1.00` and are found by the other scripts.

   ```
   python -u gerador/gerador_coreshell.py --regime mixed --mix-factor 0 0.5 1 --workers 8 --saida dados
   ```

   Curves with α ≥ 1 and β ≥ 1 take a fraction of a second; the others take about 15 to 60 s each, so a full regime takes roughly 15 to 40 CPU hours. The run can be stopped and started again: it continues from the checkpoint. `--refazer-falhas` runs again the curves that hit the time limit. Quick test: add `--max-curvas 3`.

2. Empirical relations and computational cost:

   ```
   python relacoes/relacoes_existencia_recorrencia.py dados
   python relacoes/custo_autocontido.py dados
   ```

   The folder can also be given in the variable `CORESHELL_DADOS`. In Google Colab, with no argument, the scripts mount the Drive and read `MyDrive/coreshell_final/resultados_mix`.

3. Validation:

   ```
   python validacao/fem_autocontido.py 0,1 fem_auto.json
   ```

4. Figures: open `figuras/figuras_artigo.ipynb` and `sistemas/sistemas_reais_liu_marston.ipynb` in Jupyter and run all cells. `figuras_artigo.ipynb` reads `../dados`, the folder of step 1 (or the folder in `CORESHELL_DADOS`).

## Results of the paper

The folder `resultados/` has the results used in the paper, with the same structure that the scripts create in `dados/`. The dataset checkpoints (`checkpoint*.npz`) are not included because of their size; they are recreated by step 1. `velocity` is f = 0 (`mix0.00`), `mixed` is f = 0.5 (`mix0.50`) and `density` is f = 1 (`mix1.00`).

| Paper | Files in `resultados/` | Made by |
|---|---|---|
| Fig. 1 | `figuras_do_artigo/fig1.png` | schematic |
| Fig. 2 | `figuras_do_artigo/fig2.png` | sphere x core-shell comparison |
| Fig. 3 | `esquema_analise_resultados_WT_log/ZA_ZB_<regime>_..._halo` | `figuras_artigo.ipynb` |
| Table I and Eq. 12 | `relacoes_existencia_recorrencia_final/tabela_coeficientes_existencia` and `<regime>/formula_existencia.txt` | `relacoes_existencia_recorrencia.py` |
| Existence metrics (accuracy, precision, recall, ROC-AUC) | `relacoes_existencia_recorrencia_final/<regime>/tabela_existencia` | `relacoes_existencia_recorrencia.py` |
| Table II and Eq. 13 | `relacoes_existencia_recorrencia_final/tabela_coeficientes_recorrencia_ancorada` | `relacoes_existencia_recorrencia.py` |
| Recurrence metrics | `relacoes_existencia_recorrencia_final/<regime>/tabela_recorrencia_ancorada` | `relacoes_existencia_recorrencia.py` |
| Figs. 4 and 5 | `figuras_do_artigo/fig4.png`, `fig5.png` | `sistemas_reais_liu_marston.ipynb` |
| Figs. 6 to 8, panel [a] | `coreshell_lmax10_mixed_mixX.XX/esquema_analise_resultados_<regime>/Lente3_Projecao_Delta_vs_Eta_Media_<regime>` | `figuras_artigo.ipynb` |
| Figs. 6 to 8, panels [b] to [d] and tables | `graficos_WT_mesma_malha_gerador/<regime>/` | `figuras_artigo.ipynb` |
| Computational cost | `custo_computacional/custo_computacional.csv` | `custo_autocontido.py` |

`figuras_do_artigo/` has each figure as it appears in the paper (PNG at 600 dpi). The files `*_dados.npz` next to the maps hold the numbers drawn in each map.

## Authors

Naruna E. Rodrigues, Gilberto Nakamura, Odemir M. Bruno, Alexandre S. Martinez (Instituto de Física de São Carlos, Universidade de São Paulo).
