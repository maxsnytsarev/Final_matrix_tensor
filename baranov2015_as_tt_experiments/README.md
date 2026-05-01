# Baranov 2015 AS+TT Water Experiments

This folder contains a clean, paper-oriented implementation for the water experiment from:

- V. Baranov, I. Oseledets, "Fitting high-dimensional potential energy surface using active subspace and tensor train (AS+TT) method", J. Chem. Phys. 143, 174107 (2015)

## What is implemented

- `baranov2015_faithful.py`
  - paper-specific planar water setup
  - active-subspace computation from gradient covariance in raw 4D planar coordinates
  - Chebyshev mesh in active-subspace coordinates
  - exact tensor-entry oracle
  - `LoggedBudgetedOracle` for online query logging, unique-budget truncation, and export to completion
  - `run_baranov2015_water_baseline(...)`
  - `collect_author_samples(...)`
  - `run_completion_on_author_samples(...)`
  - `run_budget_sweep_experiment(...)`
- `completion_linalg_tensor_masked.py`
  - mask-safe sibling of the Chebyshev completion code
  - observed entries are tracked by explicit sparse masks, not by `value != 0`
- `paper_as_tt_baseline.py`
  - retained as geometry / mapping / transform source
  - also provides the current `NON_FAITHFUL_FALLBACK` TT-cross backend used only if the external `ttml` backend is unavailable
- `run_baranov2015_water_experiments.py`
  - older experiment runner kept for comparison / plotting workflows

## Important interpretation

The author method is **not plain completion**.

It is an **online adaptive interpolation** method:

1. choose the next tensor entries adaptively;
2. query only those PES values from the oracle;
3. build the TT-cross approximation from those queried values.

The `shared_query_completion` section turns those queried points into a fixed observed set so that completion baselines and Cheb can be compared on exactly the same points.

## Faithful vs fallback

- `faithful`:
  - planar water geometry
  - active subspace from gradient covariance
  - Chebyshev sampling domain in reduced coordinates
  - online authors-driven querying
  - unique-query budgeting and prefix export to completion
- `NON_FAITHFUL_FALLBACK`:
  - current default TT-cross backend when neither Octave `TT-Toolbox` nor `ttpy` is operational in the environment
  - implemented via the existing custom cross routine from `paper_as_tt_baseline.py`
  - always marked explicitly in result metadata / logs

Current backend status in this workspace:

- primary online TT-cross backend: external Python package `ttml`
- default backend in `baranov2015_faithful.py`: `tt_backend="ttml"`
- recommended conda env for the full water pipeline: `matrix_approximation_final_3_11`
- direct Octave `TT-Toolbox` integration is still scaffolded but not wired end-to-end
- `NON_FAITHFUL_FALLBACK` is now only a safety net if `ttml` cannot be imported

## What Runs Where

Run everything from the repository root:

`/Users/maximsnytsarev/PycharmProjects/tensor_completion2`

Recommended env for the real water benchmark:

```bash
conda activate matrix_approximation_final_3_11
```

Why this env:

- `ttml` is installed there and provides the online `tt-cross`
- `pyscf` is installed there and provides exact HF/cc-pVDZ energies and gradients for water

Which module does what:

- `baranov2015_as_tt_experiments/baranov2015_faithful.py`
  - main experiment entrypoint
  - builds paper water context
  - computes active subspace
  - launches authors-driven online sampling via `ttml`
  - exports logged exact queries
- `baranov2015_as_tt_experiments/run_faithful_control_panel.py`
  - single-file runner with editable `FILE_RUN_CONFIG`
  - lets you switch on/off `authors_baseline`, `collect_samples`, `completion_on_samples`, `budget_sweep`
  - lets you run both `approximateLOO_masked` variants and other fixed-mask completion baselines on the same author-selected samples
- `completion_linalg_tensor_masked.py`
  - mask-safe completion backend
  - used only after the author sampler has produced `unique_indices`, `unique_values`, `shape`

Which mode to run:

- `--mode authors_baseline`
  - full paper-like pipeline
  - computes AS, runs authors sampler to convergence, builds Chebyshev surrogate, evaluates it
- `--mode collect_samples`
  - runs only the authors online sampler
  - stops at `unique_budget`
  - returns `unique_indices`, `unique_values`, `query_sequence`
- `--mode budget_sweep`
  - first collects one author trajectory up to `max(budgets)`
  - then runs completion on prefixes of that same trajectory
  - this is the main comparison mode for completion-vs-budget

## Main outputs

- paper-like Table I metric: `RMS random (meV)`
- author query counts:
  - `unique evals` = actual expensive PES evaluations
  - `total queries` = all TT-cross oracle accesses, including cache hits
- paper-like Table II metric:
  - maximal TT-rank of the coefficient tensor across Chebyshev points
  - no-AS tensor rank column for the value tensor
- paper-like Figure 1:
  - contour projections in the active-subspace variables
- completion metrics on the shared queried points:
  - `RMSE (meV)`
  - `MAE (meV)`
  - `maxAE (meV)`
  - `rRMSE`
  - `MAPE`

## Run

Faithful water baseline API:

```python
from baranov2015_as_tt_experiments.baranov2015_faithful import run_baranov2015_water_baseline

result = run_baranov2015_water_baseline(
    args=None,
    n_points=6,
    tol=1e-5,
    tt_backend="ttml",
)
```

Collect authors-driven online samples under a unique-query budget:

```python
from baranov2015_as_tt_experiments.baranov2015_faithful import collect_author_samples

trace = collect_author_samples(
    args=None,
    n_points=6,
    tol=1e-5,
    unique_budget=512,
    tt_backend="ttml",
)
```

Run completion on the same author trajectory prefix:

```python
from baranov2015_as_tt_experiments.baranov2015_faithful import run_completion_on_author_samples

completion = run_completion_on_author_samples(
    trace,
    {
        "rank": 12,
        "number_of_steps": 5,
        "tol": 1e-3,
        "validation_size": 0.1,
        "n_workers": 1,
    },
)
```

Budget sweep on one shared authors trajectory:

```python
from baranov2015_as_tt_experiments.baranov2015_faithful import run_budget_sweep_experiment

sweep = run_budget_sweep_experiment(
    args=None,
    budgets=[64, 128, 256, 512],
    n_points=6,
    tol=1e-5,
    completion_kwargs={"rank": 12, "n_workers": 1},
    seed=0,
)
```

CLI:

```bash
python -m baranov2015_as_tt_experiments.baranov2015_faithful --mode authors_baseline --n-points 6 --tol 1e-5 --tt-backend ttml --ttml-env-name matrix_approximation_final_3_11
python -m baranov2015_as_tt_experiments.baranov2015_faithful --mode collect_samples --n-points 6 --tol 1e-5 --unique-budget 512 --tt-backend ttml --ttml-env-name matrix_approximation_final_3_11
python -m baranov2015_as_tt_experiments.baranov2015_faithful --mode budget_sweep --n-points 6 --tol 1e-5 --budgets 64,128,256,512 --tt-backend ttml --ttml-env-name matrix_approximation_final_3_11
```

Small smoke run that is fast enough to sanity-check the pipeline:

```bash
python -m baranov2015_as_tt_experiments.baranov2015_faithful --mode collect_samples --n-points 2 --tol 1e-3 --unique-budget 12 --tt-backend ttml --ttml-env-name matrix_approximation_final_3_11 --water-as-samples 2
```

Control-panel runner:

```bash
python baranov2015_as_tt_experiments/run_faithful_control_panel.py
```

Edit `FILE_RUN_CONFIG` inside that file to change:

- which stages run
- author budget / `n_points` / `tol`
- `ttml` backend settings
- `approximateLOO_masked` variants
- other completion baselines such as `tt_als`, `tucker`, `cp_wopt`, `halrtc`, `cheb_auto`

Legacy runner:

```bash
python /Users/maximsnytsarev/PycharmProjects/tensor_completion2/baranov2015_as_tt_experiments/run_baranov2015_water_experiments.py
```

Edit `FILE_RUN_CONFIG` inside the runner to switch:

- `run_mode = "paper"` for author-only
- `run_mode = "completion"` for shared-query completion only
- `run_mode = "both"` for both sections

Paper sections can also be toggled independently:

- `paper_run_table_i`
- `paper_run_table_ii`
- `paper_run_fig1`

The default Table II configuration is aligned with the article:

```python
"paper_table_ii_thresholds": [1e-3, 1e-5, 1e-7],
"paper_table_ii_points": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
```

Figure 1 visualizations are written to:

```python
"paper_plots_dir": "baranov2015_as_tt_experiments/results/as_tt_water/paper_plots",
```

Shared-query budgets are configured in:

```python
"shared_query": {
    "budget_variants": [
        {"name": "frac0p2", "mode": "fraction", "fraction": 0.2},
        {"name": "full", "mode": "full_tensor"},
    ],
}
```
