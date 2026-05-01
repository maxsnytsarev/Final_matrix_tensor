# Baranov 2015 Budgeted Comparison

This runner always uses the authors-driven online sampler to choose points.
- `single_budget`: collect one authors trajectory prefix and run all baselines on that exact set of sampled points.
- `budget_sweep`: collect one longer authors trajectory once, then run all baselines on prefixes of that same trajectory.
- `authors_budgeted`: this is the authors method under the common budget. If it has not converged by that budget, its error is reported as `n/a` because no faithful surrogate exists yet.

## Settings
- Geometry: `as_tt_water_experiments/data/as_tt_water/water.xyz`
- Coordinate unit: `Bohr`
- Active subspace dim: `4`
- Active subspace sigma^2: `0.1`
- Active subspace samples: `256`
- Chebyshev interval: `[-0.3, 0.3]`
- Chebyshev points per mode: `7`
- Tensor shape: `7^4`
- Total tensor points: `2401`
- Run mode: `single_budget`
- Sampling policy: `fraction`
- Single budget resolved: `481`
- Sweep budgets resolved: `[121, 241, 481, 961]`
- Baselines: `['my_cheb', 'authors_budgeted']`
- Trace source: `npz`

## Results

### Budget = 481

| baseline | status | observed | train | val | eval | rank | train RMSE (meV) | hidden completion RMSE (meV) | off-grid RMSE (meV) | time(s) | note |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| my_cheb | ok | 481 | 433 | 48 | 4 | 2 | 1144.400396 | nan | 3879.386584 | 0.09 |  |
| authors_budgeted | budget_exhausted_before_convergence | 481 | 481 | 0 | 0 | online | nan | nan | nan | 0.00 | The authors online sampler was run under the common budget and did not converge by this prefix, so no faithful authors surrogate exists yet. |

