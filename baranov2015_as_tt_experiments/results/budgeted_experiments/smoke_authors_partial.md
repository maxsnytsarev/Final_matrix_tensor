# Baranov 2015 Budgeted Comparison

Two experiment modes:
- `single_budget`: one authors-driven online sampling run at a fixed budget; all baselines use the same sampled points.
- `budget_sweep`: one table per budget prefix; completion baselines use prefixes of the same authors-driven trajectory, and `authors_budgeted` is rerun under each budget to measure its own partial surrogate error.

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
- Single budget resolved: `121`
- Sweep budgets resolved: `[121, 241, 481, 961]`
- Baselines: `['authors_budgeted']`
- My Cheb ranks: `[50]`
- My Cheb lambda_all_list: `[0.1]`
- Trace source: `fresh`

## Completion

### Budget = 121

| baseline | status | observed | train | val | eval | rank | RMSE (meV) | MAE (meV) | maxAE (meV) | rRMSE | MAPE | off-grid RMSE (meV) | time(s) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| authors_budgeted | budget_exhausted_partial_surrogate | 121 | 121 | 0 | 1000 | 1x7x12x7x1 | 2068099.238744 | 2068098.502746 | 2075655.347276 | 2361.765162 | 6397.606363 | 2068216.789484 | 38.78 |

