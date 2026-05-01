# Faithful Control Panel Results

## Settings
- Run tag: `smoke_intuitive_budgeted`
- Experiment mode: `single_budget`
- Tensor shape: `7^4`
- Total tensor points: `2401`
- Author sampler backend: `ttml`
- TTML env: `matrix_approximation_final_3_11`
- Author tol: `1e-05`
- Author budget mode: `fraction_of_tensor` -> `481`
- Completion prefix budget mode: `fraction_of_tensor` -> `481`
- Budget sweep mode: `fraction_of_tensor` -> `[121, 241, 481, 961]`
- Random test points: `4`
- Selected model runs: `['my_cheb_smoke', 'authors_budgeted_smoke']`

## Results

### completion_on_samples

| budget | model | runner | status | observed | train | val | eval | rank | train RMSE (meV) | completion hidden RMSE (meV) | off-grid RMSE (meV) | time(s) |
|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 481 | authors_budgeted_smoke | authors_budgeted | budget_exhausted | 481 | 481 | 0 | 4 | n/a | n/a | n/a | n/a | 0.00 |
| 481 | my_cheb_smoke | approximateLOO_masked | ok | 481 | 433 | 48 | 4 | 2 | 1144.400396 | n/a | 3879.386584 | 0.03 |