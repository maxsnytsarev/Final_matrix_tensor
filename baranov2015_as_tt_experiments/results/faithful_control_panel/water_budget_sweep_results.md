# Faithful Control Panel Results

## Settings
- Run tag: `water_budget_sweep`
- Experiment mode: `single_budget`
- Tensor shape: `7^4`
- Total tensor points: `2401`
- Author sampler backend: `ttml`
- TTML env: `matrix_approximation_final_3_11`
- Author tol: `1e-05`
- Author budget mode: `fraction_of_tensor` -> `481`
- Completion prefix budget mode: `fraction_of_tensor` -> `481`
- Budget sweep mode: `fraction_of_tensor` -> `[121, 241, 481, 961]`
- Random test points: `1000`
- Selected model runs: `['my_cheb_rank50', 'cp_wopt_r12', 'halrtc_default', 'authors_budgeted_ttml']`

## Results

### completion_on_samples

| budget | model | runner | status | observed | train | val | eval | rank | train RMSE (meV) | completion hidden RMSE (meV) | off-grid RMSE (meV) | time(s) |
|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 481 | authors_budgeted_ttml | authors_budgeted | budget_exhausted | 481 | 481 | 0 | 1000 | n/a | n/a | n/a | n/a | 0.00 |
| 481 | cp_wopt_r12 | fixed_mask_baseline | ok | 481 | 481 | 0 | 1000 | 12 | 0.000000 | 742.498906 | 339.171399 | 2.07 |
| 481 | halrtc_default | fixed_mask_baseline | ok | 481 | 481 | 0 | 1000 | n/a | 0.000000 | 717.535864 | 354.942788 | 0.05 |
| 481 | my_cheb_rank50 | approximateLOO_masked | ok | 481 | 481 | 0 | 1000 | 50 | 1.571145 | 1212.854588 | 747.849714 | 9.25 |