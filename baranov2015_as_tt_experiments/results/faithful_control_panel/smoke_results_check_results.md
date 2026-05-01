# Faithful Control Panel Results

## Settings
- Run tag: `smoke_results_check`
- Tensor shape: `7^4`
- Total tensor points: `2401`
- Author sampler backend: `ttml`
- TTML env: `matrix_approximation_final_3_11`
- Author tol: `1e-05`
- Author budget mode: `fraction_of_tensor` -> `1201`
- Completion prefix budget mode: `absolute` -> `512`
- Budget sweep mode: `absolute` -> `[8]`
- Random test points: `4`
- Selected model runs: `['tt_als_smoke', 'authors_faithful_ttml_smoke']`

## Authors Baseline
- `authors_faithful_ttml_smoke`: backend=`ttml_dmrg`, unique=`2401`, total=`7154`, ranks=`[1, 7, 12, 7, 1]`, storage=`1274`, off-grid RMSE=`0.005411443507302078` meV

## Results

### budget_sweep

| budget | model | runner | status | observed | train | val | eval | rank | train RMSE (meV) | completion hidden RMSE (meV) | off-grid RMSE (meV) | time(s) |
|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 8 | authors_faithful_ttml_smoke | authors_baseline | not_ready | 2401 | 2401 | 0 | 4 | 1x7x12x7x1 | n/a | n/a | n/a | 82.22 |
| 8 | tt_als_smoke | fixed_mask_baseline | ok | 8 | 8 | 0 | 4 | 2 | 0.000000 | n/a | 1119.989045 | 0.02 |