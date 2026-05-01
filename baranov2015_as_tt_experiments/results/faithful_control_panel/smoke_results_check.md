# Faithful Control Panel Run

- `run_tag`: `smoke_results_check`
- `n_points`: `7`
- `tensor_dim`: `4`
- `tol`: `1e-05`
- `tt_backend`: `ttml`
- `ttml_env_name`: `matrix_approximation_final_3_11`
- `water_as_samples`: `256`
- `total_tensor_points`: `2401`
- `authors_budget_mode`: `fraction_of_tensor`
- `authors_budget_resolved`: `1201`
- `completion_prefix_budget_mode`: `absolute`
- `completion_prefix_budget_resolved`: `512`
- `budget_sweep_mode`: `absolute`
- `budget_sweep_resolved`: `[8]`
- `evaluation_random_test_points`: `4`
- `evaluation_exact_grid_metrics`: `False`
- `selected_model_runs`: `['tt_als_smoke', 'authors_faithful_ttml_smoke']`

## Authors Baseline

- model: `authors_faithful_ttml_smoke`
- backend: `ttml_dmrg`
- unique queries: `2401`
- total queries: `7154`
- tt ranks: `[1, 7, 12, 7, 1]`
- rms_random_mev: `0.005411443507302078`

## budget_sweep

- `tt_als_smoke` [fixed_mask_baseline] @ budget `8` status=`ok` observed=`8` train=`8` val=`0` rank=`2` completion_hidden_rmse_mev=`None` test_rmse_mev=`1119.9890450542835` train_rmse=`0.0`
- `authors_faithful_ttml_smoke` [authors_baseline] @ budget `8` status=`not_ready` observed=`2401` train=`2401` val=`0` rank=`1x7x12x7x1` completion_hidden_rmse_mev=`None` test_rmse_mev=`None` train_rmse=`None`
