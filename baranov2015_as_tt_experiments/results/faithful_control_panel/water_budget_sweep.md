# Faithful Control Panel Run

- `run_tag`: `water_budget_sweep`
- `experiment_mode`: `single_budget`
- `n_points`: `7`
- `tensor_dim`: `4`
- `tol`: `1e-05`
- `tt_backend`: `ttml`
- `ttml_env_name`: `matrix_approximation_final_3_11`
- `water_as_samples`: `256`
- `total_tensor_points`: `2401`
- `authors_budget_mode`: `fraction_of_tensor`
- `authors_budget_resolved`: `481`
- `completion_prefix_budget_mode`: `fraction_of_tensor`
- `completion_prefix_budget_resolved`: `481`
- `budget_sweep_mode`: `fraction_of_tensor`
- `budget_sweep_resolved`: `[121, 241, 481, 961]`
- `evaluation_random_test_points`: `1000`
- `evaluation_exact_grid_metrics`: `True`
- `selected_model_runs`: `['my_cheb_rank50', 'cp_wopt_r12', 'halrtc_default', 'authors_budgeted_ttml']`

## completion_on_samples

- `my_cheb_rank50` [approximateLOO_masked] @ budget `481` status=`ok` observed=`481` train=`481` val=`0` rank=`50` completion_hidden_rmse_mev=`1212.85458763688` test_rmse_mev=`747.8497140166952` train_rmse=`1.5711454996597058`
- `cp_wopt_r12` [fixed_mask_baseline] @ budget `481` status=`ok` observed=`481` train=`481` val=`0` rank=`12` completion_hidden_rmse_mev=`742.4989061999061` test_rmse_mev=`339.17139895239586` train_rmse=`0.0`
- `halrtc_default` [fixed_mask_baseline] @ budget `481` status=`ok` observed=`481` train=`481` val=`0` rank=`n/a` completion_hidden_rmse_mev=`717.5358635261396` test_rmse_mev=`354.9427884604473` train_rmse=`0.0`
- `authors_budgeted_ttml` [authors_budgeted] @ budget `481` status=`budget_exhausted` observed=`481` train=`481` val=`0` rank=`n/a` completion_hidden_rmse_mev=`None` test_rmse_mev=`None` train_rmse=`None`
