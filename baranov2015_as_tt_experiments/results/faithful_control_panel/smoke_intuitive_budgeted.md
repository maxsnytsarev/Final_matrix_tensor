# Faithful Control Panel Run

- `run_tag`: `smoke_intuitive_budgeted`
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
- `evaluation_random_test_points`: `4`
- `evaluation_exact_grid_metrics`: `False`
- `selected_model_runs`: `['my_cheb_smoke', 'authors_budgeted_smoke']`

## completion_on_samples

- `my_cheb_smoke` [approximateLOO_masked] @ budget `481` status=`ok` observed=`481` train=`433` val=`48` rank=`2` completion_hidden_rmse_mev=`None` test_rmse_mev=`3879.3865841423462` train_rmse=`1144.4003956137572`
- `authors_budgeted_smoke` [authors_budgeted] @ budget `481` status=`budget_exhausted` observed=`481` train=`481` val=`0` rank=`n/a` completion_hidden_rmse_mev=`None` test_rmse_mev=`None` train_rmse=`None`
