from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable

    setattr(tqdm, "write", staticmethod(lambda msg: print(msg, flush=True)))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_baselines.baseline_as_tt_online import ASTTOnline
from all_baselines.baseline_cp_wopt import CPWOPT
from all_baselines.baseline_halrtc import HaLRTC
from all_baselines.baseline_tt_als import TTALS
from all_baselines.baseline_tucker import TuckerCompletion
from as_tt_water_experiments import run_as_tt_water_experiments as water
from as_tt_water_experiments import run_as_tt_water_dual_experiments as dual
from baranov2015_as_tt_experiments import baranov2015_faithful as faithful
from baseline.baseline_cheb_cp import ChebCPCompletion


# Run this file from the repository root:
#   cd /Users/maximsnytsarev/PycharmProjects/tensor_completion2
# Recommended environment for the real water benchmark:
#   conda activate matrix_approximation_final_3_11
# Then launch:
#   python baranov2015_as_tt_experiments/run_faithful_control_panel.py

RUN_FROM_FILE_CONFIG = True

FILE_RUN_CONFIG = {
    "run": {
        "authors_baseline": False,
        "collect_samples": False,
        "completion_on_samples": True,
        "budget_sweep": False,
    },
    # Main user-facing control block.
    # `single_budget`: one authors-driven budget, then run all enabled models on that same set of points.
    # `budget_sweep`: one long authors-driven trajectory, then run all enabled models on each budget prefix.
    # `collect_only`: only collect and save the authors-driven sampled points.
    # `full_authors`: run the full faithful authors baseline without a budget cap.
    "experiment": {
        "mode": "single_budget",  # single_budget | budget_sweep | collect_only | full_authors
        "budget_mode": "fraction_of_tensor",  # absolute | fraction_of_tensor
        "budget": 512,
        "budget_fraction": 0.2,
        "budgets": [64, 128, 256, 512],
        "budget_fractions": [0.05, 0.1, 0.2, 0.4],
    },
    "paper_args": {
        "data_root": "baranov2015_as_tt_experiments/data/as_tt_water",
        "geometry_xyz": "as_tt_water_experiments/data/as_tt_water/water.xyz",
        "cache_tensors": True,
        "generate_if_missing": True,
        "qc_method": "RHF",
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 0,
        "scf_conv_tol": 1e-9,
        "scf_max_cycle": 50,
        "scf_init_guess": "hcore",
        "scf_verbose": 0,
        "scf_newton_fallback": True,
        "water_coordinate_unit": "Bohr",
        "water_as_dim": 4,
        "water_as_sigma2": 0.1,
        "water_as_samples": 256,
        "water_as_random_state": 1729,
        "cheb_interval": [-0.3, 0.3],
        "author_rank_cap": 12,
        "author_tol": 1e-5,
        "author_max_iter": 30,
        "author_random_state": 1729,
        "author_log_first_unique": 8,
        "author_log_every_unique": 25,
        "author_log_every_total": 250,
        "author_rms_probe_points": 1000,
        "author_rms_seed": 2025,
        "show_tqdm": True,
        "verbose": True,
    },
    "authors_sampler": {
        "n_points": 7,
        "tol": 1e-5,
        "budget_mode": "fraction_of_tensor",  # absolute | fraction_of_tensor
        "unique_budget": 512,
        "unique_budget_fraction": 0.5,
        "tt_backend": "ttml",
        "ttml_env_name": "matrix_approximation_final_3_11",
        "ttml_method": "dmrg",
        "random_state": 0,
    },
    "trace_source": {
        "mode": "fresh",  # fresh | npz
        "npz_path": "baranov2015_as_tt_experiments/results/faithful_control_panel/latest_trace.npz",
    },
    "completion_on_samples": {
        "prefix_budget_mode": "fraction_of_tensor",  # absolute | fraction_of_tensor
        "prefix_budget": 512,
        "prefix_budget_fraction": 0.2,
        "test_seed": 0,
        "run_all_enabled_variants": True,
    },
    "budget_sweep": {
        "budget_mode": "fraction_of_tensor",  # absolute | fraction_of_tensor
        "budgets": [64, 128, 256, 512],
        "budget_fractions": [0.05, 0.1, 0.2, 0.4],
        "test_seed": 0,
        "run_authors_baseline_if_trace_converged": False,
    },
    "evaluation": {
        "random_test_points": 1000,
        "compute_exact_grid_metrics": True,
    },
    "model_runs": [
        {
            "enabled": True,
            "name": "my_cheb_rank50",
            "runner": "approximateLOO_masked",
            "params": {
                "rank": 50,
                "strict_rank": True,
                "allow_rank_growth": False,
                "number_of_steps": 10,
                "tol_for_step": 1e-4,
                "begin": "oversample",
                "lambda_all": 0.1,
                "rank_eval": False,
                "rank_nest": False,
                "nest_iters": 5,
                "tol": 1e-5,
                "max_rank": 20,
                "using_qr": False,
                "eval": True,
                "seed": 179,
                "ret_best": True,
                "TQDM": True,
                "eval_fall": True,
                "validation_size": 0.1,
                "tolerance": 500,
                "min_validation_points": 8,
                "min_train_ratio_to_params": 0.5,
                "dual_guided": False,
                "pivot_topk": 8,
                "n_workers": 22,
                "completion_value_mode": "relative_mev_observed_min",
            },
        },
        {
            "enabled": False,
            "name": "tt_als_r12",
            "runner": "fixed_mask_baseline",
            "model": "tt_als",
            "params": {
                "rank": 12,
                "max_iter": 120,
                "tol": 1e-5,
                "verbose": True,
            },
        },
        {
            "enabled": False,
            "name": "tucker_r12",
            "runner": "fixed_mask_baseline",
            "model": "tucker",
            "params": {
                "rank": 12,
                "max_iter": 120,
                "tol": 1e-5,
                "verbose": True,
            },
        },
        {
            "enabled": True,
            "name": "cp_wopt_r12",
            "runner": "fixed_mask_baseline",
            "model": "cp_wopt",
            "params": {
                "rank": 12,
                "max_iter": 120,
                "tol": 1e-5,
                "random_state": 1000,
                "verbose": True,
            },
        },
        {
            "enabled": True,
            "name": "halrtc_default",
            "runner": "fixed_mask_baseline",
            "model": "halrtc",
            "params": {
                "max_iter": 120,
                "tol": 1e-5,
                "verbose": True,
            },
        },
        {
            "enabled": True,
            "name": "authors_budgeted_ttml",
            "runner": "authors_budgeted",
            "params": {},
        },
        {
            "enabled": False,
            "name": "authors_full_faithful_ttml",
            "runner": "authors_full_baseline",
            "params": {},
        },
        {
            "enabled": False,
            "name": "cheb_cp_r20_fixed",
            "runner": "fixed_mask_baseline",
            "model": "cheb_cp",
            "params": {
                "rank": 20,
                "number_of_steps": 5,
                "tol_for_step": 1e-4,
                "lambda_all": 0.0,
                "seed": 1000,
                "tolerance": 500,
                "source_file": "completion_linalg_tensor_masked.py",
                "verbose": True,
                "rank_eval": False,
                "rank_nest": False,
                "nest_iters": 5,
                "validation_size": 0.1,
                "strict_rank": True,
                "allow_rank_growth": False,
                "n_workers":22,
            },
        },
    ],
    # Legacy fallback defaults for old configs.
    # When `model_runs` is non-empty, model selection and per-model hyperparameters
    # should be edited there instead of in the blocks below.
    "completion_cheb_defaults": {
        "rank": 20,
        "strict_rank": True,
        "allow_rank_growth": False,
        "number_of_steps": 5,
        "tol_for_step": 1e-4,
        "begin": "oversample",
        "lambda_all": 0.1,
        "rank_eval": False,
        "rank_nest": False,
        "nest_iters": 5,
        "tol": 1e-5,
        "max_rank": 20,
        "using_qr": False,
        "eval": True,
        "seed": 179,
        "ret_best": True,
        "TQDM": True,
        "eval_fall": True,
        "validation_size": 0.1,
        "tolerance": 500,
        "min_validation_points": 8,
        "min_train_ratio_to_params": 0.5,
        "dual_guided": False,
        "pivot_topk": 8,
        "n_workers": 1,
        "completion_value_mode": "relative_mev_observed_min",
    },
    "completion_variants": [
        {
            "name": "approxloo_rank20_fixed",
            "enabled": True,
            "kwargs": {
            },
        },
        {
            "name": "approxloo_rank20_deeper",
            "enabled": False,
            "kwargs": {
                "rank": 20,
                "number_of_steps": 8,
                "tol_for_step": 1e-5,
                "max_rank": 24,
            },
        },
    ],
    "fixed_mask_baselines": {
        "enabled": True,
        "base_baselines": ["tt_als", "tucker", "cp_wopt", "halrtc"],
        "run_cheb": True,
        "rank": 12,
        "max_iter": 120,
        "tol": 1e-5,
        "random_state": 1000,
        "disable_joblib_multiprocessing": True,
        "cheb_ranks": [20],
        "cheb_steps": 5,
        "cheb_tol_for_step": 1e-4,
        "cheb_lambda_all": 0.0,
        "cheb_lambda_all_list": [0.0],
        "cheb_rank_eval": False,
        "cheb_rank_nest": False,
        "cheb_nest_iters": 5,
        "cheb_strict_rank": True,
        "cheb_allow_rank_growth": False,
        "cheb_tolerance": 500,
        "cheb_validation_size": 0.1,
        "cheb_validation_size_list": [0.1],
        "cheb_source_file": "completion_linalg_tensor_masked.py",
        "as_tt_online_steps": 8,
        "as_tt_online_inner_max_iter": 20,
        "as_tt_online_budget_fraction_cap": 0.2,
        "verbose": True,
    },
    "io": {
        "outputs_dir": "baranov2015_as_tt_experiments/results/faithful_control_panel",
        "run_tag": "water_budget_sweep",
        "save_summary_json": True,
        "save_summary_md": True,
        "save_results_json": True,
        "save_results_md": True,
        "save_trace_npz": True,
        "pretty_print_summary": True,
    },
}


def _log(msg: str) -> None:
    writer = getattr(tqdm, "write", None)
    if callable(writer):
        writer(msg)
    else:
        print(msg, flush=True)


def _tensor_total_points_from_cfg(cfg: dict[str, Any]) -> int:
    return int(cfg["authors_sampler"]["n_points"]) ** int(cfg["paper_args"]["water_as_dim"])


def _resolve_budget_value(*, mode: str, absolute: int | float | None, fraction: float | None, total_points: int) -> int:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "absolute":
        if absolute is None:
            raise ValueError("Absolute budget mode requires `absolute` value")
        raw = int(round(float(absolute)))
    elif mode_norm == "fraction_of_tensor":
        if fraction is None:
            raise ValueError("Fraction budget mode requires `fraction` value")
        raw = int(np.ceil(float(fraction) * float(total_points)))
    else:
        raise ValueError(f"Unsupported budget mode: {mode!r}")
    return max(1, min(int(total_points), int(raw)))


def _resolve_author_budget(cfg: dict[str, Any]) -> int:
    return _resolve_budget_value(
        mode=str(cfg["authors_sampler"]["budget_mode"]),
        absolute=cfg["authors_sampler"].get("unique_budget"),
        fraction=cfg["authors_sampler"].get("unique_budget_fraction"),
        total_points=_tensor_total_points_from_cfg(cfg),
    )


def _resolve_prefix_budget(cfg: dict[str, Any]) -> int:
    return _resolve_budget_value(
        mode=str(cfg["completion_on_samples"]["prefix_budget_mode"]),
        absolute=cfg["completion_on_samples"].get("prefix_budget"),
        fraction=cfg["completion_on_samples"].get("prefix_budget_fraction"),
        total_points=_tensor_total_points_from_cfg(cfg),
    )


def _resolve_sweep_budgets(cfg: dict[str, Any]) -> list[int]:
    mode = str(cfg["budget_sweep"]["budget_mode"]).strip().lower()
    total_points = _tensor_total_points_from_cfg(cfg)
    if mode == "absolute":
        raw = cfg["budget_sweep"]["budgets"]
        budgets = [
            _resolve_budget_value(mode="absolute", absolute=item, fraction=None, total_points=total_points)
            for item in raw
        ]
    elif mode == "fraction_of_tensor":
        raw = cfg["budget_sweep"]["budget_fractions"]
        budgets = [
            _resolve_budget_value(mode="fraction_of_tensor", absolute=None, fraction=item, total_points=total_points)
            for item in raw
        ]
    else:
        raise ValueError(f"Unsupported budget_sweep.budget_mode={mode!r}")
    return sorted(set(int(x) for x in budgets))


def _fmt_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _apply_experiment_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment = dict(cfg.get("experiment", {}))
    if not experiment:
        return cfg
    mode = str(experiment.get("mode", "single_budget")).strip().lower()
    budget_mode = str(experiment.get("budget_mode", "fraction_of_tensor")).strip().lower()
    budget = int(experiment.get("budget", cfg["authors_sampler"]["unique_budget"]))
    budget_fraction = float(experiment.get("budget_fraction", cfg["authors_sampler"]["unique_budget_fraction"]))
    budgets = [int(x) for x in experiment.get("budgets", cfg["budget_sweep"]["budgets"])]
    budget_fractions = [float(x) for x in experiment.get("budget_fractions", cfg["budget_sweep"]["budget_fractions"])]

    cfg["authors_sampler"]["budget_mode"] = budget_mode
    cfg["authors_sampler"]["unique_budget"] = budget
    cfg["authors_sampler"]["unique_budget_fraction"] = budget_fraction
    cfg["completion_on_samples"]["prefix_budget_mode"] = budget_mode
    cfg["completion_on_samples"]["prefix_budget"] = budget
    cfg["completion_on_samples"]["prefix_budget_fraction"] = budget_fraction
    cfg["budget_sweep"]["budget_mode"] = budget_mode
    cfg["budget_sweep"]["budgets"] = budgets
    cfg["budget_sweep"]["budget_fractions"] = budget_fractions

    if mode == "single_budget":
        cfg["run"]["authors_baseline"] = False
        cfg["run"]["collect_samples"] = False
        cfg["run"]["completion_on_samples"] = True
        cfg["run"]["budget_sweep"] = False
    elif mode == "budget_sweep":
        cfg["run"]["authors_baseline"] = False
        cfg["run"]["collect_samples"] = False
        cfg["run"]["completion_on_samples"] = False
        cfg["run"]["budget_sweep"] = True
    elif mode == "collect_only":
        cfg["run"]["authors_baseline"] = False
        cfg["run"]["collect_samples"] = True
        cfg["run"]["completion_on_samples"] = False
        cfg["run"]["budget_sweep"] = False
    elif mode == "full_authors":
        cfg["run"]["authors_baseline"] = True
        cfg["run"]["collect_samples"] = False
        cfg["run"]["completion_on_samples"] = False
        cfg["run"]["budget_sweep"] = False
    else:
        raise ValueError(
            f"Unsupported experiment.mode={mode!r}. "
            "Expected one of ['single_budget', 'budget_sweep', 'collect_only', 'full_authors']."
        )
    return cfg


def _log_trace_summary(label: str, trace: faithful.SampleTrace) -> None:
    _log(
        f"[control-panel] {label} status={trace.status} backend={trace.tt_backend_used} "
        f"unique={trace.unique_queries} total={trace.total_queries} shape={trace.shape}"
    )


def _log_model_start(prefix: str, *, variant: str, family: str, budget: int) -> None:
    _log(f"{prefix} START variant={variant} family={family} budget={budget}")


def _log_authors_baseline_summary(baseline: faithful.AuthorsBaselineResult) -> None:
    _log(
        f"[control-panel] authors_baseline DONE backend={baseline.backend_used} "
        f"unique={baseline.unique_queries} total={baseline.total_queries} "
        f"rms_random_mev={_fmt_metric(baseline.rms_random_mev)} "
        f"test_rmse_mev={_fmt_metric(baseline.test_metrics.get('rmse_mev'))} "
        f"test_max_abs_mev={_fmt_metric(baseline.test_metrics.get('max_abs_mev'))}"
    )


def _log_completion_summary(prefix: str, summary: dict[str, Any]) -> None:
    status = str(summary.get("status", "ok"))
    if summary.get("info", {}).get("error") or status == "fail":
        _log(
            f"{prefix} FAIL variant={summary['variant']} budget={summary['budget']} "
            f"error={summary['info']['error']}"
        )
        return
    if status == "not_ready":
        _log(
            f"{prefix} NOT_READY variant={summary['variant']} family={summary['baseline_family']} "
            f"budget={summary['budget']} required_unique={summary.get('authors_required_unique_queries')} "
            f"available_unique={summary.get('budget')}"
        )
        return
    if status == "budget_exhausted":
        _log(
            f"{prefix} BUDGET_EXHAUSTED variant={summary['variant']} family={summary['baseline_family']} "
            f"budget={summary['budget']} unique={summary.get('observed_points')} "
            f"trace_status={summary.get('info', {}).get('trace_status')}"
        )
        return
    range_min = summary.get("completed_min_hartree")
    range_max = summary.get("completed_max_hartree")
    range_part = ""
    if range_min is not None and range_max is not None:
        range_part = f" completed_range_h=[{_fmt_metric(range_min)}, {_fmt_metric(range_max)}]"
    _log(
        f"{prefix} DONE variant={summary['variant']} family={summary['baseline_family']} "
        f"budget={summary['budget']} train_rmse={_fmt_metric(summary.get('train_rmse'))} "
        f"completion_hidden_rmse_mev={_fmt_metric(summary.get('completion_hidden_rmse_mev'))} "
        f"completion_hidden_rrmse={_fmt_metric(summary.get('completion_hidden_relative_rmse'))} "
        f"grid_hidden_rmse_mev={_fmt_metric(summary.get('grid_hidden_rmse_mev'))} "
        f"grid_full_rmse_mev={_fmt_metric(summary.get('grid_full_rmse_mev'))} "
        f"test_rmse_mev={_fmt_metric(summary.get('test_rmse_mev'))} "
        f"grid_roundtrip_rmse={_fmt_metric(summary.get('grid_roundtrip_rmse'))} "
        f"test_max_abs_mev={_fmt_metric(summary.get('test_max_abs_mev'))} "
        f"time={_fmt_metric(summary.get('elapsed_sec'), digits=2)}s"
        f"{range_part}"
    )


def _log_final_summary(summary: dict[str, Any]) -> None:
    _log("[control-panel] FINAL SUMMARY START")
    if summary.get("authors_baseline") is not None:
        baseline = summary["authors_baseline"]
        _log(
            f"[control-panel][final] authors_baseline backend={baseline['backend_used']} "
            f"unique={baseline['unique_queries']} total={baseline['total_queries']} "
            f"rms_random_mev={_fmt_metric(baseline['rms_random_mev'])} "
            f"test_rmse_mev={_fmt_metric(baseline['test_metrics'].get('rmse_mev'))}"
        )
    if summary.get("collected_trace") is not None:
        trace = summary["collected_trace"]
        _log(
            f"[control-panel][final] collected_trace status={trace['status']} "
            f"unique={trace['unique_queries']} total={trace['total_queries']} "
            f"backend={trace['tt_backend_used']}"
        )
    if summary.get("authors_baseline_run") is not None:
        _log_completion_summary("[control-panel][final][authors_baseline]", summary["authors_baseline_run"])
    for row in summary.get("completion_on_samples", []):
        _log_completion_summary("[control-panel][final][completion_on_samples]", row)
    for row in summary.get("budget_sweep", []):
        _log_completion_summary("[control-panel][final][budget_sweep]", row)
    _log("[control-panel] FINAL SUMMARY END")


def _log_model_runs_catalog(model_runs: list[dict[str, Any]]) -> None:
    if not model_runs:
        _log("[control-panel] model_runs=[]")
        return
    for i, item in enumerate(model_runs, start=1):
        params = dict(item.get("params", {}))
        runner = str(item.get("runner"))
        model = item.get("model")
        _log(
            f"[control-panel] model_run[{i}] name={item.get('name')} "
            f"runner={runner} model={model} params={params}"
        )


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_cfg() -> dict[str, Any]:
    if not RUN_FROM_FILE_CONFIG:
        raise RuntimeError("Only FILE_RUN_CONFIG mode is supported in this runner.")
    cfg = json.loads(json.dumps(FILE_RUN_CONFIG))
    cfg["paper_args"]["data_root"] = str(water.resolve_project_path(cfg["paper_args"]["data_root"]))
    cfg["paper_args"]["geometry_xyz"] = str(water.resolve_project_path(cfg["paper_args"]["geometry_xyz"]))
    cfg["trace_source"]["npz_path"] = str(water.resolve_project_path(cfg["trace_source"]["npz_path"]))
    cfg["io"]["outputs_dir"] = str(water.resolve_project_path(cfg["io"]["outputs_dir"]))
    cfg["paper_args"]["cheb_interval"] = [
        float(cfg["paper_args"]["cheb_interval"][0]),
        float(cfg["paper_args"]["cheb_interval"][1]),
    ]
    cfg["authors_sampler"]["n_points"] = int(cfg["authors_sampler"]["n_points"])
    cfg["authors_sampler"]["unique_budget"] = int(cfg["authors_sampler"]["unique_budget"])
    cfg["authors_sampler"]["unique_budget_fraction"] = float(cfg["authors_sampler"]["unique_budget_fraction"])
    cfg["completion_on_samples"]["prefix_budget"] = int(cfg["completion_on_samples"]["prefix_budget"])
    cfg["completion_on_samples"]["prefix_budget_fraction"] = float(cfg["completion_on_samples"]["prefix_budget_fraction"])
    cfg["budget_sweep"]["budgets"] = [int(x) for x in cfg["budget_sweep"]["budgets"]]
    cfg["budget_sweep"]["budget_fractions"] = [float(x) for x in cfg["budget_sweep"]["budget_fractions"]]
    if "experiment" in cfg:
        cfg["experiment"] = dict(cfg["experiment"])
        cfg["experiment"]["mode"] = str(cfg["experiment"].get("mode", "single_budget"))
        cfg["experiment"]["budget_mode"] = str(cfg["experiment"].get("budget_mode", "fraction_of_tensor"))
        cfg["experiment"]["budget"] = int(cfg["experiment"].get("budget", cfg["authors_sampler"]["unique_budget"]))
        cfg["experiment"]["budget_fraction"] = float(
            cfg["experiment"].get("budget_fraction", cfg["authors_sampler"]["unique_budget_fraction"])
        )
        cfg["experiment"]["budgets"] = [int(x) for x in cfg["experiment"].get("budgets", cfg["budget_sweep"]["budgets"])]
        cfg["experiment"]["budget_fractions"] = [
            float(x) for x in cfg["experiment"].get("budget_fractions", cfg["budget_sweep"]["budget_fractions"])
        ]
    cfg["evaluation"]["random_test_points"] = int(cfg["evaluation"]["random_test_points"])
    cfg["evaluation"]["compute_exact_grid_metrics"] = bool(cfg["evaluation"]["compute_exact_grid_metrics"])
    cfg["model_runs"] = [dict(item) for item in cfg.get("model_runs", [])]
    cfg["completion_cheb_defaults"] = dict(cfg["completion_cheb_defaults"])
    cfg["io"]["save_summary_json"] = bool(cfg["io"].get("save_summary_json", True))
    cfg["io"]["save_summary_md"] = bool(cfg["io"].get("save_summary_md", True))
    cfg["io"]["save_results_json"] = bool(cfg["io"].get("save_results_json", True))
    cfg["io"]["save_results_md"] = bool(cfg["io"].get("save_results_md", True))
    cfg["io"]["save_trace_npz"] = bool(cfg["io"].get("save_trace_npz", True))
    cfg["io"]["pretty_print_summary"] = bool(cfg["io"].get("pretty_print_summary", True))
    if "base_baselines" in cfg["fixed_mask_baselines"]:
        cfg["fixed_mask_baselines"]["base_baselines"] = [str(x) for x in cfg["fixed_mask_baselines"]["base_baselines"]]
    cfg["fixed_mask_baselines"]["run_cheb"] = bool(cfg["fixed_mask_baselines"].get("run_cheb", False))
    cfg["fixed_mask_baselines"]["cheb_ranks"] = [int(x) for x in cfg["fixed_mask_baselines"]["cheb_ranks"]]
    cfg["fixed_mask_baselines"]["cheb_lambda_all_list"] = [
        float(x) for x in cfg["fixed_mask_baselines"]["cheb_lambda_all_list"]
    ]
    cfg["fixed_mask_baselines"]["cheb_validation_size_list"] = [
        float(x) for x in cfg["fixed_mask_baselines"]["cheb_validation_size_list"]
    ]
    cfg = _apply_experiment_overrides(cfg)
    return cfg


def _faithful_args_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    args = dict(cfg["paper_args"])
    args["tt_backend"] = str(cfg["authors_sampler"]["tt_backend"])
    args["ttml_env_name"] = str(cfg["authors_sampler"]["ttml_env_name"])
    args["ttml_method"] = str(cfg["authors_sampler"]["ttml_method"])
    return args


def _enabled_model_runs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [dict(item) for item in cfg.get("model_runs", []) if bool(item.get("enabled", True))]
    if runs:
        out: list[dict[str, Any]] = []
        for item in runs:
            runner = str(item.get("runner", "")).strip()
            name = str(item.get("name", "")).strip()
            if not runner or not name:
                raise ValueError(f"Each model_runs entry needs non-empty 'name' and 'runner': {item}")
            if runner not in {
                "approximateLOO_masked",
                "fixed_mask_baseline",
                "authors_budgeted",
                "authors_full_baseline",
                "authors_baseline",
            }:
                raise ValueError(
                    f"Unsupported model_runs.runner={runner!r} for {name}. "
                    "Expected 'approximateLOO_masked', 'fixed_mask_baseline', "
                    "'authors_budgeted', or 'authors_full_baseline'."
                )
            if runner == "fixed_mask_baseline":
                model_name = str(item.get("model", "")).strip().lower()
                if not model_name:
                    raise ValueError(f"fixed_mask_baseline entry {name} must provide 'model'")
                if model_name == "cheb_auto":
                    raise ValueError(
                        "model='cheb_auto' is not a separate algorithm; it was only an old auto-expansion token. "
                        "Use explicit model='cheb_cp' with the exact hyperparameters you want."
                    )
            item["params"] = dict(item.get("params", {}))
            out.append(item)
        return out

    # Legacy fallback for older configs.
    legacy: list[dict[str, Any]] = []
    for variant in cfg.get("completion_variants", []):
        if bool(variant.get("enabled", True)):
            legacy.append(
                {
                    "enabled": True,
                    "name": str(variant["name"]),
                    "runner": "approximateLOO_masked",
                    "params": dict(cfg["completion_cheb_defaults"]) | dict(variant.get("kwargs", {})),
                }
            )
    return legacy


def _model_run_names(cfg: dict[str, Any]) -> list[str]:
    return [str(item["name"]) for item in _enabled_model_runs(cfg)]


def _model_run_names_by_runner(cfg: dict[str, Any], runner: str) -> list[str]:
    return [str(item["name"]) for item in _enabled_model_runs(cfg) if str(item.get("runner")) == str(runner)]


def _dual_cfg_from_panel(cfg: dict[str, Any]) -> dict[str, Any]:
    panel = dict(cfg["fixed_mask_baselines"])
    dual_cfg = {
        "rank": int(panel["rank"]),
        "max_iter": int(panel["max_iter"]),
        "tol": float(panel["tol"]),
        "random_state": int(panel["random_state"]),
        "disable_joblib_multiprocessing": bool(panel["disable_joblib_multiprocessing"]),
        "cheb_ranks": [int(x) for x in panel["cheb_ranks"]],
        "cheb_steps": int(panel["cheb_steps"]),
        "cheb_tol_for_step": float(panel["cheb_tol_for_step"]),
        "cheb_lambda_all": float(panel["cheb_lambda_all"]),
        "cheb_lambda_all_list": [float(x) for x in panel["cheb_lambda_all_list"]],
        "cheb_rank_eval": bool(panel.get("cheb_rank_eval", False)),
        "cheb_rank_nest": bool(panel.get("cheb_rank_nest", False)),
        "cheb_nest_iters": int(panel.get("cheb_nest_iters", 5)),
        "cheb_strict_rank": bool(panel.get("cheb_strict_rank", False)),
        "cheb_allow_rank_growth": bool(panel.get("cheb_allow_rank_growth", True)),
        "cheb_tolerance": int(panel["cheb_tolerance"]),
        "cheb_validation_size": float(panel["cheb_validation_size"]),
        "cheb_validation_size_list": [float(x) for x in panel["cheb_validation_size_list"]],
        "cheb_source_file": str(panel["cheb_source_file"]),
        "as_tt_online_steps": int(panel["as_tt_online_steps"]),
        "as_tt_online_inner_max_iter": int(panel["as_tt_online_inner_max_iter"]),
        "as_tt_online_budget_fraction_cap": float(panel["as_tt_online_budget_fraction_cap"]),
        "verbose": bool(panel["verbose"]),
    }
    return dual_cfg


def _resolve_source_file(path_like: str) -> str:
    return str((PROJECT_ROOT / str(path_like)).resolve())


def _instantiate_fixed_mask_model(model_spec: dict[str, Any], shape: tuple[int, ...]) -> Any:
    model_name = str(model_spec["model"]).strip().lower()
    params = dict(model_spec.get("params", {}))
    min_dim = int(min(shape))

    if model_name == "tt_als":
        tt_rank = int(params.pop("tt_rank", params.pop("rank", min_dim)))
        return TTALS(tt_rank=max(1, tt_rank), **params)

    if model_name == "tucker":
        if "ranks" in params:
            ranks = tuple(int(x) for x in params.pop("ranks"))
        else:
            rank = int(params.pop("rank", min_dim))
            ranks = tuple(max(1, min(rank, dim)) for dim in shape)
        return TuckerCompletion(ranks=ranks, **params)

    if model_name == "cp_wopt":
        rank = int(params.pop("rank", min_dim))
        return CPWOPT(rank=max(1, rank), **params)

    if model_name == "halrtc":
        return HaLRTC(**params)

    if model_name == "as_tt_online":
        tt_rank = int(params.pop("tt_rank", params.pop("rank", min_dim)))
        return ASTTOnline(tt_rank=max(1, min(tt_rank, min_dim)), **params)

    if model_name == "cheb_cp":
        if "source_file" in params:
            params["source_file"] = _resolve_source_file(str(params["source_file"]))
        return ChebCPCompletion(**params)

    raise ValueError(
        f"Unknown fixed-mask model={model_name!r}. "
        "Available: ['tt_als', 'tucker', 'cp_wopt', 'halrtc', 'as_tt_online', 'cheb_cp']"
    )


def _context_from_payload(payload: dict[str, Any]) -> faithful.PaperWaterContext:
    return faithful.PaperWaterContext(
        symbols=tuple(payload["symbols"]),
        base_coords=np.asarray(payload["base_coords"], dtype=float),
        coord_unit=str(payload["coord_unit"]),
        raw_active_dofs=tuple(tuple(int(v) for v in pair) for pair in payload["raw_active_dofs"]),
        raw_dim=int(payload["raw_dim"]),
        as_dim=int(payload["as_dim"]),
        as_basis=np.asarray(payload["as_basis"], dtype=float),
        as_eigenvalues=np.asarray(payload["as_eigenvalues"], dtype=float),
        cheb_nodes=None if payload.get("cheb_nodes") is None else np.asarray(payload["cheb_nodes"], dtype=float),
        cheb_interval=None if payload.get("cheb_interval") is None else tuple(float(v) for v in payload["cheb_interval"]),
        as_cache_path=payload.get("as_cache_path"),
        as_cache_tag=str(payload.get("as_cache_tag", "baranov2015_water_planar")),
        metadata=dict(payload.get("metadata", {})),
    )


def _save_sample_trace_npz(path: Path, trace: faithful.SampleTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "total_queries": int(trace.total_queries),
        "unique_queries": int(trace.unique_queries),
        "shape": [int(x) for x in trace.shape],
        "status": str(trace.status),
        "tt_backend_requested": str(trace.tt_backend_requested),
        "tt_backend_used": str(trace.tt_backend_used),
        "faithful_backend": bool(trace.faithful_backend),
        "info": dict(trace.info),
        "context": asdict(trace.context),
    }
    np.savez_compressed(
        path,
        unique_indices=np.asarray(trace.unique_indices, dtype=int),
        unique_values=np.asarray(trace.unique_values, dtype=float),
        query_sequence=np.asarray(trace.query_sequence, dtype=int),
        nodes=np.asarray(trace.nodes, dtype=float),
        meta_json=np.asarray(json.dumps(meta, default=_json_default)),
    )


def _load_sample_trace_npz(path: Path) -> faithful.SampleTrace:
    payload = np.load(path, allow_pickle=False)
    meta = json.loads(str(payload["meta_json"]))
    context = _context_from_payload(meta["context"])
    return faithful.SampleTrace(
        unique_indices=np.asarray(payload["unique_indices"], dtype=int),
        unique_values=np.asarray(payload["unique_values"], dtype=float),
        query_sequence=np.asarray(payload["query_sequence"], dtype=int),
        total_queries=int(meta["total_queries"]),
        unique_queries=int(meta["unique_queries"]),
        shape=tuple(int(x) for x in meta["shape"]),
        nodes=np.asarray(payload["nodes"], dtype=float),
        context=context,
        status=str(meta["status"]),
        tt_backend_requested=str(meta["tt_backend_requested"]),
        tt_backend_used=str(meta["tt_backend_used"]),
        faithful_backend=bool(meta["faithful_backend"]),
        info=dict(meta.get("info", {})),
    )


def _prefix_trace(trace: faithful.SampleTrace, budget: int) -> faithful.SampleTrace:
    budget = int(min(int(budget), trace.unique_queries))
    return faithful.SampleTrace(
        unique_indices=np.asarray(trace.unique_indices[:budget], dtype=int).copy(),
        unique_values=np.asarray(trace.unique_values[:budget], dtype=float).copy(),
        query_sequence=np.asarray(trace.query_sequence, dtype=int).copy(),
        total_queries=int(trace.total_queries),
        unique_queries=int(budget),
        shape=tuple(int(x) for x in trace.shape),
        nodes=np.asarray(trace.nodes, dtype=float).copy(),
        context=trace.context,
        status=str(trace.status),
        tt_backend_requested=str(trace.tt_backend_requested),
        tt_backend_used=str(trace.tt_backend_used),
        faithful_backend=bool(trace.faithful_backend),
        info=dict(trace.info),
    )


def _max_requested_budget(cfg: dict[str, Any]) -> int | None:
    requested: list[int] = []
    if cfg["run"].get("collect_samples", False):
        requested.append(_resolve_author_budget(cfg))
    if cfg["run"].get("completion_on_samples", False):
        requested.append(_resolve_prefix_budget(cfg))
    if cfg["run"].get("budget_sweep", False):
        budgets = _resolve_sweep_budgets(cfg)
        if budgets:
            requested.append(int(max(budgets)))
    if not requested:
        return None
    return int(max(requested))


def _shared_trace_from_cfg(cfg: dict[str, Any], faithful_args: dict[str, Any]) -> faithful.SampleTrace | None:
    max_budget = _max_requested_budget(cfg)
    if max_budget is None:
        return None
    source_mode = str(cfg["trace_source"]["mode"]).lower()
    trace_npz_path = Path(cfg["trace_source"]["npz_path"])
    if source_mode == "npz":
        trace = _load_sample_trace_npz(trace_npz_path)
        if trace.unique_queries < int(max_budget):
            raise ValueError(
                f"Loaded trace has only {trace.unique_queries} unique points, "
                f"but current config needs at least {max_budget}"
            )
        _log(f"[control-panel] loaded trace -> {trace_npz_path}")
        _log_trace_summary("trace_loaded", trace)
        return trace
    if source_mode != "fresh":
        raise ValueError(f"Unsupported trace_source.mode={source_mode!r}")
    _log(f"[control-panel] collecting shared author trace with budget={max_budget}")
    trace = faithful.collect_author_samples(
        faithful_args,
        n_points=int(cfg["authors_sampler"]["n_points"]),
        tol=float(cfg["authors_sampler"]["tol"]),
        unique_budget=int(max_budget),
        tt_backend=str(cfg["authors_sampler"]["tt_backend"]),
        random_state=int(cfg["authors_sampler"]["random_state"]),
    )
    _log_trace_summary("trace_collected", trace)
    return trace


def _make_test_payload(cfg: dict[str, Any], faithful_args: dict[str, Any], trace: faithful.SampleTrace) -> dict[str, Any]:
    interval = tuple(float(v) for v in trace.context.cheb_interval)
    test_seed = int(cfg["budget_sweep"]["test_seed"]) if cfg["run"].get("budget_sweep", False) else int(cfg["completion_on_samples"]["test_seed"])
    test_points = faithful.sample_random_test_points(
        trace.context,
        n_test=int(cfg["evaluation"]["random_test_points"]),
        a=interval[0],
        b=interval[1],
        seed=test_seed,
    )
    energy_fn = water._build_energy_fn(faithful._coerce_args(faithful_args))
    payload = {"test_points": test_points, "energy_fn": energy_fn}
    if bool(cfg["evaluation"].get("compute_exact_grid_metrics", False)):
        _log("[control-panel] preparing exact grid tensor for node-wise diagnostics")
        exact_tensor_h, cached_nodes = faithful.load_or_generate_full_value_tensor(
            faithful._coerce_args(faithful_args),
            context=trace.context,
            n_points=int(trace.shape[0]),
        )
        if np.max(np.abs(np.asarray(cached_nodes, dtype=float) - np.asarray(trace.nodes, dtype=float))) > 1e-12:
            raise ValueError("Exact grid tensor nodes mismatch against sample trace nodes")
        payload["exact_grid_tensor_h"] = np.asarray(exact_tensor_h, dtype=float)
    return payload


def _completion_row(prefix_budget: int, variant_name: str, result: faithful.CompletionResult) -> dict[str, Any]:
    train_metrics = dict(result.train_metrics)
    test_metrics = dict(result.test_metrics)
    return {
        "budget": int(prefix_budget),
        "variant": str(variant_name),
        "train_rmse": train_metrics.get("rmse_mev", train_metrics.get("rmse")),
        "train_mae": train_metrics.get("mae_mev", train_metrics.get("mae")),
        "train_max_abs": train_metrics.get("max_abs_mev", train_metrics.get("max_abs")),
        "test_rmse_mev": test_metrics.get("rmse_mev"),
        "test_max_abs_mev": test_metrics.get("max_abs_mev"),
        "completion_hidden_rmse_mev": test_metrics.get("completion_hidden_rmse_mev"),
        "completion_hidden_mae_mev": test_metrics.get("completion_hidden_mae_mev"),
        "completion_hidden_max_abs_mev": test_metrics.get("completion_hidden_max_abs_mev"),
        "completion_hidden_relative_rmse": test_metrics.get("completion_hidden_relative_rmse"),
        "completion_hidden_mape": test_metrics.get("completion_hidden_mape"),
        "completion_full_rmse_mev": test_metrics.get("completion_full_rmse_mev"),
        "completion_full_mae_mev": test_metrics.get("completion_full_mae_mev"),
        "completion_full_max_abs_mev": test_metrics.get("completion_full_max_abs_mev"),
        "completion_full_relative_rmse": test_metrics.get("completion_full_relative_rmse"),
        "completion_full_mape": test_metrics.get("completion_full_mape"),
        "grid_hidden_rmse_mev": test_metrics.get("grid_hidden_rmse_mev"),
        "grid_hidden_max_abs_mev": test_metrics.get("grid_hidden_max_abs_mev"),
        "grid_full_rmse_mev": test_metrics.get("grid_full_rmse_mev"),
        "grid_full_max_abs_mev": test_metrics.get("grid_full_max_abs_mev"),
        "grid_roundtrip_rmse": test_metrics.get("grid_roundtrip_rmse"),
        "grid_roundtrip_max_abs": test_metrics.get("grid_roundtrip_max_abs"),
        "elapsed_sec": float(result.info.get("elapsed_sec", float("nan"))),
        "history_len": len(result.history),
        "completed_min_hartree": result.info.get("completed_value_min_hartree"),
        "completed_max_hartree": result.info.get("completed_value_max_hartree"),
        "info": dict(result.info),
    }


def _point_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    true_arr = np.asarray(true_values, dtype=float).reshape(-1)
    pred_arr = np.asarray(pred_values, dtype=float).reshape(-1)
    diff = pred_arr - true_arr
    return {
        "mae_mev": float(np.mean(np.abs(diff)) * water.HARTREE_TO_MEV),
        "rmse_mev": float(np.sqrt(np.mean(diff**2)) * water.HARTREE_TO_MEV),
        "max_abs_mev": float(np.max(np.abs(diff)) * water.HARTREE_TO_MEV),
    }


def _observed_tensor_and_mask(trace: faithful.SampleTrace) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(x) for x in trace.shape)
    observed = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    indices = np.asarray(trace.unique_indices, dtype=int)
    values = np.asarray(trace.unique_values, dtype=float)
    if indices.size:
        mask[tuple(indices.T)] = True
        observed[tuple(indices.T)] = values
    return observed, mask


def _scale_trace_observations_for_completion(trace: faithful.SampleTrace) -> tuple[np.ndarray, np.ndarray, float]:
    observed_h, mask = _observed_tensor_and_mask(trace)
    reference_h = faithful._relative_mev_reference_from_observations(trace.unique_values)
    observed_scaled = np.zeros_like(observed_h, dtype=float)
    if np.any(mask):
        observed_scaled[mask] = faithful._to_relative_mev_with_reference(observed_h[mask], reference_h)
    return observed_scaled, mask, float(reference_h)


def _other_baseline_row(prefix_budget: int, baseline_name: str, completed_tensor: np.ndarray, trace: faithful.SampleTrace, elapsed_sec: float, info: dict[str, Any], eval_payload: dict[str, Any], baseline_family: str = "fixed_mask_baseline") -> dict[str, Any]:
    indices = np.asarray(trace.unique_indices, dtype=int)
    values = np.asarray(trace.unique_values, dtype=float)
    observed_pred = np.asarray(completed_tensor, dtype=float)[tuple(indices.T)]
    train_metrics = _point_metrics(values, observed_pred)
    test_metrics = faithful.evaluate_completed_tensor_on_test(
        completed_tensor=np.asarray(completed_tensor, dtype=float),
        context=trace.context,
        nodes=trace.nodes,
        test_points=np.asarray(eval_payload["test_points"], dtype=float),
        energy_fn=eval_payload["energy_fn"],
        a=float(trace.context.cheb_interval[0]),
        b=float(trace.context.cheb_interval[1]),
    )
    if "exact_grid_tensor_h" in eval_payload:
        test_metrics.update(
            faithful.evaluate_completed_tensor_on_grid(
                completed_tensor=np.asarray(completed_tensor, dtype=float),
                exact_tensor_h=np.asarray(eval_payload["exact_grid_tensor_h"], dtype=float),
                observed_indices=indices,
            )
        )
        test_metrics.update(
            faithful.evaluate_completion_metrics_dual_style(
                completed_tensor_h=np.asarray(completed_tensor, dtype=float),
                exact_tensor_h=np.asarray(eval_payload["exact_grid_tensor_h"], dtype=float),
                observed_indices=indices,
            )
        )
    return {
        "budget": int(prefix_budget),
        "variant": str(baseline_name),
        "train_rmse": train_metrics.get("rmse_mev"),
        "train_mae": train_metrics.get("mae_mev"),
        "train_max_abs": train_metrics.get("max_abs_mev"),
        "test_rmse_mev": test_metrics.get("rmse_mev"),
        "test_max_abs_mev": test_metrics.get("max_abs_mev"),
        "completion_hidden_rmse_mev": test_metrics.get("completion_hidden_rmse_mev"),
        "completion_hidden_mae_mev": test_metrics.get("completion_hidden_mae_mev"),
        "completion_hidden_max_abs_mev": test_metrics.get("completion_hidden_max_abs_mev"),
        "completion_hidden_relative_rmse": test_metrics.get("completion_hidden_relative_rmse"),
        "completion_hidden_mape": test_metrics.get("completion_hidden_mape"),
        "completion_full_rmse_mev": test_metrics.get("completion_full_rmse_mev"),
        "completion_full_mae_mev": test_metrics.get("completion_full_mae_mev"),
        "completion_full_max_abs_mev": test_metrics.get("completion_full_max_abs_mev"),
        "completion_full_relative_rmse": test_metrics.get("completion_full_relative_rmse"),
        "completion_full_mape": test_metrics.get("completion_full_mape"),
        "grid_hidden_rmse_mev": test_metrics.get("grid_hidden_rmse_mev"),
        "grid_hidden_max_abs_mev": test_metrics.get("grid_hidden_max_abs_mev"),
        "grid_full_rmse_mev": test_metrics.get("grid_full_rmse_mev"),
        "grid_full_max_abs_mev": test_metrics.get("grid_full_max_abs_mev"),
        "grid_roundtrip_rmse": test_metrics.get("grid_roundtrip_rmse"),
        "grid_roundtrip_max_abs": test_metrics.get("grid_roundtrip_max_abs"),
        "elapsed_sec": float(elapsed_sec),
        "history_len": None,
        "completed_min_hartree": float(np.min(np.asarray(completed_tensor, dtype=float))),
        "completed_max_hartree": float(np.max(np.asarray(completed_tensor, dtype=float))),
        "info": dict(info),
        "baseline_family": str(baseline_family),
    }


def _error_row(*, budget: int, variant: str, family: str, error: Exception | str, elapsed_sec: float) -> dict[str, Any]:
    return {
        "budget": int(budget),
        "variant": str(variant),
        "train_rmse": None,
        "train_mae": None,
        "train_max_abs": None,
        "test_rmse_mev": None,
        "test_max_abs_mev": None,
        "completion_hidden_rmse_mev": None,
        "completion_hidden_mae_mev": None,
        "completion_hidden_max_abs_mev": None,
        "completion_hidden_relative_rmse": None,
        "completion_hidden_mape": None,
        "grid_hidden_rmse_mev": None,
        "grid_hidden_max_abs_mev": None,
        "grid_full_rmse_mev": None,
        "grid_full_max_abs_mev": None,
        "grid_roundtrip_rmse": None,
        "grid_roundtrip_max_abs": None,
        "elapsed_sec": float(elapsed_sec),
        "history_len": None,
        "info": {"error": str(error)},
        "baseline_family": str(family),
        "status": "fail",
    }


def _rank_label_from_params(params: dict[str, Any] | None) -> str:
    params = dict(params or {})
    if "ranks" in params:
        ranks = params["ranks"]
        if isinstance(ranks, (list, tuple)):
            return "x".join(str(int(x)) for x in ranks)
        return str(ranks)
    for key in ("rank", "tt_rank", "max_rank"):
        if key in params and params[key] is not None:
            try:
                return str(int(params[key]))
            except Exception:
                return str(params[key])
    return "n/a"


def _annotate_result_row(
    row: dict[str, Any],
    *,
    phase_name: str,
    model_run: dict[str, Any],
    prefix: faithful.SampleTrace,
    eval_points: int,
) -> dict[str, Any]:
    out = dict(row)
    info = dict(out.get("info", {}))
    runner = str(model_run["runner"])
    params = dict(model_run.get("params", {}))
    out["phase"] = str(phase_name)
    out["runner"] = runner
    out["model"] = None if runner == "authors_baseline" else model_run.get("model")
    out["params"] = params
    out["status"] = str(out.get("status") or ("fail" if info.get("error") else "ok"))
    out["observed_points"] = int(info.get("observed_points", prefix.unique_queries))
    out["train_points"] = int(info.get("train_points_planned", info.get("train_points", out["observed_points"])))
    out["val_points"] = int(info.get("val_points_planned", info.get("validation_points", 0)))
    out["eval_points"] = int(eval_points)
    out["requested_rank"] = info.get("requested_rank", params.get("rank", params.get("tt_rank")))
    out["effective_rank"] = info.get("effective_rank", out["requested_rank"])
    out["rank_label"] = (
        str(out["effective_rank"])
        if out["effective_rank"] is not None
        else _rank_label_from_params(params)
    )
    return out


def _baseline_to_summary_payload(
    baseline: faithful.AuthorsBaselineResult,
    *,
    elapsed_sec: float,
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": str(model_name),
        "params": dict(params),
        "backend_requested": str(baseline.backend_requested),
        "backend_used": str(baseline.backend_used),
        "faithful_backend": bool(baseline.faithful_backend),
        "unique_queries": int(baseline.unique_queries),
        "total_queries": int(baseline.total_queries),
        "rms_random_mev": None if baseline.rms_random_mev is None else float(baseline.rms_random_mev),
        "tt_ranks": [int(x) for x in baseline.tt_ranks],
        "storage": int(baseline.storage),
        "test_metrics": dict(baseline.test_metrics),
        "info": dict(baseline.info),
        "elapsed_sec": float(elapsed_sec),
    }


def _authors_baseline_to_row(
    *,
    baseline_payload: dict[str, Any],
    requested_budget: int,
    variant_name: str,
    phase_name: str,
    eval_points: int,
    ready: bool,
) -> dict[str, Any]:
    if ready:
        test_metrics = dict(baseline_payload.get("test_metrics", {}))
        status = "ok"
    else:
        test_metrics = {}
        status = "not_ready"
    return {
        "budget": int(requested_budget),
        "variant": str(variant_name),
        "train_rmse": None,
        "train_mae": None,
        "train_max_abs": None,
        "test_rmse_mev": test_metrics.get("rmse_mev"),
        "test_max_abs_mev": test_metrics.get("max_abs_mev"),
        "completion_hidden_rmse_mev": None,
        "completion_hidden_mae_mev": None,
        "completion_hidden_max_abs_mev": None,
        "completion_hidden_relative_rmse": None,
        "completion_hidden_mape": None,
        "grid_hidden_rmse_mev": None,
        "grid_hidden_max_abs_mev": None,
        "grid_full_rmse_mev": None,
        "grid_full_max_abs_mev": None,
        "grid_roundtrip_rmse": None,
        "grid_roundtrip_max_abs": None,
        "elapsed_sec": float(baseline_payload.get("elapsed_sec", float("nan"))),
        "history_len": None,
        "completed_min_hartree": None,
        "completed_max_hartree": None,
        "info": {
            **dict(baseline_payload.get("info", {})),
            "required_unique_queries": int(baseline_payload.get("unique_queries", 0)),
            "total_queries": int(baseline_payload.get("total_queries", 0)),
            "storage": int(baseline_payload.get("storage", 0)),
            "rms_random_mev": baseline_payload.get("rms_random_mev"),
        },
        "baseline_family": "authors_baseline",
        "phase": str(phase_name),
        "runner": "authors_baseline",
        "model": "authors_baseline",
        "status": status,
        "observed_points": int(baseline_payload.get("unique_queries", 0)),
        "train_points": int(baseline_payload.get("unique_queries", 0)),
        "val_points": 0,
        "eval_points": int(eval_points),
        "requested_rank": None,
        "effective_rank": None,
        "rank_label": "x".join(str(int(x)) for x in baseline_payload.get("tt_ranks", [])) or "n/a",
        "params": dict(baseline_payload.get("params", {})),
        "authors_required_unique_queries": int(baseline_payload.get("unique_queries", 0)),
        "authors_total_queries": int(baseline_payload.get("total_queries", 0)),
        "authors_storage": int(baseline_payload.get("storage", 0)),
        "authors_rms_random_mev": baseline_payload.get("rms_random_mev"),
    }


def _authors_budgeted_trace_row(
    *,
    prefix: faithful.SampleTrace,
    shared_trace: faithful.SampleTrace,
    variant_name: str,
    phase_name: str,
    eval_points: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    if str(shared_trace.status) == "converged_before_budget" and int(prefix.unique_queries) >= int(shared_trace.unique_queries):
        status = "converged_within_budget"
    else:
        status = "budget_exhausted"
    return {
        "budget": int(prefix.unique_queries),
        "variant": str(variant_name),
        "train_rmse": None,
        "train_mae": None,
        "train_max_abs": None,
        "test_rmse_mev": None,
        "test_max_abs_mev": None,
        "completion_hidden_rmse_mev": None,
        "completion_hidden_mae_mev": None,
        "completion_hidden_max_abs_mev": None,
        "completion_hidden_relative_rmse": None,
        "completion_hidden_mape": None,
        "grid_hidden_rmse_mev": None,
        "grid_hidden_max_abs_mev": None,
        "grid_full_rmse_mev": None,
        "grid_full_max_abs_mev": None,
        "grid_roundtrip_rmse": None,
        "grid_roundtrip_max_abs": None,
        "elapsed_sec": 0.0,
        "history_len": None,
        "completed_min_hartree": None,
        "completed_max_hartree": None,
        "info": {
            "trace_status": str(shared_trace.status),
            "shared_trace_unique_queries": int(shared_trace.unique_queries),
            "shared_trace_total_queries": int(shared_trace.total_queries),
            "tt_backend_used": str(shared_trace.tt_backend_used),
            "faithful_backend": bool(shared_trace.faithful_backend),
            "note": (
                "BUDGETED_AUTHORS_TRACE_ONLY: this row reports the honest status of the authors-driven online sampler "
                "under the requested unique-query budget. A surrogate/error is only available if the sampler actually "
                "converged within that budget."
            ),
        },
        "baseline_family": "authors_budgeted",
        "phase": str(phase_name),
        "runner": "authors_budgeted",
        "model": "authors_budgeted",
        "status": status,
        "observed_points": int(prefix.unique_queries),
        "train_points": int(prefix.unique_queries),
        "val_points": 0,
        "eval_points": int(eval_points),
        "requested_rank": params.get("rank", params.get("tt_rank")),
        "effective_rank": None,
        "rank_label": _rank_label_from_params(params),
        "params": dict(params),
        "authors_required_unique_queries": (
            int(shared_trace.unique_queries) if str(shared_trace.status) == "converged_before_budget" else None
        ),
        "authors_total_queries": int(shared_trace.total_queries),
        "authors_storage": None,
        "authors_rms_random_mev": None,
    }


def _ensure_authors_baseline(
    summary: dict[str, Any],
    cfg: dict[str, Any],
    faithful_args: dict[str, Any],
    model_run: dict[str, Any] | None = None,
) -> tuple[faithful.AuthorsBaselineResult, dict[str, Any]]:
    cache = summary.setdefault("_authors_baseline_cache", {})
    model_name = "authors_baseline" if model_run is None else str(model_run["name"])
    if model_name in cache:
        entry = cache[model_name]
        return entry["result"], entry["payload"]

    params = {} if model_run is None else dict(model_run.get("params", {}))
    local_args = dict(faithful_args)
    local_args.update(params)
    n_points = int(params.get("n_points", cfg["authors_sampler"]["n_points"]))
    tol = float(params.get("tol", cfg["authors_sampler"]["tol"]))
    tt_backend = str(params.get("tt_backend", cfg["authors_sampler"]["tt_backend"]))
    random_state = int(params.get("random_state", cfg["authors_sampler"]["random_state"]))

    _log(
        f"[control-panel] authors_baseline START model={model_name} "
        f"n_points={n_points} tol={tol:.1e} backend={tt_backend}"
    )
    t0 = time.time()
    baseline = faithful.run_baranov2015_water_baseline(
        local_args,
        n_points=n_points,
        tol=tol,
        tt_backend=tt_backend,
        random_state=random_state,
    )
    elapsed_sec = float(time.time() - t0)
    payload = _baseline_to_summary_payload(
        baseline,
        elapsed_sec=elapsed_sec,
        model_name=model_name,
        params=params,
    )
    cache[model_name] = {"result": baseline, "payload": payload}
    summary.setdefault("authors_baselines", {})[model_name] = payload
    if summary.get("authors_baseline") is None:
        summary["authors_baseline"] = payload
    _log_authors_baseline_summary(baseline)
    return baseline, payload


def _run_single_model_run(
    *,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    faithful_args: dict[str, Any],
    phase_name: str,
    model_run: dict[str, Any],
    shared_trace: faithful.SampleTrace,
    prefix: faithful.SampleTrace,
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    phase_label = f"[control-panel] {phase_name}"
    runner = str(model_run["runner"])
    name = str(model_run["name"])
    family = (
        "authors_budgeted"
        if runner == "authors_budgeted"
        else (
            "authors_baseline"
            if runner in {"authors_baseline", "authors_full_baseline"}
            else (runner if runner == "approximateLOO_masked" else f"fixed_mask:{str(model_run['model']).lower()}")
        )
    )
    _log_model_start(
        phase_label,
        variant=name,
        family=family,
        budget=int(prefix.unique_queries),
    )
    eval_points = int(len(np.asarray(eval_payload["test_points"], dtype=float)))

    if runner == "authors_budgeted":
        row = _authors_budgeted_trace_row(
            prefix=prefix,
            shared_trace=shared_trace,
            variant_name=name,
            phase_name=phase_name,
            eval_points=eval_points,
            params=dict(model_run.get("params", {})),
        )
        if row["status"] == "converged_within_budget":
            baseline_result, baseline_payload = _ensure_authors_baseline(summary, cfg, faithful_args, model_run)
            row = _authors_baseline_to_row(
                baseline_payload=baseline_payload,
                requested_budget=int(prefix.unique_queries),
                variant_name=name,
                phase_name=phase_name,
                eval_points=eval_points,
                ready=True,
            )
            row["runner"] = "authors_budgeted"
            row["baseline_family"] = "authors_budgeted"
            row["status"] = "converged_within_budget"
        _log_completion_summary(phase_label, row)
        return row

    if runner in {"authors_baseline", "authors_full_baseline"}:
        baseline_result, baseline_payload = _ensure_authors_baseline(summary, cfg, faithful_args, model_run)
        ready = int(prefix.unique_queries) >= int(baseline_result.unique_queries)
        row = _authors_baseline_to_row(
            baseline_payload=baseline_payload,
            requested_budget=int(prefix.unique_queries),
            variant_name=name,
            phase_name=phase_name,
            eval_points=eval_points,
            ready=ready,
        )
        _log_completion_summary(phase_label, row)
        return row

    if runner == "approximateLOO_masked":
        kwargs = dict(model_run.get("params", {}))
        kwargs.update(eval_payload)
        _log(
            f"{phase_label} model={name} runner=approximateLOO_masked "
            f"rank={kwargs.get('rank')} strict_rank={kwargs.get('strict_rank')} "
            f"allow_rank_growth={kwargs.get('allow_rank_growth')} "
            f"steps={kwargs.get('number_of_steps')} lambda={kwargs.get('lambda_all')} "
            f"val={kwargs.get('validation_size')}"
        )
        result = faithful.run_completion_on_author_samples(prefix, kwargs)
        row = _annotate_result_row(
            {
                **_completion_row(prefix.unique_queries, name, result),
                "baseline_family": family,
            },
            phase_name=phase_name,
            model_run=model_run,
            prefix=prefix,
            eval_points=eval_points,
        )
        _log_completion_summary(phase_label, row)
        return row

    observed_tensor_scaled, mask, reference_h = _scale_trace_observations_for_completion(prefix)
    params = dict(model_run.get("params", {}))
    params.setdefault("verbose", True)
    t0 = time.time()
    try:
        model = _instantiate_fixed_mask_model(
            {
                "model": str(model_run["model"]),
                "params": params,
            },
            tuple(prefix.shape),
        )
        result = model.fit_transform(observed_tensor=observed_tensor_scaled, mask=mask, full_tensor=None)
        completed_tensor_h = faithful._from_relative_mev_with_reference(np.asarray(result.tensor, dtype=float), reference_h)
        row = _annotate_result_row(
            _other_baseline_row(
                prefix_budget=prefix.unique_queries,
                baseline_name=name,
                completed_tensor=completed_tensor_h,
                trace=prefix,
                elapsed_sec=float(time.time() - t0),
                info={
                    **({} if result.info is None else dict(result.info)),
                    "completion_value_mode": "relative_mev_observed_min",
                    "reference_energy_hartree": float(reference_h),
                    "model": str(model_run["model"]),
                    "params": params,
                },
                eval_payload=eval_payload,
                baseline_family=family,
            ),
            phase_name=phase_name,
            model_run=model_run,
            prefix=prefix,
            eval_points=eval_points,
        )
        _log_completion_summary(phase_label, row)
        return row
    except Exception as exc:
        row = _annotate_result_row(
            _error_row(
                budget=int(prefix.unique_queries),
                variant=name,
                family=family,
                error=exc,
                elapsed_sec=float(time.time() - t0),
            ),
            phase_name=phase_name,
            model_run=model_run,
            prefix=prefix,
            eval_points=eval_points,
        )
        _log_completion_summary(phase_label, row)
        return row


def _run_model_runs_on_prefix(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    trace: faithful.SampleTrace,
    faithful_args: dict[str, Any],
    prefix_budget: int,
    *,
    phase_name: str,
) -> list[dict[str, Any]]:
    model_runs = _enabled_model_runs(cfg)
    if not model_runs:
        return []
    prefix = _prefix_trace(trace, int(prefix_budget))
    eval_payload = _make_test_payload(cfg, faithful_args, prefix)
    iterator = model_runs
    if bool(cfg["paper_args"].get("show_tqdm", False)) and len(model_runs) > 1:
        iterator = tqdm(
            model_runs,
            desc=f"{phase_name} models @B={prefix.unique_queries}",
            unit="model",
            leave=False,
        )
    rows: list[dict[str, Any]] = []
    for item in iterator:
        rows.append(
            _run_single_model_run(
                cfg=cfg,
                summary=summary,
                faithful_args=faithful_args,
                phase_name=phase_name,
                model_run=item,
                shared_trace=trace,
                prefix=prefix,
                eval_payload=eval_payload,
            )
        )
    return rows


def _run_completion_variants_on_prefix(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    trace: faithful.SampleTrace,
    faithful_args: dict[str, Any],
    prefix_budget: int,
) -> list[dict[str, Any]]:
    return _run_model_runs_on_prefix(
        cfg,
        summary,
        trace,
        faithful_args,
        prefix_budget=prefix_budget,
        phase_name="completion_on_samples",
    )


def _run_fixed_mask_baselines_on_prefix(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    trace: faithful.SampleTrace,
    faithful_args: dict[str, Any],
    prefix_budget: int,
) -> list[dict[str, Any]]:
    return _run_model_runs_on_prefix(
        cfg,
        summary,
        trace,
        faithful_args,
        prefix_budget=prefix_budget,
        phase_name="fixed_mask_baselines",
    )


def _results_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if summary.get("authors_baseline_run") is not None:
        rows.append(dict(summary["authors_baseline_run"]))
    rows.extend(dict(item) for item in summary.get("completion_on_samples", []))
    rows.extend(dict(item) for item in summary.get("budget_sweep", []))
    return rows


def _results_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "settings": dict(summary.get("settings", {})),
        "authors_baseline": summary.get("authors_baseline"),
        "authors_baselines": dict(summary.get("authors_baselines", {})),
        "rows": _results_rows(summary),
    }


def _results_markdown(summary: dict[str, Any]) -> str:
    settings = summary["settings"]
    rows = _results_rows(summary)
    lines: list[str] = []
    lines.append("# Faithful Control Panel Results")
    lines.append("")
    lines.append("## Settings")
    lines.append(f"- Run tag: `{settings['run_tag']}`")
    lines.append(f"- Experiment mode: `{settings['experiment_mode']}`")
    lines.append(f"- Tensor shape: `{settings['n_points']}^{settings['tensor_dim']}`")
    lines.append(f"- Total tensor points: `{settings['total_tensor_points']}`")
    lines.append(f"- Author sampler backend: `{settings['tt_backend']}`")
    lines.append(f"- TTML env: `{settings['ttml_env_name']}`")
    lines.append(f"- Author tol: `{settings['tol']}`")
    lines.append(f"- Author budget mode: `{settings['authors_budget_mode']}` -> `{settings['authors_budget_resolved']}`")
    lines.append(
        f"- Completion prefix budget mode: `{settings['completion_prefix_budget_mode']}` "
        f"-> `{settings['completion_prefix_budget_resolved']}`"
    )
    lines.append(f"- Budget sweep mode: `{settings['budget_sweep_mode']}` -> `{settings['budget_sweep_resolved']}`")
    lines.append(f"- Random test points: `{settings['evaluation_random_test_points']}`")
    lines.append(f"- Selected model runs: `{settings['selected_model_runs']}`")
    lines.append("")
    if summary.get("authors_baseline") is not None:
        baseline = summary["authors_baseline"]
        lines.append("## Authors Baseline")
        lines.append(
            f"- `{baseline['model_name']}`: backend=`{baseline['backend_used']}`, "
            f"unique=`{baseline['unique_queries']}`, total=`{baseline['total_queries']}`, "
            f"ranks=`{baseline['tt_ranks']}`, storage=`{baseline['storage']}`, "
            f"off-grid RMSE=`{baseline['test_metrics'].get('rmse_mev')}` meV"
        )
        lines.append("")
    if not rows:
        lines.append("## Results")
        lines.append("")
        lines.append("No model rows were produced.")
        return "\n".join(lines)

    lines.append("## Results")
    phase_order = {"authors_baseline": 0, "completion_on_samples": 1, "budget_sweep": 2, "fixed_mask_baselines": 3}
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            phase_order.get(str(row.get("phase")), 99),
            int(row.get("budget", 0)),
            str(row.get("variant", "")),
        ),
    )
    current_phase = None
    for row in rows_sorted:
        phase = str(row.get("phase", "unknown"))
        if phase != current_phase:
            current_phase = phase
            lines.append("")
            lines.append(f"### {phase}")
            lines.append("")
            lines.append(
                "| budget | model | runner | status | observed | train | val | eval | rank | "
                "train RMSE (meV) | completion hidden RMSE (meV) | off-grid RMSE (meV) | time(s) |"
            )
            lines.append("|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        lines.append(
            f"| {row.get('budget', 'n/a')} | {row.get('variant', 'n/a')} | {row.get('runner', 'n/a')} | "
            f"{row.get('status', 'n/a')} | {row.get('observed_points', 'n/a')} | "
            f"{row.get('train_points', 'n/a')} | {row.get('val_points', 'n/a')} | "
            f"{row.get('eval_points', 'n/a')} | {row.get('rank_label', 'n/a')} | "
            f"{_fmt_metric(row.get('train_rmse'))} | {_fmt_metric(row.get('completion_hidden_rmse_mev'))} | "
            f"{_fmt_metric(row.get('test_rmse_mev'))} | {_fmt_metric(row.get('elapsed_sec'), digits=2)} |"
        )
    return "\n".join(lines)


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    settings = summary["settings"]
    lines.append("# Faithful Control Panel Run")
    lines.append("")
    lines.append(f"- `run_tag`: `{settings['run_tag']}`")
    lines.append(f"- `experiment_mode`: `{settings['experiment_mode']}`")
    lines.append(f"- `n_points`: `{settings['n_points']}`")
    lines.append(f"- `tensor_dim`: `{settings['tensor_dim']}`")
    lines.append(f"- `tol`: `{settings['tol']}`")
    lines.append(f"- `tt_backend`: `{settings['tt_backend']}`")
    lines.append(f"- `ttml_env_name`: `{settings['ttml_env_name']}`")
    lines.append(f"- `water_as_samples`: `{settings['water_as_samples']}`")
    lines.append(f"- `total_tensor_points`: `{settings['total_tensor_points']}`")
    lines.append(f"- `authors_budget_mode`: `{settings['authors_budget_mode']}`")
    lines.append(f"- `authors_budget_resolved`: `{settings['authors_budget_resolved']}`")
    lines.append(f"- `completion_prefix_budget_mode`: `{settings['completion_prefix_budget_mode']}`")
    lines.append(f"- `completion_prefix_budget_resolved`: `{settings['completion_prefix_budget_resolved']}`")
    lines.append(f"- `budget_sweep_mode`: `{settings['budget_sweep_mode']}`")
    lines.append(f"- `budget_sweep_resolved`: `{settings['budget_sweep_resolved']}`")
    lines.append(f"- `evaluation_random_test_points`: `{settings['evaluation_random_test_points']}`")
    lines.append(f"- `evaluation_exact_grid_metrics`: `{settings['evaluation_exact_grid_metrics']}`")
    lines.append(f"- `selected_model_runs`: `{settings['selected_model_runs']}`")
    lines.append("")
    if summary.get("authors_baseline") is not None:
        baseline = summary["authors_baseline"]
        lines.append("## Authors Baseline")
        lines.append("")
        lines.append(f"- model: `{baseline['model_name']}`")
        lines.append(f"- backend: `{baseline['backend_used']}`")
        lines.append(f"- unique queries: `{baseline['unique_queries']}`")
        lines.append(f"- total queries: `{baseline['total_queries']}`")
        lines.append(f"- tt ranks: `{baseline['tt_ranks']}`")
        lines.append(f"- rms_random_mev: `{baseline['rms_random_mev']}`")
        lines.append("")
    if summary.get("collected_trace") is not None:
        trace = summary["collected_trace"]
        lines.append("## Collected Trace")
        lines.append("")
        lines.append(f"- status: `{trace['status']}`")
        lines.append(f"- unique queries: `{trace['unique_queries']}`")
        lines.append(f"- total queries: `{trace['total_queries']}`")
        lines.append(f"- backend used: `{trace['tt_backend_used']}`")
        lines.append("")
    for section_name in ("completion_on_samples", "budget_sweep"):
        rows = summary.get(section_name, [])
        if not rows:
            continue
        lines.append(f"## {section_name}")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['variant']}` [{row['runner']}] @ budget `{row['budget']}` "
                f"status=`{row['status']}` observed=`{row['observed_points']}` "
                f"train=`{row['train_points']}` val=`{row['val_points']}` "
                f"rank=`{row['rank_label']}` completion_hidden_rmse_mev=`{row.get('completion_hidden_rmse_mev')}` "
                f"test_rmse_mev=`{row.get('test_rmse_mev')}` train_rmse=`{row.get('train_rmse')}`"
            )
        lines.append("")
    return "\n".join(lines)


def _summary_for_output(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if not str(k).startswith("_")}


def run_from_file_config() -> dict[str, Any]:
    cfg = _resolve_cfg()
    outputs_dir = Path(cfg["io"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    trace_npz_path = Path(cfg["trace_source"]["npz_path"])
    run_tag = str(cfg["io"]["run_tag"])
    faithful_args = _faithful_args_from_cfg(cfg)
    total_tensor_points = _tensor_total_points_from_cfg(cfg)
    author_budget_resolved = _resolve_author_budget(cfg)
    prefix_budget_resolved = _resolve_prefix_budget(cfg)
    sweep_budgets_resolved = _resolve_sweep_budgets(cfg)
    model_runs = _enabled_model_runs(cfg)
    model_run_names = [str(item["name"]) for item in model_runs]

    json_path = outputs_dir / f"{run_tag}.json"
    md_path = outputs_dir / f"{run_tag}.md"
    results_json_path = outputs_dir / f"{run_tag}_results.json"
    results_md_path = outputs_dir / f"{run_tag}_results.md"

    summary: dict[str, Any] = {
        "settings": {
            "run_tag": run_tag,
            "experiment_mode": str(cfg.get("experiment", {}).get("mode", "legacy")),
            "n_points": int(cfg["authors_sampler"]["n_points"]),
            "tensor_dim": int(cfg["paper_args"]["water_as_dim"]),
            "tol": float(cfg["authors_sampler"]["tol"]),
            "tt_backend": str(cfg["authors_sampler"]["tt_backend"]),
            "ttml_env_name": str(cfg["authors_sampler"]["ttml_env_name"]),
            "water_as_samples": int(cfg["paper_args"]["water_as_samples"]),
            "fixed_mask_baselines": _model_run_names_by_runner(cfg, "fixed_mask_baseline"),
            "total_tensor_points": int(total_tensor_points),
            "authors_budget_mode": str(cfg["authors_sampler"]["budget_mode"]),
            "authors_budget_resolved": int(author_budget_resolved),
            "completion_prefix_budget_mode": str(cfg["completion_on_samples"]["prefix_budget_mode"]),
            "completion_prefix_budget_resolved": int(prefix_budget_resolved),
            "budget_sweep_mode": str(cfg["budget_sweep"]["budget_mode"]),
            "budget_sweep_resolved": [int(x) for x in sweep_budgets_resolved],
            "evaluation_random_test_points": int(cfg["evaluation"]["random_test_points"]),
            "evaluation_exact_grid_metrics": bool(cfg["evaluation"]["compute_exact_grid_metrics"]),
            "selected_model_runs": [str(x) for x in model_run_names],
            "model_runs": [dict(item) for item in model_runs],
            "outputs_dir": str(outputs_dir),
        },
        "authors_baseline": None,
        "authors_baselines": {},
        "authors_baseline_run": None,
        "collected_trace": None,
        "completion_on_samples": [],
        "budget_sweep": [],
        "artifacts": {
            "summary_json": str(json_path),
            "summary_md": str(md_path),
            "results_json": str(results_json_path),
            "results_md": str(results_md_path),
        },
    }

    _log(
        f"[control-panel] START run_tag={run_tag} n_points={cfg['authors_sampler']['n_points']} "
        f"tol={cfg['authors_sampler']['tol']:.1e} backend={cfg['authors_sampler']['tt_backend']} "
        f"tensor_points={total_tensor_points} trace_source={cfg['trace_source']['mode']}"
    )
    if "experiment" in cfg:
        _log(
            f"[control-panel] experiment mode={cfg['experiment']['mode']} "
            f"budget_mode={cfg['experiment']['budget_mode']} "
            f"budget={cfg['experiment']['budget']} "
            f"budget_fraction={cfg['experiment']['budget_fraction']} "
            f"sweep_budgets={cfg['experiment']['budgets']} "
            f"sweep_budget_fractions={cfg['experiment']['budget_fractions']}"
        )
    _log(
        f"[control-panel] run_flags authors_baseline={cfg['run']['authors_baseline']} "
        f"collect_samples={cfg['run']['collect_samples']} "
        f"completion_on_samples={cfg['run']['completion_on_samples']} "
        f"budget_sweep={cfg['run']['budget_sweep']}"
    )
    _log(
        f"[control-panel] budget authors_sampler mode={cfg['authors_sampler']['budget_mode']} "
        f"resolved={author_budget_resolved} "
        f"raw_abs={cfg['authors_sampler']['unique_budget']} "
        f"raw_frac={cfg['authors_sampler']['unique_budget_fraction']}"
    )
    _log(
        f"[control-panel] budget completion_on_samples mode={cfg['completion_on_samples']['prefix_budget_mode']} "
        f"resolved={prefix_budget_resolved} "
        f"raw_abs={cfg['completion_on_samples']['prefix_budget']} "
        f"raw_frac={cfg['completion_on_samples']['prefix_budget_fraction']}"
    )
    _log(
        f"[control-panel] budget budget_sweep mode={cfg['budget_sweep']['budget_mode']} "
        f"resolved={sweep_budgets_resolved}"
    )
    _log(f"[control-panel] selected model_runs={model_run_names}")
    _log_model_runs_catalog(model_runs)

    t0 = time.time()
    shared_trace = _shared_trace_from_cfg(cfg, faithful_args)
    if shared_trace is not None and bool(cfg["io"]["save_trace_npz"]) and str(cfg["trace_source"]["mode"]).lower() == "fresh":
        _save_sample_trace_npz(trace_npz_path, shared_trace)
        summary["artifacts"]["trace_npz"] = str(trace_npz_path)
        _log(f"[control-panel] saved trace -> {trace_npz_path}")

    if cfg["run"].get("authors_baseline", False):
        baseline_result, baseline_payload = _ensure_authors_baseline(summary, cfg, faithful_args)
        summary["authors_baseline_run"] = _authors_baseline_to_row(
            baseline_payload=baseline_payload,
            requested_budget=int(baseline_result.unique_queries),
            variant_name=str(baseline_payload["model_name"]),
            phase_name="authors_baseline",
            eval_points=int(cfg["evaluation"]["random_test_points"]),
            ready=True,
        )

    if cfg["run"].get("collect_samples", False):
        if shared_trace is None:
            raise RuntimeError("collect_samples requested, but no shared trace is available")
        collect_prefix = _prefix_trace(shared_trace, author_budget_resolved)
        summary["collected_trace"] = {
            "status": str(collect_prefix.status),
            "unique_queries": int(collect_prefix.unique_queries),
            "total_queries": int(collect_prefix.total_queries),
            "shape": [int(x) for x in collect_prefix.shape],
            "tt_backend_requested": str(collect_prefix.tt_backend_requested),
            "tt_backend_used": str(collect_prefix.tt_backend_used),
            "faithful_backend": bool(collect_prefix.faithful_backend),
            "info": dict(collect_prefix.info),
        }
        _log_trace_summary("collect_samples", collect_prefix)

    if cfg["run"].get("completion_on_samples", False):
        if shared_trace is None:
            raise RuntimeError("completion_on_samples requested, but no shared trace is available")
        prefix_budget = int(prefix_budget_resolved)
        summary["completion_on_samples"] = _run_model_runs_on_prefix(
            cfg,
            summary,
            shared_trace,
            faithful_args,
            prefix_budget=prefix_budget,
            phase_name="completion_on_samples",
        )

    if cfg["run"].get("budget_sweep", False):
        if shared_trace is None:
            raise RuntimeError("budget_sweep requested, but no shared trace is available")
        budgets = [int(x) for x in sweep_budgets_resolved]
        if cfg["budget_sweep"].get("run_authors_baseline_if_trace_converged", False):
            if summary["authors_baseline"] is None and shared_trace.status == "converged_before_budget":
                _ensure_authors_baseline(summary, cfg, faithful_args)
        sweep_rows: list[dict[str, Any]] = []
        budget_iter = budgets
        if bool(cfg["paper_args"].get("show_tqdm", False)) and len(budgets) > 1:
            budget_iter = tqdm(budgets, desc="budget_sweep budgets", unit="budget", leave=False)
        for budget in budget_iter:
            sweep_rows.extend(
                _run_model_runs_on_prefix(
                    cfg,
                    summary,
                    shared_trace,
                    faithful_args,
                    prefix_budget=int(budget),
                    phase_name="budget_sweep",
                )
            )
        summary["budget_sweep"] = sweep_rows

    summary["elapsed_sec"] = float(time.time() - t0)

    summary_out = _summary_for_output(summary)
    results_payload = _results_payload(summary_out)

    if bool(cfg["io"]["save_summary_json"]):
        json_path.write_text(json.dumps(summary_out, default=_json_default, indent=2), encoding="utf-8")
        _log(f"[control-panel] wrote summary json -> {json_path}")
    if bool(cfg["io"]["save_summary_md"]):
        md_path.write_text(_markdown_summary(summary_out), encoding="utf-8")
        _log(f"[control-panel] wrote summary md -> {md_path}")
    if bool(cfg["io"].get("save_results_json", True)):
        results_json_path.write_text(json.dumps(results_payload, default=_json_default, indent=2), encoding="utf-8")
        _log(f"[control-panel] wrote results json -> {results_json_path}")
    if bool(cfg["io"].get("save_results_md", True)):
        results_md_path.write_text(_results_markdown(summary_out), encoding="utf-8")
        _log(f"[control-panel] wrote results md -> {results_md_path}")
    if bool(cfg["io"]["pretty_print_summary"]):
        print(json.dumps(summary_out, default=_json_default, indent=2))
    _log_final_summary(summary_out)
    _log(f"[control-panel] FINISH total_elapsed={_fmt_metric(summary_out['elapsed_sec'], digits=2)}s")
    return summary_out


if __name__ == "__main__":
    run_from_file_config()
