from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tmp_cache = Path("/tmp/codex_cache")
tmp_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(tmp_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_cache))

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from common import all_metrics

from as_tt_water_experiments import run_as_tt_water_dual_experiments as dual
from as_tt_water_experiments import run_as_tt_water_experiments as water
from baranov2015_as_tt_experiments.paper_as_tt_baseline import (
    PaperASTTResult,
    evaluate_tt_chebyshev,
    load_or_generate_full_value_tensor,
    prepare_planar_identity_context,
    prepare_planar_water_context,
    run_paper_as_tt,
)


RUN_FROM_FILE_CONFIG = True
FILE_RUN_CONFIG = {
    "data_root": "baranov2015_as_tt_experiments/data/as_tt_water",
    "geometry_xyz": "baranov2015_as_tt_experiments/data/as_tt_water/water.xyz",
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
    # Paper-like water setup from Baranov & Oseledets (2015):
    # O fixed at origin, hydrogens constrained to a plane, then AS rotation.
    "water_coordinate_unit": "Bohr",
    "water_as_dim": 4,
    "water_as_sigma2": 0.1,
    "water_as_samples": 256,
    "water_as_random_state": 1729,
    "cheb_interval": [-0.3, 0.3],
    "cheb_points_list": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # Table I reference counts from the paper.
    "exact_eval_budget": {
        "2": 16,
        "3": 81,
        "4": 321,
        "5": 865,
        "6": 2145,
        "7": 3901,
        "8": 6925,
        "9": 10644,
        "10": 18202,
    },
    # Author AS+TT (paper-like TT-cross)
    "author_rank_cap": 12,
    "author_tol": 1e-5,
    "author_max_iter": 30,
    "author_random_state": 1729,
    "author_rms_probe_points": 1000,
    "author_rms_seed": 2025,
    "author_reuse_query_cache": True,
    "author_query_cache_dir": "baranov2015_as_tt_experiments/results/as_tt_water/query_cache",
    "paper_plots_dir": "baranov2015_as_tt_experiments/results/as_tt_water/paper_plots",
    # Experiment phases
    "run_mode": "both",  # paper | completion | both
    "paper_budget_mode": "unlimited",  # unlimited | paper_budget
    "paper_run_table_i": True,
    "paper_run_table_ii": True,
    "paper_run_fig1": True,
    "paper_table_i_points": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "paper_table_ii_thresholds": [1e-3, 1e-5, 1e-7],
    "paper_table_ii_points": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "paper_table_ii_rank_cap": 128,
    "paper_table_ii_no_as_points": 20,
    "paper_fig1_domain": [-0.15, 0.15],
    "paper_fig1_grid_size": 61,
    "paper_fig1_pairs": [(0, 1), (1, 2), (2, 3)],
    "shared_query": {
        "enabled": True,
        "splits": 1,
        "min_eval_points": 64,
        "budget_variants": [
            {"name": "frac0p2", "mode": "fraction", "fraction": 0.2},
            {"name": "full", "mode": "full_tensor"},
        ],
        "baselines": "tt_als,tucker,cp_wopt,halrtc,cheb_auto",
    },
    # Shared completion baseline params
    "rank": 12,
    "max_iter": 120,
    "tol": 1e-5,
    "cheb_ranks": [20, 30],
    "cheb_steps": 8,
    "cheb_tol_for_step": 1e-5,
    "cheb_lambda_all": 0.0,
    "cheb_lambda_all_list": [0.0],
    "cheb_validation_size_list": [0.1],
    "cheb_tolerance": 2000,
    "cheb_validation_size": 0.1,
    "cheb_source_file": "completion_linalg_tensor_all.py",
    "disable_joblib_multiprocessing": False,
    "random_state": 1000,
    "show_tqdm": True,
    "verbose": True,
    "output_json": "baranov2015_as_tt_experiments/results/as_tt_water/baranov2015_water_results.json",
    "output_md": "baranov2015_as_tt_experiments/results/as_tt_water/baranov2015_water_results.md",
}


def _log_line(msg: str, use_tqdm: bool = False) -> None:
    if use_tqdm and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _resolve_cfg() -> dict[str, Any]:
    if not RUN_FROM_FILE_CONFIG:
        raise RuntimeError("Only file-config mode is supported for this runner.")
    cfg = dict(FILE_RUN_CONFIG)
    cfg["data_root"] = water.resolve_project_path(cfg["data_root"])
    cfg["geometry_xyz"] = water.resolve_project_path(cfg["geometry_xyz"])
    cfg["author_query_cache_dir"] = water.resolve_project_path(cfg["author_query_cache_dir"])
    cfg["paper_plots_dir"] = water.resolve_project_path(cfg["paper_plots_dir"])
    cfg["output_json"] = water.resolve_project_path(cfg["output_json"])
    cfg["output_md"] = water.resolve_project_path(cfg["output_md"])
    cfg["cheb_points_list"] = [int(x) for x in cfg["cheb_points_list"]]
    cfg["paper_table_i_points"] = [int(x) for x in cfg.get("paper_table_i_points", cfg["cheb_points_list"])]
    cfg["paper_table_ii_points"] = [int(x) for x in cfg.get("paper_table_ii_points", [])]
    cfg["paper_table_ii_thresholds"] = [float(x) for x in cfg.get("paper_table_ii_thresholds", [1e-3, 1e-5, 1e-7])]
    cfg["paper_table_ii_no_as_points"] = int(cfg.get("paper_table_ii_no_as_points", 20))
    cfg["paper_table_ii_rank_cap"] = int(cfg.get("paper_table_ii_rank_cap", cfg["author_rank_cap"]))
    cfg["paper_fig1_domain"] = [float(cfg["paper_fig1_domain"][0]), float(cfg["paper_fig1_domain"][1])]
    cfg["paper_fig1_grid_size"] = int(cfg.get("paper_fig1_grid_size", 61))
    cfg["paper_fig1_pairs"] = [tuple(int(v) for v in pair) for pair in cfg.get("paper_fig1_pairs", [(0, 1), (1, 2), (2, 3)])]
    cfg["cheb_interval"] = [float(cfg["cheb_interval"][0]), float(cfg["cheb_interval"][1])]
    cfg["exact_eval_budget"] = {str(k): int(v) for k, v in dict(cfg["exact_eval_budget"]).items()}
    cfg["cheb_ranks"] = [int(x) for x in cfg["cheb_ranks"]]
    cfg["cheb_lambda_all_list"] = [float(x) for x in cfg.get("cheb_lambda_all_list", [cfg.get("cheb_lambda_all", 0.0)])]
    cfg["cheb_validation_size_list"] = [
        float(x) for x in cfg.get("cheb_validation_size_list", [cfg.get("cheb_validation_size", 0.1)])
    ]
    cfg["shared_query"] = dict(cfg["shared_query"])
    if "budget_variants" not in cfg["shared_query"] or not cfg["shared_query"]["budget_variants"]:
        cfg["shared_query"]["budget_variants"] = [
            {"name": "frac0p2", "mode": "fraction", "fraction": 0.2},
            {"name": "full", "mode": "full_tensor"},
        ]
    return cfg


def _paper_phase_enabled(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("run_mode", "both")).lower() in {"both", "paper"}


def _shared_phase_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg["shared_query"].get("enabled", True)) and str(cfg.get("run_mode", "both")).lower() in {
        "both",
        "completion",
    }


def _mape_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(np.asarray(y_true, dtype=float))
    valid = denom > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((np.asarray(y_pred, dtype=float)[valid] - np.asarray(y_true, dtype=float)[valid]) / denom[valid])))


def _to_relative_mev_with_reference(tensor_h: np.ndarray, reference_min_hartree: float) -> np.ndarray:
    return (np.asarray(tensor_h, dtype=float) - float(reference_min_hartree)) * water.HARTREE_TO_MEV


@dataclass
class PaperAuthorRow:
    cheb_points: int
    status: str
    max_rank: int | None
    rms_random_mev: float | None
    unique_queries: int
    total_queries: int
    paper_reference_unique_queries: int | None
    truncated_by_budget: bool
    elapsed_sec: float
    query_source: str
    query_cache_path: str
    error: str | None = None


@dataclass
class SharedCompletionRow:
    cheb_points: int
    budget_name: str
    budget_mode: str
    budget_fraction: float | None
    split: int
    baseline: str
    status: str
    requested_unique_budget: int
    observed_points: int
    eval_points: int
    metric_scope: str
    rmse_mev: float | None
    mae_mev: float | None
    max_abs_error_mev: float | None
    relative_rmse: float | None
    mape: float | None
    author_rms_random_mev: float | None
    author_unique_queries: int
    author_total_queries: int
    elapsed_sec: float
    query_source: str
    query_cache_path: str
    error: str | None = None


@dataclass
class PaperThresholdRow:
    threshold: float
    max_rank_by_points: dict[int, int | None]
    no_as_value_tensor_rank: int | None
    elapsed_sec: float
    status: str
    error: str | None = None


def _reference_budget(cfg: dict[str, Any], n_points: int) -> int | None:
    value = dict(cfg.get("exact_eval_budget", {})).get(str(int(n_points)))
    return None if value is None else int(value)


def _resolve_author_budget(cfg: dict[str, Any], n_points: int) -> tuple[int | None, str]:
    mode = str(cfg.get("paper_budget_mode", "unlimited")).lower()
    if mode == "unlimited":
        return None, "paper_unbounded"
    if mode == "paper_budget":
        return _reference_budget(cfg, n_points), "paper_budget"
    raise ValueError(f"Unsupported paper_budget_mode: {cfg.get('paper_budget_mode')!r}")


def _resolve_shared_budget_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in cfg["shared_query"]["budget_variants"]:
        item = dict(raw)
        mode = str(item.get("mode", "fraction")).lower()
        name = str(item.get("name", mode))
        fraction = None if item.get("fraction") is None else float(item.get("fraction"))
        if mode == "fraction":
            if fraction is None:
                raise ValueError(f"Budget variant {name!r} requires fraction")
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"Budget fraction must be in (0, 1], got {fraction}")
        elif mode == "full_tensor":
            fraction = 1.0
        elif mode == "paper_budget":
            fraction = None
        else:
            raise ValueError(f"Unsupported shared-query budget mode: {mode!r}")
        out.append({"name": name, "mode": mode, "fraction": fraction})
    return out


def _shared_requested_unique_budget(
    cfg: dict[str, Any],
    *,
    n_total: int,
    n_points: int,
    variant: dict[str, Any],
) -> int:
    mode = str(variant["mode"])
    if mode == "full_tensor":
        return int(n_total)
    if mode == "paper_budget":
        ref = _reference_budget(cfg, n_points)
        if ref is None:
            raise KeyError(f"Missing exact_eval_budget for n={n_points}")
        return int(min(int(ref), int(n_total)))
    if mode == "fraction":
        fraction = float(variant["fraction"])
        return int(
            dual._choose_observed_count(
                n_total=int(n_total),
                policy="fraction",
                observed_fraction=float(fraction),
                budget_requested=int(np.ceil(fraction * n_total)),
                min_eval_points=int(cfg["shared_query"].get("min_eval_points", 1)),
            )
        )
    raise ValueError(f"Unsupported budget mode: {mode!r}")


def _query_cache_path(
    cfg: dict[str, Any],
    *,
    context_cache_tag: str,
    n_points: int,
    budget_name: str,
    split: int,
) -> Path:
    interval = cfg["cheb_interval"]
    tag = (
        f"{context_cache_tag}_cheb{int(n_points)}"
        f"_a{dual._format_lambda_tag(float(interval[0]))}"
        f"_b{dual._format_lambda_tag(float(interval[1]))}"
        f"_{budget_name}_split{int(split)}"
        f"_rank{int(cfg['author_rank_cap'])}"
        f"_tol{dual._format_lambda_tag(float(cfg['author_tol']))}"
        f"_seed{int(cfg['author_random_state'])}"
    )
    return Path(cfg["author_query_cache_dir"]) / f"{tag}.npz"


def _load_cached_query_run(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    queried_flat = np.asarray(payload["queried_flat_indices"], dtype=int)
    shape = tuple(int(x) for x in np.asarray(payload["shape"], dtype=int))
    approx_tensor = np.asarray(payload["approx_tensor_hartree"], dtype=float)
    meta_json = str(np.asarray(payload["meta_json"]).item())
    meta = json.loads(meta_json)
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    if queried_flat.size:
        mask[queried_flat] = True
    return mask.reshape(shape), approx_tensor, meta


def _save_query_run(
    path: Path,
    *,
    shape: tuple[int, ...],
    result: PaperASTTResult,
    elapsed_sec: float,
    budget_name: str,
    requested_unique_budget: int | None,
    approx_tensor_hartree: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "elapsed_sec": float(elapsed_sec),
        "budget_name": str(budget_name),
        "requested_unique_budget": None if requested_unique_budget is None else int(requested_unique_budget),
        "total_query_count": int(result.total_query_count),
        "unique_query_count": int(result.unique_query_count),
        "converged": bool(result.converged),
        "truncated_by_budget": bool(result.truncated_by_budget),
        "max_rank": None if result.max_rank is None else int(result.max_rank),
        "rms_random": None if result.rms_random is None else float(result.rms_random),
        "info": result.info or {},
    }
    np.savez_compressed(
        path,
        queried_flat_indices=np.asarray(result.queried_flat_indices, dtype=int),
        shape=np.asarray(shape, dtype=int),
        approx_tensor_hartree=np.asarray(approx_tensor_hartree if approx_tensor_hartree is not None else np.zeros(shape), dtype=float),
        meta_json=np.asarray(json.dumps(meta)),
    )


def _metric_payload(full_tensor_mev: np.ndarray, pred_mev: np.ndarray, observed_mask: np.ndarray) -> tuple[str, dict[str, float], float]:
    eval_mask = ~observed_mask
    if np.any(eval_mask):
        metrics = all_metrics(full_tensor_mev, pred_mev, eval_mask)
        y_true = full_tensor_mev[eval_mask]
        y_pred = pred_mev[eval_mask]
        scope = "hidden_only"
    else:
        mask_all = np.ones_like(observed_mask, dtype=bool)
        metrics = all_metrics(full_tensor_mev, pred_mev, mask_all)
        y_true = full_tensor_mev.reshape(-1)
        y_pred = pred_mev.reshape(-1)
        scope = "full_tensor"
    return scope, metrics, _mape_ratio(y_true, y_pred)


def _author_row_from_result(
    *,
    n_points: int,
    result: PaperASTTResult,
    elapsed_sec: float,
    query_source: str,
    query_cache_path: Path,
    paper_reference_unique_queries: int | None,
    status: str = "ok",
    error: str | None = None,
) -> PaperAuthorRow:
    return PaperAuthorRow(
        cheb_points=int(n_points),
        status=str(status),
        max_rank=None if result.max_rank is None else int(result.max_rank),
        rms_random_mev=None if result.rms_random is None else float(result.rms_random),
        unique_queries=int(result.unique_query_count),
        total_queries=int(result.total_query_count),
        paper_reference_unique_queries=None if paper_reference_unique_queries is None else int(paper_reference_unique_queries),
        truncated_by_budget=bool(result.truncated_by_budget),
        elapsed_sec=float(elapsed_sec),
        query_source=str(query_source),
        query_cache_path=str(query_cache_path),
        error=error,
    )


def _completion_row_from_prediction(
    *,
    n_points: int,
    budget_name: str,
    budget_mode: str,
    budget_fraction: float | None,
    split: int,
    baseline: str,
    pred_mev: np.ndarray,
    full_tensor_mev: np.ndarray,
    observed_mask: np.ndarray,
    requested_unique_budget: int,
    author_rms_random_mev: float | None,
    author_unique_queries: int,
    author_total_queries: int,
    elapsed_sec: float,
    query_source: str,
    query_cache_path: Path,
    status: str = "ok",
    error: str | None = None,
) -> SharedCompletionRow:
    scope, metrics, mape = _metric_payload(full_tensor_mev, pred_mev, observed_mask)
    eval_points = int(np.sum(~observed_mask)) if scope == "hidden_only" else int(np.prod(observed_mask.shape))
    return SharedCompletionRow(
        cheb_points=int(n_points),
        budget_name=str(budget_name),
        budget_mode=str(budget_mode),
        budget_fraction=None if budget_fraction is None else float(budget_fraction),
        split=int(split),
        baseline=str(baseline),
        status=str(status),
        requested_unique_budget=int(requested_unique_budget),
        observed_points=int(np.sum(observed_mask)),
        eval_points=int(eval_points),
        metric_scope=str(scope),
        rmse_mev=float(metrics["rmse"]) if np.isfinite(metrics["rmse"]) else None,
        mae_mev=float(metrics["mae"]) if np.isfinite(metrics["mae"]) else None,
        max_abs_error_mev=float(metrics["max_abs_error"]) if np.isfinite(metrics["max_abs_error"]) else None,
        relative_rmse=float(metrics["relative_rmse"]) if np.isfinite(metrics["relative_rmse"]) else None,
        mape=float(mape) if np.isfinite(mape) else None,
        author_rms_random_mev=None if author_rms_random_mev is None else float(author_rms_random_mev),
        author_unique_queries=int(author_unique_queries),
        author_total_queries=int(author_total_queries),
        elapsed_sec=float(elapsed_sec),
        query_source=str(query_source),
        query_cache_path=str(query_cache_path),
        error=error,
    )


def _plot_fig1(
    cfg: dict[str, Any],
    *,
    context,
    result: PaperASTTResult,
    n_points: int,
) -> list[str]:
    if not result.coeff_tt_cores:
        return []

    plot_dir = Path(cfg["paper_plots_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = float(cfg["paper_fig1_domain"][0]), float(cfg["paper_fig1_domain"][1])
    grid_size = int(cfg["paper_fig1_grid_size"])
    xs = np.linspace(lo, hi, grid_size)
    ys = np.linspace(lo, hi, grid_size)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    out_paths: list[str] = []

    for pair in cfg["paper_fig1_pairs"]:
        i, j = int(pair[0]), int(pair[1])
        if i < 0 or j < 0 or i >= context.as_dim or j >= context.as_dim or i == j:
            continue
        Z = np.zeros_like(X, dtype=float)
        for r in range(grid_size):
            for c in range(grid_size):
                point = np.zeros(context.as_dim, dtype=float)
                point[i] = float(X[r, c])
                point[j] = float(Y[r, c])
                Z[r, c] = float(
                    evaluate_tt_chebyshev(
                        result.coeff_tt_cores,
                        point,
                        a=float(cfg["cheb_interval"][0]),
                        b=float(cfg["cheb_interval"][1]),
                    )
                )
        Z -= np.min(Z)
        Z *= water.HARTREE_TO_MEV

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        contour = ax.contourf(X, Y, Z, levels=24, cmap="viridis")
        ax.set_xlabel(f"AS variable {i + 1}")
        ax.set_ylabel(f"AS variable {j + 1}")
        ax.set_title(f"Water PES projection, n={n_points}, vars ({i + 1}, {j + 1})")
        fig.colorbar(contour, ax=ax, label="Relative energy (meV)")
        out_path = plot_dir / f"fig1_projection_n{n_points}_vars{i + 1}{j + 1}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def _run_table_ii(
    cfg: dict[str, Any],
    *,
    args_ns: Any,
    context_as,
    use_tqdm: bool,
) -> list[PaperThresholdRow]:
    rows: list[PaperThresholdRow] = []
    context_identity = prepare_planar_identity_context(args_ns)
    no_as_points = int(cfg["paper_table_ii_no_as_points"])
    for threshold in cfg["paper_table_ii_thresholds"]:
        t0 = time.time()
        max_rank_by_points: dict[int, int | None] = {}
        status = "ok"
        error: str | None = None
        try:
            for n_points in cfg["paper_table_ii_points"]:
                _log_line(
                    f"[table-ii thr={threshold:.0e} n={n_points}] START rotated-AS",
                    use_tqdm=use_tqdm,
                )
                res = run_paper_as_tt(
                    args_ns,
                    context=context_as,
                    n_points=int(n_points),
                    rank_cap=int(cfg["paper_table_ii_rank_cap"]),
                    tol=float(threshold),
                    n_iter_max=int(cfg["author_max_iter"]),
                    random_state=int(cfg["author_random_state"]),
                    max_unique_queries=None,
                    max_total_queries=None,
                    rms_probe_points=0,
                    rms_seed=int(cfg["author_rms_seed"]),
                )
                max_rank_by_points[int(n_points)] = None if res.max_rank is None else int(res.max_rank)
                _log_line(
                    f"[table-ii thr={threshold:.0e} n={n_points}] DONE max_rank={max_rank_by_points[int(n_points)]}",
                    use_tqdm=use_tqdm,
                )

            _log_line(
                f"[table-ii thr={threshold:.0e} no-as n={no_as_points}] START",
                use_tqdm=use_tqdm,
            )
            res_no_as = run_paper_as_tt(
                args_ns,
                context=context_identity,
                n_points=int(no_as_points),
                rank_cap=int(cfg["paper_table_ii_rank_cap"]),
                tol=float(threshold),
                n_iter_max=int(cfg["author_max_iter"]),
                random_state=int(cfg["author_random_state"]),
                max_unique_queries=None,
                max_total_queries=None,
                rms_probe_points=0,
                rms_seed=int(cfg["author_rms_seed"]),
            )
            value_ranks = [int(x) for x in res_no_as.info.get("value_tt_ranks", [])]
            no_as_rank = max(value_ranks) if value_ranks else None
            _log_line(
                f"[table-ii thr={threshold:.0e} no-as n={no_as_points}] DONE tensor_rank={no_as_rank}",
                use_tqdm=use_tqdm,
            )
        except Exception as exc:
            status = "fail"
            error = str(exc)
            no_as_rank = None
            _log_line(f"[table-ii thr={threshold:.0e}] FAIL error={exc}", use_tqdm=use_tqdm)
        rows.append(
            PaperThresholdRow(
                threshold=float(threshold),
                max_rank_by_points=dict(max_rank_by_points),
                no_as_value_tensor_rank=no_as_rank,
                elapsed_sec=float(time.time() - t0),
                status=status,
                error=error,
            )
        )
    return rows


def _run_or_load_author_query(
    cfg: dict[str, Any],
    *,
    context,
    n_points: int,
    budget_name: str,
    split: int,
    requested_unique_budget: int | None,
    use_tqdm: bool,
) -> tuple[np.ndarray, np.ndarray, PaperASTTResult, float, str, Path]:
    shape = tuple([int(n_points)] * int(context.as_dim))
    cache_path = _query_cache_path(
        cfg,
        context_cache_tag=str(context.as_cache_tag),
        n_points=int(n_points),
        budget_name=str(budget_name),
        split=int(split),
    )
    if bool(cfg.get("author_reuse_query_cache", True)):
        cached = _load_cached_query_run(cache_path)
        if cached is not None:
            observed_mask, approx_tensor_h, meta = cached
            if bool(cfg.get("verbose", False)):
                _log_line(f"[paper-as-tt] using cached query run -> {cache_path}", use_tqdm=use_tqdm)
            cached_result = PaperASTTResult(
                tensor=approx_tensor_h,
                coeff_tensor=None,
                coeff_tt_cores=None,
                tt_cores=None,
                queried_flat_indices=[int(x) for x in np.flatnonzero(observed_mask.reshape(-1))],
                queried_index_sequence=[],
                total_query_count=int(meta.get("total_query_count", int(np.sum(observed_mask)))),
                unique_query_count=int(meta.get("unique_query_count", int(np.sum(observed_mask)))),
                converged=bool(meta.get("converged", False)),
                truncated_by_budget=bool(meta.get("truncated_by_budget", False)),
                max_rank=None if meta.get("max_rank") is None else int(meta.get("max_rank")),
                rms_random=None if meta.get("rms_random") is None else float(meta.get("rms_random")),
                info=dict(meta.get("info", {})),
            )
            return observed_mask, approx_tensor_h, cached_result, float(meta.get("elapsed_sec", 0.0)), "cache", cache_path

    t0 = time.time()
    result = run_paper_as_tt(
        type("Args", (), cfg)(),
        context=context,
        n_points=int(n_points),
        rank_cap=int(cfg["author_rank_cap"]),
        tol=float(cfg["author_tol"]),
        n_iter_max=int(cfg["author_max_iter"]),
        random_state=int(cfg["author_random_state"]) + int(split),
        max_unique_queries=None if requested_unique_budget is None else int(requested_unique_budget),
        max_total_queries=None,
        rms_probe_points=int(cfg["author_rms_probe_points"]),
        rms_seed=int(cfg["author_rms_seed"]) + int(split),
    )
    elapsed_sec = float(time.time() - t0)
    observed_mask = np.zeros(int(np.prod(shape)), dtype=bool)
    if result.queried_flat_indices:
        observed_mask[np.asarray(result.queried_flat_indices, dtype=int)] = True
    observed_mask = observed_mask.reshape(shape)
    approx_tensor_h = np.asarray(result.tensor, dtype=float) if result.tensor is not None else np.zeros(shape, dtype=float)
    _save_query_run(
        cache_path,
        shape=shape,
        result=result,
        elapsed_sec=elapsed_sec,
        budget_name=str(budget_name),
        requested_unique_budget=requested_unique_budget,
        approx_tensor_hartree=approx_tensor_h,
    )
    if bool(cfg.get("verbose", False)):
        _log_line(f"[paper-as-tt] saved query cache -> {cache_path}", use_tqdm=use_tqdm)
    return observed_mask, approx_tensor_h, result, elapsed_sec, "generated", cache_path


def _markdown_report(
    author_rows: list[PaperAuthorRow],
    threshold_rows: list[PaperThresholdRow],
    fig1_paths: list[str],
    completion_rows: list[SharedCompletionRow],
    settings: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Baranov 2015 AS+TT Water Benchmark")
    lines.append("")
    lines.append("Protocols:")
    lines.append("- `paper_table_i`: paper-style AS+TT online approximation on the water PES.")
    lines.append("- `paper_table_ii`: TT-rank study across thresholds and Chebyshev points.")
    lines.append("- `paper_fig1`: 2D projections in active-subspace coordinates.")
    lines.append("- `shared_query_completion`: author-selected queried points are frozen and reused as the observed set for completion baselines and Cheb.")
    lines.append("")
    lines.append("## Settings")
    lines.append(f"- Geometry: `{settings['geometry_xyz']}`")
    lines.append("- Water setup: `O fixed at origin; both H atoms constrained to a plane; active-subspace rotation in 4D`")
    lines.append(f"- Coordinate unit: `{settings['water_coordinate_unit']}`")
    lines.append(f"- Active subspace dim: `{settings['water_as_dim']}`")
    lines.append(f"- Active subspace sigma^2: `{settings['water_as_sigma2']}`")
    lines.append(f"- Active subspace gradient samples: `{settings['water_as_samples']}`")
    lines.append(f"- Active subspace seed: `{settings['water_as_random_state']}`")
    if settings.get("active_subspace_cache_path"):
        lines.append(f"- Active subspace cache: `{settings['active_subspace_cache_path']}`")
    lines.append(f"- Basis / method: `{settings['qc_method']} / {settings['basis']}`")
    lines.append(f"- Chebyshev interval: `{settings['cheb_interval']}`")
    lines.append(f"- Chebyshev points: `{settings['cheb_points_list']}`")
    lines.append(f"- Author TT-cross threshold: `{settings['author_tol']}`")
    lines.append(f"- Author rank cap: `{settings['author_rank_cap']}`")
    lines.append(f"- Author RMS probe points: `{settings['author_rms_probe_points']}`")
    lines.append(f"- Paper budget mode: `{settings['paper_budget_mode']}`")
    lines.append(f"- Table I points: `{settings['paper_table_i_points']}`")
    lines.append(f"- Table II thresholds: `{settings['paper_table_ii_thresholds']}`")
    lines.append(f"- Table II points: `{settings['paper_table_ii_points']}`")
    lines.append(f"- Table II no-AS points assumption: `{settings['paper_table_ii_no_as_points']}`")
    lines.append(f"- Shared budget variants: `{settings['shared_budget_variants']}`")
    lines.append("")

    if author_rows:
        lines.append("## Table I")
        lines.append("")
        lines.append("| Cheb points | status | max rank | RMS random (meV) | unique evals | total queries | paper ref evals | truncated | time(s) | source |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(author_rows, key=lambda x: x.cheb_points):
            lines.append(
                f"| {row.cheb_points} | {row.status} | "
                f"{'nan' if row.max_rank is None else row.max_rank} | "
                f"{'nan' if row.rms_random_mev is None else f'{row.rms_random_mev:.6f}'} | "
                f"{row.unique_queries} | {row.total_queries} | "
                f"{'nan' if row.paper_reference_unique_queries is None else row.paper_reference_unique_queries} | "
                f"{str(row.truncated_by_budget).lower()} | {row.elapsed_sec:.2f} | {row.query_source} |"
            )
        lines.append("")
        lines.append("Article metric to compare against Table I: `RMS random (meV)` together with `unique evals`.")
        lines.append("")

    if threshold_rows:
        lines.append("## Table II")
        lines.append("")
        header_cols = ["threshold"] + [str(p) for p in settings["paper_table_ii_points"]] + ["Tensor rank"]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["---:"] * len(header_cols)) + "|")
        for row in threshold_rows:
            values = [f"{row.threshold:.0e}"]
            for p in settings["paper_table_ii_points"]:
                v = row.max_rank_by_points.get(int(p))
                values.append("nan" if v is None else str(v))
            values.append("nan" if row.no_as_value_tensor_rank is None else str(row.no_as_value_tensor_rank))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        lines.append(
            "`Tensor rank` is computed for the exact PES value tensor without AS transformation "
            f"using the same TT-cross threshold on a `{settings['paper_table_ii_no_as_points']}`-point Chebyshev mesh."
        )
        lines.append("")

    if fig1_paths:
        lines.append("## Figure 1")
        lines.append("")
        lines.append("Paper-style 2D projections of the PES in active-subspace coordinates:")
        for path in fig1_paths:
            lines.append(f"- `{path}`")
        lines.append("")

    if completion_rows:
        lines.append("## Shared-Query Completion")
        for n in sorted({r.cheb_points for r in completion_rows}):
            lines.append("")
            lines.append(f"### Chebyshev points = {n}")
            lines.append("")
            lines.append("| budget | split | baseline | status | req obs | actual obs | eval | scope | RMSE (meV) | MAE (meV) | maxAE (meV) | rRMSE | MAPE | author RMS random (meV) | time(s) | source |")
            lines.append("|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
            bucket = [r for r in completion_rows if r.cheb_points == n]
            bucket.sort(
                key=lambda x: (
                    x.budget_name,
                    x.split,
                    x.status != "ok",
                    x.rmse_mev if x.rmse_mev is not None else 1e18,
                    x.baseline,
                )
            )
            for row in bucket:
                lines.append(
                    f"| {row.budget_name} | {row.split} | {row.baseline} | {row.status} | "
                    f"{row.requested_unique_budget} | {row.observed_points} | {row.eval_points} | {row.metric_scope} | "
                    f"{'nan' if row.rmse_mev is None else f'{row.rmse_mev:.6f}'} | "
                    f"{'nan' if row.mae_mev is None else f'{row.mae_mev:.6f}'} | "
                    f"{'nan' if row.max_abs_error_mev is None else f'{row.max_abs_error_mev:.6f}'} | "
                    f"{'nan' if row.relative_rmse is None else f'{row.relative_rmse:.6f}'} | "
                    f"{'nan' if row.mape is None else f'{row.mape:.6f}'} | "
                    f"{'nan' if row.author_rms_random_mev is None else f'{row.author_rms_random_mev:.6f}'} | "
                    f"{row.elapsed_sec:.2f} | {row.query_source} |"
                )
        lines.append("")
        lines.append("Completion metrics are computed on `hidden_only` whenever the observed mask is non-empty.")
        lines.append("`author RMS random (meV)` is the paper-style RMS of the online AS+TT approximation for that same query run.")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- The paper method is not ordinary completion: it is online adaptive interpolation via AS + TT-cross.")
    lines.append("- The shared-query section turns the author-selected queried points into a completion mask, so other baselines and Cheb can be compared on exactly the same observed set.")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(cfg: dict[str, Any]) -> None:
    if bool(cfg["disable_joblib_multiprocessing"]):
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"
    else:
        os.environ.pop("JOBLIB_MULTIPROCESSING", None)

    args_ns = type("Args", (), cfg)()
    use_outer_tqdm = bool(cfg["show_tqdm"])
    _log_line(
        f"[baranov2015] START run_mode={cfg['run_mode']} "
        f"paper={_paper_phase_enabled(cfg)} shared={_shared_phase_enabled(cfg)} "
        f"cheb_points={cfg['cheb_points_list']}",
        use_tqdm=use_outer_tqdm,
    )
    context = prepare_planar_water_context(args_ns)

    author_rows: list[PaperAuthorRow] = []
    threshold_rows: list[PaperThresholdRow] = []
    fig1_paths: list[str] = []
    completion_rows: list[SharedCompletionRow] = []

    paper_points = sorted(set(int(x) for x in cfg["paper_table_i_points"]))
    shared_points = sorted(set(int(x) for x in cfg["cheb_points_list"]))

    completion_names = dual._expand_cheb_tokens(
        dual._resolve_completion_names(cfg["shared_query"]["baselines"]),
        cfg,
        offline=False,
    )
    completion_names = [name for name in completion_names if name != "as_tt_online"]
    if bool(cfg.get("verbose", False)):
        _log_line(
            f"[baranov2015] completion baselines={completion_names}",
            use_tqdm=use_outer_tqdm,
        )

    if _paper_phase_enabled(cfg) and bool(cfg.get("paper_run_table_i", True)):
        iter_table_i = paper_points
        if use_outer_tqdm:
            iter_table_i = tqdm(iter_table_i, desc="Table I / Chebyshev points", unit="cfg")
        for n_points in iter_table_i:
            paper_reference = _reference_budget(cfg, int(n_points))
            budget_cap, budget_name = _resolve_author_budget(cfg, int(n_points))
            _log_line(
                f"[paper-table-i n={n_points}] START budget={budget_name}",
                use_tqdm=use_outer_tqdm,
            )
            cache_path = _query_cache_path(
                cfg,
                context_cache_tag=str(context.as_cache_tag),
                n_points=int(n_points),
                budget_name=str(budget_name),
                split=0,
            )
            try:
                _mask, _pred_h, result, elapsed_sec, query_source, cache_path = _run_or_load_author_query(
                    cfg,
                    context=context,
                    n_points=int(n_points),
                    budget_name=str(budget_name),
                    split=0,
                    requested_unique_budget=budget_cap,
                    use_tqdm=use_outer_tqdm,
                )
                row = _author_row_from_result(
                    n_points=int(n_points),
                    result=result,
                    elapsed_sec=float(elapsed_sec),
                    query_source=query_source,
                    query_cache_path=cache_path,
                    paper_reference_unique_queries=paper_reference,
                )
                author_rows.append(row)
                _log_line(
                    f"[paper-table-i n={n_points}] DONE rms={row.rms_random_mev:.4f} meV "
                    f"unique={row.unique_queries} total={row.total_queries}",
                    use_tqdm=use_outer_tqdm,
                )
            except Exception as exc:
                row = PaperAuthorRow(
                    cheb_points=int(n_points),
                    status="fail",
                    max_rank=None,
                    rms_random_mev=None,
                    unique_queries=0,
                    total_queries=0,
                    paper_reference_unique_queries=paper_reference,
                    truncated_by_budget=False,
                    elapsed_sec=0.0,
                    query_source="failed",
                    query_cache_path=str(cache_path),
                    error=str(exc),
                )
                author_rows.append(row)
                _log_line(f"[paper-table-i n={n_points}] FAIL error={exc}", use_tqdm=use_outer_tqdm)

    if _paper_phase_enabled(cfg) and bool(cfg.get("paper_run_table_ii", True)):
        _log_line("[paper-table-ii] START", use_tqdm=use_outer_tqdm)
        threshold_rows = _run_table_ii(
            cfg,
            args_ns=args_ns,
            context_as=context,
            use_tqdm=use_outer_tqdm,
        )
        _log_line("[paper-table-ii] DONE", use_tqdm=use_outer_tqdm)

    if _paper_phase_enabled(cfg) and bool(cfg.get("paper_run_fig1", True)):
        fig_n = max(paper_points) if paper_points else max(shared_points)
        _log_line(f"[paper-fig1 n={fig_n}] START", use_tqdm=use_outer_tqdm)
        try:
            _mask, _pred_h, fig_result, _elapsed_sec, _query_source, _cache_path = _run_or_load_author_query(
                cfg,
                context=context,
                n_points=int(fig_n),
                budget_name="paper_unbounded",
                split=0,
                requested_unique_budget=None,
                use_tqdm=use_outer_tqdm,
            )
            if not fig_result.coeff_tt_cores:
                _log_line(
                    f"[paper-fig1 n={fig_n}] cache has no TT coefficients, regenerating author approximation",
                    use_tqdm=use_outer_tqdm,
                )
                fig_result = run_paper_as_tt(
                    args_ns,
                    context=context,
                    n_points=int(fig_n),
                    rank_cap=int(cfg["author_rank_cap"]),
                    tol=float(cfg["author_tol"]),
                    n_iter_max=int(cfg["author_max_iter"]),
                    random_state=int(cfg["author_random_state"]),
                    max_unique_queries=None,
                    max_total_queries=None,
                    rms_probe_points=0,
                    rms_seed=int(cfg["author_rms_seed"]),
                )
            fig1_paths = _plot_fig1(cfg, context=context, result=fig_result, n_points=int(fig_n))
            _log_line(f"[paper-fig1 n={fig_n}] DONE plots={len(fig1_paths)}", use_tqdm=use_outer_tqdm)
        except Exception as exc:
            _log_line(f"[paper-fig1 n={fig_n}] FAIL error={exc}", use_tqdm=use_outer_tqdm)

    points_iter = shared_points
    if use_outer_tqdm:
        points_iter = tqdm(points_iter, desc="Shared completion / Chebyshev points", unit="cfg")

    for n_points in points_iter:
        paper_reference = _reference_budget(cfg, int(n_points))
        if _shared_phase_enabled(cfg):
            _log_line(
                f"[shared n={n_points}] LOAD full paper tensor",
                use_tqdm=use_outer_tqdm,
            )
            tensor_h, _nodes = load_or_generate_full_value_tensor(args_ns, context=context, n_points=int(n_points))
            reference_min_h = float(np.min(tensor_h))
            full_tensor_mev = _to_relative_mev_with_reference(tensor_h, reference_min_h)
            shape = full_tensor_mev.shape
            n_total = int(np.prod(shape))
            baselines = dual._build_completion_baselines(cfg, shape)
            budget_variants = _resolve_shared_budget_variants(cfg)

            for variant in budget_variants:
                requested_unique_budget = _shared_requested_unique_budget(
                    cfg,
                    n_total=n_total,
                    n_points=int(n_points),
                    variant=variant,
                )
                for split in range(int(cfg["shared_query"]["splits"])):
                    _log_line(
                        f"[shared n={n_points} budget={variant['name']} split={split}] "
                        f"START author-query requested_unique={requested_unique_budget}/{n_total}",
                        use_tqdm=use_outer_tqdm,
                    )
                    try:
                        observed_mask, author_pred_h, author_result, author_elapsed, query_source, query_cache_path = _run_or_load_author_query(
                            cfg,
                            context=context,
                            n_points=int(n_points),
                            budget_name=str(variant["name"]),
                            split=int(split),
                            requested_unique_budget=int(requested_unique_budget),
                            use_tqdm=use_outer_tqdm,
                        )
                        author_pred_mev = _to_relative_mev_with_reference(np.asarray(author_pred_h, dtype=float), reference_min_h)
                        author_row = _completion_row_from_prediction(
                            n_points=int(n_points),
                            budget_name=str(variant["name"]),
                            budget_mode=str(variant["mode"]),
                            budget_fraction=variant["fraction"],
                            split=int(split),
                            baseline="paper_as_tt",
                            pred_mev=author_pred_mev,
                            full_tensor_mev=full_tensor_mev,
                            observed_mask=observed_mask,
                            requested_unique_budget=int(requested_unique_budget),
                            author_rms_random_mev=author_result.rms_random,
                            author_unique_queries=int(author_result.unique_query_count),
                            author_total_queries=int(author_result.total_query_count),
                            elapsed_sec=float(author_elapsed),
                            query_source=query_source,
                            query_cache_path=query_cache_path,
                        )
                        completion_rows.append(author_row)
                        _log_line(
                            f"[shared n={n_points} budget={variant['name']} split={split}] "
                            f"DONE author-query rmse={author_row.rmse_mev:.4f} meV obs={author_row.observed_points}",
                            use_tqdm=use_outer_tqdm,
                        )
                    except Exception as exc:
                        completion_rows.append(
                            SharedCompletionRow(
                                cheb_points=int(n_points),
                                budget_name=str(variant["name"]),
                                budget_mode=str(variant["mode"]),
                                budget_fraction=variant["fraction"],
                                split=int(split),
                                baseline="paper_as_tt",
                                status="fail",
                                requested_unique_budget=int(requested_unique_budget),
                                observed_points=0,
                                eval_points=int(n_total),
                                metric_scope="hidden_only",
                                rmse_mev=None,
                                mae_mev=None,
                                max_abs_error_mev=None,
                                relative_rmse=None,
                                mape=None,
                                author_rms_random_mev=None,
                                author_unique_queries=0,
                                author_total_queries=0,
                                elapsed_sec=0.0,
                                query_source="failed",
                                query_cache_path=str(
                                    _query_cache_path(
                                        cfg,
                                        context_cache_tag=str(context.as_cache_tag),
                                        n_points=int(n_points),
                                        budget_name=str(variant["name"]),
                                        split=int(split),
                                    )
                                ),
                                error=str(exc),
                            )
                        )
                        _log_line(
                            f"[shared n={n_points} budget={variant['name']} split={split}] "
                            f"FAIL author-query error={exc}",
                            use_tqdm=use_outer_tqdm,
                        )
                        continue

                    for baseline_name in completion_names:
                        _log_line(
                            f"[shared n={n_points} budget={variant['name']} split={split}] "
                            f"START baseline={baseline_name}",
                            use_tqdm=use_outer_tqdm,
                        )
                        t0 = time.time()
                        try:
                            model = baselines[baseline_name]
                            dual_row, _info = dual._run_completion_baseline(
                                model=model,
                                full_tensor=full_tensor_mev,
                                mask=observed_mask,
                            )
                            completion_rows.append(
                                SharedCompletionRow(
                                    cheb_points=int(n_points),
                                    budget_name=str(variant["name"]),
                                    budget_mode=str(variant["mode"]),
                                    budget_fraction=variant["fraction"],
                                    split=int(split),
                                    baseline=baseline_name,
                                    status="ok",
                                    requested_unique_budget=int(requested_unique_budget),
                                    observed_points=int(np.sum(observed_mask)),
                                    eval_points=int(dual_row.eval_points),
                                    metric_scope=str(dual_row.metric_scope),
                                    rmse_mev=float(dual_row.rmse_mev) if dual_row.rmse_mev is not None else None,
                                    mae_mev=float(dual_row.mae_mev) if dual_row.mae_mev is not None else None,
                                    max_abs_error_mev=float(dual_row.max_abs_error_mev) if dual_row.max_abs_error_mev is not None else None,
                                    relative_rmse=float(dual_row.relative_rmse) if dual_row.relative_rmse is not None else None,
                                    mape=float(dual_row.mape) if dual_row.mape is not None else None,
                                    author_rms_random_mev=None if author_result.rms_random is None else float(author_result.rms_random),
                                    author_unique_queries=int(author_result.unique_query_count),
                                    author_total_queries=int(author_result.total_query_count),
                                    elapsed_sec=float(time.time() - t0),
                                    query_source=str(query_source),
                                    query_cache_path=str(query_cache_path),
                                    error=None,
                                )
                            )
                            _log_line(
                                f"[shared n={n_points} budget={variant['name']} split={split}] "
                                f"DONE baseline={baseline_name} rmse={completion_rows[-1].rmse_mev:.4f} meV "
                                f"time={completion_rows[-1].elapsed_sec:.2f}s",
                                use_tqdm=use_outer_tqdm,
                            )
                        except Exception as exc:
                            completion_rows.append(
                                SharedCompletionRow(
                                    cheb_points=int(n_points),
                                    budget_name=str(variant["name"]),
                                    budget_mode=str(variant["mode"]),
                                    budget_fraction=variant["fraction"],
                                    split=int(split),
                                    baseline=baseline_name,
                                    status="fail",
                                    requested_unique_budget=int(requested_unique_budget),
                                    observed_points=int(np.sum(observed_mask)),
                                    eval_points=max(0, n_total - int(np.sum(observed_mask))),
                                    metric_scope="hidden_only",
                                    rmse_mev=None,
                                    mae_mev=None,
                                    max_abs_error_mev=None,
                                    relative_rmse=None,
                                    mape=None,
                                    author_rms_random_mev=None if author_result.rms_random is None else float(author_result.rms_random),
                                    author_unique_queries=int(author_result.unique_query_count),
                                    author_total_queries=int(author_result.total_query_count),
                                    elapsed_sec=float(time.time() - t0),
                                    query_source=str(query_source),
                                    query_cache_path=str(query_cache_path),
                                    error=str(exc),
                                )
                            )
                            _log_line(
                                f"[shared n={n_points} budget={variant['name']} split={split}] "
                                f"FAIL baseline={baseline_name} error={exc}",
                                use_tqdm=use_outer_tqdm,
                            )

    settings = {
        "geometry_xyz": str(cfg["geometry_xyz"]),
        "water_coordinate_unit": str(cfg["water_coordinate_unit"]),
        "water_as_dim": int(cfg["water_as_dim"]),
        "water_as_sigma2": float(cfg["water_as_sigma2"]),
        "water_as_samples": int(cfg["water_as_samples"]),
        "water_as_random_state": int(cfg["water_as_random_state"]),
        "active_subspace_cache_path": context.as_cache_path,
        "qc_method": str(cfg["qc_method"]),
        "basis": str(cfg["basis"]),
        "cheb_interval": list(cfg["cheb_interval"]),
        "cheb_points_list": list(cfg["cheb_points_list"]),
        "paper_table_i_points": list(cfg["paper_table_i_points"]),
        "paper_table_ii_thresholds": list(cfg["paper_table_ii_thresholds"]),
        "paper_table_ii_points": list(cfg["paper_table_ii_points"]),
        "paper_table_ii_no_as_points": int(cfg["paper_table_ii_no_as_points"]),
        "author_rank_cap": int(cfg["author_rank_cap"]),
        "author_tol": float(cfg["author_tol"]),
        "author_max_iter": int(cfg["author_max_iter"]),
        "author_rms_probe_points": int(cfg["author_rms_probe_points"]),
        "paper_budget_mode": str(cfg["paper_budget_mode"]),
        "paper_plots_dir": str(cfg["paper_plots_dir"]),
        "shared_budget_variants": _resolve_shared_budget_variants(cfg),
    }
    payload = {
        "settings": settings,
        "paper_author_rows": [asdict(r) for r in author_rows],
        "paper_threshold_rows": [asdict(r) for r in threshold_rows],
        "paper_fig1_paths": list(fig1_paths),
        "shared_completion_rows": [asdict(r) for r in completion_rows],
    }
    Path(cfg["output_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(cfg["output_md"]).write_text(
        _markdown_report(author_rows, threshold_rows, fig1_paths, completion_rows, settings),
        encoding="utf-8",
    )
    _log_line(f"[baranov2015] DONE json={cfg['output_json']} md={cfg['output_md']}", use_tqdm=use_outer_tqdm)


if __name__ == "__main__":
    run(_resolve_cfg())
