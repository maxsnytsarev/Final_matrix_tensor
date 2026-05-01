from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from as_tt_water_experiments import run_as_tt_water_experiments as water
from baranov2015_as_tt_experiments import baranov2015_faithful as faithful


# -----------------------------------------------------------------------------
# Fair comparison protocol
# -----------------------------------------------------------------------------
# 1) Collect ONE shared adaptive trace from the authors' online sampler.
# 2) Evaluate all completion variants on PREFIXES of the SAME trace.
# 3) Work directly in HARTREE on the value tensor V (no observed-min re-referencing).
# 4) Report two families of metrics:
#       (A) value-tensor reconstruction metrics on the Chebyshev grid
#       (B) surrogate metrics on a COMMON random test set in AS-space
# This matches the paper's end goal (PES surrogate quality) while preserving a
# direct tensor-completion diagnostic.
# -----------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
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
        "author_rms_probe_points": 1000,
        "author_rms_seed": 2025,
        "show_tqdm": True,
        "verbose": True,
    },
    "authors_sampler": {
        "n_points": 6,
        "tol": 1e-5,
        "tt_backend": "ttml",
        "ttml_env_name": "matrix_approximation_final_3_11",
        "ttml_method": "dmrg",
        "random_state": 0,
        "max_budget": 512,
    },
    "evaluation": {
        "random_test_points": 1000,
        "test_seed": 0,
        "compute_exact_grid": True,
    },
    "completion_backend": {
        # Preferred backend for fair comparison because it supports explicit masks.
        "module_candidates": [
            "completion_linalg_tensor_masked",
            "completion_linalg_tensor_all",
        ],
    },
    "completion_variants": [
        {
            "name": "approximateLOO_rank20",
            "enabled": True,
            "kwargs": {
                "rank": 20,
                "number_of_steps": 5,
                "tol_for_step": 1e-4,
                "begin": "oversample",
                "lambda_all": 0.0,
                "rank_eval": False,
                "rank_nest": False,
                "nest_iters": 5,
                "tol": 1e-5,
                "max_rank": 20,
                "using_qr": False,
                "eval": True,
                "seed": 179,
                "return_compl": True,
                "return_history": True,
                "ret_best": True,
                "TQDM": True,
                "eval_fall": True,
                "validation_size": 0.1,
                "tolerance": 500,
                "n_workers": 1,
            },
        },
    ],
    "budgets": [64, 128, 256, 512],
    "io": {
        "output_json": "baranov2015_as_tt_experiments/results/fair_paper_comparison_water.json",
    },
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_args_ns(cfg: dict[str, Any]) -> SimpleNamespace:
    args = dict(cfg["paper_args"])
    args["tt_backend"] = str(cfg["authors_sampler"]["tt_backend"])
    args["ttml_env_name"] = str(cfg["authors_sampler"]["ttml_env_name"])
    args["ttml_method"] = str(cfg["authors_sampler"]["ttml_method"])
    return SimpleNamespace(**args)


def _prefix_trace(trace: faithful.SampleTrace, budget: int) -> faithful.SampleTrace:
    budget = int(min(int(budget), int(trace.unique_queries)))
    return faithful.SampleTrace(
        unique_indices=np.asarray(trace.unique_indices[:budget], dtype=int).copy(),
        unique_values=np.asarray(trace.unique_values[:budget], dtype=float).copy(),
        query_sequence=list(trace.query_sequence),
        total_queries=int(trace.total_queries),
        unique_queries=int(budget),
        status=str(trace.status),
        shape=tuple(int(x) for x in trace.shape),
        info=dict(trace.info),
    )


def _import_completion_backend() -> Any:
    errors: list[str] = []
    for name in CONFIG["completion_backend"]["module_candidates"]:
        try:
            mod = importlib.import_module(name)
            return mod
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import any completion backend: " + " | ".join(errors))


def _build_exact_entry_oracle(context: Any, nodes: np.ndarray, energy_fn: Any):
    if hasattr(faithful, "make_exact_pes_entry_oracle"):
        return faithful.make_exact_pes_entry_oracle(context, nodes, energy_fn)

    def _entry(idx: tuple[int, ...]) -> float:
        y = np.asarray([nodes[int(i)] for i in idx], dtype=float)
        if hasattr(faithful, "reduced_to_cartesian"):
            coords = faithful.reduced_to_cartesian(context, y)
        else:
            raise RuntimeError("Cannot map reduced coordinates to Cartesian coordinates")
        return float(energy_fn(list(context.symbols), coords))

    return _entry


def _build_exact_grid_tensor(args_ns: Any, trace: faithful.SampleTrace, energy_fn: Any) -> np.ndarray:
    n_points = int(trace.shape[0])
    context = trace.context if hasattr(trace, "context") else None

    # Preferred path: reuse cached exact tensor if the faithful module exposes it.
    if context is not None and hasattr(faithful, "load_or_generate_full_value_tensor"):
        try:
            tensor_h, _nodes = faithful.load_or_generate_full_value_tensor(
                args_ns,
                context=context,
                n_points=n_points,
            )
            return np.asarray(tensor_h, dtype=float)
        except Exception as exc:
            _log(f"[warn] load_or_generate_full_value_tensor failed, falling back to brute force: {exc}")

    if context is None:
        raise RuntimeError("Sample trace does not carry context; cannot build the exact grid tensor")

    nodes = np.asarray(trace.nodes, dtype=float) if hasattr(trace, "nodes") else faithful.chebyshev_nodes(
        n_points,
        a=float(args_ns.cheb_interval[0]),
        b=float(args_ns.cheb_interval[1]),
    )
    entry_oracle = _build_exact_entry_oracle(context, nodes, energy_fn)
    tensor_h = np.zeros(tuple(int(x) for x in trace.shape), dtype=float)
    for idx in np.ndindex(tensor_h.shape):
        tensor_h[idx] = float(entry_oracle(tuple(int(i) for i in idx)))
    return tensor_h


def _sample_test_points(trace: faithful.SampleTrace, seed: int, n_test: int) -> np.ndarray:
    if not hasattr(trace, "context"):
        raise RuntimeError("Sample trace does not carry context; cannot build the random test set")
    a, b = tuple(float(v) for v in trace.context.cheb_interval)
    if hasattr(faithful, "sample_random_test_points"):
        return np.asarray(
            faithful.sample_random_test_points(trace.context, n_test=n_test, a=a, b=b, seed=seed),
            dtype=float,
        )
    rng = np.random.default_rng(seed)
    return rng.uniform(low=a, high=b, size=(int(n_test), int(trace.shape.__len__())))



def _evaluate_point_metrics(true_h: np.ndarray, pred_h: np.ndarray) -> dict[str, float]:
    err_h = np.asarray(pred_h, dtype=float) - np.asarray(true_h, dtype=float)
    return {
        "rmse_mev": float(np.sqrt(np.mean(err_h ** 2)) * water.HARTREE_TO_MEV) if err_h.size else float("nan"),
        "mae_mev": float(np.mean(np.abs(err_h)) * water.HARTREE_TO_MEV) if err_h.size else float("nan"),
        "max_abs_mev": float(np.max(np.abs(err_h)) * water.HARTREE_TO_MEV) if err_h.size else float("nan"),
        "n_points": float(err_h.size),
    }


def _evaluate_grid_metrics(exact_h: np.ndarray, pred_h: np.ndarray, observed_indices: np.ndarray) -> dict[str, float]:
    exact_h = np.asarray(exact_h, dtype=float)
    pred_h = np.asarray(pred_h, dtype=float)
    diff_mev = (pred_h - exact_h) * water.HARTREE_TO_MEV
    out = {
        "grid_full_rmse_mev": float(np.sqrt(np.mean(diff_mev ** 2))),
        "grid_full_max_abs_mev": float(np.max(np.abs(diff_mev))),
        "grid_total_points": int(diff_mev.size),
    }
    obs_mask = np.zeros(exact_h.shape, dtype=bool)
    if observed_indices.size:
        obs_mask[tuple(np.asarray(observed_indices, dtype=int).T)] = True
    hidden_mask = ~obs_mask
    if np.any(obs_mask):
        obs_err = diff_mev[obs_mask]
        out.update(
            {
                "grid_observed_rmse_mev": float(np.sqrt(np.mean(obs_err ** 2))),
                "grid_observed_max_abs_mev": float(np.max(np.abs(obs_err))),
                "grid_observed_points": int(obs_err.size),
            }
        )
    if np.any(hidden_mask):
        hid_err = diff_mev[hidden_mask]
        out.update(
            {
                "grid_hidden_rmse_mev": float(np.sqrt(np.mean(hid_err ** 2))),
                "grid_hidden_max_abs_mev": float(np.max(np.abs(hid_err))),
                "grid_hidden_points": int(hid_err.size),
            }
        )
    return out


def _evaluate_surrogate_from_completed_tensor(
    *,
    completed_h: np.ndarray,
    trace: faithful.SampleTrace,
    test_points: np.ndarray,
    energy_fn: Any,
) -> dict[str, float]:
    a, b = tuple(float(v) for v in trace.context.cheb_interval)
    coeff = faithful.values_to_chebyshev_coefficients(completed_h, np.asarray(trace.nodes, dtype=float), a=a, b=b)
    pred = np.array(
        [faithful.evaluate_chebyshev_tensor(coeff, y, a=a, b=b) for y in np.asarray(test_points, dtype=float)],
        dtype=float,
    )
    true = np.array(
        [
            float(energy_fn(list(trace.context.symbols), faithful.reduced_to_cartesian(trace.context, y)))
            for y in np.asarray(test_points, dtype=float)
        ],
        dtype=float,
    )
    out = _evaluate_point_metrics(true, pred)
    out["surrogate_points"] = int(len(test_points))
    return out


def _evaluate_author_baseline_on_common_test(
    baseline: Any,
    *,
    trace: faithful.SampleTrace,
    test_points: np.ndarray,
    energy_fn: Any,
) -> dict[str, float]:
    coeff = getattr(baseline, "coeff_tensor", None)
    if coeff is None:
        return {"rmse_mev": float("nan"), "mae_mev": float("nan"), "max_abs_mev": float("nan"), "n_points": 0.0}
    a, b = tuple(float(v) for v in trace.context.cheb_interval)
    pred = np.array(
        [faithful.evaluate_chebyshev_tensor(coeff, y, a=a, b=b) for y in np.asarray(test_points, dtype=float)],
        dtype=float,
    )
    true = np.array(
        [
            float(energy_fn(list(trace.context.symbols), faithful.reduced_point_to_cartesian(trace.context, y)))
            for y in np.asarray(test_points, dtype=float)
        ],
        dtype=float,
    )
    return _evaluate_point_metrics(true, pred)


def _run_completion_variant(
    *,
    completion_backend: Any,
    variant_name: str,
    variant_kwargs: dict[str, Any],
    prefix: faithful.SampleTrace,
    exact_grid_tensor_h: np.ndarray,
    test_points: np.ndarray,
    energy_fn: Any,
) -> dict[str, Any]:
    kwargs = dict(variant_kwargs)
    if "n_workers" in kwargs and hasattr(completion_backend, "n_workers"):
        completion_backend.n_workers = int(kwargs["n_workers"])

    t0 = time.time()
    factors, history, completed = completion_backend.approximateLOO(
        indices=np.asarray(prefix.unique_indices, dtype=int),
        values=np.asarray(prefix.unique_values, dtype=float),  # HARTREE, not re-referenced.
        shape=tuple(int(x) for x in prefix.shape),
        **kwargs,
    )
    elapsed = float(time.time() - t0)
    if completed is None:
        raise RuntimeError("Completion backend returned completed=None; set return_compl=True")
    completed_h = np.asarray(completed, dtype=float)

    train_true = np.asarray(prefix.unique_values, dtype=float)
    train_pred = completed_h[tuple(np.asarray(prefix.unique_indices, dtype=int).T)]

    row = {
        "variant": variant_name,
        "budget": int(prefix.unique_queries),
        "elapsed_sec": elapsed,
        "history_len": 0 if history is None else int(len(history)),
        "train": _evaluate_point_metrics(train_true, train_pred),
        "grid": _evaluate_grid_metrics(exact_grid_tensor_h, completed_h, np.asarray(prefix.unique_indices, dtype=int)),
        "surrogate_test": _evaluate_surrogate_from_completed_tensor(
            completed_h=completed_h,
            trace=prefix,
            test_points=test_points,
            energy_fn=energy_fn,
        ),
        "completed_min_hartree": float(np.min(completed_h)),
        "completed_max_hartree": float(np.max(completed_h)),
        "n_factors": None if factors is None else int(len(factors)),
    }
    return row


def main() -> None:
    cfg = CONFIG
    args_ns = _make_args_ns(cfg)
    n_points = int(cfg["authors_sampler"]["n_points"])
    budgets = sorted(set(int(b) for b in cfg["budgets"]))
    max_budget = int(min(max(budgets), n_points ** int(cfg["paper_args"]["water_as_dim"])))
    tol = float(cfg["authors_sampler"]["tol"])

    completion_backend = _import_completion_backend()
    energy_fn = water._build_energy_fn(args_ns)

    _log(f"[fair] collecting shared author trace up to unique budget {max_budget}")
    trace = faithful.collect_author_samples(
        args_ns,
        n_points=n_points,
        tol=tol,
        unique_budget=max_budget,
        tt_backend=str(cfg["authors_sampler"]["tt_backend"]),
        random_state=int(cfg["authors_sampler"]["random_state"]),
    )

    _log(
        f"[fair] shared trace status={trace.status} unique={trace.unique_queries} total={trace.total_queries}"
    )

    test_points = _sample_test_points(
        trace,
        seed=int(cfg["evaluation"]["test_seed"]),
        n_test=int(cfg["evaluation"]["random_test_points"]),
    )

    exact_grid_tensor_h = None
    if bool(cfg["evaluation"]["compute_exact_grid"]):
        _log("[fair] building exact grid tensor")
        exact_grid_tensor_h = _build_exact_grid_tensor(args_ns, trace, energy_fn)

    _log("[fair] running full authors baseline (for paper-style reference)")
    baseline_kwargs = dict(
        args=args_ns,
        n_points=n_points,
        tol=tol,
        tt_backend=str(cfg["authors_sampler"]["tt_backend"]),
        random_state=int(cfg["authors_sampler"]["random_state"]),
    )
    baseline_sig = inspect.signature(faithful.run_baranov2015_water_baseline)
    if "materialize_dense_for_postprocessing" in baseline_sig.parameters:
        baseline_kwargs["materialize_dense_for_postprocessing"] = True
    baseline = faithful.run_baranov2015_water_baseline(**baseline_kwargs)

    baseline_common_test = _evaluate_author_baseline_on_common_test(
        baseline,
        trace=trace,
        test_points=test_points,
        energy_fn=energy_fn,
    )

    summary: dict[str, Any] = {
        "protocol": {
            "shared_trace": "one author-driven adaptive trace, reused for all budgets",
            "working_values": "hartree",
            "primary_metric": "surrogate_test.rmse_mev on a common random AS-space test set",
            "secondary_metric": "grid.grid_hidden_rmse_mev on exact value tensor",
        },
        "trace": {
            "status": str(trace.status),
            "unique_queries": int(trace.unique_queries),
            "total_queries": int(trace.total_queries),
            "shape": [int(x) for x in trace.shape],
            "info": dict(trace.info),
        },
        "authors_baseline": {
            "unique_queries": int(getattr(baseline, "unique_queries", -1)),
            "total_queries": int(getattr(baseline, "total_queries", -1)),
            "rms_random_mev_builtin": None if getattr(baseline, "rms_random", None) is None else float(baseline.rms_random),
            "common_test": baseline_common_test,
            "info": dict(getattr(baseline, "info", {})),
        },
        "completion_results": [],
    }

    enabled_variants = [v for v in cfg["completion_variants"] if bool(v.get("enabled", True))]
    for budget in budgets:
        if budget > int(trace.unique_queries):
            _log(f"[fair] skip budget={budget}: only {trace.unique_queries} unique author samples collected")
            continue
        prefix = _prefix_trace(trace, budget)
        for variant in enabled_variants:
            name = str(variant["name"])
            _log(f"[fair] START variant={name} budget={budget}")
            row = _run_completion_variant(
                completion_backend=completion_backend,
                variant_name=name,
                variant_kwargs=dict(variant.get("kwargs", {})),
                prefix=prefix,
                exact_grid_tensor_h=np.asarray(exact_grid_tensor_h, dtype=float),
                test_points=test_points,
                energy_fn=energy_fn,
            )
            summary["completion_results"].append(row)
            _log(
                f"[fair] DONE variant={name} budget={budget} "
                f"test_rmse={row['surrogate_test']['rmse_mev']:.6f} meV "
                f"hidden_grid_rmse={row['grid'].get('grid_hidden_rmse_mev', float('nan')):.6f} meV"
            )

    out_path = Path(cfg["io"]["output_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _log(f"[fair] saved summary -> {out_path}")


if __name__ == "__main__":
    main()
