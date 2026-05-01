from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable

from all_baselines.baseline_cp_wopt import CPWOPT
from all_baselines.baseline_halrtc import HaLRTC
from all_baselines.baseline_tt_als import TTALS
from all_baselines.baseline_tucker import TuckerCompletion
from as_tt_water_experiments import run_as_tt_water_experiments as water
from baranov2015_as_tt_experiments import baranov2015_faithful as faithful


CANONICAL_FIXED_MASK_BASELINES = {
    "ttemps_rttc",
    "tensorly_tucker",
    "tensor_toolbox_cp_wopt",
    "xinychen_halrtc",
}

BASELINE_ALIASES = {
    "my_cheb": "my_cheb",
    "my_cheb_auto": "my_cheb_auto",
    "authors_budgeted": "authors_budgeted",
    "tt_als": "ttemps_rttc",
    "tt-als": "ttemps_rttc",
    "rttc": "ttemps_rttc",
    "ttemps": "ttemps_rttc",
    "ttemps_rttc": "ttemps_rttc",
    "tucker": "tensorly_tucker",
    "tensorly_tucker": "tensorly_tucker",
    "cp_wopt": "tensor_toolbox_cp_wopt",
    "cp-wopt": "tensor_toolbox_cp_wopt",
    "tensor_toolbox_cp_wopt": "tensor_toolbox_cp_wopt",
    "halrtc": "xinychen_halrtc",
    "xinychen_halrtc": "xinychen_halrtc",
}

DEFAULT_BASELINES = [
    "my_cheb",
    "ttemps_rttc",
    "tensorly_tucker",
    "tensor_toolbox_cp_wopt",
    "xinychen_halrtc",
    "authors_budgeted",
]

def _log_line(msg: str, use_tqdm: bool = False) -> None:
    if use_tqdm and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_float_tag(val: float) -> str:
    s = f"{float(val):.12g}"
    s = s.replace("-", "m").replace("+", "").replace(".", "p")
    return s


RUN_FROM_FILE_CONFIG = True

FILE_RUN_CONFIG = {
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
    "water_as_samples": 512,
    "water_as_random_state": 1729,
    "cheb_interval": [-0.3, 0.3],
    "cheb_points_list": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "n_points": 7,
    "tol": 1e-5,
    "tt_backend": "octave_tt_toolbox",
    "ttml_env_name": "matrix_approximation_final_3_11",
    "ttml_method": "dmrg",
    "octave_env_name": "octave",
    "author_rank_cap": 12,
    "author_max_iter": 30,
    "author_random_state": 1729,
    "author_log_first_unique": 8,
    "author_log_every_unique": 25,
    "author_log_every_total": 250,
    "author_rms_probe_points": 1000,
    "author_rms_seed": 2025,
    "random_state": 0,
    "show_tqdm": True,
    "verbose": True,
    # Main experiment mode.
    # single_budget: one budget, all baselines run on the same author-selected points.
    # budget_sweep: one author trajectory up to max budget, then all baselines run on its prefixes.
    # collect_only: only collect the author-selected points and save the trace.
    "run_mode": "budget_sweep",  # single_budget | budget_sweep | collect_only
    "sampling": {
        "policy": "fraction",  # fraction | absolute
        "budget_fraction": 0.2,
        "budget": 512,
        "budget_fraction_list": [0.2, 0.5, 1.0],
        "budget_fractions": [0.2, 0.5, 1.0],
        "budget_list": [64, 128, 256, 512],
        "budgets": [64, 128, 256, 512],
        "trace_source": "fresh",  # fresh | npz
        "trace_npz": "baranov2015_as_tt_experiments/results/faithful_control_panel/latest_trace.npz",
        "save_trace_npz": True,
    },
    "completion": {
        # Baselines that all see the SAME authors-driven online sampled points.
        # Canonical names:
        #   my_cheb, my_cheb_auto, ttemps_rttc, tensorly_tucker,
        #   tensor_toolbox_cp_wopt, xinychen_halrtc, authors_budgeted
        # Legacy aliases still work: tt_als, tucker, cp_wopt, halrtc.
        # Use `all` to expand to DEFAULT_BASELINES above.
        # Important: if my_cheb.ranks or my_cheb.lambda_all_list contain multiple values,
        # plain `my_cheb` is expanded automatically over that grid.
        # To run only one Cheb model, leave a single value in both lists.
        "baselines": "my_cheb,ttemps_rttc,tensorly_tucker,tensor_toolbox_cp_wopt,xinychen_halrtc,authors_budgeted",
        "random_test_points": 1000,
        "compute_exact_grid_metrics": True,
        "test_seed": 0,
    },
    "my_cheb": {
        "rank": 5,
        "ranks": [1, 5, 30, 50, 100],
        "strict_rank": True,
        "allow_rank_growth": False,
        "number_of_steps": 10,
        "tol_for_step": 1e-6,
        "begin": "oversample",
        "lambda_all": 0.1,
        "lambda_all_list": [0, 0.1, 1],
        "rank_eval": False,
        "rank_nest": False,
        "nest_iters": 5,
        "tol": 1e-6,
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
    "tt_als": {
        "rank": 12,
        "max_iter": 120,
        "tol": 1e-5,
        "verbose": True,
    },
    "tucker": {
        "rank": 12,
        "max_iter": 120,
        "tol": 1e-5,
        "verbose": True,
    },
    "cp_wopt": {
        "rank": 12,
        "max_iter": 120,
        "tol": 1e-5,
        "random_state": 1000,
        "verbose": True,
    },
    "halrtc": {
        "max_iter": 120,
        "tol": 1e-5,
        "verbose": True,
    },
    "authors_budgeted": {
        # If the authors online sampler actually converges within the requested budget,
        # we run the faithful authors surrogate once to report its error.
        "report_surrogate_if_converged": True,
    },
    "output_json": "baranov2015_as_tt_experiments/results/budgeted_experiments/baranov2015_budgeted_results.json",
    "output_md": "baranov2015_as_tt_experiments/results/budgeted_experiments/baranov2015_budgeted_results.md",
    "output_by_run_mode": True,
}





@dataclass
class RunRow:
    mode: str
    family: str
    cheb_points: int
    budget: int
    unique_budget: int
    baseline: str
    status: str
    observed_points: int
    total_queries: int
    unique_queries: int
    converged: bool | None
    sweeps_completed: int | None
    train_points: int
    val_points: int
    eval_points: int
    rank_label: str
    rmse_mev: float | None
    mae_mev: float | None
    max_abs_error_mev: float | None
    relative_rmse: float | None
    mape: float | None
    train_rmse_mev: float | None
    grid_hidden_rmse_mev: float | None
    offgrid_rmse_mev: float | None
    offgrid_max_abs_mev: float | None
    elapsed_sec: float
    returned_checkpoint: bool | None = None
    input_digest: str | None = None
    metric_profile: str | None = None
    note: str | None = None
    error: str | None = None


def _resolve_run_mode(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("run_mode", "single_budget")).strip().lower()
    valid = {"single_budget", "budget_sweep", "collect_only"}
    if mode not in valid:
        raise ValueError(f"run_mode must be one of {sorted(valid)}, got: {mode!r}")
    return mode


def _path_with_run_mode_suffix(path: Path, run_mode: str) -> Path:
    stem = path.stem
    for tag in ("single_budget", "budget_sweep", "collect_only"):
        suffix = f"_{tag}"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.with_name(f"{stem}_{run_mode}{path.suffix}")


def _path_with_cheb_points_suffix(path: Path, cheb_points: int) -> Path:
    return path.with_name(f"{path.stem}_n{int(cheb_points)}{path.suffix}")


def _resolve_baseline_names(raw: str) -> list[str]:
    raw_names = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not raw_names:
        raise ValueError("No completion baselines selected")
    names: list[str] = []
    for name in raw_names:
        if name == "all":
            names.extend(DEFAULT_BASELINES)
        elif name.startswith("my_cheb_r"):
            names.append(name)
        else:
            names.append(BASELINE_ALIASES.get(name, name))

    supported = set(BASELINE_ALIASES.values()) | set(DEFAULT_BASELINES)
    bad = [name for name in names if name not in supported and not name.startswith("my_cheb_r")]
    if bad:
        supported_human = sorted(set(BASELINE_ALIASES) | set(DEFAULT_BASELINES) | {"all"})
        raise ValueError(f"Unsupported baselines: {bad}. Supported: {supported_human}")

    dedup: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            dedup.append(name)
    return dedup


def _iter_my_cheb_specs(cfg: dict[str, Any]) -> list[tuple[int, float, str]]:
    my_cfg = dict(cfg["my_cheb"])
    raw_ranks = my_cfg.get("ranks", None)
    if raw_ranks is None:
        ranks = [int(my_cfg.get("rank", 10))]
    else:
        ranks = [int(x) for x in raw_ranks]
    raw_lambdas = my_cfg.get("lambda_all_list", None)
    if raw_lambdas is None:
        lambdas = [float(my_cfg.get("lambda_all", 0.0))]
    else:
        lambdas = [float(x) for x in raw_lambdas]
    out: list[tuple[int, float, str]] = []
    for rank in ranks:
        for lam in lambdas:
            name = f"my_cheb_r{int(rank)}"
            if len(lambdas) > 1:
                name += f"_lam{_format_float_tag(lam)}"
            out.append((int(rank), float(lam), name))
    return out


def _expand_my_cheb_tokens(names: list[str], cfg: dict[str, Any]) -> list[str]:
    auto_names = [name for _rank, _lam, name in _iter_my_cheb_specs(cfg)]
    auto_expand_plain = len(auto_names) > 1
    out: list[str] = []
    for name in names:
        if name == "my_cheb_auto":
            out.extend(auto_names)
        elif name == "my_cheb" and auto_expand_plain:
            out.extend(auto_names)
        else:
            out.append(name)
    dedup: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            seen.add(name)
            dedup.append(name)
    return dedup


def _my_cheb_params_for_name(cfg: dict[str, Any], baseline_name: str) -> dict[str, Any]:
    base = dict(cfg["my_cheb"])
    base.pop("ranks", None)
    base.pop("lambda_all_list", None)
    if baseline_name == "my_cheb":
        return base
    for rank, lam, name in _iter_my_cheb_specs(cfg):
        if baseline_name == name:
            base["rank"] = int(rank)
            base["lambda_all"] = float(lam)
            return base
    raise ValueError(f"Unknown my_cheb variant: {baseline_name}")


def _resolve_budget_value(*, policy: str, budget: int | float | None, fraction: float | None, total_points: int) -> int:
    if str(policy).strip().lower() == "absolute":
        if budget is None:
            raise ValueError("Absolute budget policy requires `budget`")
        raw = int(round(float(budget)))
    elif str(policy).strip().lower() == "fraction":
        if fraction is None:
            raise ValueError("Fraction budget policy requires `budget_fraction`")
        raw = int(np.ceil(float(fraction) * float(total_points)))
    else:
        raise ValueError(f"Unknown budget policy: {policy!r}")
    return max(1, min(int(total_points), int(raw)))


def _resolve_single_budget(cfg: dict[str, Any]) -> int:
    total_points = int(cfg["n_points"]) ** int(cfg["water_as_dim"])
    return _resolve_budget_value(
        policy=str(cfg["sampling"]["policy"]),
        budget=cfg["sampling"].get("budget"),
        fraction=cfg["sampling"].get("budget_fraction"),
        total_points=total_points,
    )


def _resolve_sweep_budgets(cfg: dict[str, Any]) -> list[int]:
    total_points = int(cfg["n_points"]) ** int(cfg["water_as_dim"])
    policy = str(cfg["sampling"]["policy"]).strip().lower()
    if policy == "absolute":
        raw = cfg["sampling"]["budgets"]
        budgets = [
            _resolve_budget_value(policy="absolute", budget=item, fraction=None, total_points=total_points)
            for item in raw
        ]
    elif policy == "fraction":
        raw = cfg["sampling"]["budget_fractions"]
        budgets = [
            _resolve_budget_value(policy="fraction", budget=None, fraction=item, total_points=total_points)
            for item in raw
        ]
    else:
        raise ValueError(f"Unknown budget policy: {policy!r}")
    return sorted(set(int(x) for x in budgets))


def _faithful_args_from_cfg(cfg: dict[str, Any]) -> Any:
    args_dict = {
        "data_root": str(cfg["data_root"]),
        "geometry_xyz": str(cfg["geometry_xyz"]),
        "cache_tensors": bool(cfg["cache_tensors"]),
        "generate_if_missing": bool(cfg["generate_if_missing"]),
        "qc_method": str(cfg["qc_method"]),
        "basis": str(cfg["basis"]),
        "charge": int(cfg["charge"]),
        "spin": int(cfg["spin"]),
        "scf_conv_tol": float(cfg["scf_conv_tol"]),
        "scf_max_cycle": int(cfg["scf_max_cycle"]),
        "scf_init_guess": str(cfg["scf_init_guess"]),
        "scf_verbose": int(cfg["scf_verbose"]),
        "scf_newton_fallback": bool(cfg["scf_newton_fallback"]),
        "water_coordinate_unit": str(cfg["water_coordinate_unit"]),
        "water_as_dim": int(cfg["water_as_dim"]),
        "water_as_sigma2": float(cfg["water_as_sigma2"]),
        "water_as_samples": int(cfg["water_as_samples"]),
        "water_as_random_state": int(cfg["water_as_random_state"]),
        "cheb_interval": [float(cfg["cheb_interval"][0]), float(cfg["cheb_interval"][1])],
        "author_rank_cap": int(cfg["author_rank_cap"]),
        "author_tol": float(cfg["tol"]),
        "author_max_iter": int(cfg["author_max_iter"]),
        "author_random_state": int(cfg["author_random_state"]),
        "author_log_first_unique": int(cfg["author_log_first_unique"]),
        "author_log_every_unique": int(cfg["author_log_every_unique"]),
        "author_log_every_total": int(cfg["author_log_every_total"]),
        "author_rms_probe_points": int(cfg["author_rms_probe_points"]),
        "author_rms_seed": int(cfg["author_rms_seed"]),
        "tt_backend": str(cfg["tt_backend"]),
        "ttml_env_name": str(cfg["ttml_env_name"]),
        "ttml_method": str(cfg["ttml_method"]),
        "octave_env_name": str(cfg["octave_env_name"]),
        "show_tqdm": bool(cfg["show_tqdm"]),
        "verbose": bool(cfg["verbose"]),
        "random_state": int(cfg["random_state"]),
    }
    return faithful._coerce_args(None, **args_dict)


def _resolve_cfg() -> dict[str, Any]:
    if not RUN_FROM_FILE_CONFIG:
        raise RuntimeError("Only FILE_RUN_CONFIG mode is enabled for this runner.")
    cfg = json.loads(json.dumps(FILE_RUN_CONFIG))
    cfg["data_root"] = str(water.resolve_project_path(cfg["data_root"]))
    cfg["geometry_xyz"] = str(water.resolve_project_path(cfg["geometry_xyz"]))
    cfg["output_json"] = str(water.resolve_project_path(cfg["output_json"]))
    cfg["output_md"] = str(water.resolve_project_path(cfg["output_md"]))
    cfg["cheb_interval"] = [float(cfg["cheb_interval"][0]), float(cfg["cheb_interval"][1])]
    cfg["n_points"] = int(cfg["n_points"])
    cfg["cheb_points_list"] = sorted(set(int(x) for x in cfg.get("cheb_points_list", [cfg["n_points"]])))
    cfg["sampling"]["trace_npz"] = str(water.resolve_project_path(cfg["sampling"]["trace_npz"]))
    cfg["run_mode"] = _resolve_run_mode(cfg)
    cfg["completion"]["baselines"] = str(cfg["completion"]["baselines"])
    cfg["completion"]["random_test_points"] = int(cfg["completion"]["random_test_points"])
    cfg["completion"]["compute_exact_grid_metrics"] = bool(cfg["completion"]["compute_exact_grid_metrics"])
    cfg["completion"]["test_seed"] = int(cfg["completion"]["test_seed"])
    cfg["my_cheb"]["rank"] = int(cfg["my_cheb"]["rank"])
    cfg["my_cheb"]["ranks"] = [int(x) for x in cfg["my_cheb"].get("ranks", [cfg["my_cheb"]["rank"]])]
    cfg["my_cheb"]["lambda_all"] = float(cfg["my_cheb"]["lambda_all"])
    cfg["my_cheb"]["lambda_all_list"] = [
        float(x) for x in cfg["my_cheb"].get("lambda_all_list", [cfg["my_cheb"]["lambda_all"]])
    ]
    cfg["sampling"]["policy"] = str(cfg["sampling"]["policy"])
    cfg["sampling"]["budget"] = int(cfg["sampling"]["budget"])
    cfg["sampling"]["budget_fraction"] = float(cfg["sampling"]["budget_fraction"])
    cfg["sampling"]["budgets"] = [
        int(x) for x in cfg["sampling"].get("budget_list", cfg["sampling"]["budgets"])
    ]
    cfg["sampling"]["budget_fractions"] = [
        float(x) for x in cfg["sampling"].get("budget_fraction_list", cfg["sampling"]["budget_fractions"])
    ]
    cfg["sampling"]["save_trace_npz"] = bool(cfg["sampling"]["save_trace_npz"])
    cfg["sampling"]["trace_source"] = str(cfg["sampling"]["trace_source"])
    cfg["output_by_run_mode"] = bool(cfg.get("output_by_run_mode", True))
    if cfg["output_by_run_mode"]:
        cfg["output_json"] = str(_path_with_run_mode_suffix(Path(cfg["output_json"]), cfg["run_mode"]))
        cfg["output_md"] = str(_path_with_run_mode_suffix(Path(cfg["output_md"]), cfg["run_mode"]))
    return cfg


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


def _total_queries_for_unique_prefix(trace: faithful.SampleTrace, unique_budget: int) -> int:
    target = int(min(int(unique_budget), int(trace.unique_queries)))
    if target <= 0:
        return 0
    seen: set[tuple[int, ...]] = set()
    total = 0
    for idx in np.asarray(trace.query_sequence, dtype=int):
        total += 1
        seen.add(tuple(int(x) for x in np.asarray(idx, dtype=int).reshape(-1)))
        if len(seen) >= target:
            return int(total)
    return int(trace.total_queries)


def _collect_shared_trace(cfg: dict[str, Any], args_ns: Any, max_budget: int, use_tqdm: bool) -> faithful.SampleTrace:
    trace_source = str(cfg["sampling"]["trace_source"]).strip().lower()
    trace_npz = _path_with_cheb_points_suffix(Path(cfg["sampling"]["trace_npz"]), int(cfg["n_points"]))
    if str(cfg["tt_backend"]) != "octave_tt_toolbox":
        raise ValueError(
            "Strict paper comparison requires `tt_backend='octave_tt_toolbox'` "
            "so that authors-driven sampling comes only from TT-Toolbox via Octave."
        )
    if trace_source == "npz":
        trace = _load_sample_trace_npz(trace_npz)
        if int(trace.unique_queries) < int(max_budget):
            raise ValueError(
                f"Loaded trace has only {trace.unique_queries} points, but current run needs at least {max_budget}"
            )
        if (not bool(trace.faithful_backend)) or ("octave_tt_toolbox" not in str(trace.tt_backend_used)):
            raise ValueError(
                "Loaded trace is not a faithful Octave TT-Toolbox trace. "
                f"Got faithful_backend={trace.faithful_backend}, backend_used={trace.tt_backend_used!r}."
            )
        _log_line(
            f"[sampling] loaded authors trajectory -> {trace_npz} "
            f"(status={trace.status}, unique={trace.unique_queries}, total={trace.total_queries})",
            use_tqdm=use_tqdm,
        )
        return trace
    if trace_source != "fresh":
        raise ValueError(f"trace_source must be 'fresh' or 'npz', got: {trace_source!r}")

    _log_line(
        f"[sampling] START authors online sampler max_unique_budget={max_budget}",
        use_tqdm=use_tqdm,
    )
    trace = faithful.collect_author_samples(
        args_ns,
        n_points=int(cfg["n_points"]),
        tol=float(cfg["tol"]),
        unique_budget=int(max_budget),
        tt_backend=str(cfg["tt_backend"]),
        random_state=int(cfg["random_state"]),
    )
    _log_line(
        f"[sampling] DONE authors online sampler status={trace.status} "
        f"unique={trace.unique_queries} total={trace.total_queries} backend={trace.tt_backend_used}",
        use_tqdm=use_tqdm,
    )
    if bool(cfg["sampling"]["save_trace_npz"]):
        _save_sample_trace_npz(trace_npz, trace)
        _log_line(f"[sampling] saved trajectory -> {trace_npz}", use_tqdm=use_tqdm)
    return trace


def _make_eval_payload(cfg: dict[str, Any], args_ns: Any, trace: faithful.SampleTrace) -> dict[str, Any]:
    interval = tuple(float(v) for v in trace.context.cheb_interval)
    test_points = faithful.sample_random_test_points(
        trace.context,
        n_test=int(cfg["completion"]["random_test_points"]),
        a=interval[0],
        b=interval[1],
        seed=int(cfg["completion"]["test_seed"]),
    )
    energy_fn = water._build_energy_fn(args_ns)
    payload: dict[str, Any] = {
        "test_points": np.asarray(test_points, dtype=float),
        "energy_fn": energy_fn,
    }
    if bool(cfg["completion"]["compute_exact_grid_metrics"]):
        try:
            exact_tensor_h, exact_nodes = faithful.load_or_generate_full_value_tensor(
                args_ns,
                context=trace.context,
                n_points=int(trace.shape[0]),
            )
            if np.max(np.abs(np.asarray(exact_nodes, dtype=float) - np.asarray(trace.nodes, dtype=float))) > 1e-12:
                raise ValueError("Exact grid tensor nodes mismatch against sample trace nodes")
            payload["exact_grid_tensor_h"] = np.asarray(exact_tensor_h, dtype=float)
        except Exception as exc:
            payload["exact_grid_error"] = str(exc)
    return payload


def _fixed_mask_config_block(cfg: dict[str, Any], canonical_name: str) -> dict[str, Any]:
    legacy_key_by_name = {
        "ttemps_rttc": "tt_als",
        "tensorly_tucker": "tucker",
        "tensor_toolbox_cp_wopt": "cp_wopt",
        "xinychen_halrtc": "halrtc",
    }
    key = legacy_key_by_name.get(str(canonical_name), str(canonical_name))
    return dict(cfg.get(key, {}))


def _build_fixed_mask_baseline(canonical_name: str, cfg: dict[str, Any], shape: tuple[int, ...]) -> Any:
    name = str(canonical_name)
    if name == "ttemps_rttc":
        params = _fixed_mask_config_block(cfg, name)
        return TTALS(
            tt_rank=max(1, int(params.get("rank", params.get("tt_rank", cfg["tt_als"]["rank"])))),
            max_iter=int(params.get("max_iter", cfg["tt_als"]["max_iter"])),
            tol=float(params.get("tol", cfg["tt_als"]["tol"])),
            verbose=bool(params.get("verbose", cfg["tt_als"]["verbose"])),
            octave_env_name=str(cfg["octave_env_name"]),
        )

    if name == "tensorly_tucker":
        params = _fixed_mask_config_block(cfg, name)
        if "ranks" in params:
            tucker_ranks = tuple(max(1, min(int(rank), int(dim))) for rank, dim in zip(params["ranks"], shape))
            if len(tucker_ranks) != len(shape):
                raise ValueError("tensorly_tucker.ranks must contain one rank per tensor mode")
        else:
            tucker_rank = int(params.get("rank", cfg["tucker"]["rank"]))
            tucker_ranks = tuple(max(1, min(tucker_rank, d)) for d in shape)
        return TuckerCompletion(
            ranks=tucker_ranks,
            max_iter=int(params.get("max_iter", cfg["tucker"]["max_iter"])),
            tol=float(params.get("tol", cfg["tucker"]["tol"])),
            verbose=bool(params.get("verbose", cfg["tucker"]["verbose"])),
            random_state=int(params.get("random_state", cfg.get("random_state", 0))),
        )

    if name == "tensor_toolbox_cp_wopt":
        params = _fixed_mask_config_block(cfg, name)
        return CPWOPT(
            rank=max(1, int(params.get("rank", cfg["cp_wopt"]["rank"]))),
            max_iter=int(params.get("max_iter", cfg["cp_wopt"]["max_iter"])),
            tol=float(params.get("tol", cfg["cp_wopt"]["tol"])),
            random_state=int(params.get("random_state", cfg["cp_wopt"]["random_state"])),
            verbose=bool(params.get("verbose", cfg["cp_wopt"]["verbose"])),
            octave_env_name=str(cfg["octave_env_name"]),
        )

    if name == "xinychen_halrtc":
        params = _fixed_mask_config_block(cfg, name)
        return HaLRTC(
            max_iter=int(params.get("max_iter", cfg["halrtc"]["max_iter"])),
            tol=float(params.get("tol", cfg["halrtc"]["tol"])),
            verbose=bool(params.get("verbose", cfg["halrtc"]["verbose"])),
            alpha=params.get("alpha", None),
            rho=float(params.get("rho", 1e-4)),
            rho_scale=float(params.get("rho_scale", 1.05)),
            rho_max=float(params.get("rho_max", 1e5)),
        )

    raise ValueError(f"Unknown fixed-mask baseline: {canonical_name!r}")


def _build_fixed_mask_baselines(
    cfg: dict[str, Any],
    shape: tuple[int, ...],
    selected_baselines: list[str] | None = None,
) -> dict[str, Any]:
    if selected_baselines is None:
        selected = sorted(CANONICAL_FIXED_MASK_BASELINES)
    else:
        selected = sorted({BASELINE_ALIASES.get(str(name), str(name)) for name in selected_baselines})
    return {
        name: _build_fixed_mask_baseline(name, cfg, shape)
        for name in selected
        if name in CANONICAL_FIXED_MASK_BASELINES
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


def _scale_trace_observations(trace: faithful.SampleTrace) -> tuple[np.ndarray, np.ndarray, float]:
    observed_h, mask = _observed_tensor_and_mask(trace)
    reference_h = faithful._relative_mev_reference_from_observations(trace.unique_values)
    observed_scaled = np.zeros_like(observed_h, dtype=float)
    if np.any(mask):
        observed_scaled[mask] = faithful._to_relative_mev_with_reference(observed_h[mask], reference_h)
    return observed_scaled, mask, float(reference_h)


def _input_digest_from_trace(trace: faithful.SampleTrace) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(trace.shape, dtype=np.int64).tobytes())
    h.update(np.asarray(trace.unique_indices, dtype=np.int64).reshape(-1).tobytes())
    h.update(np.asarray(trace.unique_values, dtype=np.float64).reshape(-1).tobytes())
    return h.hexdigest()[:16]


def _validate_completed_tensor(baseline: str, completed_tensor: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(completed_tensor, dtype=float)
    if arr.shape != tuple(expected_shape):
        raise ValueError(f"{baseline} returned tensor with shape {arr.shape}, expected {tuple(expected_shape)}")
    if not np.all(np.isfinite(arr)):
        bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
        raise ValueError(f"{baseline} returned {bad} non-finite tensor entries")
    return arr


def _rank_label_for_baseline(name: str, cfg: dict[str, Any], completion_result: faithful.CompletionResult | None = None) -> str:
    name = BASELINE_ALIASES.get(str(name), str(name))
    if name == "my_cheb" or name.startswith("my_cheb_r"):
        if completion_result is not None:
            info = dict(completion_result.info)
            effective_rank = info.get("effective_rank")
            if effective_rank is not None:
                return str(int(effective_rank))
        params = _my_cheb_params_for_name(cfg, name)
        return str(int(params["rank"]))
    if name == "ttemps_rttc":
        return str(int(_fixed_mask_config_block(cfg, name).get("rank", cfg["tt_als"]["rank"])))
    if name == "tensor_toolbox_cp_wopt":
        return str(int(_fixed_mask_config_block(cfg, name).get("rank", cfg["cp_wopt"]["rank"])))
    if name == "tensorly_tucker":
        params = _fixed_mask_config_block(cfg, name)
        if "ranks" in params:
            return "x".join(str(int(x)) for x in params["ranks"])
        return str(int(params.get("rank", cfg["tucker"]["rank"])))
    if name == "xinychen_halrtc":
        return "n/a"
    if name == "authors_budgeted":
        return "online"
    return "n/a"


def _row_from_my_cheb(
    *,
    mode: str,
    cheb_points: int,
    budget: int,
    total_queries: int,
    result: faithful.CompletionResult,
    cfg: dict[str, Any],
) -> RunRow:
    info = dict(result.info)
    test_metrics = dict(result.test_metrics)
    train_metrics = dict(result.train_metrics)
    return RunRow(
        mode=str(mode),
        family="completion",
        cheb_points=int(cheb_points),
        budget=int(budget),
        unique_budget=int(budget),
        baseline="my_cheb",
        status="ok",
        observed_points=int(info.get("observed_points", budget)),
        total_queries=int(total_queries),
        unique_queries=int(info.get("observed_points", budget)),
        converged=None,
        sweeps_completed=None,
        train_points=int(info.get("train_points_planned", budget)),
        val_points=int(info.get("val_points_planned", 0)),
        eval_points=int(test_metrics.get("n_test", cfg["completion"]["random_test_points"])),
        rank_label=_rank_label_for_baseline("my_cheb", cfg, result),
        rmse_mev=test_metrics.get("completion_hidden_rmse_mev"),
        mae_mev=test_metrics.get("completion_hidden_mae_mev"),
        max_abs_error_mev=test_metrics.get("completion_hidden_max_abs_mev"),
        relative_rmse=test_metrics.get("completion_hidden_relative_rmse"),
        mape=test_metrics.get("completion_hidden_mape"),
        train_rmse_mev=train_metrics.get("rmse_mev"),
        grid_hidden_rmse_mev=test_metrics.get("grid_hidden_rmse_mev"),
        offgrid_rmse_mev=test_metrics.get("rmse_mev"),
        offgrid_max_abs_mev=test_metrics.get("max_abs_mev"),
        elapsed_sec=float(info.get("elapsed_sec", float("nan"))),
        returned_checkpoint=None,
        note=None,
        error=None,
    )


def _row_from_fixed_mask(
    *,
    mode: str,
    cheb_points: int,
    budget: int,
    total_queries: int,
    baseline: str,
    trace: faithful.SampleTrace,
    completed_tensor_h: np.ndarray,
    elapsed_sec: float,
    info: dict[str, Any],
    eval_payload: dict[str, Any],
    cfg: dict[str, Any],
) -> RunRow:
    indices = np.asarray(trace.unique_indices, dtype=int)
    values_h = np.asarray(trace.unique_values, dtype=float)
    observed_pred = np.asarray(completed_tensor_h, dtype=float)[tuple(indices.T)]
    train_metrics = faithful._simple_point_metrics(values_h, observed_pred)
    test_metrics = faithful.evaluate_completed_tensor_on_test(
        completed_tensor=np.asarray(completed_tensor_h, dtype=float),
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
                completed_tensor=np.asarray(completed_tensor_h, dtype=float),
                exact_tensor_h=np.asarray(eval_payload["exact_grid_tensor_h"], dtype=float),
                observed_indices=indices,
            )
        )
        test_metrics.update(
            faithful.evaluate_completion_metrics_dual_style(
                completed_tensor_h=np.asarray(completed_tensor_h, dtype=float),
                exact_tensor_h=np.asarray(eval_payload["exact_grid_tensor_h"], dtype=float),
                observed_indices=indices,
            )
        )
    return RunRow(
        mode=str(mode),
        family="completion",
        cheb_points=int(cheb_points),
        budget=int(budget),
        unique_budget=int(budget),
        baseline=str(baseline),
        status="ok",
        observed_points=int(len(indices)),
        total_queries=int(total_queries),
        unique_queries=int(len(indices)),
        converged=None,
        sweeps_completed=None,
        train_points=int(len(indices)),
        val_points=0,
        eval_points=int(test_metrics.get("n_test", cfg["completion"]["random_test_points"])),
        rank_label=_rank_label_for_baseline(baseline, cfg),
        rmse_mev=test_metrics.get("completion_hidden_rmse_mev"),
        mae_mev=test_metrics.get("completion_hidden_mae_mev"),
        max_abs_error_mev=test_metrics.get("completion_hidden_max_abs_mev"),
        relative_rmse=test_metrics.get("completion_hidden_relative_rmse"),
        mape=test_metrics.get("completion_hidden_mape"),
        train_rmse_mev=train_metrics.get("rmse_mev"),
        grid_hidden_rmse_mev=test_metrics.get("grid_hidden_rmse_mev"),
        offgrid_rmse_mev=test_metrics.get("rmse_mev"),
        offgrid_max_abs_mev=test_metrics.get("max_abs_mev"),
        elapsed_sec=float(elapsed_sec),
        returned_checkpoint=None,
        note=None,
        error=None,
    )


def _row_from_error(
    mode: str,
    family: str,
    cheb_points: int,
    budget: int,
    total_queries: int,
    baseline: str,
    observed_points: int,
    elapsed_sec: float,
    error: Exception | str,
    input_digest: str | None = None,
    metric_profile: str | None = None,
) -> RunRow:
    return RunRow(
        mode=str(mode),
        family=str(family),
        cheb_points=int(cheb_points),
        budget=int(budget),
        unique_budget=int(budget),
        baseline=str(baseline),
        status="fail",
        observed_points=int(observed_points),
        total_queries=int(total_queries),
        unique_queries=int(observed_points),
        converged=None,
        sweeps_completed=None,
        train_points=int(observed_points),
        val_points=0,
        eval_points=0,
        rank_label="n/a",
        rmse_mev=None,
        mae_mev=None,
        max_abs_error_mev=None,
        relative_rmse=None,
        mape=None,
        train_rmse_mev=None,
        grid_hidden_rmse_mev=None,
        offgrid_rmse_mev=None,
        offgrid_max_abs_mev=None,
        elapsed_sec=float(elapsed_sec),
        returned_checkpoint=None,
        input_digest=input_digest,
        metric_profile=metric_profile,
        note=None,
        error=str(error),
    )


def _stamp_row_input_and_metrics(row: RunRow, *, input_digest: str | None, metric_profile: str | None) -> RunRow:
    row.input_digest = input_digest
    row.metric_profile = metric_profile
    return row


def _ensure_authors_budgeted_result(
    *,
    cache: dict[str, Any],
    cfg: dict[str, Any],
    args_ns: Any,
    budget: int,
    use_tqdm: bool,
) -> faithful.AuthorsBaselineResult:
    key = (int(cfg["n_points"]), int(budget))
    if key in cache:
        return cache[key]["result"]
    _log_line(
        f"[authors_budgeted] building authors surrogate for n={int(cfg['n_points'])} under budget={int(budget)}",
        use_tqdm=use_tqdm,
    )
    t0 = time.time()
    result = faithful.run_baranov2015_water_budgeted(
        args_ns,
        n_points=int(cfg["n_points"]),
        tol=float(cfg["tol"]),
        unique_budget=int(budget),
        tt_backend=str(cfg["tt_backend"]),
        random_state=int(cfg["random_state"]),
    )
    cache[key] = {
        "result": result,
        "elapsed_sec": float(time.time() - t0),
    }
    return result


def _ensure_authors_full_result(
    *,
    cache: dict[int, Any],
    cfg: dict[str, Any],
    args_ns: Any,
    use_tqdm: bool,
) -> faithful.AuthorsBaselineResult:
    key = int(cfg["n_points"])
    if key in cache:
        return cache[key]["result"]
    _log_line(
        f"[authors_full] building full faithful authors baseline for n={int(cfg['n_points'])}",
        use_tqdm=use_tqdm,
    )
    t0 = time.time()
    result = faithful.run_baranov2015_water_baseline(
        args_ns,
        n_points=int(cfg["n_points"]),
        tol=float(cfg["tol"]),
        tt_backend=str(cfg["tt_backend"]),
        random_state=int(cfg["random_state"]),
    )
    cache[key] = {
        "result": result,
        "elapsed_sec": float(time.time() - t0),
    }
    return result


def _row_from_authors_full(
    *,
    mode: str,
    cheb_points: int,
    result: faithful.AuthorsBaselineResult,
    eval_payload: dict[str, Any],
    elapsed_sec: float,
) -> RunRow:
    info = dict(result.info)
    converged = bool(info.get("converged", True))
    status = "ok" if converged else str(info.get("status", "max_sweeps_reached"))
    offgrid_metrics: dict[str, float] = {}
    if converged and result.coeff_tensor is not None:
        offgrid_metrics = faithful.evaluate_surrogate_on_test(
            context=result.context,
            test_points=np.asarray(eval_payload["test_points"], dtype=float),
            energy_fn=eval_payload["energy_fn"],
            a=float(result.context.cheb_interval[0]),
            b=float(result.context.cheb_interval[1]),
            coeff_tensor=result.coeff_tensor,
            coeff_tt_cores=None,
        )
    return RunRow(
        mode=str(mode),
        family="authors_full",
        cheb_points=int(cheb_points),
        budget=int(result.unique_queries),
        unique_budget=int(result.unique_queries),
        baseline="authors_full",
        status=status,
        observed_points=int(result.unique_queries),
        total_queries=int(result.total_queries),
        unique_queries=int(result.unique_queries),
        converged=converged,
        sweeps_completed=None if info.get("sweeps_completed") is None else int(info.get("sweeps_completed")),
        train_points=0,
        val_points=0,
        eval_points=int(offgrid_metrics.get("n_test", 0)),
        rank_label="x".join(str(int(x)) for x in result.tt_ranks),
        rmse_mev=None,
        mae_mev=None,
        max_abs_error_mev=None,
        relative_rmse=None,
        mape=None,
        train_rmse_mev=None,
        grid_hidden_rmse_mev=None,
        offgrid_rmse_mev=offgrid_metrics.get("rmse_mev") if converged else None,
        offgrid_max_abs_mev=offgrid_metrics.get("max_abs_mev") if converged else None,
        elapsed_sec=float(elapsed_sec),
        returned_checkpoint=None,
        input_digest=None,
        metric_profile="authors_offgrid_v1",
        note="Full faithful AS+TT baseline via Octave TT-Toolbox.",
        error=None,
    )


def _row_from_authors_budgeted(
    *,
    mode: str,
    cheb_points: int,
    budget: int,
    prefix: faithful.SampleTrace,
    shared_trace: faithful.SampleTrace,
    cfg: dict[str, Any],
    args_ns: Any,
    eval_payload: dict[str, Any],
    authors_surrogate_cache: dict[str, Any],
    use_tqdm: bool,
) -> RunRow:
    baseline = _ensure_authors_budgeted_result(
        cache=authors_surrogate_cache,
        cfg=cfg,
        args_ns=args_ns,
        budget=int(budget),
        use_tqdm=use_tqdm,
    )
    raw_status = str(baseline.info.get("status", "authors_budgeted"))
    converged = bool(baseline.info.get("converged", False)) and raw_status == "converged_before_budget"
    family = "authors_budgeted_converged" if converged else "authors_budgeted_partial"
    label = "authors_converged_under_budget" if converged else "authors_partial_under_budget"
    status = "ok" if converged else raw_status
    sweeps_completed = baseline.info.get("sweeps_completed")
    offgrid_metrics: dict[str, float] = {}
    if converged and baseline.coeff_tensor is not None:
        offgrid_metrics = faithful.evaluate_surrogate_on_test(
            context=baseline.context,
            test_points=np.asarray(eval_payload["test_points"], dtype=float),
            energy_fn=eval_payload["energy_fn"],
            a=float(baseline.context.cheb_interval[0]),
            b=float(baseline.context.cheb_interval[1]),
            coeff_tensor=baseline.coeff_tensor,
            coeff_tt_cores=None,
        )
    return RunRow(
        mode=str(mode),
        family=family,
        cheb_points=int(cheb_points),
        budget=int(budget),
        unique_budget=int(budget),
        baseline=label,
        status=status,
        observed_points=int(baseline.unique_queries),
        total_queries=int(baseline.total_queries),
        unique_queries=int(baseline.unique_queries),
        converged=bool(converged),
        sweeps_completed=None if sweeps_completed is None else int(sweeps_completed),
        train_points=0,
        val_points=0,
        eval_points=int(offgrid_metrics.get("n_test", 0)) if converged else 0,
        rank_label="x".join(str(int(x)) for x in baseline.tt_ranks),
        rmse_mev=None,
        mae_mev=None,
        max_abs_error_mev=None,
        relative_rmse=None,
        mape=None,
        train_rmse_mev=None,
        grid_hidden_rmse_mev=None,
        offgrid_rmse_mev=offgrid_metrics.get("rmse_mev") if converged else None,
        offgrid_max_abs_mev=offgrid_metrics.get("max_abs_mev") if converged else None,
        elapsed_sec=float(authors_surrogate_cache[(int(cfg["n_points"]), int(budget))]["elapsed_sec"]),
        returned_checkpoint=None if baseline.info.get("returned_checkpoint") is None else bool(baseline.info.get("returned_checkpoint")),
        note=(
            f"Authors TT-cross evaluated under budget={int(budget)}. "
            f"Returned status={raw_status}, total_queries={baseline.total_queries}, unique_queries={baseline.unique_queries}, "
            f"returned_checkpoint={bool(baseline.info.get('returned_checkpoint', False))}, "
            f"sweeps_completed={baseline.info.get('sweeps_completed')}, last_block={baseline.info.get('last_block')}."
        ),
        error=None,
    )


def _run_one_baseline(
    *,
    baseline_name: str,
    mode: str,
    cheb_points: int,
    budget: int,
    total_queries: int,
    prefix: faithful.SampleTrace,
    shared_trace: faithful.SampleTrace,
    cfg: dict[str, Any],
    args_ns: Any,
    eval_payload: dict[str, Any],
    fixed_mask_models: dict[str, Any],
    authors_surrogate_cache: dict[str, Any],
    use_tqdm: bool,
) -> RunRow:
    t0 = time.time()
    input_digest = _input_digest_from_trace(prefix)
    metric_profile = "shared_prefix_completion_v1"
    if baseline_name == "my_cheb" or baseline_name.startswith("my_cheb_r"):
        kwargs = _my_cheb_params_for_name(cfg, baseline_name)
        kwargs.update(eval_payload)
        result = faithful.run_completion_on_author_samples(prefix, kwargs)
        row = _row_from_my_cheb(
            mode=mode,
            cheb_points=cheb_points,
            budget=budget,
            total_queries=total_queries,
            result=result,
            cfg=cfg,
        )
        row.baseline = str(baseline_name)
        row.rank_label = _rank_label_for_baseline(baseline_name, cfg, result)
        return _stamp_row_input_and_metrics(row, input_digest=input_digest, metric_profile=metric_profile)

    if baseline_name == "authors_budgeted":
        row = _row_from_authors_budgeted(
            mode=mode,
            cheb_points=cheb_points,
            budget=budget,
            prefix=prefix,
            shared_trace=shared_trace,
            cfg=cfg,
            args_ns=args_ns,
            eval_payload=eval_payload,
            authors_surrogate_cache=authors_surrogate_cache,
            use_tqdm=use_tqdm,
        )
        return _stamp_row_input_and_metrics(row, input_digest=input_digest, metric_profile="authors_budgeted_offgrid_v1")

    observed_scaled, mask, reference_h = _scale_trace_observations(prefix)
    model = fixed_mask_models[baseline_name]
    result = model.fit_transform(observed_tensor=observed_scaled, mask=mask, full_tensor=None)
    completed_scaled = _validate_completed_tensor(baseline_name, result.tensor, tuple(prefix.shape))
    completed_tensor_h = faithful._from_relative_mev_with_reference(completed_scaled, reference_h)
    row = _row_from_fixed_mask(
        mode=mode,
        cheb_points=cheb_points,
        budget=budget,
        total_queries=total_queries,
        baseline=baseline_name,
        trace=prefix,
        completed_tensor_h=completed_tensor_h,
        elapsed_sec=float(time.time() - t0),
        info={} if result.info is None else dict(result.info),
        eval_payload=eval_payload,
        cfg=cfg,
    )
    return _stamp_row_input_and_metrics(row, input_digest=input_digest, metric_profile=metric_profile)


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return value
    if not np.isfinite(val):
        return None
    return val


def _requested_unique_budget_cell(row: RunRow) -> int | str | None:
    if row.family == "authors_full":
        return "full"
    return int(row.unique_budget)


def _is_main_comparison_row(row: RunRow) -> bool:
    if row.family == "authors_full":
        return True
    if row.baseline == "authors_converged_under_budget":
        return True
    return row.family == "completion"


def _is_anytime_row(row: RunRow) -> bool:
    return row.baseline in {"authors_partial_under_budget", "authors_budgeted"}


def _offgrid_rmse_sort_key(row: RunRow) -> float:
    offgrid = _finite_or_none(row.offgrid_rmse_mev)
    return float("inf") if offgrid is None else float(offgrid)


def _requested_budget_sort_key(row: RunRow) -> tuple[int, float]:
    requested = _requested_unique_budget_cell(row)
    if isinstance(requested, str) or requested is None:
        return (1, float("inf"))
    return (0, float(requested))


def _main_comparison_records(rows: list[RunRow]) -> list[dict[str, Any]]:
    selected = [row for row in rows if _is_main_comparison_row(row)]

    def _sort_key(row: RunRow) -> tuple[Any, ...]:
        return (
            int(row.cheb_points),
            _requested_budget_sort_key(row),
            _offgrid_rmse_sort_key(row),
            int(row.total_queries),
            str(row.baseline),
        )

    selected.sort(key=_sort_key)
    records: list[dict[str, Any]] = []
    for row in selected:
        is_completion = not str(row.baseline).startswith("authors")
        records.append(
            {
                "n": int(row.cheb_points),
                "baseline": str(row.baseline),
                "status": str(row.status),
                "requested unique budget": _requested_unique_budget_cell(row),
                "actual total evals": int(row.total_queries),
                "actual unique points": int(row.unique_queries),
                "observed points": int(row.observed_points),
                "rank": str(row.rank_label),
                "off-grid RMSE (meV)": _finite_or_none(row.offgrid_rmse_mev),
                "off-grid Chebyshev (meV)": _finite_or_none(row.offgrid_max_abs_mev),
                "hidden RMSE (meV)": _finite_or_none(row.rmse_mev) if is_completion else None,
                "hidden Chebyshev (meV)": _finite_or_none(row.max_abs_error_mev) if is_completion else None,
                "time(s)": _finite_or_none(row.elapsed_sec),
            }
        )
    return records


def _authors_anytime_records(rows: list[RunRow]) -> list[dict[str, Any]]:
    selected = [row for row in rows if _is_anytime_row(row)]
    selected.sort(
        key=lambda row: (
            int(row.cheb_points),
            int(row.unique_budget),
            _offgrid_rmse_sort_key(row),
            int(row.total_queries),
            str(row.baseline),
        )
    )
    records: list[dict[str, Any]] = []
    for row in selected:
        records.append(
            {
                "n": int(row.cheb_points),
                "status": str(row.status),
                "requested unique budget": int(row.unique_budget),
                "actual total evals": int(row.total_queries),
                "actual unique points": int(row.unique_queries),
                "returned_checkpoint": None if row.returned_checkpoint is None else bool(row.returned_checkpoint),
                "converged": None if row.converged is None else bool(row.converged),
                "sweeps": None if row.sweeps_completed is None else int(row.sweeps_completed),
                "rank": str(row.rank_label),
                "off-grid RMSE (meV)": _finite_or_none(row.offgrid_rmse_mev),
                "off-grid Chebyshev (meV)": _finite_or_none(row.offgrid_max_abs_mev),
                "time(s)": _finite_or_none(row.elapsed_sec),
            }
        )
    return records


def _settings_budget_summary_records(rows: list[RunRow], settings: dict[str, Any]) -> list[dict[str, Any]]:
    authors_full_by_n = {
        int(row.cheb_points): row
        for row in rows
        if row.family == "authors_full"
    }
    out: list[dict[str, Any]] = []
    run_mode = str(settings.get("run_mode", "single_budget"))
    single_map = dict(settings.get("single_budget_resolved_by_cheb_points", {}))
    sweep_map = dict(settings.get("sweep_budgets_resolved_by_cheb_points", {}))
    for n in sorted(int(x) for x in settings.get("cheb_points_list", [])):
        if run_mode == "budget_sweep":
            requested_budgets = [int(x) for x in sweep_map.get(str(n), [])]
        else:
            requested_budgets = [int(single_map[str(n)])] if str(n) in single_map else []
        authors_full = authors_full_by_n.get(int(n))
        out.append(
            {
                "n": int(n),
                "tensor size = n^4": int(n) ** 4,
                "requested unique budgets": requested_budgets,
                "authors full total evals": None if authors_full is None else int(authors_full.total_queries),
                "authors full unique evals": None if authors_full is None else int(authors_full.unique_queries),
            }
        )
    return out


def _input_audit_records(rows: list[RunRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str | None], dict[str, Any]] = {}
    for row in rows:
        if row.family != "completion":
            continue
        if not row.input_digest:
            continue
        key = (int(row.cheb_points), int(row.unique_budget), row.input_digest)
        record = grouped.setdefault(
            key,
            {
                "n": int(row.cheb_points),
                "requested unique budget": int(row.unique_budget),
                "input digest": row.input_digest,
                "observed points": int(row.observed_points),
                "metric profile": row.metric_profile,
                "rows using input": 0,
                "baselines": [],
            },
        )
        record["rows using input"] = int(record["rows using input"]) + 1
        record["baselines"].append(str(row.baseline))
    records = list(grouped.values())
    for record in records:
        record["baselines"] = ", ".join(sorted(set(record["baselines"])))
    records.sort(key=lambda x: (int(x["n"]), int(x["requested unique budget"]), str(x["input digest"])))
    return records


def _error_records(rows: list[RunRow]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if str(row.status) != "fail" and not row.error:
            continue
        records.append(
            {
                "n": int(row.cheb_points),
                "baseline": str(row.baseline),
                "status": str(row.status),
                "requested unique budget": _requested_unique_budget_cell(row),
                "input digest": row.input_digest,
                "time(s)": _finite_or_none(row.elapsed_sec),
                "error": row.error,
            }
        )
    records.sort(key=lambda x: (int(x["n"]), str(x["baseline"]), str(x["requested unique budget"])))
    return records


def _format_markdown_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value) if value else "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.6f}"
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|")
    if len(text) > 260:
        text = text[:257] + "..."
    return text


def _markdown_table(headers: list[str], records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if not records:
        lines.append("No rows.")
        return lines
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for record in records:
        lines.append("| " + " | ".join(_format_markdown_cell(record.get(header)) for header in headers) + " |")
    return lines


def _markdown_grouped_by_n(headers: list[str], records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["No rows."]
    lines: list[str] = []
    n_values = sorted({int(record["n"]) for record in records if "n" in record})
    for idx, n_value in enumerate(n_values):
        if idx > 0:
            lines.append("")
        lines.append(f"### n = {n_value}")
        lines.append("")
        group = [record for record in records if int(record.get("n", -1)) == int(n_value)]
        lines.extend(_markdown_table(headers, group))
    return lines


def _build_report_tables(rows: list[RunRow], settings: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "main_paper_comparison": _main_comparison_records(rows),
        "authors_anytime_partial_budget": _authors_anytime_records(rows),
        "settings_budget_summary": _settings_budget_summary_records(rows, settings),
        "input_metric_audit": _input_audit_records(rows),
        "errors": _error_records(rows),
    }


def _markdown_report(rows: list[RunRow], settings: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Baranov 2015 Paper-Style Comparison")
    lines.append("")
    run_status = dict(settings.get("run_status", {}))
    if run_status:
        lines.append("Run status:")
        lines.append(f"- State: `{run_status.get('state', 'unknown')}`")
        lines.append(f"- Last update: `{run_status.get('updated_at_local', 'unknown')}`")
        lines.append(f"- Rows written: `{run_status.get('rows_written', len(rows))}`")
        if run_status.get("last_event"):
            lines.append(f"- Last event: `{run_status.get('last_event')}`")
        if run_status.get("error"):
            lines.append(f"- Top-level error: `{run_status.get('error')}`")
        lines.append("")
    lines.append("Methodology:")
    lines.append("- Authors-driven sampling is produced only by Octave + TT-Toolbox `dmrg_cross`.")
    lines.append("- Completion baselines use prefixes of the same authors trace.")
    lines.append("- Fixed-mask baselines receive the same zero-filled relative-meV tensor and boolean mask for each `(n, budget)`.")
    lines.append("- Cheb receives the same prefix trace; all completion rows are scored by the same off-grid and exact-grid metric functions.")
    lines.append("- Authors surrogate evaluation uses only `dense_tensor -> coeff_tensor`, without Python TT re-rounding.")
    lines.append("- Primary comparison metric is off-grid RMSE (meV) on random AS-domain test points.")
    lines.append("- `completion_hidden_rmse_mev` and `grid_hidden_rmse_mev` are reported only for completion methods.")
    lines.append("- Partial authors runs are listed only as diagnostics and are not compared by RMSE.")
    lines.append("")
    lines.append("## Settings")
    lines.append(f"- Geometry: `{settings['geometry_xyz']}`")
    lines.append(f"- Coordinate unit: `{settings['water_coordinate_unit']}`")
    lines.append(f"- Active subspace dim: `{settings['water_as_dim']}`")
    lines.append(f"- Active subspace sigma^2: `{settings['water_as_sigma2']}`")
    lines.append(f"- Active subspace samples: `{settings['water_as_samples']}`")
    lines.append(f"- Chebyshev interval: `{settings['cheb_interval']}`")
    lines.append(f"- Chebyshev points: `{settings['cheb_points_list']}`")
    lines.append(f"- Run mode: `{settings['run_mode']}`")
    lines.append(f"- Authors TT backend: `{settings['tt_backend']}`")
    lines.append(f"- Octave env: `{settings['octave_env_name']}`")
    lines.append(f"- Sampling policy: `{settings['sampling_policy']}`")
    lines.append(f"- Single budget resolved by Chebyshev points: `{settings['single_budget_resolved_by_cheb_points']}`")
    lines.append(f"- Sweep budgets resolved by Chebyshev points: `{settings['sweep_budgets_resolved_by_cheb_points']}`")
    lines.append(f"- Baselines: `{settings['baselines']}`")
    lines.append(f"- Resolved baselines: `{settings['resolved_baselines']}`")
    lines.append(f"- My Cheb ranks: `{settings['my_cheb_ranks']}`")
    lines.append(f"- My Cheb lambda_all_list: `{settings['my_cheb_lambda_all_list']}`")
    lines.append(f"- Trace source: `{settings['trace_source']}`")
    lines.append(f"- Exact-grid fallback errors: `{settings.get('exact_grid_errors_by_cheb_points', {})}`")
    lines.append("")
    tables = _build_report_tables(rows, settings)
    if not any(tables.values()):
        lines.append("No model rows were produced.")
        return "\n".join(lines) + "\n"

    lines.append("## Main Paper Comparison")
    lines.extend(
        _markdown_grouped_by_n(
            [
                "n",
                "baseline",
                "status",
                "requested unique budget",
                "actual total evals",
                "actual unique points",
                "observed points",
                "rank",
                "off-grid RMSE (meV)",
                "off-grid Chebyshev (meV)",
                "hidden RMSE (meV)",
                "hidden Chebyshev (meV)",
                "time(s)",
            ],
            tables["main_paper_comparison"],
        )
    )
    lines.append("")
    lines.append("## Authors Anytime / Partial-Budget")
    lines.extend(
        _markdown_grouped_by_n(
            [
                "n",
                "status",
                "requested unique budget",
                "actual total evals",
                "actual unique points",
                "returned_checkpoint",
                "converged",
                "sweeps",
                "rank",
                "off-grid RMSE (meV)",
                "off-grid Chebyshev (meV)",
                "time(s)",
            ],
            tables["authors_anytime_partial_budget"],
        )
    )
    lines.append("")
    lines.append("## Settings / Budget Summary")
    lines.extend(
        _markdown_table(
            [
                "n",
                "tensor size = n^4",
                "requested unique budgets",
                "authors full total evals",
                "authors full unique evals",
            ],
            tables["settings_budget_summary"],
        )
    )
    lines.append("")
    lines.append("## Input / Metric Audit")
    lines.extend(
        _markdown_table(
            [
                "n",
                "requested unique budget",
                "input digest",
                "observed points",
                "metric profile",
                "rows using input",
                "baselines",
            ],
            tables["input_metric_audit"],
        )
    )
    lines.append("")
    lines.append("## Errors")
    lines.extend(
        _markdown_table(
            [
                "n",
                "baseline",
                "status",
                "requested unique budget",
                "input digest",
                "time(s)",
                "error",
            ],
            tables["errors"],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_outputs(
    *,
    cfg: dict[str, Any],
    rows: list[RunRow],
    settings: dict[str, Any],
    state: str,
    last_event: str,
    use_tqdm: bool,
    error: Exception | str | None = None,
    log: bool = False,
) -> None:
    status = {
        "state": str(state),
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "rows_written": int(len(rows)),
        "last_event": str(last_event),
        "error": None if error is None else str(error),
    }
    settings["run_status"] = status
    report_tables = _build_report_tables(rows, settings)
    payload = {
        "settings": settings,
        "tables": report_tables,
        "rows_raw": [asdict(row) for row in rows],
    }

    output_json = Path(cfg["output_json"])
    output_md = Path(cfg["output_md"])
    _atomic_write_text(output_json, json.dumps(payload, indent=2, default=_json_default))
    _atomic_write_text(output_md, _markdown_report(rows, settings))
    if log:
        _log_line(f"[results] wrote {len(rows)} rows -> {output_md}", use_tqdm=use_tqdm)


def run(cfg: dict[str, Any]) -> None:
    run_mode = _resolve_run_mode(cfg)
    if str(cfg["tt_backend"]) != "octave_tt_toolbox":
        raise ValueError(
            "Strict paper comparison requires `tt_backend='octave_tt_toolbox'`."
        )
    baselines = _expand_my_cheb_tokens(_resolve_baseline_names(cfg["completion"]["baselines"]), cfg)
    cheb_points_list = sorted(set(int(x) for x in cfg["cheb_points_list"]))
    use_tqdm = bool(cfg["show_tqdm"])
    my_cheb_variants = [name for name in baselines if name.startswith("my_cheb")]

    _log_line(
        f"[run-mode] {run_mode} | cheb_points={cheb_points_list} | baselines={baselines} | "
        f"sampling_policy={cfg['sampling']['policy']}",
        use_tqdm=use_tqdm,
    )
    _log_line(
        f"[resolved] my_cheb_variants={my_cheb_variants}",
        use_tqdm=use_tqdm,
    )

    rows: list[RunRow] = []
    authors_surrogate_cache: dict[tuple[int, int], Any] = {}
    authors_full_cache: dict[int, Any] = {}
    single_budget_resolved_by_cheb_points: dict[str, int] = {}
    sweep_budgets_resolved_by_cheb_points: dict[str, list[int]] = {}
    trace_status_by_cheb_points: dict[str, str] = {}
    trace_unique_by_cheb_points: dict[str, int] = {}
    trace_total_by_cheb_points: dict[str, int] = {}
    exact_grid_errors_by_cheb_points: dict[str, str] = {}

    settings = {
        "geometry_xyz": str(cfg["geometry_xyz"]),
        "water_coordinate_unit": str(cfg["water_coordinate_unit"]),
        "water_as_dim": int(cfg["water_as_dim"]),
        "water_as_sigma2": float(cfg["water_as_sigma2"]),
        "water_as_samples": int(cfg["water_as_samples"]),
        "cheb_interval": [float(cfg["cheb_interval"][0]), float(cfg["cheb_interval"][1])],
        "cheb_points_list": [int(x) for x in cheb_points_list],
        "run_mode": str(run_mode),
        "tt_backend": str(cfg["tt_backend"]),
        "octave_env_name": str(cfg["octave_env_name"]),
        "sampling_policy": str(cfg["sampling"]["policy"]),
        "single_budget_resolved_by_cheb_points": single_budget_resolved_by_cheb_points,
        "sweep_budgets_resolved_by_cheb_points": sweep_budgets_resolved_by_cheb_points,
        "baselines": [str(x) for x in baselines],
        "resolved_baselines": [str(x) for x in baselines],
        "legacy_baseline_aliases": {
            key: value for key, value in sorted(BASELINE_ALIASES.items()) if key != value
        },
        "default_baselines": list(DEFAULT_BASELINES),
        "my_cheb_ranks": [int(x) for x in cfg["my_cheb"]["ranks"]],
        "my_cheb_lambda_all_list": [float(x) for x in cfg["my_cheb"]["lambda_all_list"]],
        "trace_source": str(cfg["sampling"]["trace_source"]),
        "trace_status_by_cheb_points": trace_status_by_cheb_points,
        "trace_unique_queries_by_cheb_points": trace_unique_by_cheb_points,
        "trace_total_queries_by_cheb_points": trace_total_by_cheb_points,
        "exact_grid_errors_by_cheb_points": exact_grid_errors_by_cheb_points,
        "input_contract": (
            "For each (n, budget), fixed-mask baselines receive identical observed_tensor "
            "(missing entries zeroed, observed entries in relative meV from observed minimum) and identical boolean mask."
        ),
        "metric_contract": (
            "Completion rows use faithful.evaluate_completed_tensor_on_test plus exact-grid "
            "evaluate_completed_tensor_on_grid/evaluate_completion_metrics_dual_style when enabled."
        ),
    }
    _write_outputs(
        cfg=cfg,
        rows=rows,
        settings=settings,
        state="running",
        last_event="initialized",
        use_tqdm=use_tqdm,
        log=True,
    )

    cheb_iter = cheb_points_list
    if use_tqdm:
        cheb_iter = tqdm(cheb_iter, desc="Chebyshev grids", unit="grid")
    for n_points in cheb_iter:
        cfg_local = json.loads(json.dumps(cfg))
        cfg_local["n_points"] = int(n_points)
        args_ns = _faithful_args_from_cfg(cfg_local)
        total_tensor_points = int(cfg_local["n_points"]) ** int(cfg_local["water_as_dim"])
        single_budget = _resolve_single_budget(cfg_local)
        sweep_budgets = _resolve_sweep_budgets(cfg_local)
        max_budget = single_budget if run_mode == "single_budget" else (max(sweep_budgets) if sweep_budgets else single_budget)
        single_budget_resolved_by_cheb_points[str(n_points)] = int(single_budget)
        sweep_budgets_resolved_by_cheb_points[str(n_points)] = [int(x) for x in sweep_budgets]

        _log_line(
            f"\n[cheb-points] START n={n_points} | total={total_tensor_points} | "
            f"single_budget={single_budget} | sweep_budgets={sweep_budgets}",
            use_tqdm=use_tqdm,
        )

        shared_trace: faithful.SampleTrace | None = None
        if run_mode in {"single_budget", "budget_sweep", "collect_only"}:
            t0_trace = time.time()
            try:
                shared_trace = _collect_shared_trace(cfg_local, args_ns, max_budget=max_budget, use_tqdm=use_tqdm)
                trace_status_by_cheb_points[str(n_points)] = str(shared_trace.status)
                trace_unique_by_cheb_points[str(n_points)] = int(shared_trace.unique_queries)
                trace_total_by_cheb_points[str(n_points)] = int(shared_trace.total_queries)
            except Exception as exc:
                trace_status_by_cheb_points[str(n_points)] = "fail"
                trace_unique_by_cheb_points[str(n_points)] = 0
                trace_total_by_cheb_points[str(n_points)] = 0
                rows.append(
                    _row_from_error(
                        mode=run_mode,
                        family="sampling",
                        cheb_points=int(n_points),
                        budget=int(max_budget),
                        total_queries=0,
                        baseline="authors_sampling",
                        observed_points=0,
                        elapsed_sec=float(time.time() - t0_trace),
                        error=exc,
                        metric_profile="sampling",
                    )
                )
                _write_outputs(
                    cfg=cfg,
                    rows=rows,
                    settings=settings,
                    state="running",
                    last_event=f"sampling n={n_points} status=fail",
                    use_tqdm=use_tqdm,
                    error=exc,
                )
                _log_line(f"[sampling] FAIL n={n_points} error={exc}", use_tqdm=use_tqdm)
                continue

        if run_mode == "collect_only":
            _log_line(f"[collect-only] finished after collecting the authors trajectory for n={n_points}.", use_tqdm=use_tqdm)
            _write_outputs(
                cfg=cfg,
                rows=rows,
                settings=settings,
                state="running",
                last_event=f"collect_only n={n_points}",
                use_tqdm=use_tqdm,
            )
            continue

        if shared_trace is None:
            raise RuntimeError("Expected a shared trace for completion experiments")
        t0_eval = time.time()
        try:
            eval_payload = _make_eval_payload(cfg_local, args_ns, shared_trace)
            if eval_payload.get("exact_grid_error"):
                exact_grid_errors_by_cheb_points[str(n_points)] = str(eval_payload["exact_grid_error"])
                _log_line(
                    f"[evaluation] exact-grid metrics disabled for n={n_points}: {eval_payload['exact_grid_error']}",
                    use_tqdm=use_tqdm,
                )
        except Exception as exc:
            rows.append(
                _row_from_error(
                    mode=run_mode,
                    family="evaluation_setup",
                    cheb_points=int(n_points),
                    budget=int(max_budget),
                    total_queries=int(shared_trace.total_queries),
                    baseline="evaluation_setup",
                    observed_points=int(shared_trace.unique_queries),
                    elapsed_sec=float(time.time() - t0_eval),
                    error=exc,
                    metric_profile="shared_prefix_completion_v1",
                )
            )
            _write_outputs(
                cfg=cfg,
                rows=rows,
                settings=settings,
                state="running",
                last_event=f"evaluation_setup n={n_points} status=fail",
                use_tqdm=use_tqdm,
                error=exc,
            )
            _log_line(f"[evaluation] FAIL n={n_points} error={exc}", use_tqdm=use_tqdm)
            continue
        selected_fixed_mask_baselines = [name for name in baselines if name in CANONICAL_FIXED_MASK_BASELINES]
        fixed_mask_models = _build_fixed_mask_baselines(
            cfg_local,
            tuple(shared_trace.shape),
            selected_baselines=selected_fixed_mask_baselines,
        )
        t0_authors_full = time.time()
        try:
            full_result = _ensure_authors_full_result(
                cache=authors_full_cache,
                cfg=cfg_local,
                args_ns=args_ns,
                use_tqdm=use_tqdm,
            )
            rows.append(
                _row_from_authors_full(
                    mode=run_mode,
                    cheb_points=int(n_points),
                    result=full_result,
                    eval_payload=eval_payload,
                    elapsed_sec=float(authors_full_cache[int(n_points)]["elapsed_sec"]),
                )
            )
            _write_outputs(
                cfg=cfg,
                rows=rows,
                settings=settings,
                state="running",
                last_event=f"authors_full n={n_points}",
                use_tqdm=use_tqdm,
            )
        except Exception as exc:
            rows.append(
                _row_from_error(
                    mode=run_mode,
                    family="authors_full",
                    cheb_points=int(n_points),
                    budget=total_tensor_points,
                    total_queries=0,
                    baseline="authors_full",
                    observed_points=0,
                    elapsed_sec=float(time.time() - t0_authors_full),
                    error=exc,
                    metric_profile="authors_offgrid_v1",
                )
            )
            _write_outputs(
                cfg=cfg,
                rows=rows,
                settings=settings,
                state="running",
                last_event=f"authors_full n={n_points} status=fail",
                use_tqdm=use_tqdm,
                error=exc,
            )
            _log_line(f"[authors_full] FAIL n={n_points} error={exc}", use_tqdm=use_tqdm)

        if run_mode == "single_budget":
            prefix = _prefix_trace(shared_trace, single_budget)
            prefix_total_queries = _total_queries_for_unique_prefix(shared_trace, int(prefix.unique_queries))
            _log_line(
                f"\n[single-budget] n={n_points} authors points used={prefix.unique_queries}/{total_tensor_points} "
                f"(total_queries={prefix_total_queries})",
                use_tqdm=use_tqdm,
            )
            baseline_iter = baselines
            if use_tqdm:
                baseline_iter = tqdm(
                    baseline_iter,
                    desc=f"Baselines @ n={n_points}, total≈{prefix_total_queries}",
                    unit="model",
                    leave=False,
                )
            for i, baseline_name in enumerate(baseline_iter, start=1):
                _log_line(
                    f"[single-budget n={n_points} {i}/{len(baselines)}] START {baseline_name}",
                    use_tqdm=use_tqdm,
                )
                t0 = time.time()
                try:
                    row = _run_one_baseline(
                        baseline_name=baseline_name,
                        mode="single_budget",
                        cheb_points=int(n_points),
                        budget=int(prefix.unique_queries),
                        total_queries=int(prefix_total_queries),
                        prefix=prefix,
                        shared_trace=shared_trace,
                        cfg=cfg_local,
                        args_ns=args_ns,
                        eval_payload=eval_payload,
                        fixed_mask_models=fixed_mask_models,
                        authors_surrogate_cache=authors_surrogate_cache,
                        use_tqdm=use_tqdm,
                    )
                    rows.append(row)
                    _write_outputs(
                        cfg=cfg,
                        rows=rows,
                        settings=settings,
                        state="running",
                        last_event=f"single_budget n={n_points} baseline={baseline_name} status={row.status}",
                        use_tqdm=use_tqdm,
                    )
                    _log_line(
                        f"[single-budget n={n_points} {i}/{len(baselines)}] DONE {baseline_name} "
                        f"status={row.status} hidden_rmse={row.rmse_mev} offgrid_rmse={row.offgrid_rmse_mev}",
                        use_tqdm=use_tqdm,
                    )
                except Exception as exc:
                    row = _row_from_error(
                        mode="single_budget",
                        family="authors_budgeted" if baseline_name == "authors_budgeted" else "completion",
                        cheb_points=int(n_points),
                        budget=int(prefix.unique_queries),
                        total_queries=int(prefix_total_queries),
                        baseline=baseline_name,
                        observed_points=int(prefix.unique_queries),
                        elapsed_sec=float(time.time() - t0),
                        error=exc,
                        input_digest=_input_digest_from_trace(prefix),
                        metric_profile="authors_budgeted_offgrid_v1" if baseline_name == "authors_budgeted" else "shared_prefix_completion_v1",
                    )
                    row.rank_label = _rank_label_for_baseline(baseline_name, cfg_local)
                    rows.append(row)
                    _write_outputs(
                        cfg=cfg,
                        rows=rows,
                        settings=settings,
                        state="running",
                        last_event=f"single_budget n={n_points} baseline={baseline_name} status=fail",
                        use_tqdm=use_tqdm,
                        error=exc,
                    )
                    _log_line(
                        f"[single-budget n={n_points} {i}/{len(baselines)}] FAIL {baseline_name} error={exc}",
                        use_tqdm=use_tqdm,
                    )

        elif run_mode == "budget_sweep":
            budget_iter = sweep_budgets
            if use_tqdm:
                budget_iter = tqdm(budget_iter, desc=f"Budgets @ n={n_points}", unit="budget")
            for budget in budget_iter:
                prefix = _prefix_trace(shared_trace, budget)
                prefix_total_queries = _total_queries_for_unique_prefix(shared_trace, int(prefix.unique_queries))
                _log_line(
                    f"\n[budget-sweep] n={n_points} unique_budget={budget} authors points used={prefix.unique_queries}/{total_tensor_points} "
                    f"(total_queries={prefix_total_queries})",
                    use_tqdm=use_tqdm,
                )
                for i, baseline_name in enumerate(baselines, start=1):
                    _log_line(
                        f"[budget-sweep n={n_points} budget={budget} | {i}/{len(baselines)}] START {baseline_name}",
                        use_tqdm=use_tqdm,
                    )
                    t0 = time.time()
                    try:
                        row = _run_one_baseline(
                            baseline_name=baseline_name,
                            mode="budget_sweep",
                            cheb_points=int(n_points),
                            budget=int(prefix.unique_queries),
                            total_queries=int(prefix_total_queries),
                            prefix=prefix,
                            shared_trace=shared_trace,
                            cfg=cfg_local,
                            args_ns=args_ns,
                            eval_payload=eval_payload,
                            fixed_mask_models=fixed_mask_models,
                            authors_surrogate_cache=authors_surrogate_cache,
                            use_tqdm=use_tqdm,
                        )
                        rows.append(row)
                        _write_outputs(
                            cfg=cfg,
                            rows=rows,
                            settings=settings,
                            state="running",
                            last_event=f"budget_sweep n={n_points} budget={budget} baseline={baseline_name} status={row.status}",
                            use_tqdm=use_tqdm,
                        )
                        _log_line(
                            f"[budget-sweep n={n_points} budget={budget} | {i}/{len(baselines)}] DONE {baseline_name} "
                            f"status={row.status} hidden_rmse={row.rmse_mev} offgrid_rmse={row.offgrid_rmse_mev}",
                            use_tqdm=use_tqdm,
                        )
                    except Exception as exc:
                        row = _row_from_error(
                            mode="budget_sweep",
                            family="authors_budgeted" if baseline_name == "authors_budgeted" else "completion",
                            cheb_points=int(n_points),
                            budget=int(prefix.unique_queries),
                            total_queries=int(prefix_total_queries),
                            baseline=baseline_name,
                            observed_points=int(prefix.unique_queries),
                            elapsed_sec=float(time.time() - t0),
                            error=exc,
                            input_digest=_input_digest_from_trace(prefix),
                            metric_profile="authors_budgeted_offgrid_v1" if baseline_name == "authors_budgeted" else "shared_prefix_completion_v1",
                        )
                        row.rank_label = _rank_label_for_baseline(baseline_name, cfg_local)
                        rows.append(row)
                        _write_outputs(
                            cfg=cfg,
                            rows=rows,
                            settings=settings,
                            state="running",
                            last_event=f"budget_sweep n={n_points} budget={budget} baseline={baseline_name} status=fail",
                            use_tqdm=use_tqdm,
                            error=exc,
                        )
                        _log_line(
                            f"[budget-sweep n={n_points} budget={budget} | {i}/{len(baselines)}] FAIL {baseline_name} error={exc}",
                            use_tqdm=use_tqdm,
                        )

    _write_outputs(
        cfg=cfg,
        rows=rows,
        settings=settings,
        state="complete",
        last_event="complete",
        use_tqdm=use_tqdm,
        log=True,
    )

    _log_line(f"\nWrote JSON: {Path(cfg['output_json'])}", use_tqdm=use_tqdm)
    _log_line(f"Wrote Markdown: {Path(cfg['output_md'])}", use_tqdm=use_tqdm)


if __name__ == "__main__":
    run(_resolve_cfg())
