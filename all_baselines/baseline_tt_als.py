from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics
from all_baselines.octave_backend import PROJECT_ROOT, matlab_cell_to_arrays, output_tail, run_mat_bridge


@dataclass
class TTALS(TensorCompletionBaseline):
    """TT completion backed by EPFL TTeMPS RTTC."""

    tt_rank: int | tuple[int, ...] = 8
    max_iter: int = 120
    tol: float = 1e-5
    reg: float = 1e-8
    inner_sweeps: int = 1
    verbose: bool = False
    octave_env_name: str | None = "octave"
    ttemps_root: str | None = None
    random_state: int = 0

    @staticmethod
    def _tt_to_tensor(cores: list[np.ndarray]) -> np.ndarray:
        if not cores:
            return np.array([], dtype=float)
        out = np.asarray(cores[0], dtype=float)
        for core in cores[1:]:
            out = np.tensordot(out, np.asarray(core, dtype=float), axes=([-1], [0]))
        return np.squeeze(out, axis=(0, -1))

    def _expand_tt_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        d = len(shape)
        if d <= 1:
            return (1, 1)

        if isinstance(self.tt_rank, int):
            raw = [1] + [int(self.tt_rank)] * (d - 1) + [1]
        else:
            raw = [int(r) for r in self.tt_rank]
            if len(raw) == d - 1:
                raw = [1] + raw + [1]
            elif len(raw) != d + 1:
                raise ValueError("tt_rank must be an int, ndim - 1 tuple, or ndim + 1 full TT-rank tuple")

        if raw[0] != 1 or raw[-1] != 1:
            raise ValueError("TT rank boundary conditions require rank[0] == rank[-1] == 1")
        if any(r < 1 for r in raw):
            raise ValueError("TT ranks must be positive")
        return tuple(raw)

    def _resolve_tt_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        raw = list(self._expand_tt_ranks(shape))
        d = len(shape)
        for k in range(1, d):
            left = int(np.prod(shape[:k], dtype=np.int64))
            right = int(np.prod(shape[k:], dtype=np.int64))
            raw[k] = min(raw[k], max(1, min(left, right)))
        return tuple(raw)

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        observed = mask.astype(bool)
        missing = ~observed
        if not np.any(observed):
            raise ValueError("TTeMPS RTTC requires at least one observed entry")

        y = observed_tensor.astype(float).copy()
        y[missing] = 0.0
        raw_requested_ranks = self._expand_tt_ranks(y.shape)
        requested_ranks = self._resolve_tt_ranks(y.shape)

        ttemps_root = Path(self.ttemps_root) if self.ttemps_root else PROJECT_ROOT / "external" / "TTeMPS_1.1"
        if not (ttemps_root / "algorithms" / "completion" / "completion_orth.m").exists():
            raise RuntimeError(f"TTeMPS RTTC completion_orth.m not found at {ttemps_root}")

        payload, octave_output = run_mat_bridge(
            runner_name="run_ttemps_rttc_completion",
            input_payload={
                "observed_tensor": y,
                "mask": observed.astype(float),
            },
            config_payload={
                "shape": [int(x) for x in y.shape],
                "tt_rank": [int(x) for x in requested_ranks],
                "max_iter": int(self.max_iter),
                "tol": float(self.tol),
                "random_state": int(self.random_state),
                "verbose": bool(self.verbose),
                "ttemps_root": str(ttemps_root),
            },
            octave_env_name=self.octave_env_name,
        )

        pred = np.asarray(payload["completed"], dtype=float).reshape(y.shape)
        pred[observed] = y[observed]
        factors = matlab_cell_to_arrays(payload.get("cores"))
        effective = np.asarray(payload.get("effective_tt_ranks", []), dtype=float).reshape(-1)
        cost_history = np.asarray(payload.get("cost_history", []), dtype=float).reshape(-1)
        test_history = np.asarray(payload.get("test_history", []), dtype=float).reshape(-1)
        time_history = np.asarray(payload.get("time_history", []), dtype=float).reshape(-1)

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, pred, observed)
            test_metrics = all_metrics(full_tensor, pred, missing)

        return CompletionResult(
            tensor=pred,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "backend": "EPFL TTeMPS completion_orth (RTTC) via Octave",
                "source_repo": "https://www.epfl.ch/labs/anchp/index-html/software/ttemps/",
                "ttemps_root": str(ttemps_root),
                "octave_env_name": self.octave_env_name,
                "requested_tt_ranks_unconstrained": [int(x) for x in raw_requested_ranks],
                "requested_tt_ranks": [int(x) for x in requested_ranks],
                "effective_tt_ranks": [int(x) for x in effective] if effective.size else [],
                "cost_history": [float(x) for x in cost_history],
                "test_history": [float(x) for x in test_history],
                "time_history": [float(x) for x in time_history],
                "converged": _bool_or_false(payload.get("converged")),
                "reg_ignored_by_rttc": float(self.reg),
                "inner_sweeps_ignored_by_rttc": int(self.inner_sweeps),
                "octave_output_tail": output_tail(octave_output),
            },
        )


def _bool_or_false(value) -> bool:
    if value is None:
        return False
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return False
    return bool(arr[0])
