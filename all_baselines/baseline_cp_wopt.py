from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics
from all_baselines.octave_backend import matlab_cell_to_arrays, output_tail, run_mat_bridge, PROJECT_ROOT


@dataclass
class CPWOPT(TensorCompletionBaseline):
    """CP-WOPT from the official Tensor Toolbox for MATLAB."""

    rank: int = 5
    max_iter: int = 200
    tol: float = 1e-6
    random_state: int = 0
    verbose: bool = False
    octave_env_name: str | None = "octave"
    tensor_toolbox_root: str | None = None

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        observed = mask.astype(bool)
        missing = ~observed
        y = observed_tensor.astype(float).copy()
        y[missing] = 0.0

        toolbox_root = Path(self.tensor_toolbox_root) if self.tensor_toolbox_root else (
            PROJECT_ROOT / "external" / "tensor_toolbox-v3.3"
        )
        if not (toolbox_root / "cp_wopt.m").exists():
            raise RuntimeError(f"Tensor Toolbox cp_wopt.m not found at {toolbox_root}")

        payload, octave_output = run_mat_bridge(
            runner_name="run_tensor_toolbox_cp_wopt",
            input_payload={
                "observed_tensor": y,
                "mask": observed.astype(float),
            },
            config_payload={
                "shape": [int(x) for x in y.shape],
                "rank": int(self.rank),
                "max_iter": int(self.max_iter),
                "tol": float(self.tol),
                "random_state": int(self.random_state),
                "verbose": bool(self.verbose),
                "tensor_toolbox_root": str(toolbox_root),
            },
            octave_env_name=self.octave_env_name,
        )

        pred = np.asarray(payload["completed"], dtype=float).reshape(y.shape)
        pred[observed] = y[observed]
        factors = matlab_cell_to_arrays(payload.get("factors"))
        weights = np.asarray(payload.get("weights", []), dtype=float).reshape(-1)
        effective_rank = int(np.asarray(payload.get("effective_rank", [[len(weights)]]), dtype=float).reshape(-1)[0])

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
                "backend": "Tensor Toolbox cp_wopt (CP-WOPT) via Octave",
                "source_repo": "https://gitlab.com/tensors/tensor_toolbox",
                "method_reference": "Acar-Dunlavy-Kolda-Morup CP-WOPT",
                "tensor_toolbox_root": str(toolbox_root),
                "octave_env_name": self.octave_env_name,
                "requested_rank": int(self.rank),
                "effective_rank": effective_rank,
                "weights": weights.tolist(),
                "final_objective": _scalar_or_none(payload.get("final_objective")),
                "iterations": _scalar_or_none(payload.get("iterations")),
                "optimizer_message": _string_or_empty(payload.get("optimizer_message")),
                "missing_entries_zeroed": True,
                "mask_argument": "W/P",
                "octave_output_tail": output_tail(octave_output),
            },
        )


def _scalar_or_none(value) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def _string_or_empty(value) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(x) for x in arr.reshape(-1)).strip()
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)
