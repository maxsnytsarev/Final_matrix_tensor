from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

from common import CompletionResult, TensorCompletionBaseline, all_metrics


@dataclass
class TuckerCompletion(TensorCompletionBaseline):
    """Tucker completion backed by TensorLy's masked Tucker implementation."""

    ranks: tuple[int, ...] | None = None
    max_iter: int = 120
    tol: float = 1e-5
    inner_sweeps: int = 1
    verbose: bool = False
    random_state: int = 0

    def _resolve_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.ranks is None:
            return tuple(10 for _ in shape)
        if len(self.ranks) != len(shape):
            raise ValueError("ranks must have one rank per mode")
        return tuple(max(1, int(r)) for r in self.ranks)

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        shape = observed_tensor.shape
        ranks = self._resolve_ranks(shape)
        observed = mask.astype(bool)
        missing = ~observed

        Z = observed_tensor.astype(float).copy()
        Z[missing] = 0.0

        tucker_tensor, errors = tucker(
            Z,
            rank=ranks,
            n_iter_max=int(self.max_iter),
            init="random",
            return_errors=True,
            tol=float(self.tol),
            random_state=int(self.random_state),
            mask=observed.astype(float),
            verbose=bool(self.verbose),
        )
        projected = np.asarray(tucker_to_tensor(tucker_tensor), dtype=float)
        completed = projected.copy()
        completed[observed] = observed_tensor[observed]

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, completed, observed)
            test_metrics = all_metrics(full_tensor, completed, missing)

        info = {
            "backend": "tensorly.decomposition.tucker(mask=...)",
            "source_repo": "https://github.com/tensorly/tensorly",
            "requested_ranks": list(ranks),
            "effective_ranks": [int(factor.shape[1]) for factor in tucker_tensor.factors],
            "reconstruction_errors": [float(x) for x in errors],
        }
        info["core_shape"] = list(np.asarray(tucker_tensor.core).shape)

        return CompletionResult(
            tensor=completed,
            factors=[np.asarray(factor, dtype=float) for factor in tucker_tensor.factors],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info,
        )
