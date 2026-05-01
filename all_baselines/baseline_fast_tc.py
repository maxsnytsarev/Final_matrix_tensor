from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, fold, unfold


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    unfolded = unfold(tensor, mode)
    projected = matrix @ unfolded
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(projected, new_shape, mode)


@dataclass
class FastTensorCompletion(TensorCompletionBaseline):
    """Fast tensor completion via Richardson step + low-rank projection."""

    ranks: tuple[int, ...] | None = None
    step_size: float = 1.0
    max_iter: int = 150
    tol: float = 1e-5
    verbose: bool = False

    def _resolve_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.ranks is None:
            return tuple(max(1, min(8, dim)) for dim in shape)
        if len(self.ranks) != len(shape):
            raise ValueError("ranks must have one rank per mode")
        return tuple(max(1, min(int(r), shape[m])) for m, r in enumerate(self.ranks))

    def _project_tucker(
        self,
        tensor: np.ndarray,
        ranks: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        factors: list[np.ndarray] = []
        for mode, rank in enumerate(ranks):
            u, _, _ = np.linalg.svd(unfold(tensor, mode), full_matrices=False)
            factors.append(u[:, :rank])

        core = tensor.copy()
        for mode, factor in enumerate(factors):
            core = mode_dot(core, factor.T, mode)

        reconstructed = core.copy()
        for mode, factor in enumerate(factors):
            reconstructed = mode_dot(reconstructed, factor, mode)

        return reconstructed, core, factors

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        observed = mask.astype(bool)
        missing = ~observed
        shape = observed_tensor.shape
        ranks = self._resolve_ranks(shape)

        X = observed_tensor.astype(float).copy()
        if np.any(missing) and np.any(observed):
            X[missing] = float(np.mean(observed_tensor[observed]))

        history: list[float] = []
        core = None
        factors: list[np.ndarray] = []

        for it in range(self.max_iter):
            X_prev = X.copy()

            residual = np.zeros_like(X, dtype=float)
            residual[observed] = X[observed] - observed_tensor[observed]
            Z = X - self.step_size * residual

            X, core, factors = self._project_tucker(Z, ranks)
            X[observed] = observed_tensor[observed]

            rel_change = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
            history.append(float(rel_change))

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[FastTC] iter={it:04d} rel_change={rel_change:.3e}")

            if rel_change < self.tol:
                break

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, X, observed)
            test_metrics = all_metrics(full_tensor, X, missing)

        info = {"history": history, "ranks": list(ranks)}
        if core is not None:
            info["core_shape"] = list(core.shape)

        return CompletionResult(
            tensor=X,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info,
        )

