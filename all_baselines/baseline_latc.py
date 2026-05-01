from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics


@dataclass
class LATC(TensorCompletionBaseline):
    """Low-Rank Autoregressive Tensor Completion (practical matrix-unfolded variant)."""

    rank: int = 10
    ar_order: int = 3
    temporal_weight: float = 0.35
    ridge: float = 1e-3
    max_iter: int = 120
    tol: float = 1e-5
    verbose: bool = False

    def _fit_low_rank(self, matrix: np.ndarray) -> np.ndarray:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        r = max(1, min(self.rank, s.size))
        return (u[:, :r] * s[:r]) @ vt[:r, :]

    def _fit_ar_coeffs(self, series: np.ndarray, order: int) -> np.ndarray:
        if order <= 0 or series.size <= order:
            return np.zeros(0, dtype=float)
        target = series[order:]
        design = np.column_stack(
            [series[order - k - 1: series.size - k - 1] for k in range(order)]
        )
        lhs = design.T @ design + self.ridge * np.eye(order)
        rhs = design.T @ target
        return np.linalg.solve(lhs, rhs)

    def _apply_ar(self, matrix: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        if coeffs.size == 0:
            return matrix
        out = matrix.copy()
        order = coeffs.size
        for t in range(order, matrix.shape[1]):
            pred = np.zeros(matrix.shape[0], dtype=float)
            for k, c in enumerate(coeffs):
                pred += c * matrix[:, t - k - 1]
            out[:, t] = (1.0 - self.temporal_weight) * matrix[:, t] + self.temporal_weight * pred
        return out

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        shape = observed_tensor.shape
        if observed_tensor.ndim < 2:
            raise ValueError("LATC baseline expects at least 2D tensor with a temporal mode")

        space = int(np.prod(shape[:-1]))
        time_dim = shape[-1]

        observed_matrix = observed_tensor.reshape(space, time_dim).astype(float)
        observed_mask = mask.reshape(space, time_dim).astype(bool)
        missing_mask = ~observed_mask

        X = observed_matrix.copy()
        fallback = float(np.mean(observed_matrix[observed_mask])) if np.any(observed_mask) else 0.0
        col_means = np.full(time_dim, fallback, dtype=float)
        for t in range(time_dim):
            idx = observed_mask[:, t]
            if np.any(idx):
                col_means[t] = float(np.mean(observed_matrix[idx, t]))
        for t in range(time_dim):
            X[missing_mask[:, t], t] = col_means[t]

        history: list[float] = []
        coeffs = np.zeros(0, dtype=float)
        ar_order = min(max(0, int(self.ar_order)), max(0, time_dim - 1))

        for it in range(self.max_iter):
            X_prev = X.copy()
            low_rank = self._fit_low_rank(X)
            mean_series = low_rank.mean(axis=0)
            coeffs = self._fit_ar_coeffs(mean_series, ar_order)
            X = self._apply_ar(low_rank, coeffs)
            X[observed_mask] = observed_matrix[observed_mask]

            rel_change = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
            history.append(float(rel_change))

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[LATC] iter={it:04d} rel_change={rel_change:.3e}")

            if rel_change < self.tol:
                break

        completed = X.reshape(shape)
        observed = mask.astype(bool)
        missing = ~observed

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, completed, observed)
            test_metrics = all_metrics(full_tensor, completed, missing)

        return CompletionResult(
            tensor=completed,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"history": history, "ar_coeffs": coeffs.tolist()},
        )

