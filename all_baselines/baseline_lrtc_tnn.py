from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, fold, unfold


def truncated_svt(matrix: np.ndarray, tau: float, trunc_rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    r = max(0, min(int(trunc_rank), s.size))
    s_new = s.copy()
    if r < s.size:
        s_new[r:] = np.maximum(s[r:] - tau, 0.0)
    return (u * s_new) @ vt


@dataclass
class LRTCTNN(TensorCompletionBaseline):
    """Low-rank tensor completion with truncated tensor nuclear norm."""

    alpha: tuple[float, ...] | None = None
    trunc_rank: int = 5
    rho: float = 1e-3
    rho_scale: float = 1.05
    rho_max: float = 1e5
    max_iter: int = 300
    tol: float = 1e-5
    verbose: bool = False

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        shape = observed_tensor.shape
        n_modes = observed_tensor.ndim
        alpha = np.asarray(
            self.alpha if self.alpha is not None else np.ones(n_modes) / n_modes,
            dtype=float,
        )
        if alpha.shape != (n_modes,):
            raise ValueError("alpha must have one weight per mode")
        alpha = alpha / max(alpha.sum(), 1e-12)

        observed = mask.astype(bool)
        missing = ~observed
        X = observed_tensor.astype(float).copy()
        if np.any(missing) and np.any(observed):
            X[missing] = float(np.mean(observed_tensor[observed]))

        M = [X.copy() for _ in range(n_modes)]
        Y = [np.zeros_like(X, dtype=float) for _ in range(n_modes)]
        rho = float(self.rho)
        history: list[float] = []

        for it in range(self.max_iter):
            X_prev = X.copy()

            for mode in range(n_modes):
                unfolded = unfold(X + Y[mode] / rho, mode)
                prox = truncated_svt(unfolded, alpha[mode] / rho, self.trunc_rank)
                M[mode] = fold(prox, shape, mode)

            avg = sum(Mi - Yi / rho for Mi, Yi in zip(M, Y)) / n_modes
            X[missing] = avg[missing]
            X[observed] = observed_tensor[observed]

            for mode in range(n_modes):
                Y[mode] = Y[mode] - rho * (M[mode] - X)

            rel_change = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
            history.append(float(rel_change))

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[LRTC-TNN] iter={it:04d} rel_change={rel_change:.3e} rho={rho:.3e}")

            if rel_change < self.tol:
                break
            rho = min(rho * self.rho_scale, self.rho_max)

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, X, observed)
            test_metrics = all_metrics(full_tensor, X, missing)

        return CompletionResult(
            tensor=X,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"history": history, "final_rho": rho},
        )

