from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, fold, svt, unfold


@dataclass
class HaLRTC(TensorCompletionBaseline):
    alpha: tuple[float, ...] | None = None
    rho: float = 1e-4
    rho_scale: float = 1.05
    rho_max: float = 1e5
    max_iter: int = 200
    tol: float = 1e-5
    verbose: bool = False

    def fit_transform(self, observed_tensor: np.ndarray, mask: np.ndarray, full_tensor: np.ndarray | None = None) -> CompletionResult:
        shape = observed_tensor.shape
        n_modes = observed_tensor.ndim
        alpha = np.asarray(self.alpha if self.alpha is not None else np.ones(n_modes) / n_modes, dtype=float)
        if alpha.shape != (n_modes,):
            raise ValueError('alpha must have one weight per mode')
        alpha = alpha / alpha.sum()

        X = observed_tensor.copy().astype(float)
        M = [X.copy() for _ in range(n_modes)]
        Y = [np.zeros_like(X, dtype=float) for _ in range(n_modes)]
        rho = float(self.rho)
        history: list[float] = []

        observed = mask.astype(bool)
        missing = ~observed
        X[observed] = observed_tensor[observed]

        for it in range(self.max_iter):
            X_prev = X.copy()
            for mode in range(n_modes):
                unfolded = unfold(X + Y[mode] / rho, mode)
                M[mode] = fold(svt(unfolded, alpha[mode] / rho), shape, mode)

            avg = sum(Mi - Yi / rho for Mi, Yi in zip(M, Y)) / n_modes
            X[missing] = avg[missing]
            X[observed] = observed_tensor[observed]

            for mode in range(n_modes):
                Y[mode] = Y[mode] - rho * (M[mode] - X)

            rel_change = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
            history.append(float(rel_change))
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f'[HaLRTC] iter={it:04d} rel_change={rel_change:.3e} rho={rho:.3e}')
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
            info={'history': history, 'final_rho': rho},
        )
