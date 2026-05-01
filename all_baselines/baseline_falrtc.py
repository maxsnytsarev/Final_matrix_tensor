from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, fold, svt, unfold
from tqdm.auto import tqdm

@dataclass
class FaLRTC(TensorCompletionBaseline):
    """Fast Low-Rank Tensor Completion (Liu et al., TPAMI 2013, practical variant)."""

    alpha: tuple[float, ...] | None = None
    lam: float = 1.0
    step_size: float = 1.0
    max_iter: int = 300
    tol: float = 1e-5
    verbose: bool = False

    def _prox_average(self, tensor: np.ndarray, alpha: np.ndarray, tau: float) -> np.ndarray:
        out = np.zeros_like(tensor, dtype=float)
        shape = tensor.shape
        for mode in range(tensor.ndim):
            out += fold(svt(unfold(tensor, mode), tau * alpha[mode]), shape, mode)
        return out / tensor.ndim

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
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
        Y = X.copy()

        t = 1.0
        history: list[float] = []

        for it in tqdm(range(self.max_iter)):
            grad = np.zeros_like(Y, dtype=float)
            grad[observed] = Y[observed] - observed_tensor[observed]
            Z = Y - self.step_size * grad

            X_new = self._prox_average(Z, alpha, self.step_size * self.lam)
            X_new[observed] = observed_tensor[observed]

            rel_change = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + 1e-12)
            history.append(float(rel_change))

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[FaLRTC] iter={it:04d} rel_change={rel_change:.3e}")

            if rel_change < self.tol:
                X = X_new
                break

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Y = X_new + ((t - 1.0) / t_new) * (X_new - X)
            X = X_new
            t = t_new

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, X, observed)
            test_metrics = all_metrics(full_tensor, X, missing)

        return CompletionResult(
            tensor=X,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"history": history},
        )

