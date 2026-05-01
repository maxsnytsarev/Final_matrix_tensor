from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics
from external.xinychen_transdim.halrtc import HaLRTC_imputer

@dataclass
class HaLRTC(TensorCompletionBaseline):
    """High-accuracy Low-Rank Tensor Completion via xinychen/transdim."""

    alpha: tuple[float, ...] | None = None
    rho: float = 1e-4
    rho_scale: float = 1.05
    rho_max: float = 1e5
    max_iter: int = 200
    tol: float = 1e-5
    verbose: bool = False

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
        X[missing] = 0.0
        X, halrtc_info = HaLRTC_imputer(
            dense_tensor=None if full_tensor is None else np.asarray(full_tensor, dtype=float),
            sparse_tensor=X,
            alpha=alpha,
            rho=float(self.rho),
            epsilon=float(self.tol),
            maxiter=int(self.max_iter),
            observed_mask=observed,
            rho_scale=float(self.rho_scale),
            rho_max=float(self.rho_max),
            verbose=bool(self.verbose),
        )

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, X, observed)
            test_metrics = all_metrics(full_tensor, X, missing)

        return CompletionResult(
            tensor=X,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "backend": "xinychen/transdim HaLRTC_imputer",
                **halrtc_info,
            },
        )
