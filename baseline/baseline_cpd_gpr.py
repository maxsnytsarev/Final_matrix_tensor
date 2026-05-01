from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, cp_to_tensor


@dataclass
class CPDGPRBaseline(TensorCompletionBaseline):
    rank: int = 5
    max_als_iter: int = 100
    gp_length_scale: float = 0.2
    gp_noise: float = 1e-6
    random_state: int = 0
    verbose: bool = False

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        ell2 = self.gp_length_scale ** 2
        return np.exp(-0.5 * d2 / max(ell2, 1e-12))

    def _gp_predict_dense(self, observed_tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        obs_idx = np.argwhere(mask)
        y = observed_tensor[mask].astype(float)
        shape = observed_tensor.shape
        grid_axes = [np.linspace(-1.0, 1.0, s) for s in shape]
        X_train = np.column_stack([grid_axes[m][obs_idx[:, m]] for m in range(observed_tensor.ndim)])
        K = self._rbf_kernel(X_train, X_train)
        K.flat[:: K.shape[0] + 1] += self.gp_noise
        alpha = np.linalg.solve(K, y)
        grid_points = np.array(list(itertools.product(*grid_axes)), dtype=float)
        K_star = self._rbf_kernel(grid_points, X_train)
        pred = K_star @ alpha
        return pred.reshape(shape)

    def _cp_als(self, tensor: np.ndarray) -> list[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        shape = tensor.shape
        n_modes = tensor.ndim
        factors = [rng.standard_normal((shape[m], self.rank)) * 0.1 for m in range(n_modes)]
        for _ in range(self.max_als_iter):
            for mode in range(n_modes):
                others = [factors[m] for m in range(n_modes) if m != mode]
                kr = others[-1]
                for mat in reversed(others[:-1]):
                    kr = np.einsum('ir,jr->ijr', mat, kr).reshape(-1, self.rank)
                Xn = np.reshape(np.moveaxis(tensor, mode, 0), (shape[mode], -1))
                gram = np.ones((self.rank, self.rank))
                for m in range(n_modes):
                    if m != mode:
                        gram *= factors[m].T @ factors[m]
                factors[mode] = (Xn @ kr) @ np.linalg.pinv(gram + 1e-8 * np.eye(self.rank))
        return factors

    def fit_transform(self, observed_tensor: np.ndarray, mask: np.ndarray, full_tensor: np.ndarray | None = None) -> CompletionResult:
        dense_gp = self._gp_predict_dense(observed_tensor, mask.astype(bool))
        factors = self._cp_als(dense_gp)
        pred = cp_to_tensor(factors)
        train_metrics = test_metrics = None
        if full_tensor is not None:
            obs = mask.astype(bool)
            train_metrics = all_metrics(full_tensor, pred, obs)
            test_metrics = all_metrics(full_tensor, pred, ~obs)
        return CompletionResult(
            tensor=pred,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={'note': 'Practical CPD-GPR-style baseline, not exact paper code'},
        )
