from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, cp_to_tensor


@dataclass
class ScaledGD(TensorCompletionBaseline):
    """Scaled gradient descent baseline for CP-parameterized tensor completion."""

    rank: int = 5
    step_size: float = 0.1
    max_iter: int = 300
    tol: float = 1e-6
    reg: float = 1e-5
    damping: float = 1e-6
    random_state: int = 0
    verbose: bool = False

    def _init_factors(self, shape: tuple[int, ...]) -> list[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        return [0.1 * rng.standard_normal((dim, self.rank)) for dim in shape]

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        observed = mask.astype(bool)
        if not np.any(observed):
            raise ValueError("mask has no observed entries")

        shape = observed_tensor.shape
        n_modes = observed_tensor.ndim
        obs_idx = np.argwhere(observed)
        y_obs = observed_tensor[observed].astype(float)
        m_obs = obs_idx.shape[0]
        factors = self._init_factors(shape)

        history: list[float] = []

        for it in range(self.max_iter):
            comp = np.ones((m_obs, self.rank), dtype=float)
            for mode in range(n_modes):
                comp *= factors[mode][obs_idx[:, mode], :]
            pred_obs = np.sum(comp, axis=1)
            residual = pred_obs - y_obs
            loss = 0.5 * float(np.dot(residual, residual)) / max(m_obs, 1)
            history.append(loss)

            if self.verbose and (it % 20 == 0 or it == self.max_iter - 1):
                print(f"[ScaledGD] iter={it:04d} loss={loss:.6e}")

            if it > 0:
                rel_obj = abs(history[-2] - history[-1]) / (abs(history[-2]) + 1e-12)
                if rel_obj < self.tol:
                    break

            grads: list[np.ndarray] = []
            for mode in range(n_modes):
                contrib = np.ones((m_obs, self.rank), dtype=float)
                for other in range(n_modes):
                    if other != mode:
                        contrib *= factors[other][obs_idx[:, other], :]

                grad = np.zeros_like(factors[mode])
                np.add.at(grad, obs_idx[:, mode], residual[:, None] * contrib)
                grad = grad / max(m_obs, 1) + self.reg * factors[mode]
                grads.append(grad)

            for mode in range(n_modes):
                gram = np.ones((self.rank, self.rank), dtype=float)
                for other in range(n_modes):
                    if other != mode:
                        gram *= factors[other].T @ factors[other]
                precond = np.linalg.pinv(gram + self.damping * np.eye(self.rank))
                factors[mode] = factors[mode] - self.step_size * (grads[mode] @ precond)

        pred = cp_to_tensor(factors)
        pred[observed] = observed_tensor[observed]
        missing = ~observed

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, pred, observed)
            test_metrics = all_metrics(full_tensor, pred, missing)

        return CompletionResult(
            tensor=pred,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"objective_history": history},
        )

