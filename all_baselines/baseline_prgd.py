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
class PRGD(TensorCompletionBaseline):
    """Preconditioned Riemannian-style GD for Tucker completion (practical variant)."""

    ranks: tuple[int, ...] | None = None
    step_size: float = 0.1
    max_iter: int = 200
    tol: float = 1e-6
    reg: float = 1e-5
    precond_reg: float = 1e-4
    random_state: int = 0
    verbose: bool = False

    def _resolve_ranks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.ranks is None:
            return tuple(max(1, min(8, dim)) for dim in shape)
        if len(self.ranks) != len(shape):
            raise ValueError("ranks must have one rank per mode")
        return tuple(max(1, min(int(r), shape[m])) for m, r in enumerate(self.ranks))

    def _init_tucker(
        self,
        shape: tuple[int, ...],
        ranks: tuple[int, ...],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        rng = np.random.default_rng(self.random_state)
        factors: list[np.ndarray] = []
        for dim, rank in zip(shape, ranks):
            q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
            factors.append(q[:, :rank])
        core = 0.1 * rng.standard_normal(ranks)
        return core, factors

    def _reconstruct(self, core: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
        out = core
        for mode, factor in enumerate(factors):
            out = mode_dot(out, factor, mode)
        return out

    def _project_to_core(self, tensor: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
        out = tensor
        for mode, factor in enumerate(factors):
            out = mode_dot(out, factor.T, mode)
        return out

    def _mode_context(self, core: np.ndarray, factors: list[np.ndarray], mode: int) -> np.ndarray:
        ctx = core
        for k, factor in enumerate(factors):
            if k != mode:
                ctx = mode_dot(ctx, factor, k)
        return unfold(ctx, mode)

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

        core, factors = self._init_tucker(shape, ranks)
        history: list[float] = []
        m_obs = int(np.sum(observed))

        for it in range(self.max_iter):
            pred = self._reconstruct(core, factors)
            residual = np.zeros_like(pred, dtype=float)
            residual[observed] = pred[observed] - observed_tensor[observed]

            loss = 0.5 * float(np.dot(residual[observed], residual[observed])) / max(m_obs, 1)
            history.append(loss)

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[PRGD] iter={it:04d} loss={loss:.6e}")

            if it > 0:
                rel_obj = abs(history[-2] - history[-1]) / (abs(history[-2]) + 1e-12)
                if rel_obj < self.tol:
                    break

            grad_core = self._project_to_core(residual, factors) + self.reg * core
            core = core - self.step_size * grad_core

            for mode in range(pred.ndim):
                Hm = self._mode_context(core, factors, mode)
                Em = unfold(residual, mode)
                grad_u = Em @ Hm.T + self.reg * factors[mode]
                precond = np.linalg.pinv(Hm @ Hm.T + self.precond_reg * np.eye(Hm.shape[0]))
                candidate = factors[mode] - self.step_size * (grad_u @ precond)
                q, _ = np.linalg.qr(candidate)
                factors[mode] = q[:, :ranks[mode]]

        completed = self._reconstruct(core, factors)
        completed[observed] = observed_tensor[observed]

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, completed, observed)
            test_metrics = all_metrics(full_tensor, completed, missing)

        return CompletionResult(
            tensor=completed,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"objective_history": history, "ranks": list(ranks)},
        )

