from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def _temporal_linear_interpolation(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill missing entries independently for each spatial location by 1D interpolation."""
    out = values.copy()
    n_space, n_time = out.shape
    t = np.arange(n_time)

    for i in range(n_space):
        idx = observed[i]
        if np.sum(idx) == 0:
            out[i, :] = 0.0
            continue
        if np.sum(idx) == 1:
            out[i, :] = float(out[i, idx][0])
            continue
        out[i, :] = np.interp(t, t[idx], out[i, idx])
    return out


def _truncate_svd(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    r = max(1, min(int(rank), s.size))
    return (u[:, :r] * s[:r]) @ vt[:r, :]


def _seasonal_projection(matrix: np.ndarray, period: int, seasonal_weight: float) -> np.ndarray:
    if period <= 1 or matrix.shape[1] < period:
        return matrix

    n_space, n_time = matrix.shape
    n_cycles = n_time // period
    usable = n_cycles * period
    if usable <= 0:
        return matrix

    core = matrix[:, :usable].reshape(n_space, n_cycles, period)
    seasonal_mean = np.mean(core, axis=1, keepdims=True)
    mixed = (1.0 - seasonal_weight) * core + seasonal_weight * seasonal_mean

    out = matrix.copy()
    out[:, :usable] = mixed.reshape(n_space, usable)
    return out


def _smooth_along_time(matrix: np.ndarray, passes: int = 1) -> np.ndarray:
    if passes <= 0:
        return matrix

    out = matrix.copy()
    for _ in range(passes):
        padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    return out


@dataclass
class STTensorNDVI(TensorCompletionBaseline):
    """Approximate ST-Tensor baseline for long NDVI reconstruction.

    This is a practical, reproducible approximation of the Remote Sensing of Environment
    paper idea: low-rank spatio-temporal completion with explicit seasonal regularity.
    """

    rank: int = 20
    period: int = 12
    seasonal_weight: float = 0.35
    blend_weight: float = 0.60
    smooth_passes: int = 1
    max_iter: int = 80
    tol: float = 1e-5
    show_tqdm: bool = True
    verbose: bool = False

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if observed_tensor.ndim != 3:
            raise ValueError(f"STTensorNDVI expects 3D tensor [H, W, T], got {observed_tensor.shape}")

        h, w, t = observed_tensor.shape
        observed = mask.astype(bool)
        missing = ~observed

        matrix_obs = observed_tensor.reshape(h * w, t).astype(float)
        matrix_mask = observed.reshape(h * w, t)

        X = _temporal_linear_interpolation(matrix_obs, matrix_mask)

        history: list[float] = []
        iterator = range(self.max_iter)
        if self.show_tqdm:
            iterator = tqdm(iterator, desc="ST-Tensor", leave=False)

        for it in iterator:
            prev = X.copy()

            low_rank = _truncate_svd(X, rank=self.rank)
            seasonal = _seasonal_projection(
                low_rank,
                period=max(1, int(self.period)),
                seasonal_weight=float(self.seasonal_weight),
            )
            smooth = _smooth_along_time(seasonal, passes=max(0, int(self.smooth_passes)))

            X = (1.0 - self.blend_weight) * low_rank + self.blend_weight * smooth
            X[matrix_mask] = matrix_obs[matrix_mask]

            rel_change = np.linalg.norm(X - prev) / (np.linalg.norm(prev) + 1e-12)
            history.append(float(rel_change))

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"[STTensorNDVI] iter={it:04d} rel_change={rel_change:.3e}")

            if rel_change < self.tol:
                break

        completed = X.reshape(h, w, t)

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, completed, observed)
            test_metrics = all_metrics(full_tensor, completed, missing)

        return CompletionResult(
            tensor=completed,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "history": history,
                "rank": int(self.rank),
                "period": int(self.period),
                "seasonal_weight": float(self.seasonal_weight),
                "blend_weight": float(self.blend_weight),
            },
        )
