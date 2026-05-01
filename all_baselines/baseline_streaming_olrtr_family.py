from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def _ensure_sensor_hour_day(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")
    # Default assumption for this paper: (sensor, hour, day)
    # If data is (sensor, day, hour), swap last two axes.
    if arr.shape[1] > arr.shape[2] and arr.shape[2] <= 72:
        return np.transpose(arr, (0, 2, 1))
    return arr


def _to_matrix_shd(arr_shd: np.ndarray) -> np.ndarray:
    n_sensor, n_hour, n_day = arr_shd.shape
    return arr_shd.reshape(n_sensor * n_hour, n_day)


def _from_matrix_shd(mat: np.ndarray, shape_shd: tuple[int, int, int]) -> np.ndarray:
    n_sensor, n_hour, n_day = shape_shd
    return mat.reshape(n_sensor, n_hour, n_day)


def _safe_mean_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    obs = mask.astype(bool)
    if np.any(obs):
        out[~obs] = float(np.mean(values[obs]))
    else:
        out.fill(0.0)
    return out


def _topk_svd_reconstruct(matrix: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros_like(matrix)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    r = min(rank, s.size)
    return (u[:, :r] * s[:r]) @ vt[:r, :]


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _fiber_shrink_vector(vec: np.ndarray, n_sensor: int, n_hour: int, lam: float) -> np.ndarray:
    out = np.zeros_like(vec)
    for s in range(n_sensor):
        start = s * n_hour
        end = start + n_hour
        block = vec[start:end]
        norm = float(np.linalg.norm(block))
        if norm <= 1e-12:
            continue
        coef = max(0.0, 1.0 - lam / norm)
        out[start:end] = coef * block
    return out


def _fiber_shrink_matrix(res: np.ndarray, n_sensor: int, n_hour: int, lam: float) -> np.ndarray:
    out = np.zeros_like(res)
    for day in range(res.shape[1]):
        out[:, day] = _fiber_shrink_vector(res[:, day], n_sensor=n_sensor, n_hour=n_hour, lam=lam)
    return out


def _ridge_solve(U_obs: np.ndarray, y_obs: np.ndarray, ridge: float) -> np.ndarray:
    r = U_obs.shape[1]
    A = U_obs.T @ U_obs + ridge * np.eye(r, dtype=float)
    b = U_obs.T @ y_obs
    return np.linalg.solve(A, b)


def _derive_outlier_fiber_mask(
    sparse_matrix: np.ndarray,
    n_sensor: int,
    n_hour: int,
    n_day: int,
    threshold: float = 1e-8,
) -> np.ndarray:
    sparse_shd = _from_matrix_shd(sparse_matrix, (n_sensor, n_hour, n_day))
    norms = np.linalg.norm(sparse_shd, axis=1)  # (sensor, day)
    return norms > threshold


def _online_proxy_recover(
    observed_matrix: np.ndarray,
    observed_mask_matrix: np.ndarray,
    n_sensor: int,
    n_hour: int,
    rank: int,
    epochs: int,
    lam_sparse: float,
    ridge: float,
    lr: float,
    sparse_mode: Literal["fiber", "element", "none"],
    show_tqdm: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    m, n = observed_matrix.shape
    rank = max(1, min(rank, min(m, n)))

    rng = np.random.default_rng(0)
    U = 0.01 * rng.standard_normal((m, rank))
    for j in range(rank):
        norm = np.linalg.norm(U[:, j])
        if norm > 1e-12:
            U[:, j] /= norm

    L = np.zeros((m, n), dtype=float)
    S = np.zeros((m, n), dtype=float)
    history: list[float] = []

    epoch_iter = tqdm(range(epochs), desc="[Streaming-online]", leave=False, disable=not show_tqdm)
    for _ in epoch_iter:
        for day in range(n):
            obs = observed_mask_matrix[:, day].astype(bool)
            if not np.any(obs):
                continue
            y = observed_matrix[:, day]

            U_obs = U[obs, :]
            y_obs = y[obs]
            coeff = _ridge_solve(U_obs, y_obs, ridge=ridge)

            for _inner in range(2):
                l_col = U @ coeff
                res = np.zeros_like(y)
                res[obs] = y[obs] - l_col[obs]

                if sparse_mode == "fiber":
                    s_col = _fiber_shrink_vector(res, n_sensor=n_sensor, n_hour=n_hour, lam=lam_sparse)
                elif sparse_mode == "element":
                    s_col = np.zeros_like(res)
                    s_col[obs] = _soft_threshold(res[obs], lam_sparse)
                else:
                    s_col = np.zeros_like(res)

                y_clean = np.zeros_like(y_obs)
                y_clean[:] = y_obs - s_col[obs]
                coeff = _ridge_solve(U_obs, y_clean, ridge=ridge)

            pred_obs = U_obs @ coeff
            grad = np.outer(y_clean - pred_obs, coeff) - ridge * U_obs
            U[obs, :] = U_obs + lr * grad

            L[:, day] = U @ coeff
            S[:, day] = s_col

        rel = float(np.linalg.norm((observed_matrix - S) - L) / (np.linalg.norm(observed_matrix) + 1e-12))
        history.append(rel)
    info = {"history": history}
    return L, S, info


def _batch_proxy_recover(
    observed_matrix: np.ndarray,
    observed_mask_matrix: np.ndarray,
    n_sensor: int,
    n_hour: int,
    rank: int,
    max_iter: int,
    lam_sparse: float,
    sparse_mode: Literal["fiber", "element", "none"],
    show_tqdm: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    X = _safe_mean_fill(observed_matrix, observed_mask_matrix)
    S = np.zeros_like(X)
    history: list[float] = []

    iter_rows = tqdm(range(max_iter), desc="[Streaming-batch]", leave=False, disable=not show_tqdm)
    for _ in iter_rows:
        L = _topk_svd_reconstruct(X, rank=rank)

        R = np.zeros_like(X)
        obs = observed_mask_matrix.astype(bool)
        R[obs] = observed_matrix[obs] - L[obs]

        if sparse_mode == "fiber":
            S = _fiber_shrink_matrix(R, n_sensor=n_sensor, n_hour=n_hour, lam=lam_sparse)
        elif sparse_mode == "element":
            S = np.zeros_like(R)
            S[obs] = _soft_threshold(R[obs], lam_sparse)
        else:
            S = np.zeros_like(R)

        X_prev = X
        X = L.copy()
        X[obs] = observed_matrix[obs] - S[obs]

        rel = float(np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-12))
        history.append(rel)
        if rel < 1e-6:
            break

    info = {"history": history}
    return L, S, info


@dataclass
class _StreamingBase(TensorCompletionBaseline):
    rank: int = 20
    epochs: int = 3
    max_iter: int = 40
    lam_sparse: float = 0.2
    ridge: float = 1e-2
    lr: float = 0.02
    show_tqdm: bool = True
    verbose: bool = False
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "fiber"
    baseline_name: str = "base"

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        obs_shd = _ensure_sensor_hour_day(np.asarray(observed_tensor, dtype=float))
        mask_shd = _ensure_sensor_hour_day(np.asarray(mask, dtype=bool))

        n_sensor, n_hour, n_day = obs_shd.shape
        obs_mat = _to_matrix_shd(obs_shd)
        mask_mat = _to_matrix_shd(mask_shd)

        if self.online:
            L_mat, S_mat, extra = _online_proxy_recover(
                observed_matrix=obs_mat,
                observed_mask_matrix=mask_mat,
                n_sensor=n_sensor,
                n_hour=n_hour,
                rank=self.rank,
                epochs=self.epochs,
                lam_sparse=self.lam_sparse,
                ridge=self.ridge,
                lr=self.lr,
                sparse_mode=self.sparse_mode,
                show_tqdm=self.show_tqdm,
            )
        else:
            L_mat, S_mat, extra = _batch_proxy_recover(
                observed_matrix=obs_mat,
                observed_mask_matrix=mask_mat,
                n_sensor=n_sensor,
                n_hour=n_hour,
                rank=self.rank,
                max_iter=self.max_iter,
                lam_sparse=self.lam_sparse,
                sparse_mode=self.sparse_mode,
                show_tqdm=self.show_tqdm,
            )

        rec = _from_matrix_shd(L_mat, (n_sensor, n_hour, n_day))
        sparse_shd = _from_matrix_shd(S_mat, (n_sensor, n_hour, n_day))
        pred_outlier_fiber = _derive_outlier_fiber_mask(
            sparse_matrix=S_mat,
            n_sensor=n_sensor,
            n_hour=n_hour,
            n_day=n_day,
            threshold=1e-8,
        )

        train_metrics = test_metrics = None
        if full_tensor is not None:
            full_shd = _ensure_sensor_hour_day(np.asarray(full_tensor, dtype=float))
            train_metrics = all_metrics(full_shd, rec, mask_shd)
            test_metrics = all_metrics(full_shd, rec, ~mask_shd)

        info = {
            "baseline": self.baseline_name,
            "online": self.online,
            "sparse_mode": self.sparse_mode,
            "outlier_fiber_mask": pred_outlier_fiber,
            "sparse_tensor": sparse_shd,
        }
        info.update(extra)

        return CompletionResult(
            tensor=rec,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info,
        )


@dataclass
class OLRTRBaseline(_StreamingBase):
    baseline_name: str = "olrtr"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "fiber"
    lam_sparse: float = 0.3
    lr: float = 0.02
    ridge: float = 1e-2


@dataclass
class ORLTMBaseline(_StreamingBase):
    baseline_name: str = "orltm"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "fiber"
    lam_sparse: float = 0.22
    lr: float = 0.015
    ridge: float = 2e-2


@dataclass
class STOCRPCABaseline(_StreamingBase):
    baseline_name: str = "stoc-rpca"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "element"
    lam_sparse: float = 0.15
    lr: float = 0.02
    ridge: float = 2e-2


@dataclass
class OSTDBaseline(_StreamingBase):
    baseline_name: str = "ostd"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "none"
    lam_sparse: float = 0.0
    lr: float = 0.02
    ridge: float = 5e-2


@dataclass
class OLRSCBaseline(_StreamingBase):
    baseline_name: str = "olrsc"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "element"
    lam_sparse: float = 0.1
    lr: float = 0.01
    ridge: float = 4e-2


@dataclass
class GRASTABaseline(_StreamingBase):
    baseline_name: str = "grasta"
    online: bool = True
    sparse_mode: Literal["fiber", "element", "none"] = "none"
    lam_sparse: float = 0.0
    lr: float = 0.03
    ridge: float = 1e-1


@dataclass
class RTRBaseline(_StreamingBase):
    baseline_name: str = "rtr"
    online: bool = False
    sparse_mode: Literal["fiber", "element", "none"] = "fiber"
    lam_sparse: float = 0.18
    max_iter: int = 60


@dataclass
class TRPCABaseline(_StreamingBase):
    baseline_name: str = "trpca"
    online: bool = False
    sparse_mode: Literal["fiber", "element", "none"] = "element"
    lam_sparse: float = 0.12
    max_iter: int = 60
