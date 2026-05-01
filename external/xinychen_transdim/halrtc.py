from __future__ import annotations

from typing import Any

import numpy as np


SOURCE_REPO = "https://github.com/xinychen/transdim"
SOURCE_NOTEBOOK = "https://github.com/xinychen/transdim/blob/master/imputer/HaLRTC.ipynb"


def ten2mat(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")


def mat2ten(mat: np.ndarray, dim: np.ndarray, mode: int) -> np.ndarray:
    index = [mode]
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order="F"), 0, mode)


def svt(mat: np.ndarray, tau: float) -> np.ndarray:
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    vec = s - tau
    vec[vec < 0] = 0
    return np.matmul(np.matmul(u, np.diag(vec)), v)


def HaLRTC_imputer(
    dense_tensor: np.ndarray | None,
    sparse_tensor: np.ndarray,
    alpha: list[float] | np.ndarray,
    rho: float,
    epsilon: float,
    maxiter: int,
    *,
    observed_mask: np.ndarray | None = None,
    rho_scale: float = 1.0,
    rho_max: float = 1e5,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """HaLRTC imputer adapted from xinychen/transdim's notebook.

    The source notebook marks missing entries by zeros. This adapter accepts an
    explicit observed_mask so legitimate observed zeros are preserved.
    """

    dim = np.array(sparse_tensor.shape)
    tensor_hat = np.asarray(sparse_tensor, dtype=float).copy()
    if observed_mask is None:
        observed_mask = np.asarray(sparse_tensor != 0, dtype=bool)
    else:
        observed_mask = np.asarray(observed_mask, dtype=bool)
    pos_missing = np.where(~observed_mask)

    alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha_arr.shape != (len(dim),):
        raise ValueError("alpha must have one weight per tensor mode")

    B = [np.zeros(sparse_tensor.shape, dtype=float) for _ in range(len(dim))]
    Y = [np.zeros(sparse_tensor.shape, dtype=float) for _ in range(len(dim))]
    last_ten = tensor_hat.copy()
    snorm = np.linalg.norm(tensor_hat) + 1e-12
    history: list[float] = []

    it = 0
    while True:
        for k in range(len(dim)):
            B[k] = mat2ten(
                svt(ten2mat(tensor_hat + Y[k] / rho, k), alpha_arr[k] / rho),
                dim,
                k,
            )
        tensor_hat[pos_missing] = ((sum(B) - sum(Y) / rho) / len(dim))[pos_missing]
        tensor_hat[observed_mask] = sparse_tensor[observed_mask]

        for k in range(len(dim)):
            Y[k] = Y[k] - rho * (B[k] - tensor_hat)

        tol = np.linalg.norm(tensor_hat - last_ten) / snorm
        history.append(float(tol))
        last_ten = tensor_hat.copy()
        it += 1

        if verbose:
            print(f"[HaLRTC/transdim] iter={it:04d} tolerance={tol:.6e} rho={rho:.3e}")

        if (tol < epsilon) or (it >= maxiter):
            break
        rho = min(float(rho) * float(rho_scale), float(rho_max))

    return tensor_hat, {
        "history": history,
        "final_rho": float(rho),
        "iterations": int(it),
        "source_repo": SOURCE_REPO,
        "source_notebook": SOURCE_NOTEBOOK,
    }

