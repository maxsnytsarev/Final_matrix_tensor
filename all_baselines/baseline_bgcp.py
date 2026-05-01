from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from scipy.stats import wishart

from common import CompletionResult, TensorCompletionBaseline, all_metrics, cp_to_tensor, unfold

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def _safe_inv(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    return np.linalg.inv(sym + eps * np.eye(sym.shape[0], dtype=float))


def _safe_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=float)
    cov = np.cov(x, rowvar=False)
    return np.asarray(cov, dtype=float)


def _sample_wishart(scale: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    scale = 0.5 * (scale + scale.T)
    for jitter in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            return np.asarray(
                wishart.rvs(df=df, scale=scale + jitter * np.eye(scale.shape[0]), random_state=rng),
                dtype=float,
            )
        except Exception:
            continue
    raise RuntimeError("Could not sample Wishart matrix; scale is numerically unstable.")


def _cp_design_without_mode(factors: list[np.ndarray], mode: int) -> np.ndarray:
    """Return design matrix K with shape [prod(other dims), rank]."""
    rank = factors[0].shape[1]
    axes = [a for a in range(len(factors)) if a != mode]
    other_shape = [factors[a].shape[0] for a in axes]
    n_cols = int(np.prod(other_shape))
    K = np.empty((n_cols, rank), dtype=float)

    for r in range(rank):
        comp = np.ones(other_shape, dtype=float)
        for i, axis in enumerate(axes):
            shp = [1] * len(axes)
            shp[i] = other_shape[i]
            comp *= factors[axis][:, r].reshape(shp)
        K[:, r] = comp.reshape(-1)
    return K


def _paper_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _paper_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    if y_true.size == 0:
        return float("nan")
    idx = np.abs(y_true) > eps
    if not np.any(idx):
        return float("nan")
    return float(np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])))


@dataclass
class BGCPBase(TensorCompletionBaseline):
    """Bayesian Gaussian CP (BGCP) via Gibbs sampling."""

    cp_rank: int = 15
    burnin_iter: int = 100
    gibbs_iter: int = 200
    random_state: int = 0
    verbose: bool = False

    def _to_internal(self, tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def _from_internal(self, tensor_hat: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _run_bgcp(self, dense_tensor: np.ndarray, sparse_tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
        rng = np.random.default_rng(self.random_state)

        dim = sparse_tensor.shape
        n_modes = sparse_tensor.ndim
        rank = int(self.cp_rank)
        if rank < 1:
            raise ValueError("cp_rank must be >= 1")

        observed = mask.astype(bool)
        observed_float = observed.astype(float)
        eval_pos = (dense_tensor > 0) & (~observed)
        obs_count = int(np.sum(observed))

        factors = [0.1 * rng.standard_normal((dim[k], rank)) for k in range(n_modes)]

        beta0 = 1.0
        nu0 = float(rank)
        mu0 = np.zeros(rank, dtype=float)
        tau_eps = 1.0
        a0 = 1.0
        b0 = 1.0
        W0 = np.eye(rank, dtype=float)
        inv_W0 = np.eye(rank, dtype=float)

        rmse_history: list[float] = []
        mape_history: list[float] = []

        def sample_one_iteration() -> None:
            nonlocal tau_eps, factors

            for mode in range(n_modes):
                U = factors[mode]
                dim_k = U.shape[0]

                U_bar = np.mean(U, axis=0)
                var_mu0 = (dim_k * U_bar + beta0 * mu0) / (dim_k + beta0)
                var_nu = dim_k + nu0
                centered = U_bar - mu0
                cov_u = _safe_cov(U)
                tmp = inv_W0 + (dim_k - 1) * cov_u + (dim_k * beta0 / (dim_k + beta0)) * np.outer(centered, centered)
                var_W = _safe_inv(tmp)
                Lambda0 = _sample_wishart(var_W, var_nu, rng)
                mu_cov = _safe_inv((dim_k + beta0) * Lambda0)
                mu_draw = rng.multivariate_normal(var_mu0, mu_cov)

                Yk = unfold(sparse_tensor, mode)  # [dim_k, J]
                Mk = unfold(observed_float, mode)  # [dim_k, J]
                K = _cp_design_without_mode(factors, mode)  # [J, rank]
                KT = K.T

                new_U = np.empty_like(U, dtype=float)
                prior_term = Lambda0 @ mu_draw

                for i in range(dim_k):
                    w = Mk[i, :]
                    y = Yk[i, :]
                    weighted_K = K * w[:, None]
                    precision = tau_eps * (KT @ weighted_K) + Lambda0
                    precision = 0.5 * (precision + precision.T)
                    cov_row = _safe_inv(precision)
                    mean_row = cov_row @ (tau_eps * (KT @ (w * y)) + prior_term)
                    new_U[i, :] = rng.multivariate_normal(mean_row, cov_row)

                factors[mode] = new_U

            tensor_hat_now = cp_to_tensor(factors)
            error_obs = sparse_tensor[observed] - tensor_hat_now[observed]
            var_a = a0 + 0.5 * obs_count
            var_b = b0 + 0.5 * float(np.sum(error_obs ** 2))
            tau_eps = float(rng.gamma(shape=var_a, scale=1.0 / max(var_b, 1e-12)))

            if np.any(eval_pos):
                y_true = dense_tensor[eval_pos]
                y_pred = tensor_hat_now[eval_pos]
                rmse_history.append(_paper_rmse(y_true, y_pred))
                mape_history.append(_paper_mape(y_true, y_pred))
            else:
                rmse_history.append(float("nan"))
                mape_history.append(float("nan"))

        burnin_iter = int(self.burnin_iter)
        gibbs_iter = int(self.gibbs_iter)
        burnin_range = tqdm(range(burnin_iter), desc="[BGCP] burn-in", leave=False) if self.verbose else range(burnin_iter)
        for it in burnin_range:
            sample_one_iteration()
            if self.verbose and ((it + 1) % 10 == 0 or it + 1 == burnin_iter):
                print(f"[BGCP-burnin] iter={it + 1:04d} RMSE={rmse_history[-1]:.6f} MAPE={mape_history[-1]:.6f}")

        factor_sum = [np.zeros_like(f) for f in factors]
        tensor_sum = np.zeros(dim, dtype=float)

        gibbs_range = tqdm(range(gibbs_iter), desc="[BGCP] Gibbs", leave=False) if self.verbose else range(gibbs_iter)
        for it in gibbs_range:
            sample_one_iteration()
            for k in range(n_modes):
                factor_sum[k] += factors[k]
            tensor_sum += cp_to_tensor(factors)

            if self.verbose and ((it + 1) % 10 == 0 or it + 1 == gibbs_iter):
                print(f"[BGCP-gibbs] iter={it + 1:04d} RMSE={rmse_history[-1]:.6f} MAPE={mape_history[-1]:.6f}")

        factor_avg = [f / max(gibbs_iter, 1) for f in factor_sum]
        tensor_hat = tensor_sum / max(gibbs_iter, 1)
        tensor_hat[observed] = sparse_tensor[observed]

        info = {
            "rmse_history": rmse_history,
            "mape_history": mape_history,
            "paper_final_rmse": _paper_rmse(dense_tensor[eval_pos], tensor_hat[eval_pos]) if np.any(eval_pos) else float("nan"),
            "paper_final_mape": _paper_mape(dense_tensor[eval_pos], tensor_hat[eval_pos]) if np.any(eval_pos) else float("nan"),
        }
        return tensor_hat, factor_avg, info

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if full_tensor is None:
            dense_tensor = observed_tensor.astype(float).copy()
        else:
            dense_tensor = full_tensor.astype(float).copy()

        sparse_tensor = observed_tensor.astype(float).copy()
        observed = mask.astype(bool)

        sparse_internal, mask_internal, meta = self._to_internal(sparse_tensor, observed)
        dense_internal, _, _ = self._to_internal(dense_tensor, np.ones_like(observed, dtype=bool))

        tensor_hat_internal, factors_internal, info = self._run_bgcp(
            dense_tensor=dense_internal,
            sparse_tensor=sparse_internal,
            mask=mask_internal,
        )
        tensor_hat = self._from_internal(tensor_hat_internal, meta)
        tensor_hat[observed] = sparse_tensor[observed]

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, tensor_hat, observed)
            test_metrics = all_metrics(full_tensor, tensor_hat, ~observed)

        return CompletionResult(
            tensor=tensor_hat,
            factors=factors_internal,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info | {"representation_meta": meta},
        )


@dataclass
class BGCPMatrix(BGCPBase):
    """BGCP with matrix representation: (road segment, full time series)."""

    def _to_internal(self, tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if tensor.ndim < 2:
            raise ValueError("BGCPMatrix expects at least 2D input.")
        lead = tensor.shape[0]
        trailing = int(np.prod(tensor.shape[1:]))
        return (
            tensor.reshape(lead, trailing),
            mask.reshape(lead, trailing),
            {"orig_shape": tuple(tensor.shape)},
        )

    def _from_internal(self, tensor_hat: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        return tensor_hat.reshape(meta["orig_shape"])


@dataclass
class BGCPTensor3(BGCPBase):
    """BGCP with third-order tensor representation."""

    def _to_internal(self, tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if tensor.ndim == 3:
            return tensor.copy(), mask.copy(), {"orig_shape": tuple(tensor.shape), "direct_3d": True}
        lead = tensor.shape[0]
        if tensor.ndim > 3:
            tensor_3d = tensor.reshape(lead, tensor.shape[1], int(np.prod(tensor.shape[2:])))
            mask_3d = mask.reshape(lead, mask.shape[1], int(np.prod(mask.shape[2:])))
            return tensor_3d, mask_3d, {"orig_shape": tuple(tensor.shape), "direct_3d": False}
        raise ValueError("BGCPTensor3 expects at least 3D input.")

    def _from_internal(self, tensor_hat: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        if meta["direct_3d"]:
            return tensor_hat
        return tensor_hat.reshape(meta["orig_shape"])


@dataclass
class BGCPTensor4(BGCPBase):
    """BGCP with fourth-order representation: (segment, week, day, time)."""

    def _to_internal(self, tensor: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if tensor.ndim != 3:
            raise ValueError("BGCPTensor4 expects 3D input shaped as (segment, day, time).")

        n_seg, n_days, n_steps = tensor.shape
        n_weeks = int(ceil(n_days / 7))
        pad_days = n_weeks * 7 - n_days

        if pad_days > 0:
            pad_tensor = np.zeros((n_seg, pad_days, n_steps), dtype=float)
            pad_mask = np.zeros((n_seg, pad_days, n_steps), dtype=bool)
            tensor = np.concatenate([tensor, pad_tensor], axis=1)
            mask = np.concatenate([mask, pad_mask], axis=1)

        tensor4 = tensor.reshape(n_seg, n_weeks, 7, n_steps)
        mask4 = mask.reshape(n_seg, n_weeks, 7, n_steps)

        meta = {
            "orig_shape": (n_seg, n_days, n_steps),
            "n_weeks": n_weeks,
            "pad_days": pad_days,
        }
        return tensor4, mask4, meta

    def _from_internal(self, tensor_hat: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        n_seg, n_days, n_steps = meta["orig_shape"]
        tensor3 = tensor_hat.reshape(n_seg, meta["n_weeks"] * 7, n_steps)
        return tensor3[:, :n_days, :]
