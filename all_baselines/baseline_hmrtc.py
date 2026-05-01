from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable
def _cp_to_tensor_complex(factors: list[np.ndarray]) -> np.ndarray:
    rank = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)
    out = np.zeros(shape, dtype=np.complex128)
    for r in range(rank):
        comp = np.array(1.0 + 0.0j, dtype=np.complex128)
        for mode, factor in enumerate(factors):
            vec = factor[:, r]
            reshape = [1] * len(factors)
            reshape[mode] = vec.size
            comp = comp * vec.reshape(reshape)
        out += comp
    return out


def _soft_svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (u * s_thr) @ vh


def _make_hankel_cache(length: int) -> dict[str, np.ndarray | int]:
    s1 = int(np.ceil((length + 1) / 2.0))
    s2 = int(length - s1 + 1)
    i = np.arange(s1)[:, None]
    j = np.arange(s2)[None, :]
    index_matrix = i + j
    counts = np.bincount(index_matrix.ravel(), minlength=length).astype(float)
    return {
        "length": length,
        "s1": s1,
        "s2": s2,
        "index_matrix": index_matrix.astype(int),
        "flat_index": index_matrix.ravel().astype(int),
        "counts": counts,
    }


def _vec_to_hankel(vec: np.ndarray, cache: dict[str, np.ndarray | int]) -> np.ndarray:
    return vec[np.asarray(cache["index_matrix"], dtype=int)]


def _hankel_adjoint_sum(matrix: np.ndarray, cache: dict[str, np.ndarray | int]) -> np.ndarray:
    out = np.zeros(int(cache["length"]), dtype=np.complex128)
    np.add.at(out, np.asarray(cache["flat_index"], dtype=int), matrix.ravel())
    return out


def _hankel_average(matrix: np.ndarray, cache: dict[str, np.ndarray | int]) -> np.ndarray:
    counts = np.asarray(cache["counts"], dtype=float)
    return _hankel_adjoint_sum(matrix, cache) / np.maximum(counts, 1.0)


def _complex_rlne(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        t = true.reshape(-1)
        p = pred.reshape(-1)
    else:
        idx = mask.astype(bool)
        if not np.any(idx):
            return float("nan")
        t = true[idx]
        p = pred[idx]
    denom = np.linalg.norm(t.ravel()) + 1e-12
    return float(np.linalg.norm((p - t).ravel()) / denom)


def _build_mode_groups(
    observed_tensor: np.ndarray,
    mask: np.ndarray,
) -> list[list[dict[str, np.ndarray]]]:
    shape = observed_tensor.shape
    ndim = observed_tensor.ndim
    obs_indices = np.argwhere(mask.astype(bool))
    obs_values = observed_tensor[mask.astype(bool)].astype(np.complex128)

    grouped: list[list[dict[str, np.ndarray]]] = []
    for mode in range(ndim):
        other_modes = [m for m in range(ndim) if m != mode]
        other_shape = tuple(shape[m] for m in other_modes)
        buckets_idx: list[list[tuple[int, ...]]] = [[] for _ in range(shape[mode])]
        buckets_val: list[list[complex]] = [[] for _ in range(shape[mode])]
        buckets_col: list[list[int]] = [[] for _ in range(shape[mode])]

        for idx, val in zip(obs_indices, obs_values, strict=False):
            row = int(idx[mode])
            other_idx = tuple(int(idx[m]) for m in other_modes)
            col = int(np.ravel_multi_index(other_idx, other_shape, order="C"))
            buckets_idx[row].append(other_idx)
            buckets_val[row].append(val)
            buckets_col[row].append(col)

        rows: list[dict[str, np.ndarray]] = []
        for row in range(shape[mode]):
            if buckets_idx[row]:
                rows.append(
                    {
                        "other_idx": np.asarray(buckets_idx[row], dtype=int),
                        "values": np.asarray(buckets_val[row], dtype=np.complex128),
                        "cols": np.asarray(buckets_col[row], dtype=int),
                    }
                )
            else:
                rows.append(
                    {
                        "other_idx": np.empty((0, ndim - 1), dtype=int),
                        "values": np.empty((0,), dtype=np.complex128),
                        "cols": np.empty((0,), dtype=int),
                    }
                )
        grouped.append(rows)
    return grouped


def _mode_design_matrix(
    factors: list[np.ndarray],
    mode: int,
    max_elements: int | None,
) -> np.ndarray | None:
    other_modes = [m for m in range(len(factors)) if m != mode]
    rank = factors[0].shape[1]
    prod_other = 1
    for m in other_modes:
        prod_other *= factors[m].shape[0]
    if max_elements is not None and prod_other * rank > max_elements:
        return None

    cols: list[np.ndarray] = []
    for r in range(rank):
        comp = factors[other_modes[0]][:, r]
        for m in other_modes[1:]:
            comp = np.multiply.outer(comp, factors[m][:, r])
        cols.append(np.asarray(comp, dtype=np.complex128).reshape(-1, order="C"))
    return np.stack(cols, axis=1)


def _local_design_matrix(
    factors: list[np.ndarray],
    mode: int,
    other_idx: np.ndarray,
) -> np.ndarray:
    rank = factors[0].shape[1]
    if other_idx.size == 0:
        return np.empty((0, rank), dtype=np.complex128)
    other_modes = [m for m in range(len(factors)) if m != mode]
    design = np.ones((other_idx.shape[0], rank), dtype=np.complex128)
    for col, m in enumerate(other_modes):
        design *= factors[m][other_idx[:, col], :]
    return design


@dataclass
class HMRTC(TensorCompletionBaseline):
    """Paper-like HMRTC baseline for complex N-D exponential signal completion."""

    rank: int = 100
    lambda_reg: float = 1e3
    rho: float = 1.05
    beta0: float = 0.1
    tol: float = 1e-4
    max_iter: int = 1000
    random_state: int = 0
    init_scale: float = 0.1
    ls_reg: float = 1e-8
    max_full_design_elements: int | None = 20_000_000
    verbose: bool = False

    def _init_factors(self, shape: tuple[int, ...], observed_tensor: np.ndarray) -> list[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        scale = self.init_scale
        if np.any(np.abs(observed_tensor) > 0):
            scale *= float(np.mean(np.abs(observed_tensor[np.abs(observed_tensor) > 0])))
        factors: list[np.ndarray] = []
        for dim in shape:
            real = rng.standard_normal((dim, self.rank))
            imag = rng.standard_normal((dim, self.rank))
            mat = (real + 1j * imag) * scale / max(np.sqrt(self.rank), 1.0)
            norms = np.linalg.norm(mat, axis=0, keepdims=True) + 1e-12
            factors.append(mat / norms)
        return factors

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        y = np.asarray(observed_tensor, dtype=np.complex128)
        obs = mask.astype(bool)
        shape = y.shape
        ndim = y.ndim

        factors = self._init_factors(shape, y)
        hankel_cache = [_make_hankel_cache(dim) for dim in shape]
        groups = _build_mode_groups(y, obs)

        Z: list[list[np.ndarray]] = []
        D: list[list[np.ndarray]] = []
        for mode in range(ndim):
            z_mode: list[np.ndarray] = []
            d_mode: list[np.ndarray] = []
            for r in range(self.rank):
                h = _vec_to_hankel(factors[mode][:, r], hankel_cache[mode])
                z_mode.append(h.copy())
                d_mode.append(np.zeros_like(h, dtype=np.complex128))
            Z.append(z_mode)
            D.append(d_mode)

        beta = float(self.beta0)
        pred = _cp_to_tensor_complex(factors)
        history: list[dict[str, float]] = []

        eye = np.eye(self.rank, dtype=np.complex128)

        for it in tqdm(
            range(self.max_iter),
            disable=not self.verbose,
            leave=False,
            desc="HMRTC iters",
        ):
            pred_prev = pred

            for mode in tqdm(
                range(ndim),
                disable=not self.verbose,
                leave=False,
                desc="HMRTC modes",
            ):
                cache = hankel_cache[mode]
                counts = np.asarray(cache["counts"], dtype=float)
                U_new = np.empty_like(factors[mode], dtype=np.complex128)
                hankel_targets = np.empty_like(factors[mode], dtype=np.complex128)

                for r in range(self.rank):
                    hankel_targets[:, r] = _hankel_average(
                        Z[mode][r] - D[mode][r] / beta,
                        cache,
                    )

                design_full = _mode_design_matrix(
                    factors=factors,
                    mode=mode,
                    max_elements=self.max_full_design_elements,
                )

                for i in range(shape[mode]):
                    group = groups[mode][i]
                    b_i = hankel_targets[i, :]
                    if group["values"].size == 0:
                        U_new[i, :] = b_i
                        continue

                    if design_full is not None:
                        G_i = design_full[group["cols"], :]
                    else:
                        G_i = _local_design_matrix(factors, mode, group["other_idx"])

                    y_i = group["values"]
                    gram = G_i.conj().T @ G_i
                    rhs = G_i.conj().T @ y_i
                    weight = beta * counts[i]
                    lhs = self.lambda_reg * gram + (weight + self.ls_reg) * eye
                    rhs = self.lambda_reg * rhs + weight * b_i
                    U_new[i, :] = np.linalg.solve(lhs, rhs)

                factors[mode] = U_new

            for mode in range(ndim):
                cache = hankel_cache[mode]
                for r in range(self.rank):
                    h = _vec_to_hankel(factors[mode][:, r], cache)
                    aux = h + D[mode][r] / beta
                    Z[mode][r] = _soft_svt(aux, 1.0 / beta)
                    D[mode][r] = D[mode][r] + beta * (h - Z[mode][r])

            pred = _cp_to_tensor_complex(factors)
            rel_change = float(
                np.linalg.norm((pred - pred_prev).ravel())
                / (np.linalg.norm(pred_prev.ravel()) + 1e-12)
            )
            record: dict[str, float] = {
                "iter": float(it),
                "rel_change": rel_change,
                "beta": beta,
            }
            if full_tensor is not None:
                record["rlne_full"] = _complex_rlne(np.asarray(full_tensor, dtype=np.complex128), pred)
                record["rlne_hidden"] = _complex_rlne(
                    np.asarray(full_tensor, dtype=np.complex128),
                    pred,
                    ~obs,
                )
            history.append(record)

            if self.verbose and (it % 5 == 0 or it == self.max_iter - 1):
                msg = f"[HMRTC] iter={it:04d} rel_change={rel_change:.3e} beta={beta:.3e}"
                if "rlne_full" in record:
                    msg += f" rlne={record['rlne_full']:.6f}"
                print(msg, flush=True)

            if rel_change < self.tol:
                break
            beta *= self.rho

        train_metrics = test_metrics = None
        if full_tensor is not None:
            truth = np.asarray(full_tensor, dtype=np.complex128)
            train_metrics = {"rlne": _complex_rlne(truth, pred, obs)}
            test_metrics = {"rlne": _complex_rlne(truth, pred, ~obs)}

        tensor_out = pred
        if not np.iscomplexobj(observed_tensor) and np.max(np.abs(pred.imag)) < 1e-10:
            tensor_out = pred.real

        return CompletionResult(
            tensor=tensor_out,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "history": history,
                "final_beta": beta,
                "rank": int(self.rank),
                "lambda_reg": float(self.lambda_reg),
            },
        )
