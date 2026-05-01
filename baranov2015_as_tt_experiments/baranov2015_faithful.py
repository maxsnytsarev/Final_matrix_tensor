from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from as_tt_water_experiments import run_as_tt_water_experiments as water

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else None


class QueryBudgetExceeded(RuntimeError):
    pass


def _paper_log(msg: str, use_tqdm: bool = False) -> None:
    if use_tqdm and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg, flush=True)


@dataclass(frozen=True)
class PlanarWaterContext:
    symbols: tuple[str, ...]
    base_coords: np.ndarray
    coord_unit: str
    raw_dim: int
    as_dim: int
    raw_active_dofs: tuple[tuple[int, int], ...]
    as_basis: np.ndarray
    as_eigenvalues: np.ndarray
    as_cache_path: str | None
    as_cache_tag: str


@dataclass
class PaperASTTResult:
    tensor: np.ndarray | None
    coeff_tensor: np.ndarray | None
    coeff_tt_cores: list[np.ndarray] | None
    tt_cores: list[np.ndarray] | None
    queried_flat_indices: list[int]
    queried_index_sequence: list[list[int]]
    total_query_count: int
    unique_query_count: int
    converged: bool
    truncated_by_budget: bool
    max_rank: int | None
    rms_random: float | None
    info: dict[str, Any]


def chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    return water.chebyshev_nodes(n=n, a=a, b=b)


def _paper_water_active_dofs() -> tuple[tuple[int, int], ...]:
    # Paper setup for water: O fixed at origin, both H atoms constrained to the yz-plane.
    # Thus the free Cartesian variables are (H1_y, H1_z, H2_y, H2_z).
    return ((1, 1), (1, 2), (2, 1), (2, 2))


def _coords_from_raw_point(context: PlanarWaterContext, raw_point: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_point, dtype=float).reshape(-1)
    if x.shape != (context.raw_dim,):
        raise ValueError(f"Raw point shape mismatch: {x.shape} vs {(context.raw_dim,)}")
    coords = np.asarray(context.base_coords, dtype=float).copy()
    for mode, delta in enumerate(x):
        atom_i, xyz_i = context.raw_active_dofs[mode]
        coords[atom_i, xyz_i] += float(delta)
    return coords


def _raw_gradient_from_full_gradient(context: PlanarWaterContext, full_grad: np.ndarray) -> np.ndarray:
    grad = np.asarray(full_grad, dtype=float)
    out = np.zeros(context.raw_dim, dtype=float)
    for mode, (atom_i, xyz_i) in enumerate(context.raw_active_dofs):
        out[mode] = float(grad[atom_i, xyz_i])
    return out


def _raw_point_from_reduced(context: PlanarWaterContext, reduced_point: np.ndarray) -> np.ndarray:
    y = np.asarray(reduced_point, dtype=float).reshape(-1)
    if y.shape != (context.as_dim,):
        raise ValueError(f"Reduced point shape mismatch: {y.shape} vs {(context.as_dim,)}")
    return np.asarray(context.as_basis @ y, dtype=float)


def _coords_from_reduced(context: PlanarWaterContext, reduced_point: np.ndarray) -> np.ndarray:
    return _coords_from_raw_point(context, _raw_point_from_reduced(context, reduced_point))


def _paper_as_cache_path(
    data_root: Path,
    *,
    basis: str,
    qc_method: str,
    coord_unit: str,
    sigma2: float,
    samples: int,
    seed: int,
    raw_dim: int,
    as_dim: int,
) -> Path:
    basis_slug = water._basis_slug(basis)
    unit_slug = water._normalize_coord_unit(coord_unit).lower()
    return (
        Path(data_root)
        / "paper_active_subspace"
        / (
            f"baranov2015_water_planar_raw{raw_dim}_m{as_dim}_unit{unit_slug}"
            f"_sigma2_{sigma2:.6g}_samples{samples}_seed{seed}"
            f"_{str(qc_method).lower()}_{basis_slug}.npz"
        )
    )


def prepare_planar_water_context(args: Any) -> PlanarWaterContext:
    geometry_path = water.resolve_existing_project_path(getattr(args, "geometry_xyz"))
    symbols, base_coords_ang = water.ensure_water_geometry(geometry_path)
    coord_unit = water._normalize_coord_unit(getattr(args, "water_coordinate_unit", "Bohr"))
    base_coords = water._coords_from_angstrom(base_coords_ang, coord_unit)
    raw_active_dofs = _paper_water_active_dofs()
    raw_dim = len(raw_active_dofs)
    as_dim = int(getattr(args, "water_as_dim", raw_dim))
    if as_dim <= 0 or as_dim > raw_dim:
        raise ValueError(f"water_as_dim must be in [1, {raw_dim}], got {as_dim}")

    cache_path = _paper_as_cache_path(
        getattr(args, "data_root"),
        basis=str(getattr(args, "basis", "cc-pvdz")),
        qc_method=str(getattr(args, "qc_method", "RHF")),
        coord_unit=coord_unit,
        sigma2=float(getattr(args, "water_as_sigma2", 0.1)),
        samples=int(getattr(args, "water_as_samples", 64)),
        seed=int(getattr(args, "water_as_random_state", getattr(args, "random_state", 1000))),
        raw_dim=raw_dim,
        as_dim=as_dim,
    )
    cache_lookup_path = water.resolve_existing_project_path(cache_path)
    use_tqdm = bool(getattr(args, "show_tqdm", False))
    verbose = bool(getattr(args, "verbose", False))

    if bool(getattr(args, "cache_tensors", True)) and cache_lookup_path.exists():
        payload = np.load(cache_lookup_path)
        as_basis = np.asarray(payload["basis"], dtype=float)
        as_eigenvalues = np.asarray(payload["eigenvalues"], dtype=float)
        if verbose:
            _paper_log(f"[paper-as] using cached active subspace -> {cache_lookup_path}", use_tqdm=use_tqdm)
        return PlanarWaterContext(
            symbols=tuple(symbols),
            base_coords=base_coords,
            coord_unit=coord_unit,
            raw_dim=raw_dim,
            as_dim=as_dim,
            raw_active_dofs=raw_active_dofs,
            as_basis=as_basis,
            as_eigenvalues=as_eigenvalues,
            as_cache_path=str(cache_lookup_path),
            as_cache_tag=cache_lookup_path.stem,
        )

    if not bool(getattr(args, "generate_if_missing", True)):
        raise FileNotFoundError(
            f"Missing paper-like active-subspace cache: {cache_path}\n"
            "Set generate_if_missing=True to compute gradients and cache the active subspace."
        )

    sigma2 = float(getattr(args, "water_as_sigma2", 0.1))
    samples = int(getattr(args, "water_as_samples", 64))
    seed = int(getattr(args, "water_as_random_state", getattr(args, "random_state", 1000)))
    rng = np.random.default_rng(seed)
    std = float(np.sqrt(sigma2))
    _energy_fn, gradient_fn = water._build_qc_oracles(args)

    cov = np.zeros((raw_dim, raw_dim), dtype=float)
    iterator = range(samples)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"AS gradients (paper raw={raw_dim}->m={as_dim})", unit="pt", leave=False)
    if verbose:
        _paper_log(
            f"[paper-as] START samples={samples} sigma2={sigma2} raw_dim={raw_dim} as_dim={as_dim}",
            use_tqdm=use_tqdm,
        )
    for _ in iterator:
        raw_point = rng.normal(loc=0.0, scale=std, size=raw_dim)
        coords = _coords_from_raw_point(
            PlanarWaterContext(
                symbols=tuple(symbols),
                base_coords=base_coords,
                coord_unit=coord_unit,
                raw_dim=raw_dim,
                as_dim=as_dim,
                raw_active_dofs=raw_active_dofs,
                as_basis=np.eye(raw_dim, dtype=float),
                as_eigenvalues=np.ones(raw_dim, dtype=float),
                as_cache_path=None,
                as_cache_tag="tmp",
            ),
            raw_point,
        )
        grad_full = np.asarray(gradient_fn(list(symbols), coords), dtype=float)
        grad_raw = _raw_gradient_from_full_gradient(
            PlanarWaterContext(
                symbols=tuple(symbols),
                base_coords=base_coords,
                coord_unit=coord_unit,
                raw_dim=raw_dim,
                as_dim=as_dim,
                raw_active_dofs=raw_active_dofs,
                as_basis=np.eye(raw_dim, dtype=float),
                as_eigenvalues=np.ones(raw_dim, dtype=float),
                as_cache_path=None,
                as_cache_tag="tmp",
            ),
            grad_full,
        )
        cov += np.outer(grad_raw, grad_raw)
    cov /= float(samples)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    as_basis = water._normalize_eigenvector_signs(np.asarray(eigenvectors[:, order[:as_dim]], dtype=float))

    if bool(getattr(args, "cache_tensors", True)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            basis=as_basis,
            eigenvalues=eigenvalues,
            base_coords=base_coords,
        )
        if verbose:
            _paper_log(f"[paper-as] saved active subspace -> {cache_path}", use_tqdm=use_tqdm)
    if verbose:
        _paper_log(
            f"[paper-as] DONE top_eigs={np.asarray(eigenvalues[:min(4, len(eigenvalues))]).tolist()}",
            use_tqdm=use_tqdm,
        )

    return PlanarWaterContext(
        symbols=tuple(symbols),
        base_coords=base_coords,
        coord_unit=coord_unit,
        raw_dim=raw_dim,
        as_dim=as_dim,
        raw_active_dofs=raw_active_dofs,
        as_basis=as_basis,
        as_eigenvalues=eigenvalues,
        as_cache_path=str(cache_path),
        as_cache_tag=cache_path.stem,
    )


def prepare_planar_identity_context(args: Any) -> PlanarWaterContext:
    geometry_path = water.resolve_existing_project_path(getattr(args, "geometry_xyz"))
    symbols, base_coords_ang = water.ensure_water_geometry(geometry_path)
    coord_unit = water._normalize_coord_unit(getattr(args, "water_coordinate_unit", "Bohr"))
    base_coords = water._coords_from_angstrom(base_coords_ang, coord_unit)
    raw_active_dofs = _paper_water_active_dofs()
    raw_dim = len(raw_active_dofs)
    return PlanarWaterContext(
        symbols=tuple(symbols),
        base_coords=base_coords,
        coord_unit=coord_unit,
        raw_dim=raw_dim,
        as_dim=raw_dim,
        raw_active_dofs=raw_active_dofs,
        as_basis=np.eye(raw_dim, dtype=float),
        as_eigenvalues=np.ones(raw_dim, dtype=float),
        as_cache_path=None,
        as_cache_tag="baranov2015_water_planar_identity",
    )


class PaperGridOracle:
    def __init__(
        self,
        *,
        context: PlanarWaterContext,
        nodes: np.ndarray,
        energy_fn,
        max_total_queries: int | None = None,
        max_unique_queries: int | None = None,
    ) -> None:
        self.context = context
        self.nodes = np.asarray(nodes, dtype=float)
        self.energy_fn = energy_fn
        self.shape = tuple([len(self.nodes)] * context.as_dim)
        self.max_total_queries = None if max_total_queries is None else int(max_total_queries)
        self.max_unique_queries = None if max_unique_queries is None else int(max_unique_queries)
        self._cache: dict[tuple[int, ...], float] = {}
        self.query_sequence: list[tuple[int, ...]] = []
        self.unique_query_order: list[tuple[int, ...]] = []

    def _check_budget_before_unique_insert(self, idx: tuple[int, ...]) -> None:
        if self.max_unique_queries is not None and idx not in self._cache:
            if len(self._cache) >= self.max_unique_queries:
                raise QueryBudgetExceeded(
                    f"Unique query budget exceeded: {len(self._cache)} >= {self.max_unique_queries}"
                )

    def _check_budget_before_total_insert(self) -> None:
        if self.max_total_queries is not None and len(self.query_sequence) >= self.max_total_queries:
            raise QueryBudgetExceeded(
                f"Total query budget exceeded: {len(self.query_sequence)} >= {self.max_total_queries}"
            )

    def query(self, idx: tuple[int, ...]) -> float:
        idx = tuple(int(i) for i in idx)
        if len(idx) != len(self.shape):
            raise ValueError(f"Index dimensionality mismatch: {idx} vs ndim={len(self.shape)}")
        self._check_budget_before_total_insert()
        self._check_budget_before_unique_insert(idx)
        self.query_sequence.append(idx)
        if idx in self._cache:
            return self._cache[idx]

        reduced_point = np.asarray([self.nodes[i] for i in idx], dtype=float)
        coords = _coords_from_reduced(self.context, reduced_point)
        value = float(self.energy_fn(list(self.context.symbols), coords))
        self._cache[idx] = value
        self.unique_query_order.append(idx)
        return value

    def queried_flat_indices(self) -> list[int]:
        if not self._cache:
            return []
        return [int(np.ravel_multi_index(idx, self.shape)) for idx in self.unique_query_order]

    def total_queries(self) -> int:
        return int(len(self.query_sequence))

    def unique_queries(self) -> int:
        return int(len(self._cache))


def _tt_to_tensor(cores: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(cores[0], dtype=float)
    for core in cores[1:]:
        out = np.tensordot(out, np.asarray(core, dtype=float), axes=([-1], [0]))
    return np.squeeze(out, axis=(0, -1))


def _clip_tt_ranks(shape: tuple[int, ...], rank: list[int] | int) -> list[int]:
    d = len(shape)
    if isinstance(rank, int):
        raw = [1] + [int(rank)] * (d - 1) + [1]
    else:
        raw = [int(x) for x in rank]
    if len(raw) != d + 1:
        raise ValueError(f"rank must have length {d + 1}, got {len(raw)}")
    if raw[0] != 1 or raw[-1] != 1:
        raise ValueError("TT rank boundary conditions require rank[0] == rank[-1] == 1")

    clipped = [1] * (d + 1)
    clipped[0] = 1
    clipped[-1] = 1
    r_prev = 1
    for k in range(d - 1):
        left = r_prev * int(shape[k])
        right = int(np.prod(shape[k + 1:]))
        r_next = max(1, min(int(raw[k + 1]), left, right))
        clipped[k + 1] = r_next
        r_prev = r_next
    return clipped


def _tt_round_dense(tensor: np.ndarray, tol: float) -> tuple[list[np.ndarray], list[int]]:
    arr = np.asarray(tensor, dtype=float)
    shape = arr.shape
    d = len(shape)
    if d == 1:
        return [arr.reshape(1, shape[0], 1)], []
    norm = float(np.linalg.norm(arr))
    eps = float(tol) * max(norm, 1e-12) / np.sqrt(max(1, d - 1))
    cores: list[np.ndarray] = []
    ranks: list[int] = []
    curr = arr.copy()
    r_prev = 1
    for k in range(d - 1):
        curr = curr.reshape(r_prev * shape[k], -1)
        u, s, vt = np.linalg.svd(curr, full_matrices=False)
        tail_sq = np.cumsum((s[::-1] ** 2))
        keep = s.size
        for i in range(s.size):
            tail = np.sqrt(tail_sq[s.size - 1 - i]) if i < s.size - 1 else 0.0
            if tail <= eps:
                keep = i + 1
                break
        keep = max(1, keep)
        u = u[:, :keep]
        s = s[:keep]
        vt = vt[:keep, :]
        cores.append(u.reshape(r_prev, shape[k], keep))
        ranks.append(int(keep))
        curr = s[:, None] * vt
        r_prev = keep
    cores.append(curr.reshape(r_prev, shape[-1], 1))
    return cores, ranks


def _tt_ranks_from_cores(cores: list[np.ndarray]) -> list[int]:
    if len(cores) <= 1:
        return [1]
    return [1] + [int(core.shape[2]) for core in cores[:-1]] + [1]


def _maxvol(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    n, r = A.shape
    row_idx = np.zeros(r, dtype=int)
    rest = np.arange(n, dtype=int)
    A_new = A.copy()
    i = 0
    while i < r:
        rows_norms = np.sum(A_new ** 2, axis=1)
        if rows_norms.ndim == 0:
            row_idx[i] = int(rest)
            break
        zero_mask = rows_norms == 0
        if np.any(zero_mask):
            keep = ~zero_mask
            rest = rest[keep]
            A_new = A_new[keep]
            continue
        local_idx = int(np.argmax(rows_norms))
        max_row = A[rest[local_idx], :]
        projection = A_new @ max_row.T
        normalization = np.sqrt(rows_norms[local_idx] * rows_norms)
        projection = projection / normalization
        A_new = A_new - A_new * projection[:, None]
        mask = np.ones(A_new.shape[0], dtype=bool)
        mask[local_idx] = False
        row_idx[i] = int(rest[local_idx])
        rest = rest[mask]
        A_new = A_new[mask]
        i += 1
    inverse = np.linalg.solve(A[row_idx, :], np.eye(r, dtype=float))
    return row_idx, inverse


def _left_right_step(
    oracle: PaperGridOracle,
    k: int,
    rank: list[int],
    row_idx: list[list[tuple[int, ...]]],
    col_idx: list[list[tuple[int, ...]]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    n_k = oracle.shape[k]
    fibers: list[tuple[int, ...]] = []
    if k == 0:
        core = np.zeros((rank[k], n_k, rank[k + 1]), dtype=float)
        for j in range(rank[k + 1]):
            right = col_idx[k][j]
            for s in range(n_k):
                idx = (s,) + right
                fibers.append(idx)
                core[0, s, j] = oracle.query(idx)
    else:
        gathered = np.zeros((rank[k] * rank[k + 1], n_k), dtype=float)
        pair = 0
        for left in row_idx[k]:
            for right in col_idx[k]:
                for s in range(n_k):
                    idx = left + (s,) + right
                    fibers.append(idx)
                    gathered[pair, s] = oracle.query(idx)
                pair += 1
        core = gathered.reshape(rank[k], rank[k + 1], n_k).transpose(0, 2, 1)
    mat = core.reshape(rank[k] * n_k, rank[k + 1])
    Q, _R = np.linalg.qr(mat, mode="reduced")
    I, _ = _maxvol(Q)
    new_idx = [np.unravel_index(int(idx), (rank[k], n_k)) for idx in I]
    next_row_idx = [row_idx[k][i0] + (int(i1),) for i0, i1 in new_idx]
    return next_row_idx, fibers


def _right_left_step(
    oracle: PaperGridOracle,
    k: int,
    rank: list[int],
    row_idx: list[list[tuple[int, ...]]],
    col_idx: list[list[tuple[int, ...]]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], np.ndarray]:
    n_km1 = oracle.shape[k - 1]
    fibers: list[tuple[int, ...]] = []
    if k == len(oracle.shape):
        gathered = np.zeros((rank[k - 1], n_km1), dtype=float)
        for i in range(rank[k - 1]):
            left = row_idx[k - 1][i]
            for s in range(n_km1):
                idx = left + (s,)
                fibers.append(idx)
                gathered[i, s] = oracle.query(idx)
        core = gathered.reshape(rank[k - 1], rank[k], n_km1).transpose(0, 2, 1)
    else:
        gathered = np.zeros((rank[k - 1] * rank[k], n_km1), dtype=float)
        pair = 0
        for left in row_idx[k - 1]:
            for right in col_idx[k - 1]:
                for s in range(n_km1):
                    idx = left + (s,) + right
                    fibers.append(idx)
                    gathered[pair, s] = oracle.query(idx)
                pair += 1
        core = gathered.reshape(rank[k - 1], rank[k], n_km1).transpose(0, 2, 1)

    mat = core.reshape(rank[k - 1], n_km1 * rank[k]).T
    Q, _R = np.linalg.qr(mat, mode="reduced")
    J, Q_inv = _maxvol(Q)
    Q_skeleton = Q @ Q_inv
    new_idx = [np.unravel_index(int(idx), (n_km1, rank[k])) for idx in J]
    next_col_idx = [(int(jc0),) + col_idx[k - 1][int(jc1)] for jc0, jc1 in new_idx]
    return next_col_idx, fibers, Q_skeleton


def tensor_train_cross_oracle(
    oracle: PaperGridOracle,
    rank: list[int] | int,
    tol: float = 1e-5,
    n_iter_max: int = 100,
    random_state: int | None = None,
    verbose: bool = False,
    use_tqdm: bool = False,
    log_prefix: str = "[paper-tt-cross]",
) -> tuple[list[np.ndarray], dict[str, Any]]:
    shape = oracle.shape
    d = len(shape)
    rng = np.random.default_rng(random_state)

    rank = _clip_tt_ranks(shape, rank)
    if verbose:
        _paper_log(f"{log_prefix} effective_rank={rank}", use_tqdm=use_tqdm)

    col_idx: list[list[tuple[int, ...]] | None] = [None] * d
    for k in range(d - 1):
        choices: list[tuple[int, ...]] = []
        max_choices = int(np.prod(shape[k + 1:]))
        if rank[k + 1] > max_choices:
            raise ValueError(
                f"Internal TT-rank error after clipping: rank[{k + 1}]={rank[k + 1]} exceeds "
                f"available right-index combinations {max_choices} for shape={shape}"
            )
        while len(choices) < rank[k + 1]:
            item = tuple(int(rng.integers(shape[j])) for j in range(k + 1, d))
            if item not in choices:
                choices.append(item)
        col_idx[k] = choices

    factor_old = [np.zeros((rank[k], shape[k], rank[k + 1]), dtype=float) for k in range(d)]
    factor_new = [rng.random((rank[k], shape[k], rank[k + 1])) for k in range(d)]

    best_factors = factor_old
    converged = False
    truncated = False
    history: list[float] = []
    last_completed_iter = -1

    iterator = range(int(n_iter_max))
    if use_tqdm:
        iterator = tqdm(iterator, desc="TT-cross sweeps", unit="sweep", leave=False)
    for it in iterator:
        factor_old = [np.asarray(c, dtype=float).copy() for c in factor_new]
        factor_new = [None] * d  # type: ignore[list-item]

        try:
            row_idx: list[list[tuple[int, ...]]] = [[()]]
            for k in range(d - 1):
                assert col_idx[k] is not None
                next_row_idx, _fibers = _left_right_step(oracle, k, rank, row_idx, col_idx)  # type: ignore[arg-type]
                row_idx.append(next_row_idx)

            col_idx = [None] * d
            col_idx[-1] = [()]
            for k in range(d, 1, -1):
                assert col_idx[k - 1] is not None
                next_col_idx, _fibers, Q_skeleton = _right_left_step(
                    oracle, k, rank, row_idx, col_idx  # type: ignore[arg-type]
                )
                col_idx[k - 2] = next_col_idx
                factor_new[k - 1] = np.transpose(Q_skeleton).reshape(rank[k - 1], shape[k - 1], rank[k])

            assert col_idx[0] is not None
            core0 = np.zeros((1, shape[0], rank[1]), dtype=float)
            for j in range(rank[1]):
                right = col_idx[0][j]
                for s in range(shape[0]):
                    core0[0, s, j] = oracle.query((s,) + right)
            factor_new[0] = core0

        except QueryBudgetExceeded:
            truncated = True
            if last_completed_iter >= 0:
                best_factors = factor_old
                break
            raise

        dense_old = _tt_to_tensor([np.asarray(c, dtype=float) for c in factor_old])
        dense_new = _tt_to_tensor([np.asarray(c, dtype=float) for c in factor_new])
        err = float(np.linalg.norm(dense_old - dense_new))
        thr = float(tol) * max(float(np.linalg.norm(dense_new)), 1e-12)
        history.append(err)
        best_factors = [np.asarray(c, dtype=float) for c in factor_new]
        last_completed_iter = it
        if verbose:
            _paper_log(
                f"{log_prefix} sweep={it:03d} delta={err:.3e} thr={thr:.3e} "
                f"unique={oracle.unique_queries()} total={oracle.total_queries()}",
                use_tqdm=use_tqdm,
            )
        if err < thr:
            converged = True
            break

    info = {
        "history": history,
        "converged": bool(converged),
        "truncated_by_budget": bool(truncated),
        "iterations_completed": int(max(0, last_completed_iter + 1)),
        "effective_rank": [int(x) for x in rank],
    }
    return best_factors, info


def _cheb_basis_matrix(nodes: np.ndarray, degree: int, a: float, b: float) -> np.ndarray:
    x = (2.0 * np.asarray(nodes, dtype=float) - (a + b)) / (b - a)
    out = np.zeros((len(x), degree), dtype=float)
    out[:, 0] = 1.0
    if degree > 1:
        out[:, 1] = x
    for k in range(2, degree):
        out[:, k] = 2.0 * x * out[:, k - 1] - out[:, k - 2]
    return out


def _cheb_root_angles(nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    x = (2.0 * np.asarray(nodes, dtype=float) - (a + b)) / (b - a)
    x = np.clip(x, -1.0, 1.0)
    return np.arccos(x)


def cheb_transform_matrix(nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    degree = int(len(nodes))
    theta = _cheb_root_angles(nodes, a=float(a), b=float(b))
    k = np.arange(degree, dtype=float).reshape(-1, 1)
    transform = (2.0 / float(degree)) * np.cos(k * theta.reshape(1, -1))
    transform[0, :] *= 0.5
    return np.asarray(transform, dtype=float)


def values_to_chebyshev_coefficients(values: np.ndarray, nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    coeff = np.asarray(values, dtype=float).copy()
    degree = coeff.shape[0]
    transform = cheb_transform_matrix(nodes, a=a, b=b)
    for mode in range(coeff.ndim):
        coeff = np.moveaxis(coeff, mode, 0)
        flat = coeff.reshape(degree, -1)
        flat = transform @ flat
        coeff = flat.reshape(coeff.shape)
        coeff = np.moveaxis(coeff, 0, mode)
    return coeff


def tt_mode_transform(cores: list[np.ndarray], matrices: list[np.ndarray]) -> list[np.ndarray]:
    if len(cores) != len(matrices):
        raise ValueError(f"Need one transform matrix per TT core: {len(cores)} vs {len(matrices)}")
    out: list[np.ndarray] = []
    for core, mat in zip(cores, matrices):
        arr = np.asarray(core, dtype=float)
        M = np.asarray(mat, dtype=float)
        if M.shape[1] != arr.shape[1]:
            raise ValueError(f"Transform width mismatch: {M.shape} vs core mode size {arr.shape[1]}")
        transformed = np.einsum("ij,ajb->aib", M, arr, optimize=True)
        out.append(np.asarray(transformed, dtype=float))
    return out


def cheb_basis_values(y: float, degree: int, a: float, b: float) -> np.ndarray:
    t = (2.0 * float(y) - (a + b)) / (b - a)
    out = np.zeros(degree, dtype=float)
    out[0] = 1.0
    if degree > 1:
        out[1] = t
    for k in range(2, degree):
        out[k] = 2.0 * t * out[k - 1] - out[k - 2]
    return out


def evaluate_chebyshev_tensor(coeff_tensor: np.ndarray, point: np.ndarray, a: float, b: float) -> float:
    arr = np.asarray(coeff_tensor, dtype=float)
    y = np.asarray(point, dtype=float).reshape(-1)
    if y.shape != (arr.ndim,):
        raise ValueError(f"Point shape mismatch: {y.shape} vs {(arr.ndim,)}")
    out = arr
    for mode in range(arr.ndim):
        vals = cheb_basis_values(float(y[mode]), arr.shape[mode], a=a, b=b)
        out = np.tensordot(vals, out, axes=(0, 0))
    return float(out)


def evaluate_tt_chebyshev(coeff_tt_cores: list[np.ndarray], point: np.ndarray, a: float, b: float) -> float:
    y = np.asarray(point, dtype=float).reshape(-1)
    if len(coeff_tt_cores) != y.size:
        raise ValueError(f"Point shape mismatch: {y.shape} vs tt ndim {len(coeff_tt_cores)}")
    state = np.ones((1,), dtype=float)
    for mode, core in enumerate(coeff_tt_cores):
        basis = cheb_basis_values(float(y[mode]), core.shape[1], a=a, b=b)
        mat = np.tensordot(basis, np.asarray(core, dtype=float), axes=(0, 1))
        state = state @ mat
    return float(state.reshape(-1)[0])


def sample_random_rms(
    *,
    context: PlanarWaterContext,
    energy_fn,
    coeff_tensor: np.ndarray,
    coeff_tt_cores: list[np.ndarray] | None,
    a: float,
    b: float,
    n_samples: int,
    seed: int,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> float:
    if int(n_samples) <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    errs = np.zeros(int(n_samples), dtype=float)
    iterator = range(int(n_samples))
    if use_tqdm and int(n_samples) > 32:
        iterator = tqdm(iterator, desc="RMS probes", unit="pt", leave=False)
    if verbose:
        _paper_log(f"[paper-rms] START probe_points={n_samples}", use_tqdm=use_tqdm)
    for i in iterator:
        y = rng.uniform(low=float(a), high=float(b), size=context.as_dim)
        coords = _coords_from_reduced(context, y)
        exact = float(energy_fn(list(context.symbols), coords))
        if coeff_tt_cores is not None:
            approx = float(evaluate_tt_chebyshev(coeff_tt_cores, y, a=a, b=b))
        else:
            approx = float(evaluate_chebyshev_tensor(coeff_tensor, y, a=a, b=b))
        errs[i] = exact - approx
    rms = float(np.sqrt(np.mean(errs ** 2)) * water.HARTREE_TO_MEV)
    if verbose:
        _paper_log(f"[paper-rms] DONE rms={rms:.6f} meV", use_tqdm=use_tqdm)
    return rms


def _paper_value_tensor_cache_path(
    data_root: Path,
    *,
    context: PlanarWaterContext,
    n_points: int,
) -> Path:
    return (
        Path(data_root)
        / "paper_tensor_cache"
        / f"{context.as_cache_tag}_paper_cheb{int(n_points)}_d{context.as_dim}.npz"
    )


def load_or_generate_full_value_tensor(
    args: Any,
    *,
    context: PlanarWaterContext,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    a, b = float(getattr(args, "cheb_interval")[0]), float(getattr(args, "cheb_interval")[1])
    nodes = chebyshev_nodes(n_points, a=a, b=b)
    cache_path = _paper_value_tensor_cache_path(getattr(args, "data_root"), context=context, n_points=n_points)
    cache_lookup = water.resolve_existing_project_path(cache_path)
    shape = tuple([int(n_points)] * context.as_dim)
    use_tqdm = bool(getattr(args, "show_tqdm", False))
    verbose = bool(getattr(args, "verbose", False))
    if bool(getattr(args, "cache_tensors", True)) and cache_lookup.exists():
        payload = np.load(cache_lookup)
        tensor_h = np.asarray(payload["tensor_h"], dtype=float)
        cached_nodes = np.asarray(payload["nodes"], dtype=float)
        if tensor_h.shape != shape:
            raise ValueError(f"Cached tensor shape mismatch at {cache_lookup}: {tensor_h.shape} vs {shape}")
        if cached_nodes.shape != nodes.shape or np.max(np.abs(cached_nodes - nodes)) > 1e-12:
            raise ValueError(f"Cached nodes mismatch at {cache_lookup}")
        if verbose:
            _paper_log(f"[paper-tensor] using cached tensor -> {cache_lookup}", use_tqdm=use_tqdm)
        return tensor_h, nodes

    if not bool(getattr(args, "generate_if_missing", True)):
        raise FileNotFoundError(
            f"Missing cached paper-like tensor: {cache_path}\n"
            "Set generate_if_missing=True to generate it."
        )

    energy_fn = water._build_energy_fn(args)
    tensor_h = np.zeros(shape, dtype=float)
    total = int(np.prod(shape))
    progress = tqdm(total=total, desc=f"Paper tensor (n={n_points}, d={context.as_dim})", unit="pt", leave=False) if use_tqdm else None
    if verbose:
        _paper_log(f"[paper-tensor] START generating shape={shape}", use_tqdm=use_tqdm)
    for idx in np.ndindex(shape):
        reduced_point = np.asarray([nodes[i] for i in idx], dtype=float)
        coords = _coords_from_reduced(context, reduced_point)
        tensor_h[idx] = float(energy_fn(list(context.symbols), coords))
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()

    if bool(getattr(args, "cache_tensors", True)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, tensor_h=tensor_h, nodes=nodes)
        if verbose:
            _paper_log(f"[paper-tensor] saved tensor -> {cache_path}", use_tqdm=use_tqdm)
    if verbose:
        _paper_log(f"[paper-tensor] DONE generated shape={shape}", use_tqdm=use_tqdm)
    return tensor_h, nodes


def run_paper_as_tt(
    args: Any,
    *,
    context: PlanarWaterContext,
    n_points: int,
    rank_cap: int,
    tol: float,
    n_iter_max: int,
    random_state: int,
    max_total_queries: int | None = None,
    max_unique_queries: int | None = None,
    rms_probe_points: int = 1000,
    rms_seed: int = 0,
) -> PaperASTTResult:
    a, b = float(getattr(args, "cheb_interval")[0]), float(getattr(args, "cheb_interval")[1])
    nodes = chebyshev_nodes(n_points, a=a, b=b)
    energy_fn = water._build_energy_fn(args)
    use_tqdm = bool(getattr(args, "show_tqdm", False))
    verbose = bool(getattr(args, "verbose", False))
    oracle = PaperGridOracle(
        context=context,
        nodes=nodes,
        energy_fn=energy_fn,
        max_total_queries=max_total_queries,
        max_unique_queries=max_unique_queries,
    )

    requested_rank = [1] + [int(rank_cap)] * (context.as_dim - 1) + [1]
    effective_rank = _clip_tt_ranks(tuple([int(n_points)] * context.as_dim), requested_rank)
    if verbose:
        _paper_log(
            f"[paper-as-tt] START n={n_points} rank_cap={rank_cap} tol={tol:.1e} "
            f"budget_unique={max_unique_queries} budget_total={max_total_queries} "
            f"effective_rank={effective_rank}",
            use_tqdm=use_tqdm,
        )
    try:
        tt_cores, cross_info = tensor_train_cross_oracle(
            oracle,
            rank=effective_rank,
            tol=float(tol),
            n_iter_max=int(n_iter_max),
            random_state=int(random_state),
            verbose=verbose,
            use_tqdm=use_tqdm,
            log_prefix=f"[paper-tt-cross n={n_points}]",
        )
        approx_tensor = _tt_to_tensor(tt_cores)
        converged = bool(cross_info.get("converged", False))
        truncated = bool(cross_info.get("truncated_by_budget", False))
    except QueryBudgetExceeded as exc:
        if verbose:
            _paper_log(f"[paper-as-tt] FAIL budget exceeded before first sweep: {exc}", use_tqdm=use_tqdm)
        return PaperASTTResult(
            tensor=None,
            coeff_tensor=None,
            coeff_tt_cores=None,
            tt_cores=None,
            queried_flat_indices=oracle.queried_flat_indices(),
            queried_index_sequence=[list(idx) for idx in oracle.query_sequence],
            total_query_count=oracle.total_queries(),
            unique_query_count=oracle.unique_queries(),
            converged=False,
            truncated_by_budget=True,
            max_rank=None,
            rms_random=None,
            info={"error": str(exc), "status": "budget_exceeded_before_first_sweep"},
        )

    transform = cheb_transform_matrix(nodes, a=a, b=b)
    coeff_tt_cores = tt_mode_transform(tt_cores, [transform for _ in range(context.as_dim)])
    coeff_tensor = values_to_chebyshev_coefficients(approx_tensor, nodes, a=a, b=b)
    value_ranks = _tt_ranks_from_cores(tt_cores)
    coeff_ranks = _tt_ranks_from_cores(coeff_tt_cores)
    rms_random = sample_random_rms(
        context=context,
        energy_fn=energy_fn,
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=coeff_tt_cores,
        a=a,
        b=b,
        n_samples=int(rms_probe_points),
        seed=int(rms_seed),
        verbose=verbose,
        use_tqdm=use_tqdm,
    )
    if verbose:
        _paper_log(
            f"[paper-as-tt] DONE n={n_points} rms={rms_random:.6f} meV "
            f"unique={oracle.unique_queries()} total={oracle.total_queries()} "
            f"converged={converged} truncated={truncated}",
            use_tqdm=use_tqdm,
        )
    return PaperASTTResult(
        tensor=approx_tensor,
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=coeff_tt_cores,
        tt_cores=tt_cores,
        queried_flat_indices=oracle.queried_flat_indices(),
        queried_index_sequence=[list(idx) for idx in oracle.query_sequence],
        total_query_count=oracle.total_queries(),
        unique_query_count=oracle.unique_queries(),
        converged=converged,
        truncated_by_budget=truncated,
        max_rank=max(coeff_ranks) if coeff_ranks else 1,
        rms_random=rms_random,
        info={
            **cross_info,
            "rank_cap": int(rank_cap),
            "requested_rank": [int(x) for x in requested_rank],
            "effective_rank": [int(x) for x in effective_rank],
            "value_tt_ranks": value_ranks,
            "coefficient_tt_ranks": coeff_ranks,
            "paper_total_queries": oracle.total_queries(),
            "paper_unique_queries": oracle.unique_queries(),
        },
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "baranov2015_as_tt_experiments" / "data" / "as_tt_water"
DEFAULT_GEOMETRY = PROJECT_ROOT / "as_tt_water_experiments" / "data" / "as_tt_water" / "water.xyz"
NON_FAITHFUL_FALLBACK = "NON_FAITHFUL_FALLBACK"
TTML_EXTERNAL_BACKEND = "TTML_EXTERNAL_BACKEND"


@dataclass
class PaperWaterContext:
    symbols: tuple[str, ...]
    base_coords: np.ndarray
    coord_unit: str
    raw_active_dofs: tuple[tuple[int, int], ...]
    raw_dim: int
    as_dim: int
    as_basis: np.ndarray
    as_eigenvalues: np.ndarray
    cheb_nodes: np.ndarray | None = None
    cheb_interval: tuple[float, float] | None = None
    as_cache_path: str | None = None
    as_cache_tag: str = "baranov2015_water_planar"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleTrace:
    unique_indices: np.ndarray
    unique_values: np.ndarray
    query_sequence: np.ndarray
    total_queries: int
    unique_queries: int
    shape: tuple[int, ...]
    nodes: np.ndarray
    context: PaperWaterContext
    status: str
    tt_backend_requested: str
    tt_backend_used: str
    faithful_backend: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorsBaselineResult:
    context: PaperWaterContext
    sample_trace: SampleTrace
    tensor: np.ndarray | None
    coeff_tensor: np.ndarray | None
    coeff_tt_cores: list[np.ndarray] | None
    tt_cores: list[np.ndarray] | None
    rms_random_mev: float | None
    test_metrics: dict[str, float]
    tt_ranks: list[int]
    storage: int
    total_queries: int
    unique_queries: int
    backend_requested: str
    backend_used: str
    faithful_backend: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    completed_tensor: np.ndarray | None
    factors: list[np.ndarray] | None
    history: list[Any]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    coeff_tensor: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetSweepRow:
    budget: int
    effective_budget: int
    status: str
    completion_result: CompletionResult
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetSweepResult:
    budgets: list[int]
    rows: list[BudgetSweepRow]
    sample_trace: SampleTrace
    authors_baseline: AuthorsBaselineResult | None
    test_points: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


class BudgetExceeded(QueryBudgetExceeded):
    pass


def _default_args_dict() -> dict[str, Any]:
    return {
        "data_root": str(DEFAULT_DATA_ROOT),
        "geometry_xyz": str(DEFAULT_GEOMETRY),
        "cache_tensors": True,
        "generate_if_missing": True,
        "qc_method": "RHF",
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 0,
        "scf_conv_tol": 1e-9,
        "scf_max_cycle": 50,
        "scf_init_guess": "hcore",
        "scf_verbose": 0,
        "scf_newton_fallback": True,
        "water_coordinate_unit": "Bohr",
        "water_as_dim": 4,
        "water_as_sigma2": 0.1,
        "water_as_samples": 256,
        "water_as_random_state": 1729,
        "cheb_interval": (-0.3, 0.3),
        "author_rank_cap": 12,
        "author_tol": 1e-5,
        "author_max_iter": 30,
        "author_random_state": 1729,
        "author_log_first_unique": 8,
        "author_log_every_unique": 25,
        "author_log_every_total": 250,
        "author_rms_probe_points": 1000,
        "author_rms_seed": 2025,
        "show_tqdm": False,
        "verbose": False,
        "tt_backend": "octave_tt_toolbox",
        "ttml_env_name": "matrix_approximation_final_3_11",
        "ttml_method": "dmrg",
        "octave_env_name": "octave",
        "tt_toolbox_root": str(PROJECT_ROOT / "external" / "TT-Toolbox"),
    }


def _coerce_args(args: Any | None = None, **overrides: Any) -> SimpleNamespace:
    base = _default_args_dict()
    if args is None:
        payload: dict[str, Any] = {}
    elif isinstance(args, SimpleNamespace):
        payload = vars(args)
    elif isinstance(args, argparse.Namespace):
        payload = vars(args)
    elif isinstance(args, dict):
        payload = dict(args)
    else:
        payload = dict(vars(args))
    base.update(payload)
    base.update(overrides)
    interval = base.get("cheb_interval", (-0.3, 0.3))
    base["cheb_interval"] = (float(interval[0]), float(interval[1]))
    base["data_root"] = str(water.resolve_project_path(base["data_root"]))
    base["geometry_xyz"] = str(water.resolve_project_path(base["geometry_xyz"]))
    return SimpleNamespace(**base)


def _paper_context_from_planar(context: PlanarWaterContext, *, cheb_nodes_: np.ndarray | None = None, cheb_interval: tuple[float, float] | None = None, metadata: dict[str, Any] | None = None) -> PaperWaterContext:
    return PaperWaterContext(
        symbols=tuple(context.symbols),
        base_coords=np.asarray(context.base_coords, dtype=float).copy(),
        coord_unit=str(context.coord_unit),
        raw_active_dofs=tuple(context.raw_active_dofs),
        raw_dim=int(context.raw_dim),
        as_dim=int(context.as_dim),
        as_basis=np.asarray(context.as_basis, dtype=float).copy(),
        as_eigenvalues=np.asarray(context.as_eigenvalues, dtype=float).copy(),
        cheb_nodes=None if cheb_nodes_ is None else np.asarray(cheb_nodes_, dtype=float).copy(),
        cheb_interval=cheb_interval,
        as_cache_path=context.as_cache_path,
        as_cache_tag=str(context.as_cache_tag),
        metadata={} if metadata is None else dict(metadata),
    )


def _planar_context_from_paper(context: PaperWaterContext) -> PlanarWaterContext:
    return PlanarWaterContext(
        symbols=tuple(context.symbols),
        base_coords=np.asarray(context.base_coords, dtype=float).copy(),
        coord_unit=str(context.coord_unit),
        raw_dim=int(context.raw_dim),
        as_dim=int(context.as_dim),
        raw_active_dofs=tuple(context.raw_active_dofs),
        as_basis=np.asarray(context.as_basis, dtype=float).copy(),
        as_eigenvalues=np.asarray(context.as_eigenvalues, dtype=float).copy(),
        as_cache_path=context.as_cache_path,
        as_cache_tag=str(context.as_cache_tag),
    )


def build_paper_water_context(args: Any | None = None) -> PaperWaterContext:
    args_ns = _coerce_args(args)
    context = prepare_planar_identity_context(args_ns)
    interval = tuple(float(v) for v in getattr(args_ns, "cheb_interval"))
    n_points = getattr(args_ns, "n_points", None)
    nodes = None if n_points is None else build_chebyshev_nodes(int(n_points), interval[0], interval[1])
    return _paper_context_from_planar(
        context,
        cheb_nodes_=nodes,
        cheb_interval=interval,
        metadata={
            "sigma2": float(getattr(args_ns, "water_as_sigma2")),
            "as_samples": int(getattr(args_ns, "water_as_samples")),
            "qc_method": str(getattr(args_ns, "qc_method")),
            "basis": str(getattr(args_ns, "basis")),
        },
    )


def compute_active_subspace_paper(args: Any | None, context: PaperWaterContext) -> PaperWaterContext:
    args_ns = _coerce_args(args)
    rotated = prepare_planar_water_context(args_ns)
    return _paper_context_from_planar(
        rotated,
        cheb_nodes_=context.cheb_nodes,
        cheb_interval=context.cheb_interval,
        metadata=dict(context.metadata),
    )


def build_chebyshev_nodes(n_points: int, a: float, b: float) -> np.ndarray:
    return chebyshev_nodes(int(n_points), float(a), float(b))


def multi_index_to_reduced_point(idx: tuple[int, ...] | list[int] | np.ndarray, nodes: np.ndarray) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=int).reshape(-1)
    node_arr = np.asarray(nodes, dtype=float)
    return node_arr[idx_arr]


def reduced_point_to_cartesian(context: PaperWaterContext, y: np.ndarray) -> np.ndarray:
    planar = _planar_context_from_paper(context)
    return _coords_from_reduced(planar, np.asarray(y, dtype=float))


def make_exact_pes_entry_oracle(context: PaperWaterContext, nodes: np.ndarray, energy_fn):
    node_arr = np.asarray(nodes, dtype=float)

    def eval_entry(idx: tuple[int, ...]) -> float:
        y = multi_index_to_reduced_point(idx, node_arr)
        coords = reduced_point_to_cartesian(context, y)
        return float(energy_fn(list(context.symbols), coords))

    return eval_entry


class LoggedBudgetedOracle:
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        eval_entry,
        max_unique_queries: int | None = None,
        verbose: bool = False,
        use_tqdm: bool = False,
        log_prefix: str = "[authors-oracle]",
        log_first_unique: int = 8,
        log_every_unique: int = 25,
        log_every_total: int = 250,
    ) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.eval_entry = eval_entry
        self.max_unique_queries = None if max_unique_queries is None else int(max_unique_queries)
        self.verbose = bool(verbose)
        self.use_tqdm = bool(use_tqdm)
        self.log_prefix = str(log_prefix)
        self.log_first_unique = max(0, int(log_first_unique))
        self.log_every_unique = max(0, int(log_every_unique))
        self.log_every_total = max(0, int(log_every_total))
        self.cache: dict[tuple[int, ...], float] = {}
        self.query_sequence: list[tuple[int, ...]] = []
        self.unique_indices: list[tuple[int, ...]] = []
        self.unique_values: list[float] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            _paper_log(msg, use_tqdm=self.use_tqdm)

    def _should_log_unique(self) -> bool:
        uq = self.unique_queries()
        if uq <= self.log_first_unique:
            return True
        if self.log_every_unique > 0 and uq % self.log_every_unique == 0:
            return True
        if self.max_unique_queries is not None and uq >= self.max_unique_queries:
            return True
        return False

    def _should_log_total(self) -> bool:
        tq = self.total_queries()
        return self.log_every_total > 0 and tq > 0 and tq % self.log_every_total == 0

    def query(self, idx: tuple[int, ...]) -> float:
        idx = tuple(int(i) for i in idx)
        if len(idx) != len(self.shape):
            raise ValueError(f"Index dimensionality mismatch: {idx} vs ndim={len(self.shape)}")
        self.query_sequence.append(idx)
        if idx in self.cache:
            if self._should_log_total():
                self._log(
                    f"{self.log_prefix} total={self.total_queries()} unique={self.unique_queries()} "
                    f"cache_hits={self.total_queries() - self.unique_queries()}"
                )
            return self.cache[idx]
        if self.max_unique_queries is not None and len(self.cache) >= self.max_unique_queries:
            self._log(
                f"{self.log_prefix} budget_exhausted unique={self.unique_queries()} "
                f"total={self.total_queries()} next_idx={list(idx)}"
            )
            raise BudgetExceeded(
                f"Unique query budget exceeded: {len(self.cache)} >= {self.max_unique_queries}"
            )
        value = float(self.eval_entry(idx))
        self.cache[idx] = value
        self.unique_indices.append(idx)
        self.unique_values.append(value)
        if self._should_log_unique():
            budget_label = "inf" if self.max_unique_queries is None else str(self.max_unique_queries)
            self._log(
                f"{self.log_prefix} unique={self.unique_queries()}/{budget_label} "
                f"total={self.total_queries()} last_idx={list(idx)} value={value:.8f}"
            )
        elif self._should_log_total():
            self._log(
                f"{self.log_prefix} total={self.total_queries()} unique={self.unique_queries()} "
                f"cache_hits={self.total_queries() - self.unique_queries()}"
            )
        return value

    def export_unique_samples(self) -> tuple[np.ndarray, np.ndarray]:
        if self.unique_indices:
            indices = np.asarray(self.unique_indices, dtype=int)
        else:
            indices = np.zeros((0, len(self.shape)), dtype=int)
        values = np.asarray(self.unique_values, dtype=float)
        return indices, values

    def queried_flat_indices(self) -> list[int]:
        return [int(np.ravel_multi_index(idx, self.shape)) for idx in self.unique_indices]

    def total_queries(self) -> int:
        return int(len(self.query_sequence))

    def unique_queries(self) -> int:
        return int(len(self.unique_indices))

    def get_stats(self) -> dict[str, int]:
        return {
            "total_queries": self.total_queries(),
            "unique_queries": self.unique_queries(),
        }


def _storage_from_tt_cores(tt_cores: list[np.ndarray] | None) -> int:
    if not tt_cores:
        return 0
    return int(sum(int(np.prod(core.shape)) for core in tt_cores))


def _storage_from_tt_ranks(shape: tuple[int, ...], tt_ranks: list[int] | tuple[int, ...] | np.ndarray | None) -> int:
    if tt_ranks is None:
        return 0
    ranks = [int(x) for x in np.asarray(tt_ranks, dtype=int).reshape(-1).tolist()]
    dims = [int(x) for x in tuple(shape)]
    if len(ranks) != len(dims) + 1:
        return 0
    return int(sum(int(ranks[k]) * int(dims[k]) * int(ranks[k + 1]) for k in range(len(dims))))


def _oracle_to_trace(
    oracle: LoggedBudgetedOracle,
    *,
    context: PaperWaterContext,
    nodes: np.ndarray,
    status: str,
    tt_backend_requested: str,
    tt_backend_used: str,
    faithful_backend: bool,
    info: dict[str, Any] | None = None,
) -> SampleTrace:
    unique_indices, unique_values = oracle.export_unique_samples()
    if oracle.query_sequence:
        query_sequence = np.asarray(oracle.query_sequence, dtype=int)
    else:
        query_sequence = np.zeros((0, len(oracle.shape)), dtype=int)
    return SampleTrace(
        unique_indices=unique_indices,
        unique_values=unique_values,
        query_sequence=query_sequence,
        total_queries=oracle.total_queries(),
        unique_queries=oracle.unique_queries(),
        shape=tuple(oracle.shape),
        nodes=np.asarray(nodes, dtype=float).copy(),
        context=context,
        status=str(status),
        tt_backend_requested=str(tt_backend_requested),
        tt_backend_used=str(tt_backend_used),
        faithful_backend=bool(faithful_backend),
        info={} if info is None else dict(info),
    )


def _candidate_ttml_site_packages(ttml_env_name: str | None) -> list[Path]:
    candidates: list[Path] = []
    if ttml_env_name:
        env_root = Path.home() / "anaconda3" / "envs" / str(ttml_env_name)
        if env_root.exists():
            candidates.extend(sorted((env_root / "lib").glob("python*/site-packages")))
    return [path for path in candidates if path.exists()]


def _import_ttml_modules(ttml_env_name: str | None):
    try:
        tt_cross = importlib.import_module("ttml.tt_cross")
        tensor_train = importlib.import_module("ttml.tensor_train")
        return tt_cross, tensor_train, None
    except Exception as first_exc:
        for site_packages in _candidate_ttml_site_packages(ttml_env_name):
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.append(site_packages_str)
            try:
                tt_cross = importlib.import_module("ttml.tt_cross")
                tensor_train = importlib.import_module("ttml.tensor_train")
                return tt_cross, tensor_train, site_packages_str
            except Exception:
                continue
        raise RuntimeError(
            "Unable to import `ttml`. "
            "Run the pipeline inside the conda env where `ttml` is installed "
            f"(recommended: `matrix_approximation_final_3_11`) or set `ttml_env_name`. "
            f"Original import error: {first_exc}"
        ) from first_exc


def _make_ttml_index_function(value_callback, ndim: int):
    ndim = int(ndim)

    def index_fun(inds) -> np.ndarray:
        arr = np.asarray(inds, dtype=int)
        if arr.shape[-1] != ndim:
            raise ValueError(f"TTML index array mismatch: {arr.shape} vs ndim={ndim}")
        flat = arr.reshape(-1, ndim)
        out = np.empty(flat.shape[0], dtype=float)
        for row_idx, row in enumerate(flat):
            out[row_idx] = float(value_callback(tuple(int(x) for x in row)))
        return out.reshape(arr.shape[:-1])

    return index_fun


def _run_ttml_cross(
    *,
    shape: tuple[int, ...],
    value_callback,
    tol: float,
    random_state: int | None,
    rank_cap: int,
    n_iter_max: int,
    oracle: LoggedBudgetedOracle | None,
    ttml_env_name: str | None,
    ttml_method: str,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> dict[str, Any]:
    tt_cross, tensor_train, imported_site_packages = _import_ttml_modules(ttml_env_name)
    TensorTrain = tensor_train.TensorTrain
    tracked_oracle = oracle or LoggedBudgetedOracle(shape=shape, eval_entry=value_callback, max_unique_queries=None)
    if random_state is not None:
        np.random.seed(int(random_state))
    tt = TensorTrain.random(tuple(int(x) for x in shape), int(rank_cap), mode="r")
    index_fun = _make_ttml_index_function(tracked_oracle.query, ndim=len(shape))
    method = str(ttml_method).lower()
    if verbose:
        _paper_log(
            f"[ttml-cross] START method={method} shape={shape} rank_cap={rank_cap} "
            f"tol={float(tol):.1e} max_its={int(n_iter_max)}",
            use_tqdm=use_tqdm,
        )
    stopped_by_budget = False
    if method == "dmrg":
        try:
            tt = tt_cross.tt_cross_dmrg(
                tt,
                index_fun,
                tol=float(tol),
                max_its=int(n_iter_max),
                verbose=bool(verbose),
                inplace=True,
            )
        except BudgetExceeded:
            stopped_by_budget = True
        backend_used = "ttml_dmrg"
    elif method == "regular":
        try:
            tt = tt_cross.tt_cross_regular(
                tt,
                index_fun,
                tol=float(tol),
                max_its=int(n_iter_max),
                verbose=bool(verbose),
                inplace=True,
            )
        except BudgetExceeded:
            stopped_by_budget = True
        backend_used = "ttml_regular"
    else:
        raise ValueError(f"Unsupported ttml_method={ttml_method!r}; expected 'dmrg' or 'regular'")

    tt_cores = [np.asarray(core, dtype=float).copy() for core in tt.cores]
    dense_tensor = np.asarray(tt.dense(), dtype=float)
    tt_ranks = [1] + [int(x) for x in getattr(tt, "tt_rank", ())] + [1]
    errors = getattr(tt, "errors", None)
    info = {
        "backend_note": TTML_EXTERNAL_BACKEND,
        "ttml_method": method,
        "ttml_env_name": ttml_env_name,
        "ttml_path": getattr(tt_cross, "__file__", None),
        "ttml_site_packages": imported_site_packages,
        "value_tt_ranks": tt_ranks,
        "storage": int(tt.num_params()),
        "stopped_by_budget": bool(stopped_by_budget),
    }
    if errors is not None:
        info["errors"] = np.asarray(errors, dtype=float).tolist()
    if verbose:
        _paper_log(
            f"[ttml-cross] DONE method={method} unique={tracked_oracle.unique_queries()} "
            f"total={tracked_oracle.total_queries()} ranks={tt_ranks} storage={int(tt.num_params())} "
            f"stopped_by_budget={stopped_by_budget}",
            use_tqdm=use_tqdm,
        )
    return {
        "tt_cores": tt_cores,
        "dense_tensor": dense_tensor,
        "backend_used": backend_used,
        "faithful_backend": False,
        "info": info,
        "oracle": tracked_oracle,
        "stopped_by_budget": bool(stopped_by_budget),
    }


def _find_octave_executable(octave_env_name: str | None) -> Path | None:
    candidates: list[Path] = []
    if octave_env_name:
        candidates.append(Path.home() / "anaconda3" / "envs" / str(octave_env_name) / "bin" / "octave")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@contextlib.contextmanager
def _serve_oracle_bridge(oracle: LoggedBudgetedOracle, *, verbose: bool = False, use_tqdm: bool = False):
    stop_event = threading.Event()
    with tempfile.TemporaryDirectory(prefix="faithful_octave_bridge_") as tmpdir:
        bridge_dir = Path(tmpdir)
        request_dir = bridge_dir / "requests"
        response_dir = bridge_dir / "responses"
        request_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)

        def _worker() -> None:
            while not stop_event.is_set():
                request_paths = sorted(request_dir.glob("*.json"))
                if not request_paths:
                    stop_event.wait(0.01)
                    continue
                for request_path in request_paths:
                    processing_path = request_path.with_suffix(".processing")
                    try:
                        request_path.rename(processing_path)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        continue
                    response_payload: dict[str, Any]
                    try:
                        payload = json.loads(processing_path.read_text(encoding="utf-8"))
                        indices = np.asarray(payload.get("indices", []), dtype=int)
                        if indices.ndim == 1 and indices.size > 0:
                            indices = indices.reshape(1, -1)
                        if indices.size > 0 and oracle.max_unique_queries is not None:
                            remaining = int(oracle.max_unique_queries) - int(len(oracle.cache))
                            batch_new_unique: set[tuple[int, ...]] = set()
                            for row in indices:
                                idx = tuple(int(v) - 1 for v in np.asarray(row, dtype=int).reshape(-1))
                                if idx not in oracle.cache:
                                    batch_new_unique.add(idx)
                            if len(batch_new_unique) > remaining:
                                raise BudgetExceeded(
                                    "Unique query budget exceeded before starting TT-Toolbox block: "
                                    f"need {len(batch_new_unique)} new points, remaining {remaining}"
                                )
                        values: list[float] = []
                        for row in indices:
                            idx = tuple(int(v) - 1 for v in np.asarray(row, dtype=int).reshape(-1))
                            values.append(float(oracle.query(idx)))
                        response_payload = {"budget_exceeded": False, "values": values}
                    except BudgetExceeded as exc:
                        response_payload = {"budget_exceeded": True, "message": str(exc)}
                    except Exception as exc:  # pragma: no cover
                        response_payload = {"budget_exceeded": False, "error": str(exc)}
                    response_path = response_dir / f"{processing_path.stem}.json"
                    response_path.write_text(json.dumps(response_payload), encoding="utf-8")
                    with contextlib.suppress(FileNotFoundError):
                        processing_path.unlink()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            yield bridge_dir
        finally:
            stop_event.set()
            thread.join(timeout=5.0)


def _run_octave_tt_toolbox_cross(*, shape: tuple[int, ...], value_callback, tol: float, random_state: int | None, rank_cap: int, n_iter_max: int, oracle: LoggedBudgetedOracle | None, tt_toolbox_root: str | None, octave_env_name: str | None = None, verbose: bool = False, use_tqdm: bool = False) -> dict[str, Any]:
    root = Path(tt_toolbox_root) if tt_toolbox_root is not None else (PROJECT_ROOT / "external" / "TT-Toolbox")
    setup_m = root / "setup.m"
    if not setup_m.exists():
        raise RuntimeError(
            "TT-Toolbox checkout is unavailable or incomplete. "
            f"Expected `{setup_m}` for the faithful Octave backend."
        )
    octave_bin = _find_octave_executable(octave_env_name)
    tracked_oracle = oracle or LoggedBudgetedOracle(shape=shape, eval_entry=value_callback, max_unique_queries=None)
    bridge_dir = PROJECT_ROOT / "baranov2015_as_tt_experiments" / "octave"
    runner_m = bridge_dir / "faithful_run_dmrg_cross_bridge.m"
    eval_m = bridge_dir / "faithful_octave_http_eval.m"
    budgeted_m = bridge_dir / "faithful_dmrg_cross_budgeted.m"
    if not runner_m.exists() or not eval_m.exists() or not budgeted_m.exists():
        raise RuntimeError(
            "Octave TT-Toolbox bridge files are missing. "
            f"Expected `{runner_m}`, `{eval_m}`, and `{budgeted_m}`."
        )
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to load Octave TT-Toolbox outputs") from exc

    if verbose:
        _paper_log(
            f"[octave-tt-toolbox] START shape={shape} rank_cap={rank_cap} n_iter_max={n_iter_max} tol={tol:.1e}",
            use_tqdm=use_tqdm,
        )

    with _serve_oracle_bridge(tracked_oracle, verbose=verbose, use_tqdm=use_tqdm) as oracle_bridge_dir:
        with tempfile.TemporaryDirectory(prefix="faithful_tttoolbox_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            cfg_path = tmpdir_path / "config.json"
            out_mat = tmpdir_path / "result.mat"
            cfg_payload = {
                "shape": [int(x) for x in shape],
                "tol": float(tol),
                "rank_cap": int(rank_cap),
                "n_iter_max": int(n_iter_max),
                "random_state": None if random_state is None else int(random_state),
                "verbose": bool(verbose),
                "oracle_bridge_dir": str(oracle_bridge_dir),
                "tt_toolbox_root": str(root),
                "bridge_dir": str(bridge_dir),
                "output_mat": str(out_mat),
            }
            cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
            bridge_dir_quoted = str(bridge_dir).replace("'", "''")
            cfg_path_quoted = str(cfg_path).replace("'", "''")
            octave_eval = f"addpath('{bridge_dir_quoted}'); faithful_run_dmrg_cross_bridge('{cfg_path_quoted}');"
            if octave_env_name:
                conda_sh = Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh"
                cmd = [
                    "bash",
                    "-lc",
                    (
                        f"source {shlex.quote(str(conda_sh))} && "
                        f"conda activate {shlex.quote(str(octave_env_name))} && "
                        f"octave --quiet --eval {shlex.quote(octave_eval)}"
                    ),
                ]
            else:
                if octave_bin is None:
                    raise RuntimeError(
                        "Unable to find Octave executable for TT-Toolbox backend. "
                        "Provide `octave_env_name` or install Octave on PATH."
                    )
                cmd = [str(octave_bin), "--quiet", "--eval", octave_eval]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                check=False,
            )
            octave_output = proc.stdout or ""
            if proc.returncode != 0:
                raise RuntimeError(
                    "Octave TT-Toolbox run failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{octave_output}"
                )
            if not out_mat.exists():
                raise RuntimeError(
                    "Octave TT-Toolbox finished without producing output MAT file.\n"
                    f"Output:\n{octave_output}"
                )
            payload = loadmat(out_mat)

    dense_tensor = np.asarray(payload["dense_tensor"], dtype=float)
    dense_tensor = dense_tensor.reshape(shape, order="F")
    stopped_by_budget = bool(np.asarray(payload.get("stopped_by_budget", [[0]])).reshape(-1)[0])
    converged = bool(np.asarray(payload.get("converged", [[0]])).reshape(-1)[0])
    returned_checkpoint = bool(np.asarray(payload.get("returned_checkpoint", [[0]])).reshape(-1)[0])
    tt_ranks_octave = [int(x) for x in np.asarray(payload.get("tt_ranks", [[1]])).reshape(-1).tolist()]
    sweeps_completed = int(np.asarray(payload.get("sweeps_completed", [[0]])).reshape(-1)[0])
    last_block = int(np.asarray(payload.get("last_block", [[0]])).reshape(-1)[0])
    budget_message_arr = np.asarray(payload.get("budget_message", [[""]]), dtype=object).reshape(-1)
    budget_message = "" if budget_message_arr.size == 0 else str(budget_message_arr[0])
    storage = _storage_from_tt_ranks(shape, tt_ranks_octave)
    info = {
        "octave_env_name": octave_env_name,
        "octave_bin": None if octave_bin is None else str(octave_bin),
        "octave_output": octave_output,
        "tt_toolbox_root": str(root),
        "value_tt_ranks_octave": tt_ranks_octave,
        "value_tt_ranks": tt_ranks_octave,
        "storage": int(storage),
        "stopped_by_budget": bool(stopped_by_budget),
        "converged": bool(converged),
        "returned_checkpoint": bool(returned_checkpoint),
        "sweeps_completed": int(sweeps_completed),
        "last_block": int(last_block),
        "budget_message": budget_message,
    }
    if verbose:
        _paper_log(
            f"[octave-tt-toolbox] DONE unique={tracked_oracle.unique_queries()} total={tracked_oracle.total_queries()} "
            f"stopped_by_budget={stopped_by_budget} converged={converged} sweeps_completed={sweeps_completed} "
            f"returned_checkpoint={returned_checkpoint} octave_ranks={tt_ranks_octave}",
            use_tqdm=use_tqdm,
        )
    return {
        "tt_cores": None,
        "dense_tensor": dense_tensor,
        "backend_used": "octave_tt_toolbox_dmrg_cross",
        "faithful_backend": True,
        "info": info,
        "oracle": tracked_oracle,
        "stopped_by_budget": bool(stopped_by_budget),
        "returned_checkpoint": bool(returned_checkpoint),
    }


def _run_custom_fallback_cross(*, shape: tuple[int, ...], value_callback, tol: float, random_state: int | None, rank_cap: int, n_iter_max: int, oracle: LoggedBudgetedOracle | None, verbose: bool = False, use_tqdm: bool = False) -> dict[str, Any]:
    tracked_oracle = oracle or LoggedBudgetedOracle(shape=shape, eval_entry=value_callback, max_unique_queries=None)
    requested_rank = [1] + [int(rank_cap)] * (len(shape) - 1) + [1]
    tt_cores, cross_info = tensor_train_cross_oracle(
        tracked_oracle,
        rank=requested_rank,
        tol=float(tol),
        n_iter_max=int(n_iter_max),
        random_state=random_state,
        verbose=verbose,
        use_tqdm=use_tqdm,
        log_prefix=f"[{NON_FAITHFUL_FALLBACK}]",
    )
    dense_tensor = _tt_to_tensor(tt_cores)
    info = dict(cross_info)
    info["warning"] = NON_FAITHFUL_FALLBACK
    info["value_tt_ranks"] = _tt_ranks_from_cores(tt_cores)
    return {
        "tt_cores": tt_cores,
        "dense_tensor": dense_tensor,
        "backend_used": "custom_fallback",
        "faithful_backend": False,
        "info": info,
        "oracle": tracked_oracle,
    }


def run_authors_tt_cross(
    shape,
    value_callback,
    tol,
    *,
    tt_backend: str = "ttml",
    random_state: int | None = None,
    rank_cap: int = 12,
    n_iter_max: int = 30,
    oracle: LoggedBudgetedOracle | None = None,
    allow_non_faithful_fallback: bool = True,
    tt_toolbox_root: str | None = None,
    ttml_env_name: str | None = None,
    ttml_method: str = "dmrg",
    octave_env_name: str | None = None,
    verbose: bool = False,
    use_tqdm: bool = False,
) -> dict[str, Any]:
    shape = tuple(int(x) for x in shape)
    backend = str(tt_backend)
    errors: list[str] = []

    if backend in {"ttml", "ttml_dmrg", "ttml_regular", "tt_toolbox_like"}:
        try:
            result = _run_ttml_cross(
                shape=shape,
                value_callback=value_callback,
                tol=float(tol),
                random_state=random_state,
                rank_cap=int(rank_cap),
                n_iter_max=int(n_iter_max),
                oracle=oracle,
                ttml_env_name=ttml_env_name,
                ttml_method="regular" if backend == "ttml_regular" else ttml_method,
                verbose=verbose,
                use_tqdm=use_tqdm,
            )
            result["backend_requested"] = backend
            return result
        except BudgetExceeded:
            raise
        except Exception as exc:
            errors.append(str(exc))
            if backend in {"ttml", "ttml_dmrg", "ttml_regular"} and not allow_non_faithful_fallback:
                raise

    if backend in {"octave_tt_toolbox"}:
        try:
            result = _run_octave_tt_toolbox_cross(
                shape=shape,
                value_callback=value_callback,
                tol=float(tol),
                random_state=random_state,
                rank_cap=int(rank_cap),
                n_iter_max=int(n_iter_max),
                oracle=oracle,
                tt_toolbox_root=tt_toolbox_root,
                octave_env_name=octave_env_name,
                verbose=verbose,
                use_tqdm=use_tqdm,
            )
            result["backend_requested"] = backend
            return result
        except BudgetExceeded:
            raise
        except Exception as exc:
            errors.append(str(exc))
            if not allow_non_faithful_fallback:
                raise

    if backend == "custom_fallback" or allow_non_faithful_fallback:
        result = _run_custom_fallback_cross(
            shape=shape,
            value_callback=value_callback,
            tol=float(tol),
            random_state=random_state,
            rank_cap=int(rank_cap),
            n_iter_max=int(n_iter_max),
            oracle=oracle,
            verbose=verbose,
            use_tqdm=use_tqdm,
        )
        result["backend_requested"] = backend
        if errors:
            result["info"]["backend_errors"] = errors
        return result

    raise RuntimeError("No TT-cross backend available")


def sample_random_test_points(context: PaperWaterContext, n_test: int, a: float, b: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(float(a), float(b), size=(int(n_test), int(context.as_dim)))


def evaluate_surrogate_on_test(
    *,
    context: PaperWaterContext,
    test_points: np.ndarray,
    energy_fn,
    a: float,
    b: float,
    coeff_tensor: np.ndarray | None = None,
    coeff_tt_cores: list[np.ndarray] | None = None,
) -> dict[str, float]:
    points = np.asarray(test_points, dtype=float).reshape(-1, context.as_dim)
    exact = np.zeros(points.shape[0], dtype=float)
    pred = np.zeros(points.shape[0], dtype=float)
    for i, point in enumerate(points):
        coords = reduced_point_to_cartesian(context, point)
        exact[i] = float(energy_fn(list(context.symbols), coords))
        if coeff_tt_cores is not None:
            pred[i] = float(evaluate_tt_chebyshev(coeff_tt_cores, point, a=float(a), b=float(b)))
        elif coeff_tensor is not None:
            pred[i] = float(evaluate_chebyshev_tensor(coeff_tensor, point, a=float(a), b=float(b)))
        else:
            raise ValueError("Need coeff_tensor or coeff_tt_cores to evaluate the surrogate")
    err = pred - exact
    return {
        "rmse_mev": float(np.sqrt(np.mean(err ** 2)) * water.HARTREE_TO_MEV),
        "max_abs_mev": float(np.max(np.abs(err)) * water.HARTREE_TO_MEV),
        "chebyshev_error_mev": float(np.max(np.abs(err)) * water.HARTREE_TO_MEV),
        "n_test": float(points.shape[0]),
    }


def evaluate_completed_tensor_on_test(
    *,
    completed_tensor: np.ndarray,
    context: PaperWaterContext,
    nodes: np.ndarray,
    test_points: np.ndarray,
    energy_fn,
    a: float,
    b: float,
    working_value_mode: str = "hartree",
    reference_hartree: float | None = None,
) -> dict[str, float]:
    tensor_arr = np.asarray(completed_tensor, dtype=float)
    coeff_tensor = values_to_chebyshev_coefficients(tensor_arr, np.asarray(nodes, dtype=float), a=float(a), b=float(b))
    if str(working_value_mode) == "hartree":
        metrics = evaluate_surrogate_on_test(
            context=context,
            test_points=test_points,
            energy_fn=energy_fn,
            a=float(a),
            b=float(b),
            coeff_tensor=coeff_tensor,
            coeff_tt_cores=None,
        )
    elif str(working_value_mode) == "relative_mev":
        if reference_hartree is None:
            raise ValueError("reference_hartree is required when working_value_mode='relative_mev'")
        points = np.asarray(test_points, dtype=float).reshape(-1, context.as_dim)
        exact = np.zeros(points.shape[0], dtype=float)
        pred = np.zeros(points.shape[0], dtype=float)
        for i, point in enumerate(points):
            coords = reduced_point_to_cartesian(context, point)
            exact_h = float(energy_fn(list(context.symbols), coords))
            exact[i] = float((exact_h - float(reference_hartree)) * water.HARTREE_TO_MEV)
            pred[i] = float(evaluate_chebyshev_tensor(coeff_tensor, point, a=float(a), b=float(b)))
        err = pred - exact
        metrics = {
            "rmse_mev": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
            "max_abs_mev": float(np.max(np.abs(err))) if err.size else float("nan"),
            "chebyshev_error_mev": float(np.max(np.abs(err))) if err.size else float("nan"),
            "n_test": float(err.size),
        }
    else:
        raise ValueError(f"Unsupported working_value_mode={working_value_mode!r}")

    roundtrip = np.zeros_like(tensor_arr, dtype=float)
    node_arr = np.asarray(nodes, dtype=float)
    for idx in np.ndindex(tensor_arr.shape):
        y = multi_index_to_reduced_point(idx, node_arr)
        roundtrip[idx] = float(evaluate_chebyshev_tensor(coeff_tensor, y, a=float(a), b=float(b)))
    rt_err = roundtrip - tensor_arr
    metrics["grid_roundtrip_rmse"] = float(np.sqrt(np.mean(rt_err ** 2))) if rt_err.size else float("nan")
    metrics["grid_roundtrip_max_abs"] = float(np.max(np.abs(rt_err))) if rt_err.size else float("nan")
    return metrics


def evaluate_completed_tensor_on_grid(
    *,
    completed_tensor: np.ndarray,
    exact_tensor_h: np.ndarray,
    observed_indices: np.ndarray | None = None,
    working_value_mode: str = "hartree",
    reference_hartree: float | None = None,
) -> dict[str, float]:
    completed_arr = np.asarray(completed_tensor, dtype=float)
    exact_h = np.asarray(exact_tensor_h, dtype=float)
    if completed_arr.shape != exact_h.shape:
        raise ValueError(f"Grid metric shape mismatch: {completed_arr.shape} vs {exact_h.shape}")

    if str(working_value_mode) == "hartree":
        exact_work = exact_h
        diff_mev = (completed_arr - exact_work) * water.HARTREE_TO_MEV
    elif str(working_value_mode) == "relative_mev":
        if reference_hartree is None:
            raise ValueError("reference_hartree is required when working_value_mode='relative_mev'")
        exact_work = _to_relative_mev_with_reference(exact_h, float(reference_hartree))
        diff_mev = completed_arr - exact_work
    else:
        raise ValueError(f"Unsupported working_value_mode={working_value_mode!r}")

    metrics = {
        "grid_full_rmse_mev": float(np.sqrt(np.mean(diff_mev ** 2))) if diff_mev.size else float("nan"),
        "grid_full_max_abs_mev": float(np.max(np.abs(diff_mev))) if diff_mev.size else float("nan"),
        "grid_total_points": float(diff_mev.size),
    }
    if observed_indices is None:
        return metrics

    obs_idx = np.asarray(observed_indices, dtype=int)
    obs_mask = np.zeros(completed_arr.shape, dtype=bool)
    if obs_idx.size:
        obs_mask[tuple(obs_idx.T)] = True
    hidden_mask = ~obs_mask

    if np.any(obs_mask):
        obs_err = diff_mev[obs_mask]
        metrics["grid_observed_rmse_mev"] = float(np.sqrt(np.mean(obs_err ** 2)))
        metrics["grid_observed_max_abs_mev"] = float(np.max(np.abs(obs_err)))
        metrics["grid_observed_points"] = float(obs_err.size)
    else:
        metrics["grid_observed_rmse_mev"] = float("nan")
        metrics["grid_observed_max_abs_mev"] = float("nan")
        metrics["grid_observed_points"] = 0.0

    if np.any(hidden_mask):
        hidden_err = diff_mev[hidden_mask]
        metrics["grid_hidden_rmse_mev"] = float(np.sqrt(np.mean(hidden_err ** 2)))
        metrics["grid_hidden_max_abs_mev"] = float(np.max(np.abs(hidden_err)))
        metrics["grid_hidden_points"] = float(hidden_err.size)
    else:
        metrics["grid_hidden_rmse_mev"] = float("nan")
        metrics["grid_hidden_max_abs_mev"] = float("nan")
        metrics["grid_hidden_points"] = 0.0

    return metrics


def _mape_ratio(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    idx = np.abs(true_arr) > float(eps)
    if not np.any(idx):
        return float("nan")
    return float(np.mean(np.abs((true_arr[idx] - pred_arr[idx]) / true_arr[idx])))


def evaluate_completion_metrics_dual_style(
    *,
    completed_tensor_h: np.ndarray,
    exact_tensor_h: np.ndarray,
    observed_indices: np.ndarray | None = None,
) -> dict[str, float]:
    from common import all_metrics

    completed_h = np.asarray(completed_tensor_h, dtype=float)
    exact_h = np.asarray(exact_tensor_h, dtype=float)
    if completed_h.shape != exact_h.shape:
        raise ValueError(f"Completion metric shape mismatch: {completed_h.shape} vs {exact_h.shape}")

    e0 = float(np.min(exact_h))
    exact_rel = _to_relative_mev_with_reference(exact_h, e0)
    pred_rel = _to_relative_mev_with_reference(completed_h, e0)

    obs_idx = None if observed_indices is None else np.asarray(observed_indices, dtype=int)
    obs_mask = np.zeros(completed_h.shape, dtype=bool)
    if obs_idx is not None and obs_idx.size:
        obs_mask[tuple(obs_idx.T)] = True
    hidden_mask = ~obs_mask
    full_mask = np.ones(completed_h.shape, dtype=bool)

    out: dict[str, float] = {
        "completion_reference_energy_hartree": float(e0),
    }

    if np.any(hidden_mask):
        hidden_metrics = all_metrics(exact_rel, pred_rel, hidden_mask)
        y_true_hidden = exact_rel[hidden_mask]
        y_pred_hidden = pred_rel[hidden_mask]
        out.update(
            {
                "completion_hidden_rmse_mev": float(hidden_metrics["rmse"]),
                "completion_hidden_mae_mev": float(hidden_metrics["mae"]),
                "completion_hidden_max_abs_mev": float(hidden_metrics["max_abs_error"]),
                "completion_hidden_relative_rmse": float(hidden_metrics["relative_rmse"]),
                "completion_hidden_mape": float(_mape_ratio(y_true_hidden, y_pred_hidden)),
                "completion_hidden_points": float(np.sum(hidden_mask)),
            }
        )
    else:
        out.update(
            {
                "completion_hidden_rmse_mev": float("nan"),
                "completion_hidden_mae_mev": float("nan"),
                "completion_hidden_max_abs_mev": float("nan"),
                "completion_hidden_relative_rmse": float("nan"),
                "completion_hidden_mape": float("nan"),
                "completion_hidden_points": 0.0,
            }
        )

    full_metrics = all_metrics(exact_rel, pred_rel, full_mask)
    out.update(
        {
            "completion_full_rmse_mev": float(full_metrics["rmse"]),
            "completion_full_mae_mev": float(full_metrics["mae"]),
            "completion_full_max_abs_mev": float(full_metrics["max_abs_error"]),
            "completion_full_relative_rmse": float(full_metrics["relative_rmse"]),
            "completion_full_mape": float(_mape_ratio(exact_rel[full_mask], pred_rel[full_mask])),
            "completion_full_points": float(np.sum(full_mask)),
        }
    )
    return out


def _relative_mev_reference_from_observations(values_h: np.ndarray) -> float:
    arr = np.asarray(values_h, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Need at least one observed value to build a completion reference")
    return float(np.min(arr))


def _to_relative_mev_with_reference(values_h: np.ndarray, reference_hartree: float) -> np.ndarray:
    return (np.asarray(values_h, dtype=float) - float(reference_hartree)) * water.HARTREE_TO_MEV


def _from_relative_mev_with_reference(values_mev: np.ndarray, reference_hartree: float) -> np.ndarray:
    return np.asarray(values_mev, dtype=float) / water.HARTREE_TO_MEV + float(reference_hartree)


def _simple_point_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    true_arr = np.asarray(true_values, dtype=float).reshape(-1)
    pred_arr = np.asarray(pred_values, dtype=float).reshape(-1)
    err = pred_arr - true_arr
    return {
        "rmse_mev": float(np.sqrt(np.mean(err ** 2)) * water.HARTREE_TO_MEV) if err.size else float("nan"),
        "mae_mev": float(np.mean(np.abs(err)) * water.HARTREE_TO_MEV) if err.size else float("nan"),
        "max_abs_mev": float(np.max(np.abs(err)) * water.HARTREE_TO_MEV) if err.size else float("nan"),
        "n_points": float(err.size),
    }


def run_baranov2015_water_baseline(
    args: Any | None,
    n_points: int,
    tol: float,
    *,
    tt_backend: str = "ttml",
    random_state: int | None = None,
) -> AuthorsBaselineResult:
    args_ns = _coerce_args(args, n_points=int(n_points))
    verbose = bool(getattr(args_ns, "verbose", False))
    use_tqdm = bool(getattr(args_ns, "show_tqdm", False))
    interval = tuple(float(v) for v in getattr(args_ns, "cheb_interval"))
    if verbose:
        _paper_log(
            f"[authors-baseline] START n={int(n_points)} tol={float(tol):.1e} backend={tt_backend} "
            f"interval={interval} as_samples={int(getattr(args_ns, 'water_as_samples', 0))}",
            use_tqdm=use_tqdm,
        )
    context = compute_active_subspace_paper(args_ns, build_paper_water_context(args_ns))
    nodes = build_chebyshev_nodes(int(n_points), interval[0], interval[1])
    context = replace(context, cheb_nodes=np.asarray(nodes, dtype=float), cheb_interval=interval)
    energy_fn = water._build_energy_fn(args_ns)
    eval_entry = make_exact_pes_entry_oracle(context, nodes, energy_fn)
    oracle = LoggedBudgetedOracle(
        shape=tuple([int(n_points)] * context.as_dim),
        eval_entry=eval_entry,
        max_unique_queries=None,
        verbose=verbose,
        use_tqdm=use_tqdm,
        log_prefix="[authors-baseline/oracle]",
        log_first_unique=int(getattr(args_ns, "author_log_first_unique", 8)),
        log_every_unique=int(getattr(args_ns, "author_log_every_unique", 25)),
        log_every_total=int(getattr(args_ns, "author_log_every_total", 250)),
    )
    cross = run_authors_tt_cross(
        oracle.shape,
        oracle.query,
        float(tol),
        tt_backend=tt_backend,
        random_state=int(getattr(args_ns, "author_random_state", 1729) if random_state is None else random_state),
        rank_cap=int(getattr(args_ns, "author_rank_cap", 12)),
        n_iter_max=int(getattr(args_ns, "author_max_iter", 30)),
        oracle=oracle,
        tt_toolbox_root=str(getattr(args_ns, "tt_toolbox_root")),
        ttml_env_name=getattr(args_ns, "ttml_env_name", None),
        ttml_method=str(getattr(args_ns, "ttml_method", "dmrg")),
        octave_env_name=getattr(args_ns, "octave_env_name", None),
        verbose=verbose,
        use_tqdm=use_tqdm,
        allow_non_faithful_fallback=False,
    )
    approx_tensor = np.asarray(cross["dense_tensor"], dtype=float)
    display_tt_ranks = [int(x) for x in cross["info"].get("value_tt_ranks_octave", [])]
    storage = int(cross["info"].get("storage", _storage_from_tt_ranks(tuple(oracle.shape), display_tt_ranks)))
    if verbose:
        _paper_log(
            f"[authors-baseline] CROSS_DONE backend={cross['backend_used']} unique={oracle.unique_queries()} "
            f"total={oracle.total_queries()} value_ranks={display_tt_ranks}",
            use_tqdm=use_tqdm,
        )
    coeff_tensor = values_to_chebyshev_coefficients(approx_tensor, nodes, a=interval[0], b=interval[1])
    rms_random_mev = sample_random_rms(
        context=_planar_context_from_paper(context),
        energy_fn=energy_fn,
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=None,
        a=interval[0],
        b=interval[1],
        n_samples=int(getattr(args_ns, "author_rms_probe_points", 1000)),
        seed=int(getattr(args_ns, "author_rms_seed", 2025)),
        verbose=verbose,
        use_tqdm=use_tqdm,
    )
    test_points = sample_random_test_points(
        context=context,
        n_test=int(getattr(args_ns, "author_rms_probe_points", 1000)),
        a=interval[0],
        b=interval[1],
        seed=int(getattr(args_ns, "author_rms_seed", 2025)),
    )
    test_metrics = evaluate_surrogate_on_test(
        context=context,
        test_points=test_points,
        energy_fn=energy_fn,
        a=interval[0],
        b=interval[1],
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=None,
    )
    full_status = "converged" if bool(cross["info"].get("converged", True)) else "max_sweeps_reached"
    trace = _oracle_to_trace(
        oracle,
        context=context,
        nodes=nodes,
        status=full_status,
        tt_backend_requested=tt_backend,
        tt_backend_used=str(cross["backend_used"]),
        faithful_backend=bool(cross["faithful_backend"]),
        info={**dict(cross["info"]), "status": full_status},
    )
    if verbose:
        _paper_log(
            f"[authors-baseline] METRICS rms_random_mev={float(rms_random_mev):.6f} "
            f"test_rmse_mev={float(test_metrics.get('rmse_mev', float('nan'))):.6f} "
            f"test_max_abs_mev={float(test_metrics.get('max_abs_mev', float('nan'))):.6f} "
            f"storage={storage}",
            use_tqdm=use_tqdm,
        )
    return AuthorsBaselineResult(
        context=context,
        sample_trace=trace,
        tensor=approx_tensor,
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=None,
        tt_cores=None,
        rms_random_mev=float(rms_random_mev),
        test_metrics=test_metrics,
        tt_ranks=display_tt_ranks,
        storage=storage,
        total_queries=oracle.total_queries(),
        unique_queries=oracle.unique_queries(),
        backend_requested=str(tt_backend),
        backend_used=str(cross["backend_used"]),
        faithful_backend=bool(cross["faithful_backend"]),
        info=dict(cross["info"]),
    )


def run_baranov2015_water_budgeted(
    args: Any | None,
    *,
    n_points: int,
    tol: float,
    unique_budget: int,
    tt_backend: str = "ttml",
    random_state: int | None = None,
) -> AuthorsBaselineResult:
    args_ns = _coerce_args(args, n_points=int(n_points), tt_backend=str(tt_backend))
    verbose = bool(getattr(args_ns, "verbose", False))
    use_tqdm = bool(getattr(args_ns, "show_tqdm", False))

    if verbose:
        _paper_log(
            f"[authors-budgeted] START n_points={int(n_points)} tol={float(tol):.1e} "
            f"budget={int(unique_budget)} backend={tt_backend}",
            use_tqdm=use_tqdm,
        )

    context = build_paper_water_context(args_ns)
    context = compute_active_subspace_paper(args_ns, context)
    interval = tuple(float(v) for v in getattr(args_ns, "cheb_interval"))
    nodes = build_chebyshev_nodes(int(n_points), a=interval[0], b=interval[1])
    energy_fn = water._build_energy_fn(args_ns)
    eval_entry = make_exact_pes_entry_oracle(context, nodes, energy_fn)
    oracle = LoggedBudgetedOracle(
        shape=tuple([int(n_points)] * context.as_dim),
        eval_entry=eval_entry,
        max_unique_queries=int(unique_budget),
        verbose=verbose,
        use_tqdm=use_tqdm,
        log_prefix="[authors-budgeted/oracle]",
        log_first_unique=int(getattr(args_ns, "author_log_first_unique", 8)),
        log_every_unique=int(getattr(args_ns, "author_log_every_unique", 25)),
        log_every_total=int(getattr(args_ns, "author_log_every_total", 250)),
    )

    cross = run_authors_tt_cross(
        oracle.shape,
        oracle.query,
        float(tol),
        tt_backend=tt_backend,
        random_state=int(getattr(args_ns, "author_random_state", 1729) if random_state is None else random_state),
        rank_cap=int(getattr(args_ns, "author_rank_cap", 12)),
        n_iter_max=int(getattr(args_ns, "author_max_iter", 30)),
        oracle=oracle,
        tt_toolbox_root=str(getattr(args_ns, "tt_toolbox_root")),
        ttml_env_name=getattr(args_ns, "ttml_env_name", None),
        ttml_method=str(getattr(args_ns, "ttml_method", "dmrg")),
        octave_env_name=getattr(args_ns, "octave_env_name", None),
        verbose=verbose,
        use_tqdm=use_tqdm,
        allow_non_faithful_fallback=False,
    )

    approx_tensor = np.asarray(cross["dense_tensor"], dtype=float)
    display_tt_ranks = [int(x) for x in cross["info"].get("value_tt_ranks_octave", [])]
    storage = int(cross["info"].get("storage", _storage_from_tt_ranks(tuple(oracle.shape), display_tt_ranks)))
    if bool(cross.get("stopped_by_budget", False)):
        status = "budget_exhausted_partial_surrogate"
    elif bool(cross["info"].get("converged", True)):
        status = "converged_before_budget"
    else:
        status = "max_sweeps_reached_under_budget"
    coeff_tensor = None
    rms_random_mev = None
    test_metrics: dict[str, float] = {}
    if status == "converged_before_budget":
        coeff_tensor = values_to_chebyshev_coefficients(approx_tensor, nodes, a=interval[0], b=interval[1])
        rms_random_mev = sample_random_rms(
            context=_planar_context_from_paper(context),
            energy_fn=energy_fn,
            coeff_tensor=coeff_tensor,
            coeff_tt_cores=None,
            a=interval[0],
            b=interval[1],
            n_samples=int(getattr(args_ns, "author_rms_probe_points", 1000)),
            seed=int(getattr(args_ns, "author_rms_seed", 2025)),
            verbose=verbose,
            use_tqdm=use_tqdm,
        )
        test_points = sample_random_test_points(
            context=context,
            n_test=int(getattr(args_ns, "author_rms_probe_points", 1000)),
            a=interval[0],
            b=interval[1],
            seed=int(getattr(args_ns, "author_rms_seed", 2025)),
        )
        test_metrics = evaluate_surrogate_on_test(
            context=context,
            test_points=test_points,
            energy_fn=energy_fn,
            a=interval[0],
            b=interval[1],
            coeff_tensor=coeff_tensor,
            coeff_tt_cores=None,
        )
    trace = _oracle_to_trace(
        oracle,
        context=context,
        nodes=nodes,
        status=status,
        tt_backend_requested=tt_backend,
        tt_backend_used=str(cross["backend_used"]),
        faithful_backend=bool(cross["faithful_backend"]),
        info={**dict(cross["info"]), "status": status},
    )
    if verbose:
        _paper_log(
            f"[authors-budgeted] DONE status={status} unique={oracle.unique_queries()} total={oracle.total_queries()} "
            f"test_rmse_mev={float(test_metrics.get('rmse_mev', float('nan'))):.6f}",
            use_tqdm=use_tqdm,
        )
    return AuthorsBaselineResult(
        context=context,
        sample_trace=trace,
        tensor=approx_tensor,
        coeff_tensor=coeff_tensor,
        coeff_tt_cores=None,
        tt_cores=None,
        rms_random_mev=None if rms_random_mev is None else float(rms_random_mev),
        test_metrics=test_metrics,
        tt_ranks=display_tt_ranks,
        storage=storage,
        total_queries=oracle.total_queries(),
        unique_queries=oracle.unique_queries(),
        backend_requested=str(tt_backend),
        backend_used=str(cross["backend_used"]),
        faithful_backend=bool(cross["faithful_backend"]),
        info={**dict(cross["info"]), "status": status},
    )


def collect_author_samples(
    args: Any | None,
    n_points: int,
    tol: float,
    unique_budget: int,
    *,
    tt_backend: str = "ttml",
    random_state: int | None = None,
) -> SampleTrace:
    args_ns = _coerce_args(args, n_points=int(n_points))
    verbose = bool(getattr(args_ns, "verbose", False))
    use_tqdm = bool(getattr(args_ns, "show_tqdm", False))
    interval = tuple(float(v) for v in getattr(args_ns, "cheb_interval"))
    if verbose:
        _paper_log(
            f"[authors-samples] START n={int(n_points)} tol={float(tol):.1e} budget={int(unique_budget)} "
            f"backend={tt_backend} interval={interval}",
            use_tqdm=use_tqdm,
        )
    context = compute_active_subspace_paper(args_ns, build_paper_water_context(args_ns))
    nodes = build_chebyshev_nodes(int(n_points), interval[0], interval[1])
    context = replace(context, cheb_nodes=np.asarray(nodes, dtype=float), cheb_interval=interval)
    energy_fn = water._build_energy_fn(args_ns)
    eval_entry = make_exact_pes_entry_oracle(context, nodes, energy_fn)
    oracle = LoggedBudgetedOracle(
        shape=tuple([int(n_points)] * context.as_dim),
        eval_entry=eval_entry,
        max_unique_queries=int(unique_budget),
        verbose=verbose,
        use_tqdm=use_tqdm,
        log_prefix="[authors-samples/oracle]",
        log_first_unique=int(getattr(args_ns, "author_log_first_unique", 8)),
        log_every_unique=int(getattr(args_ns, "author_log_every_unique", 25)),
        log_every_total=int(getattr(args_ns, "author_log_every_total", 250)),
    )
    status = "converged_before_budget"
    cross_info: dict[str, Any] = {}
    backend_used = tt_backend
    faithful_backend = False
    try:
        cross = run_authors_tt_cross(
            oracle.shape,
            oracle.query,
            float(tol),
            tt_backend=tt_backend,
            random_state=int(getattr(args_ns, "author_random_state", 1729) if random_state is None else random_state),
            rank_cap=int(getattr(args_ns, "author_rank_cap", 12)),
            n_iter_max=int(getattr(args_ns, "author_max_iter", 30)),
            oracle=oracle,
            tt_toolbox_root=str(getattr(args_ns, "tt_toolbox_root")),
            ttml_env_name=getattr(args_ns, "ttml_env_name", None),
            ttml_method=str(getattr(args_ns, "ttml_method", "dmrg")),
            octave_env_name=getattr(args_ns, "octave_env_name", None),
            verbose=verbose,
            use_tqdm=use_tqdm,
            allow_non_faithful_fallback=False,
        )
        cross_info = dict(cross["info"])
        backend_used = str(cross["backend_used"])
        faithful_backend = bool(cross["faithful_backend"])
        if bool(cross.get("stopped_by_budget", False)) or bool(cross_info.get("stopped_by_budget", False)):
            status = "budget_exhausted_partial_surrogate"
        elif not bool(cross_info.get("converged", True)):
            status = "max_sweeps_reached_under_budget"
        if verbose:
            _paper_log(
                f"[authors-samples] DONE status={status} unique={oracle.unique_queries()} "
                f"total={oracle.total_queries()} backend={backend_used}",
                use_tqdm=use_tqdm,
            )
    except BudgetExceeded as exc:
        status = "budget_exhausted"
        cross_info = {"error": str(exc)}
        if verbose:
            _paper_log(
                f"[authors-samples] DONE status=budget_exhausted unique={oracle.unique_queries()} "
                f"total={oracle.total_queries()} backend={backend_used}",
                use_tqdm=use_tqdm,
            )
    return _oracle_to_trace(
        oracle,
        context=context,
        nodes=nodes,
        status=status,
        tt_backend_requested=tt_backend,
        tt_backend_used=backend_used,
        faithful_backend=faithful_backend,
        info=cross_info,
    )


def run_completion_on_author_samples(sample_trace: SampleTrace, completion_kwargs: dict[str, Any] | None = None) -> CompletionResult:
    import completion_linalg_tensor_masked as completion_backend

    kwargs = dict(completion_kwargs or {})
    if "n_workers" in kwargs:
        completion_backend.n_workers = int(kwargs.pop("n_workers"))
    exact_grid_tensor_h = kwargs.pop("exact_grid_tensor_h", None)
    indices = np.asarray(sample_trace.unique_indices, dtype=int)
    values_h = np.asarray(sample_trace.unique_values, dtype=float)
    shape = tuple(int(x) for x in sample_trace.shape)
    if indices.size == 0:
        raise ValueError("Sample trace contains no observed entries")
    completion_value_mode = str(kwargs.pop("completion_value_mode", "relative_mev_observed_min"))
    if completion_value_mode == "relative_mev_observed_min":
        reference_hartree = float(kwargs.pop("reference_energy_hartree", _relative_mev_reference_from_observations(values_h)))
        values = _to_relative_mev_with_reference(values_h, reference_hartree)
    elif completion_value_mode == "hartree":
        reference_hartree = None
        values = values_h
    else:
        raise ValueError(
            f"Unsupported completion_value_mode={completion_value_mode!r}; "
            "expected 'relative_mev_observed_min' or 'hartree'"
        )
    algo_kwargs = {
        "indices": indices,
        "values": values,
        "shape": shape,
        "rank": int(kwargs.pop("rank", 10)),
        "number_of_steps": int(kwargs.pop("number_of_steps", 5)),
        "tol_for_step": float(kwargs.pop("tol_for_step", 1e-4)),
        "begin": kwargs.pop("begin", "oversample"),
        "lambda_all": float(kwargs.pop("lambda_all", 0.0)),
        "rank_eval": bool(kwargs.pop("rank_eval", False)),
        "rank_nest": bool(kwargs.pop("rank_nest", False)),
        "nest_iters": int(kwargs.pop("nest_iters", 5)),
        "tol": float(kwargs.pop("tol", 1e-3)),
        "max_rank": int(kwargs.pop("max_rank", 20)),
        "using_qr": bool(kwargs.pop("using_qr", False)),
        "eval": bool(kwargs.pop("eval", True)),
        "seed": int(kwargs.pop("seed", 179)),
        "return_compl": True,
        "return_history": True,
        "ret_best": bool(kwargs.pop("ret_best", True)),
        "TQDM": bool(kwargs.pop("TQDM", False)),
        "eval_fall": bool(kwargs.pop("eval_fall", False)),
        "validation_size": float(kwargs.pop("validation_size", 0.1)),
        "tolerance": int(kwargs.pop("tolerance", 500)),
        "min_validation_points": int(kwargs.pop("min_validation_points", 8)),
        "min_train_ratio_to_params": float(kwargs.pop("min_train_ratio_to_params", 0.5)),
        "dual_guided": bool(kwargs.pop("dual_guided", False)),
        "pivot_topk": int(kwargs.pop("pivot_topk", 8)),
        "strict_rank": bool(kwargs.pop("strict_rank", False)),
        "allow_rank_growth": bool(kwargs.pop("allow_rank_growth", True)),
    }
    validation_plan = completion_backend.choose_validation_plan(
        shape=shape,
        observed_points=len(indices),
        rank=int(algo_kwargs["rank"]),
        validation_fraction=float(algo_kwargs["validation_size"]),
        min_validation_points=int(algo_kwargs["min_validation_points"]),
        min_train_ratio_to_params=float(algo_kwargs["min_train_ratio_to_params"]),
        strict_rank=bool(algo_kwargs["strict_rank"]),
    )
    t0 = time.time()
    factors, history, completed_tensor = completion_backend.approximateLOO_masked(**algo_kwargs)
    elapsed = time.time() - t0
    completed_work = None if completed_tensor is None else np.asarray(completed_tensor, dtype=float)
    if completed_work is None:
        completed_arr = None
    elif completion_value_mode == "relative_mev_observed_min":
        completed_arr = _from_relative_mev_with_reference(completed_work, reference_hartree)
    else:
        completed_arr = completed_work
    train_metrics: dict[str, float] = {}
    coeff_tensor = None
    if completed_arr is not None:
        observed_pred = completed_arr[tuple(indices.T)]
        train_metrics = _simple_point_metrics(values_h, observed_pred)
        coeff_tensor = values_to_chebyshev_coefficients(
            completed_arr,
            np.asarray(sample_trace.nodes, dtype=float),
            a=float(sample_trace.context.cheb_interval[0]),
            b=float(sample_trace.context.cheb_interval[1]),
        )
    validation_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    if completed_arr is not None and exact_grid_tensor_h is not None:
        test_metrics.update(
            evaluate_completed_tensor_on_grid(
                completed_tensor=completed_work if completion_value_mode == "relative_mev_observed_min" else completed_arr,
                exact_tensor_h=np.asarray(exact_grid_tensor_h, dtype=float),
                observed_indices=indices,
                working_value_mode="relative_mev" if completion_value_mode == "relative_mev_observed_min" else "hartree",
                reference_hartree=reference_hartree,
            )
        )
        test_metrics.update(
            evaluate_completion_metrics_dual_style(
                completed_tensor_h=completed_arr,
                exact_tensor_h=np.asarray(exact_grid_tensor_h, dtype=float),
                observed_indices=indices,
            )
        )
    test_points = kwargs.pop("test_points", None)
    energy_fn = kwargs.pop("energy_fn", None)
    if completed_arr is not None and test_points is not None and energy_fn is not None:
        test_metrics.update(
            evaluate_completed_tensor_on_test(
                completed_tensor=completed_work if completion_value_mode == "relative_mev_observed_min" else completed_arr,
                context=sample_trace.context,
                nodes=sample_trace.nodes,
                test_points=np.asarray(test_points, dtype=float),
                energy_fn=energy_fn,
                a=float(sample_trace.context.cheb_interval[0]),
                b=float(sample_trace.context.cheb_interval[1]),
                working_value_mode="relative_mev" if completion_value_mode == "relative_mev_observed_min" else "hartree",
                reference_hartree=reference_hartree,
            )
        )
    return CompletionResult(
        completed_tensor=completed_arr,
        factors=None if factors is None else list(factors),
        history=[] if history is None else list(history),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        coeff_tensor=coeff_tensor,
        info={
            "elapsed_sec": float(elapsed),
            "unused_kwargs": kwargs,
            "completion_value_mode": completion_value_mode,
            "reference_energy_hartree": reference_hartree,
            "observed_value_min_hartree": float(np.min(values_h)),
            "observed_value_max_hartree": float(np.max(values_h)),
            "working_value_min": float(np.min(values)),
            "working_value_max": float(np.max(values)),
            "completed_value_min_hartree": None if completed_arr is None else float(np.min(completed_arr)),
            "completed_value_max_hartree": None if completed_arr is None else float(np.max(completed_arr)),
            "observed_points": int(len(indices)),
            "train_points_planned": int(validation_plan["train_points"]),
            "val_points_planned": int(validation_plan["val_points"]),
            "requested_rank": int(algo_kwargs["rank"]),
            "effective_rank": int(validation_plan["effective_rank"]),
            "rank_cap": int(validation_plan["rank_cap"]),
        },
    )


def _prefix_sample_trace(sample_trace: SampleTrace, budget: int) -> SampleTrace:
    budget = int(min(int(budget), sample_trace.unique_queries))
    return SampleTrace(
        unique_indices=np.asarray(sample_trace.unique_indices[:budget], dtype=int).copy(),
        unique_values=np.asarray(sample_trace.unique_values[:budget], dtype=float).copy(),
        query_sequence=np.asarray(sample_trace.query_sequence, dtype=int).copy(),
        total_queries=int(sample_trace.total_queries),
        unique_queries=int(budget),
        shape=tuple(sample_trace.shape),
        nodes=np.asarray(sample_trace.nodes, dtype=float).copy(),
        context=sample_trace.context,
        status=sample_trace.status,
        tt_backend_requested=sample_trace.tt_backend_requested,
        tt_backend_used=sample_trace.tt_backend_used,
        faithful_backend=sample_trace.faithful_backend,
        info=dict(sample_trace.info),
    )


def run_budget_sweep_experiment(
    args: Any | None,
    budgets: list[int],
    n_points: int,
    tol: float,
    completion_kwargs: dict[str, Any] | None,
    *,
    seed: int = 0,
) -> BudgetSweepResult:
    args_ns = _coerce_args(args, n_points=int(n_points))
    if not budgets:
        raise ValueError("budgets must be non-empty")
    budgets_sorted = sorted(int(b) for b in budgets)
    full_trace = collect_author_samples(
        args_ns,
        n_points=int(n_points),
        tol=float(tol),
        unique_budget=int(max(budgets_sorted)),
        tt_backend=str(getattr(args_ns, "tt_backend", "ttml")),
        random_state=int(seed),
    )
    interval = tuple(float(v) for v in getattr(args_ns, "cheb_interval"))
    energy_fn = water._build_energy_fn(args_ns)
    test_points = sample_random_test_points(
        full_trace.context,
        n_test=int(getattr(args_ns, "author_rms_probe_points", 1000)),
        a=interval[0],
        b=interval[1],
        seed=int(seed),
    )
    rows: list[BudgetSweepRow] = []
    completion_kwargs = dict(completion_kwargs or {})
    completion_kwargs.setdefault("test_points", test_points)
    completion_kwargs.setdefault("energy_fn", energy_fn)
    authors_baseline = None
    if full_trace.status == "converged_before_budget":
        authors_baseline = run_baranov2015_water_baseline(
            args_ns,
            n_points=int(n_points),
            tol=float(tol),
            tt_backend=str(getattr(args_ns, "tt_backend", "ttml")),
            random_state=int(seed),
        )
    for budget in budgets_sorted:
        prefix = _prefix_sample_trace(full_trace, int(budget))
        completion_result = run_completion_on_author_samples(prefix, completion_kwargs)
        rows.append(
            BudgetSweepRow(
                budget=int(budget),
                effective_budget=int(prefix.unique_queries),
                status="ok",
                completion_result=completion_result,
                info={"author_status": full_trace.status},
            )
        )
    return BudgetSweepResult(
        budgets=budgets_sorted,
        rows=rows,
        sample_trace=full_trace,
        authors_baseline=authors_baseline,
        test_points=test_points,
        info={"n_points": int(n_points), "tol": float(tol)},
    )


def main_collect_author_samples(**kwargs: Any) -> SampleTrace:
    args_ns = _coerce_args(kwargs)
    return collect_author_samples(
        args_ns,
        n_points=int(kwargs.get("n_points", 6)),
        tol=float(kwargs.get("tol", getattr(args_ns, "author_tol", 1e-5))),
        unique_budget=int(kwargs.get("unique_budget", 100)),
        tt_backend=str(kwargs.get("tt_backend", getattr(args_ns, "tt_backend", "ttml"))),
        random_state=kwargs.get("random_state"),
    )


def main_run_authors_baseline(**kwargs: Any) -> AuthorsBaselineResult:
    args_ns = _coerce_args(kwargs)
    return run_baranov2015_water_baseline(
        args_ns,
        n_points=int(kwargs.get("n_points", 6)),
        tol=float(kwargs.get("tol", getattr(args_ns, "author_tol", 1e-5))),
        tt_backend=str(kwargs.get("tt_backend", getattr(args_ns, "tt_backend", "ttml"))),
        random_state=kwargs.get("random_state"),
    )


def main_run_budget_sweep(**kwargs: Any) -> BudgetSweepResult:
    args_ns = _coerce_args(kwargs)
    budgets = kwargs.get("budgets", [64, 128, 256])
    if isinstance(budgets, str):
        budgets = [int(x) for x in budgets.split(",") if x.strip()]
    return run_budget_sweep_experiment(
        args_ns,
        budgets=[int(x) for x in budgets],
        n_points=int(kwargs.get("n_points", 6)),
        tol=float(kwargs.get("tol", getattr(args_ns, "author_tol", 1e-5))),
        completion_kwargs=dict(kwargs.get("completion_kwargs", {})),
        seed=int(kwargs.get("seed", 0)),
    )


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baranov & Oseledets 2015 faithful water pipeline")
    parser.add_argument("--mode", choices=["authors_baseline", "collect_samples", "budget_sweep"], default="authors_baseline")
    parser.add_argument("--n-points", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--unique-budget", type=int, default=128)
    parser.add_argument("--budgets", type=str, default="64,128,256")
    parser.add_argument("--tt-backend", type=str, default="octave_tt_toolbox")
    parser.add_argument("--ttml-env-name", type=str, default="matrix_approximation_final_3_11")
    parser.add_argument("--ttml-method", type=str, choices=["dmrg", "regular"], default="dmrg")
    parser.add_argument("--octave-env-name", type=str, default="octave")
    parser.add_argument("--water-as-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-tqdm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _main_cli() -> None:
    parser = _build_cli_parser()
    ns = parser.parse_args()
    args = _coerce_args(
        ns,
        tt_backend=ns.tt_backend,
        ttml_env_name=ns.ttml_env_name,
        ttml_method=ns.ttml_method,
        octave_env_name=ns.octave_env_name,
        water_as_samples=ns.water_as_samples,
        show_tqdm=ns.show_tqdm,
        verbose=ns.verbose,
    )
    if ns.mode == "authors_baseline":
        result = run_baranov2015_water_baseline(args, n_points=ns.n_points, tol=ns.tol, tt_backend=ns.tt_backend, random_state=ns.seed)
    elif ns.mode == "collect_samples":
        result = collect_author_samples(args, n_points=ns.n_points, tol=ns.tol, unique_budget=ns.unique_budget, tt_backend=ns.tt_backend, random_state=ns.seed)
    else:
        budgets = [int(x) for x in ns.budgets.split(",") if x.strip()]
        result = run_budget_sweep_experiment(args, budgets=budgets, n_points=ns.n_points, tol=ns.tol, completion_kwargs={}, seed=ns.seed)
    print(json.dumps(result, default=_json_default, indent=2))


if __name__ == "__main__":
    _main_cli()
