from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def values_to_chebyshev_coefficients(values: np.ndarray, nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    coeff = np.asarray(values, dtype=float).copy()
    degree = coeff.shape[0]
    T = _cheb_basis_matrix(nodes, degree, a=a, b=b)
    T_inv = np.linalg.inv(T)
    for mode in range(coeff.ndim):
        coeff = np.moveaxis(coeff, mode, 0)
        flat = coeff.reshape(degree, -1)
        flat = T_inv @ flat
        coeff = flat.reshape(coeff.shape)
        coeff = np.moveaxis(coeff, 0, mode)
    return coeff


def cheb_transform_matrix(nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    degree = int(len(nodes))
    T = _cheb_basis_matrix(nodes, degree, a=a, b=b)
    return np.linalg.inv(T)


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
