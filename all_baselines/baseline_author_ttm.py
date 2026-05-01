from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


@dataclass(frozen=True)
class ThermoQuantitySpec:
    key: str
    label: str
    column: int
    modes: tuple[int, ...]


THERMO_QUANTITY_ORDER: tuple[str, ...] = (
    "gibbs",
    "potential_1",
    "potential_2",
    "potential_3",
    "second_1_1",
    "second_1_2",
    "second_1_3",
    "second_2_2",
    "second_2_3",
    "second_3_3",
)

THERMO_QUANTITY_SPECS: dict[str, ThermoQuantitySpec] = {
    "gibbs": ThermoQuantitySpec("gibbs", "Gibbs energy", 3, ()),
    "potential_1": ThermoQuantitySpec("potential_1", "Potential x1", 4, (1,)),
    "potential_2": ThermoQuantitySpec("potential_2", "Potential x2", 5, (2,)),
    "potential_3": ThermoQuantitySpec("potential_3", "Potential x3", 6, (3,)),
    "second_1_1": ThermoQuantitySpec("second_1_1", "2nd Der. x1, x1", 7, (1, 1)),
    "second_1_2": ThermoQuantitySpec("second_1_2", "2nd Der. x1, x2", 8, (1, 2)),
    "second_1_3": ThermoQuantitySpec("second_1_3", "2nd Der. x1, x3", 9, (1, 3)),
    "second_2_2": ThermoQuantitySpec("second_2_2", "2nd Der. x2, x2", 10, (2, 2)),
    "second_2_3": ThermoQuantitySpec("second_2_3", "2nd Der. x2, x3", 11, (2, 3)),
    "second_3_3": ThermoQuantitySpec("second_3_3", "2nd Der. x3, x3", 12, (3, 3)),
}

_R_GAS = 8.3144598


def get_quantity_spec(key: str) -> ThermoQuantitySpec:
    try:
        return THERMO_QUANTITY_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown thermodynamic quantity: {key}") from exc


def _composition_with_dependent(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    dep = 1.0 - np.sum(coords, axis=1, keepdims=True)
    return np.concatenate([coords, dep], axis=1)


def ideal_mixing_term(coords: np.ndarray, temperature: float, modes: tuple[int, ...]) -> np.ndarray:
    cx = _composition_with_dependent(coords)
    rt = float(_R_GAS * temperature)
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(modes) == 0:
            return rt * np.sum(cx * np.log(cx), axis=1)
        if len(modes) == 1:
            n = int(modes[0]) - 1
            return rt * (np.log(cx[:, n]) - np.log(cx[:, -1]))
        n = int(modes[0]) - 1
        m = int(modes[1]) - 1
        if n == m:
            return rt * (1.0 / cx[:, n] + 1.0 / cx[:, -1])
        return rt * (1.0 / cx[:, -1])


def preprocess_quantity(values: np.ndarray, coords: np.ndarray, temperature: float, modes: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=float) - ideal_mixing_term(coords, temperature, modes)


def reconstruct_quantity(values: np.ndarray, coords: np.ndarray, temperature: float, modes: tuple[int, ...]) -> np.ndarray:
    return np.asarray(values, dtype=float) + ideal_mixing_term(coords, temperature, modes)


def compute_error_vector(true: np.ndarray, pred: np.ndarray, error_type: str) -> np.ndarray:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    abs_err = np.abs(true - pred)
    if error_type == "absolute":
        return abs_err
    if error_type == "range":
        denom = float(np.max(true) - np.min(true))
        return abs_err / max(denom, 1e-30)
    if error_type == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            return abs_err / np.abs(true)
    raise ValueError(f"Unknown error_type: {error_type}")


def ecdf_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    vals = np.sort(vals)
    y = np.arange(1, vals.size + 1, dtype=float) / float(vals.size)
    return vals, y


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.quantile(vals, 0.5)),
        "q90": float(np.quantile(vals, 0.9)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "max": float(np.max(vals)),
    }


@dataclass
class AuthorTTMDataset:
    root: Path
    temperature: float
    shape: tuple[int, int, int]
    grid: np.ndarray
    coords: np.ndarray
    dense_indices: np.ndarray
    values: dict[str, np.ndarray]
    train_valid_mask: np.ndarray
    train_grid_indices: np.ndarray

    def quantity_values(self, quantity_key: str, *, preprocessed: bool) -> np.ndarray:
        spec = get_quantity_spec(quantity_key)
        values = np.asarray(self.values[quantity_key], dtype=float)
        if not preprocessed:
            return values
        return preprocess_quantity(values, self.coords, self.temperature, spec.modes)

    def build_dense_tensor(self, quantity_key: str, *, preprocessed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = np.zeros(self.shape, dtype=float)
        valid_mask = np.zeros(self.shape, dtype=bool)
        train_mask = np.zeros(self.shape, dtype=bool)
        values = self.quantity_values(quantity_key, preprocessed=preprocessed)
        ii, jj, kk = self.dense_indices.T
        tensor[ii, jj, kk] = values
        valid_mask[ii, jj, kk] = True
        tr_idx = self.dense_indices[self.train_valid_mask]
        if tr_idx.size:
            ti, tj, tk = tr_idx.T
            train_mask[ti, tj, tk] = True
        return tensor, valid_mask, train_mask

    def slice_mask(self, mode: int, value: float) -> tuple[np.ndarray, float]:
        axis = int(mode) - 1
        nearest = float(self.grid[np.argmin(np.abs(self.grid - float(value)))])
        mask = np.isclose(self.coords[:, axis], nearest)
        return mask, nearest


def load_author_ttm_dataset(dataset_root: str | Path, temperature: float = 1400.0, train_stride: int = 5) -> AuthorTTMDataset:
    root = Path(dataset_root)
    tdt_path = root / "1_TDT_sampling" / "output" / "TDT.mat"
    if not tdt_path.exists():
        raise FileNotFoundError(f"Missing TDT.mat: {tdt_path}")
    payload = scipy.io.loadmat(tdt_path)
    if "TDT" not in payload:
        raise KeyError(f"Expected key 'TDT' in {tdt_path}")
    tdt = np.asarray(payload["TDT"], dtype=float)
    coords = np.asarray(tdt[:, :3], dtype=float)
    grid = np.unique(coords[:, 0])
    if grid.size < 2:
        raise ValueError("Unexpected thermodynamic grid size.")
    dense_indices = np.rint(coords / float(grid[1] - grid[0])).astype(int) - 1
    if np.any(dense_indices < 0):
        raise ValueError("Encountered invalid negative dense indices while parsing TDT.")
    shape = (grid.size, grid.size, grid.size)

    train_grid_indices = np.unique(np.concatenate([np.arange(0, grid.size, int(train_stride)), [grid.size - 1]]))
    train_valid_mask = np.all(np.isin(dense_indices, train_grid_indices), axis=1)

    values = {key: np.asarray(tdt[:, spec.column], dtype=float) for key, spec in THERMO_QUANTITY_SPECS.items()}

    return AuthorTTMDataset(
        root=root,
        temperature=float(temperature),
        shape=shape,
        grid=grid,
        coords=coords,
        dense_indices=dense_indices,
        values=values,
        train_valid_mask=train_valid_mask,
        train_grid_indices=train_grid_indices,
    )


@dataclass
class OfficialTTMModel:
    rank: int
    A: tuple[np.ndarray, np.ndarray, np.ndarray]
    Ad: tuple[np.ndarray, np.ndarray, np.ndarray]
    Add: tuple[np.ndarray, np.ndarray, np.ndarray]
    alpha: float
    source_path: Path

    @classmethod
    def load(cls, dataset_root: str | Path, rank: int) -> "OfficialTTMModel":
        source_path = Path(dataset_root) / "2_TTM_computation" / "output" / f"TTMR{int(rank):02d}.mat"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing official TTM model file: {source_path}")
        payload = scipy.io.loadmat(source_path, squeeze_me=True, struct_as_record=False)
        A = tuple(np.asarray(x, dtype=float) for x in payload["A"])
        Ad = tuple(np.asarray(x, dtype=float) for x in payload["Ad"])
        Add = tuple(np.asarray(x, dtype=float) for x in payload["Add"])
        alpha = float(payload["alpha"])
        return cls(rank=int(rank), A=A, Ad=Ad, Add=Add, alpha=alpha, source_path=source_path)

    def _row_lookup(self, coords: np.ndarray) -> np.ndarray:
        rows = np.rint(np.asarray(coords, dtype=float) * 10000.0).astype(int)
        if np.any(rows < 0) or np.any(rows > 10000):
            raise ValueError("Coordinate values are outside the official TTM interpolation range [0, 1].")
        return rows

    def predict_preprocessed(self, coords: np.ndarray, quantity_key: str) -> np.ndarray:
        spec = get_quantity_spec(quantity_key)
        rows = self._row_lookup(coords)
        factors = [self.A[0][rows[:, 0]], self.A[1][rows[:, 1]], self.A[2][rows[:, 2]]]
        if len(spec.modes) == 1:
            mode = int(spec.modes[0]) - 1
            factors[mode] = self.Ad[mode][rows[:, mode]]
        elif len(spec.modes) == 2:
            a = int(spec.modes[0]) - 1
            b = int(spec.modes[1]) - 1
            if a == b:
                factors[a] = self.Add[a][rows[:, a]]
            else:
                factors[a] = self.Ad[a][rows[:, a]]
                factors[b] = self.Ad[b][rows[:, b]]
        pred = np.sum(factors[0] * factors[1] * factors[2], axis=1)
        if quantity_key == "gibbs":
            pred = pred + float(self.alpha)
        return np.asarray(pred, dtype=float)

    def predict(self, coords: np.ndarray, quantity_key: str, *, preprocessed: bool, temperature: float) -> np.ndarray:
        pred = self.predict_preprocessed(coords, quantity_key)
        if preprocessed:
            return pred
        spec = get_quantity_spec(quantity_key)
        return reconstruct_quantity(pred, coords, float(temperature), spec.modes)

    def evaluate_dataset(
        self,
        dataset: AuthorTTMDataset,
        quantity_key: str,
        *,
        preprocessed: bool,
        scope_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        truth = dataset.quantity_values(quantity_key, preprocessed=preprocessed)
        pred = self.predict(dataset.coords, quantity_key, preprocessed=preprocessed, temperature=dataset.temperature)
        if scope_mask is not None:
            idx = np.asarray(scope_mask, dtype=bool)
            truth = truth[idx]
            pred = pred[idx]
        abs_err = np.abs(truth - pred)
        summary = {
            "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))) if truth.size else float("nan"),
            "mae": float(np.mean(abs_err)) if truth.size else float("nan"),
            "max_abs_error": float(np.max(abs_err)) if truth.size else float("nan"),
            "relative": summarize_distribution(compute_error_vector(truth, pred, "relative")),
            "range": summarize_distribution(compute_error_vector(truth, pred, "range")),
            "absolute": summarize_distribution(abs_err),
        }
        return {
            "pred": pred,
            "truth": truth,
            "summary": summary,
        }
