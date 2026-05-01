from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


@dataclass
class KNNBaseline(TensorCompletionBaseline):
    """
    kNN baseline for traffic tensor completion.
    Uses segment-level neighbors over flattened temporal features.
    """

    n_neighbors: int = 10
    eps: float = 1e-8
    verbose: bool = False

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if observed_tensor.ndim != 3:
            raise ValueError("KNNBaseline expects a 3D tensor: (segment, day, time).")

        observed = mask.astype(bool)
        missing = ~observed
        shape = observed_tensor.shape

        data = observed_tensor.reshape(shape[0], -1).astype(float)
        obs = observed.reshape(shape[0], -1)

        # Initial fill with feature-wise mean for distance computation.
        col_sum = np.sum(np.where(obs, data, 0.0), axis=0)
        col_cnt = np.sum(obs, axis=0)
        col_mean = np.divide(
            col_sum,
            np.maximum(col_cnt, 1),
            out=np.zeros_like(col_sum, dtype=float),
            where=col_cnt > 0,
        )
        global_mean = float(np.mean(data[obs])) if np.any(obs) else 0.0
        col_mean[col_cnt == 0] = global_mean

        filled = data.copy()
        miss_pos = ~obs
        if np.any(miss_pos):
            filled[miss_pos] = np.broadcast_to(col_mean, filled.shape)[miss_pos]

        # Segment-level distances.
        norms = np.sum(filled * filled, axis=1, keepdims=True)
        dist2 = np.maximum(norms + norms.T - 2.0 * (filled @ filled.T), 0.0)
        dist = np.sqrt(dist2)
        np.fill_diagonal(dist, np.inf)
        neighbor_order = np.argsort(dist, axis=1)

        out = filled.copy()
        row_iter = tqdm(range(shape[0]), desc="[kNN] segments", leave=False) if self.verbose else range(shape[0])
        for i in row_iter:
            miss_cols = np.where(~obs[i])[0]
            if miss_cols.size == 0:
                continue

            neighbors = neighbor_order[i]
            for j in miss_cols:
                valid = neighbors[obs[neighbors, j]]
                if valid.size == 0:
                    out[i, j] = col_mean[j]
                    continue
                use = valid[: self.n_neighbors]
                w = 1.0 / (dist[i, use] + self.eps)
                vals = data[use, j]
                out[i, j] = float(np.dot(w, vals) / np.sum(w))

        pred = out.reshape(shape)
        pred[observed] = observed_tensor[observed]

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, pred, observed)
            test_metrics = all_metrics(full_tensor, pred, missing)

        return CompletionResult(
            tensor=pred,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"n_neighbors": int(self.n_neighbors)},
        )

