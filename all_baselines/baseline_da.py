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
class DailyAverageBaseline(TensorCompletionBaseline):
    """
    DA baseline (daily average) for traffic tensors shaped as:
    (road_segment, day, time_step).
    """

    verbose: bool = False

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if observed_tensor.ndim != 3:
            raise ValueError("DailyAverageBaseline expects a 3D tensor: (segment, day, time).")

        observed = mask.astype(bool)
        missing = ~observed
        X = observed_tensor.astype(float).copy()

        # Mean over day axis for each (segment, time) pair.
        sum_st = np.sum(np.where(observed, X, 0.0), axis=1)  # [segment, time]
        cnt_st = np.sum(observed, axis=1)  # [segment, time]
        mean_st = np.divide(
            sum_st,
            np.maximum(cnt_st, 1),
            out=np.zeros_like(sum_st, dtype=float),
            where=cnt_st > 0,
        )

        # Fallbacks when a whole (segment, time) track is missing.
        global_mean = float(np.mean(X[observed])) if np.any(observed) else 0.0
        time_mean = np.divide(
            np.sum(sum_st, axis=0),
            np.maximum(np.sum(cnt_st, axis=0), 1),
            out=np.full(mean_st.shape[1], global_mean, dtype=float),
            where=np.sum(cnt_st, axis=0) > 0,
        )
        for s in range(mean_st.shape[0]):
            miss_st = cnt_st[s] == 0
            if np.any(miss_st):
                mean_st[s, miss_st] = time_mean[miss_st]

        day_iter = tqdm(range(X.shape[1]), desc="[DA] fill days", leave=False) if self.verbose else range(X.shape[1])
        for d in day_iter:
            miss_d = missing[:, d, :]
            if np.any(miss_d):
                X[:, d, :][miss_d] = mean_st[miss_d]

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, X, observed)
            test_metrics = all_metrics(full_tensor, X, missing)

        return CompletionResult(
            tensor=X,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={"note": "Daily average baseline on (segment, day, time) tensor"},
        )

