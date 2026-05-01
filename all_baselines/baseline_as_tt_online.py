from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from all_baselines.baseline_tt_als import TTALS
from common import CompletionResult, TensorCompletionBaseline, all_metrics


def _mape_ratio(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    idx = np.abs(y_true) > eps
    if not np.any(idx):
        return float("nan")
    return float(np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])))


@dataclass
class ASTTOnline(TensorCompletionBaseline):
    """Budgeted online AS+TT-style baseline with active point acquisition."""

    tt_rank: int = 12
    online_steps: int = 8
    inner_max_iter: int = 20
    tol: float = 1e-5
    budget_fraction_cap: float = 1.0
    random_state: int = 0
    use_runner_budget: bool = True
    init_fraction: float = 0.15
    verbose: bool = False

    @staticmethod
    def _raw_tt_prediction(model: TTALS, result: CompletionResult) -> np.ndarray:
        if result.factors:
            return np.asarray(model._tt_to_tensor(result.factors), dtype=float)
        return np.asarray(result.tensor, dtype=float)

    def _empty_result(
        self,
        shape: tuple[int, ...],
        n_total: int,
        available_budget: int,
        budget_cap: int,
    ) -> CompletionResult:
        pred = np.zeros(shape, dtype=float)
        info = {
            "history": [],
            "effective_observed_points": 0,
            "available_observed_points": int(available_budget),
            "budget_cap_points": int(budget_cap),
            "budget_cap_fraction": float(self.budget_fraction_cap),
            "online_steps": int(self.online_steps),
            "inner_max_iter": int(self.inner_max_iter),
            "budget_mode": "runner_budget" if self.use_runner_budget else "fraction_cap",
            "queried_flat_indices": [],
        }
        return CompletionResult(tensor=pred, factors=None, train_metrics=None, test_metrics=None, info=info)

    def _resolve_budget(self, n_total: int, available_budget: int) -> int:
        if self.use_runner_budget:
            return max(0, int(available_budget))
        frac_cap = max(1, int(round(float(self.budget_fraction_cap) * n_total)))
        return max(0, min(int(available_budget), frac_cap))

    def _fit_with_fixed_observations(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        budget_cap: int,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        obs_tensor = np.asarray(observed_tensor, dtype=float)
        obs_mask = np.asarray(mask, dtype=bool)
        shape = obs_tensor.shape
        n_total = int(np.prod(shape))

        obs_indices = np.argwhere(obs_mask)
        n_observed_available = int(obs_indices.shape[0])
        budget_cap = min(int(budget_cap), n_observed_available)

        rng = np.random.default_rng(int(self.random_state))
        if n_observed_available > budget_cap:
            chosen = rng.choice(n_observed_available, size=budget_cap, replace=False)
            obs_indices = obs_indices[chosen]

        cur_mask = np.zeros(shape, dtype=bool)
        cur_obs = np.zeros(shape, dtype=float)
        pred = np.zeros(shape, dtype=float)
        history: list[dict[str, float | int | str]] = []

        ordered = obs_indices[rng.permutation(obs_indices.shape[0])] if obs_indices.size else obs_indices
        n_steps = max(1, min(int(self.online_steps), max(1, ordered.shape[0])))
        chunks = np.array_split(np.arange(ordered.shape[0]), n_steps)
        rank_eff = max(1, min(int(self.tt_rank), int(min(shape))))

        for step_idx, chunk in enumerate(chunks, start=1):
            if chunk.size == 0:
                continue
            add_indices = ordered[chunk]
            idx_tuple = tuple(add_indices.T)
            cur_mask[idx_tuple] = True
            cur_obs[idx_tuple] = obs_tensor[idx_tuple]

            model = TTALS(
                tt_rank=rank_eff,
                max_iter=int(self.inner_max_iter),
                tol=float(self.tol),
                verbose=bool(self.verbose),
            )
            res = model.fit_transform(observed_tensor=cur_obs, mask=cur_mask, full_tensor=None)
            pred = self._raw_tt_prediction(model, res)
            history.append(
                {
                    "step": int(step_idx),
                    "observed_points": int(np.sum(cur_mask)),
                    "observed_fraction": float(np.sum(cur_mask) / max(1, n_total)),
                    "new_queries": int(chunk.size),
                    "acquisition": "preselected_mask_fallback",
                }
            )

        train_metrics = test_metrics = None
        if full_tensor is not None:
            train_metrics = all_metrics(full_tensor, pred, cur_mask)
            test_metrics = all_metrics(full_tensor, pred, ~cur_mask)

        info = {
            "history": history,
            "effective_observed_points": int(np.sum(cur_mask)),
            "available_observed_points": int(n_observed_available),
            "budget_cap_points": int(budget_cap),
            "budget_cap_fraction": float(self.budget_fraction_cap),
            "online_steps": int(self.online_steps),
            "inner_max_iter": int(self.inner_max_iter),
            "tt_rank": int(rank_eff),
            "budget_mode": "preselected_mask_fallback",
            "queried_flat_indices": np.flatnonzero(cur_mask.reshape(-1)).astype(int).tolist(),
        }
        return CompletionResult(
            tensor=pred,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info,
        )

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        obs_tensor = np.asarray(observed_tensor, dtype=float)
        input_mask = np.asarray(mask, dtype=bool)
        shape = obs_tensor.shape
        n_total = int(np.prod(shape))
        available_budget = int(np.sum(input_mask))
        budget_cap = self._resolve_budget(n_total=n_total, available_budget=available_budget)

        if budget_cap <= 0:
            return self._empty_result(shape, n_total, available_budget, budget_cap)

        if full_tensor is None:
            return self._fit_with_fixed_observations(obs_tensor, input_mask, budget_cap, full_tensor=None)

        rng = np.random.default_rng(int(self.random_state))
        oracle = np.asarray(full_tensor, dtype=float)
        queried_mask = np.zeros(shape, dtype=bool)
        queried_obs = np.zeros(shape, dtype=float)
        pred = np.zeros(shape, dtype=float)
        pred_prev: np.ndarray | None = None
        history: list[dict[str, float | int | str]] = []

        rank_eff = max(1, min(int(self.tt_rank), int(min(shape))))
        n_steps = max(1, min(int(self.online_steps), int(budget_cap)))
        init_points = min(
            int(budget_cap),
            max(1, int(round(float(self.init_fraction) * budget_cap)), 2 * rank_eff),
        )

        initial_flat = rng.choice(n_total, size=init_points, replace=False)
        queried_mask.reshape(-1)[initial_flat] = True
        queried_obs.reshape(-1)[initial_flat] = oracle.reshape(-1)[initial_flat]
        pending_refit = True

        for step_idx in range(1, n_steps + 1):
            if pending_refit:
                model = TTALS(
                    tt_rank=rank_eff,
                    max_iter=int(self.inner_max_iter),
                    tol=float(self.tol),
                    verbose=bool(self.verbose),
                )
                res = model.fit_transform(observed_tensor=queried_obs, mask=queried_mask, full_tensor=None)
                pred = self._raw_tt_prediction(model, res)
                pending_refit = False

            observed_now = int(np.sum(queried_mask))
            history.append(
                {
                    "step": int(step_idx),
                    "observed_points": observed_now,
                    "observed_fraction": float(observed_now / max(1, n_total)),
                    "new_queries": 0,
                    "acquisition": "change_score" if pred_prev is not None else "random_warm_start",
                }
            )

            if observed_now >= budget_cap:
                break

            remaining_budget = int(budget_cap - observed_now)
            remaining_steps = max(1, int(n_steps - step_idx))
            batch_size = max(1, int(np.ceil(remaining_budget / remaining_steps)))

            flat_unqueried = np.flatnonzero(~queried_mask.reshape(-1))
            if flat_unqueried.size == 0:
                break

            if pred_prev is None:
                if np.any(queried_mask):
                    center = float(np.mean(queried_obs[queried_mask]))
                    scores = np.abs(pred.reshape(-1)[flat_unqueried] - center)
                else:
                    scores = np.abs(pred.reshape(-1)[flat_unqueried])
            else:
                scores = np.abs(pred.reshape(-1)[flat_unqueried] - pred_prev.reshape(-1)[flat_unqueried])

            scores = scores + 1e-12 * rng.standard_normal(scores.shape[0])
            if batch_size >= flat_unqueried.size:
                chosen_flat = flat_unqueried
            else:
                part = np.argpartition(scores, -batch_size)[-batch_size:]
                chosen_flat = flat_unqueried[part[np.argsort(scores[part])[::-1]]]

            queried_mask.reshape(-1)[chosen_flat] = True
            queried_obs.reshape(-1)[chosen_flat] = oracle.reshape(-1)[chosen_flat]
            history[-1]["new_queries"] = int(chosen_flat.size)

            pred_prev = pred.copy()
            pending_refit = True

        if pending_refit:
            model = TTALS(
                tt_rank=rank_eff,
                max_iter=int(self.inner_max_iter),
                tol=float(self.tol),
                verbose=bool(self.verbose),
            )
            res = model.fit_transform(observed_tensor=queried_obs, mask=queried_mask, full_tensor=None)
            pred = self._raw_tt_prediction(model, res)

        pred_before_projection = pred.copy()
        approx_full_mask = np.ones(shape, dtype=bool)
        approx_full_metrics = all_metrics(oracle, pred_before_projection, approx_full_mask)
        approx_hidden_metrics = all_metrics(oracle, pred_before_projection, ~queried_mask)
        approx_train_metrics = all_metrics(oracle, pred_before_projection, queried_mask)
        approx_full_mape = _mape_ratio(oracle[approx_full_mask], pred_before_projection[approx_full_mask])
        approx_hidden_mape = _mape_ratio(oracle[~queried_mask], pred_before_projection[~queried_mask])

        pred[queried_mask] = queried_obs[queried_mask]
        train_metrics = all_metrics(oracle, pred, queried_mask)
        test_metrics = all_metrics(oracle, pred, ~queried_mask)

        info = {
            "history": history,
            "effective_observed_points": int(np.sum(queried_mask)),
            "available_observed_points": int(available_budget),
            "budget_cap_points": int(budget_cap),
            "budget_cap_fraction": float(self.budget_fraction_cap),
            "online_steps": int(self.online_steps),
            "inner_max_iter": int(self.inner_max_iter),
            "tt_rank": int(rank_eff),
            "budget_mode": "runner_budget" if self.use_runner_budget else "fraction_cap",
            "oracle_queries": int(np.sum(queried_mask)),
            "approx_metrics_full": approx_full_metrics,
            "approx_metrics_hidden_before_projection": approx_hidden_metrics,
            "approx_metrics_train_before_projection": approx_train_metrics,
            "approx_mape_full": float(approx_full_mape),
            "approx_mape_hidden_before_projection": float(approx_hidden_mape),
            "queried_flat_indices": np.flatnonzero(queried_mask.reshape(-1)).astype(int).tolist(),
        }
        return CompletionResult(
            tensor=pred,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info=info,
        )
