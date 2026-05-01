from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from baseline.baseline_cheb_cp import ChebCPCompletion
from common import CompletionResult, TensorCompletionBaseline


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
    return float(np.linalg.norm((p - t).ravel()) / (np.linalg.norm(t.ravel()) + 1e-12))


@dataclass
class ComplexChebCPCompletion(TensorCompletionBaseline):
    rank: int = 5
    number_of_steps: int = 10
    tol_for_step: float = 1e-5
    lambda_all: float = 0.0
    seed: int = 179
    tolerance: int = 500
    source_file: str = "../completion_linalg_tensor_all.py"
    verbose: bool = False
    rank_eval: bool = False
    rank_nest: bool = False
    nest_iters: int = 5
    n_workers: int | None = None
    validation_size: float = 0.0

    def _build_real_model(self, verbose: bool) -> ChebCPCompletion:
        return ChebCPCompletion(
            rank=self.rank,
            number_of_steps=self.number_of_steps,
            tol_for_step=self.tol_for_step,
            lambda_all=self.lambda_all,
            seed=self.seed,
            tolerance=self.tolerance,
            source_file=self.source_file,
            verbose=verbose,
            rank_eval=self.rank_eval,
            rank_nest=self.rank_nest,
            nest_iters=self.nest_iters,
            n_workers=self.n_workers,
            validation_size=self.validation_size,
        )

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if not np.iscomplexobj(observed_tensor):
            return self._build_real_model(self.verbose).fit_transform(
                observed_tensor=observed_tensor,
                mask=mask,
                full_tensor=full_tensor,
            )

        t0 = time.time()
        if self.verbose:
            print(
                f"[ComplexCheb] START rank={self.rank} lambda={self.lambda_all} val={self.validation_size}",
                flush=True,
            )

        real_model = self._build_real_model(verbose=self.verbose)
        imag_model = self._build_real_model(verbose=self.verbose)

        full_real = None if full_tensor is None else np.real(full_tensor)
        full_imag = None if full_tensor is None else np.imag(full_tensor)

        if self.verbose:
            print("[ComplexCheb] phase=real START", flush=True)
        res_real = real_model.fit_transform(np.real(observed_tensor), mask, full_real)
        if self.verbose:
            print("[ComplexCheb] phase=real DONE", flush=True)

        if self.verbose:
            print("[ComplexCheb] phase=imag START", flush=True)
        res_imag = imag_model.fit_transform(np.imag(observed_tensor), mask, full_imag)
        if self.verbose:
            print("[ComplexCheb] phase=imag DONE", flush=True)

        pred = np.asarray(res_real.tensor, dtype=float) + 1j * np.asarray(res_imag.tensor, dtype=float)
        train_metrics = test_metrics = None
        if full_tensor is not None:
            truth = np.asarray(full_tensor, dtype=np.complex128)
            obs = mask.astype(bool)
            train_metrics = {"rlne": _complex_rlne(truth, pred, obs)}
            test_metrics = {"rlne": _complex_rlne(truth, pred, ~obs)}

        if self.verbose:
            msg = f"[ComplexCheb] DONE time={time.time() - t0:.2f}s"
            if test_metrics is not None and np.isfinite(test_metrics.get("rlne", np.nan)):
                msg += f" rlne={float(test_metrics['rlne']):.6f}"
            print(msg, flush=True)

        return CompletionResult(
            tensor=pred,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "real_history": (res_real.info or {}).get("history"),
                "imag_history": (res_imag.info or {}).get("history"),
            },
        )
