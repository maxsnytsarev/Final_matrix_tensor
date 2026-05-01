from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import importlib.util
import types
import sys
import time
import numpy as np

from common import CompletionResult, TensorCompletionBaseline, all_metrics, cp_to_tensor


@dataclass
class ChebCPCompletion(TensorCompletionBaseline):
    rank: int = 5
    number_of_steps: int = 10
    tol_for_step: float = 1e-5
    lambda_all: float = 0.0
    seed: int = 179
    tolerance: int = 500
    source_file: str = '../completion_linalg_tensor_all.py'
    verbose: bool = False
    rank_eval: bool = False
    rank_nest: bool = False
    nest_iters: int = 5
    strict_rank: bool = False
    allow_rank_growth: bool = True
    n_workers: int | None = None
    validation_size: float = 0.0

    def _load_source_module(self):
        path = Path(self.source_file)
        if not path.exists():
            raise FileNotFoundError(f'Could not find source file: {path}')
        if 'tensorly.cp_tensor' not in sys.modules:
            tensorly_mod = types.ModuleType('tensorly')
            cp_tensor_mod = types.ModuleType('tensorly.cp_tensor')
            cp_tensor_mod.cp_to_tensor = lambda arg: cp_to_tensor(arg[1], arg[0])
            tensorly_mod.cp_tensor = cp_tensor_mod
            sys.modules['tensorly'] = tensorly_mod
            sys.modules['tensorly.cp_tensor'] = cp_tensor_mod
        spec = importlib.util.spec_from_file_location('user_cheb_completion', str(path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def fit_transform(self, observed_tensor: np.ndarray, mask: np.ndarray, full_tensor: np.ndarray | None = None) -> CompletionResult:
        module = self._load_source_module()
        if self.n_workers is not None and hasattr(module, "n_workers"):
            module.n_workers = max(1, int(self.n_workers))
        model_tag = f"cheb_cp_r{int(self.rank)}"
        if self.verbose:
            print(
                f"[model] {model_tag} START "
                f"(lambda={float(self.lambda_all)}, val={float(self.validation_size)})",
                flush=True,
            )
        t0 = time.time()
        indices = np.argwhere(mask)
        values = observed_tensor[mask].astype(float)
        kwargs = dict(
            indices=indices,
            values=values,
            shape=observed_tensor.shape,
            rank=self.rank,
            number_of_steps=self.number_of_steps,
            tol_for_step=self.tol_for_step,
            lambda_all=self.lambda_all,
            seed=self.seed,
            return_compl=True,
            return_history=True,
            eval=self.verbose,
            TQDM=self.verbose,
            tolerance=self.tolerance,
            rank_eval=self.rank_eval,
            rank_nest=self.rank_nest,
            nest_iters=self.nest_iters,
            strict_rank=self.strict_rank,
            allow_rank_growth=self.allow_rank_growth,
            validation_size=self.validation_size,
        )
        try:
            if self.verbose:
                factors, history, reconstructed = module.approximateLOO(**kwargs)
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    factors, history, reconstructed = module.approximateLOO(**kwargs)
        except Exception as exc:
            if self.verbose:
                print(f"[model] {model_tag} FAIL error={exc}", flush=True)
            raise
        train_metrics = test_metrics = None
        if full_tensor is not None:
            obs = mask.astype(bool)
            train_metrics = all_metrics(full_tensor, reconstructed, obs)
            test_metrics = all_metrics(full_tensor, reconstructed, ~obs)
        if self.verbose:
            if test_metrics is not None and np.isfinite(test_metrics.get("rmse", np.nan)):
                print(
                    f"[model] {model_tag} DONE rmse={float(test_metrics['rmse']):.4f} "
                    f"time={float(time.time() - t0):.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[model] {model_tag} DONE time={float(time.time() - t0):.2f}s",
                    flush=True,
                )
        return CompletionResult(
            tensor=reconstructed,
            factors=factors,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={'history': history},
        )
