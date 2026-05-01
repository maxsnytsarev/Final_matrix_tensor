from copy import deepcopy

from joblib import Parallel, delayed


from tqdm.auto import tqdm
import scipy
import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, find, vstack, hstack
import os
from pathlib import Path
import subprocess
import pandas as pd
from scipy.optimize import linprog
# from scikits.umfpack import splu
# import scikits.umfpack as umf
# import sparseqr
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, find
import random
import time
from scipy import sparse
import sys
from IPython.display import clear_output
# import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from tensorly.cp_tensor import cp_to_tensor


n_workers = 22 #!!!!!!!!

DEBUG = False

# log_display = display("", display_id=True)
class StopError(Exception):
    def __init__(self, message):
        self.line = message
    def raise_error(self):
        print(f"Changed to lstsq at line {self.line}")


# находит такой вектор q, что [Q q] - ортогональная матрица, где Q - матрица с ортнормированными столбцами

def khatri_rao(matrices):
    res = matrices[0]
    for m in matrices[1:]:
        res = np.einsum('ir,jr->ijr', res, m).reshape(-1, res.shape[1])
    return res

def get_mode(indices, values, shape, mode):
    rows = indices[:, mode]

    other_dims = [i for i in range(len(shape)) if i != mode]
    other_indices = indices[:, other_dims]
    other_shape = [shape[i] for i in other_dims]

    cols = np.ravel_multi_index(other_indices.T, other_shape, order='C')

    return csc_matrix((values, (rows, cols)), shape=(shape[mode], np.prod(other_shape)))

def compute_orthogonal_complement(V: np.ndarray):
    n = V.shape[1]
    v = np.eye(1, n + 1, n)[0]
    projections = np.zeros(n + 1)
    for i in range(n):
        a_i = V[:, i]
        projections += a_i[-1] * a_i
    u = v - projections
    u = u / np.sqrt(np.dot(u, u))
    return u.reshape(-1, 1)

def best_uniform_approximation0(Q: np.ndarray, q: np.ndarray, R: np.ndarray, a:np.ndarray) -> np.ndarray:
    c_ = ((a.T @ q)) / np.linalg.norm(q, ord=1)
    w = float(c_[0]) * np.sign(q)
    u_ = scipy.linalg.solve_triangular(R, Q.T @ (a - w))
    return u_

# ищет оптимальную для замены строчку
def optimal_replacement(Q: np.ndarray, q: np.ndarray, R: np.ndarray, h: np.ndarray, e: float, a: np.ndarray) -> int:
    r = R.shape[1]
    g = scipy.linalg.solve_triangular(R.T, h, lower=True)
    y = Q @ g
    error_ = -1
    k_ = -1
    Q_ = np.column_stack((Q, q))
    Q_T = Q.T
    Base = np.eye(r + 1)
    for k in range(0, r + 1):
        base = Base[:, k:k+1]
        cur_a = a.copy()
        cur_a[k] = e
        q_k = Q_T[:, k:k+1]
        y_k = g.T @ q_k
        q_k_strich = Q_.T[:, k][-1]
        q_strich = q_k_strich * (base - y) + (y_k * q)
        error = np.abs(np.dot(q_strich.T, cur_a)) / np.sum(np.abs(q_strich))
        if error > error_:
            error_ = error
            k_ = k
    return k_

def cheb_norm(obj):
    return np.max(np.abs(obj.flatten()))

errors = 0

def nonzero(a, add=False, rank=60):
    if not add:
        return a.nonzero()[0]
    else:
        n = a.shape[0]
        head = a[:(n - rank)].nonzero()[0]
        tail = np.arange(n - rank, n, dtype=int)
        return np.concatenate([head, tail]).astype(int)


def PUREalgorithm(V: np.ndarray, a: csr_matrix, J: list[int]) -> np.ndarray:
    r = len(J) - 1
    V_ = V[J].copy()
    a_ = a[J].copy()
    a_ = a_.toarray()
    Q, R = np.linalg.qr(V_)
    q = compute_orthogonal_complement(Q)
    u_ = best_uniform_approximation0(Q, q, R, a_)
    w = a - V @ u_
    while np.max(np.abs(w[J])) < np.max(np.abs(w)):
        j = np.argmax(np.abs(w))

        k_ = optimal_replacement(Q, q, R, V[j:j+1, :].T, float(a[j].toarray()), a[J].toarray())
        a_[k_] = float(a[j].toarray())


        h = V[j:j+1, :].T
        ek = np.eye(1, r + 1, k_).T

        Q, R = scipy.linalg.qr_update(Q, R, ek, h - V_.T @ ek)
        q = compute_orthogonal_complement(Q)
        V_[k_:k_+1, :] = V[j:j+1, :]

        J[k_] = j
        u_ = best_uniform_approximation0(Q, q, R, a_)
        w = np.array(a - V @ u_)
    return u_


def PUREprocess_column_for_V(i, A, U, r):
    a = A[:, i]
    first_set = np.random.choice(A.shape[0], r + 1, replace=False)
    v = PUREalgorithm(U, csr_matrix(a), first_set)
    return v

def PUREprocess_column_for_U(i, A_t, V, r):
    a = A_t[:, i]
    first_set = np.random.choice(A_t.shape[0], r + 1, replace=False)
    u = PUREalgorithm(V, csr_matrix(a), first_set)
    return u

def algorithm_modified(rank, V: np.ndarray, a: csr_matrix, J: list[int], index_r, add=False, tolerance=500) -> np.ndarray:
    global errors
    steps = 0
    I_obs = nonzero(a, add=add, rank=rank)

    #
    J = [j for j in J if j in set(I_obs)]
    if len(J) == 0:
        return np.zeros(V.shape[1])
    V_ = V[J].copy()
    a_loc = a[J].copy()
    a_loc = a_loc.toarray().flatten()
    Q, R = np.linalg.qr(V_)
    q = compute_orthogonal_complement(Q)
    u = best_uniform_approximation0(Q, q, R, a_loc.reshape(-1, 1))
    a_obs = a[I_obs].toarray()
    V_obs = V[I_obs]
    I_obs = np.array(I_obs)
    while True:
        steps += 1
        if steps > tolerance:
            errors += 1
            raise StopError(f"{index_r}")
        w = a_obs - V_obs @ u

        J_obs = [np.where(I_obs == j)[0][0] for j in J]
        if np.max(np.abs(w[J_obs])) >= np.max(np.abs(w)):
            break

        j_idx = np.argmax(np.abs(w))
        j_new = I_obs[j_idx]
        val = float(a_obs[j_idx])
        k_replace = optimal_replacement(Q, q, R, V[j_new:j_new+1, :].T, val, a_loc.copy())
        if k_replace == -1:
            break
        a_loc[k_replace] = val
        h = V[j_new:j_new+1, :].T
        ek = np.zeros((len(J), 1))
        ek[k_replace, 0] = 1
        v = h - V_.T @ ek
        if np.allclose(v, 0):
            break

        Q, R = scipy.linalg.qr_update(Q, R, ek, h - V_.T @ ek)
        q = compute_orthogonal_complement(Q)
        V_[k_replace:k_replace+1, :] = V[j_new:j_new+1, :]
        J[k_replace] = j_new

        u = best_uniform_approximation0(Q, q, R, a_loc.reshape(-1, 1))
    return u.flatten()

c = []

def LP(V: np.ndarray, a: np.ndarray):
    """
    Решает min_u ||a - V u||_∞ через LP: min t subject to |a_i - (V u)_i| <= t.
    """
    n, r = V.shape
    c = np.zeros(r + 1)
    c[-1] = 1

    A_ub = np.zeros((2 * n, r + 1))
    b_ub = np.zeros(2 * n)

    A_ub[:n, :r] = V
    A_ub[:n, -1] = -1
    b_ub[:n] = a

    A_ub[n:, :r] = -V
    A_ub[n:, -1] = -1
    b_ub[n:] = -a

    bounds = [(None, None)] * r + [(0, None)]  # u свободны, t >= 0

    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError("LP solver failed: " + res.message)

    return res.x[:r], res.x[-1]

def process_column_for_V(i, U: np.ndarray, A: csc_matrix, r: int, type_appr: str, add=False, TQDM=True, tolerance=500, seed=179) -> np.ndarray:
    global DEBUG
    np.random.seed(seed + i)
    random.seed(seed + i)
    def return_v_small_rec(a, U, I_obs, type_appr):
        U_obs = U[I_obs, :]
        y_obs = a[I_obs].toarray().flatten()
        if U_obs.shape[0] == 0:
            v = np.zeros(r)
        else:
            if type_appr == 'LP':
                try:
                    v, t = LP(U_obs, y_obs)
                except Exception as e:
                    v, *_ = np.linalg.lstsq(U_obs, y_obs, rcond=None)
                # print(np.max(np.abs(U_obs @ v - y_obs)))
            else:
                v, *_ = np.linalg.lstsq(U_obs, y_obs, rcond=None)
        return v

    # global cnt
    local_errors = 0  # Создаем локальный счетчик

    m, n = A.shape
    a = A[:, i]
    I_obs = nonzero(a, add, rank=r)

    if len(I_obs) == 0:
        v = np.zeros(r)
    elif len(I_obs) <= r:
        v = return_v_small_rec(a, U, I_obs, type_appr)
    else:
        c.append(i)
        J = np.random.choice(I_obs, r + 1, replace=False).tolist()
        try:
            v = algorithm_modified(r, U, a, J, i, add=add, tolerance=tolerance)
        except Exception as e:
            print(":(", e)
            if isinstance(e, StopError):
                local_errors += 1  # Увеличиваем счетчик при фаллбэке
                if DEBUG:
                    pass  # Принт здесь в параллельном режиме может ломать вывод
            v = return_v_small_rec(a, U, I_obs, type_appr)

    return v, local_errors  # ВОЗВРАЩАЕМ КОРТЕЖ


def process_column_for_U(i, V: np.ndarray, A_t: csc_matrix, r: int, type_appr: str, add=False, tolerance=500, seed=179) -> np.ndarray:
    global lambda_dop, DEBUG
    np.random.seed(seed + i)
    random.seed(seed + i)
    def return_u_small_rec(a, V, I_obs, type_appr):
        V_obs = V[I_obs, :]
        y_obs = a[I_obs].toarray().flatten()
        if V_obs.shape[0] == 0:
            u = np.zeros(r)
        else:
            if type_appr == "LP":
                try:
                    u, t = LP(V_obs, y_obs)
                except Exception as e:
                    u, *_ = np.linalg.lstsq(V_obs, y_obs, rcond=None)
            else:
                u, *_ = np.linalg.lstsq(V_obs, y_obs, rcond=None)
        return u

    # global cnt
    local_errors = 0  # Создаем локальный счетчик

    n, m = A_t.shape
    a = A_t[:, i]
    I_obs = nonzero(a, add, rank=r)
    if len(I_obs) == 0:
        u = np.zeros(r)
    elif len(I_obs) <= r:
        u = return_u_small_rec(a, V, I_obs, type_appr)
    else:
        J = np.random.choice(I_obs, r + 1, replace=False).tolist()
        try:
            u = algorithm_modified(r, V, a, J, i, add=add, tolerance=tolerance)
        except Exception as e:
            print(":(", e)
            if isinstance(e, StopError):
                local_errors += 1  # Увеличиваем счетчик при фаллбэке
            u = return_u_small_rec(a, V, I_obs, type_appr)

    return u, local_errors  # ВОЗВРАЩАЕМ КОРТЕЖ


def get_V_modified(U: np.ndarray, A: csc_matrix, r: int, type_appr: str, add=False, PURE=False, eval_fall=False, seed=179, tolerance=500) -> np.ndarray:
    global errors  # Оставляем для совместимости с остальным кодом, если нужно
    if not PURE:
        results = Parallel(n_jobs=n_workers)(
            delayed(process_column_for_V)(i, U, A, r, type_appr="LP", add=add, seed=seed, tolerance=tolerance)
            for i in tqdm(range(A.shape[1]), total=A.shape[1], dynamic_ncols=True, leave=False, ncols=10)
        )

        # Разделяем векторы и счетчики ошибок
        vectors = [res[0] for res in results]
        total_errors_in_batch = sum(res[1] for res in results)

        errors += total_errors_in_batch  # Обновляем глобальную переменную в ГЛАВНОМ процессе
        if eval_fall:
            print(f"Фаллбэков на этом шаге: {total_errors_in_batch}. Всего за запуск: {errors}")

        return np.vstack(vectors)
    else:
        # PURE процесс не генерирует StopError в вашем коде,
        # поэтому здесь можно оставить как есть (но убедитесь, что PURE алгоритмы возвращают только векторы)
        results = Parallel(n_jobs=n_workers)(
            delayed(PUREprocess_column_for_V)(i, A, U, r)
            for i in tqdm(range(A.shape[1]), total=A.shape[1], dynamic_ncols=True, leave=False, ncols=10)
        )
        return np.hstack(results).T


def get_U_modified(V: np.ndarray, A_t: csc_matrix, r: int, type_appr: str, add=False, PURE=False, eval_fall=False, tolerance=500) -> np.ndarray:
    global errors
    if not PURE:
        results = Parallel(n_jobs=n_workers)(
            delayed(process_column_for_U)(i, V, A_t, r, type_appr="LP", add=add, tolerance=tolerance)
            for i in tqdm(range(A_t.shape[1]), total=A_t.shape[1], dynamic_ncols=True, leave=False, ncols=10)
        )

        vectors = [res[0] for res in results]
        total_errors_in_batch = sum(res[1] for res in results)

        errors += total_errors_in_batch
        if eval_fall:
            print(f"Фаллбэков на этом шаге: {total_errors_in_batch}. Всего за запуск: {errors}")

        return np.vstack(vectors)
    else:
        results = Parallel(n_jobs=n_workers)(
            delayed(PUREprocess_column_for_U)(i, A_t, V, r)
            for i in tqdm(range(A_t.shape[1]), total=A_t.shape[1], dynamic_ncols=True, leave=False, ncols=10)
        )
        return np.hstack(results).T

def norm(obj, type="l1"):
    # set_rnd()
    if type == "l1":
        return np.linalg.norm(obj)
    elif type == "l2":
        return np.linalg.norm(obj) ** 2
    elif type == "nuc":
        return np.linalg.norm(obj, "nuc")
    elif type == "cheb":
        return np.max(np.abs(obj.flatten()))


def get_start_rnd(A, r, seed=179):
    rng = np.random.default_rng(seed)
    shape_k = A.shape[0]
    factors = rng.standard_normal(size=(shape_k, r))
    if shape_k >= r + 10:
        factors = rng.standard_normal(size=(shape_k, r + 10))
        Q, _ = np.linalg.qr(factors, mode='reduced')
        return Q[:, :r]
    else:
        return factors


def get_start_qr(A, r, eps = 1e-6):
    columns = random.choices(range(A.shape[1]), k=r)
    A_new = A[:, columns].copy().toarray()
    A_new = A_new + eps * np.random.standard_normal(A_new.shape)
    Q_cond, _ = np.linalg.qr(A_new)
    return Q_cond

def get_start_svd(A, r):
    print(A.shape)
    # os.mkdir("U_svds")
    # U0, S, Vh0 = np.linalg.svd(A, full_matrices=False)
    # for rank in tqdm([1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000]):
    #     U = U0[:, :rank]
    #     np.save(f'U_svds/U_start_svd_{rank}.npy', U)
    return None


def tensor_cheb_error(train_indices, train_values, A_reconstructed):
    reconstructed_values = np.array([A_reconstructed[tuple(idx)] for idx in train_indices])
    cheb_error = np.max(np.abs(train_values - reconstructed_values))
    return cheb_error


def tensor_rmse_error(train_indices, train_values, A_reconstructed):
    reconstructed_values = np.array([A_reconstructed[tuple(idx)] for idx in train_indices])
    rmse_error = np.sqrt(np.mean((train_values - reconstructed_values) ** 2))
    return rmse_error





def rank_incr(indices, values, shape, rank = 10, number_of_steps = 1, tol_for_step = 1e-4, type_appr = 'LP', begin='oversample', lambda_all=0.0, tol=1e-3, max_rank=20, using_qr = False, eval=True, seed=179, return_compl=False, return_history=False, ret_best = True, TQDM=True, eval_fall=False, tolerance=500, val_indices=None, val_values=None):
    np.random.seed(seed)
    random.seed(seed)
    n_modes = len(shape)
    all_factors = []
    all_history = []
    all_errors = []
    while rank < max_rank:
        print(f"Approximate LOO with rank {rank}, lambda={lambda_all}, with val {val_indices is not None}")
        factors = []
        history = []
        cur_err = []
        for k in range(n_modes):
            A_k = get_mode(indices, values, shape, k)
            if begin == 'oversample':
                factors.append(get_start_rnd(A_k, rank, seed=seed + 1000 * k))
            else:
                print("wrong begin")

        for j in tqdm(range(number_of_steps), disable=not TQDM):
            for k in tqdm(range(n_modes), disable=not TQDM):
                other_factors = [factors[i] for i in range(n_modes) if i != k]
                V_mode = khatri_rao(other_factors)
                A_k = get_mode(indices, values, shape, k)
                if eval:
                    print(f"Updating mode {k} (dim {A_k.T.shape[0]}, rank {A_k.T.shape[1]}, {rank})")
                if lambda_all != 0:
                    dop = lambda_all * np.eye(rank)
                    V_reg = np.vstack([V_mode, dop])
                    zeros = csc_matrix((rank, A_k.shape[0]))
                    A_k_reg = csc_matrix(vstack([A_k.T, zeros]))
                    new_factor = get_V_modified(V_reg, A_k_reg, rank, type_appr, add=True, eval_fall=eval_fall, seed=seed, tolerance=tolerance)
                else:
                    new_factor = get_V_modified(V_mode, A_k.T, rank, type_appr, add=False, eval_fall=eval_fall, seed=seed, tolerance=tolerance)
                factors[k] = new_factor
            tensor_now = cp_to_tensor((None, factors))
            train_err = tensor_cheb_error(indices, values, tensor_now)
            cur_err.append(train_err)
            score_err = train_err
            if val_indices is not None and val_values is not None and len(val_indices) > 0:
                # score_err = tensor_cheb_error(val_indices, val_values, tensor_now)
                score_err = tensor_rmse_error(val_indices, val_values, tensor_now)
            if eval:
                history.append(train_err)
                print("ChebError(train):", train_err, "ChebError(score):", score_err)
            all_factors.append([f.copy() for f in factors])
            all_history.append(history.copy())
            if ret_best:
                all_errors.append(score_err)
            if len(cur_err) > 1:
                if abs(cur_err[-1] - cur_err[-2]) < tol_for_step:
                    break
        err = tensor_cheb_error(indices, values, cp_to_tensor((None, factors)))
        if err > tol:
            print(f"Increased rank to {rank + 1}, current tolerance: {err}")
            rank += 1
            continue
        else:
            break
    if ret_best:
        best_ind = np.argmin(all_errors)
        print("VALIDATION INFO")
        print(f"Best ind was {best_ind} out of {len(all_errors)}")
        print(f"Validated on {'val' if val_indices is not None else 'train'}. Best val error: {all_errors[best_ind]}, last val error: {all_errors[-1]}")
        print("VALIDATION INFO")
        factors = all_factors[best_ind]
        history = all_history[best_ind]
    else:
        factors = all_factors[-1]
        history = all_history[-1]
    print("="*50)
    for U in factors:
        print(norm(U, "cheb"), U.shape)
    print("="*50)
    if return_history:
        if return_compl:
            return factors, history, cp_to_tensor((None, factors))
        return factors, history, None
    if return_compl:
        return factors, [], cp_to_tensor((None, factors))
    return factors, [], None

def rank_nested(indices, values, shape, rank = 10, number_of_steps = 5, tol_for_step = 1e-4, type_appr = 'LP', begin='oversample', lambda_all=0.0, rank_eval = False, tol=1e-3, nest_iters=5, using_qr = False, eval=True, seed=179, return_compl=False, return_history=False, ret_best = True, TQDM=True, eval_fall=False, tolerance=500):
    np.random.seed(seed)
    random.seed(seed)
    all_factors = []
    all_history = []
    all_errors = []
    n_modes = len(shape)
    cur_values = deepcopy(values)
    last_factors = []
    for iteration in tqdm(range(nest_iters), disable=not TQDM):

        print(f"Nested Approximate LOO with rank {rank}, iter {iteration}")
        factors = []
        history = []
        cur_err = []
        for k in range(n_modes):
            A_k = get_mode(indices, cur_values, shape, k)
            if begin == "oversample":
                factors.append(get_start_rnd(A_k, rank, seed=seed + 1000 * k))

        for j in tqdm(range(number_of_steps), disable=not TQDM):
            for k in tqdm(range(n_modes), disable=not TQDM):
                other_factors = [factors[i] for i in range(n_modes) if i != k]
                V_mode = khatri_rao(other_factors)
                A_k = get_mode(indices, cur_values, shape, k)
                if eval:
                    print(f"Updating mode {k} (dim {A_k.T.shape[0]}, rank {A_k.T.shape[1]}, {rank})")
                if lambda_all != 0:
                    dop = lambda_all * np.eye(rank)
                    V_reg = np.vstack([V_mode, dop])
                    zeros = csc_matrix((rank, A_k.shape[0]))
                    A_k_reg = csc_matrix(vstack([A_k.T, zeros]))
                    new_factor = get_V_modified(V_reg, A_k_reg, rank, type_appr, add=True, eval_fall=eval_fall,
                                                seed=seed, tolerance=tolerance)
                else:
                    new_factor = get_V_modified(V_mode, A_k.T, rank, type_appr, add=False, eval_fall=eval_fall,
                                                seed=seed, tolerance=tolerance)
                factors[k] = new_factor
            cur_err.append(tensor_cheb_error(indices, cur_values, cp_to_tensor((None, factors))))
            err = tensor_cheb_error(indices, cur_values, cp_to_tensor((None, factors)))
            if eval:
                history.append(err)
                print("ChebError:", err)
            all_factors.append([f.copy() for f in factors])
            all_history.append(history.copy())
            if ret_best:
                all_errors.append(err)
            if len(cur_err) > 1:
                if abs(cur_err[-1] - cur_err[-2]) < tol_for_step:
                    break
        err = tensor_cheb_error(indices, cur_values, cp_to_tensor((None, factors)))
        last_factors.append([f.copy() for f in factors])
        if err > tol:
            print(f"Increased nested iteration to {iteration + 1}, current tolerance: {err}")
            cur_tensor = cp_to_tensor((None, factors))
            cur_tensor_at_indices = cur_tensor[tuple(indices.T)]
            cur_values = cur_values - cur_tensor_at_indices
            continue
        else:
            break
    final = np.zeros(shape)
    for factors in last_factors:
        final += cp_to_tensor((None, factors))
    if return_history:
        if return_compl:
            return [], [], final
        return [], [], None
    if return_compl:
        return [], [], final
    return [], [], None


def approximateLOO(indices, values, shape, rank = 10, number_of_steps = 5, tol_for_step = 1e-4, begin='oversample', lambda_all=0.0, rank_eval = False, rank_nest = False, nest_iters=5, tol=1e-3, max_rank=20, using_qr = False, eval=True, seed=179, return_compl=False, return_history=False, ret_best = True, TQDM=True, eval_fall=False, validation_size = 0.1, tolerance=500):
    eval_fall = True
    np.random.seed(seed)
    random.seed(seed)
    train_indices = indices
    train_values = values
    val_indices = None
    val_values = None
    total_len = len(indices)
    val_count = max(1, int(total_len * validation_size))
    if val_count < 10:
        validation_size = 0
    if validation_size > 0 and len(indices) > 1:
        val_count = min(val_count, total_len - 1)
        perm = np.random.permutation(total_len)
        val_ids = perm[:val_count]
        train_ids = perm[val_count:]
        train_indices = indices[train_ids]
        train_values = values[train_ids]
        val_indices = indices[val_ids]
        val_values = values[val_ids]
    if rank_nest:
        factors, history, completed = rank_nested(train_indices, train_values, shape, rank=rank, number_of_steps=number_of_steps,
                                                tol_for_step=tol_for_step, type_appr='LP', begin=begin,
                                                lambda_all=lambda_all, tol=tol, nest_iters=nest_iters, eval=True, seed=seed,
                                                return_compl=return_compl, return_history=return_history,
                                                ret_best=ret_best, TQDM=TQDM, eval_fall=eval_fall, tolerance=tolerance)

    elif rank_eval:
        factors, history, completed = rank_incr(train_indices, train_values, shape, rank=rank, number_of_steps=number_of_steps, tol_for_step=tol_for_step, type_appr='LP', begin=begin, lambda_all=lambda_all, tol=tol, max_rank=max_rank, eval=True, seed=seed, return_compl=return_compl, return_history=return_history, ret_best=ret_best, TQDM=TQDM, eval_fall=eval_fall, tolerance=tolerance, val_indices=val_indices, val_values=val_values)
    else:
        factors, history, completed = rank_incr(train_indices, train_values, shape, rank=rank, number_of_steps=number_of_steps, tol_for_step=tol_for_step, type_appr='LP', begin=begin, lambda_all=lambda_all, tol=tol, max_rank=rank + 1, eval=True, seed=seed, return_compl=return_compl, return_history=return_history, ret_best=ret_best, TQDM=TQDM, eval_fall=eval_fall, tolerance=tolerance, val_indices=val_indices, val_values=val_values)
    return factors, history, completed
