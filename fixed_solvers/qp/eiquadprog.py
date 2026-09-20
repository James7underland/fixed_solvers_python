"""Goldfarb–Idnani dual QP (порт testing/qp/eiquadprog.hpp).

min 0.5 x' G x + g0' x
s.t. CE^T x + ce0 = 0
     CI^T x + ci0 >= 0

G: n x n, g0: n, CE: n x p, ce0: p, CI: n x m, ci0: m, x: n
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

_EPS = sys.float_info.epsilon
_INF = math.inf


def _as_constraint_matrix(matrix, n: int, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return np.zeros((n, 0), dtype=float)
    if arr.ndim == 1:
        if arr.size == n:
            return arr.reshape(n, 1)
        raise ValueError(f"{name} has incompatible shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows, got {arr.shape}")
    return np.array(arr, dtype=float, copy=True)


def _as_vector(vector, size: int, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0 and size == 0:
        return np.zeros(0, dtype=float)
    if arr.size != size:
        if arr.size == 0 and size == 0:
            return np.zeros(0, dtype=float)
        raise ValueError(f"{name} must have length {size}, got {arr.size}")
    return np.array(arr, dtype=float, copy=True)


def distance(a: float, b: float) -> float:
    a1 = abs(a)
    b1 = abs(b)
    if a1 > b1:
        t = b1 / a1
        return a1 * math.sqrt(1.0 + t * t)
    if b1 > a1:
        t = a1 / b1
        return b1 * math.sqrt(1.0 + t * t)
    return a1 * math.sqrt(2.0)


def compute_d(d: np.ndarray, J: np.ndarray, np_vec: np.ndarray) -> None:
    d[:] = J.T @ np_vec


def update_z(z: np.ndarray, J: np.ndarray, d: np.ndarray, iq: int) -> None:
    n = z.size
    if iq >= n:
        z[:] = 0.0
        return
    z[:] = J[:, iq:] @ d[iq:]


def update_r(R: np.ndarray, r: np.ndarray, d: np.ndarray, iq: int) -> None:
    if iq <= 0:
        return
    r[:iq] = solve_triangular(R[:iq, :iq], d[:iq], lower=False, check_finite=False)


def add_constraint(R: np.ndarray, J: np.ndarray, d: np.ndarray, iq: int, R_norm: float) -> tuple[bool, int, float]:
    n = J.shape[0]
    for j in range(n - 1, iq, -1):
        cc = d[j - 1]
        ss = d[j]
        h = distance(cc, ss)
        if h == 0.0:
            continue
        d[j] = 0.0
        ss = ss / h
        cc = cc / h
        if cc < 0.0:
            cc = -cc
            ss = -ss
            d[j - 1] = -h
        else:
            d[j - 1] = h
        xny = ss / (1.0 + cc)
        for k in range(n):
            t1 = J[k, j - 1]
            t2 = J[k, j]
            J[k, j - 1] = t1 * cc + t2 * ss
            J[k, j] = xny * (t1 + J[k, j - 1]) - t2
    iq += 1
    R[:iq, iq - 1] = d[:iq]
    if abs(d[iq - 1]) <= _EPS * R_norm:
        return False, iq, R_norm
    R_norm = max(R_norm, abs(d[iq - 1]))
    return True, iq, R_norm


def delete_constraint(
    R: np.ndarray,
    J: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    p: int,
    iq: int,
    l: int,
) -> int:
    n = R.shape[0]
    qq = 0
    for i in range(p, iq):
        if A[i] == l:
            qq = i
            break
    for i in range(qq, iq - 1):
        A[i] = A[i + 1]
        u[i] = u[i + 1]
        R[:, i] = R[:, i + 1]
    A[iq - 1] = A[iq]
    u[iq - 1] = u[iq]
    A[iq] = 0
    u[iq] = 0.0
    R[:iq, iq - 1] = 0.0
    iq -= 1
    if iq == 0:
        return iq
    for j in range(qq, iq):
        cc = R[j, j]
        ss = R[j + 1, j]
        h = distance(cc, ss)
        if h == 0.0:
            continue
        cc = cc / h
        ss = ss / h
        R[j + 1, j] = 0.0
        if cc < 0.0:
            R[j, j] = -h
            cc = -cc
            ss = -ss
        else:
            R[j, j] = h
        xny = ss / (1.0 + cc)
        for k in range(j + 1, iq):
            t1 = R[j, k]
            t2 = R[j + 1, k]
            R[j, k] = t1 * cc + t2 * ss
            R[j + 1, k] = xny * (t1 + R[j, k]) - t2
        for k in range(n):
            t1 = J[k, j]
            t2 = J[k, j + 1]
            J[k, j] = t1 * cc + t2 * ss
            J[k, j + 1] = xny * (J[k, j] + t1) - t2
    return iq


def solve_quadprog2(chol, c1: float, g0: np.ndarray, CE, ce0, CI, ci0, x: np.ndarray) -> float:
    n = int(np.asarray(g0).reshape(-1).size)
    g0 = np.asarray(g0, dtype=float).reshape(-1)
    CE = _as_constraint_matrix(CE, n, "CE")
    CI = _as_constraint_matrix(CI, n, "CI")
    p = int(CE.shape[1])
    m = int(CI.shape[1])
    ce0 = _as_vector(ce0, p, "ce0")
    ci0 = _as_vector(ci0, m, "ci0")
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError(f"x must have length {n}, got {x.size}")

    # +1 — слот A(iq)/u(iq) как в C++ при iq == m+p
    work = m + p + 1
    R = np.zeros((n, n), dtype=float)
    J = np.eye(n, dtype=float)
    s = np.zeros(work, dtype=float)
    z = np.zeros(n, dtype=float)
    r = np.zeros(work, dtype=float)
    d = np.zeros(n, dtype=float)
    np_vec = np.zeros(n, dtype=float)
    u = np.zeros(work, dtype=float)
    x_old = np.zeros(n, dtype=float)
    u_old = np.zeros(work, dtype=float)
    A = np.zeros(work, dtype=int)
    A_old = np.zeros(work, dtype=int)
    iai = np.zeros(work, dtype=int)
    iaexcl = np.zeros(work, dtype=int)

    me = p
    mi = m
    R_norm = 1.0
    d[:] = 0.0
    R[:] = 0.0

    c, lower = chol
    if lower:
        J[:] = solve_triangular(c, J, lower=True, trans="T", check_finite=False)
    else:
        J[:] = solve_triangular(c, J, lower=False, trans="N", check_finite=False)
    c2 = float(np.trace(J))

    x[:] = -cho_solve(chol, g0, check_finite=False)
    f_value = 0.5 * float(g0 @ x)

    iq = 0
    for i in range(me):
        np_vec[:] = CE[:, i]
        compute_d(d, J, np_vec)
        update_z(z, J, d, iq)
        update_r(R, r, d, iq)
        t2 = 0.0
        if abs(float(z @ z)) > _EPS:
            t2 = (-float(np_vec @ x) - float(ce0[i])) / float(z @ np_vec)
        x[:] = x + t2 * z
        u[iq] = t2
        if iq > 0:
            u[:iq] -= t2 * r[:iq]
        f_value += 0.5 * (t2 * t2) * float(z @ np_vec)
        A[i] = -i - 1
        ok, iq, R_norm = add_constraint(R, J, d, iq, R_norm)
        if not ok:
            return f_value

    for i in range(mi):
        iai[i] = i

    ss = 0.0
    ip = 0
    label = "l1"
    while True:
        if label == "l1":
            for i in range(me, iq):
                ip_active = int(A[i])
                if 0 <= ip_active < iai.size:
                    iai[ip_active] = -1
            ss = 0.0
            psi = 0.0
            ip = 0
            for i in range(mi):
                iaexcl[i] = 1
                total = float(CI[:, i] @ x) + float(ci0[i])
                s[i] = total
                psi += min(0.0, total)
            if abs(psi) <= mi * _EPS * c1 * c2 * 100.0:
                return f_value
            u_old[:iq] = u[:iq]
            A_old[:iq] = A[:iq]
            x_old[:] = x
            label = "l2"
            continue

        if label == "l2":
            for i in range(mi):
                if s[i] < ss and iai[i] != -1 and iaexcl[i]:
                    ss = s[i]
                    ip = i
            if ss >= 0.0:
                return f_value
            np_vec[:] = CI[:, ip]
            u[iq] = 0.0
            A[iq] = ip
            label = "l2a"
            continue

        compute_d(d, J, np_vec)
        update_z(z, J, d, iq)
        update_r(R, r, d, iq)
        l_idx = 0
        t1 = _INF
        for k in range(me, iq):
            if r[k] > 0.0:
                tmp = u[k] / r[k]
                if tmp < t1:
                    t1 = tmp
                    l_idx = int(A[k])
        if abs(float(z @ z)) > _EPS:
            t2 = -s[ip] / float(z @ np_vec)
        else:
            t2 = _INF
        t = min(t1, t2)
        if t >= _INF:
            return _INF
        if t2 >= _INF:
            u[:iq] -= t * r[:iq]
            u[iq] += t
            if 0 <= l_idx < iai.size:
                iai[l_idx] = l_idx
            iq = delete_constraint(R, J, A, u, p, iq, l_idx)
            label = "l2a"
            continue
        x[:] = x + t * z
        f_value += t * float(z @ np_vec) * (0.5 * t + u[iq])
        u[:iq] -= t * r[:iq]
        u[iq] += t
        if t == t2:
            ok, iq, R_norm = add_constraint(R, J, d, iq, R_norm)
            if not ok:
                iaexcl[ip] = 0
                iq = delete_constraint(R, J, A, u, p, iq, ip)
                for i in range(m):
                    iai[i] = i
                for i in range(iq):
                    A[i] = A_old[i]
                    a_i = int(A[i])
                    if 0 <= a_i < iai.size:
                        iai[a_i] = -1
                    u[i] = u_old[i]
                x[:] = x_old
                label = "l2"
                continue
            if 0 <= ip < iai.size:
                iai[ip] = -1
            label = "l1"
            continue
        if 0 <= l_idx < iai.size:
            iai[l_idx] = l_idx
        iq = delete_constraint(R, J, A, u, p, iq, l_idx)
        s[ip] = float(CI[:, ip] @ x) + float(ci0[ip])
        label = "l2a"


def solve_quadprog(G, g0, CE, ce0, CI, ci0, x) -> float:
    G = np.array(G.toarray() if hasattr(G, "toarray") else G, dtype=float, copy=True)
    g0 = np.asarray(g0, dtype=float).reshape(-1)
    x_arr = np.array(np.asarray(x, dtype=float).reshape(-1), dtype=float, copy=True)
    c1 = float(np.trace(G))
    chol = cho_factor(G, lower=True, check_finite=False)
    value = solve_quadprog2(chol, c1, g0, CE, ce0, CI, ci0, x_arr)
    if hasattr(x, "__setitem__") and np.ndim(x) != 0:
        x[:] = x_arr.reshape(np.asarray(x).shape)
    return value
