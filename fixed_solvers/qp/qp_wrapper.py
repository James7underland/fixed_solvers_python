"""Обёртка квадратичного программирования и CCS-представление разреженных матриц."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse

from .eiquadprog import solve_quadprog, solve_quadprog2


def get_sparse_matrix_CCS(matrix) -> tuple[list[float], list[int], list[int]]:
    """Преобразует разреженную матрицу в формат CCS (CSC), как qp_wrapper.h."""
    csc = sparse.csc_matrix(matrix, dtype=float)
    csc.sort_indices()
    values: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    indptr = csc.indptr
    indices = csc.indices
    data = csc.data
    for col in range(csc.shape[1]):
        cols.append(len(values))
        start, end = int(indptr[col]), int(indptr[col + 1])
        for pos in range(start, end):
            values.append(float(data[pos]))
            rows.append(int(indices[pos]))
    cols.append(len(values))
    return values, rows, cols


def solve_quadprog_box(
    H,
    f,
    minimum: Sequence[tuple[int, float]],
    maximum: Sequence[tuple[int, float]],
) -> np.ndarray:
    """min 0.5 x' H x + f' x при box-ограничениях — как testing/test_main.cpp."""
    H_dense = np.asarray(H.toarray() if sparse.issparse(H) else H, dtype=float)
    f_vec = np.asarray(f, dtype=float).reshape(-1)
    n = f_vec.size
    n_cons = len(minimum) + len(maximum)
    A = np.zeros((n_cons, n), dtype=float)
    b = np.zeros(n_cons, dtype=float)
    row = 0
    for index, min_bound in minimum:
        A[row, int(index)] = -1.0
        b[row] = -float(min_bound)
        row += 1
    for index, max_bound in maximum:
        A[row, int(index)] = 1.0
        b[row] = float(max_bound)
        row += 1
    estimation = np.zeros(n, dtype=float)
    CE = np.zeros((n, 0), dtype=float)
    ce0 = np.zeros(0, dtype=float)
    CI = np.zeros((n, 0), dtype=float) if n_cons == 0 else -A.T
    solve_quadprog(H_dense, f_vec, CE, ce0, CI, b, estimation)
    return estimation
