"""Решение СЛАУ малой размерности методом Крамера."""

from __future__ import annotations


import numpy as np

from .exceptions import logic_error


def _check_finite(values) -> None:
    """Предусловие: в решении нет NaN и бесконечностей."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(arr)):
        raise logic_error("infinite value")


def determinant3(
    a11, a12, a13,
    a21, a22, a23,
    a31, a32, a33,
) -> float:
    """Определитель 3×3 в развёрнутой форме."""
    return (
        + a11 * a22 * a33
        - a11 * a23 * a32
        - a12 * a21 * a33
        + a12 * a23 * a31
        + a13 * a21 * a32
        - a13 * a22 * a31
    )


def solve_linear_system(A, b=None):
    """Решение ax=b (скаляр), 2×2 и 3×3 методом Крамера, иначе numpy.linalg.solve.

    Перегрузка пары (a, b): ``solve_linear_system((a, b))``.
    """
    if b is None:
        if isinstance(A, (tuple, list)) and len(A) == 2 and np.ndim(A[0]) == 0:
            return solve_linear_system(A[0], A[1])
        raise TypeError("b is required unless A is a (coeff, rhs) pair")

    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if A_arr.ndim == 0:
        result = float(b_arr) / float(A_arr)
        _check_finite(result)
        return result

    if A_arr.shape == (2, 2):
        d = A_arr[0, 0] * A_arr[1, 1] - A_arr[1, 0] * A_arr[0, 1]
        d1 = b_arr[0] * A_arr[1, 1] - b_arr[1] * A_arr[0, 1]
        d2 = A_arr[0, 0] * b_arr[1] - A_arr[1, 0] * b_arr[0]
        result = np.array([d1 / d, d2 / d], dtype=float)
        _check_finite(result)
        return result

    if A_arr.shape == (3, 3):
        d = determinant3(
            A_arr[0, 0], A_arr[0, 1], A_arr[0, 2],
            A_arr[1, 0], A_arr[1, 1], A_arr[1, 2],
            A_arr[2, 0], A_arr[2, 1], A_arr[2, 2],
        )
        d1 = determinant3(
            b_arr[0], A_arr[0, 1], A_arr[0, 2],
            b_arr[1], A_arr[1, 1], A_arr[1, 2],
            b_arr[2], A_arr[2, 1], A_arr[2, 2],
        )
        d2 = determinant3(
            A_arr[0, 0], b_arr[0], A_arr[0, 2],
            A_arr[1, 0], b_arr[1], A_arr[1, 2],
            A_arr[2, 0], b_arr[2], A_arr[2, 2],
        )
        d3 = determinant3(
            A_arr[0, 0], A_arr[0, 1], b_arr[0],
            A_arr[1, 0], A_arr[1, 1], b_arr[1],
            A_arr[2, 0], A_arr[2, 1], b_arr[2],
        )
        result = np.array([d1 / d, d2 / d, d3 / d], dtype=float)
        _check_finite(result)
        return result

    result = np.linalg.solve(A_arr, b_arr)
    _check_finite(result)
    return result
