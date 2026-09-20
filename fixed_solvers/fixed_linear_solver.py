"""Решение СЛАУ малой размерности методом Крамера.

Ньютон для n ≤ 3 не ходит в общий LU: явные формулы дешевле и дают
тот же порядок операций, что заложен в тестах.

Скаляр: x = b/a.

Правило Крамера (столбец i матрицы A заменён на правую часть b):
xᵢ = det(Aᵢ) / det(A).

2×2, A = [[a, b], [c, d]], правая часть (e, f):
x₁ = (e·d − b·f) / (a·d − b·c),  x₂ = (a·f − c·e) / (a·d − b·c).

3×3 — то же, det считается формулой Саррюса (см. ``determinant3``).

Если размер другой — ``numpy.linalg.solve``. Любой NaN/∞ в ответе —
``logic_error``: для солвера это «матрица/правая часть испортились», а не корень.
"""

from __future__ import annotations


import numpy as np

from .exceptions import logic_error


def _check_finite(values) -> None:
    """Срыв арифметики (деление на 0, переполнение) не маскируем как «решили»."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(arr)):
        raise logic_error("infinite value")


def determinant3(
    a11, a12, a13,
    a21, a22, a23,
    a31, a32, a33,
) -> float:
    """Определитель 3×3, формула Саррюса:
    det A = a₁₁a₂₂a₃₃ + a₁₂a₂₃a₃₁ + a₁₃a₂₁a₃₂ − a₁₃a₂₂a₃₁ − a₁₁a₂₃a₃₂ − a₁₂a₂₁a₃₃.
    """
    # Слагаемые в том же порядке, что даёт раскрытие: сначала «плюс»-тройки,
    # затем «минус». Не вызываем numpy.linalg.det — тесты фиксируют явный путь.
    return (
        + a11 * a22 * a33
        - a11 * a23 * a32
        - a12 * a21 * a33
        + a12 * a23 * a31
        + a13 * a21 * a32
        - a13 * a22 * a31
    )


def solve_linear_system(A, b=None):
    """Решает A x = b.

    Перегрузка ``solve_linear_system((a, b))`` — скалярная пара коэффициентов,
    как в вызовах без явного второго аргумента.
    """
    if b is None:
        if isinstance(A, (tuple, list)) and len(A) == 2 and np.ndim(A[0]) == 0:
            return solve_linear_system(A[0], A[1])
        raise TypeError("b is required unless A is a (coeff, rhs) pair")

    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if A_arr.ndim == 0:
        # Скалярное уравнение a x = b.
        result = float(b_arr) / float(A_arr)
        _check_finite(result)
        return result

    if A_arr.shape == (2, 2):
        # x₁ = (e·d−b·f)/(a·d−b·c),  x₂ = (a·f−c·e)/(a·d−b·c)
        d = A_arr[0, 0] * A_arr[1, 1] - A_arr[1, 0] * A_arr[0, 1]
        d1 = b_arr[0] * A_arr[1, 1] - b_arr[1] * A_arr[0, 1]
        d2 = A_arr[0, 0] * b_arr[1] - A_arr[1, 0] * b_arr[0]
        result = np.array([d1 / d, d2 / d], dtype=float)
        _check_finite(result)
        return result

    if A_arr.shape == (3, 3):
        # Крамер: Aᵢ — A с i-м столбцом, заменённым на b;  xᵢ = det(Aᵢ)/det(A).
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
