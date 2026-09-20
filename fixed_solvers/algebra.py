"""Операции над аргументом солвера: скаляр (dimension=1) или вектор."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

NAN = float("nan")
INF = float("inf")


def is_scalar_dimension(dimension: int) -> bool:
    """True, если система одномерная и аргумент хранится как float."""
    return dimension == 1


def is_variable_dimension(dimension: int) -> bool:
    """True, если длина аргумента неизвестна заранее (dimension == -1)."""
    return dimension == -1


def default_var(dimension: int, value: float = NAN, size: int | None = None) -> Any:
    """Создаёт аргумент нужной формы, заполненный ``value``.

    Для ``dimension == -1`` длина берётся из ``size`` (по умолчанию 0).
    """
    if dimension == 1:
        return float(value)
    if dimension == -1:
        n = 0 if size is None else int(size)
        return np.ones(n, dtype=float) * float(value)
    return np.full(int(dimension), float(value), dtype=float)


def as_float_array(values: Any) -> np.ndarray:
    """Преобразует скаляр или массив в одномерный ``float``-вектор."""
    return np.asarray(values, dtype=float).reshape(-1)


def is_scalar(value: Any) -> bool:
    """True для нульмерного значения (число, не массив)."""
    return np.ndim(value) == 0


def var_size(value: Any) -> int:
    """Число компонент аргумента (1 для скаляра)."""
    if is_scalar(value):
        return 1
    return int(np.asarray(value).reshape(-1).size)


def var_copy(value: Any) -> Any:
    """Копия аргумента: float для скаляра, независимый ndarray для вектора."""
    if is_scalar(value):
        return float(value)
    return np.array(value, dtype=float, copy=True)


def var_zeros_like(value: Any) -> Any:
    """Нулевой аргумент той же формы, что ``value``."""
    if is_scalar(value):
        return 0.0
    return np.zeros_like(np.asarray(value, dtype=float))


def var_add(a: Any, b: Any) -> Any:
    """Покомпонентная сумма аргументов."""
    if is_scalar(a) and is_scalar(b):
        return float(a) + float(b)
    return as_float_array(a) + as_float_array(b)


def var_sub(a: Any, b: Any) -> Any:
    """Покомпонентная разность аргументов."""
    if is_scalar(a) and is_scalar(b):
        return float(a) - float(b)
    return as_float_array(a) - as_float_array(b)


def var_scale(scalar: float, value: Any) -> Any:
    """Умножение аргумента на скаляр."""
    if is_scalar(value):
        return float(scalar) * float(value)
    return float(scalar) * as_float_array(value)


def var_div(value: Any, scalar: float) -> Any:
    """Деление аргумента на скаляр."""
    if is_scalar(value):
        return float(value) / float(scalar)
    return as_float_array(value) / float(scalar)


def var_getitem(value: Any, index: int) -> float:
    """Компонента аргумента; для скаляра допустим только индекс 0."""
    if is_scalar(value):
        if index != 0:
            raise IndexError(index)
        return float(value)
    return float(np.asarray(value, dtype=float).reshape(-1)[index])


def var_setitem(value: Any, index: int, component: float) -> Any:
    """Возвращает аргумент с заменённой компонентой (скаляр заменяется целиком)."""
    if is_scalar(value):
        return float(component)
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = np.array(arr, dtype=float, copy=True)
    arr[index] = float(component)
    return arr


def inner_prod(v1: Any, v2: Any) -> float:
    """Скалярное произведение; для двух скаляров — обычное умножение."""
    if is_scalar(v1) and is_scalar(v2):
        return float(v1) * float(v2)
    a = as_float_array(v1)
    b = as_float_array(v2)
    return float(np.dot(a, b))


def has_not_finite(value: Any) -> bool:
    """True, если среди компонент есть NaN или бесконечность."""
    if is_scalar(value):
        return not math.isfinite(float(value))
    arr = np.asarray(value, dtype=float)
    return bool(np.any(~np.isfinite(arr)))


def numeric_derivative_delta(value: float, epsilon: float) -> float:
    """Шаг двусторонней разности: ``epsilon * max(1, |value|)``."""
    return float(epsilon) * max(1.0, abs(float(value)))


def two_sided_derivative(function, value: float, epsilon: float):
    """Центральная разность ``(f(x+dx) - f(x-dx)) / (2 dx)``."""
    dx = numeric_derivative_delta(value, epsilon)
    f_plus = function(value + dx)
    f_minus = function(value - dx)
    return (f_plus - f_minus) / (2.0 * dx)


def squared_norm(value: Any) -> float:
    """Квадрат евклидовой нормы (для скаляра — квадрат значения)."""
    if is_scalar(value):
        r = float(value)
        return r * r
    arr = as_float_array(value)
    return float(np.dot(arr, arr))


def triplets_to_dense(triplets: Sequence[tuple[int, int, float]], n_rows: int, n_cols: int) -> np.ndarray:
    """Собирает плотную матрицу из троек ``(row, col, value)``; совпадающие ячейки суммируются."""
    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for row, col, val in triplets:
        matrix[int(row), int(col)] += float(val)
    return matrix
