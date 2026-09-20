"""Операции над аргументом солвера: скаляр (dimension=1) или вектор.

Солверы принимают один и тот же API при трёх формах аргумента:

    dimension =  1  →  Python float
    dimension =  n  →  ndarray длины n
    dimension = -1  →  ndarray переменной длины (size задаётся снаружи)

Функции ниже прячут эту развилку: сложение, норма, копия выглядят одинаково
и в скалярном Ньютоне, и в разреженной системе.
"""

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

    NaN по умолчанию — «ещё не задано»: так result.argument в бисекции
    отличается от явного начального приближения.
    """
    if dimension == 1:
        return float(value)
    if dimension == -1:
        # Переменная длина: без size нельзя угадать n, берём пустой вектор.
        n = 0 if size is None else int(size)
        return np.ones(n, dtype=float) * float(value)
    return np.full(int(dimension), float(value), dtype=float)


def as_float_array(values: Any) -> np.ndarray:
    """Скаляр или массив → одномерный float-вектор (удобно для J @ r и норм)."""
    return np.asarray(values, dtype=float).reshape(-1)


def is_scalar(value: Any) -> bool:
    """Нульмерное значение (число), а не ndarray формы (1,) или (1, 1)."""
    return np.ndim(value) == 0


def var_size(value: Any) -> int:
    """Число компонент: для скаляра всегда 1, иначе длина после flatten."""
    if is_scalar(value):
        return 1
    return int(np.asarray(value).reshape(-1).size)


def var_copy(value: Any) -> Any:
    """Независимая копия: float копируется значением, вектор — новым буфером.

    Нужна перед порчей аргумента при численном якобиане (x ± e).
    """
    if is_scalar(value):
        return float(value)
    return np.array(value, dtype=float, copy=True)


def var_zeros_like(value: Any) -> Any:
    """Нулевой шаг той же формы (координатный спуск начинает с p = 0)."""
    if is_scalar(value):
        return 0.0
    return np.zeros_like(np.asarray(value, dtype=float))


def var_add(a: Any, b: Any) -> Any:
    """a + b с сохранением скалярности, если оба скаляра."""
    if is_scalar(a) and is_scalar(b):
        return float(a) + float(b)
    return as_float_array(a) + as_float_array(b)


def var_sub(a: Any, b: Any) -> Any:
    """a − b, та же развилка скаляр / вектор."""
    if is_scalar(a) and is_scalar(b):
        return float(a) - float(b)
    return as_float_array(a) - as_float_array(b)


def var_scale(scalar: float, value: Any) -> Any:
    """α · v — шаг Ньютона после линейного поиска: x ← x + α p."""
    if is_scalar(value):
        return float(scalar) * float(value)
    return float(scalar) * as_float_array(value)


def var_div(value: Any, scalar: float) -> Any:
    """v / α."""
    if is_scalar(value):
        return float(value) / float(scalar)
    return as_float_array(value) / float(scalar)


def var_getitem(value: Any, index: int) -> float:
    """Компонента xᵢ. У скаляра есть только i = 0."""
    if is_scalar(value):
        if index != 0:
            raise IndexError(index)
        return float(value)
    return float(np.asarray(value, dtype=float).reshape(-1)[index])


def var_setitem(value: Any, index: int, component: float) -> Any:
    """Возвращает копию с xᵢ = component. Скаляр целиком заменяется новым float."""
    if is_scalar(value):
        return float(component)
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = np.array(arr, dtype=float, copy=True)
    arr[index] = float(component)
    return arr


def inner_prod(v1: Any, v2: Any) -> float:
    """Скалярное произведение ⟨v1, v2⟩ = Σ v1ᵢ v2ᵢ (для скаляров — обычное ·)."""
    if is_scalar(v1) and is_scalar(v2):
        return float(v1) * float(v2)
    a = as_float_array(v1)
    b = as_float_array(v2)
    return float(np.dot(a, b))


def has_not_finite(value: Any) -> bool:
    """True, если есть NaN или ±∞ — для солвера это срыв расчёта, не «корень не найден»."""
    if is_scalar(value):
        return not math.isfinite(float(value))
    arr = np.asarray(value, dtype=float)
    return bool(np.any(~np.isfinite(arr)))


def numeric_derivative_delta(value: float, epsilon: float) -> float:
    """Относительный шаг центральной разности по компоненте x:

            e = ε · max(1, |x|)

    max(1, |x|) не даёт шагу стать машинным нулём при маленьком x
    и не делает его огромным при большом x.
    """
    # Масштаб «не меньше единицы»: при x ≈ 0 шаг всё равно ε, а не ε·|x| ≈ 0.
    return float(epsilon) * max(1.0, abs(float(value)))


def two_sided_derivative(function, value: float, epsilon: float):
    """Центральная (двусторонняя) разность скалярной f:
    f′(x) ≈ (f(x+e) − f(x−e)) / (2e).
    Погрешность O(e²); у правой разности (f(x+e)−f(x))/e погрешность O(e).
    """
    dx = numeric_derivative_delta(value, epsilon)
    # Два вызова: вперёд и назад. Не используем f(x) — он сократился бы только
    # в односторонней схеме, а здесь дал бы лишнюю ошибку округления.
    f_plus = function(value + dx)
    f_minus = function(value - dx)
    return (f_plus - f_minus) / (2.0 * dx)


def squared_norm(value: Any) -> float:
    """Квадрат евклидовой нормы — целевая Ньютона / Гаусса–Ньютона: ‖v‖² = Σᵢ vᵢ² = ⟨v,v⟩.
    Для скаляра это просто v² (без вызова BLAS).
    """
    if is_scalar(value):
        r = float(value)
        return r * r
    arr = as_float_array(value)
    return float(np.dot(arr, arr))


def triplets_to_dense(triplets: Sequence[tuple[int, int, float]], n_rows: int, n_cols: int) -> np.ndarray:
    """COO (row, col, value) → плотная матрица.

    Совпадающие ячейки суммируются: так собирают якобиан из вкладов нескольких рёбер.
    """
    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for row, col, val in triplets:
        matrix[int(row), int(col)] += float(val)
    return matrix
