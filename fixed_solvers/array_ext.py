"""Операции над массивами фиксированной размерности."""

from __future__ import annotations

from typing import Callable

import numpy as np


class array_maker:
    """Создаёт массив длины n, заполненный value."""

    @staticmethod
    def make_array(n: int, value):
        n = int(n)
        if n <= 0:
            return np.array([], dtype=float if np.ndim(value) == 0 else np.asarray(value).dtype)
        return np.full(n, value)


def make_array(n: int, value):
    return array_maker.make_array(n, value)


def create_array(dimension: int, getter: Callable[[int], object]) -> np.ndarray:
    """Геттер вызывается с индексом элемента; порядок вызовов — с Dimension-1 до 0."""
    dimension = int(dimension)
    if dimension <= 0:
        return np.array([])
    values: list = [None] * dimension
    for index in range(dimension - 1, -1, -1):
        values[index] = getter(index)
    return np.array(values)


def array_add(v1, v2):
    a = np.array(v1, copy=True)
    b = np.asarray(v2)
    for index in range(a.size):
        a.reshape(-1)[index] += b.reshape(-1)[index]
    return a


def array_sub(v1, v2):
    a = np.array(v1, copy=True)
    b = np.asarray(v2)
    for index in range(a.reshape(-1).size):
        a.reshape(-1)[index] -= b.reshape(-1)[index]
    return a


def array_neg(v):
    a = np.array(v, copy=True)
    if a.ndim == 2:
        for index in range(a.shape[0]):
            a[index] = array_neg(a[index])
        return a
    flat = a.reshape(-1)
    for index in range(flat.size):
        flat[index] = -flat[index]
    return a


def array_scale(scalar: float, v):
    a = np.array(v, copy=True, dtype=float)
    flat = a.reshape(-1)
    for index in range(flat.size):
        flat[index] *= float(scalar)
    return a


def array_div(v, scalar: float):
    a = np.array(v, copy=True, dtype=float)
    flat = a.reshape(-1)
    for index in range(flat.size):
        flat[index] /= float(scalar)
    return a


def array_iadd(v1, v2):
    a = np.asarray(v1)
    b = np.asarray(v2)
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    for index in range(flat_a.size):
        flat_a[index] += flat_b[index]
    return a


def inner_prod(v1, v2):
    if np.ndim(v1) == 0 and np.ndim(v2) == 0:
        return float(v1) * float(v2)
    a = np.asarray(v1).reshape(-1)
    b = np.asarray(v2).reshape(-1)
    result = float(a[0]) * float(b[0])
    for index in range(1, a.size):
        result += float(a[index]) * float(b[index])
    return result


def matvec(matrix, vector):
    """Умножение row-major матрицы на вектор-столбец."""
    m = np.asarray(matrix)
    v = np.asarray(vector).reshape(-1)
    result = np.empty(m.shape[0], dtype=float)
    for index in range(m.shape[0]):
        result[index] = inner_prod(m[index], v)
    return result


class array_ref:
    """Ссылочный массив: элементы указывают в исходный контейнер."""

    def __init__(self, source=None, getter=None, dimension=None, refs=None):
        if refs is not None:
            self._refs = list(refs)
        elif getter is not None:
            dim = int(dimension)
            slots: list = [None] * dim
            for index in range(dim - 1, -1, -1):
                slots[index] = getter(index)
            self._refs = slots
        elif source is not None:
            arr = np.asarray(source)
            self._refs = [(arr, index) for index in range(arr.reshape(-1).size)]
        else:
            raise TypeError("array_ref requires source, getter or refs")

    def __len__(self) -> int:
        return len(self._refs)

    def _slot(self, index: int):
        return self._refs[index]

    def __getitem__(self, index: int):
        item = self._slot(index)
        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], "__getitem__"):
            return item[0][item[1]]
        return item

    def __setitem__(self, index: int, value) -> None:
        item = self._slot(index)
        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], "__setitem__"):
            item[0][item[1]] = value
            return
        self._refs[index] = value

    def to_array(self) -> np.ndarray:
        return np.array([self[i] for i in range(len(self))])

    def assign(self, other) -> None:
        if np.ndim(other) == 0:
            for index in range(len(self)):
                self[index] = other
            return
        other_arr = np.asarray(other).reshape(-1)
        for index in range(len(self)):
            self[index] = other_arr[index]

    def as_scalar(self):
        return self[0]
