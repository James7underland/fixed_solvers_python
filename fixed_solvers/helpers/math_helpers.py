"""Математические хелперы (namespace fixed_solvers в C++)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..enums import nullable_bool_t


def to_nullable_bool(value: bool) -> nullable_bool_t:
    return nullable_bool_t.True_ if value else nullable_bool_t.False_


def nullable_bool_to_str(nb: nullable_bool_t) -> str:
    if nb == nullable_bool_t.True_:
        return "True"
    if nb == nullable_bool_t.False_:
        return "False"
    return "Undefined"


def pseudo_sgn(val) -> int:
    return 2 * (val >= 0) - 1


def ensure_abs_epsilon_value(value, epsilon=1e-6):
    abs_value = abs(value)
    if abs_value >= epsilon:
        return value
    return pseudo_sgn(value) * epsilon


def inv_vector(v: np.ndarray) -> np.ndarray:
    result = np.array(v, dtype=float, copy=True)
    result = 1.0 / result
    return result


invVectorXd = inv_vector


def sgn(val) -> int:
    return int((0 < val) - (val < 0))


def sqr(x: float) -> float:
    return x * x


def ssqrt(x: float) -> float:
    if x >= 0:
        return math.sqrt(x)
    return -math.sqrt(-x)


def ssqr(x: float) -> float:
    if x >= 0:
        return sqr(x)
    return -sqr(x)


def polyval(poly_coeffs: Sequence[float], x: float) -> float:
    # НЕ используется формат Matlab: индекс коэффициента = степень.
    n = len(poly_coeffs)
    if n == 0:
        return 0.0
    result = 0.0
    power = 1.0
    for index in range(n):
        result += power * float(poly_coeffs[index])
        power *= x
    return result


def poly_differentiate(poly_coeffs: Sequence[float]) -> list[float]:
    derivative = [0.0] * (len(poly_coeffs) - 1)
    for index in range(len(derivative)):
        k = float(index + 1)
        a = float(poly_coeffs[index + 1])
        derivative[index] = a * k
    return derivative


def has_not_finite(value) -> bool:
    if np.ndim(value) == 0:
        return not math.isfinite(float(value))
    arr = np.asarray(value, dtype=float)
    for item in arr.reshape(-1):
        if not math.isfinite(float(item)):
            return True
    return False
