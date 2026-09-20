"""Математические хелперы: знаки, полиномы, проверка конечности."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..enums import nullable_bool_t


def to_nullable_bool(value: bool) -> nullable_bool_t:
    """Обычный bool → трёхзначное значение (Undefined не возникает)."""
    return nullable_bool_t.True_ if value else nullable_bool_t.False_


def nullable_bool_to_str(nb: nullable_bool_t) -> str:
    """Строка ``True`` / ``False`` / ``Undefined``."""
    if nb == nullable_bool_t.True_:
        return "True"
    if nb == nullable_bool_t.False_:
        return "False"
    return "Undefined"


def pseudo_sgn(val) -> int:
    """Знак с нулём как плюс: ``+1`` при ``val ≥ 0``, иначе ``−1``."""
    return 2 * (val >= 0) - 1


def ensure_abs_epsilon_value(value, epsilon=1e-6):
    """Поднимает слишком малые |x| до ε, сохраняя знак:

        |x| ≥ ε  →  x
        иначе    →  sign*(x) · ε     (ноль даёт +ε, т.к. pseudo_sgn(0) = +1)

    Нужно, чтобы не делить на почти-ноль в знаменателях прикладных формул.
    """
    abs_value = abs(value)
    if abs_value >= epsilon:
        return value
    return pseudo_sgn(value) * epsilon


def inv_vector(v: np.ndarray) -> np.ndarray:
    """Покомпонентное ``1/v`` (копия входного массива)."""
    result = np.array(v, dtype=float, copy=True)
    result = 1.0 / result
    return result


invVectorXd = inv_vector


def sgn(val) -> int:
    """Классический знак: -1 / 0 / +1."""
    return int((0 < val) - (val < 0))


def sqr(x: float) -> float:
    """Квадрат."""
    return x * x


def ssqrt(x: float) -> float:
    """Знаковый квадратный корень: ``sign(x) · √|x|``."""
    if x >= 0:
        return math.sqrt(x)
    return -math.sqrt(-x)


def ssqr(x: float) -> float:
    """Знаковый квадрат: ``sign(x) · x²``."""
    if x >= 0:
        return sqr(x)
    return -sqr(x)


def polyval(poly_coeffs: Sequence[float], x: float) -> float:
    """Значение полинома, индекс = степень (не Matlab, там старший первым):
    p(x) = Σₖ aₖ xᵏ. Схема: acc=0, pow=1; на каждом k: acc += aₖ·pow; pow *= x.
    Пустой список коэффициентов — тождественный ноль.
    """
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
    """(Σ aₖ xᵏ)′ = Σ k aₖ xᵏ⁻¹ → новыйₖ = (k+1) aₖ₊₁."""
    derivative = [0.0] * (len(poly_coeffs) - 1)
    for index in range(len(derivative)):
        k = float(index + 1)
        a = float(poly_coeffs[index + 1])
        derivative[index] = a * k
    return derivative


def has_not_finite(value) -> bool:
    """True, если среди компонент есть NaN или бесконечность."""
    if np.ndim(value) == 0:
        return not math.isfinite(float(value))
    arr = np.asarray(value, dtype=float)
    for item in arr.reshape(-1):
        if not math.isfinite(float(item)):
            return True
    return False
