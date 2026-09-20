"""Действительные корни кубического уравнения и экстремумы."""

from __future__ import annotations

import math
import sys

from ..exceptions import invalid_argument

_DBL_EPSILON = sys.float_info.epsilon

# Абсолютная добавка к порогу сравнения дискриминанта куба с нулём.
discriminant_zero_tol_absolute = 1e-11
# Множитель при scale^3 в пороге для дискриминанта.
discriminant_zero_tol_scale_cubed_coeff = 1e-9
# Абсолютная добавка к порогу вырождения производной (a^2 - 3b).
derivative_degenerate_tol_absolute = 1e-10
# Множитель при max(scale^2, 1) в пороге вырождения производной.
derivative_degenerate_tol_scale_sq_coeff = 1e-8


def cbrt(x: float) -> float:
    """Вещественный кубический корень с сохранением знака."""
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def normalize_to_monic_in_place(coeffs: list[float]) -> None:
    """Делит коэффициенты на старший, чтобы уравнение стало приведённым."""
    lead = coeffs[-1]
    lead_rev = 1.0 / lead
    for i, coeff in enumerate(coeffs):
        coeffs[i] = coeff * lead_rev


def monic_coefficient_scale(a: float, b: float, c: float) -> float:
    """Масштаб коэффициентов приведённого куба x^3 + a x^2 + b x + c."""
    return max(1.0, abs(a), abs(b), abs(c))


def discriminants_tolerances_for_scale(scale: float) -> tuple[float, float]:
    """Пороги «дискриминант ≈ 0» и «производная вырождена» для данного масштаба."""
    tol_disc = (
        discriminant_zero_tol_absolute
        + discriminant_zero_tol_scale_cubed_coeff * scale * scale * scale
    )
    tol_degen = (
        derivative_degenerate_tol_absolute
        + derivative_degenerate_tol_scale_sq_coeff * max(scale * scale, 1.0)
    )
    return tol_disc, tol_degen


def monic_discriminant(a: float, b: float, c: float) -> float:
    """Дискриминант приведённого кубического многочлена."""
    return (
        18.0 * a * b * c
        - 4.0 * b * b * b
        + a * a * b * b
        - 4.0 * a * a * a * c
        - 27.0 * c * c
    )


def validated_monic_coeffs_from_poly(poly: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    """Коэффициенты a, b, c приведённого куба из ``[c0, c1, c2, c3]``."""
    if len(poly) != 4:
        raise invalid_argument("count_distinct_real_roots_cubic: expected 4 coefficients")
    lead = poly[3]
    lead_abs = abs(lead)
    scale_in = max(1.0, abs(poly[0]), abs(poly[1]), abs(poly[2]))
    if lead_abs <= _DBL_EPSILON * scale_in:
        raise invalid_argument("count_distinct_real_roots_cubic: leading coefficient is zero")
    lead_rev = 1.0 / lead
    return poly[2] * lead_rev, poly[1] * lead_rev, poly[0] * lead_rev


def count_distinct_real_roots_for_monic(a: float, b: float, c: float) -> int:
    """Число различных действительных корней приведённого куба (1, 2 или 3)."""
    scale = monic_coefficient_scale(a, b, c)
    tol_disc, tol_degen = discriminants_tolerances_for_scale(scale)
    delta = monic_discriminant(a, b, c)
    if delta > tol_disc:
        return 3
    if delta < -tol_disc:
        return 1
    pprime_disc = a * a - 3.0 * b
    if abs(pprime_disc) <= tol_degen:
        return 1
    return 2


def solve_monic_s_near_zero(a: float, R: float) -> list[float]:
    """Кратный корень при S ≈ 0 (граница трёх/одного действительного корня)."""
    return [-2.0 * cbrt(R) - a / 3.0, cbrt(R) - a / 3.0]


def solve_monic_three_real_roots(a: float, Q: float, R: float) -> list[float]:
    """Три действительных корня тригонометрической формулой Виета."""
    phi = (1.0 / 3.0) * math.acos(R / (Q ** 1.5))
    shift = -a / 3.0
    radius = -2.0 * math.sqrt(Q)
    return [
        radius * math.cos(phi) + shift,
        radius * math.cos(phi + 2.0 / 3.0 * math.pi) + shift,
        radius * math.cos(phi - 2.0 / 3.0 * math.pi) + shift,
    ]


def monic_depressed_cubic_cardano_Q(a: float, b: float, c: float) -> float:
    """Дискриминант Кардано Q для приведённого к виду y^3 + p y + q куба."""
    p_dep = b - a * a / 3.0
    q_dep = c - a * b / 3.0 + 2.0 * a * a * a / 27.0
    return (p_dep / 3.0) ** 3 + (q_dep / 2.0) ** 2


def solve_monic_one_real_depressed_cardano(a: float, b: float, c: float) -> list[float]:
    """Один действительный корень формулой Кардано для приведённого куба."""
    p_dep = b - a * a / 3.0
    q_dep = c - a * b / 3.0 + 2.0 * a * a * a / 27.0
    Q_card = (p_dep / 3.0) ** 3 + (q_dep / 2.0) ** 2
    rad = math.sqrt(max(0.0, Q_card))
    d1 = -q_dep / 2.0 + rad
    d2 = -q_dep / 2.0 - rad
    y = cbrt(d1) + cbrt(d2)
    return [y - a / 3.0]


def solve_monic(a: float, b: float, c: float) -> list[float]:
    """Действительные корни x^3 + a x^2 + b x + c по знаку S = Q^3 - R^2."""
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a ** 3 - 9.0 * a * b + 27.0 * c) / 54.0
    S = Q ** 3 - R * R
    # S ≈ 0 — кратный корень; S > 0 — три действительных; иначе Кардано (один действительный).
    if abs(S) < _DBL_EPSILON:
        return solve_monic_s_near_zero(a, R)
    if S > 0:
        return solve_monic_three_real_roots(a, Q, R)
    return solve_monic_one_real_depressed_cardano(a, b, c)


def count_distinct_real_roots_cubic(poly_coeffs: list[float]) -> int:
    """Число различных действительных корней куба ``c0 + c1 x + c2 x^2 + c3 x^3``."""
    a, b, c = validated_monic_coeffs_from_poly(poly_coeffs)
    return count_distinct_real_roots_for_monic(a, b, c)


def solve_cubic_equation(poly_coeffs: list[float]) -> list[float]:
    """Действительные корни кубического уравнения по коэффициентам ``[c0..c3]``."""
    coeffs = [float(c) for c in poly_coeffs]
    normalize_to_monic_in_place(coeffs)
    a, b, c = coeffs[2], coeffs[1], coeffs[0]
    return solve_monic(a, b, c)


def monic_cubic_cardano_Q(a: float, b: float, c: float) -> float:
    """Синоним ``monic_depressed_cubic_cardano_Q``."""
    return monic_depressed_cubic_cardano_Q(a, b, c)


def solve_monic_cubic_single_real_root(a: float, b: float, c: float) -> float:
    """Один действительный корень приведённого куба (ветка Кардано)."""
    roots = solve_monic_one_real_depressed_cardano(a, b, c)
    if not roots:
        return float("nan")
    return roots[0]


def find_cubic_extremums(coeffs: list[float]) -> tuple[float, float]:
    """Локальный максимум и минимум куба; ``NaN``, если экстремума нет."""
    if len(coeffs) != 4:
        raise RuntimeError("find_cubic_extremums: expected 4 coefficients")
    D = coeffs[3]
    D_abs = abs(D)
    scale_in = max(1.0, abs(coeffs[0]), abs(coeffs[1]), abs(coeffs[2]))
    if D_abs <= _DBL_EPSILON * scale_in:
        raise RuntimeError("find_cubic_extremums: leading coefficient is zero")

    B = coeffs[1]
    C = coeffs[2]
    a_prime = 3.0 * D
    b_prime = 2.0 * C
    c_prime = B

    prime_scale = monic_coefficient_scale(a_prime, b_prime, c_prime)
    tol_degen = (
        derivative_degenerate_tol_absolute
        + derivative_degenerate_tol_scale_sq_coeff * max(prime_scale * prime_scale, 1.0)
    )

    if abs(a_prime) <= tol_degen:
        if abs(b_prime) <= tol_degen:
            return float("nan"), float("nan")
        Q = -c_prime / b_prime
        second_deriv = 2.0 * C + 6.0 * D * Q
        if second_deriv < 0:
            return Q, float("nan")
        return float("nan"), Q

    discriminant = b_prime * b_prime - 4.0 * a_prime * c_prime
    tol_disc = (
        discriminant_zero_tol_absolute
        + discriminant_zero_tol_scale_cubed_coeff * prime_scale * prime_scale * prime_scale
    )
    if discriminant < -tol_disc:
        return float("nan"), float("nan")

    sqrt_disc = math.sqrt(max(0.0, discriminant))
    Q1 = (-b_prime + sqrt_disc) / (2.0 * a_prime)
    Q2 = (-b_prime - sqrt_disc) / (2.0 * a_prime)
    second_deriv_Q1 = 2.0 * C + 6.0 * D * Q1
    # Знак второй производной отличает локальный максимум от минимума.
    if second_deriv_Q1 < 0:
        return Q1, Q2
    return Q2, Q1


def find_positive_cubic_extremums(coefficients: list[float]) -> tuple[float, float]:
    """Экстремумы на положительной полуоси; неположительные заменяются на ``NaN``."""
    Q_max, Q_min = find_cubic_extremums(coefficients)
    return (
        float("nan") if Q_max <= 0.0 else Q_max,
        float("nan") if Q_min <= 0.0 else Q_min,
    )


def solve_realpoly3_vieta(poly_coeffs: list[float]) -> list[float]:
    """Тригонометрическая формула Виета."""
    coeffs = [float(c) for c in poly_coeffs]
    higher_order_coeff = coeffs[-1]
    coeffs = [c / higher_order_coeff for c in coeffs]
    a, b, c = coeffs[2], coeffs[1], coeffs[0]
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a ** 3 - 9.0 * a * b + 27.0 * c) / 54.0
    S = Q ** 3 - R * R
    if abs(S) < _DBL_EPSILON:
        return [-2.0 * cbrt(R) - a / 3.0, cbrt(R) - a / 3.0]
    if S > 0:
        phi = (1.0 / 3.0) * math.acos(R / (Q ** 1.5))
        return [
            -2.0 * math.sqrt(Q) * math.cos(phi) - a / 3.0,
            -2.0 * math.sqrt(Q) * math.cos(phi + 2.0 / 3.0 * math.pi) - a / 3.0,
            -2.0 * math.sqrt(Q) * math.cos(phi - 2.0 / 3.0 * math.pi) - a / 3.0,
        ]
    if abs(Q) < _DBL_EPSILON:
        return [-cbrt(c - (a * a * a) / 27.0) - a / 3.0]
    from .math_helpers import pseudo_sgn
    if Q > 0:
        phi = (1.0 / 3.0) * math.acosh(abs(R) / (Q ** 1.5))
        return [-2.0 * pseudo_sgn(R) * math.sqrt(Q) * math.cosh(phi) - a / 3.0]
    phi = (1.0 / 3.0) * math.asinh(abs(R) / (abs(Q) ** 1.5))
    return [-2.0 * pseudo_sgn(R) * math.sqrt(abs(Q)) * math.sinh(phi) - a / 3.0]
