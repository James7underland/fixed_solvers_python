"""Действительные корни кубического уравнения и экстремумы.

Полином хранится как [c₀, c₁, c₂, c₃], степень = индекс: p(x) = c₀ + c₁x + c₂x² + c₃x³.
Приведение к монику: x³ + a x² + b x + c = 0, где a=c₂/c₃, b=c₁/c₃, c=c₀/c₃.

Формулы Виета: Q = (a²−3b)/9, R = (2a³ − 9a b + 27c)/54, S = Q³ − R².
    S > 0   →  три действительных корня (тригонометрия)
    S ≈ 0   →  кратные корни
    S < 0   →  один действительный (Кардано на y³ + p y + q)

Дискриминант моника: Δ = 18abc − 4b³ + a²b² − 4a³c − 27c².
"""

from __future__ import annotations

import math
import sys

from ..exceptions import invalid_argument

_DBL_EPSILON = sys.float_info.epsilon

# Абсолютная добавка к порогу сравнения дискриминанта куба с нулём.
discriminant_zero_tol_absolute = 1e-11
# Множитель при scale³ в пороге для дискриминанта.
discriminant_zero_tol_scale_cubed_coeff = 1e-9
# Абсолютная добавка к порогу вырождения производной (a² − 3b).
derivative_degenerate_tol_absolute = 1e-10
# Множитель при max(scale², 1) в пороге вырождения производной.
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
    """Масштаб коэффициентов приведённого куба x³ + a x² + b x + c."""
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
    """Δ = 18abc − 4b³ + a²b² − 4a³c − 27c²  для  x³ + a x² + b x + c."""
    return (
        18.0 * a * b * c
        - 4.0 * b * b * b
        + a * a * b * b
        - 4.0 * a * a * a * c
        - 27.0 * c * c
    )


def validated_monic_coeffs_from_poly(poly: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    """Коэффициенты a, b, c приведённого куба из ``[c₀, c₁, c₂, c₃]``."""
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
    """Три действительных корня (тригонометрическая формула Виета).
    φ = arccos(R / Q³⁄²) / 3,
    xₖ = −2√Q · cos(φ + 2πk/3) − a/3,  k = 0, 1, −1.
    В коде radius = −2√Q, затем cos(φ), cos(φ+2π/3), cos(φ−2π/3).
    Сдвиг −a/3 возвращает из приведённой переменной y = x + a/3.
    """
    phi = (1.0 / 3.0) * math.acos(R / (Q ** 1.5))
    shift = -a / 3.0
    radius = -2.0 * math.sqrt(Q)
    return [
        radius * math.cos(phi) + shift,
        radius * math.cos(phi + 2.0 / 3.0 * math.pi) + shift,
        radius * math.cos(phi - 2.0 / 3.0 * math.pi) + shift,
    ]


def monic_depressed_cubic_cardano_Q(a: float, b: float, c: float) -> float:
    """Сдвиг x = y − a/3 убирает квадратичный член и даёт y³ + p y + q = 0:
    p = b − a²/3,  q = c − a b/3 + 2 a³/27,  Q = (p/3)³ + (q/2)².
    """
    p_dep = b - a * a / 3.0
    q_dep = c - a * b / 3.0 + 2.0 * a * a * a / 27.0
    return (p_dep / 3.0) ** 3 + (q_dep / 2.0) ** 2


def solve_monic_one_real_depressed_cardano(a: float, b: float, c: float) -> list[float]:
    """Один действительный корень по Кардано:
    y = ∛(−q/2 + √Q) + ∛(−q/2 − √Q),  x = y − a/3.
    max(0, Q) страхует от крошечного отрицательного дискриминанта
    из-за округления на границе S ≈ 0.
    """
    p_dep = b - a * a / 3.0
    q_dep = c - a * b / 3.0 + 2.0 * a * a * a / 27.0
    Q_card = (p_dep / 3.0) ** 3 + (q_dep / 2.0) ** 2
    rad = math.sqrt(max(0.0, Q_card))
    d1 = -q_dep / 2.0 + rad
    d2 = -q_dep / 2.0 - rad
    y = cbrt(d1) + cbrt(d2)
    return [y - a / 3.0]


def solve_monic(a: float, b: float, c: float) -> list[float]:
    """Действительные корни x³ + a x² + b x + c по знаку S = Q³ − R²."""
    Q = (a * a - 3.0 * b) / 9.0
    R = (2.0 * a ** 3 - 9.0 * a * b + 27.0 * c) / 54.0
    S = Q ** 3 - R * R
    # S ≈ 0 — кратный корень; S > 0 — три действительных; иначе Кардано.
    if abs(S) < _DBL_EPSILON:
        return solve_monic_s_near_zero(a, R)
    if S > 0:
        return solve_monic_three_real_roots(a, Q, R)
    return solve_monic_one_real_depressed_cardano(a, b, c)


def count_distinct_real_roots_cubic(poly_coeffs: list[float]) -> int:
    """Число различных действительных корней куба ``c₀ + c₁ x + c₂ x² + c₃ x³``."""
    a, b, c = validated_monic_coeffs_from_poly(poly_coeffs)
    return count_distinct_real_roots_for_monic(a, b, c)


def solve_cubic_equation(poly_coeffs: list[float]) -> list[float]:
    """Действительные корни кубического уравнения по коэффициентам ``[c₀..c₃]``."""
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
    """Экстремумы p(x) = c₀ + c₁ x + c₂ x² + c₃ x³ из p′(x) = 0.
    p′(x) = 3 D x² + 2 C x + B = 0, где D=c₃, C=c₂, B=c₁.
    Корни: x = (−2C ± √((2C)² − 4·3D·B)) / (2·3D).
    Знак p″(x) = 2C + 6D x отличает max (p″<0) от min (p″>0).
    Вырожденная квадратичная производная (3D ≈ 0) → линейное p′ = 2C x + B,
    одна критическая точка или ни одной. Возвращает (xₘₐₓ, xₘᵢₙ);
    отсутствующая точка — NaN.
    """
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
        # 3D ≈ 0: производная почти линейна  2C x + B = 0.
        if abs(b_prime) <= tol_degen:
            # p′ ≈ константа — стационарных точек нет.
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
        # p′ не имеет действительных корней — куб строго монотонен.
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
    """Тригонометрическая формула Виета: xₖ = −2√Q · cos(φ + 2πk/3) − a/3."""
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
