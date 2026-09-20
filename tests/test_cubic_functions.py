"""Тесты кубических уравнений: Виета, Кардано, экстремумы."""
import math

import pytest

from fixed_solvers import (
    count_distinct_real_roots_cubic,
    find_cubic_extremums,
    invalid_argument,
    solve_cubic_equation,
)


def eval_poly3(k, x):
    """Схема Горнера: p(x) = c₀ + x (c₁ + x (c₂ + x c₃))."""
    return k[0] + x * (k[1] + x * (k[2] + x * k[3]))


def count_unique_real_from_solver(coeffs, sep_tol):
    """Сколько различных корней вернул солвер, склеивая близкие ближе sep_tol."""
    roots = solve_cubic_equation(coeffs)
    roots = sorted(roots)
    n = 0
    for i, r in enumerate(roots):
        if i == 0 or abs(r - roots[i - 1]) > sep_tol:
            n += 1
    return n


def unique_sorted_roots(roots, sep_tol):
    """Сортирует и склеивает кратные корни ближе sep_tol в один представитель."""
    u = sorted(roots)
    out = []
    for r in u:
        if not out or abs(r - out[-1]) > sep_tol:
            out.append(r)
    return out


def eval_cubic_derivative(k, x):
    """p′(x) = c₁ + 2 c₂ x + 3 c₃ x² — проверка, что экстремум действительно стационарен."""
    return k[1] + x * (2.0 * k[2] + x * (3.0 * k[3]))


def eval_cubic_second_derivative(k, x):
    """p″(x) = 2 c₂ + 6 c₃ x — знак отличает max (p″<0) от min (p″>0)."""
    return 2.0 * k[2] + x * (6.0 * k[3])


def test_returns_three_for_three_distinct_roots():
    # Arrange: (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6  →  [c₀..c₃] = [-6, 11, -6, 1]
    p = [-6.0, 11.0, -6.0, 1.0]
    # Act
    n_disc = count_distinct_real_roots_cubic(p)
    n_solver = count_unique_real_from_solver(p, 1e-7)
    # Assert
    assert n_disc == 3
    assert n_solver == 3


def test_returns_one_for_single_real_root():
    # Arrange: x³ − 1 = 0  →  единственный действительный корень x = 1
    # (два комплексных кубических корня из единицы нас не интересуют).
    p = [-1.0, 0.0, 0.0, 1.0]
    # Act
    n_disc = count_distinct_real_roots_cubic(p)
    n_solver = count_unique_real_from_solver(p, 1e-7)
    # Assert
    assert n_disc == 1
    assert n_solver == 1


def test_returns_two_for_double_and_simple_root():
    # Arrange: x³ − 3x + 2 = (x − 1)² (x + 2)
    # два различных корня: 1 (кратность 2) и −2.
    p = [2.0, -3.0, 0.0, 1.0]
    # Act
    n_disc = count_distinct_real_roots_cubic(p)
    n_solver = count_unique_real_from_solver(p, 1e-7)
    # Assert
    assert n_disc == 2
    assert n_solver == 2


def test_returns_one_for_triple_root():
    # Arrange: (x−1)³ = x³ − 3x² + 3x − 1  →  один различный корень x = 1.
    p = [-1.0, 3.0, -3.0, 1.0]
    # Act
    n_disc = count_distinct_real_roots_cubic(p)
    n_solver = count_unique_real_from_solver(p, 1e-7)
    # Assert
    assert n_disc == 1
    assert n_solver == 1


def test_ignores_leading_coefficient_scale():
    # Arrange: 2x³ − 2 = 2(x³ − 1) — тот же корень x = 1, что и у немасштабированного.
    p = [-2.0, 0.0, 0.0, 2.0]
    # Act
    n = count_distinct_real_roots_cubic(p)
    # Assert
    assert n == 1


def test_returns_three_when_zero_is_root():
    # Arrange: x³ − x = x(x−1)(x+1)  →  корни −1, 0, 1.
    p = [0.0, -1.0, 0.0, 1.0]
    # Act
    n_disc = count_distinct_real_roots_cubic(p)
    n_solver = count_unique_real_from_solver(p, 1e-7)
    # Assert
    assert n_disc == 3
    assert n_solver == 3


def test_throws_when_coefficient_count_is_not_four():
    # Arrange: куб всегда ровно 4 коэффициента [c₀..c₃]; тройка — ошибка входа.
    p = [1.0, 2.0, 3.0]
    # Act / Assert
    with pytest.raises(invalid_argument):
        count_distinct_real_roots_cubic(p)


def test_throws_when_leading_coefficient_is_zero():
    # Arrange: c₃ = 0 — это уже не куб, приведение к монику делило бы на ноль.
    p = [1.0, 2.0, 3.0, 0.0]
    # Act / Assert
    with pytest.raises(invalid_argument):
        count_distinct_real_roots_cubic(p)


def test_small_residual_for_three_distinct_roots():
    # Arrange: те же (x−1)(x−2)(x−3); солвер обязан попасть в корни с |p(x)| < 1e-9.
    p = [-6.0, 11.0, -6.0, 1.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) == 3
    for r in roots:
        assert eval_poly3(p, r) == pytest.approx(0.0, abs=1e-9)


def test_small_residual_for_single_real_root():
    # Arrange: x³ − 1 = 0, единственный действительный корень ровно 1.
    p = [-1.0, 0.0, 0.0, 1.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) == 1
    assert roots[0] == pytest.approx(1.0, abs=1e-9)
    assert eval_poly3(p, roots[0]) == pytest.approx(0.0, abs=1e-9)


def test_small_residual_for_double_root_case():
    # Arrange: (x−1)²(x+2); солвер может вернуть кратный корень дважды —
    # unique_sorted_roots склеивает их в {−2, 1}.
    p = [2.0, -3.0, 0.0, 1.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) == 2
    for r in roots:
        assert eval_poly3(p, r) == pytest.approx(0.0, abs=1e-9)
    u = unique_sorted_roots(roots, 1e-7)
    assert len(u) == 2
    assert u[0] == pytest.approx(-2.0, abs=1e-9)
    assert u[1] == pytest.approx(1.0, abs=1e-9)


def test_small_residual_for_triple_root():
    # Arrange: (x−1)³. Все возвращённые корни около 1, после склейки — один.
    p = [-1.0, 3.0, -3.0, 1.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) >= 1
    for r in roots:
        assert eval_poly3(p, r) == pytest.approx(0.0, abs=1e-9)
    u = unique_sorted_roots(roots, 1e-7)
    assert len(u) == 1
    assert u[0] == pytest.approx(1.0, abs=1e-9)


def test_scaled_equation_same_residuals():
    # Arrange: 2(x³ − 1) = 0 — тот же корень, что у x³ − 1, невязка на нём ноль.
    p = [-2.0, 0.0, 0.0, 2.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) == 1
    assert eval_poly3(p, roots[0]) == pytest.approx(0.0, abs=1e-9)
    assert roots[0] == pytest.approx(1.0, abs=1e-9)


def test_three_roots_with_zero():
    # Arrange: x(x−1)(x+1) = x³ − x, корни −1, 0, 1.
    p = [0.0, -1.0, 0.0, 1.0]
    # Act
    roots = solve_cubic_equation(p)
    # Assert
    assert len(roots) == 3
    for r in roots:
        assert eval_poly3(p, r) == pytest.approx(0.0, abs=1e-9)
    u = unique_sorted_roots(roots, 1e-7)
    assert len(u) == 3
    assert u[0] == pytest.approx(-1.0, abs=1e-9)
    assert u[1] == pytest.approx(0.0, abs=1e-9)
    assert u[2] == pytest.approx(1.0, abs=1e-9)


def test_sorted_roots_match_linear_factors():
    # Arrange: (x−1)(x−2)(x−3), после сортировки ровно 1, 2, 3.
    p = [-6.0, 11.0, -6.0, 1.0]
    # Act
    roots = sorted(solve_cubic_equation(p))
    # Assert
    assert len(roots) == 3
    assert roots[0] == pytest.approx(1.0, abs=1e-9)
    assert roots[1] == pytest.approx(2.0, abs=1e-9)
    assert roots[2] == pytest.approx(3.0, abs=1e-9)


def test_returns_max_and_min_for_two_stationary_points():
    # Arrange: p(x)=x³/3−2x²+3x, p′(x)=(x−1)(x−3), p″(x)=2x−4 → max в 1, min в 3.
    coeffs = [0.0, 3.0, -2.0, 1.0 / 3.0]
    # Act
    Q_max, Q_min = find_cubic_extremums(coeffs)
    # Assert
    assert Q_max == pytest.approx(1.0, abs=1e-9)
    assert Q_min == pytest.approx(3.0, abs=1e-9)
    assert eval_cubic_derivative(coeffs, Q_max) == pytest.approx(0.0, abs=1e-9)
    assert eval_cubic_derivative(coeffs, Q_min) == pytest.approx(0.0, abs=1e-9)
    assert eval_cubic_second_derivative(coeffs, Q_max) < 0.0
    assert eval_cubic_second_derivative(coeffs, Q_min) > 0.0


def test_returns_nans_for_no_real_extremums():
    # Arrange: p(x) = x³/3 + x,  p′(x) = x² + 1 > 0 всегда — действительных экстремумов нет.
    coeffs = [0.0, 1.0, 0.0, 1.0 / 3.0]
    # Act
    Q_max, Q_min = find_cubic_extremums(coeffs)
    # Assert
    assert math.isnan(Q_max)
    assert math.isnan(Q_min)


def test_returns_single_extremum_for_linear_derivative():
    # Arrange: почти квадратичный полином (c₃ ≈ 0), p′ почти линейна.
    # Единственная стационарная точка — минимум около x = 1.
    coeffs = [0.0, -2.0, 1.0, 1e-12]
    # Act
    Q_max, Q_min = find_cubic_extremums(coeffs)
    # Assert
    assert math.isnan(Q_max)
    assert Q_min == pytest.approx(1.0, abs=1e-7)
    assert eval_cubic_second_derivative(coeffs, Q_min) > 0.0


def test_throws_on_invalid_arguments():
    # Arrange: экстремумы тоже требуют ровно 4 коэффициента и ненулевой c₃.
    wrong_size = [1.0, 2.0, 3.0]
    zero_lead = [1.0, 2.0, 3.0, 0.0]
    # Act / Assert
    with pytest.raises(RuntimeError):
        find_cubic_extremums(wrong_size)
    with pytest.raises(RuntimeError):
        find_cubic_extremums(zero_lead)
