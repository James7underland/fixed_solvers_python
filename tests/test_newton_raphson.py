"""Тесты Ньютона–Рафсона: плотный/разреженный якобиан и сходимость."""
import numpy as np
import pytest

from fixed_solvers import (
    fixed_newton_raphson,
    fixed_solver_parameters_t,
    fixed_solver_result_analysis_t,
    fixed_solver_result_t,
    fixed_system_t,
    golden_section_search,
    numerical_result_code_t,
    step_constraint_algorithm_t,
)


class simple_equation_fixed(fixed_system_t):
    """r(x) = x − (4, 5): корень очевиден, якобиан = I."""
    dimension = 2

    def residuals(self, x):
        x0 = np.array([4.0, 5.0])
        return np.asarray(x, dtype=float) - x0


class simple_equation_var(fixed_system_t):
    """Та же линейная система, но dimension = −1 (длина аргумента задаётся снаружи)."""
    dimension = -1

    def residuals(self, x):
        x0 = np.array([4.0, 5.0])
        return np.asarray(x, dtype=float) - x0


class diagnostic_equation_var(fixed_system_t):
    """Линейная r(x) = x − (1, −2) плюс крючок custom_line_research:

    на каждом шаге Ньютона запоминает сетку α = 0, 0.01, …, 1,
    чтобы сверить её с analysis.line_search_explore.
    """
    dimension = -1

    def __init__(self):
        super().__init__(dimension=-1)
        self.alphas_per_step = []

    def residuals(self, x):
        x0 = np.array([1.0, -2.0])
        return np.asarray(x, dtype=float) - x0

    def custom_line_research(self, argument, argument_increment):
        alphas = []
        research_step_count = 100
        for index in range(research_step_count + 1):
            alphas.append(1.0 * index / research_step_count)
        self.alphas_per_step.append(alphas)


class cubic_equation_fixed(fixed_system_t):
    """(x−3)³ = 0. Целевая |r|, не r² — иначе линейный поиск ведёт себя иначе у нуля."""
    dimension = 1

    def residuals(self, x):
        return (float(x) - 3.0) ** 3

    def objective_function(self, r) -> float:
        return abs(float(r))


class simple_equation_with_custom_criteria(fixed_system_t):
    """Останов только по custom_success_criteria: max |pᵢ| < 1e-6.

    residuals_norm и argument_increment_norm в тесте выключены (NaN),
    иначе Ньютон сошёлся бы раньше по стандартной метрике шага.
    """
    dimension = -1

    def residuals(self, x):
        x0 = np.array([4.0, 5.0])
        return np.asarray(x, dtype=float) - x0

    def custom_success_criteria(self, r, x, p) -> bool:
        return float(np.max(np.abs(p))) < 1e-6

    def get_solution_for_testing(self):
        return np.array([4.0, 5.0])


class sample_system(fixed_system_t):
    """(x−2)³ = 0, (y−1)³ = 0 — учебный пример из README."""
    dimension = 2

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([(x[0] - 2.0) ** 3, (x[1] - 1.0) ** 3])


def test_use_case_converges_sample_system():
    # Arrange: старт (0, 0), корень (2, 1); кубическая невязка, Ньютон всё равно сходится.
    system = sample_system()
    parameters = fixed_solver_parameters_t(2, 0)
    result = fixed_solver_result_t(2)
    # Act
    fixed_newton_raphson[2].solve_dense(system, np.array([0.0, 0.0]), parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    np.testing.assert_allclose(result.argument, [2.0, 1.0], atol=1e-3)


def test_handles_constrained_equations_fixed_quadprog():
    # Arrange
    # Без box корень (4, 5). Потолок x₀≤3 и пол x₁≥8 несовместимы с корнем —
    # шаг QP должен упереться в (3, 8).
    eq = simple_equation_fixed()
    parameters = fixed_solver_parameters_t(2, 0, golden_section_search)
    parameters.step_constraint_as_optimization = True
    parameters.step_constraint_algorithm = step_constraint_algorithm_t.Quadprog
    parameters.constraints.maximum[0] = 3
    parameters.constraints.minimum[1] = 8
    x0 = np.array([0.0, 0.0])
    x0 = parameters.constraints.ensure_constraints(x0)
    result = fixed_solver_result_t(2)
    # Act
    fixed_newton_raphson[2].solve(eq, x0, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument[0] == pytest.approx(parameters.constraints.maximum[0], abs=1e-8)
    assert result.argument[1] == pytest.approx(parameters.constraints.minimum[1], abs=1e-8)


def test_handles_constrained_equations_coordinate_descent():
    # Arrange: те же box x₀≤3, x₁≥8, но шаг у границы ищем покоординатным lstsq,
    # не dual-QP. Ответ тот же (3, 8) — оба алгоритма упираются в активный box.
    eq = simple_equation_fixed()
    parameters = fixed_solver_parameters_t(2, 0, golden_section_search)
    parameters.step_constraint_as_optimization = True
    parameters.step_constraint_algorithm = step_constraint_algorithm_t.CoordinateDescent
    parameters.constraints.maximum[0] = 3
    parameters.constraints.minimum[1] = 8
    x0 = np.array([0.0, 0.0])
    x0 = parameters.constraints.ensure_constraints(x0)
    result = fixed_solver_result_t(2)
    # Act
    fixed_newton_raphson[2].solve(eq, x0, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument[0] == pytest.approx(parameters.constraints.maximum[0], abs=1e-8)
    assert result.argument[1] == pytest.approx(parameters.constraints.minimum[1], abs=1e-8)


def test_handles_constrained_equations_var():
    # Arrange: dimension = −1, потолок только на x₀ ≤ 3. Свободная компонента x₁
    # должна выйти в корень 5 (residuals[1] ≈ 0), а x₀ упереться в 3.
    eq = simple_equation_var()
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    parameters.step_constraint_as_optimization = True
    parameters.step_constraint_algorithm = step_constraint_algorithm_t.Quadprog
    parameters.constraints.maximum = [(0, 3.0)]
    initial = np.zeros(2)
    initial = parameters.constraints.ensure_constraints(initial)
    result = fixed_solver_result_t(-1, argument_size=2)
    # Act
    fixed_newton_raphson[-1].solve(eq, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument[0] == pytest.approx(parameters.constraints.maximum[0][1], abs=1e-8)
    assert result.residuals[1] == pytest.approx(0.0, abs=1e-8)


def test_handles_line_search_custom_diagnostics():
    # Arrange: включаем сетку α вдоль луча. Солвер пишет 101 значение f в analysis,
    # система дублирует те же α в custom_line_research — длины должны совпасть.
    eq = diagnostic_equation_var()
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    parameters.analysis.line_search_explore = True
    initial = np.zeros(2)
    result = fixed_solver_result_t(-1, argument_size=2)
    analysis = fixed_solver_result_analysis_t()
    # Act
    fixed_newton_raphson[-1].solve(eq, initial, parameters, result, analysis)
    # Assert
    assert analysis.target_function
    assert eq.alphas_per_step
    assert len(analysis.target_function) == len(eq.alphas_per_step)
    assert len(analysis.target_function[0]) == len(eq.alphas_per_step[0])
    assert len(eq.alphas_per_step[0]) > 0


def test_handles_residuals_norm():
    # Arrange: (x−3)³, целевая |r| вместо r². Порог residuals_norm = 1.5 с early exit:
    # Ньютон имеет право остановиться, не доведя ρ-метрику шага до ε.
    equation = cubic_equation_fixed()
    parameters = fixed_solver_parameters_t(1, 0, golden_section_search)
    parameters.residuals_norm = 1.5
    parameters.residuals_norm_allow_early_exit = True
    initial_x = 10.0
    result = fixed_solver_result_t(1)
    # Act
    fixed_newton_raphson[1].solve(equation, initial_x, parameters, result)
    # Assert
    norm = equation(result.argument)
    assert norm <= parameters.residuals_norm
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument_increment_criteria is False
    assert result.residuals_norm_criteria is True


def test_handles_stop_by_custom_criteria_when_other_criteria_disabled():
    # Arrange: оба стандартных критерия выключены (NaN). Сходимость — только
    # когда max|p| < 1e-6, то есть шаг Ньютона уже численный нуль.
    equation = simple_equation_with_custom_criteria()
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    parameters.residuals_norm = float("nan")
    parameters.argument_increment_norm = float("nan")
    parameters.step_criteria_assuming_search_step = False
    initial = np.zeros(2)
    result = fixed_solver_result_t(-1, argument_size=2)
    # Act
    fixed_newton_raphson[-1].solve(equation, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    solution = equation.get_solution_for_testing()
    assert result.argument[0] == pytest.approx(solution[0], abs=1e-8)
    assert result.argument[1] == pytest.approx(solution[1], abs=1e-8)
