"""Тесты Ньютона при domain_violation и политике line_search_explore."""
import math

import numpy as np
import pytest

from fixed_solvers import (
    domain_discovery_mode_t,
    domain_violation,
    fixed_newton_raphson,
    fixed_solver_parameters_t,
    fixed_solver_result_analysis_t,
    fixed_solver_result_t,
    fixed_system_t,
    golden_section_search,
    golden_section_search_domain_discovery,
    line_search_explore_domain_violation_action_t,
    numerical_result_code_t,
)


class residual_domain_violation_var(fixed_system_t):
    dimension = -1

    def __init__(self):
        super().__init__(dimension=-1)
        self.upper_bound_for_domain = math.inf
        self.linear_residual_root = 1.0

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        if x[0] < 0.0:
            raise domain_violation
        if math.isfinite(self.upper_bound_for_domain) and x[0] > self.upper_bound_for_domain:
            raise domain_violation
        return np.array([x[0] - self.linear_residual_root])


class jacobian_domain_violation_var(fixed_system_t):
    dimension = -1

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0])

    def jacobian_sparse(self, x):
        raise domain_violation


class residual_domain_violation_fixed2(fixed_system_t):
    dimension = 2

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        if x[0] < 0.0:
            raise domain_violation
        return np.array([x[0] - 1.0, x[1] - 2.0])


class jacobian_domain_violation_fixed2(fixed_system_t):
    dimension = 2

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, x[1] - 2.0])

    def jacobian_dense(self, x):
        raise domain_violation


def test_catches_residual_violation_at_initial_point_var():
    # Arrange
    equation = residual_domain_violation_var()
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    initial = np.array([-1.0])
    result = fixed_solver_result_t(-1, argument_size=1)
    # Act
    fixed_newton_raphson[-1].solve(equation, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.NumericalNanValues


def test_catches_jacobian_violation_var():
    # Arrange
    equation = jacobian_domain_violation_var()
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    initial = np.array([0.5])
    result = fixed_solver_result_t(-1, argument_size=1)
    # Act
    fixed_newton_raphson[-1].solve(equation, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.NumericalNanValues


def test_catches_residual_violation_at_initial_point_fixed2():
    # Arrange
    equation = residual_domain_violation_fixed2()
    parameters = fixed_solver_parameters_t(2, 0, golden_section_search)
    initial = np.array([-1.0, 0.0])
    result = fixed_solver_result_t(2)
    # Act
    fixed_newton_raphson[2].solve(equation, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.NumericalNanValues


def test_catches_jacobian_violation_fixed2():
    # Arrange
    equation = jacobian_domain_violation_fixed2()
    parameters = fixed_solver_parameters_t(2, 0, golden_section_search)
    initial = np.array([0.0, 0.0])
    result = fixed_solver_result_t(2)
    # Act
    fixed_newton_raphson[2].solve(equation, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.NumericalNanValues


def test_records_nan_and_indices_when_line_search_explore_hits_domain():
    # Arrange
    equation = residual_domain_violation_var()
    equation.upper_bound_for_domain = 1.0
    equation.linear_residual_root = 1.2
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search_domain_discovery)
    parameters.iteration_count = 3
    parameters.analysis.line_search_explore = True
    parameters.analysis.line_search_explore_on_domain_violation = (
        line_search_explore_domain_violation_action_t.record_nan
    )
    parameters.line_search.mode = domain_discovery_mode_t.allow_disconnected_domain
    initial = np.array([0.95])
    result = fixed_solver_result_t(-1, argument_size=1)
    analysis = fixed_solver_result_analysis_t()
    # Act
    fixed_newton_raphson[-1].solve(equation, initial, parameters, result, analysis)
    # Assert
    assert len(analysis.target_function) == len(analysis.line_search_explore_domain_violation_indices)
    found_violation = False
    for step, indices in enumerate(analysis.line_search_explore_domain_violation_indices):
        if not indices:
            continue
        found_violation = True
        for grid_index in indices:
            assert grid_index < len(analysis.target_function[step])
            assert not math.isfinite(analysis.target_function[step][grid_index])
    assert found_violation


def test_propagates_domain_violation_when_line_search_explore_rethrow():
    # Arrange
    equation = residual_domain_violation_var()
    equation.upper_bound_for_domain = 1.0
    equation.linear_residual_root = 1.2
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    parameters.iteration_count = 3
    parameters.analysis.line_search_explore = True
    parameters.analysis.line_search_explore_on_domain_violation = (
        line_search_explore_domain_violation_action_t.rethrow
    )
    initial = np.array([0.95])
    result = fixed_solver_result_t(-1, argument_size=1)
    # Act / Assert
    with pytest.raises(domain_violation):
        fixed_newton_raphson[-1].solve(equation, initial, parameters, result)


def test_domain_violation_is_not_a_regular_exception():
    # Arrange / Act / Assert
    assert issubclass(domain_violation, BaseException)
    assert not issubclass(domain_violation, Exception)
    caught_as_exception = False
    try:
        raise domain_violation()
    except Exception:
        caught_as_exception = True
    except domain_violation:
        caught_as_exception = False
    assert caught_as_exception is False
