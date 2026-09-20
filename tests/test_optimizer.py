import numpy as np
import pytest

from fixed_solvers import (
    fixed_least_squares_function_t,
    fixed_optimize_gauss_newton,
    fixed_optimizer_parameters_t,
    fixed_optimizer_result_analysis_t,
    fixed_optimizer_result_t,
    numerical_result_code_t,
    rosenbrock_function_t,
)


class simple_sum_of_squares_function(fixed_least_squares_function_t):
    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 2.0, x[1] - 1.0])


def test_converges_simple_function():
    # Arrange
    initial = np.zeros(2)
    function = simple_sum_of_squares_function()
    parameters = fixed_optimizer_parameters_t()
    result = fixed_optimizer_result_t()
    # Act
    fixed_optimize_gauss_newton.optimize(function, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument[0] == pytest.approx(2.0, abs=parameters.argument_increment_norm)
    assert result.argument[1] == pytest.approx(1.0, abs=parameters.argument_increment_norm)


def test_converges_rosenbrock_function():
    # Arrange
    initial = np.zeros(2)
    function = rosenbrock_function_t()
    parameters = fixed_optimizer_parameters_t()
    result = fixed_optimizer_result_t()
    # Act
    fixed_optimize_gauss_newton.optimize(function, initial, parameters, result)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    assert result.argument[0] == pytest.approx(1.0, abs=parameters.argument_increment_norm)
    assert result.argument[1] == pytest.approx(1.0, abs=parameters.argument_increment_norm)


def test_performs_learning_curve_analysis():
    # Arrange
    initial = np.zeros(2)
    function = rosenbrock_function_t()
    parameters = fixed_optimizer_parameters_t()
    parameters.analysis.objective_function_history = True
    parameters.analysis.steps = True
    parameters.analysis.argument_history = True
    result = fixed_optimizer_result_t()
    analysis = fixed_optimizer_result_analysis_t()
    # Act
    fixed_optimize_gauss_newton.optimize(function, initial, parameters, result, analysis)
    # Assert
    assert result.result_code == numerical_result_code_t.Converged
    learning_curve = analysis.get_learning_curve()
    assert learning_curve
    for index in range(1, len(learning_curve)):
        assert learning_curve[index] < learning_curve[index - 1]
