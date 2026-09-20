import math

import pytest

from fixed_solvers import (
    fixed_bisection_result_analysis_t,
    fixed_bisection_result_t,
    fixed_bisectional,
    fixed_bisectional_parameters_t,
    fixed_bisectional_solution_type,
    fixed_system_t,
)


class simple_linear(fixed_system_t):
    dimension = 1

    def residuals(self, x):
        return 5.0 - float(x)

    def expected_solution(self):
        return 5.0


class simple_quadratic(fixed_system_t):
    dimension = 1

    def residuals(self, x):
        return 3.5 - float(x) * float(x)

    def expected_solution(self):
        return math.sqrt(3.5)


def test_solves_linear_equation():
    # Arrange
    eqn = simple_linear()
    p = fixed_bisectional_parameters_t()
    p.argument_limit_min = 0
    p.argument_limit_max = 7.8
    p.argument_history = True
    p.secant_treshhold_iterations = 0
    p.solution_type = fixed_bisectional_solution_type.Secant
    p.verbose = False
    res = fixed_bisection_result_t(1)
    ana = fixed_bisection_result_analysis_t()
    # Act
    fixed_bisectional[1].solve(p, eqn, res, ana)
    # Assert
    assert res.argument == eqn.expected_solution()


def test_solves_quadratic_equation():
    # Arrange
    eqn = simple_quadratic()
    p = fixed_bisectional_parameters_t()
    p.argument_limit_min = 0
    p.argument_limit_max = 27.8
    p.argument_history = True
    p.solution_type = fixed_bisectional_solution_type.Combined
    p.secant_treshhold_max = 0.01
    p.secant_treshhold_min = 0.001
    p.verbose = False
    res = fixed_bisection_result_t(1)
    ana = fixed_bisection_result_analysis_t()
    # Act
    fixed_bisectional[1].solve(p, eqn, res, ana)
    # Assert
    assert res.argument == pytest.approx(eqn.expected_solution(), abs=p.residual_precision)


def test_prints_verbose_trace_to_stdout_and_stderr(capsys):
    # Arrange
    eqn = simple_linear()
    p = fixed_bisectional_parameters_t()
    p.argument_limit_min = 0
    p.argument_limit_max = 7.8
    p.solution_type = fixed_bisectional_solution_type.Bisection
    p.verbose = True
    res = fixed_bisection_result_t(1)
    # Act
    fixed_bisectional[1].solve(p, eqn, res)
    captured = capsys.readouterr()
    # Assert
    assert "next_argument_value d:" in captured.out
    assert " of " in captured.out
    assert "check:" in captured.err
    assert "with:" in captured.err
    assert res.argument == pytest.approx(eqn.expected_solution(), abs=p.residual_precision)
