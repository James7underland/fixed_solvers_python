"""Тесты порога шума function_target_value в золотом сечении."""
import math

from fixed_solvers import (
    domain_discovery_mode_t,
    golden_section_domain_discovery_parameters,
    golden_section_parameters,
    golden_section_search,
    golden_section_search_domain_discovery,
)


def test_returns_zero_step_when_both_boundary_values_below_target():
    # Arrange
    parameters = golden_section_parameters()
    parameters.iteration_count = 10
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = 1e-8
    f_a = 1e-10
    f_b = 1e-11
    function = lambda x: (x - 0.5) ** 2
    # Act
    step, iterations = golden_section_search.search(parameters, function, 0.0, 1.0, f_a, f_b)
    # Assert
    assert step == 1.0
    assert iterations == 0


def test_returns_smaller_boundary_when_only_one_boundary_value_below_target():
    # Arrange
    parameters = golden_section_parameters()
    parameters.iteration_count = 10
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = 1e-8
    f_a = 1e-10
    f_b = 1.0
    function = lambda x: (x - 0.5) ** 2
    # Act
    step, iterations = golden_section_search.search(parameters, function, 0.0, 1.0, f_a, f_b)
    # Assert
    assert step == 0.0
    assert iterations == 0


def test_returns_right_boundary_when_only_right_boundary_value_below_target():
    # Arrange
    parameters = golden_section_parameters()
    parameters.iteration_count = 10
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = 1e-8
    f_a = 1.0
    f_b = 1e-10
    function = lambda x: (x - 0.5) ** 2
    # Act
    step, iterations = golden_section_search.search(parameters, function, 0.0, 1.0, f_a, f_b)
    # Assert
    assert step == 1.0
    assert iterations == 0


def test_runs_iterations_when_both_boundary_values_above_target():
    # Arrange
    parameters = golden_section_parameters()
    parameters.iteration_count = 3
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = 1e-8
    function = lambda x: (x - 0.3) ** 2 + 0.1
    # Act
    step, iterations = golden_section_search.search(parameters, function, 0.0, 1.0, function(0.0))
    # Assert
    assert math.isfinite(step)
    assert iterations > 0


def test_domain_discovery_returns_zero_step_when_both_boundary_values_below_target():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 10
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = 1e-8
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    f_a = 1e-10
    f_b = 1e-11
    function = lambda x: (x - 0.5) ** 2
    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, f_a, f_b
    )
    # Assert
    assert step == 1.0
    assert iterations == 0
