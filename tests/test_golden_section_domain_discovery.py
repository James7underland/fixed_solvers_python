import math

import pytest

from fixed_solvers import (
    domain_discovery_mode_t,
    domain_violation,
    golden_section_domain_discovery_parameters,
    golden_section_search_domain_discovery,
    logic_error,
)


def test_returns_fail_contract_when_interval_cannot_be_localized():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 14
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.6) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return (x - 0.03) ** 2
        return 10.0 + (x - 0.9) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert not math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_throws_logic_error_when_disconnected_domain_detected():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 12
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    domain_border = 0.55

    def function(x):
        if x >= domain_border:
            raise domain_violation
        return (x - 0.2) ** 2

    # Act / Assert
    with pytest.raises(logic_error):
        golden_section_search_domain_discovery.search(
            parameters, function, 0.0, 1.0, function(0.0)
        )


def test_throws_runtime_error_when_mode_is_forbid_exit():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 8
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.forbid_exit
    domain_border = 0.5

    def function(x):
        if x >= domain_border:
            raise domain_violation
        return (x - 0.1) ** 2

    # Act / Assert
    with pytest.raises(RuntimeError):
        golden_section_search_domain_discovery.search(
            parameters, function, 0.0, 1.0, function(0.0)
        )


def test_returns_fail_contract_when_function_returns_nan():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 10
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain

    def function(x):
        if x > 0.6:
            return float("nan")
        return (x - 0.2) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert not math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule01():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    function = lambda x: (x - 0.2) ** 2
    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1
    assert step < 0.7


def test_returns_finite_step_rule02():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    function = lambda x: (x - 0.9) ** 2
    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1
    assert step > 0.3


def test_returns_finite_step_rule03():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain

    def function(x):
        if x >= 0.99:
            raise domain_violation
        return (x - 0.2) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule04():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain

    def function(x):
        if x >= 0.99:
            raise domain_violation
        return (x - 0.85) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule05():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    search_a = 0.0
    search_b = 0.56

    def function(x):
        in_left = (x >= 0.0) and (x < 0.22)
        in_right = (x > 0.55) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return (x - 0.18) ** 2
        return (x - 0.9) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, search_a, search_b, function(search_a)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule06():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain
    search_a = 0.0
    search_b = 0.56

    def function(x):
        in_left = (x >= 0.0) and (x < 0.22)
        in_right = (x > 0.55) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return 0.1 * x + 0.2
        return 1e-6

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, search_a, search_b, function(search_a)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_throws_logic_error_rule07():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain
    search_a = 0.0
    search_b = 0.56

    def function(x):
        in_left = (x >= 0.0) and (x < 0.22)
        in_right = (x > 0.55) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return 0.1 * x
        return 1.0 + (x - 0.9) ** 2

    # Act / Assert
    with pytest.raises(logic_error):
        golden_section_search_domain_discovery.search(
            parameters, function, search_a, search_b, function(search_a)
        )


def test_returns_finite_step_rule08():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.6) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        return (x - 0.9) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule09():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.6) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return 2.0 + (x - 0.05) ** 2
        return 0.1 + (x - 0.9) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_throws_logic_error_rule10():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.6) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return 0.01 * x
        return 0.5 + (x - 0.6) ** 2

    # Act / Assert
    with pytest.raises(logic_error):
        golden_section_search_domain_discovery.search(
            parameters, function, 0.0, 1.0, function(0.0)
        )


def test_returns_finite_step_rule11():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        if x >= 0.55:
            raise domain_violation
        return (x - 0.2) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_throws_logic_error_rule12():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.require_connected_domain

    def function(x):
        if x >= 0.55:
            raise domain_violation
        return (x - 0.2) ** 2

    # Act / Assert
    with pytest.raises(logic_error):
        golden_section_search_domain_discovery.search(
            parameters, function, 0.0, 1.0, function(0.0)
        )


def test_returns_finite_step_rule13():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.6) and (x < 0.9)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return 2.0 + (x - 0.05) ** 2
        return 0.1 + (x - 0.8) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule14():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 1
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_mid = (x > 0.55) and (x < 0.8)
        if not in_left and not in_mid:
            raise domain_violation
        if in_left:
            return 2.0 + (x - 0.05) ** 2
        return 0.2 + (x - 0.62) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule15():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 2
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        in_left = (x >= 0.0) and (x < 0.2)
        in_right = (x > 0.95) and (x <= 1.0)
        if not in_left and not in_right:
            raise domain_violation
        if in_left:
            return (x - 0.15) ** 2
        return 10.0 + (x - 1.0) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1


def test_returns_finite_step_rule16():
    # Arrange
    parameters = golden_section_domain_discovery_parameters()
    parameters.iteration_count = 3
    parameters.function_decrement_factor = float("nan")
    parameters.function_target_value = float("nan")
    parameters.mode = domain_discovery_mode_t.allow_disconnected_domain

    def function(x):
        if x >= 0.3:
            raise domain_violation
        return (x - 0.15) ** 2

    # Act
    step, iterations = golden_section_search_domain_discovery.search(
        parameters, function, 0.0, 1.0, function(0.0)
    )
    # Assert
    assert math.isfinite(step)
    assert iterations == parameters.iteration_count + 1
