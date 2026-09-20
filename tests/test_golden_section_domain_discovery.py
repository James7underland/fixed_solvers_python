"""Тесты золотого сечения с обнаружением ООФ и режимами domain_discovery."""
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
    # Две ямы: [0, 0.2) и (0.6, 1]; дырка между ними. Режим связной ООФ
    # не может честно локализовать минимум → контракт fail (NaN, N+1).
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
    # Arrange: связная ООФ, но функция снова определена после дырки → logic_error.
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
    # Arrange: forbid_exit — любой domain_violation это программная ошибка,
    # не «сузь отрезок». ООФ x < 0.5, луч [0, 1] обязан выйти за границу.
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
    # Arrange: NaN — сбой расчёта, не ООФ. Контракт тот же, что у нелокализуемого
    # интервала: (NaN, iteration_count+1), без logic_error.
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
    # Arrange: парабола (x−0.2)² на [0, 1], одна итерация ЗС.
    # Минимум слева → шаг должен быть заметно меньше 0.7.
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
    # Arrange: минимум параболы у правого конца (x−0.9)² → шаг > 0.3.
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
    # Arrange: минимум слева (x−0.2)², дырка ООФ только у самого правого конца (x≥0.99).
    # Связная ООФ, ЗС должно сузить отрезок и вернуть конечный шаг.
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
    # Arrange: минимум ближе к дырке (x−0.85)², ООФ обрывается в 0.99.
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
    # Arrange: две ямы, поиск только по [0, 0.56] — правая яма почти не видна.
    # Минимум на левом куске около 0.18; режим связной ООФ, один шаг ЗС.
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
    # Arrange: allow_disconnected_domain, левый кусок линейно растёт, справа почти ноль.
    # Эвристика имеет право отрезать дырку и вернуть конечный шаг.
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
    # Arrange: связная ООФ + две ямы на [0, 0.56]: внутренние точки не могут честно
    # указать, где минимум — «minimum is not in [a, b]».
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
    # Arrange: allow_disconnected_domain, минимум в правой яме (x−0.9)²,
    # обе внутренние точки ЗС могут попасть в дырку — эвристика отрезает справа.
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
    # Arrange: левая яма выше (2+(x−0.05)²), правая ниже (0.1+(x−0.9)²).
    # Эвристика disconnected domain должна оставить шанс шагу вправо.
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
    # Arrange: левый кусок почти константа 0.01x, правый горб 0.5+(x−0.6)² —
    # внутренние точки не указывают однозначный минимум, даже с эвристикой.
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
    # Arrange: ООФ x < 0.55, минимум (x−0.2)² внутри. allow_disconnected_domain
    # спокойно отрезает правый хвост, где function бросает domain_violation.
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
    # Arrange: та же функция, что rule11, но require_connected_domain:
    # после первой дырки повторный провал (или «вышли и вернулись») — logic_error.
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
    # Arrange: правая яма не доходит до b=1 (обрыв в 0.9). Эвристика disconnected.
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
    # Arrange: средняя яма (0.55, 0.8), не правый конец. Минимум около 0.62.
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
    # Arrange: правая яма крошечная (x>0.95) и на порядок хуже левой.
    # Две итерации ЗС с эвристикой должны удержать конечный шаг на левом куске.
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
    # Arrange: ООФ только [0, 0.3), минимум в 0.15. Три сжатия золотого сечения.
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
