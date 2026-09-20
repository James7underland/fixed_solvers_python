"""Тесты ensure_abs_epsilon_value."""
from fixed_solvers import ensure_abs_epsilon_value


def test_positive_value_less_than_epsilon():
    # Arrange
    input_value = 1e-10
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == 1e-6


def test_negative_value_less_than_epsilon():
    # Arrange
    input_value = -1e-10
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == -1e-6


def test_positive_value_greater_than_epsilon():
    # Arrange
    input_value = 1.0
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == 1.0


def test_negative_value_greater_than_epsilon():
    # Arrange
    input_value = -1.0
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == -1.0
