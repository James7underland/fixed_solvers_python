"""Тесты ensure_abs_epsilon_value: |x| < ε поднимается до sign(x)·ε."""
from fixed_solvers import ensure_abs_epsilon_value


def test_positive_value_less_than_epsilon():
    # Arrange: 1e-10 << 1e-6, знак плюс → результат ровно +ε
    input_value = 1e-10
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == 1e-6


def test_negative_value_less_than_epsilon():
    # Arrange: отрицательный микроскоп → −ε, не «почти ноль»
    input_value = -1e-10
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == -1e-6


def test_positive_value_greater_than_epsilon():
    # Arrange: |x| ≥ ε — не трогаем
    input_value = 1.0
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == 1.0


def test_negative_value_greater_than_epsilon():
    # Arrange: |x| ≥ ε, знак минус сохраняется как есть.
    input_value = -1.0
    # Act
    result = ensure_abs_epsilon_value(input_value)
    # Assert
    assert result == -1.0
