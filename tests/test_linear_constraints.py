"""Тесты линейных ограничений ax ≤ b и обрезки 2D-шага."""
import numpy as np

from fixed_solvers import fixed_linear_constraints


def test_linear_constraints_approaches_bound():
    # Arrange
    x = np.array([0.0, 0.0])
    dx = np.array([1.0, 1.0])
    linear_constraints = fixed_linear_constraints(2, 1)
    linear_constraints.a = np.array([1.0, 1.0])
    linear_constraints.b = 1.0
    # Act
    linear_constraints.trim(x, dx)
    # Assert
    np.testing.assert_allclose(dx, [0.5, 0.5], atol=1e-12)


def test_linear_constraints_projects_from_border_case2():
    # Arrange
    x = np.array([0.0, 0.0])
    dx = np.array([0.0, 1.0])
    linear_constraints = fixed_linear_constraints(2, 1)
    linear_constraints.a = np.array([-1.0, 1.0])
    linear_constraints.b = 0.0
    # Act
    linear_constraints.trim(x, dx)
    # Assert
    np.testing.assert_allclose(dx, [0.5, 0.5], atol=1e-12)


def test_linear_constraints_projects_from_border_case3():
    # Arrange
    x = np.array([0.0, 0.0])
    dx = np.array([0.0, 1.0])
    linear_constraints = fixed_linear_constraints(2, 1)
    linear_constraints.a = np.array([1.0, 1.0])
    linear_constraints.b = 0.0
    # Act
    linear_constraints.trim(x, dx)
    # Assert
    np.testing.assert_allclose(dx, [-0.5, 0.5], atol=1e-12)


def test_linear_constraints_projects_from_border_case4():
    # Arrange
    x = np.array([0.0, 0.0])
    dx = np.array([1.0, 0.0])
    linear_constraints = fixed_linear_constraints(2, 1)
    linear_constraints.a = np.array([1.0, 1.0])
    linear_constraints.b = 0.0
    # Act
    linear_constraints.trim(x, dx)
    # Assert
    np.testing.assert_allclose(dx, [0.5, -0.5], atol=1e-12)


def test_linear_constraints_projects_from_border_case5():
    # Arrange
    x = np.array([0.0, 0.0])
    dx = np.array([1.0, 1.0])
    linear_constraints = fixed_linear_constraints(2, 1)
    linear_constraints.a = np.array([1.0, 1.0])
    linear_constraints.b = 0.0
    # Act
    linear_constraints.trim(x, dx)
    # Assert
    np.testing.assert_allclose(dx, [0.0, 0.0], atol=1e-12)
