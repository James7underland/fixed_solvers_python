"""Тесты разреженной сборки box-ограничений."""
import numpy as np

from fixed_solvers import fixed_solver_constraints


def test_variable_dimension_sparse_inequalities_are_relative():
    # Arrange
    constraints = fixed_solver_constraints(-1)
    constraints.minimum = [(0, 1.0)]
    constraints.maximum = [(1, 5.0)]
    argument = np.array([2.0, 3.0])
    # Act
    A, b = constraints.get_inequalities_constraints_sparse(argument)
    # Assert
    dense = A.toarray()
    np.testing.assert_allclose(dense, [[-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(b, [1.0, 2.0])


def test_fixed_dimension_sparse_inequalities_skip_nan():
    # Arrange
    constraints = fixed_solver_constraints(2)
    constraints.minimum[0] = 1.0
    constraints.maximum[1] = 5.0
    argument = np.array([2.0, 3.0])
    # Act
    A, b = constraints.get_inequalities_constraints_sparse(argument)
    # Assert
    dense = A.toarray()
    np.testing.assert_allclose(dense, [[-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(b, [1.0, 2.0])
