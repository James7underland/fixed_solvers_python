import numpy as np

from fixed_solvers import solve_linear_system


def test_linear_equation_solver_2x2():
    # Arrange
    A = np.array([[2.0, 2.0], [3.0, 4.0]])
    b = np.array([6.0, 11.0])
    # Act
    x = solve_linear_system(A, b)
    # Assert
    np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-12)
