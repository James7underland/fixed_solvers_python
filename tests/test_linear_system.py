"""Тесты СЛАУ малой размерности методом Крамера."""
import numpy as np

from fixed_solvers import solve_linear_system


def test_linear_equation_solver_2x2():
    # Arrange: 2x+2y=6, 3x+4y=11 → (x,y)=(1,2); Крамер: det A=2.
    #     Крамер:  det A = 2,  x = 1,  y = 2
    #
    A = np.array([[2.0, 2.0], [3.0, 4.0]])
    b = np.array([6.0, 11.0])
    # Act
    x = solve_linear_system(A, b)
    # Assert
    np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-12)
