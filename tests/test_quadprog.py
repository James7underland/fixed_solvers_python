import math

import numpy as np
from scipy import sparse

from fixed_solvers import get_sparse_matrix_CCS, solve_quadprog, solve_quadprog_box


def test_solves_unconstrained_qp():
    # Arrange
    G = np.eye(2)
    g0 = np.array([-3.0, -4.0])
    x = np.zeros(2)
    CE = np.zeros((2, 0))
    ce0 = np.zeros(0)
    CI = np.zeros((2, 0))
    ci0 = np.zeros(0)
    # Act
    value = solve_quadprog(G, g0, CE, ce0, CI, ci0, x)
    # Assert
    np.testing.assert_allclose(x, [3.0, 4.0], atol=1e-10)
    expected = 0.5 * float(x @ G @ x) + float(g0 @ x)
    assert abs(value - expected) < 1e-10


def test_solves_equality_and_inequality_quadprog_example():
    # Arrange: классический пример QuadProg++ / eiquadprog
    G = np.array([[4.0, -2.0], [-2.0, 4.0]])
    g0 = np.array([6.0, 0.0])
    CE = np.array([[1.0], [1.0]])
    ce0 = np.array([-3.0])
    CI = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    ci0 = np.array([0.0, 0.0, -2.0])
    x = np.zeros(2)
    # Act
    solve_quadprog(G, g0, CE, ce0, CI, ci0, x)
    # Assert
    np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-8)


def test_solves_box_qp_with_goldfarb_idnani():
    # Arrange
    H = sparse.eye(2, format="csc")
    f = np.array([-3.0, -4.0])
    # Act
    x = solve_quadprog_box(H, f, minimum=[], maximum=[(0, 1.0)])
    # Assert
    np.testing.assert_allclose(x, [1.0, 4.0], atol=1e-8)


def test_solves_box_qp_with_lower_bound():
    # Arrange
    H = np.eye(2)
    f = np.array([-3.0, -4.0])
    # Act
    x = solve_quadprog_box(H, f, minimum=[(1, 5.0)], maximum=[])
    # Assert
    np.testing.assert_allclose(x, [3.0, 5.0], atol=1e-8)


def test_returns_infinity_when_qp_is_infeasible():
    # Arrange
    G = np.eye(1)
    g0 = np.array([0.0])
    x = np.zeros(1)
    CE = np.zeros((1, 0))
    ce0 = np.zeros(0)
    CI = np.array([[1.0, -1.0]])
    ci0 = np.array([-2.0, 1.0])
    # Act
    value = solve_quadprog(G, g0, CE, ce0, CI, ci0, x)
    # Assert
    assert math.isinf(value)


def test_converts_sparse_matrix_to_ccs():
    # Arrange
    matrix = np.array([[2.0, 0.0, 3.0], [0.0, 5.0, 0.0], [7.0, 0.0, 9.0]])
    # Act
    values, rows, cols = get_sparse_matrix_CCS(matrix)
    # Assert
    assert values == [2.0, 7.0, 5.0, 3.0, 9.0]
    assert rows == [0, 2, 1, 0, 2]
    assert cols == [0, 2, 3, 5]
