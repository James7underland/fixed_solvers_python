"""Тесты поэлементных операций array_ext и array_ref."""
import numpy as np

from fixed_solvers import (
    array_add,
    array_div,
    array_iadd,
    array_maker,
    array_neg,
    array_ref,
    array_scale,
    array_sub,
    create_array,
    make_array,
    matvec,
)
from fixed_solvers.array_ext import inner_prod


def test_make_array_repeats_value():
    # Arrange / Act
    values = make_array(3, 1.5)
    empty = array_maker.make_array(0, 1.0)
    # Assert
    np.testing.assert_allclose(values, [1.5, 1.5, 1.5])
    assert empty.size == 0


def test_create_array_calls_getter_from_last_index():
    # Arrange
    order = []

    def getter(index):
        order.append(index)
        return index * 2

    # Act
    values = create_array(3, getter)
    # Assert
    assert order == [2, 1, 0]
    np.testing.assert_array_equal(values, [0, 2, 4])


def test_array_algebra_matches_fixed_dimension_operators():
    # Arrange
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # Act
    summed = array_add(v1, v2)
    diff = array_sub(v2, v1)
    scaled = array_scale(2.0, v1)
    divided = array_div(v1, 2.0)
    negated = array_neg(v1)
    acc = np.array([1.0, 1.0, 1.0])
    array_iadd(acc, v1)
    product = inner_prod(v1, v2)
    mv = matvec(matrix, v1)
    # Assert
    np.testing.assert_allclose(summed, [5.0, 7.0, 9.0])
    np.testing.assert_allclose(diff, [3.0, 3.0, 3.0])
    np.testing.assert_allclose(scaled, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(divided, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(negated, [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(acc, [2.0, 3.0, 4.0])
    assert product == 32.0
    np.testing.assert_allclose(mv, [14.0, 32.0, 50.0])


def test_array_ref_writes_through_source():
    # Arrange
    source = np.array([1.0, 2.0, 3.0])
    refs = array_ref(source=source)
    # Act
    refs[1] = 8.0
    refs.assign(np.array([9.0, 8.0, 7.0]))
    # Assert
    np.testing.assert_allclose(source, [9.0, 8.0, 7.0])
    np.testing.assert_allclose(refs.to_array(), source)
    assert refs.as_scalar() == 9.0
