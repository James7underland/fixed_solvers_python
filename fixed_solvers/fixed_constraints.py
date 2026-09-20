"""Ограничения солвера: box, relative и линейные ax <= b."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import sparse

from .algebra import default_var, inner_prod, is_scalar, var_copy, var_div, var_size
from .exceptions import logic_error
from .fixed_linear_solver import solve_linear_system
from .helpers.math_helpers import sgn

eps_constraints = 1e-8


def prepare_box_constraints(
    minimum: Sequence[tuple[int, float]],
    maximum: Sequence[tuple[int, float]],
    callback: Callable[[int, float, float], None],
) -> None:
    iimin = 0
    iimax = 0
    have_min = iimin < len(minimum)
    have_max = iimax < len(maximum)
    while have_min or have_max:
        min_index = math.inf
        max_index = math.inf
        if have_min:
            min_index = minimum[iimin][0]
        if have_max:
            max_index = maximum[iimax][0]

        min_value = -math.inf
        max_value = math.inf
        if have_min and have_max:
            if min_index < max_index:
                var_index = min_index
                min_value = minimum[iimin][1]
                iimin += 1
            elif max_index < min_index:
                var_index = max_index
                max_value = maximum[iimax][1]
                iimax += 1
            else:
                var_index = min_index
                min_value = minimum[iimin][1]
                max_value = maximum[iimax][1]
                iimin += 1
                iimax += 1
        elif have_min:
            var_index = min_index
            min_value = minimum[iimin][1]
            iimin += 1
        else:
            var_index = max_index
            max_value = maximum[iimax][1]
            iimax += 1
        callback(int(var_index), float(min_value), float(max_value))
        have_min = iimin < len(minimum)
        have_max = iimax < len(maximum)


class fixed_solver_constraints:
    """Ограничения min/max/relative.

    Для dimension == -1 поля ``minimum``, ``maximum``, ``relative_boundary`` —
    списки пар ``(index, value)``. Для фиксированной размерности — скаляр
    или массив той же длины, ``NaN`` означает «ограничение не задано».
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        if dimension == -1:
            self.relative_boundary: list[tuple[int, float]] = []
            self.minimum: list[tuple[int, float]] = []
            self.maximum: list[tuple[int, float]] = []
        else:
            self.relative_boundary = default_var(dimension)
            self.minimum = default_var(dimension)
            self.maximum = default_var(dimension)

    def get_constraint_count(self) -> int:
        if self.dimension == -1:
            return len(self.minimum) + len(self.maximum)
        count = 0
        mins = np.atleast_1d(self.minimum)
        maxs = np.atleast_1d(self.maximum)
        for value in mins:
            if math.isfinite(float(value)):
                count += 1
        for value in maxs:
            if math.isfinite(float(value)):
                count += 1
        return count

    def get_relative_constraints(self, current_argument) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        if self.dimension == 1:
            raise RuntimeError("not impl")
        if self.dimension == -1:
            mins = [(i, v - float(current_argument[i])) for i, v in self.minimum]
            maxs = [(i, v - float(current_argument[i])) for i, v in self.maximum]
            return mins, maxs
        mins: list[tuple[int, float]] = []
        maxs: list[tuple[int, float]] = []
        for index in range(self.dimension):
            if math.isfinite(float(self.minimum[index])):
                mins.append((index, float(self.minimum[index]) - float(current_argument[index])))
            if math.isfinite(float(self.maximum[index])):
                maxs.append((index, float(self.maximum[index]) - float(current_argument[index])))
        return mins, maxs

    @staticmethod
    def get_inequalities_constraints_vectors_dense(
        argument_dimension: int,
        boundaries: Sequence[tuple[int, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        A = np.zeros((len(boundaries), argument_dimension), dtype=float)
        b = np.zeros(len(boundaries), dtype=float)
        for row_index, (idx, value) in enumerate(boundaries):
            A[row_index, idx] = 1.0
            b[row_index] = value
        return A, b

    def get_inequalities_constraints_dense(self, argument_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = argument_size
        A = np.zeros((self.get_constraint_count(), n), dtype=float)
        B = np.zeros(self.get_constraint_count(), dtype=float)
        offset = 0
        a_max, b_max = self.get_inequalities_constraints_vectors_dense(n, self.maximum)
        A[offset:offset + a_max.shape[0], :] = a_max
        B[offset:offset + b_max.size] = b_max
        offset += a_max.shape[0]
        a_min, b_min = self.get_inequalities_constraints_vectors_dense(n, self.minimum)
        A[offset:offset + a_min.shape[0], :] = -a_min
        B[offset:offset + b_min.size] = -b_min
        return A, B

    def get_inequalities_constraints_sparse(self, current_argument) -> tuple[sparse.csc_matrix, np.ndarray]:
        arg = np.asarray(current_argument, dtype=float).reshape(-1)
        n = int(arg.size)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []

        def add_pair(var_index: int, value: float, sign: float) -> None:
            row_index = len(rhs)
            rows.append(row_index)
            cols.append(int(var_index))
            data.append(sign * 1.0)
            rhs.append(sign * float(value))

        if self.dimension == -1:
            for var_index, value in self.minimum:
                add_pair(var_index, value, -1.0)
            for var_index, value in self.maximum:
                add_pair(var_index, value, 1.0)
        else:
            mins = np.atleast_1d(np.asarray(self.minimum, dtype=float))
            maxs = np.atleast_1d(np.asarray(self.maximum, dtype=float))
            dim = 1 if self.dimension == 1 else self.dimension
            for var_index in range(dim):
                value = float(mins[var_index])
                if not math.isnan(value):
                    add_pair(var_index, value, -1.0)
            for var_index in range(dim):
                value = float(maxs[var_index])
                if not math.isnan(value):
                    add_pair(var_index, value, 1.0)

        n_rows = len(rhs)
        if n_rows == 0:
            A_matrix = sparse.csc_matrix((0, n), dtype=float)
            b_vec = np.zeros(0, dtype=float)
            return A_matrix, b_vec
        A_matrix = sparse.csc_matrix((data, (rows, cols)), shape=(n_rows, n), dtype=float)
        b_vec = np.asarray(rhs, dtype=float) - A_matrix @ arg
        return A_matrix, b_vec

    def has_active_constraints(self, argument) -> bool:
        if self.dimension == -1:
            for index, min_value in self.minimum:
                if abs(float(argument[index]) - min_value) < eps_constraints:
                    return True
            for index, max_value in self.maximum:
                if abs(float(argument[index]) - max_value) < eps_constraints:
                    return True
            return False
        if self.dimension == 1:
            if not math.isnan(float(self.minimum)):
                if abs(float(argument) - float(self.minimum)) < eps_constraints:
                    return True
            if not math.isnan(float(self.maximum)):
                if abs(float(argument) - float(self.maximum)) < eps_constraints:
                    return True
            return False
        for index in range(self.dimension):
            if not math.isnan(float(self.minimum[index])):
                if abs(float(argument[index]) - float(self.minimum[index])) < eps_constraints:
                    return True
            if not math.isnan(float(self.maximum[index])):
                if abs(float(argument[index]) - float(self.maximum[index])) < eps_constraints:
                    return True
        return False

    def trim_relative(self, increment) -> Any:
        if self.dimension == 1:
            if math.isnan(float(self.relative_boundary)):
                return increment
            abs_inc = abs(float(increment))
            if abs_inc > float(self.relative_boundary):
                factor = abs_inc / float(self.relative_boundary)
                return float(increment) / factor
            return increment

        if self.dimension == -1:
            factor = 1.0
            inc = np.asarray(increment, dtype=float)
            for index, bound in self.relative_boundary:
                sign = sgn(inc[index])
                if sign * inc[index] > bound:
                    current_factor = sign * inc[index] / bound
                    factor = max(factor, current_factor)
            if factor > 1:
                inc = inc / factor
            increment[:] = inc
            return increment

        factor = 1.0
        inc = np.asarray(increment, dtype=float).copy()
        for index in range(inc.size):
            if math.isnan(float(self.relative_boundary[index])):
                continue
            abs_inc = abs(inc[index])
            if abs_inc > float(self.relative_boundary[index]):
                current_factor = abs_inc / float(self.relative_boundary[index])
                factor = max(factor, current_factor)
        if factor > 1:
            inc = inc / factor
        increment[:] = inc
        return increment

    def trim_max(self, argument, increment) -> Any:
        if self.dimension == 1:
            if math.isnan(float(self.maximum)):
                return increment
            if float(argument) + float(increment) > float(self.maximum):
                return float(self.maximum) - float(argument)
            return increment

        if self.dimension == -1:
            factor = 1.0
            inc = np.asarray(increment, dtype=float)
            arg = np.asarray(argument, dtype=float)
            for index, max_value in self.maximum:
                if arg[index] + inc[index] > max_value:
                    if abs(arg[index] - max_value) < eps_constraints:
                        inc[index] = 0.0
                    else:
                        allowed_increment = max_value - arg[index]
                        factor = max(factor, abs(inc[index]) / allowed_increment)
            if factor > 1:
                inc = inc / factor
            increment[:] = inc
            return increment

        factor = 1.0
        inc = np.asarray(increment, dtype=float).copy()
        arg = np.asarray(argument, dtype=float)
        for index in range(inc.size):
            if math.isnan(float(self.maximum[index])):
                continue
            if arg[index] + inc[index] > float(self.maximum[index]):
                if abs(arg[index] - float(self.maximum[index])) < eps_constraints:
                    inc[index] = 0.0
                else:
                    allowed_increment = float(self.maximum[index]) - arg[index]
                    factor = max(factor, inc[index] / allowed_increment)
        if factor > 1:
            inc = inc / factor
        increment[:] = inc
        return increment

    def trim_min(self, argument, increment) -> Any:
        if self.dimension == 1:
            if math.isnan(float(self.minimum)):
                return increment
            if float(argument) + float(increment) < float(self.minimum):
                return float(self.minimum) - float(argument)
            return increment

        if self.dimension == -1:
            factor = 1.0
            inc = np.asarray(increment, dtype=float)
            arg = np.asarray(argument, dtype=float)
            for index, min_value in self.minimum:
                if arg[index] + inc[index] < min_value:
                    if abs(arg[index] - min_value) < eps_constraints:
                        inc[index] = 0.0
                    else:
                        allowed_decrement = arg[index] - min_value
                        factor = max(factor, abs(inc[index]) / allowed_decrement)
            if factor > 1:
                inc = inc / factor
            increment[:] = inc
            return increment

        factor = 1.0
        inc = np.asarray(increment, dtype=float).copy()
        arg = np.asarray(argument, dtype=float)
        for index in range(inc.size):
            if math.isnan(float(self.minimum[index])):
                continue
            if arg[index] + inc[index] < float(self.minimum[index]):
                if abs(arg[index] - float(self.minimum[index])) < eps_constraints:
                    inc[index] = 0.0
                else:
                    allowed_decrement = arg[index] - float(self.minimum[index])
                    factor = max(factor, abs(inc[index]) / allowed_decrement)
        if factor > 1:
            inc = inc / factor
        increment[:] = inc
        return increment

    def ensure_constraints(self, argument) -> Any:
        if self.dimension == 1:
            value = float(argument)
            if not math.isnan(float(self.maximum)):
                value = min(value, float(self.maximum))
            if not math.isnan(float(self.minimum)):
                value = max(value, float(self.minimum))
            return value
        if self.dimension == -1:
            arg = np.asarray(argument, dtype=float)
            for index, min_value in self.minimum:
                if arg[index] < min_value:
                    arg[index] = min_value
            for index, max_value in self.maximum:
                if arg[index] > max_value:
                    arg[index] = max_value
            return arg
        arg = np.asarray(argument, dtype=float)
        for index in range(arg.size):
            if not math.isnan(float(self.maximum[index])):
                arg[index] = min(arg[index], float(self.maximum[index]))
            if not math.isnan(float(self.minimum[index])):
                arg[index] = max(arg[index], float(self.minimum[index]))
        return arg


class fixed_linear_constraints:
    def __init__(self, dimension: int, count: int = 0) -> None:
        self.dimension = int(dimension)
        self.count = int(count)
        if dimension == 2 and count == 1:
            self.a = np.array([float("nan"), float("nan")], dtype=float)
            self.b = float("nan")
        else:
            self.a = None
            self.b = float("nan")

    def check_constraint_satisfaction(self, x) -> bool:
        if math.isfinite(self.b):
            return inner_prod(self.a, x) <= self.b
        return True

    def check_constraint_border(self, x) -> bool:
        if math.isfinite(self.b):
            return abs(inner_prod(self.a, x) - self.b) < eps_constraints
        return True

    @staticmethod
    def get_line_coeffs(p1, p2) -> tuple[np.ndarray, float]:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return np.array([-k, 1.0], dtype=float), b

    def trim(self, x, dx) -> Any:
        if self.count == 0 or self.a is None:
            return dx
        if not math.isfinite(self.b):
            return dx
        p1 = np.asarray(x, dtype=float)
        p2 = p1 + np.asarray(dx, dtype=float)
        if self.check_constraint_satisfaction(p2):
            return dx
        if self.check_constraint_border(p1):
            k = -self.a[0] / self.a[1]
            alpha = math.atan(k)
            p = math.sqrt(float(dx[0]) ** 2 + float(dx[1]) ** 2)
            beta = math.acos(float(dx[0]) / p)
            gamma = beta - alpha
            p_dash = p * math.cos(gamma)
            px = p_dash * math.cos(alpha)
            py = p_dash * math.sin(alpha)
            dx[0] = px
            dx[1] = py
            return dx
        a2, b2 = self.get_line_coeffs(p1, p2)
        A = np.array([self.a, a2], dtype=float)
        rhs = np.array([self.b, b2], dtype=float)
        x_star = solve_linear_system(A, rhs)
        dx[:] = x_star - p1
        return dx
