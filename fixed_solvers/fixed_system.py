"""Системы алгебраических уравнений r(x) = 0.

Пользователь задаёт ``residuals(x)``. Якобиан по умолчанию — центральные разности
по каждой компоненте xᵢ: Jᵢ = (r(x + e eᵢ) − r(x − e eᵢ)) / (2e),
e = ε · max(1, |xᵢ|).

Целевая для линейного поиска — квадрат невязки: f(x) = ‖r(x)‖² (для скаляра просто r²).

dimension: 1 — float; n>1 — вектор длины n; −1 — переменная длина (sparse COO).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .algebra import (
    as_float_array,
    default_var,
    is_scalar,
    numeric_derivative_delta,
    squared_norm,
    two_sided_derivative,
    var_copy,
)


class fixed_system_types:
    """Фабрика аргумента той же формы, что у системы."""

    @staticmethod
    def default_var(dimension: int, value: float = float("nan"), size: int | None = None):
        """NaN-вектор / NaN-скаляр под данную dimension (и size при −1)."""
        return default_var(dimension, value, size)


class fixed_system_t(ABC):
    """Базовый класс: невязки + якобиан + крючки диагностики."""

    dimension: int = -1

    def __init__(self, dimension: int | None = None, epsilon: float = 1e-6) -> None:
        if dimension is not None:
            self.dimension = int(dimension)
        # Относительный шаг численной производной; 1e-6 — типичный баланс
        # между ошибкой аппроксимации O(e²) и шумом float64.
        self.epsilon = float(epsilon)

    def objective_function(self, r) -> float:
        """f = ‖r‖². Линейный поиск минимизирует именно это, не сырой r."""
        if self.dimension == 1 and is_scalar(r):
            value = float(r)
            return value * value
        return squared_norm(r)

    def __call__(self, x) -> float:
        """f(x) = ‖r(x)‖² — удобно передавать систему как function(alpha) вдоль луча."""
        r = self.residuals(x)
        return self.objective_function(r)

    @abstractmethod
    def residuals(self, x):
        """r(x). Вне ООФ принято бросать domain_violation, а не возвращать NaN."""
        raise NotImplementedError

    def jacobian_dense(self, x):
        """Плотный J. Переопределите, если есть аналитическая производная."""
        return self.jacobian_dense_numeric(x)

    def jacobian_sparse(self, x) -> list[tuple[int, int, float]]:
        """Разреженный J тройками (row, col, value). Для dimension = −1."""
        return self.jacobian_sparse_numeric(x)

    def jacobian_sparse_column(self, reduced, desired_col_index: int) -> list[tuple[int, int, float]]:
        """Столбец Jᵢ в COO, но col переписан в 0 (одноколоночная матрица для lstsq)."""
        J = self.jacobian_sparse(reduced)
        result = []
        for row, col, value in J:
            if int(col) == int(desired_col_index):
                result.append((int(row), 0, float(value)))
        return result

    def jacobian_column(self, x, desired_col_index: int):
        """Численный столбец плотного J — для координатного спуска по одной переменной."""
        if self.dimension == 1:
            raise RuntimeError("Jacobian column for dimension = 1 is sensless")
        arg = var_copy(x)
        e = numeric_derivative_delta(float(arg[desired_col_index]), self.epsilon)
        arg_plus = var_copy(arg)
        arg_plus[desired_col_index] = float(x[desired_col_index]) + e
        f_plus = as_float_array(self.residuals(arg_plus))
        arg_minus = var_copy(arg)
        arg_minus[desired_col_index] = float(x[desired_col_index]) - e
        f_minus = as_float_array(self.residuals(arg_minus))
        return (f_plus - f_minus) / (2.0 * e)

    def custom_success_criteria(self, r, x, p) -> bool:
        """Доп. останов Ньютона (например, «невязка мала в прикладных единицах»). По умолчанию выкл."""
        return False

    def custom_line_research(self, argument, argument_increment) -> None:
        """Крючок после сетки line_search_explore: можно логировать траекторию шага."""
        return None

    def custom_line_search_start(self) -> None:
        """Крючок перед линейным поиском (резерв API)."""
        return None

    def custom_line_search_sample(self, alpha: float, x_alpha) -> None:
        """Крючок на пробной точке α (резерв API)."""
        return None

    def jacobian_dense_numeric(self, x):
        """Центральные разности по каждой компоненте; для dimension=1 — скаляр f′(x)."""
        if self.dimension == 1:
            return float(two_sided_derivative(self.residuals, float(x), self.epsilon))

        arg = var_copy(x)
        n = int(np.asarray(x).reshape(-1).size)
        J = None
        for component in range(n):
            # Портим одну компоненту, считаем r±, возвращаем xᵢ как было —
            # так не копируем весь вектор на каждый столбец.
            e = numeric_derivative_delta(float(arg[component]), self.epsilon)
            arg[component] = float(x[component]) + e
            f_plus = as_float_array(self.residuals(arg))
            arg[component] = float(x[component]) - e
            f_minus = as_float_array(self.residuals(arg))
            arg[component] = float(x[component])
            Jcol = (f_plus - f_minus) / (2.0 * e)
            if J is None:
                J = np.zeros((Jcol.size, n), dtype=float)
            J[:, component] = Jcol
        return J

    def jacobian_sparse_numeric(self, x) -> list[tuple[int, int, float]]:
        """Тот же численный J, но COO. Нули столбца тоже пишутся (плотная колонка в sparse-обёртке)."""
        if self.dimension == 1:
            raise RuntimeError("Must not be called")
        arg = var_copy(x)
        n = int(np.asarray(x).reshape(-1).size)
        result: list[tuple[int, int, float]] = []
        dim = n if self.dimension == -1 else self.dimension
        for component in range(dim):
            e = numeric_derivative_delta(float(arg[component]), self.epsilon)
            arg[component] = float(x[component]) + e
            f_plus = as_float_array(self.residuals(arg))
            arg[component] = float(x[component]) - e
            f_minus = as_float_array(self.residuals(arg))
            arg[component] = float(x[component])
            Jcol = (f_plus - f_minus) / (2.0 * e)
            for row in range(Jcol.size):
                result.append((row, component, float(Jcol[row])))
        return result


class fixed_scalar_wrapper_t(fixed_system_t):
    """Превращает f: ℝ → ℝ в систему dimension=1 (бисекция / скалярный Ньютон)."""

    dimension = 1

    def __init__(self, function: Callable[[float], float], epsilon: float = float("nan")) -> None:
        super().__init__(dimension=1)
        self.function = function
        if not np.isnan(epsilon):
            self.epsilon = float(epsilon)

    def residuals(self, x):
        """Просто f(x); якобиан — f′(x) через two_sided_derivative."""
        return self.function(float(x))
