"""Оптимизация методом Гаусса–Ньютона (сумма квадратов)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .algebra import as_float_array, var_add, var_copy, var_scale
from .enums import numerical_result_code_t
from .fixed_nonlinear_solver import (
    fixed_solver_parameters_t,
    fixed_solver_result_analysis_t,
    fixed_solver_result_t,
)
from .line_search.divider import divider_search
from .line_search.golden_section import golden_section_search


class fixed_optimizer_parameters_t(fixed_solver_parameters_t):
    """Параметры Гаусса–Ньютона (переменная размерность)."""
    def __init__(self, line_search=divider_search) -> None:
        super().__init__(-1, 0, line_search)


class fixed_optimizer_result_t(fixed_solver_result_t):
    """Результат оптимизации: те же поля, что у Ньютона, dimension=-1."""
    def __init__(self, argument_size: int | None = None) -> None:
        super().__init__(-1, argument_size)


class fixed_optimizer_result_analysis_t(fixed_solver_result_analysis_t):
    """История оптимизации (кривая ц.ф. и аргумент)."""
    pass


class fixed_least_squares_function_t(ABC):
    """Целевая функция МНК: сумма квадратов невязок и численный якобиан."""
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = float(epsilon)

    def objective_function(self, r) -> float:
        """||r||²."""
        arr = as_float_array(r)
        return float(np.dot(arr, arr))

    def __call__(self, x) -> float:
        r = self.residuals(x)
        return self.objective_function(r)

    @abstractmethod
    def residuals(self, x):
        """Вектор невязок наименьших квадратов."""
        raise NotImplementedError

    def jacobian_dense(self, x):
        """Плотный якобиан; по умолчанию численный."""
        return self.jacobian_dense_numeric(x)

    def jacobian_dense_numeric(self, x):
        """Центральные разности по каждой компоненте аргумента."""
        arg = var_copy(x)
        J = None
        n = int(np.asarray(x).reshape(-1).size)
        for arg_index in range(n):
            e = self.epsilon * max(1.0, abs(float(arg[arg_index])))
            arg[arg_index] = float(x[arg_index]) + e
            f_plus = as_float_array(self.residuals(arg))
            arg[arg_index] = float(x[arg_index]) - e
            f_minus = as_float_array(self.residuals(arg))
            arg[arg_index] = float(x[arg_index])
            Jcol = (f_plus - f_minus) / (2.0 * e)
            if J is None:
                J = np.zeros((Jcol.size, n), dtype=float)
            J[:, arg_index] = Jcol
        return J

    def custom_success_criteria(self, r, x, p) -> bool:
        """Для оптимизатора критерий по умолчанию всегда истинен (остановка — по шагу)."""
        return True


class rosenbrock_function_t(fixed_least_squares_function_t):
    """Классическая функция Розенброка в виде двух невязок."""
    def residuals(self, x):
        """Невязки Розенброка: 10(x1-x0²) и 1-x0."""
        x = as_float_array(x)
        result = np.empty(2, dtype=float)
        result[0] = 10.0 * (x[1] - x[0] * x[0])
        result[1] = 1.0 - x[0]
        return result


class fixed_optimize_gauss_newton:
    """Итерации Гаусса–Ньютона: lstsq(J, -r) и линейный поиск вдоль направления."""
    @staticmethod
    def _argument_increment_factor(argument, argument_increment) -> float:
        """||Δx|| / n — относительная длина шага оптимизатора."""
        inc = as_float_array(argument_increment)
        return float(np.linalg.norm(inc) / argument.size)

    @staticmethod
    def _perform_line_search(line_search_cls, line_search_parameters, function, argument, r, p) -> float:
        """Линейный поиск вдоль направления Гаусса–Ньютона."""
        def directed_function(step: float):
            return function(var_add(argument, var_scale(step, p)))

        a = 0.0
        b = line_search_parameters.maximum_step
        function_a = directed_function(a)
        search_step, _elapsed = line_search_cls.search(
            line_search_parameters, directed_function, a, b, function_a
        )
        return search_step

    @staticmethod
    def optimize(function, initial_argument, solver_parameters, result, analysis=None):
        """Минимизирует сумму квадратов; пишет код и аргумент в ``result``."""
        if analysis is not None and solver_parameters.analysis.line_search_explore:
            raise RuntimeError("Line search exploring not implemented")

        initial = as_float_array(initial_argument)
        n = int(initial.size)
        if n == 0:
            result.result_code = numerical_result_code_t.Converged
            return

        result.argument = var_copy(initial)
        if analysis is not None and solver_parameters.analysis.argument_history:
            analysis.argument_history.append(var_copy(result.argument))

        result.residuals = function.residuals(result.argument)
        result.residuals_norm = function.objective_function(result.residuals)
        if analysis is not None and solver_parameters.analysis.objective_function_history:
            analysis.target_function.append([result.residuals_norm])

        line_search_cls = getattr(solver_parameters, "_line_search_cls", divider_search)

        for _iteration in range(solver_parameters.iteration_count):
            J = function.jacobian_dense(result.argument)
            search_direction, *_ = np.linalg.lstsq(J, -as_float_array(result.residuals), rcond=None)

            search_step = fixed_optimize_gauss_newton._perform_line_search(
                line_search_cls,
                solver_parameters.line_search,
                function,
                result.argument,
                result.residuals,
                search_direction,
            )
            if analysis is not None and solver_parameters.analysis.steps:
                analysis.steps.append(search_step)

            if not np.isfinite(search_step):
                result.result_code = numerical_result_code_t.LineSearchFailed
                break

            argument_increment = var_scale(search_step, search_direction)
            result.argument = var_add(result.argument, argument_increment)

            if analysis is not None and solver_parameters.analysis.argument_history:
                analysis.argument_history.append(var_copy(result.argument))

            result.residuals = function.residuals(result.argument)
            result.residuals_norm = function.objective_function(result.residuals)
            if analysis is not None and solver_parameters.analysis.objective_function_history:
                analysis.target_function.append([result.residuals_norm])

            argument_increment_metric = (
                fixed_optimize_gauss_newton._argument_increment_factor(result.argument, argument_increment)
                if solver_parameters.step_criteria_assuming_search_step
                else fixed_optimize_gauss_newton._argument_increment_factor(result.argument, search_direction)
            )
            argument_increment_criteria = argument_increment_metric < solver_parameters.argument_increment_norm
            custom_criteria = False
            if custom_criteria or argument_increment_criteria:
                result.result_code = numerical_result_code_t.Converged
                break


def optimize_gauss_newton(function, initial) -> np.ndarray:
    """Удобная обёртка: золотое сечение и исключение, если нет сходимости."""
    parameters = fixed_solver_parameters_t(-1, 0, golden_section_search)
    result = fixed_solver_result_t(-1, argument_size=len(initial))
    fixed_optimize_gauss_newton.optimize(function, initial, parameters, result)
    if result.result_code != numerical_result_code_t.Converged:
        raise RuntimeError("Gauss-Newton optimizer not converged")
    return result.argument
