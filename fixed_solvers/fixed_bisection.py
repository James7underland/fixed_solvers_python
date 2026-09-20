"""Бисекция, метод секущих и комбинированный скалярный солвер."""

from __future__ import annotations

from dataclasses import dataclass
import math

import sys

import numpy as np

from .algebra import default_var
from .enums import (
    convergence_score_t,
    fixed_bisectional_solution_type,
    numerical_result_code_t,
)
from .exceptions import logic_error

_DBL_EPSILON = sys.float_info.epsilon
_FLT_EPSILON = float(np.finfo(np.float32).eps)


@dataclass
class fixed_bisectional_parameters_t:
    """Параметры бисекции / секущих: точность, границы и пороги комбинированного режима."""
    argument_history: bool = False
    residual_history: bool = False
    argument_precision: float = _DBL_EPSILON
    residual_precision: float = _FLT_EPSILON
    solution_type: fixed_bisectional_solution_type = fixed_bisectional_solution_type.Bisection
    secant_treshhold_iterations: int = 3
    secant_treshhold_min: float = float("nan")
    secant_treshhold_max: float = float("nan")
    argument_limit_min: float = float("nan")
    argument_limit_max: float = float("nan")
    verbose: bool = False
    use_Illinois: bool = True
    check_boundary_before: bool = True

    def secant_thresholds_satisfied(self, x_lower: float, x_upper: float) -> bool:
        """True, если ширина отрезка попадает в [secant_treshhold_min, secant_treshhold_max]."""
        delta_current = abs(x_upper - x_lower)
        max_threshold_satisfied = (
            delta_current < self.secant_treshhold_max if math.isfinite(self.secant_treshhold_max) else True
        )
        min_threshold_satisfied = (
            delta_current > self.secant_treshhold_min if math.isfinite(self.secant_treshhold_min) else True
        )
        return max_threshold_satisfied and min_threshold_satisfied


class fixed_bisection_result_t:
    """Результат скалярного корнеискателя: код, балл, аргумент, невязка."""
    def __init__(self, dimension: int = 1) -> None:
        self.dimension = int(dimension)
        self.result_code = numerical_result_code_t.NotConverged
        self.score = convergence_score_t.Poor
        self.residuals = default_var(dimension)
        self.residuals_norm = 0.0
        self.argument = default_var(dimension)
        self.iteration_count = 0
        self.max_allowed_iterations = 0
        self.reached_precision = _DBL_EPSILON


class fixed_bisection_result_analysis_t:
    """История аргумента и невязки по итерациям."""
    def __init__(self) -> None:
        self.residual_history = []
        self.argument_history = []


class _BisectionView:
    """Солвер бисекции для выбранной размерности (скаляр — 1)."""
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)

    def solve(self, solver_parameters, residuals, result, analysis=None, initial_argument=None):
        if initial_argument is None:
            return _solve(solver_parameters, float("nan"), residuals, result, analysis)
        return _solve(solver_parameters, initial_argument, residuals, result, analysis)


class _BisectionFactory:
    """Фабрика: ``fixed_bisectional[1].solve(...)`` или ``fixed_bisectional.solve(...)``."""
    def __getitem__(self, dimension: int) -> _BisectionView:
        return _BisectionView(dimension)

    def solve(self, solver_parameters, residuals, result, analysis=None, initial_argument=None):
        return _BisectionView(1).solve(solver_parameters, residuals, result, analysis, initial_argument)


fixed_bisectional = _BisectionFactory()


def _get_max_allowed_iterations(initial_delta: float, argument_precision: float) -> int:
    """Оценка числа делений отрезка пополам до заданной точности."""
    return int(math.floor(math.log(1.0 / argument_precision) / math.log(2.0)))


def _is_within_machine_epsilon(x1: float, x2: float) -> bool:
    """True, если |x1-x2| не больше машинного эпсилона относительно масштаба точек."""
    return abs(x1 - x2) <= max(abs(x2), abs(x1)) * _DBL_EPSILON


def _next_argument_secant(x1, x2, y1, y2):
    """Следующая точка методом секущих; NaN, если точка вышла за (x1, x2)."""
    denom = abs(y2) + abs(y1)
    x3 = (abs(y2) / denom * x1 + abs(y1) / denom * x2)
    if not math.isfinite(x3):
        return float("nan"), numerical_result_code_t.NumericalNanValues
    if x3 > x1 and x3 < x2:
        return x3, numerical_result_code_t.Converged
    return float("nan"), numerical_result_code_t.CustomCriteriaFailed


def _next_argument_bisection(x1, x2, y1, y2):
    """Середина отрезка."""
    x3 = 0.5 * (x2 + x1)
    if not math.isfinite(x3):
        return float("nan"), numerical_result_code_t.NumericalNanValues
    return x3, numerical_result_code_t.Converged


def _next_argument_value(solver_parameters, x1, x2, y1, y2, iterations, use_secant_ref: list):
    """Выбирает бисекцию, секущие или комбинированный шаг; Illinois включается снаружи."""
    if solver_parameters.verbose:
        print(f"next_argument_value d:{abs(x2 - x1)}", file=sys.stdout)
    use_secant_ref[0] = False
    if solver_parameters.solution_type == fixed_bisectional_solution_type.Bisection:
        return _next_argument_bisection(x1, x2, y1, y2)
    if solver_parameters.solution_type == fixed_bisectional_solution_type.Secant:
        use_secant_ref[0] = True
        return _next_argument_secant(x1, x2, y1, y2)
    if solver_parameters.solution_type == fixed_bisectional_solution_type.Combined:
        use_secant_ref[0] = (
            solver_parameters.secant_thresholds_satisfied(x1, x2)
            and int(iterations) > solver_parameters.secant_treshhold_iterations
        )
        if use_secant_ref[0]:
            x3, code = _next_argument_secant(x1, x2, y1, y2)
            if code != numerical_result_code_t.Converged:
                use_secant_ref[0] = False
                return _next_argument_bisection(x1, x2, y1, y2)
            return x3, code
        return _next_argument_bisection(x1, x2, y1, y2)
    raise RuntimeError("unsupported solution_type")


def _residual_exit_criterium(solver_parameters, r, argument, analysis, result) -> bool:
    """True, если невязка NaN или уже меньше residual_precision (успех)."""
    if solver_parameters.verbose:
        print(f"check:{r} with:{argument}", file=sys.stderr)
    if analysis is not None:
        if solver_parameters.argument_history:
            analysis.argument_history.append(argument)
        if solver_parameters.residual_history:
            analysis.residual_history.append(r)
    if not math.isfinite(r):
        result.result_code = numerical_result_code_t.NumericalNanValues
        result.score = convergence_score_t.Error
        return True
    if abs(r) < solver_parameters.residual_precision:
        result.result_code = numerical_result_code_t.Converged
        result.score = convergence_score_t.Excellent
        result.residuals = r
        return True
    return False


def _solve_limited(solver_parameters, residuals, x1, x2, result, analysis) -> None:
    """Итерации на отрезке [x1, x2] с возможным Illinois-ослаблением секущих."""
    x3 = result.argument
    y1 = residuals.residuals(x1)
    y2 = residuals.residuals(x2)
    y3 = residuals.residuals(x3)
    result.residuals = y3
    result.reached_precision = abs(x2 - x1)
    result.max_allowed_iterations = _get_max_allowed_iterations(
        abs(x2 - x1), solver_parameters.argument_precision
    )

    previous_residual_sign = 0
    for iteration in range(result.max_allowed_iterations + 1):
        result.iteration_count = iteration
        use_secant_ref = [False]
        x3, result.result_code = _next_argument_value(
            solver_parameters, x1, x2, y1, y2, iteration, use_secant_ref
        )
        result.argument = x3
        if result.result_code != numerical_result_code_t.Converged:
            if solver_parameters.verbose:
                print(f"{iteration}\t{result.max_allowed_iterations}", file=sys.stdout)
                print(
                    f"{abs(x2 - x1)}\t{2. * solver_parameters.argument_precision}\t{solver_parameters.residual_precision}",
                    file=sys.stdout,
                )
            result.score = convergence_score_t.Error
            return
        y3 = residuals.residuals(x3)
        result.residuals = y3
        if solver_parameters.verbose:
            print(
                f"{x3:.20g}\t{y3:.20g}\t{iteration} of {result.max_allowed_iterations}",
                file=sys.stdout,
            )
        if _residual_exit_criterium(solver_parameters, y3, x3, analysis, result):
            result.score = convergence_score_t.Excellent
            return
        use_secant = use_secant_ref[0]
        if y3 > 0:
            x1 = x3
            y1 = y3
            # Illinois: при двух положительных невязках подряд ослабляем противоположный конец.
            if use_secant and solver_parameters.use_Illinois and previous_residual_sign == +1:
                y2 /= 2.0
            previous_residual_sign = +1
        elif y3 < 0:
            x2 = x3
            y2 = y3
            if use_secant and solver_parameters.use_Illinois and previous_residual_sign == -1:
                y1 /= 2.0
            previous_residual_sign = -1
        result.reached_precision = abs(x2 - x1)
        if result.reached_precision <= solver_parameters.argument_precision:
            break
        if _is_within_machine_epsilon(x1, x2):
            break
    else:
        result.iteration_count = result.max_allowed_iterations

    if result.iteration_count >= result.max_allowed_iterations:
        result.result_code = numerical_result_code_t.Converged
        result.score = convergence_score_t.Satisfactory
    else:
        result.result_code = numerical_result_code_t.Converged
        if result.reached_precision < solver_parameters.argument_precision:
            result.score = convergence_score_t.Excellent
        else:
            result.score = convergence_score_t.Good


def _solve(solver_parameters, initial_argument, residuals, result, analysis) -> None:
    """Проверяет границы, сужает отрезок по знаку невязки и запускает итерации."""
    minx = solver_parameters.argument_limit_min
    maxx = solver_parameters.argument_limit_max
    if not math.isfinite(minx):
        raise logic_error("Не задана нижняя граница")
    if not math.isfinite(maxx):
        raise logic_error("Не задана верхняя граница")

    if not math.isfinite(initial_argument):
        result.argument = (maxx + minx) / 2.0
    else:
        # Проверка идёт по result.argument (NaN по умолчанию), а не по initial_argument.
        if result.argument < minx or result.argument > maxx:
            raise logic_error("Не верно задано начальное приближение")
        result.argument = initial_argument

    if solver_parameters.check_boundary_before:
        miny = residuals.residuals(minx)
        maxy = residuals.residuals(maxx)
        if _residual_exit_criterium(solver_parameters, miny, minx, analysis, result):
            return
        if _residual_exit_criterium(solver_parameters, maxy, maxx, analysis, result):
            return

    r = residuals.residuals(result.argument)
    result.residuals = r
    if _residual_exit_criterium(solver_parameters, r, result.argument, analysis, result):
        return

    # Знак невязки сужает отрезок: корень ищем там, где функция меняет знак.
    if r > 0:
        minx = result.argument
    if r < 0:
        maxx = result.argument

    result.result_code = numerical_result_code_t.NotConverged
    result.score = convergence_score_t.Excellent
    _solve_limited(solver_parameters, residuals, minx, maxx, result, analysis)
