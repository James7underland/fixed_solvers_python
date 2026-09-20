"""Ньютон–Рафсон для системы r(x) = 0.

Итерация без ограничений: J(xₖ)·p = −r(xₖ), затем xₖ₊₁ = xₖ + α·p
(α — линейный поиск по f = ‖r‖²).

Останов по относительной норме шага:
ρ = √(Σᵢ (Δxᵢ / max(1, |xᵢ|))²) / n  <  ε
(для dimension = -1 в знаменателе длина вектора, не константа dimension).

Если линейный поиск сорвался и включён ``step_constraint_as_optimization``
при активном box — шаг ищут как QP Гаусса–Ньютона:
minₚ ½·‖J·p + r‖²  при p в относительном box
или покоординатно: каждая компонента — lstsq по i-му столбцу J.

Обрезка шага до line search: линейные aᵀx ≤ b, затем max, min, relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .algebra import (
    as_float_array,
    default_var,
    has_not_finite,
    is_scalar,
    var_add,
    var_copy,
    var_scale,
    var_size,
    var_zeros_like,
)
from .enums import (
    convergence_score_t,
    line_search_explore_domain_violation_action_t,
    line_search_fail_action_t,
    numerical_result_code_t,
    small_step_threshold,
    step_constraint_algorithm_t,
)
from .exceptions import domain_violation, logic_error
from .fixed_constraints import fixed_linear_constraints, fixed_solver_constraints
from .fixed_linear_solver import solve_linear_system
from .line_search.divider import divider_search
from .qp.qp_wrapper import solve_quadprog_box


@dataclass
class fixed_solver_analysis_parameters_t:
    """Флаги диагностики Ньютона: истории, шаги, сетка line_search_explore."""
    argument_history: bool = False
    objective_function_history: bool = False
    steps: bool = False
    line_search_explore: bool = False
    line_search_explore_on_domain_violation: line_search_explore_domain_violation_action_t = (
        line_search_explore_domain_violation_action_t.record_nan
    )


@dataclass
class step_line_search_explore_result_t:
    """Значения ц.ф. по сетке α и индексы узлов с domain_violation."""
    values: list[float] = field(default_factory=list)
    domain_violation_indices: list[int] = field(default_factory=list)


@dataclass
class no_line_search_parameters:
    """Заглушка линейного поиска: всегда полный шаг."""
    maximum_step: float = 1.0

    def step_on_search_fail(self) -> float:
        return 1.0


class no_line_search:
    """Линейный поиск, который сразу возвращает α = 1."""
    parameters_type = no_line_search_parameters

    @staticmethod
    def search(parameters, function, a, b, f_a, f_b=float("nan")):
        """Всегда полный шаг (1.0) и одна «итерация» поиска."""
        return 1.0, 1


class fixed_solver_parameters_t:
    """Параметры Ньютона: итерации, нормы, ограничения и выбранный line search."""
    def __init__(
        self,
        dimension: int,
        linear_constraints_count: int = 0,
        line_search=divider_search,
    ) -> None:
        self.dimension = int(dimension)
        self.analysis = fixed_solver_analysis_parameters_t()
        self.constraints = fixed_solver_constraints(dimension)
        self.step_constraint_as_optimization = False
        self.step_constraint_algorithm = step_constraint_algorithm_t.CoordinateDescent
        self.linear_constraints = fixed_linear_constraints(dimension, linear_constraints_count)
        self._line_search_cls = line_search
        self.line_search = line_search.parameters_type()
        self.line_search_fail_action = line_search_fail_action_t.TreatAsFail
        self.iteration_count = 100
        self.argument_increment_norm = 1e-4
        self.step_criteria_assuming_search_step = False
        self.residuals_norm = float("nan")
        self.residuals_norm_allow_early_exit = False
        self.residuals_norm_allow_force_success = False


class fixed_solver_result_t:
    """Результат Ньютона: код, балл, аргумент, невязки и метрика шага."""
    def __init__(self, dimension: int, argument_size: int | None = None) -> None:
        self.dimension = int(dimension)
        self.argument_increment_metric = 0.0
        self.argument_increment_criteria = False
        self.result_code = numerical_result_code_t.NotConverged
        self.score = convergence_score_t.Poor
        self.residuals = default_var(dimension, size=argument_size)
        self.residuals_norm = 0.0
        self.residuals_norm_criteria = False
        self.argument = default_var(dimension, size=argument_size)
        self.iteration_count = 0


class fixed_solver_result_analysis_t:
    """История аргумента, шагов и сетки целевой функции по итерациям."""
    def __init__(self) -> None:
        self.target_function: list[list[float]] = []
        self.argument_history: list[Any] = []
        self.steps: list[float] = []
        self.line_search_explore_domain_violation_indices: list[list[int]] = []

    def get_learning_curve(self) -> list[float]:
        """Кривая обучения: одно значение ц.ф. на шаг (без explore-сетки)."""
        result: list[float] = []
        for objective_function_value in self.target_function:
            if len(objective_function_value) != 1:
                raise RuntimeError("Unexpected objective function values per step")
            result.append(objective_function_value[0])
        return result


class _NewtonView:
    """Солвер Ньютона для заданной dimension."""
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)

    def solve(self, residuals, initial_argument, solver_parameters, result, analysis=None):
        return _solve(self.dimension, residuals, initial_argument, solver_parameters, result, analysis)

    def solve_dense(self, residuals, initial_argument, solver_parameters, result, analysis=None):
        return self.solve(residuals, initial_argument, solver_parameters, result, analysis)


class _NewtonFactory:
    """Фабрика: ``fixed_newton_raphson[n].solve_dense(...)``."""
    def __getitem__(self, dimension: int) -> _NewtonView:
        return _NewtonView(dimension)

    def solve(self, residuals, initial_argument, solver_parameters, result, analysis=None):
        dimension = getattr(residuals, "dimension", solver_parameters.dimension)
        return _NewtonView(dimension).solve(
            residuals, initial_argument, solver_parameters, result, analysis
        )

    def solve_dense(self, residuals, initial_argument, solver_parameters, result, analysis=None):
        return self.solve(residuals, initial_argument, solver_parameters, result, analysis)


fixed_newton_raphson = _NewtonFactory()


def argument_increment_factor(dimension: int, argument, argument_increment) -> float:
    """Относительная норма шага ρ.
    Скаляр: ρ = |Δx| / max(1, |x|).
    Вектор: ρ = √(Σᵢ (Δxᵢ / max(1, |xᵢ|))²) / n.
    max(1, |xᵢ|) не даёт компоненте около нуля раздуть метрику.
    При dimension = -1 знаменатель — фактическая длина вектора.
    """
    if dimension == 1 and is_scalar(argument):
        arg = max(1.0, abs(float(argument)))
        inc = float(argument_increment)
        return abs(inc / arg)
    arg = as_float_array(argument)
    inc = as_float_array(argument_increment)
    squared_sum = 0.0
    for component in range(inc.size):
        scale = max(1.0, abs(float(arg[component])))
        squared_sum += (float(inc[component]) / scale) ** 2
    if dimension == -1:
        return math.sqrt(squared_sum) / arg.size
    return math.sqrt(squared_sum) / dimension


def _perform_step_research(residuals, argument, p, on_domain_violation) -> step_line_search_explore_result_t:
    """Диагностика: 101 точка α = 0, 0.01, …, 1 вдоль x + α p.

    domain_violation на узле: либо проброс, либо NaN + индекс узла в списке.
    """
    research_step_count = 100
    result = step_line_search_explore_result_t()
    for index in range(research_step_count + 1):
        alpha = 1.0 * float(index) / float(research_step_count)
        step_argument = var_add(argument, var_scale(alpha, p))
        try:
            norm = residuals(step_argument)
            result.values.append(float(norm))
        except domain_violation:
            if on_domain_violation == line_search_explore_domain_violation_action_t.rethrow:
                raise
            result.values.append(float("nan"))
            result.domain_violation_indices.append(index)
    return result


def _perform_line_search(line_search_cls, line_search_parameters, residuals, argument, r, p) -> float:
    """Линейный поиск вдоль p по целевой функции невязок."""
    def directed_function(step: float):
        return residuals(var_add(argument, var_scale(step, p)))

    a = 0.0
    b = line_search_parameters.maximum_step
    function_a = residuals.objective_function(r)
    search_step, _elapsed = line_search_cls.search(
        line_search_parameters, directed_function, a, b, function_a
    )
    return search_step


def _triplets_to_csc(triplets, n_rows: int, n_cols: int):
    """COO-тройки → CSC-матрица SciPy."""
    if not triplets:
        return sparse.csc_matrix((n_rows, n_cols), dtype=float)
    rows, cols, data = zip(*triplets)
    return sparse.csc_matrix((np.asarray(data, dtype=float), (rows, cols)), shape=(n_rows, n_cols))


def _solve_newton(dimension: int, residuals, current_residuals_value, argument):
    """Направление Ньютона: решаем J·p = −r, то есть p = −J⁻¹ r.

    При dimension = −1 якобиан приходит COO-тройками, собирается в CSC
    и факторизуется SuperLU. Иначе — плотный J и Крамер/LU.
    """
    if dimension == -1:
        J_triplets = residuals.jacobian_sparse(argument)
        n = int(np.asarray(argument).reshape(-1).size)
        J = _triplets_to_csc(J_triplets, n, n)
        try:
            solver = splu(J)
        except Exception as exc:
            raise RuntimeError("cannot calc newton step") from exc
        result = -solver.solve(as_float_array(current_residuals_value))
        return result
    J = residuals.jacobian_dense(argument)
    return -solve_linear_system(J, current_residuals_value)


def _solve_quadprog(dimension, solver_parameters, residuals, current_residuals_value, argument):
    """Шаг как box-QP Гаусса–Ньютона:
    minₚ ½·pᵀ(JᵀJ)p + (rᵀJ)p  при  minᵢ−xᵢ ≤ pᵢ ≤ maxᵢ−xᵢ.
    H = JᵀJ, линейный член f = rᵀJ. Границы — относительные (шаг, не точка).
    """
    if dimension == 1:
        raise RuntimeError("Dimension=1 should not be called with quadprog")
    if dimension == -1:
        J_triplets = residuals.jacobian_sparse(argument)
        n = int(np.asarray(argument).reshape(-1).size)
        J = _triplets_to_csc(J_triplets, n, n)
        H = J.T @ J
        r = as_float_array(current_residuals_value)
        f = r @ J
        mins, maxs = solver_parameters.constraints.get_relative_constraints(argument)
        return solve_quadprog_box(H, f, mins, maxs)
    J_triplets = residuals.jacobian_sparse(argument)
    J = _triplets_to_csc(J_triplets, dimension, dimension)
    H = J.T @ J
    r = as_float_array(current_residuals_value)
    f = r @ J
    mins, maxs = solver_parameters.constraints.get_relative_constraints(argument)
    result = solve_quadprog_box(H, f, mins, maxs)
    return np.asarray(result, dtype=float)


def _solve_coordinate_descent(dimension, residuals, current_residuals_value, argument, var_index: int) -> float:
    """Одномерный МНК по столбцу якобиана для компоненты ``var_index``."""
    if dimension == 1:
        raise RuntimeError("Coordinate descent for dimension = 1 is senseless")
    if dimension == -1:
        Jcol = residuals.jacobian_sparse_column(argument, var_index)
        n = int(np.asarray(current_residuals_value).reshape(-1).size)
        J = _triplets_to_csc(Jcol, n, 1).toarray()
        r = as_float_array(current_residuals_value)
        var_result, *_ = np.linalg.lstsq(J, -r, rcond=None)
        return float(var_result[0])
    Jcol = residuals.jacobian_column(argument, var_index)
    J = np.asarray(Jcol, dtype=float).reshape(-1, 1)
    r = as_float_array(current_residuals_value)
    var_result, *_ = np.linalg.lstsq(J, -r, rcond=None)
    return float(var_result[0])


def _trim_step(solver_parameters, argument, p):
    """Последовательная обрезка шага: линейные, max, min, relative."""
    p = solver_parameters.linear_constraints.trim(argument, p)
    p = solver_parameters.constraints.trim_max(argument, p)
    p = solver_parameters.constraints.trim_min(argument, p)
    p = solver_parameters.constraints.trim_relative(p)
    return p


def _perform_vector_step(dimension, solver_parameters, optimization_step, residuals, result, analysis) -> bool:
    """Один шаг: направление → trim → line search → x += α p.

    True = пора выходить из внешнего цикла (сходимость, NaN, fail search).
    Сначала можно остановиться по residuals_norm, ещё не считая J.
    """
    # Ранний выход: невязка уже меньше порога — якобиан и шаг не нужны.
    if solver_parameters.residuals_norm_allow_early_exit and math.isfinite(solver_parameters.residuals_norm):
        if result.residuals_norm < solver_parameters.residuals_norm:
            result.residuals_norm_criteria = True
            result.result_code = numerical_result_code_t.Converged
            return True

    try:
        if optimization_step:
            # Обычный Ньютон сорвался у границы: ищем допустимый p как box-QP.
            p = _solve_quadprog(dimension, solver_parameters, residuals, result.residuals, result.argument)
        else:
            p = _solve_newton(dimension, residuals, result.residuals, result.argument)
    except domain_violation:
        # Якобиан/невязка вне ООФ на текущей точке — это не «не сошлись», а NaN-код.
        result.result_code = numerical_result_code_t.NumericalNanValues
        return True

    if solver_parameters.step_criteria_assuming_search_step is False:
        # Критерий по полному направлению p, ещё до умножения на α линейного поиска.
        result.argument_increment_metric = argument_increment_factor(dimension, result.argument, p)
        result.argument_increment_criteria = (
            result.argument_increment_metric < solver_parameters.argument_increment_norm
        )
        custom_criteria = residuals.custom_success_criteria(result.residuals, result.argument, p)
        if result.argument_increment_criteria or custom_criteria:
            result.result_code = numerical_result_code_t.Converged
            return True

    # Обрезка: линейные полуплоскости, затем box max/min и относительный потолок |pᵢ|.
    p = _trim_step(solver_parameters, result.argument, p)

    if analysis is not None and solver_parameters.analysis.line_search_explore:
        # Диагностика: 101 узел α ∈ [0, 1], чтобы увидеть ямы/дыры ООФ вдоль луча.
        explore = _perform_step_research(
            residuals,
            result.argument,
            p,
            solver_parameters.analysis.line_search_explore_on_domain_violation,
        )
        analysis.target_function.append(explore.values)
        analysis.line_search_explore_domain_violation_indices.append(explore.domain_violation_indices)
        residuals.custom_line_research(result.argument, p)

    search_step = _perform_line_search(
        solver_parameters._line_search_cls,
        solver_parameters.line_search,
        residuals,
        result.argument,
        result.residuals,
        p,
    )
    if analysis is not None and solver_parameters.analysis.steps:
        analysis.steps.append(search_step)

    if math.isfinite(search_step):
        if search_step < small_step_threshold:
            # α < 0.1: направление ок, но линия почти не пошла — балл не выше Good.
            result.score = min(result.score, convergence_score_t.Good)
    else:
        # Поиск не нашёл α. Три политики — см. line_search_fail_action_t.
        if solver_parameters.line_search_fail_action == line_search_fail_action_t.PerformMinStep:
            result.score = min(result.score, convergence_score_t.Satisfactory)
            search_step = solver_parameters.line_search.step_on_search_fail()
        elif solver_parameters.line_search_fail_action == line_search_fail_action_t.TreatAsFail:
            result.result_code = numerical_result_code_t.LineSearchFailed
            return True
        elif solver_parameters.line_search_fail_action == line_search_fail_action_t.TreatAsSuccess:
            result.result_code = numerical_result_code_t.Converged
            return True
        else:
            raise logic_error("solver_parameters.line_search_fail_action is unknown")

    # x ← x + α p. Дальше пересчитываем невязку в новой точке.
    argument_increment = var_scale(search_step, p)
    result.argument = var_add(result.argument, argument_increment)

    if analysis is not None and solver_parameters.analysis.argument_history:
        analysis.argument_history.append(var_copy(result.argument))

    result.residuals = residuals.residuals(result.argument)
    result.residuals_norm = residuals.objective_function(result.residuals)
    if has_not_finite(result.residuals):
        result.residuals = residuals.residuals(result.argument)
        result.result_code = numerical_result_code_t.NumericalNanValues
        return True
    if solver_parameters.residuals_norm_allow_early_exit and math.isfinite(solver_parameters.residuals_norm):
        if result.residuals_norm < solver_parameters.residuals_norm:
            result.residuals_norm_criteria = True
            result.result_code = numerical_result_code_t.Converged
            return True

    custom_criteria = residuals.custom_success_criteria(result.residuals, result.argument, argument_increment)
    if custom_criteria:
        result.result_code = numerical_result_code_t.Converged
        return True

    result.argument_increment_metric = (
        argument_increment_factor(dimension, result.argument, argument_increment)
        if solver_parameters.step_criteria_assuming_search_step
        else argument_increment_factor(dimension, result.argument, p)
    )
    result.argument_increment_criteria = (
        result.argument_increment_metric < solver_parameters.argument_increment_norm
    )
    result.result_code = (
        numerical_result_code_t.Converged
        if result.argument_increment_criteria
        else numerical_result_code_t.InProgress
    )
    return result.argument_increment_criteria


def _perform_coordinate_descent_step(dimension, solver_parameters, residuals, result, analysis) -> bool:
    """Цикл по компонентам: координатный спуск с линейным поиском на каждой.

    На шаге i вектор p почти нулевой, кроме pᵢ = arg min ‖Jᵢ pᵢ + r‖²
    (одномерный lstsq). После trim и line search обновляем x и переходим к i+1.
    Метрика шага — максимум |pᵢ| (или α|pᵢ|, если учитываем длину поиска).
    """
    has_succeeded_search_step = False
    p = var_zeros_like(result.argument)
    result.argument_increment_metric = 0.0
    substep_count = var_size(result.argument)

    for substep in range(substep_count):
        if substep != 0:
            # Предыдущая компонента уже сделала свой шаг; в p её обнуляем,
            # чтобы текущий lstsq не тащил старое направление.
            p = var_copy(p)
            p[substep - 1] = 0.0
        try:
            var_p = _solve_coordinate_descent(
                dimension, residuals, result.residuals, result.argument, substep
            )
        except domain_violation:
            result.result_code = numerical_result_code_t.NumericalNanValues
            return True
        p = var_copy(p)
        p[substep] = var_p
        p = _trim_step(solver_parameters, result.argument, p)
        var_p = float(p[substep])
        result.argument_increment_metric = max(abs(var_p), result.argument_increment_metric)

        if analysis is not None and solver_parameters.analysis.line_search_explore:
            explore = _perform_step_research(
                residuals,
                result.argument,
                p,
                solver_parameters.analysis.line_search_explore_on_domain_violation,
            )
            analysis.target_function.append(explore.values)
            analysis.line_search_explore_domain_violation_indices.append(explore.domain_violation_indices)
            residuals.custom_line_research(result.argument, p)

        search_step = _perform_line_search(
            solver_parameters._line_search_cls,
            solver_parameters.line_search,
            residuals,
            result.argument,
            result.residuals,
            p,
        )
        if analysis is not None and solver_parameters.analysis.steps:
            analysis.steps.append(search_step)

        if math.isfinite(search_step):
            if search_step < small_step_threshold:
                result.score = min(result.score, convergence_score_t.Good)
            has_succeeded_search_step = True
        else:
            if solver_parameters.line_search_fail_action == line_search_fail_action_t.PerformMinStep:
                result.score = min(result.score, convergence_score_t.Satisfactory)
                search_step = solver_parameters.line_search.step_on_search_fail()
                has_succeeded_search_step = True
            else:
                continue

        argument_increment = var_scale(search_step, p)
        result.argument = var_add(result.argument, argument_increment)

        if analysis is not None and solver_parameters.analysis.argument_history:
            analysis.argument_history.append(var_copy(result.argument))

        result.residuals = residuals.residuals(result.argument)
        if has_not_finite(result.residuals):
            result.residuals = residuals.residuals(result.argument)
            result.result_code = numerical_result_code_t.NumericalNanValues
            return True

        custom_criteria = residuals.custom_success_criteria(
            result.residuals, result.argument, argument_increment
        )
        if custom_criteria:
            result.result_code = numerical_result_code_t.Converged
            return True

        var_increment_metric = (
            search_step * abs(var_p)
            if solver_parameters.step_criteria_assuming_search_step
            else abs(var_p)
        )
        result.argument_increment_metric = max(var_increment_metric, result.argument_increment_metric)

    result.residuals_norm = residuals.objective_function(result.residuals)
    result.argument_increment_criteria = (
        result.argument_increment_metric < solver_parameters.argument_increment_norm
    )
    if result.argument_increment_criteria:
        result.result_code = numerical_result_code_t.Converged
        return True

    if has_succeeded_search_step is False:
        if solver_parameters.line_search_fail_action == line_search_fail_action_t.TreatAsFail:
            result.result_code = numerical_result_code_t.LineSearchFailed
            return True
        if solver_parameters.line_search_fail_action == line_search_fail_action_t.TreatAsSuccess:
            result.result_code = numerical_result_code_t.Converged
            return True
        raise RuntimeError("Wrong branch")
    result.result_code = numerical_result_code_t.InProgress
    return False


def _solve(dimension, residuals, initial_argument, solver_parameters, result, analysis):
    """Основной цикл Ньютона: невязка в x₀, затем шаги и оценка балла."""
    result.argument = var_copy(initial_argument)
    if analysis is not None and solver_parameters.analysis.argument_history:
        analysis.argument_history.append(var_copy(result.argument))

    try:
        result.residuals = residuals.residuals(result.argument)
    except domain_violation:
        result.result_code = numerical_result_code_t.NumericalNanValues
        result.score = convergence_score_t.Error
        return

    result.residuals_norm = residuals.objective_function(result.residuals)
    if has_not_finite(result.residuals):
        result.residuals = residuals.residuals(result.argument)
        result.result_code = numerical_result_code_t.NumericalNanValues
        result.score = convergence_score_t.Error
        return

    result.result_code = numerical_result_code_t.NotConverged
    result.score = convergence_score_t.Excellent

    for iteration in range(solver_parameters.iteration_count):
        result.iteration_count = iteration
        stop_iterations = _perform_vector_step(
            dimension, solver_parameters, False, residuals, result, analysis
        )
        # Ньютон не нашёл α (LineSearchFailed) и box активен — пробуем
        # допустимый шаг: QP (Quadprog) или покоординатный lstsq.
        optimization_step = (
            solver_parameters.step_constraint_as_optimization
            and solver_parameters.constraints.has_active_constraints(result.argument)
            and result.result_code == numerical_result_code_t.LineSearchFailed
        )
        if optimization_step:
            if dimension == 1:
                stop_iterations = _perform_vector_step(
                    dimension, solver_parameters, True, residuals, result, analysis
                )
            elif solver_parameters.step_constraint_algorithm == step_constraint_algorithm_t.CoordinateDescent:
                stop_iterations = _perform_coordinate_descent_step(
                    dimension, solver_parameters, residuals, result, analysis
                )
            else:
                stop_iterations = _perform_vector_step(
                    dimension, solver_parameters, True, residuals, result, analysis
                )
        if stop_iterations:
            break
    else:
        result.iteration_count = solver_parameters.iteration_count

    iteration = result.iteration_count
    # Чем больше итераций относительно лимита, тем хуже балл (даже при Converged):
    #   > 30% бюджета → не выше Satisfactory
    #   > 15% бюджета → не выше Good
    if iteration > 0.3 * solver_parameters.iteration_count:
        result.score = min(result.score, convergence_score_t.Satisfactory)
    elif iteration > 0.15 * solver_parameters.iteration_count:
        result.score = min(result.score, convergence_score_t.Good)

    if result.result_code != numerical_result_code_t.Converged:
        result.score = convergence_score_t.Poor

    if math.isfinite(solver_parameters.residuals_norm):
        # Жёсткий порог по невязке после цикла: если не уложились — NotConverged,
        # если уложились и allow_force_success — поднимаем код до Converged
        # (балл не лучше Satisfactory, если он был хуже).
        if result.residuals_norm > solver_parameters.residuals_norm:
            result.result_code = numerical_result_code_t.NotConverged
            result.score = convergence_score_t.Poor
        elif solver_parameters.residuals_norm_allow_force_success:
            result.result_code = numerical_result_code_t.Converged
            score = max(int(result.score), int(convergence_score_t.Satisfactory))
            result.score = convergence_score_t(score)
