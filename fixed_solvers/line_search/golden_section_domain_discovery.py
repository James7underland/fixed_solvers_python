"""Золотое сечение при неизвестной области определения (ООФ).

Инвариант: левая граница a всегда в ООФ. Точки α, β, b могут выбросить
domain_violation. NaN в значении — ошибка расчёта, не «вне ООФ».

Режимы:
  forbid_exit              — любой выход за ООФ → RuntimeError
  require_connected_domain — ООФ связна: повторный «провал» после дырки → logic_error
  allow_disconnected_domain — эвристика: отбрасываем кусок, где минимума точно нет

Сужение отрезка (унимодальность). Точки a < α < β < b:

  все в ООФ, f(α) < f(β)  →  новый [a, β]
  все в ООФ, f(α) ≥ f(β)  →  новый [α, b]
  α и β вне ООФ           →  эвристика: [a, α]  (минимум не правее первой дырки)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..enums import domain_discovery_mode_t
from ..exceptions import domain_violation, logic_error
from .golden_section import golden_section_parameters, golden_section_search


@dataclass
class golden_section_domain_discovery_parameters(golden_section_parameters):
    """Параметры ЗС с режимом ООФ и флагом допуска неунимодальности."""
    mode: domain_discovery_mode_t = domain_discovery_mode_t.require_connected_domain
    allow_non_unimodal: bool = False


@dataclass
class _evaluation_t:
    """Результат одной оценки: в ООФ / NaN / значение."""
    in_domain: bool = False
    has_nan: bool = False
    value: float = float("nan")


class golden_section_search_domain_discovery:
    """Золотое сечение, которое сужает отрезок при ``domain_violation``."""
    parameters_type = golden_section_domain_discovery_parameters

    @staticmethod
    def search(
        parameters: golden_section_domain_discovery_parameters,
        function,
        a: float,
        b: float,
        f_a: float,
        f_b: float = float("nan"),
    ) -> tuple[float, int]:
        """Ищет шаг при возможном выходе за ООФ; NaN-значение функции — ошибка расчёта."""
        def check_convergence(f_min: float, f_0: float) -> bool:
            # Те же два критерия, что у обычного ЗС: сильное падение или порог шума.
            return parameters.decrement_factor_criteria(f_min, f_0) or parameters.target_value_criteria(f_min)

        seen_domain_gap = False

        def evaluate(x: float) -> _evaluation_t:
            """Вызов function(x): domain_violation сужает ООФ, NaN — сбой расчёта."""
            nonlocal seen_domain_gap
            result = _evaluation_t()
            try:
                result.value = function(x)
                result.in_domain = True
                result.has_nan = not math.isfinite(result.value)
            except domain_violation:
                # Это маркер ООФ, не Exception: широкий except его не съест.
                if parameters.mode == domain_discovery_mode_t.forbid_exit:
                    raise RuntimeError("domain violation detected in forbid_exit mode")
                result.in_domain = False
                if parameters.mode == domain_discovery_mode_t.require_connected_domain:
                    # Связная ООФ: после первого провала (x > a) второй провал запрещён,
                    # если уже «выходили и вернулись» — seen_domain_gap.
                    if x > a:
                        if seen_domain_gap:
                            raise logic_error("disconnected domain detected")
                        seen_domain_gap = True
            return result

        def fail_result():
            return float("nan"), parameters.iteration_count + 1

        if not math.isfinite(f_a):
            return fail_result()

        if math.isfinite(f_b):
            b_eval = _evaluation_t(in_domain=True, has_nan=False, value=f_b)
        else:
            b_eval = evaluate(b)
        if b_eval.has_nan:
            return fail_result()

        f_0 = f_a
        if b_eval.in_domain:
            boundary_alpha = parameters.try_resolve_step_by_target_value(f_a, b_eval.value, a, b)
            if boundary_alpha is not None:
                return boundary_alpha, 0

        best_x = a
        best_f = f_a
        if b_eval.in_domain and b_eval.value < best_f:
            best_x = b
            best_f = b_eval.value

        if check_convergence(best_f, f_0):
            return best_x, 0

        def update_minimum(alpha, alpha_eval, beta, beta_eval):
            nonlocal best_x, best_f
            if alpha_eval.in_domain and alpha_eval.value < best_f:
                best_x = alpha
                best_f = alpha_eval.value
            if beta_eval.in_domain and beta_eval.value < best_f:
                best_x = beta
                best_f = beta_eval.value

        for index in range(parameters.iteration_count):
            alpha = golden_section_search.get_alpha(a, b)
            beta = golden_section_search.get_beta(a, b)
            alpha_eval = evaluate(alpha)
            beta_eval = evaluate(beta)
            if alpha_eval.has_nan or beta_eval.has_nan:
                return fail_result()
            update_minimum(alpha, alpha_eval, beta, beta_eval)

            def check_local_max(allow_non_unimodal: bool) -> None:
                # Локальный максимум внутри [a, b] ломает унимодальность ЗС:
                #   f(α) > f(a) и f(α) > f(β)  →  горб в α
                #   f(β) > f(α) и f(β) > f(b)  →  горб в β
                if allow_non_unimodal:
                    return
                if not alpha_eval.in_domain or not beta_eval.in_domain:
                    return
                if alpha_eval.value > f_a and alpha_eval.value > beta_eval.value:
                    raise logic_error("non-unimodal function detected, f_alpha is max")
                if b_eval.in_domain and beta_eval.value > alpha_eval.value and beta_eval.value > b_eval.value:
                    raise logic_error("non-unimodal function detected, f_beta is max")

            check_local_max(parameters.allow_non_unimodal)

            defined_b = b_eval.in_domain
            defined_alpha = alpha_eval.in_domain
            defined_beta = beta_eval.in_domain
            allow_heuristic = parameters.mode == domain_discovery_mode_t.allow_disconnected_domain

            if not defined_alpha and not defined_beta:
                # Эвристика ООФ: обе внутренние точки вне области — отбрасываем [α, b].
                if not allow_heuristic:
                    return fail_result()
                b = alpha
                b_eval = alpha_eval
            elif not defined_alpha:
                # α вне ООФ, β в ООФ: смотрим, куда сдвигать по f(β) vs f(a), f(b).
                if defined_b and (beta_eval.value > b_eval.value):
                    a = beta
                    f_a = beta_eval.value
                elif beta_eval.value < f_a:
                    b = beta
                    b_eval = beta_eval
                else:
                    raise logic_error("minimum is not in [a, b]")
            elif not defined_beta:
                # β вне ООФ. Если b тоже вне — нужна эвристика (отрезать справа).
                if not defined_b:
                    if not allow_heuristic:
                        return fail_result()
                    if alpha_eval.value < f_a:
                        b = beta
                        b_eval = beta_eval
                    else:
                        b = alpha
                        b_eval = alpha_eval
                else:
                    if alpha_eval.value < f_a:
                        b = alpha
                        b_eval = alpha_eval
                    elif alpha_eval.value > b_eval.value:
                        a = alpha
                        f_a = alpha_eval.value
                    else:
                        raise logic_error("minimum is not in [a, b]")
            else:
                # Классическое ЗС: все внутренние точки в ООФ.
                if alpha_eval.value < beta_eval.value:
                    b = beta
                    b_eval = beta_eval
                else:
                    a = alpha
                    f_a = alpha_eval.value

            if check_convergence(best_f, f_0):
                return best_x, index + 1

        if best_f < f_0:
            return best_x, parameters.iteration_count + 1
        return fail_result()
