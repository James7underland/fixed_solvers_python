"""Золотое сечение и исключение выхода за ООФ."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..exceptions import domain_violation

__all__ = ["domain_violation", "golden_section_parameters", "golden_section_search"]


@dataclass
class golden_section_parameters:
    maximum_step: float = 1.0
    iteration_count: int = 10
    function_decrement_factor: float = 2.0
    function_target_value: float = float("nan")
    fail_step_size: float = 0.05

    def step_on_search_fail(self) -> float:
        return self.fail_step_size

    def get_final_section_length(self) -> float:
        return 0.618 ** self.iteration_count

    def decrement_factor_criteria(self, f_current_min: float, f_0: float) -> bool:
        if math.isfinite(self.function_decrement_factor):
            decrement = f_0 / f_current_min
            return decrement > self.function_decrement_factor
        return False

    def target_value_criteria(self, f_current_min: float) -> bool:
        return math.isfinite(self.function_target_value) and (f_current_min < self.function_target_value)

    def try_resolve_step_by_target_value(self, f_a: float, f_b: float, a: float, b: float) -> float | None:
        if not math.isfinite(self.function_target_value):
            return None
        a_below = f_a < self.function_target_value
        b_below = f_b < self.function_target_value
        if not a_below and not b_below:
            return None
        if a_below and b_below:
            return b
        if a_below:
            return a
        return b


class golden_section_search:
    parameters_type = golden_section_parameters

    @staticmethod
    def get_alpha(a: float, b: float) -> float:
        return a + 2.0 / (3.0 + math.sqrt(5.0)) * (b - a)

    @staticmethod
    def get_beta(a: float, b: float) -> float:
        return a + 2.0 / (1.0 + math.sqrt(5.0)) * (b - a)

    @staticmethod
    def search(
        parameters: golden_section_parameters,
        function,
        a: float,
        b: float,
        f_a: float,
        f_b: float = float("nan"),
    ) -> tuple[float, int]:
        def check_convergence(f_min: float, f_0: float) -> bool:
            return parameters.decrement_factor_criteria(f_min, f_0) or parameters.target_value_criteria(f_min)

        f_0 = f_a
        if not math.isfinite(f_b):
            f_b = function(b)

        boundary_alpha = parameters.try_resolve_step_by_target_value(f_a, f_b, a, b)
        if boundary_alpha is not None:
            return boundary_alpha, 0

        if f_b < f_a:
            x_min, f_min = b, f_b
        else:
            x_min, f_min = a, f_a
        if check_convergence(f_min, f_0):
            return x_min, 0

        alpha = golden_section_search.get_alpha(a, b)
        f_alpha = function(alpha)
        beta = golden_section_search.get_beta(a, b)
        f_beta = function(beta)

        def update_minimum():
            nonlocal x_min, f_min
            if f_alpha < f_beta:
                if f_alpha < f_a:
                    x_min, f_min = alpha, f_alpha
                else:
                    x_min, f_min = a, f_a
            else:
                if f_b < f_beta:
                    x_min, f_min = b, f_b
                else:
                    x_min, f_min = beta, f_beta

        update_minimum()
        if check_convergence(f_min, f_0):
            return x_min, 1

        for index in range(1, parameters.iteration_count):
            if f_alpha < f_beta:
                b = beta
                f_b = f_beta
                beta = alpha
                f_beta = f_alpha
                alpha = golden_section_search.get_alpha(a, b)
                f_alpha = function(alpha)
            else:
                a = alpha
                f_a = f_alpha
                alpha = beta
                f_alpha = f_beta
                beta = golden_section_search.get_beta(a, b)
                f_beta = function(beta)
            update_minimum()
            if check_convergence(f_min, f_0):
                return x_min, index + 1

        if f_min < f_0:
            return x_min, parameters.iteration_count + 1
        return float("nan"), parameters.iteration_count + 1
