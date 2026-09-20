"""Линейный поиск методом дробления шага."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class divider_search_parameters:
    maximum_step: float = 1.0
    minimum_step: float = 0.05
    step_divider: float = 1.5

    def step_on_search_fail(self) -> float:
        return self.minimum_step


class divider_search:
    parameters_type = divider_search_parameters

    @staticmethod
    def search(
        parameters: divider_search_parameters,
        function,
        a: float,
        b: float,
        f_a: float,
        f_b: float = float("nan"),
    ) -> tuple[float, int]:
        found_better_than_initial = False
        function_initial = f_a
        alpha_curr = b
        if not math.isfinite(f_b):
            function_current = function(b)
        else:
            function_current = f_b
        index = 1
        while alpha_curr >= parameters.minimum_step:
            alpha_prev = alpha_curr
            alpha_curr /= parameters.step_divider
            function_prev = function_current
            function_current = function(alpha_curr)
            if function_current < function_initial:
                found_better_than_initial = True
            if found_better_than_initial:
                if function_current > function_prev:
                    return alpha_prev, index
                continue
            index += 1
        return float("nan"), index
