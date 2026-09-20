"""Линейный поиск дроблением шага (Armijo-подобная эвристика без условия Вольфе).

Идём от полного шага b к нулю, деля α на step_divider: α ← α/d, d=1.5 по умолчанию.

Пока f(α) не стала меньше f(0), просто дробим. Как только нашли улучшение,
продолжаем, пока f снова не вырастет — тогда возвращаем предыдущий (ещё лучший) α.

Если до minimum_step улучшения не было — (NaN, n): Ньютон сам решит,
считать это провалом, успехом или взять fail_step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class divider_search_parameters:
    """Полный шаг, пол, делитель. fail_step = minimum_step."""

    maximum_step: float = 1.0    # стартовый α = b (обычно 1)
    minimum_step: float = 0.05   # пол: ниже не дробим, возвращаем NaN
    step_divider: float = 1.5    # α ← α / d на каждой итерации

    def step_on_search_fail(self) -> float:
        """Запасной α при PerformMinStep."""
        return self.minimum_step


class divider_search:
    """Дробление α, пока целевая убывает относительно f(a)."""

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
        """(α, сколько раз вызвали function после старта). a обычно 0, b = maximum_step."""
        found_better_than_initial = False
        function_initial = f_a
        alpha_curr = b
        # f(b) часто уже известен (конец луча), иначе это первая оценка в бюджете.
        if not math.isfinite(f_b):
            function_current = function(b)
        else:
            function_current = f_b
        index = 1
        while alpha_curr >= parameters.minimum_step:
            alpha_prev = alpha_curr
            # Дробим шаг: α ← α / d. Следующая точка ближе к старту a = 0.
            alpha_curr /= parameters.step_divider
            function_prev = function_current
            function_current = function(alpha_curr)
            if function_current < function_initial:
                found_better_than_initial = True
            if found_better_than_initial:
                # Первое улучшение уже было: рост f значит, что прошли минимум —
                # откатываемся на предыдущий (ещё меньший по α, но лучший по f) шаг.
                if function_current > function_prev:
                    return alpha_prev, index
                # Ещё убывает — продолжаем дробить, индекс не увеличиваем:
                # вызывающий интересуется числом оценок до первого улучшения+отката.
                continue
            index += 1
        # Дошли до пола без улучшения относительно f(a).
        return float("nan"), index
