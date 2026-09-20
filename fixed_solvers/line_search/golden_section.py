"""Золотое сечение: поиск шага α ∈ [a, b] вдоль направления Ньютона.

Ищем минимум унимодальной f на отрезке. Золотое отношение:
φ = (1+√5)/2 ≈ 1.618.

Две пробные точки делят [a, b] так, что одна из них переиспользуется
после сужения (считается ровно одно новое значение f за итерацию):
α = a + 2/(3+√5)·(b-a)     ближе к a
β = a + 2/(1+√5)·(b-a)     ближе к b

Сужение (унимодальность: ровно один минимум на отрезке):

            f(α) < f(β)   →   новый отрезок [a, β],  старый α становится β
            f(α) ≥ f(β)   →   новый отрезок [α, b],  старый β становится α

Досрочный выход из цикла:
  •  f(0) / fₘᵢₙ  >  function_decrement_factor   (шаг уже «достаточно хорош»);
  •  fₘᵢₙ  <  function_target_value              (ниже порога шума).

Порог шума до цикла:
  оба конца ниже порога  →  вернуть b  (Ньютон: «на полу, полный шаг-маркер»);
  только a ниже          →  вернуть a  (остаёмся у старта);
  только b ниже          →  вернуть b  (полный шаг ещё осмыслен).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..exceptions import domain_violation

__all__ = ["domain_violation", "golden_section_parameters", "golden_section_search"]


@dataclass
class golden_section_parameters:
    """Параметры ЗС: длина луча, число сжатий, критерии останова, запасной шаг."""

    maximum_step: float = 1.0          # правый конец поиска, обычно полный шаг Ньютона
    iteration_count: int = 10          # сколько раз сжать отрезок (длина × ≈0.618ᵏ)
    function_decrement_factor: float = 2.0  # досрочный выход, если f(0)/fₘᵢₙ больше этого
    function_target_value: float = float("nan")  # порог шума; NaN = выключен
    fail_step_size: float = 0.05       # запасной α при политике PerformMinStep

    def step_on_search_fail(self) -> float:
        """α при политике PerformMinStep, если поиск вернул NaN."""
        return self.fail_step_size

    def get_final_section_length(self) -> float:
        """После k сжатий длина ≈ 0.618ᵏ · (b − a); коэффициент 0.618 = 1/φ."""
        return 0.618 ** self.iteration_count

    def decrement_factor_criteria(self, f_current_min: float, f_0: float) -> bool:
        """True, если f(0) / fₘᵢₙ > factor — шаг уже «достаточно хорош»."""
        if math.isfinite(self.function_decrement_factor):
            decrement = f_0 / f_current_min
            return decrement > self.function_decrement_factor
        return False

    def target_value_criteria(self, f_current_min: float) -> bool:
        """True, если минимум ниже порога шума (дальше итерировать бессмысленно)."""
        return math.isfinite(self.function_target_value) and (f_current_min < self.function_target_value)

    def try_resolve_step_by_target_value(self, f_a: float, f_b: float, a: float, b: float) -> float | None:
        """До ЗС: границы уже в шуме.

        Порог не задан / оба конца выше порога → None (гоняем ЗС).
        Только a ниже → вернуть a (остаёмся у старта).
        Только b ниже → вернуть b (полный шаг ещё осмыслен).
        Оба ниже → вернуть b: предельная точность достигнута, цикл ЗС не запускаем.
        """
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
    """Локализация минимума унимодальной f на [a, b]."""

    parameters_type = golden_section_parameters

    @staticmethod
    def get_alpha(a: float, b: float) -> float:
        """Левая пробная точка: α = a + 2/(3+√5)·(b-a) = a+(b-a)/φ² ≈ a+0.382·(b-a)."""
        return a + 2.0 / (3.0 + math.sqrt(5.0)) * (b - a)

    @staticmethod
    def get_beta(a: float, b: float) -> float:
        """Правая пробная точка: β = a + 2/(1+√5)·(b-a) = a+(b-a)/φ ≈ a+0.618·(b-a)."""
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
        """Возвращает (αₘᵢₙ, число_оценок). NaN — минимум не лучше f(a)."""

        def check_convergence(f_min: float, f_0: float) -> bool:
            # Достаточно любого из двух критериев: сильное падение относительно
            # старта или уже ниже порога шума — дальше сжимать отрезок незачем.
            return parameters.decrement_factor_criteria(f_min, f_0) or parameters.target_value_criteria(f_min)

        f_0 = f_a
        # f(b) часто не посчитан вызывающим (Ньютон знает только f(x)); считаем сами.
        if not math.isfinite(f_b):
            f_b = function(b)

        # Короткое замыкание: границы уже в шуме — цикл ЗС не запускаем.
        boundary_alpha = parameters.try_resolve_step_by_target_value(f_a, f_b, a, b)
        if boundary_alpha is not None:
            return boundary_alpha, 0

        # Стартовый претендент — лучший из концов [a, b].
        if f_b < f_a:
            x_min, f_min = b, f_b
        else:
            x_min, f_min = a, f_a
        if check_convergence(f_min, f_0):
            return x_min, 0

        # Первая пара пробных точек: две оценки f, дальше будет по одной за итерацию.
        alpha = golden_section_search.get_alpha(a, b)
        f_alpha = function(alpha)
        beta = golden_section_search.get_beta(a, b)
        f_beta = function(beta)

        def update_minimum():
            # Лучшая из четырёх точек {a, α, β, b} — ответ, если ЗС оборвём критерием.
            # Сравниваем сначала внутреннюю пару, затем победителя с ближайшим концом:
            # так не теряем минимум на границе, если внутренние точки хуже.
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
            # Уже после первой пары точек критерий выполнен — вторая оценка не нужна.
            return x_min, 1

        for index in range(1, parameters.iteration_count):
            if f_alpha < f_beta:
                # Минимум левее β. Правый конец ← β, старый α становится новым β
                # (его f уже известна). Считаем только новую α на укороченном [a, b].
                b = beta
                f_b = f_beta
                beta = alpha
                f_beta = f_alpha
                alpha = golden_section_search.get_alpha(a, b)
                f_alpha = function(alpha)
            else:
                # Минимум правее α. Левый конец ← α, старый β становится новым α.
                # Считаем только новую β справа.
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
            # Цикл исчерпан, но относительно старта есть улучшение — шаг годится.
            return x_min, parameters.iteration_count + 1
        # Нет улучшения относительно f(a): Ньютон увидит NaN и применит
        # line_search_fail_action (провал / успех / минимальный шаг).
        return float("nan"), parameters.iteration_count + 1
