"""Кусочно-заданные функции и полиномы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from ..exceptions import logic_error
from .cubic_equation_functions import solve_cubic_equation
from .math_helpers import polyval

Coeffs = TypeVar("Coeffs")


def value_in_range(value: float, range_begin: float, range_end: float) -> bool:
    """True, если ``value`` лежит на отрезке независимо от порядка границ."""
    if range_begin > range_end:
        return value_in_range(value, range_end, range_begin)
    return range_begin <= value <= range_end


def poly_integral_coefficients(poly_coeffs: Sequence[float]) -> list[float]:
    """Коэффициенты первообразной: ``a_k / (k+1)`` со свободным членом 0."""
    n = len(poly_coeffs)
    result = [0.0] * (n + 1)
    for index in range(1, n + 1):
        result[index] = float(poly_coeffs[index - 1]) / index
    return result


@dataclass
class function_range_t(Generic[Coeffs]):
    """Один кусок кусочной функции: полуинтервал и коэффициенты на нём."""
    range_start: float
    range_end: float
    coefficients: Coeffs


class ranged_function_t(Generic[Coeffs]):
    """Кусочная функция: диапазоны отсортированы по ``range_start``."""
    def __init__(self, ranges: Sequence[function_range_t[Coeffs]] | None = None) -> None:
        self.ranges: list[function_range_t[Coeffs]] = list(ranges or [])
        self.ranges.sort(key=lambda item: item.range_start)

    def get_range_index(self, x: float) -> int:
        """Индекс куска, содержащего ``x`` (по правому концу отрезка)."""
        for index, rng in enumerate(self.ranges):
            if x < rng.range_end:
                if x > rng.range_end:
                    raise logic_error("approximation range not found")
                return index
        raise logic_error("approximation range not found")

    def get_ranges(self) -> list[function_range_t[Coeffs]]:
        """Список кусков в порядке возрастания левой границы."""
        return self.ranges

    def get_whole_range(self) -> tuple[float, float]:
        """Объединение всех кусков: от первой левой границы до последней правой."""
        if not self.ranges:
            raise RuntimeError("cannot get whole range for empty range list")
        return self.ranges[0].range_start, self.ranges[-1].range_end


class ranged_polynom_t(ranged_function_t[Sequence[float]]):
    """Кусочный полином с проверкой стыковки значений на границах кусков."""
    def __init__(
        self,
        ranges: Sequence[function_range_t[Sequence[float]]] | None = None,
        gain: float = 1.0,
        offset: float = 0.0,
        dy_error: float = 1e-8,
        _raw_gain_offset: bool = False,
    ) -> None:
        if ranges is None:
            super().__init__([])
            self.gain = float("nan")
            self.offset = float("nan")
            self.polynom_integral: list[list[float]] = []
            self.boundary_values: list[tuple[float, float]] = []
            return

        super().__init__(ranges)
        self.gain = float(gain)
        self.offset = float(offset)
        self.polynom_integral = []
        self.boundary_values = []
        for rng in self.ranges:
            self.boundary_values.append(
                (
                    self.get_polynom_value_on_range(rng, rng.range_start),
                    self.get_polynom_value_on_range(rng, rng.range_end),
                )
            )
        for index in range(len(self.boundary_values) - 1):
            y_prev = self.boundary_values[index][1]
            y_next = self.boundary_values[index + 1][0]
            if abs(y_prev - y_next) > dy_error:
                raise logic_error("dy between ranges is too large")

    def get_polynom_value_on_range(self, rng: function_range_t[Sequence[float]], x: float) -> float:
        """Значение полинома куска ``rng`` в точке ``x`` с учётом gain/offset."""
        result = polyval(rng.coefficients, x)
        return result * self.gain + self.offset

    def get_polynom_value(self, x: float, rng: function_range_t[Sequence[float]] | None = None) -> float:
        """Значение кусочного полинома; кусок ищется по ``x``, если не задан явно."""
        if rng is None:
            range_index = self.get_range_index(x)
            rng = self.ranges[range_index]
        return self.get_polynom_value_on_range(rng, x)

    def get_polynom_value_integral(self, x: float) -> float:
        """Первообразная на текущем куске (кэширует интегральные коэффициенты)."""
        if not self.polynom_integral:
            for rng in self.ranges:
                self.polynom_integral.append(poly_integral_coefficients(rng.coefficients))
        range_index = self.get_range_index(x)
        result = polyval(self.polynom_integral[range_index], x)
        return result * self.gain + self.offset * x

    def get_inv_range_index(self, y: float) -> int:
        """Кусок, на котором значение полинома покрывает ``y``."""
        for index, bounds in enumerate(self.boundary_values):
            if value_in_range(y, bounds[0], bounds[1]):
                return index
        raise logic_error("approximation range not found")

    def get_inv_polynom_value(self, y: float) -> float:
        """Обратная функция: x такой, что p(x) = y, корень должен быть один на куске."""
        if not self.ranges:
            raise logic_error("No polynom ranges defined")
        range_index = self.get_inv_range_index(y)
        rng = self.ranges[range_index]
        polynom_order = len(rng.coefficients) - 1
        if polynom_order == 1:
            c = rng.coefficients
            return (y - c[0]) / c[1]
        if polynom_order == 2:
            raise logic_error("inv polynom order 2 not implemented")
        if polynom_order == 3:
            equation = list(rng.coefficients)
            equation[0] -= y
            roots = solve_cubic_equation(equation)
        else:
            raise logic_error("inv polynom higher order not implemented")
        root_selected = [x for x in roots if rng.range_start <= x <= rng.range_end]
        if len(root_selected) != 1:
            raise logic_error("wrong root count")
        return root_selected[0]
