"""Перечисления солверов."""

from __future__ import annotations

from enum import IntEnum


class step_constraint_algorithm_t(IntEnum):
    Quadprog = 0
    CoordinateDescent = 1


class line_search_explore_domain_violation_action_t(IntEnum):
    record_nan = 0
    rethrow = 1


class line_search_fail_action_t(IntEnum):
    TreatAsFail = 0
    TreatAsSuccess = 1
    PerformMinStep = 2


class numerical_result_code_t(IntEnum):
    NoNumericalError = 0
    IllConditionedMatrix = 1
    LargeConditionNumber = 2
    CustomCriteriaFailed = 3
    NotConverged = 4
    NumericalNanValues = 5
    LineSearchFailed = 6
    Converged = 7
    InProgress = 8

    def __str__(self) -> str:
        return f"{self.name}({int(self)})"


class convergence_score_t(IntEnum):
    Excellent = 5
    Good = 4
    Satisfactory = 3
    Poor = 2
    Error = 1

    def __str__(self) -> str:
        return get_score_strings()[self]


class domain_discovery_mode_t(IntEnum):
    forbid_exit = 0
    require_connected_domain = 1
    allow_disconnected_domain = 2


class fixed_bisectional_solution_type(IntEnum):
    Bisection = 0
    Secant = 1
    Combined = 2


class nullable_bool_t(IntEnum):
    False_ = 0b0
    True_ = 0b1
    Undefined = 0b10

    def __str__(self) -> str:
        if self == nullable_bool_t.True_:
            return "True"
        if self == nullable_bool_t.False_:
            return "False"
        if self == nullable_bool_t.Undefined:
            return "Undefined"
        return "unknown nullable bool value"


SCORE_STRINGS = {
    convergence_score_t.Excellent: "Excellent(5)",
    convergence_score_t.Good: "Good(4)",
    convergence_score_t.Satisfactory: "Satisfactory(3)",
    convergence_score_t.Poor: "Poor(2)",
    convergence_score_t.Error: "Error(1)",
}


def get_score_strings() -> dict[convergence_score_t, str]:
    return SCORE_STRINGS


def get_score_wstrings() -> dict[convergence_score_t, str]:
    return SCORE_STRINGS


def get_score_total_calculations(score: dict[convergence_score_t, int]) -> int:
    return int(sum(score.values()))


def get_score_string(score: dict[convergence_score_t, int]) -> str:
    total = get_score_total_calculations(score)
    parts: list[str] = []
    for sc, count in score.items():
        percent = 100.0 * float(count) / total if total else 0.0
        parts.append(f"{SCORE_STRINGS[sc]}: {percent:.4g}%")
    return " ".join(parts) + (" " if parts else "")


def get_converged_percent(score: dict[convergence_score_t, int]) -> float:
    total = get_score_total_calculations(score)
    if total == 0:
        return 0.0
    converged_count = 0
    for sc, count in score.items():
        if sc in (
            convergence_score_t.Excellent,
            convergence_score_t.Good,
            convergence_score_t.Satisfactory,
        ):
            converged_count += count
    return float(converged_count) / total * 100.0


small_step_threshold = 0.1
