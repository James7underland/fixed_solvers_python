"""Исключения библиотеки (аналоги C++-типов из fixed_solvers)."""


class domain_violation(BaseException):
    """Выход за область определения функции.

    В C++ тип `domain_violation` намеренно не наследует `std::exception`,
    чтобы его ловили только явные `catch (const domain_violation&)`.
    Здесь класс наследует `BaseException`, а не `Exception`, поэтому
    широкий `except Exception` его не перехватывает.
    """


class logic_error(Exception):
    """Аналог std::logic_error."""


class invalid_argument(ValueError):
    """Аналог std::invalid_argument."""
