"""Исключения библиотеки.

domain_violation специально не Exception: солвер ловит его точечно,
а пользовательский ``except Exception`` не спрячет выход за ООФ как «обычный сбой».
NaN из функции при этом НЕ считается ООФ — это NumericalNanValues.
"""


class domain_violation(BaseException):
    """Выход за область определения функции.

    Наследует `BaseException`, а не `Exception`, поэтому широкий
    `except Exception` его не перехватывает — ловить нужно явно.
    """


class logic_error(Exception):
    """Нарушение предусловия или инварианта."""


class invalid_argument(ValueError):
    """Некорректный аргумент вызова."""
