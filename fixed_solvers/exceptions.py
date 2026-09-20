"""Исключения библиотеки."""


class domain_violation(BaseException):
    """Выход за область определения функции.

    Наследует `BaseException`, а не `Exception`, поэтому широкий
    `except Exception` его не перехватывает — ловить нужно явно.
    """


class logic_error(Exception):
    """Нарушение предусловия или инварианта."""


class invalid_argument(ValueError):
    """Некорректный аргумент вызова."""
