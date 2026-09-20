from io import StringIO

from fixed_solvers import (
    UTF8_to_wchar,
    int2str,
    int2wstr,
    load_vector,
    save_vector,
    string2wide,
    string_ends_with,
    string_replace,
    wchar_to_UTF8,
    wide2string,
)


def test_string_ends_with():
    # Arrange / Act / Assert
    assert string_ends_with("filename.cpp", ".cpp")
    assert not string_ends_with("filename.cpp", ".hpp")
    assert not string_ends_with("ab", "abc")


def test_utf8_roundtrip_cyrillic():
    # Arrange
    text = "Привет"
    # Act
    wide = UTF8_to_wchar(text.encode("utf-8"))
    encoded = wchar_to_UTF8(wide)
    # Assert
    assert wide == text
    assert encoded == text
    assert wide2string(wide) == text


def test_string_replace_skips_inserted_fragment():
    # Arrange
    # C++ string_replace и Python str.replace для "x"→"xx" на "xxx" дают 6 иксов:
    # после каждой вставки поиск продолжается за концом `to`.
    # Act
    result = string_replace("xxx", "x", "xx")
    # Assert
    assert result == "xxxxxx"
    assert result == "xxx".replace("x", "xx")


def test_string_replace_empty_from_is_noop():
    # Arrange / Act
    result = string_replace("abc", "", "x")
    # Assert
    assert result == "abc"


def test_int_to_string_helpers():
    # Arrange / Act / Assert
    assert int2str(42) == "42"
    assert int2wstr(-7) == "-7"


def test_save_and_load_vector():
    # Arrange
    stream = StringIO()
    # Act
    save_vector(stream, [1, 2, 3])
    stream.seek(0)
    loaded = load_vector(stream)
    # Assert
    assert loaded == [1, 2, 3]


def test_string2wide_accepts_unicode_text():
    # Arrange
    text = "solver"
    # Act
    wide = string2wide(text)
    # Assert
    assert wide == text
