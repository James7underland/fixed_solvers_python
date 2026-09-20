"""Строковые хелперы."""

from __future__ import annotations

import os
import sys
from typing import Iterable, TextIO


def string_ends_with(value: str, ending: str) -> bool:
    if len(ending) > len(value):
        return False
    return value.endswith(ending)


def UTF8_to_wchar(text: str | bytes) -> str:
    """Декодер UTF-8 побайтово."""
    if isinstance(text, str):
        data = text.encode("utf-8")
    else:
        data = bytes(text)
    out: list[str] = []
    codepoint = 0
    index = 0
    length = len(data)
    while index < length:
        ch = data[index]
        if ch <= 0x7F:
            codepoint = ch
        elif ch <= 0xBF:
            codepoint = (codepoint << 6) | (ch & 0x3F)
        elif ch <= 0xDF:
            codepoint = ch & 0x1F
        elif ch <= 0xEF:
            codepoint = ch & 0x0F
        else:
            codepoint = ch & 0x07
        index += 1
        nxt = data[index] if index < length else 0
        if ((nxt & 0xC0) != 0x80) and (codepoint <= 0x10FFFF):
            if sys.maxunicode > 0xFFFF:
                out.append(chr(codepoint))
            elif codepoint > 0xFFFF:
                out.append(chr(0xD800 + (codepoint >> 10)))
                out.append(chr(0xDC00 + (codepoint & 0x03FF)))
            elif codepoint < 0xD800 or codepoint >= 0xE000:
                out.append(chr(codepoint))
    return "".join(out)


def wchar_to_UTF8(text: str) -> str:
    """Кодирование широкой строки в UTF-8."""
    out = bytearray()
    codepoint = 0
    for ch in text:
        value = ord(ch)
        if 0xD800 <= value <= 0xDBFF:
            codepoint = ((value - 0xD800) << 10) + 0x10000
            continue
        if 0xDC00 <= value <= 0xDFFF:
            codepoint |= value - 0xDC00
        else:
            codepoint = value
        if codepoint <= 0x7F:
            out.append(codepoint)
        elif codepoint <= 0x7FF:
            out.append(0xC0 | ((codepoint >> 6) & 0x1F))
            out.append(0x80 | (codepoint & 0x3F))
        elif codepoint <= 0xFFFF:
            out.append(0xE0 | ((codepoint >> 12) & 0x0F))
            out.append(0x80 | ((codepoint >> 6) & 0x3F))
            out.append(0x80 | (codepoint & 0x3F))
        else:
            out.append(0xF0 | ((codepoint >> 18) & 0x07))
            out.append(0x80 | ((codepoint >> 12) & 0x3F))
            out.append(0x80 | ((codepoint >> 6) & 0x3F))
            out.append(0x80 | (codepoint & 0x3F))
        codepoint = 0
    return out.decode("utf-8")


def string2wide(text: str | bytes) -> str:
    """Windows: интерпретация байт как cp1251; иначе UTF-8."""
    if os.name != "nt":
        raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        return UTF8_to_wchar(raw)
    try:
        if isinstance(text, (bytes, bytearray)):
            return bytes(text).decode("cp1251")
        raw = text.encode("latin-1")
        return raw.decode("cp1251")
    except Exception:
        if isinstance(text, (bytes, bytearray)):
            return "".join(chr(b) for b in text)
        return str(text)


def wide2string(text: str) -> str:
    return wchar_to_UTF8(text)


def int2str(value) -> str:
    return str(value)


def int2wstr(value) -> str:
    return str(value)


def string_replace(text: str, frm: str, to: str) -> str:
    """Замена подстроки: не пересканирует вставленный фрагмент."""
    if not frm:
        return text
    start_pos = 0
    while True:
        found = text.find(frm, start_pos)
        if found < 0:
            break
        text = text[:found] + to + text[found + len(frm) :]
        start_pos = found + len(to)
    return text


def save_vector(stream: TextIO, values: Iterable) -> TextIO:
    values = list(values)
    stream.write(f"{len(values)}\n")
    for val in values:
        stream.write(f"{val}\n")
    stream.write("\n")
    return stream


def load_vector(stream: TextIO) -> list:
    raw = stream.readline()
    if not raw:
        raise RuntimeError("bad v_size")
    try:
        v_size = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError("bad v_size") from exc
    result = []
    for _ in range(v_size):
        line = stream.readline()
        if not line:
            raise RuntimeError("v read error")
        token = line.strip()
        try:
            if any(ch in token for ch in ".eE"):
                result.append(float(token))
            else:
                result.append(int(token))
        except ValueError as exc:
            raise RuntimeError("v read error") from exc
    return result
