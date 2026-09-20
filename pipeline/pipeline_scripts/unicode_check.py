#!/usr/bin/env python3
"""Проверка, что исходники репозитория сохранены в UTF-8 / ASCII.

Аналог pipeline/pipeline_scripts/unicode_check.sh из C++-проекта:
там проверялись *.m *.cpp *.h, здесь — Python-исходники и тексты пайплайна.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "pipeline" / "pipeline_result"
ERROR_LIST = RESULT_DIR / "encoding_error_list.txt"
SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".ini"}
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "fixed_solvers.egg-info",
}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def is_utf8(data: bytes) -> bool:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff") or data.startswith(b"\x00\x00\xfe\xff"):
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if ERROR_LIST.exists():
        ERROR_LIST.unlink()

    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or is_skipped(path):
            continue
        if path.suffix.lower() not in SUFFIXES:
            continue
        data = path.read_bytes()
        if not is_utf8(data):
            errors.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    print("--------------- Result ---------------")
    if errors:
        ERROR_LIST.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(
            "Error: some files are not in UTF-8 encoding. "
            "See the list of affected files in pipeline/pipeline_result/encoding_error_list.txt"
        )
        print("\n".join(errors))
        print("--------------------------------------")
        return 1
    print("All files are in Unicode encoding.")
    print("--------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
