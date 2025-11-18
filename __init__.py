"""Минимальная реализация функции load_dotenv для локального использования."""

# Все комментарии должны быть на русском языке согласно инструкции пользователя

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple


def _iter_env_lines(path: Path) -> Iterable[Tuple[str, str]]:
    """Парсит файл .env построчно и возвращает пары ключ-значение."""

    if not path.exists():
        return []

    pairs = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        pairs.append((key.strip(), value.strip().strip('"').strip("'")))
    return pairs


def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, *, override: bool = False) -> bool:
    """Загружает переменные окружения из файла .env."""

    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    loaded = False
    for key, value in _iter_env_lines(path):
        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True
    return loaded

__all__ = ["load_dotenv"]