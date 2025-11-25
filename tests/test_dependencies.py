"""Тесты для проверки корректной обработки отсутствующих зависимостей."""

import builtins
import sys
from pathlib import Path

import pytest


def test_missing_aiohttp_triggers_help(monkeypatch, capsys):
    """Проверяет, что при отсутствии aiohttp выводится подсказка и выполнение прекращается."""

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("aiohttp missing for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("async_bybit_client", None)

    with pytest.raises(SystemExit) as excinfo:
        __import__("async_bybit_client")

    assert excinfo.value.code == 1

    captured = capsys.readouterr().out
    assert "aiohttp" in captured
    assert "pip install -r requirements.txt" in captured

    sys.modules.pop("async_bybit_client", None)
