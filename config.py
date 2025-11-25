"""Обёртка для экспорта конфигурации из монолитного модуля."""

import importlib.util
import sys
from pathlib import Path


def _load_config_class():
    """Лениво импортирует Config, чтобы избежать циклических зависимостей."""

    module_path = Path(__file__).resolve().parent / "arbitrage_bot" / "core" / "config.py"
    spec = importlib.util.spec_from_file_location("arbitrage_bot.core.config", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Не удалось создать спецификацию для arbitrage_bot.core.config")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Config


class Config:
    """Прокси-класс, создающий экземпляры реальной конфигурации."""

    def __new__(cls, *args, **kwargs):
        real_class = _load_config_class()
        instance = real_class(*args, **kwargs)
        return instance


__all__ = ["Config"]
