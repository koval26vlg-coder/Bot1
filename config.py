"""Обёртка для экспорта конфигурации из монолитного модуля."""

import importlib


def _load_config_class():
    """Лениво импортирует Config, чтобы избежать циклических зависимостей."""

    module = importlib.import_module("bot_bundle")
    return module.Config


class Config:
    """Прокси-класс, создающий экземпляры реальной конфигурации."""

    def __new__(cls, *args, **kwargs):
        real_class = _load_config_class()
        instance = real_class(*args, **kwargs)
        return instance


__all__ = ["Config"]
