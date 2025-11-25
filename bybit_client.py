"""Обёртка для экспорта клиента и менеджера Bybit из монолитного модуля."""

import importlib


def _load_manager_class():
    """Лениво импортирует реализацию менеджера, чтобы избежать циклов."""

    module = importlib.import_module("arbitrage_bot.exchanges.bybit_client")
    return module.BybitWebSocketManager


def _load_client_class():
    """Лениво импортирует REST-клиент, определённый в монолите."""

    module = importlib.import_module("arbitrage_bot.exchanges.bybit_client")
    return module.BybitClient


class BybitWebSocketManager:
    """Прокси-класс, перенаправляющий создание реального менеджера."""

    def __new__(cls, *args, **kwargs):
        real_class = _load_manager_class()
        instance = real_class(*args, **kwargs)
        return instance


class BybitClient:
    """Прокси-класс, перенаправляющий создание REST-клиента."""

    def __new__(cls, *args, **kwargs):
        real_class = _load_client_class()
        instance = real_class(*args, **kwargs)
        return instance


__all__ = ["BybitWebSocketManager", "BybitClient"]
