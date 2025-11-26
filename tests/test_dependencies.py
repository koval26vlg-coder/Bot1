"""Тесты для проверки корректной обработки отсутствующих зависимостей."""

import asyncio
import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest


def test_missing_aiohttp_triggers_help(monkeypatch, caplog):
    """Проверяет, что при отсутствии aiohttp выдаётся понятная ошибка при инициализации."""

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("aiohttp missing for test")
        if name in sys.modules:
            return sys.modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    project_root = Path(__file__).resolve().parents[1]
    package_root = project_root / "arbitrage_bot"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("arbitrage_bot.core.async_bybit_client", None)
    sys.modules.pop("async_bybit_client", None)

    fake_aiohttp_placeholder = types.ModuleType("aiohttp")

    fake_package = types.ModuleType("arbitrage_bot")
    fake_package.__path__ = [str(package_root)]
    fake_core = types.ModuleType("arbitrage_bot.core")
    fake_core.__path__ = [str(package_root / "core")]
    fake_exchanges = types.ModuleType("arbitrage_bot.exchanges")
    fake_exchanges.__path__ = [str(package_root / "exchanges")]

    fake_config = types.ModuleType("arbitrage_bot.core.config")

    class _Config:
        def __init__(self):
            self.API_BASE_URL = "https://api.test"
            self.MARKET_CATEGORY = "spot"
            self.WEBSOCKET_PRICE_ONLY = False
            self.SYMBOLS = []

    fake_config.Config = _Config

    fake_bybit_client = types.ModuleType("arbitrage_bot.exchanges.bybit_client")

    class _DummyWS:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *args, **kwargs):
            return None

        def stop(self):
            return None

        def get_cached_tickers(self, *args, **kwargs):
            return {}, []

        def update_cache(self, *args, **kwargs):
            return None

    fake_bybit_client.BybitWebSocketManager = _DummyWS

    monkeypatch.setattr(
        sys,
        "modules",
        {
            **sys.modules,
            "aiohttp": fake_aiohttp_placeholder,
            "arbitrage_bot": fake_package,
            "arbitrage_bot.core": fake_core,
            "arbitrage_bot.exchanges": fake_exchanges,
            "arbitrage_bot.core.config": fake_config,
            "arbitrage_bot.exchanges.bybit_client": fake_bybit_client,
        },
    )

    module = importlib.import_module("arbitrage_bot.core.async_bybit_client")
    client = module.AsyncBybitClient(allow_missing_aiohttp=True)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client._ensure_session())

    assert "aiohttp" in str(excinfo.value)
    assert "pip install -r requirements.txt" in str(excinfo.value)
    assert any("aiohttp" in record.message for record in caplog.records)

    sys.modules.pop("arbitrage_bot.core.async_bybit_client", None)
    sys.modules.pop("async_bybit_client", None)
