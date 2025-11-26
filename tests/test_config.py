import builtins
import importlib
import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config_module():
    """Подгружает модуль конфигурации после применения переменных окружения."""

    if "arbitrage_bot.core.config" in sys.modules:
        return importlib.reload(sys.modules["arbitrage_bot.core.config"])
    return importlib.import_module("arbitrage_bot.core.config")


class DummyResponse:
    """Простая заглушка для имитации ответа requests."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        """Имитирует успешный статус без исключений."""

        return None

    def json(self):
        """Возвращает заранее подготовленный JSON."""

        return self._payload


@pytest.mark.parametrize(
    "market_hint,expected_category",
    [("spot", "spot"), ("linear", "linear")],
)
def test_market_category_autodetect_success(monkeypatch, market_hint, expected_category):
    """Успешный автодетект выбирает сегмент и кэширует результат."""

    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("ARBITRAGE_TICKER_CATEGORY", market_hint)
    monkeypatch.delenv("MARKET_CATEGORY_OVERRIDE", raising=False)

    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["category"])
        payload = {"retCode": 0, "result": {"list": [{"category": params["category"]}]}}
        return DummyResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    module = _load_config_module()
    config = module.Config()

    assert config.MARKET_CATEGORY == expected_category
    assert config._detected_market_category == expected_category
    assert calls == [expected_category]

    _ = config.MARKET_CATEGORY
    assert calls == [expected_category]


def test_market_category_autodetect_fallback(monkeypatch):
    """Неуспешный автодетект использует запасной сегмент и кэширует его."""

    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("ARBITRAGE_TICKER_CATEGORY", "spot")
    monkeypatch.delenv("MARKET_CATEGORY_OVERRIDE", raising=False)

    calls: list[str] = []

    def failing_get(url, params=None, timeout=None):
        calls.append(params["category"])
        raise requests.RequestException("network error")

    monkeypatch.setattr(requests, "get", failing_get)

    module = _load_config_module()
    config = module.Config()

    assert config.MARKET_CATEGORY == "linear"
    assert config._detected_market_category == "linear"
    assert calls == ["spot", "linear"]

    _ = config.MARKET_CATEGORY
    assert calls == ["spot", "linear"]


def test_config_loads_without_aiohttp(monkeypatch):
    """Config создаётся без импорта aiohttp и запуска arbitrage_bot.__init__."""

    for name in ["arbitrage_bot", "arbitrage_bot.core", "arbitrage_bot.core.config"]:
        sys.modules.pop(name, None)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("aiohttp"):
            raise ModuleNotFoundError("aiohttp отсутствует для теста")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = _load_config_module()
    config = module.Config()

    assert hasattr(config, "API_BASE_URL")
    assert "aiohttp" not in sys.modules


def test_api_base_url_uses_testnet_spot(monkeypatch):
    """URL API для тестнета корректно учитывает выбранный сегмент."""

    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("AUTO_DETECT_MARKET_CATEGORY", "false")
    monkeypatch.setenv("TESTNET_SPOT_API_BASE_URL", "https://custom.test")
    monkeypatch.setenv("ARBITRAGE_TICKER_CATEGORY", "spot")

    module = _load_config_module()
    config = module.Config()

    assert config.MARKET_CATEGORY == "spot"
    assert config.API_BASE_URL == "https://custom.test"


def test_min_profit_and_numeric_env_overrides(monkeypatch):
    """Числовые параметры окружения корректно парсятся и имеют фолбэки."""

    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("MIN_TRIANGULAR_PROFIT", "0.42")
    monkeypatch.setenv("EMPTY_CYCLE_RELAX_STEP", "нечисло")
    monkeypatch.setenv("MARKET_SYMBOLS_LIMIT", "abc")

    module = _load_config_module()
    config = module.Config()

    assert config.MIN_TRIANGULAR_PROFIT == 0.42
    assert pytest.approx(config.EMPTY_CYCLE_RELAX_STEP, rel=1e-6) == 0.05
    assert config._market_symbols_limit == 0


def test_triangular_pairs_cache_reset(monkeypatch):
    """Кэш треугольников обновляется по TTL и после сброса."""

    monkeypatch.setenv("TESTNET", "true")

    fetch_calls: list[float] = []
    template_calls: list[int] = []
    available_symbols = {"AAAUSDT", "AAABBB", "BBBUSD"}

    def fake_fetch(self):
        fetch_calls.append(time.time())
        return available_symbols

    def fake_templates(self, symbols):
        template_calls.append(1)
        suffix = len(template_calls)
        return [
            {
                "name": f"triangle-{suffix}",
                "legs": ["AAAUSDT", "AAABBB", "BBBUSD"],
            }
        ]

    module = _load_config_module()
    config = module.Config()
    real_cls = config.__class__

    monkeypatch.setattr(real_cls, "_fetch_market_symbols", fake_fetch)
    monkeypatch.setattr(real_cls, "_build_triangle_templates", fake_templates)
    config._triangles_cache_ttl = 60

    first = config.TRIANGULAR_PAIRS
    second = config.TRIANGULAR_PAIRS

    assert first is second
    assert len(fetch_calls) == 1
    assert len(template_calls) == 1

    config.reset_symbol_caches()

    third = config.TRIANGULAR_PAIRS

    assert len(fetch_calls) == 2
    assert len(template_calls) == 2
    assert third is config._triangular_pairs_cache
    assert third[0]["name"] != first[0]["name"]
