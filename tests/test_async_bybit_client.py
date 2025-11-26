import importlib
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock


class DummyWebSocketManager:
    """Псевдо-менеджер, проверяющий использование параметра свежести."""

    def __init__(self, ages: dict[str, float]):
        self.ages = ages
        self.used_max_age: float | None = None

    def get_cached_tickers(self, symbols, max_age=None):
        """Возвращает только те тикеры, чья давность меньше порога."""

        self.used_max_age = max_age
        fresh = {}
        missing = []
        now_ms = int(time.time() * 1000)

        for symbol in symbols:
            age = self.ages.get(symbol)
            if age is None or (max_age is not None and age > max_age):
                missing.append(symbol)
                continue

            fresh[symbol] = {
                "bid": 1.0,
                "ask": 1.1,
                "last": 1.05,
                "timestamp": now_ms,
            }

        return fresh, missing

    def update_cache(self, tickers):
        """Заглушка для совместимости с интерфейсом менеджера."""

        return None


class AsyncBybitClientCacheTests(unittest.IsolatedAsyncioTestCase):
    """Проверка влияния параметра свежести на кэшированные котировки."""

    def setUp(self):
        """Подготавливает поддельные зависимости перед импортом клиента."""

        project_root = Path(__file__).resolve().parents[1]
        package_root = project_root / "arbitrage_bot"

        fake_aiohttp = types.ModuleType("aiohttp")

        class _ClientError(Exception):
            pass

        class _ContentTypeError(Exception):
            pass

        class _ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class _ClientSession:
            def __init__(self, timeout=None):
                self.timeout = timeout
                self.closed = False

            async def close(self):
                self.closed = True

        fake_aiohttp.ClientError = _ClientError
        fake_aiohttp.ContentTypeError = _ContentTypeError
        fake_aiohttp.ClientTimeout = _ClientTimeout
        fake_aiohttp.ClientSession = _ClientSession

        fake_config = types.ModuleType("arbitrage_bot.core.config")

        class _Config:
            def __init__(self):
                self.API_BASE_URL = "https://api.test"
                self.MARKET_CATEGORY = "spot"
                self.WEBSOCKET_PRICE_ONLY = False
                self._ticker_staleness_warning = 5.0

            @property
            def TICKER_STALENESS_WARNING_SEC(self):
                return self._ticker_staleness_warning

        fake_config.Config = _Config

        fake_bybit_client = types.ModuleType("arbitrage_bot.exchanges.bybit_client")

        fake_package = types.ModuleType("arbitrage_bot")
        fake_package.__path__ = [str(package_root)]
        fake_core = types.ModuleType("arbitrage_bot.core")
        fake_core.__path__ = [str(package_root / "core")]
        fake_exchanges = types.ModuleType("arbitrage_bot.exchanges")
        fake_exchanges.__path__ = [str(package_root / "exchanges")]
        fake_package.core = fake_core
        fake_package.exchanges = fake_exchanges
        fake_core.config = fake_config
        fake_exchanges.bybit_client = fake_bybit_client

        class _BybitWebSocketManager:
            def __init__(self, *args, **kwargs):
                pass

        fake_bybit_client.BybitWebSocketManager = _BybitWebSocketManager

        self.modules_patcher = mock.patch.dict(
            sys.modules,
            {
                "aiohttp": fake_aiohttp,
                "arbitrage_bot": fake_package,
                "arbitrage_bot.core": fake_core,
                "arbitrage_bot.exchanges": fake_exchanges,
                "arbitrage_bot.core.config": fake_config,
                "arbitrage_bot.exchanges.bybit_client": fake_bybit_client,
            },
        )
        self.modules_patcher.start()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        self.async_client_module = importlib.import_module("arbitrage_bot.core.async_bybit_client")
        self.AsyncBybitClient = self.async_client_module.AsyncBybitClient
        from arbitrage_bot.core.config import Config

        self.Config = Config

    def tearDown(self):
        """Очищает подмену модулей после теста."""

        self.modules_patcher.stop()
        for module_name in [
            "arbitrage_bot.core.async_bybit_client",
        ]:
            if module_name in sys.modules:
                del sys.modules[module_name]

    async def test_configurable_staleness_filters_cached_tickers(self):
        """Изменение порога свежести в конфиге влияет на выдачу из кэша."""

        config = self.Config()
        config.WEBSOCKET_PRICE_ONLY = True
        config._ticker_staleness_warning = 10.0

        ws_manager = DummyWebSocketManager({
            "BTCUSDT": 8.0,
            "ETHUSDT": 12.0,
        })

        with mock.patch.object(self.AsyncBybitClient, "_initialize_websocket_streams", lambda self: None):
            client = self.AsyncBybitClient(config=config)
            client.ws_manager = ws_manager
            tickers = await client.get_tickers_async(["BTCUSDT", "ETHUSDT"])

        self.assertEqual(ws_manager.used_max_age, config.TICKER_STALENESS_WARNING_SEC)
        self.assertIn("BTCUSDT", tickers)
        self.assertNotIn("ETHUSDT", tickers)


if __name__ == "__main__":
    unittest.main()
