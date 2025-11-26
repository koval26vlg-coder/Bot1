import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyConfig:
    PRIMARY_EXCHANGE = "bybit"
    ENABLE_ASYNC_MARKET_CLIENT = False
    USE_LEGACY_TICKER_CLIENT = False
    ML_MODEL_PATH = "dummy.pkl"
    ML_FALLBACK_THRESHOLD = 0.07
    MIN_TRIANGULAR_PROFIT = 0.07
    TRIANGULAR_PAIRS = [
        {"name": "TEST", "legs": ["AAA", "BBB", "CCC"]},
    ]
    SYMBOLS = ["AAA", "BBB", "CCC"]
    UPDATE_INTERVAL = 1
    API_KEY = ""
    API_SECRET = ""


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")

    def get_tickers(self, symbols):
        return {symbol: {} for symbol in symbols}


class DummyAsyncBybitClient:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get("config")

    async def close(self):
        return None


class DummyMonitor:
    def __init__(self, *_):
        self.started = False

    def start_monitoring_loop(self):
        self.started = True


class DummyRealTradingExecutor:
    def __init__(self, *_):
        self.started = True


class DummyStrategyManager:
    def __init__(self, *_):
        self.configured = True

    def update_config(self, *_):
        return None


class DummyPerformanceOptimizer:
    def __init__(self, *_):
        self.configured = True

    def update_config(self, *_):
        return None

    def get_optimized_triangles(self):
        return []


class DummyMLProfitOptimizer:
    def __init__(self, model_path, fallback_threshold):
        self.model_path = Path(model_path)
        self.fallback_threshold = fallback_threshold
        self.ml_supported = False
        self.model = None

    def predict_threshold(self, context):
        return self.fallback_threshold


class AdvancedEngineFallbackTests(unittest.TestCase):
    """Проверяет, что движок корректно работает без scikit-learn."""

    def test_engine_initializes_without_ml_dependency(self):
        """Создание движка не падает, даже если ML отключён."""

        for module_name in [
            "arbitrage_bot.core.advanced_arbitrage_engine",
            "arbitrage_bot.core.config",
        ]:
            sys.modules.pop(module_name, None)

        engine_module = importlib.import_module("arbitrage_bot.core.advanced_arbitrage_engine")

        with mock.patch.multiple(
            engine_module,
            AsyncBybitClient=DummyAsyncBybitClient,
            BybitClient=DummyClient,
            OkxClient=DummyClient,
            AdvancedMonitor=DummyMonitor,
            RealTradingExecutor=DummyRealTradingExecutor,
            StrategyManager=DummyStrategyManager,
            PerformanceOptimizer=DummyPerformanceOptimizer,
            MLProfitOptimizer=DummyMLProfitOptimizer,
        ):
            engine = engine_module.AdvancedArbitrageEngine(config=DummyConfig())

        for module_name in [
            "arbitrage_bot.core.advanced_arbitrage_engine",
            "arbitrage_bot.core.config",
        ]:
            sys.modules.pop(module_name, None)

        self.assertIsNotNone(engine)
        self.assertFalse(engine.ml_profit_optimizer.ml_supported)
        self.assertAlmostEqual(
            engine.ml_profit_optimizer.predict_threshold({}), DummyConfig.ML_FALLBACK_THRESHOLD
        )


if __name__ == "__main__":
    unittest.main()
