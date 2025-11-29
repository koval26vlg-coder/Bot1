import builtins
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from arbitrage_bot.utils.ml_profit_optimizer import MLProfitOptimizer


class MLProfitOptimizerTests(unittest.TestCase):
    """Тесты предсказания и фолбэка ML-оптимизатора порога прибыли."""

    def test_predict_uses_fallback_when_model_missing(self):
        """Если модель отсутствует, возвращается фолбэк-порог."""

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "missing_model.pkl"
            optimizer = MLProfitOptimizer(model_path, fallback_threshold=0.123)
            prediction = optimizer.predict_threshold(
                {
                    "overall_volatility": 0.5,
                    "average_spread_percent": 0.1,
                    "orderbook_imbalance": 0.0,
                    "empty_cycles": 2,
                    "market_regime": "testnet_spot",
                }
            )

            self.assertAlmostEqual(prediction, 0.123)

    def test_train_and_predict_threshold(self):
        """После обучения предсказание отличается от фолбэка и ближе к целям."""

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model.pkl"
            optimizer = MLProfitOptimizer(model_path, fallback_threshold=0.05)

            if not optimizer.ml_supported:
                self.skipTest("scikit-learn недоступен, ML-обучение пропущено")

            contexts = [
                {
                    "overall_volatility": 0.2,
                    "average_spread_percent": 0.1,
                    "orderbook_imbalance": 0.0,
                    "empty_cycles": 0,
                    "market_regime": "live_spot_normal",
                },
                {
                    "overall_volatility": 1.2,
                    "average_spread_percent": 0.3,
                    "orderbook_imbalance": 0.05,
                    "empty_cycles": 1,
                    "market_regime": "live_spot_high_volatility",
                },
                {
                    "overall_volatility": 0.05,
                    "average_spread_percent": 0.05,
                    "orderbook_imbalance": -0.02,
                    "empty_cycles": 3,
                    "market_regime": "testnet_spot",
                },
                {
                    "overall_volatility": 0.8,
                    "average_spread_percent": 0.2,
                    "orderbook_imbalance": 0.1,
                    "empty_cycles": 0,
                    "market_regime": "live_spot_normal",
                },
                {
                    "overall_volatility": 1.5,
                    "average_spread_percent": 0.4,
                    "orderbook_imbalance": 0.12,
                    "empty_cycles": 2,
                    "market_regime": "live_spot_high_volatility",
                },
            ]
            targets = [0.08, 0.12, 0.05, 0.1, 0.14]

            optimizer.train(contexts, targets)

            loaded_optimizer = MLProfitOptimizer(model_path, fallback_threshold=0.05)
            prediction = loaded_optimizer.predict_threshold(contexts[1])

            self.assertNotAlmostEqual(prediction, loaded_optimizer.fallback_threshold)
            self.assertTrue(0.05 <= prediction <= 0.2)
            self.assertAlmostEqual(prediction, targets[1], delta=0.05)

    def test_missing_sklearn_does_not_break_optimizer(self):
        """Отсутствие scikit-learn переводит оптимизатор в фолбэк без исключений."""

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name.startswith("sklearn") or name == "joblib":
                raise ImportError("sklearn missing for test")
            return original_import(name, *args, **kwargs)

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model.pkl"
            with mock.patch("builtins.__import__", side_effect=failing_import):
                optimizer = MLProfitOptimizer(model_path, fallback_threshold=0.42)

        self.assertFalse(optimizer.ml_supported)
        self.assertAlmostEqual(optimizer.predict_threshold({}), 0.42)
        self.assertAlmostEqual(optimizer.train([], []), 0.42)


if __name__ == "__main__":
    unittest.main()
