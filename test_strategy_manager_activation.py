import unittest
from datetime import datetime, timedelta, timezone

from config import Config
from indicator_strategies import StrategyManager


class StrategyManagerActivationTest(unittest.TestCase):
    """Эмулируем первые циклы обновления и проверяем смену стратегии."""

    def setUp(self):
        self.manager = StrategyManager(Config())
        self.base_time = datetime.now(timezone.utc)

    def _build_rows(self, closes):
        rows = []
        for idx, price in enumerate(closes):
            rows.append({
                'timestamp': self.base_time + timedelta(minutes=idx),
                'symbol': 'BTCUSDT',
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1000 + idx * 10,
            })
        return rows

    def test_strategy_switches_after_history_threshold(self):
        context = {'volatility': 0.2, 'liquidity': 100000}

        short_series = self._build_rows([100 + i for i in range(5)])
        early_result = self.manager.evaluate(short_series, context)
        self.assertEqual(self.manager.get_active_strategy_name(), 'volatility_adaptive')
        self.assertEqual(early_result.name, 'volatility_adaptive')
        self.assertNotEqual(early_result.signal, 'wait_for_data')

        long_series = self._build_rows([100 + i for i in range(20)])
        rich_result = self.manager.evaluate(long_series, context)
        self.assertEqual(self.manager.get_active_strategy_name(), 'momentum_bias')
        self.assertEqual(rich_result.name, 'momentum_bias')
        self.assertNotEqual(rich_result.signal, 'wait_for_data')


if __name__ == '__main__':
    unittest.main()
