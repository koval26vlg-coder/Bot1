import unittest
from collections import deque
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine
from indicator_strategies import StrategyManager


class StrategyFreshDataFlowTest(unittest.TestCase):
    """Проверяем, что стратегия переключается сразу после добора истории."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        config = SimpleNamespace(
            SYMBOLS=['BTCUSDT'],
            TRIANGULAR_PAIRS=[],
            MIN_TRIANGULAR_PROFIT=0.1,
            TRADING_FEE=0.001,
            STRATEGY_MODE='auto'
        )
        self.engine.config = config
        self.engine.strategy_manager = StrategyManager(config)
        self.engine.price_history = {
            'BTCUSDT': {
                'timestamps': deque(maxlen=500),
                'bids': deque(maxlen=500),
                'asks': deque(maxlen=500),
                'spreads': deque(maxlen=500),
            }
        }
        self.engine.volatility_data = {
            'BTCUSDT': {
                'short_term': deque(maxlen=50),
                'long_term': deque(maxlen=200),
            }
        }
        self.engine.ohlcv_history = {
            'BTCUSDT': {
                'timestamps': deque(maxlen=500),
                'open': deque(maxlen=500),
                'high': deque(maxlen=500),
                'low': deque(maxlen=500),
                'close': deque(maxlen=500),
                'volume': deque(maxlen=500),
            }
        }
        self.engine.last_strategy_context = {}
        self.engine._last_candidates = []
        self.engine._last_market_analysis = {'market_conditions': 'normal'}

    def _build_ticker(self, index):
        if index < 11:
            base_price = 100 + index * 0.01
        else:
            base_price = 200
        return {
            'bid': base_price,
            'ask': base_price + 0.5,
            'open': base_price,
            'high': base_price + 1,
            'low': base_price - 1,
            'last_price': base_price + 0.2,
            'volume': 1000 + index * 10,
        }

    def test_strategy_switches_after_sufficient_bars(self):
        for i in range(11):
            self.engine.update_market_data({'BTCUSDT': self._build_ticker(i)})

        partial_snapshot = self.engine._build_market_dataframe()
        initial_result = self.engine.evaluate_strategies(partial_snapshot)
        self.assertIsNotNone(initial_result)
        self.assertEqual(self.engine.strategy_manager.get_active_strategy_name(), 'volatility_adaptive')

        self.engine.update_market_data({'BTCUSDT': self._build_ticker(11)})
        full_snapshot = self.engine._build_market_dataframe()
        fresh_result = self.engine.evaluate_strategies(full_snapshot)

        self.assertIsNotNone(fresh_result)
        self.assertEqual(len(full_snapshot), 12)
        self.assertEqual(fresh_result.name, 'momentum_bias')
        self.assertEqual(self.engine.strategy_manager.get_active_strategy_name(), 'momentum_bias')
        self.assertNotEqual(fresh_result.signal, 'wait_for_data')


if __name__ == '__main__':
    unittest.main()
