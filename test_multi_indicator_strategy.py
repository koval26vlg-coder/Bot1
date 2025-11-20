import unittest
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine
from indicator_strategies import StrategyManager, StrategyResult


class MultiIndicatorStrategyTest(unittest.TestCase):
    """Проверяем расчёт комбинированной стратегии и её сигналы."""

    def setUp(self):
        self.manager = StrategyManager(SimpleNamespace())
        self.base_time = datetime.utcnow()

    def _build_rows(self, closes):
        rows = []
        for idx, close in enumerate(closes):
            rows.append({
                'timestamp': self.base_time + timedelta(minutes=idx),
                'symbol': 'BTCUSDT',
                'open': close - 0.5,
                'high': close + 1,
                'low': close - 1,
                'close': close,
                'volume': 1000 + idx * 5,
            })
        return rows

    def test_indicator_values_and_signal(self):
        closes = [100 + i for i in range(25)]
        market_df = self._build_rows(closes)
        result = self.manager._multi_indicator_strategy(market_df, {'volatility': 0.2, 'liquidity': 10000})

        self.assertEqual(result.name, 'multi_indicator')
        self.assertEqual(result.signal, 'long')
        self.assertGreater(result.score, 0)
        self.assertGreater(result.confidence, 0)
        self.assertGreater(result.meta['rsi'], 60)
        self.assertGreater(result.meta['short_ema'], result.meta['long_ema'])
        self.assertGreater(result.meta['atr'], 0)


class MultiIndicatorIntegrationTest(unittest.TestCase):
    """Проверяем выбор стратегии и влияние на динамический порог."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        config = SimpleNamespace(
            SYMBOLS=['BTCUSDT'],
            TRIANGULAR_PAIRS=[],
            MIN_TRIANGULAR_PROFIT=0.1,
            TRADING_FEE=0.001,
            STRATEGY_MODE='auto',
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
        self.engine.no_opportunity_cycles = 0
        self.engine.triangle_stats = {}
        self.engine.monitor = None

    def _build_ticker(self, base_price):
        return {
            'bid': base_price,
            'ask': base_price + 0.5,
            'open': base_price,
            'high': base_price + 1,
            'low': base_price - 1,
            'last_price': base_price + 0.2,
            'volume': 1000 + base_price,
        }

    def test_multi_indicator_selected_on_trend(self):
        for i in range(25):
            price = 100 + i
            self.engine.update_market_data({'BTCUSDT': self._build_ticker(price)})

        snapshot = self.engine._build_market_dataframe()
        result = self.engine.evaluate_strategies(snapshot)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, 'multi_indicator')
        self.assertEqual(self.engine.strategy_manager.get_active_strategy_name(), 'multi_indicator')

    def test_dynamic_threshold_responds_to_multi_indicator(self):
        strategy_result = StrategyResult(
            name='multi_indicator',
            signal='long',
            score=2.0,
            confidence=0.5,
            meta={'atr_percent': 0.2},
        )
        opportunities = self.engine.detect_triangular_arbitrage({}, strategy_result=strategy_result)

        self.assertEqual(opportunities, [])
        self.assertLess(self.engine._last_dynamic_threshold, self.engine.config.MIN_TRIANGULAR_PROFIT)


if __name__ == '__main__':
    unittest.main()
