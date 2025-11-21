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

    def test_candlestick_pattern_boosts_score(self):
        """Проверяем, что свечные паттерны добавляют сигнальную окраску и метаданные."""

        rows = []
        for idx in range(20):
            close = 100 + idx * 0.2
            rows.append({
                'timestamp': self.base_time + timedelta(minutes=idx),
                'symbol': 'BTCUSDT',
                'open': close - 0.1,
                'high': close + 0.3,
                'low': close - 0.4,
                'close': close,
                'volume': 1000 + idx,
            })

        rows.append({
            'timestamp': self.base_time + timedelta(minutes=20),
            'symbol': 'BTCUSDT',
            'open': 101.5,
            'high': 102.0,
            'low': 100.0,
            'close': 100.5,
            'volume': 2000,
        })
        rows.append({
            'timestamp': self.base_time + timedelta(minutes=21),
            'symbol': 'BTCUSDT',
            'open': 100.0,
            'high': 103.5,
            'low': 99.5,
            'close': 103.0,
            'volume': 2500,
        })

        result = self.manager._multi_indicator_strategy(rows, {'volatility': 0.4, 'liquidity': 8000})

        self.assertEqual(result.meta['pattern_name'], 'bullish_engulfing')
        self.assertEqual(result.meta['pattern_bias'], 'bullish')
        self.assertGreater(result.meta['pattern_strength'], 0)
        self.assertNotEqual(result.signal, 'short')


class MultiIndicatorIntegrationTest(unittest.TestCase):
    """Проверяем выбор стратегии и влияние на динамический порог."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        config = SimpleNamespace(
            SYMBOLS=['BTCUSDT'],
            TRIANGULAR_PAIRS=[],
            MIN_TRIANGULAR_PROFIT=0.1,
            TRADING_FEE=0.001,
            SLIPPAGE_PROFIT_BUFFER=0.02,
            VOLATILITY_PROFIT_MULTIPLIER=0.05,
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
                'raw_spreads': deque(maxlen=500),
                'bid_volumes': deque(maxlen=500),
                'ask_volumes': deque(maxlen=500),
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
        expected_min_threshold = (
            self.engine.config.MIN_TRIANGULAR_PROFIT
            + self.engine.config.TRADING_FEE * 3 * 100
            + self.engine.config.SLIPPAGE_PROFIT_BUFFER
        )
        self.assertGreaterEqual(self.engine._last_dynamic_threshold, expected_min_threshold)


if __name__ == '__main__':
    unittest.main()
