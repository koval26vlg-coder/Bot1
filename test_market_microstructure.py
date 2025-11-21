import unittest
from collections import deque
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine


class MarketMicrostructureContextTest(unittest.TestCase):
    """Проверяем обновление спреда и дисбаланса в контексте."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = SimpleNamespace(
            SYMBOLS=['BTCUSDT'],
            TRIANGULAR_PAIRS=[],
            MIN_TRIANGULAR_PROFIT=0.1,
            TRADING_FEE=0.001,
            SLIPPAGE_PROFIT_BUFFER=0.02,
            VOLATILITY_PROFIT_MULTIPLIER=0.05,
            STRATEGY_MODE='auto',
            EMPTY_CYCLE_RELAX_STEP=0.01,
            EMPTY_CYCLE_RELAX_MAX=0.05,
            MIN_DYNAMIC_PROFIT_FLOOR=0.0,
        )

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
        self.engine.strategy_manager = type('StubStrategyManager', (), {'evaluate': lambda *args, **kwargs: None})()

    def _build_ticker(self, bid, ask, bid_size, ask_size, volume):
        return {
            'bid': bid,
            'ask': ask,
            'open': bid,
            'high': ask,
            'low': bid,
            'last_price': ask,
            'volume': volume,
            'bid_size': bid_size,
            'ask_size': ask_size,
        }

    def test_microstructure_metrics_added_to_context(self):
        for i in range(5):
            ticker = self._build_ticker(100 + i, 100.5 + i, 500 + i * 10, 400 + i * 5, 1000 + i)
            self.engine.update_market_data({'BTCUSDT': ticker})

        market_state = self.engine.analyze_market_conditions()
        self.assertGreater(market_state['average_spread_percent'], 0)
        self.assertNotEqual(market_state['orderbook_imbalance'], 0)

        market_df = self.engine._build_market_dataframe()
        strategy_context = self.engine.evaluate_strategies(market_df)
        # evaluate_strategies возвращает StrategyResult или None, но нам нужен контекст
        self.assertIn('average_spread_percent', self.engine.last_strategy_context)
        self.assertIn('orderbook_imbalance', self.engine.last_strategy_context)
        self.assertAlmostEqual(
            self.engine.last_strategy_context['average_spread_percent'],
            market_state['average_spread_percent'],
            delta=1e-6
        )


class DynamicSpreadThresholdTest(unittest.TestCase):
    """Проверяем влияние спреда на динамический порог прибыли."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = SimpleNamespace(
            SYMBOLS=['ABCUST', 'BCDUST', 'ABCD'],
            TRIANGULAR_PAIRS=[{'name': 'TEST', 'legs': ['ABCUST', 'BCDUST', 'ABCD'], 'priority': 1}],
            MIN_TRIANGULAR_PROFIT=0.1,
            TRADING_FEE=0.001,
            SLIPPAGE_PROFIT_BUFFER=0.02,
            VOLATILITY_PROFIT_MULTIPLIER=0.05,
            STRATEGY_MODE='auto',
            EMPTY_CYCLE_RELAX_STEP=0.01,
            EMPTY_CYCLE_RELAX_MAX=0.05,
            MIN_DYNAMIC_PROFIT_FLOOR=0.0,
        )

        self.engine.price_history = {
            'ABCUST': {'timestamps': deque(maxlen=500), 'bids': deque(maxlen=500), 'asks': deque(maxlen=500), 'spreads': deque(maxlen=500), 'raw_spreads': deque(maxlen=500), 'bid_volumes': deque(maxlen=500), 'ask_volumes': deque(maxlen=500)},
            'BCDUST': {'timestamps': deque(maxlen=500), 'bids': deque(maxlen=500), 'asks': deque(maxlen=500), 'spreads': deque(maxlen=500), 'raw_spreads': deque(maxlen=500), 'bid_volumes': deque(maxlen=500), 'ask_volumes': deque(maxlen=500)},
            'ABCD': {'timestamps': deque(maxlen=500), 'bids': deque(maxlen=500), 'asks': deque(maxlen=500), 'spreads': deque(maxlen=500), 'raw_spreads': deque(maxlen=500), 'bid_volumes': deque(maxlen=500), 'ask_volumes': deque(maxlen=500)},
        }
        self.engine.volatility_data = {symbol: {'short_term': deque(maxlen=50), 'long_term': deque(maxlen=200)} for symbol in self.engine.price_history}
        for vols in self.engine.volatility_data.values():
            vols['short_term'].append(1.0)
        self.engine.ohlcv_history = {symbol: {'timestamps': deque(maxlen=500), 'open': deque(maxlen=500), 'high': deque(maxlen=500), 'low': deque(maxlen=500), 'close': deque(maxlen=500), 'volume': deque(maxlen=500)} for symbol in self.engine.price_history}
        self.engine.triangle_stats = {'TEST': {'opportunities_found': 0, 'executed_trades': 0, 'failures': 0, 'total_profit': 0, 'last_execution': None, 'success_rate': 0}}
        self.engine.no_opportunity_cycles = 0
        self.engine._last_candidates = []
        self.engine._last_market_analysis = {'market_conditions': 'normal'}
        self.engine._check_liquidity = lambda *args, **kwargs: True
        self.engine._check_triangle_volatility = lambda *args, **kwargs: True
        self.engine._prepare_direction_sequences = lambda legs, direction: [legs]
        self.engine._build_universal_path = lambda *args, **kwargs: ['path']
        self.engine._calculate_triangular_profit_path = lambda *args, **kwargs: 0.8

    def _ticker(self, bid, ask):
        return {'bid': bid, 'ask': ask, 'open': bid, 'high': ask, 'low': bid, 'last_price': ask, 'volume': 1000}

    def test_threshold_reacts_to_spread(self):
        tight_tickers = {leg: self._ticker(100, 100.1) for leg in self.engine.config.TRIANGULAR_PAIRS[0]['legs']}
        opportunities = self.engine.detect_triangular_arbitrage(tight_tickers, strategy_result=None)
        min_with_costs = (
            self.engine.config.MIN_TRIANGULAR_PROFIT
            + self.engine.config.TRADING_FEE * 3 * 100
            + self.engine.config.SLIPPAGE_PROFIT_BUFFER
        )
        self.assertGreaterEqual(self.engine._last_dynamic_threshold, min_with_costs)
        self.assertTrue(any(op['profit_percent'] > self.engine._last_dynamic_threshold for op in opportunities))

        self.engine._calculate_triangular_profit_path = lambda *args, **kwargs: 0.3
        wide_tickers = {leg: self._ticker(100, 102) for leg in self.engine.config.TRIANGULAR_PAIRS[0]['legs']}
        opportunities = self.engine.detect_triangular_arbitrage(wide_tickers, strategy_result=None)
        self.assertGreater(self.engine._last_dynamic_threshold, min_with_costs)
        self.assertEqual(opportunities, [])


if __name__ == '__main__':
    unittest.main()
