import unittest
from collections import defaultdict
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine


class AggressiveModeFilteringTest(unittest.TestCase):
    """Проверяем, что агрессивный режим не выдает отрицательные проценты."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = SimpleNamespace(MIN_TRIANGULAR_PROFIT=0.2)
        self.engine._last_market_analysis = {'market_conditions': 'normal'}
        self.engine._last_candidates = []
        self.engine.triangle_stats = {
            'NEG_TRIANGLE': {'opportunities_found': 0},
            'LOW_TRIANGLE': {'opportunities_found': 0},
            'VALID_TRIANGLE': {'opportunities_found': 0}
        }
        self.engine.aggressive_filter_metrics = defaultdict(int)
        self.engine._calculate_aggressive_alpha = lambda *args, **kwargs: 0.05

    def test_negative_and_low_profit_candidates_filtered(self):
        strategy_result = SimpleNamespace(signal='neutral', score=0)
        self.engine._last_candidates = [
            {
                'triangle_name': 'NEG_TRIANGLE',
                'triangle': {'legs': ['A', 'B', 'C'], 'priority': 1, 'base_currency': 'USDT'},
                'best_direction': {
                    'profit_percent': -0.5,
                    'direction': 'forward',
                    'path': ['A', 'B', 'C']
                },
                'prices': {}
            },
            {
                'triangle_name': 'LOW_TRIANGLE',
                'triangle': {'legs': ['D', 'E', 'F'], 'priority': 1, 'base_currency': 'USDT'},
                'best_direction': {
                    'profit_percent': 0.05,
                    'direction': 'forward',
                    'path': ['D', 'E', 'F']
                },
                'prices': {}
            },
            {
                'triangle_name': 'VALID_TRIANGLE',
                'triangle': {'legs': ['G', 'H', 'I'], 'priority': 1, 'base_currency': 'USDT'},
                'best_direction': {
                    'profit_percent': 0.3,
                    'direction': 'forward',
                    'path': ['G', 'H', 'I']
                },
                'prices': {}
            }
        ]

        opportunities = self.engine._generate_aggressive_opportunities_from_cache(strategy_result)

        self.assertEqual(len(opportunities), 1)
        self.assertGreaterEqual(opportunities[0]['profit_percent'], self.engine.config.MIN_TRIANGULAR_PROFIT)
        self.assertGreaterEqual(opportunities[0]['raw_profit_percent'], 0)
        self.assertEqual(self.engine.aggressive_filter_metrics['negative_raw_filtered'], 1)
        self.assertEqual(self.engine.aggressive_filter_metrics['below_min_profit_filtered'], 1)


if __name__ == '__main__':
    unittest.main()
