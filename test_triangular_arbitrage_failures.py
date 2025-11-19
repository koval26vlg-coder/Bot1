import unittest
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine


class TriangularArbitrageFailureTest(unittest.TestCase):
    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.client = SimpleNamespace(get_tickers=lambda symbols: {})
        self.engine.real_trader = SimpleNamespace(execute_arbitrage_trade=lambda plan: None)
        self.engine._validate_opportunity_still_exists = lambda opportunity, tickers: True
        self.engine.triangle_stats = {
            'TEST': {
                'opportunities_found': 0,
                'executed_trades': 0,
                'failures': 0,
                'total_profit': 0,
                'last_execution': None,
                'success_rate': 0,
            }
        }

    def test_failure_increments_counter(self):
        opportunity = {
            'triangle_name': 'TEST',
            'symbols': ['BTCUSDT', 'ETHUSDT', 'ETHBTC'],
            'direction': 'forward',
            'profit_percent': 0.8,
            'execution_path': [],
            'market_conditions': 'normal',
            'type': 'triangular',
        }
        trade_plan = {
            'estimated_profit_usdt': 0.0,
            'initial_amount': 10,
        }

        result = self.engine.execute_triangular_arbitrage(opportunity, trade_plan)

        self.assertFalse(result)
        self.assertEqual(self.engine.triangle_stats['TEST']['failures'], 1)
        self.assertEqual(self.engine.triangle_stats['TEST']['success_rate'], 0)


if __name__ == '__main__':
    unittest.main()
