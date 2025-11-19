import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from advanced_arbitrage_engine import AdvancedArbitrageEngine


class MonitorStub:
    def __init__(self):
        self.balance_snapshots = []
        self.health_checks = []

    def update_balance_snapshot(self, balance):
        self.balance_snapshots.append(balance)

    def check_balance_health(self, balance):
        self.health_checks.append(balance)

    def notify_performance(self, report):
        self.last_report = report

    def track_cooldown_violation(self, symbol):
        self.last_violation = symbol


class ExecuteArbitrageBalanceTest(unittest.TestCase):
    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = SimpleNamespace(
            TRADE_AMOUNT=50,
            TRADING_FEE=0.0006,
            COOLDOWN_PERIOD=10,
            MIN_TRIANGULAR_PROFIT=0.2
        )
        self.engine.last_arbitrage_time = {}
        self.engine.monitor = MonitorStub()
        self.engine.trade_history = []
        self.engine.triangle_stats = {
            'TEST': {
                'executed_trades': 0,
                'total_profit': 0,
                'last_execution': None,
                'success_rate': 0,
                'failures': 0,
                'opportunities_found': 0
            }
        }

    def _make_opportunity(self):
        return {
            'triangle_name': 'TEST',
            'profit_percent': 1.0,
            'market_conditions': 'normal',
            'execution_path': [],
            'direction': 'forward',
            'symbols': [],
            'prices': {},
            'type': 'triangular'
        }

    def test_execute_arbitrage_declines_with_zero_balance(self):
        self.engine.client = SimpleNamespace(get_balance=lambda coin: {'available': 0.0, 'total': 0.0, 'coin': coin})
        self.engine.execute_triangular_arbitrage = Mock()
        opportunity = self._make_opportunity()

        result = self.engine.execute_arbitrage(opportunity)

        self.assertFalse(result)
        self.assertEqual(self.engine.monitor.health_checks[-1], 0.0)
        self.engine.execute_triangular_arbitrage.assert_not_called()

    def test_calculate_trade_receives_real_balance(self):
        expected_balance = 42.5
        self.engine.client = SimpleNamespace(get_balance=lambda coin: {'available': expected_balance, 'total': expected_balance, 'coin': coin})
        trade_plan = {'initial_amount': 10, 'estimated_profit_usdt': 0.1}
        calculate_mock = Mock(return_value=trade_plan)
        self.engine.calculate_advanced_trade = calculate_mock
        self.engine.execute_triangular_arbitrage = Mock(return_value=True)
        opportunity = self._make_opportunity()

        self.engine.execute_arbitrage(opportunity)

        calculate_mock.assert_called_once_with(opportunity, expected_balance)
        self.assertIn(expected_balance, self.engine.monitor.balance_snapshots)
        self.engine.execute_triangular_arbitrage.assert_called_once_with(opportunity, trade_plan)


if __name__ == '__main__':
    unittest.main()
