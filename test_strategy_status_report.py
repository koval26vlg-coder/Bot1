import unittest
from types import SimpleNamespace

from advanced_arbitrage_engine import AdvancedArbitrageEngine


class StrategyManagerStub:
    def __init__(self, active, snapshot):
        self._active = active
        self._snapshot = snapshot

    def get_active_strategy_name(self):
        return self._active

    def get_strategy_snapshot(self):
        return self._snapshot


class StrategyStatusReportTest(unittest.TestCase):
    def test_returns_mode_and_active_strategy(self):
        engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        engine.config = SimpleNamespace(STRATEGY_MODE='auto')
        engine.strategy_manager = StrategyManagerStub(
            'momentum_bias',
            {'momentum_bias': {'signal': 'bull'}}
        )
        engine.last_strategy_context = {'volatility': 'low'}

        status = engine.get_strategy_status()

        self.assertEqual(status['mode'], 'auto')
        self.assertEqual(status['active'], 'momentum_bias')
        self.assertEqual(status['context'], engine.last_strategy_context)
        self.assertEqual(status['strategies'], {'momentum_bias': {'signal': 'bull'}})


if __name__ == '__main__':
    unittest.main()
