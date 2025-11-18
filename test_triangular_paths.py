import unittest

from advanced_arbitrage_engine import AdvancedArbitrageEngine
from config import Config


class TriangularPathFlowTest(unittest.TestCase):
    """Проверяем, что оба маршрута ETH/BTC/USDT остаются валидными."""

    def setUp(self):
        # Создаем экземпляр без побочных эффектов конструктора
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = Config()

    def test_eth_btc_usdt_routes_return_realistic_profit(self):
        prices = {
            'BTCUSDT': {'bid': 26005.0, 'ask': 26010.0},
            'ETHBTC': {'bid': 0.06155, 'ask': 0.06158},
            'ETHUSDT': {'bid': 1610.0, 'ask': 1610.5},
            'BTCETH': {'bid': 16.0, 'ask': 16.05}
        }

        triangle_btc_eth = {
            'name': 'USDT-BTC-ETH',
            'legs': ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
            'base_currency': 'USDT'
        }
        triangle_eth_btc = {
            'name': 'USDT-ETH-BTC',
            'legs': ['ETHUSDT', 'BTCETH', 'BTCUSDT'],
            'base_currency': 'USDT'
        }

        dir_one = self.engine._calculate_direction(
            {symbol: prices[symbol] for symbol in triangle_btc_eth['legs']},
            triangle_btc_eth,
            1
        )
        dir_two = self.engine._calculate_direction(
            {symbol: prices[symbol] for symbol in triangle_eth_btc['legs']},
            triangle_eth_btc,
            1
        )

        self.assertGreater(dir_one['profit_percent'], 0)
        self.assertLess(dir_one['profit_percent'], 1)
        self.assertGreater(dir_two['profit_percent'], 0)
        self.assertLess(dir_two['profit_percent'], 1)


if __name__ == '__main__':
    unittest.main()