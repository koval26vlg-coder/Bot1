import unittest
from unittest.mock import patch

import advanced_arbitrage_engine
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
            'ETHUSDT': {'bid': 1610.0, 'ask': 1610.5}
        }

        triangle_btc_eth = {
            'name': 'USDT-BTC-ETH',
            'legs': ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
            'base_currency': 'USDT'
        }
        triangle_eth_btc = {
            'name': 'USDT-ETH-BTC',
            'legs': ['ETHUSDT', 'ETHBTC', 'BTCUSDT'],
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
            3
        )

        self.assertGreater(dir_one['profit_percent'], 0)
        self.assertLess(dir_one['profit_percent'], 1)
        self.assertGreater(dir_two['profit_percent'], -5)
        self.assertLess(dir_two['profit_percent'], 5)
        self.assertEqual(len(dir_two['path']), 3)


class ConfigTriangularPairsBaseCurrencyTest(unittest.TestCase):
    """Проверяем, что все маршруты начинаются и заканчиваются на поддерживаемой базе."""

    @patch.object(Config, '_fetch_market_symbols', return_value=set())
    def test_all_triangles_use_usdt_base(self, _mock_symbols):
        config = Config()
        triangles = config.TRIANGULAR_PAIRS

        self.assertGreater(len(triangles), 0)
        for triangle in triangles:
            self.assertEqual(triangle['base_currency'], 'USDT')
            self.assertTrue(triangle['legs'][0].endswith('USDT'))
            self.assertTrue(triangle['legs'][-1].endswith('USDT'))


class ExoticFiatSymbolPathTest(unittest.TestCase):
    """Проверяем построение маршрутов для тикеров с фиатными котировками."""

    def setUp(self):
        self.engine = AdvancedArbitrageEngine.__new__(AdvancedArbitrageEngine)
        self.engine.config = Config()
        self.engine._quote_suffix_cache = None
        self.engine.config._available_cross_map_cache = {
            'SOL': {'USDT', 'BRL'},
            'USDC': {'USDT', 'BRL'},
            'BRL': {'USDT'},
            'WIF': {'USDT', 'EUR'},
            'EUR': {'USDT'}
        }

    def _assert_exotic_path(self, legs, base_currency='USDT'):
        with patch.object(advanced_arbitrage_engine.logger, 'warning') as mock_warning:
            path = self.engine._build_universal_path(legs, base_currency, 'Test', 1)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        mock_warning.assert_not_called()

    def test_solbrl_path(self):
        self._assert_exotic_path(['SOLUSDT', 'SOLBRL', 'BRLUSDT'])

    def test_usdcbrl_path(self):
        self._assert_exotic_path(['USDCUSDT', 'USDCBRL', 'BRLUSDT'])

    def test_wifeur_path(self):
        self._assert_exotic_path(['WIFUSDT', 'WIFEUR', 'EURUSDT'])


if __name__ == '__main__':
    unittest.main()
