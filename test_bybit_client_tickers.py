import unittest
from unittest.mock import MagicMock, patch

from bybit_client import BybitClient


class BatchedTickerFetchTest(unittest.TestCase):
    """Проверяем, что get_tickers использует bulk и укладывается во временной бюджет."""

    def setUp(self):
        patcher = patch('bybit_client.BybitClient._create_session')
        self.addCleanup(patcher.stop)
        self.mock_create_session = patcher.start()
        self.session_mock = MagicMock()
        self.mock_create_session.return_value = self.session_mock
        self.client = BybitClient()

    @patch('bybit_client.time.time')
    def test_bulk_fetches_all_symbols_faster_than_two_seconds(self, mock_time):
        symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']

        mock_time.side_effect = [100.0, 101.5] + [101.5] * 10

        bulk_response = {
            'retCode': 0,
            'result': {
                'list': [
                    {'symbol': 'BTCUSDT', 'bid1Price': '50000', 'ask1Price': '50010', 'lastPrice': '50005', 'time': 1},
                    {'symbol': 'ETHUSDT', 'bid1Price': '3000', 'ask1Price': '3001', 'lastPrice': '3000.5', 'time': 1},
                ]
            }
        }

        fallback_response = {
            'retCode': 0,
            'result': {
                'list': [
                    {'symbol': 'XRPUSDT', 'bid1Price': '0.5', 'ask1Price': '0.51', 'lastPrice': '0.505', 'time': 1}
                ]
            }
        }

        def fake_get_tickers(**kwargs):
            if kwargs.get('symbol') == 'XRPUSDT':
                return fallback_response
            return bulk_response

        self.session_mock.get_tickers.side_effect = fake_get_tickers

        with self.assertLogs('bybit_client', level='INFO') as log_capture:
            tickers = self.client.get_tickers(symbols)

        self.assertEqual(len(tickers), 3)
        self.assertTrue(any('меньше 2 секунд' in message for message in log_capture.output))
        self.assertEqual(self.session_mock.get_tickers.call_count, 2)


if __name__ == '__main__':
    unittest.main()
