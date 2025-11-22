import unittest
from datetime import datetime, timedelta, timezone

from config import Config
from indicator_strategies import StrategyManager


class PatternRecognitionStrategyTest(unittest.TestCase):
    """Проверяем распознавание базовых свечных паттернов."""

    def setUp(self):
        self.manager = StrategyManager(Config())
        self.base_time = datetime.now(timezone.utc)

    def _build_rows(self, candles):
        rows = []
        for idx, (o, h, l, c, v) in enumerate(candles):
            rows.append({
                'timestamp': self.base_time + timedelta(minutes=idx),
                'symbol': 'BTCUSDT',
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v,
            })
        return rows

    def test_bullish_engulfing_triggers_long(self):
        """Бычье поглощение с объёмом должно давать длинный сигнал."""

        candles = [
            (100, 101, 94, 95, 1000),
            (94, 106, 93, 106, 1800),
        ]
        market_df = self._build_rows(candles)

        result = self.manager._pattern_recognition_strategy(market_df, {})

        self.assertEqual(result.name, 'pattern_recognition')
        self.assertEqual(result.signal, 'long')
        self.assertEqual(result.meta['pattern_name'], 'bullish_engulfing')
        self.assertGreater(result.confidence, 0.6)

    def test_bearish_engulfing_triggers_short(self):
        """Медвежье поглощение должно давать короткий сигнал."""

        candles = [
            (100, 112, 99, 110, 1400),
            (112, 114, 98, 99, 1700),
        ]
        market_df = self._build_rows(candles)

        result = self.manager._pattern_recognition_strategy(market_df, {})

        self.assertEqual(result.signal, 'short')
        self.assertEqual(result.meta['pattern_name'], 'bearish_engulfing')
        self.assertGreater(result.confidence, 0.4)

    def test_flag_pattern_detected(self):
        """Флаг после импульса должен определяться как бычий паттерн."""

        candles = [
            (95, 130, 94, 100, 1000),
            (100, 170, 99, 150, 1100),
            (150, 155, 149, 152.5, 900),
            (152.5, 156, 151.5, 153.5, 1300),
        ]
        market_df = self._build_rows(candles)

        result = self.manager._pattern_recognition_strategy(market_df, {})

        self.assertEqual(result.signal, 'long')
        self.assertEqual(result.meta['pattern_name'], 'bull_flag')
        self.assertGreater(result.confidence, 0.45)

    def test_neutral_when_no_pattern(self):
        """При отсутствии паттернов стратегия возвращает нейтральный сигнал."""

        candles = [
            (100, 101, 99, 100, 1000),
            (101, 102, 100, 101, 980),
            (102, 103, 101, 102, 990),
        ]
        market_df = self._build_rows(candles)

        result = self.manager._pattern_recognition_strategy(market_df, {})

        self.assertEqual(result.signal, 'neutral')
        self.assertLess(result.confidence, 0.4)


if __name__ == '__main__':
    unittest.main()
