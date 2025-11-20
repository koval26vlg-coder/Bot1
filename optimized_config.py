"""Оптимизированная конфигурация параметров для тестнета."""

from config import Config


class OptimizedConfig(Config):
    """Конфигурация с более агрессивными параметрами для тестовой среды."""

    @property
    def MIN_TRIANGULAR_PROFIT(self):
        """Пониженный порог прибыли для ускоренного поиска сделок в тестнете."""
        if self._min_triangular_profit_override is not None:
            return self._min_triangular_profit_override
        if self.TESTNET:
            return 0.03
        return Config.MIN_TRIANGULAR_PROFIT.fget(self)

    @property
    def UPDATE_INTERVAL(self):
        """Чаще обновляем данные в тестнете для ускорения экспериментов."""
        if self.TESTNET:
            return 3
        return Config.UPDATE_INTERVAL.fget(self)

    @property
    def TRADE_AMOUNT(self):
        """Увеличенная сумма сделки в тестнете для более заметных результатов."""
        if self.TESTNET:
            return 15
        return Config.TRADE_AMOUNT.fget(self)

    @property
    def COOLDOWN_PERIOD(self):
        """Сокращенный кулдаун в тестнете для быстрого повторного тестирования."""
        if self.TESTNET:
            return 90
        return Config.COOLDOWN_PERIOD
