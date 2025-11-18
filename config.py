import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API настройки
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'

    @property
    def MARKET_CATEGORY(self):
        """Возвращает тип сегмента рынка в зависимости от режима"""
        return 'linear' if self.TESTNET else 'spot'
    STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'auto')
    MANUAL_STRATEGY_NAME = os.getenv('MANUAL_STRATEGY_NAME') or ''
    
    # Основные символы для мониторинга
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'XRPUSDT']
    STRATEGY_MODE = 'adaptive'
    
    @property
    def TRIANGULAR_PAIRS(self):
        """Динамическая конфигурация треугольников в зависимости от тестнета"""
        if self.TESTNET:
            # В тестнете реально доступны только пары с ETHBTC, поэтому оставляем
            # лишь те треугольники, которые можно действительно исполнить
            return [
                {
                    'name': 'USDT-BTC-ETH-USDT',
                    # USDT -> BTC (BTCUSDT), BTC -> ETH (ETHBTC), ETH -> USDT (ETHUSDT)
                    'legs': ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 1
                },
                {
                    'name': 'USDT-ETH-BTC-USDT',
                    # USDT -> ETH (ETHUSDT), ETH -> BTC (ETHBTC), BTC -> USDT (BTCUSDT)
                    'legs': ['ETHUSDT', 'ETHBTC', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 1
                }
            ]
        else:
            # Полные треугольники для основной сети
            return [
                # Основные треугольники с BTC и ETH
                {
                    'name': 'USDT-BTC-ETH-USDT',
                    'legs': ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 1
                },
                {
                    'name': 'USDT-ETH-BTC-USDT',
                    'legs': ['ETHUSDT', 'ETHBTC', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 1
                },
                
                # Треугольники с BNB
                {
                    'name': 'USDT-BNB-BTC',
                    'legs': ['BNBUSDT', 'BTCBNB', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 2
                },
                {
                    'name': 'USDT-BNB-ETH',
                    'legs': ['BNBUSDT', 'ETHBNB', 'ETHUSDT'],
                    'base_currency': 'USDT', 
                    'priority': 2
                },
                
                # Треугольники с SOL
                {
                    'name': 'USDT-SOL-BTC',
                    'legs': ['SOLUSDT', 'BTCSOL', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 2
                },
                {
                    'name': 'USDT-SOL-ETH',
                    'legs': ['SOLUSDT', 'ETHSOL', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 2
                },
                
                # Треугольники с ADA
                {
                    'name': 'USDT-ADA-BTC',
                    'legs': ['ADAUSDT', 'BTCADA', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                {
                    'name': 'USDT-ADA-ETH',
                    'legs': ['ADAUSDT', 'ETHADA', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                
                # Дополнительные треугольники
                {
                    'name': 'USDT-DOT-BTC',
                    'legs': ['DOTUSDT', 'BTCDOT', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                {
                    'name': 'USDT-LINK-ETH',
                    'legs': ['LINKUSDT', 'ETHLINK', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                {
                    'name': 'USDT-AVAX-BTC',
                    'legs': ['AVAXUSDT', 'BTCAVAX', 'BTCUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                {
                    'name': 'USDT-XRP-ETH',
                    'legs': ['XRPUSDT', 'ETHXRP', 'ETHUSDT'],
                    'base_currency': 'USDT',
                    'priority': 3
                },
                
                # Сложные треугольники с 3 альткойнами
                {
                    'name': 'USDT-BNB-SOL',
                    'legs': ['BNBUSDT', 'SOLBNB', 'SOLUSDT'],
                    'base_currency': 'USDT',
                    'priority': 4
                },
                {
                    'name': 'USDT-ADA-DOT',
                    'legs': ['ADAUSDT', 'DOTADA', 'DOTUSDT'],
                    'base_currency': 'USDT',
                    'priority': 4
                }
            ]
    
    @property 
    def MIN_TRIANGULAR_PROFIT(self):
        """Динамический порог прибыли для тестнета"""
        return 0.0 if self.TESTNET else 0.25
    
    @property
    def UPDATE_INTERVAL(self):
        """Интервал обновления в зависимости от режима"""
        return 5 if self.TESTNET else 2  # Увеличиваем интервал для тестнета
    
    # Параметры арбитража (базовые значения, могут переопределяться свойствами)
    _MIN_PROFIT_PERCENT = 0.15  # Более агрессивный порог
    _TRADE_AMOUNT = 10  # Увеличим сумму для тестов
    
    # Настройки риска
    MAX_TRADE_PERCENT = 8
    MAX_DAILY_TRADES = 100
    MAX_LOSS_PERCENT = 0.8
    
    # Комиссии (можно настроить под разные биржи)
    TRADING_FEE = 0.001  # 0.1% комиссия за сделку
    WITHDRAWAL_FEE = 0.0  # Комиссия на вывод (не учитываем для треугольного)
    
    # Настройки логгирования
    LOG_FILE = 'triangular_arbitrage_bot.log'
    LOG_LEVEL = 'INFO'
    
    # Настройки кулдауна
    COOLDOWN_PERIOD = 180  # 3 минуты между сделками для одного треугольника
    
    # Фильтрация аномальных спредов
    @property
    def MAX_SPREAD_PERCENT(self):
        """Фильтр спреда: в тестнете позволяем огромные расхождения по ETHBTC"""
        return 1000 if self.TESTNET else 10
    
    # Дополнительные настройки для треугольного арбитража
    MAX_TRIANGLE_EXECUTION_TIME = 30  # Максимальное время выполнения треугольника (секунды)
    MIN_LIQUIDITY = 1000  # Минимальная ликвидность для торговли (USDT)

    @property
    def MARKET_CATEGORY(self):
        """Возвращает сегмент рынка для запросов к Bybit"""
        return "linear" if self.TESTNET else "spot"

    @property
    def MIN_PROFIT_PERCENT(self):
        """Порог прибыли для простого арбитража"""
        return self._MIN_PROFIT_PERCENT

    @property
    def TRADE_AMOUNT(self):
        """Сумма для торговли"""
        return self._TRADE_AMOUNT