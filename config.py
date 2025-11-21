import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    # API настройки
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'
    STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'auto')
    MANUAL_STRATEGY_NAME = os.getenv('MANUAL_STRATEGY_NAME') or ''

    # Базовый список наблюдаемых символов (фильтруется динамически)
    DEFAULT_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT',
        'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'XRPUSDT'
    ]

    # Приоритеты базовых активов для генерации треугольников
    TRIANGLE_BASES = {
        'BTC': 1,
        'ETH': 1,
        'BNB': 2,
        'SOL': 2,
        'ADA': 3,
        'DOT': 3,
        'LINK': 3,
        'MATIC': 3,
        'AVAX': 3,
        'XRP': 3
    }

    KNOWN_QUOTES = [
        'USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB',
        # Основные фиатные валюты, встречающиеся на Bybit
        'USD', 'EUR', 'BRL', 'TRY', 'AUD', 'GBP', 'JPY', 'MXN', 'ARS', 'CHF'
    ]
    # Поддерживаемые стартовые валюты для построения треугольников
    TRADING_BASE_CURRENCIES = ['USDT']

    def __init__(self):
        self._triangular_pairs_cache = None
        self._available_symbols_cache = None
        self._available_cross_map_cache = None
        self._symbol_watchlist_cache = None
        self._symbol_details_map = {}
        self._min_triangular_profit_override = self._load_min_triangular_profit_override()
        # Настройки динамического ослабления порогов
        self._empty_cycle_relax_step = self._load_float_env(
            'EMPTY_CYCLE_RELAX_STEP',
            0.02 if not self.TESTNET else 0.05
        )
        self._empty_cycle_relax_max = self._load_float_env(
            'EMPTY_CYCLE_RELAX_MAX',
            0.25 if not self.TESTNET else 0.5
        )
        self._min_dynamic_profit_floor = self._load_float_env(
            'MIN_DYNAMIC_PROFIT_FLOOR',
            0.02 if not self.TESTNET else 0.0
        )
        self._ticker_staleness_warning = self._load_float_env(
            'TICKER_STALENESS_WARNING_SEC',
            5.0
        )

    @property
    def MARKET_CATEGORY(self):
        """Возвращает тип сегмента рынка в зависимости от режима"""
        return 'linear' if self.TESTNET else 'spot'

    @property
    def API_BASE_URL(self):
        """Базовый URL публичного API Bybit"""
        return 'https://api-testnet.bybit.com' if self.TESTNET else 'https://api.bybit.com'

    @property
    def AVAILABLE_CROSSES(self):
        """Карта доступных кроссов (BASE -> список тикеров)"""
        cross_map = self._build_available_cross_map()
        if not cross_map:
            return {}

        summary = {}
        for base, quotes in cross_map.items():
            symbols = [f"{base}{quote}" for quote in sorted(quotes)]
            summary[base] = symbols
        return summary

    @property
    def SYMBOLS(self):
        """Список наблюдаемых тикеров, согласованный с биржей"""
        if self._symbol_watchlist_cache is None:
            watchlist = set(self.DEFAULT_SYMBOLS)
            for triangle in self.TRIANGULAR_PAIRS:
                watchlist.update(triangle['legs'])
            self._symbol_watchlist_cache = sorted(watchlist)
        return self._symbol_watchlist_cache

    @property
    def TRIANGULAR_PAIRS(self):
        """Динамическая конфигурация треугольников в зависимости от доступных тикеров"""
        if self._triangular_pairs_cache is not None:
            return self._triangular_pairs_cache

        templates = self._build_triangle_templates()
        available_symbols = self._fetch_market_symbols()

        if available_symbols:
            valid_pairs = []
            for triangle in templates:
                if all(leg in available_symbols for leg in triangle['legs']):
                    valid_pairs.append(triangle)
                else:
                    missing = [leg for leg in triangle['legs'] if leg not in available_symbols]
                    logger.debug(
                        "Пропускаем треугольник %s из-за отсутствующих ног: %s",
                        triangle['name'],
                        ', '.join(missing)
                    )

            if valid_pairs:
                self._triangular_pairs_cache = valid_pairs
                return self._triangular_pairs_cache

            logger.warning(
                "API не вернул треугольники с полным покрытием. Используем шаблон как запасной вариант."
            )

        # Фолбэк (например, оффлайн среда или ошибки сети)
        self._triangular_pairs_cache = templates
        return self._triangular_pairs_cache
    
    @property
    def MIN_TRIANGULAR_PROFIT(self):
        """Динамический порог прибыли для тестнета"""
        if self._min_triangular_profit_override is not None:
            return self._min_triangular_profit_override
        return 0.05 if self.TESTNET else 0.15
    
    @property
    def UPDATE_INTERVAL(self):
        """Интервал обновления в зависимости от режима"""
        # Сокращаем интервал до 1 секунды для ускорения циклов арбитража даже в тестнете
        return 1
    
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
    SLIPPAGE_PROFIT_BUFFER = 0.02  # Запас на проскальзывание в процентах
    VOLATILITY_PROFIT_MULTIPLIER = 0.05  # Усилитель порога прибыли от волатильности
    
    # Настройки логгирования
    LOG_FILE = 'triangular_arbitrage_bot.log'
    LOG_LEVEL = 'INFO'
    # Количество выводимых котировок в снапшоте рынка
    MARKET_SNAPSHOT_SYMBOLS = int(os.getenv('MARKET_SNAPSHOT_SYMBOLS', '3'))
    
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
    ORDERBOOK_DEPTH_LEVELS = 5  # Количество уровней стакана для оценки ликвидности
    MAX_ORDERBOOK_IMPACT = 0.25  # Максимальная доля, которую может занимать наша заявка от глубины

    @property
    def MIN_PROFIT_PERCENT(self):
        """Порог прибыли для простого арбитража"""
        return self._MIN_PROFIT_PERCENT

    @property
    def TRADE_AMOUNT(self):
        """Сумма для торговли"""
        return self._TRADE_AMOUNT

    @property
    def EMPTY_CYCLE_RELAX_STEP(self):
        """Шаг снижения порога прибыли при серии пустых циклов"""
        return self._empty_cycle_relax_step

    @property
    def EMPTY_CYCLE_RELAX_MAX(self):
        """Максимальное снижение порога на фоне отсутствия сделок"""
        return self._empty_cycle_relax_max

    @property
    def MIN_DYNAMIC_PROFIT_FLOOR(self):
        """Абсолютный минимум динамического порога прибыли"""
        return self._min_dynamic_profit_floor

    @property
    def TICKER_STALENESS_WARNING_SEC(self):
        """Максимально допустимая давность котировок перед предупреждением"""
        return self._ticker_staleness_warning

    def _load_min_triangular_profit_override(self):
        """Читает переопределение порога прибыли из переменной окружения"""
        raw_value = os.getenv('MIN_TRIANGULAR_PROFIT')
        if raw_value is None:
            return None

        normalized_value = raw_value.replace(',', '.')
        try:
            profit_value = float(normalized_value)
        except ValueError:
            logger.warning(
                "Некорректное значение MIN_TRIANGULAR_PROFIT='%s'. Используем дефолтный порог.",
                raw_value
            )
            return None

        if profit_value < 0:
            logger.warning(
                "MIN_TRIANGULAR_PROFIT не может быть отрицательным. Используем дефолтный порог."
            )
            return None

        return profit_value

    def _load_float_env(self, var_name, default):
        """Унифицированное чтение числовых параметров окружения"""
        raw_value = os.getenv(var_name)
        if raw_value is None:
            return default

        normalized_value = raw_value.replace(',', '.')
        try:
            return float(normalized_value)
        except ValueError:
            logger.warning(
                "Некорректное значение %s='%s'. Используем дефолт %.4f.",
                var_name,
                raw_value,
                default
            )
            return default

    def _build_triangle_templates(self):
        """Создает список потенциальных треугольников, согласованный с реальными тикерами"""
        available_symbols = self._fetch_market_symbols()
        dynamic_templates = self._build_dynamic_triangle_templates(available_symbols)

        if dynamic_templates:
            return dynamic_templates

        # Фолбэк на статическую конфигурацию, если API недоступно или ничего не построено
        return self._build_static_triangle_templates()

    def _build_dynamic_triangle_templates(self, available_symbols):
        """Генерация всех возможных треугольников на основе доступных инструментов"""
        if not available_symbols:
            return []

        templates = []
        registered = set()
        base_candidates = self._discover_base_currencies(available_symbols)
        if self.TRADING_BASE_CURRENCIES:
            base_candidates = [
                base for base in base_candidates
                if base in self.TRADING_BASE_CURRENCIES
            ]
        if not base_candidates:
            return []

        for base_currency in base_candidates:
            connected_assets = sorted(self._collect_connected_assets(base_currency, available_symbols))
            if len(connected_assets) < 2:
                continue

            for primary_asset in connected_assets:
                if primary_asset == base_currency:
                    continue

                for secondary_asset in connected_assets:
                    if secondary_asset in {primary_asset, base_currency}:
                        continue

                    legs = self._resolve_triangle_legs(
                        base_currency,
                        primary_asset,
                        secondary_asset,
                        available_symbols
                    )
                    if not legs:
                        continue

                    triangle_name = f'{base_currency}-{primary_asset}-{secondary_asset}-{base_currency}'
                    if triangle_name in registered:
                        continue

                    priority = min(
                        self.TRIANGLE_BASES.get(primary_asset, 5),
                        self.TRIANGLE_BASES.get(secondary_asset, 5)
                    )

                    templates.append({
                        'name': triangle_name,
                        'legs': legs,
                        'base_currency': base_currency,
                        'priority': priority
                    })
                    registered.add(triangle_name)

        return templates

    def _build_static_triangle_templates(self):
        """Резервный список треугольников на случай оффлайн режима"""
        templates = [
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
            }
        ]

        for asset, priority in self.TRIANGLE_BASES.items():
            if asset in {'BTC', 'ETH'}:
                continue

            templates.append({
                'name': f'USDT-{asset}-BTC-USDT',
                'legs': [f'{asset}USDT', f'{asset}BTC', 'BTCUSDT'],
                'base_currency': 'USDT',
                'priority': priority
            })
            templates.append({
                'name': f'USDT-{asset}-ETH-USDT',
                'legs': [f'{asset}USDT', f'{asset}ETH', 'ETHUSDT'],
                'base_currency': 'USDT',
                'priority': priority
            })

        return templates

    def _discover_base_currencies(self, available_symbols):
        """Определяет подходящие базовые валюты для старта треугольников"""
        currencies = set()
        for symbol in available_symbols:
            base, quote = self._split_symbol(symbol)
            if base:
                currencies.add(base)
            if quote:
                currencies.add(quote)

        ordered = []
        if self.TRADING_BASE_CURRENCIES:
            for currency in self.TRADING_BASE_CURRENCIES:
                if currency in currencies:
                    ordered.append(currency)
        else:
            preferred_order = [
                'USDT', 'USDC', 'BTC', 'ETH', 'BNB', 'EUR', 'USD', 'DAI'
            ]

            for currency in preferred_order:
                if currency in currencies:
                    ordered.append(currency)

            for currency in sorted(currencies):
                if currency not in ordered:
                    ordered.append(currency)

        result = []
        for currency in ordered:
            connected = self._collect_connected_assets(currency, available_symbols)
            if len(connected) >= 2:
                result.append(currency)

        return result

    def _collect_connected_assets(self, anchor_currency, available_symbols):
        """Находит все валюты, которые можно напрямую обменять на указанную"""
        connected = set()
        for symbol in available_symbols:
            base, quote = self._split_symbol(symbol)
            if base == anchor_currency and quote:
                connected.add(quote)
            elif quote == anchor_currency and base:
                connected.add(base)

        connected.discard(anchor_currency)
        return connected

    def _resolve_triangle_legs(self, base_currency, primary_asset, secondary_asset, available_symbols):
        """Подбирает реальные тикеры для треугольника Base -> Primary -> Secondary -> Base"""
        legs = []
        currency_pairs = [
            (primary_asset, base_currency),
            (secondary_asset, primary_asset),
            (secondary_asset, base_currency)
        ]

        for base, quote in currency_pairs:
            symbol = self._resolve_market_symbol(base, quote, available_symbols)
            if not symbol:
                return None
            legs.append(symbol)

        return legs

    def _resolve_market_symbol(self, base_currency, quote_currency, available_symbols):
        """Находит фактический тикер между двумя валютами в любой ориентации"""
        base_currency = base_currency.upper()
        quote_currency = quote_currency.upper()
        direct = f"{base_currency}{quote_currency}"
        reverse = f"{quote_currency}{base_currency}"

        if direct in available_symbols:
            return direct
        if reverse in available_symbols:
            return reverse

        return None

    def _fetch_market_symbols(self):
        """Получает доступные тикеры через публичный REST Bybit"""
        if self._available_symbols_cache is not None:
            return self._available_symbols_cache

        url = f"{self.API_BASE_URL}/v5/market/instruments-info"
        params = {'category': self.MARKET_CATEGORY}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('retCode') == 0:
                self._symbol_details_map = {}
                symbols = set()

                for item in data.get('result', {}).get('list', []):
                    symbol = item.get('symbol')
                    base_coin = item.get('baseCoin')
                    quote_coin = item.get('quoteCoin')

                    if not symbol:
                        continue

                    symbols.add(symbol)
                    if base_coin and quote_coin:
                        self._symbol_details_map[symbol] = (base_coin.upper(), quote_coin.upper())

                self._available_symbols_cache = symbols
                logger.info(
                    "Получено %s тикеров для категории %s", len(symbols), self.MARKET_CATEGORY
                )
                return self._available_symbols_cache

            logger.warning(
                "REST ответил кодом %s: %s", data.get('retCode'), data.get('retMsg')
            )
        except requests.RequestException as exc:
            logger.warning(
                "Не удалось получить инструменты Bybit (%s): %s", self.MARKET_CATEGORY, exc
            )

        self._available_symbols_cache = set()
        return self._available_symbols_cache

    def _build_available_cross_map(self):
        """Формирует карту доступных кроссов BASE -> QUOTE"""
        if self._available_cross_map_cache is not None:
            return self._available_cross_map_cache

        symbols = self._fetch_market_symbols()
        cross_map = {}
        for symbol in symbols:
            base, quote = self._split_symbol(symbol)
            if not base or not quote:
                continue
            cross_map.setdefault(base, set()).add(quote)

        self._available_cross_map_cache = cross_map
        return self._available_cross_map_cache

    def _split_symbol(self, symbol):
        """Разделяет тикер на базовую и котируемую валюты"""
        normalized_symbol = symbol.upper()

        # Проверяем кэш деталей тикеров, если ранее уже разобрали тикер
        if normalized_symbol in self._symbol_details_map:
            base_coin, quote_coin = self._symbol_details_map[normalized_symbol]
            if base_coin and quote_coin:
                return base_coin, quote_coin

        # Перебираем известные котируемые валюты по длине для корректного парсинга составных суффиксов
        for quote in sorted(self.KNOWN_QUOTES, key=len, reverse=True):
            if normalized_symbol.endswith(quote) and len(normalized_symbol) > len(quote):
                base_coin = normalized_symbol[:-len(quote)]
                # Требуем минимум два символа в базовой валюте
                if len(base_coin) >= 2:
                    self._symbol_details_map[normalized_symbol] = (base_coin, quote)
                    return base_coin, quote

        # Отдельно проверяем популярные котируемые валюты, которые могут не попасть в KNOWN_QUOTES
        popular_quotes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
        for quote in popular_quotes:
            if normalized_symbol.endswith(quote) and len(normalized_symbol) > len(quote):
                base_coin = normalized_symbol[:-len(quote)]
                if len(base_coin) >= 2:
                    self._symbol_details_map[normalized_symbol] = (base_coin, quote)
                    return base_coin, quote

        # Резервное разбиение пополам используется только когда других вариантов нет
        midpoint = len(normalized_symbol) // 2
        base_coin, quote_coin = normalized_symbol[:midpoint], normalized_symbol[midpoint:]
        if len(base_coin) >= 2 and len(quote_coin) >= 1:
            self._symbol_details_map[normalized_symbol] = (base_coin, quote_coin)
            return base_coin, quote_coin

        return None, None
