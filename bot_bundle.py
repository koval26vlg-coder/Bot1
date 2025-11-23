"""Монолитный модуль, объединяющий ключевые компоненты арбитражного бота."""

from __future__ import annotations

import sys as _sys

# Совместимость с историческими путями модулей при объединении в один файл.
for _alias in [
    'config',
    'optimized_config',
    'math_stats',
    'performance_optimizer',
    'indicator_strategies',
    'monitoring',
    'visualization',
    'bybit_client',
    'advanced_arbitrage_engine',
    'advanced_bot',
    'real_trading',
]:
    _sys.modules.setdefault(_alias, _sys.modules[__name__])

# ==== Начало config.py ====
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
            available_symbols = list(self._fetch_market_symbols())
            watchlist = list(available_symbols)

            triangle_legs = set()
            for triangle in self.TRIANGULAR_PAIRS:
                triangle_legs.update(triangle['legs'])

            for leg in sorted(triangle_legs):
                if leg not in watchlist:
                    watchlist.append(leg)

            self._symbol_watchlist_cache = watchlist
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
        """Порог прибыли для треугольного арбитража (приоритет у MIN_TRIANGULAR_PROFIT)."""
        if self._min_triangular_profit_override is not None:
            return self._min_triangular_profit_override
        # Базовый дефолт: для тестнета используем более консервативные 0.10,
        # для продакшена повышаем порог до 0.18+ чтобы отсеивать шум.
        return 0.10 if self.TESTNET else 0.18
    
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
                detailed_symbols = {}
                market_entries = []

                for item in data.get('result', {}).get('list', []):
                    symbol = item.get('symbol')
                    base_coin = item.get('baseCoin')
                    quote_coin = item.get('quoteCoin')
                    turnover_raw = item.get('turnover24h')

                    if not symbol:
                        continue

                    if base_coin and quote_coin:
                        detailed_symbols[symbol] = (base_coin.upper(), quote_coin.upper())

                    try:
                        turnover_value = float(turnover_raw) if turnover_raw is not None else 0.0
                    except (TypeError, ValueError):
                        turnover_value = 0.0

                    market_entries.append((symbol, turnover_value))

                market_entries.sort(key=lambda entry: entry[1], reverse=True)
                top_entries = market_entries[:20]

                self._available_symbols_cache = [symbol for symbol, _ in top_entries]
                self._symbol_details_map = {
                    symbol: detailed_symbols.get(symbol) for symbol, _ in top_entries
                    if detailed_symbols.get(symbol)
                }
                logger.info(
                    "Получено %s тикеров для категории %s (отсортировано по обороту)",
                    len(self._available_symbols_cache),
                    self.MARKET_CATEGORY
                )
                return self._available_symbols_cache

            logger.warning(
                "REST ответил кодом %s: %s", data.get('retCode'), data.get('retMsg')
            )
        except requests.RequestException as exc:
            logger.warning(
                "Не удалось получить инструменты Bybit (%s): %s", self.MARKET_CATEGORY, exc
            )

        self._available_symbols_cache = []
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

# ==== Конец config.py ====

# ==== Начало optimized_config.py ====
"""Оптимизированная конфигурация параметров для тестнета."""

from config import Config


class OptimizedConfig(Config):
    """Конфигурация с более агрессивными параметрами для тестовой среды."""

    # Ограничиваем количество треугольников для ускоренного режима поиска
    ACCELERATED_TRIANGLE_LIMIT = 50

    @property
    def MIN_TRIANGULAR_PROFIT(self):
        """Пониженный порог прибыли для ускоренного поиска сделок в тестнете."""
        if self._min_triangular_profit_override is not None:
            return self._min_triangular_profit_override
        if self.TESTNET:
            return 0.01
        return 0.05

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

# ==== Конец optimized_config.py ====

# ==== Начало math_stats.py ====
"""Вспомогательные функции для статистических расчётов без numpy/pandas."""

from statistics import fmean
from typing import Iterable, Optional


def mean(values: Iterable[float]) -> float:
    """Вычисляет среднее значение с защитой от пустых коллекций."""
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return 0.0
    return float(fmean(filtered))


def rolling_mean(values: Iterable[float], window: int, min_periods: Optional[int] = None) -> Optional[float]:
    """Возвращает среднее для последних *window* значений либо None, если данных мало."""
    if window <= 0:
        raise ValueError("Размер окна должен быть положительным")

    min_required = min_periods or window
    data = [float(v) for v in values if v is not None]
    if len(data) < min_required:
        return None

    recent = data[-window:]
    if not recent:
        return None
    return float(fmean(recent))
# ==== Конец math_stats.py ====

# ==== Начало performance_optimizer.py ====
"""Утилита для приоритизации и быстрой фильтрации треугольников по ликвидности."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional


class PerformanceOptimizer:
    """Строит оптимизированный список треугольников с учетом ликвидности и приоритетов."""

    def __init__(self, config):
        # Конфигурация передается снаружи, чтобы использовать существующие настройки.
        self.config = config
        # Базовые валюты с приоритетом для арбитража.
        self._preferred_bases = {"USDT", "USDC"}
        # Ключевые активы, которые стоит обрабатывать первыми.
        self._core_assets = {"BTC", "ETH", "BNB"}

    def update_config(self, config):
        """Обновляет конфигурацию оптимизатора без пересоздания объекта."""
        self.config = config

    def get_optimized_triangles(
        self,
        tickers: Optional[Dict[str, Dict[str, float]]] = None,
        max_count: int = 50,
    ) -> List[Dict]:
        """Возвращает отсортированный и при необходимости отфильтрованный список треугольников."""

        candidates = list(self.config.TRIANGULAR_PAIRS)
        prioritized = sorted(candidates, key=self._triangle_sort_key)

        if tickers:
            prioritized = self.parallel_check_liquidity(prioritized, tickers)

        return prioritized[:max_count]

    def _triangle_sort_key(self, triangle: Dict) -> tuple:
        """Формирует ключ сортировки с учетом базовой валюты, активов и приоритета."""

        base_currency = triangle.get("base_currency", "")
        base_score = 0 if base_currency in self._preferred_bases else 1

        # Чем больше ключевых активов в треугольнике, тем выше его приоритет.
        core_hits = sum(
            1
            for leg in triangle.get("legs", [])
            for asset in self._core_assets
            if leg.startswith(asset) or leg.endswith(asset)
        )
        asset_score = -core_hits  # Большее количество совпадений уменьшает ключ сортировки.

        priority_score = triangle.get("priority", 999)

        return (base_score, asset_score, priority_score, triangle.get("name", ""))

    def _quick_liquidity_check(
        self,
        triangle: Dict,
        tickers: Dict[str, Dict[str, float]],
        max_spread_percent: float = 5.0,
    ) -> bool:
        """Быстрая проверка ликвидности по верхним котировкам и спреду."""

        for symbol in triangle.get("legs", []):
            ticker = tickers.get(symbol)
            if not ticker:
                return False

            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if not bid or not ask or bid <= 0 or ask <= 0:
                return False

            spread = ((ask - bid) / bid) * 100
            if spread > max_spread_percent:
                return False

        return True

    def parallel_check_liquidity(
        self,
        triangles: Iterable[Dict],
        tickers: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        """Параллельная фильтрация треугольников по результатам быстрой проверки ликвидности."""

        triangles_list = list(triangles)
        filtered: List[Dict] = []
        # Используем ограниченный пул потоков, чтобы не перегружать систему.
        max_workers = min(8, len(triangles_list) or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_triangle = {
                executor.submit(self._quick_liquidity_check, triangle, tickers): triangle
                for triangle in triangles_list
            }
            for future in future_to_triangle:
                if future.result():
                    filtered.append(future_to_triangle[future])

        return filtered

# ==== Конец performance_optimizer.py ====

# ==== Начало indicator_strategies.py ====
"""Простые индикаторные стратегии для определения рыночного контекста."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

# Используем функции из реального локального модуля math_stats вместо utils.math_stats
from math_stats import mean



@dataclass
class StrategyResult:
    """Результат работы стратегии."""

    name: str
    signal: str
    score: float
    confidence: float
    meta: Optional[Dict[str, float]] = None


class StrategyManager:
    """Менеджер стратегий, который выбирает лучший результат по скору."""

    def __init__(self, config):
        self.config = config
        self.active_strategy = 'volatility_adaptive'
        self._strategies: Dict[str, Callable[[Sequence[dict], Dict[str, float]], StrategyResult]] = {
            'volatility_adaptive': self._volatility_adaptive_strategy,
            'momentum_bias': self._momentum_bias_strategy,
            'multi_indicator': self._multi_indicator_strategy,
            'orderbook_balance': self._orderbook_balance_strategy,
            'pattern_recognition': self._pattern_recognition_strategy,
        }

    def update_config(self, config):
        """Обновляет конфигурацию без пересоздания менеджера."""
        self.config = config

    def evaluate(self, market_df: Sequence[dict], context: Dict[str, float]) -> Optional[StrategyResult]:
        best_result: Optional[StrategyResult] = None
        for name, strategy in self._strategies.items():
            try:
                result = strategy(market_df, context)
            except Exception:
                result = None

            if result and (best_result is None or result.score > best_result.score):
                best_result = result

        if best_result:
            self.active_strategy = best_result.name

        return best_result

    def get_active_strategy_name(self) -> str:
        return self.active_strategy

    def get_strategy_snapshot(self) -> Dict[str, Dict[str, str]]:
        snapshot = {}
        for name in self._strategies:
            snapshot[name] = {
                'name': name,
                'description': self._strategies[name].__doc__ or ''
            }
        return snapshot

    def _volatility_adaptive_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Корректирует агрессивность торговли в зависимости от волатильности."""

        volatility = context.get('volatility', 0.0)
        liquidity = context.get('liquidity', 0.0)
        spread = context.get('average_spread_percent', 0.0)

        if spread > 1.0:
            signal = 'reduce_risk'
            score = max(0.1, 1.8 - spread)
        elif volatility > 1.5:
            signal = 'reduce_risk'
            score = max(0.1, 2.5 - volatility)
        elif volatility < 0.3 and liquidity > 0:
            signal = 'increase_risk'
            score = 1.5 + min(0.5, liquidity / 100000)
        else:
            signal = 'neutral'
            score = 1.0

        confidence_anchor = abs(volatility - 1.0) / 2
        confidence_spread = min(1.0, max(0.0, (1.0 - spread)))
        confidence = float(min(1.0, (confidence_anchor + confidence_spread) / 2))
        return StrategyResult(
            name='volatility_adaptive',
            signal=signal,
            score=float(score),
            confidence=confidence,
        )

    def _orderbook_balance_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Использует средний спред и дисбаланс стакана для оценки риска."""

        spread = context.get('average_spread_percent', 0.0)
        imbalance = context.get('orderbook_imbalance', 0.0)

        if spread > 1.2:
            signal = 'reduce_risk'
            base_score = max(0.1, 2 - spread)
        elif abs(imbalance) > 0.25 and spread < 0.6:
            signal = 'long_bias' if imbalance > 0 else 'short_bias'
            base_score = 1.5 + abs(imbalance)
        elif spread < 0.25:
            signal = 'increase_risk'
            base_score = 1.2 + (0.3 - spread)
        else:
            signal = 'neutral'
            base_score = 1.0

        confidence = float(min(1.0, max(0.0, (1 - spread)) + abs(imbalance)))
        return StrategyResult(
            name='orderbook_balance',
            signal=signal,
            score=float(base_score),
            confidence=confidence,
            meta={
                'average_spread_percent': float(spread),
                'orderbook_imbalance': float(imbalance),
            },
        )

    def _momentum_bias_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Оценивает импульс рынка и сама контролирует достаточность выборки."""

        closes_by_symbol = defaultdict(list)
        for row in market_df:
            close_value = row.get('close')
            if close_value is None:
                continue
            closes_by_symbol[row.get('symbol', 'default')].append(close_value)

        closes: Sequence[float] = []
        if closes_by_symbol:
            closes = max(closes_by_symbol.values(), key=len)

        min_required = 12
        if len(closes) < min_required:
            readiness = len(closes) / min_required if min_required else 0
            return StrategyResult(
                name='momentum_bias',
                signal='wait_for_data',
                score=float(readiness * 0.5),
                confidence=float(min(0.3, readiness / 2)),
            )

        short_window = min(10, len(closes))
        long_window = min(25, len(closes))

        short_ma = mean(closes[-short_window:])
        long_ma = mean(closes[-long_window:])

        if long_ma == 0:
            return StrategyResult(
                name='momentum_bias',
                signal='neutral',
                score=0.0,
                confidence=0.0,
            )

        momentum = float(((short_ma - long_ma) / long_ma) * 100)

        if momentum > 0.1:
            signal = 'long_bias'
        elif momentum < -0.1:
            signal = 'short_bias'
        else:
            signal = 'neutral'

        score = abs(momentum)
        confidence = float(min(1.0, abs(momentum) / 5))

        return StrategyResult(
            name='momentum_bias',
            signal=signal,
            score=score,
            confidence=confidence,
        )

    def _multi_indicator_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Комбинирует RSI, EMA и ATR для оценки импульса и риска."""

        rows_by_symbol = defaultdict(list)
        for row in market_df:
            symbol = row.get('symbol', 'default')
            rows_by_symbol[symbol].append(row)

        if not rows_by_symbol:
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        # Берем самый заполненный набор, чтобы избежать смены символов на коротких сериях
        symbol_rows = max(rows_by_symbol.values(), key=len)
        closes = [float(r['close']) for r in symbol_rows if r.get('close') is not None]
        highs = [float(r['high']) for r in symbol_rows if r.get('high') is not None]
        lows = [float(r['low']) for r in symbol_rows if r.get('low') is not None]
        opens = [float(r['open']) for r in symbol_rows if r.get('open') is not None]

        min_required = 15
        if (
            len(closes) < min_required
            or len(highs) < min_required
            or len(lows) < min_required
            or len(opens) < min_required
        ):
            readiness = len(closes) / min_required if min_required else 0
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=float(readiness * 0.4),
                confidence=float(min(0.25, readiness / 3)),
                meta={},
            )

        rsi = self._calculate_rsi(closes)
        short_ema = self._calculate_ema(closes, period=9)
        long_ema = self._calculate_ema(closes, period=21)
        atr = self._calculate_atr(highs, lows, closes)

        if long_ema is None or short_ema is None or rsi is None or atr is None:
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        ema_diff = short_ema - long_ema
        trend_strength = abs(ema_diff) / long_ema * 100 if long_ema else 0.0
        momentum_component = abs(rsi - 50) / 50

        if rsi > 60 and short_ema > long_ema:
            signal = 'long'
        elif rsi < 40 and short_ema < long_ema:
            signal = 'short'
        else:
            signal = 'flat'

        score = float(trend_strength + momentum_component * 2)
        confidence = float(min(1.0, (trend_strength / 10) + (momentum_component / 2)))

        pattern = self._detect_candlestick_pattern(opens, highs, lows, closes)
        if pattern['bias'] == 'bullish':
            # Поддерживаем бычьи паттерны повышением скора и уверенностью
            score += pattern['strength']
            confidence = float(min(1.0, confidence + pattern['strength'] / 2))
            if signal == 'flat':
                signal = 'long_bias'
            elif signal == 'short':
                signal = 'flat'
        elif pattern['bias'] == 'bearish':
            # Медвежьи паттерны уменьшают уверенность и агрессивность
            score += pattern['strength']
            confidence = float(max(0.0, confidence - pattern['strength'] / 3))
            if signal == 'flat':
                signal = 'short_bias'
            elif signal == 'long':
                signal = 'flat'

        meta = {
            'rsi': float(rsi),
            'short_ema': float(short_ema),
            'long_ema': float(long_ema),
            'atr': float(atr),
            'atr_percent': float((atr / closes[-1]) * 100) if closes and closes[-1] else 0.0,
            'trend_strength': float(trend_strength),
            'pattern_name': pattern['name'],
            'pattern_bias': pattern['bias'],
            'pattern_strength': pattern['strength'],
        }

        return StrategyResult(
            name='multi_indicator',
            signal=signal,
            score=score,
            confidence=confidence,
            meta=meta,
        )

    def _calculate_rsi(self, closes: Sequence[float], period: int = 14) -> Optional[float]:
        """Простой расчёт RSI без сторонних библиотек."""

        if len(closes) < period + 1:
            return None

        gains = []
        losses = []
        for prev, curr in zip(closes[-(period + 1):-1], closes[-period:]):
            change = curr - prev
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = mean(gains) if gains else 0.0
        avg_loss = mean(losses) if losses else 0.0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_ema(self, closes: Sequence[float], period: int) -> Optional[float]:
        """Экспоненциальное среднее с инициализацией от простого среднего."""

        if len(closes) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = mean(closes[:period])
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    def _calculate_atr(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> Optional[float]:
        """ATR на основе истинного диапазона."""

        if not (len(highs) == len(lows) == len(closes)):
            return None
        if len(closes) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:]
        if not recent_tr:
            return None
        return mean(recent_tr)

    def _detect_candlestick_pattern(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> Dict[str, str | float]:
        """Определяем базовые свечные паттерны без сторонних библиотек."""

        if min(len(opens), len(highs), len(lows), len(closes)) < 2:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = max(h1 - l1, 1e-9)
        range2 = max(h2 - l2, 1e-9)

        patterns = []

        # Бычье поглощение
        if c1 > o1 and c2 < o2 and o1 <= c2 and c1 >= o2:
            strength = min(0.6, (body1 + body2) / max(body2, 1e-9) * 0.1)
            patterns.append(('bullish_engulfing', 'bullish', strength))

        # Медвежье поглощение
        if c1 < o1 and c2 > o2 and o1 >= c2 and c1 <= o2:
            strength = min(0.6, (body1 + body2) / max(body2, 1e-9) * 0.1)
            patterns.append(('bearish_engulfing', 'bearish', strength))

        # Молот
        lower_shadow = o1 - l1 if o1 >= c1 else c1 - l1
        upper_shadow = h1 - max(o1, c1)
        if body1 <= range1 * 0.35 and lower_shadow >= body1 * 2 and upper_shadow <= body1:
            strength = min(0.4, lower_shadow / range1)
            patterns.append(('hammer', 'bullish', strength))

        # Падающая звезда
        if body1 <= range1 * 0.35 and upper_shadow >= body1 * 2 and lower_shadow <= body1:
            strength = min(0.4, upper_shadow / range1)
            patterns.append(('shooting_star', 'bearish', strength))

        if not patterns:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        # Берём самый сильный паттерн
        name, bias, strength = max(patterns, key=lambda item: item[2])
        return {'name': name, 'bias': bias, 'strength': float(strength)}

    def _pattern_recognition_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float],
    ) -> StrategyResult:
        """Распознаёт свечные паттерны и подтверждает их объёмами."""

        rows_by_symbol = defaultdict(list)
        for row in market_df:
            symbol = row.get('symbol', 'default')
            rows_by_symbol[symbol].append(row)

        if not rows_by_symbol:
            return StrategyResult(
                name='pattern_recognition',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        symbol_rows = max(rows_by_symbol.values(), key=len)
        opens = [float(r['open']) for r in symbol_rows if r.get('open') is not None]
        highs = [float(r['high']) for r in symbol_rows if r.get('high') is not None]
        lows = [float(r['low']) for r in symbol_rows if r.get('low') is not None]
        closes = [float(r['close']) for r in symbol_rows if r.get('close') is not None]
        volumes = [float(r['volume']) for r in symbol_rows if r.get('volume') is not None]

        if min(len(opens), len(highs), len(lows), len(closes), len(volumes)) < 2:
            return StrategyResult(
                name='pattern_recognition',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        pattern = self._detect_pattern_with_flag(opens, highs, lows, closes)

        avg_volume = mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        volume_ratio = float(volumes[-1] / avg_volume) if avg_volume else 1.0
        volume_confirmation = float(min(1.0, max(0.0, volume_ratio - 0.5)))

        signal_map = {
            'bullish': 'long',
            'bearish': 'short',
            'neutral': 'neutral',
        }
        signal = signal_map.get(pattern['bias'], 'neutral')

        pattern_strength = pattern['strength']
        if pattern_strength < 0.1:
            signal = 'neutral'

        score = float(1 + pattern_strength * 2 + volume_confirmation)
        confidence = float(min(1.0, pattern_strength * 0.7 + volume_confirmation * 0.5))

        meta = {
            'pattern_name': pattern['name'],
            'pattern_bias': pattern['bias'],
            'pattern_strength': float(pattern_strength),
            'volume_ratio': volume_ratio,
        }

        return StrategyResult(
            name='pattern_recognition',
            signal=signal,
            score=score,
            confidence=confidence,
            meta=meta,
        )

    def _detect_pattern_with_flag(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> Dict[str, str | float]:
        """Расширенный поиск паттернов, включая флаги."""

        base_pattern = self._detect_candlestick_pattern(opens, highs, lows, closes)
        candidates = [base_pattern] if base_pattern['name'] != 'none' else []

        if len(closes) >= 4:
            range_values = [h - l for h, l in zip(highs, lows)]
            impulse_up = closes[-3] > closes[-4] and closes[-2] >= closes[-3]
            impulse_down = closes[-3] < closes[-4] and closes[-2] <= closes[-3]
            compact_consolidation = max(range_values[-2:]) < min(range_values[-4:-2]) * 0.8

            if impulse_up and compact_consolidation:
                strength = min(0.5, (closes[-2] - closes[-4]) / max(closes[-4], 1e-9) * 0.3)
                candidates.append({'name': 'bull_flag', 'bias': 'bullish', 'strength': float(strength)})

            if impulse_down and compact_consolidation:
                strength = min(0.5, (closes[-4] - closes[-2]) / max(closes[-4], 1e-9) * 0.3)
                candidates.append({'name': 'bear_flag', 'bias': 'bearish', 'strength': float(strength)})

        if not candidates:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        return max(candidates, key=lambda p: p['strength'])


__all__ = ['StrategyManager', 'StrategyResult']

# ==== Конец indicator_strategies.py ====

# ==== Начало monitoring.py ====
import logging
import time
import json
import csv
import os
from datetime import datetime
import importlib.util

psutil = None
if importlib.util.find_spec('psutil') is not None:
    import psutil
from config import Config

logger = logging.getLogger(__name__)

class AdvancedMonitor:
    def __init__(self, engine):
        self.config = Config()
        self.engine = engine
        self.start_time = datetime.now()
        self.api_response_times = []
        self.system_metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io': [],
            'network_io': []
        }
        self.trade_history = []
        self.alert_thresholds = {
            'max_api_latency': 2.0,  # секунды
            'min_profit_rate': 0.8,  # 80% успешных сделок
            'max_consecutive_losses': 3,
            'min_balance': 10.0,     # USDT
            'max_cpu_usage': 95.0,   # %
            'max_memory_usage': 95.0 # %
        }
        self.cooldown_violations = 0
        self.api_errors = 0
        self.last_performance_report = None
        self._psutil_warning_logged = False
        self.last_balance_snapshot = None

    def _get_strategy_status(self):
        """Безопасно возвращает статус стратегии из движка."""
        if not self.engine or not hasattr(self.engine, 'get_strategy_status'):
            return {}

        try:
            return self.engine.get_strategy_status() or {}
        except Exception as exc:
            logger.debug(f"Не удалось получить статус стратегий: {exc}")
            return {}

    def track_api_call(self, endpoint, duration):
        """Отслеживание времени ответа API"""
        self.api_response_times.append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'duration': duration
        })
        
        # Очистка старых данных (хранить только последние 1000 записей)
        if len(self.api_response_times) > 1000:
            self.api_response_times.pop(0)
        
        # Проверка на аномальную задержку
        if duration > self.alert_thresholds['max_api_latency']:
            self._log_api_latency_alert(endpoint, duration)
    
    def _log_api_latency_alert(self, endpoint, duration):
        """Логирование алерта о высокой задержке API"""
        logger.warning(
            f"Высокая задержка API эндпоинта '{endpoint}': {duration:.2f} сек\n"
            f"Порог: {self.alert_thresholds['max_api_latency']} сек\n"
            f"Рекомендуется проверить подключение или снизить частоту запросов."
        )
    
    def track_system_metrics(self):
        """Отслеживание системных метрик"""
        if not self._ensure_psutil_available():
            return

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            self.system_metrics['cpu_percent'].append(cpu_percent)
            self.system_metrics['memory_percent'].append(memory.percent)
            self.system_metrics['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)
            self.system_metrics['network_io'].append(net_io.bytes_sent + net_io.bytes_recv)
            
            # Очистка старых данных
            for metric in self.system_metrics.values():
                if len(metric) > 1000:
                    metric.pop(0)
            
            # Проверка на высокую нагрузку системы
            if cpu_percent > self.alert_thresholds['max_cpu_usage']:
                self._log_system_load_alert('CPU', cpu_percent)
                
            if memory.percent > self.alert_thresholds['max_memory_usage']:
                self._log_system_load_alert('Memory', memory.percent)
                
        except Exception as e:
            logger.error(f"Error tracking system metrics: {str(e)}")
    
    def _log_system_load_alert(self, component, usage_percent):
        """Логирование алерта о высокой нагрузке системы"""
        logger.warning(
            f"Высокая нагрузка {component}:\n"
            f"Текущее использование: {usage_percent}%\n"
            f"Порог: {self.alert_thresholds[f'max_{component.lower()}_usage']}%\n"
            f"Рекомендуется оптимизировать код или увеличить ресурсы сервера."
        )
    
    def track_trade(self, trade_data):
        """Отслеживание сделки"""
        self.trade_history.append(trade_data)
        
        # Очистка старых данных (хранить только последние 1000 сделок)
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)

        # Анализ эффективности сделок
        self._analyze_trade_performance()

    def log_profit_threshold(self, final_threshold, rejected_candidates, *, base_threshold, adjustments,
                              market_conditions=None, total_candidates=0):
        """Логирование итогового порога и статистики отбора кандидатов"""
        adjustments = adjustments or []
        adjustments_summary = ', '.join(
            f"{adj['reason']}: {adj['value']:+.4f}"
            for adj in adjustments
        ) or 'без корректировок'

        logger.info(
            "🎚️ Итоговый порог прибыли %.4f%% (база %.4f%%) | Условия: %s | Кандидатов: %s | Отброшено: %s",
            final_threshold,
            base_threshold,
            market_conditions or 'неизвестно',
            total_candidates,
            rejected_candidates
        )
        logger.debug("Корректировки порога: %s", adjustments_summary)

    def _analyze_trade_performance(self):
        """Анализ эффективности сделок"""
        if len(self.trade_history) < 10:
            return
        
        # Расчет процента успешных сделок
        successful_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        success_rate = successful_trades / 10
        
        # Проверка на низкую эффективность
        if success_rate < self.alert_thresholds['min_profit_rate']:
            self._log_performance_alert(success_rate)
        
        # Проверка на серию убытков
        consecutive_losses = 0
        for trade in reversed(self.trade_history[-10:]):
            if trade.get('profit', 0) <= 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= self.alert_thresholds['max_consecutive_losses']:
            self._log_consecutive_losses_alert(consecutive_losses)
    
    def _log_performance_alert(self, success_rate):
        """Логирование алерта о низкой эффективности"""
        logger.warning(
            f"Низкая эффективность сделок:\n"
            f"Успешных сделок за последние 10: {success_rate*100:.1f}%\n"
            f"Порог: {self.alert_thresholds['min_profit_rate']*100:.1f}%\n"
            f"Рекомендуется проверить стратегию или параметры."
        )
    
    def _log_consecutive_losses_alert(self, consecutive_losses):
        """Логирование алерта о серии убытков"""
        logger.error(
            f"Серия убыточных сделок:\n"
            f"{consecutive_losses} последовательных убытков\n"
            f"Порог: {self.alert_thresholds['max_consecutive_losses']}\n"
            f"Рекомендуется приостановить торговлю и проанализировать стратегию."
        )
    
    def track_cooldown_violation(self, symbol):
        """Отслеживание нарушений кулдауна"""
        self.cooldown_violations += 1
        if self.cooldown_violations >= 10:  # 10 нарушений подряд
            logger.warning(
                f"Множественные нарушения кулдауна:\n"
                f"Количество нарушений: {self.cooldown_violations}\n"
                f"Символ: {symbol}\n"
                f"Рекомендуется проверить логику кулдауна."
            )
    
    def track_api_error(self, endpoint, error_message):
        """Отслеживание ошибок API"""
        self.api_errors += 1
        if self.api_errors >= 10:  # 10 ошибок подряд
            logger.critical(
                f"Критическое количество ошибок API:\n"
                f"Количество ошибок: {self.api_errors}\n"
                f"Последний эндпоинт: {endpoint}\n"
                f"Ошибка: {error_message}\n"
                f"Рекомендуется перезапустить бота или проверить API ключи."
            )

    def check_balance_health(self, balance_usdt):
        """Проверка здоровья баланса"""
        if balance_usdt < self.alert_thresholds['min_balance']:
            logger.error(
                f"⚠️ Низкий баланс:\n"
                f"Текущий баланс: {balance_usdt:.2f} USDT\n"
                f"Минимальный порог: {self.alert_thresholds['min_balance']} USDT\n"
                f"Торговля может быть приостановлена из-за недостатка средств."
            )

    def update_balance_snapshot(self, balance_usdt):
        """Сохраняет последнее значение баланса для мониторинга."""
        self.last_balance_snapshot = {
            'timestamp': datetime.now(),
            'balance': balance_usdt
        }
    
    def generate_performance_report(self):
        """Генерация отчета о производительности"""
        if not self.trade_history:
            return None
        
        total_trades = len(self.trade_history)
        successful_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0

        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Убираем микросекунды

        cpu_usage = self._get_cpu_usage_string()
        memory_usage = self._get_memory_usage_string()

        report = {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'runtime': runtime_str,
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_stats': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'cooldown_violations': self.cooldown_violations,
                'api_errors': self.api_errors
            }
        }
        
        logger.info(
            f"📊 Отчет о производительности:\n"
            f"   Всего сделок: {report['total_trades']}\n"
            f"   Успешных: {report['successful_trades']} ({report['success_rate']:.1f}%)\n"
            f"   Общая прибыль: {report['total_profit']:.4f} USDT\n"
            f"   Средняя прибыль: {report['avg_profit']:.4f} USDT\n"
            f"   Время работы: {report['runtime']}"
        )
        
        self.last_performance_report = report
        
        return report
    
    def export_trade_history(self, filename=None):
        """Экспорт истории сделок"""
        if not self.trade_history:
            return False

        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        def _convert_datetime_values(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {k: _convert_datetime_values(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert_datetime_values(item) for item in value]
            return value

        def _prepare_trade_plan(trade_plan):
            if not trade_plan:
                return {}
            prepared = _convert_datetime_values(trade_plan)
            return prepared if isinstance(prepared, dict) else {'value': prepared}

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'symbol', 'side', 'amount', 'price', 'profit', 'simulated', 'trade_details']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                
                for trade in self.trade_history:
                    results = trade.get('results') or []

                    if not results:
                        details = trade.get('details', {})
                        symbols = details.get('symbols') or trade.get('symbol') or ''
                        if isinstance(symbols, (list, tuple)):
                            symbols = ','.join(symbols)
                        # Создаем упрощённую запись для сделок старого формата
                        results = [{
                            'symbol': symbols,
                            'side': details.get('direction', trade.get('direction', '')),
                            'qty': details.get('initial_amount', 0),
                            'price': details.get('price', 0)
                        }]

                    for result in results:
                        timestamp = trade['timestamp']
                        if hasattr(timestamp, 'strftime'):
                            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            timestamp_str = str(timestamp)

                        writer.writerow({
                            'timestamp': timestamp_str,
                            'symbol': result.get('symbol', ''),
                            'side': result.get('side', ''),
                            'amount': result.get('qty', result.get('cumExecQty', 0)),
                            'price': result.get('avgPrice', result.get('price', 0)),
                            'profit': trade.get('total_profit', 0) if result == results[-1] else 0,
                            'simulated': trade.get('simulated', False),
                            'trade_details': json.dumps(
                                _prepare_trade_plan(trade.get('trade_plan', {})),
                                default=str
                            )
                        })
            
            logger.info(f"✅ Trade history exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error exporting trade history: {str(e)}")
            return None
    
    def health_check(self):
        """Проверка здоровья системы"""
        try:
            psutil_available = self._ensure_psutil_available()
            if psutil_available:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
            else:
                cpu_percent = 0.0
                memory_percent = 0.0

            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'uptime': str(datetime.now() - self.start_time).split('.')[0],
                'api_latency': self._get_avg_api_latency(),
                'cpu_usage': f"{cpu_percent}%" if psutil_available else 'N/A',
                'memory_usage': f"{memory_percent}%" if psutil_available else 'N/A',
                'active_trades': len(self.trade_history),
                'last_trade_time': self.trade_history[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if self.trade_history else 'N/A',
                'cooldown_violations': self.cooldown_violations,
                'api_errors': self.api_errors,
                'strategy': self._get_strategy_status()
            }

            # Определение статуса
            if health_status['api_latency'] > 2.0 or (psutil_available and cpu_percent > 90):
                health_status['status'] = 'warning'

            if psutil_available and memory_percent > 95:
                health_status['status'] = 'critical'
            
            if self.cooldown_violations > 5 or self.api_errors > 10:
                health_status['status'] = 'critical'
            
            return health_status
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                'status': 'error',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': str(e)
            }
    
    def _get_avg_api_latency(self):
        """Получение средней задержки API"""
        if not self.api_response_times:
            return 0.0

        recent_times = [call['duration'] for call in self.api_response_times[-10:]]
        return sum(recent_times) / len(recent_times) if recent_times else 0.0

    def _ensure_psutil_available(self):
        """Проверяет доступность psutil и логирует предупреждение один раз"""
        if psutil is not None:
            return True

        if not self._psutil_warning_logged:
            logger.warning(
                "Модуль psutil не установлен. Системный мониторинг будет ограничен. "
                "Установите пакет psutil для получения детальной статистики."
            )
            self._psutil_warning_logged = True

        return False

    def _get_cpu_usage_string(self):
        """Возвращает строку с загрузкой CPU либо N/A"""
        if self._ensure_psutil_available():
            return f"{psutil.cpu_percent()}%"
        return 'N/A'

    def _get_memory_usage_string(self):
        """Возвращает строку с загрузкой памяти либо N/A"""
        if self._ensure_psutil_available():
            return f"{psutil.virtual_memory().percent}%"
        return 'N/A'
    
    def send_system_summary(self):
        """Отправка сводки по системе (теперь просто логирование)"""
        health = self.health_check()
        report = self.last_performance_report or {}
        strategy_status = health.get('strategy') or self._get_strategy_status()

        logger.info(
            f"🖥️ Системная сводка:\n"
            f"   ⏱️ Время работы: {health.get('uptime', 'N/A')}\n"
            f"   📊 Статус: {health.get('status', 'N/A').upper()}\n"
            f"   📈 Всего сделок: {report.get('total_trades', 0)}\n"
            f"   💰 Общая прибыль: {report.get('total_profit', 0):.4f} USDT\n"
            f"   🔧 CPU: {health.get('cpu_usage', 'N/A')}\n"
            f"   💾 Память: {health.get('memory_usage', 'N/A')}\n"
            f"   ⚡ API latency: {health.get('api_latency', 0):.2f}с\n"
            f"   ❌ Ошибок API: {health.get('api_errors', 0)}\n"
            f"   ⏳ Нарушений кулдауна: {health.get('cooldown_violations', 0)}\n"
            f"   🧠 Режим стратегии: {strategy_status.get('mode', 'N/A')} | Активная: {strategy_status.get('active', 'N/A')}"
        )
    
    def start_monitoring_loop(self):
        """Запуск цикла мониторинга"""
        import threading
        
        def monitoring_loop():
            while True:
                try:
                    # Отслеживание системных метрик каждые 30 секунд
                    if int(time.time()) % 30 == 0:
                        self.track_system_metrics()
                    
                    # Генерация отчета каждый час
                    if int(time.time()) % 3600 == 0:
                        self.generate_performance_report()
                    
                    # Проверка здоровья системы каждые 5 минут
                    if int(time.time()) % 300 == 0:
                        health = self.health_check()
                        if health['status'] != 'healthy':
                            logger.warning(f"⚠️ Состояние системы: {health['status']} - {health}")
                    
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(60)  # При ошибке ждем минуту
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info("🔄 Advanced monitoring loop started")
# ==== Конец monitoring.py ====

# ==== Начало visualization.py ====
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime
import threading
import time
from config import Config

class Dashboard:
    def __init__(self, engine):
        self.engine = engine
        self.config = Config()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.trade_history = []
        self.price_history = {symbol: [] for symbol in self.config.SYMBOLS}
        self.timestamps = []
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Настройка макета дашборда"""
        self.app.layout = dbc.Container([
            # Заголовок
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Bybit Arbitrage Bot Dashboard", 
                           className="text-center mb-4 text-primary"),
                    html.H5(f"{'TESTNET' if self.config.TESTNET else 'REAL'} MODE", 
                           className="text-center text-warning")
                ], width=12)
            ], className="mb-4"),
            
            # Статистика
            dbc.Row([
                dbc.Col(self._create_stat_card("💰 Total Profit", "profit_value", "0.00 USDT"), width=3),
                dbc.Col(self._create_stat_card("🎯 Total Trades", "trades_value", "0"), width=3),
                dbc.Col(self._create_stat_card("⚡ Avg Profit/Trade", "avg_profit_value", "0.00 USDT"), width=3),
                dbc.Col(self._create_stat_card("⏱️ Last Update", "time_value", "00:00:00"), width=3),
            ], className="mb-4"),
            
            # Графики
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='price-chart', config={'displayModeBar': False})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id='profit-chart', config={'displayModeBar': False})
                ], width=4),
            ], className="mb-4"),
            
            # Спреды и сделки
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='spread-chart', config={'displayModeBar': False})
                ], width=6),
                dbc.Col([
                    html.H4("📈 Recent Trades", className="mb-3"),
                    dbc.Table(id='trades-table', bordered=True, hover=True, 
                             className="bg-dark text-light"),
                    html.Div(id='cooldown-status', className="mt-3")
                ], width=6),
            ], className="mb-4"),
            
            # Управление
            dbc.Row([
                dbc.Col([
                    html.H4("⚙️ Bot Controls", className="mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button("▶️ Start", id="start-btn", color="success", className="me-2"),
                        dbc.Button("⏹️ Stop", id="stop-btn", color="danger", className="me-2"),
                        dbc.Button("🔄 Refresh", id="refresh-btn", color="info"),
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Settings", className="card-title"),
                            dbc.Label("Update Interval (s):"),
                            dcc.Slider(
                                id='interval-slider',
                                min=1,
                                max=10,
                                step=1,
                                value=self.config.UPDATE_INTERVAL,
                                marks={i: str(i) for i in range(1, 11)}
                            ),
                            dbc.Label("Min Profit Threshold (%):"),
                            dcc.Slider(
                                id='profit-slider',
                                min=0.01,
                                max=1.0,
                                step=0.01,
                                value=self.config.MIN_PROFIT_PERCENT,
                                marks={0.1: '0.1%', 0.5: '0.5%', 1.0: '1.0%'}
                            ),
                            dbc.Label("Trade Amount (USDT):"),
                            dcc.Slider(
                                id='trade-amount-slider',
                                min=1,
                                max=100,
                                step=1,
                                value=self.config.TRADE_AMOUNT,
                                marks={10: '10', 50: '50', 100: '100'}
                            ),
                        ])
                    ], className="mt-3")
                ], width=12),
            ]),
            
            # Интервал обновления
            dcc.Interval(
                id='update-interval',
                interval=self.config.UPDATE_INTERVAL * 1000,
                n_intervals=0
            )
        ], fluid=True)
    
    def _create_stat_card(self, title, id, value):
        """Создание карточки статистики"""
        return dbc.Card([
            dbc.CardBody([
                html.H5(title, className="card-title text-muted"),
                html.H3(id=id, children=value, className="card-text text-success fw-bold")
            ])
        ], className="bg-dark border-primary")
    
    def setup_callbacks(self):
        """Настройка callback-функций для интерактивности"""
        
        @self.app.callback(
            [Output('profit_value', 'children'),
             Output('trades_value', 'children'),
             Output('avg_profit_value', 'children'),
             Output('time_value', 'children'),
             Output('price-chart', 'figure'),
             Output('profit-chart', 'figure'),
             Output('spread-chart', 'figure'),
             Output('trades-table', 'children'),
             Output('cooldown-status', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_dashboard(n):
            # Обновление статистики
            profit = sum(trade.get('estimated_profit_usdt', 0) for trade in self.trade_history)
            trades_count = len(self.trade_history)
            avg_profit = profit / trades_count if trades_count > 0 else 0
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Обновление графиков
            price_fig = self._create_price_chart()
            profit_fig = self._create_profit_chart()
            spread_fig = self._create_spread_chart()
            
            # Обновление таблицы сделок
            trades_table = self._create_trades_table()
            cooldown_status = self._create_cooldown_status()
            
            return (
                f"{profit:.2f} USDT",
                str(trades_count),
                f"{avg_profit:.4f} USDT",
                current_time,
                price_fig,
                profit_fig,
                spread_fig,
                trades_table,
                cooldown_status
            )
        
        @self.app.callback(
            Output('update-interval', 'interval'),
            [Input('interval-slider', 'value')]
        )
        def update_interval(value):
            return value * 1000
        
        @self.app.callback(
            [Output('start-btn', 'disabled'),
             Output('stop-btn', 'disabled')],
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks')],
            [State('start-btn', 'disabled'),
             State('stop-btn', 'disabled')]
        )
        def control_bot(start_clicks, stop_clicks, start_disabled, stop_disabled):
            # Логика управления ботом будет добавлена позже
            return start_disabled, stop_disabled
        
        @self.app.callback(
            Output('refresh-btn', 'n_clicks'),
            [Input('refresh-btn', 'n_clicks')]
        )
        def refresh_data(n_clicks):
            if n_clicks:
                # Принудительное обновление данных
                self.update_data()
            return 0
    
    def _create_price_chart(self):
        """Создание графика цен"""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, symbol in enumerate(self.config.SYMBOLS):
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                prices = [p['mid'] for p in self.price_history[symbol]]
                times = [p['timestamp'] for p in self.price_history[symbol]]
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=prices,
                    mode='lines+markers',
                    name=symbol,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f"{symbol}<br>Price: %{{y:.2f}} USDT<br>Time: %{{x}}<extra></extra>"
                ))
        
        fig.update_layout(
            title="💰 Real-time Prices",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_profit_chart(self):
        """Создание графика прибыли"""
        if not self.trade_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="📈 Cumulative Profit",
                template="plotly_dark",
                height=300
            )
            return fig
        
        cumulative_profit = []
        running_sum = 0
        timestamps = []
        
        for trade in self.trade_history:
            running_sum += trade.get('estimated_profit_usdt', 0)
            cumulative_profit.append(running_sum)
            timestamps.append(trade.get('timestamp', datetime.now()))
        
        fig = go.Figure()
        
        # Основная линия прибыли
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative_profit,
            mode='lines+markers',
            name='Cumulative Profit',
            line=dict(color='#00FF00', width=3),
            marker=dict(size=6, color='#00FF00'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # Линия нулевой прибыли
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="📈 Cumulative Profit",
            xaxis_title="Time",
            yaxis_title="Profit (USDT)",
            template="plotly_dark",
            hovermode="x unified",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_spread_chart(self):
        """Создание графика спредов"""
        if not hasattr(self.engine, 'last_tickers'):
            fig = go.Figure()
            fig.add_annotation(
                text="Waiting for data...",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="📊 Spreads Analysis",
                template="plotly_dark",
                height=300
            )
            return fig
        
        symbols = []
        spreads = []
        colors = []
        
        for symbol, data in self.engine.last_tickers.items():
            if data['bid'] > 0 and data['ask'] > 0:
                spread = ((data['ask'] - data['bid']) / data['bid']) * 100
                symbols.append(symbol)
                spreads.append(spread)
                colors.append('#FF6B6B' if spread > 1 else '#4ECDC4')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=spreads,
            marker_color=colors,
            text=[f"{spread:.2f}%" for spread in spreads],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Spread: %{y:.2f}%<extra></extra>"
        ))
        
        # Порог прибыльности
        fig.add_hline(
            y=self.config.MIN_PROFIT_PERCENT * 2,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Threshold: {self.config.MIN_PROFIT_PERCENT * 2:.2f}%",
            annotation_position="right"
        )
        
        fig.update_layout(
            title="📊 Spreads Analysis",
            xaxis_title="Symbols",
            yaxis_title="Spread (%)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(range=[0, max(2, max(spreads) * 1.2) if spreads else 2])
        )
        
        return fig
    
    def _create_trades_table(self):
        """Создание таблицы последних сделок"""
        if not self.trade_history:
            return [
                html.Thead(html.Tr([html.Th("No trades executed yet")])),
                html.Tbody([])
            ]
        
        # Берем последние 10 сделок
        recent_trades = self.trade_history[-10:]
        
        table_header = [
            html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Symbol"),
                html.Th("Type"),
                html.Th("Profit (USDT)"),
                html.Th("Status")
            ]))
        ]
        
        table_body = []
        for trade in reversed(recent_trades):
            timestamp = trade.get('timestamp', datetime.now()).strftime("%H:%M:%S")
            symbol = trade.get('opportunity', {}).get('symbol', 'N/A')
            trade_type = trade.get('opportunity', {}).get('type', 'N/A').upper()
            profit = trade.get('estimated_profit_usdt', 0)
            status = "✅" if profit > 0 else "❌"
            
            row_color = "table-success" if profit > 0 else "table-danger"
            
            table_body.append(html.Tr([
                html.Td(timestamp),
                html.Td(symbol),
                html.Td(trade_type),
                html.Td(f"{profit:.4f}"),
                html.Td(status)
            ], className=row_color))
        
        return table_header + [html.Tbody(table_body)]
    
    def _create_cooldown_status(self):
        """Создание статуса кулдауна"""
        cooldown_period = (
            getattr(self.engine, 'cooldown_period', None)
            or getattr(getattr(self.engine, 'config', None), 'COOLDOWN_PERIOD', None)
        )

        if not cooldown_period or cooldown_period <= 0:
            return html.Div("No cooldowns active", className="text-muted")

        if not hasattr(self.engine, 'last_arbitrage_time') or not self.engine.last_arbitrage_time:
            return html.Div("No cooldowns active", className="text-muted")

        now = datetime.now()
        cooldown_items = []

        for symbol, last_time in self.engine.last_arbitrage_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = max(0, cooldown_period - elapsed)

            if remaining > 0:
                progress = (elapsed / cooldown_period) * 100
                cooldown_items.append(
                    dbc.Progress(
                        value=progress,
                        label=f"{symbol}: {remaining:.0f}s",
                        color="warning" if remaining < 60 else "info",
                        className="mb-2"
                    )
                )
        
        if not cooldown_items:
            return html.Div("✅ No active cooldowns", className="text-success")
        
        return html.Div([
            html.H5("⏳ Active Cooldowns", className="mb-2"),
            *cooldown_items
        ])
    
    def update_data(self):
        """Обновление данных для визуализации"""
        # Обновление истории цен
        if hasattr(self.engine, 'last_tickers'):
            current_time = datetime.now()
            
            for symbol, data in self.engine.last_tickers.items():
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                mid_price = (data['bid'] + data['ask']) / 2
                self.price_history[symbol].append({
                    'timestamp': current_time,
                    'mid': mid_price,
                    'bid': data['bid'],
                    'ask': data['ask']
                })
                
                # Ограничиваем историю последними 100 точками
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
        
        # Обновление истории сделок
        if hasattr(self.engine, 'trade_history'):
            self.trade_history = self.engine.trade_history.copy()
    
    def run_dashboard(self, port=8050):
        """Запуск дашборда в отдельном потоке"""
        def run_app():
            self.app.run_server(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )
        
        dashboard_thread = threading.Thread(target=run_app, daemon=True)
        dashboard_thread.start()
        print(f"📊 Dashboard started at http://localhost:{port}")
        
        # Запускаем цикл обновления данных
        self._start_data_update_loop()
    
    def _start_data_update_loop(self):
        """Цикл обновления данных для визуализации"""
        def update_loop():
            while True:
                try:
                    self.update_data()
                    time.sleep(1)  # Обновление данных каждую секунду
                except Exception as e:
                    print(f"Error updating dashboard data: {e}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
# ==== Конец visualization.py ====

# ==== Начало bybit_client.py ====
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

try:
    from pybit.unified_trading import HTTP, WebSocket
except ModuleNotFoundError:
    HTTP = None
    WebSocket = None

logger = logging.getLogger(__name__)


class BybitWebSocketManager:
    """Управление WebSocket-подключениями к Bybit для котировок и ордеров."""

    def __init__(self, config: Config, *, order_callback=None):
        self.config = config
        self._order_callback = order_callback
        self._ticker_cache = {}
        self._cache_lock = threading.Lock()
        self._public_ws = None
        self._private_ws = None
        self._symbols = set()
        self._order_listeners = []
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._last_ticker_ts = 0
        self._max_staleness = max(getattr(self.config, '_ticker_staleness_warning', 5.0) * 2, 1.0)

    def start(self, symbols):
        """Запуск стримов по списку тикеров."""

        self._symbols = set(symbols)
        self._connect_public_ws()
        self._ensure_monitor()

    def stop(self):
        """Остановка всех подключений (используется при завершении работы)."""

        self._stop_event.set()
        if self._public_ws and hasattr(self._public_ws, 'exit'):
            self._public_ws.exit()
        if self._private_ws and hasattr(self._private_ws, 'exit'):
            self._private_ws.exit()

    def register_order_listener(self, callback):
        """Регистрирует обработчик событий ордеров и инициирует приватный стрим."""

        if not callback:
            return

        if callback not in self._order_listeners:
            self._order_listeners.append(callback)

        self._connect_private_ws()
        self._ensure_monitor()

    def get_cached_tickers(self, symbols, max_age=None):
        """Возвращает свежие котировки из кэша и список недостающих тикеров."""

        max_age = max_age or self._max_staleness
        now = time.time()
        fresh = {}
        missing = []

        with self._cache_lock:
            for symbol in symbols:
                cached = self._ticker_cache.get(symbol)
                if cached and now - cached['ts'] <= max_age:
                    fresh[symbol] = cached['data']
                else:
                    missing.append(symbol)

        return fresh, missing

    def update_cache(self, tickers):
        """Принудительное обновление кэша внешними данными (например, после REST-запроса)."""

        now = time.time()
        with self._cache_lock:
            for symbol, data in tickers.items():
                self._ticker_cache[symbol] = {'data': data, 'ts': now}
                self._last_ticker_ts = now

    def _ensure_monitor(self):
        """Запускает фоновый мониторинг состояния подключений."""

        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
        self._monitor_thread.start()

    def _monitor_connections(self):
        """Следит за обрывами соединений и восстанавливает стримы."""

        while not self._stop_event.is_set():
            try:
                now = time.time()
                if self._symbols and (self._public_ws is None or now - self._last_ticker_ts > self._max_staleness):
                    logger.debug("Переподключение публичного стрима котировок")
                    self._restart_public_ws()

                if self._order_listeners and self._private_ws is None:
                    logger.debug("Переподключение приватного стрима ордеров")
                    self._connect_private_ws()
            except Exception as exc:
                logger.warning("Ошибка мониторинга WebSocket: %s", exc)

            time.sleep(3)

    def _connect_public_ws(self):
        """Создаёт подключение для получения котировок."""

        if WebSocket is None:
            logger.warning("pybit не установлен, WebSocket котировок недоступен")
            return

        try:
            self._public_ws = WebSocket(
                channel_type=self.config.MARKET_CATEGORY,
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
            )
            self._public_ws.ticker_stream(symbol=list(self._symbols), callback=self._handle_ticker)
            self._last_ticker_ts = time.time()
            logger.info("📡 WebSocket котировок запущен для %s символов", len(self._symbols))
        except Exception as exc:
            logger.warning("Не удалось подключиться к публичному стриму котировок: %s", exc)
            self._public_ws = None

    def _restart_public_ws(self):
        """Перезапускает публичное подключение."""

        try:
            if self._public_ws and hasattr(self._public_ws, 'exit'):
                self._public_ws.exit()
        finally:
            self._connect_public_ws()

    def _connect_private_ws(self):
        """Создаёт приватное подключение для событий ордеров."""

        if WebSocket is None:
            logger.warning("pybit не установлен, приватный WebSocket недоступен")
            return

        if not self.config.API_KEY or not self.config.API_SECRET:
            logger.warning("API ключи не заданы, пропускаем подписку на приватные события")
            return

        try:
            self._private_ws = WebSocket(
                channel_type="private",
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
            )
            self._private_ws.order_stream(callback=self._handle_order)
            logger.info("🔔 WebSocket ордеров активирован")
        except Exception as exc:
            logger.warning("Не удалось подключиться к приватному стриму ордеров: %s", exc)
            self._private_ws = None

    def _handle_ticker(self, message):
        """Нормализация входящих котировок и сохранение в кэше."""

        data = message.get('data') if isinstance(message, dict) else None
        if not data:
            return

        if isinstance(data, dict):
            entries = [data]
        else:
            entries = data

        now = time.time()

        with self._cache_lock:
            for entry in entries:
                symbol = entry.get('symbol') or entry.get('s')
                if not symbol:
                    continue

                bid = self._safe_float(
                    entry.get('bid1Price')
                    or entry.get('bestBidPrice')
                    or entry.get('bp')
                    or entry.get('b1'),
                    0,
                )
                ask = self._safe_float(
                    entry.get('ask1Price')
                    or entry.get('bestAskPrice')
                    or entry.get('ap')
                    or entry.get('a1'),
                    0,
                )

                ticker = {
                    'bid': bid,
                    'ask': ask,
                    'last_price': self._safe_float(entry.get('lastPrice') or entry.get('lp') or entry.get('price'), 0),
                    'bid_size': self._safe_float(entry.get('bid1Size') or entry.get('b1Size') or entry.get('bidSize') or entry.get('bq')),  
                    'ask_size': self._safe_float(entry.get('ask1Size') or entry.get('a1Size') or entry.get('askSize') or entry.get('aq')),
                }

                self._ticker_cache[symbol] = {'data': ticker, 'ts': now}
                self._last_ticker_ts = now

        if self._order_callback:
            # Хук на внешний обработчик может использовать котировки для актуализации внутренних структур
            try:
                self._order_callback({'type': 'ticker', 'symbols': [e.get('symbol') for e in entries if e.get('symbol')]})
            except Exception:
                logger.debug("Ошибка колбэка на котировки", exc_info=True)

    def _handle_order(self, message):
        """Пробрасывает события ордеров зарегистрированным слушателям."""

        data = message.get('data') if isinstance(message, dict) else None
        if not data:
            return

        events = data if isinstance(data, list) else [data]

        for event in events:
            normalized = {
                'orderId': event.get('orderId'),
                'symbol': event.get('symbol'),
                'orderStatus': event.get('orderStatus'),
                'side': event.get('side'),
                'leavesQty': self._safe_float(event.get('leavesQty')),
                'cumExecQty': self._safe_float(event.get('cumExecQty')),
                'avgPrice': self._safe_float(event.get('avgPrice') or event.get('lastPrice')),
                'execType': event.get('execType') or event.get('eventType'),
                'updatedTime': event.get('updatedTime') or event.get('ts') or int(time.time() * 1000),
            }

            for listener in self._order_listeners:
                try:
                    listener(normalized)
                except Exception:
                    logger.debug("Ошибка в обработчике событий ордеров", exc_info=True)

    def _safe_float(self, value, default=0.0):
        """Безопасное приведение к float для всех входящих данных."""

        try:
            if value is None:
                return default

            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default

            return float(value)
        except (TypeError, ValueError):
            return default

class BybitClient:
    def __init__(self):
        self.config = Config()
        self.session = self._create_session()
        self.account_type = "UNIFIED" if not self.config.TESTNET else "CONTRACT"
        # Всегда заранее сохраняем сегмент рынка, чтобы одинаково использовать его во всех запросах
        self.market_category = getattr(self.config, "MARKET_CATEGORY", "spot")
        self.ws_manager = None
        self.order_error_metrics = defaultdict(int)
        self._initialize_websocket_streams()
        logger.info(
            f"Bybit client initialized. Testnet: {self.config.TESTNET}, "
            f"Account type: {self.account_type}"
        )
        logger.info(f"🎯 Market category set to: {self.market_category}")

    def _classify_error(self, *, response=None, exception=None):
        """Определяет тип ошибки для логирования и метрик."""

        if exception is not None:
            message = str(exception)

            if isinstance(exception, (TimeoutError, )):
                return "network", message

            if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                return "network", message

            return "unknown", message

        if response is None:
            return "unknown", "Пустой ответ от API"

        message = response.get('retMsg', '') or ''
        normalized = message.lower()
        ret_code = response.get('retCode')

        validation_keywords = (
            'invalid', 'parameter', 'qty', 'quantity', 'insufficient', 'leverage', 'precision', 'required'
        )
        refusal_keywords = (
            'reject', 'rejected', 'blocked', 'limit', 'risk', 'system busy', 'maintenance', 'forbidden', 'denied'
        )

        if ret_code and str(ret_code).startswith('100'):
            return "validation", message or f"Код ошибки {ret_code}"

        if any(keyword in normalized for keyword in validation_keywords):
            return "validation", message or "Ошибка валидации параметров"

        if any(keyword in normalized for keyword in refusal_keywords):
            return "exchange_refusal", message or "Биржа отвергла запрос"

        return "unknown", message or f"Код ошибки {ret_code}"

    def _record_error_metric(self, error_type):
        """Увеличивает счётчик ошибок указанного типа."""

        self.order_error_metrics[error_type] += 1

    def _log_attempt_result(self, operation, attempt, max_attempts, success, error_type, message):
        """Единообразное логирование попыток с типами ошибок."""

        status_label = "успех" if success else "ошибка"
        logger_method = logger.info if success else logger.warning
        logger_method(
            "%s: попытка %s/%s завершилась как %s (%s). %s",
            operation,
            attempt,
            max_attempts,
            status_label,
            error_type,
            message,
        )

    def _is_status_uncertain(self, status: str | None) -> bool:
        """Определяет, требуется ли доуточнение статуса ордера."""

        if not status:
            return True

        uncertain_statuses = {
            'Created', 'New', 'Untriggered', 'PartiallyFilled', 'Pending', 'Triggered'
        }
        return status in uncertain_statuses

    def _ensure_order_finalized(self, order_id, symbol, initial_status, fallback_payload=None):
        """Дополнительный опрос/отмена ордера при неоднозначном статусе."""

        if not order_id:
            logger.warning("Не удалось уточнить статус: отсутствует orderId для %s", symbol)
            return fallback_payload

        logger.warning(
            "Статус ордера %s для %s неоднозначен (%s). Запускаем уточнение/отмену.",
            order_id,
            symbol,
            initial_status or 'unknown',
        )

        for attempt in range(1, 3):
            fetched = self.get_order_status(order_id, symbol)
            if fetched and not self._is_status_uncertain(fetched.get('orderStatus')):
                return fetched

            cancel_result = self.cancel_order(order_id, symbol)
            if cancel_result:
                fetched_after_cancel = self.get_order_status(order_id, symbol)
                if fetched_after_cancel:
                    return fetched_after_cancel
            time.sleep(0.5 * attempt)

        return fallback_payload

    def _initialize_websocket_streams(self):
        """Настраивает WebSocket для котировок и приватных событий."""

        if WebSocket is None:
            logger.warning("pybit не установлен, пропускаем запуск WebSocket")
            return

        try:
            self.ws_manager = BybitWebSocketManager(self.config)
            self.ws_manager.start(self.config.SYMBOLS)
        except Exception as exc:
            logger.warning("Не удалось инициализировать WebSocket-стримы: %s", exc, exc_info=True)
            self.ws_manager = None
    
    def _create_session(self):
        """Создание сессии для работы с Bybit API"""
        if HTTP is None:
            raise RuntimeError(
                "pybit is not installed. Install dependencies with 'pip install -r requirements.txt'"
            )

        try:
            return HTTP(
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
                recv_window=10000  # Увеличенное время ожидания для тестнета
            )
        except Exception as e:
            logger.error(f"❌ Failed to create Bybit session: {str(e)}")
            raise
    
    def get_tickers(self, symbols):
        """Получение котировок пачками с минимальным числом HTTP-запросов"""
        requested_symbols = sorted(set(symbols))
        tickers = {}

        if not requested_symbols:
            return tickers

        cache_hits = {}
        remaining_symbols = set(requested_symbols)

        if self.ws_manager:
            cache_hits, fresh_missing = self.ws_manager.get_cached_tickers(
                requested_symbols,
                max_age=getattr(self.config, '_ticker_staleness_warning', 5.0),
            )
            tickers.update(cache_hits)
            remaining_symbols = set(fresh_missing)

            if not remaining_symbols:
                logger.debug("♻️ Используем кэш WebSocket для всех тикеров")
                self._validate_ticker_freshness(tickers)
                return tickers

        logger.debug(f"🔍 Requesting {len(requested_symbols)} symbols: {requested_symbols}")

        start_time = time.time()
        request_count = 0

        def _extract_from_response(response, label):
            """Извлекает данные тикеров из ответа и обновляет остаток"""
            nonlocal tickers
            if not response:
                logger.debug(f"❌ Пустой ответ в блоке {label}")
                return

            if response.get('retCode') != 0 or not response.get('result'):
                logger.debug(f"❌ API error in {label}: {response.get('retMsg')}")
                return

            ticker_list = response['result'].get('list', [])
            for ticker_data in ticker_list:
                symbol = ticker_data.get('symbol')
                if symbol not in remaining_symbols:
                    continue

                tickers[symbol] = {
                    'bid': self._safe_float(ticker_data.get('bid1Price', 0)),
                    'ask': self._safe_float(ticker_data.get('ask1Price', 0)),
                    'last': self._safe_float(ticker_data.get('lastPrice', 0)),
                    'timestamp': ticker_data.get('time')
                }
                remaining_symbols.discard(symbol)
                logger.debug(
                    f"✅ {symbol}: bid={tickers[symbol]['bid']}, ask={tickers[symbol]['ask']} (source={label})"
                )

        # Основной bulk-запрос без параметра symbol
        try:
            cursor = None
            while True:
                params = {'category': self.market_category}
                if cursor:
                    params['cursor'] = cursor

                response = self.session.get_tickers(**params)
                request_count += 1
                _extract_from_response(response, 'bulk')

                cursor = response.get('result', {}).get('nextPageCursor') if response else None
                if not cursor or not remaining_symbols:
                    break

        except Exception as e:
            logger.debug(f"🔥 Bulk request failed: {str(e)}")

        # Фолбэк: запрашиваем оставшиеся символы параллельно
        if remaining_symbols:
            logger.debug(
                f"⚙️ Bulk вернул не все данные, догружаем {len(remaining_symbols)} символов параллельно"
            )

            def _fetch_symbol(symbol):
                try:
                    return self.session.get_tickers(category=self.market_category, symbol=symbol)
                except Exception as exc:
                    logger.debug(f"🔥 Exception for {symbol}: {str(exc)}")
                    return None

            with ThreadPoolExecutor(max_workers=min(8, len(remaining_symbols))) as executor:
                future_to_symbol = {
                    executor.submit(_fetch_symbol, symbol): symbol for symbol in list(remaining_symbols)
                }

                for future in as_completed(future_to_symbol):
                    request_count += 1
                    symbol = future_to_symbol[future]
                    response = future.result()
                    _extract_from_response(response, f'fallback:{symbol}')

        duration = time.time() - start_time
        logger.debug(
            f"📊 Total tickers received: {len(tickers)} (requests: {request_count}, missing: {len(remaining_symbols)})"
        )

        if duration < 2:
            logger.info(
                f"⚡️ Сбор {len(tickers)} тикеров занял {duration:.2f} с (меньше 2 секунд, запросов: {request_count})"
            )
        else:
            logger.warning(
                f"⏱️ Сбор {len(tickers)} тикеров занял {duration:.2f} с (запросов: {request_count})"
            )

        if self.ws_manager and tickers:
            self.ws_manager.update_cache(tickers)

        self._validate_ticker_freshness(tickers)

        return tickers

    def add_order_listener(self, callback):
        """Подключает внешний обработчик событий ордеров."""

        if not self.ws_manager:
            logger.warning("WebSocket менеджер не инициализирован, события ордеров недоступны")
            return

        self.ws_manager.register_order_listener(callback)

    def _validate_ticker_freshness(self, tickers):
        """Проверяет насколько свежие котировки получены от Bybit"""
        if not tickers:
            return

        freshness_limit_ms = int(self.config.TICKER_STALENESS_WARNING_SEC * 1000)
        now_ms = int(time.time() * 1000)
        stale = []

        for symbol, data in tickers.items():
            timestamp = data.get('timestamp')
            if not timestamp:
                continue

            try:
                age_ms = now_ms - int(float(timestamp))
            except (TypeError, ValueError):
                continue

            if age_ms > freshness_limit_ms:
                stale.append((symbol, age_ms / 1000))

        if stale:
            preview = ', '.join(f"{sym} ({age:.1f}с)" for sym, age in stale[:5])
            logger.warning(
                "🕒 Обнаружены устаревшие котировки (>%.1fс): %s",
                self.config.TICKER_STALENESS_WARNING_SEC,
                preview
            )
        else:
            logger.debug(
                "Котировки %s инструментов свежее %.1f секунд",
                len(tickers),
                self.config.TICKER_STALENESS_WARNING_SEC
            )

    def get_order_book(self, symbol, depth=5):
        """Возвращает стакан для инструмента с указанной глубиной"""
        try:
            response = self.session.get_orderbook(
                category=self.market_category,
                symbol=symbol,
                limit=depth
            )

            if response.get('retCode') != 0 or 'result' not in response:
                logger.debug(
                    "Не удалось получить стакан для %s: %s",
                    symbol,
                    response.get('retMsg') if isinstance(response, dict) else 'unknown error'
                )
                return {'bids': [], 'asks': []}

            orderbook = response['result']
            bids = orderbook.get('b', [])
            asks = orderbook.get('a', [])

            # Форматируем в удобный вид: список словарей с price/size
            def _normalize(side):
                normalized = []
                for level in side:
                    if len(level) < 2:
                        continue
                    price = self._safe_float(level[0])
                    size = self._safe_float(level[1])
                    normalized.append({'price': price, 'size': size})
                return normalized

            return {
                'bids': _normalize(bids),
                'asks': _normalize(asks)
            }
        except Exception as exc:
            logger.debug("Ошибка при получении стакана %s: %s", symbol, str(exc))
            return {'bids': [], 'asks': []}
        
    def get_balance(self, coin='USDT'):
        """Получение баланса для конкретной монеты - исправленная версия для тестнета"""
        try:
            # Всегда используем фиктивный баланс для тестнета для избежания ошибок
            if self.config.TESTNET:
                logger.info("🧪 Using mock balance for testnet")
                return {'available': 100.0, 'total': 100.0, 'coin': coin}
        
            # Реальная логика для основной сети
            response = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin=coin
            )
        
            if response.get('retCode') == 0 and response.get('result'):
                balance_list = response['result'].get('list', [])
                if balance_list:
                    for account in balance_list:
                        coin_balances = account.get('coin', [])
                        for coin_balance in coin_balances:
                            if coin_balance.get('coin') == coin:
                                available = self._safe_float(coin_balance.get('availableToWithdraw', 0))
                                total = self._safe_float(coin_balance.get('walletBalance', 0))
                                return {
                                    'available': available,
                                    'total': total,
                                    'coin': coin
                                }
            logger.warning(f"No balance data for {coin}: {response.get('retMsg', 'Unknown error')}")
            return {'available': 0.0, 'total': 0.0, 'coin': coin}
        except Exception as e:
            logger.error(f"Error getting balance for {coin}: {str(e)}")
            return {'available': 0.0, 'total': 0.0, 'coin': coin}

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование к float, чтобы пустые строки не ломали расчеты."""
        try:
            if value is None:
                return default

            # Пустые строки и значения с пробелами должны превращаться в дефолт сразу
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default

            return float(value)
        except (TypeError, ValueError):
            return default
    
    def place_order(
        self,
        symbol,
        side,
        qty,
        price=None,
        order_type='Market',
        trigger_price=None,
        trigger_by='LastPrice',
        reduce_only=False,
    ):
        """Размещение ордера на бирже с улучшенной обработкой ошибок и поддержкой контингентных триггеров"""
        try:
            # Проверка минимальных объемов для тестнета
            if self.config.TESTNET:
                if qty < 0.001 and symbol in ['BTCUSDT', 'ETHUSDT']:
                    logger.warning(f"🧪 Testnet: Increasing quantity for {symbol} from {qty} to 0.001")
                    qty = 0.001

            params = {
                'category': self.market_category,
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'timeInForce': 'GTC' if order_type == 'Limit' else 'IOC',
                'isLeverage': 0,
                'orderFilter': 'Order',
                'reduceOnly': 1 if reduce_only else 0,
            }

            if price and order_type == 'Limit':
                params['price'] = str(price)

            if trigger_price is not None:
                params['triggerPrice'] = str(trigger_price)
                params['triggerBy'] = trigger_by
                params['orderFilter'] = 'tpslOrder' if order_type.lower() != 'market' else 'Order'

            logger.info(f"🚀 Placing {order_type} order: {params}")

            # В тестнете не выполняем реальные ордера, только имитируем
            if self.config.TESTNET:
                logger.info(f"🧪 TESTNET MODE: Simulating order execution (no real order placed)")
                return {
                    'orderId': f"test_order_{int(time.time())}",
                    'orderStatus': 'Filled',
                    'price': str(price) if price else 'market',
                    'avgPrice': str(price) if price else 'market',
                    'qty': str(qty),
                    'cumExecQty': str(qty),
                    'symbol': symbol
                }

            max_attempts = 3
            base_delay = 0.5
            last_result = None

            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.place_order(**params)
                except Exception as exc:
                    error_type, error_message = self._classify_error(exception=exc)
                    self._record_error_metric(error_type)
                    self._log_attempt_result(
                        "place_order",
                        attempt,
                        max_attempts,
                        False,
                        error_type,
                        f"Исключение при размещении: {error_message}",
                    )

                    if attempt < max_attempts:
                        time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue

                if response and response.get('retCode') == 0 and response.get('result'):
                    result = response['result']
                    order_id = result.get('orderId')
                    status = result.get('orderStatus')

                    self._log_attempt_result(
                        "place_order",
                        attempt,
                        max_attempts,
                        True,
                        "ok",
                        f"Статус {status}, orderId={order_id}",
                    )

                    if self._is_status_uncertain(status):
                        return self._ensure_order_finalized(order_id, symbol, status, fallback_payload=result)

                    return result

                error_type, error_message = self._classify_error(response=response)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "place_order",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Код {response.get('retCode') if response else 'N/A'}",
                )

                last_result = response

                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))

            if last_result and last_result.get('result', {}).get('orderId'):
                uncertain = last_result['result']
                return self._ensure_order_finalized(
                    uncertain.get('orderId'),
                    symbol,
                    uncertain.get('orderStatus'),
                    fallback_payload=uncertain,
                )

            return None

        except Exception as e:
            logger.error(f"🔥 Critical error placing order: {str(e)}", exc_info=True)
            self._record_error_metric("unknown")
            return None
    
    def get_order_status(self, order_id, symbol):
        """Получение статуса ордера"""
        try:
            max_attempts = 3
            base_delay = 0.5

            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.get_order_history(
                        category=self.market_category,
                        orderId=order_id,
                        symbol=symbol
                    )
                except Exception as exc:
                    error_type, error_message = self._classify_error(exception=exc)
                    self._record_error_metric(error_type)
                    self._log_attempt_result(
                        "get_order_status",
                        attempt,
                        max_attempts,
                        False,
                        error_type,
                        f"Исключение при запросе статуса: {error_message}",
                    )

                    if attempt < max_attempts:
                        time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue

                if response.get('retCode') == 0 and response.get('result'):
                    order_list = response['result'].get('list', [])
                    if order_list:
                        order = order_list[0]
                        logger.debug(
                            f"Order status: {order.get('orderStatus')}, Filled: {order.get('cumExecQty')}/{order.get('qty')}"
                        )
                        self._log_attempt_result(
                            "get_order_status",
                            attempt,
                            max_attempts,
                            True,
                            "ok",
                            f"Получен статус {order.get('orderStatus')}",
                        )
                        return order

                error_type, error_message = self._classify_error(response=response)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "get_order_status",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Статус не найден для {order_id}",
                )

                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))

            logger.warning(f"No order found for ID {order_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            self._record_error_metric("unknown")
            return None
    
    def cancel_order(self, order_id, symbol):
        """Отмена ордера"""
        try:
            # В тестнете не выполняем реальную отмену
            if self.config.TESTNET:
                logger.info(f"🧪 TESTNET MODE: Simulating order cancellation for {order_id}")
                return True
            
            response = self.session.cancel_order(
                category=self.market_category,
                orderId=order_id,
                symbol=symbol
            )
            
            if response.get('retCode') == 0:
                logger.info(f"CloseOperation: Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"CloseOperation failed: {response.get('retMsg', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_open_orders(self, symbol=None):
        """Получение открытых ордеров"""
        try:
            params = {'category': self.market_category}
            if symbol:
                params['symbol'] = symbol
            
            response = self.session.get_open_orders(**params)
            
            if response.get('retCode') == 0 and response.get('result'):
                return response['result'].get('list', [])
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []
# ==== Конец bybit_client.py ====

# ==== Начало advanced_arbitrage_engine.py ====
import inspect
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from itertools import permutations
from pathlib import Path

from bybit_client import BybitClient
from config import Config
from monitoring import AdvancedMonitor
from performance_optimizer import PerformanceOptimizer
from real_trading import RealTradingExecutor
# Импортируем менеджер стратегий напрямую из локального модуля без пакета strategies
from indicator_strategies import StrategyManager
# Используем реальный локальный модуль math_stats вместо устаревшего utils.math_stats
from math_stats import mean, rolling_mean


logger = logging.getLogger(__name__)

class AdvancedArbitrageEngine:
    def __init__(self, config=None):
        self._log_module_origin()
        self._ensure_integrity()

        self.client = BybitClient()
        self.monitor = AdvancedMonitor(self)
        self.real_trader = RealTradingExecutor()
        self.strategy_manager = None
        self.performance_optimizer = None

        base_config = config or Config()
        self._apply_config(base_config, reset_state=True, recreate_client=False)

        self.monitor.start_monitoring_loop()
        logger.info("🚀 Advanced Triangular Arbitrage Engine initialized")

    def _apply_config(self, new_config, *, reset_state=True, recreate_client=True):
        """Подменяет конфигурацию и пересобирает связанные структуры."""

        if not self._validate_config(new_config):
            logger.error("❌ Переданная конфигурация не прошла валидацию")
            return False

        self.config = new_config

        if recreate_client:
            self.client = BybitClient()

        self.cooldown_period = getattr(self.config, 'COOLDOWN_PERIOD', None) or 180

        if self.strategy_manager is None:
            self.strategy_manager = StrategyManager(self.config)
        else:
            self.strategy_manager.update_config(self.config)

        if self.performance_optimizer is None:
            self.performance_optimizer = PerformanceOptimizer(self.config)
        else:
            self.performance_optimizer.update_config(self.config)

        if reset_state:
            self._initialize_data_structures()

        self._initialize_triangle_stats()
        self._initialize_symbols()
        self.optimized_triangles = self.performance_optimizer.get_optimized_triangles()
        return True

    def _log_module_origin(self):
        """Фиксирует путь к модулю, откуда загружен движок."""
        module_path = Path(__file__).resolve()
        logger.info("📂 AdvancedArbitrageEngine загружен из %s", module_path)

    def _ensure_integrity(self):
        """Проверяет наличие и исходник критичных методов."""
        if not hasattr(self.__class__, "_initialize_triangle_stats"):
            raise AttributeError("Метод _initialize_triangle_stats не найден в AdvancedArbitrageEngine")

        method = getattr(self.__class__, "_initialize_triangle_stats")
        method_file = Path(inspect.getsourcefile(method)).resolve()
        module_path = Path(__file__).resolve()

        if method_file != module_path:
            raise ImportError(
                f"Метод _initialize_triangle_stats загружен из другого файла: {method_file}. Ожидался {module_path}"
            )

    def _initialize_data_structures(self):
        """Вынос инициализации структур данных в отдельный метод."""
        self.price_history = {}
        self.volatility_data = {}
        self.trade_history = []
        self.performance_stats = defaultdict(lambda: {'success': 0, 'failures': 0, 'total_profit': 0})
        self.last_arbitrage_time = {}
        self.triangle_cooldown = {}
        self.ohlcv_history = {}
        self.last_strategy_context = {}
        self.last_tickers = {}
        self._last_candidates = []
        self._last_market_analysis = {'market_conditions': 'normal', 'overall_volatility': 0}
        self.no_opportunity_cycles = 0
        self.aggressive_filter_metrics = defaultdict(int)
        self._last_reported_balance = None
        self._last_dynamic_threshold = None
        self._quote_suffix_cache = None
        self.optimized_triangles = []

    def _initialize_triangle_stats(self):
        """Формирует базовую статистику по треугольникам."""
        self.triangle_stats = {}
        for triangle in self.config.TRIANGULAR_PAIRS:
            self.triangle_stats[triangle['name']] = {
                'opportunities_found': 0,
                'executed_trades': 0,
                'failures': 0,
                'total_profit': 0,
                'last_execution': None,
                'success_rate': 0
            }

    def _validate_config(self, config=None):
        """Быстрая валидация критичных параметров конфигурации."""
        cfg = config or self.config
        is_valid = True

        if not cfg.TRIANGULAR_PAIRS:
            logger.error("❌ Конфигурация не содержит треугольников для арбитража")
            is_valid = False

        if getattr(cfg, 'MIN_TRIANGULAR_PROFIT', 0) <= 0:
            logger.warning(
                "⚠️ MIN_TRIANGULAR_PROFIT должен быть положительным. Текущее значение: %s", cfg.MIN_TRIANGULAR_PROFIT
            )

        if getattr(cfg, 'UPDATE_INTERVAL', 0) <= 0:
            logger.warning(
                "⚠️ UPDATE_INTERVAL должен быть больше нуля. Текущее значение: %s", cfg.UPDATE_INTERVAL
            )

        if not cfg.API_KEY or not cfg.API_SECRET:
            logger.warning("🔒 API ключи не заданы или пустые. Торговля может быть недоступна")

        return is_valid

    def reload_config(self):
        """Перезагрузка конфигурации без перезапуска бота."""
        try:
            new_config = Config()
            if not self._apply_config(new_config, reset_state=True, recreate_client=True):
                logger.error("❌ Перезагрузка конфигурации отменена из-за ошибок валидации")
                return False

            logger.info("🔄 Конфигурация успешно перезагружена без рестарта")
            return True
        except Exception as exc:
            logger.exception("Ошибка при перезагрузке конфигурации: %s", exc)
            return False

    def _initialize_symbols(self):
        """Инициализация всех необходимых символов"""
        all_symbols = set(self.config.SYMBOLS)
        for triangle in self.config.TRIANGULAR_PAIRS:
            for symbol in triangle['legs']:
                all_symbols.add(symbol)
        
        for symbol in all_symbols:
            self.price_history[symbol] = {
                'timestamps': deque(maxlen=500),
                'bids': deque(maxlen=500),
                'asks': deque(maxlen=500),
                'spreads': deque(maxlen=500),
                'raw_spreads': deque(maxlen=500),
                'bid_volumes': deque(maxlen=500),
                'ask_volumes': deque(maxlen=500)
            }
            self.volatility_data[symbol] = {
                'short_term': deque(maxlen=50),
                'long_term': deque(maxlen=200)
            }
            self.ohlcv_history[symbol] = {
                'timestamps': deque(maxlen=500),
                'open': deque(maxlen=500),
                'high': deque(maxlen=500),
                'low': deque(maxlen=500),
                'close': deque(maxlen=500),
                'volume': deque(maxlen=500)
            }

    def update_market_data(self, tickers):
        """Обновление рыночных данных с расширенной аналитикой"""
        current_time = datetime.now()

        for symbol, data in tickers.items():
            try:
                if symbol not in self.price_history:
                    continue

                bid = data.get('bid', 0)
                ask = data.get('ask', 0)
                bid_volume = data.get('bid_size') or data.get('bid_qty') or data.get('bid_volume') or data.get('bidVol') or data.get('bidVol24h') or 0
                ask_volume = data.get('ask_size') or data.get('ask_qty') or data.get('ask_volume') or data.get('askVol') or data.get('askVol24h') or 0

                # Обновление истории цен
                self.price_history[symbol]['timestamps'].append(current_time)
                self.price_history[symbol]['bids'].append(bid)
                self.price_history[symbol]['asks'].append(ask)
                self.price_history[symbol]['bid_volumes'].append(float(bid_volume))
                self.price_history[symbol]['ask_volumes'].append(float(ask_volume))

                # Расчет спреда
                if bid > 0 and ask > 0:
                    spread_percent = ((ask - bid) / bid) * 100
                    self.price_history[symbol]['spreads'].append(spread_percent)
                    self.price_history[symbol]['raw_spreads'].append(ask - bid)

                # Обновление волатильности
                mid_price = (bid + ask) / 2
                if len(self.price_history[symbol]['bids']) > 1:
                    prev_mid = (self.price_history[symbol]['bids'][-2] +
                               self.price_history[symbol]['asks'][-2]) / 2
                    price_change = ((mid_price - prev_mid) / prev_mid) * 100
                    self.volatility_data[symbol]['short_term'].append(abs(price_change))

                # Агрегируем OHLCV для индикаторов
                ohlcv = self.ohlcv_history[symbol]
                open_price = data.get('open', bid)
                high_price = data.get('high', max(bid, ask))
                low_price = data.get('low', min(bid, ask))
                close_price = data.get('last_price', mid_price)
                volume = data.get('volume', data.get('turnover24h', 0))

                ohlcv['timestamps'].append(current_time)
                ohlcv['open'].append(open_price)
                ohlcv['high'].append(high_price)
                ohlcv['low'].append(low_price)
                ohlcv['close'].append(close_price)
                ohlcv['volume'].append(volume)
            except KeyError as exc:
                logger.warning("Пропуск тикера %s из-за отсутствующего ключа: %s. Сырой ответ: %s", symbol, exc, data)
                continue

    def analyze_market_conditions(self):
        """Анализ рыночных условий для оптимизации арбитража"""
        market_analysis = {
            'overall_volatility': 0,
            'best_triangles': [],
            'market_conditions': 'normal',
            'average_spread_percent': 0.0,
            'orderbook_imbalance': 0.0
        }
        
        volatilities = []
        for symbol, data in self.volatility_data.items():
            if data['short_term']:
                vol = mean(data['short_term'])
                volatilities.append(vol)

        if volatilities:
            market_analysis['overall_volatility'] = mean(volatilities)

        micro = self._calculate_microstructure_metrics()
        market_analysis['average_spread_percent'] = micro['average_spread_percent']
        market_analysis['orderbook_imbalance'] = micro['orderbook_imbalance']

        # Определение рыночных условий
        if market_analysis['overall_volatility'] > 2:
            market_analysis['market_conditions'] = 'high_volatility'
        elif market_analysis['overall_volatility'] < 0.1:
            market_analysis['market_conditions'] = 'low_volatility'

        return market_analysis

    def _calc_dynamic_threshold_testnet(self, base_profit_threshold, market_analysis, commission_buffer, slippage_buffer, volatility_buffer):
        """Расчет динамического порога для тестовой среды."""
        dynamic_profit_threshold = base_profit_threshold
        threshold_adjustments = []

        dynamic_profit_threshold += commission_buffer
        threshold_adjustments.append({'reason': 'комиссии цикла', 'value': commission_buffer})

        if slippage_buffer:
            dynamic_profit_threshold += slippage_buffer
            threshold_adjustments.append({'reason': 'запас на проскальзывание', 'value': slippage_buffer})

        spread_adjustment = min(0.02, (market_analysis.get('average_spread_percent', 0) or 0) / 150)
        if spread_adjustment:
            dynamic_profit_threshold += spread_adjustment
            threshold_adjustments.append({'reason': 'средний спред', 'value': spread_adjustment})

        if market_analysis['market_conditions'] == 'high_volatility':
            dynamic_profit_threshold += 0.01
            threshold_adjustments.append({'reason': 'высокая волатильность', 'value': 0.01})
        elif market_analysis['market_conditions'] == 'low_volatility':
            dynamic_profit_threshold -= 0.005
            threshold_adjustments.append({'reason': 'низкая волатильность', 'value': -0.005})

        if self.no_opportunity_cycles:
            relax = -min(0.03, self.no_opportunity_cycles * 0.005)
            dynamic_profit_threshold += relax
            threshold_adjustments.append({'reason': 'накопленные пустые циклы', 'value': relax})

        if volatility_buffer:
            dynamic_profit_threshold += volatility_buffer
            threshold_adjustments.append({'reason': 'реальная волатильность', 'value': volatility_buffer})

        min_dynamic_floor = max(
            getattr(self.config, 'MIN_DYNAMIC_PROFIT_FLOOR', 0.0),
            base_profit_threshold + commission_buffer + slippage_buffer
        )
        if dynamic_profit_threshold < min_dynamic_floor:
            threshold_adjustments.append({'reason': 'нижняя граница тестнета', 'value': min_dynamic_floor - dynamic_profit_threshold})
            dynamic_profit_threshold = min_dynamic_floor

        return dynamic_profit_threshold, threshold_adjustments

    def _calc_dynamic_threshold_live(self, base_profit_threshold, market_analysis, commission_buffer, slippage_buffer, volatility_buffer, tickers, strategy_result):
        """Расчет динамического порога для боевого режима."""
        dynamic_profit_threshold = base_profit_threshold
        threshold_adjustments = []

        dynamic_profit_threshold += commission_buffer
        threshold_adjustments.append({
            'reason': 'комиссии цикла',
            'value': commission_buffer
        })

        if slippage_buffer:
            dynamic_profit_threshold += slippage_buffer
            threshold_adjustments.append({
                'reason': 'запас на проскальзывание',
                'value': slippage_buffer
            })

        spread_adjustment = self._calculate_spread_adjustment(tickers)
        if spread_adjustment != 0:
            dynamic_profit_threshold += spread_adjustment
            threshold_adjustments.append({
                'reason': 'рыночный спред',
                'value': spread_adjustment
            })

        safe_strategy_context = getattr(self, 'last_strategy_context', {}) or {}
        context_spread = safe_strategy_context.get('average_spread_percent') if isinstance(safe_strategy_context, dict) else None
        if context_spread is not None:
            context_spread_adjustment = 0.0
            if context_spread > 1.0:
                context_spread_adjustment = min(0.08, context_spread / 120)
            elif context_spread < 0.25:
                context_spread_adjustment = -0.03

            if context_spread_adjustment:
                dynamic_profit_threshold += context_spread_adjustment
                threshold_adjustments.append({
                    'reason': 'средний спред контекста',
                    'value': context_spread_adjustment
                })

        context_imbalance = safe_strategy_context.get('orderbook_imbalance') if isinstance(safe_strategy_context, dict) else None
        if context_imbalance is not None:
            imbalance_strength = abs(context_imbalance)
            if imbalance_strength > 0.2:
                imbalance_adjustment = -0.02 * min(1.5, 1 + imbalance_strength)
                dynamic_profit_threshold += imbalance_adjustment
                threshold_adjustments.append({
                    'reason': 'дисбаланс стакана',
                    'value': imbalance_adjustment
                })

        if market_analysis['market_conditions'] == 'high_volatility':
            dynamic_profit_threshold += 0.03
            threshold_adjustments.append({'reason': 'высокая волатильность', 'value': 0.03})
        elif market_analysis['market_conditions'] == 'low_volatility':
            dynamic_profit_threshold -= 0.02
            threshold_adjustments.append({'reason': 'низкая волатильность', 'value': -0.02})

        if volatility_buffer:
            dynamic_profit_threshold += volatility_buffer
            threshold_adjustments.append({
                'reason': 'реальная волатильность',
                'value': volatility_buffer
            })

        if strategy_result:
            signal = (strategy_result.signal or '').lower()
            confidence = getattr(strategy_result, 'confidence', 0) or 0
            confidence = max(0.0, min(1.0, confidence))
            strategy_bias_map = {
                'increase_risk': -0.04,
                'long_bias': -0.03,
                'reduce_risk': 0.04,
                'short_bias': 0.03
            }
            if signal in strategy_bias_map:
                strategy_adjustment = strategy_bias_map[signal] * (1 + confidence)
                dynamic_profit_threshold += strategy_adjustment
                threshold_adjustments.append({
                    'reason': f'сигнал стратегии {signal}',
                    'value': strategy_adjustment
                })

            if getattr(strategy_result, 'name', '') == 'multi_indicator':
                extended_bias_map = {
                    'long': -0.05,
                    'short': 0.05,
                    'flat': 0.01,
                }
                bias_shift = extended_bias_map.get(signal, 0.0) * (1 + confidence)
                dynamic_profit_threshold += bias_shift
                threshold_adjustments.append({
                    'reason': f'мульти-индикаторный сигнал {signal}',
                    'value': bias_shift
                })

                meta = getattr(strategy_result, 'meta', {}) or {}
                atr_percent = meta.get('atr_percent', 0.0)
                if atr_percent > 1:
                    atr_adjustment = min(0.08, 0.02 * atr_percent)
                    dynamic_profit_threshold += atr_adjustment
                    threshold_adjustments.append({
                        'reason': 'высокий ATR',
                        'value': atr_adjustment
                    })
                elif atr_percent < 0.4 and signal == 'long':
                    atr_adjustment = -0.015 * (1 + confidence)
                    dynamic_profit_threshold += atr_adjustment
                    threshold_adjustments.append({
                        'reason': 'низкий ATR, подтверждение импульса',
                        'value': atr_adjustment
                    })

        if self.no_opportunity_cycles > 0:
            relax_step = getattr(self.config, 'EMPTY_CYCLE_RELAX_STEP', 0.01)
            relax_cap = getattr(self.config, 'EMPTY_CYCLE_RELAX_MAX', 0.05)
            empty_cycle_adjustment = -min(self.no_opportunity_cycles * relax_step, relax_cap)
            dynamic_profit_threshold += empty_cycle_adjustment
            threshold_adjustments.append({
                'reason': f'{self.no_opportunity_cycles} пустых циклов',
                'value': empty_cycle_adjustment
            })

        min_dynamic_floor = max(
            getattr(self.config, 'MIN_DYNAMIC_PROFIT_FLOOR', 0.0),
            base_profit_threshold + commission_buffer + slippage_buffer
        )
        if dynamic_profit_threshold < min_dynamic_floor:
            threshold_adjustments.append({
                'reason': 'динамический минимум',
                'value': min_dynamic_floor - dynamic_profit_threshold
            })
            dynamic_profit_threshold = min_dynamic_floor

        return dynamic_profit_threshold, threshold_adjustments

    def _build_market_dataframe(self, symbol=None, min_points=5):
        """Формирует список баров по одному символу или агрегирует несколько."""

        def _rows_for_symbol(sym):
            history = self.ohlcv_history.get(sym)
            if not history:
                return []

            if len(history['close']) < min_points:
                return []

            rows = []
            for ts, o, h, l, c, v in zip(
                history['timestamps'],
                history['open'],
                history['high'],
                history['low'],
                history['close'],
                history['volume']
            ):
                rows.append({
                    'timestamp': ts,
                    'symbol': sym,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v
                })
            return rows

        if symbol:
            return _rows_for_symbol(symbol)

        aggregated_rows = []
        for sym in sorted(self.ohlcv_history.keys()):
            aggregated_rows.extend(_rows_for_symbol(sym))

        if not aggregated_rows:
            # Возвращаем наилучший доступный набор данных даже если точек меньше минимума
            fallback_symbol = max(
                self.ohlcv_history.keys(),
                key=lambda s: len(self.ohlcv_history[s]['close']),
                default=None
            )
            if fallback_symbol:
                history = self.ohlcv_history[fallback_symbol]
                for ts, o, h, l, c, v in zip(
                    history['timestamps'],
                    history['open'],
                    history['high'],
                    history['low'],
                    history['close'],
                    history['volume']
                ):
                    aggregated_rows.append({
                        'timestamp': ts,
                        'symbol': fallback_symbol,
                        'open': o,
                        'high': h,
                        'low': l,
                        'close': c,
                        'volume': v
                    })

        aggregated_rows.sort(key=lambda row: row['timestamp'])
        return aggregated_rows

    def evaluate_strategies(self, market_data=None):
        if market_data is None:
            market_data = self._build_market_dataframe()
        if not market_data:
            return None

        closes_by_symbol = defaultdict(list)
        for row in market_data:
            close_value = row.get('close')
            if close_value is None:
                continue
            symbol_key = row.get('symbol', 'default')
            closes_by_symbol[symbol_key].append(close_value)

        closes = []
        if closes_by_symbol:
            closes = max(closes_by_symbol.values(), key=len)

        price_changes = []
        for previous, current in zip(closes, closes[1:]):
            if previous:
                price_change = ((current - previous) / previous) * 100
                price_changes.append(abs(price_change))

        volatility = rolling_mean(price_changes, window=20, min_periods=5)
        liquidity_values = [row['volume'] for row in market_data[-50:] if row['volume'] is not None]
        liquidity = mean(liquidity_values)

        micro = self._calculate_microstructure_metrics()

        market_context = {
            'volatility': float(volatility) if volatility is not None else 0.0,
            'liquidity': float(liquidity),
            'prepared_market': market_data,
            'average_spread_percent': micro['average_spread_percent'],
            'orderbook_imbalance': micro['orderbook_imbalance'],
        }

        # Сохраняем рассчитанный микроструктурный контекст для последующих решений
        self.last_strategy_context = {
            **market_context,
            'average_spread_percent': micro['average_spread_percent'],
            'orderbook_imbalance': micro['orderbook_imbalance'],
        }

        strategy_result = self.strategy_manager.evaluate(market_data, self.last_strategy_context)

        if strategy_result:
            logger.info(
                "🧠 Strategy %s selected signal=%s score=%.3f confidence=%.2f",
                strategy_result.name,
                strategy_result.signal,
                strategy_result.score,
                strategy_result.confidence
            )
        else:
            logger.debug("No strategy result available, fallback to triangular arbitrage")

        return strategy_result

    def detect_triangular_arbitrage(self, tickers, strategy_result=None):
        """Улучшенное обнаружение треугольного арбитража"""
        opportunities = []
        market_analysis = self.analyze_market_conditions()
        self._last_market_analysis = market_analysis
        self._last_candidates = []

        # Динамический порог прибыли с отдельной веткой для тестнета
        rejected_candidates = 0
        rejected_by_profit = 0
        rejected_by_liquidity = 0
        rejected_by_volatility = 0

        fee_rate = getattr(self.config, 'TRADING_FEE', 0)
        commission_buffer = max(0.0, fee_rate * 3 * 100)
        slippage_buffer = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
        volatility_component = max(0.0, market_analysis.get('overall_volatility', 0) or 0)
        volatility_buffer = min(
            0.2,
            volatility_component * getattr(self.config, 'VOLATILITY_PROFIT_MULTIPLIER', 0.05)
        )

        if getattr(self.config, 'TESTNET', False):
            base_profit_threshold = getattr(self.config, 'MIN_TRIANGULAR_PROFIT', 0.01)
            dynamic_profit_threshold, threshold_adjustments = self._calc_dynamic_threshold_testnet(
                base_profit_threshold,
                market_analysis,
                commission_buffer,
                slippage_buffer,
                volatility_buffer
            )
        else:
            # Динамический порог прибыли в зависимости от внешних факторов (боевой режим)
            base_profit_threshold = getattr(self.config, 'MIN_TRIANGULAR_PROFIT', 0.05)
            dynamic_profit_threshold, threshold_adjustments = self._calc_dynamic_threshold_live(
                base_profit_threshold,
                market_analysis,
                commission_buffer,
                slippage_buffer,
                volatility_buffer,
                tickers,
                strategy_result
            )

        dynamic_profit_threshold = max(0.0, dynamic_profit_threshold)
        self._last_dynamic_threshold = dynamic_profit_threshold
        logger.info(
            "Выбран порог прибыли %.4f%% (базовый %.4f%%, тестнет=%s, корректировки=%d)",
            dynamic_profit_threshold,
            base_profit_threshold,
            getattr(self.config, 'TESTNET', False),
            len(threshold_adjustments),
        )

        performance_optimizer = getattr(self, 'performance_optimizer', None)
        if performance_optimizer:
            prioritized_triangles = performance_optimizer.get_optimized_triangles()
            quick_filtered_triangles = performance_optimizer.parallel_check_liquidity(
                prioritized_triangles,
                tickers
            )
        else:
            prioritized_triangles = getattr(self.config, 'TRIANGULAR_PAIRS', [])
            quick_filtered_triangles = prioritized_triangles
        max_triangles = getattr(self.config, 'MAX_TRIANGLES_PER_CYCLE', 20)
        limited_triangles = quick_filtered_triangles[:max_triangles]
        self.optimized_triangles = limited_triangles
        rejected_by_liquidity += max(0, len(prioritized_triangles) - len(quick_filtered_triangles))

        for triangle in limited_triangles:
            triangle_name = triangle.get('name', 'triangle')
            try:
                # Проверяем доступность всех пар
                if not all(leg in tickers for leg in triangle['legs']):
                    continue

                # Быстрая проверка ликвидности по котировкам без обращения к стакану
                if not self._quick_triangle_liquidity_check(triangle, tickers):
                    rejected_by_liquidity += 1
                    continue

                # Проверяем ликвидность
                if not self._check_liquidity(triangle, tickers):
                    rejected_by_liquidity += 1
                    continue

                # Проверяем волатильность треугольника
                if not self._check_triangle_volatility(triangle):
                    rejected_by_volatility += 1
                    continue
                
                leg1, leg2, leg3 = triangle['legs']
                
                prices = {
                    leg1: tickers[leg1],
                    leg2: tickers[leg2], 
                    leg3: tickers[leg3]
                }
                
                # Расчет прибыли для всех направлений
                directions = [
                    self._calculate_direction(prices, triangle, 1),
                    self._calculate_direction(prices, triangle, 2),
                    self._calculate_direction(prices, triangle, 3)
                ]

                # Фильтрация явных ошибок построения пути
                valid_directions = [
                    d for d in directions
                    if d.get('path') and d.get('profit_percent', -100) > -90
                ]
                if not valid_directions:
                    rejected_candidates += 1
                    continue

                # Выбираем лучшее направление и пересчитываем прибыль перед сравнением с динамическим порогом
                best_direction = max(valid_directions, key=lambda x: x['profit_percent'])
                recalculated_profit = best_direction['profit_percent']
                if best_direction.get('path'):
                    base_currency = triangle.get('base_currency', 'USDT')
                    recalculated_profit = self._calculate_triangular_profit_path(
                        prices,
                        best_direction['path'],
                        base_currency,
                        trade_amount=getattr(self.config, 'TRADE_AMOUNT', None)
                    )
                    best_direction['profit_percent'] = recalculated_profit
                self._last_candidates.append({
                    'triangle': triangle,
                    'triangle_name': triangle_name,
                    'best_direction': best_direction,
                    'prices': prices
                })

                if recalculated_profit > dynamic_profit_threshold:
                    opportunity = {
                        'type': 'triangular',
                        'triangle_name': triangle_name,
                        'direction': best_direction['direction'],
                        'profit_percent': best_direction['profit_percent'],
                        'symbols': triangle['legs'],
                        'prices': prices,
                        'execution_path': best_direction['path'],
                        'timestamp': datetime.now(),
                        'market_conditions': market_analysis['market_conditions'],
                        'priority': triangle.get('priority', 999),
                        'base_currency': triangle.get('base_currency', 'USDT')
                    }
                    
                    historical_success = self.triangle_stats[triangle_name]['success_rate']
                    if historical_success > 0.7:  # Повышаем приоритет успешных треугольников
                        opportunity['profit_percent'] += 0.0

                    opportunities.append(opportunity)

                    self.triangle_stats[triangle_name]['opportunities_found'] += 1

                    logger.info(f"🔺 {triangle['name']} - Direction {best_direction['direction']} - "
                              f"Profit: {best_direction['profit_percent']:.4f}% - "
                              f"Market: {market_analysis['market_conditions']}")
                    
                    logger.info(f"🔺 {triangle_name} - Direction {best_direction['direction']} - "
                              f"Profit: {best_direction['profit_percent']:.4f}% - "
                              f"Market: {market_analysis['market_conditions']}")

                else:
                    rejected_candidates += 1
                    rejected_by_profit += 1

            except Exception as e:
                logger.error(f"Error in triangle {triangle_name}: {str(e)}")

        # Сортировка по прибыльности
        opportunities.sort(key=lambda x: x['profit_percent'], reverse=True)

        total_candidates = len(self._last_candidates)
        logger.info(
            "Итоги отбора: кандидатов=%d, принято=%d, отклонено=%d (прибыль=%d, ликвидность=%d, волатильность=%d)",
            total_candidates,
            len(opportunities),
            rejected_candidates,
            rejected_by_profit,
            rejected_by_liquidity,
            rejected_by_volatility,
        )
        if hasattr(self, 'monitor') and hasattr(self.monitor, 'log_profit_threshold'):
            self.monitor.log_profit_threshold(
                final_threshold=dynamic_profit_threshold,
                rejected_candidates=rejected_candidates,
                base_threshold=base_profit_threshold,
                adjustments=threshold_adjustments,
                market_conditions=market_analysis['market_conditions'],
                total_candidates=total_candidates
            )
        return opportunities

    def _calculate_microstructure_metrics(self, window: int = 20):
        """Подсчет среднего спреда и дисбаланса стакана на коротком окне."""
        spreads = []
        imbalances = []

        for symbol, history in self.price_history.items():
            if history['spreads']:
                spreads.extend(list(history['spreads'])[-window:])

            bid_vols = list(history.get('bid_volumes', []))[-window:]
            ask_vols = list(history.get('ask_volumes', []))[-window:]
            for bid_vol, ask_vol in zip(bid_vols, ask_vols):
                total = bid_vol + ask_vol
                if total > 0:
                    imbalances.append((bid_vol - ask_vol) / total)

        avg_spread = mean(spreads) if spreads else 0.0
        avg_imbalance = mean(imbalances) if imbalances else 0.0

        return {
            'average_spread_percent': float(avg_spread),
            'orderbook_imbalance': float(avg_imbalance)
        }

    def _calculate_spread_adjustment(self, tickers):
        """Корректировка порога прибыли в зависимости от среднего спреда ног."""
        spreads = []
        for triangle in self.config.TRIANGULAR_PAIRS:
            legs = triangle['legs']
            if all(leg in tickers for leg in legs):
                for leg in legs:
                    data = tickers[leg]
                    bid = data.get('bid', 0)
                    ask = data.get('ask', 0)
                    if bid > 0 and ask > 0:
                        spreads.append(((ask - bid) / bid) * 100)

        if not spreads:
            return 0.0

        avg_spread = mean(spreads)

        # Чем шире средний спред, тем жестче должен быть порог, и наоборот
        if avg_spread > 1.0:
            return min(0.1, avg_spread / 100)
        if avg_spread < 0.2:
            return -0.05
        return 0.0

    def _calculate_direction(self, prices, triangle, direction):
        """Расчет прибыли для конкретного направления"""
        leg1, leg2, leg3 = triangle['legs']
        base_currency = triangle.get('base_currency', 'USDT')

        direction_sequences = self._prepare_direction_sequences([leg1, leg2, leg3], direction)
        path = None

        for sequence in direction_sequences:
            path = self._build_universal_path(
                sequence,
                base_currency,
                triangle.get('name', 'unknown'),
                direction
            )
            if path:
                break

        if not path:
            profit = -100
        else:
            profit = self._calculate_triangular_profit_path(
                prices,
                path,
                base_currency,
                trade_amount=getattr(self.config, 'TRADE_AMOUNT', None)
            )

        return {
            'direction': direction,
            'profit_percent': profit,
            'path': path
        }

    def _prepare_direction_sequences(self, legs, direction):
        """Подбор последовательностей ног в зависимости от направления"""

        def _rotations(sequence):
            base = list(sequence)
            return [base[i:] + base[:i] for i in range(len(base))]

        if direction == 1:
            return _rotations(legs)
        if direction == 2:
            reversed_legs = list(reversed(legs))
            return _rotations(reversed_legs)

        unique_sequences = []
        for perm in permutations(legs):
            perm_list = list(perm)
            if perm_list not in unique_sequences:
                unique_sequences.append(perm_list)
        return unique_sequences

    def _build_universal_path(self, legs_sequence, base_currency, triangle_name, direction):
        """Универсальное построение пути на основе текущей валюты"""
        current_asset = base_currency
        path = []
        remaining_symbols = list(legs_sequence)
        max_iterations = len(remaining_symbols) * 3 or 3
        iterations = 0

        while remaining_symbols and iterations < max_iterations:
            step_found = False
            iterations += 1

            for symbol in list(remaining_symbols):
                base_cur, quote_cur = self._get_symbol_currencies(symbol)

                logger.debug(
                    f"Текущая валюта: {current_asset}, рассматриваем символ {symbol}"
                )

                if current_asset == quote_cur:
                    path.append({'symbol': symbol, 'side': 'Buy', 'price_type': 'ask'})
                    current_asset = base_cur
                    remaining_symbols.remove(symbol)
                    step_found = True
                    logger.debug(
                        f"Добавлен шаг покупки {symbol}, новая валюта {current_asset}, осталось {len(remaining_symbols)} символов"
                    )
                    break
                if current_asset == base_cur:
                    path.append({'symbol': symbol, 'side': 'Sell', 'price_type': 'bid'})
                    current_asset = quote_cur
                    remaining_symbols.remove(symbol)
                    step_found = True
                    logger.debug(
                        f"Добавлен шаг продажи {symbol}, новая валюта {current_asset}, осталось {len(remaining_symbols)} символов"
                    )
                    break

            if not step_found:
                logger.debug(
                    f"Не удалось подобрать шаг с текущей валютой {current_asset}, пытаемся аварийный шаг"
                )

                fallback_symbol = None
                fallback_side = None
                fallback_next_asset = None

                for symbol in list(remaining_symbols):
                    base_cur, quote_cur = self._get_symbol_currencies(symbol)
                    if base_cur == base_currency:
                        fallback_symbol = symbol
                        fallback_side = 'Sell'
                        fallback_next_asset = quote_cur
                        break
                    if quote_cur == base_currency:
                        fallback_symbol = symbol
                        fallback_side = 'Buy'
                        fallback_next_asset = base_cur
                        break

                if fallback_symbol:
                    path.append({
                        'symbol': fallback_symbol,
                        'side': fallback_side,
                        'price_type': 'bid' if fallback_side == 'Sell' else 'ask'
                    })
                    remaining_symbols.remove(fallback_symbol)
                    current_asset = fallback_next_asset
                    logger.debug(
                        f"Аварийно добавлен шаг {fallback_side} для {fallback_symbol}, новая валюта {current_asset}, осталось {len(remaining_symbols)} символов"
                    )
                    continue

                logger.warning(
                    f"Невозможно построить путь для {triangle_name} (направление {direction}): "
                    f"текущая валюта {current_asset} не совпадает ни с одной котируемой валютой"
                )
                return None

        if iterations >= max_iterations and remaining_symbols:
            logger.warning(
                f"Превышен лимит итераций при построении пути для {triangle_name} (направление {direction})"
            )
            return None

        if current_asset != base_currency:
            logger.warning(
                f"Путь для {triangle_name} (направление {direction}) не возвращает базовую валюту {base_currency}"
            )
            return None

        if remaining_symbols:
            logger.warning(
                f"Путь для {triangle_name} (направление {direction}) построен не полностью, осталось {len(remaining_symbols)} символов"
            )
            return None

        return path

    def _refresh_quote_suffix_cache(self):
        """Формирует список известных окончаний котировок с учетом данных биржи."""
        quotes = set(self.config.KNOWN_QUOTES)

        try:
            available_crosses = self.config.AVAILABLE_CROSSES
        except Exception:  # pragma: no cover - на случай сетевых ошибок
            available_crosses = {}

        for base_currency, symbols in (available_crosses or {}).items():
            for cross_symbol in symbols:
                if cross_symbol.startswith(base_currency):
                    quote = cross_symbol[len(base_currency):]
                    if quote:
                        quotes.add(quote)
                elif cross_symbol.endswith(base_currency):
                    quote = cross_symbol[:-len(base_currency)]
                    if quote:
                        quotes.add(quote)

        self._quote_suffix_cache = sorted(filter(None, quotes), key=len, reverse=True)

    def _get_symbol_currencies(self, symbol):
        """Определение базовой и котируемой валюты с максимальным использованием справочников."""
        base, quote = self.config._split_symbol(symbol)
        if base and quote:
            return base, quote

        if not hasattr(self, '_quote_suffix_cache') or self._quote_suffix_cache is None:
            self._refresh_quote_suffix_cache()

        for quote_candidate in self._quote_suffix_cache:
            if symbol.endswith(quote_candidate) and len(symbol) > len(quote_candidate):
                return symbol[:-len(quote_candidate)], quote_candidate

        # Резервный случай для неизвестных пар: делим тикер пополам
        midpoint = len(symbol) // 2
        return symbol[:midpoint], symbol[midpoint:]

    def _calculate_triangular_profit_path(self, prices, path, base_currency, trade_amount=None):
        """Расчет прибыли по пути с учётом доступных объёмов, комиссий и проскальзывания"""
        try:
            fee_rate = getattr(self.config, 'TRADING_FEE', 0)
            slippage_buffer = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
            initial_amount = float(trade_amount or getattr(self.config, 'TRADE_AMOUNT', 0) or 1000.0)
            current_amount = initial_amount
            current_asset = base_currency

            def _normalize_levels(levels):
                """Нормализация уровней стакана в формат со стандартными ключами"""
                normalized = []
                for level in levels or []:
                    if isinstance(level, dict):
                        price = self._safe_float(level.get('price'))
                        size = self._safe_float(level.get('size'))
                    elif isinstance(level, (list, tuple)) and len(level) >= 2:
                        price = self._safe_float(level[0])
                        size = self._safe_float(level[1])
                    else:
                        continue

                    if price > 0 and size > 0:
                        normalized.append({'price': price, 'size': size})
                return normalized

            def _estimate_execution(side, price_data, amount, price_type):
                """Оценка средней цены исполнения с проверкой объёмов и проскальзывания"""
                best_price = price_data['ask'] if price_type == 'ask' else price_data['bid']
                if best_price <= 0:
                    return None, None

                best_volume = self._safe_float(
                    price_data.get('ask_size') if price_type == 'ask' else price_data.get('bid_size')
                )

                depth_key = 'asks' if price_type == 'ask' else 'bids'
                depth = _normalize_levels(price_data.get(depth_key) or price_data.get('order_book', {}).get(depth_key))

                if side == 'Buy':
                    remaining_quote = amount
                    acquired_base = 0.0

                    def _consume(price, size):
                        nonlocal remaining_quote, acquired_base
                        if remaining_quote <= 0:
                            return
                        max_base_for_quote = remaining_quote / price
                        taken_base = min(size, max_base_for_quote)
                        acquired_base += taken_base
                        remaining_quote -= taken_base * price

                    if best_volume > 0:
                        _consume(best_price, best_volume)

                    if remaining_quote > 0 and depth:
                        for level in depth:
                            _consume(level['price'], level['size'])
                            if remaining_quote <= 0:
                                break

                    if remaining_quote > 0 and acquired_base > 0:
                        worse_price = best_price * (1 + slippage_buffer)
                        acquired_base += (remaining_quote / worse_price)
                        remaining_quote = 0

                    if acquired_base <= 0:
                        return None, None

                    spent_quote = amount - remaining_quote
                    effective_price = spent_quote / acquired_base if acquired_base > 0 else None
                    return effective_price, acquired_base

                remaining_base = amount
                realized_quote = 0.0

                def _sell_consume(price, size):
                    nonlocal remaining_base, realized_quote
                    if remaining_base <= 0:
                        return
                    sold_base = min(size, remaining_base)
                    realized_quote += sold_base * price
                    remaining_base -= sold_base

                if best_volume > 0:
                    _sell_consume(best_price, best_volume)

                if remaining_base > 0 and depth:
                    for level in depth:
                        _sell_consume(level['price'], level['size'])
                        if remaining_base <= 0:
                            break

                if remaining_base > 0:
                    worse_price = best_price * (1 - slippage_buffer)
                    realized_quote += remaining_base * worse_price
                    remaining_base = 0

                effective_price = realized_quote / amount if amount > 0 else None
                return effective_price, amount

            for step in path:
                symbol = step['symbol']
                price_data = prices[symbol]
                symbol_base, symbol_quote = self._get_symbol_currencies(symbol)

                if step['side'] == 'Buy':
                    if current_asset != symbol_quote:
                        logger.warning(
                            f"Путь отклонен: покупка {symbol} требует {symbol_quote},"
                            f" но текущая валюта {current_asset}"
                        )
                        return -100

                    price, acquired = _estimate_execution('Buy', price_data, current_amount, step['price_type'])
                    if not price or not acquired:
                        return -100

                    quantity = acquired * (1 - fee_rate)
                    current_amount = quantity
                    current_asset = symbol_base
                else:  # Sell
                    if current_asset != symbol_base:
                        logger.warning(
                            f"Путь отклонен: продажа {symbol} требует {symbol_base},"
                            f" но текущая валюта {current_asset}"
                        )
                        return -100

                    price, executed_amount = _estimate_execution('Sell', price_data, current_amount, step['price_type'])
                    if not price or not executed_amount:
                        return -100

                    proceeds = (executed_amount * price) * (1 - fee_rate)
                    current_amount = proceeds
                    current_asset = symbol_quote

            profit_percent = ((current_amount - initial_amount) / initial_amount) * 100
            return profit_percent

        except (ZeroDivisionError, ValueError) as e:
            logger.debug(f"Profit calculation error: {str(e)}")
            return -100

    def _quick_triangle_liquidity_check(self, triangle, tickers):
        """Упрощенная проверка ликвидности на основе bid/ask и спреда"""
        max_spread = getattr(self.config, 'MAX_SPREAD_PERCENT', 10)
        planned_amount = getattr(self.config, 'TRADE_AMOUNT', 0) or getattr(self.config, 'MIN_LIQUIDITY', 0)
        minimum_threshold = max(getattr(self.config, 'MIN_LIQUIDITY', 0) * 0.5, planned_amount * 0.25)

        for symbol in triangle['legs']:
            ticker = tickers.get(symbol)
            if not ticker:
                logger.debug("Пропускаем %s из-за отсутствия тикера %s", triangle.get('name', 'triangle'), symbol)
                return False

            bid = float(ticker.get('bid', 0) or 0)
            ask = float(ticker.get('ask', 0) or 0)

            if bid <= 0 or ask <= 0 or ask < bid:
                logger.debug("Невалидные котировки для %s: bid=%s, ask=%s", symbol, bid, ask)
                return False

            spread_percent = ((ask - bid) / bid) * 100
            if spread_percent > max_spread:
                logger.debug(
                    "Слишком широкий спред для %s: %.2f%% (лимит %.2f%%)",
                    symbol,
                    spread_percent,
                    max_spread
                )
                return False

            bid_volume = float(
                ticker.get('bid_size')
                or ticker.get('bid_qty')
                or ticker.get('bid_volume')
                or ticker.get('bidVol')
                or 0
            )
            ask_volume = float(
                ticker.get('ask_size')
                or ticker.get('ask_qty')
                or ticker.get('ask_volume')
                or ticker.get('askVol')
                or 0
            )

            available_notional = max(bid * bid_volume, ask * ask_volume)
            if available_notional and available_notional < minimum_threshold:
                logger.debug(
                    "Недостаточный объём по %s: доступно %.2f, требуется минимум %.2f",
                    symbol,
                    available_notional,
                    minimum_threshold
                )
                return False

        return True

    def _check_liquidity(self, triangle, tickers):
        """Проверка ликвидности для треугольника"""
        spread_limit = 50 if self.config.TESTNET else self.config.MAX_SPREAD_PERCENT
        depth_levels = getattr(self.config, 'ORDERBOOK_DEPTH_LEVELS', 5)
        max_impact = getattr(self.config, 'MAX_ORDERBOOK_IMPACT', 0.25)
        planned_amount = getattr(self.config, 'TRADE_AMOUNT', 0) or self.config.MIN_LIQUIDITY

        # Проверяем глубокую ликвидность по каждому плечу
        def _depth_value(levels):
            # Суммируем денежный объём верхних уровней стакана
            return sum(
                max(0.0, level.get('price', 0)) * max(0.0, level.get('size', 0))
                for level in levels[:depth_levels]
            )

        for symbol in triangle['legs']:
            if symbol not in tickers:
                logger.warning(
                    "Пропускаем треугольник %s из-за отсутствия тикера %s",
                    triangle['name'],
                    symbol
                )
                return False

            bid, ask = tickers[symbol]['bid'], tickers[symbol]['ask']
            if bid <= 0 or ask <= 0:
                logger.warning(
                    "Пропускаем тикер %s из-за нулевых значений: bid=%s, ask=%s",
                    symbol,
                    bid,
                    ask
                )
                return False

            # Проверка спреда
            spread = ((ask - bid) / bid) * 100
            if spread > spread_limit:
                logger.debug(
                    "Слишком широкий спред для %s: %.2f%% (лимит %.2f%%)",
                    symbol,
                    spread,
                    spread_limit
                )
                return False

            # Получаем стакан и оцениваем глубину
            order_book = self.client.get_order_book(symbol, depth_levels)
            total_bid_value = _depth_value(order_book.get('bids', []))
            total_ask_value = _depth_value(order_book.get('asks', []))
            available_liquidity = min(total_bid_value, total_ask_value)

            if available_liquidity < self.config.MIN_LIQUIDITY:
                logger.debug(
                    "Недостаточная ликвидность в стакане %s: доступно %.2f USDT (порог %.2f)",
                    symbol,
                    available_liquidity,
                    self.config.MIN_LIQUIDITY
                )
                return False

            if planned_amount > available_liquidity * max_impact:
                logger.debug(
                    "Планируемый объем %.2f USDT превышает допустимую долю стакана %.2f%% для %s",
                    planned_amount,
                    max_impact * 100,
                    symbol
                )
                return False

        return True

    def _check_triangle_volatility(self, triangle):
        """Проверка волатильности треугольника"""
        volatilities = []
        for symbol in triangle['legs']:
            if (symbol in self.volatility_data and
                self.volatility_data[symbol]['short_term']):
                vol = mean(self.volatility_data[symbol]['short_term'])
                volatilities.append(vol)

        if volatilities:
            avg_volatility = mean(volatilities)
            # Фильтруем слишком волатильные треугольники
            return avg_volatility < 5.0  # Максимум 5% волатильность
        
        return True

    def calculate_advanced_trade(self, opportunity, balance_usdt):
        """Расчет параметров сделки с улучшенным управлением рисками"""
        try:
            # Динамический расчет суммы на основе волатильности
            base_amount = min(self.config.TRADE_AMOUNT, balance_usdt * 0.7)
            
            # Корректировка суммы в зависимости от рыночных условий
            if opportunity['market_conditions'] == 'high_volatility':
                trade_amount = base_amount * 0.5  # Уменьшаем размер при высокой волатильности
            elif opportunity['market_conditions'] == 'low_volatility':
                trade_amount = base_amount * 1.2  # Увеличиваем при низкой волатильности
            else:
                trade_amount = base_amount
            
            if trade_amount < 5:  # Минимальная сумма
                return None
            
            path = opportunity['execution_path']
            direction = opportunity['direction']
            
            trade_plan = {
                'type': 'triangular',
                'triangle_name': opportunity['triangle_name'],
                'direction': direction,
                'initial_amount': trade_amount,
                'base_currency': opportunity.get('base_currency', 'USDT'),
                'estimated_profit_usdt': trade_amount * (opportunity['profit_percent'] / 100),
                'market_conditions': opportunity['market_conditions'],
                'timestamp': datetime.now()
            }
            
            # Расчет шагов на основе пути исполнения
            current_amount = trade_amount
            steps = {}
            
            for i, step in enumerate(path):
                symbol = step['symbol']
                price_data = opportunity['prices'][symbol]
                price = price_data['ask'] if step['price_type'] == 'ask' else price_data['bid']
                
                if step['side'] == 'Buy':
                    quantity = current_amount / price
                    # Учитываем комиссию при покупке
                    quantity *= (1 - self.config.TRADING_FEE)
                    steps[f'step{i+1}'] = {
                        'symbol': symbol,
                        'side': 'Buy',
                        'amount': quantity,
                        'price': price,
                        'type': 'Limit',
                        'calculated_amount': quantity
                    }
                    current_amount = quantity
                else:  # Sell
                    amount = current_amount  # Текущее количество для продажи
                    usd_value = amount * price
                    # Учитываем комиссию при продаже
                    usd_value *= (1 - self.config.TRADING_FEE)
                    steps[f'step{i+1}'] = {
                        'symbol': symbol,
                        'side': 'Sell',
                        'amount': amount,
                        'price': price,
                        'type': 'Limit',
                        'calculated_amount': amount
                    }
                    current_amount = usd_value
            
            trade_plan.update(steps)
            return trade_plan
            
        except Exception as e:
            logger.error(f"Error calculating advanced trade: {str(e)}")
            return None

    def execute_triangular_arbitrage(self, opportunity, trade_plan):
        """Мгновенное выполнение треугольного арбитража на текущих котировках"""
        logger.info(f"🔺 Начало исполнения треугольного арбитража: {opportunity['triangle_name']}")

        start_time = datetime.now()

        try:
            # Немедленно запрашиваем актуальные тикеры по всем ногам
            current_tickers = self.client.get_tickers(opportunity['symbols'])
            if not current_tickers:
                logger.warning("❌ Не удалось получить тикеры для проверки возможности")
                triangle_name = opportunity.get('triangle_name')
                if triangle_name in self.triangle_stats:
                    self.triangle_stats[triangle_name]['failures'] += 1
                    self._update_triangle_success_rate(triangle_name)
                return False

            # Перепроверяем сохранение возможности на свежих ценах
            if not self._validate_opportunity_still_exists(opportunity, current_tickers):
                logger.warning("❌ Возможность исчезла на момент проверки тикеров")
                return False

            # Быстрая проверка ликвидности и спредов без ожидания стаканов
            if not self._quick_liquidity_check(opportunity, trade_plan, current_tickers):
                logger.warning("❌ Быстрая проверка ликвидности не пройдена, пропускаем исполнение")
                triangle_name = opportunity['triangle_name']
                self.triangle_stats[triangle_name]['failures'] += 1
                self._update_triangle_success_rate(triangle_name)
                return False

            recalculated_profit = self._recalculate_trade_plan_profit(
                trade_plan,
                current_tickers,
                opportunity
            )

            if recalculated_profit is None:
                logger.warning("❌ Не удалось пересчитать прибыль на актуальных ценах, исполнение отменено")
                return False

            if recalculated_profit <= 0:
                logger.info(
                    "📉 Обновленный расчёт прибыли %.6f USDT не удовлетворяет требованиям, сделка отменена",
                    recalculated_profit
                )
                return False

            # Мгновенно исполняем торговый план по подтвержденным ценам
            trade_result = self.real_trader.execute_arbitrage_trade(trade_plan)
            execution_time = (datetime.now() - start_time).total_seconds()

            actual_profit = trade_result.get(
                'total_profit',
                trade_plan.get('estimated_profit_usdt', 0)
            ) if trade_result else 0

            if trade_result:
                triangle_name = opportunity['triangle_name']
                self.triangle_stats[triangle_name]['executed_trades'] += 1
                self.triangle_stats[triangle_name]['total_profit'] += trade_plan['estimated_profit_usdt']
                self.triangle_stats[triangle_name]['last_execution'] = datetime.now()
                self._update_triangle_success_rate(triangle_name)

                logger.info(
                    "✅ Треугольный арбитраж выполнен успешно! Время: %.2fs, Прибыль: %.4f USDT",
                    execution_time,
                    trade_plan['estimated_profit_usdt']
                )

                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': opportunity['triangle_name'],
                    'type': 'triangular',
                    'profit': actual_profit,
                    'profit_percent': opportunity['profit_percent'],
                    'direction': opportunity['direction'],
                    'execution_time': execution_time,
                    'market_conditions': opportunity['market_conditions'],
                    'triangle_stats': self.triangle_stats[triangle_name],
                    'trade_plan': trade_plan,
                    'results': trade_result.get('results', []),
                    'total_profit': actual_profit,
                    'details': {
                        'triangle': opportunity['triangle_name'],
                        'symbols': opportunity['symbols'],
                        'direction': opportunity['direction'],
                        'initial_amount': trade_plan['initial_amount'],
                        'execution_path': opportunity['execution_path'],
                        'real_executed': True
                    }
                }

                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor.track_trade(trade_record)

                self._record_trade(
                    opportunity,
                    trade_plan,
                    trade_result.get('results', []),
                    actual_profit
                )
                return True

            logger.error("❌ Исполнение треугольного арбитража завершилось ошибкой")
            triangle_name = opportunity['triangle_name']
            self.triangle_stats[triangle_name]['failures'] += 1
            self._update_triangle_success_rate(triangle_name)
            return False

        except Exception as e:
            logger.error(f"🔥 Критическая ошибка при исполнении треугольного арбитража: {str(e)}", exc_info=True)
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'notify_alert'):
                self.monitor.notify_alert(f"Ошибка треугольного арбитража: {str(e)}", "critical")
            return False

    def _quick_liquidity_check(self, opportunity, trade_plan, current_tickers):
        """Быстрая оценка валидности бид-аск и спреда по всем ногам"""
        protective_spread = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
        max_spread = getattr(self.config, 'MAX_SPREAD_PERCENT', 10)

        for i, step in enumerate(opportunity['execution_path']):
            plan_key = f"step{i+1}"
            order_details = trade_plan.get(plan_key)
            ticker = current_tickers.get(step['symbol']) if current_tickers else None

            if not order_details or not ticker:
                logger.debug("Недостаточно данных для оценки шага %s", plan_key)
                return False

            bid = float(ticker.get('bid', 0) or 0)
            ask = float(ticker.get('ask', 0) or 0)

            if bid <= 0 or ask <= 0 or ask < bid:
                logger.debug("Невалидные котировки для %s: bid=%s, ask=%s", step['symbol'], bid, ask)
                return False

            spread_percent = ((ask - bid) / bid) * 100
            if spread_percent > max_spread:
                logger.debug("Спред %.4f%% для %s превышает лимит %.4f%%", spread_percent, step['symbol'], max_spread)
                return False

            base_price = ask if order_details['side'] == 'Buy' else bid
            adjusted_price = (
                base_price * (1 + protective_spread)
                if order_details['side'] == 'Buy'
                else base_price * (1 - protective_spread)
            )

            trade_plan[plan_key]['price'] = adjusted_price
            trade_plan[plan_key]['book_validation'] = {
                'bid': bid,
                'ask': ask,
                'spread_percent': spread_percent,
                'checked_at': datetime.now()
            }

        return True

    def _recalculate_trade_plan_profit(self, trade_plan, current_tickers, opportunity):
        """Пересчет ожидаемой прибыли по актуальным ценам с учётом комиссий и проскальзывания"""
        initial_amount = float(trade_plan.get('initial_amount', 0) or 0)
        if initial_amount <= 0:
            logger.warning("⚠️ Некорректный стартовый объём для пересчёта прибыли")
            return None

        fee_rate = getattr(self.config, 'TRADING_FEE', 0)
        protective_spread = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
        current_amount = initial_amount
        current_asset = opportunity.get('base_currency', 'USDT')

        for i, step in enumerate(opportunity['execution_path']):
            plan_key = f"step{i+1}"
            ticker = current_tickers.get(step['symbol'])
            plan_step = trade_plan.get(plan_key)

            if not ticker or not plan_step:
                logger.debug("Недостаточно данных для пересчёта шага %s", plan_key)
                return None

            bid = float(ticker.get('bid', 0) or 0)
            ask = float(ticker.get('ask', 0) or 0)

            if bid <= 0 or ask <= 0:
                logger.debug("Невалидные котировки для пересчёта %s: bid=%s, ask=%s", step['symbol'], bid, ask)
                return None

            base_currency, quote_currency = self._get_symbol_currencies(step['symbol'])
            if not base_currency or not quote_currency:
                logger.debug("Не удалось определить валюты тикера %s", step['symbol'])
                return None
            if step['side'] == 'Buy':
                if current_asset != quote_currency:
                    logger.debug("Несогласованная валюта шага %s: ожидается %s, текущая %s", plan_key, quote_currency, current_asset)
                    return None

                base_price = ask if step['price_type'] == 'ask' else bid
                price = base_price * (1 + protective_spread)
                quantity = (current_amount / price) * (1 - fee_rate)

                trade_plan[plan_key]['price'] = price
                trade_plan[plan_key]['calculated_amount'] = quantity
                current_amount = quantity
                current_asset = base_currency
            else:
                if current_asset != base_currency:
                    logger.debug("Несогласованная валюта шага %s: ожидается %s, текущая %s", plan_key, base_currency, current_asset)
                    return None

                base_price = bid if step['price_type'] == 'bid' else ask
                price = base_price * (1 - protective_spread)
                proceeds = (current_amount * price) * (1 - fee_rate)

                trade_plan[plan_key]['price'] = price
                trade_plan[plan_key]['calculated_amount'] = current_amount
                current_amount = proceeds
                current_asset = quote_currency

        recalculated_profit = current_amount - initial_amount
        trade_plan['estimated_profit_usdt'] = recalculated_profit
        trade_plan['recalculated_at'] = datetime.now()
        return recalculated_profit

    def _update_triangle_success_rate(self, triangle_name):
        """Пересчитывает успешность треугольника с учетом всех попыток."""
        stats = self.triangle_stats.get(triangle_name)
        if not stats:
            return

        total_attempts = stats['executed_trades'] + stats.get('failures', 0)
        if total_attempts == 0:
            stats['success_rate'] = 0
            return

        stats['success_rate'] = stats['executed_trades'] / total_attempts

    def _validate_opportunity_still_exists(self, opportunity, current_tickers):
        """Проверка, что арбитражная возможность все еще существует"""
        try:
            # Пересчитываем прибыль с текущими ценами
            recalculated_profit = self._calculate_direction(
                current_tickers,
                {
                    'name': opportunity['triangle_name'],
                    'legs': opportunity['symbols'],
                    'base_currency': opportunity.get('base_currency', 'USDT')
                },
                opportunity['direction']
            )['profit_percent']
            
            # Возможность все еще существует если прибыль > 50% от исходной
            return recalculated_profit > (opportunity['profit_percent'] * 0.5)
        except Exception:
            return False

    def get_triangle_performance_report(self):
        """Генерация отчета по эффективности треугольников"""
        report = {
            'timestamp': datetime.now(),
            'total_opportunities_found': sum(stats['opportunities_found'] for stats in self.triangle_stats.values()),
            'total_executed_trades': sum(stats['executed_trades'] for stats in self.triangle_stats.values()),
            'total_profit': sum(stats['total_profit'] for stats in self.triangle_stats.values()),
            'triangle_details': {}
        }
        
        for triangle_name, stats in self.triangle_stats.items():
            report['triangle_details'][triangle_name] = {
                'opportunities_found': stats['opportunities_found'],
                'executed_trades': stats['executed_trades'],
                'success_rate': stats['success_rate'],
                'total_profit': stats['total_profit'],
                'last_execution': stats['last_execution'],
                'efficiency': stats['executed_trades'] / stats['opportunities_found'] if stats['opportunities_found'] > 0 else 0
            }
        
        return report

    def _record_trade(self, opportunity, trade_plan, orders, total_profit=None):
        """Запись информации о сделке в историю"""
        trade_record = {
            'timestamp': datetime.now(),
            'type': opportunity['type'],
            'triangle_name': opportunity['triangle_name'],
            'profit_percent': opportunity['profit_percent'],
            'estimated_profit_usdt': trade_plan.get('estimated_profit_usdt', 0),
            'actual_profit_usdt': total_profit if total_profit is not None else trade_plan.get('estimated_profit_usdt', 0),
            'direction': opportunity['direction'],
            'market_conditions': opportunity['market_conditions'],
            'orders': orders,
            'opportunity': opportunity
        }
        self.trade_history.append(trade_record)

        # Ограничиваем длину истории
        if len(self.trade_history) > 2000:
            self.trade_history.pop(0)

    def get_strategy_status(self):
        """Возвращает актуальный режим и состояние стратегий."""
        return {
            'mode': getattr(self.config, 'STRATEGY_MODE', 'adaptive'),
            'active': self.strategy_manager.get_active_strategy_name(),
            'context': self.last_strategy_context,
            'strategies': self.strategy_manager.get_strategy_snapshot()
        }

    def detect_opportunities(self):
        """Основной метод для обнаружения арбитражных возможностей"""
        # Получаем все необходимые символы
        all_symbols = set(self.config.SYMBOLS)
        for triangle in self.config.TRIANGULAR_PAIRS:
            for symbol in triangle['legs']:
                all_symbols.add(symbol)

        tickers = self.client.get_tickers(list(all_symbols))

        if not tickers:
            logger.warning("❌ No ticker data received")
            return []

        # Обновляем рыночные данные
        self.update_market_data(tickers)

        # Сохраняем последние цены для визуализации
        self.last_tickers = tickers

        market_data = self._build_market_dataframe()

        # Оцениваем стратегии уже на свежих данных
        strategy_result = self.evaluate_strategies(market_data)
        active_strategy_name = self.strategy_manager.get_active_strategy_name()
        logger.info(
            "⚙️ Strategy mode=%s | Active=%s",
            self.config.STRATEGY_MODE,
            active_strategy_name
        )

        # Обнаружение треугольного арбитража
        opportunities = self.detect_triangular_arbitrage(tickers, strategy_result)

        if opportunities:
            self.no_opportunity_cycles = 0
        else:
            self.no_opportunity_cycles += 1
            logger.debug(
                "Нет треугольных возможностей %s циклов подряд",
                self.no_opportunity_cycles
            )

            if self.no_opportunity_cycles >= 3:
                aggressive = self._generate_aggressive_opportunities_from_cache(strategy_result)
                if aggressive:
                    logger.warning(
                        "⚡ Активирован агрессивный режим: добавлено %s синтетических возможностей",
                        len(aggressive)
                    )
                    opportunities.extend(aggressive)
                    self.no_opportunity_cycles = 0

        if strategy_result:
            for opportunity in opportunities:
                opportunity['strategy'] = strategy_result.name
                opportunity['strategy_signal'] = strategy_result.signal
                opportunity['strategy_confidence'] = strategy_result.confidence
        else:
            for opportunity in opportunities:
                opportunity['strategy'] = active_strategy_name
                opportunity['strategy_signal'] = 'neutral'
                opportunity['strategy_confidence'] = 0

        # Логируем результаты
        if opportunities:
            logger.info(f"🎯 Found {len(opportunities)} triangular arbitrage opportunities:")
            for i, opp in enumerate(opportunities[:5], 1):  # Показываем топ-5
                logger.info(f"   {i}. {opp['triangle_name']} - {opp['profit_percent']:.4f}% - "
                          f"Direction: {opp['direction']}")
        else:
            logger.info("🔍 No arbitrage opportunities found")
        
        return opportunities

    def _calculate_aggressive_alpha(self, strategy_result, candidate):
        """Расчет надбавки к ожидаемой прибыли на основе стратегий"""
        base_boost = 0.1

        market_state = self._last_market_analysis.get('market_conditions', 'normal')
        if market_state == 'low_volatility':
            base_boost += 0.05
        elif market_state == 'high_volatility':
            base_boost *= 0.5

        if strategy_result:
            signal = getattr(strategy_result, 'signal', '')
            score = getattr(strategy_result, 'score', 0)

            if signal in {'increase_risk', 'long_bias'}:
                base_boost += 0.15
            elif signal in {'reduce_risk', 'short_bias'}:
                base_boost *= 0.5

            base_boost += min(0.4, max(0, score) * 0.05)

        # Минимальный буст чтобы гарантировать торговую активность
        return max(0.05, base_boost)

    def _generate_aggressive_opportunities_from_cache(self, strategy_result):
        """Создание синтетических возможностей, когда рынок спокоен"""
        if not self._last_candidates:
            return []

        if not hasattr(self, 'aggressive_filter_metrics'):
            self.aggressive_filter_metrics = defaultdict(int)

        aggressive_ops = []
        last_dynamic_threshold = getattr(self, '_last_dynamic_threshold', None)
        adaptive_threshold = max(
            getattr(self.config, 'MIN_DYNAMIC_PROFIT_FLOOR', 0.0),
            last_dynamic_threshold or self.config.MIN_TRIANGULAR_PROFIT
        )
        sorted_candidates = sorted(
            self._last_candidates,
            key=lambda item: item['best_direction']['profit_percent'],
            reverse=True
        )

        filtered_negative = 0
        filtered_below_threshold = 0

        for candidate in sorted_candidates[:3]:
            path = candidate['best_direction'].get('path')
            if not path:
                continue

            raw_profit = candidate['best_direction']['profit_percent']
            if raw_profit <= 0:
                filtered_negative += 1
                self.aggressive_filter_metrics['negative_raw_filtered'] += 1
                logger.debug(
                    "⚠️ Агрессивный кандидат %s отброшен: отрицательная сырая прибыль %.4f%%",
                    candidate['triangle_name'],
                    raw_profit
                )
                continue

            boost = self._calculate_aggressive_alpha(strategy_result, candidate)
            clamped_raw = max(raw_profit, 0)
            adjusted_profit = clamped_raw + boost

            if adjusted_profit < adaptive_threshold:
                filtered_below_threshold += 1
                self.aggressive_filter_metrics['below_min_profit_filtered'] += 1
                logger.debug(
                    "⚠️ Агрессивный кандидат %s отброшен: скорректированная прибыль %.4f%% ниже порога %.4f%%",
                    candidate['triangle_name'],
                    adjusted_profit,
                    adaptive_threshold
                )
                continue

            opportunity = {
                'type': 'triangular',
                'triangle_name': candidate['triangle_name'],
                'direction': candidate['best_direction']['direction'],
                'profit_percent': adjusted_profit,
                'symbols': candidate['triangle']['legs'],
                'prices': candidate['prices'],
                'execution_path': path,
                'timestamp': datetime.now(),
                'market_conditions': self._last_market_analysis.get('market_conditions', 'normal'),
                'priority': candidate['triangle'].get('priority', 999),
                'base_currency': candidate['triangle'].get('base_currency', 'USDT'),
                'aggressive_mode': True,
                'raw_profit_percent': raw_profit,
            }

            self.triangle_stats[candidate['triangle_name']]['opportunities_found'] += 1
            aggressive_ops.append(opportunity)

        if filtered_negative or filtered_below_threshold:
            logger.info(
                "📊 Фильтрация агрессивных кандидатов: отрицательных=%s, ниже порога=%s",
                filtered_negative,
                filtered_below_threshold
            )

        return aggressive_ops

    def execute_arbitrage(self, opportunity):
        """Основной метод выполнения арбитража"""
        triangle_name = opportunity.get('triangle_name', 'triangular')

        # Проверяем кулдаун треугольника
        if self._is_triangle_on_cooldown(triangle_name):
            logger.debug(
                "⏳ Треугольник %s находится на кулдауне, пропускаем сделку",
                triangle_name
            )
            return False

        logger.info(f"🎯 Executing arbitrage: {opportunity['triangle_name']}")
        logger.info(f"   Profit: {opportunity['profit_percent']:.4f}%")
        logger.info(f"   Market: {opportunity['market_conditions']}")

        # Получаем фактический баланс
        balance = self._fetch_actual_balance()
        balance_usdt = balance['available']

        # Обновляем снимок баланса в мониторинге
        if hasattr(self, 'monitor') and hasattr(self.monitor, 'update_balance_snapshot'):
            self.monitor.update_balance_snapshot(balance_usdt)

        configured_amount = getattr(self.config, 'TRADE_AMOUNT', 0)
        if configured_amount and balance_usdt + 1e-6 < configured_amount:
            logger.warning(
                "⚖️ Доступный баланс %.2f USDT ниже настроенного объёма сделки %.2f USDT",
                balance_usdt,
                configured_amount
            )

        min_required = max(5, self.config.TRADE_AMOUNT * 0.5)
        if balance_usdt < min_required:
            logger.warning(
                "❌ Недостаточный баланс: доступно %.2f USDT, требуется минимум %.2f USDT",
                balance_usdt,
                min_required
            )
            self.monitor.check_balance_health(balance_usdt)
            return False

        # Рассчитываем объемы сделок
        trade_plan = self.calculate_advanced_trade(opportunity, balance_usdt)

        if not trade_plan:
            logger.error("❌ Failed to calculate trade amounts")
            return False

        logger.info(
            f"📋 Trade plan: Initial amount: {trade_plan['initial_amount']} USDT, "
            f"Estimated profit: {trade_plan['estimated_profit_usdt']:.4f} USDT"
        )

        # Выполняем арбитраж
        success = self.execute_triangular_arbitrage(opportunity, trade_plan)

        if success:
            self.last_arbitrage_time[triangle_name] = datetime.now()
            logger.info("✅ Arbitrage execution completed successfully")

            # Отправляем отчёт о производительности каждые 10 сделок через монитор
            if len(self.trade_history) % 10 == 0:
                performance_report = self.get_triangle_performance_report()
                if hasattr(self, 'monitor') and hasattr(self.monitor, 'notify_performance'):
                    self.monitor.notify_performance(performance_report)
        else:
            logger.error("❌ Arbitrage execution failed")

        return success

    def get_effective_balance(self, coin='USDT'):
        """Прокси-метод для получения баланса с учётом симуляции"""
        real_trader = getattr(self, 'real_trader', None)
        if real_trader and getattr(real_trader, 'simulation_mode', False):
            return real_trader.get_balance(coin)

        client = getattr(self, 'client', None)
        if client and hasattr(client, 'get_balance'):
            return client.get_balance(coin)

        logger.warning("⚠️ Недоступен источник баланса, возвращаем нулевые значения")
        return {'available': 0.0, 'total': 0.0, 'coin': coin}

    def _fetch_actual_balance(self, coin='USDT'):
        """Возвращает нормализованный баланс с обработкой ошибок."""
        default_balance = {'available': 0.0, 'total': 0.0, 'coin': coin}

        try:
            balance = self.get_effective_balance(coin)
            if not isinstance(balance, dict):
                raise ValueError('Некорректный формат ответа баланса')
        except Exception as exc:
            logger.error("🔥 Ошибка получения баланса %s: %s", coin, str(exc))
            return default_balance

        available = self._safe_float(balance.get('available', 0.0))
        total = self._safe_float(balance.get('total', available))

        if total > 0:
            discrepancy = abs(total - available)
            if discrepancy > total * 0.05:
                logger.warning(
                    "📉 Расхождение баланса %s: доступно %.2f USDT из %.2f USDT",
                    coin,
                    available,
                    total
                )

        self._last_reported_balance = available

        return {'available': available, 'total': total, 'coin': balance.get('coin', coin)}

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование значений в float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _is_triangle_on_cooldown(self, triangle_name):
        """Проверка кулдауна треугольника без побочных эффектов"""
        last_time = self.last_arbitrage_time.get(triangle_name)

        if not last_time:
            return False

        cooldown_elapsed = (datetime.now() - last_time).total_seconds()
        return cooldown_elapsed < self.config.COOLDOWN_PERIOD

    def check_cooldown(self, symbol):
        """Проверка кулдауна для символа/треугольника"""
        if symbol not in self.last_arbitrage_time:
            return True
        
        last_time = self.last_arbitrage_time[symbol]
        cooldown_elapsed = (datetime.now() - last_time).total_seconds()

        if cooldown_elapsed < self.config.COOLDOWN_PERIOD:
            remaining = self.config.COOLDOWN_PERIOD - cooldown_elapsed
            logger.debug(f"⏳ Cooldown active for {symbol}: {remaining:.1f} seconds remaining")
            return False

        return True

# ==== Конец advanced_arbitrage_engine.py ====

# ==== Начало advanced_bot.py ====
import logging
import time
import signal
import sys
import os
import importlib
from pathlib import Path
from datetime import datetime
import inspect

# 👇 Гарантируем, что локальная папка проекта всегда есть в sys.path,
#    чтобы импорты работали даже при запуске скрипта из другой директории
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimized_config import OptimizedConfig
from advanced_arbitrage_engine import AdvancedArbitrageEngine

logger = logging.getLogger(__name__)


def ensure_psutil_available():
    """Проверяет доступность psutil перед запуском мониторинга"""
    if importlib.util.find_spec("psutil") is None:
        message = (
            "❗ Модуль psutil не найден. Установите зависимости командой "
            "'pip install -r requirements.txt'."
        )
        print(message, file=sys.stderr)
        sys.exit(1)

def setup_logging():
    """Настройка расширенного логгирования"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, OptimizedConfig().LOG_LEVEL, 'INFO'))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Файловый обработчик
    file_handler = logging.FileHandler(OptimizedConfig().LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик с цветами
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_market_snapshot(engine, max_symbols=None):
    """Выводит несколько актуальных котировок bid/ask для наглядности"""
    if not hasattr(engine, 'last_tickers'):
        return

    # Определяем количество отображаемых символов с учетом конфигурации
    if max_symbols is None:
        if hasattr(engine, 'config') and hasattr(engine.config, 'MARKET_SNAPSHOT_SYMBOLS'):
            max_symbols = engine.config.MARKET_SNAPSHOT_SYMBOLS
        else:
            max_symbols = 3

    tickers = getattr(engine, 'last_tickers', {})
    if not tickers:
        logger.info("📉 Нет актуальных котировок для отображения")
        return

    logger.info("📈 Текущие котировки (bid/ask):")
    for symbol in sorted(tickers.keys())[:max_symbols]:
        data = tickers[symbol]
        bid = data.get('bid')
        ask = data.get('ask')

        if bid is None or ask is None:
            logger.info(f"   {symbol}: данные bid/ask отсутствуют")
            continue

        if bid <= 0 or ask <= 0:
            logger.info(f"   {symbol}: bid={bid}, ask={ask} (некорректные значения для расчета спреда)")
            continue

        spread_percent = ((ask - bid) / ((ask + bid) / 2)) * 100 if (ask + bid) > 0 else 0
        logger.info(
            f"   {symbol}: bid={bid:.6f}, ask={ask:.6f}, спред={spread_percent:.4f}%"
        )

class GracefulKiller:
    """Обработчик сигналов для graceful shutdown"""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        self.kill_now = True

def main():
    """Основная функция запуска улучшенного бота"""
    global logger
    logger = setup_logging()
    # Используем оптимизированные параметры и явно включаем тестнет для безопасных прогонов
    config = OptimizedConfig()
    config.TESTNET = True  # Принудительный перевод в тестнет

    logger.info("=" * 70)
    logger.info("🚀 ADVANCED TRIANGULAR ARBITRAGE BOT STARTING 🚀")
    logger.info(f"🔧 Принудительный режим тестнета: {config.TESTNET}")
    logger.info(f"💰 Минимальный порог прибыли для ускоренного поиска: {config.MIN_TRIANGULAR_PROFIT}%")
    logger.info(
        "🧭 Ограничение на количество треугольников в ускоренном режиме: "
        f"{getattr(config, 'ACCELERATED_TRIANGLE_LIMIT', 0)}"
    )
    logger.info(f"📈 Monitoring {len(config.TRIANGULAR_PAIRS)} triangular pairs")
    logger.info(f"⚖️  Trade amount: {config.TRADE_AMOUNT} USDT")
    logger.info(f"🛡️  Max daily trades: {config.MAX_DAILY_TRADES}")
    logger.info(f"⏰ Update interval: {config.UPDATE_INTERVAL} seconds")
    logger.info(f"📊 Dashboard: http://localhost:{os.getenv('DASHBOARD_PORT', '8050')}")
    logger.info("=" * 70)

    ensure_psutil_available()

    engine_module_path = Path(inspect.getfile(AdvancedArbitrageEngine)).resolve()
    if PROJECT_ROOT not in engine_module_path.parents and PROJECT_ROOT != engine_module_path:
        logger.warning("⚠️ AdvancedArbitrageEngine импортирован не из корня проекта: %s", engine_module_path)
    else:
        logger.info("📂 Используется локальная версия AdvancedArbitrageEngine: %s", engine_module_path)

    engine = AdvancedArbitrageEngine()
    killer = GracefulKiller()
    
    try:
        iteration_count = 0
        start_time = datetime.now()
        total_opportunities_found = 0
        
        while not killer.kill_now:
            iteration_count += 1
            cycle_start = time.time()
            
            if iteration_count % 10 == 0:  # Каждые 10 итераций
                logger.info(f"\n{'=' * 30} Iteration #{iteration_count} {'=' * 30}")
                logger.info(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"🕐 Running for: {str(datetime.now() - start_time).split('.')[0]}")
            
            try:
                # Получение баланса с учетом режима симуляции
                balance = engine.get_effective_balance('USDT')
                balance_usdt = balance['available']
                
                if iteration_count % 10 == 0:
                    logger.info(f"💰 Account balance: {balance_usdt:.2f} USDT available")

                opportunities = engine.detect_opportunities()
                if iteration_count % 5 == 0:
                    # Каждые несколько циклов выводим фактические bid/ask значения
                    log_market_snapshot(engine)
                total_opportunities_found += len(opportunities)
                
                if opportunities:
                    if iteration_count % 5 == 0:  # Реже логируем найденные возможности
                        logger.info(f"🎯 Found {len(opportunities)} triangular arbitrage opportunities")
                    
                    # Фильтрация и выбор лучшей возможности
                    best_opportunity = opportunities[0]  # Уже отсортированы по прибыльности
                    
                    # Дополнительные проверки для лучшей возможности
                    balance_check_passed = balance_usdt > config.TRADE_AMOUNT * 0.5
                    if engine.real_trader.simulation_mode:
                        balance_check_passed = True

                    if (balance_check_passed and
                        engine.check_cooldown(best_opportunity['triangle_name'])):
                        
                        logger.info(f"⭐ Selected: {best_opportunity['triangle_name']} - "
                                  f"{best_opportunity['profit_percent']:.4f}% profit")
                        
                        # Выполнение арбитража
                        success = engine.execute_arbitrage(best_opportunity)
                        
                        if success:
                            logger.info(f"✅ SUCCESS! Triangular arbitrage executed")
                            
                            # Периодический отчет о производительности
                            if len(engine.trade_history) % 5 == 0:
                                report = engine.get_triangle_performance_report()
                                logger.info(f"📊 Performance: {report['total_executed_trades']} trades, "
                                          f"Total profit: {report['total_profit']:.4f} USDT")
                        else:
                            logger.error("❌ FAILED! Arbitrage execution failed")
                    else:
                        if iteration_count % 20 == 0:
                            logger.info("🔍 Opportunities found but skipped due to risk management")
                
                # Отправка системной сводки каждые 50 итераций
                if hasattr(engine, 'monitor') and hasattr(engine.monitor, 'send_system_summary'):
                    if iteration_count % 50 == 0:
                        engine.monitor.send_system_summary()
                
            except Exception as e:
                logger.error(f"🔥 Critical error during iteration: {str(e)}", exc_info=True)
                if hasattr(engine, 'monitor') and hasattr(engine.monitor, 'track_api_error'):
                    engine.monitor.track_api_error("main_loop", str(e))
            
            # Соблюдение интервала обновления
            cycle_time = time.time() - cycle_start
            sleep_time = max(0, config.UPDATE_INTERVAL - cycle_time)
            
            if sleep_time > 0 and iteration_count % 20 != 0:  # Реже логируем sleep
                time.sleep(sleep_time)
            elif cycle_time > config.UPDATE_INTERVAL:
                logger.warning(f"⚡ Cycle took longer than interval: {cycle_time:.2f}s")
            
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.critical(f"🔥 Bot crashed unexpectedly: {str(e)}", exc_info=True)
    finally:
        logger.info("🔧 Bot shutdown complete")
        
        # Финальные отчеты и экспорт
        if hasattr(engine, 'monitor') and hasattr(engine.monitor, 'export_trade_history'):
            engine.monitor.export_trade_history()
        
        if hasattr(engine, 'get_triangle_performance_report'):
            final_report = engine.get_triangle_performance_report()
            logger.info("📈 FINAL PERFORMANCE REPORT:")
            logger.info(f"   Total iterations: {iteration_count}")
            logger.info(f"   Total opportunities found: {total_opportunities_found}")
            logger.info(f"   Total trades executed: {final_report['total_executed_trades']}")
            logger.info(f"   Total profit: {final_report['total_profit']:.4f} USDT")
            
            # Лучшие треугольники
            best_triangles = sorted(
                final_report['triangle_details'].items(),
                key=lambda x: x[1]['total_profit'],
                reverse=True
            )[:3]
            
            logger.info("🏆 TOP 3 TRIANGLES:")
            for name, stats in best_triangles:
                logger.info(f"   {name}: {stats['executed_trades']} trades, "
                          f"{stats['total_profit']:.4f} USDT profit, "
                          f"{stats['success_rate']:.1%} success rate")
        
        logger.info("=" * 70)

if __name__ == "__main__":
    main()
# ==== Конец advanced_bot.py ====

# ==== Начало real_trading.py ====
import logging
import time
from collections import deque
from datetime import datetime
import os  # Исправлено: добавлен импорт os
from config import Config  # Исправлено: проверьте правильность пути импорта
from bybit_client import BybitClient  # Исправлено: проверьте правильность пути импорта

logger = logging.getLogger(__name__)

class RiskManager:
    """Менеджер рисков для реальной торговли"""
    
    def __init__(self):
        self.max_daily_loss = 5.0  # Максимальный убыток в день в USDT
        self.max_trade_size_percent = 10  # Максимальный размер сделки в процентах от баланса
        self.max_consecutive_losses = 3  # Максимальное количество убыточных сделок подряд
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.min_trade_interval = 60  # Минимальный интервал между сделками в секундах
    
    def can_execute_trade(self, trade_plan):
        """Проверка возможности выполнения сделки"""
        current_time = datetime.now()
        
        # Проверка интервала между сделками
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
            logger.warning(f"⏳ Слишком частые сделки. Ожидайте {(current_time - self.last_trade_time).total_seconds():.0f} секунд")
            return False
        
        # Проверка максимального размера сделки
        estimated_profit = trade_plan.get('estimated_profit_usdt', 0)
        if estimated_profit < 0.01:  # Минимальная прибыль 0.01 USDT
            logger.warning(f"📉 Слишком маленькая прибыль: {estimated_profit:.4f} USDT")
            return False
        
        return True
    
    def update_after_trade(self, trade_record):
        """Обновление статистики после сделки"""
        profit = trade_record.get('total_profit', 0)
        
        if profit < 0:
            self.daily_loss += abs(profit)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.last_trade_time = datetime.now()
        
        # Проверка лимитов
        if self.daily_loss > self.max_daily_loss:
            logger.critical(f"🔥 Достигнут максимальный дневной убыток: {self.daily_loss:.2f} USDT")
        
        if self.consecutive_losses > self.max_consecutive_losses:
            logger.critical(f"🔥 Достигнуто максимальное количество убыточных сделок подряд: {self.consecutive_losses}")


class ContingentOrderOrchestrator:
    """Оркестратор контингентных ордеров и поэтапного исполнения с хеджированием"""

    def __init__(self, client: BybitClient, config: Config):
        self.client = client
        self.config = config
        self.default_timeout = getattr(config, 'MAX_TRIANGLE_EXECUTION_TIME', 30)
        self.loss_limit_usdt = float(os.getenv('CONTINGENT_MAX_LOSS_USDT', '10'))

    def execute_sequence(self, legs: list[dict], hedge_leg: dict | None = None, max_loss_usdt: float | None = None, timeout: int | None = None):
        """Выполняет цепочку ног с проверкой статусов и хеджем при сбоях"""

        if not legs:
            logger.warning("⚠️ Передан пустой список ног для оркестратора")
            return None

        timeout_sec = timeout or self.default_timeout
        loss_cap = max_loss_usdt if max_loss_usdt is not None else self.loss_limit_usdt

        executed_orders = []
        hedge_actions = []
        amount_scale = 1.0

        for idx, leg in enumerate(legs, start=1):
            leg_payload = dict(leg)
            leg_payload['amount'] = float(leg_payload.get('amount', 0) or 0) * amount_scale

            logger.info("🧭 Оркестратор: шаг %s/%s для %s", idx, len(legs), leg_payload.get('symbol'))
            order_result, status, fill_ratio = self._place_and_monitor(leg_payload, timeout_sec)

            if order_result:
                executed_orders.append(order_result)

            if status in {'timeout', 'cancelled', 'failed'} or fill_ratio <= 0:
                hedge = self._apply_hedge(hedge_leg, leg_payload, executed_orders, loss_cap, reason="сбой шага")
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('failed', executed_orders, hedge_actions, amount_scale, loss_cap)

            if status == 'partial' and fill_ratio < 1.0:
                amount_scale *= fill_ratio
                hedge = self._apply_hedge(
                    hedge_leg,
                    leg_payload,
                    executed_orders,
                    loss_cap,
                    reason="частичное исполнение",
                    unfilled_ratio=1 - fill_ratio,
                )
                if hedge:
                    hedge_actions.append(hedge)

        return self._build_report('completed', executed_orders, hedge_actions, amount_scale, loss_cap)

    def _place_and_monitor(self, leg: dict, timeout_sec: int):
        """Размещает ордер и ждёт его исполнения или тайм-аута"""

        order = self.client.place_order(
            symbol=leg['symbol'],
            side=leg['side'],
            qty=leg['amount'],
            price=leg.get('price'),
            order_type=leg.get('type', 'Market'),
            trigger_price=leg.get('trigger_price'),
            trigger_by=leg.get('trigger_by', 'LastPrice'),
            reduce_only=leg.get('reduce_only', False),
        )

        if not order:
            logger.error("❌ Ордер шага не размещён")
            return None, 'failed', 0.0

        order_id = order.get('orderId')
        symbol = leg['symbol']
        status = self._normalize_status(order.get('orderStatus'))
        fill_ratio = self._calc_fill_ratio(order)

        if status in {'filled', 'cancelled'} or not order_id:
            return order, status, fill_ratio

        start_time = time.time()
        last_status = status

        while time.time() - start_time < timeout_sec:
            fetched = self.client.get_order_status(order_id, symbol) or {}
            if fetched:
                order.update(fetched)
                last_status = self._normalize_status(fetched.get('orderStatus'))
                fill_ratio = self._calc_fill_ratio(fetched)

                if last_status in {'filled', 'cancelled'}:
                    return order, last_status, fill_ratio

            time.sleep(1)

        logger.error("⏳ Тайм-аут исполнения ордера %s", order_id)
        return order, last_status or 'timeout', fill_ratio

    def _apply_hedge(self, hedge_leg, failed_leg, executed_orders, loss_cap, reason: str, unfilled_ratio: float | None = None):
        """Проводит компенсирующее действие при сбое или частичном исполнении"""

        hedge_payload = self._prepare_hedge_payload(hedge_leg, failed_leg, executed_orders, loss_cap, unfilled_ratio)
        if not hedge_payload:
            logger.warning("⚠️ Хедж не выполнен: нет подходящего payload")
            return None

        logger.warning("🛡️ Запуск хеджа (%s) для %s", reason, hedge_payload['symbol'])
        hedge_result = self.client.place_order(**hedge_payload)

        if hedge_result:
            hedge_status = self._normalize_status(hedge_result.get('orderStatus'))
            return {
                'reason': reason,
                'payload': hedge_payload,
                'result': hedge_result,
                'status': hedge_status,
            }

        logger.error("❌ Хедж не размещён")
        return None

    def _prepare_hedge_payload(self, hedge_leg, failed_leg, executed_orders, loss_cap, unfilled_ratio):
        """Формирует параметры хедж-ордера, ограничивая риск по сумме"""

        last_fill = self._extract_last_fill(executed_orders, failed_leg)
        if not last_fill:
            return None

        qty, price, symbol, side = last_fill
        target_symbol = (hedge_leg or {}).get('symbol', symbol)
        target_side = (hedge_leg or {}).get('side')

        hedge_side = target_side or ('sell' if side.lower() == 'buy' else 'buy')
        hedge_price = (hedge_leg or {}).get('price', price)
        hedge_type = (hedge_leg or {}).get('type', 'Market')

        effective_unfilled = qty * (unfilled_ratio or 1.0)
        capped_qty = self._cap_qty_by_loss(effective_unfilled, hedge_price, loss_cap)
        if capped_qty <= 0:
            return None

        return {
            'symbol': target_symbol,
            'side': hedge_side,
            'qty': capped_qty,
            'price': hedge_price,
            'order_type': hedge_type,
            'reduce_only': True,
        }

    def _extract_last_fill(self, executed_orders, fallback_leg):
        """Извлекает информацию о последнем исполненном объёме"""

        source = executed_orders[-1] if executed_orders else None
        if not source and fallback_leg:
            qty = float(fallback_leg.get('amount', 0) or 0)
            price = float(fallback_leg.get('price', 0) or 0)
            return qty, price, fallback_leg.get('symbol'), fallback_leg.get('side', '')

        if not source:
            return None

        qty = self._safe_float(source.get('cumExecQty') or source.get('qty'))
        price = self._safe_float(source.get('avgPrice') or source.get('price'))
        symbol = source.get('symbol') or fallback_leg.get('symbol')
        side = source.get('side') or fallback_leg.get('side', '')
        return qty, price, symbol, side

    def _cap_qty_by_loss(self, qty: float, price: float | None, loss_cap: float) -> float:
        """Ограничивает объём для хеджа, чтобы потенциальный убыток не превысил лимит"""

        if not price or price <= 0 or loss_cap <= 0:
            return qty

        max_qty = loss_cap / price
        return min(qty, max_qty)

    def _calc_fill_ratio(self, order_data: dict) -> float:
        """Рассчитывает долю исполнения ордера"""

        qty = self._safe_float(order_data.get('qty'))
        filled = self._safe_float(order_data.get('cumExecQty'))
        if qty <= 0:
            return 0.0
        return min(1.0, filled / qty)

    def _normalize_status(self, status: str | None) -> str:
        """Приводит статус ордера к нормализованной форме"""

        if not status:
            return 'unknown'
        status_lower = status.lower()
        if 'partial' in status_lower:
            return 'partial'
        if 'cancel' in status_lower:
            return 'cancelled'
        if 'filled' in status_lower:
            return 'filled'
        return status_lower

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование к float"""

        try:
            if value is None:
                return default
            if isinstance(value, str):
                value = value.strip()
                if value == '':
                    return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_report(self, status, executed_orders, hedges, amount_scale, loss_cap=None):
        """Формирует итоговый отчёт по цепочке"""

        hedge_cost = 0.0
        for hedge in hedges or []:
            payload = hedge.get('payload') or {}
            hedge_cost += self._safe_float(payload.get('price')) * self._safe_float(payload.get('qty'))

        estimated_profit = 0.0
        if hedge_cost > 0:
            limit = loss_cap if loss_cap is not None else hedge_cost
            estimated_profit = -min(limit, hedge_cost)

        return {
            'status': status,
            'executed': executed_orders,
            'hedges': hedges,
            'effective_scale': amount_scale,
            'estimated_profit': estimated_profit,
        }


class RealTradingExecutor:
    """Исполнение реальных ордеров с режимом симуляции и постепенного перехода к реальной торговле"""
    
    def __init__(self):
        self.config = Config()
        self.client = BybitClient()
        self.is_real_mode = False
        self.trade_history = []
        self.risk_manager = RiskManager()
        self.contingent_orchestrator = ContingentOrderOrchestrator(self.client, self.config)
        # Фиктивный баланс для симуляции, чтобы можно было управлять проверками ликвидности
        self._simulated_balance_usdt = self._load_simulated_balance()
        self.recent_order_events = deque(maxlen=200)
        self._last_execution_hint = None

        # Режим симуляции (True = симуляция, False = реальные ордера)
        simulation_env = os.getenv('TRADE_SIMULATION_MODE')
        legacy_simulation_env = os.getenv('SIMULATION_MODE')

        if simulation_env is not None:
            self.simulation_mode = simulation_env.lower() == 'true'
            mode_source = 'TRADE_SIMULATION_MODE'
        elif legacy_simulation_env is not None:
            self.simulation_mode = legacy_simulation_env.lower() == 'true'
            mode_source = 'SIMULATION_MODE'
        else:
            self.simulation_mode = self.config.TESTNET
            mode_source = 'TESTNET'

        logger.info(
            "🔄 Режим торговли: %s (источник: %s)",
            'симуляция' if self.simulation_mode else 'реальные ордера',
            mode_source
        )
        logger.info(
            "📡 Режим котировок Bybit: %s",
            'testnet' if self.config.TESTNET else 'mainnet'
        )

        # Подписываемся на события ордеров для быстрого реагирования
        self.client.add_order_listener(self._handle_order_event)

        if self.simulation_mode and not self.config.TESTNET:
            logger.warning(
                "🧪 Симуляция исполнения сделок при работе с реальными котировками Bybit"
            )
            logger.debug(
                "💳 Тестовый симулированный баланс: %.2f USDT", self._simulated_balance_usdt
            )
    
    def set_real_mode(self, enable_real_mode):
        """Переключение в реальный режим торговли"""
        if enable_real_mode and self.simulation_mode:
            # Запрашиваем подтверждение перед переходом в реальный режим
            confirmation = self._request_real_mode_confirmation()
            if confirmation:
                self.simulation_mode = False
                self.is_real_mode = True
                logger.info("✅ Переключено в реальный режим торговли")
                return True
            else:
                logger.warning("❌ Отмена перехода в реальный режим")
                return False
        return False
    
    def _request_real_mode_confirmation(self):
        """Запрос подтверждения перед переходом в реальный режим"""
        logger.warning("⚠️  ВНИМАНИЕ! Вы собираетесь перейти в реальный режим торговли!")
        logger.warning("⚠️  Будут выполняться реальные ордера с вашими средствами!")
        logger.warning("⚠️  Убедитесь, что вы протестировали стратегию в симуляционном режиме!")
        
        # В реальном приложении здесь должен быть запрос подтверждения
        # Пока возвращаем False для безопасности
        return False
    
    def execute_arbitrage_trade(self, trade_plan):
        """Выполнение арбитражной сделки"""
        if self.simulation_mode:
            return self._simulate_trade(trade_plan)
        else:
            return self._execute_real_trade(trade_plan)

    def execute_orchestrated_trade(self, legs: list[dict], hedge_leg: dict | None = None, max_loss_usdt: float | None = None, timeout: int | None = None):
        """Запуск оркестратора контингентных цепочек с учётом риск-менеджмента"""

        if not legs:
            logger.warning("⚠️ Невозможно запустить оркестратор без списка ног")
            return None

        safety_plan = {'estimated_profit_usdt': max(0.02, (max_loss_usdt or 0) * -1)}
        if not self.risk_manager.can_execute_trade(safety_plan):
            logger.error("❌ Риск-менеджер запретил запуск оркестратора")
            return None

        result = self.contingent_orchestrator.execute_sequence(legs, hedge_leg, max_loss_usdt, timeout)
        if not result:
            return None

        trade_record = {
            'timestamp': datetime.now(),
            'trade_plan': {'legs': legs, 'hedge': hedge_leg, 'max_loss_usdt': max_loss_usdt, 'timeout': timeout},
            'results': result.get('executed'),
            'hedges': result.get('hedges'),
            'status': result.get('status'),
            'total_profit': result.get('estimated_profit', 0),
            'simulated': self.simulation_mode,
        }

        self.trade_history.append(trade_record)
        if not self.simulation_mode:
            self.risk_manager.update_after_trade(trade_record)

        logger.info(
            "🏁 Оркестратор завершён со статусом %s (хеджей: %s)",
            trade_record['status'],
            len(trade_record.get('hedges') or []),
        )
        return trade_record

    def get_balance(self, coin='USDT'):
        """Возвращает баланс в зависимости от режима исполнения"""
        if self.simulation_mode:
            return {
                'available': self._simulated_balance_usdt,
                'total': self._simulated_balance_usdt,
                'coin': coin
            }

        return self.client.get_balance(coin)

    def _load_simulated_balance(self):
        """Загружает виртуальный баланс из окружения либо использует дефолт"""
        env_balance = os.getenv('SIMULATION_BALANCE_USDT')

        try:
            return float(env_balance) if env_balance is not None else 100.0
        except (TypeError, ValueError):
            logger.warning("⚠️ Некорректное значение SIMULATION_BALANCE_USDT, используем 100.0 USDT")
            return 100.0

    def _handle_order_event(self, event):
        """Реагирует на обновления по ордерам (исполнения и частичные исполнения)."""

        normalized = {
            'orderId': event.get('orderId'),
            'symbol': event.get('symbol'),
            'status': (event.get('orderStatus') or '').lower(),
            'side': event.get('side'),
            'filled_qty': self._safe_float(event.get('cumExecQty')),
            'remaining_qty': self._safe_float(event.get('leavesQty')),
            'avg_price': self._safe_float(event.get('avgPrice')),
            'execType': event.get('execType'),
            'updatedTime': event.get('updatedTime'),
        }

        self.recent_order_events.appendleft(normalized)
        self._last_execution_hint = normalized

        status = normalized['status']
        if status in ('partiallyfilled', 'partially_filled', 'partial_fill'):
            logger.info(
                "🔔 Частичное исполнение %s: исполнено=%s, осталось=%s",
                normalized['orderId'],
                normalized['filled_qty'],
                normalized['remaining_qty'],
            )
        elif status == 'filled':
            logger.info(
                "✅ Полное исполнение %s по цене %s",
                normalized['orderId'],
                normalized['avg_price'],
            )

    def get_live_order_events(self):
        """Возвращает последние события по ордерам для оперативных решений движка."""

        return list(self.recent_order_events)

    def _safe_float(self, value, default=0.0):
        """Безопасное приведение к float для данных из WebSocket."""

        try:
            if value is None:
                return default

            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default

            return float(value)
        except (TypeError, ValueError):
            return default
    
    def _simulate_trade(self, trade_plan):
        """Симуляция торговли"""
        logger.info("🧪 SIMULATION MODE: Симуляция исполнения ордеров")
        
        results = []
        total_profit = 0
        
        for step_name, step in trade_plan.items():
            if step_name.startswith('step') or step_name in ['leg1', 'leg2']:
                simulated_result = {
                    'orderId': f"sim_{int(time.time())}_{step_name}",
                    'orderStatus': 'Filled',
                    'symbol': step['symbol'],
                    'side': step['side'],
                    'qty': step['amount'],
                    'price': step['price'],
                    'avgPrice': step['price'],
                    'cumExecQty': step['amount'],
                    'simulated': True,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(simulated_result)
                logger.info(f"✅ SIMULATED: {step['side']} {step['amount']:.6f} {step['symbol']} @ {step['price']:.2f}")
        
        # Расчет прибыли для симуляции
        if 'estimated_profit_usdt' in trade_plan:
            total_profit = trade_plan['estimated_profit_usdt']
        
        trade_record = {
            'timestamp': datetime.now(),
            'trade_plan': trade_plan,
            'results': results,
            'total_profit': total_profit,
            'simulated': True
        }
        
        self.trade_history.append(trade_record)
        logger.info(f"💰 SIMULATED PROFIT: {total_profit:.4f} USDT")
        
        return trade_record

    def _get_live_price(self, symbol: str, side: str) -> tuple[float | None, dict]:
        """Возвращает актуальную цену из WebSocket/REST и источник данных."""

        market_price = None
        ticker_snapshot = {}

        if self.client.ws_manager:
            cached, missing = self.client.ws_manager.get_cached_tickers([symbol])
            ticker_snapshot = cached.get(symbol) or {}
            if not missing and ticker_snapshot:
                market_price = ticker_snapshot.get('ask') if side.lower() == 'buy' else ticker_snapshot.get('bid')

        if market_price is None:
            fresh = self.client.get_tickers([symbol]) or {}
            ticker_snapshot = fresh.get(symbol) or ticker_snapshot
            if ticker_snapshot:
                market_price = ticker_snapshot.get('ask') if side.lower() == 'buy' else ticker_snapshot.get('bid')

        if market_price:
            market_price = float(market_price)
        else:
            logger.warning("⚠️ Не удалось получить актуальную цену для %s", symbol)

        return market_price, ticker_snapshot

    def _ensure_price_alignment(self, planned_price: float | None, market_price: float | None, tolerance: float) -> bool:
        """Проверяет, что расчётная цена не отклоняется от рыночной выше допуска."""

        if not market_price or not planned_price:
            return True

        deviation = abs(planned_price - market_price) / planned_price if planned_price else 0
        if deviation > tolerance:
            logger.warning(
                "❌ Отклонение цены %.4f%% превышает допуск %.4f%%", deviation * 100, tolerance * 100
            )
            return False

        return True

    def _calculate_limit_price(self, side: str, market_price: float, tolerance: float) -> float:
        """Формирует лимитную цену с учётом направления и допуска проскальзывания."""

        if side.lower() == 'buy':
            return market_price * (1 + tolerance)
        return market_price * (1 - tolerance)

    def _handle_partial_fill(self, step: dict, order_result: dict, tolerance: float, market_price: float | None) -> float:
        """Обрабатывает частичное исполнение и возвращает коэффициент масштабирования для следующих шагов."""

        requested_qty = float(step.get('amount', 0) or 0)
        executed_qty = self._safe_float(order_result.get('cumExecQty'))
        avg_price = self._safe_float(order_result.get('avgPrice')) or step.get('price')

        if requested_qty <= 0 or executed_qty is None or executed_qty <= 0:
            return 1.0

        fill_ratio = min(1.0, executed_qty / requested_qty)

        if market_price and avg_price:
            deviation = abs(avg_price - market_price) / market_price
            if deviation > tolerance:
                logger.error(
                    "🔥 Фактическая цена исполнения отклоняется на %.4f%% (допуск %.4f%%)",
                    deviation * 100,
                    tolerance * 100,
                )
                return -1.0

        if fill_ratio < 1.0:
            logger.warning(
                "⚠️ Частичное исполнение: исполнено %.4f из %.4f (%.2f%%)",
                executed_qty,
                requested_qty,
                fill_ratio * 100,
            )

        return fill_ratio
    
    def _execute_real_trade(self, trade_plan):
        """Реальное исполнение торговли"""
        logger.warning("🔥 REAL MODE: Выполнение реальных ордеров")
        
        if not self.risk_manager.can_execute_trade(trade_plan):
            logger.error("❌ Риск-менеджер запретил выполнение сделки")
            return None
        
        try:
            results = []
            total_profit = 0
            slippage_tolerance = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
            amount_scale = 1.0

            # Выполняем ордера последовательно
            for step_name, step in trade_plan.items():
                if step_name.startswith('step') or step_name in ['leg1', 'leg2']:
                    # Пересчитываем объём с учётом предыдущих частичных исполнений
                    planned_amount = float(step.get('amount', 0) or 0) * amount_scale
                    trade_plan[step_name]['amount'] = planned_amount

                    market_price, _ = self._get_live_price(step['symbol'], step['side'])

                    if not self._ensure_price_alignment(step.get('price'), market_price, slippage_tolerance):
                        logger.error("🚫 Цепочка отменена из-за отклонения котировок на шаге %s", step_name)
                        self._cancel_previous_orders(results)
                        return None

                    if step.get('type', 'Limit').lower() == 'limit' and market_price:
                        new_limit = self._calculate_limit_price(step['side'], market_price, slippage_tolerance)
                        trade_plan[step_name]['price'] = new_limit
                        logger.info(
                            "🔧 Обновлена лимитная цена для %s: %.6f (рынок %.6f)",
                            step_name,
                            new_limit,
                            market_price,
                        )

                    order_result = self.client.place_order(
                        symbol=step['symbol'],
                        side=step['side'],
                        qty=planned_amount,
                        price=trade_plan[step_name].get('price'),
                        order_type=step.get('type', 'Limit')
                    )

                    if order_result:
                        results.append(order_result)
                        logger.info(
                            f"✅ REAL ORDER: {step['side']} {planned_amount:.6f} {step['symbol']} @ {trade_plan[step_name].get('price', '_MARKET_')}"
                        )

                        fill_ratio = self._handle_partial_fill(
                            trade_plan[step_name],
                            order_result,
                            slippage_tolerance,
                            market_price,
                        )

                        if fill_ratio < 0:
                            self._cancel_previous_orders(results)
                            return None

                        if 0 < fill_ratio < 1:
                            amount_scale *= fill_ratio

                    else:
                        logger.error(f"❌ FAILED ORDER: {step['side']} {planned_amount:.6f} {step['symbol']}")
                        # Отменяем предыдущие ордера при ошибке
                        self._cancel_previous_orders(results)
                        return None
            
            # Расчет реальной прибыли
            if results:
                total_profit = self._calculate_real_profit(results, trade_plan)
            
            trade_record = {
                'timestamp': datetime.now(),
                'trade_plan': trade_plan,
                'results': results,
                'total_profit': total_profit,
                'simulated': False
            }
            
            self.trade_history.append(trade_record)
            self.risk_manager.update_after_trade(trade_record)
            
            logger.info(f"💰 REAL PROFIT: {total_profit:.4f} USDT")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"🔥 CRITICAL ERROR during real trade execution: {str(e)}", exc_info=True)
            return None
    
    def _cancel_previous_orders(self, results):
        """Отмена предыдущих ордеров при ошибке"""
        for order in results:
            if 'orderId' in order:
                self.client.cancel_order(order['orderId'], order['symbol'])
    
    def _calculate_real_profit(self, results, trade_plan):
        """Расчет реальной прибыли на основе исполненных ордеров"""
        try:
            initial_amount = float(trade_plan.get('initial_amount', 0))
            if initial_amount <= 0:
                logger.warning("⚠️ Стартовый капитал не задан или некорректен, расчёт прибыли невозможен")
                return 0

            if any(order.get('simulated') for order in results):
                return float(trade_plan.get('estimated_profit_usdt', 0))

            base_currency = trade_plan.get('base_currency', 'USDT')
            fee_rate = getattr(self.config, 'TRADING_FEE', 0)

            # Инициализируем балансы: стартуем только с базовой валюты
            balances = {base_currency: initial_amount}

            for order in results:
                symbol = order.get('symbol') or ''
                side = (order.get('side') or '').lower()
                price = float(order.get('avgPrice') or order.get('price') or 0)
                quantity = float(order.get('cumExecQty') or order.get('qty') or 0)

                base, quote = self._split_symbol(symbol)
                if not base or not quote or price <= 0 or quantity <= 0:
                    logger.warning("⚠️ Пропуск ордера из-за некорректных данных при расчёте прибыли")
                    continue

                if side == 'buy':
                    # Покупаем базовый актив за котируемую валюту, комиссия уменьшает получаемое количество
                    cost = price * quantity
                    balances[quote] = balances.get(quote, 0) - cost
                    received = quantity * (1 - fee_rate)
                    balances[base] = balances.get(base, 0) + received
                elif side == 'sell':
                    # Продаём базовый актив за котируемую валюту, комиссия уменьшает итоговую выручку
                    balances[base] = balances.get(base, 0) - quantity
                    proceeds = price * quantity * (1 - fee_rate)
                    balances[quote] = balances.get(quote, 0) + proceeds
                else:
                    logger.warning("⚠️ Неизвестная сторона сделки при расчёте прибыли")

            real_profit = balances.get(base_currency, 0) - initial_amount
            trade_plan['estimated_profit_usdt'] = real_profit
            return real_profit
        except Exception as e:
            logger.error(f"Ошибка расчета реальной прибыли: {str(e)}")
            return 0

    def _split_symbol(self, symbol):
        """Разделяет тикер на базовую и котируемую валюты"""
        for quote in sorted(self.config.KNOWN_QUOTES, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        return None, None
    
    def get_performance_stats(self):
        """Получение статистики производительности"""
        if not self.trade_history:
            return {}
        
        total_trades = len(self.trade_history)
        successful_trades = sum(1 for trade in self.trade_history if trade.get('total_profit', 0) > 0)
        total_profit = sum(trade.get('total_profit', 0) for trade in self.trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        
        runtime = datetime.now() - min(trade['timestamp'] for trade in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'runtime': str(runtime).split('.')[0],
            'simulation_mode': self.simulation_mode,
            'real_mode': self.is_real_mode
        }
    
    def export_trade_history(self, filename=None):
        """Экспорт истории сделок"""
        import csv
        import json

        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        def _convert_datetime_values(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {k: _convert_datetime_values(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert_datetime_values(item) for item in value]
            return value

        def _prepare_trade_plan(trade_plan):
            if not trade_plan:
                return {}
            prepared = _convert_datetime_values(trade_plan)
            return prepared if isinstance(prepared, dict) else {'value': prepared}

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'symbol', 'side', 'amount', 'price', 'profit', 'simulated', 'trade_details']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                for trade in self.trade_history:
                    results = trade.get('results') or []

                    if not results:
                        details = trade.get('details', {})
                        symbols = details.get('symbols') or trade.get('symbol') or ''
                        if isinstance(symbols, (list, tuple)):
                            symbols = ','.join(symbols)
                        results = [{
                            'symbol': symbols,
                            'side': details.get('direction', trade.get('direction', '')),
                            'qty': details.get('initial_amount', 0),
                            'price': details.get('price', 0)
                        }]

                    for result in results:
                        timestamp = trade['timestamp']
                        if hasattr(timestamp, 'strftime'):
                            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            timestamp_str = str(timestamp)

                        writer.writerow({
                            'timestamp': timestamp_str,
                            'symbol': result.get('symbol', ''),
                            'side': result.get('side', ''),
                            'amount': result.get('qty', result.get('cumExecQty', 0)),
                            'price': result.get('avgPrice', result.get('price', 0)),
                            'profit': trade.get('total_profit', 0) if result == results[-1] else 0,
                            'simulated': trade.get('simulated', False),
                            'trade_details': json.dumps(
                                _prepare_trade_plan(trade.get('trade_plan', {})),
                                default=str
                            )
                        })
            
            logger.info(f"✅ Trade history exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error exporting trade history: {str(e)}")
            return None
# ==== Конец real_trading.py ====

