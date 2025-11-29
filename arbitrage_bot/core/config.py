import logging
import os
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import requests
from dotenv import load_dotenv
from logging_utils import (
    LOG_FORMAT,
    ContextFilter,
    configure_root_logging,
    create_adapter,
    generate_cycle_id,
)

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    # API настройки
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'
    PRIMARY_EXCHANGE = os.getenv('PRIMARY_EXCHANGE', 'bybit').lower()
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
    # Базовая допустимая давность котировок перед предупреждением (секунды)
    TICKER_STALENESS_WARNING_SEC_DEFAULT = 5.0

    def __init__(self):
        # Подчищаем возможный импорт aiohttp от других тестов/модулей, чтобы базовая загрузка не зависела от него
        sys.modules.pop("aiohttp", None)

        self._triangular_pairs_cache = None
        self._available_symbols_cache = None
        self._available_cross_map_cache = None
        self._symbol_watchlist_cache = None
        self._triangles_last_update = None
        self._okx_symbol_map = {}
        self._symbol_details_map = {}
        self._symbol_market_source = {}
        self._market_category_override = os.getenv('MARKET_CATEGORY_OVERRIDE')
        self._auto_detect_market_category = os.getenv('AUTO_DETECT_MARKET_CATEGORY', 'true').lower() == 'true'
        self._enable_spot_in_testnet = os.getenv('TESTNET_SPOT_ENABLED', 'false').lower() == 'true'
        self._arbitrage_market_hint = os.getenv('ARBITRAGE_TICKER_CATEGORY', 'spot').lower()
        self._testnet_spot_api_base = os.getenv('TESTNET_SPOT_API_BASE_URL', 'https://api-testnet.bybit.com')
        self._detected_market_category = None
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
            self.TICKER_STALENESS_WARNING_SEC_DEFAULT,
        )
        self._simulation_slippage_tolerance = self._load_float_env(
            'SIMULATION_SLIPPAGE_TOLERANCE',
            getattr(self, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
        )
        self._simulation_latency_range = self._load_latency_range()
        self._simulation_partial_fill_probability = self._load_float_env(
            'SIMULATION_PARTIAL_FILL_PROB',
            0.0,
        )
        self._simulation_reject_probability = self._load_float_env(
            'SIMULATION_REJECT_PROB',
            0.0,
        )
        self._simulation_liquidity_buffer = self._load_float_env(
            'SIMULATION_LIQUIDITY_BUFFER',
            0.0,
        )
        self._simulation_auto_complete_partials = os.getenv(
            'SIMULATION_AUTO_COMPLETE_PARTIALS',
            'true',
        ).lower() == 'true'
        self._market_symbols_limit = self._load_int_env('MARKET_SYMBOLS_LIMIT', 100)
        # Длительность кэша для треугольников (в секундах)
        self._triangles_cache_ttl = 60
        self.WEBSOCKET_PRICE_ONLY = os.getenv('WEBSOCKET_PRICE_ONLY', 'false').lower() == 'true'
        self.ENABLE_ASYNC_MARKET_CLIENT = os.getenv('ENABLE_ASYNC_MARKET_CLIENT', 'true').lower() == 'true'
        self.USE_LEGACY_TICKER_CLIENT = os.getenv('USE_LEGACY_TICKER_CLIENT', 'false').lower() == 'true'
        self.ASYNC_TICKER_CONCURRENCY = self._load_int_env('ASYNC_TICKER_CONCURRENCY', 6)
        self.PAPER_TRADING_MODE = os.getenv('PAPER_TRADING_MODE', 'false').lower() == 'true'
        self.PAPER_BOOK_IMPACT = self._load_float_env('PAPER_BOOK_IMPACT', 0.05)
        self.REPLAY_DATA_PATH = os.getenv('REPLAY_DATA_PATH')
        self.REPLAY_SPEED = self._load_float_env('REPLAY_SPEED', 1.0)
        self.REPLAY_MAX_RECORDS = self._load_int_env('REPLAY_MAX_RECORDS', 0)

        # Параметры ML-оптимизатора порога прибыли
        self.ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/profit_threshold_model.pkl')
        self.ML_FALLBACK_THRESHOLD = self._load_float_env(
            'ML_FALLBACK_THRESHOLD',
            self.MIN_TRIANGULAR_PROFIT,
        )
        self.ML_MIN_THRESHOLD = self._load_float_env('ML_MIN_THRESHOLD', 0.0)

    @property
    def MARKET_CATEGORY(self):
        """Возвращает тип сегмента рынка с учётом среды и арбитражных тикеров."""

        return self.MARKET_CATEGORIES[0]

    @property
    def MARKET_CATEGORIES(self):
        """Перечень приоритетных сегментов рынка (spot + фьючерсы)"""

        if self._market_category_override:
            return [self._market_category_override]

        prefers_spot = self._arbitrage_market_hint == 'spot' or self._enable_spot_in_testnet
        detected_category = None

        ordered = ['spot', 'linear'] if prefers_spot else ['linear', 'spot']

        if self.TESTNET:
            if self._auto_detect_market_category:
                detected_category = self._detect_testnet_market_category(prefers_spot)
                if detected_category:
                    ordered = [detected_category] + [cat for cat in ordered if cat != detected_category]

            if (
                not self._enable_spot_in_testnet
                and not prefers_spot
                and detected_category != 'spot'
            ):
                ordered = [cat for cat in ordered if cat != 'spot'] + (
                    ['spot'] if prefers_spot else []
                )

        return list(dict.fromkeys(ordered))

    @property
    def API_BASE_URL(self):
        """Базовый URL публичного API Bybit с учётом выбранного сегмента рынка."""

        if self.PRIMARY_EXCHANGE == 'okx':
            return 'https://www.okx.com'

        if self.TESTNET:
            if self.MARKET_CATEGORY == 'spot':
                return self._testnet_spot_api_base
            return 'https://api-testnet.bybit.com'

        return 'https://api.bybit.com'

    def _detect_testnet_market_category(self, prefers_spot: bool) -> str | None:
        """Пытается автоматически подобрать категорию для тестовой среды."""

        if self._detected_market_category:
            return self._detected_market_category

        order = ['spot', 'linear'] if prefers_spot else ['linear', 'spot']
        base_url = 'https://api-testnet.bybit.com'

        for candidate in order:
            try:
                response = requests.get(
                    f"{base_url}/v5/market/instruments-info",
                    params={'category': candidate},
                    timeout=5,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get('retCode') == 0 and payload.get('result', {}).get('list'):
                    self._detected_market_category = candidate
                    logger.info(
                        "Автоматически выбран сегмент %s для тестовой среды", candidate
                    )
                    return self._detected_market_category
            except requests.RequestException as exc:
                logger.debug(
                    "Автодетект категории %s не удался: %s", candidate, exc
                )

        logger.warning(
            "Не удалось определить доступную категорию тестнета, используем запасной порядок"
        )
        fallback = 'linear' if 'linear' in order else order[0]
        self._detected_market_category = fallback
        return self._detected_market_category

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
            available_symbols = set(self._fetch_market_symbols())
            watchlist = list(available_symbols)

            triangle_legs = set()
            for triangle in self.TRIANGULAR_PAIRS:
                triangle_legs.update(triangle['legs'])

            missing_triangle_legs = []

            for leg in sorted(triangle_legs):
                if available_symbols and leg not in available_symbols:
                    missing_triangle_legs.append(leg)
                    continue
                if leg not in watchlist:
                    watchlist.append(leg)

            if available_symbols and len(available_symbols) < 3:
                logger.warning(
                    "Получено слишком мало символов (%s) для построения треугольников. "
                    "Добавляем обязательные ноги и базовый список по умолчанию.",
                    len(available_symbols),
                )

                for leg in sorted(missing_triangle_legs):
                    if leg not in watchlist:
                        watchlist.append(leg)

                for default_symbol in self.DEFAULT_SYMBOLS:
                    if default_symbol not in watchlist:
                        watchlist.append(default_symbol)

            filtered_watchlist = self._filter_symbols_by_market_category(watchlist)
            limit = self._market_symbols_limit

            if limit and len(filtered_watchlist) > limit:
                logger.warning(
                    "Список наблюдаемых тикеров сокращён с %s до лимита %s",
                    len(filtered_watchlist),
                    limit,
                )
                filtered_watchlist = filtered_watchlist[:limit]

            self._symbol_watchlist_cache = filtered_watchlist
        return self._symbol_watchlist_cache

    @property
    def TRIANGULAR_PAIRS(self):
        """Динамическая конфигурация треугольников в зависимости от доступных тикеров"""
        if (
            self._triangular_pairs_cache is not None
            and self._triangles_last_update is not None
            and (time.time() - self._triangles_last_update) < self._triangles_cache_ttl
        ):
            return self._triangular_pairs_cache

        available_symbols = set(self._fetch_market_symbols())
        templates = self._build_triangle_templates(available_symbols)

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
                self._triangles_last_update = time.time()
                return self._triangular_pairs_cache

            filtered_templates = [
                triangle
                for triangle in templates
                if all(leg in available_symbols for leg in triangle['legs'])
            ]

            if filtered_templates:
                logger.warning(
                    "Используем отфильтрованный статический шаблон: доступно %s треугольников",
                    len(filtered_templates),
                )
                self._triangular_pairs_cache = filtered_templates
                self._triangles_last_update = time.time()
                return self._triangular_pairs_cache

            logger.warning(
                "API не вернул треугольники с полным покрытием. Используем шаблон как запасной вариант без фильтра."
            )

        # Фолбэк (например, оффлайн среда или ошибки сети)
        self._triangular_pairs_cache = templates
        self._triangles_last_update = time.time()
        return self._triangular_pairs_cache

    def reset_symbol_caches(self):
        """Сбрасывает кэш доступных символов и зависимых структур."""

        self._available_symbols_cache = None
        self._available_cross_map_cache = None
        self._symbol_watchlist_cache = None
        self._triangular_pairs_cache = None
        self._triangles_last_update = None

    def _filter_symbols_by_market_category(self, symbols: list[str]) -> list[str]:
        """Фильтрует тикеры по текущему сегменту рынка с учётом источника."""

        market_category = self.MARKET_CATEGORY
        filtered: list[str] = []
        mismatched: list[str] = []
        missing_category: list[str] = []

        for symbol in symbols:
            sources = self._symbol_market_source.get(symbol)
            if sources:
                if market_category in sources:
                    filtered.append(symbol)
                else:
                    mismatched.append(symbol)
            else:
                if market_category == 'spot':
                    missing_category.append(symbol)
                else:
                    filtered.append(symbol)

        dropped_total = len(mismatched) + len(missing_category)
        if dropped_total:
            logger.info(
                "Отброшено %s тикеров из-за несоответствия категории %s (без данных: %s). Итоговый список: %s",
                dropped_total,
                market_category,
                len(missing_category),
                len(filtered),
            )

        return filtered
        self._symbol_market_source = {}
    
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
    MAX_TRADE_AMOUNT = 100  # Жёсткий потолок размера сделки в USDT для ограничений риск-менеджмента
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

    @property
    def SIMULATION_SLIPPAGE_TOLERANCE(self):
        """Допустимое проскальзывание при симуляции"""
        return self._simulation_slippage_tolerance

    @property
    def SIMULATION_LATENCY_RANGE(self):
        """Диапазон искусственной задержки в секундах для симуляции"""
        return self._simulation_latency_range

    @property
    def SIMULATION_PARTIAL_FILL_PROBABILITY(self):
        """Вероятность частичного исполнения ордера в симуляции (0..1)."""
        return self._simulation_partial_fill_probability

    @property
    def SIMULATION_REJECT_PROBABILITY(self):
        """Вероятность принудительного отказа ордера в симуляции (0..1)."""
        return self._simulation_reject_probability

    @property
    def SIMULATION_LIQUIDITY_BUFFER(self):
        """Доля недоступной ликвидности в симуляции (0..1), имитирует пустой стакан."""
        return self._simulation_liquidity_buffer

    @property
    def SIMULATION_AUTO_COMPLETE_PARTIALS(self):
        """Флаг, разрешающий автоматически добирать остаток после частичного исполнения."""
        return self._simulation_auto_complete_partials

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

    def _load_int_env(self, var_name, default: int) -> int:
        """Читает целочисленный параметр из окружения с фолбэком."""

        raw_value = os.getenv(var_name)
        if raw_value is None:
            return default

        try:
            return int(raw_value)
        except ValueError:
            logger.warning(
                "Некорректное значение %s='%s'. Используем целочисленный дефолт %s.",
                var_name,
                raw_value,
                default,
            )
            return default

    def _load_latency_range(self):
        """Читает диапазон задержек для симуляции из окружения"""
        raw_range = os.getenv('SIMULATION_LATENCY_RANGE')
        if not raw_range:
            return (0.05, 0.2)

        normalized = raw_range.replace(' ', '').replace(';', ',')
        parts = [p for p in normalized.split(',') if p]

        if len(parts) != 2:
            logger.warning(
                "Некорректный формат SIMULATION_LATENCY_RANGE='%s'. Используем диапазон по умолчанию.",
                raw_range
            )
            return (0.05, 0.2)

        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            logger.warning(
                "Не удалось распарсить SIMULATION_LATENCY_RANGE='%s'. Используем диапазон по умолчанию.",
                raw_range
            )
            return (0.05, 0.2)

        if start < 0 or end < 0:
            logger.warning(
                "Задержка не может быть отрицательной. Используем диапазон по умолчанию."
            )
            return (0.05, 0.2)

        if start > end:
            start, end = end, start

        return (start, end)

    def _build_triangle_templates(self, available_symbols=None):
        """Создает список потенциальных треугольников, согласованный с реальными тикерами"""
        limit = getattr(self, "ACCELERATED_TRIANGLE_LIMIT", 0) or None
        dynamic_templates = self._build_dynamic_triangle_templates(
            available_symbols,
            limit=limit,
        )

        if dynamic_templates:
            if limit and len(dynamic_templates) > limit:
                logger.info(
                    "Ограничиваем количество динамических треугольников до %s из %s",  # noqa: E501
                    limit,
                    len(dynamic_templates),
                )
                return dynamic_templates[:limit]
            return dynamic_templates

        # Фолбэк на статическую конфигурацию, если API недоступно или ничего не построено
        static_templates = self._build_static_triangle_templates(available_symbols)
        if available_symbols:
            filtered_static = [
                triangle
                for triangle in static_templates
                if all(leg in available_symbols for leg in triangle['legs'])
            ]
            if filtered_static:
                return filtered_static

        return static_templates

    def _build_dynamic_triangle_templates(self, available_symbols, limit: int | None = None):
        """Построение динамических шаблонов треугольников с возможностью прерывания."""
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

        processed_count = 0
        total_symbols = len(base_candidates)

        logger.info(
            f"🔄 Начинаю построение треугольных пар из {total_symbols} доступных символов..."
        )

        for base_currency in base_candidates:
            if not threading.current_thread().is_alive():
                logger.warning("⚠️ Процесс прерван пользователем во время построения треугольников")
                break

            processed_count += 1

            if processed_count % max(1, total_symbols // 10) == 0:
                progress = (processed_count / total_symbols) * 100
                logger.info(
                    f"📊 Прогресс построения треугольников: {progress:.1f}% "
                    f"({processed_count}/{total_symbols})"
                )

            connected_assets = sorted(self._collect_connected_assets(base_currency, available_symbols))
            if len(connected_assets) < 2:
                continue

            for primary_asset in connected_assets:
                if primary_asset == base_currency:
                    continue

                for secondary_asset in connected_assets:
                    if secondary_asset in {primary_asset, base_currency}:
                        continue

                    try:
                        leg_options = self._collect_leg_combinations(
                            base_currency,
                            primary_asset,
                            secondary_asset,
                            available_symbols,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"⚠️ Ошибка при обработке символа {base_currency}: {str(e)}"
                        )
                        continue

                    if not leg_options:
                        continue

                    triangle_name = f'{base_currency}-{primary_asset}-{secondary_asset}-{base_currency}'
                    for legs, leg_types in leg_options:
                        combination_name = f"{triangle_name}|{'-'.join(legs)}|{'-'.join(leg_types)}"
                        if combination_name in registered:
                            continue

                        priority = min(
                            self.TRIANGLE_BASES.get(primary_asset, 5),
                            self.TRIANGLE_BASES.get(secondary_asset, 5)
                        )

                        leg_type_map = {leg: leg_type for leg, leg_type in zip(legs, leg_types)}

                        templates.append({
                            'name': triangle_name,
                            'legs': legs,
                            'leg_types': leg_types,
                            'leg_type_map': leg_type_map,
                            'base_currency': base_currency,
                            'priority': priority
                        })
                        registered.add(combination_name)

                        if limit and len(templates) >= limit:
                            logger.info(f"🎯 Достигнут лимит треугольных пар: {limit}")
                            return templates

        logger.info(f"✅ Успешно построено {len(templates)} треугольных пар")
        return templates

    def _build_static_triangle_templates(self, available_symbols=None):
        """Резервный список треугольников на случай оффлайн режима"""
        templates = [
            {
                'name': 'USDT-BTC-ETH-USDT',
                'legs': ['BTCUSDT', 'ETHBTC', 'ETHUSDT'],
                'leg_types': ['spot', 'spot', 'spot'],
                'leg_type_map': {'BTCUSDT': 'spot', 'ETHBTC': 'spot', 'ETHUSDT': 'spot'},
                'base_currency': 'USDT',
                'priority': 1
            },
            {
                'name': 'USDT-ETH-BTC-USDT',
                'legs': ['ETHUSDT', 'ETHBTC', 'BTCUSDT'],
                'leg_types': ['spot', 'spot', 'spot'],
                'leg_type_map': {'ETHUSDT': 'spot', 'ETHBTC': 'spot', 'BTCUSDT': 'spot'},
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
                'leg_types': ['spot', 'spot', 'spot'],
                'leg_type_map': {
                    f'{asset}USDT': 'spot',
                    f'{asset}BTC': 'spot',
                    'BTCUSDT': 'spot'
                },
                'base_currency': 'USDT',
                'priority': priority
            })
            templates.append({
                'name': f'USDT-{asset}-ETH-USDT',
                'legs': [f'{asset}USDT', f'{asset}ETH', 'ETHUSDT'],
                'leg_types': ['spot', 'spot', 'spot'],
                'leg_type_map': {
                    f'{asset}USDT': 'spot',
                    f'{asset}ETH': 'spot',
                    'ETHUSDT': 'spot'
                },
                'base_currency': 'USDT',
                'priority': priority
            })

        if available_symbols:
            available_set = set(available_symbols)
            templates = [
                triangle
                for triangle in templates
                if all(leg in available_set for leg in triangle['legs'])
            ]

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

    def _collect_leg_combinations(self, base_currency, primary_asset, secondary_asset, available_symbols):
        """Комбинирует все варианты ног для гибридных треугольников."""

        leg1_options = self._resolve_market_leg_options(primary_asset, base_currency, available_symbols)
        leg2_options = self._resolve_market_leg_options(secondary_asset, primary_asset, available_symbols)
        leg3_options = self._resolve_market_leg_options(secondary_asset, base_currency, available_symbols)

        if not leg1_options or not leg2_options or not leg3_options:
            return []

        combinations = []
        registered = set()

        for leg1, leg2, leg3 in product(leg1_options, leg2_options, leg3_options):
            legs = [leg1[0], leg2[0], leg3[0]]
            leg_types = [leg1[1], leg2[1], leg3[1]]
            key = tuple(legs + leg_types)
            if key in registered:
                continue

            combinations.append((legs, leg_types))
            registered.add(key)

        return combinations

    def _resolve_triangle_legs(self, base_currency, primary_asset, secondary_asset, available_symbols):
        """Подбирает первую доступную комбинацию тикеров (для обратной совместимости)."""

        combinations = self._collect_leg_combinations(
            base_currency,
            primary_asset,
            secondary_asset,
            available_symbols,
        )

        if not combinations:
            return None

        return combinations[0][0]

    def _resolve_market_leg_options(self, base_currency, quote_currency, available_symbols):
        """Возвращает все тикеры между двумя валютами с указанием сегмента рынка."""

        normalized_available = {symbol.upper() for symbol in available_symbols}
        options = []
        target_pairs = {
            (base_currency.upper(), quote_currency.upper()),
            (quote_currency.upper(), base_currency.upper()),
        }

        for symbol in normalized_available:
            base_coin, quote_coin = self._symbol_details_map.get(symbol, (None, None))
            if not base_coin or not quote_coin:
                base_coin, quote_coin = self._split_symbol(symbol)

            if not base_coin or not quote_coin:
                continue

            if (base_coin, quote_coin) not in target_pairs and (quote_coin, base_coin) not in target_pairs:
                continue

            market_types = self._symbol_market_source.get(symbol)
            if not market_types:
                market_types = {self.MARKET_CATEGORY}
            if isinstance(market_types, str):
                market_types = {market_types}

            for market_type in market_types:
                options.append((symbol, market_type))

        return options

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

        if self.PRIMARY_EXCHANGE == 'okx':
            return self._fetch_okx_market_symbols()

        categories = self.MARKET_CATEGORIES
        aggregated_symbols: list[str] = []
        aggregated_details: dict[str, tuple[str, str]] = {}
        aggregated_sources: dict[str, set[str]] = {}

        def _resolve_api_base(category: str) -> str:
            if self.PRIMARY_EXCHANGE == 'okx':
                return self.API_BASE_URL
            if self.TESTNET:
                if category == 'spot' and self._enable_spot_in_testnet:
                    return self._testnet_spot_api_base
                return 'https://api-testnet.bybit.com'
            return 'https://api.bybit.com'

        def _load_category_symbols(category: str):
            url = f"{_resolve_api_base(category)}/v5/market/instruments-info"
            params = {'category': category}
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return category, response.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Не удалось получить инструменты Bybit (%s): %s", category, exc
                )
                return category, None

        with ThreadPoolExecutor(max_workers=len(categories)) as executor:
            futures = [executor.submit(_load_category_symbols, category) for category in categories]
            for future in futures:
                category, data = future.result()
                if not data:
                    continue
                if data.get('retCode') != 0:
                    logger.warning(
                        "REST ответил кодом %s для %s: %s",
                        data.get('retCode'),
                        category,
                        data.get('retMsg'),
                    )
                    continue

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
                limit = self._market_symbols_limit if self._market_symbols_limit > 0 else None
                selected_entries = market_entries if limit is None else market_entries[:limit]

                aggregated_symbols.extend(symbol for symbol, _ in selected_entries)
                aggregated_details.update({
                    symbol: detailed_symbols.get(symbol)
                    for symbol, _ in selected_entries
                    if detailed_symbols.get(symbol)
                })
                for symbol, _ in selected_entries:
                    aggregated_sources.setdefault(symbol, set()).add(category)

                logger.info(
                    "Получено %s тикеров для категории %s (ограничение %s)",
                    len(selected_entries),
                    category,
                    limit if limit is not None else 'без ограничений',
                )

        self._available_symbols_cache = list(dict.fromkeys(aggregated_symbols))
        self._symbol_details_map = aggregated_details
        self._symbol_market_source = aggregated_sources

        return self._available_symbols_cache

    def _fetch_okx_market_symbols(self):
        """Получает список инструментов OKX и сопоставляет формат тикеров."""
        try:
            instruments_resp = requests.get(
                f"{self.API_BASE_URL}/api/v5/public/instruments",
                params={'instType': 'SPOT'},
                timeout=10,
            )
            instruments_resp.raise_for_status()
            instruments_data = instruments_resp.json()
            instruments = instruments_data.get('data', []) or []

            tickers_resp = requests.get(
                f"{self.API_BASE_URL}/api/v5/market/tickers",
                params={'instType': 'SPOT'},
                timeout=10,
            )
            tickers_resp.raise_for_status()
            tickers_data = tickers_resp.json()
            volumes = {item.get('instId'): float(item.get('volCcy24h', 0) or 0) for item in tickers_data.get('data', [])}

            ranked = []
            for item in instruments:
                inst_id = item.get('instId')
                base_coin = (item.get('baseCcy') or '').upper()
                quote_coin = (item.get('quoteCcy') or '').upper()
                state = item.get('state')

                if not inst_id or state not in {'live', 'trading'}:
                    continue

                normalized_symbol = inst_id.replace('-', '').upper()
                self._okx_symbol_map[normalized_symbol] = inst_id
                if base_coin and quote_coin:
                    self._symbol_details_map[normalized_symbol] = (base_coin, quote_coin)

                ranked.append((normalized_symbol, volumes.get(inst_id, 0.0)))

            ranked.sort(key=lambda pair: pair[1], reverse=True)
            limit = self._market_symbols_limit if self._market_symbols_limit > 0 else None
            selected = ranked if limit is None else ranked[:limit]

            self._available_symbols_cache = [symbol for symbol, _ in selected]

            logger.info(
                "Получено %s тикеров с OKX (ограничение %s)",
                len(self._available_symbols_cache),
                limit if limit is not None else 'без ограничений',
            )
            return self._available_symbols_cache

        except requests.RequestException as exc:
            logger.warning("Не удалось получить инструменты OKX: %s", exc)

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

    def get_okx_inst_id(self, symbol: str):
        """Возвращает instId для тикера OKX при необходимости."""

        if not symbol:
            return None

        normalized = symbol.replace('-', '').upper()
        return self._okx_symbol_map.get(normalized)


__all__ = ["Config"]
