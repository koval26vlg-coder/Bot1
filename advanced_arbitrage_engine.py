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
    def __init__(self):
        self._log_module_origin()
        self._ensure_integrity()

        self.config = Config()
        self._validate_config()

        self.client = BybitClient()
        self.monitor = AdvancedMonitor(self)
        self.real_trader = RealTradingExecutor()
        self.strategy_manager = StrategyManager(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)

        self._initialize_data_structures()
        self._initialize_triangle_stats()

        # Инициализация всех символов
        self._initialize_symbols()

        # Предварительная оптимизация списка треугольников
        self.optimized_triangles = self.performance_optimizer.get_optimized_triangles()

        self.monitor.start_monitoring_loop()
        logger.info("🚀 Advanced Triangular Arbitrage Engine initialized")

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
            if not self._validate_config(new_config):
                logger.error("❌ Перезагрузка конфигурации отменена из-за ошибок валидации")
                return False

            self.config = new_config
            self.strategy_manager.update_config(new_config)
            self.performance_optimizer.update_config(new_config)
            self.client = BybitClient()
            self._initialize_triangle_stats()
            self._initialize_symbols()
            self.optimized_triangles = self.performance_optimizer.get_optimized_triangles()
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
                        base_currency
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
            profit = self._calculate_triangular_profit_path(prices, path, base_currency)

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

    def _calculate_triangular_profit_path(self, prices, path, base_currency):
        """Расчет прибыли для конкретного пути с контролем валют"""
        try:
            initial_amount = 1000.0  # Базовый расчет на 1000 USDT
            current_amount = initial_amount
            current_asset = base_currency

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

                    price = price_data['ask'] if step['price_type'] == 'ask' else price_data['bid']
                    if price <= 0:
                        return -100
                    # Покупаем: количество = текущая сумма / цена
                    quantity = current_amount / price
                    # Применяем комиссию
                    quantity *= (1 - self.config.TRADING_FEE)
                    current_amount = quantity
                    current_asset = symbol_base
                else:  # Sell
                    if current_asset != symbol_base:
                        logger.warning(
                            f"Путь отклонен: продажа {symbol} требует {symbol_base},"
                            f" но текущая валюта {current_asset}"
                        )
                        return -100
                    price = price_data['bid'] if step['price_type'] == 'bid' else price_data['ask']
                    if price <= 0:
                        return -100
                    # Продаем: сумма = количество * цена
                    current_amount = current_amount * price
                    # Применяем комиссию
                    current_amount *= (1 - self.config.TRADING_FEE)
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
