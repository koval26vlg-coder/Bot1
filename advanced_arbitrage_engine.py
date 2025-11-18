import logging
from collections import defaultdict, deque
from datetime import datetime

from bybit_client import BybitClient
from config import Config
from monitoring import AdvancedMonitor
from real_trading import RealTradingExecutor
# Импортируем менеджер стратегий напрямую из локального модуля без пакета strategies
from indicator_strategies import StrategyManager
# Используем реальный локальный модуль math_stats вместо устаревшего utils.math_stats
from math_stats import mean, rolling_mean


logger = logging.getLogger(__name__)

class AdvancedArbitrageEngine:
    def __init__(self):
        self.config = Config()
        self.client = BybitClient()
        self.monitor = AdvancedMonitor(self)
        self.real_trader = RealTradingExecutor()
        self.strategy_manager = StrategyManager(self.config)

        # Расширенные структуры данных
        self.price_history = {}
        self.volatility_data = {}
        self.trade_history = []
        self.performance_stats = defaultdict(lambda: {'success': 0, 'failures': 0, 'total_profit': 0})
        self.last_arbitrage_time = {}
        self.triangle_cooldown = {}
        self.ohlcv_history = {}
        self.last_strategy_context = {}
        self.last_tickers = {}
        
        # Статистика по треугольникам
        self.triangle_stats = {}
        for triangle in self.config.TRIANGULAR_PAIRS:
            self.triangle_stats[triangle['name']] = {
                'opportunities_found': 0,
                'executed_trades': 0,
                'total_profit': 0,
                'last_execution': None,
                'success_rate': 0
            }
        
        self.ohlcv_history = {}
        self.last_strategy_context = {}

        # Инициализация всех символов
        self._initialize_symbols()
        
        self.monitor.start_monitoring_loop()
        logger.info("🚀 Advanced Triangular Arbitrage Engine initialized")

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
                'spreads': deque(maxlen=500)
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
            if symbol not in self.price_history:
                continue

            bid, ask = data['bid'], data['ask']

            # Обновление истории цен
            self.price_history[symbol]['timestamps'].append(current_time)
            self.price_history[symbol]['bids'].append(bid)
            self.price_history[symbol]['asks'].append(ask)

            # Расчет спреда
            if bid > 0 and ask > 0:
                spread = ((ask - bid) / bid) * 100
                self.price_history[symbol]['spreads'].append(spread)

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

    def analyze_market_conditions(self):
        """Анализ рыночных условий для оптимизации арбитража"""
        market_analysis = {
            'overall_volatility': 0,
            'best_triangles': [],
            'market_conditions': 'normal'
        }
        
        volatilities = []
        for symbol, data in self.volatility_data.items():
            if data['short_term']:
                vol = mean(data['short_term'])
                volatilities.append(vol)

        if volatilities:
            market_analysis['overall_volatility'] = mean(volatilities)
        
        # Определение рыночных условий
        if market_analysis['overall_volatility'] > 2:
            market_analysis['market_conditions'] = 'high_volatility'
        elif market_analysis['overall_volatility'] < 0.1:
            market_analysis['market_conditions'] = 'low_volatility'

        return market_analysis

    def _build_market_dataframe(self, symbol=None):
        symbol = symbol or self.config.SYMBOLS[0]
        if symbol not in self.ohlcv_history:
            return []

        ohlcv = self.ohlcv_history[symbol]
        if len(ohlcv['close']) < 30:
            return []

        market_rows = []
        for ts, o, h, l, c, v in zip(
            ohlcv['timestamps'],
            ohlcv['open'],
            ohlcv['high'],
            ohlcv['low'],
            ohlcv['close'],
            ohlcv['volume']
        ):
            market_rows.append({
                'timestamp': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            })

        return market_rows

    def evaluate_strategies(self):
        market_data = self._build_market_dataframe()
        if not market_data:
            return None

        closes = [row['close'] for row in market_data if row['close'] is not None]
        price_changes = []
        for previous, current in zip(closes, closes[1:]):
            if previous:
                price_change = ((current - previous) / previous) * 100
                price_changes.append(abs(price_change))

        volatility = rolling_mean(price_changes, window=20, min_periods=5)
        liquidity_values = [row['volume'] for row in market_data[-50:] if row['volume'] is not None]
        liquidity = mean(liquidity_values)

        market_context = {
            'volatility': float(volatility) if volatility is not None else 0.0,
            'liquidity': float(liquidity)
        }

        strategy_result = self.strategy_manager.evaluate(market_data, market_context)
        self.last_strategy_context = market_context

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

    def detect_triangular_arbitrage(self, tickers):
        """Улучшенное обнаружение треугольного арбитража"""
        opportunities = []
        market_analysis = self.analyze_market_conditions()
        
        # Динамический порог прибыли в зависимости от волатильности
        dynamic_profit_threshold = self.config.MIN_TRIANGULAR_PROFIT
        if market_analysis['market_conditions'] == 'high_volatility':
            dynamic_profit_threshold += 0.1  # Увеличиваем порог при высокой волатильности
        elif market_analysis['market_conditions'] == 'low_volatility':
            dynamic_profit_threshold -= 0.05  # Уменьшаем порог при низкой волатильности
        
        for triangle in sorted(self.config.TRIANGULAR_PAIRS,
                             key=lambda x: x.get('priority', 999)):
            triangle_name = triangle.get('name', 'triangle')
            try:
                # Проверяем доступность всех пар
                if not all(leg in tickers for leg in triangle['legs']):
                    continue
                
                # Проверяем ликвидность
                if not self._check_liquidity(triangle, tickers):
                    continue
                
                # Проверяем волатильность треугольника
                if not self._check_triangle_volatility(triangle):
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
                
                # Выбираем лучшее направление
                best_direction = max(directions, key=lambda x: x['profit_percent'])
                
                if best_direction['profit_percent'] > dynamic_profit_threshold:
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
                    if historical_success > 0.7:  # Повышаем приоритет успешных треугольников␊
                        opportunity['profit_percent'] += 0.0

                    opportunities.append(opportunity)

                    self.triangle_stats[triangle_name]['opportunities_found'] += 1

                    logger.info(f"🔺 {triangle['name']} - Direction {best_direction['direction']} - "
                              f"Profit: {best_direction['profit_percent']:.4f}% - "
                              f"Market: {market_analysis['market_conditions']}")
                    
                    logger.info(f"🔺 {triangle_name} - Direction {best_direction['direction']} - "
                              f"Profit: {best_direction['profit_percent']:.4f}% - "
                              f"Market: {market_analysis['market_conditions']}")

            except Exception as e:
                logger.error(f"Error in triangle {triangle_name}: {str(e)}")
        
        # Сортировка по прибыльности и приоритету
        opportunities.sort(key=lambda x: (x['profit_percent'], -x['priority']), reverse=True)
        return opportunities

    def _calculate_direction(self, prices, triangle, direction):
        """Расчет прибыли для конкретного направления"""
        leg1, leg2, leg3 = triangle['legs']
        base_currency = triangle.get('base_currency', 'USDT')

        if direction in (1, 2):
            legs_sequence = [leg1, leg2, leg3] if direction == 1 else [leg3, leg2, leg1]
            path = self._build_direction_path(
                legs_sequence,
                base_currency,
                triangle.get('name', 'unknown'),
                direction
            )

            if not path:
                profit = -100
            else:
                profit = self._calculate_triangular_profit_path(prices, path, base_currency)

        else:  # direction == 3
            path = self._build_direction_three_path(triangle, base_currency)
            if not path:
                logger.warning(
                    f"Путь для направления 3 отклонен: невозможно построить последовательность "
                    f"торгов для {triangle['name']}"
                )
                profit = -100
            else:
                profit = self._calculate_triangular_profit_path(prices, path, base_currency)

        return {
            'direction': direction,
            'profit_percent': profit,
            'path': path
        }

    def _build_direction_path(self, legs_sequence, base_currency, triangle_name, direction):
        """Итеративное построение пути для направлений 1 и 2"""
        current_asset = base_currency
        path = []

        for symbol in legs_sequence:
            base_cur, quote_cur = self._get_symbol_currencies(symbol)

            if current_asset == quote_cur:
                path.append({'symbol': symbol, 'side': 'Buy', 'price_type': 'ask'})
                current_asset = base_cur
            elif current_asset == base_cur:
                path.append({'symbol': symbol, 'side': 'Sell', 'price_type': 'bid'})
                current_asset = quote_cur
            else:
                logger.warning(
                    f"Невозможно построить путь для {triangle_name} (направление {direction}): "
                    f"текущая валюта {current_asset} не подходит для сделки {symbol}"
                )
                return None

        if current_asset != base_currency:
            logger.warning(
                f"Путь для {triangle_name} (направление {direction}) не возвращает базовую валюту {base_currency}"
            )
            return None

        return path

    def _build_direction_three_path(self, triangle, base_currency):
        """Построение альтернативного пути (направление №3) с контролем валют"""
        leg1, leg2, leg3 = triangle['legs']
        first_base, first_quote = self._get_symbol_currencies(leg1)

        if base_currency != first_quote:
            logger.warning(
                f"Путь отклонен: для первого шага {leg1} требуется {first_quote}, "
                f"но базовая валюта треугольника {base_currency}"
            )
            return None

        initial_step = {'symbol': leg1, 'side': 'Buy', 'price_type': 'ask'}
        current_asset = first_base

        remaining_orders = [
            (leg2, leg3),
            (leg3, leg2)
        ]

        for order in remaining_orders:
            path = [initial_step.copy()]
            asset = current_asset
            valid_path = True

            for symbol in order:
                base_cur, quote_cur = self._get_symbol_currencies(symbol)

                if asset == quote_cur:
                    path.append({'symbol': symbol, 'side': 'Buy', 'price_type': 'ask'})
                    asset = base_cur
                elif asset == base_cur:
                    path.append({'symbol': symbol, 'side': 'Sell', 'price_type': 'bid'})
                    asset = quote_cur
                else:
                    valid_path = False
                    logger.debug(
                        f"Невозможно использовать {symbol} на шаге направления 3: текущая валюта {asset}"
                    )
                    break

            if valid_path and asset == base_currency:
                return path

        logger.warning(
            f"Не найден валидный альтернативный путь для {triangle['name']} (направление 3)"
        )
        return None

    def _get_symbol_currencies(self, symbol):
        """Определение базовой и котируемой валюты символа"""
        known_quotes = [
            'USDT', 'USDC', 'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT',
            'LINK', 'MATIC', 'AVAX', 'XRP', 'DOGE', 'LTC', 'TRX', 'ETC'
        ]

        for quote in sorted(known_quotes, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:
                    return base, quote

        # Резервный случай для неизвестных пар
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

    def _check_liquidity(self, triangle, tickers):
        """Проверка ликвидности для треугольника"""
        for symbol in triangle['legs']:
            if symbol not in tickers:
                return False
            
            bid, ask = tickers[symbol]['bid'], tickers[symbol]['ask']
            if bid <= 0 or ask <= 0:
                return False
            
            # Проверка спреда
            spread = ((ask - bid) / bid) * 100
            if spread > self.config.MAX_SPREAD_PERCENT:
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
            # Фильтруем слишком волатильные треугольники␊
            return avg_volatility < 5.0  # Максимум 5% волатильность␊
        
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
        """Выполнение треугольного арбитража с улучшенным отслеживанием"""
        logger.info(f"🔺 Executing triangular arbitrage: {opportunity['triangle_name']}")
        
        start_time = datetime.now()
        
        try:
            # Проверяем, не изменились ли условия
            current_tickers = self.client.get_tickers(opportunity['symbols'])
            if not self._validate_opportunity_still_exists(opportunity, current_tickers):
                logger.warning("❌ Opportunity disappeared before execution")
                return False
            
            # Исполняем сделку
            trade_result = self.real_trader.execute_arbitrage_trade(trade_plan)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if trade_result:
                # Обновляем статистику треугольника
                triangle_name = opportunity['triangle_name']
                self.triangle_stats[triangle_name]['executed_trades'] += 1
                self.triangle_stats[triangle_name]['total_profit'] += trade_plan['estimated_profit_usdt']
                self.triangle_stats[triangle_name]['last_execution'] = datetime.now()
                
                # Расчет успешности
                total_trades = self.triangle_stats[triangle_name]['executed_trades']
                successful_trades = self.triangle_stats[triangle_name]['executed_trades']  # Пока все успешные
                self.triangle_stats[triangle_name]['success_rate'] = successful_trades / total_trades
                
                logger.info(f"✅ Triangular arbitrage executed successfully! "
                          f"Time: {execution_time:.2f}s, "
                          f"Profit: {trade_plan['estimated_profit_usdt']:.4f} USDT")
                
                # Запись расширенной информации о сделке
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': opportunity['triangle_name'],
                    'type': 'triangular',
                    'profit': trade_plan['estimated_profit_usdt'],
                    'profit_percent': opportunity['profit_percent'],
                    'direction': opportunity['direction'],
                    'execution_time': execution_time,
                    'market_conditions': opportunity['market_conditions'],
                    'triangle_stats': self.triangle_stats[triangle_name],
                    'details': {
                        'triangle': opportunity['triangle_name'],
                        'symbols': opportunity['symbols'],
                        'direction': opportunity['direction'],
                        'initial_amount': trade_plan['initial_amount'],
                        'execution_path': opportunity['execution_path'],
                        'real_executed': True
                    }
                }
                
                # Передаем запись в мониторинг
                if hasattr(self, 'monitor') and self.monitor:
                    self.monitor.track_trade(trade_record)
                
                self._record_trade(opportunity, trade_plan, trade_result.get('results', []))
                return True
            else:
                logger.error("❌ Triangular arbitrage execution failed")
                # Обновляем статистику неудач
                triangle_name = opportunity['triangle_name']
                self.triangle_stats[triangle_name]['failures'] += 1
                return False
                
        except Exception as e:
            logger.error(f"🔥 Critical error executing triangular arbitrage: {str(e)}", exc_info=True)
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'notify_alert'):
                self.monitor.notify_alert(f"Ошибка треугольного арбитража: {str(e)}", "critical")
            return False

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

    def _record_trade(self, opportunity, trade_plan, orders):
        """Запись информации о сделке в историю"""
        trade_record = {
            'timestamp': datetime.now(),
            'type': opportunity['type'],
            'triangle_name': opportunity['triangle_name'],
            'profit_percent': opportunity['profit_percent'],
            'estimated_profit_usdt': trade_plan.get('estimated_profit_usdt', 0),
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
        return {
            'mode': getattr(self.config, 'STRATEGY_MODE', 'adaptive'),
            'active': self.strategy_manager.get_active_strategy_name(),
            'context': self.last_strategy_context,
            'strategies': self.strategy_manager.get_strategy_snapshot()
        }

    def get_strategy_status(self):
        return {
            'mode': self.config.STRATEGY_MODE,
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

        # Оцениваем стратегии до обновления буферов
        strategy_result = self.evaluate_strategies()
        active_strategy_name = self.strategy_manager.get_active_strategy_name()
        logger.info(
            "⚙️ Strategy mode=%s | Active=%s",
            self.config.STRATEGY_MODE,
            active_strategy_name
        )

        # Обновляем рыночные данные
        self.update_market_data(tickers)

        # Сохраняем последние цены для визуализации
        self.last_tickers = tickers

        # Обнаружение треугольного арбитража
        opportunities = self.detect_triangular_arbitrage(tickers)

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

    def execute_arbitrage(self, opportunity):
        """Основной метод выполнения арбитража"""
        symbol = opportunity.get('triangle_name', 'triangular')
        
        # Проверка кулдауна
        if not self.check_cooldown(symbol):
            return False
        
        logger.info(f"🎯 Executing arbitrage: {opportunity['triangle_name']}")
        logger.info(f"   Profit: {opportunity['profit_percent']:.4f}%")
        logger.info(f"   Market: {opportunity['market_conditions']}")
        
        # Получаем баланс
        balance = {'available': 100.0}  # Временно для тестнета
        balance_usdt = balance['available']
        
        if balance_usdt < max(5, self.config.TRADE_AMOUNT * 0.1):
            logger.warning(f"❌ Insufficient balance. Available: {balance_usdt:.2f} USDT")
            self.monitor.check_balance_health(balance_usdt)
            return False
        
        # Рассчитываем объемы сделок
        trade_plan = self.calculate_advanced_trade(opportunity, balance_usdt)
        
        if not trade_plan:
            logger.error("❌ Failed to calculate trade amounts")
            return False
        
        logger.info(f"📋 Trade plan: Initial amount: {trade_plan['initial_amount']} USDT, "
                  f"Estimated profit: {trade_plan['estimated_profit_usdt']:.4f} USDT")
        
        # Выполняем арбитраж
        success = self.execute_triangular_arbitrage(opportunity, trade_plan)
        
        if success:
            self.last_arbitrage_time[symbol] = datetime.now()
            logger.info("✅ Arbitrage execution completed successfully")
            
            # Отправляем отчет о производительности каждые 10 сделок
            if len(self.trade_history) % 10 == 0:
                performance_report = self.get_triangle_performance_report()
                if hasattr(self, 'monitor') and hasattr(self.monitor, 'notify_performance'):
                    self.monitor.notify_performance(performance_report)
        else:
            logger.error("❌ Arbitrage execution failed")

        return success

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
            logger.info(f"⏳ Cooldown active for {symbol}: {remaining:.1f} seconds remaining")
            self.monitor.track_cooldown_violation(symbol)
            return False

        return True