import logging
import time
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

class RealTradingExecutor:
    """Исполнение реальных ордеров с режимом симуляции и постепенного перехода к реальной торговле"""
    
    def __init__(self):
        self.config = Config()
        self.client = BybitClient()
        self.is_real_mode = False
        self.trade_history = []
        self.risk_manager = RiskManager()
        # Фиктивный баланс для симуляции, чтобы можно было управлять проверками ликвидности
        self._simulated_balance_usdt = self._load_simulated_balance()

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
    
    def _execute_real_trade(self, trade_plan):
        """Реальное исполнение торговли"""
        logger.warning("🔥 REAL MODE: Выполнение реальных ордеров")
        
        if not self.risk_manager.can_execute_trade(trade_plan):
            logger.error("❌ Риск-менеджер запретил выполнение сделки")
            return None
        
        try:
            results = []
            total_profit = 0
            
            # Выполняем ордера последовательно
            for step_name, step in trade_plan.items():
                if step_name.startswith('step') or step_name in ['leg1', 'leg2']:
                    order_result = self.client.place_order(
                        symbol=step['symbol'],
                        side=step['side'],
                        qty=step['amount'],
                        price=step.get('price'),
                        order_type=step.get('type', 'Limit')
                    )
                    
                    if order_result:
                        results.append(order_result)
                        logger.info(f"✅ REAL ORDER: {step['side']} {step['amount']:.6f} {step['symbol']} @ {step.get('price', '_MARKET_')}")
                    else:
                        logger.error(f"❌ FAILED ORDER: {step['side']} {step['amount']:.6f} {step['symbol']}")
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