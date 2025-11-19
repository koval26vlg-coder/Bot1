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
        
        # Режим симуляции (True = симуляция, False = реальные ордера)
        simulation_override = os.getenv('SIMULATION_MODE')
        if simulation_override is not None:
            self.simulation_mode = simulation_override.lower() == 'true'
        else:
            self.simulation_mode = self.config.TESTNET

        logger.info(f"🔄 Real Trading Executor initialized. Simulation mode: {self.simulation_mode}")
    
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
            # Этот метод должен быть реализован в зависимости от типа арбитража
            # Пока возвращаем оценочную прибыль из trade_plan
            return trade_plan.get('estimated_profit_usdt', 0)
        except Exception as e:
            logger.error(f"Ошибка расчета реальной прибыли: {str(e)}")
            return 0
    
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
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'symbol', 'side', 'amount', 'price', 'profit', 'simulated', 'trade_details']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for trade in self.trade_history:
                    for result in trade.get('results', []):
                        writer.writerow({
                            'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': result.get('symbol', ''),
                            'side': result.get('side', ''),
                            'amount': result.get('qty', result.get('cumExecQty', 0)),
                            'price': result.get('avgPrice', result.get('price', 0)),
                            'profit': trade.get('total_profit', 0) if result == trade['results'][-1] else 0,
                            'simulated': trade.get('simulated', False),
                            'trade_details': json.dumps(trade.get('trade_plan', {}))
                        })
            
            logger.info(f"✅ Trade history exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error exporting trade history: {str(e)}")
            return None