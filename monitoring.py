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
        if self.cooldown_violations >= 5:  # 5 нарушений подряд
            logger.critical(
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