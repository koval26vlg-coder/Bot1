import logging
import time
import json
import csv
import statistics
from datetime import datetime
import asyncio
import importlib.util
import threading
from collections import Counter

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from arbitrage_bot.core.config import Config

psutil = None
if importlib.util.find_spec('psutil') is not None:
    import psutil

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

    def _monitor_tick(self, now: float | None = None):
        """Единичный проход мониторинга со всеми проверками."""

        current_ts = now or time.time()

        if int(current_ts) % 30 == 0:
            self.track_system_metrics()

        if int(current_ts) % 3600 == 0:
            self.generate_performance_report()

        if int(current_ts) % 300 == 0:
            health = self.health_check()
            if health.get('status') != 'healthy':
                logger.warning(f"⚠️ Состояние системы: {health['status']} - {health}")

    async def monitor_tick_async(self):
        """Асинхронный враппер для интеграции мониторинга в event loop."""

        await asyncio.to_thread(self._monitor_tick, time.time())

    def _format_numeric_metric(self, value, precision: int = 4) -> str:
        """Аккуратно форматирует числовые метрики для логов."""

        if value is None:
            return "н/д"
        if isinstance(value, float) and value == float("inf"):
            return "∞"
        return f"{value:.{precision}f}"

    def _calculate_performance_metrics(self) -> dict:
        """Вычисляет расширенные показатели эффективности сделок."""

        if not self.trade_history:
            return {}

        profits = [trade.get('profit', 0) for trade in self.trade_history]
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = -sum(p for p in profits if p < 0)
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]

        profit_factor = None
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float('inf')

        win_rate = (len(win_trades) / len(profits) * 100) if profits else 0.0

        avg_win = sum(win_trades) / len(win_trades) if win_trades else 0.0
        avg_loss = -sum(loss_trades) / len(loss_trades) if loss_trades else 0.0
        avg_win_loss_ratio = None
        if avg_loss > 0:
            avg_win_loss_ratio = avg_win / avg_loss
        elif avg_win > 0:
            avg_win_loss_ratio = float('inf')

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for profit in profits:
            cumulative += profit
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        volatility = 0.0
        if len(profits) > 1:
            volatility = statistics.pstdev(profits)
        average_return = statistics.mean(profits) if profits else 0.0
        sharpe_ratio = average_return / volatility if volatility > 0 else 0.0

        return {
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win_loss_ratio': avg_win_loss_ratio,
        }

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
            f"Рекомендуется оптимизировать код или увеличить реурсы сервера.",
        )

    def track_trade(self, trade_data):
        """Отслеживание сделки"""
        execution_path = (trade_data.get('details') or {}).get('execution_path')
        if isinstance(execution_path, list):
            market_types = [step.get('market_type') for step in execution_path if isinstance(step, dict)]
            filtered_types = [m for m in market_types if m]
            if filtered_types:
                trade_data['market_mix'] = dict(Counter(filtered_types))

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
                f"Рекомендуется проверить логику кулдауна.",
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
                f"Рекомендуется перезапустить бота или проверить API ключи.",
            )

    def check_balance_health(self, balance_usdt):
        """Проверка здоровья баланса"""
        if balance_usdt < self.alert_thresholds['min_balance']:
            logger.error(
                f"⚠️ Низкий баланс:\n"
                f"Текущий баланс: {balance_usdt:.2f} USDT\n"
                f"Минимальный порог: {self.alert_thresholds['min_balance']} USDT\n"
                f"Торговля может быть приостановлена из-за недостатка средств.",
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

        performance_metrics = self._calculate_performance_metrics()

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
            },
            'performance_metrics': performance_metrics,
        }

        logger.info(
            f"📊 Отчет о производительности:\n"
            f"   Всего сделок: {report['total_trades']}\n"
            f"   Успешных: {report['successful_trades']} ({report['success_rate']:.1f}%)\n"
            f"   Общая прибыль: {report['total_profit']:.4f} USDT\n"
            f"   Средняя прибыль: {report['avg_profit']:.4f} USDT\n"
            f"   Время работы: {report['runtime']}\n"
            f"   📈 Profit Factor: {self._format_numeric_metric(performance_metrics.get('profit_factor'))}\n"
            f"   ⚖️ Sharpe Ratio: {self._format_numeric_metric(performance_metrics.get('sharpe_ratio'))}\n"
            f"   📉 Max Drawdown: {performance_metrics.get('max_drawdown', 0.0):.4f} USDT\n"
            f"   🎯 Win Rate: {performance_metrics.get('win_rate', 0.0):.1f}%\n"
            f"   ⚔️ Avg Win/Loss: {self._format_numeric_metric(performance_metrics.get('avg_win_loss_ratio'))}"
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
        def monitoring_loop():
            while True:
                try:
                    self._monitor_tick(time.time())
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(60)  # При ошибке ждем минуту

        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info("🔄 Advanced monitoring loop started")


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
                    hovertemplate=f"{symbol}<br>Price: %{y:.2f} USDT<br>Time: %{x}<extra></extra>"
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


__all__ = ["AdvancedMonitor", "Dashboard"]
