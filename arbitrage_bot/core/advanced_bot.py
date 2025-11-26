import importlib
import asyncio
import csv
import inspect
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .advanced_arbitrage_engine import AdvancedArbitrageEngine
from .optimized_config import OptimizedConfig
from logging_utils import configure_root_logging, create_adapter, generate_cycle_id

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_psutil_available():
    """Проверяет доступность psutil перед запуском мониторинга"""
    if importlib.util.find_spec("psutil") is None:
        message = (
            "❗ Модуль psutil не найден. Установите зависимости командой "
            "'pip install -r requirements.txt'."
        )
        print(message, file=sys.stderr)
        sys.exit(1)

def setup_logging(mode: str, environment: str, *, cycle_id: str | None = None):
    """Настройка расширенного логирования с контекстом."""

    log_level = getattr(logging, OptimizedConfig().LOG_LEVEL.upper(), logging.INFO)
    file_handler = logging.FileHandler(OptimizedConfig().LOG_FILE)
    console_handler = logging.StreamHandler()
    handlers = [file_handler, console_handler]

    configure_root_logging(
        logging.getLevelName(log_level),
        mode=mode,
        environment=environment,
        handlers=handlers,
    )

    return create_adapter(
        logging.getLogger(__name__),
        mode=mode,
        environment=environment,
        cycle_id=cycle_id,
    )


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


async def _async_trading_loop(engine: AdvancedArbitrageEngine, killer: GracefulKiller):
    """Асинхронный главный цикл, использующий event loop для сбора котировок."""

    iteration_count = 0
    start_time = datetime.now()
    total_opportunities_found = 0
    update_interval = getattr(engine.config, 'UPDATE_INTERVAL', 3)

    while not killer.kill_now:
        logger.extra["cycle_id"] = generate_cycle_id()
        iteration_count += 1
        cycle_start = time.time()

        if iteration_count % 10 == 0:
            logger.info(f"\n{'=' * 30} Iteration #{iteration_count} {'=' * 30}")
            logger.info(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🕐 Running for: {str(datetime.now() - start_time).split('.')[0]}")

        try:
            balance = await asyncio.to_thread(engine.get_effective_balance, 'USDT')
            balance_usdt = balance['available']

            if iteration_count % 10 == 0:
                logger.info(f"💰 Account balance: {balance_usdt:.2f} USDT available")

            opportunities = await engine.detect_opportunities_async()
            if iteration_count % 5 == 0:
                await asyncio.to_thread(log_market_snapshot, engine)
            total_opportunities_found += len(opportunities)

            if opportunities:
                if iteration_count % 5 == 0:
                    logger.info(f"🎯 Found {len(opportunities)} triangular arbitrage opportunities")

                best_opportunity = opportunities[0]
                balance_check_passed = balance_usdt > engine.config.TRADE_AMOUNT * 0.5
                if engine.real_trader.simulation_mode:
                    balance_check_passed = True

                if balance_check_passed and engine.check_cooldown(best_opportunity['triangle_name']):
                    logger.info(
                        f"⭐ Selected: {best_opportunity['triangle_name']} - "
                        f"{best_opportunity['profit_percent']:.4f}% profit"
                    )

                    success = await asyncio.to_thread(engine.execute_arbitrage, best_opportunity)

                    if success:
                        logger.info("✅ SUCCESS! Triangular arbitrage executed")

                        if len(engine.trade_history) % 5 == 0:
                            report = engine.get_triangle_performance_report()
                            logger.info(
                                f"📊 Performance: {report['total_executed_trades']} trades, "
                                f"Total profit: {report['total_profit']:.4f} USDT"
                            )
                    else:
                        logger.error("❌ FAILED! Arbitrage execution failed")
                else:
                    if iteration_count % 20 == 0:
                        logger.info("🔍 Opportunities found but skipped due to risk management")

            elapsed = time.time() - cycle_start
            await asyncio.sleep(max(0, update_interval - elapsed))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка в асинхронном торговом цикле: %s", exc)
            await asyncio.sleep(update_interval)

def main(logger_adapter=None, *, mode: str = "standard", environment: str | None = None):
    """Основная функция запуска улучшенного бота с контекстным логированием."""

    global logger

    config = OptimizedConfig()
    effective_environment = environment or (
        "simulation"
        if os.getenv("SIMULATION_MODE", "false").lower() == "true"
        else "testnet"
        if config.TESTNET
        else "production"
    )
    cycle_id = generate_cycle_id()

    if logger_adapter is not None:
        logger = logger_adapter
        logger.extra.update({
            "mode": mode,
            "environment": effective_environment,
            "cycle_id": cycle_id,
        })
    else:
        logger = setup_logging(mode, effective_environment, cycle_id=cycle_id)

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

    if engine._should_use_async_market():
        logger.info("🚦 Включён асинхронный режим сбора котировок через AsyncBybitClient")
        asyncio.run(_async_trading_loop(engine, killer))
        return

    try:
        iteration_count = 0
        start_time = datetime.now()
        total_opportunities_found = 0

        while not killer.kill_now:
            logger.extra["cycle_id"] = generate_cycle_id()
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
                logger.info(
                    f"   {name}: {stats['executed_trades']} trades, "
                    f"{stats['total_profit']:.4f} USDT profit, "
                    f"{stats['success_rate']:.1%} success rate"
                )

        logger.info("=" * 70)


class HistoricalReplayer:
    """Стресс-тестирует движок на исторических данных (режим replay)."""

    def __init__(self, engine: AdvancedArbitrageEngine, data_path: str, *, speed: float = 1.0, max_records: int | None = None):
        self.engine = engine
        self.data_path = Path(data_path)
        self.speed = max(speed, 0.001)
        self.max_records = max_records if max_records and max_records > 0 else None

    def _parse_timestamp(self, raw_ts):
        """Преобразует строковое время в datetime для корректного воспроизведения задержек."""

        if not raw_ts:
            return None

        try:
            return datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
        except ValueError:
            try:
                return datetime.fromtimestamp(float(raw_ts))
            except (TypeError, ValueError):
                return None

    def replay(self):
        """Проигрывает исторические котировки, обновляя движок и вычисляя арбитраж."""

        if not self.data_path.exists():
            logger.error("❌ Файл исторических данных не найден: %s", self.data_path)
            return False

        logger.info(
            "🚦 Запуск стресс-теста на исторических данных: %s (скорость x%.2f)",
            self.data_path,
            self.speed,
        )

        processed = 0
        last_timestamp = None

        with self.data_path.open('r', encoding='utf-8') as history_file:
            reader = csv.DictReader(history_file)

            for row in reader:
                if self.max_records and processed >= self.max_records:
                    break

                symbol = row.get('symbol')
                if not symbol:
                    continue

                ticker = {
                    'bid': self.engine._safe_float(row.get('bid')) if hasattr(self.engine, '_safe_float') else float(row.get('bid') or 0),
                    'ask': self.engine._safe_float(row.get('ask')) if hasattr(self.engine, '_safe_float') else float(row.get('ask') or 0),
                    'bid_size': float(row.get('bid_size') or row.get('bidSize') or 0),
                    'ask_size': float(row.get('ask_size') or row.get('askSize') or 0),
                    'last_price': float(row.get('last_price') or row.get('last') or 0),
                    'volume': float(row.get('volume') or 0),
                }

                current_ts = self._parse_timestamp(row.get('timestamp'))
                if last_timestamp and current_ts:
                    delay = max((current_ts - last_timestamp).total_seconds() / self.speed, 0)
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                if current_ts:
                    last_timestamp = current_ts

                self.engine.update_market_data({symbol: ticker})
                self.engine.last_tickers = {symbol: ticker}
                self.engine.detect_triangular_arbitrage({symbol: ticker}, None)
                processed += 1

        logger.info("✅ Стресс-тест завершён, обработано %s записей", processed)
        return True


__all__ = ["HistoricalReplayer", "main"]
