import time
import logging
import signal
import sys
import os
import importlib
from pathlib import Path
from datetime import datetime

# 👇 Гарантируем, что локальная папка проекта всегда есть в sys.path,
#    чтобы импорты работали даже при запуске скрипта из другой директории
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
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
    logger.setLevel(getattr(logging, Config().LOG_LEVEL, 'INFO'))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Файловый обработчик
    file_handler = logging.FileHandler(Config().LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик с цветами
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_market_snapshot(engine, max_symbols=3):
    """Выводит несколько актуальных котировок bid/ask для наглядности"""
    if not hasattr(engine, 'last_tickers'):
        return

    tickers = getattr(engine, 'last_tickers', {})
    if not tickers:
        logger.info("📉 Нет актуальных котировок для отображения")
        return

    logger.info("📈 Текущие котировки (bid/ask):")
    for symbol in sorted(tickers.keys())[:max_symbols]:
        data = tickers[symbol]
        bid = data.get('bid', 0)
        ask = data.get('ask', 0)
        logger.info(f"   {symbol}: bid={bid:.6f}, ask={ask:.6f}")

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
    config = Config()
    
    logger.info("=" * 70)
    logger.info("🚀 ADVANCED TRIANGULAR ARBITRAGE BOT STARTING 🚀")
    logger.info(f"🔧 Testnet mode: {config.TESTNET}")
    logger.info(f"📈 Monitoring {len(config.TRIANGULAR_PAIRS)} triangular pairs")
    logger.info(f"💰 Min profit threshold: {config.MIN_TRIANGULAR_PROFIT}%")
    logger.info(f"⚖️  Trade amount: {config.TRADE_AMOUNT} USDT")
    logger.info(f"🛡️  Max daily trades: {config.MAX_DAILY_TRADES}")
    logger.info(f"⏰ Update interval: {config.UPDATE_INTERVAL} seconds")
    logger.info(f"📊 Dashboard: http://localhost:{os.getenv('DASHBOARD_PORT', '8050')}")
    logger.info("=" * 70)

    ensure_psutil_available()

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
                # Получение баланса
                balance = engine.client.get_balance('USDT')
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
                    if (balance_usdt > config.TRADE_AMOUNT * 0.5 and 
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