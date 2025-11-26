"""Публичный API арбитражного бота без ленивых импортов и прокси."""

from arbitrage_bot.core.advanced_arbitrage_engine import AdvancedArbitrageEngine
from arbitrage_bot.core.async_bybit_client import AsyncBybitClient
from arbitrage_bot.core.config import Config
from arbitrage_bot.core.engine import run_advanced_bot
from arbitrage_bot.core.optimized_config import OptimizedConfig
from arbitrage_bot.core.real_trading import RealTradingExecutor
from arbitrage_bot.exchanges.bybit_client import BybitClient, BybitWebSocketManager
from arbitrage_bot.exchanges.okx_client import OkxClient
from arbitrage_bot.monitoring.monitoring import AdvancedMonitor
from arbitrage_bot.monitoring.visualization import Dashboard
from arbitrage_bot.strategies.indicator_strategies import StrategyManager
from arbitrage_bot.core.advanced_bot import HistoricalReplayer

advanced_main = run_advanced_bot
main = run_advanced_bot

__all__ = [
    "AdvancedArbitrageEngine",
    "AdvancedMonitor",
    "AsyncBybitClient",
    "BybitClient",
    "BybitWebSocketManager",
    "Config",
    "Dashboard",
    "HistoricalReplayer",
    "OkxClient",
    "OptimizedConfig",
    "RealTradingExecutor",
    "StrategyManager",
    "advanced_main",
    "main",
    "run_advanced_bot",
]
