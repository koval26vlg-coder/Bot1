"""Пакет с компонентами арбитражного бота."""

from arbitrage_bot.core.advanced_arbitrage_engine import AdvancedArbitrageEngine
from arbitrage_bot.core.advanced_bot import HistoricalReplayer, main
from arbitrage_bot.core.config import Config
from arbitrage_bot.core.optimized_config import OptimizedConfig
from arbitrage_bot.core.real_trading import RealTradingExecutor
from arbitrage_bot.exchanges.bybit_client import BybitClient, BybitWebSocketManager
from arbitrage_bot.exchanges.okx_client import OkxClient
from arbitrage_bot.monitoring.monitoring import AdvancedMonitor
from arbitrage_bot.monitoring.visualization import Dashboard
from arbitrage_bot.strategies.indicator_strategies import StrategyManager

__all__ = [
    "AdvancedArbitrageEngine",
    "AdvancedMonitor",
    "BybitClient",
    "BybitWebSocketManager",
    "Config",
    "Dashboard",
    "HistoricalReplayer",
    "OkxClient",
    "OptimizedConfig",
    "RealTradingExecutor",
    "StrategyManager",
    "main",
]
