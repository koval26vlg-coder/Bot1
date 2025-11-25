"""Совместимость с устаревшим монолитным модулем."""

from arbitrage_bot import (
    AdvancedArbitrageEngine,
    AdvancedMonitor,
    BybitClient,
    BybitWebSocketManager,
    Config,
    Dashboard,
    HistoricalReplayer,
    OkxClient,
    OptimizedConfig,
    RealTradingExecutor,
    StrategyManager,
    main,
)
from arbitrage_bot.utils.math_stats import mean, rolling_mean
from arbitrage_bot.utils.performance_optimizer import PerformanceOptimizer

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
    "PerformanceOptimizer",
    "RealTradingExecutor",
    "StrategyManager",
    "main",
    "mean",
    "rolling_mean",
]
