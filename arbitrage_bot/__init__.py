"""Пакет с компонентами арбитражного бота."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

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
    "advanced_main",
    "main",
    "run_advanced_bot",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AdvancedArbitrageEngine": ("arbitrage_bot.core.advanced_arbitrage_engine", "AdvancedArbitrageEngine"),
    "AdvancedMonitor": ("arbitrage_bot.monitoring.monitoring", "AdvancedMonitor"),
    "BybitClient": ("arbitrage_bot.exchanges.bybit_client", "BybitClient"),
    "BybitWebSocketManager": ("arbitrage_bot.exchanges.bybit_client", "BybitWebSocketManager"),
    "Config": ("arbitrage_bot.core.config", "Config"),
    "Dashboard": ("arbitrage_bot.monitoring.visualization", "Dashboard"),
    "HistoricalReplayer": ("arbitrage_bot.core.advanced_bot", "HistoricalReplayer"),
    "OkxClient": ("arbitrage_bot.exchanges.okx_client", "OkxClient"),
    "OptimizedConfig": ("arbitrage_bot.core.optimized_config", "OptimizedConfig"),
    "RealTradingExecutor": ("arbitrage_bot.core.real_trading", "RealTradingExecutor"),
    "StrategyManager": ("arbitrage_bot.strategies.indicator_strategies", "StrategyManager"),
    "advanced_main": ("arbitrage_bot.core.engine", "run_advanced_bot"),
    "main": ("arbitrage_bot.core.engine", "run_advanced_bot"),
    "run_advanced_bot": ("arbitrage_bot.core.engine", "run_advanced_bot"),
}


def _load_export(name: str) -> Any:
    """Лениво импортирует целевой объект, избегая побочных эффектов."""

    module_name, attr_name = _EXPORTS[name]
    module: ModuleType = importlib.import_module(module_name)
    value: Any = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Возвращает экспортируемый атрибут по требованию."""

    if name in _EXPORTS:
        return _load_export(name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Возвращает доступные атрибуты модуля."""

    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))
