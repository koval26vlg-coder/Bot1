"""Устаревшая обёртка для клиентов Bybit с прямыми импортами."""

import warnings

from arbitrage_bot.exchanges.bybit_client import BybitClient, BybitWebSocketManager

warnings.warn(
    "bybit_client.* устарели; импортируйте классы из arbitrage_bot.exchanges.bybit_client",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BybitWebSocketManager", "BybitClient"]
