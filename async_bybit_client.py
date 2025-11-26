"""Устаревшая обёртка для асинхронного клиента Bybit."""

import warnings

from arbitrage_bot.core.async_bybit_client import AsyncBybitClient

warnings.warn(
    "async_bybit_client.AsyncBybitClient устарел; используйте arbitrage_bot.core.async_bybit_client",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AsyncBybitClient"]
