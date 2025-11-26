"""Устаревшая обёртка для прямого импорта Config из пакета."""

import warnings

from arbitrage_bot.core.config import Config

warnings.warn(
    "config.Config устарел; импортируйте Config из arbitrage_bot.core.config",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Config"]
