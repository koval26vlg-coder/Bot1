"""Вспомогательные утилиты для контекстного логирования."""

from __future__ import annotations

import logging
import uuid


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [mode=%(mode)s env=%(environment)s cycle=%(cycle_id)s]: %(message)s"


class ContextFilter(logging.Filter):
    """Добавляет обязательные поля контекста в запись лога."""

    def __init__(self, default_mode: str = "-", default_environment: str = "-"):
        super().__init__()
        self._default_mode = default_mode
        self._default_environment = default_environment

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.mode = getattr(record, "mode", self._default_mode)
        record.environment = getattr(record, "environment", self._default_environment)
        record.cycle_id = getattr(record, "cycle_id", "-")
        return True


def configure_root_logging(
    level: str,
    *,
    mode: str,
    environment: str,
    handlers: list[logging.Handler] | None = None,
) -> logging.Logger:
    """Настраивает корневой логгер и применяет фильтр контекста."""

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)
    context_filter = ContextFilter(mode, environment)

    target_handlers = handlers or [logging.StreamHandler()]
    for handler in target_handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root_logger.addHandler(handler)

    return root_logger


def create_adapter(
    logger: logging.Logger,
    *,
    mode: str,
    environment: str,
    cycle_id: str | None = None,
) -> logging.LoggerAdapter:
    """Создает адаптер с заполненными полями контекста."""

    return logging.LoggerAdapter(
        logger,
        {
            "mode": mode,
            "environment": environment,
            "cycle_id": cycle_id or "-",
        },
    )


def generate_cycle_id() -> str:
    """Возвращает новый идентификатор цикла."""

    return uuid.uuid4().hex[:8]
