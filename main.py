"""Единая точка входа с CLI для разных режимов запуска бота."""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from arbitrage_bot.core.engine import AdvancedArbitrageEngine, HistoricalReplayer, run_advanced_bot
from arbitrage_bot.core.optimized_config import OptimizedConfig
from logging_utils import configure_root_logging, create_adapter, generate_cycle_id

advanced_main = run_advanced_bot


def signal_handler(sig, frame):
    """Глобальный обработчик KeyboardInterrupt для плавного завершения."""

    print("\n\n🛑 Получен сигнал прерывания (Ctrl+C). Выполняется плавное завершение...")
    print("⏳ Ожидание завершения текущих операций...")
    shutdown_flag.set()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
shutdown_flag = threading.Event()


@dataclass(frozen=True)
class ModeDefinition:
    """Описание режима CLI: требуемые переменные окружения, конфигурация и раннер."""

    description: str
    env_builder: Callable[[argparse.Namespace], Dict[str, str]]
    config_builder: Optional[
        Callable[[argparse.Namespace, logging.LoggerAdapter], Optional[OptimizedConfig]]
    ]
    runner: Callable[[argparse.Namespace, logging.LoggerAdapter, str, Optional[OptimizedConfig]], None]


def _configure_logging(level: str, mode: str, environment: str):
    """Создает адаптер логирования с дополнительными полями контекста."""

    configure_root_logging(level, mode=mode, environment=environment)
    return create_adapter(
        logging.getLogger(__name__),
        mode=mode,
        environment=environment,
        cycle_id=generate_cycle_id(),
    )


def _build_threshold_env(
    min_profit: float | int | str | None,
    trade_amount: float | int | str | None,
    *,
    default_min_profit: float | int | str | None = None,
    default_trade_amount: float | int | str | None = None,
) -> Dict[str, str]:
    """Формирует словарь переменных окружения для порогов и суммы сделки."""

    env: Dict[str, str] = {}
    if min_profit is not None or default_min_profit is not None:
        env["MIN_TRIANGULAR_PROFIT"] = str(
            min_profit if min_profit is not None else default_min_profit
        )
    if trade_amount is not None or default_trade_amount is not None:
        env["TRADE_AMOUNT"] = str(
            trade_amount if trade_amount is not None else default_trade_amount
        )
    return env


def _standard_env(args: argparse.Namespace) -> Dict[str, str]:
    """Пороговые настройки для стандартного запуска."""

    return _build_threshold_env(args.min_profit, args.trade_amount)


def _aggressive_env(args: argparse.Namespace) -> Dict[str, str]:
    """Расширенные настройки для агрессивного режима."""

    default_trade_amount = os.environ.get("TRADE_AMOUNT", 25)

    return {
        "TESTNET": "true",
        "SIMULATION_MODE": "true",
        **_build_threshold_env(
            args.min_profit,
            args.trade_amount,
            default_min_profit=0.01,
            default_trade_amount=default_trade_amount,
        ),
    }


def _quick_env(args: argparse.Namespace) -> Dict[str, str]:
    """Включает тестнет для быстрого теста и применяет пользовательские пороги."""

    return {
        "TESTNET": "true",
        **_build_threshold_env(args.min_profit, args.trade_amount),
    }


def _replay_env(_: argparse.Namespace) -> Dict[str, str]:
    """Включает режим симуляции для воспроизведения исторических данных."""

    return {"SIMULATION_MODE": "true", "PAPER_TRADING_MODE": "true"}


def _prepare_quick_config(
    min_profit: float | None,
    trade_amount: float | None,
    logger,
) -> OptimizedConfig:
    """Создает конфигурацию для быстрого тестового цикла."""

    optimized = OptimizedConfig()
    optimized.TESTNET = True

    if min_profit is not None:
        optimized._min_triangular_profit_override = min_profit
    if trade_amount is not None:
        optimized._TRADE_AMOUNT = trade_amount

    logger.info("Используется OptimizedConfig в режиме тестнета")

    if trade_amount is not None:
        logger.info(
            "Применён пользовательский объём сделки для режима quick: %s USDT",
            optimized.TRADE_AMOUNT,
        )
    else:
        logger.info(
            "Используется стандартный объём сделки для режима quick: %s USDT",
            optimized.TRADE_AMOUNT,
        )
    return optimized


def _quick_test(engine, logger) -> None:
    """Выполняет три цикла поиска возможностей с выводом результатов."""

    for cycle in range(1, 4):
        logger.extra["cycle_id"] = generate_cycle_id()
        start_time = time.time()
        opportunities = engine.detect_opportunities()
        duration = time.time() - start_time

        if opportunities:
            logger.info("Цикл %s: найдено возможностей %s", cycle, len(opportunities))
            logger.info("Список возможностей: %s", opportunities)
        else:
            logger.info("Цикл %s: возможности не найдены", cycle)

        logger.info("Время цикла %s: %.2f секунд", cycle, duration)
        if duration > 30:
            logger.warning(
                "Цикл %s превысил 30 секунд и может требовать оптимизации", cycle
            )


def _apply_environment(overrides: Dict[str, str]) -> None:
    """Применяет вычисленные переменные окружения единообразно."""

    for key, value in overrides.items():
        os.environ[key] = str(value)


def _resolve_environment() -> str:
    """Определяет окружение для логирования."""

    simulation_mode = os.getenv("SIMULATION_MODE", "false").lower() == "true"
    if simulation_mode:
        return "simulation"

    testnet_enabled = os.getenv("TESTNET", "false").lower() == "true"
    if testnet_enabled:
        return "testnet"

    return os.getenv("ENVIRONMENT", "production")


def _run_standard_mode(
    args: argparse.Namespace,
    logger: logging.LoggerAdapter,
    environment: str,
    _: Optional[OptimizedConfig],
) -> None:
    """Стандартный запуск улучшенного бота."""

    advanced_main(logger_adapter=logger, mode=args.mode, environment=environment)


def _run_quick_mode(
    args: argparse.Namespace,
    logger: logging.LoggerAdapter,
    _: str,
    optimized_config: Optional[OptimizedConfig],
) -> None:
    """Быстрый тестовый прогон арбитражного движка с оптимизированной конфигурацией."""

    if optimized_config is None:
        optimized_config = OptimizedConfig()
    engine = AdvancedArbitrageEngine(config=optimized_config)

    logger.info("Запуск тестового цикла обнаружения возможностей")
    _quick_test(engine, logger)


def _run_replay_mode(
    args: argparse.Namespace,
    logger: logging.LoggerAdapter,
    _: str,
    __: Optional[OptimizedConfig],
) -> None:
    """Стресс-тест движка на исторических данных (режим replay)."""

    engine = AdvancedArbitrageEngine()
    data_path = args.replay_path or getattr(engine.config, "REPLAY_DATA_PATH", None)

    if not data_path:
        logger.error(
            "Для режима replay требуется указать путь к файлу через --replay-path или переменную REPLAY_DATA_PATH"
        )
        return

    replayer = HistoricalReplayer(
        engine,
        data_path,
        speed=args.replay_speed or getattr(engine.config, "REPLAY_SPEED", 1.0),
        max_records=args.replay_limit or getattr(engine.config, "REPLAY_MAX_RECORDS", None),
    )
    replayer.replay()


MODE_DEFINITIONS: Dict[str, ModeDefinition] = {
    "standard": ModeDefinition(
        description="Стандартный запуск улучшенного арбитражного бота",
        env_builder=_standard_env,
        config_builder=None,
        runner=_run_standard_mode,
    ),
    "aggressive": ModeDefinition(
        description="Тестнет с агрессивными параметрами по умолчанию",
        env_builder=_aggressive_env,
        config_builder=None,
        runner=_run_standard_mode,
    ),
    "quick": ModeDefinition(
        description="Быстрый тестовый цикл с OptimizedConfig",
        env_builder=_quick_env,
        config_builder=lambda args, logger: _prepare_quick_config(
            args.min_profit, args.trade_amount, logger
        ),
        runner=_run_quick_mode,
    ),
    "replay": ModeDefinition(
        description="Воспроизведение исторических данных для стресс-теста",
        env_builder=_replay_env,
        config_builder=None,
        runner=_run_replay_mode,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер на основе описаний доступных режимов."""

    modes_help = "; ".join(
        f"{name}: {definition.description}" for name, definition in MODE_DEFINITIONS.items()
    )
    parser = argparse.ArgumentParser(
        description=(
            "Универсальная точка входа для стандартного запуска, агрессивного режима, "
            "быстрого тестового цикла или исторического replay."
        )
    )

    parser.add_argument(
        "--mode",
        choices=list(MODE_DEFINITIONS.keys()),
        default="standard",
        help=f"Выбор режима: {modes_help}",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=None,
        help="Порог прибыли для поиска треугольников (MIN_TRIANGULAR_PROFIT)",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=None,
        help="Сумма сделки для тестовых прогонов",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Уровень логирования (например, INFO или DEBUG)",
    )
    parser.add_argument(
        "--replay-path",
        default=None,
        help="Путь к CSV с историческими котировками для режима replay",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=None,
        help="Коэффициент ускорения воспроизведения исторических данных",
    )
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=None,
        help="Ограничение числа записей для быстрого стресс-теста",
    )

    return parser


def _execute_mode(args: argparse.Namespace) -> None:
    """Применяет настройки для выбранного режима и запускает соответствующий раннер."""

    mode_definition = MODE_DEFINITIONS[args.mode]
    env_overrides = mode_definition.env_builder(args)
    _apply_environment(env_overrides)

    environment = _resolve_environment()
    logger = _configure_logging(args.log_level, args.mode, environment)
    config = None

    if mode_definition.config_builder is not None:
        config = mode_definition.config_builder(args, logger)

    mode_definition.runner(args, logger, environment, config)


def main() -> None:
    """Точка входа для CLI, маршрутизирующая запуск по выбранному режиму."""

    parser = _build_parser()
    args = parser.parse_args()

    _execute_mode(args)


if __name__ == "__main__":
    main()
