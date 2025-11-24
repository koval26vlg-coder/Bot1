"""Единая точка входа с CLI для разных режимов запуска бота."""

import argparse
import logging
import os
import time

import bot_bundle
from bot_bundle import AdvancedArbitrageEngine, HistoricalReplayer, OptimizedConfig


# Функция advanced_main получает функцию main из advanced_bot внутри монолитного bot_bundle
advanced_main = bot_bundle.main


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _configure_logging(level: str) -> None:
    """Настраивает базовое логирование с указанным уровнем."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


def _apply_threshold_overrides(min_profit: float | None, trade_amount: float | None) -> None:
    """Передает значения порогов через переменные окружения, если они заданы."""

    if min_profit is not None:
        os.environ["MIN_TRIANGULAR_PROFIT"] = str(min_profit)
    if trade_amount is not None:
        os.environ["TRADE_AMOUNT"] = str(trade_amount)


def _prepare_aggressive_env(min_profit: float | None, trade_amount: float | None) -> None:
    """Включает тестнет/симуляцию и устанавливает агрессивные параметры по умолчанию."""

    os.environ["TESTNET"] = "true"
    os.environ["SIMULATION_MODE"] = "true"

    effective_min_profit = min_profit if min_profit is not None else 0.01
    os.environ["MIN_TRIANGULAR_PROFIT"] = str(effective_min_profit)

    effective_trade_amount = trade_amount if trade_amount is not None else os.environ.get("TRADE_AMOUNT", "25")
    os.environ.setdefault("TRADE_AMOUNT", str(effective_trade_amount))


def _prepare_quick_config(min_profit: float | None, trade_amount: float | None) -> OptimizedConfig:
    """Создает конфигурацию для быстрого тестового цикла."""

    os.environ["TESTNET"] = "true"
    optimized = OptimizedConfig()
    optimized.TESTNET = True

    if min_profit is not None:
        optimized._min_triangular_profit_override = min_profit
    if trade_amount is not None:
        optimized._TRADE_AMOUNT = trade_amount

    logging.info("Используется OptimizedConfig в режиме тестнета")

    if trade_amount is not None:
        logging.info(
            "Применён пользовательский объём сделки для режима quick: %s USDT",
            optimized.TRADE_AMOUNT,
        )
    else:
        logging.info(
            "Используется стандартный объём сделки для режима quick: %s USDT",
            optimized.TRADE_AMOUNT,
        )
    return optimized


def _quick_test(engine) -> None:
    """Выполняет три цикла поиска возможностей с выводом результатов."""

    for cycle in range(1, 4):
        start_time = time.time()
        opportunities = engine.detect_opportunities()
        duration = time.time() - start_time

        if opportunities:
            logging.info("Цикл %s: найдено возможностей %s", cycle, len(opportunities))
            logging.info("Список возможностей: %s", opportunities)
        else:
            logging.info("Цикл %s: возможности не найдены", cycle)

        logging.info("Время цикла %s: %.2f секунд", cycle, duration)
        if duration > 30:
            logging.warning(
                "Цикл %s превысил 30 секунд и может требовать оптимизации", cycle
            )


def _run_standard_mode(args) -> None:
    """Стандартный запуск улучшенного бота."""

    _configure_logging(args.log_level)
    _apply_threshold_overrides(args.min_profit, args.trade_amount)

    advanced_main()


def _run_aggressive_mode(args) -> None:
    """Запуск с агрессивными настройками и предустановленными переменными окружения."""

    _configure_logging(args.log_level)
    _prepare_aggressive_env(args.min_profit, args.trade_amount)

    advanced_main()


def _run_quick_mode(args) -> None:
    """Быстрый тестовый прогон арбитражного движка с оптимизированной конфигурацией."""

    _configure_logging(args.log_level)
    optimized_config = _prepare_quick_config(args.min_profit, args.trade_amount)

    engine = AdvancedArbitrageEngine(config=optimized_config)

    logging.info("Запуск тестового цикла обнаружения возможностей")
    _quick_test(engine)


def _run_replay_mode(args) -> None:
    """Стресс-тест движка на исторических данных (режим replay)."""

    _configure_logging(args.log_level)
    os.environ.setdefault("SIMULATION_MODE", "true")
    os.environ.setdefault("PAPER_TRADING_MODE", "true")

    engine = AdvancedArbitrageEngine()
    data_path = args.replay_path or getattr(engine.config, "REPLAY_DATA_PATH", None)

    if not data_path:
        logging.error(
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


def _build_parser() -> argparse.ArgumentParser:
    """Создает CLI-парсер для выбора режимов и порогов."""

    parser = argparse.ArgumentParser(
        description=(
            "Универсальная точка входа для стандартного запуска, агрессивного режима, "
            "быстрого тестового цикла или исторического replay."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "aggressive", "quick", "replay"],
        default="standard",
        help="Выбор режима: стандартный, агрессивный, быстрый тест или replay",
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


def main() -> None:
    """Точка входа для CLI, маршрутизирующая запуск по выбранному режиму."""

    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "aggressive":
        _run_aggressive_mode(args)
    elif args.mode == "quick":
        _run_quick_mode(args)
    elif args.mode == "replay":
        _run_replay_mode(args)
    else:
        _run_standard_mode(args)


if __name__ == "__main__":
    main()
