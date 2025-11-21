"""Быстрый запуск арбитражного движка с оптимизированной конфигурацией."""

import logging
import os
import time

import config as config_module
from optimized_config import OptimizedConfig


# Настраиваем базовое логирование с русскими сообщениями
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _prepare_config() -> OptimizedConfig:
    """Создает конфигурацию для тестнета и обновляет используемый класс Config."""

    # Обеспечиваем работу в тестнете через переменную окружения
    os.environ["TESTNET"] = "true"

    # Подменяем базовый класс конфигурации на оптимизированную версию
    config_module.Config = OptimizedConfig

    optimized = OptimizedConfig()
    optimized.TESTNET = True

    logging.info("Используется OptimizedConfig в режиме тестнета")
    return optimized


def quick_test(engine) -> None:
    """Выполняет три цикла поиска возможностей с выводом результатов."""

    for cycle in range(1, 4):
        start_time = time.time()
        opportunities = engine.detect_opportunities()
        duration = time.time() - start_time

        if opportunities:
            logging.info(
                "Цикл %s: найдено возможностей %s", cycle, len(opportunities)
            )
            logging.info("Список возможностей: %s", opportunities)
        else:
            logging.info("Цикл %s: возможности не найдены", cycle)

        logging.info("Время цикла %s: %.2f секунд", cycle, duration)
        if duration > 30:
            logging.warning(
                "Цикл %s превысил 30 секунд и может требовать оптимизации", cycle
            )


def main() -> None:
    """Точка входа для быстрого запуска арбитражного движка."""

    optimized_config = _prepare_config()

    # Импортируем после подмены конфигурации, чтобы движок использовал OptimizedConfig
    from advanced_arbitrage_engine import AdvancedArbitrageEngine

    engine = AdvancedArbitrageEngine()
    engine.config = optimized_config

    logging.info("Запуск тестового цикла обнаружения возможностей")
    quick_test(engine)


if __name__ == "__main__":
    main()
