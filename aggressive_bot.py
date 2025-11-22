"""Запуск бота в агрессивном тестовом режиме с предустановленными переменными окружения."""

import os

# Предварительно задаем переменные окружения, чтобы они подхватились при импорте движка
os.environ["TESTNET"] = "true"
os.environ["SIMULATION_MODE"] = "true"
os.environ["MIN_TRIANGULAR_PROFIT"] = "0.01"
# Рекомендуемая сумма сделки для агрессивного режима (можно скорректировать при запуске)
os.environ.setdefault("TRADE_AMOUNT", "25")

# Импорт основной точки входа после установки переменных окружения
from advanced_bot import main as run_advanced_bot


def main():
    """Запуск расширенного бота с агрессивными настройками."""
    run_advanced_bot()


if __name__ == "__main__":
    main()
