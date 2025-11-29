# Algorithmic Trading Bot (Bybit Triangular Arbitrage)

Текущая версия проекта намеренно упрощена и сведена к двум основным исходникам, чтобы облегчить тестирование и демонстрацию логики. Вся функциональность (конфигурация, логирование, движок и воспроизведение истории) собрана в монолите `arbitrage_bot.py`, а запуск управляется через единый CLI `main.py`.

## Структура проекта

- `main.py` — CLI с режимами `standard`, `aggressive`, `quick` и `replay`.
- `arbitrage_bot.py` — монолитный модуль с классами `OptimizedConfig`, `AdvancedArbitrageEngine`, `HistoricalReplayer` и утилитами логирования.
- `requirements.txt` — зависимости для запуска.

## Режимы запуска через CLI

`main.py` предоставляет четыре режима работы, переключаемые через параметр `--mode`:

- `standard` — базовый запуск упрощённого движка.
- `aggressive` — включает тестнет и симуляцию, по умолчанию снижает `MIN_TRIANGULAR_PROFIT` до `0.01` и позволяет переопределять пороги через параметры CLI.
- `quick` — быстрый тестовый прогон с `OptimizedConfig` и тремя циклами поиска возможностей без выхода в продакшн; пользовательские `--min-profit` и `--trade-amount` сразу применяются к рабочей конфигурации.
- `replay` — воспроизведение исторических данных (CSV) с ограничением по скорости и количеству строк.

Примеры:

```bash
# Стандартный режим
python main.py --mode standard

# Агрессивный режим с кастомным порогом прибыли и суммой сделки
python main.py --mode aggressive --min-profit 0.01 --trade-amount 25

# Быстрый тестовый прогон с повышенным логированием
python main.py --mode quick --log-level DEBUG

# Replay с историческими данными (ускорение в 4 раза)
python main.py --mode replay --replay-path data/bybit_ticks.csv --replay-speed 4
```

Основные параметры:

- `--mode` — `standard`, `aggressive`, `quick` или `replay`.
- `--min-profit` — порог прибыли `MIN_TRIANGULAR_PROFIT`.
- `--trade-amount` — сумма сделки для тестовых прогонов.
- `--log-level` — уровень логирования.
- `--replay-path` — путь к CSV с полями `timestamp,symbol,bid,ask,bid_size,ask_size,last_price,volume`.
- `--replay-speed` — ускорение воспроизведения исторических данных.
- `--replay-limit` — ограничение числа строк для быстрого стресс-теста.

## Установка

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows (PowerShell)
# source .venv/bin/activate  # Linux/macOS

pip install .
```

Обязательно выполните установку зависимостей из `requirements.txt`. При ограниченном доступе к интернету можно заранее скачать колёса и установить через `pip install --no-index --find-links=wheelhouse -r requirements.txt`.

### Переменные окружения

Базовые параметры конфигурации можно задавать через переменные окружения:

- `TESTNET` — включает тестовую среду.
- `MIN_TRIANGULAR_PROFIT` — минимальный порог прибыли для поиска треугольников.
- `TRADE_AMOUNT` — объём сделки для тестовых прогонов.
- `REPLAY_DATA_PATH`, `REPLAY_SPEED`, `REPLAY_MAX_RECORDS` — настройки режима `replay`.

Пример `.env`:

```bash
TESTNET=true
MIN_TRIANGULAR_PROFIT=0.03
TRADE_AMOUNT=50
```

После установки запустите стандартный режим, чтобы убедиться, что зависимости корректно подтянулись:

```bash
python main.py --mode standard
```
