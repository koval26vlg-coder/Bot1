# Algorithmic Trading Bot (SMA + Triangular Arbitrage)

Продакшн-скелет для алгоритмической торговли:
- SMA-стратегия с фильтрами (RSI, MACD, Bollinger Bands, волатильность, deadband, double-confirm).
- Треугольный арбитраж BTC/ETH/USDT.
- Авто-переключение стратегий через `StrategyManager`.
- Централизованный `risk-gate` (дневной минус, серия лоссов, лимит сделок).
- Единый логгер, mock/testnet/prod режимы.

## Структура проекта

- `main_live.py` — точка входа, выбор режима, цикл тиков.
- `trading_bot.py` — SMA-бот.
- `fixed_arbitrage.py` — TriangularArbBot (BTCUSDT/ETHBTC/ETHUSDT).
- `strategies.py` — логика сигналов и индикаторов.
- `strategy_manager.py` — переключение между SMA и арбитражем в режимах `AUTO` / `SMA_ONLY` / `ARB_ONLY`.
- `risk.py` — ограничения по рискам.
- `binance_client.py` — обёртка над Binance (real / testnet / mock).
- `logger.py` — настройка логов (stdout + JSON).
- `analytics.py` — утилиты: сводка, мониторинг, backtest.
- `test_connection.py` — проверка доступа к Binance.
- `test_trading_bot.py` — базовые тесты SMA-бота.
- `config_live.py` / `config_real.py` — конфиги.
- `requirements.txt` — зависимости.

Загрузка конфига:
1. Пытаемся импортировать `config_real.py` (боевой режим).
2. Если нет — `config_live.py` (testnet / mock).
3. Если нет — встроенные безопасные дефолты.

## Установка

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows (PowerShell)
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
# при работе с тестами/аналитикой:
pip install -r requirements-dev.txt  # если используете отдельно

Обязательно выполните установку зависимостей перед запуском бота:

- `pip install -r requirements.txt`
- модуль `psutil==6.1.0` входит в список обязательных пакетов и отвечает за сбор системных метрик; без него мониторинг ресурсов и алерты не будут работать.

### Переменные окружения

Для удобной работы со значениями API-ключей используется пакет `python-dotenv`.
Создайте файл `.env` в корне проекта и пропишите в нём, например:

```
BYBIT_API_KEY="ваш_ключ"
BYBIT_API_SECRET="ваш_секрет"
TESTNET=True
```

При запуске `load_dotenv()` автоматически подтянет эти значения, поэтому важно
установить зависимости из `requirements.txt`.

### Единый CLI (`main.py`)

Для запуска бота теперь используется единый скрипт `main.py` с переключателем режимов и параметрами порогов:

```bash
# Стандартный режим
python main.py --mode standard

# Агрессивный режим с пониженным порогом прибыли и симуляцией
python main.py --mode aggressive --min-profit 0.01 --trade-amount 25

# Быстрый тестовый прогон (3 цикла поиска возможностей)
python main.py --mode quick --log-level DEBUG
```

Основные параметры CLI:
- `--mode` — `standard`, `aggressive` или `quick`.
- `--min-profit` — порог прибыли `MIN_TRIANGULAR_PROFIT` (по умолчанию 0.01 в агрессивном режиме).
- `--trade-amount` — сумма сделки для тестовых прогонов.
- `--log-level` — уровень логирования (например, `INFO` или `DEBUG`).

Агрессивный режим автоматически включает `TESTNET=true` и `SIMULATION_MODE=true`, а быстрый прогон использует `OptimizedConfig` и тестнет.
