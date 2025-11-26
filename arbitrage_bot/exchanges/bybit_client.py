import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode

import aiohttp
import requests
from arbitrage_bot.core.config import Config

try:
    from pybit.unified_trading import HTTP, WebSocket
except ModuleNotFoundError:
    HTTP = None
    WebSocket = None

logger = logging.getLogger(__name__)


class BybitWebSocketManager:
    """Управление WebSocket-подключениями к Bybit для котировок и ордеров."""

    def __init__(self, config: Config, *, order_callback=None):
        self.config = config
        self._order_callback = order_callback
        self._ticker_cache = {}
        self._cache_lock = threading.Lock()
        self._public_ws = None
        self._private_ws = None
        self._symbols = set()
        self._order_listeners = []
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._last_ticker_ts = 0
        self._max_staleness = max(
            getattr(self.config, 'TICKER_STALENESS_WARNING_SEC', 5.0) * 2,
            1.0,
        )
        # Сегмент рынка заранее синхронизируем с REST-клиентом, чтобы избежать ошибок доступа
        self.market_category = getattr(self.config, "MARKET_CATEGORY", "spot")

    def start(self, symbols):
        """Запуск стримов по списку тикеров."""

        self._symbols = set(symbols)
        self._connect_public_ws()
        self._ensure_monitor()

    def stop(self):
        """Остановка всех подключений (используется при завершении работы)."""

        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        self._shutdown_ws(self._public_ws)
        self._shutdown_ws(self._private_ws)

        self._public_ws = None
        self._private_ws = None

    def register_order_listener(self, callback):
        """Регистрирует обработчик событий ордеров и инициирует приватный стрим."""

        if not callback:
            return

        if callback not in self._order_listeners:
            self._order_listeners.append(callback)

        self._connect_private_ws()
        self._ensure_monitor()

    def get_cached_tickers(self, symbols, max_age=None):
        """Возвращает свежие котировки из кэша и список недостающих тикеров."""

        max_age = max_age or self._max_staleness
        now = time.time()
        fresh = {}
        missing = []

        with self._cache_lock:
            for symbol in symbols:
                cached = self._ticker_cache.get(symbol)
                if cached and now - cached['ts'] <= max_age:
                    fresh[symbol] = cached['data']
                else:
                    missing.append(symbol)

        return fresh, missing

    def update_cache(self, tickers):
        """Принудительное обновление кэша внешними данными (например, после REST-запроса)."""

        now = time.time()
        with self._cache_lock:
            for symbol, data in tickers.items():
                self._ticker_cache[symbol] = {'data': data, 'ts': now}
                self._last_ticker_ts = now

    def _ensure_monitor(self):
        """Запускает фоновый мониторинг состояния подключений."""

        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
        self._monitor_thread.start()

    def _monitor_connections(self):
        """Следит за обрывами соединений и восстанавливает стримы."""

        while not self._stop_event.is_set():
            try:
                now = time.time()
                public_alive = self._is_ws_active(self._public_ws)
                if self._symbols and (self._public_ws is None or not public_alive or now - self._last_ticker_ts > self._max_staleness):
                    if self._public_ws is None or not public_alive:
                        logger.debug("Обнаружено закрытое публичное подключение, инициируем переподключение")
                        self._shutdown_ws(self._public_ws)
                        self._public_ws = None
                    logger.debug("Переподключение публичного стрима котировок")
                    self._restart_public_ws()

                private_alive = self._is_ws_active(self._private_ws)
                if self._order_listeners and (self._private_ws is None or not private_alive):
                    if self._private_ws is None or not private_alive:
                        logger.debug("Обнаружено закрытое приватное подключение, инициируем переподключение")
                        self._shutdown_ws(self._private_ws)
                        self._private_ws = None
                    logger.debug("Переподключение приватного стрима ордеров")
                    self._connect_private_ws()
            except Exception as exc:
                logger.warning("Ошибка мониторинга WebSocket: %s", exc)

            time.sleep(3)

    def _is_ws_active(self, ws):
        """Проверяет наличие живого соединения WebSocket."""

        if not ws:
            return False

        inner_ws = getattr(ws, "ws", None)
        if inner_ws is None:
            return False

        connected_flag = getattr(inner_ws, "connected", None)
        if isinstance(connected_flag, bool):
            return connected_flag

        sock = getattr(inner_ws, "sock", None)
        if sock is None:
            return False

        if hasattr(sock, "connected"):
            return bool(sock.connected)

        return True

    def _connect_public_ws(self):
        """Создаёт подключение для получения котировок."""

        if WebSocket is None:
            logger.warning("pybit не установлен, WebSocket котировок недоступен")
            return

        symbols = list(self._symbols)
        if not symbols:
            logger.info("Нет символов для подписки на публичный стрим котировок")
            return

        batch_size = 10
        symbol_batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

        try:
            self._public_ws = WebSocket(
                channel_type=self.market_category,
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
            )
            logger.info(
                "Подготовлено %s пакетов подписки на котировки для %s символов",
                len(symbol_batches),
                len(symbols),
            )

            for idx, batch in enumerate(symbol_batches, start=1):
                logger.debug("Подписка на пакет %s/%s: %s", idx, len(symbol_batches), batch)
                self._public_ws.ticker_stream(symbol=batch, callback=self._handle_ticker)
                self._last_ticker_ts = time.time()

            logger.info("📡 WebSocket котировок запущен для %s символов", len(symbols))
        except Exception as exc:
            logger.warning("Не удалось подключиться к публичному стриму котировок: %s", exc)
            self._public_ws = None

    def _restart_public_ws(self):
        """Перезапускает публичное подключение."""

        try:
            self._shutdown_ws(self._public_ws)
            self._public_ws = None
        finally:
            self._connect_public_ws()

    def _connect_private_ws(self):
        """Создаёт приватное подключение для событий ордеров."""

        if WebSocket is None:
            logger.warning("pybit не установлен, приватный WebSocket недоступен")
            return

        if not self.config.API_KEY or not self.config.API_SECRET:
            logger.warning("API ключи не заданы, пропускаем подписку на приватные события")
            return

        try:
            self._private_ws = WebSocket(
                channel_type="private",
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
            )
            self._private_ws.order_stream(callback=self._handle_order)
            logger.info("🔔 WebSocket ордеров активирован")
        except Exception as exc:
            logger.warning("Не удалось подключиться к приватному стриму ордеров: %s", exc)
            self._private_ws = None

    def _shutdown_ws(self, ws):
        """Аккуратное завершение WebSocket с остановкой служебных потоков."""

        if not ws:
            return

        try:
            ping_timer = getattr(ws, "custom_ping_timer", None)
            if ping_timer:
                stop_method = getattr(ping_timer, "cancel", None) or getattr(ping_timer, "stop", None)
                if callable(stop_method):
                    stop_method()
                if hasattr(ping_timer, "is_alive") and ping_timer.is_alive():
                    ping_timer.join(timeout=1)

            if hasattr(ws, "custom_ping_running"):
                try:
                    ws.custom_ping_running = False
                except Exception:
                    logger.debug("Не удалось обновить флаг custom_ping_running", exc_info=True)

            ping_thread = getattr(ws, "ping_thread", None)
            if ping_thread and hasattr(ping_thread, "join"):
                ping_thread.join(timeout=2)

            if hasattr(ws, "exit"):
                ws.exit()
            elif hasattr(ws, "close"):
                ws.close()
        except Exception:
            logger.debug("Ошибка при завершении WebSocket", exc_info=True)

    def _handle_ticker(self, message):
        """Нормализация входящих котировок и сохранение в кэше."""

        data = message.get('data') if isinstance(message, dict) else None
        if not data:
            return

        if isinstance(data, dict):
            entries = [data]
        else:
            entries = data

        now = time.time()

        with self._cache_lock:
            for entry in entries:
                symbol = entry.get('symbol') or entry.get('s')
                if not symbol:
                    continue

                bid = self._safe_float(
                    entry.get('bid1Price')
                    or entry.get('bestBidPrice')
                    or entry.get('bp')
                    or entry.get('b1'),
                    0,
                )
                ask = self._safe_float(
                    entry.get('ask1Price')
                    or entry.get('bestAskPrice')
                    or entry.get('ap')
                    or entry.get('a1'),
                    0,
                )

                ticker = {
                    'bid': bid,
                    'ask': ask,
                    'last_price': self._safe_float(entry.get('lastPrice') or entry.get('lp') or entry.get('price'), 0),
                    'bid_size': self._safe_float(entry.get('bid1Size') or entry.get('b1Size') or entry.get('bidSize') or entry.get('bq')),  
                    'ask_size': self._safe_float(entry.get('ask1Size') or entry.get('a1Size') or entry.get('askSize') or entry.get('aq')),
                }

                self._ticker_cache[symbol] = {'data': ticker, 'ts': now}
                self._last_ticker_ts = now

        if self._order_callback:
            # Хук на внешний обработчик может использовать котировки для актуализации внутренних структур
            try:
                self._order_callback({'type': 'ticker', 'symbols': [e.get('symbol') for e in entries if e.get('symbol')]})
            except Exception:
                logger.debug("Ошибка колбэка на котировки", exc_info=True)

    def _handle_order(self, message):
        """Пробрасывает события ордеров зарегистрированным слушателям."""

        data = message.get('data') if isinstance(message, dict) else None
        if not data:
            return

        events = data if isinstance(data, list) else [data]

        for event in events:
            normalized = {
                'orderId': event.get('orderId'),
                'symbol': event.get('symbol'),
                'orderStatus': event.get('orderStatus'),
                'side': event.get('side'),
                'leavesQty': self._safe_float(event.get('leavesQty')),
                'cumExecQty': self._safe_float(event.get('cumExecQty')),
                'avgPrice': self._safe_float(event.get('avgPrice') or event.get('lastPrice')),
                'execType': event.get('execType') or event.get('eventType'),
                'updatedTime': event.get('updatedTime') or event.get('ts') or int(time.time() * 1000),
            }

            for listener in self._order_listeners:
                try:
                    listener(normalized)
                except Exception:
                    logger.debug("Ошибка в обработчике событий ордеров", exc_info=True)

    def _safe_float(self, value, default=0.0):
        """Безопасное приведение к float для всех входящих данных."""

        try:
            if value is None:
                return default

            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default

            return float(value)
        except (TypeError, ValueError):
            return default

class BybitClient:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.session = self._create_session()
        self.account_type = "UNIFIED" if not self.config.TESTNET else "CONTRACT"
        # Всегда заранее сохраняем сегмент рынка, чтобы одинаково использовать его во всех запросах
        self.market_category = getattr(self.config, "MARKET_CATEGORY", "spot")
        self.ws_manager = None
        self.order_error_metrics = defaultdict(int)
        self._temporarily_unavailable_symbols = set()
        self._async_http_session: aiohttp.ClientSession | None = None
        self._initialize_websocket_streams()
        logger.info(
            "Bybit client initialized. Testnet: %s, Account type: %s, Market category: %s",
            self.config.TESTNET,
            self.account_type,
            self.market_category,
        )

    def _classify_error(self, *, response=None, exception=None):
        """Определяет тип ошибки для логирования и метрик."""

        if exception is not None:
            message = str(exception)

            if isinstance(exception, (TimeoutError, )):
                return "network", message

            if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                return "network", message

            return "unknown", message

        if response is None:
            return "unknown", "Пустой ответ от API"

        message = response.get('retMsg', '') or ''
        normalized = message.lower()
        ret_code = response.get('retCode')

        validation_keywords = (
            'invalid', 'parameter', 'qty', 'quantity', 'insufficient', 'leverage', 'precision', 'required'
        )
        refusal_keywords = (
            'reject', 'rejected', 'blocked', 'limit', 'risk', 'system busy', 'maintenance', 'forbidden', 'denied'
        )

        if ret_code and str(ret_code).startswith('100'):
            return "validation", message or f"Код ошибки {ret_code}"

        if any(keyword in normalized for keyword in validation_keywords):
            return "validation", message or "Ошибка валидации параметров"

        if any(keyword in normalized for keyword in refusal_keywords):
            return "exchange_refusal", message or "Биржа отвергла запрос"

        return "unknown", message or f"Код ошибки {ret_code}"

    def _record_error_metric(self, error_type):
        """Увеличивает счётчик ошибок указанного типа."""

        self.order_error_metrics[error_type] += 1

    async def _ensure_async_session(self) -> aiohttp.ClientSession:
        """Создаёт HTTP-сессию aiohttp для приватных запросов."""

        if self._async_http_session is None or self._async_http_session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._async_http_session = aiohttp.ClientSession(timeout=timeout)
        return self._async_http_session

    def _build_signed_headers(self, params: dict[str, str], body: str = "") -> dict[str, str]:
        """Формирует подпись для приватных запросов Bybit v5."""

        api_key = getattr(self.config, "API_KEY", None)
        api_secret = getattr(self.config, "API_SECRET", None)
        if not api_key or not api_secret:
            logger.warning("Отсутствуют API ключи для подписанного запроса")
            return {}

        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = urlencode(sorted(params.items()))
        sign_payload = f"{timestamp}{api_key}{recv_window}{query}{body}".encode()
        signature = hmac.new(api_secret.encode(), sign_payload, hashlib.sha256).hexdigest()

        return {
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-SIGN": signature,
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

    def _log_attempt_result(self, operation, attempt, max_attempts, success, error_type, message):
        """Единообразное логирование попыток с типами ошибок."""

        status_label = "успех" if success else "ошибка"
        logger_method = logger.info if success else logger.warning
        logger_method(
            "%s: попытка %s/%s завершилась как %s (%s). %s",
            operation,
            attempt,
            max_attempts,
            status_label,
            error_type,
            message,
        )

    def _is_status_uncertain(self, status: str | None) -> bool:
        """Определяет, требуется ли доуточнение статуса ордера."""

        if not status:
            return True

        uncertain_statuses = {
            'Created', 'New', 'Untriggered', 'PartiallyFilled', 'Pending', 'Triggered'
        }
        return status in uncertain_statuses

    async def _ensure_order_finalized(self, order_id, symbol, initial_status, fallback_payload=None):
        """Асинхронно уточняет или отменяет ордер с ограничением попыток."""

        if not order_id:
            logger.warning("Не удалось уточнить статус: отсутствует orderId для %s", symbol)
            return fallback_payload

        logger.warning(
            "Статус ордера %s для %s неоднозначен (%s). Запускаем уточнение/отмену.",
            order_id,
            symbol,
            initial_status or 'unknown',
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            status_task = self.get_order_status_async(order_id, symbol)
            cancel_task = self.cancel_order_async(order_id, symbol)
            fetched, cancel_result = await asyncio.gather(status_task, cancel_task)

            if fetched and not self._is_status_uncertain(fetched.get('orderStatus')):
                return fetched

            if cancel_result:
                fetched_after_cancel = await self.get_order_status_async(order_id, symbol)
                if fetched_after_cancel:
                    return fetched_after_cancel

            if attempt < max_attempts:
                await asyncio.sleep(0.5 * attempt)

        return fallback_payload

    def _ensure_order_finalized_sync(self, order_id, symbol, initial_status, fallback_payload=None):
        """Синхронный адаптер для асинхронного уточнения статуса ордера."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._ensure_order_finalized(order_id, symbol, initial_status, fallback_payload=fallback_payload)
            )

        future = asyncio.run_coroutine_threadsafe(
            self._ensure_order_finalized(order_id, symbol, initial_status, fallback_payload=fallback_payload),
            loop,
        )
        return future.result()

    def _initialize_websocket_streams(self):
        """Настраивает WebSocket для котировок и приватных событий."""

        if WebSocket is None:
            logger.warning("pybit не установлен, пропускаем запуск WebSocket")
            return

        try:
            self.ws_manager = BybitWebSocketManager(self.config)
            self.ws_manager.start(self.config.SYMBOLS)
        except Exception as exc:
            logger.warning("Не удалось инициализировать WebSocket-стримы: %s", exc, exc_info=True)
            self.ws_manager = None
    
    def _create_session(self):
        """Создание сессии для работы с Bybit API"""
        if HTTP is None:
            raise RuntimeError(
                "pybit is not installed. Install dependencies with 'pip install -r requirements.txt'"
            )

        try:
            return HTTP(
                testnet=self.config.TESTNET,
                api_key=self.config.API_KEY,
                api_secret=self.config.API_SECRET,
                recv_window=10000  # Увеличенное время ожидания для тестнета
            )
        except Exception as e:
            logger.error(f"❌ Failed to create Bybit session: {str(e)}")
            raise
    
    def get_tickers(self, symbols):
        """Получение котировок пачками с минимальным числом HTTP-запросов"""
        requested_symbols = sorted(set(symbols))
        tickers = {}

        if not requested_symbols:
            return tickers

        blocked_symbols = set(requested_symbols) & self._temporarily_unavailable_symbols
        remaining_symbols = set(requested_symbols)

        if blocked_symbols:
            logger.warning(
                "⏳ Пропускаем REST-запросы для временно недоступных тикеров: %s",
                ", ".join(sorted(blocked_symbols)),
            )

        cache_hits = {}

        if self.ws_manager:
            cache_hits, fresh_missing = self.ws_manager.get_cached_tickers(
                requested_symbols,
                max_age=getattr(self.config, 'TICKER_STALENESS_WARNING_SEC', 5.0),
            )
            tickers.update(cache_hits)
            remaining_symbols = set(fresh_missing)

            recovered = set(cache_hits) & self._temporarily_unavailable_symbols
            if recovered:
                logger.info(
                    "✅ Тикеры снова появились и удалены из карантина: %s",
                    ", ".join(sorted(recovered)),
                )
                self._temporarily_unavailable_symbols.difference_update(recovered)

            if not remaining_symbols:
                logger.debug("♻️ Используем кэш WebSocket для всех тикеров")
                self._validate_ticker_freshness(tickers)
                return tickers

        if getattr(self.config, 'WEBSOCKET_PRICE_ONLY', False) and remaining_symbols:
            logger.warning(
                "📡 Включён режим WEBSOCKET_PRICE_ONLY, пропускаем REST для %s тикеров", len(remaining_symbols)
            )
            self._validate_ticker_freshness(tickers)
            return tickers

        logger.debug(f"🔍 Requesting {len(requested_symbols)} symbols: {requested_symbols}")

        start_time = time.time()
        request_count = 0

        max_retries = getattr(self.config, 'TICKER_MAX_RETRIES', 3)
        base_backoff = getattr(self.config, 'TICKER_BACKOFF_BASE', 0.25)
        heavy_backoff = getattr(self.config, 'TICKER_HEAVY_BACKOFF_BASE', 1.0)
        pause_required = False

        def _calculate_delay(attempt, is_heavy):
            """Подбирает задержку с экспоненциальным ростом"""
            base = heavy_backoff if is_heavy else base_backoff
            return base * (2 ** (attempt - 1))

        def _request_with_retries(request_fn, label):
            """Оборачивает запрос в цикл повторов с экспоненциальным бэкоффом"""
            nonlocal pause_required
            last_exc = None

            for attempt in range(1, max_retries + 1):
                try:
                    return request_fn()
                except Exception as exc:
                    status_code = None
                    if hasattr(exc, 'response') and getattr(exc, 'response') is not None:
                        status_code = getattr(exc.response, 'status_code', None)

                    is_rate_limited = status_code == 429
                    is_server_error = isinstance(status_code, int) and status_code >= 500
                    is_heavy = is_rate_limited or is_server_error
                    delay = _calculate_delay(attempt, is_heavy)

                    logger.warning(
                        "♻️ Ошибка запроса %s (попытка %s/%s, код: %s). Ждём %.2f c перед повтором",
                        label,
                        attempt,
                        max_retries,
                        status_code or 'n/a',
                        delay,
                    )
                    time.sleep(delay)
                    last_exc = exc

                    if attempt == max_retries and is_heavy:
                        pause_required = True

            if pause_required:
                raise RuntimeError(
                    "Достигнут предел повторов для запросов тикеров, требуется временная пауза",
                ) from last_exc

            raise last_exc or RuntimeError("Неизвестная ошибка при запросе тикеров")

        def _extract_from_response(response, label):
            """Извлекает данные тикеров из ответа и обновляет остаток"""
            nonlocal tickers
            if not response:
                logger.debug(f"❌ Пустой ответ в блоке {label}")
                return

            if response.get('retCode') != 0 or not response.get('result'):
                logger.debug(f"❌ API error in {label}: {response.get('retMsg')}")
                return

            ticker_list = response.get('result', {}).get('list')
            if ticker_list is None:
                logger.debug(f"ℹ️ Пустой список тикеров в блоке {label}")
                return

            if not isinstance(ticker_list, Iterable) or isinstance(ticker_list, (str, bytes, dict)):
                logger.debug(
                    f"❌ Некорректный формат списка тикеров в блоке {label}: {type(ticker_list)}"
                )
                return

            for ticker_data in ticker_list:
                symbol = ticker_data.get('symbol')
                if symbol not in remaining_symbols:
                    continue

                tickers[symbol] = {
                    'bid': self._safe_float(ticker_data.get('bid1Price', 0)),
                    'ask': self._safe_float(ticker_data.get('ask1Price', 0)),
                    'last': self._safe_float(ticker_data.get('lastPrice', 0)),
                    'timestamp': ticker_data.get('time')
                }
                remaining_symbols.discard(symbol)
                self._temporarily_unavailable_symbols.discard(symbol)
                logger.debug(
                    f"✅ {symbol}: bid={tickers[symbol]['bid']}, ask={tickers[symbol]['ask']} (source={label})"
                )

        # Основной bulk-запрос без параметра symbol
        try:
            cursor = None
            while True:
                fetchable_symbols = remaining_symbols - blocked_symbols
                if not fetchable_symbols:
                    break

                params = {'category': self.market_category}
                if cursor:
                    params['cursor'] = cursor

                response = _request_with_retries(
                    lambda: self.session.get_tickers(**params),
                    'bulk',
                )
                request_count += 1
                _extract_from_response(response, 'bulk')

                cursor = response.get('result', {}).get('nextPageCursor') if response else None
                if not cursor or not (remaining_symbols - blocked_symbols):
                    break

        except RuntimeError:
            raise
        except Exception as e:
            logger.debug(f"🔥 Bulk request failed: {str(e)}")

        # Фолбэк: запрашиваем оставшиеся символы параллельно
        if remaining_symbols - blocked_symbols:
            logger.debug(
                f"⚙️ Bulk вернул не все данные, догружаем {len(remaining_symbols)} символов параллельно"
            )

            def _fetch_symbol(symbol):
                try:
                    return _request_with_retries(
                        lambda: self.session.get_tickers(
                            category=self.market_category,
                            symbol=symbol,
                        ),
                        f'fallback:{symbol}',
                    )
                except Exception as exc:
                    logger.debug(f"🔥 Exception for {symbol}: {str(exc)}")
                    raise

            fetchable = list(remaining_symbols - blocked_symbols)
            with ThreadPoolExecutor(max_workers=min(8, len(fetchable))) as executor:
                future_to_symbol = {
                    executor.submit(_fetch_symbol, symbol): symbol for symbol in fetchable
                }

                for future in as_completed(future_to_symbol):
                    request_count += 1
                    symbol = future_to_symbol[future]
                    try:
                        response = future.result()
                    except Exception as exc:
                        logger.debug(f"🔥 Exception for {symbol}: {str(exc)}")
                        if pause_required:
                            raise RuntimeError(
                                "Достигнут предел повторов для фолбэк-запросов, требуется пауза",
                            ) from exc
                        continue

                    _extract_from_response(response, f'fallback:{symbol}')

        if remaining_symbols:
            missing_preview = ', '.join(sorted(remaining_symbols))
            logger.warning(
                "🚫 После всех запросов отсутствуют тикеры: %s. Помечаем как временно недоступные.",
                missing_preview,
            )
            self._temporarily_unavailable_symbols.update(remaining_symbols)
            remaining_symbols.clear()

        duration = time.time() - start_time
        logger.debug(
            f"📊 Total tickers received: {len(tickers)} (requests: {request_count}, missing: {len(remaining_symbols)})"
        )

        if duration < 2:
            logger.info(
                f"⚡️ Сбор {len(tickers)} тикеров занял {duration:.2f} с (меньше 2 секунд, запросов: {request_count})"
            )
        else:
            logger.warning(
                f"⏱️ Сбор {len(tickers)} тикеров занял {duration:.2f} с (запросов: {request_count})"
            )

        if self.ws_manager and tickers:
            self.ws_manager.update_cache(tickers)

        self._validate_ticker_freshness(tickers)

        return tickers

    def get_unavailable_symbols(self):
        """Возвращает текущее множество временно исключённых тикеров."""

        return set(self._temporarily_unavailable_symbols)

    def add_order_listener(self, callback):
        """Подключает внешний обработчик событий ордеров."""

        if not self.ws_manager:
            logger.warning("WebSocket менеджер не инициализирован, события ордеров недоступны")
            return

        self.ws_manager.register_order_listener(callback)

    def _validate_ticker_freshness(self, tickers):
        """Проверяет насколько свежие котировки получены от Bybit"""
        if not tickers:
            return

        freshness_limit_ms = int(self.config.TICKER_STALENESS_WARNING_SEC * 1000)
        now_ms = int(time.time() * 1000)
        stale = []

        for symbol, data in tickers.items():
            timestamp = data.get('timestamp')
            if not timestamp:
                continue

            try:
                age_ms = now_ms - int(float(timestamp))
            except (TypeError, ValueError):
                continue

            if age_ms > freshness_limit_ms:
                stale.append((symbol, age_ms / 1000))

        if stale:
            preview = ', '.join(f"{sym} ({age:.1f}с)" for sym, age in stale[:5])
            logger.warning(
                "🕒 Обнаружены устаревшие котировки (>%.1fс): %s",
                self.config.TICKER_STALENESS_WARNING_SEC,
                preview
            )
        else:
            logger.debug(
                "Котировки %s инструментов свежее %.1f секунд",
                len(tickers),
                self.config.TICKER_STALENESS_WARNING_SEC
            )

    def get_order_book(self, symbol, depth=5):
        """Возвращает стакан для инструмента с указанной глубиной"""
        try:
            response = self.session.get_orderbook(
                category=self.market_category,
                symbol=symbol,
                limit=depth
            )

            if response.get('retCode') != 0 or 'result' not in response:
                logger.debug(
                    "Не удалось получить стакан для %s: %s",
                    symbol,
                    response.get('retMsg') if isinstance(response, dict) else 'unknown error'
                )
                return {'bids': [], 'asks': []}

            orderbook = response['result']
            bids = orderbook.get('b', [])
            asks = orderbook.get('a', [])

            # Форматируем в удобный вид: список словарей с price/size
            def _normalize(side):
                normalized = []
                for level in side:
                    if len(level) < 2:
                        continue
                    price = self._safe_float(level[0])
                    size = self._safe_float(level[1])
                    normalized.append({'price': price, 'size': size})
                return normalized

            return {
                'bids': _normalize(bids),
                'asks': _normalize(asks)
            }
        except Exception as exc:
            logger.debug("Ошибка при получении стакана %s: %s", symbol, str(exc))
            return {'bids': [], 'asks': []}
        
    def get_balance(self, coin='USDT'):
        """Получение баланса для конкретной монеты - исправленная версия для тестнета"""
        try:
            # Всегда используем фиктивный баланс для тестнета для избежания ошибок
            if self.config.TESTNET:
                logger.info("🧪 Using mock balance for testnet")
                return {'available': 100.0, 'total': 100.0, 'coin': coin}
        
            # Реальная логика для основной сети
            response = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin=coin
            )
        
            if response.get('retCode') == 0 and response.get('result'):
                balance_list = response['result'].get('list', [])
                if balance_list:
                    for account in balance_list:
                        coin_balances = account.get('coin', [])
                        for coin_balance in coin_balances:
                            if coin_balance.get('coin') == coin:
                                available = self._safe_float(coin_balance.get('availableToWithdraw', 0))
                                total = self._safe_float(coin_balance.get('walletBalance', 0))
                                return {
                                    'available': available,
                                    'total': total,
                                    'coin': coin
                                }
            logger.warning(f"No balance data for {coin}: {response.get('retMsg', 'Unknown error')}")
            return {'available': 0.0, 'total': 0.0, 'coin': coin}
        except Exception as e:
            logger.error(f"Error getting balance for {coin}: {str(e)}")
            return {'available': 0.0, 'total': 0.0, 'coin': coin}

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование к float, чтобы пустые строки не ломали расчеты."""
        try:
            if value is None:
                return default

            # Пустые строки и значения с пробелами должны превращаться в дефолт сразу
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default

            return float(value)
        except (TypeError, ValueError):
            return default
    
    def place_order(
        self,
        symbol,
        side,
        qty,
        price=None,
        order_type='Market',
        trigger_price=None,
        trigger_by='LastPrice',
        reduce_only=False,
    ):
        """Размещение ордера на бирже с улучшенной обработкой ошибок и поддержкой контингентных триггеров"""
        try:
            # Проверка минимальных объемов для тестнета
            if self.config.TESTNET:
                if qty < 0.001 and symbol in ['BTCUSDT', 'ETHUSDT']:
                    logger.warning(f"🧪 Testnet: Increasing quantity for {symbol} from {qty} to 0.001")
                    qty = 0.001

            params = {
                'category': self.market_category,
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'timeInForce': 'GTC' if order_type == 'Limit' else 'IOC',
                'isLeverage': 0,
                'orderFilter': 'Order',
                'reduceOnly': 1 if reduce_only else 0,
            }

            if price and order_type == 'Limit':
                params['price'] = str(price)

            if trigger_price is not None:
                params['triggerPrice'] = str(trigger_price)
                params['triggerBy'] = trigger_by
                params['orderFilter'] = 'tpslOrder' if order_type.lower() != 'market' else 'Order'

            logger.info(f"🚀 Placing {order_type} order: {params}")

            # В тестнете не выполняем реальные ордера, только имитируем
            if self.config.TESTNET:
                logger.info(f"🧪 TESTNET MODE: Simulating order execution (no real order placed)")
                return {
                    'orderId': f"test_order_{int(time.time())}",
                    'orderStatus': 'Filled',
                    'price': str(price) if price else 'market',
                    'avgPrice': str(price) if price else 'market',
                    'qty': str(qty),
                    'cumExecQty': str(qty),
                    'symbol': symbol
                }

            max_attempts = 3
            base_delay = 0.5
            last_result = None

            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.place_order(**params)
                except Exception as exc:
                    error_type, error_message = self._classify_error(exception=exc)
                    self._record_error_metric(error_type)
                    self._log_attempt_result(
                        "place_order",
                        attempt,
                        max_attempts,
                        False,
                        error_type,
                        f"Исключение при размещении: {error_message}",
                    )

                    if attempt < max_attempts:
                        time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue

                if response and response.get('retCode') == 0 and response.get('result'):
                    result = response['result']
                    order_id = result.get('orderId')
                    status = result.get('orderStatus')

                    self._log_attempt_result(
                        "place_order",
                        attempt,
                        max_attempts,
                        True,
                        "ok",
                        f"Статус {status}, orderId={order_id}",
                    )

                    if self._is_status_uncertain(status):
                        return self._ensure_order_finalized_sync(order_id, symbol, status, fallback_payload=result)

                    return result

                error_type, error_message = self._classify_error(response=response)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "place_order",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Код {response.get('retCode') if response else 'N/A'}",
                )

                last_result = response

                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))

            if last_result and last_result.get('result', {}).get('orderId'):
                uncertain = last_result['result']
                return self._ensure_order_finalized_sync(
                    uncertain.get('orderId'),
                    symbol,
                    uncertain.get('orderStatus'),
                    fallback_payload=uncertain,
                )

            return None

        except Exception as e:
            logger.error(f"🔥 Critical error placing order: {str(e)}", exc_info=True)
            self._record_error_metric("unknown")
            return None
    
    def get_order_status(self, order_id, symbol):
        """Получение статуса ордера"""
        try:
            max_attempts = 3
            base_delay = 0.5

            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.get_order_history(
                        category=self.market_category,
                        orderId=order_id,
                        symbol=symbol
                    )
                except Exception as exc:
                    error_type, error_message = self._classify_error(exception=exc)
                    self._record_error_metric(error_type)
                    self._log_attempt_result(
                        "get_order_status",
                        attempt,
                        max_attempts,
                        False,
                        error_type,
                        f"Исключение при запросе статуса: {error_message}",
                    )

                    if attempt < max_attempts:
                        time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue

                if response.get('retCode') == 0 and response.get('result'):
                    order_list = response['result'].get('list', [])
                    if order_list:
                        order = order_list[0]
                        logger.debug(
                            f"Order status: {order.get('orderStatus')}, Filled: {order.get('cumExecQty')}/{order.get('qty')}"
                        )
                        self._log_attempt_result(
                            "get_order_status",
                            attempt,
                            max_attempts,
                            True,
                            "ok",
                            f"Получен статус {order.get('orderStatus')}",
                        )
                        return order

                error_type, error_message = self._classify_error(response=response)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "get_order_status",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Статус не найден для {order_id}",
                )

                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))

            logger.warning(f"No order found for ID {order_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            self._record_error_metric("unknown")
            return None

    async def get_order_status_async(self, order_id: str, symbol: str) -> dict | None:
        """Асинхронно получает статус ордера через REST v5 с ретраями."""

        max_attempts = 3
        base_delay = 0.5
        params = {"category": self.market_category, "orderId": order_id, "symbol": symbol}

        for attempt in range(1, max_attempts + 1):
            try:
                session = await self._ensure_async_session()
                headers = self._build_signed_headers(params)
                async with session.get(
                    f"{self.config.API_BASE_URL}/v5/order/history", params=params, headers=headers
                ) as response:
                    payload = await response.json()

                if payload.get("retCode") == 0:
                    order_list = payload.get("result", {}).get("list") or []
                    if order_list:
                        return order_list[0]

                error_type, error_message = self._classify_error(response=payload)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "get_order_status_async",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Статус не найден для {order_id}",
                )
            except aiohttp.ClientError as exc:
                error_type, error_message = self._classify_error(exception=exc)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "get_order_status_async",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or str(exc),
                )

            if attempt < max_attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

        logger.warning("Не удалось получить статус ордера %s после асинхронных повторов", order_id)
        return None
    
    def cancel_order(self, order_id, symbol):
        """Отмена ордера"""
        try:
            # В тестнете не выполняем реальную отмену
            if self.config.TESTNET:
                logger.info(f"🧪 TESTNET MODE: Simulating order cancellation for {order_id}")
                return True
            
            response = self.session.cancel_order(
                category=self.market_category,
                orderId=order_id,
                symbol=symbol
            )
            
            if response.get('retCode') == 0:
                logger.info(f"CloseOperation: Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"CloseOperation failed: {response.get('retMsg', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False

    async def cancel_order_async(self, order_id: str, symbol: str) -> bool:
        """Асинхронно отменяет ордер с ретраями и таймаутом."""

        if self.config.TESTNET:
            logger.info("🧪 TESTNET MODE: Симулируем отмену ордера %s", order_id)
            return True

        max_attempts = 2
        base_delay = 0.5
        payload = {"category": self.market_category, "orderId": order_id, "symbol": symbol}
        body = json.dumps(payload, separators=(",", ":"))

        for attempt in range(1, max_attempts + 1):
            try:
                session = await self._ensure_async_session()
                headers = self._build_signed_headers(payload, body)
                async with session.post(
                    f"{self.config.API_BASE_URL}/v5/order/cancel", data=body, headers=headers
                ) as response:
                    result = await response.json()

                if result.get("retCode") == 0:
                    return True

                error_type, error_message = self._classify_error(response=result)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "cancel_order_async",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or f"Не удалось отменить {order_id}",
                )
            except aiohttp.ClientError as exc:
                error_type, error_message = self._classify_error(exception=exc)
                self._record_error_metric(error_type)
                self._log_attempt_result(
                    "cancel_order_async",
                    attempt,
                    max_attempts,
                    False,
                    error_type,
                    error_message or str(exc),
                )

            if attempt < max_attempts:
                await asyncio.sleep(base_delay * attempt)

        logger.warning("Не удалось отменить ордер %s после асинхронных попыток", order_id)
        return False
    
    def get_open_orders(self, symbol=None):
        """Получение открытых ордеров"""
        try:
            params = {'category': self.market_category}
            if symbol:
                params['symbol'] = symbol

            response = self.session.get_open_orders(**params)

            if response.get('retCode') == 0 and response.get('result'):
                return response['result'].get('list', [])
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []


__all__ = ["BybitClient", "BybitWebSocketManager"]
