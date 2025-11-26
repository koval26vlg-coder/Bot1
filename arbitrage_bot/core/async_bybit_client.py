"""Асинхронный клиент Bybit для параллельного получения тикеров."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp
    from arbitrage_bot.exchanges.bybit_client import BybitWebSocketManager

from arbitrage_bot.core.config import Config

logger = logging.getLogger(__name__)


class AsyncBybitClient:
    """Асинхронный REST-клиент для получения котировок по API Bybit."""

    def __init__(self, config: Config | None = None, allow_missing_aiohttp: bool = False):
        self.config = config or Config()
        self.base_url = self.config.API_BASE_URL
        self.market_category = getattr(self.config, "MARKET_CATEGORY", "spot")
        self.allow_missing_aiohttp = allow_missing_aiohttp
        self._aiohttp: Any | None = None
        self._session: "aiohttp.ClientSession | None" = None
        self.ws_manager: "BybitWebSocketManager | None" = None
        self._temporarily_unavailable_symbols: set[str] = set()
        self._initialize_websocket_streams()

    async def __aenter__(self) -> "AsyncBybitClient":
        """Позволяет использовать клиент в async with и гарантировать закрытие сессии."""

        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Закрывает HTTP-сессию при выходе из контекста."""

        await self.close()

    async def close(self) -> None:
        """Закрывает HTTP-сессию и останавливает WebSocket-менеджер."""

        if self._session and not self._session.closed:
            await self._session.close()
        if self.ws_manager:
            self.ws_manager.stop()

    def _get_aiohttp(self):
        """Лениво импортирует aiohttp и возвращает модуль или выдаёт понятную ошибку."""

        if self._aiohttp is not None:
            return self._aiohttp

        message = (
            "Пакет 'aiohttp' не установлен. "
            "Установите зависимости командой: pip install -r requirements.txt"
        )

        try:
            import aiohttp
        except ImportError as exc:
            logger.error(message)
            if self.allow_missing_aiohttp:
                raise RuntimeError(message) from exc
            raise

        if not (hasattr(aiohttp, "ClientSession") and hasattr(aiohttp, "ClientTimeout")):
            logger.error(message)
            if self.allow_missing_aiohttp:
                raise RuntimeError(message)

        self._aiohttp = aiohttp
        return aiohttp

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Создаёт или возвращает существующую HTTP-сессию."""

        aiohttp = self._get_aiohttp()

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _initialize_websocket_streams(self) -> None:
        """Запускает WebSocket-менеджер для кэширования котировок."""

        try:
            from arbitrage_bot.exchanges.bybit_client import BybitWebSocketManager

            self.ws_manager = BybitWebSocketManager(self.config)
            self.ws_manager.start(self.config.SYMBOLS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось инициализировать WebSocket-стримы: %s", exc, exc_info=True)
            self.ws_manager = None

    def _classify_response(self, status: int, payload: dict | None) -> tuple[bool, str]:
        """Возвращает признак тяжёлой ошибки и текст для логирования."""

        if status == 429:
            return True, "Достигнут лимит запросов"
        if status >= 500:
            return True, f"Сервер вернул код {status}"
        if payload and payload.get("retCode") not in (0, None):
            return False, payload.get("retMsg") or f"Код ошибки {payload.get('retCode')}"
        return False, ""

    async def _request_with_backoff(self, params: dict, label: str) -> dict | None:
        """Выполняет запрос с экспоненциальным бэкоффом и единой обработкой ошибок."""

        max_attempts = getattr(self.config, "TICKER_MAX_RETRIES", 3)
        base_backoff = getattr(self.config, "TICKER_BACKOFF_BASE", 0.25)
        heavy_backoff = getattr(self.config, "TICKER_HEAVY_BACKOFF_BASE", 1.0)
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                session = await self._ensure_session()
                async with session.get(
                    f"{self.base_url}/v5/market/tickers", params=params
                ) as response:
                    status = response.status
                    payload = None

                    try:
                        payload = await response.json()
                    except self._get_aiohttp().ContentTypeError:
                        payload = None

                    is_heavy, message = self._classify_response(status, payload)

                    if status == 200 and payload and payload.get("retCode") == 0:
                        return payload

                    delay_base = heavy_backoff if is_heavy else base_backoff
                    delay = delay_base * (2 ** (attempt - 1))
                    last_error = message or f"HTTP {status}"

                    logger.warning(
                        "♻️ Ошибка запроса %s (попытка %s/%s, код: %s). Ждём %.2f c перед повтором",
                        label,
                        attempt,
                        max_attempts,
                        status,
                        delay,
                    )
                    await asyncio.sleep(delay)
            except self._get_aiohttp().ClientError as exc:
                last_error = str(exc)
                delay = heavy_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "🌐 Сетевая ошибка запроса %s (попытка %s/%s): %s. Пауза %.2f c",
                    label,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        logger.error("🚫 Запрос %s не удался после повторов: %s", label, last_error)
        return None

    def _extract_from_response(self, response: dict | None, remaining_symbols: set[str], label: str) -> dict:
        """Формирует словарь тикеров из ответа и обновляет набор недостающих."""

        tickers: dict[str, dict] = {}
        if not response or response.get("retCode") != 0:
            logger.debug("❌ Некорректный ответ %s: %s", label, response)
            return tickers

        ticker_list = response.get("result", {}).get("list")
        if ticker_list is None:
            logger.debug("ℹ️ Пустой список тикеров в блоке %s", label)
            return tickers

        if not isinstance(ticker_list, Iterable) or isinstance(ticker_list, (str, bytes, dict)):
            logger.debug("❌ Неверный формат списка тикеров в блоке %s", label)
            return tickers

        for ticker_data in ticker_list:
            symbol = ticker_data.get("symbol")
            if symbol not in remaining_symbols:
                continue

            tickers[symbol] = {
                "bid": self._safe_float(ticker_data.get("bid1Price", 0)),
                "ask": self._safe_float(ticker_data.get("ask1Price", 0)),
                "last": self._safe_float(ticker_data.get("lastPrice", 0)),
                "timestamp": ticker_data.get("time"),
            }
            remaining_symbols.discard(symbol)
            self._temporarily_unavailable_symbols.discard(symbol)

        return tickers

    def _safe_float(self, value):
        """Безопасно приводит значение к float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _validate_ticker_freshness(self, tickers: dict) -> set[str]:
        """Проверяет свежесть котировок, логирует и возвращает устаревшие символы."""

        if not tickers:
            return set()

        freshness_limit_ms = int(getattr(self.config, "TICKER_STALENESS_WARNING_SEC", 5.0) * 1000)
        now_ms = int(time.time() * 1000)
        stale: list[tuple[str, float]] = []

        for symbol, data in tickers.items():
            timestamp = data.get("timestamp")
            if not timestamp:
                continue

            try:
                age_ms = now_ms - int(float(timestamp))
            except (TypeError, ValueError):
                continue

            if age_ms > freshness_limit_ms:
                stale.append((symbol, age_ms / 1000))

        if stale:
            preview = ", ".join(f"{sym} ({age:.1f}с)" for sym, age in stale[:5])
            logger.warning(
                "🕒 Обнаружены устаревшие котировки (>%.1fс): %s",
                getattr(self.config, "TICKER_STALENESS_WARNING_SEC", 5.0),
                preview,
            )

        return {symbol for symbol, _ in stale}

    async def _refresh_stale_tickers(self, stale_symbols: set[str]) -> tuple[dict[str, dict], set[str]]:
        """Повторно загружает устаревшие котировки через альтернативные источники."""

        if not stale_symbols:
            return {}, set()

        logger.info(
            "🔁 Запускаем восстановление для %s устаревших тикеров: %s",
            len(stale_symbols),
            ", ".join(sorted(stale_symbols)),
        )

        refreshed: dict[str, dict] = {}
        failed = set(stale_symbols)
        concurrency = max(1, getattr(self.config, "ASYNC_TICKER_CONCURRENCY", 6))
        semaphore = asyncio.Semaphore(concurrency)

        async def _pull_symbol(symbol: str) -> tuple[str, dict | None]:
            async with semaphore:
                params = {"category": self.market_category, "symbol": symbol}
                return symbol, await self._request_with_backoff(params, f"stale:{symbol}")

        tasks = [asyncio.create_task(_pull_symbol(symbol)) for symbol in stale_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.debug("🔥 Исключение восстановления тикера: %s", result)
                continue

            symbol, response = result
            extracted = self._extract_from_response(response, {symbol}, f"stale:{symbol}")
            if extracted:
                refreshed.update(extracted)
                failed.discard(symbol)

        if refreshed:
            logger.info(
                "✅ Успешно восстановлены устаревшие тикеры: %s",
                ", ".join(sorted(refreshed)),
            )
        if failed:
            logger.warning(
                "🚧 Не удалось обновить устаревшие тикеры: %s",
                ", ".join(sorted(failed)),
            )

        return refreshed, failed

    async def _finalize_tickers(self, tickers: dict[str, dict]) -> dict[str, dict]:
        """Фильтрует устаревшие данные, восстанавливает их и обновляет кэш."""

        stale_symbols = self._validate_ticker_freshness(tickers)
        if stale_symbols:
            for symbol in stale_symbols:
                tickers.pop(symbol, None)

            refreshed, failed = await self._refresh_stale_tickers(stale_symbols)
            if refreshed:
                tickers.update(refreshed)

            if failed:
                self._temporarily_unavailable_symbols.update(failed)
                logger.debug(
                    "🛑 Помечаем устаревшие тикеры как временно недоступные: %s",
                    ", ".join(sorted(failed)),
                )

        if self.ws_manager and tickers:
            self.ws_manager.update_cache(tickers)

        return tickers

    async def get_tickers_async(self, symbols: list[str]) -> dict[str, dict]:
        """Асинхронно получает тикеры, используя кэш WebSocket и параллельные запросы."""

        requested_symbols = sorted(set(symbols))
        tickers: dict[str, dict] = {}

        if not requested_symbols:
            return tickers

        blocked_symbols = set(requested_symbols) & self._temporarily_unavailable_symbols
        remaining_symbols = set(requested_symbols)

        if blocked_symbols:
            logger.warning(
                "⏳ Пропускаем REST-запросы для временно недоступных тикеров: %s",
                ", ".join(sorted(blocked_symbols)),
            )

        cache_hits: dict[str, dict] = {}
        if self.ws_manager:
            cache_hits, fresh_missing = self.ws_manager.get_cached_tickers(
                requested_symbols,
                max_age=getattr(self.config, "TICKER_STALENESS_WARNING_SEC", 5.0),
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
                return await self._finalize_tickers(tickers)

        if getattr(self.config, "WEBSOCKET_PRICE_ONLY", False) and remaining_symbols:
            logger.warning(
                "📡 Включён режим WEBSOCKET_PRICE_ONLY, пропускаем REST для %s тикеров",
                len(remaining_symbols),
            )
            return await self._finalize_tickers(tickers)

        request_count = 0
        start_time = time.time()

        try:
            cursor = None
            while remaining_symbols - blocked_symbols:
                params: dict[str, str] = {"category": self.market_category}
                if cursor:
                    params["cursor"] = cursor

                response = await self._request_with_backoff(params, "bulk")
                request_count += 1
                tickers.update(self._extract_from_response(response, remaining_symbols, "bulk"))

                cursor = response.get("result", {}).get("nextPageCursor") if response else None
                if not cursor:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("🔥 Ошибка bulk-запроса: %s", exc)

        if remaining_symbols - blocked_symbols:
            logger.debug(
                "⚙️ Bulk вернул не все данные, догружаем %s символов параллельно",
                len(remaining_symbols - blocked_symbols),
            )

            fetch_tasks = []
            concurrency = max(1, getattr(self.config, "ASYNC_TICKER_CONCURRENCY", 6))
            semaphore = asyncio.Semaphore(concurrency)

            async def _fetch_symbol(symbol: str):
                async with semaphore:
                    params = {"category": self.market_category, "symbol": symbol}
                    return await self._request_with_backoff(params, f"fallback:{symbol}")

            fetchable = list(remaining_symbols - blocked_symbols)
            for symbol in fetchable:
                fetch_tasks.append(_fetch_symbol(symbol))

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for symbol, response in zip(fetchable, results):
                request_count += 1
                if isinstance(response, Exception):
                    logger.debug("🔥 Исключение для %s: %s", symbol, response)
                    continue
                tickers.update(self._extract_from_response(response, remaining_symbols, f"fallback:{symbol}"))

        if remaining_symbols:
            missing_preview = ", ".join(sorted(remaining_symbols))
            logger.warning(
                "🚫 После всех запросов отсутствуют тикеры: %s. Помечаем как временно недоступные.",
                missing_preview,
            )
            self._temporarily_unavailable_symbols.update(remaining_symbols)
            remaining_symbols.clear()

        duration = time.time() - start_time
        logger.debug(
            "📊 Получено %s тикеров (запросов: %s, заняло: %.2f с)",
            len(tickers),
            request_count,
            duration,
        )

        return await self._finalize_tickers(tickers)

    def get_unavailable_symbols(self) -> set[str]:
        """Возвращает копию множества временно исключённых тикеров."""

        return set(self._temporarily_unavailable_symbols)
