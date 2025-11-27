import logging

import requests

from arbitrage_bot.core.config import Config


logger = logging.getLogger(__name__)


class OkxClient:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.session = requests.Session()
        self.base_url = getattr(self.config, 'API_BASE_URL', 'https://www.okx.com')
        self.market_category = 'spot'
        self._temporarily_unavailable_symbols = set()

        # Убеждаемся, что маппинг instId заполнен
        try:
            _ = self.config.SYMBOLS
        except Exception:
            logger.debug("Не удалось предзагрузить список тикеров OKX")

        logger.info("OKX client initialized. Base URL: %s", self.base_url)

    def _safe_float(self, value, default=0.0):
        """Безопасно приводит значение к float, возвращает default при ошибке."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _inst_id(self, symbol):
        """Преобразует тикер без дефиса в instId OKX."""
        if not symbol:
            return None

        explicit = self.config.get_okx_inst_id(symbol)
        if explicit:
            return explicit

        normalized = symbol.replace('-', '').upper()
        if len(normalized) < 3:
            return None

        base, quote = normalized[:-4], normalized[-4:]
        return f"{base}-{quote}"

    def add_order_listener(self, callback):
        """Поддержка WebSocket не реализована, метод оставлен для совместимости."""

        logger.debug("Слушатели ордеров для OKX не поддерживаются")

    def get_tickers(self, symbols):
        """Возвращает котировки OKX, фильтруя по нужным тикерам."""

        requested = {sym.replace('-', '').upper() for sym in symbols}
        tickers = {}

        if not requested:
            return tickers

        try:
            response = self.session.get(
                f"{self.base_url}/api/v5/market/tickers",
                params={'instType': 'SPOT'},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get('data', []) or []:
                inst_id = item.get('instId')
                normalized = (inst_id or '').replace('-', '').upper()
                if normalized not in requested:
                    continue

                tickers[normalized] = {
                    'bid': self._safe_float(item.get('bidPx')),
                    'ask': self._safe_float(item.get('askPx')),
                    'last': self._safe_float(item.get('last')),
                    'timestamp': item.get('ts')
                }

            missing = requested - set(tickers)
            if missing:
                logger.warning(
                    "⚠️ OKX не вернул тикеры: %s", ', '.join(sorted(missing))
                )
            return tickers
        except requests.RequestException as exc:
            logger.warning("Не удалось получить котировки OKX: %s", exc)
            return tickers

    def get_funding_rates(self, symbols):
        """Заглушка для фандинга OKX: возвращаем пустой словарь для совместимости."""

        logger.debug("Запрос фандинга для OKX не реализован, возвращаем пустой ответ")
        return {}

    def get_order_book(self, symbol, depth=5):
        """Запрашивает стакан для тикера OKX."""

        inst_id = self._inst_id(symbol)
        if not inst_id:
            logger.warning("Не удалось построить instId для %s", symbol)
            return {}

        try:
            response = self.session.get(
                f"{self.base_url}/api/v5/market/books",
                params={'instId': inst_id, 'sz': depth},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            entries = (data.get('data') or [{}])[0]
            return {
                'bids': entries.get('bids') or [],
                'asks': entries.get('asks') or [],
            }
        except requests.RequestException as exc:
            logger.warning("Ошибка получения стакана OKX для %s: %s", inst_id, exc)
            return {}

    def place_order(self, **params):
        """Заглушка размещения ордеров для OKX."""

        logger.error("Размещение ордеров на OKX не реализовано в текущем профиле")
        return {'orderId': None, 'orderStatus': 'failed', 'instType': 'SPOT'}

    def cancel_order(self, order_id, symbol):
        """Заглушка отмены ордеров."""

        logger.warning("Отмена ордеров на OKX не поддерживается: %s", order_id)
        return None

    def get_order_status(self, order_id, symbol=None):
        """Заглушка статуса ордера."""

        logger.warning("Запрос статуса ордера на OKX не реализован: %s", order_id)
        return None

    def get_balance(self, coin='USDT'):
        """Возвращает пустой баланс для совместимости, чтобы не падать в режиме OKX."""

        logger.warning("Баланс OKX не поддерживается, возвращаем нулевой")
        return {'coin': coin, 'available': 0.0, 'total': 0.0}


__all__ = ["OkxClient"]
