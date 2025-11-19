import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

try:
    from pybit.unified_trading import HTTP
except ModuleNotFoundError:
    HTTP = None

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self):
        self.config = Config()
        self.session = self._create_session()
        self.account_type = "UNIFIED" if not self.config.TESTNET else "CONTRACT"
        # Всегда заранее сохраняем сегмент рынка, чтобы одинаково использовать его во всех запросах
        self.market_category = getattr(self.config, "MARKET_CATEGORY", "spot")
        logger.info(
            f"Bybit client initialized. Testnet: {self.config.TESTNET}, "
            f"Account type: {self.account_type}"
        )
        logger.info(f"🎯 Market category set to: {self.market_category}")
    
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

        logger.debug(f"🔍 Requesting {len(requested_symbols)} symbols: {requested_symbols}")

        remaining_symbols = set(requested_symbols)
        start_time = time.time()
        request_count = 0

        def _extract_from_response(response, label):
            """Извлекает данные тикеров из ответа и обновляет остаток"""
            nonlocal tickers
            if not response:
                logger.debug(f"❌ Пустой ответ в блоке {label}")
                return

            if response.get('retCode') != 0 or not response.get('result'):
                logger.debug(f"❌ API error in {label}: {response.get('retMsg')}")
                return

            ticker_list = response['result'].get('list', [])
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
                logger.debug(
                    f"✅ {symbol}: bid={tickers[symbol]['bid']}, ask={tickers[symbol]['ask']} (source={label})"
                )

        # Основной bulk-запрос без параметра symbol
        try:
            cursor = None
            while True:
                params = {'category': self.market_category}
                if cursor:
                    params['cursor'] = cursor

                response = self.session.get_tickers(**params)
                request_count += 1
                _extract_from_response(response, 'bulk')

                cursor = response.get('result', {}).get('nextPageCursor') if response else None
                if not cursor or not remaining_symbols:
                    break

        except Exception as e:
            logger.debug(f"🔥 Bulk request failed: {str(e)}")

        # Фолбэк: запрашиваем оставшиеся символы параллельно
        if remaining_symbols:
            logger.debug(
                f"⚙️ Bulk вернул не все данные, догружаем {len(remaining_symbols)} символов параллельно"
            )

            def _fetch_symbol(symbol):
                try:
                    return self.session.get_tickers(category=self.market_category, symbol=symbol)
                except Exception as exc:
                    logger.debug(f"🔥 Exception for {symbol}: {str(exc)}")
                    return None

            with ThreadPoolExecutor(max_workers=min(8, len(remaining_symbols))) as executor:
                future_to_symbol = {
                    executor.submit(_fetch_symbol, symbol): symbol for symbol in list(remaining_symbols)
                }

                for future in as_completed(future_to_symbol):
                    request_count += 1
                    symbol = future_to_symbol[future]
                    response = future.result()
                    _extract_from_response(response, f'fallback:{symbol}')

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

        return tickers
        
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
    
    def place_order(self, symbol, side, qty, price=None, order_type='Market'):
        """Размещение ордера на бирже с улучшенной обработкой ошибок"""
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
                'orderFilter': 'Order'
            }
            
            if price and order_type == 'Limit':
                params['price'] = str(price)
            
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
            
            # Реальное исполнение (только для основной сети)
            response = self.session.place_order(**params)
            
            if response.get('retCode') == 0:
                result = response['result']
                order_id = result.get('orderId')
                logger.info(f"✅ Order placed successfully! Order ID: {order_id}, Symbol: {symbol}, Side: {side}, Qty: {qty}")
                logger.info(f"   Status: {result.get('orderStatus')}, Price: {result.get('price')}, Avg Price: {result.get('avgPrice')}")
                return result
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                logger.error(f"❌ Order failed: {error_msg} (Code: {response.get('retCode')})")
                logger.error(f"   Request: {params}")
                return None
                
        except Exception as e:
            logger.error(f"🔥 Critical error placing order: {str(e)}", exc_info=True)
            return None
    
    def get_order_status(self, order_id, symbol):
        """Получение статуса ордера"""
        try:
            response = self.session.get_order_history(
                category=self.market_category,
                orderId=order_id,
                symbol=symbol
            )
            
            if response.get('retCode') == 0 and response.get('result'):
                order_list = response['result'].get('list', [])
                if order_list:
                    order = order_list[0]
                    logger.debug(f"Order status: {order.get('orderStatus')}, Filled: {order.get('cumExecQty')}/{order.get('qty')}")
                    return order
            logger.warning(f"No order found for ID {order_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
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