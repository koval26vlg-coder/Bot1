import logging
import time
import threading
import random
import math
import asyncio
from collections import deque
from datetime import (
    datetime,
    timedelta,
)
from statistics import NormalDist, pstdev
import os  # Исправлено: добавлен импорт os
from arbitrage_bot.core.config import Config
from arbitrage_bot.exchanges.bybit_client import BybitClient
from arbitrage_bot.exchanges.okx_client import OkxClient

logger = logging.getLogger(__name__)

class RiskManager:
    """Менеджер рисков для реальной торговли"""

    def __init__(self, client: BybitClient | OkxClient | None = None, config: Config | None = None):
        self.max_daily_loss = 5.0  # Максимальный убыток в день в USDT
        self.max_trade_size_percent = 10  # Максимальный размер сделки в процентах от баланса
        self.max_consecutive_losses = 3  # Максимальное количество убыточных сделок подряд
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.min_trade_interval = 60  # Минимальный интервал между сделками в секундах
        self.var_confidence = 0.95  # Уровень доверия для расчёта VaR
        self.var_tolerance = 0.1  # Доля VaR, допускаемая в качестве планируемого риска
        self.capital_risk_fraction = 0.05  # Максимальная доля капитала под риск на сделку
        self.returns_history = deque(maxlen=500)  # История доходностей для расчёта волатильности
        self.last_portfolio_value = None
        self.trading_blocked_until: datetime | None = None
        self.client = client
        self.last_reset_date = datetime.now().date()
        self.config = config or Config()

    def get_risk_level(self) -> str:
        """Оценивает текущий уровень риска на основе потерь и серии неудачных сделок."""

        if self.daily_loss >= self.max_daily_loss * 0.8 or self.consecutive_losses >= self.max_consecutive_losses:
            return 'high'

        if self.daily_loss >= self.max_daily_loss * 0.4 or self.consecutive_losses >= (self.max_consecutive_losses // 2):
            return 'medium'

        return 'low'

    def calculate_dynamic_trade_size(self, balance: float, market_volatility: float | None = None) -> float:
        """Рассчитывает динамический объём сделки с учётом волатильности и текущего уровня риска.

        Базовый размер ограничивается 10% от баланса и константой MAX_TRADE_AMOUNT.
        Повышенная волатильность снижает коэффициент объёма, а высокий уровень риска дополнительно
        сжимает размер сделки. При спокойном рынке допускается небольшой рост объёма относительно базы."""

        if balance is None or balance <= 0:
            return 0.0

        max_trade_amount = getattr(self.config, 'MAX_TRADE_AMOUNT', balance * 0.1)
        base_size = min(balance * 0.1, max_trade_amount)

        volatility = market_volatility or 0.0
        if volatility >= 3:
            volatility_factor = 0.6
        elif volatility >= 1:
            volatility_factor = 0.85
        else:
            volatility_factor = 1.1

        risk_level = self.get_risk_level()
        risk_factor_map = {'high': 0.5, 'medium': 0.75, 'low': 1.0}
        risk_factor = risk_factor_map.get(risk_level, 1.0)

        dynamic_size = base_size * volatility_factor * risk_factor
        return max(0.0, min(dynamic_size, max_trade_amount))

    def calculate_var(self, portfolio_value: float, time_horizon: int = 1) -> float:
        """Оценивает VaR на основе истории доходностей."""

        if portfolio_value is None or portfolio_value <= 0:
            return 0.0

        if len(self.returns_history) < 2:
            # Недостаточно данных, используем консервативную оценку от капитала
            return portfolio_value * self.capital_risk_fraction

        volatility = pstdev(self.returns_history)
        if volatility <= 0:
            return portfolio_value * self.capital_risk_fraction

        z_score = abs(NormalDist().inv_cdf(1 - self.var_confidence))
        horizon_scale = math.sqrt(max(time_horizon, 1))
        return portfolio_value * volatility * z_score * horizon_scale

    def _reset_daily_counters_if_needed(self, current_time: datetime):
        """Сбрасывает дневные лимиты и блокировки при смене суток."""

        if self.last_reset_date and current_time.date() > self.last_reset_date:
            self.daily_loss = 0.0
            self.consecutive_losses = 0
            self.trading_blocked_until = None
            self.last_reset_date = current_time.date()
            logger.info("🔄 Сброшены дневные лимиты и блокировки по смене суток")

    def _calculate_end_of_day(self, current_time: datetime) -> datetime:
        """Возвращает временную метку конца текущих суток."""

        next_day = current_time.date() + timedelta(days=1)
        return datetime.combine(next_day, datetime.min.time())

    def _activate_block(self, reason: str, current_time: datetime):
        """Устанавливает блокировку торгов до конца суток с логированием причины."""

        end_of_day = self._calculate_end_of_day(current_time)
        self.trading_blocked_until = end_of_day
        logger.critical(
            "🚫 Блокировка торгов до %s из-за превышения лимита: %s",
            end_of_day.strftime("%Y-%m-%d %H:%M:%S"),
            reason,
        )

    def _resolve_portfolio_value(self, trade_plan, explicit_value: float | None):
        """Пытается определить актуальную стоимость портфеля."""

        for candidate in [explicit_value, trade_plan.get('portfolio_value'), self.last_portfolio_value]:
            if candidate is not None and candidate > 0:
                return float(candidate)

        if self.client:
            try:
                balance = self.client.get_balance('USDT')
                balance_value = balance.get('total') or balance.get('available')
                if balance_value:
                    return float(balance_value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️ Не удалось получить баланс для расчёта лимита сделки: %s", exc)

        return None

    def _extract_trade_notional(self, trade_plan) -> float:
        """Извлекает или оценивает объём сделки в USDT."""

        numeric_keys = [
            'trade_amount',
            'amount_usdt',
            'notional',
            'notional_usdt',
            'order_amount',
            'trade_value',
            'estimated_trade_value',
        ]

        for key in numeric_keys:
            value = trade_plan.get(key)
            if value is not None:
                try:
                    numeric_value = float(value)
                    if numeric_value > 0:
                        return numeric_value
                except (TypeError, ValueError):
                    continue

        # Пытаемся вычислить нотионал по шагам сделки
        for key, payload in trade_plan.items():
            if isinstance(payload, dict):
                amount = payload.get('amount') or payload.get('qty')
                price = payload.get('price')
                try:
                    if amount is not None and price is not None:
                        notional = float(amount) * float(price)
                        if notional > 0:
                            return notional
                except (TypeError, ValueError):
                    continue

        return 0.0

    def can_execute_trade(self, trade_plan, portfolio_value: float | None = None):
        """Проверка возможности выполнения сделки"""
        current_time = datetime.now()

        self._reset_daily_counters_if_needed(current_time)

        if self.trading_blocked_until and current_time < self.trading_blocked_until:
            logger.error(
                "🚫 Торговля заблокирована до %s", self.trading_blocked_until.strftime("%Y-%m-%d %H:%M:%S")
            )
            return False

        if self.daily_loss >= self.max_daily_loss:
            self._activate_block("превышен дневной убыток", current_time)
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._activate_block("серия убыточных сделок", current_time)
            return False

        portfolio_value = self._resolve_portfolio_value(trade_plan, portfolio_value)
        var_value = self.calculate_var(portfolio_value, time_horizon=1) if portfolio_value else 0.0

        planned_loss = abs(trade_plan.get('max_loss_usdt') or trade_plan.get('estimated_loss_usdt') or trade_plan.get('estimated_profit_usdt', 0))
        var_limit = var_value * self.var_tolerance if var_value else 0.0
        capital_limit = portfolio_value * self.capital_risk_fraction if portfolio_value else 0.0

        effective_loss_limit = None
        for candidate in [var_limit, capital_limit]:
            if candidate > 0:
                effective_loss_limit = candidate if effective_loss_limit is None else min(effective_loss_limit, candidate)

        if effective_loss_limit is None and portfolio_value:
            effective_loss_limit = portfolio_value * 0.02  # Минимальный ограничитель при отсутствии статистики

        if effective_loss_limit is not None and planned_loss > effective_loss_limit:
            logger.warning(
                "🚫 Планируемый убыток %.4f USDT превышает допустимый лимит %.4f USDT на основе VaR",
                planned_loss,
                effective_loss_limit,
            )
            return False

        # Проверка максимального размера сделки относительно баланса
        trade_notional = self._extract_trade_notional(trade_plan)
        if portfolio_value and trade_notional > 0:
            allowed_notional = portfolio_value * (self.max_trade_size_percent / 100)
            if trade_notional > allowed_notional:
                logger.warning(
                    "🚫 Объём сделки %.4f USDT превышает разрешённый предел %.4f USDT (%.1f%% от баланса)",
                    trade_notional,
                    allowed_notional,
                    self.max_trade_size_percent,
                )
                return False

        # Проверка интервала между сделками
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
            logger.warning(f"⏳ Слишком частые сделки. Ожидайте {(current_time - self.last_trade_time).total_seconds():.0f} секунд")
            return False

        # Проверка минимальной ожидаемой прибыли
        estimated_profit = trade_plan.get('estimated_profit_usdt', 0)
        if estimated_profit < 0.01:  # Минимальная прибыль 0.01 USDT
            logger.warning(f"📉 Слишком маленькая прибыль: {estimated_profit:.4f} USDT")
            return False

        return True
    
    def update_after_trade(self, trade_record):
        """Обновление статистики после сделки"""
        profit = trade_record.get('total_profit', 0)
        portfolio_value = trade_record.get('portfolio_value') or self.last_portfolio_value

        if portfolio_value and portfolio_value > 0:
            self.last_portfolio_value = portfolio_value + profit
            period_return = profit / portfolio_value
            self.returns_history.append(period_return)

        if profit < 0:
            self.daily_loss += abs(profit)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.last_trade_time = datetime.now()

        # Проверка лимитов
        if self.daily_loss >= self.max_daily_loss:
            self._activate_block("превышен дневной убыток", self.last_trade_time)
            logger.critical(f"🔥 Достигнут максимальный дневной убыток: {self.daily_loss:.2f} USDT")

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._activate_block("серия убыточных сделок", self.last_trade_time)
            logger.critical(f"🔥 Достигнуто максимальное количество убыточных сделок подряд: {self.consecutive_losses}")


class ContingentOrderOrchestrator:
    """Оркестратор контингентных ордеров и поэтапного исполнения с хеджированием"""

    def __init__(self, client: BybitClient, config: Config):
        self.client = client
        self.config = config
        self.default_timeout = getattr(config, 'MAX_TRIANGLE_EXECUTION_TIME', 30)
        self.loss_limit_usdt = float(os.getenv('CONTINGENT_MAX_LOSS_USDT', '10'))
        self._order_events: dict[str, dict] = {}
        self._order_events_lock = threading.Lock()

        if hasattr(self.client, 'add_order_listener'):
            self.client.add_order_listener(self._handle_order_event)

    def execute_sequence(self, legs: list[dict], hedge_leg: dict | None = None, max_loss_usdt: float | None = None, timeout: int | None = None):
        """Выполняет цепочку ног с проверкой статусов и хеджем при сбоях"""

        if not legs:
            logger.warning("⚠️ Передан пустой список ног для оркестратора")
            return None

        timeout_sec = timeout if timeout is not None else self.default_timeout
        loss_cap = max_loss_usdt if max_loss_usdt is not None else self.loss_limit_usdt
        slippage_tolerance = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)

        self._subscribe_leg_tickers(legs)

        executed_orders = []
        hedge_actions = []
        active_orders = []
        amount_scale = 1.0
        start_time = time.time()
        deadline_ts = start_time + timeout_sec

        for idx, leg in enumerate(legs, start=1):
            leg_payload = dict(leg)
            leg_payload['amount'] = float(leg_payload.get('amount', 0) or 0) * amount_scale

            if time.time() >= deadline_ts:
                logger.error("⏳ Тайм-аут цепочки до размещения шага %s/%s", idx, len(legs))
                self._cancel_previous_orders(executed_orders + active_orders)
                hedge = self._apply_hedge(
                    hedge_leg,
                    leg_payload,
                    executed_orders,
                    loss_cap,
                    reason="тайм-аут цепочки",
                )
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('timeout', executed_orders, hedge_actions, amount_scale, loss_cap)

            if not self._check_quote_alignment(leg_payload, slippage_tolerance):
                logger.error("🚫 Несоответствие котировок для %s, цепочка будет остановлена", leg_payload.get('symbol'))
                self._cancel_all_active(active_orders)
                hedge = self._apply_hedge(hedge_leg, leg_payload, executed_orders, loss_cap, reason="несоответствие котировок")
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('failed', executed_orders, hedge_actions, amount_scale, loss_cap)

            logger.info("🧭 Оркестратор: шаг %s/%s для %s", idx, len(legs), leg_payload.get('symbol'))
            remaining_time = max(0.0, deadline_ts - time.time())
            if remaining_time <= 0:
                logger.error("⏳ Тайм-аут цепочки перед мониторингом шага %s", idx)
                self._cancel_previous_orders(executed_orders + active_orders)
                hedge = self._apply_hedge(
                    hedge_leg,
                    leg_payload,
                    executed_orders,
                    loss_cap,
                    reason="тайм-аут цепочки",
                )
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('timeout', executed_orders, hedge_actions, amount_scale, loss_cap)

            order_result, status, fill_ratio = self._place_and_monitor(
                leg_payload,
                min(timeout_sec, remaining_time),
                slippage_tolerance,
                deadline_ts,
            )

            if order_result:
                executed_orders.append(order_result)
                if self._normalize_status(order_result.get('orderStatus')) not in {'filled', 'cancelled'}:
                    active_orders.append(order_result)

            if status in {'timeout', 'cancelled', 'failed'} or fill_ratio <= 0:
                self._cancel_all_active(active_orders)
                hedge = self._apply_hedge(hedge_leg, leg_payload, executed_orders, loss_cap, reason="сбой шага")
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('failed', executed_orders, hedge_actions, amount_scale, loss_cap)

            if status == 'partial' and fill_ratio < 1.0:
                amount_scale *= fill_ratio
                self._cancel_all_active(active_orders)
                hedge = self._apply_hedge(
                    hedge_leg,
                    leg_payload,
                    executed_orders,
                    loss_cap,
                    reason="частичное исполнение",
                    unfilled_ratio=1 - fill_ratio,
                )
                if hedge:
                    hedge_actions.append(hedge)
                return self._build_report('partial', executed_orders, hedge_actions, amount_scale, loss_cap)

        return self._build_report('completed', executed_orders, hedge_actions, amount_scale, loss_cap)

    def _place_and_monitor(self, leg: dict, timeout_sec: int, slippage_tolerance: float, deadline_ts: float | None = None):
        """Размещает ордер и ждёт его исполнения или тайм-аута"""

        order = self.client.place_order(
            symbol=leg['symbol'],
            side=leg['side'],
            qty=leg['amount'],
            price=leg.get('price'),
            order_type=leg.get('type', 'Market'),
            trigger_price=leg.get('trigger_price'),
            trigger_by=leg.get('trigger_by', 'LastPrice'),
            reduce_only=leg.get('reduce_only', False),
        )

        if not order:
            logger.error("❌ Ордер шага не размещён")
            return None, 'failed', 0.0

        order_id = order.get('orderId')
        symbol = leg['symbol']
        status = self._normalize_status(order.get('orderStatus'))
        fill_ratio = self._calc_fill_ratio(order)

        if status in {'filled', 'cancelled'} or not order_id:
            return order, status, fill_ratio

        start_time = time.time()
        last_status = status

        while time.time() - start_time < timeout_sec and (deadline_ts is None or time.time() < deadline_ts):
            cached_status, cached_fill = self._get_ws_order_status(order_id)
            if cached_status:
                last_status = self._normalize_status(cached_status)
                fill_ratio = cached_fill if cached_fill is not None else fill_ratio
                if last_status in {'filled', 'cancelled'}:
                    order['orderStatus'] = cached_status
                    base_qty = self._safe_float(order.get('qty'))
                    order['cumExecQty'] = (
                        order.get('cumExecQty') or base_qty if cached_fill is None else cached_fill * base_qty
                    )
                    return order, last_status, fill_ratio

            if not self._check_quote_alignment(leg, slippage_tolerance):
                logger.error("📉 Котировки ушли за предел допуска по %s", symbol)
                self.client.cancel_order(order_id, symbol)
                return order, 'failed', fill_ratio

            fetched = self.client.get_order_status(order_id, symbol) or {}
            if fetched:
                order.update(fetched)
                last_status = self._normalize_status(fetched.get('orderStatus'))
                fill_ratio = self._calc_fill_ratio(fetched)

                if last_status in {'filled', 'cancelled'}:
                    return order, last_status, fill_ratio

            time.sleep(1)

        logger.error("⏳ Тайм-аут исполнения ордера %s", order_id)
        return order, last_status or 'timeout', fill_ratio

    def _subscribe_leg_tickers(self, legs: list[dict]):
        """Подписывается на все ногами цепочки для получения свежих котировок."""

        if not getattr(self.client, 'ws_manager', None):
            return

        symbols = {leg.get('symbol') for leg in legs if leg.get('symbol')}
        try:
            existing = getattr(self.client.ws_manager, '_symbols', set()) or set()
            updated = list(existing.union(symbols))
            self.client.ws_manager.start(updated)
            logger.info("📡 Подписка на WebSocket тикеры для ног: %s", ', '.join(sorted(symbols)))
        except Exception as exc:
            logger.debug("Не удалось обновить подписку на котировки: %s", exc)

    def _handle_order_event(self, event: dict):
        """Сохраняет последнюю информацию об ордерах из приватного стрима."""

        order_id = event.get('orderId')
        if not order_id:
            return

        with self._order_events_lock:
            self._order_events[order_id] = {
                'status': event.get('orderStatus'),
                'cumExecQty': event.get('cumExecQty'),
                'qty': event.get('qty') or event.get('leavesQty'),
            }

        if hasattr(self.client, '_is_status_uncertain') and hasattr(self.client, '_ensure_order_finalized'):
            status = event.get('orderStatus')
            symbol = event.get('symbol')
            if self.client._is_status_uncertain(status):
                self._fire_and_forget(
                    self._refresh_order_status_async(order_id, symbol, status, fallback_payload=event)
                )

    async def _refresh_order_status_async(self, order_id: str, symbol: str, status: str | None, fallback_payload: dict):
        """Асинхронно обновляет сведения по ордеру и записывает финальный статус."""

        finalized = await self.client._ensure_order_finalized(
            order_id,
            symbol,
            status,
            fallback_payload=fallback_payload,
        )

        if finalized:
            with self._order_events_lock:
                self._order_events[order_id] = {
                    'status': finalized.get('orderStatus'),
                    'cumExecQty': finalized.get('cumExecQty'),
                    'qty': finalized.get('qty'),
                }

    def _fire_and_forget(self, coro):
        """Запускает корутину без блокировки основного цикла."""

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            threading.Thread(target=asyncio.run, args=(coro,), daemon=True).start()

    def _get_ws_order_status(self, order_id: str) -> tuple[str | None, float | None]:
        """Возвращает статус и долю исполнения из WebSocket, если доступно."""

        with self._order_events_lock:
            event = self._order_events.get(order_id)

        if not event:
            return None, None

        status = self._normalize_status(event.get('status'))
        fill_ratio = None
        try:
            qty = self._safe_float(event.get('qty'))
            filled = self._safe_float(event.get('cumExecQty'))
            if qty > 0:
                fill_ratio = min(1.0, filled / qty)
        except Exception:
            fill_ratio = None

        return status, fill_ratio

    def _check_quote_alignment(self, leg: dict, tolerance: float) -> bool:
        """Сверяет плановую цену с актуальными котировками из WebSocket."""

        symbol = leg.get('symbol')
        target_price = leg.get('price')
        side = (leg.get('side') or '').lower()

        if not symbol or not target_price or not getattr(self.client, 'ws_manager', None):
            return True

        cached, _ = self.client.ws_manager.get_cached_tickers([symbol])
        ticker = cached.get(symbol) if cached else None
        if not ticker:
            return True

        market_price = ticker.get('ask') if side == 'buy' else ticker.get('bid')
        if not market_price:
            return True

        deviation = abs(target_price - market_price) / target_price if target_price else 0
        if deviation > tolerance:
            logger.warning(
                "⚠️ Цена для %s ушла на %.4f%% (допуск %.4f%%)",
                symbol,
                deviation * 100,
                tolerance * 100,
            )
            return False

        return True

    def _cancel_all_active(self, active_orders: list[dict]):
        """Отменяет все активные ордера цепочки."""

        for order in active_orders or []:
            order_id = order.get('orderId')
            symbol = order.get('symbol')
            if order_id and symbol:
                try:
                    self.client.cancel_order(order_id, symbol)
                    logger.info("🛑 Отменён ордер %s для %s", order_id, symbol)
                except Exception as exc:
                    logger.warning("Не удалось отменить %s: %s", order_id, exc)

    def _cancel_previous_orders(self, orders: list[dict]):
        """Отменяет уже размещённые ордера при глобальном тайм-ауте."""

        for order in orders or []:
            order_id = order.get('orderId')
            symbol = order.get('symbol')
            if order_id and symbol:
                try:
                    self.client.cancel_order(order_id, symbol)
                    logger.info("🛑 Отменён ранее размещённый ордер %s для %s", order_id, symbol)
                except Exception as exc:
                    logger.warning("Не удалось отменить ранее размещённый ордер %s: %s", order_id, exc)

    def _apply_hedge(self, hedge_leg, failed_leg, executed_orders, loss_cap, reason: str, unfilled_ratio: float | None = None):
        """Проводит компенсирующее действие при сбое или частичном исполнении"""

        hedge_payload = self._prepare_hedge_payload(hedge_leg, failed_leg, executed_orders, loss_cap, unfilled_ratio)
        if not hedge_payload:
            logger.warning("⚠️ Хедж не выполнен: нет подходящего payload")
            return None

        logger.warning("🛡️ Запуск хеджа (%s) для %s", reason, hedge_payload['symbol'])
        hedge_result = self.client.place_order(**hedge_payload)

        if hedge_result:
            hedge_status = self._normalize_status(hedge_result.get('orderStatus'))
            return {
                'reason': reason,
                'payload': hedge_payload,
                'result': hedge_result,
                'status': hedge_status,
            }

        logger.error("❌ Хедж не размещён")
        return None

    def _prepare_hedge_payload(self, hedge_leg, failed_leg, executed_orders, loss_cap, unfilled_ratio):
        """Формирует параметры хедж-ордера, ограничивая риск по сумме"""

        last_fill = self._extract_last_fill(executed_orders, failed_leg)
        if not last_fill:
            return None

        qty, price, symbol, side = last_fill
        target_symbol = (hedge_leg or {}).get('symbol', symbol)
        target_side = (hedge_leg or {}).get('side')

        hedge_side = target_side or ('sell' if side.lower() == 'buy' else 'buy')
        hedge_price = (hedge_leg or {}).get('price', price)
        hedge_type = (hedge_leg or {}).get('type', 'Market')

        effective_unfilled = qty * (unfilled_ratio or 1.0)
        capped_qty = self._cap_qty_by_loss(effective_unfilled, hedge_price, loss_cap)
        if capped_qty <= 0:
            return None

        return {
            'symbol': target_symbol,
            'side': hedge_side,
            'qty': capped_qty,
            'price': hedge_price,
            'order_type': hedge_type,
            'reduce_only': True,
        }

    def _extract_last_fill(self, executed_orders, fallback_leg):
        """Извлекает информацию о последнем исполненном объёме"""

        source = executed_orders[-1] if executed_orders else None
        if not source and fallback_leg:
            qty = float(fallback_leg.get('amount', 0) or 0)
            price = float(fallback_leg.get('price', 0) or 0)
            return qty, price, fallback_leg.get('symbol'), fallback_leg.get('side', '')

        if not source:
            return None

        qty = self._safe_float(source.get('cumExecQty') or source.get('qty'))
        price = self._safe_float(source.get('avgPrice') or source.get('price'))
        symbol = source.get('symbol') or fallback_leg.get('symbol')
        side = source.get('side') or fallback_leg.get('side', '')
        return qty, price, symbol, side

    def _cap_qty_by_loss(self, qty: float, price: float | None, loss_cap: float) -> float:
        """Ограничивает объём для хеджа, чтобы потенциальный убыток не превысил лимит"""

        if not price or price <= 0 or loss_cap <= 0:
            return qty

        max_qty = loss_cap / price
        return min(qty, max_qty)

    def _calc_fill_ratio(self, order_data: dict) -> float:
        """Рассчитывает долю исполнения ордера"""

        qty = self._safe_float(order_data.get('qty'))
        filled = self._safe_float(order_data.get('cumExecQty'))
        if qty <= 0:
            return 0.0
        return min(1.0, filled / qty)

    def _normalize_status(self, status: str | None) -> str:
        """Приводит статус ордера к нормализованной форме"""

        if not status:
            return 'unknown'
        status_lower = status.lower()
        if 'partial' in status_lower:
            return 'partial'
        if 'cancel' in status_lower:
            return 'cancelled'
        if 'filled' in status_lower:
            return 'filled'
        return status_lower

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование к float"""

        try:
            if value is None:
                return default
            if isinstance(value, str):
                value = value.strip()
                if value == '':
                    return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_report(self, status, executed_orders, hedges, amount_scale, loss_cap=None):
        """Формирует итоговый отчёт по цепочке"""

        hedge_cost = 0.0
        for hedge in hedges or []:
            payload = hedge.get('payload') or {}
            hedge_cost += self._safe_float(payload.get('price')) * self._safe_float(payload.get('qty'))

        estimated_profit = 0.0
        if hedge_cost > 0:
            limit = loss_cap if loss_cap is not None else hedge_cost
            estimated_profit = -min(limit, hedge_cost)

        return {
            'status': status,
            'executed': executed_orders,
            'hedges': hedges,
            'effective_scale': amount_scale,
            'estimated_profit': estimated_profit,
        }


class RealTradingExecutor:
    """Исполнение реальных ордеров с режимом симуляции и постепенного перехода к реальной торговле"""
    
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.client = self._select_client()
        self.is_real_mode = False
        self.trade_history = []
        self.risk_manager = RiskManager(client=self.client, config=self.config)
        self.contingent_orchestrator = ContingentOrderOrchestrator(self.client, self.config)
        # Фиктивный баланс для симуляции, чтобы можно было управлять проверками ликвидности
        self._simulated_balance_usdt = self._load_simulated_balance()
        self.recent_order_events = deque(maxlen=200)
        self._last_execution_hint = None

        # Режим симуляции (True = симуляция, False = реальные ордера)
        simulation_env = os.getenv('TRADE_SIMULATION_MODE')
        legacy_simulation_env = os.getenv('SIMULATION_MODE')

        if simulation_env is not None:
            self.simulation_mode = simulation_env.lower() == 'true'
            mode_source = 'TRADE_SIMULATION_MODE'
        elif legacy_simulation_env is not None:
            self.simulation_mode = legacy_simulation_env.lower() == 'true'
            mode_source = 'SIMULATION_MODE'
        else:
            self.simulation_mode = self.config.TESTNET
            mode_source = 'TESTNET'

        self.paper_trading_mode = getattr(self.config, 'PAPER_TRADING_MODE', False) or (
            os.getenv('PAPER_TRADING_MODE', 'false').lower() == 'true'
        )
        if self.paper_trading_mode:
            self.simulation_mode = True
            mode_source = 'PAPER_TRADING_MODE'

        logger.info(
            "🔄 Режим торговли: %s (источник: %s)",
            'симуляция' if self.simulation_mode else 'реальные ордера',
            mode_source
        )
        if self.paper_trading_mode:
            logger.info("📄 Paper trading активирован: используется эмуляция стакана и безрисковое исполнение")
        logger.info(
            "📡 Режим котировок %s: %s",
            self.config.PRIMARY_EXCHANGE.upper(),
            'testnet' if self.config.TESTNET else 'mainnet'
        )

        # Подписываемся на события ордеров для быстрого реагирования
        self.client.add_order_listener(self._handle_order_event)

        if self.simulation_mode and not self.config.TESTNET:
            logger.warning(
                "🧪 Симуляция исполнения сделок при работе с реальными котировками Bybit"
            )
            logger.debug(
                "💳 Тестовый симулированный баланс: %.2f USDT", self._simulated_balance_usdt
            )
    
    def set_real_mode(self, enable_real_mode):
        """Переключение в реальный режим торговли"""
        if enable_real_mode and self.simulation_mode:
            # Запрашиваем подтверждение перед переходом в реальный режим
            confirmation = self._request_real_mode_confirmation()
            if confirmation:
                self.simulation_mode = False
                self.is_real_mode = True
                logger.info("✅ Переключено в реальный режим торговли")
                return True
            else:
                logger.warning("❌ Отмена перехода в реальный режим")
                return False
        return False

    def _select_client(self):
        """Выбирает биржевой клиент в зависимости от конфигурации."""

        exchange = getattr(self.config, 'PRIMARY_EXCHANGE', 'bybit').lower()
        if exchange == 'okx':
            return OkxClient(config=self.config)
        return BybitClient(config=self.config)
    
    def _request_real_mode_confirmation(self):
        """Запрос подтверждения перед переходом в реальный режим"""
        logger.warning("⚠️  ВНИМАНИЕ! Вы собираетесь перейти в реальный режим торговли!")
        logger.warning("⚠️  Будут выполняться реальные ордера с вашими средствами!")
        logger.warning("⚠️  Убедитесь, что вы протестировали стратегию в симуляционном режиме!")
        
        # В реальном приложении здесь должен быть запрос подтверждения
        # Пока возвращаем False для безопасности
        return False
    
    def execute_arbitrage_trade(self, trade_plan):
        """Выполнение арбитражной сделки"""
        if self.simulation_mode:
            return self._simulate_trade(trade_plan)
        else:
            return self._execute_real_trade(trade_plan)

    def execute_orchestrated_trade(self, legs: list[dict], hedge_leg: dict | None = None, max_loss_usdt: float | None = None, timeout: int | None = None):
        """Запуск оркестратора контингентных цепочек с учётом риск-менеджмента"""

        if not legs:
            logger.warning("⚠️ Невозможно запустить оркестратор без списка ног")
            return None

        balance_snapshot = self.get_balance()
        portfolio_value = float(balance_snapshot.get('total') or balance_snapshot.get('available') or 0)

        safety_plan = {
            'estimated_profit_usdt': max(0.02, (max_loss_usdt or 0) * -1),
            'portfolio_value': portfolio_value,
            'max_loss_usdt': max_loss_usdt or 0,
        }

        if not self.risk_manager.can_execute_trade(safety_plan, portfolio_value=portfolio_value):
            logger.error("❌ Риск-менеджер запретил запуск оркестратора")
            return None

        result = self.contingent_orchestrator.execute_sequence(legs, hedge_leg, max_loss_usdt, timeout)
        if not result:
            return None

        trade_record = {
            'timestamp': datetime.now(),
            'trade_plan': {'legs': legs, 'hedge': hedge_leg, 'max_loss_usdt': max_loss_usdt, 'timeout': timeout},
            'results': result.get('executed'),
            'hedges': result.get('hedges'),
            'status': result.get('status'),
            'total_profit': result.get('estimated_profit', 0),
            'simulated': self.simulation_mode,
            'portfolio_value': portfolio_value,
        }

        self.trade_history.append(trade_record)
        if not self.simulation_mode:
            self.risk_manager.update_after_trade(trade_record)

        logger.info(
            "🏁 Оркестратор завершён со статусом %s (хеджей: %s)",
            trade_record['status'],
            len(trade_record.get('hedges') or []),
        )
        return trade_record

    def get_balance(self, coin='USDT'):
        """Возвращает баланс в зависимости от режима исполнения"""
        if self.simulation_mode:
            return {
                'available': self._simulated_balance_usdt,
                'total': self._simulated_balance_usdt,
                'coin': coin
            }

        return self.client.get_balance(coin)

    def _load_simulated_balance(self):
        """Загружает виртуальный баланс из окружения либо использует дефолт"""
        env_balance = os.getenv('SIMULATION_BALANCE_USDT')

        try:
            return float(env_balance) if env_balance is not None else 100.0
        except (TypeError, ValueError):
            logger.warning("⚠️ Некорректное значение SIMULATION_BALANCE_USDT, используем 100.0 USDT")
            return 100.0

    def _handle_order_event(self, event):
        """Реагирует на обновления по ордерам (исполнения и частичные исполнения)."""

        normalized = {
            'orderId': event.get('orderId'),
            'symbol': event.get('symbol'),
            'status': (event.get('orderStatus') or '').lower(),
            'side': event.get('side'),
            'filled_qty': self._safe_float(event.get('cumExecQty')),
            'remaining_qty': self._safe_float(event.get('leavesQty')),
            'avg_price': self._safe_float(event.get('avgPrice')),
            'execType': event.get('execType'),
            'updatedTime': event.get('updatedTime'),
        }

        self.recent_order_events.appendleft(normalized)
        self._last_execution_hint = normalized

        status = normalized['status']
        if status in ('partiallyfilled', 'partially_filled', 'partial_fill'):
            logger.info(
                "🔔 Частичное исполнение %s: исполнено=%s, осталось=%s",
                normalized['orderId'],
                normalized['filled_qty'],
                normalized['remaining_qty'],
            )
        elif status == 'filled':
            logger.info(
                "✅ Полное исполнение %s по цене %s",
                normalized['orderId'],
                normalized['avg_price'],
            )

    def get_live_order_events(self):
        """Возвращает последние события по ордерам для оперативных решений движка."""

        return list(self.recent_order_events)

    def _safe_float(self, value, default=0.0):
        """Безопасное приведение к float для данных из WebSocket."""

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

    def _emulate_orderbook_fill(self, planned_amount: float, side: str, ticker_snapshot: dict, base_price: float | None):
        """Эмулирует исполнение по лучшему уровню стакана для paper trading."""

        if planned_amount <= 0:
            return 0.0, base_price or 0.0, 0.0

        side_lower = (side or '').lower()
        book_price = self._safe_float(
            ticker_snapshot.get('ask') if side_lower == 'buy' else ticker_snapshot.get('bid')
        )
        book_size = self._safe_float(
            ticker_snapshot.get('ask_size') if side_lower == 'buy' else ticker_snapshot.get('bid_size')
        )

        effective_price = book_price or base_price or 0.0
        if book_size <= 0 or effective_price <= 0:
            return 0.0, effective_price, 0.0

        fill_ratio = min(1.0, book_size / planned_amount if planned_amount else 0)
        impact = getattr(self.config, 'PAPER_BOOK_IMPACT', 0.05) * (1 - fill_ratio)

        if impact > 0 and effective_price > 0:
            if side_lower == 'buy':
                effective_price *= (1 + impact)
            else:
                effective_price *= (1 - impact)

        return fill_ratio, effective_price, book_size
    
    def _simulate_trade(self, trade_plan):
        """Симуляция торговли"""
        logger.info("🧪 SIMULATION MODE: Симуляция исполнения ордеров")

        results = []
        total_profit = 0
        slippage_tolerance = getattr(
            self.config,
            'SIMULATION_SLIPPAGE_TOLERANCE',
            getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
        )
        latency_range = getattr(self.config, 'SIMULATION_LATENCY_RANGE', (0.05, 0.2))
        partial_probability = getattr(self.config, 'SIMULATION_PARTIAL_FILL_PROBABILITY', 0.0)
        reject_probability = getattr(self.config, 'SIMULATION_REJECT_PROBABILITY', 0.0)
        liquidity_buffer = getattr(self.config, 'SIMULATION_LIQUIDITY_BUFFER', 0.0)
        auto_complete_partials = getattr(self.config, 'SIMULATION_AUTO_COMPLETE_PARTIALS', True)
        amount_scale = 1.0
        overall_status = 'completed'
        status_reason = None
        rejected_orders: list[dict] = []

        for step_name, step in trade_plan.items():
            if step_name.startswith('step') or step_name in ['leg1', 'leg2']:
                planned_amount = float(step.get('amount', 0) or 0) * amount_scale
                trade_plan[step_name]['amount'] = planned_amount

                latency = 0.0
                if isinstance(latency_range, (list, tuple)) and len(latency_range) == 2:
                    latency = max(0.0, random.uniform(latency_range[0], latency_range[1]))
                    time.sleep(latency)
                    logger.info(
                        "⏱️ Имитируем сетевую задержку %.3fс перед шагом %s",
                        latency,
                        step_name
                    )

                market_price, ticker_snapshot = self._get_live_price(step['symbol'], step['side'])
                base_price = float(step.get('price') or 0)
                if market_price:
                    base_price = market_price

                forced_rejection = False
                liquidity_shortage = False
                slippage = 0.0
                fill_ratio = 1.0
                execution_price = base_price
                available_liquidity = planned_amount

                if self.paper_trading_mode:
                    fill_ratio, execution_price, available_liquidity = self._emulate_orderbook_fill(
                        planned_amount,
                        step.get('side', ''),
                        ticker_snapshot or {},
                        base_price,
                    )
                    liquidity_shortage = fill_ratio < 1.0
                    forced_rejection = fill_ratio == 0
                    if base_price and execution_price:
                        slippage = abs(execution_price - base_price) / base_price
                    if liquidity_shortage:
                        logger.warning(
                            "⚖️ Paper trading: нехватка ликвидности %.4f против заявки %.4f для %s",
                            available_liquidity,
                            planned_amount,
                            step['symbol'],
                        )
                else:
                    forced_rejection = random.random() < max(0.0, reject_probability)
                    available_liquidity = self._safe_float(
                        ticker_snapshot.get('bidSize') if (step.get('side') or '').lower() == 'sell' else ticker_snapshot.get('askSize'),
                        planned_amount,
                    )
                    if available_liquidity <= 0:
                        available_liquidity = planned_amount * max(0.0, random.uniform(0.5, 1.2))
                    liquidity_shortage = planned_amount > 0 and available_liquidity < planned_amount * (1 - max(0.0, liquidity_buffer))

                    slippage = max(0.0, min(slippage_tolerance, random.uniform(0, slippage_tolerance)))

                    if base_price > 0:
                        if (step.get('side') or '').lower() == 'buy':
                            execution_price = base_price * (1 + slippage)
                        else:
                            execution_price = base_price * (1 - slippage)
                    else:
                        execution_price = base_price

                    logger.info(
                        "📉 Применено проскальзывание %.4f%% для %s: базовая цена %.6f -> %.6f",
                        slippage * 100,
                        step['symbol'],
                        base_price,
                        execution_price if execution_price else base_price
                    )

                    triggered_partial = planned_amount > 0 and random.random() < partial_probability
                    fill_ratio = random.uniform(0.4, 0.95) if triggered_partial else 1.0

                    if liquidity_shortage:
                        fill_ratio = min(fill_ratio, available_liquidity / planned_amount if planned_amount else 0)

                    if forced_rejection:
                        fill_ratio = 0.0

                executed_qty = planned_amount * fill_ratio if planned_amount else 0
                remaining_qty = max(planned_amount - executed_qty, 0)

                simulated_result = {
                    'orderId': f"sim_{int(time.time())}_{step_name}",
                    'orderStatus': 'Rejected' if forced_rejection else 'PartiallyFilled' if fill_ratio < 1 else 'Filled',
                    'symbol': step['symbol'],
                    'side': step['side'],
                    'qty': planned_amount,
                    'price': execution_price or step['price'],
                    'avgPrice': execution_price or step['price'],
                    'cumExecQty': executed_qty,
                    'leavesQty': remaining_qty,
                    'isActive': fill_ratio < 1,
                    'simulated': True,
                    'timestamp': datetime.now().isoformat(),
                    'applied_slippage': slippage,
                    'simulated_latency': latency,
                    'simulatedStatus': 'rejected' if forced_rejection else 'partial' if fill_ratio < 1 else 'filled',
                }
                if forced_rejection:
                    status_reason = "Вероятностный отказ симуляции"
                    overall_status = 'rejected'
                    rejected_orders.append(simulated_result)
                    results.append(simulated_result)
                    logger.error(
                        "❌ Симулированный отказ: %s %.6f %s @ %.2f (причина: %s)",
                        step['side'],
                        planned_amount,
                        step['symbol'],
                        execution_price or step['price'],
                        status_reason,
                    )
                    self._cancel_previous_orders(results)
                    break

                if liquidity_shortage and fill_ratio == 0:
                    status_reason = "Недостаточная ликвидность в симулированном стакане"
                    overall_status = 'rejected'
                    rejected_orders.append(simulated_result)
                    results.append(simulated_result)
                    logger.error(
                        "❌ Симулированный отказ по ликвидности: %s %.6f %s (доступно %.6f)",
                        step['side'],
                        planned_amount,
                        step['symbol'],
                        available_liquidity,
                    )
                    self._cancel_previous_orders(results)
                    break

                results.append(simulated_result)
                if liquidity_shortage and fill_ratio < 1:
                    status_reason = "Частичное исполнение из-за нехватки ликвидности"
                    overall_status = 'partial'
                    logger.warning(
                        "⚠️ Симуляция: нехватка ликвидности, исполнено %.6f из %.6f для %s",
                        executed_qty,
                        planned_amount,
                        step['symbol'],
                    )
                elif fill_ratio < 1:
                    overall_status = 'partial'
                    logger.info(
                        "ℹ️ Симуляция: частичное исполнение %.6f из %.6f для %s",
                        executed_qty,
                        planned_amount,
                        step['symbol'],
                    )
                else:
                    logger.info(
                        "✅ Симулированный ордер: %s %.6f %s @ %.2f",
                        step['side'],
                        planned_amount,
                        step['symbol'],
                        execution_price or step['price']
                    )

                fill_ratio_effective = self._handle_partial_fill(
                    trade_plan[step_name],
                    simulated_result,
                    slippage_tolerance,
                    market_price,
                )
                simulated_result['fillRatio'] = fill_ratio_effective

                if (
                    fill_ratio < 1
                    and auto_complete_partials
                    and remaining_qty > 0
                    and not liquidity_shortage
                ):
                    completion_latency = max(
                        0.0,
                        random.uniform(latency_range[0], latency_range[1])
                    ) if isinstance(latency_range, (list, tuple)) and len(latency_range) == 2 else 0.0
                    if completion_latency:
                        time.sleep(completion_latency)
                        logger.info(
                            "⏱️ Дособираем остаток после частичного исполнения через %.3fс",
                            completion_latency,
                        )

                    completion_result = {
                        **simulated_result,
                        'orderStatus': 'Filled',
                        'cumExecQty': planned_amount,
                        'leavesQty': 0.0,
                        'isActive': False,
                        'fillRatio': 1.0,
                        'simulated_latency': latency + completion_latency,
                    }
                    results.append(completion_result)
                    fill_ratio_effective = 1.0
                    logger.info(
                        "🔄 Остаток %.6f %s добран, ордер закрыт",
                        remaining_qty,
                        step['symbol'],
                    )

                amount_scale *= fill_ratio_effective if fill_ratio_effective > 0 else 1.0

        # Расчет прибыли для симуляции
        if 'estimated_profit_usdt' in trade_plan:
            total_profit = trade_plan['estimated_profit_usdt']

        trade_record = {
            'timestamp': datetime.now(),
            'trade_plan': trade_plan,
            'results': results,
            'total_profit': total_profit,
            'simulated': True,
            'status': overall_status,
            'rejected': rejected_orders,
            'status_reason': status_reason,
        }

        self.trade_history.append(trade_record)
        if overall_status == 'rejected':
            logger.error(
                "💔 Симуляция завершена отказом: прибыль %.4f USDT, причина: %s",
                total_profit,
                status_reason or 'неизвестно'
            )
        elif overall_status == 'partial':
            logger.warning(
                "⚠️ Симуляция завершена частично: прибыль %.4f USDT, есть незакрытые объёмы",
                total_profit,
            )
        else:
            logger.info(f"💰 SIMULATED PROFIT: {total_profit:.4f} USDT")

        return trade_record

    def _get_live_price(self, symbol: str, side: str) -> tuple[float | None, dict]:
        """Возвращает актуальную цену из WebSocket/REST и источник данных."""

        market_price = None
        ticker_snapshot = {}

        if self.client.ws_manager:
            cached, missing = self.client.ws_manager.get_cached_tickers([symbol])
            ticker_snapshot = cached.get(symbol) or {}
            if not missing and ticker_snapshot:
                market_price = ticker_snapshot.get('ask') if side.lower() == 'buy' else ticker_snapshot.get('bid')

        if market_price is None:
            fresh = self.client.get_tickers([symbol]) or {}
            ticker_snapshot = fresh.get(symbol) or ticker_snapshot
            if ticker_snapshot:
                market_price = ticker_snapshot.get('ask') if side.lower() == 'buy' else ticker_snapshot.get('bid')

        if market_price:
            market_price = float(market_price)
        else:
            logger.warning("⚠️ Не удалось получить актуальную цену для %s", symbol)

        return market_price, ticker_snapshot

    def _ensure_price_alignment(self, planned_price: float | None, market_price: float | None, tolerance: float) -> bool:
        """Проверяет, что расчётная цена не отклоняется от рыночной выше допуска."""

        if not market_price or not planned_price:
            return True

        deviation = abs(planned_price - market_price) / planned_price if planned_price else 0
        if deviation > tolerance:
            logger.warning(
                "❌ Отклонение цены %.4f%% превышает допуск %.4f%%", deviation * 100, tolerance * 100
            )
            return False

        return True

    def _calculate_limit_price(self, side: str, market_price: float, tolerance: float) -> float:
        """Формирует лимитную цену с учётом направления и допуска проскальзывания."""

        if side.lower() == 'buy':
            return market_price * (1 + tolerance)
        return market_price * (1 - tolerance)

    def _handle_partial_fill(self, step: dict, order_result: dict, tolerance: float, market_price: float | None) -> float:
        """Обрабатывает частичное исполнение и возвращает коэффициент масштабирования для следующих шагов."""

        requested_qty = float(step.get('amount', 0) or 0)
        executed_qty = self._safe_float(order_result.get('cumExecQty'))
        avg_price = self._safe_float(order_result.get('avgPrice')) or step.get('price')

        if requested_qty <= 0 or executed_qty is None or executed_qty <= 0:
            return 1.0

        fill_ratio = min(1.0, executed_qty / requested_qty)

        if not market_price and getattr(self.client, 'ws_manager', None):
            cached, _ = self.client.ws_manager.get_cached_tickers([step['symbol']])
            ticker = cached.get(step['symbol']) if cached else None
            if ticker:
                market_price = ticker.get('ask') if (step.get('side') or '').lower() == 'buy' else ticker.get('bid')

        if market_price and avg_price:
            deviation = abs(avg_price - market_price) / market_price
            if deviation > tolerance:
                logger.error(
                    "🔥 Фактическая цена исполнения отклоняется на %.4f%% (допуск %.4f%%) по данным WebSocket",
                    deviation * 100,
                    tolerance * 100,
                )
                return -1.0

        if fill_ratio < 1.0:
            logger.warning(
                "⚠️ Частичное исполнение: исполнено %.4f из %.4f (%.2f%%)",
                executed_qty,
                requested_qty,
                fill_ratio * 100,
            )

        return fill_ratio
    
    def _execute_real_trade(self, trade_plan):
        """Реальное исполнение торговли"""
        logger.warning("🔥 REAL MODE: Выполнение реальных ордеров")

        balance_snapshot = self.get_balance()
        portfolio_value = float(balance_snapshot.get('total') or balance_snapshot.get('available') or 0)
        trade_plan = dict(trade_plan)
        trade_plan.setdefault('portfolio_value', portfolio_value)

        if not self.risk_manager.can_execute_trade(trade_plan, portfolio_value=portfolio_value):
            logger.error("❌ Риск-менеджер запретил выполнение сделки")
            return None

        try:
            results = []
            total_profit = 0
            slippage_tolerance = getattr(self.config, 'SLIPPAGE_PROFIT_BUFFER', 0.02)
            amount_scale = 1.0
            start_time = time.time()
            max_exec_time = getattr(self.config, 'MAX_TRIANGLE_EXECUTION_TIME', 30)

            # Выполняем ордера последовательно
            for step_name, step in trade_plan.items():
                if step_name.startswith('step') or step_name in ['leg1', 'leg2']:
                    # Пересчитываем объём с учётом предыдущих частичных исполнений
                    planned_amount = float(step.get('amount', 0) or 0) * amount_scale
                    trade_plan[step_name]['amount'] = planned_amount

                    if time.time() - start_time > max_exec_time:
                        logger.error("⏳ Тайм-аут исполнения сделки на шаге %s", step_name)
                        self._cancel_previous_orders(results)

                        timeout_record = {
                            'timestamp': datetime.now(),
                            'trade_plan': trade_plan,
                            'results': results,
                            'total_profit': -abs(trade_plan.get('estimated_profit_usdt', 0) or 0.01),
                            'status': 'timeout',
                            'simulated': False,
                            'portfolio_value': portfolio_value,
                        }

                        self.trade_history.append(timeout_record)
                        self.risk_manager.update_after_trade(timeout_record)
                        return None

                    market_price, _ = self._get_live_price(step['symbol'], step['side'])

                    if not self._ensure_price_alignment(step.get('price'), market_price, slippage_tolerance):
                        logger.error("🚫 Цепочка отменена из-за отклонения котировок на шаге %s", step_name)
                        self._cancel_previous_orders(results)
                        return None

                    if step.get('type', 'Limit').lower() == 'limit' and market_price:
                        new_limit = self._calculate_limit_price(step['side'], market_price, slippage_tolerance)
                        trade_plan[step_name]['price'] = new_limit
                        logger.info(
                            "🔧 Обновлена лимитная цена для %s: %.6f (рынок %.6f)",
                            step_name,
                            new_limit,
                            market_price,
                        )

                    order_result = self.client.place_order(
                        symbol=step['symbol'],
                        side=step['side'],
                        qty=planned_amount,
                        price=trade_plan[step_name].get('price'),
                        order_type=step.get('type', 'Limit')
                    )

                    if order_result:
                        results.append(order_result)
                        logger.info(
                            f"✅ REAL ORDER: {step['side']} {planned_amount:.6f} {step['symbol']} @ {trade_plan[step_name].get('price', '_MARKET_')}"
                        )

                        fill_ratio = self._handle_partial_fill(
                            trade_plan[step_name],
                            order_result,
                            slippage_tolerance,
                            market_price,
                        )

                        if fill_ratio < 0:
                            self._cancel_previous_orders(results)
                            return None

                        if 0 < fill_ratio < 1:
                            amount_scale *= fill_ratio

                    else:
                        logger.error(f"❌ FAILED ORDER: {step['side']} {planned_amount:.6f} {step['symbol']}")
                        # Отменяем предыдущие ордера при ошибке
                        self._cancel_previous_orders(results)
                        return None
            
            # Расчет реальной прибыли
            if results:
                total_profit = self._calculate_real_profit(results, trade_plan)
            
            trade_record = {
                'timestamp': datetime.now(),
                'trade_plan': trade_plan,
                'results': results,
                'total_profit': total_profit,
                'simulated': False,
                'portfolio_value': portfolio_value,
            }
            
            self.trade_history.append(trade_record)
            self.risk_manager.update_after_trade(trade_record)
            
            logger.info(f"💰 REAL PROFIT: {total_profit:.4f} USDT")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"🔥 CRITICAL ERROR during real trade execution: {str(e)}", exc_info=True)
            return None
    
    def _cancel_previous_orders(self, results):
        """Отмена предыдущих ордеров при ошибке"""
        for order in results:
            if 'orderId' in order:
                if self.simulation_mode or order.get('simulated'):
                    logger.warning(
                        "🛑 Отмена симулированного ордера %s по %s", order['orderId'], order.get('symbol')
                    )
                else:
                    self.client.cancel_order(order['orderId'], order['symbol'])
    
    def _calculate_real_profit(self, results, trade_plan):
        """Расчет реальной прибыли на основе исполненных ордеров"""
        try:
            initial_amount = float(trade_plan.get('initial_amount', 0))
            if initial_amount <= 0:
                logger.warning("⚠️ Стартовый капитал не задан или некорректен, расчёт прибыли невозможен")
                return 0

            if any(order.get('simulated') for order in results):
                return float(trade_plan.get('estimated_profit_usdt', 0))

            base_currency = trade_plan.get('base_currency', 'USDT')
            fee_rate = getattr(self.config, 'TRADING_FEE', 0)

            # Инициализируем балансы: стартуем только с базовой валюты
            balances = {base_currency: initial_amount}

            for order in results:
                symbol = order.get('symbol') or ''
                side = (order.get('side') or '').lower()
                price = float(order.get('avgPrice') or order.get('price') or 0)
                quantity = float(order.get('cumExecQty') or order.get('qty') or 0)

                base, quote = self._split_symbol(symbol)
                if not base or not quote or price <= 0 or quantity <= 0:
                    logger.warning("⚠️ Пропуск ордера из-за некорректных данных при расчёте прибыли")
                    continue

                if side == 'buy':
                    # Покупаем базовый актив за котируемую валюту, комиссия уменьшает получаемое количество
                    cost = price * quantity
                    balances[quote] = balances.get(quote, 0) - cost
                    received = quantity * (1 - fee_rate)
                    balances[base] = balances.get(base, 0) + received
                elif side == 'sell':
                    # Продаём базовый актив за котируемую валюту, комиссия уменьшает итоговую выручку
                    balances[base] = balances.get(base, 0) - quantity
                    proceeds = price * quantity * (1 - fee_rate)
                    balances[quote] = balances.get(quote, 0) + proceeds
                else:
                    logger.warning("⚠️ Неизвестная сторона сделки при расчёте прибыли")

            real_profit = balances.get(base_currency, 0) - initial_amount
            trade_plan['estimated_profit_usdt'] = real_profit
            return real_profit
        except Exception as e:
            logger.error(f"Ошибка расчета реальной прибыли: {str(e)}")
            return 0

    def _split_symbol(self, symbol):
        """Разделяет тикер на базовую и котируемую валюты"""
        for quote in sorted(self.config.KNOWN_QUOTES, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        return None, None
    
    def get_performance_stats(self):
        """Получение статистики производительности"""
        if not self.trade_history:
            return {}
        
        total_trades = len(self.trade_history)
        successful_trades = sum(1 for trade in self.trade_history if trade.get('total_profit', 0) > 0)
        total_profit = sum(trade.get('total_profit', 0) for trade in self.trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        
        runtime = datetime.now() - min(trade['timestamp'] for trade in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'runtime': str(runtime).split('.')[0],
            'simulation_mode': self.simulation_mode,
            'real_mode': self.is_real_mode
        }
    
    def export_trade_history(self, filename=None):
        """Экспорт истории сделок"""
        import csv
        import json

        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        def _convert_datetime_values(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {k: _convert_datetime_values(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert_datetime_values(item) for item in value]
            return value

        def _prepare_trade_plan(trade_plan):
            if not trade_plan:
                return {}
            prepared = _convert_datetime_values(trade_plan)
            return prepared if isinstance(prepared, dict) else {'value': prepared}

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'symbol', 'side', 'amount', 'price', 'profit', 'simulated', 'trade_details']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                for trade in self.trade_history:
                    results = trade.get('results') or []

                    if not results:
                        details = trade.get('details', {})
                        symbols = details.get('symbols') or trade.get('symbol') or ''
                        if isinstance(symbols, (list, tuple)):
                            symbols = ','.join(symbols)
                        results = [{
                            'symbol': symbols,
                            'side': details.get('direction', trade.get('direction', '')),
                            'qty': details.get('initial_amount', 0),
                            'price': details.get('price', 0)
                        }]

                    for result in results:
                        timestamp = trade['timestamp']
                        if hasattr(timestamp, 'strftime'):
                            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            timestamp_str = str(timestamp)

                        writer.writerow({
                            'timestamp': timestamp_str,
                            'symbol': result.get('symbol', ''),
                            'side': result.get('side', ''),
                            'amount': result.get('qty', result.get('cumExecQty', 0)),
                            'price': result.get('avgPrice', result.get('price', 0)),
                            'profit': trade.get('total_profit', 0) if result == results[-1] else 0,
                            'simulated': trade.get('simulated', False),
                            'trade_details': json.dumps(
                                _prepare_trade_plan(trade.get('trade_plan', {})),
                                default=str
                            )
                        })
            
            logger.info(f"✅ Trade history exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error exporting trade history: {str(e)}")
            return None


__all__ = ["RealTradingExecutor"]
