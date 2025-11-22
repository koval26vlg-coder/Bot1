"""Простые индикаторные стратегии для определения рыночного контекста."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

# Используем функции из реального локального модуля math_stats вместо utils.math_stats
from math_stats import mean



@dataclass
class StrategyResult:
    """Результат работы стратегии."""

    name: str
    signal: str
    score: float
    confidence: float
    meta: Optional[Dict[str, float]] = None


class StrategyManager:
    """Менеджер стратегий, который выбирает лучший результат по скору."""

    def __init__(self, config):
        self.config = config
        self.active_strategy = 'volatility_adaptive'
        self._strategies: Dict[str, Callable[[Sequence[dict], Dict[str, float]], StrategyResult]] = {
            'volatility_adaptive': self._volatility_adaptive_strategy,
            'momentum_bias': self._momentum_bias_strategy,
            'multi_indicator': self._multi_indicator_strategy,
            'orderbook_balance': self._orderbook_balance_strategy,
            'pattern_recognition': self._pattern_recognition_strategy,
        }

    def update_config(self, config):
        """Обновляет конфигурацию без пересоздания менеджера."""
        self.config = config

    def evaluate(self, market_df: Sequence[dict], context: Dict[str, float]) -> Optional[StrategyResult]:
        best_result: Optional[StrategyResult] = None
        for name, strategy in self._strategies.items():
            try:
                result = strategy(market_df, context)
            except Exception:
                result = None

            if result and (best_result is None or result.score > best_result.score):
                best_result = result

        if best_result:
            self.active_strategy = best_result.name

        return best_result

    def get_active_strategy_name(self) -> str:
        return self.active_strategy

    def get_strategy_snapshot(self) -> Dict[str, Dict[str, str]]:
        snapshot = {}
        for name in self._strategies:
            snapshot[name] = {
                'name': name,
                'description': self._strategies[name].__doc__ or ''
            }
        return snapshot

    def _volatility_adaptive_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Корректирует агрессивность торговли в зависимости от волатильности."""

        volatility = context.get('volatility', 0.0)
        liquidity = context.get('liquidity', 0.0)
        spread = context.get('average_spread_percent', 0.0)

        if spread > 1.0:
            signal = 'reduce_risk'
            score = max(0.1, 1.8 - spread)
        elif volatility > 1.5:
            signal = 'reduce_risk'
            score = max(0.1, 2.5 - volatility)
        elif volatility < 0.3 and liquidity > 0:
            signal = 'increase_risk'
            score = 1.5 + min(0.5, liquidity / 100000)
        else:
            signal = 'neutral'
            score = 1.0

        confidence_anchor = abs(volatility - 1.0) / 2
        confidence_spread = min(1.0, max(0.0, (1.0 - spread)))
        confidence = float(min(1.0, (confidence_anchor + confidence_spread) / 2))
        return StrategyResult(
            name='volatility_adaptive',
            signal=signal,
            score=float(score),
            confidence=confidence,
        )

    def _orderbook_balance_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Использует средний спред и дисбаланс стакана для оценки риска."""

        spread = context.get('average_spread_percent', 0.0)
        imbalance = context.get('orderbook_imbalance', 0.0)

        if spread > 1.2:
            signal = 'reduce_risk'
            base_score = max(0.1, 2 - spread)
        elif abs(imbalance) > 0.25 and spread < 0.6:
            signal = 'long_bias' if imbalance > 0 else 'short_bias'
            base_score = 1.5 + abs(imbalance)
        elif spread < 0.25:
            signal = 'increase_risk'
            base_score = 1.2 + (0.3 - spread)
        else:
            signal = 'neutral'
            base_score = 1.0

        confidence = float(min(1.0, max(0.0, (1 - spread)) + abs(imbalance)))
        return StrategyResult(
            name='orderbook_balance',
            signal=signal,
            score=float(base_score),
            confidence=confidence,
            meta={
                'average_spread_percent': float(spread),
                'orderbook_imbalance': float(imbalance),
            },
        )

    def _momentum_bias_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Оценивает импульс рынка и сама контролирует достаточность выборки."""

        closes_by_symbol = defaultdict(list)
        for row in market_df:
            close_value = row.get('close')
            if close_value is None:
                continue
            closes_by_symbol[row.get('symbol', 'default')].append(close_value)

        closes: Sequence[float] = []
        if closes_by_symbol:
            closes = max(closes_by_symbol.values(), key=len)

        min_required = 12
        if len(closes) < min_required:
            readiness = len(closes) / min_required if min_required else 0
            return StrategyResult(
                name='momentum_bias',
                signal='wait_for_data',
                score=float(readiness * 0.5),
                confidence=float(min(0.3, readiness / 2)),
            )

        short_window = min(10, len(closes))
        long_window = min(25, len(closes))

        short_ma = mean(closes[-short_window:])
        long_ma = mean(closes[-long_window:])

        if long_ma == 0:
            return StrategyResult(
                name='momentum_bias',
                signal='neutral',
                score=0.0,
                confidence=0.0,
            )

        momentum = float(((short_ma - long_ma) / long_ma) * 100)

        if momentum > 0.1:
            signal = 'long_bias'
        elif momentum < -0.1:
            signal = 'short_bias'
        else:
            signal = 'neutral'

        score = abs(momentum)
        confidence = float(min(1.0, abs(momentum) / 5))

        return StrategyResult(
            name='momentum_bias',
            signal=signal,
            score=score,
            confidence=confidence,
        )

    def _multi_indicator_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> StrategyResult:
        """Комбинирует RSI, EMA и ATR для оценки импульса и риска."""

        rows_by_symbol = defaultdict(list)
        for row in market_df:
            symbol = row.get('symbol', 'default')
            rows_by_symbol[symbol].append(row)

        if not rows_by_symbol:
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        # Берем самый заполненный набор, чтобы избежать смены символов на коротких сериях
        symbol_rows = max(rows_by_symbol.values(), key=len)
        closes = [float(r['close']) for r in symbol_rows if r.get('close') is not None]
        highs = [float(r['high']) for r in symbol_rows if r.get('high') is not None]
        lows = [float(r['low']) for r in symbol_rows if r.get('low') is not None]
        opens = [float(r['open']) for r in symbol_rows if r.get('open') is not None]

        min_required = 15
        if (
            len(closes) < min_required
            or len(highs) < min_required
            or len(lows) < min_required
            or len(opens) < min_required
        ):
            readiness = len(closes) / min_required if min_required else 0
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=float(readiness * 0.4),
                confidence=float(min(0.25, readiness / 3)),
                meta={},
            )

        rsi = self._calculate_rsi(closes)
        short_ema = self._calculate_ema(closes, period=9)
        long_ema = self._calculate_ema(closes, period=21)
        atr = self._calculate_atr(highs, lows, closes)

        if long_ema is None or short_ema is None or rsi is None or atr is None:
            return StrategyResult(
                name='multi_indicator',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        ema_diff = short_ema - long_ema
        trend_strength = abs(ema_diff) / long_ema * 100 if long_ema else 0.0
        momentum_component = abs(rsi - 50) / 50

        if rsi > 60 and short_ema > long_ema:
            signal = 'long'
        elif rsi < 40 and short_ema < long_ema:
            signal = 'short'
        else:
            signal = 'flat'

        score = float(trend_strength + momentum_component * 2)
        confidence = float(min(1.0, (trend_strength / 10) + (momentum_component / 2)))

        pattern = self._detect_candlestick_pattern(opens, highs, lows, closes)
        if pattern['bias'] == 'bullish':
            # Поддерживаем бычьи паттерны повышением скора и уверенностью
            score += pattern['strength']
            confidence = float(min(1.0, confidence + pattern['strength'] / 2))
            if signal == 'flat':
                signal = 'long_bias'
            elif signal == 'short':
                signal = 'flat'
        elif pattern['bias'] == 'bearish':
            # Медвежьи паттерны уменьшают уверенность и агрессивность
            score += pattern['strength']
            confidence = float(max(0.0, confidence - pattern['strength'] / 3))
            if signal == 'flat':
                signal = 'short_bias'
            elif signal == 'long':
                signal = 'flat'

        meta = {
            'rsi': float(rsi),
            'short_ema': float(short_ema),
            'long_ema': float(long_ema),
            'atr': float(atr),
            'atr_percent': float((atr / closes[-1]) * 100) if closes and closes[-1] else 0.0,
            'trend_strength': float(trend_strength),
            'pattern_name': pattern['name'],
            'pattern_bias': pattern['bias'],
            'pattern_strength': pattern['strength'],
        }

        return StrategyResult(
            name='multi_indicator',
            signal=signal,
            score=score,
            confidence=confidence,
            meta=meta,
        )

    def _calculate_rsi(self, closes: Sequence[float], period: int = 14) -> Optional[float]:
        """Простой расчёт RSI без сторонних библиотек."""

        if len(closes) < period + 1:
            return None

        gains = []
        losses = []
        for prev, curr in zip(closes[-(period + 1):-1], closes[-period:]):
            change = curr - prev
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = mean(gains) if gains else 0.0
        avg_loss = mean(losses) if losses else 0.0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_ema(self, closes: Sequence[float], period: int) -> Optional[float]:
        """Экспоненциальное среднее с инициализацией от простого среднего."""

        if len(closes) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = mean(closes[:period])
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)

    def _calculate_atr(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> Optional[float]:
        """ATR на основе истинного диапазона."""

        if not (len(highs) == len(lows) == len(closes)):
            return None
        if len(closes) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        recent_tr = true_ranges[-period:]
        if not recent_tr:
            return None
        return mean(recent_tr)

    def _detect_candlestick_pattern(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> Dict[str, str | float]:
        """Определяем базовые свечные паттерны без сторонних библиотек."""

        if min(len(opens), len(highs), len(lows), len(closes)) < 2:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = max(h1 - l1, 1e-9)
        range2 = max(h2 - l2, 1e-9)

        patterns = []

        # Бычье поглощение
        if c1 > o1 and c2 < o2 and o1 <= c2 and c1 >= o2:
            strength = min(0.6, (body1 + body2) / max(body2, 1e-9) * 0.1)
            patterns.append(('bullish_engulfing', 'bullish', strength))

        # Медвежье поглощение
        if c1 < o1 and c2 > o2 and o1 >= c2 and c1 <= o2:
            strength = min(0.6, (body1 + body2) / max(body2, 1e-9) * 0.1)
            patterns.append(('bearish_engulfing', 'bearish', strength))

        # Молот
        lower_shadow = o1 - l1 if o1 >= c1 else c1 - l1
        upper_shadow = h1 - max(o1, c1)
        if body1 <= range1 * 0.35 and lower_shadow >= body1 * 2 and upper_shadow <= body1:
            strength = min(0.4, lower_shadow / range1)
            patterns.append(('hammer', 'bullish', strength))

        # Падающая звезда
        if body1 <= range1 * 0.35 and upper_shadow >= body1 * 2 and lower_shadow <= body1:
            strength = min(0.4, upper_shadow / range1)
            patterns.append(('shooting_star', 'bearish', strength))

        if not patterns:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        # Берём самый сильный паттерн
        name, bias, strength = max(patterns, key=lambda item: item[2])
        return {'name': name, 'bias': bias, 'strength': float(strength)}

    def _pattern_recognition_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float],
    ) -> StrategyResult:
        """Распознаёт свечные паттерны и подтверждает их объёмами."""

        rows_by_symbol = defaultdict(list)
        for row in market_df:
            symbol = row.get('symbol', 'default')
            rows_by_symbol[symbol].append(row)

        if not rows_by_symbol:
            return StrategyResult(
                name='pattern_recognition',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        symbol_rows = max(rows_by_symbol.values(), key=len)
        opens = [float(r['open']) for r in symbol_rows if r.get('open') is not None]
        highs = [float(r['high']) for r in symbol_rows if r.get('high') is not None]
        lows = [float(r['low']) for r in symbol_rows if r.get('low') is not None]
        closes = [float(r['close']) for r in symbol_rows if r.get('close') is not None]
        volumes = [float(r['volume']) for r in symbol_rows if r.get('volume') is not None]

        if min(len(opens), len(highs), len(lows), len(closes), len(volumes)) < 2:
            return StrategyResult(
                name='pattern_recognition',
                signal='wait_for_data',
                score=0.0,
                confidence=0.0,
                meta={},
            )

        pattern = self._detect_pattern_with_flag(opens, highs, lows, closes)

        avg_volume = mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        volume_ratio = float(volumes[-1] / avg_volume) if avg_volume else 1.0
        volume_confirmation = float(min(1.0, max(0.0, volume_ratio - 0.5)))

        signal_map = {
            'bullish': 'long',
            'bearish': 'short',
            'neutral': 'neutral',
        }
        signal = signal_map.get(pattern['bias'], 'neutral')

        pattern_strength = pattern['strength']
        if pattern_strength < 0.1:
            signal = 'neutral'

        score = float(1 + pattern_strength * 2 + volume_confirmation)
        confidence = float(min(1.0, pattern_strength * 0.7 + volume_confirmation * 0.5))

        meta = {
            'pattern_name': pattern['name'],
            'pattern_bias': pattern['bias'],
            'pattern_strength': float(pattern_strength),
            'volume_ratio': volume_ratio,
        }

        return StrategyResult(
            name='pattern_recognition',
            signal=signal,
            score=score,
            confidence=confidence,
            meta=meta,
        )

    def _detect_pattern_with_flag(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> Dict[str, str | float]:
        """Расширенный поиск паттернов, включая флаги."""

        base_pattern = self._detect_candlestick_pattern(opens, highs, lows, closes)
        candidates = [base_pattern] if base_pattern['name'] != 'none' else []

        if len(closes) >= 4:
            range_values = [h - l for h, l in zip(highs, lows)]
            impulse_up = closes[-3] > closes[-4] and closes[-2] >= closes[-3]
            impulse_down = closes[-3] < closes[-4] and closes[-2] <= closes[-3]
            compact_consolidation = max(range_values[-2:]) < min(range_values[-4:-2]) * 0.8

            if impulse_up and compact_consolidation:
                strength = min(0.5, (closes[-2] - closes[-4]) / max(closes[-4], 1e-9) * 0.3)
                candidates.append({'name': 'bull_flag', 'bias': 'bullish', 'strength': float(strength)})

            if impulse_down and compact_consolidation:
                strength = min(0.5, (closes[-4] - closes[-2]) / max(closes[-4], 1e-9) * 0.3)
                candidates.append({'name': 'bear_flag', 'bias': 'bearish', 'strength': float(strength)})

        if not candidates:
            return {'name': 'none', 'bias': 'neutral', 'strength': 0.0}

        return max(candidates, key=lambda p: p['strength'])


__all__ = ['StrategyManager', 'StrategyResult']
