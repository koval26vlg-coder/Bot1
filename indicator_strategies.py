"""Простые индикаторные стратегии для определения рыночного контекста."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from utils.math_stats import mean



@dataclass
class StrategyResult:
    """Результат работы стратегии."""

    name: str
    signal: str
    score: float
    confidence: float


class StrategyManager:
    """Менеджер стратегий, который выбирает лучший результат по скору."""

    def __init__(self, config):
        self.config = config
        self.active_strategy = 'volatility_adaptive'
        self._strategies: Dict[str, Callable[[Sequence[dict], Dict[str, float]], Optional[StrategyResult]]] = {
            'volatility_adaptive': self._volatility_adaptive_strategy,
            'momentum_bias': self._momentum_bias_strategy,
        }

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
    ) -> Optional[StrategyResult]:
        """Корректирует агрессивность торговли в зависимости от волатильности."""

        volatility = context.get('volatility', 0.0)
        liquidity = context.get('liquidity', 0.0)

        if volatility > 1.5:
            signal = 'reduce_risk'
            score = max(0.1, 2.5 - volatility)
        elif volatility < 0.3 and liquidity > 0:
            signal = 'increase_risk'
            score = 1.5 + min(0.5, liquidity / 100000)
        else:
            signal = 'neutral'
            score = 1.0

        confidence = float(min(1.0, abs(volatility - 1.0) / 2))
        return StrategyResult(
            name='volatility_adaptive',
            signal=signal,
            score=float(score),
            confidence=confidence,
        )

    def _momentum_bias_strategy(
        self,
        market_df: Sequence[dict],
        context: Dict[str, float]
    ) -> Optional[StrategyResult]:
        """Оценивает импульс рынка на основе средних значений."""

        if len(market_df) < 30:
            return None

        closes = [row['close'] for row in market_df if row.get('close') is not None]
        if len(closes) < 30:
            return None

        short_ma = mean(closes[-10:])
        long_ma = mean(closes[-30:])

        if long_ma == 0:
            return None

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


__all__ = ['StrategyManager', 'StrategyResult']