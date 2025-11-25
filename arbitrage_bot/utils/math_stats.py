"""Вспомогательные функции для статистических расчётов без numpy/pandas."""

from statistics import fmean
from typing import Iterable, Optional


def mean(values: Iterable[float]) -> float:
    """Вычисляет среднее значение с защитой от пустых коллекций."""
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return 0.0
    return float(fmean(filtered))


def rolling_mean(values: Iterable[float], window: int, min_periods: Optional[int] = None) -> Optional[float]:
    """Возвращает среднее для последних *window* значений либо None, если данных мало."""
    if window <= 0:
        raise ValueError("Размер окна должен быть положительным")

    min_required = min_periods or window
    data = [float(v) for v in values if v is not None]
    if len(data) < min_required:
        return None

    recent = data[-window:]
    if not recent:
        return None
    return float(fmean(recent))


__all__ = ["mean", "rolling_mean"]
