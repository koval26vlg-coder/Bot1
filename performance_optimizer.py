"""Утилита для приоритизации и быстрой фильтрации треугольников по ликвидности."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional


class PerformanceOptimizer:
    """Строит оптимизированный список треугольников с учетом ликвидности и приоритетов."""

    def __init__(self, config):
        # Конфигурация передается снаружи, чтобы использовать существующие настройки.
        self.config = config
        # Базовые валюты с приоритетом для арбитража.
        self._preferred_bases = {"USDT", "USDC"}
        # Ключевые активы, которые стоит обрабатывать первыми.
        self._core_assets = {"BTC", "ETH", "BNB"}

    def get_optimized_triangles(
        self,
        tickers: Optional[Dict[str, Dict[str, float]]] = None,
        max_count: int = 50,
    ) -> List[Dict]:
        """Возвращает отсортированный и при необходимости отфильтрованный список треугольников."""

        candidates = list(self.config.TRIANGULAR_PAIRS)
        prioritized = sorted(candidates, key=self._triangle_sort_key)

        if tickers:
            prioritized = self.parallel_check_liquidity(prioritized, tickers)

        return prioritized[:max_count]

    def _triangle_sort_key(self, triangle: Dict) -> tuple:
        """Формирует ключ сортировки с учетом базовой валюты, активов и приоритета."""

        base_currency = triangle.get("base_currency", "")
        base_score = 0 if base_currency in self._preferred_bases else 1

        # Чем больше ключевых активов в треугольнике, тем выше его приоритет.
        core_hits = sum(
            1
            for leg in triangle.get("legs", [])
            for asset in self._core_assets
            if leg.startswith(asset) or leg.endswith(asset)
        )
        asset_score = -core_hits  # Большее количество совпадений уменьшает ключ сортировки.

        priority_score = triangle.get("priority", 999)

        return (base_score, asset_score, priority_score, triangle.get("name", ""))

    def _quick_liquidity_check(
        self,
        triangle: Dict,
        tickers: Dict[str, Dict[str, float]],
        max_spread_percent: float = 5.0,
    ) -> bool:
        """Быстрая проверка ликвидности по верхним котировкам и спреду."""

        for symbol in triangle.get("legs", []):
            ticker = tickers.get(symbol)
            if not ticker:
                return False

            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if not bid or not ask or bid <= 0 or ask <= 0:
                return False

            spread = ((ask - bid) / bid) * 100
            if spread > max_spread_percent:
                return False

        return True

    def parallel_check_liquidity(
        self,
        triangles: Iterable[Dict],
        tickers: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        """Параллельная фильтрация треугольников по результатам быстрой проверки ликвидности."""

        triangles_list = list(triangles)
        filtered: List[Dict] = []
        # Используем ограниченный пул потоков, чтобы не перегружать систему.
        max_workers = min(8, len(triangles_list) or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_triangle = {
                executor.submit(self._quick_liquidity_check, triangle, tickers): triangle
                for triangle in triangles_list
            }
            for future in future_to_triangle:
                if future.result():
                    filtered.append(future_to_triangle[future])

        return filtered
