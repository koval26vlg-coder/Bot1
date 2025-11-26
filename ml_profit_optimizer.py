"""Легковесный ML-оптимизатор порога прибыли."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

logger = logging.getLogger(__name__)


class MLProfitOptimizer:
    """Упрощённый оптимизатор порога прибыли на базе RandomForestRegressor."""

    FEATURE_ORDER = [
        "overall_volatility",
        "average_spread_percent",
        "orderbook_imbalance",
        "empty_cycles",
        "market_regime",
    ]

    def __init__(self, model_path: str | Path, fallback_threshold: float = 0.0):
        self.model_path = Path(model_path)
        self.fallback_threshold = float(fallback_threshold)
        self.model = None
        self._sklearn_modules: SimpleNamespace | None = None
        self._sklearn_import_error: Exception | None = None

        self._ensure_sklearn()
        if self.ml_supported:
            self._load_model()
        else:
            logger.warning(
                "ML-функциональность отключена: scikit-learn недоступен (%s). Используется фолбэк %.4f%%",
                self._sklearn_import_error,
                self.fallback_threshold,
            )

    @property
    def ml_supported(self) -> bool:
        """Показывает, доступна ли ML-функциональность (импортирована ли scikit-learn)."""

        return self._sklearn_modules is not None

    def _ensure_sklearn(self) -> None:
        """Лениво импортирует scikit-learn и связанные зависимости."""

        if self._sklearn_modules or self._sklearn_import_error:
            return

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.exceptions import NotFittedError
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error
            from joblib import dump, load

            self._sklearn_modules = SimpleNamespace(
                RandomForestRegressor=RandomForestRegressor,
                NotFittedError=NotFittedError,
                train_test_split=train_test_split,
                mean_absolute_error=mean_absolute_error,
                dump=dump,
                load=load,
            )
        except Exception as exc:  # noqa: BLE001
            self._sklearn_import_error = exc

    def _load_model(self) -> None:
        """Пытается загрузить модель с диска, при отсутствии использует фолбэк."""

        if not self.ml_supported:
            return

        if not self.model_path.exists():
            logger.info(
                "ML-модель для порога не найдена по пути %s, используется фолбэк %.4f%%",
                self.model_path,
                self.fallback_threshold,
            )
            return

        try:
            self.model = self._sklearn_modules.load(self.model_path)
            logger.info("ML-модель порога загружена из %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Не удалось загрузить ML-модель %s (%s). Будет использован фолбэк %.4f%%",
                self.model_path,
                exc,
                self.fallback_threshold,
            )
            self.model = None

    def _encode_market_regime(self, regime: str | None) -> float:
        """Кодирует строковое описание режима рынка в числовой признак."""

        if not regime:
            return 0.0
        normalized = regime.lower()
        mapping = {
            "live_spot_high_volatility": 3.0,
            "live_spot_normal": 2.0,
            "live_spot_low_volatility": 1.0,
            "testnet_spot": -1.0,
            "live_linear": 1.5,
            "testnet_linear": -1.5,
        }
        return mapping.get(normalized, float(hash(normalized) % 1000) / 1000)

    def _prepare_features(self, context: dict) -> List[float]:
        """Формирует вектор признаков из контекста рынка."""

        features: List[float] = []
        for name in self.FEATURE_ORDER:
            if name == "market_regime":
                features.append(self._encode_market_regime(context.get(name)))
                continue
            value = context.get(name, 0.0)
            try:
                features.append(float(value))
            except (TypeError, ValueError):
                features.append(0.0)
        return features

    def train(self, contexts: Iterable[dict], targets: Sequence[float]) -> float:
        """Обучает модель и сохраняет её на диск, возвращая MAE на валидации."""

        if not self.ml_supported:
            logger.warning(
                "Обучение ML-модели пропущено: scikit-learn недоступен (%s). Возвращаем фолбэк %.4f%%",
                self._sklearn_import_error,
                self.fallback_threshold,
            )
            return self.fallback_threshold

        rows = [self._prepare_features(ctx) for ctx in contexts]
        X_train, X_val, y_train, y_val = self._sklearn_modules.train_test_split(
            rows, targets, test_size=0.2, random_state=42
        )

        model = self._sklearn_modules.RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        mae = float(self._sklearn_modules.mean_absolute_error(y_val, val_pred))

        self.model = model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._sklearn_modules.dump(model, self.model_path)
        logger.info("Модель порога обучена и сохранена в %s (MAE=%.5f)", self.model_path, mae)
        return mae

    def predict_threshold(self, context: dict) -> float:
        """Возвращает прогноз порога прибыли или фолбэк при отсутствии модели."""

        features = self._prepare_features(context)
        if not self.model or not self.ml_supported:
            return self.fallback_threshold

        try:
            prediction = float(self.model.predict([features])[0])
            return prediction
        except self._sklearn_modules.NotFittedError:
            logger.warning("Модель RandomForest не обучена, возвращаем фолбэк")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка предсказания порога (%s), используем фолбэк", exc)

        return self.fallback_threshold

