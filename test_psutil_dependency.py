"""Смоук-тест для проверки доступности psutil"""

import importlib


def test_psutil_is_available():
    """Проверяем, что psutil доступен для импорта"""
    if importlib.util.find_spec("psutil") is None:
        raise AssertionError(
            "psutil недоступен. Установите зависимости командой 'pip install -r requirements.txt'."
        )
