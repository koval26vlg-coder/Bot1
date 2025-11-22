import pathlib
import sys

import requests

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _build_payload(with_extra_legs):
    market_list = []
    for idx in range(22):
        market_list.append({
            "symbol": f"COIN{idx}USDT",
            "baseCoin": f"COIN{idx}",
            "quoteCoin": "USDT",
            "turnover24h": 1000 - idx,
        })

    for leg_idx, leg in enumerate(with_extra_legs, start=1):
        market_list.append({
            "symbol": leg,
            "baseCoin": leg.replace("USDT", ""),
            "quoteCoin": "USDT",
            "turnover24h": leg_idx,
        })

    return {
        "retCode": 0,
        "result": {
            "list": market_list,
        },
    }


def test_fetch_limits_and_preserves_triangle_legs(monkeypatch):
    """Проверяет ограничение до топ-20 и сохранение уже настроенных ног."""
    extra_legs = ["KEEP1USDT", "KEEP2USDT", "KEEP3USDT"]
    payload = _build_payload(extra_legs)

    def fake_get(url, params=None, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    config = Config()
    config._triangular_pairs_cache = [{"name": "custom", "legs": extra_legs}]

    available = config._fetch_market_symbols()
    assert len(available) == 20
    assert available == [f"COIN{idx}USDT" for idx in range(20)]

    watchlist = config.SYMBOLS
    for leg in extra_legs:
        assert leg in watchlist
    assert len(watchlist) == len(set(available)) + len(extra_legs)
