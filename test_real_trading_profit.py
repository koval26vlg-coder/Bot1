import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_trading


class FakeBybitClient:
    def __init__(self, orders=None):
        self.orders = list(orders or [])

    def place_order(self, **kwargs):
        if not self.orders:
            return None
        return self.orders.pop(0)

    def cancel_order(self, order_id, symbol):
        return None


class FakeRiskManager:
    def __init__(self):
        self.last_profit = None

    def can_execute_trade(self, trade_plan):
        return True

    def update_after_trade(self, trade_record):
        self.last_profit = trade_record.get('total_profit')


@pytest.fixture
def triangular_orders():
    return [
        {
            'orderId': '1',
            'orderStatus': 'Filled',
            'symbol': 'BTCUSDT',
            'side': 'Buy',
            'qty': 0.01,
            'price': 10000,
            'avgPrice': 10000,
            'cumExecQty': 0.01,
        },
        {
            'orderId': '2',
            'orderStatus': 'Filled',
            'symbol': 'BTCETH',
            'side': 'Sell',
            'qty': 0.00999,
            'price': 20,
            'avgPrice': 20,
            'cumExecQty': 0.00999,
        },
        {
            'orderId': '3',
            'orderStatus': 'Filled',
            'symbol': 'ETHUSDT',
            'side': 'Sell',
            'qty': 0.1996002,
            'price': 2000,
            'avgPrice': 2000,
            'cumExecQty': 0.1996002,
        },
    ]


def test_calculate_real_profit(monkeypatch, triangular_orders):
    monkeypatch.setattr(real_trading, "BybitClient", lambda: FakeBybitClient())

    executor = real_trading.RealTradingExecutor()

    trade_plan = {
        'initial_amount': 100.0,
        'estimated_profit_usdt': 0.0,
    }

    profit = executor._calculate_real_profit(triangular_orders, trade_plan)

    assert profit == pytest.approx(298.8012, rel=1e-4)


def test_execute_real_trade_uses_actual_profit(monkeypatch, triangular_orders):
    monkeypatch.setattr(real_trading, "BybitClient", lambda: FakeBybitClient(triangular_orders))

    executor = real_trading.RealTradingExecutor()
    executor.risk_manager = FakeRiskManager()

    trade_plan = {
        'initial_amount': 100.0,
        'estimated_profit_usdt': 1.0,
        'step1': {'symbol': 'BTCUSDT', 'side': 'Buy', 'amount': 0.01, 'price': 10000},
        'step2': {'symbol': 'BTCETH', 'side': 'Sell', 'amount': 0.00999, 'price': 20},
        'step3': {'symbol': 'ETHUSDT', 'side': 'Sell', 'amount': 0.1996002, 'price': 2000},
    }

    trade_record = executor._execute_real_trade(trade_plan)

    assert trade_record
    assert trade_record['total_profit'] == pytest.approx(298.8012, rel=1e-4)
    assert executor.risk_manager.last_profit == pytest.approx(298.8012, rel=1e-4)
