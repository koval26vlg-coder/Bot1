import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitoring import AdvancedMonitor


class DummyEngine:
    def get_strategy_status(self):
        return {}


def test_export_trade_history_contains_order_data(tmp_path):
    monitor = AdvancedMonitor(engine=DummyEngine())

    trade_plan = {
        'initial_amount': 100,
        'estimated_profit_usdt': 5.0,
        'execution_path': ['BTCUSDT', 'ETHUSDT', 'BTCETH']
    }

    trade_record = {
        'timestamp': datetime.now(),
        'symbol': 'TEST_TRIANGLE',
        'type': 'triangular',
        'profit': trade_plan['estimated_profit_usdt'],
        'profit_percent': 0.5,
        'direction': 'forward',
        'execution_time': 0.5,
        'market_conditions': 'normal',
        'triangle_stats': {},
        'trade_plan': trade_plan,
        'results': [
            {
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'qty': 0.003,
                'avgPrice': 30000,
                'cumExecQty': 0.003
            }
        ],
        'total_profit': trade_plan['estimated_profit_usdt'],
        'simulated': False,
        'details': {
            'triangle': 'TEST_TRIANGLE',
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BTCETH'],
            'direction': 'forward',
            'initial_amount': trade_plan['initial_amount'],
            'execution_path': trade_plan['execution_path'],
            'real_executed': True
        }
    }

    monitor.track_trade(trade_record)

    export_path = tmp_path / 'trade_history.csv'
    exported_file = monitor.export_trade_history(str(export_path))

    assert exported_file

    with open(exported_file, newline='', encoding='utf-8') as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert len(rows) >= 1
    assert rows[0]['symbol'] == 'BTCUSDT'
    assert rows[0]['trade_details']
