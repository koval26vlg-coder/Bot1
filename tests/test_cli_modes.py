import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


def _reset_env(monkeypatch):
    """Очищает ключевые переменные окружения перед запуском режима."""

    for key in [
        "MIN_TRIANGULAR_PROFIT",
        "TRADE_AMOUNT",
        "TESTNET",
        "SIMULATION_MODE",
        "PAPER_TRADING_MODE",
        "ENVIRONMENT",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_standard_mode_applies_thresholds(monkeypatch):
    """Стандартный режим должен передавать пороги через переменные окружения."""

    _reset_env(monkeypatch)
    captured = {}

    def fake_run(*, logger_adapter, mode, environment):
        captured["mode"] = mode
        captured["environment"] = environment
        captured["logger_extra"] = getattr(logger_adapter, "extra", {})

    monkeypatch.setattr(main, "advanced_main", fake_run)

    parser = main._build_parser()
    args = parser.parse_args(
        ["--mode", "standard", "--min-profit", "0.2", "--trade-amount", "50"]
    )
    main._execute_mode(args)

    assert float(os.environ["MIN_TRIANGULAR_PROFIT"]) == 0.2
    assert float(os.environ["TRADE_AMOUNT"]) == 50
    assert captured["mode"] == "standard"
    assert captured["environment"] == "production"
    assert captured["logger_extra"].get("mode") == "standard"


def test_aggressive_mode_defaults(monkeypatch):
    """Агрессивный режим должен проставлять значения по умолчанию."""

    _reset_env(monkeypatch)
    captured = {}

    def fake_run(*, logger_adapter, mode, environment):
        captured["environment"] = environment
        captured["mode"] = mode

    monkeypatch.setattr(main, "advanced_main", fake_run)

    parser = main._build_parser()
    args = parser.parse_args(["--mode", "aggressive"])
    main._execute_mode(args)

    assert os.environ["MIN_TRIANGULAR_PROFIT"] == "0.01"
    assert os.environ["TRADE_AMOUNT"] == "25"
    assert os.environ["TESTNET"] == "true"
    assert os.environ["SIMULATION_MODE"] == "true"
    assert captured["environment"] == "simulation"


def test_quick_mode_builds_custom_config(monkeypatch):
    """Быстрый режим должен создавать OptimizedConfig с пользовательскими параметрами."""

    _reset_env(monkeypatch)
    captured = {}

    class DummyEngine:
        def __init__(self, config=None):
            captured["config"] = config
            self.config = config or SimpleNamespace()

    def fake_quick_test(engine, logger):
        captured["engine"] = engine
        captured["logger_level"] = logger.logger.level

    monkeypatch.setattr(main, "AdvancedArbitrageEngine", DummyEngine)
    monkeypatch.setattr(main, "_quick_test", fake_quick_test)

    parser = main._build_parser()
    args = parser.parse_args(
        ["--mode", "quick", "--min-profit", "0.5", "--trade-amount", "10"]
    )
    main._execute_mode(args)

    config = captured["config"]
    assert isinstance(config, main.OptimizedConfig)
    assert config.MIN_TRIANGULAR_PROFIT == 0.5
    assert config.TRADE_AMOUNT == 10
    assert os.environ["TESTNET"] == "true"


def test_replay_mode_uses_cli_parameters(monkeypatch):
    """Режим replay должен прокидывать путь и параметры ускорения в HistoricalReplayer."""

    _reset_env(monkeypatch)
    captured = {}

    class DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(REPLAY_SPEED=2.0, REPLAY_MAX_RECORDS=5, REPLAY_DATA_PATH=None)

    class DummyReplayer:
        def __init__(self, engine, data_path, speed, max_records):
            captured.update(
                {
                    "engine": engine,
                    "data_path": data_path,
                    "speed": speed,
                    "max_records": max_records,
                }
            )

        def replay(self):
            captured["replayed"] = True

    monkeypatch.setattr(main, "AdvancedArbitrageEngine", DummyEngine)
    monkeypatch.setattr(main, "HistoricalReplayer", DummyReplayer)

    parser = main._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "replay",
            "--replay-path",
            "sample.csv",
            "--replay-speed",
            "3.0",
            "--replay-limit",
            "10",
        ]
    )
    main._execute_mode(args)

    assert os.environ["SIMULATION_MODE"] == "true"
    assert os.environ["PAPER_TRADING_MODE"] == "true"
    assert captured["data_path"] == "sample.csv"
    assert captured["speed"] == 3.0
    assert captured["max_records"] == 10
    assert captured.get("replayed") is True
