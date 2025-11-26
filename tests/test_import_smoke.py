"""Проверяет, что пакет импортируется без ручного изменения sys.path."""

import os
import subprocess
import sys


def test_advanced_bot_import_keeps_sys_path_stable(tmp_path):
    """Импорт модуля не добавляет сторонние пути."""

    install_dir = tmp_path / "site"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            ".",
            "-t",
            str(install_dir),
        ]
    )

    env = {"PYTHONPATH": str(install_dir), **os.environ}
    sys.modules.pop("arbitrage_bot.core.advanced_bot", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arbitrage_bot.core.advanced_bot as mod; print(mod.main)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "main" in result.stdout
