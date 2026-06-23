from __future__ import annotations

import pytest

from arktor_bench.utils.log import bar_disabled, log


def test_log_writes_flushed_line(capsys: pytest.CaptureFixture[str]) -> None:
    log("hello")
    assert capsys.readouterr().out == "hello\n"


def test_bar_disabled_is_bool() -> None:
    assert isinstance(bar_disabled(), bool)             # off under pytest capture (no TTY)
