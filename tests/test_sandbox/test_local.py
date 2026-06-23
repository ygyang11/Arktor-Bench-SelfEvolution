from __future__ import annotations

from typing import Any

import pytest

import arktor_bench.sandbox.local as local_mod
from arktor_bench.sandbox.local import LocalBackend


class _FakeProc:
    def __init__(self, out: bytes) -> None:
        self._out = out

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, b""


async def test_start_errors_without_conda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_exec(*_a: Any, **_k: Any) -> _FakeProc:
        return _FakeProc(b"# conda environments:\nbase  /x\narktor  /y\n")

    monkeypatch.setattr(local_mod.asyncio, "create_subprocess_exec", _fake_exec)
    with pytest.raises(SystemExit):
        await LocalBackend().start("/tmp/ws")
