from __future__ import annotations

from pathlib import Path

import pytest

from arktor_bench.sandbox.docker import DockerBackend
from arktor_bench.sandbox.workspace import Workspace, _place_files

pytestmark = pytest.mark.docker


async def test_start_execute_stop(tmp_path: Path) -> None:
    backend = DockerBackend("arktor-bench:base")
    await backend.start(str(tmp_path))
    try:
        r = await backend.execute("echo hi")
        assert r.exit_code == 0
        assert "hi" in r.stdout
    finally:
        await backend.stop()


async def test_place_files_round_trip(tmp_path: Path) -> None:
    host = tmp_path / "src.txt"
    host.write_text("secret-content")
    backend = DockerBackend("arktor-bench:base")
    await backend.start(str(tmp_path))
    try:
        ws = Workspace(tmp_path, backend)
        await _place_files(ws, {"/root/cfg/src.txt": str(host)})
        r = await ws.execute("cat /root/cfg/src.txt")
        assert r.stdout.strip() == "secret-content"
    finally:
        await backend.stop()
