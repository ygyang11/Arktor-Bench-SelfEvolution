from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import arktor_bench.sandbox.docker as docker_mod
from arktor_bench.sandbox.docker import DockerBackend
from arktor_bench.sandbox.workspace import Workspace, _place_files


@pytest.mark.docker
async def test_start_execute_stop(tmp_path: Path) -> None:
    backend = DockerBackend("arktor-bench:base")
    await backend.start(str(tmp_path))
    try:
        r = await backend.execute("echo hi")
        assert r.exit_code == 0
        assert "hi" in r.stdout
    finally:
        await backend.stop()


@pytest.mark.docker
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


def _backend(stream: object) -> tuple[DockerBackend, SimpleNamespace]:
    backend = DockerBackend("unused")
    api = SimpleNamespace(start_kwargs={})
    api.exec_create = lambda *_args, **_kwargs: {"Id": "exec-1"}

    def exec_start(_exec_id: str, **kwargs: object) -> object:
        api.start_kwargs = kwargs
        return stream

    api.exec_start = exec_start
    api.exec_inspect = lambda _exec_id: {"ExitCode": 0, "Pid": 0}
    backend._client = SimpleNamespace(api=api)
    backend._container = SimpleNamespace(id="container-1", exec_run=lambda *_args, **_kwargs: None)
    return backend, api


async def test_execute_reads_demuxed_stream() -> None:
    stream = MagicMock()
    stream.__iter__.return_value = iter([(b"out", None), (None, b"err")])
    backend, api = _backend(stream)

    result = await backend.execute("echo test")

    assert result.exit_code == 0
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert api.start_kwargs == {"stream": True, "demux": True}
    stream._response.close.assert_called()


async def test_execute_timeout_keeps_streamed_output() -> None:
    closed = threading.Event()

    def frames() -> object:
        yield b'{"type":"step"}\n', None
        closed.wait()

    stream = MagicMock()
    stream.__iter__.side_effect = frames
    stream.close.side_effect = closed.set
    backend, _ = _backend(stream)

    result = await backend.execute("slow", timeout=0.01)

    assert result.exit_code is None
    assert result.stdout == '{"type":"step"}\n'
    assert result.stderr == "timeout after 0.01s"
    stream.close.assert_called()
    stream._response.close.assert_called()


async def test_start_failure_closes_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = MagicMock()
    client.containers.run.side_effect = RuntimeError("create failed")
    monkeypatch.setattr(docker_mod._docker, "from_env", lambda: client)

    with pytest.raises(RuntimeError, match="create failed"):
        await DockerBackend("unused").start(str(tmp_path))

    client.close.assert_called_once_with()


async def test_stop_closes_client_when_container_removal_fails() -> None:
    backend = DockerBackend("unused")
    backend._client = client = MagicMock()
    backend._container = container = MagicMock()
    container.remove.side_effect = RuntimeError("remove failed")

    with pytest.raises(RuntimeError, match="remove failed"):
        await backend.stop()

    client.close.assert_called_once_with()
    assert backend._client is None
    assert backend._container is None
