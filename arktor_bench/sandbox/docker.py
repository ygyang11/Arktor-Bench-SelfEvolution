from __future__ import annotations

import asyncio
import atexit
import threading
from pathlib import Path
from typing import Any

import docker as _docker
from arktor_bench.sandbox.backend import ExecuteResult

_WORKDIR = "/workspace"


class DockerBackend:
    def __init__(self, image: str, *, network: str = "bridge",
                 memory: str | None = None, cpus: float = 0.0,
                 mounts: dict[str, str] | None = None) -> None:
        self._image = image
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._mounts = mounts or {}
        self._client: Any = None
        self._container: Any = None

    async def start(self, workspace: str) -> None:
        def _create() -> tuple[Any, Any]:
            client = _docker.from_env()
            try:
                kwargs: dict[str, Any] = {}
                if self._memory:
                    kwargs["mem_limit"] = self._memory
                if self._cpus > 0:
                    kwargs["nano_cpus"] = int(self._cpus * 1e9)
                volumes: dict[str, dict[str, str]] = {workspace: {"bind": _WORKDIR, "mode": "rw"}}
                for container_path, host_path in self._mounts.items():
                    src = Path(host_path).expanduser().resolve()
                    if not src.exists():                       # fail loud, not a silent empty mount
                        raise FileNotFoundError(f"harness mount source missing: {host_path}")
                    volumes[str(src)] = {"bind": container_path, "mode": "ro"}
                container = client.containers.run(
                    self._image, "sleep infinity", detach=True, working_dir=_WORKDIR,
                    network_mode=self._network, volumes=volumes, **kwargs,
                )
                return client, container
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass
                raise

        loop = asyncio.get_running_loop()
        self._client, self._container = await loop.run_in_executor(None, _create)
        atexit.register(self._sync_cleanup)

    async def execute(self, command: str, *, timeout: float = 30.0,
                      env: dict[str, str] | None = None) -> ExecuteResult:
        if self._container is None:
            raise RuntimeError("backend not started")
        api = self._client.api
        loop = asyncio.get_running_loop()
        exec_id = await loop.run_in_executor(None, lambda: api.exec_create(
            self._container.id, ["bash", "-c", command],
            workdir=_WORKDIR, stdout=True, stderr=True, environment=env)["Id"])
        out, err = bytearray(), bytearray()
        stream: Any = None
        stop = threading.Event()
        read_error: list[str] = []

        def close_stream() -> None:
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - an already-closed socket is harmless
                    pass
                finally:
                    response = getattr(stream, "_response", None)
                    if response is not None:
                        try:
                            response.close()
                        except Exception:  # noqa: BLE001 - best-effort SDK compatibility cleanup
                            pass

        def read() -> None:
            nonlocal stream
            try:
                stream = api.exec_start(exec_id, stream=True, demux=True)
                if stop.is_set():
                    return
                for stdout, stderr in stream:
                    if stdout:
                        out.extend(stdout)
                    if stderr:
                        err.extend(stderr)
            except Exception as exc:  # Docker disconnects leave an incomplete stream.
                read_error.append(str(exc))
            finally:
                close_stream()

        reader = loop.run_in_executor(None, read)
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=timeout)
        except TimeoutError:
            stop.set()
            close_stream()
            await self._kill(exec_id)
            stderr = err.decode(errors="replace")
            timeout_error = f"timeout after {timeout}s"
            return ExecuteResult(
                None,
                out.decode(errors="replace"),
                f"{stderr}\n{timeout_error}" if stderr else timeout_error,
            )
        except asyncio.CancelledError:
            stop.set()
            close_stream()
            await self._kill(exec_id)
            raise
        except Exception as e:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"docker error: {e}")
        if read_error:
            return ExecuteResult(None, out.decode(errors="replace"),
                                 "\n".join([err.decode(errors="replace"), read_error[0]]))
        code = await loop.run_in_executor(None, lambda: api.exec_inspect(exec_id).get("ExitCode"))
        if not isinstance(code, int):
            return ExecuteResult(None, out.decode(errors="replace"), err.decode(errors="replace"))
        return ExecuteResult(
            exit_code=code,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
        )

    async def _kill(self, exec_id: str) -> None:
        api = self._client.api

        def _do() -> None:
            pid = api.exec_inspect(exec_id).get("Pid", 0)
            if pid > 0:
                self._container.exec_run(
                    ["bash", "-c", f"kill -9 -{pid} 2>/dev/null; kill -9 {pid} 2>/dev/null"])

        await asyncio.get_running_loop().run_in_executor(None, _do)

    async def stop(self) -> None:
        if self._container is None and self._client is None:
            return
        container, self._container = self._container, None
        client, self._client = self._client, None

        def _dispose() -> None:
            try:
                if container is not None:
                    container.remove(force=True)
            finally:
                if client is not None:
                    client.close()

        await asyncio.get_running_loop().run_in_executor(None, _dispose)

    def _sync_cleanup(self) -> None:
        container, self._container = self._container, None
        client, self._client = self._client, None
        try:
            if container is not None:
                container.remove(force=True)
        except Exception:
            pass
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
