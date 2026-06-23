from __future__ import annotations

import asyncio
import atexit
from typing import Any

import docker as _docker
from arktor_bench.sandbox.backend import ExecuteResult

_WORKDIR = "/workspace"


class DockerBackend:
    def __init__(self, image: str, *, network: str = "bridge",
                 memory: str | None = None, cpus: float = 0.0) -> None:
        self._image = image
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._client: Any = None
        self._container: Any = None

    async def start(self, workspace: str) -> None:
        def _create() -> tuple[Any, Any]:
            client = _docker.from_env()
            kwargs: dict[str, Any] = {}
            if self._memory:
                kwargs["mem_limit"] = self._memory
            if self._cpus > 0:
                kwargs["nano_cpus"] = int(self._cpus * 1e9)
            container = client.containers.run(
                self._image, "sleep infinity", detach=True, working_dir=_WORKDIR,
                network_mode=self._network,
                volumes={workspace: {"bind": _WORKDIR, "mode": "rw"}}, **kwargs,
            )
            return client, container

        loop = asyncio.get_running_loop()
        self._client, self._container = await loop.run_in_executor(None, _create)
        atexit.register(self._sync_cleanup)

    async def execute(self, command: str, *, timeout: float = 30.0,
                      env: dict[str, str] | None = None) -> ExecuteResult:
        if self._container is None:
            raise RuntimeError("backend not started")
        loop = asyncio.get_running_loop()
        api = self._client.api
        exec_id = await loop.run_in_executor(None, lambda: api.exec_create(
            self._container.id, ["bash", "-c", command],
            workdir=_WORKDIR, stdout=True, stderr=True, environment=env)["Id"])
        try:
            out, err = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: api.exec_start(exec_id, demux=True)),
                timeout=timeout,
            )
        except TimeoutError:
            await self._kill(exec_id)
            return ExecuteResult(exit_code=None, stdout=f"timeout after {timeout}s")
        except asyncio.CancelledError:
            await self._kill(exec_id)
            raise
        except Exception as e:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"docker error: {e}")
        code = await loop.run_in_executor(
            None, lambda: int(api.exec_inspect(exec_id).get("ExitCode", -1)))
        return ExecuteResult(
            exit_code=code,
            stdout=(out or b"").decode(errors="replace"),
            stderr=(err or b"").decode(errors="replace"),
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
        if self._container is None:
            return
        c, self._container = self._container, None
        await asyncio.get_running_loop().run_in_executor(None, lambda: c.remove(force=True))

    def _sync_cleanup(self) -> None:
        if self._container:
            try:
                self._container.remove(force=True)
            except Exception:
                pass
