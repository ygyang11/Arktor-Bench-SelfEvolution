from __future__ import annotations

import asyncio
import os
import signal

from arktor_bench.sandbox.backend import ExecuteResult

_LOCAL_ENV = "arktor-bench"


class LocalBackend:
    def __init__(self) -> None:
        self._root: str = ""

    async def start(self, workspace: str) -> None:
        self._root = workspace
        proc = await asyncio.create_subprocess_exec(
            "conda", "env", "list",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        names = {ln.split()[0] for ln in out.decode().splitlines()
                 if ln.strip() and not ln.startswith("#")}
        if _LOCAL_ENV not in names:
            raise SystemExit(
                f"local backend needs conda env '{_LOCAL_ENV}'; "
                "create it: conda env create -f environment.yml")

    def _kill(self, proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def execute(self, command: str, *, timeout: float = 30.0,
                      env: dict[str, str] | None = None) -> ExecuteResult:
        # Strip conda-nesting vars: `conda run` (v26.x) breaks when invoked from a deeply nested
        # activation (CONDA_SHLVL>1), failing every command. A clean env makes it reliable.
        base = {k: v for k, v in os.environ.items()
                if k not in ("CONDA_SHLVL", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
                             "CONDA_PROMPT_MODIFIER", "CONDA_STACKED_2")}
        try:
            proc = await asyncio.create_subprocess_exec(
                "conda", "run", "--no-capture-output", "-n", _LOCAL_ENV, "bash", "-c", command,
                cwd=self._root, env={**base, **(env or {})},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"local exec failed: {e}")
        # Drain the pipes incrementally rather than via communicate(): on a wall-clock timeout
        # communicate() discards everything it has buffered, so a capped run loses its whole
        # streamed trajectory (token/context/step accounting). Draining into our own buffers
        # keeps the partial stream-json the process emitted before it was killed.
        out_buf, err_buf = bytearray(), bytearray()

        async def _drain(stream: asyncio.StreamReader | None, buf: bytearray) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)

        readers = asyncio.gather(_drain(proc.stdout, out_buf), _drain(proc.stderr, err_buf),
                                 return_exceptions=True)
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            self._kill(proc)
            await proc.wait()
        except asyncio.CancelledError:
            self._kill(proc)
            await proc.wait()
            readers.cancel()
            raise
        # once the process group is gone the pipes hit EOF and the readers finish (return_exceptions
        # keeps a stream error from propagating); bound the wait so a stuck pipe can't hang us, then
        # keep whatever we drained. a genuine outer cancellation still propagates.
        try:
            await asyncio.wait_for(readers, timeout=5.0)
        except TimeoutError:
            readers.cancel()
        return ExecuteResult(
            exit_code=None if timed_out else proc.returncode,
            stdout=bytes(out_buf).decode(errors="replace"),
            stderr=bytes(err_buf).decode(errors="replace"),
        )

    async def stop(self) -> None:
        return
