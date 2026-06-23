from __future__ import annotations

import base64
import hashlib
import json
import shlex
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from arktor_bench.config import HarnessInvocation, get_config
from arktor_bench.models import Produced, TaskSpec
from arktor_bench.sandbox.backend import Backend, ExecuteResult
from arktor_bench.sandbox.docker import DockerBackend
from arktor_bench.sandbox.local import LocalBackend


class Workspace:
    def __init__(self, root: Path, backend: Backend) -> None:
        self.root = root
        self.baseline: dict[str, str] = {}
        self._backend = backend

    async def execute(self, cmd: str, *, timeout: float = 30.0, check: bool = False,
                      env: dict[str, str] | None = None) -> ExecuteResult:
        r = await self._backend.execute(cmd, timeout=timeout, env=env)
        if check and r.exit_code != 0:
            raise AssertionError(f"command failed ({r.exit_code}): {cmd}\n{r.stdout}")
        return r

    async def run(self, cmd: str, *, check: bool = True, timeout: float = 30.0) -> ExecuteResult:
        return await self.execute(cmd, timeout=timeout, check=check)

    def read_text(self, rel: str) -> str | None:
        p = self.root / rel
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None

    def read_json(self, rel: str) -> Any | None:
        t = self.read_text(rel)
        if t is None:
            return None
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return None

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()


def _materialize(task: TaskSpec, root: Path) -> None:
    for w in task.workspace_files:
        dst = root / w.dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((task.dir / w.source).read_bytes())


# Tooling byproducts no task delivers — caches, deps, VCS/editor/OS metadata (zero false-positives).
_IGNORE_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",   # python tool caches
    "node_modules", ".venv",                                        # deps / virtualenv
    ".git", ".hg", ".svn",                                          # vcs metadata
    ".idea", ".vscode",                                             # editor metadata
})
_IGNORE_SUFFIXES = frozenset({".pyc", ".pyo"})                      # python bytecode
_IGNORE_NAMES = frozenset({".DS_Store", "Thumbs.db"})              # os junk


def is_artifact(rel: Path) -> bool:
    """A workspace path that counts as a deliverable, excluding tooling byproducts."""
    return (rel.suffix not in _IGNORE_SUFFIXES
            and rel.name not in _IGNORE_NAMES
            and _IGNORE_DIRS.isdisjoint(rel.parts))


def _snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and is_artifact(rel):
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def classify_produced(ws: Workspace) -> list[Produced]:
    """Agent deliverable = files new or changed vs the post-setup baseline."""
    out: list[Produced] = []
    for rel, h in _snapshot(ws.root).items():
        base = ws.baseline.get(rel)
        if base is None:
            out.append(Produced(path=rel, status="created"))
        elif base != h:
            out.append(Produced(path=rel, status="modified"))
    return sorted(out, key=lambda p: p.path)


async def _place_files(ws: Workspace, files: dict[str, str]) -> None:
    for container_path, host_path in files.items():
        src = Path(host_path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"harness files source missing: {host_path}")
        b64 = base64.b64encode(src.read_bytes()).decode()
        parent = str(PurePosixPath(container_path).parent)
        await ws.execute(
            f"mkdir -p {shlex.quote(parent)} && "
            f"echo {b64} | base64 -d > {shlex.quote(container_path)} && "
            f"chmod 600 {shlex.quote(container_path)}",
            check=True,
        )


@asynccontextmanager
async def _make(setup: str | None, root: Path, backend_mode: str,
                image: str | None) -> AsyncIterator[Workspace]:
    cfg = get_config()
    backend: Backend = (
        DockerBackend(image or cfg.docker_image, memory=cfg.mem_limit, cpus=cfg.cpus)
        if backend_mode == "docker" else LocalBackend()
    )
    await backend.start(str(root))
    ws = Workspace(root, backend)
    try:
        if setup:
            await ws.execute(setup, timeout=cfg.wall_s, check=True)
        ws.baseline = _snapshot(root)
        yield ws
    finally:
        await backend.stop()


@asynccontextmanager
async def task_workspace(task: TaskSpec, inv: HarnessInvocation) -> AsyncIterator[Workspace]:
    root = Path(tempfile.mkdtemp(prefix="abx-"))
    try:
        _materialize(task, root)
        async with _make(task.setup, root, inv.backend, inv.image) as ws:
            if inv.backend == "docker":
                await _place_files(ws, inv.files)
            yield ws
    finally:                                      # backend already stopped; drop the host temp dir
        shutil.rmtree(root, ignore_errors=True)


@asynccontextmanager
async def scratch_workspace(src: Path | None) -> AsyncIterator[Workspace]:
    root = Path(tempfile.mkdtemp(prefix="abx-val-"))
    try:
        if src and src.is_dir():
            shutil.copytree(src, root, dirs_exist_ok=True)
        async with _make(None, root, "docker", None) as ws:
            yield ws
    finally:
        shutil.rmtree(root, ignore_errors=True)
