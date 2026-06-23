from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from arktor_bench.config import HarnessInvocation
from arktor_bench.models import Produced, TaskSpec
from arktor_bench.sandbox.backend import ExecuteResult
from arktor_bench.sandbox.workspace import Workspace, classify_produced
from arktor_bench.trajectory.record import TrajectoryRecord


class RunResult(BaseModel):
    trajectory: TrajectoryRecord
    produced: list[Produced]


class Adapter(ABC):
    name: str

    @abstractmethod
    async def run(self, task: TaskSpec, ws: Workspace, inv: HarnessInvocation) -> RunResult: ...

    @abstractmethod
    def to_trajectory(self, raw: list[dict[str, Any]]) -> TrajectoryRecord: ...


def parse_ndjson(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def args_str(args: Any) -> str:
    """Full string form of a tool call's args (no truncation; compact trims later)."""
    if args is None:
        return ""
    return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)


def finish(adapter: Adapter, task: TaskSpec, ws: Workspace,
           res: ExecuteResult, t0: float) -> RunResult:
    traj = adapter.to_trajectory(parse_ndjson(res.stdout))
    traj.wall_ms = int((time.monotonic() - t0) * 1000)
    if res.exit_code != 0:
        traj.cap_hit = True
        if not traj.error:
            traj.error = res.stderr.strip() or f"exit code {res.exit_code}"
    return RunResult(trajectory=traj, produced=classify_produced(ws))
