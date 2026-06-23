from __future__ import annotations

import shlex
import time
import uuid
from typing import Any

from arktor_bench.config import HarnessInvocation, get_config
from arktor_bench.harness.base import Adapter, RunResult, args_str, finish
from arktor_bench.models import TaskSpec
from arktor_bench.sandbox.workspace import Workspace
from arktor_bench.trajectory.record import StepRecord, TokenUsage, ToolEvent, TrajectoryRecord


class ArktorAdapter(Adapter):
    name = "arktor"

    async def run(self, task: TaskSpec, ws: Workspace, inv: HarnessInvocation) -> RunResult:
        env = dict(inv.env)
        if inv.model:
            env["ARKTOR_LLM_MODEL"] = inv.model
        cmd = (f"arktor -p {shlex.quote(task.prompt)} "
               f"--output-format json -s {uuid.uuid4().hex}")
        t0 = time.monotonic()
        res = await ws.execute(cmd, timeout=get_config().wall_s, env=env)
        return finish(self, task, ws, res, t0)

    def to_trajectory(self, raw: list[dict[str, Any]]) -> TrajectoryRecord:
        steps: list[StepRecord] = []
        tokens = TokenUsage()
        cap = False
        error = ""
        for ev in raw:
            if ev.get("type") == "step":
                obs = {o.get("tool_call_id"): o for o in (ev.get("observation") or [])}
                tools = [
                    ToolEvent(
                        name=a.get("name", "unknown"),
                        args=args_str(a.get("arguments")),
                        is_error=bool(obs.get(a.get("id"), {}).get("is_error")),
                        output=str(obs.get(a.get("id"), {}).get("content", "")),
                    )
                    for a in (ev.get("action") or [])
                ]
                steps.append(StepRecord(
                    index=int(ev.get("index", len(steps))),
                    think=ev.get("thought") or "",
                    response=ev.get("response") or "", tools=tools,
                ))
            elif ev.get("type") == "result":
                u = ev.get("usage") or {}
                tokens = TokenUsage(
                    input=u.get("prompt_tokens", 0),
                    cached_input=u.get("cache_read_tokens", 0),
                    output=u.get("completion_tokens", 0),
                    reasoning=u.get("reasoning_tokens", 0),
                )
                if ev.get("is_error"):
                    cap = True
                    error = str(ev.get("error") or "")
        return TrajectoryRecord(steps=steps, tokens=tokens, cap_hit=cap, error=error)
