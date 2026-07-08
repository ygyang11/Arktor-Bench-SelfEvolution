from __future__ import annotations

import shlex
import time
from typing import Any

from arktor_bench.config import HarnessInvocation, get_config
from arktor_bench.harness.base import Adapter, RunResult, args_str, finish
from arktor_bench.models import TaskSpec
from arktor_bench.sandbox.workspace import Workspace
from arktor_bench.trajectory.record import StepRecord, TokenUsage, ToolEvent, TrajectoryRecord


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class ClaudeCodeAdapter(Adapter):
    name = "claude_code"

    async def run(self, task: TaskSpec, ws: Workspace, inv: HarnessInvocation) -> RunResult:
        model = f"--model {shlex.quote(inv.model)} " if inv.model else ""
        cmd = (f"claude -p {shlex.quote(task.prompt)} --output-format stream-json --verbose "
               f"--permission-mode bypassPermissions {model}").rstrip()
        t0 = time.monotonic()
        res = await ws.execute(cmd, timeout=get_config().wall_s, env=inv.env)
        return finish(self, task, ws, res, t0)

    def to_trajectory(self, raw: list[dict[str, Any]]) -> TrajectoryRecord:
        steps: list[StepRecord] = []
        tokens = TokenUsage()
        cap = False
        error = ""
        think = ""
        response = ""
        ctx_peak = 0
        pending: dict[str, dict[str, Any]] = {}

        for ev in raw:
            t = ev.get("type")
            if t == "assistant":
                # each assistant message carries THIS call's usage (not cumulative); its full prompt
                # (input + cache_read + cache_creation) is that turn's window occupancy, so the max
                # over turns is the peak occupancy
                mu = ev.get("message", {}).get("usage") or {}
                ctx_peak = max(ctx_peak, mu.get("input_tokens", 0)
                               + mu.get("cache_read_input_tokens", 0)
                               + mu.get("cache_creation_input_tokens", 0))
                for b in (ev.get("message", {}).get("content") or []):
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "thinking":
                        think += (b.get("thinking") or "") + "\n"
                    elif bt == "text":
                        response += (b.get("text") or "") + "\n"
                    elif bt == "tool_use":
                        pending[str(b.get("id"))] = {"name": b.get("name"), "input": b.get("input")}
            elif t == "user":
                tools: list[ToolEvent] = []
                for b in (ev.get("message", {}).get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        call = pending.pop(str(b.get("tool_use_id")), {})
                        tools.append(ToolEvent(
                            name=call.get("name", "unknown"),
                            args=args_str(call.get("input")),
                            is_error=bool(b.get("is_error")),
                            output=content_text(b.get("content")),
                        ))
                if tools or think or response:
                    steps.append(StepRecord(
                        index=len(steps), think=think.strip(),
                        response=response.strip(), tools=tools,
                    ))
                    think, response = "", ""
            elif t == "result":
                u = ev.get("usage") or {}
                tokens = TokenUsage(
                    input=(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                           + u.get("cache_creation_input_tokens", 0)),
                    cached_input=u.get("cache_read_input_tokens", 0),
                    output=u.get("output_tokens", 0),
                    reasoning=0,
                    context=ctx_peak,
                )
                if ev.get("subtype") != "success":
                    cap = True
                    error = str(ev.get("subtype") or "error")
        if think or response:
            steps.append(StepRecord(
                index=len(steps), think=think.strip(), response=response.strip()))
        return TrajectoryRecord(steps=steps, tokens=tokens, cap_hit=cap, error=error)
