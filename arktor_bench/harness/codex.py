from __future__ import annotations

import shlex
import time
from typing import Any

from arktor_bench.config import HarnessInvocation, get_config
from arktor_bench.harness.base import Adapter, RunResult, args_str, finish
from arktor_bench.models import TaskSpec
from arktor_bench.sandbox.workspace import Workspace
from arktor_bench.trajectory.record import StepRecord, TokenUsage, ToolEvent, TrajectoryRecord


class CodexAdapter(Adapter):
    name = "codex"

    async def run(self, task: TaskSpec, ws: Workspace, inv: HarnessInvocation) -> RunResult:
        model = f"-m {shlex.quote(inv.model)} " if inv.model else ""
        cmd = (
            f"codex exec --json --dangerously-bypass-approvals-and-sandbox "
            f"--skip-git-repo-check --ephemeral "
            f"{model}-- {shlex.quote(task.prompt)} </dev/null"
        )
        t0 = time.monotonic()
        res = await ws.execute(cmd, timeout=get_config().wall_s, env=inv.env)
        return finish(self, task, ws, res, t0)

    def to_trajectory(self, raw: list[dict[str, Any]]) -> TrajectoryRecord:
        steps: list[StepRecord] = []
        tokens = TokenUsage()
        cap = False
        errors: list[str] = []
        think = ""
        response = ""

        def flush(tool: ToolEvent | None) -> None:
            nonlocal think, response
            if tool is None and not (think or response):
                return
            steps.append(StepRecord(
                index=len(steps), think=think.strip(), response=response.strip(),
                tools=[tool] if tool else [],
            ))
            think, response = "", ""

        for ev in raw:
            t = ev.get("type")
            if t == "turn.completed":
                u = ev.get("usage") or {}
                tokens = TokenUsage(
                    input=u.get("input_tokens", 0),
                    cached_input=u.get("cached_input_tokens", 0),
                    output=u.get("output_tokens", 0),
                    reasoning=u.get("reasoning_output_tokens", 0),
                )
            elif t == "turn.failed":
                cap = True
                errors.append(str((ev.get("error") or {}).get("message", "")))
            elif t == "error":
                errors.append(str(ev.get("message", "")))
            elif t == "item.completed":
                it = ev.get("item", {})
                ity = it.get("type")
                if ity == "reasoning":
                    think += (it.get("text") or "") + "\n"
                elif ity == "agent_message":
                    response += (it.get("text") or "") + "\n"
                elif ity == "command_execution":
                    flush(ToolEvent(
                        name="command_execution",
                        args=str(it.get("command", "")),
                        is_error=str(it.get("exit_code")) != "0" or it.get("status") == "failed",
                        output=str(it.get("aggregated_output", "")),
                    ))
                elif ity == "file_change":
                    paths = ", ".join(
                        f"{c.get('kind')}:{c.get('path')}" for c in it.get("changes", [])
                    )
                    flush(ToolEvent(
                        name="file_change", args=paths,
                        is_error=it.get("status") == "failed", output=paths,
                    ))
                elif ity == "web_search":
                    flush(ToolEvent(name="web_search", args=str(it.get("query", "")),
                                    is_error=False, output="[results consumed server-side]"))
                elif ity == "mcp_tool_call":
                    flush(ToolEvent(
                        name=f"mcp:{it.get('server')}.{it.get('tool')}",
                        args=args_str(it.get("arguments")),
                        is_error=it.get("status") == "failed",
                        output=str(it.get("status", "")),
                    ))
                elif ity == "todo_list":
                    items = "\n".join(
                        f"[{'x' if i.get('completed') else ' '}] {i.get('text', '')}"
                        for i in it.get("items", [])
                    )
                    flush(ToolEvent(name="todo_list", args="", is_error=False, output=items))
                elif ity:
                    rest = {k: v for k, v in it.items() if k not in ("id", "type")}
                    flush(ToolEvent(
                        name=str(ity), args=args_str(rest),
                        is_error=it.get("status") == "failed",
                        output=str(
                            it.get("aggregated_output") or it.get("output")
                            or it.get("text") or ""),
                    ))
        flush(None)
        return TrajectoryRecord(
            steps=steps, tokens=tokens, cap_hit=cap, error="\n".join(e for e in errors if e),
        )
