from __future__ import annotations

import json
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
        sid = uuid.uuid4().hex
        cmd = (f"arktor -p {shlex.quote(task.prompt)} "
               f"--output-format json -s {sid}")
        t0 = time.monotonic()
        res = await ws.execute(cmd, timeout=get_config().wall_s, env=env)
        rr = finish(self, task, ws, res, t0)
        if not rr.trajectory.tokens.context:      # capped: the `result` event never arrived
            await self._context_from_session(ws, env, sid, rr.trajectory)
        return rr

    @staticmethod
    async def _context_from_session(ws: Workspace, env: dict[str, str], sid: str,
                                    traj: TrajectoryRecord) -> None:
        """A capped run never emits the final `result`, losing window occupancy. arktor persists
        the last call's usage in its session file under `metadata._call_snapshot.input_tokens`,
        which is the same last-call input the `result` event reports as `context.input_tokens` on a
        finished run. Recover ONLY that occupancy by the session id we passed with -s; the snapshot
        is per-last-call, so cumulative input/output are not recoverable from it and stay unset."""
        dashed = str(uuid.UUID(sid))
        cmd = ('d="$HOME/.arktor/sessions"; '
               f'f="$d/{dashed}.json"; [ -f "$f" ] || f="$d/{sid}.json"; '
               f'[ -f "$f" ] || f=$(grep -laE "{sid}|{dashed}" "$d"/*.json 2>/dev/null | head -1); '
               'cat "$f" 2>/dev/null || true')
        try:
            out = (await ws.execute(cmd, timeout=30, env=env)).stdout
            snap = (json.loads(out).get("metadata") or {}).get("_call_snapshot") or {}
        except (ValueError, AttributeError):
            return
        occ = snap.get("input_tokens", 0)
        if isinstance(occ, int) and occ > 0:
            traj.tokens.context = occ

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
                u = ev.get("usage") or {}                 # usage is cumulative over the whole run
                tokens = TokenUsage(
                    input=u.get("prompt_tokens", 0),
                    cached_input=u.get("cache_read_tokens", 0),
                    output=u.get("completion_tokens", 0),
                    reasoning=u.get("reasoning_tokens", 0),
                    # window occupancy = the CLI's final input_tokens (last call's input); the
                    # cumulative prompt_tokens would inflate it, so it is NOT used as a fallback
                    context=(ev.get("context") or {}).get("input_tokens", 0),
                )
                if ev.get("is_error"):
                    cap = True
                    error = str(ev.get("error") or "")
        return TrajectoryRecord(steps=steps, tokens=tokens, cap_hit=cap, error=error)
