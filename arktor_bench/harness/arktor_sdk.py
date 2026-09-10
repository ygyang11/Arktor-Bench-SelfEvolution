from __future__ import annotations

import json
import shlex
import time
from typing import Any

from arktor_bench.config import HarnessInvocation, get_config
from arktor_bench.harness.arktor import ArktorAdapter
from arktor_bench.harness.base import RunResult, finish
from arktor_bench.models import TaskSpec
from arktor_bench.sandbox.workspace import Workspace


def parse_arktor_sdk_ndjson(
    text: str,
    *,
    exit_code: int | None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_result = False

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid NDJSON at stdout line {line_no}: {e.msg}") from e
        if not isinstance(event, dict):
            raise ValueError(f"stdout line {line_no} must be a JSON object")

        event_type = event.get("type")
        if event_type not in {"step", "result"}:
            raise ValueError(f"unsupported event type at stdout line {line_no}")
        if seen_result:
            raise ValueError("result must be the final stdout event")
        if event_type == "result":
            if not isinstance(event.get("is_error"), bool):
                raise ValueError("result.is_error must be a boolean")
            seen_result = True
        events.append(event)

    if exit_code == 0 and not seen_result:
        raise ValueError("successful exit requires a final result event")

    return events


class ArktorSdkAdapter(ArktorAdapter):
    def __init__(self, method: str) -> None:
        self.method = method
        self.name = f"arktor_sdk.{method}"

    async def run(
        self,
        task: TaskSpec,
        ws: Workspace,
        inv: HarnessInvocation,
    ) -> RunResult:
        env = dict(inv.env)
        if inv.model:
            env["ARKTOR_LLM_MODEL"] = inv.model
        cmd = shlex.join([
            *inv.command,
            "-p",
            task.prompt,
            "--output-format",
            "json",
        ])
        t0 = time.monotonic()
        res = await ws.execute(cmd, timeout=get_config().wall_s, env=env)
        events = parse_arktor_sdk_ndjson(res.stdout, exit_code=res.exit_code)
        return finish(self, task, ws, res, t0, events=events)
