from __future__ import annotations

from arktor_bench.trajectory.record import TrajectoryRecord
from arktor_bench.utils.token_counter import clip_tokens

_THINK, _OUTPUT = 4000, 5000
_SAMPLE = 4096


def _media(s: str) -> bool:
    chunk = s[:_SAMPLE]
    return "\x00" in chunk or chunk.count("�") > len(chunk) // 8


def _field(s: str, limit: int | None) -> str:
    if _media(s):
        return f"[non-text / media, {len(s)} chars omitted]"
    return clip_tokens(s, limit) if limit else s


def compact_trajectory(traj: TrajectoryRecord) -> str:
    blocks: list[str] = []
    seen: dict[tuple[str, str, str], int] = {}
    for s in traj.steps:
        lines = [f"### Step {s.index}"]
        if s.think:
            lines.append(f"Think: {_field(s.think, _THINK)}")
        if s.response:
            lines.append(f"Response: {_field(s.response, None)}")
        if s.tools:
            lines.append("Tools:")
            for t in s.tools:
                sig = (t.name, t.args, t.output)
                status = "ERROR" if t.is_error else "ok"
                if sig in seen:
                    body = f"(same as step {seen[sig]})"
                else:
                    seen[sig] = s.index
                    body = t.output if t.is_error else _field(t.output, _OUTPUT)
                lines.append(f"  - {t.name}({_field(t.args, None)}) -> {status}: {body}")
        blocks.append("\n".join(lines))
    tail: list[str] = []
    if traj.cap_hit:
        tail.append("The run was stopped before completing "
                    "(the CLI reported a terminal failure / non-success).")
    if traj.error:
        tail.append(f"Run-level failure the agent never saw: {_field(traj.error, _THINK)}")
    if tail:
        blocks.append("## Run outcome\n" + "\n".join(tail))
    return "\n\n".join(blocks)
