from __future__ import annotations

from pathlib import Path

from arktor_bench.harness.base import parse_ndjson
from arktor_bench.harness.codex import CodexAdapter

_FIX = Path(__file__).parent / "fixtures"


def test_splits_items_and_usage() -> None:
    raw = parse_ndjson((_FIX / "codex_basic.jsonl").read_text())
    traj = CodexAdapter().to_trajectory(raw)
    assert len(traj.steps) == 2
    s0 = traj.steps[0]
    assert s0.think == "Plan the file"
    assert s0.tools[0].name == "command_execution"
    assert s0.tools[0].output == "added"
    assert not s0.tools[0].is_error
    assert traj.steps[1].response == "All set."
    assert (traj.tokens.input, traj.tokens.cached_input,
            traj.tokens.output, traj.tokens.reasoning) == (900, 100, 250, 80)


def test_turn_failed_sets_cap() -> None:
    traj = CodexAdapter().to_trajectory([{"type": "turn.failed", "error": {"message": "exceeded steps"}}])
    assert traj.cap_hit is True
    assert "exceeded steps" in traj.error


def test_unknown_item_type_falls_back() -> None:
    traj = CodexAdapter().to_trajectory([
        {"type": "item.completed",
         "item": {"id": "x", "type": "mystery_tool", "status": "completed", "output": "weird"}},
    ])
    assert len(traj.steps) == 1
    tool = traj.steps[0].tools[0]
    assert tool.name == "mystery_tool"
    assert tool.output == "weird"
    assert not tool.is_error
