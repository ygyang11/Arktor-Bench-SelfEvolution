from __future__ import annotations

from pathlib import Path

from arktor_bench.harness.arktor import ArktorAdapter
from arktor_bench.harness.base import parse_ndjson

_FIX = Path(__file__).parent / "fixtures"


def _traj(name: str) -> object:
    raw = parse_ndjson((_FIX / name).read_text())
    return ArktorAdapter().to_trajectory(raw)


def test_maps_steps_and_usage() -> None:
    traj = _traj("arktor_basic.jsonl")
    assert len(traj.steps) == 2
    s0 = traj.steps[0]
    assert s0.think == "I should create the file"
    assert s0.response == "Creating todo.py"
    assert s0.tools[0].name == "write_file"
    assert s0.tools[0].output == "wrote 1 file"
    assert not s0.tools[0].is_error
    assert '"path": "todo.py"' in s0.tools[0].args
    assert traj.steps[1].tools == []
    assert (traj.tokens.input, traj.tokens.cached_input,
            traj.tokens.output, traj.tokens.reasoning) == (1200, 400, 300, 120)
    # context is the CLI's window field (input_tokens), NOT the cumulative prompt_tokens (1200)
    assert traj.tokens.context == 950
    assert not traj.cap_hit


def test_context_never_falls_back_to_cumulative() -> None:
    # a run with big cumulative usage but no `context` field must NOT report cumulative as occupancy
    traj = ArktorAdapter().to_trajectory([
        {"type": "result", "is_error": False, "usage": {"prompt_tokens": 999999}},
    ])
    assert traj.tokens.input == 999999
    assert traj.tokens.context == 0


def test_result_is_error_sets_cap() -> None:
    traj = ArktorAdapter().to_trajectory([
        {"type": "step", "index": 0, "thought": "t", "response": "r", "action": [], "observation": []},
        {"type": "result", "is_error": True, "error": "boom", "usage": {}},
    ])
    assert traj.cap_hit is True
    assert traj.error == "boom"


async def test_capped_run_recovers_context_from_session() -> None:
    # a killed run never emits `result`; occupancy is read back from the arktor session file
    from types import SimpleNamespace

    from arktor_bench.trajectory.record import TokenUsage, TrajectoryRecord

    class _WS:
        async def execute(self, cmd: str, timeout: float | None = None,
                          env: dict | None = None) -> object:
            return SimpleNamespace(stdout=(
                '{"session_id":"s","metadata":{"_call_snapshot":'
                '{"input_tokens":48000,"completion_tokens":900,"cache_read":40000}}}'),
                stderr="", exit_code=0)
    traj = TrajectoryRecord(steps=[], tokens=TokenUsage())
    await ArktorAdapter._context_from_session(_WS(), {}, "0123456789abcdef0123456789abcdef", traj)
    assert traj.tokens.context == 48000        # last call's input = window occupancy, recovered
    assert traj.tokens.output == 0             # snapshot is last-call only; cumulative stays unset
