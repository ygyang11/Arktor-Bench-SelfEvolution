from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_stream_carries_no_context() -> None:
    raw = parse_ndjson((_FIX / "codex_basic.jsonl").read_text())
    traj = CodexAdapter().to_trajectory(raw)
    # exec --json only carries the cumulative turn total; occupancy is read from the rollout in run()
    assert traj.tokens.input == 900
    assert traj.tokens.context == 0


class _FakeWS:
    def __init__(self, stdout: str) -> None:
        self._stdout = stdout

    async def execute(self, cmd: str, timeout: float | None = None,
                      env: dict | None = None) -> object:
        return SimpleNamespace(stdout=self._stdout, stderr="", exit_code=0)


async def test_usage_from_rollout_is_peak_and_totals() -> None:
    def line(inp: int, tot: int) -> str:
        return ('{"type":"event_msg","payload":{"type":"token_count","info":{'
                f'"last_token_usage":{{"input_tokens":{inp}}},'
                f'"total_token_usage":{{"input_tokens":{tot},"output_tokens":7,"total_tokens":{tot}}},'
                '"model_context_window":258400}}}')
    rollout = "\n".join([line(30000, 100000), line(95000, 500000), line(80000, 700000),
                         '{"type":"other","payload":{}}'])
    peak, totals = await CodexAdapter._usage_from_rollout(_FakeWS(rollout), {})
    assert peak == 95000                      # peak last_token_usage.input_tokens (window occupancy)
    assert totals["input_tokens"] == 700000   # largest total_token_usage — recovers a capped run


async def test_usage_from_rollout_empty_is_zero() -> None:
    peak, totals = await CodexAdapter._usage_from_rollout(_FakeWS(""), {})
    assert peak == 0 and totals == {}


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
