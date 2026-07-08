from __future__ import annotations

from pathlib import Path

from arktor_bench.harness.base import parse_ndjson
from arktor_bench.harness.claude_code import ClaudeCodeAdapter

_FIX = Path(__file__).parent / "fixtures"


def _traj() -> object:
    raw = parse_ndjson((_FIX / "claude_code_basic.jsonl").read_text())
    return ClaudeCodeAdapter().to_trajectory(raw)


def test_pairs_tool_use_result() -> None:
    traj = _traj()
    assert len(traj.steps) == 2
    s0 = traj.steps[0]
    assert s0.think == "Let me write the file"
    assert s0.response == "I'll create todo.py"
    assert s0.tools[0].name == "Write"               # paired by tool_use_id
    assert s0.tools[0].output == "File created"
    assert not s0.tools[0].is_error
    assert traj.steps[1].response == "Done"


def test_input_tokens_include_cache() -> None:
    traj = _traj()
    # CC input_tokens exclude cache -> adapter must add cache_read + cache_creation
    assert traj.tokens.input == 800                  # 500 + 200 + 100
    assert traj.tokens.cached_input == 200
    assert traj.tokens.output == 150


def test_context_is_peak_per_turn_not_sum() -> None:
    # each assistant message's input+cache is that turn's window occupancy; context = the peak
    def asst(inp: int, cr: int, cc: int) -> dict:
        return {"type": "assistant", "message": {
            "usage": {"input_tokens": inp, "cache_read_input_tokens": cr,
                      "cache_creation_input_tokens": cc},
            "content": [{"type": "text", "text": "x"}]}}
    traj = ClaudeCodeAdapter().to_trajectory([
        asst(100, 200, 0),     # 300
        asst(500, 300, 100),   # 900
        asst(50, 100, 0),      # 150
        {"type": "result", "subtype": "success", "usage": {}},
    ])
    assert traj.tokens.context == 900   # peak, not the last (150) and not the sum (1350)


def test_subtype_non_success_sets_cap() -> None:
    traj = ClaudeCodeAdapter().to_trajectory([
        {"type": "result", "subtype": "error_max_turns", "usage": {}},
    ])
    assert traj.cap_hit is True
    assert traj.error == "error_max_turns"
