from __future__ import annotations

from arktor_bench.diagnose.levers import ARKTOR_ANNEX, load_annex, load_lever_spec


def test_load_annex_arktor_vs_external() -> None:
    assert load_annex("arktor") == ARKTOR_ANNEX
    assert "agent_app/tools" in load_annex("arktor")     # real code map
    external = load_annex("codex")
    assert external == load_annex("claude_code")         # both share _EXTERNAL
    assert "third-party" in external                     # lever-level, no code map
    assert load_annex("unknown") == ""
    assert "harness" in load_lever_spec()
