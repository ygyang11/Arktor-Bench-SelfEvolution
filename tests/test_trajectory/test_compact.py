from __future__ import annotations

from typing import Any

from arktor_bench.trajectory.compact import compact_trajectory


def test_clips_think_and_tool_output(sample_trajectory: Any) -> None:
    out = compact_trajectory(sample_trajectory)
    assert out.count("[...truncated") >= 2          # both the big think and the big tool output


def test_marks_media_field(sample_trajectory: Any) -> None:
    out = compact_trajectory(sample_trajectory)
    assert "[non-text / media" in out


def test_dedupes_repeated_output(sample_trajectory: Any) -> None:
    out = compact_trajectory(sample_trajectory)
    assert "(same as step 0)" in out


def test_appends_run_outcome_tail(sample_trajectory: Any) -> None:
    out = compact_trajectory(sample_trajectory)
    assert "## Run outcome" in out
    assert "transport failed 503" in out
    assert "stopped before completing" in out       # cap_hit message
