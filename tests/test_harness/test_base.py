from __future__ import annotations

import time
from pathlib import Path

from arktor_bench.harness.base import args_str, finish, parse_ndjson
from arktor_bench.harness.codex import CodexAdapter
from arktor_bench.models import (
    Complexity,
    Domain,
    Layer,
    ModelCapability,
    TaskLabels,
    TaskSpec,
)
from arktor_bench.sandbox.backend import ExecuteResult
from arktor_bench.sandbox.workspace import Workspace


def _task() -> TaskSpec:
    return TaskSpec(
        id="T", name="t", dir=Path("."),
        labels=TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli",
                          model_capability=[ModelCapability.CODE], harness_focus=[Layer.LOOP],
                          complexity=Complexity.LINEAR),
        prompt="p")


def test_parse_ndjson_skips_blank_and_bad() -> None:
    text = '{"a": 1}\n\n  \nnot json\n{"b": 2}\n'
    assert parse_ndjson(text) == [{"a": 1}, {"b": 2}]


def test_args_str_forms() -> None:
    assert args_str(None) == ""
    assert args_str("raw") == "raw"
    assert args_str({"k": "v"}) == '{"k": "v"}'


def test_finish_sets_wall_ms_and_cap_on_no_exit(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, None)  # backend unused by finish / classify_produced
    res = ExecuteResult(exit_code=None, stdout="", stderr="timeout after 30s")
    rr = finish(CodexAdapter(), _task(), ws, res, time.monotonic() - 0.01)
    assert rr.trajectory.cap_hit is True            # None exit -> cap
    assert rr.trajectory.error == "timeout after 30s"
    assert rr.trajectory.wall_ms >= 0
