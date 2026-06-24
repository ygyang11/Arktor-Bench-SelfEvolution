from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

import arktor_bench.report.aggregate as agg
from arktor_bench.config import BenchConfig
from arktor_bench.diagnose.levers import attr_label
from arktor_bench.models import Attribution, Finding, Layer, ModelCapability, ToolLever
from arktor_bench.report.aggregate import (
    RunFailure,
    _dedup,
    _impact,
    _merge,
    _MergeOut,
    _render_backlog,
    build_report,
)
from arktor_bench.run.scoreboard import CellMetrics
from arktor_bench.trajectory.record import TokenUsage

FakeFactory = Callable[[Sequence[Any]], Any]


def _cfg() -> BenchConfig:
    return BenchConfig(judge={"model": "m", "base_url": "http://x", "api_key": "k"})


def _finding(*, root: str = "cause A", task: str = "T001_min", rp: float = 1.0,
             layer: Layer | None = Layer.TOOLS, tool: ToolLever | None = ToolLever.ERROR,
             model: ModelCapability | None = None) -> Finding:
    return Finding(task_id=task, trial=0, criterion_ids=["c_ok"],
                   attribution=Attribution(layer=layer, tool=tool, model=model),
                   root_cause=root, recoverable_points=rp)


def test_dedup_folds_identical_root_cause() -> None:
    groups = _dedup([_finding(root="A"), _finding(root="A"), _finding(root="B")])
    assert [len(g) for g in groups] == [2, 1]            # identical causes folded, order kept


def test_attr_label_formats_tool_sublever() -> None:
    assert attr_label(Attribution(layer=Layer.TOOLS, tool=ToolLever.ERROR)) == "tools.error"
    assert attr_label(Attribution(layer=Layer.LOOP)) == "loop"


def test_attr_label_formats_model_capability() -> None:
    assert attr_label(Attribution(model=ModelCapability.REASONING)) == "model.reasoning"


def test_impact_normalizes_per_task_per_trial() -> None:
    members = [_finding(rp=1.0, task="T1")]
    assert _impact(members, {"T1": 2.0}, 1, 1) == pytest.approx(0.5)    # (1/1)/2 /1
    assert _impact(members, {"T1": 2.0}, 1, 2) == pytest.approx(0.25)   # per-trial halves it


async def test_merge_assigns_findings_to_clusters(fake_llm: FakeFactory) -> None:
    group = [_finding(root="A"), _finding(root="B")]
    out = {"clusters": [{"summary": "shared", "fix": "do x"}], "assignments": [0, 0]}
    issues = await _merge(group, {"T001_min": 3.0}, 1, 1, fake_llm([out]), "code map")
    assert len(issues) == 1
    assert issues[0].recurrence == 2 and len(issues[0].findings) == 2


def test_merge_out_rejects_bad_assignments() -> None:
    with pytest.raises(ValidationError):                 # index out of range
        _MergeOut.model_validate({"clusters": [{"summary": "s", "fix": "f"}], "assignments": [5]})
    with pytest.raises(ValidationError):                 # length != n_findings (from context)
        _MergeOut.model_validate(
            {"clusters": [{"summary": "s", "fix": "f"}], "assignments": [0, 0]},
            context={"n_findings": 1})


def test_run_failures_section() -> None:
    md = _render_backlog("arktor", [],
                         [RunFailure(task="T001_min", trial="0", error="boom 503", ran=False)])
    assert "## Run failures" in md and "boom 503" in md and "produced no trajectory" in md


def test_degraded_cells_fold_into_run_failures() -> None:
    md = _render_backlog("arktor", [],
                         [RunFailure(task="T001_min", trial="1", error="exit code -15", ran=True)])
    assert "## Run failures" in md and "exit code -15" in md   # one section, sub-point distinguishes
    assert "exited non-clean" in md and "## Degraded cells" not in md


def test_render_backlog_shows_empty_section_explicitly() -> None:
    from arktor_bench.report.aggregate import Issue
    issue = Issue(attribution=Attribution(model=ModelCapability.CODE), summary="s", fix="f",
                  impact=0.1, recurrence=1,
                  findings=[_finding(layer=None, tool=None, model=ModelCapability.CODE)])
    md = _render_backlog("arktor", [issue], [])
    assert "## Harness issues" in md and "None found" in md   # empty harness section kept + noted
    assert "## Model capabilities" in md and "model.code" in md


async def test_build_report_writes_backlog_per_harness(run_tree: Any, fake_llm: FakeFactory,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=3, wall_ms=1, cap_hit=False)
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics, findings=[_finding()])
    out = {"clusters": [{"summary": "shared", "fix": "do x"}], "assignments": [0]}
    await build_report(run_tree.out, fake_llm([out]))
    data = json.loads((run_tree.out / "arktor" / "backlog.json").read_text())
    assert len(data["issues"]) == 1
    assert (run_tree.out / "arktor" / "backlog.md").is_file()


async def test_build_report_flags_degraded_cell(run_tree: Any, fake_llm: FakeFactory,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=4, wall_ms=1, cap_hit=True,  # ran but killed
                          error="exit code -15")
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics)   # scored, no findings -> clean backlog
    await build_report(run_tree.out, fake_llm([]))
    md = (run_tree.out / "arktor" / "backlog.md").read_text()
    assert "## Run failures" in md and "exit code -15" in md and "exited non-clean" in md
    data = json.loads((run_tree.out / "arktor" / "backlog.json").read_text())
    assert len(data["run_failures"]) == 1 and data["run_failures"][0]["ran"] is True


async def test_skips_corrupt_findings_json(run_tree: Any, fake_llm: FakeFactory,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=2, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=3, wall_ms=1, cap_hit=False)
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics, findings="{bad json")  # corrupt -> skipped
    run_tree.cell("arktor", "T001_min", 1, metrics=metrics, findings=[_finding()])  # valid -> kept
    out = {"clusters": [{"summary": "shared", "fix": "do x"}], "assignments": [0]}
    await build_report(run_tree.out, fake_llm([out]))    # corrupt cell skipped, valid one still reported
    data = json.loads((run_tree.out / "arktor" / "backlog.json").read_text())
    assert len(data["issues"]) == 1


async def test_writes_empty_backlog_for_clean_harness(run_tree: Any, fake_llm: FakeFactory,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    # a harness that scored clean (no findings, no run-failures) still gets a backlog stating so
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=3, wall_ms=1, cap_hit=False)
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics)    # no findings.json -> clean
    await build_report(run_tree.out, fake_llm([]))
    assert "No issues found" in (run_tree.out / "arktor" / "backlog.md").read_text()
    data = json.loads((run_tree.out / "arktor" / "backlog.json").read_text())
    assert data["issues"] == [] and data["run_failures"] == []


async def test_bucket_failure_degrades_to_empty(run_tree: Any, fake_llm: FakeFactory,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=3, wall_ms=1, cap_hit=False)
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics, findings=[_finding()])
    await build_report(run_tree.out, fake_llm([RuntimeError("merge boom")]))
    data = json.loads((run_tree.out / "arktor" / "backlog.json").read_text())
    assert data["issues"] == []                          # bad bucket degraded, report still written


async def test_fatal_llm_aborts_report(run_tree: Any, fake_llm: FakeFactory,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from arktor_bench.llm import FatalLLMError
    monkeypatch.setattr(agg, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    metrics = CellMetrics(tokens=TokenUsage(), steps=3, wall_ms=1, cap_hit=False)
    run_tree.cell("arktor", "T001_min", 0, metrics=metrics, findings=[_finding()])
    with pytest.raises(FatalLLMError):
        await build_report(run_tree.out, fake_llm([FatalLLMError("auth")]))
