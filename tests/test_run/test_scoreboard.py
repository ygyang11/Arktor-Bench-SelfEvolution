from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arktor_bench.models import (
    Complexity,
    CriterionResult,
    Domain,
    Layer,
    ModelCapability,
    OutcomeScore,
    TaskLabels,
)
from arktor_bench.run.scoreboard import (
    CellMetrics,
    HarnessBoard,
    Scoreboard,
    _cell,
    _efficiency,
    _render,
    _slice,
    build_board,
    build_summary,
    write_scoreboard,
)
from arktor_bench.trajectory.record import TokenUsage


def _score(s_ok: float, s_q: float, *, q_errored: bool = False) -> OutcomeScore:
    return OutcomeScore(task_id="T001_min", trial=0, results=[
        CriterionResult(id="c_ok", mode="auto", score=s_ok, weight=2.0),
        CriterionResult(id="c_quality", mode="judge", score=s_q, weight=1.0, errored=q_errored),
    ])


def _metrics(steps: int = 5, cap: bool = False, tin: int = 100, tout: int = 50) -> CellMetrics:
    return CellMetrics(tokens=TokenUsage(input=tin, output=tout), steps=steps, wall_ms=10, cap_hit=cap)


def _labels(c: Complexity) -> TaskLabels:
    return TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli",
                      model_capability=[ModelCapability.CODE], harness_focus=[Layer.LOOP], complexity=c)


def test_build_summary_mean_and_std() -> None:
    s = build_summary("T001_min", [_score(1.0, 1.0), _score(0.0, 0.0)],
                      [_metrics(), _metrics()])
    assert s.mean == pytest.approx(0.5)
    assert s.std == pytest.approx(0.5)                   # pstdev([1,0])
    means = {c.id: c.mean for c in s.criteria}
    assert means == {"c_ok": pytest.approx(0.5), "c_quality": pytest.approx(0.5)}


def test_summary_skips_errored_criteria() -> None:
    s = build_summary("T001_min", [_score(1.0, 0.0, q_errored=True)], [_metrics()])
    ids = {c.id for c in s.criteria}
    assert ids == {"c_ok"}                               # errored judge criterion excluded


def test_slice_buckets_by_label() -> None:
    labels = {"A": _labels(Complexity.LINEAR), "B": _labels(Complexity.COMPOSITE),
              "C": _labels(Complexity.LINEAR)}
    out = _slice({"A": 1.0, "B": 0.0, "C": 0.5}, labels, "complexity")
    assert out["linear"] == pytest.approx(0.75)          # mean of A,C
    assert out["composite"] == pytest.approx(0.0)


def test_efficiency_and_cap_rate() -> None:
    eff = _efficiency([_metrics(steps=4, cap=True), _metrics(steps=6, cap=False)])
    assert eff.steps == pytest.approx(5.0)
    assert eff.cap_hit_rate == pytest.approx(0.5)
    assert eff.tokens_total == pytest.approx(150.0)


def _board(per_task: dict[str, float], trials: dict[str, int]) -> HarnessBoard:
    return build_board(per_task, {k: 100.0 for k in per_task}, trials,
                       {}, [_metrics()])


def test_render_marks_missing_cell() -> None:
    sb = Scoreboard(tasks=["T1", "T2"], trials=2, harnesses={
        "arktor": _board({"T1": 0.8}, {"T1": 2}),        # T2 missing
    })
    md = _render(sb)
    assert "| T2 |" in md and "-" in md


def test_render_flags_dropped_trials() -> None:
    board = _board({"T1": 0.8}, {"T1": 1})
    assert "(1/3)" in _cell(board, "T1", 3)              # mean built on fewer trials
    assert "(1/3)" not in _cell(board, "T1", 1)


def test_skips_corrupt_cell_json(run_tree: Any) -> None:
    run_tree.manifest(harnesses=["arktor"], trials=2, tasks=["T001_min"])
    run_tree.cell("arktor", "T001_min", 0, score=_score(1.0, 1.0), metrics=_metrics())
    run_tree.cell("arktor", "T001_min", 1, score="{bad json", metrics=_metrics())
    sb = write_scoreboard(run_tree.out)
    assert sb.harnesses["arktor"].per_task_trials["T001_min"] == 1   # corrupt cell dropped
    assert (run_tree.out / "scoreboard.md").is_file()
