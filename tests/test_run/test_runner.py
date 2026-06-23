from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

import arktor_bench.run.runner as runner
from arktor_bench.config import BenchConfig, HarnessConfigs, HarnessInvocation
from arktor_bench.harness.base import Adapter, RunResult
from arktor_bench.llm import FatalLLMError
from arktor_bench.models import (
    AutoCheckCriterion,
    Complexity,
    CriterionResult,
    Domain,
    JudgeCriterion,
    JudgeLevel,
    Layer,
    ModelCapability,
    OutcomeScore,
    Produced,
    TaskLabels,
    TaskSpec,
)
from arktor_bench.run.runner import _persist, _persist_failure, run_matrix
from arktor_bench.run.scoreboard import CellMetrics
from arktor_bench.sandbox.workspace import Workspace
from arktor_bench.spec.loader import load_pack
from arktor_bench.trajectory.record import StepRecord, TokenUsage, TrajectoryRecord


def _task() -> TaskSpec:
    return TaskSpec(
        id="T", name="t", dir=Path("."),
        labels=TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli",
                          model_capability=[ModelCapability.CODE], harness_focus=[Layer.LOOP],
                          complexity=Complexity.LINEAR),
        prompt="p",
        auto_checks=[AutoCheckCriterion(id="a", desc="d", weight=1.0)],
        judge=[JudgeCriterion(id="j", desc="d", weight=1.0,
                              levels=[JudgeLevel(score=0.0, desc="lo"), JudgeLevel(score=1.0, desc="hi")])])


def _rr() -> RunResult:
    return RunResult(
        trajectory=TrajectoryRecord(steps=[StepRecord(index=0, response="hi")],
                                    tokens=TokenUsage(input=10, output=5), wall_ms=100),
        produced=[Produced(path="a.py", status="created")])


def test_persist_writes_cell_files(tmp_path: Path) -> None:
    score = OutcomeScore(task_id="T", trial=0,
                         results=[CriterionResult(id="a", mode="auto", score=1.0, weight=1.0)])
    _persist(tmp_path, _rr(), score)
    for name in ("trajectory.json", "score.json", "produced.json", "metrics.json"):
        assert (tmp_path / name).is_file()
    m = CellMetrics.model_validate_json((tmp_path / "metrics.json").read_text())
    assert m.steps == 1 and m.tokens.input == 10


def test_cell_failure_persists_zero(tmp_path: Path) -> None:
    cap = _persist_failure(tmp_path, _task(), 0, RuntimeError("boom"), None)
    assert cap is True                                   # no rr -> cap_hit
    score = OutcomeScore.model_validate_json((tmp_path / "score.json").read_text())
    assert score.score == 0.0
    assert all(r.errored for r in score.results)
    assert {r.id for r in score.results} == {"a", "j"}


def test_run_failure_sets_metrics_error(tmp_path: Path) -> None:
    _persist_failure(tmp_path, _task(), 0, RuntimeError("boom"), None)
    m = CellMetrics.model_validate_json((tmp_path / "metrics.json").read_text())
    assert m.steps == 0
    assert m.error is not None and "cell failed: boom" in m.error


async def test_fatal_llm_aborts_run(tmp_path: Path, tmp_pack: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BenchConfig(judge={"model": "m", "base_url": "http://x", "api_key": "k"},
                      harness=HarnessConfigs(arktor=HarnessInvocation(model="m", backend="docker")))
    monkeypatch.setattr(runner, "get_config", lambda: cfg)

    @asynccontextmanager
    async def fake_ws(task: Any, inv: Any) -> Any:
        yield Workspace(tmp_path, None)
    monkeypatch.setattr(runner, "task_workspace", fake_ws)

    class FatalAdapter(Adapter):
        name = "arktor"

        async def run(self, task: Any, ws: Any, inv: Any) -> RunResult:
            raise FatalLLMError("auth down")

        def to_trajectory(self, raw: Any) -> TrajectoryRecord:
            return TrajectoryRecord(steps=[], tokens=TokenUsage())

    tasks = load_pack("general", ["T001_min"], base=tmp_pack)
    with pytest.raises(FatalLLMError):
        await run_matrix(tasks, [FatalAdapter()], 1, tmp_path / "out")
