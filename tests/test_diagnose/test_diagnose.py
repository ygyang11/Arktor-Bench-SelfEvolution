from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import arktor_bench.diagnose.diagnose as diag
from arktor_bench.config import BenchConfig
from arktor_bench.diagnose.diagnose import diagnose_dir, diagnose_run
from arktor_bench.llm import FatalLLMError
from arktor_bench.models import (
    AutoCheckCriterion,
    Complexity,
    CriterionResult,
    Domain,
    Layer,
    ModelCapability,
    OutcomeScore,
    TaskLabels,
    TaskSpec,
)
from arktor_bench.trajectory.record import StepRecord, TokenUsage, TrajectoryRecord

FakeFactory = Callable[[Sequence[Any]], Any]

_FINDING = {"findings": [{"criterion_ids": ["c_ok"],
                          "root_cause": "the shell tool's error buries stderr",
                          "attribution": {"layer": "tools", "tool": "error"}}]}


def _cfg() -> BenchConfig:
    return BenchConfig(judge={"model": "m", "base_url": "http://x", "api_key": "k"})


def _task() -> TaskSpec:
    return TaskSpec(
        id="T001_min", name="t", dir=Path("."),
        labels=TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli",
                          model_capability=[ModelCapability.CODE], harness_focus=[Layer.TOOLS],
                          complexity=Complexity.LINEAR),
        prompt="p", auto_checks=[AutoCheckCriterion(id="c_ok", desc="persist", weight=2.0)])


def _score(s: float, *, errored: bool = False) -> OutcomeScore:
    return OutcomeScore(task_id="T001_min", trial=0, results=[
        CriterionResult(id="c_ok", mode="auto", score=s, weight=2.0, message="m", errored=errored)])


def _traj() -> TrajectoryRecord:
    return TrajectoryRecord(steps=[StepRecord(index=0, response="did stuff")], tokens=TokenUsage())


async def test_no_deficiencies_yields_nothing(fake_llm: FakeFactory) -> None:
    out = await diagnose_run(_score(1.0), _traj(), _task(), "spec", fake_llm([]))
    assert out == []


async def test_recoverable_points_weighted_loss(fake_llm: FakeFactory) -> None:
    out = await diagnose_run(_score(0.5), _traj(), _task(), "spec", fake_llm([_FINDING]))
    assert len(out) == 1
    assert out[0].recoverable_points == pytest.approx(1.0)   # 2 * (1 - 0.5) / 1 cover
    assert out[0].attribution.tool is not None


async def test_drops_unknown_criterion_ids(fake_llm: FakeFactory) -> None:
    bad = {"findings": [{"criterion_ids": ["nope"], "root_cause": "x",
                         "attribution": {"layer": "loop"}}]}
    out = await diagnose_run(_score(0.5), _traj(), _task(), "spec", fake_llm([bad]))
    assert out == []                                     # finding referenced no known criterion


async def test_skips_errored_criteria(fake_llm: FakeFactory) -> None:
    out = await diagnose_run(_score(0.0, errored=True), _traj(), _task(), "spec", fake_llm([_FINDING]))
    assert out == []                                     # errored deficiency not diagnosed


async def test_diagnose_dir_writes_findings_per_cell(run_tree: Any, fake_llm: FakeFactory,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    cell = run_tree.cell("arktor", "T001_min", 0, score=_score(0.0), trajectory=_traj())
    await diagnose_dir(run_tree.out, fake_llm([_FINDING]))
    findings = json.loads((cell / "findings.json").read_text())
    assert len(findings) == 1 and findings[0]["criterion_ids"] == ["c_ok"]


async def test_skips_corrupt_score_json(run_tree: Any, fake_llm: FakeFactory,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    run_tree.cell("arktor", "T001_min", 0, score="{bad json", trajectory=_traj())
    await diagnose_dir(run_tree.out, fake_llm([_FINDING]))   # must not raise


async def test_fatal_llm_aborts_diagnose(run_tree: Any, fake_llm: FakeFactory,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "get_config", _cfg)
    run_tree.manifest(harnesses=["arktor"], trials=1, tasks=["T001_min"])
    run_tree.cell("arktor", "T001_min", 0, score=_score(0.0), trajectory=_traj())
    with pytest.raises(FatalLLMError):
        await diagnose_dir(run_tree.out, fake_llm([FatalLLMError("auth")]))


@pytest.mark.llm
async def test_diagnose_run_live_attributes_failure() -> None:
    from arktor_bench.config import get_config
    from arktor_bench.diagnose.levers import load_lever_spec
    from arktor_bench.llm import StructuredLLM
    from arktor_bench.models import Layer
    from arktor_bench.trajectory.record import ToolEvent

    llm = StructuredLLM(get_config().diagnose_endpoint, timeout=120.0)
    traj = TrajectoryRecord(
        steps=[StepRecord(index=0, response="ran the command",
                          tools=[ToolEvent(name="shell", args="python todo.py done",
                                           is_error=True, output="exit 0; no message")])],
        tokens=TokenUsage())
    findings = await diagnose_run(_score(0.0), traj, _task(), load_lever_spec(), llm)
    assert findings
    for f in findings:
        assert set(f.criterion_ids) <= {"c_ok"}                         # only the real failed criterion
        a = f.attribution
        assert (a.layer is not None) ^ (a.model is not None)            # exactly one side
        assert (a.layer == Layer.TOOLS) == (a.tool is not None)         # tools <-> sub-lever
        assert f.recoverable_points > 0
        assert len(f.root_cause) > 10                                   # a real explanation, not a stub
    # the loss attributed across findings can't exceed the criterion's recoverable points (weight 2, score 0)
    assert sum(f.recoverable_points for f in findings) <= 2.0 + 1e-9
