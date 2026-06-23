from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from arktor_bench.grading.check import CheckFn
from arktor_bench.grading.score import score_task
from arktor_bench.llm import FatalLLMError
from arktor_bench.models import (
    AutoCheckCriterion,
    Complexity,
    Domain,
    JudgeCriterion,
    JudgeLevel,
    Layer,
    ModelCapability,
    Produced,
    TaskLabels,
    TaskSpec,
)
from arktor_bench.sandbox.backend import Backend
from arktor_bench.sandbox.workspace import Workspace

FakeFactory = Callable[[Sequence[Any]], Any]


def _labels() -> TaskLabels:
    return TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli", model_capability=[ModelCapability.CODE],
                      harness_focus=[Layer.INSTRUCTIONS], complexity=Complexity.LINEAR)


def _task(tmp: Path, *, auto: list[AutoCheckCriterion], judge: list[JudgeCriterion]) -> TaskSpec:
    return TaskSpec(id="T", name="t", dir=tmp, labels=_labels(), prompt="do it",
                    auto_checks=auto, judge=judge)


def _check(score: float) -> CheckFn:
    async def fn(ws: Workspace) -> tuple[float, str]:
        return score, "m"
    return fn


def _ws(tmp: Path) -> Workspace:
    return Workspace(tmp, cast(Backend, None))


def _judge_crit() -> JudgeCriterion:
    return JudgeCriterion(id="q", desc="quality", weight=1.0,
                          levels=[JudgeLevel(score=0.0, desc="lo"), JudgeLevel(score=1.0, desc="hi")])


async def test_clamps_auto_score(tmp_path: Path, fake_llm: FakeFactory) -> None:
    task = _task(tmp_path, auto=[AutoCheckCriterion(id="hi", desc="d", weight=1.0),
                                 AutoCheckCriterion(id="lo", desc="d", weight=1.0)], judge=[])
    checks: dict[str, CheckFn] = {"hi": _check(1.5), "lo": _check(-0.5)}
    out = await score_task(task, checks, _ws(tmp_path), [], tmp_path, 0, fake_llm([]))
    by = {r.id: r.score for r in out.results}
    assert by == {"hi": 1.0, "lo": 0.0}


async def test_appends_judge_results(tmp_path: Path, fake_llm: FakeFactory) -> None:
    task = _task(tmp_path, auto=[AutoCheckCriterion(id="a", desc="d", weight=1.0)], judge=[_judge_crit()])
    llm = fake_llm([{"q": {"rationale": "great", "level": 2}}])
    out = await score_task(task, {"a": _check(1.0)}, _ws(tmp_path), [], tmp_path, 0, llm)
    judged = [r for r in out.results if r.mode == "judge"]
    assert len(judged) == 1
    assert judged[0].id == "q" and judged[0].score == 1.0 and not judged[0].errored
    assert any(r.mode == "auto" for r in out.results)


async def test_judge_failure_voids_judge_only_auto_still_runs(tmp_path: Path, fake_llm: FakeFactory) -> None:
    task = _task(tmp_path, auto=[AutoCheckCriterion(id="a", desc="d", weight=1.0)], judge=[_judge_crit()])
    llm = fake_llm([RuntimeError("judge boom")])
    out = await score_task(task, {"a": _check(1.0)}, _ws(tmp_path), [], tmp_path, 0, llm)
    auto = next(r for r in out.results if r.mode == "auto")
    judge = next(r for r in out.results if r.mode == "judge")
    assert auto.score == 1.0 and not auto.errored          # auto unaffected
    assert judge.errored and judge.score == 0.0


async def test_judge_failure_marks_errored(tmp_path: Path, fake_llm: FakeFactory) -> None:
    task = _task(tmp_path, auto=[], judge=[_judge_crit()])
    out = await score_task(task, {}, _ws(tmp_path), [], tmp_path, 0, fake_llm([RuntimeError("x")]))
    assert out.results[0].errored
    assert "judge raised" in out.results[0].message
    assert out.score == 0.0                                # errored-only cell -> 0


async def test_fatal_llm_propagates(tmp_path: Path, fake_llm: FakeFactory) -> None:
    task = _task(tmp_path, auto=[AutoCheckCriterion(id="a", desc="d", weight=1.0)], judge=[_judge_crit()])
    llm = fake_llm([FatalLLMError("auth")])
    with pytest.raises(FatalLLMError):
        await score_task(task, {"a": _check(1.0)}, _ws(tmp_path), [], tmp_path, 0, llm)


@pytest.mark.llm
async def test_score_task_live_combines_auto_and_judge(tmp_path: Path) -> None:
    # real judge endpoint, NO docker: a host-side auto check + a real judge merge into one OutcomeScore
    from arktor_bench.config import get_config
    from arktor_bench.llm import StructuredLLM

    async def has_answer(ws: Workspace) -> tuple[float, str]:
        return (1.0, "") if (ws.read_text("answer.txt") or "").strip() == "42" else (0.0, "not 42")

    task = _task(
        tmp_path,
        auto=[AutoCheckCriterion(id="has_answer", desc="answer.txt is 42", weight=1.0)],
        judge=[JudgeCriterion(id="explains", desc="notes.txt states the final answer is 42", weight=1.0,
                              levels=[JudgeLevel(score=0.0, desc="does not state the answer is 42"),
                                      JudgeLevel(score=1.0, desc="states the final answer is 42")])])
    (tmp_path / "answer.txt").write_text("42")
    (tmp_path / "notes.txt").write_text("After working it through, the final answer is 42.")
    llm = StructuredLLM(get_config().judge_endpoint, timeout=120.0)
    out = await score_task(task, {"has_answer": has_answer}, _ws(tmp_path),
                           [Produced(path="notes.txt", status="created")], tmp_path, 0, llm)
    by = {r.id: r for r in out.results}
    assert by["has_answer"].mode == "auto" and by["has_answer"].score == 1.0   # host-side grader ran
    assert by["explains"].mode == "judge" and by["explains"].score == 1.0       # real judge read notes.txt
    assert out.score == 1.0                                                     # weighted mean of both
