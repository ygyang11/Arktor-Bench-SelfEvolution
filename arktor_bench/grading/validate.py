from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from arktor_bench.grading.check import CheckFn
from arktor_bench.grading.score import score_task
from arktor_bench.llm import StructuredLLM
from arktor_bench.models import OutcomeScore, Produced, TaskSpec
from arktor_bench.sandbox.workspace import is_artifact, scratch_workspace
from arktor_bench.spec.loader import load


class ValidityResult(BaseModel):
    task_id: str
    ok: bool
    oracle: float | None
    null: float
    issues: list[str]


async def _score_dir(
    task: TaskSpec, checks: dict[str, CheckFn], src: Path | None, judge: StructuredLLM,
) -> OutcomeScore:
    async with scratch_workspace(src) as ws:
        produced = ([Produced(path=str(rel), status="created")
                     for p in ws.root.rglob("*")
                     if p.is_file() and is_artifact(rel := p.relative_to(ws.root))] if src else [])
        return await score_task(task, checks, ws, produced, ws.root, 0, judge)


def validate_static(task_dir: Path) -> ValidityResult:
    try:
        task, _ = load(task_dir)
    except ValueError as e:
        return ValidityResult(
            task_id=task_dir.name, ok=False, oracle=None, null=0.0, issues=[str(e)])
    return ValidityResult(task_id=task.id, ok=True, oracle=None, null=0.0, issues=[])


async def validate_judge(task_dir: Path, judge: StructuredLLM) -> ValidityResult:
    task, checks = load(task_dir)
    ref = task_dir / "reference"
    issues: list[str] = []
    oracle: float | None = None
    if ref.is_dir() and any(p.is_file() for p in ref.rglob("*")):
        o = await _score_dir(task, checks, ref, judge)
        oracle = o.score
        if o.score < 0.95:
            issues.append(f"oracle {o.score:.2f} < 0.95")
        for r in o.results:
            if r.score < 1.0:
                issues.append(f"criterion {r.id} scored {r.score:.2f}: {r.message or 'no message'}")
    null = await _score_dir(task, checks, None, judge)
    if null.score > 0.05:
        issues.append(f"null {null.score:.2f} > 0.05")
    return ValidityResult(
        task_id=task.id, ok=not issues, oracle=oracle, null=null.score, issues=issues)
