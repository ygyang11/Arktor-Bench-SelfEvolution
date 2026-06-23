from __future__ import annotations

from pathlib import Path

from arktor_bench.grading.check import CheckFn
from arktor_bench.grading.judge import judge_task
from arktor_bench.llm import FatalLLMError, StructuredLLM
from arktor_bench.models import CriterionResult, OutcomeScore, Produced, TaskSpec
from arktor_bench.sandbox.workspace import Workspace


async def score_task(
    task: TaskSpec, checks: dict[str, CheckFn], ws: Workspace,
    produced: list[Produced], produced_root: Path, trial: int, judge: StructuredLLM,
) -> OutcomeScore:
    results: list[CriterionResult] = []
    if task.judge:
        try:
            results.extend(await judge_task(task, produced, produced_root, judge))
        except FatalLLMError:
            raise
        except Exception as e:  # noqa: BLE001 — judge hiccup voids judge only
            results.extend(CriterionResult(id=c.id, mode="judge", score=0.0, weight=c.weight,
                                           message=f"judge raised: {e}", errored=True)
                           for c in task.judge)
    for c in task.auto_checks:
        try:
            score, message = await checks[c.id](ws)
        except Exception as e:  # noqa: BLE001 — one bad check scores 0, not the cell
            score, message = 0.0, f"check raised: {e}"
        results.append(CriterionResult(
            id=c.id, mode="auto", score=max(0.0, min(1.0, score)), weight=c.weight, message=message,
        ))
    return OutcomeScore(task_id=task.id, trial=trial, results=results)
