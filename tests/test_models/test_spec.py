from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from arktor_bench.models import (
    AutoCheckCriterion,
    Complexity,
    Domain,
    JudgeCriterion,
    JudgeLevel,
    Layer,
    ModelCapability,
    TaskLabels,
    TaskSpec,
    WorkspaceFile,
)


def _labels() -> TaskLabels:
    return TaskLabels(
        domain=Domain.SOFTWARE_ENGINEERING,
        subdomain="cli_tool",
        model_capability=[ModelCapability.CODE],
        harness_focus=[Layer.INSTRUCTIONS, Layer.TOOLS],
        complexity=Complexity.COMPOSITE,
    )


def test_workspace_file_rejects_absolute_and_escape() -> None:
    WorkspaceFile(source="assets/in.txt", dest="data/in.txt")
    with pytest.raises(ValidationError):
        WorkspaceFile(source="/etc/passwd", dest="x")
    with pytest.raises(ValidationError):
        WorkspaceFile(source="assets/in.txt", dest="../escape")


def test_judge_levels_strictly_ascend_to_one() -> None:
    JudgeCriterion(
        id="c", desc="d", weight=1.0,
        levels=[JudgeLevel(score=0.0, desc="lo"), JudgeLevel(score=0.5, desc="mid"),
                JudgeLevel(score=1.0, desc="hi")],
    )
    with pytest.raises(ValidationError):  # not ascending / duplicate
        JudgeCriterion(id="c", desc="d", weight=1.0,
                       levels=[JudgeLevel(score=0.5, desc="a"), JudgeLevel(score=0.5, desc="b")])
    with pytest.raises(ValidationError):  # top is not 1.0
        JudgeCriterion(id="c", desc="d", weight=1.0,
                       levels=[JudgeLevel(score=0.0, desc="a"), JudgeLevel(score=0.9, desc="b")])
    with pytest.raises(ValidationError):  # no levels
        JudgeCriterion(id="c", desc="d", weight=1.0, levels=[])


def test_task_total_weight() -> None:
    task = TaskSpec(
        id="T", name="t", dir=Path("."), labels=_labels(), prompt="do it",
        auto_checks=[AutoCheckCriterion(id="a", desc="d", weight=2.0)],
        judge=[JudgeCriterion(id="j", desc="d", weight=1.0,
                              levels=[JudgeLevel(score=0.0, desc="lo"),
                                      JudgeLevel(score=1.0, desc="hi")])],
    )
    assert task.total_weight == 3.0
