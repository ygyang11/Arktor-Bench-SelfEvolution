from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from arktor_bench.config import get_config
from arktor_bench.grading.judge import _gather, _load_text, judge_task
from arktor_bench.llm import StructuredLLM
from arktor_bench.models import (
    Complexity,
    Domain,
    JudgeCriterion,
    JudgeLevel,
    Layer,
    ModelCapability,
    Produced,
    TaskLabels,
    TaskSpec,
    WorkspaceFile,
)

FakeFactory = Callable[[Sequence[Any]], Any]


def _labels() -> TaskLabels:
    return TaskLabels(domain=Domain.SOFTWARE_ENGINEERING, subdomain="cli", model_capability=[ModelCapability.CODE],
                      harness_focus=[Layer.INSTRUCTIONS], complexity=Complexity.LINEAR)


def test_gather_filters_artifacts_by_glob() -> None:
    crit = JudgeCriterion(id="c", desc="d", weight=1.0, artifacts=["*.py"],
                          levels=[JudgeLevel(score=0.0, desc="lo"), JudgeLevel(score=1.0, desc="hi")])
    task = TaskSpec(id="T", name="t", dir=Path("."), labels=_labels(), prompt="p", judge=[crit],
                    workspace_files=[WorkspaceFile(source="assets/seed.py", dest="seed.py")])
    produced = [Produced(path="main.py", status="created"), Produced(path="notes.txt", status="created")]
    gathered = dict(_gather(task, produced))
    assert gathered == {"main.py": "created", "seed.py": "provided"}   # .txt filtered, seed kept as provided


def test_load_text_marks_binary(tmp_path: Path) -> None:
    (tmp_path / "bin").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "txt").write_text("hello text")
    assert "[binary file" in _load_text(tmp_path / "bin")
    assert _load_text(tmp_path / "txt") == "hello text"
    assert _load_text(tmp_path / "missing") == "[file not present]"


async def test_maps_level_to_score(tmp_path: Path, fake_llm: FakeFactory) -> None:
    crit = JudgeCriterion(id="q", desc="d", weight=2.0,
                          levels=[JudgeLevel(score=0.0, desc="lo"),
                                  JudgeLevel(score=0.5, desc="mid"),
                                  JudgeLevel(score=1.0, desc="hi")])
    task = TaskSpec(id="T", name="t", dir=tmp_path, labels=_labels(), prompt="p", judge=[crit])
    (tmp_path / "out.txt").write_text("answer")
    llm = fake_llm([{"q": {"rationale": "decent", "level": 2}}])
    results = await judge_task(task, [Produced(path="out.txt", status="created")], tmp_path, llm)
    assert results[0].score == 0.5                         # level 2 -> middle score
    assert results[0].message == "decent"


@pytest.mark.llm
async def test_judge_task_live_discriminates(tmp_path: Path) -> None:
    # real judge endpoint, NO docker: the judge must READ the artifact and discriminate,
    # not rubber-stamp — a matching artifact scores 1.0, a non-matching one (same rubric) 0.0.
    crit = JudgeCriterion(
        id="defines_add", desc="The file defines a top-level function named `add`", weight=1.0,
        levels=[JudgeLevel(score=0.0, desc="no function named add"),
                JudgeLevel(score=1.0, desc="defines a top-level function named add")])
    task = TaskSpec(id="T", name="t", dir=tmp_path, labels=_labels(),
                    prompt="Write add(a, b).", judge=[crit])
    llm = StructuredLLM(get_config().judge_endpoint, timeout=120.0)

    (tmp_path / "sol.py").write_text("def add(a, b):\n    return a + b\n")
    good = await judge_task(task, [Produced(path="sol.py", status="created")], tmp_path, llm)
    assert len(good) == 1 and good[0].id == "defines_add"
    assert good[0].score == 1.0
    assert good[0].message and "add" in good[0].message.lower()   # rationale cites the evidence

    (tmp_path / "sol.py").write_text("def multiply(a, b):\n    return a * b\n")
    bad = await judge_task(task, [Produced(path="sol.py", status="created")], tmp_path, llm)
    assert bad[0].score == 0.0                                    # no `add` -> bottom level
