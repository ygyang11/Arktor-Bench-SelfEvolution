from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from arktor_bench.grading.check import CheckFn, collect_checks
from arktor_bench.models import (
    AutoCheckCriterion,
    JudgeCriterion,
    TaskLabels,
    TaskSpec,
    WorkspaceFile,
)

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class _FrontMatter(BaseModel):
    id: str
    name: str
    labels: TaskLabels
    workspace_files: list[WorkspaceFile] = Field(default_factory=list)
    setup: str | None = None
    model_config = {"extra": "forbid"}


def _sections(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        h = re.match(r"^##\s+(.+)$", line)
        if h:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = h.group(1).strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def _yaml_list(text: str, where: str) -> list[dict[str, object]]:
    try:
        data = yaml.safe_load(text) or []
    except yaml.YAMLError as e:
        raise ValueError(f"{where}: invalid yaml: {e}") from e
    if not isinstance(data, list):
        raise ValueError(f"{where}: expected a yaml list")
    return data


def load_task(task_dir: Path) -> TaskSpec:
    path = task_dir / "task.md"
    if not path.is_file():
        raise ValueError(f"{task_dir}: missing task.md")
    m = _FM.match(path.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{task_dir}: missing '---' front-matter")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{task_dir}: bad front-matter yaml: {e}") from e
    secs = _sections(m.group(2))
    if not secs.get("Prompt", "").strip():
        raise ValueError(f"{task_dir}: missing or empty '## Prompt' section")
    try:
        fm = _FrontMatter(**meta)
        auto = [AutoCheckCriterion.model_validate(c)
                for c in _yaml_list(secs.get("Auto Checks", ""), f"{task_dir} Auto Checks")]
        judge = [JudgeCriterion.model_validate(c)
                 for c in _yaml_list(secs.get("Judge Rubric", ""), f"{task_dir} Judge Rubric")]
    except ValidationError as e:
        raise ValueError(f"{task_dir}: invalid task spec: {e}") from e
    return TaskSpec(
        id=fm.id, name=fm.name, dir=task_dir, labels=fm.labels,
        prompt=secs["Prompt"].strip(), auto_checks=auto, judge=judge,
        workspace_files=fm.workspace_files, setup=fm.setup,
    )


def load_checks(task_dir: Path) -> dict[str, CheckFn]:
    path = task_dir / "grader.py"
    if not path.is_file():                              # fall back to grader/grader.py — a grader
        path = task_dir / "grader" / "grader.py"       
    if not path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location(f"grader_{task_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{task_dir}: cannot load grader")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{task_dir}: grader failed to import: {e}") from e
    return collect_checks(module)


def load(task_dir: Path) -> tuple[TaskSpec, dict[str, CheckFn]]:
    task = load_task(task_dir)
    if task.id != task_dir.name:
        raise ValueError(f"{task_dir}: task id {task.id!r} must equal its directory name")
    checks = load_checks(task_dir)
    auto_ids = {c.id for c in task.auto_checks}
    judge_ids = {c.id for c in task.judge}
    if auto_ids != set(checks):
        raise ValueError(f"{task_dir}: auto_checks {auto_ids} != grader.py {set(checks)}")
    if auto_ids & judge_ids:
        raise ValueError(f"{task_dir}: id reused by auto & judge: {auto_ids & judge_ids}")
    if not (task.auto_checks or task.judge):
        raise ValueError(f"{task_dir}: has no criteria")
    criteria: list[AutoCheckCriterion | JudgeCriterion] = [*task.auto_checks, *task.judge]
    for c in criteria:
        if c.weight <= 0:
            raise ValueError(f"{task_dir}: criterion {c.id} weight must be > 0")
    for w in task.workspace_files:
        if not (task_dir / w.source).is_file():
            raise ValueError(f"{task_dir}: workspace_files source missing: {w.source}")
    return task, checks


def load_pack(pack: str, tasks: list[str] | None = None,
              base: Path | None = None) -> list[TaskSpec]:
    root = base if base is not None else Path("packs") / pack
    dirs = {d.name: d for d in root.iterdir() if d.is_dir()}
    names = tasks if tasks else sorted(dirs)
    missing = [n for n in names if n not in dirs]
    if missing:
        raise ValueError(f"pack {pack!r}: unknown task(s) {sorted(missing)}")
    return [load(dirs[n])[0] for n in names]
