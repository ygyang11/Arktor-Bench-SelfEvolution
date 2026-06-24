from __future__ import annotations

from pathlib import Path

import pytest

from arktor_bench.spec.loader import load, load_pack, load_task

_GRADER_OK = """\
from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace


@check(id="c_ok")
async def c_ok(ws: Workspace) -> tuple[float, str]:
    return 1.0, ""
"""


def _task_md(*, id: str = "T", auto: str = '- {id: c_ok, desc: "d", weight: 1}',
             judge: str = "") -> str:
    judge_block = f"\n## Judge Rubric\n\n{judge}\n" if judge else ""
    return (
        f"---\nid: {id}\nname: n\nlabels:\n  domain: software_engineering\n"
        "  subdomain: cli_tool\n  model_capability: [code]\n  harness_focus: [instructions]\n"
        "  complexity: linear\nworkspace_files: []\n---\n\n## Prompt\n\nDo a thing.\n\n"
        f"## Auto Checks\n\n{auto}\n{judge_block}"
    )


def _write(root: Path, name: str, md: str, grader: str = _GRADER_OK) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "task.md").write_text(md)
    (d / "grader.py").write_text(grader)
    return d


def test_parses_frontmatter_and_sections(tmp_pack: Path) -> None:
    task = load_task(tmp_pack / "T001_min")
    assert task.id == "T001_min"
    assert "answer.txt" in task.prompt
    assert {c.id for c in task.auto_checks} == {"c_ok"}
    assert {c.id for c in task.judge} == {"c_quality"}
    assert task.total_weight == 3.0


def test_cross_checks_auto_ids_against_grader(tmp_path: Path) -> None:
    # auto check id 'c_ok' but grader exposes 'other' -> mismatch
    grader = _GRADER_OK.replace('id="c_ok"', 'id="other"').replace("c_ok(", "other(")
    d = _write(tmp_path, "T", _task_md(id="T"), grader=grader)
    with pytest.raises(ValueError):
        load(d)


def test_rejects_id_reused_by_auto_and_judge(tmp_path: Path) -> None:
    judge = ('- id: c_ok\n  desc: "dup id"\n  weight: 1\n  levels:\n'
             '    - {score: 0.0, desc: "lo"}\n    - {score: 1.0, desc: "hi"}')
    d = _write(tmp_path, "T", _task_md(id="T", judge=judge))
    with pytest.raises(ValueError):
        load(d)


def test_load_pack_selects_subset(tmp_path: Path) -> None:
    _write(tmp_path, "one", _task_md(id="one"))
    _write(tmp_path, "two", _task_md(id="two"))
    only = load_pack("general", ["one"], base=tmp_path)
    assert [t.id for t in only] == ["one"]
    both = load_pack("general", base=tmp_path)
    assert sorted(t.id for t in both) == ["one", "two"]


def test_load_pack_validates_and_rejects_unknown(tmp_path: Path) -> None:
    _write(tmp_path, "one", _task_md(id="one"))
    with pytest.raises(ValueError):
        load_pack("general", ["nope"], base=tmp_path)


def test_loads_dir_form_grader(tmp_path: Path) -> None:
    # top-level grader.py absent -> loader falls back to grader/grader.py (helpers alongside)
    d = tmp_path / "T"
    (d / "grader").mkdir(parents=True)
    (d / "task.md").write_text(_task_md(id="T"))
    (d / "grader" / "grader.py").write_text(_GRADER_OK)
    _, checks = load(d)
    assert set(checks) == {"c_ok"}
