from __future__ import annotations

from pathlib import Path
from typing import cast

from arktor_bench.sandbox.backend import Backend
from arktor_bench.sandbox.workspace import Workspace, _snapshot, classify_produced, is_artifact


def _ws(root: Path) -> Workspace:
    return Workspace(root, cast(Backend, None))


def test_is_artifact_excludes_tooling_byproducts() -> None:
    # kept — could each be a deliverable (binaries / build output are not assumed noise)
    keep = ["todo.py", "src/main.go", "data/todo.json", "out.so", "ext.pyd", "target/release/app"]
    drop = ["__pycache__/todo.cpython-311.pyc", "todo.pyc", "x.pyo",
            ".pytest_cache/v/cache", ".mypy_cache/3.11/x.json", ".ruff_cache/x",
            "node_modules/lodash/index.js", ".venv/bin/python",
            ".git/HEAD", ".idea/workspace.xml", ".DS_Store", "Thumbs.db"]
    assert all(is_artifact(Path(p)) for p in keep)
    assert not any(is_artifact(Path(p)) for p in drop)


def test_diff_excludes_tooling_byproducts(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.baseline = _snapshot(tmp_path)
    (tmp_path / "todo.py").write_text("x")                      # the real deliverable
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "todo.cpython-311.pyc").write_text("bytecode")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    assert {p.path for p in classify_produced(ws)} == {"todo.py"}   # caches/vcs excluded


def test_diff_excludes_seeded(tmp_path: Path) -> None:
    (tmp_path / "seed.txt").write_text("original")
    ws = _ws(tmp_path)
    ws.baseline = _snapshot(tmp_path)               # baseline taken after seeding
    (tmp_path / "new.py").write_text("print(1)")    # created by the agent
    (tmp_path / "seed.txt").write_text("changed")   # modified by the agent
    produced = {p.path: p.status for p in classify_produced(ws)}
    assert produced == {"new.py": "created", "seed.txt": "modified"}


def test_read_json_none_on_bad(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (tmp_path / "good.json").write_text('{"a": 1}')
    (tmp_path / "bad.json").write_text("{not json")
    assert ws.read_json("good.json") == {"a": 1}
    assert ws.read_json("bad.json") is None
    assert ws.read_json("missing.json") is None
