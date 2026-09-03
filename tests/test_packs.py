from __future__ import annotations

from pathlib import Path

import pytest

from arktor_bench.grading.validate import validate_static

_PACKS = Path(__file__).resolve().parents[1] / "packs"


def _task_dirs() -> list[Path]:
    # A task dir is one holding a task.md; container dirs (e.g. a pack's `_heldout/` of tasks held
    # out of the official suite) have none and are skipped, matching how load_pack enumerates a pack.
    return [d for pack in _PACKS.iterdir() if pack.is_dir()
            for d in pack.iterdir() if d.is_dir() and (d / "task.md").exists()]


@pytest.mark.parametrize("task_dir", _task_dirs(), ids=lambda d: d.name)
def test_every_task_passes_validate_static(task_dir: Path) -> None:
    r = validate_static(task_dir)
    assert r.ok, r.issues
