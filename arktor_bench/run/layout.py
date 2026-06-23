from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class RunManifest(BaseModel):
    pack: str
    pack_dir: str
    tasks: list[str]
    trials: int
    harnesses: list[str]


@dataclass(frozen=True)
class CellPath:
    harness: str
    task: str
    trial: str
    dir: Path

    @property
    def score(self) -> Path:
        return self.dir / "score.json"

    @property
    def trajectory(self) -> Path:
        return self.dir / "trajectory.json"

    @property
    def metrics(self) -> Path:
        return self.dir / "metrics.json"

    @property
    def findings(self) -> Path:
        return self.dir / "findings.json"


def iter_cells(run_dir: Path) -> Iterator[CellPath]:
    for hdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for tdir in sorted(p for p in hdir.iterdir() if p.is_dir()):
            for trdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
                yield CellPath(hdir.name, tdir.name, trdir.name, trdir)


def read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "tasks.json").read_text())


def write_manifest(run_dir: Path, pack: str, tasks: list[str],
                   trials: int, harnesses: list[str]) -> None:
    path = run_dir / "tasks.json"
    pack_dir = str((Path("packs") / pack).resolve())
    if path.is_file():
        prev = RunManifest.model_validate_json(path.read_text())
        harnesses = sorted(set(prev.harnesses) | set(harnesses))
        tasks = sorted(set(prev.tasks) | set(tasks)) if (prev.tasks and tasks) else []
    path.write_text(RunManifest(
        pack=pack, pack_dir=pack_dir, tasks=tasks, trials=trials,
        harnesses=harnesses).model_dump_json(indent=2))
