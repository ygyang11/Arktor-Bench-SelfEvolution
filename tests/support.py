"""Reusable test infrastructure (not fixtures): a StructuredLLM stand-in and a run-dir builder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from arktor_bench.run.layout import RunManifest


class FakeLLM:
    """A StructuredLLM stand-in: returns canned instances of the schema it is handed.

    Each scripted response is a dict validated into the requested output model, a
    callable `out_type -> dict`, or an Exception to raise. Prompts/contexts are retained.
    """

    def __init__(self, scripted: Sequence[Any]) -> None:
        self._scripted = list(scripted)
        self.prompts: list[str] = []
        self.contexts: list[dict[str, object] | None] = []

    async def complete(self, prompt: str, out: type[BaseModel], *,
                       system: str | None = None,
                       context: dict[str, object] | None = None) -> BaseModel:
        self.prompts.append(prompt)
        self.contexts.append(context)
        item = self._scripted.pop(0) if self._scripted else {}
        if isinstance(item, Exception):
            raise item
        data = item(out) if callable(item) else item
        return out.model_validate(data, context=context)


class RunTree:
    """Builds a temporary run dir (`<out>/<harness>/<task>/<trial>/...` + tasks.json) over a pack."""

    def __init__(self, out: Path, pack_dir: Path) -> None:
        self.out = out
        self.pack_dir = pack_dir

    def manifest(self, *, harnesses: Sequence[str], trials: int = 1,
                 tasks: Sequence[str] = ()) -> None:
        (self.out / "tasks.json").write_text(RunManifest(
            pack="general", pack_dir=str(self.pack_dir), tasks=list(tasks),
            trials=trials, harnesses=list(harnesses)).model_dump_json(indent=2))

    def cell(self, harness: str, task: str, trial: int, *, score: Any = None,
             metrics: Any = None, trajectory: Any = None, findings: Any = None) -> Path:
        d = self.out / harness / task / str(trial)
        d.mkdir(parents=True, exist_ok=True)
        if score is not None:
            self._write(d / "score.json", score)
        if metrics is not None:
            self._write(d / "metrics.json", metrics)
        if trajectory is not None:
            self._write(d / "trajectory.json", trajectory)
        if findings is not None:
            d.joinpath("findings.json").write_text(
                findings if isinstance(findings, str)
                else json.dumps([f.model_dump(mode="json") for f in findings]))
        return d

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.write_text(value if isinstance(value, str) else value.model_dump_json())
