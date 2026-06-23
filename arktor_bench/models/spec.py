from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, model_validator

from arktor_bench.models.taxonomy import Complexity, Domain, Layer, ModelCapability


class TaskLabels(BaseModel):
    domain: Domain
    subdomain: str
    model_capability: list[ModelCapability]
    harness_focus: list[Layer]
    complexity: Complexity
    multimodal: bool = False
    online: bool = False
    model_config = {"extra": "forbid"}


class WorkspaceFile(BaseModel):
    source: str
    dest: str
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_paths(self) -> WorkspaceFile:
        for label, rel in (("source", self.source), ("dest", self.dest)):
            p = PurePosixPath(rel)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(
                    f"workspace_files {label} must be a relative non-escaping path: {rel}")
        return self


class JudgeLevel(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    desc: str
    model_config = {"extra": "forbid"}


class JudgeCriterion(BaseModel):
    id: str
    desc: str
    weight: float = Field(gt=0.0)
    levels: list[JudgeLevel]
    artifacts: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_levels(self) -> JudgeCriterion:
        scores = [lv.score for lv in self.levels]
        if not scores:
            raise ValueError(f"criterion {self.id}: needs >=1 level")
        if scores != sorted(scores) or len(set(scores)) != len(scores):
            raise ValueError(f"criterion {self.id}: level scores must strictly ascend")
        if scores[-1] != 1.0:
            raise ValueError(f"criterion {self.id}: top level score must be 1.0")
        return self


class AutoCheckCriterion(BaseModel):
    id: str
    desc: str
    weight: float = Field(gt=0.0)
    model_config = {"extra": "forbid"}


class TaskSpec(BaseModel):
    id: str
    name: str
    dir: Path
    labels: TaskLabels
    prompt: str
    auto_checks: list[AutoCheckCriterion] = Field(default_factory=list)
    judge: list[JudgeCriterion] = Field(default_factory=list)
    workspace_files: list[WorkspaceFile] = Field(default_factory=list)
    setup: str | None = None
    model_config = {"extra": "forbid"}

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.auto_checks) + sum(c.weight for c in self.judge)
