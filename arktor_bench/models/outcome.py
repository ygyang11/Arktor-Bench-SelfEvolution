from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from arktor_bench.models.taxonomy import Layer, ModelCapability, ToolLever


class CriterionResult(BaseModel):
    id: str
    mode: Literal["auto", "judge"]
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(gt=0.0)
    message: str = ""
    errored: bool = False
    model_config = {"extra": "forbid"}


class OutcomeScore(BaseModel):
    task_id: str
    trial: int = Field(ge=0)
    results: list[CriterionResult]
    model_config = {"extra": "forbid"}

    @property
    def score(self) -> float:
        scored = [r for r in self.results if not r.errored]
        t = sum(r.weight for r in scored)
        return sum(r.score * r.weight for r in scored) / t if t else 0.0

    @property
    def deficiencies(self) -> list[CriterionResult]:
        return [r for r in self.results if r.score < 1.0]


class Attribution(BaseModel):
    layer: Layer | None = None
    tool: ToolLever | None = None
    model: ModelCapability | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> Attribution:
        if (self.layer is None) == (self.model is None):
            raise ValueError("name exactly one of: a harness layer, or a model capability")
        if (self.layer == Layer.TOOLS) != (self.tool is not None):
            raise ValueError(
                "layer=tools needs a tool sub-lever, and a tool sub-lever needs layer=tools")
        return self


class Finding(BaseModel):
    task_id: str
    trial: int = Field(ge=0)
    criterion_ids: list[str]
    attribution: Attribution
    root_cause: str
    recoverable_points: float = Field(ge=0.0)
    model_config = {"extra": "forbid"}


class Produced(BaseModel):
    path: str
    status: Literal["created", "modified"]
    model_config = {"extra": "forbid"}
