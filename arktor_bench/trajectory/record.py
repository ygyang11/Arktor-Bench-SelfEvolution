from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """cached_input ⊆ input; reasoning ⊆ output (sub-views, not additive)."""

    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0


class ToolEvent(BaseModel):
    name: str
    args: str
    is_error: bool
    output: str


class StepRecord(BaseModel):
    index: int
    think: str = ""
    response: str = ""
    tools: list[ToolEvent] = Field(default_factory=list)


class TrajectoryRecord(BaseModel):
    steps: list[StepRecord]
    tokens: TokenUsage
    wall_ms: int = 0
    cap_hit: bool = False
    error: str = ""
