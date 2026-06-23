from __future__ import annotations

import pytest
from pydantic import ValidationError

from arktor_bench.models import (
    Attribution,
    CriterionResult,
    Layer,
    ModelCapability,
    OutcomeScore,
    ToolLever,
)


def _r(id: str, score: float, weight: float, *, errored: bool = False) -> CriterionResult:
    return CriterionResult(id=id, mode="auto", score=score, weight=weight, errored=errored)


def test_outcome_score_weighted_mean() -> None:
    s = OutcomeScore(task_id="T", trial=0, results=[_r("a", 1.0, 2.0), _r("b", 0.0, 1.0)])
    assert s.score == pytest.approx(2.0 / 3.0)


def test_score_excludes_errored_keeps_run_failure_zero() -> None:
    # an errored criterion contributes to neither numerator nor denominator
    s = OutcomeScore(task_id="T", trial=0,
                     results=[_r("a", 1.0, 1.0), _r("b", 0.0, 1.0, errored=True)])
    assert s.score == 1.0
    # a wholly-errored cell -> empty denominator -> 0.0
    allerr = OutcomeScore(task_id="T", trial=0,
                          results=[_r("a", 0.0, 1.0, errored=True)])
    assert allerr.score == 0.0


def test_deficiencies_lists_below_full() -> None:
    s = OutcomeScore(task_id="T", trial=0,
                     results=[_r("a", 1.0, 1.0), _r("b", 0.4, 1.0), _r("c", 0.0, 1.0)])
    assert [r.id for r in s.deficiencies] == ["b", "c"]


def test_attribution_tools_layer_xor_sublever() -> None:
    Attribution(layer=Layer.TOOLS, tool=ToolLever.ERROR)
    with pytest.raises(ValidationError):  # tools without a sub-lever
        Attribution(layer=Layer.TOOLS)
    with pytest.raises(ValidationError):  # sub-lever without layer=tools
        Attribution(layer=Layer.INSTRUCTIONS, tool=ToolLever.ERROR)


def test_attribution_layer_xor_model() -> None:
    Attribution(layer=Layer.INSTRUCTIONS)
    Attribution(model=ModelCapability.REASONING)
    with pytest.raises(ValidationError):  # both sides
        Attribution(layer=Layer.LOOP, model=ModelCapability.PLANNING)
    with pytest.raises(ValidationError):  # neither side
        Attribution()
