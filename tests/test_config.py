from __future__ import annotations

import pytest
from pydantic import ValidationError

from arktor_bench.config import BenchConfig, HarnessConfigs, HarnessInvocation, ModelEndpoint


def _cfg(**kw: object) -> BenchConfig:
    base: dict[str, object] = {"judge": {"model": "j", "base_url": "http://x", "api_key": "k"}}
    base.update(kw)
    return BenchConfig(**base)


def test_load_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        BenchConfig(judge={"model": "j", "base_url": "http://x"}, bogus=1)


def test_diagnose_endpoint_overlays_judge() -> None:
    cfg = _cfg(diagnose={"model": "d-model"})
    ep = cfg.diagnose_endpoint
    assert ep.model == "d-model"            # overridden
    assert ep.base_url == "http://x"        # inherited from judge
    assert ep.api_key == "k"
    # empty overlay == judge identity
    assert _cfg().diagnose_endpoint.model == "j"


def test_missing_harness_exits() -> None:
    cfg = _cfg(harness=HarnessConfigs(arktor=HarnessInvocation(model="m")))
    assert cfg.harness_invocation("arktor").model == "m"
    with pytest.raises(SystemExit):
        cfg.harness_invocation("codex")


def test_judge_endpoint_is_model_endpoint() -> None:
    assert isinstance(_cfg().judge_endpoint, ModelEndpoint)
