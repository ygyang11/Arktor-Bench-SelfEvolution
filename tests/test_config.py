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


def test_resolves_arktor_sdk_method_config() -> None:
    cfg = _cfg(harness={
        "arktor_sdk": {
            "comagent": {
                "command": ["python", "-m", "baselines.comagent"],
                "model": "m",
            },
        },
    })
    inv = cfg.harness_invocation("arktor_sdk.comagent")
    assert inv.command == ["python", "-m", "baselines.comagent"]
    assert inv.model == "m"


@pytest.mark.parametrize("command", [[], ["python", ""]])
def test_arktor_sdk_method_requires_command(command: list[str]) -> None:
    with pytest.raises(ValidationError, match="requires a non-empty command argv"):
        _cfg(harness={"arktor_sdk": {"comagent": {"command": command}}})


def test_missing_arktor_sdk_method_exits() -> None:
    cfg = _cfg(harness={"arktor_sdk": {}})
    with pytest.raises(SystemExit, match="arktor_sdk.comagent"):
        cfg.harness_invocation("arktor_sdk.comagent")


@pytest.mark.parametrize("name", [
    "arktor_sdk",
    "arktor_sdk.",
    "arktor_sdk.a.b",
    "arktor_sdk../x",
])
def test_rejects_invalid_arktor_sdk_runtime_name(name: str) -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="invalid harness name"):
        cfg.harness_invocation(name)


def test_rejects_invalid_arktor_sdk_config_key() -> None:
    with pytest.raises(ValidationError, match="invalid arktor_sdk method name"):
        _cfg(harness={
            "arktor_sdk": {
                "ComAgent": {"command": ["python", "-m", "baselines.comagent"]},
            },
        })


@pytest.mark.parametrize("name, expected", [
    ("arktor_sdk.comagent", "comagent"),
    ("arktor_sdk.llm_cot", "llm_cot"),
    ("arktor", None),
    ("unknown", None),
])
def test_parses_arktor_sdk_method_name(name: str, expected: str | None) -> None:
    assert HarnessConfigs.arktor_sdk_method(name) == expected


def test_judge_endpoint_is_model_endpoint() -> None:
    assert isinstance(_cfg().judge_endpoint, ModelEndpoint)
