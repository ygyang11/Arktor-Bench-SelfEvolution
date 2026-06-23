from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AuthenticationError
from pydantic import BaseModel, ValidationInfo, model_validator

from arktor_bench.config import ModelEndpoint
from arktor_bench.llm import FatalLLMError, StructuredLLM


class _Out(BaseModel):
    n: int
    model_config = {"extra": "forbid"}


class _Ctx(BaseModel):
    n: int
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _v(self, info: ValidationInfo) -> _Ctx:
        want = (info.context or {}).get("want")
        if want is not None and self.n != want:
            raise ValueError(f"n must be {want}")
        return self


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeCompletions:
    def __init__(self, scripted: list[Any]) -> None:
        self._scripted = scripted
        self.calls: list[list[dict[str, str]]] = []

    async def create(self, *, model: str, messages: Any, response_format: Any) -> SimpleNamespace:
        self.calls.append([dict(m) for m in messages])
        item = self._scripted[min(len(self.calls) - 1, len(self._scripted) - 1)]
        if isinstance(item, Exception):
            raise item
        return _resp(item)


def _wire(llm: StructuredLLM, comp: _FakeCompletions) -> None:
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=comp))  # type: ignore[assignment]


def _ep() -> ModelEndpoint:
    return ModelEndpoint(model="m", base_url="http://x", api_key="k")


async def test_returns_validated_model() -> None:
    llm = StructuredLLM(_ep())
    comp = _FakeCompletions(['{"n": 7}'])
    _wire(llm, comp)
    out = await llm.complete("p", _Out)
    assert out.n == 7
    assert len(comp.calls) == 1


async def test_validation_error_triggers_repair_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import arktor_bench.llm.client as client_mod

    async def _noop(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _noop)
    llm = StructuredLLM(_ep())
    comp = _FakeCompletions(['{"n": "bad"}', '{"n": 5}'])
    _wire(llm, comp)
    out = await llm.complete("p", _Out)
    assert out.n == 5
    assert len(comp.calls) == 2                          # repaired on the second call
    assert any(m["role"] == "assistant" for m in comp.calls[1])  # prior answer fed back


async def test_passes_context_to_validator() -> None:
    llm = StructuredLLM(_ep(), max_retries=0)
    _wire(llm, _FakeCompletions(['{"n": 5}']))
    assert (await llm.complete("p", _Ctx, context={"want": 5})).n == 5

    llm2 = StructuredLLM(_ep(), max_retries=0)
    _wire(llm2, _FakeCompletions(['{"n": 5}']))
    with pytest.raises(RuntimeError):                    # context constraint kept failing -> exhausted
        await llm2.complete("p", _Ctx, context={"want": 6})


async def test_fatal_4xx_raises_without_retry() -> None:
    err = AuthenticationError(
        "nope", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
    llm = StructuredLLM(_ep())
    comp = _FakeCompletions([err])
    _wire(llm, comp)
    with pytest.raises(FatalLLMError):
        await llm.complete("p", _Out)
    assert len(comp.calls) == 1                          # fatal -> no retry
