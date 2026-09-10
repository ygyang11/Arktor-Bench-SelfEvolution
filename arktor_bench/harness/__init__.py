from __future__ import annotations

from arktor_bench.config import ARKTOR_SDK_FAMILY
from arktor_bench.harness.arktor import ArktorAdapter
from arktor_bench.harness.arktor_sdk import ArktorSdkAdapter
from arktor_bench.harness.base import Adapter
from arktor_bench.harness.claude_code import ClaudeCodeAdapter
from arktor_bench.harness.codex import CodexAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "arktor": ArktorAdapter,
    "codex": CodexAdapter,
    "claude_code": ClaudeCodeAdapter,
}


def create_adapter(name: str) -> Adapter:
    adapter = ADAPTERS.get(name)
    if adapter is not None:
        return adapter()
    family, _, method = name.partition(".")
    if family == ARKTOR_SDK_FAMILY:
        return ArktorSdkAdapter(method)
    raise ValueError(
        f"unknown harness '{name}'; choose a built-in harness or "
        "configured arktor_sdk.<method>"
    )


__all__ = [
    "ADAPTERS",
    "Adapter",
    "ArktorAdapter",
    "ArktorSdkAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "create_adapter",
]
