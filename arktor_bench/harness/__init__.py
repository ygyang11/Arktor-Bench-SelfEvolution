from __future__ import annotations

from arktor_bench.harness.arktor import ArktorAdapter
from arktor_bench.harness.base import Adapter
from arktor_bench.harness.claude_code import ClaudeCodeAdapter
from arktor_bench.harness.codex import CodexAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "arktor": ArktorAdapter,
    "codex": CodexAdapter,
    "claude_code": ClaudeCodeAdapter,
}

__all__ = ["ADAPTERS", "Adapter", "ArktorAdapter", "ClaudeCodeAdapter", "CodexAdapter"]
