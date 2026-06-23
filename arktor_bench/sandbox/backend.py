from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExecuteResult:
    exit_code: int | None
    stdout: str
    stderr: str = ""


class Backend(Protocol):
    async def start(self, workspace: str) -> None: ...
    async def execute(self, command: str, *, timeout: float = 30.0,
                      env: dict[str, str] | None = None) -> ExecuteResult: ...
    async def stop(self) -> None: ...
