from __future__ import annotations

import types
from collections.abc import Awaitable, Callable

from arktor_bench.sandbox.workspace import Workspace

CheckFn = Callable[[Workspace], Awaitable[tuple[float, str]]]


def check(*, id: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        fn._check_id = id  # type: ignore[attr-defined]
        return fn
    return deco


def collect_checks(module: types.ModuleType) -> dict[str, CheckFn]:
    out: dict[str, CheckFn] = {}
    for obj in vars(module).values():
        cid = getattr(obj, "_check_id", None)
        if cid is None:
            continue
        if cid in out:
            raise ValueError(f"duplicate check id in {module.__name__}: {cid}")
        out[cid] = obj
    return out
