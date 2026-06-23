from __future__ import annotations

import types

import pytest

from arktor_bench.grading import check
from arktor_bench.grading.check import collect_checks


def test_decorator_tags_id() -> None:
    @check(id="abc")
    async def f(ws: object) -> tuple[float, str]:
        return 1.0, ""

    assert f._check_id == "abc"  # type: ignore[attr-defined]
    mod = types.ModuleType("m")
    mod.f = f  # type: ignore[attr-defined]
    assert set(collect_checks(mod)) == {"abc"}


def test_collect_rejects_duplicate_id() -> None:
    @check(id="dup")
    async def a(ws: object) -> tuple[float, str]:
        return 1.0, ""

    @check(id="dup")
    async def b(ws: object) -> tuple[float, str]:
        return 1.0, ""

    mod = types.ModuleType("m")
    mod.a = a  # type: ignore[attr-defined]
    mod.b = b  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        collect_checks(mod)
