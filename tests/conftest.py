from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from support import FakeLLM, RunTree

from arktor_bench.trajectory.record import StepRecord, TokenUsage, ToolEvent, TrajectoryRecord

# --- external-resource gates: skip a marked test unless its flag is passed ---

_GATES = {"llm": "--run-llm", "docker": "--run-docker"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-llm", action="store_true", default=False,
                     help="run tests that need a real LLM API")
    parser.addoption("--run-docker", action="store_true", default=False,
                     help="run tests that need a Docker daemon + the bench image")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, flag in _GATES.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"pass {flag} to run")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


# --- fixtures ---

_FIXTURE_PACK = Path(__file__).parent / "fixtures" / "packs" / "general"


@pytest.fixture
def tmp_pack(tmp_path: Path) -> Path:
    """A copy of the minimal fixture pack (one task: T001_min) under a tmp dir."""
    dest = tmp_path / "general"
    shutil.copytree(_FIXTURE_PACK, dest)
    return dest


@pytest.fixture
def fake_llm() -> Callable[[Sequence[Any]], FakeLLM]:
    return FakeLLM


@pytest.fixture
def run_tree(tmp_path: Path, tmp_pack: Path) -> RunTree:
    out = tmp_path / "run"
    out.mkdir()
    return RunTree(out, tmp_pack)


@pytest.fixture
def sample_trajectory() -> TrajectoryRecord:
    """A multi-step trajectory exercising clipping, media, dedup, and the run-outcome tail."""
    repeated = ToolEvent(name="shell", args="ls -la", is_error=False, output="beta " * 9000)
    return TrajectoryRecord(
        steps=[
            StepRecord(index=0, think="alpha " * 6000, response="short reply", tools=[repeated]),
            StepRecord(index=1, tools=[repeated]),                          # dup -> "(same as step 0)"
            StepRecord(index=2, tools=[ToolEvent(name="img", args="load",
                                                 is_error=False, output="\x00\x00\x00media-bytes")]),
            StepRecord(index=3, tools=[ToolEvent(name="bad", args="x",
                                                 is_error=True, output="boom error")]),
        ],
        tokens=TokenUsage(input=1000, cached_input=200, output=500, reasoning=100),
        cap_hit=True, error="transport failed 503",
    )
