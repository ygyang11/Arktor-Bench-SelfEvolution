from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import arktor_bench.harness.arktor_sdk as sdk_mod
from arktor_bench.config import HarnessInvocation
from arktor_bench.harness import create_adapter
from arktor_bench.harness.arktor_sdk import ArktorSdkAdapter, parse_arktor_sdk_ndjson
from arktor_bench.harness.base import RunResult
from arktor_bench.models import Complexity, Domain, Layer, ModelCapability, TaskLabels, TaskSpec
from arktor_bench.sandbox.backend import ExecuteResult
from arktor_bench.trajectory.record import TokenUsage, TrajectoryRecord

_FIX = Path(__file__).parent / "fixtures"


def _task(prompt: str = "do it") -> TaskSpec:
    return TaskSpec(
        id="T",
        name="t",
        dir=Path("."),
        labels=TaskLabels(
            domain=Domain.SOFTWARE_ENGINEERING,
            subdomain="cli",
            model_capability=[ModelCapability.CODE],
            harness_focus=[Layer.LOOP],
            complexity=Complexity.LINEAR,
        ),
        prompt=prompt,
    )


def test_creates_dynamic_adapter() -> None:
    adapter = create_adapter("arktor_sdk.comagent")
    assert isinstance(adapter, ArktorSdkAdapter)
    assert adapter.method == "comagent"
    assert adapter.name == "arktor_sdk.comagent"


def test_create_adapter_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown harness"):
        create_adapter("unknown.method")


@pytest.mark.parametrize("model", ["gpt-test", ""])
async def test_run_builds_argv_and_copies_env(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "write 'quoted' && touch nope"
    inv = HarnessInvocation(
        command=["python", "-m", "baselines.comagent"],
        model=model,
        env={"BASE": "value"},
    )

    class FakeWorkspace:
        command = ""
        env: dict[str, str] = {}

        async def execute(
            self,
            command: str,
            *,
            timeout: float,
            env: dict[str, str],
        ) -> ExecuteResult:
            self.command = command
            self.env = env
            assert timeout == 42
            return ExecuteResult(
                exit_code=0,
                stdout='{"type":"result","is_error":false}\n',
            )

    def fake_finish(*args: Any, **kwargs: Any) -> RunResult:
        assert kwargs["events"] == [{"type": "result", "is_error": False}]
        return RunResult(trajectory=TrajectoryRecord(steps=[], tokens=TokenUsage()), produced=[])

    monkeypatch.setattr(sdk_mod, "get_config", lambda: SimpleNamespace(wall_s=42))
    monkeypatch.setattr(sdk_mod, "finish", fake_finish)
    workspace = FakeWorkspace()

    await ArktorSdkAdapter("comagent").run(_task(prompt), workspace, inv)  # type: ignore[arg-type]

    assert shlex.split(workspace.command) == [
        "python",
        "-m",
        "baselines.comagent",
        "-p",
        prompt,
        "--output-format",
        "json",
    ]
    expected_env = {"BASE": "value"}
    if model:
        expected_env["ARKTOR_LLM_MODEL"] = model
    assert workspace.env == expected_env
    assert inv.env == {"BASE": "value"}


def test_parser_accepts_partial_events_after_failed_exit() -> None:
    assert parse_arktor_sdk_ndjson('{"type":"step","index":7}\n', exit_code=None) == [
        {"type": "step", "index": 7},
    ]


def test_maps_multi_agent_trajectory() -> None:
    raw = parse_arktor_sdk_ndjson(
        (_FIX / "arktor_sdk_multi_agent.jsonl").read_text(),
        exit_code=0,
    )
    traj = ArktorSdkAdapter("comagent").to_trajectory(raw)

    assert [(step.index, step.agent, step.phase) for step in traj.steps] == [
        (0, "planner", "planning"),
        (1, "executor", "execution"),
    ]
    assert traj.steps[0].tools[0].name == "write_file"
    assert traj.steps[0].tools[0].output == "wrote 1 file"
    assert (traj.tokens.input, traj.tokens.output, traj.tokens.context) == (1300, 520, 4200)
    assert (traj.tokens.cached_input, traj.tokens.reasoning) == (210, 90)
    assert set(traj.agents) == {"planner", "executor"}
    assert traj.agents["planner"].tokens.input == 400
    assert traj.agents["planner"].tokens.context == 2100
    assert traj.agents["executor"].tokens.output == 380
    assert traj.agents["executor"].tokens.context == 4200


@pytest.mark.parametrize(
    "text, exit_code, message",
    [
        ("not json\n", 0, "invalid NDJSON"),
        ("[]\n", 0, "must be a JSON object"),
        ('{"type":"progress"}\n', 0, "unsupported event type"),
        ('{"type":"result","is_error":false}\n{"type":"step"}\n', 0, "must be the final"),
        ('{"type":"result"}\n', 0, "is_error must be a boolean"),
        ('{"type":"step"}\n', 0, "requires a final result"),
    ],
)
def test_parser_rejects_invalid_protocol(text: str, exit_code: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_arktor_sdk_ndjson(text, exit_code=exit_code)
