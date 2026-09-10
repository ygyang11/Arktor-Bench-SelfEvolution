from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

import arktor_bench.cli as cli
from arktor_bench.cli import app
from arktor_bench.config import BenchConfig

runner = CliRunner()


def test_validate_command_reports_pack() -> None:
    result = runner.invoke(app, ["validate", "test"])
    assert result.exit_code == 0                         # every shipped task is statically valid
    assert "T001_todo_cli" in result.stdout
    assert "FAIL" not in result.stdout


def test_run_rejects_unknown_harness() -> None:
    # a typo'd harness must fail cleanly (BadParameter), not crash with KeyError or spin docker
    result = runner.invoke(app, ["run", "test", "bogus"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, KeyError)    # the bug we are guarding against
    assert "unknown harness" in result.output


@pytest.mark.parametrize("harness", ["arktor,", ",arktor", "arktor,,codex"])
def test_run_rejects_empty_harness_name(harness: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(typer.BadParameter, match="empty name"):
        cli._run("test", harness, "", 1, out)
    assert not out.exists()


def test_run_rejects_duplicate_harness_name(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(typer.BadParameter, match="duplicate harness"):
        cli._run("test", "arktor, arktor", "", 1, out)
    assert not out.exists()


def test_run_rejects_invalid_arktor_sdk_name_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = BenchConfig(
        judge={"model": "j", "base_url": "http://x"},
        harness={"arktor_sdk": {}},
    )
    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    out = tmp_path / "run"
    with pytest.raises(typer.BadParameter, match="invalid harness name"):
        cli._run("test", "arktor_sdk.a.b", "", 1, out)
    assert not out.exists()


def test_config_validation_error_is_not_reported_as_bad_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError) as caught:
        BenchConfig(judge={"model": "j", "base_url": "http://x"}, extra_field=True)

    def invalid_config() -> BenchConfig:
        raise caught.value

    monkeypatch.setattr(cli, "get_config", invalid_config)
    out = tmp_path / "run"
    with pytest.raises(ValidationError):
        cli._run("test", "arktor", "", 1, out)
    assert not out.exists()


def test_run_accepts_configured_arktor_sdk_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arktor_bench.run import runner as run_mod
    from arktor_bench.run import scoreboard as scoreboard_mod

    cfg = BenchConfig(
        judge={"model": "j", "base_url": "http://x"},
        harness={
            "arktor_sdk": {
                "comagent": {"command": ["python", "-m", "baselines.comagent"]},
            },
        },
    )

    async def fake_run_matrix(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    monkeypatch.setattr(run_mod, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(
        scoreboard_mod,
        "write_scoreboard",
        lambda _out: SimpleNamespace(harnesses={}),
    )
    out = tmp_path / "run"

    assert cli._run("test", "arktor_sdk.comagent", "", 1, out) == out
    assert out.is_dir()


def test_run_rejects_unknown_pack() -> None:
    result = runner.invoke(app, ["run", "nope_pack", "arktor"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    assert "unknown pack" in result.output


def test_run_rejects_unknown_task() -> None:
    result = runner.invoke(app, ["run", "test", "arktor", "--tasks", "T999_nope"])
    assert result.exit_code != 0
    assert "unknown task" in result.output


def test_diagnose_rejects_non_run_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["diagnose", str(tmp_path)],     # empty dir, no tasks.json
                           env={"COLUMNS": "200"})                # avoid Rich panel wrapping the message
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)
    assert "not a run directory" in result.output
