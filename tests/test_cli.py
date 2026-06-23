from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from arktor_bench.cli import app

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
