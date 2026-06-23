from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import typer

from arktor_bench.config import BENCH_HOME, get_config
from arktor_bench.harness import ADAPTERS
from arktor_bench.llm import StructuredLLM

app = typer.Typer(no_args_is_help=True)


def _stamp() -> Path:
    return BENCH_HOME / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")


def _pack_dir(pack: str) -> Path:
    d = Path("packs") / pack
    if not d.is_dir():
        raise typer.BadParameter(f"unknown pack '{pack}' (no packs/{pack})")
    return d


def _require_run(run_dir: Path) -> Path:
    if not (run_dir / "tasks.json").is_file():
        raise typer.BadParameter(f"'{run_dir}' is not a run directory (no tasks.json)")
    return run_dir


def _run(pack: str, harness: str, tasks: str, trials: int, out: Path) -> Path:
    from arktor_bench.run.layout import write_manifest
    from arktor_bench.run.runner import run_matrix
    from arktor_bench.run.scoreboard import write_scoreboard
    from arktor_bench.spec.loader import load_pack
    sel = tasks.split(",") if tasks else None
    try:  # _pack_dir validates the pack; load_pack then only fails on bad task name / spec
        ts = load_pack(pack, sel, base=_pack_dir(pack))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    names = harness.split(",")
    unknown = [h for h in names if h not in ADAPTERS]
    if unknown:
        raise typer.BadParameter(f"unknown harness {unknown}; choose from {sorted(ADAPTERS)}")
    adapters = [ADAPTERS[h]() for h in names]
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, pack, [t.id for t in ts] if sel else [], trials, names)
    typer.echo(f"run: pack={pack} harness={names} tasks={len(ts)} trials={trials} -> {out}")
    asyncio.run(run_matrix(ts, adapters, trials, out))
    sb = write_scoreboard(out)                            # rebuilt over all harnesses in `out`
    typer.echo(f"run: scoreboard -> {out / 'scoreboard.md'}")
    for h, b in sb.harnesses.items():
        typer.echo(f"  {h}: overall {b.overall:.3f}  "
                   f"tokens~{b.efficiency.tokens_total:.0f}  steps~{b.efficiency.steps:.1f}")
    return out


@app.command()
def run(pack: str, harness: str, tasks: str = "", trials: int = 3,
        out: Path | None = None) -> None:
    _run(pack, harness, tasks, trials, out or _stamp())


@app.command()
def diagnose(run_dir: Path) -> None:
    from arktor_bench.diagnose.diagnose import diagnose_dir
    _require_run(run_dir)

    async def _go() -> None:
        async with StructuredLLM(get_config().diagnose_endpoint) as llm:
            await diagnose_dir(run_dir, llm)
    asyncio.run(_go())


@app.command()
def report(run_dir: Path) -> None:
    from arktor_bench.report.aggregate import build_report
    _require_run(run_dir)

    async def _go() -> None:
        async with StructuredLLM(get_config().diagnose_endpoint) as llm:
            await build_report(run_dir, llm)
    asyncio.run(_go())


@app.command()
def eval(pack: str, harness: str, tasks: str = "", trials: int = 3,
         out: Path | None = None) -> None:
    """run -> diagnose -> report in one shot."""
    from arktor_bench.diagnose.diagnose import diagnose_dir
    from arktor_bench.report.aggregate import build_report
    typer.echo("run: starting")
    dest = _run(pack, harness, tasks, trials, out or _stamp())

    async def _diagnose_report() -> None:            # one loop + one client for both stages
        async with StructuredLLM(get_config().diagnose_endpoint) as diag:
            typer.echo("diagnose: starting")
            await diagnose_dir(dest, diag)
            typer.echo("report: starting")
            await build_report(dest, diag)
    asyncio.run(_diagnose_report())


@app.command()
def validate(pack: str, judge: bool = False) -> None:
    from arktor_bench.grading.validate import ValidityResult, validate_judge, validate_static
    dirs = [d for d in sorted(_pack_dir(pack).iterdir()) if d.is_dir()]

    async def _judged() -> list[ValidityResult]:     # one loop + one client across all tasks
        async with StructuredLLM(get_config().judge_endpoint) as jllm:
            return [await validate_judge(d, jllm) for d in dirs]

    results = asyncio.run(_judged()) if judge else [validate_static(d) for d in dirs]
    failed = 0
    for r in results:
        failed += not r.ok
        typer.echo(f"{'OK  ' if r.ok else 'FAIL'} {r.task_id}: {r.issues}")
    if failed:
        raise typer.Exit(1)
