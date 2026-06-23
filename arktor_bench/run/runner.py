from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from arktor_bench.config import get_config
from arktor_bench.grading.score import score_task
from arktor_bench.harness.base import Adapter, RunResult
from arktor_bench.llm import FatalLLMError, StructuredLLM
from arktor_bench.models import CriterionResult, OutcomeScore, Produced, TaskSpec
from arktor_bench.run.scoreboard import CellMetrics
from arktor_bench.sandbox.workspace import task_workspace
from arktor_bench.spec.loader import load
from arktor_bench.trajectory.record import TokenUsage, TrajectoryRecord
from arktor_bench.utils.log import bar_disabled


def _snapshot_produced(dest: Path, root: Path, produced: list[Produced], seeds: list[str]) -> None:
    """Freeze the agent's diff + seeds' final state before graders touch the workspace."""
    for rel in {p.path for p in produced} | set(seeds):
        src = root / rel
        if src.is_file():
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def _persist(d: Path, rr: RunResult, score: OutcomeScore) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "trajectory.json").write_text(rr.trajectory.model_dump_json(indent=2))
    (d / "score.json").write_text(score.model_dump_json(indent=2))
    (d / "produced.json").write_text(
        json.dumps([p.model_dump() for p in rr.produced], ensure_ascii=False))
    (d / "metrics.json").write_text(CellMetrics(
        tokens=rr.trajectory.tokens, steps=len(rr.trajectory.steps),
        wall_ms=rr.trajectory.wall_ms, cap_hit=rr.trajectory.cap_hit,
        error=rr.trajectory.error,
    ).model_dump_json())


def _persist_failure(d: Path, task: TaskSpec, trial: int, err: Exception,
                     rr: RunResult | None) -> bool:
    msg = f"cell failed: {err}"
    results = [
        CriterionResult(
            id=c.id, mode="auto", score=0.0, weight=c.weight, message=msg, errored=True)
        for c in task.auto_checks]
    results += [
        CriterionResult(
            id=c.id, mode="judge", score=0.0, weight=c.weight, message=msg, errored=True)
        for c in task.judge]
    if rr is not None:                       # ran fine; scoring/teardown failed — keep its traj
        traj, produced = rr.trajectory, rr.produced
        traj.error = traj.error or msg
    else:                                    # setup/run never yielded a result
        traj, produced = (
            TrajectoryRecord(steps=[], tokens=TokenUsage(), cap_hit=True, error=msg), [])
    _persist(d, RunResult(trajectory=traj, produced=produced),
             OutcomeScore(task_id=task.id, trial=trial, results=results))
    return traj.cap_hit


async def run_matrix(
    tasks: list[TaskSpec], adapters: list[Adapter], trials: int, out: Path,
) -> None:
    cfg = get_config()
    checks = {t.id: load(t.dir)[1] for t in tasks}
    docker_sem = asyncio.Semaphore(cfg.docker_concurrency)
    local_sem = asyncio.Semaphore(cfg.local_concurrency)

    async with StructuredLLM(cfg.judge_endpoint) as judge:
        async def cell(
            adapter: Adapter, task: TaskSpec, trial: int) -> tuple[str, str, int, float, bool]:
            inv = cfg.harness_invocation(adapter.name)
            sem = local_sem if inv.backend == "local" else docker_sem
            d = out / adapter.name / task.id / str(trial)
            rr: RunResult | None = None
            async with sem:                                  # docker / local bounded apart
                try:
                    async with task_workspace(task, inv) as ws:   # fresh ws per cell
                        rr = await adapter.run(task, ws, inv)
                        _snapshot_produced(d / "produced", ws.root, rr.produced,
                                           [w.dest for w in task.workspace_files])  # before graders
                        score = await score_task(task, checks[task.id], ws, rr.produced,
                                                 d / "produced", trial, judge)
                    _persist(d, rr, score)
                    return adapter.name, task.id, trial, score.score, rr.trajectory.cap_hit
                except FatalLLMError:  # config/auth hits every cell — abort, no fake 0
                    raise
                except Exception as e:  # noqa: BLE001 — a failed cell scores 0, never vanishes
                    cap = _persist_failure(d, task, trial, e, rr)
                    return adapter.name, task.id, trial, 0.0, cap

        jobs = [cell(a, t, k) for a in adapters for t in tasks for k in range(trials)]
        with tqdm(total=len(jobs), desc="run", unit="cell", disable=bar_disabled()) as bar:
            for fut in asyncio.as_completed(jobs):
                try:
                    name, tid, k, sc, cap = await fut
                    tqdm.write(f"[{name}] {tid}/{k} -> {sc:.3f}{' CAP' if cap else ''}")
                except FatalLLMError:                   # propagate the abort past the drain loop
                    raise
                except Exception as e:  # noqa: BLE001 — cell self-persists; guard the drain loop
                    tqdm.write(f"! cell crashed past persist: {e}")
                bar.update(1)
