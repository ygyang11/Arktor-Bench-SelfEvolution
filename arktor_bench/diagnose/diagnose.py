from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from arktor_bench.config import get_config
from arktor_bench.diagnose.levers import attr_label, load_lever_spec
from arktor_bench.llm import FatalLLMError, StructuredLLM
from arktor_bench.models import (
    Attribution,
    AutoCheckCriterion,
    Finding,
    JudgeCriterion,
    OutcomeScore,
    TaskSpec,
)
from arktor_bench.run.layout import CellPath, iter_cells, read_manifest
from arktor_bench.spec.loader import load_pack
from arktor_bench.trajectory.compact import compact_trajectory
from arktor_bench.trajectory.record import TrajectoryRecord
from arktor_bench.utils.log import bar_disabled, log

_SYSTEM = """\
You are an agent diagnostician. The candidate is a model run inside a harness — the \
scaffold (instructions, tools, loop, context) that turns a model into an agent. You \
are given the task, the criteria the run failed, and a compact trajectory of what the \
agent did. For each failed criterion, trace the failure to its underlying cause and \
attribute that cause to one category — a harness lever or a model capability — by the \
taxonomy and rule given below.

State the root_cause as a durable property of the failing lever or capability — one \
that holds beyond this run, not a recap of it.
- Name what is at fault by what it durably is — a harness tool (use real name) lacks , \
the model by the capability that fell short — never by the agent's generated functions or \
files, even where the failed criteria quote them, which carry no meaning beyond this run.
- Keep out the rest specific to this run — the task's wording, the steps taken, and \
the code the agent wrote — though the trajectory shows them all.
- Well-formed, one per side: "the shell tool's error result returns only the exit \
code, not stderr, so the agent cannot tell why a command failed and retries it \
unchanged"; "the model mis-derives the recursion's base case and returns off-by-one \
results though the task was unambiguous and the tools sufficed.\""""

_USER = """\
{lever_spec}

# Task
{task}

# Failed criteria
{failed}

# Compact trajectory
{compact}

Now produce findings for the failed criteria above, following the rules above: each \
finding carries its criterion_ids, its root_cause, and its attribution."""


class _FindingItem(BaseModel):
    criterion_ids: list[str]
    root_cause: str
    attribution: Attribution
    model_config = {"extra": "forbid"}


class _DiagnoseOut(BaseModel):
    findings: list[_FindingItem]
    model_config = {"extra": "forbid"}


async def diagnose_run(
    score: OutcomeScore, traj: TrajectoryRecord, task: TaskSpec,
    lever_spec: str, llm: StructuredLLM,
) -> list[Finding]:
    failed = [c for c in score.deficiencies if not c.errored]   # skip evaluator errors
    if not failed or not traj.steps:        # nothing to diagnose, or an empty/aborted run
        return []
    by = {c.id: c for c in failed}
    all_crit: list[AutoCheckCriterion | JudgeCriterion] = [*task.auto_checks, *task.judge]
    desc = {c.id: c.desc for c in all_crit}
    failed_md = "\n".join(
        f"- [{c.id}] {desc[c.id]} — {c.message}" if c.id in desc else f"- [{c.id}] {c.message}"
        for c in failed)
    prompt = _USER.format(lever_spec=lever_spec, task=task.prompt,
                          failed=failed_md, compact=compact_trajectory(traj))
    out = await llm.complete(prompt, _DiagnoseOut, system=_SYSTEM)
    raw: list[tuple[_FindingItem, list[str]]] = []
    for f in out.findings:
        ids = list(dict.fromkeys(i for i in f.criterion_ids if i in by))   # known, de-duped
        if ids:
            raw.append((f, ids))
    cover = Counter(i for _, ids in raw for i in ids)           # findings sharing each criterion
    return [
        Finding(
            task_id=score.task_id, trial=score.trial, criterion_ids=ids,
            attribution=f.attribution, root_cause=f.root_cause,
            recoverable_points=sum(by[i].weight * (1 - by[i].score) / cover[i] for i in ids),
        )
        for f, ids in raw
    ]


async def diagnose_dir(run_dir: Path, llm: StructuredLLM) -> None:
    manifest = read_manifest(run_dir)
    cells = list(iter_cells(run_dir))
    tasks = {t.id: t
             for t in load_pack(manifest.pack, sorted({c.task for c in cells}),
                                base=Path(manifest.pack_dir))}
    lever_spec = load_lever_spec()
    sem = asyncio.Semaphore(get_config().llm_concurrency)

    async def one(cell: CellPath) -> int | None:
        task = tasks.get(cell.task)
        if task is None:
            return None
        try:
            score = OutcomeScore.model_validate_json(cell.score.read_text())
        except (OSError, ValueError) as e:           # can't read score → can't tell if deficient
            who = f"{cell.harness}/{cell.task}/{cell.trial}"
            tqdm.write(f"  ! diagnose skip {who}: unreadable score ({e})")
            return None
        if not score.deficiencies:
            return None
        try:
            traj = TrajectoryRecord.model_validate_json(cell.trajectory.read_text())
            async with sem:
                findings = await diagnose_run(score, traj, task, lever_spec, llm)
            cell.findings.write_text(
                json.dumps([f.model_dump(mode="json") for f in findings], ensure_ascii=False))
            if findings:
                tags = ", ".join(attr_label(f.attribution) for f in findings)
                tqdm.write(f"  diagnosed {cell.harness}/{cell.task}/{cell.trial}: "
                           f"{len(findings)} finding(s) [{tags}]")
            return len(findings)
        except FatalLLMError:                        # config/auth: abort the whole diagnose pass
            raise
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  ! diagnose {cell.harness}/{cell.task}/{cell.trial}: {e}")
            return None

    log(f"diagnose: scanning {len(cells)} cells in {run_dir}")
    results: list[int | None] = []
    with tqdm(total=len(cells), desc="diagnose", unit="cell", disable=bar_disabled()) as bar:
        for fut in asyncio.as_completed([one(c) for c in cells]):
            results.append(await fut)
            bar.update(1)
    done = [r for r in results if r is not None]
    log(f"diagnose: {len(done)} deficient trials diagnosed, {sum(done)} findings written")
