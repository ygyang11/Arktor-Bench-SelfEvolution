from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable
from pathlib import Path

from pydantic import BaseModel, ValidationInfo, model_validator
from tqdm import tqdm

from arktor_bench.config import get_config
from arktor_bench.diagnose.levers import attr_label, load_annex
from arktor_bench.llm import FatalLLMError, StructuredLLM
from arktor_bench.models import Attribution, Finding
from arktor_bench.run.layout import iter_cells, read_manifest
from arktor_bench.run.scoreboard import CellMetrics
from arktor_bench.spec.loader import load_pack
from arktor_bench.utils.log import bar_disabled, log


class Issue(BaseModel):
    attribution: Attribution
    summary: str
    fix: str
    impact: float
    recurrence: int
    findings: list[Finding]


class RunFailure(BaseModel):
    task: str
    trial: str
    error: str
    ran: bool


def _impact(members: list[Finding], task_weight: dict[str, float],
            n_tasks: int, trials: int) -> float:
    by_task: dict[str, list[Finding]] = defaultdict(list)
    for m in members:
        by_task[m.task_id].append(m)
    total = 0.0
    for tid, ms in by_task.items():
        w = task_weight.get(tid, 0.0)
        if w > 0:
            total += (sum(m.recoverable_points for m in ms) / trials) / w
    return total / n_tasks if n_tasks else 0.0


class _Cluster(BaseModel):
    summary: str
    fix: str
    model_config = {"extra": "forbid"}


class _MergeOut(BaseModel):
    clusters: list[_Cluster]
    assignments: list[int]
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> _MergeOut:
        if not self.clusters:
            raise ValueError("clusters must be non-empty")
        if any(not 0 <= a < len(self.clusters) for a in self.assignments):
            raise ValueError(f"every assignment must be in [0, {len(self.clusters)})")
        n = (info.context or {}).get("n_findings")
        if n is not None and len(self.assignments) != n:
            raise ValueError(
                f"assignments must have exactly {n} entries (one per finding), "
                f"got {len(self.assignments)}")
        return self


_MERGE_SYSTEM = """\
You turn diagnostic findings — root causes from a agent failed, all sharing the one attribution \
shown at the top — into a small backlog of distinct, actionable issues. Each finding is one root \
cause.

Cluster by fix: merge two findings only when one fix would resolve both, and keep apart \
whatever needs a different fix — so paraphrases of one cause fold into a single issue \
while genuinely distinct problems stay separate, neither lumped (it hides work) nor \
split (it inflates the backlog).

Return:
- clusters — each one issue, with:
  - summary: the standing, durable problem it shares — true beyond any one run, in \
general terms, never a retelling of a run or a mention of the agent's generated functions or files.
  - fix: the durable remedy, in those same terms — never an edit to the agent's generated code
    - harness, code map given: the change to the harness code that owns the lever, and why.
    - harness, no code map (external CLI): the change to the lever, not code.
    - model: durable guidance for the model's trainer to strengthen the capability, NOT \
a code change.
- assignments — one entry per finding in listed order; assignments[k] is the cluster \
finding k belongs to. Length equals the number of findings; every value is a valid \
cluster index."""


def _dedup(group: list[Finding]) -> list[list[Finding]]:
    order: list[str] = []
    by_cause: dict[str, list[Finding]] = {}
    for f in group:
        key = f.root_cause.strip()
        if key not in by_cause:
            by_cause[key] = []
            order.append(key)
        by_cause[key].append(f)
    return [by_cause[k] for k in order]


async def _merge(
    group: list[Finding], task_weight: dict[str, float],
    n_tasks: int, trials: int, llm: StructuredLLM, annex: str,
) -> list[Issue]:
    a = group[0].attribution
    causes = _dedup(group)
    kind = "model" if a.model else "harness"
    listing = "\n".join(
        f"[{i}] {ms[0].root_cause.strip()}" for i, ms in enumerate(causes))
    code_map = "" if a.model else f"\n\n# Code map\n{annex}"
    prompt = f"# Attribution\nkind: {kind}\n{attr_label(a)}{code_map}\n\n# Findings\n{listing}"
    out = await llm.complete(prompt, _MergeOut, system=_MERGE_SYSTEM,
                             context={"n_findings": len(causes)})
    issues: list[Issue] = []
    for ci, cl in enumerate(out.clusters):
        members = [f for i, cid in enumerate(out.assignments) if cid == ci for f in causes[i]]
        if members:
            issues.append(Issue(
                attribution=a, summary=cl.summary, fix=cl.fix,
                impact=_impact(members, task_weight, n_tasks, trials),
                recurrence=len(members), findings=members,
            ))
    return issues


async def aggregate(
    findings_by_harness: dict[str, list[Finding]], task_weight: dict[str, float],
    n_tasks: int, trials: int, llm: StructuredLLM,
) -> dict[str, list[Issue]]:
    sem = asyncio.Semaphore(get_config().llm_concurrency)

    async def bucket(h: str, group: list[Finding]) -> tuple[str, list[Issue]]:
        async with sem:
            try:
                return h, await _merge(group, task_weight, n_tasks, trials, llm, load_annex(h))
            except FatalLLMError:                    # config/auth: abort the whole report
                raise
            except Exception as e:  # noqa: BLE001 — one bad bucket degrades to no issues
                tqdm.write(
                    f"  ! merge failed {h}/{attr_label(group[0].attribution)} "
                    f"({len(group)} findings kept in findings.json; re-run `report`): {e}")
                return h, []

    jobs: list[Awaitable[tuple[str, list[Issue]]]] = []
    for h, fs in findings_by_harness.items():
        buckets: dict[str, list[Finding]] = defaultdict(list)
        for f in fs:
            buckets[attr_label(f.attribution)].append(f)
        jobs += [bucket(h, g) for g in buckets.values()]

    result: dict[str, list[Issue]] = defaultdict(list)
    with tqdm(total=len(jobs), desc="merge", unit="bucket", disable=bar_disabled()) as bar:
        for fut in asyncio.as_completed(jobs):
            h, issues = await fut
            result[h].extend(issues)
            bar.update(1)
    for issues in result.values():
        issues.sort(key=lambda x: x.impact, reverse=True)
    return dict(result)


def _issue_section(title: str, action: str, issues: list[Issue]) -> list[str]:
    out = [f"## {title}", ""]
    if not issues:                                     # keep the heading, say so explicitly
        return out + ["_None found in this run._", ""]
    for n, i in enumerate(issues, 1):
        evidence: dict[str, set[str]] = {}
        for f in i.findings:
            evidence.setdefault(f.root_cause.strip(), set()).add(
                f"{f.task_id}/{f.trial}({'/'.join(f.criterion_ids)})")
        out += [f"### {n}. {attr_label(i.attribution)} · impact {i.impact:.3f} · "
                f"recurrence {i.recurrence}",
                f"**{i.summary}**", "", f"{action}: {i.fix}", "", "Evidence:"]
        out += [f"- [{', '.join(sorted(where))}] {cause}" for cause, where in evidence.items()]
        out.append("")
    return out


def _run_failures_section(faults: list[RunFailure]) -> list[str]:
    out = ["## Run failures", "",                       # one section; ran tells the two kinds apart
           "Cells that did not complete cleanly — investigate before trusting their row.", ""]
    out += _by_error([f for f in faults if not f.ran],
                     "produced no trajectory and scored 0 — not diagnosed a real weakness "
                     "(infra or harness-launch)")
    out += _by_error([f for f in faults if f.ran],
                     "ran and was scored, but exited non-clean — cost is untrustworthy "
                     "and excluded from the means")
    out.append("")
    return out


def _by_error(faults: list[RunFailure], note: str) -> list[str]:
    by_err: dict[str, list[str]] = {}
    for rf in faults:
        by_err.setdefault(rf.error, []).append(f"{rf.task}/{rf.trial}")
    return [f"- [{', '.join(sorted(where))}] {err} — {note}" for err, where in by_err.items()]


def _render_backlog(harness: str, issues: list[Issue], faults: list[RunFailure]) -> str:
    harness_issues = [i for i in issues if i.attribution.layer is not None]
    model_issues = [i for i in issues if i.attribution.model is not None]
    out = [f"# Backlog — {harness}", ""]
    if not (issues or faults):
        out += ["No issues found in this run — no diagnosed weaknesses and no run failures.", ""]
        return "\n".join(out)
    out += _issue_section("Harness issues", "Fix", harness_issues)   # both sections always render,
    out += _issue_section(                                            # empty one says "None found"
        "Model capabilities (not a harness fix — model-trainer guidance)",
        "Guidance", model_issues)
    if faults:
        out += _run_failures_section(faults)
    return "\n".join(out)


async def build_report(run_dir: Path, llm: StructuredLLM) -> None:
    manifest = read_manifest(run_dir)
    by_harness: dict[str, list[Finding]] = defaultdict(list)
    faults: dict[str, list[RunFailure]] = defaultdict(list)
    present: set[str] = set()
    harnesses: set[str] = set()
    for cell in iter_cells(run_dir):
        present.add(cell.task)
        harnesses.add(cell.harness)
        try:  # a cell that didn't complete cleanly (the complement of a metered cell) — surface it
            m = CellMetrics.model_validate_json(cell.metrics.read_text())
            if m.error or m.steps == 0:                  # ran tells no-trajectory from degraded
                faults[cell.harness].append(RunFailure(
                    task=cell.task, trial=cell.trial, error=m.error or "unknown", ran=m.steps > 0))
        except (OSError, ValueError):
            pass
        if not cell.findings.is_file():
            continue
        try:  # materialize fully — no partial extend on a bad row
            found = [Finding.model_validate(x) for x in json.loads(cell.findings.read_text())]
        except (OSError, ValueError) as e:  # missing/corrupt findings.json → skip this cell
            log(f"report: skipping findings {cell.harness}/{cell.task}/{cell.trial}: {e}")
            continue
        by_harness[cell.harness].extend(found)
    tasks = load_pack(manifest.pack, sorted(present), base=Path(manifest.pack_dir))
    task_weight = {t.id: t.total_weight for t in tasks}
    log(f"report: aggregating findings for {sorted(by_harness) or 'no deficiencies'}")
    backlog = await aggregate(by_harness, task_weight, len(tasks), manifest.trials, llm)
    for harness in sorted(harnesses | set(backlog) | set(faults)):     # every harness in the run
        issues = backlog.get(harness, [])
        fs = faults.get(harness, [])
        d = run_dir / harness
        d.mkdir(parents=True, exist_ok=True)
        (d / "backlog.json").write_text(json.dumps({
            "issues": [i.model_dump(mode="json") for i in issues],
            "run_failures": [f.model_dump() for f in fs],
        }, ensure_ascii=False, indent=2))
        (d / "backlog.md").write_text(_render_backlog(harness, issues, fs))
        log(f"report: {harness} -> {len(issues)} issues, {len(fs)} run-failures "
            f"({d / 'backlog.md'})")
