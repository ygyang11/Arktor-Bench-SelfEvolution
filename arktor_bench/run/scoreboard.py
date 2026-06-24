from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Literal

from pydantic import BaseModel

from arktor_bench.models import OutcomeScore, TaskLabels
from arktor_bench.run.layout import iter_cells, read_manifest
from arktor_bench.spec.loader import load_pack
from arktor_bench.trajectory.record import TokenUsage


class CellMetrics(BaseModel):
    tokens: TokenUsage
    steps: int
    wall_ms: int
    cap_hit: bool
    error: str | None = None


class CriterionSummary(BaseModel):
    id: str
    mode: Literal["auto", "judge"]
    weight: float
    mean: float
    scores: list[float]


class TaskSummary(BaseModel):
    task_id: str
    mean: float
    trials: list[float]
    std: float
    tokens_total: float
    steps: float
    criteria: list[CriterionSummary]


class Efficiency(BaseModel):
    tokens_in: float
    tokens_out: float
    tokens_total: float
    steps: float
    cap_hit_rate: float
    metered: int                                          # cells the token/step means are built on
    total: int                                            # cells attempted (metered + bad-debt)


class HarnessBoard(BaseModel):
    overall: float
    per_task: dict[str, float]
    per_task_tokens: dict[str, float]
    per_task_trials: dict[str, int]
    by_complexity: dict[str, float]
    by_domain: dict[str, float]
    by_harness_focus: dict[str, float]
    by_model_capability: dict[str, float]
    efficiency: Efficiency


class Scoreboard(BaseModel):
    tasks: list[str]
    trials: int
    harnesses: dict[str, HarnessBoard]


def _metered(m: CellMetrics) -> bool:                     # trustworthy cost needs a clean exit
    return not m.error and m.steps > 0


def build_summary(task_id: str, scores: list[OutcomeScore],
                  metrics: list[CellMetrics]) -> TaskSummary:
    trials = [s.score for s in scores]
    order: list[str] = []
    modes: dict[str, Literal["auto", "judge"]] = {}
    weights: dict[str, float] = {}
    acc: dict[str, list[float]] = {}
    for s in scores:
        for r in s.results:
            if r.errored:                                 # evaluator failures don't enter the mean
                continue
            if r.id not in acc:
                order.append(r.id)
                acc[r.id] = []
                modes[r.id] = r.mode
                weights[r.id] = r.weight
            acc[r.id].append(r.score)
    criteria = [
        CriterionSummary(id=cid, mode=modes[cid], weight=weights[cid],
                         mean=sum(acc[cid]) / len(acc[cid]), scores=acc[cid])
        for cid in order
    ]
    metered = [m for m in metrics if _metered(m)]         # exclude bad-debt cells from cost means
    n = len(metered) or 1
    return TaskSummary(
        task_id=task_id, mean=sum(trials) / len(trials) if trials else 0.0,
        trials=trials, std=pstdev(trials) if len(trials) > 1 else 0.0,
        tokens_total=sum(m.tokens.input + m.tokens.output for m in metered) / n,
        steps=sum(m.steps for m in metered) / n, criteria=criteria,
    )


def _slice(per_task: dict[str, float], labels: dict[str, TaskLabels],
           attr: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for tid, sc in per_task.items():
        lb = labels.get(tid)
        if lb is None:
            continue
        val = getattr(lb, attr)
        for v in (val if isinstance(val, list) else [val]):
            buckets[v.value].append(sc)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _efficiency(metrics: list[CellMetrics]) -> Efficiency:
    metered = [m for m in metrics if _metered(m)]         # cost means exclude bad-debt cells;
    n = len(metered) or 1                                 # cap_hit_rate stays over all attempts
    total = len(metrics) or 1
    return Efficiency(
        tokens_in=sum(m.tokens.input for m in metered) / n,
        tokens_out=sum(m.tokens.output for m in metered) / n,
        tokens_total=sum(m.tokens.input + m.tokens.output for m in metered) / n,
        steps=sum(m.steps for m in metered) / n,
        cap_hit_rate=sum(m.cap_hit for m in metrics) / total,
        metered=len(metered), total=len(metrics),
    )


def build_board(per_task: dict[str, float], per_task_tokens: dict[str, float],
                per_task_trials: dict[str, int],
                labels: dict[str, TaskLabels], metrics: list[CellMetrics]) -> HarnessBoard:
    return HarnessBoard(
        overall=sum(per_task.values()) / len(per_task) if per_task else 0.0,
        per_task=per_task,
        per_task_tokens=per_task_tokens,
        per_task_trials=per_task_trials,
        by_complexity=_slice(per_task, labels, "complexity"),
        by_domain=_slice(per_task, labels, "domain"),
        by_harness_focus=_slice(per_task, labels, "harness_focus"),
        by_model_capability=_slice(per_task, labels, "model_capability"),
        efficiency=_efficiency(metrics),
    )


def _num(v: float) -> str:
    return "-" if v != v else f"{v:.3f}"


def _cell(b: HarnessBoard, task: str, expected: int) -> str:
    if task not in b.per_task:
        return "-"
    got = b.per_task_trials.get(task, 0)
    short = f" ({got}/{expected})" if got < expected else ""   # flag a mean built on fewer trials
    return f"{b.per_task[task]:.3f} · {b.per_task_tokens.get(task, 0.0):.0f}{short}"


def _render(sb: Scoreboard) -> str:
    hs = list(sb.harnesses)
    head = "| | " + " | ".join(hs) + " |"
    sep = "|---" * (len(hs) + 1) + "|"
    out = ["# Scoreboard", "", "## Overall", head, sep,
           "| overall | " + " | ".join(f"{sb.harnesses[h].overall:.3f}" for h in hs) + " |"]
    for title, attr in (("By complexity", "by_complexity"), ("By domain", "by_domain"),
                        ("By harness_focus", "by_harness_focus"),
                        ("By model_capability", "by_model_capability")):
        keys = sorted({k for h in hs for k in getattr(sb.harnesses[h], attr)})
        out += ["", f"## {title}", head, sep]
        for r in keys:
            cells = " | ".join(
                _num(getattr(sb.harnesses[h], attr).get(r, float("nan"))) for h in hs)
            out.append(f"| {r} | {cells} |")
    out += ["", "## By task — score · tokens", head, sep]
    for r in sb.tasks:
        cells = " | ".join(_cell(sb.harnesses[h], r, sb.trials) for h in hs)
        out.append(f"| {r} | {cells} |")
    out += ["", "## Efficiency", head, sep]
    for field in ("tokens_in", "tokens_out", "tokens_total", "steps", "cap_hit_rate"):
        cells = " | ".join(f"{getattr(sb.harnesses[h].efficiency, field):.2f}" for h in hs)
        out.append(f"| {field} | {cells} |")
    metered = " | ".join(f"{(e := sb.harnesses[h].efficiency).metered}/{e.total}" for h in hs)
    out.append(f"| metered (cost means) | {metered} |")    # how many cells the cost means rest on
    return "\n".join(out)


def write_scoreboard(out: Path) -> Scoreboard:
    manifest = read_manifest(out)
    cells = list(iter_cells(out))
    labels = {t.id: t.labels for t in load_pack(manifest.pack, sorted({c.task for c in cells}),
                                                base=Path(manifest.pack_dir))}
    scores: dict[str, dict[str, list[OutcomeScore]]] = defaultdict(lambda: defaultdict(list))
    metrics: dict[str, dict[str, list[CellMetrics]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        try:  # read both before appending — keep the two lists aligned
            sc = OutcomeScore.model_validate_json(cell.score.read_text())
            mt = CellMetrics.model_validate_json(cell.metrics.read_text())
        except (OSError, ValueError) as e:            # drop the cell, but say which and why
            who = f"{cell.harness}/{cell.task}/{cell.trial}"
            print(f"scoreboard: dropping {who}: unreadable cell ({e})")
            continue
        scores[cell.harness][cell.task].append(sc)
        metrics[cell.harness][cell.task].append(mt)
    boards: dict[str, HarnessBoard] = {}
    tasks: set[str] = set()
    for harness, tmap in scores.items():
        per_task: dict[str, float] = {}
        per_task_tokens: dict[str, float] = {}
        per_task_trials: dict[str, int] = {}
        flat: list[CellMetrics] = []
        for tid, sc_list in tmap.items():
            summary = build_summary(tid, sc_list, metrics[harness][tid])
            (out / harness / tid / "summary.json").write_text(summary.model_dump_json(indent=2))
            per_task[tid] = summary.mean
            per_task_tokens[tid] = summary.tokens_total
            per_task_trials[tid] = len(sc_list)
            flat += metrics[harness][tid]
            tasks.add(tid)
        boards[harness] = build_board(per_task, per_task_tokens, per_task_trials, labels, flat)
    sb = Scoreboard(tasks=sorted(tasks), trials=manifest.trials, harnesses=boards)
    (out / "scoreboard.json").write_text(sb.model_dump_json(indent=2))
    (out / "scoreboard.md").write_text(_render(sb))
    return sb
