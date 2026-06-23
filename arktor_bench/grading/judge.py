from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

from arktor_bench.llm import StructuredLLM
from arktor_bench.models import CriterionResult, Produced, TaskSpec
from arktor_bench.utils.token_counter import clip_tokens

_ARTIFACT_TOKENS = 5000
_BINARY_SAMPLE = 8192


_SYSTEM = """\
You are a rigorous, impartial evaluator. You are given a task, a rubric of criteria, and \
the deliverable an AI agent produced for it; you decide how well that deliverable \
satisfies each criterion.

Judge the final artifacts by the evidence in them. Each artifact is tagged [created] (the \
agent wrote it), [modified] (the agent changed a provided file), or [provided] (given to \
the agent, unchanged).

Scoring rules:
- Score each criterion ONLY against its own rubric levels — never on general \
impression, and never on assumed intent.
- Match the artifact to the level whose defining behavior it actually shows: judge \
by what it substantively does, not by incidental details or partial resemblance.
- When the artifact falls genuinely between two adjacent levels, take the lower.

Fairness rules:
- Independence: never let a strong or weak showing on one criterion shift your \
judgment of another — no halo, no cross-criterion averaging.
- Ignore surface style: formatting, comments, tone, and authoritative-sounding \
claims are not evidence; a claim the work makes about itself earns nothing. \
Judge only what the artifacts actually do and contain.
- Length neutrality: at equal substance a concise artifact scores no lower than \
a verbose one; never reward length or effort.

For each criterion, first state the rationale in general terms — a brief, accurate, faithful \
description of the decisive behavior the artifact shows or lacks — rather than quoting \
its code or specific identifiers. Then choose its level number."""

_USER = """\
# Task
{task}

# Criteria and rubric levels
{rubric}

# Reference solution
{reference}

# Candidate artifacts
{artifacts}

For every criterion id, give its brief rationale then choose the level number."""


def _looks_binary(raw: bytes) -> bool:
    chunk = raw[:_BINARY_SAMPLE]
    if b"\x00" in chunk:
        return True
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _load_text(path: Path) -> str:
    if not path.is_file():
        return "[file not present]"
    raw = path.read_bytes()
    if _looks_binary(raw):
        return f"[binary file, {len(raw)} bytes, not shown]"
    return clip_tokens(raw.decode("utf-8", errors="replace"), _ARTIFACT_TOKENS)


def _judge_model(task: TaskSpec) -> type[BaseModel]:
    crit = {
        c.id: (create_model(
            f"_Crit_{c.id}",
            rationale=(str, ...),
            level=(int, Field(ge=1, le=len(c.levels))),
            __config__=ConfigDict(extra="forbid"),
        ), ...)
        for c in task.judge
    }
    out_model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        "_JudgeOut", __config__=ConfigDict(extra="forbid"), **crit)
    return out_model


def _rubric(task: TaskSpec) -> str:
    blocks = []
    for c in task.judge:
        lvls = "\n".join(f"  Level {i} (score {lv.score}): {lv.desc}"
                         for i, lv in enumerate(c.levels, 1))
        blocks.append(f"[{c.id}] {c.desc}\n{lvls}")
    return "\n\n".join(blocks)


def _gather(task: TaskSpec, produced: list[Produced]) -> list[tuple[str, str]]:
    pool: dict[str, str] = {p.path: p.status for p in produced}
    for w in task.workspace_files:
        if w.dest not in pool:
            pool[w.dest] = "provided"
    wanted: dict[str, str] = {}
    for c in task.judge:
        pats = c.artifacts or ["*"]
        for path, tag in pool.items():
            if any(fnmatch(path, pat) for pat in pats):
                wanted[path] = tag
    return sorted(wanted.items())


def _artifacts(produced_root: Path, gathered: list[tuple[str, str]]) -> str:
    if not gathered:
        return "No artifacts were produced."
    return "\n\n".join(
        f"### {path} [{tag}]\n{_load_text(produced_root / path)}" for path, tag in gathered)


def _reference(task: TaskSpec) -> str:
    ref = task.dir / "reference"
    files = sorted(p for p in ref.rglob("*") if p.is_file()) if ref.is_dir() else []
    if not files:
        return "No reference provided; score strictly against the rubric levels."
    body = "\n\n".join(f"--- {p.relative_to(ref)} ---\n{_load_text(p)}" for p in files)
    return f"A correct solution that would earn the top level on every criterion:\n{body}"


async def judge_task(
    task: TaskSpec, produced: list[Produced], produced_root: Path, llm: StructuredLLM,
) -> list[CriterionResult]:
    prompt = _USER.format(
        task=task.prompt, rubric=_rubric(task),
        reference=_reference(task), artifacts=_artifacts(produced_root, _gather(task, produced)),
    )
    out = await llm.complete(prompt, _judge_model(task), system=_SYSTEM)
    results: list[CriterionResult] = []
    for c in task.judge:
        picked = getattr(out, c.id)
        results.append(CriterionResult(
            id=c.id, mode="judge", weight=c.weight,
            score=c.levels[picked.level - 1].score, message=picked.rationale,
        ))
    return results
