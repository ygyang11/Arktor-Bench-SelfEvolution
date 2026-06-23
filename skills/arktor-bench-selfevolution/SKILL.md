---
name: arktor-bench-selfevolution
description: Drive Arktor-Bench-SelfEvolution to score an agent harness, work down a ranked backlog of its weaknesses, and improve it — human-in-the-loop or self-evolution.
---

# Arktor-Bench-SelfEvolution

## Overview

Arktor-Bench-SelfEvolution scores an agent **harness** — the scaffold (instructions, tools, loop,
context) wrapped around a fixed model — on a pack of from-scratch tasks. It produces
two things: a **scoreboard** of how the harness did, and a ranked **backlog** of the
weaknesses behind its lost points. The scoreboard is what you raise; the backlog is where
to raise it. This skill drives the loop: improve the harness against the backlog, re-run,
and keep a change when the scoreboard rises — repeat. You **orchestrate** by calling the
`arktor-bench` CLI; the CLI runs the benchmark and writes the results. The loop is
harness-agnostic. If the user only wants scores, run the benchmark, show the scoreboard,
and stop — do not start improving.

## Working mode

Users rarely state how they want to work, so settle this first — **ask if it is not
explicit**, before anything else.

- **Human-in-the-loop (default).** You analyse and propose; the user decides. Work the
  backlog top-down by impact: bring the user a concrete proposal — the weakness, the
  change you would make, and the evidence behind it — then ask. Apply it
  only once the user approves, and never edit harness code before that.
- **Self-evolution.** You run the loop yourself, working the backlog top-down by impact:
  make the change, re-run, keep it if the scores rose and drop it otherwise, and continue
  — reporting at each round instead of asking permission. You still obey every rule below
  and stop by the policy below, never running open-endedly.

### Stop policy

Neither grind on forever nor quit early — judge it each round:

- Keep going while the top remaining issues carry meaningful impact and recent changes
  are still raising the overall score.
- Stop when the largest remaining impact has fallen below a small threshold (diminishing
  returns), or several changes in a row fail to raise the score, or an agreed budget of
  iterations is spent.
- A single failed change is not a stop signal — drop it and move to the next issue;
  sustained lack of progress is.
- In human-in-the-loop mode, present this read each round — remaining impact, the recent
  score deltas, a continue-or-stop recommendation — and let the user decide.

## Rules

- **Fix the lever, not the task** — address the weakness in general, never by hard-coding
  task-specific behavior or tuning to the pack's cases; a fix that does not generalize is
  worthless.
- **Never game the metric** — the scoreboard is a proxy. Raising it by overfitting to the
  tasks, or by weakening anything that is graded, is a regression, not progress.
- **Ground every change in the evidence** — the issue's root causes, and the task or
  trajectory when you need more; never edit on speculation.
- **An external harness has no source you can edit** — give a lever-level recommendation
  instead.

## Commands

`arktor-bench` is the installed CLI. It reads config from `./arktor-bench.yaml` if
present, otherwise `~/.arktor-bench/arktor-bench.yaml`. In the commands below, `<pack>` is
the task set to evaluate on (including `general`, `communication`) and `<harness>` is the harness under test.

| command | what it does |
|---|---|
| `arktor-bench eval <pack> <harness>` | run the harness → diagnose failures → report the backlog; the normal entry point |
| `arktor-bench run <pack> <harness>` | run + score only — scoreboard, no backlog |
| `arktor-bench diagnose <run_dir>` | diagnose a run and write its findings (run again on a run_dir to refresh them) |
| `arktor-bench report <run_dir>` | build the backlog from a run's findings (run again to refresh it) |

`eval` and `run` take these options — each has a default, so pass one only to override:

- `--trials N` — runs per task (default 3).
- `--tasks a,b` — restrict to some tasks (default: the whole pack).
- `--out DIR` — where the run lands; defaults to a timestamped dir under
  `~/.arktor-bench/runs/`. Name it yourself when iterating.

## Result directory

A evaluation writes everything it produces here.

```
<out>/
  tasks.json                  # how this run was produced: pack, pack_dir (absolute), tasks, trials, harnesses
  scoreboard.md / .json       # cross-harness board: overall, label slices, per-task score·tokens, efficiency
  <harness>/
    backlog.md / .json        # this harness's ranked weaknesses — start here
    <task>/
      summary.json            # this task across trials: mean/std score, per-criterion means, mean tokens·steps
      <trial>/
        score.json            # this trial's per-criterion outcome
        trajectory.json       # the full run: per step, the model's thinking, reply, tool calls and their output
        metrics.json          # this trial's tokens, steps, wall_ms, cap_hit
        produced.json         # the agent's new/changed files vs baseline: [{path, status: created|modified}]
        findings.json         # this trial's diagnosed failures: criterion_ids, root_cause, attribution
        produced/             # the trial's output files — what the agent created or changed (plus any provided).
```

Read **`<harness>/backlog.md`** first. Each issue is a recurring weakness:

- `attribution` — a harness lever (`instructions`, `tools.{surface,schema,success,error}`,
  `loop`, `context`) or a model capability (`model.{reasoning,code,math,domain_knowledge,planning,tool_use}`),
  split into two sections in the backlog. The harness levers are your work; `model.*` you cannot
  fix by changing the scaffold — it's for model trainers, not you.
- `impact` — the scoreboard gain expected if it is resolved, as a fraction of total
  score; your priority order.
- `recurrence` — how many runs it was merged from (how systemic it is).
- `summary` — the underlying weakness it identifies.
- `Fix` — the change it proposes at that lever; weigh it, don't follow it blindly.
- `Evidence` — the per-run root causes, each tagged `task/trial(criteria)`.

Go deeper only when the backlog is not enough:

- what happened in a run → `<harness>/<task>/<trial>/trajectory.json`.
- what a task asks the agent to do → `<pack_dir>/<task>/task.md`.
- the current scores → `scoreboard.md`.

## Workflow

### Step 1 — Establish

Settle the working mode, the `<harness>`, the `<pack>`, and where runs go — name a
baseline `exp-1` and number iterations from there (`exp-2`, …; `exp-2-affected-<tasks>`
for a partial re-check). Note the defaults you are accepting (the whole pack, 3 trials),
and ask whether to run evals in the foreground or background — a full eval can take a while.

### Step 2 — Baseline

Produce the first run — `arktor-bench eval <pack> <harness> --out xx/exp-1` — or reuse a
current one. It is where you start.

### Step 3 — Iterate

Each round makes a change, re-runs to measure it, and keeps it only if the scoreboard rises:

1. **Change.** Improve the harness against the backlog, highest impact first. Human-in-the-loop:
   propose the change with its evidence and apply on approval. Self-evolution: apply it.
2. **Re-run** into the next directory — `arktor-bench eval <pack> <harness> --out xx/exp-2`.
   If you want a quick signal first, prefer re-run just the affected tasks (`--tasks
   <affected> --out xx/exp-2-affected-<tasks>`) and then full run when you have confidence.
3. **Verify** against the previous run: in the new one the targeted issues are gone from
   (or much smaller in) the backlog, the affected tasks' and overall scores rose, and nothing else
   regressed. Keep the change only if all hold — otherwise drop it.

### Step 4 — Stop and report

Stop when the stop policy fires, then report the whole run:
where the harness started and ended (the overall score before and after, and the net gain);
each round in order (the issue it targeted, the change, its score delta); and the backlog
now — which high-impact weaknesses are still open.
Point to the `xx/exp-*` directories so any round can be re-examined; in either mode this
write-up is the deliverable in xx/report.md.
