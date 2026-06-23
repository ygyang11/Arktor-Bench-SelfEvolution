# Arktor-Bench-SelfEvolution

Arktor-Bench-SelfEvolution evaluates the **harness (scaffold)**. The same model is dropped into different harnesses (Arktor / Codex CLI / Claude Code) and run over the same batch of from-scratch tasks, to locate where harness falls short and produce an actionable list of improvements.

The core difficulty in evaluating a harness is **attribution**: when a task fails, was it the scaffold that failed to carry the model, or a limitation of the model itself? Fusing *evaluation* and *attribution* — e.g. handing the harness a single synthetic score blended with execution quality — sacrifices cross-harness comparability and once again blurs which share of the failure belongs to the model. Arktor-Bench-SelfEvolution rests on **decoupling the two**: scoring (rubric + judge) answers only *how well the task was completed* — neutral across harnesses, one ruler for all; *who to charge for this failure* is left to an independent `diagnose` that follows the trajectory and attributes the cause to a specific lever, or rules it an inherent model limitation. Precisely because scoring never touches attribution, shortcomings sort cleanly — harness ones into each harness's backlog, model ones handed to the trainer. From the backlog the harness is then **iteratively optimized**: change the harness by impact, re-run, and keep the change when the scoreboard rises without regressing — round after round, closing a single evaluation into continuous evolution. **The bench is the fitness function; the harness is the scaffold under evolution.** A single evaluation thus extends naturally into a loop: **measure completion → attribute root cause → iteratively optimize**.

## Pipeline

`run → diagnose → report` (or the one-shot `eval` chain):

- **run** — execute the matrix of (harness × task × trial) in an isolated sandbox; a neutral judge scores task completion against each task's rubric.
- **diagnose** — for every failed criterion, independently attribute the root cause to a harness lever *or* a model capability.
- **report** — aggregate findings into a per-harness backlog, ranked by impact.

## Quick start

```bash
# install (Python 3.11)
pip install -e ".[dev]"

# statically validate the bundled smoke task — no LLM or Docker needed
arktor-bench validate test

# point the bench at its judge/diagnose LLM and the harnesses under test
mkdir -p ~/.arktor-bench
cp arktor-bench.example.yaml ~/.arktor-bench/arktor-bench.yaml   # then fill in your values

# run the full pipeline on the smoke task
arktor-bench eval test arktor,codex,claude_code --trials 3
```

`packs/test/` holds a single end-to-end smoke task (`T001_todo_cli`) for validating an install; the real task packs live beside it under `packs/`.
