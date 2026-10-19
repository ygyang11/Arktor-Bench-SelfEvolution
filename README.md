<h2 align="center">Arktor-Bench-SelfEvolution</h2>

<p align="center">
  <a href="https://github.com/ygyang11/Arktor-Bench-SelfEvolution/actions/workflows/ci.yml"><img src="https://github.com/ygyang11/Arktor-Bench-SelfEvolution/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"></a>
  <a href="README_ZH.md"><img src="https://img.shields.io/badge/%F0%9F%93%84_%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3-click-lightgrey" alt="中文文档"></a>
</p>

Arktor-Bench-SelfEvolution is a framework for evaluating and improving **agent
harnesses**. It separates task scoring from trajectory-based
diagnosis, providing comparable results and evidence to guide further
development.

<p align="center"><code>run → diagnose → report</code></p>

## Features

### Evaluation capabilities

- **Outcome-based grading.** Score task deliverables with executable checks
  and rubric-based LLM judging, using weighted criteria.

- **Trajectory-based diagnosis.** Examine failed criteria and execution traces
  to distinguish harness shortcomings from model capability limitations.

- **Comparative reports.** Compare scores and execution metrics, and aggregate
  recurring findings into a prioritized backlog for each harness.

- **Harness comparisons.** Evaluate Arktor, Codex CLI, Claude Code, and custom
  Arktor SDK methods on shared task packs, with repeated trials.

### Framework design

- **Task packs.** Organize task instructions, evaluation criteria, and supporting
  resources in a shared format.

- **Execution backends.** Run in fresh workspaces through Docker or a local
  Conda environment, with configurable time and concurrency limits.

- **Structured evidence.** Record standardized trajectories, criterion scores,
  produced artifacts, token usage, and execution metrics for further analysis.

## Getting started

### Installation

Install from source in your Python environment:

```bash
git clone https://github.com/ygyang11/Arktor-Bench-SelfEvolution.git
cd Arktor-Bench-SelfEvolution
python -m pip install -e ".[dev]"
```

Validate the bundled task specification locally, without a model API or Docker:

```bash
arktor-bench validate test
```

### Configuration

Create a user configuration from the repository root:

```bash
mkdir -p ~/.arktor-bench
cp arktor-bench.example.yaml ~/.arktor-bench/arktor-bench.yaml
```

For project-specific settings, place `arktor-bench.yaml` in the working directory;
the CLI gives this file priority.

Set `judge` to an OpenAI-compatible endpoint. Diagnosis and reporting reuse it
unless fields are overridden in `diagnose`:

```yaml
judge:
  model: "__SET_JUDGE_MODEL__"
  base_url: "__SET_BASE_URL__"
  api_key: "__SET_API_KEY__"
diagnose: {}
```

Configure the harnesses you intend to evaluate, including their models and
authentication, under `harness`. See
[arktor-bench.example.yaml](arktor-bench.example.yaml) for the supported settings.

### Execution environment

With Docker running, build the supplied image, which includes the supported
agent CLIs:

```bash
docker build -f docker/Dockerfile.base -t arktor-bench:base .
```

For `backend: local`, create the environment defined in
[environment.yml](environment.yml) and install the selected harness in it:

```bash
conda env create -f environment.yml
```

## CLI

From the repository root, run the complete pipeline on the bundled smoke task
with a configured harness:

```bash
arktor-bench eval test arktor --trials 1
```

Use comma-separated names to compare harnesses, such as `arktor,codex,claude_code`.

| Command | Purpose |
| --- | --- |
| `validate <pack>` | Validate task specifications and grader declarations. |
| `run <pack> <harnesses>` | Execute tasks and score their deliverables. |
| `diagnose <run_dir>` | Analyse failed criteria using execution trajectories. |
| `report <run_dir>` | Aggregate findings into a backlog for each harness. |
| `eval <pack> <harnesses>` | Run execution, scoring, diagnosis, and reporting together. |

Run `arktor-bench --help` for CLI usage.

## Task packs and extension

### Define a task pack

Use the [bundled task](packs/test/T001_todo_cli/) as a starting point. Each task
has a `task.md` specification containing its prompt, labels, and grading criteria;
`grader.py` implements any executable checks. Tasks can also provide input files,
setup steps, and reference artifacts.

Place task directories under `packs/<pack>/`, then check them with
`arktor-bench validate <pack>` before evaluating the pack.

### Connect and extend harnesses

Built-in [harness adapters](arktor_bench/harness/) support Arktor, Codex CLI,
and Claude Code. Extend the adapters and [configuration](arktor_bench/config.py)
to connect other harnesses, or adapt execution, grading, and analysis to your
own experiments.

Custom Arktor SDK methods can also be evaluated through the
[SDK adapter](arktor_bench/harness/arktor_sdk.py).

## Citation

To cite Arktor-Bench-SelfEvolution, use the entry below or
[CITATION.cff](CITATION.cff).

```bibtex
@software{arktor_bench,
  author = {{ygyang11}},
  title  = {Arktor-Bench-SelfEvolution},
  url    = {https://github.com/ygyang11/Arktor-Bench-SelfEvolution}
}
```

## License

Released under the [MIT License](LICENSE).
