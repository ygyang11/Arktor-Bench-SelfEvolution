<h2 align="center">Arktor-Bench-SelfEvolution</h2>

<p align="center">
  <a href="https://github.com/ygyang11/Arktor-Bench-SelfEvolution/actions/workflows/ci.yml"><img src="https://github.com/ygyang11/Arktor-Bench-SelfEvolution/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="许可证：MIT"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/%F0%9F%93%84_English-click-lightgrey" alt="English"></a>
</p>

Arktor-Bench-SelfEvolution 是一个用于评估和改进 **Agent Harness** 的框架。
任务评分与执行轨迹诊断相互独立，为不同 Harness 提供可比较的结果，并为后续改进提供依据。

<p align="center"><code>run → diagnose → report</code></p>

## 特性

### 评估能力

- **任务评分。** 结合 Auto Checks 与 LLM judge，依据 rubric 评价任务产出，
  按各评分项的权重汇总结果。

- **执行轨迹诊断。** 结合未通过的评分项与执行轨迹分析问题，
  区分 Harness 的不足和模型能力的局限。

- **对比与报告。** 比较得分和执行指标，汇总反复出现的问题，
  为各 Harness 生成按优先级排序的改进清单。

- **Harness 对比。** 在同一任务包上重复运行 Arktor、Codex CLI、
  Claude Code 及自定义 Arktor SDK 方法。

### 框架设计

- **任务。** 使用统一格式组织任务描述、评分标准与相关资源。

- **执行后端。** 支持 Docker 和本地 Conda 环境，每次运行使用独立的新工作目录，
  可配置运行时限与并发数。

- **结构化记录。** 保存标准化执行轨迹、评分明细、任务产出、
  token 用量与执行指标，便于进一步分析。

## 快速开始

### 安装

在 Python 环境中从源码安装：

```bash
git clone https://github.com/ygyang11/Arktor-Bench-SelfEvolution.git
cd Arktor-Bench-SelfEvolution
python -m pip install -e ".[dev]"
```

先校验仓库内的测试任务，无需调用模型 API 或启动 Docker：

```bash
arktor-bench validate test
```

### 配置

在仓库根目录执行以下命令，创建用户配置文件：

```bash
mkdir -p ~/.arktor-bench
cp arktor-bench.example.yaml ~/.arktor-bench/arktor-bench.yaml
```

如需单独配置某个项目，可在工作目录放置 `arktor-bench.yaml`，CLI 会优先读取此文件。

在 `judge` 中配置 OpenAI 兼容服务。诊断和报告默认复用该配置，
也可通过 `diagnose` 覆盖其中的字段：

```yaml
judge:
  model: "__SET_JUDGE_MODEL__"
  base_url: "__SET_BASE_URL__"
  api_key: "__SET_API_KEY__"
diagnose: {}
```

在 `harness` 下配置待评估的 Harness，包括模型和认证信息。
完整配置见 [arktor-bench.example.yaml](arktor-bench.example.yaml)。

### 执行环境

启动 Docker 后，构建仓库提供的镜像，其中已包含支持的 Agent CLI：

```bash
docker build -f docker/Dockerfile.base -t arktor-bench:base .
```

使用 `backend: local` 时，根据 [environment.yml](environment.yml) 创建环境，
并在其中安装所选 Harness：

```bash
conda env create -f environment.yml
```

## CLI

在仓库根目录，用已配置的 Harness 对内置 smoke test 任务运行完整流程：

```bash
arktor-bench eval test arktor --trials 1
```

使用逗号分隔的名称比较多个 Harness，例如 `arktor,codex,claude_code`。

| 命令 | 用途 |
| --- | --- |
| `validate <pack>` | 校验任务规范与 grader 声明。 |
| `run <pack> <harnesses>` | 运行任务并对任务产出评分。 |
| `diagnose <run_dir>` | 根据执行轨迹分析未通过的评分项。 |
| `report <run_dir>` | 汇总诊断结果，生成各 Harness 的改进清单。 |
| `eval <pack> <harnesses>` | 连续执行任务、评分、诊断与报告。 |

使用 `arktor-bench --help` 查看 CLI 用法。

## 任务与扩展

### 定义任务

可参考仓库内的[测试任务](packs/test/T001_todo_cli/)。每道任务通过 `task.md` 定义 prompt、
标签与评分标准，通过 `grader.py` 实现自动检查，
还可包含输入文件、环境准备步骤和参考产出。

将任务目录放在 `packs/<pack>/` 下，并在评估前通过
`arktor-bench validate <pack>` 校验任务。

### 接入与扩展 Harness

内置 [Harness adapter](arktor_bench/harness/) 支持 Arktor、Codex CLI 与 Claude Code。
可扩展 adapter 和[配置](arktor_bench/config.py)接入其他 Harness，
也可根据实验需要调整任务执行、评分和分析组件。

自定义 Arktor SDK 方法也可通过 [SDK adapter](arktor_bench/harness/arktor_sdk.py) 进行评估。

## 引用

引用 Arktor-Bench-SelfEvolution 时，可使用以下条目或 [CITATION.cff](CITATION.cff)。

```bibtex
@software{arktor_bench,
  author = {{ygyang11}},
  title  = {Arktor-Bench-SelfEvolution},
  url    = {https://github.com/ygyang11/Arktor-Bench-SelfEvolution}
}
```

## 许可证

本项目采用 [MIT License](LICENSE)。
