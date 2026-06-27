from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

BENCH_HOME = Path.home() / ".arktor-bench"


class ModelEndpoint(BaseModel):
    model: str
    base_url: str
    api_key: str = ""
    model_config = {"extra": "forbid"}


class EndpointOverride(BaseModel):
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model_config = {"extra": "forbid"}


class HarnessInvocation(BaseModel):
    backend: Literal["docker", "local"] = "docker"
    model: str = ""
    image: str | None = None
    mounts: dict[str, str] = Field(default_factory=dict)   # container_path -> host_path, read-only
    files: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class HarnessConfigs(BaseModel):
    arktor: HarnessInvocation | None = None
    codex: HarnessInvocation | None = None
    claude_code: HarnessInvocation | None = None
    model_config = {"extra": "forbid"}


class BenchConfig(BaseModel):
    harness: HarnessConfigs = Field(default_factory=HarnessConfigs)
    judge: ModelEndpoint
    diagnose: EndpointOverride = Field(default_factory=EndpointOverride)
    docker_image: str = "arktor-bench:base"
    mem_limit: str | None = "2g"
    cpus: float = 0.0
    wall_s: int = 1800
    docker_concurrency: int = 3
    local_concurrency: int = 2
    llm_concurrency: int = 8
    model_config = {"extra": "forbid"}

    def harness_invocation(self, name: str) -> HarnessInvocation:
        inv: HarnessInvocation | None = getattr(self.harness, name, None)
        if inv is None:
            raise SystemExit(f"harness '{name}' not configured in arktor-bench.yaml")
        return inv

    @property
    def judge_endpoint(self) -> ModelEndpoint:
        return self.judge

    @property
    def diagnose_endpoint(self) -> ModelEndpoint:
        upd = {k: v for k, v in self.diagnose.model_dump().items() if v is not None}
        return self.judge.model_copy(update=upd)

    @classmethod
    def load(cls) -> BenchConfig:
        path = Path("arktor-bench.yaml")
        if not path.is_file():
            path = BENCH_HOME / "arktor-bench.yaml"
        if not path.is_file():
            raise SystemExit(
                f"no arktor-bench.yaml in cwd or {BENCH_HOME}; "
                "copy arktor-bench.example.yaml and fill it in"
            )
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: invalid yaml: {e}") from e
        return cls(**data)


@lru_cache
def get_config() -> BenchConfig:
    return BenchConfig.load()
