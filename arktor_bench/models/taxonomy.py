from __future__ import annotations

from enum import StrEnum


class Domain(StrEnum):
    SOFTWARE_ENGINEERING = "software_engineering"
    RESEARCH = "research"
    DATA = "data"


class ModelCapability(StrEnum):
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    PLANNING = "planning"
    TOOL_USE = "tool_use"


class Complexity(StrEnum):
    LINEAR = "linear"
    COMPOSITE = "composite"
    LONG_HORIZON = "long_horizon"


class Layer(StrEnum):
    INSTRUCTIONS = "instructions"
    TOOLS = "tools"
    LOOP = "loop"
    CONTEXT = "context"


class ToolLever(StrEnum):
    SURFACE = "surface"
    SCHEMA = "schema"
    SUCCESS = "success"
    ERROR = "error"
