from __future__ import annotations

from arktor_bench.models import Attribution

LEVER_SPEC = """\
A run can fail on two sides. The harness — the scaffold around the model: \
instructions, tools (sub-levers surface, schema, success, error), loop, context. \
Or the model itself — a capability it fell short on: reasoning, code, math, \
domain_knowledge, planning, tool_use. Attribute each finding to exactly one target from \
this closed set on the evidence, and do not invent targets. Hold both sides to the same bar.

How to attribute
- Place it on the side the fix belongs to: the harness when telling the model how to work or \
giving it what it lacked would have prevented the failure; the model when it was already \
equipped to succeed and only better craft in the work itself would have.
- Name the single lever or capability the failure traces to, not the visible \
symptom — the one whose change would stop it recurring.
- Derive each label from the rules above, the same way each time — the same cause recurs \
across runs judged separately, so only a rule applied identically keeps it to a single label.
- One failure, one attribution; a criterion that failed for several independent \
causes gets a separate finding for each.

The harness levers

A harness lever is the cause when the scaffold fell short of what the model needed, \
and a concrete change to that lever would have prevented the failure. Name the lever:

instructions — What the system prompt tells the model about how to work: its role, \
its principles, and how to use its tools (which tool to reach for, the intended way \
of working). General guidance, not the spec of any one tool (that is tools.schema) \
nor the control loop (that is loop). For example: reaching for a raw shell command \
where a dedicated tool exists is instructions; calling the right tool wrongly \
because its interface was unclear is tools.schema; finishing without checking the work against \
the task, which a general verify-before-finishing instruction would prevent, is \
instructions, not model.planning.

tools — A specific tool as the model meets it: finding it, calling it, and reading what \
comes back. Four sub-levers, each the cause when:
  surface — a tool the task needs is missing, or present but the model never finds it.
  schema — the tool is found, but its declared interface — the name, description, and \
parameters the model is given to call it by — does not make clear when and how to call it \
correctly.
  success — a call succeeds, but its result omits or buries the information the model \
needed.
  error — a call fails, but its message does not show what went wrong or how to \
recover.

loop — The harness's control over the run: noticing the model is stuck, nudging or \
stopping it, and deciding when the run ends. The cause only when that control \
misjudged — it ended a run still making progress, or never intervened in one plainly \
stuck. A model behaving badly inside a working loop is instructions or tools.

context — What the harness keeps in the model's window across turns: what \
accumulates, what is compressed when it fills, and what is injected. The cause when \
information the model needed was dropped, never included, or crowded out.

The model capabilities

A failure here is the model's own: the scaffold was adequate and it still fell \
short. Name the capability:
  reasoning — flawed logic, analysis, or inference, with no misleading cue from the \
scaffold.
  code — flawed implementation, structure, or use of the language and its libraries — \
the coding ability itself, on a clear task with working tools.
  math — flawed calculation, derivation, or quantitative argument — the mathematical \
work itself, not the general reasoning around it.
  domain_knowledge — a fact the task needed that the model lacked and the context was \
not meant to supply.
  planning — failed to break the task into the right steps or sequence them sensibly — \
what to do and in what order (not code organization), though the loop gave the run room to finish.
  tool_use — the model mishandled adequate tools: it never reached for an obvious tool \
or ignored results or errors it could have acted on.

tools versus tool_use is the easy confusion. A tool's interface and the results it \
returns are the harness's; using an adequate interface and adequate results well is \
the model's. You cannot see the declared interface directly — infer its adequacy from \
the run: if the results and errors the model received were enough to act on and it \
still mis-called or ignored the tool, it is tool_use; if a needed tool was missing, \
its interface unusable, or a result or error uninformative, it is the tools lever."""

ARKTOR_ANNEX = """\
Code map for Arktor — where each lever's code lives. The named area owns the lever; \
read it to understand the issue and make the change.

instructions   The agent's behavioral guidance in the system prompt — role/intro, \
working guidelines, per-tool usage notes, and skill instructions — assembled in \
agent_harness/prompt/sections.py and agent_harness/prompt/instructions.py.
tools.surface  Which tools are built for the agent and exposed to it: the tool set \
in agent_app/tools/ and the registry in agent_harness/tool/registry.py.
tools.schema   A tool's declared name, description, and parameters as the model \
sees them — defined by each tool in agent_app/tools/<tool> (on the ToolSchema type \
in agent_harness/tool/base.py).
tools.success  What a tool returns to the model on success, and how compactly — \
the result each tool builds in agent_app/tools/<tool>.
tools.error    A tool's failure message and whether it guides recovery — the error \
each tool raises in agent_app/tools/<tool>; the generic wrapper for uncaught errors \
is agent_harness/tool/executor.py.
loop           The ReAct cycle and its guards: the step in agent_harness/agent/\
react.py, the stuck-detector in agent_harness/utils/loop_detector.py, and the run \
loop and self-heal in agent_harness/agent/base.py.
context        What enters the window over time — accumulation, compression, and \
injection — in agent_harness/context/context.py and the compressor in \
agent_harness/memory/."""

_EXTERNAL = """\
This harness is a third-party agent CLI; its source code is not available. Give \
each fix at the lever level: the concrete change this harness would need at the \
attributed lever for the failure to stop recurring (for example, a tool error \
message that surfaces the failing input, a clearer tool description, or a stop \
condition that triggers sooner)."""

_ANNEXES: dict[str, str] = {
    "arktor": ARKTOR_ANNEX,
    "codex": _EXTERNAL,
    "claude_code": _EXTERNAL,
}


def load_lever_spec() -> str:
    return LEVER_SPEC


def load_annex(harness: str) -> str:
    return _ANNEXES.get(harness, "")


def attr_label(a: Attribution) -> str:
    if a.model is not None:
        return f"model.{a.model.value}"
    if a.layer is None:
        raise ValueError("attribution has neither a harness layer nor a model capability")
    return f"{a.layer.value}.{a.tool.value}" if a.tool else a.layer.value
