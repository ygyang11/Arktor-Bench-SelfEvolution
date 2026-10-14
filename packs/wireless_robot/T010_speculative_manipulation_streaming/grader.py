from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

SCENES = ("free_motion", "contact_transition", "multi_stage", "high_latency", "drift_recovery")
HORIZON = 48
DELAYS = {
    "free_motion": 2,
    "contact_transition": 4,
    "multi_stage": 3,
    "high_latency": 6,
    "drift_recovery": 4,
}
CONTACT = {
    "free_motion": ((20.0, 20.0), 0.0),
    "contact_transition": ((2.35, 1.45), 0.85),
    "multi_stage": ((5.05, 1.05), 0.90),
    "high_latency": ((2.35, 1.45), 0.85),
    "drift_recovery": ((6.0, 2.8), 0.95),
}
DISTURBANCES = {
    "free_motion": ({15, 16}, (0.06, -0.03)),
    "contact_transition": ({17, 18, 19}, (-0.10, 0.05)),
    "multi_stage": ({12, 13, 30}, (0.08, 0.08)),
    "high_latency": ({8, 9, 10, 27}, (-0.08, 0.06)),
    "drift_recovery": ({22, 23, 24}, (0.12, -0.10)),
}
STAGES = (
    (0, 15, (0.0, 0.0), (2.5, 1.5)),
    (16, 31, (2.5, 1.5), (5.8, 0.8)),
    (32, 47, (5.8, 0.8), (8.6, 4.8)),
)
TRACE_KEYS = {
    "t",
    "target",
    "position",
    "velocity",
    "segment",
    "cache_size",
    "cache_head",
    "cache_error",
    "decision",
    "action",
    "request_depth",
    "request_id",
    "post_position",
    "post_velocity",
    "safety_state",
    "segment_complete",
}
HEAD_KEYS = {
    "request_id",
    "command_index",
    "apply_t",
    "action",
    "nominal_position",
    "nominal_velocity",
}
REQUEST_KEYS = {
    "request_id",
    "send_t",
    "base_position",
    "base_velocity",
    "depth",
    "delay",
    "arrival_t",
    "status",
    "commands",
}
COMMAND_KEYS = {"command_index", "apply_t", "action", "nominal_position", "nominal_velocity"}
EVENT_KEYS = {"t", "event", "request_id", "command_index", "cache_size", "reason"}
SEGMENT_KEYS = {"segment", "start_t", "end_t", "target_end", "finish_t", "completed", "max_error"}
VALID_ACTIONS = {-1, 0, 1}
EPS = 1e-6
ERROR_LIMIT = 0.15
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scenes": 0.33,
    },
    "result_evidence": {
        "result_evidence.scene_shape": 0.1,
        "result_evidence.trace_shape": 0.1,
        "result_evidence.request_shape": 0.1,
        "result_evidence.event_shape": 0.1,
        "result_evidence.segment_shape": 0.1,
        "result_evidence.field_values": 0.1,
        "result_evidence.ordering": 0.1,
        "result_evidence.links": 0.1,
        "result_evidence.finite_values": 0.1,
        "result_evidence.nonempty_evidence": 0.1,
    },
    "decision_legality": {
        "decision_legality.action_relations": 0.16666666666666666,
        "decision_legality.cache_validation": 0.16666666666666666,
        "decision_legality.request_depth": 0.16666666666666666,
        "decision_legality.request_spacing": 0.16666666666666666,
        "decision_legality.request_payload": 0.16666666666666666,
        "decision_legality.information_time": 0.16666666666666666,
    },
    "replay_validity": {
        "replay_validity.state": 0.1111111111111111,
        "replay_validity.nominal_fragments": 0.1111111111111111,
        "replay_validity.cache_events": 0.1111111111111111,
        "replay_validity.dynamics": 0.1111111111111111,
        "replay_validity.safety": 0.1111111111111111,
        "replay_validity.stage_records": 0.1111111111111111,
        "replay_validity.requests": 0.1111111111111111,
        "replay_validity.trace_evidence": 0.1111111111111111,
        "replay_validity.ordering": 0.1111111111111111,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = bool(passed)
        self.detail = detail
        self.share = share


class EvalState:
    def __init__(self) -> None:
        self.run_started = False
        self.run_finished = False
        self.exit_code: int | None = None
        self.timed_out = False
        self.elapsed_s: float | None = None
        self.official_ok = False
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.data: Any = None
        self.parse_ok = False
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, (int, float)) or _finite(value)


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    key = f"{group}.{name}"
    return AtomVerdict(key, passed, detail, ATOM_SHARES[group][key])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("atom shares must sum to one")
    if any(
        not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from declaration")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return (
        "failed_atoms=["
        + ",".join(atom.id for atom in failed)
        + "]; "
        + " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    )


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    return (
        data.get("scenarios", [])
        if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list)
        else []
    )


def _stage(t: int) -> int:
    for index, (start, end, _, _) in enumerate(STAGES):
        if start <= t <= end:
            return index
    return 2


def _target(t: int) -> tuple[float, float]:
    start, end, p0, p1 = STAGES[_stage(t)]
    alpha = (t - start + 1) / (end - start + 1)
    return (p0[0] + alpha * (p1[0] - p0[0]), p0[1] + alpha * (p1[1] - p0[1]))


def _contact(scene: str, position: tuple[float, float]) -> bool:
    centre, radius = CONTACT[scene]
    return (position[0] - centre[0]) ** 2 + (position[1] - centre[1]) ** 2 <= radius * radius + EPS


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _action_ok(action: Any) -> bool:
    return (
        isinstance(action, list)
        and len(action) == 2
        and all(_integer(x) and x in VALID_ACTIONS for x in action)
    )


def _transition(
    scene: str,
    position: tuple[float, float],
    velocity: tuple[float, float],
    action: list[int],
    slot: int | None = None,
    disturbance: bool = False,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    gain = 0.55 if _contact(scene, position) else 1.0
    next_velocity = (
        _clip(0.85 * velocity[0] + 0.18 * gain * action[0], -1.5, 1.5),
        _clip(0.85 * velocity[1] + 0.18 * gain * action[1], -1.5, 1.5),
    )
    next_position = (position[0] + next_velocity[0], position[1] + next_velocity[1])
    if disturbance and slot is not None and slot in DISTURBANCES[scene][0]:
        d = DISTURBANCES[scene][1]
        next_position = (next_position[0] + d[0], next_position[1] + d[1])
    clipped = (_clip(next_position[0], 0.0, 10.0), _clip(next_position[1], 0.0, 8.0))
    violation = (
        abs(clipped[0] - next_position[0]) > EPS
        or abs(clipped[1] - next_position[1]) > EPS
        or max(abs(next_velocity[0]), abs(next_velocity[1])) > 1.5 + EPS
    )
    state = "violation" if violation else ("contact" if _contact(scene, position) else "normal")
    return clipped, next_velocity, state


def _head(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "request_id": value.get("request_id"),
        "command_index": value.get("command_index"),
        "apply_t": value.get("apply_t"),
        "action": value.get("action"),
        "nominal_position": value.get("nominal_position"),
        "nominal_velocity": value.get("nominal_velocity"),
    }


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = _shape(data, {"scenarios"})
    rows = data.get("scenarios") if root else None
    containers = isinstance(rows, list) and all(
        _shape(row, {"scenario", "trace", "requests", "cache_events", "segments"}) for row in rows
    )
    return [
        _atom("result_json", "parse", parsed, "results.json must be strict JSON"),
        _atom("result_json", "root", root, "root keys must be exactly scenarios"),
        _atom("result_json", "scenes", containers, "scene containers must have exact keys"),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = _scene_rows(data)
    scene_shape = len(rows) == 5 and all(isinstance(row, Mapping) for row in rows)
    trace_shape = request_shape = event_shape = segment_shape = values_ok = ordering = links = (
        finite
    ) = nonempty = True
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        trace, requests, events, segments = (
            row.get(key) for key in ("trace", "requests", "cache_events", "segments")
        )
        trace_shape &= isinstance(trace, list) and all(_shape(item, TRACE_KEYS) for item in trace)
        request_shape &= isinstance(requests, list) and all(
            _shape(item, REQUEST_KEYS) for item in requests
        )
        event_shape &= isinstance(events, list) and all(_shape(item, EVENT_KEYS) for item in events)
        segment_shape &= isinstance(segments, list) and all(
            _shape(item, SEGMENT_KEYS) for item in segments
        )
        ordering &= isinstance(trace, list) and [
            item.get("t") for item in trace if isinstance(item, Mapping)
        ] == list(range(HORIZON))
        ordering &= isinstance(segments, list) and [
            item.get("segment") for item in segments if isinstance(item, Mapping)
        ] == [0, 1, 2]
        ordering &= isinstance(requests, list) and [
            item.get("request_id") for item in requests if isinstance(item, Mapping)
        ] == [f"q{i}" for i in range(len(requests))]
        for sequence in (trace or [], requests or [], events or [], segments or []):
            if isinstance(sequence, Mapping):
                sequence = [sequence]
            finite &= all(_all_finite(item) for item in sequence if isinstance(item, Mapping))
        nonempty &= isinstance(trace, list) and isinstance(segments, list)
        for request in requests or []:
            if isinstance(request, Mapping):
                commands = request.get("commands")
                links &= (
                    isinstance(commands, list)
                    and request.get("depth") == len(commands)
                    and all(
                        isinstance(command, Mapping) and set(command) == COMMAND_KEYS
                        for command in commands
                    )
                )
        for event in events or []:
            if isinstance(event, Mapping):
                links &= event.get("event") in {"append", "consume", "clear", "safe_hold"}
    values_ok &= scene_shape and trace_shape and request_shape and event_shape and segment_shape
    return [
        _atom("result_evidence", "scene_shape", scene_shape, "five scene records are present"),
        _atom("result_evidence", "trace_shape", trace_shape, "trace keys and list shape"),
        _atom("result_evidence", "request_shape", request_shape, "request keys and list shape"),
        _atom("result_evidence", "event_shape", event_shape, "cache event keys and list shape"),
        _atom("result_evidence", "segment_shape", segment_shape, "segment keys and list shape"),
        _atom(
            "result_evidence", "field_values", values_ok, "declared field containers are present"
        ),
        _atom("result_evidence", "ordering", ordering, "canonical sequence order"),
        _atom("result_evidence", "links", links, "request commands and event links"),
        _atom("result_evidence", "finite_values", finite, "all numeric values are finite"),
        _atom(
            "result_evidence",
            "nonempty_evidence",
            nonempty,
            "trace and stage evidence are nonempty",
        ),
    ]


def _index_records(data: Any) -> dict[Hashable, Any]:
    result: dict[Hashable, Any] = {}
    for row in _scene_rows(data):
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            for key in ("trace", "requests", "cache_events", "segments"):
                result[(row["scenario"], key)] = row.get(key)
    return result


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _scene_rows(data)
    names = [row.get("scenario") for row in rows if isinstance(row, Mapping)]
    if names != list(SCENES):
        defects.append(f"scene order expected {list(SCENES)!r}, actual {names!r}")
    for name in SCENES:
        if names.count(name) != 1:
            defects.append(f"scene {name} count={names.count(name)}")
        row = next(
            (item for item in rows if isinstance(item, Mapping) and item.get("scenario") == name),
            None,
        )
        if row is None:
            continue
        for field, expected, key in (
            ("trace", set(range(HORIZON)), "t"),
            ("segments", {0, 1, 2}, "segment"),
        ):
            sequence = row.get(field)
            actual = (
                [item.get(key) for item in sequence if isinstance(item, Mapping)]
                if isinstance(sequence, list)
                else []
            )
            if set(actual) != expected or len(actual) != len(expected):
                defects.append(f"{name}.{field} keys={actual!r}")
        requests = row.get("requests")
        if isinstance(requests, list):
            ids = [item.get("request_id") for item in requests if isinstance(item, Mapping)]
            if ids != [f"q{i}" for i in range(len(ids))]:
                defects.append(f"{name}.requests IDs={ids!r}")
        events = row.get("cache_events")
        if isinstance(events, list) and any(
            not isinstance(item, Mapping)
            or not _integer(item.get("t"))
            or not 0 <= item["t"] < HORIZON
            for item in events
        ):
            defects.append(f"{name}.cache_events has invalid slot")
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _legality_atoms(records: Mapping[Hashable, Any]) -> list[AtomVerdict]:
    names = (
        "action_relations",
        "cache_validation",
        "request_depth",
        "request_spacing",
        "request_payload",
        "information_time",
    )
    flags = {name: True for name in names}
    details = {name: "constraint satisfied" for name in names}
    for scene in SCENES:
        trace = records.get((scene, "trace"), [])
        requests = records.get((scene, "requests"), [])
        sends: list[int] = []
        if not isinstance(trace, list):
            for key in names:
                flags[key] = False
                details[key] = f"{scene}.trace is not a list"
            continue
        for row in trace:
            if not isinstance(row, Mapping):
                flags["action_relations"] = False
                details["action_relations"] = f"{scene}: malformed trace row"
                continue
            t, decision, action, depth, request_id = (
                row.get("t"),
                row.get("decision"),
                row.get("action"),
                row.get("request_depth"),
                row.get("request_id"),
            )
            if (
                decision not in {"cache", "safe_hold"}
                or not _action_ok(action)
                or (decision == "safe_hold" and action != [0, 0])
            ):
                flags["action_relations"] = False
                details["action_relations"] = f"{scene}.t={t}: invalid decision/action"
            if decision == "cache" and not isinstance(row.get("cache_head"), Mapping):
                flags["cache_validation"] = False
                details["cache_validation"] = f"{scene}.t={t}: cache decision without head"
            if not (_integer(depth) and (depth == 0 or 2 <= depth <= 8)):
                flags["request_depth"] = False
                details["request_depth"] = f"{scene}.t={t}: depth={depth!r}"
            if depth == 0 and request_id is not None:
                flags["request_depth"] = False
                details["request_depth"] = f"{scene}.t={t}: ID with zero depth"
            if depth and _integer(t):
                sends.append(t)
                if request_id != f"q{len(sends) - 1}":
                    flags["request_payload"] = False
                    details["request_payload"] = f"{scene}.t={t}: request ID mismatch"
        if any(b - a < 2 for a, b in zip(sends, sends[1:])) or len(sends) > 18:
            flags["request_spacing"] = False
            details["request_spacing"] = f"{scene}: sends={sends!r}"
        if not isinstance(requests, list) or len(requests) != len(sends):
            flags["request_payload"] = False
            details["request_payload"] = (
                f"{scene}: trace sends={len(sends)}, "
                f"records={len(requests) if isinstance(requests, list) else 'invalid'}"
            )
        else:
            send_order: list[int] = []
            for request in requests:
                if not isinstance(request, Mapping):
                    flags["request_payload"] = False
                    continue
                depth, commands = request.get("depth"), request.get("commands")
                if not (
                    _integer(depth)
                    and 2 <= depth <= 8
                    and isinstance(commands, list)
                    and len(commands) == depth
                ):
                    flags["request_payload"] = False
                    details["request_payload"] = (
                        f"{scene}.{request.get('request_id')}: depth/commands mismatch"
                    )
                if request.get("base_position") is None or request.get("base_velocity") is None:
                    flags["information_time"] = False
                    details["information_time"] = (
                        f"{scene}.{request.get('request_id')}: missing base state"
                    )
                if _integer(request.get("send_t")):
                    send_order.append(request["send_t"])
                if send_order != sorted(send_order) or len(send_order) != len(set(send_order)):
                    flags["request_payload"] = False
                    details["request_payload"] = f"{scene}: request records are not in send order"
                if request.get("delay") != DELAYS[scene]:
                    flags["information_time"] = False
                    details["information_time"] = (
                        f"{scene}.{request.get('request_id')}: delay field mismatch"
                    )
                if (
                    not _integer(request.get("send_t"))
                    or not _integer(request.get("arrival_t"))
                    or request["arrival_t"] != request["send_t"] + DELAYS[scene]
                ):
                    flags["information_time"] = False
                    details["information_time"] = (
                        f"{scene}.{request.get('request_id')}: delay mismatch"
                    )
                status = request.get("status")
                arrival = request.get("arrival_t")
                if (
                    status not in {"delivered", "late"}
                    or not _integer(arrival)
                    or (status == "delivered") != (arrival <= 47)
                ):
                    flags["information_time"] = False
                    details["information_time"] = (
                        f"{scene}.{request.get('request_id')}: status/arrival mismatch"
                    )
    return [_atom("decision_legality", name, flags[name], details[name]) for name in names]


def _compare(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            errors.append(f"{path}: expected keys {sorted(expected)!r}, actual {sorted(actual)!r}")
            return
        for key in expected:
            _compare(actual.get(key), expected[key], f"{path}.{key}", errors)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            errors.append(f"{path}: expected length {len(expected)}, actual {len(actual)}")
            return
        for index, (got, want) in enumerate(zip(actual, expected)):
            _compare(got, want, f"{path}[{index}]", errors)
        return
    if _finite(actual) and _finite(expected):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=EPS):
            errors.append(f"{path}: expected {expected!r}, actual {actual!r}, tolerance={EPS}")
    elif actual != expected:
        errors.append(f"{path}: expected {expected!r}, actual {actual!r}")


def _check_nominal(scene: str, request: Mapping[str, Any], errors: list[str]) -> None:
    depth, commands = request.get("depth"), request.get("commands")
    base_p, base_v = request.get("base_position"), request.get("base_velocity")
    if not (
        _integer(depth)
        and isinstance(commands, list)
        and len(commands) == depth
        and isinstance(base_p, list)
        and isinstance(base_v, list)
        and len(base_p) == 2
        and len(base_v) == 2
        and all(_finite(x) for x in base_p + base_v)
    ):
        return
    delay, arrival = request.get("delay"), request.get("arrival_t")
    if not (_integer(delay) and _integer(arrival)):
        return
    position, velocity = (float(base_p[0]), float(base_p[1])), (float(base_v[0]), float(base_v[1]))
    for _ in range(delay):
        position, velocity, _ = _transition(scene, position, velocity, [0, 0], None, False)
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping) or set(command) != COMMAND_KEYS:
            errors.append(f"{scene}.{request.get('request_id')}.commands[{index}]: malformed")
            continue
        _compare(
            command.get("command_index"),
            index,
            f"{scene}.{request.get('request_id')}.commands[{index}].command_index",
            errors,
        )
        _compare(
            command.get("apply_t"),
            arrival + index,
            f"{scene}.{request.get('request_id')}.commands[{index}].apply_t",
            errors,
        )
        if not _action_ok(command.get("action")):
            errors.append(f"{scene}.{request.get('request_id')}.commands[{index}].action: invalid")
            continue
        _compare(
            command.get("nominal_position"),
            list(position),
            f"{scene}.{request.get('request_id')}.commands[{index}].nominal_position",
            errors,
        )
        _compare(
            command.get("nominal_velocity"),
            list(velocity),
            f"{scene}.{request.get('request_id')}.commands[{index}].nominal_velocity",
            errors,
        )
        position, velocity, _ = _transition(
            scene, position, velocity, command["action"], None, False
        )


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    trace = records.get((scene, "trace"), [])
    requests = records.get((scene, "requests"), [])
    events = records.get((scene, "cache_events"), [])
    segments = records.get((scene, "segments"), [])
    trace_rows = (
        {row.get("t"): row for row in trace if isinstance(row, Mapping)}
        if isinstance(trace, list)
        else {}
    )
    req_rows = (
        [row for row in requests if isinstance(row, Mapping)] if isinstance(requests, list) else []
    )
    errors: list[str] = []
    arrivals: dict[int, list[Mapping[str, Any]]] = {}
    for request in req_rows:
        arrival = request.get("arrival_t")
        if request.get("delay") != DELAYS[scene]:
            errors.append(
                f"{scene}.{request.get('request_id')}.delay: expected {DELAYS[scene]!r}, "
                f"actual {request.get('delay')!r}"
            )
        if (
            not _integer(request.get("send_t"))
            or not _integer(arrival)
            or arrival != request.get("send_t") + DELAYS[scene]
        ):
            errors.append(
                f"{scene}.{request.get('request_id')}.arrival_t: inconsistent with send_t and delay"
            )
        if (
            request.get("status") not in {"delivered", "late"}
            or not _integer(arrival)
            or (request.get("status") == "delivered") != (arrival <= 47)
        ):
            errors.append(
                f"{scene}.{request.get('request_id')}.status: inconsistent with arrival_t"
            )
        if request.get("status") == "delivered" and _integer(request.get("arrival_t")):
            arrivals.setdefault(request["arrival_t"], []).append(request)
    cache: list[dict[str, Any]] = []
    position, velocity = (0.0, 0.0), (0.0, 0.0)
    expected_events: list[dict[str, Any]] = []
    request_sends = {request.get("send_t"): request for request in req_rows}
    segment_errors, segment_violations, completed = [0.0] * 3, [False] * 3, [False] * 3
    safe_holds = mismatches = 0
    expected_trace: list[dict[str, Any]] = []
    for t in range(HORIZON):
        for request in sorted(arrivals.get(t, []), key=lambda item: str(item.get("request_id"))):
            commands = request.get("commands", [])
            if not isinstance(commands, list):
                errors.append(f"{scene}.{request.get('request_id')}.commands: not a list")
                continue
            for command in commands:
                if not isinstance(command, Mapping):
                    errors.append(f"{scene}.{request.get('request_id')}.command: malformed")
                    continue
                entry = dict(command)
                entry["request_id"] = request.get("request_id")
                cache.append(entry)
                expected_events.append(
                    {
                        "t": t,
                        "event": "append",
                        "request_id": request.get("request_id"),
                        "command_index": command.get("command_index"),
                        "cache_size": len(cache),
                        "reason": "returned",
                    }
                )
        row = trace_rows.get(t)
        target = _target(t)
        stage = _stage(t)
        head = _head(cache[0]) if cache else None
        cache_error = None
        if cache:
            hp, hv = cache[0].get("nominal_position"), cache[0].get("nominal_velocity")
            if (
                isinstance(hp, list)
                and isinstance(hv, list)
                and len(hp) == 2
                and len(hv) == 2
                and all(_finite(x) for x in hp + hv)
            ):
                cache_error = math.hypot(
                    position[0] - hp[0], position[1] - hp[1]
                ) + 0.4 * math.hypot(velocity[0] - hv[0], velocity[1] - hv[1])
            else:
                errors.append(f"{scene}.t={t}.cache_head: invalid nominal state")
        if not isinstance(row, Mapping):
            errors.append(f"{scene}.trace[{t}]: missing")
            continue
        for key, expected in (
            ("t", t),
            ("target", list(target)),
            ("position", list(position)),
            ("velocity", list(velocity)),
            ("segment", stage),
            ("cache_size", len(cache)),
            ("cache_head", head),
        ):
            _compare(row.get(key), expected, f"{scene}.trace[{t}].{key}", errors)
        if cache_error is None:
            _compare(row.get("cache_error"), None, f"{scene}.trace[{t}].cache_error", errors)
        elif not _finite(row.get("cache_error")) or not math.isclose(
            float(row.get("cache_error")), cache_error, rel_tol=0.0, abs_tol=EPS
        ):
            errors.append(
                f"{scene}.trace[{t}].cache_error: expected {cache_error}, "
                f"actual {row.get('cache_error')!r}"
            )
        accepts = bool(cache and cache_error is not None and cache_error <= ERROR_LIMIT + EPS)
        decision, action = row.get("decision"), row.get("action")
        if accepts:
            _compare(decision, "cache", f"{scene}.trace[{t}].decision", errors)
            _compare(action, cache[0].get("action"), f"{scene}.trace[{t}].action", errors)
            consumed = cache.pop(0)
            expected_events.append(
                {
                    "t": t,
                    "event": "consume",
                    "request_id": consumed.get("request_id"),
                    "command_index": consumed.get("command_index"),
                    "cache_size": len(cache),
                    "reason": "validated",
                }
            )
        else:
            if cache:
                cache.clear()
                mismatches += 1
                expected_events.append(
                    {
                        "t": t,
                        "event": "clear",
                        "request_id": None,
                        "command_index": None,
                        "cache_size": 0,
                        "reason": "mismatch",
                    }
                )
            safe_holds += 1
            expected_events.append(
                {
                    "t": t,
                    "event": "safe_hold",
                    "request_id": None,
                    "command_index": None,
                    "cache_size": 0,
                    "reason": "empty",
                }
            )
            _compare(decision, "safe_hold", f"{scene}.trace[{t}].decision", errors)
            _compare(action, [0, 0], f"{scene}.trace[{t}].action", errors)
            action = [0, 0]
        if not _action_ok(action):
            errors.append(f"{scene}.trace[{t}].action: invalid")
            action = [0, 0]
        post_position, post_velocity, safety = _transition(
            scene, position, velocity, action, t, True
        )
        _compare(
            row.get("post_position"),
            list(post_position),
            f"{scene}.trace[{t}].post_position",
            errors,
        )
        _compare(
            row.get("post_velocity"),
            list(post_velocity),
            f"{scene}.trace[{t}].post_velocity",
            errors,
        )
        _compare(row.get("safety_state"), safety, f"{scene}.trace[{t}].safety_state", errors)
        tracking_error = math.hypot(post_position[0] - target[0], post_position[1] - target[1])
        segment_errors[stage] = max(segment_errors[stage], tracking_error)
        segment_violations[stage] |= safety == "violation"
        segment_complete = (
            t == STAGES[stage][1] and tracking_error <= 0.9 + EPS and not segment_violations[stage]
        )
        if t == STAGES[stage][1]:
            completed[stage] = segment_complete
        _compare(
            row.get("segment_complete"),
            segment_complete,
            f"{scene}.trace[{t}].segment_complete",
            errors,
        )
        expected_trace.append(
            {
                "t": t,
                "position": position,
                "velocity": velocity,
                "action": action,
                "post_position": post_position,
                "post_velocity": post_velocity,
            }
        )
        depth, request_id = row.get("request_depth"), row.get("request_id")
        if _integer(depth) and depth > 0:
            request = request_sends.get(t)
            if request is None:
                errors.append(f"{scene}.t={t}: request_depth without request")
            else:
                if request.get("request_id") != request_id:
                    errors.append(f"{scene}.t={t}: request ID mismatch")
                base_errors: list[str] = []
                _compare(
                    request.get("base_position"),
                    list(position),
                    f"{scene}.t={t}.request.base_position",
                    base_errors,
                )
                _compare(
                    request.get("base_velocity"),
                    list(velocity),
                    f"{scene}.t={t}.request.base_velocity",
                    base_errors,
                )
                errors.extend(base_errors)
                _check_nominal(scene, request, errors)
        elif request_id is not None:
            errors.append(f"{scene}.t={t}: request ID with zero depth")
        position, velocity = post_position, post_velocity
    expected_segments = [
        {
            "segment": i,
            "start_t": start,
            "end_t": end,
            "target_end": list(end_point),
            "finish_t": end if completed[i] else None,
            "completed": completed[i],
            "max_error": segment_errors[i],
        }
        for i, (start, end, _, end_point) in enumerate(STAGES)
    ]
    segments_match = isinstance(segments, list) and len(segments) == len(expected_segments)
    if segments_match:
        for actual, expected in zip(segments, expected_segments):
            if not isinstance(actual, Mapping):
                segments_match = False
                break
            for key in SEGMENT_KEYS - {"max_error"}:
                if actual.get(key) != expected[key]:
                    segments_match = False
            if not _finite(actual.get("max_error")) or not math.isclose(
                float(actual.get("max_error")), expected["max_error"], rel_tol=0.0, abs_tol=EPS
            ):
                segments_match = False
    if not segments_match:
        errors.append(f"{scene}.segments: expected {expected_segments!r}, actual {segments!r}")
    if events != expected_events:
        errors.append(f"{scene}.cache_events: expected {expected_events!r}, actual {events!r}")
    request_ids = [request.get("request_id") for request in req_rows]
    send_times = [request.get("send_t") for request in req_rows]
    if request_ids != [f"q{i}" for i in range(len(req_rows))] or send_times != sorted(send_times):
        errors.append(f"{scene}.ordering: request IDs and send times are not ordered")
    return {
        "errors": errors,
        "completed": completed,
        "segment_errors": segment_errors,
        "safe_holds": safe_holds,
        "mismatches": mismatches,
        "requests": len(req_rows),
        "trace": expected_trace,
    }


def _replay_atoms(records: Mapping[Hashable, Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    replayed = {scene: _replay_scene(scene, records) for scene in SCENES}
    names = (
        "state",
        "nominal_fragments",
        "cache_events",
        "dynamics",
        "safety",
        "stage_records",
        "requests",
        "trace_evidence",
        "ordering",
    )
    flags = {name: True for name in names}
    details = {name: "replay matched" for name in names}
    for scene, result in replayed.items():
        for error in result["errors"]:
            if ".ordering" in error:
                target = "ordering"
            elif "cache_events" in error:
                target = "cache_events"
            elif "nominal" in error or "commands" in error:
                target = "nominal_fragments"
            elif ".post_" in error or ".action" in error:
                target = "dynamics"
            elif "safety_state" in error:
                target = "safety"
            elif "segments" in error or "segment_complete" in error:
                target = "stage_records"
            elif "request" in error or "delay" in error or "status" in error:
                target = "requests"
            elif ".trace[" in error or "cache_head" in error or "cache_error" in error:
                target = "trace_evidence"
            else:
                target = "state"
            flags[target] = False
            if details[target] == "replay matched":
                details[target] = error
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names], replayed


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    completion = sum(sum(item["completed"]) for item in replayed.values()) / 15.0
    tracking = 0.0
    for item in replayed.values():
        mse = (
            sum(
                (row["post_position"][0] - _target(t)[0]) ** 2
                + (row["post_position"][1] - _target(t)[1]) ** 2
                for t, row in enumerate(item["trace"])
            )
            / HORIZON
        )
        tracking += max(0.0, min(1.0, 1.0 - mse / 4.0))
    tracking /= len(SCENES)
    request_eff = sum(
        max(0.0, min(1.0, 1.0 - item["requests"] / 18.0)) for item in replayed.values()
    ) / len(SCENES)
    hold_cost = sum(
        max(0.0, min(1.0, 1.0 - (item["safe_holds"] + item["mismatches"]) / HORIZON))
        for item in replayed.values()
    ) / len(SCENES)
    return {
        "primary_metrics_1": completion,
        "primary_metrics_2": tracking,
        "auxiliary_metrics_1": request_eff,
        "auxiliary_metrics_2": hold_cost,
    }


def _gated_metrics(state: EvalState) -> dict[str, float] | None:
    required = (
        "official_run",
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    )
    if not state.official_ok or any(
        state.scores.get(name, 0.0) < 1.0 - EPS for name in required[1:]
    ):
        state.validity_gate = False
        return None
    state.validity_gate = True
    return _metrics(state.replayed)


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    lines, fence, headings = text.splitlines(), False, []
    for index, line in enumerate(lines):
        if line.strip().startswith(chr(96) * 3):
            fence = not fence
            continue
        if not fence:
            match = re.match(r"^ {0,3}#{1,6} +(.+?)\s*#*$", line)
            if match:
                headings.append((index, match.group(1).strip()))
    for offset, (index, title) in enumerate(headings):
        end = headings[offset + 1][0] if offset + 1 < len(headings) else len(lines)
        found.setdefault(title, "\n".join(lines[index + 1 : end]).strip())
    return found


def _report_score(ws: Workspace) -> tuple[float, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected regular file"
    try:
        found = _headings(ws.read_text("report.md") or "")
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md: read failed: {exc}"
    missing = sorted(name for name in required if not found.get(name))
    return max(0.0, (2.0 - 0.5 * len(missing)) / 2.0), f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    start = time.monotonic()
    state.run_started = True
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        run = await ws.run("python run.py", check=False, timeout=600.0)
        elapsed = time.monotonic() - start
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = elapsed
        stderr_text = str(getattr(run, "stderr", "")).lower()
        state.timed_out = (
            elapsed >= 600.0 - EPS
            or state.exit_code is None
            or "timeout" in stderr_text
            or "timed out" in stderr_text
        )
        state.result_file_is_new_regular = _regular_file(ws, "results.json")
        state.official_ok = (
            getattr(removed, "exit_code", None) == 0
            and state.exit_code == 0
            and not state.timed_out
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={elapsed:.3f}s, "
            f"new_results={state.result_file_is_new_regular}, timed_out={state.timed_out}"
        )
        if _regular_file(ws, "results.json"):
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            if not state.official_ok:
                state.scores["result_json"] = 0.0
                state.diagnostics["result_json"] = (
                    "official run failed; result_json score gated to zero"
                )
            for check_id in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = "official run or strict parse failed"
        else:
            state.records = _index_records(state.data)
            evidence = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
            state.diagnostics["result_evidence"] = _diagnose(evidence)
            coverage, defects = _coverage_analysis(state.data)
            state.scores["canonical_coverage"] = coverage
            state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:8]}"
            legality = _legality_atoms(state.records)
            state.scores["decision_legality"] = _score_atoms(
                legality, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose(legality)
            replay_atoms, state.replayed = _replay_atoms(state.records)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
            state.metrics = _gated_metrics(state)
    except Exception as exc:
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        state.official_ok = False
        state.run_finished = True
        state.elapsed_s = time.monotonic() - start
        state.timed_out = (
            isinstance(exc, TimeoutError)
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or state.elapsed_s >= 600.0 - EPS
        )
        state.official_message = detail
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, detail)
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, check_id: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(check_id, 0.0), state.diagnostics.get(check_id, "")


for _check_id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, _name: str = _check_id) -> tuple[float, str]:
        return await _score(ws, _name)

    globals()[_check_id] = check(id=_check_id)(_check)
del _check


for _metric_id in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, _name: str = _metric_id) -> tuple[float, str]:
        state = await _build_state(ws)
        return (state.metrics or {}).get(
            _name, 0.0
        ), "validity gate passed" if state.metrics else "validity gate failed; metric=0"

    globals()[_metric_id] = check(id=_metric_id)(_metric)
del _metric
