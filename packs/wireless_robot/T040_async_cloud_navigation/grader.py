# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
HORIZON = 100
DT = 0.25
WORKSPACE_MAX = 12.0
MAX_SPEED = 0.8
TARGET_SLEW = 0.05
SAFETY_CLEARANCE = 0.35
RISK_CLEARANCE = 0.60
LIDAR_RANGE = 2.5
COMMAND_MAX_AGE = 10
GOAL_RADIUS = 0.4
WAYPOINT_STEP = 0.4
WAYPOINT_COUNT = 5
N_RAYS = 16

SCENARIO_NAMES = (
    "static_chicane",
    "dynamic_crossing",
    "latency_spike",
    "narrow_doorway",
    "actuator_lag",
)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
GATE_CHECKS = (
    "official_run",
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.30,
        "result_json.scenario_shape": 0.50,
    },
    "result_evidence": {
        "result_evidence.unit_fields": 0.25,
        "result_evidence.command_fields": 0.75,
    },
    "decision_legality": {
        "decision_legality.packet_reference": 0.30,
        "decision_legality.null_braking": 0.20,
        "decision_legality.target_speed": 0.20,
        "decision_legality.target_slew": 0.30,
    },
    "replay_validity": {
        "replay_validity.packet_generation": 0.18,
        "replay_validity.lidar": 0.16,
        "replay_validity.dynamics": 0.22,
        "replay_validity.safety": 0.18,
        "replay_validity.arrival": 0.14,
        "replay_validity.terminal_metrics": 0.12,
    },
}


def _scenario(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    static: tuple[tuple[float, float, float], ...],
    dynamic: str | None,
    deadline: int,
    base_delay: int | str,
    loss: float,
) -> dict[str, Any]:
    return {
        "start": start,
        "goal": goal,
        "static": static,
        "dynamic": dynamic,
        "deadline": deadline,
        "base_delay": base_delay,
        "loss": loss,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "static_chicane": _scenario(
        (1.0, 1.0, math.pi / 4.0),
        (10.0, 10.0),
        ((4.0, 3.0, 0.8), (4.0, 7.0, 0.8), (8.0, 5.0, 0.8)),
        None,
        82,
        "alternating",
        0.08,
    ),
    "dynamic_crossing": _scenario(
        (1.0, 10.0, -math.pi / 4.0),
        (10.0, 1.0),
        ((5.0, 5.0, 0.7),),
        "horizontal7",
        86,
        1,
        0.10,
    ),
    "latency_spike": _scenario(
        (1.0, 6.0, 0.0),
        (10.0, 6.0),
        ((5.0, 3.0, 0.7), (5.0, 9.0, 0.7)),
        None,
        88,
        "spike",
        0.08,
    ),
    "narrow_doorway": _scenario(
        (1.0, 1.0, math.pi / 4.0),
        (10.0, 10.0),
        ((6.0, 4.0, 1.0), (6.0, 8.0, 1.0)),
        None,
        84,
        2,
        0.14,
    ),
    "actuator_lag": _scenario(
        (1.0, 10.0, -math.pi / 4.0),
        (10.0, 2.0),
        ((4.0, 6.0, 0.7), (8.0, 6.0, 0.7)),
        "vertical5",
        90,
        2,
        0.10,
    ),
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = bool(passed)
        self.detail = str(detail)
        self.share = float(share)


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
        self.parse_ok = False
        self.data: Any = None
        self.scenarios: dict[tuple[str, int], Any] = {}
        self.replayed: dict[tuple[str, int], Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _rotate(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    )


def _add(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    return first[0] + second[0], first[1] + second[1]


def _subtract(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    return first[0] - second[0], first[1] - second[1]


def _scale(vector: tuple[float, float], amount: float) -> tuple[float, float]:
    return vector[0] * amount, vector[1] * amount


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return _norm(_subtract(first, second))


def _clip_vec(vector: tuple[float, float], limit: float) -> tuple[float, float]:
    magnitude = _norm(vector)
    if magnitude <= limit or magnitude <= EPS:
        return vector
    return _scale(vector, limit / magnitude)


def _brake(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = _norm(vector)
    if magnitude <= TARGET_SLEW:
        return (0.0, 0.0)
    return _scale(vector, (magnitude - TARGET_SLEW) / magnitude)


def _vector_close(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return math.isclose(first[0], second[0], rel_tol=0.0, abs_tol=EPS) and math.isclose(first[1], second[1], rel_tol=0.0, abs_tol=EPS)


def _tri(period_half: int, value: int) -> int:
    return period_half - abs((value % (2 * period_half)) - period_half)


def _dynamic_circle(name: str, variant: int, boundary: int) -> tuple[float, float, float] | None:
    kind = SCENARIOS[name]["dynamic"]
    if kind == "horizontal7":
        return 2.0 + _tri(7, boundary + 3 * variant), 6.0, 0.30
    if kind == "vertical5":
        return 6.0, 2.0 + _tri(5, boundary + 3 * variant), 0.30
    return None


def _circles(name: str, variant: int, boundary: int) -> tuple[tuple[float, float, float], ...]:
    dynamic = _dynamic_circle(name, variant, boundary)
    static = tuple(SCENARIOS[name]["static"])
    return static if dynamic is None else static + (dynamic,)


def _workspace_clearance(point: tuple[float, float]) -> float:
    return min(point[0], WORKSPACE_MAX - point[0], point[1], WORKSPACE_MAX - point[1])


def _clearance(name: str, variant: int, boundary: int, point: tuple[float, float]) -> float:
    result = _workspace_clearance(point)
    for center_x, center_y, radius in _circles(name, variant, boundary):
        result = min(result, _distance(point, (center_x, center_y)) - radius)
    return result


def _ray_circle_distance(
    origin: tuple[float, float],
    direction: tuple[float, float],
    circle: tuple[float, float, float],
) -> float | None:
    center = (circle[0], circle[1])
    radius = circle[2]
    offset = _subtract(origin, center)
    if _norm(offset) <= radius + EPS:
        return 0.0
    projection = offset[0] * direction[0] + offset[1] * direction[1]
    constant = offset[0] ** 2 + offset[1] ** 2 - radius**2
    discriminant = projection**2 - constant
    if discriminant < -EPS:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [value for value in (-projection - root, -projection + root) if value >= -EPS]
    return max(0.0, min(candidates)) if candidates else None


def _ray_workspace_distance(origin: tuple[float, float], direction: tuple[float, float]) -> float:
    candidates: list[float] = []
    if direction[0] > EPS:
        candidates.append((WORKSPACE_MAX - origin[0]) / direction[0])
    elif direction[0] < -EPS:
        candidates.append((0.0 - origin[0]) / direction[0])
    if direction[1] > EPS:
        candidates.append((WORKSPACE_MAX - origin[1]) / direction[1])
    elif direction[1] < -EPS:
        candidates.append((0.0 - origin[1]) / direction[1])
    nonnegative = [value for value in candidates if value >= -EPS]
    return max(0.0, min(nonnegative)) if nonnegative else 0.0


def _lidar(name: str, variant: int, boundary: int, pose: tuple[float, float, float]) -> list[float]:
    origin = (pose[0], pose[1])
    values: list[float] = []
    for index in range(N_RAYS):
        angle = pose[2] + 2.0 * math.pi * index / N_RAYS
        direction = (math.cos(angle), math.sin(angle))
        distances = [_ray_workspace_distance(origin, direction)]
        for circle in _circles(name, variant, boundary):
            distance = _ray_circle_distance(origin, direction, circle)
            if distance is not None:
                distances.append(distance)
        values.append(_clip(min(distances), 0.0, LIDAR_RANGE))
    return values


def _generation_boundaries(variant: int) -> tuple[int, ...]:
    return tuple(range(1 + variant, HORIZON, 5))


def _packet_id(name: str, variant: int, generated: int) -> str:
    return f"packet={name}|variant={variant}|g={generated}"


def _base_delay(name: str, variant: int, generated: int) -> int:
    rule = SCENARIOS[name]["base_delay"]
    if rule == "alternating":
        return 1 + generated % 2
    if rule == "spike":
        return 4 if 36 + 3 * variant <= generated <= 48 + 3 * variant else 1
    return int(rule)


def _packet_delay(name: str, variant: int, generated: int) -> int:
    return _base_delay(name, variant, generated) + ((generated + variant) % 2)


def _packet_loss_sample(packet_id: str, variant: int) -> float:
    text = f"{packet_id}|loss_phase={4 * variant}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


def _best_direction_endpoint(
    options: Sequence[tuple[float, int, tuple[float, float]]],
) -> tuple[float, float]:
    best_alignment = max(option[0] for option in options)
    tied = [option for option in options if best_alignment - option[0] <= EPS]
    return min(tied, key=lambda option: option[1])[2]


def _planner_waypoints(
    name: str,
    variant: int,
    generated: int,
    anchor_position: tuple[float, float],
) -> list[tuple[float, float]]:
    goal = tuple(SCENARIOS[name]["goal"])
    previous = anchor_position
    waypoints: list[tuple[float, float]] = []
    for _ in range(WAYPOINT_COUNT):
        to_goal = _subtract(goal, previous)
        magnitude = _norm(to_goal)
        goal_direction = (0.0, 0.0) if magnitude <= EPS else _scale(to_goal, 1.0 / magnitude)
        options: list[tuple[float, int, tuple[float, float]]] = []
        for index in range(16):
            angle = 2.0 * math.pi * index / 16.0
            direction = (math.cos(angle), math.sin(angle))
            endpoint = _add(previous, _scale(direction, WAYPOINT_STEP))
            if _clearance(name, variant, generated, endpoint) + EPS < RISK_CLEARANCE:
                continue
            alignment = direction[0] * goal_direction[0] + direction[1] * goal_direction[1]
            options.append((alignment, index, endpoint))
        if options:
            previous = _best_direction_endpoint(options)
        waypoints.append(previous)
    return waypoints


def _generate_packet(
    name: str,
    variant: int,
    generated: int,
    anchor_pose: tuple[float, float, float],
) -> dict[str, Any]:
    packet_id = _packet_id(name, variant, generated)
    anchor_position = (anchor_pose[0], anchor_pose[1])
    world_waypoints = _planner_waypoints(name, variant, generated, anchor_position)
    ego_waypoints = [
        _rotate(_subtract(waypoint, anchor_position), -anchor_pose[2])
        for waypoint in world_waypoints
    ]
    sample = _packet_loss_sample(packet_id, variant)
    delay = _packet_delay(name, variant, generated)
    return {
        "packet_id": packet_id,
        "generated_boundary": generated,
        "arrival_boundary": generated + delay,
        "delay": delay,
        "loss_sample": sample,
        "lost": sample < float(SCENARIOS[name]["loss"]),
        "anchor_pose": anchor_pose,
        "ego_waypoints": ego_waypoints,
        "world_waypoints": world_waypoints,
    }


def _public_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "generated_boundary": packet["generated_boundary"],
        "arrival_boundary": packet["arrival_boundary"],
        "ego_waypoints": [tuple(waypoint) for waypoint in packet["ego_waypoints"]],
    }


def _world_polyline(packet: Mapping[str, Any], poses: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
    generated = int(packet["generated_boundary"])
    anchor = poses[generated]
    anchor_position = (anchor[0], anchor[1])
    return [anchor_position] + [
        _add(anchor_position, _rotate(tuple(waypoint), anchor[2]))
        for waypoint in packet["ego_waypoints"]
    ]


def _point_segment_distance(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    delta = _subtract(second, first)
    length_squared = delta[0] ** 2 + delta[1] ** 2
    if length_squared <= EPS**2:
        return _distance(point, first)
    offset = _subtract(point, first)
    fraction = _clip((offset[0] * delta[0] + offset[1] * delta[1]) / length_squared)
    return _distance(point, _add(first, _scale(delta, fraction)))


def _polyline_distance(point: tuple[float, float], polyline: Sequence[tuple[float, float]]) -> float:
    return min(
        _point_segment_distance(point, polyline[index], polyline[index + 1])
        for index in range(len(polyline) - 1)
    )


def _idle_command(slot: int) -> dict[str, Any]:
    return {"slot": slot, "packet_id": None, "u_x": 0.0, "u_y": 0.0}


def _command_issue(command: Any, path: str) -> str | None:
    expected = {"slot", "packet_id", "u_x", "u_y"}
    if not isinstance(command, dict):
        return f"{path}: expected object, actual={type(command).__name__}"
    if set(command) != expected:
        return f"{path}: expected exact fields={sorted(expected)!r}, actual={sorted(command)!r}"
    if not _integer(command.get("slot")):
        return f"{path}.slot: expected JSON integer, actual={command.get('slot')!r}"
    packet_id = command.get("packet_id")
    if packet_id is not None and (not isinstance(packet_id, str) or not packet_id):
        return f"{path}.packet_id: expected non-empty string or null, actual={packet_id!r}"
    for field in ("u_x", "u_y"):
        if not _finite(command.get(field)):
            return f"{path}.{field}: expected finite JSON number, actual={command.get(field)!r}"
    return None


def _unit_issue(unit: Any, path: str) -> str | None:
    expected = {"scenario", "variant", "commands"}
    if not isinstance(unit, dict):
        return f"{path}: expected object, actual={type(unit).__name__}"
    if set(unit) != expected:
        return f"{path}: expected exact fields={sorted(expected)!r}, actual={sorted(unit)!r}"
    if not isinstance(unit.get("scenario"), str):
        return f"{path}.scenario: expected string, actual={unit.get('scenario')!r}"
    if not _integer(unit.get("variant")):
        return f"{path}.variant: expected JSON integer, actual={unit.get('variant')!r}"
    return None


def _shape_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_ok = isinstance(data, dict) and set(data) == {"scenarios"}
    scenarios_ok = root_ok and isinstance(data.get("scenarios"), list)
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parsed" if parse_ok else "strict JSON parse failed"),
        _atom("result_json", "root", root_ok, f"root_keys={sorted(data) if isinstance(data, dict) else type(data).__name__}"),
        _atom("result_json", "scenario_shape", scenarios_ok, "scenarios is an array" if scenarios_ok else "scenarios is not an array"),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, dict) else None
    unit_issue = None
    if not isinstance(rows, list):
        unit_issue = f"scenarios: expected array, actual={type(rows).__name__}"
    else:
        for unit_index, unit in enumerate(rows):
            unit_issue = _unit_issue(unit, f"scenarios[{unit_index}]")
            if unit_issue is not None:
                break
    unit_ok = unit_issue is None
    command_issue = None
    if isinstance(rows, list):
        for unit_index, unit in enumerate(rows):
            if not isinstance(unit, dict):
                command_issue = (
                    f"scenarios[{unit_index}].commands: unavailable because unit is "
                    f"{type(unit).__name__}"
                )
                break
            commands = unit.get("commands")
            if not isinstance(commands, list):
                command_issue = (
                    f"scenarios[{unit_index}].commands: expected array, "
                    f"actual={type(commands).__name__}"
                )
                break
            for command_index, command in enumerate(commands):
                command_issue = _command_issue(
                    command, f"scenarios[{unit_index}].commands[{command_index}]"
                )
                if command_issue is not None:
                    break
            if command_issue is not None:
                break
    else:
        command_issue = f"commands unavailable because scenarios is {type(rows).__name__}"
    command_ok = command_issue is None
    return [
        _atom("result_evidence", "unit_fields", unit_ok, "unit records have exact fields" if unit_ok else str(unit_issue)),
        _atom("result_evidence", "command_fields", command_ok, "command records have exact finite fields" if command_ok else str(command_issue)),
    ]


def _index(data: Any) -> dict[tuple[str, int], Any]:
    indexed: dict[tuple[str, int], Any] = {}
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return indexed
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            hash(key)
        except TypeError:
            continue
        indexed.setdefault(key, row)
    return indexed


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return 0.0, ["scenarios is not an array"]
    counts: dict[Any, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            defects.append("non-object scenario row")
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            counts[key] = counts.get(key, 0) + 1
        except TypeError:
            defects.append(f"unhashable scenario key={key!r}")
    for key in SCENARIO_KEYS:
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"missing scenario={key[0]}|variant={key[1]}")
        elif count > 1:
            defects.extend([f"duplicate scenario={key!r}"] * (count - 1))
    defects.extend(f"unknown scenario key={key!r}" for key in counts if key not in SCENARIO_KEYS)
    expected_slots = set(range(HORIZON))
    for key in SCENARIO_KEYS:
        matching = [
            row
            for row in rows
            if isinstance(row, dict) and (row.get("scenario"), row.get("variant")) == key
        ]
        if len(matching) != 1:
            continue
        commands = matching[0].get("commands")
        if not isinstance(commands, list):
            defects.append(f"{key} commands is not an array")
            continue
        slot_counts: dict[Any, int] = {}
        for command in commands:
            if not isinstance(command, dict):
                defects.append(f"{key} non-object command")
                continue
            slot = command.get("slot")
            try:
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
            except TypeError:
                defects.append(f"{key} unhashable slot={slot!r}")
        for slot in expected_slots:
            count = slot_counts.get(slot, 0)
            if count == 0:
                defects.append(f"{key} missing slot={slot}")
            elif count > 1:
                defects.extend([f"{key} duplicate slot={slot}"] * (count - 1))
        defects.extend(f"{key} unknown slot={slot!r}" for slot in slot_counts if slot not in expected_slots)
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _commands_by_slot(row: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    commands = row.get("commands", ()) if isinstance(row, Mapping) else ()
    if not isinstance(commands, list):
        return result
    for command in commands:
        if not isinstance(command, Mapping):
            continue
        slot = command.get("slot")
        if _integer(slot):
            result.setdefault(slot, command)
    return result


def _replay(name: str, variant: int, row: Mapping[str, Any]) -> dict[str, Any]:
    scene = SCENARIOS[name]
    commands = _commands_by_slot(row)
    poses: list[tuple[float, float, float]] = [tuple(scene["start"])]
    actual_velocities: list[tuple[float, float]] = [(0.0, 0.0)]
    target_velocities: list[tuple[float, float]] = [(0.0, 0.0)]
    generated_packets: list[dict[str, Any]] = []
    arrivals: dict[int, list[dict[str, Any]]] = {}
    delivered: dict[str, dict[str, Any]] = {}
    lidar_records: list[list[float]] = []
    action_start_snapshots: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    motion_records: list[dict[str, Any]] = []
    selected_packet_ids: set[str] = set()
    command_ages: list[int] = []
    intent_errors: list[float] = []
    accelerations: list[float] = []
    safety_events = 0
    risk_steps = 0
    motion_energy = 0.0
    first_arrival: int | None = None
    legality: dict[str, list[str]] = {
        "packet_reference": [],
        "null_braking": [],
        "target_speed": [],
        "target_slew": [],
    }
    previous_decisions: list[dict[str, Any]] = []
    generation_set = set(_generation_boundaries(variant))

    for boundary in range(HORIZON + 1):
        current_pose = poses[boundary]
        if boundary in generation_set:
            packet = _generate_packet(name, variant, boundary, current_pose)
            generated_packets.append(packet)
            if not packet["lost"] and packet["arrival_boundary"] <= HORIZON:
                arrivals.setdefault(packet["arrival_boundary"], []).append(packet)

        for packet in sorted(
            arrivals.pop(boundary, []),
            key=lambda item: (item["arrival_boundary"], item["generated_boundary"], item["packet_id"]),
        ):
            delivered[packet["packet_id"]] = packet

        if boundary == HORIZON:
            break

        current_lidar = _lidar(name, variant, boundary, current_pose)
        lidar_records.append(current_lidar)
        snapshot = {
            "slot": boundary,
            "scenario": name,
            "variant": variant,
            "deadline": int(scene["deadline"]),
            "public_context": {
                "generation_phase": variant,
                "base_delay": scene["base_delay"],
                "loss_threshold": scene["loss"],
                "command_max_age": COMMAND_MAX_AGE,
                "actuator_lag": 0.65 + 0.08 * variant if name == "actuator_lag" else 0.65,
            },
            "poses": [tuple(pose) for pose in poses],
            "previous_actual_velocity": actual_velocities[boundary],
            "previous_target_velocity": target_velocities[boundary],
            "delivered_packets": [
                _public_packet(packet)
                for packet in sorted(
                    delivered.values(),
                    key=lambda item: (item["arrival_boundary"], item["generated_boundary"], item["packet_id"]),
                )
            ],
            "lidar": list(current_lidar),
            "previous_decisions": [dict(item) for item in previous_decisions],
            "first_arrival": first_arrival,
        }
        action_start_snapshots.append(snapshot)

        raw = commands.get(boundary, _idle_command(boundary))
        command = dict(raw) if _command_issue(raw, "command") is None else _idle_command(boundary)
        packet_id = command["packet_id"]
        ego_target = (float(command["u_x"]), float(command["u_y"]))
        submitted_world = _rotate(ego_target, current_pose[2])
        previous_target = target_velocities[boundary]
        expected_brake = _brake(previous_target)
        selected_packet: dict[str, Any] | None = None
        reference_ok = True
        if packet_id is not None:
            selected_packet = delivered.get(packet_id)
            if selected_packet is None:
                legality["packet_reference"].append(
                    f"{name}|{variant}|slot={boundary}: packet_id={packet_id!r} is not delivered at action start"
                )
                reference_ok = False
            else:
                age = boundary - int(selected_packet["arrival_boundary"])
                if age < 0 or age > COMMAND_MAX_AGE:
                    legality["packet_reference"].append(
                        f"{name}|{variant}|slot={boundary}: packet_id={packet_id!r}, age={age}, expected=0..{COMMAND_MAX_AGE}"
                    )
                    reference_ok = False
        elif not _vector_close(submitted_world, expected_brake):
            legality["null_braking"].append(
                f"{name}|{variant}|slot={boundary}: null packet requires world_target={expected_brake!r}, actual={submitted_world!r}, tolerance={EPS}"
            )

        if _norm(submitted_world) > MAX_SPEED + EPS:
            legality["target_speed"].append(
                f"{name}|{variant}|slot={boundary}: target_speed={_norm(submitted_world):.12g}, maximum={MAX_SPEED}, tolerance={EPS}"
            )
        slew = _distance(submitted_world, previous_target)
        if slew > TARGET_SLEW + EPS:
            legality["target_slew"].append(
                f"{name}|{variant}|slot={boundary}: world_target_slew={slew:.12g}, maximum={TARGET_SLEW}, tolerance={EPS}"
            )

        executed_target = submitted_world
        if packet_id is None or not reference_ok:
            executed_target = expected_brake
        executed_target = _clip_vec(executed_target, MAX_SPEED)
        delta = _subtract(executed_target, previous_target)
        if _norm(delta) > TARGET_SLEW:
            executed_target = _add(previous_target, _clip_vec(delta, TARGET_SLEW))

        lag = 0.65 + 0.08 * variant if name == "actuator_lag" else 0.65
        previous_actual = actual_velocities[boundary]
        lagged_velocity = _clip_vec(
            _add(_scale(previous_actual, lag), _scale(executed_target, 1.0 - lag)),
            MAX_SPEED,
        )
        position = (current_pose[0], current_pose[1])
        proposed = _add(position, _scale(lagged_velocity, DT))
        proposed_clearance = _clearance(name, variant, boundary + 1, proposed)
        unsafe = proposed_clearance + EPS < SAFETY_CLEARANCE
        if unsafe:
            next_position = position
            actual = (0.0, 0.0)
            next_heading = current_pose[2]
            safety_events += 1
        else:
            next_position = proposed
            actual = lagged_velocity
            next_heading = math.atan2(actual[1], actual[0]) if _norm(actual) > EPS else current_pose[2]
        next_clearance = _clearance(name, variant, boundary + 1, next_position)
        risky = next_clearance + EPS < RISK_CLEARANCE
        if risky:
            risk_steps += 1
        if first_arrival is None and _distance(next_position, tuple(scene["goal"])) <= GOAL_RADIUS + EPS:
            first_arrival = boundary + 1

        acceleration = _distance(actual, previous_actual) / DT
        energy = 0.5 * _norm(actual) ** 2
        accelerations.append(acceleration)
        motion_energy += energy
        if reference_ok and selected_packet is not None:
            polyline = _world_polyline(selected_packet, poses)
            intent_error = _polyline_distance(position, polyline)
            selected_packet_ids.add(selected_packet["packet_id"])
            command_ages.append(boundary - int(selected_packet["arrival_boundary"]))
        else:
            intent_error = MAX_SPEED
        intent_errors.append(intent_error)

        decision_record = {
            "slot": boundary,
            "packet_id": packet_id,
            "command_age": (
                boundary - int(selected_packet["arrival_boundary"])
                if reference_ok and selected_packet is not None
                else None
            ),
            "ego_target": ego_target,
            "submitted_world_target": submitted_world,
            "executed_world_target": executed_target,
            "reference_ok": reference_ok,
            "intent_error": intent_error,
        }
        decision_records.append(decision_record)
        previous_decisions.append(
            {
                "slot": boundary,
                "packet_id": packet_id,
                "u_x": ego_target[0],
                "u_y": ego_target[1],
            }
        )
        motion_records.append(
            {
                "slot": boundary,
                "before_pose": current_pose,
                "previous_actual": previous_actual,
                "world_target": executed_target,
                "lag": lag,
                "lagged_velocity": lagged_velocity,
                "proposed_position": proposed,
                "proposed_clearance": proposed_clearance,
                "unsafe": unsafe,
                "actual_velocity": actual,
                "after_pose": (next_position[0], next_position[1], next_heading),
                "next_clearance": next_clearance,
                "risky": risky,
                "acceleration": acceleration,
                "energy": energy,
            }
        )
        poses.append((next_position[0], next_position[1], next_heading))
        actual_velocities.append(actual)
        target_velocities.append(executed_target)

    delivered_before_terminal = {
        packet_id
        for packet_id, packet in delivered.items()
        if packet["arrival_boundary"] < HORIZON
    }
    return {
        "name": name,
        "variant": variant,
        "commands": {slot: dict(command) for slot, command in commands.items()},
        "poses": poses,
        "actual_velocities": actual_velocities,
        "target_velocities": target_velocities,
        "generated_packets": generated_packets,
        "delivered_packets": delivered,
        "delivered_before_terminal": delivered_before_terminal,
        "lidar_records": lidar_records,
        "action_start_snapshots": action_start_snapshots,
        "decision_records": decision_records,
        "motion_records": motion_records,
        "selected_packet_ids": selected_packet_ids,
        "command_ages": command_ages,
        "intent_errors": intent_errors,
        "accelerations": accelerations,
        "safety_events": safety_events,
        "risk_steps": risk_steps,
        "motion_energy": motion_energy,
        "first_arrival": first_arrival,
        "legality": legality,
    }


def _legality(records: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures = {name: [] for name in ATOM_SHARES["decision_legality"]}
    for key in SCENARIO_KEYS:
        if key not in records:
            for atom_id in failures:
                failures[atom_id].append(f"missing canonical unit {key}")
            continue
        try:
            state = _replay(key[0], key[1], records[key])
        except Exception as exc:
            for atom_id in failures:
                failures[atom_id].append(
                    f"{key}: legality replay raised {type(exc).__name__}: {exc}"
                )
            continue
        for name, details in state["legality"].items():
            failures[f"decision_legality.{name}"].extend(details)
    return [
        AtomVerdict(
            atom_id,
            not details,
            "passed" if not details else " | ".join(details[:10]),
            ATOM_SHARES["decision_legality"][atom_id],
        )
        for atom_id, details in failures.items()
    ]


def _audit_replay(name: str, variant: int, state: Mapping[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {
        "packet_generation": [],
        "lidar": [],
        "dynamics": [],
        "safety": [],
        "arrival": [],
        "terminal_metrics": [],
    }
    poses = state.get("poses")
    if not isinstance(poses, list) or len(poses) != HORIZON + 1:
        failures["dynamics"].append(
            f"{name}|{variant}: expected {HORIZON + 1} pose boundaries"
        )
        return failures
    generated = state.get("generated_packets")
    expected_boundaries = _generation_boundaries(variant)
    if not isinstance(generated, list) or len(generated) != len(expected_boundaries):
        failures["packet_generation"].append(
            f"{name}|{variant}: expected generation boundaries={expected_boundaries!r}"
        )
    else:
        for expected_boundary, actual_packet in zip(expected_boundaries, generated):
            expected = _generate_packet(name, variant, expected_boundary, tuple(poses[expected_boundary]))
            for field in (
                "packet_id",
                "generated_boundary",
                "arrival_boundary",
                "delay",
                "lost",
            ):
                if actual_packet.get(field) != expected[field]:
                    failures["packet_generation"].append(
                        f"{name}|{variant}: packet {expected_boundary} field={field}, expected={expected[field]!r}, actual={actual_packet.get(field)!r}"
                    )
            if not math.isclose(
                float(actual_packet.get("loss_sample", -1.0)),
                expected["loss_sample"],
                rel_tol=0.0,
                abs_tol=EPS,
            ):
                failures["packet_generation"].append(
                    f"{name}|{variant}: packet {expected_boundary} loss sample mismatch"
                )
            actual_waypoints = actual_packet.get("ego_waypoints")
            if not isinstance(actual_waypoints, list) or len(actual_waypoints) != WAYPOINT_COUNT or any(
                not _vector_close(tuple(actual), tuple(wanted))
                for actual, wanted in zip(actual_waypoints, expected["ego_waypoints"])
            ):
                failures["packet_generation"].append(
                    f"{name}|{variant}: packet {expected_boundary} ego waypoints mismatch"
                )

    lidar_records = state.get("lidar_records")
    if not isinstance(lidar_records, list) or len(lidar_records) != HORIZON:
        failures["lidar"].append(f"{name}|{variant}: expected {HORIZON} LiDAR records")
    else:
        for slot, actual_values in enumerate(lidar_records):
            expected_values = _lidar(name, variant, slot, tuple(poses[slot]))
            if not isinstance(actual_values, list) or len(actual_values) != N_RAYS or any(
                not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=EPS)
                for actual, expected in zip(actual_values, expected_values)
            ):
                failures["lidar"].append(
                    f"{name}|{variant}: LiDAR mismatch at slot={slot}"
                )
                break

    motion_records = state.get("motion_records")
    actual_velocities = state.get("actual_velocities")
    target_velocities = state.get("target_velocities")
    decision_records = state.get("decision_records")
    snapshots = state.get("action_start_snapshots")
    raw_commands = state.get("commands")
    if (
        not isinstance(motion_records, list)
        or len(motion_records) != HORIZON
        or not isinstance(actual_velocities, list)
        or len(actual_velocities) != HORIZON + 1
        or not isinstance(target_velocities, list)
        or len(target_velocities) != HORIZON + 1
    ):
        failures["dynamics"].append(f"{name}|{variant}: motion history lengths mismatch")
    else:
        for slot, record in enumerate(motion_records):
            before = tuple(poses[slot])
            previous_actual = tuple(actual_velocities[slot])
            target = tuple(target_velocities[slot + 1])
            lag = 0.65 + 0.08 * variant if name == "actuator_lag" else 0.65
            expected_lagged = _clip_vec(
                _add(_scale(previous_actual, lag), _scale(target, 1.0 - lag)),
                MAX_SPEED,
            )
            expected_proposed = _add((before[0], before[1]), _scale(expected_lagged, DT))
            if (
                not _vector_close(tuple(record.get("lagged_velocity", (999.0, 999.0))), expected_lagged)
                or not _vector_close(tuple(record.get("proposed_position", (999.0, 999.0))), expected_proposed)
                or not _vector_close(tuple(record.get("previous_actual", (999.0, 999.0))), previous_actual)
                or not _vector_close(tuple(record.get("world_target", (999.0, 999.0))), target)
            ):
                failures["dynamics"].append(
                    f"{name}|{variant}: lagged transition mismatch at slot={slot}"
                )
                break
            if not _vector_close(tuple(record.get("actual_velocity", (999.0, 999.0))), tuple(actual_velocities[slot + 1])) or not all(
                math.isclose(float(record.get("after_pose", (999.0, 999.0, 999.0))[index]), float(poses[slot + 1][index]), rel_tol=0.0, abs_tol=EPS)
                for index in range(3)
                ):
                failures["dynamics"].append(
                    f"{name}|{variant}: stored motion state mismatch at slot={slot}"
                )
                break

            expected_acceleration = _distance(
                tuple(actual_velocities[slot + 1]), previous_actual
            ) / DT
            expected_energy = 0.5 * _norm(tuple(actual_velocities[slot + 1])) ** 2
            if not math.isclose(
                float(record.get("acceleration", -1.0)),
                expected_acceleration,
                rel_tol=0.0,
                abs_tol=EPS,
            ) or not math.isclose(
                float(record.get("energy", -1.0)),
                expected_energy,
                rel_tol=0.0,
                abs_tol=EPS,
            ):
                failures["terminal_metrics"].append(
                    f"{name}|{variant}: acceleration/energy mismatch at slot={slot}"
                )

            if isinstance(raw_commands, Mapping) and isinstance(decision_records, list) and slot < len(decision_records):
                raw = raw_commands.get(slot, _idle_command(slot))
                decision = decision_records[slot]
                if _command_issue(raw, f"{name}|{variant}|slot={slot}") is None:
                    submitted = _rotate(
                        (float(raw["u_x"]), float(raw["u_y"])), before[2]
                    )
                    if not _vector_close(
                        tuple(decision.get("submitted_world_target", (999.0, 999.0))),
                        submitted,
                    ):
                        failures["dynamics"].append(
                            f"{name}|{variant}: submitted target mismatch at slot={slot}"
                        )
                    expected_target = submitted
                    packet_id = raw.get("packet_id")
                    if packet_id is None or not bool(decision.get("reference_ok")):
                        expected_target = _brake(tuple(target_velocities[slot]))
                    expected_target = _clip_vec(expected_target, MAX_SPEED)
                    target_delta = _subtract(expected_target, tuple(target_velocities[slot]))
                    if _norm(target_delta) > TARGET_SLEW:
                        expected_target = _add(
                            tuple(target_velocities[slot]),
                            _clip_vec(target_delta, TARGET_SLEW),
                        )
                    if not _vector_close(tuple(decision.get("executed_world_target", (999.0, 999.0))), expected_target):
                        failures["dynamics"].append(
                            f"{name}|{variant}: executed target mismatch at slot={slot}"
                        )

    counted_safety = 0
    counted_risk = 0
    if isinstance(motion_records, list):
        for slot, record in enumerate(motion_records):
            proposed = tuple(record.get("proposed_position", (999.0, 999.0)))
            expected_unsafe = _clearance(name, variant, slot + 1, proposed) + EPS < SAFETY_CLEARANCE
            after_pose = tuple(record.get("after_pose", (999.0, 999.0, 0.0)))
            expected_risky = _clearance(name, variant, slot + 1, (after_pose[0], after_pose[1])) + EPS < RISK_CLEARANCE
            if bool(record.get("unsafe")) != expected_unsafe:
                failures["safety"].append(
                    f"{name}|{variant}: safety filter mismatch at slot={slot}"
                )
            if bool(record.get("risky")) != expected_risky:
                failures["safety"].append(
                    f"{name}|{variant}: risk state mismatch at slot={slot}"
                )
            counted_safety += int(expected_unsafe)
            counted_risk += int(expected_risky)
    if counted_safety != state.get("safety_events") or counted_risk != state.get("risk_steps"):
        failures["safety"].append(
            f"{name}|{variant}: safety counters mismatch, expected=({counted_safety},{counted_risk}), actual=({state.get('safety_events')},{state.get('risk_steps')})"
        )

    expected_arrival = next(
        (
            boundary
            for boundary, pose in enumerate(poses)
            if _distance((pose[0], pose[1]), tuple(SCENARIOS[name]["goal"])) <= GOAL_RADIUS + EPS
        ),
        None,
    )
    if state.get("first_arrival") != expected_arrival:
        failures["arrival"].append(
            f"{name}|{variant}: arrival mismatch, expected={expected_arrival}, actual={state.get('first_arrival')}"
        )

    accelerations = state.get("accelerations")
    intent_errors = state.get("intent_errors")
    command_ages = state.get("command_ages")
    if not isinstance(accelerations, list) or len(accelerations) != HORIZON or any(
        not _finite(value) or value < 0 for value in accelerations
    ):
        failures["terminal_metrics"].append(f"{name}|{variant}: invalid acceleration evidence")
    if not isinstance(intent_errors, list) or len(intent_errors) != HORIZON or any(
        not _finite(value) or value < 0 for value in intent_errors
    ):
        failures["terminal_metrics"].append(f"{name}|{variant}: invalid intent-error evidence")
    if not isinstance(command_ages, list) or any(
        not _integer(value) or value < 0 or value > COMMAND_MAX_AGE for value in command_ages
    ):
        failures["terminal_metrics"].append(f"{name}|{variant}: invalid command-age evidence")
    if not _finite(state.get("motion_energy")) or state["motion_energy"] < 0:
        failures["terminal_metrics"].append(f"{name}|{variant}: invalid motion energy")

    if isinstance(motion_records, list) and len(motion_records) == HORIZON:
        expected_accelerations = [
            _distance(tuple(actual_velocities[slot + 1]), tuple(actual_velocities[slot])) / DT
            for slot in range(HORIZON)
        ]
        expected_energy = sum(
            0.5 * _norm(tuple(actual_velocities[slot + 1])) ** 2
            for slot in range(HORIZON)
        )
        if isinstance(accelerations, list) and len(accelerations) == HORIZON and any(
            not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=EPS)
            for actual, expected in zip(accelerations, expected_accelerations)
        ):
            failures["terminal_metrics"].append(
                f"{name}|{variant}: acceleration series is not recomputed from velocity transitions"
            )
        if not math.isclose(float(state.get("motion_energy", -1.0)), expected_energy, rel_tol=0.0, abs_tol=EPS):
            failures["terminal_metrics"].append(
                f"{name}|{variant}: motion energy is not recomputed from actual velocities"
            )

    expected_intent_errors: list[float] = []
    expected_command_ages: list[int] = []
    expected_selected: set[str] = set()
    if not isinstance(decision_records, list) or len(decision_records) != HORIZON:
        failures["terminal_metrics"].append(
            f"{name}|{variant}: decision evidence length is not {HORIZON}"
        )
    elif not isinstance(snapshots, list) or len(snapshots) != HORIZON:
        failures["terminal_metrics"].append(
            f"{name}|{variant}: action-start snapshot length is not {HORIZON}"
        )
    else:
        for slot, decision in enumerate(decision_records):
            packet_id = decision.get("packet_id")
            selected = None
            if packet_id is not None:
                selected = next(
                    (
                        packet
                        for packet in snapshots[slot].get("delivered_packets", ())
                        if packet.get("packet_id") == packet_id
                    ),
                    None,
                )
            if selected is None:
                expected_intent_errors.append(MAX_SPEED)
                expected_age = None
            else:
                expected_age = slot - int(selected["arrival_boundary"])
                expected_intent_errors.append(
                    _polyline_distance(
                        (poses[slot][0], poses[slot][1]),
                        _world_polyline(selected, snapshots[slot]["poses"]),
                    )
                )
                expected_command_ages.append(expected_age)
                expected_selected.add(str(packet_id))
            if decision.get("command_age") != expected_age:
                failures["terminal_metrics"].append(
                    f"{name}|{variant}: command age mismatch at slot={slot}"
                )
            if not math.isclose(
                float(decision.get("intent_error", -1.0)),
                expected_intent_errors[-1],
                rel_tol=0.0,
                abs_tol=EPS,
            ):
                failures["terminal_metrics"].append(
                    f"{name}|{variant}: intent error mismatch at slot={slot}"
                )
        if isinstance(intent_errors, list) and len(intent_errors) == HORIZON and any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=EPS)
            for actual, expected in zip(intent_errors, expected_intent_errors)
        ):
            failures["terminal_metrics"].append(
                f"{name}|{variant}: intent-error series is not recomputed from selected packet geometry"
            )
        if command_ages != expected_command_ages:
            failures["terminal_metrics"].append(
                f"{name}|{variant}: command-age series mismatch"
            )
        if state.get("selected_packet_ids") != expected_selected:
            failures["terminal_metrics"].append(
                f"{name}|{variant}: selected packet set mismatch"
            )
    return failures


def _replay_atoms(
    records: Mapping[tuple[str, int], Any],
    replayed: Mapping[tuple[str, int], Any],
) -> list[AtomVerdict]:
    failures = {name: [] for name in ATOM_SHARES["replay_validity"]}
    for key in SCENARIO_KEYS:
        if key not in records or key not in replayed:
            for atom_id in failures:
                failures[atom_id].append(f"missing replay for {key}")
            continue
        try:
            unit_failures = _audit_replay(key[0], key[1], replayed[key])
        except Exception as exc:
            for atom_id in failures:
                failures[atom_id].append(
                    f"{key}: replay audit raised {type(exc).__name__}: {exc}"
                )
            continue
        for name, details in unit_failures.items():
            failures[f"replay_validity.{name}"].extend(details)
    return [
        AtomVerdict(
            atom_id,
            not details,
            "passed" if not details else " | ".join(details[:10]),
            ATOM_SHARES["replay_validity"][atom_id],
        )
        for atom_id, details in failures.items()
    ]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name, variant in SCENARIO_KEYS:
        state = replayed[(name, variant)]
        arrival = state["first_arrival"]
        deadline = int(SCENARIOS[name]["deadline"])
        if arrival is None:
            arrival_quality = 0.0
        elif arrival <= deadline:
            arrival_quality = 1.0
        else:
            arrival_quality = 0.5 * _clip(
                1.0 - (arrival - deadline) / max(1, HORIZON - deadline)
            )
        totals["primary_metrics_1"] += arrival_quality
        totals["primary_metrics_2"] += 0.7 * _clip(
            1.0 - state["risk_steps"] / HORIZON
        ) + 0.3 * _clip(1.0 - state["safety_events"] / HORIZON)
        mean_error = sum(state["intent_errors"]) / HORIZON
        totals["primary_metrics_3"] += _clip(1.0 - mean_error / MAX_SPEED)

        delivered_count = len(state["delivered_before_terminal"])
        useful_fraction = (
            len(state["selected_packet_ids"]) / delivered_count
            if delivered_count
            else 0.0
        )
        freshness = (
            sum(_clip(1.0 - age / COMMAND_MAX_AGE) for age in state["command_ages"])
            / len(state["command_ages"])
            if state["command_ages"]
            else 0.0
        )
        totals["auxiliary_metrics_1"] += _clip(0.6 * useful_fraction + 0.4 * freshness)

        valid_fraction = len(state["command_ages"]) / HORIZON
        mean_acceleration = sum(state["accelerations"]) / HORIZON
        resource_quality = 0.5 * _clip(1.0 - mean_acceleration / 0.20) + 0.5 * _clip(
            1.0 - state["motion_energy"] / 32.0
        )
        totals["auxiliary_metrics_2"] += _clip(valid_fraction * resource_quality)
    return {
        key: _clip(value / len(SCENARIO_KEYS))
        for key, value in totals.items()
    }


def _gated_metrics(
    replayed: Mapping[tuple[str, int], Any], validity_gate: bool
) -> dict[str, float]:
    return _metrics(replayed) if validity_gate else {key: 0.0 for key in METRIC_KEYS}


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared fixed set")
    if any(
        not math.isfinite(value) or value <= 0 for value in expected.values()
    ) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid fixed atom shares")
    if any(
        not math.isclose(
            atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12
        )
        for atom in atoms
    ):
        raise ValueError("atom share differs from declaration")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    return "failed_atoms=[]" if not failed else "failed_atoms=[{}]; {}".format(
        ",".join(atom.id for atom in failed),
        " | ".join(f"{atom.id}: {atom.detail}" for atom in failed),
    )


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular file"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for index, line in enumerate(lines):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_char, fence_len = True, marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
            continue
        if in_fence:
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$", line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))
    found: set[str] = set()
    for index, (line_no, level, title) in enumerate(headings):
        if title not in required:
            continue
        end = len(lines)
        for later_line, later_level, _ in headings[index + 1 :]:
            if later_level <= level:
                end = later_line
                break
        if any(line.strip() for line in lines[line_no + 1 : end]):
            found.add(title)
    normalized = max(0.0, 2.0 - 0.5 * (len(required) - len(found))) / 2.0
    return normalized, f"missing_sections={sorted(required - found)}; normalized={normalized:.9g}"


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
        remove_ok = getattr(removed, "exit_code", None) == 0
        state.run_started = True
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600)
            state.exit_code = getattr(result, "exit_code", None)
            output = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".lower()
            state.timed_out = state.exit_code is None and "timeout" in output
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
        finally:
            state.elapsed_s = time.monotonic() - started
            state.run_finished = True
        result_path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = bool(
            remove_ok and result_path.is_file() and not result_path.is_symlink()
        )
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s is not None
            and state.elapsed_s <= 600 + EPS
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"{state.official_message}; " if state.official_message else ""
        ) + f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        state.scores["official_run"] = 1.0 if state.official_ok else 0.0
        parse_failure = ""
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                parse_failure = f"strict parse failed: {type(exc).__name__}: {exc}"
        shape = _shape_atoms(state.data, state.parse_ok)
        evidence = _evidence_atoms(state.data)
        state.scenarios = _index(state.data)
        coverage_score, coverage_defects = _coverage(state.data)
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = [
                _atom(
                    "decision_legality",
                    name,
                    False,
                    f"legality replay raised {type(exc).__name__}: {exc}",
                )
                for name in (
                    "packet_reference",
                    "null_braking",
                    "target_speed",
                    "target_slew",
                )
            ]
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(
            evidence, ATOM_SHARES["result_evidence"]
        )
        state.scores["canonical_coverage"] = coverage_score
        state.scores["decision_legality"] = _score_atoms(
            legality, ATOM_SHARES["decision_legality"]
        )
        can_replay = all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS
            for check_id in (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
            )
        )
        if can_replay:
            state.replayed = {
                key_: _replay(key_[0], key_[1], state.scenarios[key_])
                for key_ in SCENARIO_KEYS
            }
        replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        state.scores["replay_validity"] = _score_atoms(
            replay_atoms, ATOM_SHARES["replay_validity"]
        )
        state.diagnostics["result_json"] = (
            f"{parse_failure}; " if parse_failure else ""
        ) + _diagnose(shape)
        state.diagnostics["result_evidence"] = (
            f"{parse_failure}; affected gate=result_evidence; "
            if parse_failure
            else ""
        ) + _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = (
            (
                f"{parse_failure}; affected gate=canonical_coverage; "
                if parse_failure
                else ""
            )
            + f"defect_count={len(coverage_defects)}; "
            + " | ".join(coverage_defects[:12])
            if coverage_defects
            else (
                f"{parse_failure}; affected gate=canonical_coverage; defect_count=0"
                if parse_failure
                else "defect_count=0"
            )
        )
        state.diagnostics["decision_legality"] = (
            f"{parse_failure}; affected gate=decision_legality; "
            if parse_failure
            else ""
        ) + _diagnose(legality)
        state.diagnostics["replay_validity"] = (
            f"{parse_failure}; affected gate=replay_validity; "
            if parse_failure
            else ""
        ) + _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in GATE_CHECKS
        )
        state.metrics = _gated_metrics(state.replayed, state.validity_gate)
        if not state.official_ok:
            for check_id in GATE_CHECKS:
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = (
                    f"official_run failed: {state.official_message}; "
                    f"{state.diagnostics.get(check_id, '')}"
                )
    except Exception as exc:
        state.run_finished = state.run_finished or state.run_started
        state.official_message = f"grader state failed: {type(exc).__name__}: {exc}"
        for check_id in GATE_CHECKS:
            state.scores[check_id] = 0.0
            state.diagnostics[check_id] = state.official_message
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink():
        return 0.0, "run.py is not a regular file"
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    except Exception as exc:
        return 0.0, f"static compilation raised {type(exc).__name__}: {exc}"
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (
        (1.0, state.official_message)
        if state.official_ok
        else (0.0, state.official_message)
    )


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


for _id in GATE_CHECKS:
    async def _check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    if _id != "official_run":
        globals()[_id] = check(id=_id)(_check)


for _metric in METRIC_KEYS:
    async def _metric_check(ws: Workspace, ident: str = _metric) -> tuple[float, str]:
        state = await _build_state(ws)
        value = (
            state.metrics.get(ident, 0.0)
            if state.validity_gate and state.metrics
            else 0.0
        )
        return value, f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}"

    globals()[_metric] = check(id=_metric)(_metric_check)


del _check, _metric_check
