from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

SCENES = (
    "open_yard",
    "shielded_turns",
    "interference_corridor",
    "shared_bottleneck",
    "long_return",
)
ROBOTS = tuple(range(3))
CHECKPOINTS = tuple(range(6))
HORIZON = 48
TERMINAL_BOUNDARY = 48
ACCESS_POINT = (10.0, 10.0)
ROUTE_SPEED = 1.5
INITIAL_ENERGY = 20.0
POWER_LEVELS = (2, 6, 10, 14, 18, 22)
MAX_CONCURRENT = 2
MAX_STREAK = 3
MAX_ADJACENT_CHANGE = 6
RSSI_THRESHOLD = -58.0
EPS = 1e-6

VALUES = (
    (4, 7, 5, 9, 6, 10),
    (6, 5, 8, 6, 10, 7),
    (5, 8, 6, 10, 7, 9),
)

SCENE_DATA: dict[str, dict[str, Any]] = {
    "open_yard": {
        "routes": (
            ((8, 8), (6, 8), (4, 8), (4, 12), (6, 14), (9, 13)),
            ((12, 8), (14, 8), (16, 8), (16, 12), (14, 14), (11, 13)),
            ((10, 12), (10, 15), (7, 16), (5, 14), (7, 11), (10, 9)),
        ),
        "windows": (2, 2, 2, 2, 2, 2),
        "base_interference": 0.0,
        "obstacles": (),
    },
    "shielded_turns": {
        "routes": (
            ((7, 7), (5, 7), (3, 9), (3, 13), (6, 16), (9, 15)),
            ((13, 7), (15, 7), (17, 9), (17, 13), (14, 16), (11, 15)),
            ((10, 13), (10, 16), (7, 18), (4, 16), (5, 12), (8, 10)),
        ),
        "windows": (2, 3, 2, 3, 2, 4),
        "base_interference": 1.0,
        "obstacles": (
            (2.0, 7.5, 8.5, 14.5, 4.5),
            (12.5, 18.0, 8.5, 14.5, 4.5),
            (6.0, 14.0, 15.0, 18.0, 3.0),
        ),
    },
    "interference_corridor": {
        "routes": (
            ((4, 8), (6, 8), (8, 8), (10, 8), (12, 8), (14, 8)),
            ((4, 10), (6, 10), (8, 10), (10, 10), (12, 10), (14, 10)),
            ((4, 12), (6, 12), (8, 12), (10, 12), (12, 12), (14, 12)),
        ),
        "windows": (2, 2, 3, 2, 3, 3),
        "base_interference": 3.0,
        "obstacles": ((7.0, 13.0, 7.0, 13.0, 2.0),),
    },
    "shared_bottleneck": {
        "routes": (
            ((3, 7), (6, 8), (9, 9), (12, 10), (15, 11), (17, 13)),
            ((3, 10), (6, 10), (9, 10), (12, 10), (15, 10), (18, 10)),
            ((3, 13), (6, 12), (9, 11), (12, 10), (15, 9), (17, 7)),
        ),
        "windows": (2, 2, 2, 3, 2, 3),
        "base_interference": 1.5,
        "obstacles": (
            (7.5, 9.5, 5.5, 14.5, 3.5),
            (10.5, 12.5, 5.5, 14.5, 3.5),
        ),
    },
    "long_return": {
        "routes": (
            ((10, 8), (14, 6), (18, 4), (18, 10), (16, 16), (10, 18)),
            ((8, 10), (5, 6), (2, 3), (2, 10), (4, 17), (10, 19)),
            ((12, 10), (15, 7), (18, 7), (19, 13), (15, 18), (9, 17)),
        ),
        "windows": (3, 3, 4, 3, 4, 4),
        "base_interference": 1.5,
        "obstacles": (
            (0.0, 6.0, 0.0, 8.0, 5.0),
            (14.0, 20.0, 0.0, 8.0, 5.0),
            (0.0, 6.0, 14.0, 20.0, 6.0),
            (14.0, 20.0, 14.0, 20.0, 6.0),
        ),
    },
}

SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "send", "power_dbm"}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.25,
        "result_evidence.action_containers": 0.25,
        "result_evidence.send_fields": 0.25,
        "result_evidence.power_fields": 0.25,
    },
    "decision_legality": {
        "decision_legality.power_selection": 1 / 6,
        "decision_legality.concurrent_capacity": 1 / 6,
        "decision_legality.report_availability": 1 / 6,
        "decision_legality.energy_budget": 1 / 6,
        "decision_legality.attempt_streak": 1 / 6,
        "decision_legality.power_slew": 1 / 6,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.baseline_observations": 1 / 6,
        "replay_validity.interference_outcomes": 1 / 6,
        "replay_validity.arrival_confirmation": 1 / 6,
        "replay_validity.route_progression": 1 / 6,
        "replay_validity.energy_accounting": 1 / 6,
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
        self.parse_ok = False
        self.data: Any = None
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _safe_repr(value: Any, limit: int = 160) -> str:
    try:
        rendered = repr(value)
    except (OverflowError, ValueError):
        if _integer(value):
            return f"<integer bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _record_shape(value: Any, keys: set[str], canonical_key: str) -> bool:
    # Canonical identity fields are required parts of each public container;
    # coverage handles their uniqueness/order separately.
    del canonical_key
    return isinstance(value, Mapping) and set(value) == keys


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(value)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[group][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    return []


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = _shape(data, {"scenarios"})
    scenarios = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = isinstance(scenarios, list) and all(
        _record_shape(row, SCENE_KEYS, "scenario") and isinstance(row.get("slots"), list)
        for row in scenarios
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires a slots array and no unknown non-canonical keys",
        ),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = ("slot_containers", "action_containers", "send_fields", "power_fields")
    flags = {name: True for name in names}
    details = {name: "all submitted decision fields satisfy their local contract" for name in names}
    rows = _rows(data)
    if not rows:
        for name in names:
            flags[name] = False
            details[name] = "scenarios: no decision records"
    for scene_index, scene_row in enumerate(rows):
        if not isinstance(scene_row, Mapping) or not isinstance(scene_row.get("slots"), list):
            continue
        for slot_index, slot in enumerate(scene_row["slots"]):
            # Canonical slot IDs and required send/power fields have their own
            # atoms; this container atom owns the container type and unknown
            # keys so a missing field is not double-counted.
            slot_ok = (
                isinstance(slot, Mapping)
                and set(slot) <= SLOT_KEYS
                and isinstance(slot.get("actions"), list)
            )
            if not slot_ok and flags["slot_containers"]:
                details["slot_containers"] = (
                    f"scenarios[{scene_index}].slots[{slot_index}]: "
                    "invalid keys or actions container"
                )
            flags["slot_containers"] &= slot_ok
            if not isinstance(slot, Mapping) or not isinstance(slot.get("actions"), list):
                continue
            for action_index, action in enumerate(slot["actions"]):
                # Field presence/type and explicit robot IDs are scored by
                # their dedicated atoms; this atom owns container type and
                # unknown keys only.
                action_ok = isinstance(action, Mapping) and set(action) <= ACTION_KEYS
                if not action_ok and flags["action_containers"]:
                    details["action_containers"] = (
                        f"scenarios[{scene_index}].slots[{slot_index}]."
                        f"actions[{action_index}]: invalid container or unknown keys"
                    )
                flags["action_containers"] &= action_ok
                if not isinstance(action, Mapping):
                    continue
                send_ok = isinstance(action.get("send"), bool)
                if not send_ok and flags["send_fields"]:
                    details["send_fields"] = (
                        f"scenarios[{scene_index}].slots[{slot_index}]."
                        f"actions[{action_index}].send: expected Boolean"
                    )
                flags["send_fields"] &= send_ok
                power_ok = _integer(action.get("power_dbm"))
                if not power_ok and flags["power_fields"]:
                    details["power_fields"] = (
                        f"scenarios[{scene_index}].slots[{slot_index}]."
                        f"actions[{action_index}].power_dbm: expected integer"
                    )
                flags["power_fields"] &= power_ok
    return [_atom("result_evidence", name, flags[name], details[name]) for name in names]


def _canonical_defects(
    actual: Sequence[Hashable], expected: Sequence[Hashable], label: str
) -> list[str]:
    defects: list[str] = []
    for key in expected:
        count = actual.count(key)
        if count == 0:
            defects.append(f"{label}: missing key={_safe_repr(key)}")
        elif count > 1:
            defects.extend(
                f"{label}: duplicate key={_safe_repr(key)}" for _ in range(count - 1)
            )
    expected_set = set(expected)
    defects.extend(
        f"{label}: unknown key={_safe_repr(key)}"
        for key in actual
        if key not in expected_set
    )
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping) and "scenario" in row:
            name = row.get("scenario")
            scene_keys.append(name if isinstance(name, str) else ("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))
    for scene in SCENES:
        row = next(
            (item for item in rows if isinstance(item, Mapping) and item.get("scenario") == scene),
            None,
        )
        slots = row.get("slots") if isinstance(row, Mapping) else None
        if not isinstance(slots, list):
            defects.extend(
                f"{scene}.slots: missing key={t!r}" for t in range(HORIZON)
            )
            defects.extend(
                f"{scene}.actions: missing key={(t, robot)!r}"
                for t in range(HORIZON)
                for robot in ROBOTS
            )
            continue
        slot_keys: list[Hashable] = []
        action_keys: list[Hashable] = []
        for slot_index, slot in enumerate(slots):
            if isinstance(slot, Mapping) and "t" in slot:
                slot_t = slot.get("t")
                slot_keys.append(slot_t if _integer(slot_t) else ("invalid_slot", slot_index))
                actions = slot.get("actions")
                if _integer(slot_t) and isinstance(actions, list):
                    for action_index, action in enumerate(actions):
                        if isinstance(action, Mapping) and "robot" in action:
                            robot = action.get("robot")
                            action_keys.append(
                                (slot_t, robot)
                                if _integer(robot)
                                else ("invalid_robot", slot_t, action_index)
                            )
        defects.extend(_canonical_defects(slot_keys, list(range(HORIZON)), f"{scene}.slots"))
        expected_actions = [(t, robot) for t in range(HORIZON) for robot in ROBOTS]
        defects.extend(_canonical_defects(action_keys, expected_actions, f"{scene}.actions"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    seen_scenes: set[str] = set()
    for row in _rows(data):
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            continue
        scene = row["scenario"]
        if scene in seen_scenes:
            continue
        seen_scenes.add(scene)
        slots = row.get("slots")
        if not isinstance(slots, list):
            continue
        for slot in slots:
            if (
                isinstance(slot, Mapping)
                and _integer(slot.get("t"))
                and (scene, slot["t"]) not in records
            ):
                records[(scene, slot["t"])] = slot
    return records


def _raw_robot_action(slot: Any, robot: int) -> Mapping[str, Any]:
    if not isinstance(slot, Mapping) or not isinstance(slot.get("actions"), list):
        return {}
    for action in slot["actions"]:
        if (
            isinstance(action, Mapping)
            and _integer(action.get("robot"))
            and action.get("robot") == robot
        ):
            return action
    return {}


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _move_toward(position: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    distance = math.hypot(dx, dy)
    if distance <= ROUTE_SPEED + EPS:
        return (float(target[0]), float(target[1]))
    scale = ROUTE_SPEED / distance
    return (float(position[0]) + scale * dx, float(position[1]) + scale * dy)


def _obstruction(scene: str, position: Sequence[float]) -> float:
    total = 0.0
    for x0, x1, y0, y1, loss in SCENE_DATA[scene]["obstacles"]:
        if x0 - EPS <= position[0] <= x1 + EPS and y0 - EPS <= position[1] <= y1 + EPS:
            total += loss
    return total


def _tail_component(scene_index: int, robot: int, t: int) -> float:
    residue = (11 * t + 7 * robot + 5 * scene_index) % 31
    if residue == 0:
        return 9.0
    if residue <= 2:
        return 5.0
    if residue <= 6:
        return 2.0
    return 0.0


def _disturbance(scene: str, robot: int, t: int) -> float:
    scene_index = SCENES.index(scene)
    value = float(SCENE_DATA[scene]["base_interference"])
    value += _tail_component(scene_index, robot, t)
    if scene == "shielded_turns" and (t + 2 * robot) % 17 in {8, 9}:
        value += 3.0
    elif scene == "interference_corridor" and (t // 4 + robot) % 4 == 1:
        value += 4.0
    elif scene == "shared_bottleneck" and 15 <= t <= 29 and (t + robot) % 3 != 0:
        value += 3.0
    elif scene == "long_return" and t >= 24 and (t + 2 * robot) % 7 in {0, 1}:
        value += 4.0
    return value


def _baseline(scene: str, robot: int, t: int, position: Sequence[float]) -> float:
    distance = _distance(position, ACCESS_POINT)
    return (
        -44.0
        - 12.0 * math.log10(1.0 + distance)
        - _obstruction(scene, position)
        - _disturbance(scene, robot, t)
    )


def _energy_cost(power_dbm: int) -> float:
    return 1.0 + 0.015 * power_dbm * power_dbm


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _boundary_snapshot(
    t: int, states: Sequence[Mapping[str, Any]], baselines: Sequence[float] | None
) -> dict[str, Any]:
    return {
        "t": t,
        "baselines": None if baselines is None else [float(value) for value in baselines],
        "robots": [
            {
                "position": [float(value) for value in state["position"]],
                "mode": state["mode"],
                "checkpoint": state["checkpoint"],
                "release": state["release"],
                "deadline": state["deadline"],
                "outcomes": list(state["outcomes"]),
                "release_slots": list(state["release_slots"]),
                "confirm_slots": list(state["confirm_slots"]),
                "depart_until": state["depart_until"],
                "energy": float(state["energy"]),
                "last_request_t": state["last_request_t"],
                "last_request_power": state["last_request_power"],
                "request_streak": state["request_streak"],
            }
            for state in states
        ],
    }


def _simulate_scene(
    scene: str,
    before_source: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    routes = SCENE_DATA[scene]["routes"]
    windows = SCENE_DATA[scene]["windows"]
    states: list[dict[str, Any]] = []
    for robot in ROBOTS:
        states.append(
            {
                "position": tuple(float(value) for value in routes[robot][0]),
                "mode": "awaiting_report",
                "checkpoint": 0,
                "release": 0,
                "deadline": int(windows[0]),
                "energy": INITIAL_ENERGY,
                "outcomes": ["pending", *(["unreleased"] * (len(CHECKPOINTS) - 1))],
                "release_slots": [0, *([None] * (len(CHECKPOINTS) - 1))],
                "confirm_slots": [None for _ in CHECKPOINTS],
                "depart_until": None,
                "last_request_t": None,
                "last_request_power": None,
                "request_streak": 0,
            }
        )

    arrivals: dict[int, list[dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    consumed: dict[str, dict[tuple[int, int], Any]] = {"send": {}, "power_dbm": {}}
    decision_slots: list[dict[str, Any]] = []
    boundary_trace: list[dict[str, Any]] = []
    legality_names = (
        "power_selection",
        "concurrent_capacity",
        "report_availability",
        "energy_budget",
        "attempt_streak",
        "power_slew",
    )
    legality = {name: True for name in legality_names}
    legality_errors: dict[str, list[str]] = {name: [] for name in legality_names}
    for t in range(TERMINAL_BOUNDARY + 1):
        due = sorted(
            arrivals.get(t, []),
            key=lambda item: (item["arrival_slot"], item["generated_slot"], item["robot"]),
        )
        for attempt in due:
            robot = int(attempt["robot"])
            state = states[robot]
            if int(attempt["arrival_slot"]) != t or int(attempt["generated_slot"]) >= t:
                continue
            checkpoint = int(attempt["checkpoint"])
            if (
                state["mode"] == "awaiting_report"
                and state["checkpoint"] == checkpoint
                and state["outcomes"][checkpoint] == "pending"
                and t <= int(state["deadline"])
            ):
                state["outcomes"][checkpoint] = "verified"
                state["confirm_slots"][checkpoint] = t
                attempt["confirmed"] = True

        for state in states:
            if (
                state["mode"] == "awaiting_report"
                and state["outcomes"][state["checkpoint"]] == "pending"
                and t >= int(state["deadline"])
            ):
                state["outcomes"][state["checkpoint"]] = "missed"

        for robot, state in enumerate(states):
            if state["mode"] == "awaiting_report" and state["outcomes"][state["checkpoint"]] in {
                "verified",
                "missed",
            }:
                if state["checkpoint"] == CHECKPOINTS[-1]:
                    state["mode"] = "complete"
                else:
                    state["checkpoint"] += 1
                    state["mode"] = "departing"
                    state["depart_until"] = t + 1
            elif state["mode"] == "departing" and t >= int(state["depart_until"]):
                state["mode"] = "travel"
                state["depart_until"] = None
            if (
                state["mode"] == "travel"
                and _distance(state["position"], routes[robot][state["checkpoint"]]) <= EPS
            ):
                checkpoint = state["checkpoint"]
                state["mode"] = "awaiting_report"
                state["release"] = t
                state["deadline"] = t + int(windows[checkpoint])
                state["outcomes"][checkpoint] = "pending"
                state["release_slots"][checkpoint] = t

        baselines = (
            None
            if t == TERMINAL_BOUNDARY
            else [_baseline(scene, robot, t, states[robot]["position"]) for robot in ROBOTS]
        )
        boundary_trace.append(_boundary_snapshot(t, states, baselines))
        if t == TERMINAL_BOUNDARY:
            break
        assert baselines is not None

        raw_slot = before_source({"t": t})
        view = {
            "scenario": scene,
            "t": t,
            "robots": [
                {
                    "robot": robot,
                    "position": list(states[robot]["position"]),
                    "route_state": states[robot]["mode"],
                    "checkpoint": (
                        None if states[robot]["mode"] == "complete" else states[robot]["checkpoint"]
                    ),
                    "release_slot": (
                        states[robot]["release"]
                        if states[robot]["mode"] == "awaiting_report"
                        else None
                    ),
                    "deadline": (
                        states[robot]["deadline"]
                        if states[robot]["mode"] == "awaiting_report"
                        else None
                    ),
                    "energy_remaining": states[robot]["energy"],
                    "baseline_rssi_dbm": baselines[robot],
                    "preceding_send_streak": (
                        states[robot]["request_streak"]
                        if states[robot]["last_request_t"] == t - 1
                        else 0
                    ),
                    "preceding_send_power_dbm": (
                        states[robot]["last_request_power"]
                        if states[robot]["last_request_t"] == t - 1
                        else None
                    ),
                    "checkpoint_values": list(VALUES[robot]),
                    "deadline_windows": list(windows),
                    "route": [list(point) for point in routes[robot]],
                    "outcomes": list(states[robot]["outcomes"]),
                    "confirm_slots": list(states[robot]["confirm_slots"]),
                }
                for robot in ROBOTS
            ],
        }
        if raw_slot is None:
            raw_slot = before_source(view)

        raw_actions: list[Mapping[str, Any]] = []
        normalized: list[dict[str, Any]] = []
        for robot in ROBOTS:
            supplied = _raw_robot_action(raw_slot, robot)
            raw_actions.append(supplied)
            raw_send = supplied.get("send")
            raw_power = supplied.get("power_dbm")
            consumed["send"][(t, robot)] = raw_send
            consumed["power_dbm"][(t, robot)] = raw_power
            typed = isinstance(raw_send, bool) and _integer(raw_power)
            send = bool(raw_send) if typed else False
            power = int(raw_power) if typed else 0
            pair_ok = typed and ((not send and power == 0) or (send and power in POWER_LEVELS))
            if typed and not pair_ok:
                legality["power_selection"] = False
                legality_errors["power_selection"].append(
                    f"{scene}.slots[{t}].actions[{robot}]: send={send}, "
                    f"power_dbm={_safe_repr(power)}"
                )
                send = False
                power = 0
            normalized.append({"robot": robot, "send": send, "power_dbm": power})

        physical_requests: list[int] = []
        for robot in ROBOTS:
            action = normalized[robot]
            state = states[robot]
            if not action["send"]:
                continue
            if state["mode"] != "awaiting_report" or t >= int(state["deadline"]):
                legality["report_availability"] = False
                legality_errors["report_availability"].append(
                    f"{scene}.slots[{t}].actions[{robot}]: no sendable current report"
                )
                continue
            cost = _energy_cost(action["power_dbm"])
            if state["energy"] + EPS < cost:
                legality["energy_budget"] = False
                legality_errors["energy_budget"].append(
                    f"{scene}.slots[{t}].actions[{robot}]: cost={cost:.6f}, "
                    f"remaining={state['energy']:.6f}"
                )
                continue
            preceding_streak = state["request_streak"] if state["last_request_t"] == t - 1 else 0
            if preceding_streak + 1 > MAX_STREAK:
                legality["attempt_streak"] = False
                legality_errors["attempt_streak"].append(
                    f"{scene}.slots[{t}].actions[{robot}]: "
                    f"consecutive_attempts={preceding_streak + 1}"
                )
            if (
                state["last_request_t"] == t - 1
                and abs(action["power_dbm"] - int(state["last_request_power"]))
                > MAX_ADJACENT_CHANGE
            ):
                legality["power_slew"] = False
                legality_errors["power_slew"].append(
                    f"{scene}.slots[{t}].actions[{robot}]: adjacent power change "
                    f"from {state['last_request_power']} to {action['power_dbm']}"
                )
            physical_requests.append(robot)

        if len(physical_requests) > MAX_CONCURRENT:
            legality["concurrent_capacity"] = False
            legality_errors["concurrent_capacity"].append(
                f"{scene}.slots[{t}]: physical_send_count={len(physical_requests)} "
                f"exceeds {MAX_CONCURRENT}"
            )

        current_attempts: list[dict[str, Any]] = []
        for robot in physical_requests:
            action = normalized[robot]
            state = states[robot]
            cost = _energy_cost(action["power_dbm"])
            state["energy"] -= cost
            attempt = {
                "robot": robot,
                "checkpoint": state["checkpoint"],
                "generated_slot": t,
                "power_dbm": action["power_dbm"],
                "baseline_rssi_dbm": baselines[robot],
                "energy_cost": cost,
                "arrival_slot": None,
                "confirmed": False,
            }
            current_attempts.append(attempt)
            attempts.append(attempt)

        for attempt in current_attempts:
            robot = int(attempt["robot"])
            interference = 0.35 * sum(
                int(other["power_dbm"])
                for other in current_attempts
                if int(other["robot"]) != robot
            )
            rssi = float(attempt["baseline_rssi_dbm"]) + int(attempt["power_dbm"]) - interference
            margin = rssi - RSSI_THRESHOLD
            attempt["interferers"] = sorted(
                int(other["robot"]) for other in current_attempts if int(other["robot"]) != robot
            )
            attempt["rssi_dbm"] = rssi
            attempt["margin_db"] = margin
            if margin >= 0.0:
                delay = 2 if margin < 6.0 else 1
                arrival_slot = t + delay
                attempt["arrival_slot"] = arrival_slot
                if arrival_slot <= TERMINAL_BOUNDARY:
                    arrivals.setdefault(arrival_slot, []).append(attempt)
        for robot in ROBOTS:
            state = states[robot]
            if robot in physical_requests:
                previous = state["request_streak"] if state["last_request_t"] == t - 1 else 0
                state["request_streak"] = previous + 1
                state["last_request_t"] = t
                state["last_request_power"] = normalized[robot]["power_dbm"]
            else:
                state["request_streak"] = 0

        for robot, state in enumerate(states):
            if state["mode"] == "travel":
                state["position"] = _move_toward(
                    state["position"], routes[robot][state["checkpoint"]]
                )

        decision_slots.append(
            {
                "t": t,
                "actions": [
                    {
                        "robot": robot,
                        "send": raw_actions[robot].get("send"),
                        "power_dbm": raw_actions[robot].get("power_dbm"),
                    }
                    for robot in ROBOTS
                ],
            }
        )

    energy_used = [INITIAL_ENERGY - float(state["energy"]) for state in states]

    return {
        "scenario_record": {"scenario": scene, "slots": decision_slots},
        "legality": legality,
        "legality_errors": legality_errors,
        "consumed": consumed,
        "attempts": attempts,
        "states": states,
        "energy_used": energy_used,
        "boundary_trace": boundary_trace,
        "decision_stream_complete": True,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    def before(view: Mapping[str, Any]) -> Any:
        return records.get((scene, int(view["t"])), {})

    result = _simulate_scene(scene, before)
    incomplete: list[int] = []
    for t in range(HORIZON):
        slot = records.get((scene, t))
        actions = slot.get("actions") if isinstance(slot, Mapping) else None
        if not isinstance(actions, list):
            incomplete.append(t)
            continue
        for robot in ROBOTS:
            action = next(
                (
                    row
                    for row in actions
                    if isinstance(row, Mapping)
                    and _integer(row.get("robot"))
                    and row.get("robot") == robot
                ),
                None,
            )
            if (
                action is None
                or not isinstance(action.get("send"), bool)
                or not _integer(action.get("power_dbm"))
            ):
                incomplete.append(t)
                break
    result["decision_stream_complete"] = not incomplete
    result["first_incomplete_t"] = min(incomplete) if incomplete else None
    return result


def _reference_baseline(
    scene: str, robot: int, t: int, position: Sequence[float]
) -> float:
    distance = math.hypot(
        float(position[0]) - ACCESS_POINT[0], float(position[1]) - ACCESS_POINT[1]
    )
    obstruction = sum(
        loss
        for x0, x1, y0, y1, loss in SCENE_DATA[scene]["obstacles"]
        if x0 - EPS <= float(position[0]) <= x1 + EPS
        and y0 - EPS <= float(position[1]) <= y1 + EPS
    )
    scene_index = SCENES.index(scene)
    residue = (11 * t + 7 * robot + 5 * scene_index) % 31
    tail = 9.0 if residue == 0 else 5.0 if residue in {1, 2} else 2.0 if 3 <= residue <= 6 else 0.0
    burst = 0.0
    if scene == "shielded_turns" and (t + 2 * robot) % 17 in {8, 9}:
        burst = 3.0
    elif scene == "interference_corridor" and (t // 4 + robot) % 4 == 1:
        burst = 4.0
    elif scene == "shared_bottleneck" and 15 <= t <= 29 and (t + robot) % 3 != 0:
        burst = 3.0
    elif scene == "long_return" and t >= 24 and (t + 2 * robot) % 7 in {0, 1}:
        burst = 4.0
    disturbance = float(SCENE_DATA[scene]["base_interference"]) + tail + burst
    return -44.0 - 12.0 * math.log10(1.0 + distance) - obstruction - disturbance


def _reference_move(
    position: Sequence[float], target: Sequence[float]
) -> tuple[float, float]:
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    distance = math.hypot(dx, dy)
    if distance <= ROUTE_SPEED + EPS:
        return (float(target[0]), float(target[1]))
    return (
        float(position[0]) + ROUTE_SPEED * dx / distance,
        float(position[1]) + ROUTE_SPEED * dy / distance,
    )


def _reference_replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    routes = SCENE_DATA[scene]["routes"]
    windows = SCENE_DATA[scene]["windows"]
    states = [
        {
            "position": tuple(float(value) for value in routes[robot][0]),
            "mode": "awaiting_report",
            "checkpoint": 0,
            "release": 0,
            "deadline": int(windows[0]),
            "energy": INITIAL_ENERGY,
            "outcomes": ["pending", *("unreleased" for _ in CHECKPOINTS[1:])],
            "release_slots": [0, *(None for _ in CHECKPOINTS[1:])],
            "confirm_slots": [None for _ in CHECKPOINTS],
            "depart_until": None,
            "last_request_t": None,
            "last_request_power": None,
            "request_streak": 0,
        }
        for robot in ROBOTS
    ]
    arrivals: dict[int, list[dict[str, Any]]] = {}
    attempts: list[dict[str, Any]] = []
    boundary_trace: list[dict[str, Any]] = []

    for t in range(TERMINAL_BOUNDARY + 1):
        for attempt in sorted(
            arrivals.get(t, []), key=lambda item: (item["generated_slot"], item["robot"])
        ):
            robot = int(attempt["robot"])
            state = states[robot]
            checkpoint = int(attempt["checkpoint"])
            if (
                state["mode"] == "awaiting_report"
                and state["checkpoint"] == checkpoint
                and state["outcomes"][checkpoint] == "pending"
                and t <= int(state["deadline"])
            ):
                state["outcomes"][checkpoint] = "verified"
                state["confirm_slots"][checkpoint] = t
                attempt["confirmed"] = True

        for state in states:
            if (
                state["mode"] == "awaiting_report"
                and state["outcomes"][state["checkpoint"]] == "pending"
                and t >= int(state["deadline"])
            ):
                state["outcomes"][state["checkpoint"]] = "missed"

        for robot, state in enumerate(states):
            if state["mode"] == "awaiting_report" and state["outcomes"][state["checkpoint"]] in {
                "verified",
                "missed",
            }:
                if state["checkpoint"] == CHECKPOINTS[-1]:
                    state["mode"] = "complete"
                else:
                    state["checkpoint"] += 1
                    state["mode"] = "departing"
                    state["depart_until"] = t + 1
            elif state["mode"] == "departing" and t >= int(state["depart_until"]):
                state["mode"] = "travel"
                state["depart_until"] = None
            target = routes[robot][state["checkpoint"]]
            if state["mode"] == "travel" and math.hypot(
                float(state["position"][0]) - float(target[0]),
                float(state["position"][1]) - float(target[1]),
            ) <= EPS:
                checkpoint = int(state["checkpoint"])
                state["mode"] = "awaiting_report"
                state["release"] = t
                state["deadline"] = t + int(windows[checkpoint])
                state["outcomes"][checkpoint] = "pending"
                state["release_slots"][checkpoint] = t

        baselines = (
            None
            if t == TERMINAL_BOUNDARY
            else [
                _reference_baseline(scene, robot, t, states[robot]["position"])
                for robot in ROBOTS
            ]
        )
        boundary_trace.append(_boundary_snapshot(t, states, baselines))
        if t == TERMINAL_BOUNDARY:
            break
        assert baselines is not None

        slot = records.get((scene, t), {})
        normalized: list[dict[str, Any]] = []
        for robot in ROBOTS:
            action = _raw_robot_action(slot, robot)
            raw_send = action.get("send")
            raw_power = action.get("power_dbm")
            typed = isinstance(raw_send, bool) and _integer(raw_power)
            send = bool(raw_send) if typed else False
            power = int(raw_power) if typed else 0
            if not (typed and ((not send and power == 0) or (send and power in POWER_LEVELS))):
                send = False
                power = 0
            normalized.append({"send": send, "power_dbm": power})

        physical: list[int] = []
        for robot, action in enumerate(normalized):
            state = states[robot]
            if not action["send"]:
                continue
            cost = 1.0 + 0.015 * action["power_dbm"] * action["power_dbm"]
            if state["mode"] != "awaiting_report" or t >= int(state["deadline"]):
                continue
            if cost > float(state["energy"]) + EPS:
                continue
            physical.append(robot)

        current: list[dict[str, Any]] = []
        for robot in physical:
            power = int(normalized[robot]["power_dbm"])
            cost = 1.0 + 0.015 * power * power
            states[robot]["energy"] -= cost
            attempt = {
                "robot": robot,
                "checkpoint": int(states[robot]["checkpoint"]),
                "generated_slot": t,
                "power_dbm": power,
                "baseline_rssi_dbm": float(baselines[robot]),
                "energy_cost": cost,
                "arrival_slot": None,
                "confirmed": False,
            }
            attempts.append(attempt)
            current.append(attempt)

        for attempt in current:
            robot = int(attempt["robot"])
            other_powers = [
                int(other["power_dbm"])
                for other in current
                if int(other["robot"]) != robot
            ]
            interference = 0.35 * sum(other_powers)
            rssi = float(attempt["baseline_rssi_dbm"]) + int(attempt["power_dbm"]) - interference
            margin = rssi - RSSI_THRESHOLD
            attempt["interferers"] = sorted(
                int(other["robot"]) for other in current if int(other["robot"]) != robot
            )
            attempt["rssi_dbm"] = rssi
            attempt["margin_db"] = margin
            if margin >= 0.0:
                arrival = t + (1 if margin >= 6.0 else 2)
                attempt["arrival_slot"] = arrival
                if arrival <= TERMINAL_BOUNDARY:
                    arrivals.setdefault(arrival, []).append(attempt)

        for robot, state in enumerate(states):
            if robot in physical:
                previous = int(state["request_streak"]) if state["last_request_t"] == t - 1 else 0
                state["request_streak"] = previous + 1
                state["last_request_t"] = t
                state["last_request_power"] = normalized[robot]["power_dbm"]
            else:
                state["request_streak"] = 0
            if state["mode"] == "travel":
                state["position"] = _reference_move(
                    state["position"], routes[robot][state["checkpoint"]]
                )

    return {
        "attempts": attempts,
        "states": states,
        "energy_used": [INITIAL_ENERGY - float(state["energy"]) for state in states],
        "boundary_trace": boundary_trace,
    }


def _online_scene(scene: str, policy: Any) -> dict[str, Any]:
    if hasattr(policy, "reset"):
        policy.reset()

    def before(view: Mapping[str, Any]) -> Any:
        if set(view) == {"t"}:
            return None
        return {"t": view["t"], "actions": policy.decide_joint(view)}

    return _simulate_scene(scene, before)["scenario_record"]


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = (
        "power_selection",
        "concurrent_capacity",
        "report_availability",
        "energy_budget",
        "attempt_streak",
        "power_slew",
    )
    flags = {name: True for name in names}
    details = {name: "legal in all scenes" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("legality", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("legality_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if scene_flags.get(name, False):
                continue
            failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
            if name in {
                "concurrent_capacity",
                "report_availability",
                "energy_budget",
                "attempt_streak",
                "power_slew",
            } and isinstance(result, Mapping):
                cutoff = result.get("first_incomplete_t")
                attributable: list[str] = []
                for detail in failures:
                    slot_match = re.search(r"\.slots\[(\d+)\]", detail)
                    if slot_match is None:
                        continue
                    t = int(slot_match.group(1))
                    if not _integer(cutoff):
                        attributable.append(detail)
                    elif name == "concurrent_capacity" and t < cutoff:
                        attributable.append(detail)
                    elif name != "concurrent_capacity" and t <= cutoff:
                        attributable.append(detail)
                failures = attributable
                if not failures:
                    continue
            flags[name] = False
            details[name] = failures[0] if failures else f"{scene}: {name} unavailable"
    return [_atom("decision_legality", name, flags[name], details[name]) for name in names]


def _consumption_mismatch(
    scene: str,
    records: Mapping[Hashable, Any],
    result: Mapping[str, Any],
) -> str | None:
    consumed = result.get("consumed")
    if not isinstance(consumed, Mapping):
        return f"{scene}: consumption audit unavailable"
    for t in range(HORIZON):
        slot = records.get((scene, t))
        for robot in ROBOTS:
            action = _raw_robot_action(slot, robot)
            for field in ("send", "power_dbm"):
                values = consumed.get(field)
                if not isinstance(values, Mapping) or (t, robot) not in values:
                    return f"{scene}.slots[{t}].actions[{robot}].{field}: unavailable"
                if action.get(field) != values[(t, robot)]:
                    return (
                        f"{scene}.slots[{t}].actions[{robot}].{field}: "
                        f"submitted={_safe_repr(action.get(field))}, "
                        f"consumed={_safe_repr(values[(t, robot)])}"
                    )
    return None


def _first_mismatch(expected: Any, actual: Any, path: str) -> str | None:
    if isinstance(expected, float) or isinstance(actual, float):
        if (
            isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(expected))
            and math.isfinite(float(actual))
            and math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=EPS)
        ):
            return None
        return (
            f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}, "
            f"tolerance={EPS}"
        )
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected mapping, actual={type(actual).__name__}"
        if set(expected) != set(actual):
            return f"{path}: expected keys={sorted(expected)}, actual keys={sorted(actual)}"
        for key in expected:
            mismatch = _first_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f"{path}: expected sequence, actual={type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path}: expected length={len(expected)}, actual length={len(actual)}"
        for index, value in enumerate(expected):
            mismatch = _first_mismatch(value, actual[index], f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}"
    return None


def _trace_projection(
    result: Mapping[str, Any], robot_fields: Sequence[str], include_baselines: bool = False
) -> Any:
    trace = result.get("boundary_trace")
    if not isinstance(trace, list):
        return None
    projected = []
    for boundary in trace:
        if not isinstance(boundary, Mapping) or not isinstance(boundary.get("robots"), list):
            projected.append(None)
            continue
        item: dict[str, Any] = {
            "t": boundary.get("t"),
            "robots": [
                {field: robot.get(field) for field in robot_fields}
                if isinstance(robot, Mapping)
                else None
                for robot in boundary["robots"]
            ],
        }
        if include_baselines:
            item["baselines"] = boundary.get("baselines")
        projected.append(item)
    return projected


def _attempt_projection(result: Mapping[str, Any], fields: Sequence[str]) -> Any:
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        return None
    return [
        {field: attempt.get(field) for field in fields}
        if isinstance(attempt, Mapping)
        else None
        for attempt in attempts
    ]


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Any]
) -> list[AtomVerdict]:
    names = (
        "decision_consumption",
        "baseline_observations",
        "interference_outcomes",
        "arrival_confirmation",
        "route_progression",
        "energy_accounting",
    )
    flags = {name: True for name in names}
    details = {name: "independent reference replay matched" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        if not isinstance(result, Mapping):
            for name in names:
                flags[name] = False
                details[name] = f"{scene}: replay unavailable"
            continue
        try:
            expected = _reference_replay_scene(scene, records)
        except Exception as exc:  # noqa: BLE001
            for name in names:
                flags[name] = False
                details[name] = f"{scene}: reference replay raised {type(exc).__name__}: {exc}"
            continue

        consumption = _consumption_mismatch(scene, records, result)
        if consumption is None:
            consumption = _first_mismatch(
                _trace_projection(
                    expected,
                    ("last_request_t", "last_request_power", "request_streak"),
                ),
                _trace_projection(
                    result,
                    ("last_request_t", "last_request_power", "request_streak"),
                ),
                f"{scene}.decision_trace",
            )
        if consumption is not None:
            flags["decision_consumption"] = False
            details["decision_consumption"] = consumption

        comparisons = {
            "baseline_observations": (
                _trace_projection(expected, (), include_baselines=True),
                _trace_projection(result, (), include_baselines=True),
                f"{scene}.baseline_trace",
            ),
            "interference_outcomes": (
                _attempt_projection(
                    expected,
                    (
                        "robot",
                        "checkpoint",
                        "generated_slot",
                        "power_dbm",
                        "interferers",
                        "rssi_dbm",
                        "margin_db",
                    ),
                ),
                _attempt_projection(
                    result,
                    (
                        "robot",
                        "checkpoint",
                        "generated_slot",
                        "power_dbm",
                        "interferers",
                        "rssi_dbm",
                        "margin_db",
                    ),
                ),
                f"{scene}.attempts",
            ),
            "arrival_confirmation": (
                {
                    "attempts": _attempt_projection(
                        expected,
                        ("robot", "checkpoint", "generated_slot", "arrival_slot", "confirmed"),
                    ),
                    "trace": _trace_projection(expected, ("outcomes", "confirm_slots")),
                },
                {
                    "attempts": _attempt_projection(
                        result,
                        ("robot", "checkpoint", "generated_slot", "arrival_slot", "confirmed"),
                    ),
                    "trace": _trace_projection(result, ("outcomes", "confirm_slots")),
                },
                f"{scene}.arrival_confirmation",
            ),
            "route_progression": (
                _trace_projection(
                    expected,
                    (
                        "position",
                        "mode",
                        "checkpoint",
                        "release",
                        "deadline",
                        "release_slots",
                        "depart_until",
                    ),
                ),
                _trace_projection(
                    result,
                    (
                        "position",
                        "mode",
                        "checkpoint",
                        "release",
                        "deadline",
                        "release_slots",
                        "depart_until",
                    ),
                ),
                f"{scene}.route_trace",
            ),
            "energy_accounting": (
                {
                    "costs": _attempt_projection(expected, ("energy_cost",)),
                    "trace": _trace_projection(expected, ("energy",)),
                    "used": expected.get("energy_used"),
                },
                {
                    "costs": _attempt_projection(result, ("energy_cost",)),
                    "trace": _trace_projection(result, ("energy",)),
                    "used": result.get("energy_used"),
                },
                f"{scene}.energy",
            ),
        }
        for name, (expected_value, actual_value, path) in comparisons.items():
            mismatch = _first_mismatch(expected_value, actual_value, path)
            if mismatch is not None:
                flags[name] = False
                details[name] = mismatch
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    primary_1_values: list[float] = []
    primary_2_values: list[float] = []
    auxiliary_1_values: list[float] = []
    auxiliary_2_values: list[float] = []
    for scene in SCENES:
        result = replayed[scene]
        windows = SCENE_DATA[scene]["windows"]
        total_value = sum(sum(values) for values in VALUES)
        verified_value = 0.0
        deadline_quality = 0.0
        for robot in ROBOTS:
            state = result["states"][robot]
            robot_verified_value = 0.0
            for checkpoint in CHECKPOINTS:
                if state["outcomes"][checkpoint] != "verified":
                    continue
                value = float(VALUES[robot][checkpoint])
                verified_value += value
                robot_verified_value += value
                release = state["release_slots"][checkpoint]
                confirm = state["confirm_slots"][checkpoint]
                if _integer(release) and _integer(confirm):
                    deadline_quality += _clip(
                        1.0 - (confirm - release) / float(windows[checkpoint])
                    )
            energy_used = float(result["energy_used"][robot])
            denominator = robot_verified_value + sum(VALUES[robot]) * energy_used / INITIAL_ENERGY
            auxiliary_1_values.append(
                robot_verified_value / denominator if denominator > 0.0 else 0.0
            )
        primary_1_values.append(verified_value / total_value)
        primary_2_values.append(deadline_quality / (len(ROBOTS) * len(CHECKPOINTS)))

        attempt_scores: list[float] = []
        previous_power: dict[int, int] = {}
        for attempt in result["attempts"]:
            robot = int(attempt["robot"])
            power = int(attempt["power_dbm"])
            prior = previous_power.get(robot, 0)
            margin_score = _clip((float(attempt["margin_db"]) + 8.0) / 16.0)
            stability_score = _clip(1.0 - abs(power - prior) / 22.0)
            attempt_scores.append(0.6 * margin_score + 0.4 * stability_score)
            previous_power[robot] = power
        auxiliary_2_values.append(
            sum(attempt_scores) / len(attempt_scores) if attempt_scores else 0.0
        )
    return {
        "primary_metrics_1": sum(primary_1_values) / len(primary_1_values),
        "primary_metrics_2": sum(primary_2_values) / len(primary_2_values),
        "auxiliary_metrics_1": sum(auxiliary_1_values) / (len(SCENES) * len(ROBOTS)),
        "auxiliary_metrics_2": sum(auxiliary_2_values) / len(auxiliary_2_values),
    }


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    visible_lines = ["" for _ in lines]
    visible = [False] * len(lines)
    in_fence = False
    fence_marker = ""
    in_comment = False
    for index, line in enumerate(lines):
        if in_fence:
            if line.lstrip().startswith(fence_marker):
                in_fence = False
            continue
        remainder = line
        kept = ""
        while remainder:
            if in_comment:
                end = remainder.find("-->")
                if end < 0:
                    remainder = ""
                    continue
                in_comment = False
                remainder = remainder[end + 3 :]
                continue
            start = remainder.find("<!--")
            if start < 0:
                kept += remainder
                remainder = ""
                continue
            kept += remainder[:start]
            in_comment = True
            remainder = remainder[start + 4 :]
        stripped = kept.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = True
            fence_marker = stripped[:3]
            continue
        visible_lines[index] = kept
        visible[index] = True
    headings: list[tuple[int, int, int, str]] = []
    for index, line in enumerate(visible_lines):
        if not visible[index]:
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
        if match:
            headings.append((index, index + 1, len(match.group(1)), match.group(2).strip()))
        elif (
            line.strip()
            and index + 1 < len(lines)
            and visible[index + 1]
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", visible_lines[index + 1])
        ):
            level = 1 if "=" in visible_lines[index + 1] else 2
            headings.append((index, index + 2, level, line.strip()))
    heading_markup = {
        line_index
        for start, body_start, _, _ in headings
        for line_index in range(start, body_start)
    }
    result: dict[str, str] = {}
    for order, (_, body_start, level, title) in enumerate(headings):
        end = next(
            (start for start, _, next_level, _ in headings[order + 1 :] if next_level <= level),
            len(lines),
        )
        body = "\n".join(
            visible_lines[index]
            for index in range(body_start, end)
            if visible[index] and index not in heading_markup
        ).strip()
        if title not in result or (body and not result[title]):
            result[title] = body
    return result


def _report_score(ws: Workspace) -> tuple[float, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    if not _regular(ws, "report.md"):
        return 0.0, "report.md missing or not a regular file"
    try:
        report_text = ws.read_text("report.md") or ""
    except (OSError, TypeError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    headings = _headings(report_text)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


def _fail_run_checks(state: EvalState, detail: str) -> EvalState:
    state.official_ok = False
    state.official_message = detail
    for name in (
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    ):
        state.scores[name] = 0.0
        state.diagnostics[name] = f"official run failed; {detail}"
    return state


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        except TimeoutError:
            state.timed_out = True
            return _fail_run_checks(state, "pre-run results cleanup timed out after 30s")
        except Exception as exc:  # noqa: BLE001
            return _fail_run_checks(
                state, f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        start = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - start
            state.timed_out = True
            return _fail_run_checks(
                state, f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - start
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; elapsed={state.elapsed_s:.3f}s",
            )
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - start
        state.timed_out = state.exit_code is None and state.elapsed_s >= 600.0
        state.result_file_is_new_regular = getattr(removed, "exit_code", None) == 0 and _regular(
            ws, "results.json"
        )
        state.official_ok = (
            state.run_started and state.exit_code == 0 and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = (
            _score_atoms(result_atoms, ATOM_SHARES["result_json"]) if state.official_ok else 0.0
        )
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.official_ok and state.parse_ok:
            state.records = _index_records(state.data)
            evidence_atoms = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(
                evidence_atoms, ATOM_SHARES["result_evidence"]
            )
            state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
            coverage, defects = _coverage_score(state.data)
            state.scores["canonical_coverage"] = coverage
            state.diagnostics["canonical_coverage"] = (
                f"defect_count={len(defects)}; defects={defects[:12]}"
            )
            state.replayed = {scene: _replay_scene(scene, state.records) for scene in SCENES}
            legality_atoms = _legality_atoms(state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                legality_atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
            replay_atoms = _replay_atoms(state.records, state.replayed)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
            required = (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            )
            if all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in required):
                state.metrics = _metrics(state.replayed)
                state.validity_gate = True
        else:
            for name in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[name] = 0.0
                state.diagnostics[name] = "official run or strict JSON parse failed"
    except Exception as exc:  # noqa: BLE001
        state.official_message = f"grader failure: {type(exc).__name__}: {exc}"
        for name in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(name, 0.0)
            state.diagnostics.setdefault(name, state.official_message)
    return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    task = _INFLIGHT.get(key)
    if task is None:
        task = asyncio.create_task(_build_state_uncached(ws))
        _INFLIGHT[key] = task
    try:
        state = await asyncio.shield(task)
        _CACHE[key] = state
        return state
    finally:
        if task.done() and _INFLIGHT.get(key) is task:
            del _INFLIGHT[key]


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular(ws, "run.py"):
        return 0.0, "run.py missing or not a regular file"
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


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        if state.metrics is None:
            failed = []
            if not state.official_ok:
                failed.append("official_run")
            failed.extend(
                f"{name}={state.scores.get(name, 0.0):.6f}"
                for name in (
                    "result_json",
                    "result_evidence",
                    "canonical_coverage",
                    "decision_legality",
                    "replay_validity",
                )
                if state.scores.get(name, 0.0) < 1.0 - EPS
            )
            return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
