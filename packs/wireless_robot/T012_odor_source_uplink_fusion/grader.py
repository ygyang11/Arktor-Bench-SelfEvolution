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

SCENES = (
    "crosswind_plume",
    "sheltered_plume",
    "split_plume",
    "wall_source",
    "weak_signal",
)
HORIZON = 48
ROBOTS = tuple(range(4))
GRID_MAX = 11
BASE = (6, 6)
REPORT_LIMIT = 36
SCENE_DATA = {
    "crosswind_plume": {
        "source": (9, 6),
        "starts": ((0, 0), (0, 11), (3, 5), (11, 2)),
        "wind": (1.0, 0.0),
        "a": 12.0,
        "beta": 0.18,
    },
    "sheltered_plume": {
        "source": (3, 8),
        "starts": ((0, 0), (11, 0), (2, 10), (10, 11)),
        "wind": (0.0, -1.0),
        "a": 14.0,
        "beta": 0.12,
    },
    "split_plume": {
        "source": (8, 7),
        "starts": ((0, 1), (1, 11), (10, 0), (11, 10)),
        "wind": (1 / math.sqrt(2), 1 / math.sqrt(2)),
        "a": 11.0,
        "beta": 0.16,
    },
    "wall_source": {
        "source": (10, 2),
        "starts": ((0, 2), (0, 9), (5, 0), (8, 11)),
        "wind": (-1.0, 0.0),
        "a": 13.0,
        "beta": 0.20,
    },
    "weak_signal": {
        "source": (2, 9),
        "starts": ((0, 0), (11, 1), (1, 11), (10, 10)),
        "wind": (1 / math.sqrt(2), -1 / math.sqrt(2)),
        "a": 7.0,
        "beta": 0.24,
    },
}
ACTIONS = ("north", "east", "south", "west", "stay")
DELTA = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
    "stay": (0, 0),
}
EPS = 1e-6
COMMAND_TIE_EPS = 1e-12
DECISION_KEYS = {"t", "service_report_id", "robots"}
ROBOT_DECISION_KEYS = {"robot", "fallback_action", "send"}
ROBOT_TRACE_KEYS = {
    "t",
    "robot",
    "position",
    "concentration",
    "gradient",
    "command_id",
    "command_age",
    "executed_action",
    "post_position",
    "blocked",
    "collision",
}
BASE_TRACE_KEYS = {
    "t",
    "queue_report_ids",
    "fusion_report_ids",
    "estimate",
    "command_id",
    "command_target",
    "command_action",
    "support_report_ids",
    "confirmed",
}
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.decisions": 0.125,
        "result_evidence.robot_state": 0.125,
        "result_evidence.robot_command": 0.125,
        "result_evidence.robot_motion": 0.125,
        "result_evidence.base_queue": 0.125,
        "result_evidence.base_fusion": 0.125,
        "result_evidence.base_command": 0.125,
        "result_evidence.confirmation": 0.125,
    },
    "decision_legality": {
        "decision_legality.action_domain": 0.25,
        "decision_legality.report_budget": 0.25,
        "decision_legality.service_available": 0.25,
        "decision_legality.service_once": 0.25,
    },
    "replay_validity": {
        "replay_validity.robot_state": 0.125,
        "replay_validity.robot_observation": 0.125,
        "replay_validity.robot_command": 0.125,
        "replay_validity.robot_motion": 0.125,
        "replay_validity.queue": 0.125,
        "replay_validity.fusion": 0.125,
        "replay_validity.base_command": 0.125,
        "replay_validity.confirmation": 0.125,
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
        self.replayed: dict[Hashable, Any] = {}
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


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_integer(item) and 0 <= item <= GRID_MAX for item in value)
    )


def _vector(value: Any, low: float | None = None, high: float | None = None) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            _finite(item)
            and (low is None or float(item) >= low)
            and (high is None or float(item) <= high)
            for item in value
        )
    )


def _id(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"{prefix}[0-9]+", value) is not None


def _id_list(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, list)
        and all(_id(item, prefix) for item in value)
        and len(value) == len(set(value))
    )


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
    scenes = data.get("scenarios") if root_ok else None
    containers_ok = isinstance(scenes, list) and all(
        _shape(row, {"scenario", "decisions", "robot_trace", "base_trace"}) for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires scenario,decisions,robot_trace,base_trace",
        ),
    ]


def _decision_shape(value: Any) -> bool:
    return _shape(value, DECISION_KEYS) and isinstance(value.get("robots"), list) and all(
        _shape(row, ROBOT_DECISION_KEYS) for row in value["robots"]
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    flags = {
        name: True
        for name in (
            "decisions",
            "robot_state",
            "robot_command",
            "robot_motion",
            "base_queue",
            "base_fusion",
            "base_command",
            "confirmation",
        )
    }
    details = {name: "all submitted fields satisfy their local contract" for name in flags}
    rows = _rows(data)
    if not rows:
        for name in flags:
            flags[name] = False
            details[name] = "scenarios: no evidence records"
    for scene_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            for name in flags:
                flags[name] = False
                details[name] = f"scenarios[{scene_index}]: expected object"
            continue
        decisions = row.get("decisions")
        robot_trace = row.get("robot_trace")
        base_trace = row.get("base_trace")
        previous = flags["decisions"]
        flags["decisions"] &= isinstance(decisions, list) and all(
            _decision_shape(item)
            and _integer(item.get("t"))
            and (item.get("service_report_id") is None or _id(item.get("service_report_id"), "r"))
            and isinstance(item.get("robots"), list)
            and len(item["robots"]) == len(ROBOTS)
            and [robot.get("robot") for robot in item["robots"]] == list(ROBOTS)
            and all(
                _integer(robot.get("robot"))
                and isinstance(robot.get("fallback_action"), str)
                and isinstance(robot.get("send"), bool)
                for robot in item["robots"]
            )
            for item in decisions
        )
        if previous and not flags["decisions"]:
            details["decisions"] = (
                f"scenarios[{scene_index}].decisions: invalid record keys, field type, "
                "or robot order"
            )
        previous = flags["robot_state"]
        flags["robot_state"] &= isinstance(robot_trace, list) and all(
            isinstance(item, Mapping)
            and set(item).issubset(ROBOT_TRACE_KEYS)
            and all(
                field in item
                for field in ("t", "robot", "position", "concentration", "gradient")
            )
            and _integer(item.get("t"))
            and _integer(item.get("robot"))
            and _position(item.get("position"))
            and _finite(item.get("concentration"))
            and float(item["concentration"]) >= 0
            and _vector(item.get("gradient"))
            for item in robot_trace
        )
        if previous and not flags["robot_state"]:
            details["robot_state"] = (
                f"scenarios[{scene_index}].robot_trace: invalid state/observation field "
                "or unknown record key"
            )
        previous = flags["robot_command"]
        flags["robot_command"] &= isinstance(robot_trace, list) and all(
            isinstance(item, Mapping)
            and "command_id" in item
            and "command_age" in item
            and (
                (item.get("command_id") is None and item.get("command_age") is None)
                or (
                    _id(item.get("command_id"), "c")
                    and _integer(item.get("command_age"))
                    and 1 <= item["command_age"] <= 3
                )
            )
            for item in robot_trace
        )
        if previous and not flags["robot_command"]:
            details["robot_command"] = (
                f"scenarios[{scene_index}].robot_trace.command_id/command_age: "
                "invalid pair"
            )
        previous = flags["robot_motion"]
        flags["robot_motion"] &= isinstance(robot_trace, list) and all(
            isinstance(item, Mapping)
            and all(
                field in item
                for field in ("executed_action", "post_position", "blocked", "collision")
            )
            and isinstance(item.get("executed_action"), str)
            and item["executed_action"] in ACTIONS
            and _position(item.get("post_position"))
            and isinstance(item.get("blocked"), bool)
            and isinstance(item.get("collision"), bool)
            for item in robot_trace
        )
        if previous and not flags["robot_motion"]:
            details["robot_motion"] = (
                f"scenarios[{scene_index}].robot_trace motion fields: invalid type or range"
            )
        previous = flags["base_queue"]
        flags["base_queue"] &= isinstance(base_trace, list) and all(
            isinstance(item, Mapping)
            and set(item).issubset(BASE_TRACE_KEYS)
            and "t" in item
            and "queue_report_ids" in item
            and _integer(item.get("t"))
            and _id_list(item.get("queue_report_ids"), "r")
            for item in base_trace
        )
        if previous and not flags["base_queue"]:
            details["base_queue"] = (
                f"scenarios[{scene_index}].base_trace queue fields: invalid type "
                "or unknown record key"
            )
        previous = flags["base_fusion"]
        flags["base_fusion"] &= isinstance(base_trace, list) and all(
            isinstance(item, Mapping)
            and "fusion_report_ids" in item
            and "estimate" in item
            and _id_list(item.get("fusion_report_ids"), "r")
            and _vector(item.get("estimate"), 0.0, 11.0)
            for item in base_trace
        )
        if previous and not flags["base_fusion"]:
            details["base_fusion"] = (
                f"scenarios[{scene_index}].base_trace fusion_report_ids/estimate: "
                "invalid type or range"
            )
        previous = flags["base_command"]
        flags["base_command"] &= isinstance(base_trace, list) and all(
            isinstance(item, Mapping)
            and "command_id" in item
            and "command_target" in item
            and "command_action" in item
            and (
                (
                    item.get("command_id") is None
                    and item.get("command_target") is None
                    and item.get("command_action") is None
                )
                or (
                    _id(item.get("command_id"), "c")
                    and _integer(item.get("command_target"))
                    and item["command_target"] in ROBOTS
                    and isinstance(item.get("command_action"), str)
                    and item["command_action"] in ACTIONS
                )
            )
            for item in base_trace
        )
        if previous and not flags["base_command"]:
            details["base_command"] = (
                f"scenarios[{scene_index}].base_trace command fields: invalid null group "
                "or type"
            )
        previous = flags["confirmation"]
        flags["confirmation"] &= isinstance(base_trace, list) and all(
            isinstance(item, Mapping)
            and "support_report_ids" in item
            and "confirmed" in item
            and _id_list(item.get("support_report_ids"), "r")
            and isinstance(item.get("confirmed"), bool)
            for item in base_trace
        )
        if previous and not flags["confirmation"]:
            details["confirmation"] = (
                f"scenarios[{scene_index}].base_trace support/confirmed fields: invalid type"
            )
    return [_atom("result_evidence", name, flags[name], details[name]) for name in flags]


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for row in _rows(data):
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            for field in ("decisions", "robot_trace", "base_trace"):
                records[(row["scenario"], field)] = row.get(field)
    return records


def _canonical_defects(
    actual: Sequence[Hashable], expected: Sequence[Hashable], label: str
) -> list[str]:
    defects: list[str] = []
    for key in expected:
        count = actual.count(key)
        if count == 0:
            defects.append(f"{label}: missing key={key!r}")
        elif count > 1:
            defects.extend(f"{label}: duplicate key={key!r}" for _ in range(count - 1))
    expected_set = set(expected)
    defects.extend(f"{label}: unknown key={key!r}" for key in actual if key not in expected_set)
    same_keys = len(actual) == len(expected) and set(actual) == expected_set
    if same_keys and list(actual) != list(expected):
        defects.append(f"{label}: canonical order")
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    for index, row in enumerate(rows):
        name = row.get("scenario") if isinstance(row, Mapping) else None
        scene_keys.append(name if isinstance(name, str) else ("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))

    for scene in SCENES:
        row = next(
            (item for item in rows if isinstance(item, Mapping) and item.get("scenario") == scene),
            None,
        )
        for field, expected in (
            ("decisions", list(range(HORIZON))),
            ("robot_trace", [(t, robot) for t in range(HORIZON) for robot in ROBOTS]),
            ("base_trace", list(range(HORIZON))),
        ):
            raw = row.get(field) if isinstance(row, Mapping) else None
            if not isinstance(raw, list):
                defects.append(f"{scene}.{field}: expected list")
                continue
            actual: list[Hashable] = []
            for index, item in enumerate(raw):
                if field == "robot_trace":
                    if (
                        isinstance(item, Mapping)
                        and _integer(item.get("t"))
                        and _integer(item.get("robot"))
                    ):
                        actual.append((item["t"], item["robot"]))
                    else:
                        actual.append(("invalid_robot_trace", index))
                elif isinstance(item, Mapping) and _integer(item.get("t")):
                    actual.append(item["t"])
                else:
                    actual.append((f"invalid_{field}", index))
            defects.extend(_canonical_defects(actual, expected, f"{scene}.{field}"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _concentration(scene: str, position: Sequence[int]) -> float:
    data = SCENE_DATA[scene]
    dx = float(position[0] - data["source"][0])
    dy = float(position[1] - data["source"][1])
    distance = math.hypot(dx, dy)
    wind_term = data["wind"][0] * dx + data["wind"][1] * dy
    return float(data["a"]) / (1.0 + distance) * math.exp(
        -float(data["beta"]) * max(0.0, wind_term)
    )


def _gradient(scene: str, position: Sequence[int]) -> list[float]:
    x, y = int(position[0]), int(position[1])
    if x == 0:
        gx = _concentration(scene, (1, y)) - _concentration(scene, (0, y))
    elif x == GRID_MAX:
        gx = _concentration(scene, (GRID_MAX, y)) - _concentration(scene, (GRID_MAX - 1, y))
    else:
        gx = (_concentration(scene, (x + 1, y)) - _concentration(scene, (x - 1, y))) / 2
    if y == 0:
        gy = _concentration(scene, (x, 1)) - _concentration(scene, (x, 0))
    elif y == GRID_MAX:
        gy = _concentration(scene, (x, GRID_MAX)) - _concentration(scene, (x, GRID_MAX - 1))
    else:
        gy = (_concentration(scene, (x, y + 1)) - _concentration(scene, (x, y - 1))) / 2
    return [gx, gy]


def _report_number(report_id: str) -> int:
    return int(report_id[1:])


def _decision_maps(records: Mapping[Hashable, Any], scene: str) -> dict[int, dict[str, Any]]:
    raw = records.get((scene, "decisions"), [])
    mapped: dict[int, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return mapped
    for row in raw:
        if isinstance(row, Mapping) and _integer(row.get("t")) and row["t"] not in mapped:
            robots: dict[int, dict[str, Any]] = {}
            raw_robots = row.get("robots")
            if isinstance(raw_robots, list):
                for robot_row in raw_robots:
                    if (
                        isinstance(robot_row, Mapping)
                        and _integer(robot_row.get("robot"))
                        and robot_row["robot"] not in robots
                    ):
                        robots[robot_row["robot"]] = dict(robot_row)
            mapped[row["t"]] = {
                "service_report_id": row.get("service_report_id"),
                "robots": robots,
            }
    return mapped


def _choose_command(estimate: Sequence[float], beliefs: Sequence[Sequence[int]]) -> tuple[int, str]:
    target_distances = [
        (
            (beliefs[robot][0] - estimate[0]) ** 2
            + (beliefs[robot][1] - estimate[1]) ** 2
        )
        for robot in ROBOTS
    ]
    maximum = max(target_distances)
    target = next(
        robot
        for robot in ROBOTS
        if target_distances[robot] >= maximum - COMMAND_TIE_EPS
    )
    action_distances: list[float] = []
    for action in ACTIONS:
        dx, dy = DELTA[action]
        destination = (
            max(0, min(GRID_MAX, beliefs[target][0] + dx)),
            max(0, min(GRID_MAX, beliefs[target][1] + dy)),
        )
        distance = (destination[0] - estimate[0]) ** 2 + (destination[1] - estimate[1]) ** 2
        action_distances.append(distance)
    minimum = min(action_distances)
    best_action = next(
        action
        for action, distance in zip(ACTIONS, action_distances)
        if distance <= minimum + COMMAND_TIE_EPS
    )
    return target, best_action


def _compare(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            errors.append(
                f"{path}: expected keys={sorted(expected)}, "
                f"actual_type={type(actual).__name__}"
            )
            return
        for key, value in expected.items():
            _compare(actual[key], value, f"{path}.{key}", errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append(
                f"{path}: expected list length={len(expected)}, actual_length="
                f"{len(actual) if isinstance(actual, list) else 'not-list'}"
            )
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare(actual_item, expected_item, f"{path}[{index}]", errors)
    elif _finite(expected):
        if not _finite(actual) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=EPS
        ):
            errors.append(f"{path}: expected={expected!r}, actual={actual!r}, tolerance={EPS}")
    elif actual != expected:
        errors.append(f"{path}: expected={expected!r}, actual={actual!r}")


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    scene_index = SCENES.index(scene)
    decisions = _decision_maps(records, scene)
    positions = [list(position) for position in SCENE_DATA[scene]["starts"]]
    beliefs = [list(position) for position in SCENE_DATA[scene]["starts"]]
    estimate = [6.0, 6.0]
    reports: list[dict[str, Any]] = []
    report_by_id: dict[str, dict[str, Any]] = {}
    queue: list[str] = []
    latest_served: dict[int, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    stored_commands: list[dict[str, Any] | None] = [None] * len(ROBOTS)
    collision_history: list[list[bool]] = [[] for _ in ROBOTS]
    confirmed = False
    confirm_t: int | None = None
    expected_robot: dict[tuple[int, int], dict[str, Any]] = {}
    expected_base: dict[int, dict[str, Any]] = {}
    legality = {
        "action_domain": True,
        "report_budget": True,
        "service_available": True,
        "service_once": True,
    }
    legality_errors = {name: [] for name in legality}
    send_slots: list[int] = []
    freshness_sum = 0.0

    for t in range(HORIZON):
        arrivals = sorted(
            (
                report
                for report in reports
                if report["status"] == "delivered" and report["arrival_t"] == t
            ),
            key=lambda report: (report["arrival_t"], _report_number(report["id"])),
        )
        queue.extend(report["id"] for report in arrivals)
        queue.sort(
            key=lambda report_id: (
                report_by_id[report_id]["arrival_t"],
                _report_number(report_id),
            )
        )
        for command in commands:
            if command["created_t"] + 1 == t:
                stored_commands[command["target"]] = command
        queue_before = list(queue)

        decision = decisions.get(t, {"service_report_id": None, "robots": {}})
        robot_decisions = decision["robots"]
        fallbacks: list[str] = []
        sends: list[bool] = []
        for robot in ROBOTS:
            robot_decision = robot_decisions.get(robot, {})
            action = robot_decision.get("fallback_action")
            send = robot_decision.get("send")
            if isinstance(action, str) and action not in ACTIONS:
                legality["action_domain"] = False
                legality_errors["action_domain"].append(
                    f"{scene}.decisions[{t}].robots[{robot}].fallback_action={action!r}"
                )
                action = "stay"
            elif not isinstance(action, str):
                action = "stay"
            if not isinstance(send, bool):
                send = False
            fallbacks.append(action)
            sends.append(send)

        service_id = decision.get("service_report_id")
        serviced: dict[str, Any] | None = None
        if service_id is not None:
            if (
                _id(service_id, "r")
                and service_id in report_by_id
                and report_by_id[service_id]["service_t"] is not None
            ):
                legality["service_once"] = False
                legality_errors["service_once"].append(
                    f"{scene}.decisions[{t}].service_report_id={service_id!r} already serviced"
                )
            elif _id(service_id, "r") and service_id in queue:
                queue.remove(service_id)
                serviced = report_by_id[service_id]
                serviced["service_t"] = t
            elif _id(service_id, "r"):
                legality["service_available"] = False
                legality_errors["service_available"].append(
                    f"{scene}.decisions[{t}].service_report_id={service_id!r} "
                    "is not in the service-before queue"
                )

        fusion_ids: list[str] = []
        command: dict[str, Any] | None = None
        if serviced is not None:
            robot = serviced["robot"]
            beliefs[robot] = list(serviced["position"])
            latest_served[robot] = serviced
            fusion = [
                latest_served[candidate]
                for candidate in ROBOTS
                if candidate in latest_served and t - latest_served[candidate]["generated_t"] <= 3
            ]
            fusion_ids = [report["id"] for report in fusion]
            weighted = [
                (
                    report["concentration"]
                    * max(0.0, min(1.0, 1.0 - (t - report["generated_t"]) / 3)),
                    report,
                )
                for report in fusion
            ]
            weight_sum = sum(weight for weight, _ in weighted)
            if weight_sum > 0:
                estimate = []
                for coordinate in range(2):
                    numerator = 0.0
                    for weight, report in weighted:
                        norm = math.hypot(report["gradient"][0], report["gradient"][1])
                        z_value = report["position"][coordinate] + (
                            2 * report["gradient"][coordinate] / (norm + 1e-9)
                        )
                        numerator += weight * z_value
                    estimate.append(max(0.0, min(11.0, numerator / weight_sum)))
            target, action = _choose_command(estimate, beliefs)
            command = {
                "id": f"c{len(commands)}",
                "created_t": t,
                "target": target,
                "action": action,
                "fusion_ids": list(fusion_ids),
            }
            commands.append(command)
            age = t - serviced["generated_t"]
            if serviced["concentration"] * max(0.0, min(1.0, 1.0 - age / 3)) > 0:
                freshness_sum += max(0.0, min(1.0, 1.0 - age / 3))

        pre_positions = [list(position) for position in positions]
        active_commands: list[dict[str, Any] | None] = []
        executed: list[str] = []
        for robot in ROBOTS:
            stored = stored_commands[robot]
            if stored is not None and 1 <= t - stored["created_t"] <= 3:
                active_commands.append(stored)
                executed.append(stored["action"])
            else:
                active_commands.append(None)
                executed.append(fallbacks[robot])

        proposals: list[list[int]] = []
        blocked = [False] * len(ROBOTS)
        movers = [False] * len(ROBOTS)
        for robot in ROBOTS:
            dx, dy = DELTA[executed[robot]]
            proposal = [positions[robot][0] + dx, positions[robot][1] + dy]
            if not (0 <= proposal[0] <= GRID_MAX and 0 <= proposal[1] <= GRID_MAX):
                proposal = list(positions[robot])
                blocked[robot] = True
            elif executed[robot] != "stay":
                movers[robot] = True
            proposals.append(proposal)
        collision = [False] * len(ROBOTS)
        for robot in ROBOTS:
            if not movers[robot]:
                continue
            occupied = any(
                proposals[robot] == pre_positions[other] for other in ROBOTS if other != robot
            )
            shared = sum(
                movers[other] and proposals[other] == proposals[robot] for other in ROBOTS
            ) > 1
            collision[robot] = occupied or shared
        for robot in ROBOTS:
            if not blocked[robot] and not collision[robot]:
                positions[robot] = proposals[robot]
            collision_history[robot].append(collision[robot])

        for robot in ROBOTS:
            active = active_commands[robot]
            expected_robot[(t, robot)] = {
                "t": t,
                "robot": robot,
                "position": pre_positions[robot],
                "concentration": _concentration(scene, pre_positions[robot]),
                "gradient": _gradient(scene, pre_positions[robot]),
                "command_id": active["id"] if active else None,
                "command_age": t - active["created_t"] if active else None,
                "executed_action": executed[robot],
                "post_position": list(positions[robot]),
                "blocked": blocked[robot],
                "collision": collision[robot],
            }

        for robot in ROBOTS:
            if not sends[robot]:
                continue
            send_slots.append(t)
            report_id = f"r{len(reports)}"
            position = list(positions[robot])
            distance = abs(position[0] - BASE[0]) + abs(position[1] - BASE[1])
            deterministic_loss = (3 * t + 5 * robot + 7 * scene_index + distance) % 11 in {0, 1}
            delay = 1 + ((t + robot + scene_index + distance) % 3)
            arrival_t = t + delay
            status = (
                "dropped"
                if distance > 8 or deterministic_loss
                else ("late" if arrival_t > HORIZON - 1 else "delivered")
            )
            report = {
                "id": report_id,
                "robot": robot,
                "generated_t": t,
                "position": position,
                "concentration": _concentration(scene, position),
                "gradient": _gradient(scene, position),
                "arrival_t": arrival_t,
                "status": status,
                "service_t": None,
            }
            reports.append(report)
            report_by_id[report_id] = report

        if len(reports) > REPORT_LIMIT:
            legality["report_budget"] = False
            legality_errors["report_budget"].append(
                f"{scene}: generated_reports={len(reports)} exceeds limit={REPORT_LIMIT}"
            )
        window_count = sum(max(0, t - 3) <= slot <= t for slot in send_slots)
        if window_count > 3:
            legality["report_budget"] = False
            legality_errors["report_budget"].append(
                f"{scene}: slots {max(0, t - 3)}..{t} generated {window_count} reports; limit=3"
            )

        support = [
            report
            for report in reports
            if report["service_t"] is not None
            and max(0, t - 4) <= report["generated_t"] <= t
            and report["service_t"] <= t
            and report["service_t"] - report["generated_t"] <= 4
            and abs(report["position"][0] - SCENE_DATA[scene]["source"][0])
            + abs(report["position"][1] - SCENE_DATA[scene]["source"][1])
            <= 2
        ]
        support.sort(key=lambda report: _report_number(report["id"]))
        support_ids = [report["id"] for report in support]
        support_robots = {report["robot"] for report in support}
        close_and_safe = any(
            abs(positions[robot][0] - SCENE_DATA[scene]["source"][0])
            + abs(positions[robot][1] - SCENE_DATA[scene]["source"][1])
            <= 1
            and not any(collision_history[robot][-3:])
            for robot in ROBOTS
        )
        if not confirmed and close_and_safe and len(support_robots) >= 2:
            confirmed = True
            confirm_t = t
        expected_base[t] = {
            "t": t,
            "queue_report_ids": queue_before,
            "fusion_report_ids": fusion_ids,
            "estimate": list(estimate),
            "command_id": command["id"] if command else None,
            "command_target": command["target"] if command else None,
            "command_action": command["action"] if command else None,
            "support_report_ids": support_ids,
            "confirmed": confirmed,
        }

    submitted_robot = records.get((scene, "robot_trace"), [])
    submitted_base = records.get((scene, "base_trace"), [])
    robot_map = {
        (row["t"], row["robot"]): row
        for row in submitted_robot
        if isinstance(row, Mapping)
        and _integer(row.get("t"))
        and _integer(row.get("robot"))
    } if isinstance(submitted_robot, list) else {}
    base_map = {
        row["t"]: row
        for row in submitted_base
        if isinstance(row, Mapping) and _integer(row.get("t"))
    } if isinstance(submitted_base, list) else {}
    errors = {
        name: []
        for name in (
            "robot_state",
            "robot_observation",
            "robot_command",
            "robot_motion",
            "queue",
            "fusion",
            "base_command",
            "confirmation",
        )
    }
    for key, expected in expected_robot.items():
        actual = robot_map.get(key)
        t, robot = key
        if not isinstance(actual, Mapping):
            for group in ("robot_state", "robot_observation", "robot_command", "robot_motion"):
                errors[group].append(f"{scene}.robot_trace[{t},{robot}]: missing")
            continue
        for group, fields in (
            ("robot_state", ("t", "robot", "position")),
            ("robot_observation", ("concentration", "gradient")),
            ("robot_command", ("command_id", "command_age")),
            ("robot_motion", ("executed_action", "post_position", "blocked", "collision")),
        ):
            for field in fields:
                _compare(
                    actual.get(field),
                    expected[field],
                    f"{scene}.robot[{t},{robot}].{field}",
                    errors[group],
                )
    for t, expected in expected_base.items():
        actual = base_map.get(t)
        if not isinstance(actual, Mapping):
            for group in ("queue", "fusion", "base_command", "confirmation"):
                errors[group].append(f"{scene}.base_trace[{t}]: missing")
            continue
        for group, fields in (
            ("queue", ("t", "queue_report_ids")),
            ("fusion", ("fusion_report_ids", "estimate")),
            ("base_command", ("command_id", "command_target", "command_action")),
            ("confirmation", ("support_report_ids", "confirmed")),
        ):
            for field in fields:
                _compare(
                    actual.get(field),
                    expected[field],
                    f"{scene}.base[{t}].{field}",
                    errors[group],
                )
    return {
        "legality": legality,
        "legality_errors": legality_errors,
        "errors": errors,
        "confirmed": confirmed,
        "confirm_t": confirm_t,
        "n_reports": len(reports),
        "freshness": freshness_sum / HORIZON,
    }


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    flags = {
        name: True
        for name in ("action_domain", "report_budget", "service_available", "service_once")
    }
    details = {name: "legal in all scenes" for name in flags}
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("legality", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("legality_errors", {}) if isinstance(result, Mapping) else {}
        for name in flags:
            if not scene_flags.get(name, False):
                flags[name] = False
                errors = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                details[name] = errors[0] if errors else f"{scene}: {name} unavailable"
    return [_atom("decision_legality", name, flags[name], details[name]) for name in flags]


def _replay_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = (
        "robot_state",
        "robot_observation",
        "robot_command",
        "robot_motion",
        "queue",
        "fusion",
        "base_command",
        "confirmation",
    )
    flags = {name: True for name in names}
    details = {name: "fixed-model replay matched" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_errors = result.get("errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            errors = scene_errors.get(name, [f"{scene}: replay unavailable"])
            if errors:
                flags[name] = False
                details[name] = errors[0]
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    primary_1 = sum(bool(replayed[scene]["confirmed"]) for scene in SCENES) / len(SCENES)
    primary_2 = sum(
        max(0.0, 1.0 - replayed[scene]["confirm_t"] / 47)
        if replayed[scene]["confirm_t"] is not None
        else 0.0
        for scene in SCENES
    ) / len(SCENES)
    auxiliary_1 = sum(
        max(0.0, min(1.0, 1.0 - replayed[scene]["n_reports"] / REPORT_LIMIT))
        for scene in SCENES
    ) / len(SCENES)
    auxiliary_2 = sum(replayed[scene]["freshness"] for scene in SCENES) / len(SCENES)
    return {
        "primary_metrics_1": primary_1,
        "primary_metrics_2": primary_2,
        "auxiliary_metrics_1": auxiliary_1,
        "auxiliary_metrics_2": auxiliary_2,
    }


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    visible = [False] * len(lines)
    in_fence = False
    fence_marker = ""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            continue
        visible[index] = not in_fence

    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        if not visible[index]:
            continue
        match = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
        if match:
            headings.append((index, index + 1, match.group(1).strip()))
            continue
        if (
            line.strip()
            and index + 1 < len(lines)
            and visible[index + 1]
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[index + 1])
        ):
            headings.append((index, index + 2, line.strip()))
    result: dict[str, str] = {}
    for order, (_, body_start, title) in enumerate(headings):
        end = headings[order + 1][0] if order + 1 < len(headings) else len(lines)
        body = "\n".join(lines[body_start:end]).strip()
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
        text = ws.read_text("report.md") or ""
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    headings = _headings(text)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    start = time.monotonic()
    state.run_started = True
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        run = await ws.run("python run.py", check=False, timeout=600.0)
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - start
        stderr_text = str(getattr(run, "stderr", "")).lower()
        state.timed_out = (
            state.elapsed_s >= 600.0 - EPS
            or state.exit_code is None
            or "timeout" in stderr_text
            or "timed out" in stderr_text
        )
        state.result_file_is_new_regular = _regular(ws, "results.json")
        state.official_ok = (
            getattr(removed, "exit_code", None) == 0
            and state.exit_code == 0
            and not state.timed_out
            and state.result_file_is_new_regular
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
            state.replayed = {
                scene: _replay_scene(scene, state.records) for scene in SCENES
            }
            legality_atoms = _legality_atoms(state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                legality_atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
            replay_atoms = _replay_atoms(state.replayed)
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
    except Exception as exc:
        state.run_finished = True
        state.elapsed_s = time.monotonic() - start
        state.timed_out = (
            isinstance(exc, TimeoutError)
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or state.elapsed_s >= 600.0 - EPS
        )
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
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular(ws, "run.py"):
        return 0.0, "run.py missing or not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    if getattr(result, "exit_code", None) == 0:
        return 1.0, "static compilation passed"
    return 0.0, "static compilation failed"


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
            return 0.0, "validity gate failed; metric=0"
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
