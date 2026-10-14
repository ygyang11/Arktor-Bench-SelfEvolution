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

HORIZON = 60
DT = 0.5
MAX_SPEED = 0.4
MIN_SEPARATION = 0.75
GOAL_RADIUS = 0.3
PACKET_BUDGET = 120
EPS = 1e-6
DIAGONAL = math.hypot(12.0, 12.0)
SCENES = (
    "opposing_goals",
    "fading_funnel",
    "narrow_bridge",
    "jam_burst",
    "dense_targets",
)
ROBOTS = tuple(range(4))
INITIAL = {
    "opposing_goals": ((2.0, 4.5), (4.0, 4.5), (6.0, 4.5), (8.0, 4.5)),
    "fading_funnel": ((1.5, 2.0), (1.5, 4.0), (1.5, 8.0), (1.5, 10.0)),
    "narrow_bridge": ((2.0, 3.5), (2.0, 5.0), (2.0, 7.0), (2.0, 8.5)),
    "jam_burst": ((2.0, 3.0), (2.0, 5.0), (2.0, 7.0), (2.0, 9.0)),
    "dense_targets": ((4.5, 4.5), (5.5, 4.5), (4.5, 5.5), (5.5, 5.5)),
}
GOALS = {
    "opposing_goals": ((9.0, 7.5), (7.0, 7.5), (5.0, 7.5), (3.0, 7.5)),
    "fading_funnel": ((10.5, 4.0), (10.5, 5.0), (10.5, 7.0), (10.5, 8.0)),
    "narrow_bridge": ((10.0, 8.5), (10.0, 7.0), (10.0, 5.0), (10.0, 3.5)),
    "jam_burst": ((10.0, 3.0), (10.0, 5.0), (10.0, 7.0), (10.0, 9.0)),
    "dense_targets": ((7.5, 7.5), (6.5, 7.5), (7.5, 6.5), (6.5, 6.5)),
}
DEADLINES = {
    "opposing_goals": (52, 48, 48, 52),
    "fading_funnel": (58, 58, 58, 58),
    "narrow_bridge": (60, 58, 58, 60),
    "jam_burst": (56, 56, 56, 56),
    "dense_targets": (36, 34, 34, 36),
}
OBSTACLES = {
    "opposing_goals": ((5.25, 0.0, 5.75, 3.5), (5.25, 8.5, 5.75, 12.0)),
    "fading_funnel": ((4.0, 0.0, 8.0, 3.2), (4.0, 8.8, 8.0, 12.0)),
    "narrow_bridge": ((4.5, 0.0, 7.5, 4.2), (4.5, 7.8, 7.5, 12.0)),
    "jam_burst": ((5.0, 4.2, 7.0, 7.8),),
    "dense_targets": (),
}
REQUIRED = frozenset({(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)})
CONTROL_OPTIONAL = frozenset({(0, 2), (1, 3), (2, 0), (3, 1)})
ALL_EDGES = tuple((i, j) for i in ROBOTS for j in ROBOTS if i != j)
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "velocity", "broadcast", "relax"}
RELAX_KEYS = {"receiver", "duration"}
VALIDITY_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
METRIC_NAMES = (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.25,
        "result_evidence.action_containers": 0.25,
        "result_evidence.velocities_and_broadcasts": 0.25,
        "result_evidence.relaxation_fields": 0.25,
    },
    "decision_legality": {
        "decision_legality.speed": 0.20,
        "decision_legality.local_link_thresholds": 0.20,
        "decision_legality.relaxation_state": 0.20,
        "decision_legality.packet_budget": 0.20,
        "decision_legality.physical_safety": 0.20,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.motion": 1 / 6,
        "replay_validity.local_records_and_messages": 1 / 6,
        "replay_validity.channel": 1 / 6,
        "replay_validity.relaxation_and_recovery": 1 / 6,
        "replay_validity.safety_and_goals": 1 / 6,
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


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _safe_repr(value: Any, limit: int = 180) -> str:
    try:
        rendered = repr(value)
    except (OverflowError, ValueError):
        if _integer(value):
            return f"<integer bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate object member {key!r}")
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


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = parsed and _shape(data, ROOT_KEYS) and isinstance(data["scenarios"], list)
    scenarios = data.get("scenarios") if parsed and isinstance(data, Mapping) else None
    containers_ok = isinstance(scenarios, list) and all(
        _shape(row, SCENE_KEYS)
        and isinstance(row["slots"], list)
        for row in scenarios
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "parse failed"),
        _atom(
            "result_json",
            "root",
            root_ok,
            "exact root key and scenarios list" if root_ok else "expected exact root key scenarios",
        ),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            (
                "scene containers have exact keys and slots arrays"
                if containers_ok
                else "each scene needs exact keys scenario and slots, with slots as an array"
            ),
        ),
    ]


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return []
    return [row for row in data["scenarios"] if isinstance(row, Mapping)]


def _raw_slot_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return entries
    for scene_index, scene in enumerate(data["scenarios"]):
        if not isinstance(scene, Mapping) or not isinstance(scene.get("slots"), list):
            continue
        for slot_index, slot in enumerate(scene["slots"]):
            entries.append((f"scenarios[{scene_index}].slots[{slot_index}]", slot))
    return entries


def _raw_action_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for path, slot in _raw_slot_entries(data):
        if not _shape(slot, SLOT_KEYS) or not isinstance(slot["actions"], list):
            continue
        for action_index, action in enumerate(slot["actions"]):
            entries.append((f"{path}.actions[{action_index}]", action))
    return entries


def _velocity_valid(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_number(item) for item in value)


def _relax_fields_valid(value: Any) -> bool:
    return value is None or (
        _shape(value, RELAX_KEYS) and _integer(value["receiver"]) and _integer(value["duration"])
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    slots = _raw_slot_entries(data)
    slot_issue = next(
        (
            f"{path}: expected exact keys {sorted(SLOT_KEYS)} and an actions list, "
            f"got {_safe_repr(slot)}"
            for path, slot in slots
            if not _shape(slot, SLOT_KEYS) or not isinstance(slot["actions"], list)
        ),
        None,
    )
    actions = _raw_action_entries(data)
    action_issue = next(
        (
            f"{path}: expected exact keys {sorted(ACTION_KEYS)}, got {_safe_repr(action)}"
            for path, action in actions
            if not _shape(action, ACTION_KEYS)
        ),
        None,
    )
    shaped = [(path, action) for path, action in actions if _shape(action, ACTION_KEYS)]
    velocity_issue = next(
        (
            f"{path}: velocity must be a length-two finite-number list and broadcast must be "
            f"Boolean; got velocity={_safe_repr(action['velocity'])}, "
            f"broadcast={_safe_repr(action['broadcast'])}"
            for path, action in shaped
            if not _velocity_valid(action["velocity"]) or not isinstance(action["broadcast"], bool)
        ),
        None,
    )
    relax_issue = next(
        (
            f"{path}.relax: expected null or exact integer receiver/duration object, "
            f"got {_safe_repr(action['relax'])}"
            for path, action in shaped
            if not _relax_fields_valid(action["relax"])
        ),
        None,
    )
    return [
        _atom(
            "result_evidence",
            "slot_containers",
            slot_issue is None,
            "slot containers valid" if slot_issue is None else slot_issue,
        ),
        _atom(
            "result_evidence",
            "action_containers",
            action_issue is None,
            "action containers valid" if action_issue is None else action_issue,
        ),
        _atom(
            "result_evidence",
            "velocities_and_broadcasts",
            velocity_issue is None,
            "velocity and broadcast fields valid" if velocity_issue is None else velocity_issue,
        ),
        _atom(
            "result_evidence",
            "relaxation_fields",
            relax_issue is None,
            "relaxation fields valid" if relax_issue is None else relax_issue,
        ),
    ]


def _key_defects(values: Sequence[Any], expected: set[Any], label: str) -> list[str]:
    counts: dict[Any, int] = {}
    defects: list[str] = []
    for value in values:
        try:
            counts[value] = counts.get(value, 0) + 1
        except TypeError:
            defects.append(f"{label} unhashable key={_safe_repr(value)}")
    for key in sorted(expected, key=str):
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"{label} missing key={key!r}")
        elif count > 1:
            defects.extend(f"{label} duplicate key={key!r}" for _ in range(count - 1))
    for key, count in counts.items():
        if key not in expected:
            defects.extend(f"{label} unknown key={_safe_repr(key)}" for _ in range(count))
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    rows = _scene_rows(data)
    defects = _key_defects([row.get("scenario") for row in rows], set(SCENES), "scenario")
    by_scene: dict[str, list[Mapping[str, Any]]] = {scene: [] for scene in SCENES}
    for row in rows:
        scene = row.get("scenario")
        if isinstance(scene, str) and scene in by_scene:
            by_scene[scene].append(row)
    for scene, matches in by_scene.items():
        if len(matches) != 1 or not isinstance(matches[0].get("slots"), list):
            continue
        slots = matches[0]["slots"]
        slot_values = [
            slot.get("t")
            if isinstance(slot, Mapping) and _integer(slot.get("t"))
            else (
                "invalid_slot_key",
                _safe_repr(slot.get("t") if isinstance(slot, Mapping) else slot),
            )
            for slot in slots
        ]
        defects.extend(_key_defects(slot_values, set(range(HORIZON)), f"{scene} slot"))
        valid_slots: dict[int, list[Mapping[str, Any]]] = {t: [] for t in range(HORIZON)}
        for slot in slots:
            if isinstance(slot, Mapping) and _integer(slot.get("t")) and slot["t"] in valid_slots:
                valid_slots[slot["t"]].append(slot)
        for t, slot_matches in valid_slots.items():
            if len(slot_matches) != 1 or not isinstance(slot_matches[0].get("actions"), list):
                continue
            actions = slot_matches[0]["actions"]
            action_values = [
                action.get("robot")
                if isinstance(action, Mapping) and _integer(action.get("robot"))
                else (
                    "invalid_robot_key",
                    _safe_repr(action.get("robot") if isinstance(action, Mapping) else action),
                )
                for action in actions
            ]
            defects.extend(_key_defects(action_values, set(ROBOTS), f"{scene} slot={t} robot"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene_row in _scene_rows(data):
        scene = scene_row.get("scenario")
        slots = scene_row.get("slots")
        if not isinstance(scene, str) or scene not in SCENES or not isinstance(slots, list):
            continue
        for slot in slots:
            if not isinstance(slot, Mapping) or not _integer(slot.get("t")):
                continue
            t = slot["t"]
            actions = slot.get("actions")
            if not 0 <= t < HORIZON or not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, Mapping) or not _integer(action.get("robot")):
                    continue
                robot = action["robot"]
                if robot not in ROBOTS:
                    continue
                key = (scene, t, robot)
                if key not in records:
                    records[key] = action
    return records


def _raw_action(records: Mapping[Hashable, Any], scene: str, t: int, robot: int) -> dict[str, Any]:
    value = records.get((scene, t, robot))
    if not isinstance(value, Mapping):
        return {"robot": robot, "velocity": [0.0, 0.0], "broadcast": False, "relax": None}
    return {
        "robot": robot,
        "velocity": value.get("velocity"),
        "broadcast": value.get("broadcast"),
        "relax": value.get("relax"),
    }


def _normalize_policy_action(value: Any, robot: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"robot": robot, "velocity": [0.0, 0.0], "broadcast": False, "relax": None}
    velocity = value.get("velocity", [0.0, 0.0])
    if isinstance(velocity, tuple):
        velocity = list(velocity)
    relax = value.get("relax")
    if isinstance(relax, Mapping):
        relax = {"receiver": relax.get("receiver"), "duration": relax.get("duration")}
    return {
        "robot": robot,
        "velocity": velocity,
        "broadcast": value.get("broadcast", False),
        "relax": relax,
    }


def _interference(scene: str, t: int) -> float:
    if scene == "opposing_goals":
        return 0.15 if t % 20 < 10 else 0.25
    if scene == "fading_funnel":
        return 0.20 if t < 18 else 0.65 if t <= 34 else 0.35
    if scene == "narrow_bridge":
        return 0.25 if t < 24 else 0.45 if t <= 40 else 0.30
    if scene == "jam_burst":
        return 0.15 if t < 18 else 0.85 if t <= 25 else 0.55 if t <= 31 else 0.20
    return 0.30 if t % 2 == 0 else 0.40


def _q(scene: str, sender: int, receiver: int, t: int) -> float:
    s = SCENES.index(scene)
    return (
        (37 * (s + 1) + 19 * (sender + 1) + 11 * (receiver + 1) + 23 * (t + 1)) % 101
    ) / 50.0 - 1.0


def _u(scene: str, sender: int, receiver: int, t: int) -> float:
    s = SCENES.index(scene)
    return (
        (29 * (s + 1) + 31 * (sender + 1) + 17 * (receiver + 1) + 43 * (t + 1)) % 997 + 0.5
    ) / 997.0


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _omega_hat(
    endpoint: Sequence[float], neighbor: Sequence[float], m_prev: int, age: int, interference: float
) -> float:
    d_pred = _clip(_distance(endpoint, neighbor) / DIAGONAL)
    return _clip(0.96 + 0.03 * m_prev - 0.30 * d_pred - 0.18 * interference - 0.012 * min(age, 10))


def _omega_true(
    scene: str,
    sender: int,
    receiver: int,
    t: int,
    distance_norm: float,
    m_prev: int,
    interference: float,
) -> float:
    return _clip(
        0.90
        + 0.04 * m_prev
        - 0.34 * distance_norm
        - 0.35 * interference
        + 0.10 * _q(scene, sender, receiver, t)
    )


def _inside_obstacle(point: Sequence[float], rectangle: Sequence[float]) -> bool:
    return (
        rectangle[0] - EPS <= point[0] <= rectangle[2] + EPS
        and rectangle[1] - EPS <= point[1] <= rectangle[3] + EPS
    )


def _local_view(
    robot: int,
    t: int,
    position: Sequence[float],
    records: Mapping[int, Mapping[str, Any]],
    m_prev: Mapping[tuple[int, int], int],
    d_prev: Mapping[tuple[int, int], float],
    active: Mapping[str, Any] | None,
    launch_count: int,
    sent_packets: int,
    interference: float,
    scene: str,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "t": t,
        "robot": robot,
        "position": tuple(position),
        "goal": GOALS[scene][robot],
        "deadline": DEADLINES[scene][robot],
        "obstacles": OBSTACLES[scene],
        "interference": interference,
        "neighbors": {
            neighbor: {
                "position": tuple(records[neighbor]["position"]),
                "generation_boundary": records[neighbor]["generation_boundary"],
                "age": t - records[neighbor]["generation_boundary"],
                "outgoing_monitor": m_prev[(robot, neighbor)],
                "previous_true_distance": d_prev[(robot, neighbor)],
                "required_outgoing": (robot, neighbor) in REQUIRED,
            }
            for neighbor in ROBOTS
            if neighbor != robot
        },
        "active_relaxation": None if active is None else dict(active),
        "relaxation_launches": launch_count,
        "sent_packets": sent_packets,
        "remaining_packet_budget": PACKET_BUDGET - sent_packets,
        "history": tuple(history),
    }


def _simulate_scene(
    scene: str,
    records: Mapping[Hashable, Any] | None = None,
    policies: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    positions = [tuple(point) for point in INITIAL[scene]]
    position_history: list[tuple[tuple[float, float], ...]] = [tuple(positions)]
    local_records: list[dict[int, dict[str, Any]]] = [
        {
            neighbor: {"position": tuple(INITIAL[scene][neighbor]), "generation_boundary": 0}
            for neighbor in ROBOTS
            if neighbor != robot
        }
        for robot in ROBOTS
    ]
    m_prev = {(i, j): 1 for i, j in ALL_EDGES}
    d_prev = {(i, j): _clip(_distance(positions[i], positions[j]) / DIAGONAL) for i, j in ALL_EDGES}
    active_relax: list[dict[str, Any] | None] = [None] * len(ROBOTS)
    launch_counts = [0] * len(ROBOTS)
    pending: dict[int, list[dict[str, Any]]] = {}
    sent_packets = 0
    delivered_copies = 0
    messages: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    open_window: dict[str, Any] | None = None
    window_armed = True
    goal_arrivals: list[int | None] = [
        0 if _distance(INITIAL[scene][robot], GOALS[scene][robot]) <= GOAL_RADIUS + EPS else None
        for robot in ROBOTS
    ]
    unsafe_robots: set[int] = set()
    decisions: list[dict[str, Any]] = []
    executed_velocities: list[tuple[tuple[float, float], ...]] = []
    slot_logs: list[dict[str, Any]] = []
    link_logs: list[dict[str, Any]] = []
    histories: list[list[dict[str, Any]]] = [[] for _ in ROBOTS]
    legality_issues: dict[str, list[str]] = {
        "speed": [],
        "local_link_thresholds": [],
        "relaxation_state": [],
        "packet_budget": [],
        "physical_safety": [],
    }

    for t in range(HORIZON):
        for robot in ROBOTS:
            if active_relax[robot] is not None and active_relax[robot]["end"] <= t:
                active_relax[robot] = None
        for copy in pending.pop(t, []):
            local_records[copy["receiver"]][copy["sender"]] = {
                "position": copy["position"],
                "generation_boundary": copy["generation_boundary"],
            }
        interference = _interference(scene, t)
        views = [
            _local_view(
                robot,
                t,
                positions[robot],
                local_records[robot],
                m_prev,
                d_prev,
                active_relax[robot],
                launch_counts[robot],
                sent_packets,
                interference,
                scene,
                histories[robot],
            )
            for robot in ROBOTS
        ]
        actions: list[dict[str, Any]] = []
        for robot in ROBOTS:
            if policies is not None:
                action = _normalize_policy_action(policies[robot](views[robot]), robot)
            else:
                action = _raw_action(records or {}, scene, t, robot)
            actions.append(action)
            histories[robot].append({"view": views[robot], "decision": dict(action)})
        decisions.append({"t": t, "actions": actions})

        executed: list[tuple[float, float]] = [(0.0, 0.0)] * len(ROBOTS)
        broadcasts = [False] * len(ROBOTS)
        prospective_relax = [active_relax[robot] for robot in ROBOTS]
        valid_action = [True] * len(ROBOTS)
        new_episodes: list[dict[str, Any] | None] = [None] * len(ROBOTS)
        for robot, action in enumerate(actions):
            velocity = action["velocity"]
            relax = action["relax"]
            if not isinstance(action["broadcast"], bool):
                valid_action[robot] = False
            if not _velocity_valid(velocity):
                valid_action[robot] = False
            else:
                speed = math.hypot(float(velocity[0]), float(velocity[1]))
                if speed > MAX_SPEED + EPS:
                    legality_issues["speed"].append(
                        f"scene={scene},t={t},robot={robot},actual_speed={speed:.9f},"
                        f"expected_max={MAX_SPEED},abs_tolerance={EPS}"
                    )
                    valid_action[robot] = False
            if not _relax_fields_valid(relax):
                valid_action[robot] = False
            elif relax is not None:
                receiver = relax["receiver"]
                duration = relax["duration"]
                reason = None
                if receiver not in ROBOTS or receiver == robot:
                    reason = "receiver is not another canonical robot"
                elif (robot, receiver) not in CONTROL_OPTIONAL:
                    reason = "edge is not this robot's optional coordination edge"
                elif duration not in {1, 2, 3}:
                    reason = "duration must be 1, 2, or 3"
                elif active_relax[robot] is not None:
                    reason = "robot already has an active episode"
                elif launch_counts[robot] >= 2:
                    reason = "robot already launched two episodes"
                if reason is not None:
                    legality_issues["relaxation_state"].append(
                        f"scene={scene},t={t},robot={robot},relax={_safe_repr(relax)}: {reason}"
                    )
                    valid_action[robot] = False
                else:
                    episode = {
                        "id": f"RL|{robot}|{receiver}|{t}",
                        "robot": robot,
                        "receiver": receiver,
                        "start": t,
                        "end": t + duration,
                    }
                    prospective_relax[robot] = episode
                    new_episodes[robot] = episode

        for robot, action in enumerate(actions):
            if not valid_action[robot]:
                continue
            velocity = (float(action["velocity"][0]), float(action["velocity"][1]))
            endpoint = (
                positions[robot][0] + DT * velocity[0],
                positions[robot][1] + DT * velocity[1],
            )
            violations: list[str] = []
            for neighbor, record in local_records[robot].items():
                edge = (robot, neighbor)
                if edge not in REQUIRED and edge not in CONTROL_OPTIONAL:
                    continue
                active = prospective_relax[robot]
                threshold = (
                    0.50
                    if active is not None and active["receiver"] == neighbor
                    else 0.50
                    if edge in REQUIRED
                    else 0.66
                )
                age = t - record["generation_boundary"]
                estimate = _omega_hat(
                    endpoint, record["position"], m_prev[(robot, neighbor)], age, interference
                )
                if estimate + EPS < threshold:
                    violations.append(
                        f"edge=({robot},{neighbor}),actual_hat={estimate:.9f},"
                        f"expected_min={threshold:.2f},abs_tolerance={EPS},age={age}"
                    )
            if violations:
                legality_issues["local_link_thresholds"].append(
                    f"scene={scene},t={t},robot={robot}: {';'.join(violations)}"
                )
                valid_action[robot] = False
                prospective_relax[robot] = active_relax[robot]
                new_episodes[robot] = None
                continue
            executed[robot] = velocity
            broadcasts[robot] = bool(action["broadcast"]) and communication_enabled

        for robot in ROBOTS:
            if valid_action[robot] and new_episodes[robot] is not None:
                active_relax[robot] = new_episodes[robot]
                launch_counts[robot] += 1
                episodes.append(new_episodes[robot])
        requested_packets = sum(broadcasts)
        if requested_packets > PACKET_BUDGET - sent_packets:
            legality_issues["packet_budget"].append(
                f"scene={scene},t={t},requested={requested_packets},"
                f"remaining={PACKET_BUDGET - sent_packets}"
            )
            broadcasts = [False] * len(ROBOTS)

        endpoints = [
            (
                positions[robot][0] + DT * executed[robot][0],
                positions[robot][1] + DT * executed[robot][1],
            )
            for robot in ROBOTS
        ]
        for robot, point in enumerate(endpoints):
            if not (-EPS <= point[0] <= 12.0 + EPS and -EPS <= point[1] <= 12.0 + EPS):
                legality_issues["physical_safety"].append(
                    f"scene={scene},t={t},robot={robot},actual_endpoint={point},"
                    f"expected_coordinate_range=[0,12],abs_tolerance={EPS}: outside workspace"
                )
                unsafe_robots.add(robot)
            for obstacle_index, obstacle in enumerate(OBSTACLES[scene]):
                if _inside_obstacle(point, obstacle):
                    legality_issues["physical_safety"].append(
                        f"scene={scene},t={t},robot={robot},actual_endpoint={point},"
                        f"expected_outside_closed_obstacle={obstacle_index},abs_tolerance={EPS}"
                    )
                    unsafe_robots.add(robot)
        for left in ROBOTS:
            for right in range(left + 1, len(ROBOTS)):
                separation = _distance(endpoints[left], endpoints[right])
                if separation + EPS < MIN_SEPARATION:
                    legality_issues["physical_safety"].append(
                        f"scene={scene},t={t},pair=({left},{right}),"
                        f"actual_distance={separation:.9f},expected_min={MIN_SEPARATION},"
                        f"abs_tolerance={EPS}"
                    )
                    unsafe_robots.update({left, right})
        positions = endpoints
        position_history.append(tuple(positions))
        executed_velocities.append(tuple(executed))

        current_m: dict[tuple[int, int], int] = {}
        current_d: dict[tuple[int, int], float] = {}
        states: dict[tuple[int, int], str] = {}
        for sender, receiver in ALL_EDGES:
            distance_norm = _clip(_distance(positions[sender], positions[receiver]) / DIAGONAL)
            omega = _omega_true(
                scene,
                sender,
                receiver,
                t,
                distance_norm,
                m_prev[(sender, receiver)],
                interference,
            )
            event = int(_u(scene, sender, receiver, t) <= omega)
            state = (
                "reliable" if omega >= 0.75 else "vulnerable" if omega >= 0.50 else "disconnected"
            )
            current_m[(sender, receiver)] = event
            current_d[(sender, receiver)] = distance_norm
            states[(sender, receiver)] = state
            link_logs.append(
                {
                    "t": t,
                    "sender": sender,
                    "receiver": receiver,
                    "distance": distance_norm,
                    "m_prev": m_prev[(sender, receiver)],
                    "interference": interference,
                    "q": _q(scene, sender, receiver, t),
                    "u": _u(scene, sender, receiver, t),
                    "omega_true": omega,
                    "monitor": event,
                    "state": state,
                }
            )
        for sender, requested in enumerate(broadcasts):
            if not requested:
                continue
            sent_packets += 1
            for receiver in ROBOTS:
                if receiver == sender:
                    continue
                delivered = bool(current_m[(sender, receiver)])
                message = {
                    "sender": sender,
                    "receiver": receiver,
                    "slot": t,
                    "generation_boundary": t + 1,
                    "delivered": delivered,
                }
                messages.append(message)
                if delivered:
                    delivered_copies += 1
                    pending.setdefault(t + 1, []).append(
                        {
                            **message,
                            "position": tuple(positions[sender]),
                        }
                    )

        required_states = [states[edge] for edge in REQUIRED]
        all_reliable = all(state == "reliable" for state in required_states)
        has_disconnected = any(state == "disconnected" for state in required_states)
        if open_window is not None:
            if all_reliable:
                open_window["recovery"] = t
                open_window["success"] = True
                windows.append(open_window)
                open_window = None
                window_armed = True
            elif t >= open_window["last"]:
                open_window["recovery"] = None
                open_window["success"] = False
                windows.append(open_window)
                open_window = None
                window_armed = not has_disconnected
        elif has_disconnected and window_armed:
            open_window = {
                "id": len(windows),
                "start": t,
                "last": min(t + 4, HORIZON - 1),
                "recovery": None,
                "success": False,
            }
            window_armed = False
            if open_window["last"] == t:
                windows.append(open_window)
                open_window = None
        elif not has_disconnected:
            window_armed = True

        for robot in ROBOTS:
            if (
                goal_arrivals[robot] is None
                and _distance(positions[robot], GOALS[scene][robot]) <= GOAL_RADIUS + EPS
            ):
                goal_arrivals[robot] = t + 1
        slot_logs.append(
            {
                "t": t,
                "positions": tuple(positions),
                "velocities": tuple(executed),
                "broadcasts": tuple(broadcasts),
                "sent_packets": sent_packets,
                "all_required_reliable": all_reliable,
                "has_required_disconnected": has_disconnected,
                "active_relaxations": tuple(
                    None if episode is None else episode["id"] for episode in active_relax
                ),
            }
        )
        m_prev = current_m
        d_prev = current_d

    if open_window is not None:
        open_window["recovery"] = None
        open_window["success"] = False
        windows.append(open_window)
    goals = [
        {
            "robot": robot,
            "arrival": goal_arrivals[robot],
            "deadline": DEADLINES[scene][robot],
            "unsafe": robot in unsafe_robots,
            "success": (
                goal_arrivals[robot] is not None
                and goal_arrivals[robot] <= DEADLINES[scene][robot]
                and robot not in unsafe_robots
            ),
        }
        for robot in ROBOTS
    ]
    return {
        "scene": scene,
        "slots": decisions,
        "consumed_decisions": sum(len(slot["actions"]) for slot in decisions),
        "position_history": position_history,
        "executed_velocities": executed_velocities,
        "slot_logs": slot_logs,
        "link_logs": link_logs,
        "messages": messages,
        "episodes": episodes,
        "windows": windows,
        "goals": goals,
        "sent_packets": sent_packets,
        "delivered_copies": delivered_copies,
        "unsafe_robots": tuple(sorted(unsafe_robots)),
        "legality_issues": legality_issues,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    return _simulate_scene(scene, records=records)


def _online_scene(
    scene: str,
    policies: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]],
    communication_enabled: bool = True,
) -> dict[str, Any]:
    return _simulate_scene(scene, policies=policies, communication_enabled=communication_enabled)


def _result_from_online(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": [{"scenario": scene, "slots": replayed[scene]["slots"]} for scene in SCENES]
    }


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    atoms: list[AtomVerdict] = []
    for name in (
        "speed",
        "local_link_thresholds",
        "relaxation_state",
        "packet_budget",
        "physical_safety",
    ):
        issues = [issue for scene in SCENES for issue in replayed[scene]["legality_issues"][name]]
        atoms.append(
            _atom(
                "decision_legality",
                name,
                not issues,
                "no violations" if not issues else f"violations={issues[:10]}",
            )
        )
    return atoms


def _decision_issue(replay: Mapping[str, Any]) -> str | None:
    if replay.get("consumed_decisions") != HORIZON * len(ROBOTS):
        return (
            f"expected consumed_decisions={HORIZON * len(ROBOTS)}, "
            f"got {replay.get('consumed_decisions')}"
        )
    slots = replay.get("slots")
    if not isinstance(slots, list) or len(slots) != HORIZON:
        return f"expected {HORIZON} slot decisions"
    for t, slot in enumerate(slots):
        if not isinstance(slot, Mapping) or slot.get("t") != t:
            return f"slot index mismatch at t={t}"
        actions = slot.get("actions")
        if not isinstance(actions, list) or [item.get("robot") for item in actions] != list(ROBOTS):
            return f"action sequence mismatch at t={t}"
    return None


def _motion_issue(replay: Mapping[str, Any]) -> str | None:
    history = replay.get("position_history")
    velocities = replay.get("executed_velocities")
    if not isinstance(history, list) or len(history) != HORIZON + 1:
        return "position history must contain 61 boundaries"
    if not isinstance(velocities, list) or len(velocities) != HORIZON:
        return "executed velocity history must contain 60 slots"
    for t, velocity_row in enumerate(velocities):
        if len(history[t]) != len(ROBOTS) or len(history[t + 1]) != len(ROBOTS):
            return f"position row malformed at t={t}"
        for robot in ROBOTS:
            expected = (
                history[t][robot][0] + DT * velocity_row[robot][0],
                history[t][robot][1] + DT * velocity_row[robot][1],
            )
            actual = history[t + 1][robot]
            if _distance(expected, actual) > EPS:
                return (
                    f"motion mismatch at t={t},robot={robot},expected={expected},"
                    f"actual={actual},abs_tolerance={EPS}"
                )
    return None


def _message_issue(replay: Mapping[str, Any]) -> str | None:
    sent = replay.get("sent_packets")
    delivered = replay.get("delivered_copies")
    messages = replay.get("messages")
    if not _integer(sent) or not 0 <= sent <= PACKET_BUDGET:
        return f"sent packet count invalid={sent}"
    if not isinstance(messages, list) or len(messages) != 3 * sent:
        expected = 3 * sent if _integer(sent) else "unknown"
        actual = len(messages) if isinstance(messages, list) else "invalid"
        return f"expected receiver copies={expected}, got={actual}"
    actual_delivered = sum(
        isinstance(message, Mapping) and message.get("delivered") is True for message in messages
    )
    if delivered != actual_delivered:
        return f"delivered copy mismatch expected={actual_delivered},actual={delivered}"
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or message.get("sender") == message.get("receiver")
            or message.get("generation_boundary") != message.get("slot", -2) + 1
            or not isinstance(message.get("delivered"), bool)
        ):
            return f"message malformed={_safe_repr(message)}"
    return None


def _channel_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    links = replay.get("link_logs")
    if not isinstance(links, list) or len(links) != HORIZON * len(ALL_EDGES):
        return f"expected link records={HORIZON * len(ALL_EDGES)}"
    for link in links:
        if not isinstance(link, Mapping):
            return "link log contains non-object"
        t = link.get("t")
        sender = link.get("sender")
        receiver = link.get("receiver")
        if not _integer(t) or not _integer(sender) or not _integer(receiver):
            return f"link key malformed={_safe_repr(link)}"
        expected = _omega_true(
            scene,
            sender,
            receiver,
            t,
            link["distance"],
            link["m_prev"],
            link["interference"],
        )
        actual = link.get("omega_true", -2.0)
        if abs(actual - expected) > EPS:
            return (
                f"omega mismatch at t={t},edge=({sender},{receiver}),expected={expected},"
                f"actual={actual},abs_tolerance={EPS}"
            )
        event = int(_u(scene, sender, receiver, t) <= expected)
        state = (
            "reliable" if expected >= 0.75 else "vulnerable" if expected >= 0.50 else "disconnected"
        )
        if link.get("monitor") != event or link.get("state") != state:
            return f"link outcome mismatch at t={t},edge=({sender},{receiver})"
    return None


def _relaxation_recovery_issue(replay: Mapping[str, Any]) -> str | None:
    episodes = replay.get("episodes")
    windows = replay.get("windows")
    if not isinstance(episodes, list) or not isinstance(windows, list):
        return "episode or recovery-window log missing"
    launches = {robot: 0 for robot in ROBOTS}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            return "relaxation episode malformed"
        robot = episode.get("robot")
        receiver = episode.get("receiver")
        start = episode.get("start")
        end = episode.get("end")
        if (
            robot not in ROBOTS
            or receiver not in ROBOTS
            or robot == receiver
            or (robot, receiver) not in CONTROL_OPTIONAL
            or not _integer(start)
            or not _integer(end)
            or end - start not in {1, 2, 3}
            or episode.get("id") != f"RL|{robot}|{receiver}|{start}"
        ):
            return f"invalid relaxation episode={_safe_repr(episode)}"
        launches[robot] += 1
    if any(count > 2 for count in launches.values()):
        return f"relaxation launch count invalid={launches}"
    previous_start = -1
    for window in windows:
        if not isinstance(window, Mapping):
            return "recovery window malformed"
        start = window.get("start")
        last = window.get("last")
        recovery = window.get("recovery")
        success = window.get("success")
        if (
            not _integer(start)
            or not _integer(last)
            or not 0 <= start <= last < HORIZON
            or last != min(start + 4, HORIZON - 1)
            or start <= previous_start
            or not isinstance(success, bool)
            or (success and (not _integer(recovery) or not start <= recovery <= last))
            or (not success and recovery is not None)
        ):
            return f"invalid recovery window={_safe_repr(window)}"
        previous_start = start
    return None


def _safety_goal_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    goals = replay.get("goals")
    unsafe = replay.get("unsafe_robots")
    if not isinstance(goals, list) or len(goals) != len(ROBOTS) or not isinstance(unsafe, tuple):
        return "goal or safety summary malformed"
    for robot, goal in enumerate(goals):
        if not isinstance(goal, Mapping) or goal.get("robot") != robot:
            return f"goal key malformed for robot={robot}"
        arrival = goal.get("arrival")
        if arrival is not None and (not _integer(arrival) or not 0 <= arrival <= HORIZON):
            return f"goal arrival invalid robot={robot},arrival={arrival}"
        expected_success = (
            arrival is not None and arrival <= DEADLINES[scene][robot] and robot not in unsafe
        )
        if (
            goal.get("deadline") != DEADLINES[scene][robot]
            or goal.get("success") is not expected_success
        ):
            return f"goal success mismatch robot={robot}"
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    del records
    issues: dict[str, list[str]] = {
        "decision_consumption": [],
        "motion": [],
        "local_records_and_messages": [],
        "channel": [],
        "relaxation_and_recovery": [],
        "safety_and_goals": [],
    }
    for scene in SCENES:
        replay = replayed.get(scene)
        if not isinstance(replay, Mapping):
            for values in issues.values():
                values.append(f"{scene}: replay missing")
            continue
        checks = {
            "decision_consumption": _decision_issue(replay),
            "motion": _motion_issue(replay),
            "local_records_and_messages": _message_issue(replay),
            "channel": _channel_issue(scene, replay),
            "relaxation_and_recovery": _relaxation_recovery_issue(replay),
            "safety_and_goals": _safety_goal_issue(scene, replay),
        }
        for name, issue in checks.items():
            if issue is not None:
                issues[name].append(f"{scene}: {issue}")
    return [
        _atom(
            "replay_validity",
            name,
            not values,
            "replay invariant passed" if not values else f"issues={values[:8]}",
        )
        for name, values in issues.items()
    ]


def _scene_metrics(replay: Mapping[str, Any]) -> dict[str, float]:
    goal_quality = sum(
        goal["arrival"] is not None and goal["arrival"] <= goal["deadline"]
        for goal in replay["goals"]
    ) / len(ROBOTS)
    reliable_slots = sum(log["all_required_reliable"] for log in replay["slot_logs"])
    windows = replay["windows"]
    recovery_fraction = (
        sum(window["success"] for window in windows) / len(windows)
        if windows
        else reliable_slots / HORIZON
    )
    communication_quality = 0.6 * reliable_slots / HORIZON + 0.4 * recovery_fraction
    sent = replay["sent_packets"]
    delivery = replay["delivered_copies"] / (3 * sent) if sent else 0.0
    initial = replay["position_history"][0]
    final = replay["position_history"][-1]
    progress = sum(
        max(
            0.0,
            _distance(initial[robot], GOALS[replay["scene"]][robot])
            - _distance(final[robot], GOALS[replay["scene"]][robot]),
        )
        for robot in ROBOTS
    )
    progress_norm = _clip(progress / (len(ROBOTS) * HORIZON * DT * MAX_SPEED))
    smooth_samples: list[float] = []
    previous = [(0.0, 0.0)] * len(ROBOTS)
    for row in replay["executed_velocities"]:
        for robot in ROBOTS:
            smooth_samples.append(_clip(1.0 - _distance(row[robot], previous[robot]) / 0.8))
        previous = list(row)
    smoothness = sum(smooth_samples) / len(smooth_samples)
    packet_efficiency = _clip(1.0 - sent / PACKET_BUDGET)
    action_efficiency = progress_norm * (0.6 * smoothness + 0.4 * packet_efficiency)
    return {
        "primary_metrics_1": goal_quality,
        "primary_metrics_2": communication_quality,
        "auxiliary_metrics_1": delivery,
        "auxiliary_metrics_2": action_efficiency,
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values = {
        name: sum(_scene_metrics(replayed[scene])[name] for scene in SCENES) / len(SCENES)
        for name in METRIC_NAMES
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"metric out of range: {values}")
    return values


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8.0 * metrics["primary_metrics_1"]
        + 7.0 * metrics["primary_metrics_2"]
        + 3.0 * metrics["auxiliary_metrics_1"]
        + 2.0 * metrics["auxiliary_metrics_2"]
    ) / 20.0


def _gated_metric(
    scores: Mapping[str, float],
    official_ok: bool,
    metrics: Mapping[str, float] | None,
    name: str | None,
) -> tuple[float, str]:
    failed: list[str] = []
    if not official_ok:
        failed.append("official_run")
    failed.extend(
        f"{check_id}={scores.get(check_id, 0.0):.6f}"
        for check_id in VALIDITY_CHECKS
        if scores.get(check_id, 0.0) < 1.0 - EPS
    )
    if failed:
        return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
    if name is None:
        return 1.0, "validity gate passed"
    if name not in METRIC_NAMES:
        raise ValueError(f"unknown metric name {name!r}")
    if metrics is None or name not in metrics:
        return 0.0, f"validity gate passed but {name} is unavailable; metric=0"
    value = metrics[name]
    if not _number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"invalid gated metric {name}={value!r}")
    return float(value), "validity gate passed"


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
            stripped = line.lstrip()
            if stripped.startswith(fence_marker):
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
    try:
        if not _regular(ws, "report.md"):
            return 0.0, "report.md missing or not a regular file"
        report = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(report)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


def _fail_run_checks(state: EvalState, detail: str) -> EvalState:
    state.official_ok = False
    state.official_message = detail
    for name in VALIDITY_CHECKS:
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
        started = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - started
            state.timed_out = True
            return _fail_run_checks(
                state, f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - started
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; elapsed={state.elapsed_s:.3f}s",
            )
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - started
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
        if not state.official_ok:
            return _fail_run_checks(state, state.official_message)
        try:
            state.data = _strict_json(ws.read_text("results.json") or "")
            state.parse_ok = True
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.parse_ok:
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
            gate_score, _ = _gated_metric(state.scores, state.official_ok, None, None)
            state.validity_gate = gate_score >= 1.0 - EPS
            if state.validity_gate:
                state.metrics = _metrics(state.replayed)
        else:
            for name in VALIDITY_CHECKS[1:]:
                state.scores[name] = 0.0
                state.diagnostics[name] = "strict JSON parse failed"
    except Exception as exc:  # noqa: BLE001
        detail = f"grader failure: {type(exc).__name__}: {exc}"
        state.official_message = (
            f"{state.official_message}; {detail}" if state.official_message else detail
        )
        for name in VALIDITY_CHECKS:
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
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
    try:
        if not _regular(ws, "run.py"):
            return 0.0, "run.py missing or not a regular file"
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except TimeoutError:
        return 0.0, "static compilation timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"static compilation raised {type(exc).__name__}: {exc}"
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
    try:
        return _report_score(ws)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report contract check raised {type(exc).__name__}: {exc}"


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in VALIDITY_CHECKS:

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in METRIC_NAMES:

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        return _gated_metric(state.scores, state.official_ok, state.metrics, _id)

    globals()[_name] = check(id=_name)(_metric)
del _metric
