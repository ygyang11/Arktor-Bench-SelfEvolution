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
    "opposing_goals",
    "four_way_crossing",
    "narrow_merge",
    "offset_swap",
    "dense_ring",
)
ROBOTS = tuple(range(4))
HORIZON = 60
REGION_MAX = 6.0
MAX_ACTION = 0.4
GOAL_RADIUS = 0.25
COLLISION_DISTANCE = 0.4
SAFE_DISTANCE = 0.8
COMM_RADIUS = 1.8
BROADCAST_BUDGET = 48
ROLLING_LIMIT = 3
EPS = 1e-6

STARTS = {
    "opposing_goals": ((0.8, 2.0), (5.2, 2.0), (0.8, 4.0), (5.2, 4.0)),
    "four_way_crossing": ((3.0, 0.6), (5.4, 3.0), (3.0, 5.4), (0.6, 3.0)),
    "narrow_merge": ((0.6, 2.0), (0.6, 4.0), (5.4, 2.5), (5.4, 3.5)),
    "offset_swap": ((0.8, 1.0), (5.2, 1.8), (0.8, 4.2), (5.2, 5.0)),
    "dense_ring": ((3.0, 1.9), (4.1, 3.0), (3.0, 4.1), (1.9, 3.0)),
}
GOALS = {
    "opposing_goals": ((5.2, 2.0), (0.8, 2.0), (5.2, 4.0), (0.8, 4.0)),
    "four_way_crossing": ((3.0, 5.4), (0.6, 3.0), (3.0, 0.6), (5.4, 3.0)),
    "narrow_merge": ((5.4, 2.0), (5.4, 4.0), (0.6, 2.5), (0.6, 3.5)),
    "offset_swap": ((5.2, 1.8), (0.8, 1.0), (5.2, 5.0), (0.8, 4.2)),
    "dense_ring": ((3.0, 4.1), (1.9, 3.0), (3.0, 1.9), (4.1, 3.0)),
}

SCENE_KEYS = {"scenario", "frames"}
FRAME_KEYS = {"t", "robots"}
ROBOT_KEYS = {"robot", "action", "broadcast"}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.frame_containers": 0.25,
        "result_evidence.robot_containers": 0.25,
        "result_evidence.action_fields": 0.25,
        "result_evidence.broadcast_fields": 0.25,
    },
    "decision_legality": {
        "decision_legality.action_norm": 0.20,
        "decision_legality.stationary_actions": 0.20,
        "decision_legality.inactive_broadcasts": 0.20,
        "decision_legality.rolling_broadcasts": 0.20,
        "decision_legality.broadcast_budget": 0.20,
    },
    "replay_validity": {
        "replay_validity.action_consumption": 0.16666666666666666,
        "replay_validity.motion_update": 0.16666666666666666,
        "replay_validity.message_delivery": 0.16666666666666666,
        "replay_validity.neighbor_histories": 0.16666666666666666,
        "replay_validity.collision_state": 0.16666666666666666,
        "replay_validity.goal_state": 0.16666666666666666,
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


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _vector(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite(item) for item in value)


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _record_shape(value: Any, keys: set[str], canonical_key: str) -> bool:
    return isinstance(value, Mapping) and set(value) - {canonical_key} == keys - {
        canonical_key
    }


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
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = isinstance(scenes, list) and all(
        _record_shape(row, SCENE_KEYS, "scenario")
        and isinstance(row.get("frames"), list)
        for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires frames and no unknown non-canonical keys",
        ),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = ("frame_containers", "robot_containers", "action_fields", "broadcast_fields")
    flags = {name: True for name in names}
    details = {name: "all submitted decision fields satisfy their local contract" for name in names}
    rows = _rows(data)
    if not rows:
        for name in names:
            flags[name] = False
            details[name] = "scenarios: no decision records"
    for scene_index, scene_row in enumerate(rows):
        if not isinstance(scene_row, Mapping) or "frames" not in scene_row:
            continue
        frames = scene_row.get("frames")
        if not isinstance(frames, list):
            continue
        for frame_index, frame in enumerate(frames):
            frame_ok = (
                isinstance(frame, Mapping)
                and set(frame) <= FRAME_KEYS
                and isinstance(frame.get("robots"), list)
            )
            if not frame_ok and flags["frame_containers"]:
                details["frame_containers"] = (
                    f"scenarios[{scene_index}].frames[{frame_index}]: "
                    "invalid keys or robots container"
                )
            flags["frame_containers"] &= frame_ok
            if not isinstance(frame, Mapping) or not isinstance(frame.get("robots"), list):
                continue
            for robot_index, robot in enumerate(frame["robots"]):
                robot_ok = isinstance(robot, Mapping) and set(robot) <= ROBOT_KEYS
                if not robot_ok and flags["robot_containers"]:
                    details["robot_containers"] = (
                        f"scenarios[{scene_index}].frames[{frame_index}]."
                        f"robots[{robot_index}]: invalid container or unknown keys"
                    )
                flags["robot_containers"] &= robot_ok
                if not isinstance(robot, Mapping):
                    continue
                action_ok = _vector(robot.get("action"))
                if not action_ok and flags["action_fields"]:
                    details["action_fields"] = (
                        f"scenarios[{scene_index}].frames[{frame_index}]."
                        f"robots[{robot_index}].action: expected two finite numbers"
                    )
                flags["action_fields"] &= action_ok
                broadcast_ok = isinstance(robot.get("broadcast"), bool)
                if not broadcast_ok and flags["broadcast_fields"]:
                    details["broadcast_fields"] = (
                        f"scenarios[{scene_index}].frames[{frame_index}]."
                        f"robots[{robot_index}].broadcast: expected Boolean"
                    )
                flags["broadcast_fields"] &= broadcast_ok
    return [_atom("result_evidence", name, flags[name], details[name]) for name in names]


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
    if (
        len(actual) == len(expected)
        and set(actual) == expected_set
        and list(actual) != list(expected)
    ):
        defects.append(f"{label}: canonical order")
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    structurally_unavailable: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            if index < len(SCENES):
                scene_keys.append(SCENES[index])
                structurally_unavailable.add(SCENES[index])
            continue
        name = row.get("scenario") if isinstance(row, Mapping) else None
        scene_keys.append(name if isinstance(name, str) else ("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))
    for scene in SCENES:
        if scene in structurally_unavailable:
            continue
        row = next(
            (item for item in rows if isinstance(item, Mapping) and item.get("scenario") == scene),
            None,
        )
        frames = row.get("frames") if isinstance(row, Mapping) else None
        if isinstance(row, Mapping) and not isinstance(frames, list):
            continue
        if not isinstance(frames, list):
            defects.append(f"{scene}.frames: expected list")
            continue
        frame_keys: list[Hashable] = []
        robot_keys: list[Hashable] = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                if frame_index < HORIZON:
                    frame_keys.append(frame_index)
                    robot_keys.extend((frame_index, robot) for robot in ROBOTS)
                continue
            if isinstance(frame, Mapping) and _integer(frame.get("t")):
                frame_key: Hashable = frame["t"]
                frame_keys.append(frame_key)
                robots = frame.get("robots")
                if isinstance(robots, list):
                    for robot_index, robot in enumerate(robots):
                        if isinstance(robot, Mapping) and _integer(robot.get("robot")):
                            robot_keys.append((frame_key, robot["robot"]))
                        elif not isinstance(robot, Mapping) and robot_index < len(ROBOTS):
                            robot_keys.append((frame_key, ROBOTS[robot_index]))
                        else:
                            robot_keys.append(("invalid_robot", frame_index, robot_index))
                else:
                    robot_keys.extend((frame_key, robot) for robot in ROBOTS)
            else:
                frame_keys.append(("invalid_frame", frame_index))
        defects.extend(_canonical_defects(frame_keys, list(range(HORIZON)), f"{scene}.frames"))
        expected_robots = [(t, robot) for t in range(HORIZON) for robot in ROBOTS]
        defects.extend(_canonical_defects(robot_keys, expected_robots, f"{scene}.robots"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for row in _rows(data):
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            continue
        scene = row["scenario"]
        frames = row.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if (
                isinstance(frame, Mapping)
                and _integer(frame.get("t"))
                and (scene, frame["t"]) not in records
            ):
                records[(scene, frame["t"])] = frame
    return records


def _raw_robot_action(frame: Any, robot: int) -> Mapping[str, Any]:
    if not isinstance(frame, Mapping) or not isinstance(frame.get("robots"), list):
        return {}
    for row in frame["robots"]:
        if isinstance(row, Mapping) and row.get("robot") == robot:
            return row
    return {}


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _nominal(
    position: tuple[float, float],
    goal: tuple[float, float],
    stationary: bool,
) -> tuple[float, float]:
    if stationary:
        return (0.0, 0.0)
    delta = (goal[0] - position[0], goal[1] - position[1])
    distance = _norm(delta)
    if distance <= EPS:
        return (0.0, 0.0)
    scale = min(MAX_ACTION, distance) / distance
    return (delta[0] * scale, delta[1] * scale)


def _clip_action(vector: tuple[float, float]) -> tuple[float, float]:
    length = _norm(vector)
    if length <= MAX_ACTION or length <= EPS:
        return vector
    scale = MAX_ACTION / length
    return (vector[0] * scale, vector[1] * scale)


def _clip_endpoint(
    position: tuple[float, float], vector: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    endpoint = (
        min(REGION_MAX, max(0.0, position[0] + vector[0])),
        min(REGION_MAX, max(0.0, position[1] + vector[1])),
    )
    return endpoint, (endpoint[0] - position[0], endpoint[1] - position[1])


def _public_message(message: Mapping[str, Any], t: int) -> dict[str, Any]:
    return {
        "sender": message["sender"],
        "generation_t": message["generation_t"],
        "position": list(message["position"]),
        "previous_action": list(message["previous_action"]),
        "age": t - int(message["generation_t"]),
    }


def _simulate_scene(
    scene: str,
    before_source: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    scene_index = SCENES.index(scene)
    goals = GOALS[scene]
    positions = [tuple(value) for value in STARTS[scene]]
    previous_actions = [(0.0, 0.0) for _ in ROBOTS]
    collided = [False for _ in ROBOTS]
    completed = [False for _ in ROBOTS]
    collided_at: list[int | None] = [None for _ in ROBOTS]
    completed_at: list[int | None] = [None for _ in ROBOTS]
    latest_messages: list[dict[int, dict[str, Any]]] = [dict() for _ in ROBOTS]
    message_arrivals: dict[int, list[dict[str, Any]]] = {}
    broadcast_times: list[list[int]] = [[] for _ in ROBOTS]
    broadcast_count = 0
    modification_sums = [0.0 for _ in ROBOTS]
    freshness_sum = 0.0
    alert_count = 0
    frames: list[dict[str, Any]] = []

    legality_names = (
        "action_norm",
        "stationary_actions",
        "inactive_broadcasts",
        "rolling_broadcasts",
        "broadcast_budget",
    )
    legality = {name: True for name in legality_names}
    legality_errors: dict[str, list[str]] = {name: [] for name in legality_names}
    replay_names = (
        "action_consumption",
        "motion_update",
        "message_delivery",
        "neighbor_histories",
        "collision_state",
        "goal_state",
    )
    replay = {name: True for name in replay_names}
    replay_errors: dict[str, list[str]] = {name: [] for name in replay_names}
    consumed_actions: dict[str, dict[tuple[int, int], Any]] = {
        "action": {},
        "broadcast": {},
    }

    for t in range(HORIZON):
        for message in message_arrivals.get(t, []):
            generation_t = int(message["generation_t"])
            receiver = int(message["receiver"])
            sender = int(message["sender"])
            if not (0 <= generation_t < t and sender != receiver):
                replay["message_delivery"] = False
                replay_errors["message_delivery"].append(
                    f"{scene}.t={t}: noncausal message sender={sender}, receiver={receiver}"
                )
                continue
            previous = latest_messages[receiver].get(sender)
            if previous is None or generation_t > int(previous["generation_t"]):
                latest_messages[receiver][sender] = message

        for receiver in ROBOTS:
            for sender in ROBOTS:
                if receiver == sender:
                    continue
                message = latest_messages[receiver].get(sender)
                if message is not None:
                    age = t - int(message["generation_t"])
                    freshness_sum += max(0.0, 1.0 - age / 3.0)

        raw_frame = before_source({"t": t})
        pre_positions = list(positions)
        nominals = [
            _nominal(positions[robot], goals[robot], collided[robot] or completed[robot])
            for robot in ROBOTS
        ]
        requested_actions: list[tuple[float, float]] = []
        processed_broadcasts: list[bool] = []
        raw_actions: list[Any] = []
        earlier_broadcasts = 0

        for robot in ROBOTS:
            previous_three = sum(
                max(0, t - 3) <= sent_t < t for sent_t in broadcast_times[robot]
            )
            messages = [
                _public_message(latest_messages[robot][sender], t)
                for sender in ROBOTS
                if sender != robot and sender in latest_messages[robot]
            ]
            view = {
                "t": t,
                "robot": robot,
                "start": list(STARTS[scene][robot]),
                "goal": list(goals[robot]),
                "position": list(positions[robot]),
                "previous_action": list(previous_actions[robot]),
                "nominal_action": list(nominals[robot]),
                "collided": collided[robot],
                "completed": completed[robot],
                "latest_messages": messages,
                "remaining_broadcasts": max(0, BROADCAST_BUDGET - broadcast_count),
                "broadcasts_in_previous_three_slots": previous_three,
                "earlier_current_broadcasts": earlier_broadcasts,
            }
            supplied = (
                before_source(view) if raw_frame is None else _raw_robot_action(raw_frame, robot)
            )
            raw_actions.append(supplied)
            action = supplied if isinstance(supplied, Mapping) else {}
            raw_vector = action.get("action")
            raw_broadcast = action.get("broadcast")
            consumed_actions["action"][(t, robot)] = raw_vector
            consumed_actions["broadcast"][(t, robot)] = raw_broadcast

            vector_ok = _vector(raw_vector)
            vector = (
                (float(raw_vector[0]), float(raw_vector[1])) if vector_ok else (0.0, 0.0)
            )
            if vector_ok and _norm(vector) > MAX_ACTION + EPS:
                legality["action_norm"] = False
                legality_errors["action_norm"].append(
                    f"{scene}.frames[{t}].robots[{robot}].action norm={_norm(vector):.9f} "
                    f"exceeds {MAX_ACTION}"
                )
            if vector_ok and (collided[robot] or completed[robot]) and _norm(vector) > EPS:
                legality["stationary_actions"] = False
                legality_errors["stationary_actions"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: nonzero action after stationary state"
                )
            vector = _clip_action(vector)
            if collided[robot] or completed[robot]:
                vector = (0.0, 0.0)
            requested_actions.append(vector)

            broadcast_ok = isinstance(raw_broadcast, bool)
            broadcast = bool(raw_broadcast) if broadcast_ok else False
            if broadcast_ok and collided[robot] and broadcast:
                legality["inactive_broadcasts"] = False
                legality_errors["inactive_broadcasts"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: broadcast after collision"
                )
                broadcast = False
            if broadcast:
                broadcast_times[robot].append(t)
                broadcast_count += 1
                earlier_broadcasts += 1
                if previous_three + 1 > ROLLING_LIMIT:
                    legality["rolling_broadcasts"] = False
                    legality_errors["rolling_broadcasts"].append(
                        f"{scene}.frames[{t}].robots[{robot}]: "
                        f"four-slot broadcasts={previous_three + 1}"
                    )
                if broadcast_count > BROADCAST_BUDGET:
                    legality["broadcast_budget"] = False
                    legality_errors["broadcast_budget"].append(
                        f"{scene}.frames[{t}]: broadcasts={broadcast_count} "
                        f"exceeds {BROADCAST_BUDGET}"
                    )
            processed_broadcasts.append(broadcast)

        for sender in ROBOTS:
            if not processed_broadcasts[sender]:
                continue
            for receiver in ROBOTS:
                separation = _distance(pre_positions[sender], pre_positions[receiver])
                if sender == receiver or separation > COMM_RADIUS + EPS:
                    continue
                lost = (5 * t + 2 * sender + 3 * receiver + scene_index) % 13 in {0, 1}
                if lost:
                    continue
                delay = 1 + ((t + sender + receiver + scene_index) % 3)
                arrival_t = t + delay
                if arrival_t < HORIZON:
                    message_arrivals.setdefault(arrival_t, []).append(
                        {
                            "sender": sender,
                            "receiver": receiver,
                            "generation_t": t,
                            "position": pre_positions[sender],
                            "previous_action": previous_actions[sender],
                            "arrival_t": arrival_t,
                        }
                    )

        endpoints: list[tuple[float, float]] = []
        executed_actions: list[tuple[float, float]] = []
        for robot in ROBOTS:
            endpoint, executed = _clip_endpoint(pre_positions[robot], requested_actions[robot])
            endpoints.append(endpoint)
            executed_actions.append(executed)
            if not (
                -EPS <= endpoint[0] <= REGION_MAX + EPS
                and -EPS <= endpoint[1] <= REGION_MAX + EPS
                and _norm(executed) <= MAX_ACTION + EPS
            ):
                replay["motion_update"] = False
                replay_errors["motion_update"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: invalid executed motion"
                )
            difference = _distance(executed, nominals[robot])
            modification_sums[robot] += max(0.0, min(1.0, 1.0 - difference / 0.8))

        newly_collided: set[int] = set()
        for left in ROBOTS:
            for right in ROBOTS:
                if left >= right:
                    continue
                pair_distance = _distance(endpoints[left], endpoints[right])
                if pair_distance < COLLISION_DISTANCE - EPS:
                    newly_collided.update((left, right))
                elif pair_distance < SAFE_DISTANCE - EPS:
                    alert_count += 1

        old_collided = list(collided)
        positions = endpoints
        previous_actions = executed_actions
        for robot in ROBOTS:
            collided[robot] = collided[robot] or robot in newly_collided
            if not old_collided[robot] and collided[robot]:
                collided_at[robot] = t
            if old_collided[robot] and not collided[robot]:
                replay["collision_state"] = False
                replay_errors["collision_state"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: collision state cleared"
                )
            goal_distance = _distance(positions[robot], goals[robot])
            old_completed = completed[robot]
            if not collided[robot] and goal_distance <= GOAL_RADIUS + EPS:
                completed[robot] = True
            if not old_completed and completed[robot]:
                completed_at[robot] = t
            if completed[robot] and goal_distance > GOAL_RADIUS + EPS:
                replay["goal_state"] = False
                replay_errors["goal_state"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: completed robot left goal"
                )

        frames.append(
            {
                "t": t,
                "robots": [
                    {
                        "robot": robot,
                        "action": (
                            raw_actions[robot].get("action")
                            if isinstance(raw_actions[robot], Mapping)
                            else None
                        ),
                        "broadcast": (
                            raw_actions[robot].get("broadcast")
                            if isinstance(raw_actions[robot], Mapping)
                            else None
                        ),
                    }
                    for robot in ROBOTS
                ],
            }
        )

    for arrival_t, messages in message_arrivals.items():
        for message in messages:
            if int(message["arrival_t"]) != arrival_t or int(message["generation_t"]) >= arrival_t:
                replay["message_delivery"] = False
                replay_errors["message_delivery"].append(
                    f"{scene}: inconsistent queued message at arrival={arrival_t}"
                )
    for receiver in ROBOTS:
        for sender, message in latest_messages[receiver].items():
            if sender == receiver or int(message["generation_t"]) >= HORIZON:
                replay["neighbor_histories"] = False
                replay_errors["neighbor_histories"].append(
                    f"{scene}: invalid retained history receiver={receiver}, sender={sender}"
                )

    qualities = [
        0.0
        if collided[robot]
        else 0.6 * float(completed[robot]) + 0.4 * modification_sums[robot] / HORIZON
        for robot in ROBOTS
    ]
    return {
        "scenario_record": {"scenario": scene, "frames": frames},
        "legality": legality,
        "legality_errors": legality_errors,
        "replay": replay,
        "replay_errors": replay_errors,
        "decision_stream_complete": True,
        "consumed_actions": consumed_actions,
        "collision_free": [not value for value in collided],
        "completed": completed,
        "collided_at": collided_at,
        "completed_at": completed_at,
        "qualities": qualities,
        "broadcast_count": broadcast_count,
        "freshness": freshness_sum / (HORIZON * len(ROBOTS) * (len(ROBOTS) - 1)),
        "alert_count": alert_count,
        "positions": positions,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    def before(view: Mapping[str, Any]) -> Any:
        return records.get((scene, int(view["t"])), {})

    result = _simulate_scene(scene, before)
    missing = [t for t in range(HORIZON) if (scene, t) not in records]
    incomplete = []
    for t in range(HORIZON):
        frame = records.get((scene, t))
        robots = frame.get("robots") if isinstance(frame, Mapping) else None
        robot_ids = (
            [row.get("robot") for row in robots if isinstance(row, Mapping)]
            if isinstance(robots, list)
            else []
        )
        if len(robot_ids) != len(ROBOTS) or robot_ids != list(ROBOTS):
            incomplete.append(t)
    if missing or incomplete:
        result["decision_stream_complete"] = False
        result["first_incomplete_t"] = min(missing + incomplete)
    else:
        result["first_incomplete_t"] = None
    return result


def _online_scene(scene: str, policy: Any) -> dict[str, Any]:
    if hasattr(policy, "reset"):
        policy.reset()

    def before(view: Mapping[str, Any]) -> Any:
        if set(view) == {"t"}:
            return None
        return policy.decide_robot(view)

    return _simulate_scene(scene, before)["scenario_record"]


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = (
        "action_norm",
        "stationary_actions",
        "inactive_broadcasts",
        "rolling_broadcasts",
        "broadcast_budget",
    )
    flags = {name: True for name in names}
    details = {name: "legal in all scenes" for name in names}

    def attributable_state_failure(
        name: str, detail: str, result: Mapping[str, Any]
    ) -> bool:
        first_incomplete = result.get("first_incomplete_t")
        if not _integer(first_incomplete):
            return True
        slot_match = re.search(r"\.frames\[(\d+)\]", detail)
        robot_match = re.search(r"\.robots\[(\d+)\]", detail)
        if slot_match is None or robot_match is None:
            return False
        slot = int(slot_match.group(1))
        robot = int(robot_match.group(1))
        if slot < first_incomplete:
            return True
        if robot not in ROBOTS:
            return False
        collided_at = result.get("collided_at")
        completed_at = result.get("completed_at")
        collision_start = (
            collided_at[robot]
            if isinstance(collided_at, list) and len(collided_at) == len(ROBOTS)
            else None
        )
        completion_start = (
            completed_at[robot]
            if isinstance(completed_at, list) and len(completed_at) == len(ROBOTS)
            else None
        )
        if name == "inactive_broadcasts":
            return _integer(collision_start) and collision_start < first_incomplete
        starts = [value for value in (collision_start, completion_start) if _integer(value)]
        return bool(starts) and min(starts) < first_incomplete

    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("legality", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("legality_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if not scene_flags.get(name, False):
                failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                if name in {"stationary_actions", "inactive_broadcasts"} and isinstance(
                    result, Mapping
                ):
                    failures = [
                        detail
                        for detail in failures
                        if attributable_state_failure(name, detail, result)
                    ]
                    if not failures:
                        continue
                flags[name] = False
                details[name] = failures[0] if failures else f"{scene}: {name} unavailable"
    return [_atom("decision_legality", name, flags[name], details[name]) for name in names]


def _consumption_mismatch(
    scene: str,
    records: Mapping[Hashable, Any],
    result: Mapping[str, Any],
    field: str,
) -> str | None:
    consumed = result.get("consumed_actions")
    values = consumed.get(field) if isinstance(consumed, Mapping) else None
    if not isinstance(values, Mapping):
        return f"{scene}: consumed {field} audit unavailable"
    for t in range(HORIZON):
        frame = records.get((scene, t))
        for robot in ROBOTS:
            action = _raw_robot_action(frame, robot)
            expected = action.get(field)
            key = (t, robot)
            if key not in values:
                return f"{scene}.frames[{t}].robots[{robot}].{field}: consumption unavailable"
            actual = values[key]
            if expected != actual:
                return (
                    f"{scene}.frames[{t}].robots[{robot}].{field}: "
                    f"submitted={expected!r}, consumed={actual!r}"
                )
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Any]
) -> list[AtomVerdict]:
    names = (
        "action_consumption",
        "motion_update",
        "message_delivery",
        "neighbor_histories",
        "collision_state",
        "goal_state",
    )
    flags = {name: True for name in names}
    details = {name: "fixed-model replay completed" for name in names}
    field_owners = {
        "action_consumption": "action",
        "message_delivery": "broadcast",
    }
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("replay", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("replay_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if not scene_flags.get(name, False):
                flags[name] = False
                failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                details[name] = failures[0] if failures else f"{scene}: replay unavailable"
        if isinstance(result, Mapping):
            for name, field in field_owners.items():
                mismatch = _consumption_mismatch(scene, records, result, field)
                if mismatch is not None:
                    flags[name] = False
                    details[name] = mismatch
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    primary_1 = sum(
        sum(float(value) for value in replayed[scene]["collision_free"]) for scene in SCENES
    ) / (len(SCENES) * len(ROBOTS))
    primary_2 = sum(
        sum(float(value) for value in replayed[scene]["qualities"]) for scene in SCENES
    ) / (len(SCENES) * len(ROBOTS))
    auxiliary_1 = sum(
        max(
            0.0,
            min(1.0, 1.0 - float(replayed[scene]["broadcast_count"]) / BROADCAST_BUDGET),
        )
        for scene in SCENES
    ) / len(SCENES)
    auxiliary_2 = sum(float(replayed[scene]["freshness"]) for scene in SCENES) / len(SCENES)
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
                state,
                f"pre-run results cleanup raised {type(exc).__name__}: {exc}",
            )
        state.run_started = True
        start = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - start
            state.timed_out = True
            return _fail_run_checks(
                state,
                f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s",
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - start
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; "
                f"elapsed={state.elapsed_s:.3f}s",
            )
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
        state.result_file_is_new_regular = getattr(removed, "exit_code", None) == 0 and _regular(
            ws, "results.json"
        )
        state.official_ok = (
            state.run_started
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
        state.run_finished = True
        state.timed_out = (
            isinstance(exc, TimeoutError)
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS)
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
