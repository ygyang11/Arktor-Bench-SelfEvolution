from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

SCENES = (
    "initial_split",
    "crossing_obstacles",
    "narrow_passage",
    "temporary_separation",
    "dense_cluster",
)
ROBOTS = tuple(range(4))
HORIZON = 60
BOUNDARIES = tuple(range(HORIZON + 1))
WORKSPACE_MAX = 14.0
STEP = 0.35
OBSTACLE_MARGIN = 0.25
SAFE_DISTANCE = 0.8
COMM_RADIUS = 2.4
FRESHNESS_LIMIT = 5.0
MARGIN_TARGET = 0.30
GOAL_RADIUS = 0.35
BROADCAST_BUDGET = 120
ROLLING_LIMIT = 3
EPS = 1e-6

ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "direction", "broadcast"}

DIAGONAL_STEP = STEP / math.sqrt(2.0)
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "N": (0.0, STEP),
    "NE": (DIAGONAL_STEP, DIAGONAL_STEP),
    "E": (STEP, 0.0),
    "SE": (DIAGONAL_STEP, -DIAGONAL_STEP),
    "S": (0.0, -STEP),
    "SW": (-DIAGONAL_STEP, -DIAGONAL_STEP),
    "W": (-STEP, 0.0),
    "NW": (-DIAGONAL_STEP, DIAGONAL_STEP),
    "stay": (0.0, 0.0),
}
DIRECTIONS = tuple(DIRECTION_VECTORS)

SCENE_DATA: dict[str, dict[str, Any]] = {
    "initial_split": {
        "starts": ((1.4, 5.0), (1.4, 6.2), (4.4, 5.0), (4.4, 6.2)),
        "goals": ((10.8, 5.0), (10.8, 6.2), (12.2, 5.0), (12.2, 6.2)),
        "obstacles": ((6.2, 7.4, 0.0, 4.0), (6.2, 7.4, 7.2, 14.0)),
        "recovery_limit": 12,
    },
    "crossing_obstacles": {
        "starts": ((1.2, 3.2), (1.2, 4.4), (2.4, 3.2), (2.4, 4.4)),
        "goals": ((11.4, 8.6), (11.4, 9.8), (12.7, 8.6), (12.7, 9.8)),
        "obstacles": (
            (4.0, 5.0, 0.0, 6.2),
            (7.0, 8.0, 7.8, 14.0),
            (10.0, 11.0, 0.0, 6.2),
        ),
        "recovery_limit": 8,
    },
    "narrow_passage": {
        "starts": ((1.2, 5.4), (1.2, 6.6), (2.4, 5.4), (2.4, 6.6)),
        "goals": ((11.4, 5.4), (11.4, 6.6), (12.6, 5.4), (12.6, 6.6)),
        "obstacles": ((5.0, 9.0, 0.0, 4.8), (5.0, 9.0, 7.2, 14.0)),
        "recovery_limit": 8,
    },
    "temporary_separation": {
        "starts": ((1.2, 6.4), (1.2, 7.6), (2.4, 6.4), (2.4, 7.6)),
        "goals": ((11.4, 6.4), (11.4, 7.6), (12.6, 6.4), (12.6, 7.6)),
        "obstacles": ((5.2, 8.8, 5.3, 8.7),),
        "recovery_limit": 10,
    },
    "dense_cluster": {
        "starts": ((2.0, 5.0), (2.0, 6.0), (3.0, 5.0), (3.0, 6.0)),
        "goals": ((11.0, 6.0), (11.0, 5.0), (12.0, 6.0), (12.0, 5.0)),
        "obstacles": ((5.3, 6.4, 4.0, 5.2), (7.4, 8.5, 5.8, 7.0)),
        "recovery_limit": 6,
    },
}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.20,
        "result_evidence.action_containers": 0.20,
        "result_evidence.direction_fields": 0.20,
        "result_evidence.broadcast_fields": 0.20,
        "result_evidence.local_record_relations": 0.20,
    },
    "decision_legality": {
        "decision_legality.direction_domain": 1 / 6,
        "decision_legality.workspace_boundary": 1 / 6,
        "decision_legality.obstacle_clearance": 1 / 6,
        "decision_legality.swept_separation": 1 / 6,
        "decision_legality.rolling_broadcasts": 1 / 6,
        "decision_legality.broadcast_budget": 1 / 6,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.motion_update": 1 / 6,
        "replay_validity.broadcast_delivery": 1 / 6,
        "replay_validity.cache_merge": 1 / 6,
        "replay_validity.weighted_connectivity": 1 / 6,
        "replay_validity.recovery_and_goals": 1 / 6,
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


def _record_shape(value: Any, keys: set[str], canonical_key: str) -> bool:
    del canonical_key
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


def _rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    return []


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = _shape(data, ROOT_KEYS)
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = isinstance(scenes, list) and all(
        _record_shape(row, SCENE_KEYS, "scenario") and isinstance(row.get("slots"), list)
        for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected exact root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires only canonical ID plus a slots list",
        ),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = (
        "slot_containers",
        "action_containers",
        "direction_fields",
        "broadcast_fields",
        "local_record_relations",
    )
    flags = {name: True for name in names}
    details = {name: "all submitted local records satisfy the declared format" for name in names}
    rows = _rows(data)
    for scene_index, scene_row in enumerate(rows):
        slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
        if not isinstance(slots, list):
            continue
        for slot_index, slot in enumerate(slots):
            slot_ok = _record_shape(slot, SLOT_KEYS, "t") and isinstance(slot.get("actions"), list)
            if not slot_ok and flags["slot_containers"]:
                details["slot_containers"] = (
                    f"scenarios[{scene_index}].slots[{slot_index}]: "
                    "expected only canonical t plus an actions list"
                )
            flags["slot_containers"] &= slot_ok
            actions = slot.get("actions") if isinstance(slot, Mapping) else None
            if not isinstance(actions, list):
                continue
            for action_index, action in enumerate(actions):
                path = f"scenarios[{scene_index}].slots[{slot_index}].actions[{action_index}]"
                container_ok = _record_shape(action, ACTION_KEYS, "robot")
                if not container_ok and flags["action_containers"]:
                    details["action_containers"] = f"{path}: unexpected or missing decision keys"
                flags["action_containers"] &= container_ok
                direction_ok = (
                    not isinstance(action, Mapping)
                    or "direction" not in action
                    or isinstance(action.get("direction"), str)
                )
                if not direction_ok and flags["direction_fields"]:
                    details["direction_fields"] = f"{path}.direction: expected string"
                flags["direction_fields"] &= direction_ok
                broadcast_ok = (
                    not isinstance(action, Mapping)
                    or "broadcast" not in action
                    or isinstance(action.get("broadcast"), bool)
                )
                if not broadcast_ok and flags["broadcast_fields"]:
                    details["broadcast_fields"] = f"{path}.broadcast: expected Boolean"
                flags["broadcast_fields"] &= broadcast_ok
                relation_ok = (
                    not isinstance(action, Mapping)
                    or "robot" not in action
                    or _integer(action.get("robot"))
                )
                if not relation_ok and flags["local_record_relations"]:
                    details["local_record_relations"] = (
                        f"{path}.robot: expected integer canonical ID, got "
                        f"{_safe_repr(action.get('robot'))}"
                    )
                flags["local_record_relations"] &= relation_ok
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
            defects.extend(f"{label}: duplicate key={_safe_repr(key)}" for _ in range(count - 1))
    expected_set = set(expected)
    defects.extend(
        f"{label}: unknown key={_safe_repr(key)}" for key in actual if key not in expected_set
    )
    return defects


def _first_scene(data: Any, scene: str) -> Mapping[str, Any] | None:
    for row in _rows(data):
        if isinstance(row, Mapping) and row.get("scenario") == scene:
            return row
    return None


def _first_slot(scene_row: Any, t: int) -> Mapping[str, Any] | None:
    slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if isinstance(slot, Mapping) and _integer(slot.get("t")) and slot.get("t") == t:
            return slot
    return None


def _first_action(slot: Any, robot: int) -> Mapping[str, Any] | None:
    actions = slot.get("actions") if isinstance(slot, Mapping) else None
    if not isinstance(actions, list):
        return None
    for action in actions:
        if (
            isinstance(action, Mapping)
            and _integer(action.get("robot"))
            and action.get("robot") == robot
        ):
            return action
    return None


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            scene_keys.append(row["scenario"])
        else:
            scene_keys.append(("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))

    for scene in SCENES:
        scene_row = _first_scene(data, scene)
        slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
        if not isinstance(slots, list):
            continue
        slot_keys: list[Hashable] = []
        for index, slot in enumerate(slots):
            if isinstance(slot, Mapping) and _integer(slot.get("t")):
                slot_keys.append(slot["t"])
            else:
                slot_keys.append(("invalid_slot", index))
        defects.extend(_canonical_defects(slot_keys, list(range(HORIZON)), f"{scene}.slots"))
        for t in range(HORIZON):
            slot = _first_slot(scene_row, t)
            if slot is None:
                continue
            actions = slot.get("actions") if isinstance(slot, Mapping) else None
            if not isinstance(actions, list):
                continue
            action_keys: list[Hashable] = []
            for index, action in enumerate(actions):
                if isinstance(action, Mapping) and _integer(action.get("robot")):
                    action_keys.append(action["robot"])
                else:
                    action_keys.append(("invalid_robot", index))
            defects.extend(_canonical_defects(action_keys, ROBOTS, f"{scene}.slots[{t}].actions"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene in SCENES:
        scene_row = _first_scene(data, scene)
        if scene_row is None:
            continue
        for t in range(HORIZON):
            slot = _first_slot(scene_row, t)
            if slot is not None:
                records[(scene, t)] = slot
    return records


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _expanded(rectangle: Sequence[float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = rectangle
    return (
        xmin - OBSTACLE_MARGIN,
        xmax + OBSTACLE_MARGIN,
        ymin - OBSTACLE_MARGIN,
        ymax + OBSTACLE_MARGIN,
    )


def _point_in_closed_rectangle(point: Sequence[float], rectangle: Sequence[float]) -> bool:
    xmin, xmax, ymin, ymax = rectangle
    return xmin - EPS <= point[0] <= xmax + EPS and ymin - EPS <= point[1] <= ymax + EPS


def _segment_intersects_closed_rectangle(
    start: Sequence[float], end: Sequence[float], rectangle: Sequence[float]
) -> bool:
    if _point_in_closed_rectangle(start, rectangle) or _point_in_closed_rectangle(end, rectangle):
        return True
    xmin, xmax, ymin, ymax = rectangle
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    lower = 0.0
    upper = 1.0
    for origin, delta, low, high in (
        (float(start[0]), dx, xmin, xmax),
        (float(start[1]), dy, ymin, ymax),
    ):
        if abs(delta) <= 1e-15:
            if origin < low - EPS or origin > high + EPS:
                return False
            continue
        first = (low - origin) / delta
        second = (high - origin) / delta
        enter, leave = min(first, second), max(first, second)
        lower = max(lower, enter)
        upper = min(upper, leave)
        if lower > upper + EPS:
            return False
    return upper >= -EPS and lower <= 1.0 + EPS and lower <= upper + EPS


def _inside_workspace(point: Sequence[float]) -> bool:
    return all(-EPS <= float(value) <= WORKSPACE_MAX + EPS for value in point)


def _obstacle_clear(scene: str, start: Sequence[float], end: Sequence[float]) -> bool:
    return not any(
        _segment_intersects_closed_rectangle(start, end, _expanded(rectangle))
        for rectangle in SCENE_DATA[scene]["obstacles"]
    )


def _requested_endpoint(position: Sequence[float], direction: str) -> tuple[float, float]:
    dx, dy = DIRECTION_VECTORS[direction]
    return float(position[0]) + dx, float(position[1]) + dy


def _swept_distance(
    left_start: Sequence[float],
    left_end: Sequence[float],
    right_start: Sequence[float],
    right_end: Sequence[float],
) -> float:
    relative_start = (
        float(left_start[0]) - float(right_start[0]),
        float(left_start[1]) - float(right_start[1]),
    )
    relative_velocity = (
        (float(left_end[0]) - float(left_start[0])) - (float(right_end[0]) - float(right_start[0])),
        (float(left_end[1]) - float(left_start[1])) - (float(right_end[1]) - float(right_start[1])),
    )
    speed_sq = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    if speed_sq <= 1e-18:
        alpha = 0.0
    else:
        alpha = (
            -(relative_start[0] * relative_velocity[0] + relative_start[1] * relative_velocity[1])
            / speed_sq
        )
        alpha = max(0.0, min(1.0, alpha))
    return math.hypot(
        relative_start[0] + alpha * relative_velocity[0],
        relative_start[1] + alpha * relative_velocity[1],
    )


def _physical_edges(
    scene: str, positions: Sequence[Sequence[float]]
) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for left in ROBOTS:
        for right in range(left + 1, len(ROBOTS)):
            if _distance(positions[left], positions[right]) > COMM_RADIUS + EPS:
                continue
            if _obstacle_clear(scene, positions[left], positions[right]):
                edges.append((left, right))
    return tuple(edges)


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    values = [list(map(float, row)) for row in matrix]
    size = len(values)
    for _ in range(80):
        p, q = 0, 1
        largest = 0.0
        for i in range(size):
            for j in range(i + 1, size):
                if abs(values[i][j]) > largest:
                    largest = abs(values[i][j])
                    p, q = i, j
        if largest <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * values[p][q], values[q][q] - values[p][p])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        app, aqq, apq = values[p][p], values[q][q], values[p][q]
        for k in range(size):
            if k in (p, q):
                continue
            akp, akq = values[k][p], values[k][q]
            values[k][p] = values[p][k] = cosine * akp - sine * akq
            values[k][q] = values[q][k] = sine * akp + cosine * akq
        values[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        values[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        values[p][q] = values[q][p] = 0.0
    return tuple(sorted(max(0.0, values[i][i]) for i in range(size)))


def _coordination_graph(
    scene: str,
    boundary: int,
    positions: Sequence[Sequence[float]],
    caches: Sequence[Sequence[tuple[int, tuple[float, float]]]],
) -> dict[str, Any]:
    physical = _physical_edges(scene, positions)
    weights: dict[tuple[int, int], float] = {}
    laplacian = [[0.0 for _ in ROBOTS] for _ in ROBOTS]
    for left, right in physical:
        left_generation = caches[left][right][0]
        right_generation = caches[right][left][0]
        age = boundary - min(left_generation, right_generation)
        freshness = max(0.0, 1.0 - age / FRESHNESS_LIMIT)
        distance = _distance(positions[left], positions[right])
        range_factor = max(0.0, 1.0 - (distance / COMM_RADIUS) ** 2)
        weight = freshness * range_factor
        weights[(left, right)] = weight
        laplacian[left][left] += weight
        laplacian[right][right] += weight
        laplacian[left][right] -= weight
        laplacian[right][left] -= weight
    eigenvalues = _jacobi_eigenvalues(laplacian)
    lambda2 = eigenvalues[1]
    return {
        "physical_edges": physical,
        "weights": weights,
        "lambda2": lambda2,
        "connected": lambda2 > 1e-9,
        "maintained": lambda2 >= MARGIN_TARGET - EPS,
    }


def _cache_copy(
    caches: Sequence[Sequence[tuple[int, tuple[float, float]]]],
) -> list[list[tuple[int, tuple[float, float]]]]:
    return [[(entry[0], tuple(entry[1])) for entry in row] for row in caches]


def _cache_view(
    cache: Sequence[tuple[int, tuple[float, float]]], boundary: int
) -> list[dict[str, Any]]:
    return [
        {
            "robot": robot,
            "generation": entry[0],
            "position": [entry[1][0], entry[1][1]],
            "age": boundary - entry[0],
        }
        for robot, entry in enumerate(cache)
    ]


def _process_arrivals(
    boundary: int,
    messages: Sequence[Mapping[str, Any]],
    caches: list[list[tuple[int, tuple[float, float]]]],
) -> dict[str, Any]:
    before = _cache_copy(caches)
    selected: dict[tuple[int, int], tuple[int, tuple[float, float], int]] = {}
    novel_updates = 0
    freshness_total = 0.0
    payload_entries = 0
    events: list[dict[str, Any]] = []
    for message in sorted(messages, key=lambda item: (item["sender"], item["receiver"])):
        sender = int(message["sender"])
        receiver = int(message["receiver"])
        payload = message["payload"]
        event_novel = 0
        for robot, entry in enumerate(payload):
            generation, position = int(entry[0]), tuple(entry[1])
            payload_entries += 1
            freshness_total += max(0.0, min(1.0, 1.0 - (boundary - generation) / 5.0))
            if generation > before[receiver][robot][0]:
                novel_updates += 1
                event_novel += 1
            key = (receiver, robot)
            retained = selected.get(key)
            candidate = (generation, position, sender)
            if (
                retained is None
                or generation > retained[0]
                or (generation == retained[0] and sender < retained[2])
            ):
                selected[key] = candidate
        events.append(
            {
                "sender": sender,
                "receiver": receiver,
                "payload": tuple((int(item[0]), tuple(item[1])) for item in payload),
                "novel_updates": event_novel,
            }
        )
    for (receiver, robot), (generation, position, _sender) in selected.items():
        if generation > caches[receiver][robot][0]:
            caches[receiver][robot] = (generation, position)
    return {
        "events": events,
        "recipient_events": len(events),
        "novel_updates": novel_updates,
        "payload_entries": payload_entries,
        "freshness_total": freshness_total,
        "before": before,
    }


def _policy_view(
    scene: str,
    t: int,
    robot: int,
    positions: Sequence[Sequence[float]],
    caches: Sequence[Sequence[tuple[int, tuple[float, float]]]],
    physical_edges: Sequence[tuple[int, int]],
    broadcast_history: Sequence[Sequence[bool]],
    team_broadcasts: int,
) -> dict[str, Any]:
    neighbors = sorted(
        right if left == robot else left for left, right in physical_edges if robot in (left, right)
    )
    data = SCENE_DATA[scene]
    return {
        "t": t,
        "robot": robot,
        "position": [positions[robot][0], positions[robot][1]],
        "goal": list(data["goals"][robot]),
        "starts": [list(point) for point in data["starts"]],
        "goals": [list(point) for point in data["goals"]],
        "obstacles": [list(rectangle) for rectangle in data["obstacles"]],
        "recovery_limit": data["recovery_limit"],
        "cache": _cache_view(caches[robot], t),
        "physical_neighbors": neighbors,
        "preceding_three_broadcasts": sum(broadcast_history[robot][max(0, t - 3) : t]),
        "remaining_team_broadcasts": max(0, BROADCAST_BUDGET - team_broadcasts),
    }


def _submitted_actions(
    scene: str,
    t: int,
    records: Mapping[Hashable, Any] | None,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    positions: Sequence[Sequence[float]],
    caches: Sequence[Sequence[tuple[int, tuple[float, float]]]],
    physical_edges: Sequence[tuple[int, int]],
    broadcast_history: Sequence[Sequence[bool]],
    team_broadcasts: int,
) -> tuple[list[Any], bool]:
    if policy is not None:
        actions: list[Any] = []
        for robot in ROBOTS:
            proposed = policy(
                _policy_view(
                    scene,
                    t,
                    robot,
                    positions,
                    caches,
                    physical_edges,
                    broadcast_history,
                    team_broadcasts,
                )
            )
            if isinstance(proposed, Mapping):
                actions.append(
                    {
                        "robot": robot,
                        "direction": proposed.get("direction"),
                        "broadcast": proposed.get("broadcast"),
                    }
                )
            else:
                actions.append(proposed)
        return actions, True

    slot = records.get((scene, t)) if records is not None else None
    slot_ok = _shape(slot, SLOT_KEYS) and slot.get("t") == t
    if not slot_ok:
        return [None for _ in ROBOTS], False
    actions = [_first_action(slot, robot) for robot in ROBOTS]
    return actions, True


def _recovery_windows(lambda_history: Sequence[float], limit: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    boundary = 0
    while boundary <= HORIZON:
        if lambda_history[boundary] >= MARGIN_TARGET - EPS:
            boundary += 1
            continue
        start = boundary
        while boundary <= HORIZON and lambda_history[boundary] < MARGIN_TARGET - EPS:
            boundary += 1
        recovered = boundary if boundary <= HORIZON else None
        windows.append(
            {
                "start": start,
                "last_deficient": HORIZON if recovered is None else recovered - 1,
                "recovered": recovered,
                "success": recovered is not None and recovered - start <= limit,
            }
        )
    return windows


def _goal_summary(scene: str, boundaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for robot in ROBOTS:
        distances = [
            _distance(boundary["positions"][robot], SCENE_DATA[scene]["goals"][robot])
            for boundary in boundaries
        ]
        reached = next(
            (index for index, distance in enumerate(distances) if distance <= GOAL_RADIUS + EPS),
            None,
        )
        summaries.append(
            {
                "robot": robot,
                "first_reached": reached,
                "minimum_distance": min(distances),
                "final_distance": distances[-1],
            }
        )
    return summaries


def _simulate_scene(
    scene: str,
    *,
    records: Mapping[Hashable, Any] | None = None,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    resolver: Callable[
        [Sequence[Any], Sequence[Sequence[float]], Sequence[Sequence[float]]], Sequence[Any]
    ]
    | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    if (records is None) == (policy is None):
        raise ValueError("provide exactly one decision source")
    positions = [tuple(point) for point in SCENE_DATA[scene]["starts"]]
    caches: list[list[tuple[int, tuple[float, float]]]] = [
        [(0, tuple(point)) for point in SCENE_DATA[scene]["starts"]] for _ in ROBOTS
    ]
    pending: dict[int, list[dict[str, Any]]] = {boundary: [] for boundary in BOUNDARIES}
    broadcast_history: list[list[bool]] = [[] for _ in ROBOTS]
    team_broadcasts = 0
    boundaries: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    causal_active = True
    cutoff: int | None = None
    issues: dict[str, list[str]] = {
        "direction_domain": [],
        "workspace_boundary": [],
        "obstacle_clearance": [],
        "swept_separation": [],
        "rolling_broadcasts": [],
        "broadcast_budget": [],
    }
    communication = {
        "sent": 0,
        "delivered_broadcasts": 0,
        "recipient_events": 0,
        "novel_updates": 0,
        "payload_entries": 0,
        "freshness_total": 0.0,
    }

    for boundary in BOUNDARIES:
        arrival = _process_arrivals(boundary, pending[boundary], caches)
        communication["recipient_events"] += arrival["recipient_events"]
        communication["novel_updates"] += arrival["novel_updates"]
        communication["payload_entries"] += arrival["payload_entries"]
        communication["freshness_total"] += arrival["freshness_total"]
        communication["delivered_broadcasts"] += len(
            {event["sender"] for event in arrival["events"]}
        )
        for robot in ROBOTS:
            caches[robot][robot] = (boundary, tuple(positions[robot]))
        graph = _coordination_graph(scene, boundary, positions, caches)
        boundaries.append(
            {
                "b": boundary,
                "positions": tuple(tuple(point) for point in positions),
                "caches": tuple(tuple(row) for row in _cache_copy(caches)),
                "arrivals": tuple(arrival["events"]),
                "graph": graph,
            }
        )
        if boundary == HORIZON:
            break

        raw_actions, slot_container_ok = _submitted_actions(
            scene,
            boundary,
            records,
            policy,
            positions,
            caches,
            graph["physical_edges"],
            broadcast_history,
            team_broadcasts,
        )
        if policy is not None and resolver is not None:
            raw_actions = list(resolver(raw_actions, positions, SCENE_DATA[scene]["obstacles"]))
        action_logs: list[dict[str, Any]] = []
        slot_format_ok = slot_container_ok
        for robot, raw in enumerate(raw_actions):
            format_ok = (
                _shape(raw, ACTION_KEYS)
                and raw.get("robot") == robot
                and isinstance(raw.get("direction"), str)
                and isinstance(raw.get("broadcast"), bool)
            )
            slot_format_ok &= format_ok
            if format_ok:
                submitted_direction = raw["direction"]
                submitted_broadcast = raw["broadcast"]
            else:
                submitted_direction = "stay"
                submitted_broadcast = False
            action_logs.append(
                {
                    "robot": robot,
                    "raw": deepcopy(raw),
                    "format_ok": format_ok,
                    "submitted_direction": submitted_direction,
                    "submitted_broadcast": submitted_broadcast,
                }
            )
        state_attributable = causal_active and slot_format_ok
        if causal_active and not slot_format_ok:
            causal_active = False
            cutoff = boundary

        endpoints: list[tuple[float, float]] = []
        for log in action_logs:
            robot = log["robot"]
            direction = log["submitted_direction"]
            if log["format_ok"] and direction not in DIRECTION_VECTORS:
                issues["direction_domain"].append(
                    f"scene={scene}, t={boundary}, robot={robot}, direction={_safe_repr(direction)}"
                )
                direction = "stay"
            if direction not in DIRECTION_VECTORS:
                direction = "stay"
            requested = _requested_endpoint(positions[robot], direction)
            workspace_ok = _inside_workspace(requested)
            obstacle_ok = _obstacle_clear(scene, positions[robot], requested)
            if (
                state_attributable
                and log["format_ok"]
                and log["submitted_direction"] in DIRECTION_VECTORS
            ):
                if not workspace_ok:
                    issues["workspace_boundary"].append(
                        f"scene={scene}, t={boundary}, robot={robot}, endpoint={requested}, "
                        f"bounds=[0,{WORKSPACE_MAX}], tolerance={EPS}"
                    )
                if workspace_ok and not obstacle_ok:
                    issues["obstacle_clearance"].append(
                        f"scene={scene}, t={boundary}, robot={robot}, segment="
                        f"{positions[robot]}->{requested}, margin={OBSTACLE_MARGIN}, "
                        f"tolerance={EPS}"
                    )
            executed_direction = direction if workspace_ok and obstacle_ok else "stay"
            endpoint = _requested_endpoint(positions[robot], executed_direction)
            log["corrected_direction"] = executed_direction
            log["endpoint"] = endpoint
            endpoints.append(endpoint)

        if state_attributable:
            for left in ROBOTS:
                for right in range(left + 1, len(ROBOTS)):
                    separation = _swept_distance(
                        positions[left], endpoints[left], positions[right], endpoints[right]
                    )
                    if separation < SAFE_DISTANCE - EPS:
                        issues["swept_separation"].append(
                            f"scene={scene}, t={boundary}, pair=({left},{right}), "
                            f"minimum={separation:.12g}, required>={SAFE_DISTANCE - EPS:.12g}"
                        )

        accepted_broadcasts: list[bool] = []
        for log in action_logs:
            robot = log["robot"]
            requested = bool(log["submitted_broadcast"])
            rolling_ok = sum(broadcast_history[robot][max(0, boundary - 3) : boundary]) < 3
            budget_ok = team_broadcasts < BROADCAST_BUDGET
            accepted = requested and rolling_ok and budget_ok
            if requested and not rolling_ok:
                preceding_three = sum(broadcast_history[robot][max(0, boundary - 3) : boundary])
                issues["rolling_broadcasts"].append(
                    f"scene={scene}, t={boundary}, robot={robot}, "
                    f"preceding_three={preceding_three}, "
                    f"limit={ROLLING_LIMIT}"
                )
            if requested and not budget_ok:
                issues["broadcast_budget"].append(
                    f"scene={scene}, t={boundary}, robot={robot}, used={team_broadcasts}, "
                    f"budget={BROADCAST_BUDGET}"
                )
            accepted_broadcasts.append(accepted)
            broadcast_history[robot].append(requested)
            if accepted:
                team_broadcasts += 1
                communication["sent"] += 1
            log["accepted_broadcast"] = accepted

        start_positions = tuple(tuple(point) for point in positions)
        positions = endpoints
        post_edges = _physical_edges(scene, positions)
        generated: list[dict[str, Any]] = []
        for robot, accepted in enumerate(accepted_broadcasts):
            if not accepted:
                continue
            payload = _cache_copy(caches)[robot]
            payload[robot] = (boundary + 1, tuple(positions[robot]))
            recipients = sorted(
                right if left == robot else left
                for left, right in post_edges
                if robot in (left, right)
            )
            generated.append(
                {
                    "sender": robot,
                    "recipients": tuple(recipients),
                    "payload": tuple(payload),
                }
            )
            if communication_enabled:
                for receiver in recipients:
                    pending[boundary + 1].append(
                        {
                            "sender": robot,
                            "receiver": receiver,
                            "payload": tuple(payload),
                        }
                    )
        slots.append(
            {
                "t": boundary,
                "start_positions": start_positions,
                "actions": action_logs,
                "end_positions": tuple(tuple(point) for point in positions),
                "post_physical_edges": post_edges,
                "generated": tuple(generated),
                "state_attributable": state_attributable,
            }
        )

    lambda_history = [float(boundary["graph"]["lambda2"]) for boundary in boundaries]
    recovery = _recovery_windows(lambda_history, int(SCENE_DATA[scene]["recovery_limit"]))
    goals = _goal_summary(scene, boundaries)
    return {
        "scene": scene,
        "boundaries": boundaries,
        "slots": slots,
        "issues": issues,
        "cutoff": cutoff,
        "communication": communication,
        "recovery_windows": recovery,
        "goals": goals,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    return _simulate_scene(scene, records=records)


def _online_scene(
    scene: str,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    resolver: Callable[
        [Sequence[Any], Sequence[Sequence[float]], Sequence[Sequence[float]]], Sequence[Any]
    ]
    | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    return _simulate_scene(
        scene,
        policy=policy,
        resolver=resolver,
        communication_enabled=communication_enabled,
    )


def _result_from_online(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": [
            {
                "scenario": scene,
                "slots": [
                    {
                        "t": slot["t"],
                        "actions": [
                            {
                                "robot": action["robot"],
                                "direction": action["submitted_direction"],
                                "broadcast": action["submitted_broadcast"],
                            }
                            for action in slot["actions"]
                        ],
                    }
                    for slot in replayed[scene]["slots"]
                ],
            }
            for scene in SCENES
        ]
    }


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    atoms: list[AtomVerdict] = []
    for name in (
        "direction_domain",
        "workspace_boundary",
        "obstacle_clearance",
        "swept_separation",
        "rolling_broadcasts",
        "broadcast_budget",
    ):
        found = [detail for scene in SCENES for detail in replayed[scene]["issues"][name]]
        atoms.append(
            _atom(
                "decision_legality",
                name,
                not found,
                "no violations" if not found else found[0],
            )
        )
    return atoms


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=EPS)


def _point_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) == 2 and _close(left[0], right[0]) and _close(left[1], right[1])


def _decision_consumption_issue(
    scene: str, records: Mapping[Hashable, Any], replayed: Mapping[str, Any]
) -> str | None:
    for t in range(HORIZON):
        slot = records.get((scene, t))
        for robot in ROBOTS:
            expected = _first_action(slot, robot)
            actual = replayed["slots"][t]["actions"][robot]["raw"]
            if expected != actual:
                return (
                    f"scene={scene}, t={t}, robot={robot}, field=raw decision, "
                    f"expected={_safe_repr(expected)}, actual={_safe_repr(actual)}"
                )
    return None


def _motion_issue(scene: str, replayed: Mapping[str, Any]) -> str | None:
    for slot in replayed["slots"]:
        t = slot["t"]
        for robot in ROBOTS:
            start = slot["start_positions"][robot]
            direction = slot["actions"][robot]["corrected_direction"]
            expected = _requested_endpoint(start, direction)
            actual = slot["end_positions"][robot]
            if not _point_close(expected, actual):
                return (
                    f"scene={scene}, t={t}, robot={robot}, field=end_position, "
                    f"expected={expected}, actual={actual}, tolerance={EPS}"
                )
            next_position = replayed["boundaries"][t + 1]["positions"][robot]
            if not _point_close(actual, next_position):
                return (
                    f"scene={scene}, boundary={t + 1}, robot={robot}, field=position carry, "
                    f"expected={actual}, actual={next_position}, tolerance={EPS}"
                )
    return None


def _broadcast_issue(scene: str, replayed: Mapping[str, Any]) -> str | None:
    for slot in replayed["slots"]:
        t = slot["t"]
        expected_edges = _physical_edges(scene, slot["end_positions"])
        if tuple(slot["post_physical_edges"]) != expected_edges:
            return (
                f"scene={scene}, t={t}, field=post_physical_edges, "
                f"expected={expected_edges}, actual={slot['post_physical_edges']}"
            )
        expected_messages: set[tuple[int, int]] = set()
        for action in slot["actions"]:
            if not action["accepted_broadcast"]:
                continue
            sender = action["robot"]
            for left, right in expected_edges:
                if sender == left:
                    expected_messages.add((sender, right))
                elif sender == right:
                    expected_messages.add((sender, left))
        actual_messages = {
            (event["sender"], event["receiver"])
            for event in replayed["boundaries"][t + 1]["arrivals"]
        }
        if actual_messages != expected_messages:
            return (
                f"scene={scene}, boundary={t + 1}, field=broadcast recipients, "
                f"expected={sorted(expected_messages)}, actual={sorted(actual_messages)}"
            )
    return None


def _cache_issue(scene: str, replayed: Mapping[str, Any]) -> str | None:
    starts = SCENE_DATA[scene]["starts"]
    expected: list[list[tuple[int, tuple[float, float]]]] = [
        [(0, tuple(point)) for point in starts] for _ in ROBOTS
    ]
    for boundary in BOUNDARIES:
        if boundary > 0:
            _process_arrivals(boundary, replayed["boundaries"][boundary]["arrivals"], expected)
        for robot in ROBOTS:
            expected[robot][robot] = (
                boundary,
                tuple(replayed["boundaries"][boundary]["positions"][robot]),
            )
        actual = replayed["boundaries"][boundary]["caches"]
        for receiver in ROBOTS:
            for robot in ROBOTS:
                expected_entry = expected[receiver][robot]
                actual_entry = actual[receiver][robot]
                if expected_entry[0] != actual_entry[0] or not _point_close(
                    expected_entry[1], actual_entry[1]
                ):
                    return (
                        f"scene={scene}, boundary={boundary}, receiver={receiver}, "
                        f"robot={robot}, field=cache, expected={expected_entry}, "
                        f"actual={actual_entry}, tolerance={EPS}"
                    )
    return None


def _connectivity_issue(scene: str, replayed: Mapping[str, Any]) -> str | None:
    for boundary in replayed["boundaries"]:
        b = boundary["b"]
        expected = _coordination_graph(scene, b, boundary["positions"], boundary["caches"])
        actual = boundary["graph"]
        if expected["physical_edges"] != actual["physical_edges"]:
            return (
                f"scene={scene}, boundary={b}, field=physical_edges, "
                f"expected={expected['physical_edges']}, actual={actual['physical_edges']}"
            )
        if set(expected["weights"]) != set(actual["weights"]):
            return (
                f"scene={scene}, boundary={b}, field=weight keys, "
                f"expected={sorted(expected['weights'])}, actual={sorted(actual['weights'])}"
            )
        for edge, value in expected["weights"].items():
            if not _close(value, actual["weights"][edge]):
                return (
                    f"scene={scene}, boundary={b}, edge={edge}, field=weight, "
                    f"expected={value:.12g}, actual={actual['weights'][edge]:.12g}, "
                    f"tolerance={EPS}"
                )
        if not _close(expected["lambda2"], actual["lambda2"]):
            return (
                f"scene={scene}, boundary={b}, field=lambda2, "
                f"expected={expected['lambda2']:.12g}, actual={actual['lambda2']:.12g}, "
                f"tolerance={EPS}"
            )
    return None


def _recovery_goal_issue(scene: str, replayed: Mapping[str, Any]) -> str | None:
    lambdas = [boundary["graph"]["lambda2"] for boundary in replayed["boundaries"]]
    expected_windows = _recovery_windows(lambdas, int(SCENE_DATA[scene]["recovery_limit"]))
    if expected_windows != replayed["recovery_windows"]:
        return (
            f"scene={scene}, field=recovery_windows, expected={expected_windows}, "
            f"actual={replayed['recovery_windows']}"
        )
    expected_goals = _goal_summary(scene, replayed["boundaries"])
    for expected, actual in zip(expected_goals, replayed["goals"], strict=True):
        if (
            expected["robot"] != actual["robot"]
            or expected["first_reached"] != actual["first_reached"]
            or not _close(expected["minimum_distance"], actual["minimum_distance"])
            or not _close(expected["final_distance"], actual["final_distance"])
        ):
            return (
                f"scene={scene}, robot={expected['robot']}, field=goal summary, "
                f"expected={expected}, actual={actual}, tolerance={EPS}"
            )
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    checks: tuple[tuple[str, Callable[[str, Mapping[str, Any]], str | None]], ...] = (
        (
            "decision_consumption",
            lambda scene, result: _decision_consumption_issue(scene, records, result),
        ),
        ("motion_update", _motion_issue),
        ("broadcast_delivery", _broadcast_issue),
        ("cache_merge", _cache_issue),
        ("weighted_connectivity", _connectivity_issue),
        ("recovery_and_goals", _recovery_goal_issue),
    )
    atoms: list[AtomVerdict] = []
    for name, function in checks:
        issue = next(
            (found for scene in SCENES if (found := function(scene, replayed[scene])) is not None),
            None,
        )
        atoms.append(
            _atom(
                "replay_validity",
                name,
                issue is None,
                "independent replay relation matched" if issue is None else issue,
            )
        )
    return atoms


def _scene_metrics(scene: str, replayed: Mapping[str, Any]) -> dict[str, float]:
    goal_scores: list[float] = []
    for goal in replayed["goals"]:
        robot = goal["robot"]
        initial = _distance(SCENE_DATA[scene]["starts"][robot], SCENE_DATA[scene]["goals"][robot])
        progress = max(
            0.0,
            min(
                1.0,
                (initial - float(goal["minimum_distance"])) / (initial - GOAL_RADIUS),
            ),
        )
        reached = 1.0 if goal["first_reached"] is not None else 0.0
        goal_scores.append(0.7 * reached + 0.3 * progress)

    margin_quality = (
        sum(
            max(0.0, min(1.0, boundary["graph"]["lambda2"] / MARGIN_TARGET))
            for boundary in replayed["boundaries"][1:]
        )
        / HORIZON
    )
    windows = replayed["recovery_windows"]
    recovery_quality = (
        sum(1.0 for window in windows if window["success"]) / len(windows)
        if windows
        else margin_quality
    )

    communication = replayed["communication"]
    sent = int(communication["sent"])
    delivery_fraction = communication["delivered_broadcasts"] / max(1, sent)
    payload_freshness = communication["freshness_total"] / max(1, communication["payload_entries"])
    recipient_events = int(communication["recipient_events"])
    novel_updates = int(communication["novel_updates"])
    if novel_updates == 0:
        synchronization = 0.0
    else:
        synchronization = 0.7 * novel_updates / (4 * recipient_events) + 0.3 * max(
            0.0, min(1.0, 1.0 - sent / BROADCAST_BUDGET)
        )
    return {
        "primary_metrics_1": sum(goal_scores) / len(goal_scores),
        "primary_metrics_2": 0.65 * margin_quality + 0.35 * recovery_quality,
        "auxiliary_metrics_1": 0.5 * delivery_fraction + 0.5 * payload_freshness,
        "auxiliary_metrics_2": synchronization,
        "margin_quality": margin_quality,
        "recovery_quality": recovery_quality,
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    names = (
        "primary_metrics_1",
        "primary_metrics_2",
        "auxiliary_metrics_1",
        "auxiliary_metrics_2",
    )
    values = {
        name: sum(_scene_metrics(scene, replayed[scene])[name] for scene in SCENES) / len(SCENES)
        for name in names
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
        text = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
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
        detail = f"grader failure: {type(exc).__name__}: {exc}"
        state.official_message = (
            f"{state.official_message}; {detail}" if state.official_message else detail
        )
        for name in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
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
            failed: list[str] = []
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
            return 0.0, (f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0")
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
