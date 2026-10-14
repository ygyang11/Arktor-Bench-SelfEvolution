from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

SCENES = (
    "open_rooms",
    "branched_corridor",
    "narrow_cave",
    "request_burst",
    "operator_shift",
)
ROBOTS = tuple(range(4))
NODES = tuple(range(12))
HORIZON = 60
TERMINAL_BOUNDARY = 60
INITIAL_ENERGY = 36.0
TIMELY_BOUND = 8
ASSISTANCE_SLOTS = 4
EPS = 1e-6

SCENE_DATA: dict[str, dict[str, Any]] = {
    "open_rooms": {
        "edges": (
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 7),
            (4, 7),
            (4, 8),
            (5, 8),
            (5, 9),
            (6, 9),
            (7, 10),
            (8, 10),
            (8, 11),
            (9, 11),
        ),
        "values": (0, 4, 5, 6, 5, 7, 4, 8, 6, 7, 9, 10),
        "requests": {
            "R0": ("region", 8, 25, (7, 8, 10)),
            "R1": ("region", 20, 43, (9, 10, 11)),
            "R2": ("assistance", 28, 48, 2),
        },
    },
    "branched_corridor": {
        "edges": (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (2, 5),
            (5, 6),
            (6, 7),
            (3, 8),
            (8, 9),
            (1, 10),
            (10, 11),
        ),
        "values": (0, 3, 4, 5, 8, 6, 9, 10, 7, 11, 5, 12),
        "requests": {
            "R0": ("region", 7, 24, (4, 6, 7)),
            "R1": ("region", 21, 45, (9, 11)),
            "R2": ("assistance", 30, 52, 3),
        },
    },
    "narrow_cave": {
        "edges": (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (2, 6),
            (6, 7),
            (7, 8),
            (4, 9),
            (9, 10),
            (10, 11),
        ),
        "values": (0, 3, 5, 6, 8, 10, 4, 7, 11, 5, 9, 12),
        "requests": {
            "R0": ("region", 8, 26, (5, 8)),
            "R1": ("region", 23, 47, (10, 11)),
            "R2": ("assistance", 31, 54, 1),
        },
    },
    "request_burst": {
        "edges": (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (3, 9),
            (4, 9),
            (9, 10),
            (10, 11),
        ),
        "values": (0, 5, 4, 7, 6, 8, 9, 10, 11, 5, 12, 13),
        "requests": {
            "R0": ("region", 14, 30, (7, 9, 10)),
            "R1": ("region", 16, 38, (8, 10, 11)),
            "R2": ("assistance", 18, 40, 3),
        },
    },
    "operator_shift": {
        "edges": (
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (3, 7),
            (4, 7),
            (5, 8),
            (6, 9),
            (7, 10),
            (8, 11),
            (9, 11),
            (10, 11),
        ),
        "values": (0, 4, 6, 5, 8, 7, 9, 10, 6, 11, 12, 14),
        "requests": {
            "R0": ("region", 9, 28, (6, 7, 8)),
            "R1": ("region", 24, 48, (10, 11)),
            "R2": ("assistance", 20, 42, 0),
        },
    },
}

SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "move_to", "parent", "map_node"}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.20,
        "result_evidence.action_containers": 0.20,
        "result_evidence.movement_fields": 0.20,
        "result_evidence.parent_fields": 0.20,
        "result_evidence.map_fields": 0.20,
    },
    "decision_legality": {
        "decision_legality.movement_destination": 1 / 6,
        "decision_legality.parent_selection": 1 / 6,
        "decision_legality.parent_topology": 1 / 6,
        "decision_legality.linked_mobility": 1 / 6,
        "decision_legality.map_availability": 1 / 6,
        "decision_legality.energy_budget": 1 / 6,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.movement_discovery": 1 / 6,
        "replay_validity.topology_activation": 1 / 6,
        "replay_validity.map_propagation": 1 / 6,
        "replay_validity.request_service": 1 / 6,
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


def _record_shape(value: Any, keys: set[str], canonical_key: str) -> bool:
    # Canonical identity fields are required by the public record shape;
    # coverage separately owns uniqueness and order.
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

    def integer(value: str) -> int:
        try:
            return int(Decimal(value))
        except (InvalidOperation, OverflowError, ValueError) as exc:
            raise ValueError(f"invalid JSON integer {value[:80]!r}") from exc

    def finite_decimal(value: str) -> Decimal:
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid JSON number {value[:80]!r}") from exc
        if not parsed.is_finite():
            raise ValueError(f"non-finite JSON number {value[:80]!r}")
        return parsed

    return json.loads(
        raw,
        parse_constant=reject,
        parse_int=integer,
        parse_float=finite_decimal,
        object_pairs_hook=pairs,
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
            "each scene requires only its canonical ID and a slots array",
        ),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = (
        "slot_containers",
        "action_containers",
        "movement_fields",
        "parent_fields",
        "map_fields",
    )
    flags = {name: True for name in names}
    details = {name: "all submitted local fields satisfy their format" for name in names}
    rows = _rows(data)
    if not rows:
        for name in names:
            flags[name] = False
            details[name] = "scenarios: no submitted decision records"
    for scene_index, scene_row in enumerate(rows):
        if not isinstance(scene_row, Mapping) or not isinstance(scene_row.get("slots"), list):
            continue
        for slot_index, slot in enumerate(scene_row["slots"]):
            slot_ok = (
                isinstance(slot, Mapping)
                and set(slot) <= SLOT_KEYS
                and isinstance(slot.get("actions"), list)
            )
            if not slot_ok and flags["slot_containers"]:
                details["slot_containers"] = (
                    f"scenarios[{scene_index}].slots[{slot_index}]: "
                    "expected an actions array and no unknown keys"
                )
            flags["slot_containers"] &= slot_ok
            if not isinstance(slot, Mapping) or not isinstance(slot.get("actions"), list):
                continue
            for action_index, action in enumerate(slot["actions"]):
                action_ok = _record_shape(action, ACTION_KEYS, "robot")
                path = f"scenarios[{scene_index}].slots[{slot_index}].actions[{action_index}]"
                if not action_ok and flags["action_containers"]:
                    details["action_containers"] = (
                        f"{path}: expected decision fields and no unknown keys"
                    )
                flags["action_containers"] &= action_ok
                if not isinstance(action, Mapping):
                    continue
                move_ok = _integer(action.get("move_to"))
                if not move_ok and flags["movement_fields"]:
                    details["movement_fields"] = f"{path}.move_to: expected integer"
                flags["movement_fields"] &= move_ok
                parent = action.get("parent")
                parent_ok = parent is None or _integer(parent)
                if not parent_ok and flags["parent_fields"]:
                    details["parent_fields"] = f"{path}.parent: expected null or integer"
                flags["parent_fields"] &= parent_ok
                map_node = action.get("map_node")
                map_ok = map_node is None or _integer(map_node)
                if not map_ok and flags["map_fields"]:
                    details["map_fields"] = f"{path}.map_node: expected null or integer"
                flags["map_fields"] &= map_ok
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
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            if row["scenario"] == scene:
                return row
    return None


def _first_slot(scene_row: Any, t: int) -> Mapping[str, Any] | None:
    slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if isinstance(slot, Mapping) and _integer(slot.get("t")) and slot["t"] == t:
            return slot
    return None


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    scene_keys: list[Hashable] = []
    for index, row in enumerate(_rows(data)):
        if isinstance(row, Mapping):
            value = row.get("scenario")
            scene_keys.append(value if isinstance(value, str) else ("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))

    for scene in SCENES:
        scene_row = _first_scene(data, scene)
        slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
        slot_keys: list[Hashable] = []
        if isinstance(slots, list):
            for slot_index, slot in enumerate(slots):
                if isinstance(slot, Mapping):
                    value = slot.get("t")
                    slot_keys.append(value if _integer(value) else ("invalid_slot", slot_index))
        defects.extend(_canonical_defects(slot_keys, list(range(HORIZON)), f"{scene}.slots"))

        action_keys: list[Hashable] = []
        for t in range(HORIZON):
            slot = _first_slot(scene_row, t)
            actions = slot.get("actions") if isinstance(slot, Mapping) else None
            if not isinstance(actions, list):
                continue
            for action_index, action in enumerate(actions):
                if isinstance(action, Mapping):
                    value = action.get("robot")
                    action_keys.append(
                        (t, value) if _integer(value) else ("invalid_robot", t, action_index)
                    )
        expected_actions = [(t, robot) for t in range(HORIZON) for robot in ROBOTS]
        defects.extend(_canonical_defects(action_keys, expected_actions, f"{scene}.actions"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene in SCENES:
        row = _first_scene(data, scene)
        if row is None:
            continue
        for t in range(HORIZON):
            slot = _first_slot(row, t)
            if slot is not None:
                records[(scene, t)] = slot
    return records


def _first_action(slot: Any, robot: int) -> Mapping[str, Any] | None:
    actions = slot.get("actions") if isinstance(slot, Mapping) else None
    if not isinstance(actions, list):
        return None
    for action in actions:
        if (
            isinstance(action, Mapping)
            and _integer(action.get("robot"))
            and action["robot"] == robot
        ):
            return action
    return None


def _edge_set(scene: str) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in SCENE_DATA[scene]["edges"]}


def _operator_node(scene: str, boundary: int) -> int:
    if scene != "operator_shift" or boundary <= 19:
        return 0
    if boundary <= 21:
        return 1
    return 3


def _in_radio(scene: str, left: int, right: int) -> bool:
    return left == right or frozenset((left, right)) in _edge_set(scene)


def _topology_issue(
    scene: str,
    parents: Sequence[int | None],
    positions: Sequence[int],
    operator_node: int,
) -> str | None:
    robot_children = {robot: 0 for robot in ROBOTS}
    operator_children = 0
    for child, parent in enumerate(parents):
        if parent == -1:
            operator_children += 1
        elif parent is not None:
            robot_children[parent] += 1
    if operator_children > 2:
        return f"operator child count={operator_children}, maximum=2"
    overloaded = [robot for robot, count in robot_children.items() if count > 1]
    if overloaded:
        return f"robots with more than one child={overloaded}"

    for child, parent in enumerate(parents):
        if parent is None:
            continue
        seen = {child}
        current = child
        while True:
            next_parent = parents[current]
            if next_parent == -1:
                break
            if next_parent is None:
                return f"chain from robot={child} does not terminate at operator"
            if next_parent in seen:
                return f"cycle reached from robot={child} at robot={next_parent}"
            seen.add(next_parent)
            current = next_parent

    for child, parent in enumerate(parents):
        if parent is None:
            continue
        parent_node = operator_node if parent == -1 else positions[parent]
        if not _in_radio(scene, positions[child], parent_node):
            return (
                f"edge child={child}, parent={parent} out of range at "
                f"nodes=({positions[child]},{parent_node})"
            )
    return None


def _active_edges(
    scene: str,
    parents: Sequence[int | None],
    stored: Sequence[int | None],
    positions: Sequence[int],
    operator_node: int,
) -> tuple[bool, ...]:
    result: list[bool] = []
    for child, parent in enumerate(parents):
        parent_node = (
            operator_node if parent == -1 else positions[parent] if parent is not None else None
        )
        result.append(
            parent is not None
            and parent == stored[child]
            and parent_node is not None
            and _in_radio(scene, positions[child], parent_node)
        )
    return tuple(result)


def _incident_robots(parents: Sequence[int | None]) -> set[int]:
    incident: set[int] = set()
    for child, parent in enumerate(parents):
        if parent is None:
            continue
        incident.add(child)
        if parent != -1:
            incident.add(parent)
    return incident


def _decision_costs(
    positions: Sequence[int],
    moves: Sequence[int],
    parents: Sequence[int | None],
    stored: Sequence[int | None],
    map_nodes: Sequence[int | None],
) -> tuple[list[float], float]:
    costs = [1.0 if moves[robot] != positions[robot] else 0.0 for robot in ROBOTS]
    communication_cost = 0.0
    for child, parent in enumerate(parents):
        if parent is None:
            continue
        costs[child] += 0.25
        communication_cost += 0.25
        if parent != -1:
            costs[parent] += 0.25
            communication_cost += 0.25
    for robot in ROBOTS:
        if parents[robot] != stored[robot]:
            costs[robot] += 0.5
        if map_nodes[robot] is not None:
            costs[robot] += 0.1
            communication_cost += 0.1
    return costs, communication_cost


def _request_order(scene: str, request_id: str) -> tuple[int, str]:
    return int(SCENE_DATA[scene]["requests"][request_id][1]), request_id


def _released_requests(scene: str, boundary: int) -> set[str]:
    return {
        request_id
        for request_id, request in SCENE_DATA[scene]["requests"].items()
        if int(request[1]) <= boundary
    }


def _has_active_chain(robot: int, parents: Sequence[int | None], active: Sequence[bool]) -> bool:
    seen: set[int] = set()
    current = robot
    while current not in seen and parents[current] is not None:
        seen.add(current)
        if not active[current]:
            return False
        parent = parents[current]
        if parent == -1:
            return True
        assert parent is not None
        current = parent
    return False


def _knowledge_signature(knowledge: Sequence[Mapping[int, int]]) -> tuple[Any, ...]:
    return tuple(tuple(sorted(item.items())) for item in knowledge)


def _request_signature(requests: Sequence[Mapping[str, int]]) -> tuple[Any, ...]:
    return tuple(tuple(sorted(item.items())) for item in requests)


def _policy_view(
    scene: str,
    t: int,
    positions: Sequence[int],
    knowledge: Sequence[Mapping[int, int]],
    robot_requests: Sequence[Mapping[str, int]],
    stored: Sequence[int | None],
    previous_active: Sequence[bool],
    energy: Sequence[float],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    operator_node = _operator_node(scene, t)
    values = SCENE_DATA[scene]["values"]
    robot_views: list[dict[str, Any]] = []
    for robot in ROBOTS:
        neighbors = [
            other
            for other in ROBOTS
            if other != robot and _in_radio(scene, positions[robot], positions[other])
        ]
        if _in_radio(scene, positions[robot], operator_node):
            neighbors.append(-1)
        delivered = []
        for request_id, received in sorted(
            robot_requests[robot].items(), key=lambda item: _request_order(scene, item[0])
        ):
            kind, release, deadline, target = SCENE_DATA[scene]["requests"][request_id]
            delivered.append(
                {
                    "request_id": request_id,
                    "kind": kind,
                    "release_slot": release,
                    "deadline": deadline,
                    "target": list(target) if isinstance(target, tuple) else target,
                    "received_boundary": received,
                }
            )
        robot_views.append(
            {
                "robot": robot,
                "node": positions[robot],
                "known_nodes": [
                    {"node": node, "generation": generation, "value": values[node]}
                    for node, generation in sorted(knowledge[robot].items())
                ],
                "delivered_requests": delivered,
                "stored_parent": stored[robot],
                "preceding_edge_active": bool(previous_active[robot]),
                "energy_remaining": float(energy[robot]),
                "radio_neighbors": neighbors,
            }
        )
    return {
        "scene": scene,
        "scene_index": SCENES.index(scene),
        "t": t,
        "operator_node": operator_node,
        "graph": {
            "nodes": list(NODES),
            "edges": [list(edge) for edge in SCENE_DATA[scene]["edges"]],
        },
        "robots": robot_views,
        "history": list(history),
    }


def _select_submitted_actions(source: Any) -> list[Mapping[str, Any] | None]:
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        return [None for _ in ROBOTS]
    selected: list[Mapping[str, Any] | None] = []
    for robot in ROBOTS:
        selected.append(
            next(
                (
                    action
                    for action in source
                    if isinstance(action, Mapping)
                    and _integer(action.get("robot"))
                    and action["robot"] == robot
                ),
                None,
            )
        )
    return selected


def _action_well_typed(action: Any, robot: int) -> bool:
    return (
        isinstance(action, Mapping)
        and set(action) == ACTION_KEYS
        and _integer(action.get("robot"))
        and action["robot"] == robot
        and _integer(action.get("move_to"))
        and (action.get("parent") is None or _integer(action.get("parent")))
        and (action.get("map_node") is None or _integer(action.get("map_node")))
    )


def _admit_boundary(
    boundary: int,
    pending_discoveries: Mapping[int, list[tuple[int, int, int]]],
    pending_maps: Mapping[int, list[tuple[int, int, int]]],
    pending_requests: Mapping[int, list[tuple[int, str]]],
    knowledge: list[dict[int, int]],
    robot_requests: list[dict[str, int]],
    operator_knowledge: dict[int, int],
    operator_arrivals: dict[int, int],
) -> None:
    for robot, node, generation in pending_discoveries.get(boundary, []):
        knowledge[robot][node] = min(knowledge[robot].get(node, generation), generation)
    for receiver, node, generation in pending_maps.get(boundary, []):
        if receiver == -1:
            if node not in operator_knowledge:
                operator_arrivals[node] = boundary
            operator_knowledge[node] = min(operator_knowledge.get(node, generation), generation)
        else:
            knowledge[receiver][node] = min(knowledge[receiver].get(node, generation), generation)
    for receiver, request_id in pending_requests.get(boundary, []):
        robot_requests[receiver].setdefault(request_id, boundary)


def _simulate_scene(
    scene: str,
    action_source: Callable[[Mapping[str, Any]], Any],
    *,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    positions = [0 for _ in ROBOTS]
    knowledge = [{0: 0} for _ in ROBOTS]
    robot_requests: list[dict[str, int]] = [{} for _ in ROBOTS]
    stored: list[int | None] = [None for _ in ROBOTS]
    previous_active = [False for _ in ROBOTS]
    energy = [INITIAL_ENERGY for _ in ROBOTS]
    team_discovery = {0: 0}
    operator_knowledge = {0: 0}
    operator_arrivals = {0: 0}
    pending_discoveries: dict[int, list[tuple[int, int, int]]] = {}
    pending_maps: dict[int, list[tuple[int, int, int]]] = {}
    pending_requests: dict[int, list[tuple[int, str]]] = {}
    assistance_current = 0
    assistance_max = 0
    communication_energy = 0.0
    parent_changes = 0
    cutoff: int | None = None
    violations: dict[str, list[str]] = {
        "movement_destination": [],
        "parent_selection": [],
        "parent_topology": [],
        "linked_mobility": [],
        "map_availability": [],
        "energy_budget": [],
    }
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    signatures = {
        "positions": [],
        "discoveries": [],
        "stored": [],
        "active": [],
        "knowledge": [],
        "operator": [],
        "requests": [],
        "assistance": [],
        "energy": [],
        "communication_energy_total": [],
        "parent_changes_total": [],
    }

    for t in range(HORIZON):
        _admit_boundary(
            t,
            pending_discoveries,
            pending_maps,
            pending_requests,
            knowledge,
            robot_requests,
            operator_knowledge,
            operator_arrivals,
        )
        operator_node = _operator_node(scene, t)
        view = _policy_view(
            scene,
            t,
            positions,
            knowledge,
            robot_requests,
            stored,
            previous_active,
            energy,
            history,
        )
        selected = _select_submitted_actions(action_source(view))
        malformed = [robot for robot in ROBOTS if not _action_well_typed(selected[robot], robot)]
        if malformed and cutoff is None:
            cutoff = t
        attributable = cutoff is None

        moves = list(positions)
        parents: list[int | None] = [None for _ in ROBOTS]
        map_nodes: list[int | None] = [None for _ in ROBOTS]
        for robot in ROBOTS:
            action = selected[robot]
            if robot in malformed:
                continue
            assert action is not None
            move = int(action["move_to"])
            parent = action["parent"]
            map_node = action["map_node"]

            if move not in NODES:
                violations["movement_destination"].append(
                    f"{scene}.t={t}.robot={robot}.move_to={_safe_repr(move)}: outside node domain"
                )
            elif not _in_radio(scene, positions[robot], move):
                if attributable:
                    violations["movement_destination"].append(
                        f"{scene}.t={t}.robot={robot}: nonadjacent move {positions[robot]}->{move}"
                    )
            else:
                moves[robot] = move

            if parent is not None and (parent not in (-1, *ROBOTS) or parent == robot):
                violations["parent_selection"].append(
                    f"{scene}.t={t}.robot={robot}.parent={_safe_repr(parent)}: outside legal domain"
                )
            else:
                parents[robot] = parent

            if map_node is not None and map_node not in NODES:
                violations["map_availability"].append(
                    f"{scene}.t={t}.robot={robot}.map_node={_safe_repr(map_node)}: "
                    "outside node domain"
                )
            else:
                map_nodes[robot] = map_node

        topology_issue = _topology_issue(scene, parents, positions, operator_node)
        if topology_issue is not None:
            if attributable:
                violations["parent_topology"].append(f"{scene}.t={t}: {topology_issue}")
            parents = [None for _ in ROBOTS]

        incident = _incident_robots(parents)
        for robot in ROBOTS:
            if robot in incident and moves[robot] != positions[robot]:
                if attributable:
                    violations["linked_mobility"].append(
                        f"{scene}.t={t}.robot={robot}: desired-edge endpoint moved"
                    )
                moves[robot] = positions[robot]

        active = _active_edges(scene, parents, stored, positions, operator_node)
        for robot in ROBOTS:
            node = map_nodes[robot]
            if node is None:
                continue
            if not active[robot] or node not in knowledge[robot]:
                if attributable:
                    violations["map_availability"].append(
                        f"{scene}.t={t}.robot={robot}.map_node={node}: "
                        f"known={node in knowledge[robot]}, active_parent={active[robot]}"
                    )
                map_nodes[robot] = None

        costs, slot_communication_energy = _decision_costs(
            positions, moves, parents, stored, map_nodes
        )
        energy_rejected = any(costs[robot] > energy[robot] + EPS for robot in ROBOTS)
        accepted = not energy_rejected
        if energy_rejected:
            if attributable:
                violations["energy_budget"].append(
                    f"{scene}.t={t}: costs={costs}, remaining={energy}"
                )
            moves = list(positions)
            parents = list(stored)
            map_nodes = [None for _ in ROBOTS]
            active = _active_edges(scene, parents, stored, positions, operator_node)
            costs, slot_communication_energy = _decision_costs(
                positions, moves, parents, stored, map_nodes
            )

        if communication_enabled:
            boundary_knowledge = [dict(item) for item in knowledge]
            boundary_requests = [set(item) for item in robot_requests]
            released = _released_requests(scene, t)
            for child, parent in enumerate(parents):
                if not active[child]:
                    continue
                node = map_nodes[child]
                if node is not None:
                    pending_maps.setdefault(t + 1, []).append(
                        (parent, node, boundary_knowledge[child][node])
                    )
                parent_requests = released if parent == -1 else boundary_requests[parent]
                candidates = sorted(
                    parent_requests - boundary_requests[child],
                    key=lambda request_id: _request_order(scene, request_id),
                )
                if candidates:
                    pending_requests.setdefault(t + 1, []).append((child, candidates[0]))

        for robot in ROBOTS:
            if moves[robot] != positions[robot]:
                generation = t + 1
                pending_discoveries.setdefault(generation, []).append(
                    (robot, moves[robot], generation)
                )
                team_discovery.setdefault(moves[robot], generation)

        assistance = SCENE_DATA[scene]["requests"]["R2"]
        _, _, assistance_deadline, target = assistance
        eligible = (
            communication_enabled
            and t < int(assistance_deadline)
            and "R2" in robot_requests[int(target)]
            and _has_active_chain(int(target), parents, active)
        )
        if eligible:
            assistance_current += 1
            assistance_max = max(assistance_max, assistance_current)
        else:
            assistance_current = 0

        if accepted:
            parent_changes += sum(parents[robot] != stored[robot] for robot in ROBOTS)
        communication_energy += slot_communication_energy
        energy = [energy[robot] - costs[robot] for robot in ROBOTS]
        positions = list(moves)
        stored = list(parents)
        previous_active = list(active)

        submitted_for_history = [
            dict(action) if isinstance(action, Mapping) else {"robot": robot}
            for robot, action in enumerate(selected)
        ]
        executed = [
            {
                "robot": robot,
                "move_to": moves[robot],
                "parent": parents[robot],
                "map_node": map_nodes[robot],
            }
            for robot in ROBOTS
        ]
        history.append({"t": t, "actions": submitted_for_history})
        trace.append(
            {
                "t": t,
                "submitted": submitted_for_history,
                "executed": executed,
                "accepted": accepted,
                "active": tuple(active),
                "costs": tuple(costs),
                "communication_energy": slot_communication_energy,
                "causal_attributable": attributable,
            }
        )
        signatures["positions"].append(tuple(positions))
        signatures["discoveries"].append(tuple(sorted(team_discovery.items())))
        signatures["stored"].append(tuple(stored))
        signatures["active"].append(tuple(active))
        signatures["knowledge"].append(_knowledge_signature(knowledge))
        signatures["operator"].append(
            (tuple(sorted(operator_knowledge.items())), tuple(sorted(operator_arrivals.items())))
        )
        signatures["requests"].append(_request_signature(robot_requests))
        signatures["assistance"].append((assistance_current, assistance_max))
        signatures["energy"].append(tuple(energy))
        signatures["communication_energy_total"].append(communication_energy)
        signatures["parent_changes_total"].append(parent_changes)

    _admit_boundary(
        TERMINAL_BOUNDARY,
        pending_discoveries,
        pending_maps,
        pending_requests,
        knowledge,
        robot_requests,
        operator_knowledge,
        operator_arrivals,
    )
    signatures["knowledge"].append(_knowledge_signature(knowledge))
    signatures["operator"].append(
        (tuple(sorted(operator_knowledge.items())), tuple(sorted(operator_arrivals.items())))
    )
    signatures["requests"].append(_request_signature(robot_requests))
    return {
        "scene": scene,
        "trace": trace,
        "violations": violations,
        "causal_cutoff": cutoff,
        "positions": tuple(positions),
        "knowledge": _knowledge_signature(knowledge),
        "operator_knowledge": tuple(sorted(operator_knowledge.items())),
        "operator_arrivals": dict(operator_arrivals),
        "team_discovery": dict(team_discovery),
        "robot_requests": _request_signature(robot_requests),
        "assistance_max": assistance_max,
        "energy": tuple(energy),
        "communication_energy": communication_energy,
        "parent_changes": parent_changes,
        "signatures": signatures,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    def source(view: Mapping[str, Any]) -> list[Any]:
        slot = records.get((scene, int(view["t"])))
        return [_first_action(slot, robot) for robot in ROBOTS]

    return _simulate_scene(scene, source)


def _online_scene(scene: str, policy: Any, *, communication_enabled: bool = True) -> dict[str, Any]:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()

    replayed = _simulate_scene(
        scene,
        lambda view: policy.decide_joint(view),
        communication_enabled=communication_enabled,
    )
    return {
        "scenario": scene,
        "slots": [{"t": item["t"], "actions": item["submitted"]} for item in replayed["trace"]],
    }


def _reference_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    edges = {frozenset(edge) for edge in SCENE_DATA[scene]["edges"]}

    def in_radio(left: int, right: int) -> bool:
        return left == right or frozenset((left, right)) in edges

    def operator_at(boundary: int) -> int:
        if scene != "operator_shift" or boundary <= 19:
            return 0
        return 1 if boundary <= 21 else 3

    def submitted_actions(t: int) -> list[Mapping[str, Any] | None]:
        slot = records.get((scene, t))
        actions = slot.get("actions") if isinstance(slot, Mapping) else None
        if not isinstance(actions, list):
            return [None for _ in ROBOTS]
        result: list[Mapping[str, Any] | None] = []
        for robot in ROBOTS:
            result.append(
                next(
                    (
                        action
                        for action in actions
                        if isinstance(action, Mapping)
                        and _integer(action.get("robot"))
                        and action["robot"] == robot
                    ),
                    None,
                )
            )
        return result

    def well_typed(action: Any, robot: int) -> bool:
        return (
            isinstance(action, Mapping)
            and set(action) == ACTION_KEYS
            and _integer(action.get("robot"))
            and action["robot"] == robot
            and _integer(action.get("move_to"))
            and (action.get("parent") is None or _integer(action.get("parent")))
            and (action.get("map_node") is None or _integer(action.get("map_node")))
        )

    def topology_valid(
        parents: Sequence[int | None], positions: Sequence[int], operator_node: int
    ) -> bool:
        if sum(parent == -1 for parent in parents) > 2:
            return False
        if any(sum(parent == robot for parent in parents) > 1 for robot in ROBOTS):
            return False
        for child, parent in enumerate(parents):
            if parent is None:
                continue
            seen = {child}
            current = child
            while parents[current] != -1:
                nxt = parents[current]
                if nxt is None or nxt in seen:
                    return False
                seen.add(nxt)
                current = int(nxt)
        for child, parent in enumerate(parents):
            if parent is None:
                continue
            parent_node = operator_node if parent == -1 else positions[int(parent)]
            if not in_radio(positions[child], parent_node):
                return False
        return True

    def active_edges(
        parents: Sequence[int | None],
        stored_parents: Sequence[int | None],
        positions: Sequence[int],
        operator_node: int,
    ) -> tuple[bool, ...]:
        result: list[bool] = []
        for child, parent in enumerate(parents):
            parent_node = (
                operator_node
                if parent == -1
                else positions[int(parent)]
                if parent is not None
                else None
            )
            result.append(
                parent is not None
                and parent == stored_parents[child]
                and parent_node is not None
                and in_radio(positions[child], parent_node)
            )
        return tuple(result)

    def decision_costs(
        positions: Sequence[int],
        moves: Sequence[int],
        parents: Sequence[int | None],
        stored_parents: Sequence[int | None],
        map_nodes: Sequence[int | None],
    ) -> tuple[list[float], float]:
        costs = [1.0 if moves[robot] != positions[robot] else 0.0 for robot in ROBOTS]
        communication_cost = 0.0
        for child, parent in enumerate(parents):
            if parent is None:
                continue
            costs[child] += 0.25
            communication_cost += 0.25
            if parent != -1:
                costs[int(parent)] += 0.25
                communication_cost += 0.25
        for robot in ROBOTS:
            if parents[robot] != stored_parents[robot]:
                costs[robot] += 0.5
            if map_nodes[robot] is not None:
                costs[robot] += 0.1
                communication_cost += 0.1
        return costs, communication_cost

    def admit(boundary: int) -> None:
        for robot, node, generation in pending_discoveries.get(boundary, []):
            knowledge[robot][node] = min(knowledge[robot].get(node, generation), generation)
        for receiver, node, generation in pending_maps.get(boundary, []):
            if receiver == -1:
                if node not in operator_knowledge:
                    operator_arrivals[node] = boundary
                operator_knowledge[node] = min(operator_knowledge.get(node, generation), generation)
            else:
                knowledge[receiver][node] = min(
                    knowledge[receiver].get(node, generation), generation
                )
        for receiver, request_id in pending_requests.get(boundary, []):
            robot_requests[receiver].setdefault(request_id, boundary)

    def active_chain(robot: int, parents: Sequence[int | None], active: Sequence[bool]) -> bool:
        seen: set[int] = set()
        current = robot
        while current not in seen and parents[current] is not None:
            seen.add(current)
            if not active[current]:
                return False
            parent = parents[current]
            if parent == -1:
                return True
            current = int(parent)
        return False

    positions = [0 for _ in ROBOTS]
    knowledge = [{0: 0} for _ in ROBOTS]
    robot_requests: list[dict[str, int]] = [{} for _ in ROBOTS]
    stored: list[int | None] = [None for _ in ROBOTS]
    energy = [INITIAL_ENERGY for _ in ROBOTS]
    team_discovery = {0: 0}
    operator_knowledge = {0: 0}
    operator_arrivals = {0: 0}
    pending_discoveries: dict[int, list[tuple[int, int, int]]] = {}
    pending_maps: dict[int, list[tuple[int, int, int]]] = {}
    pending_requests: dict[int, list[tuple[int, str]]] = {}
    assistance_current = 0
    assistance_max = 0
    communication_energy = 0.0
    parent_changes = 0
    signatures = {
        "positions": [],
        "discoveries": [],
        "stored": [],
        "active": [],
        "knowledge": [],
        "operator": [],
        "requests": [],
        "assistance": [],
        "energy": [],
        "communication_energy_total": [],
        "parent_changes_total": [],
    }
    reference_trace: list[dict[str, Any]] = []

    for t in range(HORIZON):
        admit(t)
        selected = submitted_actions(t)
        moves = list(positions)
        parents: list[int | None] = [None for _ in ROBOTS]
        map_nodes: list[int | None] = [None for _ in ROBOTS]
        for robot in ROBOTS:
            action = selected[robot]
            if not well_typed(action, robot):
                continue
            assert action is not None
            move = int(action["move_to"])
            if move in NODES and in_radio(positions[robot], move):
                moves[robot] = move
            parent = action["parent"]
            if parent is None or parent in (-1, *ROBOTS) and parent != robot:
                parents[robot] = parent
            map_node = action["map_node"]
            if map_node is None or map_node in NODES:
                map_nodes[robot] = map_node

        operator_node = operator_at(t)
        if not topology_valid(parents, positions, operator_node):
            parents = [None for _ in ROBOTS]

        incident: set[int] = set()
        for child, parent in enumerate(parents):
            if parent is None:
                continue
            incident.add(child)
            if parent != -1:
                incident.add(int(parent))
        for robot in incident:
            moves[robot] = positions[robot]

        active = active_edges(parents, stored, positions, operator_node)
        for robot, node in enumerate(map_nodes):
            if node is not None and (not active[robot] or node not in knowledge[robot]):
                map_nodes[robot] = None

        costs, slot_communication_energy = decision_costs(
            positions, moves, parents, stored, map_nodes
        )
        accepted = not any(costs[robot] > energy[robot] + EPS for robot in ROBOTS)
        if not accepted:
            moves = list(positions)
            parents = list(stored)
            map_nodes = [None for _ in ROBOTS]
            active = active_edges(parents, stored, positions, operator_node)
            costs, slot_communication_energy = decision_costs(
                positions, moves, parents, stored, map_nodes
            )

        boundary_knowledge = [dict(item) for item in knowledge]
        boundary_requests = [set(item) for item in robot_requests]
        released = {
            request_id
            for request_id, request in SCENE_DATA[scene]["requests"].items()
            if int(request[1]) <= t
        }
        for child, parent in enumerate(parents):
            if not active[child]:
                continue
            node = map_nodes[child]
            if node is not None:
                pending_maps.setdefault(t + 1, []).append(
                    (parent, node, boundary_knowledge[child][node])
                )
            parent_requests = released if parent == -1 else boundary_requests[parent]
            candidates = sorted(
                parent_requests - boundary_requests[child],
                key=lambda request_id: (
                    int(SCENE_DATA[scene]["requests"][request_id][1]),
                    request_id,
                ),
            )
            if candidates:
                pending_requests.setdefault(t + 1, []).append((child, candidates[0]))

        for robot in ROBOTS:
            if moves[robot] != positions[robot]:
                generation = t + 1
                pending_discoveries.setdefault(generation, []).append(
                    (robot, moves[robot], generation)
                )
                team_discovery.setdefault(moves[robot], generation)

        _, _, assistance_deadline, target = SCENE_DATA[scene]["requests"]["R2"]
        eligible = (
            t < int(assistance_deadline)
            and "R2" in robot_requests[int(target)]
            and active_chain(int(target), parents, active)
        )
        if eligible:
            assistance_current += 1
            assistance_max = max(assistance_max, assistance_current)
        else:
            assistance_current = 0

        if accepted:
            parent_changes += sum(parents[robot] != stored[robot] for robot in ROBOTS)
        communication_energy += slot_communication_energy
        energy = [energy[robot] - costs[robot] for robot in ROBOTS]
        positions = moves
        stored = parents

        reference_trace.append(
            {
                "t": t,
                "executed": [
                    {
                        "robot": robot,
                        "move_to": moves[robot],
                        "parent": parents[robot],
                        "map_node": map_nodes[robot],
                    }
                    for robot in ROBOTS
                ],
                "accepted": accepted,
                "active": tuple(active),
                "costs": tuple(costs),
                "communication_energy": slot_communication_energy,
            }
        )

        signatures["positions"].append(tuple(positions))
        signatures["discoveries"].append(tuple(sorted(team_discovery.items())))
        signatures["stored"].append(tuple(stored))
        signatures["active"].append(tuple(active))
        signatures["knowledge"].append(_knowledge_signature(knowledge))
        signatures["operator"].append(
            (tuple(sorted(operator_knowledge.items())), tuple(sorted(operator_arrivals.items())))
        )
        signatures["requests"].append(_request_signature(robot_requests))
        signatures["assistance"].append((assistance_current, assistance_max))
        signatures["energy"].append(tuple(energy))
        signatures["communication_energy_total"].append(communication_energy)
        signatures["parent_changes_total"].append(parent_changes)

    admit(TERMINAL_BOUNDARY)
    signatures["knowledge"].append(_knowledge_signature(knowledge))
    signatures["operator"].append(
        (tuple(sorted(operator_knowledge.items())), tuple(sorted(operator_arrivals.items())))
    )
    signatures["requests"].append(_request_signature(robot_requests))
    return {
        "trace": reference_trace,
        "positions": tuple(positions),
        "knowledge": _knowledge_signature(knowledge),
        "operator_knowledge": tuple(sorted(operator_knowledge.items())),
        "operator_arrivals": dict(operator_arrivals),
        "team_discovery": dict(team_discovery),
        "robot_requests": _request_signature(robot_requests),
        "assistance_max": assistance_max,
        "energy": tuple(energy),
        "communication_energy": communication_energy,
        "parent_changes": parent_changes,
        "signatures": signatures,
    }


def _replay_difference(expected: Any, actual: Any, path: str) -> str | None:
    expected_number = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    actual_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if expected_number or actual_number:
        if (
            expected_number
            and actual_number
            and math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=EPS)
        ):
            return None
        return f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}, EPS={EPS}"
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}"
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}.keys: expected={_safe_repr(sorted(expected_keys, key=repr))}, "
                f"actual={_safe_repr(sorted(actual_keys, key=repr))}"
            )
        for key in expected:
            difference = _replay_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}"
        if len(expected) != len(actual):
            return f"{path}.length: expected={len(expected)}, actual={len(actual)}, EPS={EPS}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = _replay_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected={_safe_repr(expected)}, actual={_safe_repr(actual)}"
    return None


def _timeline_difference(expected: Any, actual: Any, scene: str, field: str) -> str | None:
    if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
        return _replay_difference(expected, actual, f"{scene}.{field}")
    if len(expected) != len(actual):
        return f"{scene}.{field}.length: expected={len(expected)}, actual={len(actual)}, EPS={EPS}"
    for boundary, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
        difference = _replay_difference(
            expected_item,
            actual_item,
            f"{scene}.boundary={boundary}.{field}",
        )
        if difference is not None:
            return difference
    return None


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    atoms: list[AtomVerdict] = []
    for name in ATOM_SHARES["decision_legality"]:
        short = name.split(".", 1)[1]
        failures: list[str] = []
        for scene in SCENES:
            failures.extend(replayed[scene]["violations"][short])
        atoms.append(
            _atom(
                "decision_legality",
                short,
                not failures,
                "all attributable decisions satisfy this rule"
                if not failures
                else "; ".join(failures[:4]),
            )
        )
    return atoms


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    references = {scene: _reference_scene(scene, records) for scene in SCENES}
    fields = {
        "decision_consumption": ("trace",),
        "movement_discovery": ("positions", "discoveries"),
        "topology_activation": ("stored", "active"),
        "map_propagation": ("knowledge", "operator"),
        "request_service": ("requests", "assistance"),
        "energy_accounting": (
            "energy",
            "communication_energy_total",
            "parent_changes_total",
        ),
    }
    differences: dict[str, str | None] = {name: None for name in fields}
    for scene in SCENES:
        primary_scene = replayed.get(scene, {})
        reference_scene = references[scene]
        primary_signatures = primary_scene.get("signatures", {})
        reference_signatures = reference_scene["signatures"]
        for name, names in fields.items():
            if differences[name] is not None:
                continue
            for field in names:
                if field == "trace":
                    expected = [row["executed"] for row in reference_scene["trace"]]
                    primary_trace = primary_scene.get("trace", [])
                    actual = (
                        [
                            row.get("executed") if isinstance(row, Mapping) else row
                            for row in primary_trace
                        ]
                        if isinstance(primary_trace, list)
                        else primary_trace
                    )
                    difference = _timeline_difference(expected, actual, scene, "executed_decisions")
                else:
                    expected = reference_signatures[field]
                    actual = primary_signatures.get(field)
                    difference = _timeline_difference(
                        expected, actual, scene, f"signatures.{field}"
                    )
                if difference is not None:
                    differences[name] = difference
                    break

    return [
        _atom(
            "replay_validity",
            name,
            differences[name] is None,
            "primary replay and independent raw-decision replay agree"
            if differences[name] is None
            else str(differences[name]),
        )
        for name in fields
    ]


def _scene_metrics(scene: str, replayed: Mapping[str, Any]) -> dict[str, float]:
    values = SCENE_DATA[scene]["values"]
    total_value = float(sum(values))
    discoveries = replayed["team_discovery"]
    arrivals = replayed["operator_arrivals"]
    timely_value = sum(
        values[node]
        for node in NODES
        if node in discoveries
        and node in arrivals
        and arrivals[node] - discoveries[node] <= TIMELY_BOUND
    )
    primary_1 = timely_value / total_value

    request_qualities: dict[str, float] = {}
    for request_id in ("R0", "R1"):
        _, _, deadline, targets = SCENE_DATA[scene]["requests"][request_id]
        denominator = float(sum(values[node] for node in targets))
        numerator = sum(
            values[node] for node in targets if node in arrivals and arrivals[node] <= int(deadline)
        )
        request_qualities[request_id] = numerator / denominator
    request_qualities["R2"] = min(1.0, float(replayed["assistance_max"]) / ASSISTANCE_SLOTS)
    primary_2 = (
        request_qualities["R0"] + 2 * request_qualities["R1"] + 2 * request_qualities["R2"]
    ) / 5.0

    fresh_value = 0.0
    for node in NODES:
        if node not in discoveries or node not in arrivals:
            continue
        delay = arrivals[node] - discoveries[node]
        fresh_value += values[node] * min(1.0, max(0.0, 1.0 - delay / TIMELY_BOUND))
    auxiliary_1 = fresh_value / total_value

    denominator = timely_value + total_value * (
        float(replayed["communication_energy"]) / (len(ROBOTS) * INITIAL_ENERGY)
        + float(replayed["parent_changes"]) / (len(ROBOTS) * HORIZON)
    )
    auxiliary_2 = timely_value / denominator if denominator > 0.0 else 0.0
    return {
        "primary_metrics_1": primary_1,
        "primary_metrics_2": primary_2,
        "auxiliary_metrics_1": auxiliary_1,
        "auxiliary_metrics_2": auxiliary_2,
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    per_scene = [_scene_metrics(scene, replayed[scene]) for scene in SCENES]
    result = {
        name: sum(item[name] for item in per_scene) / len(per_scene)
        for name in (
            "primary_metrics_1",
            "primary_metrics_2",
            "auxiliary_metrics_1",
            "auxiliary_metrics_2",
        )
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError(f"non-finite or out-of-range task metric: {result}")
    return result


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8 * metrics["primary_metrics_1"]
        + 7 * metrics["primary_metrics_2"]
        + 3 * metrics["auxiliary_metrics_1"]
        + 2 * metrics["auxiliary_metrics_2"]
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
        return 0.0, f"report read failed with {type(exc).__name__}: {exc}"
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
        return 0.0, f"report read failed with {type(exc).__name__}: {exc}"


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
            return 0.0, (f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0")
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
