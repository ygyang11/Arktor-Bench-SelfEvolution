# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 84
GRID_X = 24
GRID_Y = 18
N_TASK = 3
N_MID = 2
NODES = ("t0", "t1", "t2", "m0", "m1")
BASE = "base"
ENERGY0 = 120.0
TASK_MOVE_COST = 1.0
MID_MOVE_COST = 1.5
STAY_COST = 0.2
SEND_COST = 0.5
QUEUE_CAPACITY = 6
MAX_HOPS = 4
MAX_ATTEMPTS = 2
SLOT_CAPACITY = 6
SCENARIO_BUDGET = 504
EPS = 1e-6
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
SCENARIO_NAMES = (
    "dispersing_patrol",
    "crossing_flows",
    "support_gap",
    "priority_burst",
    "moving_endpoints",
)
ACTION_KEYS = frozenset({"slot", "node", "dx", "dy", "send_unit", "relay_candidates"})
SCENARIO_KEYS = frozenset({"scenario", "actions"})
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
GATE_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.scenario_rows": 0.15,
        "result_evidence.action_rows": 0.25,
        "result_evidence.action_values": 0.35,
        "result_evidence.request_values": 0.25,
    },
    "decision_legality": {
        "decision_legality.motion": 0.25,
        "decision_legality.requests": 0.25,
        "decision_legality.relays": 0.25,
        "decision_legality.energy": 0.25,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.packets": 0.25,
        "replay_validity.queues": 0.20,
        "replay_validity.mission": 0.20,
        "replay_validity.metrics": 0.15,
    },
}


def _scenario(
    task_starts: tuple[tuple[int, int], ...],
    mid_starts: tuple[tuple[int, int], ...],
    station: tuple[int, int],
    obstacle: tuple[int, int, int, int],
    goals: tuple[tuple[tuple[int, int, int, int], ...], ...],
    flows: tuple[tuple[int, int, int], ...],
    radius: int,
) -> dict[str, Any]:
    return {
        "task_starts": task_starts,
        "mid_starts": mid_starts,
        "station": station,
        "obstacles": (obstacle,),
        "goals": goals,
        "flows": flows,
        "radius": radius,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "dispersing_patrol": _scenario(
        ((2, 2), (2, 9), (2, 16)),
        ((11, 2), (11, 16)),
        (22, 9),
        (8, 9, 4, 13),
        (
            ((6, 2, 2, 28), (10, 4, 2, 42), (16, 2, 3, 58)),
            ((6, 9, 2, 30), (12, 9, 3, 48), (18, 12, 3, 66)),
            ((6, 16, 2, 32), (14, 16, 3, 54), (20, 14, 4, 72)),
        ),
        ((6, 2, 40), (6, 2, 48), (6, 1, 56)),
        6,
    ),
    "crossing_flows": _scenario(
        ((2, 3), (2, 14), (20, 3)),
        ((10, 3), (10, 15)),
        (22, 9),
        (9, 10, 6, 11),
        (
            ((7, 12, 2, 34), (12, 6, 3, 50), (18, 12, 3, 68)),
            ((7, 6, 2, 36), (12, 12, 3, 52), (18, 6, 3, 70)),
            ((8, 9, 2, 40), (14, 3, 3, 58), (20, 9, 4, 76)),
        ),
        ((8, 3, 52), (8, 2, 60), (8, 1, 68)),
        5,
    ),
    "support_gap": _scenario(
        ((3, 2), (3, 15), (19, 9)),
        ((4, 9), (17, 9)),
        (22, 9),
        (7, 8, 3, 14),
        (
            ((8, 2, 2, 30), (12, 4, 3, 46), (18, 4, 3, 64)),
            ((8, 15, 2, 32), (12, 14, 3, 50), (18, 14, 4, 70)),
            ((10, 9, 2, 36), (14, 9, 3, 56), (20, 9, 4, 78)),
        ),
        ((7, 2, 50), (7, 2, 60), (7, 3, 72)),
        5,
    ),
    "priority_burst": _scenario(
        ((2, 2), (2, 16), (20, 9)),
        ((11, 4), (11, 14)),
        (22, 9),
        (10, 11, 5, 13),
        (
            ((6, 3, 2, 26), (10, 3, 3, 40), (16, 3, 4, 58)),
            ((6, 15, 2, 28), (10, 15, 3, 42), (16, 15, 4, 60)),
            ((12, 9, 5, 34), (18, 9, 5, 46), (21, 9, 5, 62)),
        ),
        ((10, 1, 46), (10, 1, 48), (14, 3, 54)),
        6,
    ),
    "moving_endpoints": _scenario(
        ((2, 5), (2, 13), (20, 9)),
        ((10, 5), (10, 13)),
        (22, 9),
        (6, 7, 6, 12),
        (
            ((7, 5, 2, 30), (13, 5, 3, 48), (19, 5, 4, 66)),
            ((7, 13, 2, 32), (13, 13, 3, 50), (19, 13, 4, 68)),
            ((11, 9, 3, 36), (16, 9, 4, 54), (21, 9, 5, 76)),
        ),
        ((8, 2, 52), (8, 2, 58), (10, 3, 70)),
        4,
    ),
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id = ident
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
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_PAIR_DISTANCE_CACHE: dict[tuple[str, tuple[int, int], tuple[int, int]], int] = {}
_STATION_DISTANCE_CACHE: dict[str, dict[tuple[int, int], int]] = {}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _brief(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _blocked(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    x, y = position
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in spec["obstacles"])


def _traversable(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    return 0 <= position[0] < GRID_X and 0 <= position[1] < GRID_Y and not _blocked(spec, position)


def _cell_value(spec: Mapping[str, Any], cell: tuple[int, int]) -> int:
    values = [goal[2] for goals in spec["goals"] for goal in goals if goal[:2] == cell]
    return max(values, default=1)


def _distance_map(name: str) -> dict[tuple[int, int], int]:
    if name in _STATION_DISTANCE_CACHE:
        return _STATION_DISTANCE_CACHE[name]
    spec = SCENARIOS[name]
    target = tuple(spec["station"])
    distances = {target: 0}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for dx, dy in MOVES[1:]:
            nxt = current[0] + dx, current[1] + dy
            if nxt not in distances and _traversable(spec, nxt):
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    _STATION_DISTANCE_CACHE[name] = distances
    return distances


def _pair_distance(name: str, start: tuple[int, int], target: tuple[int, int]) -> int:
    key = (name, start, target)
    if key in _PAIR_DISTANCE_CACHE:
        return _PAIR_DISTANCE_CACHE[key]
    spec = SCENARIOS[name]
    distances = {start: 0}
    queue = deque([start])
    while queue and target not in distances:
        current = queue.popleft()
        for dx, dy in MOVES[1:]:
            nxt = current[0] + dx, current[1] + dy
            if nxt not in distances and _traversable(spec, nxt):
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    value = distances.get(target, GRID_X * GRID_Y + 1)
    _PAIR_DISTANCE_CACHE[key] = value
    _PAIR_DISTANCE_CACHE[(name, target, start)] = value
    return value


def _release_boundary(name: str, source: int, unit: int) -> int:
    if name == "priority_burst":
        return 2 * unit if source < 2 else 18 + unit
    return 3 * unit


def _unit_id(source: int, unit: int) -> str:
    return f"t{source}:u{unit}"


def _parse_unit(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str) or not re.fullmatch(r"t[0-2]:u[0-9]+", value):
        return None
    source, unit = value.split(":u")
    return int(source[1:]), int(unit)


def _goal_threshold(unit_count: int, index: int) -> int:
    return (unit_count * (index + 1) + 2) // 3


def _relay_sequence(unit: Mapping[str, Any]) -> tuple[str, ...]:
    source = f"t{unit['source']}"
    return tuple(node for node in unit["path"] if node not in {source, BASE})


def _shape_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    root_keys = root and set(data) == {"scenarios"}
    rows = data.get("scenarios") if root else None
    list_ok = isinstance(rows, list)
    records_ok = bool(
        list_ok
        and all(
            isinstance(item, Mapping)
            and set(item) == SCENARIO_KEYS
            and isinstance(item.get("actions"), list)
            for item in rows
        )
    )
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parse"),
        _atom("result_json", "root", bool(root_keys), f"expected exact root keys ['scenarios']; got {_brief(data)}"),
        _atom("result_json", "scenarios", list_ok, f"scenarios={_brief(rows)}; expected array"),
        _atom("result_json", "records", records_ok, "scenario records must have exact keys and action arrays"),
    ]


def _index(data: Any) -> dict[str, Mapping[str, Any]]:
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    result: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for item in rows:
        if isinstance(item, Mapping) and isinstance(item.get("scenario"), str):
            result.setdefault(item["scenario"], item)
    return result


def _typed_action(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and set(row) == ACTION_KEYS
        and _integer(row.get("slot"))
        and isinstance(row.get("node"), str)
        and _integer(row.get("dx"))
        and _integer(row.get("dy"))
        and (row.get("send_unit") is None or isinstance(row.get("send_unit"), str))
        and isinstance(row.get("relay_candidates"), list)
        and all(isinstance(value, str) for value in row["relay_candidates"])
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, Mapping) else []
    rows = rows if isinstance(rows, list) else []
    scenario_errors: list[str] = []
    action_errors: list[str] = []
    value_errors: list[str] = []
    request_errors: list[str] = []
    for scenario_index, item in enumerate(rows):
        if not isinstance(item, Mapping) or not isinstance(item.get("scenario"), str):
            scenario_errors.append(f"scenarios[{scenario_index}].scenario must be a string")
        actions = item.get("actions") if isinstance(item, Mapping) else []
        if not isinstance(actions, list):
            action_errors.append(f"scenarios[{scenario_index}].actions must be an array")
            continue
        for action_index, row in enumerate(actions):
            path = f"scenarios[{scenario_index}].actions[{action_index}]"
            if not isinstance(row, Mapping) or set(row) != ACTION_KEYS:
                action_errors.append(f"{path} must have exact keys {sorted(ACTION_KEYS)}")
                continue
            if not (_integer(row.get("slot")) and isinstance(row.get("node"), str)):
                value_errors.append(f"{path}.slot/node have invalid types")
            if not (_integer(row.get("dx")) and _integer(row.get("dy"))):
                value_errors.append(f"{path}.dx/dy must be JSON integers")
            if row.get("send_unit") is not None and not isinstance(row.get("send_unit"), str):
                request_errors.append(f"{path}.send_unit must be string or null")
            candidates = row.get("relay_candidates")
            if not isinstance(candidates, list) or not all(isinstance(x, str) for x in candidates):
                request_errors.append(f"{path}.relay_candidates must be a string array")
    return [
        _atom("result_evidence", "scenario_rows", not scenario_errors, "passed" if not scenario_errors else " | ".join(scenario_errors[:8])),
        _atom("result_evidence", "action_rows", not action_errors, "passed" if not action_errors else " | ".join(action_errors[:8])),
        _atom("result_evidence", "action_values", not value_errors, "passed" if not value_errors else " | ".join(value_errors[:8])),
        _atom("result_evidence", "request_values", not request_errors, "passed" if not request_errors else " | ".join(request_errors[:8])),
    ]


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = data.get("scenarios") if isinstance(data, Mapping) else []
    rows = rows if isinstance(rows, list) else []
    counts: dict[str, int] = {}
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping) or not isinstance(item.get("scenario"), str):
            defects.append(f"scenarios[{index}] has no string scenario key")
            continue
        name = item["scenario"]
        counts[name] = counts.get(name, 0) + 1
        if name not in SCENARIO_NAMES:
            defects.append(f"unknown scenario {name!r}")
    for name in SCENARIO_NAMES:
        if counts.get(name, 0) == 0:
            defects.append(f"missing scenario {name}")
        elif counts[name] > 1:
            defects.extend([f"duplicate scenario {name}"] * (counts[name] - 1))
    indexed = _index(data)
    for name in SCENARIO_NAMES:
        item = indexed.get(name)
        if item is None:
            continue
        action_counts: dict[tuple[int, str], int] = {}
        actions = item.get("actions")
        for action_index, row in enumerate(actions if isinstance(actions, list) else []):
            if not isinstance(row, Mapping) or not _integer(row.get("slot")) or not isinstance(row.get("node"), str):
                defects.append(f"{name}.actions[{action_index}] has no canonical (slot,node) key")
                continue
            key = (row["slot"], row["node"])
            action_counts[key] = action_counts.get(key, 0) + 1
            if key[0] not in range(HORIZON) or key[1] not in NODES:
                defects.append(f"{name} unknown action key {key}")
        for slot in range(HORIZON):
            for node in NODES:
                count = action_counts.get((slot, node), 0)
                if count == 0:
                    defects.append(f"{name} missing action {(slot, node)}")
                elif count > 1:
                    defects.extend([f"{name} duplicate action {(slot, node)}"] * (count - 1))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _actions(record: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    rows = record.get("actions")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and _integer(row.get("slot")) and isinstance(row.get("node"), str):
            result.setdefault((row["slot"], row["node"]), row)
    return result


def _blank_action(slot: int, node: str) -> dict[str, Any]:
    return {"slot": slot, "node": node, "dx": 0, "dy": 0, "send_unit": None, "relay_candidates": []}


def _node_position(spec: Mapping[str, Any], positions: Sequence[tuple[int, int]], node: str) -> tuple[int, int]:
    return tuple(spec["station"]) if node == BASE else positions[NODES.index(node)]


def _unit_sort_key(unit: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    source, number = unit["source"], unit["number"]
    return (spec["flows"][source][2], -spec["flows"][source][1], unit["release"], source, number)


def _relay_error(
    name: str,
    spec: Mapping[str, Any],
    positions: Sequence[tuple[int, int]],
    sender: str,
    candidate: str,
    unit: Mapping[str, Any],
) -> str | None:
    if candidate not in (*NODES, BASE):
        return f"candidate={candidate!r}; expected one of {list((*NODES, BASE))}"
    if candidate == sender:
        return f"candidate={candidate!r}; expected a node different from sender={sender}"
    sender_position = _node_position(spec, positions, sender)
    receiver_position = _node_position(spec, positions, candidate)
    distance = abs(sender_position[0] - receiver_position[0]) + abs(
        sender_position[1] - receiver_position[1]
    )
    if distance > spec["radius"]:
        return f"candidate={candidate}, distance={distance}; expected <= radius={spec['radius']}"
    distances = _distance_map(name)
    sender_distance = distances[sender_position]
    receiver_distance = distances[receiver_position]
    if receiver_distance >= sender_distance:
        return (
            f"candidate={candidate}, station_distance={receiver_distance}; expected strictly less "
            f"than sender {sender} distance={sender_distance}"
        )
    if candidate != BASE and unit["hops"] >= MAX_HOPS - 1:
        return (
            f"candidate={candidate}, prior_hops={unit['hops']}; mobile relay requires prior_hops "
            f"< {MAX_HOPS - 1}"
        )
    if candidate == BASE and unit["hops"] >= MAX_HOPS:
        return f"candidate=base, prior_hops={unit['hops']}; expected prior_hops < {MAX_HOPS}"
    return None


def _replay(
    name: str, record: Mapping[str, Any], stop: int = HORIZON
) -> dict[str, Any]:
    if stop not in range(HORIZON + 1):
        raise ValueError(f"stop={stop}; expected integer boundary 0..{HORIZON}")
    spec = SCENARIOS[name]
    rows = _actions(record)
    errors: dict[str, list[str]] = {
        key: [] for key in ("motion", "requests", "relays", "energy")
    }
    positions = [*spec["task_starts"], *spec["mid_starts"]]
    energies = [ENERGY0] * len(NODES)
    units: dict[str, dict[str, Any]] = {}
    queues: list[list[str]] = [[] for _ in NODES]
    for source, (count, priority, deadline) in enumerate(spec["flows"]):
        for number in range(count):
            ident = _unit_id(source, number)
            units[ident] = {
                "id": ident,
                "source": source,
                "number": number,
                "priority": priority,
                "deadline": deadline,
                "release": _release_boundary(name, source, number),
                "carrier": None,
                "hops": 0,
                "attempts": 0,
                "path": [],
                "status": "pending",
                "arrival": None,
                "drop_reason": None,
            }

    def release(boundary: int) -> None:
        for unit in units.values():
            if unit["status"] == "pending" and unit["release"] == boundary:
                unit["carrier"] = unit["source"]
                queues[unit["source"]].append(unit["id"])

    def retain() -> int:
        dropped = 0
        for node_index, queue in enumerate(queues):
            eligible = [
                ident
                for ident in queue
                if units[ident]["status"] == "pending"
                and units[ident]["carrier"] == node_index
            ]
            eligible = sorted(set(eligible), key=lambda ident: _unit_sort_key(units[ident], spec))
            queues[node_index] = eligible[:QUEUE_CAPACITY]
            for ident in eligible[QUEUE_CAPACITY:]:
                units[ident]["status"] = "dropped"
                units[ident]["drop_reason"] = "queue"
                units[ident]["carrier"] = None
                dropped += 1
        return dropped

    release(0)
    dropped_units = retain()
    coverage: set[tuple[int, int]] = set(spec["task_starts"])
    goal_index = [0] * N_TASK
    completed_goals: list[list[int]] = [[] for _ in range(N_TASK)]
    blocked_goals = [False] * N_TASK
    delivered_counts = [0] * N_TASK
    position_history = [tuple(positions)]
    energy_history = [tuple(energies)]
    queue_history = [tuple(tuple(queue) for queue in queues)]
    safety_events = 0
    mid_moves = 0
    task_moves = 0
    admitted_attempts = 0

    for slot in range(stop):
        action_start_positions = tuple(positions)
        action_rows = {
            node: rows.get((slot, node), _blank_action(slot, node)) for node in NODES
        }
        valid_requests: dict[str, dict[str, Any]] = {}
        requested_units: set[str] = set()
        for node_index, node in enumerate(NODES):
            row = action_rows[node]
            if not _typed_action(row):
                continue
            move = row["dx"], row["dy"]
            if move not in MOVES:
                errors["motion"].append(
                    f"{name}.actions[slot={slot},node={node}].move={move}; expected one of {list(MOVES)}"
                )
            send = row["send_unit"]
            candidates = row["relay_candidates"]
            if send is None:
                if candidates:
                    errors["requests"].append(
                        f"{name}.actions[slot={slot},node={node}] has candidates={candidates} with null send_unit"
                    )
                continue
            parsed = _parse_unit(send)
            if parsed is None or send not in units:
                errors["requests"].append(
                    f"{name}.actions[slot={slot},node={node}].send_unit={send!r}; expected a canonical unit"
                )
                continue
            if send in requested_units:
                errors["requests"].append(
                    f"{name}.actions[slot={slot}] unit={send} requested more than once"
                )
                continue
            if units[send]["carrier"] != node_index or send not in queues[node_index]:
                errors["requests"].append(
                    f"{name}.actions[slot={slot},node={node}].send_unit={send}; expected an action-start queued unit"
                )
                continue
            if not 1 <= len(candidates) <= 3 or len(set(candidates)) != len(candidates):
                errors["relays"].append(
                    f"{name}.actions[slot={slot},node={node}].relay_candidates={candidates}; expected 1..3 distinct IDs"
                )
                continue
            candidate_errors = [
                issue
                for candidate in candidates
                if (
                    issue := _relay_error(
                        name,
                        spec,
                        action_start_positions,
                        node,
                        candidate,
                        units[send],
                    )
                )
                is not None
            ]
            if candidate_errors:
                errors["relays"].extend(
                    f"{name}.actions[slot={slot},node={node}].{issue}"
                    for issue in candidate_errors
                )
                continue
            requested_units.add(send)
            valid_requests[send] = {
                "node": node,
                "node_index": node_index,
                "candidates": tuple(candidates),
            }

        entries: list[tuple[tuple[int, int, int, int, int], int, int, str, str]] = []
        for ident, request in valid_requests.items():
            for list_index, candidate in enumerate(request["candidates"]):
                entries.append(
                    (
                        _unit_sort_key(units[ident], spec),
                        request["node_index"],
                        list_index,
                        ident,
                        candidate,
                    )
                )
        entries.sort()
        remaining = max(0, SCENARIO_BUDGET - admitted_attempts)
        admitted = entries[: min(SLOT_CAPACITY, remaining)]
        admitted_attempts += len(admitted)
        admitted_by_unit: dict[str, list[tuple[int, str]]] = {}
        for _priority, _sender_index, list_index, ident, candidate in admitted:
            admitted_by_unit.setdefault(ident, []).append((list_index, candidate))
        concurrent_senders = len(admitted_by_unit)
        selected_receiver: dict[str, str | None] = {}
        for ident, candidates in admitted_by_unit.items():
            unit = units[ident]
            request = valid_requests[ident]
            sender = request["node"]
            sender_position = _node_position(spec, action_start_positions, sender)
            successful: list[tuple[int, str]] = []
            for list_index, candidate in candidates:
                receiver_position = _node_position(spec, action_start_positions, candidate)
                distance = abs(sender_position[0] - receiver_position[0]) + abs(
                    sender_position[1] - receiver_position[1]
                )
                obstruction = int(
                    _pair_distance(name, sender_position, receiver_position) > distance
                )
                normalized_distance = min(1.0, distance / spec["radius"])
                probability = _clip(
                    0.95
                    - 0.20 * normalized_distance
                    - 0.10 * obstruction
                    - 0.05 * (concurrent_senders - 1)
                )
                sample_input = (
                    f"{name}|{slot}|{ident}|{sender}|{candidate}|{unit['hops']}|{unit['attempts']}"
                )
                sample = int(
                    hashlib.sha256(sample_input.encode("utf-8")).hexdigest()[:16], 16
                ) / 2**64
                if sample < probability:
                    successful.append((list_index, candidate))
            selected_receiver[ident] = min(successful)[1] if successful else None
            energies[request["node_index"]] -= SEND_COST

        accepted_destinations: list[tuple[int, int]] = []
        next_positions: list[tuple[int, int]] = []
        for node_index, node in enumerate(NODES):
            row = action_rows[node]
            move = row.get("dx"), row.get("dy")
            if move not in MOVES:
                move = (0, 0)
            current = action_start_positions[node_index]
            destination = current[0] + move[0], current[1] + move[1]
            rejected = bool(
                not _traversable(spec, destination)
                or destination in accepted_destinations
                or destination in action_start_positions[node_index + 1 :]
            )
            if rejected:
                destination = current
                if move != (0, 0):
                    safety_events += 1
            accepted_destinations.append(destination)
            next_positions.append(destination)
        positions = next_positions
        for node_index in range(len(NODES)):
            moved = positions[node_index] != action_start_positions[node_index]
            if moved:
                energies[node_index] -= (
                    TASK_MOVE_COST if node_index < N_TASK else MID_MOVE_COST
                )
                task_moves += int(node_index < N_TASK)
                mid_moves += int(node_index >= N_TASK)
            else:
                energies[node_index] -= STAY_COST
            if energies[node_index] < -EPS:
                errors["energy"].append(
                    f"{name}.actions[slot={slot},node={NODES[node_index]}] resulting_energy={energies[node_index]:.9g}; expected >= {-EPS} with tolerance {EPS}"
                )

        for ident, request in valid_requests.items():
            if ident not in admitted_by_unit:
                continue
            unit = units[ident]
            sender_index = request["node_index"]
            receiver = selected_receiver[ident]
            if receiver is None:
                unit["attempts"] += 1
                if unit["attempts"] >= MAX_ATTEMPTS:
                    unit["status"] = "dropped"
                    unit["drop_reason"] = "retry"
                    unit["carrier"] = None
                    queues[sender_index] = [value for value in queues[sender_index] if value != ident]
                    dropped_units += 1
            else:
                queues[sender_index] = [value for value in queues[sender_index] if value != ident]
                unit["attempts"] = 0
                unit["hops"] += 1
                unit["path"].append(receiver)
                if receiver == BASE:
                    unit["status"] = "delivered"
                    unit["carrier"] = None
                    unit["arrival"] = slot + 1
                    delivered_counts[unit["source"]] += 1
                else:
                    receiver_index = NODES.index(receiver)
                    unit["carrier"] = receiver_index
                    queues[receiver_index].append(ident)

        release(slot + 1)
        dropped_units += retain()
        coverage.update(positions[:N_TASK])
        for task in range(N_TASK):
            if blocked_goals[task] or goal_index[task] >= len(spec["goals"][task]):
                continue
            x, y, _value, deadline = spec["goals"][task][goal_index[task]]
            if slot + 1 > deadline:
                blocked_goals[task] = True
            elif positions[task] == (x, y) and delivered_counts[task] >= _goal_threshold(
                spec["flows"][task][0], goal_index[task]
            ):
                completed_goals[task].append(goal_index[task])
                goal_index[task] += 1
        position_history.append(tuple(positions))
        energy_history.append(tuple(energies))
        queue_history.append(tuple(tuple(queue) for queue in queues))

    goal_value = sum(
        spec["goals"][task][index][2]
        for task in range(N_TASK)
        for index in completed_goals[task]
    )
    total_goal_value = sum(goal[2] for goals in spec["goals"] for goal in goals)
    total_coverage_value = sum(
        _cell_value(spec, (x, y))
        for y in range(GRID_Y)
        for x in range(GRID_X)
        if _traversable(spec, (x, y))
    )
    coverage_value = sum(_cell_value(spec, cell) for cell in coverage)
    delivered = tuple(unit for unit in units.values() if unit["status"] == "delivered")
    return {
        "positions": tuple(position_history),
        "energies": tuple(energy_history),
        "queues": tuple(queue_history),
        "units": units,
        "delivered": delivered,
        "goal_value": goal_value,
        "total_goal_value": total_goal_value,
        "coverage_value": coverage_value,
        "total_coverage_value": total_coverage_value,
        "completed_goals": tuple(tuple(values) for values in completed_goals),
        "safety_events": safety_events,
        "mid_moves": mid_moves,
        "task_moves": task_moves,
        "admitted_attempts": admitted_attempts,
        "dropped_units": dropped_units,
        "energy_used": len(NODES) * ENERGY0 - sum(energies),
        "errors": errors,
    }


def _legality(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "requests", "relays", "energy")
    }
    for name in SCENARIO_NAMES:
        record = scenarios.get(name)
        if not isinstance(record, Mapping):
            continue
        replayed = _replay(name, record)
        for key, details in replayed["errors"].items():
            failures[key].extend(details)

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("decision_legality", "motion", not failures["motion"], detail("motion")),
        _atom("decision_legality", "requests", not failures["requests"], detail("requests")),
        _atom("decision_legality", "relays", not failures["relays"], detail("relays")),
        _atom("decision_legality", "energy", not failures["energy"], detail("energy")),
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name in SCENARIO_NAMES:
        state = replayed[name]
        spec = SCENARIOS[name]
        quality_numerator = 0.0
        quality_denominator = 0.0
        for source, (count, priority, deadline) in enumerate(spec["flows"]):
            for number in range(count):
                unit = state["units"][_unit_id(source, number)]
                if unit["status"] != "delivered":
                    quality = 0.0
                elif unit["arrival"] <= deadline:
                    quality = 1.0
                else:
                    quality = 0.5 * _clip(
                        1.0
                        - (unit["arrival"] - deadline)
                        / max(1, HORIZON - deadline)
                    )
                quality_numerator += priority * quality
                quality_denominator += priority
        totals["primary_metrics_1"] += _clip(
            quality_numerator / max(1.0, quality_denominator)
        )

        goal_progress = state["goal_value"] / max(1, state["total_goal_value"])
        coverage_progress = state["coverage_value"] / max(
            1, state["total_coverage_value"]
        )
        totals["primary_metrics_2"] += _clip(
            0.6 * goal_progress + 0.4 * coverage_progress
        )

        totals["primary_metrics_3"] += _clip(
            0.4
            * _clip(1.0 - state["safety_events"] / (HORIZON * len(NODES)))
            + 0.3 * _clip(1.0 - state["mid_moves"] / (HORIZON * N_MID))
            + 0.3
            * _clip(1.0 - state["energy_used"] / (len(NODES) * ENERGY0))
        )

        by_flow: dict[int, list[Mapping[str, Any]]] = {
            source: [] for source in range(N_TASK)
        }
        for unit in state["delivered"]:
            by_flow[unit["source"]].append(unit)
        diverse = 0
        for delivered in by_flow.values():
            paths = {_relay_sequence(unit) for unit in delivered}
            if len(delivered) >= 2 and len(paths) >= 2:
                diverse += 1
        mean_hops = (
            sum(unit["hops"] for unit in state["delivered"])
            / len(state["delivered"])
            if state["delivered"]
            else 4.0
        )
        totals["auxiliary_metrics_1"] += _clip(
            0.5 * diverse / N_TASK
            + 0.5 * _clip(1.0 - (mean_hops - 1.0) / 3.0)
        )

        total_units = sum(flow[0] for flow in spec["flows"])
        totals["auxiliary_metrics_2"] += _clip(
            0.5
            * _clip(1.0 - state["admitted_attempts"] / SCENARIO_BUDGET)
            + 0.5
            * _clip(1.0 - state["dropped_units"] / max(1, total_units))
        )
    return {
        key: _clip(value / len(SCENARIO_NAMES)) for key, value in totals.items()
    }


def _replay_atoms(
    scenarios: Mapping[str, Any], replayed: Mapping[str, Any]
) -> list[AtomVerdict]:
    del scenarios
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "packets", "queues", "mission", "metrics")
    }
    if set(replayed) != set(SCENARIO_NAMES):
        missing = sorted(set(SCENARIO_NAMES) - set(replayed))
        unknown = sorted(set(replayed) - set(SCENARIO_NAMES))
        for key in failures:
            failures[key].append(
                f"replayed scenario keys missing={missing}, unknown={unknown}; expected {list(SCENARIO_NAMES)}"
            )

    for name in SCENARIO_NAMES:
        state = replayed.get(name)
        if not isinstance(state, Mapping):
            continue
        spec = SCENARIOS[name]
        positions = state.get("positions", ())
        energies = state.get("energies", ())
        if len(positions) != HORIZON + 1:
            failures["motion"].append(
                f"{name}.positions length={len(positions)}; expected {HORIZON + 1}"
            )
        if len(energies) != HORIZON + 1:
            failures["motion"].append(
                f"{name}.energies length={len(energies)}; expected {HORIZON + 1}"
            )
        for boundary, values in enumerate(positions):
            if len(values) != len(NODES) or len(set(values)) != len(NODES):
                failures["motion"].append(
                    f"{name}.positions[{boundary}]={_brief(values)}; expected {len(NODES)} distinct positions"
                )
                continue
            invalid = [position for position in values if not _traversable(spec, tuple(position))]
            if invalid:
                failures["motion"].append(
                    f"{name}.positions[{boundary}] invalid={invalid}; expected traversable cells"
                )
        for boundary, values in enumerate(energies):
            invalid = [
                (node, value)
                for node, value in zip(NODES, values, strict=False)
                if not _finite_number(value) or value < -EPS or value > ENERGY0 + EPS
            ]
            if len(values) != len(NODES) or invalid:
                failures["motion"].append(
                    f"{name}.energies[{boundary}] invalid={invalid}; expected {len(NODES)} finite balances in [{-EPS},{ENERGY0}] with tolerance {EPS}"
                )

        units = state.get("units", {})
        expected_units = {
            _unit_id(source, number)
            for source, flow in enumerate(spec["flows"])
            for number in range(flow[0])
        }
        if not isinstance(units, Mapping) or set(units) != expected_units:
            failures["packets"].append(
                f"{name}.unit keys missing={sorted(expected_units - set(units)) if isinstance(units, Mapping) else sorted(expected_units)}, unknown={sorted(set(units) - expected_units) if isinstance(units, Mapping) else []}"
            )
        if isinstance(units, Mapping):
            for ident, unit in units.items():
                if not isinstance(unit, Mapping):
                    failures["packets"].append(
                        f"{name}.{ident}={_brief(unit)}; expected replayed unit object"
                    )
                    continue
                status = unit.get("status")
                if status not in {"pending", "delivered", "dropped"}:
                    failures["packets"].append(
                        f"{name}.{ident}.status={status!r}; expected pending/delivered/dropped"
                    )
                hops = unit.get("hops")
                path = unit.get("path")
                if not _integer(hops) or hops not in range(MAX_HOPS + 1):
                    failures["packets"].append(
                        f"{name}.{ident}.hops={hops!r}; expected integer 0..{MAX_HOPS}"
                    )
                if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or len(path) != hops:
                    failures["packets"].append(
                        f"{name}.{ident}.path={_brief(path)}; expected length equal to hops={hops!r}"
                    )
                if status == "delivered" and (
                    not _integer(unit.get("arrival"))
                    or not unit.get("path")
                    or unit["path"][-1] != BASE
                    or unit["arrival"] not in range(unit["release"] + 1, HORIZON + 1)
                ):
                    failures["packets"].append(
                        f"{name}.{ident} delivered arrival/path={unit.get('arrival')!r}/{_brief(unit.get('path'))}; expected base-terminated path and boundary {unit['release'] + 1}..{HORIZON}"
                    )

        queues = state.get("queues", ())
        if len(queues) != HORIZON + 1:
            failures["queues"].append(
                f"{name}.queues length={len(queues)}; expected {HORIZON + 1} boundaries"
            )
        else:
            for boundary, snapshot in enumerate(queues):
                flat: list[str] = []
                if len(snapshot) != len(NODES):
                    failures["queues"].append(
                        f"{name}.queues[{boundary}] node_count={len(snapshot)}; expected {len(NODES)}"
                    )
                    continue
                for node_index, queue in enumerate(snapshot):
                    flat.extend(queue)
                    if len(queue) > QUEUE_CAPACITY:
                        failures["queues"].append(
                            f"{name}.queues[{boundary}][{NODES[node_index]}] length={len(queue)}; expected <= {QUEUE_CAPACITY}"
                        )
                if len(flat) != len(set(flat)):
                    failures["queues"].append(
                        f"{name}.queues[{boundary}] has duplicate live unit carriers"
                    )
        attempts = state.get("admitted_attempts")
        dropped = state.get("dropped_units")
        if not _integer(attempts) or attempts not in range(SCENARIO_BUDGET + 1):
            failures["queues"].append(
                f"{name}.admitted_attempts={attempts!r}; expected integer 0..{SCENARIO_BUDGET}"
            )
        if not _integer(dropped) or dropped not in range(len(expected_units) + 1):
            failures["queues"].append(
                f"{name}.dropped_units={dropped!r}; expected integer 0..{len(expected_units)}"
            )

        goal_value = state.get("goal_value")
        total_goal_value = state.get("total_goal_value")
        coverage_value = state.get("coverage_value")
        total_coverage_value = state.get("total_coverage_value")
        if not all(
            _integer(value)
            for value in (
                goal_value,
                total_goal_value,
                coverage_value,
                total_coverage_value,
            )
        ) or not (0 <= goal_value <= total_goal_value) or not (
            0 <= coverage_value <= total_coverage_value
        ):
            failures["mission"].append(
                f"{name} mission values goal={goal_value}/{total_goal_value}, coverage={coverage_value}/{total_coverage_value}; expected bounded integer values"
            )
        completed = state.get("completed_goals", ())
        if len(completed) != N_TASK or any(
            tuple(values) != tuple(range(len(values))) or len(values) > 3
            for values in completed
        ):
            failures["mission"].append(
                f"{name}.completed_goals={_brief(completed)}; expected three ordered prefixes of 0..2"
            )
        for field, maximum in (
            ("safety_events", HORIZON * len(NODES)),
            ("mid_moves", HORIZON * N_MID),
            ("task_moves", HORIZON * N_TASK),
        ):
            value = state.get(field)
            if not _integer(value) or value not in range(maximum + 1):
                failures["mission"].append(
                    f"{name}.{field}={value!r}; expected integer 0..{maximum}"
                )

    try:
        metrics = _metrics(replayed)
        invalid = [
            (key, value)
            for key, value in metrics.items()
            if not _finite_number(value) or not 0 <= value <= 1
        ]
        if set(metrics) != set(METRIC_KEYS) or invalid:
            failures["metrics"].append(
                f"metric keys/values={_brief(metrics)}; expected {list(METRIC_KEYS)} finite in [0,1]"
            )
    except Exception as exc:
        failures["metrics"].append(
            f"metric replay raised {type(exc).__name__}: {exc}; expected five finite values"
        )

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("replay_validity", "motion", not failures["motion"], detail("motion")),
        _atom("replay_validity", "packets", not failures["packets"], detail("packets")),
        _atom("replay_validity", "queues", not failures["queues"], detail("queues")),
        _atom("replay_validity", "mission", not failures["mission"], detail("mission")),
        _atom("replay_validity", "metrics", not failures["metrics"], detail("metrics")),
    ]


def _score_atoms(
    atoms: Sequence[AtomVerdict], expected: Mapping[str, float]
) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


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
            state.timed_out = state.exit_code is None and any(
                marker in output for marker in ("timeout", "timed out")
            )
        except Exception as exc:
            state.official_message = (
                f"official run raised {type(exc).__name__}: {exc}"
            )
            state.timed_out = isinstance(exc, TimeoutError) or any(
                marker in str(exc).lower() for marker in ("timeout", "timed out")
            )
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
        status = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
            f"timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        )
        state.official_message = (
            f"{state.official_message}; {status}"
            if state.official_message
            else status
        )

        parse_failure = ""
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                parse_failure = f"strict parse failed: {type(exc).__name__}: {exc}"

        shape = _shape_atoms(state.data, state.parse_ok)
        state.scenarios = _index(state.data)
        evidence = _evidence_atoms(state.data)
        coverage_score, coverage_defects = _coverage(state.data)
        legality = _legality(state.scenarios)
        state.scores["result_json"] = _score_atoms(
            shape, ATOM_SHARES["result_json"]
        )
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
        state.replayed = (
            {
                name: _replay(name, state.scenarios[name])
                for name in SCENARIO_NAMES
            }
            if can_replay
            else {}
        )
        replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        state.scores["replay_validity"] = _score_atoms(
            replay_atoms, ATOM_SHARES["replay_validity"]
        )

        state.diagnostics["result_json"] = _diagnose(shape)
        if parse_failure:
            state.diagnostics["result_json"] = (
                f"{parse_failure}; {state.diagnostics['result_json']}"
            )
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = (
            f"defect_count={len(coverage_defects)}; "
            + " | ".join(coverage_defects[:12])
            if coverage_defects
            else "defect_count=0"
        )
        state.diagnostics["decision_legality"] = _diagnose(legality)
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS
            for check_id in GATE_CHECKS
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
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


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular file"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return 0.0, f"report.md could not be read: {type(exc).__name__}: {exc}"
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    lines = text.splitlines()
    headings: list[tuple[int, int, int, str]] = []
    in_fence = False
    fence_character = ""
    fence_length = 0
    previous_visible = False
    for index, line in enumerate(lines):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                in_fence = False
            previous_visible = False
            continue
        if in_fence:
            previous_visible = False
            continue
        atx = re.match(
            r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$", line
        )
        if atx:
            headings.append((index, index + 1, len(atx.group(1)), atx.group(2).strip()))
            previous_visible = False
            continue
        setext = re.match(r"^ {0,3}(=+|-+)[ \t]*$", line)
        if setext and previous_visible and index > 0:
            title = lines[index - 1].strip()
            headings.append((index - 1, index + 1, 1 if line.strip()[0] == "=" else 2, title))
            previous_visible = False
            continue
        previous_visible = bool(line.strip()) and len(line) - len(line.lstrip(" ")) <= 3
    found: set[str] = set()
    for index, (heading_line, body_start, level, title) in enumerate(headings):
        if title not in required:
            continue
        end = len(lines)
        for later in headings[index + 1 :]:
            if later[2] <= level:
                end = later[0]
                break
        if any(line.strip() for line in lines[body_start:end]):
            found.add(title)
    score = max(0.0, 2.0 - 0.5 * (len(required) - len(found))) / 2.0
    return (
        score,
        f"missing_sections={[title for title in required if title not in found]}; normalized={score:.9g}",
    )


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

    globals()[_id] = check(id=_id)(_check)


for _id in METRIC_KEYS:

    async def _metric(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        value = (state.metrics or {}).get(ident, 0.0)
        detail = (
            f"{ident}: replayed value={value:.9g}"
            if state.metrics is not None
            else f"{ident}: validity gate not satisfied"
        )
        return value, detail

    globals()[_id] = check(id=_id)(_metric)


del _check, _metric
