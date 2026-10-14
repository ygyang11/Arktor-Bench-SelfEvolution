from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

GRID_X = 26
GRID_Y = 16
HORIZON = 90
N_ROBOTS = 4
ROBOTS = tuple(f"r{i}" for i in range(N_ROBOTS))
MOVES = ("stay", "N", "E", "S", "W")
CARDINAL_MOVES = ("N", "E", "S", "W")
MOVE_DELTAS = {
    "stay": (0, 0),
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}
ENCODINGS = {"compact": (1, 0.60), "full": (2, 1.0)}
CLASS_LIMITS = {"map": 12, "critical": 8, "alarm": 4}
CLASS_WEIGHTS = {"map": 1.0, "critical": 4.0, "alarm": 3.0}
QUEUE_CAPACITY = 12
SLOT_CAPACITY = 8
SCENARIO_BUDGET = HORIZON * SLOT_CAPACITY
ENERGY0 = 100.0
EPS = 1e-6
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.35,
        "result_json.root": 0.30,
        "result_json.scenario_rows": 0.35,
    },
    "result_evidence": {
        "result_evidence.scenario_rows": 0.20,
        "result_evidence.action_rows": 0.20,
        "result_evidence.action_values": 0.30,
        "result_evidence.request_values": 0.30,
    },
    "decision_legality": {
        "decision_legality.motion": 0.20,
        "decision_legality.deployment": 0.25,
        "decision_legality.requests": 0.25,
        "decision_legality.routes": 0.15,
        "decision_legality.energy": 0.15,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.deployments": 0.20,
        "replay_validity.packets": 0.25,
        "replay_validity.queues": 0.15,
        "replay_validity.mission": 0.20,
    },
}
GATE_CHECKS = (
    "official_run",
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)


def _scenario(
    starts: tuple[tuple[int, int], ...],
    base: tuple[int, int],
    obstacles: tuple[tuple[int, int, int, int], ...],
    regions: tuple[tuple[int, int, int, int], ...],
    interference: str,
) -> dict[str, Any]:
    return {
        "starts": starts,
        "base": base,
        "obstacles": obstacles,
        "regions": regions,
        "interference": interference,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "winding_mine": _scenario(
        ((1, 2), (1, 13), (24, 2), (24, 13)),
        (0, 7),
        ((6, 7, 0, 5), (6, 7, 10, 15), (15, 16, 4, 11)),
        (
            (3, 2, 1, 24),
            (3, 11, 1, 30),
            (9, 2, 2, 42),
            (9, 12, 2, 48),
            (18, 2, 3, 60),
            (18, 11, 3, 68),
            (22, 4, 4, 78),
            (22, 11, 4, 86),
        ),
        "none",
    ),
    "split_tunnel": _scenario(
        ((2, 2), (2, 13), (10, 2), (10, 13)),
        (0, 7),
        ((9, 10, 4, 11),),
        (
            (4, 2, 1, 26),
            (4, 11, 1, 32),
            (12, 2, 2, 44),
            (12, 11, 2, 50),
            (18, 3, 3, 62),
            (18, 10, 3, 68),
            (22, 4, 4, 80),
            (22, 10, 4, 88),
        ),
        "split",
    ),
    "relay_overlap": _scenario(
        ((1, 7), (4, 2), (4, 13), (9, 7)),
        (0, 7),
        ((7, 11, 0, 4), (7, 11, 11, 15)),
        (
            (3, 5, 1, 24),
            (3, 9, 1, 30),
            (12, 5, 2, 42),
            (12, 9, 2, 48),
            (17, 2, 3, 60),
            (17, 12, 3, 66),
            (22, 5, 4, 78),
            (22, 9, 4, 86),
        ),
        "overlap",
    ),
    "critical_burst": _scenario(
        ((2, 2), (2, 13), (23, 2), (23, 13)),
        (12, 7),
        ((8, 9, 5, 10),),
        (
            (4, 3, 1, 20),
            (4, 11, 1, 25),
            (10, 2, 2, 34),
            (10, 12, 2, 39),
            (17, 3, 3, 52),
            (17, 11, 3, 58),
            (22, 4, 4, 70),
            (22, 10, 4, 78),
        ),
        "none",
    ),
    "interference_shift": _scenario(
        ((1, 1), (1, 14), (24, 1), (24, 14)),
        (12, 7),
        ((5, 6, 5, 10), (18, 19, 5, 10)),
        (
            (3, 2, 1, 24),
            (3, 12, 1, 30),
            (9, 2, 2, 44),
            (9, 12, 2, 50),
            (15, 3, 3, 64),
            (15, 11, 3, 70),
            (22, 4, 4, 82),
            (22, 10, 4, 88),
        ),
        "shift",
    ),
}
SCENARIO_NAMES = tuple(SCENARIOS)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))


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
        # ``records`` is the canonical index required by the grader contract;
        # ``scenarios`` is retained as a compatibility alias for older callers.
        self.records: dict[tuple[str, int], Any] = {}
        self.scenarios = self.records
        self.replayed: dict[tuple[str, int], Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_DISTANCE_CACHE: dict[str, dict[tuple[int, int], int]] = {}
_PATH_DISTANCE_CACHE: dict[
    tuple[str, tuple[int, int], tuple[int, int]], int
] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _brief(value: Any, limit: int = 180) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _action_context(name: str, variant: int, slot: int, robot: str) -> str:
    return f"{name}|v{variant}.actions[slot={slot},robot={robot}]"


def _blocked(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    x, y = position
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in spec["obstacles"])


def _traversable(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    return 0 <= position[0] < GRID_X and 0 <= position[1] < GRID_Y and not _blocked(spec, position)


def _distance_map(name: str) -> dict[tuple[int, int], int]:
    if name in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[name]
    spec = SCENARIOS[name]
    target = tuple(spec["base"])
    distances = {target: 0}
    queue: deque[tuple[int, int]] = deque([target])
    while queue:
        current = queue.popleft()
        for move in CARDINAL_MOVES:
            dx, dy = MOVE_DELTAS[move]
            nxt = current[0] + dx, current[1] + dy
            if nxt not in distances and _traversable(spec, nxt):
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    _DISTANCE_CACHE[name] = distances
    return distances


def _path_distance(name: str, start: tuple[int, int], target: tuple[int, int]) -> int:
    cache_key = (name, start, target)
    if cache_key in _PATH_DISTANCE_CACHE:
        return _PATH_DISTANCE_CACHE[cache_key]
    spec = SCENARIOS[name]
    if not _traversable(spec, start) or not _traversable(spec, target):
        distance = GRID_X * GRID_Y + 1
        _PATH_DISTANCE_CACHE[cache_key] = distance
        return distance
    distances = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        current = queue.popleft()
        if current == target:
            distance = distances[current]
            _PATH_DISTANCE_CACHE[cache_key] = distance
            return distance
        for move in CARDINAL_MOVES:
            dx, dy = MOVE_DELTAS[move]
            nxt = current[0] + dx, current[1] + dy
            if _traversable(spec, nxt) and nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    distance = GRID_X * GRID_Y + 1
    _PATH_DISTANCE_CACHE[cache_key] = distance
    return distance


def _region_index(spec: Mapping[str, Any], position: tuple[int, int]) -> int | None:
    x, y = position
    for index, (x0, y0, _value, _deadline) in enumerate(spec["regions"]):
        if x0 <= x < x0 + 2 and y0 <= y < y0 + 2:
            return index
    return None


def _cell_value_deadline(spec: Mapping[str, Any], position: tuple[int, int]) -> tuple[int, int]:
    region = _region_index(spec, position)
    if region is None:
        return 1, HORIZON
    _x, _y, value, deadline = spec["regions"][region]
    return value, deadline


def _packet_id(name: str, variant: int, robot: int, slot: int) -> str:
    return f"scenario={name}|variant={variant}|robot={robot}|slot={slot}"


def _parse_packet_id(value: Any) -> tuple[str, int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"scenario=([a-z_]+)\|variant=([0-2])\|robot=([0-3])\|slot=([0-9]+)",
        value,
    )
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


def _packet_class(name: str, variant: int, slot: int) -> str:
    if name == "critical_burst" and 20 + 5 * variant <= slot <= 31 + 5 * variant:
        return "critical"
    phase = (slot + variant) % 12
    return "map" if phase <= 5 else "critical" if phase <= 8 else "alarm"


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


def _expected_action_keys() -> set[tuple[int, str]]:
    return {(slot, robot) for slot in range(HORIZON) for robot in ROBOTS}


def _shape_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    rows = data.get("scenarios") if root else None
    return [
        _atom(
            "result_json",
            "parse",
            parse_ok,
            "strict JSON parsed" if parse_ok else "strict JSON parse failed",
        ),
        _atom(
            "result_json",
            "root",
            root and set(data) == {"scenarios"},
            f"expected exact root keys ['scenarios'], got {_brief(set(data) if root else data)}",
        ),
        _atom(
            "result_json",
            "scenario_rows",
            isinstance(rows, list)
            and all(
                isinstance(row, Mapping) and set(row) == {"scenario", "variant", "actions"}
                for row in rows
            ),
            "expected a list of records with exact keys scenario, variant, actions",
        ),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    scenario_ok = isinstance(rows, list) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("scenario"), str)
        and _integer(row.get("variant"))
        and row.get("variant") in range(3)
        for row in rows
    )
    action_rows_ok = True
    action_values_ok = True
    request_values_ok = True
    if not isinstance(rows, list):
        action_rows_ok = action_values_ok = request_values_ok = False
    else:
        for row in rows:
            actions = row.get("actions") if isinstance(row, Mapping) else None
            if not isinstance(actions, list):
                action_rows_ok = action_values_ok = request_values_ok = False
                continue
            for index, action in enumerate(actions):
                if not isinstance(action, Mapping) or set(action) != {
                    "slot",
                    "robot",
                    "move",
                    "drop",
                    "encode",
                    "send_packet",
                    "relay_path",
                }:
                    action_rows_ok = False
                    continue
                if not _integer(action.get("slot")) or not isinstance(action.get("robot"), str):
                    action_values_ok = False
                if not isinstance(action.get("move"), str) or not isinstance(
                    action.get("encode"), str
                ):
                    action_values_ok = False
                if not isinstance(action.get("drop"), bool):
                    action_values_ok = False
                if not isinstance(action.get("send_packet"), (str, type(None))):
                    request_values_ok = False
                path = action.get("relay_path")
                if (
                    not isinstance(path, list)
                    or any(not isinstance(item, str) for item in path)
                    or len(path) > 2
                ):
                    request_values_ok = False
    return [
        _atom("result_evidence", "scenario_rows", scenario_ok, "scenario and variant types"),
        _atom("result_evidence", "action_rows", action_rows_ok, "action objects have exact keys"),
        _atom(
            "result_evidence",
            "action_values",
            action_values_ok,
            "move, drop, encode, slot, and robot types",
        ),
        _atom(
            "result_evidence",
            "request_values",
            request_values_ok,
            "send_packet and ordered relay_path types",
        ),
    ]


def _index_records(data: Any) -> dict[tuple[str, int], Any]:
    out: dict[tuple[str, int], Any] = {}
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = (row.get("scenario"), row.get("variant"))
        if isinstance(key[0], str) and _integer(key[1]) and key not in out:
            out[key] = row
    return out


def _index(data: Any) -> dict[tuple[str, int], Any]:
    """Backward-compatible shorthand for the canonical record index."""
    return _index_records(data)


def _coverage_score(defect_count: int) -> float:
    """Normalize the fixed four-point coverage deduction."""
    if not _integer(defect_count) or defect_count < 0:
        raise ValueError("coverage defect count must be a non-negative integer")
    return max(0.0, 4.0 - 0.5 * defect_count) / 4.0


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    expected_scenarios = set(SCENARIO_KEYS)
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    counts: dict[Any, int] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                defects.append(f"unknown scenario row={_brief(row)}")
                continue
            scenario_value = row.get("scenario")
            variant_value = row.get("variant")
            if not isinstance(scenario_value, str) or not _integer(variant_value):
                defects.append(f"unknown scenario key={_brief((scenario_value, variant_value))}")
                continue
            key = (scenario_value, variant_value)
            counts[key] = counts.get(key, 0) + 1
            if key not in expected_scenarios:
                defects.append(f"unknown scenario key={key!r}")
    for key in sorted(expected_scenarios - set(counts)):
        defects.append(f"missing scenario key={key!r}")
    for key, count in sorted(counts.items(), key=repr):
        if count > 1:
            defects.extend(
                f"duplicate scenario key={key!r} count={count}"
                for _ in range(count - 1)
            )
    indexed = _index(data)
    expected_actions = _expected_action_keys()
    for key in sorted(expected_scenarios & set(indexed)):
        actions = indexed[key].get("actions")
        keys: list[Any] = []
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, Mapping):
                    action_key = (action.get("slot"), action.get("robot"))
                    try:
                        hash(action_key)
                    except TypeError:
                        defects.append(f"{key} unhashable action key={_brief(action_key)}")
                    else:
                        keys.append(action_key)
                else:
                    keys.append(None)
        action_counts: dict[Any, int] = {}
        for action_key in keys:
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
        for action_key in sorted(expected_actions - set(action_counts), key=repr):
            defects.append(f"{key} missing action key={action_key!r}")
        for action_key, count in sorted(action_counts.items(), key=repr):
            if action_key not in expected_actions:
                defects.append(f"{key} unknown action key={action_key!r}")
            elif count > 1:
                defects.extend(
                    f"{key} duplicate action key={action_key!r} count={count}"
                    for _ in range(count - 1)
                )
    return _coverage_score(len(defects)), defects


def _node_position(
    deployed: Mapping[str, tuple[int, int]], spec: Mapping[str, Any], ident: str
) -> tuple[int, int]:
    return tuple(spec["base"]) if ident == "base" else deployed[ident]


def _route_error(
    name: str,
    deployed: Mapping[str, tuple[int, int]],
    start: tuple[int, int],
    path: Sequence[str],
) -> str | None:
    spec = SCENARIOS[name]
    if not isinstance(path, (list, tuple)):
        return "relay path must be an array"
    if any(not isinstance(relay, str) for relay in path):
        return "relay path entries must be strings"
    if len(path) > 2 or len(set(path)) != len(path):
        return "relay path must contain zero to two distinct deployed nodes"
    if any(relay == "base" for relay in path):
        return "relay path entries must be deployed node IDs, not 'base'"
    previous = start
    previous_distance = _distance_map(name).get(previous, GRID_X * GRID_Y + 1)
    for relay in path:
        if relay not in deployed:
            return f"relay={relay!r} is not an action-start deployed node"
        target = _node_position(deployed, spec, relay)
        manhattan = abs(previous[0] - target[0]) + abs(previous[1] - target[1])
        if manhattan > 10:
            return f"relay={relay!r} Manhattan distance={manhattan}; expected <=10"
        target_distance = _distance_map(name).get(target, GRID_X * GRID_Y + 1)
        if target_distance >= previous_distance:
            return (
                f"relay={relay!r} base_distance={target_distance}; "
                f"expected strictly below {previous_distance}"
            )
        previous, previous_distance = target, target_distance
    target = tuple(spec["base"])
    manhattan = abs(previous[0] - target[0]) + abs(previous[1] - target[1])
    if manhattan > 10:
        return f"relay='base' Manhattan distance={manhattan}; expected <=10"
    return None


def _interference(
    name: str, variant: int, slot: int, first: tuple[int, int], second: tuple[int, int]
) -> float:
    spec = SCENARIOS[name]
    midpoint_x = (first[0] + second[0]) / 2.0
    midpoint_y = (first[1] + second[1]) / 2.0
    if spec["interference"] == "split" and midpoint_x >= 12:
        return 0.08
    if spec["interference"] == "shift":
        center = (4 + (slot + 7 * variant) % 18, 7)
        if max(abs(midpoint_x - center[0]), abs(midpoint_y - center[1])) <= 2:
            return 0.16
    return 0.0


def _link_sample(
    name: str,
    variant: int,
    slot: int,
    packet_id: str,
    attempt: int,
    leg: int,
    sender: str,
    receiver: str,
    first: tuple[int, int],
    second: tuple[int, int],
    deployed: Mapping[str, tuple[int, int]],
    concurrent: int,
) -> tuple[float, float, bool]:
    manhattan = abs(first[0] - second[0]) + abs(first[1] - second[1])
    d = min(1.0, manhattan / 10.0)
    obstruction = int(_path_distance(name, first, second) > manhattan)
    midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    overlap = sum(
        int(
            ident not in {sender, receiver}
            and max(abs(position[0] - midpoint[0]), abs(position[1] - midpoint[1])) <= 1
        )
        for ident, position in deployed.items()
    )
    probability = max(
        0.08,
        min(
            0.96,
            0.96
            - 0.28 * d
            - 0.10 * obstruction
            - 0.04 * (concurrent - 1)
            - 0.025 * overlap
            - _interference(name, variant, slot, first, second),
        ),
    )
    hop_id = f"{name}|{variant}|{packet_id}|a{attempt}|g{leg}|{sender}|{receiver}"
    sample = (
        int(hashlib.sha256(f"{hop_id}|salt={11 * variant}".encode()).hexdigest()[:16], 16) / 2**64
    )
    return probability, sample, sample < probability


def _action_map(record: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    actions = record.get("actions") if isinstance(record, Mapping) else None
    if not isinstance(actions, list):
        return {}
    out: dict[tuple[int, str], Mapping[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        slot = action.get("slot")
        robot = action.get("robot")
        if _integer(slot) and isinstance(robot, str):
            out[(slot, robot)] = action
    return out


def _packet_age(packet: Mapping[str, Any], boundary: int) -> int:
    return boundary - int(packet["generation"])


def _remove_from_queue(queues: list[list[str]], robot: int, packet_id: str) -> None:
    queues[robot] = [ident for ident in queues[robot] if ident != packet_id]


def _replay(
    name: str, variant: int, record: Mapping[str, Any], stop: int | None = None
) -> dict[str, Any]:
    spec = SCENARIOS[name]
    actions = _action_map(record)
    if stop is not None and (not _integer(stop) or stop not in range(HORIZON + 1)):
        raise ValueError(f"stop={stop!r}; expected an integer boundary in 0..{HORIZON}")
    limit = HORIZON if stop is None else stop
    positions = [tuple(position) for position in spec["starts"]]
    energies = [ENERGY0] * N_ROBOTS
    queues: list[list[str]] = [[] for _ in ROBOTS]
    packets: dict[str, dict[str, Any]] = {}
    deployed: dict[str, tuple[int, int]] = {}
    node_owner: dict[str, int] = {}
    next_drop = [2, 2, 2, 2]
    last_drop = [-100] * N_ROBOTS
    coverage = set(positions)
    first_visit = {cell: 0 for cell in positions}
    report_ack: list[int | None] = [None] * len(spec["regions"])
    report_owner: list[int | None] = [None] * len(spec["regions"])
    safety_events = 0
    disconnected_required_slots = 0
    accepted_hop_units = 0
    dropped_packets = 0
    errors: dict[str, list[str]] = {
        key: [] for key in ("motion", "deployment", "requests", "routes", "energy")
    }
    positions_history = [tuple(positions)]
    energies_history = [tuple(energies)]
    deployed_history = [tuple(sorted(deployed))]
    queue_history: list[tuple[tuple[str, ...], ...]] = [tuple(tuple() for _ in ROBOTS)]
    report_history = [tuple(report_ack)]
    report_owner_history = [tuple(report_owner)]
    admitted_history: list[int] = []

    def expire_queues(boundary: int, close_retry: bool = False) -> None:
        """Apply the boundary ageing rule, optionally closing a terminal retry window."""
        nonlocal dropped_packets
        for packet_id in list(packets):
            packet = packets[packet_id]
            if packet["status"] != "queued":
                continue
            if _packet_age(packet, boundary) > packet["max_age"]:
                _remove_from_queue(queues, packet["robot"], packet_id)
                packet["status"] = "dropped"
                packet["drop_reason"] = "age"
                dropped_packets += 1
            elif close_retry and packet.get("retry_due") == boundary:
                _remove_from_queue(queues, packet["robot"], packet_id)
                packet["status"] = "dropped"
                packet["drop_reason"] = "retry_window"
                dropped_packets += 1

    for slot in range(limit):
        boundary = slot
        expire_queues(boundary)
        # A boundary snapshot is the action-start state after ageing and before
        # the action at this slot.  Recording it here avoids charging an
        # over-age packet to the next boundary's queue-age metric.
        queue_history[slot] = tuple(tuple(queue) for queue in queues)

        rows = {robot: actions.get((slot, robot), {}) for robot in ROBOTS}
        starts = tuple(positions)
        available_retry = {
            packet_id
            for packet_id, packet in packets.items()
            if packet["status"] == "queued" and packet.get("retry_due") == slot
        }
        current_report_holders = [
            packet_id
            for packet_id, packet in packets.items()
            if packet["status"] == "queued"
            and packet.get("region") is not None
            and report_ack[packet["region"]] is None
        ]
        if current_report_holders:
            disconnected = True
            for packet_id in current_report_holders:
                packet = packets[packet_id]
                if _route_error(name, deployed, starts[packet["robot"]], ()) is None:
                    disconnected = False
                    break
                route_candidates = [
                    route for length in (1, 2) for route in permutations(deployed, length)
                ]
                for route in route_candidates:
                    if _route_error(name, deployed, starts[packet["robot"]], route) is None:
                        disconnected = False
                        break
                if not disconnected:
                    break
            if disconnected:
                disconnected_required_slots += 1

        request_rows: list[dict[str, Any]] = []
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            context = _action_context(name, variant, slot, robot)
            move = row.get("move")
            if not isinstance(move, str) or move not in MOVES:
                errors["motion"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].move={move!r}"
                )
            drop = row.get("drop")
            drop_action = isinstance(drop, bool) and drop
            if not isinstance(drop, bool):
                errors["deployment"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].drop is not boolean"
                )
            encode = row.get("encode")
            if not isinstance(encode, str) or encode not in ENCODINGS:
                errors["requests"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].encode={encode!r}"
                )
            send_packet = row.get("send_packet")
            raw_path = row.get("relay_path")
            path_valid = (
                isinstance(raw_path, list)
                and len(raw_path) <= 2
                and all(isinstance(item, str) for item in raw_path)
            )
            path = raw_path if path_valid else []
            if not path_valid:
                target_errors = errors["deployment"] if drop_action else errors["routes"]
                target_errors.append(
                    f"{context}.relay_path must be a string array of length 0..2"
                )
            if send_packet is None and path:
                target_errors = errors["deployment"] if drop_action else errors["routes"]
                target_errors.append(f"{context} null send with non-empty path")
            if send_packet is not None:
                packet = packets.get(send_packet) if isinstance(send_packet, str) else None
                if not isinstance(send_packet, str):
                    errors["requests"].append(
                        f"{context}.send_packet must be a string or null"
                    )
                if (
                    packet is None
                    or packet["status"] != "queued"
                    or packet["robot"] != robot_index
                    or send_packet not in queues[robot_index]
                ):
                    errors["requests"].append(
                        f"{context}.send_packet={send_packet!r}; expected action-start queue packet"
                    )
                else:
                    if drop_action:
                        errors["deployment"].append(
                            f"{context} drop cannot transmit"
                        )
                    else:
                        retry_path_ok = True
                        if (
                            packet.get("retry_due") is not None
                            and send_packet not in available_retry
                        ):
                            errors["requests"].append(
                                f"{context} retry packet is only requestable at boundary "
                                f"{packet.get('retry_due')}"
                            )
                            retry_path_ok = False
                        if send_packet in available_retry and tuple(path) != tuple(
                            packet.get("retry_path", ())
                        ):
                            errors["routes"].append(
                                f"{context} retry path differs from failed route"
                            )
                            retry_path_ok = False
                        route_error = _route_error(name, deployed, starts[robot_index], path)
                        if route_error is not None:
                            errors["routes"].append(
                                f"{context} {route_error}"
                            )
                        if (
                            path_valid
                            and route_error is None
                            and retry_path_ok
                            and not any(
                                item["packet_id"] == send_packet for item in request_rows
                            )
                        ):
                            request_rows.append(
                                {
                                    "robot": robot_index,
                                    "packet_id": send_packet,
                                    "path": tuple(path),
                                    "cost": packet["payload_units"] * (len(path) + 1),
                                    "retry": send_packet in available_retry,
                                }
                            )
            if isinstance(drop, bool) and drop:
                if move != "stay":
                    errors["deployment"].append(
                        f"{context} drop requires stay"
                    )
                if next_drop[robot_index] <= 0:
                    errors["deployment"].append(
                        f"{context} no carried node remains"
                    )
                if slot - last_drop[robot_index] < 6:
                    errors["deployment"].append(
                        f"{context} drop cooldown"
                    )
                if starts[robot_index] in deployed.values():
                    errors["deployment"].append(
                        f"{context} deployed-node cell is occupied"
                    )

        request_rows.sort(
            key=lambda request: (
                packets[request["packet_id"]]["generation"]
                + packets[request["packet_id"]]["max_age"],
                0 if request["retry"] else 1,
                packets[request["packet_id"]]["generation"],
                request["robot"],
                request["packet_id"],
            )
        )
        remaining = SLOT_CAPACITY
        admitted: list[dict[str, Any]] = []
        for request in request_rows:
            if request["cost"] <= remaining:
                admitted.append(request)
                remaining -= request["cost"]
        admitted_history.append(sum(request["cost"] for request in admitted))
        accepted_hop_units += sum(request["cost"] for request in admitted)
        outcomes: dict[str, list[bool]] = {}
        for request in admitted:
            packet = packets[request["packet_id"]]
            attempt = 1 if request["retry"] else 0
            sender = ROBOTS[request["robot"]]
            endpoints = [starts[request["robot"]]]
            endpoints.extend(deployed[relay] for relay in request["path"])
            endpoints.append(tuple(spec["base"]))
            outcome = []
            for leg in range(len(endpoints) - 1):
                receiver = request["path"][leg] if leg < len(request["path"]) else "base"
                probability, sample, success = _link_sample(
                    name,
                    variant,
                    slot,
                    request["packet_id"],
                    attempt,
                    leg,
                    sender if leg == 0 else request["path"][leg - 1],
                    receiver,
                    endpoints[leg],
                    endpoints[leg + 1],
                    deployed,
                    len(admitted),
                )
                outcome.append(success)
                packet.setdefault("legs", []).append(
                    {
                        "slot": slot,
                        "leg": leg,
                        "attempt": attempt,
                        "probability": probability,
                        "sample": sample,
                        "success": success,
                    }
                )
            outcomes[request["packet_id"]] = outcome

        accepted_destinations: list[tuple[int, int]] = []
        next_positions: list[tuple[int, int]] = []
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            move = row.get("move")
            delta = MOVE_DELTAS.get(move, (0, 0)) if isinstance(move, str) else (0, 0)
            destination = starts[robot_index][0] + delta[0], starts[robot_index][1] + delta[1]
            invalid = bool(
                not _traversable(spec, destination)
                or destination in accepted_destinations
                or destination in starts[robot_index + 1 :]
            )
            if invalid:
                destination = starts[robot_index]
                if move != "stay":
                    safety_events += 1
            accepted_destinations.append(destination)
            next_positions.append(destination)
        positions = next_positions

        # Evaluate the complete drop predicate once at the post-motion
        # boundary.  The same predicate controls both deployment and its
        # energy charge, so malformed drops cannot mutate state or consume
        # deployment energy while still being reported as illegal.
        drop_valid_flags: list[bool] = []
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            raw_path = row.get("relay_path")
            drop_valid_flags.append(
                bool(
                    isinstance(row.get("drop"), bool)
                    and row["drop"]
                    and row.get("move") == "stay"
                    and row.get("send_packet") is None
                    and isinstance(raw_path, list)
                    and len(raw_path) == 0
                    and next_drop[robot_index] > 0
                    and slot - last_drop[robot_index] >= 6
                    and positions[robot_index] not in deployed.values()
                    and _traversable(spec, positions[robot_index])
                )
            )

        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            moved = positions[robot_index] != starts[robot_index]
            cost = 1.0 if moved else 0.2
            if drop_valid_flags[robot_index]:
                cost += 8.0
            for request in admitted:
                if request["robot"] == robot_index:
                    cost += request["cost"] * 0.05
            energies[robot_index] -= cost
            if energies[robot_index] < -EPS:
                errors["energy"].append(
                    f"{_action_context(name, variant, slot, robot)} "
                    f"resulting_energy={energies[robot_index]:.9g}; "
                    f"expected >= {-EPS} with tolerance {EPS}"
                )

        admitted_ids = {request["packet_id"] for request in admitted}
        for request in admitted:
            packet = packets[request["packet_id"]]
            if all(outcomes[request["packet_id"]]):
                _remove_from_queue(queues, request["robot"], request["packet_id"])
                packet["status"] = "delivered"
                packet["arrival"] = slot + 1
                packet["retry_due"] = None
                packet["route"] = request["path"]
                if packet.get("region") is not None and report_ack[packet["region"]] is None:
                    report_ack[packet["region"]] = slot + 1
            elif request["retry"]:
                _remove_from_queue(queues, request["robot"], request["packet_id"])
                packet["status"] = "dropped"
                packet["drop_reason"] = "retry"
                packet["retry_due"] = None
                packet["route"] = request["path"]
                dropped_packets += 1
            else:
                packet["attempt"] = 1
                packet["retry_due"] = slot + 1
                packet["retry_path"] = request["path"]
                packet["route"] = request["path"]

        for packet_id in list(available_retry):
            if packet_id not in admitted_ids and packets[packet_id]["status"] == "queued":
                _remove_from_queue(queues, packets[packet_id]["robot"], packet_id)
                packets[packet_id]["status"] = "dropped"
                packets[packet_id]["drop_reason"] = "retry_window"
                dropped_packets += 1
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            if drop_valid_flags[robot_index]:
                node_id = f"{robot}d{2 - next_drop[robot_index]}"
                deployed[node_id] = positions[robot_index]
                node_owner[node_id] = robot_index
                next_drop[robot_index] -= 1
                last_drop[robot_index] = slot

        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            packet_id = _packet_id(name, variant, robot_index, slot)
            packet_class = _packet_class(name, variant, slot)
            region = _region_index(spec, positions[robot_index])
            if (
                region is not None
                and report_ack[region] is None
                and report_owner[region] == robot_index
            ):
                packet_region = region
            else:
                packet_region = None
            encoding = row.get("encode")
            if not isinstance(encoding, str) or encoding not in ENCODINGS:
                encoding = "compact"
            payload_units, retained_fraction = ENCODINGS[encoding]
            packet = {
                "id": packet_id,
                "robot": robot_index,
                "generation": slot,
                "class": packet_class,
                "max_age": CLASS_LIMITS[packet_class],
                "weight": CLASS_WEIGHTS[packet_class],
                "payload_units": payload_units,
                "retained_fraction": retained_fraction,
                "cell": positions[robot_index],
                "region": packet_region,
                "status": "queued",
                "attempt": 0,
                "retry_due": None,
                "route": (),
                "legs": [],
            }
            packets[packet_id] = packet
            queues[robot_index].append(packet_id)

        for robot_index in range(N_ROBOTS):
            packets_for_queue = [
                packets[packet_id]
                for packet_id in queues[robot_index]
                if packets[packet_id]["status"] == "queued"
            ]
            packets_for_queue.sort(
                key=lambda packet: (
                    packet["generation"] + packet["max_age"],
                    -packet["weight"],
                    packet["generation"],
                    packet["id"],
                )
            )
            kept: list[str] = []
            used = 0
            capacity_exhausted = False
            for packet in packets_for_queue:
                if not capacity_exhausted and used + packet["payload_units"] <= QUEUE_CAPACITY:
                    kept.append(packet["id"])
                    used += packet["payload_units"]
                else:
                    capacity_exhausted = True
                    packet["status"] = "dropped"
                    packet["drop_reason"] = "queue"
                    dropped_packets += 1
            queues[robot_index] = kept

        # Coverage and report ownership are observations produced by the
        # completed movement. They become visible at the next action-start
        # boundary, so a newly entered region is not attached to the packet
        # generated in this same slot.
        for robot_index, position in enumerate(positions):
            coverage.add(position)
            first_visit.setdefault(position, slot + 1)
            region = _region_index(spec, position)
            if region is not None and report_ack[region] is None and report_owner[region] is None:
                report_owner[region] = robot_index

        positions_history.append(tuple(positions))
        energies_history.append(tuple(energies))
        deployed_history.append(tuple(sorted(deployed)))
        queue_history.append(tuple(tuple(queue) for queue in queues))
        report_history.append(tuple(report_ack))
        report_owner_history.append(tuple(report_owner))

    # The returned final snapshot is an action-start boundary for a partial
    # replay, so apply ageing there while leaving a due retry available.  A
    # complete replay has no following action at boundary 90 and therefore
    # closes both ageing and the retry window for its terminal metric snapshot.
    if stop is None or limit == HORIZON:
        expire_queues(HORIZON, close_retry=True)
    else:
        expire_queues(limit)
    queue_history[-1] = tuple(tuple(queue) for queue in queues)

    expected_packets = HORIZON * N_ROBOTS
    return {
        "positions": tuple(positions_history),
        "energies": tuple(energies_history),
        "deployed": dict(deployed),
        "deployed_history": tuple(deployed_history),
        "packets": packets,
        "queues": tuple(queue_history),
        "reports": tuple(report_ack),
        "report_history": tuple(report_history),
        "report_owners": tuple(report_owner),
        "report_owner_history": tuple(report_owner_history),
        "coverage": frozenset(coverage),
        "first_visit": dict(first_visit),
        "safety_events": safety_events,
        "disconnected_required_slots": disconnected_required_slots,
        "accepted_hop_units": accepted_hop_units,
        "dropped_packets": dropped_packets,
        "admitted_history": tuple(admitted_history),
        "errors": errors,
        "expected_packets": expected_packets,
    }


def _legality(scenarios: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "deployment", "requests", "routes", "energy")
    }
    for key in SCENARIO_KEYS:
        record = scenarios.get(key)
        if not isinstance(record, Mapping):
            continue
        state = _replay(key[0], key[1], record)
        for error_key, details in state["errors"].items():
            failures[error_key].extend(details)

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("decision_legality", "motion", not failures["motion"], detail("motion")),
        _atom("decision_legality", "deployment", not failures["deployment"], detail("deployment")),
        _atom("decision_legality", "requests", not failures["requests"], detail("requests")),
        _atom("decision_legality", "routes", not failures["routes"], detail("routes")),
        _atom("decision_legality", "energy", not failures["energy"], detail("energy")),
    ]


def _replay_atoms(
    scenarios: Mapping[tuple[str, int], Any], replayed: Mapping[tuple[str, int], Any]
) -> list[AtomVerdict]:
    del scenarios
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "deployments", "packets", "queues", "mission")
    }
    replay_keys = set(replayed)
    expected_keys = set(SCENARIO_KEYS)
    if replay_keys != expected_keys:
        detail = f"scenario keys={sorted(replay_keys, key=repr)}; expected={list(SCENARIO_KEYS)}"
        # A missing or extra canonical replay unit means that no complete
        # fixed-model evidence exists for any replay atom.  Do not let the
        # unaffected-looking atoms pass vacuously when the replay map is empty
        # or only partially populated.
        for failure_list in failures.values():
            failure_list.append(detail)
    def is_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

    def valid_position(spec: Mapping[str, Any], value: Any) -> bool:
        if not is_sequence(value) or len(value) != 2:
            return False
        if any(not _integer(coordinate) for coordinate in value):
            return False
        return _traversable(spec, (value[0], value[1]))

    for key in SCENARIO_KEYS:
        state = replayed.get(key)
        if not isinstance(state, Mapping):
            for failure_list in failures.values():
                failure_list.append(f"{key!r} has no replay state mapping")
            continue
        name, variant = key
        spec = SCENARIOS[name]
        positions = state.get("positions", ())
        energies = state.get("energies", ())
        positions_ok = is_sequence(positions)
        energies_ok = is_sequence(energies)
        if not positions_ok:
            failures["motion"].append(
                f"{name}|v{variant}.positions is not a sequence"
            )
            positions = ()
        if not energies_ok:
            failures["motion"].append(
                f"{name}|v{variant}.energies is not a sequence"
            )
            energies = ()
        if len(positions) != HORIZON + 1:
            failures["motion"].append(
                f"{name}|v{variant}.positions length={len(positions)}; expected {HORIZON + 1}"
            )
        if len(energies) != HORIZON + 1:
            failures["motion"].append(
                f"{name}|v{variant}.energies length={len(energies)}; expected {HORIZON + 1}"
            )
        for boundary, values in enumerate(positions):
            if not is_sequence(values):
                failures["motion"].append(
                    f"{name}|v{variant}.positions[{boundary}] is not a sequence"
                )
                continue
            valid_cells = [
                tuple(position) for position in values if valid_position(spec, position)
            ]
            if (
                len(values) != N_ROBOTS
                or len(valid_cells) != N_ROBOTS
                or len(set(valid_cells)) != N_ROBOTS
            ):
                failures["motion"].append(
                    f"{name}|v{variant}.positions[{boundary}]={_brief(values)}; "
                    "expected four distinct cells"
                )
        for boundary, values in enumerate(energies):
            if not is_sequence(values):
                failures["motion"].append(
                    f"{name}|v{variant}.energies[{boundary}] is not a sequence"
                )
                continue
            if len(values) != N_ROBOTS or any(
                not _finite_number(value) or value < -EPS or value > ENERGY0 + EPS
                for value in values
            ):
                failures["motion"].append(
                    f"{name}|v{variant}.energies[{boundary}]={_brief(values)}; "
                    "expected four balances in [-1e-6,100]"
                )

        deployed = state.get("deployed", {})
        if not isinstance(deployed, Mapping):
            failures["deployments"].append(f"{name}|v{variant}.deployed is not a mapping")
        else:
            if len(deployed) > 8:
                failures["deployments"].append(
                    f"{name}|v{variant}.deployed count={len(deployed)}; expected <=8"
                )
            for ident, position in deployed.items():
                if not re.fullmatch(r"r[0-3]d[01]", str(ident)) or not valid_position(
                    spec, position
                ):
                    failures["deployments"].append(
                        f"{name}|v{variant}.deployed[{ident!r}]={position!r}; "
                        "expected valid node and traversable cell"
                    )

        packets = state.get("packets", {})
        expected = {
            _packet_id(name, variant, robot, slot)
            for robot in range(N_ROBOTS)
            for slot in range(HORIZON)
        }
        packet_keys: set[Any] = set()
        packet_keys_ok = isinstance(packets, Mapping)
        if packet_keys_ok:
            try:
                packet_keys = set(packets)
            except TypeError:
                packet_keys_ok = False
                failures["packets"].append(
                    f"{name}|v{variant}.packets has unhashable keys"
                )
        if not packet_keys_ok or packet_keys != expected:
            missing = sorted(expected - packet_keys) if packet_keys_ok else "all"
            unknown = sorted(packet_keys - expected) if packet_keys_ok else []
            failures["packets"].append(
                f"{name}|v{variant}.packet keys missing={missing}, unknown={unknown}"
            )
        if packet_keys_ok:
            for packet_id, packet in packets.items():
                if not isinstance(packet, Mapping):
                    failures["packets"].append(
                        f"{name}|v{variant}.{packet_id} is not a packet mapping"
                    )
                    continue
                if packet.get("status") not in {"queued", "delivered", "dropped"}:
                    failures["packets"].append(
                        f"{name}|v{variant}.{packet_id}.status={packet.get('status')!r}"
                    )
                if not _integer(packet.get("payload_units")) or packet.get("payload_units") not in {
                    1,
                    2,
                } or not _finite_number(
                    packet.get("retained_fraction")
                ):
                    failures["packets"].append(
                        f"{name}|v{variant}.{packet_id} encoding evidence invalid"
                    )
                if packet.get("status") == "delivered" and packet.get("arrival") not in range(
                    1, HORIZON + 1
                ):
                    failures["packets"].append(
                        f"{name}|v{variant}.{packet_id}.arrival={packet.get('arrival')!r}; "
                        "expected 1..90"
                    )

        queues = state.get("queues", ())
        queues_ok = is_sequence(queues)
        if not queues_ok:
            failures["queues"].append(f"{name}|v{variant}.queues is not a sequence")
            queues = ()
        if len(queues) != HORIZON + 1:
            failures["queues"].append(f"{name}|v{variant}.queues length={len(queues)}; expected 91")
        else:
            for boundary, snapshot in enumerate(queues):
                if not is_sequence(snapshot):
                    failures["queues"].append(
                        f"{name}|v{variant}.queues[{boundary}] is not a sequence"
                    )
                    continue
                if len(snapshot) != N_ROBOTS:
                    failures["queues"].append(
                        f"{name}|v{variant}.queues[{boundary}] node_count={len(snapshot)}; "
                        "expected 4"
                    )
                    continue
                flat: list[str] = []
                for robot, queue in enumerate(snapshot):
                    if not is_sequence(queue):
                        failures["queues"].append(
                            f"{name}|v{variant}.queues[{boundary}][r{robot}] is not a sequence"
                        )
                        continue
                    flat.extend(item for item in queue if isinstance(item, str))
                    if any(not isinstance(item, str) for item in queue):
                        failures["queues"].append(
                            f"{name}|v{variant}.queues[{boundary}][r{robot}] "
                            "has non-string packet ID"
                        )
                    if (
                        sum(
                            packets[ident]["payload_units"]
                            for ident in queue
                            if packet_keys_ok
                            and isinstance(ident, str)
                            and ident in packets
                            and isinstance(packets[ident], Mapping)
                            and _integer(packets[ident].get("payload_units"))
                            and packets[ident].get("payload_units") in {1, 2}
                        )
                        > QUEUE_CAPACITY
                    ):
                        failures["queues"].append(
                            f"{name}|v{variant}.queues[{boundary}][r{robot}] "
                            "payload units exceed 12"
                        )
                if len(flat) != len(set(flat)):
                    failures["queues"].append(
                        f"{name}|v{variant}.queues[{boundary}] duplicate packet carriers"
                    )
        owner_history = state.get("report_owner_history", ())
        if not is_sequence(owner_history) or len(owner_history) != HORIZON + 1:
            failures["mission"].append(
                f"{name}|v{variant}.report_owner_history length/type invalid"
            )
        else:
            for boundary, owners in enumerate(owner_history):
                if not is_sequence(owners) or len(owners) != len(spec["regions"]):
                    failures["mission"].append(
                        f"{name}|v{variant}.report_owner_history[{boundary}] invalid"
                    )
                    continue
                if any(
                    owner is not None
                    and (not _integer(owner) or owner not in range(N_ROBOTS))
                    for owner in owners
                ):
                    failures["mission"].append(
                        f"{name}|v{variant}.report_owner_history[{boundary}] has invalid owner"
                    )
        first_visit = state.get("first_visit", {})
        if not isinstance(first_visit, Mapping) or any(
            first_visit.get(start) != 0 for start in spec["starts"]
        ):
            failures["mission"].append(
                f"{name}|v{variant}.start cells must have first_visit=0"
            )
        if not _integer(state.get("safety_events")) or state["safety_events"] not in range(
            HORIZON * N_ROBOTS + 1
        ):
            failures["mission"].append(
                f"{name}|v{variant}.safety_events={state.get('safety_events')!r}; expected 0..360"
            )
        if not _integer(state.get("disconnected_required_slots")) or state[
            "disconnected_required_slots"
        ] not in range(HORIZON + 1):
            failures["mission"].append(f"{name}|v{variant}.disconnected_required_slots invalid")
        if not _integer(state.get("accepted_hop_units")) or state[
            "accepted_hop_units"
        ] not in range(SCENARIO_BUDGET + 1):
            failures["mission"].append(
                f"{name}|v{variant}.accepted_hop_units="
                f"{state.get('accepted_hop_units')!r}; expected 0..720"
            )
    try:
        metrics = _metrics(replayed)
        if set(metrics) != set(METRIC_KEYS) or any(
            not _finite_number(value) or not 0 <= value <= 1 for value in metrics.values()
        ):
            failures["mission"].append(
                f"metrics={_brief(metrics)}; expected five finite values in [0,1]"
            )
    except Exception as exc:
        failures["mission"].append(f"metrics raised {type(exc).__name__}: {exc}")

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("replay_validity", "motion", not failures["motion"], detail("motion")),
        _atom("replay_validity", "deployments", not failures["deployments"], detail("deployments")),
        _atom("replay_validity", "packets", not failures["packets"], detail("packets")),
        _atom("replay_validity", "queues", not failures["queues"], detail("queues")),
        _atom("replay_validity", "mission", not failures["mission"], detail("mission")),
    ]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for key in SCENARIO_KEYS:
        name, variant = key
        spec = SCENARIOS[name]
        state = replayed[key]
        traversable = {
            (x, y) for x in range(GRID_X) for y in range(GRID_Y) if _traversable(spec, (x, y))
        }
        cell_value = {cell: _cell_value_deadline(spec, cell)[0] for cell in traversable}
        cell_deadline = {cell: _cell_value_deadline(spec, cell)[1] for cell in traversable}
        best_fraction = {cell: 0.0 for cell in traversable}
        service_numerator = 0.0
        service_denominator = sum(
            CLASS_WEIGHTS[_packet_class(name, variant, slot)]
            for slot in range(HORIZON)
            for _robot in ROBOTS
        )
        for packet in state["packets"].values():
            if packet["status"] != "delivered":
                continue
            arrival = packet.get("arrival", HORIZON + 1)
            if arrival <= cell_deadline[tuple(packet["cell"])]:
                best_fraction[tuple(packet["cell"])] = max(
                    best_fraction[tuple(packet["cell"])], packet["retained_fraction"]
                )
            if arrival <= packet["generation"] + packet["max_age"]:
                service_numerator += packet["weight"] * packet["retained_fraction"]
        total_value = max(1, sum(cell_value.values()))
        totals["primary_metrics_1"] += _clip(
            sum(cell_value[cell] * best_fraction[cell] for cell in traversable) / total_value
        )

        report_score = 0.0
        report_weight = 0.0
        for region_index, (_x, _y, value, deadline) in enumerate(spec["regions"]):
            report_weight += value
            arrival = state["reports"][region_index]
            if arrival is None:
                quality = 0.0
            elif arrival <= deadline:
                quality = 1.0
            else:
                quality = 0.5 * _clip(1.0 - (arrival - deadline) / max(1, HORIZON - deadline))
            report_score += value * quality
        totals["primary_metrics_2"] += _clip(report_score / max(1.0, report_weight))

        progress = (
            sum(
                cell_value[cell]
                for cell, boundary in state["first_visit"].items()
                if cell in cell_value and boundary <= cell_deadline[cell]
            )
            / total_value
        )
        totals["primary_metrics_3"] += _clip(
            0.45 * _clip(progress)
            + 0.30 * _clip(1.0 - state["disconnected_required_slots"] / HORIZON)
            + 0.25 * _clip(1.0 - state["safety_events"] / (HORIZON * N_ROBOTS))
        )

        queue_age_total = 0.0
        for boundary, snapshot in enumerate(state["queues"]):
            for queue in snapshot:
                units = 0
                age_units = 0.0
                for packet_id in queue:
                    packet = state["packets"].get(packet_id)
                    if packet is not None:
                        units += packet["payload_units"]
                        age_units += packet["payload_units"] * max(
                            0, boundary - packet["generation"]
                        )
                queue_age_total += age_units / units if units else 0.0
        mean_queue_age = queue_age_total / ((HORIZON + 1) * N_ROBOTS)
        totals["auxiliary_metrics_1"] += _clip(
            0.5 * _clip(service_numerator / max(1.0, service_denominator))
            + 0.5 * _clip(1.0 - mean_queue_age / 12.0)
        )
        totals["auxiliary_metrics_2"] += _clip(
            0.5 * _clip(1.0 - len(state["deployed"]) / 8.0)
            + 0.5 * _clip(1.0 - state["accepted_hop_units"] / SCENARIO_BUDGET)
        )
    return {key: _clip(value / len(SCENARIO_KEYS)) for key, value in totals.items()}


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from declared share")
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
                in_fence, fence_character, fence_length = True, marker[0], len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                in_fence = False
            previous_visible = False
            continue
        if in_fence:
            previous_visible = False
            continue
        atx = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$", line)
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
        del heading_line
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
        f"missing_sections={[title for title in required if title not in found]}; "
        f"normalized={score:.9g}",
    )


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
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
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
            f"timed_out={state.timed_out}, "
            f"new_regular_results={state.result_file_is_new_regular}"
        )
        state.official_message = (
            f"{state.official_message}; {status}" if state.official_message else status
        )
        parse_failure = ""
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                parse_failure = f"strict parse failed: {type(exc).__name__}: {exc}"
        shape = _shape_atoms(state.data, state.parse_ok)
        state.records = _index_records(state.data)
        state.scenarios = state.records
        evidence = _evidence_atoms(state.data)
        coverage_score, coverage_defects = _coverage(state.data)
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = [
                _atom(
                    "decision_legality",
                    atom_name,
                    False,
                    f"legality replay raised {type(exc).__name__}: {exc}",
                )
                for atom_name in ("motion", "deployment", "requests", "routes", "energy")
            ]
        state.scores["official_run"] = 1.0 if state.official_ok else 0.0
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.scores["canonical_coverage"] = coverage_score
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
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
            {key_: _replay(key_[0], key_[1], state.scenarios[key_]) for key_ in SCENARIO_KEYS}
            if can_replay
            else {}
        )
        replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["result_json"] = f"{parse_failure}; " if parse_failure else ""
        state.diagnostics["result_json"] += _diagnose(shape)
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = (
            f"defect_count={len(coverage_defects)}; " + " | ".join(coverage_defects[:12])
            if coverage_defects
            else "defect_count=0"
        )
        state.diagnostics["decision_legality"] = _diagnose(legality)
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in GATE_CHECKS
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
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _state_check(ws: Workspace, ident: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]:
    return await _state_check(ws, "result_json")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]:
    return await _state_check(ws, "result_evidence")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace) -> tuple[float, str]:
    return await _state_check(ws, "canonical_coverage")


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]:
    return await _state_check(ws, "decision_legality")


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]:
    return await _state_check(ws, "replay_validity")


async def _metric_check(ws: Workspace, ident: str) -> tuple[float, str]:
    state = await _build_state(ws)
    value = state.metrics.get(ident, 0.0) if state.validity_gate and state.metrics else 0.0
    failed_gates = [
        check_id
        for check_id in GATE_CHECKS
        if state.scores.get(check_id, 0.0) < 1.0 - EPS
    ]
    gate_detail = "none" if not failed_gates else "; ".join(
        f"{check_id}={state.diagnostics.get(check_id, '')}" for check_id in failed_gates
    )
    return value, (
        f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}; "
        f"failed_gates=[{', '.join(failed_gates)}]; {gate_detail}"
    )


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_2")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_3")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_2")
