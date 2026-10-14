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


HORIZON = 72
GRID = 16
UAV = 3
UGVS = (0, 1, 2)
UGV_COUNT = 3
UGV_SENSOR = 2
UAV_SENSOR = 4
AGE_MAX = 8
MAP_BUDGET = 120
REQUEST_BUDGET = 12
CHANNEL_CAPACITY = 4
EPS = 1e-6
MOVE_UGV = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
MOVE_UAV = {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
SCENARIO_NAMES = (
    "forked_corridors",
    "shared_bottleneck",
    "supporter_detour",
    "late_hazard",
    "asymmetric_deadlines",
)
SCENARIOS: dict[str, dict[str, Any]] = {
    "forked_corridors": {
        "uav": (8, 8), "starts": ((2, 2), (2, 13), (13, 2)),
        "goals": ((5, 13), (13, 13), (13, 8)), "deadlines": (48, 60, 56),
        "rectangles": ((7, 8, 1, 10), (7, 8, 12, 15)), "radius": 6,
        "base": 0.10, "burst": (24, 29, 0.35), "constant": 11, "delay": 1,
    },
    "shared_bottleneck": {
        "uav": (8, 8), "starts": ((2, 2), (2, 14), (14, 8)),
        "goals": ((14, 2), (14, 14), (8, 8)), "deadlines": (58, 62, 50),
        "rectangles": ((6, 10, 7, 7), (6, 10, 9, 9)), "radius": 5,
        "base": 0.20, "burst": (30, 39, 0.30), "constant": 23, "delay": 2,
    },
    "supporter_detour": {
        "uav": (2, 14), "starts": ((2, 2), (14, 2), (14, 14)),
        "goals": ((14, 4), (4, 14), (8, 8)), "deadlines": (64, 66, 54),
        "rectangles": ((4, 5, 4, 12), (11, 12, 5, 13)), "radius": 6,
        "base": 0.15, "burst": (18, 27, 0.25), "constant": 41, "delay": 2,
    },
    "late_hazard": {
        "uav": (14, 8), "starts": ((2, 2), (2, 14), (14, 2)),
        "goals": ((5, 5), (12, 12), (9, 6)), "deadlines": (40, 58, 52),
        "rectangles": ((5, 6, 6, 10), (10, 11, 2, 5)), "radius": 5,
        "base": 0.25, "burst": (36, 47, 0.40), "constant": 59, "delay": 3,
    },
    "asymmetric_deadlines": {
        "uav": (8, 2), "starts": ((2, 2), (14, 2), (2, 14)),
        "goals": ((4, 14), (12, 12), (14, 14)), "deadlines": (36, 54, 70),
        "rectangles": ((7, 9, 5, 6),), "radius": 6,
        "base": 0.20, "burst": (12, 23, 0.30), "constant": 73, "delay": 2,
    },
}
OPPORTUNITIES = tuple(range(0, 67, 6))
REQUEST_KEYS = frozenset(("request_id", "request_version", "path_cell_ids", "uncertainty"))
PACKET_ENQUEUE_KEYS = frozenset((
    "receiver", "packet_id", "attempt", "cell_id", "cell_state", "cell_cost", "cell_version", "retry_of",
))
PACKET_EVENT_KEYS = frozenset((
    "packet_id", "attempt", "sender", "receiver", "cell_id", "cell_state", "cell_cost", "cell_version",
    "enqueue_slot", "send_slot", "arrival_state", "retry_of", "receiver_distance", "interference",
    "concurrency", "delivered", "fusion_status",
))
REQUEST_OPPORTUNITY_KEYS = frozenset((
    "robot", "opportunity_t", "active", "request_id", "request_version", "path_cell_ids", "uncertainty",
    "queue_before", "queue_after", "generated", "request_packet_id", "send_slot", "arrival_state", "status",
))
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.scenario_keys": 0.25,
    },
    "result_evidence": {
        "result_evidence.supporter": 0.14285714285714285,
        "result_evidence.seekers": 0.14285714285714285,
        "result_evidence.requests": 0.14285714285714285,
        "result_evidence.packets": 0.14285714285714285,
        "result_evidence.channels": 0.14285714285714285,
        "result_evidence.mission": 0.14285714285714285,
        "result_evidence.types": 0.14285714285714285,
    },
    "decision_legality": {
        "decision_legality.moves": 0.125,
        "decision_legality.paths": 0.125,
        "decision_legality.queues": 0.125,
        "decision_legality.packets": 0.125,
        "decision_legality.channel": 0.125,
        "decision_legality.budgets": 0.125,
        "decision_legality.information": 0.125,
        "decision_legality.terminal": 0.125,
    },
    "replay_validity": {
        "replay_validity.supporter": 0.14285714285714285,
        "replay_validity.seekers": 0.14285714285714285,
        "replay_validity.requests": 0.14285714285714285,
        "replay_validity.packets": 0.14285714285714285,
        "replay_validity.channels": 0.14285714285714285,
        "replay_validity.mission": 0.14285714285714285,
        "replay_validity.metrics": 0.14285714285714285,
    },
}
GATE_CHECKS = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = ident, bool(passed), detail, float(share)


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
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    return AtomVerdict(f"{group}.{name}", passed, detail, ATOM_SHARES[group][f"{group}.{name}"])


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def _cell_id(pos: Sequence[int]) -> int:
    return GRID * int(pos[1]) + int(pos[0])


def _cell(cell_id: int) -> tuple[int, int]:
    return int(cell_id) % GRID, int(cell_id) // GRID


def _obstacles(name: str) -> set[int]:
    out: set[int] = set()
    for xmin, xmax, ymin, ymax in SCENARIOS[name]["rectangles"]:
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                out.add(_cell_id((x, y)))
    # Canonical starts and goals take precedence if a future rectangle overlaps one.
    for pos in tuple(SCENARIOS[name]["starts"]) + tuple(SCENARIOS[name]["goals"]):
        out.discard(_cell_id(pos))
    return out


def _truth(name: str, cell_id: int) -> tuple[str, int]:
    return ("obstacle", 1) if int(cell_id) in _obstacles(name) else ("free", 1)


def _sense_uav(pos: Sequence[int]) -> set[int]:
    x0, y0 = int(pos[0]), int(pos[1])
    return {_cell_id((x, y)) for x in range(GRID) for y in range(GRID) if max(abs(x - x0), abs(y - y0)) <= UAV_SENSOR}


def _sense_ugv(pos: Sequence[int]) -> set[int]:
    x0, y0 = int(pos[0]), int(pos[1])
    return {_cell_id((x, y)) for x in range(GRID) for y in range(GRID) if abs(x - x0) + abs(y - y0) <= UGV_SENSOR}


def _dist(a: Sequence[int], b: Sequence[int]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _interference(name: str, t: int) -> float:
    spec = SCENARIOS[name]
    lo, hi, extra = spec["burst"]
    return min(1.0, float(spec["base"]) + (float(extra) if lo <= t <= hi else 0.0))


def _randoms(name: str, t: int, m: int) -> tuple[float, int]:
    c = int(SCENARIOS[name]["constant"])
    u = ((37 * (t + 1) + 101 * (m + 1) + c) % 997) / 996.0
    d = int(SCENARIOS[name]["delay"])
    delta = 1 + int(((53 * (t + 1) + 89 * (m + 1) + c) % 10) < d)
    return u, delta


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _by_t(rows: Any, field: str = "t") -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for row in _rows(rows):
        if isinstance(row, Mapping) and _integer(row.get(field)):
            out.setdefault(int(row[field]), row)
    return out


def _map_entries(value: Any, value_key: str) -> dict[int, int]:
    out: dict[int, int] = {}
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, Mapping) and _integer(item.get("cell_id")) and _integer(item.get(value_key)):
            out[int(item["cell_id"])] = int(item[value_key])
    return out


def _path_unknown_value(values: Sequence[bool], candidate: Any) -> Any:
    """Keep the candidate's compact or expanded path annotation encoding."""
    if isinstance(candidate, bool):
        return bool(any(values))
    return list(values)


def _path_cost_value(values: Sequence[float], candidate: Any) -> Any:
    """Keep list costs per path cell, or canonicalize a scalar to movement cost."""
    if _finite(candidate):
        return sum(float(value) for value in values[1:])
    return list(values)


def _queue_sizes_encoding(value: Any) -> str | None:
    if (isinstance(value, list) and len(value) == UGV_COUNT
            and all(_integer(size) and int(size) >= 0 for size in value)):
        return "compact"
    if not isinstance(value, list) or len(value) != UGV_COUNT:
        return None
    robots: list[int] = []
    for item in value:
        if (not isinstance(item, Mapping) or set(item) != {"robot", "size"}
                or not _integer(item.get("robot")) or not _integer(item.get("size"))
                or int(item["size"]) < 0):
            return None
        robots.append(int(item["robot"]))
    return "rows" if sorted(robots) == list(UGVS) and len(set(robots)) == UGV_COUNT else None


def _queue_sizes_canonical(value: Any) -> list[int] | None:
    encoding = _queue_sizes_encoding(value)
    if encoding == "compact":
        return [int(size) for size in value]
    if encoding == "rows":
        by_robot = {int(item["robot"]): int(item["size"]) for item in value}
        return [by_robot[robot] for robot in UGVS]
    return None


def _channel_key_canonical(key: Any) -> Any:
    if not isinstance(key, Mapping):
        return key
    if key.get("kind") == "map" and set(key) == {"kind", "packet_id", "attempt", "receiver"}:
        return {name: value for name, value in key.items() if name != "receiver"}
    return dict(key)


def _channel_keys_canonical(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [_channel_key_canonical(key) for key in value]


def _path_from_state(row: Mapping[str, Any]) -> list[int]:
    raw = row.get("belief_path_cell_ids", [])
    return [int(x) for x in raw] if isinstance(raw, list) and all(_integer(x) for x in raw) else []


def _legal_full_path(name: str, robot: int, pos: Sequence[int], path: Sequence[int], known: Mapping[int, tuple[str, int]]) -> bool:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        return False
    if not path:
        return True
    goal = _cell_id(SCENARIOS[name]["goals"][robot])
    if path[0] != _cell_id(pos) or path[-1] != goal:
        return False
    for index, cid in enumerate(path):
        if not _integer(cid) or not 0 <= int(cid) < GRID * GRID:
            return False
        if int(cid) in known and known[int(cid)][0] == "obstacle":
            return False
        if index and _manhattan(_cell(path[index - 1]), _cell(int(cid))) != 1:
            return False
    return True


def _legal_prefix(full_path: Sequence[int], prefix: Sequence[int], pos: Sequence[int]) -> bool:
    if not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes)) or not 1 <= len(prefix) <= 6:
        return False
    if not isinstance(full_path, Sequence) or isinstance(full_path, (str, bytes)) or not full_path:
        return False
    try:
        if not all(_integer(x) for x in full_path) or not all(_integer(x) for x in prefix):
            return False
        if list(prefix) != list(full_path[: min(6, len(full_path))]):
            return False
        if int(prefix[0]) != _cell_id(pos):
            return False
        return all(
            _manhattan(_cell(int(prefix[i - 1])), _cell(int(prefix[i]))) == 1
            for i in range(1, len(prefix))
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def _state_row(seekers: list[Mapping[str, Any]], robot: int, t: int) -> Mapping[str, Any]:
    rows = seekers[robot].get("states", []) if robot < len(seekers) else []
    by = _by_t(rows, "state_t")
    return by.get(t, {})


def _action_row(seekers: list[Mapping[str, Any]], robot: int, t: int) -> Mapping[str, Any]:
    rows = seekers[robot].get("actions", []) if robot < len(seekers) else []
    return _by_t(rows, "t").get(t, {})


def _supporter_state(supporter: Mapping[str, Any], t: int) -> Mapping[str, Any]:
    return _by_t(supporter.get("states", []), "state_t").get(t, {})


def _supporter_action(supporter: Mapping[str, Any], t: int) -> Mapping[str, Any]:
    return _by_t(supporter.get("actions", []), "t").get(t, {})


def _key_sort(key: Any) -> tuple[Any, ...]:
    if not isinstance(key, Mapping):
        return ("invalid",)
    return (str(key.get("kind", "")), int(key.get("robot", -1)) if _integer(key.get("robot")) else -1,
            int(key.get("packet_id", -1)) if _integer(key.get("packet_id")) else -1,
            int(key.get("attempt", -1)) if _integer(key.get("attempt")) else -1)


def _replay_scenario(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    supporter = _as_mapping(row.get("supporter_trace"))
    seeker_rows = [x for x in _rows(row.get("seeker_traces")) if isinstance(x, Mapping)]
    seeker_by_robot: list[Mapping[str, Any]] = [{} for _ in UGVS]
    for item in seeker_rows:
        if _integer(item.get("robot")) and 0 <= int(item["robot"]) < UGV_COUNT:
            seeker_by_robot[int(item["robot"])] = item
    channel_input = _by_t(row.get("channel_slots", []), "t")
    positions = [list(spec["uav"])] + [list(x) for x in spec["starts"]]
    uav_seen: dict[int, tuple[str, int, int]] = {}
    ugv_maps: list[dict[int, tuple[str, int, int]]] = [{} for _ in UGVS]
    receiver_versions: list[dict[int, int]] = [{} for _ in UGVS]
    ages: list[dict[int, int]] = [{} for _ in UGVS]
    path_fusion: list[dict[int, int]] = [{} for _ in UGVS]
    uav_versions: dict[int, int] = {}
    uav_seen_ids = _sense_uav(positions[0])
    for cid in sorted(uav_seen_ids):
        state, cost = _truth(name, cid)
        uav_versions[cid] = 1
        uav_seen[cid] = (state, cost, 1)
    for robot in UGVS:
        for cid in sorted(_sense_ugv(positions[robot + 1])):
            state, cost = _truth(name, cid)
            ugv_maps[robot][cid] = (state, cost, 0)

    pending_maps: dict[int, list[dict[str, Any]]] = {}
    pending_requests: dict[int, list[dict[str, Any]]] = {}
    request_queues: list[list[dict[str, Any]]] = [[] for _ in UGVS]
    arrived_requests: list[list[dict[str, Any]]] = [[] for _ in UGVS]
    request_events: list[dict[str, Any]] = []
    packet_events: dict[tuple[int, int], dict[str, Any]] = {}
    packet_queues: list[list[tuple[int, int]]] = [[] for _ in UGVS]
    seen_packet_ids: set[int] = set()
    request_versions = [0] * UGV_COUNT
    request_budget = [REQUEST_BUDGET] * UGV_COUNT
    map_remaining = MAP_BUDGET
    counters = {"ground": 0.0, "uav": 0.0, "blocked": 0, "separation": 0, "detours": 0}
    safety = [True, True, True, True]
    safety_violation = False
    goals = [False] * UGV_COUNT
    arrivals: list[int | None] = [None] * UGV_COUNT
    position_history: list[list[list[int]]] = []
    move_cost_by_robot = [0.0] * UGV_COUNT
    useful_cells: set[tuple[int, int, int]] = set()
    freshness_samples: list[list[float]] = [[] for _ in UGVS]
    expected_supporter_states: list[dict[str, Any]] = []
    expected_supporter_actions: list[dict[str, Any]] = []
    expected_seeker_states: list[list[dict[str, Any]]] = [[] for _ in UGVS]
    expected_seeker_actions: list[list[dict[str, Any]]] = [[] for _ in UGVS]
    expected_opportunities: list[dict[str, Any]] = []
    expected_channels: list[dict[str, Any]] = []
    expected_mission: list[dict[str, Any]] = []
    defects: list[str] = []
    illegal: dict[str, list[str]] = {x: [] for x in ("moves", "paths", "queues", "packets", "channel", "budgets", "information", "terminal")}
    prior_paths = [[] for _ in UGVS]

    def current_supporter_state(t: int) -> dict[str, Any]:
        return {
            "state_t": t, "x": int(positions[0][0]), "y": int(positions[0][1]),
            "observed_cell_ids": sorted(uav_seen),
            "map_versions": [{"cell_id": cid, "version": uav_versions[cid]} for cid in sorted(uav_versions)],
            "map_queue_sizes": [{"robot": r, "size": len(packet_queues[r])} for r in UGVS],
            "remaining_map_attempts": map_remaining, "safe": bool(safety[0]),
        }

    def current_seeker_state(robot: int, t: int) -> dict[str, Any]:
        source = _state_row(seeker_by_robot, robot, t)
        pos = positions[robot + 1]
        known = ugv_maps[robot]
        path = _path_from_state(source)
        if path and not _legal_full_path(name, robot, pos, path, {k: (v[0], v[1]) for k, v in known.items()}):
            illegal["paths"].append(f"robot={robot},state={t}: illegal belief path")
        observed = [{"cell_id": cid, "state": known[cid][0], "cost": known[cid][1], "version": known[cid][2]} for cid in sorted(known)]
        path_unknown = [cid not in known for cid in path]
        path_cost = [known[cid][1] if cid in known else 1 for cid in path]
        return {
            "state_t": t, "x": int(pos[0]), "y": int(pos[1]), "observed_cells": observed,
            "belief_path_cell_ids": path,
            "belief_path_unknown": _path_unknown_value(path_unknown, source.get("belief_path_unknown")),
            "belief_path_cost": _path_cost_value(path_cost, source.get("belief_path_cost")),
            "request_queue": [int(e["request_id"]) for e in request_queues[robot]],
            "request_versions": request_versions[robot],
            "map_versions": [{"cell_id": cid, "version": receiver_versions[robot][cid]} for cid in sorted(receiver_versions[robot])],
            "ages": [{"cell_id": cid, "age": ages[robot][cid]} for cid in sorted(ages[robot])],
            "goal_reached": bool(goals[robot]), "safe": bool(safety[robot + 1]),
            "remaining_request_budget": request_budget[robot],
        }

    def mission_row(t: int) -> dict[str, Any]:
        return {
            "state_t": t, "goal_reached": list(goals), "arrival_states": list(arrivals),
            "ground_move_cost": float(counters["ground"]), "uav_move_cost": float(counters["uav"]),
            "blocked_moves": int(counters["blocked"]), "separation_interventions": int(counters["separation"]),
            "redundant_detours": int(counters["detours"]), "remaining_map_attempts": int(map_remaining),
            "safety_violation": bool(safety_violation), "terminal": t == HORIZON,
        }

    def compare_state(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
        mismatch = _first_mismatch(actual, expected, label)
        if mismatch:
            defects.append(mismatch)

    def process_boundary(t: int) -> None:
        nonlocal safety_violation
        if t > 0:
            for robot in UGVS:
                for cid in list(ages[robot]):
                    ages[robot][cid] = min(AGE_MAX, ages[robot][cid] + 1)
                for cid in list(path_fusion[robot]):
                    path_fusion[robot][cid] = min(AGE_MAX, path_fusion[robot][cid] + 1)
        for event in pending_maps.pop(t, []):
            receiver = int(event["receiver"])
            cid = int(event["cell_id"])
            current = receiver_versions[receiver].get(cid, 0)
            if int(event["cell_version"]) > current:
                event["fusion_status"] = "fused"
                receiver_versions[receiver][cid] = int(event["cell_version"])
                ages[receiver][cid] = 0
                ugv_maps[receiver][cid] = (event["cell_state"], int(event["cell_cost"]), int(event["cell_version"]))
                if cid in prior_paths[receiver] and cid not in known_before_by_robot[receiver]:
                    useful_cells.add((receiver, cid, int(event["cell_version"])))
                path_fusion[receiver][cid] = 0
            elif int(event["cell_version"]) == current:
                event["fusion_status"] = "duplicate"
            else:
                event["fusion_status"] = "stale"
        for event in pending_requests.pop(t, []):
            robot = int(event["robot"])
            if t >= int(event["generation_slot"]) + 8:
                event["status"] = "expired"
                continue
            event["status"] = "delivered"
            arrived_requests[robot] = [old for old in arrived_requests[robot] if int(old["request_version"]) >= int(event["request_version"])]
            arrived_requests[robot].append(event)
        for robot in UGVS:
            while request_queues[robot] and t >= int(request_queues[robot][0]["generation_slot"]) + 8:
                event = request_queues[robot].pop(0)
                event["status"] = "expired"
        for robot in UGVS:
            for cid in sorted(_sense_ugv(positions[robot + 1])):
                state, cost = _truth(name, cid)
                old = ugv_maps[robot].get(cid)
                version = old[2] if old else 0
                ugv_maps[robot][cid] = (state, cost, version)

    for t in range(HORIZON + 1):
        known_before_by_robot: list[set[int]] = [set() for _ in UGVS]
        if t > 0:
            for robot in UGVS:
                known_before_by_robot[robot].update(ugv_maps[robot])
        process_boundary(t)
        # Fixed opportunity creation is driven by the candidate's local request action.
        if t < HORIZON and t in OPPORTUNITIES:
            for robot in UGVS:
                active = not goals[robot]
                action = _action_row(seeker_by_robot, robot, t)
                raw_request = action.get("request") if isinstance(action, Mapping) else None
                queue_before = len(request_queues[robot])
                event: dict[str, Any] | None = None
                generated = bool(active and raw_request is not None)
                if generated:
                    if request_budget[robot] <= 0:
                        illegal["budgets"].append(f"robot={robot},slot={t}: request budget exhausted")
                    request_budget[robot] = max(0, request_budget[robot] - 1)
                    if not isinstance(raw_request, Mapping):
                        illegal["paths"].append(f"robot={robot},slot={t}: request must be an object")
                        request: Mapping[str, Any] = {}
                    else:
                        request = raw_request
                        if set(request) != set(REQUEST_KEYS):
                            illegal["paths"].append(
                                f"robot={robot},slot={t}: request fields expected={sorted(REQUEST_KEYS)} actual={sorted(request)}"
                            )
                    rid = len(request_events)
                    supplied_id = request.get("request_id")
                    if not _integer(supplied_id) or int(supplied_id) != rid:
                        illegal["paths"].append(
                            f"robot={robot},slot={t}: request_id expected={rid} actual={supplied_id!r}"
                        )
                    version = request.get("request_version")
                    previous_version = request_versions[robot]
                    request_versions[robot] = int(version) if _integer(version) else request_versions[robot]
                    event = {
                        "request_id": rid, "robot": robot, "opportunity_t": t,
                        "request_version": int(version) if _integer(version) else -1,
                        "path_cell_ids": list(request.get("path_cell_ids", [])) if isinstance(request.get("path_cell_ids", []), list) else [],
                        "uncertainty": request.get("uncertainty"), "generation_slot": t,
                        "status": "queued", "send_slot": None, "arrival_state": None,
                    }
                    request_events.append(event)
                    path = _path_from_state(_state_row(seeker_by_robot, robot, t))
                    if not _integer(version) or int(version) <= previous_version:
                        illegal["paths"].append(f"robot={robot},slot={t}: request version not increasing")
                    if not _finite(event["uncertainty"]) or not 0 <= float(event["uncertainty"]) <= 1:
                        illegal["paths"].append(f"robot={robot},slot={t}: uncertainty")
                    if not _legal_prefix(path, event["path_cell_ids"], positions[robot + 1]):
                        illegal["paths"].append(f"robot={robot},slot={t}: request is not path prefix")
                    if queue_before >= 2:
                        event["status"] = "queue_full"
                        generated = True
                    else:
                        request_queues[robot].append(event)
                elif raw_request is not None:
                    illegal["terminal"].append(f"robot={robot},slot={t}: inactive request")
                queue_after = len(request_queues[robot])
                expected_opportunities.append({
                    "robot": robot, "opportunity_t": t, "active": active,
                    "request_id": event["request_id"] if event else None,
                    "request_version": event["request_version"] if event else None,
                    "path_cell_ids": event["path_cell_ids"] if event else None,
                    "uncertainty": event["uncertainty"] if event else None,
                    "queue_before": queue_before, "queue_after": queue_after,
                    "generated": generated, "request_packet_id": event["request_id"] if event else None,
                    "send_slot": None, "arrival_state": None,
                    "status": event["status"] if event else "none",
                })
        # Action-start evidence.
        supporter_state = current_supporter_state(t)
        expected_supporter_states.append(supporter_state)
        compare_state(_supporter_state(supporter, t), supporter_state, f"{name}.supporter.states[{t}]")
        for robot in UGVS:
            seeker_state = current_seeker_state(robot, t)
            expected_seeker_states[robot].append(seeker_state)
            compare_state(_state_row(seeker_by_robot, robot, t), seeker_state, f"{name}.seeker[{robot}].states[{t}]")
            path = seeker_state["belief_path_cell_ids"]
            if any(cid not in ugv_maps[robot] for cid in path):
                fusion_ages = [path_fusion[robot][cid] for cid in path if cid in path_fusion[robot]]
                freshness_samples[robot].append(_clip(1 - min(fusion_ages) / AGE_MAX) if fusion_ages else 0.0)
            prior_paths[robot] = list(path)
        expected_mission.append(mission_row(t))
        compare_state(_rows(row.get("mission_trace"))[t] if t < len(_rows(row.get("mission_trace"))) else {}, expected_mission[-1], f"{name}.mission[{t}]")
        position_history.append([list(x) for x in positions])
        if t == HORIZON:
            break

        # Candidate packet enqueue decisions are made before channel admission.
        sa = _supporter_action(supporter, t)
        move_uav = (sa.get("move_dx", 0), sa.get("move_dy", 0)) if isinstance(sa, Mapping) else (0, 0)
        enqueue = sa.get("enqueue_packets", []) if isinstance(sa, Mapping) else []
        if not isinstance(enqueue, list):
            illegal["packets"].append(f"slot={t}: enqueue_packets not list")
            enqueue = []
        if len(enqueue) > 3:
            illegal["queues"].append(f"slot={t}: more than three map attempts enqueued")
        for raw_item in enqueue:
            if not isinstance(raw_item, Mapping):
                illegal["packets"].append(f"slot={t}: map enqueue item must be an object")
                continue
            item = raw_item
            if set(item) != set(PACKET_ENQUEUE_KEYS):
                illegal["packets"].append(
                    f"slot={t}: map enqueue fields expected={sorted(PACKET_ENQUEUE_KEYS)} actual={sorted(item)}"
                )
            receiver, pid, attempt = item.get("receiver"), item.get("packet_id"), item.get("attempt")
            cid = item.get("cell_id")
            if not (_integer(receiver) and receiver in UGVS and _integer(pid) and pid >= 0 and _integer(attempt) and attempt >= 0 and _integer(cid) and 0 <= cid < GRID * GRID):
                illegal["packets"].append(f"slot={t}: malformed map enqueue")
                continue
            receiver, pid, attempt, cid = int(receiver), int(pid), int(attempt), int(cid)
            if item.get("cell_state") not in ("free", "obstacle"):
                illegal["packets"].append(f"slot={t}: invalid map cell_state={item.get('cell_state')!r}")
            if not _integer(item.get("cell_cost")) or int(item.get("cell_cost", -1)) < 0:
                illegal["packets"].append(f"slot={t}: invalid map cell_cost={item.get('cell_cost')!r}")
            if not _integer(item.get("cell_version")) or int(item.get("cell_version", 0)) <= 0:
                illegal["packets"].append(f"slot={t}: invalid map cell_version={item.get('cell_version')!r}")
            if attempt == 0:
                if pid in seen_packet_ids:
                    illegal["packets"].append(f"slot={t}: duplicate logical packet {pid}")
                seen_packet_ids.add(pid)
                if item.get("retry_of") is not None:
                    illegal["packets"].append(f"slot={t}: attempt-0 retry_of must be null")
                retry_of = None
            else:
                previous = packet_events.get((pid, attempt - 1))
                retry_of = item.get("retry_of")
                if previous is None or previous.get("fusion_status") not in ("out_of_range", "interference_lost") or retry_of != attempt - 1:
                    illegal["packets"].append(f"slot={t}: invalid retry {pid}/{attempt}")
                elif any(item.get(field) != previous.get(field) for field in ("receiver", "cell_id", "cell_state", "cell_cost", "cell_version")):
                    illegal["packets"].append(f"slot={t}: retry {pid}/{attempt} changes logical packet identity or payload")
            if (pid, attempt) in packet_events:
                illegal["packets"].append(f"slot={t}: duplicate packet attempt key {(pid, attempt)!r}")
                continue
            # A failed enqueue into a full receiver queue is not an enqueued
            # attempt, so it must not consume the map-attempt budget.  The
            # public contract charges the budget at successful queue entry.
            if len(packet_queues[receiver]) >= 6:
                illegal["queues"].append(f"slot={t}: map queue full receiver={receiver}")
                continue
            if map_remaining <= 0:
                illegal["budgets"].append(f"slot={t}: map budget exhausted")
            else:
                map_remaining -= 1
            if cid not in uav_seen:
                illegal["information"].append(f"slot={t}: UAV has not sensed cell {cid}")
                state, cost, ver = "free", 1, 0
            else:
                state, cost, ver = uav_seen[cid]
            if item.get("cell_state") != state or item.get("cell_cost") != cost or item.get("cell_version") != ver:
                illegal["information"].append(f"slot={t}: map payload is not latest sensed truth cell={cid}")
            event = {
                "packet_id": pid, "attempt": attempt, "sender": UAV, "receiver": receiver,
                "cell_id": cid, "cell_state": item.get("cell_state", state), "cell_cost": item.get("cell_cost", cost),
                "cell_version": item.get("cell_version", ver), "enqueue_slot": t, "send_slot": None,
                "arrival_state": None, "retry_of": retry_of, "receiver_distance": None,
                "interference": None, "concurrency": None, "delivered": False, "fusion_status": "pending",
            }
            packet_events[(pid, attempt)] = event
            packet_queues[receiver].append((pid, attempt))

        ch = _as_mapping(channel_input.get(t, {}))
        submitted = ch.get("submitted_keys", []) if isinstance(ch.get("submitted_keys", []), list) else []
        request_sizes_before_channel = [len(request_queues[r]) for r in UGVS]
        map_sizes_before_channel = [len(packet_queues[r]) for r in UGVS]
        for field in ("request_queue_sizes", "map_queue_sizes"):
            if _queue_sizes_encoding(ch.get(field)) is None:
                illegal["channel"].append(f"slot={t}: {field} must use a supported queue-size encoding")
        admitted: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for key in submitted:
            km = _as_mapping(key)
            kind = km.get("kind")
            if kind == "request" and set(km) != {"kind", "robot", "opportunity_t"}:
                illegal["channel"].append(f"slot={t}: request key fields")
            if kind == "map" and set(km) not in (
                    {"kind", "packet_id", "attempt"},
                    {"kind", "packet_id", "attempt", "receiver"},
            ):
                illegal["channel"].append(f"slot={t}: map key fields")
            if kind == "map" and "receiver" in km and (
                    not _integer(km.get("receiver")) or int(km["receiver"]) not in UGVS):
                illegal["channel"].append(f"slot={t}: map key receiver")
            if kind not in ("request", "map"):
                illegal["channel"].append(f"slot={t}: unknown key kind")
            canonical_key = _channel_key_canonical(km)
            canon = tuple(sorted((str(k), json.dumps(v, sort_keys=True, separators=(",", ":"))) for k, v in canonical_key.items()))
            if canon in seen_keys:
                illegal["channel"].append(f"slot={t}: duplicate submitted key")
                continue
            seen_keys.add(canon)
            event: dict[str, Any] | None = None
            queue_owner: tuple[str, int] | None = None
            if kind == "request" and _integer(km.get("robot")) and _integer(km.get("opportunity_t")):
                for candidate in request_queues[int(km["robot"])] if int(km["robot"]) in UGVS else []:
                    if candidate["opportunity_t"] == int(km["opportunity_t"]):
                        event, queue_owner = candidate, ("request", int(km["robot"]))
                        break
            elif kind == "map" and _integer(km.get("packet_id")) and _integer(km.get("attempt")):
                pair = (int(km["packet_id"]), int(km["attempt"]))
                for receiver in UGVS:
                    if packet_queues[receiver] and packet_queues[receiver][0] == pair:
                        event, queue_owner = packet_events.get(pair), ("map", receiver)
                        break
            if event is None:
                illegal["channel"].append(f"slot={t}: submitted key is not a queue head")
                continue
            if (kind == "map" and "receiver" in km and _integer(km.get("receiver"))
                    and queue_owner is not None and int(km["receiver"]) != queue_owner[1]):
                illegal["channel"].append(f"slot={t}: map key receiver does not match destination queue")
            if len(admitted) >= CHANNEL_CAPACITY:
                continue
            admitted.append(event)
            if queue_owner and queue_owner[0] == "request":
                request_queues[queue_owner[1]].remove(event)
            elif queue_owner:
                packet_queues[queue_owner[1]].pop(0)

        h = _interference(name, t)
        for m, event in enumerate(admitted):
            event["send_slot"] = t
            event["interference"] = h
            event["concurrency"] = len(admitted)
            if event.get("sender") == UAV:
                receiver_pos = positions[int(event["receiver"]) + 1]
                sender_pos = positions[0]
            else:
                receiver_pos = positions[0]
                sender_pos = positions[int(event["robot"]) + 1]
            distance = _manhattan(sender_pos, receiver_pos)
            event["receiver_distance"] = distance
            u, delta = _randoms(name, t, m)
            good = distance <= int(spec["radius"]) and h + 0.1 * len(admitted) <= 0.9 and u <= 0.85 - 0.05 * h
            if not good:
                event["delivered"] = False
                event["fusion_status"] = "out_of_range" if distance > int(spec["radius"]) else "interference_lost"
                if event.get("sender") != UAV:
                    event["status"] = "lost"
            else:
                event["delivered"] = True
                event["arrival_state"] = t + delta
                if event.get("sender") != UAV:
                    event["status"] = "sent"
                    event["arrival_state"] = t + delta
                    pending_requests.setdefault(t + delta, []).append(event)
                elif t + delta <= HORIZON:
                    pending_maps.setdefault(t + delta, []).append(event)
        # Write send metadata into request opportunity records.
        for op in expected_opportunities:
            if op["opportunity_t"] <= t and op["request_id"] is not None:
                event = request_events[op["request_id"]]
                op["status"] = event["status"]
                op["send_slot"] = event["send_slot"]
                op["arrival_state"] = event["arrival_state"]

        expected_admitted_keys = [
            {"kind": "request", "robot": int(e["robot"]), "opportunity_t": int(e["opportunity_t"])} if e.get("sender") != UAV else
            {"kind": "map", "packet_id": int(e["packet_id"]), "attempt": int(e["attempt"])} for e in admitted
        ]
        actual_admitted_keys = ch.get("admitted_keys", [])
        if isinstance(actual_admitted_keys, list):
            for index, key in enumerate(actual_admitted_keys):
                km = _as_mapping(key)
                if km.get("kind") != "map" or "receiver" not in km:
                    continue
                if (not _integer(km.get("receiver")) or index >= len(admitted)
                        or admitted[index].get("sender") != UAV
                        or int(km["receiver"]) != int(admitted[index]["receiver"])):
                    illegal["channel"].append(f"slot={t}: admitted map key receiver does not match destination queue")
        expected_channels.append({
            "t": t, "interference": h,
            "request_queue_sizes": list(request_sizes_before_channel),
            "map_queue_sizes": list(map_sizes_before_channel),
            "submitted_keys": _channel_keys_canonical(submitted),
            "admitted_keys": expected_admitted_keys,
            "units": len(admitted), "capacity_remaining": CHANNEL_CAPACITY - len(admitted),
        })
        actual_channel = dict(ch)
        for field in ("request_queue_sizes", "map_queue_sizes"):
            canonical = _queue_sizes_canonical(actual_channel.get(field))
            if canonical is not None:
                actual_channel[field] = canonical
        for field in ("submitted_keys", "admitted_keys"):
            actual_channel[field] = _channel_keys_canonical(actual_channel.get(field))
        compare_state(actual_channel, expected_channels[-1], f"{name}.channel[{t}]")

        # Motion and fixed safety executor.
        proposed_ugv: list[tuple[int, int]] = []
        for robot in UGVS:
            action = _action_row(seeker_by_robot, robot, t)
            dx, dy = action.get("move_dx", 0), action.get("move_dy", 0)
            if not (_integer(dx) and _integer(dy) and (int(dx), int(dy)) in MOVE_UGV):
                illegal["moves"].append(f"robot={robot},slot={t}: move vector")
                proposed_ugv.append((0, 0))
            else:
                proposed_ugv.append((int(dx), int(dy)))
            if action.get("request") is not None and t not in OPPORTUNITIES:
                illegal["terminal"].append(f"robot={robot},slot={t}: request outside opportunity")
        old_ugv = [list(positions[r + 1]) for r in UGVS]
        accepted_ugv = [list(x) for x in old_ugv]
        for robot in UGVS:
            dx, dy = proposed_ugv[robot]
            if goals[robot] and (dx, dy) != (0, 0):
                illegal["terminal"].append(f"robot={robot},slot={t}: moved after goal")
                continue
            if (dx, dy) == (0, 0):
                continue
            dest = [old_ugv[robot][0] + dx, old_ugv[robot][1] + dy]
            if not (0 <= dest[0] < GRID and 0 <= dest[1] < GRID):
                illegal["moves"].append(f"robot={robot},slot={t}: out of domain")
                continue
            cid = _cell_id(dest)
            if cid in ugv_maps[robot] and ugv_maps[robot][cid][0] == "obstacle":
                illegal["moves"].append(f"robot={robot},slot={t}: known obstacle")
                continue
            if any(_dist(dest, accepted_ugv[other]) < 1 - EPS for other in UGVS if other < robot) or _dist(dest, positions[0]) < 0.5 - EPS:
                counters["separation"] += 1
                safety[robot + 1] = False
                safety_violation = True
                continue
            if any(proposed_ugv[other] == (old_ugv[robot][0] - old_ugv[other][0], old_ugv[robot][1] - old_ugv[other][1]) and proposed_ugv[robot] == (old_ugv[other][0] - old_ugv[robot][0], old_ugv[other][1] - old_ugv[robot][1]) for other in UGVS if other != robot):
                counters["separation"] += 1
                safety[robot + 1] = False
                safety_violation = True
                continue
            if cid in _obstacles(name):
                counters["blocked"] += 1
                safety[robot + 1] = False
                safety_violation = True
                continue
            if _manhattan(old_ugv[robot], spec["goals"][robot]) < _manhattan(dest, spec["goals"][robot]):
                counters["detours"] += 1
            accepted_ugv[robot] = dest
            counters["ground"] += 1
            move_cost_by_robot[robot] += 1
        # UAV action and safety.
        if not (_integer(move_uav[0]) and _integer(move_uav[1]) and (int(move_uav[0]), int(move_uav[1])) in MOVE_UAV):
            illegal["moves"].append(f"uav,slot={t}: move vector")
            move_uav = (0, 0)
        proposed_uav = [positions[0][0] + int(move_uav[0]), positions[0][1] + int(move_uav[1])]
        accepted_uav = list(positions[0])
        if not (0 <= proposed_uav[0] < GRID and 0 <= proposed_uav[1] < GRID):
            illegal["moves"].append(f"uav,slot={t}: out of domain")
        elif any(_dist(proposed_uav, p) < 0.5 - EPS for p in accepted_ugv):
            counters["separation"] += 1
            safety[0] = False
            safety_violation = True
        else:
            accepted_uav = proposed_uav
            if move_uav != (0, 0):
                counters["uav"] += 1.5
        positions = [accepted_uav] + [list(x) for x in accepted_ugv]
        for robot in UGVS:
            if not goals[robot] and tuple(positions[robot + 1]) == tuple(spec["goals"][robot]):
                goals[robot] = True
                arrivals[robot] = t + 1
        for cid in sorted(_sense_uav(positions[0])):
            state, cost = _truth(name, cid)
            if cid not in uav_versions:
                uav_versions[cid] = 1
                uav_seen[cid] = (state, cost, 1)
        for robot in UGVS:
            for cid in sorted(_sense_ugv(positions[robot + 1])):
                state, cost = _truth(name, cid)
                old = ugv_maps[robot].get(cid)
                ugv_maps[robot][cid] = (state, cost, old[2] if old else 0)
        # Preserve raw action evidence but use the exact expected communication row.
        expected_supporter_actions.append({
            "t": t, "move_dx": int(move_uav[0]), "move_dy": int(move_uav[1]),
            "enqueue_packets": enqueue,
        })
        for robot in UGVS:
            action = _action_row(seeker_by_robot, robot, t)
            expected_seeker_actions[robot].append({
                "t": t, "move_dx": proposed_ugv[robot][0], "move_dy": proposed_ugv[robot][1],
                "request": action.get("request") if t in OPPORTUNITIES else None,
            })
    # Finalize packet rows, opportunity statuses and action evidence.
    packet_list = sorted(packet_events.values(), key=lambda x: (int(x["packet_id"]), int(x["attempt"])))
    for event in packet_list:
        if event["send_slot"] is None:
            event["receiver_distance"] = None
    for event in request_events:
        if event["status"] == "queued" and HORIZON >= event["generation_slot"] + 8:
            event["status"] = "expired"
        elif event["status"] == "sent" and (
            event["arrival_state"] is None
            or int(event["arrival_state"]) >= event["generation_slot"] + 8
        ):
            event["status"] = "expired"
        for op in expected_opportunities:
            if op["request_id"] == event["request_id"]:
                op.update(send_slot=event["send_slot"], arrival_state=event["arrival_state"], status=event["status"])
    actual_opportunities = _rows(row.get("request_opportunities"))
    # Queue snapshots are optional for opportunities that did not generate a
    # request.  Normalize an explicit null to the independently replayed
    # snapshot for comparison, while retaining all other fields and preserving
    # missing-key diagnostics.
    if len(actual_opportunities) == len(expected_opportunities):
        normalized_opportunities: list[Any] = []
        for actual, expected in zip(actual_opportunities, expected_opportunities):
            if isinstance(actual, Mapping) and isinstance(expected, Mapping) and expected.get("generated") is False:
                item = dict(actual)
                for field in ("queue_before", "queue_after"):
                    if field in item and item[field] is None:
                        item[field] = expected[field]
                normalized_opportunities.append(item)
            else:
                normalized_opportunities.append(actual)
        actual_opportunities = normalized_opportunities
    compare_state(actual_opportunities, expected_opportunities, f"{name}.request_opportunities")
    compare_state(_rows(row.get("map_packets")), packet_list, f"{name}.map_packets")
    compare_state(_rows(supporter.get("actions")), expected_supporter_actions, f"{name}.supporter.actions")
    for robot in UGVS:
        compare_state(_rows(seeker_by_robot[robot].get("actions")), expected_seeker_actions[robot], f"{name}.seeker[{robot}].actions")
    expected = {
        "scenario": name,
        "supporter_trace": {"states": expected_supporter_states, "actions": expected_supporter_actions},
        "seeker_traces": [{"robot": r, "states": expected_seeker_states[r], "actions": expected_seeker_actions[r]} for r in UGVS],
        "request_opportunities": expected_opportunities,
        "map_packets": packet_list,
        "channel_slots": expected_channels,
        "mission_trace": expected_mission,
    }
    metrics = _scenario_metrics(name, position_history, arrivals, move_cost_by_robot, counters, freshness_samples, useful_cells, packet_list)
    return {"expected": expected, "defects": defects, "illegal": illegal, "metrics": metrics}


def _scenario_metrics(name: str, position_history: list[list[list[int]]], arrivals: list[int | None], move_cost: list[float], counters: Mapping[str, Any], freshness: list[list[float]], useful: set[tuple[int, int, int]], packets: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    spec = SCENARIOS[name]
    completion: list[float] = []
    paths: list[float] = []
    progress: list[float] = []
    for robot in UGVS:
        arrival = arrivals[robot]
        deadline = int(spec["deadlines"][robot])
        if arrival is not None and arrival <= deadline:
            completion.append(0.7 + 0.3 * (1 - arrival / deadline))
        elif arrival is not None and arrival <= HORIZON:
            completion.append(0.4 * (HORIZON - arrival) / max(1, HORIZON - deadline))
        else:
            completion.append(0.0)
        start = spec["starts"][robot]
        goal = spec["goals"][robot]
        d0 = _manhattan(start, goal)
        state = min(deadline, HORIZON)
        final_pos = position_history[state][robot + 1] if position_history else list(start)
        progress.append(_clip(1 - _manhattan(final_pos, goal) / max(1, d0)))
        cstar = _shortest_cost(name, start, goal)
        paths.append(_clip(cstar / max(move_cost[robot], 1e-6)) if arrival is not None and arrival <= deadline else 0.0)
    p1 = sum(completion) / 3
    safety = _clip(1 - (int(counters["blocked"]) + int(counters["separation"])) / HORIZON)
    q = [x * safety for x in progress]
    p2 = 0.7 * (sum(q) / 3) + 0.3 * min(q)
    p3 = sum(paths) / 3
    freshness_scores = []
    for values in freshness:
        freshness_scores.append(sum(values) / len(values) if values else 0.5)
    a1 = sum(freshness_scores) / 3
    admitted = sum(1 for p in packets if p.get("send_slot") is not None)
    a2 = 0.5 * _clip(1 - admitted / MAP_BUDGET) + 0.5 * len(useful) / max(1, admitted)
    return {"primary_metrics_1": _clip(p1), "primary_metrics_2": _clip(p2), "primary_metrics_3": _clip(p3), "auxiliary_metrics_1": _clip(a1), "auxiliary_metrics_2": _clip(a2)}


def _metrics(replayed: Mapping[str, Mapping[str, Any]], records: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Aggregate task metrics from independently replayed scenario states only."""
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("metrics require every canonical scenario")
    values = [replayed[name]["metrics"] for name in SCENARIO_NAMES]
    if not values or any(set(row) != set(values[0]) for row in values):
        raise ValueError("replayed metric keys are inconsistent")
    result = {key: sum(float(row[key]) for row in values) / len(values) for key in values[0]}
    if any(not _finite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return result


def _shortest_cost(name: str, start: Sequence[int], goal: Sequence[int]) -> float:
    blocked = _obstacles(name)
    source, target = tuple(start), tuple(goal)
    q = [source]
    dist = {source: 0}
    while q:
        p = q.pop(0)
        if p == target:
            return float(dist[p])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (p[0] + dx, p[1] + dy)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID) or _cell_id(nxt) in blocked or nxt in dist:
                continue
            dist[nxt] = dist[p] + 1
            q.append(nxt)
    return float(max(1, _manhattan(start, goal)))


def _first_mismatch(actual: Any, expected: Any, path: str) -> str | None:
    if isinstance(expected, float):
        if not _finite(actual) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=EPS):
            return f"{path}: expected={expected!r}, actual={actual!r}"
        return None
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected object"
        if set(actual) != set(expected):
            return f"{path}: keys expected={sorted(expected)} actual={sorted(actual)}"
        for key, value in expected.items():
            mismatch = _first_mismatch(actual[key], value, f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path}: list length/type mismatch"
        for i, (a, e) in enumerate(zip(actual, expected)):
            mismatch = _first_mismatch(a, e, f"{path}[{i}]")
            if mismatch:
                return mismatch
        return None
    if actual != expected:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


def _scenario_shape(row: Any) -> bool:
    return isinstance(row, Mapping) and set(row) == {"scenario", "supporter_trace", "seeker_traces", "request_opportunities", "map_packets", "channel_slots", "mission_trace"}


def _valid_rows(data: Mapping[str, Any]) -> dict[str, bool]:
    out = {k: True for k in ("supporter", "seekers", "requests", "packets", "channels", "mission", "types")}
    scenarios = data.get("scenarios", []) if isinstance(data, Mapping) else []
    if not isinstance(scenarios, list):
        return {k: False for k in out}
    for row in scenarios:
        if not isinstance(row, Mapping):
            for k in out: out[k] = False
            continue
        supporter = _as_mapping(row.get("supporter_trace"))
        if set(supporter) != {"states", "actions"} or len(_rows(supporter.get("states"))) != HORIZON + 1 or len(_rows(supporter.get("actions"))) != HORIZON:
            out["supporter"] = False
        seekers = row.get("seeker_traces")
        if not isinstance(seekers, list) or len(seekers) != UGV_COUNT:
            out["seekers"] = False
        else:
            for item in seekers:
                if not isinstance(item, Mapping) or set(item) != {"robot", "states", "actions"} or len(_rows(item.get("states"))) != HORIZON + 1 or len(_rows(item.get("actions"))) != HORIZON:
                    out["seekers"] = False
        if len(_rows(row.get("request_opportunities"))) != 36: out["requests"] = False
        if not isinstance(row.get("map_packets"), list): out["packets"] = False
        if len(_rows(row.get("channel_slots"))) != HORIZON: out["channels"] = False
        if len(_rows(row.get("mission_trace"))) != HORIZON + 1: out["mission"] = False
        for field in ("supporter_trace", "seeker_traces", "request_opportunities", "map_packets", "channel_slots", "mission_trace"):
            if field not in row: out["types"] = False
    return out


def _index_records(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return result
    for row in data["scenarios"]:
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            result[row["scenario"]] = row
    return result


def _coverage_sequence(
    rows: Any,
    key_fn: Any,
    expected: Sequence[Any] | None,
    label: str,
    defects: list[str],
) -> None:
    """Record one missing/duplicate/unknown defect per canonical key.

    Coverage is deliberately independent of field validity and replay.  A row with
    an unusable key is an unknown canonical record, while a repeated usable key is
    one duplicate; the fixed coverage formula below then applies the specification's
    half-point deduction to each defect.
    """
    if not isinstance(rows, list):
        defects.append(f"{label}: rows is not a list")
        return
    seen: list[Any] = []
    actual_keys: list[Any] = []
    expected_set = set(expected) if expected is not None else None
    for index, row in enumerate(rows):
        try:
            key = key_fn(row)
        except Exception:
            key = None
        if key is None or (expected_set is not None and key not in expected_set):
            defects.append(f"{label}: unknown key at row[{index}]={key!r}")
            continue
        actual_keys.append(key)
        if key in seen:
            defects.append(f"{label}: duplicate key={key!r}")
        else:
            seen.append(key)
    if expected is not None:
        for key in expected:
            if key not in seen:
                defects.append(f"{label}: missing key={key!r}")


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    """Check every fixed canonical sequence and apply the mandated score formula."""
    defects: list[str] = []
    scenarios = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios is not a list"]

    _coverage_sequence(
        scenarios,
        lambda row: row.get("scenario") if isinstance(row, Mapping) else None,
        SCENARIO_NAMES,
        "scenario",
        defects,
    )
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in scenarios:
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            by_name.setdefault(str(row["scenario"]), row)

    state_keys = tuple(range(HORIZON + 1))
    action_keys = tuple(range(HORIZON))
    opportunity_keys = tuple((robot, t) for t in OPPORTUNITIES for robot in UGVS)
    for name in SCENARIO_NAMES:
        row = by_name.get(name)
        if row is None:
            # The missing scenario defect above is sufficient; do not manufacture
            # thousands of secondary missing-unit defects for an absent container.
            continue
        supporter = row.get("supporter_trace") if isinstance(row, Mapping) else None
        supporter = supporter if isinstance(supporter, Mapping) else {}
        _coverage_sequence(
            supporter.get("states"),
            lambda item: item.get("state_t") if isinstance(item, Mapping) else None,
            state_keys,
            f"{name}.supporter.states",
            defects,
        )
        _coverage_sequence(
            supporter.get("actions"),
            lambda item: item.get("t") if isinstance(item, Mapping) else None,
            action_keys,
            f"{name}.supporter.actions",
            defects,
        )

        seekers = row.get("seeker_traces") if isinstance(row, Mapping) else None
        _coverage_sequence(
            seekers,
            lambda item: item.get("robot") if isinstance(item, Mapping) else None,
            UGVS,
            f"{name}.seeker_traces",
            defects,
        )
        seeker_by_robot: dict[int, Mapping[str, Any]] = {}
        if isinstance(seekers, list):
            for item in seekers:
                if isinstance(item, Mapping) and _integer(item.get("robot")):
                    seeker_by_robot.setdefault(int(item["robot"]), item)
        for robot in UGVS:
            seeker = seeker_by_robot.get(robot)
            if seeker is None:
                continue
            _coverage_sequence(
                seeker.get("states"),
                lambda item: item.get("state_t") if isinstance(item, Mapping) else None,
                state_keys,
                f"{name}.seeker[{robot}].states",
                defects,
            )
            _coverage_sequence(
                seeker.get("actions"),
                lambda item: item.get("t") if isinstance(item, Mapping) else None,
                action_keys,
                f"{name}.seeker[{robot}].actions",
                defects,
            )

        _coverage_sequence(
            row.get("request_opportunities"),
            lambda item: (
                (int(item["robot"]), int(item["opportunity_t"]))
                if isinstance(item, Mapping)
                and _integer(item.get("robot"))
                and _integer(item.get("opportunity_t"))
                else None
            ),
            opportunity_keys,
            f"{name}.request_opportunities",
            defects,
        )
        _coverage_sequence(
            row.get("channel_slots"),
            lambda item: item.get("t") if isinstance(item, Mapping) else None,
            action_keys,
            f"{name}.channel_slots",
            defects,
        )
        _coverage_sequence(
            row.get("mission_trace"),
            lambda item: item.get("state_t") if isinstance(item, Mapping) else None,
            state_keys,
            f"{name}.mission_trace",
            defects,
        )

        packets = row.get("map_packets")
        if not isinstance(packets, list):
            defects.append(f"{name}.map_packets: rows is not a list")
        else:
            packet_keys: list[tuple[int, int]] = []
            for index, item in enumerate(packets):
                if not isinstance(item, Mapping) or not _integer(item.get("packet_id")) or not _integer(item.get("attempt")):
                    defects.append(f"{name}.map_packets: unknown key at row[{index}]")
                    continue
                key = (int(item["packet_id"]), int(item["attempt"]))
                if key in packet_keys:
                    defects.append(f"{name}.map_packets: duplicate key={key!r}")
                else:
                    packet_keys.append(key)

    # The construction/evaluation specifications define this exact normalization;
    # do not substitute a denominator based on the number of units.
    score = max(0.0, 4.0 - 0.5 * len(defects)) / 4.0
    return score, defects


def _evidence_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    groups = ("supporter", "seekers", "requests", "packets", "channels", "mission", "types")
    flags = {k: True for k in groups}
    details: dict[str, list[str]] = {k: [] for k in groups}

    def bad(group: str, path: str, reason: str, *, type_error: bool = True) -> None:
        flags[group] = False
        if len(details[group]) < 8:
            details[group].append(f"{path}: {reason}")
        if type_error and group != "types":
            flags["types"] = False
            if len(details["types"]) < 8:
                details["types"].append(f"{path}: {reason}")

    def mapping(value: Any, keys: frozenset[str], group: str, path: str) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            bad(group, path, "expected object")
            return None
        if set(value) != set(keys):
            bad(group, path, f"keys expected={sorted(keys)} actual={sorted(value)}")
        return value

    def rows(value: Any, group: str, path: str) -> list[Any]:
        if not isinstance(value, list):
            bad(group, path, "expected list")
            return []
        return value

    def integer(value: Any, group: str, path: str, low: int | None = None, high: int | None = None) -> None:
        if not _integer(value) or (low is not None and int(value) < low) or (high is not None and int(value) > high):
            bound = "" if low is None and high is None else f" in [{low},{high}]"
            bad(group, path, f"expected integer{bound}, actual={value!r}")

    def finite(value: Any, group: str, path: str, low: float | None = None, high: float | None = None) -> None:
        if not _finite(value) or (low is not None and float(value) < low - EPS) or (high is not None and float(value) > high + EPS):
            bad(group, path, f"expected finite number in [{low},{high}], actual={value!r}")

    def bool_value(value: Any, group: str, path: str) -> None:
        if not isinstance(value, bool):
            bad(group, path, f"expected boolean, actual={value!r}")

    def cell_list(value: Any, group: str, path: str) -> None:
        if not isinstance(value, list):
            bad(group, path, "expected cell-id list")
            return
        for i, cell in enumerate(value):
            integer(cell, group, f"{path}[{i}]", 0, GRID * GRID - 1)

    def packet_enqueue(value: Any, path: str) -> None:
        item = mapping(value, PACKET_ENQUEUE_KEYS, "supporter", path)
        if item is None:
            return
        integer(item.get("receiver"), "supporter", f"{path}.receiver", 0, UGV_COUNT - 1)
        integer(item.get("packet_id"), "supporter", f"{path}.packet_id", 0)
        integer(item.get("attempt"), "supporter", f"{path}.attempt", 0)
        integer(item.get("cell_id"), "supporter", f"{path}.cell_id", 0, GRID * GRID - 1)
        if item.get("cell_state") not in ("free", "obstacle"):
            bad("supporter", f"{path}.cell_state", "unknown cell state")
        integer(item.get("cell_cost"), "supporter", f"{path}.cell_cost", 0)
        integer(item.get("cell_version"), "supporter", f"{path}.cell_version", 1)
        retry = item.get("retry_of")
        if retry is not None:
            integer(retry, "supporter", f"{path}.retry_of", 0)

    for name in SCENARIO_NAMES:
        row = scenarios.get(name) if isinstance(scenarios, Mapping) else None
        if not isinstance(row, Mapping):
            for group in groups:
                bad(group, name, "scenario record missing", type_error=False)
            continue

        supporter = mapping(row.get("supporter_trace"), frozenset(("states", "actions")), "supporter", f"{name}.supporter_trace")
        if supporter is not None:
            sup_states = rows(supporter.get("states"), "supporter", f"{name}.supporter.states")
            sup_actions = rows(supporter.get("actions"), "supporter", f"{name}.supporter.actions")
            for i, state_value in enumerate(sup_states):
                path = f"{name}.supporter.states[{i}]"
                state = mapping(state_value, frozenset(("state_t", "x", "y", "observed_cell_ids", "map_versions", "map_queue_sizes", "remaining_map_attempts", "safe")), "supporter", path)
                if state is None:
                    continue
                integer(state.get("state_t"), "supporter", f"{path}.state_t", 0, HORIZON)
                integer(state.get("x"), "supporter", f"{path}.x", 0, GRID - 1)
                integer(state.get("y"), "supporter", f"{path}.y", 0, GRID - 1)
                cell_list(state.get("observed_cell_ids"), "supporter", f"{path}.observed_cell_ids")
                versions = rows(state.get("map_versions"), "supporter", f"{path}.map_versions")
                for j, value in enumerate(versions):
                    item = mapping(value, frozenset(("cell_id", "version")), "supporter", f"{path}.map_versions[{j}]")
                    if item is not None:
                        integer(item.get("cell_id"), "supporter", f"{path}.map_versions[{j}].cell_id", 0, GRID * GRID - 1)
                        integer(item.get("version"), "supporter", f"{path}.map_versions[{j}].version", 1)
                queues = rows(state.get("map_queue_sizes"), "supporter", f"{path}.map_queue_sizes")
                for j, value in enumerate(queues):
                    item = mapping(value, frozenset(("robot", "size")), "supporter", f"{path}.map_queue_sizes[{j}]")
                    if item is not None:
                        integer(item.get("robot"), "supporter", f"{path}.map_queue_sizes[{j}].robot", 0, UGV_COUNT - 1)
                        integer(item.get("size"), "supporter", f"{path}.map_queue_sizes[{j}].size", 0, 6)
                integer(state.get("remaining_map_attempts"), "supporter", f"{path}.remaining_map_attempts", 0, MAP_BUDGET)
                bool_value(state.get("safe"), "supporter", f"{path}.safe")
            for i, action_value in enumerate(sup_actions):
                path = f"{name}.supporter.actions[{i}]"
                action = mapping(action_value, frozenset(("t", "move_dx", "move_dy", "enqueue_packets")), "supporter", path)
                if action is None:
                    continue
                integer(action.get("t"), "supporter", f"{path}.t", 0, HORIZON - 1)
                integer(action.get("move_dx"), "supporter", f"{path}.move_dx")
                integer(action.get("move_dy"), "supporter", f"{path}.move_dy")
                packets = rows(action.get("enqueue_packets"), "supporter", f"{path}.enqueue_packets")
                for j, packet in enumerate(packets):
                    packet_enqueue(packet, f"{path}.enqueue_packets[{j}]")

        seekers = rows(row.get("seeker_traces"), "seekers", f"{name}.seeker_traces")
        for i, seeker_value in enumerate(seekers):
            path = f"{name}.seeker_traces[{i}]"
            seeker = mapping(seeker_value, frozenset(("robot", "states", "actions")), "seekers", path)
            if seeker is None:
                continue
            integer(seeker.get("robot"), "seekers", f"{path}.robot", 0, UGV_COUNT - 1)
            state_rows = rows(seeker.get("states"), "seekers", f"{path}.states")
            action_rows = rows(seeker.get("actions"), "seekers", f"{path}.actions")
            state_keys = frozenset(("state_t", "x", "y", "observed_cells", "belief_path_cell_ids", "belief_path_unknown", "belief_path_cost", "request_queue", "request_versions", "map_versions", "ages", "goal_reached", "safe", "remaining_request_budget"))
            for j, state_value in enumerate(state_rows):
                spath = f"{path}.states[{j}]"
                state = mapping(state_value, state_keys, "seekers", spath)
                if state is None:
                    continue
                integer(state.get("state_t"), "seekers", f"{spath}.state_t", 0, HORIZON)
                integer(state.get("x"), "seekers", f"{spath}.x", 0, GRID - 1)
                integer(state.get("y"), "seekers", f"{spath}.y", 0, GRID - 1)
                observed = rows(state.get("observed_cells"), "seekers", f"{spath}.observed_cells")
                for k, value in enumerate(observed):
                    opath = f"{spath}.observed_cells[{k}]"
                    item = mapping(value, frozenset(("cell_id", "state", "cost", "version")), "seekers", opath)
                    if item is not None:
                        integer(item.get("cell_id"), "seekers", f"{opath}.cell_id", 0, GRID * GRID - 1)
                        if item.get("state") not in ("free", "obstacle"):
                            bad("seekers", f"{opath}.state", "unknown cell state")
                        integer(item.get("cost"), "seekers", f"{opath}.cost", 0)
                        integer(item.get("version"), "seekers", f"{opath}.version", 0)
                path_cells = state.get("belief_path_cell_ids")
                cell_list(path_cells, "seekers", f"{spath}.belief_path_cell_ids")
                plen = len(path_cells) if isinstance(path_cells, list) else -1
                unknown = state.get("belief_path_unknown")
                if not isinstance(unknown, bool) and not (isinstance(unknown, list) and len(unknown) == plen and all(isinstance(v, bool) for v in unknown)):
                    bad("seekers", f"{spath}.belief_path_unknown", "invalid compact or expanded encoding")
                costs = state.get("belief_path_cost")
                if _finite(costs):
                    if float(costs) < -EPS:
                        bad("seekers", f"{spath}.belief_path_cost", "negative cost")
                elif not (isinstance(costs, list) and len(costs) == plen and all(_finite(v) and float(v) >= -EPS for v in costs)):
                    bad("seekers", f"{spath}.belief_path_cost", "invalid compact or expanded encoding")
                queue = state.get("request_queue")
                if not isinstance(queue, list):
                    bad("seekers", f"{spath}.request_queue", "expected list")
                else:
                    for k, value in enumerate(queue):
                        integer(value, "seekers", f"{spath}.request_queue[{k}]", 0)
                integer(state.get("request_versions"), "seekers", f"{spath}.request_versions", 0)
                versions = rows(state.get("map_versions"), "seekers", f"{spath}.map_versions")
                for k, value in enumerate(versions):
                    item = mapping(value, frozenset(("cell_id", "version")), "seekers", f"{spath}.map_versions[{k}]")
                    if item is not None:
                        integer(item.get("cell_id"), "seekers", f"{spath}.map_versions[{k}].cell_id", 0, GRID * GRID - 1)
                        integer(item.get("version"), "seekers", f"{spath}.map_versions[{k}].version", 1)
                ages = rows(state.get("ages"), "seekers", f"{spath}.ages")
                for k, value in enumerate(ages):
                    item = mapping(value, frozenset(("cell_id", "age")), "seekers", f"{spath}.ages[{k}]")
                    if item is not None:
                        integer(item.get("cell_id"), "seekers", f"{spath}.ages[{k}].cell_id", 0, GRID * GRID - 1)
                        integer(item.get("age"), "seekers", f"{spath}.ages[{k}].age", 0, AGE_MAX)
                bool_value(state.get("goal_reached"), "seekers", f"{spath}.goal_reached")
                bool_value(state.get("safe"), "seekers", f"{spath}.safe")
                integer(state.get("remaining_request_budget"), "seekers", f"{spath}.remaining_request_budget", 0, REQUEST_BUDGET)
            action_keys = frozenset(("t", "move_dx", "move_dy", "request"))
            for j, action_value in enumerate(action_rows):
                apath = f"{path}.actions[{j}]"
                action = mapping(action_value, action_keys, "seekers", apath)
                if action is None:
                    continue
                integer(action.get("t"), "seekers", f"{apath}.t", 0, HORIZON - 1)
                integer(action.get("move_dx"), "seekers", f"{apath}.move_dx")
                integer(action.get("move_dy"), "seekers", f"{apath}.move_dy")
                request = action.get("request")
                if request is not None:
                    req = mapping(request, REQUEST_KEYS, "requests", f"{apath}.request")
                    if req is not None:
                        integer(req.get("request_id"), "requests", f"{apath}.request.request_id", 0)
                        integer(req.get("request_version"), "requests", f"{apath}.request.request_version", 1)
                        cell_list(req.get("path_cell_ids"), "requests", f"{apath}.request.path_cell_ids")
                        finite(req.get("uncertainty"), "requests", f"{apath}.request.uncertainty", 0.0, 1.0)

        request_rows = rows(row.get("request_opportunities"), "requests", f"{name}.request_opportunities")
        request_keys = REQUEST_OPPORTUNITY_KEYS
        for i, value in enumerate(request_rows):
            path = f"{name}.request_opportunities[{i}]"
            req = mapping(value, request_keys, "requests", path)
            if req is None:
                continue
            integer(req.get("robot"), "requests", f"{path}.robot", 0, UGV_COUNT - 1)
            integer(req.get("opportunity_t"), "requests", f"{path}.opportunity_t", 0, HORIZON - 1)
            bool_value(req.get("active"), "requests", f"{path}.active")
            for field in ("request_id", "request_version", "request_packet_id", "send_slot", "arrival_state"):
                value = req.get(field)
                if value is not None:
                    integer(value, "requests", f"{path}.{field}", 0, HORIZON + 1)
            if req.get("path_cell_ids") is not None:
                cell_list(req.get("path_cell_ids"), "requests", f"{path}.path_cell_ids")
            if req.get("uncertainty") is not None:
                finite(req.get("uncertainty"), "requests", f"{path}.uncertainty", 0.0, 1.0)
            generated = req.get("generated")
            bool_value(generated, "requests", f"{path}.generated")
            for field in ("queue_before", "queue_after"):
                value = req.get(field)
                # The public contract permits null queue snapshots when no
                # request was generated; supplied snapshots remain typed.
                if value is None and generated is False:
                    continue
                integer(value, "requests", f"{path}.{field}", 0, 2)
            if req.get("status") not in ("none", "queued", "queue_full", "sent", "delivered", "lost", "expired"):
                bad("requests", f"{path}.status", "unknown request status")

        packet_rows = rows(row.get("map_packets"), "packets", f"{name}.map_packets")
        for i, value in enumerate(packet_rows):
            path = f"{name}.map_packets[{i}]"
            packet = mapping(value, PACKET_EVENT_KEYS, "packets", path)
            if packet is None:
                continue
            integer(packet.get("packet_id"), "packets", f"{path}.packet_id", 0)
            integer(packet.get("attempt"), "packets", f"{path}.attempt", 0)
            integer(packet.get("sender"), "packets", f"{path}.sender", 0, UAV)
            integer(packet.get("receiver"), "packets", f"{path}.receiver", 0, UGV_COUNT - 1)
            integer(packet.get("cell_id"), "packets", f"{path}.cell_id", 0, GRID * GRID - 1)
            if packet.get("cell_state") not in ("free", "obstacle"):
                bad("packets", f"{path}.cell_state", "unknown cell state")
            integer(packet.get("cell_cost"), "packets", f"{path}.cell_cost", 0)
            integer(packet.get("cell_version"), "packets", f"{path}.cell_version", 1)
            integer(packet.get("enqueue_slot"), "packets", f"{path}.enqueue_slot", 0, HORIZON - 1)
            for field in ("send_slot", "arrival_state"):
                if packet.get(field) is not None:
                    integer(packet.get(field), "packets", f"{path}.{field}", 0, HORIZON + 1)
            if packet.get("retry_of") is not None:
                integer(packet.get("retry_of"), "packets", f"{path}.retry_of", 0)
            receiver_distance = packet.get("receiver_distance")
            interference = packet.get("interference")
            concurrency = packet.get("concurrency")
            send_slot = packet.get("send_slot")
            if send_slot is None:
                if packet.get("arrival_state") is not None:
                    bad("packets", f"{path}.arrival_state", "must be null before admission")
                for field, value in (("receiver_distance", receiver_distance), ("interference", interference), ("concurrency", concurrency)):
                    if value is not None:
                        bad("packets", f"{path}.{field}", "must be null before admission")
            else:
                finite(receiver_distance, "packets", f"{path}.receiver_distance", 0.0)
                finite(interference, "packets", f"{path}.interference", 0.0, 1.0)
                integer(concurrency, "packets", f"{path}.concurrency", 0, CHANNEL_CAPACITY)
            bool_value(packet.get("delivered"), "packets", f"{path}.delivered")
            if packet.get("fusion_status") not in ("pending", "out_of_range", "interference_lost", "fused", "duplicate", "stale"):
                bad("packets", f"{path}.fusion_status", "unknown fusion status")

        channel_rows = rows(row.get("channel_slots"), "channels", f"{name}.channel_slots")
        channel_keys = frozenset(("t", "interference", "request_queue_sizes", "map_queue_sizes", "submitted_keys", "admitted_keys", "units", "capacity_remaining"))
        for i, value in enumerate(channel_rows):
            path = f"{name}.channel_slots[{i}]"
            channel = mapping(value, channel_keys, "channels", path)
            if channel is None:
                continue
            integer(channel.get("t"), "channels", f"{path}.t", 0, HORIZON - 1)
            finite(channel.get("interference"), "channels", f"{path}.interference", 0.0, 1.0)
            for field in ("request_queue_sizes", "map_queue_sizes"):
                if _queue_sizes_encoding(channel.get(field)) is None:
                    bad("channels", f"{path}.{field}", "invalid compact or explicit queue-size encoding")
            for field in ("submitted_keys", "admitted_keys"):
                keys = rows(channel.get(field), "channels", f"{path}.{field}")
                for j, key in enumerate(keys):
                    kpath = f"{path}.{field}[{j}]"
                    if not isinstance(key, Mapping) or key.get("kind") not in ("request", "map"):
                        bad("channels", kpath, "invalid channel key")
                        continue
                    kind = key.get("kind")
                    expected = {"kind", "robot", "opportunity_t"} if kind == "request" else None
                    if kind == "request" and set(key) != expected:
                        bad("channels", kpath, "request key fields")
                    if kind == "request":
                        integer(key.get("robot"), "channels", f"{kpath}.robot", 0, UGV_COUNT - 1)
                        integer(key.get("opportunity_t"), "channels", f"{kpath}.opportunity_t", 0, HORIZON - 1)
                    else:
                        if set(key) not in ({"kind", "packet_id", "attempt"}, {"kind", "packet_id", "attempt", "receiver"}):
                            bad("channels", kpath, "map key fields")
                        integer(key.get("packet_id"), "channels", f"{kpath}.packet_id", 0)
                        integer(key.get("attempt"), "channels", f"{kpath}.attempt", 0)
                        if "receiver" in key:
                            integer(key.get("receiver"), "channels", f"{kpath}.receiver", 0, UGV_COUNT - 1)
            integer(channel.get("units"), "channels", f"{path}.units", 0, CHANNEL_CAPACITY)
            integer(channel.get("capacity_remaining"), "channels", f"{path}.capacity_remaining", 0, CHANNEL_CAPACITY)

        mission_rows = rows(row.get("mission_trace"), "mission", f"{name}.mission_trace")
        mission_keys = frozenset(("state_t", "goal_reached", "arrival_states", "ground_move_cost", "uav_move_cost", "blocked_moves", "separation_interventions", "redundant_detours", "remaining_map_attempts", "safety_violation", "terminal"))
        for i, value in enumerate(mission_rows):
            path = f"{name}.mission_trace[{i}]"
            mission = mapping(value, mission_keys, "mission", path)
            if mission is None:
                continue
            integer(mission.get("state_t"), "mission", f"{path}.state_t", 0, HORIZON)
            goals = mission.get("goal_reached")
            if not isinstance(goals, list) or len(goals) != UGV_COUNT:
                bad("mission", f"{path}.goal_reached", "expected three booleans")
            else:
                for j, goal in enumerate(goals):
                    bool_value(goal, "mission", f"{path}.goal_reached[{j}]")
            arrivals = mission.get("arrival_states")
            if not isinstance(arrivals, list) or len(arrivals) != UGV_COUNT:
                bad("mission", f"{path}.arrival_states", "expected three nullable integers")
            else:
                for j, arrival in enumerate(arrivals):
                    if arrival is not None:
                        integer(arrival, "mission", f"{path}.arrival_states[{j}]", 0, HORIZON + 1)
            for field in ("ground_move_cost", "uav_move_cost"):
                finite(mission.get(field), "mission", f"{path}.{field}", 0.0)
            for field in ("blocked_moves", "separation_interventions", "redundant_detours"):
                integer(mission.get(field), "mission", f"{path}.{field}", 0)
            integer(mission.get("remaining_map_attempts"), "mission", f"{path}.remaining_map_attempts", 0, MAP_BUDGET)
            bool_value(mission.get("safety_violation"), "mission", f"{path}.safety_violation")
            bool_value(mission.get("terminal"), "mission", f"{path}.terminal")

    return [
        _atom("result_evidence", group, flags[group], "; ".join(details[group]) or f"{group} evidence valid")
        for group in groups
    ]


def _legality_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    flags = {k: True for k in ("moves", "paths", "queues", "packets", "channel", "budgets", "information", "terminal")}
    details: dict[str, list[str]] = {k: [] for k in flags}
    for name in SCENARIO_NAMES:
        if name not in scenarios:
            for k in flags: flags[k] = False
            continue
        result = _replay_scenario(name, scenarios[name])
        for k, values in result["illegal"].items():
            if values:
                flags[k] = False
                details[k].extend(values[:3])
    return [_atom("decision_legality", k, flags[k], "; ".join(details[k]) or "legal") for k in flags]


def _replay_atoms(scenarios: Mapping[str, Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    names = ("supporter", "seekers", "requests", "packets", "channels", "mission", "metrics")
    flags = {k: True for k in names}
    details: dict[str, list[str]] = {k: [] for k in names}
    replayed: dict[str, Any] = {}
    for name in SCENARIO_NAMES:
        row = scenarios.get(name)
        if not isinstance(row, Mapping):
            for k in flags: flags[k] = False
            continue
        result = _replay_scenario(name, row)
        replayed[name] = result
        for defect in result["defects"]:
            if ".supporter." in defect: bucket = "supporter"
            elif ".seeker[" in defect: bucket = "seekers"
            elif ".request_opportunities" in defect: bucket = "requests"
            elif ".map_packets" in defect: bucket = "packets"
            elif ".channel[" in defect: bucket = "channels"
            elif ".mission[" in defect: bucket = "mission"
            else: bucket = "metrics"
            flags[bucket] = False
            if len(details[bucket]) < 20:
                details[bucket].append(defect)
        if result["defects"]:
            flags["metrics"] = False
    return [_atom("replay_validity", k, flags[k], "; ".join(details[k]) or "replayed") for k in names], replayed


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in expected.values()):
        raise ValueError("declared atom shares must be finite and positive")
    if not math.isclose(sum(float(value) for value in expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    for atom in atoms:
        if not math.isfinite(float(atom.share)) or not math.isclose(float(atom.share), float(expected[atom.id]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"atom share mismatch: {atom.id}")
    return max(0.0, min(1.0, sum(float(atom.share) for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [x for x in atoms if not x.passed]
    return "failed_atoms=[" + ",".join(x.id for x in failed) + "]; " + " | ".join(f"{x.id}: {x.detail}" for x in failed) if failed else "failed_atoms=[]"


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular non-symlink file"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md could not be read as UTF-8: {exc}"
    if not text.strip():
        return 0.0, "report.md is empty"
    required = ("Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations")
    found = []
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.MULTILINE))
    for i, heading in enumerate(headings):
        title = heading.group(2)
        if title not in required:
            continue
        level = len(heading.group(1)); end = len(text)
        for later in headings[i + 1:]:
            if len(later.group(1)) <= level:
                end = later.start(); break
        if text[heading.end():end].strip():
            found.append(title)
    missing = [x for x in required if x not in found]
    return max(0.0, 1.0 - len(missing) / 4.0), f"missing_sections={missing}"


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    scenarios = root and isinstance(data.get("scenarios"), list)
    rows = bool(scenarios and all(isinstance(x, Mapping) and isinstance(x.get("scenario"), str) for x in data["scenarios"]))
    shape = bool(scenarios and all(_scenario_shape(x) for x in data["scenarios"]))
    return [_atom("result_json", "parse", parsed, "strict JSON parse"), _atom("result_json", "root", root and set(data) == {"scenarios"}, "root keys"), _atom("result_json", "scenarios", bool(scenarios), "scenario list"), _atom("result_json", "scenario_keys", rows and shape, "scenario record keys")]


def _validity_gate(scores: Mapping[str, float], official: bool) -> bool:
    return official and all(scores.get(x, 0.0) >= 1 - EPS for x in GATE_CHECKS)


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    try:
        removed = (await ws.run("rm -f -- results.json", check=False, timeout=30.0)).exit_code == 0
        state.run_started = True
        started = time.monotonic()
        result = await ws.run("python run.py", check=False, timeout=600.0)
        state.exit_code = getattr(result, "exit_code", None)
        state.timed_out = state.exit_code is None
        state.run_finished = True
        state.elapsed_s = time.monotonic() - started
        state.result_file_is_new_regular = removed and _regular_file(ws, "results.json")
        state.official_ok = bool(state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600 + EPS and state.result_file_is_new_regular)
        state.official_message = f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(Path(ws.root, "results.json").read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"]) if state.official_ok else 0.0
        state.diagnostics["result_json"] = _diagnose(atoms)
        if not state.official_ok or not state.parse_ok:
            for name in GATE_CHECKS[1:]: state.scores[name] = 0.0
            state.validity_gate = False
            _CACHE[key] = state
            return state
        state.records = _index_records(state.data)
        state.scenarios = state.records
        evidence = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:10]}"
        legality = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose(legality)
        replay, replayed = _replay_atoms(state.scenarios)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose(replay)
        state.validity_gate = _validity_gate(state.scores, state.official_ok)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.records)
    except Exception as exc:
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        state.official_message = state.official_message or detail
        for name in GATE_CHECKS: state.scores[name] = 0.0; state.diagnostics[name] = detail
        state.validity_gate = False
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (1.0, "static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "static compilation failed")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if name.startswith(("primary_metrics_", "auxiliary_metrics_")):
        if state.metrics is None:
            return 0.0, f"{name}: validity gate not satisfied"
        return state.metrics[name], f"{name}: independently replayed value={state.metrics[name]:.9g}"
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "")


for _id in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity", "primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2"):
    async def _check(ws: Workspace, _name: str = _id) -> tuple[float, str]:
        return await _score(ws, _name)
    globals()[_id] = check(id=_id)(_check)

del _check, _id
