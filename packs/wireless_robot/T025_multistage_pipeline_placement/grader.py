from __future__ import annotations

import asyncio
import heapq
import json
import math
import re
import time
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 48
WINDOW_LENGTH = 4
WINDOWS = 12
DT = 0.5
EPS = 1e-6
ROBOTS = (0, 1)
SCENES = ("balanced_pipeline", "robot_cpu_burst", "edge_queue_burst", "link_jitter", "deadline_race")
STAGES = ("Sensing", "Perception", "Planning", "Control")
MIGRATABLE = ("Perception", "Planning")
NODES = ("robot_0", "robot_1", "edge")
QUEUE_CAPACITY = 2
EDGE_RATE = 2.4
LOCAL_RATE = 1.2
PACKET_BITS = 256
CYCLE_BITS = 1024
MIGRATION_COST = 0.18
DEADLINE = 0.8
MAX_V = 1.2
MAX_OMEGA = 1.0
GOAL_RADIUS = 0.4
COMMUNICATION_WINDOWS = len(ROBOTS) * WINDOWS
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "placements", "commands"}
PLACEMENT_KEYS = {"robot", "window", "perception", "planning"}
COMMAND_KEYS = {"robot", "cycle", "command"}
ATOM_SHARES = {
    "result_json": {"result_json.parse": 0.34, "result_json.root": 0.33, "result_json.scene_containers": 0.33},
    "result_evidence": {"result_evidence.placements": 0.5, "result_evidence.commands": 0.5},
    "decision_legality": {
        "decision_legality.residence": 0.4,
        "decision_legality.command_update": 0.6,
    },
    "replay_validity": {
        "replay_validity.dag": 0.16666666666666666,
        "replay_validity.communication": 0.16666666666666666,
        "replay_validity.resources": 0.16666666666666666,
        "replay_validity.motion": 0.16666666666666666,
        "replay_validity.deadline": 0.16666666666666666,
        "replay_validity.safety": 0.16666666666666666,
    },
}
VALIDITY_CHECKS = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
METRIC_NAMES = ("primary_metrics_1", "primary_metrics_2", "auxiliary_metrics_1", "auxiliary_metrics_2")
BASE_WORK = {"Sensing": 0.10, "Perception": 0.14, "Planning": 0.11, "Control": 0.09}
SCENE_DATA = {
    "balanced_pipeline": {"starts": ((1.0, 1.0, 0.0), (1.0, 9.0, 0.0)), "goals": ((18.0, 10.0), (18.0, 2.0)), "deadlines": (42, 42), "obstacles": ((8.0, 0.0, 10.0, 4.0), (8.0, 8.0, 10.0, 12.0))},
    "robot_cpu_burst": {"starts": ((1.0, 2.0, 0.0), (1.0, 10.0, 0.0)), "goals": ((18.0, 10.0), (18.0, 2.0)), "deadlines": (44, 44), "obstacles": ((9.0, 0.0, 11.0, 3.0), (9.0, 9.0, 11.0, 12.0))},
    "edge_queue_burst": {"starts": ((1.0, 1.0, 0.0), (1.0, 9.0, 0.0)), "goals": ((18.0, 11.0), (18.0, 1.0)), "deadlines": (43, 43), "obstacles": ((7.0, 4.0, 13.0, 6.0),)},
    "link_jitter": {"starts": ((1.0, 2.0, 0.0), (1.0, 10.0, 0.0)), "goals": ((18.0, 9.0), (18.0, 3.0)), "deadlines": (45, 45), "obstacles": ((6.0, 0.0, 8.0, 5.0), (6.0, 7.0, 8.0, 12.0))},
    "deadline_race": {"starts": ((1.0, 1.0, 0.0), (1.0, 9.0, 0.0)), "goals": ((18.0, 10.0), (18.0, 2.0)), "deadlines": (38, 38), "obstacles": ((8.0, 0.0, 10.0, 4.0), (8.0, 8.0, 10.0, 12.0))},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, bool(passed), detail, share


class EvalState:
    def __init__(self) -> None:
        self.run_started = self.run_finished = self.official_ok = False
        self.exit_code: int | None = None
        self.timed_out = False
        self.elapsed_s: float | None = None
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


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _number(value: Any) -> bool:
    return _finite_float(value) is not None


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate object member {key!r}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[group][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0.0, abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share differs")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    return "failed_atoms=[]" if not failed else "failed_atoms=[{}]; {}".format(
        ",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed)
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _link(scene: str, cycle: int) -> tuple[str, float]:
    if scene == "balanced_pipeline": state = "good"
    elif scene == "robot_cpu_burst": state = "good" if cycle % 6 < 4 else "jitter"
    elif scene == "edge_queue_burst": state = "good" if cycle < 12 or cycle % 8 < 3 else "jitter"
    elif scene == "link_jitter": state = "jitter" if cycle % 10 in (3, 4, 5) else "good"
    else: state = "jitter" if 10 <= cycle <= 21 else "good"
    return state, 0.05 if state == "good" else 0.20


def _work(scene: str, robot: int, cycle: int, stage: str) -> float:
    q = SCENES.index(scene)
    value = BASE_WORK[stage] + 0.02 * ((q + 2 * robot + cycle) % 3)
    if scene == "robot_cpu_burst" and robot == 0 and 12 <= cycle <= 23:
        value += 0.36 if stage == "Sensing" else 0.18 if stage == "Control" else 0.0
    if scene == "edge_queue_burst" and stage in MIGRATABLE and 8 <= cycle <= 19: value += 0.32
    if scene == "deadline_race" and 8 <= cycle <= 19 and stage in MIGRATABLE: value += 0.14
    return value


def _node(robot: int, stage: str, placement: Mapping[str, str]) -> str:
    if stage in ("Sensing", "Control"): return f"robot_{robot}"
    return "edge" if placement[stage.lower()] == "edge" else f"robot_{robot}"


def _transfer_priority(ready: float, robot: int, cycle: int, stage_index: int) -> tuple[float, int, int, int]:
    return ready, robot, cycle, stage_index


def _reserve_transfer_slot(transfer_slots: dict[int, int], predecessor_finish: float) -> int:
    slot = max(0, int(math.ceil((predecessor_finish - EPS) / DT)))
    while transfer_slots.get(slot, 0) + PACKET_BITS > CYCLE_BITS:
        slot += 1
    transfer_slots[slot] = transfer_slots.get(slot, 0) + PACKET_BITS
    return slot


def _distance_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> float:
    x, y = point
    return math.hypot(max(rect[0] - x, 0.0, x - rect[2]), max(rect[1] - y, 0.0, y - rect[3]))


def _obstacle_distance(scene: str, point: tuple[float, float]) -> float:
    x, y = point
    boundary = min(x, 20.0 - x, y, 12.0 - y)
    return min([boundary, *(_distance_rect(point, rect) for rect in SCENE_DATA[scene]["obstacles"])])


def _simulate_scene(scene: str, placements: Mapping[tuple[int, int], Mapping[str, Any]], commands: Mapping[tuple[int, int], Mapping[str, Any]]) -> dict[str, Any]:
    placement_map = {(r, w): {"perception": placements[(r, w)]["perception"], "planning": placements[(r, w)]["planning"]} for r in ROBOTS for w in range(WINDOWS)}
    node_available = {node: 0.0 for node in NODES}
    queue_seq = {node: 0 for node in NODES}
    intervals: dict[str, list[tuple[float, float, tuple[int, int, str], float, float]]] = {node: [] for node in NODES}
    stage_info: dict[tuple[int, int, str], dict[str, Any]] = {}
    transfer_slots: dict[int, int] = {}
    transfers: list[dict[str, Any]] = []
    transfer_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    # Event kind 0 is a transfer request and kind 1 is a service-ready stage.
    # Their independent resources may be processed in either cross-kind order,
    # while the keys within each kind implement the two public arbitration rules.
    ready_heap: list[tuple[float, int, int, int, int, int, int, int, str | None]] = []

    def push_service(ready: float, stage_index: int, robot: int, cycle: int, predecessor: str | None) -> None:
        heapq.heappush(ready_heap, (ready, 1, stage_index, robot, cycle, stage_index, robot, cycle, predecessor))

    def push_transfer(ready: float, stage_index: int, robot: int, cycle: int, predecessor: str) -> None:
        priority = _transfer_priority(ready, robot, cycle, stage_index)
        heapq.heappush(ready_heap, (priority[0], 0, priority[1], priority[2], priority[3], stage_index, robot, cycle, predecessor))

    for robot in ROBOTS:
        for cycle in range(HORIZON):
            push_service(cycle * DT, 0, robot, cycle, None)
    while ready_heap:
        nominal_ready, event_kind, _, _, _, stage_index, robot, cycle, predecessor = heapq.heappop(ready_heap)
        stage = STAGES[stage_index]
        key = (robot, cycle, stage)
        placement = placement_map[(robot, cycle // WINDOW_LENGTH)]
        node = _node(robot, stage, placement)
        pred_info = stage_info.get((robot, cycle, predecessor)) if predecessor else None
        pred_finish = pred_info["end"] if pred_info and pred_info["status"] == "completed" else None
        communication_wait = 0.0
        ready = nominal_ready
        if predecessor and pred_finish is None:
            stage_info[key] = {"stage": stage, "node": node, "predecessor_finish": None, "start": None, "end": None, "service_time": 0.0, "communication_wait": 0.0, "queue_sequence": None, "status": "dropped", "ready": ready}
            continue
        if event_kind == 0:
            assert predecessor is not None and pred_finish is not None
            slot = _reserve_transfer_slot(transfer_slots, pred_finish)
            _, delay = _link(scene, slot)
            arrival = slot * DT + delay
            transfer = {"robot": robot, "cycle": cycle, "stage": stage, "predecessor_finish": pred_finish, "start_cycle": slot, "arrival": arrival, "wait_cycles": max(0, math.ceil((arrival - pred_finish - EPS) / DT)), "delivered": arrival <= HORIZON * DT + EPS}
            transfer_cache[key] = transfer
            transfers.append(transfer)
            push_service(arrival, stage_index, robot, cycle, predecessor)
            continue
        if predecessor and pred_info["node"] != node:
            transfer = transfer_cache[key]
            communication_wait = transfer["arrival"] - pred_finish
        waiting = sum(1 for start, end, _, _, item_ready in intervals[node] if item_ready <= ready + EPS and start > ready + EPS and end > ready + EPS)
        if waiting >= QUEUE_CAPACITY:
            status = "dropped"
            stage_info[key] = {"stage": stage, "node": node, "predecessor_finish": pred_finish, "start": None, "end": None, "service_time": 0.0, "communication_wait": communication_wait, "queue_sequence": None, "status": status, "ready": ready}
            continue
        changed = stage in MIGRATABLE and cycle % WINDOW_LENGTH == 0 and placement[stage.lower()] != ("local" if cycle == 0 else placement_map[(robot, cycle // WINDOW_LENGTH - 1)][stage.lower()])
        service_time = _work(scene, robot, cycle, stage) / (EDGE_RATE if node == "edge" else LOCAL_RATE) + (MIGRATION_COST if changed else 0.0)
        start = max(ready, node_available[node])
        end = start + service_time
        sequence = queue_seq[node]
        queue_seq[node] += 1
        node_available[node] = end
        intervals[node].append((start, end, key, service_time, ready))
        stage_info[key] = {"stage": stage, "node": node, "predecessor_finish": pred_finish, "start": start, "end": end, "service_time": service_time, "communication_wait": communication_wait, "queue_sequence": sequence, "status": "completed", "ready": ready}
        if stage_index + 1 < len(STAGES):
            successor_index = stage_index + 1
            successor_node = _node(robot, STAGES[successor_index], placement)
            if successor_node == node:
                push_service(end, successor_index, robot, cycle, stage)
            else:
                push_transfer(end, successor_index, robot, cycle, stage)
    for robot in ROBOTS:
        for cycle in range(HORIZON):
            placement = placement_map[(robot, cycle // WINDOW_LENGTH)]
            for stage_index, stage in enumerate(STAGES):
                key = (robot, cycle, stage)
                if key in stage_info: continue
                predecessor = STAGES[stage_index - 1] if stage_index else None
                pred_info = stage_info.get((robot, cycle, predecessor)) if predecessor else None
                stage_info[key] = {"stage": stage, "node": _node(robot, stage, placement), "predecessor_finish": pred_info["end"] if pred_info and pred_info["status"] == "completed" else None, "start": None, "end": None, "service_time": 0.0, "communication_wait": 0.0, "queue_sequence": None, "status": "dropped", "ready": cycle * DT}
    stages = []
    for robot in ROBOTS:
        for cycle in range(HORIZON):
            for stage in STAGES:
                info = stage_info[(robot, cycle, stage)]
                stages.append({"scenario": scene, "robot": robot, "cycle": cycle, "stage": stage, "node": info["node"], "predecessor_finish": info["predecessor_finish"], "start": info["start"], "end": info["end"], "service_time": info["service_time"], "communication_wait": info["communication_wait"], "queue_sequence": info["queue_sequence"], "status": info["status"]})
    state = [list(SCENE_DATA[scene]["starts"][r]) for r in ROBOTS]
    unsafe_history = {r: [] for r in ROBOTS}
    goal_first: dict[int, int | None] = {r: None for r in ROBOTS}
    robot_rows, safety_rows, cycle_rows = [], [], []
    control_end = {(r, c): stage_info[(r, c, "Control")]["end"] if stage_info[(r, c, "Control")]["status"] == "completed" else None for r in ROBOTS for c in range(HORIZON)}
    for cycle in range(HORIZON):
        previous_boundary = (cycle - 1) * DT
        updates = {
            r: max(
                (
                    c for c in range(cycle)
                    if control_end[(r, c)] is not None
                    and control_end[(r, c)] > previous_boundary + EPS
                    and control_end[(r, c)] <= cycle * DT + EPS
                ),
                default=None,
            )
            for r in ROBOTS
        }
        proposed = []
        for robot in ROBOTS:
            v, omega = commands[(robot, cycle)]["command"]
            x, y, theta = state[robot]
            proposed.append([x + DT * v * math.cos(theta), y + DT * v * math.sin(theta), _wrap(theta + DT * omega)])
        out_of_bounds = [p[0] < -EPS or p[0] > 20.0 + EPS or p[1] < -EPS or p[1] > 12.0 + EPS for p in proposed]
        obstacle_hit = [any(_distance_rect((p[0], p[1]), rect) <= EPS for rect in SCENE_DATA[scene]["obstacles"]) for p in proposed]
        pair_distance = math.hypot(proposed[0][0] - proposed[1][0], proposed[0][1] - proposed[1][1])
        collision = pair_distance < 0.5 - EPS
        unsafe = [out_of_bounds[r] or obstacle_hit[r] or collision for r in ROBOTS]
        for robot in ROBOTS:
            if not unsafe[robot]: state[robot] = proposed[robot]
            unsafe_history[robot].append(unsafe[robot])
            goal_distance = math.hypot(state[robot][0] - SCENE_DATA[scene]["goals"][robot][0], state[robot][1] - SCENE_DATA[scene]["goals"][robot][1])
            if goal_first[robot] is None and goal_distance <= GOAL_RADIUS + EPS: goal_first[robot] = cycle + 1
            robot_rows.append({"scenario": scene, "robot": robot, "t": cycle, "position": [state[robot][0], state[robot][1]], "goal_distance": goal_distance, "active_command": list(commands[(robot, cycle)]["command"]), "new_control_cycle_id": updates[robot], "unsafe": unsafe[robot]})
            cinfo = stage_info[(robot, cycle, "Control")]
            completion = cinfo["end"]
            latency = completion - cycle * DT if completion is not None else None
            cycle_rows.append({"scenario": scene, "robot": robot, "cycle": cycle, "window": cycle // WINDOW_LENGTH, "placement": dict(placement_map[(robot, cycle // WINDOW_LENGTH)]), "release_time": cycle * DT, "completion_time": completion, "latency": latency, "deadline": DEADLINE, "new_control_cycle_id": updates[robot], "deadline_miss": completion is None or latency > DEADLINE + EPS, "active_command": list(commands[(robot, cycle)]["command"])})
        safety_rows.append({"scenario": scene, "t": cycle, "obstacle_distance": [_obstacle_distance(scene, (proposed[r][0], proposed[r][1])) for r in ROBOTS], "pair_distance": pair_distance, "collision": collision, "out_of_bounds": out_of_bounds, "goal_deadline": [goal_first[r] is not None and goal_first[r] <= SCENE_DATA[scene]["deadlines"][r] for r in ROBOTS]})
    resources = []
    for cycle in range(HORIZON):
        now = cycle * DT
        for node in NODES:
            active = [item for item in intervals[node] if item[0] <= now + EPS < item[1]]
            waiting = [item for item in intervals[node] if item[4] <= now + EPS and item[0] > now + EPS and item[1] > now + EPS]
            demand = sum(item[3] for item in intervals[node] if now - EPS <= item[4] < now + DT - EPS)
            remaining = sum(max(0.0, item[1] - now) * (EDGE_RATE if node == "edge" else LOCAL_RATE) for item in active + waiting)
            resources.append({"scenario": scene, "node": node, "t": cycle, "queue_jobs": [f"{scene}:{key[0]}:{key[1]}:{key[2]}" for _, _, key, _, _ in sorted(waiting, key=lambda x: (x[4], x[2]))], "remaining_work": remaining, "scheduled_bits": transfer_slots.get(cycle, 0), "service_state": "busy" if active else "idle", "demand_time": demand, "available_time": DT, "U": demand / DT})
    placements_out = [{"scenario": scene, "robot": r, "window": w, "perception": placement_map[(r, w)]["perception"], "planning": placement_map[(r, w)]["planning"]} for r in ROBOTS for w in range(WINDOWS)]
    deadline_ok = sum(1 for row in cycle_rows if not row["deadline_miss"] and row["completion_time"] is not None) / (len(ROBOTS) * HORIZON)
    timely = sum(1 for r in ROBOTS if goal_first[r] is not None and goal_first[r] <= SCENE_DATA[scene]["deadlines"][r] and not any(unsafe_history[r][: goal_first[r] - 1])) / len(ROBOTS)
    generated_bits = len(transfers) * PACKET_BITS
    communication_window_values = []
    for robot in ROBOTS:
        for window in range(WINDOWS):
            delivered = [
                item for item in transfers
                if item["delivered"] and item["robot"] == robot and item["cycle"] // WINDOW_LENGTH == window
            ]
            if not delivered:
                communication_window_values.append(0.0)
                continue
            freshness = sum(_clip(1.0 - item["wait_cycles"] / HORIZON, 0.0, 1.0) for item in delivered) / len(delivered)
            communication_window_values.append(0.6 + 0.4 * freshness)
    migrations = sum(1 for r in ROBOTS for w in range(WINDOWS) for stage in MIGRATABLE if placement_map[(r, w)][stage.lower()] != ("local" if w == 0 else placement_map[(r, w - 1)][stage.lower()]))
    aux1 = sum(communication_window_values) / COMMUNICATION_WINDOWS
    aux2 = 0.6 * _clip(1.0 - migrations / 24.0, 0.0, 1.0) + 0.4 * _clip(1.0 - generated_bits / (HORIZON * CYCLE_BITS), 0.0, 1.0)
    transfer_order = [
        _transfer_priority(item["predecessor_finish"], item["robot"], item["cycle"], STAGES.index(item["stage"]))
        for item in transfers
    ]
    communication_ok = transfer_order == sorted(transfer_order) and len(transfers) == len(transfer_cache) and all(bits <= CYCLE_BITS for bits in transfer_slots.values()) and all(
        item["wait_cycles"] == max(0, math.ceil((item["arrival"] - item["predecessor_finish"] - EPS) / DT))
        and _integer(item["start_cycle"])
        and item["start_cycle"] >= math.ceil((item["predecessor_finish"] - EPS) / DT)
        and math.isclose(item["arrival"], item["start_cycle"] * DT + _link(scene, item["start_cycle"])[1], rel_tol=0.0, abs_tol=EPS)
        for item in transfers
    )
    replay_issues = {"dag": len(stages) == ROBOTS.__len__() * HORIZON * len(STAGES), "communication": communication_ok, "resources": all(row["U"] >= -EPS and row["available_time"] > 0 for row in resources), "motion": all(_number(value) for row in robot_rows for value in row["position"]), "deadline": all(row["deadline_miss"] == (row["completion_time"] is None or row["latency"] > DEADLINE + EPS) for row in cycle_rows), "safety": all(len(row["obstacle_distance"]) == 2 and row["pair_distance"] >= -EPS for row in safety_rows)}
    return {"scene": scene, "placements": placements_out, "cycles": cycle_rows, "stages": stages, "resources": resources, "robots": robot_rows, "safety": safety_rows, "deadline_ok": deadline_ok, "timely": timely, "aux1": aux1, "aux2": aux2, "transfers": transfers, "replay_issues": replay_issues}


def _decision_maps(data: Any) -> tuple[dict[str, dict[tuple[int, int], Any]], dict[str, dict[tuple[int, int], Any]]]:
    placements: dict[str, dict[tuple[int, int], Any]] = {}
    commands: dict[str, dict[tuple[int, int], Any]] = {}
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list): return placements, commands
    for row in data["scenarios"]:
        if not isinstance(row, Mapping) or row.get("scenario") not in SCENES: continue
        scene = row["scenario"]
        placements[scene] = {}
        commands[scene] = {}
        for record in row.get("placements", []) if isinstance(row.get("placements"), list) else []:
            if not isinstance(record, Mapping):
                continue
            key = (record.get("robot"), record.get("window"))
            try:
                hash(key)
            except TypeError:
                continue
            placements[scene][key] = record
        for record in row.get("commands", []) if isinstance(row.get("commands"), list) else []:
            if not isinstance(record, Mapping):
                continue
            key = (record.get("robot"), record.get("cycle"))
            try:
                hash(key)
            except TypeError:
                continue
            commands[scene][key] = record
    return placements, commands


def _expected_keys() -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for scene in SCENES:
        keys.add((scene, "scene"))
        for robot in ROBOTS:
            for window in range(WINDOWS): keys.add((scene, "placement", robot, window))
            for cycle in range(HORIZON): keys.add((scene, "command", robot, cycle))
    return keys


def _index_records(data: Any) -> dict[Hashable, Any]:
    out: dict[Hashable, Any] = {}
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list): return out
    for row in data["scenarios"]:
        if not isinstance(row, Mapping) or row.get("scenario") not in SCENES: continue
        scene = row["scenario"]
        out.setdefault((scene, "scene"), row)
        for field, tag, parts in (("placements", "placement", ("robot", "window")), ("commands", "command", ("robot", "cycle"))):
            for record in row.get(field, []) if isinstance(row.get(field), list) else []:
                if isinstance(record, Mapping) and all(part in record for part in parts):
                    key = (scene, tag, *(record[part] for part in parts))
                    try:
                        hash(key)
                    except TypeError:
                        continue
                    out.setdefault(key, record)
    return out


def _result_json_atoms(data: Any, parse_ok: bool = True, parse_error: str | None = None) -> list[AtomVerdict]:
    if not parse_ok: return [_atom("result_json", "parse", False, parse_error or "strict JSON parse failed"), _atom("result_json", "root", False, "root unavailable"), _atom("result_json", "scene_containers", False, "scenarios unavailable")]
    root = _shape(data, ROOT_KEYS) and isinstance(data.get("scenarios"), list)
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    containers = isinstance(scenes, list) and all(_shape(row, SCENE_KEYS) for row in scenes)
    return [_atom("result_json", "parse", True, "strict JSON parsed"), _atom("result_json", "root", root, "exact scenarios root" if root else "expected root key scenarios"), _atom("result_json", "scene_containers", containers, "scene containers valid" if containers else "each scene needs exact scenario, placements, commands keys")]


def _valid_placement(row: Any) -> bool:
    return _shape(row, PLACEMENT_KEYS) and _integer(row["robot"]) and row["robot"] in ROBOTS and _integer(row["window"]) and row["window"] in range(WINDOWS) and isinstance(row["perception"], str) and row["perception"] in {"local", "edge"} and isinstance(row["planning"], str) and row["planning"] in {"local", "edge"}


def _valid_command(row: Any) -> bool:
    command = row.get("command") if isinstance(row, Mapping) else None
    if not (_shape(row, COMMAND_KEYS) and _integer(row["robot"]) and row["robot"] in ROBOTS and _integer(row["cycle"]) and row["cycle"] in range(HORIZON) and isinstance(command, list) and len(command) == 2):
        return False
    v = _finite_float(command[0])
    omega = _finite_float(command[1])
    return v is not None and omega is not None and -EPS <= v <= MAX_V + EPS and -MAX_OMEGA - EPS <= omega <= MAX_OMEGA + EPS


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    issues = {"placements": [], "commands": []}
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(scenes, list):
        issues["placements"].append("path=scenarios: expected list")
        issues["commands"].append("path=scenarios: expected list")
    for i, row in enumerate(scenes if isinstance(scenes, list) else []):
        for name, validator in (("placements", _valid_placement), ("commands", _valid_command)):
            values = row.get(name) if isinstance(row, Mapping) else None
            if not isinstance(values, list): issues[name].append(f"path=scenarios[{i}].{name}: expected list")
            else:
                for j, value in enumerate(values):
                    if not validator(value): issues[name].append(f"path=scenarios[{i}].{name}[{j}]: invalid field type/range")
    return [_atom("result_evidence", name, not values, "decision evidence valid" if not values else " | ".join(values[:6])) for name, values in issues.items()]


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    expected = _expected_keys(); seen: dict[tuple[Any, ...], int] = {}; defects: list[str] = []
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(scenes, list): return 0.0, ["missing scenarios list"]
    for row in scenes:
        if not isinstance(row, Mapping): defects.append("invalid scene record"); continue
        scene = row.get("scenario")
        if not isinstance(scene, str):
            defects.append(f"unknown scene key {scene!r}")
            continue
        key = (scene, "scene")
        if key not in expected: defects.append(f"unknown scene key {scene!r}")
        elif key in seen: defects.append(f"duplicate scene key {scene!r}")
        seen[key] = seen.get(key, 0) + 1
        for field, tag, parts in (("placements", "placement", ("robot", "window")), ("commands", "command", ("robot", "cycle"))):
            values = row.get(field, [])
            if not isinstance(values, list): continue
            for pos, record in enumerate(values):
                if not isinstance(record, Mapping) or any(part not in record for part in parts): defects.append(f"invalid {field} record at {pos}"); continue
                values = tuple(record[part] for part in parts)
                try:
                    hash((scene, tag, *values))
                except TypeError:
                    defects.append(f"unhashable {field} key at {pos}")
                    continue
                k = (scene, tag, *values)
                if k not in expected: defects.append(f"unknown {field} key {k!r}")
                elif k in seen: defects.append(f"duplicate {field} key {k!r}")
                seen[k] = seen.get(k, 0) + 1
    for key in sorted(expected, key=repr):
        if key not in seen: defects.append(f"missing canonical key {key!r}")
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _legality_atoms(replayed: Mapping[str, Any], data: Any, prerequisite: Sequence[str] | None) -> list[AtomVerdict]:
    names = ("residence", "command_update")
    if prerequisite: return [_atom("decision_legality", name, False, f"not assessed because prerequisite checks failed: {list(prerequisite)!r}") for name in names]
    issues = {name: [] for name in names}
    for scene in SCENES:
        row = next(r for r in data["scenarios"] if r.get("scenario") == scene)
        pmap = {(r["robot"], r["window"]): r for r in row["placements"]}; cmap = {(r["robot"], r["cycle"]): r for r in row["commands"]}
        cycle_map = {(r["robot"], r["cycle"]): r for r in replayed[scene]["cycles"]}
        for (robot, window), record in pmap.items():
            if window == 0:
                continue
            for stage in MIGRATABLE:
                field = stage.lower()
                changed_now = record[field] != pmap[(robot, window - 1)][field]
                changed_before = pmap[(robot, window - 1)][field] != (
                    "local" if window == 1 else pmap[(robot, window - 2)][field]
                )
                if changed_now and changed_before:
                    issues["residence"].append(f"{scene}: robot={robot} window={window} stage={stage} migrated in consecutive windows")
        for robot in ROBOTS:
            previous = [0.0, 0.0]
            for cycle in range(HORIZON):
                command = cmap[(robot, cycle)]["command"]
                update = cycle_map[(robot, cycle)]["new_control_cycle_id"]
                if update is None and any(abs(float(command[i]) - previous[i]) > EPS for i in (0, 1)): issues["command_update"].append(f"{scene}: robot={robot} cycle={cycle} changed command without a newly exposed prior Control")
                previous = [float(command[0]), float(command[1])]
    return [_atom("decision_legality", name, not values, "decision relation valid" if not values else " | ".join(values[:8])) for name, values in issues.items()]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=EPS)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_value(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def _replay_atoms(data: Any, replayed: Mapping[str, Any], prerequisite: Sequence[str] | None) -> list[AtomVerdict]:
    names = ("dag", "communication", "resources", "motion", "deadline", "safety")
    if prerequisite: return [_atom("replay_validity", name, False, f"not assessed because prerequisite checks failed: {list(prerequisite)!r}") for name in names]
    groups = {name: [] for name in names}
    for scene in SCENES:
        rep = replayed.get(scene)
        if not isinstance(rep, Mapping):
            for name in names: groups[name].append(f"{scene}: replay missing")
            continue
        fresh = _simulate_scene(scene, {(r["robot"], r["window"]): r for r in next(x for x in data["scenarios"] if x.get("scenario") == scene)["placements"]}, {(r["robot"], r["cycle"]): r for r in next(x for x in data["scenarios"] if x.get("scenario") == scene)["commands"]})
        for name in names:
            if rep.get("replay_issues", {}).get(name) is not True: groups[name].append(f"{scene}: fixed replay invariant failed")
        projections = {
            "dag": "stages",
            "communication": "transfers",
            "resources": "resources",
            "motion": "robots",
            "deadline": "cycles",
            "safety": "safety",
        }
        for name, field in projections.items():
            if not _same_value(rep.get(field, []), fresh.get(field, [])):
                groups[name].append(f"{scene}: {field} differs from independent recomputation")
        for name, fields in {
            "deadline": ("deadline_ok", "timely"),
            "communication": ("aux1",),
            "resources": ("aux2",),
        }.items():
            for field in fields:
                if not _same_value(rep.get(field), fresh.get(field)):
                    groups[name].append(f"{scene}: derived summary {field} differs from independent recomputation")
    return [_atom("replay_validity", name, not issues, "replay audit passed" if not issues else " | ".join(issues[:8])) for name, issues in groups.items()]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    return {"primary_metrics_1": sum(float(r["deadline_ok"]) for r in replayed.values()) / len(SCENES), "primary_metrics_2": sum(float(r["timely"]) for r in replayed.values()) / len(SCENES), "auxiliary_metrics_1": sum(float(r["aux1"]) for r in replayed.values()) / len(SCENES), "auxiliary_metrics_2": sum(float(r["aux2"]) for r in replayed.values()) / len(SCENES)}


def _evaluate_data(data: Any, parse_ok: bool = True, parse_error: str | None = None) -> dict[str, Any]:
    scores: dict[str, float] = {}; diagnostics: dict[str, str] = {}
    atoms = _result_json_atoms(data, parse_ok, parse_error); scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"]); diagnostics["result_json"] = _diagnose_atoms(atoms)
    atoms = _evidence_atoms(data); scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"]); diagnostics["result_evidence"] = _diagnose_atoms(atoms)
    coverage, defects = _coverage_score(data); scores["canonical_coverage"] = coverage; diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; defects={defects[:12]}"
    prerequisites = [name for name in ("result_json", "result_evidence", "canonical_coverage") if scores[name] < 1.0 - EPS]
    replayed: dict[str, Any] = {}
    if not prerequisites:
        placements, commands = _decision_maps(data)
        try:
            for scene in SCENES: replayed[scene] = _simulate_scene(scene, placements[scene], commands[scene])
        except Exception as exc:
            prerequisites.append("replay_setup"); diagnostics["replay_validity"] = f"replay setup failed: {type(exc).__name__}: {exc}"
    atoms = _legality_atoms(replayed, data, prerequisites); scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"]); diagnostics["decision_legality"] = _diagnose_atoms(atoms)
    if scores["decision_legality"] < 1.0 - EPS: prerequisites = [*prerequisites, "decision_legality"]
    atoms = _replay_atoms(data, replayed, prerequisites); scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"]); diagnostics["replay_validity"] = _diagnose_atoms(atoms)
    gate = all(scores.get(name, 0.0) >= 1.0 - EPS for name in VALIDITY_CHECKS)
    metrics = _metrics(replayed) if gate else None; scores.update(metrics or {name: 0.0 for name in METRIC_NAMES})
    for name in METRIC_NAMES: diagnostics[name] = "validity gate passed" if metrics is not None else f"validity gate failed; {name}=0"
    return {"scores": scores, "diagnostics": diagnostics, "records": _index_records(data), "replayed": replayed, "metrics": metrics, "validity_gate": gate}


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    visible_lines = [""] * len(lines)
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
    required = {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"}
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink(): return 0.0, "report.md missing or not regular"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    missing = sorted(name for name in required if not _headings(text).get(name))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        await ws.run("rm -f -- results.json", check=False, timeout=30); state.run_started = True; started = time.monotonic()
        try: result = await ws.run("python run.py", check=False, timeout=600)
        except (asyncio.TimeoutError, TimeoutError):
            state.timed_out = True; state.elapsed_s = time.monotonic() - started; state.official_message = "official run timed out after 600 seconds"
            for name in VALIDITY_CHECKS + METRIC_NAMES: state.scores[name] = 0.0; state.diagnostics[name] = state.official_message
            return state
        state.elapsed_s = time.monotonic() - started; state.run_finished = True; state.exit_code = result.exit_code
        stderr_text = str(getattr(result, "stderr", "")).lower()
        state.timed_out = (state.elapsed_s >= 600.0 - EPS or state.exit_code is None or "timeout" in stderr_text or "timed out" in stderr_text)
        path = Path(ws.root) / "results.json"; state.result_file_is_new_regular = path.is_file() and not path.is_symlink(); state.official_ok = result.exit_code == 0 and not state.timed_out and state.result_file_is_new_regular; state.official_message = f"exit={result.exit_code}, elapsed={state.elapsed_s:.3f}s, timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        if not state.official_ok:
            for name in VALIDITY_CHECKS + METRIC_NAMES: state.scores[name] = 0.0; state.diagnostics[name] = f"official run failed; {state.official_message}"
            return state
        parse_error = None
        try: state.data = _strict_json(path.read_text(encoding="utf-8")); state.parse_ok = True
        except Exception as exc: state.data = None; state.parse_ok = False; parse_error = f"strict JSON parse failed: {type(exc).__name__}: {exc}"
        evaluated = _evaluate_data(state.data, state.parse_ok, parse_error); state.scores.update(evaluated["scores"]); state.records = evaluated["records"]; state.replayed = evaluated["replayed"]; state.metrics = evaluated["metrics"]; state.validity_gate = evaluated["validity_gate"]; state.diagnostics.update(evaluated["diagnostics"])
        return state
    except Exception as exc:
        state.run_finished = True
        state.timed_out = (isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower() or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS))
        message = f"grader exception {type(exc).__name__}: {exc}"
        if not state.official_ok:
            state.official_message = message
        for name in VALIDITY_CHECKS + METRIC_NAMES: state.scores[name] = 0.0; state.diagnostics[name] = message
        return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE: return _CACHE[key]
    if key not in _INFLIGHT: _INFLIGHT[key] = asyncio.create_task(_build_state_uncached(ws))
    state = await _INFLIGHT[key]; _CACHE[key] = state; return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink(): return 0.0, "run.py missing or not regular"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (1.0, "static compilation passed") if result.exit_code == 0 else (0.0, f"static compilation failed; exit={result.exit_code}")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws); return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]: return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws); return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity", *METRIC_NAMES): globals()[_name] = check(id=_name)(lambda ws, name=_name: _score(ws, name))
