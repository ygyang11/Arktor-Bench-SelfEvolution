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

EPS = 1e-6
HORIZON = 80
N_ROBOTS = 4
GRID_X = 22
GRID_Y = 18
QUEUE_CAPACITY = 8
CHANNEL_CAPACITY = 12
BURST_CAPACITY = 6
TOTAL_BUDGET = 640
SERVICE_UNITS = 2
GRAPH_RESIDUAL_REF = 16.0
POSE_MSE_REF = 22.0**2
SCENARIO_NAMES = (
    "dense_overlap",
    "ambiguous_similarity",
    "budget_burst",
    "heterogeneous_drift",
    "crossing_trajectories",
)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
GATE_CHECKS = (
    "official_run",
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.30,
        "result_json.scenario_shape": 0.50,
    },
    "result_evidence": {
        "result_evidence.scenario_fields": 0.25,
        "result_evidence.action_fields": 0.75,
    },
    "decision_legality": {
        "decision_legality.action_values": 0.25,
        "decision_legality.visible_edges": 0.30,
        "decision_legality.request_history": 0.20,
        "decision_legality.energy": 0.15,
        "decision_legality.replayable_events": 0.10,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.observations": 0.20,
        "replay_validity.candidates": 0.20,
        "replay_validity.communication": 0.20,
        "replay_validity.mission": 0.20,
    },
}


SCENARIOS: dict[str, dict[str, Any]] = {
    "dense_overlap": {
        "starts": ((1, 2), (1, 15), (20, 2), (20, 15)),
        "goals": ((20, 8), (11, 9), (2, 8), (11, 2)),
        "obstacles": (((8, 9), (0, 5)), ((8, 9), (12, 17))),
        "landmarks": ((4, 4), (4, 13), (11, 4), (11, 13), (18, 9)),
        "deadline": 62,
        "drift": 0.020,
        "multipliers": (1.0, 1.0, 1.0, 1.0),
        "burst": False,
    },
    "ambiguous_similarity": {
        "starts": ((2, 2), (2, 15), (19, 2), (19, 15)),
        "goals": ((19, 8), (11, 9), (2, 8), (11, 15)),
        "obstacles": (((9, 12), (6, 7)), ((9, 12), (10, 11))),
        "landmarks": ((5, 4), (5, 13), (16, 4), (16, 13)),
        "deadline": 66,
        "drift": 0.024,
        "multipliers": (1.0, 1.0, 1.0, 1.0),
        "burst": False,
    },
    "budget_burst": {
        "starts": ((1, 1), (1, 16), (20, 1), (20, 16)),
        "goals": ((20, 8), (11, 9), (2, 8), (11, 1)),
        "obstacles": (((6, 7), (5, 12)), ((15, 16), (5, 12))),
        "landmarks": ((4, 2), (4, 16), (11, 3), (11, 15), (19, 9)),
        "deadline": 58,
        "drift": 0.018,
        "multipliers": (1.0, 1.0, 1.0, 1.0),
        "burst": True,
    },
    "heterogeneous_drift": {
        "starts": ((2, 8), (10, 2), (19, 8), (10, 15)),
        "goals": ((2, 15), (11, 9), (19, 2), (20, 8)),
        "obstacles": (((8, 9), (3, 14)),),
        "landmarks": ((4, 8), (11, 2), (11, 16), (18, 8)),
        "deadline": 70,
        "drift": 0.028,
        "multipliers": (0.8, 1.0, 1.3, 1.6),
        "burst": False,
    },
    "crossing_trajectories": {
        "starts": ((1, 3), (1, 14), (20, 3), (20, 14)),
        "goals": ((20, 14), (11, 9), (2, 9), (11, 3)),
        "obstacles": (((10, 11), (0, 5)), ((10, 11), (12, 17))),
        "landmarks": ((4, 4), (4, 13), (17, 4), (17, 13)),
        "deadline": 64,
        "drift": 0.022,
        "multipliers": (1.0, 1.0, 1.0, 1.0),
        "burst": False,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = passed
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
        self.scenarios: dict[tuple[str, int], Any] = {}
        self.replayed: dict[tuple[str, int], Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _u64(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


def _hash_float(text: str) -> float:
    return _u64(text)


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[group][atom_id])


def _shape_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_ok = isinstance(data, dict) and set(data) == {"scenarios"}
    scenarios_ok = root_ok and isinstance(data.get("scenarios"), list)
    return [
        _atom(
            "result_json",
            "parse",
            parse_ok,
            "strict JSON parse" if parse_ok else "strict JSON parsing failed",
        ),
        _atom(
            "result_json",
            "root",
            root_ok,
            f"root_keys={sorted(data) if isinstance(data, dict) else type(data).__name__}",
        ),
        _atom(
            "result_json",
            "scenario_shape",
            scenarios_ok,
            "scenarios is an array" if scenarios_ok else "root.scenarios is not an array",
        ),
    ]


def _action_shape(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    slot = action.get("slot")
    robot = action.get("robot")
    move = action.get("move")
    validate_edge = action.get("validate_edge")
    return (
        set(action) == {"slot", "robot", "move", "broadcast_metadata", "validate_edge"}
        and _integer(slot)
        and isinstance(robot, str)
        and robot in {f"r{i}" for i in range(N_ROBOTS)}
        and isinstance(move, str)
        and move in {"stay", "N", "E", "S", "W"}
        and isinstance(action["broadcast_metadata"], bool)
        and (validate_edge is None or isinstance(validate_edge, str))
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    records = data.get("scenarios") if isinstance(data, dict) else None
    scenario_ok = isinstance(records, list) and all(
        isinstance(record, dict)
        and set(record) == {"scenario", "variant", "actions"}
        and isinstance(record.get("scenario"), str)
        and _integer(record.get("variant"))
        and isinstance(record.get("actions"), list)
        for record in records or []
    )
    action_ok = scenario_ok and all(
        _action_shape(action) for record in records for action in record.get("actions", [])
    )
    return [
        _atom(
            "result_evidence",
            "scenario_fields",
            scenario_ok,
            "scenario records have the exact declared fields"
            if scenario_ok
            else "scenario record fields/types invalid",
        ),
        _atom(
            "result_evidence",
            "action_fields",
            action_ok,
            "all action records have exact declared fields"
            if action_ok
            else "one or more action records have invalid fields/types",
        ),
    ]


def _index(data: Any) -> dict[tuple[str, int], Any]:
    result: dict[tuple[str, int], Any] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return result
    for record in data["scenarios"]:
        if not isinstance(record, dict):
            continue
        key = (record.get("scenario"), record.get("variant"))
        try:
            hash(key)
        except TypeError:
            continue
        if key not in result:
            result[key] = record
    return result


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    records = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return 0.0, ["scenarios is not an array"]
    counts: dict[tuple[Any, Any], int] = {}
    for record in records:
        if isinstance(record, dict):
            key = (record.get("scenario"), record.get("variant"))
            try:
                counts[key] = counts.get(key, 0) + 1
            except TypeError:
                defects.append(f"unhashable scenario key={key!r}")
        else:
            defects.append("non-object scenario record")
    for key in SCENARIO_KEYS:
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"missing scenario={key[0]}|variant={key[1]}")
        elif count > 1:
            defects.extend([f"duplicate scenario={key[0]}|variant={key[1]}"] * (count - 1))
    for key in counts:
        if key not in SCENARIO_KEYS:
            defects.append(f"unknown scenario key={key!r}")
    for key in SCENARIO_KEYS:
        records_for_key = [
            r
            for r in records
            if isinstance(r, dict) and (r.get("scenario"), r.get("variant")) == key
        ]
        if len(records_for_key) != 1:
            continue
        actions = records_for_key[0].get("actions")
        if not isinstance(actions, list):
            defects.append(f"{key} actions is not an array")
            continue
        action_counts: dict[tuple[Any, Any], int] = {}
        for action in actions:
            if isinstance(action, dict):
                action_key = (action.get("slot"), action.get("robot"))
                try:
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                except TypeError:
                    defects.append(f"{key} unhashable action key={action_key!r}")
            else:
                defects.append(f"{key} non-object action")
        expected = {(slot, f"r{robot}") for slot in range(HORIZON) for robot in range(N_ROBOTS)}
        for action_key in expected:
            count = action_counts.get(action_key, 0)
            if count == 0:
                defects.append(f"{key} missing action={action_key!r}")
            elif count > 1:
                defects.extend([f"{key} duplicate action={action_key!r}"] * (count - 1))
        for action_key in action_counts:
            if action_key not in expected:
                defects.append(f"{key} unknown action={action_key!r}")
    normalized = max(0.0, 4.0 - 0.5 * len(defects)) / 4.0
    return normalized, defects


def _inside_obstacle(spec: Mapping[str, Any], point: tuple[int, int]) -> bool:
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for (x0, x1), (y0, y1) in spec["obstacles"])


def _traversable(spec: Mapping[str, Any], point: tuple[int, int]) -> bool:
    return 0 <= point[0] < GRID_X and 0 <= point[1] < GRID_Y and not _inside_obstacle(spec, point)


def _move_point(point: tuple[int, int], move: str) -> tuple[int, int]:
    dxdy = {"stay": (0, 0), "N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
    dx, dy = dxdy.get(move, (0, 0))
    return point[0] + dx, point[1] + dy


def _shortest_distance(
    spec: Mapping[str, Any], start: tuple[int, int], goal: tuple[int, int]
) -> int:
    if start == goal:
        return 0
    queue: deque[tuple[int, int]] = deque([start])
    distance = {start: 0}
    while queue:
        current = queue.popleft()
        for move in ("N", "E", "S", "W"):
            nxt = _move_point(current, move)
            if nxt in distance or not _traversable(spec, nxt):
                continue
            distance[nxt] = distance[current] + 1
            if nxt == goal:
                return distance[nxt]
            queue.append(nxt)
    return GRID_X + GRID_Y


def _descriptor(
    spec: Mapping[str, Any],
    point: tuple[int, int],
    drift: tuple[float, float],
    variant: int,
    robot: int,
) -> dict[str, Any]:
    sigma = 0.10 + 0.05 * variant
    sigma *= spec["multipliers"][robot]
    visible: list[int] = []
    quality: dict[int, float] = {}
    descriptor: list[float] = []
    anchors: list[tuple[float, float]] = []
    for landmark_id, landmark in enumerate(spec["landmarks"]):
        dx = landmark[0] - point[0]
        dy = landmark[1] - point[1]
        distance = math.hypot(dx, dy)
        is_visible = distance <= 3.0 + EPS
        if is_visible:
            visible.append(landmark_id)
        d = _clip(distance / 3.0)
        q = _clip(1.0 - 0.10 * sigma - 0.03 * d)
        quality[landmark_id] = q
        if is_visible:
            anchors.append((float(landmark[0]), float(landmark[1])))
            descriptor.extend((1.0, dx / 3.0, dy / 3.0, q))
        else:
            descriptor.extend((0.0, 0.0, 0.0, 0.0))
    if anchors:
        anchor = (
            sum(value[0] for value in anchors) / len(anchors),
            sum(value[1] for value in anchors) / len(anchors),
        )
    else:
        anchor = (0.0, 0.0)
    return {
        "visible": tuple(visible),
        "quality": quality,
        "descriptor": tuple(descriptor),
        "pose": (point[0] + drift[0], point[1] + drift[1]),
        "anchor": anchor,
    }


def _edge_residual(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    delta_hat = (first["pose"][0] - second["pose"][0], first["pose"][1] - second["pose"][1])
    delta_meta = (
        first["anchor"][0] - second["anchor"][0],
        first["anchor"][1] - second["anchor"][1],
    )
    return math.hypot(delta_hat[0] - delta_meta[0], delta_hat[1] - delta_meta[1])


def _edge_id(first_id: str, second_id: str) -> str:
    a, b = sorted((first_id, second_id))
    return f"edge={a}--{b}"


def _packet_sample(
    name: str, variant: int, kind: str, owner: int, generation: int, edge_id: str | None
) -> float:
    edge_part = edge_id if edge_id is not None else "none"
    packet_id = f"tx={name}|{variant}|{kind}|{owner}|{generation}|{edge_part}"
    return _hash_float(packet_id + f"|salt={17 * variant}")


def _actions_by_slot(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    actions = record.get("actions", []) if isinstance(record, Mapping) else []
    if not isinstance(actions, list):
        return result
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        slot, robot = action.get("slot"), action.get("robot")
        if (
            _integer(slot)
            and isinstance(robot, str)
            and robot.startswith("r")
            and robot[1:].isdigit()
        ):
            result.setdefault((slot, int(robot[1:])), action)
    return result


def _replay(
    name: str, variant: int, record: Mapping[str, Any], collect_errors: bool = False
) -> dict[str, Any]:
    spec = SCENARIOS[name]
    actions = _actions_by_slot(record)
    positions: list[list[tuple[int, int]]] = [list(spec["starts"])]
    drifts: list[list[tuple[float, float]]] = [[(0.0, 0.0) for _ in range(N_ROBOTS)]]
    energies: list[list[float]] = [[100.0 for _ in range(N_ROBOTS)]]
    goal_arrivals: list[int | None] = [
        0 if spec["starts"][i] == spec["goals"][i] else None for i in range(N_ROBOTS)
    ]
    received: dict[str, dict[str, Any]] = {}
    pending: dict[str, dict[str, Any]] = {}
    attempted: set[str] = set()
    generated_edge_pairs: set[tuple[str, str]] = set()
    arrivals: dict[int, list[dict[str, Any]]] = {}
    waiting: deque[dict[str, Any]] = deque()
    active: dict[str, Any] | None = None
    graph_edges: list[dict[str, Any]] = []
    errors: list[str] = []
    safety_events = 0
    motion_cost = 0.0
    accepted_units = 0
    total_candidates = 0
    dropped_candidates = 0
    validation_wait_slots = 0
    informative_units = 0
    seen_informative: set[str] = set()
    actions_valid = True
    generated_edges: list[str] = []
    candidate_records: list[dict[str, Any]] = []
    metadata_requests = 0
    metadata_successes = 0
    verification_requests = 0
    verification_successes = 0
    verification_completions = 0

    def add_candidates(boundary: int) -> None:
        nonlocal total_candidates
        items = list(received.items())
        for index, (first_id, first) in enumerate(items):
            for second_id, second in items[index + 1 :]:
                if first["robot"] == second["robot"]:
                    continue
                pair = tuple(sorted((first_id, second_id)))
                if pair in generated_edge_pairs:
                    continue
                common = set(first["visible"]) & set(second["visible"])
                if not common:
                    continue
                distance = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(first["descriptor"], second["descriptor"]))
                )
                similarity = _clip(1.0 - distance / 2.0)
                if similarity + EPS < 0.55:
                    continue
                edge = _edge_id(first_id, second_id)
                generated_edge_pairs.add(pair)
                pending[edge] = {
                    "edge_id": edge,
                    "first_id": first_id,
                    "second_id": second_id,
                    "generation": boundary,
                    "expiry": boundary + 10,
                    "similarity": similarity,
                    "residual": _edge_residual(first, second),
                    "consistent": _edge_residual(first, second) <= 0.35 + EPS,
                }
                total_candidates += 1
                generated_edges.append(edge)
                candidate_records.append(dict(pending[edge]))

    for slot in range(HORIZON):
        boundary = slot
        # Arrivals are processed before service. Metadata becomes globally visible here.
        for packet in arrivals.pop(boundary, []):
            if packet["kind"] == "metadata":
                received[packet["obs_id"]] = packet["observation"]
                add_candidates(boundary)
            elif packet["kind"] == "verification":
                if len(waiting) >= QUEUE_CAPACITY:
                    dropped_candidates += 1
                else:
                    packet["arrival"] = boundary
                    waiting.append(packet)
        if active is None and waiting:
            active = waiting.popleft()
            active["service_start"] = boundary
            validation_wait_slots += boundary - active["arrival"]
        if active is not None:
            active["remaining"] -= 1
            if active["remaining"] <= 0:
                verification_completions += 1
                edge = active["edge"]
                if active["hash_pass"]:
                    if edge["consistent"]:
                        first_robot = received[edge["first_id"]]["robot"]
                        second_robot = received[edge["second_id"]]["robot"]
                        current_drifts = drifts[-1]
                        updated = list(current_drifts)
                        for robot in (first_robot, second_robot):
                            updated[robot] = (updated[robot][0] * 0.62, updated[robot][1] * 0.62)
                        drifts[-1] = updated
                        graph_edges.append(
                            {
                                "edge_id": edge["edge_id"],
                                "residual": edge["residual"],
                                "correct": True,
                            }
                        )
                        if edge["edge_id"] not in seen_informative:
                            informative_units += 4
                            seen_informative.add(edge["edge_id"])
                    else:
                        first_robot = received[edge["first_id"]]["robot"]
                        second_robot = received[edge["second_id"]]["robot"]
                        current_drifts = drifts[-1]
                        updated = list(current_drifts)
                        for robot in (first_robot, second_robot):
                            updated[robot] = (updated[robot][0] + 0.08, updated[robot][1] + 0.08)
                        drifts[-1] = updated
                        graph_edges.append(
                            {
                                "edge_id": edge["edge_id"],
                                "residual": edge["residual"],
                                "correct": False,
                            }
                        )
                elif not edge["consistent"] and edge["edge_id"] not in seen_informative:
                    informative_units += 4
                    seen_informative.add(edge["edge_id"])
                active = None
        expired = [edge_id for edge_id, edge in pending.items() if boundary > edge["expiry"]]
        for edge_id in expired:
            pending.pop(edge_id, None)
            dropped_candidates += 1

        # Recover action records. Missing values are retained as errors so replay stays deterministic.
        current_actions = [actions.get((slot, robot), {}) for robot in range(N_ROBOTS)]
        for robot, action in enumerate(current_actions):
            if not _action_shape(action):
                actions_valid = False
                errors.append(f"slot={slot},robot={robot}: action shape/type invalid")

        requests: list[dict[str, Any]] = []
        for robot, action in enumerate(current_actions):
            move = action.get("move")
            broadcast = action.get("broadcast_metadata")
            edge_id = action.get("validate_edge")
            if broadcast is True:
                metadata_requests += 1
                requests.append(
                    {
                        "kind": "metadata",
                        "owner": robot,
                        "generation": slot,
                        "payload": 1,
                        "edge_id": None,
                    }
                )
            if edge_id is not None:
                if not isinstance(edge_id, str) or edge_id not in pending:
                    actions_valid = False
                    errors.append(
                        f"slot={slot},robot={robot}: edge {edge_id!r} not visible/unexpired"
                    )
                elif edge_id in attempted:
                    actions_valid = False
                    errors.append(f"slot={slot},robot={robot}: edge {edge_id!r} requested twice")
                else:
                    edge = pending.pop(edge_id)
                    attempted.add(edge_id)
                    verification_requests += 1
                    requests.append(
                        {
                            "kind": "verification",
                            "owner": robot,
                            "generation": slot,
                            "payload": 4,
                            "edge_id": edge_id,
                            "edge": edge,
                        }
                    )

        capacity = BURST_CAPACITY if spec["burst"] and 24 <= slot <= 35 else CHANNEL_CAPACITY
        requests.sort(
            key=lambda item: (
                item["generation"],
                0 if item["kind"] == "metadata" else 1,
                item["owner"],
                item["edge_id"] or "",
            )
        )
        accepted_this_slot = 0
        comm_cost = [0.0] * N_ROBOTS
        for request in requests:
            if (
                request["payload"] > capacity - accepted_this_slot
                or request["payload"] > TOTAL_BUDGET - accepted_units
            ):
                if request["kind"] == "verification":
                    dropped_candidates += 1
                continue
            accepted_this_slot += request["payload"]
            accepted_units += request["payload"]
            comm_cost[request["owner"]] += request["payload"] * 0.05
            distance = math.hypot(
                positions[-1][request["owner"]][0] - 11, positions[-1][request["owner"]][1] - 9
            )
            d_link = _clip(distance / math.sqrt(21**2 + 17**2))
            busy = 1 if spec["burst"] and 24 <= slot <= 35 else 0
            n = accepted_this_slot
            probability = _clip(0.93 - 0.04 * d_link - 0.06 * busy - 0.025 * (n - 1), 0.10, 0.93)
            sample = _packet_sample(
                name, variant, request["kind"], request["owner"], slot, request["edge_id"]
            )
            request["success"] = sample < probability
            if not request["success"]:
                if request["kind"] == "verification":
                    dropped_candidates += 1
                continue
            if request["kind"] == "metadata":
                metadata_successes += 1
                request["obs_id"] = (
                    f"scenario={name}|variant={variant}|robot={request['owner']}|slot={slot}"
                )
                request["observation_pending"] = True
                arrivals.setdefault(slot + 1, []).append(request)
            else:
                verification_successes += 1
                validation_sample = _hash_float(
                    f"{name}|{variant}|{request['edge_id']}|validate|{17 * variant}"
                )
                request["hash_pass"] = validation_sample < 0.95 - 0.05 * variant
                request["remaining"] = SERVICE_UNITS
                arrivals.setdefault(slot + 1, []).append(request)

        proposed = [action.get("move") for action in current_actions]
        accepted_destinations: set[tuple[int, int]] = set()
        next_positions: list[tuple[int, int]] = []
        start_positions = positions[-1]
        for robot, move in enumerate(proposed):
            current = start_positions[robot]
            destination = _move_point(current, move) if move in {"N", "E", "S", "W"} else current
            unsafe = move != "stay" and (
                not _traversable(spec, destination)
                or destination in accepted_destinations
                or destination in {start_positions[j] for j in range(robot + 1, N_ROBOTS)}
            )
            if unsafe:
                safety_events += 1
                destination = current
            accepted_destinations.add(destination)
            next_positions.append(destination)
            motion_cost += 1.0 if destination != current else 0.2
        next_energies = [
            energies[-1][robot]
            - (1.0 if next_positions[robot] != start_positions[robot] else 0.2)
            - comm_cost[robot]
            for robot in range(N_ROBOTS)
        ]
        if any(value < -EPS for value in next_energies):
            actions_valid = False
            errors.append(f"slot={slot}: energy below zero after admitted work")
        for robot in range(N_ROBOTS):
            if next_positions[robot] == spec["goals"][robot] and goal_arrivals[robot] is None:
                goal_arrivals[robot] = slot + 1
        next_drifts = list(drifts[-1])
        for robot in range(N_ROBOTS):
            point = next_positions[robot]
            observation = _descriptor(spec, point, next_drifts[robot], variant, robot)
            action = current_actions[robot]
            for request in requests:
                if (
                    request.get("kind") == "metadata"
                    and request.get("owner") == robot
                    and request.get("observation_pending")
                ):
                    request["observation"] = {
                        "obs_id": f"scenario={name}|variant={variant}|robot={robot}|slot={slot}",
                        "robot": robot,
                        "slot": slot,
                        **observation,
                    }
            drift_increment = []
            for axis in range(2):
                sample = _hash_float(f"{name}|{variant}|r{robot}|t{slot}|drift|{axis}")
                drift_increment.append(
                    spec["drift"]
                    * (1.0 + 0.2 * variant)
                    * spec["multipliers"][robot]
                    * (2.0 * sample - 1.0)
                )
            next_drifts[robot] = (
                next_drifts[robot][0] + drift_increment[0],
                next_drifts[robot][1] + drift_increment[1],
            )
        positions.append(next_positions)
        energies.append(next_energies)
        drifts.append(next_drifts)

    # Terminal boundary: process one final arrival/service event and close open work.
    boundary = HORIZON
    for packet in arrivals.pop(boundary, []):
        if packet["kind"] == "metadata":
            received[packet["obs_id"]] = packet["observation"]
            add_candidates(boundary)
        elif packet["kind"] == "verification":
            if len(waiting) >= QUEUE_CAPACITY:
                dropped_candidates += 1
            else:
                packet["arrival"] = boundary
                waiting.append(packet)
    if active is None and waiting:
        active = waiting.popleft()
        validation_wait_slots += boundary - active["arrival"]
    if active is not None:
        active["remaining"] -= 1
        if active["remaining"] <= 0:
            verification_completions += 1
            edge = active["edge"]
            if active["hash_pass"] and edge["consistent"]:
                first_robot = received[edge["first_id"]]["robot"]
                second_robot = received[edge["second_id"]]["robot"]
                updated = list(drifts[-1])
                for robot in (first_robot, second_robot):
                    updated[robot] = (updated[robot][0] * 0.62, updated[robot][1] * 0.62)
                drifts[-1] = updated
                graph_edges.append(
                    {"edge_id": edge["edge_id"], "residual": edge["residual"], "correct": True}
                )
                if edge["edge_id"] not in seen_informative:
                    informative_units += 4
                    seen_informative.add(edge["edge_id"])
            elif active["hash_pass"]:
                first_robot = received[edge["first_id"]]["robot"]
                second_robot = received[edge["second_id"]]["robot"]
                updated = list(drifts[-1])
                for robot in (first_robot, second_robot):
                    updated[robot] = (updated[robot][0] + 0.08, updated[robot][1] + 0.08)
                drifts[-1] = updated
                graph_edges.append(
                    {"edge_id": edge["edge_id"], "residual": edge["residual"], "correct": False}
                )
            elif not edge["consistent"] and edge["edge_id"] not in seen_informative:
                informative_units += 4
                seen_informative.add(edge["edge_id"])
            active = None
    dropped_candidates += len(pending) + len(waiting) + (1 if active is not None else 0)
    pending.clear()
    waiting.clear()

    # Attach packet-level state needed by metrics and check diagnostics.
    for packet in arrivals.values():
        dropped_candidates += sum(1 for item in packet if item.get("kind") == "verification")
    return {
        "positions": positions,
        "drifts": drifts,
        "energies": energies,
        "goal_arrivals": goal_arrivals,
        "graph_edges": graph_edges,
        "safety_events": safety_events,
        "motion_cost": motion_cost,
        "accepted_units": accepted_units,
        "total_candidates": total_candidates,
        "dropped_candidates": dropped_candidates,
        "validation_wait_slots": validation_wait_slots,
        "informative_units": informative_units,
        "generated_edges": generated_edges,
        "candidate_records": candidate_records,
        "metadata_requests": metadata_requests,
        "metadata_successes": metadata_successes,
        "verification_requests": verification_requests,
        "verification_successes": verification_successes,
        "verification_completions": verification_completions,
        "actions_valid": actions_valid,
        "errors": errors,
    }


def _legality(record_map: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures = {
        "action_values": [],
        "visible_edges": [],
        "request_history": [],
        "energy": [],
        "replayable_events": [],
    }
    for key in SCENARIO_KEYS:
        record = record_map.get(key, {})
        state = _replay(key[0], key[1], record)
        for error in state["errors"]:
            if "action shape" in error:
                failures["action_values"].append(f"{key}: {error}")
            elif "edge" in error and "requested twice" not in error:
                failures["visible_edges"].append(f"{key}: {error}")
            elif "twice" in error:
                failures["request_history"].append(f"{key}: {error}")
            elif "energy" in error:
                failures["energy"].append(f"{key}: {error}")
            else:
                failures["replayable_events"].append(f"{key}: {error}")
    return [
        _atom(
            "decision_legality",
            name,
            not failures[name],
            "passed" if not failures[name] else " | ".join(failures[name][:8]),
        )
        for name in failures
    ]


def _replay_atoms(
    record_map: Mapping[tuple[str, int], Any], replayed: Mapping[tuple[str, int], Any]
) -> list[AtomVerdict]:
    failures = {
        name: [] for name in ("motion", "observations", "candidates", "communication", "mission")
    }
    for key in SCENARIO_KEYS:
        state = replayed.get(key)
        if not isinstance(state, Mapping):
            for name in failures:
                failures[name].append(f"{key}: no replay state")
            continue
        name, variant = key
        spec = SCENARIOS[name]
        positions = state.get("positions", ())
        if (
            len(positions) != HORIZON + 1
            or len(state.get("drifts", ())) != HORIZON + 1
        ):
            failures["motion"].append(f"{key}: boundary history length is not {HORIZON + 1}")
        expected_safety = 0
        expected_motion_cost = 0.0
        actions = _actions_by_slot(record_map.get(key, {}))
        for boundary, values in enumerate(positions):
            if len(values) != N_ROBOTS:
                failures["motion"].append(f"{key}: positions[{boundary}] has wrong robot count")
                continue
            if len(set(values)) != N_ROBOTS:
                failures["motion"].append(f"{key}: positions[{boundary}] are not distinct")
            if any(not _traversable(spec, tuple(point)) for point in values):
                failures["motion"].append(f"{key}: positions[{boundary}] contains blocked cell")
        for slot in range(HORIZON):
            start = positions[slot] if slot < len(positions) else ()
            observed_next = positions[slot + 1] if slot + 1 < len(positions) else ()
            accepted: set[tuple[int, int]] = set()
            for robot in range(N_ROBOTS):
                action = actions.get((slot, robot), {})
                move = action.get("move")
                current = tuple(start[robot]) if len(start) == N_ROBOTS else (0, 0)
                proposed = _move_point(current, move) if move in {"N", "E", "S", "W"} else current
                unsafe = move != "stay" and (
                    not _traversable(spec, proposed)
                    or proposed in accepted
                    or proposed in {tuple(start[j]) for j in range(robot + 1, N_ROBOTS)}
                )
                expected = current if unsafe else proposed
                if unsafe and move in {"N", "E", "S", "W"}:
                    expected_safety += 1
                accepted.add(expected)
                expected_motion_cost += 1.0 if expected != current else 0.2
                if len(observed_next) == N_ROBOTS and tuple(observed_next[robot]) != expected:
                    failures["motion"].append(
                        f"{key}: transition slot={slot},robot={robot} disagrees with fixed safety execution"
                    )
        if not _finite(state.get("motion_cost")) or not math.isclose(
            state["motion_cost"], expected_motion_cost, rel_tol=0.0, abs_tol=EPS
        ):
            failures["motion"].append(f"{key}: motion_cost disagrees with replayed transitions")
        if state.get("safety_events") != expected_safety:
            failures["motion"].append(f"{key}: safety event count disagrees with replayed transitions")
        if any(not _finite(value) for values in state.get("energies", ()) for value in values):
            failures["motion"].append(f"{key}: non-finite energy")
        if any(
            value < -EPS or value > 100.0 + EPS
            for values in state.get("energies", ())
            for value in values
            if _finite(value)
        ):
            failures["motion"].append(f"{key}: energy outside the fixed range")
        metadata_requests = state.get("metadata_requests", -1)
        metadata_successes = state.get("metadata_successes", -1)
        if (
            not _integer(metadata_requests)
            or not _integer(metadata_successes)
            or metadata_requests < 0
            or metadata_successes < 0
            or metadata_successes > metadata_requests
        ):
            failures["observations"].append(f"{key}: metadata request/success counts are inconsistent")
        candidate_records = state.get("candidate_records", ())
        generated_ids = [record.get("edge_id") for record in candidate_records if isinstance(record, Mapping)]
        if (
            not isinstance(candidate_records, list)
            or state.get("total_candidates") != len(candidate_records)
            or len(generated_ids) != len(set(generated_ids))
        ):
            failures["candidates"].append(f"{key}: candidate generation records are inconsistent")
        for candidate in candidate_records if isinstance(candidate_records, list) else ():
            if (
                not isinstance(candidate, Mapping)
                or not isinstance(candidate.get("edge_id"), str)
                or not _integer(candidate.get("generation"))
                or not _integer(candidate.get("expiry"))
                or candidate["expiry"] != candidate["generation"] + 10
                or not 0 <= candidate["generation"] <= HORIZON
            ):
                failures["candidates"].append(f"{key}: candidate expiry or ID relation is invalid")
        verification_requests = state.get("verification_requests", -1)
        verification_successes = state.get("verification_successes", -1)
        verification_completions = state.get("verification_completions", -1)
        if (
            not all(_integer(value) for value in (verification_requests, verification_successes, verification_completions))
            or verification_requests < 0
            or verification_successes < 0
            or verification_completions < 0
            or verification_successes > verification_requests
            or verification_completions > verification_successes
        ):
            failures["communication"].append(f"{key}: verification packet lifecycle is inconsistent")
        if (
            state.get("accepted_units", -1) < 0
            or state.get("accepted_units", TOTAL_BUDGET + 1) > TOTAL_BUDGET
        ):
            failures["communication"].append(f"{key}: accepted units exceed budget")
        if not _finite(state.get("validation_wait_slots")) or state["validation_wait_slots"] < 0:
            failures["communication"].append(f"{key}: validation wait is invalid")
        if (
            not _integer(state.get("dropped_candidates"))
            or state["dropped_candidates"] < 0
            or state["dropped_candidates"] > max(0, state.get("total_candidates", 0))
        ):
            failures["candidates"].append(f"{key}: dropped candidate count is invalid")
        if (
            state.get("safety_events", -1) < 0
            or state.get("safety_events", HORIZON * N_ROBOTS + 1) > HORIZON * N_ROBOTS
        ):
            failures["mission"].append(f"{key}: invalid safety event count")
        for vector in (item for boundary in state.get("drifts", ()) for item in boundary):
            if (
                not isinstance(vector, (list, tuple))
                or len(vector) != 2
                or any(not _finite(value) for value in vector)
            ):
                failures["observations"].append(f"{key}: invalid drift vector")
        if any(
            not _finite(edge.get("residual")) or edge.get("residual") < 0
            for edge in state.get("graph_edges", ())
        ):
            failures["mission"].append(f"{key}: invalid graph residual")
        arrivals = state.get("goal_arrivals", ())
        if not isinstance(arrivals, list) or len(arrivals) != N_ROBOTS or any(
            arrival is not None and (not _integer(arrival) or not 0 <= arrival <= HORIZON)
            for arrival in arrivals
        ):
            failures["mission"].append(f"{key}: goal arrival boundaries are invalid")
    return [
        _atom(
            "replay_validity",
            name,
            not failures[name],
            "passed" if not failures[name] else " | ".join(failures[name][:8]),
        )
        for name in failures
    ]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name, variant in SCENARIO_KEYS:
        spec = SCENARIOS[name]
        state = replayed[(name, variant)]
        navigation = 0.0
        for robot in range(N_ROBOTS):
            arrival = state["goal_arrivals"][robot]
            if arrival is None:
                final_distance = _shortest_distance(
                    spec, tuple(state["positions"][-1][robot]), tuple(spec["goals"][robot])
                )
                start_distance = _shortest_distance(
                    spec, tuple(spec["starts"][robot]), tuple(spec["goals"][robot])
                )
                quality = 0.5 * _clip(1.0 - final_distance / max(1, start_distance))
            elif arrival <= spec["deadline"]:
                quality = 1.0
            else:
                quality = 0.5 * _clip(
                    1.0 - (arrival - spec["deadline"]) / max(1, HORIZON - spec["deadline"])
                )
            navigation += quality / N_ROBOTS
        totals["primary_metrics_1"] += _clip(navigation)
        drift_vectors = [vector for boundary in state["drifts"] for vector in boundary]
        pose_mse = sum(dx * dx + dy * dy for dx, dy in drift_vectors) / max(1, len(drift_vectors))
        edges = state["graph_edges"]
        graph_residual = (
            sum(edge["residual"] for edge in edges) / len(edges) if edges else GRAPH_RESIDUAL_REF
        )
        totals["primary_metrics_2"] += _clip(
            0.6 * _clip(1.0 - pose_mse / POSE_MSE_REF)
            + 0.4 * _clip(1.0 - graph_residual / GRAPH_RESIDUAL_REF)
        )
        totals["primary_metrics_3"] += _clip(
            0.65 * _clip(1.0 - state["safety_events"] / (HORIZON * N_ROBOTS))
            + 0.35 * _clip(1.0 - state["motion_cost"] / (HORIZON * N_ROBOTS))
        )
        totals["auxiliary_metrics_1"] += _clip(
            0.5 * _clip(1.0 - state["accepted_units"] / TOTAL_BUDGET)
            + 0.5 * _clip(state["informative_units"] / max(1, state["accepted_units"]))
        )
        totals["auxiliary_metrics_2"] += _clip(
            0.5 * _clip(1.0 - state["validation_wait_slots"] / HORIZON)
            + 0.5 * _clip(1.0 - state["dropped_candidates"] / max(1, state["total_candidates"]))
        )
    return {key: _clip(value / len(SCENARIO_KEYS)) for key, value in totals.items()}


def _gated_metrics(
    replayed: Mapping[tuple[str, int], Any], validity_gate: bool
) -> dict[str, float]:
    if not validity_gate:
        return {key: 0.0 for key in METRIC_KEYS}
    return _metrics(replayed)


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared fixed set")
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
    return "failed_atoms=[{}]; {}".format(
        ",".join(atom.id for atom in failed),
        " | ".join(f"{atom.id}: {atom.detail}" for atom in failed),
    )


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular file"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for index, line in enumerate(lines):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_char, fence_len = True, marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
            continue
        if in_fence:
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$", line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))
            continue
    found: set[str] = set()
    for index, (line_no, level, title) in enumerate(headings):
        if title not in required:
            continue
        end = len(lines)
        for later_line, later_level, _ in headings[index + 1 :]:
            if later_level <= level:
                end = later_line
                break
        if any(line.strip() for line in lines[line_no + 1 : end]):
            found.add(title)
    score = max(0.0, 2.0 - 0.5 * (len(required) - len(found))) / 2.0
    return score, f"missing_sections={sorted(required - found)}; normalized={score:.9g}"


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
            state.timed_out = state.exit_code is None and "timeout" in output
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
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
        state.official_message = (
            (f"{state.official_message}; " if state.official_message else "")
            + f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        )
        parse_failure = ""
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                parse_failure = f"strict parse failed: {type(exc).__name__}: {exc}"
        shape = _shape_atoms(state.data, state.parse_ok)
        evidence = _evidence_atoms(state.data)
        state.scenarios = _index(state.data)
        coverage_score, coverage_defects = _coverage(state.data)
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = [
                _atom(
                    "decision_legality",
                    name,
                    False,
                    f"legality replay raised {type(exc).__name__}: {exc}",
                )
                for name in (
                    "action_values",
                    "visible_edges",
                    "request_history",
                    "energy",
                    "replayable_events",
                )
            ]
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
        if can_replay:
            state.replayed = {
                key_: _replay(key_[0], key_[1], state.scenarios[key_]) for key_ in SCENARIO_KEYS
            }
        replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["result_json"] = (
            f"{parse_failure}; " if parse_failure else ""
        ) + _diagnose(shape)
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
        state.metrics = _gated_metrics(state.replayed, state.validity_gate)
        if not state.official_ok:
            for check_id in GATE_CHECKS:
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = (
                    f"official_run failed: {state.official_message}; {state.diagnostics.get(check_id, '')}"
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


for _id in GATE_CHECKS:

    async def _check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    if _id != "official_run":
        globals()[_id] = check(id=_id)(_check)


for _metric in METRIC_KEYS:

    async def _metric_check(ws: Workspace, ident: str = _metric) -> tuple[float, str]:
        state = await _build_state(ws)
        value = state.metrics.get(ident, 0.0) if state.validity_gate and state.metrics else 0.0
        return value, f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}"

    globals()[_metric] = check(id=_metric)(_metric_check)


del _check, _metric_check
