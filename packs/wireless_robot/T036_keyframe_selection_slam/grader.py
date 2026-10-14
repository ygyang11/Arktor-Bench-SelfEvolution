from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

GRID_X = 22
GRID_Y = 18
HORIZON = 72
N_ROBOTS = 3
ROBOTS = tuple(f"r{i}" for i in range(N_ROBOTS))
MOVES = ("stay", "N", "E", "S", "W")
MOVE_DELTAS = {"stay": (0, 0), "N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
ENCODINGS = {
    "drop": (0, 0.0, 0.0),
    "sparse": (1, 0.50, 0.18),
    "balanced": (2, 0.75, 0.30),
    "full": (4, 1.00, 0.46),
}
LOCAL_QUEUE_CAPACITY = 8
EDGE_QUEUE_CAPACITY = 6
SLOT_CAPACITY = 8
SERVICE_RATE = 2
MAX_AGE = 12
ENERGY0 = 100.0
POSE_ERROR_REF = 22.0
MAP_RESIDUAL_REF = 16.0
GRID_DIAGONAL = math.hypot(GRID_X - 1, GRID_Y - 1)
EDGE_POSITION = (11, 9)
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
        "result_evidence.action_rows": 0.25,
        "result_evidence.action_values": 0.30,
        "result_evidence.request_values": 0.25,
    },
    "decision_legality": {
        "decision_legality.motion": 0.20,
        "decision_legality.encoding": 0.15,
        "decision_legality.requests": 0.25,
        "decision_legality.admission": 0.15,
        "decision_legality.energy": 0.15,
        "decision_legality.retry": 0.10,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.observations": 0.20,
        "replay_validity.transmissions": 0.20,
        "replay_validity.fusion": 0.20,
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
    goals: tuple[tuple[int, int], ...],
    obstacles: tuple[tuple[int, int, int, int], ...],
    landmarks: tuple[tuple[int, int], ...],
    deadline: int,
    drift: float,
    noise: tuple[float, ...] = (1.0, 1.0, 1.0),
    burst: bool = False,
) -> dict[str, Any]:
    return {
        "starts": starts,
        "goals": goals,
        "obstacles": obstacles,
        "landmarks": landmarks,
        "deadline": deadline,
        "drift": drift,
        "noise": noise,
        "burst": burst,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "loop_rich": _scenario(
        ((1, 2), (1, 15), (20, 8)),
        ((19, 15), (19, 2), (11, 9)),
        ((7, 8, 0, 6), (7, 8, 11, 17)),
        ((4, 3), (4, 14), (11, 4), (11, 14), (18, 9)),
        56,
        0.018,
    ),
    "drift_corridor": _scenario(
        ((2, 2), (2, 15), (19, 2)),
        ((19, 15), (11, 9), (19, 8)),
        ((9, 12, 6, 7), (9, 12, 10, 11)),
        ((5, 4), (5, 13), (15, 4), (15, 13)),
        64,
        0.026,
    ),
    "bandwidth_burst": _scenario(
        ((1, 1), (1, 16), (20, 1)),
        ((20, 16), (11, 9), (20, 8)),
        ((6, 7, 5, 12), (15, 16, 5, 12)),
        ((4, 2), (4, 16), (11, 3), (11, 15), (19, 9)),
        60,
        0.020,
        burst=True,
    ),
    "heterogeneous_observation": _scenario(
        ((2, 8), (10, 2), (19, 8)),
        ((2, 15), (11, 9), (19, 2)),
        ((8, 9, 3, 14),),
        ((4, 8), (11, 2), (11, 16), (18, 8)),
        68,
        0.022,
        noise=(1.0, 1.5, 0.7),
    ),
    "crossing_routes": _scenario(
        ((1, 3), (1, 14), (20, 3)),
        ((20, 14), (11, 9), (20, 8)),
        ((10, 11, 0, 5), (10, 11, 12, 17)),
        ((4, 4), (4, 13), (17, 4), (17, 13)),
        62,
        0.024,
    ),
}
SCENARIO_NAMES = tuple(SCENARIOS)
SCENARIO_KEYS = tuple(product(SCENARIO_NAMES, range(3)))


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
        self.scenarios: dict[tuple[str, int], Any] = {}
        self.replayed: dict[tuple[str, int], Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_DISTANCE_CACHE: dict[str, dict[tuple[int, int], int]] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _brief(value: Any, limit: int = 180) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _blocked(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    x, y = position
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in spec["obstacles"])


def _traversable(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    return 0 <= position[0] < GRID_X and 0 <= position[1] < GRID_Y and not _blocked(spec, position)


def _distance_map(name: str) -> dict[tuple[int, int], int]:
    if name in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[name]
    spec = SCENARIOS[name]
    distances = {tuple(spec["goals"][0]): 0}
    # The goal-specific maps used for metrics are built separately; this map is
    # retained only for deterministic nearest-cell policy helpers.
    target = tuple(spec["goals"][0])
    queue: deque[tuple[int, int]] = deque([target])
    while queue:
        current = queue.popleft()
        for delta in MOVE_DELTAS.values():
            candidate = current[0] + delta[0], current[1] + delta[1]
            if candidate in distances or not _traversable(spec, candidate):
                continue
            distances[candidate] = distances[current] + 1
            queue.append(candidate)
    _DISTANCE_CACHE[name] = distances
    return distances


def _shortest_distance(
    spec: Mapping[str, Any], start: tuple[int, int], target: tuple[int, int]
) -> int:
    if start == target:
        return 0
    queue: deque[tuple[int, int]] = deque([start])
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        for delta in MOVE_DELTAS.values():
            candidate = current[0] + delta[0], current[1] + delta[1]
            if candidate in distances or not _traversable(spec, candidate):
                continue
            distances[candidate] = distances[current] + 1
            if candidate == target:
                return distances[candidate]
            queue.append(candidate)
    return GRID_X * GRID_Y


def _hash_unit(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


def _drift_delta(name: str, variant: int, robot: int, slot: int, axis: str) -> float:
    base = SCENARIOS[name]["drift"] * (1.0 + 0.2 * variant)
    sample = _hash_unit(f"{name}|{variant}|r{robot}|t{slot}|drift|{axis}")
    return base * (2.0 * sample - 1.0)


def _visible_observations(
    name: str,
    variant: int,
    robot: int,
    slot: int,
    position: tuple[int, int],
    drift: tuple[float, float],
) -> dict[int, dict[str, Any]]:
    spec = SCENARIOS[name]
    multiplier = spec["noise"][robot]
    observations: dict[int, dict[str, Any]] = {}
    sigma = multiplier * (0.10 + 0.08 * variant)
    for landmark_id, landmark in enumerate(spec["landmarks"]):
        distance = math.dist(position, landmark)
        if distance > 3.0 + EPS:
            continue
        normalized = min(1.0, distance / 3.0)
        quality = _clip(1.0 - 0.12 * sigma - 0.04 * normalized)
        ux = _hash_unit(f"{name}|{variant}|r{robot}|t{slot}|l{landmark_id}|ax|obs")
        uy = _hash_unit(f"{name}|{variant}|r{robot}|t{slot}|l{landmark_id}|ay|obs")
        noise_scale = 0.25 * sigma * (1.0 + normalized)
        observed = (
            landmark[0] + drift[0] + noise_scale * (2.0 * ux - 1.0),
            landmark[1] + drift[1] + noise_scale * (2.0 * uy - 1.0),
        )
        observations[landmark_id] = {
            "quality": quality,
            "sigma": sigma,
            "distance": normalized,
            "observed": observed,
        }
    return observations


def _retain_observations(
    observations: Mapping[int, Mapping[str, Any]], retained_fraction: float
) -> dict[int, Mapping[str, Any]]:
    if not observations:
        return {}
    keep = max(1, math.ceil(len(observations) * retained_fraction))
    ordered = sorted(
        observations.items(),
        key=lambda item: (-float(item[1]["quality"]), item[0]),
    )
    return dict(ordered[:keep])


def _frame_id(name: str, variant: int, robot: int, slot: int) -> str:
    return f"scenario={name}|variant={variant}|robot={robot}|slot={slot}"


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
            "expected scenario records with exact keys scenario, variant, actions",
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
            for action in actions:
                if not isinstance(action, Mapping) or set(action) != {
                    "slot",
                    "robot",
                    "move",
                    "encode",
                    "send_keyframe",
                }:
                    action_rows_ok = False
                    continue
                if not _integer(action.get("slot")) or not isinstance(action.get("robot"), str):
                    action_values_ok = False
                if not isinstance(action.get("move"), str) or not isinstance(
                    action.get("encode"), str
                ):
                    action_values_ok = False
                send = action.get("send_keyframe")
                if not isinstance(send, (str, type(None))):
                    request_values_ok = False
    return [
        _atom("result_evidence", "scenario_rows", scenario_ok, "scenario and variant types"),
        _atom("result_evidence", "action_rows", action_rows_ok, "action objects have exact keys"),
        _atom(
            "result_evidence",
            "action_values",
            action_values_ok,
            "slot, robot, move and encode types",
        ),
        _atom(
            "result_evidence", "request_values", request_values_ok, "send_keyframe string or null"
        ),
    ]


def _index_records(data: Any) -> dict[tuple[str, int], Any]:
    """Index canonical scenario rows without scoring or repairing them."""
    out: dict[tuple[str, int], Any] = {}
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name, variant = row.get("scenario"), row.get("variant")
        if isinstance(name, str) and _integer(variant):
            out.setdefault((name, variant), row)
    return out


# Historical private spelling retained for compatibility with older probes.
def _index(data: Any) -> dict[tuple[str, int], Any]:
    return _index_records(data)


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    expected = set(SCENARIO_KEYS)
    counts: dict[tuple[str, int], int] = {}
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                defects.append(f"unknown scenario row={_brief(row)}")
                continue
            name, variant = row.get("scenario"), row.get("variant")
            if not isinstance(name, str) or not _integer(variant):
                defects.append(f"unknown scenario key={_brief((name, variant))}")
                continue
            key = (name, variant)
            counts[key] = counts.get(key, 0) + 1
            if key not in expected:
                defects.append(f"unknown scenario key={key!r}")
    for key in sorted(expected - set(counts)):
        defects.append(f"missing scenario key={key!r}")
    for key, count in sorted(counts.items(), key=repr):
        if count > 1:
            defects.append(f"duplicate scenario key={key!r} count={count}")
    indexed = _index_records(data)
    expected_actions = _expected_action_keys()
    for key in sorted(expected & set(indexed)):
        actions = indexed[key].get("actions")
        action_counts: dict[tuple[Any, Any], int] = {}
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, Mapping):
                    defects.append(f"{key} non-mapping action={_brief(action)}")
                    continue
                action_key = (action.get("slot"), action.get("robot"))
                try:
                    hash(action_key)
                except TypeError:
                    defects.append(f"{key} unhashable action key={_brief(action_key)}")
                    continue
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
        else:
            defects.append(f"{key} actions is not a list")
        for action_key in sorted(expected_actions - set(action_counts), key=repr):
            defects.append(f"{key} missing action key={action_key!r}")
        for action_key, count in sorted(action_counts.items(), key=repr):
            if action_key not in expected_actions:
                defects.append(f"{key} unknown action key={action_key!r}")
            elif count > 1:
                defects.append(f"{key} duplicate action key={action_key!r} count={count}")
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _action_map(record: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    actions = record.get("actions") if isinstance(record, Mapping) else None
    if not isinstance(actions, list):
        return {}
    out: dict[tuple[int, str], Mapping[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        slot, robot = action.get("slot"), action.get("robot")
        if _integer(slot) and isinstance(robot, str):
            out[(slot, robot)] = action
    return out


def _link_sample(
    name: str,
    variant: int,
    slot: int,
    robot: int,
    generation: int,
    attempt: int,
    first: tuple[int, int],
    concurrent: int,
) -> tuple[float, float, bool, str]:
    distance = min(1.0, math.dist(first, EDGE_POSITION) / GRID_DIAGONAL)
    burst = 1 if SCENARIOS[name]["burst"] and 24 <= slot <= 35 else 0
    probability = _clip(0.93 - 0.05 * distance - 0.06 * burst - 0.03 * (concurrent - 1))
    tx_id = (
        f"scenario={name}|variant={variant}|robot={robot}|"
        f"slot={generation}|attempt={attempt}"
    )
    sample = _hash_unit(f"{tx_id}|salt={13 * variant}")
    return probability, sample, sample < probability, tx_id


def _replay(
    name: str, variant: int, record: Mapping[str, Any], stop: int | None = None
) -> dict[str, Any]:
    spec = SCENARIOS[name]
    actions = _action_map(record)
    limit = HORIZON if stop is None else max(0, min(HORIZON, stop))
    positions = [tuple(value) for value in spec["starts"]]
    drifts = [(0.0, 0.0) for _ in ROBOTS]
    energies = [ENERGY0 for _ in ROBOTS]
    local_queues: list[list[str]] = [[] for _ in ROBOTS]
    frames: dict[str, dict[str, Any]] = {}
    pending_arrivals: dict[int, list[str]] = {}
    edge_queue: list[str] = []
    active_id: str | None = None
    active_remaining = 0
    mapped_landmarks: set[int] = set()
    best_residual: dict[int, float] = {}
    graph_edges: list[dict[str, Any]] = []
    transmissions: list[dict[str, Any]] = []
    edge_fusion: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = {
        key: [] for key in ("motion", "encoding", "requests", "admission", "energy", "retry")
    }
    positions_history = [tuple(positions)]
    drifts_history = [tuple(drifts)]
    energies_history = [tuple(energies)]
    queue_history: list[tuple[tuple[str, ...], ...]] = [
        tuple(tuple(queue) for queue in local_queues)
    ]
    edge_history: list[tuple[str | None, int, tuple[str, ...]]] = [(None, 0, tuple())]
    goal_arrivals: list[int | None] = [
        0 if positions[i] == spec["goals"][i] else None for i in range(N_ROBOTS)
    ]
    safety_events = 0
    motion_cost = 0.0
    accepted_payload = 0
    informative_payload = 0
    dropped_frames = 0
    total_retained = 0
    edge_busy_slots = 0
    busy_flags: list[bool] = []

    def remove_local(robot: int, frame_id: str) -> None:
        local_queues[robot] = [ident for ident in local_queues[robot] if ident != frame_id]

    def drop_frame(frame_id: str, reason: str) -> None:
        nonlocal dropped_frames
        frame = frames.get(frame_id)
        if frame is None or frame["status"] in {"dropped", "fused"}:
            return
        remove_local(frame["robot"], frame_id)
        frame["status"] = "dropped"
        frame["drop_reason"] = reason
        dropped_frames += 1

    def fuse(frame_id: str, boundary: int) -> None:
        nonlocal active_id, active_remaining, informative_payload
        frame = frames[frame_id]
        improved = False
        for landmark_id in sorted(frame["observations"]):
            observed = frame["observations"][landmark_id]["observed"]
            landmark = spec["landmarks"][landmark_id]
            raw_residual = math.dist(observed, landmark)
            residual = raw_residual * (0.85 if landmark_id in mapped_landmarks else 1.0)
            if residual < best_residual.get(landmark_id, math.inf) - EPS:
                improved = True
                best_residual[landmark_id] = residual
            mapped_landmarks.add(landmark_id)
            graph_edges.append(
                {"frame": frame_id, "landmark": landmark_id, "residual": residual, "valid": True}
            )
        if improved:
            informative_payload += frame["payload_units"]
        gain = frame["correction_gain"]
        robot = frame["robot"]
        drifts[robot] = (drifts[robot][0] * (1.0 - gain), drifts[robot][1] * (1.0 - gain))
        frame["status"] = "fused"
        frame["fusion_boundary"] = boundary
        edge_fusion.append(
            {
                "frame": frame_id,
                "boundary": boundary,
                "robot": robot,
                "payload_units": frame["payload_units"],
            }
        )
        active_id = None
        active_remaining = 0

    def advance(boundary: int) -> None:
        nonlocal active_id, active_remaining
        for frame_id in pending_arrivals.pop(boundary, []):
            frame = frames.get(frame_id)
            if frame is None or frame["status"] != "pending_arrival":
                continue
            if len(edge_queue) >= EDGE_QUEUE_CAPACITY:
                drop_frame(frame_id, "edge_queue")
            else:
                frame["status"] = "edge_queue"
                edge_queue.append(frame_id)
        if active_id is None and edge_queue:
            active_id = edge_queue.pop(0)
            active_remaining = frames[active_id]["payload_units"]
        if active_id is not None:
            active_remaining -= SERVICE_RATE
            if active_remaining <= 0:
                fuse(active_id, boundary)

    def expire(boundary: int, terminal: bool = False) -> None:
        for frame_id, frame in list(frames.items()):
            if frame["status"] not in {"queued", "retry"}:
                continue
            age = boundary - frame["generation"]
            if age > MAX_AGE:
                drop_frame(frame_id, "age")
            elif frame.get("retry_due") is not None and (
                frame["retry_due"] < boundary or (terminal and frame["retry_due"] == boundary)
            ):
                drop_frame(frame_id, "retry_window")

    def record_boundary() -> None:
        positions_history.append(tuple(positions))
        drifts_history.append(tuple(drifts))
        energies_history.append(tuple(energies))
        queue_history.append(tuple(tuple(queue) for queue in local_queues))
        edge_history.append((active_id, active_remaining, tuple(edge_queue)))

    for slot in range(limit):
        if slot > 0:
            advance(slot)
            expire(slot)
            record_boundary()
        busy = active_id is not None or bool(edge_queue)
        busy_flags.append(busy)
        if busy:
            edge_busy_slots += 1
        rows = {robot: actions.get((slot, robot), {}) for robot in ROBOTS}
        starts = tuple(positions)
        request_rows: list[dict[str, Any]] = []
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            move = row.get("move")
            if not isinstance(move, str) or move not in MOVES:
                errors["motion"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].move={move!r}"
                )
            encoding = row.get("encode")
            if not isinstance(encoding, str) or encoding not in ENCODINGS:
                errors["encoding"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].encode={encoding!r}"
                )
            send = row.get("send_keyframe")
            if send is None:
                continue
            frame = frames.get(send) if isinstance(send, str) else None
            valid = (
                frame is not None
                and frame["robot"] == robot_index
                and frame["status"] in {"queued", "retry"}
                and send in local_queues[robot_index]
            )
            if not valid:
                errors["requests"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}].send_keyframe={send!r}; "
                    "expected action-start local-queue frame"
                )
                continue
            retry = frame.get("retry_due") is not None
            if retry and frame.get("retry_due") != slot:
                errors["retry"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}] "
                    f"retry_due={frame.get('retry_due')}; expected {slot}"
                )
                continue
            request_rows.append(
                {
                    "robot": robot_index,
                    "frame": send,
                    "cost": frame["payload_units"],
                    "priority": frame["priority"],
                    "generation": frame["generation"],
                    "attempt": 1 if retry else 0,
                }
            )
        request_rows.sort(
            key=lambda request: (
                request["generation"],
                -request["priority"],
                request["robot"],
                request["frame"],
                request["attempt"],
            )
        )
        capacity = 4 if spec["burst"] and 24 <= slot <= 35 else SLOT_CAPACITY
        remaining = capacity
        admitted: list[dict[str, Any]] = []
        for request in request_rows:
            if request["cost"] <= remaining:
                admitted.append(request)
                remaining -= request["cost"]
        accepted_payload += sum(request["cost"] for request in admitted)
        admitted_ids = {request["frame"] for request in admitted}

        # Admission is an environment outcome rather than a candidate error
        # when a request does not fit.  Still verify the fixed admission
        # ledger here so the dedicated atom is not vacuous: every complete
        # request must be considered once, admitted requests must match the
        # deterministic greedy capacity pass, and no payload may be split or
        # exceed the slot capacity.
        admission_remaining = capacity
        admission_seen: set[str] = set()
        for request in request_rows:
            frame_id = request.get("frame")
            cost = request.get("cost")
            if (
                not isinstance(frame_id, str)
                or not isinstance(cost, int)
                or isinstance(cost, bool)
                or cost not in {1, 2, 4}
            ):
                errors["admission"].append(
                    f"{name}|v{variant}.actions[slot={slot}] admission request has invalid "
                    f"frame/cost={frame_id!r}/{cost!r}"
                )
                continue
            if frame_id in admission_seen:
                errors["admission"].append(
                    f"{name}|v{variant}.actions[slot={slot}] duplicate admission request "
                    f"frame={frame_id!r}"
                )
            admission_seen.add(frame_id)
            if frame_id in admitted_ids:
                if cost > admission_remaining:
                    errors["admission"].append(
                        f"{name}|v{variant}.actions[slot={slot}] admitted frame={frame_id!r} "
                        f"cost={cost} exceeds remaining capacity={admission_remaining}"
                    )
                else:
                    admission_remaining -= cost
            elif cost <= admission_remaining:
                errors["admission"].append(
                    f"{name}|v{variant}.actions[slot={slot}] fitting frame={frame_id!r} "
                    "was omitted from deterministic admission"
                )
        admitted_cost = sum(request["cost"] for request in admitted)
        if admitted_cost != capacity - admission_remaining:
            errors["admission"].append(
                f"{name}|v{variant}.actions[slot={slot}] admission ledger mismatch: "
                f"admitted_cost={admitted_cost}, ledger={capacity - admission_remaining}"
            )
        for request in admitted:
            frame = frames[request["frame"]]
            probability, sample, success, tx_id = _link_sample(
                name,
                variant,
                slot,
                request["robot"],
                request["generation"],
                request["attempt"],
                starts[request["robot"]],
                len(admitted),
            )
            transmissions.append(
                {
                    "tx_id": tx_id,
                    "frame": request["frame"],
                    "slot": slot,
                    "attempt": request["attempt"],
                    "probability": probability,
                    "sample": sample,
                    "success": success,
                    "payload_units": request["cost"],
                }
            )
            if success:
                remove_local(request["robot"], request["frame"])
                frame["status"] = "pending_arrival"
                frame["attempt"] = request["attempt"]
                pending_arrivals.setdefault(slot + 1, []).append(request["frame"])
            elif request["attempt"] == 0:
                frame["status"] = "retry"
                frame["retry_due"] = slot + 1
                frame["attempt"] = 1
            else:
                drop_frame(request["frame"], "retry_failed")
        for frame_id, frame in list(frames.items()):
            if (
                frame["status"] == "retry"
                and frame_id not in admitted_ids
                and frame.get("retry_due") == slot
            ):
                drop_frame(frame_id, "retry_window")

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
                if move in MOVES and move != "stay":
                    safety_events += 1
            accepted_destinations.append(destination)
            next_positions.append(destination)
        positions = next_positions
        for robot_index, robot in enumerate(ROBOTS):
            row = rows[robot]
            moved = positions[robot_index] != starts[robot_index]
            move_cost = 1.0 if moved else 0.2
            upload_cost = sum(
                request["cost"] * 0.05 for request in admitted if request["robot"] == robot_index
            )
            energies[robot_index] -= move_cost + upload_cost
            motion_cost += move_cost
            if energies[robot_index] < -EPS:
                errors["energy"].append(
                    f"{name}|v{variant}.actions[slot={slot},robot={robot}] "
                    f"resulting_energy={energies[robot_index]:.9g}; expected >= {-EPS}"
                )
            row_encoding = row.get("encode")
            if not isinstance(row_encoding, str) or row_encoding not in ENCODINGS:
                row_encoding = "drop"
            payload, retained, gain = ENCODINGS[row_encoding]
            if row_encoding == "drop":
                continue
            frame_id = _frame_id(name, variant, robot_index, slot)
            observations = _visible_observations(
                name, variant, robot_index, slot, positions[robot_index], drifts[robot_index]
            )
            priority = sum(item["quality"] for item in observations.values()) + 0.02 * len(
                observations
            )
            frames[frame_id] = {
                "id": frame_id,
                "robot": robot_index,
                "generation": slot,
                "payload_units": payload,
                "retained_fraction": retained,
                "correction_gain": gain,
                "priority": priority,
                "observations": _retain_observations(observations, retained),
                "status": "queued",
                "attempt": 0,
                "retry_due": None,
            }
            local_queues[robot_index].append(frame_id)
            total_retained += 1
            local_queues[robot_index].sort(
                key=lambda ident: (
                    frames[ident]["generation"] + MAX_AGE,
                    -frames[ident]["priority"],
                    frames[ident]["generation"],
                    ident,
                )
            )
            while len(local_queues[robot_index]) > LOCAL_QUEUE_CAPACITY:
                drop_frame(local_queues[robot_index][-1], "queue_overflow")
        for robot_index in range(N_ROBOTS):
            drifts[robot_index] = (
                drifts[robot_index][0] + _drift_delta(name, variant, robot_index, slot, "x"),
                drifts[robot_index][1] + _drift_delta(name, variant, robot_index, slot, "y"),
            )
        for robot_index in range(N_ROBOTS):
            if (
                goal_arrivals[robot_index] is None
                and positions[robot_index] == spec["goals"][robot_index]
            ):
                goal_arrivals[robot_index] = slot + 1

    advance(limit)
    expire(limit, terminal=True)
    if limit > 0 or stop is None:
        record_boundary()
    return {
        "positions": tuple(positions_history),
        "drifts": tuple(drifts_history),
        "energies": tuple(energies_history),
        "queues": tuple(queue_history),
        "edge_history": tuple(edge_history),
        "frames": frames,
        "transmissions": tuple(transmissions),
        "edge_fusion": tuple(edge_fusion),
        "graph_edges": tuple(graph_edges),
        "goal_arrivals": tuple(goal_arrivals),
        "safety_events": safety_events,
        "motion_cost": motion_cost,
        "accepted_payload": accepted_payload,
        "informative_payload": informative_payload,
        "dropped_frames": dropped_frames,
        "total_retained": total_retained,
        "edge_busy_slots": edge_busy_slots,
        "expected_frames": HORIZON * N_ROBOTS,
        "errors": errors,
    }


def _legality(scenarios: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "encoding", "requests", "admission", "energy", "retry")
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
        _atom("decision_legality", "encoding", not failures["encoding"], detail("encoding")),
        _atom("decision_legality", "requests", not failures["requests"], detail("requests")),
        _atom("decision_legality", "admission", not failures["admission"], detail("admission")),
        _atom("decision_legality", "energy", not failures["energy"], detail("energy")),
        _atom("decision_legality", "retry", not failures["retry"], detail("retry")),
    ]


def _replay_atoms(
    scenarios: Mapping[tuple[str, int], Any], replayed: Mapping[tuple[str, int], Any]
) -> list[AtomVerdict]:
    del scenarios
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "observations", "transmissions", "fusion", "mission")
    }
    if set(replayed) != set(SCENARIO_KEYS):
        failures["mission"].append(
            f"scenario keys={sorted(replayed)}; expected={list(SCENARIO_KEYS)}"
        )
    for key in SCENARIO_KEYS:
        state = replayed.get(key)
        if not isinstance(state, Mapping):
            continue
        name, variant = key
        spec = SCENARIOS[name]
        positions = state.get("positions", ())
        drifts = state.get("drifts", ())
        energies = state.get("energies", ())
        if (
            len(positions) != HORIZON + 1
            or len(drifts) != HORIZON + 1
            or len(energies) != HORIZON + 1
        ):
            failures["motion"].append(
                f"{name}|v{variant} boundary histories must have length {HORIZON + 1}"
            )
        for boundary, values in enumerate(positions):
            if len(values) != N_ROBOTS or len(set(values)) != N_ROBOTS:
                failures["motion"].append(
                    f"{name}|v{variant}.positions[{boundary}]={_brief(values)}; "
                    "expected distinct robot cells"
                )
            elif any(not _traversable(spec, tuple(position)) for position in values):
                failures["motion"].append(
                    f"{name}|v{variant}.positions[{boundary}] contains blocked/out-of-grid cell"
                )
        for boundary, values in enumerate(drifts):
            if len(values) != N_ROBOTS or any(
                len(tuple(value)) != 2 or any(not _finite_number(component) for component in value)
                for value in values
            ):
                failures["motion"].append(f"{name}|v{variant}.drifts[{boundary}] invalid")
        for boundary, values in enumerate(energies):
            if len(values) != N_ROBOTS or any(
                not _finite_number(value) or value < -EPS or value > ENERGY0 + EPS
                for value in values
            ):
                failures["motion"].append(
                    f"{name}|v{variant}.energies[{boundary}]={_brief(values)}; expected [0,100]"
                )
        frames = state.get("frames", {})
        if not isinstance(frames, Mapping):
            failures["observations"].append(f"{name}|v{variant}.frames is not a mapping")
        else:
            for frame_id, frame in frames.items():
                if not isinstance(frame, Mapping) or not re.fullmatch(
                    rf"scenario={re.escape(name)}\|variant={variant}\|robot=[0-2]\|slot=[0-9]+",
                    str(frame_id),
                ):
                    failures["observations"].append(
                        f"{name}|v{variant}.frame={frame_id!r} has invalid ID"
                    )
                    continue
                if frame.get("payload_units") not in {1, 2, 4} or not _finite_number(
                    frame.get("priority")
                ):
                    failures["observations"].append(
                        f"{name}|v{variant}.{frame_id} encoding evidence invalid"
                    )
                if not isinstance(frame.get("observations"), Mapping):
                    failures["observations"].append(
                        f"{name}|v{variant}.{frame_id}.observations invalid"
                    )
                if frame.get("status") not in {
                    "queued",
                    "retry",
                    "pending_arrival",
                    "edge_queue",
                    "fused",
                    "dropped",
                }:
                    failures["observations"].append(f"{name}|v{variant}.{frame_id}.status invalid")
        transmissions = state.get("transmissions", ())
        seen_tx: set[str] = set()
        if not isinstance(transmissions, (list, tuple)):
            failures["transmissions"].append(f"{name}|v{variant}.transmissions is not a sequence")
        else:
            for transmission in transmissions:
                if not isinstance(transmission, Mapping):
                    failures["transmissions"].append(
                        f"{name}|v{variant} transmission is not a mapping"
                    )
                    continue
                tx_id = transmission.get("tx_id")
                if not isinstance(tx_id, str) or tx_id in seen_tx:
                    failures["transmissions"].append(
                        f"{name}|v{variant} duplicate/invalid tx_id={tx_id!r}"
                    )
                if isinstance(tx_id, str):
                    seen_tx.add(tx_id)
                if not _finite_number(transmission.get("probability")) or not _finite_number(
                    transmission.get("sample")
                ):
                    failures["transmissions"].append(
                        f"{name}|v{variant}.{tx_id} probability/sample invalid"
                    )
                if transmission.get("attempt") not in {0, 1}:
                    failures["transmissions"].append(f"{name}|v{variant}.{tx_id}.attempt invalid")
        for edge in state.get("graph_edges", ()):
            if (
                not isinstance(edge, Mapping)
                or not edge.get("valid")
                or not _finite_number(edge.get("residual"))
            ):
                failures["fusion"].append(f"{name}|v{variant} invalid graph edge={_brief(edge)}")
        if not _integer(state.get("safety_events")) or state["safety_events"] not in range(
            HORIZON * N_ROBOTS + 1
        ):
            failures["mission"].append(f"{name}|v{variant}.safety_events invalid")
        if not _finite_number(state.get("motion_cost")) or state["motion_cost"] < 0:
            failures["mission"].append(f"{name}|v{variant}.motion_cost invalid")
        if not _integer(state.get("accepted_payload")) or state["accepted_payload"] not in range(
            HORIZON * SLOT_CAPACITY + 1
        ):
            failures["mission"].append(
                f"{name}|v{variant}.accepted_payload={state.get('accepted_payload')!r}; "
                f"expected 0..{HORIZON * SLOT_CAPACITY}"
            )
        if len(state.get("goal_arrivals", ())) != N_ROBOTS:
            failures["mission"].append(f"{name}|v{variant}.goal_arrivals invalid")
    try:
        metrics = _metrics(replayed)
        if set(metrics) != set(METRIC_KEYS) or any(
            not _finite_number(value) or not 0 <= value <= 1 for value in metrics.values()
        ):
            failures["mission"].append(
                f"metrics={_brief(metrics)}; expected finite values in [0,1]"
            )
    except Exception as exc:
        failures["mission"].append(f"metrics raised {type(exc).__name__}: {exc}")

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("replay_validity", "motion", not failures["motion"], detail("motion")),
        _atom(
            "replay_validity", "observations", not failures["observations"], detail("observations")
        ),
        _atom(
            "replay_validity",
            "transmissions",
            not failures["transmissions"],
            detail("transmissions"),
        ),
        _atom("replay_validity", "fusion", not failures["fusion"], detail("fusion")),
        _atom("replay_validity", "mission", not failures["mission"], detail("mission")),
    ]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name, variant in SCENARIO_KEYS:
        state = replayed[(name, variant)]
        spec = SCENARIOS[name]
        arrival_score = 0.0
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
            arrival_score += quality / N_ROBOTS
        totals["primary_metrics_1"] += _clip(arrival_score)
        drift_vectors = [vector for boundary in state["drifts"] for vector in boundary]
        rmse = math.sqrt(sum(dx * dx + dy * dy for dx, dy in drift_vectors) / len(drift_vectors))
        edges = state["graph_edges"]
        residual = (
            sum(edge["residual"] for edge in edges) / len(edges) if edges else MAP_RESIDUAL_REF
        )
        totals["primary_metrics_2"] += _clip(
            0.6 * _clip(1.0 - rmse / POSE_ERROR_REF)
            + 0.4 * _clip(1.0 - residual / MAP_RESIDUAL_REF)
        )
        totals["primary_metrics_3"] += _clip(
            0.6 * _clip(1.0 - state["safety_events"] / (HORIZON * N_ROBOTS))
            + 0.4 * _clip(1.0 - state["motion_cost"] / (HORIZON * N_ROBOTS))
        )
        payload = state["accepted_payload"]
        totals["auxiliary_metrics_1"] += _clip(
            0.5 * _clip(1.0 - payload / 864.0)
            + 0.5 * _clip(state["informative_payload"] / max(1, payload))
        )
        totals["auxiliary_metrics_2"] += _clip(
            0.5 * _clip(1.0 - state["edge_busy_slots"] / HORIZON)
            + 0.5 * _clip(1.0 - state["dropped_frames"] / max(1, state["total_retained"]))
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
    previous_visible = False
    for index, line in enumerate(lines):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_char, fence_len = True, marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
            previous_visible = False
            continue
        if in_fence:
            previous_visible = False
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$", line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))
            previous_visible = False
            continue
        setext = re.match(r"^ {0,3}(=+|-+)[ \t]*$", line)
        if setext and previous_visible and index > 0:
            headings.append(
                (index - 1, 1 if line.strip()[0] == "=" else 2, lines[index - 1].strip())
            )
            previous_visible = False
            continue
        previous_visible = bool(line.strip()) and len(line) - len(line.lstrip(" ")) <= 3
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
        state.official_message = (
            (f"{state.official_message}; " if state.official_message else "")
            + f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
            f"timed_out={state.timed_out}, "
            f"new_regular_results={state.result_file_is_new_regular}"
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
        state.scenarios = _index_records(state.data)
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
                for name in ("motion", "encoding", "requests", "admission", "energy", "retry")
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
        state.replayed = (
            {key_: _replay(key_[0], key_[1], state.scenarios[key_]) for key_ in SCENARIO_KEYS}
            if can_replay
            else {}
        )
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
