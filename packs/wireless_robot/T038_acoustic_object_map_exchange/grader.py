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
HORIZON = 75
N_ROBOTS = 3
GRID_X = 24
GRID_Y = 18
ENERGY0 = 60.0
CHANNEL_CAPACITY = 12
QUEUE_CAPACITY = 10
SUMMARY_UNITS = 1
VALIDATION_UNITS = 6
TOTAL_BUDGET = 600
SERVICE_UNITS = 2
POSE_MSE_REF = 24.0**2
GRAPH_RESIDUAL_REF = 16.0
SCENARIO_NAMES = (
    "repeated_pylons",
    "sparse_landmarks",
    "depth_link_shift",
    "crossing_surveys",
    "acoustic_burst",
)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
ROBOTS = tuple(f"r{i}" for i in range(N_ROBOTS))
MOVES = ("stay", "N", "E", "S", "W")
MOVE_DELTAS = {"stay": (0, 0), "N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
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
        "decision_legality.action_values": 0.22,
        "decision_legality.visible_edges": 0.22,
        "decision_legality.retry_feedback": 0.18,
        "decision_legality.communication": 0.20,
        "decision_legality.energy": 0.08,
        "decision_legality.acceptance": 0.10,
    },
    "replay_validity": {
        "replay_validity.motion": 0.18,
        "replay_validity.observations": 0.18,
        "replay_validity.candidates": 0.18,
        "replay_validity.communication": 0.20,
        "replay_validity.mission": 0.16,
        "replay_validity.feedback": 0.10,
    },
}


def _scene(
    starts: tuple[tuple[int, int], ...],
    station: tuple[int, int],
    objects: tuple[tuple[int, int], ...],
    targets: tuple[tuple[int, int, int, int], ...],
    drift: float,
    *,
    depth_shift: bool = False,
    burst: bool = False,
) -> dict[str, Any]:
    categories = ("A", "B", "A", "C", "B", "C", "A", "B")
    sizes = {
        "A": (1.0, 0.5),
        "B": (0.7, 0.7),
        "C": (1.2, 0.4),
    }
    return {
        "starts": starts,
        "station": station,
        "objects": objects,
        "categories": categories,
        "sizes": sizes,
        "targets": targets,
        "drift": drift,
        "depth_shift": depth_shift,
        "burst": burst,
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "repeated_pylons": _scene(
        ((1, 2), (1, 9), (1, 15)),
        (0, 9),
        ((4, 3), (4, 8), (4, 13), (10, 3), (10, 8), (10, 13), (18, 5), (18, 12)),
        ((0, 1, 20), (2, 1, 26), (3, 2, 38), (5, 2, 48), (6, 3, 62), (7, 3, 70)),
        0.024,
    ),
    "sparse_landmarks": _scene(
        ((2, 2), (2, 15), (12, 9)),
        (0, 9),
        ((3, 3), (6, 14), (10, 5), (13, 13), (17, 3), (19, 10), (22, 5), (22, 15)),
        ((0, 1, 18), (1, 1, 28), (2, 2, 36), (4, 2, 50), (5, 3, 60), (7, 4, 72)),
        0.028,
    ),
    "depth_link_shift": _scene(
        ((1, 3), (1, 14), (20, 9)),
        (0, 9),
        ((4, 4), (4, 13), (9, 3), (9, 14), (15, 4), (15, 13), (21, 5), (21, 12)),
        ((0, 1, 20), (1, 1, 26), (2, 2, 38), (3, 2, 44), (6, 4, 62), (7, 4, 68)),
        0.026,
        depth_shift=True,
    ),
    "crossing_surveys": _scene(
        ((1, 2), (1, 15), (22, 8)),
        (12, 9),
        ((4, 4), (4, 13), (9, 7), (9, 11), (15, 7), (15, 11), (20, 4), (20, 13)),
        ((0, 1, 22), (1, 1, 28), (2, 2, 38), (4, 2, 48), (6, 3, 62), (7, 3, 70)),
        0.022,
    ),
    "acoustic_burst": _scene(
        ((2, 2), (2, 15), (21, 9)),
        (12, 9),
        ((4, 3), (4, 14), (8, 6), (8, 11), (16, 6), (16, 11), (21, 3), (21, 14)),
        ((0, 2, 18), (1, 2, 24), (2, 3, 34), (3, 3, 40), (6, 4, 56), (7, 4, 62)),
        0.025,
        burst=True,
    ),
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
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


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _brief(value: Any, limit: int = 140) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # huge/malformed values must remain diagnosable
        return f"<{type(value).__name__} repr failed: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _u64(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

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
        parse_float=finite_float,
        object_pairs_hook=reject_duplicate,
    )


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _action_shape(action: Any) -> bool:
    retry = action.get("retry") if isinstance(action, dict) else None
    return (
        isinstance(action, dict)
        and set(action) == {
            "slot",
            "robot",
            "move",
            "broadcast_metadata",
            "validate_edge",
            "retry",
            "accept_validation",
        }
        and _integer(action.get("slot"))
        and isinstance(action.get("robot"), str)
        and action.get("robot") in ROBOTS
        and isinstance(action.get("move"), str)
        and action.get("move") in MOVES
        and isinstance(action.get("broadcast_metadata"), bool)
        and (action.get("validate_edge") is None or isinstance(action.get("validate_edge"), str))
        and (retry is None or (isinstance(retry, str) and retry in ("metadata", "verification")))
        and (action.get("accept_validation") is None or isinstance(action.get("accept_validation"), bool))
    )


def _shape_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_ok = isinstance(data, dict) and set(data) == {"scenarios"}
    scenarios_ok = root_ok and isinstance(data.get("scenarios"), list)
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parse" if parse_ok else "strict JSON parse failed"),
        _atom("result_json", "root", root_ok, f"root_keys={_brief(list(data) if isinstance(data, dict) else type(data).__name__)}"),
        _atom("result_json", "scenario_shape", scenarios_ok, "scenarios array" if scenarios_ok else "scenarios is not an array"),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, dict) else None
    scenario_ok = isinstance(rows, list) and all(
        isinstance(row, dict)
        and set(row) == {"scenario", "variant", "actions"}
        and isinstance(row.get("scenario"), str)
        and _integer(row.get("variant"))
        and isinstance(row.get("actions"), list)
        for row in rows or ()
    )
    action_ok = scenario_ok and all(
        _action_shape(action) for row in rows for action in row.get("actions", ())
    )
    return [
        _atom("result_evidence", "scenario_fields", scenario_ok, "scenario rows have exact fields" if scenario_ok else "invalid scenario row fields"),
        _atom("result_evidence", "action_fields", action_ok, "actions have exact fields" if action_ok else "invalid action fields or types"),
    ]


def _index_records(data: Any) -> dict[tuple[str, int], Any]:
    indexed: dict[tuple[str, int], Any] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for row in data["scenarios"]:
        if not isinstance(row, dict):
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            hash(key)
        except TypeError:
            continue
        indexed.setdefault(key, row)
    return indexed


# Keep the short historical name for the boundary probes and any local tooling;
# the named entry point above is the canonical record-indexing interface.
_index = _index_records


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return 0.0, ["scenarios is not an array"]
    counts: dict[Any, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            defects.append("non-object scenario row")
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            counts[key] = counts.get(key, 0) + 1
        except TypeError:
            defects.append(f"unhashable scenario key={_brief(key)}")
    for key in SCENARIO_KEYS:
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"missing scenario={key[0]}|variant={key[1]}")
        elif count > 1:
            defects.extend([f"duplicate scenario={_brief(key)}"] * (count - 1))
    defects.extend(f"unknown scenario key={_brief(key)}" for key in counts if key not in SCENARIO_KEYS)
    expected_actions = {(slot, f"r{robot}") for slot in range(HORIZON) for robot in range(N_ROBOTS)}
    for key in SCENARIO_KEYS:
        matching = [row for row in rows if isinstance(row, dict) and (row.get("scenario"), row.get("variant")) == key]
        if len(matching) != 1:
            continue
        actions = matching[0].get("actions")
        if not isinstance(actions, list):
            defects.append(f"{key} actions is not an array")
            continue
        action_counts: dict[Any, int] = {}
        for action in actions:
            if not isinstance(action, dict):
                defects.append(f"{key} non-object action")
                continue
            action_key = (action.get("slot"), action.get("robot"))
            try:
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
            except TypeError:
                defects.append(f"{_brief(key)} unhashable action key={_brief(action_key)}")
        for action_key in expected_actions:
            count = action_counts.get(action_key, 0)
            if count == 0:
                defects.append(f"{_brief(key)} missing action={_brief(action_key)}")
            elif count > 1:
                defects.extend([f"{_brief(key)} duplicate action={_brief(action_key)}"] * (count - 1))
        defects.extend(f"{_brief(key)} unknown action={_brief(action_key)}" for action_key in action_counts if action_key not in expected_actions)
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _move_point(point: tuple[int, int], move: str) -> tuple[int, int]:
    delta = MOVE_DELTAS.get(move, (0, 0))
    return point[0] + delta[0], point[1] + delta[1]


def _traversable(point: tuple[int, int]) -> bool:
    return 0 <= point[0] < GRID_X and 0 <= point[1] < GRID_Y


def _obs_id(name: str, variant: int, robot: int, slot: int) -> str:
    return f"obs={name}|variant={variant}|robot={robot}|slot={slot}"


def _observation(name: str, variant: int, robot: int, slot: int, position: tuple[int, int], drift: tuple[float, float]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    visible = [
        (abs(obj[0] - position[0]) + abs(obj[1] - position[1]), object_id)
        for object_id, obj in enumerate(spec["objects"])
        if abs(obj[0] - position[0]) + abs(obj[1] - position[1]) <= 4
    ]
    object_id = min(visible)[1] if visible else None
    sigma = 0.06 + 0.06 * variant
    if spec["depth_shift"] and slot >= 30 + 6 * variant:
        sigma += 0.04
    if object_id is None:
        return {"obs_id": _obs_id(name, variant, robot, slot), "robot": robot, "slot": slot, "position": position, "visible": False, "_true_object": None, "category": None, "size": None, "g": None, "d": 1.0, "sigma": sigma, "quality": 0.0}
    obj = spec["objects"][object_id]
    distance = abs(obj[0] - position[0]) + abs(obj[1] - position[1])
    d = _clip(distance / 4.0)
    quality = _clip(1.0 - 0.04 * d - 0.10 * sigma)
    width, height = spec["sizes"][spec["categories"][object_id]]
    g = (obj[0] - position[0] - drift[0], obj[1] - position[1] - drift[1], width, height)
    return {"obs_id": _obs_id(name, variant, robot, slot), "robot": robot, "slot": slot, "position": position, "visible": True, "_true_object": object_id, "category": spec["categories"][object_id], "size": (width, height), "g": g, "d": d, "sigma": sigma, "quality": quality}


def _edge_id(first: str, second: str) -> str:
    a, b = sorted((first, second))
    return f"edge={a}--{b}"


def _public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: observation[key]
        for key in (
            "obs_id",
            "visible",
            "category",
            "size",
            "g",
            "d",
            "sigma",
            "quality",
        )
    }


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _candidate_record(
    spec: Mapping[str, Any],
    first_id: str,
    first: Mapping[str, Any],
    second_id: str,
    second: Mapping[str, Any],
    boundary: int,
) -> dict[str, Any] | None:
    if (
        not first.get("visible")
        or not second.get("visible")
        or first["robot"] == second["robot"]
        or first.get("category") != second.get("category")
    ):
        return None
    g1, g2 = first["g"], second["g"]
    distance = math.sqrt(sum((float(g1[k]) - float(g2[k])) ** 2 for k in range(4)))
    similarity = _clip(1.0 - distance / 12.0)
    if similarity + EPS < 0.55:
        return None
    first_object = int(first["_true_object"])
    second_object = int(second["_true_object"])
    p1 = (
        spec["objects"][first_object][0] - first["position"][0],
        spec["objects"][first_object][1] - first["position"][1],
    )
    p2 = (
        spec["objects"][second_object][0] - second["position"][0],
        spec["objects"][second_object][1] - second["position"][1],
    )
    expected = (p1[0] - p2[0], p1[1] - p2[1])
    observed = (float(g1[0]) - float(g2[0]), float(g1[1]) - float(g2[1]))
    residual = (
        math.hypot(observed[0] - expected[0], observed[1] - expected[1])
        + 0.5 * abs(float(g1[2]) - float(g2[2]))
        + 0.5 * abs(float(g1[3]) - float(g2[3]))
    )
    return {
        "edge_id": _edge_id(first_id, second_id),
        "first_id": first_id,
        "second_id": second_id,
        "generation": boundary,
        "expiry": boundary + 10,
        "similarity": similarity,
        "residual": residual,
        "consistent": first_object == second_object and residual <= 0.35 + EPS,
        "attempts": 0,
    }


def _packet_sample(name: str, variant: int, packet_id: str, attempt: int) -> float:
    return _u64(f"{name}|{variant}|{packet_id}|{attempt}|salt={19 * variant}")


def _actions_by_slot(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    actions = row.get("actions", ()) if isinstance(row, Mapping) else ()
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    if not isinstance(actions, list):
        return result
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        slot, robot = action.get("slot"), action.get("robot")
        if _integer(slot) and isinstance(robot, str) and robot in ROBOTS:
            result.setdefault((slot, int(robot[1:])), action)
    return result


def _replay(name: str, variant: int, row: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    actions = _actions_by_slot(row)
    positions: list[list[tuple[int, int]]] = [list(spec["starts"])]
    drifts: list[list[tuple[float, float]]] = [[(0.0, 0.0) for _ in range(N_ROBOTS)]]
    energies: list[list[float]] = [[ENERGY0 for _ in range(N_ROBOTS)]]
    local_history: list[deque[dict[str, Any]]] = [deque(maxlen=3) for _ in range(N_ROBOTS)]
    observation_truth: dict[str, dict[str, Any]] = {}
    received: dict[str, dict[str, Any]] = {}
    generated_pairs: set[tuple[str, str]] = set()
    pending: dict[str, dict[str, Any]] = {}
    attempted: set[str] = set()
    arrivals: dict[int, list[dict[str, Any]]] = {}
    waiting: deque[dict[str, Any]] = deque()
    active: dict[str, Any] | None = None
    feedback: list[deque[dict[str, Any]]] = [deque() for _ in range(N_ROBOTS)]
    failed_retry: list[deque[dict[str, Any]]] = [deque() for _ in range(N_ROBOTS)]
    graph_edges: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    candidate_views: list[dict[str, Any]] = []
    feedback_events: list[dict[str, Any]] = []
    failure_events: list[dict[str, Any]] = []
    candidate_snapshots: list[list[dict[str, Any]]] = []
    feedback_snapshots: list[list[list[dict[str, Any]]]] = []
    queue_snapshots: list[dict[str, Any]] = []
    observation_records: list[dict[str, Any]] = []
    action_start_drifts: list[list[tuple[float, float]]] = []
    transmission_records: list[dict[str, Any]] = []
    sampled: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    safety_events = 0
    motion_cost = 0.0
    accepted_units = 0
    informative_units = 0
    accepted_false = 0
    total_constraints = 0
    total_candidates = 0
    dropped_candidates = 0
    queue_wait_slots = 0
    metadata_requests = metadata_successes = 0
    verification_requests = verification_successes = verification_completions = 0

    def add_candidates(boundary: int) -> None:
        nonlocal total_candidates
        items = list(received.items())
        for i, (first_id, first) in enumerate(items):
            for second_id, second in items[i + 1 :]:
                pair = tuple(sorted((first_id, second_id)))
                if pair in generated_pairs:
                    continue
                candidate = _candidate_record(
                    spec, first_id, first, second_id, second, boundary
                )
                if candidate is None:
                    continue
                generated_pairs.add(pair)
                edge = candidate["edge_id"]
                pending[edge] = candidate
                candidate_records.append(dict(candidate))
                candidate_views.append(
                    {
                        key: candidate[key]
                        for key in (
                            "edge_id",
                            "first_id",
                            "second_id",
                            "generation",
                            "expiry",
                            "similarity",
                        )
                    }
                )
                total_candidates += 1

    def complete_validation(packet: dict[str, Any], boundary: int) -> None:
        nonlocal verification_completions
        verification_completions += 1
        edge = packet["edge"]
        result = {"edge_id": edge["edge_id"], "consistent": edge["consistent"], "residual": edge["residual"], "sample_pass": packet["sample_pass"], "packet": packet, "boundary": boundary, "owner": packet["owner"]}
        feedback[packet["owner"]].append(result)
        feedback_events.append(result)

    for slot in range(HORIZON):
        boundary = slot
        expired = [edge_id for edge_id, edge in pending.items() if edge["expiry"] < boundary]
        for edge_id in expired:
            pending.pop(edge_id)
            dropped_candidates += 1
        for packet in arrivals.pop(boundary, []):
            if packet["kind"] == "metadata":
                for observation in packet["payload"]:
                    received[observation["obs_id"]] = observation_truth[observation["obs_id"]]
                add_candidates(boundary)
            else:
                if len(waiting) >= QUEUE_CAPACITY:
                    dropped_candidates += 1
                    packet["failure"] = "queue"
                    packet["fail_slot"] = boundary
                    packet["queue_admitted"] = False
                    failed_retry[packet["owner"]].append({**packet, "fail_slot": boundary, "failure": "queue"})
                    failure_events.append(
                        {
                            "kind": packet["kind"],
                            "owner": packet["owner"],
                            "attempt": packet["attempt"],
                            "deadline": packet["deadline"],
                            "fail_slot": boundary,
                        }
                    )
                else:
                    packet["arrival"] = boundary
                    packet["queue_admitted"] = True
                    waiting.append(packet)
        if active is not None and active["completion"] <= boundary:
            complete_validation(active, boundary)
            active = None
        if active is None and waiting:
            active = waiting.popleft()
            active["service_start"] = boundary
            active["completion"] = boundary + SERVICE_UNITS
            queue_wait_slots += boundary - active["arrival"]

        candidate_snapshots.append(
            [
                {
                    key: candidate[key]
                    for key in (
                        "edge_id",
                        "first_id",
                        "second_id",
                        "generation",
                        "expiry",
                        "similarity",
                    )
                }
                for candidate in sorted(
                    pending.values(), key=lambda item: str(item["edge_id"])
                )
            ]
        )
        feedback_snapshots.append(
            [
                [
                    {
                        key: item[key]
                        for key in (
                            "edge_id",
                            "consistent",
                            "residual",
                            "sample_pass",
                            "boundary",
                            "owner",
                        )
                    }
                    for item in robot_feedback
                ]
                for robot_feedback in feedback
            ]
        )
        queue_snapshots.append(
            {
                "waiting": len(waiting),
                "active_completion": active["completion"] if active is not None else None,
            }
        )

        current_actions = [actions.get((slot, robot), {}) for robot in range(N_ROBOTS)]
        action_start_drifts.append(list(drifts[-1]))
        observations = [_observation(name, variant, robot, slot, positions[-1][robot], drifts[-1][robot]) for robot in range(N_ROBOTS)]
        observation_records.extend(observations)
        for robot, observation in enumerate(observations):
            local_history[robot].append(observation)
            observation_truth[observation["obs_id"]] = observation

        # Applying a validation decision is a separate action from requesting a packet.
        for robot, action in enumerate(current_actions):
            accept = action.get("accept_validation")
            if accept is not None:
                if not isinstance(accept, bool) or not feedback[robot]:
                    errors.append(f"slot={slot},robot=r{robot}: accept_validation has no feedback")
                else:
                    result = feedback[robot].popleft()
                    total_constraints += 1
                    if accept and result["consistent"] and result["sample_pass"]:
                        for endpoint in (result["packet"]["edge"]["first_id"], result["packet"]["edge"]["second_id"]):
                            endpoint_robot = received[endpoint]["robot"]
                            drifts[-1][endpoint_robot] = (drifts[-1][endpoint_robot][0] * 0.60, drifts[-1][endpoint_robot][1] * 0.60)
                        graph_edges.append({"edge_id": result["edge_id"], "residual": result["residual"], "correct": True})
                        informative_units += VALIDATION_UNITS
                    elif accept and not result["consistent"] and result["sample_pass"]:
                        for endpoint in (result["packet"]["edge"]["first_id"], result["packet"]["edge"]["second_id"]):
                            endpoint_robot = received[endpoint]["robot"]
                            drifts[-1][endpoint_robot] = (drifts[-1][endpoint_robot][0] + 0.12, drifts[-1][endpoint_robot][1] + 0.12)
                        accepted_false += 1
                        graph_edges.append({"edge_id": result["edge_id"], "residual": result["residual"], "correct": False})
                    elif (not accept) and (not result["consistent"]) and result["sample_pass"]:
                        informative_units += VALIDATION_UNITS

        requests: list[dict[str, Any]] = []
        for robot, action in enumerate(current_actions):
            if not _action_shape(action):
                errors.append(f"slot={slot},robot=r{robot}: action shape/value invalid")
                continue
            retry_kind = action.get("retry")
            if retry_kind is not None and (action.get("broadcast_metadata") or action.get("validate_edge") is not None):
                errors.append(f"slot={slot},robot=r{robot}: retry combined with a new request")
            if action.get("broadcast_metadata") and action.get("validate_edge") is not None:
                errors.append(f"slot={slot},robot=r{robot}: two new packets in one slot")
            if retry_kind is not None:
                candidates = [
                    p
                    for p in failed_retry[robot]
                    if p.get("fail_slot", -1) + 1 == slot
                    and p.get("attempt", 0) == 0
                    and p.get("deadline", -1) >= slot
                ]
                if not candidates:
                    errors.append(f"slot={slot},robot=r{robot}: no eligible {retry_kind} feedback")
                else:
                    candidates = [p for p in candidates if p.get("kind") == retry_kind]
                    if not candidates:
                        errors.append(f"slot={slot},robot=r{robot}: retry kind has no eligible packet")
                    else:
                        previous = candidates[0]
                        failed_retry[robot].remove(previous)
                        if retry_kind == "metadata":
                            metadata_requests += 1
                        else:
                            verification_requests += 1
                        packet = {
                            key: previous[key]
                            for key in (
                                "kind",
                                "owner",
                                "units",
                                "deadline",
                                "generation",
                                "packet_id",
                            )
                        }
                        if retry_kind == "metadata":
                            packet["payload"] = previous["payload"]
                        else:
                            packet["edge"] = previous["edge"]
                            packet["edge_id"] = previous["edge_id"]
                        packet.update(
                            attempt=1,
                            request_slot=slot,
                            queue_age=slot - previous["generation"],
                        )
                        requests.append(packet)
            elif action.get("broadcast_metadata"):
                requests.append({"kind": "metadata", "owner": robot, "payload": [_public_observation(item) for item in local_history[robot]], "units": SUMMARY_UNITS, "deadline": slot + 8, "queue_age": 0, "attempt": 0, "generation": slot, "packet_id": f"metadata|r{robot}|generation={slot}", "request_slot": slot})
                metadata_requests += 1
            elif action.get("validate_edge") is not None:
                edge_id = action.get("validate_edge")
                edge = pending.get(edge_id)
                if edge is None or edge["expiry"] < slot:
                    errors.append(
                        f"slot={slot},robot=r{robot}: validate_edge is not visible at action start: {_brief(edge_id)}"
                    )
                elif edge_id in attempted:
                    errors.append(f"slot={slot},robot=r{robot}: edge requested more than once without retry")
                else:
                    attempted.add(edge_id)
                    edge["attempts"] += 1
                    pending.pop(edge_id)
                    requests.append({"kind": "verification", "owner": robot, "edge": edge, "edge_id": edge_id, "units": VALIDATION_UNITS, "deadline": edge["expiry"], "queue_age": 0, "attempt": 0, "generation": slot, "packet_id": f"verification|{edge_id}|r{robot}|generation={slot}", "request_slot": slot})
                    verification_requests += 1

        # Channel admission uses the public priority rule and one deterministic sample per packet.
        requests.sort(
            key=lambda p: (
                p["deadline"],
                p.get("queue_age", 0),
                p["owner"],
                p["kind"],
                p.get("edge_id", ""),
                p.get("attempt", 0),
            )
        )
        admitted_units = 0
        admitted_packets = 0
        for packet in requests:
            packet["request_position"] = positions[-1][packet["owner"]]
            transmission_records.append(packet)
            if (
                slot > packet["deadline"]
                or packet["units"] > CHANNEL_CAPACITY - admitted_units
                or accepted_units + packet["units"] > TOTAL_BUDGET
            ):
                packet["admitted"] = False
                packet["success"] = False
                if slot > packet["deadline"]:
                    packet["failure"] = "deadline"
                elif packet["units"] > CHANNEL_CAPACITY - admitted_units:
                    packet["failure"] = "capacity"
                else:
                    packet["failure"] = "budget"
                packet["fail_slot"] = slot
                if packet["kind"] == "verification":
                    dropped_candidates += 1
                failed_retry[packet["owner"]].append({**packet, "fail_slot": slot, "failure": "capacity"})
                failure_events.append(
                    {
                        "kind": packet["kind"],
                        "owner": packet["owner"],
                        "attempt": packet["attempt"],
                        "deadline": packet["deadline"],
                        "fail_slot": slot,
                    }
                )
                continue
            n = admitted_packets + 1
            distance = _distance(positions[-1][packet["owner"]], spec["station"])
            z = (spec["depth_shift"] and slot >= 30 + 6 * variant) or (spec["burst"] and 22 + 6 * variant <= slot <= 33 + 6 * variant)
            probability = _clip(0.92 - 0.05 * _clip(distance / 12.0) - 0.08 * int(z) - 0.04 * (n - 1), 0.08, 0.92)
            sample = _packet_sample(name, variant, packet["packet_id"], packet["attempt"])
            packet["sample"] = sample
            packet["probability"] = probability
            admitted_units += packet["units"]
            admitted_packets += 1
            accepted_units += packet["units"]
            packet["admitted"] = True
            delay = 1 + int(distance // 8)
            packet["arrival_boundary"] = slot + delay
            if sample >= probability:
                packet["success"] = False
                packet["failure"] = "link"
                packet["fail_slot"] = slot
                if packet["kind"] == "verification":
                    dropped_candidates += 1
                failed_retry[packet["owner"]].append({**packet, "fail_slot": slot, "failure": "link"})
                failure_events.append(
                    {
                        "kind": packet["kind"],
                        "owner": packet["owner"],
                        "attempt": packet["attempt"],
                        "deadline": packet["deadline"],
                        "fail_slot": slot,
                    }
                )
                continue
            if packet["kind"] == "metadata":
                packet["success"] = True
                metadata_successes += 1
                arrivals.setdefault(slot + delay, []).append(packet)
            else:
                packet["success"] = True
                verification_successes += 1
                packet["sample_pass"] = _u64(f"{name}|{variant}|{packet['edge_id']}|validate|{packet['attempt']}|salt={19 * variant}") < 0.95 - 0.05 * variant
                arrivals.setdefault(slot + delay, []).append(packet)

        # Safe motion executes after communication decisions and service updates.
        start_positions = positions[-1]
        next_positions: list[tuple[int, int]] = []
        accepted_destinations: set[tuple[int, int]] = set()
        for robot, action in enumerate(current_actions):
            move = action.get("move")
            proposed = _move_point(start_positions[robot], move) if move in MOVES else start_positions[robot]
            unsafe = move != "stay" and (not _traversable(proposed) or proposed in accepted_destinations or proposed in {start_positions[j] for j in range(robot + 1, N_ROBOTS)})
            if unsafe:
                safety_events += 1
                proposed = start_positions[robot]
            accepted_destinations.add(proposed)
            next_positions.append(proposed)
            motion_cost += 1.0 if proposed != start_positions[robot] else 0.2
        next_energy = [energies[-1][robot] - (1.0 if next_positions[robot] != start_positions[robot] else 0.2) for robot in range(N_ROBOTS)]
        if any(value < -EPS for value in next_energy):
            errors.append(f"slot={slot}: energy below zero")
        for robot, position in enumerate(next_positions):
            for object_id, (x, y) in enumerate(spec["objects"]):
                if object_id in {target[0] for target in spec["targets"]} and object_id not in sampled and position == (x, y):
                    sampled[object_id] = {"slot": slot + 1, "robot": robot, "error": math.hypot(*drifts[-1][robot])}
        next_drift = [list(vector) for vector in drifts[-1]]
        for robot in range(N_ROBOTS):
            for axis in range(2):
                sample = _u64(f"{name}|{variant}|r{robot}|t{slot}|drift|{axis}")
                next_drift[robot][axis] += spec["drift"] * (1.0 + 0.2 * variant) * (2.0 * sample - 1.0)
        positions.append(next_positions)
        energies.append(next_energy)
        drifts.append([tuple(vector) for vector in next_drift])

    # Terminal boundary applies arrivals and one service step; any remaining work is expired.
    boundary = HORIZON
    expired = [edge_id for edge_id, edge in pending.items() if edge["expiry"] < boundary]
    for edge_id in expired:
        pending.pop(edge_id)
        dropped_candidates += 1
    for packet in arrivals.pop(boundary, []):
        if packet["kind"] == "verification":
            if len(waiting) >= QUEUE_CAPACITY:
                dropped_candidates += 1
            else:
                packet["arrival"] = boundary
                waiting.append(packet)
        else:
            for observation in packet["payload"]:
                received[observation["obs_id"]] = observation_truth[observation["obs_id"]]
            add_candidates(boundary)
    if active is not None and active["completion"] <= boundary:
        complete_validation(active, boundary)
        active = None
    dropped_candidates += len(pending) + len(waiting) + (1 if active is not None else 0)
    return {
        "positions": positions,
        "drifts": drifts,
        "energies": energies,
        "sampled": sampled,
        "graph_edges": graph_edges,
        "candidate_records": candidate_records,
        "candidate_views": candidate_views,
        "feedback_events": feedback_events,
        "failure_events": failure_events,
        "candidate_snapshots": candidate_snapshots,
        "feedback_snapshots": feedback_snapshots,
        "queue_snapshots": queue_snapshots,
        "observation_records": observation_records,
        "action_start_drifts": action_start_drifts,
        "transmission_records": transmission_records,
        "safety_events": safety_events,
        "motion_cost": motion_cost,
        "accepted_units": accepted_units,
        "informative_units": informative_units,
        "accepted_false_constraints": accepted_false,
        "total_constraints": total_constraints,
        "total_candidates": total_candidates,
        "dropped_candidates": dropped_candidates,
        "queue_wait_slots": queue_wait_slots,
        "metadata_requests": metadata_requests,
        "metadata_successes": metadata_successes,
        "verification_requests": verification_requests,
        "verification_successes": verification_successes,
        "verification_completions": verification_completions,
        "errors": errors,
        "pending_feedback": sum(len(queue) for queue in feedback),
    }


def _legality(record_map: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {name: [] for name in ("action_values", "visible_edges", "retry_feedback", "communication", "energy", "acceptance")}
    for key in SCENARIO_KEYS:
        state = _replay(key[0], key[1], record_map.get(key, {}))
        for error in state["errors"]:
            if "action shape" in error or "two new packets" in error:
                failures["action_values"].append(f"{key}: {error}")
            elif "visible" in error or "edge requested" in error:
                failures["visible_edges"].append(f"{key}: {error}")
            elif "accept_validation" in error:
                failures["acceptance"].append(f"{key}: {error}")
            elif "retry" in error or "feedback" in error:
                failures["retry_feedback"].append(f"{key}: {error}")
            elif "energy" in error:
                failures["energy"].append(f"{key}: {error}")
            elif "accept" in error:
                failures["acceptance"].append(f"{key}: {error}")
            else:
                failures["communication"].append(f"{key}: {error}")
    return [_atom("decision_legality", name, not failures[name], "passed" if not failures[name] else " | ".join(failures[name][:8])) for name in failures]


def _replay_atoms(record_map: Mapping[tuple[str, int], Any], replayed: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {name: [] for name in ("motion", "observations", "candidates", "communication", "mission", "feedback")}
    for key in SCENARIO_KEYS:
        state = replayed.get(key)
        if not isinstance(state, Mapping):
            for name in failures:
                failures[name].append(f"{key}: no replay state")
            continue
        name, variant = key
        spec = SCENARIOS[name]
        positions = state.get("positions", ())
        drifts = state.get("drifts", ())
        energies = state.get("energies", ())
        action_start_drifts = state.get("action_start_drifts", ())
        actions = _actions_by_slot(record_map.get(key, {}))
        if len(positions) != HORIZON + 1 or len(drifts) != HORIZON + 1 or len(energies) != HORIZON + 1:
            failures["motion"].append(f"{key}: boundary histories have wrong length")
            continue

        expected_safety = 0
        expected_motion = 0.0
        for boundary, cells in enumerate(positions):
            if len(cells) != N_ROBOTS or len(set(map(tuple, cells))) != N_ROBOTS:
                failures["motion"].append(f"{key}: invalid position set at boundary={boundary}")
            elif any(not _traversable(tuple(cell)) for cell in cells):
                failures["motion"].append(f"{key}: out-of-grid position at boundary={boundary}")
        for slot in range(HORIZON):
            accepted: set[tuple[int, int]] = set()
            start_cells = [tuple(cell) for cell in positions[slot]]
            for robot in range(N_ROBOTS):
                move = actions[(slot, robot)]["move"]
                proposed = _move_point(start_cells[robot], move)
                unsafe = move != "stay" and (
                    not _traversable(proposed)
                    or proposed in accepted
                    or proposed in {start_cells[j] for j in range(robot + 1, N_ROBOTS)}
                )
                expected = start_cells[robot] if unsafe else proposed
                accepted.add(expected)
                expected_safety += int(unsafe)
                cost = 1.0 if expected != start_cells[robot] else 0.2
                expected_motion += cost
                if tuple(positions[slot + 1][robot]) != expected:
                    failures["motion"].append(f"{key}: position transition mismatch at slot={slot},robot=r{robot}")
                if not math.isclose(energies[slot + 1][robot], energies[slot][robot] - cost, rel_tol=0.0, abs_tol=EPS):
                    failures["motion"].append(f"{key}: energy transition mismatch at slot={slot},robot=r{robot}")
        if state.get("safety_events") != expected_safety or not math.isclose(state.get("motion_cost", -1.0), expected_motion, rel_tol=0.0, abs_tol=EPS):
            failures["motion"].append(f"{key}: safety or motion total mismatch")

        observations = state.get("observation_records", ())
        if len(observations) != HORIZON * N_ROBOTS or len(action_start_drifts) != HORIZON:
            failures["observations"].append(f"{key}: observation or action-start history length mismatch")
        else:
            for slot in range(HORIZON):
                for robot in range(N_ROBOTS):
                    expected = _observation(
                        name,
                        variant,
                        robot,
                        slot,
                        tuple(positions[slot][robot]),
                        tuple(action_start_drifts[slot][robot]),
                    )
                    actual = observations[slot * N_ROBOTS + robot]
                    if actual != expected:
                        failures["observations"].append(f"{key}: observation mismatch at slot={slot},robot=r{robot}")
                        break
        observation_index = {
            item["obs_id"]: item for item in observations if isinstance(item, Mapping) and isinstance(item.get("obs_id"), str)
        }

        candidates = state.get("candidate_records", ())
        candidate_ids: list[str] = []
        for candidate in candidates if isinstance(candidates, list) else ():
            if not isinstance(candidate, Mapping):
                failures["candidates"].append(f"{key}: non-object candidate")
                continue
            first = observation_index.get(candidate.get("first_id"))
            second = observation_index.get(candidate.get("second_id"))
            if first is None or second is None or not _integer(candidate.get("generation")):
                failures["candidates"].append(f"{key}: candidate endpoint or generation invalid")
                continue
            expected = _candidate_record(
                spec,
                str(candidate["first_id"]),
                first,
                str(candidate["second_id"]),
                second,
                int(candidate["generation"]),
            )
            candidate_ids.append(str(candidate.get("edge_id")))
            if expected is None or any(
                candidate.get(field) != expected[field]
                for field in ("edge_id", "expiry", "consistent")
            ) or any(
                not math.isclose(float(candidate.get(field, math.inf)), float(expected[field]), rel_tol=0.0, abs_tol=EPS)
                for field in ("similarity", "residual")
            ):
                failures["candidates"].append(
                    f"{_brief(key)}: candidate formula mismatch for {_brief(candidate.get('edge_id'))}"
                )
        if len(candidate_ids) != len(set(candidate_ids)) or state.get("total_candidates") != len(candidate_ids):
            failures["candidates"].append(f"{key}: candidate IDs or count mismatch")

        transmissions = state.get("transmission_records", ())
        accepted_sum = 0
        request_counts = {"metadata": 0, "verification": 0}
        success_counts = {"metadata": 0, "verification": 0}
        admitted_per_slot: dict[int, int] = {}
        expected_wait = 0
        for packet in transmissions if isinstance(transmissions, list) else ():
            kind = packet.get("kind")
            owner = packet.get("owner")
            generation = packet.get("generation")
            attempt = packet.get("attempt")
            request_slot = packet.get("request_slot")
            if kind not in request_counts or not all(_integer(value) for value in (owner, generation, attempt, request_slot)):
                failures["communication"].append(f"{key}: malformed transmission record")
                continue
            request_counts[kind] += 1
            expected_id = (
                f"metadata|r{owner}|generation={generation}"
                if kind == "metadata"
                else f"verification|{packet.get('edge_id')}|r{owner}|generation={generation}"
            )
            expected_units = SUMMARY_UNITS if kind == "metadata" else VALIDATION_UNITS
            if packet.get("packet_id") != expected_id or packet.get("units") != expected_units or packet.get("queue_age") != request_slot - generation:
                failures["communication"].append(f"{key}: packet identity, payload, or age mismatch")
            if packet.get("admitted"):
                admitted_index = admitted_per_slot.get(request_slot, 0) + 1
                admitted_per_slot[request_slot] = admitted_index
                accepted_sum += expected_units
                request_position = tuple(packet.get("request_position", ()))
                distance = _distance(request_position, spec["station"])
                stress = (
                    spec["depth_shift"] and request_slot >= 30 + 6 * variant
                ) or (
                    spec["burst"] and 22 + 6 * variant <= request_slot <= 33 + 6 * variant
                )
                probability = _clip(
                    0.92
                    - 0.05 * _clip(distance / 12.0)
                    - 0.08 * int(stress)
                    - 0.04 * (admitted_index - 1),
                    0.08,
                    0.92,
                )
                sample = _packet_sample(name, variant, expected_id, attempt)
                if not math.isclose(packet.get("probability", math.inf), probability, rel_tol=0.0, abs_tol=EPS) or not math.isclose(packet.get("sample", math.inf), sample, rel_tol=0.0, abs_tol=EPS):
                    failures["communication"].append(f"{key}: packet probability or sample mismatch for {expected_id}")
                if packet.get("success") != (sample < probability):
                    failures["communication"].append(f"{key}: link outcome mismatch for {expected_id}")
                if packet.get("arrival_boundary") != request_slot + 1 + int(distance // 8):
                    failures["communication"].append(f"{key}: arrival boundary mismatch for {expected_id}")
                success_counts[kind] += int(bool(packet.get("success")))
            if "service_start" in packet:
                if packet.get("completion") != packet["service_start"] + SERVICE_UNITS:
                    failures["communication"].append(f"{key}: service duration mismatch")
                expected_wait += packet["service_start"] - packet["arrival"]
        if state.get("accepted_units") != accepted_sum or state.get("queue_wait_slots") != expected_wait:
            failures["communication"].append(f"{key}: payload or queue-wait total mismatch")
        if state.get("metadata_requests") != request_counts["metadata"] or state.get("verification_requests") != request_counts["verification"] or state.get("metadata_successes") != success_counts["metadata"] or state.get("verification_successes") != success_counts["verification"]:
            failures["communication"].append(f"{key}: request or link-success count mismatch")

        feedback_events = state.get("feedback_events", ())
        feedback_queues: list[deque[Mapping[str, Any]]] = [deque() for _ in range(N_ROBOTS)]
        expected_graph: list[dict[str, Any]] = []
        expected_informative = 0
        expected_false = 0
        expected_constraints = 0
        for slot in range(HORIZON):
            for event in feedback_events if isinstance(feedback_events, list) else ():
                if event.get("boundary") == slot and _integer(event.get("owner")):
                    feedback_queues[event["owner"]].append(event)
            expected_post = [tuple(vector) for vector in action_start_drifts[slot]]
            for robot in range(N_ROBOTS):
                accept = actions[(slot, robot)]["accept_validation"]
                if accept is None:
                    continue
                expected_constraints += 1
                event = feedback_queues[robot].popleft()
                edge = event["packet"]["edge"]
                if accept and event["consistent"] and event["sample_pass"]:
                    for endpoint in (edge["first_id"], edge["second_id"]):
                        endpoint_robot = observation_index[endpoint]["robot"]
                        current = expected_post[endpoint_robot]
                        expected_post[endpoint_robot] = (current[0] * 0.60, current[1] * 0.60)
                    expected_graph.append({"edge_id": edge["edge_id"], "residual": edge["residual"], "correct": True})
                    expected_informative += VALIDATION_UNITS
                elif accept and not event["consistent"] and event["sample_pass"]:
                    for endpoint in (edge["first_id"], edge["second_id"]):
                        endpoint_robot = observation_index[endpoint]["robot"]
                        current = expected_post[endpoint_robot]
                        expected_post[endpoint_robot] = (current[0] + 0.12, current[1] + 0.12)
                    expected_graph.append({"edge_id": edge["edge_id"], "residual": edge["residual"], "correct": False})
                    expected_false += 1
                elif not accept and not event["consistent"] and event["sample_pass"]:
                    expected_informative += VALIDATION_UNITS
            if any(
                not math.isclose(drifts[slot][robot][axis], expected_post[robot][axis], rel_tol=0.0, abs_tol=EPS)
                for robot in range(N_ROBOTS)
                for axis in range(2)
            ):
                failures["feedback"].append(f"{key}: acceptance drift mismatch at slot={slot}")
            expected_next: list[tuple[float, float]] = []
            for robot in range(N_ROBOTS):
                increments = [
                    spec["drift"]
                    * (1.0 + 0.2 * variant)
                    * (2.0 * _u64(f"{name}|{variant}|r{robot}|t{slot}|drift|{axis}") - 1.0)
                    for axis in range(2)
                ]
                expected_next.append((expected_post[robot][0] + increments[0], expected_post[robot][1] + increments[1]))
            actual_next = action_start_drifts[slot + 1] if slot + 1 < HORIZON else drifts[HORIZON]
            if any(
                not math.isclose(actual_next[robot][axis], expected_next[robot][axis], rel_tol=0.0, abs_tol=EPS)
                for robot in range(N_ROBOTS)
                for axis in range(2)
            ):
                failures["feedback"].append(f"{key}: drift increment mismatch at slot={slot}")
        terminal_feedback = sum(1 for event in feedback_events if event.get("boundary") == HORIZON)
        expected_pending = sum(len(queue) for queue in feedback_queues) + terminal_feedback
        if state.get("graph_edges") != expected_graph or state.get("informative_units") != expected_informative or state.get("accepted_false_constraints") != expected_false or state.get("total_constraints") != expected_constraints or state.get("pending_feedback") != expected_pending:
            failures["feedback"].append(f"{key}: feedback, graph, or constraint totals mismatch")

        expected_sampled: dict[int, dict[str, Any]] = {}
        target_ids = {target[0] for target in spec["targets"]}
        for slot in range(HORIZON):
            for robot in range(N_ROBOTS):
                for object_id, point in enumerate(spec["objects"]):
                    if object_id in target_ids and object_id not in expected_sampled and tuple(positions[slot + 1][robot]) == tuple(point):
                        expected_sampled[object_id] = {
                            "slot": slot + 1,
                            "robot": robot,
                            "error": math.hypot(*drifts[slot][robot]),
                        }
        if state.get("sampled") != expected_sampled:
            failures["mission"].append(f"{key}: sampled-target lifecycle mismatch")
        for edge in state.get("graph_edges", ()):
            if not isinstance(edge, Mapping) or not _finite(edge.get("residual")) or edge.get("residual", -1) < 0:
                failures["mission"].append(f"{key}: graph edge evidence invalid")
    return [_atom("replay_validity", name, not failures[name], "passed" if not failures[name] else " | ".join(failures[name][:8])) for name in failures]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name, variant in SCENARIO_KEYS:
        spec = SCENARIOS[name]
        state = replayed[(name, variant)]
        total_value = sum(target[1] for target in spec["targets"])
        navigation = 0.0
        for object_id, value, deadline in spec["targets"]:
            result = state["sampled"].get(object_id)
            if result is None or result["slot"] > deadline:
                continue
            error = result["error"]
            quality = 1.0 if error <= 0.8 else 0.5 * _clip(1.0 - error / 0.8)
            navigation += value * quality
        totals["primary_metrics_1"] += _clip(navigation / max(1, total_value))
        vectors = [vector for boundary in state["drifts"] for vector in boundary]
        pose_mse = sum(dx * dx + dy * dy for dx, dy in vectors) / max(1, len(vectors))
        edges = state["graph_edges"]
        graph_residual = (
            sum(edge["residual"] if edge["correct"] else GRAPH_RESIDUAL_REF for edge in edges)
            / len(edges)
            if edges
            else GRAPH_RESIDUAL_REF
        )
        totals["primary_metrics_2"] += _clip(0.6 * _clip(1.0 - pose_mse / POSE_MSE_REF) + 0.4 * _clip(1.0 - graph_residual / GRAPH_RESIDUAL_REF))
        totals["primary_metrics_3"] += _clip(0.6 * _clip(1.0 - state["safety_events"] / (HORIZON * N_ROBOTS)) + 0.4 * _clip(1.0 - state["motion_cost"] / (HORIZON * N_ROBOTS)))
        totals["auxiliary_metrics_1"] += _clip(0.5 * _clip(1.0 - state["accepted_units"] / TOTAL_BUDGET) + 0.5 * state["informative_units"] / max(1, state["accepted_units"]))
        totals["auxiliary_metrics_2"] += _clip(0.5 * _clip(1.0 - state["queue_wait_slots"] / HORIZON) + 0.5 * _clip(1.0 - state["accepted_false_constraints"] / max(1, state["total_constraints"])))
    return {key: _clip(value / len(SCENARIO_KEYS)) for key, value in totals.items()}


def _gated_metrics(
    replayed: Mapping[tuple[str, int], Any], validity_gate: bool
) -> dict[str, float]:
    return _metrics(replayed) if validity_gate else {key: 0.0 for key in METRIC_KEYS}


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid fixed atom shares")
    if any(not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share differs from declaration")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    return "failed_atoms=[]" if not failed else "failed_atoms=[{}]; {}".format(",".join(atom.id for atom in failed), " | ".join(f"{atom.id}: {atom.detail}" for atom in failed))


def _failed_atoms(group: str, detail: str) -> list[AtomVerdict]:
    return [
        _atom(group, key.removeprefix(f"{group}."), False, detail)
        for key in ATOM_SHARES[group]
    ]


def _headings(text: str) -> dict[str, str]:
    """Parse visible Markdown headings and their non-empty bodies."""
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
    path = Path(ws.root) / "report.md"
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    try:
        if not path.is_file() or path.is_symlink():
            return 0.0, "report.md is not a regular file"
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(text)
    missing = [title for title in required if not headings.get(title)]
    score = max(0.0, 2.0 - 0.5 * len(missing)) / 2.0
    return score, f"missing_sections={missing}; normalized={score:.9g}"


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = False
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
            remove_ok = getattr(removed, "exit_code", None) == 0
        except TimeoutError:
            state.timed_out = True
            state.official_message = "pre-run results cleanup timed out after 30s"
        except Exception as exc:
            state.official_message = (
                f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        started = time.monotonic()
        try:
            result = (
                await ws.run("python run.py", check=False, timeout=600)
                if not state.official_message
                else None
            )
            state.exit_code = getattr(result, "exit_code", None)
            output = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".lower()
            state.timed_out = state.timed_out or (
                state.exit_code is None
                and any(word in output for word in ("timeout", "timed out"))
            )
        except TimeoutError as exc:
            state.timed_out = True
            state.official_message = f"official run timed out after 600s: {exc}"
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or any(
                word in str(exc).lower() for word in ("timeout", "timed out")
            )
        finally:
            state.elapsed_s = time.monotonic() - started
            state.run_finished = True
        result_path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = bool(remove_ok and result_path.is_file() and not result_path.is_symlink())
        state.official_ok = bool(remove_ok and state.exit_code == 0 and not state.timed_out and state.elapsed_s is not None and state.elapsed_s <= 600 + EPS and state.result_file_is_new_regular)
        state.official_message = (f"{state.official_message}; " if state.official_message else "") + f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        state.scores["official_run"] = 1.0 if state.official_ok else 0.0
        parse_failure = ""
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                parse_failure = f"strict parse failed: {type(exc).__name__}: {exc}"
        shape = _shape_atoms(state.data, state.parse_ok)
        try:
            evidence = _evidence_atoms(state.data)
        except Exception as exc:
            evidence = _failed_atoms(
                "result_evidence", f"evidence check raised {type(exc).__name__}: {exc}"
            )
        state.scenarios = _index_records(state.data)
        try:
            coverage_score, coverage_defects = _coverage(state.data)
        except Exception as exc:
            coverage_score, coverage_defects = 0.0, [
                f"coverage check raised {type(exc).__name__}: {exc}"
            ]
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = _failed_atoms(
                "decision_legality", f"legality replay raised {type(exc).__name__}: {exc}"
            )
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.scores["canonical_coverage"] = coverage_score
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        can_replay = all(state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in ("result_json", "result_evidence", "canonical_coverage", "decision_legality"))
        if can_replay:
            try:
                state.replayed = {
                    key_: _replay(key_[0], key_[1], state.scenarios[key_])
                    for key_ in SCENARIO_KEYS
                }
            except Exception as exc:
                state.replayed = {}
                state.diagnostics["replay_validity"] = (
                    f"replay raised {type(exc).__name__}: {exc}"
                )
        else:
            state.replayed = {}
        try:
            replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        except Exception as exc:
            replay_atoms = _failed_atoms(
                "replay_validity", f"replay atom check raised {type(exc).__name__}: {exc}"
            )
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["result_json"] = (f"{parse_failure}; " if parse_failure else "") + _diagnose(shape)
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = f"defect_count={len(coverage_defects)}; " + " | ".join(coverage_defects[:12]) if coverage_defects else "defect_count=0"
        state.diagnostics["decision_legality"] = _diagnose(legality)
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in GATE_CHECKS)
        if state.validity_gate:
            try:
                state.metrics = _metrics(state.replayed)
            except Exception as exc:
                state.metrics = None
                state.validity_gate = False
                state.scores["replay_validity"] = 0.0
                state.diagnostics["replay_validity"] = (
                    "failed_atoms=[replay_validity.metrics]; "
                    f"replay_validity.metrics: metric calculation raised {type(exc).__name__}: {exc}"
                )
        else:
            state.metrics = {key: 0.0 for key in METRIC_KEYS}
        if not state.official_ok:
            for check_id in GATE_CHECKS:
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = f"official_run failed: {state.official_message}; {state.diagnostics.get(check_id, '')}"
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
    return (1.0, "static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "static compilation failed")


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
