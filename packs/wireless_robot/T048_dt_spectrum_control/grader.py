from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
HORIZON = 120
WINDOWS = 12
QUEUE_CAPACITY = 3840
PACKET_BITS = 960
MAX_ADMITTED_PACKETS = 80
SPECTRUM_UNITS = 12
INITIAL_ENERGY = 26.0
PACKET_ENERGY = 0.00025 * PACKET_BITS
DELTA = 0.25
LINKS = ("L_F1", "L_F2")
LINK_RANK = {link: index for index, link in enumerate(LINKS)}
ROBOTS = ("leader", "follower_1", "follower_2")
OFFSETS = {"follower_1": (-0.9, -0.6), "follower_2": (-0.9, 0.6)}
FOLLOWER_LINK = {"follower_1": "L_F1", "follower_2": "L_F2"}
LINK_FOLLOWER = {value: key for key, value in FOLLOWER_LINK.items()}
GAIN_INDEX = {"L_F1": 1, "L_F2": 2}

SCENARIOS: dict[str, dict[str, Any]] = {
    "formation_clearance": {
        "target": (18.0, 7.0),
        "gate": (10.0, 7.0),
        "gate_deadline": 68,
        "phase": 1,
        "obstacles": (
            (7.6, 8.4, 0.0, 5.1),
            (7.6, 8.4, 8.9, 14.0),
            (13.2, 14.0, 0.0, 3.5),
            (13.2, 14.0, 10.5, 14.0),
        ),
        "gains": (
            (0.90, 0.75, 0.65, 1.05, 0.55, 0.95, 0.70, 1.10, 0.60, 0.85, 0.75, 1.00),
            (0.70, 1.00, 0.60, 0.75, 1.10, 0.55, 0.95, 0.65, 1.05, 0.70, 0.85, 0.60),
        ),
    },
    "obstacle_detour": {
        "target": (18.0, 11.2),
        "gate": (14.0, 11.2),
        "gate_deadline": 82,
        "phase": 5,
        "obstacles": ((6.5, 12.8, 4.1, 9.9), (14.4, 15.1, 0.0, 7.0)),
        "gains": (
            (0.85, 0.55, 0.70, 0.60, 1.05, 0.75, 0.60, 1.10, 0.65, 0.90, 0.55, 0.80),
            (0.65, 1.05, 0.85, 0.70, 0.55, 0.95, 1.10, 0.65, 0.80, 0.60, 0.90, 0.75),
        ),
    },
    "shared_bottleneck": {
        "target": (18.0, 7.0),
        "gate": (10.8, 7.0),
        "gate_deadline": 72,
        "phase": 9,
        "obstacles": (
            (8.0, 9.0, 0.0, 5.0),
            (8.0, 9.0, 9.0, 14.0),
            (13.0, 13.8, 0.0, 5.2),
            (13.0, 13.8, 8.8, 14.0),
        ),
        "gains": (
            (0.70, 0.60, 1.05, 0.75, 0.55, 1.10, 0.65, 0.90, 0.60, 0.85, 1.00, 0.70),
            (1.00, 0.75, 0.60, 1.10, 0.65, 0.55, 0.95, 0.70, 1.05, 0.60, 0.80, 0.90),
        ),
    },
    "asymmetric_links": {
        "target": (18.0, 3.0),
        "gate": (13.0, 3.0),
        "gate_deadline": 86,
        "phase": 13,
        "obstacles": ((6.0, 10.0, 5.2, 8.8), (13.0, 13.7, 7.5, 14.0)),
        "gains": (
            (1.10, 1.00, 0.95, 0.80, 0.70, 1.05, 0.90, 0.75, 1.10, 0.85, 0.95, 0.80),
            (0.55, 0.60, 0.65, 0.70, 0.55, 0.60, 0.65, 0.55, 0.70, 0.60, 0.55, 0.65),
        ),
    },
    "spectrum_surge": {
        "target": (18.0, 7.0),
        "gate": (10.0, 7.0),
        "gate_deadline": 76,
        "phase": 17,
        "obstacles": (
            (7.8, 8.6, 0.0, 5.4),
            (7.8, 8.6, 8.6, 14.0),
            (13.5, 14.3, 0.0, 4.6),
            (13.5, 14.3, 9.4, 14.0),
        ),
        "gains": (
            (0.85, 0.75, 0.60, 1.10, 0.55, 0.65, 1.15, 0.70, 0.60, 1.05, 0.55, 0.90),
            (0.70, 1.05, 0.65, 0.60, 1.10, 0.55, 0.70, 1.15, 0.60, 0.65, 1.05, 0.55),
        ),
    },
}
SCENARIO_NAMES = tuple(SCENARIOS)
STARTS = {"leader": (1.5, 7.0), "follower_1": (0.6, 6.4), "follower_2": (0.6, 7.6)}
TARGET_OFFSETS = {"leader": (0.0, 0.0), **OFFSETS}
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.state_types": 0.14285714285714285,
        "result_evidence.slot_types": 0.14285714285714285,
        "result_evidence.allocation_types": 0.14285714285714285,
        "result_evidence.opportunity_types": 0.14285714285714285,
        "result_evidence.packet_types": 0.14285714285714285,
        "result_evidence.queue_types": 0.14285714285714285,
        "result_evidence.references": 0.14285714285714285,
    },
    "decision_legality": {
        "decision_legality.allocation_budget": 0.2,
        "decision_legality.opportunity_consistency": 0.2,
        "decision_legality.trajectory_constraints": 0.2,
        "decision_legality.command_dynamics": 0.2,
        "decision_legality.prediction_use": 0.2,
    },
    "replay_validity": {
        "replay_validity.allocation_timing": 0.25,
        "replay_validity.queue_service": 0.25,
        "replay_validity.packet_outcomes": 0.25,
        "replay_validity.robot_state": 0.25,
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
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=reject_duplicate)


def _shape_error(value: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object; actual_type={type(value).__name__}"
    if set(value) != keys:
        return f"{path}: expected keys={sorted(keys)}; actual_keys={sorted(value)}"
    return None


def _state_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"slot", *ROBOTS}, path)
    if error:
        return error
    if not isinstance(value["slot"], int) or isinstance(value["slot"], bool):
        return f"{path}.slot: expected integer"
    for robot in ROBOTS:
        item = value[robot]
        error = _shape_error(item, {"x", "y", "energy"}, f"{path}.{robot}")
        if error:
            return error
        if not all(_finite_number(item[key]) for key in ("x", "y", "energy")):
            return f"{path}.{robot}: expected finite coordinates and energy"
        if (
            not -EPS <= float(item["x"]) <= 20.0 + EPS
            or not -EPS <= float(item["y"]) <= 14.0 + EPS
        ):
            return f"{path}.{robot}: coordinate outside [0,20]x[0,14]"
        if not -EPS <= float(item["energy"]) <= INITIAL_ENERGY + EPS:
            return f"{path}.{robot}.energy: expected [0,26]; actual={item['energy']!r}"
    return None


def _velocity_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"vx", "vy"}, path)
    if error:
        return error
    if not all(_finite_number(value[key]) for key in ("vx", "vy")):
        return f"{path}: expected finite vx/vy"
    if (
        not -1.0 - EPS <= float(value["vx"]) <= 1.0 + EPS
        or not -1.0 - EPS <= float(value["vy"]) <= 1.0 + EPS
    ):
        return f"{path}: velocity component outside [-1,1]"
    return None


def _slot_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "slot",
            "leader_velocity",
            "follower_1_velocity",
            "follower_2_velocity",
            "follower_1_prediction_id",
            "follower_2_prediction_id",
        },
        path,
    )
    if error:
        return error
    if not isinstance(value["slot"], int) or isinstance(value["slot"], bool):
        return f"{path}.slot: expected integer"
    for robot in ROBOTS:
        error = _velocity_error(value[f"{robot}_velocity"], f"{path}.{robot}_velocity")
        if error:
            return error
    for robot in ("follower_1", "follower_2"):
        ref = value[f"{robot}_prediction_id"]
        if ref is not None and not isinstance(ref, str):
            return f"{path}.{robot}_prediction_id: expected string or null"
    return None


def _allocation_error(value: Any, path: str) -> str | None:
    keys = {
        "decision_window",
        "decision_slot",
        "effective_window",
        "follower_1_units",
        "follower_2_units",
        "observed_gain_follower_1",
        "observed_gain_follower_2",
    }
    error = _shape_error(value, keys, path)
    if error:
        return error
    for key in (
        "decision_window",
        "decision_slot",
        "effective_window",
        "follower_1_units",
        "follower_2_units",
    ):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            return f"{path}.{key}: expected integer"
    for key in ("observed_gain_follower_1", "observed_gain_follower_2"):
        if not _finite_number(value[key]) or not 0.55 - EPS <= float(value[key]) <= 1.15 + EPS:
            return f"{path}.{key}: expected finite gain in [0.55,1.15]"
    if any(value[key] not in range(13) for key in ("follower_1_units", "follower_2_units")):
        return f"{path}: spectrum units outside 0..12"
    return None


def _opportunity_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"link", "slot", "send", "packet_id", "trajectory"}, path)
    if error:
        return error
    if (
        value["link"] not in LINKS
        or not isinstance(value["slot"], int)
        or isinstance(value["slot"], bool)
        or value["slot"] % 2
    ):
        return f"{path}: expected link and even opportunity slot"
    if not isinstance(value["send"], bool):
        return f"{path}.send: expected boolean"
    if value["send"]:
        if not isinstance(value["packet_id"], str) or not value["packet_id"]:
            return f"{path}.packet_id: expected nonempty string when sending"
        trajectory = value["trajectory"]
        if not isinstance(trajectory, list) or len(trajectory) != 9:
            return f"{path}.trajectory: expected nine points"
        for index, point in enumerate(trajectory):
            error = _shape_error(point, {"x", "y"}, f"{path}.trajectory[{index}]")
            if error:
                return error
            if not _finite_number(point["x"]) or not _finite_number(point["y"]):
                return f"{path}.trajectory[{index}]: expected finite coordinates"
    elif value["packet_id"] is not None or value["trajectory"] is not None:
        return f"{path}: wait opportunity must have null packet_id and trajectory"
    return None


def _packet_error(value: Any, path: str) -> str | None:
    keys = {
        "packet_id",
        "link",
        "source_slot",
        "bits",
        "admitted",
        "queue_enter_slot",
        "service_start_slot",
        "service_complete_slot",
        "arrival_slot",
        "ack_slot",
        "outcome",
    }
    error = _shape_error(value, keys, path)
    if error:
        return error
    if not isinstance(value["packet_id"], str) or not value["packet_id"]:
        return f"{path}.packet_id: expected nonempty string"
    if value["link"] not in LINKS:
        return f"{path}.link: unknown link"
    if not isinstance(value["source_slot"], int) or value["source_slot"] not in range(0, 119, 2):
        return f"{path}.source_slot: expected even slot 0..118"
    if value["bits"] != PACKET_BITS or not isinstance(value["bits"], int):
        return f"{path}.bits: expected 960"
    if not isinstance(value["admitted"], bool):
        return f"{path}.admitted: expected boolean"
    if value["outcome"] not in {"delivered", "lost", "unfinished", "dropped"}:
        return f"{path}.outcome: unknown outcome"
    for key in (
        "queue_enter_slot",
        "service_start_slot",
        "service_complete_slot",
        "arrival_slot",
        "ack_slot",
    ):
        if value[key] is not None and (
            not isinstance(value[key], int) or isinstance(value[key], bool)
        ):
            return f"{path}.{key}: expected integer or null"
    for key in ("queue_enter_slot", "service_start_slot", "service_complete_slot"):
        if value[key] is not None and value[key] not in range(HORIZON):
            return f"{path}.{key}: expected slot in 0..119 or null"
    for key in ("arrival_slot", "ack_slot"):
        if value[key] is not None and value[key] not in range(1, HORIZON + 1):
            return f"{path}.{key}: expected boundary in 1..120 or null"
    queue_enter = value["queue_enter_slot"]
    service_start = value["service_start_slot"]
    service_complete = value["service_complete_slot"]
    arrival = value["arrival_slot"]
    ack = value["ack_slot"]
    outcome = value["outcome"]
    if not value["admitted"]:
        if outcome != "dropped" or any(
            item is not None
            for item in (queue_enter, service_start, service_complete, arrival, ack)
        ):
            return f"{path}: dropped packet must be unadmitted with all event slots null"
    elif outcome == "dropped" or queue_enter != value["source_slot"]:
        return f"{path}: admitted packet must enter at its source slot and cannot be dropped"
    elif outcome == "unfinished":
        if service_complete is not None or arrival is not None or ack is not None:
            return f"{path}: unfinished packet cannot have completion, arrival, or ACK"
    elif outcome in {"lost", "delivered"}:
        if not all(isinstance(item, int) for item in (service_start, service_complete, ack)):
            return f"{path}: completed packet requires service start, completion, and ACK"
        if not queue_enter <= service_start <= service_complete or ack != service_complete + 1:
            return f"{path}: packet service and ACK slots have an invalid order"
        if outcome == "lost" and arrival is not None:
            return f"{path}: lost packet must have null arrival"
        if outcome == "delivered" and arrival != ack:
            return f"{path}: delivered packet must have arrival_slot=ack_slot"
    return None


def _queue_error(value: Any, path: str) -> str | None:
    keys = {"slot", "link", "bits_start", "bits_end", "bits_admitted", "bits_served"}
    error = _shape_error(value, keys, path)
    if error:
        return error
    if value["link"] not in LINKS or not isinstance(value["slot"], int):
        return f"{path}: expected link and integer slot"
    for key in ("bits_start", "bits_end", "bits_admitted", "bits_served"):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            return f"{path}.{key}: expected integer"
    if (
        not 0 <= value["bits_start"] <= QUEUE_CAPACITY
        or not 0 <= value["bits_end"] <= QUEUE_CAPACITY
    ):
        return f"{path}: queue bits outside [0,3840]"
    if value["bits_admitted"] not in {0, PACKET_BITS} or not 0 <= value["bits_served"] <= 552:
        return f"{path}: invalid admitted/service bits"
    return None


def _mission_error(value: Any, path: str) -> str | None:
    keys = {
        "leader_goal_arrival_slot",
        "follower_1_goal_arrival_slot",
        "follower_2_goal_arrival_slot",
        "gate_success_slot",
        "collision_events",
        "energy_used",
        "mean_formation_error",
        "mean_used_prediction_age",
    }
    error = _shape_error(value, keys, path)
    if error:
        return error
    for key in (
        "leader_goal_arrival_slot",
        "follower_1_goal_arrival_slot",
        "follower_2_goal_arrival_slot",
        "gate_success_slot",
    ):
        if value[key] is not None and (
            not isinstance(value[key], int)
            or isinstance(value[key], bool)
            or not 0 <= value[key] < HORIZON
        ):
            return f"{path}.{key}: expected integer in 0..119 or null"
    if (
        not isinstance(value["collision_events"], int)
        or isinstance(value["collision_events"], bool)
        or not 0 <= value["collision_events"] <= 360
    ):
        return f"{path}.collision_events: expected integer in 0..360"
    for key in ("energy_used", "mean_formation_error"):
        if not _finite_number(value[key]) or float(value[key]) < -EPS:
            return f"{path}.{key}: expected finite nonnegative value"
    if float(value["energy_used"]) > 3 * INITIAL_ENERGY + EPS:
        return f"{path}.energy_used: expected value in [0,78]"
    age = value["mean_used_prediction_age"]
    if age is not None and (not _finite_number(age) or float(age) < -EPS):
        return f"{path}.mean_used_prediction_age: expected finite nonnegative value or null"
    return None


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    scenario_keys = {
        "scenario",
        "states",
        "slots",
        "allocation_decisions",
        "prediction_opportunities",
        "packets",
        "queue_states",
    }
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    values = {
        "parse": parse_ok,
        "root": isinstance(data, dict) and set(data) == {"scenarios"},
        "scenarios": isinstance(scenarios, list),
        "records": isinstance(scenarios, list)
        and all(isinstance(record, dict) and set(record) == scenario_keys for record in scenarios),
    }
    details = {
        "parse": "strict JSON parse succeeded" if parse_ok else "strict JSON parse failed",
        "root": "results.json root has exactly scenarios"
        if values["root"]
        else "results.json: expected root keys={'scenarios'}",
        "scenarios": "scenarios is an array"
        if values["scenarios"]
        else "results.json.scenarios: expected an array",
        "records": "all scenario records have exactly the declared top-level fields"
        if values["records"]
        else "results.json.scenarios: expected records with exactly the declared fields",
    }
    return [
        _atom("result_json", key, values[key], details[key])
        for key in ("parse", "root", "scenarios", "records")
    ]


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(data["scenarios"]):
        if (
            isinstance(record, dict)
            and isinstance(record.get("scenario"), str)
            and record["scenario"] in SCENARIOS
        ):
            result.setdefault(record["scenario"], record)
    return result


def _known_scenarios(records: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(name for name in SCENARIO_NAMES if name in records)


def _atom(check_id: str, rule: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{rule}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


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
        raise ValueError("atom share differs from declared share")
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


def _point_rect_distance(
    point: tuple[float, float], rect: tuple[float, float, float, float]
) -> float:
    x, y = point
    xmin, xmax, ymin, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def _segments_intersect(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> bool:
    c1, c2, c3, c4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((c1 > EPS and c2 < -EPS) or (c1 < -EPS and c2 > EPS)) and (
        (c3 > EPS and c4 < -EPS) or (c3 < -EPS and c4 > EPS)
    ):
        return True
    return (
        (abs(c1) <= EPS and _on_segment(a, b, c))
        or (abs(c2) <= EPS and _on_segment(a, b, d))
        or (abs(c3) <= EPS and _on_segment(c, d, a))
        or (abs(c4) <= EPS and _on_segment(c, d, b))
    )


def _point_segment_distance(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom <= EPS:
        return math.dist(point, a)
    u = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom))
    return math.dist(point, (a[0] + u * dx, a[1] + u * dy))


def _segment_segment_distance(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _segment_rect_distance(
    a: tuple[float, float], b: tuple[float, float], rect: tuple[float, float, float, float]
) -> float:
    xmin, xmax, ymin, ymax = rect
    edges = (
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    )
    if any(_segments_intersect(a, b, c, d) for c, d in edges):
        return 0.0
    return min(_segment_segment_distance(a, b, c, d) for c, d in edges)


def _clearance(
    point: tuple[float, float], obstacles: Sequence[tuple[float, float, float, float]]
) -> float:
    x, y = point
    values = [x, 20.0 - x, y, 14.0 - y]
    values.extend(_point_rect_distance(point, rect) for rect in obstacles)
    return min(values)


def _segment_valid(
    a: tuple[float, float],
    b: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
) -> bool:
    if (
        min(a[0], b[0]) < 0.4 - EPS
        or max(a[0], b[0]) > 19.6 + EPS
        or min(a[1], b[1]) < 0.4 - EPS
        or max(a[1], b[1]) > 13.6 + EPS
    ):
        return False
    return all(_segment_rect_distance(a, b, rect) >= 0.4 - EPS for rect in obstacles)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite_number(value) else default


def _trajectory_valid(name: str, trajectory: Any, source: tuple[float, float]) -> bool:
    if not isinstance(trajectory, list) or len(trajectory) != 9:
        return False
    points: list[tuple[float, float]] = []
    for point in trajectory:
        if (
            not isinstance(point, dict)
            or set(point) != {"x", "y"}
            or not _finite_number(point.get("x"))
            or not _finite_number(point.get("y"))
        ):
            return False
        points.append((float(point["x"]), float(point["y"])))
    if math.dist(points[0], source) > EPS:
        return False
    obstacles = SCENARIOS[name]["obstacles"]
    return all(_clearance(point, obstacles) >= 0.4 - EPS for point in points) and all(
        math.dist(a, b) <= 0.25 + EPS for a, b in zip(points, points[1:])
    )


def _current_gain(name: str, link: str, slot: int) -> float:
    return float(SCENARIOS[name]["gains"][GAIN_INDEX[link] - 1][slot // 10])


def _loss(name: str, link: str, slot: int) -> bool:
    rank = 1 if link == "L_F1" else 2
    return (17 * slot + 5 * SCENARIOS[name]["phase"] + 7 * rank) % 37 == 0


def _initial_allocation() -> tuple[int, int]:
    return (6, 6)


def _formation_error(positions: Mapping[str, tuple[float, float]]) -> float:
    values = []
    leader = positions["leader"]
    for robot in ("follower_1", "follower_2"):
        offset = OFFSETS[robot]
        actual = positions[robot]
        values.append(math.dist((actual[0] - leader[0], actual[1] - leader[1]), offset))
    return 0.5 * sum(values)


def _decision_map(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    slots = record.get("slots")
    allocations = record.get("allocation_decisions")
    opportunities = record.get("prediction_opportunities")
    if not all(isinstance(value, list) for value in (slots, allocations, opportunities)):
        return None
    if len(slots) != HORIZON or len(allocations) != 11 or len(opportunities) != 120:
        return None
    slot_map: dict[int, Any] = {}
    for row in slots:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("slot"), int)
            or row["slot"] in slot_map
        ):
            return None
        slot_map[row["slot"]] = row
    alloc_map: dict[int, Any] = {}
    for row in allocations:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("decision_window"), int)
            or row["decision_window"] in alloc_map
        ):
            return None
        alloc_map[row["decision_window"]] = row
    opp_map: dict[tuple[int, str], Any] = {}
    for row in opportunities:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("slot"), int)
            or row.get("link") not in LINKS
        ):
            return None
        key = (row["slot"], row["link"])
        if key in opp_map:
            return None
        opp_map[key] = row
    if (
        set(slot_map) != set(range(HORIZON))
        or set(alloc_map) != set(range(11))
        or set(opp_map) != {(slot, link) for slot in range(0, HORIZON, 2) for link in LINKS}
    ):
        return None
    return {"slots": slot_map, "allocations": alloc_map, "opportunities": opp_map}


def _velocity_from_row(row: Any, robot: str) -> tuple[float, float]:
    value = row.get(f"{robot}_velocity") if isinstance(row, dict) else None
    if not isinstance(value, dict):
        return (0.0, 0.0)
    return (_safe_float(value.get("vx")), _safe_float(value.get("vy")))


def _compare(actual: Any, expected: Any, path: str) -> tuple[bool, str]:
    if isinstance(expected, bool):
        ok = isinstance(actual, bool) and actual is expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if _finite_number(expected):
        ok = _finite_number(actual) and math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=EPS
        )
        return (
            ok,
            "" if ok else f"{path}: expected={expected!r}; actual={actual!r}; tolerance={EPS}",
        )
    if isinstance(expected, (str, type(None), int)):
        ok = actual == expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return (
                False,
                f"{path}: expected list length={len(expected)}; "
                f"actual_type={type(actual).__name__}",
            )
        for index, (a, e) in enumerate(zip(actual, expected)):
            ok, detail = _compare(a, e, f"{path}[{index}]")
            if not ok:
                return False, detail
        return True, ""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return (
                False,
                f"{path}: expected keys={sorted(expected)}; "
                f"actual_keys={sorted(actual) if isinstance(actual, dict) else None}",
            )
        for key in expected:
            ok, detail = _compare(actual[key], expected[key], f"{path}.{key}")
            if not ok:
                return False, detail
        return True, ""
    ok = actual == expected
    return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"


def _empty_trajectory(position: tuple[float, float]) -> list[dict[str, float]]:
    return [{"x": position[0], "y": position[1]} for _ in range(9)]


def _robot_target(name: str, robot: str) -> tuple[float, float]:
    target = SCENARIOS[name]["target"]
    offset = TARGET_OFFSETS[robot]
    return (target[0] + offset[0], target[1] + offset[1])


def _view_state(
    positions: Mapping[str, tuple[float, float]],
    energies: Mapping[str, float],
    previous: Mapping[str, tuple[float, float]],
    name: str,
    slot: int,
    allocations: tuple[int, int],
    queues: Mapping[str, list[dict[str, Any]]],
    packets: Mapping[str, dict[str, Any]],
    gains: Mapping[str, float],
    acks: Sequence[dict[str, Any]],
    gate_success: int | None,
) -> dict[str, Any]:
    def robot_view(robot: str) -> dict[str, Any]:
        return {
            "position": positions[robot],
            "energy": energies[robot],
            "previous_velocity": previous[robot],
            "target": _robot_target(name, robot),
        }

    queue_view = {
        link: tuple(
            {"packet_id": p["packet_id"], "remaining_bits": int(p["_remaining"])}
            for p in queues[link]
        )
        for link in LINKS
    }
    delivered: dict[str, list[dict[str, Any]]] = {link: [] for link in LINKS}
    for packet in packets.values():
        if (
            packet.get("outcome") == "delivered"
            and packet.get("arrival_slot") is not None
            and packet["arrival_slot"] <= slot
        ):
            delivered[packet["link"]].append(
                {
                    "packet_id": packet["packet_id"],
                    "source_slot": packet["source_slot"],
                    "trajectory": packet["trajectory"],
                }
            )
    for link in LINKS:
        delivered[link].sort(key=lambda row: (row["source_slot"], row["packet_id"]))
    common = {
        "slot": slot,
        "region": (0.0, 20.0, 0.0, 14.0),
        "static_obstacles": SCENARIOS[name]["obstacles"],
        "gate": SCENARIOS[name]["gate"],
        "desired_offsets": dict(OFFSETS),
        "allocations": {"follower_1_units": allocations[0], "follower_2_units": allocations[1]},
        "queue_summaries": queue_view,
        "acks": tuple(acks),
        "current_gains": dict(gains),
        "gate_success_slot": gate_success,
    }
    return {
        **common,
        "leader": {
            "slot": slot,
            "static_obstacles": SCENARIOS[name]["obstacles"],
            **robot_view("leader"),
            "queues": queue_view,
            "acks": tuple(acks),
            "delivered_predictions": delivered,
        },
        "follower_1": {
            "slot": slot,
            "static_obstacles": SCENARIOS[name]["obstacles"],
            **robot_view("follower_1"),
            "delivered_predictions": delivered["L_F1"],
            "acks": tuple(row for row in acks if row.get("link") == "L_F1"),
        },
        "follower_2": {
            "slot": slot,
            "static_obstacles": SCENARIOS[name]["obstacles"],
            **robot_view("follower_2"),
            "delivered_predictions": delivered["L_F2"],
            "acks": tuple(row for row in acks if row.get("link") == "L_F2"),
        },
        "edge": {
            "positions": dict(positions),
            "energies": dict(energies),
            "previous_velocities": dict(previous),
            "gate_success_slot": gate_success,
            "queue_summaries": queue_view,
            "acks": tuple(acks),
            "current_gains": dict(gains),
        },
    }


def _normalise_alloc(row: Any, current: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(row, dict):
        return current
    values = (row.get("follower_1_units"), row.get("follower_2_units"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return current
    return (max(0, min(12, values[0])), max(0, min(12, values[1])))


def _simulate_scenario(
    name: str,
    decisions: Mapping[str, Any] | None = None,
    provider: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if name not in SCENARIOS or (decisions is None) == (provider is None):
        raise ValueError("provide exactly one known scenario and one decision source")
    if decisions is not None:
        source = decisions
    else:
        source = None
    positions = dict(STARTS)
    energies = {robot: INITIAL_ENERGY for robot in ROBOTS}
    previous = {robot: (0.0, 0.0) for robot in ROBOTS}
    queues: dict[str, list[dict[str, Any]]] = {link: [] for link in LINKS}
    packets: dict[str, dict[str, Any]] = {}
    packet_order: list[str] = []
    acks: list[dict[str, Any]] = []
    allocations_by_window: dict[int, tuple[int, int]] = {0: _initial_allocation()}
    current_allocation = _initial_allocation()
    allocation_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    uses: list[dict[str, Any]] = []
    collision_events = 0
    goal_slots: dict[str, int | None] = {robot: None for robot in ROBOTS}
    gate_success: int | None = None
    admitted_count = 0
    legality = {
        key.removeprefix("decision_legality."): [True, ""]
        for key in ATOM_SHARES["decision_legality"]
    }

    def fail(rule: str, detail: str) -> None:
        if legality[rule][0]:
            legality[rule] = [False, detail]

    def decision_row(kind: str, key: Any, default: Any) -> Any:
        if source is not None:
            return source.get(kind, {}).get(key, default)
        return default

    def record_state(slot: int) -> None:
        states.append(
            {
                "slot": slot,
                **{
                    robot: {
                        "x": positions[robot][0],
                        "y": positions[robot][1],
                        "energy": energies[robot],
                    }
                    for robot in ROBOTS
                },
            }
        )

    record_state(0)
    for t in range(HORIZON):
        gains = {link: _current_gain(name, link, t) for link in LINKS}
        current_allocation = allocations_by_window.get(t // 10, current_allocation)
        if t % 10 == 0 and t // 10 <= 10:
            k = t // 10
            row = decision_row("allocations", k, {})
            if provider is not None:
                # Allocation is returned with the other actions by the edge policy.
                pass
            current_row = row if isinstance(row, dict) else {}
            proposed_allocation = _normalise_alloc(current_row, current_allocation)
            if proposed_allocation[0] + proposed_allocation[1] > SPECTRUM_UNITS:
                fail("allocation_budget", f"{name}.allocation_decisions[{k}]: shares exceed 12")
                proposed_allocation = (
                    min(12, proposed_allocation[0]),
                    min(12, proposed_allocation[1]),
                )
            if source is not None and isinstance(current_row, dict):
                if (
                    current_row.get("decision_slot") != t
                    or current_row.get("effective_window") != k + 1
                ):
                    fail(
                        "allocation_budget",
                        f"{name}.allocation_decisions[{k}]: "
                        "decision/effective window relation invalid",
                    )
            allocations_by_window[k + 1] = proposed_allocation
            allocation_rows.append(
                {
                    "decision_window": k,
                    "decision_slot": t,
                    "effective_window": k + 1,
                    "follower_1_units": proposed_allocation[0],
                    "follower_2_units": proposed_allocation[1],
                    "observed_gain_follower_1": gains["L_F1"],
                    "observed_gain_follower_2": gains["L_F2"],
                }
            )

        if provider is not None:
            views = _view_state(
                positions,
                energies,
                previous,
                name,
                t,
                current_allocation,
                queues,
                packets,
                gains,
                acks,
                gate_success,
            )
            action = dict(provider(t, views) or {})
            command_source = action.get("commands", {})
            send_source = action.get("opportunities", {})
            if t % 10 == 0 and t // 10 <= 10:
                alloc_action = action.get("allocation", {})
                if isinstance(alloc_action, Mapping):
                    proposed_allocation = _normalise_alloc(alloc_action, current_allocation)
                    if proposed_allocation[0] + proposed_allocation[1] > SPECTRUM_UNITS:
                        fail("allocation_budget", f"{name}: provider shares exceed 12")
                        proposed_allocation = (
                            min(12, proposed_allocation[0]),
                            min(12, proposed_allocation[1]),
                        )
                    allocations_by_window[t // 10 + 1] = proposed_allocation
                    allocation_rows[-1].update(
                        {
                            "follower_1_units": proposed_allocation[0],
                            "follower_2_units": proposed_allocation[1],
                        }
                    )
            slot_decision = {"commands": command_source, "opportunities": send_source}
        else:
            raw_slot = decision_row("slots", t, {})
            if not isinstance(raw_slot, Mapping):
                raw_slot = {}
            command_source = {
                robot: {
                    "velocity": raw_slot.get(f"{robot}_velocity"),
                    "prediction_id": raw_slot.get(f"{robot}_prediction_id"),
                }
                for robot in ROBOTS
            }
            slot_decision = {
                "commands": command_source,
                "opportunities": {
                    link: decision_row("opportunities", (t, link), {}) for link in LINKS
                },
            }

        opportunity_source = slot_decision["opportunities"]
        current_opps: dict[str, Any] = {}
        if t % 2 == 0:
            for link in LINKS:
                row = (
                    opportunity_source.get(link, {})
                    if isinstance(opportunity_source, Mapping)
                    else {}
                )
                if not isinstance(row, dict):
                    row = {}
                send = row.get("send")
                if not isinstance(send, bool):
                    fail(
                        "opportunity_consistency",
                        f"{name}.prediction_opportunities[{t},{link}].send: expected boolean",
                    )
                    send = False
                trajectory = row.get("trajectory") if send else None
                packet_id = row.get("packet_id") if send else None
                if send and (not isinstance(packet_id, str) or not packet_id):
                    fail(
                        "opportunity_consistency",
                        f"{name}.{link}@{t}: sent opportunity lacks packet ID",
                    )
                    packet_id = f"auto-{link}-{t}"
                if send and not _trajectory_valid(name, trajectory, positions["leader"]):
                    fail(
                        "trajectory_constraints",
                        f"{name}.{link}@{t}: trajectory violates source, clearance, or step bound",
                    )
                    trajectory = _empty_trajectory(positions["leader"])
                if not send:
                    trajectory = None
                    packet_id = None
                if send and packet_id in packets:
                    fail("opportunity_consistency", f"{name}: duplicate packet_id={packet_id!r}")
                    packet_id = f"auto-{link}-{t}"
                current_opps[link] = {
                    "link": link,
                    "slot": t,
                    "send": send,
                    "packet_id": packet_id,
                    "trajectory": trajectory,
                }
                opportunity_rows.append(current_opps[link])
        command_source = slot_decision["commands"]
        command_vectors: dict[str, tuple[float, float]] = {}
        references: dict[str, str | None] = {}
        no_motion_debit: set[str] = set()
        for robot in ROBOTS:
            row = command_source.get(robot, {}) if isinstance(command_source, Mapping) else {}
            if not isinstance(row, Mapping):
                row = {}
            raw = row.get("velocity", row)
            if (
                not isinstance(raw, Mapping)
                or not _finite_number(raw.get("vx"))
                or not _finite_number(raw.get("vy"))
            ):
                fail(
                    "command_dynamics",
                    f"{name}.slots[{t}].{robot}_velocity: expected finite velocity",
                )
                vector = (0.0, 0.0)
            else:
                vector = (float(raw["vx"]), float(raw["vy"]))
            ref = row.get("prediction_id") if robot in ("follower_1", "follower_2") else None
            references[robot] = ref if isinstance(ref, str) else None
            forced_fallback = False
            if robot in ("follower_1", "follower_2"):
                link = FOLLOWER_LINK[robot]
                packet = packets.get(references[robot] or "")
                valid_ref = bool(
                    packet
                    and packet.get("outcome") == "delivered"
                    and packet.get("link") == link
                    and packet.get("arrival_slot") is not None
                    and packet["arrival_slot"] <= t
                    and 0 <= t - packet["source_slot"] <= 8
                    and _trajectory_valid(
                        name,
                        packet.get("trajectory"),
                        positions["leader"]
                        if packet.get("source_slot") == t
                        else (
                            packet["trajectory"][0]["x"],
                            packet["trajectory"][0]["y"],
                        )
                        if packet.get("trajectory")
                        else positions["leader"],
                    )
                )
                if references[robot] is None:
                    if math.dist(vector, (0.0, 0.0)) > EPS:
                        fail(
                            "prediction_use",
                            f"{name}.slots[{t}].{robot}: nonzero command "
                            "without prediction reference",
                        )
                    vector = (0.0, 0.0)
                    forced_fallback = True
                elif not valid_ref:
                    fail(
                        "prediction_use",
                        f"{name}.slots[{t}].{robot}_prediction_id: "
                        "invalid or unavailable prediction",
                    )
                    vector = (0.0, 0.0)
                    forced_fallback = True
                else:
                    uses.append(
                        {
                            "slot": t,
                            "robot": robot,
                            "packet_id": references[robot],
                            "age": t - packet["source_slot"],
                            "error": math.dist(
                                (
                                    packet["trajectory"][t - packet["source_slot"]]["x"],
                                    packet["trajectory"][t - packet["source_slot"]]["y"],
                                ),
                                positions["leader"],
                            ),
                        }
                    )
            speed = math.hypot(*vector)
            acceleration = math.dist(vector, previous[robot])
            if speed > 1.0 + EPS or (not forced_fallback and acceleration > 0.35 + EPS):
                fail(
                    "command_dynamics",
                    f"{name}.slots[{t}].{robot}: speed/acceleration bound violated",
                )
                vector = (0.0, 0.0)
            command_vectors[robot] = vector

        # Admit packets after all decisions, in the fixed link order.
        queue_bits_start = {
            link: sum(int(item["_remaining"]) for item in queues[link]) for link in LINKS
        }
        admitted_by_link = {link: False for link in LINKS}
        for link in LINKS:
            if t % 2 != 0 or link not in current_opps or not current_opps[link]["send"]:
                continue
            opp = current_opps[link]
            packet_id = opp["packet_id"]
            queue_bits = sum(int(item["_remaining"]) for item in queues[link])
            admitted = (
                queue_bits + PACKET_BITS <= QUEUE_CAPACITY
                and admitted_count < MAX_ADMITTED_PACKETS
                and energies["leader"] >= PACKET_ENERGY
            )
            packet = {
                "packet_id": packet_id,
                "link": link,
                "source_slot": t,
                "bits": PACKET_BITS,
                "admitted": admitted,
                "queue_enter_slot": t if admitted else None,
                "service_start_slot": None,
                "service_complete_slot": None,
                "arrival_slot": None,
                "ack_slot": None,
                "outcome": "dropped" if not admitted else None,
                "_remaining": PACKET_BITS if admitted else 0,
                "trajectory": opp["trajectory"],
            }
            packets[packet_id] = packet
            packet_order.append(packet_id)
            if admitted:
                admitted_count += 1
                energies["leader"] -= PACKET_ENERGY
                queues[link].append(packet)
                admitted_by_link[link] = True

        # Packet admission precedes motion. Check command energy against the
        # post-admission balance so simultaneous sends cannot make a later
        # command drive the leader below zero energy.
        for robot in ROBOTS:
            vector = command_vectors[robot]
            cost = 0.04 * (vector[0] * vector[0] + vector[1] * vector[1]) + 0.02 * math.dist(
                vector, previous[robot]
            )
            if cost > energies[robot]:
                fail(
                    "command_dynamics",
                    f"{name}.slots[{t}].{robot}: command exceeds post-admission energy",
                )
                command_vectors[robot] = (0.0, 0.0)
                no_motion_debit.add(robot)

        for link in LINKS:
            bits_start = queue_bits_start[link]
            admitted_bits = PACKET_BITS if admitted_by_link[link] else 0
            budget = int(math.floor(40 * current_allocation[LINK_RANK[link]] * gains[link]))
            served = 0
            while budget > 0 and queues[link]:
                head = queues[link][0]
                if head["service_start_slot"] is None:
                    head["service_start_slot"] = t
                amount = min(budget, int(head["_remaining"]))
                head["_remaining"] -= amount
                budget -= amount
                served += amount
                if head["_remaining"] <= 0:
                    head["service_complete_slot"] = t
                    head["outcome"] = "lost" if _loss(name, link, t) else "delivered"
                    head["ack_slot"] = t + 1
                    head["arrival_slot"] = None if head["outcome"] == "lost" else t + 1
                    queues[link].pop(0)
                    acks.append(
                        {
                            "packet_id": head["packet_id"],
                            "link": link,
                            "outcome": head["outcome"],
                            "ack_slot": t + 1,
                        }
                    )
            bits_end = sum(int(item["_remaining"]) for item in queues[link])
            queue_rows.append(
                {
                    "slot": t,
                    "link": link,
                    "bits_start": bits_start,
                    "bits_end": bits_end,
                    "bits_admitted": admitted_bits,
                    "bits_served": served,
                }
            )

        slot_rows.append(
            {
                "slot": t,
                **{
                    f"{robot}_velocity": {
                        "vx": command_vectors[robot][0],
                        "vy": command_vectors[robot][1],
                    }
                    for robot in ROBOTS
                },
                "follower_1_prediction_id": references["follower_1"],
                "follower_2_prediction_id": references["follower_2"],
            }
        )
        for robot in ROBOTS:
            vector = command_vectors[robot]
            cost = (
                0.0
                if robot in no_motion_debit
                else 0.04 * (vector[0] * vector[0] + vector[1] * vector[1])
                + 0.02 * math.dist(vector, previous[robot])
            )
            energies[robot] -= cost
        segments = {
            robot: (
                positions[robot],
                (
                    positions[robot][0] + DELTA * command_vectors[robot][0],
                    positions[robot][1] + DELTA * command_vectors[robot][1],
                ),
            )
            for robot in ROBOTS
        }
        involved: set[str] = set()
        obstacles = SCENARIOS[name]["obstacles"]
        for robot, (a, b) in segments.items():
            if not _segment_valid(a, b, obstacles):
                involved.add(robot)
        for index, first in enumerate(ROBOTS):
            for second in ROBOTS[index + 1 :]:
                if _segment_segment_distance(*segments[first], *segments[second]) < 0.8 - EPS:
                    involved.update((first, second))
        collision_events += len(involved)
        for robot in ROBOTS:
            positions[robot] = segments[robot][0] if robot in involved else segments[robot][1]
            if (
                goal_slots[robot] is None
                and math.dist(positions[robot], _robot_target(name, robot)) <= 0.35 + EPS
            ):
                goal_slots[robot] = t
        if gate_success is None and 20 <= t <= int(SCENARIOS[name]["gate_deadline"]):
            if (
                math.dist(positions["leader"], SCENARIOS[name]["gate"]) <= 0.35 + EPS
                and _formation_error(positions) <= 0.55 + EPS
                and all(
                    math.dist(positions[a], positions[b]) >= 0.8 - EPS
                    for index, a in enumerate(ROBOTS)
                    for b in ROBOTS[index + 1 :]
                )
            ):
                gate_success = t
        previous = command_vectors
        record_state(t + 1)

    for packet in packets.values():
        if packet["admitted"] and packet["outcome"] is None:
            packet["outcome"] = "unfinished"
            packet["_remaining"] = max(0, int(packet["_remaining"]))
    gate_range = range(20, int(SCENARIOS[name]["gate_deadline"]) + 1)
    mean_form = sum(
        _formation_error(
            {robot: (states[t][robot]["x"], states[t][robot]["y"]) for robot in ROBOTS}
        )
        for t in gate_range
    ) / len(tuple(gate_range))
    age_values = [row["age"] for row in uses]
    energy_used = 3 * INITIAL_ENERGY - sum(energies.values())
    mission = {
        "leader_goal_arrival_slot": goal_slots["leader"],
        "follower_1_goal_arrival_slot": goal_slots["follower_1"],
        "follower_2_goal_arrival_slot": goal_slots["follower_2"],
        "gate_success_slot": gate_success,
        "collision_events": collision_events,
        "energy_used": energy_used,
        "mean_formation_error": mean_form,
        "mean_used_prediction_age": sum(age_values) / len(age_values) if age_values else None,
    }
    packets_out = []
    for packet_id in packet_order:
        packet = packets[packet_id]
        packets_out.append(
            {
                key: packet[key]
                for key in (
                    "packet_id",
                    "link",
                    "source_slot",
                    "bits",
                    "admitted",
                    "queue_enter_slot",
                    "service_start_slot",
                    "service_complete_slot",
                    "arrival_slot",
                    "ack_slot",
                    "outcome",
                )
            }
        )
    record = {
        "scenario": name,
        "states": states,
        "slots": slot_rows,
        "allocation_decisions": allocation_rows,
        "prediction_opportunities": opportunity_rows,
        "packets": packets_out,
        "queue_states": queue_rows,
        "mission": mission,
    }
    return {
        "record": record,
        "legality": legality,
        "uses": uses,
        "packets": packets,
        "allocations": allocations_by_window,
        "metrics": _scenario_metrics(name, record, uses, collision_events),
    }


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _scenario_metrics(
    name: str, record: Mapping[str, Any], uses: Sequence[Mapping[str, Any]], collision_events: int
) -> dict[str, float]:
    scenario = SCENARIOS[name]
    states = record["states"]
    slots = record["slots"]
    deadline = int(scenario["gate_deadline"])
    gate_slots = range(20, deadline + 1)
    targets = {robot: _robot_target(name, robot) for robot in ROBOTS}
    progress: dict[str, float] = {}
    for robot in ROBOTS:
        arrival = record["mission"][f"{robot}_goal_arrival_slot"]
        initial_distance = max(1.0, math.dist(STARTS[robot], targets[robot]))
        if isinstance(arrival, int) and arrival <= 112:
            progress[robot] = 1.0
        else:
            end = states[113][robot]
            progress[robot] = _clip(
                1.0 - math.dist((end["x"], end["y"]), targets[robot]) / initial_distance
            )
    individual = (
        0.4 * progress["leader"] + 0.3 * progress["follower_1"] + 0.3 * progress["follower_2"]
    )
    gate_value = 0.0
    gate_distance = max(1.0, math.dist(STARTS["leader"], scenario["gate"]))
    for t in gate_slots:
        pos = {robot: (states[t + 1][robot]["x"], states[t + 1][robot]["y"]) for robot in ROBOTS}
        pair_ok = all(
            math.dist(pos[first], pos[second]) >= 0.8 - EPS
            for index, first in enumerate(ROBOTS)
            for second in ROBOTS[index + 1 :]
        )
        value = (
            _clip(1.0 - math.dist(pos["leader"], scenario["gate"]) / gate_distance)
            * _clip(1.0 - _formation_error(pos) / 0.55)
            * (1.0 if pair_ok else 0.0)
        )
        gate_value = max(gate_value, value)
    task = 0.55 * individual + 0.45 * gate_value
    clear_samples = []
    safe_count = 0
    form_values = []
    for t in gate_slots:
        pos = {robot: (states[t][robot]["x"], states[t][robot]["y"]) for robot in ROBOTS}
        form_values.append(_clip(1.0 - _formation_error(pos) / 0.55))
        pair_ok = all(
            math.dist(pos[first], pos[second]) >= 0.8 - EPS
            for index, first in enumerate(ROBOTS)
            for second in ROBOTS[index + 1 :]
        )
        clear_ok = all(
            _clearance(pos[robot], scenario["obstacles"]) >= 0.4 - EPS for robot in ROBOTS
        )
        safe_count += int(pair_ok and clear_ok)
        clear_samples.extend(
            _clip((_clearance(pos[robot], scenario["obstacles"]) - 0.4) / 0.6) for robot in ROBOTS
        )
    safe = safe_count / len(tuple(gate_slots))
    valid_uses = [row for row in uses if row["slot"] in gate_slots and row["error"] <= 0.75 + EPS]
    valid = len(valid_uses) / (2 * len(tuple(gate_slots)))
    form = sum(form_values) / len(form_values)
    clear = sum(clear_samples) / len(clear_samples)
    primary_two = 0.45 * safe + 0.25 * valid + 0.30 * (0.5 * form + 0.5 * clear)
    variation = 0.0
    previous = {robot: (0.0, 0.0) for robot in ROBOTS}
    for row in slots:
        for robot in ROBOTS:
            current = (row[f"{robot}_velocity"]["vx"], row[f"{robot}_velocity"]["vy"])
            variation += math.dist(current, previous[robot])
            previous[robot] = current
    energy_used = float(record["mission"]["energy_used"])
    safe_motion = _clip(1.0 - collision_events / (3 * HORIZON))
    smooth = _clip(1.0 - variation / (0.35 * HORIZON * 3))
    energy_eff = _clip(1.0 - energy_used / 78.0)
    primary_three = task * (0.4 * safe_motion + 0.3 * smooth + 0.3 * energy_eff)
    accepted = [packet for packet in record["packets"] if packet["admitted"]]
    useful_ids = {row["packet_id"] for row in valid_uses}
    support = len(valid_uses) / (2 * len(tuple(gate_slots)))
    useful = len(useful_ids) / len(accepted) if accepted else 0.0
    auxiliary_one = support * (0.4 + 0.6 * useful)
    if useful_ids:
        useful_packets = [packet for packet in accepted if packet["packet_id"] in useful_ids]
        age = sum(int(row["age"]) for row in valid_uses) / len(valid_uses) if valid_uses else 0.0
        wait = sum(
            int(packet["service_start_slot"]) - int(packet["queue_enter_slot"])
            for packet in useful_packets
            if packet["service_start_slot"] is not None
        ) / len(useful_packets)
        fresh = support * _clip(1.0 - age / 8.0)
        wait_eff = support * _clip(1.0 - wait / HORIZON)
    else:
        fresh = 0.0
        wait_eff = 0.0
    h_numerator = {link: 0.0 for link in LINKS}
    h_denominator = {link: 0.0 for link in LINKS}
    for k in range(WINDOWS):
        t = 10 * k
        urgency: dict[str, float] = {}
        shares: dict[str, int] = {}
        for link in LINKS:
            follower = LINK_FOLLOWER[link]
            arrived = [
                packet
                for packet in record["packets"]
                if packet["link"] == link
                and packet["outcome"] == "delivered"
                and packet["arrival_slot"] is not None
                and packet["arrival_slot"] <= t
                and 0 <= t - packet["source_slot"] <= 8
            ]
            age_at = t - max(
                (int(packet["source_slot"]) for packet in arrived),
                default=t - 8,
            )
            position = (states[t][follower]["x"], states[t][follower]["y"])
            leader = (states[t]["leader"]["x"], states[t]["leader"]["y"])
            offset = OFFSETS[follower]
            formation_error = math.dist((position[0] - leader[0], position[1] - leader[1]), offset)
            urgency[link] = 1.0 + (1.0 if t in gate_slots else 0.0) * (
                1.0 + _clip(age_at / 8.0) + _clip(formation_error / 0.55)
            )
            shares[link] = (
                int(record["allocation_decisions"][k - 1][f"{follower}_units"]) if k > 0 else 6
            )
        total_urgency = sum(urgency.values())
        for link in LINKS:
            demand = 12.0 * urgency[link] / total_urgency
            h_numerator[link] += urgency[link] * min(1.0, shares[link] / demand)
            h_denominator[link] += urgency[link]
    h1 = h_numerator["L_F1"] / h_denominator["L_F1"]
    h2 = h_numerator["L_F2"] / h_denominator["L_F2"]
    balance = (
        0.0
        if h1 + h2 <= EPS
        else ((h1 + h2) / 2.0) * ((h1 + h2) ** 2 / (2.0 * (h1 * h1 + h2 * h2)))
    )
    auxiliary_two = 0.4 * fresh + 0.3 * wait_eff + 0.3 * support * balance
    return {
        "primary_metrics_1": task,
        "primary_metrics_2": primary_two,
        "primary_metrics_3": primary_three,
        "auxiliary_metrics_1": auxiliary_one,
        "auxiliary_metrics_2": auxiliary_two,
    }


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, list):
        return 0.0, ["results.scenarios: missing array"]
    seen: set[str] = set()
    scenario_order: list[str] = []
    for index, record in enumerate(scenarios):
        key = record.get("scenario") if isinstance(record, dict) else None
        if not isinstance(key, str) or key not in SCENARIOS:
            defects.append(f"results.scenarios[{index}]: unknown scenario key={key!r}")
        elif key in seen:
            defects.append(f"results.scenarios[{index}]: duplicate scenario key={key!r}")
        else:
            seen.add(key)
            scenario_order.append(key)
    for name in SCENARIO_NAMES:
        if name not in seen:
            defects.append(f"results.scenarios: missing scenario key={name!r}")
    if len(scenarios) == len(SCENARIO_NAMES) and seen == set(SCENARIO_NAMES):
        if scenario_order != list(SCENARIO_NAMES):
            defects.append("results.scenarios: scenario records are not in canonical order")
    expected: dict[str, set[Hashable]] = {}
    for name in SCENARIO_NAMES:
        expected[name] = {
            "states": set(range(HORIZON + 1)),
            "slots": set(range(HORIZON)),
            "allocation_decisions": set(range(11)),
            "prediction_opportunities": {
                (slot, link) for slot in range(0, HORIZON, 2) for link in LINKS
            },
            "queue_states": {(slot, link) for slot in range(HORIZON) for link in LINKS},
        }
    for name in SCENARIO_NAMES:
        records_for_name = [
            record
            for record in scenarios
            if isinstance(record, dict) and record.get("scenario") == name
        ]
        if not records_for_name:
            continue
        record = records_for_name[0]
        for field, wanted in expected[name].items():
            rows = record.get(field)
            if not isinstance(rows, list):
                defects.append(f"{name}.{field}: missing canonical array")
                continue
            keys: list[Hashable] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if field == "states" or field == "slots":
                    raw_key: Any = row.get("slot")
                elif field == "allocation_decisions":
                    raw_key = row.get("decision_window")
                else:
                    raw_key = (row.get("slot"), row.get("link"))
                # Keep malformed canonical keys diagnostic-only instead of
                # allowing an unhashable JSON value to abort grading.
                try:
                    hash(raw_key)
                except TypeError:
                    raw_key = ("<unhashable>", repr(raw_key))
                keys.append(raw_key)
            counts: dict[Hashable, int] = defaultdict(int)
            for key in keys:
                counts[key] += 1
            for key, count in counts.items():
                if key not in wanted:
                    defects.append(f"{name}.{field}: unknown key {key!r}")
                elif count > 1:
                    defects.append(f"{name}.{field}: duplicate key {key!r}")
            for key in wanted:
                if counts.get(key, 0) == 0:
                    defects.append(f"{name}.{field}: missing key {key!r}")
            if len(keys) == len(wanted) and set(keys) == wanted:
                if field in {"states", "slots"}:
                    ordered = list(range(HORIZON + (1 if field == "states" else 0)))
                elif field == "allocation_decisions":
                    ordered = list(range(11))
                else:
                    ordered = [
                        (slot, link)
                        for slot in range(HORIZON)
                        if field == "queue_states" or slot % 2 == 0
                        for link in LINKS
                    ]
                if keys != ordered:
                    defects.append(f"{name}.{field}: records are not in canonical order")
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _evidence_atoms(records: Mapping[str, Any]) -> list[AtomVerdict]:
    rules = [key.removeprefix("result_evidence.") for key in ATOM_SHARES["result_evidence"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(records):
        record = records[name]
        for index, row in enumerate(
            record.get("states", []) if isinstance(record.get("states"), list) else []
        ):
            error = _state_error(row, f"{name}.states[{index}]")
            if error:
                fail("state_types", error)
        for index, row in enumerate(
            record.get("slots", []) if isinstance(record.get("slots"), list) else []
        ):
            error = _slot_error(row, f"{name}.slots[{index}]")
            if error:
                fail("slot_types", error)
        for index, row in enumerate(
            record.get("allocation_decisions", [])
            if isinstance(record.get("allocation_decisions"), list)
            else []
        ):
            error = _allocation_error(row, f"{name}.allocation_decisions[{index}]")
            if error:
                fail("allocation_types", error)
        for index, row in enumerate(
            record.get("prediction_opportunities", [])
            if isinstance(record.get("prediction_opportunities"), list)
            else []
        ):
            error = _opportunity_error(row, f"{name}.prediction_opportunities[{index}]")
            if error:
                fail("opportunity_types", error)
        packets_value = record.get("packets")
        if not isinstance(packets_value, list):
            fail("packet_types", f"{name}.packets: expected an array")
            packets_value = []
        packet_ids: set[str] = set()
        packet_order: list[tuple[int, int, str]] = []
        for index, row in enumerate(packets_value):
            error = _packet_error(row, f"{name}.packets[{index}]")
            if error:
                fail("packet_types", error)
            elif row["packet_id"] in packet_ids:
                fail(
                    "packet_types",
                    f"{name}.packets[{index}].packet_id: duplicate id={row['packet_id']!r}",
                )
            else:
                packet_ids.add(row["packet_id"])
                packet_order.append((row["source_slot"], LINK_RANK[row["link"]], row["packet_id"]))
        if packet_order != sorted(packet_order):
            fail("packet_types", f"{name}.packets: records are not in canonical order")
        for index, row in enumerate(
            record.get("queue_states", []) if isinstance(record.get("queue_states"), list) else []
        ):
            error = _queue_error(row, f"{name}.queue_states[{index}]")
            if error:
                fail("queue_types", error)
        opportunities = record.get("prediction_opportunities")
        packets = record.get("packets")
        if isinstance(opportunities, list) and isinstance(packets, list):
            by_key = {
                (row.get("link"), row.get("source_slot"), row.get("packet_id")): row
                for row in packets
                if isinstance(row, dict)
                and row.get("link") in LINKS
                and isinstance(row.get("source_slot"), int)
                and isinstance(row.get("packet_id"), str)
            }
            for index, row in enumerate(opportunities):
                if not isinstance(row, dict) or not row.get("send"):
                    continue
                key = (row.get("link"), row.get("slot"), row.get("packet_id"))
                if key not in by_key:
                    fail(
                        "references",
                        f"{name}.prediction_opportunities[{index}]: "
                        "sent opportunity lacks matching packet",
                    )
            sent_keys = {
                (row.get("link"), row.get("slot"), row.get("packet_id"))
                for row in opportunities
                if isinstance(row, dict)
                and row.get("send")
                and row.get("link") in LINKS
                and isinstance(row.get("slot"), int)
                and isinstance(row.get("packet_id"), str)
            }
            packet_keys = {
                (row.get("link"), row.get("source_slot"), row.get("packet_id"))
                for row in packets
                if isinstance(row, dict)
                and row.get("link") in LINKS
                and isinstance(row.get("source_slot"), int)
                and isinstance(row.get("packet_id"), str)
            }
            if sent_keys != packet_keys:
                fail("references", f"{name}: sent opportunity and packet reference sets differ")
    return [
        _atom(
            "result_evidence",
            rule,
            passed[rule],
            details[rule] or "all canonical evidence fields have the declared form",
        )
        for rule in rules
    ]


def _decision_evidence_valid(record: Mapping[str, Any]) -> bool:
    validators = (
        ("slots", _slot_error),
        ("allocation_decisions", _allocation_error),
        ("prediction_opportunities", _opportunity_error),
    )
    for field, validator in validators:
        rows = record.get(field)
        if not isinstance(rows, list) or any(
            validator(row, f"{field}[{index}]") is not None for index, row in enumerate(rows)
        ):
            return False
    return True


def _legality_atoms(records: Mapping[str, Any]) -> list[AtomVerdict]:
    rules = [key.removeprefix("decision_legality.") for key in ATOM_SHARES["decision_legality"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(records):
        if not _decision_evidence_valid(records[name]):
            continue
        decisions = _decision_map(records[name])
        if decisions is None:
            continue
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception as exc:  # noqa: BLE001
            for rule in rules:
                fail(rule, f"{name}: legality replay raised {type(exc).__name__}: {exc}")
            continue
        for rule, value in replay["legality"].items():
            if not value[0]:
                fail(rule, f"{name}: {value[1]}")
    return [
        _atom(
            "decision_legality",
            rule,
            passed[rule],
            details[rule] or "all canonical decisions satisfy this constraint",
        )
        for rule in rules
    ]


def _replay_atoms(
    records: Mapping[str, Any],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(records):
        if not _decision_evidence_valid(records[name]):
            continue
        decisions = _decision_map(records[name])
        if decisions is None:
            continue
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception as exc:  # noqa: BLE001
            for rule in rules:
                fail(rule, f"{name}: replay raised {type(exc).__name__}: {exc}")
            continue
        replayed[name] = replay
        legal = {rule: bool(value[0]) for rule, value in replay["legality"].items()}
        actual = records[name]
        expected = replay["record"]
        if all(legal.values()):
            for rule, field, label in (
                ("allocation_timing", "allocation_decisions", "allocation_decisions"),
                ("queue_service", "queue_states", "queue_states"),
                ("packet_outcomes", "packets", "packets"),
                ("robot_state", "states", "states"),
            ):
                actual_value = actual.get(field)
                expected_value = expected.get(field)
                if not isinstance(actual_value, list):
                    continue
                validators = {
                    "allocation_decisions": _allocation_error,
                    "queue_states": _queue_error,
                    "packets": _packet_error,
                    "states": _state_error,
                }
                if any(
                    validators[field](row, f"{name}.{field}[{index}]") is not None
                    for index, row in enumerate(actual_value)
                ):
                    continue
                canonical_keys: dict[str, set[Hashable] | None] = {
                    "allocation_decisions": set(range(11)),
                    "queue_states": {(slot, link) for slot in range(HORIZON) for link in LINKS},
                    "packets": None,
                    "states": set(range(HORIZON + 1)),
                }
                keyers: dict[str, Callable[[Mapping[str, Any]], Hashable]] = {
                    "allocation_decisions": lambda row: row["decision_window"],
                    "queue_states": lambda row: (row["slot"], row["link"]),
                    "packets": lambda row: (
                        row["link"],
                        row["source_slot"],
                        row["packet_id"],
                    ),
                    "states": lambda row: row["slot"],
                }
                actual_keys = [keyers[field](row) for row in actual_value]
                wanted = canonical_keys[field]
                if wanted is not None and (
                    len(actual_keys) != len(wanted) or set(actual_keys) != wanted
                ):
                    continue
                if field == "packets":
                    opportunities = actual.get("prediction_opportunities")
                    if not isinstance(opportunities, list):
                        continue
                    sent_keys = {
                        (row["link"], row["slot"], row["packet_id"])
                        for row in opportunities
                        if isinstance(row, dict)
                        and row.get("send") is True
                        and row.get("link") in LINKS
                        and isinstance(row.get("slot"), int)
                        and isinstance(row.get("packet_id"), str)
                    }
                    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != sent_keys:
                        continue
                sorters: dict[str, Callable[[Mapping[str, Any]], Any]] = {
                    "allocation_decisions": lambda row: row["decision_window"],
                    "queue_states": lambda row: (row["slot"], LINK_RANK[row["link"]]),
                    "packets": lambda row: (
                        row["source_slot"],
                        LINK_RANK[row["link"]],
                        row["packet_id"],
                    ),
                    "states": lambda row: row["slot"],
                }
                ordered_actual = sorted(actual_value, key=sorters[field])
                ordered_expected = sorted(expected_value, key=sorters[field])
                ok, detail = _compare(ordered_actual, ordered_expected, f"{name}.{label}")
                if not ok:
                    fail(rule, detail)
    return [
        _atom(
            "replay_validity",
            rule,
            passed[rule],
            details[rule] or "fixed-model replay matched the submitted state and event evidence",
        )
        for rule in rules
    ], replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all canonical scenario replays are required for metrics")
    metric_names = (
        "primary_metrics_1",
        "primary_metrics_2",
        "primary_metrics_3",
        "auxiliary_metrics_1",
        "auxiliary_metrics_2",
    )
    values = {
        metric: sum(float(replayed[name]["metrics"][metric]) for name in SCENARIO_NAMES)
        / len(SCENARIO_NAMES)
        for metric in metric_names
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return values


def _gated_metric(metrics: Mapping[str, float] | None, name: str) -> float:
    if metrics is None:
        return 0.0
    value = metrics.get(name, 0.0)
    return float(value) if _finite_number(value) and 0.0 <= float(value) <= 1.0 else 0.0


def _simulate_policy(
    name: str, provider: Callable[[int, Mapping[str, Any]], Mapping[str, Any]]
) -> dict[str, Any]:
    return _simulate_scenario(name, provider=provider)


def _submission_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "scenario",
            "states",
            "slots",
            "allocation_decisions",
            "prediction_opportunities",
            "packets",
            "queue_states",
        )
    }


def _headings(text: str) -> dict[str, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    raw_lines = text.splitlines()
    lines = [""] * len(raw_lines)
    in_fence = False
    fence_marker = ""
    in_comment = False
    for index, raw_line in enumerate(raw_lines):
        if in_fence:
            stripped = raw_line.lstrip()
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        remainder = raw_line
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
        lines[index] = kept
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match or match.group(2).strip() not in required:
            continue
        level = len(match.group(1))
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            next_heading = re.match(r"^(#{1,6})\s+.+", lines[cursor])
            if next_heading and len(next_heading.group(1)) <= level:
                break
            body.append(lines[cursor])
            cursor += 1
        found.setdefault(match.group(2).strip(), "\n".join(body).strip())
    return found


def _report_score(text: str | None) -> tuple[float, str]:
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    return score, f"missing_sections={missing}; normalized={score:.9g}"


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = False
    try:
        state.run_started = True
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(removed, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
            state.official_message = (
                f"could not remove previous results.json: {type(exc).__name__}: {exc}"
            )
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = (
                state.exit_code is None
                or "timeout" in stderr.lower()
                or "timed out" in stderr.lower()
            )
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.timed_out = state.timed_out or state.elapsed_s >= 600.0 - EPS
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        if not state.official_message:
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok
                else (
                    f"official run failed: exit={state.exit_code}, "
                    f"elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, "
                    f"new_regular_results={state.result_file_is_new_regular}"
                )
            )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except Exception as exc:  # noqa: BLE001
                state.diagnostics["result_json"] = (
                    f"results.json: strict parse failed: {type(exc).__name__}: {exc}"
                )
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            for check_id in (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = reason
            _CACHE[key] = state
            return state
        state.records = _index_records(state.data)
        evidence_atoms = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(
            evidence_atoms, ATOM_SHARES["result_evidence"]
        )
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = (
            "failed_atoms=[]; defect_count=0"
            if not defects
            else f"defect_count={len(defects)}; " + " | ".join(defects[:6])
        )
        legality_atoms = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(
            legality_atoms, ATOM_SHARES["decision_legality"]
        )
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
        replay_atoms, replayed = _replay_atoms(state.records)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
        gate_names = (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        )
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in gate_names
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001
        message = f"grader analysis failed: {type(exc).__name__}: {exc}"
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, message)
    finally:
        state.run_finished = True
        _CACHE[key] = state
    return state


async def _metric_check(ws: Workspace, metric: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{metric}: validity gate not satisfied; metric not computed"
    value = _gated_metric(state.metrics, metric)
    return value, f"{metric}: independently replayed value={value:.9g}"


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected a regular file"
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"run.py: static compilation raised {type(exc).__name__}: {exc}"
    if getattr(result, "exit_code", None) == 0:
        return 1.0, "run.py: static compilation passed"
    return 0.0, "run.py: static compilation failed"


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a readable nonempty regular file"
    try:
        return _report_score(ws.read_text("report.md"))
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_json", 0.0), state.diagnostics.get("result_json", "")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_evidence", 0.0), state.diagnostics.get("result_evidence", "")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("canonical_coverage", 0.0), state.diagnostics.get(
        "canonical_coverage", ""
    )


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("decision_legality", 0.0), state.diagnostics.get(
        "decision_legality", ""
    )


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("replay_validity", 0.0), state.diagnostics.get("replay_validity", "")


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
