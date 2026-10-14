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
HORIZON = 72
N_ROBOTS = 2
GRID_X = 18
GRID_Y = 14
ENERGY0 = 90.0
SENSE_RADIUS = 3.0
RESULT_MAX_AGE = 4
CONFIDENCE_THRESHOLD = 0.75
EDGE_COORDINATE = (9, 7)
FRAGMENT_BITS = 480
UPLINK_CAPACITY = 960
EDGE_BIT_BUDGET = 9600
EDGE_QUEUE_CAPACITY = 3

SCENARIO_NAMES = (
    "clear_rows",
    "occluded_cluster",
    "edge_queue_burst",
    "weak_link_turn",
    "deadline_competition",
)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
ROBOTS = ("r0", "r1")
TARGET_IDS = tuple(f"T{i}" for i in range(8))
TARGET_VALUE_DEADLINE = {
    "T0": (1, 24),
    "T1": (1, 30),
    "T2": (2, 36),
    "T3": (2, 42),
    "T4": (3, 50),
    "T5": (3, 56),
    "T6": (4, 64),
    "T7": (4, 70),
}
KINDS = ("idle", "move", "submit", "cancel", "retry", "harvest")
DIRECTIONS = ("N", "E", "S", "W")
DELTAS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
MODES = ("local", "edge_compact", "edge_full")
FRAGMENT_COUNTS = {"edge_compact": 1, "edge_full": 2}
SERVICE_BITS = {"edge_compact": 480.0, "edge_full": 960.0}
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
        "decision_legality.visible_observations": 0.20,
        "decision_legality.job_capacity": 0.12,
        "decision_legality.job_references": 0.16,
        "decision_legality.retry_window": 0.18,
        "decision_legality.harvest_results": 0.18,
        "decision_legality.energy": 0.16,
    },
    "replay_validity": {
        "replay_validity.observation_processing": 0.18,
        "replay_validity.uplink": 0.20,
        "replay_validity.queue": 0.18,
        "replay_validity.harvest": 0.18,
        "replay_validity.motion_energy": 0.16,
        "replay_validity.terminal_metrics": 0.10,
    },
}


def _scene(
    starts: tuple[tuple[int, int], tuple[int, int]],
    rectangles: tuple[tuple[int, int, int, int], ...],
    coordinates: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    return {
        "starts": starts,
        "rectangles": rectangles,
        "targets": {target: coordinates[index] for index, target in enumerate(TARGET_IDS)},
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    "clear_rows": _scene(
        ((1, 2), (1, 11)),
        (),
        ((3, 2), (3, 11), (7, 3), (7, 10), (12, 2), (12, 11), (16, 4), (16, 9)),
    ),
    "occluded_cluster": _scene(
        ((2, 7), (15, 7)),
        ((8, 9, 4, 10),),
        ((3, 2), (3, 11), (12, 2), (12, 11), (7, 3), (7, 10), (16, 4), (16, 9)),
    ),
    "edge_queue_burst": _scene(
        ((2, 2), (2, 11)),
        ((6, 7, 5, 9),),
        ((3, 11), (7, 3), (7, 10), (12, 2), (12, 11), (16, 4), (16, 9), (3, 2)),
    ),
    "weak_link_turn": _scene(
        ((1, 6), (16, 6)),
        ((8, 9, 0, 4), (8, 9, 9, 13)),
        ((3, 11), (3, 2), (7, 10), (7, 3), (12, 2), (12, 11), (16, 4), (16, 9)),
    ),
    "deadline_competition": _scene(
        ((2, 3), (2, 10)),
        ((5, 6, 5, 8),),
        ((3, 2), (7, 3), (12, 2), (16, 4), (3, 11), (7, 10), (12, 11), (16, 9)),
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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _u64(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2**64


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

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _idle(slot: int, robot: int) -> dict[str, Any]:
    return {
        "slot": slot,
        "robot": f"r{robot}",
        "kind": "idle",
        "direction": None,
        "target": None,
        "mode": None,
        "observation_id": None,
        "job_id": None,
    }


def _action_issue(action: Any, path: str) -> str | None:
    expected = {
        "slot",
        "robot",
        "kind",
        "direction",
        "target",
        "mode",
        "observation_id",
        "job_id",
    }
    if not isinstance(action, dict):
        return f"{path}: expected object, actual={type(action).__name__}"
    if set(action) != expected:
        return f"{path}: expected exact fields={sorted(expected)!r}, actual={sorted(action)!r}"
    if not _integer(action.get("slot")):
        return f"{path}.slot: expected JSON integer, actual={action.get('slot')!r}"
    if not isinstance(action.get("robot"), str):
        return f"{path}.robot: expected string canonical-key component, actual={action.get('robot')!r}"
    if action.get("kind") not in KINDS:
        return f"{path}.kind: expected one of {KINDS!r}, actual={action.get('kind')!r}"
    kind = action["kind"]
    direction = action.get("direction")
    target = action.get("target")
    mode = action.get("mode")
    observation_id = action.get("observation_id")
    job_id = action.get("job_id")
    if kind == "idle":
        if not all(value is None for value in (direction, target, mode, observation_id, job_id)):
            return f"{path}: idle requires direction,target,mode,observation_id,job_id all null"
        return None
    if kind == "move":
        if direction not in DIRECTIONS:
            return f"{path}.direction: move expects one of {DIRECTIONS!r}, actual={direction!r}"
        if not all(value is None for value in (target, mode, observation_id, job_id)):
            return f"{path}: move requires target,mode,observation_id,job_id all null"
        return None
    if kind == "submit":
        if target not in TARGET_IDS:
            return f"{path}.target: submit expects one of {TARGET_IDS!r}, actual={target!r}"
        if mode not in MODES:
            return f"{path}.mode: submit expects one of {MODES!r}, actual={mode!r}"
        if not isinstance(observation_id, str) or not observation_id:
            return f"{path}.observation_id: submit expects non-empty string, actual={observation_id!r}"
        if direction is not None or job_id is not None:
            return f"{path}: submit requires direction and job_id null"
        return None
    if kind in ("cancel", "retry"):
        if not isinstance(job_id, str) or not job_id:
            return f"{path}.job_id: {kind} expects non-empty string, actual={job_id!r}"
        if not all(value is None for value in (direction, target, mode, observation_id)):
            return f"{path}: {kind} requires direction,target,mode,observation_id all null"
        return None
    if target not in TARGET_IDS:
        return f"{path}.target: harvest expects one of {TARGET_IDS!r}, actual={target!r}"
    if not all(value is None for value in (direction, mode, observation_id, job_id)):
        return f"{path}: harvest requires direction,mode,observation_id,job_id all null"
    return None


def _action_shape(action: Any) -> bool:
    return _action_issue(action, "action") is None


def _scenario_issue(row: Any, path: str) -> str | None:
    expected = {"scenario", "variant", "actions"}
    if not isinstance(row, dict):
        return f"{path}: expected object, actual={type(row).__name__}"
    if set(row) != expected:
        return f"{path}: expected exact fields={sorted(expected)!r}, actual={sorted(row)!r}"
    if not isinstance(row.get("scenario"), str):
        return f"{path}.scenario: expected string, actual={row.get('scenario')!r}"
    if not _integer(row.get("variant")):
        return f"{path}.variant: expected JSON integer, actual={row.get('variant')!r}"
    if not isinstance(row.get("actions"), list):
        return f"{path}.actions: expected array, actual={type(row.get('actions')).__name__}"
    return None


def _shape_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_ok = isinstance(data, dict) and set(data) == {"scenarios"}
    scenarios_ok = root_ok and isinstance(data.get("scenarios"), list)
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parsed" if parse_ok else "strict JSON parse failed"),
        _atom("result_json", "root", root_ok, f"root_keys={sorted(data) if isinstance(data, dict) else type(data).__name__}"),
        _atom("result_json", "scenario_shape", scenarios_ok, "scenarios is an array" if scenarios_ok else "scenarios is not an array"),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, dict) else None
    scenario_issue = None
    if not isinstance(rows, list):
        scenario_issue = f"scenarios: expected array, actual={type(rows).__name__}"
    else:
        for row_index, row in enumerate(rows):
            scenario_issue = _scenario_issue(row, f"scenarios[{row_index}]")
            if scenario_issue is not None:
                break
    scenario_ok = scenario_issue is None
    action_issue = None
    if scenario_ok and isinstance(rows, list):
        for row_index, row in enumerate(rows):
            for action_index, action in enumerate(row["actions"]):
                action_issue = _action_issue(
                    action, f"scenarios[{row_index}].actions[{action_index}]"
                )
                if action_issue is not None:
                    break
            if action_issue is not None:
                break
    elif scenario_issue is not None:
        action_issue = f"action records unavailable because {scenario_issue}"
    action_ok = action_issue is None
    return [
        _atom("result_evidence", "scenario_fields", scenario_ok, "scenario rows have exact fields" if scenario_ok else str(scenario_issue)),
        _atom("result_evidence", "action_fields", action_ok, "action records have exact action-specific fields" if action_ok else str(action_issue)),
    ]


def _index(data: Any) -> dict[tuple[str, int], Any]:
    indexed: dict[tuple[str, int], Any] = {}
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return indexed
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            hash(key)
        except TypeError:
            continue
        indexed.setdefault(key, row)
    return indexed


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
            defects.append(f"unhashable scenario key={key!r}")
    for key in SCENARIO_KEYS:
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"missing scenario={key[0]}|variant={key[1]}")
        elif count > 1:
            defects.extend([f"duplicate scenario={key!r}"] * (count - 1))
    defects.extend(f"unknown scenario key={key!r}" for key in counts if key not in SCENARIO_KEYS)
    expected = {(slot, robot) for slot in range(HORIZON) for robot in ROBOTS}
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
                defects.append(f"{key} unhashable action key={action_key!r}")
        for action_key in expected:
            count = action_counts.get(action_key, 0)
            if count == 0:
                defects.append(f"{key} missing action={action_key!r}")
            elif count > 1:
                defects.extend([f"{key} duplicate action={action_key!r}"] * (count - 1))
        defects.extend(f"{key} unknown action={action_key!r}" for action_key in action_counts if action_key not in expected)
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _actions_by_slot(row: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    actions = row.get("actions", ()) if isinstance(row, Mapping) else ()
    if not isinstance(actions, list):
        return result
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        slot = action.get("slot")
        robot = action.get("robot")
        if _integer(slot) and robot in ROBOTS:
            result.setdefault((slot, int(str(robot)[1:])), action)
    return result


def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _point_in_rectangle(point: tuple[int, int], rectangle: tuple[int, int, int, int]) -> bool:
    xmin, xmax, ymin, ymax = rectangle
    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax


def _segment_intersects_rectangle(
    first: tuple[int, int], second: tuple[int, int], rectangle: tuple[int, int, int, int]
) -> bool:
    xmin, xmax, ymin, ymax = rectangle
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    low = 0.0
    high = 1.0
    for p, q in ((-dx, first[0] - xmin), (dx, xmax - first[0]), (-dy, first[1] - ymin), (dy, ymax - first[1])):
        if abs(p) <= EPS:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low > high + EPS:
            return False
    return True


def _traversable(name: str, point: tuple[int, int]) -> bool:
    return 0 <= point[0] < GRID_X and 0 <= point[1] < GRID_Y and not any(
        _point_in_rectangle(point, rectangle) for rectangle in SCENARIOS[name]["rectangles"]
    )


def _observation_id(name: str, variant: int, robot: int, target: str, slot: int) -> str:
    return f"obs={name}|variant={variant}|robot={robot}|target={target}|slot={slot}"


def _job_id(name: str, variant: int, robot: int, slot: int) -> str:
    return f"job={name}|variant={variant}|robot={robot}|slot={slot}"


def _observation(
    name: str,
    variant: int,
    robot: int,
    target: str,
    slot: int,
    robot_position: tuple[int, int],
) -> dict[str, Any]:
    target_position = SCENARIOS[name]["targets"][target]
    d_o = _clip(_distance(robot_position, target_position) / SENSE_RADIUS)
    occluded = any(
        _segment_intersects_rectangle(robot_position, target_position, rectangle)
        for rectangle in SCENARIOS[name]["rectangles"]
    )
    return {
        "observation_id": _observation_id(name, variant, robot, target, slot),
        "robot": robot,
        "target": target,
        "observation_slot": slot,
        "robot_position": robot_position,
        "target_position": target_position,
        "d_o": d_o,
        "o": int(occluded),
    }


def _observation_quality(observation: Mapping[str, Any], variant: int, slot: int) -> float:
    age = slot - int(observation["observation_slot"])
    return _clip(1.0 - 0.12 * float(observation["d_o"]) - 0.20 * int(observation["o"]) - 0.04 * age - 0.04 * variant)


def _confidence(mode: str, quality: float) -> float:
    if mode == "local":
        return _clip(0.40 + 0.36 * quality)
    if mode == "edge_compact":
        return _clip(0.55 + 0.28 * quality)
    return _clip(0.72 + 0.22 * quality)


def _packet_sample(name: str, variant: int, fragment_id: str, attempt: int) -> float:
    return _u64(f"{name}|{variant}|{fragment_id}|a{attempt}|salt={5 * variant}")


def _service_rate(name: str, variant: int, slot: int) -> float:
    if name == "edge_queue_burst" and 24 + 3 * variant <= slot <= 35 + 3 * variant:
        return 240.0
    return 480.0


def _link_penalty(name: str, variant: int, slot: int, position: tuple[int, int]) -> float:
    if name == "weak_link_turn" and 7 <= position[0] <= 10:
        return 0.15
    if name == "deadline_competition" and 12 + 3 * variant <= slot <= 19 + 3 * variant:
        return 0.20
    return 0.0


def _copy_public_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "target",
            "mode",
            "observation_id",
            "submit_slot",
            "deadline",
            "status",
            "next_fragment",
            "failure_due",
            "queue_entry",
            "service_start",
        )
    }


def _copy_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: observation[key]
        for key in (
            "observation_id",
            "target",
            "observation_slot",
            "robot_position",
            "target_position",
            "d_o",
            "o",
        )
    }


def _copy_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "job_id",
            "target",
            "mode",
            "observation_id",
            "observation_slot",
            "completion_boundary",
            "confidence",
        )
    }


def _replay(name: str, variant: int, row: Mapping[str, Any]) -> dict[str, Any]:
    scene = SCENARIOS[name]
    submitted = _actions_by_slot(row)
    positions: list[list[tuple[int, int]]] = [[tuple(point) for point in scene["starts"]]]
    energies: list[list[float]] = [[ENERGY0 for _ in range(N_ROBOTS)]]
    cached_observations: list[dict[str, dict[str, Any]]] = [{} for _ in range(N_ROBOTS)]
    pending: list[set[str]] = [set() for _ in range(N_ROBOTS)]
    robot_results: list[list[dict[str, Any]]] = [[] for _ in range(N_ROBOTS)]
    known_harvested: list[set[str]] = [set() for _ in range(N_ROBOTS)]
    jobs: dict[str, dict[str, Any]] = {}
    outcomes: dict[int, list[dict[str, Any]]] = {}
    waiting: deque[str] = deque()
    active: str | None = None
    observations: list[dict[str, Any]] = []
    transmissions: list[dict[str, Any]] = []
    queue_events: list[dict[str, Any]] = []
    queue_snapshots: list[dict[str, Any]] = []
    service_records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    harvests: list[dict[str, Any]] = []
    harvested: dict[str, dict[str, Any]] = {}
    move_events: list[dict[str, Any]] = []
    energy_charges: list[dict[str, Any]] = []
    action_start_snapshots: list[list[dict[str, Any]]] = []
    legality: dict[str, list[str]] = {
        "visible_observations": [],
        "job_capacity": [],
        "job_references": [],
        "retry_window": [],
        "harvest_results": [],
        "energy": [],
    }
    safety_events = 0
    successful_moves = 0
    total_edge_bits = 0
    total_local_compute_energy = 0.0

    def terminate(job: dict[str, Any], status: str, boundary: int) -> None:
        nonlocal active, waiting
        job["status"] = status
        job["terminal_boundary"] = boundary
        pending[job["owner"]].discard(job["job_id"])
        if active == job["job_id"]:
            active = None
        if job["job_id"] in waiting:
            waiting = deque(item for item in waiting if item != job["job_id"])

    def complete(job: dict[str, Any], boundary: int) -> None:
        job["status"] = "completed"
        job["completion_boundary"] = boundary
        pending[job["owner"]].discard(job["job_id"])
        result = {
            "job_id": job["job_id"],
            "owner": job["owner"],
            "target": job["target"],
            "mode": job["mode"],
            "observation_id": job["observation_id"],
            "observation_slot": job["observation_slot"],
            "completion_boundary": boundary,
            "confidence": job["confidence"],
        }
        results.append(result)
        robot_results[job["owner"]].append(result)

    for boundary in range(HORIZON + 1):
        current_positions = positions[boundary]
        current_energies = list(energies[boundary])

        if active is not None and jobs[active]["service_remaining"] <= EPS:
            completed_job = jobs[active]
            active = None
            complete(completed_job, boundary)

        for job in sorted(jobs.values(), key=lambda item: item["job_id"]):
            if job["status"] == "local_pending" and job["completion_due"] == boundary:
                complete(job, boundary)

        upload_complete: list[dict[str, Any]] = []
        current_failures: list[list[dict[str, Any]]] = [[] for _ in range(N_ROBOTS)]
        for event in sorted(outcomes.pop(boundary, []), key=lambda item: item["offer_key"]):
            job = jobs.get(event["job_id"])
            if job is None or job.get("status") != "transmitting" or job.get("inflight_event") != event["event_id"]:
                continue
            job["inflight_event"] = None
            if event["admitted"] and event["success"]:
                if event["fragment_index"] + 1 < job["fragment_count"]:
                    job["next_fragment"] = event["fragment_index"] + 1
                    job["status"] = "ready"
                    job["ready_boundary"] = boundary
                else:
                    job["next_fragment"] = job["fragment_count"]
                    job["status"] = "upload_complete"
                    upload_complete.append(job)
            elif event["attempt"] == 0:
                job["status"] = "await_retry"
                job["failure_due"] = boundary
                feedback = {
                    "job_id": job["job_id"],
                    "fragment_id": event["fragment_id"],
                    "fragment_index": event["fragment_index"],
                    "attempt": 0,
                    "outcome": event["outcome"],
                    "feedback_boundary": boundary,
                }
                current_failures[job["owner"]].append(feedback)
            else:
                terminate(job, "failed", boundary)

        for job in sorted(
            upload_complete,
            key=lambda item: (item["submit_slot"], item["deadline"], item["owner"], item["job_id"]),
        ):
            load = (1 if active is not None else 0) + len(waiting)
            if load >= EDGE_QUEUE_CAPACITY:
                queue_events.append({"boundary": boundary, "kind": "rejected", "job_id": job["job_id"], "load_before": load})
                terminate(job, "rejected", boundary)
            else:
                job["status"] = "queued"
                job["queue_entry"] = boundary
                waiting.append(job["job_id"])
                queue_events.append({"boundary": boundary, "kind": "enqueued", "job_id": job["job_id"], "load_before": load})

        if boundary < HORIZON and active is None and waiting:
            active = waiting.popleft()
            jobs[active]["status"] = "active"
            jobs[active]["service_start"] = boundary
            queue_events.append({"boundary": boundary, "kind": "service_start", "job_id": active})

        queue_snapshots.append(
            {
                "boundary": boundary,
                "active": active,
                "waiting": list(waiting),
                "queue_load": (1 if active is not None else 0) + len(waiting),
            }
        )

        if boundary == HORIZON:
            break

        for robot in range(N_ROBOTS):
            for target, event in harvested.items():
                if event["robot"] == robot or _distance(current_positions[robot], scene["targets"][target]) <= SENSE_RADIUS + EPS:
                    known_harvested[robot].add(target)

        current_observations: list[list[dict[str, Any]]] = [[] for _ in range(N_ROBOTS)]
        for robot in range(N_ROBOTS):
            for target in TARGET_IDS:
                if target in harvested:
                    continue
                if _distance(current_positions[robot], scene["targets"][target]) <= SENSE_RADIUS + EPS:
                    observation = _observation(name, variant, robot, target, boundary, current_positions[robot])
                    observations.append(observation)
                    cached_observations[robot][observation["observation_id"]] = observation
                    current_observations[robot].append(observation)

        snapshots: list[dict[str, Any]] = []
        for robot in range(N_ROBOTS):
            snapshots.append(
                {
                    "slot": boundary,
                    "robot": f"r{robot}",
                    "position": current_positions[robot],
                    "energy": current_energies[robot],
                    "current_observations": [_copy_public_observation(item) for item in current_observations[robot]],
                    "cached_observations": [_copy_public_observation(item) for item in cached_observations[robot].values()],
                    "pending_jobs": [_copy_public_job(jobs[job_id]) for job_id in sorted(pending[robot])],
                    "jobs": [
                        _copy_public_job(job)
                        for job in sorted(
                            (item for item in jobs.values() if item["owner"] == robot),
                            key=lambda item: item["job_id"],
                        )
                    ],
                    "failure_feedback": [dict(item) for item in current_failures[robot]],
                    "results": [_copy_result(item) for item in robot_results[robot]],
                    "known_harvested": sorted(known_harvested[robot]),
                    "queue_load": (1 if active is not None else 0) + len(waiting),
                }
            )
        action_start_snapshots.append(snapshots)

        operations: list[dict[str, Any]] = []
        for robot in range(N_ROBOTS):
            raw_action = submitted.get((boundary, robot), _idle(boundary, robot))
            action = dict(raw_action) if _action_shape(raw_action) else _idle(boundary, robot)
            kind = action["kind"]
            valid = True
            selected_result: dict[str, Any] | None = None
            if kind == "submit":
                observation = cached_observations[robot].get(action["observation_id"])
                if observation is None or observation["robot"] != robot or observation["target"] != action["target"] or observation["observation_slot"] > boundary:
                    legality["visible_observations"].append(f"{name}|{variant}|slot={boundary}|r{robot}: observation {action['observation_id']!r} is not an owned visible cached observation for {action['target']}")
                    valid = False
                if len(pending[robot]) >= 2:
                    legality["job_capacity"].append(f"{name}|{variant}|slot={boundary}|r{robot}: unfinished_jobs={len(pending[robot])}, limit=2")
                    valid = False
                deadline = TARGET_VALUE_DEADLINE[action["target"]][1]
                if boundary >= deadline:
                    legality["job_references"].append(f"{name}|{variant}|slot={boundary}|r{robot}: submit target={action['target']} at or after deadline={deadline}")
                    valid = False
                required_energy = 5.0 if action["mode"] == "local" else 1.0
                if current_energies[robot] + EPS < required_energy:
                    legality["energy"].append(f"{name}|{variant}|slot={boundary}|r{robot}: submit mode={action['mode']} needs {required_energy}, available={current_energies[robot]}")
                    valid = False
            elif kind == "cancel":
                job = jobs.get(action["job_id"])
                if job is None or job["owner"] != robot or action["job_id"] not in pending[robot]:
                    legality["job_references"].append(f"{name}|{variant}|slot={boundary}|r{robot}: cancel job {action['job_id']!r} is not an owned unfinished job")
                    valid = False
            elif kind == "retry":
                job = jobs.get(action["job_id"])
                if job is None or job["owner"] != robot or job.get("status") != "await_retry" or job.get("failure_due") != boundary or boundary >= (job.get("deadline") if job else -1):
                    legality["retry_window"].append(f"{name}|{variant}|slot={boundary}|r{robot}: retry job {action['job_id']!r} has no owned attempt-0 failure due now before deadline")
                    valid = False
                if current_energies[robot] + EPS < 1.0:
                    legality["energy"].append(f"{name}|{variant}|slot={boundary}|r{robot}: retry needs 1 possible transmission energy, available={current_energies[robot]}")
                    valid = False
            elif kind == "harvest":
                target = action["target"]
                eligible = [
                    result
                    for result in robot_results[robot]
                    if result["target"] == target
                    and result["confidence"] + EPS >= CONFIDENCE_THRESHOLD
                    and 0 <= boundary - result["completion_boundary"] <= RESULT_MAX_AGE
                ]
                if target in harvested or current_positions[robot] != scene["targets"][target] or not eligible:
                    legality["harvest_results"].append(f"{name}|{variant}|slot={boundary}|r{robot}: harvest target={target}, position={current_positions[robot]}, expected_position={scene['targets'][target]}, eligible_results={len(eligible)}, already_harvested={target in harvested}")
                    valid = False
                else:
                    selected_result = min(eligible, key=lambda item: (-item["confidence"], item["completion_boundary"], item["job_id"]))
                if current_energies[robot] + EPS < 3.0:
                    legality["energy"].append(f"{name}|{variant}|slot={boundary}|r{robot}: harvest needs 3, available={current_energies[robot]}")
                    valid = False
            elif kind == "move" and current_energies[robot] + EPS < 1.0:
                legality["energy"].append(f"{name}|{variant}|slot={boundary}|r{robot}: move needs 1, available={current_energies[robot]}")
                valid = False
            operations.append({"robot": robot, "action": action if valid else _idle(boundary, robot), "selected_result": selected_result if valid else None})

        retry_jobs: set[str] = set()
        for operation in operations:
            robot = operation["robot"]
            action = operation["action"]
            kind = action["kind"]
            if kind == "move":
                current_energies[robot] -= 1.0
                energy_charges.append({"slot": boundary, "robot": robot, "amount": 1.0, "kind": "move"})
            elif kind == "harvest":
                current_energies[robot] -= 3.0
                energy_charges.append({"slot": boundary, "robot": robot, "amount": 3.0, "kind": "harvest"})
            elif kind == "submit":
                observation = cached_observations[robot][action["observation_id"]]
                quality = _observation_quality(observation, variant, boundary)
                job_id = _job_id(name, variant, robot, boundary)
                mode = action["mode"]
                job = {
                    "job_id": job_id,
                    "owner": robot,
                    "target": action["target"],
                    "mode": mode,
                    "observation_id": action["observation_id"],
                    "observation_slot": observation["observation_slot"],
                    "observation_o": observation["o"],
                    "submit_slot": boundary,
                    "deadline": TARGET_VALUE_DEADLINE[action["target"]][1],
                    "quality": quality,
                    "confidence": _confidence(mode, quality),
                    "status": "local_pending" if mode == "local" else "ready",
                    "fragment_count": 0 if mode == "local" else FRAGMENT_COUNTS[mode],
                    "next_fragment": None if mode == "local" else 0,
                    "failure_due": None,
                    "queue_entry": None,
                    "service_start": None,
                    "service_remaining": None if mode == "local" else SERVICE_BITS[mode],
                    "completion_due": boundary + 1 if mode == "local" else None,
                    "completion_boundary": None,
                    "terminal_boundary": None,
                    "inflight_event": None,
                }
                jobs[job_id] = job
                pending[robot].add(job_id)
                if mode == "local":
                    current_energies[robot] -= 5.0
                    total_local_compute_energy += 5.0
                    energy_charges.append({"slot": boundary, "robot": robot, "amount": 5.0, "kind": "local"})
            elif kind == "cancel":
                terminate(jobs[action["job_id"]], "canceled", boundary)
                queue_events.append({"boundary": boundary, "kind": "canceled", "job_id": action["job_id"]})
            elif kind == "retry":
                retry_jobs.add(action["job_id"])

        offers: list[dict[str, Any]] = []
        for job in jobs.values():
            if job["status"] == "ready":
                fragment_index = int(job["next_fragment"])
                offers.append({"job": job, "fragment_index": fragment_index, "attempt": 0})
            elif job["status"] == "await_retry" and job["job_id"] in retry_jobs:
                offers.append({"job": job, "fragment_index": int(job["next_fragment"]), "attempt": 1})
        offers.sort(
            key=lambda item: (
                item["job"]["submit_slot"],
                item["job"]["deadline"],
                item["job"]["owner"],
                item["job"]["job_id"],
                item["fragment_index"],
                item["attempt"],
            )
        )
        admitted_this_slot = 0
        admitted_bits_this_slot = 0
        for offer_index, offer in enumerate(offers):
            job = offer["job"]
            owner = job["owner"]
            fragment_index = offer["fragment_index"]
            attempt = offer["attempt"]
            fragment_id = f"{job['job_id']}|f{fragment_index}"
            offer_key = (job["submit_slot"], job["deadline"], owner, job["job_id"], fragment_index, attempt)
            admitted = False
            success = False
            rate: float | None = None
            probability: float | None = None
            sample: float | None = None
            admission_index: int | None = None
            if admitted_bits_this_slot + FRAGMENT_BITS > UPLINK_CAPACITY:
                outcome = "capacity"
            elif total_edge_bits + FRAGMENT_BITS > EDGE_BIT_BUDGET:
                outcome = "budget"
            elif current_energies[owner] + EPS < 1.0:
                outcome = "energy"
            else:
                admitted = True
                admission_index = admitted_this_slot
                admitted_this_slot += 1
                admitted_bits_this_slot += FRAGMENT_BITS
                total_edge_bits += FRAGMENT_BITS
                current_energies[owner] -= 1.0
                energy_charges.append({"slot": boundary, "robot": owner, "amount": 1.0, "kind": "transmission"})
                position = current_positions[owner]
                d_n = _clip(_distance(position, EDGE_COORDINATE) / 18.0)
                rate = _clip(960.0 / (1.0 + 2.0 * d_n + 0.2 * admission_index + 0.15 * job["observation_o"]), 120.0, 960.0)
                probability = _clip(rate / 960.0 - _link_penalty(name, variant, boundary, position), 0.05, 1.0)
                sample = _packet_sample(name, variant, fragment_id, attempt)
                success = sample < probability
                outcome = "success" if success else "loss"
            event_id = f"{fragment_id}|a{attempt}|slot={boundary}"
            event = {
                "event_id": event_id,
                "slot": boundary,
                "offer_index": offer_index,
                "offer_key": offer_key,
                "job_id": job["job_id"],
                "owner": owner,
                "fragment_id": fragment_id,
                "fragment_index": fragment_index,
                "attempt": attempt,
                "position": current_positions[owner],
                "occlusion": job["observation_o"],
                "admitted": admitted,
                "admission_index": admission_index,
                "rate": rate,
                "probability": probability,
                "sample": sample,
                "success": success,
                "outcome": outcome,
            }
            transmissions.append(event)
            outcomes.setdefault(boundary + 1, []).append(event)
            job["status"] = "transmitting"
            job["inflight_event"] = event_id

        for job in list(jobs.values()):
            if job["status"] == "await_retry" and job.get("failure_due") == boundary:
                terminate(job, "failed", boundary)

        if active is not None:
            rate = _service_rate(name, variant, boundary)
            job = jobs[active]
            remaining_before = float(job["service_remaining"])
            job["service_remaining"] = remaining_before - rate
            service_records.append(
                {
                    "slot": boundary,
                    "job_id": active,
                    "rate": rate,
                    "remaining_before": remaining_before,
                    "remaining_after": job["service_remaining"],
                }
            )

        for operation in operations:
            action = operation["action"]
            if action["kind"] != "harvest":
                continue
            target = action["target"]
            robot = operation["robot"]
            result = operation["selected_result"]
            succeeded = target not in harvested
            event = {
                "action_slot": boundary,
                "boundary": boundary + 1,
                "robot": robot,
                "target": target,
                "success": succeeded,
                "job_id": result["job_id"] if result else None,
                "confidence": result["confidence"] if result else None,
                "result_completion": result["completion_boundary"] if result else None,
                "on_time": succeeded and boundary + 1 <= TARGET_VALUE_DEADLINE[target][1],
            }
            harvests.append(event)
            if succeeded:
                harvested[target] = event

        accepted_destinations: set[tuple[int, int]] = set()
        next_positions: list[tuple[int, int]] = []
        for operation in operations:
            robot = operation["robot"]
            action = operation["action"]
            before = current_positions[robot]
            attempted = action["kind"] == "move"
            if attempted:
                delta = DELTAS[action["direction"]]
                proposed = (before[0] + delta[0], before[1] + delta[1])
                unsafe = (
                    not _traversable(name, proposed)
                    or proposed in accepted_destinations
                    or proposed in {current_positions[index] for index in range(robot + 1, N_ROBOTS)}
                )
                after = before if unsafe else proposed
                if unsafe:
                    safety_events += 1
                else:
                    successful_moves += 1
            else:
                proposed = before
                unsafe = False
                after = before
            accepted_destinations.add(after)
            next_positions.append(after)
            move_events.append(
                {
                    "slot": boundary,
                    "robot": robot,
                    "attempted": attempted,
                    "direction": action["direction"] if attempted else None,
                    "before": before,
                    "proposed": proposed,
                    "after": after,
                    "unsafe": unsafe,
                }
            )
        positions.append(next_positions)
        energies.append(current_energies)

    return {
        "name": name,
        "variant": variant,
        "positions": positions,
        "energies": energies,
        "observations": observations,
        "jobs": jobs,
        "transmissions": transmissions,
        "queue_events": queue_events,
        "queue_snapshots": queue_snapshots,
        "service_records": service_records,
        "results": results,
        "harvests": harvests,
        "harvested": harvested,
        "move_events": move_events,
        "energy_charges": energy_charges,
        "action_start_snapshots": action_start_snapshots,
        "legality": legality,
        "safety_events": safety_events,
        "successful_moves": successful_moves,
        "total_edge_bits": total_edge_bits,
        "total_local_compute_energy": total_local_compute_energy,
    }


def _legality(records: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures = {name: [] for name in ATOM_SHARES["decision_legality"]}
    for key in SCENARIO_KEYS:
        if key not in records:
            for atom_id in failures:
                failures[atom_id].append(f"missing canonical unit {key}")
            continue
        try:
            state = _replay(key[0], key[1], records[key])
        except Exception as exc:
            for atom_id in failures:
                failures[atom_id].append(f"{key}: legality replay raised {type(exc).__name__}: {exc}")
            continue
        for name, details in state["legality"].items():
            failures[f"decision_legality.{name}"].extend(details)
    return [
        AtomVerdict(atom_id, not details, "passed" if not details else " | ".join(details[:10]), ATOM_SHARES["decision_legality"][atom_id])
        for atom_id, details in failures.items()
    ]


def _audit_replay(name: str, variant: int, state: Mapping[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {
        "observation_processing": [],
        "uplink": [],
        "queue": [],
        "harvest": [],
        "motion_energy": [],
        "terminal_metrics": [],
    }
    scene = SCENARIOS[name]
    positions = state.get("positions")
    energies = state.get("energies")
    if not isinstance(positions, list) or len(positions) != HORIZON + 1:
        failures["motion_energy"].append(f"{name}|{variant}: expected {HORIZON + 1} position boundaries")
        return failures
    if not isinstance(energies, list) or len(energies) != HORIZON + 1:
        failures["motion_energy"].append(f"{name}|{variant}: expected {HORIZON + 1} energy boundaries")
        return failures

    successful_harvests = {
        event["target"]: event
        for event in state.get("harvests", ())
        if isinstance(event, Mapping) and event.get("success")
    }
    expected_observations: dict[str, dict[str, Any]] = {}
    for slot in range(HORIZON):
        for robot in range(N_ROBOTS):
            position = tuple(positions[slot][robot])
            for target in TARGET_IDS:
                harvest = successful_harvests.get(target)
                if harvest is not None and harvest["boundary"] <= slot:
                    continue
                if _distance(position, scene["targets"][target]) <= SENSE_RADIUS + EPS:
                    observation = _observation(name, variant, robot, target, slot, position)
                    expected_observations[observation["observation_id"]] = observation
    actual_observations = {
        item.get("observation_id"): item
        for item in state.get("observations", ())
        if isinstance(item, Mapping) and isinstance(item.get("observation_id"), str)
    }
    if set(actual_observations) != set(expected_observations):
        failures["observation_processing"].append(f"{name}|{variant}: observation ID set mismatch, expected={len(expected_observations)}, actual={len(actual_observations)}")
    for observation_id, expected in expected_observations.items():
        actual = actual_observations.get(observation_id)
        if actual is None:
            continue
        if any(actual.get(field) != expected[field] for field in ("robot", "target", "observation_slot", "robot_position", "target_position", "o")) or not math.isclose(float(actual.get("d_o", math.inf)), expected["d_o"], rel_tol=0.0, abs_tol=EPS):
            failures["observation_processing"].append(f"{name}|{variant}: observation mismatch for {observation_id}")

    jobs = state.get("jobs", {})
    if not isinstance(jobs, Mapping):
        failures["observation_processing"].append(f"{name}|{variant}: jobs is not a mapping")
        jobs = {}
    for job_id, job in jobs.items():
        if not isinstance(job, Mapping):
            failures["observation_processing"].append(f"{name}|{variant}: malformed job {job_id}")
            continue
        observation = actual_observations.get(job.get("observation_id"))
        if observation is None:
            failures["observation_processing"].append(f"{name}|{variant}: job {job_id} references missing observation")
            continue
        expected_quality = _observation_quality(observation, variant, int(job["submit_slot"]))
        expected_confidence = _confidence(str(job["mode"]), expected_quality)
        if not math.isclose(float(job.get("quality", math.inf)), expected_quality, rel_tol=0.0, abs_tol=EPS) or not math.isclose(float(job.get("confidence", math.inf)), expected_confidence, rel_tol=0.0, abs_tol=EPS):
            failures["observation_processing"].append(f"{name}|{variant}: quality/confidence mismatch for {job_id}")
        if job.get("mode") == "local" and job.get("completion_boundary") is not None and job.get("completion_boundary") != job.get("submit_slot") + 1:
            failures["observation_processing"].append(f"{name}|{variant}: local completion mismatch for {job_id}")
    result_ids: set[str] = set()
    for result in state.get("results", ()):
        if not isinstance(result, Mapping) or result.get("job_id") not in jobs:
            failures["observation_processing"].append(f"{name}|{variant}: result references unknown job")
            continue
        job = jobs[result["job_id"]]
        result_ids.add(result["job_id"])
        if any(result.get(field) != job.get(field) for field in ("owner", "target", "mode", "observation_id", "observation_slot")) or result.get("completion_boundary") != job.get("completion_boundary") or not math.isclose(float(result.get("confidence", math.inf)), float(job.get("confidence", -math.inf)), rel_tol=0.0, abs_tol=EPS):
            failures["observation_processing"].append(f"{name}|{variant}: result fields mismatch for {result.get('job_id')}")
    for job_id, job in jobs.items():
        if isinstance(job, Mapping) and job.get("status") == "completed" and job_id not in result_ids:
            failures["observation_processing"].append(f"{name}|{variant}: completed job {job_id} has no result")

    transmissions = state.get("transmissions", ())
    admitted_total = 0
    by_slot: dict[int, list[Mapping[str, Any]]] = {}
    for event in transmissions if isinstance(transmissions, list) else ():
        if not isinstance(event, Mapping) or not _integer(event.get("slot")):
            failures["uplink"].append(f"{name}|{variant}: malformed transmission")
            continue
        by_slot.setdefault(event["slot"], []).append(event)
    for slot, events in by_slot.items():
        events.sort(key=lambda item: int(item.get("offer_index", -1)))
        admitted_index = 0
        admitted_bits = 0
        for event in events:
            if event.get("admitted"):
                if event.get("admission_index") != admitted_index:
                    failures["uplink"].append(f"{name}|{variant}|slot={slot}: admission index mismatch")
                if admitted_bits + FRAGMENT_BITS > UPLINK_CAPACITY:
                    failures["uplink"].append(f"{name}|{variant}|slot={slot}: admitted capacity exceeds {UPLINK_CAPACITY}")
                admitted_index += 1
                admitted_bits += FRAGMENT_BITS
                admitted_total += FRAGMENT_BITS
                expected_rate = _clip(
                    960.0
                    / (
                        1.0
                        + 2.0 * _clip(_distance(tuple(event["position"]), EDGE_COORDINATE) / 18.0)
                        + 0.2 * int(event["admission_index"])
                        + 0.15 * int(event["occlusion"])
                    ),
                    120.0,
                    960.0,
                )
                expected_probability = _clip(expected_rate / 960.0 - _link_penalty(name, variant, slot, tuple(event["position"])), 0.05, 1.0)
                expected_sample = _packet_sample(name, variant, str(event["fragment_id"]), int(event["attempt"]))
                if not math.isclose(float(event.get("rate", math.inf)), expected_rate, rel_tol=0.0, abs_tol=EPS) or not math.isclose(float(event.get("probability", math.inf)), expected_probability, rel_tol=0.0, abs_tol=EPS) or not math.isclose(float(event.get("sample", math.inf)), expected_sample, rel_tol=0.0, abs_tol=EPS):
                    failures["uplink"].append(f"{name}|{variant}: link fields mismatch for {event.get('event_id')}")
                expected_success = expected_sample < expected_probability
                if event.get("success") != expected_success or event.get("outcome") != ("success" if expected_success else "loss"):
                    failures["uplink"].append(f"{name}|{variant}: link outcome mismatch for {event.get('event_id')}")
            elif event.get("outcome") not in ("capacity", "budget", "energy") or any(event.get(field) is not None for field in ("admission_index", "rate", "probability", "sample")) or event.get("success") is not False:
                failures["uplink"].append(f"{name}|{variant}: rejected transmission fields mismatch for {event.get('event_id')}")
    if admitted_total != state.get("total_edge_bits") or admitted_total > EDGE_BIT_BUDGET:
        failures["uplink"].append(f"{name}|{variant}: total_edge_bits expected={admitted_total}, actual={state.get('total_edge_bits')}")

    snapshots = state.get("queue_snapshots", ())
    if not isinstance(snapshots, list) or len(snapshots) != HORIZON + 1:
        failures["queue"].append(f"{name}|{variant}: expected {HORIZON + 1} queue boundaries")
    else:
        for snapshot in snapshots:
            waiting_ids = snapshot.get("waiting", ()) if isinstance(snapshot, Mapping) else ()
            load = (1 if snapshot.get("active") is not None else 0) + len(waiting_ids)
            if snapshot.get("queue_load") != load or load > EDGE_QUEUE_CAPACITY or len(waiting_ids) != len(set(waiting_ids)) or snapshot.get("active") in waiting_ids:
                failures["queue"].append(f"{name}|{variant}: invalid queue snapshot at boundary={snapshot.get('boundary') if isinstance(snapshot, Mapping) else '?'}")
    service_by_job: dict[str, list[Mapping[str, Any]]] = {}
    for record in state.get("service_records", ()):
        if not isinstance(record, Mapping) or record.get("job_id") not in jobs:
            failures["queue"].append(f"{name}|{variant}: malformed service record")
            continue
        service_by_job.setdefault(record["job_id"], []).append(record)
        expected_rate = _service_rate(name, variant, int(record["slot"]))
        if not math.isclose(float(record.get("rate", math.inf)), expected_rate, rel_tol=0.0, abs_tol=EPS) or not math.isclose(float(record.get("remaining_after", math.inf)), float(record.get("remaining_before", 0.0)) - expected_rate, rel_tol=0.0, abs_tol=EPS):
            failures["queue"].append(f"{name}|{variant}: service transition mismatch for {record.get('job_id')} at slot={record.get('slot')}")
    for job_id, job in jobs.items():
        if not isinstance(job, Mapping) or job.get("mode") == "local" or job.get("queue_entry") is None:
            continue
        if job.get("service_start") is not None and job["service_start"] < job["queue_entry"]:
            failures["queue"].append(f"{name}|{variant}: service precedes queue entry for {job_id}")
        if job.get("status") == "completed":
            records = service_by_job.get(job_id, [])
            if not records or records[-1]["remaining_after"] > EPS or job.get("completion_boundary") != records[-1]["slot"] + 1:
                failures["queue"].append(f"{name}|{variant}: edge completion mismatch for {job_id}")

    results_by_id = {result["job_id"]: result for result in state.get("results", ()) if isinstance(result, Mapping)}
    seen_targets: set[str] = set()
    for event in state.get("harvests", ()):
        if not isinstance(event, Mapping) or event.get("job_id") not in results_by_id:
            failures["harvest"].append(f"{name}|{variant}: harvest has no result")
            continue
        result = results_by_id[event["job_id"]]
        slot = int(event["action_slot"])
        robot = int(event["robot"])
        target = str(event["target"])
        eligible = result["target"] == target and result["confidence"] + EPS >= CONFIDENCE_THRESHOLD and 0 <= slot - result["completion_boundary"] <= RESULT_MAX_AGE and tuple(positions[slot][robot]) == scene["targets"][target]
        expected_success = eligible and target not in seen_targets
        if not eligible or event.get("success") != expected_success or event.get("boundary") != slot + 1:
            failures["harvest"].append(f"{name}|{variant}: harvest transition mismatch at slot={slot}, robot={robot}, target={target}")
        if event.get("success"):
            seen_targets.add(target)
            if event.get("on_time") != (event["boundary"] <= TARGET_VALUE_DEADLINE[target][1]):
                failures["harvest"].append(f"{name}|{variant}: harvest deadline flag mismatch for {target}")
    if seen_targets != set(state.get("harvested", {})):
        failures["harvest"].append(f"{name}|{variant}: harvested target index mismatch")

    move_events = state.get("move_events", ())
    if not isinstance(move_events, list) or len(move_events) != HORIZON * N_ROBOTS:
        failures["motion_energy"].append(f"{name}|{variant}: expected {HORIZON * N_ROBOTS} move records")
    else:
        for slot in range(HORIZON):
            slot_events = move_events[slot * N_ROBOTS : (slot + 1) * N_ROBOTS]
            if [tuple(event.get("after", ())) for event in slot_events] != [tuple(point) for point in positions[slot + 1]] or any(tuple(event.get("before", ())) != tuple(positions[slot][robot]) for robot, event in enumerate(slot_events)):
                failures["motion_energy"].append(f"{name}|{variant}: position transition mismatch at slot={slot}")
    charges: dict[tuple[int, int], float] = {}
    for charge in state.get("energy_charges", ()):
        if not isinstance(charge, Mapping) or not _integer(charge.get("slot")) or not _integer(charge.get("robot")) or not _finite(charge.get("amount")):
            failures["motion_energy"].append(f"{name}|{variant}: malformed energy charge")
            continue
        key = (charge["slot"], charge["robot"])
        charges[key] = charges.get(key, 0.0) + float(charge["amount"])
    for slot in range(HORIZON):
        for robot in range(N_ROBOTS):
            expected_energy = float(energies[slot][robot]) - charges.get((slot, robot), 0.0)
            if not math.isclose(float(energies[slot + 1][robot]), expected_energy, rel_tol=0.0, abs_tol=EPS) or expected_energy < -EPS:
                failures["motion_energy"].append(f"{name}|{variant}: energy transition mismatch at slot={slot}, robot={robot}")
    counted_unsafe = sum(bool(event.get("unsafe")) for event in move_events if isinstance(event, Mapping))
    counted_moves = sum(bool(event.get("attempted")) and not bool(event.get("unsafe")) for event in move_events if isinstance(event, Mapping))
    if counted_unsafe != state.get("safety_events") or counted_moves != state.get("successful_moves"):
        failures["motion_energy"].append(f"{name}|{variant}: motion counters mismatch")

    if state.get("total_local_compute_energy") != 5.0 * sum(1 for job in jobs.values() if isinstance(job, Mapping) and job.get("mode") == "local"):
        failures["terminal_metrics"].append(f"{name}|{variant}: local-compute energy total mismatch")
    for field in ("safety_events", "successful_moves", "total_edge_bits", "total_local_compute_energy"):
        if not _finite(state.get(field)) or state[field] < 0:
            failures["terminal_metrics"].append(f"{name}|{variant}: invalid terminal field {field}")
    return failures


def _replay_atoms(
    records: Mapping[tuple[str, int], Any], replayed: Mapping[tuple[str, int], Any]
) -> list[AtomVerdict]:
    failures = {name: [] for name in ATOM_SHARES["replay_validity"]}
    for key in SCENARIO_KEYS:
        if key not in records or key not in replayed:
            for atom_id in failures:
                failures[atom_id].append(f"missing replay for {key}")
            continue
        try:
            unit_failures = _audit_replay(key[0], key[1], replayed[key])
        except Exception as exc:
            for atom_id in failures:
                failures[atom_id].append(f"{key}: replay audit raised {type(exc).__name__}: {exc}")
            continue
        for name, details in unit_failures.items():
            failures[f"replay_validity.{name}"].extend(details)
    return [
        AtomVerdict(atom_id, not details, "passed" if not details else " | ".join(details[:10]), ATOM_SHARES["replay_validity"][atom_id])
        for atom_id, details in failures.items()
    ]


def _shortest_distance(name: str, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return 0
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        point, distance = queue.popleft()
        for delta in DELTAS.values():
            neighbor = (point[0] + delta[0], point[1] + delta[1])
            if neighbor in seen or not _traversable(name, neighbor):
                continue
            if neighbor == goal:
                return distance + 1
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return GRID_X * GRID_Y


def _first_qualifying_result(state: Mapping[str, Any], target: str) -> Mapping[str, Any] | None:
    deadline = TARGET_VALUE_DEADLINE[target][1]
    eligible = [
        result
        for result in state["results"]
        if result["target"] == target
        and result["completion_boundary"] <= deadline
        and result["confidence"] + EPS >= CONFIDENCE_THRESHOLD
    ]
    return min(eligible, key=lambda item: (item["completion_boundary"], -item["confidence"], item["job_id"])) if eligible else None


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    total_value = sum(value for value, _ in TARGET_VALUE_DEADLINE.values())
    for name, variant in SCENARIO_KEYS:
        state = replayed[(name, variant)]
        results_by_id = {result["job_id"]: result for result in state["results"]}
        completion_quality = 0.0
        perception_quality = 0.0
        for target in TARGET_IDS:
            value, deadline = TARGET_VALUE_DEADLINE[target]
            harvest = state["harvested"].get(target)
            on_time = harvest is not None and harvest["boundary"] <= deadline
            first_result = _first_qualifying_result(state, target)
            if on_time:
                completion_quality += value
                chosen = results_by_id[harvest["job_id"]]
                use_boundary = harvest["boundary"]
            elif first_result is not None:
                completion_quality += 0.5 * value
                chosen = first_result
                use_boundary = chosen["completion_boundary"]
            else:
                continue
            delay = chosen["completion_boundary"] - chosen["observation_slot"]
            age_at_use = use_boundary - chosen["completion_boundary"]
            quality = 0.6 * _clip(1.0 - delay / max(1, deadline - chosen["observation_slot"])) + 0.4 * _clip((chosen["confidence"] - CONFIDENCE_THRESHOLD) / 0.25) * _clip(1.0 - age_at_use / RESULT_MAX_AGE)
            perception_quality += value * _clip(quality)
        totals["primary_metrics_1"] += _clip(completion_quality / total_value)
        totals["primary_metrics_2"] += _clip(perception_quality / total_value)

        safe_ratio = len(state["harvested"]) / len(TARGET_IDS) * _clip(1.0 - state["safety_events"] / (N_ROBOTS * HORIZON))
        shortest_cost = 0
        for robot in range(N_ROBOTS):
            previous = SCENARIOS[name]["starts"][robot]
            events = [event for event in state["harvests"] if event["success"] and event["robot"] == robot]
            for event in events:
                target_point = SCENARIOS[name]["targets"][event["target"]]
                shortest_cost += _shortest_distance(name, previous, target_point)
                previous = target_point
        motion_efficiency = _clip(shortest_cost / max(1, state["successful_moves"])) if state["harvested"] else 0.0
        totals["primary_metrics_3"] += _clip(0.6 * safe_ratio + 0.4 * motion_efficiency)

        useful_edge_bits = 0
        for target in TARGET_IDS:
            first_result = _first_qualifying_result(state, target)
            if first_result is not None and first_result["mode"] in FRAGMENT_COUNTS:
                useful_edge_bits += FRAGMENT_COUNTS[first_result["mode"]] * FRAGMENT_BITS
        total_edge_bits = state["total_edge_bits"]
        useful_fraction = useful_edge_bits / max(total_edge_bits, 1) if total_edge_bits else 0.0
        totals["auxiliary_metrics_1"] += _clip(0.6 * useful_fraction + 0.4 * _clip(1.0 - total_edge_bits / EDGE_BIT_BUDGET))

        waits: list[float] = []
        for job in state["jobs"].values():
            if job["mode"] == "local" or job.get("queue_entry") is None:
                continue
            end = job["service_start"] if job.get("service_start") is not None else HORIZON
            waits.append(float(end - job["queue_entry"]))
        mean_wait = sum(waits) / len(waits) if waits else 0.0
        totals["auxiliary_metrics_2"] += _clip(0.5 * _clip(1.0 - state["total_local_compute_energy"] / (N_ROBOTS * ENERGY0)) + 0.5 * _clip(1.0 - mean_wait / HORIZON))
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
    return "failed_atoms=[]" if not failed else "failed_atoms=[{}]; {}".format(
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
        state.result_file_is_new_regular = bool(remove_ok and result_path.is_file() and not result_path.is_symlink())
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s is not None
            and state.elapsed_s <= 600 + EPS
            and state.result_file_is_new_regular
        )
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
        evidence = _evidence_atoms(state.data)
        state.scenarios = _index(state.data)
        coverage_score, coverage_defects = _coverage(state.data)
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = [
                _atom("decision_legality", name, False, f"legality replay raised {type(exc).__name__}: {exc}")
                for name in ("visible_observations", "job_capacity", "job_references", "retry_window", "harvest_results", "energy")
            ]
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.scores["canonical_coverage"] = coverage_score
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        can_replay = all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS
            for check_id in ("result_json", "result_evidence", "canonical_coverage", "decision_legality")
        )
        if can_replay:
            state.replayed = {
                key_: _replay(key_[0], key_[1], state.scenarios[key_])
                for key_ in SCENARIO_KEYS
            }
        replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["result_json"] = (f"{parse_failure}; " if parse_failure else "") + _diagnose(shape)
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
