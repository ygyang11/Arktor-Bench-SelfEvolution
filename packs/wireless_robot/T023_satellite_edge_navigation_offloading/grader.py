from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace


HORIZON = 60
EPS = 1e-6
ROBOTS = tuple(range(4))
SCENES = (
    "open_service_area",
    "satellite_only_corridor",
    "edge_congestion",
    "crossing_targets",
    "late_jobs",
)
STARTS = {
    "open_service_area": ((1.0, 1.0), (1.0, 19.0), (19.0, 1.0), (19.0, 19.0)),
    "satellite_only_corridor": ((2.0, 10.0), (5.0, 10.0), (15.0, 10.0), (18.0, 10.0)),
    "edge_congestion": ((2.0, 2.0), (2.0, 18.0), (18.0, 2.0), (18.0, 18.0)),
    "crossing_targets": ((2.0, 10.0), (10.0, 2.0), (18.0, 10.0), (10.0, 18.0)),
    "late_jobs": ((1.0, 10.0), (6.0, 1.0), (19.0, 10.0), (14.0, 19.0)),
}
TARGETS = {
    "open_service_area": ((4.0, 4.0), (4.0, 16.0), (16.0, 4.0), (16.0, 16.0)),
    "satellite_only_corridor": ((3.0, 3.0), (3.0, 17.0), (17.0, 3.0), (17.0, 17.0)),
    "edge_congestion": ((6.0, 6.0), (6.0, 14.0), (14.0, 6.0), (14.0, 14.0)),
    "crossing_targets": ((16.0, 4.0), (16.0, 16.0), (4.0, 16.0), (4.0, 4.0)),
    "late_jobs": ((3.0, 15.0), (8.0, 4.0), (17.0, 6.0), (12.0, 17.0)),
}
OBSTACLES = {
    "open_service_area": (),
    "satellite_only_corridor": ((8.0, 0.0, 12.0, 7.0), (8.0, 13.0, 12.0, 20.0)),
    "edge_congestion": ((9.0, 4.0, 11.0, 16.0),),
    "crossing_targets": ((9.0, 8.0, 11.0, 12.0),),
    "late_jobs": ((9.0, 0.0, 11.0, 8.0), (9.0, 12.0, 11.0, 20.0)),
}
EDGE = (10.0, 10.0)
SATELLITES = {
    "open_service_area": (10.0, 19.0),
    "satellite_only_corridor": (10.0, 2.0),
    "edge_congestion": (19.0, 10.0),
    "crossing_targets": (2.0, 10.0),
    "late_jobs": (18.0, 2.0),
}
TARGET_DEADLINES = (34, 38, 42, 46)
TARGET_VALUES = (2, 3, 4, 5)
DIAGONAL = 0.5 / math.sqrt(2.0)
MOVE_VECTORS = (
    (0.0, 0.0),
    (0.5, 0.0),
    (-0.5, 0.0),
    (0.0, 0.5),
    (0.0, -0.5),
    (DIAGONAL, DIAGONAL),
    (DIAGONAL, -DIAGONAL),
    (-DIAGONAL, DIAGONAL),
    (-DIAGONAL, -DIAGONAL),
)
EDGE_RADIUS = 7.0
SATELLITE_RADIUS = 20.0
LOCAL_RATE = 400_000.0
EDGE_RATE = 6_000_000.0
SATELLITE_RATE = 4_000_000.0
LOCAL_CAPACITY = 2
REMOTE_CAPACITY = 3
INITIAL_ENERGY = 120.0

ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "move", "target_id", "job_id", "placement"}
TARGET_IDS = {f"T{k}" for k in range(4)}
JOB_IDS = {f"J{k:02d}" for k in range(12)}
PLACEMENTS = {None, "local", "edge", "satellite"}
VALIDITY_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
METRIC_NAMES = (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.20,
        "result_evidence.action_fields": 0.30,
        "result_evidence.action_values": 0.50,
    },
    "decision_legality": {
        "decision_legality.job_references": 0.30,
        "decision_legality.target_requests": 0.15,
        "decision_legality.energy": 0.15,
        "decision_legality.placements": 0.20,
        "decision_legality.safety": 0.20,
    },
    "replay_validity": {
        "replay_validity.motion": 0.16666666666666666,
        "replay_validity.targets": 0.16666666666666666,
        "replay_validity.jobs": 0.16666666666666666,
        "replay_validity.uploads": 0.16666666666666666,
        "replay_validity.services": 0.16666666666666666,
        "replay_validity.metrics": 0.16666666666666666,
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
        self.replayed: dict[Hashable, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


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

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[group][atom_id])


def _score_atoms(
    atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]
) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in expected_share.values()
    ):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(
            atom.share,
            expected_share[atom.id],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(
        sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("declared atom shares must sum to 1.0")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return []
    return [row for row in data["scenarios"] if isinstance(row, Mapping)]


def _result_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = (
        parsed
        and _shape(data, ROOT_KEYS)
        and isinstance(data.get("scenarios"), list)
    )
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = (
        isinstance(rows, list)
        and bool(rows)
        and all(
            _shape(row, SCENE_KEYS) and isinstance(row.get("slots"), list)
            for row in rows
        )
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "strict JSON parsing failed"),
        _atom("result_json", "root", root_ok, "exact root key scenarios" if root_ok else "expected exact root key scenarios with an array value"),
        _atom("result_json", "scene_containers", containers_ok, "scene containers have exact keys" if containers_ok else "each scene must have exact keys scenario and slots"),
    ]


def _raw_slots(data: Any) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for scene_index, scene in enumerate(_scene_rows(data)):
        slots = scene.get("slots")
        if isinstance(slots, list):
            result.extend(
                (f"scenarios[{scene_index}].slots[{slot_index}]", slot)
                for slot_index, slot in enumerate(slots)
            )
    return result


def _valid_move(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_number(component) for component in value)
        and any(
            abs(float(value[0]) - candidate[0]) <= EPS
            and abs(float(value[1]) - candidate[1]) <= EPS
            for candidate in MOVE_VECTORS
        )
    )


def _valid_action_values(action: Any) -> bool:
    if not isinstance(action, Mapping):
        return False
    robot = action.get("robot")
    target = action.get("target_id")
    job = action.get("job_id")
    placement = action.get("placement")
    return (
        _integer(robot)
        and robot in ROBOTS
        and _valid_move(action.get("move"))
        and (target is None or (isinstance(target, str) and target in TARGET_IDS))
        and (job is None or (isinstance(job, str) and job in JOB_IDS))
        and (
            placement is None
            or (isinstance(placement, str) and placement in PLACEMENTS)
        )
        and ((job is None and placement is None) or (job is not None and placement is not None))
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    slots = _raw_slots(data)
    slot_ok = bool(slots) and all(
        _shape(slot, SLOT_KEYS)
        and _integer(slot.get("t"))
        and isinstance(slot.get("actions"), list)
        for _, slot in slots
    )
    actions = [
        (f"{path}.actions[{index}]", action)
        for path, slot in slots
        if isinstance(slot, Mapping) and isinstance(slot.get("actions"), list)
        for index, action in enumerate(slot["actions"])
    ]
    fields_ok = bool(actions) and all(_shape(action, ACTION_KEYS) for _, action in actions)
    values_ok = bool(actions) and all(_valid_action_values(action) for _, action in actions)
    return [
        _atom("result_evidence", "slot_containers", slot_ok, "all slot containers and integer t fields are valid" if slot_ok else "a slot must have exact keys t/actions, integer t, and an actions array"),
        _atom("result_evidence", "action_fields", fields_ok, "all actions have the exact five fields" if fields_ok else "each action must have exact keys robot/move/target_id/job_id/placement"),
        _atom("result_evidence", "action_values", values_ok, "all action values satisfy their local domains" if values_ok else "invalid robot, move, target, job, placement, or null relationship"),
    ]


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _scene_rows(data)
    scene_counts: Counter[str] = Counter()
    for scene_index, row in enumerate(rows):
        scene = row.get("scenario")
        if isinstance(scene, str):
            scene_counts[scene] += 1
        else:
            defects.append(
                f"unknown scenario key at scenarios[{scene_index}]: {scene!r}"
            )
    for scene in SCENES:
        count = scene_counts.get(scene, 0)
        if count == 0:
            defects.append(f"missing scenario {scene}")
        elif count > 1:
            defects.extend(f"duplicate scenario {scene}" for _ in range(count - 1))
    for scene, count in scene_counts.items():
        if scene not in SCENES:
            defects.extend(f"unknown scenario {scene!r}" for _ in range(count))

    slot_counts: Counter[tuple[str, int]] = Counter()
    action_counts: Counter[tuple[str, int, int]] = Counter()
    for scene_index, row in enumerate(rows):
        scene = row.get("scenario")
        slots = row.get("slots")
        if not isinstance(slots, list):
            continue
        for slot_index, slot in enumerate(slots):
            if not isinstance(slot, Mapping):
                defects.append(f"unknown slot key at scenarios[{scene_index}].slots[{slot_index}]")
                continue
            t = slot.get("t")
            valid_slot_key = (
                isinstance(scene, str)
                and scene in SCENES
                and _integer(t)
                and t in range(HORIZON)
            )
            if valid_slot_key:
                slot_counts[(scene, t)] += 1
            else:
                defects.append(f"unknown slot key scene={scene!r}, t={t!r}")
            actions = slot.get("actions")
            if not isinstance(actions, list):
                continue
            for action_index, action in enumerate(actions):
                robot = action.get("robot") if isinstance(action, Mapping) else None
                if valid_slot_key and _integer(robot) and robot in ROBOTS:
                    action_counts[(scene, t, robot)] += 1
                else:
                    defects.append(
                        "unknown action key at "
                        f"scenarios[{scene_index}].slots[{slot_index}].actions[{action_index}]"
                    )

    for key in ((scene, t) for scene in SCENES for t in range(HORIZON)):
        count = slot_counts.get(key, 0)
        if count == 0:
            defects.append(f"missing slot key {key}")
        elif count > 1:
            defects.extend(f"duplicate slot key {key}" for _ in range(count - 1))
    for key in (
        (scene, t, robot)
        for scene in SCENES
        for t in range(HORIZON)
        for robot in ROBOTS
    ):
        count = action_counts.get(key, 0)
        if count == 0:
            defects.append(f"missing action key {key}")
        elif count > 1:
            defects.extend(f"duplicate action key {key}" for _ in range(count - 1))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for row in _scene_rows(data):
        scene = row.get("scenario")
        slots = row.get("slots")
        if (
            not isinstance(scene, str)
            or scene not in SCENES
            or not isinstance(slots, list)
        ):
            continue
        for slot in slots:
            if (
                not isinstance(slot, Mapping)
                or not _integer(slot.get("t"))
                or slot["t"] not in range(HORIZON)
                or not isinstance(slot.get("actions"), list)
            ):
                continue
            for action in slot["actions"]:
                if (
                    isinstance(action, Mapping)
                    and _integer(action.get("robot"))
                    and action["robot"] in ROBOTS
                ):
                    records.setdefault((scene, slot["t"], action["robot"]), action)
    return records


def job_spec(scene: str, job_index: int) -> dict[str, Any]:
    scene_index = SCENES.index(scene)
    owner = job_index % 4
    release = 3 * (job_index // 4) + scene_index
    return {
        "id": f"J{job_index:02d}",
        "owner": owner,
        "release": release,
        "bits": 48_000 + 8_000 * (job_index % 3),
        "cycles": 3_000_000 + 500_000 * (job_index % 4),
        "deadline": release + 10 + (job_index % 2),
        "value": 1 + (job_index % 4),
    }


def _inside(point: Sequence[float], rectangle: Sequence[float]) -> bool:
    return (
        rectangle[0] - EPS <= point[0] <= rectangle[2] + EPS
        and rectangle[1] - EPS <= point[1] <= rectangle[3] + EPS
    )


def _dist(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _move_value(action: Any) -> tuple[float, float]:
    if isinstance(action, Mapping) and _valid_move(action.get("move")):
        return float(action["move"][0]), float(action["move"][1])
    return 0.0, 0.0


def _new_node(rate: float, capacity: int) -> dict[str, Any]:
    return {
        "rate": rate,
        "capacity": capacity,
        "active": None,
        "queue": [],
        "admissions": [],
        "starts": [],
        "max_occupancy": 0,
    }


def _replay(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    positions = [list(point) for point in STARTS[scene]]
    energy = [INITIAL_ENERGY for _ in ROBOTS]
    assigned: list[int | None] = [None for _ in ROBOTS]
    target_owner: list[int | None] = [None for _ in ROBOTS]
    target_arrival: list[int | None] = [None for _ in ROBOTS]
    jobs = {
        spec["id"]: {
            **spec,
            "status": "unreleased",
            "submit": None,
            "placement": None,
            "upload_left": None,
            "upload_rate": None,
            "upload_complete": None,
            "submit_position": None,
            "service_start": None,
            "cycles_left": None,
            "finish": None,
        }
        for spec in (job_spec(scene, index) for index in range(12))
    }
    nodes = {
        **{
            f"local_{robot}": _new_node(LOCAL_RATE, LOCAL_CAPACITY)
            for robot in ROBOTS
        },
        "edge": _new_node(EDGE_RATE, REMOTE_CAPACITY),
        "satellite": _new_node(SATELLITE_RATE, REMOTE_CAPACITY),
    }
    uploads: dict[str, list[str]] = {"edge": [], "satellite": []}
    submitted: set[str] = set()
    remote_events: list[str] = []
    sent_bits = 0.0
    delivered_bits = 0.0
    safety_events = 0
    traces: list[dict[str, Any]] = []
    issues = {
        "job_references": [],
        "target_requests": [],
        "energy": [],
        "placements": [],
        "safety": [],
    }

    def remove_from_resources(job_id: str) -> None:
        for node in nodes.values():
            if node["active"] == job_id:
                node["active"] = None
            node["queue"] = [queued for queued in node["queue"] if queued != job_id]
        for node_name in uploads:
            uploads[node_name] = [queued for queued in uploads[node_name] if queued != job_id]

    def begin_boundary(boundary: int) -> None:
        nonlocal delivered_bits
        for node in nodes.values():
            job_id = node["active"]
            if job_id is not None and jobs[job_id]["finish"] == boundary:
                jobs[job_id]["status"] = (
                    "completed" if boundary <= jobs[job_id]["deadline"] else "failed"
                )
                node["active"] = None

        for node_name in ("edge", "satellite"):
            arrivals = [
                jobs[job_id]
                for job_id in tuple(uploads[node_name])
                if jobs[job_id]["status"] == "upload_arrival"
                and jobs[job_id]["upload_complete"] == boundary
            ]
            arrivals.sort(key=lambda job: (job["release"], job["owner"], job["id"]))
            for job in arrivals:
                uploads[node_name].remove(job["id"])
                delivered_bits += job["bits"]
                if job["deadline"] < boundary:
                    job["status"] = "failed"
                    continue
                node = nodes[node_name]
                occupied = (1 if node["active"] is not None else 0) + len(node["queue"])
                if occupied >= node["capacity"]:
                    job["status"] = "failed"
                    continue
                job["status"] = "queued"
                job["cycles_left"] = float(job["cycles"])
                node["queue"].append(job["id"])
                node["admissions"].append(job["id"])
                node["max_occupancy"] = max(node["max_occupancy"], occupied + 1)

        for job in jobs.values():
            if job["status"] == "unreleased" and job["release"] <= boundary:
                job["status"] = "visible"
            elif job["status"] not in {"completed", "failed"} and job["deadline"] < boundary:
                job["status"] = "failed"
                remove_from_resources(job["id"])

    def advance_uploads(boundary: int) -> None:
        for node_name in ("edge", "satellite"):
            ordered = sorted(
                (
                    jobs[job_id]
                    for job_id in uploads[node_name]
                    if jobs[job_id]["status"] == "transferring"
                ),
                key=lambda job: (job["release"], job["owner"], job["id"]),
            )
            for job in ordered:
                amount = min(float(job["upload_left"]), float(job["upload_rate"]))
                job["upload_left"] -= amount
                if job["upload_left"] <= EPS:
                    job["upload_left"] = 0.0
                    job["upload_complete"] = boundary + 1
                    job["status"] = "upload_arrival"

    def advance_services(boundary: int) -> None:
        for node in nodes.values():
            if node["active"] is None and node["queue"]:
                job_id = node["queue"].pop(0)
                node["active"] = job_id
                node["starts"].append(job_id)
                job = jobs[job_id]
                job["status"] = "active"
                job["service_start"] = boundary
            job_id = node["active"]
            if job_id is None:
                continue
            job = jobs[job_id]
            job["cycles_left"] -= node["rate"]
            if job["cycles_left"] <= EPS:
                job["cycles_left"] = 0.0
                job["finish"] = boundary + 1
                job["status"] = "finishing"
            occupied = (1 if node["active"] is not None else 0) + len(node["queue"])
            node["max_occupancy"] = max(node["max_occupancy"], occupied)

    for t in range(HORIZON):
        begin_boundary(t)
        actions = [records.get((scene, t, robot), {}) for robot in ROBOTS]
        energy_before_motion = list(energy)

        contenders: dict[int, list[int]] = {target: [] for target in ROBOTS}
        for robot, action in enumerate(actions):
            target_id = action.get("target_id") if isinstance(action, Mapping) else None
            if not isinstance(target_id, str) or target_id not in TARGET_IDS:
                continue
            target = int(target_id[1:])
            if assigned[robot] is not None:
                if assigned[robot] != target:
                    issues["target_requests"].append(
                        f"{scene} t={t} r={robot}: locked T{assigned[robot]} but requested {target_id}"
                    )
                continue
            if target_owner[target] is not None:
                # A losing simultaneous intention is a legal no-op; only a
                # robot that already holds a lock is forbidden to switch.
                continue
            contenders[target].append(robot)
        for target, robots in contenders.items():
            if robots:
                winner = min(robots)
                assigned[winner] = target
                target_owner[target] = winner

        start_positions = [list(point) for point in positions]
        final_positions: list[list[float]] = []
        unsafe_robots: set[int] = set()
        for robot, action in enumerate(actions):
            move = _move_value(action)
            move_cost = 1.0 + 2.0 * (move[0] ** 2 + move[1] ** 2)
            affordable = energy[robot] >= move_cost - EPS
            if not affordable:
                issues["energy"].append(
                    f"{scene} t={t} r={robot}: move cost={move_cost:.6f}, energy={energy[robot]:.6f}"
                )
                proposal = list(start_positions[robot])
            else:
                proposal = [
                    start_positions[robot][0] + move[0],
                    start_positions[robot][1] + move[1],
                ]
            unsafe = (
                proposal[0] < -EPS
                or proposal[0] > 20.0 + EPS
                or proposal[1] < -EPS
                or proposal[1] > 20.0 + EPS
                or any(_inside(proposal, rectangle) for rectangle in OBSTACLES[scene])
                or any(_dist(proposal, lower) < 0.8 - EPS for lower in final_positions)
                or any(
                    _dist(proposal, start_positions[higher]) < 0.8 - EPS
                    for higher in range(robot + 1, len(ROBOTS))
                )
            )
            if unsafe or not affordable:
                unsafe_robots.add(robot)
                if unsafe:
                    safety_events += 1
                    issues["safety"].append(
                        f"{scene} t={t} r={robot}: unsafe proposed endpoint={proposal}"
                    )
                final_positions.append(list(start_positions[robot]))
            else:
                energy[robot] -= move_cost
                final_positions.append(proposal)
        positions = final_positions
        energy_after_motion = list(energy)

        for robot, target in enumerate(assigned):
            if (
                target is not None
                and target_arrival[target] is None
                and robot not in unsafe_robots
                and _dist(positions[robot], TARGETS[scene][target]) <= 0.6 + EPS
            ):
                target_arrival[target] = t + 1

        for robot, action in enumerate(actions):
            if not isinstance(action, Mapping):
                continue
            job_id = action.get("job_id")
            if job_id is None:
                continue
            placement = action.get("placement")
            job = jobs.get(job_id) if isinstance(job_id, str) else None
            valid_reference = (
                job is not None
                and job["owner"] == robot
                and job["release"] <= t < job["deadline"]
                and job_id not in submitted
                and job["status"] == "visible"
                and isinstance(placement, str)
                and placement in {"local", "edge", "satellite"}
            )
            if not valid_reference:
                issues["job_references"].append(
                    f"{scene} t={t} r={robot}: unavailable or repeated job={job_id!r}"
                )
                continue
            submitted.add(job_id)
            job["submit"] = t
            job["placement"] = placement

            if placement == "local":
                node = nodes[f"local_{robot}"]
                occupied = (1 if node["active"] is not None else 0) + len(node["queue"])
                if occupied >= node["capacity"]:
                    job["status"] = "failed"
                    continue
                if energy[robot] < 5.0 - EPS:
                    issues["energy"].append(
                        f"{scene} t={t} r={robot}: local cost=5, energy={energy[robot]:.6f}"
                    )
                    job["status"] = "failed"
                    continue
                energy[robot] -= 5.0
                job["status"] = "queued"
                job["cycles_left"] = float(job["cycles"])
                node["queue"].append(job_id)
                node["admissions"].append(job_id)
                node["max_occupancy"] = max(node["max_occupancy"], occupied + 1)
                continue

            if not isinstance(placement, str) or placement not in {"edge", "satellite"}:
                issues["placements"].append(
                    f"{scene} t={t} r={robot}: invalid placement={placement!r}"
                )
                job["status"] = "failed"
                continue
            node_position = EDGE if placement == "edge" else SATELLITES[scene]
            radius = EDGE_RADIUS if placement == "edge" else SATELLITE_RADIUS
            distance = _dist(positions[robot], node_position)
            if distance > radius + EPS:
                issues["placements"].append(
                    f"{scene} t={t} r={robot}: {placement} distance={distance:.6f}, radius={radius:.6f}"
                )
                job["status"] = "failed"
                continue
            remote_cost = float(math.ceil(job["bits"] / 48_000.0))
            if energy[robot] < remote_cost - EPS:
                issues["energy"].append(
                    f"{scene} t={t} r={robot}: upload cost={remote_cost:.6f}, energy={energy[robot]:.6f}"
                )
                job["status"] = "failed"
                continue
            energy[robot] -= remote_cost
            job["status"] = "transferring"
            job["upload_left"] = float(job["bits"])
            job["submit_position"] = list(positions[robot])
            job["upload_rate"] = max(
                20_000.0,
                min(120_000.0, 120_000.0 / (1.0 + distance)),
            )
            uploads[placement].append(job_id)
            remote_events.append(job_id)
            sent_bits += job["bits"]

        advance_uploads(t)
        advance_services(t)
        traces.append(
            {
                "boundary": t + 1,
                "start_positions": start_positions,
                "positions": [list(point) for point in positions],
                "energy_before_motion": energy_before_motion,
                "energy_after_motion": energy_after_motion,
                "energy": list(energy),
                "occupancy": {
                    name: (1 if node["active"] is not None else 0) + len(node["queue"])
                    for name, node in nodes.items()
                },
            }
        )

    begin_boundary(HORIZON)
    for job in jobs.values():
        if job["status"] not in {"completed", "failed"}:
            job["status"] = "failed"
            remove_from_resources(job["id"])

    target_value = sum(
        TARGET_VALUES[target]
        for target, arrival in enumerate(target_arrival)
        if arrival is not None and arrival <= TARGET_DEADLINES[target]
    ) / sum(TARGET_VALUES)
    total_job_value = sum(job["value"] for job in jobs.values())
    job_value = sum(
        job["value"]
        for job in jobs.values()
        if job["status"] == "completed"
        and job["finish"] is not None
        and job["finish"] <= job["deadline"]
    ) / total_job_value

    transfer_timeliness = 0.0
    queue_efficiency = 0.0
    if remote_events:
        event_jobs = [jobs[job_id] for job_id in remote_events]
        transfer_timeliness = sum(
            job["bits"]
            * (
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (job["upload_complete"] - job["submit"])
                        / max(1, job["deadline"] - job["submit"]),
                    ),
                )
                if job["upload_complete"] is not None
                and job["upload_complete"] <= min(job["deadline"], HORIZON)
                else 0.0
            )
            for job in event_jobs
        ) / sum(job["bits"] for job in event_jobs)
        queue_efficiency = sum(
            max(
                0.0,
                min(
                    1.0,
                    1.0 - (job["service_start"] - job["upload_complete"]) / HORIZON,
                ),
            )
            if job["service_start"] is not None and job["upload_complete"] is not None
            else 0.0
            for job in event_jobs
        ) / len(event_jobs)
    energy_fraction = sum(
        max(0.0, min(1.0, value / INITIAL_ENERGY)) for value in energy
    ) / len(ROBOTS)

    return {
        "scene": scene,
        "target_value": target_value,
        "job_value": job_value,
        "transfer_timeliness": transfer_timeliness,
        "queue_efficiency": queue_efficiency,
        "sent_bits": sent_bits,
        "delivered_bits": delivered_bits,
        "energy_fraction": energy_fraction,
        "safety_events": safety_events,
        "jobs": jobs,
        "target_arrival": target_arrival,
        "assigned": assigned,
        "target_owner": target_owner,
        "final_positions": positions,
        "traces": traces,
        "nodes": nodes,
        "remote_events": remote_events,
        "legality_issues": issues,
    }


def _legality_atoms(replayed: Mapping[Hashable, Any]) -> list[AtomVerdict]:
    issue_names = ("job_references", "target_requests", "energy", "placements", "safety")
    combined = {name: [] for name in issue_names}
    for scene in SCENES:
        replay = replayed.get(scene)
        if not isinstance(replay, Mapping):
            for name in issue_names:
                combined[name].append(f"{scene}: missing replay")
            continue
        source = replay.get("legality_issues", {})
        for name in issue_names:
            combined[name].extend(source.get(name, []))
    return [
        _atom(
            "decision_legality",
            name,
            not combined[name],
            "no violations" if not combined[name] else f"violations={combined[name][:6]}",
        )
        for name in issue_names
    ]


def _close(actual: Any, expected: Any, tolerance: float = EPS) -> bool:
    return _number(actual) and _number(expected) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
    )


def _point_close(actual: Any, expected: Sequence[float]) -> bool:
    return (
        isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
        and len(actual) == 2
        and _close(actual[0], expected[0])
        and _close(actual[1], expected[1])
    )


def _is_subsequence(values: Sequence[str], source: Sequence[str]) -> bool:
    cursor = iter(source)
    return all(any(candidate == value for candidate in cursor) for value in values)


def _audit_replay(
    scene: str,
    records: Mapping[Hashable, Any],
    replay: Mapping[str, Any],
) -> dict[str, list[str]]:
    failures = {
        "motion": [],
        "targets": [],
        "jobs": [],
        "uploads": [],
        "services": [],
        "metrics": [],
    }
    traces = replay.get("traces")
    jobs = replay.get("jobs")
    nodes = replay.get("nodes")
    remote_events = replay.get("remote_events")
    if not isinstance(traces, list) or len(traces) != HORIZON:
        failures["motion"].append(
            f"expected {HORIZON} transition records, got {len(traces) if isinstance(traces, list) else 'invalid'}"
        )
        traces = []
    if not isinstance(jobs, Mapping) or set(jobs) != JOB_IDS:
        failures["jobs"].append("terminal job map does not contain exactly J00..J11")
        jobs = {}
    if not isinstance(nodes, Mapping) or set(nodes) != {
        "local_0",
        "local_1",
        "local_2",
        "local_3",
        "edge",
        "satellite",
    }:
        failures["services"].append("service-node state set is incomplete")
        nodes = {}
    if not isinstance(remote_events, list) or any(
        not isinstance(job_id, str) for job_id in remote_events
    ):
        failures["uploads"].append("remote-event index is invalid")
        remote_events = []

    accepted_by_slot: list[list[bool]] = []
    previous_positions = [list(point) for point in STARTS[scene]]
    previous_energy = [INITIAL_ENERGY for _ in ROBOTS]
    for t, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            failures["motion"].append(f"t={t}: transition record is not an object")
            continue
        starts = trace.get("start_positions")
        ends = trace.get("positions")
        energy_before = trace.get("energy_before_motion")
        energy_motion = trace.get("energy_after_motion")
        energy_end = trace.get("energy")
        valid_arrays = (
            isinstance(starts, list)
            and isinstance(ends, list)
            and isinstance(energy_before, list)
            and isinstance(energy_motion, list)
            and isinstance(energy_end, list)
            and all(len(value) == len(ROBOTS) for value in (starts, ends, energy_before, energy_motion, energy_end))
        )
        if not valid_arrays:
            failures["motion"].append(f"t={t}: malformed transition arrays")
            continue
        if trace.get("boundary") != t + 1:
            failures["motion"].append(f"t={t}: wrong boundary index")
        if any(not _point_close(starts[r], previous_positions[r]) for r in ROBOTS):
            failures["motion"].append(f"t={t}: start positions break continuity")
        if any(not _close(energy_before[r], previous_energy[r]) for r in ROBOTS):
            failures["motion"].append(f"t={t}: starting energy breaks continuity")

        expected_positions: list[list[float]] = []
        expected_motion_energy = [float(value) for value in energy_before]
        accepted: list[bool] = []
        for robot in ROBOTS:
            action = records.get((scene, t, robot), {})
            move = _move_value(action)
            cost = 1.0 + 2.0 * (move[0] ** 2 + move[1] ** 2)
            proposal = [
                float(starts[robot][0]) + move[0],
                float(starts[robot][1]) + move[1],
            ]
            safe = (
                -EPS <= proposal[0] <= 20.0 + EPS
                and -EPS <= proposal[1] <= 20.0 + EPS
                and not any(_inside(proposal, rectangle) for rectangle in OBSTACLES[scene])
                and all(_dist(proposal, lower) >= 0.8 - EPS for lower in expected_positions)
                and all(
                    _dist(proposal, starts[higher]) >= 0.8 - EPS
                    for higher in range(robot + 1, len(ROBOTS))
                )
            )
            can_move = safe and float(energy_before[robot]) >= cost - EPS
            accepted.append(can_move)
            if can_move:
                expected_positions.append(proposal)
                expected_motion_energy[robot] -= cost
            else:
                expected_positions.append([float(starts[robot][0]), float(starts[robot][1])])
        accepted_by_slot.append(accepted)
        if any(not _point_close(ends[r], expected_positions[r]) for r in ROBOTS):
            failures["motion"].append(f"t={t}: post-motion position does not follow resolution rules")
        if any(not _close(energy_motion[r], expected_motion_energy[r]) for r in ROBOTS):
            failures["motion"].append(f"t={t}: movement energy accounting mismatch")

        expected_end_energy = list(expected_motion_energy)
        if jobs and nodes:
            for job in jobs.values():
                if not isinstance(job, Mapping) or job.get("submit") != t:
                    continue
                job_id = job.get("id")
                owner = job.get("owner")
                if not _integer(owner) or owner not in ROBOTS or not isinstance(job_id, str):
                    continue
                if job.get("placement") == "local" and job_id in nodes[f"local_{owner}"].get("admissions", []):
                    expected_end_energy[owner] -= 5.0
                elif job_id in remote_events:
                    expected_end_energy[owner] -= math.ceil(float(job["bits"]) / 48_000.0)
        if any(not _close(energy_end[r], expected_end_energy[r]) for r in ROBOTS):
            failures["motion"].append(f"t={t}: submission energy accounting mismatch")
        if any(float(value) < -EPS or not math.isfinite(float(value)) for value in energy_end):
            failures["motion"].append(f"t={t}: non-finite or negative post-slot energy")
        if any(
            not (-EPS <= float(point[0]) <= 20.0 + EPS and -EPS <= float(point[1]) <= 20.0 + EPS)
            or any(_inside(point, rectangle) for rectangle in OBSTACLES[scene])
            for point in ends
        ):
            failures["motion"].append(f"t={t}: invalid post-slot geometry")
        if any(
            _dist(ends[first], ends[second]) < 0.8 - EPS
            for first in ROBOTS
            for second in range(first + 1, len(ROBOTS))
        ):
            failures["motion"].append(f"t={t}: post-slot collision")
        previous_positions = [list(point) for point in ends]
        previous_energy = [float(value) for value in energy_end]

    assigned: list[int | None] = [None for _ in ROBOTS]
    target_owner: list[int | None] = [None for _ in ROBOTS]
    arrivals: list[int | None] = [None for _ in ROBOTS]
    for t, trace in enumerate(traces):
        contenders: dict[int, list[int]] = {target: [] for target in ROBOTS}
        for robot in ROBOTS:
            action = records.get((scene, t, robot), {})
            target_id = action.get("target_id") if isinstance(action, Mapping) else None
            if not isinstance(target_id, str) or target_id not in TARGET_IDS:
                continue
            target = int(target_id[1:])
            if assigned[robot] is not None or target_owner[target] is not None:
                continue
            contenders[target].append(robot)
        for target, contenders_for_target in contenders.items():
            if contenders_for_target:
                winner = min(contenders_for_target)
                assigned[winner] = target
                target_owner[target] = winner
        ends = trace.get("positions", []) if isinstance(trace, Mapping) else []
        if len(ends) != len(ROBOTS) or t >= len(accepted_by_slot):
            continue
        for robot, target in enumerate(assigned):
            if (
                target is not None
                and arrivals[target] is None
                and accepted_by_slot[t][robot]
                and _dist(ends[robot], TARGETS[scene][target]) <= 0.6 + EPS
            ):
                arrivals[target] = t + 1
    if replay.get("assigned") != assigned or replay.get("target_owner") != target_owner:
        failures["targets"].append("target-lock replay does not match simultaneous tie resolution")
    if replay.get("target_arrival") != arrivals:
        failures["targets"].append("target-arrival boundaries do not match accepted motion")
    if len([owner for owner in target_owner if owner is not None]) != len(
        set(owner for owner in target_owner if owner is not None)
    ):
        failures["targets"].append("a target owner is not unique")

    if jobs:
        for job_index in range(12):
            job_id = f"J{job_index:02d}"
            job = jobs.get(job_id)
            spec = job_spec(scene, job_index)
            if not isinstance(job, Mapping):
                failures["jobs"].append(f"{job_id}: missing terminal state")
                continue
            if any(job.get(name) != spec[name] for name in ("id", "owner", "release", "bits", "cycles", "deadline", "value")):
                failures["jobs"].append(f"{job_id}: public specification changed during replay")
            if job.get("status") not in {"completed", "failed"}:
                failures["jobs"].append(f"{job_id}: nonterminal at boundary 60")
            submit = job.get("submit")
            if submit is not None and (
                not _integer(submit) or not spec["release"] <= submit < spec["deadline"]
            ):
                failures["jobs"].append(f"{job_id}: invalid submission boundary {submit!r}")
            service_start = job.get("service_start")
            finish = job.get("finish")
            if service_start is not None and (not _integer(service_start) or submit is None or service_start < submit):
                failures["jobs"].append(f"{job_id}: invalid service-start boundary")
            if finish is not None and (not _integer(finish) or service_start is None or finish <= service_start):
                failures["jobs"].append(f"{job_id}: invalid finish boundary")
            if job.get("status") == "completed" and (
                finish is None or finish > spec["deadline"] or finish > HORIZON
            ):
                failures["jobs"].append(f"{job_id}: completed outside its legal deadline")
            if job.get("placement") in {"edge", "satellite"} and service_start is not None:
                upload_complete = job.get("upload_complete")
                if not _integer(upload_complete) or service_start < upload_complete:
                    failures["jobs"].append(f"{job_id}: remote service precedes upload arrival")

    if jobs:
        if len(remote_events) != len(set(remote_events)):
            failures["uploads"].append("a remote job appears more than once")
        expected_sent = 0.0
        expected_delivered = 0.0
        for job_id in remote_events:
            job = jobs.get(job_id)
            if not isinstance(job, Mapping) or job.get("placement") not in {"edge", "satellite"}:
                failures["uploads"].append(f"{job_id}: remote-event metadata is inconsistent")
                continue
            submit = job.get("submit")
            position = job.get("submit_position")
            rate = job.get("upload_rate")
            owner = job.get("owner")
            if (
                not _integer(submit)
                or not _integer(owner)
                or owner not in ROBOTS
                or not _point_close(position, traces[submit]["positions"][owner])
            ):
                failures["uploads"].append(f"{job_id}: upload position is not the post-motion submission position")
                continue
            node_position = EDGE if job["placement"] == "edge" else SATELLITES[scene]
            expected_rate = max(20_000.0, min(120_000.0, 120_000.0 / (1.0 + _dist(position, node_position))))
            if not _close(rate, expected_rate):
                failures["uploads"].append(f"{job_id}: fixed upload rate mismatch")
                continue
            natural_completion = submit + math.ceil(float(job["bits"]) / expected_rate)
            expected_completion = (
                natural_completion
                if natural_completion <= min(int(job["deadline"]) + 1, HORIZON)
                else None
            )
            if job.get("upload_complete") != expected_completion:
                failures["uploads"].append(
                    f"{job_id}: expected upload completion {expected_completion!r}, got {job.get('upload_complete')!r}"
                )
            expected_sent += float(job["bits"])
            if expected_completion is not None:
                expected_delivered += float(job["bits"])
        if not _close(replay.get("sent_bits"), expected_sent) or not _close(
            replay.get("delivered_bits"), expected_delivered
        ):
            failures["uploads"].append("remote bit accounting mismatch")

    if nodes and jobs:
        for node_name, node in nodes.items():
            if not isinstance(node, Mapping):
                failures["services"].append(f"{node_name}: invalid node state")
                continue
            admissions = node.get("admissions")
            starts = node.get("starts")
            capacity = node.get("capacity")
            if not isinstance(admissions, list) or not isinstance(starts, list):
                failures["services"].append(f"{node_name}: missing FIFO event lists")
                continue
            if len(admissions) != len(set(admissions)) or len(starts) != len(set(starts)):
                failures["services"].append(f"{node_name}: duplicate admission or service start")
            if not _is_subsequence(starts, admissions):
                failures["services"].append(f"{node_name}: service order violates FIFO admission order")
            if not _integer(capacity) or node.get("max_occupancy", capacity + 1) > capacity:
                failures["services"].append(f"{node_name}: capacity exceeded")
            prior_release = 0
            rate = float(node.get("rate", 0.0))
            for job_id in starts:
                job = jobs.get(job_id)
                if not isinstance(job, Mapping) or not _integer(job.get("service_start")):
                    failures["services"].append(f"{node_name}/{job_id}: missing service start")
                    continue
                start = int(job["service_start"])
                if start < prior_release:
                    failures["services"].append(f"{node_name}/{job_id}: overlaps preceding FIFO service")
                finish = job.get("finish")
                if finish is not None:
                    expected_finish = start + math.ceil(float(job["cycles"]) / rate)
                    if finish != expected_finish:
                        failures["services"].append(f"{node_name}/{job_id}: service duration mismatch")
                    prior_release = int(finish)
                else:
                    prior_release = int(job["deadline"]) + 1

    if jobs:
        expected_target = sum(
            TARGET_VALUES[target]
            for target, arrival in enumerate(replay.get("target_arrival", []))
            if _integer(arrival) and arrival <= TARGET_DEADLINES[target]
        ) / sum(TARGET_VALUES)
        expected_job = sum(
            job["value"]
            for job in jobs.values()
            if isinstance(job, Mapping)
            and job.get("status") == "completed"
            and _integer(job.get("finish"))
            and job["finish"] <= job["deadline"]
        ) / sum(job["value"] for job in jobs.values() if isinstance(job, Mapping))
        expected_transfer = 0.0
        expected_queue = 0.0
        if remote_events:
            event_jobs = [jobs[job_id] for job_id in remote_events if job_id in jobs]
            expected_transfer = sum(
                job["bits"]
                * (
                    max(0.0, min(1.0, 1.0 - (job["upload_complete"] - job["submit"]) / max(1, job["deadline"] - job["submit"])))
                    if job.get("upload_complete") is not None
                    and job["upload_complete"] <= min(job["deadline"], HORIZON)
                    else 0.0
                )
                for job in event_jobs
            ) / sum(job["bits"] for job in event_jobs)
            expected_queue = sum(
                max(0.0, min(1.0, 1.0 - (job["service_start"] - job["upload_complete"]) / HORIZON))
                if job.get("service_start") is not None and job.get("upload_complete") is not None
                else 0.0
                for job in event_jobs
            ) / len(event_jobs)
        expected_energy = (
            sum(max(0.0, min(1.0, value / INITIAL_ENERGY)) for value in previous_energy)
            / len(ROBOTS)
        )
        for name, expected in (
            ("target_value", expected_target),
            ("job_value", expected_job),
            ("transfer_timeliness", expected_transfer),
            ("queue_efficiency", expected_queue),
            ("energy_fraction", expected_energy),
        ):
            if not _close(replay.get(name), expected):
                failures["metrics"].append(
                    f"{name}: expected {expected:.9f}, got {replay.get(name)!r}"
                )
    return failures


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[Hashable, Any]
) -> list[AtomVerdict]:
    names = ("motion", "targets", "jobs", "uploads", "services", "metrics")
    failures = {name: [] for name in names}
    expected_keys = {
        (scene, t, robot)
        for scene in SCENES
        for t in range(HORIZON)
        for robot in ROBOTS
    }
    if set(records) != expected_keys or any(
        not _valid_action_values(action) for action in records.values()
    ):
        for name in names:
            failures[name].append("complete locally valid action evidence is required before replay")
    else:
        for scene in SCENES:
            replay = replayed.get(scene)
            if not isinstance(replay, Mapping):
                for name in names:
                    failures[name].append(f"{scene}: replay is missing")
                continue
            audited = _audit_replay(scene, records, replay)
            for name in names:
                failures[name].extend(f"{scene}: {detail}" for detail in audited[name])
    return [
        _atom(
            "replay_validity",
            name,
            not failures[name],
            "transition audit passed" if not failures[name] else f"issues={failures[name][:6]}",
        )
        for name in names
    ]


def _metrics(replayed: Mapping[Hashable, Any]) -> dict[str, float]:
    scene_results = [replayed[scene] for scene in SCENES]
    primary_1 = sum(result["target_value"] for result in scene_results) / len(SCENES)
    primary_2 = sum(result["job_value"] for result in scene_results) / len(SCENES)
    auxiliary_1 = sum(
        0.6 * result["transfer_timeliness"] + 0.4 * result["queue_efficiency"]
        for result in scene_results
    ) / len(SCENES)
    auxiliary_2 = sum(
        0.0
        if result["sent_bits"] <= EPS
        else 0.5 * result["delivered_bits"] / result["sent_bits"]
        + 0.5 * result["energy_fraction"]
        for result in scene_results
    ) / len(SCENES)
    values = {
        "primary_metrics_1": primary_1,
        "primary_metrics_2": primary_2,
        "auxiliary_metrics_1": auxiliary_1,
        "auxiliary_metrics_2": auxiliary_2,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite task metric")
    return {name: max(0.0, min(1.0, float(value))) for name, value in values.items()}


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8.0 * metrics["primary_metrics_1"]
        + 7.0 * metrics["primary_metrics_2"]
        + 3.0 * metrics["auxiliary_metrics_1"]
        + 2.0 * metrics["auxiliary_metrics_2"]
    ) / 20.0


def _evaluate_data(data: Any, parsed: bool = True) -> dict[str, Any]:
    scores: dict[str, float] = {}
    diagnostics: dict[str, str] = {}
    atoms = _result_atoms(data, parsed)
    scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"])
    diagnostics["result_json"] = _diagnose_atoms(atoms)
    atoms = _evidence_atoms(data)
    scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"])
    diagnostics["result_evidence"] = _diagnose_atoms(atoms)
    coverage, defects = _coverage_score(data)
    scores["canonical_coverage"] = coverage
    diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; defects={defects[:10]}"
    records = _index_records(data)
    replayed = {scene: _replay(scene, records) for scene in SCENES}
    atoms = _legality_atoms(replayed)
    scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"])
    diagnostics["decision_legality"] = _diagnose_atoms(atoms)
    atoms = _replay_atoms(records, replayed)
    scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"])
    diagnostics["replay_validity"] = _diagnose_atoms(atoms)
    gate = all(scores.get(name, 0.0) >= 1.0 - EPS for name in VALIDITY_CHECKS)
    metrics = _metrics(replayed) if gate else None
    for name in METRIC_NAMES:
        scores[name] = metrics[name] if metrics is not None else 0.0
        diagnostics[name] = (
            "validity gate passed"
            if metrics is not None
            else "validity gate failed; metric=0"
        )
    return {
        "scores": scores,
        "diagnostics": diagnostics,
        "records": records,
        "replayed": replayed,
        "metrics": metrics,
        "validity_gate": gate,
    }


def _regular(ws: Workspace, name: str) -> bool:
    path = Path(ws.root) / name
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if not match:
            continue
        title = match.group(1).strip()
        body: list[str] = []
        for following in lines[index + 1 :]:
            if re.match(r"^#{1,6}\s+", following):
                break
            body.append(following)
        found[title] = "\n".join(body).strip()
    return found


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
        text = ws.read_text("report.md") or ""
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    sections = _headings(text)
    missing = sorted(title for title in required if not sections.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        await ws.run("rm -f -- results.json", check=False, timeout=30)
        state.run_started = True
        started = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600)
        except (asyncio.TimeoutError, TimeoutError):
            state.timed_out = True
            state.elapsed_s = time.monotonic() - started
            state.official_message = f"timed_out=True, elapsed={state.elapsed_s:.3f}s, new_results=False"
            for name in VALIDITY_CHECKS:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"official run failed; {state.official_message}"
            return state
        state.elapsed_s = time.monotonic() - started
        state.exit_code = run.exit_code
        state.run_finished = True
        stderr_text = str(getattr(run, "stderr", "")).lower()
        state.timed_out = (
            state.elapsed_s >= 600.0 - EPS
            or state.exit_code is None
            or "timeout" in stderr_text
            or "timed out" in stderr_text
        )
        result_path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = result_path.is_file() and not result_path.is_symlink()
        state.official_ok = (
            run.exit_code == 0
            and not state.timed_out
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={run.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        )
        if not state.official_ok:
            for name in VALIDITY_CHECKS:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"official run failed; {state.official_message}"
            return state
        try:
            state.data = _strict_json(result_path.read_text(encoding="utf-8"))
            state.parse_ok = True
        except Exception as exc:
            state.data = None
            state.parse_ok = False
            state.diagnostics["result_json"] = f"parse failed: {type(exc).__name__}: {exc}"
        evaluated = _evaluate_data(state.data, state.parse_ok)
        state.scores.update(evaluated["scores"])
        for check_name, diagnostic in evaluated["diagnostics"].items():
            if check_name != "result_json" or state.parse_ok:
                state.diagnostics[check_name] = diagnostic
        state.records = evaluated["records"]
        state.replayed = evaluated["replayed"]
        state.metrics = evaluated["metrics"]
        state.validity_gate = evaluated["validity_gate"]
        for name in METRIC_NAMES:
            if state.validity_gate and state.metrics is not None:
                state.scores[name] = state.metrics[name]
                state.diagnostics[name] = "validity gate passed"
            else:
                state.scores[name] = 0.0
                failed = [
                    check_name
                    for check_name in VALIDITY_CHECKS
                    if state.scores.get(check_name, 0.0) < 1.0 - EPS
                ]
                state.diagnostics[name] = f"validity gate failed; failed_checks={failed}; metric=0"
        return state
    except Exception as exc:
        state.official_ok = False
        state.run_finished = True
        state.timed_out = (
            isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS)
        )
        state.official_message = f"grader exception {type(exc).__name__}: {exc}"
        for name in VALIDITY_CHECKS + METRIC_NAMES:
            state.scores[name] = 0.0
            state.diagnostics[name] = state.official_message
        return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE:
        return _CACHE[key]
    if key not in _INFLIGHT:
        _INFLIGHT[key] = asyncio.create_task(_build_state_uncached(ws))
    state = await _INFLIGHT[key]
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular(ws, "run.py"):
        return 0.0, "run.py missing or not a regular file"
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
        if result.exit_code == 0:
            return 1.0, "static compilation passed"
        return 0.0, f"static compilation failed; exit={result.exit_code}"
    except Exception as exc:
        return 0.0, f"compile check failed: {type(exc).__name__}: {exc}"


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):
    globals()[_name] = check(id=_name)(lambda ws, name=_name: _score(ws, name))
