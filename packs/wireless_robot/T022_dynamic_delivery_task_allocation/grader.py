from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 60
EPS = 1e-6
HUBS = tuple(range(3))
ROBOTS = tuple(range(6))
SCENES = (
    "balanced_stream",
    "overlap_hotspot",
    "sparse_link",
    "tight_windows",
    "burst_arrivals",
)
HUB_POSITIONS = ((1.0, 1.0), (11.0, 1.0), (6.0, 10.0))
ROBOT_HUB = (0, 0, 1, 1, 2, 2)
HUB_ROBOTS = {hub: tuple(r for r in ROBOTS if ROBOT_HUB[r] == hub) for hub in HUBS}
SENSING_RADIUS = 7.25
QUEUE_CAPACITY = 2
MESSAGE_BITS = 32
SCENE_CAPACITY_BITS = 128
MAX_SERVED_MESSAGES = SCENE_CAPACITY_BITS // MESSAGE_BITS
CANCEL_DELAY = 2
SERVICE_DURATION = 2
TASK_IDS = tuple(f"K{k:02d}" for k in range(18))
TASK_VALUES = (4, 4, 3, 5, 5, 5, 3, 4, 4, 3, 5, 4, 2, 2, 4, 5, 3, 5)
TOTAL_VALUE = sum(TASK_VALUES)
POSITION_CODES = {
    "A": (2.0, 2.0),
    "B": (3.0, 2.0),
    "C": (4.0, 4.0),
    "D": (5.0, 4.0),
    "E": (6.0, 4.0),
    "F": (7.0, 4.0),
    "G": (8.0, 4.0),
    "H": (9.0, 2.0),
    "I": (10.0, 2.0),
    "J": (5.0, 8.0),
    "K": (6.0, 9.0),
    "L": (7.0, 8.0),
    "M": (3.5, 5.0),
    "N": (8.5, 5.0),
    "O": (6.0, 2.0),
    "P": (5.0, 5.0),
    "Q": (6.0, 5.0),
    "R": (7.0, 5.0),
}
SCENE_TABLES = {
    "balanced_stream": {
        "codes": tuple("AIKCGEMNJLOPDFBHQR"),
        "release": (0, 0, 0, 4, 5, 7, 10, 11, 13, 14, 17, 19, 22, 23, 26, 28, 32, 36),
        "start": (1, 1, 2, 6, 7, 9, 12, 13, 15, 16, 19, 21, 24, 25, 28, 30, 34, 38),
        "end": (10, 10, 14, 18, 19, 22, 26, 27, 30, 31, 33, 35, 39, 40, 43, 45, 50, 56),
    },
    "overlap_hotspot": {
        "codes": tuple("DEFPQRDEFPQRDEFPQR"),
        "release": (0, 0, 0, 0, 1, 1, 10, 10, 10, 11, 11, 11, 22, 22, 23, 23, 24, 24),
        "start": (2, 2, 2, 3, 3, 3, 12, 12, 12, 13, 13, 13, 24, 24, 25, 25, 26, 26),
        "end": (14, 14, 15, 15, 16, 16, 25, 25, 25, 26, 26, 27, 38, 38, 39, 39, 40, 40),
    },
    "sparse_link": {
        "codes": tuple("DEFCGPQROMNDEFGPQR"),
        "release": (0, 0, 2, 3, 6, 8, 12, 13, 16, 18, 22, 24, 28, 30, 34, 36, 40, 43),
        "start": (1, 1, 4, 5, 8, 10, 14, 15, 18, 20, 24, 26, 30, 32, 36, 38, 42, 45),
        "end": (15, 15, 17, 18, 21, 23, 27, 28, 31, 33, 37, 39, 43, 45, 49, 51, 55, 58),
    },
    "tight_windows": {
        "codes": tuple("ABHIJKLCGOMNDEPQRF"),
        "release": (0, 0, 0, 1, 4, 5, 6, 8, 10, 12, 14, 17, 20, 23, 27, 31, 35, 40),
        "start": (0, 1, 1, 2, 5, 6, 7, 9, 11, 13, 15, 18, 21, 24, 28, 32, 36, 41),
        "end": (6, 7, 7, 7, 11, 10, 13, 18, 20, 22, 25, 28, 31, 33, 37, 40, 45, 51),
    },
    "burst_arrivals": {
        "codes": tuple("ABHIKPDEFPQRJLMNGO"),
        "release": (0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 36, 36, 36, 36, 36, 36),
        "start": (1, 1, 1, 2, 2, 2, 20, 20, 20, 21, 21, 21, 38, 38, 38, 39, 39, 39),
        "end": (13, 13, 13, 15, 15, 16, 32, 32, 33, 33, 34, 34, 51, 51, 52, 53, 54, 55),
    },
}
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "claims", "hub_actions"}
CLAIM_KEYS = {"robot", "task"}
HUB_ACTION_KEYS = {"hub", "enqueue"}
ENQUEUE_KEYS = {"receiver", "robot", "task", "status", "event_boundary"}
EVENT_STATUSES = {"claim", "cancel", "complete"}
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
        "result_evidence.slot_containers": 0.25,
        "result_evidence.claim_fields": 0.25,
        "result_evidence.hub_action_fields": 0.25,
        "result_evidence.enqueue_fields": 0.25,
    },
    "decision_legality": {
        "decision_legality.robot_availability": 0.25,
        "decision_legality.task_local_knowledge": 0.25,
        "decision_legality.enqueue_routes": 0.25,
        "decision_legality.enqueue_events": 0.25,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.release_and_views": 1 / 6,
        "replay_validity.commitments_and_conflicts": 1 / 6,
        "replay_validity.fifo_messages": 1 / 6,
        "replay_validity.travel_service_return": 1 / 6,
        "replay_validity.task_outcomes": 1 / 6,
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
        self.replayed: dict[str, Any] = {}
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


def _safe_repr(value: Any, limit: int = 180) -> str:
    try:
        rendered = repr(value)
    except (OverflowError, TypeError, ValueError):
        if _integer(value):
            return f"<integer bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


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
        raise ValueError("atom share differs from the declared share")
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


def _task_index(task_id: str) -> int:
    return int(task_id[1:])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _visible_hubs(position: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        hub for hub in HUBS if _distance(position, HUB_POSITIONS[hub]) <= SENSING_RADIUS + EPS
    )


def _task_specs(scene: str) -> dict[str, dict[str, Any]]:
    table = SCENE_TABLES[scene]
    return {
        task_id: {
            "task": task_id,
            "position": POSITION_CODES[table["codes"][index]],
            "value": TASK_VALUES[index],
            "release": table["release"][index],
            "start": table["start"][index],
            "end": table["end"][index],
            "visible_hubs": _visible_hubs(POSITION_CODES[table["codes"][index]]),
        }
        for index, task_id in enumerate(TASK_IDS)
    }


def _topology(scene: str, t: int) -> frozenset[tuple[int, int]]:
    if scene == "balanced_stream":
        return frozenset((a, b) for a in HUBS for b in HUBS if a != b)
    if scene in {"overlap_hotspot", "burst_arrivals"}:
        return frozenset({(0, 1), (1, 2), (2, 0)})
    if scene == "tight_windows":
        return frozenset({(0, 2), (2, 0), (1, 2), (2, 1)})
    if t < 20:
        return frozenset({(0, 1), (1, 2)})
    if t < 40:
        return frozenset({(2, 1), (1, 0)})
    return frozenset({(0, 2), (2, 0)})


def _base_duration(hub: int, task: Mapping[str, Any]) -> int:
    point = task["position"]
    home = HUB_POSITIONS[hub]
    return math.ceil(abs(point[0] - home[0]) + abs(point[1] - home[1]))


def _travel_delta(scene: str, robot: int, task_id: str, departure: int, leg: int) -> int:
    scenario_index = SCENES.index(scene)
    task_index = _task_index(task_id)
    return (
        17 * (scenario_index + 1)
        + 11 * (robot + 1)
        + 7 * (task_index + 1)
        + 5 * (departure + 1)
        + 3 * leg
    ) % 2


def _message_lost(scene: str, generated: int, sequence: int) -> bool:
    return (SCENES.index(scene) + generated + sequence) % 9 == 0


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = parsed and _shape(data, ROOT_KEYS) and isinstance(data["scenarios"], list)
    scenarios = data.get("scenarios") if parsed and isinstance(data, Mapping) else None
    containers_ok = isinstance(scenarios, list) and all(
        _shape(row, SCENE_KEYS) and isinstance(row["slots"], list) for row in scenarios
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "parse failed"),
        _atom(
            "result_json",
            "root",
            root_ok,
            "exact root key and scenarios array"
            if root_ok
            else "expected exact root key scenarios",
        ),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            (
                "scene containers have exact keys and slots arrays"
                if containers_ok
                else "each scene needs exact keys scenario and slots, with slots as an array"
            ),
        ),
    ]


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return []
    return [row for row in data["scenarios"] if isinstance(row, Mapping)]


def _raw_slot_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    scenarios = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(scenarios, list):
        return entries
    for scene_index, scene in enumerate(scenarios):
        if not isinstance(scene, Mapping) or not isinstance(scene.get("slots"), list):
            continue
        for slot_index, slot in enumerate(scene["slots"]):
            entries.append((f"scenarios[{scene_index}].slots[{slot_index}]", slot))
    return entries


def _raw_claim_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for path, slot in _raw_slot_entries(data):
        if not _shape(slot, SLOT_KEYS) or not isinstance(slot["claims"], list):
            continue
        for index, claim in enumerate(slot["claims"]):
            entries.append((f"{path}.claims[{index}]", claim))
    return entries


def _raw_hub_action_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for path, slot in _raw_slot_entries(data):
        if not _shape(slot, SLOT_KEYS) or not isinstance(slot["hub_actions"], list):
            continue
        for index, action in enumerate(slot["hub_actions"]):
            entries.append((f"{path}.hub_actions[{index}]", action))
    return entries


def _enqueue_fields_valid(value: Any) -> bool:
    return (
        _shape(value, ENQUEUE_KEYS)
        and _integer(value["receiver"])
        and value["receiver"] in HUBS
        and _integer(value["robot"])
        and value["robot"] in ROBOTS
        and isinstance(value["task"], str)
        and value["task"] in TASK_IDS
        and isinstance(value["status"], str)
        and value["status"] in EVENT_STATUSES
        and _integer(value["event_boundary"])
        and 0 <= value["event_boundary"] <= HORIZON
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    slots = _raw_slot_entries(data)
    slot_issue = next(
        (
            f"{path}: expected exact keys {sorted(SLOT_KEYS)} and claims/hub_actions arrays, "
            f"got {_safe_repr(slot)}"
            for path, slot in slots
            if not _shape(slot, SLOT_KEYS)
            or not isinstance(slot["claims"], list)
            or not isinstance(slot["hub_actions"], list)
        ),
        None,
    )
    claims = _raw_claim_entries(data)
    claim_issue = next(
        (
            f"{path}: expected exact robot/task fields with task null or canonical string, "
            f"got {_safe_repr(claim)}"
            for path, claim in claims
            if not _shape(claim, CLAIM_KEYS)
            or not (
                claim["task"] is None
                or (isinstance(claim["task"], str) and claim["task"] in TASK_IDS)
            )
        ),
        None,
    )
    hub_actions = _raw_hub_action_entries(data)
    hub_issue = next(
        (
            f"{path}: expected exact hub/enqueue fields and enqueue array, got {_safe_repr(action)}"
            for path, action in hub_actions
            if not _shape(action, HUB_ACTION_KEYS) or not isinstance(action["enqueue"], list)
        ),
        None,
    )
    enqueue_issue = next(
        (
            f"{path}.enqueue[{index}]: malformed intention {_safe_repr(request)}"
            for path, action in hub_actions
            if _shape(action, HUB_ACTION_KEYS) and isinstance(action["enqueue"], list)
            for index, request in enumerate(action["enqueue"])
            if not _enqueue_fields_valid(request)
        ),
        None,
    )
    return [
        _atom(
            "result_evidence",
            "slot_containers",
            slot_issue is None,
            "slot containers valid" if slot_issue is None else slot_issue,
        ),
        _atom(
            "result_evidence",
            "claim_fields",
            claim_issue is None,
            "claim fields valid" if claim_issue is None else claim_issue,
        ),
        _atom(
            "result_evidence",
            "hub_action_fields",
            hub_issue is None,
            "hub action fields valid" if hub_issue is None else hub_issue,
        ),
        _atom(
            "result_evidence",
            "enqueue_fields",
            enqueue_issue is None,
            "enqueue fields valid" if enqueue_issue is None else enqueue_issue,
        ),
    ]


def _key_defects(values: Sequence[Any], expected: set[Any], label: str) -> list[str]:
    counts: dict[Any, int] = {}
    defects: list[str] = []
    for value in values:
        try:
            counts[value] = counts.get(value, 0) + 1
        except TypeError:
            defects.append(f"{label} unhashable key={_safe_repr(value)}")
    for key in sorted(expected, key=str):
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"{label} missing key={key!r}")
        elif count > 1:
            defects.extend(f"{label} duplicate key={key!r}" for _ in range(count - 1))
    for key, count in counts.items():
        if key not in expected:
            defects.extend(f"{label} unknown key={_safe_repr(key)}" for _ in range(count))
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    rows = _scene_rows(data)
    defects = _key_defects([row.get("scenario") for row in rows], set(SCENES), "scenario")
    by_scene: dict[str, list[Mapping[str, Any]]] = {scene: [] for scene in SCENES}
    for row in rows:
        scene = row.get("scenario")
        if isinstance(scene, str) and scene in by_scene:
            by_scene[scene].append(row)
    for scene, matches in by_scene.items():
        if len(matches) != 1 or not isinstance(matches[0].get("slots"), list):
            continue
        slots = matches[0]["slots"]
        slot_values = [
            slot.get("t")
            if isinstance(slot, Mapping) and _integer(slot.get("t"))
            else (
                "invalid_slot_key",
                _safe_repr(slot.get("t") if isinstance(slot, Mapping) else slot),
            )
            for slot in slots
        ]
        defects.extend(_key_defects(slot_values, set(range(HORIZON)), f"{scene} slot"))
        by_t: dict[int, list[Mapping[str, Any]]] = {t: [] for t in range(HORIZON)}
        for slot in slots:
            if isinstance(slot, Mapping) and _integer(slot.get("t")) and slot["t"] in by_t:
                by_t[slot["t"]].append(slot)
        for t, slot_matches in by_t.items():
            if len(slot_matches) != 1:
                continue
            slot = slot_matches[0]
            if isinstance(slot.get("claims"), list):
                values = [
                    claim.get("robot")
                    if isinstance(claim, Mapping) and _integer(claim.get("robot"))
                    else (
                        "invalid_robot_key",
                        _safe_repr(claim.get("robot") if isinstance(claim, Mapping) else claim),
                    )
                    for claim in slot["claims"]
                ]
                defects.extend(_key_defects(values, set(ROBOTS), f"{scene} slot={t} robot"))
            if isinstance(slot.get("hub_actions"), list):
                values = [
                    action.get("hub")
                    if isinstance(action, Mapping) and _integer(action.get("hub"))
                    else (
                        "invalid_hub_key",
                        _safe_repr(action.get("hub") if isinstance(action, Mapping) else action),
                    )
                    for action in slot["hub_actions"]
                ]
                defects.extend(_key_defects(values, set(HUBS), f"{scene} slot={t} hub"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene_row in _scene_rows(data):
        scene = scene_row.get("scenario")
        slots = scene_row.get("slots")
        if not isinstance(scene, str) or scene not in SCENES or not isinstance(slots, list):
            continue
        for slot in slots:
            if not isinstance(slot, Mapping) or not _integer(slot.get("t")):
                continue
            t = slot["t"]
            if not 0 <= t < HORIZON:
                continue
            for claim in slot.get("claims", []) if isinstance(slot.get("claims"), list) else []:
                if (
                    isinstance(claim, Mapping)
                    and _integer(claim.get("robot"))
                    and claim["robot"] in ROBOTS
                ):
                    records.setdefault((scene, t, "robot", claim["robot"]), claim)
            for action in (
                slot.get("hub_actions", []) if isinstance(slot.get("hub_actions"), list) else []
            ):
                if (
                    isinstance(action, Mapping)
                    and _integer(action.get("hub"))
                    and action["hub"] in HUBS
                ):
                    records.setdefault((scene, t, "hub", action["hub"]), action)
    return records


def _raw_claim(records: Mapping[Hashable, Any], scene: str, t: int, robot: int) -> dict[str, Any]:
    value = records.get((scene, t, "robot", robot))
    if not isinstance(value, Mapping) or not _shape(value, CLAIM_KEYS):
        return {"robot": robot, "task": None}
    task = value.get("task")
    if task is not None and (not isinstance(task, str) or task not in TASK_IDS):
        task = None
    return {"robot": robot, "task": task}


def _raw_hub_action(
    records: Mapping[Hashable, Any], scene: str, t: int, hub: int
) -> dict[str, Any]:
    value = records.get((scene, t, "hub", hub))
    if not isinstance(value, Mapping) or not _shape(value, HUB_ACTION_KEYS):
        return {"hub": hub, "enqueue": []}
    enqueue = value.get("enqueue")
    if not isinstance(enqueue, list):
        enqueue = []
    return {
        "hub": hub,
        "enqueue": [dict(request) for request in enqueue if _enqueue_fields_valid(request)],
    }


def _normalize_policy_action(value: Any, hub: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    raw_claims = value.get("claims", {})
    claims: dict[int, Any] = {}
    if isinstance(raw_claims, Mapping):
        claims = {robot: raw_claims.get(robot) for robot in HUB_ROBOTS[hub]}
    elif isinstance(raw_claims, list):
        for item in raw_claims:
            if isinstance(item, Mapping) and item.get("robot") in HUB_ROBOTS[hub]:
                claims.setdefault(item["robot"], item.get("task"))
    enqueue = value.get("enqueue", [])
    if not isinstance(enqueue, list):
        enqueue = []
    return {
        "claims": [
            {
                "robot": robot,
                "task": claims.get(robot)
                if claims.get(robot) is None or claims.get(robot) in TASK_IDS
                else None,
            }
            for robot in HUB_ROBOTS[hub]
        ],
        "hub_action": {
            "hub": hub,
            "enqueue": [dict(request) for request in enqueue if _enqueue_fields_valid(request)],
        },
    }


def _event_key(event: Mapping[str, Any]) -> tuple[int, str, str, int]:
    return (
        int(event["robot"]),
        str(event["task"]),
        str(event["status"]),
        int(event["event_boundary"]),
    )


def _public_robot_state(robot: Mapping[str, Any], t: int) -> dict[str, Any]:
    phase = str(robot["phase"])
    due = robot.get("due")
    public_due = due if phase in {"waiting", "service", "cancel_delay"} else None
    departure = robot.get("departure")
    return {
        "phase": phase,
        "task": robot.get("task"),
        "departure_boundary": departure,
        "elapsed_slots": t - departure if _integer(departure) else None,
        "base_duration": robot.get("base"),
        "known_transition_boundary": public_due,
    }


def _local_view(
    scene: str,
    t: int,
    hub: int,
    tasks: Mapping[str, Mapping[str, Any]],
    robots: Sequence[Mapping[str, Any]],
    attempted: Mapping[int, set[str]],
    own_events: Mapping[int, Sequence[Mapping[str, Any]]],
    remote_status: Mapping[int, Mapping[tuple[int, str], Mapping[str, Any]]],
    queues: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    current_edges: frozenset[tuple[int, int]],
    observed_legs: Mapping[int, Sequence[Mapping[str, Any]]],
    histories: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    known_tasks = {
        task_id: {
            "task": task_id,
            "position": list(task["position"]),
            "value": task["value"],
            "release": task["release"],
            "window_start": task["start"],
            "window_end": task["end"],
            "base_duration": _base_duration(hub, task),
        }
        for task_id, task in tasks.items()
        if task["release"] <= t and hub in task["visible_hubs"]
    }
    public_queues: dict[int, list[dict[str, Any]]] = {}
    for receiver in HUBS:
        if receiver == hub:
            continue
        public_queues[receiver] = [
            {
                "receiver": receiver,
                "generated_boundary": message["generated_boundary"],
                "robot": message["robot"],
                "task": message["task"],
                "status": message["status"],
                "event_boundary": message["event_boundary"],
            }
            for message in queues[(hub, receiver)]
        ]
    remote = [
        dict(event)
        for _, event in sorted(
            remote_status[hub].items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    return {
        "t": t,
        "hub": hub,
        "tasks": known_tasks,
        "attempted_tasks": tuple(sorted(attempted[hub])),
        "robots": {robot: _public_robot_state(robots[robot], t) for robot in HUB_ROBOTS[hub]},
        "own_events": tuple(dict(event) for event in own_events[hub]),
        "observed_legs": tuple(dict(leg) for leg in observed_legs[hub]),
        "remote_status": tuple(remote),
        "current_receivers": tuple(
            receiver for receiver in HUBS if (hub, receiver) in current_edges
        ),
        "outgoing_queues": public_queues,
        "history": tuple(dict(item) for item in histories[hub]),
    }


def _remote_blocks_task(status: Mapping[tuple[int, str], Mapping[str, Any]], task_id: str) -> bool:
    return any(
        remote_task == task_id and event["status"] in {"claim", "complete"}
        for (_, remote_task), event in status.items()
    )


def _claim_reason(
    claim: Mapping[str, Any],
    t: int,
    hub: int,
    tasks: Mapping[str, Mapping[str, Any]],
    robots: Sequence[Mapping[str, Any]],
    attempted: Mapping[int, set[str]],
    remote_status: Mapping[int, Mapping[tuple[int, str], Mapping[str, Any]]],
) -> tuple[str | None, str | None]:
    robot = int(claim["robot"])
    task_id = claim["task"]
    if task_id is None:
        return None, None
    if robots[robot]["phase"] != "idle":
        return "robot_availability", f"robot phase is {robots[robot]['phase']!r}, expected 'idle'"
    task = tasks[task_id]
    if task["release"] > t or hub not in task["visible_hubs"]:
        return "task_local_knowledge", "task is not released and sensed at the home hub"
    if not t < task["end"]:
        return "task_local_knowledge", f"claim boundary must be < window_end={task['end']}"
    if task_id in attempted[hub]:
        return "task_local_knowledge", "task is in the home hub's permanent attempted set"
    if _remote_blocks_task(remote_status[hub], task_id):
        return "task_local_knowledge", "arrived remote claim or completion blocks the task"
    return None, None


def _apply_remote_event(
    store: dict[tuple[int, str], dict[str, Any]], message: Mapping[str, Any]
) -> None:
    key = (int(message["robot"]), str(message["task"]))
    candidate = {
        "robot": key[0],
        "task": key[1],
        "status": str(message["status"]),
        "event_boundary": int(message["event_boundary"]),
    }
    current = store.get(key)
    precedence = {"claim": 0, "cancel": 1, "complete": 2}
    if current is None or (candidate["event_boundary"], precedence[candidate["status"]]) > (
        current["event_boundary"],
        precedence[current["status"]],
    ):
        store[key] = candidate


def _simulate_scene(
    scene: str,
    records: Mapping[Hashable, Any] | None = None,
    policies: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    if (records is None) == (policies is None):
        raise ValueError("provide exactly one of records or policies")
    if policies is not None and len(policies) != len(HUBS):
        raise ValueError("one policy callable is required per hub")

    tasks = _task_specs(scene)
    robots: list[dict[str, Any]] = [
        {
            "phase": "idle",
            "task": None,
            "departure": None,
            "due": None,
            "base": None,
        }
        for _ in ROBOTS
    ]
    task_state = {
        task_id: {"owner": None, "completed": None, "expired": False} for task_id in TASK_IDS
    }
    attempted = {hub: set() for hub in HUBS}
    own_events: dict[int, list[dict[str, Any]]] = {hub: [] for hub in HUBS}
    remote_status: dict[int, dict[tuple[int, str], dict[str, Any]]] = {hub: {} for hub in HUBS}
    queues: dict[tuple[int, int], list[dict[str, Any]]] = {
        (sender, receiver): [] for sender in HUBS for receiver in HUBS if sender != receiver
    }
    pending: dict[int, list[dict[str, Any]]] = {}
    observed_legs: dict[int, list[dict[str, Any]]] = {hub: [] for hub in HUBS}
    histories: dict[int, list[dict[str, Any]]] = {hub: [] for hub in HUBS}
    first_attempt: dict[tuple[int, str], int] = {}
    decisions: list[dict[str, Any]] = []
    views_log: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    delivered_messages: list[dict[str, Any]] = []
    service_audits: list[dict[str, Any]] = []
    leg_logs: list[dict[str, Any]] = []
    service_logs: list[dict[str, Any]] = []
    duplicate_claims = 0
    next_sequence = 0
    legality_issues: dict[str, list[str]] = {
        "robot_availability": [],
        "task_local_knowledge": [],
        "enqueue_routes": [],
        "enqueue_events": [],
    }

    def emit(robot: int, task_id: str, status: str, boundary: int) -> None:
        own_events[ROBOT_HUB[robot]].append(
            {
                "robot": robot,
                "task": task_id,
                "status": status,
                "event_boundary": boundary,
            }
        )

    def start_return(robot: int, boundary: int) -> None:
        task_id = str(robots[robot]["task"])
        hub = ROBOT_HUB[robot]
        duration = _base_duration(hub, tasks[task_id]) + _travel_delta(
            scene, robot, task_id, boundary, 1
        )
        robots[robot].update(
            phase="return",
            departure=boundary,
            due=boundary + duration,
            base=_base_duration(hub, tasks[task_id]),
        )
        leg_logs.append(
            {
                "leg": 1,
                "robot": robot,
                "task": task_id,
                "departure": boundary,
                "duration": duration,
                "arrival": boundary + duration,
            }
        )

    def cancel(robot: int, boundary: int) -> None:
        if robots[robot]["phase"] not in {"outbound", "waiting", "service"}:
            return
        task_id = str(robots[robot]["task"])
        if task_state[task_id]["owner"] == robot:
            task_state[task_id]["owner"] = None
        robots[robot].update(phase="cancel_delay", departure=boundary, due=boundary + CANCEL_DELAY)
        emit(robot, task_id, "cancel", boundary)

    def process_boundary(boundary: int) -> None:
        for message in sorted(pending.pop(boundary, []), key=lambda item: item["sequence"]):
            _apply_remote_event(remote_status[message["receiver"]], message)
            delivered_messages.append(message)

        for robot in ROBOTS:
            if robots[robot]["phase"] == "return" and robots[robot]["due"] == boundary:
                observed_legs[ROBOT_HUB[robot]].append(
                    {
                        "leg": 1,
                        "robot": robot,
                        "task": robots[robot]["task"],
                        "departure_boundary": robots[robot]["departure"],
                        "arrival_boundary": boundary,
                        "actual_duration": boundary - robots[robot]["departure"],
                    }
                )
                robots[robot] = {
                    "phase": "idle",
                    "task": None,
                    "departure": None,
                    "due": None,
                    "base": None,
                }

        completing = [
            robot
            for robot in ROBOTS
            if robots[robot]["phase"] == "service" and robots[robot]["due"] == boundary
        ]
        for robot in completing:
            task_id = str(robots[robot]["task"])
            if task_state[task_id]["completed"] is not None:
                cancel(robot, boundary)
                continue
            task_state[task_id]["completed"] = boundary
            task_state[task_id]["owner"] = robot
            emit(robot, task_id, "complete", boundary)
            service_logs.append({"robot": robot, "task": task_id, "completion": boundary})
            start_return(robot, boundary)
            for other in ROBOTS:
                if other != robot and robots[other]["task"] == task_id:
                    cancel(other, boundary)

        arriving = sorted(
            robot
            for robot in ROBOTS
            if robots[robot]["phase"] == "outbound" and robots[robot]["due"] == boundary
        )
        for robot in arriving:
            task_id = str(robots[robot]["task"])
            task = tasks[task_id]
            observed_legs[ROBOT_HUB[robot]].append(
                {
                    "leg": 0,
                    "robot": robot,
                    "task": task_id,
                    "departure_boundary": robots[robot]["departure"],
                    "arrival_boundary": boundary,
                    "actual_duration": boundary - robots[robot]["departure"],
                }
            )
            owner = task_state[task_id]["owner"]
            eligible = max(boundary, task["start"]) + SERVICE_DURATION <= task["end"]
            if (
                task_state[task_id]["completed"] is not None
                or owner not in {None, robot}
                or not eligible
            ):
                cancel(robot, boundary)
                continue
            task_state[task_id]["owner"] = robot
            if boundary < task["start"]:
                robots[robot].update(phase="waiting", due=task["start"])
            else:
                robots[robot].update(phase="service", due=boundary + SERVICE_DURATION)
                service_logs.append({"robot": robot, "task": task_id, "start": boundary})

        for robot in ROBOTS:
            if robots[robot]["phase"] == "waiting" and robots[robot]["due"] == boundary:
                task_id = str(robots[robot]["task"])
                robots[robot].update(phase="service", due=boundary + SERVICE_DURATION)
                service_logs.append({"robot": robot, "task": task_id, "start": boundary})

        for robot in ROBOTS:
            if robots[robot]["phase"] == "cancel_delay" and robots[robot]["due"] == boundary:
                start_return(robot, boundary)

        for task_id, task in tasks.items():
            if task["end"] != boundary or task_state[task_id]["completed"] is not None:
                continue
            task_state[task_id]["expired"] = True
            for robot in ROBOTS:
                if robots[robot]["task"] == task_id:
                    cancel(robot, boundary)

    for t in range(HORIZON):
        process_boundary(t)
        current_edges = _topology(scene, t) if communication_enabled else frozenset()
        views = {
            hub: _local_view(
                scene,
                t,
                hub,
                tasks,
                robots,
                attempted,
                own_events,
                remote_status,
                queues,
                current_edges,
                observed_legs,
                histories,
            )
            for hub in HUBS
        }
        views_log.extend({"t": t, "hub": hub, "view": views[hub]} for hub in HUBS)

        actions: dict[int, dict[str, Any]] = {}
        for hub in HUBS:
            if policies is None:
                actions[hub] = {
                    "claims": [
                        _raw_claim(records or {}, scene, t, robot) for robot in HUB_ROBOTS[hub]
                    ],
                    "hub_action": _raw_hub_action(records or {}, scene, t, hub),
                }
            else:
                actions[hub] = _normalize_policy_action(policies[hub](views[hub]), hub)

        claims = sorted(
            [claim for hub in HUBS for claim in actions[hub]["claims"]],
            key=lambda claim: claim["robot"],
        )
        hub_actions = [actions[hub]["hub_action"] for hub in HUBS]
        decisions.append({"t": t, "claims": claims, "hub_actions": hub_actions})

        legal_claims: dict[int, str] = {}
        by_hub_task: dict[tuple[int, str], list[int]] = {}
        for claim in claims:
            robot = int(claim["robot"])
            task_id = claim["task"]
            if task_id is None:
                continue
            hub = ROBOT_HUB[robot]
            group, reason = _claim_reason(claim, t, hub, tasks, robots, attempted, remote_status)
            if group is not None:
                legality_issues[group].append(
                    f"scene={scene},t={t},hub={hub},robot={robot},task={task_id}: {reason}"
                )
                continue
            legal_claims[robot] = str(task_id)
            by_hub_task.setdefault((hub, str(task_id)), []).append(robot)
            first_attempt.setdefault((hub, str(task_id)), t)

        winners: list[tuple[int, str]] = []
        for (hub, task_id), contenders in sorted(by_hub_task.items()):
            attempted[hub].add(task_id)
            winner = min(contenders)
            winners.append((winner, task_id))
            duplicate_claims += len(contenders) - 1

        for robot, task_id in sorted(winners):
            other_commitment = any(
                other != robot
                and ROBOT_HUB[other] != ROBOT_HUB[robot]
                and robots[other]["task"] == task_id
                and robots[other]["phase"] in {"outbound", "waiting", "service"}
                for other in ROBOTS
            )
            if task_state[task_id]["completed"] is not None or other_commitment:
                duplicate_claims += 1
            hub = ROBOT_HUB[robot]
            base = _base_duration(hub, tasks[task_id])
            duration = base + _travel_delta(scene, robot, task_id, t, 0)
            robots[robot].update(
                phase="outbound",
                task=task_id,
                departure=t,
                due=t + duration,
                base=base,
            )
            leg_logs.append(
                {
                    "leg": 0,
                    "robot": robot,
                    "task": task_id,
                    "departure": t,
                    "duration": duration,
                    "arrival": t + duration,
                }
            )
            emit(robot, task_id, "claim", t)

        accepted_requests: list[tuple[int, dict[str, Any]]] = []
        for hub, action in zip(HUBS, hub_actions):
            requests = action["enqueue"]
            route_failures: list[str] = []
            event_failures: list[str] = []
            receivers: set[int] = set()
            frozen_events = {_event_key(event) for event in views[hub]["own_events"]}
            for request in requests:
                receiver = int(request["receiver"])
                if receiver in receivers:
                    route_failures.append(f"receiver={receiver} appears more than once")
                receivers.add(receiver)
                if (hub, receiver) not in current_edges:
                    route_failures.append(
                        f"edge=({hub},{receiver}) is unavailable at this boundary"
                    )
                if ROBOT_HUB[int(request["robot"])] != hub:
                    event_failures.append(
                        f"robot={request['robot']} is not owned by sender hub={hub}"
                    )
                elif _event_key(request) not in frozen_events:
                    event_failures.append(
                        "event was not authentically observed before the frozen view"
                    )
            if route_failures:
                legality_issues["enqueue_routes"].append(
                    f"scene={scene},t={t},hub={hub}: {'; '.join(route_failures)}"
                )
            if event_failures:
                legality_issues["enqueue_events"].append(
                    f"scene={scene},t={t},hub={hub}: {'; '.join(event_failures)}"
                )
            if not route_failures and not event_failures:
                accepted_requests.extend((hub, dict(request)) for request in requests)

        for hub, request in sorted(
            accepted_requests, key=lambda item: (item[0], item[1]["receiver"])
        ):
            message = {
                **request,
                "sender": hub,
                "generated_boundary": t,
                "sequence": next_sequence,
                "queue_full_drop": False,
                "service_slot": None,
                "lost": None,
                "arrival_boundary": None,
            }
            next_sequence += 1
            edge = (hub, int(request["receiver"]))
            if len(queues[edge]) >= QUEUE_CAPACITY:
                message["queue_full_drop"] = True
            else:
                queues[edge].append(message)
            messages.append(message)

        candidate_heads = [
            queue[0] for edge, queue in queues.items() if edge in current_edges and queue
        ]
        selected = sorted(
            candidate_heads,
            key=lambda message: (
                message["generated_boundary"],
                message["sender"],
                message["sequence"],
            ),
        )[:MAX_SERVED_MESSAGES]
        selected_sequences = {message["sequence"] for message in selected}
        service_audits.append(
            {
                "t": t,
                "candidate_sequences": tuple(message["sequence"] for message in candidate_heads),
                "selected_sequences": tuple(message["sequence"] for message in selected),
            }
        )
        for edge in sorted(current_edges):
            if queues[edge] and queues[edge][0]["sequence"] in selected_sequences:
                message = queues[edge].pop(0)
                message["service_slot"] = t
                message["lost"] = _message_lost(
                    scene, message["generated_boundary"], message["sequence"]
                )
                if not message["lost"]:
                    message["arrival_boundary"] = t + 1
                    pending.setdefault(t + 1, []).append(message)

        for hub in HUBS:
            prior_view = {key: value for key, value in views[hub].items() if key != "history"}
            histories[hub].append(
                {
                    "t": t,
                    "view": prior_view,
                    "claims": tuple(
                        {"robot": claim["robot"], "task": claim["task"]}
                        for claim in claims
                        if ROBOT_HUB[claim["robot"]] == hub
                    ),
                    "enqueue": tuple(
                        dict(request) for request in actions[hub]["hub_action"]["enqueue"]
                    ),
                    "observed_event_count": len(views[hub]["own_events"]),
                }
            )

    process_boundary(HORIZON)
    return {
        "scene": scene,
        "slots": decisions,
        "consumed_decisions": HORIZON * (len(ROBOTS) + len(HUBS)),
        "views": views_log,
        "tasks": tasks,
        "task_state": task_state,
        "robots": robots,
        "attempted": attempted,
        "own_events": own_events,
        "remote_status": remote_status,
        "queues": queues,
        "messages": messages,
        "delivered_messages": delivered_messages,
        "service_audits": service_audits,
        "leg_logs": leg_logs,
        "service_logs": service_logs,
        "first_attempt": first_attempt,
        "duplicate_claims": duplicate_claims,
        "legality_issues": legality_issues,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    return _simulate_scene(scene, records=records)


def _online_scene(
    scene: str,
    policies: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]],
    communication_enabled: bool = True,
) -> dict[str, Any]:
    return _simulate_scene(scene, policies=policies, communication_enabled=communication_enabled)


def _result_from_online(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": [{"scenario": scene, "slots": replayed[scene]["slots"]} for scene in SCENES]
    }


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    return [
        _atom(
            "decision_legality",
            name,
            not (
                issues := [
                    issue
                    for scene in SCENES
                    if isinstance(replayed.get(scene), Mapping)
                    for issue in replayed[scene]["legality_issues"][name]
                ]
            ),
            "no violations" if not issues else f"violations={issues[:10]}",
        )
        for name in (
            "robot_availability",
            "task_local_knowledge",
            "enqueue_routes",
            "enqueue_events",
        )
    ]


def _canonical_enqueue(value: Any) -> tuple[tuple[int, int, str, str, int], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            (
                request["receiver"],
                request["robot"],
                request["task"],
                request["status"],
                request["event_boundary"],
            )
            for request in value
            if _enqueue_fields_valid(request)
        )
    )


def _decision_issue(
    scene: str,
    replay: Mapping[str, Any],
    records: Mapping[Hashable, Any],
) -> str | None:
    expected = HORIZON * (len(ROBOTS) + len(HUBS))
    if replay.get("consumed_decisions") != expected:
        return f"expected consumed_decisions={expected}, got={replay.get('consumed_decisions')}"
    slots = replay.get("slots")
    if not isinstance(slots, list) or len(slots) != HORIZON:
        return f"expected {HORIZON} decision slots"
    for t, slot in enumerate(slots):
        if not isinstance(slot, Mapping) or slot.get("t") != t:
            return f"slot order mismatch at expected t={t}"
        claims = slot.get("claims")
        hubs = slot.get("hub_actions")
        if not isinstance(claims, list) or [item.get("robot") for item in claims] != list(ROBOTS):
            return f"claim consumption mismatch at t={t}"
        if not isinstance(hubs, list) or [item.get("hub") for item in hubs] != list(HUBS):
            return f"hub-action consumption mismatch at t={t}"
        for robot, actual in zip(ROBOTS, claims):
            key = (scene, t, "robot", robot)
            if key not in records:
                continue
            expected_claim = _raw_claim(records, scene, t, robot)
            if actual != expected_claim:
                return (
                    f"submitted claim was not consumed at scene={scene},t={t},robot={robot},"
                    f"expected={_safe_repr(expected_claim)},actual={_safe_repr(actual)}"
                )
        for hub, actual in zip(HUBS, hubs):
            key = (scene, t, "hub", hub)
            if key not in records:
                continue
            expected_action = _raw_hub_action(records, scene, t, hub)
            if actual.get("hub") != hub or _canonical_enqueue(
                actual.get("enqueue")
            ) != _canonical_enqueue(expected_action["enqueue"]):
                return (
                    f"submitted hub action was not consumed at scene={scene},t={t},hub={hub},"
                    f"expected={_safe_repr(expected_action)},actual={_safe_repr(actual)}"
                )
    return None


def _view_issue(replay: Mapping[str, Any]) -> str | None:
    views = replay.get("views")
    tasks = replay.get("tasks")
    if not isinstance(views, list) or len(views) != HORIZON * len(HUBS):
        return f"expected frozen views={HORIZON * len(HUBS)}"
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASK_IDS):
        return "canonical task table missing from replay"
    for item in views:
        if (
            not isinstance(item, Mapping)
            or item.get("hub") not in HUBS
            or item.get("t") not in range(HORIZON)
        ):
            return f"malformed frozen view key={_safe_repr(item)}"
        hub = item["hub"]
        t = item["t"]
        view = item.get("view")
        if not isinstance(view, Mapping):
            return f"missing view at t={t},hub={hub}"
        if t == 0:
            robot_states = view.get("robots")
            queues = view.get("outgoing_queues")
            initial_ok = (
                isinstance(robot_states, Mapping)
                and all(state.get("phase") == "idle" for state in robot_states.values())
                and not view.get("attempted_tasks")
                and not view.get("own_events")
                and not view.get("observed_legs")
                and not view.get("remote_status")
                and not view.get("history")
                and isinstance(queues, Mapping)
                and all(not queue for queue in queues.values())
            )
            if not initial_ok:
                return f"public initial state mismatch at t=0,hub={hub}"
        expected_tasks = {
            task_id
            for task_id, task in tasks.items()
            if task["release"] <= t and hub in task["visible_hubs"]
        }
        actual_tasks = (
            set(view.get("tasks", {})) if isinstance(view.get("tasks"), Mapping) else set()
        )
        if actual_tasks != expected_tasks:
            return (
                f"release/view mismatch at t={t},hub={hub},"
                f"expected={sorted(expected_tasks)},actual={sorted(actual_tasks)}"
            )
    return None


def _commitment_issue(replay: Mapping[str, Any]) -> str | None:
    states = replay.get("task_state")
    robots = replay.get("robots")
    duplicate = replay.get("duplicate_claims")
    if not isinstance(states, Mapping) or set(states) != set(TASK_IDS):
        return "task state set is incomplete"
    if not isinstance(robots, list) or len(robots) != len(ROBOTS):
        return "robot terminal state set is incomplete"
    if not _integer(duplicate) or duplicate < 0:
        return f"duplicate-claim count invalid={_safe_repr(duplicate)}"
    valid_phases = {"idle", "outbound", "waiting", "service", "cancel_delay", "return"}
    for robot, state in enumerate(robots):
        if not isinstance(state, Mapping) or state.get("phase") not in valid_phases:
            return f"robot={robot} terminal phase invalid"
        if state["phase"] == "idle" and state.get("task") is not None:
            return f"idle robot={robot} retains a task"
    for task_id, state in states.items():
        if not isinstance(state, Mapping) or state.get("owner") not in {*ROBOTS, None}:
            return f"task={task_id} owner invalid"
        active_service = [
            robot
            for robot in ROBOTS
            if robots[robot].get("task") == task_id
            and robots[robot].get("phase") in {"waiting", "service"}
        ]
        if len(active_service) > 1:
            return f"task={task_id} has multiple service owners={active_service}"
        if active_service and state.get("owner") != active_service[0]:
            return f"task={task_id} owner does not match active service robot"
    return None


def _fifo_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    messages = replay.get("messages")
    queues = replay.get("queues")
    audits = replay.get("service_audits")
    if not isinstance(messages, list) or not isinstance(queues, Mapping):
        return "message or queue replay missing"
    if [message.get("sequence") for message in messages] != list(range(len(messages))):
        return "message sequences are not consecutive in canonical generation order"
    if any(not isinstance(queue, list) or len(queue) > QUEUE_CAPACITY for queue in queues.values()):
        return f"terminal FIFO exceeds capacity={QUEUE_CAPACITY}"
    message_by_sequence = {message["sequence"]: message for message in messages}
    if not isinstance(audits, list) or len(audits) != HORIZON:
        return f"expected FIFO service audits={HORIZON}"
    for audit in audits:
        candidates = [message_by_sequence[seq] for seq in audit["candidate_sequences"]]
        expected = tuple(
            message["sequence"]
            for message in sorted(
                candidates,
                key=lambda message: (
                    message["generated_boundary"],
                    message["sender"],
                    message["sequence"],
                ),
            )[:MAX_SERVED_MESSAGES]
        )
        if tuple(audit["selected_sequences"]) != expected:
            return f"FIFO head selection mismatch at t={audit.get('t')}"
    for message in messages:
        if message["queue_full_drop"] and message["service_slot"] is not None:
            return f"queue-full message sequence={message['sequence']} was served"
        if message["service_slot"] is not None:
            expected_lost = _message_lost(scene, message["generated_boundary"], message["sequence"])
            if message["lost"] is not expected_lost:
                return f"loss mismatch for message sequence={message['sequence']}"
            expected_arrival = None if expected_lost else message["service_slot"] + 1
            if message["arrival_boundary"] != expected_arrival:
                return f"arrival mismatch for message sequence={message['sequence']}"
    return None


def _travel_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    tasks = replay.get("tasks")
    legs = replay.get("leg_logs")
    if not isinstance(tasks, Mapping) or not isinstance(legs, list):
        return "travel replay missing"
    for leg in legs:
        if not isinstance(leg, Mapping):
            return "travel log contains a non-object"
        robot = leg.get("robot")
        task_id = leg.get("task")
        departure = leg.get("departure")
        direction = leg.get("leg")
        if (
            robot not in ROBOTS
            or task_id not in TASK_IDS
            or not _integer(departure)
            or direction not in {0, 1}
        ):
            return f"travel leg key invalid={_safe_repr(leg)}"
        expected = _base_duration(ROBOT_HUB[robot], tasks[task_id]) + _travel_delta(
            scene, robot, task_id, departure, direction
        )
        if leg.get("duration") != expected or leg.get("arrival") != departure + expected:
            return (
                f"travel mismatch robot={robot},task={task_id},leg={direction},"
                f"expected_duration={expected},actual_duration={leg.get('duration')},"
                f"abs_tolerance={EPS}"
            )
    return None


def _outcome_issue(replay: Mapping[str, Any]) -> str | None:
    tasks = replay.get("tasks")
    states = replay.get("task_state")
    services = replay.get("service_logs")
    if (
        not isinstance(tasks, Mapping)
        or not isinstance(states, Mapping)
        or not isinstance(services, list)
    ):
        return "task outcome replay missing"
    completion_logs: dict[str, list[int]] = {task_id: [] for task_id in TASK_IDS}
    for item in services:
        if "completion" in item:
            completion_logs[item["task"]].append(item["completion"])
    for task_id in TASK_IDS:
        completed = states[task_id]["completed"]
        logs = completion_logs[task_id]
        if completed is None and logs:
            return f"task={task_id} has an unregistered completion log"
        if completed is not None:
            if logs != [completed]:
                return f"task={task_id} completion is not unique"
            if completed > tasks[task_id]["end"]:
                return (
                    f"task={task_id},actual_completion={completed},"
                    f"expected_latest={tasks[task_id]['end']},abs_tolerance={EPS}"
                )
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    issues: dict[str, list[str]] = {
        name.split(".", 1)[1]: [] for name in ATOM_SHARES["replay_validity"]
    }
    for scene in SCENES:
        replay = replayed.get(scene)
        if not isinstance(replay, Mapping):
            for values in issues.values():
                values.append(f"scene={scene}: replay missing")
            continue
        checks = {
            "decision_consumption": _decision_issue(scene, replay, records),
            "release_and_views": _view_issue(replay),
            "commitments_and_conflicts": _commitment_issue(replay),
            "fifo_messages": _fifo_issue(scene, replay),
            "travel_service_return": _travel_issue(scene, replay),
            "task_outcomes": _outcome_issue(replay),
        }
        for name, issue in checks.items():
            if issue is not None:
                issues[name].append(f"scene={scene}: {issue}")
    return [
        _atom(
            "replay_validity",
            name,
            not values,
            "replay invariant passed" if not values else f"issues={values[:8]}",
        )
        for name, values in issues.items()
    ]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scene_metrics(replay: Mapping[str, Any]) -> dict[str, float]:
    tasks = replay["tasks"]
    states = replay["task_state"]
    completed_value = sum(
        task["value"]
        for task_id, task in tasks.items()
        if states[task_id]["completed"] is not None and states[task_id]["completed"] <= task["end"]
    )
    temporal_value = 0.0
    for task_id, task in tasks.items():
        completion = states[task_id]["completed"]
        if completion is None or completion > task["end"]:
            continue
        quality = _clip(1.0 - (completion - task["start"]) / (task["end"] - task["start"]))
        temporal_value += task["value"] * quality

    generated = len(replay["messages"])
    delivered = len(replay["delivered_messages"])
    timely = sum(
        message["arrival_boundary"] <= tasks[message["task"]]["end"]
        for message in replay["delivered_messages"]
    )
    delivery_quality = (
        0.5 * delivered / generated + 0.5 * (timely / delivered if delivered else 0.0)
        if generated
        else 0.0
    )

    preclaim_keys: set[tuple[int, str]] = set()
    for message in sorted(
        replay["delivered_messages"],
        key=lambda item: (item["arrival_boundary"], item["sequence"]),
    ):
        receiver = message["receiver"]
        task_id = message["task"]
        task = tasks[task_id]
        key = (receiver, task_id)
        if key in preclaim_keys or message["status"] not in {"claim", "complete"}:
            continue
        arrival = message["arrival_boundary"]
        first = replay["first_attempt"].get(key, HORIZON + 1)
        if (
            receiver in task["visible_hubs"]
            and task["release"] <= arrival <= task["end"]
            and arrival < first
        ):
            preclaim_keys.add(key)
    coordination = (
        0.6 * len(preclaim_keys) / generated
        + 0.4 * _clip(1.0 - replay["duplicate_claims"] / len(TASK_IDS))
        if generated
        else 0.0
    )
    return {
        "primary_metrics_1": completed_value / TOTAL_VALUE,
        "primary_metrics_2": temporal_value / TOTAL_VALUE,
        "auxiliary_metrics_1": delivery_quality,
        "auxiliary_metrics_2": coordination,
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values = {
        name: sum(_scene_metrics(replayed[scene])[name] for scene in SCENES) / len(SCENES)
        for name in METRIC_NAMES
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"metric out of range: {values}")
    return values


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8.0 * metrics["primary_metrics_1"]
        + 7.0 * metrics["primary_metrics_2"]
        + 3.0 * metrics["auxiliary_metrics_1"]
        + 2.0 * metrics["auxiliary_metrics_2"]
    ) / 20.0


def _gated_metric(
    scores: Mapping[str, float],
    official_ok: bool,
    metrics: Mapping[str, float] | None,
    name: str | None,
) -> tuple[float, str]:
    failed: list[str] = []
    if not official_ok:
        failed.append("official_run")
    failed.extend(
        f"{check_id}={scores.get(check_id, 0.0):.6f}"
        for check_id in VALIDITY_CHECKS
        if scores.get(check_id, 0.0) < 1.0 - EPS
    )
    if failed:
        return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
    if name is None:
        return 1.0, "validity gate passed"
    if name not in METRIC_NAMES:
        raise ValueError(f"unknown metric name {name!r}")
    if metrics is None or name not in metrics:
        return 0.0, f"validity gate passed but {name} is unavailable; metric=0"
    value = metrics[name]
    if not _number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"invalid gated metric {name}={value!r}")
    return float(value), "validity gate passed"


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    visible_lines = ["" for _ in lines]
    visible = [False] * len(lines)
    in_fence = False
    fence_marker = ""
    in_comment = False
    for index, line in enumerate(lines):
        if in_fence:
            if line.lstrip().startswith(fence_marker):
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
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    try:
        if not _regular(ws, "report.md"):
            return 0.0, "report.md missing or not a regular file"
        report = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(report)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


def _fail_run_checks(state: EvalState, detail: str) -> EvalState:
    state.official_ok = False
    state.official_message = detail
    for name in VALIDITY_CHECKS:
        state.scores[name] = 0.0
        state.diagnostics[name] = f"official run failed; {detail}"
    return state


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        except TimeoutError:
            state.timed_out = True
            return _fail_run_checks(state, "pre-run results cleanup timed out after 30s")
        except Exception as exc:  # noqa: BLE001
            return _fail_run_checks(
                state, f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        started = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - started
            state.timed_out = True
            return _fail_run_checks(
                state, f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - started
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; elapsed={state.elapsed_s:.3f}s",
            )
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - started
        state.timed_out = state.exit_code is None and state.elapsed_s >= 600.0
        state.result_file_is_new_regular = getattr(removed, "exit_code", None) == 0 and _regular(
            ws, "results.json"
        )
        state.official_ok = (
            state.run_started and state.exit_code == 0 and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        )
        if not state.official_ok:
            return _fail_run_checks(state, state.official_message)
        try:
            state.data = _strict_json(ws.read_text("results.json") or "")
            state.parse_ok = True
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.parse_ok:
            state.records = _index_records(state.data)
            evidence_atoms = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(
                evidence_atoms, ATOM_SHARES["result_evidence"]
            )
            state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
            coverage, defects = _coverage_score(state.data)
            state.scores["canonical_coverage"] = coverage
            state.diagnostics["canonical_coverage"] = (
                f"defect_count={len(defects)}; defects={defects[:12]}"
            )
            state.replayed = {scene: _replay_scene(scene, state.records) for scene in SCENES}
            legality_atoms = _legality_atoms(state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                legality_atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
            replay_atoms = _replay_atoms(state.records, state.replayed)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
            gate_score, _ = _gated_metric(state.scores, state.official_ok, None, None)
            state.validity_gate = gate_score >= 1.0 - EPS
            if state.validity_gate:
                state.metrics = _metrics(state.replayed)
        else:
            for name in VALIDITY_CHECKS[1:]:
                state.scores[name] = 0.0
                state.diagnostics[name] = "strict JSON parse failed"
    except Exception as exc:  # noqa: BLE001
        detail = f"grader failure: {type(exc).__name__}: {exc}"
        state.official_message = (
            f"{state.official_message}; {detail}" if state.official_message else detail
        )
        for name in VALIDITY_CHECKS:
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
    return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    task = _INFLIGHT.get(key)
    if task is None:
        task = asyncio.create_task(_build_state_uncached(ws))
        _INFLIGHT[key] = task
    try:
        state = await asyncio.shield(task)
        _CACHE[key] = state
        return state
    finally:
        if task.done() and _INFLIGHT.get(key) is task:
            del _INFLIGHT[key]


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    try:
        if not _regular(ws, "run.py"):
            return 0.0, "run.py missing or not a regular file"
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except TimeoutError:
        return 0.0, "static compilation timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"static compilation raised {type(exc).__name__}: {exc}"
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    try:
        return _report_score(ws)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report contract check raised {type(exc).__name__}: {exc}"


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in VALIDITY_CHECKS:

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in METRIC_NAMES:

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        return _gated_metric(state.scores, state.official_ok, state.metrics, _id)

    globals()[_name] = check(id=_name)(_metric)
del _metric
