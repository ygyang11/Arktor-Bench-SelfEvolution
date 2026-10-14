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
DT = 0.5
EPS = 1e-6
SCENES = ("open_crossing", "narrow_merge", "delay_corridor", "shared_bottleneck", "recovery_turn")
ROBOTS = (0, 1, 2)
MAX_V = 0.6
MAX_OMEGA = 1.2
GOAL_RADIUS = 0.3
BASE_SEPARATION = 0.5
QOC_ERROR_SCALE = (8 * DT * MAX_V) ** 2
QUEUE_CAPACITY = 4
CHANNEL_CAPACITY = 2
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots", "audit"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "velocity", "send", "mode"}
AUDIT_KEYS = {"checkpoints", "packet_summary", "goal_arrivals"}
CHECKPOINT_KEYS = {"boundary", "robots"}
AUDIT_ROBOT_KEYS = {
    "robot",
    "position",
    "heading",
    "active_velocity",
    "active_generation",
    "age",
    "unsafe",
}
PACKET_SUMMARY_KEYS = {"attempts", "admitted", "effective_updates"}
AUDIT_BOUNDARIES = (20, 40, 60)
MODES = {"best_effort", "reliable"}
STARTS = {
    "open_crossing": ((1.0, 1.0, 0.0), (1.0, 7.0, 0.0), (10.0, 4.0, 3.141593)),
    "narrow_merge": ((1.0, 1.0, 0.0), (1.0, 7.0, 0.0), (2.0, 4.0, 0.0)),
    "delay_corridor": ((1.0, 4.0, 0.0), (2.0, 1.0, 1.570796), (2.0, 7.0, -1.570796)),
    "shared_bottleneck": ((1.0, 1.0, 0.0), (1.0, 7.0, 0.0), (10.0, 4.0, 3.141593)),
    "recovery_turn": ((1.0, 1.0, 0.0), (1.0, 7.0, 0.0), (10.0, 4.0, 3.141593)),
}
GOALS = {
    "open_crossing": ((11.0, 7.0), (11.0, 1.0), (1.0, 4.0)),
    "narrow_merge": ((11.0, 4.0), (10.0, 6.0), (10.0, 2.0)),
    "delay_corridor": ((11.0, 4.0), (10.0, 7.0), (10.0, 1.0)),
    "shared_bottleneck": ((11.0, 7.0), (11.0, 1.0), (1.0, 4.0)),
    "recovery_turn": ((10.0, 4.0), (10.0, 6.0), (2.0, 4.0)),
}
DEADLINES = {
    "open_crossing": (42, 42, 42),
    "narrow_merge": (54, 54, 54),
    "delay_corridor": (52, 52, 52),
    "shared_bottleneck": (48, 48, 48),
    "recovery_turn": (50, 50, 50),
}
OBSTACLES = {
    "open_crossing": (),
    "narrow_merge": ((5.0, 0.0, 7.0, 3.0), (5.0, 5.0, 7.0, 8.0)),
    "delay_corridor": ((4.0, 0.0, 8.0, 2.5), (4.0, 5.5, 8.0, 8.0)),
    "shared_bottleneck": ((5.0, 3.0, 7.0, 5.0),),
    "recovery_turn": ((4.0, 0.0, 6.0, 3.0), (4.0, 5.0, 6.0, 8.0)),
}
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
        "result_evidence.action_containers": 0.20,
        "result_evidence.action_values": 0.30,
        "result_evidence.audit_containers": 0.15,
        "result_evidence.audit_values": 0.15,
    },
    "decision_legality": {
        "decision_legality.command": 0.25,
        "decision_legality.queue": 0.20,
        "decision_legality.mode": 0.15,
        "decision_legality.motion": 0.20,
        "decision_legality.safety": 0.20,
    },
    "replay_validity": {
        "replay_validity.timeline": 0.16666666666666666,
        "replay_validity.motion": 0.16666666666666666,
        "replay_validity.channel": 0.16666666666666666,
        "replay_validity.commands": 0.16666666666666666,
        "replay_validity.goals": 0.16666666666666666,
        "replay_validity.safety": 0.16666666666666666,
    },
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


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0, abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share mismatch")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[{}]; {}".format(
        ",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed)
    )


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    return (
        data.get("scenarios", [])
        if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list)
        else []
    )


def _result_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = parsed and _shape(data, ROOT_KEYS) and isinstance(data.get("scenarios"), list)
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    containers = isinstance(rows, list) and all(
        _shape(row, SCENE_KEYS)
        and isinstance(row.get("slots"), list)
        and isinstance(row.get("audit"), Mapping)
        for row in rows
    )
    actual_root = sorted(data) if isinstance(data, Mapping) else type(data).__name__
    invalid_scenes = (
        []
        if not isinstance(rows, list)
        else [
            f"scenarios[{index}]"
            for index, row in enumerate(rows)
            if not (
                _shape(row, SCENE_KEYS)
                and isinstance(row.get("slots"), list)
                and isinstance(row.get("audit"), Mapping)
            )
        ]
    )
    return [
        _atom(
            "result_json",
            "parse",
            parsed,
            "strict JSON parsed" if parsed else "strict JSON parse failed",
        ),
        _atom(
            "result_json",
            "root",
            root,
            "exact scenarios root"
            if root
            else (
                "path=$: expected exact keys=['scenarios'] with a list value, "
                f"actual={actual_root!r}"
            ),
        ),
        _atom(
            "result_json",
            "scene_containers",
            containers,
            "scene containers valid"
            if containers
            else (
                "path=scenarios: expected a list of exact scenario/slots/audit objects; "
                f"invalid_paths={invalid_scenes[:6]}"
            ),
        ),
    ]


def _valid_velocity(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_number(v) for v in value)
        and -EPS <= float(value[0]) <= MAX_V + EPS
        and -MAX_OMEGA - EPS <= float(value[1]) <= MAX_OMEGA + EPS
    )


def _valid_action(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not _valid_velocity(value.get("velocity")) or not isinstance(value.get("send"), bool):
        return False
    mode = value.get("mode")
    return mode is None or (isinstance(mode, str) and mode in MODES)


def _action_value_issue(action: Mapping[str, Any], path: str) -> str | None:
    if not _valid_velocity(action.get("velocity")):
        return (
            f"path={path}.velocity: expected [v,omega] with finite numbers, "
            f"v in [0,{MAX_V}], omega in [-{MAX_OMEGA},{MAX_OMEGA}], "
            f"tolerance={EPS}; actual={action.get('velocity')!r}"
        )
    if not isinstance(action.get("send"), bool):
        return f"path={path}.send: expected Boolean; actual={action.get('send')!r}"
    mode = action.get("mode")
    if not (mode is None or (isinstance(mode, str) and mode in MODES)):
        return f"path={path}.mode: expected null or one of {sorted(MODES)!r}; actual={mode!r}"
    return None


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = _scene_rows(data)
    if not (
        isinstance(data, Mapping)
        and isinstance(data.get("scenarios"), list)
        and all(isinstance(row, Mapping) for row in rows)
    ):
        reason = (
            "path=scenarios: expected a list of scene objects before record evidence "
            "can be assessed"
        )
        return [
            _atom("result_evidence", "slot_containers", False, reason),
            _atom("result_evidence", "action_containers", False, reason),
            _atom("result_evidence", "action_values", False, reason),
            _atom("result_evidence", "audit_containers", False, reason),
            _atom("result_evidence", "audit_values", False, reason),
        ]
    slot_entries = [
        (f"scenarios[{scene_index}].slots[{slot_index}]", slot)
        for scene_index, row in enumerate(rows)
        if isinstance(row, Mapping) and isinstance(row.get("slots"), list)
        for slot_index, slot in enumerate(row["slots"])
    ]
    slot_issues = [
        f"path={path}: expected exact keys={sorted(SLOT_KEYS)!r} and actions list"
        for path, slot in slot_entries
        if not (_shape(slot, SLOT_KEYS) and isinstance(slot.get("actions"), list))
    ]
    action_entries = [
        (f"{path}.actions[{action_index}]", action)
        for path, slot in slot_entries
        if isinstance(slot, Mapping) and isinstance(slot.get("actions"), list)
        for action_index, action in enumerate(slot["actions"])
    ]
    action_shape_issues = [
        f"path={path}: expected exact keys={sorted(ACTION_KEYS)!r}"
        for path, action in action_entries
        if not _shape(action, ACTION_KEYS)
    ]
    value_issues = [
        issue
        for path, action in action_entries
        if _shape(action, ACTION_KEYS)
        for issue in [_action_value_issue(action, path)]
        if issue is not None
    ]
    audit_container_issues: list[str] = []
    audit_value_issues: list[str] = []
    for scene_index, row in enumerate(rows):
        audit_path = f"scenarios[{scene_index}].audit"
        audit = row.get("audit")
        if not _shape(audit, AUDIT_KEYS):
            audit_container_issues.append(
                f"path={audit_path}: expected exact keys={sorted(AUDIT_KEYS)!r}; actual={audit!r}"
            )
            continue
        checkpoints = audit.get("checkpoints")
        packet_summary = audit.get("packet_summary")
        goal_arrivals = audit.get("goal_arrivals")
        if not isinstance(checkpoints, list) or len(checkpoints) != len(AUDIT_BOUNDARIES):
            audit_container_issues.append(
                f"path={audit_path}.checkpoints: expected list "
                f"length={len(AUDIT_BOUNDARIES)}; actual={checkpoints!r}"
            )
        else:
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                checkpoint_path = f"{audit_path}.checkpoints[{checkpoint_index}]"
                if not (
                    _shape(checkpoint, CHECKPOINT_KEYS)
                    and isinstance(checkpoint.get("robots"), list)
                    and len(checkpoint["robots"]) == len(ROBOTS)
                ):
                    audit_container_issues.append(
                        f"path={checkpoint_path}: expected exact boundary/robots object "
                        f"with robots length={len(ROBOTS)}; actual={checkpoint!r}"
                    )
                    continue
                if not _integer(checkpoint.get("boundary")):
                    audit_value_issues.append(
                        f"path={checkpoint_path}.boundary: expected integer; "
                        f"actual={checkpoint.get('boundary')!r}"
                    )
                for robot_index, robot_row in enumerate(checkpoint["robots"]):
                    robot_path = f"{checkpoint_path}.robots[{robot_index}]"
                    if not _shape(robot_row, AUDIT_ROBOT_KEYS):
                        audit_container_issues.append(
                            f"path={robot_path}: expected exact "
                            f"keys={sorted(AUDIT_ROBOT_KEYS)!r}; actual={robot_row!r}"
                        )
                        continue
                    robot = robot_row.get("robot")
                    position = robot_row.get("position")
                    heading = robot_row.get("heading")
                    active_velocity = robot_row.get("active_velocity")
                    active_generation = robot_row.get("active_generation")
                    age = robot_row.get("age")
                    unsafe = robot_row.get("unsafe")
                    if not _integer(robot):
                        audit_value_issues.append(
                            f"path={robot_path}.robot: expected integer; actual={robot!r}"
                        )
                    if not (
                        isinstance(position, list)
                        and len(position) == 2
                        and all(_number(value) for value in position)
                        and -EPS <= float(position[0]) <= 12.0 + EPS
                        and -EPS <= float(position[1]) <= 8.0 + EPS
                    ):
                        audit_value_issues.append(
                            f"path={robot_path}.position: expected two finite workspace "
                            f"coordinates with tolerance={EPS}; actual={position!r}"
                        )
                    if not (_number(heading) and -math.pi - EPS <= float(heading) < math.pi + EPS):
                        audit_value_issues.append(
                            f"path={robot_path}.heading: expected finite angle in [-pi,pi), "
                            f"tolerance={EPS}; actual={heading!r}"
                        )
                    if not _valid_velocity(active_velocity):
                        audit_value_issues.append(
                            f"path={robot_path}.active_velocity: expected [v,omega] within "
                            f"command bounds, tolerance={EPS}; actual={active_velocity!r}"
                        )
                    if not (_integer(active_generation) and -1 <= active_generation < HORIZON):
                        audit_value_issues.append(
                            f"path={robot_path}.active_generation: expected integer in "
                            f"[-1,{HORIZON - 1}]; actual={active_generation!r}"
                        )
                    if not (_integer(age) and 0 <= age <= 8):
                        audit_value_issues.append(
                            f"path={robot_path}.age: expected integer in [0,8]; actual={age!r}"
                        )
                    if not isinstance(unsafe, bool):
                        audit_value_issues.append(
                            f"path={robot_path}.unsafe: expected Boolean; actual={unsafe!r}"
                        )
        if not _shape(packet_summary, PACKET_SUMMARY_KEYS):
            audit_container_issues.append(
                f"path={audit_path}.packet_summary: expected exact "
                f"keys={sorted(PACKET_SUMMARY_KEYS)!r}; actual={packet_summary!r}"
            )
        else:
            limits = {
                "attempts": CHANNEL_CAPACITY * HORIZON,
                "admitted": len(ROBOTS) * HORIZON,
                "effective_updates": CHANNEL_CAPACITY * HORIZON,
            }
            for name, limit in limits.items():
                value = packet_summary.get(name)
                if not (_integer(value) and 0 <= value <= limit):
                    audit_value_issues.append(
                        f"path={audit_path}.packet_summary.{name}: expected integer in "
                        f"[0,{limit}]; actual={value!r}"
                    )
        if not isinstance(goal_arrivals, list) or len(goal_arrivals) != len(ROBOTS):
            audit_container_issues.append(
                f"path={audit_path}.goal_arrivals: expected list "
                f"length={len(ROBOTS)}; actual={goal_arrivals!r}"
            )
        else:
            for robot, value in enumerate(goal_arrivals):
                if value is not None and not (_integer(value) and 1 <= value <= HORIZON):
                    audit_value_issues.append(
                        f"path={audit_path}.goal_arrivals[{robot}]: expected null or integer "
                        f"in [1,{HORIZON}]; actual={value!r}"
                    )
    return [
        _atom(
            "result_evidence",
            "slot_containers",
            not slot_issues,
            "slot containers valid" if not slot_issues else " | ".join(slot_issues[:6]),
        ),
        _atom(
            "result_evidence",
            "action_containers",
            not action_shape_issues,
            "action keys valid" if not action_shape_issues else " | ".join(action_shape_issues[:6]),
        ),
        _atom(
            "result_evidence",
            "action_values",
            not value_issues,
            "action values valid" if not value_issues else " | ".join(value_issues[:6]),
        ),
        _atom(
            "result_evidence",
            "audit_containers",
            not audit_container_issues,
            "audit containers valid"
            if not audit_container_issues
            else " | ".join(audit_container_issues[:6]),
        ),
        _atom(
            "result_evidence",
            "audit_values",
            not audit_value_issues,
            "audit values valid" if not audit_value_issues else " | ".join(audit_value_issues[:6]),
        ),
    ]


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _scene_rows(data)
    scene_counts: Counter[str] = Counter(
        row.get("scenario")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str)
    )
    for scene in SCENES:
        if scene_counts[scene] == 0:
            defects.append(f"missing scenario {scene}")
        if scene_counts[scene] > 1:
            defects.extend([f"duplicate scenario {scene}"] * (scene_counts[scene] - 1))
    for scene, count in scene_counts.items():
        if scene not in SCENES:
            defects.extend([f"unknown scenario {scene!r}"] * count)
    counts: Counter[tuple[str, int, int]] = Counter()
    for si, row in enumerate(rows):
        scene = row.get("scenario") if isinstance(row, Mapping) else None
        slots = row.get("slots") if isinstance(row, Mapping) else None
        if not isinstance(slots, list):
            continue
        for ti, slot in enumerate(slots):
            if not isinstance(slot, Mapping):
                defects.append(f"invalid slot at scenarios[{si}].slots[{ti}]")
                continue
            t = slot.get("t")
            valid_slot = (
                isinstance(scene, str) and scene in SCENES and _integer(t) and t in range(HORIZON)
            )
            if not valid_slot:
                defects.append(f"unknown slot key scene={scene!r}, t={t!r}")
            acts = slot.get("actions")
            if not isinstance(acts, list):
                continue
            for ai, action in enumerate(acts):
                robot = action.get("robot") if isinstance(action, Mapping) else None
                if valid_slot and _integer(robot) and robot in ROBOTS:
                    counts[(scene, t, robot)] += 1
                else:
                    defects.append(
                        f"unknown action key at scenarios[{si}].slots[{ti}].actions[{ai}]"
                    )
    for key in ((s, t, r) for s in SCENES for t in range(HORIZON) for r in ROBOTS):
        if counts[key] == 0:
            defects.append(f"missing action key {key}")
        elif counts[key] > 1:
            defects.extend([f"duplicate action key {key}"] * (counts[key] - 1))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    out: dict[Hashable, Any] = {}
    for row in _scene_rows(data):
        scene = row.get("scenario") if isinstance(row, Mapping) else None
        slots = row.get("slots") if isinstance(row, Mapping) else None
        if not isinstance(scene, str) or scene not in SCENES or not isinstance(slots, list):
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
                    out.setdefault((scene, slot["t"], action["robot"]), action)
    return out


def _index_audits(data: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in _scene_rows(data):
        scene = row.get("scenario") if isinstance(row, Mapping) else None
        if isinstance(scene, str) and scene in SCENES:
            out.setdefault(scene, row.get("audit"))
    return out


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _inside(point: Sequence[float], rect: Sequence[float]) -> bool:
    return rect[0] - EPS <= point[0] <= rect[2] + EPS and rect[1] - EPS <= point[1] <= rect[3] + EPS


def _level(scene: str, t: int) -> str:
    if scene == "open_crossing":
        return "good" if t % 12 < 8 else "degraded"
    if scene == "narrow_merge":
        if t < 14 or t >= 42:
            return "good"
        return "outage" if 22 <= t <= 29 else "degraded"
    if scene == "delay_corridor":
        return "degraded" if t < 18 or 28 <= t <= 39 else ("outage" if 18 <= t <= 27 else "good")
    if scene == "shared_bottleneck":
        if 30 <= t <= 35:
            return "outage"
        return "good" if t % 10 < 4 else "degraded"
    if t < 16 or t >= 36:
        return "good"
    return "outage" if 16 <= t <= 22 else "degraded"


def _u(scene: str, t: int, robot: int, seq: int, attempt: int) -> float:
    s = SCENES.index(scene)
    return (
        ((37 * (s + 1) + 19 * (t + 1) + 11 * (robot + 1) + 7 * (seq + 1) + 5 * (attempt + 1)) % 997)
        + 0.5
    ) / 997.0


def _reference(scene: str) -> list[list[tuple[float, float, float]]]:
    state = [list(start) for start in STARTS[scene]]
    result: list[list[tuple[float, float, float]]] = [[] for _ in ROBOTS]
    for _ in range(HORIZON):
        for robot in ROBOTS:
            x, y, theta = state[robot]
            gx, gy = GOALS[scene][robot]
            desired = math.atan2(gy - y, gx - x)
            omega = max(-MAX_OMEGA, min(MAX_OMEGA, 2.0 * _wrap(desired - theta)))
            v = max(0.0, min(MAX_V, 0.45 * math.hypot(gx - x, gy - y)))
            state[robot] = [
                x + DT * v * math.cos(theta),
                y + DT * v * math.sin(theta),
                _wrap(theta + DT * omega),
            ]
            result[robot].append(tuple(state[robot]))
    return result


def _replay(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    state = [list(start) for start in STARTS[scene]]
    active = [[0.0, 0.0] for _ in ROBOTS]
    active_gen = [-1] * len(ROBOTS)
    ages = [8] * len(ROBOTS)
    generated = [[0.0, 0.0] for _ in ROBOTS]
    queues: list[list[dict[str, Any]]] = [[] for _ in ROBOTS]
    next_seq = [0] * len(ROBOTS)
    packets: dict[tuple[int, int], dict[str, Any]] = {}
    arrivals: dict[int, list[dict[str, Any]]] = {}
    issues = {name: [] for name in ("command", "queue", "mode", "motion", "safety")}
    traces: list[dict[str, Any]] = []
    age_history: list[list[int]] = []
    goal_arrivals: list[int | None] = [None] * len(ROBOTS)
    attempts = 0
    effective = 0
    admitted = 0
    delivered = 0

    def process_boundary(boundary: int, initial: bool = False) -> None:
        nonlocal effective, delivered
        if not initial:
            candidates = arrivals.pop(boundary, [])
            for robot in ROBOTS:
                robot_packets = sorted(
                    (p for p in candidates if p["robot"] == robot),
                    key=lambda p: (p["generation"], p["seq"]),
                )
                for packet in robot_packets:
                    delivered += 1
                    packet["status"] = "delivered"
                    packet["arrival"] = boundary
                eligible = [p for p in robot_packets if p["generation"] > active_gen[robot]]
                if eligible:
                    packet = eligible[-1]
                    active[robot] = [packet["v"], packet["omega"]]
                    active_gen[robot] = packet["generation"]
                    ages[robot] = 0
                    packet["effective"] = True
                    effective += 1
                else:
                    ages[robot] = min(8, ages[robot] + 1)
        age_history.append(list(ages))

    process_boundary(0, initial=True)
    for t in range(HORIZON):
        actions = [records.get((scene, t, robot), {}) for robot in ROBOTS]
        for robot, action in enumerate(actions):
            velocity = action.get("velocity") if isinstance(action, Mapping) else None
            if _valid_velocity(velocity):
                v, omega = float(velocity[0]), float(velocity[1])
                delta_v = abs(v - generated[robot][0])
                delta_omega = abs(omega - generated[robot][1])
                if delta_v > 0.2 + EPS or delta_omega > 0.6 + EPS:
                    issues["command"].append(
                        f"path=scenarios[{scene}].slots[{t}].actions[robot={robot}].velocity: "
                        f"expected |delta_v|<={0.2}+{EPS} and |delta_omega|<={0.6}+{EPS}; "
                        f"actual_delta={[delta_v, delta_omega]!r}"
                    )
                generated[robot] = [v, omega]
            else:
                v, omega = 0.0, 0.0
        proposed = []
        for robot in ROBOTS:
            x, y, theta = state[robot]
            v, omega = active[robot]
            proposed.append(
                [
                    x + DT * v * math.cos(theta),
                    y + DT * v * math.sin(theta),
                    _wrap(theta + DT * omega),
                ]
            )
        motion_unsafe = [False] * len(ROBOTS)
        separation_unsafe = [False] * len(ROBOTS)
        for robot in ROBOTS:
            x, y, _ = proposed[robot]
            motion_unsafe[robot] = (
                x < -EPS
                or x > 12 + EPS
                or y < -EPS
                or y > 8 + EPS
                or any(_inside((x, y), rect) for rect in OBSTACLES[scene])
            )
            for other in ROBOTS:
                if (
                    other != robot
                    and _dist(proposed[robot], proposed[other])
                    < BASE_SEPARATION + 0.05 * max(ages[robot], ages[other]) - EPS
                ):
                    separation_unsafe[robot] = True
            if motion_unsafe[robot]:
                issues["motion"].append(
                    f"path=replay.{scene}.slots[{t}].proposed_state[robot={robot}]: "
                    "expected center in workspace and outside closed obstacles "
                    f"with tolerance={EPS}; "
                    f"actual={proposed[robot]!r}"
                )
            if separation_unsafe[robot]:
                _, actual_distance, violating_other, required = min(
                    (
                        (
                            _dist(proposed[robot], proposed[other])
                            - (BASE_SEPARATION + 0.05 * max(ages[robot], ages[other])),
                            _dist(proposed[robot], proposed[other]),
                            other,
                            BASE_SEPARATION + 0.05 * max(ages[robot], ages[other]),
                        )
                        for other in ROBOTS
                        if other != robot
                    ),
                    default=(math.inf, math.inf, -1, math.inf),
                )
                issues["safety"].append(
                    f"path=replay.{scene}.slots[{t}].proposed_state[robot={robot}]: "
                    f"expected pairwise center distance>={required} with tolerance={EPS}; "
                    f"actual_distance={actual_distance}, other_robot={violating_other}"
                )
        unsafe = [motion_unsafe[robot] or separation_unsafe[robot] for robot in ROBOTS]
        for robot in ROBOTS:
            if unsafe[robot]:
                state[robot] = list(state[robot])
            else:
                state[robot] = proposed[robot]
        for robot in ROBOTS:
            if (
                goal_arrivals[robot] is None
                and _dist(state[robot], GOALS[scene][robot]) <= GOAL_RADIUS + EPS
            ):
                goal_arrivals[robot] = t + 1
        for robot, action in enumerate(actions):
            if not isinstance(action, Mapping):
                continue
            send = action.get("send")
            mode = action.get("mode")
            if not isinstance(send, bool):
                continue
            mode_field_valid = mode is None or (isinstance(mode, str) and mode in MODES)
            if mode_field_valid:
                mode_ok = (not send and mode is None) or (
                    send and isinstance(mode, str) and mode in MODES
                )
                if not mode_ok:
                    issues["mode"].append(
                        f"path=scenarios[{scene}].slots[{t}].actions[robot={robot}]: "
                        "expected send=false with mode=null or send=true with mode in "
                        f"{sorted(MODES)!r}; "
                        f"actual_send={send!r}, actual_mode={mode!r}"
                    )
            if not send:
                continue
            if not isinstance(mode, str) or mode not in MODES:
                continue
            if len(queues[robot]) >= QUEUE_CAPACITY:
                issues["queue"].append(
                    f"path=scenarios[{scene}].slots[{t}].actions[robot={robot}].send: "
                    f"expected pre-admission queue_length<{QUEUE_CAPACITY}; "
                    f"actual_queue_length={len(queues[robot])}, tolerance=0"
                )
                continue
            seq = next_seq[robot]
            next_seq[robot] += 1
            packet = {
                "robot": robot,
                "seq": seq,
                "generation": t,
                "v": generated[robot][0],
                "omega": generated[robot][1],
                "mode": mode,
                "attempts": 0,
                "status": "queued",
                "arrival": None,
                "effective": False,
            }
            queues[robot].append(packet)
            packets[(robot, seq)] = packet
            admitted += 1
        candidates = sorted(
            (packet for queue in queues for packet in queue),
            key=lambda p: (p["generation"], p["robot"], p["seq"]),
        )
        chosen = candidates[:CHANNEL_CAPACITY]
        attempts += len(chosen)
        for packet in chosen:
            queues[packet["robot"]].remove(packet)
            packet["attempts"] += 1
            level = _level(scene, t)
            success = level == "good" or (
                level == "degraded"
                and _u(scene, t, packet["robot"], packet["seq"], packet["attempts"]) <= 0.65
            )
            if success:
                delay = 1 if level == "good" else 2
                arrival = t + delay
                if arrival <= HORIZON:
                    packet["status"] = "scheduled"
                    packet["arrival_boundary"] = arrival
                    arrivals.setdefault(arrival, []).append(packet)
                else:
                    packet["status"] = "late"
            elif packet["mode"] == "reliable" and packet["attempts"] < 3:
                queues[packet["robot"]].append(packet)
            else:
                packet["status"] = "lost"
        process_boundary(t + 1)
        traces.append(
            {
                "boundary": t + 1,
                "positions": [list(p) for p in state],
                "active": [list(c) for c in active],
                "active_generation": list(active_gen),
                "ages": list(ages),
                "unsafe": list(unsafe),
            }
        )
    reference = _reference(scene)
    error_auc = []
    for robot in ROBOTS:
        auc = sum(
            (traces[b]["positions"][robot][0] - reference[robot][b][0]) ** 2
            + (traces[b]["positions"][robot][1] - reference[robot][b][1]) ** 2
            for b in range(HORIZON)
        )
        error_auc.append(max(0.0, min(1.0, 1.0 - auc / (HORIZON * QOC_ERROR_SCALE))))
    qoc = sum(error_auc) / len(error_auc)
    timely = sum(
        1
        for robot, arrival in enumerate(goal_arrivals)
        if arrival is not None
        and arrival <= DEADLINES[scene][robot]
        and not any(tr["unsafe"][robot] for tr in traces[:arrival])
    ) / len(ROBOTS)
    effective_updates = sum(1 for p in packets.values() if p.get("effective"))
    aux1 = 0.0 if effective_updates == 0 or attempts == 0 else effective_updates / attempts
    mean_freshness = sum(
        max(0.0, min(1.0, 1.0 - age / 8.0))
        for ages_at_boundary in age_history[:-1]
        for age in ages_at_boundary
    ) / (HORIZON * len(ROBOTS))
    aux2 = 0.6 * mean_freshness + 0.4 * (effective_updates / admitted if admitted else 0.0)
    return {
        "scene": scene,
        "traces": traces,
        "packets": packets,
        "goal_arrivals": goal_arrivals,
        "qoc": qoc,
        "timely": timely,
        "aux1": aux1,
        "aux2": aux2,
        "attempts": attempts,
        "admitted": admitted,
        "effective": effective,
        "delivered": delivered,
        "issues": issues,
    }


def _legality_atoms(
    replayed: Mapping[str, Any], prerequisite: str | None = None
) -> list[AtomVerdict]:
    if prerequisite is not None:
        return [
            _atom(
                "decision_legality",
                name,
                False,
                f"not assessed because {prerequisite}",
            )
            for name in ("command", "queue", "mode", "motion", "safety")
        ]
    combined = {name: [] for name in ("command", "queue", "mode", "motion", "safety")}
    for scene in SCENES:
        rep = replayed.get(scene)
        if not isinstance(rep, Mapping):
            for name in combined:
                combined[name].append(f"{scene}: missing replay")
            continue
        for name in combined:
            combined[name].extend(rep.get("issues", {}).get(name, []))
    return [
        _atom(
            "decision_legality",
            name,
            not combined[name],
            "no violations" if not combined[name] else f"violations={combined[name][:6]}",
        )
        for name in combined
    ]


def _same_value(left: Any, right: Any) -> bool:
    """Compare fixed replay evidence without relying on object identity."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=EPS)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_value(left[key], right[key]) for key in left)
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def _audit_from_replay(replayed: Mapping[str, Any]) -> dict[str, Any]:
    """Build the sparse public audit projection; formal metrics remain grader-only."""
    by_boundary = {
        trace["boundary"]: trace
        for trace in replayed.get("traces", [])
        if isinstance(trace, Mapping) and _integer(trace.get("boundary"))
    }
    checkpoints = []
    for boundary in AUDIT_BOUNDARIES:
        trace = by_boundary[boundary]
        robots = []
        for robot in ROBOTS:
            position = trace["positions"][robot]
            robots.append(
                {
                    "robot": robot,
                    "position": [position[0], position[1]],
                    "heading": position[2],
                    "active_velocity": list(trace["active"][robot]),
                    "active_generation": trace["active_generation"][robot],
                    "age": trace["ages"][robot],
                    "unsafe": trace["unsafe"][robot],
                }
            )
        checkpoints.append({"boundary": boundary, "robots": robots})
    return {
        "checkpoints": checkpoints,
        "packet_summary": {
            "attempts": replayed.get("attempts"),
            "admitted": replayed.get("admitted"),
            "effective_updates": replayed.get("effective"),
        },
        "goal_arrivals": list(replayed.get("goal_arrivals", [])),
    }


def _audit_timeline(audit: Mapping[str, Any]) -> list[tuple[Any, list[Any]]]:
    timeline = []
    checkpoints = audit.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return timeline
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            timeline.append((None, []))
            continue
        robots = checkpoint.get("robots", [])
        robot_ids = (
            [row.get("robot") for row in robots if isinstance(row, Mapping)]
            if isinstance(robots, list)
            else []
        )
        timeline.append((checkpoint.get("boundary"), sorted(robot_ids, key=repr)))
    return sorted(timeline, key=repr)


def _audit_checkpoint_index(audit: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    checkpoints = audit.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return index
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping) or not _integer(checkpoint.get("boundary")):
            continue
        robots = checkpoint.get("robots", [])
        if not isinstance(robots, list):
            continue
        for row in robots:
            if isinstance(row, Mapping) and _integer(row.get("robot")):
                index.setdefault((checkpoint["boundary"], row["robot"]), row)
    return index


def _append_mismatch(
    failures: dict[str, list[str]],
    group: str,
    path: str,
    actual: Any,
    expected: Any,
    tolerance: float | int,
) -> None:
    if not _same_value(actual, expected):
        failures[group].append(
            f"path={path}: expected={expected!r}, actual={actual!r}, tolerance={tolerance}"
        )


def _replay_atoms(
    audits: Mapping[str, Any],
    replayed: Mapping[str, Any],
    prerequisite: str | None = None,
) -> list[AtomVerdict]:
    names = ("timeline", "motion", "channel", "commands", "goals", "safety")
    if prerequisite is not None:
        return [
            _atom(
                "replay_validity",
                name,
                False,
                f"not assessed because {prerequisite}",
            )
            for name in names
        ]
    failures = {name: [] for name in names}
    for scene in SCENES:
        rep = replayed.get(scene)
        submitted = audits.get(scene)
        if not isinstance(rep, Mapping) or not isinstance(submitted, Mapping):
            reason = f"scene={scene}: submitted audit or independent replay missing"
            for name in names:
                failures[name].append(reason)
            continue
        expected = _audit_from_replay(rep)
        submitted_timeline = _audit_timeline(submitted)
        expected_timeline = _audit_timeline(expected)
        timeline_ok = submitted_timeline == expected_timeline
        if not timeline_ok:
            failures["timeline"].append(
                f"path=scenarios[{scene}].audit.checkpoints: expected_ids={expected_timeline!r}, "
                f"actual_ids={submitted_timeline!r}, tolerance=0"
            )
        submitted_index = _audit_checkpoint_index(submitted)
        expected_index = _audit_checkpoint_index(expected)
        if timeline_ok:
            for boundary in AUDIT_BOUNDARIES:
                for robot in ROBOTS:
                    actual_row = submitted_index[(boundary, robot)]
                    expected_row = expected_index[(boundary, robot)]
                    base = (
                        f"scenarios[{scene}].audit.checkpoints[boundary={boundary}]"
                        f".robots[robot={robot}]"
                    )
                    for component in range(2):
                        _append_mismatch(
                            failures,
                            "motion",
                            f"{base}.position[{component}]",
                            actual_row["position"][component],
                            expected_row["position"][component],
                            EPS,
                        )
                    _append_mismatch(
                        failures,
                        "motion",
                        f"{base}.heading",
                        actual_row["heading"],
                        expected_row["heading"],
                        EPS,
                    )
                    for component in range(2):
                        _append_mismatch(
                            failures,
                            "commands",
                            f"{base}.active_velocity[{component}]",
                            actual_row["active_velocity"][component],
                            expected_row["active_velocity"][component],
                            EPS,
                        )
                    for field in ("active_generation", "age"):
                        _append_mismatch(
                            failures,
                            "commands",
                            f"{base}.{field}",
                            actual_row[field],
                            expected_row[field],
                            0,
                        )
                    _append_mismatch(
                        failures,
                        "safety",
                        f"{base}.unsafe",
                        actual_row["unsafe"],
                        expected_row["unsafe"],
                        0,
                    )
        else:
            for group in ("motion", "commands", "safety"):
                failures[group].append(
                    f"path=scenarios[{scene}].audit.checkpoints: values not assessable "
                    "because checkpoint IDs differ"
                )
        actual_summary = submitted.get("packet_summary", {})
        expected_summary = expected["packet_summary"]
        for field in ("attempts", "admitted", "effective_updates"):
            _append_mismatch(
                failures,
                "channel",
                f"scenarios[{scene}].audit.packet_summary.{field}",
                actual_summary.get(field) if isinstance(actual_summary, Mapping) else None,
                expected_summary[field],
                0,
            )
        actual_goals = submitted.get("goal_arrivals")
        for robot, expected_arrival in enumerate(expected["goal_arrivals"]):
            actual_arrival = (
                actual_goals[robot]
                if isinstance(actual_goals, list) and robot < len(actual_goals)
                else None
            )
            _append_mismatch(
                failures,
                "goals",
                f"scenarios[{scene}].audit.goal_arrivals[{robot}]",
                actual_arrival,
                expected_arrival,
                0,
            )
    return [
        _atom(
            "replay_validity",
            name,
            not failures[name],
            "submitted audit matches independent replay"
            if not failures[name]
            else " | ".join(failures[name][:6]),
        )
        for name in names
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    qoc = sum(float(replayed[s]["qoc"]) for s in SCENES) / len(SCENES)
    timely = sum(float(replayed[s]["timely"]) for s in SCENES) / len(SCENES)
    aux1 = sum(float(replayed[s]["aux1"]) for s in SCENES) / len(SCENES)
    aux2 = sum(float(replayed[s]["aux2"]) for s in SCENES) / len(SCENES)
    return {
        "primary_metrics_1": qoc,
        "primary_metrics_2": timely,
        "auxiliary_metrics_1": aux1,
        "auxiliary_metrics_2": aux2,
    }


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
    audits = _index_audits(data)
    base_checks = ("result_json", "result_evidence", "canonical_coverage")
    failed_prerequisites = [name for name in base_checks if scores.get(name, 0.0) < 1.0 - EPS]
    prerequisite = (
        None
        if not failed_prerequisites
        else f"prerequisite checks failed: {failed_prerequisites!r}"
    )
    replayed = {scene: _replay(scene, records) for scene in SCENES} if prerequisite is None else {}
    atoms = _legality_atoms(replayed, prerequisite)
    scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"])
    diagnostics["decision_legality"] = _diagnose_atoms(atoms)
    atoms = _replay_atoms(audits, replayed, prerequisite)
    scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"])
    diagnostics["replay_validity"] = _diagnose_atoms(atoms)
    gate = all(scores.get(name, 0.0) >= 1.0 - EPS for name in VALIDITY_CHECKS)
    metrics = _metrics(replayed) if gate else None
    if metrics is not None:
        scores.update(metrics)
    else:
        scores.update({name: 0.0 for name in METRIC_NAMES})
    failed_gate_checks = [name for name in VALIDITY_CHECKS if scores.get(name, 0.0) < 1.0 - EPS]
    for name in METRIC_NAMES:
        diagnostics[name] = (
            "validity gate passed"
            if metrics is not None
            else f"validity gate failed on {failed_gate_checks!r}; {name}=0"
        )
    return {
        "scores": scores,
        "diagnostics": diagnostics,
        "records": records,
        "audits": audits,
        "replayed": replayed,
        "metrics": metrics,
        "validity_gate": gate,
    }


def _regular(ws: Workspace, name: str) -> bool:
    path = Path(ws.root) / name
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if fence is not None:
            if stripped.startswith(fence) and len(stripped) >= 3:
                fence = None
            continue
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_match:
            fence = fence_match.group(1)[0]
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        if not match:
            continue
        title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
        if title:
            headings.append((index, len(match.group(1)), title))

    def substantive(body_lines: list[str]) -> str:
        kept: list[str] = []
        body_fence: str | None = None
        for body_line in body_lines:
            stripped = body_line.lstrip()
            if body_fence is not None:
                if stripped.startswith(body_fence) and len(stripped) >= 3:
                    body_fence = None
                else:
                    kept.append(body_line)
                continue
            fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", body_line)
            if fence_match:
                body_fence = fence_match.group(1)[0]
                continue
            if re.match(r"^ {0,3}#{1,6}[ \t]+", body_line):
                continue
            kept.append(body_line)
        return "\n".join(kept).strip()

    result: dict[str, str] = {}
    for position, (start, level, title) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[position + 1 :]:
            if next_level <= level:
                end = next_start
                break
        result[title] = substantive(lines[start + 1 : end])
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
    if not _regular(ws, "report.md"):
        return 0.0, "report.md missing"
    try:
        text = ws.read_text("report.md") or ""
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    sections = _headings(text)
    missing = sorted(name for name in required if not sections.get(name))
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
        except TimeoutError:
            state.timed_out = True
            state.elapsed_s = time.monotonic() - started
            state.official_message = "timed out"
            for name in VALIDITY_CHECKS:
                state.scores[name] = 0.0
                state.diagnostics[name] = "official run timed out; check not evaluated"
            for name in METRIC_NAMES:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"official run timed out; {name}=0"
            return state
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.exit_code = run.exit_code
        stderr_text = str(getattr(run, "stderr", "")).lower()
        state.timed_out = (
            state.elapsed_s >= 600.0 - EPS
            or state.exit_code is None
            or "timeout" in stderr_text
            or "timed out" in stderr_text
        )
        path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = path.is_file() and not path.is_symlink()
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
            for name in VALIDITY_CHECKS + METRIC_NAMES:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"official run failed; {state.official_message}"
            return state
        try:
            state.data = _strict_json(path.read_text(encoding="utf-8"))
            state.parse_ok = True
        except Exception as exc:
            state.data = None
            state.parse_ok = False
            state.diagnostics["result_json"] = f"parse failed: {type(exc).__name__}: {exc}"
        evaluated = _evaluate_data(state.data, state.parse_ok)
        state.scores.update(evaluated["scores"])
        state.records = evaluated["records"]
        state.replayed = evaluated["replayed"]
        state.metrics = evaluated["metrics"]
        state.validity_gate = evaluated["validity_gate"]
        for name, diagnostic in evaluated["diagnostics"].items():
            if name != "result_json" or state.parse_ok:
                state.diagnostics[name] = diagnostic
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
        message = f"grader exception {type(exc).__name__}: {exc}"
        state.official_message = message
        for name in VALIDITY_CHECKS + METRIC_NAMES:
            state.scores[name] = 0.0
            state.diagnostics[name] = message
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
        return 0.0, "run.py missing or not regular"
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
        return (
            (1.0, "static compilation passed")
            if result.exit_code == 0
            else (0.0, f"static compilation failed; exit={result.exit_code}")
        )
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
    *METRIC_NAMES,
):
    globals()[_name] = check(id=_name)(lambda ws, name=_name: _score(ws, name))
