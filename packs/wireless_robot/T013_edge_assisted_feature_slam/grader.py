from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

SCENES = (
    "hall_loop",
    "split_route",
    "delayed_merge",
    "sparse_landmarks",
    "drift_corridor",
)
ROBOTS = tuple(range(3))
HORIZON = 40
EVALUATION_FRAMES = (9, 19, 29, 39)
EDGE = (5.5, 5.5)
BYTE_LIMIT = 2400
FRAME_CAPACITY = 120
ROLLING_LIMIT = 3
STORE_LIMIT = 12
LOOP_RESIDUAL_LIMIT = 0.5
RMSE_NORM = 2.0
LOOP_QUALITY_NORM = 4.0
EPS = 1e-6

SCENE_DATA = {
    "hall_loop": {
        "x_phase": (0, 4, 8),
        "y_phase": (5, 9, 13),
        "radius": 3.2,
        "drift_scale": 0.85,
    },
    "split_route": {
        "x_phase": (0, 10, 5),
        "y_phase": (10, 0, 15),
        "radius": 3.0,
        "drift_scale": 1.0,
    },
    "delayed_merge": {
        "x_phase": (0, 7, 14),
        "y_phase": (14, 7, 0),
        "radius": 2.8,
        "drift_scale": 1.2,
    },
    "sparse_landmarks": {
        "x_phase": (0, 6, 12),
        "y_phase": (3, 11, 19),
        "radius": 2.35,
        "drift_scale": 1.1,
    },
    "drift_corridor": {
        "x_phase": (0, 2, 4),
        "y_phase": (10, 12, 14),
        "radius": 2.9,
        "drift_scale": 1.45,
    },
}
LANDMARKS = tuple(
    (f"f{index:02d}", (1.5 + 3.0 * (index % 4), 1.5 + 3.0 * (index // 4))) for index in range(16)
)
TRANSMIT_ACTIONS = ("skip", "summary", "keyframe", "retry")
DECISION_KEYS = {
    "t",
    "service_packet_id",
    "retain_serviced_keyframe",
    "evict_packet_id",
    "robots",
}
ROBOT_DECISION_KEYS = {"robot", "transmit", "retry_packet_id"}
ROBOT_TRACE_KEYS = {
    "t",
    "robot",
    "local_pose",
    "uncertainty",
    "visible_tokens",
    "applied_feedback_ids",
    "retryable_packet_ids",
}
CLOUD_TRACE_KEYS = {
    "t",
    "queue_packet_ids",
    "retained_keyframe_ids",
    "accepted_loop_pairs",
    "feedback_id",
}
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.decisions": 0.20,
        "result_evidence.robot_state": 0.20,
        "result_evidence.robot_feedback": 0.20,
        "result_evidence.cloud_queue": 0.20,
        "result_evidence.cloud_updates": 0.20,
    },
    "decision_legality": {
        "decision_legality.transmit_relations": 0.125,
        "decision_legality.byte_budget": 0.125,
        "decision_legality.frame_capacity": 0.125,
        "decision_legality.rolling_attempts": 0.125,
        "decision_legality.retry_eligibility": 0.125,
        "decision_legality.service_available": 0.125,
        "decision_legality.service_once": 0.125,
        "decision_legality.retention": 0.125,
    },
    "replay_validity": {
        "replay_validity.robot_state": 1 / 6,
        "replay_validity.robot_feedback": 1 / 6,
        "replay_validity.queue": 1 / 6,
        "replay_validity.retention": 1 / 6,
        "replay_validity.loops": 1 / 6,
        "replay_validity.feedback": 1 / 6,
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


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _vector(value: Any, length: int = 2) -> bool:
    return isinstance(value, list) and len(value) == length and all(_finite(item) for item in value)


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _id(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"{prefix}[0-9]+", value) is not None


def _id_list(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, list)
        and all(_id(item, prefix) for item in value)
        and len(value) == len(set(value))
    )


def _token_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(item, str) and re.fullmatch(r"f(?:0[0-9]|1[0-5])", item) for item in value
        )
        and value == sorted(set(value))
    )


def _loop_pairs(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(pair, list) and len(pair) == 2 and all(_id(item, "p") for item in pair)
            for pair in value
        )
        and len({tuple(pair) for pair in value}) == len(value)
    )


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(value)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key}")
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


def _triangular(value: int) -> float:
    phase = value % 20
    return float(10 - abs(phase - 10))


def _truth_pose(scene: str, robot: int, t: int) -> list[float]:
    data = SCENE_DATA[scene]
    return [
        0.5 + _triangular(t + data["x_phase"][robot]),
        0.5 + _triangular(t + data["y_phase"][robot]),
    ]


def _tokens(scene: str, robot: int, t: int) -> list[str]:
    pose = _truth_pose(scene, robot, t)
    radius = float(SCENE_DATA[scene]["radius"])
    return [
        token
        for token, landmark in LANDMARKS
        if math.hypot(pose[0] - landmark[0], pose[1] - landmark[1]) <= radius + 1e-12
    ]


def _odometry_delta(scene: str, robot: int, t: int) -> list[float]:
    previous = _truth_pose(scene, robot, t - 1)
    current = _truth_pose(scene, robot, t)
    scene_index = SCENES.index(scene)
    scale = float(SCENE_DATA[scene]["drift_scale"])
    error_x = scale * (0.032 * (robot - 1) + 0.006 * (((t + robot + scene_index) % 3) - 1))
    error_y = scale * (0.026 * (1 - robot) + 0.005 * (((2 * t + robot + scene_index) % 3) - 1))
    return [current[0] - previous[0] + error_x, current[1] - previous[1] + error_y]


def _packet_bytes(mode: str, token_count: int) -> int:
    if mode == "summary":
        return 8 + 6 * token_count
    if mode == "keyframe":
        return 32 + 14 * token_count
    raise ValueError(mode)


def _packet_number(packet_id: str) -> int:
    return int(packet_id[1:])


def _feedback_number(feedback_id: str) -> int:
    return int(feedback_id[1:])


def _link_result(
    scene: str, robot: int, t: int, packet_number: int, retry_index: int
) -> tuple[str, int | None]:
    truth = _truth_pose(scene, robot, t)
    distance = math.floor(abs(truth[0] - EDGE[0]) + abs(truth[1] - EDGE[1]) + 1e-12)
    value = (
        5 * t + 3 * robot + 7 * SCENES.index(scene) + packet_number + 2 * retry_index + distance
    ) % 13
    if distance > 8 or value in {0, 1}:
        return "dropped", None
    delay = 1 + ((t + robot + SCENES.index(scene) + packet_number + retry_index + distance) % 3)
    arrival = t + delay
    return ("late", arrival) if arrival >= HORIZON else ("delivered", arrival)


def _rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    return []


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = _shape(data, {"scenarios"})
    scenes = data.get("scenarios") if root_ok else None
    containers_ok = isinstance(scenes, list) and all(
        _shape(row, {"scenario", "decisions", "robot_trace", "cloud_trace"}) for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires scenario,decisions,robot_trace,cloud_trace",
        ),
    ]


def _decision_shape(value: Any) -> bool:
    return (
        _shape(value, DECISION_KEYS)
        and isinstance(value.get("robots"), list)
        and all(_shape(row, ROBOT_DECISION_KEYS) for row in value["robots"])
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = ("decisions", "robot_state", "robot_feedback", "cloud_queue", "cloud_updates")
    flags = {name: True for name in names}
    details = {name: "all submitted fields satisfy their local contract" for name in names}
    rows = _rows(data)
    if not rows:
        for name in names:
            flags[name] = False
            details[name] = "scenarios: no evidence records"
    for scene_index, scene_row in enumerate(rows):
        if not isinstance(scene_row, Mapping):
            for name in names:
                flags[name] = False
                details[name] = f"scenarios[{scene_index}]: expected object"
            continue
        decisions = scene_row.get("decisions")
        robot_trace = scene_row.get("robot_trace")
        cloud_trace = scene_row.get("cloud_trace")

        current = flags["decisions"]
        flags["decisions"] &= isinstance(decisions, list) and all(
            _decision_shape(item)
            and _integer(item.get("t"))
            and (item.get("service_packet_id") is None or _id(item.get("service_packet_id"), "p"))
            and isinstance(item.get("retain_serviced_keyframe"), bool)
            and (item.get("evict_packet_id") is None or _id(item.get("evict_packet_id"), "p"))
            and len(item["robots"]) == len(ROBOTS)
            and [robot.get("robot") for robot in item["robots"]] == list(ROBOTS)
            and all(
                _integer(robot.get("robot"))
                and isinstance(robot.get("transmit"), str)
                and (robot.get("retry_packet_id") is None or _id(robot.get("retry_packet_id"), "p"))
                for robot in item["robots"]
            )
            for item in decisions
        )
        if current and not flags["decisions"]:
            details["decisions"] = (
                f"scenarios[{scene_index}].decisions: invalid keys, types, or robot order"
            )

        current = flags["robot_state"]
        flags["robot_state"] &= isinstance(robot_trace, list) and all(
            isinstance(item, Mapping)
            and set(item).issubset(ROBOT_TRACE_KEYS)
            and all(field in item for field in ("t", "robot", "local_pose", "uncertainty"))
            and _integer(item.get("t"))
            and _integer(item.get("robot"))
            and _vector(item.get("local_pose"))
            and _finite(item.get("uncertainty"))
            and 0 <= float(item["uncertainty"])
            for item in robot_trace
        )
        if current and not flags["robot_state"]:
            details["robot_state"] = (
                f"scenarios[{scene_index}].robot_trace: invalid state field or unknown key"
            )

        current = flags["robot_feedback"]
        flags["robot_feedback"] &= isinstance(robot_trace, list) and all(
            isinstance(item, Mapping)
            and _token_list(item.get("visible_tokens"))
            and _id_list(item.get("applied_feedback_ids"), "f")
            and _id_list(item.get("retryable_packet_ids"), "p")
            for item in robot_trace
        )
        if current and not flags["robot_feedback"]:
            details["robot_feedback"] = (
                f"scenarios[{scene_index}].robot_trace: invalid token or feedback list"
            )

        current = flags["cloud_queue"]
        flags["cloud_queue"] &= isinstance(cloud_trace, list) and all(
            isinstance(item, Mapping)
            and set(item).issubset(CLOUD_TRACE_KEYS)
            and _integer(item.get("t"))
            and _id_list(item.get("queue_packet_ids"), "p")
            and _id_list(item.get("retained_keyframe_ids"), "p")
            for item in cloud_trace
        )
        if current and not flags["cloud_queue"]:
            details["cloud_queue"] = (
                f"scenarios[{scene_index}].cloud_trace: invalid queue/store field or unknown key"
            )

        current = flags["cloud_updates"]
        flags["cloud_updates"] &= isinstance(cloud_trace, list) and all(
            isinstance(item, Mapping)
            and _loop_pairs(item.get("accepted_loop_pairs"))
            and (item.get("feedback_id") is None or _id(item.get("feedback_id"), "f"))
            for item in cloud_trace
        )
        if current and not flags["cloud_updates"]:
            details["cloud_updates"] = (
                f"scenarios[{scene_index}].cloud_trace: invalid loop or feedback field"
            )
    return [_atom("result_evidence", name, flags[name], details[name]) for name in names]


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for row in _rows(data):
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            for field in ("decisions", "robot_trace", "cloud_trace"):
                records[(row["scenario"], field)] = row.get(field)
    return records


def _canonical_defects(
    actual: Sequence[Hashable], expected: Sequence[Hashable], label: str
) -> list[str]:
    defects: list[str] = []
    for key in expected:
        count = actual.count(key)
        if count == 0:
            defects.append(f"{label}: missing key={key!r}")
        elif count > 1:
            defects.extend(f"{label}: duplicate key={key!r}" for _ in range(count - 1))
    expected_set = set(expected)
    defects.extend(f"{label}: unknown key={key!r}" for key in actual if key not in expected_set)
    if (
        len(actual) == len(expected)
        and set(actual) == expected_set
        and list(actual) != list(expected)
    ):
        defects.append(f"{label}: canonical order")
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    for index, row in enumerate(rows):
        name = row.get("scenario") if isinstance(row, Mapping) else None
        scene_keys.append(name if isinstance(name, str) else ("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))
    for scene in SCENES:
        row = next(
            (item for item in rows if isinstance(item, Mapping) and item.get("scenario") == scene),
            None,
        )
        for field, expected in (
            ("decisions", list(range(HORIZON))),
            ("robot_trace", [(t, robot) for t in range(HORIZON) for robot in ROBOTS]),
            ("cloud_trace", list(range(HORIZON))),
        ):
            raw = row.get(field) if isinstance(row, Mapping) else None
            if not isinstance(raw, list):
                defects.append(f"{scene}.{field}: expected list")
                continue
            actual: list[Hashable] = []
            for index, item in enumerate(raw):
                if field == "robot_trace":
                    if (
                        isinstance(item, Mapping)
                        and _integer(item.get("t"))
                        and _integer(item.get("robot"))
                    ):
                        actual.append((item["t"], item["robot"]))
                    else:
                        actual.append(("invalid_robot_trace", index))
                elif isinstance(item, Mapping) and _integer(item.get("t")):
                    actual.append(item["t"])
                else:
                    actual.append((f"invalid_{field}", index))
            defects.extend(_canonical_defects(actual, expected, f"{scene}.{field}"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _decision_maps(records: Mapping[Hashable, Any], scene: str) -> dict[int, dict[str, Any]]:
    raw = records.get((scene, "decisions"), [])
    mapped: dict[int, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return mapped
    for row in raw:
        if isinstance(row, Mapping) and _integer(row.get("t")) and row["t"] not in mapped:
            mapped[row["t"]] = dict(row)
    return mapped


def _normalize_decision(value: Any, t: int) -> dict[str, Any]:
    row = value if isinstance(value, Mapping) else {}
    robot_rows: dict[int, Mapping[str, Any]] = {}
    raw_robots = row.get("robots")
    if isinstance(raw_robots, list):
        for robot_row in raw_robots:
            if (
                isinstance(robot_row, Mapping)
                and _integer(robot_row.get("robot"))
                and robot_row["robot"] not in robot_rows
            ):
                robot_rows[robot_row["robot"]] = robot_row
    return {
        "t": t,
        "service_packet_id": row.get("service_packet_id"),
        "retain_serviced_keyframe": row.get("retain_serviced_keyframe"),
        "evict_packet_id": row.get("evict_packet_id"),
        "robots": [
            {
                "robot": robot,
                "transmit": robot_rows.get(robot, {}).get("transmit"),
                "retry_packet_id": robot_rows.get(robot, {}).get("retry_packet_id"),
            }
            for robot in ROBOTS
        ],
    }


def _public_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["id"],
        "robot": packet["robot"],
        "generated_t": packet["generated_t"],
        "mode": packet["mode"],
        "bytes": packet["bytes"],
        "tokens": list(packet["tokens"]),
        "local_pose": list(packet["local_pose"]),
        "uncertainty": packet["uncertainty"],
    }


def _compare(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            errors.append(
                f"{path}: expected keys={sorted(expected)}, actual_type={type(actual).__name__}"
            )
            return
        for key, value in expected.items():
            _compare(actual[key], value, f"{path}.{key}", errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            actual_length = len(actual) if isinstance(actual, list) else "not-list"
            errors.append(
                f"{path}: expected list length={len(expected)}, actual_length={actual_length}"
            )
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare(actual_item, expected_item, f"{path}[{index}]", errors)
    elif _finite(expected):
        if not _finite(actual) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=EPS
        ):
            errors.append(f"{path}: expected={expected!r}, actual={actual!r}, tolerance={EPS}")
    elif actual != expected:
        errors.append(f"{path}: expected={expected!r}, actual={actual!r}")


def _simulate_scene(
    scene: str,
    decision_source: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    local_poses = [_truth_pose(scene, robot, 0) for robot in ROBOTS]
    uncertainties = [0.05 for _ in ROBOTS]
    packets: list[dict[str, Any]] = []
    packet_by_id: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    queue: list[str] = []
    retained: list[str] = []
    feedbacks: list[dict[str, Any]] = []
    bytes_sent = 0
    loop_counter = 0
    best_loop_quality: dict[tuple[int, int, str, str], float] = {}
    latest_applied_generation: list[int | None] = [None] * len(ROBOTS)
    freshness_sum = 0.0
    decisions: list[dict[str, Any]] = []
    expected_robot: dict[tuple[int, int], dict[str, Any]] = {}
    expected_cloud: dict[int, dict[str, Any]] = {}
    legality_names = (
        "transmit_relations",
        "byte_budget",
        "frame_capacity",
        "rolling_attempts",
        "retry_eligibility",
        "service_available",
        "service_once",
        "retention",
    )
    legality = {name: True for name in legality_names}
    legality_errors: dict[str, list[str]] = {name: [] for name in legality_names}

    for t in range(HORIZON):
        arrivals: list[str] = []
        for attempt in attempts:
            if attempt["status"] != "delivered" or attempt["arrival_t"] != t:
                continue
            packet = packet_by_id[attempt["packet_id"]]
            if packet["delivered_t"] is None:
                packet["delivered_t"] = t
                arrivals.append(packet["id"])
        queue.extend(arrivals)
        queue.sort(
            key=lambda packet_id: (
                packet_by_id[packet_id]["delivered_t"],
                _packet_number(packet_id),
            )
        )

        applied_by_robot: list[list[str]] = [[] for _ in ROBOTS]
        for feedback in sorted(feedbacks, key=lambda item: _feedback_number(item["id"])):
            if feedback["arrival_t"] != t:
                continue
            robot = feedback["robot"]
            local_poses[robot] = [
                local_poses[robot][coordinate] + feedback["delta"][coordinate]
                for coordinate in range(2)
            ]
            uncertainties[robot] *= 1.0 - feedback["gain"]
            latest_applied_generation[robot] = feedback["packet_generated_t"]
            applied_by_robot[robot].append(feedback["id"])

        if t > 0:
            for robot in ROBOTS:
                delta = _odometry_delta(scene, robot, t)
                local_poses[robot] = [
                    local_poses[robot][coordinate] + delta[coordinate] for coordinate in range(2)
                ]
                uncertainties[robot] += 0.07 * float(SCENE_DATA[scene]["drift_scale"])

        retryable_by_robot: list[list[str]] = [[] for _ in ROBOTS]
        for packet in packets:
            if (
                packet["last_status"] == "dropped"
                and packet["retry_count"] == 0
                and packet["last_attempt_t"] < t
                and packet["delivered_t"] is None
                and packet["service_t"] is None
            ):
                retryable_by_robot[packet["robot"]].append(packet["id"])
        for values in retryable_by_robot:
            values.sort(key=_packet_number)

        queue_before = list(queue)
        retained_before = list(retained)
        robot_view = [
            {
                "t": t,
                "robot": robot,
                "local_pose": list(local_poses[robot]),
                "uncertainty": uncertainties[robot],
                "visible_tokens": _tokens(scene, robot, t),
                "applied_feedback_ids": list(applied_by_robot[robot]),
                "retryable_packet_ids": list(retryable_by_robot[robot]),
            }
            for robot in ROBOTS
        ]
        view = {
            "t": t,
            "robots": robot_view,
            "queue": [_public_packet(packet_by_id[packet_id]) for packet_id in queue_before],
            "retained_keyframes": [
                _public_packet(packet_by_id[packet_id]) for packet_id in retained_before
            ],
            "remaining_bytes": max(0, BYTE_LIMIT - bytes_sent),
            "attempts_in_previous_three_frames": sum(
                max(0, t - 3) <= attempt["t"] < t for attempt in attempts
            ),
        }
        raw_decision = decision_source(view)
        decision = _normalize_decision(raw_decision, t)
        decisions.append(decision)

        raw_retain = decision["retain_serviced_keyframe"]
        raw_evict = decision["evict_packet_id"]
        service_id = decision["service_packet_id"]
        serviced: dict[str, Any] | None = None
        if service_id is not None:
            if (
                _id(service_id, "p")
                and service_id in packet_by_id
                and packet_by_id[service_id]["service_t"] is not None
            ):
                legality["service_once"] = False
                legality_errors["service_once"].append(
                    f"{scene}.decisions[{t}].service_packet_id={service_id!r} already serviced"
                )
            elif _id(service_id, "p") and service_id in queue:
                queue.remove(service_id)
                serviced = packet_by_id[service_id]
                serviced["service_t"] = t
            else:
                legality["service_available"] = False
                legality_errors["service_available"].append(
                    f"{scene}.decisions[{t}].service_packet_id={service_id!r} "
                    "not in service-before queue"
                )

        accepted_pairs: list[list[str]] = []
        accepted_details: list[tuple[dict[str, Any], float, tuple[str, str]]] = []
        if serviced is not None and serviced["mode"] == "keyframe":
            for previous_id in sorted(retained, key=_packet_number):
                previous = packet_by_id[previous_id]
                common = sorted(set(serviced["tokens"]) & set(previous["tokens"]))
                if previous["robot"] == serviced["robot"] or len(common) < 2:
                    continue
                current_truth = _truth_pose(scene, serviced["robot"], serviced["generated_t"])
                previous_truth = _truth_pose(scene, previous["robot"], previous["generated_t"])
                residual_vector = [
                    (serviced["local_pose"][coordinate] - previous["local_pose"][coordinate])
                    - (current_truth[coordinate] - previous_truth[coordinate])
                    for coordinate in range(2)
                ]
                residual = math.hypot(*residual_vector)
                if residual <= LOOP_RESIDUAL_LIMIT + 1e-12:
                    accepted_pairs.append([serviced["id"], previous_id])
                    accepted_details.append((previous, residual, (common[0], common[1])))
                    loop_counter += 1
                    signature = (
                        min(serviced["robot"], previous["robot"]),
                        max(serviced["robot"], previous["robot"]),
                        common[0],
                        common[1],
                    )
                    quality = max(0.0, 1.0 - residual / LOOP_RESIDUAL_LIMIT)
                    best_loop_quality[signature] = max(
                        best_loop_quality.get(signature, 0.0), quality
                    )

        feedback: dict[str, Any] | None = None
        if serviced is not None:
            if serviced["mode"] == "summary":
                gain = 0.20
                delay = 1
            else:
                gain = 0.75 if accepted_details else 0.40
                delay = 2
            truth = _truth_pose(scene, serviced["robot"], serviced["generated_t"])
            error = [
                serviced["local_pose"][coordinate] - truth[coordinate] for coordinate in range(2)
            ]
            feedback = {
                "id": f"f{len(feedbacks)}",
                "robot": serviced["robot"],
                "created_t": t,
                "arrival_t": t + delay,
                "packet_generated_t": serviced["generated_t"],
                "gain": gain,
                "delta": [-gain * value for value in error],
            }
            feedbacks.append(feedback)

        retention_ok = True
        if serviced is None or serviced["mode"] != "keyframe":
            retention_ok = raw_retain is False and raw_evict is None
        elif not isinstance(raw_retain, bool):
            retention_ok = False
        elif not raw_retain:
            retention_ok = raw_evict is None
        elif len(retained) < STORE_LIMIT:
            retention_ok = raw_evict is None
            if retention_ok:
                retained.append(serviced["id"])
        else:
            retention_ok = _id(raw_evict, "p") and raw_evict in retained
            if retention_ok:
                retained.remove(raw_evict)
                retained.append(serviced["id"])
        if not retention_ok:
            legality["retention"] = False
            legality_errors["retention"].append(
                f"{scene}.decisions[{t}]: retain/evict relation invalid for serviced packet"
            )

        attempts_this_frame = 0
        bytes_this_frame = 0
        for robot in ROBOTS:
            robot_decision = decision["robots"][robot]
            transmit = robot_decision["transmit"]
            retry_id = robot_decision["retry_packet_id"]
            relation_ok = (
                transmit in TRANSMIT_ACTIONS
                and ((transmit == "retry") == (retry_id is not None))
                and (retry_id is None or _id(retry_id, "p"))
            )
            if not relation_ok:
                legality["transmit_relations"] = False
                legality_errors["transmit_relations"].append(
                    f"{scene}.decisions[{t}].robots[{robot}]: transmit/retry relation invalid"
                )
                continue
            packet: dict[str, Any] | None = None
            retry_index = 0
            if transmit in {"summary", "keyframe"}:
                packet_id = f"p{len(packets)}"
                tokens = _tokens(scene, robot, t)
                packet = {
                    "id": packet_id,
                    "robot": robot,
                    "generated_t": t,
                    "mode": transmit,
                    "bytes": _packet_bytes(transmit, len(tokens)),
                    "tokens": tokens,
                    "local_pose": list(local_poses[robot]),
                    "uncertainty": uncertainties[robot],
                    "retry_count": 0,
                    "last_status": None,
                    "last_attempt_t": t,
                    "delivered_t": None,
                    "service_t": None,
                }
                packets.append(packet)
                packet_by_id[packet_id] = packet
            elif transmit == "retry":
                eligible = retry_id in retryable_by_robot[robot]
                if not eligible:
                    legality["retry_eligibility"] = False
                    legality_errors["retry_eligibility"].append(
                        f"{scene}.decisions[{t}].robots[{robot}]."
                        f"retry_packet_id={retry_id!r} not eligible"
                    )
                    continue
                packet = packet_by_id[retry_id]
                packet["retry_count"] = 1
                retry_index = 1
            if packet is None:
                continue
            attempts_this_frame += 1
            bytes_this_frame += packet["bytes"]
            bytes_sent += packet["bytes"]
            status, arrival_t = _link_result(
                scene, robot, t, _packet_number(packet["id"]), retry_index
            )
            packet["last_status"] = status
            packet["last_attempt_t"] = t
            attempts.append(
                {
                    "packet_id": packet["id"],
                    "robot": robot,
                    "t": t,
                    "retry_index": retry_index,
                    "status": status,
                    "arrival_t": arrival_t,
                }
            )

        if bytes_sent > BYTE_LIMIT:
            legality["byte_budget"] = False
            legality_errors["byte_budget"].append(
                f"{scene}: bytes_sent={bytes_sent} exceeds limit={BYTE_LIMIT} at t={t}"
            )
        if bytes_this_frame > FRAME_CAPACITY:
            legality["frame_capacity"] = False
            legality_errors["frame_capacity"].append(
                f"{scene}: frame={t} bytes={bytes_this_frame} exceeds capacity={FRAME_CAPACITY}"
            )
        rolling_count = sum(max(0, t - 3) <= attempt["t"] <= t for attempt in attempts)
        if rolling_count > ROLLING_LIMIT:
            legality["rolling_attempts"] = False
            legality_errors["rolling_attempts"].append(
                f"{scene}: frames {max(0, t - 3)}..{t} "
                f"attempts={rolling_count} exceeds {ROLLING_LIMIT}"
            )

        for robot in ROBOTS:
            expected_robot[(t, robot)] = robot_view[robot]
            latest_generation = latest_applied_generation[robot]
            if latest_generation is not None:
                freshness_sum += max(0.0, 1.0 - (t - latest_generation) / 6.0)
        expected_cloud[t] = {
            "t": t,
            "queue_packet_ids": queue_before,
            "retained_keyframe_ids": list(retained),
            "accepted_loop_pairs": accepted_pairs,
            "feedback_id": feedback["id"] if feedback is not None else None,
        }

    rmse_terms = []
    for t in EVALUATION_FRAMES:
        for robot in ROBOTS:
            actual = expected_robot[(t, robot)]["local_pose"]
            truth = _truth_pose(scene, robot, t)
            rmse_terms.append((actual[0] - truth[0]) ** 2 + (actual[1] - truth[1]) ** 2)
    rmse = math.sqrt(sum(rmse_terms) / len(rmse_terms))

    return {
        "scenario_record": {
            "scenario": scene,
            "decisions": decisions,
            "robot_trace": [expected_robot[(t, robot)] for t in range(HORIZON) for robot in ROBOTS],
            "cloud_trace": [expected_cloud[t] for t in range(HORIZON)],
        },
        "expected_robot": expected_robot,
        "expected_cloud": expected_cloud,
        "legality": legality,
        "legality_errors": legality_errors,
        "rmse": rmse,
        "loop_quality": min(1.0, sum(best_loop_quality.values()) / LOOP_QUALITY_NORM),
        "bytes_sent": bytes_sent,
        "freshness": freshness_sum / (HORIZON * len(ROBOTS)),
        "packets": packets,
        "attempts": attempts,
        "feedbacks": feedbacks,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    decisions = _decision_maps(records, scene)
    replayed = _simulate_scene(scene, lambda view: decisions.get(int(view["t"]), {}))
    submitted_robot = records.get((scene, "robot_trace"), [])
    submitted_cloud = records.get((scene, "cloud_trace"), [])
    robot_map = (
        {
            (row["t"], row["robot"]): row
            for row in submitted_robot
            if isinstance(row, Mapping) and _integer(row.get("t")) and _integer(row.get("robot"))
        }
        if isinstance(submitted_robot, list)
        else {}
    )
    cloud_map = (
        {
            row["t"]: row
            for row in submitted_cloud
            if isinstance(row, Mapping) and _integer(row.get("t"))
        }
        if isinstance(submitted_cloud, list)
        else {}
    )
    errors = {
        name: []
        for name in ("robot_state", "robot_feedback", "queue", "retention", "loops", "feedback")
    }
    for (t, robot), expected in replayed["expected_robot"].items():
        actual = robot_map.get((t, robot))
        if not isinstance(actual, Mapping):
            errors["robot_state"].append(f"{scene}.robot_trace[{t},{robot}]: missing")
            errors["robot_feedback"].append(f"{scene}.robot_trace[{t},{robot}]: missing")
            continue
        for field in ("t", "robot", "local_pose", "uncertainty"):
            _compare(
                actual.get(field),
                expected[field],
                f"{scene}.robot[{t},{robot}].{field}",
                errors["robot_state"],
            )
        for field in ("visible_tokens", "applied_feedback_ids", "retryable_packet_ids"):
            _compare(
                actual.get(field),
                expected[field],
                f"{scene}.robot[{t},{robot}].{field}",
                errors["robot_feedback"],
            )
    for t, expected in replayed["expected_cloud"].items():
        actual = cloud_map.get(t)
        if not isinstance(actual, Mapping):
            for name in ("queue", "retention", "loops", "feedback"):
                errors[name].append(f"{scene}.cloud_trace[{t}]: missing")
            continue
        _compare(
            actual.get("queue_packet_ids"),
            expected["queue_packet_ids"],
            f"{scene}.cloud[{t}].queue_packet_ids",
            errors["queue"],
        )
        _compare(
            actual.get("retained_keyframe_ids"),
            expected["retained_keyframe_ids"],
            f"{scene}.cloud[{t}].retained_keyframe_ids",
            errors["retention"],
        )
        _compare(
            actual.get("accepted_loop_pairs"),
            expected["accepted_loop_pairs"],
            f"{scene}.cloud[{t}].accepted_loop_pairs",
            errors["loops"],
        )
        for field in ("t", "feedback_id"):
            _compare(
                actual.get(field),
                expected[field],
                f"{scene}.cloud[{t}].{field}",
                errors["feedback"],
            )
    replayed["errors"] = errors
    return replayed


def _online_scene(scene: str, policy: Any) -> dict[str, Any]:
    if hasattr(policy, "reset"):
        policy.reset()
    return _simulate_scene(scene, policy.decide)["scenario_record"]


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = (
        "transmit_relations",
        "byte_budget",
        "frame_capacity",
        "rolling_attempts",
        "retry_eligibility",
        "service_available",
        "service_once",
        "retention",
    )
    flags = {name: True for name in names}
    details = {name: "legal in all scenes" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("legality", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("legality_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if not scene_flags.get(name, False):
                flags[name] = False
                failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                details[name] = failures[0] if failures else f"{scene}: {name} unavailable"
    return [_atom("decision_legality", name, flags[name], details[name]) for name in names]


def _replay_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = ("robot_state", "robot_feedback", "queue", "retention", "loops", "feedback")
    flags = {name: True for name in names}
    details = {name: "fixed-model replay matched" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_errors = result.get("errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            failures = scene_errors.get(name, [f"{scene}: replay unavailable"])
            if failures:
                flags[name] = False
                details[name] = failures[0]
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    primary_1 = sum(
        max(0.0, min(1.0, 1.0 - replayed[scene]["rmse"] / RMSE_NORM)) for scene in SCENES
    ) / len(SCENES)
    primary_2 = sum(replayed[scene]["loop_quality"] for scene in SCENES) / len(SCENES)
    auxiliary_1 = sum(
        max(0.0, min(1.0, 1.0 - replayed[scene]["bytes_sent"] / BYTE_LIMIT)) for scene in SCENES
    ) / len(SCENES)
    auxiliary_2 = sum(replayed[scene]["freshness"] for scene in SCENES) / len(SCENES)
    return {
        "primary_metrics_1": primary_1,
        "primary_metrics_2": primary_2,
        "auxiliary_metrics_1": auxiliary_1,
        "auxiliary_metrics_2": auxiliary_2,
    }


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    visible = [False] * len(lines)
    in_fence = False
    fence_marker = ""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            continue
        visible[index] = not in_fence
    headings: list[tuple[int, int, int, str]] = []
    for index, line in enumerate(lines):
        if not visible[index]:
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
        if match:
            headings.append((index, index + 1, len(match.group(1)), match.group(2).strip()))
        elif (
            line.strip()
            and index + 1 < len(lines)
            and visible[index + 1]
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[index + 1])
        ):
            level = 1 if "=" in lines[index + 1] else 2
            headings.append((index, index + 2, level, line.strip()))
    heading_markup = {
        line_index
        for start, body_start, _, _ in headings
        for line_index in range(start, body_start)
    }
    result: dict[str, str] = {}
    for order, (_, body_start, level, title) in enumerate(headings):
        end = next(
            (
                start
                for start, _, next_level, _ in headings[order + 1 :]
                if next_level <= level
            ),
            len(lines),
        )
        body = "\n".join(
            lines[index]
            for index in range(body_start, end)
            if index not in heading_markup
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
    if not _regular(ws, "report.md"):
        return 0.0, "report.md missing or not a regular file"
    try:
        headings = _headings(ws.read_text("report.md") or "")
    except OSError as exc:
        return 0.0, f"report.md read failed: {exc}"
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        absent = await ws.run(
            "test ! -e results.json && test ! -L results.json", check=False, timeout=30.0
        )
        state.run_started = True
        start = time.monotonic()
        run = await ws.run("python run.py", check=False, timeout=600.0)
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - start
        state.timed_out = state.exit_code is None and state.elapsed_s >= 600.0
        remove_ok = (
            getattr(removed, "exit_code", None) == 0
            and getattr(absent, "exit_code", None) == 0
        )
        state.result_file_is_new_regular = remove_ok and _regular(ws, "results.json")
        state.official_ok = (
            state.run_started
            and state.exit_code == 0
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, "
            f"new_results={state.result_file_is_new_regular}"
        )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = (
            _score_atoms(result_atoms, ATOM_SHARES["result_json"]) if state.official_ok else 0.0
        )
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.official_ok and state.parse_ok:
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
            replay_atoms = _replay_atoms(state.replayed)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
            required = (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            )
            if all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in required):
                state.metrics = _metrics(state.replayed)
                state.validity_gate = True
        else:
            for name in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[name] = 0.0
                state.diagnostics[name] = "official run or strict JSON parse failed"
    except Exception as exc:
        state.official_message = f"grader failure: {type(exc).__name__}: {exc}"
        for name in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(name, 0.0)
            state.diagnostics.setdefault(name, state.official_message)
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular(ws, "run.py"):
        return 0.0, "run.py missing or not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
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
):

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        if state.metrics is None:
            return 0.0, "validity gate failed; metric=0"
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
