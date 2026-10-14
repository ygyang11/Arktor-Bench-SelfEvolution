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

SCENES = (
    "steady_cell",
    "coverage_gap",
    "handover_route",
    "congested_return",
    "branching_search",
)
ROBOTS = tuple(range(3))
GRID_SIZE = 14
HORIZON = 48
ATTEMPT_BUDGET = 2400
CACHE_LIMIT = 600
EDGE_SERVICE_CAPACITY = 120
ROLLING_LIMIT = 3
EPS = 1e-6

STARTS = {
    "steady_cell": ((1, 1), (12, 1), (1, 12)),
    "coverage_gap": ((1, 1), (12, 2), (2, 12)),
    "handover_route": ((1, 1), (12, 1), (6, 12)),
    "congested_return": ((1, 1), (12, 1), (1, 12)),
    "branching_search": ((1, 1), (12, 1), (6, 10)),
}
MOVES = {
    "stay": (0, 0),
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
SCENE_KEYS = {"scenario", "frames"}
FRAME_KEYS = {"t", "robots"}
ROBOT_KEYS = {"robot", "move", "send_block_id", "priority", "drop_block_ids"}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.frame_containers": 0.20,
        "result_evidence.robot_containers": 0.20,
        "result_evidence.movement_fields": 0.20,
        "result_evidence.upload_fields": 0.20,
        "result_evidence.discard_fields": 0.20,
    },
    "decision_legality": {
        "decision_legality.known_free_moves": 0.125,
        "decision_legality.send_membership": 0.125,
        "decision_legality.priority_relation": 0.125,
        "decision_legality.rolling_attempts": 0.125,
        "decision_legality.byte_budget": 0.125,
        "decision_legality.discard_membership": 0.125,
        "decision_legality.discard_order": 0.125,
        "decision_legality.cache_capacity": 0.125,
    },
    "replay_validity": {
        "replay_validity.map_generation": 1 / 6,
        "replay_validity.radio_delivery": 1 / 6,
        "replay_validity.heartbeats": 1 / 6,
        "replay_validity.edge_service": 1 / 6,
        "replay_validity.global_summaries": 1 / 6,
        "replay_validity.block_lifecycle": 1 / 6,
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


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _record_shape(value: Any, keys: set[str], canonical_key: str) -> bool:
    # Every record's canonical field is part of the public container contract;
    # omitting it must not be mistaken for a valid shape.  Coverage owns
    # uniqueness/order, while this helper owns the exact key set.
    del canonical_key
    return isinstance(value, Mapping) and set(value) == keys


def _block_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"b[0-9]+", value) is not None


def _block_number(block_id: str) -> int:
    return int(block_id[1:])


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


def _obstacles(scene: str) -> set[tuple[int, int]]:
    if scene == "steady_cell":
        return {(4, y) for y in range(11) if y not in {2, 8}} | {
            (9, y) for y in range(3, 14) if y not in {5, 11}
        }
    if scene == "coverage_gap":
        return {(5, y) for y in range(13) if y not in {3, 10}} | {
            (8, y) for y in range(1, 14) if y not in {4, 11}
        }
    if scene == "handover_route":
        return {(x, 5) for x in range(12) if x not in {2, 9}} | {
            (x, 9) for x in range(2, 14) if x not in {4, 11}
        }
    if scene == "congested_return":
        return {(6, y) for y in range(1, 13) if y not in {3, 10}} | {
            (x, 7) for x in range(1, 13) if x not in {3, 9}
        }
    if scene == "branching_search":
        return (
            {(6, y) for y in range(2, 14) if y not in {5, 10}}
            | {(x, 6) for x in range(6) if x != 2}
            | {(x, 6) for x in range(7, 14) if x != 10}
        )
    raise ValueError(scene)


def _free_cells(scene: str) -> set[tuple[int, int]]:
    return {
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in _obstacles(scene)
    }


def _footprint(position: tuple[int, int]) -> set[tuple[int, int]]:
    x, y = position
    return {
        (nx, ny)
        for nx, ny in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
    }


def _capacity(scene: str, t: int) -> int:
    if scene == "steady_cell":
        return 144
    if scene == "coverage_gap":
        return 120 if t % 10 < 6 else 88
    if scene == "handover_route":
        return 96 if (t // 6) % 2 == 0 else 128
    if scene == "congested_return":
        return (144, 112, 80, 96)[(t % 12) // 3]
    if scene == "branching_search":
        return 112
    raise ValueError(scene)


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _radio_class(scene: str, position: tuple[int, int], t: int) -> int:
    x, y = position
    if scene == "steady_cell":
        return 2 if _manhattan(position, (6, 6)) <= 9 else 1
    if scene == "coverage_gap":
        if 5 <= x <= 8 and 2 <= y <= 11:
            return 0
        return 2 if _manhattan(position, (6, 6)) <= 8 else 1
    if scene == "handover_route":
        distance = min(_manhattan(position, (1, 6)), _manhattan(position, (12, 6)))
        value = 2 if distance <= 5 else 1 if distance <= 8 else 0
        return max(0, value - 1) if x in {6, 7} else value
    if scene == "congested_return":
        value = 2 if _manhattan(position, (6, 6)) <= 9 else 1
        return max(0, value - 1) if t % 12 in {7, 8, 9} else value
    if scene == "branching_search":
        distance = _manhattan(position, (6, 1))
        return 2 if distance <= 6 else 1 if distance <= 10 else 0
    raise ValueError(scene)


def _rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    return []


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = _shape(data, {"scenarios"})
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = isinstance(scenes, list) and all(
        _record_shape(row, SCENE_KEYS, "scenario") for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each scene requires frames and no unknown non-canonical keys",
        ),
    ]


def _id_list(value: Any) -> bool:
    return isinstance(value, list) and all(_block_id(item) for item in value)


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = (
        "frame_containers",
        "robot_containers",
        "movement_fields",
        "upload_fields",
        "discard_fields",
    )
    flags = {name: True for name in names}
    details = {name: "all submitted decision fields satisfy their local contract" for name in names}
    rows = _rows(data)
    if not rows:
        for name in names:
            flags[name] = False
            details[name] = "scenarios: no decision records"
    for scene_index, scene_row in enumerate(rows):
        frames = scene_row.get("frames") if isinstance(scene_row, Mapping) else None
        current = flags["frame_containers"]
        flags["frame_containers"] &= isinstance(frames, list) and all(
            _record_shape(frame, FRAME_KEYS, "t") and isinstance(frame.get("robots"), list)
            for frame in frames
        )
        if current and not flags["frame_containers"]:
            details["frame_containers"] = f"scenarios[{scene_index}].frames: invalid container"
        current = flags["robot_containers"]
        flags["robot_containers"] &= isinstance(frames, list) and all(
            isinstance(frame, Mapping)
            and isinstance(frame.get("robots"), list)
            and all(_record_shape(robot, ROBOT_KEYS, "robot") for robot in frame["robots"])
            for frame in frames
        )
        if current and not flags["robot_containers"]:
            details["robot_containers"] = (
                f"scenarios[{scene_index}].frames.robots: invalid keys or container"
            )
        current = flags["movement_fields"]
        flags["movement_fields"] &= isinstance(frames, list) and all(
            isinstance(frame, Mapping)
            and isinstance(frame.get("robots"), list)
            and all(
                isinstance(robot, Mapping)
                and isinstance(robot.get("move"), str)
                for robot in frame["robots"]
            )
            for frame in frames
        )
        if current and not flags["movement_fields"]:
            details["movement_fields"] = f"scenarios[{scene_index}]: invalid move type"
        current = flags["upload_fields"]
        flags["upload_fields"] &= isinstance(frames, list) and all(
            isinstance(frame, Mapping)
            and isinstance(frame.get("robots"), list)
            and all(
                isinstance(robot, Mapping)
                and (robot.get("send_block_id") is None or _block_id(robot.get("send_block_id")))
                and (robot.get("priority") is None or _integer(robot.get("priority")))
                for robot in frame["robots"]
            )
            for frame in frames
        )
        if current and not flags["upload_fields"]:
            details["upload_fields"] = f"scenarios[{scene_index}]: invalid send or priority type"
        current = flags["discard_fields"]
        flags["discard_fields"] &= isinstance(frames, list) and all(
            isinstance(frame, Mapping)
            and isinstance(frame.get("robots"), list)
            and all(
                isinstance(robot, Mapping) and _id_list(robot.get("drop_block_ids"))
                for robot in frame["robots"]
            )
            for frame in frames
        )
        if current and not flags["discard_fields"]:
            details["discard_fields"] = f"scenarios[{scene_index}]: invalid discard ID array"
    return [_atom("result_evidence", name, flags[name], details[name]) for name in names]


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
        frames = row.get("frames") if isinstance(row, Mapping) else None
        if not isinstance(frames, list):
            defects.append(f"{scene}.frames: expected list")
            continue
        frame_keys: list[Hashable] = []
        robot_keys: list[Hashable] = []
        for frame_index, frame in enumerate(frames):
            if isinstance(frame, Mapping) and _integer(frame.get("t")):
                frame_key: Hashable = frame["t"]
                frame_keys.append(frame_key)
                robots = frame.get("robots")
                if isinstance(robots, list):
                    for robot_index, robot in enumerate(robots):
                        if isinstance(robot, Mapping) and _integer(robot.get("robot")):
                            robot_keys.append((frame_key, robot["robot"]))
                        else:
                            robot_keys.append(("invalid_robot", frame_index, robot_index))
                else:
                    robot_keys.append(("invalid_robots", frame_index))
            else:
                frame_keys.append(("invalid_frame", frame_index))
        defects.extend(_canonical_defects(frame_keys, list(range(HORIZON)), f"{scene}.frames"))
        expected_robots = [(t, robot) for t in range(HORIZON) for robot in ROBOTS]
        defects.extend(_canonical_defects(robot_keys, expected_robots, f"{scene}.robots"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for row in _rows(data):
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            continue
        scene = row["scenario"]
        frames = row.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if (
                isinstance(frame, Mapping)
                and _integer(frame.get("t"))
                and (scene, frame["t"]) not in records
            ):
                records[(scene, frame["t"])] = frame
    return records


def _raw_robot_action(frame: Any, robot: int) -> Mapping[str, Any]:
    if not isinstance(frame, Mapping) or not isinstance(frame.get("robots"), list):
        return {}
    for row in frame["robots"]:
        if isinstance(row, Mapping) and row.get("robot") == robot:
            return row
    return {}


def _public_block(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block["id"],
        "generated_t": block["generated_t"],
        "cells": [list(cell) for cell in block["cells"]],
        "bytes": block["bytes"],
    }


def _simulate_scene(
    scene: str,
    before_source: Callable[[Mapping[str, Any]], Any],
    drop_source: Callable[[Mapping[str, Any], Any], Any],
) -> dict[str, Any]:
    free = _free_cells(scene)
    obstacles = _obstacles(scene)
    positions = [STARTS[scene][robot] for robot in ROBOTS]
    local_free: list[set[tuple[int, int]]] = []
    local_obstacles: list[set[tuple[int, int]]] = []
    caches: list[list[str]] = [[] for _ in ROBOTS]
    blocks: dict[str, dict[str, Any]] = {}
    next_block = 0
    for robot in ROBOTS:
        observed = _footprint(positions[robot])
        known_free = observed & free
        local_free.append(set(known_free))
        local_obstacles.append(observed & obstacles)
        block_id = f"b{next_block}"
        next_block += 1
        cells = sorted(known_free)
        blocks[block_id] = {
            "id": block_id,
            "robot": robot,
            "generated_t": -1,
            "cells": cells,
            "bytes": 16 + 8 * len(cells),
            "state": "cached",
            "priority": None,
            "arrival_t": None,
        }
        caches[robot].append(block_id)

    map_arrivals: dict[int, list[str]] = {}
    heartbeat_arrivals: dict[int, list[dict[str, Any]]] = {}
    summary_arrivals: dict[int, list[dict[str, Any]]] = {}
    latest_heartbeat: list[dict[str, Any] | None] = [None for _ in ROBOTS]
    latest_summary: list[dict[str, Any] | None] = [None for _ in ROBOTS]
    edge_queue: list[str] = []
    merged: set[tuple[int, int]] = set()
    attempt_bytes = 0
    attempts_by_robot: list[list[int]] = [[] for _ in ROBOTS]
    frames: list[dict[str, Any]] = []
    coverage_series: list[float] = []
    cache_score_sum = 0.0
    summary_score_sum = 0.0
    legality_names = (
        "known_free_moves",
        "send_membership",
        "priority_relation",
        "rolling_attempts",
        "byte_budget",
        "discard_membership",
        "discard_order",
        "cache_capacity",
    )
    legality = {name: True for name in legality_names}
    legality_errors: dict[str, list[str]] = {name: [] for name in legality_names}
    replay_names = (
        "map_generation",
        "radio_delivery",
        "heartbeats",
        "edge_service",
        "global_summaries",
        "block_lifecycle",
    )
    replay = {name: True for name in replay_names}
    replay_errors: dict[str, list[str]] = {name: [] for name in replay_names}
    consumed_actions: dict[str, dict[tuple[int, int], Any]] = {
        name: {} for name in ("move", "send_block_id", "priority", "drop_block_ids")
    }

    for t in range(HORIZON):
        for block_id in map_arrivals.get(t, []):
            block = blocks[block_id]
            if block["state"] != "in_flight" or block["arrival_t"] != t:
                replay["radio_delivery"] = False
                replay_errors["radio_delivery"].append(
                    f"{scene}.t={t}: inconsistent arrival for {block_id}"
                )
                continue
            block["state"] = "edge_queue"
            edge_queue.append(block_id)

        edge_queue.sort(
            key=lambda block_id: (
                -int(blocks[block_id]["priority"]),
                int(blocks[block_id]["generated_t"]),
                _block_number(block_id),
            )
        )
        service_remaining = EDGE_SERVICE_CAPACITY
        kept_queue: list[str] = []
        serviced: list[str] = []
        for block_id in edge_queue:
            block = blocks[block_id]
            if block["bytes"] <= service_remaining:
                service_remaining -= block["bytes"]
                block["state"] = "serviced"
                serviced.append(block_id)
                merged.update(block["cells"])
            else:
                kept_queue.append(block_id)
        edge_queue = kept_queue
        if sum(blocks[block_id]["bytes"] for block_id in serviced) > EDGE_SERVICE_CAPACITY:
            replay["edge_service"] = False
            replay_errors["edge_service"].append(f"{scene}.t={t}: service capacity exceeded")
        if not merged <= free:
            replay["edge_service"] = False
            replay_errors["edge_service"].append(f"{scene}.t={t}: merged non-free cell")
        coverage_series.append(len(merged) / len(free))

        for report in heartbeat_arrivals.get(t, []):
            robot = int(report["robot"])
            previous = latest_heartbeat[robot]
            if previous is None or report["sample_t"] > previous["sample_t"]:
                latest_heartbeat[robot] = report
        for summary in summary_arrivals.get(t, []):
            robot = int(summary["robot"])
            previous = latest_summary[robot]
            if previous is None or summary["version"] > previous["version"]:
                latest_summary[robot] = summary

        raw_frame = before_source({"t": t})
        radio_classes = [_radio_class(scene, positions[robot], t) for robot in ROBOTS]
        capacity = _capacity(scene, t)
        remaining_capacity = capacity
        current_attempt_bytes = 0
        normalized_actions: list[dict[str, Any]] = []
        raw_actions: list[Any] = []
        proposed_moves: list[str] = []

        for robot in ROBOTS:
            summary = latest_summary[robot]
            heartbeat = latest_heartbeat[robot]
            previous_three = sum(
                max(0, t - 3) <= attempt_t < t for attempt_t in attempts_by_robot[robot]
            )
            view = {
                "t": t,
                "robot": robot,
                "position": list(positions[robot]),
                "local_free_cells": [list(cell) for cell in sorted(local_free[robot])],
                "local_obstacle_cells": [list(cell) for cell in sorted(local_obstacles[robot])],
                "latest_heartbeat": dict(heartbeat) if heartbeat is not None else None,
                "global_summary": (
                    {
                        "version": summary["version"],
                        "cells": [list(cell) for cell in summary["cells"]],
                    }
                    if summary is not None
                    else None
                ),
                "cache": [_public_block(blocks[block_id]) for block_id in caches[robot]],
                "remaining_attempt_bytes": max(0, ATTEMPT_BUDGET - attempt_bytes),
                "attempts_in_previous_three_slots": previous_three,
                "earlier_current_attempt_bytes": current_attempt_bytes,
            }
            supplied = (
                before_source(view) if raw_frame is None else _raw_robot_action(raw_frame, robot)
            )
            raw_actions.append(supplied)
            action = supplied if isinstance(supplied, Mapping) else {}
            move = action.get("move")
            send_id = action.get("send_block_id")
            priority = action.get("priority")
            consumed_actions["send_block_id"][(t, robot)] = send_id
            consumed_actions["priority"][(t, robot)] = priority
            proposed_moves.append(move if isinstance(move, str) else "")

            upload_types_ok = (send_id is None or _block_id(send_id)) and (
                priority is None or _integer(priority)
            )
            relation_ok = False
            send_valid = False
            if upload_types_ok:
                relation_ok = (send_id is None and priority is None) or (
                    _block_id(send_id) and _integer(priority) and 0 <= priority <= 3
                )
                if not relation_ok:
                    legality["priority_relation"] = False
                    legality_errors["priority_relation"].append(
                        f"{scene}.frames[{t}].robots[{robot}]: send/priority relation invalid"
                    )
                send_valid = _block_id(send_id) and send_id in caches[robot]
                if send_id is not None and not send_valid:
                    legality["send_membership"] = False
                    legality_errors["send_membership"].append(
                        f"{scene}.frames[{t}].robots[{robot}].send_block_id={send_id!r} not cached"
                    )
            if send_valid and relation_ok:
                block = blocks[send_id]
                caches[robot].remove(send_id)
                block["state"] = "attempted"
                block["priority"] = priority
                attempts_by_robot[robot].append(t)
                attempt_bytes += block["bytes"]
                current_attempt_bytes += block["bytes"]
                if previous_three + 1 > ROLLING_LIMIT:
                    legality["rolling_attempts"] = False
                    legality_errors["rolling_attempts"].append(
                        f"{scene}.frames[{t}].robots[{robot}]: "
                        f"four-slot attempts={previous_three + 1}"
                    )
                if attempt_bytes > ATTEMPT_BUDGET:
                    legality["byte_budget"] = False
                    legality_errors["byte_budget"].append(
                        f"{scene}.frames[{t}]: attempt_bytes={attempt_bytes} "
                        f"exceeds {ATTEMPT_BUDGET}"
                    )
                if radio_classes[robot] > 0 and block["bytes"] <= remaining_capacity:
                    remaining_capacity -= block["bytes"]
                    arrival_t = t + 3 - radio_classes[robot]
                    block["arrival_t"] = arrival_t
                    if arrival_t < HORIZON:
                        block["state"] = "in_flight"
                        map_arrivals.setdefault(arrival_t, []).append(send_id)
                    else:
                        block["state"] = "late"
                else:
                    block["state"] = "lost"

        for robot in ROBOTS:
            radio_class = radio_classes[robot]
            if radio_class == 0:
                continue
            delay = (
                3
                - radio_class
                + int(current_attempt_bytes > 0.75 * capacity)
                + int(current_attempt_bytes > capacity)
            )
            report = {
                "robot": robot,
                "sample_t": t,
                "radio_class": radio_class,
                "capacity": capacity,
                "attempted_map_bytes": current_attempt_bytes,
                "delay": delay,
            }
            arrival_t = t + delay
            if arrival_t < HORIZON:
                heartbeat_arrivals.setdefault(arrival_t, []).append(report)

        for robot in ROBOTS:
            move = proposed_moves[robot]
            submitted_move = raw_actions[robot].get("move") if isinstance(
                raw_actions[robot], Mapping
            ) else None
            consumed_actions["move"][(t, robot)] = submitted_move
            displacement = MOVES.get(move) if isinstance(submitted_move, str) else None
            destination = None
            if displacement is not None:
                destination = (
                    positions[robot][0] + displacement[0],
                    positions[robot][1] + displacement[1],
                )
            if not isinstance(submitted_move, str):
                pass
            elif destination is None or destination not in local_free[robot]:
                legality["known_free_moves"] = False
                legality_errors["known_free_moves"].append(
                    f"{scene}.frames[{t}].robots[{robot}].move={move!r} "
                    "does not reach known free cell"
                )
            else:
                positions[robot] = destination

            observed = _footprint(positions[robot])
            newly_free = sorted((observed & free) - local_free[robot])
            local_free[robot].update(observed & free)
            local_obstacles[robot].update(observed & obstacles)
            generated_id: str | None = None
            if newly_free:
                generated_id = f"b{next_block}"
                next_block += 1
                blocks[generated_id] = {
                    "id": generated_id,
                    "robot": robot,
                    "generated_t": t,
                    "cells": newly_free,
                    "bytes": 16 + 8 * len(newly_free),
                    "state": "cached",
                    "priority": None,
                    "arrival_t": None,
                }
                caches[robot].append(generated_id)
                if not newly_free or len(newly_free) > 5 or not set(newly_free) <= free:
                    replay["map_generation"] = False
                    replay_errors["map_generation"].append(
                        f"{scene}.frames[{t}].robots[{robot}]: invalid generated block"
                    )

            post_view = {
                "t": t,
                "robot": robot,
                "position": list(positions[robot]),
                "new_block_id": generated_id,
                "cache": [_public_block(blocks[block_id]) for block_id in caches[robot]],
            }
            raw_drops = drop_source(post_view, raw_actions[robot])
            consumed_actions["drop_block_ids"][(t, robot)] = raw_drops
            drops = list(raw_drops) if _id_list(raw_drops) else []
            if drops:
                order_ok = len(drops) == len(set(drops)) and drops == sorted(
                    drops, key=_block_number
                )
                if not order_ok:
                    legality["discard_order"] = False
                    legality_errors["discard_order"].append(
                        f"{scene}.frames[{t}].robots[{robot}].drop_block_ids: "
                        "expected unique numeric order"
                    )
            cache_before_drops = set(caches[robot])
            unknown_drops = [block_id for block_id in drops if block_id not in cache_before_drops]
            if unknown_drops:
                legality["discard_membership"] = False
                legality_errors["discard_membership"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: "
                    f"discard {unknown_drops[0]!r} not cached"
                )
            for block_id in drops:
                if block_id not in caches[robot]:
                    continue
                caches[robot].remove(block_id)
                blocks[block_id]["state"] = "discarded"
            cache_bytes = sum(blocks[block_id]["bytes"] for block_id in caches[robot])
            if cache_bytes > CACHE_LIMIT:
                legality["cache_capacity"] = False
                legality_errors["cache_capacity"].append(
                    f"{scene}.frames[{t}].robots[{robot}]: "
                    f"cache_bytes={cache_bytes} exceeds {CACHE_LIMIT}"
                )
            cache_score_sum += max(0.0, min(1.0, 1.0 - cache_bytes / CACHE_LIMIT))
            summary = latest_summary[robot]
            age = 4 if summary is None else min(4, t - int(summary["version"]))
            summary_score_sum += max(0.0, 1.0 - age / 4.0)
            normalized_actions.append(
                {
                    "robot": robot,
                    "move": move,
                    "send_block_id": (
                        raw_actions[robot].get("send_block_id")
                        if isinstance(raw_actions[robot], Mapping)
                        else None
                    ),
                    "priority": (
                        raw_actions[robot].get("priority")
                        if isinstance(raw_actions[robot], Mapping)
                        else None
                    ),
                    "drop_block_ids": list(drops),
                }
            )

        snapshot = tuple(sorted(merged))
        for robot in ROBOTS:
            radio_class = _radio_class(scene, positions[robot], t)
            if radio_class == 0:
                continue
            arrival_t = t + 3 - radio_class
            if arrival_t < HORIZON:
                summary_arrivals.setdefault(arrival_t, []).append(
                    {
                        "robot": robot,
                        "version": t,
                        "cells": snapshot,
                    }
                )
        frames.append({"t": t, "robots": normalized_actions})

    for block_id, block in blocks.items():
        if block["bytes"] != 16 + 8 * len(block["cells"]):
            replay["map_generation"] = False
            replay_errors["map_generation"].append(f"{scene}.{block_id}: byte formula mismatch")
        if block["state"] not in {
            "cached",
            "in_flight",
            "edge_queue",
            "serviced",
            "lost",
            "late",
            "discarded",
        }:
            replay["block_lifecycle"] = False
            replay_errors["block_lifecycle"].append(
                f"{scene}.{block_id}: invalid terminal state={block['state']!r}"
            )
    if any(
        report["sample_t"] >= t for t, reports in heartbeat_arrivals.items() for report in reports
    ):
        replay["heartbeats"] = False
        replay_errors["heartbeats"].append(f"{scene}: noncausal heartbeat arrival")
    if any(summary["version"] >= t for t, values in summary_arrivals.items() for summary in values):
        replay["global_summaries"] = False
        replay_errors["global_summaries"].append(f"{scene}: noncausal summary arrival")

    objects = HORIZON * len(ROBOTS)
    return {
        "scenario_record": {"scenario": scene, "frames": frames},
        "legality": legality,
        "legality_errors": legality_errors,
        "replay": replay,
        "replay_errors": replay_errors,
        "decision_stream_complete": True,
        "consumed_actions": consumed_actions,
        "terminal_coverage": len(merged) / len(free),
        "progress": sum(coverage_series) / HORIZON,
        "attempt_bytes": attempt_bytes,
        "cache_summary_quality": (cache_score_sum + summary_score_sum) / (2 * objects),
        "blocks": blocks,
        "merged_cells": merged,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    def before(view: Mapping[str, Any]) -> Any:
        return records.get((scene, int(view["t"])), {})

    def drops(_view: Mapping[str, Any], raw_action: Any) -> Any:
        return raw_action.get("drop_block_ids") if isinstance(raw_action, Mapping) else None

    result = _simulate_scene(scene, before, drops)
    missing = [t for t in range(HORIZON) if (scene, t) not in records]
    incomplete = []
    for t in range(HORIZON):
        frame = records.get((scene, t))
        robots = frame.get("robots") if isinstance(frame, Mapping) else None
        robot_ids = (
            [row.get("robot") for row in robots if isinstance(row, Mapping)]
            if isinstance(robots, list)
            else []
        )
        if len(robot_ids) != len(ROBOTS) or robot_ids != list(ROBOTS):
            incomplete.append(t)
    if missing or incomplete:
        result["decision_stream_complete"] = False
        detail = (
            f"first missing frame={missing[0]}"
            if missing
            else f"first incomplete robot frame={incomplete[0]}"
        )
        for name in result["replay"]:
            result["replay"][name] = False
            result["replay_errors"][name].append(
                f"{scene}: decision stream unavailable; {detail}"
            )
    return result


def _online_scene(scene: str, policy: Any) -> dict[str, Any]:
    if hasattr(policy, "reset"):
        policy.reset()

    def before(view: Mapping[str, Any]) -> Any:
        if set(view) == {"t"}:
            return None
        return policy.decide_robot(view)

    def drops(view: Mapping[str, Any], _raw_action: Any) -> Any:
        return policy.decide_drops(view)

    return _simulate_scene(scene, before, drops)["scenario_record"]


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    names = (
        "known_free_moves",
        "send_membership",
        "priority_relation",
        "rolling_attempts",
        "byte_budget",
        "discard_membership",
        "discard_order",
        "cache_capacity",
    )
    flags = {name: True for name in names}
    details = {name: "legal in all scenes" for name in names}
    for scene in SCENES:
        result = replayed.get(scene, {})
        if isinstance(result, Mapping) and not result.get("decision_stream_complete", False):
            continue
        scene_flags = result.get("legality", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("legality_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if not scene_flags.get(name, False):
                flags[name] = False
                failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                details[name] = failures[0] if failures else f"{scene}: {name} unavailable"
    return [_atom("decision_legality", name, flags[name], details[name]) for name in names]


def _consumption_mismatch(
    scene: str,
    records: Mapping[Hashable, Any],
    result: Mapping[str, Any],
    field: str,
) -> str | None:
    consumed = result.get("consumed_actions")
    values = consumed.get(field) if isinstance(consumed, Mapping) else None
    if not isinstance(values, Mapping):
        return f"{scene}: consumed {field} audit unavailable"
    for t in range(HORIZON):
        frame = records.get((scene, t))
        for robot in ROBOTS:
            action = _raw_robot_action(frame, robot)
            expected = action.get(field)
            key = (t, robot)
            if key not in values:
                return f"{scene}.frames[{t}].robots[{robot}].{field}: consumption unavailable"
            actual = values[key]
            if expected != actual:
                return (
                    f"{scene}.frames[{t}].robots[{robot}].{field}: "
                    f"submitted={expected!r}, consumed={actual!r}"
                )
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Any]
) -> list[AtomVerdict]:
    names = (
        "map_generation",
        "radio_delivery",
        "heartbeats",
        "edge_service",
        "global_summaries",
        "block_lifecycle",
    )
    flags = {name: True for name in names}
    details = {name: "fixed-model replay completed" for name in names}
    field_owners = {
        "map_generation": "move",
        "radio_delivery": "send_block_id",
        "edge_service": "priority",
        "block_lifecycle": "drop_block_ids",
    }
    for scene in SCENES:
        result = replayed.get(scene, {})
        scene_flags = result.get("replay", {}) if isinstance(result, Mapping) else {}
        scene_errors = result.get("replay_errors", {}) if isinstance(result, Mapping) else {}
        for name in names:
            if not scene_flags.get(name, False):
                flags[name] = False
                failures = scene_errors.get(name, []) if isinstance(scene_errors, Mapping) else []
                details[name] = failures[0] if failures else f"{scene}: replay unavailable"
        if isinstance(result, Mapping):
            for name, field in field_owners.items():
                mismatch = _consumption_mismatch(scene, records, result, field)
                if mismatch is not None:
                    flags[name] = False
                    details[name] = mismatch
    return [_atom("replay_validity", name, flags[name], details[name]) for name in names]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    primary_1 = sum(float(replayed[scene]["terminal_coverage"]) for scene in SCENES) / len(SCENES)
    primary_2 = sum(float(replayed[scene]["progress"]) for scene in SCENES) / len(SCENES)
    auxiliary_1 = sum(
        max(0.0, min(1.0, 1.0 - float(replayed[scene]["attempt_bytes"]) / ATTEMPT_BUDGET))
        for scene in SCENES
    ) / len(SCENES)
    auxiliary_2 = sum(float(replayed[scene]["cache_summary_quality"]) for scene in SCENES) / len(
        SCENES
    )
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
            (start for start, _, next_level, _ in headings[order + 1 :] if next_level <= level),
            len(lines),
        )
        body = "\n".join(
            lines[index] for index in range(body_start, end) if index not in heading_markup
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
        report_text = ws.read_text("report.md") or ""
    except (OSError, TypeError, UnicodeError) as exc:
        return 0.0, f"report.md read failed: {exc}"
    headings = _headings(report_text)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        state.run_started = True
        start = time.monotonic()
        run = await ws.run("python run.py", check=False, timeout=600.0)
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - start
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
            replay_atoms = _replay_atoms(state.records, state.replayed)
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
    except Exception as exc:  # noqa: BLE001
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
            failed = []
            if not state.official_ok:
                failed.append("official_run")
            failed.extend(
                f"{name}={state.scores.get(name, 0.0):.6f}"
                for name in (
                    "result_json",
                    "result_evidence",
                    "canonical_coverage",
                    "decision_legality",
                    "replay_validity",
                )
                if state.scores.get(name, 0.0) < 1.0 - EPS
            )
            return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
