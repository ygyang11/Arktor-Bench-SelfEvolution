from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace


# Fixed task model.
GRID_SIZE = 9
HORIZON = 40
ROBOT_COUNT = 3
TASK_COUNT = 8
SENSE_RANGE = 1
REPORT_RANGE = 3
CHANNEL_COUNT = 2
REPORT_BUDGET = 24
BURST_WINDOW = 4
BURST_LIMIT = 3
FRESH_AGE = 3
EPS = 1e-6
MOVES = ("north", "east", "south", "west", "stay")
MOVE_DELTA = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
    "stay": (0, 0),
}
PHASES = ("idle", "to_pickup", "to_delivery")
SCENARIOS = ("cross_aisles", "dual_bottleneck", "ring_aisle", "staggered", "crossing")

LAYOUTS: dict[str, tuple[str, ...]] = {
    "cross_aisles": (
        "#########",
        "#.......#",
        "#..###..#",
        "#.......#",
        "#..#....#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ),
    "dual_bottleneck": (
        "#########",
        "#...#...#",
        "#...#...#",
        "#.......#",
        "###.#####",
        "#.......#",
        "#...#...#",
        "#...#...#",
        "#########",
    ),
    "ring_aisle": (
        "#########",
        "#.......#",
        "#.#####.#",
        "#.......#",
        "#...#...#",
        "#.......#",
        "#.#####.#",
        "#.......#",
        "#########",
    ),
    "staggered": (
        "#########",
        "#..#....#",
        "#..#....#",
        "#.......#",
        "#....#..#",
        "#....#..#",
        "#.......#",
        "#.......#",
        "#########",
    ),
    "crossing": (
        "#########",
        "#.......#",
        "#...#...#",
        "#...#...#",
        "#.......#",
        "#...#...#",
        "#...#...#",
        "#.......#",
        "#########",
    ),
}

STARTS: dict[str, tuple[tuple[int, int], ...]] = {
    "cross_aisles": ((1, 1), (7, 1), (1, 7)),
    "dual_bottleneck": ((1, 1), (7, 1), (7, 7)),
    "ring_aisle": ((1, 1), (7, 1), (1, 7)),
    "staggered": ((1, 1), (7, 1), (7, 7)),
    "crossing": ((1, 1), (7, 1), (1, 7)),
}

# (release, pickup, delivery, deadline, predecessor)
TASKS: dict[str, tuple[dict[str, Any], ...]] = {
    "cross_aisles": (
        {"release": 0, "pickup": (2, 1), "delivery": (6, 1), "deadline": 14, "predecessor": None},
        {"release": 0, "pickup": (6, 7), "delivery": (2, 7), "deadline": 16, "predecessor": None},
        {"release": 4, "pickup": (1, 3), "delivery": (7, 3), "deadline": 23, "predecessor": 0},
        {"release": 5, "pickup": (7, 5), "delivery": (1, 5), "deadline": 25, "predecessor": 1},
        {"release": 9, "pickup": (3, 1), "delivery": (3, 7), "deadline": 31, "predecessor": 2},
        {"release": 11, "pickup": (5, 7), "delivery": (5, 1), "deadline": 33, "predecessor": 3},
        {"release": 17, "pickup": (1, 4), "delivery": (7, 4), "deadline": 37, "predecessor": 4},
        {"release": 19, "pickup": (7, 6), "delivery": (1, 6), "deadline": 39, "predecessor": 5},
    ),
    "dual_bottleneck": (
        {"release": 0, "pickup": (2, 1), "delivery": (6, 1), "deadline": 15, "predecessor": None},
        {"release": 0, "pickup": (6, 7), "delivery": (2, 7), "deadline": 17, "predecessor": None},
        {"release": 4, "pickup": (1, 3), "delivery": (7, 3), "deadline": 24, "predecessor": 0},
        {"release": 5, "pickup": (7, 5), "delivery": (1, 5), "deadline": 26, "predecessor": 1},
        {"release": 9, "pickup": (3, 2), "delivery": (3, 6), "deadline": 32, "predecessor": 2},
        {"release": 11, "pickup": (5, 6), "delivery": (5, 2), "deadline": 34, "predecessor": 3},
        {"release": 17, "pickup": (3, 5), "delivery": (7, 5), "deadline": 38, "predecessor": 4},
        {"release": 19, "pickup": (5, 3), "delivery": (1, 3), "deadline": 39, "predecessor": 5},
    ),
    "ring_aisle": (
        {"release": 0, "pickup": (2, 1), "delivery": (6, 1), "deadline": 14, "predecessor": None},
        {"release": 0, "pickup": (6, 7), "delivery": (2, 7), "deadline": 16, "predecessor": None},
        {"release": 4, "pickup": (1, 5), "delivery": (7, 5), "deadline": 23, "predecessor": 0},
        {"release": 5, "pickup": (7, 3), "delivery": (1, 3), "deadline": 25, "predecessor": 1},
        {"release": 9, "pickup": (3, 5), "delivery": (3, 3), "deadline": 31, "predecessor": 2},
        {"release": 11, "pickup": (5, 3), "delivery": (5, 5), "deadline": 33, "predecessor": 3},
        {"release": 17, "pickup": (1, 4), "delivery": (7, 4), "deadline": 37, "predecessor": 4},
        {"release": 19, "pickup": (7, 6), "delivery": (1, 6), "deadline": 39, "predecessor": 5},
    ),
    "staggered": (
        {"release": 0, "pickup": (2, 1), "delivery": (6, 1), "deadline": 14, "predecessor": None},
        {"release": 0, "pickup": (6, 7), "delivery": (2, 7), "deadline": 16, "predecessor": None},
        {"release": 4, "pickup": (1, 5), "delivery": (7, 5), "deadline": 23, "predecessor": 0},
        {"release": 5, "pickup": (7, 3), "delivery": (1, 3), "deadline": 25, "predecessor": 1},
        {"release": 9, "pickup": (2, 2), "delivery": (6, 2), "deadline": 31, "predecessor": 2},
        {"release": 11, "pickup": (6, 6), "delivery": (2, 6), "deadline": 33, "predecessor": 3},
        {"release": 17, "pickup": (1, 4), "delivery": (7, 4), "deadline": 37, "predecessor": 4},
        {"release": 19, "pickup": (7, 5), "delivery": (1, 5), "deadline": 39, "predecessor": 5},
    ),
    "crossing": (
        {"release": 0, "pickup": (2, 1), "delivery": (6, 1), "deadline": 14, "predecessor": None},
        {"release": 0, "pickup": (6, 7), "delivery": (2, 7), "deadline": 16, "predecessor": None},
        {"release": 4, "pickup": (1, 4), "delivery": (7, 4), "deadline": 23, "predecessor": 0},
        {"release": 5, "pickup": (7, 4), "delivery": (1, 4), "deadline": 25, "predecessor": 1},
        {"release": 9, "pickup": (3, 1), "delivery": (3, 7), "deadline": 31, "predecessor": 2},
        {"release": 11, "pickup": (5, 7), "delivery": (5, 1), "deadline": 33, "predecessor": 3},
        {"release": 17, "pickup": (1, 6), "delivery": (7, 6), "deadline": 37, "predecessor": 4},
        {"release": 19, "pickup": (7, 2), "delivery": (1, 2), "deadline": 39, "predecessor": 5},
    ),
}


# Relative atom shares are task-specific and sum to one within each check.
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 1.0 / 3.0,
        "result_json.root": 1.0 / 3.0,
        "result_json.scenarios": 1.0 / 3.0,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 1.0 / 8.0,
        "result_evidence.decision_shape": 1.0 / 8.0,
        "result_evidence.decision_types": 1.0 / 8.0,
        "result_evidence.trace_shape": 1.0 / 8.0,
        "result_evidence.trace_values": 1.0 / 8.0,
        "result_evidence.packet_evidence": 1.0 / 8.0,
        "result_evidence.task_shape": 1.0 / 8.0,
        "result_evidence.task_types": 1.0 / 8.0,
    },
    "decision_legality": {
        "decision_legality.time_domain": 0.1,
        "decision_legality.robot_domain": 0.1,
        "decision_legality.move_domain": 0.1,
        "decision_legality.claim_domain": 0.1,
        "decision_legality.claim_eligibility": 0.1,
        "decision_legality.report_target": 0.1,
        "decision_legality.commitment_relation": 0.1,
        "decision_legality.commitment_execution": 0.1,
        "decision_legality.report_budget": 0.1,
        "decision_legality.burst_limit": 0.1,
    },
    "replay_validity": {
        "replay_validity.packet_events": 1.0 / 8.0,
        "replay_validity.report_ages": 1.0 / 8.0,
        "replay_validity.pre_positions": 1.0 / 8.0,
        "replay_validity.post_positions": 1.0 / 8.0,
        "replay_validity.blocked_flags": 1.0 / 8.0,
        "replay_validity.collision_flags": 1.0 / 8.0,
        "replay_validity.robot_task_state": 1.0 / 8.0,
        "replay_validity.task_records": 1.0 / 8.0,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = bool(passed)
        self.detail = detail
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
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[Hashable, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _coordinate(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(
        _integer(part) and 0 <= part < GRID_SIZE for part in value
    )


def _close(a: Any, b: Any) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= EPS


def _layout_cell(scenario: str, x: int, y: int) -> str:
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return "#"
    return LAYOUTS[scenario][GRID_SIZE - 1 - y][x]


def _free_cells(scenario: str) -> set[tuple[int, int]]:
    return {
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if _layout_cell(scenario, x, y) == "."
    }


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique_pairs)


def _require_replay_decisions(decisions: Any) -> None:
    if not isinstance(decisions, Mapping) or set(decisions) != set(range(HORIZON)):
        raise ValueError("replay requires one valid decision record for every time slot")
    for t in range(HORIZON):
        if not _decision_valid(decisions[t]):
            raise ValueError(f"replay decision at t={t} is malformed or outside its declared domain")


def _report_delivered(scenario_index: int, t: int, sender: int, receiver: int, distance: int, channel_unique: bool) -> bool:
    return (
        distance <= REPORT_RANGE
        and channel_unique
        and (t + 3 * sender + 5 * receiver + scenario_index) % 7 != 0
    )


def _initial_task_state() -> tuple[list[int | None], list[str], list[bool]]:
    return [None] * ROBOT_COUNT, ["idle"] * ROBOT_COUNT, [False] * ROBOT_COUNT


def _robot_rows(decision_row: Any) -> dict[int, Mapping[str, Any]]:
    if not isinstance(decision_row, Mapping) or not isinstance(decision_row.get("robots"), list):
        return {}
    rows: dict[int, Mapping[str, Any]] = {}
    for item in decision_row["robots"]:
        if isinstance(item, Mapping) and _integer(item.get("robot")):
            rows.setdefault(int(item["robot"]), item)
    return rows


def _replay_scenario(scenario: str, decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    _require_replay_decisions(decisions)
    scenario_index = SCENARIOS.index(scenario)
    positions = [tuple(point) for point in STARTS[scenario]]
    active, phases, loaded = _initial_task_state()
    assigned: list[int | None] = [None] * TASK_COUNT
    pickup_times: list[int | None] = [None] * TASK_COUNT
    delivery_times: list[int | None] = [None] * TASK_COUNT
    completed: set[int] = set()
    pending: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reports: list[list[dict[str, Any] | None]] = [[None] * ROBOT_COUNT for _ in range(ROBOT_COUNT)]
    trace: list[dict[str, Any]] = []
    freshness_values: list[float] = []
    collision_slots = 0
    report_count = 0
    claim_violations: list[str] = []
    commitment_violations: list[str] = []
    required_moves: list[str | None] = [None] * ROBOT_COUNT

    for t in range(HORIZON):
        for event in sorted(pending.pop(t, []), key=lambda item: (item["sender"], item["receiver"])):
            reports[event["receiver"]][event["sender"]] = event

        report_ages: list[list[int | None]] = []
        for receiver in range(ROBOT_COUNT):
            row: list[int | None] = []
            for sender in range(ROBOT_COUNT):
                event = reports[receiver][sender]
                row.append(None if event is None else t - int(event["generation"]))
                if sender != receiver and event is not None and t - int(event["generation"]) <= FRESH_AGE:
                    freshness_values.append(1.0)
                elif sender != receiver:
                    freshness_values.append(0.0)
            report_ages.append(row)

        rows = _robot_rows(decisions.get(t, {}))
        for robot in range(ROBOT_COUNT):
            required = required_moves[robot]
            if required is not None and rows.get(robot, {}).get("move") != required:
                commitment_violations.append(
                    f"t={t},robot={robot}: move {rows.get(robot, {}).get('move')} "
                    f"does not execute prior commitment {required}"
                )
        task_specs = TASKS[scenario]
        eligible: set[int] = set()
        for task_id, spec in enumerate(task_specs):
            predecessor = spec["predecessor"]
            if (
                spec["release"] <= t
                and assigned[task_id] is None
                and (predecessor is None or predecessor in completed)
            ):
                eligible.add(task_id)

        claims: dict[int, int | None] = {}
        for robot in range(ROBOT_COUNT):
            value = rows.get(robot, {}).get("claim")
            claims[robot] = int(value) if _integer(value) else None
        claim_groups: dict[int, list[int]] = defaultdict(list)
        for robot, task_id in claims.items():
            if task_id is None:
                continue
            if active[robot] is not None:
                claim_violations.append(f"t={t},robot={robot}: claim while carrying task")
            elif task_id not in eligible:
                claim_violations.append(f"t={t},robot={robot}: task {task_id} is not eligible")
            else:
                claim_groups[task_id].append(robot)
        for task_id, robots in claim_groups.items():
            winner = min(robots)
            assigned[task_id] = winner
            active[winner] = task_id
            phases[winner] = "to_pickup"
            loaded[winner] = False

        proposed: list[tuple[int, int]] = []
        blocked = [False] * ROBOT_COUNT
        collision = [False] * ROBOT_COUNT
        for robot in range(ROBOT_COUNT):
            move = rows.get(robot, {}).get("move")
            dx, dy = MOVE_DELTA.get(move, (0, 0))
            target = (positions[robot][0] + dx, positions[robot][1] + dy)
            proposed.append(target)
            if _layout_cell(scenario, target[0], target[1]) != ".":
                blocked[robot] = True

        for first in range(ROBOT_COUNT):
            for second in range(first + 1, ROBOT_COUNT):
                if proposed[first] == positions[second] and proposed[second] == positions[first]:
                    collision[first] = collision[second] = True
                elif not blocked[first] and proposed[first] == positions[second] and proposed[second] != positions[first]:
                    blocked[first] = True
                elif not blocked[second] and proposed[second] == positions[first] and proposed[first] != positions[second]:
                    blocked[second] = True
        current = set(positions)
        for first in range(ROBOT_COUNT):
            for second in range(first + 1, ROBOT_COUNT):
                if (
                    not blocked[first] and not blocked[second]
                    and proposed[first] == proposed[second]
                    and proposed[first] not in current
                ):
                    collision[first] = collision[second] = True
        post_positions = [
            proposed[robot] if not blocked[robot] and not collision[robot] else positions[robot]
            for robot in range(ROBOT_COUNT)
        ]
        if any(collision):
            collision_slots += 1

        for robot in range(ROBOT_COUNT):
            task_id = active[robot]
            if task_id is None:
                continue
            spec = task_specs[task_id]
            if phases[robot] == "to_pickup" and post_positions[robot] == spec["pickup"]:
                phases[robot] = "to_delivery"
                loaded[robot] = True
                pickup_times[task_id] = t + 1
            elif phases[robot] == "to_delivery" and loaded[robot] and post_positions[robot] == spec["delivery"]:
                delivery_times[task_id] = t + 1
                completed.add(task_id)
                active[robot] = None
                phases[robot] = "idle"
                loaded[robot] = False

        # Reports describe the post-move state and commit the sender's next proposal.
        generated: list[dict[str, Any]] = []
        channel_counts: dict[int, int] = defaultdict(int)
        next_required: list[str | None] = [None] * ROBOT_COUNT
        if t < HORIZON - 1:
            for sender in range(ROBOT_COUNT):
                row = rows.get(sender, {})
                receiver_value = row.get("announce_to")
                if receiver_value is None or not _integer(receiver_value) or receiver_value == sender:
                    continue
                receiver = int(receiver_value)
                if not 0 <= receiver < ROBOT_COUNT:
                    continue
                commitment = row.get("committed_move")
                committed_move = commitment if commitment in MOVES else "stay"
                channel = (t + sender + receiver) % CHANNEL_COUNT
                channel_counts[channel] += 1
                next_required[sender] = committed_move
                generated.append({
                    "sender": sender,
                    "receiver": receiver,
                    "channel": channel,
                    "position": [post_positions[sender][0], post_positions[sender][1]],
                    "committed_move": committed_move,
                    "task": active[sender],
                    "phase": phases[sender],
                    "loaded": loaded[sender],
                })
        packet_events: list[dict[str, Any]] = []
        for event in generated:
            sender, receiver, channel = event["sender"], event["receiver"], event["channel"]
            distance = abs(post_positions[sender][0] - post_positions[receiver][0]) + abs(
                post_positions[sender][1] - post_positions[receiver][1]
            )
            in_range = distance <= REPORT_RANGE
            delivered = _report_delivered(
                scenario_index, t, sender, receiver, distance, channel_counts[channel] == 1
            )
            arrival = t + 1 if delivered else None
            packet = {
                **event,
                "in_range": in_range,
                "delivered": delivered,
                "arrival": arrival,
            }
            packet_events.append(packet)
            report_count += 1
            if delivered:
                pending[arrival].append({**packet, "generation": t})

        trace.append({
            "t": t,
            "positions": [[point[0], point[1]] for point in positions],
            "post_positions": [[point[0], point[1]] for point in post_positions],
            "active_tasks": list(active),
            "phases": list(phases),
            "loaded": list(loaded),
            "blocked": list(blocked),
            "collision": list(collision),
            "report_ages": report_ages,
            "packets": sorted(packet_events, key=lambda item: (item["sender"], item["receiver"])),
        })
        positions = post_positions
        required_moves = next_required

    task_results: list[dict[str, Any]] = []
    for task_id, spec in enumerate(TASKS[scenario]):
        delivery = delivery_times[task_id]
        task_results.append({
            "task": task_id,
            "assigned_robot": assigned[task_id],
            "pickup_time": pickup_times[task_id],
            "delivery_time": delivery,
            "on_time": delivery is not None and delivery <= spec["deadline"],
        })
    return {
        "trace": trace,
        "tasks": task_results,
        "freshness_values": freshness_values,
        "collision_slots": collision_slots,
        "report_count": report_count,
        "claim_violations": claim_violations,
        "commitment_violations": commitment_violations,
    }


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"scenario", "decisions", "trace", "tasks"}


def _decision_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"t", "robots"}


def _robot_decision_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "robot", "move", "claim", "announce_to", "committed_move",
    }


def _trace_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "t", "positions", "post_positions", "active_tasks", "phases", "loaded",
        "blocked", "collision", "report_ages", "packets",
    }


def _packet_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "sender", "receiver", "channel", "in_range", "delivered", "arrival",
        "position", "committed_move", "task", "phase", "loaded",
    }


def _task_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "task", "assigned_robot", "pickup_time", "delivery_time", "on_time",
    }


def _decision_fields_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not _integer(value.get("t")) or not 0 <= value["t"] < HORIZON:
        return False
    robots = value.get("robots")
    if not isinstance(robots, list) or len(robots) != ROBOT_COUNT:
        return False
    seen: set[int] = set()
    for row in robots:
        if not _robot_decision_shape(row) or not _integer(row.get("robot")):
            return False
        robot = int(row["robot"])
        if robot in seen or not 0 <= robot < ROBOT_COUNT:
            return False
        seen.add(robot)
        if row.get("move") not in MOVES:
            return False
        claim = row.get("claim")
        if claim is not None and (not _integer(claim) or not 0 <= claim < TASK_COUNT):
            return False
        target = row.get("announce_to")
        if target is not None and (not _integer(target) or not 0 <= target < ROBOT_COUNT or target == robot):
            return False
        commitment = row.get("committed_move")
        if commitment is not None and commitment not in MOVES:
            return False
    return seen == set(range(ROBOT_COUNT))


def _decision_valid(value: Any) -> bool:
    return _decision_shape(value) and _decision_fields_valid(value)


def _packet_valid(value: Any) -> bool:
    if not _packet_shape(value):
        return False
    if not _integer(value.get("sender")) or not 0 <= value["sender"] < ROBOT_COUNT:
        return False
    if not _integer(value.get("receiver")) or not 0 <= value["receiver"] < ROBOT_COUNT:
        return False
    if value["sender"] == value["receiver"]:
        return False
    if not _integer(value.get("channel")) or not 0 <= value["channel"] < CHANNEL_COUNT:
        return False
    if not isinstance(value.get("in_range"), bool) or not isinstance(value.get("delivered"), bool):
        return False
    arrival = value.get("arrival")
    if arrival is not None and (not _integer(arrival) or not 1 <= arrival < HORIZON):
        return False
    if not _coordinate(value.get("position")):
        return False
    if value.get("committed_move") not in MOVES or value.get("phase") not in PHASES:
        return False
    task = value.get("task")
    if task is not None and (not _integer(task) or not 0 <= task < TASK_COUNT):
        return False
    return isinstance(value.get("loaded"), bool)


def _trace_state_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not _integer(value.get("t")) or not 0 <= value["t"] < HORIZON:
        return False
    for field in ("positions", "post_positions", "active_tasks", "phases", "loaded", "blocked", "collision", "report_ages"):
        if not isinstance(value.get(field), list):
            return False
    if len(value["positions"]) != ROBOT_COUNT or len(value["post_positions"]) != ROBOT_COUNT:
        return False
    if not all(_coordinate(point) for point in value["positions"] + value["post_positions"]):
        return False
    if len(value["active_tasks"]) != ROBOT_COUNT or any(
        task is not None and (not _integer(task) or not 0 <= task < TASK_COUNT)
        for task in value["active_tasks"]
    ):
        return False
    if len(value["phases"]) != ROBOT_COUNT or any(phase not in PHASES for phase in value["phases"]):
        return False
    for field in ("loaded", "blocked", "collision"):
        if len(value[field]) != ROBOT_COUNT or not all(isinstance(item, bool) for item in value[field]):
            return False
    ages = value["report_ages"]
    if len(ages) != ROBOT_COUNT or any(
        not isinstance(row, list) or len(row) != ROBOT_COUNT
        or any(age is not None and (not _integer(age) or age < 0 or age >= HORIZON) for age in row)
        for row in ages
    ):
        return False
    return True


def _trace_valid(value: Any) -> bool:
    if not _trace_shape(value) or not _trace_state_valid(value):
        return False
    packets = value.get("packets")
    return isinstance(packets, list) and all(_packet_valid(packet) for packet in packets)


def _task_fields_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    task = value.get("task")
    if not _integer(task) or not 0 <= task < TASK_COUNT:
        return False
    assigned = value.get("assigned_robot")
    if assigned is not None and (not _integer(assigned) or not 0 <= assigned < ROBOT_COUNT):
        return False
    for field in ("pickup_time", "delivery_time"):
        item = value.get(field)
        if item is not None and (not _integer(item) or not 1 <= item <= HORIZON):
            return False
    return isinstance(value.get("on_time"), bool)


def _task_valid(value: Any) -> bool:
    return _task_shape(value) and _task_fields_valid(value)


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for row in data["scenarios"]:
        if not isinstance(row, dict) or not isinstance(row.get("scenario"), str):
            continue
        name = row["scenario"]
        if name in indexed:
            continue
        decisions_rows = row.get("decisions") if isinstance(row.get("decisions"), list) else []
        trace_rows = row.get("trace") if isinstance(row.get("trace"), list) else []
        task_rows = row.get("tasks") if isinstance(row.get("tasks"), list) else []
        decisions: dict[int, Any] = {}
        traces: dict[int, Any] = {}
        tasks: dict[int, Any] = {}
        for item in decisions_rows:
            if isinstance(item, dict) and _integer(item.get("t")) and 0 <= item["t"] < HORIZON:
                decisions.setdefault(int(item["t"]), item)
        for item in trace_rows:
            if isinstance(item, dict) and _integer(item.get("t")) and 0 <= item["t"] < HORIZON:
                traces.setdefault(int(item["t"]), item)
        for item in task_rows:
            if isinstance(item, dict) and _integer(item.get("task")) and 0 <= item["task"] < TASK_COUNT:
                tasks.setdefault(int(item["task"]), item)
        indexed[name] = {
            "record": row,
            "decisions_rows": decisions_rows,
            "trace_rows": trace_rows,
            "task_rows": task_rows,
            "decisions": decisions,
            "traces": traces,
            "tasks": tasks,
        }
    return indexed


def _atom(check_id: str, rule: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{rule}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[check_id][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("atom shares must sum to one")
    for atom in atoms:
        if not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"atom share mismatch: {atom.id}")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root = parse_ok and isinstance(data, dict)
    root_ok = bool(root and set(data) == {"scenarios"})
    scenarios_ok = bool(root_ok and isinstance(data.get("scenarios"), list))
    return [
        _atom("result_json", "parse", bool(parse_ok), "strict JSON parsed" if parse_ok else "strict JSON parsing failed"),
        _atom("result_json", "root", root_ok, "root has exactly the scenarios key" if root_ok else "root must be an object with exactly scenarios"),
        _atom("result_json", "scenarios", scenarios_ok, "scenarios is an array" if scenarios_ok else "scenarios must be an array"),
    ]


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key.removeprefix("result_evidence."): True for key in ATOM_SHARES["result_evidence"]}
    details = {key: "" for key in checks}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if set(records) != set(SCENARIOS):
        fail("scenario_shape", "all canonical scenario records are required")
    for scenario in SCENARIOS:
        item = records.get(scenario)
        if item is None:
            continue
        record = item["record"]
        if not _scenario_shape(record):
            fail("scenario_shape", f"{scenario}: expected exactly scenario/decisions/trace/tasks")
        decisions_rows, trace_rows, task_rows = item["decisions_rows"], item["trace_rows"], item["task_rows"]
        if not isinstance(record.get("decisions"), list):
            fail("decision_shape", f"{scenario}.decisions: expected an array")
        elif len(decisions_rows) != HORIZON or not all(_decision_shape(row) for row in decisions_rows):
            fail(
                "decision_shape",
                f"{scenario}.decisions: expected exactly {HORIZON} rows with keys t and robots",
            )
        if isinstance(record.get("decisions"), list) and not all(
            _decision_fields_valid(row) for row in decisions_rows
        ):
            fail("decision_types", f"{scenario}.decisions: robot, move, claim or report fields have invalid types/domains")
        if not isinstance(record.get("trace"), list):
            fail("trace_shape", f"{scenario}.trace: expected an array")
        elif len(trace_rows) != HORIZON or not all(_trace_shape(row) for row in trace_rows):
            fail(
                "trace_shape",
                f"{scenario}.trace: expected exactly {HORIZON} rows with the declared keys",
            )
        if isinstance(record.get("trace"), list) and not all(
            _trace_state_valid(row) for row in trace_rows
        ):
            fail("trace_values", f"{scenario}.trace: state arrays, flags, ages or coordinates have invalid values")
        if isinstance(record.get("trace"), list) and any(
            not isinstance(row, dict) or not isinstance(row.get("packets"), list)
            or not all(_packet_shape(packet) and _packet_valid(packet) for packet in row.get("packets", []))
            for row in trace_rows
        ):
            fail("packet_evidence", f"{scenario}.trace.packets: every report must expose valid payload and outcome fields")
        if not isinstance(record.get("tasks"), list):
            fail("task_shape", f"{scenario}.tasks: expected an array")
        elif len(task_rows) != TASK_COUNT or not all(_task_shape(row) for row in task_rows):
            fail(
                "task_shape",
                f"{scenario}.tasks: expected exactly {TASK_COUNT} rows with the declared task result keys",
            )
        if isinstance(record.get("tasks"), list) and not all(
            _task_fields_valid(row) for row in task_rows
        ):
            fail("task_types", f"{scenario}.tasks: task, robot, time and on_time fields have invalid values")
    return [_atom("result_evidence", rule, checks[rule], details[rule] or "declared evidence fields passed") for rule in checks]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios: missing or not a list"]
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIOS}
    for index, row in enumerate(data["scenarios"]):
        name = row.get("scenario") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"scenarios[{index}]: unknown scenario key")
        else:
            by_name[name].append(row)

    def count_keys(record: dict[str, Any], field: str, key_name: str, expected: set[int], label: str) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.append(f"{label}: missing list")
            return
        counts = {key: 0 for key in expected}
        for index, row in enumerate(rows):
            key = row.get(key_name) if isinstance(row, dict) and _integer(row.get(key_name)) else None
            if key in counts:
                counts[key] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key")
        defects.extend(f"{label}: missing key {key}" for key, count in sorted(counts.items()) if count == 0)
        defects.extend(f"{label}: duplicate key {key}" for key, count in sorted(counts.items()) for _ in range(max(0, count - 1)))

    for name in SCENARIOS:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(record, "decisions", "t", set(range(HORIZON)), f"scenario {name}.decisions")
        count_keys(record, "trace", "t", set(range(HORIZON)), f"scenario {name}.trace")
        count_keys(record, "tasks", "task", set(range(TASK_COUNT)), f"scenario {name}.tasks")
    score = max(0.0, 1.0 - len(defects) / 8.0)
    return score, defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return f"failed_atoms=[canonical_coverage.defect]; defect_count={len(defects)}; {shown}; normalized={score:.9g}"


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    names = [key.removeprefix("decision_legality.") for key in ATOM_SHARES["decision_legality"]]
    checks = {name: True for name in names}
    details = {name: "" for name in names}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if set(records) != set(SCENARIOS):
        for rule in names:
            fail(rule, "all canonical scenario records are required")
    for scenario in SCENARIOS:
        item = records.get(scenario)
        if item is None:
            continue
        rows = item["decisions_rows"]
        if any(not isinstance(row, dict) or not _integer(row.get("t")) or not 0 <= row["t"] < HORIZON for row in rows):
            fail("time_domain", f"{scenario}.decisions: t must be an integer in 0..39")
        seen_times = [row.get("t") for row in rows if isinstance(row, dict) and _integer(row.get("t"))]
        if len(seen_times) != len(set(seen_times)) or len(seen_times) != len(rows):
            fail("time_domain", f"{scenario}.decisions: duplicate or malformed time keys")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("robots"), list) or len(row["robots"]) != ROBOT_COUNT:
                fail("robot_domain", f"{scenario}.decisions[{row.get('t') if isinstance(row, dict) else '?'}]: exactly three robot rows required")
                continue
            robot_ids = [r.get("robot") for r in row["robots"] if isinstance(r, dict)]
            if (
                len(robot_ids) != ROBOT_COUNT
                or any(not _integer(robot_id) for robot_id in robot_ids)
                or set(robot_ids) != set(range(ROBOT_COUNT))
            ):
                fail("robot_domain", f"{scenario}.decisions[{row.get('t')}].robots: expected unique IDs 0,1,2")
            for robot in row["robots"]:
                if not isinstance(robot, dict):
                    continue
                if robot.get("move") not in MOVES:
                    fail("move_domain", f"{scenario}.decisions[{row.get('t')}]: move must be one of {MOVES}")
                claim = robot.get("claim")
                if claim is not None and (not _integer(claim) or not 0 <= claim < TASK_COUNT):
                    fail("claim_domain", f"{scenario}.decisions[{row.get('t')}]: claim must be null or an integer in 0..7")
                target = robot.get("announce_to")
                rid = robot.get("robot")
                if target is not None and (
                    not _integer(target) or not 0 <= target < ROBOT_COUNT or target == rid
                ):
                    fail("report_target", f"{scenario}.decisions[{row.get('t')}]: announce_to must name a distinct robot or null")
                commitment = robot.get("committed_move")
                if commitment is not None and commitment not in MOVES:
                    fail("commitment_relation", f"{scenario}.decisions[{row.get('t')}]: committed_move must be null or one of {MOVES}")
                if (target is None) != (commitment is None):
                    fail("commitment_relation", f"{scenario}.decisions[{row.get('t')}]: announce_to and committed_move must both be null or both be set")
                if row.get("t") == HORIZON - 1 and (target is not None or commitment is not None):
                    fail("commitment_relation", f"{scenario}.decisions[39]: final-slot report and commitment must be null")

        decisions = item["decisions"]
        if set(decisions) == set(range(HORIZON)):
            for t in range(HORIZON - 1):
                current = _robot_rows(decisions[t])
                following = _robot_rows(decisions[t + 1])
                for robot in range(ROBOT_COUNT):
                    commitment = current.get(robot, {}).get("committed_move")
                    if commitment is not None and following.get(robot, {}).get("move") != commitment:
                        fail(
                            "commitment_execution",
                            f"{scenario}.decisions[{t + 1}].robot[{robot}].move must execute "
                            f"the slot-{t} commitment {commitment}",
                        )
                        break
                if not checks["commitment_execution"]:
                    break
        else:
            fail("commitment_execution", f"{scenario}: complete decision history is required to check next-slot commitments")

        if len(item["decisions_rows"]) == HORIZON and all(_decision_valid(row) for row in item["decisions_rows"]):
            replay = _replay_scenario(scenario, item["decisions"])
            if replay["claim_violations"]:
                fail("claim_eligibility", f"{scenario}: {replay['claim_violations'][0]}")
        else:
            fail("claim_eligibility", f"{scenario}: complete valid decisions are required to check claim eligibility")

        counts: list[int] = []
        for row in rows:
            robots = row.get("robots") if isinstance(row, dict) else []
            counts.append(sum(1 for robot in robots if isinstance(robot, dict) and robot.get("announce_to") is not None))
        if sum(counts) > REPORT_BUDGET:
            fail("report_budget", f"{scenario}: generated reports={sum(counts)} exceeds budget {REPORT_BUDGET}")
        for start in range(max(0, HORIZON - BURST_WINDOW + 1)):
            if sum(counts[start:start + BURST_WINDOW]) > BURST_LIMIT:
                fail("burst_limit", f"{scenario}: slots {start}..{start + BURST_WINDOW - 1} exceed {BURST_LIMIT} reports")
                break
    return [_atom("decision_legality", name, checks[name], details[name] or "decision constraint passed") for name in names]


def _replay_atoms(records: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    names = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    checks = {name: True for name in names}
    details = {name: "" for name in names}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if set(records) != set(SCENARIOS):
        for name in names:
            fail(name, "all canonical scenario records are required")
    for scenario in SCENARIOS:
        item = records.get(scenario)
        if item is None:
            continue
        decisions = item["decisions"]
        traces = item["traces"]
        if set(decisions) != set(range(HORIZON)) or any(not _decision_valid(decisions[t]) for t in range(HORIZON)):
            for name in names:
                fail(name, f"{scenario}.decisions: complete valid decisions required for replay")
            continue
        try:
            expected = _replay_scenario(scenario, decisions)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            for name in names:
                fail(name, f"{scenario}.decisions: replay input rejected: {type(exc).__name__}: {exc}")
            continue
        replayed[scenario] = expected
        if set(traces) != set(range(HORIZON)) or any(not _trace_valid(traces[t]) for t in range(HORIZON)):
            for name in names:
                fail(name, f"{scenario}.trace: complete valid trace required for replay")
            continue
        for t in range(HORIZON):
            actual = traces[t]
            wanted = expected["trace"][t]
            if actual.get("packets") != wanted["packets"]:
                fail("packet_events", f"{scenario}.trace[{t}].packets: expected={wanted['packets']}; actual={actual.get('packets')}")
            if actual.get("report_ages") != wanted["report_ages"]:
                fail("report_ages", f"{scenario}.trace[{t}].report_ages: expected={wanted['report_ages']}; actual={actual.get('report_ages')}")
            if actual.get("positions") != wanted["positions"]:
                fail("pre_positions", f"{scenario}.trace[{t}].positions: expected={wanted['positions']}; actual={actual.get('positions')}")
            if actual.get("post_positions") != wanted["post_positions"]:
                fail("post_positions", f"{scenario}.trace[{t}].post_positions: expected={wanted['post_positions']}; actual={actual.get('post_positions')}")
            if actual.get("blocked") != wanted["blocked"]:
                fail("blocked_flags", f"{scenario}.trace[{t}].blocked: expected={wanted['blocked']}; actual={actual.get('blocked')}")
            if actual.get("collision") != wanted["collision"]:
                fail("collision_flags", f"{scenario}.trace[{t}].collision: expected={wanted['collision']}; actual={actual.get('collision')}")
            if (
                actual.get("active_tasks") != wanted["active_tasks"]
                or actual.get("phases") != wanted["phases"]
                or actual.get("loaded") != wanted["loaded"]
            ):
                fail("robot_task_state", f"{scenario}.trace[{t}]: task phase/state differs from replay")
        if item["task_rows"]:
            if set(item["tasks"]) != set(range(TASK_COUNT)) or any(not _task_valid(item["tasks"][task]) for task in range(TASK_COUNT)):
                fail("task_records", f"{scenario}.tasks: complete valid task records required")
            else:
                for task_id in range(TASK_COUNT):
                    if item["tasks"][task_id] != expected["tasks"][task_id]:
                        fail("task_records", f"{scenario}.tasks[{task_id}]: expected={expected['tasks'][task_id]}; actual={item['tasks'][task_id]}")
        else:
            fail("task_records", f"{scenario}.tasks: task records missing")
    return [_atom("replay_validity", name, checks[name], details[name] or "fixed online scheduling replay comparison passed") for name in names], replayed


def _metrics(replayed: Mapping[str, dict[str, Any]], records: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    total_tasks = float(len(SCENARIOS) * TASK_COUNT)
    on_time = 0
    complete = 0
    collision_slots = 0
    report_count = 0
    freshness_sum = 0.0
    for scenario in SCENARIOS:
        result = replayed[scenario]
        on_time += sum(1 for row in result["tasks"] if row["on_time"])
        complete += sum(1 for row in result["tasks"] if row["delivery_time"] is not None)
        collision_slots += result["collision_slots"]
        report_count += result["report_count"]
        freshness_sum += sum(result["freshness_values"])
    metrics = {
        "deadline_service": on_time / total_tasks,
        "completion_rate": complete / total_tasks,
        "collision_free": max(0.0, 1.0 - collision_slots / float(len(SCENARIOS) * HORIZON)),
        "communication_efficiency": max(0.0, min(1.0, 1.0 - report_count / 120.0)),
        "intent_freshness": max(
            0.0,
            min(1.0, freshness_sum / float(len(SCENARIOS) * REPORT_BUDGET * FRESH_AGE)),
        ),
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _headings(text: str) -> dict[str, str]:
    required = {
        "Problem Formulation", "Method Design", "Experimental Setup", "Results",
        "Research Analysis", "Conclusion and Limitations",
    }
    lines = text.splitlines()
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match or match.group(2).strip() not in required:
            continue
        title = match.group(2).strip()
        level = len(match.group(1))
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            next_match = re.match(r"^(#{1,6})\s+.+", lines[cursor])
            if next_match and len(next_match.group(1)) <= level:
                break
            body.append(lines[cursor])
            cursor += 1
        found.setdefault(title, "\n".join(body).strip())
    return found


def _report_score(text: str | None) -> tuple[float, str]:
    required = (
        "Problem Formulation", "Method Design", "Experimental Setup", "Results",
        "Research Analysis", "Conclusion and Limitations",
    )
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    return score, f"missing_sections={missing}"


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


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
            state.official_message = f"results.json removal failed: {type(exc).__name__}: {exc}"
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = state.exit_code is None and (
                "timeout" in stderr.lower()
                or "timed out" in stderr.lower()
                or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS)
            )
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        if state.exit_code is None and state.elapsed_s >= 600.0 - EPS:
            state.timed_out = True
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok and state.exit_code == 0 and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS and state.result_file_is_new_regular
        )
        if not state.official_message:
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok else
                f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
            )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"results.json: strict parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose_atoms(result_atoms))
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            state.scores["result_json"] = 0.0
            state.diagnostics["result_json"] = f"{reason}; {state.diagnostics['result_json']}"
            for name in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[name] = 0.0
                state.diagnostics[name] = reason
            _CACHE[key] = state
            return state

        state.records = _index_records(state.data)
        evidence_atoms = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(evidence_atoms, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
        coverage_score, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage_score
        state.diagnostics["canonical_coverage"] = _coverage_diagnostic(coverage_score, defects)
        legality_atoms = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(legality_atoms, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
        replay_atoms, replayed = _replay_atoms(state.records)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
        gate_names = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        state.validity_gate = state.official_ok and all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in gate_names)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.records)
    except Exception as exc:  # noqa: BLE001
        state.official_ok = False
        state.official_message = state.official_message or f"grader state construction failed: {type(exc).__name__}: {exc}"
        for name in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
            state.scores.setdefault(name, 0.0)
            state.diagnostics.setdefault(name, state.official_message)
    finally:
        _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
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
        return 0.0, "report.md: expected a regular file"
    return _report_score(ws.read_text("report.md"))


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
    return state.scores.get("canonical_coverage", 0.0), state.diagnostics.get("canonical_coverage", "")


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("decision_legality", 0.0), state.diagnostics.get("decision_legality", "")


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("replay_validity", 0.0), state.diagnostics.get("replay_validity", "")


async def _metric_check(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{name}: validity gate not satisfied; metric not computed"
    value = state.metrics.get(name, 0.0)
    return value, f"{name}: independently replayed value={value:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "deadline_service")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "completion_rate")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "collision_free")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "communication_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "intent_freshness")
