from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

# Fixed task model.
HORIZON = 96
GRID_X = 20
GRID_Y = 14
INITIAL_ENERGY = 120.0
QUEUE_CAPACITY = 12
INITIAL_TTL = 4
SEND_BUDGET = 240
PROBE_BUDGET = 96
QUALITY_THRESHOLD = 0.65
EPS = 1e-6

NODE_IDS = ("G", "C", "R1", "R2")
NODE_RANK = {node: rank for rank, node in enumerate(NODE_IDS)}
MOVING_NODES = ("C", "R1", "R2")
GOAL_IDS = tuple(f"g{i}" for i in range(1, 7))
SCENARIO_NAMES = (
    "cross_corridor",
    "doorway_shift",
    "branching_mission",
    "moving_blocker",
    "relay_bottleneck",
)
MOVES = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "stay": (0, 0),
}
MOVE_COST = {"G": 0.0, "C": 1.0, "R1": 1.25, "R2": 1.25}
GOAL_RELEASES = (0, 16, 32, 48, 64, 80)
GOAL_BASE = (
    ((4, 3), 1.00, 24),
    ((15, 10), 1.10, 24),
    ((7, 11), 1.15, 25),
    ((13, 2), 1.20, 24),
    ((5, 8), 1.25, 23),
    ((17, 6), 1.30, 15),
)


def _wall(x: int, ys: Sequence[int]) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for y in ys)


SCENARIOS: dict[str, dict[str, Any]] = {
    "cross_corridor": {
        "phase": 1,
        "starts": {"G": (1, 7), "C": (18, 7), "R1": (6, 5), "R2": (12, 9)},
        "obstacles": _wall(9, tuple(range(0, 6)) + tuple(range(8, 14))),
        "windows": ((34, 46),),
    },
    "doorway_shift": {
        "phase": 4,
        "starts": {"G": (1, 2), "C": (18, 11), "R1": (6, 3), "R2": (13, 10)},
        "obstacles": _wall(8, tuple(range(0, 5)) + tuple(range(6, 14))),
        "windows": ((42, 54),),
    },
    "branching_mission": {
        "phase": 7,
        "starts": {"G": (1, 11), "C": (18, 2), "R1": (6, 10), "R2": (13, 3)},
        "obstacles": frozenset({(7, y) for y in range(3, 9)} | {(12, y) for y in range(5, 11)}),
        "windows": ((20, 32), (62, 74)),
    },
    "moving_blocker": {
        "phase": 10,
        "starts": {"G": (1, 7), "C": (18, 7), "R1": (5, 6), "R2": (14, 8)},
        "obstacles": _wall(10, tuple(range(0, 4)) + tuple(range(5, 14))),
        "windows": ((28, 40),),
    },
    "relay_bottleneck": {
        "phase": 13,
        "starts": {"G": (1, 6), "C": (18, 6), "R1": (6, 4), "R2": (13, 9)},
        "obstacles": _wall(9, tuple(range(0, 3)) + tuple(range(4, 11)) + (12, 13)),
        "windows": ((48, 66),),
    },
}


# Atom identities and fixed task-specific shares. Every group sums to one.
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 1.0 / 3.0,
        "result_json.root": 1.0 / 3.0,
        "result_json.scenarios": 1.0 / 3.0,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 0.10,
        "result_evidence.slot_request_types": 0.15,
        "result_evidence.robot_types": 0.15,
        "result_evidence.goal_types": 0.10,
        "result_evidence.packet_types": 0.15,
        "result_evidence.hop_types": 0.15,
        "result_evidence.probe_types": 0.10,
        "result_evidence.mission_types": 0.10,
    },
    "decision_legality": {
        "decision_legality.request_domains": 0.20,
        "decision_legality.motion": 0.20,
        "decision_legality.active_goal": 0.20,
        "decision_legality.fifo_routing": 0.20,
        "decision_legality.budgets": 0.20,
    },
    "replay_validity": {
        "replay_validity.link_attempts": 1.0 / 6.0,
        "replay_validity.packet_queues": 1.0 / 6.0,
        "replay_validity.robot_motion": 1.0 / 6.0,
        "replay_validity.goal_events": 1.0 / 6.0,
        "replay_validity.probe_feedback": 1.0 / 6.0,
        "replay_validity.mission_terminal": 1.0 / 6.0,
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
        self.scenarios: dict[str, dict[str, Any]] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nullable_integer(value: Any, low: int = 0, high: int = HORIZON) -> bool:
    return value is None or (_integer(value) and low <= value <= high)


def _nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _close(actual: Any, expected: Any) -> bool:
    return (
        _finite_number(actual)
        and _finite_number(expected)
        and abs(float(actual) - float(expected)) <= EPS
    )


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


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "scenario",
        "slots",
        "robots",
        "goals",
        "packets",
        "hops",
        "probes",
        "mission",
    }


def _slot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "slot",
        "requests",
        "arbitrated_request_id",
        "active_command_goal",
        "state",
    }


def _request_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "node",
        "move",
        "kind",
        "packet_id",
        "next_hop",
    }


def _slot_state_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "interference",
        "budget_used",
        "collision_nodes",
        "client_position",
    }


def _robot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "node",
        "positions",
        "energy",
        "queue_lengths",
        "local_feedback",
    }


def _goal_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "goal_id",
        "release_slot",
        "target_x",
        "target_y",
        "value",
        "deadline",
        "confirmation_window_end",
        "command_packet_id",
        "command_arrival_slot",
        "completion_slot",
        "progress_packet_id",
        "confirmation_slot",
        "status",
    }


def _packet_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "packet_id",
        "class",
        "goal_id",
        "source",
        "created_slot",
        "ttl",
        "initial_node",
        "final_node",
        "outcome",
    }


def _hop_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "hop_id",
        "packet_id",
        "attempt_no",
        "retransmission",
        "from",
        "to",
        "request_slot",
        "arrival_slot",
        "quality",
        "success",
        "reason",
    }


def _probe_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "probe_id",
        "node",
        "request_slot",
        "feedback_slot",
        "neighbors",
        "qualities",
    }


def _mission_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "send_attempts",
        "useful_deliveries",
        "queue_wait_sum",
        "probe_count",
        "retransmissions",
        "collision_slots",
        "motion_cost",
        "energy_used",
        "completed_value",
        "terminal_arrival_packet_ids",
    }


def _request_types(value: Any) -> bool:
    return (
        _request_shape(value)
        and isinstance(value["node"], str)
        and isinstance(value["move"], str)
        and isinstance(value["kind"], str)
        and _nullable_string(value["packet_id"])
        and _nullable_string(value["next_hop"])
    )


def _slot_valid(value: Any) -> bool:
    if not _slot_shape(value) or not _integer(value["slot"]):
        return False
    requests = value["requests"]
    state = value["state"]
    return (
        isinstance(requests, list)
        and len(requests) == len(NODE_IDS)
        and all(_request_types(row) for row in requests)
        and _nullable_string(value["arbitrated_request_id"])
        and _nullable_string(value["active_command_goal"])
        and _slot_state_shape(state)
        and isinstance(state["interference"], bool)
        and _integer(state["budget_used"])
        and state["budget_used"] >= 0
        and isinstance(state["collision_nodes"], list)
        and all(isinstance(node, str) for node in state["collision_nodes"])
        and isinstance(state["client_position"], list)
        and len(state["client_position"]) == 2
        and all(_integer(coordinate) for coordinate in state["client_position"])
    )


def _position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_integer(coordinate) for coordinate in value)
        and 0 <= value[0] < GRID_X
        and 0 <= value[1] < GRID_Y
    )


def _robot_valid(value: Any) -> bool:
    return (
        _robot_shape(value)
        and isinstance(value["node"], str)
        and isinstance(value["positions"], list)
        and len(value["positions"]) == HORIZON + 1
        and all(_position(position) for position in value["positions"])
        and isinstance(value["energy"], list)
        and len(value["energy"]) == HORIZON + 1
        and all(
            _finite_number(energy) and -EPS <= float(energy) <= INITIAL_ENERGY + EPS
            for energy in value["energy"]
        )
        and isinstance(value["queue_lengths"], list)
        and len(value["queue_lengths"]) == HORIZON
        and all(
            _integer(length) and 0 <= length <= QUEUE_CAPACITY for length in value["queue_lengths"]
        )
        and isinstance(value["local_feedback"], list)
        and len(value["local_feedback"]) == HORIZON
        and all(
            isinstance(ids, list) and all(isinstance(probe_id, str) for probe_id in ids)
            for ids in value["local_feedback"]
        )
    )


def _goal_valid(value: Any) -> bool:
    return (
        _goal_shape(value)
        and isinstance(value["goal_id"], str)
        and all(
            _integer(value[key])
            for key in (
                "release_slot",
                "target_x",
                "target_y",
                "deadline",
                "confirmation_window_end",
            )
        )
        and _finite_number(value["value"])
        and float(value["value"]) >= 0.0
        and isinstance(value["command_packet_id"], str)
        and _nullable_integer(value["command_arrival_slot"])
        and _nullable_integer(value["completion_slot"], 0, HORIZON - 1)
        and _nullable_string(value["progress_packet_id"])
        and _nullable_integer(value["confirmation_slot"])
        and value["status"] in {"commanded", "completed", "confirmed", "expired"}
    )


def _packet_valid(value: Any) -> bool:
    return (
        _packet_shape(value)
        and all(
            isinstance(value[key], str)
            for key in ("packet_id", "class", "goal_id", "source", "initial_node", "final_node")
        )
        and value["class"] in {"command", "progress"}
        and _integer(value["created_slot"])
        and 0 <= value["created_slot"] < HORIZON
        and _integer(value["ttl"])
        and 0 <= value["ttl"] <= INITIAL_TTL
        and value["outcome"] in {"delivered", "expired", "dropped", "unfinished"}
    )


def _hop_valid(value: Any) -> bool:
    return (
        _hop_shape(value)
        and all(
            isinstance(value[key], str) for key in ("hop_id", "packet_id", "from", "to", "reason")
        )
        and _integer(value["attempt_no"])
        and value["attempt_no"] >= 0
        and isinstance(value["retransmission"], bool)
        and _integer(value["request_slot"])
        and 0 <= value["request_slot"] < HORIZON
        and _nullable_integer(value["arrival_slot"], 1, HORIZON)
        and _finite_number(value["quality"])
        and 0.0 <= float(value["quality"]) <= 1.0
        and isinstance(value["success"], bool)
        and value["reason"] in {"delivered", "quality", "loss", "ttl", "not_selected"}
    )


def _probe_valid(value: Any) -> bool:
    return (
        _probe_shape(value)
        and isinstance(value["probe_id"], str)
        and isinstance(value["node"], str)
        and _integer(value["request_slot"])
        and 0 <= value["request_slot"] < HORIZON
        and _integer(value["feedback_slot"])
        and 1 <= value["feedback_slot"] <= HORIZON
        and isinstance(value["neighbors"], list)
        and len(value["neighbors"]) == len(NODE_IDS) - 1
        and all(isinstance(node, str) for node in value["neighbors"])
        and isinstance(value["qualities"], list)
        and len(value["qualities"]) == len(NODE_IDS) - 1
        and all(
            _finite_number(quality) and 0.0 <= float(quality) <= 1.0
            for quality in value["qualities"]
        )
    )


def _mission_valid(value: Any) -> bool:
    if not _mission_shape(value):
        return False
    integer_fields = (
        "send_attempts",
        "useful_deliveries",
        "queue_wait_sum",
        "probe_count",
        "retransmissions",
        "collision_slots",
        "motion_cost",
    )
    return (
        all(_integer(value[key]) and value[key] >= 0 for key in integer_fields)
        and _finite_number(value["energy_used"])
        and float(value["energy_used"]) >= 0.0
        and _finite_number(value["completed_value"])
        and float(value["completed_value"]) >= 0.0
        and isinstance(value["terminal_arrival_packet_ids"], list)
        and all(isinstance(packet_id, str) for packet_id in value["terminal_arrival_packet_ids"])
    )


def _ids_unique(rows: Sequence[Any], key: str) -> bool:
    ids = [row.get(key) for row in rows if isinstance(row, dict)]
    return (
        len(ids) == len(rows)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(set(ids))
    )


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for root_index, record in enumerate(data["scenarios"]):
        if not isinstance(record, dict) or not isinstance(record.get("scenario"), str):
            continue
        name = record["scenario"]
        if name in indexed:
            continue

        def rows_and_index(field: str, key: str) -> tuple[list[Any], dict[Any, Any]]:
            rows = record.get(field) if isinstance(record.get(field), list) else []
            row_index: dict[Any, Any] = {}
            for row in rows:
                if isinstance(row, dict) and key in row and isinstance(row[key], Hashable):
                    row_index.setdefault(row[key], row)
            return rows, row_index

        slot_rows, slots = rows_and_index("slots", "slot")
        robot_rows, robots = rows_and_index("robots", "node")
        goal_rows, goals = rows_and_index("goals", "goal_id")
        indexed[name] = {
            "record": record,
            "root_index": root_index,
            "slot_rows": slot_rows,
            "slots": slots,
            "robot_rows": robot_rows,
            "robots": robots,
            "goal_rows": goal_rows,
            "goals": goals,
        }
    return indexed


def _in_grid(cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < GRID_X and 0 <= cell[1] < GRID_Y


def _open_cell(name: str, cell: tuple[int, int]) -> bool:
    return _in_grid(cell) and cell not in SCENARIOS[name]["obstacles"]


@cache
def _shortest_distance(name: str, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    if not _open_cell(name, start) or not _open_cell(name, goal):
        return None
    if start == goal:
        return 0
    frontier: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    for_cell = ((0, 1), (1, 0), (0, -1), (-1, 0))
    while frontier:
        cell, distance = frontier.popleft()
        for dx, dy in for_cell:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if neighbor in seen or not _open_cell(name, neighbor):
                continue
            if neighbor == goal:
                return distance + 1
            seen.add(neighbor)
            frontier.append((neighbor, distance + 1))
    return None


def _interference(name: str, slot: int) -> bool:
    phase = int(SCENARIOS[name]["phase"])
    in_window = any(start <= slot < end for start, end in SCENARIOS[name]["windows"])
    return in_window or (5 * slot + 2 * phase) % 29 < 3


def _link_quality(
    name: str,
    slot: int,
    sender: str,
    receiver: str,
    positions: Mapping[str, tuple[int, int]],
) -> float:
    distance = _shortest_distance(name, positions[sender], positions[receiver])
    if distance is None:
        return 0.0
    interference = 0.15 if _interference(name, slot) else 0.0
    return max(0.0, min(1.0, 1.0 - 0.045 * distance - 0.18 * interference))


def _physical_success(name: str, slot: int, sender: str, receiver: str, quality: float) -> bool:
    phase = int(SCENARIOS[name]["phase"])
    loss_term = (13 * slot + 3 * NODE_RANK[sender] + 5 * NODE_RANK[receiver] + phase) % 17
    return quality + EPS >= QUALITY_THRESHOLD and loss_term > 1


def _goal_models(name: str) -> dict[str, dict[str, Any]]:
    phase = int(SCENARIOS[name]["phase"])
    delta_x = (phase // 3) % 3 - 1
    result: dict[str, dict[str, Any]] = {}
    for index, goal_id in enumerate(GOAL_IDS):
        base, value, offset = GOAL_BASE[index]
        shifted = (base[0] + delta_x, base[1])
        if not _open_cell(name, shifted):
            for move in ("N", "E", "S", "W"):
                dx, dy = MOVES[move]
                candidate = (shifted[0] + dx, shifted[1] + dy)
                if _open_cell(name, candidate):
                    shifted = candidate
                    break
        release = GOAL_RELEASES[index]
        deadline = release + offset
        result[goal_id] = {
            "goal_id": goal_id,
            "release_slot": release,
            "target_x": shifted[0],
            "target_y": shifted[1],
            "value": value,
            "deadline": deadline,
            "confirmation_window_end": min(deadline + 8, HORIZON),
            "command_packet_id": f"cmd-{goal_id}",
            "command_arrival_slot": None,
            "completion_slot": None,
            "progress_packet_id": None,
            "confirmation_slot": None,
            "status": "expired",
        }
    return result


def _packet_public(
    packet: Mapping[str, Any], goals: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    goal = goals[str(packet["goal_id"])]
    return {
        "packet_id": packet["packet_id"],
        "class": packet["class"],
        "goal_id": packet["goal_id"],
        "created_slot": packet["created_slot"],
        "ttl": packet["ttl"],
        "entry_slot": packet.get("entry_slot"),
        "target": [goal["target_x"], goal["target_y"]],
        "value": goal["value"],
        "deadline": goal["deadline"],
        "confirmation_window_end": goal["confirmation_window_end"],
        "final_node": packet["final_node"],
    }


def _default_request(node: str) -> dict[str, Any]:
    return {
        "node": node,
        "move": "stay",
        "kind": "none",
        "packet_id": None,
        "next_hop": None,
    }


def _decision_map(item: Mapping[str, Any]) -> dict[int, Mapping[str, Any]] | None:
    slots = item.get("slots")
    if not isinstance(slots, dict):
        return None
    if not set(range(HORIZON)).issubset(slots):
        return None
    decisions: dict[int, Mapping[str, Any]] = {}
    for slot in range(HORIZON):
        row = slots[slot]
        if not isinstance(row, dict) or row.get("slot") != slot:
            return None
        requests = row.get("requests")
        if (
            not isinstance(requests, list)
            or len(requests) != len(NODE_IDS)
            or not all(_request_types(request) for request in requests)
            or [request.get("node") for request in requests] != list(NODE_IDS)
            or not _nullable_string(row.get("active_command_goal"))
        ):
            return None
        decisions[slot] = {
            "requests": requests,
            "active_command_goal": row.get("active_command_goal"),
        }
    return decisions


DecisionProvider = Callable[[int, Mapping[str, Any]], Mapping[str, Any]]


def _resolve_collisions(
    starts: Mapping[str, tuple[int, int]], moves: Mapping[str, str]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    destinations = {
        node: (
            starts[node][0] + MOVES[moves[node]][0],
            starts[node][1] + MOVES[moves[node]][1],
        )
        for node in NODE_IDS
    }
    collision: set[str] = set()
    while True:
        implicated: set[str] = set()
        by_destination: dict[tuple[int, int], list[str]] = {}
        for node in NODE_IDS:
            by_destination.setdefault(destinations[node], []).append(node)
        for group in by_destination.values():
            if len(group) > 1:
                implicated.update(group)
        for left_index, left in enumerate(NODE_IDS):
            for right in NODE_IDS[left_index + 1 :]:
                if (
                    destinations[left] == starts[right]
                    and destinations[right] == starts[left]
                    and starts[left] != starts[right]
                ):
                    implicated.update((left, right))
        stopped = {node for node in NODE_IDS if moves[node] == "stay" or node in collision}
        for moving in NODE_IDS:
            if destinations[moving] == starts[moving]:
                continue
            for occupying in stopped:
                if moving != occupying and destinations[moving] == starts[occupying]:
                    implicated.update((moving, occupying))
        newly_implicated = implicated - collision
        if not newly_implicated:
            break
        collision.update(newly_implicated)
        for node in newly_implicated:
            destinations[node] = starts[node]
    for node in collision:
        destinations[node] = starts[node]
    return destinations, [node for node in NODE_IDS if node in collision]


def _simulate_scenario(
    name: str,
    decisions: Mapping[int, Mapping[str, Any]] | None = None,
    decision_provider: DecisionProvider | None = None,
) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    if (decisions is None) == (decision_provider is None):
        raise ValueError("provide exactly one of decisions or decision_provider")

    goals = _goal_models(name)
    positions = {node: tuple(SCENARIOS[name]["starts"][node]) for node in NODE_IDS}
    energy = {node: INITIAL_ENERGY for node in NODE_IDS}
    queues: dict[str, list[str]] = {node: [] for node in NODE_IDS}
    packets: dict[str, dict[str, Any]] = {}
    arrivals: dict[int, list[tuple[str, str]]] = {}
    feedback_due: dict[int, dict[str, list[dict[str, Any]]]] = {}
    attempt_history: dict[str, list[dict[str, Any]]] = {}
    delivered_to_role: dict[str, list[str]] = {node: [] for node in NODE_IDS}

    slots_out: list[dict[str, Any]] = []
    hops_out: list[dict[str, Any]] = []
    probes_out: list[dict[str, Any]] = []
    position_tracks = {node: [[*positions[node]]] for node in NODE_IDS}
    energy_tracks = {node: [energy[node]] for node in NODE_IDS}
    queue_tracks: dict[str, list[int]] = {node: [] for node in NODE_IDS}
    feedback_tracks: dict[str, list[list[str]]] = {node: [] for node in NODE_IDS}
    terminal_arrivals: list[str] = []

    send_requested = 0
    probe_requested = 0
    queue_wait_sum = 0
    collision_slots = 0
    motion_cost = 0

    legality: dict[str, list[Any]] = {
        rule.removeprefix("decision_legality."): [True, ""]
        for rule in ATOM_SHARES["decision_legality"]
    }

    def fail(rule: str, detail: str) -> None:
        if legality[rule][0]:
            legality[rule] = [False, detail]

    def packet_expired(packet: Mapping[str, Any], slot: int) -> bool:
        goal = goals[str(packet["goal_id"])]
        if packet["class"] == "command":
            return slot >= int(goal["deadline"])
        return slot >= int(goal["confirmation_window_end"])

    def enqueue(packet: dict[str, Any], node: str, slot: int) -> bool:
        if len(queues[node]) >= QUEUE_CAPACITY:
            packet["outcome"] = "dropped"
            packet["current_node"] = None
            packet["entry_slot"] = None
            return False
        packet["current_node"] = node
        packet["entry_slot"] = slot
        queues[node].append(str(packet["packet_id"]))
        queues[node].sort(key=lambda packet_id: (int(packets[packet_id]["entry_slot"]), packet_id))
        return True

    def create_packet(packet_class: str, goal_id: str, slot: int) -> dict[str, Any]:
        command = packet_class == "command"
        packet_id = f"cmd-{goal_id}" if command else f"prog-{goal_id}"
        source = "G" if command else "C"
        packet = {
            "packet_id": packet_id,
            "class": packet_class,
            "goal_id": goal_id,
            "source": source,
            "created_slot": slot,
            "ttl": INITIAL_TTL,
            "initial_node": source,
            "final_node": "C" if command else "G",
            "outcome": "unfinished",
            "current_node": None,
            "entry_slot": None,
        }
        packets[packet_id] = packet
        attempt_history[packet_id] = []
        enqueue(packet, source, slot)
        return packet

    def process_arrivals(slot: int) -> None:
        for packet_id, receiver in arrivals.pop(slot, []):
            if slot == HORIZON:
                terminal_arrivals.append(packet_id)
            packet = packets[packet_id]
            packet["ttl"] = int(packet["ttl"]) - 1
            if receiver == packet["final_node"]:
                packet["outcome"] = "delivered"
                packet["current_node"] = None
                packet["entry_slot"] = None
                delivered_to_role[receiver].append(packet_id)
                goal = goals[str(packet["goal_id"])]
                if packet["class"] == "command":
                    if goal["command_arrival_slot"] is None:
                        goal["command_arrival_slot"] = slot
                elif (
                    goal["completion_slot"] is not None
                    and int(goal["deadline"]) < slot <= int(goal["confirmation_window_end"])
                    and goal["confirmation_slot"] is None
                ):
                    goal["confirmation_slot"] = slot
                continue
            if int(packet["ttl"]) == 0 or packet_expired(packet, slot):
                packet["outcome"] = "expired"
                packet["current_node"] = None
                packet["entry_slot"] = None
                continue
            enqueue(packet, receiver, slot)

    for slot in range(HORIZON):
        process_arrivals(slot)
        slot_feedback = feedback_due.pop(slot, {})
        visible_feedback: dict[str, list[dict[str, Any]]] = {}
        for node in NODE_IDS:
            rows = sorted(slot_feedback.get(node, []), key=lambda row: row["probe_id"])
            visible_feedback[node] = rows
            feedback_tracks[node].append([str(row["probe_id"]) for row in rows])

        for node in NODE_IDS:
            while queues[node]:
                packet = packets[queues[node][0]]
                if not packet_expired(packet, slot):
                    break
                queues[node].pop(0)
                packet["outcome"] = "expired"
                packet["current_node"] = None
                packet["entry_slot"] = None

        if slot in GOAL_RELEASES:
            goal_id = GOAL_IDS[GOAL_RELEASES.index(slot)]
            create_packet("command", goal_id, slot)

        static_view = {
            "grid": [GRID_X, GRID_Y],
            "obstacles": [list(cell) for cell in sorted(SCENARIOS[name]["obstacles"])],
            "initial_positions": {node: list(SCENARIOS[name]["starts"][node]) for node in NODE_IDS},
            "goal_base": [
                {
                    "goal_id": GOAL_IDS[index],
                    "base_target": list(GOAL_BASE[index][0]),
                    "value": GOAL_BASE[index][1],
                    "release_slot": GOAL_RELEASES[index],
                    "deadline_offset": GOAL_BASE[index][2],
                }
                for index in range(len(GOAL_IDS))
            ],
        }
        role_views: dict[str, dict[str, Any]] = {}
        for node in NODE_IDS:
            queue_view = [_packet_public(packets[packet_id], goals) for packet_id in queues[node]]
            role_views[node] = {
                "node": node,
                "slot": slot,
                "position": list(positions[node]),
                "energy": energy[node],
                "queue": queue_view,
                "probe_feedback": visible_feedback[node],
                "delivered_packets": [
                    _packet_public(packets[packet_id], goals)
                    for packet_id in delivered_to_role[node]
                ],
                "released_goals": [
                    {
                        "goal_id": goal_id,
                        "target": [goals[goal_id]["target_x"], goals[goal_id]["target_y"]],
                        "value": goals[goal_id]["value"],
                        "deadline": goals[goal_id]["deadline"],
                    }
                    for goal_id in GOAL_IDS
                    if node == "G" and int(goals[goal_id]["release_slot"]) <= slot
                ],
                "completed_goals": [
                    goal_id for goal_id in GOAL_IDS if goals[goal_id]["completion_slot"] is not None
                ]
                if node == "C"
                else [],
                "static": static_view,
            }
        views = {"roles": role_views}

        if decision_provider is not None:
            supplied = decision_provider(slot, views)
        else:
            supplied = decisions.get(slot, {}) if decisions is not None else {}
        if not isinstance(supplied, Mapping):
            supplied = {}
            fail("request_domains", f"{name}.slots[{slot}]: decision provider returned no mapping")

        supplied_requests = supplied.get("requests")
        if isinstance(supplied_requests, Mapping):
            request_rows = [
                supplied_requests.get(node, _default_request(node)) for node in NODE_IDS
            ]
        elif isinstance(supplied_requests, list):
            request_rows = list(supplied_requests)
        else:
            request_rows = [_default_request(node) for node in NODE_IDS]
            fail("request_domains", f"{name}.slots[{slot}].requests: expected four requests")

        if len(request_rows) != len(NODE_IDS):
            fail(
                "request_domains",
                f"{name}.slots[{slot}].requests: expected length=4; actual={len(request_rows)}",
            )
        normalized_rows: list[dict[str, Any]] = []
        requests_by_node: dict[str, dict[str, Any]] = {}
        for node_index, node in enumerate(NODE_IDS):
            raw = (
                request_rows[node_index]
                if node_index < len(request_rows)
                else _default_request(node)
            )
            if not isinstance(raw, Mapping):
                raw = _default_request(node)
                fail(
                    "request_domains",
                    f"{name}.slots[{slot}].requests[{node_index}]: expected request object",
                )
            row = {
                "node": raw.get("node"),
                "move": raw.get("move"),
                "kind": raw.get("kind"),
                "packet_id": raw.get("packet_id"),
                "next_hop": raw.get("next_hop"),
            }
            normalized_rows.append(row)
            requests_by_node[node] = row
            if row["node"] != node:
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].request[{node_index}].node: "
                    f"expected={node!r}; actual={row['node']!r}",
                )

        active_raw = supplied.get("active_command_goal")
        active_goal: str | None = None
        if active_raw is not None and not isinstance(active_raw, str):
            fail(
                "active_goal",
                f"{name}.slots[{slot}].active_command_goal: expected string or null",
            )
        elif active_raw is not None:
            if active_raw not in goals:
                fail(
                    "active_goal",
                    f"{name}.slots[{slot}].active_command_goal: unknown goal={active_raw!r}",
                )
            else:
                goal = goals[active_raw]
                legal_active = (
                    goal["command_arrival_slot"] is not None
                    and int(goal["command_arrival_slot"]) <= slot
                    and goal["completion_slot"] is None
                    and slot <= int(goal["deadline"])
                )
                if legal_active:
                    active_goal = active_raw
                else:
                    fail(
                        "active_goal",
                        f"{name}.slot[{slot}].active_command_goal: "
                        f"goal {active_raw!r} is not currently executable",
                    )

        move_names: dict[str, str] = {}
        eligible: list[dict[str, Any]] = []
        for node in NODE_IDS:
            row = requests_by_node[node]
            move = row["move"]
            kind = row["kind"]
            packet_id = row["packet_id"]
            next_hop = row["next_hop"]

            if move not in MOVES:
                fail(
                    "request_domains",
                    f"{name}.slots[{slot}].requests[{node}].move: invalid enum={move!r}",
                )
                move = "stay"
            if kind not in {"send", "probe", "none"}:
                fail(
                    "request_domains",
                    f"{name}.slots[{slot}].requests[{node}].kind: invalid enum={kind!r}",
                )
                kind = "none"
            pairing_ok = (
                kind == "send" and isinstance(packet_id, str) and isinstance(next_hop, str)
            ) or (kind in {"probe", "none"} and packet_id is None and next_hop is None)
            if not pairing_ok:
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].request[{node}]: "
                    f"packet_id/next_hop do not match kind={kind!r}",
                )
                kind = "none"

            dx, dy = MOVES[move]
            destination = (positions[node][0] + dx, positions[node][1] + dy)
            motion_ok = True
            if node == "G" and move != "stay":
                motion_ok = False
            elif move != "stay" and not _open_cell(name, destination):
                motion_ok = False
            elif move != "stay" and energy[node] + EPS < MOVE_COST[node]:
                motion_ok = False
            if not motion_ok:
                fail(
                    "motion",
                    f"{name}.slot[{slot}].request[{node}].move: {move!r} "
                    "violates station, map, or energy boundary",
                )
                move = "stay"
            move_names[node] = move

            raw_send = row["kind"] == "send"
            raw_probe = row["kind"] == "probe"
            if raw_send:
                send_requested += 1
                if send_requested > SEND_BUDGET:
                    fail(
                        "budgets",
                        f"{name}.slot[{slot}].request[{node}]: send request "
                        f"{send_requested} exceeds {SEND_BUDGET}",
                    )
                    kind = "none"
            if raw_probe:
                probe_requested += 1
                if probe_requested > PROBE_BUDGET:
                    fail(
                        "budgets",
                        f"{name}.slot[{slot}].request[{node}]: probe request "
                        f"{probe_requested} exceeds {PROBE_BUDGET}",
                    )
                    kind = "none"

            if kind == "send":
                head = queues[node][0] if queues[node] else None
                if packet_id != head or packet_id not in packets:
                    fail(
                        "fifo_routing",
                        f"{name}.slot[{slot}].request[{node}]: expected FIFO "
                        f"head={head!r}; actual={packet_id!r}",
                    )
                    kind = "none"
                elif next_hop not in NODE_IDS or next_hop == node:
                    fail(
                        "fifo_routing",
                        f"{name}.slot[{slot}].request[{node}].next_hop: "
                        f"expected a different node; actual={next_hop!r}",
                    )
                    kind = "none"
                else:
                    packet = packets[str(packet_id)]
                    history = attempt_history[str(packet_id)]
                    attempt_no = len(history)
                    quality = _link_quality(name, slot, node, str(next_hop), positions)
                    retransmission = bool(
                        history and not bool(history[-1]["success"]) and history[-1]["from"] == node
                    )
                    eligible.append(
                        {
                            "request_id": f"req-{slot:02d}-{node}",
                            "packet": packet,
                            "packet_id": packet_id,
                            "attempt_no": attempt_no,
                            "retransmission": retransmission,
                            "from": node,
                            "to": next_hop,
                            "request_slot": slot,
                            "quality": quality,
                        }
                    )
            elif kind == "probe":
                neighbors = [other for other in NODE_IDS if other != node]
                probe = {
                    "probe_id": f"probe-{slot:02d}-{node}",
                    "node": node,
                    "request_slot": slot,
                    "feedback_slot": slot + 1,
                    "neighbors": neighbors,
                    "qualities": [
                        _link_quality(name, slot, node, other, positions) for other in neighbors
                    ],
                }
                probes_out.append(probe)
                feedback_due.setdefault(slot + 1, {}).setdefault(node, []).append(probe)

        selected: dict[str, Any] | None = None
        if eligible:
            selected = min(
                eligible,
                key=lambda attempt: (
                    0 if attempt["packet"]["class"] == "command" else 1,
                    int(attempt["packet"]["created_slot"]),
                    NODE_RANK[str(attempt["from"])],
                    str(attempt["packet_id"]),
                    int(attempt["attempt_no"]),
                ),
            )
        arbitrated_request_id = selected["request_id"] if selected is not None else None

        for attempt in eligible:
            success = False
            arrival_slot: int | None = None
            reason = "not_selected"
            if attempt is selected:
                packet = attempt["packet"]
                queue_wait_sum += slot - int(packet["entry_slot"])
                if float(attempt["quality"]) + EPS < QUALITY_THRESHOLD:
                    reason = "quality"
                elif not _physical_success(
                    name,
                    slot,
                    str(attempt["from"]),
                    str(attempt["to"]),
                    float(attempt["quality"]),
                ):
                    reason = "loss"
                else:
                    success = True
                    arrival_slot = slot + 1
                    reason = (
                        "ttl"
                        if int(packet["ttl"]) - 1 == 0 and attempt["to"] != packet["final_node"]
                        else "delivered"
                    )
                    sender = str(attempt["from"])
                    packet_id = str(attempt["packet_id"])
                    if queues[sender] and queues[sender][0] == packet_id:
                        queues[sender].pop(0)
                    packet["current_node"] = None
                    packet["entry_slot"] = None
                    arrivals.setdefault(slot + 1, []).append((packet_id, str(attempt["to"])))
            hop = {
                "hop_id": f"{attempt['packet_id']}#a{attempt['attempt_no']}",
                "packet_id": attempt["packet_id"],
                "attempt_no": attempt["attempt_no"],
                "retransmission": attempt["retransmission"],
                "from": attempt["from"],
                "to": attempt["to"],
                "request_slot": slot,
                "arrival_slot": arrival_slot,
                "quality": attempt["quality"],
                "success": success,
                "reason": reason,
            }
            hops_out.append(hop)
            attempt_history[str(attempt["packet_id"])].append(hop)

        starts = dict(positions)
        destinations, collision_nodes = _resolve_collisions(starts, move_names)
        collision = set(collision_nodes)
        if collision_nodes:
            collision_slots += 1
        for node in NODE_IDS:
            if node in collision:
                destinations[node] = starts[node]
            moved = destinations[node] != starts[node]
            positions[node] = destinations[node]
            if moved:
                energy[node] -= MOVE_COST[node]
                if node in MOVING_NODES:
                    motion_cost += 1

        if active_goal is not None:
            goal = goals[active_goal]
            target = (int(goal["target_x"]), int(goal["target_y"]))
            if positions["C"] == target and goal["completion_slot"] is None:
                goal["completion_slot"] = slot
                progress = create_packet("progress", active_goal, slot)
                goal["progress_packet_id"] = progress["packet_id"]

        slots_out.append(
            {
                "slot": slot,
                "requests": normalized_rows,
                "arbitrated_request_id": arbitrated_request_id,
                "active_command_goal": active_goal,
                "state": {
                    "interference": _interference(name, slot),
                    "budget_used": send_requested,
                    "collision_nodes": collision_nodes,
                    "client_position": list(positions["C"]),
                },
            }
        )
        for node in NODE_IDS:
            position_tracks[node].append(list(positions[node]))
            energy_tracks[node].append(energy[node])
            queue_tracks[node].append(len(queues[node]))

    process_arrivals(HORIZON)
    terminal_arrivals.sort()

    for goal_id in GOAL_IDS:
        goal = goals[goal_id]
        if goal["confirmation_slot"] is not None:
            goal["status"] = "confirmed"
        elif goal["completion_slot"] is not None:
            goal["status"] = "completed"
        elif goal["command_arrival_slot"] is not None and int(goal["command_arrival_slot"]) <= int(
            goal["deadline"]
        ):
            goal["status"] = "commanded"
        else:
            goal["status"] = "expired"

    packet_rows = []
    packet_keys = (
        "packet_id",
        "class",
        "goal_id",
        "source",
        "created_slot",
        "ttl",
        "initial_node",
        "final_node",
        "outcome",
    )
    for packet in sorted(packets.values(), key=lambda row: (row["created_slot"], row["packet_id"])):
        packet_rows.append({key: packet[key] for key in packet_keys})

    goal_rows = [{key: goals[goal_id][key] for key in goals[goal_id]} for goal_id in GOAL_IDS]
    robot_rows = [
        {
            "node": node,
            "positions": position_tracks[node],
            "energy": energy_tracks[node],
            "queue_lengths": queue_tracks[node],
            "local_feedback": feedback_tracks[node],
        }
        for node in NODE_IDS
    ]
    useful_commands = sum(
        int(
            goals[goal_id]["command_arrival_slot"] is not None
            and int(goals[goal_id]["command_arrival_slot"]) <= int(goals[goal_id]["deadline"])
        )
        for goal_id in GOAL_IDS
    )
    confirmed_progress = sum(
        int(goals[goal_id]["confirmation_slot"] is not None) for goal_id in GOAL_IDS
    )
    mission = {
        "send_attempts": len(hops_out),
        "useful_deliveries": useful_commands + confirmed_progress,
        "queue_wait_sum": queue_wait_sum,
        "probe_count": len(probes_out),
        "retransmissions": sum(int(hop["retransmission"]) for hop in hops_out),
        "collision_slots": collision_slots,
        "motion_cost": motion_cost,
        "energy_used": sum(INITIAL_ENERGY - energy[node] for node in MOVING_NODES),
        "completed_value": sum(
            float(goals[goal_id]["value"])
            for goal_id in GOAL_IDS
            if goals[goal_id]["completion_slot"] is not None
        ),
        "terminal_arrival_packet_ids": terminal_arrivals,
    }
    record = {
        "scenario": name,
        "slots": slots_out,
        "robots": robot_rows,
        "goals": goal_rows,
        "packets": packet_rows,
        "hops": hops_out,
        "probes": probes_out,
        "mission": mission,
    }
    return {
        "record": record,
        "legality": {rule: (bool(value[0]), str(value[1])) for rule, value in legality.items()},
    }


def _simulate_policy(name: str, provider: DecisionProvider) -> dict[str, Any]:
    """Run the fixed model with an author-side state-driven decision provider."""

    return _simulate_scenario(name, decision_provider=provider)


def _atom(check_id: str, rule: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{rule}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    for atom in atoms:
        if not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"atom share differs for {atom.id}")
    score = sum(atom.share for atom in atoms if atom.passed)
    return (
        1.0 if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-12) else max(0.0, min(1.0, score))
    )


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_object = parse_ok and isinstance(data, dict)
    root_shape = root_object and set(data) == {"scenarios"}
    scenarios_list = root_object and isinstance(data.get("scenarios"), list)
    return [
        _atom(
            "result_json",
            "parse",
            bool(parse_ok),
            f"results.json: expected strict JSON; actual_type={type(data).__name__}",
        ),
        _atom(
            "result_json",
            "root",
            bool(root_shape),
            "results.json: expected root keys={'scenarios'}; "
            f"actual_keys={sorted(data) if isinstance(data, dict) else None}",
        ),
        _atom(
            "result_json",
            "scenarios",
            bool(scenarios_list),
            "results.json.scenarios: expected an array",
        ),
    ]


def _known_scenarios(scenarios: Mapping[str, dict[str, Any]]) -> tuple[str, ...]:
    return tuple(name for name in SCENARIO_NAMES if name in scenarios)


def _evidence_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    known = _known_scenarios(scenarios)
    scenario_ok = all(_scenario_shape(scenarios[name]["record"]) for name in known)

    checks: dict[str, tuple[bool, str]] = {
        "slot_request_types": (True, "slot and request records satisfy declared types"),
        "robot_types": (True, "robot records satisfy declared types and lengths"),
        "goal_types": (True, "goal records satisfy declared types"),
        "packet_types": (True, "packet records satisfy declared types and unique IDs"),
        "hop_types": (True, "hop records satisfy declared types and unique IDs"),
        "probe_types": (True, "probe records satisfy declared types and unique IDs"),
        "mission_types": (True, "mission records satisfy declared summary types"),
    }

    def set_failure(rule: str, detail: str) -> None:
        if checks[rule][0]:
            checks[rule] = (False, detail)

    for name in known:
        record = scenarios[name]["record"]
        slot_rows = record.get("slots")
        if not isinstance(slot_rows, list) or not all(_slot_valid(row) for row in slot_rows):
            set_failure(
                "slot_request_types",
                f"{name}.slots: expected declared slot, request, and state field types",
            )
        elif any(
            [request.get("node") for request in row["requests"]] != list(NODE_IDS)
            for row in slot_rows
        ):
            set_failure(
                "slot_request_types",
                f"{name}.slots.requests: expected G,C,R1,R2 order in every slot",
            )

        robot_rows = record.get("robots")
        if not isinstance(robot_rows, list) or not all(_robot_valid(row) for row in robot_rows):
            set_failure("robot_types", f"{name}.robots: expected declared robot records")

        goal_rows = record.get("goals")
        if not isinstance(goal_rows, list) or not all(_goal_valid(row) for row in goal_rows):
            set_failure("goal_types", f"{name}.goals: expected declared goal records")

        packet_rows = record.get("packets")
        if (
            not isinstance(packet_rows, list)
            or not all(_packet_valid(row) for row in packet_rows)
            or not _ids_unique(packet_rows, "packet_id")
        ):
            set_failure(
                "packet_types", f"{name}.packets: expected valid records with unique packet_id"
            )

        hop_rows = record.get("hops")
        if (
            not isinstance(hop_rows, list)
            or not all(_hop_valid(row) for row in hop_rows)
            or not _ids_unique(hop_rows, "hop_id")
        ):
            set_failure("hop_types", f"{name}.hops: expected valid records with unique hop_id")

        probe_rows = record.get("probes")
        if (
            not isinstance(probe_rows, list)
            or not all(_probe_valid(row) for row in probe_rows)
            or not _ids_unique(probe_rows, "probe_id")
        ):
            set_failure(
                "probe_types", f"{name}.probes: expected valid records with unique probe_id"
            )

        if not _mission_valid(record.get("mission")):
            set_failure("mission_types", f"{name}.mission: expected declared summary fields")

    return [
        _atom(
            "result_evidence",
            "scenario_shape",
            scenario_ok,
            "known scenario records require the exact declared key set",
        ),
        *[
            _atom("result_evidence", rule, checks[rule][0], checks[rule][1])
            for rule in (
                "slot_request_types",
                "robot_types",
                "goal_types",
                "packet_types",
                "hop_types",
                "probe_types",
                "mission_types",
            )
        ],
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not a list"] * 8
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    for index, row in enumerate(data["scenarios"]):
        name = row.get("scenario") if isinstance(row, dict) else None
        if name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key={name!r}")
        else:
            by_name[name].append(row)
    supplied_scenarios = [
        row.get("scenario") if isinstance(row, dict) else None for row in data["scenarios"]
    ]
    if (
        len(supplied_scenarios) == len(SCENARIO_NAMES)
        and set(supplied_scenarios) == set(SCENARIO_NAMES)
        and supplied_scenarios != list(SCENARIO_NAMES)
    ):
        defects.append("results.scenarios: canonical order mismatch")

    def count_keys(
        record: Mapping[str, Any], field: str, key: str, expected: Sequence[Any], label: str
    ) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {value!r}" for value in expected)
            return
        counts = {value: 0 for value in expected}
        for index, row in enumerate(rows):
            value = row.get(key) if isinstance(row, dict) else None
            if value in counts and type(value) is type(expected[0]):
                counts[value] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key={value!r}")
        for value, count in counts.items():
            if count == 0:
                defects.append(f"{label}: missing key {value!r}")
            defects.extend(f"{label}: duplicate key {value!r}" for _ in range(max(0, count - 1)))
        supplied = [row.get(key) if isinstance(row, dict) else None for row in rows]
        if (
            len(supplied) == len(expected)
            and all(count == 1 for count in counts.values())
            and supplied != list(expected)
        ):
            defects.append(f"{label}: canonical order mismatch")

    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            defects.extend(f"{name}.slots: missing key {slot!r}" for slot in range(HORIZON))
            defects.extend(f"{name}.robots: missing key {node!r}" for node in NODE_IDS)
            defects.extend(f"{name}.goals: missing key {goal!r}" for goal in GOAL_IDS)
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(record, "slots", "slot", list(range(HORIZON)), f"{name}.slots")
        count_keys(record, "robots", "node", NODE_IDS, f"{name}.robots")
        count_keys(record, "goals", "goal_id", GOAL_IDS, f"{name}.goals")
    score = max(0.0, 4.0 - 0.5 * len(defects)) / 4.0
    return score, defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return (
        "failed_atoms=[canonical_coverage.defect]; "
        f"defect_count={len(defects)}; {shown}; normalized={score:.9g}"
    )


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules = [name.removeprefix("decision_legality.") for name in ATOM_SHARES["decision_legality"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(scenarios):
        decisions = _decision_map(scenarios[name])
        if decisions is None:
            continue
        replay = _simulate_scenario(name, decisions=decisions)
        for rule in rules:
            ok, detail = replay["legality"][rule]
            if not ok:
                fail(rule, str(detail))
    return [
        _atom(
            "decision_legality",
            rule,
            passed[rule],
            details[rule] or "all decisions satisfy this rule",
        )
        for rule in rules
    ]


def _compare(actual: Any, expected: Any, path: str) -> tuple[bool, str]:
    if isinstance(expected, bool):
        ok = isinstance(actual, bool) and actual is expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if _finite_number(expected):
        ok = _close(actual, expected)
        return (
            ok,
            "" if ok else f"{path}: expected={expected!r}; actual={actual!r}; tolerance={EPS}",
        )
    if isinstance(expected, str) or expected is None:
        ok = actual == expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            actual_length = len(actual) if isinstance(actual, list) else None
            return (
                False,
                f"{path}: expected list length={len(expected)}; "
                f"actual_type={type(actual).__name__}; actual_length={actual_length}",
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            ok, detail = _compare(actual_item, expected_item, f"{path}[{index}]")
            if not ok:
                return False, detail
        return True, ""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            actual_keys = sorted(actual) if isinstance(actual, dict) else None
            return False, f"{path}: expected keys={sorted(expected)}; actual_keys={actual_keys}"
        for key in expected:
            ok, detail = _compare(actual[key], expected[key], f"{path}.{key}")
            if not ok:
                return False, detail
        return True, ""
    ok = actual == expected
    return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"


def _replay_atoms(
    scenarios: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = [name.removeprefix("replay_validity.") for name in ATOM_SHARES["replay_validity"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(scenarios):
        item = scenarios[name]
        decisions = _decision_map(item)
        if decisions is None:
            continue
        replay = _simulate_scenario(name, decisions=decisions)
        replayed[name] = replay
        actual = item["record"]
        expected = replay["record"]
        actual_slots = [item["slots"][slot] for slot in range(HORIZON)]
        expected_slots = expected["slots"]
        robot_keys = [node for node in NODE_IDS if node in item["robots"]]
        actual_robots = [item["robots"][node] for node in robot_keys]
        expected_robot_index = {row["node"]: row for row in expected["robots"]}
        expected_robots = [expected_robot_index[node] for node in robot_keys]
        goal_keys = [goal for goal in GOAL_IDS if goal in item["goals"]]
        actual_goals = [item["goals"][goal] for goal in goal_keys]
        expected_goal_index = {row["goal_id"]: row for row in expected["goals"]}
        expected_goals = [expected_goal_index[goal] for goal in goal_keys]

        groups = {
            "link_attempts": (
                {
                    "hops": actual.get("hops"),
                    "slots": [
                        {
                            "arbitrated_request_id": row.get("arbitrated_request_id"),
                            "interference": row.get("state", {}).get("interference")
                            if isinstance(row.get("state"), dict)
                            else None,
                        }
                        for row in actual_slots
                        if isinstance(row, dict)
                    ],
                },
                {
                    "hops": expected["hops"],
                    "slots": [
                        {
                            "arbitrated_request_id": row["arbitrated_request_id"],
                            "interference": row["state"]["interference"],
                        }
                        for row in expected_slots
                    ],
                },
            ),
            "packet_queues": (
                {
                    "packets": actual.get("packets"),
                    "queues": [
                        row.get("queue_lengths") if isinstance(row, dict) else None
                        for row in actual_robots
                    ],
                },
                {
                    "packets": expected["packets"],
                    "queues": [row["queue_lengths"] for row in expected_robots],
                },
            ),
            "robot_motion": (
                {
                    "robots": [
                        {
                            "node": row.get("node"),
                            "positions": row.get("positions"),
                            "energy": row.get("energy"),
                        }
                        for row in actual_robots
                        if isinstance(row, dict)
                    ],
                    "slots": [
                        {
                            "collision_nodes": row.get("state", {}).get("collision_nodes")
                            if isinstance(row.get("state"), dict)
                            else None,
                            "client_position": row.get("state", {}).get("client_position")
                            if isinstance(row.get("state"), dict)
                            else None,
                        }
                        for row in actual_slots
                        if isinstance(row, dict)
                    ],
                },
                {
                    "robots": [
                        {
                            "node": row["node"],
                            "positions": row["positions"],
                            "energy": row["energy"],
                        }
                        for row in expected_robots
                    ],
                    "slots": [
                        {
                            "collision_nodes": row["state"]["collision_nodes"],
                            "client_position": row["state"]["client_position"],
                        }
                        for row in expected_slots
                    ],
                },
            ),
            "goal_events": (actual_goals, expected_goals),
            "probe_feedback": (
                {
                    "probes": actual.get("probes"),
                    "feedback": [
                        row.get("local_feedback") if isinstance(row, dict) else None
                        for row in actual_robots
                    ],
                },
                {
                    "probes": expected["probes"],
                    "feedback": [row["local_feedback"] for row in expected_robots],
                },
            ),
            "mission_terminal": (
                {
                    "mission": actual.get("mission"),
                    "budget_used": [
                        row.get("state", {}).get("budget_used")
                        if isinstance(row.get("state"), dict)
                        else None
                        for row in actual_slots
                        if isinstance(row, dict)
                    ],
                },
                {
                    "mission": expected["mission"],
                    "budget_used": [row["state"]["budget_used"] for row in expected_slots],
                },
            ),
        }
        for rule, (actual_group, expected_group) in groups.items():
            ok, detail = _compare(actual_group, expected_group, f"{name}.{rule}")
            if not ok:
                fail(rule, detail)

    atoms = [
        _atom(
            "replay_validity",
            rule,
            passed[rule],
            details[rule] or "fixed-model evidence comparison passed",
        )
        for rule in rules
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all canonical scenario replays are required for metrics")
    totals = {
        "goal_progress": 0.0,
        "command_connectivity": 0.0,
        "safe_motion_efficiency": 0.0,
        "delivery_efficiency": 0.0,
        "communication_overhead_efficiency": 0.0,
    }
    total_goal_value = sum(float(row[1]) for row in GOAL_BASE)
    for name in SCENARIO_NAMES:
        record = replayed[name]["record"]
        goals = record["goals"]
        slots = record["slots"]
        mission = record["mission"]

        progress_value = 0.0
        prompt_arrivals = 0.0
        active_fractions = 0.0
        for goal in goals:
            if goal["confirmation_slot"] is not None:
                credit = 1.0
            elif goal["completion_slot"] is not None:
                credit = 0.5
            else:
                credit = 0.0
            progress_value += float(goal["value"]) * credit
            arrival = goal["command_arrival_slot"]
            prompt_arrivals += float(
                arrival is not None and int(arrival) <= int(goal["release_slot"]) + 8
            )
            active_slots = sum(
                int(
                    int(goal["release_slot"]) <= row["slot"] <= int(goal["deadline"])
                    and row["active_command_goal"] == goal["goal_id"]
                )
                for row in slots
            )
            active_fractions += active_slots / (
                int(goal["deadline"]) - int(goal["release_slot"]) + 1
            )
        totals["goal_progress"] += progress_value / total_goal_value
        totals["command_connectivity"] += 0.6 * prompt_arrivals / len(
            GOAL_IDS
        ) + 0.4 * active_fractions / len(GOAL_IDS)

        safety = max(0.0, min(1.0, 1.0 - mission["collision_slots"] / HORIZON))
        motion = max(0.0, min(1.0, 1.0 - mission["motion_cost"] / (3 * HORIZON)))
        energy_score = max(0.0, min(1.0, 1.0 - mission["energy_used"] / 360.0))
        totals["safe_motion_efficiency"] += 0.4 * safety + 0.3 * motion + 0.3 * energy_score

        attempts = int(mission["send_attempts"])
        confirmed = sum(int(goal["confirmation_slot"] is not None) for goal in goals)
        delivery = (
            0.5 * int(mission["useful_deliveries"]) / attempts + 0.5 * confirmed / 6.0
            if attempts > 0
            else 0.0
        )
        totals["delivery_efficiency"] += max(0.0, min(1.0, delivery))

        if int(mission["useful_deliveries"]) > 0:
            selected_attempts = sum(int(row["arbitrated_request_id"] is not None) for row in slots)
            mean_wait = (
                float(mission["queue_wait_sum"]) / selected_attempts
                if selected_attempts > 0
                else 0.0
            )
            overhead = (
                0.4 * max(0.0, min(1.0, 1.0 - mean_wait / 12.0))
                + 0.3 * max(0.0, min(1.0, 1.0 - mission["probe_count"] / PROBE_BUDGET))
                + 0.3
                * max(
                    0.0,
                    min(1.0, 1.0 - mission["retransmissions"] / max(1, attempts)),
                )
            )
        else:
            overhead = 0.0
        totals["communication_overhead_efficiency"] += overhead

    metrics = {
        metric: max(0.0, min(1.0, total / len(SCENARIO_NAMES))) for metric, total in totals.items()
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _gated_metric(metrics: Mapping[str, float] | None, name: str) -> float:
    if metrics is None:
        return 0.0
    value = metrics.get(name, 0.0)
    return float(value) if _finite_number(value) and 0.0 <= float(value) <= 1.0 else 0.0


def _headings(text: str) -> dict[str, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*(```+|~~~+)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    found: dict[str, str] = {}
    for position, (index, level, title) in enumerate(headings):
        if title not in required:
            continue
        end = len(lines)
        for next_index, next_level, _ in headings[position + 1:]:
            if next_level <= level:
                end = next_index
                break
        body = "\n".join(lines[index + 1:end]).strip()
        if title not in found or len(body) > len(found[title]):
            found[title] = body
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
    return (
        score,
        f"missing_sections={missing}; normalized={score:.9g}" if missing else "missing_sections=[]",
    )


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = True
    try:
        state.run_started = True
        try:
            remove_result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(remove_result, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001 - retain failure in the cached state
            remove_ok = False
            state.official_message = f"could not remove previous results.json: {exc}"
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = state.exit_code is None and "timeout" in stderr.lower()
        except Exception as exc:  # noqa: BLE001 - every execution path is cached
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = (
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        if not state.official_message:
            run_summary = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"new_regular_results={state.result_file_is_new_regular}"
            )
            failure_summary = (
                f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"timed_out={state.timed_out}, "
                f"new_regular_results={state.result_file_is_new_regular}"
            )
            state.official_message = run_summary if state.official_ok else failure_summary
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"results.json: strict parse failed: {exc}"

        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            state.scores["result_json"] = 0.0
            state.diagnostics["result_json"] = f"{reason}; {state.diagnostics['result_json']}"
            for check_id in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = reason
            _CACHE[key] = state
            return state

        state.scenarios = _index_records(state.data)
        state.records = state.scenarios
        evidence_atoms = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(
            evidence_atoms, ATOM_SHARES["result_evidence"]
        )
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)

        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = _coverage_diagnostic(coverage, defects)

        legality_atoms = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(
            legality_atoms, ATOM_SHARES["decision_legality"]
        )
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)

        replay_atoms, replayed = _replay_atoms(state.scenarios)
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
    except Exception as exc:  # noqa: BLE001 - cache a deterministic failed state
        analysis_message = f"grader analysis failed: {type(exc).__name__}: {exc}"
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, analysis_message)
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


async def _metric_check(ws: Workspace, metric: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{metric}: validity gate not satisfied; metric not computed"
    value = _gated_metric(state.metrics, metric)
    return value, f"{metric}: independently replayed value={value:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "goal_progress")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "command_connectivity")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "safe_motion_efficiency")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "delivery_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "communication_overhead_efficiency")
