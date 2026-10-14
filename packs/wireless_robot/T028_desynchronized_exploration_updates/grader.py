from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 80
GRID = 12
ROBOT_COUNT = 4
SENSOR_RADIUS = 2
AGE_MAX = 8
BROADCAST_BUDGET = 20
CHANNEL_CAPACITY = 2
EPS = 1e-6
GATE_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
MOVE_SET = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
SCENARIO_NAMES = (
    "synchrony_tax",
    "range_limited",
    "clustered_frontiers",
    "interference_burst",
    "heterogeneous_value",
)
SCENARIOS: dict[str, dict[str, Any]] = {
    "synchrony_tax": {
        "starts": ((1, 1), (1, 10), (10, 1), (10, 10)),
        "rectangles": ((5, 6, 1, 10),),
        "high": {(3, 3): 3, (8, 8): 3},
        "radius": 6,
        "bursts": (),
        "deadline": 80,
    },
    "range_limited": {
        "starts": ((1, 1), (1, 10), (10, 1), (10, 10)),
        "rectangles": ((3, 4, 2, 9), (8, 9, 3, 10)),
        "high": {(2, 6): 4, (10, 6): 4},
        "radius": 4,
        "bursts": (),
        "deadline": 80,
    },
    "clustered_frontiers": {
        "starts": ((2, 2), (2, 9), (9, 2), (9, 9)),
        "rectangles": ((5, 7, 5, 7),),
        "high": {(4, 4): 5, (4, 5): 5, (8, 8): 2},
        "radius": 5,
        "bursts": (),
        "deadline": 80,
    },
    "interference_burst": {
        "starts": ((1, 6), (6, 1), (6, 10), (10, 6)),
        "rectangles": ((5, 6, 2, 4), (5, 6, 8, 10)),
        "high": {(3, 3): 3, (9, 9): 3},
        "radius": 5,
        "bursts": ((18, 31, 0.8), (52, 59, 0.6)),
        "deadline": 80,
    },
    "heterogeneous_value": {
        "starts": ((1, 1), (1, 10), (10, 1), (10, 10)),
        "rectangles": ((2, 3, 5, 9), (9, 10, 2, 7)),
        "high": {(4, 9): 8, (8, 3): 2, (6, 6): 6},
        "radius": 5,
        "bursts": (),
        "deadline": 72,
    },
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.scenario_keys": 0.25,
    },
    "result_evidence": {
        "result_evidence.robots": 1 / 7,
        "result_evidence.belief": 1 / 7,
        "result_evidence.frontiers": 1 / 7,
        "result_evidence.messages": 1 / 7,
        "result_evidence.channels": 1 / 7,
        "result_evidence.coverage": 1 / 7,
        "result_evidence.types": 1 / 7,
    },
    "decision_legality": {
        "decision_legality.moves": 0.14,
        "decision_legality.known_free": 0.14,
        "decision_legality.safety": 0.14,
        "decision_legality.claims": 0.14,
        "decision_legality.capacity": 0.11,
        "decision_legality.budget": 0.11,
        "decision_legality.retry": 0.11,
        "decision_legality.terminal": 0.11,
    },
    "replay_validity": {
        "replay_validity.robots": 1 / 7,
        "replay_validity.belief": 1 / 7,
        "replay_validity.frontiers": 1 / 7,
        "replay_validity.messages": 1 / 7,
        "replay_validity.channels": 1 / 7,
        "replay_validity.coverage": 1 / 7,
        "replay_validity.metrics": 1 / 7,
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
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    id = f"{group}.{name}"
    return AtomVerdict(id, passed, detail, ATOM_SHARES[group][id])


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return _finite(actual) and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=EPS)
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_same(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, b) for a, b in zip(actual, expected))
        )
    return actual == expected


def _first_mismatch(actual: Any, expected: Any, path: str) -> str | None:
    if isinstance(expected, float):
        if not _finite(actual) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=EPS
        ):
            return f"{path}: expected={expected!r}, actual={actual!r}, tolerance={EPS}"
        return None
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected object, actual={type(actual).__name__}"
        if set(actual) != set(expected):
            return f"{path}: expected_keys={sorted(expected)}, actual_keys={sorted(actual)}"
        for key, value in expected.items():
            mismatch = _first_mismatch(actual[key], value, f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected list, actual={type(actual).__name__}"
        if len(actual) != len(expected):
            return f"{path}: expected_length={len(expected)}, actual_length={len(actual)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatch = _first_mismatch(actual_item, expected_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    return json.loads(raw, parse_constant=reject)


def _cell_id(cell: Sequence[int]) -> int:
    return GRID * int(cell[1]) + int(cell[0])


def _cell(cell_id: int) -> tuple[int, int]:
    return cell_id % GRID, cell_id // GRID


def _obstacles(name: str) -> set[int]:
    result: set[int] = set()
    for xmin, xmax, ymin, ymax in SCENARIOS[name]["rectangles"]:
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                result.add(_cell_id((x, y)))
    return result


def _free_cells(name: str) -> set[int]:
    return set(range(GRID * GRID)) - _obstacles(name)


def _value(name: str, cell_id: int) -> int:
    return int(SCENARIOS[name]["high"].get(_cell(cell_id), 1))


def _interference(name: str, t: int) -> float:
    for start, end, value in SCENARIOS[name]["bursts"]:
        if start <= t <= end:
            return float(value)
    return 0.0


def _sense(position: Sequence[int]) -> set[int]:
    px, py = int(position[0]), int(position[1])
    return {
        _cell_id((x, y))
        for x in range(GRID)
        for y in range(GRID)
        if abs(x - px) + abs(y - py) <= SENSOR_RADIUS
    }


def _frontier(observed: set[int], free: set[int]) -> list[int]:
    result: set[int] = set()
    for cell_id in observed & free:
        x, y = _cell(cell_id)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID and 0 <= ny < GRID:
                candidate = _cell_id((nx, ny))
                if candidate not in observed:
                    result.add(candidate)
    return sorted(result)


def _nearby(positions: Sequence[Sequence[int]], robot: int) -> list[dict[str, int]]:
    x, y = positions[robot]
    return [
        {"robot": other, "x": int(positions[other][0]), "y": int(positions[other][1])}
        for other in range(ROBOT_COUNT)
        if other != robot
        and abs(int(positions[other][0]) - int(x)) + abs(int(positions[other][1]) - int(y))
        <= SENSOR_RADIUS
    ]


def _rank_frontier(name: str, frontier: Sequence[int]) -> list[int]:
    ranked = sorted(set(frontier), key=lambda cell_id: (-_value(name, cell_id), cell_id))[:12]
    return sorted(ranked)


def _encode_payload(
    sender: int,
    source_slot: int,
    version: int,
    x: int,
    y: int,
    frontier_ids: Sequence[int],
) -> str:
    ids = list(frontier_ids)
    fields: list[tuple[int, int]] = [
        (sender, 2),
        (source_slot, 7),
        (version, 7),
        (x, 4),
        (y, 4),
        (len(ids), 4),
    ]
    fields.extend((cell_id, 8) for cell_id in ids)
    fields.extend((255, 8) for _ in range(12 - len(ids)))
    fields.append((0, 4))
    packed = 0
    for value, width in fields:
        packed = (packed << width) | int(value)
    return packed.to_bytes(16, byteorder="big", signed=False).hex()


def _decode_payload(payload_hex: str) -> dict[str, Any]:
    bits = f"{int(payload_hex, 16):0128b}"
    cursor = 0

    def take(width: int) -> int:
        nonlocal cursor
        value = int(bits[cursor:cursor + width], 2)
        cursor += width
        return value

    sender = take(2)
    source_slot = take(7)
    version = take(7)
    x = take(4)
    y = take(4)
    count = take(4)
    raw_ids = [take(8) for _ in range(12)]
    reserved = take(4)
    return {
        "sender": sender,
        "source_slot": source_slot,
        "version": version,
        "x": x,
        "y": y,
        "frontier_count": count,
        "frontier_cell_ids": raw_ids[:count],
        "reserved": reserved,
    }


def _safe_action(row: Mapping[str, Any], key: str, default: Any) -> Any:
    return row.get(key, default)


def _release_claim(
    claim: dict[str, Any],
    state_t: int,
    reason: str,
    active_by_robot: list[int | None],
    active_by_cell: dict[int, dict[str, Any]],
    visited_state: int | None = None,
) -> None:
    event = claim["event"]
    if event["release_state"] is None:
        event["release_state"] = state_t
        event["release_reason"] = reason
        event["visited_state"] = visited_state
    active_by_robot[claim["owner"]] = None
    active_by_cell.pop(claim["cell_id"], None)


def _replay_scenario(name: str, rows: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    free = _free_cells(name)
    positions = [list(cell) for cell in spec["starts"]]
    observed = [_sense(position) for position in positions]
    visited = {_cell_id(position) for position in positions}
    unique_value = sum(_value(name, cell_id) for cell_id in visited)
    redundant_visits = 0
    claim_conflicts = 0
    total_travel = 0
    safety_violation = False
    newly_visited = sorted(visited)

    robot_input = rows.get("robots", {}) if isinstance(rows.get("robots"), Mapping) else {}
    caches: list[dict[int, dict[str, Any]]] = [{} for _ in range(ROBOT_COUNT)]
    pending: dict[int, list[dict[str, Any]]] = {}
    sender_versions = [0] * ROBOT_COUNT
    attempts_by_robot = [0] * ROBOT_COUNT
    payloads: dict[int, dict[str, Any]] = {}
    active_by_robot: list[int | None] = [None] * ROBOT_COUNT
    active_by_cell: dict[int, dict[str, Any]] = {}

    next_broadcast_id = 0
    next_attempt_id = 0
    next_claim_sequence = 0
    robots: list[dict[str, Any]] = []
    beliefs: list[dict[str, Any]] = []
    frontier_events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    for state_t in range(HORIZON + 1):
        if state_t > 0:
            for receiver in range(ROBOT_COUNT):
                for cache in caches[receiver].values():
                    cache["age"] = min(AGE_MAX, int(cache["age"]) + 1)

            arrivals = [message for message in pending.pop(state_t, []) if message["delivered"]]
            grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for message in arrivals:
                grouped.setdefault((message["receiver"], message["sender"]), []).append(message)
            for (receiver, sender), group in grouped.items():
                group.sort(key=lambda item: (-item["decoded"]["version"], item["attempt_id"]))
                for message in group:
                    decoded = message["decoded"]
                    current = caches[receiver].get(sender)
                    current_version = int(current["summary_version"]) if current else 0
                    if decoded["version"] > current_version:
                        caches[receiver][sender] = {
                            "summary_version": decoded["version"],
                            "source_slot": decoded["source_slot"],
                            "cached_x": decoded["x"],
                            "cached_y": decoded["y"],
                            "frontier_cell_ids": list(decoded["frontier_cell_ids"]),
                            "age": 0,
                            "last_broadcast_id": message["broadcast_id"],
                        }
                        message["fusion_status"] = "fused"
                    elif decoded["version"] == current_version:
                        message["fusion_status"] = "duplicate"
                    else:
                        message["fusion_status"] = "stale"

        for robot in range(ROBOT_COUNT):
            observed[robot].update(_sense(positions[robot]))

        for claim in list(active_by_cell.values()):
            if int(claim["expiry_state"]) <= state_t:
                _release_claim(
                    claim,
                    state_t,
                    "expired",
                    active_by_robot,
                    active_by_cell,
                )
        if state_t == HORIZON:
            for claim in list(active_by_cell.values()):
                _release_claim(
                    claim,
                    state_t,
                    "horizon",
                    active_by_robot,
                    active_by_cell,
                )

        local_frontiers = [_frontier(observed[robot], free) for robot in range(ROBOT_COUNT)]
        shared_frontiers: list[list[int]] = []
        for receiver in range(ROBOT_COUNT):
            shared: set[int] = set()
            for cache in caches[receiver].values():
                if int(cache["age"]) < AGE_MAX:
                    shared.update(cache["frontier_cell_ids"])
            shared.difference_update(observed[receiver])
            shared_frontiers.append(sorted(shared))

        coverage.append({
            "state_t": state_t,
            "newly_visited_cell_ids": list(newly_visited),
            "cumulative_visited_cell_ids": sorted(visited),
            "unique_value": unique_value,
            "redundant_visits": redundant_visits,
            "claim_conflicts": claim_conflicts,
            "total_travel": total_travel,
            "safety_violation": safety_violation,
            "deadline_state": int(spec["deadline"]),
        })

        for receiver in range(ROBOT_COUNT):
            for sender in range(ROBOT_COUNT):
                if receiver == sender:
                    continue
                cache = caches[receiver].get(sender)
                beliefs.append({
                    "receiver": receiver,
                    "sender": sender,
                    "state_t": state_t,
                    "summary_version": cache["summary_version"] if cache else None,
                    "source_slot": cache["source_slot"] if cache else None,
                    "cached_x": cache["cached_x"] if cache else None,
                    "cached_y": cache["cached_y"] if cache else None,
                    "frontier_cell_ids": list(cache["frontier_cell_ids"]) if cache else [],
                    "age": int(cache["age"]) if cache else AGE_MAX,
                    "available": bool(cache and int(cache["age"]) < AGE_MAX),
                    "last_broadcast_id": cache["last_broadcast_id"] if cache else None,
                })

        if state_t == HORIZON:
            for robot in range(ROBOT_COUNT):
                robots.append({
                    "robot": robot,
                    "state_t": state_t,
                    "x": int(positions[robot][0]),
                    "y": int(positions[robot][1]),
                    "observed_cell_ids": sorted(observed[robot]),
                    "local_frontier_cell_ids": list(local_frontiers[robot]),
                    "shared_frontier_cell_ids": list(shared_frontiers[robot]),
                    "nearby_robots": _nearby(positions, robot),
                    "active_claim_cell_id": active_by_robot[robot],
                    "remaining_broadcast_budget": max(0, BROADCAST_BUDGET - attempts_by_robot[robot]),
                    "move_dx": 0,
                    "move_dy": 0,
                    "claim_cell_id": None,
                    "tx_kind": "none",
                    "retry_broadcast_id": None,
                    "emitted_broadcast_id": None,
                    "safe": not safety_violation,
                    "deadline_state": int(spec["deadline"]),
                })
            break

        start_positions = [list(position) for position in positions]
        start_claims = list(active_by_robot)
        start_claims_by_cell = dict(active_by_cell)
        start_attempts = list(attempts_by_robot)
        start_safe = not safety_violation
        actions: list[dict[str, Any]] = []
        for robot in range(ROBOT_COUNT):
            source = robot_input.get((robot, state_t), {})
            source = source if isinstance(source, Mapping) else {}
            actions.append({
                "move_dx": _safe_action(source, "move_dx", 0),
                "move_dy": _safe_action(source, "move_dy", 0),
                "claim_cell_id": _safe_action(source, "claim_cell_id", None),
                "tx_kind": _safe_action(source, "tx_kind", "none"),
                "retry_broadcast_id": _safe_action(source, "retry_broadcast_id", None),
                "emitted_broadcast_id": None,
            })

        requested_senders = [
            robot for robot, action in enumerate(actions)
            if action["tx_kind"] in ("new", "retry")
        ]
        served_senders = requested_senders[:CHANNEL_CAPACITY]
        interference = _interference(name, state_t)
        attempt_ids: list[int] = []
        served_attempt_ids: list[int] = []
        attempt_info: list[tuple[int, int, dict[str, Any], bool]] = []
        for robot in requested_senders:
            action = actions[robot]
            payload: dict[str, Any] | None = None
            if action["tx_kind"] == "new":
                sender_versions[robot] += 1
                frontier_ids = _rank_frontier(name, local_frontiers[robot])
                broadcast_id = next_broadcast_id
                next_broadcast_id += 1
                payload_hex = _encode_payload(
                    robot,
                    state_t,
                    sender_versions[robot],
                    int(positions[robot][0]),
                    int(positions[robot][1]),
                    frontier_ids,
                )
                payload = {
                    "broadcast_id": broadcast_id,
                    "sender": robot,
                    "payload_hex": payload_hex,
                    "decoded": _decode_payload(payload_hex),
                }
                payloads[broadcast_id] = payload
            elif _integer(action["retry_broadcast_id"]):
                candidate = payloads.get(int(action["retry_broadcast_id"]))
                if candidate is not None and candidate["sender"] == robot:
                    payload = candidate
            if payload is None:
                continue
            attempt_id = next_attempt_id
            next_attempt_id += 1
            attempts_by_robot[robot] += 1
            action["emitted_broadcast_id"] = payload["broadcast_id"]
            attempt_ids.append(attempt_id)
            served = robot in served_senders
            if served:
                served_attempt_ids.append(attempt_id)
            attempt_info.append((attempt_id, robot, payload, served))

        concurrency = len(served_attempt_ids)
        channels.append({
            "t": state_t,
            "interference": interference,
            "requested_sender_ids": requested_senders,
            "served_sender_ids": [robot for _, robot, _, served in attempt_info if served],
            "attempt_ids": attempt_ids,
            "served_attempt_ids": served_attempt_ids,
            "concurrency": concurrency,
            "capacity_ok": len(requested_senders) <= CHANNEL_CAPACITY,
        })

        for attempt_id, sender, payload, served in attempt_info:
            sx, sy = start_positions[sender]
            for receiver in range(ROBOT_COUNT):
                if receiver == sender:
                    continue
                distance = abs(int(start_positions[receiver][0]) - int(sx)) + abs(int(start_positions[receiver][1]) - int(sy))
                in_range = distance <= int(spec["radius"])
                delivered = bool(
                    served
                    and in_range
                    and interference + 0.2 * (concurrency - 1) <= 0.8 + EPS
                )
                if not served:
                    status = "capacity_lost"
                elif not in_range:
                    status = "out_of_range"
                elif not delivered:
                    status = "interference_lost"
                else:
                    status = "fused"
                message = {
                    "attempt_id": attempt_id,
                    "broadcast_id": payload["broadcast_id"],
                    "receiver": receiver,
                    "sender": sender,
                    "transmit_slot": state_t,
                    "tx_kind": actions[sender]["tx_kind"],
                    "arrival_state": state_t + 1,
                    "receiver_distance": distance,
                    "in_range": in_range,
                    "interference": interference,
                    "concurrency": concurrency,
                    "payload_hex": payload["payload_hex"],
                    "decoded": dict(payload["decoded"]),
                    "delivered": delivered,
                    "fusion_status": status,
                }
                messages.append(message)
                if delivered:
                    pending.setdefault(state_t + 1, []).append(message)

        requests: dict[int, list[int]] = {}
        valid_candidates: list[set[int]] = [
            set(local_frontiers[robot]) | set(shared_frontiers[robot])
            for robot in range(ROBOT_COUNT)
        ]
        request_events: dict[int, dict[str, Any]] = {}
        for robot, action in enumerate(actions):
            cell_id = action["claim_cell_id"]
            if cell_id is None:
                continue
            event = {
                "claim_sequence": next_claim_sequence,
                "robot": robot,
                "cell_id": int(cell_id) if _integer(cell_id) else -1,
                "request_slot": state_t,
                "outcome": "invalid",
                "owner_robot": None,
                "expiry_state": None,
                "release_state": None,
                "release_reason": None,
                "visited_state": None,
            }
            next_claim_sequence += 1
            frontier_events.append(event)
            request_events[robot] = event
            if _integer(cell_id) and int(cell_id) in valid_candidates[robot]:
                requests.setdefault(int(cell_id), []).append(robot)

        winning_claims: list[tuple[int, int, int, dict[str, Any]]] = []
        for cell_id, claimants in requests.items():
            existing = start_claims_by_cell.get(cell_id)
            if existing is not None:
                for robot in claimants:
                    event = request_events[robot]
                    event["outcome"] = "conflict"
                    event["owner_robot"] = existing["owner"]
                    event["expiry_state"] = existing["expiry_state"]
                    claim_conflicts += 1
                continue
            winner = min(claimants)
            expiry_state = state_t + 6
            for robot in claimants:
                event = request_events[robot]
                event["owner_robot"] = winner
                event["expiry_state"] = expiry_state
                if robot != winner:
                    event["outcome"] = "conflict"
                    claim_conflicts += 1
            event = request_events[winner]
            event["outcome"] = "won"
            winning_claims.append((winner, cell_id, expiry_state, event))

        for winner, cell_id, expiry_state, event in sorted(winning_claims):
            previous_cell = active_by_robot[winner]
            if previous_cell is not None:
                previous = active_by_cell.get(previous_cell)
                if previous is not None:
                    _release_claim(
                        previous,
                        state_t + 1,
                        "replaced",
                        active_by_robot,
                        active_by_cell,
                    )
            claim = {
                "owner": winner,
                "cell_id": cell_id,
                "expiry_state": expiry_state,
                "event": event,
            }
            active_by_robot[winner] = cell_id
            active_by_cell[cell_id] = claim

        next_positions = [list(position) for position in positions]
        moved: list[tuple[int, int]] = []
        for robot, action in enumerate(actions):
            dx = action["move_dx"]
            dy = action["move_dy"]
            valid_vector = _integer(dx) and _integer(dy) and (int(dx), int(dy)) in MOVE_SET
            if not valid_vector:
                safety_violation = True
                continue
            if int(dx) == 0 and int(dy) == 0:
                continue
            nx = int(positions[robot][0]) + int(dx)
            ny = int(positions[robot][1]) + int(dy)
            destination = _cell_id((nx, ny)) if 0 <= nx < GRID and 0 <= ny < GRID else -1
            if destination not in observed[robot] or destination not in free:
                safety_violation = True
                continue
            next_positions[robot] = [nx, ny]
            total_travel += 1
            moved.append((robot, destination))

        if len({tuple(position) for position in next_positions}) != ROBOT_COUNT:
            safety_violation = True

        newly_set: set[int] = set()
        for robot, destination in sorted(moved):
            if destination not in visited:
                visited.add(destination)
                newly_set.add(destination)
                unique_value += _value(name, destination)
            else:
                redundant_visits += 1
        positions = next_positions
        newly_visited = sorted(newly_set)

        entered_cells = {destination for _, destination in moved}
        for cell_id in entered_cells:
            claim = active_by_cell.get(cell_id)
            if claim is not None:
                _release_claim(
                    claim,
                    state_t + 1,
                    "visited",
                    active_by_robot,
                    active_by_cell,
                    visited_state=state_t + 1,
                )

        for robot, action in enumerate(actions):
            robots.append({
                "robot": robot,
                "state_t": state_t,
                "x": int(start_positions[robot][0]),
                "y": int(start_positions[robot][1]),
                "observed_cell_ids": sorted(observed[robot]),
                "local_frontier_cell_ids": list(local_frontiers[robot]),
                "shared_frontier_cell_ids": list(shared_frontiers[robot]),
                "nearby_robots": _nearby(start_positions, robot),
                "active_claim_cell_id": start_claims[robot],
                "remaining_broadcast_budget": max(0, BROADCAST_BUDGET - start_attempts[robot]),
                "move_dx": int(action["move_dx"]) if _integer(action["move_dx"]) else 0,
                "move_dy": int(action["move_dy"]) if _integer(action["move_dy"]) else 0,
                "claim_cell_id": action["claim_cell_id"] if _integer(action["claim_cell_id"]) else None,
                "tx_kind": action["tx_kind"] if action["tx_kind"] in ("none", "new", "retry") else "none",
                "retry_broadcast_id": action["retry_broadcast_id"] if _integer(action["retry_broadcast_id"]) else None,
                "emitted_broadcast_id": action["emitted_broadcast_id"],
                "safe": start_safe,
                "deadline_state": int(spec["deadline"]),
            })

    robots.sort(key=lambda row: (row["robot"], row["state_t"]))
    beliefs.sort(key=lambda row: (row["receiver"], row["sender"], row["state_t"]))
    frontier_events.sort(key=lambda row: row["claim_sequence"])
    messages.sort(key=lambda row: (row["attempt_id"], row["receiver"]))
    return {
        "scenario": name,
        "robots": robots,
        "belief": beliefs,
        "frontiers": frontier_events,
        "messages": messages,
        "channels": channels,
        "coverage": coverage,
    }


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {
        "scenario", "robots", "belief", "frontiers", "messages", "channels", "coverage"
    }


def _sorted_ids(value: Any) -> bool:
    if not isinstance(value, list) or not all(
        _integer(cell_id) and 0 <= cell_id < GRID * GRID for cell_id in value
    ):
        return False
    return value == sorted(set(value))


def _nullable_id(value: Any, upper: int | None = None) -> bool:
    return value is None or (
        _integer(value) and value >= 0 and (upper is None or value <= upper)
    )


def _valid_robot(row: Any) -> bool:
    keys = {
        "robot", "state_t", "x", "y", "observed_cell_ids", "local_frontier_cell_ids",
        "shared_frontier_cell_ids", "nearby_robots", "active_claim_cell_id",
        "remaining_broadcast_budget", "move_dx", "move_dy", "claim_cell_id", "tx_kind",
        "retry_broadcast_id", "emitted_broadcast_id", "safe", "deadline_state",
    }
    if not isinstance(row, Mapping) or set(row) != keys:
        return False
    nearby = row["nearby_robots"]
    nearby_items_ok = (
        isinstance(nearby, list)
        and all(
            isinstance(item, Mapping)
            and set(item) == {"robot", "x", "y"}
            and all(_integer(item[key]) for key in ("robot", "x", "y"))
            and 0 <= item["robot"] < ROBOT_COUNT
            and 0 <= item["x"] < GRID
            and 0 <= item["y"] < GRID
            for item in nearby
        )
    )
    nearby_ok = bool(
        nearby_items_ok
        and nearby == sorted(nearby, key=lambda item: item["robot"])
        and len({item["robot"] for item in nearby}) == len(nearby)
    )
    retry_ok = (
        row["tx_kind"] == "retry"
        and _integer(row["retry_broadcast_id"])
        and row["retry_broadcast_id"] >= 0
        or row["tx_kind"] in ("none", "new") and row["retry_broadcast_id"] is None
    )
    emitted_ok = (
        row["tx_kind"] in ("new", "retry")
        and _integer(row["emitted_broadcast_id"])
        and row["emitted_broadcast_id"] >= 0
        or row["tx_kind"] == "none" and row["emitted_broadcast_id"] is None
    )
    return bool(
        _integer(row["robot"]) and 0 <= row["robot"] < ROBOT_COUNT
        and _integer(row["state_t"]) and 0 <= row["state_t"] <= HORIZON
        and _integer(row["x"]) and 0 <= row["x"] < GRID
        and _integer(row["y"]) and 0 <= row["y"] < GRID
        and _sorted_ids(row["observed_cell_ids"])
        and _sorted_ids(row["local_frontier_cell_ids"])
        and _sorted_ids(row["shared_frontier_cell_ids"])
        and nearby_ok
        and _nullable_id(row["active_claim_cell_id"], GRID * GRID - 1)
        and _integer(row["remaining_broadcast_budget"])
        and 0 <= row["remaining_broadcast_budget"] <= BROADCAST_BUDGET
        and _integer(row["move_dx"]) and _integer(row["move_dy"])
        and (row["move_dx"], row["move_dy"]) in MOVE_SET
        and _nullable_id(row["claim_cell_id"], GRID * GRID - 1)
        and row["tx_kind"] in ("none", "new", "retry")
        and retry_ok
        and emitted_ok
        and isinstance(row["safe"], bool)
        and _integer(row["deadline_state"])
        and 0 <= row["deadline_state"] <= HORIZON
    )


def _valid_belief(row: Any) -> bool:
    keys = {
        "receiver", "sender", "state_t", "summary_version", "source_slot", "cached_x",
        "cached_y", "frontier_cell_ids", "age", "available", "last_broadcast_id",
    }
    if not isinstance(row, Mapping) or set(row) != keys:
        return False
    absent = row["summary_version"] is None
    age_ok = _integer(row["age"]) and 0 <= row["age"] <= AGE_MAX
    content_ok = (
        absent
        and all(row[key] is None for key in ("source_slot", "cached_x", "cached_y", "last_broadcast_id"))
        and row["frontier_cell_ids"] == []
        and row["age"] == AGE_MAX
        and row["available"] is False
    ) or (
        _integer(row["summary_version"]) and 1 <= row["summary_version"] <= 127
        and _integer(row["source_slot"]) and 0 <= row["source_slot"] < HORIZON
        and _integer(row["cached_x"]) and 0 <= row["cached_x"] < GRID
        and _integer(row["cached_y"]) and 0 <= row["cached_y"] < GRID
        and _sorted_ids(row["frontier_cell_ids"])
        and _nullable_id(row["last_broadcast_id"])
        and age_ok
        and row["available"] == (row["age"] < AGE_MAX)
    )
    return bool(
        _integer(row["receiver"]) and 0 <= row["receiver"] < ROBOT_COUNT
        and _integer(row["sender"]) and 0 <= row["sender"] < ROBOT_COUNT
        and row["receiver"] != row["sender"]
        and _integer(row["state_t"]) and 0 <= row["state_t"] <= HORIZON
        and age_ok
        and isinstance(row["available"], bool)
        and content_ok
    )


def _valid_frontier(row: Any) -> bool:
    keys = {
        "claim_sequence", "robot", "cell_id", "request_slot", "outcome", "owner_robot",
        "expiry_state", "release_state", "release_reason", "visited_state",
    }
    if not isinstance(row, Mapping) or set(row) != keys:
        return False
    base = (
        _integer(row["claim_sequence"]) and row["claim_sequence"] >= 0
        and _integer(row["robot"]) and 0 <= row["robot"] < ROBOT_COUNT
        and _integer(row["cell_id"]) and -1 <= row["cell_id"] < GRID * GRID
        and _integer(row["request_slot"]) and 0 <= row["request_slot"] < HORIZON
        and row["outcome"] in ("won", "conflict", "invalid")
    )
    if not base:
        return False
    if row["outcome"] == "won":
        return bool(
            row["owner_robot"] == row["robot"]
            and _integer(row["expiry_state"])
            and 0 <= row["expiry_state"] <= HORIZON + 5
            and _integer(row["release_state"]) and 0 <= row["release_state"] <= HORIZON
            and row["release_reason"] in ("visited", "expired", "replaced", "horizon")
            and _nullable_id(row["visited_state"], HORIZON)
        )
    if row["outcome"] == "conflict":
        return bool(
            _integer(row["owner_robot"]) and 0 <= row["owner_robot"] < ROBOT_COUNT
            and _integer(row["expiry_state"])
            and 0 <= row["expiry_state"] <= HORIZON + 5
            and row["release_state"] is None
            and row["release_reason"] is None
            and row["visited_state"] is None
        )
    return all(row[key] is None for key in (
        "owner_robot", "expiry_state", "release_state", "release_reason", "visited_state"
    ))


def _valid_message(row: Any) -> bool:
    keys = {
        "attempt_id", "broadcast_id", "receiver", "sender", "transmit_slot", "tx_kind",
        "arrival_state", "receiver_distance", "in_range", "interference", "concurrency",
        "payload_hex", "decoded", "delivered", "fusion_status",
    }
    decoded_keys = {
        "sender", "source_slot", "version", "x", "y", "frontier_count",
        "frontier_cell_ids", "reserved",
    }
    if not isinstance(row, Mapping) or set(row) != keys:
        return False
    decoded = row["decoded"]
    hex_ok = isinstance(row["payload_hex"], str) and re.fullmatch(r"[0-9a-f]{32}", row["payload_hex"])
    decoded_ok = (
        isinstance(decoded, Mapping) and set(decoded) == decoded_keys
        and _integer(decoded["sender"]) and decoded["sender"] == row["sender"]
        and _integer(decoded["source_slot"]) and 0 <= decoded["source_slot"] < HORIZON
        and _integer(decoded["version"]) and 1 <= decoded["version"] <= 127
        and _integer(decoded["x"]) and 0 <= decoded["x"] < GRID
        and _integer(decoded["y"]) and 0 <= decoded["y"] < GRID
        and _integer(decoded["frontier_count"])
        and decoded["frontier_count"] == len(decoded["frontier_cell_ids"])
        and 0 <= decoded["frontier_count"] <= 12
        and _sorted_ids(decoded["frontier_cell_ids"])
        and decoded["reserved"] == 0
    )
    delivered_status = row["fusion_status"] in ("fused", "duplicate", "stale")
    return bool(
        _integer(row["attempt_id"]) and row["attempt_id"] >= 0
        and _integer(row["broadcast_id"]) and row["broadcast_id"] >= 0
        and _integer(row["receiver"]) and 0 <= row["receiver"] < ROBOT_COUNT
        and _integer(row["sender"]) and 0 <= row["sender"] < ROBOT_COUNT
        and row["receiver"] != row["sender"]
        and _integer(row["transmit_slot"]) and 0 <= row["transmit_slot"] < HORIZON
        and row["tx_kind"] in ("new", "retry")
        and row["arrival_state"] == row["transmit_slot"] + 1
        and _integer(row["receiver_distance"]) and row["receiver_distance"] >= 0
        and isinstance(row["in_range"], bool)
        and _finite(row["interference"]) and 0 <= row["interference"] <= 1
        and _integer(row["concurrency"]) and 0 <= row["concurrency"] <= CHANNEL_CAPACITY
        and bool(hex_ok) and decoded_ok
        and isinstance(row["delivered"], bool)
        and row["fusion_status"] in (
            "capacity_lost", "out_of_range", "interference_lost", "fused", "duplicate", "stale"
        )
        and row["delivered"] == delivered_status
    )


def _valid_channel(row: Any) -> bool:
    keys = {
        "t", "interference", "requested_sender_ids", "served_sender_ids", "attempt_ids",
        "served_attempt_ids", "concurrency", "capacity_ok",
    }
    if not isinstance(row, Mapping) or set(row) != keys:
        return False
    requested = row["requested_sender_ids"]
    served = row["served_sender_ids"]
    attempts = row["attempt_ids"]
    served_attempts = row["served_attempt_ids"]
    lists_ok = (
        isinstance(requested, list)
        and all(_integer(robot) and 0 <= robot < ROBOT_COUNT for robot in requested)
        and isinstance(served, list)
        and all(_integer(robot) and 0 <= robot < ROBOT_COUNT for robot in served)
        and isinstance(attempts, list)
        and all(_integer(attempt) and attempt >= 0 for attempt in attempts)
        and isinstance(served_attempts, list)
        and all(_integer(attempt) and attempt >= 0 for attempt in served_attempts)
    )
    return bool(
        lists_ok
        and _integer(row["t"]) and 0 <= row["t"] < HORIZON
        and _finite(row["interference"]) and 0 <= row["interference"] <= 1
        and requested == sorted(set(requested))
        and served == requested[:CHANNEL_CAPACITY]
        and _integer(row["concurrency"])
        and row["concurrency"] == len(served_attempts)
        and 0 <= row["concurrency"] <= CHANNEL_CAPACITY
        and isinstance(row["capacity_ok"], bool)
        and row["capacity_ok"] == (len(requested) <= CHANNEL_CAPACITY)
    )


def _valid_coverage(row: Any) -> bool:
    keys = {
        "state_t", "newly_visited_cell_ids", "cumulative_visited_cell_ids", "unique_value",
        "redundant_visits", "claim_conflicts", "total_travel", "safety_violation",
        "deadline_state",
    }
    return bool(
        isinstance(row, Mapping) and set(row) == keys
        and _integer(row["state_t"]) and 0 <= row["state_t"] <= HORIZON
        and _sorted_ids(row["newly_visited_cell_ids"])
        and _sorted_ids(row["cumulative_visited_cell_ids"])
        and all(
            _integer(row[key]) and row[key] >= 0
            for key in ("unique_value", "redundant_visits", "claim_conflicts", "total_travel")
        )
        and isinstance(row["safety_violation"], bool)
        and _integer(row["deadline_state"]) and 0 <= row["deadline_state"] <= HORIZON
    )


def _index_records(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scenarios = data.get("scenarios", []) if isinstance(data, Mapping) else []
    if not isinstance(scenarios, list):
        return result
    for row in scenarios:
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            continue
        name = row["scenario"]
        indexed = dict(row)
        for field, key_fn in (
            ("robots", lambda item: (item.get("robot"), item.get("state_t"))),
            ("belief", lambda item: (item.get("receiver"), item.get("sender"), item.get("state_t"))),
            ("frontiers", lambda item: item.get("claim_sequence")),
            ("messages", lambda item: (item.get("attempt_id"), item.get("receiver"))),
            ("channels", lambda item: item.get("t")),
            ("coverage", lambda item: item.get("state_t")),
        ):
            values = row.get(field, [])
            mapping: dict[Hashable, Any] = {}
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        try:
                            mapping[key_fn(item)] = item
                        except (TypeError, ValueError):
                            pass
            indexed[field] = mapping
            indexed[f"{field}_rows"] = values
        result[name] = indexed
    return result


def _evidence_atoms(sc: Mapping[str, Any]) -> list[AtomVerdict]:
    complete = all(name in sc for name in SCENARIO_NAMES)

    def rows(name: str, field: str) -> list[Any]:
        value = sc[name].get(f"{field}_rows", []) if name in sc else []
        return value if isinstance(value, list) else []

    validators = {
        "robots": (_valid_robot, ROBOT_COUNT * (HORIZON + 1)),
        "belief": (_valid_belief, ROBOT_COUNT * (ROBOT_COUNT - 1) * (HORIZON + 1)),
        "frontiers": (_valid_frontier, None),
        "messages": (_valid_message, None),
        "channels": (_valid_channel, HORIZON),
        "coverage": (_valid_coverage, HORIZON + 1),
    }
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    for field, (validator, expected_length) in validators.items():
        checks[field] = complete
        details[field] = f"{field} evidence contract passed"
        if not complete:
            details[field] = "missing canonical scenario"
            continue
        for name in SCENARIO_NAMES:
            values = rows(name, field)
            if expected_length is not None and len(values) != expected_length:
                checks[field] = False
                details[field] = (
                    f"{name}.{field}: expected_length={expected_length}, "
                    f"actual_length={len(values)}"
                )
                break
            invalid_index = next(
                (index for index, row in enumerate(values) if not validator(row)),
                None,
            )
            if invalid_index is not None:
                checks[field] = False
                details[field] = f"{name}.{field}[{invalid_index}]: invalid structure/type/range"
                break
    checks["types"] = complete and all(
        isinstance(sc[name].get(f"{field}_rows"), list)
        for name in SCENARIO_NAMES
        for field in validators
    )
    details["types"] = (
        "all evidence collections are lists"
        if checks["types"]
        else "one or more scenario evidence collections are not lists"
    )
    return [
        _atom("result_evidence", name, checks[name], details[name])
        for name in checks
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    scenarios = data.get("scenarios", []) if isinstance(data, Mapping) else []
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios is not a list"]
    names = [
        row.get("scenario")
        for row in scenarios
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str)
    ]
    invalid_name_count = sum(
        not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str)
        for row in scenarios
    )
    if invalid_name_count:
        defects.append(f"scenario keys: invalid_count={invalid_name_count}")
    for expected in SCENARIO_NAMES:
        count = names.count(expected)
        if count != 1:
            defects.append(f"scenario {expected}: count={count}")
    for unknown in sorted({name for name in names if name not in SCENARIO_NAMES}):
        defects.append(f"unknown scenario {unknown}")

    expected_keys: dict[str, list[Hashable]] = {
        "robots": [(robot, t) for robot in range(ROBOT_COUNT) for t in range(HORIZON + 1)],
        "belief": [(receiver, sender, t) for receiver in range(ROBOT_COUNT) for sender in range(ROBOT_COUNT) if sender != receiver for t in range(HORIZON + 1)],
        "channels": list(range(HORIZON)),
        "coverage": list(range(HORIZON + 1)),
    }
    key_fns = {
        "robots": lambda row: (row.get("robot"), row.get("state_t")),
        "belief": lambda row: (row.get("receiver"), row.get("sender"), row.get("state_t")),
        "channels": lambda row: row.get("t"),
        "coverage": lambda row: row.get("state_t"),
    }
    for name in SCENARIO_NAMES:
        matching = [row for row in scenarios if isinstance(row, Mapping) and row.get("scenario") == name]
        if len(matching) != 1:
            continue
        scenario = matching[0]
        for field, expected in expected_keys.items():
            values = scenario.get(field)
            actual = [
                key_fns[field](row)
                for row in values
            ] if isinstance(values, list) and all(isinstance(row, Mapping) for row in values) else []
            hashable = True
            try:
                actual_set = set(actual)
            except TypeError:
                hashable = False
                actual_set = set()
            if not hashable or actual_set != set(expected):
                defects.append(f"{name}.{field}: key set")
            if hashable and len(actual) != len(actual_set):
                defects.append(f"{name}.{field}: duplicate")
            if actual != expected:
                defects.append(f"{name}.{field}: order")
        frontiers = scenario.get("frontiers")
        sequences = [row.get("claim_sequence") for row in frontiers] if isinstance(frontiers, list) and all(isinstance(row, Mapping) for row in frontiers) else []
        if not all(_integer(sequence) for sequence in sequences) or len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
            defects.append(f"{name}.frontiers: duplicate/order")
        messages = scenario.get("messages")
        message_keys = [(row.get("attempt_id"), row.get("receiver")) for row in messages] if isinstance(messages, list) and all(isinstance(row, Mapping) for row in messages) else []
        message_keys_valid = all(
            _integer(attempt_id) and _integer(receiver)
            for attempt_id, receiver in message_keys
        )
        if not message_keys_valid or len(message_keys) != len(set(message_keys)) or message_keys != sorted(message_keys):
            defects.append(f"{name}.messages: duplicate/order")
    return max(0.0, 1.0 - len(defects) / 8.0), defects


def _legality_atoms(sc: Mapping[str, Any]) -> list[AtomVerdict]:
    names = [id.split(".", 1)[1] for id in ATOM_SHARES["decision_legality"]]
    checks = {name: True for name in names}
    details = {name: "passed" for name in checks}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    for name in SCENARIO_NAMES:
        if name not in sc:
            for rule in checks:
                fail(rule, f"{name}: missing scenario")
            continue
        robot_rows = sc[name].get("robots_rows", [])
        if not isinstance(robot_rows, list):
            for rule in checks:
                fail(rule, f"{name}: robots must be a list")
            continue
        by_key = {
            (row.get("robot"), row.get("state_t")): row
            for row in robot_rows
            if isinstance(row, Mapping)
            and _integer(row.get("robot"))
            and _integer(row.get("state_t"))
        }
        for robot in range(ROBOT_COUNT):
            attempts = 0
            owned_broadcasts: set[int] = set()
            for t in range(HORIZON):
                row = by_key.get((robot, t), {})
                dx, dy = row.get("move_dx"), row.get("move_dy")
                if not (_integer(dx) and _integer(dy) and (dx, dy) in MOVE_SET):
                    fail("moves", f"{name}.robots[{robot},{t}]: move vector")
                if _integer(dx) and _integer(dy) and (dx, dy) != (0, 0):
                    x, y = row.get("x"), row.get("y")
                    if not (_integer(x) and _integer(y)):
                        fail("known_free", f"{name}.robots[{robot},{t}]: position")
                    else:
                        nx, ny = x + dx, y + dy
                        destination = _cell_id((nx, ny)) if 0 <= nx < GRID and 0 <= ny < GRID else -1
                        if destination not in row.get("observed_cell_ids", []) or destination not in _free_cells(name):
                            fail("known_free", f"{name}.robots[{robot},{t}]: unknown/blocked destination")
                claim = row.get("claim_cell_id")
                if claim is not None:
                    local = row.get("local_frontier_cell_ids", [])
                    shared = row.get("shared_frontier_cell_ids", [])
                    if not (
                        _integer(claim)
                        and _sorted_ids(local)
                        and _sorted_ids(shared)
                        and claim in set(local) | set(shared)
                    ):
                        fail("claims", f"{name}.robots[{robot},{t}]: claim outside candidate frontiers")
                kind = row.get("tx_kind")
                if kind in ("new", "retry"):
                    attempts += 1
                    emitted = row.get("emitted_broadcast_id")
                    if kind == "new" and _integer(emitted):
                        owned_broadcasts.add(emitted)
                    retry_id = row.get("retry_broadcast_id")
                    if kind == "retry" and (
                        not _integer(retry_id) or retry_id not in owned_broadcasts
                    ):
                        fail("retry", f"{name}.robots[{robot},{t}]: retry is not an earlier own payload")
                elif kind != "none":
                    fail("retry", f"{name}.robots[{robot},{t}]: tx_kind")
            if attempts > BROADCAST_BUDGET:
                fail("budget", f"{name}.robot[{robot}]: attempts={attempts}")
            terminal = by_key.get((robot, HORIZON), {})
            if not (
                terminal.get("move_dx") == 0
                and terminal.get("move_dy") == 0
                and terminal.get("claim_cell_id") is None
                and terminal.get("tx_kind") == "none"
                and terminal.get("retry_broadcast_id") is None
                and terminal.get("emitted_broadcast_id") is None
            ):
                fail("terminal", f"{name}.robot[{robot}]: terminal action")
        for t in range(HORIZON):
            count = sum(by_key.get((robot, t), {}).get("tx_kind") in ("new", "retry") for robot in range(ROBOT_COUNT))
            if count > CHANNEL_CAPACITY:
                fail("capacity", f"{name}.slot[{t}]: requests={count}")
        try:
            expected = _replay_scenario(name, sc[name])
            if expected["coverage"][-1]["safety_violation"]:
                fail("safety", f"{name}: replayed safety violation")
        except Exception as exc:
            fail("safety", f"{name}: replay failed: {exc}")
    return [_atom("decision_legality", name, checks[name], details[name]) for name in checks]


def _metrics(replayed: Mapping[str, Any], sc: Mapping[str, Any] | None = None) -> dict[str, float]:
    coverage_scores: list[float] = []
    efficiency_scores: list[float] = []
    freshness_scores: list[float] = []
    communication_scores: list[float] = []
    for name in SCENARIO_NAMES:
        result = replayed[name]
        deadline = int(SCENARIOS[name]["deadline"])
        boundary = result["coverage"][deadline]
        total_value = sum(_value(name, cell_id) for cell_id in _free_cells(name))
        coverage_scores.append(max(0.0, min(1.0, boundary["unique_value"] / total_value)))
        travel_score = max(0.0, min(1.0, 1.0 - boundary["total_travel"] / (4 * deadline)))
        repeat_score = max(0.0, min(1.0, 1.0 - boundary["redundant_visits"] / (4 * deadline)))
        efficiency_scores.append(0.6 * travel_score + 0.4 * repeat_score)
        beliefs = {
            (row["receiver"], row["sender"], row["state_t"]): row
            for row in result["belief"]
        }
        peaks: list[float] = []
        for state_t in range(1, deadline + 1):
            peaks.append(max(
                (beliefs[(i, j, state_t)]["age"] + beliefs[(j, i, state_t)]["age"]) / 2
                for i in range(ROBOT_COUNT)
                for j in range(i + 1, ROBOT_COUNT)
            ))
        mean_peak = sum(peaks) / deadline
        freshness_scores.append(max(0.0, min(1.0, 1.0 - mean_peak / AGE_MAX)))
        attempts = sum(
            len(channel["attempt_ids"])
            for channel in result["channels"] if channel["t"] < deadline
        )
        communication_scores.append(max(0.0, min(1.0, 1.0 - attempts / 80.0)))
    return {
        "primary_metrics_1": sum(coverage_scores) / len(coverage_scores),
        "primary_metrics_2": sum(efficiency_scores) / len(efficiency_scores),
        "auxiliary_metrics_1": sum(freshness_scores) / len(freshness_scores),
        "auxiliary_metrics_2": sum(communication_scores) / len(communication_scores),
    }


def _replay_atoms(sc: Mapping[str, Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    names = [id.split(".", 1)[1] for id in ATOM_SHARES["replay_validity"]]
    passed = {name: True for name in names}
    details = {name: "independent replay passed" for name in names}
    replayed: dict[str, Any] = {}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    validators = {
        "robots": _valid_robot,
        "belief": _valid_belief,
        "frontiers": _valid_frontier,
        "messages": _valid_message,
        "channels": _valid_channel,
        "coverage": _valid_coverage,
    }
    for name in SCENARIO_NAMES:
        if name not in sc:
            for rule in names:
                fail(rule, f"{name}: missing scenario")
            continue
        data = sc[name]
        typed = all(
            isinstance(data.get(f"{field}_rows"), list)
            and all(validator(row) for row in data[f"{field}_rows"])
            for field, validator in validators.items()
        )
        if not typed:
            for rule in names:
                fail(rule, f"{name}: complete typed evidence required")
            continue
        expected = _replay_scenario(name, data)
        replayed[name] = expected
        for field, rule, key_fn in (
            ("robots", "robots", lambda row: (row["robot"], row["state_t"])),
            ("belief", "belief", lambda row: (row["receiver"], row["sender"], row["state_t"])),
            ("frontiers", "frontiers", lambda row: row["claim_sequence"]),
            ("messages", "messages", lambda row: (row["attempt_id"], row["receiver"])),
            ("channels", "channels", lambda row: row["t"]),
            ("coverage", "coverage", lambda row: row["state_t"]),
        ):
            actual_rows = data[f"{field}_rows"]
            actual = {key_fn(row): row for row in actual_rows}
            wanted = {key_fn(row): row for row in expected[field]}
            if set(actual) != set(wanted):
                fail(rule, f"{name}.{field}: key set mismatch")
                continue
            for key, wanted_row in wanted.items():
                mismatch = _first_mismatch(
                    actual[key], wanted_row, f"{name}.{field}[{key}]"
                )
                if mismatch is not None:
                    fail(rule, mismatch)
                    break
        try:
            values = _metrics({name: expected}) if len(SCENARIO_NAMES) == 1 else None
            if values is not None and any(not _finite(value) for value in values.values()):
                fail("metrics", f"{name}: non-finite metric")
        except Exception as exc:
            fail("metrics", f"{name}: metric replay failed: {exc}")
    if len(replayed) == len(SCENARIO_NAMES):
        try:
            values = _metrics(replayed)
            if set(values) != {"primary_metrics_1", "primary_metrics_2", "auxiliary_metrics_1", "auxiliary_metrics_2"} or any(not _finite(value) or not 0 <= value <= 1 for value in values.values()):
                fail("metrics", "metric names or ranges")
        except Exception as exc:
            fail("metrics", f"metric replay failed: {exc}")
    else:
        fail("metrics", "all scenarios required")
    return [
        _atom("replay_validity", name, passed[name], details[name])
        for name in names
    ], replayed


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs mismatch")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("invalid atom shares")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("atom shares must sum to one")
    if any(not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share mismatch")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(atom.id for atom in failed) + "]; " + " | ".join(
        f"{atom.id}: {atom.detail}" for atom in failed
    )


def _report_score(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md is not a regular non-symlink file"
    text = ws.read_text("report.md") or ""
    required = (
        "Problem Formulation", "Method Design", "Experimental Setup", "Results",
        "Research Analysis", "Conclusion and Limitations",
    )
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.MULTILINE))
    found: list[str] = []
    for index, heading in enumerate(headings):
        title = heading.group(2)
        if title not in required:
            continue
        level = len(heading.group(1))
        end = len(text)
        for later in headings[index + 1:]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        prose = "\n".join(
            line for line in text[heading.end():end].splitlines()
            if not re.match(r"^#{1,6}\s+", line)
        )
        if prose.strip():
            found.append(title)
    missing = [title for title in required if title not in found]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    return score, f"missing_sections={missing}; normalized={score:.9g}"


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    scenarios = root and isinstance(data.get("scenarios"), list)
    names = scenarios and all(
        isinstance(row, Mapping) and isinstance(row.get("scenario"), str)
        for row in data["scenarios"]
    )
    keys = scenarios and all(_scenario_shape(row) for row in data["scenarios"])
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root and set(data) == {"scenarios"}, "root key set"),
        _atom("result_json", "scenarios", bool(scenarios), "scenarios list"),
        _atom("result_json", "scenario_keys", bool(names and keys), "scenario record keys"),
    ]


def _validity_gate(scores: Mapping[str, float], official_ok: bool = True) -> bool:
    return official_ok and all(scores.get(name, 0.0) >= 1 - EPS for name in GATE_CHECKS)


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = False
    try:
        try:
            result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(result, "exit_code", None) == 0
        except Exception as exc:
            state.official_message = f"could not remove old results.json: {exc}"
        started = time.monotonic()
        state.run_started = True
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            state.timed_out = state.exit_code is None
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError)
        finally:
            state.run_finished = True
        state.elapsed_s = time.monotonic() - started
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok and state.exit_code == 0 and not state.timed_out
            and state.elapsed_s <= 600 + EPS and state.result_file_is_new_regular
        )
        state.official_message = state.official_message or (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
            f"new_regular_results={state.result_file_is_new_regular}"
        )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except Exception as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = _diagnose(atoms)
        if not state.official_ok:
            state.scores["result_json"] = 0.0
            state.diagnostics["result_json"] = "official run failed; result_json is gated to zero"
        if not state.official_ok or not state.parse_ok:
            for name in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[name] = 0.0
                state.diagnostics[name] = "official run or strict parse failed"
            _CACHE[key] = state
            return state
        state.records = _index_records(state.data)
        state.scenarios = state.records
        evidence = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        coverage_score, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage_score
        state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:20]}"
        legality = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose(legality)
        replay, replayed = _replay_atoms(state.scenarios)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose(replay)
        state.validity_gate = _validity_gate(state.scores, state.official_ok)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.scenarios)
    except Exception as exc:
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        state.official_message = state.official_message or detail
        state.metrics = None
        state.validity_gate = False
        for name in GATE_CHECKS:
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    if getattr(result, "exit_code", None) == 0:
        return 1.0, "static compilation passed"
    return 0.0, "static compilation failed"


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if name.startswith(("primary_metrics_", "auxiliary_metrics_")):
        if state.metrics is None:
            return 0.0, f"{name}: validity gate not satisfied"
        return state.metrics[name], f"{name}: independently replayed value={state.metrics[name]:.9g}"
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "")


for _id in (
    "result_json", "result_evidence", "canonical_coverage", "decision_legality",
    "replay_validity", "primary_metrics_1", "primary_metrics_2",
    "auxiliary_metrics_1", "auxiliary_metrics_2",
):
    async def _check(ws: Workspace, _name: str = _id) -> tuple[float, str]:
        return await _score(ws, _name)

    globals()[_id] = check(id=_id)(_check)

del _check, _id
