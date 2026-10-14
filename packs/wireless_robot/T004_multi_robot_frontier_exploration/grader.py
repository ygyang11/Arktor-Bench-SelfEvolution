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
GRID_SIZE = 8
HORIZON = 32
ROBOT_COUNT = 3
COMM_RANGE = 2
SYNC_BUDGET = 24
BURST_WINDOW = 4
BURST_LIMIT = 3
EPS = 1e-6
MOVES = ("north", "east", "south", "west", "stay")
MOVE_DELTA = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
    "stay": (0, 0),
}
ENVIRONMENTS = (
    "split_corridor",
    "central_ring",
    "branching_rooms",
    "offset_cross",
    "gated_loop",
)
LAYOUTS: dict[str, tuple[str, ...]] = {
    "split_corridor": (
        "########",
        "#...#..#",
        "#...#..#",
        "#......#",
        "#...#..#",
        "#...#..#",
        "#...#..#",
        "########",
    ),
    "central_ring": (
        "########",
        "#......#",
        "#..##..#",
        "#..##..#",
        "#..##..#",
        "#..##..#",
        "#......#",
        "########",
    ),
    "branching_rooms": (
        "########",
        "#...#..#",
        "#...#..#",
        "#......#",
        "#..#...#",
        "#...#..#",
        "#...#..#",
        "########",
    ),
    "offset_cross": (
        "########",
        "#......#",
        "#..##..#",
        "#......#",
        "#..##..#",
        "#......#",
        "#......#",
        "########",
    ),
    "gated_loop": (
        "########",
        "#..#...#",
        "#..#...#",
        "#......#",
        "#...#..#",
        "#...#..#",
        "#......#",
        "########",
    ),
}
STARTS: dict[str, tuple[tuple[int, int], ...]] = {
    "split_corridor": ((1, 1), (6, 1), (1, 6)),
    "central_ring": ((1, 1), (6, 1), (1, 6)),
    "branching_rooms": ((1, 1), (6, 1), (6, 6)),
    "offset_cross": ((1, 1), (6, 1), (3, 6)),
    "gated_loop": ((1, 1), (6, 1), (1, 6)),
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
        "result_evidence.map_encoding": 1.0 / 8.0,
        "result_evidence.sync_evidence": 1.0 / 8.0,
        "result_evidence.record_relations": 1.0 / 8.0,
    },
    "decision_legality": {
        "decision_legality.time_domain": 1.0 / 8.0,
        "decision_legality.robot_domain": 1.0 / 8.0,
        "decision_legality.move_domain": 1.0 / 8.0,
        "decision_legality.sync_domain": 1.0 / 8.0,
        "decision_legality.packet_count": 1.0 / 8.0,
        "decision_legality.burst_limit": 1.0 / 8.0,
        "decision_legality.unique_decisions": 1.0 / 8.0,
        "decision_legality.decision_container": 1.0 / 8.0,
    },
    "replay_validity": {
        "replay_validity.packet_admission": 1.0 / 10.0,
        "replay_validity.sensor_update": 1.0 / 10.0,
        "replay_validity.sync_events": 1.0 / 10.0,
        "replay_validity.pre_positions": 1.0 / 10.0,
        "replay_validity.post_positions": 1.0 / 10.0,
        "replay_validity.map_codes": 1.0 / 10.0,
        "replay_validity.known_counts": 1.0 / 10.0,
        "replay_validity.blocked": 1.0 / 10.0,
        "replay_validity.collisions": 1.0 / 10.0,
        "replay_validity.map_values": 1.0 / 10.0,
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


def _layout_cell(environment: str, x: int, y: int) -> str:
    if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
        return "#"
    # The task displays rows from north to south; map codes use y=0 first.
    return LAYOUTS[environment][GRID_SIZE - 1 - y][x]


def _free_cells(environment: str) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        if _layout_cell(environment, x, y) == "."
    }


def _packet_delivered(environment_index: int, t: int, sender: int, receiver: int, distance: int) -> bool:
    return distance <= COMM_RANGE and (t + 3 * sender + 5 * receiver + environment_index) % 7 != 0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coordinate(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_integer(v) and 0 <= v < GRID_SIZE for v in value)
    )


def _close(a: Any, b: Any, tolerance: float = EPS) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tolerance


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique_pairs)


def _require_replay_decisions(decisions: Any) -> None:
    if not isinstance(decisions, Mapping) or set(decisions) != set(range(HORIZON)):
        raise ValueError("replay requires one valid decision record for every time slot")
    for t in range(HORIZON):
        if not _decision_valid(decisions[t]):
            raise ValueError(f"replay decision at t={t} is malformed or outside its declared domain")


def _map_code(values: Sequence[str]) -> str:
    return "".join(values)


def _decode_map_code(code: Any) -> list[str] | None:
    if not isinstance(code, str) or len(code) != GRID_SIZE * GRID_SIZE or any(c not in "?.#" for c in code):
        return None
    return list(code)


def _initial_maps(environment: str) -> tuple[list[list[str]], list[list[int]], list[list[int]]]:
    values = [["?"] * (GRID_SIZE * GRID_SIZE) for _ in range(ROBOT_COUNT)]
    stamps = [[-1] * (GRID_SIZE * GRID_SIZE) for _ in range(ROBOT_COUNT)]
    sources = [[ROBOT_COUNT] * (GRID_SIZE * GRID_SIZE) for _ in range(ROBOT_COUNT)]
    for robot, (x, y) in enumerate(STARTS[environment]):
        idx = y * GRID_SIZE + x
        values[robot][idx] = "."
        sources[robot][idx] = robot
    return values, stamps, sources


def _snapshot(values: list[list[str]], stamps: list[list[int]], sources: list[list[int]], robot: int) -> tuple[list[str], list[int], list[int]]:
    return list(values[robot]), list(stamps[robot]), list(sources[robot])


def _merge_snapshot(
    values: list[list[str]], stamps: list[list[int]], sources: list[list[int]], receiver: int,
    packet: tuple[int, int, list[str], list[int], list[int]],
) -> None:
    # Newer observation timestamps win; equal timestamps are resolved by the lower source index.
    sender, _receiver, incoming_values, incoming_stamps, incoming_sources = packet
    for idx in range(GRID_SIZE * GRID_SIZE):
        incoming_key = (incoming_stamps[idx], -incoming_sources[idx])
        current_key = (stamps[receiver][idx], -sources[receiver][idx])
        if incoming_key > current_key:
            values[receiver][idx] = incoming_values[idx]
            stamps[receiver][idx] = incoming_stamps[idx]
            sources[receiver][idx] = incoming_sources[idx]
        elif incoming_key == current_key and sender < sources[receiver][idx]:
            values[receiver][idx] = incoming_values[idx]
            sources[receiver][idx] = sender


def _replay_environment(environment: str, decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    _require_replay_decisions(decisions)
    environment_index = ENVIRONMENTS.index(environment)
    values, stamps, sources = _initial_maps(environment)
    positions = [tuple(point) for point in STARTS[environment]]
    pending: dict[int, list[tuple[int, int, list[str], list[int], list[int]]]] = defaultdict(list)
    trace: list[dict[str, Any]] = []
    collision_slots = 0
    sync_count = 0
    for t in range(HORIZON):
        # Packet admissions occur before sensing and decisions.
        for packet in sorted(pending.pop(t, []), key=lambda item: (item[0], item[1])):
            _merge_snapshot(values, stamps, sources, packet[1], packet)

        # Each robot senses its current cell and its four-neighbourhood.
        for robot, (x, y) in enumerate(positions):
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                sx, sy = x + dx, y + dy
                if 0 <= sx < GRID_SIZE and 0 <= sy < GRID_SIZE and abs(dx) + abs(dy) <= 1:
                    idx = sy * GRID_SIZE + sx
                    values[robot][idx] = _layout_cell(environment, sx, sy)
                    stamps[robot][idx] = t
                    sources[robot][idx] = robot

        rows = decisions.get(t, {})
        robot_rows = rows.get("robots", []) if isinstance(rows, Mapping) else []
        by_robot: dict[int, Mapping[str, Any]] = {}
        if isinstance(robot_rows, list):
            for row in robot_rows:
                if isinstance(row, Mapping) and _integer(row.get("robot")):
                    by_robot[row["robot"]] = row

        sync_events: list[dict[str, Any]] = []
        for sender in range(ROBOT_COUNT):
            row = by_robot.get(sender, {})
            target = row.get("sync_to") if isinstance(row, Mapping) else None
            if target is None:
                continue
            if not _integer(target) or target < 0 or target >= ROBOT_COUNT or target == sender:
                continue
            receiver = int(target)
            distance = abs(positions[sender][0] - positions[receiver][0]) + abs(positions[sender][1] - positions[receiver][1])
            in_range = distance <= COMM_RANGE
            delivered = _packet_delivered(environment_index, t, sender, receiver, distance)
            arrival = t + 1 if delivered and t + 1 < HORIZON else None
            sync_events.append({
                "sender": sender,
                "receiver": receiver,
                "in_range": in_range,
                "delivered": delivered,
                "arrival": arrival,
            })
            sync_count += 1
            if delivered and arrival is not None:
                snap_values, snap_stamps, snap_sources = _snapshot(values, stamps, sources, sender)
                pending[arrival].append((sender, receiver, snap_values, snap_stamps, snap_sources))

        proposed: list[tuple[int, int]] = []
        blocked = [False] * ROBOT_COUNT
        collision = [False] * ROBOT_COUNT
        for robot in range(ROBOT_COUNT):
            row = by_robot.get(robot, {})
            move = row.get("move") if isinstance(row, Mapping) else None
            dx, dy = MOVE_DELTA.get(move, (0, 0))
            target = (positions[robot][0] + dx, positions[robot][1] + dy)
            if _layout_cell(environment, target[0], target[1]) != ".":
                blocked[robot] = True
            proposed.append(target)

        # A pair exchanging current cells is a collision.  Other proposals into
        # an occupied current cell are blocked before same-target testing.
        for first in range(ROBOT_COUNT):
            for second in range(first + 1, ROBOT_COUNT):
                if proposed[first] == positions[second] and proposed[second] == positions[first]:
                    collision[first] = collision[second] = True
                elif not blocked[first] and proposed[first] == positions[second] and proposed[second] != positions[first]:
                    blocked[first] = True
                elif not blocked[second] and proposed[second] == positions[first] and proposed[first] != positions[second]:
                    blocked[second] = True

        # Same-target collisions apply only to a free target.  A target occupied
        # at the start of the slot has already been handled as a block above.
        current_positions = set(positions)
        for first in range(ROBOT_COUNT):
            for second in range(first + 1, ROBOT_COUNT):
                if (
                    not blocked[first]
                    and not blocked[second]
                    and not collision[first]
                    and not collision[second]
                    and proposed[first] == proposed[second]
                    and proposed[first] not in current_positions
                ):
                    collision[first] = collision[second] = True

        post_positions = [
            proposed[robot] if not blocked[robot] and not collision[robot] else positions[robot]
            for robot in range(ROBOT_COUNT)
        ]
        if any(collision):
            collision_slots += 1
        trace.append({
            "t": t,
            "positions": [[p[0], p[1]] for p in positions],
            "post_positions": [[p[0], p[1]] for p in post_positions],
            "map_codes": [_map_code(values[robot]) for robot in range(ROBOT_COUNT)],
            "known_counts": [sum(c != "?" for c in values[robot]) for robot in range(ROBOT_COUNT)],
            "blocked": blocked,
            "collision": collision,
            "sync_events": sync_events,
        })
        positions = post_positions

    final_values = [list(row) for row in values]
    union_known = {idx for idx in range(GRID_SIZE * GRID_SIZE) if any(final_values[r][idx] != "?" for r in range(ROBOT_COUNT))}
    t90 = None
    for index, row in enumerate(trace, 1):
        known = {idx for idx in range(GRID_SIZE * GRID_SIZE) if any(row["map_codes"][r][idx] != "?" for r in range(ROBOT_COUNT))}
        if len(known) >= math.ceil(0.90 * GRID_SIZE * GRID_SIZE):
            t90 = index
            break
    return {
        "trace": trace,
        "final_values": final_values,
        "final_union_known": union_known,
        "t90": t90,
        "collision_slots": collision_slots,
        "sync_count": sync_count,
    }


def _scenario_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"scenario", "decisions", "trace"}


def _decision_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"t", "robots"}


def _robot_decision_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"robot", "move", "sync_to"}


def _trace_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {
        "t", "positions", "post_positions", "map_codes", "known_counts", "blocked", "collision", "sync_events",
    }


def _sync_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"sender", "receiver", "in_range", "delivered", "arrival"}


def _decision_valid(row: Any) -> bool:
    if not _decision_shape(row) or not _integer(row.get("t")) or not 0 <= row["t"] < HORIZON:
        return False
    robots = row.get("robots")
    if not isinstance(robots, list) or len(robots) != ROBOT_COUNT:
        return False
    seen: set[int] = set()
    for robot in robots:
        if not _robot_decision_shape(robot) or not _integer(robot.get("robot")) or robot["robot"] in seen or not 0 <= robot["robot"] < ROBOT_COUNT:
            return False
        seen.add(robot["robot"])
        if robot.get("move") not in MOVES:
            return False
        target = robot.get("sync_to")
        if target is not None and (not _integer(target) or not 0 <= target < ROBOT_COUNT or target == robot["robot"]):
            return False
    return seen == set(range(ROBOT_COUNT))


def _trace_valid(row: Any) -> bool:
    if not _trace_shape(row) or not _integer(row.get("t")) or not 0 <= row["t"] < HORIZON:
        return False
    for field in ("positions", "post_positions", "map_codes", "known_counts", "blocked", "collision"):
        if not isinstance(row.get(field), list) or len(row[field]) != ROBOT_COUNT:
            return False
    if not all(_coordinate(point) for point in row["positions"] + row["post_positions"]):
        return False
    if not all(isinstance(code, str) and len(code) == GRID_SIZE * GRID_SIZE and all(c in "?.#" for c in code) for code in row["map_codes"]):
        return False
    if not all(_integer(value) and 0 <= value <= GRID_SIZE * GRID_SIZE for value in row["known_counts"]):
        return False
    if not all(isinstance(value, bool) for value in row["blocked"] + row["collision"]):
        return False
    events = row.get("sync_events")
    if not isinstance(events, list):
        return False
    for event in events:
        if not _sync_shape(event):
            return False
        if not _integer(event.get("sender")) or not 0 <= event["sender"] < ROBOT_COUNT:
            return False
        if not _integer(event.get("receiver")) or not 0 <= event["receiver"] < ROBOT_COUNT or event["receiver"] == event["sender"]:
            return False
        if not isinstance(event.get("in_range"), bool) or not isinstance(event.get("delivered"), bool):
            return False
        arrival = event.get("arrival")
        if arrival is not None and (not _integer(arrival) or not 1 <= arrival <= HORIZON):
            return False
    return True


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for record in data["scenarios"]:
        if not isinstance(record, dict) or not isinstance(record.get("scenario"), str):
            continue
        name = record["scenario"]
        if name in indexed:
            continue
        decisions_rows = record.get("decisions") if isinstance(record.get("decisions"), list) else []
        trace_rows = record.get("trace") if isinstance(record.get("trace"), list) else []
        decisions: dict[int, Any] = {}
        traces: dict[int, Any] = {}
        for row in decisions_rows:
            if _decision_shape(row) and _integer(row.get("t")) and 0 <= row["t"] < HORIZON:
                decisions.setdefault(row["t"], row)
        for row in trace_rows:
            if _trace_shape(row) and _integer(row.get("t")) and 0 <= row["t"] < HORIZON:
                traces.setdefault(row["t"], row)
        indexed[name] = {
            "record": record,
            "decisions_rows": decisions_rows,
            "trace_rows": trace_rows,
            "decisions": decisions,
            "traces": traces,
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
    root_keys = root and set(data) == {"scenarios"}
    scenarios = root_keys and isinstance(data.get("scenarios"), list)
    return [
        _atom("result_json", "parse", parse_ok, f"results.json: strict JSON parse; actual_type={type(data).__name__}"),
        _atom("result_json", "root", root_keys, f"results.json: expected keys={{'scenarios'}}; actual_keys={sorted(data) if isinstance(data, dict) else None}"),
        _atom("result_json", "scenarios", scenarios, f"results.json.scenarios: expected array; actual_type={type(data.get('scenarios')).__name__ if isinstance(data, dict) else None}"),
    ]


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {rule: True for rule in [key.removeprefix("result_evidence.") for key in ATOM_SHARES["result_evidence"]]}
    details = {rule: "" for rule in checks}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if not records:
        for rule in checks:
            fail(rule, "no scenario records available")
    for name, item in records.items():
        record = item["record"]
        if not _scenario_shape(record):
            fail("scenario_shape", f"{name}: expected keys scenario/decisions/trace")
        decisions_rows = item["decisions_rows"]
        traces_rows = item["trace_rows"]
        if not all(_decision_shape(row) for row in decisions_rows):
            fail("decision_shape", f"{name}.decisions: every row must have exactly t and robots")
        if not all(_decision_valid(row) for row in decisions_rows):
            fail("decision_types", f"{name}.decisions: robot records, moves and sync targets have invalid types or domains")
        if not all(_trace_shape(row) for row in traces_rows):
            fail("trace_shape", f"{name}.trace: every row has the declared trace keys")
        if not all(_trace_valid(row) for row in traces_rows):
            fail("trace_values", f"{name}.trace: positions, flags, counts or event fields have invalid types or ranges")
        if any(not isinstance(row, dict) or not all(isinstance(code, str) and len(code) == 64 and all(c in "?.#" for c in code) for code in row.get("map_codes", [])) for row in traces_rows):
            fail("map_encoding", f"{name}.trace.map_codes: expected three 64-character ?/.# encodings")
        if any(not isinstance(row, dict) or not all(_sync_shape(event) for event in row.get("sync_events", []) if isinstance(row.get("sync_events"), list)) for row in traces_rows):
            fail("sync_evidence", f"{name}.trace.sync_events: every event must expose sender, receiver, range, delivery and arrival")
        if any(not isinstance(row, dict) or not isinstance(row.get("robots"), list) for row in decisions_rows):
            fail("record_relations", f"{name}: decisions and trace containers are not associated lists")
    return [
        _atom("result_evidence", rule, checks[rule], details[rule] or "declared evidence fields passed")
        for rule in checks
    ]


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    names = [key.removeprefix("decision_legality.") for key in ATOM_SHARES["decision_legality"]]
    checks = {name: True for name in names}
    details = {name: "" for name in names}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if not records:
        for rule in names:
            fail(rule, "no scenario records available")
    for name, item in records.items():
        rows = item["decisions_rows"]
        if any(not _integer(row.get("t")) or not 0 <= row["t"] < HORIZON for row in rows if isinstance(row, dict)):
            fail("time_domain", f"{name}.decisions: t must be an integer in 0..31")
        for row in rows:
            robots = row.get("robots") if isinstance(row, dict) else None
            if not isinstance(robots, list) or len(robots) != ROBOT_COUNT:
                fail("robot_domain", f"{name}.decisions: exactly three robot records are required")
                continue
            robot_ids = [r.get("robot") for r in robots if isinstance(r, dict)]
            if (
                len(robot_ids) != ROBOT_COUNT
                or any(not _integer(robot_id) for robot_id in robot_ids)
                or set(robot_ids) != set(range(ROBOT_COUNT))
            ):
                fail("robot_domain", f"{name}.decisions[{row.get('t')}].robots: expected unique robot IDs 0,1,2")
            for robot in robots:
                if not isinstance(robot, dict):
                    continue
                if robot.get("move") not in MOVES:
                    fail("move_domain", f"{name}.decisions[{row.get('t')}]: move must be one of {MOVES}")
                target = robot.get("sync_to")
                if target is not None and (not _integer(target) or not 0 <= target < ROBOT_COUNT or target == robot.get("robot")):
                    fail("sync_domain", f"{name}.decisions[{row.get('t')}]: sync_to must name a distinct robot or null")
        counts = [0] * HORIZON
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("robots"), list):
                if _integer(row.get("t")) and 0 <= row["t"] < HORIZON:
                    counts[row["t"]] += sum(
                        1 for robot in row["robots"]
                        if isinstance(robot, dict) and robot.get("sync_to") is not None
                    )
        if sum(counts) > SYNC_BUDGET:
            fail("packet_count", f"{name}: generated synchronization packets={sum(counts)} exceeds budget {SYNC_BUDGET}")
        for start in range(HORIZON - BURST_WINDOW + 1):
            if sum(counts[start:start + BURST_WINDOW]) > BURST_LIMIT:
                fail("burst_limit", f"{name}: slots {start}..{start + BURST_WINDOW - 1} exceed {BURST_LIMIT} packets")
        time_values = [row.get("t") for row in rows if isinstance(row, dict) and _integer(row.get("t"))]
        if len(time_values) != len(rows) or len(set(time_values)) != len(time_values):
            fail("unique_decisions", f"{name}.decisions: duplicate t keys")
        if not isinstance(item["record"].get("decisions"), list):
            fail("decision_container", f"{name}.decisions: expected list")
    return [_atom("decision_legality", rule, checks[rule], details[rule] or "decision constraint passed") for rule in names]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios: missing or not a list"]
    rows = data["scenarios"]
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in ENVIRONMENTS}
    for index, row in enumerate(rows):
        name = row.get("scenario") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"scenarios[{index}]: unknown environment key")
        else:
            by_name[name].append(row)

    def count_keys(record: dict[str, Any], field: str, expected: set[Any], key_fn: Any, label: str) -> None:
        values = record.get(field)
        if not isinstance(values, list):
            defects.append(f"{label}: missing list")
            return
        counts = {key: 0 for key in expected}
        for index, value in enumerate(values):
            key = key_fn(value) if isinstance(value, dict) else None
            if key in counts:
                counts[key] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key")
        defects.extend(f"{label}: missing key {key}" for key, count in counts.items() if count == 0)
        defects.extend(f"{label}: duplicate key {key}" for key, count in counts.items() for _ in range(max(0, count - 1)))

    for name in ENVIRONMENTS:
        matches = by_name[name]
        if not matches:
            defects.append(f"environment {name}: missing record")
            continue
        defects.extend(f"environment {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(record, "decisions", set(range(HORIZON)), lambda value: value.get("t") if _integer(value.get("t")) else None, f"{name}.decisions")
        count_keys(record, "trace", set(range(HORIZON)), lambda value: value.get("t") if _integer(value.get("t")) else None, f"{name}.trace")
    score = max(0.0, 1.0 - 0.5 * len(defects) / 4.0)
    return score, defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return f"failed_atoms=[canonical_coverage.defect]; defect_count={len(defects)}; {shown}; normalized={score:.9g}"


def _replay_atoms(records: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    names = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    checks = {name: True for name in names}
    details = {name: "" for name in names}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    for environment in ENVIRONMENTS:
        if environment not in records:
            for rule in names:
                fail(rule, f"{environment}: canonical record missing")
            continue
        item = records[environment]
        decisions = item["decisions"]
        traces = item["traces"]
        if set(decisions) != set(range(HORIZON)) or any(not _decision_valid(decisions[t]) for t in range(HORIZON)):
            for rule in names:
                fail(rule, f"{environment}.decisions: complete valid decisions required for replay")
            continue
        try:
            replay = _replay_environment(environment, decisions)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            for rule in names:
                fail(rule, f"{environment}.decisions: replay input rejected: {type(exc).__name__}: {exc}")
            continue
        replayed[environment] = replay
        if set(traces) != set(range(HORIZON)) or any(not _trace_valid(traces[t]) for t in range(HORIZON)):
            for rule in names:
                fail(rule, f"{environment}.trace: complete valid trace required for replay")
            continue
        expected_trace = replay["trace"]
        for t in range(HORIZON):
            actual = traces[t]
            expected = expected_trace[t]
            if actual.get("positions") != expected["positions"]:
                fail("pre_positions", f"{environment}.trace[{t}].positions: expected={expected['positions']}; actual={actual.get('positions')}")
            if actual.get("post_positions") != expected["post_positions"]:
                fail("post_positions", f"{environment}.trace[{t}].post_positions: expected={expected['post_positions']}; actual={actual.get('post_positions')}")
            if actual.get("map_codes") != expected["map_codes"]:
                fail("map_codes", f"{environment}.trace[{t}].map_codes: fixed observation/merge map differs")
            if actual.get("known_counts") != expected["known_counts"]:
                fail("known_counts", f"{environment}.trace[{t}].known_counts: expected={expected['known_counts']}; actual={actual.get('known_counts')}")
            if actual.get("blocked") != expected["blocked"]:
                fail("blocked", f"{environment}.trace[{t}].blocked: expected={expected['blocked']}; actual={actual.get('blocked')}")
            if actual.get("collision") != expected["collision"]:
                fail("collisions", f"{environment}.trace[{t}].collision: expected={expected['collision']}; actual={actual.get('collision')}")
            if actual.get("sync_events") != expected["sync_events"]:
                fail("sync_events", f"{environment}.trace[{t}].sync_events: expected={expected['sync_events']}; actual={actual.get('sync_events')}")
        # Packet admission and sensor update are independently represented by the map sequence.
        if any(len(row.get("map_codes", [])) != ROBOT_COUNT for row in traces.values()):
            fail("packet_admission", f"{environment}: map admission evidence is incomplete")
        if any(any(code.count("?") + code.count(".") + code.count("#") != 64 for code in row.get("map_codes", [])) for row in traces.values()):
            fail("sensor_update", f"{environment}: map codes do not contain exactly 64 cells")
        if any(any(code not in ("?", ".", "#") for code in "".join(row.get("map_codes", []))) for row in traces.values()):
            fail("map_values", f"{environment}: map encoding contains an invalid value")
    atoms = [_atom("replay_validity", rule, checks[rule], details[rule] or "fixed model replay comparison passed") for rule in names]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]], records: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    coverage_values: list[float] = []
    speed_values: list[float] = []
    collisions = 0
    consensus_values: list[float] = []
    sync_count = 0
    for environment in ENVIRONMENTS:
        result = replayed[environment]
        coverage_values.append(len(result["final_union_known"]) / 64.0)
        t90 = result["t90"]
        speed_values.append(0.0 if t90 is None else max(0.0, (HORIZON - t90) / (HORIZON - 1)))
        collisions += result["collision_slots"]
        sync_count += result["sync_count"]
        final_maps = result["final_values"]
        pair_scores: list[float] = []
        for first in range(ROBOT_COUNT):
            for second in range(first + 1, ROBOT_COUNT):
                denominator = 0
                numerator = 0
                for idx in range(64):
                    a, b = final_maps[first][idx], final_maps[second][idx]
                    if a != "?" or b != "?":
                        denominator += 1
                        if a != "?" and a == b:
                            numerator += 1
                pair_scores.append(numerator / denominator if denominator else 0.0)
        consensus_values.append(sum(pair_scores) / len(pair_scores))
    metrics = {
        "coverage_quality": sum(coverage_values) / len(coverage_values),
        "completion_speed": sum(speed_values) / len(speed_values),
        "safe_movement": max(0.0, 1.0 - collisions / (len(ENVIRONMENTS) * HORIZON)),
        "map_consensus": sum(consensus_values) / len(consensus_values),
        "sync_efficiency": max(0.0, min(1.0, 1.0 - sync_count / (len(ENVIRONMENTS) * SYNC_BUDGET))),
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _headings(text: str) -> dict[str, str]:
    required = {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"}
    lines = text.splitlines()
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not match or match.group(1).strip() not in required:
            continue
        title = match.group(1).strip()
        level = len(line) - len(line.lstrip("#"))
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
    required = ("Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations")
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    return max(0.0, (2.0 - 0.5 * len(missing)) / 2.0), f"missing_sections={missing}"


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    started = time.monotonic()
    remove_ok = False
    try:
        state.run_started = True
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(removed, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
            state.official_message = f"results.json removal failed: {type(exc).__name__}: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            state.timed_out = state.exit_code is None and "timeout" in str(getattr(result, "stderr", "")).lower()
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok and state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600.0 + EPS and state.result_file_is_new_regular
        )
        if not state.official_message:
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok
                else f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
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
        records = _index_records(state.data)
        state.records = records
        evidence_atoms = _evidence_atoms(records)
        state.scores["result_evidence"] = _score_atoms(evidence_atoms, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
        coverage_score, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage_score
        state.diagnostics["canonical_coverage"] = _coverage_diagnostic(coverage_score, defects)
        legality_atoms = _legality_atoms(records)
        state.scores["decision_legality"] = _score_atoms(legality_atoms, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
        replay_atoms, replayed = _replay_atoms(records)
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
    return (1.0, "run.py: static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "run.py: static compilation failed")


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
    return await _metric_check(ws, "coverage_quality")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "completion_speed")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "safe_movement")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "map_consensus")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "sync_efficiency")
