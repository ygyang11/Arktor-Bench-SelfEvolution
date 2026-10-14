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

EPS = 1e-6
HORIZON = 180
PERIOD = 30
PERIOD_COUNT = 6
QUEUE_CAPACITY = 12
PACKET_LIMIT = 120
HOP_LIMIT = 3
DEADLINE_WINDOW = 45
INITIAL_ENERGY = 220.0
TRAVEL_COST = 1.0
STAY_COST = 0.05
SEND_COST = 0.2

ROVERS = ("R1", "R2", "R3")
NODES = (*ROVERS, "B")
SCENARIO_NAMES = (
    "balanced_loop",
    "long_bridge",
    "relay_gap",
    "priority_shift",
    "intermittent_base",
)
SCENARIO_INDEX = {name: index for index, name in enumerate(SCENARIO_NAMES)}
PHI = {
    "balanced_loop": 0,
    "long_bridge": 1,
    "relay_gap": 2,
    "priority_shift": 0,
    "intermittent_base": 1,
}
VERTICES = tuple(f"v_{x}{y}" for x in range(6) for y in range(4))
VERTEX_SET = set(VERTICES)
SENSING_VERTICES = tuple(f"v_{x}{y}" for x in range(6) for y in range(1, 4))
SENSING_SET = set(SENSING_VERTICES)
BASE_VERTEX = "v_00"
STARTS = {"R1": "v_01", "R2": "v_02", "R3": "v_03"}
PAIR_ORDER = (
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R3"),
    ("R1", "B"),
    ("R2", "B"),
    ("R3", "B"),
)
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_ORDER)}
BLOCKED_OFFSETS = {
    "balanced_loop": {
        ("R1", "R2"): {5, 17},
        ("R1", "R3"): {11},
        ("R2", "R3"): {23},
        ("R1", "B"): {2, 20},
        ("R2", "B"): {14},
        ("R3", "B"): {8, 26},
    },
    "long_bridge": {
        ("R1", "R2"): {3, 19},
        ("R1", "R3"): {7},
        ("R2", "R3"): {13, 28},
        ("R1", "B"): {1, 16},
        ("R2", "B"): {9},
        ("R3", "B"): {21, 27},
    },
    "relay_gap": {
        ("R1", "R2"): {4},
        ("R1", "R3"): {10, 24},
        ("R2", "R3"): {18},
        ("R1", "B"): {6, 22},
        ("R2", "B"): {15},
        ("R3", "B"): {0, 25},
    },
    "priority_shift": {
        ("R1", "R2"): {1, 14},
        ("R1", "R3"): {6},
        ("R2", "R3"): {20, 29},
        ("R1", "B"): {5},
        ("R2", "B"): {12, 24},
        ("R3", "B"): {17},
    },
    "intermittent_base": {
        ("R1", "R2"): {2, 16},
        ("R1", "R3"): {9},
        ("R2", "R3"): {21, 27},
        ("R1", "B"): {4, 19},
        ("R2", "B"): {1, 14},
        ("R3", "B"): {6, 23},
    },
}


def _vertex_xy(vertex: str) -> tuple[int, int]:
    return int(vertex[2]), int(vertex[3])


def _adjacent(first: str, second: str) -> bool:
    x1, y1 = _vertex_xy(first)
    x2, y2 = _vertex_xy(second)
    return abs(x1 - x2) + abs(y1 - y2) == 1


EDGES = tuple(
    (first, second)
    for index, first in enumerate(VERTICES)
    for second in VERTICES[index + 1 :]
    if _adjacent(first, second)
)

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.moves": 0.5,
        "result_evidence.transmission_proposals": 0.5,
    },
    "decision_legality": {
        "decision_legality.movement_state": 0.35,
        "decision_legality.movement_energy": 0.20,
        "decision_legality.proposal_key": 0.15,
        "decision_legality.proposal_contact": 0.30,
    },
    "replay_validity": {
        "replay_validity.motion": 0.15,
        "replay_validity.visits": 0.15,
        "replay_validity.contacts": 0.15,
        "replay_validity.custody": 0.30,
        "replay_validity.resources": 0.25,
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
        self.records: dict[str, dict[str, Any]] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any, lower: int, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _shape(value: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object"
    if set(value) != keys:
        return f"{path}: expected keys={sorted(keys)!r}, actual={sorted(value)!r}"
    return None


def _scenario_error(value: Any, path: str) -> str | None:
    return _shape(value, {"scenario", "moves", "transmission_proposals"}, path)


def _move_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "robot", "destination"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["robot"] not in ROVERS:
        return f"{path}.robot: expected one of {ROVERS!r}"
    destination = value["destination"]
    if destination is not None and (
        not isinstance(destination, str) or destination not in VERTEX_SET
    ):
        return f"{path}.destination: expected null or a canonical vertex"
    return None


_RECORD_ID = re.compile(r"^(v_[0-5][1-3])@([0-9]{3})$")


def _record_id_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    match = _RECORD_ID.fullmatch(value)
    return bool(match and int(match.group(2)) < HORIZON)


def _proposal_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "src", "dst", "record_ids", "priority"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["src"] not in ROVERS:
        return f"{path}.src: expected one of {ROVERS!r}"
    if value["dst"] not in NODES or value["dst"] == value["src"]:
        return f"{path}.dst: expected a different rover or B"
    record_ids = value["record_ids"]
    if (
        not isinstance(record_ids, list)
        or not 1 <= len(record_ids) <= 3
        or not all(_record_id_valid(item) for item in record_ids)
        or len(record_ids) != len(set(record_ids))
    ):
        return f"{path}.record_ids: expected one to three distinct canonical record IDs"
    if not _integer(value["priority"], 0, 9):
        return f"{path}.priority: expected integer 0..9"
    return None


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9, rel_tol=0.0):
        raise ValueError("atom shares must sum to one")
    if any(
        not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12, rel_tol=0.0)
        for atom in atoms
    ):
        raise ValueError("atom share differs from declaration")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[{}]; {}".format(
        ",".join(atom.id for atom in failed),
        " | ".join(f"{atom.id}: {atom.detail}" for atom in failed),
    )


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    verdicts = {
        "result_json.parse": (parse_ok, "strict JSON parse"),
        "result_json.root": (
            isinstance(data, dict) and set(data) == {"scenarios"},
            "root must contain exactly scenarios",
        ),
        "result_json.scenarios": (isinstance(scenarios, list), "scenarios must be an array"),
        "result_json.records": (
            isinstance(scenarios, list)
            and all(
                _scenario_error(row, f"scenarios[{index}]") is None
                for index, row in enumerate(scenarios)
            ),
            "each scenario record must have the exact published keys",
        ),
    }
    return [
        AtomVerdict(atom_id, passed, detail, ATOM_SHARES["result_json"][atom_id])
        for atom_id, (passed, detail) in verdicts.items()
    ]


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in data["scenarios"]:
        if (
            isinstance(row, dict)
            and isinstance(row.get("scenario"), str)
            and row["scenario"] in SCENARIO_NAMES
        ):
            records.setdefault(row["scenario"], row)
    return records


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules: dict[str, list[Any]] = {
        "moves": [True, "movement fields are well formed"],
        "transmission_proposals": [True, "proposal fields are well formed"],
    }

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            continue
        for field, validator in (
            ("moves", _move_error),
            ("transmission_proposals", _proposal_error),
        ):
            rows = record.get(field)
            if not isinstance(rows, list):
                fail(field, f"{name}.{field}: expected array")
                continue
            for index, row in enumerate(rows):
                error = validator(row, f"{name}.{field}[{index}]")
                if error:
                    fail(field, error)
                    break
    return [
        AtomVerdict(
            f"result_evidence.{rule}",
            bool(verdict[0]),
            str(verdict[1]),
            ATOM_SHARES["result_evidence"][f"result_evidence.{rule}"],
        )
        for rule, verdict in rules.items()
    ]


def _hashable(value: Any) -> Hashable:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    try:
        hash(value)
    except Exception:  # noqa: BLE001
        return repr(value)
    return value


def _key_defects(label: str, actual: list[Hashable], expected: set[Hashable]) -> list[str]:
    defects: list[str] = []
    seen: set[Hashable] = set()
    duplicate = False
    for key in actual:
        duplicate = duplicate or key in seen
        seen.add(key)
    if duplicate:
        defects.append(f"{label}: duplicate canonical key")
    if expected - seen:
        defects.append(f"{label}: missing {len(expected - seen)} canonical keys")
    if seen - expected:
        defects.append(f"{label}: unknown {len(seen - expected)} canonical keys")
    return defects


def _coverage(data: Any) -> tuple[float, list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios: missing array"]
    rows = data["scenarios"]
    defects = _key_defects(
        "scenarios",
        [
            _hashable(row.get("scenario")) if isinstance(row, dict) else _hashable(row)
            for row in rows
        ],
        set(SCENARIO_NAMES),
    )
    expected_moves = {(slot, rover) for slot in range(HORIZON) for rover in ROVERS}
    for name in SCENARIO_NAMES:
        matches = [row for row in rows if isinstance(row, dict) and row.get("scenario") == name]
        if not matches:
            continue
        moves = matches[0].get("moves")
        if not isinstance(moves, list):
            defects.append(f"{name}.moves: missing array")
            continue
        defects.extend(
            _key_defects(
                f"{name}.moves",
                [
                    _hashable((row.get("slot"), row.get("robot")))
                    if isinstance(row, dict)
                    else _hashable(row)
                    for row in moves
                ],
                expected_moves,
            )
        )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _decision_map(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record.get("moves"), list) or not isinstance(
        record.get("transmission_proposals"), list
    ):
        return None
    moves: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["moves"]:
        if not isinstance(row, dict) or _move_error(row, "move"):
            continue
        key = (row["slot"], row["robot"])
        moves.setdefault(key, row)
    expected = {(slot, rover) for slot in range(HORIZON) for rover in ROVERS}
    for slot, rover in expected - set(moves):
        moves[(slot, rover)] = {"slot": slot, "robot": rover, "destination": None}
    proposals: dict[tuple[int, str], dict[str, Any]] = {}
    duplicate_proposals: list[tuple[int, str]] = []
    for row in record["transmission_proposals"]:
        if not isinstance(row, dict) or _proposal_error(row, "proposal"):
            continue
        key = (row["slot"], row["src"])
        if key in proposals:
            duplicate_proposals.append(key)
        else:
            proposals[key] = row
    return {
        "moves": moves,
        "proposals": proposals,
        "duplicate_proposals": duplicate_proposals,
    }


def _edge_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _narrow(first: str, second: str) -> bool:
    x1, y1 = _vertex_xy(first)
    x2, y2 = _vertex_xy(second)
    return y1 == y2 and {x1, x2} == {2, 3}


def _travel_duration(name: str, first: str, second: str) -> int:
    x1, y1 = _vertex_xy(first)
    x2, y2 = _vertex_xy(second)
    duration = 1 + ((x1 + y1 + x2 + y2 + PHI[name]) % 3)
    if _narrow(first, second) and name == "long_bridge":
        return 5
    if _narrow(first, second) and name == "relay_gap":
        return 4
    if name == "intermittent_base" and BASE_VERTEX in {first, second}:
        duration += 1
    return duration


def _priority(name: str, vertex: str, period: int) -> int:
    x, y = _vertex_xy(vertex)
    source_x = 5 - x if name == "priority_shift" and period >= 3 else x
    index = 4 * source_x + y
    value = 1 + ((index + period + PHI[name]) % 3)
    if name == "long_bridge" and x >= 3:
        value = min(3, value + 1)
    return value


def _pair(first: str, second: str) -> tuple[str, str]:
    for pair in PAIR_ORDER:
        if {first, second} == set(pair):
            return pair
    raise ValueError(f"invalid node pair: {first},{second}")


def _interference(name: str, slot: int, first: str, second: str) -> bool:
    pair = _pair(first, second)
    offset = slot % PERIOD
    if name == "intermittent_base" and "B" in pair and offset in {8, 9, 10, 11, 12, 13}:
        return True
    return offset in BLOCKED_OFFSETS[name][pair]


def _base_contact(
    name: str,
    first: str,
    second: str,
    positions: Mapping[str, str | None],
) -> bool:
    first_vertex = BASE_VERTEX if first == "B" else positions[first]
    second_vertex = BASE_VERTEX if second == "B" else positions[second]
    if first_vertex is None or second_vertex is None:
        return False
    x1, y1 = _vertex_xy(first_vertex)
    x2, y2 = _vertex_xy(second_vertex)
    if math.dist((2 * x1, 2 * y1), (2 * x2, 2 * y2)) > 3.0 + EPS:
        return False
    if name == "relay_gap" and (x1 <= 2 < x2 or x2 <= 2 < x1):
        return {first_vertex, second_vertex} == {"v_20", "v_30"}
    return True


def _contacts(
    name: str,
    slot: int,
    positions: Mapping[str, str | None],
) -> dict[tuple[str, str], bool]:
    available: dict[tuple[str, str], bool] = {}
    for first, second in PAIR_ORDER:
        value = _base_contact(name, first, second, positions) and not _interference(
            name, slot, first, second
        )
        available[(first, second)] = value
        available[(second, first)] = value
    return available


def _record_id(location: str, visit_slot: int) -> str:
    return f"{location}@{visit_slot:03d}"


def _local_views(
    name: str,
    slot: int,
    positions: Mapping[str, str | None],
    transit: Mapping[str, Mapping[str, Any] | None],
    energies: Mapping[str, float],
    queues: Mapping[str, list[str]],
    records: Mapping[str, Mapping[str, Any]],
    seen: Mapping[str, set[str]],
    known_last_visit: Mapping[str, Mapping[str, int]],
    priorities: Mapping[str, int],
    contacts: Mapping[tuple[str, str], bool],
) -> dict[str, dict[str, Any]]:
    travel_durations = {
        f"{first}|{second}": _travel_duration(name, first, second)
        for first, second in EDGES
    }
    views: dict[str, dict[str, Any]] = {}
    for rover in ROVERS:
        queue_rows = [
            {
                "record_id": record_id,
                "location": records[record_id]["location"],
                "visit_slot": records[record_id]["visit_slot"],
                "priority": records[record_id]["priority"],
                "deadline": records[record_id]["deadline"],
                "hop_count": records[record_id]["hop_count"],
                "queue_available": records[record_id]["queue_available"],
            }
            for record_id in queues[rover]
        ]
        views[rover] = {
            "node": rover,
            "slot": slot,
            "vertex": positions[rover],
            "transit": dict(transit[rover]) if transit[rover] is not None else None,
            "energy": energies[rover],
            "queue": queue_rows,
            "seen_record_ids": sorted(seen[rover]),
            "known_last_visit": dict(known_last_visit[rover]),
            "priorities": dict(priorities),
            "available_contacts": [
                node for node in NODES if node != rover and contacts.get((rover, node), False)
            ],
            "travel_durations": travel_durations,
            "sensing_vertices": list(SENSING_VERTICES),
        }
    return views


def _simulate(
    name: str,
    decisions: Mapping[str, Any] | None = None,
    provider: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if name not in SCENARIO_NAMES or (decisions is None) == (provider is None):
        raise ValueError("provide exactly one valid scenario and decision source")

    positions: dict[str, str | None] = dict(STARTS)
    transit: dict[str, dict[str, Any] | None] = {rover: None for rover in ROVERS}
    energies = {rover: INITIAL_ENERGY for rover in ROVERS}
    queues: dict[str, list[str]] = {rover: [] for rover in ROVERS}
    seen: dict[str, set[str]] = {rover: set() for rover in ROVERS}
    confirmed: set[str] = set()
    known_last_visit = {
        rover: {vertex: -1 for vertex in SENSING_VERTICES} for rover in ROVERS
    }
    last_visit = {vertex: -1 for vertex in SENSING_VERTICES}
    records: dict[str, dict[str, Any]] = {}
    arrivals: dict[int, dict[str, Any]] = {}
    packets: list[dict[str, Any]] = []
    hops: list[dict[str, Any]] = []
    visits: list[dict[str, Any]] = []
    expiration_events: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    period_covered: set[tuple[str, int]] = set()
    move_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    idle_max = 0.0
    selected_packet_count = 0
    eligible_contact_slots = 0
    productive_contact_slots = 0
    successful_novel_hops = 0
    duplicate_arrivals = 0
    record_arrivals = 0
    motion_robot_slots = 0
    unproductive_robot_slots = 0
    collision_robot_slots = 0
    violations: dict[str, list[str]] = {
        "movement_state": [],
        "movement_energy": [],
        "proposal_contact": [],
    }
    replay_violations: dict[str, list[str]] = {
        "visits": [],
        "custody": [],
        "resources": [],
    }
    invariants = {
        "motion": True,
        "visits": True,
        "contacts": True,
        "custody": True,
        "resources": True,
    }

    def fail(rule: str, detail: str) -> None:
        violations[rule].append(detail)

    def replay_fail(rule: str, detail: str) -> None:
        replay_violations[rule].append(detail)

    def destination_seen(node: str, record_id: str) -> bool:
        return record_id in (confirmed if node == "B" else seen[node])

    def terminate(record_id: str, reason: str) -> None:
        record = records[record_id]
        record["holder"] = None
        record["terminal"] = reason

    def admit_batch(node: str, batch: list[str], boundary: int) -> None:
        novel = [record_id for record_id in batch if record_id not in seen[node]]
        for record_id in novel:
            seen[node].add(record_id)
            record = records[record_id]
            known_last_visit[node][record["location"]] = max(
                known_last_visit[node][record["location"]], record["visit_slot"]
            )
        accepted = [
            record_id for record_id in novel if records[record_id]["hop_count"] < HOP_LIMIT
        ]
        for record_id in novel:
            if records[record_id]["hop_count"] >= HOP_LIMIT:
                terminate(record_id, "hop_limit")
        while len(queues[node]) + len(accepted) > QUEUE_CAPACITY:
            terminate(queues[node].pop(0), "overflow")
        for record_id in accepted:
            records[record_id]["holder"] = node
            records[record_id]["queue_available"] = boundary
            queues[node].append(record_id)

    def process_arrival(boundary: int) -> set[str]:
        nonlocal productive_contact_slots
        nonlocal successful_novel_hops
        nonlocal duplicate_arrivals
        nonlocal record_arrivals
        received_novel: set[str] = set()
        packet = arrivals.pop(boundary, None)
        if packet is None:
            return received_novel
        accepted_by_rover: list[str] = []
        productive = False
        dst = packet["dst"]
        for record_id, hop in zip(packet["record_ids"], packet["hops"], strict=True):
            record_arrivals += 1
            record = records[record_id]
            if boundary > record["deadline"]:
                hop["outcome"] = "late"
                terminate(record_id, "late")
                continue
            if dst == "B":
                if record_id in confirmed:
                    duplicate_arrivals += 1
                    hop["outcome"] = "duplicate"
                    terminate(record_id, "duplicate")
                else:
                    confirmed.add(record_id)
                    record["base_arrival"] = boundary
                    hop["outcome"] = "confirmed"
                    terminate(record_id, "confirmed")
                    successful_novel_hops += 1
                    productive = True
                continue
            if record_id in seen[dst]:
                duplicate_arrivals += 1
                hop["outcome"] = "duplicate"
                terminate(record_id, "duplicate")
                continue
            if record["hop_count"] >= HOP_LIMIT:
                seen[dst].add(record_id)
                known_last_visit[dst][record["location"]] = max(
                    known_last_visit[dst][record["location"]], record["visit_slot"]
                )
                hop["outcome"] = "hop_limit"
                terminate(record_id, "hop_limit")
                continue
            hop["outcome"] = "accepted"
            accepted_by_rover.append(record_id)
            received_novel.add(dst)
            successful_novel_hops += 1
            productive = True
        if dst != "B" and accepted_by_rover:
            admit_batch(dst, accepted_by_rover, boundary)
        if productive and packet["eligible_at_send"]:
            productive_contact_slots += 1
        return received_novel

    def expire(boundary: int) -> None:
        for rover in ROVERS:
            kept: list[str] = []
            for record_id in queues[rover]:
                if records[record_id]["deadline"] < boundary:
                    expiration_events.append(
                        {"boundary": boundary, "record_id": record_id}
                    )
                    terminate(record_id, "expired")
                else:
                    kept.append(record_id)
            queues[rover] = kept

    def check_custody() -> None:
        nonlocal invariants
        queue_ids = [record_id for rover in ROVERS for record_id in queues[rover]]
        inflight_ids = [
            record_id
            for packet in arrivals.values()
            for record_id in packet["record_ids"]
        ]
        if (
            any(len(queues[rover]) > QUEUE_CAPACITY for rover in ROVERS)
            or len(queue_ids + inflight_ids) != len(set(queue_ids + inflight_ids))
        ):
            invariants["custody"] = False
        for rover in ROVERS:
            for record_id in queues[rover]:
                if records[record_id]["holder"] != rover or records[record_id]["terminal"]:
                    invariants["custody"] = False

    for slot in range(HORIZON):
        received_novel = process_arrival(slot)
        expire(slot)
        priorities = {
            vertex: _priority(name, vertex, slot // PERIOD) for vertex in SENSING_VERTICES
        }
        available = _contacts(name, slot, positions)
        for vertex in SENSING_VERTICES:
            weighted = priorities[vertex] / 3.0 * (slot - last_visit[vertex])
            idle_max = max(idle_max, weighted)

        if selected_packet_count < PACKET_LIMIT:
            for src in ROVERS:
                if positions[src] is None or not queues[src]:
                    continue
                head = queues[src][0]
                for dst in NODES:
                    if (
                        dst != src
                        and available.get((src, dst), False)
                        and not destination_seen(dst, head)
                    ):
                        eligible_contact_slots += 1

        views = _local_views(
            name,
            slot,
            positions,
            transit,
            energies,
            queues,
            records,
            seen,
            known_last_visit,
            priorities,
            available,
        )
        cycle_moves: dict[str, Any] = {}
        cycle_proposals: dict[str, dict[str, Any]] = {}
        if provider is not None:
            for rover in ROVERS:
                supplied = provider(slot, views[rover])
                cycle_moves[rover] = supplied.get("destination")
                proposal = supplied.get("proposal")
                if proposal is not None:
                    cycle_proposals[rover] = {
                        "slot": slot,
                        "src": rover,
                        "dst": proposal.get("dst"),
                        "record_ids": list(proposal.get("record_ids", [])),
                        "priority": proposal.get("priority"),
                    }
        else:
            assert decisions is not None
            cycle_moves = {
                rover: decisions["moves"][(slot, rover)]["destination"] for rover in ROVERS
            }
            cycle_proposals = {
                src: row
                for (row_slot, src), row in decisions["proposals"].items()
                if row_slot == slot
            }

        for rover in ROVERS:
            move_rows.append(
                {"slot": slot, "robot": rover, "destination": cycle_moves[rover]}
            )
        proposal_rows.extend(
            dict(cycle_proposals[rover]) for rover in ROVERS if rover in cycle_proposals
        )

        valid_entry: dict[str, tuple[str, str, int] | None] = {rover: None for rover in ROVERS}
        for rover in ROVERS:
            destination = cycle_moves[rover]
            has_proposal = rover in cycle_proposals
            if transit[rover] is not None:
                if destination is not None:
                    fail(
                        "movement_state",
                        f"{name}.moves[{slot},{rover}]: destination must be null in transit",
                    )
                if energies[rover] + EPS < TRAVEL_COST:
                    fail(
                        "movement_energy",
                        f"{name}.moves[{slot},{rover}]: insufficient reserved transit energy",
                    )
                continue
            source = positions[rover]
            assert source is not None
            if destination is None:
                reserve = STAY_COST + (SEND_COST if has_proposal else 0.0)
                if energies[rover] + EPS < reserve:
                    fail(
                        "movement_energy",
                        f"{name}.moves[{slot},{rover}]: stay and proposal reserve exceeds energy",
                    )
                continue
            if not _adjacent(source, destination):
                fail(
                    "movement_state",
                    f"{name}.moves[{slot},{rover}]: destination is not adjacent to {source}",
                )
                continue
            duration = _travel_duration(name, source, destination)
            reserve = duration * TRAVEL_COST + (SEND_COST if has_proposal else 0.0)
            if energies[rover] + EPS < reserve:
                fail(
                    "movement_energy",
                    f"{name}.moves[{slot},{rover}]: edge completion reserve requires {reserve:g}",
                )
                continue
            valid_entry[rover] = (source, destination, duration)

        valid_proposals: list[dict[str, Any]] = []
        for src in ROVERS:
            row = cycle_proposals.get(src)
            if row is None:
                continue
            record_ids = row["record_ids"]
            if any(record_id not in records for record_id in record_ids):
                replay_fail(
                    "visits",
                    f"{name}.transmission_proposals[{slot},{src}]: unknown or unreleased record ID",
                )
                continue
            dst = row["dst"]
            if positions[src] is None or not available.get((src, dst), False):
                fail(
                    "proposal_contact",
                    f"{name}.transmission_proposals[{slot},{src}]: source must be at a "
                    "vertex and dst currently available",
                )
                continue
            if any(
                records[record_id]["deadline"] < slot
                or records[record_id]["hop_count"] >= HOP_LIMIT
                for record_id in record_ids
            ):
                replay_fail(
                    "resources",
                    f"{name}.transmission_proposals[{slot},{src}]: expired or hop-limited token",
                )
                continue
            prefix = queues[src][: len(record_ids)]
            if prefix != record_ids:
                replay_fail(
                    "custody",
                    f"{name}.transmission_proposals[{slot},{src}]: record_ids must be "
                    "the current FIFO prefix",
                )
                continue
            valid_proposals.append(row)

        selected: dict[str, Any] | None = None
        if selected_packet_count < PACKET_LIMIT and valid_proposals:
            selected = min(
                valid_proposals,
                key=lambda row: (
                    -row["priority"],
                    row["src"],
                    row["dst"],
                    row["record_ids"][0],
                ),
            )
            src = selected["src"]
            dst = selected["dst"]
            record_ids = list(selected["record_ids"])
            eligible_at_send = bool(
                record_ids and not destination_seen(dst, record_ids[0])
            )
            packet_id = f"pkt_{selected_packet_count:03d}"
            packet_hops: list[dict[str, Any]] = []
            for record_id in record_ids:
                popped = queues[src].pop(0)
                if popped != record_id:
                    invariants["custody"] = False
                record = records[record_id]
                record["total_queue_wait"] += slot - record["queue_available"]
                record["hop_count"] += 1
                record["holder"] = None
                hop = {
                    "packet_id": packet_id,
                    "packet_size": len(record_ids),
                    "record_id": record_id,
                    "send_slot": slot,
                    "arrival": slot + 1,
                    "src": src,
                    "dst": dst,
                    "outcome": None,
                }
                hops.append(hop)
                packet_hops.append(hop)
            packet = {
                "packet_id": packet_id,
                "source_slot": slot,
                "arrival": slot + 1,
                "src": src,
                "dst": dst,
                "record_ids": record_ids,
                "hops": packet_hops,
                "eligible_at_send": eligible_at_send,
            }
            packets.append(packet)
            arrivals[slot + 1] = packet
            selected_packet_count += 1
            energies[src] -= SEND_COST

        rejected: set[str] = set()
        existing = [state for state in transit.values() if state is not None]
        for rover, request in valid_entry.items():
            if request is None or not _narrow(request[0], request[1]):
                continue
            for state in existing:
                if (
                    _edge_key(request[0], request[1]) == _edge_key(state["src"], state["dst"])
                    and request[0] == state["dst"]
                    and request[1] == state["src"]
                ):
                    rejected.add(rover)
        for index, first in enumerate(ROVERS):
            first_request = valid_entry[first]
            if first_request is None or not _narrow(first_request[0], first_request[1]):
                continue
            for second in ROVERS[index + 1 :]:
                second_request = valid_entry[second]
                if second_request is None:
                    continue
                if first_request[0] == second_request[1] and first_request[1] == second_request[0]:
                    rejected.update((first, second))

        arrived: dict[str, str] = {}
        moved_this_slot: set[str] = set()
        for rover in ROVERS:
            state = transit[rover]
            if state is not None:
                energies[rover] -= TRAVEL_COST
                motion_robot_slots += 1
                moved_this_slot.add(rover)
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    positions[rover] = state["dst"]
                    arrived[rover] = state["dst"]
                    transit[rover] = None
                continue
            request = valid_entry[rover]
            if request is None or rover in rejected:
                energies[rover] -= STAY_COST
                if rover in rejected:
                    collision_robot_slots += 1
                    collision_events.append({"slot": slot, "robot": rover})
                continue
            source, destination, duration = request
            energies[rover] -= TRAVEL_COST
            motion_robot_slots += 1
            moved_this_slot.add(rover)
            if duration == 1:
                positions[rover] = destination
                arrived[rover] = destination
            else:
                positions[rover] = None
                transit[rover] = {
                    "src": source,
                    "dst": destination,
                    "remaining": duration - 1,
                }

        visit_winners: set[str] = set()
        for vertex in SENSING_VERTICES:
            entrants = [rover for rover in ROVERS if arrived.get(rover) == vertex]
            if not entrants:
                continue
            winner = min(entrants, key=ROVERS.index)
            visit_winners.add(winner)
            last_visit[vertex] = slot
            period_covered.add((vertex, slot // PERIOD))
            record_id = _record_id(vertex, slot)
            if record_id in records:
                invariants["visits"] = False
                continue
            record = {
                "record_id": record_id,
                "winner": winner,
                "location": vertex,
                "visit_slot": slot,
                "priority": priorities[vertex],
                "deadline": slot + DEADLINE_WINDOW,
                "hop_count": 0,
                "holder": winner,
                "queue_available": slot + 1,
                "base_arrival": None,
                "terminal": None,
                "total_queue_wait": 0,
            }
            records[record_id] = record
            visits.append(record)
            seen[winner].add(record_id)
            known_last_visit[winner][vertex] = slot
            while len(queues[winner]) + 1 > QUEUE_CAPACITY:
                terminate(queues[winner].pop(0), "overflow")
            queues[winner].append(record_id)

        selected_sender = selected["src"] if selected is not None else None
        for rover in ROVERS:
            if (
                rover not in moved_this_slot
                and rover not in visit_winners
                and rover != selected_sender
                and rover not in received_novel
            ):
                unproductive_robot_slots += 1

        occupancy: list[tuple[str, str]] = []
        for state in transit.values():
            if state is not None and _narrow(state["src"], state["dst"]):
                occupancy.append((state["src"], state["dst"]))
        if any(
            first[0] == second[1] and first[1] == second[0]
            for index, first in enumerate(occupancy)
            for second in occupancy[index + 1 :]
        ):
            invariants["motion"] = False
        if any(
            not _finite(energies[rover])
            or energies[rover] < -EPS
            or energies[rover] > INITIAL_ENERGY + EPS
            for rover in ROVERS
        ):
            invariants["resources"] = False
        if selected_packet_count > PACKET_LIMIT or any(
            record["hop_count"] > HOP_LIMIT for record in records.values()
        ):
            invariants["resources"] = False
        if selected is not None and not available.get((selected["src"], selected["dst"]), False):
            invariants["contacts"] = False
        check_custody()

    process_arrival(HORIZON)
    check_custody()
    for record in records.values():
        expected = _record_id(record["location"], record["visit_slot"])
        if (
            record["record_id"] != expected
            or record["deadline"] != record["visit_slot"] + DEADLINE_WINDOW
            or record["priority"]
            != _priority(name, record["location"], record["visit_slot"] // PERIOD)
        ):
            invariants["visits"] = False
        if record["base_arrival"] is not None and record["base_arrival"] > record["deadline"]:
            invariants["resources"] = False
    if arrivals:
        invariants["contacts"] = False

    return {
        "scenario": name,
        "positions": positions,
        "transit": transit,
        "energies": energies,
        "queues": queues,
        "seen": seen,
        "confirmed": confirmed,
        "last_visit": last_visit,
        "records": records,
        "visits": visits,
        "packets": packets,
        "hops": hops,
        "expiration_events": expiration_events,
        "collision_events": collision_events,
        "period_covered": period_covered,
        "idle_max": idle_max,
        "selected_packet_count": selected_packet_count,
        "eligible_contact_slots": eligible_contact_slots,
        "productive_contact_slots": productive_contact_slots,
        "successful_novel_hops": successful_novel_hops,
        "duplicate_arrivals": duplicate_arrivals,
        "record_arrivals": record_arrivals,
        "motion_robot_slots": motion_robot_slots,
        "unproductive_robot_slots": unproductive_robot_slots,
        "collision_robot_slots": collision_robot_slots,
        "energy_used": sum(INITIAL_ENERGY - energies[rover] for rover in ROVERS),
        "violations": violations,
        "replay_violations": replay_violations,
        "invariants": invariants,
        "record": {
            "scenario": name,
            "moves": move_rows,
            "transmission_proposals": proposal_rows,
        },
    }


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules: dict[str, list[Any]] = {
        "movement_state": [True, "movement state and topology are legal"],
        "movement_energy": [True, "movement and proposal reserves are legal"],
        "proposal_key": [True, "proposal node-slot keys are unique"],
        "proposal_contact": [True, "proposal sources and contacts are legal"],
    }

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    for name in SCENARIO_NAMES:
        record = records.get(name)
        decisions = _decision_map(record) if record is not None else None
        if decisions is None:
            continue
        if decisions["duplicate_proposals"]:
            fail(
                "proposal_key",
                f"{name}: duplicate proposal key {decisions['duplicate_proposals'][0]!r}",
            )
        replayed = _simulate(name, decisions=decisions)
        for rule, details in replayed["violations"].items():
            if details:
                fail(rule, details[0])
    return [
        AtomVerdict(
            f"decision_legality.{rule}",
            bool(verdict[0]),
            str(verdict[1]),
            ATOM_SHARES["decision_legality"][f"decision_legality.{rule}"],
        )
        for rule, verdict in rules.items()
    ]


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules: dict[str, list[Any]] = {
        "motion": [True, "motion and narrow-edge state are consistent"],
        "visits": [True, "visit winners, priorities, and record metadata are consistent"],
        "contacts": [True, "contacts, arbitration, and arrivals are consistent"],
        "custody": [True, "FIFO capacity and single custody are preserved"],
        "resources": [True, "energy, hop, packet, and deadline bounds are preserved"],
    }
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    for name in SCENARIO_NAMES:
        record = records.get(name)
        decisions = _decision_map(record) if record is not None else None
        if decisions is None:
            continue
        state = _simulate(name, decisions=decisions)
        replayed[name] = state
        for rule, details in state["replay_violations"].items():
            if details:
                fail(rule, details[0])
        for rule, passed in state["invariants"].items():
            if not passed:
                fail(rule, f"{name}: fixed replay invariant failed")
    atoms = [
        AtomVerdict(
            f"replay_validity.{rule}",
            bool(verdict[0]),
            str(verdict[1]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{rule}"],
        )
        for rule, verdict in rules.items()
    ]
    return atoms, replayed


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scenario_metrics(state: Mapping[str, Any]) -> dict[str, float]:
    coverage = len(state["period_covered"]) / (len(SENSING_VERTICES) * PERIOD_COUNT)
    primary_1 = 0.6 * _clip(1.0 - state["idle_max"] / HORIZON) + 0.4 * coverage

    records = list(state["records"].values())
    total_priority = sum(record["priority"] for record in records)
    on_time = [
        record
        for record in records
        if record["base_arrival"] is not None
        and record["base_arrival"] <= record["deadline"]
    ]
    delivered_priority = sum(record["priority"] for record in on_time)
    fresh_priority = sum(
        record["priority"]
        * _clip(
            1.0
            - (record["base_arrival"] - record["visit_slot"]) / DEADLINE_WINDOW
        )
        for record in on_time
    )
    if total_priority:
        primary_2 = (
            0.65 * delivered_priority / total_priority
            + 0.35 * fresh_priority / total_priority
        )
    else:
        primary_2 = 0.0

    primary_3 = coverage * (
        0.25 * _clip(1.0 - state["motion_robot_slots"] / (len(ROVERS) * HORIZON))
        + 0.25
        * _clip(1.0 - state["unproductive_robot_slots"] / (len(ROVERS) * HORIZON))
        + 0.25
        * _clip(1.0 - state["collision_robot_slots"] / (len(ROVERS) * HORIZON))
        + 0.25 * _clip(1.0 - state["energy_used"] / (len(ROVERS) * INITIAL_ENERGY))
    )

    on_time_ids = {record["record_id"] for record in on_time}
    useful_bits = sum(
        32.0 + 64.0 / hop["packet_size"]
        for hop in state["hops"]
        if hop["record_id"] in on_time_ids
    )
    sent_bits = 160.0 * state["selected_packet_count"]
    bit_score = useful_bits / sent_bits if sent_bits else 0.0
    eligible = state["eligible_contact_slots"]
    service_score = state["productive_contact_slots"] / eligible if eligible else 0.0
    auxiliary_1 = 0.6 * bit_score + 0.4 * service_score

    waits = [record["total_queue_wait"] for record in on_time]
    wait_score = _clip(1.0 - sum(waits) / len(waits) / DEADLINE_WINDOW) if waits else 0.0
    hop_count = len(state["hops"])
    hop_score = state["successful_novel_hops"] / hop_count if hop_count else 0.0
    arrivals = state["record_arrivals"]
    redundancy_score = 1.0 - state["duplicate_arrivals"] / arrivals if arrivals else 0.0
    auxiliary_2 = 0.4 * wait_score + 0.3 * hop_score + 0.3 * redundancy_score

    return {
        "primary_metrics_1": _clip(primary_1),
        "primary_metrics_2": _clip(primary_2),
        "primary_metrics_3": _clip(primary_3),
        "auxiliary_metrics_1": _clip(auxiliary_1),
        "auxiliary_metrics_2": _clip(auxiliary_2),
    }


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all scenario replays are required for metrics")
    scenario_metrics = {name: _scenario_metrics(replayed[name]) for name in SCENARIO_NAMES}
    return {
        metric: sum(scenario_metrics[name][metric] for name in SCENARIO_NAMES)
        / len(SCENARIO_NAMES)
        for metric in scenario_metrics[SCENARIO_NAMES[0]]
    }


def _headings(text: str) -> dict[str, str]:
    raw_lines = text.splitlines()
    lines: list[str] = []
    in_html_comment = False
    for raw_line in raw_lines:
        remainder = raw_line
        kept = ""
        while remainder:
            if in_html_comment:
                end = remainder.find("-->")
                if end < 0:
                    remainder = ""
                    continue
                in_html_comment = False
                remainder = remainder[end + 3 :]
                continue
            start = remainder.find("<!--")
            if start < 0:
                kept += remainder
                remainder = ""
                continue
            kept += remainder[:start]
            in_html_comment = True
            remainder = remainder[start + 4 :]
        lines.append(kept)
    headings: list[tuple[int, int, int, str]] = []
    fence_char: str | None = None
    fence_length = 0
    for line_index, line in enumerate(lines):
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)
        fence = re.match(r"(`{3,}|~{3,})", stripped) if indent <= 3 else None
        if fence_char is not None:
            if (
                fence
                and fence.group(1)[0] == fence_char
                and len(fence.group(1)) >= fence_length
                and stripped[len(fence.group(1)) :].strip() == ""
            ):
                fence_char = None
                fence_length = 0
            continue
        if fence:
            fence_char = fence.group(1)[0]
            fence_length = len(fence.group(1))
            continue
        match = re.match(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*$", line)
        if match:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
            if title:
                headings.append((line_index, line_index + 1, len(match.group(1)), title))
            continue
        setext = re.match(r"^[ \t]{0,3}(=+|-+)[ \t]*$", line)
        if not setext or line_index == 0:
            continue
        previous = lines[line_index - 1]
        previous_stripped = previous.strip()
        if (
            not previous_stripped
            or len(previous) - len(previous.lstrip(" \t")) > 3
            or re.match(r"^[ \t]{0,3}(#{1,6})[ \t]+", previous)
            or re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", previous)
        ):
            continue
        headings.append(
            (
                line_index - 1,
                line_index + 1,
                1 if setext.group(1)[0] == "=" else 2,
                previous_stripped,
            )
        )
    occurrences: dict[str, list[str]] = {}
    for index, (line_index, content_start, level, title) in enumerate(headings):
        end = len(lines)
        for next_line, _, next_level, _ in headings[index + 1 :]:
            if next_level <= level:
                end = next_line
                break
        occurrences.setdefault(title, []).append("\n".join(lines[content_start:end]).strip())
    return {
        title: "\n\n".join(body for body in bodies if body)
        for title, bodies in occurrences.items()
    }


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
    score = max(0.0, (2.0 - 0.5 * len(missing)) / 2.0)
    return score, f"missing_sections={missing!r}; normalized={score:.9g}"


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


_CACHE: dict[str, EvalState] = {}


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
            state.official_message = (
                f"could not remove previous results.json: {type(exc).__name__}: {exc}"
            )
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = (
                state.exit_code is None
                or "timeout" in stderr.lower()
                or "timed out" in stderr.lower()
            )
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = (
                isinstance(exc, (asyncio.TimeoutError, TimeoutError))
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.timed_out = state.timed_out or state.elapsed_s >= 600.0 - EPS
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        if not state.official_message:
            if state.official_ok:
                state.official_message = (
                    f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                    f"new_regular_results={state.result_file_is_new_regular}"
                )
            else:
                state.official_message = (
                    f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                    f"timed_out={state.timed_out}, "
                    f"new_regular_results={state.result_file_is_new_regular}"
                )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except Exception as exc:  # noqa: BLE001
                state.diagnostics["result_json"] = (
                    f"results.json: strict parse failed: {type(exc).__name__}: {exc}"
                )
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            if not state.official_ok:
                state.scores["result_json"] = 0.0
                if state.parse_ok:
                    state.diagnostics["result_json"] = reason
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
        state.records = _index_records(state.data)
        evidence_atoms = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(
            evidence_atoms, ATOM_SHARES["result_evidence"]
        )
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
        coverage, defects = _coverage(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = (
            "failed_atoms=[]; defect_count=0"
            if not defects
            else f"defect_count={len(defects)}; " + " | ".join(defects[:8])
        )
        legality_atoms = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(
            legality_atoms, ATOM_SHARES["decision_legality"]
        )
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
        replay_atoms, replayed = _replay_atoms(state.records)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(
            replay_atoms, ATOM_SHARES["replay_validity"]
        )
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
    except Exception as exc:  # noqa: BLE001
        state.timed_out = state.timed_out or (
            isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
        )
        message = f"grader analysis failed: {type(exc).__name__}: {exc}"
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, message)
    finally:
        state.run_finished = True
        _CACHE[key] = state
    return state


async def _metric_check(ws: Workspace, metric: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{metric}: validity gate not satisfied; metric not computed"
    value = state.metrics.get(metric, 0.0)
    return value, f"{metric}: independently replayed value={value:.9g}"


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected a regular file"
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"run.py: static compilation raised {type(exc).__name__}: {exc}"
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
        return 0.0, "report.md: expected a readable nonempty regular file"
    try:
        return _report_score(ws.read_text("report.md"))
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_json", 0.0), state.diagnostics.get("result_json", "")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_evidence", 0.0), state.diagnostics.get(
        "result_evidence", ""
    )


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
    return state.scores.get("replay_validity", 0.0), state.diagnostics.get(
        "replay_validity", ""
    )


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_2")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_3")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_2")
