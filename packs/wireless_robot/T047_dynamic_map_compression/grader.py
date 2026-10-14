from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

GRID_SIZE = 16
HORIZON = 100
QUEUE_CAPACITY = 1024
SCENE_BIT_BUDGET = 30720
SENDER_INITIAL_ENERGY = 80.0
NAVIGATOR_INITIAL_ENERGY = 120.0
SENDER_ENERGY_PER_BIT = 0.002
EPS = 1e-6
MOVES = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "stay": (0, 0),
}
MOVE_ORDER = ("N", "E", "S", "W")


def _cells(*groups: tuple[str, Sequence[int]]) -> frozenset[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for axis, values in groups:
        if axis.startswith("x="):
            x = int(axis[2:])
            result.update((x, y) for y in values)
        else:
            y = int(axis[2:])
            result.update((x, y) for x in values)
    return frozenset(result)


SCENARIOS: dict[str, dict[str, Any]] = {
    "static_narrow_passage": {
        "phase": 1,
        "start": (1, 1),
        "goal": (14, 14),
        "deadline": 64,
        "obstacles": _cells(
            ("x=7", (*range(0, 6), *range(7, 16))),
            ("x=11", range(9, 12)),
        ),
        "paths": (
            ((6, 6), (7, 6), (8, 6), (8, 7), (8, 8), (8, 7)),
            ((10, 12), (11, 12), (12, 12), (13, 12), (12, 12), (11, 12)),
        ),
        "base_capacity": 48,
    },
    "moving_crossing": {
        "phase": 4,
        "start": (1, 13),
        "goal": (14, 2),
        "deadline": 72,
        "obstacles": _cells(
            ("x=5", range(3, 12)),
            ("x=10", range(4, 13)),
            ("y=7", range(6, 10)),
        ),
        "paths": (
            (
                (7, 11),
                (8, 11),
                (9, 11),
                (9, 10),
                (9, 9),
                (8, 9),
                (7, 9),
                (6, 9),
                (6, 10),
                (6, 11),
            ),
            ((11, 3), (12, 3), (13, 3), (12, 3)),
        ),
        "base_capacity": 48,
    },
    "bandwidth_burst": {
        "phase": 7,
        "start": (1, 8),
        "goal": (14, 8),
        "deadline": 68,
        "obstacles": _cells(
            ("x=5", (*range(0, 7), *range(9, 16))),
            ("x=10", (*range(0, 6), *range(8, 16))),
        ),
        "paths": (
            ((4, 7), (5, 7), (6, 7), (6, 8), (6, 9), (6, 8)),
            ((9, 6), (10, 6), (11, 6), (11, 7), (11, 8), (11, 7)),
        ),
        "base_capacity": 48,
    },
    "innovation_shift": {
        "phase": 10,
        "start": (2, 2),
        "goal": (13, 13),
        "deadline": 78,
        "obstacles": _cells(
            ("x=3", range(5, 14)),
            ("x=9", range(1, 10)),
            ("y=10", range(10, 15)),
        ),
        "paths": (
            ((4, 4), (5, 4), (6, 4), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (4, 5)),
            ((10, 11), (11, 11), (12, 11), (12, 12), (12, 13), (11, 13), (10, 13), (10, 12)),
        ),
        "base_capacity": 48,
    },
    "long_range_update": {
        "phase": 13,
        "start": (1, 14),
        "goal": (14, 1),
        "deadline": 86,
        "obstacles": _cells(
            ("x=6", (*range(0, 5), *range(6, 16))),
            ("x=11", range(2, 14)),
        ),
        "paths": (
            ((5, 5), (6, 5), (7, 5), (7, 6), (7, 7), (7, 8), (6, 8), (5, 8), (5, 7), (5, 6)),
            ((10, 1), (11, 1), (12, 1), (13, 1), (13, 2), (13, 3), (12, 3), (12, 2)),
        ),
        "base_capacity": 36,
    },
}
SCENARIO_NAMES = tuple(SCENARIOS)


ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.3333333333333333,
        "result_json.root": 0.3333333333333333,
        "result_json.scenarios": 0.3333333333333333,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 0.16666666666666666,
        "result_evidence.state_types": 0.16666666666666666,
        "result_evidence.slot_types": 0.16666666666666666,
        "result_evidence.packet_types": 0.16666666666666666,
        "result_evidence.map_types": 0.16666666666666666,
        "result_evidence.mission_types": 0.16666666666666666,
    },
    "decision_legality": {
        "decision_legality.move_domain": 0.16666666666666666,
        "decision_legality.packet_reference": 0.16666666666666666,
        "decision_legality.node_selection": 0.16666666666666666,
        "decision_legality.packet_encoding": 0.16666666666666666,
        "decision_legality.queue_budget": 0.16666666666666666,
        "decision_legality.energy": 0.16666666666666666,
    },
    "replay_validity": {
        "replay_validity.navigator_state": 0.16666666666666666,
        "replay_validity.sender_packet_service": 0.16666666666666666,
        "replay_validity.receiver_map": 0.16666666666666666,
        "replay_validity.collision_goal": 0.16666666666666666,
        "replay_validity.mission_resources": 0.16666666666666666,
        "replay_validity.event_summaries": 0.16666666666666666,
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


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _nullable_integer(value: Any, low: int, high: int) -> bool:
    return value is None or (_integer(value) and low <= value <= high)


def _nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _close(actual: Any, expected: Any) -> bool:
    return _finite_number(actual) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=EPS
    )


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_pairs)


def _dynamic_positions(name: str, slot: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scenario = SCENARIOS[name]
    phase = int(scenario["phase"])
    return tuple(path[(slot + phase) % len(path)] for path in scenario["paths"])  # type: ignore[return-value]


def _true_occupied(name: str, cell: tuple[int, int], slot: int) -> bool:
    return cell in SCENARIOS[name]["obstacles"] or cell in _dynamic_positions(name, slot)


def _probability_code(occupied: bool) -> int:
    return 3686 if occupied else 410


def _capacity(name: str, slot: int) -> int:
    scenario = SCENARIOS[name]
    phase = int(scenario["phase"])
    if (11 * slot + 5 * phase) % 23 == 0:
        return 0
    if name == "bandwidth_burst" and 36 <= slot <= 48:
        return 0
    return int(scenario["base_capacity"]) * (1 + ((7 * slot + 3 * phase) % 4))


def _lost(name: str, complete_slot: int) -> bool:
    return (13 * complete_slot + 4 * int(SCENARIOS[name]["phase"])) % 31 == 0


def _safe_cells(name: str) -> tuple[tuple[int, int], ...]:
    scenario = SCENARIOS[name]
    result = {scenario["start"], scenario["goal"]}
    for path in scenario["paths"]:
        for x, y in path:
            for dx, dy in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
                cell = (x + dx, y + dy)
                if 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE:
                    result.add(cell)
    return tuple(sorted(result, key=lambda cell: (cell[1], cell[0])))


def _safe_weight(name: str, cell: tuple[int, int]) -> int:
    paths = SCENARIOS[name]["paths"]
    return 2 if any(cell in path for path in paths) else 1


def _parse_node(node: Any) -> tuple[int, str, int] | None:
    if not isinstance(node, str):
        return None
    match = re.fullmatch(r"([0-4]):([0-3]*)", node)
    if match is None:
        return None
    depth = int(match.group(1))
    path = match.group(2)
    if len(path) != depth:
        return None
    value = int(path, 4) if path else 0
    node_id = (4**depth - 1) // 3 + value
    return depth, path, node_id


def _node_region(node: str) -> tuple[tuple[int, int], ...]:
    parsed = _parse_node(node)
    if parsed is None:
        return ()
    _, path, _ = parsed
    x0, x1, y0, y1 = 0, GRID_SIZE - 1, 0, GRID_SIZE - 1
    for digit in path:
        xmid = (x0 + x1 + 1) // 2
        ymid = (y0 + y1 + 1) // 2
        if digit == "0":
            x1, y0 = xmid - 1, ymid
        elif digit == "1":
            x0, y0 = xmid, ymid
        elif digit == "2":
            x1, y1 = xmid - 1, ymid - 1
        else:
            x0, y1 = xmid, ymid - 1
    return tuple((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def _node_code(name: str, node: str, slot: int) -> int:
    region = _node_region(node)
    if not region:
        raise ValueError(f"invalid node {node!r}")
    occupied = sum(_true_occupied(name, cell, slot) for cell in region)
    count = len(region)
    numerator = 2 * 4095 * (count + 8 * occupied) + 10 * count
    return numerator // (20 * count)


def _prefix_free(nodes: Sequence[str]) -> bool:
    parsed = [_parse_node(node) for node in nodes]
    if any(item is None for item in parsed):
        return False
    paths = [item[1] for item in parsed if item is not None]
    return len(paths) == len(set(paths)) and not any(
        first != second and second.startswith(first) for first in paths for second in paths
    )


def _shape_error(value: Any, expected: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object; actual_type={type(value).__name__}"
    if set(value) != expected:
        return f"{path}: expected keys={sorted(expected)}; actual_keys={sorted(value)}"
    return None


def _state_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "slot",
            "x",
            "y",
            "sender_energy",
            "navigator_energy",
            "goal_reached",
            "collision",
        },
        path,
    )
    if error:
        return error
    for key in ("slot", "x", "y"):
        if not _integer(value[key]):
            return f"{path}.{key}: expected integer; actual={value[key]!r}"
    if not 0 <= value["slot"] <= HORIZON:
        return f"{path}.slot: expected integer in [0,100]; actual={value['slot']!r}"
    if not 0 <= value["x"] < GRID_SIZE or not 0 <= value["y"] < GRID_SIZE:
        return (
            f"{path}.x/y: expected grid coordinates in [0,15]; actual={[value['x'], value['y']]!r}"
        )
    for key in ("sender_energy", "navigator_energy"):
        if not _finite_number(value[key]) or not -EPS <= float(value[key]) <= 120.0 + EPS:
            return f"{path}.{key}: expected finite value in [0,120]; actual={value[key]!r}"
    for key in ("goal_reached", "collision"):
        if not isinstance(value[key], bool):
            return f"{path}.{key}: expected boolean; actual={value[key]!r}"
    return None


def _slot_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"slot", "move", "packet_id"}, path)
    if error:
        return error
    if not _integer(value["slot"]) or not 0 <= value["slot"] < HORIZON:
        return f"{path}.slot: expected integer in [0,99]; actual={value['slot']!r}"
    if not isinstance(value["move"], str):
        return f"{path}.move: expected string; actual={value['move']!r}"
    if not _nullable_string(value["packet_id"]):
        return f"{path}.packet_id: expected string or null; actual={value['packet_id']!r}"
    return None


def _entry_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"node", "node_id", "code"}, path)
    if error:
        return error
    if not isinstance(value["node"], str):
        return f"{path}.node: expected string; actual={value['node']!r}"
    if not _integer(value["node_id"]) or not 0 <= value["node_id"] <= 340:
        return f"{path}.node_id: expected integer in [0,340]; actual={value['node_id']!r}"
    if not _integer(value["code"]) or not 0 <= value["code"] <= 4095:
        return f"{path}.code: expected integer in [0,4095]; actual={value['code']!r}"
    return None


def _packet_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "packet_id",
            "created_slot",
            "entries",
            "bits",
            "queue_enter_slot",
            "service_start_slot",
            "service_complete_slot",
            "arrival_slot",
            "outcome",
        },
        path,
    )
    if error:
        return error
    if not isinstance(value["packet_id"], str):
        return f"{path}.packet_id: expected string; actual={value['packet_id']!r}"
    if not _integer(value["created_slot"]) or not 0 <= value["created_slot"] < HORIZON:
        return f"{path}.created_slot: expected integer in [0,99]; actual={value['created_slot']!r}"
    entries = value["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
        length = len(entries) if isinstance(entries, list) else None
        return f"{path}.entries: expected array length in [1,16]; actual_length={length}"
    for index, entry in enumerate(entries):
        error = _entry_error(entry, f"{path}.entries[{index}]")
        if error:
            return error
    if not _integer(value["bits"]) or value["bits"] <= 0:
        return f"{path}.bits: expected positive integer; actual={value['bits']!r}"
    for key in ("queue_enter_slot", "service_start_slot", "service_complete_slot"):
        if not _nullable_integer(value[key], 0, HORIZON - 1):
            return f"{path}.{key}: expected null or integer in [0,99]; actual={value[key]!r}"
    if not _nullable_integer(value["arrival_slot"], 1, HORIZON):
        return (
            f"{path}.arrival_slot: expected null or integer in [1,100]; "
            f"actual={value['arrival_slot']!r}"
        )
    if value["outcome"] not in {"delivered", "lost", "unfinished"}:
        return f"{path}.outcome: expected declared enum; actual={value['outcome']!r}"
    expected_id = f"pkt-{value['created_slot']:03d}"
    if value["packet_id"] != expected_id:
        return f"{path}.packet_id: expected={expected_id!r}; actual={value['packet_id']!r}"
    if value["queue_enter_slot"] != value["created_slot"]:
        return (
            f"{path}.queue_enter_slot: expected={value['created_slot']}; "
            f"actual={value['queue_enter_slot']!r}"
        )
    start = value["service_start_slot"]
    complete = value["service_complete_slot"]
    arrival = value["arrival_slot"]
    if start is not None and start < value["created_slot"]:
        return f"{path}.service_start_slot: expected >= created_slot; actual={start}"
    if complete is not None and (start is None or complete < start):
        return f"{path}.service_complete_slot: expected >= non-null start; actual={complete}"
    if value["outcome"] == "delivered":
        if complete is None or arrival != complete + 1:
            return f"{path}.arrival_slot: expected completion+1 for delivered; actual={arrival!r}"
    elif value["outcome"] == "lost":
        if complete is None or arrival is not None:
            return f"{path}: lost requires completion and null arrival"
    elif complete is not None or arrival is not None:
        return f"{path}: unfinished requires null completion and arrival"
    return None


def _cell_error(value: Any, path: str, slot: int) -> str | None:
    error = _shape_error(value, {"x", "y", "code", "source_slot", "source_kind"}, path)
    if error:
        return error
    for key in ("x", "y", "code", "source_slot"):
        if not _integer(value[key]):
            return f"{path}.{key}: expected integer; actual={value[key]!r}"
    if not 0 <= value["x"] < GRID_SIZE or not 0 <= value["y"] < GRID_SIZE:
        return f"{path}.x/y: expected grid coordinates in [0,15]"
    if not 0 <= value["code"] <= 4095:
        return f"{path}.code: expected integer in [0,4095]; actual={value['code']!r}"
    if not -1 <= value["source_slot"] <= slot:
        return (
            f"{path}.source_slot: expected integer in [-1,{slot}]; actual={value['source_slot']!r}"
        )
    if value["source_kind"] not in {"initial", "local", "remote"}:
        return f"{path}.source_kind: expected declared enum; actual={value['source_kind']!r}"
    return None


def _map_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"slot", "cells"}, path)
    if error:
        return error
    if not _integer(value["slot"]) or not 0 <= value["slot"] < HORIZON:
        return f"{path}.slot: expected integer in [0,99]; actual={value['slot']!r}"
    if not isinstance(value["cells"], list):
        return f"{path}.cells: expected array; actual_type={type(value['cells']).__name__}"
    for index, cell in enumerate(value["cells"]):
        error = _cell_error(cell, f"{path}.cells[{index}]", value["slot"])
        if error:
            return error
    return None


def _mission_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "goal_arrival_slot",
            "collision_slots",
            "sent_bits",
            "energy_used",
            "remote_event_value",
            "mean_useful_age",
        },
        path,
    )
    if error:
        return error
    if not _nullable_integer(value["goal_arrival_slot"], 0, HORIZON - 1):
        return f"{path}.goal_arrival_slot: expected null or integer in [0,99]"
    if not _integer(value["collision_slots"]) or value["collision_slots"] < 0:
        return f"{path}.collision_slots: expected nonnegative integer"
    if not _integer(value["sent_bits"]) or not 0 <= value["sent_bits"] <= SCENE_BIT_BUDGET:
        return f"{path}.sent_bits: expected integer in [0,30720]"
    for key, high in (("energy_used", 200.0), ("remote_event_value", math.inf)):
        if not _finite_number(value[key]) or not 0.0 <= float(value[key]) <= high + EPS:
            return f"{path}.{key}: expected finite nonnegative value; actual={value[key]!r}"
    age = value["mean_useful_age"]
    if age is not None and (not _finite_number(age) or float(age) < 0.0):
        return f"{path}.mean_useful_age: expected null or finite nonnegative value"
    return None


def _ids_unique(rows: Sequence[Any], key: str) -> bool:
    values = [row.get(key) for row in rows if isinstance(row, dict)]
    return (
        len(values) == len(rows)
        and all(isinstance(value, str) for value in values)
        and len(values) == len(set(values))
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

        state_rows, states = rows_and_index("states", "slot")
        slot_rows, slots = rows_and_index("slots", "slot")
        packet_rows, packets = rows_and_index("packets", "packet_id")
        map_rows, maps = rows_and_index("map_estimates", "slot")
        indexed[name] = {
            "record": record,
            "root_index": root_index,
            "state_rows": state_rows,
            "states": states,
            "slot_rows": slot_rows,
            "slots": slots,
            "packet_rows": packet_rows,
            "packets": packets,
            "map_rows": map_rows,
            "maps": maps,
        }
    return indexed


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
    return 1.0 if math.isclose(score, 1.0, abs_tol=1e-12) else max(0.0, min(1.0, score))


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
    scenario_ok = True
    scenario_detail = "known scenario records have the exact declared key set"
    checks: dict[str, tuple[bool, str]] = {
        "state_types": (True, "state records satisfy declared types"),
        "slot_types": (True, "slot records satisfy declared types"),
        "packet_types": (True, "packet records satisfy declared types and local relations"),
        "map_types": (True, "map records satisfy declared types"),
        "mission_types": (True, "mission records satisfy declared types"),
    }

    def fail(rule: str, detail: str) -> None:
        if checks[rule][0]:
            checks[rule] = (False, detail)

    for name in _known_scenarios(scenarios):
        record = scenarios[name]["record"]
        error = _shape_error(
            record,
            {"scenario", "states", "slots", "packets", "map_estimates", "mission"},
            name,
        )
        if error and scenario_ok:
            scenario_ok = False
            scenario_detail = error

        state_rows = record.get("states")
        if not isinstance(state_rows, list):
            fail("state_types", f"{name}.states: expected array")
        else:
            for index, row in enumerate(state_rows):
                error = _state_error(row, f"{name}.states[{index}]")
                if error:
                    fail("state_types", error)
                    break

        slot_rows = record.get("slots")
        if not isinstance(slot_rows, list):
            fail("slot_types", f"{name}.slots: expected array")
        else:
            for index, row in enumerate(slot_rows):
                error = _slot_error(row, f"{name}.slots[{index}]")
                if error:
                    fail("slot_types", error)
                    break

        packet_rows = record.get("packets")
        if not isinstance(packet_rows, list):
            fail("packet_types", f"{name}.packets: expected array")
        else:
            for index, row in enumerate(packet_rows):
                error = _packet_error(row, f"{name}.packets[{index}]")
                if error:
                    fail("packet_types", error)
                    break
            if checks["packet_types"][0] and not _ids_unique(packet_rows, "packet_id"):
                fail("packet_types", f"{name}.packets.packet_id: expected unique IDs")
            if checks["packet_types"][0]:
                created = [row["created_slot"] for row in packet_rows]
                if created != sorted(created) or len(created) != len(set(created)):
                    fail(
                        "packet_types",
                        f"{name}.packets: expected strictly increasing created_slot order",
                    )

        map_rows = record.get("map_estimates")
        if not isinstance(map_rows, list):
            fail("map_types", f"{name}.map_estimates: expected array")
        else:
            for index, row in enumerate(map_rows):
                error = _map_error(row, f"{name}.map_estimates[{index}]")
                if error:
                    fail("map_types", error)
                    break

        error = _mission_error(record.get("mission"), f"{name}.mission")
        if error:
            fail("mission_types", error)

    return [
        _atom("result_evidence", "scenario_shape", scenario_ok, scenario_detail),
        *[
            _atom("result_evidence", rule, checks[rule][0], checks[rule][1])
            for rule in (
                "state_types",
                "slot_types",
                "packet_types",
                "map_types",
                "mission_types",
            )
        ],
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not a list"] * 8
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    supplied_names: list[Any] = []
    for index, row in enumerate(data["scenarios"]):
        name = row.get("scenario") if isinstance(row, dict) else None
        supplied_names.append(name)
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key={name!r}")
        else:
            by_name[name].append(row)
    if (
        len(supplied_names) == len(SCENARIO_NAMES)
        and all(isinstance(name, str) for name in supplied_names)
        and set(supplied_names) == set(SCENARIO_NAMES)
        and supplied_names != list(SCENARIO_NAMES)
    ):
        defects.append("results.scenarios: canonical order mismatch")

    def count_rows(
        record: Mapping[str, Any], field: str, expected: Sequence[Any], label: str
    ) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {value!r}" for value in expected)
            return
        counts = {value: 0 for value in expected}
        supplied: list[Any] = []
        for index, row in enumerate(rows):
            value = row.get("slot") if isinstance(row, Mapping) else None
            supplied.append(value)
            known = False
            try:
                known = value in counts
            except TypeError:
                pass
            if known and type(value) is type(expected[0]):
                counts[value] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key={value!r}")
        for value, count in counts.items():
            if count == 0:
                defects.append(f"{label}: missing key {value!r}")
            defects.extend(f"{label}: duplicate key {value!r}" for _ in range(max(0, count - 1)))
        if len(supplied) == len(expected) and all(count == 1 for count in counts.values()):
            if supplied != list(expected):
                defects.append(f"{label}: canonical order mismatch")

    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_rows(record, "states", list(range(HORIZON + 1)), f"{name}.states")
        count_rows(record, "slots", list(range(HORIZON)), f"{name}.slots")
        count_rows(record, "map_estimates", list(range(HORIZON)), f"{name}.map_estimates")
        map_rows = record.get("map_estimates")
        if not isinstance(map_rows, list):
            continue
        expected_cells = _safe_cells(name)
        for map_index, map_row in enumerate(map_rows):
            if not isinstance(map_row, Mapping) or not _integer(map_row.get("slot")):
                continue
            slot = map_row["slot"]
            if not 0 <= slot < HORIZON:
                continue
            cells = map_row.get("cells")
            label = f"{name}.map_estimates[{slot}].cells"
            if not isinstance(cells, list):
                defects.extend(f"{label}: missing key {cell!r}" for cell in expected_cells)
                continue
            counts = {cell: 0 for cell in expected_cells}
            supplied: list[Any] = []
            for cell_index, row in enumerate(cells):
                key = (row.get("x"), row.get("y")) if isinstance(row, Mapping) else None
                supplied.append(key)
                key_known = False
                try:
                    key_known = key in counts
                except TypeError:
                    key_known = False
                if key_known and all(_integer(value) for value in key):
                    counts[key] += 1
                else:
                    defects.append(f"{label}[{cell_index}]: unknown key={key!r}")
            for cell, count in counts.items():
                if count == 0:
                    defects.append(f"{label}: missing key {cell!r}")
                defects.extend(f"{label}: duplicate key {cell!r}" for _ in range(max(0, count - 1)))
            if (
                len(supplied) == len(expected_cells)
                and all(count == 1 for count in counts.values())
                and supplied != list(expected_cells)
            ):
                defects.append(f"{label}: canonical order mismatch")
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


def _decision_map(item: Mapping[str, Any]) -> dict[int, dict[str, Any]] | None:
    slots = item.get("slots")
    if not isinstance(slots, Mapping) or any(slot not in slots for slot in range(HORIZON)):
        return None
    record = item.get("record")
    if not isinstance(record, Mapping):
        return None
    packet_rows = record.get("packets")
    if not isinstance(packet_rows, list) or any(
        _packet_error(row, "packet") is not None for row in packet_rows
    ):
        return None
    if not _ids_unique(packet_rows, "packet_id"):
        return None
    created = [row["created_slot"] for row in packet_rows]
    if created != sorted(created) or len(created) != len(set(created)):
        return None
    packets_by_slot = {row["created_slot"]: row for row in packet_rows}
    decisions: dict[int, dict[str, Any]] = {}
    for slot in range(HORIZON):
        row = slots[slot]
        if _slot_error(row, f"slot[{slot}]") is not None:
            return None
        decisions[slot] = {
            "move": row["move"],
            "packet_id": row["packet_id"],
            "packet": packets_by_slot.get(slot),
        }
    return decisions


DecisionProvider = Callable[[int, Mapping[str, Any]], Mapping[str, Any]]


def _initial_receiver_map(name: str) -> dict[tuple[int, int], dict[str, Any]]:
    static = SCENARIOS[name]["obstacles"]
    return {
        (x, y): {
            "code": _probability_code((x, y) in static),
            "source_slot": -1,
            "source_kind": "initial",
            "packet_id": None,
        }
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
    }


def _packet_from_nodes(name: str, slot: int, nodes: Sequence[str]) -> dict[str, Any]:
    parsed = [(node, _parse_node(node)) for node in nodes]
    if any(item is None for _, item in parsed):
        raise ValueError(f"provider returned an invalid quadtree node at slot {slot}")
    ordered = sorted(parsed, key=lambda item: item[1][2])  # type: ignore[index]
    entries = [
        {
            "node": node,
            "node_id": item[2],
            "code": _node_code(name, node, slot),
        }
        for node, item in ordered
        if item is not None
    ]
    return {
        "packet_id": f"pkt-{slot:03d}",
        "created_slot": slot,
        "entries": entries,
        "bits": 16 + 21 * len(entries),
        "queue_enter_slot": slot,
        "service_start_slot": None,
        "service_complete_slot": None,
        "arrival_slot": None,
        "outcome": "unfinished",
    }


def _packet_record(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "created_slot": packet["created_slot"],
        "entries": [dict(entry) for entry in packet["entries"]],
        "bits": packet["bits"],
        "queue_enter_slot": packet["queue_enter_slot"],
        "service_start_slot": packet["service_start_slot"],
        "service_complete_slot": packet["service_complete_slot"],
        "arrival_slot": packet["arrival_slot"],
        "outcome": packet["outcome"],
    }


def _map_cells(
    receiver_map: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "x": cell[0],
            "y": cell[1],
            "code": receiver_map[cell]["code"],
            "source_slot": receiver_map[cell]["source_slot"],
            "source_kind": receiver_map[cell]["source_kind"],
        }
        for cell in cells
    ]


def _provider_views(
    name: str,
    slot: int,
    position: tuple[int, int],
    sender_energy: float,
    navigator_energy: float,
    receiver_map: Mapping[tuple[int, int], Mapping[str, Any]],
    local_cells: Sequence[tuple[int, int]],
    queue: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    arrived: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenario = SCENARIOS[name]
    static = {
        "grid_size": GRID_SIZE,
        "start": list(scenario["start"]),
        "goal": list(scenario["goal"]),
        "static_obstacles": [list(cell) for cell in sorted(scenario["obstacles"])],
        "deadline": scenario["deadline"],
        "horizon": HORIZON,
    }
    sender_map = [
        {"x": x, "y": y, "code": _probability_code(_true_occupied(name, (x, y), slot))}
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
    ]
    queue_view = [
        {
            "packet_id": packet["packet_id"],
            "created_slot": packet["created_slot"],
            "remaining_bits": packet["remaining_bits"],
            "bits": packet["bits"],
            "entries": [dict(entry) for entry in packet["entries"]],
        }
        for packet in queue
    ]
    receiver_cells = tuple((x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE))
    return {
        "sender": {
            "role": "sender",
            "slot": slot,
            "static": static,
            "energy": sender_energy,
            "capacity": _capacity(name, slot),
            "queue": queue_view,
            "feedback": [dict(row) for row in feedback],
            "true_map": sender_map,
        },
        "navigator": {
            "role": "navigator",
            "slot": slot,
            "static": static,
            "position": list(position),
            "energy": navigator_energy,
            "receiver_map": _map_cells(receiver_map, receiver_cells),
            "local_observation": _map_cells(receiver_map, local_cells),
            "arrivals": [dict(row) for row in arrived],
        },
    }


def _event_analysis(
    name: str,
    remote_writes: Sequence[Mapping[str, Any]],
    packets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    deadline = int(SCENARIOS[name]["deadline"])
    hit_value = 0.0
    total_value = 0.0
    assigned_packets: set[str] = set()
    for cell in _safe_cells(name):
        changes = [
            slot
            for slot in range(1, deadline + 1)
            if _true_occupied(name, cell, slot) != _true_occupied(name, cell, slot - 1)
        ]
        for index, event_slot in enumerate(changes):
            next_change = changes[index + 1] if index + 1 < len(changes) else deadline + 1
            value = 0.8 * _safe_weight(name, cell)
            total_value += value
            last_slot = min(deadline, event_slot + 6, next_change - 1)
            candidates = [
                write
                for write in remote_writes
                if write["cell"] == cell
                and event_slot <= write["slot"] <= last_slot
                and write["source_slot"] >= event_slot
                and abs(
                    int(write["code"])
                    - _probability_code(_true_occupied(name, cell, int(write["slot"])))
                )
                <= 1
            ]
            if not candidates:
                continue
            selected = min(candidates, key=lambda row: (row["slot"], row["packet_id"]))
            hit_value += value
            assigned_packets.add(str(selected["packet_id"]))
    timely = hit_value / total_value if total_value > 0.0 else 0.0
    useful = [packets[packet_id] for packet_id in sorted(assigned_packets)]
    ages = [int(packet["arrival_slot"]) - int(packet["created_slot"]) for packet in useful]
    waits = [
        int(packet["service_start_slot"]) - int(packet["queue_enter_slot"]) for packet in useful
    ]
    return {
        "timely": timely,
        "hit_value": hit_value,
        "assigned_packet_ids": tuple(sorted(assigned_packets)),
        "mean_age": sum(ages) / len(ages) if ages else None,
        "mean_wait": sum(waits) / len(waits) if waits else None,
    }


def _scenario_metrics(
    name: str,
    states: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    map_estimates: Sequence[Mapping[str, Any]],
    sent_bits: int,
    collision_slots: int,
    goal_arrival_slot: int | None,
    event_analysis: Mapping[str, Any],
) -> dict[str, float]:
    scenario = SCENARIOS[name]
    deadline = int(scenario["deadline"])
    goal = tuple(scenario["goal"])
    start = tuple(scenario["start"])
    initial_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if goal_arrival_slot is not None and goal_arrival_slot <= deadline:
        progress = 1.0
    else:
        deadline_position = (states[deadline + 1]["x"], states[deadline + 1]["y"])
        remaining = abs(deadline_position[0] - goal[0]) + abs(deadline_position[1] - goal[1])
        progress = max(0.0, min(1.0, 1.0 - remaining / initial_distance))
    safe = max(0.0, min(1.0, 1.0 - collision_slots / HORIZON))

    fidelity_sum = 0.0
    weight_sum = sum(_safe_weight(name, cell) for cell in _safe_cells(name))
    for slot in range(deadline + 1):
        cells = {(row["x"], row["y"]): row for row in map_estimates[slot]["cells"]}
        for cell in _safe_cells(name):
            expected = 0.9 if _true_occupied(name, cell, slot) else 0.1
            estimate = int(cells[cell]["code"]) / 4095.0
            fidelity_sum += _safe_weight(name, cell) * (1.0 - abs(estimate - expected))
    fidelity = fidelity_sum / ((deadline + 1) * weight_sum)
    timely = float(event_analysis["timely"])

    moves = sum(row["move"] != "stay" for row in slots)
    route_eff = max(
        0.0,
        min(
            1.0,
            1.0 - max(0, moves - initial_distance) / max(1, HORIZON - initial_distance),
        ),
    )
    energy = SENDER_ENERGY_PER_BIT * sent_bits + moves
    energy_eff = max(0.0, min(1.0, 1.0 - energy / 200.0))
    load_eff = timely * (0.5 + 0.5 * max(0.0, min(1.0, 1.0 - sent_bits / 30720.0)))
    mean_age = event_analysis["mean_age"]
    mean_wait = event_analysis["mean_wait"]
    if mean_age is None or mean_wait is None:
        freshness_eff = 0.0
    else:
        freshness_eff = timely * (
            0.5 * max(0.0, min(1.0, 1.0 - float(mean_age) / 6.0))
            + 0.5 * max(0.0, min(1.0, 1.0 - float(mean_wait) / HORIZON))
        )
    return {
        "navigation_safety": 0.75 * progress + 0.25 * safe,
        "map_fidelity": 0.65 * fidelity + 0.35 * timely,
        "route_energy_efficiency": progress * (0.6 * route_eff + 0.4 * energy_eff),
        "compression_load_efficiency": load_eff,
        "update_freshness_efficiency": freshness_eff,
        "progress": progress,
        "fidelity": fidelity,
        "timely": timely,
    }


def _simulate_scenario(
    name: str,
    decisions: Mapping[int, Mapping[str, Any]] | None = None,
    decision_provider: DecisionProvider | None = None,
) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    if (decisions is None) == (decision_provider is None):
        raise ValueError("provide exactly one of decisions or decision_provider")

    rules = [item.removeprefix("decision_legality.") for item in ATOM_SHARES["decision_legality"]]
    legality: dict[str, list[Any]] = {rule: [True, ""] for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if legality[rule][0]:
            legality[rule] = [False, detail]

    scenario = SCENARIOS[name]
    position = tuple(scenario["start"])
    goal = tuple(scenario["goal"])
    sender_energy = SENDER_INITIAL_ENERGY
    navigator_energy = NAVIGATOR_INITIAL_ENERGY
    receiver_map = _initial_receiver_map(name)
    queue: list[dict[str, Any]] = []
    packets: dict[str, dict[str, Any]] = {}
    packet_order: list[str] = []
    arrivals: dict[int, list[str]] = defaultdict(list)
    feedback: list[dict[str, Any]] = []
    remote_writes: list[dict[str, Any]] = []
    sent_bits = 0
    collision_slots = 0
    goal_arrival_slot: int | None = None
    goal_reached = False

    states_out = [
        {
            "slot": 0,
            "x": position[0],
            "y": position[1],
            "sender_energy": sender_energy,
            "navigator_energy": navigator_energy,
            "goal_reached": False,
            "collision": False,
        }
    ]
    slots_out: list[dict[str, Any]] = []
    maps_out: list[dict[str, Any]] = []

    for slot in range(HORIZON):
        arrived_records: list[dict[str, Any]] = []
        for packet_id in sorted(arrivals.pop(slot, [])):
            packet = packets[packet_id]
            arrived_records.append(_packet_record(packet))
            for entry in packet["entries"]:
                for cell in _node_region(str(entry["node"])):
                    saved = receiver_map[cell]
                    newer = packet["created_slot"] > saved["source_slot"]
                    tied_remote = (
                        packet["created_slot"] == saved["source_slot"]
                        and saved["source_kind"] == "remote"
                        and packet_id > str(saved["packet_id"])
                    )
                    if not (newer or tied_remote):
                        continue
                    receiver_map[cell] = {
                        "code": entry["code"],
                        "source_slot": packet["created_slot"],
                        "source_kind": "remote",
                        "packet_id": packet_id,
                    }
                    remote_writes.append(
                        {
                            "slot": slot,
                            "cell": cell,
                            "source_slot": packet["created_slot"],
                            "packet_id": packet_id,
                            "code": entry["code"],
                        }
                    )

        local_cells = tuple(
            (x, y)
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
            if abs(x - position[0]) + abs(y - position[1]) <= 1
        )
        for cell in local_cells:
            receiver_map[cell] = {
                "code": _probability_code(_true_occupied(name, cell, slot)),
                "source_slot": slot,
                "source_kind": "local",
                "packet_id": None,
            }
        maps_out.append(
            {
                "slot": slot,
                "cells": _map_cells(receiver_map, _safe_cells(name)),
            }
        )

        provider_packet = False
        if decision_provider is not None:
            action = decision_provider(
                slot,
                _provider_views(
                    name,
                    slot,
                    position,
                    sender_energy,
                    navigator_energy,
                    receiver_map,
                    local_cells,
                    queue,
                    feedback,
                    arrived_records,
                ),
            )
            if not isinstance(action, Mapping):
                raise ValueError(f"provider slot {slot}: expected mapping")
            move = action.get("move")
            raw_nodes = action.get("nodes")
            if raw_nodes is None:
                packet = None
            elif isinstance(raw_nodes, list) and all(isinstance(node, str) for node in raw_nodes):
                packet = _packet_from_nodes(name, slot, raw_nodes) if raw_nodes else None
                provider_packet = packet is not None
            else:
                raise ValueError(f"provider slot {slot}: nodes must be a string array or null")
            slot_packet_id = packet["packet_id"] if packet is not None else None
        else:
            decision = decisions[slot]  # type: ignore[index]
            move = decision.get("move")
            packet = decision.get("packet")
            slot_packet_id = decision.get("packet_id")

        if move not in MOVES:
            fail("move_domain", f"{name}.slots[{slot}].move: unknown move={move!r}")
            replay_move = "stay"
        else:
            replay_move = str(move)

        expected_reference = packet["packet_id"] if isinstance(packet, Mapping) else None
        reference_ok = slot_packet_id == expected_reference
        if not reference_ok:
            fail(
                "packet_reference",
                f"{name}.slots[{slot}].packet_id: expected={expected_reference!r}; "
                f"actual={slot_packet_id!r}",
            )

        admitted_packet: dict[str, Any] | None = None
        if isinstance(packet, Mapping):
            entries = packet["entries"]
            parsed = [_parse_node(entry["node"]) for entry in entries]
            node_ok = (
                all(item is not None for item in parsed)
                and [entry["node_id"] for entry in entries]
                == [item[2] for item in parsed if item is not None]
                and [entry["node_id"] for entry in entries]
                == sorted(entry["node_id"] for entry in entries)
                and len({entry["node_id"] for entry in entries}) == len(entries)
                and _prefix_free([entry["node"] for entry in entries])
            )
            if not node_ok:
                fail(
                    "node_selection",
                    f"{name}.packets[{packet['packet_id']}].entries: invalid node bijection, "
                    "order, uniqueness, or prefix relation",
                )
            correct_codes = all(
                parsed[index] is None or entry["code"] == _node_code(name, entry["node"], slot)
                for index, entry in enumerate(entries)
            )
            expected_bits = 16 + 21 * len(entries)
            encoding_ok = correct_codes and packet["bits"] == expected_bits
            if not encoding_ok:
                fail(
                    "packet_encoding",
                    f"{name}.packets[{packet['packet_id']}]: expected current-map codes and "
                    f"bits={expected_bits}; actual_bits={packet['bits']!r}",
                )
            queued_bits = sum(int(item["remaining_bits"]) for item in queue)
            queue_ok = queued_bits + expected_bits <= QUEUE_CAPACITY
            budget_ok = sent_bits + expected_bits <= SCENE_BIT_BUDGET
            if not queue_ok or not budget_ok:
                fail(
                    "queue_budget",
                    f"{name}.slot[{slot}]: queued={queued_bits}, packet={expected_bits}, "
                    f"sent={sent_bits}, queue_limit={QUEUE_CAPACITY}, bit_limit={SCENE_BIT_BUDGET}",
                )
            sender_ok = sender_energy + EPS >= SENDER_ENERGY_PER_BIT * expected_bits
            if not sender_ok:
                fail(
                    "energy",
                    f"{name}.slot[{slot}].sender_energy: available={sender_energy}, "
                    f"required={SENDER_ENERGY_PER_BIT * expected_bits}",
                )
            admission_ok = (
                reference_ok and node_ok and encoding_ok and queue_ok and budget_ok and sender_ok
            )
            if admission_ok:
                admitted_packet = {
                    **_packet_record(packet),
                    "bits": expected_bits,
                    "remaining_bits": expected_bits,
                    "service_start_slot": None,
                    "service_complete_slot": None,
                    "arrival_slot": None,
                    "outcome": "unfinished",
                }
                packet_id = str(admitted_packet["packet_id"])
                packets[packet_id] = admitted_packet
                packet_order.append(packet_id)
                queue.append(admitted_packet)
                sent_bits += expected_bits
                sender_energy -= SENDER_ENERGY_PER_BIT * expected_bits
            elif provider_packet:
                failed = [rule for rule, value in legality.items() if not value[0]]
                raise ValueError(f"provider proposed an illegal packet at slot {slot}: {failed}")

        slots_out.append(
            {
                "slot": slot,
                "move": move if move in MOVES else str(move),
                "packet_id": (
                    admitted_packet["packet_id"]
                    if decision_provider is not None and admitted_packet is not None
                    else (None if decision_provider is not None else slot_packet_id)
                ),
            }
        )

        if queue:
            head = queue[0]
            served = min(_capacity(name, slot), int(head["remaining_bits"]))
            if served > 0:
                if head["service_start_slot"] is None:
                    head["service_start_slot"] = slot
                head["remaining_bits"] -= served
                if head["remaining_bits"] == 0:
                    head["service_complete_slot"] = slot
                    if _lost(name, slot):
                        head["outcome"] = "lost"
                    else:
                        head["outcome"] = "delivered"
                        head["arrival_slot"] = slot + 1
                        arrivals[slot + 1].append(str(head["packet_id"]))
                    feedback.append(
                        {
                            "packet_id": head["packet_id"],
                            "complete_slot": slot,
                            "outcome": head["outcome"],
                        }
                    )
                    queue.pop(0)

        collision = False
        if replay_move != "stay":
            if navigator_energy + EPS < 1.0:
                fail(
                    "energy",
                    f"{name}.slots[{slot}].move: non-stay requested with "
                    f"navigator_energy={navigator_energy}",
                )
                replay_move = "stay"
            else:
                navigator_energy -= 1.0
        dx, dy = MOVES[replay_move]
        target = (position[0] + dx, position[1] + dy)
        current_blocked = (
            not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE)
            or target in scenario["obstacles"]
            or target in _dynamic_positions(name, slot)
        )
        if current_blocked:
            collision = True
        else:
            position = target
        if position in _dynamic_positions(name, slot + 1):
            collision = True
        if collision:
            collision_slots += 1
        if not goal_reached and position == goal:
            goal_reached = True
            goal_arrival_slot = slot
        states_out.append(
            {
                "slot": slot + 1,
                "x": position[0],
                "y": position[1],
                "sender_energy": sender_energy,
                "navigator_energy": navigator_energy,
                "goal_reached": goal_reached,
                "collision": collision,
            }
        )

    packet_records = [_packet_record(packets[packet_id]) for packet_id in packet_order]
    packet_index = {packet["packet_id"]: packet for packet in packet_records}
    event_analysis = _event_analysis(name, remote_writes, packet_index)
    moves = sum(row["move"] != "stay" for row in slots_out)
    mission = {
        "goal_arrival_slot": goal_arrival_slot,
        "collision_slots": collision_slots,
        "sent_bits": sent_bits,
        "energy_used": SENDER_ENERGY_PER_BIT * sent_bits + moves,
        "remote_event_value": event_analysis["hit_value"],
        "mean_useful_age": event_analysis["mean_age"],
    }
    record = {
        "scenario": name,
        "states": states_out,
        "slots": slots_out,
        "packets": packet_records,
        "map_estimates": maps_out,
        "mission": mission,
    }
    return {
        "record": record,
        "legality": legality,
        "remote_writes": remote_writes,
        "event_analysis": event_analysis,
        "metrics": _scenario_metrics(
            name,
            states_out,
            slots_out,
            maps_out,
            sent_bits,
            collision_slots,
            goal_arrival_slot,
            event_analysis,
        ),
    }


def _simulate_policy(name: str, provider: DecisionProvider) -> dict[str, Any]:
    return _simulate_scenario(name, decision_provider=provider)


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules = [key.removeprefix("decision_legality.") for key in ATOM_SHARES["decision_legality"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(scenarios):
        item = scenarios[name]
        decisions = _decision_map(item)
        if decisions is None:
            continue
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception as exc:  # malformed records are a failed legality trace
            for rule in rules:
                fail(rule, f"{name}: legality replay raised {type(exc).__name__}: {exc}")
            continue
        for rule in rules:
            legal, detail = replay["legality"][rule]
            if not legal:
                fail(rule, f"{name}: {detail}")
    return [
        _atom(
            "decision_legality",
            rule,
            passed[rule],
            details[rule] or "all canonical scenario decisions satisfy this constraint",
        )
        for rule in rules
    ]


def _replay_atoms(
    scenarios: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
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
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception as exc:
            for rule in rules:
                fail(rule, f"{name}: replay raised {type(exc).__name__}: {exc}")
            continue
        replayed[name] = replay
        expected = replay["record"]
        actual = item["record"]
        legal = {rule: bool(value[0]) for rule, value in replay["legality"].items()}

        def dependencies_pass(*dependencies: str) -> bool:
            return all(legal.get(dependency, False) for dependency in dependencies)

        actual_states = actual.get("states")
        expected_states = expected.get("states")
        states_replayable = (
            isinstance(actual_states, list)
            and len(actual_states) == HORIZON + 1
            and all(
                _state_error(row, f"{name}.states[{index}]") is None
                for index, row in enumerate(actual_states)
            )
            and [row["slot"] for row in actual_states] == list(range(HORIZON + 1))
        )
        if (
            dependencies_pass("move_domain", "energy")
            and states_replayable
            and isinstance(expected_states, list)
        ):
            motion_keys = ("slot", "x", "y", "navigator_energy")
            actual_motion = [{key: row.get(key) for key in motion_keys} for row in actual_states]
            expected_motion = [
                {key: row.get(key) for key in motion_keys} for row in expected_states
            ]
            ok, detail = _compare(actual_motion, expected_motion, f"{name}.states.motion")
            if not ok:
                fail("navigator_state", detail)
            collision_keys = ("goal_reached", "collision")
            actual_collision = [
                {key: row.get(key) for key in collision_keys} for row in actual_states
            ]
            expected_collision = [
                {key: row.get(key) for key in collision_keys} for row in expected_states
            ]
            ok, detail = _compare(
                actual_collision, expected_collision, f"{name}.states.collision_goal"
            )
            if not ok:
                fail("collision_goal", detail)
        packet_dependencies = (
            "packet_reference",
            "node_selection",
            "packet_encoding",
            "queue_budget",
            "energy",
        )
        if dependencies_pass(*packet_dependencies):
            actual_packets = actual.get("packets")
            expected_packets = expected.get("packets")
            if isinstance(actual_packets, list) and isinstance(expected_packets, list):
                service_keys = (
                    "packet_id",
                    "service_start_slot",
                    "service_complete_slot",
                    "arrival_slot",
                    "outcome",
                )
                ok, detail = _compare(
                    [{key: row.get(key) for key in service_keys} for row in actual_packets],
                    [{key: row.get(key) for key in service_keys} for row in expected_packets],
                    f"{name}.packets.service",
                )
                if not ok:
                    fail("sender_packet_service", detail)
            else:
                fail("sender_packet_service", f"{name}.packets: expected replayable arrays")
            actual_maps = actual.get("map_estimates")
            maps_replayable = (
                isinstance(actual_maps, list)
                and len(actual_maps) == HORIZON
                and all(
                    _map_error(row, f"{name}.map_estimates[{index}]") is None
                    for index, row in enumerate(actual_maps)
                )
                and [row["slot"] for row in actual_maps] == list(range(HORIZON))
                and all(
                    [(cell["x"], cell["y"]) for cell in row["cells"]] == list(_safe_cells(name))
                    for row in actual_maps
                )
            )
            if dependencies_pass("move_domain", "energy") and maps_replayable:
                ok, detail = _compare(
                    actual_maps,
                    expected.get("map_estimates"),
                    f"{name}.map_estimates",
                )
                if not ok:
                    fail("receiver_map", detail)

        actual_mission = actual.get("mission")
        expected_mission = expected.get("mission")
        mission_replayable = _mission_error(actual_mission, f"{name}.mission") is None
        if (
            dependencies_pass("move_domain", *packet_dependencies)
            and mission_replayable
            and isinstance(actual_mission, Mapping)
            and isinstance(expected_mission, Mapping)
        ):
            resource_keys = ("sent_bits", "energy_used")
            ok, detail = _compare(
                {key: actual_mission.get(key) for key in resource_keys},
                {key: expected_mission.get(key) for key in resource_keys},
                f"{name}.mission.resources",
            )
            if not ok:
                fail("mission_resources", detail)
            if states_replayable and isinstance(expected_states, list):
                actual_sender_energy = [row.get("sender_energy") for row in actual_states]
                expected_sender_energy = [row.get("sender_energy") for row in expected_states]
                ok, detail = _compare(
                    actual_sender_energy,
                    expected_sender_energy,
                    f"{name}.states.sender_energy",
                )
                if not ok:
                    fail("mission_resources", detail)
        if (
            dependencies_pass("move_domain", "energy")
            and mission_replayable
            and isinstance(actual_mission, Mapping)
            and isinstance(expected_mission, Mapping)
        ):
            collision_keys = ("goal_arrival_slot", "collision_slots")
            ok, detail = _compare(
                {key: actual_mission.get(key) for key in collision_keys},
                {key: expected_mission.get(key) for key in collision_keys},
                f"{name}.mission.collision_goal",
            )
            if not ok:
                fail("collision_goal", detail)

        if dependencies_pass(*packet_dependencies):
            expected_events = replay["event_analysis"]
            if mission_replayable and isinstance(actual_mission, Mapping):
                event_checks = (
                    ("remote_event_value", expected_events["hit_value"]),
                    ("mean_useful_age", expected_events["mean_age"]),
                )
                for key, value in event_checks:
                    ok, detail = _compare(actual_mission.get(key), value, f"{name}.mission.{key}")
                    if not ok:
                        fail("event_summaries", detail)

    atoms = [
        _atom(
            "replay_validity",
            rule,
            passed[rule],
            details[rule] or "fixed-model state and event replay matched the submitted evidence",
        )
        for rule in rules
    ]
    return atoms, replayed


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
    if isinstance(expected, (str, type(None))):
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


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all canonical scenario replays are required for metrics")
    names = (
        "navigation_safety",
        "map_fidelity",
        "route_energy_efficiency",
        "compression_load_efficiency",
        "update_freshness_efficiency",
    )
    metrics = {
        metric: sum(float(replayed[name]["metrics"][metric]) for name in SCENARIO_NAMES)
        / len(SCENARIO_NAMES)
        for metric in names
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
    raw_lines = text.splitlines()
    lines = [""] * len(raw_lines)
    in_fence = False
    fence_marker = ""
    in_comment = False
    for index, raw_line in enumerate(raw_lines):
        if in_fence:
            stripped = raw_line.lstrip()
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        remainder = raw_line
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
        lines[index] = kept
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match or match.group(2).strip() not in required:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            next_heading = re.match(r"^(#{1,6})\s+.+", lines[cursor])
            if next_heading and len(next_heading.group(1)) <= level:
                break
            body.append(lines[cursor])
            cursor += 1
        found.setdefault(title, "\n".join(body).strip())
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
    detail = f"missing_sections={missing}; normalized={score:.9g}"
    return score, detail


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
            state.official_message = f"could not remove previous results.json: {exc}"
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
            state.timed_out = "timeout" in str(exc).lower()
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
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok
                else (
                    f"official run failed: exit={state.exit_code}, "
                    f"elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, "
                    f"new_regular_results={state.result_file_is_new_regular}"
                )
            )
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
    except Exception as exc:  # noqa: BLE001
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
        _CACHE[key] = state
    return state


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
        text = ws.read_text("report.md")
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"
    return _report_score(text)


async def _metric_check(ws: Workspace, metric: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{metric}: validity gate not satisfied; metric not computed"
    value = _gated_metric(state.metrics, metric)
    return value, f"{metric}: independently replayed value={value:.9g}"


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


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "navigation_safety")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "map_fidelity")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "route_energy_efficiency")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "compression_load_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "update_freshness_efficiency")
