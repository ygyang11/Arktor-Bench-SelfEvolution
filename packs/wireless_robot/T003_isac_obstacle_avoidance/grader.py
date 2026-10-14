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
HORIZON = 24
AGE_LIMIT = 3
MAP_VALID_AGE = 2
SIGNAL_BUDGET = 18
BURST_WINDOW = 4
BURST_LIMIT = 3
VELOCITY_STEP_LIMIT = 0.8
SAFETY_RADIUS = 1.0
COLLISION_RADIUS = 0.75
WAYPOINT_TOL = 1e-6
EPS = 1e-6
DEADLINES = (0, 6, 12, 18, 24)
WAYPOINT_KEYS = frozenset(zip(range(1, 5), DEADLINES[1:]))
SCENARIO_NAMES = (
    "crossing_corridor",
    "north_gate",
    "switchback",
    "diagonal_drift",
    "return_arc",
)
WAYPOINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "crossing_corridor": ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (8.0, 4.0), (8.0, 8.0)),
    "north_gate": ((0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 8.0), (8.0, 8.0)),
    "switchback": ((0.0, 0.0), (3.0, 1.0), (1.0, 5.0), (6.0, 6.0), (8.0, 8.0)),
    "diagonal_drift": ((0.0, 0.0), (2.0, 3.0), (5.0, 1.0), (6.0, 6.0), (8.0, 8.0)),
    "return_arc": ((0.0, 0.0), (4.0, 2.0), (2.0, 6.0), (6.0, 7.0), (8.0, 8.0)),
}
OBSTACLES: dict[str, tuple[tuple[tuple[float, float], tuple[float, float]], ...]] = {
    "crossing_corridor": (((2.0, -0.8), (0.0, 0.18)), ((4.4, 2.0), (-0.12, 0.0))),
    "north_gate": (((-0.9, 2.3), (0.18, 0.0)), ((3.2, 4.4), (0.0, 0.18))),
    "switchback": (((2.0, 2.0), (0.12, 0.12)), ((3.2, 5.2), (-0.20, 0.0))),
    "diagonal_drift": (((2.0, 1.8), (0.15, -0.05)), ((5.4, 3.1), (-0.10, 0.16))),
    "return_arc": (((3.5, 1.8), (-0.10, 0.14)), ((4.2, 6.0), (0.16, -0.08))),
}
SENSE_LOST: dict[str, frozenset[int]] = {
    "crossing_corridor": frozenset({5, 6, 7}),
    "north_gate": frozenset({2, 3, 4, 14}),
    "switchback": frozenset({8, 9, 10, 17}),
    "diagonal_drift": frozenset({0, 1, 12, 13, 20}),
    "return_arc": frozenset({4, 5, 6, 18, 19}),
}
COMMAND_LOST: dict[str, frozenset[int]] = {
    "crossing_corridor": frozenset({4, 5, 6, 15}),
    "north_gate": frozenset({0, 1, 10, 11, 12}),
    "switchback": frozenset({3, 4, 5, 16, 17}),
    "diagonal_drift": frozenset({7, 8, 9, 18}),
    "return_arc": frozenset({2, 3, 14, 15, 16}),
}


# Atom identities and relative shares are task-specific; every mapping sums to one.
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 1.0 / 3.0,
        "result_json.root": 1.0 / 3.0,
        "result_json.scenarios": 1.0 / 3.0,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 1.0 / 9.0,
        "result_evidence.decision_shape": 1.0 / 9.0,
        "result_evidence.decision_types": 1.0 / 9.0,
        "result_evidence.trace_shape": 1.0 / 9.0,
        "result_evidence.trace_values": 1.0 / 9.0,
        "result_evidence.trace_state": 1.0 / 9.0,
        "result_evidence.trace_flags": 1.0 / 9.0,
        "result_evidence.waypoint_shape": 1.0 / 9.0,
        "result_evidence.waypoint_values": 1.0 / 9.0,
    },
    "decision_legality": {
        "decision_legality.time_domain": 1.0 / 9.0,
        "decision_legality.transmission_flags": 1.0 / 9.0,
        "decision_legality.null_payload": 1.0 / 9.0,
        "decision_legality.velocity_pair": 1.0 / 9.0,
        "decision_legality.velocity_box": 1.0 / 9.0,
        "decision_legality.velocity_norm": 1.0 / 9.0,
        "decision_legality.signal_budget": 1.0 / 9.0,
        "decision_legality.burst_limit": 1.0 / 9.0,
        "decision_legality.velocity_continuity": 1.0 / 9.0,
    },
    "replay_validity": {
        "replay_validity.trace_position": 1.0 / 10.0,
        "replay_validity.command_sequence": 1.0 / 10.0,
        "replay_validity.command_age": 1.0 / 10.0,
        "replay_validity.map_sequence": 1.0 / 10.0,
        "replay_validity.map_age": 1.0 / 10.0,
        "replay_validity.safety_stop": 1.0 / 10.0,
        "replay_validity.executed_velocity": 1.0 / 10.0,
        "replay_validity.collision": 1.0 / 10.0,
        "replay_validity.waypoint_geometry": 1.0 / 10.0,
        "replay_validity.waypoint_success": 1.0 / 10.0,
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


def _link(kind: str, scenario: str, t: int) -> tuple[bool, int]:
    if kind == "sense":
        lost = SENSE_LOST[scenario]
        if scenario == "crossing_corridor":
            delay = 1 if t % 3 == 0 else 0
        elif scenario == "north_gate":
            delay = 2 if t in {1, 8, 15, 22} else 1
        elif scenario == "switchback":
            delay = 1 if t % 2 == 0 else 0
        elif scenario == "diagonal_drift":
            delay = 2 if t % 4 == 0 else 1
        else:
            delay = 1 if t % 2 == 1 else 0
    else:
        lost = COMMAND_LOST[scenario]
        if scenario == "crossing_corridor":
            delay = 1 if t in {3, 4, 11, 12, 19, 20} else 0
        elif scenario == "north_gate":
            delay = 2 if t % 2 == 0 else 1
        elif scenario == "switchback":
            delay = 1 if t % 3 == 0 else 0
        elif scenario == "diagonal_drift":
            delay = 2 if t in {6, 14, 22} else 0
        else:
            delay = 1 if t % 2 == 1 else 0
    return t not in lost, delay


def _obstacle_centres(scenario: str, t: int) -> tuple[tuple[float, float], ...]:
    return tuple(
        (initial[0] + t * velocity[0], initial[1] + t * velocity[1])
        for initial, velocity in OBSTACLES[scenario]
    )


def _reference(scenario: str, t: int) -> tuple[float, float]:
    if not 0 <= t <= HORIZON:
        raise ValueError(f"reference time out of range: {t}")
    for index in range(len(DEADLINES) - 1):
        if DEADLINES[index] <= t <= DEADLINES[index + 1]:
            fraction = (t - DEADLINES[index]) / 6.0
            left = WAYPOINTS[scenario][index]
            right = WAYPOINTS[scenario][index + 1]
            return (
                (1.0 - fraction) * left[0] + fraction * right[0],
                (1.0 - fraction) * left[1] + fraction * right[1],
            )
    raise ValueError(f"reference time out of range: {t}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite_number(x) for x in value)


def _close(a: Any, b: Any, tolerance: float = EPS) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tolerance


def _close_pair(a: Any, b: Sequence[float], tolerance: float = EPS) -> bool:
    return _pair(a) and len(b) == 2 and _close(a[0], b[0], tolerance) and _close(a[1], b[1], tolerance)


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
        raise ValueError("replay requires one valid decision record for every control slot")
    for t in range(HORIZON):
        if not _decision_valid(decisions[t]):
            raise ValueError(f"replay decision at t={t} is malformed or outside its declared domain")


def _replay_scenario(scenario: str, decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    _require_replay_decisions(decisions)
    pending_sense: dict[int, list[tuple[int, tuple[tuple[float, float], ...]]]] = defaultdict(list)
    pending_command: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    active_map: tuple[int, tuple[tuple[float, float], ...]] | None = None
    active_command: tuple[int, tuple[float, float]] | None = None
    position = [0.0, 0.0]
    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    trace: list[dict[str, Any]] = []

    for t in range(HORIZON):
        for generation, snapshot in sorted(pending_sense.pop(t, []), key=lambda item: item[0]):
            if active_map is None or generation > active_map[0]:
                active_map = (generation, snapshot)
        for generation, velocity in sorted(pending_command.pop(t, []), key=lambda item: item[0]):
            if active_command is None or generation > active_command[0]:
                active_command = (generation, velocity)

        decision = decisions[t]
        if decision["sense"]:
            delivered, delay = _link("sense", scenario, t)
            arrival = t + delay
            if delivered and arrival < HORIZON:
                snapshot = _obstacle_centres(scenario, t)
                if delay == 0:
                    if active_map is None or t > active_map[0]:
                        active_map = (t, snapshot)
                else:
                    pending_sense[arrival].append((t, snapshot))

        if decision["transmit"]:
            velocity = (float(decision["velocity"][0]), float(decision["velocity"][1]))
            delivered, delay = _link("command", scenario, t)
            arrival = t + delay
            if delivered and arrival < HORIZON:
                if delay == 0:
                    if active_command is None or t > active_command[0]:
                        active_command = (t, velocity)
                else:
                    pending_command[arrival].append((t, velocity))

        if active_command is not None and t - active_command[0] > AGE_LIMIT:
            active_command = None
        command_age = None if active_command is None else t - active_command[0]
        commanded = active_command[1] if active_command is not None else (0.0, 0.0)
        map_age = None if active_map is None else t - active_map[0]
        safety_stop = False
        executed = commanded
        if active_map is not None and map_age is not None and map_age <= MAP_VALID_AGE:
            predicted = _obstacle_centres(scenario, t + 1)
            candidate = (position[0] + commanded[0], position[1] + commanded[1])
            if any(math.dist(candidate, centre) <= SAFETY_RADIUS + EPS for centre in predicted):
                executed = (0.0, 0.0)
                safety_stop = True
        post_position = (position[0] + executed[0], position[1] + executed[1])
        true_centres = _obstacle_centres(scenario, t + 1)
        collision = any(math.dist(post_position, centre) <= COLLISION_RADIUS + EPS for centre in true_centres)
        trace.append(
            {
                "t": t,
                "position": [position[0], position[1]],
                "active_sequence": None if active_command is None else active_command[0],
                "command_age": command_age,
                "map_sequence": None if active_map is None else active_map[0],
                "map_age": map_age,
                "executed_velocity": [executed[0], executed[1]],
                "safety_stop": safety_stop,
                "collision": collision,
            }
        )
        position[0], position[1] = post_position
        positions.append(post_position)

    waypoints: list[dict[str, Any]] = []
    for waypoint, deadline in zip(range(1, 5), DEADLINES[1:]):
        point = positions[deadline]
        target = WAYPOINTS[scenario][waypoint]
        distance = math.dist(point, target)
        waypoints.append(
            {
                "waypoint": waypoint,
                "t": deadline,
                "position": [point[0], point[1]],
                "distance": distance,
                "success": distance <= 1.0 + WAYPOINT_TOL,
            }
        )
    return {"trace": trace, "positions": positions, "waypoints": waypoints}


def _scenario_record_shape(record: Any) -> bool:
    return isinstance(record, dict) and set(record) == {"scenario", "decisions", "trace", "waypoints"}


def _decision_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"t", "sense", "transmit", "velocity"}


def _trace_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {
        "t", "position", "active_sequence", "command_age", "map_sequence", "map_age",
        "executed_velocity", "safety_stop", "collision",
    }


def _waypoint_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"waypoint", "t", "position", "distance", "success"}


def _decision_valid(row: Any) -> bool:
    if not _decision_shape(row) or not _integer(row.get("t")) or not 0 <= row["t"] < HORIZON:
        return False
    if not isinstance(row.get("sense"), bool) or not isinstance(row.get("transmit"), bool):
        return False
    if not row["transmit"]:
        return row.get("velocity") is None
    velocity = row.get("velocity")
    return (
        _pair(velocity)
        and all(-1.0 - EPS <= float(value) <= 1.0 + EPS for value in velocity)
        and sum(float(value) ** 2 for value in velocity) <= 1.0 + EPS
    )


def _trace_valid(row: Any) -> bool:
    return (
        _trace_shape(row)
        and _integer(row.get("t"))
        and 0 <= row["t"] < HORIZON
        and _pair(row.get("position"))
        and _pair(row.get("executed_velocity"))
        and (row.get("active_sequence") is None or (_integer(row.get("active_sequence")) and 0 <= row["active_sequence"] < HORIZON))
        and (row.get("map_sequence") is None or (_integer(row.get("map_sequence")) and 0 <= row["map_sequence"] < HORIZON))
        and (row.get("command_age") is None or (_integer(row.get("command_age")) and 0 <= row["command_age"] <= AGE_LIMIT))
        and (row.get("map_age") is None or (_integer(row.get("map_age")) and row["map_age"] >= 0))
        and isinstance(row.get("safety_stop"), bool)
        and isinstance(row.get("collision"), bool)
    )


def _waypoint_valid(row: Any) -> bool:
    return (
        _waypoint_shape(row)
        and _integer(row.get("waypoint"))
        and _integer(row.get("t"))
        and (row["waypoint"], row["t"]) in WAYPOINT_KEYS
        and _pair(row.get("position"))
        and _finite_number(row.get("distance"))
        and isinstance(row.get("success"), bool)
    )


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
        waypoint_rows = record.get("waypoints") if isinstance(record.get("waypoints"), list) else []
        decisions: dict[int, Any] = {}
        traces: dict[int, Any] = {}
        waypoint_map: dict[tuple[int, int], Any] = {}
        for row in decisions_rows:
            if _decision_shape(row) and _integer(row.get("t")) and 0 <= row["t"] < HORIZON:
                decisions.setdefault(row["t"], row)
        for row in trace_rows:
            if _trace_shape(row) and _integer(row.get("t")) and 0 <= row["t"] < HORIZON:
                traces.setdefault(row["t"], row)
        for row in waypoint_rows:
            if _waypoint_shape(row) and _integer(row.get("waypoint")) and _integer(row.get("t")):
                waypoint_map.setdefault((row["waypoint"], row["t"]), row)
        indexed[name] = {
            "record": record,
            "decisions_rows": decisions_rows,
            "trace_rows": trace_rows,
            "waypoint_rows": waypoint_rows,
            "decisions": decisions,
            "traces": traces,
            "waypoint_map": waypoint_map,
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
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


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
    scenarios_list = root_shape and isinstance(data.get("scenarios"), list)
    return [
        _atom("result_json", "parse", bool(parse_ok), f"results.json: strict JSON object; actual_type={type(data).__name__}"),
        _atom("result_json", "root", bool(root_shape), f"results.json: expected root keys={{'scenarios'}}; actual_keys={sorted(data) if isinstance(data, dict) else None}"),
        _atom("result_json", "scenarios", bool(scenarios_list), f"results.json.scenarios: expected array; actual_type={type(data.get('scenarios')).__name__ if isinstance(data, dict) else None}"),
    ]


def _complete_scenarios(scenarios: Mapping[str, dict[str, Any]]) -> bool:
    return set(scenarios) == set(SCENARIO_NAMES)


def _every_rows(
    scenarios: Mapping[str, dict[str, Any]], field: str, predicate: Any, expectation: str,
) -> tuple[bool, str]:
    names = [name for name in SCENARIO_NAMES if name in scenarios]
    if not names:
        return False, "scenarios: no known canonical scenario record is present"
    for name in names:
        rows = scenarios[name]["record"].get(field)
        if not isinstance(rows, list):
            return False, f"{name}.{field}: expected a list"
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not predicate(row):
                return False, f"{name}.{field}[{index}]: expected {expectation}; actual={row!r}"
    return True, f"{field}: all supplied records satisfy {expectation}"


def _evidence_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    names = [name for name in SCENARIO_NAMES if name in scenarios]
    scenario_ok = bool(names) and all(_scenario_record_shape(scenarios[name]["record"]) for name in names)
    decision_shape, decision_shape_detail = _every_rows(scenarios, "decisions", _decision_shape, "keys {'t','sense','transmit','velocity'}")
    decision_types, decision_types_detail = _every_rows(
        scenarios, "decisions",
        lambda row: _integer(row.get("t")) and isinstance(row.get("sense"), bool) and isinstance(row.get("transmit"), bool) and (row.get("velocity") is None or _pair(row.get("velocity"))),
        "integer t, boolean flags, and finite velocity pair or null",
    )
    trace_shape, trace_shape_detail = _every_rows(scenarios, "trace", _trace_shape, "the declared cache and outcome keys")
    trace_values, trace_values_detail = _every_rows(scenarios, "trace", lambda row: _integer(row.get("t")) and _pair(row.get("position")) and _pair(row.get("executed_velocity")), "integer t and finite position and executed velocity")
    trace_state, trace_state_detail = _every_rows(
        scenarios, "trace",
        lambda row: (row.get("active_sequence") is None or (_integer(row.get("active_sequence")) and 0 <= row["active_sequence"] < HORIZON))
        and (row.get("map_sequence") is None or (_integer(row.get("map_sequence")) and 0 <= row["map_sequence"] < HORIZON))
        and (row.get("command_age") is None or (_integer(row.get("command_age")) and 0 <= row["command_age"] <= AGE_LIMIT))
        and (row.get("map_age") is None or (_integer(row.get("map_age")) and row["map_age"] >= 0)),
        "cache sequence and age fields in their domains",
    )
    trace_flags, trace_flags_detail = _every_rows(scenarios, "trace", lambda row: isinstance(row.get("safety_stop"), bool) and isinstance(row.get("collision"), bool), "boolean safety_stop and collision flags")
    waypoint_shape, waypoint_shape_detail = _every_rows(scenarios, "waypoints", _waypoint_shape, "keys {'waypoint','t','position','distance','success'}")
    waypoint_values, waypoint_values_detail = _every_rows(scenarios, "waypoints", lambda row: _integer(row.get("waypoint")) and _integer(row.get("t")) and _pair(row.get("position")) and _finite_number(row.get("distance")) and isinstance(row.get("success"), bool), "typed waypoint index, position, distance and success")
    return [
        _atom("result_evidence", "scenario_shape", scenario_ok, "scenario records have the declared key set"),
        _atom("result_evidence", "decision_shape", decision_shape, decision_shape_detail),
        _atom("result_evidence", "decision_types", decision_types, decision_types_detail),
        _atom("result_evidence", "trace_shape", trace_shape, trace_shape_detail),
        _atom("result_evidence", "trace_values", trace_values, trace_values_detail),
        _atom("result_evidence", "trace_state", trace_state, trace_state_detail),
        _atom("result_evidence", "trace_flags", trace_flags, trace_flags_detail),
        _atom("result_evidence", "waypoint_shape", waypoint_shape, waypoint_shape_detail),
        _atom("result_evidence", "waypoint_values", waypoint_values, waypoint_values_detail),
    ]


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = bool(scenarios)
    rows_by_name: dict[str, list[Any]] = {}
    for name in SCENARIO_NAMES:
        rows = scenarios.get(name, {}).get("decisions_rows", [])
        rows_by_name[name] = rows
    all_rows = [row for name, rows in rows_by_name.items() if name in scenarios for row in rows]
    rows_well_formed = complete and bool(all_rows) and all(isinstance(row, dict) for row in all_rows)
    time_ok = rows_well_formed and all(_integer(row.get("t")) and 0 <= row["t"] < HORIZON for row in all_rows)
    flags_ok = rows_well_formed and all(isinstance(row.get("sense"), bool) and isinstance(row.get("transmit"), bool) for row in all_rows)
    null_ok = rows_well_formed and all((not row.get("transmit") and row.get("velocity") is None) or row.get("transmit") for row in all_rows)
    pair_ok = rows_well_formed and all((not row.get("transmit")) or _pair(row.get("velocity")) for row in all_rows)
    box_ok = pair_ok and all(all(-1.0 - EPS <= float(value) <= 1.0 + EPS for value in row["velocity"]) for rows in rows_by_name.values() for row in rows if isinstance(row, dict) and row.get("transmit") and _pair(row.get("velocity")))
    norm_ok = pair_ok and all(sum(float(value) ** 2 for value in row["velocity"]) <= 1.0 + EPS for rows in rows_by_name.values() for row in rows if isinstance(row, dict) and row.get("transmit") and _pair(row.get("velocity")))
    budget_ok = rows_well_formed
    budget_detail = "all scenarios use at most 18 generated packets"
    for name, rows in rows_by_name.items():
        if not all(isinstance(row, dict) for row in rows):
            budget_ok = False
            budget_detail = f"{name}.decisions: every decision must be an object"
            break
        count = sum(int(row.get("sense") is True) + int(row.get("transmit") is True) for row in rows)
        if count > SIGNAL_BUDGET:
            budget_ok = False
            budget_detail = f"{name}: expected at most {SIGNAL_BUDGET} packets; actual={count}"
            break
    burst_ok = rows_well_formed
    burst_detail = "every four-slot window contains at most three packets"
    for name, rows in rows_by_name.items():
        by_t = {row.get("t"): row for row in rows if isinstance(row, dict) and _integer(row.get("t"))}
        for start in range(HORIZON - BURST_WINDOW + 1):
            count = sum(int(by_t[t].get("sense") is True) + int(by_t[t].get("transmit") is True) for t in range(start, start + BURST_WINDOW) if isinstance(by_t.get(t), dict))
            if count > BURST_LIMIT:
                burst_ok = False
                burst_detail = f"{name}.t={start}:{start + BURST_WINDOW}: expected at most {BURST_LIMIT}; actual={count}"
                break
        if not burst_ok:
            break
    continuity_ok = rows_well_formed
    continuity_detail = "successive generated command velocities obey the 0.8 m/s limit"
    for name, rows in rows_by_name.items():
        ordered = sorted((row for row in rows if isinstance(row, dict) and _integer(row.get("t")) and row.get("transmit") is True and _pair(row.get("velocity"))), key=lambda row: row["t"])
        for previous, current in zip(ordered, ordered[1:]):
            distance = math.dist(previous["velocity"], current["velocity"])
            if distance > VELOCITY_STEP_LIMIT + EPS:
                continuity_ok = False
                continuity_detail = f"{name}.t={previous['t']},{current['t']}: expected velocity distance <= {VELOCITY_STEP_LIMIT}; actual={distance:.9g}; tolerance={EPS}"
                break
        if not continuity_ok:
            break
    details = {
        "time_domain": "decision t values are in [0,24)",
        "transmission_flags": "sense and transmit are booleans",
        "null_payload": "non-transmitted commands have null velocity",
        "velocity_pair": "each transmitted velocity is a finite pair",
        "velocity_box": "transmitted velocity components lie in [-1,1]",
        "velocity_norm": "transmitted velocity norm is at most 1 m/s",
    }
    return [
        _atom("decision_legality", "time_domain", time_ok, details["time_domain"] if time_ok else "a decision has an invalid t"),
        _atom("decision_legality", "transmission_flags", flags_ok, details["transmission_flags"] if flags_ok else "a decision has a non-boolean transmission flag"),
        _atom("decision_legality", "null_payload", null_ok, details["null_payload"] if null_ok else "a non-transmitted decision carries a velocity"),
        _atom("decision_legality", "velocity_pair", pair_ok, details["velocity_pair"] if pair_ok else "a transmitted decision lacks a finite two-number velocity"),
        _atom("decision_legality", "velocity_box", box_ok, details["velocity_box"] if box_ok else "a velocity component is outside [-1,1] m/s"),
        _atom("decision_legality", "velocity_norm", norm_ok, details["velocity_norm"] if norm_ok else "a velocity norm exceeds 1 m/s"),
        _atom("decision_legality", "signal_budget", budget_ok, budget_detail),
        _atom("decision_legality", "burst_limit", burst_ok, burst_detail),
        _atom("decision_legality", "velocity_continuity", continuity_ok, continuity_detail),
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not a list"]
    records = data["scenarios"]
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    for index, row in enumerate(records):
        name = row.get("scenario") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key")
        else:
            by_name[name].append(row)

    def count_keys(record: dict[str, Any], field: str, key_fn: Any, expected: set[Any], label: str) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.append(f"{label}: missing list")
            return
        counts = {key: 0 for key in expected}
        for index, row in enumerate(rows):
            key = key_fn(row) if isinstance(row, dict) else None
            if key in counts:
                counts[key] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key")
        defects.extend(f"{label}: missing key {key}" for key, count in counts.items() if count == 0)
        defects.extend(f"{label}: duplicate key {key}" for key, count in counts.items() for _ in range(max(0, count - 1)))

    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(record, "decisions", lambda row: row.get("t") if _integer(row.get("t")) else None, set(range(HORIZON)), f"scenario {name}.decisions")
        count_keys(record, "trace", lambda row: row.get("t") if _integer(row.get("t")) else None, set(range(HORIZON)), f"scenario {name}.trace")
        count_keys(record, "waypoints", lambda row: (row.get("waypoint"), row.get("t")) if _integer(row.get("waypoint")) and _integer(row.get("t")) else None, set(WAYPOINT_KEYS), f"scenario {name}.waypoints")
    # The canonical check has eight half-point defect units; excess defects are still diagnostic but score is clipped.
    return max(0.0, 1.0 - len(defects) / 8.0), defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return f"failed_atoms=[canonical_coverage.defect]; defect_count={len(defects)}; {shown}; normalized={score:.9g}"


def _replay_atoms(scenarios: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    names = [name.removeprefix("replay_validity.") for name in ATOM_SHARES["replay_validity"]]
    checks = {name: True for name in names}
    details = {name: "" for name in names}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    present = [scenario for scenario in SCENARIO_NAMES if scenario in scenarios]
    if not present:
        for rule in names:
            fail(rule, "no known canonical scenario record is present")
    else:
        for scenario in present:
            item = scenarios[scenario]
            decisions = item["decisions"]
            traces = item["traces"]
            waypoints = item["waypoint_map"]
            if set(decisions) != set(range(HORIZON)) or any(not _decision_valid(decisions[t]) for t in range(HORIZON)):
                for rule in names:
                    fail(rule, f"{scenario}.decisions: complete valid decision map required")
                continue
            try:
                replay = _replay_scenario(scenario, decisions)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                for rule in names:
                    fail(rule, f"{scenario}.decisions: replay input rejected: {type(exc).__name__}: {exc}")
                continue
            replayed[scenario] = replay
            if set(traces) != set(range(HORIZON)) or any(not _trace_valid(traces[t]) for t in range(HORIZON)):
                for rule in names[:8]:
                    fail(rule, f"{scenario}.trace: complete valid trace required")
            else:
                for t in range(HORIZON):
                    actual = traces[t]
                    wanted = replay["trace"][t]
                    if not _close_pair(actual.get("position"), wanted["position"]):
                        fail("trace_position", f"{scenario}.trace[{t}].position: expected={wanted['position']}; actual={actual.get('position')}; tolerance={EPS}")
                    if actual.get("active_sequence") != wanted["active_sequence"]:
                        fail("command_sequence", f"{scenario}.trace[{t}].active_sequence: expected={wanted['active_sequence']}; actual={actual.get('active_sequence')}")
                    if actual.get("command_age") != wanted["command_age"]:
                        fail("command_age", f"{scenario}.trace[{t}].command_age: expected={wanted['command_age']}; actual={actual.get('command_age')}")
                    if actual.get("map_sequence") != wanted["map_sequence"]:
                        fail("map_sequence", f"{scenario}.trace[{t}].map_sequence: expected={wanted['map_sequence']}; actual={actual.get('map_sequence')}")
                    if actual.get("map_age") != wanted["map_age"]:
                        fail("map_age", f"{scenario}.trace[{t}].map_age: expected={wanted['map_age']}; actual={actual.get('map_age')}")
                    if actual.get("safety_stop") is not wanted["safety_stop"]:
                        fail("safety_stop", f"{scenario}.trace[{t}].safety_stop: expected={wanted['safety_stop']}; actual={actual.get('safety_stop')}")
                    if not _close_pair(actual.get("executed_velocity"), wanted["executed_velocity"]):
                        fail("executed_velocity", f"{scenario}.trace[{t}].executed_velocity: expected={wanted['executed_velocity']}; actual={actual.get('executed_velocity')}; tolerance={EPS}")
                    if actual.get("collision") is not wanted["collision"]:
                        fail("collision", f"{scenario}.trace[{t}].collision: expected={wanted['collision']}; actual={actual.get('collision')}")
            if set(waypoints) != set(WAYPOINT_KEYS) or any(not _waypoint_valid(waypoints[key]) for key in WAYPOINT_KEYS):
                fail("waypoint_geometry", f"{scenario}.waypoints: complete valid waypoint records required")
                fail("waypoint_success", f"{scenario}.waypoints: complete valid waypoint records required")
            else:
                wanted_by_key = {(row["waypoint"], row["t"]): row for row in replay["waypoints"]}
                for key in WAYPOINT_KEYS:
                    wanted = wanted_by_key[key]
                    actual = waypoints[key]
                    if not _close_pair(actual.get("position"), wanted["position"]) or not _close(actual.get("distance"), wanted["distance"]):
                        fail("waypoint_geometry", f"{scenario}.waypoints{key}: expected position/distance={wanted['position'], wanted['distance']}; actual={actual.get('position'), actual.get('distance')}; tolerance={EPS}")
                    if actual.get("success") is not wanted["success"]:
                        fail("waypoint_success", f"{scenario}.waypoints{key}.success: expected={wanted['success']}; actual={actual.get('success')}")
    atoms = [_atom("replay_validity", rule, checks[rule], details[rule] or "fixed model comparison passed") for rule in names]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]], records: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    collision_free = 0
    total_slots = len(SCENARIO_NAMES) * HORIZON
    successes = 0
    mse = 0.0
    signal_count = 0
    freshness = 0.0
    for scenario in SCENARIO_NAMES:
        result = replayed[scenario]
        for row in result["trace"]:
            collision_free += int(not row["collision"])
            age = row["map_age"]
            freshness += max(0.0, 1.0 - age / MAP_VALID_AGE) if age is not None and age <= MAP_VALID_AGE else 0.0
        for t in range(1, HORIZON + 1):
            point = result["positions"][t]
            reference = _reference(scenario, t)
            mse += (point[0] - reference[0]) ** 2 + (point[1] - reference[1]) ** 2
        successes += sum(int(row["success"]) for row in result["waypoints"])
        for row in records[scenario]["decisions"].values():
            signal_count += int(row["sense"] is True) + int(row["transmit"] is True)
    metrics = {
        "collision_free_rate": collision_free / total_slots,
        "waypoint_completion": successes / (len(SCENARIO_NAMES) * 4),
        "tracking_quality": max(0.0, 1.0 - (mse / total_slots) / 64.0),
        "signal_efficiency": max(0.0, min(1.0, 1.0 - signal_count / 90.0)),
        "map_freshness": freshness / total_slots,
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
    return max(0.0, 1.0 - len(missing) / 4.0), (f"missing_sections={missing}" if missing else "missing_sections=[]")


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    started = time.monotonic()
    remove_ok = True
    try:
        state.run_started = True
        try:
            result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            removed = getattr(result, "exit_code", None) == 0
            absence = await ws.run(
                "test ! -e results.json && test ! -L results.json",
                check=False,
                timeout=30.0,
            )
            remove_ok = removed and getattr(absence, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
            remove_ok = False
            state.official_message = f"could not remove previous results.json: {exc}"
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
        state.official_ok = remove_ok and state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600.0 + EPS and state.result_file_is_new_regular
        if not state.official_message:
            state.official_message = f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}" if state.official_ok else f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
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
        state.scenarios = _index_records(state.data)
        state.records = state.scenarios
        evidence_atoms = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(evidence_atoms, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
        coverage_score, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage_score
        state.diagnostics["canonical_coverage"] = _coverage_diagnostic(coverage_score, defects)
        legality_atoms = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(legality_atoms, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
        replay_atoms, replayed = _replay_atoms(state.scenarios)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
        gate_names = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        state.validity_gate = state.official_ok and all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in gate_names)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.scenarios)
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
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a regular non-symlink file"
    try:
        text = ws.read_text("report.md")
    except (OSError, TypeError, ValueError) as exc:
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"
    return _report_score(text)


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
    return await _metric_check(ws, "collision_free_rate")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "waypoint_completion")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "tracking_quality")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "signal_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "map_freshness")
