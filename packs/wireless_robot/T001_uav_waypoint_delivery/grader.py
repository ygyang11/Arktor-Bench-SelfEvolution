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
AGE_LIMIT = 4
EPS = 1e-6
WAYPOINT_TOL = 1e-6
TRANSMISSION_BUDGET = 12
BURST_WINDOW = 4
BURST_LIMIT = 3
VELOCITY_STEP_LIMIT = 0.75
SCENARIO_NAMES = (
    "steady_square",
    "crossing_route",
    "zigzag_route",
    "late_turn",
    "radial_return",
)
DEADLINES = (0, 6, 12, 18, 24)
WAYPOINT_KEYS = frozenset(zip(range(1, 5), DEADLINES[1:]))
WAYPOINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "steady_square": ((0, 0), (4, 0), (4, 4), (0, 4), (0, 0)),
    "crossing_route": ((0, 0), (3, 3), (-3, 3), (-3, -3), (3, -3)),
    "zigzag_route": ((0, 0), (-3, 2), (2, 4), (-2, 5), (0, 0)),
    "late_turn": ((0, 0), (1, 0), (2, 0), (2, 5), (-3, 5)),
    "radial_return": ((0, 0), (-2, -2), (0, -4), (4, -4), (0, 0)),
}
LOST: dict[str, set[int]] = {
    "steady_square": set(),
    "crossing_route": {5, 6, 7, 8},
    "zigzag_route": {2, 3, 4, 14, 15, 16, 22},
    "late_turn": {0, 1, 12, 13, 14, 20},
    "radial_return": {3, 4, 5, 18, 19},
}


# Atom identities and relative shares are task-specific. Every mapping sums to one.
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
        "result_evidence.trace_state": 1.0 / 8.0,
        "result_evidence.waypoint_shape": 1.0 / 8.0,
        "result_evidence.waypoint_values": 1.0 / 8.0,
    },
    "decision_legality": {
        "decision_legality.time_domain": 1.0 / 9.0,
        "decision_legality.transmit_flag": 1.0 / 9.0,
        "decision_legality.null_payload": 1.0 / 9.0,
        "decision_legality.velocity_pair": 1.0 / 9.0,
        "decision_legality.velocity_box": 1.0 / 9.0,
        "decision_legality.velocity_norm": 1.0 / 9.0,
        "decision_legality.resource_budget": 1.0 / 9.0,
        "decision_legality.burst_limit": 1.0 / 9.0,
        "decision_legality.velocity_continuity": 1.0 / 9.0,
    },
    "replay_validity": {
        "replay_validity.trace_position": 1.0 / 8.0,
        "replay_validity.command_admission": 1.0 / 8.0,
        "replay_validity.command_age_expiry": 1.0 / 8.0,
        "replay_validity.executed_velocity": 1.0 / 8.0,
        "replay_validity.waypoint_position": 1.0 / 8.0,
        "replay_validity.waypoint_distance": 1.0 / 8.0,
        "replay_validity.waypoint_success": 1.0 / 8.0,
        "replay_validity.deadline_rule": 1.0 / 8.0,
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
        self.replayed: dict[Hashable, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _link(scenario: str, t: int) -> tuple[bool, int]:
    """Return the fixed delivery flag and delay for a generated packet."""
    if scenario == "crossing_route":
        delay = 2 if 9 <= t <= 12 else 0
    elif scenario == "zigzag_route":
        delay = 1 if t % 2 == 0 else 0
    elif scenario == "late_turn":
        delay = 3 if t in {4, 5, 6, 7, 16, 17} else 1
    elif scenario == "radial_return":
        delay = 2 if t % 2 == 0 else 0
    else:
        delay = 0
    return t not in LOST[scenario], delay


def _reference(scenario: str, t: int) -> tuple[float, float]:
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
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_finite_number(x) for x in value)


def _close(a: Any, b: Any, tolerance: float = EPS) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tolerance


def _close_pair(a: Any, b: Sequence[float], tolerance: float = EPS) -> bool:
    return (
        _pair(a)
        and len(b) == 2
        and _close(a[0], b[0], tolerance)
        and _close(a[1], b[1], tolerance)
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


def _replay_scenario(scenario: str, decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Replay the fixed downlink event model from a complete decision map."""
    pending: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    active: tuple[int, tuple[float, float]] | None = None
    position = [0.0, 0.0]
    trace: list[dict[str, Any]] = []
    positions: list[tuple[float, float]] = [(0.0, 0.0)]

    for t in range(HORIZON):
        for generation, velocity in pending.pop(t, []):
            if active is None or generation > active[0]:
                active = (generation, velocity)

        decision = decisions[t]
        if decision["transmit"]:
            velocity = (float(decision["velocity"][0]), float(decision["velocity"][1]))
            delivered, delay = _link(scenario, t)
            arrival = t + delay
            if delivered and arrival < HORIZON:
                if delay == 0:
                    if active is None or t > active[0]:
                        active = (t, velocity)
                else:
                    pending[arrival].append((t, velocity))

        if active is not None and t - active[0] > AGE_LIMIT:
            active = None
        if active is None:
            sequence: int | None = None
            age: int | None = None
            executed = (0.0, 0.0)
        else:
            sequence = active[0]
            age = t - active[0]
            executed = active[1]

        trace.append({
            "t": t,
            "position": [position[0], position[1]],
            "active_sequence": sequence,
            "command_age": age,
            "executed_velocity": [executed[0], executed[1]],
        })
        position[0] += executed[0]
        position[1] += executed[1]
        positions.append((position[0], position[1]))

    return {"trace": trace, "positions": positions}


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


def _scenario_record_shape(record: Any) -> bool:
    return isinstance(record, dict) and set(record) == {"scenario", "decisions", "trace", "waypoints"}


def _decision_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"t", "transmit", "velocity"}


def _trace_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {
        "t", "position", "active_sequence", "command_age", "executed_velocity",
    }


def _waypoint_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"waypoint", "t", "position", "distance", "success"}


def _decision_valid(row: Any) -> bool:
    if not _decision_shape(row) or not _integer(row.get("t")):
        return False
    if not 0 <= row["t"] < HORIZON or not isinstance(row.get("transmit"), bool):
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
        and (
            row.get("active_sequence") is None
            or (_integer(row.get("active_sequence")) and 0 <= row["active_sequence"] < HORIZON)
        )
        and (
            row.get("command_age") is None
            or (_integer(row.get("command_age")) and 0 <= row["command_age"] <= AGE_LIMIT)
        )
    )


def _waypoint_valid(row: Any) -> bool:
    return (
        _waypoint_shape(row)
        and _integer(row.get("waypoint"))
        and 1 <= row["waypoint"] <= 4
        and _integer(row.get("t"))
        and (row["waypoint"], row["t"]) in WAYPOINT_KEYS
        and _pair(row.get("position"))
        and _finite_number(row.get("distance"))
        and isinstance(row.get("success"), bool)
    )


def _constraint_checks(rows: Sequence[Any]) -> tuple[dict[str, bool], dict[str, str]]:
    checks = {name: False for name in ("budget", "burst", "continuity")}
    details = {name: "" for name in checks}
    ordered = sorted(
        (
            row for row in rows
            if _decision_shape(row) and _integer(row.get("t")) and 0 <= row["t"] < HORIZON
        ),
        key=lambda row: row["t"],
    )
    transmitted = [row for row in ordered if row.get("transmit") is True]
    checks["budget"] = len(transmitted) <= TRANSMISSION_BUDGET
    if not checks["budget"]:
        details["budget"] = (
            f"decisions: expected at most {TRANSMISSION_BUDGET} transmissions; actual={len(transmitted)}"
        )
    checks["burst"] = True
    for start in range(0, HORIZON - BURST_WINDOW + 1):
        count = sum(
            row.get("transmit") is True
            for row in ordered
            if start <= row.get("t", -1) < start + BURST_WINDOW
        )
        if count > BURST_LIMIT:
            checks["burst"] = False
            details["burst"] = (
                f"decisions[t={start}:{start + BURST_WINDOW}]: expected at most "
                f"{BURST_LIMIT} transmissions; actual={count}"
            )
            break
    checks["continuity"] = True
    valid_transmitted = [
        row for row in transmitted
        if _pair(row.get("velocity"))
        and all(-1.0 - EPS <= float(value) <= 1.0 + EPS for value in row["velocity"])
        and sum(float(value) ** 2 for value in row["velocity"]) <= 1.0 + EPS
    ]
    for previous, current in zip(valid_transmitted, valid_transmitted[1:]):
        distance = math.dist(previous["velocity"], current["velocity"])
        if distance > VELOCITY_STEP_LIMIT + EPS:
            checks["continuity"] = False
            details["continuity"] = (
                f"decisions[t={previous['t']},t={current['t']}].velocity: expected distance <= "
                f"{VELOCITY_STEP_LIMIT} m/s; actual={distance:.9g}; tolerance={EPS}"
            )
            break
    return checks, details


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
    score = sum(atom.share for atom in atoms if atom.passed)
    return 1.0 if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-12) else max(0.0, min(1.0, score))


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
        _atom(
            "result_json", "parse", bool(parse_ok),
            f"results.json: expected a strict JSON object; actual_type={type(data).__name__}",
        ),
        _atom(
            "result_json", "root", bool(root_shape),
            f"results.json: expected root keys={{'scenarios'}}; "
            f"actual_keys={sorted(data) if isinstance(data, dict) else None}",
        ),
        _atom(
            "result_json", "scenarios",
            bool(scenarios_list),
            "results.json.scenarios: expected a top-level array; "
            f"actual_type={type(data.get('scenarios')).__name__ if isinstance(data, dict) else None}",
        ),
    ]


def _complete_scenarios(scenarios: Mapping[str, dict[str, Any]]) -> bool:
    return set(scenarios) == set(SCENARIO_NAMES)


def _every_rows(
    scenarios: Mapping[str, dict[str, Any]],
    field: str,
    predicate: Any,
    expectation: str,
) -> tuple[bool, str]:
    if not _complete_scenarios(scenarios):
        return False, "scenarios: all canonical scenario records are required"
    for name in SCENARIO_NAMES:
        item = scenarios[name]
        rows = item["record"].get(field)
        if not isinstance(rows, list):
            return False, f"{name}.{field}: expected a list"
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not predicate(row):
                return False, f"{name}.{field}[{index}]: expected {expectation}; actual={row!r}"
    return True, f"{field}: all supplied records satisfy {expectation}"


def _evidence_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = _complete_scenarios(scenarios)
    scenario_ok = complete and all(_scenario_record_shape(scenarios[name]["record"]) for name in SCENARIO_NAMES)
    decision_shape, decision_shape_detail = _every_rows(
        scenarios, "decisions", _decision_shape, "keys {'t','transmit','velocity'}")
    decision_types, decision_types_detail = _every_rows(
        scenarios,
        "decisions",
        lambda row: _integer(row.get("t")) and isinstance(row.get("transmit"), bool)
        and (row.get("velocity") is None or _pair(row.get("velocity"))),
        "integer t, boolean transmit, and finite velocity pair or null",
    )
    trace_shape, trace_shape_detail = _every_rows(
        scenarios, "trace", _trace_shape, "the declared trace keys")
    trace_values, trace_values_detail = _every_rows(
        scenarios,
        "trace",
        lambda row: _integer(row.get("t")) and _pair(row.get("position"))
        and _pair(row.get("executed_velocity")),
        "integer t and finite position and executed_velocity pairs",
    )
    trace_state, trace_state_detail = _every_rows(
        scenarios,
        "trace",
        lambda row: (
            row.get("active_sequence") is None
            or (_integer(row.get("active_sequence")) and 0 <= row["active_sequence"] < HORIZON)
        ) and (
            row.get("command_age") is None
            or (_integer(row.get("command_age")) and 0 <= row["command_age"] <= AGE_LIMIT)
        ),
        "active sequence and command age in their declared domains",
    )
    waypoint_shape, waypoint_shape_detail = _every_rows(
        scenarios, "waypoints", _waypoint_shape, "the declared waypoint keys")
    waypoint_values, waypoint_values_detail = _every_rows(
        scenarios, "waypoints", _waypoint_valid, "finite waypoint evidence in the declared domain")
    return [
        _atom("result_evidence", "scenario_shape", scenario_ok,
              "scenario records: expected exact keys {'scenario','decisions','trace','waypoints'}"),
        _atom("result_evidence", "decision_shape", decision_shape, decision_shape_detail),
        _atom("result_evidence", "decision_types", decision_types, decision_types_detail),
        _atom("result_evidence", "trace_shape", trace_shape, trace_shape_detail),
        _atom("result_evidence", "trace_values", trace_values, trace_values_detail),
        _atom("result_evidence", "trace_state", trace_state, trace_state_detail),
        _atom("result_evidence", "waypoint_shape", waypoint_shape, waypoint_shape_detail),
        _atom("result_evidence", "waypoint_values", waypoint_values, waypoint_values_detail),
    ]


def _all_decision_rows(
    scenarios: Mapping[str, dict[str, Any]], predicate: Any, expectation: str,
) -> tuple[bool, str]:
    if not _complete_scenarios(scenarios):
        return False, "decisions: all canonical scenario records are required"
    for name in SCENARIO_NAMES:
        rows = scenarios[name]["decisions_rows"]
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not predicate(row):
                return False, f"{name}.decisions[{index}]: expected {expectation}; actual={row!r}"
    return True, f"decisions: all supplied rows satisfy {expectation}"


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    time_ok, time_detail = _all_decision_rows(
        scenarios, lambda row: _integer(row.get("t")) and 0 <= row["t"] < HORIZON,
        f"integer t in [0,{HORIZON})",
    )
    flag_ok, flag_detail = _all_decision_rows(
        scenarios, lambda row: isinstance(row.get("transmit"), bool), "boolean transmit",
    )
    null_ok, null_detail = _all_decision_rows(
        scenarios, lambda row: row.get("transmit") is not False or row.get("velocity") is None,
        "null velocity when transmit=false",
    )
    pair_ok, pair_detail = _all_decision_rows(
        scenarios, lambda row: row.get("transmit") is not True or _pair(row.get("velocity")),
        "a finite two-element velocity when transmit=true",
    )
    box_ok, box_detail = _all_decision_rows(
        scenarios,
        lambda row: row.get("transmit") is not True or (
            _pair(row.get("velocity"))
            and all(-1.0 - EPS <= float(value) <= 1.0 + EPS for value in row["velocity"])
        ),
        f"velocity components in [-1,1] with tolerance={EPS}",
    )
    norm_ok, norm_detail = _all_decision_rows(
        scenarios,
        lambda row: row.get("transmit") is not True or (
            _pair(row.get("velocity"))
            and sum(float(value) ** 2 for value in row["velocity"]) <= 1.0 + EPS
        ),
        f"squared velocity norm at most 1 with tolerance={EPS}",
    )
    constraint_by_name = {
        name: _constraint_checks(scenarios[name]["decisions_rows"])
        for name in SCENARIO_NAMES if name in scenarios
    }
    budget_ok = _complete_scenarios(scenarios) and all(
        constraint_by_name[name][0]["budget"] for name in SCENARIO_NAMES if name in constraint_by_name
    )
    burst_ok = _complete_scenarios(scenarios) and all(
        constraint_by_name[name][0]["burst"] for name in SCENARIO_NAMES if name in constraint_by_name
    )
    continuity_ok = _complete_scenarios(scenarios) and all(
        constraint_by_name[name][0]["continuity"] for name in SCENARIO_NAMES if name in constraint_by_name
    )
    budget_detail = next(
        (constraint_by_name[name][1]["budget"] for name in SCENARIO_NAMES
         if name in constraint_by_name and not constraint_by_name[name][0]["budget"]),
        f"decisions: each scenario has at most {TRANSMISSION_BUDGET} transmissions",
    )
    burst_detail = next(
        (constraint_by_name[name][1]["burst"] for name in SCENARIO_NAMES
         if name in constraint_by_name and not constraint_by_name[name][0]["burst"]),
        f"decisions: every {BURST_WINDOW}-slot block has at most {BURST_LIMIT} transmissions",
    )
    continuity_detail = next(
        (constraint_by_name[name][1]["continuity"] for name in SCENARIO_NAMES
         if name in constraint_by_name and not constraint_by_name[name][0]["continuity"]),
        f"decisions: successive transmitted velocities differ by at most {VELOCITY_STEP_LIMIT} m/s",
    )
    return [
        _atom("decision_legality", "time_domain", time_ok, time_detail),
        _atom("decision_legality", "transmit_flag", flag_ok, flag_detail),
        _atom("decision_legality", "null_payload", null_ok, null_detail),
        _atom("decision_legality", "velocity_pair", pair_ok, pair_detail),
        _atom("decision_legality", "velocity_box", box_ok, box_detail),
        _atom("decision_legality", "velocity_norm", norm_ok, norm_detail),
        _atom("decision_legality", "resource_budget", budget_ok, budget_detail),
        _atom("decision_legality", "burst_limit", burst_ok, burst_detail),
        _atom("decision_legality", "velocity_continuity", continuity_ok, continuity_detail),
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        defects.extend(["results.scenarios: missing or not a list"] * 8)
        return 0.0, defects
    records = data["scenarios"]
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    for index, record in enumerate(records):
        name = record.get("scenario") if isinstance(record, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key")
        else:
            by_name[name].append(record)

    def count_integer_keys(record: dict[str, Any], field: str, expected: set[int], label: str) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {key}" for key in sorted(expected))
            return
        counts = {key: 0 for key in expected}
        for index, row in enumerate(rows):
            key = row.get("t") if isinstance(row, dict) and _integer(row.get("t")) else None
            if key in counts:
                counts[key] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key")
        defects.extend(f"{label}: missing key {key}" for key, count in sorted(counts.items()) if count == 0)
        defects.extend(
            f"{label}: duplicate key {key}"
            for key, count in sorted(counts.items())
            for _ in range(max(0, count - 1))
        )

    def count_waypoints(record: dict[str, Any], label: str) -> None:
        rows = record.get("waypoints")
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {key}" for key in sorted(WAYPOINT_KEYS))
            return
        counts = {key: 0 for key in WAYPOINT_KEYS}
        for index, row in enumerate(rows):
            key = (
                (row.get("waypoint"), row.get("t"))
                if isinstance(row, dict) and _integer(row.get("waypoint")) and _integer(row.get("t"))
                else None
            )
            if key in counts:
                counts[key] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key")
        defects.extend(f"{label}: missing key {key}" for key, count in sorted(counts.items()) if count == 0)
        defects.extend(
            f"{label}: duplicate key {key}"
            for key, count in sorted(counts.items())
            for _ in range(max(0, count - 1))
        )

    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            defects.extend(f"scenario {name}.decisions: missing key {key}" for key in range(HORIZON))
            defects.extend(f"scenario {name}.trace: missing key {key}" for key in range(HORIZON))
            defects.extend(f"scenario {name}.waypoints: missing key {key}" for key in sorted(WAYPOINT_KEYS))
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_integer_keys(record, "decisions", set(range(HORIZON)), f"scenario {name}.decisions")
        count_integer_keys(record, "trace", set(range(HORIZON)), f"scenario {name}.trace")
        count_waypoints(record, f"scenario {name}.waypoints")
    score = max(0.0, 1.0 - len(defects) / 8.0)
    return score, defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return f"failed_atoms=[canonical_coverage.defect]; defect_count={len(defects)}; {shown}; normalized={score:.9g}"


def _replay_atoms(
    scenarios: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = tuple(ATOM_SHARES["replay_validity"])
    rule_names = [rule.removeprefix("replay_validity.") for rule in rules]
    checks = {name: True for name in rule_names}
    details = {name: "" for name in rule_names}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if not _complete_scenarios(scenarios):
        for rule in rule_names:
            fail(rule, "scenarios: all canonical scenario records are required")
    else:
        for name in SCENARIO_NAMES:
            item = scenarios[name]
            decisions = item["decisions"]
            constraints, constraint_details = _constraint_checks(item["decisions_rows"])
            decision_complete = (
                set(decisions) == set(range(HORIZON))
                and len(item["decisions_rows"]) == HORIZON
                and len({row.get("t") for row in item["decisions_rows"] if isinstance(row, dict)}) == HORIZON
                and all(_decision_valid(decisions[t]) for t in range(HORIZON))
                and all(constraints.values())
            )
            if not decision_complete:
                detail = f"{name}.decisions: complete legal decisions required"
                for key in ("budget", "burst", "continuity"):
                    if not constraints[key] and constraint_details[key]:
                        detail = constraint_details[key]
                        break
                for rule in rule_names:
                    fail(rule, detail)
                continue

            expected = _replay_scenario(name, decisions)
            replayed[name] = expected
            traces = item["traces"]
            trace_complete = (
                set(traces) == set(range(HORIZON))
                and len(item["trace_rows"]) == HORIZON
                and len({row.get("t") for row in item["trace_rows"] if isinstance(row, dict)}) == HORIZON
                and all(_trace_valid(traces[t]) for t in range(HORIZON))
            )
            if not trace_complete:
                for rule in ("trace_position", "command_admission", "command_age_expiry", "executed_velocity"):
                    fail(rule, f"{name}.trace: expected one valid row for every t in [0,{HORIZON})")
            else:
                for t in range(HORIZON):
                    actual = traces[t]
                    wanted = expected["trace"][t]
                    if not _close_pair(actual.get("position"), wanted["position"]):
                        fail("trace_position", f"{name}.trace[{t}].position: expected={wanted['position']}; actual={actual.get('position')}; tolerance={EPS}")
                    if actual.get("active_sequence") != wanted["active_sequence"]:
                        fail("command_admission", f"{name}.trace[{t}].active_sequence: expected={wanted['active_sequence']}; actual={actual.get('active_sequence')}")
                    if actual.get("command_age") != wanted["command_age"]:
                        fail("command_age_expiry", f"{name}.trace[{t}].command_age: expected={wanted['command_age']}; actual={actual.get('command_age')}")
                    if not _close_pair(actual.get("executed_velocity"), wanted["executed_velocity"]):
                        fail("executed_velocity", f"{name}.trace[{t}].executed_velocity: expected={wanted['executed_velocity']}; actual={actual.get('executed_velocity')}; tolerance={EPS}")

            waypoints = item["waypoint_map"]
            waypoint_complete = (
                set(waypoints) == WAYPOINT_KEYS
                and len(item["waypoint_rows"]) == len(WAYPOINT_KEYS)
                and len({(row.get("waypoint"), row.get("t")) for row in item["waypoint_rows"] if isinstance(row, dict)}) == len(WAYPOINT_KEYS)
                and all(_waypoint_valid(waypoints[key]) for key in WAYPOINT_KEYS)
            )
            if not waypoint_complete:
                for rule in ("waypoint_position", "waypoint_distance", "waypoint_success", "deadline_rule"):
                    fail(rule, f"{name}.waypoints: expected one valid row for each key in {sorted(WAYPOINT_KEYS)}")
                continue
            for index, t in enumerate(DEADLINES[1:], start=1):
                actual = waypoints[(index, t)]
                position = expected["positions"][t]
                target = WAYPOINTS[name][index]
                distance = math.hypot(position[0] - target[0], position[1] - target[1])
                if not _close_pair(actual.get("position"), position, WAYPOINT_TOL):
                    fail("waypoint_position", f"{name}.waypoints[{index},{t}].position: expected={list(position)}; actual={actual.get('position')}; tolerance={WAYPOINT_TOL}")
                if not _close(actual.get("distance"), distance, WAYPOINT_TOL):
                    fail("waypoint_distance", f"{name}.waypoints[{index},{t}].distance: expected={distance:.9g}; actual={actual.get('distance')}; tolerance={WAYPOINT_TOL}")
                expected_success = distance <= 1.0 + WAYPOINT_TOL
                if actual.get("success") is not expected_success:
                    fail("waypoint_success", f"{name}.waypoints[{index},{t}].success: expected={expected_success}; actual={actual.get('success')}")
                actual_distance = actual.get("distance")
                rule_success = _finite_number(actual_distance) and float(actual_distance) <= 1.0 + WAYPOINT_TOL
                if actual.get("success") is not rule_success:
                    fail("deadline_rule", f"{name}.waypoints[{index},{t}]: success must equal distance <= 1.0 m; distance={actual_distance}; tolerance={WAYPOINT_TOL}")

    atoms = [
        _atom("replay_validity", name, checks[name], details[name] or "fixed downlink model comparison passed")
        for name in rule_names
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]], scenarios: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    successes = 0
    squared_error = 0.0
    transmissions = 0
    freshness_sum = 0.0
    slots = len(SCENARIO_NAMES) * HORIZON
    for name in SCENARIO_NAMES:
        result = replayed[name]
        transmissions += sum(
            1 for row in scenarios[name]["decisions_rows"]
            if isinstance(row, dict) and row.get("transmit") is True
        )
        for t in range(1, HORIZON + 1):
            position = result["positions"][t]
            reference = _reference(name, t)
            squared_error += (position[0] - reference[0]) ** 2 + (position[1] - reference[1]) ** 2
        for row in result["trace"]:
            age = row["command_age"]
            freshness_sum += max(0.0, 1.0 - age / AGE_LIMIT) if age is not None else 0.0
        for index, t in enumerate(DEADLINES[1:], start=1):
            position = result["positions"][t]
            target = WAYPOINTS[name][index]
            successes += int(math.dist(position, target) <= 1.0 + WAYPOINT_TOL)
    tracking_mse = squared_error / slots
    metrics = {
        "completion_rate": successes / float(len(SCENARIO_NAMES) * len(WAYPOINT_KEYS)),
        "tracking_quality": max(0.0, min(1.0, 1.0 - tracking_mse / 1800.0)),
        "transmission_efficiency": max(0.0, min(1.0, 1.0 - transmissions / 60.0)),
        "freshness_quality": max(0.0, min(1.0, freshness_sum / slots)),
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _report_score(text: str | None) -> tuple[float, str]:
    required = (
        "Problem Formulation", "Method Design", "Experimental Setup", "Results",
        "Research Analysis", "Conclusion and Limitations",
    )
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    if missing:
        return score, f"missing_sections={missing}; normalized={score:.9g}"
    return score, "missing_sections=[]"


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = True
    started = time.monotonic()
    try:
        state.run_started = True
        try:
            remove_result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(remove_result, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001 - preserve failure in cached state
            remove_ok = False
            state.official_message = f"could not remove previous results.json: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = state.exit_code is None and "timeout" in stderr.lower()
        except Exception as exc:  # noqa: BLE001 - every path is cached
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
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok
                else f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
            )

        if state.result_file_is_new_regular:
            try:
                raw = ws.read_text("results.json")
                state.data = _strict_json(raw or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.parse_ok = False
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
        state.validity_gate = state.official_ok and all(
            state.scores.get(name, 0.0) >= 1.0 - EPS for name in gate_names
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.scenarios)
    except Exception as exc:  # noqa: BLE001 - cache a deterministic failed state
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
    return await _metric_check(ws, "completion_rate")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "tracking_quality")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "transmission_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "freshness_quality")
