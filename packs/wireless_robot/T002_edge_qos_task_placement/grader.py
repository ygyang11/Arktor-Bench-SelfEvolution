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


# Fixed task model.
WINDOWS = 16
CYCLES_PER_WINDOW = 4
CYCLES = WINDOWS * CYCLES_PER_WINDOW
JITTER = (0.0, 2.0, 1.0, 3.0)
DEADLINE_MS = 42.0
CPU_LIMIT = 90.0
LATENCY_BOUND_MS = 100.0
LATENCY_SCALE_MS = 85.0
MIGRATION_WORKLOAD = 0.65
MIGRATION_PENALTY_MS = 12.0
MIGRATION_CPU = 6.0
EPS = 1e-6
PLACEMENTS = frozenset({"local", "edge"})
SCENARIO_NAMES = (
    "nominal", "robot_stress", "network_fault", "edge_contention", "alternating_conditions",
)
WORKLOAD = (
    1.86, 2.48, 1.705, 2.79, 2.17, 2.635, 1.55, 2.945,
    2.015, 2.48, 1.705, 2.79, 2.325, 1.86, 2.635, 2.17,
)
SCENARIOS: dict[str, dict[str, tuple[float, ...]]] = {
    "nominal": {
        "robot_pressure": (12.0,) * WINDOWS,
        "network_delay": (2.0,) * WINDOWS,
        "edge_pressure": (14.0,) * WINDOWS,
    },
    "robot_stress": {
        "robot_pressure": (12.0, 12.0, 12.0, 12.0, 58.0, 58.0, 58.0, 58.0,
                            28.0, 28.0, 28.0, 28.0, 50.0, 50.0, 50.0, 50.0),
        "network_delay": (2.0,) * WINDOWS,
        "edge_pressure": (14.0,) * WINDOWS,
    },
    "network_fault": {
        "robot_pressure": (12.0, 12.0, 12.0, 12.0, 20.0, 20.0, 20.0, 20.0,
                            36.0, 36.0, 36.0, 36.0, 18.0, 18.0, 18.0, 18.0),
        "network_delay": (2.0, 2.0, 2.0, 2.0, 24.0, 24.0, 24.0, 24.0,
                           8.0, 8.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0),
        "edge_pressure": (14.0,) * WINDOWS,
    },
    "edge_contention": {
        "robot_pressure": (12.0, 12.0, 12.0, 12.0, 22.0, 22.0, 22.0, 22.0,
                            18.0, 18.0, 18.0, 18.0, 34.0, 34.0, 34.0, 34.0),
        "network_delay": (2.0,) * WINDOWS,
        "edge_pressure": (14.0, 14.0, 14.0, 14.0, 58.0, 58.0, 58.0, 58.0,
                           25.0, 25.0, 25.0, 25.0, 48.0, 48.0, 48.0, 48.0),
    },
    "alternating_conditions": {
        "robot_pressure": (55.0, 55.0, 16.0, 16.0) * 4,
        "network_delay": (2.0, 2.0, 17.0, 17.0) * 4,
        "edge_pressure": (14.0,) * WINDOWS,
    },
}


# Atom identities and relative shares are task-specific. Every mapping sums to one.
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 1.0 / 3.0,
        "result_json.root": 1.0 / 3.0,
        "result_json.scenarios": 1.0 / 3.0,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 1.0 / 7.0,
        "result_evidence.window_shape": 1.0 / 7.0,
        "result_evidence.window_types": 1.0 / 7.0,
        "result_evidence.cycle_shape": 1.0 / 7.0,
        "result_evidence.cycle_identity_types": 1.0 / 7.0,
        "result_evidence.cycle_numeric_values": 1.0 / 7.0,
        "result_evidence.cycle_flags": 1.0 / 7.0,
    },
    "decision_legality": {
        "decision_legality.window_index": 1.0 / 6.0,
        "decision_legality.placement_enum": 1.0 / 6.0,
        "decision_legality.cycle_index": 1.0 / 6.0,
        "decision_legality.cycle_window_relation": 1.0 / 6.0,
        "decision_legality.cycle_placement_relation": 1.0 / 6.0,
        "decision_legality.window_residency": 1.0 / 6.0,
    },
    "replay_validity": {
        "replay_validity.latency_ms": 1.0 / 5.0,
        "replay_validity.robot_cpu_pct": 1.0 / 5.0,
        "replay_validity.edge_cpu_pct": 1.0 / 5.0,
        "replay_validity.switch_penalty_ms": 1.0 / 5.0,
        "replay_validity.outcome_flags": 1.0 / 5.0,
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


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _placement_valid(value: Any) -> bool:
    return isinstance(value, str) and value in PLACEMENTS


def _close(a: Any, b: Any, tolerance: float = EPS) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tolerance


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


def _strict_json_bytes(raw: bytes) -> Any:
    """Decode result bytes as strict UTF-8 before applying the JSON policy."""
    return _strict_json(raw.decode("utf-8"))


def _scenario_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"scenario", "windows", "cycles"}


def _window_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"window", "placement"}


def _cycle_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {
        "cycle", "window", "placement", "latency_ms", "robot_cpu_pct", "edge_cpu_pct",
        "switch_penalty_ms", "deadline_miss", "resource_safe",
    }


def _window_valid(row: Any) -> bool:
    return (
        _window_shape(row)
        and _integer(row.get("window"))
        and 0 <= row["window"] < WINDOWS
        and _placement_valid(row.get("placement"))
    )


def _cycle_valid(row: Any) -> bool:
    return (
        _cycle_shape(row)
        and _integer(row.get("cycle"))
        and 0 <= row["cycle"] < CYCLES
        and _integer(row.get("window"))
        and 0 <= row["window"] < WINDOWS
        and _placement_valid(row.get("placement"))
        and all(_finite_number(row.get(key)) for key in (
            "latency_ms", "robot_cpu_pct", "edge_cpu_pct", "switch_penalty_ms",
        ))
        and all(isinstance(row.get(key), bool) for key in ("deadline_miss", "resource_safe"))
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
        window_rows = record.get("windows") if isinstance(record.get("windows"), list) else []
        cycle_rows = record.get("cycles") if isinstance(record.get("cycles"), list) else []
        windows: dict[int, Any] = {}
        cycles: dict[int, Any] = {}
        for row in window_rows:
            if _window_shape(row) and _integer(row.get("window")) and 0 <= row["window"] < WINDOWS:
                windows.setdefault(row["window"], row)
        for row in cycle_rows:
            if _cycle_shape(row) and _integer(row.get("cycle")) and 0 <= row["cycle"] < CYCLES:
                cycles.setdefault(row["cycle"], row)
        indexed[name] = {
            "record": record,
            "window_rows": window_rows,
            "cycle_rows": cycle_rows,
            "windows": windows,
            "cycles": cycles,
        }
    return indexed


def _replay_scenario(name: str, placements: Mapping[int, str]) -> dict[str, Any]:
    """Replay the fixed backlog, QoS and migration model."""
    profile = SCENARIOS[name]
    local_backlog = 0.0
    edge_backlog = 0.0
    previous = "local"
    cycles: list[dict[str, Any]] = []
    switches = 0
    for window in range(WINDOWS):
        placement = placements[window]
        changed = placement != previous
        switches += int(changed)
        if changed:
            if placement == "local":
                local_backlog += MIGRATION_WORKLOAD
            else:
                edge_backlog += MIGRATION_WORKLOAD
        for offset in range(CYCLES_PER_WINDOW):
            if placement == "local":
                local_backlog += WORKLOAD[window] / CYCLES_PER_WINDOW
            else:
                edge_backlog += WORKLOAD[window] / CYCLES_PER_WINDOW
            robot_pressure = profile["robot_pressure"][window]
            network_delay = profile["network_delay"][window]
            edge_pressure = profile["edge_pressure"][window]
            cap_local = max(0.25, 0.53 - 0.003 * robot_pressure)
            cap_edge = max(0.30, 0.72 - 0.004 * edge_pressure)
            selected_backlog = local_backlog if placement == "local" else edge_backlog
            migration_penalty = MIGRATION_PENALTY_MS if changed and offset == 0 else 0.0
            migration_overhead = MIGRATION_CPU if changed and offset == 0 else 0.0
            if placement == "local":
                latency = (
                    16.0 + 0.32 * robot_pressure + 7.0 * selected_backlog
                    + JITTER[offset] + migration_penalty
                )
            else:
                latency = (
                    20.0 + network_delay + 0.25 * edge_pressure + 7.0 * selected_backlog
                    + JITTER[offset] + migration_penalty
                )
            local_backlog = max(0.0, local_backlog - cap_local)
            edge_backlog = max(0.0, edge_backlog - cap_edge)
            if placement == "local":
                robot_cpu = 35.0 + robot_pressure + 12.0 * local_backlog + migration_overhead
                edge_cpu = 28.0 + 0.10 * edge_pressure + 6.0 * edge_backlog + migration_overhead
            else:
                robot_cpu = 25.0 + 0.12 * robot_pressure + 6.0 * local_backlog + migration_overhead
                edge_cpu = 38.0 + edge_pressure + 12.0 * edge_backlog + migration_overhead
            cycle = window * CYCLES_PER_WINDOW + offset
            cycles.append({
                "cycle": cycle,
                "window": window,
                "placement": placement,
                "latency_ms": latency,
                "robot_cpu_pct": robot_cpu,
                "edge_cpu_pct": edge_cpu,
                "switch_penalty_ms": migration_penalty,
                "deadline_miss": latency > DEADLINE_MS + EPS,
                "resource_safe": robot_cpu <= CPU_LIMIT + EPS and edge_cpu <= CPU_LIMIT + EPS,
            })
        previous = placement
    return {"cycles": cycles, "switches": switches}


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
        _atom("result_json", "parse", bool(parse_ok),
              f"results.json: expected strict JSON object; actual_type={type(data).__name__}"),
        _atom("result_json", "root", bool(root_shape),
              f"results.json: expected root keys={{'scenarios'}}; actual_keys={sorted(data) if isinstance(data, dict) else None}"),
        _atom("result_json", "scenarios",
              bool(scenarios_list),
              "results.json.scenarios: expected a top-level array; "
              f"actual_type={type(data.get('scenarios')).__name__ if isinstance(data, dict) else None}"),
    ]


def _complete_scenarios(scenarios: Mapping[str, dict[str, Any]]) -> bool:
    # Canonical completeness is required for evidence, legality and replay;
    # extra keys are owned exclusively by canonical_coverage and must not
    # invalidate otherwise complete canonical records a second time.
    return set(SCENARIO_NAMES).issubset(scenarios)


def _every_rows(
    scenarios: Mapping[str, dict[str, Any]],
    field: str,
    predicate: Any,
    expectation: str,
    *,
    skip_non_dict: bool = False,
) -> tuple[bool, str]:
    if not _complete_scenarios(scenarios):
        return False, "scenarios: all canonical scenario records are required"
    for name in SCENARIO_NAMES:
        rows = scenarios[name]["record"].get(field)
        if not isinstance(rows, list):
            return False, f"{name}.{field}: expected a list"
        for index, row in enumerate(rows):
            if skip_non_dict and not isinstance(row, dict):
                # The corresponding shape atom owns non-object rows.
                continue
            if not isinstance(row, dict) or not predicate(row):
                return False, f"{name}.{field}[{index}]: expected {expectation}; actual={row!r}"
    return True, f"{field}: all supplied records satisfy {expectation}"


def _evidence_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = _complete_scenarios(scenarios)
    scenario_ok = complete and all(_scenario_shape(scenarios[name]["record"]) for name in SCENARIO_NAMES)
    window_shape, window_shape_detail = _every_rows(
        scenarios, "windows", _window_shape, "keys {'window','placement'}")
    window_types, window_types_detail = _every_rows(
        scenarios, "windows",
        lambda row: _integer(row.get("window")) and isinstance(row.get("placement"), str),
        "integer window and string placement",
        skip_non_dict=True,
    )
    cycle_shape, cycle_shape_detail = _every_rows(
        scenarios, "cycles", _cycle_shape, "the declared cycle keys")
    cycle_identity, cycle_identity_detail = _every_rows(
        scenarios, "cycles",
        lambda row: _integer(row.get("cycle")) and _integer(row.get("window"))
        and isinstance(row.get("placement"), str),
        "integer cycle/window and string placement",
        skip_non_dict=True,
    )
    cycle_numeric, cycle_numeric_detail = _every_rows(
        scenarios, "cycles",
        lambda row: all(_finite_number(row.get(key)) for key in (
            "latency_ms", "robot_cpu_pct", "edge_cpu_pct", "switch_penalty_ms",
        )),
        "finite numeric QoS and resource fields",
        skip_non_dict=True,
    )
    cycle_flags, cycle_flags_detail = _every_rows(
        scenarios, "cycles",
        lambda row: all(isinstance(row.get(key), bool) for key in ("deadline_miss", "resource_safe")),
        "boolean deadline_miss and resource_safe",
        skip_non_dict=True,
    )
    return [
        _atom("result_evidence", "scenario_shape", scenario_ok,
              "scenario records: expected exact keys {'scenario','windows','cycles'}"),
        _atom("result_evidence", "window_shape", window_shape, window_shape_detail),
        _atom("result_evidence", "window_types", window_types, window_types_detail),
        _atom("result_evidence", "cycle_shape", cycle_shape, cycle_shape_detail),
        _atom("result_evidence", "cycle_identity_types", cycle_identity, cycle_identity_detail),
        _atom("result_evidence", "cycle_numeric_values", cycle_numeric, cycle_numeric_detail),
        _atom("result_evidence", "cycle_flags", cycle_flags, cycle_flags_detail),
    ]


def _all_rows(
    scenarios: Mapping[str, dict[str, Any]], predicate: Any, expectation: str,
) -> tuple[bool, str]:
    if not _complete_scenarios(scenarios):
        return False, "scenarios: all canonical scenario records are required"
    for name in SCENARIO_NAMES:
        for field, rows in (("windows", scenarios[name]["window_rows"]), ("cycles", scenarios[name]["cycle_rows"])):
            for index, row in enumerate(rows):
                # A malformed row belongs to the field currently being
                # checked; do not charge a malformed cycle to window-only
                # atoms (or vice versa).
                candidate = row if isinstance(row, dict) else {}
                if not predicate(field, candidate):
                    return False, f"{name}.{field}[{index}]: expected {expectation}; actual={row!r}"
    return True, f"all supplied rows satisfy {expectation}"


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    window_index, window_index_detail = _all_rows(
        scenarios,
        lambda field, row: field != "windows" or (_integer(row.get("window")) and 0 <= row["window"] < WINDOWS),
        f"window in [0,{WINDOWS})",
    )
    placement_enum, placement_enum_detail = _all_rows(
        scenarios,
        lambda field, row: field != "windows" or _placement_valid(row.get("placement")),
        "placement in {'local','edge'} for every window",
    )
    cycle_index, cycle_index_detail = _all_rows(
        scenarios,
        lambda field, row: field != "cycles" or (_integer(row.get("cycle")) and 0 <= row["cycle"] < CYCLES),
        f"cycle in [0,{CYCLES})",
    )
    cycle_window, cycle_window_detail = _all_rows(
        scenarios,
        lambda field, row: field != "cycles" or (
            not (_integer(row.get("cycle")) and 0 <= row["cycle"] < CYCLES)
            or (
                _integer(row.get("window"))
                and row["window"] == row["cycle"] // CYCLES_PER_WINDOW
            )
        ),
        "window == cycle // 4",
    )
    cycle_placement, cycle_placement_detail = _all_rows(
        scenarios,
        lambda field, row: field != "cycles" or _placement_valid(row.get("placement")),
        "a legal cycle placement",
    )
    residency_ok = _complete_scenarios(scenarios)
    residency_detail = "each cycle placement equals its window placement"
    if residency_ok:
        for name in SCENARIO_NAMES:
            windows = scenarios[name]["windows"]
            for index, row in enumerate(scenarios[name]["cycle_rows"]):
                if not isinstance(row, dict):
                    # Cycle shape/type evidence owns a non-object row; there
                    # is no cross-record residency relation to evaluate.
                    continue
                window = row.get("window")
                cycle = row.get("cycle")
                if not (
                    _integer(cycle)
                    and 0 <= cycle < CYCLES
                    and _integer(window)
                    and 0 <= window < WINDOWS
                    and window == cycle // CYCLES_PER_WINDOW
                ):
                    # The cycle-index/window relation atom owns this malformed
                    # or inconsistent identity; residency is evaluated only
                    # after that relation is established.
                    continue
                expected = windows.get(window, {}).get("placement") if isinstance(window, int) else None
                if not _placement_valid(expected):
                    # The window enum atom reports this malformed value; do not
                    # charge it again as a cross-record violation.
                    continue
                if not _placement_valid(row.get("placement")):
                    # The cycle-placement enum atom owns this malformed value;
                    # residency compares only two legal placement enums.
                    continue
                if row.get("placement") != expected:
                    residency_ok = False
                    residency_detail = (
                        f"{name}.cycles[{index}].placement: expected window placement={expected}; "
                        f"actual={row.get('placement')}"
                    )
                    break
            if not residency_ok:
                break
    else:
        residency_detail = "all canonical scenario records are required"
    # The last two atoms are deliberately separate: one checks the cycle field's enum,
    # the other checks its cross-record residency relation.
    return [
        _atom("decision_legality", "window_index", window_index, window_index_detail),
        _atom("decision_legality", "placement_enum", placement_enum, placement_enum_detail),
        _atom("decision_legality", "cycle_index", cycle_index, cycle_index_detail),
        _atom("decision_legality", "cycle_window_relation", cycle_window, cycle_window_detail),
        _atom("decision_legality", "cycle_placement_relation", cycle_placement, cycle_placement_detail),
        _atom("decision_legality", "window_residency", residency_ok, residency_detail),
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        defects.extend(["results.scenarios: missing or not a list"] * 8)
        return 0.0, defects
    records = data["scenarios"]
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    for index, row in enumerate(records):
        name = row.get("scenario") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key")
        else:
            by_name[name].append(row)

    def count_keys(record: dict[str, Any], field: str, key_name: str, expected: set[int], label: str) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {key}" for key in sorted(expected))
            return
        counts = {key: 0 for key in expected}
        for index, row in enumerate(rows):
            key = row.get(key_name) if isinstance(row, dict) and _integer(row.get(key_name)) else None
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
            defects.extend(f"scenario {name}.windows: missing key {key}" for key in range(WINDOWS))
            defects.extend(f"scenario {name}.cycles: missing key {key}" for key in range(CYCLES))
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(record, "windows", "window", set(range(WINDOWS)), f"scenario {name}.windows")
        count_keys(record, "cycles", "cycle", set(range(CYCLES)), f"scenario {name}.cycles")
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
    rule_names = [name.removeprefix("replay_validity.") for name in ATOM_SHARES["replay_validity"]]
    checks = {name: True for name in rule_names}
    details = {name: "" for name in rule_names}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    if _complete_scenarios(scenarios):
        for name in SCENARIO_NAMES:
            item = scenarios[name]
            windows = item["windows"]
            windows_complete = (
                set(windows) == set(range(WINDOWS))
                and all(_window_valid(windows[k]) for k in range(WINDOWS))
            )
            if not windows_complete:
                # Missing or illegal decisions are owned by coverage/evidence/
                # legality.  There is no fixed-model replay to compare here.
                continue
            expected = _replay_scenario(name, {k: windows[k]["placement"] for k in range(WINDOWS)})
            replayed[name] = expected
            cycles = item["cycles"]
            for index in range(CYCLES):
                actual = cycles.get(index)
                if not isinstance(actual, dict):
                    # Canonical key/shape coverage is checked elsewhere.
                    continue
                wanted = expected["cycles"][index]
                # Cycle/window/placement identity and field types are owned by
                # evidence and legality.  Replay atoms compare only finite,
                # well-typed derived values on rows with valid identity.
                if (
                    actual.get("cycle") != wanted["cycle"]
                    or actual.get("window") != wanted["window"]
                    or actual.get("placement") != wanted["placement"]
                ):
                    continue
                for field in ("latency_ms", "robot_cpu_pct", "edge_cpu_pct", "switch_penalty_ms"):
                    value = actual.get(field)
                    if not _finite_number(value):
                        continue
                    if not _close(value, wanted[field]):
                        fail(field, f"{name}.cycles[{index}].{field}: expected={wanted[field]:.9g}; actual={value}; tolerance={EPS}")
                if (
                    isinstance(actual.get("deadline_miss"), bool)
                    and isinstance(actual.get("resource_safe"), bool)
                    and (
                        actual.get("deadline_miss") is not wanted["deadline_miss"]
                        or actual.get("resource_safe") is not wanted["resource_safe"]
                    )
                ):
                    fail("outcome_flags", f"{name}.cycles[{index}]: expected flags={{deadline_miss:{wanted['deadline_miss']}, resource_safe:{wanted['resource_safe']}}}; actual={{deadline_miss:{actual.get('deadline_miss')}, resource_safe:{actual.get('resource_safe')}}}")
    atoms = [
        _atom("replay_validity", name, checks[name], details[name] or "fixed backlog and QoS replay comparison passed")
        for name in rule_names
    ]
    return atoms, replayed


def _replay_diagnostic(
    scenarios: Mapping[str, dict[str, Any]], atoms: Sequence[AtomVerdict],
) -> str:
    """Report when replay deliberately defers a malformed record to its owner."""
    notes: list[str] = []
    if not _complete_scenarios(scenarios):
        notes.append("missing_canonical_scenario")
    else:
        for name in SCENARIO_NAMES:
            item = scenarios[name]
            windows = item["windows"]
            if (
                set(windows) != set(range(WINDOWS))
                or any(not _window_valid(windows[k]) for k in range(WINDOWS))
            ):
                notes.append(f"{name}.windows_owned_by_coverage_or_legality")
                continue
            expected = _replay_scenario(name, {k: windows[k]["placement"] for k in range(WINDOWS)})
            cycles = item["cycles"]
            for index in range(CYCLES):
                actual = cycles.get(index)
                wanted = expected["cycles"][index]
                if not isinstance(actual, dict):
                    notes.append(f"{name}.cycles[{index}]_owned_by_shape_or_coverage")
                    continue
                if (
                    actual.get("cycle") != wanted["cycle"]
                    or actual.get("window") != wanted["window"]
                    or actual.get("placement") != wanted["placement"]
                ):
                    notes.append(f"{name}.cycles[{index}]_identity_owned_by_evidence_or_legality")
                    continue
                if any(not _finite_number(actual.get(field)) for field in (
                    "latency_ms", "robot_cpu_pct", "edge_cpu_pct", "switch_penalty_ms",
                )):
                    notes.append(f"{name}.cycles[{index}]_numeric_type_owned_by_evidence")
                if not all(isinstance(actual.get(field), bool) for field in ("deadline_miss", "resource_safe")):
                    notes.append(f"{name}.cycles[{index}]_flag_type_owned_by_evidence")
    if notes:
        unique = list(dict.fromkeys(notes))
        return f"{_diagnose_atoms(atoms)}; not_evaluated_due_to=[{','.join(unique[:24])}]"
    return _diagnose_atoms(atoms)


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    misses = 0
    safe = 0
    switches = 0
    latency_quality = 0.0
    total = float(len(SCENARIO_NAMES) * CYCLES)
    for name in SCENARIO_NAMES:
        result = replayed[name]
        switches += int(result["switches"])
        for row in result["cycles"]:
            misses += int(row["deadline_miss"])
            safe += int(row["resource_safe"])
            latency_quality += max(0.0, min(1.0, (LATENCY_BOUND_MS - row["latency_ms"]) / LATENCY_SCALE_MS))
    metrics = {
        "deadline_compliance": 1.0 - misses / total,
        "latency_quality": latency_quality / total,
        "resource_safety": safe / total,
        "placement_stability": max(0.0, min(1.0, 1.0 - switches / 80.0)),
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
        "Problem Formulation", "Method Design", "Experimental Setup", "Results",
        "Research Analysis", "Conclusion and Limitations",
    )
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    return score, (f"missing_sections={missing}; normalized={score:.9g}" if missing else "missing_sections=[]")


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
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
                if state.official_ok
                else f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                     f"timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
            )
        if state.result_file_is_new_regular:
            try:
                result_path = Path(ws.root) / "results.json"
                state.data = _strict_json_bytes(result_path.read_bytes())
                state.parse_ok = True
            except (OSError, UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.parse_ok = False
                state.diagnostics["result_json"] = f"results.json: strict UTF-8/JSON parse failed: {exc}"

        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose_atoms(result_atoms))
        if not state.official_ok or not state.parse_ok:
            blocked_checks = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
            reason = (
                f"blocked_checks=[{','.join(blocked_checks)}]; official_ok={state.official_ok}; "
                f"parse_ok={state.parse_ok}; official_message={state.official_message}"
            )
            state.scores["result_json"] = 0.0
            state.diagnostics["result_json"] = f"{reason}; {state.diagnostics['result_json']}"
            for name in blocked_checks[1:]:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"{reason}; {name}=not_evaluated"
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
        state.diagnostics["replay_validity"] = _replay_diagnostic(state.scenarios, replay_atoms)

        gate_names = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        state.validity_gate = state.official_ok and all(
            state.scores.get(name, 0.0) >= 1.0 - EPS for name in gate_names
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001 - cache a deterministic failed state
        state.official_ok = False
        state.official_message = state.official_message or f"grader state construction failed: {type(exc).__name__}: {exc}"
        state.validity_gate = False
        state.metrics = None
        failed_checks = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        detail = (
            f"grader_exception={type(exc).__name__}: {exc}; "
            f"affected_checks=[{','.join(failed_checks)}]; validity_gate=False"
        )
        for name in failed_checks:
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
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
    report_path = Path(ws.root) / "report.md"
    if not report_path.is_file() or report_path.is_symlink():
        return 0.0, "report.md: expected a regular UTF-8 file"
    try:
        text = report_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return 0.0, f"report.md: strict UTF-8 read failed: {exc}"
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
        gate_names = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        failed_gate = [gate for gate in gate_names if state.scores.get(gate, 0.0) < 1.0 - EPS]
        return 0.0, (
            f"{name}: validity gate not satisfied; metric not computed; "
            f"failed_gate=[{','.join(failed_gate)}]; gate_scores="
            + ",".join(f"{gate}={state.scores.get(gate, 0.0):.9g}" for gate in gate_names)
        )
    value = state.metrics.get(name, 0.0)
    return value, f"{name}: independently replayed value={value:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "deadline_compliance")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "latency_quality")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "resource_safety")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "placement_stability")
