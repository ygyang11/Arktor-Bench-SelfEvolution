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


HORIZON = 40
N_ROBOTS = 4
N_TASKS = 12
GRID = 12
RADIUS = 5
MAX_BROADCASTS = 18
WINDOW = 5
WINDOW_LIMIT = 2
MAX_ENTRIES = 4
EPS = 1e-6
SCENARIO_NAMES = ("dense_arrival", "capability_bottleneck", "relay_gap", "late_urgent", "balanced")
CAPABILITIES = ("red", "green", "blue", "blue")
STARTS = ((0, 0), (11, 0), (0, 11), (11, 11))
ACTIONS = ("move", "inspect", "claim", "execute", "abandon", "wait")
MOVES = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0), "stay": (0, 0)}
REQUIREMENTS = {"red", "green", "blue"}
ACTION_STATUSES = {"none", "moved", "inspected", "claimed", "executed", "failed", "abandoned", "invalid"}
TASK_STATES = {"unreleased", "available", "claimed", "completed", "expired"}


def _make_tasks(rows: Sequence[tuple[tuple[int, int], int, int, str]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"task": i, "position": tuple(pos), "release_t": release, "deadline_t": deadline, "requirement": req}
        for i, (pos, release, deadline, req) in enumerate(rows)
    )


TASKS: dict[str, tuple[dict[str, Any], ...]] = {
    "dense_arrival": _make_tasks((
        ((3, 3), 0, 16, "red"), ((4, 3), 1, 18, "green"), ((3, 4), 2, 20, "blue"),
        ((4, 4), 3, 22, "blue"), ((5, 4), 4, 24, "red"), ((6, 4), 5, 26, "green"),
        ((5, 5), 6, 27, "blue"), ((6, 5), 8, 29, "red"), ((5, 6), 10, 31, "green"),
        ((6, 6), 12, 33, "blue"), ((4, 6), 14, 35, "red"), ((7, 6), 16, 37, "blue"),
    )),
    "capability_bottleneck": _make_tasks((
        ((2, 9), 0, 18, "red"), ((3, 8), 0, 20, "red"), ((4, 7), 1, 22, "red"),
        ((8, 9), 1, 19, "green"), ((9, 8), 2, 23, "green"), ((8, 3), 3, 20, "blue"),
        ((9, 4), 4, 24, "blue"), ((6, 6), 4, 25, "red"), ((5, 5), 6, 27, "red"),
        ((6, 3), 8, 29, "green"), ((3, 2), 10, 31, "blue"), ((7, 1), 12, 34, "red"),
    )),
    "relay_gap": _make_tasks((
        ((1, 5), 0, 20, "red"), ((2, 6), 1, 22, "green"), ((1, 7), 2, 24, "blue"),
        ((2, 8), 3, 25, "blue"), ((10, 5), 0, 20, "red"), ((9, 6), 1, 22, "green"),
        ((10, 7), 2, 24, "blue"), ((9, 8), 3, 25, "blue"), ((5, 1), 4, 27, "red"),
        ((6, 2), 6, 29, "green"), ((5, 10), 8, 31, "blue"), ((6, 9), 10, 33, "red"),
    )),
    "late_urgent": _make_tasks((
        ((2, 2), 10, 18, "red"), ((9, 2), 10, 19, "green"), ((2, 9), 11, 21, "blue"),
        ((9, 9), 11, 22, "blue"), ((4, 2), 12, 22, "red"), ((7, 2), 13, 23, "green"),
        ((4, 9), 14, 25, "blue"), ((7, 9), 14, 26, "red"), ((5, 5), 16, 28, "green"),
        ((6, 5), 17, 29, "blue"), ((5, 6), 18, 30, "red"), ((6, 6), 20, 32, "blue"),
    )),
    "balanced": _make_tasks((
        ((2, 2), 0, 20, "red"), ((9, 2), 0, 21, "green"), ((2, 9), 1, 22, "blue"),
        ((9, 9), 1, 23, "blue"), ((4, 4), 3, 24, "red"), ((7, 4), 4, 25, "green"),
        ((4, 7), 5, 26, "blue"), ((7, 7), 6, 27, "red"), ((5, 3), 8, 29, "green"),
        ((6, 3), 9, 30, "blue"), ((5, 8), 10, 32, "red"), ((6, 8), 11, 34, "blue"),
    )),
}
BLACKOUTS = {
    "dense_arrival": frozenset({7, 8}),
    "capability_bottleneck": frozenset({12, 13, 14}),
    "relay_gap": frozenset({4, 5, 6, 20, 21}),
    "late_urgent": frozenset({0, 1, 2, 15, 16}),
    "balanced": frozenset({10, 11, 24}),
}
FAILURE_SLOTS = {
    "dense_arrival": frozenset({9, 21}),
    "capability_bottleneck": frozenset({8, 17, 28}),
    "relay_gap": frozenset({11, 22}),
    "late_urgent": frozenset({18, 27}),
    "balanced": frozenset({13, 25}),
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {"result_json.parse": 0.34, "result_json.root": 0.33, "result_json.scenarios": 0.33},
    "result_evidence": {f"result_evidence.{name}": 0.1 for name in (
        "scenario_shape", "robot_shape", "robot_fields", "task_shape", "task_fields",
        "message_shape", "message_fields", "fact_fields", "timeline_fields", "array_values",
    )},
    "decision_legality": {f"decision_legality.{name}": 0.125 for name in (
        "robot_keys", "action_domain", "position_move", "information_boundary", "task_pair_limits",
        "local_preconditions", "broadcast_resources", "message_summary",
    )},
    "replay_validity": {f"replay_validity.{name}": 1.0 / 9.0 for name in (
        "positions_moves", "arrivals_knowledge", "inspections", "claims", "executions",
        "task_timelines", "messages", "action_results", "task_records",
    )},
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
FACT_KEYS = {"task", "requirement", "claim_robot", "version"}
ROBOT_KEYS = {"robot", "t", "position", "capability", "action", "move", "task", "known_facts", "received_message_ids", "broadcast", "broadcast_entries", "inspection_result", "action_result"}
TASK_KEYS = {"task", "position", "release_t", "deadline_t", "requirement", "timeline", "claimed_by", "inspect_t", "complete_t", "completion_reason"}
TIMELINE_KEYS = {"t", "state", "claimed_by"}
MESSAGE_KEYS = {"message_id", "sender", "send_t", "targets", "delivered_to", "entries", "arrival_t", "status"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fact(task: int, requirement: str, claim_robot: int | None, version: int) -> dict[str, Any]:
    return {"task": int(task), "requirement": requirement, "claim_robot": claim_robot, "version": int(version)}


def _fact_key(value: Mapping[str, Any]) -> tuple[int, int, int]:
    claim = -1 if value.get("claim_robot") is None else int(value["claim_robot"])
    return int(value["task"]), int(value["version"]), claim


def _facts_list(facts: Mapping[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(facts[t]) for t in sorted(facts)]


def _task_truth(scenario: str, task: int) -> dict[str, Any]:
    return TASKS[scenario][task]


def _failure(scenario: str, t: int, robot: int, task: int) -> bool:
    return t in FAILURE_SLOTS[scenario] and (t + robot + task) % 4 == 0


def _claim_winner(
    task_state: str,
    owner: int | None,
    task: int,
    claimants: Sequence[int],
    facts: Sequence[Mapping[int, Mapping[str, Any]]],
) -> int | None:
    if task_state != "available" or owner is not None:
        return None
    eligible = [
        robot for robot in sorted(claimants)
        if facts[robot].get(task, {}).get("requirement") == CAPABILITIES[robot]
    ]
    return eligible[0] if eligible else None


def _register_pair_attempt(attempts: set[tuple[int, int]], robot: int, task: int) -> bool:
    pair = (robot, task)
    if pair in attempts:
        return False
    attempts.add(pair)
    return True


def _owns_active_claim(
    task: Any,
    robot: int,
    task_states: Mapping[int, str],
    owners: Mapping[int, int | None],
) -> bool:
    return _integer(task) and 0 <= task < N_TASKS and owners.get(task) == robot and task_states.get(task) == "claimed"


def _resolve_movements(
    start_positions: Sequence[tuple[int, int]],
    proposed: Sequence[tuple[int, int]],
) -> tuple[set[int], list[tuple[int, int]]]:
    conflicts: set[int] = set()
    for i in range(N_ROBOTS):
        for j in range(i + 1, N_ROBOTS):
            same_destination = proposed[i] == proposed[j]
            exchanged = proposed[i] == start_positions[j] and proposed[j] == start_positions[i]
            if same_destination or exchanged:
                conflicts.update((i, j))
    actual = [start_positions[robot] if robot in conflicts else proposed[robot] for robot in range(N_ROBOTS)]
    return conflicts, actual


def _targets(scenario: str, positions: Sequence[tuple[int, int]], sender: int, t: int) -> list[int]:
    if t in BLACKOUTS[scenario]:
        return []
    sx, sy = positions[sender]
    return [
        r for r, (x, y) in enumerate(positions)
        if r != sender and abs(sx - x) + abs(sy - y) <= RADIUS
    ]


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


def _pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_integer(v) and 0 <= v < GRID for v in value)


def _fact_valid(value: Any) -> bool:
    return (
        isinstance(value, dict) and set(value) == FACT_KEYS and _integer(value.get("task")) and 0 <= value["task"] < N_TASKS
        and value.get("requirement") in REQUIREMENTS
        and (value.get("claim_robot") is None or (_integer(value.get("claim_robot")) and 0 <= value["claim_robot"] < N_ROBOTS))
        and _integer(value.get("version")) and 0 <= value["version"] < HORIZON
    )


def _facts_distinct(values: Any) -> bool:
    """Fact arrays represent a map, so each task may occur at most once."""
    return isinstance(values, list) and len({value.get("task") for value in values if isinstance(value, dict)}) == len(values)


def _robot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == ROBOT_KEYS


def _task_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == TASK_KEYS


def _message_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == MESSAGE_KEYS


def _robot_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    inspection = value.get("inspection_result")
    action_result = value.get("action_result")
    inspection_ok = inspection is None or (
        isinstance(inspection, dict) and set(inspection) == {"task", "requirement", "version"}
        and _integer(inspection.get("task")) and 0 <= inspection["task"] < N_TASKS
        and inspection.get("requirement") in REQUIREMENTS and _integer(inspection.get("version"))
        and 0 <= inspection["version"] < HORIZON
    )
    result_ok = isinstance(action_result, dict) and set(action_result) == {"status", "task"}
    result_ok = result_ok and action_result.get("status") in ACTION_STATUSES and (
        action_result.get("task") is None or (_integer(action_result.get("task")) and 0 <= action_result["task"] < N_TASKS)
    )
    return (
        _integer(value.get("robot")) and 0 <= value["robot"] < N_ROBOTS
        and _integer(value.get("t")) and 0 <= value["t"] < HORIZON and _pair(value.get("position"))
        and value.get("capability") in REQUIREMENTS and value.get("action") in ACTIONS and value.get("move") in MOVES
        and (value.get("task") is None or (_integer(value.get("task")) and 0 <= value["task"] < N_TASKS))
        and isinstance(value.get("broadcast"), bool) and inspection_ok and result_ok
    )


def _timeline_valid(value: Any) -> bool:
    return (
        isinstance(value, dict) and set(value) == TIMELINE_KEYS and _integer(value.get("t")) and 0 <= value["t"] < HORIZON
        and value.get("state") in TASK_STATES
        and (value.get("claimed_by") is None or (_integer(value.get("claimed_by")) and 0 <= value["claimed_by"] < N_ROBOTS))
    )


def _task_valid(value: Any) -> bool:
    return (
        isinstance(value, dict) and _integer(value.get("task")) and 0 <= value["task"] < N_TASKS and _pair(value.get("position"))
        and _integer(value.get("release_t")) and 0 <= value["release_t"] < HORIZON
        and _integer(value.get("deadline_t")) and value["deadline_t"] >= value["release_t"] and value["deadline_t"] < HORIZON
        and value.get("requirement") in REQUIREMENTS
        and (value.get("claimed_by") is None or (_integer(value.get("claimed_by")) and 0 <= value["claimed_by"] < N_ROBOTS))
        and (value.get("inspect_t") is None or (_integer(value.get("inspect_t")) and 0 <= value["inspect_t"] < HORIZON))
        and (value.get("complete_t") is None or (_integer(value.get("complete_t")) and 0 <= value["complete_t"] < HORIZON))
        and value.get("completion_reason") in {"completed", "expired"}
    )


def _message_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("message_id"), str) and _integer(value.get("sender")) and 0 <= value["sender"] < N_ROBOTS
        and _integer(value.get("send_t")) and 0 <= value["send_t"] < HORIZON
        and _integer(value.get("arrival_t")) and 1 <= value["arrival_t"] <= HORIZON
        and value.get("status") in {"delivered", "lost"}
    )


def _robot_fact_fields_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    known = value.get("known_facts")
    entries = value.get("broadcast_entries")
    return (
        isinstance(known, list)
        and _facts_distinct(known)
        and all(_fact_valid(fact) for fact in known)
        and isinstance(entries, list)
        and all(_fact_valid(fact) for fact in entries)
    )


def _message_fact_fields_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        return False
    return all(_fact_valid(fact) for fact in value["entries"])


def _task_timeline_fields_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("timeline"), list):
        return False
    return all(_timeline_valid(entry) for entry in value["timeline"])


def _array_fields_valid(scenario: Any) -> bool:
    if not isinstance(scenario, dict):
        return False
    robots = scenario.get("robots")
    tasks = scenario.get("tasks")
    messages = scenario.get("messages")
    if not isinstance(robots, list) or not isinstance(tasks, list) or not isinstance(messages, list):
        return False
    robot_arrays = all(
        isinstance(row, dict)
        and isinstance(row.get("received_message_ids"), list)
        and all(isinstance(message_id, str) for message_id in row["received_message_ids"])
        for row in robots
    )
    message_arrays = all(
        isinstance(row, dict)
        and isinstance(row.get("targets"), list)
        and isinstance(row.get("delivered_to"), list)
        and all(
            _integer(robot) and 0 <= robot < N_ROBOTS
            for robot in row["targets"] + row["delivered_to"]
        )
        for row in messages
    )
    return robot_arrays and message_arrays


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"scenario", "robots", "tasks", "messages"}


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("scenario"), str):
            continue
        name = scenario["scenario"]
        if name in indexed:
            continue
        robots = scenario.get("robots") if isinstance(scenario.get("robots"), list) else []
        tasks = scenario.get("tasks") if isinstance(scenario.get("tasks"), list) else []
        messages = scenario.get("messages") if isinstance(scenario.get("messages"), list) else []
        robot_map: dict[tuple[int, int], Any] = {}
        task_map: dict[int, Any] = {}
        message_map: dict[str, Any] = {}
        for row in robots:
            if isinstance(row, dict) and _integer(row.get("robot")) and _integer(row.get("t")):
                robot_map.setdefault((row["robot"], row["t"]), row)
        for row in tasks:
            if isinstance(row, dict) and _integer(row.get("task")):
                task_map.setdefault(row["task"], row)
        for row in messages:
            if isinstance(row, dict) and isinstance(row.get("message_id"), str):
                message_map.setdefault(row["message_id"], row)
        indexed[name] = {"record": scenario, "robot_rows": robots, "task_rows": tasks, "message_rows": messages,
                         "robots": robot_map, "tasks": task_map, "messages": message_map}
    return indexed


def _atom(check_id: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{name}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share differs")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[{}]; {}".format(",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed))


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root = parse_ok and isinstance(data, dict)
    keys_ok = bool(root and set(data) == {"scenarios"})
    scenarios_ok = bool(keys_ok and isinstance(data.get("scenarios"), list))
    return [
        _atom("result_json", "parse", parse_ok, "results.json: strict JSON parse"),
        _atom("result_json", "root", keys_ok, "results.json: exact root key set"),
        _atom("result_json", "scenarios", scenarios_ok, "results.json.scenarios: array required"),
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not an array"]
    by_name: dict[str, list[Any]] = {name: [] for name in SCENARIO_NAMES}
    for index, row in enumerate(data["scenarios"]):
        name = row.get("scenario") if isinstance(row, dict) else None
        if name not in by_name:
            defects.append(f"scenarios[{index}]: unknown scenario")
        else:
            by_name[name].append(row)
    supplied = [row.get("scenario") for row in data["scenarios"] if isinstance(row, dict)]
    if supplied != list(SCENARIO_NAMES):
        defects.append(f"scenarios: expected canonical order {list(SCENARIO_NAMES)}, actual={supplied}")
    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"{name}: missing scenario")
            continue
        defects.extend(f"{name}: duplicate scenario" for _ in range(len(matches) - 1))
        row = matches[0]
        expected_robot = {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}
        counts = {key: 0 for key in expected_robot}
        robots = row.get("robots") if isinstance(row, dict) else None
        if not isinstance(robots, list):
            defects.append(f"{name}.robots: missing array")
        else:
            for index, item in enumerate(robots):
                key = (item.get("robot"), item.get("t")) if isinstance(item, dict) else None
                if key in counts:
                    counts[key] += 1
                else:
                    defects.append(f"{name}.robots[{index}]: unknown key")
            defects.extend(f"{name}.robots: missing key {key}" for key, count in counts.items() if count == 0)
            defects.extend(f"{name}.robots: duplicate key {key}" for key, count in counts.items() for _ in range(max(0, count - 1)))
        task_counts = {task: 0 for task in range(N_TASKS)}
        tasks = row.get("tasks") if isinstance(row, dict) else None
        if not isinstance(tasks, list):
            defects.append(f"{name}.tasks: missing array")
        else:
            for index, item in enumerate(tasks):
                key = item.get("task") if isinstance(item, dict) else None
                if key in task_counts:
                    task_counts[key] += 1
                else:
                    defects.append(f"{name}.tasks[{index}]: unknown task key")
            defects.extend(f"{name}.tasks: missing key {key}" for key, count in task_counts.items() if count == 0)
            defects.extend(f"{name}.tasks: duplicate key {key}" for key, count in task_counts.items() for _ in range(max(0, count - 1)))
        if isinstance(tasks, list):
            for index, item in enumerate(tasks):
                timeline = item.get("timeline") if isinstance(item, dict) else None
                counts_t = {t: 0 for t in range(HORIZON)}
                if not isinstance(timeline, list):
                    defects.append(f"{name}.tasks[{index}].timeline: missing array")
                    continue
                for entry in timeline:
                    key = entry.get("t") if isinstance(entry, dict) else None
                    if key in counts_t:
                        counts_t[key] += 1
                    else:
                        defects.append(f"{name}.tasks[{index}].timeline: unknown t={key}")
                defects.extend(f"{name}.tasks[{index}].timeline: missing t={t}" for t, count in counts_t.items() if count == 0)
                defects.extend(f"{name}.tasks[{index}].timeline: duplicate t={t}" for t, count in counts_t.items() for _ in range(max(0, count - 1)))
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _evidence_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = set(indexed) == set(SCENARIO_NAMES)
    scenario_shape = complete and all(_scenario_shape(indexed[n]["record"]) for n in SCENARIO_NAMES)
    robot_shape = complete and all(all(_robot_shape(row) for row in indexed[n]["robot_rows"]) for n in SCENARIO_NAMES)
    robot_fields = complete and all(all(_robot_valid(row) for row in indexed[n]["robot_rows"]) for n in SCENARIO_NAMES)
    task_shape = complete and all(all(_task_shape(row) for row in indexed[n]["task_rows"]) for n in SCENARIO_NAMES)
    task_fields = complete and all(all(_task_valid(row) for row in indexed[n]["task_rows"]) for n in SCENARIO_NAMES)
    message_shape = complete and all(all(_message_shape(row) for row in indexed[n]["message_rows"]) for n in SCENARIO_NAMES)
    message_fields = complete and all(all(_message_valid(row) for row in indexed[n]["message_rows"]) for n in SCENARIO_NAMES)
    fact_fields = complete and all(
        all(_robot_fact_fields_valid(row) for row in indexed[n]["robot_rows"])
        and all(_message_fact_fields_valid(row) for row in indexed[n]["message_rows"])
        for n in SCENARIO_NAMES
    )
    timeline_fields = complete and all(
        all(_task_timeline_fields_valid(row) for row in indexed[n]["task_rows"])
        for n in SCENARIO_NAMES
    )
    array_values = complete and all(_array_fields_valid(indexed[n]["record"]) for n in SCENARIO_NAMES)
    return [
        _atom("result_evidence", "scenario_shape", scenario_shape, "scenario objects have exact keys"),
        _atom("result_evidence", "robot_shape", robot_shape, "robot records have exact keys"),
        _atom("result_evidence", "robot_fields", robot_fields, "robot fields have declared domains"),
        _atom("result_evidence", "task_shape", task_shape, "task records have exact keys"),
        _atom("result_evidence", "task_fields", task_fields, "task fields have declared domains"),
        _atom("result_evidence", "message_shape", message_shape, "message records have exact keys"),
        _atom("result_evidence", "message_fields", message_fields, "message fields have declared domains"),
        _atom("result_evidence", "fact_fields", fact_fields, "fact arrays have declared fields"),
        _atom("result_evidence", "timeline_fields", timeline_fields, "task timelines have declared fields"),
        _atom("result_evidence", "array_values", array_values, "arrival and variable arrays have valid containers"),
    ]


def _legality_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = set(indexed) == set(SCENARIO_NAMES)
    robot_keys = complete and all(len(indexed[n]["robot_rows"]) == N_ROBOTS * HORIZON and set(indexed[n]["robots"]) == {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)} for n in SCENARIO_NAMES)
    action_domain = complete and all(
        all(
            isinstance(row, dict) and row.get("action") in ACTIONS and row.get("move") in MOVES
            and _integer(row.get("robot")) and 0 <= row["robot"] < N_ROBOTS
            and row.get("capability") == CAPABILITIES[row["robot"]]
            for row in indexed[n]["robot_rows"]
        )
        for n in SCENARIO_NAMES
    )
    position_move = complete and all(
        all(
            _pair(row.get("position"))
            and (row.get("action") == "move" or row.get("move") == "stay")
            and (
                (row.get("action") in {"inspect", "claim", "execute", "abandon"} and _integer(row.get("task")))
                or (row.get("action") in {"move", "wait"} and row.get("task") is None)
            )
            for row in indexed[n]["robot_rows"] if isinstance(row, dict)
        )
        for n in SCENARIO_NAMES
    )
    information_boundary = complete
    task_pair_limits = complete
    local_preconditions = complete
    broadcast_resources = complete
    message_summary = complete
    for name in SCENARIO_NAMES:
        if name not in indexed:
            continue
        rows = indexed[name]["robots"]
        for robot in range(N_ROBOTS):
            inspect_pairs: set[int] = set()
            claim_pairs: set[int] = set()
            broadcasts: list[int] = []
            for t in range(HORIZON):
                row = rows.get((robot, t), {})
                action = row.get("action")
                task = row.get("task")
                position = row.get("position")
                move = row.get("move")
                if action in {"inspect", "claim"} and not _integer(task):
                    task_pair_limits = False
                if action == "inspect" and _integer(task):
                    if task in inspect_pairs:
                        task_pair_limits = False
                    inspect_pairs.add(task)
                if action == "claim" and _integer(task):
                    if task in claim_pairs:
                        task_pair_limits = False
                    claim_pairs.add(task)
                known_values = row.get("known_facts", [])
                known_iter = known_values if isinstance(known_values, list) else []
                known = {
                    fact.get("task"): fact
                    for fact in known_iter if isinstance(fact, dict)
                }
                if action == "move" and move in MOVES and _pair(position):
                    dx, dy = MOVES[move]
                    candidate = (position[0] + dx, position[1] + dy)
                    if not (0 <= candidate[0] < GRID and 0 <= candidate[1] < GRID):
                        local_preconditions = False
                elif action in {"inspect", "claim", "execute", "abandon"}:
                    if not (_integer(task) and 0 <= task < N_TASKS and _pair(position)):
                        local_preconditions = False
                    else:
                        truth = _task_truth(name, task)
                        in_public_window = truth["release_t"] <= t <= truth["deadline_t"]
                        fact = known.get(task)
                        if action == "inspect" and (not in_public_window or tuple(position) != truth["position"]):
                            local_preconditions = False
                        elif action == "claim" and (
                            not in_public_window
                            or fact is None
                            or fact.get("requirement") != CAPABILITIES[robot]
                        ):
                            local_preconditions = False
                        elif action == "execute" and (
                            not in_public_window
                            or fact is None
                            or fact.get("claim_robot") != robot
                            or tuple(position) != truth["position"]
                        ):
                            local_preconditions = False
                        elif action == "abandon" and (
                            not in_public_window or fact is None or fact.get("claim_robot") != robot
                        ):
                            local_preconditions = False
                entries_value = row.get("broadcast_entries", [])
                entries = entries_value if isinstance(entries_value, list) else []
                if not isinstance(entries_value, list):
                    information_boundary = False
                    message_summary = False
                if row.get("broadcast"):
                    broadcasts.append(t)
                    if row.get("action") == "move" and row.get("move") not in MOVES:
                        broadcast_resources = False
                    if any(f not in known.values() for f in entries):
                        information_boundary = False
                    if len(entries) > MAX_ENTRIES or not _facts_distinct(entries):
                        message_summary = False
                elif entries:
                    information_boundary = False
            if len(broadcasts) > MAX_BROADCASTS or any(broadcasts[i + 1] == broadcasts[i] + 1 for i in range(len(broadcasts) - 1)):
                broadcast_resources = False
            for start in range(HORIZON - WINDOW + 1):
                if sum(t in broadcasts for t in range(start, start + WINDOW)) > WINDOW_LIMIT:
                    broadcast_resources = False
    return [
        _atom("decision_legality", "robot_keys", robot_keys, "exact robot-slot keys are required"),
        _atom("decision_legality", "action_domain", action_domain, "actions and capabilities use declared domains"),
        _atom("decision_legality", "position_move", position_move, "positions, movement, and action-task relations obey the declared rules"),
        _atom("decision_legality", "information_boundary", information_boundary, "summaries are selected from known facts"),
        _atom("decision_legality", "task_pair_limits", task_pair_limits, "inspection and claim pairs are not repeated"),
        _atom("decision_legality", "local_preconditions", local_preconditions, "locally knowable movement and task-action preconditions are satisfied"),
        _atom("decision_legality", "broadcast_resources", broadcast_resources, "broadcast budgets and windows are respected"),
        _atom("decision_legality", "message_summary", message_summary, "broadcast summaries respect entry cardinality and uniqueness"),
    ]


def _status_result(status: str, task: int | None) -> dict[str, Any]:
    return {"status": status, "task": task}


def _expected_timeline(states: Mapping[int, str], owners: Mapping[int, int | None], t: int) -> dict[str, Any]:
    return {"t": t, "state": states[t], "claimed_by": owners[t]}


def _replay_scenario(name: str, item: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, bool], dict[str, str]]:
    rules = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    checks = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    robots = item["robots"]
    if set(robots) != {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}:
        for rule in rules:
            fail(rule, f"{name}.robots: complete robot-slot records required")
        return None, checks, details
    positions = list(STARTS)
    facts: list[dict[int, dict[str, Any]]] = [dict() for _ in range(N_ROBOTS)]
    pending: dict[int, list[dict[str, Any]]] = {}
    expected_messages: dict[str, dict[str, Any]] = {}
    task_states = {task: "unreleased" for task in range(N_TASKS)}
    owners: dict[int, int | None] = {task: None for task in range(N_TASKS)}
    inspect_attempts: set[tuple[int, int]] = set()
    claim_attempts: set[tuple[int, int]] = set()
    inspect_times: dict[int, int | None] = {task: None for task in range(N_TASKS)}
    complete_times: dict[int, int | None] = {task: None for task in range(N_TASKS)}
    reasons: dict[int, str] = {task: "expired" for task in range(N_TASKS)}
    timelines: dict[int, list[dict[str, Any]]] = {task: [] for task in range(N_TASKS)}
    received_by_robot: list[list[str]] = [[] for _ in range(N_ROBOTS)]
    broadcast_count = 0
    avoidable_inspect = 0
    conflicting_claim = 0
    completed = 0
    utility_sum = 0.0

    for t in range(HORIZON):
        # Start-of-slot order: expiry, message delivery, then releases.
        for task in range(N_TASKS):
            truth = _task_truth(name, task)
            if task_states[task] in {"available", "claimed"} and t > truth["deadline_t"]:
                task_states[task] = "expired"
                owners[task] = None
        received = [[] for _ in range(N_ROBOTS)]
        for event in sorted(pending.pop(t, []), key=lambda value: (value["sender"], value["message_id"])):
            for receiver in event["targets"]:
                received[receiver].append(event["message_id"])
                for entry in event["entries"]:
                    old = facts[receiver].get(entry["task"])
                    if old is None or entry["version"] > old["version"]:
                        facts[receiver][entry["task"]] = dict(entry)
        for receiver in range(N_ROBOTS):
            received[receiver] = sorted(received[receiver])
        for task in range(N_TASKS):
            truth = _task_truth(name, task)
            if task_states[task] == "unreleased" and t >= truth["release_t"]:
                task_states[task] = "available"

        pre_facts = [_facts_list(local) for local in facts]
        start_positions = list(positions)
        planned: list[tuple[int, int]] = []
        row_results = [_status_result("none", None) for _ in range(N_ROBOTS)]
        row_inspections: list[dict[str, Any] | None] = [None] * N_ROBOTS
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            if row.get("position") != list(start_positions[robot]):
                fail("positions_moves", f"{name}.robots[{robot},{t}].position: expected {list(start_positions[robot])}, actual={row.get('position')}")
            if row.get("received_message_ids") != received[robot]:
                fail("arrivals_knowledge", f"{name}.robots[{robot},{t}].received_message_ids: expected {received[robot]}, actual={row.get('received_message_ids')}")
            if row.get("known_facts") != pre_facts[robot]:
                fail("arrivals_knowledge", f"{name}.robots[{robot},{t}].known_facts: expected {pre_facts[robot]}, actual={row.get('known_facts')}")
            action = row.get("action")
            move = row.get("move")
            proposed = start_positions[robot]
            if action == "move" and move in MOVES:
                dx, dy = MOVES[move]
                candidate = (start_positions[robot][0] + dx, start_positions[robot][1] + dy)
                if 0 <= candidate[0] < GRID and 0 <= candidate[1] < GRID:
                    proposed = candidate
                else:
                    row_results[robot] = _status_result("invalid", row.get("task"))
            elif action != "move" and move != "stay":
                fail("positions_moves", f"{name}.robots[{robot},{t}]: non-move action must stay")
                row_results[robot] = _status_result("invalid", row.get("task"))
            planned.append(proposed)
            if row.get("broadcast"):
                broadcast_count += 1

        conflicts, actual_positions = _resolve_movements(start_positions, planned)
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            if row.get("action") == "move":
                if robot in conflicts or planned[robot] == start_positions[robot]:
                    if row.get("move") != "stay" and row_results[robot]["status"] != "invalid":
                        row_results[robot] = _status_result("invalid", row.get("task"))
                else:
                    row_results[robot] = _status_result("moved", None)

        # Inspections happen before claims, and reveal only the inspecting robot's task fact.
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            task = row.get("task")
            if row.get("action") != "inspect":
                continue
            valid = _integer(task) and 0 <= task < N_TASKS and _register_pair_attempt(inspect_attempts, robot, task)
            if valid and task_states[task] in {"available", "claimed"} and tuple(start_positions[robot]) == tuple(_task_truth(name, task)["position"]):
                truth = _task_truth(name, task)
                inspect_times[task] = inspect_times[task] if inspect_times[task] is not None else t
                facts[robot][task] = _fact(task, truth["requirement"], owners[task], t)
                row_results[robot] = _status_result("inspected", task)
                row_inspections[robot] = {"task": task, "requirement": truth["requirement"], "version": t}
            else:
                row_results[robot] = _status_result("invalid", task if _integer(task) else None)
            local_known = pre_facts[robot]
            if _integer(task) and 0 <= task < N_TASKS and (
                any(f["task"] == task for f in local_known)
                or t > _task_truth(name, task)["deadline_t"]
            ):
                avoidable_inspect += 1

        # Claims are resolved atomically by robot ID.
        claims_by_task: dict[int, list[int]] = {}
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            if row.get("action") != "claim":
                continue
            task = row.get("task")
            if _integer(task) and 0 <= task < N_TASKS and _register_pair_attempt(claim_attempts, robot, task):
                claims_by_task.setdefault(task, []).append(robot)
                known = facts[robot].get(task)
                if known is not None and known.get("claim_robot") not in {None, robot}:
                    conflicting_claim += 1
            else:
                row_results[robot] = _status_result("invalid", task if _integer(task) and 0 <= task < N_TASKS else None)
        for task, claimants in claims_by_task.items():
            truth = _task_truth(name, task)
            winner = _claim_winner(task_states[task], owners[task], task, claimants, facts)
            if winner is not None:
                owners[task] = winner
                task_states[task] = "claimed"
                facts[winner][task] = _fact(task, truth["requirement"], winner, t)
            for robot in claimants:
                row_results[robot] = _status_result("claimed" if robot == winner else "invalid", task)

        # Execution and abandonment use the state after claims.
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            task = row.get("task")
            action = row.get("action")
            if action not in {"execute", "abandon"}:
                continue
            owns_active_claim = _owns_active_claim(task, robot, task_states, owners)
            if action == "abandon":
                if owns_active_claim:
                    owners[task] = None
                    task_states[task] = "available"
                    facts[robot][task] = _fact(task, _task_truth(name, task)["requirement"], None, t)
                    row_results[robot] = _status_result("abandoned", task)
                else:
                    row_results[robot] = _status_result("invalid", task if _integer(task) else None)
            elif action == "execute":
                at_task = bool(
                    owns_active_claim
                    and tuple(start_positions[robot]) == tuple(_task_truth(name, task)["position"])
                )
                if not at_task:
                    row_results[robot] = _status_result("invalid", task if _integer(task) else None)
                elif _failure(name, t, robot, task):
                    owners[task] = None
                    task_states[task] = "available"
                    facts[robot][task] = _fact(task, _task_truth(name, task)["requirement"], None, t)
                    row_results[robot] = _status_result("failed", task)
                else:
                    task_states[task] = "completed"
                    owners[task] = None
                    complete_times[task] = t
                    reasons[task] = "completed"
                    facts[robot][task] = _fact(task, _task_truth(name, task)["requirement"], None, t)
                    row_results[robot] = _status_result("executed", task)

        for task in range(N_TASKS):
            truth = _task_truth(name, task)
            if task_states[task] in {"available", "claimed"} and t >= truth["deadline_t"]:
                if task_states[task] != "completed":
                    task_states[task] = "expired"
                    owners[task] = None
                    reasons[task] = "expired"

        # Create communication events from the pre-action summaries and beginning positions.
        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            if row.get("broadcast"):
                entries = [dict(entry) for entry in row.get("broadcast_entries", [])]
                targets = _targets(name, start_positions, robot, t)
                message = {
                    "message_id": f"r{robot}_t{t}", "sender": robot, "send_t": t,
                    "targets": targets, "delivered_to": list(targets), "entries": entries,
                    "arrival_t": t + 1, "status": "delivered" if targets else "lost",
                }
                expected_messages[message["message_id"]] = message
                if targets and t + 1 < HORIZON:
                    pending.setdefault(t + 1, []).append(message)

        for robot in range(N_ROBOTS):
            row = robots[(robot, t)]
            if row.get("inspection_result") != row_inspections[robot]:
                fail("inspections", f"{name}.robots[{robot},{t}].inspection_result: expected {row_inspections[robot]}, actual={row.get('inspection_result')}")
            if row.get("action_result") != row_results[robot]:
                action = row.get("action")
                if action == "claim":
                    rule = "claims"
                elif action in {"execute", "abandon"}:
                    rule = "executions"
                elif action == "inspect":
                    rule = "inspections"
                else:
                    rule = "action_results"
                fail(rule, f"{name}.robots[{robot},{t}].action_result: expected {row_results[robot]}, actual={row.get('action_result')}")
            if row.get("action") == "move" and robot not in conflicts and row.get("move") != "stay" and list(actual_positions[robot]) != list(planned[robot]):
                fail("positions_moves", f"{name}.robots[{robot},{t}]: movement mismatch")
        positions = actual_positions
        for task in range(N_TASKS):
            timelines[task].append({"t": t, "state": task_states[task], "claimed_by": owners[task]})

    # Compare message and task evidence after causal replay.
    actual_messages = item["messages"]
    if len(actual_messages) != len(item["message_rows"]):
        fail("messages", f"{name}.messages: duplicate message IDs are not allowed")
    for message_id, expected in expected_messages.items():
        if item["messages"].get(message_id) != expected:
            fail("messages", f"{name}.messages[{message_id}]: expected {expected}, actual={item['messages'].get(message_id)}")
    if set(item["messages"]) != set(expected_messages):
        fail("messages", f"{name}.messages: expected IDs {sorted(expected_messages)}, actual={sorted(item['messages'])}")
    for task in range(N_TASKS):
        actual = item["tasks"].get(task)
        truth = _task_truth(name, task)
        expected = {
            "task": task, "position": list(truth["position"]), "release_t": truth["release_t"], "deadline_t": truth["deadline_t"],
            "requirement": truth["requirement"], "timeline": timelines[task], "claimed_by": owners[task],
            "inspect_t": inspect_times[task], "complete_t": complete_times[task], "completion_reason": reasons[task],
        }
        if actual != expected:
            fail("task_records", f"{name}.tasks[{task}]: expected fixed replay record, actual={actual}")
        if actual is not None and actual.get("timeline") != timelines[task]:
            fail("task_timelines", f"{name}.tasks[{task}].timeline: expected replay timeline")

    for task in range(N_TASKS):
        if complete_times[task] is not None:
            completed += 1
            truth = _task_truth(name, task)
            if truth["deadline_t"] > truth["release_t"]:
                utility_sum += max(0.0, min(1.0, 1.0 - (complete_times[task] - truth["release_t"]) / (truth["deadline_t"] - truth["release_t"])))
    return {
        "completed": completed / N_TASKS,
        "utility": utility_sum / N_TASKS,
        "broadcasts": broadcast_count,
        "avoidable_inspect": avoidable_inspect,
        "conflicting_claim": conflicting_claim,
    }, checks, details


def _replay_atoms(indexed: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    names = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    aggregate = {name: True for name in names}
    details = {name: "" for name in names}
    replayed: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIO_NAMES:
        if scenario not in indexed:
            for name in names:
                aggregate[name] = False
                details[name] = "all canonical scenarios are required"
            continue
        try:
            result, checks, local = _replay_scenario(scenario, indexed[scenario])
        except Exception as exc:  # noqa: BLE001
            result = None
            checks = {name: False for name in names}
            local = {name: f"replay raised {type(exc).__name__}: {exc}" for name in names}
        if result is not None:
            replayed[scenario] = result
        for name, ok in checks.items():
            if not ok:
                aggregate[name] = False
                if not details[name]:
                    details[name] = local[name]
    return [_atom("replay_validity", name, aggregate[name], details[name] or "replay matched") for name in names], replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    completion = sum(replayed[name]["completed"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    utility = sum(replayed[name]["utility"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    communication = sum(max(0.0, min(1.0, 1.0 - replayed[name]["broadcasts"] / 72.0)) for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    avoidable = sum(replayed[name]["avoidable_inspect"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    conflicts = sum(replayed[name]["conflicting_claim"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    control = sum(0.5 * max(0.0, min(1.0, 1.0 - replayed[name]["avoidable_inspect"] / 48.0)) + 0.5 * max(0.0, min(1.0, 1.0 - replayed[name]["conflicting_claim"] / 48.0)) for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    return {
        "primary_metrics_1": completion, "primary_metrics_2": utility,
        "auxiliary_metrics_1": communication, "auxiliary_metrics_2": control,
        "normalized_task_performance": (8 * completion + 7 * utility + 3 * communication + 2 * control) / 20,
    }


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
    required = {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"}
    lines = text.splitlines()
    parsed: list[tuple[int, int, int, str]] = []
    heading_lines: set[int] = set()
    fence_lines: set[int] = set()
    fence_char: str | None = None
    fence_length = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_char is not None:
            if fence:
                marker = fence.group(1)
                if marker[0] == fence_char and len(marker) >= fence_length and not fence.group(2).strip():
                    fence_char = None
                    fence_length = 0
                fence_lines.add(i)
            i += 1
            continue
        if fence:
            marker = fence.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            fence_lines.add(i)
            i += 1
            continue

        atx = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", line)
        if atx:
            title = (atx.group(2) or "").strip()
            title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
            parsed.append((i, i, len(atx.group(1)), title))
            heading_lines.add(i)
            i += 1
            continue

        if i + 1 < len(lines):
            title_match = re.match(r"^ {0,3}(\S.*?)\s*$", line)
            underline = re.match(r"^ {0,3}(=+|-+)[ \t]*$", lines[i + 1])
            if title_match and underline:
                level = 1 if underline.group(1)[0] == "=" else 2
                parsed.append((i, i + 1, level, title_match.group(1).strip()))
                heading_lines.update((i, i + 1))
                i += 2
                continue
        i += 1

    found: dict[str, str] = {}
    for index, (start, end, level, title) in enumerate(parsed):
        if title not in required:
            continue
        stop = len(lines)
        for next_start, _, next_level, _ in parsed[index + 1:]:
            if next_level <= level:
                stop = next_start
                break
        body = "\n".join(
            lines[line_index]
            for line_index in range(end + 1, stop)
            if line_index not in heading_lines and line_index not in fence_lines
        ).strip()
        if title not in found or not found[title]:
            found[title] = body
    return found


def _report_score(ws: Workspace) -> tuple[float, str]:
    required = {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"}
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a regular non-symlink file"
    try:
        text = ws.read_text("report.md")
    except OSError as exc:
        return 0.0, f"report.md: read failed: {exc}"
    headings = _headings(text or "")
    missing = sorted(name for name in required if not headings.get(name))
    score = max(0.0, (2.0 - 0.5 * len(missing)) / 2.0)
    return score, f"report.md: missing_or_empty={missing}; normalized={score:.9g}"


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
            removed_ok = getattr(removed, "exit_code", None) == 0
            absent = await ws.run("test ! -e results.json && test ! -L results.json", check=False, timeout=30.0)
            remove_ok = removed_ok and getattr(absent, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
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
        state.official_ok = bool(remove_ok and state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600 + EPS and state.result_file_is_new_regular)
        if not state.official_message:
            state.official_message = f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, new_regular_results={state.result_file_is_new_regular}"
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"results.json: strict parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose(result_atoms))
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            if not state.official_ok:
                state.scores["result_json"] = 0.0
                state.diagnostics["result_json"] = reason
            for check_id in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = reason
            _CACHE[key] = state
            return state
        state.scenarios = _index_records(state.data)
        state.records = state.scenarios
        evidence = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:20]}"
        legality = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose(legality)
        replay, state.replayed = _replay_atoms(state.scenarios)
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose(replay)
        gate_names = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        state.validity_gate = state.official_ok and all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in gate_names)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        if not state.run_finished:
            state.official_ok = False
            state.official_message = state.official_message or detail
        else:
            state.metrics = None
            state.validity_gate = False
            state.scores.setdefault("result_json", 0.0)
            state.diagnostics.setdefault("result_json", detail)
            for check_id in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = detail
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
    return _report_score(ws)


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


async def _metric(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if not state.validity_gate or state.metrics is None:
        return 0.0, f"{name}: validity gate not satisfied; metric not computed"
    return state.metrics[name], f"{name}: independently replayed value={state.metrics[name]:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "primary_metrics_2")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "auxiliary_metrics_2")
