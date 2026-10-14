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


HORIZON = 32
N_ROBOTS = 4
EPS = 1e-6
MAX_FACTS = 12
MAX_BROADCASTS = 20
WINDOW = 4
WINDOW_LIMIT = 2
SCENARIO_NAMES = ("corridor", "rooms", "crossing", "dead_end", "open_field")
STARTS = ((1, 1), (1, 8), (8, 1), (8, 8))
MAP_ROWS: dict[str, tuple[str, ...]] = {
    "corridor": (
        "....##....", "....##....", "....##....", "....##....", "..........",
        "....##....", "....##....", "....##....", "....##....", "....##....",
    ),
    "rooms": (
        "...#..#...", "...#..#...", "..........", "...#..#...", "...#..#...",
        "...#..#...", "...#..#...", "..........", "...#..#...", "...#..#...",
    ),
    "crossing": (
        "....##....", "....##....", "....##....", "....##....", "..........",
        "..........", "....##....", "....##....", "....##....", "....##....",
    ),
    "dead_end": (
        "..........", "..........", "....##....", "....##....", "....##....",
        ".#..#...#.", "....##....", "....##....", "....##....", "....##....",
    ),
    "open_field": (
        "..........", "..#.......", ".......#..", "..........", "....#.....",
        ".....#....", "..........", "..#.......", ".......#..", "..........",
    ),
}
MOVES: dict[str, tuple[int, int]] = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0), "stay": (0, 0)}
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {"result_json.parse": 0.34, "result_json.root": 0.33, "result_json.scenarios": 0.33},
    "result_evidence": {
        "result_evidence.scenario_shape": 1 / 9, "result_evidence.robot_shape": 1 / 9,
        "result_evidence.robot_fields": 1 / 9, "result_evidence.event_shape": 1 / 9,
        "result_evidence.event_fields": 1 / 9, "result_evidence.coverage_shape": 1 / 9,
        "result_evidence.sync_shape": 1 / 9, "result_evidence.fact_fields": 1 / 9,
        "result_evidence.array_values": 1 / 9,
    },
    "decision_legality": {
        "decision_legality.robot_keys": 1 / 8, "decision_legality.role_domain": 1 / 8,
        "decision_legality.move_domain": 1 / 8, "decision_legality.position_domain": 1 / 8,
        "decision_legality.role_dwell": 1 / 8, "decision_legality.switch_occupancy": 1 / 8,
        "decision_legality.broadcast_role": 1 / 8, "decision_legality.broadcast_budget": 1 / 8,
    },
    "replay_validity": {
        "replay_validity.positions": 1 / 9, "replay_validity.roles": 1 / 9,
        "replay_validity.events": 1 / 9, "replay_validity.receipts": 1 / 9,
        "replay_validity.observations": 1 / 9, "replay_validity.moves": 1 / 9,
        "replay_validity.coverage": 1 / 9, "replay_validity.sync": 1 / 9,
        "replay_validity.resource_state": 1 / 9,
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


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _close(a: Any, b: Any) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= EPS


def _cell_xy(cell: int) -> tuple[int, int]:
    return cell % 10, cell // 10


def _cell(x: int, y: int) -> int:
    return x + 10 * y


def _state_at(scenario: str, x: int, y: int) -> str:
    return "free" if 0 <= x < 10 and 0 <= y < 10 and MAP_ROWS[scenario][9 - y][x] == "." else "obstacle"


def _free_cells(scenario: str) -> set[int]:
    return {_cell(x, y) for y in range(10) for x in range(10) if _state_at(scenario, x, y) == "free"}


def _shortest_clear(scenario: str, start: tuple[int, int], end: tuple[int, int]) -> bool:
    if start == end or _state_at(scenario, *start) != "free" or _state_at(scenario, *end) != "free":
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    nx, ny = abs(dx), abs(dy)
    memo: dict[tuple[int, int], bool] = {}

    def walk(x: int, y: int, rx: int, ry: int) -> bool:
        key = (x * 1000 + y, rx * 100 + ry)
        if key in memo:
            return memo[key]
        if _state_at(scenario, x, y) != "free":
            memo[key] = False
            return False
        if rx == 0 and ry == 0:
            memo[key] = True
            return True
        ok = False
        if rx:
            ok = walk(x + sx, y, rx - 1, ry)
        if not ok and ry:
            ok = walk(x, y + sy, rx, ry - 1)
        memo[key] = ok
        return ok

    return walk(start[0], start[1], nx, ny)


def _targets(scenario: str, positions: Sequence[tuple[int, int]], sender: int) -> list[int]:
    result: list[int] = []
    sx, sy = positions[sender]
    for receiver, (rx, ry) in enumerate(positions):
        if receiver == sender:
            continue
        if abs(sx - rx) + abs(sy - ry) <= 3 and _shortest_clear(scenario, (sx, sy), (rx, ry)):
            result.append(receiver)
    return result


def _fact(cell: int, state: str, version: int) -> dict[str, Any]:
    return {"cell": int(cell), "state": state, "version": int(version)}


def _facts_equal(left: Any, right: Any) -> bool:
    return isinstance(left, list) and isinstance(right, list) and left == right


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
    """Apply strict UTF-8 decoding before the JSON duplicate-key policy."""
    return _strict_json(raw.decode("utf-8"))


def _fact_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"cell", "state", "version"}:
        return False
    cell = value.get("cell")
    state = value.get("state")
    version = value.get("version")
    return (
        _integer(cell) and 0 <= cell < 100
        and isinstance(state, str) and state in {"free", "obstacle"}
        and _integer(version) and 0 <= version < HORIZON
    )


def _robot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "robot", "t", "position", "role", "role_age", "move", "broadcast",
        "broadcast_facts", "observed_facts", "received_event_ids",
    }


def _event_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "event_id", "sender", "send_t", "facts", "targets", "delivered_to", "arrival_t", "status",
    }


def _coverage_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"t", "cells"}


def _sync_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"event_id", "sender", "receiver", "arrival_t", "facts"}


def _pair_int(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_integer(v) for v in value)


def _robot_valid(value: Any) -> bool:
    if not _robot_shape(value):
        return False
    robot = value.get("robot")
    slot = value.get("t")
    position = value.get("position")
    role = value.get("role")
    age = value.get("role_age")
    move = value.get("move")
    broadcast_facts = value.get("broadcast_facts")
    observed_facts = value.get("observed_facts")
    received = value.get("received_event_ids")
    if not (_integer(robot) and 0 <= robot < N_ROBOTS
            and _integer(slot) and 0 <= slot < HORIZON
            and _pair_int(position) and all(0 <= v < 10 for v in position)
            and isinstance(role, str) and role in {"explorer", "messenger"}
            and _integer(age) and 1 <= age <= HORIZON
            and isinstance(move, str) and move in MOVES
            and isinstance(value.get("broadcast"), bool)):
        return False
    if not isinstance(broadcast_facts, list) or len(broadcast_facts) > MAX_FACTS:
        return False
    if not all(_fact_valid(f) for f in broadcast_facts):
        return False
    broadcast_cells = [f["cell"] for f in broadcast_facts]
    if len(set(broadcast_cells)) != len(broadcast_cells):
        return False
    if not isinstance(observed_facts, list) or not all(_fact_valid(f) for f in observed_facts):
        return False
    observed_cells = [f["cell"] for f in observed_facts]
    if observed_cells != sorted(set(observed_cells)):
        return False
    return isinstance(received, list) and all(isinstance(v, str) for v in received) and received == sorted(set(received))


def _event_valid(value: Any) -> bool:
    if not _event_shape(value):
        return False
    event_id = value.get("event_id")
    sender = value.get("sender")
    send_t = value.get("send_t")
    facts = value.get("facts")
    targets = value.get("targets")
    delivered_to = value.get("delivered_to")
    arrival_t = value.get("arrival_t")
    status = value.get("status")
    target_lists_ok = (
        isinstance(targets, list)
        and all(_integer(v) and 0 <= v < N_ROBOTS for v in targets)
        and targets == sorted(set(targets))
        and isinstance(delivered_to, list)
        and all(_integer(v) and 0 <= v < N_ROBOTS for v in delivered_to)
        and delivered_to == sorted(set(delivered_to))
        and isinstance(sender, int) and sender not in targets
    )
    facts_ok = (
        isinstance(facts, list)
        and len(facts) <= MAX_FACTS
        and all(_fact_valid(f) for f in facts)
        and len({f["cell"] for f in facts}) == len(facts)
    )
    return (
        isinstance(event_id, str)
        and _integer(sender) and 0 <= sender < N_ROBOTS
        and _integer(send_t) and 0 <= send_t < HORIZON
        and event_id == f"r{sender}_t{send_t}"
        and facts_ok
        and target_lists_ok
        and delivered_to == targets
        and _integer(arrival_t) and arrival_t == send_t + 1
        and isinstance(status, str)
        and status == ("delivered" if bool(targets) else "lost")
    )


def _coverage_valid(value: Any) -> bool:
    if not _coverage_shape(value):
        return False
    slot = value.get("t")
    cells = value.get("cells")
    return (
        _integer(slot) and 0 <= slot < HORIZON
        and isinstance(cells, list)
        and all(_integer(c) and 0 <= c < 100 for c in cells)
        and cells == sorted(set(cells))
    )


def _sync_valid(value: Any) -> bool:
    if not _sync_shape(value):
        return False
    sender = value.get("sender")
    receiver = value.get("receiver")
    facts = value.get("facts")
    return (
        isinstance(value.get("event_id"), str)
        and _integer(sender) and 0 <= sender < N_ROBOTS
        and _integer(receiver) and 0 <= receiver < N_ROBOTS and receiver != sender
        and _integer(value.get("arrival_t")) and 0 <= value["arrival_t"] < HORIZON
        and isinstance(facts, list) and all(_fact_valid(f) for f in facts)
        and len({f["cell"] for f in facts}) == len(facts)
    )


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"scenario", "robots", "events", "coverage", "sync"}


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for record in data["scenarios"]:
        if not isinstance(record, dict) or not isinstance(record.get("scenario"), str):
            continue
        name = record["scenario"]
        if name not in SCENARIO_NAMES:
            # Unknown units are reported by canonical-coverage analysis, but
            # never enter the canonical replay/index used by other atoms.
            continue
        if name in indexed:
            continue
        robot_map: dict[tuple[int, int], Any] = {}
        for row in record.get("robots", []) if isinstance(record.get("robots"), list) else []:
            if _robot_shape(row) and _integer(row.get("robot")) and _integer(row.get("t")):
                robot_map.setdefault((row["robot"], row["t"]), row)
        event_map: dict[str, Any] = {}
        for event in record.get("events", []) if isinstance(record.get("events"), list) else []:
            if isinstance(event, dict) and isinstance(event.get("event_id"), str):
                event_map.setdefault(event["event_id"], event)
        coverage_map: dict[int, Any] = {}
        for row in record.get("coverage", []) if isinstance(record.get("coverage"), list) else []:
            if isinstance(row, dict) and _integer(row.get("t")):
                coverage_map.setdefault(row["t"], row)
        sync_rows = record.get("sync", []) if isinstance(record.get("sync"), list) else []
        sync_map: dict[tuple[str, int], Any] = {}
        for row in sync_rows:
            if isinstance(row, dict) and isinstance(row.get("event_id"), str) and _integer(row.get("receiver")):
                sync_map.setdefault((row["event_id"], row["receiver"]), row)
        indexed[name] = {
            "record": record, "robot_rows": record.get("robots", []), "event_rows": record.get("events", []),
            "coverage_rows": record.get("coverage", []), "sync_rows": sync_rows,
            "robots": robot_map, "events": event_map, "coverage": coverage_map, "sync": sync_map,
        }
    return indexed


def _atom(check_id: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{name}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share differs")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[{}]; {}".format(",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed))


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root = parse_ok and isinstance(data, dict)
    keys = root and set(data) == {"scenarios"}
    scenarios = keys and isinstance(data.get("scenarios"), list)
    actual_keys = list(data) if root else None
    return [
        _atom("result_json", "parse", bool(parse_ok), "results.json: strict JSON parse"),
        _atom("result_json", "root", bool(keys), f"results.json: expected root keys {{'scenarios'}}, actual={actual_keys}"),
        _atom("result_json", "scenarios", bool(scenarios), "results.json.scenarios: expected array"),
    ]


def _complete(indexed: Mapping[str, Any]) -> bool:
    # Extra scenario records are a canonical-coverage defect only; they must
    # not make otherwise complete canonical records unevaluable by every
    # evidence/legality/replay atom.
    return set(SCENARIO_NAMES).issubset(indexed)


def _evidence_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = _complete(indexed)
    scenario_ok = complete and all(_scenario_shape(indexed[n]["record"]) for n in SCENARIO_NAMES)
    def rows(name: str, key: str) -> list[Any] | None:
        value = indexed[name].get(key)
        return value if isinstance(value, list) else None

    robot_shape = complete and all(
        (values := rows(n, "robot_rows")) is not None
        and len(values) == N_ROBOTS * HORIZON
        and all(_robot_shape(r) for r in values)
        for n in SCENARIO_NAMES
    )
    robot_fields = complete and all(
        (values := rows(n, "robot_rows")) is not None
        and len(values) == N_ROBOTS * HORIZON
        and all(_robot_valid(r) for r in values)
        for n in SCENARIO_NAMES
    )
    event_shape = complete and all(
        (values := rows(n, "event_rows")) is not None
        and all(_event_shape(e) for e in values)
        for n in SCENARIO_NAMES
    )
    event_fields = complete and all(
        (values := rows(n, "event_rows")) is not None
        and all(_event_valid(e) for e in values)
        for n in SCENARIO_NAMES
    )
    coverage_shape = complete and all(
        (values := rows(n, "coverage_rows")) is not None
        and len(values) == HORIZON
        and all(_coverage_shape(c) for c in values)
        for n in SCENARIO_NAMES
    )
    sync_shape = complete and all(
        (values := rows(n, "sync_rows")) is not None
        and all(_sync_shape(s) for s in values)
        for n in SCENARIO_NAMES
    )

    def facts_valid(name: str) -> bool:
        values = rows(name, "robot_rows")
        if values is None or len(values) != N_ROBOTS * HORIZON:
            return False
        for row in values:
            if not isinstance(row, dict):
                return False
            broadcast_facts = row.get("broadcast_facts")
            observed_facts = row.get("observed_facts")
            if not isinstance(broadcast_facts, list) or not isinstance(observed_facts, list):
                return False
            if not all(_fact_valid(f) for f in broadcast_facts + observed_facts):
                return False
        return True

    fact_fields = complete and all(facts_valid(n) for n in SCENARIO_NAMES)
    array_values = complete and all(
        (coverage := rows(n, "coverage_rows")) is not None
        and len(coverage) == HORIZON
        and (sync := rows(n, "sync_rows")) is not None
        and all(_coverage_valid(c) for c in coverage)
        and all(_sync_valid(s) for s in sync)
        for n in SCENARIO_NAMES
    )
    return [
        _atom("result_evidence", "scenario_shape", scenario_ok, "scenario objects have exact keys"),
        _atom("result_evidence", "robot_shape", robot_shape, "robot records have exact keys"),
        _atom("result_evidence", "robot_fields", robot_fields, "robot fields satisfy declared types and domains"),
        _atom("result_evidence", "event_shape", event_shape, "event records have exact keys"),
        _atom("result_evidence", "event_fields", event_fields, "event fields satisfy declared types and domains"),
        _atom("result_evidence", "coverage_shape", coverage_shape, "coverage records have exact keys"),
        _atom("result_evidence", "sync_shape", sync_shape, "sync records have exact keys"),
        _atom("result_evidence", "fact_fields", fact_fields, "all fact objects satisfy cell/state/version contract"),
        _atom("result_evidence", "array_values", array_values, "coverage and sync arrays have valid values"),
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not a list"]
    by_name: dict[str, list[Any]] = {n: [] for n in SCENARIO_NAMES}
    for idx, rec in enumerate(data["scenarios"]):
        name = rec.get("scenario") if isinstance(rec, dict) else None
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"scenarios[{idx}]: unknown scenario")
        else:
            by_name[name].append(rec)
    supplied_order = [rec.get("scenario") for rec in data["scenarios"] if isinstance(rec, dict)]
    if supplied_order != list(SCENARIO_NAMES):
        defects.append(f"scenarios: expected canonical order {list(SCENARIO_NAMES)}, actual={supplied_order}")
    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"{name}: missing scenario")
            continue
        defects.extend(f"{name}: duplicate scenario" for _ in range(len(matches) - 1))
        rec = matches[0]
        robots = rec.get("robots") if isinstance(rec, dict) else None
        expected = {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}
        counts = {key: 0 for key in expected}
        if not isinstance(robots, list):
            defects.append(f"{name}.robots: missing array")
        else:
            for idx, row in enumerate(robots):
                key = (
                    (row.get("robot"), row.get("t"))
                    if isinstance(row, dict)
                    and _integer(row.get("robot"))
                    and _integer(row.get("t"))
                    else None
                )
                if key is not None and key in counts:
                    counts[key] += 1
                else:
                    defects.append(f"{name}.robots[{idx}]: unknown key")
            defects.extend(f"{name}.robots: missing key {k}" for k, c in sorted(counts.items()) if c == 0)
            defects.extend(f"{name}.robots: duplicate key {k}" for k, c in sorted(counts.items()) for _ in range(max(0, c - 1)))
        cov = rec.get("coverage") if isinstance(rec, dict) else None
        cc = {t: 0 for t in range(HORIZON)}
        if not isinstance(cov, list):
            defects.append(f"{name}.coverage: missing array")
        else:
            for idx, row in enumerate(cov):
                key = (
                    row.get("t")
                    if isinstance(row, dict) and _integer(row.get("t"))
                    else None
                )
                if key is not None and key in cc:
                    cc[key] += 1
                else:
                    defects.append(f"{name}.coverage[{idx}]: unknown key")
            defects.extend(f"{name}.coverage: missing key {k}" for k, c in sorted(cc.items()) if c == 0)
            defects.extend(f"{name}.coverage: duplicate key {k}" for k, c in sorted(cc.items()) for _ in range(max(0, c - 1)))
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _legality_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    complete = _complete(indexed)
    def robot_rows(name: str) -> list[Any] | None:
        value = indexed[name].get("robot_rows")
        return value if isinstance(value, list) else None

    def every_row(name: str, predicate: Any) -> bool:
        values = robot_rows(name)
        return values is not None and len(values) == N_ROBOTS * HORIZON and all(
            isinstance(row, dict) and predicate(row) for row in values
        )

    robot_keys = complete and all(
        isinstance(indexed[n].get("robot_rows"), list)
        and len(indexed[n]["robot_rows"]) == N_ROBOTS * HORIZON
        and set(indexed[n]["robots"]) == {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}
        for n in SCENARIO_NAMES
    )
    role_domain = complete and all(
        every_row(n, lambda r: isinstance(r.get("role"), str) and r.get("role") in {"explorer", "messenger"})
        for n in SCENARIO_NAMES
    )
    move_domain = complete and all(
        every_row(n, lambda r: isinstance(r.get("move"), str) and r.get("move") in MOVES)
        for n in SCENARIO_NAMES
    )
    position_domain = complete and all(
        every_row(n, lambda r: _pair_int(r.get("position")) and all(0 <= v < 10 for v in r["position"]))
        for n in SCENARIO_NAMES
    )
    dwell_ok = complete
    switch_ok = complete
    broadcast_role = complete
    budget_ok = complete
    for name in SCENARIO_NAMES:
        if name not in indexed:
            continue
        m = indexed[name]["robots"]
        for robot in range(N_ROBOTS):
            previous_role = "explorer"
            previous_switch = -100
            count = 0
            for t in range(HORIZON):
                row = m.get((robot, t))
                if not isinstance(row, dict):
                    dwell_ok = switch_ok = broadcast_role = budget_ok = False
                    continue
                changed = row.get("role") != previous_role
                if changed:
                    if t == 0 or t - previous_switch < 4:
                        switch_ok = False
                    previous_switch = t
                if changed or t == 0:
                    expected_age = 1
                else:
                    previous_row = m.get((robot, t - 1), {})
                    previous_age = previous_row.get("role_age") if isinstance(previous_row, dict) else None
                    if _integer(previous_age):
                        expected_age = previous_age + 1
                    else:
                        expected_age = None
                        dwell_ok = False
                if row.get("role_age") != expected_age:
                    dwell_ok = False
                if changed and (row.get("move") != "stay" or row.get("broadcast") is not False):
                    switch_ok = False
                broadcast_value = row.get("broadcast")
                facts_value = row.get("broadcast_facts")
                if not isinstance(broadcast_value, bool):
                    broadcast_role = False
                elif broadcast_value:
                    if row.get("role") != "messenger":
                        broadcast_role = False
                    if (
                        not isinstance(facts_value, list)
                        or len(facts_value) > MAX_FACTS
                        or not all(_fact_valid(fact) for fact in facts_value)
                        or len({fact["cell"] for fact in facts_value if _fact_valid(fact)}) != len(facts_value)
                    ):
                        broadcast_role = False
                elif isinstance(facts_value, list) and facts_value:
                    broadcast_role = False
                if broadcast_value is True:
                    count += 1
                previous_role = row.get("role", previous_role)
            if count > MAX_BROADCASTS:
                budget_ok = False
            for start in range(HORIZON - WINDOW + 1):
                if sum(bool(m.get((robot, t), {}).get("broadcast")) for t in range(start, start + WINDOW)) > WINDOW_LIMIT:
                    budget_ok = False
    return [
        _atom("decision_legality", "robot_keys", robot_keys, "exact robot-slot keys are required"),
        _atom("decision_legality", "role_domain", role_domain, "role is explorer or messenger"),
        _atom("decision_legality", "move_domain", move_domain, "move is in the declared alphabet"),
        _atom("decision_legality", "position_domain", position_domain, "positions are integer grid coordinates"),
        _atom("decision_legality", "role_dwell", dwell_ok, "role ages and four-slot dwell rule"),
        _atom("decision_legality", "switch_occupancy", switch_ok, "role changes occupy slot and require stay/no broadcast"),
        _atom("decision_legality", "broadcast_role", broadcast_role, "only stable messengers broadcast a valid summary"),
        _atom("decision_legality", "broadcast_budget", budget_ok, "per-robot budget and sliding window"),
    ]


def _same(a: Any, b: Any) -> bool:
    return a == b


def _replay_scenario(name: str, item: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, bool], dict[str, str]]:
    rules = [k.removeprefix("replay_validity.") for k in ATOM_SHARES["replay_validity"]]
    checks = {r: True for r in rules}
    details = {r: "" for r in rules}

    def fail(rule: str, detail: str) -> None:
        if checks[rule]:
            checks[rule] = False
            details[rule] = detail

    expected_robot_keys = {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}
    robots = item.get("robots") if isinstance(item, dict) else None
    if not isinstance(robots, dict) or set(robots) != expected_robot_keys:
        # Canonical key and record-shape checks own an incomplete decision map.
        # Do not charge every fixed-model atom for a replay that cannot be run.
        for rule in rules:
            details[rule] = f"{name}.robots: replay not evaluated; complete robot-slot index is unavailable"
        return None, checks, details

    def valid_position(value: Any) -> bool:
        return _pair_int(value) and all(0 <= coordinate < 10 for coordinate in value)

    def valid_role(value: Any) -> bool:
        return isinstance(value, str) and value in {"explorer", "messenger"}

    def valid_move(value: Any) -> bool:
        return isinstance(value, str) and value in MOVES

    def valid_observed(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(_fact_valid(f) for f in value)
            and [f["cell"] for f in value] == sorted({f["cell"] for f in value})
        )

    def valid_received(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(v, str) for v in value) and value == sorted(set(value))

    positions = [tuple(p) for p in STARTS]
    roles = ["explorer"] * N_ROBOTS
    role_ages = [1] * N_ROBOTS
    known: list[dict[int, dict[str, Any]]] = [{_cell(*STARTS[r]): _fact(_cell(*STARTS[r]), "free", 0)} for r in range(N_ROBOTS)]
    acquired: list[dict[int, int]] = [{_cell(*STARTS[r]): 0} for r in range(N_ROBOTS)]
    pending: dict[int, list[tuple[str, int, list[dict[str, Any]]]]] = {}
    expected_events: dict[str, dict[str, Any]] = {}
    expected_sync: dict[tuple[str, int], dict[str, Any]] = {}
    expected_coverage: dict[int, dict[str, Any]] = {}
    global_free: set[int] = set()
    tx_count = 0
    switch_count = 0
    freshness_sum = 0.0
    freshness_den = 0

    for t in range(HORIZON):
        received: list[list[str]] = [[] for _ in range(N_ROBOTS)]
        for event_id, sender, facts in pending.pop(t, []):
            event = expected_events[event_id]
            for receiver in event["targets"]:
                received[receiver].append(event_id)
                adopted: list[dict[str, Any]] = []
                for fact in facts:
                    old = known[receiver].get(fact["cell"])
                    if old is None or fact["version"] > old["version"]:
                        known[receiver][fact["cell"]] = dict(fact)
                        acquired[receiver][fact["cell"]] = t
                        adopted.append(dict(fact))
                expected_sync[(event_id, receiver)] = {
                    "event_id": event_id, "sender": sender, "receiver": receiver,
                    "arrival_t": t, "facts": adopted,
                }

        proposed: list[tuple[int, int]] = []
        for r in range(N_ROBOTS):
            row = robots.get((r, t))
            if not isinstance(row, dict):
                row = {}
            position_value = row.get("position")
            if valid_position(position_value) and position_value != list(positions[r]):
                fail("positions", f"{name}.robots[{r},{t}].position: expected {list(positions[r])}, actual={row.get('position')}")
            new_role = row.get("role")
            role_is_valid = valid_role(new_role)
            # An invalid role is owned by evidence/legality.  Keep the last
            # replay role so one malformed row does not manufacture a cascade
            # of fixed-model failures in later slots.
            is_changed = role_is_valid and new_role != roles[r]
            if is_changed:
                switch_count += 1
                if t == 0 or role_ages[r] < 4:
                    fail("roles", f"{name}.robots[{r},{t}]: role change before four held slots")
                role_ages[r] = 1
            else:
                role_ages[r] += 1 if t > 0 else 0
            role_age_value = row.get("role_age")
            if _integer(role_age_value) and 1 <= role_age_value <= HORIZON and role_age_value != role_ages[r]:
                fail("roles", f"{name}.robots[{r},{t}].role_age: expected {role_ages[r]}, actual={row.get('role_age')}")
            move_value = row.get("move")
            move_is_valid = valid_move(move_value)
            broadcast_value = row.get("broadcast")
            broadcast_is_valid = isinstance(broadcast_value, bool)
            if is_changed and move_is_valid and broadcast_is_valid and (move_value != "stay" or broadcast_value is not False):
                fail("roles", f"{name}.robots[{r},{t}]: role change must stay and not broadcast")
            move = move_value if move_is_valid else "stay"
            dx, dy = MOVES[move]
            proposed.append((positions[r][0] + dx, positions[r][1] + dy))
            if is_changed:
                proposed[-1] = positions[r]
            if move_is_valid and move != "stay" and _state_at(name, *proposed[-1]) != "free":
                fail("moves", f"{name}.robots[{r},{t}].move: destination {proposed[-1]} is not free")
            broadcast = broadcast_value if broadcast_is_valid else False
            facts_value = row.get("broadcast_facts")
            facts_is_valid = (
                isinstance(facts_value, list)
                and len(facts_value) <= MAX_FACTS
                and all(_fact_valid(f) for f in facts_value)
                and len({f["cell"] for f in facts_value}) == len(facts_value)
            )
            if broadcast:
                tx_count += 1
                if role_is_valid and (new_role != "messenger" or is_changed):
                    fail("roles", f"{name}.robots[{r},{t}]: broadcast requires stable messenger role")
            facts = facts_value if isinstance(facts_value, list) else []
            if broadcast_is_valid and broadcast and not facts_is_valid:
                fail("resource_state", f"{name}.robots[{r},{t}].broadcast_facts: invalid, duplicate, or over-limit summary")
            if broadcast_is_valid and broadcast and facts_is_valid and any(
                f["cell"] not in known[r] or known[r][f["cell"]] != f for f in facts
            ):
                fail("resource_state", f"{name}.robots[{r},{t}].broadcast_facts: fact is not currently known")
            if broadcast_is_valid and not broadcast and isinstance(facts_value, list) and facts:
                fail("resource_state", f"{name}.robots[{r},{t}].broadcast_facts: expected empty when broadcast=false")
            if broadcast:
                event_id = f"r{r}_t{t}"
                targets = _targets(name, positions, r)
                event_facts = [dict(f) for f in facts if _fact_valid(f)]
                event = {
                    "event_id": event_id, "sender": r, "send_t": t, "facts": event_facts,
                    "targets": targets, "delivered_to": list(targets), "arrival_t": t + 1,
                    "status": "delivered" if targets else "lost",
                }
                expected_events[event_id] = event
                if t + 1 < HORIZON and targets:
                    pending.setdefault(t + 1, []).append((event_id, r, event_facts))
            if role_is_valid:
                roles[r] = new_role

        # Same destination and exchanged positions are resolved as all-stay conflicts.
        conflict: set[int] = set()
        for i in range(N_ROBOTS):
            for j in range(i + 1, N_ROBOTS):
                if proposed[i] == proposed[j] or (proposed[i] == positions[j] and proposed[j] == positions[i]):
                    conflict.update((i, j))
        actual_positions = [positions[r] if r in conflict else proposed[r] for r in range(N_ROBOTS)]
        for r in range(N_ROBOTS):
            row = robots.get((r, t), {})
            move_value = row.get("move") if isinstance(row, dict) else None
            role_value = row.get("role") if isinstance(row, dict) else None
            if valid_role(role_value) and valid_move(move_value) and move_value != "stay" and r not in conflict and list(actual_positions[r]) != list(proposed[r]):
                fail("moves", f"{name}.robots[{r},{t}]: movement mismatch")
            obs_cells = []
            if roles[r] == "explorer":
                for x in range(10):
                    for y in range(10):
                        if abs(x - actual_positions[r][0]) + abs(y - actual_positions[r][1]) <= 1:
                            obs_cells.append(_cell(x, y))
            else:
                obs_cells = [_cell(*actual_positions[r])]
            observed = [_fact(c, _state_at(name, *_cell_xy(c)), t) for c in sorted(obs_cells)]
            for fact in observed:
                old = known[r].get(fact["cell"])
                if old is None or fact["version"] >= old["version"]:
                    known[r][fact["cell"]] = dict(fact)
                    acquired[r][fact["cell"]] = t
            row_observed = row.get("observed_facts")
            # A malformed role/move/observation field is owned by the
            # corresponding evidence/legality atom; compare observations only
            # when the submitted local fields provide an interpretable record.
            if valid_role(role_value) and valid_move(move_value) and valid_observed(row_observed) and not _facts_equal(row_observed, observed):
                fail("observations", f"{name}.robots[{r},{t}].observed_facts: expected {observed}, actual={row_observed}")
            row_received = row.get("received_event_ids") if isinstance(row, dict) else None
            if valid_received(row_received) and row_received != sorted(received[r]):
                fail("receipts", f"{name}.robots[{r},{t}].received_event_ids: expected {sorted(received[r])}, actual={row.get('received_event_ids')}")
        positions = actual_positions
        global_free.update(c for r in range(N_ROBOTS) for c, f in known[r].items() if f["state"] == "free")
        expected_coverage[t] = {"t": t, "cells": sorted(global_free)}
        for r in range(N_ROBOTS):
            for cell in global_free:
                freshness_den += 1
                k = acquired[r].get(cell)
                freshness_sum += max(0.0, 1.0 - (t - k) / 4.0) if k is not None else 0.0

    # Compare variable evidence after the full causal replay.  Structural
    # array/type defects belong to evidence; keep replay safe and diagnostic.
    event_rows = item.get("event_rows")
    event_map = item.get("events") if isinstance(item.get("events"), dict) else {}
    events_interpretable = isinstance(event_rows, list) and all(_event_shape(row) for row in event_rows)
    if events_interpretable:
        if len(event_rows) != len(event_map):
            fail("events", f"{name}.events: duplicate event_id records are not allowed")
        for event_id, expected in expected_events.items():
            actual = event_map.get(event_id)
            if actual is None:
                fail("events", f"{name}.events[{event_id}]: expected event is missing")
            elif _event_valid(actual) and actual != expected:
                fail("events", f"{name}.events[{event_id}]: expected fixed event, actual={actual}")
        extra_events = set(event_map) - set(expected_events)
        if extra_events:
            fail("events", f"{name}.events: unknown event IDs {sorted(extra_events)}")
    sync_rows = item.get("sync_rows")
    sync_map = item.get("sync") if isinstance(item.get("sync"), dict) else {}
    sync_interpretable = isinstance(sync_rows, list) and all(_sync_shape(row) for row in sync_rows)
    if sync_interpretable:
        if len(sync_rows) != len(sync_map):
            fail("sync", f"{name}.sync: duplicate (event_id,receiver) records are not allowed")
        for key, expected in expected_sync.items():
            actual = sync_map.get(key)
            if actual is None:
                fail("sync", f"{name}.sync[{key}]: expected sync record is missing")
            elif _sync_valid(actual) and actual != expected:
                fail("sync", f"{name}.sync[{key}]: expected {expected}, actual={actual}")
        if set(sync_map) != set(expected_sync):
            fail("sync", f"{name}.sync: expected keys {sorted(expected_sync)}, actual={sorted(sync_map)}")
    coverage_rows = item.get("coverage_rows")
    coverage_map = item.get("coverage") if isinstance(item.get("coverage"), dict) else {}
    coverage_interpretable = isinstance(coverage_rows, list) and all(_coverage_shape(row) for row in coverage_rows)
    if coverage_interpretable:
        for t, expected in expected_coverage.items():
            actual = coverage_map.get(t)
            if actual is not None and _coverage_valid(actual) and actual != expected:
                fail("coverage", f"{name}.coverage[{t}]: expected {expected}, actual={actual}")
        if set(coverage_map) != set(expected_coverage):
            fail("coverage", f"{name}.coverage: all slots required")

    return {
        "free_cells": _free_cells(name), "final_coverage": len(global_free) / max(1, len(_free_cells(name))),
        "freshness": freshness_sum / freshness_den if freshness_den else 0.0,
        "tx": tx_count, "switches": switch_count,
    }, checks, details


def _replay_atoms(indexed: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    aggregate = {k.removeprefix("replay_validity."): True for k in ATOM_SHARES["replay_validity"]}
    details = {k: "" for k in aggregate}
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIO_NAMES:
        if name not in indexed:
            for rule in aggregate:
                aggregate[rule] = False
                details[rule] = "all canonical scenarios are required"
            continue
        result, checks, local_details = _replay_scenario(name, indexed[name])
        if result is not None:
            replayed[name] = result
        for rule, ok in checks.items():
            if not ok:
                aggregate[rule] = False
                if not details[rule]:
                    details[rule] = local_details[rule]
    return [
        _atom("replay_validity", rule, aggregate[rule], details[rule] or f"{rule}: replay matched")
        for rule in [k.removeprefix("replay_validity.") for k in ATOM_SHARES["replay_validity"]]
    ], replayed


def _replay_diagnostic(indexed: Mapping[str, dict[str, Any]], atoms: Sequence[AtomVerdict]) -> str:
    """Explain replay deferrals without charging malformed evidence twice."""
    notes: list[str] = []
    expected_robot_keys = {(r, t) for r in range(N_ROBOTS) for t in range(HORIZON)}
    for name in SCENARIO_NAMES:
        if name not in indexed:
            notes.append(f"{name}.missing_canonical_scenario")
            continue
        item = indexed[name]
        robots = item.get("robots")
        if not isinstance(robots, dict) or set(robots) != expected_robot_keys:
            notes.append(f"{name}.robots_owned_by_canonical_or_evidence")
            continue
        bad_fields: set[str] = set()
        for row in robots.values():
            if not isinstance(row, dict):
                bad_fields.add("robot_row")
                continue
            if not _pair_int(row.get("position")) or not all(0 <= v < 10 for v in row.get("position", [])):
                bad_fields.add("position")
            if not (isinstance(row.get("role"), str) and row.get("role") in {"explorer", "messenger"}):
                bad_fields.add("role")
            if not (isinstance(row.get("move"), str) and row.get("move") in MOVES):
                bad_fields.add("move")
            if not (isinstance(row.get("observed_facts"), list)
                    and all(_fact_valid(f) for f in row.get("observed_facts", []))
                    and [f["cell"] for f in row.get("observed_facts", [])] == sorted({f["cell"] for f in row.get("observed_facts", [])})):
                bad_fields.add("observed_facts")
            received = row.get("received_event_ids")
            if not (isinstance(received, list) and all(isinstance(v, str) for v in received) and received == sorted(set(received))):
                bad_fields.add("received_event_ids")
        notes.extend(f"{name}.{field}_owned_by_evidence_or_legality" for field in sorted(bad_fields))
        event_rows = item.get("event_rows")
        if not (isinstance(event_rows, list) and all(_event_shape(row) for row in event_rows)):
            notes.append(f"{name}.events_owned_by_evidence")
        sync_rows = item.get("sync_rows")
        if not (isinstance(sync_rows, list) and all(_sync_shape(row) for row in sync_rows)):
            notes.append(f"{name}.sync_owned_by_evidence")
        coverage_rows = item.get("coverage_rows")
        if not (isinstance(coverage_rows, list) and all(_coverage_shape(row) for row in coverage_rows)):
            notes.append(f"{name}.coverage_owned_by_evidence_or_canonical")
    unique = list(dict.fromkeys(notes))
    diagnostic = _diagnose(atoms)
    if unique:
        return f"{diagnostic}; not_evaluated_due_to=[{','.join(unique[:24])}]"
    return diagnostic


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if not set(SCENARIO_NAMES).issubset(replayed):
        raise ValueError("metrics require a replay result for every canonical scenario")
    coverage = sum(replayed[n]["final_coverage"] for n in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    freshness = sum(replayed[n]["freshness"] for n in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    communication = sum(max(0.0, min(1.0, 1.0 - replayed[n]["tx"] / 80.0)) for n in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    stability = sum(max(0.0, min(1.0, 1.0 - replayed[n]["switches"] / 32.0)) for n in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    metrics = {
        "primary_metrics_1": coverage, "primary_metrics_2": freshness,
        "auxiliary_metrics_1": communication, "auxiliary_metrics_2": stability,
        "normalized_task_performance": (8 * coverage + 7 * freshness + 3 * communication + 2 * stability) / 20,
    }
    if not all(_finite(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _headings(text: str) -> dict[str, str]:
    required = {
        "Problem Formulation", "Method Design", "Experimental Setup",
        "Results", "Research Analysis", "Conclusion and Limitations",
    }
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:\x60{3,}|~{3,})", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^\s{0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        if match:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2).strip())
            headings.append((index, len(match.group(1)), title))

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


def _report_score(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a regular non-symlink file"
    try:
        text = (Path(ws.root) / "report.md").read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return 0.0, f"report.md: strict UTF-8 read failed: {exc}"
    headings = _headings(text or "")
    missing = sorted(k for k in {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"} if not headings.get(k))
    score = max(0.0, (2.0 - 0.5 * len(missing)) / 2.0)
    return score, f"missing_sections={missing}; normalized={score:.9g}" if missing else "missing_sections=[]"


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
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
            result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(result, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
            state.official_message = f"could not remove previous results.json: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr_text = str(getattr(result, "stderr", ""))
            state.timed_out = state.exit_code is None and (
                "timeout" in stderr_text.lower() or "timed out" in stderr_text.lower()
            )
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        if state.exit_code is None and state.elapsed_s >= 600.0 - EPS:
            state.timed_out = True
        state.run_finished = True
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
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
                f"timed_out={state.timed_out}, "
                f"new_regular_results={state.result_file_is_new_regular}"
            )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json_bytes((Path(ws.root) / "results.json").read_bytes())
                state.parse_ok = True
            except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
                state.diagnostics["result_json"] = f"results.json: strict UTF-8/JSON parse failed: {exc}"
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose(result_atoms))
        if not state.official_ok or not state.parse_ok:
            blocked = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
            reason = (
                f"blocked_checks=[{','.join(blocked)}]; official_ok={state.official_ok}; "
                f"parse_ok={state.parse_ok}; official_message={state.official_message}"
            )
            for name in blocked:
                state.scores[name] = 0.0
                state.diagnostics[name] = f"{reason}; {name}=not_evaluated"
            return state
        state.scenarios = _index_records(state.data)
        state.records = state.scenarios
        evidence = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        cov, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = cov
        state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:20]}"
        legality = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose(legality)
        replay, state.replayed = _replay_atoms(state.scenarios)
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _replay_diagnostic(state.scenarios, replay)
        gate = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        state.validity_gate = state.official_ok and all(state.scores.get(n, 0.0) >= 1.0 - EPS for n in gate)
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001
        state.official_ok = False
        state.run_finished = True
        state.metrics = None
        state.validity_gate = False
        state.official_message = state.official_message or (
            f"grader state construction failed: {type(exc).__name__}: {exc}"
        )
        failed = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        detail = (
            f"grader_exception={type(exc).__name__}: {exc}; "
            f"affected_checks=[{','.join(failed)}]; validity_gate=False"
        )
        for name in failed:
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


async def _metric_check(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        gate = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
        failed = ([] if state.official_ok else ["official_run"])
        failed.extend(item for item in gate if state.scores.get(item, 0.0) < 1.0 - EPS)
        return (
            0.0,
            f"{name}: validity gate not satisfied; metric not computed; "
            f"failed_gate=[{','.join(failed)}]; "
            f"official_ok={state.official_ok}; "
            + ",".join(f"{item}={state.scores.get(item, 0.0):.9g}" for item in gate),
        )
    value = state.metrics.get(name, 0.0)
    return value, f"{name}: independently replayed value={value:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_2")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_2")
