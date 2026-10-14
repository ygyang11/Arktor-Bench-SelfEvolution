# ruff: noqa: E501
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 96
STAGE_LENGTH = 32
STAGES = 3
ROBOTS = 4
GRID_X = 24
GRID_Y = 18
SENSOR_RADIUS = 2
ENERGY0 = 100.0
PLAN_COST = 1.0
MOVE_COST = 1.0
STAY_COST = 0.2
UNIT_COST = 0.5
RENDEZVOUS_CAPACITY = 8
UNIT_BUDGET = 20
MAX_SUMMARY_CELLS = 4
EPS = 1e-6

SCENARIO_NAMES = (
    "long_corridor",
    "branching_frontiers",
    "rendezvous_bottleneck",
    "late_frontier",
    "range_shift",
)
REGION_LOWER_LEFT = ((2, 2), (2, 13), (8, 2), (8, 13), (16, 2), (16, 13))
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
SCENARIOS: dict[str, dict[str, Any]] = {
    "long_corridor": {
        "starts": ((1, 1), (1, 16), (22, 1), (22, 16)),
        "obstacles": ((11, 12, 0, 6), (11, 12, 11, 17)),
        "regions": ((1, 28), (1, 34), (2, 44), (2, 52), (3, 72), (4, 88)),
        "points": ((6, 9), (12, 9), (18, 9)),
        "ranges": (5, 5, 5),
    },
    "branching_frontiers": {
        "starts": ((1, 2), (1, 15), (22, 2), (22, 15)),
        "obstacles": ((5, 8, 7, 8), (15, 18, 9, 10)),
        "regions": ((2, 24), (1, 36), (3, 48), (2, 60), (4, 76), (3, 92)),
        "points": ((5, 5), (12, 9), (19, 12)),
        "ranges": (5, 4, 4),
    },
    "rendezvous_bottleneck": {
        "starts": ((2, 1), (2, 16), (21, 1), (21, 16)),
        "obstacles": ((10, 13, 6, 11),),
        "regions": ((1, 26), (2, 38), (2, 50), (3, 62), (4, 78), (4, 94)),
        "points": ((8, 5), (12, 5), (16, 12)),
        "ranges": (4, 4, 3),
    },
    "late_frontier": {
        "starts": ((1, 1), (1, 16), (22, 1), (22, 16)),
        "obstacles": ((6, 7, 3, 14), (18, 19, 4, 13)),
        "regions": ((3, 20), (3, 28), (2, 44), (2, 56), (4, 72), (4, 90)),
        "points": ((4, 9), (12, 3), (20, 9)),
        "ranges": (5, 5, 5),
    },
    "range_shift": {
        "starts": ((2, 2), (2, 15), (21, 2), (21, 15)),
        "obstacles": ((10, 11, 4, 13),),
        "regions": ((1, 30), (2, 42), (2, 54), (3, 66), (4, 82), (3, 94)),
        "points": ((6, 9), (12, 5), (18, 9)),
        "ranges": (6, 4, 3),
    },
}

SCENARIO_KEYS = frozenset({"scenario", "proposals", "actions"})
PROPOSAL_KEYS = frozenset(
    {"stage", "robot", "proposal_slot", "members", "point_id", "window_start"}
)
ACTION_KEYS = frozenset({"t", "robot", "dx", "dy", "intent_cell", "summary_cells"})
OPTIONAL_ID_FIELDS = ("proposal_slot", "point_id", "window_start")
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
GATE_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.scenario_rows": 0.10,
        "result_evidence.proposal_rows": 0.25,
        "result_evidence.proposal_values": 0.20,
        "result_evidence.action_rows": 0.20,
        "result_evidence.action_values": 0.25,
    },
    "decision_legality": {
        "decision_legality.proposals": 0.20,
        "decision_legality.motion": 0.20,
        "decision_legality.intents": 0.20,
        "decision_legality.summaries": 0.20,
        "decision_legality.energy": 0.20,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.rendezvous": 0.20,
        "replay_validity.maps": 0.20,
        "replay_validity.resources": 0.20,
        "replay_validity.metrics": 0.20,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id = ident
        self.passed = bool(passed)
        self.detail = str(detail)
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
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_integer(value: Any) -> bool:
    return value is None or _integer(value)


def _null_id(value: Any) -> int | None:
    if value is None or value == -1:
        return None
    return int(value) if _integer(value) else None


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _brief(value: Any, limit: int = 140) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # huge/malformed values must remain diagnosable
        return f"<{type(value).__name__} repr failed: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _blocked(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    x, y = position
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in spec["obstacles"])


def _traversable(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    return (
        position[0] in range(GRID_X)
        and position[1] in range(GRID_Y)
        and not _blocked(spec, position)
    )


def _cell_id(position: tuple[int, int]) -> int:
    return GRID_X * position[1] + position[0]


def _cell_position(cell: int) -> tuple[int, int]:
    return cell % GRID_X, cell // GRID_X


def _region_cells(region: int) -> tuple[int, ...]:
    x0, y0 = REGION_LOWER_LEFT[region]
    return tuple(_cell_id((x, y)) for y in range(y0, y0 + 2) for x in range(x0, x0 + 2))


REGION_CELLS = tuple(_region_cells(region) for region in range(len(REGION_LOWER_LEFT)))
VALUABLE_CELLS = frozenset(cell for region in REGION_CELLS for cell in region)


def _cell_value(spec: Mapping[str, Any], cell: int) -> int:
    for region, cells in enumerate(REGION_CELLS):
        if cell in cells:
            return int(spec["regions"][region][0])
    return 1


def _shape_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    root_keys = root and set(data) == {"scenarios"}
    scenarios = data.get("scenarios") if root else None
    scenario_list = isinstance(scenarios, list)
    records = bool(
        scenario_list
        and all(
            isinstance(record, Mapping)
            and set(record) == SCENARIO_KEYS
            and isinstance(record.get("proposals"), list)
            and isinstance(record.get("actions"), list)
            for record in scenarios
        )
    )
    record_errors: list[str] = []
    if scenario_list:
        for index, record in enumerate(scenarios):
            if not isinstance(record, Mapping):
                record_errors.append(f"scenarios[{index}]={_brief(record)}; expected an object")
            elif set(record) != SCENARIO_KEYS:
                record_errors.append(
                    f"scenarios[{index}] keys={sorted(map(str, record))}; expected {sorted(SCENARIO_KEYS)}"
                )
            elif not isinstance(record.get("proposals"), list) or not isinstance(
                record.get("actions"), list
            ):
                record_errors.append(f"scenarios[{index}] proposals/actions must both be arrays")
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parse"),
        _atom(
            "result_json",
            "root",
            bool(root_keys),
            f"expected exact root keys ['scenarios']; got {_brief(data)}",
        ),
        _atom(
            "result_json",
            "scenarios",
            scenario_list,
            f"scenarios={_brief(scenarios)}; expected a JSON array",
        ),
        _atom(
            "result_json",
            "records",
            records,
            "passed"
            if records
            else " | ".join(record_errors[:8])
            or "scenario records unavailable because scenarios is not an array",
        ),
    ]


def _index(data: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    rows = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(rows, list):
        return result
    for record in rows:
        if not isinstance(record, Mapping):
            continue
        name = record.get("scenario")
        if isinstance(name, str) and name not in result:
            result[name] = record
    return result


def _proposal_rows(record: Mapping[str, Any]) -> list[Any]:
    rows = record.get("proposals")
    return rows if isinstance(rows, list) else []


def _action_rows(record: Mapping[str, Any]) -> list[Any]:
    rows = record.get("actions")
    return rows if isinstance(rows, list) else []


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    raw = data.get("scenarios") if isinstance(data, Mapping) else None
    records = raw if isinstance(raw, list) else []
    scenario_errors = [
        f"scenarios[{index}].scenario={_brief(record.get('scenario') if isinstance(record, Mapping) else None)}; expected string"
        for index, record in enumerate(records)
        if not isinstance(record, Mapping) or not isinstance(record.get("scenario"), str)
    ]
    scenario_rows = not scenario_errors
    proposals = [
        row for record in records if isinstance(record, Mapping) for row in _proposal_rows(record)
    ]
    proposal_row_errors: list[str] = []
    proposal_value_errors: list[str] = []
    action_row_errors: list[str] = []
    action_value_errors: list[str] = []
    for scenario_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        for row_index, row in enumerate(_proposal_rows(record)):
            path = f"scenarios[{scenario_index}].proposals[{row_index}]"
            if not isinstance(row, Mapping):
                proposal_row_errors.append(f"{path}={_brief(row)}; expected object")
            elif set(row) != PROPOSAL_KEYS:
                proposal_row_errors.append(
                    f"{path} keys={sorted(map(str, row))}; expected {sorted(PROPOSAL_KEYS)}"
                )
            elif not (
                all(_integer(row.get(field)) for field in ("stage", "robot"))
                and all(_optional_integer(row.get(field)) for field in OPTIONAL_ID_FIELDS)
                and isinstance(row.get("members"), list)
                and all(_integer(member) for member in row.get("members", []))
            ):
                proposal_value_errors.append(
                    f"{path}={_brief(row)}; expected integer keys, integer/null optional IDs, and integer member array"
                )
        for row_index, row in enumerate(_action_rows(record)):
            path = f"scenarios[{scenario_index}].actions[{row_index}]"
            if not isinstance(row, Mapping):
                action_row_errors.append(f"{path}={_brief(row)}; expected object")
            elif set(row) != ACTION_KEYS:
                action_row_errors.append(
                    f"{path} keys={sorted(map(str, row))}; expected {sorted(ACTION_KEYS)}"
                )
            elif not (
                all(_integer(row.get(field)) for field in ("t", "robot", "dx", "dy"))
                and _optional_integer(row.get("intent_cell"))
                and isinstance(row.get("summary_cells"), list)
                and all(_integer(cell) for cell in row.get("summary_cells", []))
            ):
                action_value_errors.append(
                    f"{path}={_brief(row)}; expected integer keys/move, integer/null intent, and integer summary array"
                )
    proposal_rows = not proposal_row_errors
    typed_proposal_rows = [
        row for row in proposals if isinstance(row, Mapping) and set(row) == PROPOSAL_KEYS
    ]
    proposal_values = not proposal_value_errors and all(
        all(_integer(row.get(field)) for field in ("stage", "robot"))
        and all(_optional_integer(row.get(field)) for field in OPTIONAL_ID_FIELDS)
        and isinstance(row.get("members"), list)
        and all(_integer(member) for member in row.get("members", []))
        for row in typed_proposal_rows
    )
    actions = [
        row for record in records if isinstance(record, Mapping) for row in _action_rows(record)
    ]
    action_rows = not action_row_errors
    typed_action_rows = [
        row for row in actions if isinstance(row, Mapping) and set(row) == ACTION_KEYS
    ]
    action_values = not action_value_errors and all(
        all(_integer(row.get(field)) for field in ("t", "robot", "dx", "dy"))
        and _optional_integer(row.get("intent_cell"))
        and isinstance(row.get("summary_cells"), list)
        and all(_integer(cell) for cell in row.get("summary_cells", []))
        for row in typed_action_rows
    )
    return [
        _atom(
            "result_evidence",
            "scenario_rows",
            scenario_rows,
            "passed" if scenario_rows else " | ".join(scenario_errors[:8]),
        ),
        _atom(
            "result_evidence",
            "proposal_rows",
            proposal_rows,
            "passed" if proposal_rows else " | ".join(proposal_row_errors[:8]),
        ),
        _atom(
            "result_evidence",
            "proposal_values",
            proposal_values,
            "passed" if proposal_values else " | ".join(proposal_value_errors[:8]),
        ),
        _atom(
            "result_evidence",
            "action_rows",
            action_rows,
            "passed" if action_rows else " | ".join(action_row_errors[:8]),
        ),
        _atom(
            "result_evidence",
            "action_values",
            action_values,
            "passed" if action_values else " | ".join(action_value_errors[:8]),
        ),
    ]


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    raw = data.get("scenarios") if isinstance(data, Mapping) else None
    rows = raw if isinstance(raw, list) else []
    scenario_counts: dict[str, int] = {}
    for index, record in enumerate(rows):
        if not isinstance(record, Mapping) or not isinstance(record.get("scenario"), str):
            defects.append(f"scenarios[{index}] has no string canonical key")
            continue
        name = record["scenario"]
        scenario_counts[name] = scenario_counts.get(name, 0) + 1
        if name not in SCENARIO_NAMES:
            defects.append(f"unknown scenario key {name!r}")
    for name in SCENARIO_NAMES:
        count = scenario_counts.get(name, 0)
        if count == 0:
            defects.append(f"missing scenario {name}")
        elif count > 1:
            defects.extend([f"duplicate scenario {name}"] * (count - 1))

    indexed = _index(data)
    for name in SCENARIO_NAMES:
        record = indexed.get(name)
        if record is None:
            continue
        proposal_counts: dict[tuple[int, int], int] = {}
        for index, row in enumerate(_proposal_rows(record)):
            if (
                not isinstance(row, Mapping)
                or not _integer(row.get("stage"))
                or not _integer(row.get("robot"))
            ):
                defects.append(f"{name}.proposals[{index}] has no integer (stage,robot) key")
                continue
            key = (row["stage"], row["robot"])
            proposal_counts[key] = proposal_counts.get(key, 0) + 1
            if key[0] not in range(STAGES) or key[1] not in range(ROBOTS):
                defects.append(f"{name} has unknown proposal key {_brief(key)}")
        for stage in range(STAGES):
            for robot in range(ROBOTS):
                count = proposal_counts.get((stage, robot), 0)
                if count == 0:
                    defects.append(f"{name} missing proposal {(stage, robot)}")
                elif count > 1:
                    defects.extend([f"{name} duplicate proposal {(stage, robot)}"] * (count - 1))

        action_counts: dict[tuple[int, int], int] = {}
        for index, row in enumerate(_action_rows(record)):
            if (
                not isinstance(row, Mapping)
                or not _integer(row.get("t"))
                or not _integer(row.get("robot"))
            ):
                defects.append(f"{name}.actions[{index}] has no integer (t,robot) key")
                continue
            key = (row["t"], row["robot"])
            action_counts[key] = action_counts.get(key, 0) + 1
            if key[0] not in range(HORIZON) or key[1] not in range(ROBOTS):
                defects.append(f"{name} has unknown action key {_brief(key)}")
        for t in range(HORIZON):
            for robot in range(ROBOTS):
                count = action_counts.get((t, robot), 0)
                if count == 0:
                    defects.append(f"{name} missing action {(t, robot)}")
                elif count > 1:
                    defects.extend([f"{name} duplicate action {(t, robot)}"] * (count - 1))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _proposal_index(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in _proposal_rows(record):
        if isinstance(row, Mapping) and _integer(row.get("stage")) and _integer(row.get("robot")):
            result.setdefault((row["stage"], row["robot"]), row)
    return result


def _action_index(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in _action_rows(record):
        if isinstance(row, Mapping) and _integer(row.get("t")) and _integer(row.get("robot")):
            result.setdefault((row["t"], row["robot"]), row)
    return result


def _blank_proposal(stage: int, robot: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "robot": robot,
        "proposal_slot": None,
        "members": [],
        "point_id": None,
        "window_start": None,
    }


def _blank_action(t: int, robot: int) -> dict[str, Any]:
    return {
        "t": t,
        "robot": robot,
        "dx": 0,
        "dy": 0,
        "intent_cell": None,
        "summary_cells": [],
    }


def _typed_proposal(row: Mapping[str, Any]) -> bool:
    return (
        set(row) == PROPOSAL_KEYS
        and all(_integer(row.get(field)) for field in ("stage", "robot"))
        and all(_optional_integer(row.get(field)) for field in OPTIONAL_ID_FIELDS)
        and isinstance(row.get("members"), list)
        and all(_integer(member) for member in row["members"])
    )


def _typed_action(row: Mapping[str, Any]) -> bool:
    return (
        set(row) == ACTION_KEYS
        and all(_integer(row.get(field)) for field in ("t", "robot", "dx", "dy"))
        and _optional_integer(row.get("intent_cell"))
        and isinstance(row.get("summary_cells"), list)
        and all(_integer(cell) for cell in row["summary_cells"])
    )


def _observe(
    spec: Mapping[str, Any], local_map: dict[int, int], position: tuple[int, int], timestamp: int
) -> None:
    for x in range(
        max(0, position[0] - SENSOR_RADIUS), min(GRID_X, position[0] + SENSOR_RADIUS + 1)
    ):
        for y in range(
            max(0, position[1] - SENSOR_RADIUS), min(GRID_Y, position[1] + SENSOR_RADIUS + 1)
        ):
            cell_position = (x, y)
            if abs(x - position[0]) + abs(y - position[1]) <= SENSOR_RADIUS and _traversable(
                spec, cell_position
            ):
                local_map[_cell_id(cell_position)] = timestamp


def _is_frontier(spec: Mapping[str, Any], local_map: Mapping[int, int], cell: int) -> bool:
    if cell not in range(GRID_X * GRID_Y):
        return False
    position = _cell_position(cell)
    if not _traversable(spec, position) or cell in local_map:
        return False
    return any(
        _traversable(spec, (position[0] + dx, position[1] + dy))
        and _cell_id((position[0] + dx, position[1] + dy)) in local_map
        for dx, dy in MOVES[1:]
    )


def _normalize_proposals(
    name: str,
    proposal_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    errors: dict[str, list[str]],
) -> dict[int, dict[str, Any] | None]:
    submitted: dict[tuple[int, int], tuple[tuple[int, ...], int, int, int]] = {}
    for stage in range(STAGES):
        for robot in range(ROBOTS):
            row = proposal_rows.get((stage, robot), _blank_proposal(stage, robot))
            if not _typed_proposal(row):
                continue
            raw_slot, raw_point, raw_window = (row[field] for field in OPTIONAL_ID_FIELDS)
            members = row["members"]
            no_proposal = (
                raw_slot in (None, -1)
                and raw_point in (None, -1)
                and raw_window in (None, -1)
                and members == []
            )
            if no_proposal:
                continue
            if any(value in (None, -1) for value in (raw_slot, raw_point, raw_window)):
                errors["proposals"].append(
                    f"{name}.proposals[{stage},{robot}] must be either fully null/empty or a complete proposal"
                )
                continue
            proposal_slot = int(raw_slot)
            point_id = int(raw_point)
            window_start = int(raw_window)
            members_valid = (
                2 <= len(members) <= ROBOTS
                and len(set(members)) == len(members)
                and all(member in range(ROBOTS) for member in members)
                and robot in members
            )
            ranges_valid = (
                stage * STAGE_LENGTH <= proposal_slot <= stage * STAGE_LENGTH + 3
                and point_id in range(3)
                and stage * STAGE_LENGTH + 4 <= window_start <= stage * STAGE_LENGTH + 29
            )
            if not members_valid or not ranges_valid:
                errors["proposals"].append(
                    f"{name}.proposals[{stage},{robot}] has invalid members={_brief(members)}, proposal_slot={_brief(proposal_slot)}, point_id={_brief(point_id)}, window_start={_brief(window_start)}"
                )
                continue
            submitted[(stage, robot)] = (
                tuple(sorted(members)),
                point_id,
                window_start,
                proposal_slot,
            )

    selected: dict[int, dict[str, Any] | None] = {}
    for stage in range(STAGES):
        tuples = {
            (members, point_id, window_start)
            for (candidate_stage, _robot), (
                members,
                point_id,
                window_start,
                _slot,
            ) in submitted.items()
            if candidate_stage == stage
        }
        complete: list[dict[str, Any]] = []
        for members, point_id, window_start in tuples:
            slots: list[int] = []
            for member in members:
                proposal = submitted.get((stage, member))
                if proposal is None or proposal[:3] != (members, point_id, window_start):
                    break
                slots.append(proposal[3])
            else:
                complete.append(
                    {
                        "stage": stage,
                        "members": members,
                        "point_id": point_id,
                        "window_start": window_start,
                        "latest_proposal": max(slots),
                    }
                )
        selected[stage] = (
            min(
                complete,
                key=lambda plan: (
                    -len(plan["members"]),
                    plan["latest_proposal"],
                    plan["members"],
                    plan["point_id"],
                    plan["window_start"],
                ),
            )
            if complete
            else None
        )
    return selected


def _replay(name: str, scenario: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    proposals = _proposal_index(scenario)
    actions = _action_index(scenario)
    errors: dict[str, list[str]] = {
        key: [] for key in ("proposals", "motion", "intents", "summaries", "energy")
    }
    selected = _normalize_proposals(name, proposals, errors)

    positions = [tuple(position) for position in spec["starts"]]
    energies = [ENERGY0] * ROBOTS
    local_maps: list[dict[int, int]] = [dict() for _ in range(ROBOTS)]
    received_maps: list[dict[int, int]] = [dict() for _ in range(ROBOTS)]
    for robot in range(ROBOTS):
        _observe(spec, local_maps[robot], positions[robot], 0)

    position_history: list[tuple[tuple[int, int], ...]] = [tuple(positions)]
    energy_history: list[tuple[float, ...]] = [tuple(energies)]
    submitted_moves: dict[tuple[int, int], tuple[int, int]] = {}
    plans: dict[int, dict[str, Any] | None] = {stage: None for stage in range(STAGES)}
    accepted_units: list[dict[str, Any]] = []
    accepted_count = 0
    successful_rendezvous = 0
    safety_events = 0
    wait_slots = 0
    region_completions: dict[int, int] = {}
    freshness_sum = 0.0

    def update_region_completions(boundary: int) -> None:
        for region, cells in enumerate(REGION_CELLS):
            if (
                region not in region_completions
                and sum(all(cell in local_maps[robot] for cell in cells) for robot in range(ROBOTS))
                >= 3
            ):
                region_completions[region] = boundary

    update_region_completions(0)

    for t in range(HORIZON):
        stage = t // STAGE_LENGTH
        if t == stage * STAGE_LENGTH + 4:
            candidate = selected[stage]
            if candidate is not None:
                plan = dict(candidate)
                plan["status"] = "pending"
                plan["success_boundary"] = None
                plans[stage] = plan
                for robot in plan["members"]:
                    energies[robot] -= PLAN_COST
                    if energies[robot] < -EPS:
                        errors["energy"].append(
                            f"{name} robot {robot} is negative after stage-{stage} plan adoption: {energies[robot]:.6g}"
                        )

        plan = plans[stage]
        slot_rows: dict[int, Mapping[str, Any]] = {}
        slot_summaries: dict[int, tuple[int, ...]] = {}
        action_start_timestamps: dict[tuple[int, int], int] = {}
        for robot in range(ROBOTS):
            row = actions.get((t, robot), _blank_action(t, robot))
            if not _typed_action(row):
                row = _blank_action(t, robot)
            slot_rows[robot] = row

            move = (row["dx"], row["dy"])
            if move not in MOVES:
                errors["motion"].append(
                    f"{name}.actions[{t},{robot}] move {_brief(move)} is outside the five-move domain"
                )
                move = (0, 0)
            submitted_moves[(t, robot)] = move

            raw_intent = row["intent_cell"]
            intent = _null_id(raw_intent)
            if intent is not None and not _is_frontier(spec, local_maps[robot], intent):
                errors["intents"].append(
                    f"{name}.actions[{t},{robot}].intent_cell={_brief(intent)} is not an action-start local frontier"
                )

            summaries = tuple(row["summary_cells"])
            valid_summary = True
            if len(summaries) > MAX_SUMMARY_CELLS or len(set(summaries)) != len(summaries):
                errors["summaries"].append(
                    f"{name}.actions[{t},{robot}].summary_cells must contain at most four distinct cells"
                )
                valid_summary = False
            allowed = bool(
                plan
                and plan["status"] == "pending"
                and robot in plan["members"]
                and plan["window_start"] <= t < plan["window_start"] + 3
            )
            if summaries and not allowed:
                errors["summaries"].append(
                    f"{name}.actions[{t},{robot}] submits summaries without a pending member window"
                )
                valid_summary = False
            unknown = [cell for cell in summaries if cell not in local_maps[robot]]
            if unknown:
                errors["summaries"].append(
                    f"{name}.actions[{t},{robot}] summary cells are not in the action-start local map: {_brief(unknown[:8])}"
                )
                valid_summary = False
            if valid_summary:
                slot_summaries[robot] = summaries
                for cell in summaries:
                    action_start_timestamps[(robot, cell)] = local_maps[robot][cell]

            if (
                plan
                and plan["status"] == "pending"
                and robot in plan["members"]
                and plan["window_start"] <= t < plan["window_start"] + 3
                and abs(positions[robot][0] - spec["points"][plan["point_id"]][0])
                + abs(positions[robot][1] - spec["points"][plan["point_id"]][1])
                <= 2
                and move == (0, 0)
            ):
                wait_slots += 1

        accepted_destinations: list[tuple[int, int]] = []
        next_positions: list[tuple[int, int]] = []
        for robot in range(ROBOTS):
            current = positions[robot]
            move = submitted_moves[(t, robot)]
            destination = current[0] + move[0], current[1] + move[1]
            rejected = bool(
                not _traversable(spec, destination)
                or destination in accepted_destinations
                or destination in positions[robot + 1 :]
            )
            if rejected:
                destination = current
                if move != (0, 0):
                    safety_events += 1
            accepted_destinations.append(destination)
            next_positions.append(destination)

        for robot in range(ROBOTS):
            moved = next_positions[robot] != positions[robot]
            energies[robot] -= MOVE_COST if moved else STAY_COST
            if energies[robot] < -EPS:
                errors["energy"].append(
                    f"{name}.actions[{t},{robot}] executed motion makes energy negative: {energies[robot]:.6g}"
                )
        positions = next_positions
        for robot in range(ROBOTS):
            _observe(spec, local_maps[robot], positions[robot], t + 1)

        if (
            plan
            and plan["status"] == "pending"
            and plan["window_start"] <= t < plan["window_start"] + 3
        ):
            members = plan["members"]
            point = spec["points"][plan["point_id"]]
            at_point = all(
                abs(positions[robot][0] - point[0]) + abs(positions[robot][1] - point[1]) <= 2
                for robot in members
            )
            max_pairwise = max(
                (
                    abs(positions[first][0] - positions[second][0])
                    + abs(positions[first][1] - positions[second][1])
                    for index, first in enumerate(members)
                    for second in members[index + 1 :]
                ),
                default=0,
            )
            if at_point and max_pairwise <= spec["ranges"][stage]:
                plan["status"] = "success"
                plan["success_boundary"] = t + 1
                successful_rendezvous += 1
                units = [
                    {
                        "stage": stage,
                        "slot": t,
                        "source": source,
                        "cell": cell,
                        "timestamp": action_start_timestamps[(source, cell)],
                        "useful": False,
                    }
                    for source in members
                    for cell in slot_summaries.get(source, ())
                ]
                units.sort(key=lambda unit: (-unit["timestamp"], unit["source"], unit["cell"]))
                admitted = units[: min(RENDEZVOUS_CAPACITY, max(0, UNIT_BUDGET - accepted_count))]
                for unit in admitted:
                    accepted_count += 1
                    source = unit["source"]
                    timestamp = unit["timestamp"]
                    useful = False
                    for recipient in members:
                        if recipient == source:
                            continue
                        received_maps[recipient][unit["cell"]] = max(
                            timestamp, received_maps[recipient].get(unit["cell"], -1)
                        )
                        if timestamp > local_maps[recipient].get(unit["cell"], -1):
                            local_maps[recipient][unit["cell"]] = timestamp
                            useful = True
                    unit["useful"] = useful
                    energies[source] -= UNIT_COST
                    if energies[source] < -EPS:
                        errors["energy"].append(
                            f"{name}.actions[{t},{source}] admitted summaries make energy negative: {energies[source]:.6g}"
                        )
                    accepted_units.append(unit)
            elif t == plan["window_start"] + 2:
                plan["status"] = "cancelled"

        update_region_completions(t + 1)
        for robot in range(ROBOTS):
            for cell in VALUABLE_CELLS:
                timestamp = received_maps[robot].get(cell)
                if timestamp is not None:
                    freshness_sum += _clip(1.0 - (t + 1 - timestamp) / STAGE_LENGTH)
        position_history.append(tuple(positions))
        energy_history.append(tuple(energies))

    final_coverage = set().union(*(set(local_map) for local_map in local_maps))
    return {
        "name": name,
        "positions": tuple(position_history),
        "energies": tuple(energy_history),
        "local_maps": tuple(dict(local_map) for local_map in local_maps),
        "received_maps": tuple(dict(local_map) for local_map in received_maps),
        "submitted_moves": submitted_moves,
        "plans": plans,
        "selected_proposals": selected,
        "accepted_units": tuple(accepted_units),
        "accepted_count": accepted_count,
        "successful_rendezvous": successful_rendezvous,
        "safety_events": safety_events,
        "wait_slots": wait_slots,
        "region_completions": region_completions,
        "final_coverage": frozenset(final_coverage),
        "freshness_sum": freshness_sum,
        "energy_used": ROBOTS * ENERGY0 - sum(energies),
        "errors": errors,
    }


def _legality(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {
        key: [] for key in ("proposals", "motion", "intents", "summaries", "energy")
    }
    for name in SCENARIO_NAMES:
        record = scenarios.get(name)
        if not isinstance(record, Mapping):
            continue
        replayed = _replay(name, record)
        for key, details in replayed["errors"].items():
            failures[key].extend(details)

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom("decision_legality", "proposals", not failures["proposals"], detail("proposals")),
        _atom("decision_legality", "motion", not failures["motion"], detail("motion")),
        _atom("decision_legality", "intents", not failures["intents"], detail("intents")),
        _atom("decision_legality", "summaries", not failures["summaries"], detail("summaries")),
        _atom("decision_legality", "energy", not failures["energy"], detail("energy")),
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name in SCENARIO_NAMES:
        state = replayed[name]
        spec = SCENARIOS[name]
        traversable_cells = [
            _cell_id((x, y))
            for y in range(GRID_Y)
            for x in range(GRID_X)
            if _traversable(spec, (x, y))
        ]
        maximum_value = sum(_cell_value(spec, cell) for cell in traversable_cells)
        covered_value = sum(_cell_value(spec, cell) for cell in state["final_coverage"])
        totals["primary_metrics_1"] += _clip(covered_value / maximum_value)

        quality_numerator = 0.0
        quality_denominator = 0.0
        for region, (value, deadline) in enumerate(spec["regions"]):
            completion = state["region_completions"].get(region)
            if completion is None:
                quality = 0.0
            elif completion <= deadline:
                quality = 1.0
            else:
                quality = 0.5 * _clip(1.0 - (completion - deadline) / max(1, HORIZON - deadline))
            quality_numerator += value * quality
            quality_denominator += value
        totals["primary_metrics_2"] += _clip(quality_numerator / quality_denominator)

        efficiency = (
            0.35 * _clip(1.0 - state["wait_slots"] / (HORIZON * ROBOTS))
            + 0.35 * _clip(1.0 - state["safety_events"] / (HORIZON * ROBOTS))
            + 0.30 * _clip(1.0 - state["energy_used"] / (ROBOTS * ENERGY0))
        )
        totals["primary_metrics_3"] += _clip(efficiency)

        accepted = state["accepted_count"]
        useful_units = [unit for unit in state["accepted_units"] if unit["useful"]]
        source_stage_coverage = len({(unit["source"], unit["stage"]) for unit in useful_units})
        communication = (
            0.0
            if accepted == 0
            else _clip(source_stage_coverage / (ROBOTS * STAGES))
            * (0.6 + 0.4 * _clip(len(useful_units) / accepted))
        )
        totals["auxiliary_metrics_1"] += _clip(communication)

        freshness = state["freshness_sum"] / (ROBOTS * len(VALUABLE_CELLS) * HORIZON)
        map_quality = 0.7 * _clip(freshness) + 0.3 * _clip(state["successful_rendezvous"] / STAGES)
        totals["auxiliary_metrics_2"] += _clip(map_quality)

    return {key: _clip(value / len(SCENARIO_NAMES)) for key, value in totals.items()}


def _replay_atoms(scenarios: Mapping[str, Any], replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    complete = set(replayed) == set(SCENARIO_NAMES)
    failures: dict[str, list[str]] = {
        key: [] for key in ("motion", "rendezvous", "maps", "resources", "metrics")
    }
    if not complete:
        missing = sorted(set(SCENARIO_NAMES) - set(replayed))
        unknown = sorted(set(replayed) - set(SCENARIO_NAMES))
        for key in failures:
            failures[key].append(
                f"replayed scenario keys missing={missing}, unknown={unknown}; expected {list(SCENARIO_NAMES)}"
            )

    for name in SCENARIO_NAMES:
        state = replayed.get(name)
        if not isinstance(state, Mapping):
            continue
        positions = state.get("positions", ())
        energies = state.get("energies", ())
        if len(positions) != HORIZON + 1:
            failures["motion"].append(
                f"{name}.positions length={len(positions)}; expected {HORIZON + 1} boundaries"
            )
        if len(energies) != HORIZON + 1:
            failures["motion"].append(
                f"{name}.energies length={len(energies)}; expected {HORIZON + 1} boundaries"
            )
        for boundary_index, boundary in enumerate(positions):
            if len(boundary) != ROBOTS or len(set(boundary)) != ROBOTS:
                failures["motion"].append(
                    f"{name}.positions[{boundary_index}]={_brief(boundary)}; expected {ROBOTS} distinct robot positions"
                )
                continue
            invalid = [
                position for position in boundary if not _traversable(SCENARIOS[name], position)
            ]
            if invalid:
                failures["motion"].append(
                    f"{name}.positions[{boundary_index}] has non-traversable positions {invalid}"
                )

        plans = state.get("plans", {})
        if not isinstance(plans, Mapping) or set(plans) != set(range(STAGES)):
            failures["rendezvous"].append(
                f"{name}.plans keys={_brief(list(plans) if isinstance(plans, Mapping) else plans)}; expected {list(range(STAGES))}"
            )
        else:
            for stage, plan in plans.items():
                if plan is None:
                    continue
                status = plan.get("status")
                success_boundary = plan.get("success_boundary")
                if status not in {"success", "cancelled"}:
                    failures["rendezvous"].append(
                        f"{name}.plans[{stage}].status={_brief(status)}; expected success or cancelled"
                    )
                elif status == "success" and not (
                    _integer(success_boundary) and 1 <= success_boundary <= HORIZON
                ):
                    failures["rendezvous"].append(
                        f"{name}.plans[{stage}].success_boundary={_brief(success_boundary)}; expected integer 1..{HORIZON}"
                    )
        successful = state.get("successful_rendezvous")
        if not _integer(successful) or successful not in range(STAGES + 1):
            failures["rendezvous"].append(
                f"{name}.successful_rendezvous={_brief(successful)}; expected integer 0..{STAGES}"
            )

        local_maps = state.get("local_maps", ())
        received_maps = state.get("received_maps", ())
        if len(local_maps) != ROBOTS or len(received_maps) != ROBOTS:
            failures["maps"].append(
                f"{name} local_maps/received_maps lengths={len(local_maps)}/{len(received_maps)}; expected {ROBOTS}/{ROBOTS}"
            )
        for collection_name, maps in (("local_maps", local_maps), ("received_maps", received_maps)):
            for robot, local_map in enumerate(maps):
                if not isinstance(local_map, Mapping):
                    failures["maps"].append(
                        f"{name}.{collection_name}[{robot}]={_brief(local_map)}; expected map"
                    )
                    continue
                invalid = [
                    (cell, timestamp)
                    for cell, timestamp in local_map.items()
                    if not _integer(cell)
                    or cell not in range(GRID_X * GRID_Y)
                    or not _traversable(SCENARIOS[name], _cell_position(cell))
                    or not _integer(timestamp)
                    or timestamp not in range(HORIZON + 1)
                ]
                if invalid:
                    failures["maps"].append(
                        f"{name}.{collection_name}[{robot}] invalid cell/timestamps={_brief(invalid)}; expected traversable cells and integer timestamps 0..{HORIZON}"
                    )
        completions = state.get("region_completions", {})
        invalid_completions = [
            (region, boundary)
            for region, boundary in completions.items()
            if not _integer(region)
            or region not in range(len(REGION_CELLS))
            or not _integer(boundary)
            or boundary not in range(HORIZON + 1)
        ]
        if invalid_completions:
            failures["maps"].append(
                f"{name}.region_completions invalid entries={_brief(invalid_completions)}; expected region 0..{len(REGION_CELLS) - 1}, boundary 0..{HORIZON}"
            )

        accepted = state.get("accepted_count")
        units = state.get("accepted_units", ())
        if not _integer(accepted) or accepted not in range(UNIT_BUDGET + 1):
            failures["resources"].append(
                f"{name}.accepted_count={_brief(accepted)}; expected integer 0..{UNIT_BUDGET}"
            )
        if not isinstance(units, Sequence) or len(units) != accepted:
            failures["resources"].append(
                f"{name}.accepted_units length={len(units) if isinstance(units, Sequence) else 'non-sequence'}; expected accepted_count={_brief(accepted)}"
            )
        if isinstance(units, Sequence):
            for stage in range(STAGES):
                stage_count = sum(
                    isinstance(unit, Mapping) and unit.get("stage") == stage for unit in units
                )
                if stage_count > RENDEZVOUS_CAPACITY:
                    failures["resources"].append(
                        f"{name}.accepted_units stage {stage} count={stage_count}; expected <= {RENDEZVOUS_CAPACITY}"
                    )
        for field, maximum in (
            ("wait_slots", HORIZON * ROBOTS),
            ("safety_events", HORIZON * ROBOTS),
        ):
            value = state.get(field)
            if not _integer(value) or value not in range(maximum + 1):
                failures["resources"].append(
                    f"{name}.{field}={_brief(value)}; expected integer 0..{maximum}"
                )
        if energies:
            for robot, energy in enumerate(energies[-1]):
                if (
                    not isinstance(energy, (int, float))
                    or not math.isfinite(energy)
                    or energy < -EPS
                ):
                    failures["resources"].append(
                        f"{name}.energies[{HORIZON}][{robot}]={_brief(energy)}; expected finite >= {-EPS} (tolerance {EPS})"
                    )

    metric_values: dict[str, float] = {}
    if complete:
        try:
            metric_values = _metrics(replayed)
        except Exception as exc:
            failures["metrics"].append(
                f"metric replay raised {type(exc).__name__}: {exc}; expected five finite values in [0,1]"
            )
    if set(metric_values) != set(METRIC_KEYS):
        failures["metrics"].append(
            f"metric keys={sorted(metric_values)}; expected {list(METRIC_KEYS)}"
        )
    invalid_metrics = [
        (key, value)
        for key, value in metric_values.items()
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
    ]
    if invalid_metrics:
        failures["metrics"].append(
            f"invalid metrics={_brief(invalid_metrics)}; expected finite values in [0,1]"
        )

    def detail(key: str) -> str:
        return "passed" if not failures[key] else " | ".join(failures[key][:8])

    return [
        _atom(
            "replay_validity",
            "motion",
            not failures["motion"],
            detail("motion"),
        ),
        _atom(
            "replay_validity",
            "rendezvous",
            not failures["rendezvous"],
            detail("rendezvous"),
        ),
        _atom(
            "replay_validity",
            "maps",
            not failures["maps"],
            detail("maps"),
        ),
        _atom(
            "replay_validity",
            "resources",
            not failures["resources"],
            detail("resources"),
        ),
        _atom(
            "replay_validity",
            "metrics",
            not failures["metrics"],
            detail("metrics"),
        ),
    ]


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _failed_atoms(group: str, detail: str) -> list[AtomVerdict]:
    return [
        _atom(group, key.removeprefix(f"{group}."), False, detail)
        for key in ATOM_SHARES[group]
    ]


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = False
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
            remove_ok = getattr(removed, "exit_code", None) == 0
        except TimeoutError:
            state.timed_out = True
            state.official_message = "pre-run results cleanup timed out after 30s"
        except Exception as exc:
            state.official_message = (
                f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        started = time.monotonic()
        try:
            result = (
                await ws.run("python run.py", check=False, timeout=600)
                if not state.official_message
                else None
            )
            state.exit_code = getattr(result, "exit_code", None)
            output = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".lower()
            state.timed_out = state.timed_out or (
                state.exit_code is None
                and any(word in output for word in ("timeout", "timed out"))
            )
        except TimeoutError as exc:
            state.timed_out = True
            state.official_message = f"official run timed out after 600s: {exc}"
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or any(
                word in str(exc).lower() for word in ("timeout", "timed out")
            )
        finally:
            state.elapsed_s = time.monotonic() - started
            state.run_finished = True

        result_path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = (
            remove_ok and result_path.is_file() and not result_path.is_symlink()
        )
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s is not None
            and state.elapsed_s <= 600 + EPS
            and state.result_file_is_new_regular
        )
        status = f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        state.official_message = (
            f"{state.official_message}; {status}" if state.official_message else status
        )

        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(result_path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"

        parse_failure = state.diagnostics.get("result_json", "")
        shape = _shape_atoms(state.data, state.parse_ok)
        state.scenarios = _index(state.data)
        try:
            evidence = _evidence_atoms(state.data)
        except Exception as exc:
            evidence = _failed_atoms(
                "result_evidence", f"evidence check raised {type(exc).__name__}: {exc}"
            )
        try:
            coverage_score, coverage_defects = _coverage(state.data)
        except Exception as exc:
            coverage_score, coverage_defects = 0.0, [
                f"coverage check raised {type(exc).__name__}: {exc}"
            ]
        try:
            legality = _legality(state.scenarios)
        except Exception as exc:
            legality = _failed_atoms(
                "decision_legality", f"legality check raised {type(exc).__name__}: {exc}"
            )
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.scores["canonical_coverage"] = coverage_score
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])

        can_replay = all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS
            for check_id in (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
            )
        )
        if can_replay:
            try:
                state.replayed = {
                    name: _replay(name, state.scenarios[name]) for name in SCENARIO_NAMES
                }
            except Exception as exc:
                state.replayed = {}
                state.diagnostics["replay_validity"] = (
                    f"replay raised {type(exc).__name__}: {exc}"
                )
        else:
            state.replayed = {}
        try:
            replay_atoms = _replay_atoms(state.scenarios, state.replayed)
        except Exception as exc:
            replay_atoms = _failed_atoms(
                "replay_validity", f"replay atom check raised {type(exc).__name__}: {exc}"
            )
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])

        state.diagnostics["result_json"] = _diagnose(shape)
        if parse_failure:
            state.diagnostics["result_json"] = (
                f"{parse_failure}; {state.diagnostics['result_json']}"
            )
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = (
            f"defect_count={len(coverage_defects)}; " + " | ".join(coverage_defects[:12])
            if coverage_defects
            else "defect_count=0"
        )
        state.diagnostics["decision_legality"] = _diagnose(legality)
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in GATE_CHECKS
        )
        if state.validity_gate:
            try:
                state.metrics = _metrics(state.replayed)
            except Exception as exc:
                state.metrics = None
                state.validity_gate = False
                state.scores["replay_validity"] = 0.0
                state.diagnostics["replay_validity"] = (
                    "failed_atoms=[replay_validity.metrics]; "
                    f"replay_validity.metrics: metric calculation raised {type(exc).__name__}: {exc}"
                )
        if not state.official_ok:
            for check_id in GATE_CHECKS:
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = (
                    f"official_run failed: {state.official_message}; {state.diagnostics.get(check_id, '')}"
                )
    except Exception as exc:
        state.run_finished = state.run_finished or state.run_started
        state.official_message = f"grader state failed: {type(exc).__name__}: {exc}"
        for check_id in GATE_CHECKS:
            state.scores[check_id] = 0.0
            state.diagnostics[check_id] = state.official_message
    _CACHE[key] = state
    return state


def _headings(text: str) -> dict[str, str]:
    """Parse visible Markdown headings and their non-empty bodies."""
    lines = text.splitlines()
    visible_lines = ["" for _ in lines]
    visible = [False] * len(lines)
    in_fence = False
    fence_marker = ""
    in_comment = False
    for index, line in enumerate(lines):
        if in_fence:
            stripped = line.lstrip()
            if stripped.startswith(fence_marker):
                in_fence = False
            continue
        remainder = line
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
        visible_lines[index] = kept
        visible[index] = True

    headings: list[tuple[int, int, int, str]] = []
    for index, line in enumerate(visible_lines):
        if not visible[index]:
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", line)
        if match:
            headings.append((index, index + 1, len(match.group(1)), match.group(2).strip()))
        elif (
            line.strip()
            and index + 1 < len(lines)
            and visible[index + 1]
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", visible_lines[index + 1])
        ):
            level = 1 if "=" in visible_lines[index + 1] else 2
            headings.append((index, index + 2, level, line.strip()))

    heading_markup = {
        line_index
        for start, body_start, _, _ in headings
        for line_index in range(start, body_start)
    }
    result: dict[str, str] = {}
    for order, (_, body_start, level, title) in enumerate(headings):
        end = next(
            (start for start, _, next_level, _ in headings[order + 1 :] if next_level <= level),
            len(lines),
        )
        body = "\n".join(
            visible_lines[index]
            for index in range(body_start, end)
            if visible[index] and index not in heading_markup
        ).strip()
        if title not in result or (body and not result[title]):
            result[title] = body
    return result


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    try:
        if not path.is_file() or path.is_symlink():
            return 0.0, "report.md is not a regular file"
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(text)
    missing = [title for title in required if not headings.get(title)]
    score = max(0.0, 2.0 - 0.5 * len(missing)) / 2.0
    return score, f"missing_sections={missing}; normalized={score:.9g}"


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink():
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


for _id in GATE_CHECKS:

    async def _check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    globals()[_id] = check(id=_id)(_check)


for _id in METRIC_KEYS:

    async def _metric(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        value = (state.metrics or {}).get(ident, 0.0)
        detail = (
            f"{ident}: replayed value={value:.9g}"
            if state.metrics is not None
            else f"{ident}: validity gate not satisfied"
        )
        return value, detail

    globals()[_id] = check(id=_id)(_metric)


del _check, _metric
