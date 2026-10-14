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
GRID = 16
HORIZON = 96
CYCLES = 24
ROBOTS = ("R1", "R2")
DIRECTIONS = ("R1->R2", "R2->R1")
DIRECTION_RANK = {direction: index for index, direction in enumerate(DIRECTIONS)}
ACTIONS = ("stay", "north", "east", "south", "west")
ACTION_DELTAS = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))
CELLS = tuple((x, y) for x in range(2, 14) for y in range(2, 13))
PACKET_BITS = 192
OBSERVATION_BITS = 40
QUEUE_CAPACITY = 3
MAX_PACKETS = 32
INITIAL_ENERGY = 96.0

SCENARIO_NAMES = (
    "crossing_hazard",
    "split_targets",
    "narrow_merge",
    "moving_target",
    "communication_blackout",
)
OFFSETS = (0, 1, 2, 0, 1)
SCENARIOS: dict[str, dict[str, Any]] = {
    "crossing_hazard": {
        "starts": ((1, 7), (14, 7)),
        "obstacles": ((6, 9, 0, 5), (6, 9, 10, 15)),
        "targets": ((4, 6), (11, 8), (5, 9), (10, 5)),
        "hazards": ((5, 7), (10, 7), (8, 6), (7, 9)),
        "deadline": 18,
    },
    "split_targets": {
        "starts": ((2, 2), (13, 13)),
        "obstacles": ((5, 7, 5, 10), (9, 11, 5, 10)),
        "targets": ((3, 4), (12, 11), (4, 12), (11, 3)),
        "hazards": ((4, 5), (11, 10), (7, 4), (8, 11)),
        "deadline": 20,
    },
    "narrow_merge": {
        "starts": ((1, 3), (1, 12)),
        "obstacles": ((4, 11, 0, 4), (4, 11, 11, 15)),
        "targets": ((3, 8), (12, 7), (7, 6), (8, 9)),
        "hazards": ((5, 8), (10, 7), (7, 5), (8, 10)),
        "deadline": 19,
    },
    "moving_target": {
        "starts": ((2, 7), (13, 8)),
        "obstacles": ((5, 8, 0, 4), (8, 11, 11, 15)),
        "targets": ((4, 6), (11, 9), (6, 10), (9, 5)),
        "hazards": ((5, 6), (10, 9), (7, 8), (8, 7)),
        "deadline": 22,
    },
    "communication_blackout": {
        "starts": ((1, 1), (14, 14)),
        "obstacles": ((5, 10, 4, 6), (5, 10, 9, 11)),
        "targets": ((3, 3), (12, 12), (3, 12), (12, 3)),
        "hazards": ((4, 4), (11, 11), (4, 11), (11, 4)),
        "deadline": 21,
    },
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.evaluations": 0.20,
        "result_evidence.communication": 0.20,
        "result_evidence.protocol": 0.20,
        "result_evidence.replay_checkpoints": 0.40,
    },
    "decision_legality": {
        "decision_legality.evaluation_relations": 0.20,
        "decision_legality.packet_identity": 0.20,
        "decision_legality.observation_selection": 0.25,
        "decision_legality.information_versions": 0.15,
        "decision_legality.commitments": 0.20,
    },
    "replay_validity": {
        "replay_validity.observations_beliefs": 0.20,
        "replay_validity.packet_events": 0.20,
        "replay_validity.protocol_actions": 0.20,
        "replay_validity.motion_safety": 0.20,
        "replay_validity.targets_metrics": 0.20,
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


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value: Any, lower: int, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper


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

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )


def _shape(value: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object"
    if set(value) != keys:
        return f"{path}: expected keys={sorted(keys)!r}, actual={sorted(value)!r}"
    return None


def _scenario_error(value: Any, path: str) -> str | None:
    return _shape(
        value,
        {
            "scenario",
            "action_evaluations",
            "communication_opportunities",
            "protocol_submissions",
            "replay_checkpoints",
        },
        path,
    )


def _evaluation_group_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {"robot", "cycle", "ranked_joint_action_ids", "possible_preferred_ids"},
        path,
    )
    if error:
        return error
    if value["robot"] not in ROBOTS or not _integer(value["cycle"], 0, CYCLES - 1):
        return f"{path}: invalid robot or cycle"
    possible = value["possible_preferred_ids"]
    if not isinstance(possible, list) or any(not _integer(item, 0, 24) for item in possible):
        return f"{path}.possible_preferred_ids: expected action-ID array"
    ranked = value["ranked_joint_action_ids"]
    if not isinstance(ranked, list) or any(not _integer(item, 0, 24) for item in ranked):
        return f"{path}.ranked_joint_action_ids: expected action-ID array"
    return None


def _opportunity_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {"cycle", "direction", "send", "observation_ids", "packet_id", "trigger_reason"},
        path,
    )
    if error:
        return error
    if not _integer(value["cycle"], 0, CYCLES - 1) or value["direction"] not in DIRECTIONS:
        return f"{path}: invalid cycle or direction"
    if not isinstance(value["send"], bool):
        return f"{path}.send: expected boolean"
    ids = value["observation_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) > 4
        or any(not isinstance(item, str) or not item for item in ids)
    ):
        return f"{path}.observation_ids: expected at most four nonempty strings"
    if value["send"]:
        if not ids:
            return f"{path}.observation_ids: send requires one to four IDs"
        if not isinstance(value["packet_id"], str) or not value["packet_id"]:
            return f"{path}.packet_id: send requires a nonempty string"
        if not isinstance(value["trigger_reason"], str) or not value["trigger_reason"].strip():
            return f"{path}.trigger_reason: send requires nonempty text"
    elif value["packet_id"] is not None or ids:
        return f"{path}: wait requires null packet_id and empty observation_ids"
    elif not isinstance(value["trigger_reason"], str):
        return f"{path}.trigger_reason: expected string"
    return None


def _protocol_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {"robot", "cycle", "proposal", "commitment_id", "information_version"},
        path,
    )
    if error:
        return error
    if value["robot"] not in ROBOTS or not _integer(value["cycle"], 0, CYCLES - 1):
        return f"{path}: invalid robot or cycle"
    proposal = value["proposal"]
    if proposal is not None and not _integer(proposal, 0, 24):
        return f"{path}.proposal: expected action ID or null"
    commitment = value["commitment_id"]
    if commitment is not None and (not isinstance(commitment, str) or not commitment):
        return f"{path}.commitment_id: expected nonempty string or null"
    if (
        not isinstance(value["information_version"], int)
        or isinstance(value["information_version"], bool)
        or value["information_version"] < 0
    ):
        return f"{path}.information_version: expected nonnegative integer"
    return None


def _checkpoint_robot_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {
            "x",
            "y",
            "energy",
            "information_version",
            "target_probability_sum",
            "hazard_probability_sum",
        },
        path,
    )
    if error:
        return error
    if not _integer(value["x"], 0, GRID - 1) or not _integer(value["y"], 0, GRID - 1):
        return f"{path}.x/y: expected grid coordinates 0..{GRID - 1}"
    if not _finite(value["energy"]) or not -EPS <= float(value["energy"]) <= INITIAL_ENERGY + EPS:
        return f"{path}.energy: expected finite value in [0,{INITIAL_ENERGY:g}]"
    if not _integer(value["information_version"], 0, CYCLES * 3 + MAX_PACKETS * 4):
        return f"{path}.information_version: expected integer in [0,{CYCLES * 3 + MAX_PACKETS * 4}]"
    for field in ("target_probability_sum", "hazard_probability_sum"):
        if not _finite(value[field]) or not -EPS <= float(value[field]) <= len(CELLS) + EPS:
            return f"{path}.{field}: expected finite value in [0,{len(CELLS)}]"
    return None


def _checkpoint_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {
            "boundary",
            "R1",
            "R2",
            "queue_R1_to_R2",
            "queue_R2_to_R1",
            "confirmed_target_ids",
            "safety_violation_count",
            "fallback_count",
            "admitted_packet_count",
            "shared_evidence_version",
        },
        path,
    )
    if error:
        return error
    if not _integer(value["boundary"], 0, CYCLES):
        return f"{path}.boundary: expected integer 0..{CYCLES}"
    for robot in ROBOTS:
        error = _checkpoint_robot_error(value[robot], f"{path}.{robot}")
        if error:
            return error
    for field in ("queue_R1_to_R2", "queue_R2_to_R1"):
        queue = value[field]
        if (
            not isinstance(queue, list)
            or len(queue) > QUEUE_CAPACITY
            or any(not isinstance(packet_id, str) or not packet_id for packet_id in queue)
            or len(queue) != len(set(queue))
        ):
            return f"{path}.{field}: expected at most {QUEUE_CAPACITY} distinct packet IDs"
    identities = value["confirmed_target_ids"]
    if not isinstance(identities, list) or len(identities) > 4:
        return f"{path}.confirmed_target_ids: expected an array of at most four identities"
    normalized: list[tuple[int, int]] = []
    for index, identity in enumerate(identities):
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or any(not _integer(coordinate, 0, GRID - 1) for coordinate in identity)
        ):
            return f"{path}.confirmed_target_ids[{index}]: expected [x,y] grid identity"
        normalized.append((identity[0], identity[1]))
    if len(normalized) != len(set(normalized)):
        return f"{path}.confirmed_target_ids: identities must be distinct"
    for field, upper in (
        ("safety_violation_count", 2 * HORIZON),
        ("fallback_count", CYCLES),
        ("admitted_packet_count", MAX_PACKETS),
        ("shared_evidence_version", MAX_PACKETS * 4),
    ):
        if not _integer(value[field], 0, upper):
            return f"{path}.{field}: expected integer 0..{upper}"
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
    rules = {
        "evaluations": [True, "action-ranking fields are well formed"],
        "communication": [True, "communication fields are well formed"],
        "protocol": [True, "protocol fields are well formed"],
        "replay_checkpoints": [True, "replay checkpoint fields are well formed"],
    }

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            continue
        groups = record.get("action_evaluations")
        if not isinstance(groups, list):
            fail("evaluations", f"{name}.action_evaluations: expected array")
        else:
            for index, group in enumerate(groups):
                error = _evaluation_group_error(group, f"{name}.action_evaluations[{index}]")
                if error:
                    fail("evaluations", error)
                    break
        opportunities = record.get("communication_opportunities")
        if not isinstance(opportunities, list):
            fail("communication", f"{name}.communication_opportunities: expected array")
        else:
            for index, row in enumerate(opportunities):
                error = _opportunity_error(row, f"{name}.communication_opportunities[{index}]")
                if error:
                    fail("communication", error)
                    break
        submissions = record.get("protocol_submissions")
        if not isinstance(submissions, list):
            fail("protocol", f"{name}.protocol_submissions: expected array")
        else:
            for index, row in enumerate(submissions):
                error = _protocol_error(row, f"{name}.protocol_submissions[{index}]")
                if error:
                    fail("protocol", error)
                    break
        checkpoints = record.get("replay_checkpoints")
        if not isinstance(checkpoints, list):
            fail("replay_checkpoints", f"{name}.replay_checkpoints: expected array")
        else:
            for index, row in enumerate(checkpoints):
                error = _checkpoint_error(row, f"{name}.replay_checkpoints[{index}]")
                if error:
                    fail("replay_checkpoints", error)
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
        if key in seen:
            duplicate = True
        seen.add(key)
    if duplicate:
        defects.append(f"{label}: duplicate canonical key")
    missing = expected - seen
    unknown = seen - expected
    if missing:
        defects.append(f"{label}: missing {len(missing)} canonical keys")
    if unknown:
        defects.append(f"{label}: unknown {len(unknown)} canonical keys")
    return defects


def _coverage(data: Any) -> tuple[float, list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios: missing array"]
    defects: list[str] = []
    scenario_rows = data["scenarios"]
    names = [
        _hashable(row.get("scenario")) if isinstance(row, dict) else _hashable(row)
        for row in scenario_rows
    ]
    defects.extend(_key_defects("scenarios", names, set(SCENARIO_NAMES)))
    for name in SCENARIO_NAMES:
        matches = [
            row for row in scenario_rows if isinstance(row, dict) and row.get("scenario") == name
        ]
        if not matches:
            continue
        record = matches[0]
        groups = record.get("action_evaluations")
        if isinstance(groups, list):
            group_keys = [
                _hashable((row.get("robot"), row.get("cycle")))
                if isinstance(row, dict)
                else _hashable(row)
                for row in groups
            ]
            expected_groups = {(robot, cycle) for robot in ROBOTS for cycle in range(CYCLES)}
            defects.extend(_key_defects(f"{name}.action_evaluations", group_keys, expected_groups))
        else:
            defects.append(f"{name}.action_evaluations: missing array")
        for field, expected in (
            (
                "communication_opportunities",
                {(cycle, direction) for cycle in range(CYCLES) for direction in DIRECTIONS},
            ),
            (
                "protocol_submissions",
                {(robot, cycle) for robot in ROBOTS for cycle in range(CYCLES)},
            ),
        ):
            rows = record.get(field)
            if not isinstance(rows, list):
                defects.append(f"{name}.{field}: missing array")
                continue
            if field == "communication_opportunities":
                actual = [
                    _hashable((row.get("cycle"), row.get("direction")))
                    if isinstance(row, dict)
                    else _hashable(row)
                    for row in rows
                ]
            else:
                actual = [
                    _hashable((row.get("robot"), row.get("cycle")))
                    if isinstance(row, dict)
                    else _hashable(row)
                    for row in rows
                ]
            defects.extend(_key_defects(f"{name}.{field}", actual, expected))
        checkpoints = record.get("replay_checkpoints")
        if not isinstance(checkpoints, list):
            defects.append(f"{name}.replay_checkpoints: missing array")
        else:
            checkpoint_keys = [
                _hashable(row.get("boundary")) if isinstance(row, dict) else _hashable(row)
                for row in checkpoints
            ]
            defects.extend(
                _key_defects(f"{name}.replay_checkpoints", checkpoint_keys, set(range(CYCLES + 1)))
            )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _current_targets(name: str, cycle: int) -> dict[tuple[int, int], tuple[int, int]]:
    targets = {identity: identity for identity in SCENARIOS[name]["targets"]}
    if name == "moving_target":
        moving = (4, 6)
        if cycle >= 18:
            location = (4, 7)
        elif cycle >= 12:
            location = (3, 7)
        elif cycle >= 6:
            location = (3, 6)
        else:
            location = moving
        targets.pop(moving)
        targets[location] = moving
    return targets


def _state_kind(name: str, cycle: int, cell: tuple[int, int]) -> str:
    if cell in _current_targets(name, cycle):
        return "target"
    if cell in SCENARIOS[name]["hazards"]:
        return "hazard"
    return "empty"


def _signal(name: str, cycle: int, robot_index: int, cell: tuple[int, int], channel: str) -> int:
    x, y = cell
    scenario_index = SCENARIO_NAMES.index(name)
    if channel == "T":
        value = (13 * x + 7 * y + 5 * cycle + 3 * robot_index + scenario_index) % 20
        truth = _state_kind(name, cycle, cell) == "target"
    else:
        value = (11 * x + 5 * y + 3 * cycle + 5 * robot_index + 2 * scenario_index) % 20
        truth = _state_kind(name, cycle, cell) == "hazard"
    return int(value < (17 if truth else 2))


def _observe(position: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(
        cell for cell in CELLS if abs(cell[0] - position[0]) + abs(cell[1] - position[1]) <= 2
    )[:3]


def _observation_id(robot: str, cycle: int, cell: tuple[int, int]) -> str:
    return f"obs-{robot}-{cycle}-{cell[0]}-{cell[1]}"


def _bayes(
    prior: tuple[float, float, float], target_signal: int, hazard_signal: int
) -> tuple[float, float, float]:
    p_t, p_h, p_e = prior
    lt_t = 0.85 if target_signal else 0.15
    lt_n = 0.10 if target_signal else 0.90
    lh_h = 0.85 if hazard_signal else 0.15
    lh_n = 0.10 if hazard_signal else 0.90
    values = (p_t * lt_t * lh_n, p_h * lt_n * lh_h, p_e * lt_n * lh_n)
    total = sum(values)
    clipped = tuple(max(0.01, value / total) for value in values)
    clipped_total = sum(clipped)
    return tuple(value / clipped_total for value in clipped)


def _is_blackout(name: str, cycle: int) -> bool:
    return name == "communication_blackout" and 7 <= cycle <= 9


def _obstacle(name: str, cell: tuple[int, int]) -> bool:
    return any(
        x0 <= cell[0] <= x1 and y0 <= cell[1] <= y1
        for x0, x1, y0, y1 in SCENARIOS[name]["obstacles"]
    )


def _target_value(identity: tuple[int, int]) -> float:
    return 1.0 + ((3 * identity[0] + identity[1]) % 4) / 3.0


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def _evaluation_row(
    robot: str,
    cycle: int,
    preferred: int,
    possible: Sequence[int] | None = None,
    scores: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    values = {
        action_id: float(scores[action_id]) if scores is not None else -abs(action_id - preferred)
        for action_id in range(25)
    }
    ordering = sorted(range(25), key=lambda action_id: (-values[action_id], action_id))
    actual_preferred = ordering[0]
    possible_ids = sorted(set(possible if possible is not None else [actual_preferred]))
    if actual_preferred not in possible_ids:
        possible_ids.append(actual_preferred)
        possible_ids.sort()
    return {
        "robot": robot,
        "cycle": cycle,
        "ranked_joint_action_ids": ordering,
        "possible_preferred_ids": possible_ids,
    }


def _decision_map(
    record: Mapping[str, Any], *, require_checkpoints: bool = True
) -> dict[str, Any] | None:
    fields = ("action_evaluations", "communication_opportunities", "protocol_submissions")
    if any(not isinstance(record.get(field), list) for field in fields):
        return None
    evaluations: dict[tuple[str, int], dict[str, Any]] = {}
    for row in record["action_evaluations"]:
        if not isinstance(row, dict) or _evaluation_group_error(row, "evaluation"):
            return None
        key = (row["robot"], row["cycle"])
        if key in evaluations:
            return None
        evaluations[key] = row
    opportunities: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["communication_opportunities"]:
        if not isinstance(row, dict) or _opportunity_error(row, "opportunity"):
            return None
        key = (row["cycle"], row["direction"])
        if key in opportunities:
            return None
        opportunities[key] = row
    submissions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in record["protocol_submissions"]:
        if not isinstance(row, dict) or _protocol_error(row, "submission"):
            return None
        key = (row["robot"], row["cycle"])
        if key in submissions:
            return None
        submissions[key] = row
    if set(evaluations) != {(robot, cycle) for robot in ROBOTS for cycle in range(CYCLES)}:
        return None
    if set(opportunities) != {
        (cycle, direction) for cycle in range(CYCLES) for direction in DIRECTIONS
    }:
        return None
    if set(submissions) != {(robot, cycle) for robot in ROBOTS for cycle in range(CYCLES)}:
        return None
    result = {
        "evaluations": evaluations,
        "opportunities": opportunities,
        "submissions": submissions,
    }
    if require_checkpoints:
        rows = record.get("replay_checkpoints")
        if not isinstance(rows, list):
            return None
        checkpoints: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or _checkpoint_error(row, "checkpoint"):
                return None
            boundary = row["boundary"]
            if boundary in checkpoints:
                return None
            checkpoints[boundary] = row
        if set(checkpoints) != set(range(CYCLES + 1)):
            return None
        result["checkpoints"] = checkpoints
    return result


def _movement_step(
    name: str,
    cycle: int,
    joint_action_id: int | None,
    positions: dict[str, tuple[int, int]],
    energies: dict[str, float],
) -> tuple[set[str], int]:
    old = dict(positions)
    action_indices = (0, 0) if joint_action_id is None else divmod(joint_action_id, 5)
    effective: dict[str, int] = {}
    requested_moves = 0
    proposed: dict[str, tuple[int, int]] = {}
    for robot, action_index in zip(ROBOTS, action_indices):
        if action_index != 0 and energies[robot] >= 1.0 - EPS:
            effective[robot] = action_index
            energies[robot] -= 1.0
            requested_moves += 1
        else:
            effective[robot] = 0
        dx, dy = ACTION_DELTAS[effective[robot]]
        proposed[robot] = (old[robot][0] + dx, old[robot][1] + dy)

    blocked: set[str] = set()
    for robot in ROBOTS:
        cell = proposed[robot]
        if (
            not 0 <= cell[0] < GRID
            or not 0 <= cell[1] < GRID
            or _obstacle(name, cell)
            or _state_kind(name, cycle, cell) == "hazard"
        ):
            blocked.add(robot)
    if proposed["R1"] == proposed["R2"]:
        blocked.update(ROBOTS)
    if proposed["R1"] == old["R2"] and proposed["R2"] == old["R1"] and old["R1"] != old["R2"]:
        blocked.update(ROBOTS)

    retained = {robot: old[robot] if robot in blocked else proposed[robot] for robot in ROBOTS}
    if retained["R1"] == retained["R2"]:
        for robot in ROBOTS:
            if robot not in blocked and effective[robot] != 0:
                blocked.add(robot)
        retained = {robot: old[robot] if robot in blocked else proposed[robot] for robot in ROBOTS}
    positions.update(retained)
    return blocked, requested_moves


def _scenario_metrics(name: str, replay: Mapping[str, Any]) -> dict[str, float]:
    first_confirmations = replay["first_confirmations"]
    deadline = int(SCENARIOS[name]["deadline"])
    total_value = sum(_target_value(identity) for identity in SCENARIOS[name]["targets"])
    deadline_value = (
        sum(
            _target_value(identity)
            for identity, event in first_confirmations.items()
            if event["cycle"] <= deadline
        )
        / total_value
    )

    h0 = _binary_entropy(0.25)
    reduction = 0.0
    beliefs = replay["beliefs"]
    for cell in CELLS:
        mean_probability = sum(beliefs[robot][cell][0] for robot in ROBOTS) / len(ROBOTS)
        current_truth = _state_kind(name, 23, cell) == "target"
        if (mean_probability >= 0.5) == current_truth:
            reduction += max(0.0, h0 - _binary_entropy(mean_probability))
    entropy_efficiency = max(0.0, min(1.0, reduction / (h0 * len(CELLS))))
    primary_1 = 0.6 * deadline_value + 0.4 * entropy_efficiency

    confirmed_fraction = sum(action["confirmed"] for action in replay["actions"]) / CYCLES
    safe_fraction = 1.0 - replay["safety_violations"] / (2 * HORIZON)
    primary_2 = 0.6 * confirmed_fraction + 0.4 * safe_fraction

    requested_moves = replay["requested_moves"]
    fallback_count = sum(action["fallback"] for action in replay["actions"])
    admitted_count = sum(packet["admitted"] for packet in replay["packets"])
    energy_used = requested_moves + 0.25 * admitted_count
    primary_3 = (
        0.35 * max(0.0, min(1.0, 1.0 - requested_moves / (2 * HORIZON)))
        + 0.35 * max(0.0, min(1.0, 1.0 - fallback_count / CYCLES))
        + 0.30 * max(0.0, min(1.0, 1.0 - energy_used / 200.0))
    )

    delivered_packets = [packet for packet in replay["packets"] if packet["outcome"] == "delivered"]
    delivered_ids = {
        observation_id
        for packet in delivered_packets
        for observation_id in packet["observation_ids"]
    }
    summary_apps = replay["summary_applications"]
    application_cycles: dict[str, list[int]] = {robot: [] for robot in ROBOTS}
    for event in summary_apps:
        application_cycles[event["receiver"]].append(event["arrival_cycle"])
    for robot in ROBOTS:
        application_cycles[robot] = sorted(set(application_cycles[robot]))

    target_events = replay["target_events"]
    useful: dict[str, dict[str, Any]] = {}
    actions = replay["actions"]
    for event in summary_apps:
        arrival = event["arrival_cycle"]
        receiver = event["receiver"]
        later_cycles = [cycle for cycle in application_cycles[receiver] if cycle > arrival]
        final_cycle = (later_cycles[0] - 1) if later_cycles else CYCLES - 1
        qualifying: list[int] = []
        for cycle in range(arrival, final_cycle + 1):
            if (
                actions[cycle]["confirmed"]
                and not actions[cycle]["fallback"]
                and actions[cycle]["joint_action_id"] != 0
            ):
                qualifying.append(cycle)
            if any(
                target_event["robot"] == receiver
                and target_event["cycle"] == cycle
                and cycle <= deadline
                for target_event in target_events
            ):
                qualifying.append(cycle)
        if qualifying:
            useful[event["observation_id"]] = {**event, "qualifying_cycle": min(qualifying)}

    if admitted_count == 0 or not delivered_ids:
        auxiliary_1 = 0.0
    else:
        useful_bits = OBSERVATION_BITS * len(useful)
        admitted_bits = PACKET_BITS * admitted_count
        auxiliary_1 = 0.5 * min(1.0, useful_bits / admitted_bits) + 0.5 * (
            len(useful) / len(delivered_ids)
        )

    useful_packets: dict[str, int] = {}
    for event in useful.values():
        age = event["qualifying_cycle"] - event["arrival_cycle"]
        packet_id = event["packet_id"]
        useful_packets[packet_id] = min(age, useful_packets.get(packet_id, age))
    if admitted_count == 0 or not useful_packets:
        auxiliary_2 = 0.0
    else:
        freshness = sum(
            max(0.0, min(1.0, 1.0 - age / 12.0)) for age in useful_packets.values()
        ) / len(useful_packets)
        total_attempts = sum(
            len(packet["attempts"]) for packet in replay["packets"] if packet["admitted"]
        )
        retry_efficiency = max(
            0.0,
            min(1.0, 1.0 - max(0, total_attempts - admitted_count) / admitted_count),
        )
        fallback_runs: list[tuple[int, int]] = []
        start: int | None = None
        for cycle, action in enumerate(actions):
            if action["fallback"] and start is None:
                start = cycle
            if start is not None and (not action["fallback"] or cycle == CYCLES - 1):
                end = cycle - 1 if not action["fallback"] else cycle
                fallback_runs.append((start, end))
                start = None
        recovered = 0
        for run_start, run_end in fallback_runs:
            candidates = [
                cycle
                for cycle in range(run_end + 1, min(CYCLES, run_end + 13))
                if actions[cycle]["confirmed"] and not actions[cycle]["fallback"]
            ]
            if not candidates:
                continue
            recovery_cycle = candidates[0]
            if any(
                event["arrival_cycle"] >= run_start
                and event["arrival_cycle"] <= recovery_cycle
                and event["qualifying_cycle"] == recovery_cycle
                for event in useful.values()
            ):
                recovered += 1
        recovery = recovered / len(fallback_runs) if fallback_runs else 0.0
        auxiliary_2 = 0.4 * freshness + 0.3 * retry_efficiency + 0.3 * recovery

    return {
        "primary_metrics_1": max(0.0, min(1.0, primary_1)),
        "primary_metrics_2": max(0.0, min(1.0, primary_2)),
        "primary_metrics_3": max(0.0, min(1.0, primary_3)),
        "auxiliary_metrics_1": max(0.0, min(1.0, auxiliary_1)),
        "auxiliary_metrics_2": max(0.0, min(1.0, auxiliary_2)),
    }


def _simulate(
    name: str,
    decisions: Mapping[str, Any] | None = None,
    provider: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if name not in SCENARIOS or (decisions is None) == (provider is None):
        raise ValueError("provide exactly one valid scenario and decision source")
    positions = {
        robot: tuple(SCENARIOS[name]["starts"][index]) for index, robot in enumerate(ROBOTS)
    }
    energies = {robot: INITIAL_ENERGY for robot in ROBOTS}
    beliefs = {robot: {cell: (0.25, 0.15, 0.60) for cell in CELLS} for robot in ROBOTS}
    versions = {robot: 0 for robot in ROBOTS}
    applied = {robot: set() for robot in ROBOTS}
    shared_observations: set[str] = set()
    shared_evidence_version = 0
    observations: dict[str, dict[str, Any]] = {}
    packets: dict[str, dict[str, Any]] = {}
    packet_order: list[str] = []
    queues: dict[str, list[str]] = {direction: [] for direction in DIRECTIONS}
    inflight: dict[str, str | None] = {direction: None for direction in DIRECTIONS}
    requested_observations: dict[str, set[str]] = {direction: set() for direction in DIRECTIONS}
    admitted_count = 0

    evaluation_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    execution_feedback: list[dict[str, Any]] = []
    admission_feedback: list[dict[str, Any]] = []
    summary_applications: list[dict[str, Any]] = []
    target_events: list[dict[str, Any]] = []
    first_confirmations: dict[tuple[int, int], dict[str, Any]] = {}
    safety_violations = 0
    requested_moves = 0
    legality = {
        "packet_identity": [True, "packet IDs are unique"],
        "observation_selection": [True, "summary observations are own, visible and unrepeated"],
        "information_versions": [True, "submitted information versions match replay"],
        "commitments": [
            True,
            "commitments encode the submitted action and shared evidence version",
        ],
    }

    def fail(rule: str, detail: str) -> None:
        if legality[rule][0]:
            legality[rule] = [False, detail]

    def record_state(slot: int) -> None:
        states.append(
            {
                "slot": slot,
                "positions": dict(positions),
                "energies": dict(energies),
                "safety_violations": safety_violations,
                "requested_moves": requested_moves,
                "admitted_packets": admitted_count,
            }
        )

    def record_checkpoint(boundary: int) -> None:
        checkpoints.append(
            {
                "boundary": boundary,
                **{
                    robot: {
                        "x": positions[robot][0],
                        "y": positions[robot][1],
                        "energy": energies[robot],
                        "information_version": versions[robot],
                        "target_probability_sum": sum(
                            beliefs[robot][cell][0] for cell in CELLS
                        ),
                        "hazard_probability_sum": sum(
                            beliefs[robot][cell][1] for cell in CELLS
                        ),
                    }
                    for robot in ROBOTS
                },
                "queue_R1_to_R2": list(queues["R1->R2"]),
                "queue_R2_to_R1": list(queues["R2->R1"]),
                "confirmed_target_ids": [
                    list(identity) for identity in sorted(first_confirmations)
                ],
                "safety_violation_count": safety_violations,
                "fallback_count": sum(action["fallback"] for action in actions),
                "admitted_packet_count": admitted_count,
                "shared_evidence_version": shared_evidence_version,
            }
        )

    record_state(0)
    record_checkpoint(0)
    for cycle in range(CYCLES):
        arriving = sorted(
            (
                packet
                for packet in packets.values()
                if packet["outcome"] == "delivered" and packet["arrival_cycle"] == cycle
            ),
            key=lambda packet: (
                packet["source_cycle"],
                DIRECTION_RANK[packet["direction"]],
                packet["packet_id"],
            ),
        )
        arrived_by_robot: dict[str, list[str]] = {robot: [] for robot in ROBOTS}
        received_packets: dict[str, list[str]] = {robot: [] for robot in ROBOTS}
        acked_packets: dict[str, list[str]] = {robot: [] for robot in ROBOTS}
        for packet in arriving:
            receiver = "R2" if packet["direction"] == "R1->R2" else "R1"
            sender = "R1" if packet["direction"] == "R1->R2" else "R2"
            received_packets[receiver].append(packet["packet_id"])
            acked_packets[sender].append(packet["packet_id"])
            for observation_id in packet["observation_ids"]:
                if observation_id not in shared_observations:
                    shared_observations.add(observation_id)
                    shared_evidence_version += 1
                if observation_id in applied[receiver]:
                    continue
                observation = observations[observation_id]
                cell = observation["cell"]
                beliefs[receiver][cell] = _bayes(
                    beliefs[receiver][cell],
                    observation["target_signal"],
                    observation["hazard_signal"],
                )
                applied[receiver].add(observation_id)
                versions[receiver] += 1
                arrived_by_robot[receiver].append(observation_id)
                summary_applications.append(
                    {
                        "observation_id": observation_id,
                        "packet_id": packet["packet_id"],
                        "receiver": receiver,
                        "arrival_cycle": cycle,
                    }
                )

        local_by_robot: dict[str, list[str]] = {robot: [] for robot in ROBOTS}
        for robot_index, robot in enumerate(ROBOTS):
            for cell in _observe(positions[robot]):
                observation_id = _observation_id(robot, cycle, cell)
                observation = {
                    "observation_id": observation_id,
                    "robot": robot,
                    "cycle": cycle,
                    "cell": cell,
                    "target_signal": _signal(name, cycle, robot_index, cell, "T"),
                    "hazard_signal": _signal(name, cycle, robot_index, cell, "H"),
                }
                observations[observation_id] = observation
                beliefs[robot][cell] = _bayes(
                    beliefs[robot][cell],
                    observation["target_signal"],
                    observation["hazard_signal"],
                )
                applied[robot].add(observation_id)
                versions[robot] += 1
                local_by_robot[robot].append(observation_id)

        view = {
            "scenario": name,
            "cycle": cycle,
            "positions": dict(positions),
            "energies": dict(energies),
            "shared_evidence_version": shared_evidence_version,
            "shared_arrived_packet_ids": [packet["packet_id"] for packet in arriving],
            "admitted_packet_count": admitted_count,
            "remaining_packet_budget": MAX_PACKETS - admitted_count,
            "confirmed_target_ids": [list(identity) for identity in sorted(first_confirmations)],
            "admission_feedback": [dict(row) for row in admission_feedback],
            "previous_execution_feedback": [
                {
                    **row,
                    "positions": dict(row["positions"]),
                    "energies": dict(row["energies"]),
                    "blocked": dict(row["blocked"]),
                }
                for row in execution_feedback
                if row["cycle"] == cycle - 1
            ],
            "robot_views": {
                robot: {
                    "beliefs": dict(beliefs[robot]),
                    "information_version": versions[robot],
                    "arrived_observation_ids": list(arrived_by_robot[robot]),
                    "received_packet_ids": list(received_packets[robot]),
                    "acked_packet_ids": list(acked_packets[robot]),
                    "local_observation_ids": list(local_by_robot[robot]),
                    "own_observations": [
                        dict(observation)
                        for observation in observations.values()
                        if observation["robot"] == robot
                    ],
                    "outgoing_queue": list(queues["R1->R2" if robot == "R1" else "R2->R1"]),
                    "outgoing_link_available": not _is_blackout(name, cycle),
                }
                for robot in ROBOTS
            },
            "action_history": list(actions),
        }
        if provider is not None:
            selected = provider(cycle, view)
            proposals = {robot: selected.get("proposals", {}).get(robot) for robot in ROBOTS}
            commitments = {robot: selected.get("commitments", {}).get(robot) for robot in ROBOTS}
            information_versions = {
                robot: selected.get("information_versions", {}).get(robot, versions[robot])
                for robot in ROBOTS
            }
            provided_evaluations = selected.get("evaluations", {})
            cycle_evaluations = {
                robot: provided_evaluations.get(robot)
                or _evaluation_row(
                    robot,
                    cycle,
                    proposals[robot] if isinstance(proposals[robot], int) else 0,
                )
                for robot in ROBOTS
            }
            provided_opportunities = selected.get("opportunities", {})
            cycle_opportunities = {
                direction: {
                    "cycle": cycle,
                    "direction": direction,
                    **provided_opportunities.get(
                        direction,
                        {
                            "send": False,
                            "observation_ids": [],
                            "packet_id": None,
                            "trigger_reason": "wait",
                        },
                    ),
                }
                for direction in DIRECTIONS
            }
            cycle_submissions = {
                robot: {
                    "robot": robot,
                    "cycle": cycle,
                    "proposal": proposals[robot],
                    "commitment_id": commitments[robot],
                    "information_version": information_versions[robot],
                }
                for robot in ROBOTS
            }
        else:
            assert decisions is not None
            cycle_evaluations = {
                robot: decisions["evaluations"][(robot, cycle)] for robot in ROBOTS
            }
            cycle_opportunities = {
                direction: decisions["opportunities"][(cycle, direction)]
                for direction in DIRECTIONS
            }
            cycle_submissions = {
                robot: decisions["submissions"][(robot, cycle)] for robot in ROBOTS
            }
            proposals = {robot: cycle_submissions[robot]["proposal"] for robot in ROBOTS}
            commitments = {robot: cycle_submissions[robot]["commitment_id"] for robot in ROBOTS}

        evaluation_rows.extend(dict(cycle_evaluations[robot]) for robot in ROBOTS)
        opportunity_rows.extend(dict(cycle_opportunities[direction]) for direction in DIRECTIONS)
        submission_rows.extend(dict(cycle_submissions[robot]) for robot in ROBOTS)
        for robot in ROBOTS:
            actual_version = cycle_submissions[robot]["information_version"]
            if actual_version != versions[robot]:
                fail(
                    "information_versions",
                    f"{name}.protocol_submissions[{robot},{cycle}].information_version: "
                    f"expected={versions[robot]}, actual={actual_version}",
                )

        for direction in DIRECTIONS:
            packet_id = inflight[direction]
            if packet_id is None:
                continue
            packet = packets[packet_id]
            attempt = packet["attempts"][-1]
            if attempt["completion_cycle"] != cycle:
                continue
            if not queues[direction] or queues[direction][0] != packet_id:
                raise ValueError(f"{name}.{direction}: inflight packet is not queue head")
            queues[direction].pop(0)
            inflight[direction] = None
            scenario_index = SCENARIO_NAMES.index(name)
            loss = (
                11 * cycle
                + 3 * DIRECTION_RANK[direction]
                + 2 * scenario_index
                + OFFSETS[scenario_index]
            ) % 17 == 0
            if loss:
                attempt["outcome"] = "lost"
                if cycle < packet["expiry_cycle"]:
                    queues[direction].append(packet_id)
                else:
                    packet["outcome"] = "expired"
            else:
                attempt["outcome"] = "delivered"
                packet["outcome"] = "delivered"
                packet["arrival_cycle"] = cycle + 1

        for direction in DIRECTIONS:
            if inflight[direction] is not None or _is_blackout(name, cycle):
                continue
            while queues[direction]:
                packet_id = queues[direction][0]
                packet = packets[packet_id]
                delay = (
                    1
                    + (5 * cycle + OFFSETS[SCENARIO_NAMES.index(name)] + DIRECTION_RANK[direction])
                    % 3
                )
                completion_cycle = cycle + delay
                if completion_cycle > packet["expiry_cycle"]:
                    queues[direction].pop(0)
                    packet["outcome"] = "expired"
                    continue
                packet["attempts"].append(
                    {
                        "attempt_no": len(packet["attempts"]),
                        "service_cycle": cycle,
                        "completion_cycle": completion_cycle,
                        "outcome": None,
                    }
                )
                inflight[direction] = packet_id
                break

        for direction in DIRECTIONS:
            opportunity = cycle_opportunities[direction]
            if not opportunity["send"]:
                continue
            packet_id = opportunity["packet_id"]
            observation_ids = list(opportunity["observation_ids"])
            source_robot = "R1" if direction == "R1->R2" else "R2"
            identity_ok = isinstance(packet_id, str) and packet_id not in packets
            if not identity_ok:
                fail(
                    "packet_identity",
                    f"{name}.communication_opportunities[{cycle},{direction}]: duplicate packet_id",
                )
            selection_ok = (
                len(observation_ids) == len(set(observation_ids))
                and observation_ids == sorted(observation_ids)
                and all(
                    observation_id in observations
                    and observations[observation_id]["robot"] == source_robot
                    and observations[observation_id]["cycle"] <= cycle
                    and observation_id not in requested_observations[direction]
                    for observation_id in observation_ids
                )
            )
            if not selection_ok:
                fail(
                    "observation_selection",
                    f"{name}.communication_opportunities[{cycle},{direction}]: "
                    "IDs must be distinct, sorted, own, visible and unrepeated",
                )
            requested_observations[direction].update(observation_ids)
            if not identity_ok or not selection_ok:
                continue
            expiry_cycle = min(
                min(23, int(observations[observation_id]["cycle"]) + 6)
                for observation_id in observation_ids
            )
            admitted = len(queues[direction]) < QUEUE_CAPACITY and admitted_count < MAX_PACKETS
            packet = {
                "packet_id": packet_id,
                "direction": direction,
                "source_cycle": cycle,
                "observation_ids": observation_ids,
                "expiry_cycle": expiry_cycle,
                "admitted": admitted,
                "attempts": [],
                "arrival_cycle": None,
                "outcome": None if admitted else "dropped",
            }
            packets[packet_id] = packet
            packet_order.append(packet_id)
            if admitted:
                queues[direction].append(packet_id)
                admitted_count += 1
            admission_feedback.append(
                {
                    "cycle": cycle,
                    "direction": direction,
                    "packet_id": packet_id,
                    "outcome": "admitted" if admitted else "dropped",
                }
            )

        commitment_ok: dict[str, bool] = {}
        for robot in ROBOTS:
            proposal = proposals[robot]
            expected_commitment = (
                None if proposal is None else f"ac:{proposal}:{shared_evidence_version}"
            )
            commitment_ok[robot] = commitments[robot] == expected_commitment
            if not commitment_ok[robot]:
                fail(
                    "commitments",
                    f"{name}.protocol_submissions[{robot},{cycle}].commitment_id: "
                    f"expected={expected_commitment!r}, actual={commitments[robot]!r}",
                )
        confirmed = (
            _integer(proposals["R1"], 0, 24)
            and proposals["R1"] == proposals["R2"]
            and commitment_ok["R1"]
            and commitment_ok["R2"]
            and commitments["R1"] == commitments["R2"]
        )
        joint_action_id = proposals["R1"] if confirmed else None
        action = {
            "cycle": cycle,
            "proposal_R1": proposals["R1"],
            "proposal_R2": proposals["R2"],
            "commitment_R1": commitments["R1"],
            "commitment_R2": commitments["R2"],
            "confirmed": confirmed,
            "fallback": not confirmed,
            "joint_action_id": joint_action_id,
        }
        actions.append(action)
        for primitive_index in range(4):
            blocked, move_count = _movement_step(
                name,
                cycle,
                joint_action_id,
                positions,
                energies,
            )
            safety_violations += len(blocked)
            requested_moves += move_count
            execution_feedback.append(
                {
                    "cycle": cycle,
                    "slot": 4 * cycle + primitive_index + 1,
                    "positions": dict(positions),
                    "energies": dict(energies),
                    "blocked": {robot: robot in blocked for robot in ROBOTS},
                }
            )
            current_targets = _current_targets(name, cycle)
            for robot in ROBOTS:
                identity = current_targets.get(positions[robot])
                if identity is None or identity in first_confirmations:
                    continue
                if beliefs[robot][positions[robot]][0] >= 0.9 - EPS:
                    event = {
                        "target_identity": identity,
                        "robot": robot,
                        "cycle": cycle,
                        "slot": 4 * cycle + primitive_index + 1,
                    }
                    first_confirmations[identity] = event
                    target_events.append(event)
            record_state(4 * cycle + primitive_index + 1)
        record_checkpoint(cycle + 1)

    for packet in packets.values():
        if packet["admitted"] and packet["outcome"] is None:
            packet["outcome"] = "unfinished"
    replay = {
        "scenario": name,
        "observations": observations,
        "beliefs": beliefs,
        "versions": versions,
        "packets": [packets[packet_id] for packet_id in packet_order],
        "actions": actions,
        "states": states,
        "summary_applications": summary_applications,
        "target_events": target_events,
        "first_confirmations": first_confirmations,
        "safety_violations": safety_violations,
        "requested_moves": requested_moves,
        "shared_evidence_version": shared_evidence_version,
        "checkpoints": checkpoints,
        "execution_feedback": execution_feedback,
        "admission_feedback": admission_feedback,
        "legality": legality,
    }
    replay["metrics"] = _scenario_metrics(name, replay)
    replay["record"] = {
        "scenario": name,
        "action_evaluations": evaluation_rows,
        "communication_opportunities": opportunity_rows,
        "protocol_submissions": submission_rows,
        "replay_checkpoints": checkpoints,
    }
    return replay


def _evaluation_relations(records: Mapping[str, dict[str, Any]]) -> tuple[bool, str]:
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None or not isinstance(record.get("action_evaluations"), list):
            continue
        for index, group in enumerate(record["action_evaluations"]):
            if not isinstance(group, dict) or _evaluation_group_error(group, "evaluation"):
                continue
            possible = group["possible_preferred_ids"]
            if possible != sorted(set(possible)):
                return (
                    False,
                    f"{name}.action_evaluations[{index}]: possible IDs not strictly increasing",
                )
            if not possible:
                return (
                    False,
                    f"{name}.action_evaluations[{index}].possible_preferred_ids: must be nonempty",
                )
            ranked = group["ranked_joint_action_ids"]
            if len(ranked) != 25 or set(ranked) != set(range(25)):
                return (
                    False,
                    f"{name}.action_evaluations[{index}].ranked_joint_action_ids: "
                    "expected a permutation of 0..24",
                )
            if ranked[0] not in possible:
                return (
                    False,
                    f"{name}.action_evaluations[{index}]: first ranked ID absent from possible set",
                )
    return True, "ranked action IDs and possible sets satisfy the published local relations"


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    relation_ok, relation_detail = _evaluation_relations(records)
    rules: dict[str, tuple[bool, str]] = {
        "evaluation_relations": (relation_ok, relation_detail),
        "packet_identity": (True, "packet IDs are unique"),
        "observation_selection": (True, "summary selections obey visibility and reuse rules"),
        "information_versions": (True, "information versions match replayed visible observations"),
        "commitments": (True, "commitments encode action and shared evidence version"),
    }
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            continue
        decisions = _decision_map(record, require_checkpoints=False)
        if decisions is None:
            continue
        try:
            replay = _simulate(name, decisions=decisions)
        except Exception as exc:  # noqa: BLE001
            detail = f"{name}: legality replay raised {type(exc).__name__}: {exc}"
            for rule in (
                "packet_identity",
                "observation_selection",
                "information_versions",
                "commitments",
            ):
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        for rule in (
            "packet_identity",
            "observation_selection",
            "information_versions",
            "commitments",
        ):
            verdict = replay["legality"][rule]
            if rule in {"information_versions", "commitments"} and (
                not replay["legality"]["packet_identity"][0]
                or not replay["legality"]["observation_selection"][0]
            ):
                continue
            if not verdict[0] and rules[rule][0]:
                rules[rule] = (False, str(verdict[1]))
    return [
        AtomVerdict(
            f"decision_legality.{rule}",
            passed,
            detail,
            ATOM_SHARES["decision_legality"][f"decision_legality.{rule}"],
        )
        for rule, (passed, detail) in rules.items()
    ]


def _packet_integrity(replay: Mapping[str, Any]) -> tuple[bool, str]:
    seen: set[str] = set()
    for packet in replay["packets"]:
        packet_id = packet["packet_id"]
        if packet_id in seen:
            return False, f"duplicate derived packet {packet_id!r}"
        seen.add(packet_id)
        if not packet["admitted"] and packet["outcome"] != "dropped":
            return False, f"{packet_id}: non-admitted packet is not dropped"
        for attempt_no, attempt in enumerate(packet["attempts"]):
            if attempt["attempt_no"] != attempt_no:
                return False, f"{packet_id}: non-contiguous attempt number"
            if attempt["completion_cycle"] <= attempt["service_cycle"]:
                return False, f"{packet_id}: completion does not follow service"
        if packet["outcome"] == "delivered":
            if packet["arrival_cycle"] is None or not packet["attempts"]:
                return False, f"{packet_id}: delivered packet lacks arrival or attempt"
            if packet["arrival_cycle"] != packet["attempts"][-1]["completion_cycle"] + 1:
                return False, f"{packet_id}: arrival cycle mismatch"
    return True, "derived FIFO, attempt, expiry, loss and arrival events are internally consistent"


def _checkpoint_mismatch(
    name: str,
    submitted: Mapping[int, Mapping[str, Any]],
    derived: Sequence[Mapping[str, Any]],
    rule: str,
) -> str | None:
    expected_by_boundary = {row["boundary"]: row for row in derived}

    def exact(path: str, expected: Any, actual: Any) -> str | None:
        if actual != expected:
            return f"{name}.replay_checkpoints[{path}]: expected={expected!r}, actual={actual!r}"
        return None

    def real(path: str, expected: Any, actual: Any) -> str | None:
        if not _finite(actual) or not math.isclose(
            float(actual), float(expected), abs_tol=EPS, rel_tol=0.0
        ):
            return (
                f"{name}.replay_checkpoints[{path}]: expected={expected!r}, "
                f"actual={actual!r}, abs_tol={EPS}"
            )
        return None

    for boundary in range(CYCLES + 1):
        expected = expected_by_boundary[boundary]
        actual = submitted[boundary]
        prefix = f"boundary={boundary}"
        if rule == "observations_beliefs":
            for robot in ROBOTS:
                for field in (
                    "information_version",
                    "target_probability_sum",
                    "hazard_probability_sum",
                ):
                    path = f"{prefix}.{robot}.{field}"
                    mismatch = (
                        exact(path, expected[robot][field], actual[robot][field])
                        if field == "information_version"
                        else real(path, expected[robot][field], actual[robot][field])
                    )
                    if mismatch:
                        return mismatch
        elif rule == "packet_events":
            for field in (
                "queue_R1_to_R2",
                "queue_R2_to_R1",
                "admitted_packet_count",
            ):
                mismatch = exact(f"{prefix}.{field}", expected[field], actual[field])
                if mismatch:
                    return mismatch
        elif rule == "protocol_actions":
            for field in ("fallback_count", "shared_evidence_version"):
                mismatch = exact(f"{prefix}.{field}", expected[field], actual[field])
                if mismatch:
                    return mismatch
        elif rule == "motion_safety":
            for robot in ROBOTS:
                for field in ("x", "y", "energy"):
                    path = f"{prefix}.{robot}.{field}"
                    mismatch = (
                        real(path, expected[robot][field], actual[robot][field])
                        if field == "energy"
                        else exact(path, expected[robot][field], actual[robot][field])
                    )
                    if mismatch:
                        return mismatch
            mismatch = exact(
                f"{prefix}.safety_violation_count",
                expected["safety_violation_count"],
                actual["safety_violation_count"],
            )
            if mismatch:
                return mismatch
        elif rule == "targets_metrics":
            expected_ids = sorted(tuple(identity) for identity in expected["confirmed_target_ids"])
            actual_ids = sorted(tuple(identity) for identity in actual["confirmed_target_ids"])
            mismatch = exact(f"{prefix}.confirmed_target_ids", expected_ids, actual_ids)
            if mismatch:
                return mismatch
        else:
            raise ValueError(f"unknown replay checkpoint rule {rule!r}")
    return None


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules: dict[str, tuple[bool, str]] = {
        "observations_beliefs": (True, "fixed observations and three-state beliefs replayed"),
        "packet_events": (True, "fixed packet events replayed"),
        "protocol_actions": (True, "protocol predicate and fallback replayed"),
        "motion_safety": (True, "motion, energy and safety replayed"),
        "targets_metrics": (True, "target identities and metric inputs replayed"),
    }
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            detail = f"{name}: missing scenario record; replay checkpoints unavailable"
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        decisions = _decision_map(record)
        if decisions is None:
            detail = f"{name}: required decision or replay-checkpoint map is unusable"
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        try:
            replay = _simulate(name, decisions=decisions)
        except Exception as exc:  # noqa: BLE001
            detail = f"{name}: replay raised {type(exc).__name__}: {exc}"
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        replayed[name] = replay
        probability_ok = all(
            all(
                all(_finite(value) and 0.0 <= value <= 1.0 for value in belief)
                and math.isclose(sum(belief), 1.0, abs_tol=EPS, rel_tol=0.0)
                for belief in replay["beliefs"][robot].values()
            )
            for robot in ROBOTS
        )
        if not probability_ok:
            rules["observations_beliefs"] = (False, f"{name}: invalid replayed belief")
        packet_ok, packet_detail = _packet_integrity(replay)
        if not packet_ok and rules["packet_events"][0]:
            rules["packet_events"] = (False, f"{name}: {packet_detail}")
        protocol_ok = len(replay["actions"]) == CYCLES and all(
            action["fallback"] == (not action["confirmed"])
            and action["joint_action_id"]
            == (action["proposal_R1"] if action["confirmed"] else None)
            for action in replay["actions"]
        )
        if not protocol_ok:
            rules["protocol_actions"] = (False, f"{name}: derived protocol/action invariant failed")
        motion_ok = (
            len(replay["states"]) == HORIZON + 1
            and 0 <= replay["safety_violations"] <= 2 * HORIZON
            and 0 <= replay["requested_moves"] <= 2 * HORIZON
            and all(
                all(
                    0 <= cell[0] < GRID
                    and 0 <= cell[1] < GRID
                    and -EPS <= state["energies"][robot] <= INITIAL_ENERGY + EPS
                    for robot, cell in state["positions"].items()
                )
                for state in replay["states"]
            )
        )
        if not motion_ok:
            rules["motion_safety"] = (False, f"{name}: derived motion/state invariant failed")
        target_ok = set(replay["first_confirmations"]).issubset(set(SCENARIOS[name]["targets"]))
        metric_ok = set(replay["metrics"]) == {
            "primary_metrics_1",
            "primary_metrics_2",
            "primary_metrics_3",
            "auxiliary_metrics_1",
            "auxiliary_metrics_2",
        } and all(
            _finite(value) and -EPS <= value <= 1.0 + EPS for value in replay["metrics"].values()
        )
        if not target_ok or not metric_ok:
            rules["targets_metrics"] = (
                False,
                f"{name}: target identity or metric invariant failed",
            )
        for rule in rules:
            mismatch = _checkpoint_mismatch(
                name,
                decisions["checkpoints"],
                replay["checkpoints"],
                rule,
            )
            if mismatch and rules[rule][0]:
                rules[rule] = (False, mismatch)
    atoms = [
        AtomVerdict(
            f"replay_validity.{rule}",
            passed,
            detail,
            ATOM_SHARES["replay_validity"][f"replay_validity.{rule}"],
        )
        for rule, (passed, detail) in rules.items()
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all scenario replays are required for metrics")
    return {
        metric: sum(replayed[name]["metrics"][metric] for name in SCENARIO_NAMES)
        / len(SCENARIO_NAMES)
        for metric in replayed[SCENARIO_NAMES[0]]["metrics"]
    }


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
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
        if not match:
            continue
        title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
        if title:
            headings.append((line_index, len(match.group(1)), title))

    occurrences: dict[str, list[str]] = {}
    for index, (line_index, level, title) in enumerate(headings):
        end = len(lines)
        for next_line, next_level, _ in headings[index + 1 :]:
            if next_level <= level:
                end = next_line
                break
        occurrences.setdefault(title, []).append("\n".join(lines[line_index + 1 : end]).strip())
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
            if state.official_ok:
                state.official_message = (
                    f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                    f"new_regular_results={state.result_file_is_new_regular}"
                )
            else:
                state.official_message = (
                    f"official run failed: exit={state.exit_code}, "
                    f"elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, "
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
        state.timed_out = (
            isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS)
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
