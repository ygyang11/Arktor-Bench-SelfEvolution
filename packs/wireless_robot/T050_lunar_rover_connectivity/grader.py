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
GRID_X = 24
GRID_Y = 20
HORIZON = 150
ROVERS = ("R1", "R2", "R3")
NODES = ("L", "R1", "R2", "R3")
LANDER_POSITION = (1, 9)
INITIAL_POSITIONS = {"R1": (2, 8), "R2": (2, 9), "R3": (2, 10)}
INITIAL_ENERGY = 185.0
QUEUE_CAPACITY = 8
ATTEMPT_LIMIT = 120
PACKET_AGE_LIMIT = 30
CHECKPOINT_BOUNDARIES = tuple(range(0, HORIZON + 1, 10))

SCENARIO_NAMES = (
    "ridge_shadow",
    "distributed_ice",
    "relay_saddle",
    "narrow_crater",
    "radio_dead_zone",
)
PHI = {
    "ridge_shadow": 0,
    "distributed_ice": 1,
    "relay_saddle": 2,
    "narrow_crater": 0,
    "radio_dead_zone": 1,
}
GOAL_CELLS = (
    (4, 3),
    (4, 7),
    (4, 12),
    (4, 16),
    (8, 2),
    (8, 6),
    (8, 10),
    (8, 14),
    (8, 18),
    (12, 3),
    (12, 7),
    (12, 12),
    (12, 16),
    (16, 2),
    (16, 6),
    (16, 10),
    (16, 14),
    (16, 18),
    (21, 5),
    (21, 15),
)
GOAL_IDS = tuple(f"G{index:02d}" for index in range(len(GOAL_CELLS)))
GOAL_BY_CELL = dict(zip(GOAL_CELLS, GOAL_IDS))
ACTION_DELTAS = {
    "stay": (0, 0),
    "north": (0, 1),
    "northeast": (1, 1),
    "east": (1, 0),
    "southeast": (1, -1),
    "south": (0, -1),
    "southwest": (-1, -1),
    "west": (-1, 0),
    "northwest": (-1, 1),
}
PAIRS = (
    ("L", "R1"),
    ("L", "R2"),
    ("L", "R3"),
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R3"),
)

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.moves": 0.25,
        "result_evidence.summary_requests": 0.20,
        "result_evidence.transmission_proposals": 0.20,
        "result_evidence.replay_checkpoints": 0.35,
    },
    "decision_legality": {
        "decision_legality.movement": 0.25,
        "decision_legality.summary_identity": 0.15,
        "decision_legality.summary_records": 0.20,
        "decision_legality.proposal_fifo": 0.25,
        "decision_legality.proposal_routes": 0.15,
    },
    "replay_validity": {
        "replay_validity.movement_energy": 0.20,
        "replay_validity.packet_queues": 0.20,
        "replay_validity.radio_arbitration": 0.20,
        "replay_validity.goal_knowledge": 0.20,
        "replay_validity.mission_state": 0.20,
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

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def _shape(value: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object"
    if set(value) != keys:
        return f"{path}: expected keys={sorted(keys)!r}, actual={sorted(value)!r}"
    return None


def _scenario_error(value: Any, path: str) -> str | None:
    return _shape(
        value,
        {"scenario", "moves", "summary_requests", "transmission_proposals", "replay_checkpoints"},
        path,
    )


def _move_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "rover", "action"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["rover"] not in ROVERS:
        return f"{path}.rover: expected one of {ROVERS!r}"
    if not isinstance(value["action"], str) or value["action"] not in ACTION_DELTAS:
        return f"{path}.action: expected one of {tuple(ACTION_DELTAS)!r}"
    return None


def _summary_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "source", "packet_id", "goal_ids"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1) or value["source"] not in ROVERS:
        return f"{path}: invalid slot or source"
    if not isinstance(value["packet_id"], str) or not value["packet_id"]:
        return f"{path}.packet_id: expected nonempty string"
    goals = value["goal_ids"]
    if (
        not isinstance(goals, list)
        or not 1 <= len(goals) <= 2
        or any(goal not in GOAL_IDS for goal in goals)
        or goals != sorted(set(goals))
    ):
        return f"{path}.goal_ids: expected one or two strictly increasing valid goal IDs"
    return None


def _proposal_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "src", "dst", "packet_id", "priority"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["src"] not in NODES or value["dst"] not in NODES or value["src"] == value["dst"]:
        return f"{path}.src/dst: expected two different public nodes"
    if not isinstance(value["packet_id"], str) or not value["packet_id"]:
        return f"{path}.packet_id: expected nonempty string"
    if not _integer(value["priority"], 0, 9):
        return f"{path}.priority: expected integer 0..9"
    return None


def _goal_set_error(value: Any, path: str) -> str | None:
    if not isinstance(value, list) or any(goal not in GOAL_IDS for goal in value):
        return f"{path}: expected valid goal-ID array"
    if len(value) != len(set(value)):
        return f"{path}: goal IDs must be distinct"
    return None


def _checkpoint_rover_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {"x", "y", "energy", "known_visited_goal_ids", "known_confirmed_goal_ids"},
        path,
    )
    if error:
        return error
    if not _integer(value["x"], 0, GRID_X - 1) or not _integer(value["y"], 0, GRID_Y - 1):
        return f"{path}.x/y: expected public grid coordinates"
    if not _finite(value["energy"]) or not -EPS <= float(value["energy"]) <= INITIAL_ENERGY + EPS:
        return f"{path}.energy: expected finite value in [0,{INITIAL_ENERGY:g}]"
    for field in ("known_visited_goal_ids", "known_confirmed_goal_ids"):
        error = _goal_set_error(value[field], f"{path}.{field}")
        if error:
            return error
    return None


def _checkpoint_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {
            "boundary",
            "R1",
            "R2",
            "R3",
            "queue_L",
            "queue_R1",
            "queue_R2",
            "queue_R3",
            "visited_goal_ids",
            "confirmed_goal_ids",
            "duplicate_visit_count",
            "selected_attempt_count",
            "terrain_risk",
        },
        path,
    )
    if error:
        return error
    if not _integer(value["boundary"], 0, HORIZON) or value[
        "boundary"
    ] not in CHECKPOINT_BOUNDARIES:
        return f"{path}.boundary: expected one of {CHECKPOINT_BOUNDARIES!r}"
    for rover in ROVERS:
        error = _checkpoint_rover_error(value[rover], f"{path}.{rover}")
        if error:
            return error
    for node in NODES:
        queue = value[f"queue_{node}"]
        if (
            not isinstance(queue, list)
            or len(queue) > QUEUE_CAPACITY
            or any(not isinstance(packet_id, str) or not packet_id for packet_id in queue)
            or len(queue) != len(set(queue))
        ):
            return f"{path}.queue_{node}: expected at most eight distinct packet IDs"
    for field in ("visited_goal_ids", "confirmed_goal_ids"):
        error = _goal_set_error(value[field], f"{path}.{field}")
        if error:
            return error
    if not _integer(value["duplicate_visit_count"], 0, len(ROVERS) * HORIZON):
        return f"{path}.duplicate_visit_count: expected integer 0..{len(ROVERS) * HORIZON}"
    if not _integer(value["selected_attempt_count"], 0, ATTEMPT_LIMIT):
        return f"{path}.selected_attempt_count: expected integer 0..{ATTEMPT_LIMIT}"
    if not _finite(value["terrain_risk"]) or not -EPS <= float(value["terrain_risk"]) <= 1800 + EPS:
        return f"{path}.terrain_risk: expected finite value in [0,1800]"
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
        "moves": [True, "move fields are well formed"],
        "summary_requests": [True, "summary-request fields are well formed"],
        "transmission_proposals": [True, "proposal fields are well formed"],
        "replay_checkpoints": [True, "checkpoint fields are well formed"],
    }

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    validators = {
        "moves": _move_error,
        "summary_requests": _summary_error,
        "transmission_proposals": _proposal_error,
        "replay_checkpoints": _checkpoint_error,
    }
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            continue
        for field, validator in validators.items():
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
    expected_moves = {(slot, rover) for slot in range(HORIZON) for rover in ROVERS}
    for name in SCENARIO_NAMES:
        matches = [
            row for row in scenario_rows if isinstance(row, dict) and row.get("scenario") == name
        ]
        if not matches:
            continue
        record = matches[0]
        moves = record.get("moves")
        if isinstance(moves, list):
            keys = [
                _hashable((row.get("slot"), row.get("rover")))
                if isinstance(row, dict)
                else _hashable(row)
                for row in moves
            ]
            defects.extend(_key_defects(f"{name}.moves", keys, expected_moves))
        else:
            defects.append(f"{name}.moves: missing array")
        checkpoints = record.get("replay_checkpoints")
        if isinstance(checkpoints, list):
            keys = [
                _hashable(row.get("boundary")) if isinstance(row, dict) else _hashable(row)
                for row in checkpoints
            ]
            defects.extend(
                _key_defects(f"{name}.replay_checkpoints", keys, set(CHECKPOINT_BOUNDARIES))
            )
        else:
            defects.append(f"{name}.replay_checkpoints: missing array")
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _decision_map(
    record: Mapping[str, Any], *, require_checkpoints: bool = True
) -> dict[str, Any] | None:
    fields = ("moves", "summary_requests", "transmission_proposals")
    if any(not isinstance(record.get(field), list) for field in fields):
        return None
    moves: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["moves"]:
        if not isinstance(row, dict) or _move_error(row, "move"):
            return None
        key = (row["slot"], row["rover"])
        if key in moves:
            return None
        moves[key] = row
    if set(moves) != {(slot, rover) for slot in range(HORIZON) for rover in ROVERS}:
        return None
    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["summary_requests"]:
        if not isinstance(row, dict) or _summary_error(row, "summary_request"):
            return None
        key = (row["slot"], row["source"])
        if key in summaries:
            return None
        summaries[key] = row
    proposals: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["transmission_proposals"]:
        if not isinstance(row, dict) or _proposal_error(row, "proposal"):
            return None
        key = (row["slot"], row["src"])
        if key in proposals:
            return None
        proposals[key] = row
    result: dict[str, Any] = {"moves": moves, "summaries": summaries, "proposals": proposals}
    if require_checkpoints:
        rows = record.get("replay_checkpoints")
        if not isinstance(rows, list):
            return None
        checkpoints: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or _checkpoint_error(row, "checkpoint"):
                return None
            if row["boundary"] in checkpoints:
                return None
            checkpoints[row["boundary"]] = row
        if set(checkpoints) != set(CHECKPOINT_BOUNDARIES):
            return None
        result["checkpoints"] = checkpoints
    return result


def _elevation(name: str, cell: tuple[int, int]) -> int:
    x, y = cell
    base = (x + 2 * y + PHI[name]) % 3
    extra = 0
    if name == "ridge_shadow" and x in {10, 11} and y not in {8, 9, 10}:
        extra = 4
    elif name == "relay_saddle" and x in {10, 11, 12, 13} and y not in {8, 9, 10, 11}:
        extra = 4
    elif (
        name == "narrow_crater"
        and 16 <= (x - 12) ** 2 + (y - 10) ** 2 <= 36
        and y not in {9, 10}
    ):
        extra = 3
    return base + extra


def _goal_value(name: str, goal_id: str) -> float:
    index = GOAL_IDS.index(goal_id)
    return float(1 + ((index + PHI[name]) % 5))


def _visit_deadline(goal_id: str) -> int:
    return 70 + 4 * (GOAL_IDS.index(goal_id) % 10)


def _confirm_deadline(goal_id: str) -> int:
    return min(149, _visit_deadline(goal_id) + 24)


def _round_half_down(numerator: int, denominator: int = 8) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient if 2 * remainder <= denominator else quotient + 1


def _node_position(node: str, positions: Mapping[str, tuple[int, int]]) -> tuple[int, int]:
    return LANDER_POSITION if node == "L" else positions[node]


def _pair_key(src: str, dst: str) -> tuple[str, str]:
    for pair in PAIRS:
        if {src, dst} == set(pair):
            return pair
    raise ValueError(f"unknown node pair {src!r},{dst!r}")


def _margin(
    name: str,
    src: str,
    dst: str,
    positions: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    p_u = _node_position(src, positions)
    p_v = _node_position(dst, positions)
    distance = math.hypot(p_u[0] - p_v[0], p_u[1] - p_v[1])
    z_u = _elevation(name, p_u)
    z_v = _elevation(name, p_v)
    excess = 0.0
    dead_samples = 0
    for sample_index in range(1, 8):
        x_num = (8 - sample_index) * p_u[0] + sample_index * p_v[0]
        y_num = (8 - sample_index) * p_u[1] + sample_index * p_v[1]
        sample = (_round_half_down(x_num), _round_half_down(y_num))
        line_height = ((8 - sample_index) * z_u + sample_index * z_v) / 8.0
        excess = max(excess, _elevation(name, sample) - line_height)
        if (
            name == "radio_dead_zone"
            and 14 <= sample[0] <= 19
            and 4 <= sample[1] <= 15
        ):
            dead_samples += 1
    value = 8.0 - 0.45 * distance - 1.25 * excess - 0.6 * dead_samples
    return {
        "pair": "-".join(_pair_key(src, dst)),
        "distance": distance,
        "elevation_excess": excess,
        "dead_zone_samples": dead_samples,
        "margin": value,
        "available": value >= -EPS,
    }


def _local_views(
    name: str,
    slot: int,
    positions: Mapping[str, tuple[int, int]],
    energies: Mapping[str, float],
    queues: Mapping[str, list[str]],
    packets: Mapping[str, dict[str, Any]],
    known_visited: Mapping[str, set[str]],
    known_confirmed: Mapping[str, set[str]],
    goal_state: Mapping[str, dict[str, Any]],
    margins: Mapping[tuple[str, str], float],
    selected_count: int,
    collision_feedback: Mapping[str, bool],
    summary_feedback: Sequence[Mapping[str, Any]],
    proposal_feedback: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    node_views: dict[str, dict[str, Any]] = {}
    for node in NODES:
        queue_packets = [
            {
                "packet_id": packet_id,
                "type": packets[packet_id]["type"],
                "goal_ids": list(packets[packet_id]["goal_ids"]),
                "destination": packets[packet_id]["destination"],
                "created_slot": packets[packet_id]["created_slot"],
                "path": list(packets[packet_id]["path"]),
            }
            for packet_id in queues[node]
        ]
        view: dict[str, Any] = {
            "node": node,
            "slot": slot,
            "queue": queue_packets,
            "incident_margins": {
                other: margins[_pair_key(node, other)] for other in NODES if other != node
            },
            "remaining_attempts": ATTEMPT_LIMIT - selected_count,
            "own_proposal_feedback": [
                dict(item) for item in proposal_feedback if item["src"] == node
            ],
        }
        if node in ROVERS:
            view.update(
                {
                    "position": positions[node],
                    "energy": energies[node],
                    "collision_blocked_last_slot": collision_feedback[node],
                    "known_visited_goal_ids": sorted(known_visited[node]),
                    "known_confirmed_goal_ids": sorted(known_confirmed[node]),
                    "own_summary_feedback": [
                        dict(item) for item in summary_feedback if item["source"] == node
                    ],
                    "own_first_visit_records": [
                        {
                            "goal_id": goal_id,
                            "visit_slot": state["visit_slot"],
                            "packetized": state["packetized"],
                        }
                        for goal_id, state in goal_state.items()
                        if state["first_visitor"] == node
                    ],
                }
            )
        else:
            view["confirmed_goal_ids"] = sorted(
                goal_id
                for goal_id, state in goal_state.items()
                if state["confirm_slot"] is not None
            )
        node_views[node] = view
    return {"scenario": name, "slot": slot, "node_views": node_views}


def _scenario_metrics(name: str, replay: Mapping[str, Any]) -> dict[str, float]:
    goals = replay["goal_state"]
    total_value = 60.0
    confirm_value = sum(
        _goal_value(name, goal_id)
        for goal_id, state in goals.items()
        if state["confirm_slot"] is not None
        and state["confirm_slot"] <= _confirm_deadline(goal_id)
    )
    visit_value = sum(
        _goal_value(name, goal_id)
        for goal_id, state in goals.items()
        if state["visit_slot"] is not None and state["visit_slot"] <= _visit_deadline(goal_id)
    )
    primary_1 = 0.75 * confirm_value / total_value + 0.25 * visit_value / total_value

    visited = [goal_id for goal_id, state in goals.items() if state["visit_slot"] is not None]
    visited_value = sum(_goal_value(name, goal_id) for goal_id in visited)
    if not visited:
        complete = 0.0
        fresh = 0.0
    else:
        complete = sum(
            _goal_value(name, goal_id)
            for goal_id in visited
            if goals[goal_id]["confirm_slot"] is not None
            and goals[goal_id]["confirm_slot"] <= _confirm_deadline(goal_id)
        ) / visited_value
        fresh = sum(
            _goal_value(name, goal_id)
            * max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (goals[goal_id]["confirm_slot"] - goals[goal_id]["visit_slot"]) / 30.0,
                ),
            )
            for goal_id in visited
            if goals[goal_id]["confirm_slot"] is not None
            and goals[goal_id]["confirm_slot"] <= _confirm_deadline(goal_id)
        ) / visited_value
    primary_2 = 0.6 * complete + 0.4 * fresh

    progress = len(visited) / len(GOAL_IDS)
    duplicate = replay["duplicate_visits"]
    visits = len(visited) + duplicate
    energy_used = len(ROVERS) * INITIAL_ENERGY - sum(replay["energies"].values())
    primary_3 = progress * (
        0.4 * max(0.0, min(1.0, 1.0 - replay["terrain_risk"] / 1800.0))
        + 0.3 * (1.0 - duplicate / max(1, visits))
        + 0.3 * max(0.0, min(1.0, 1.0 - energy_used / 555.0))
    )

    attempts = replay["attempts"]
    packets = replay["packets"]
    useful_hops = sum(
        1
        for attempt in attempts
        if attempt["outcome"] == "delivered"
        and packets[attempt["packet_id"]]["type"] == "summary"
        and (
            bool(packets[attempt["packet_id"]]["confirmed_on_time_goals"])
            or attempt.get("new_visit_knowledge", False)
        )
    )
    generated = [packet for packet in packets.values() if packet["type"] == "summary"]
    on_time = [packet for packet in generated if packet["confirmed_on_time_goals"]]
    auxiliary_1 = (
        0.6 * (useful_hops / len(attempts) if attempts else 0.0)
        + 0.4 * (len(on_time) / len(generated) if generated else 0.0)
    )

    on_time_goals = {
        goal_id
        for packet in on_time
        for goal_id in packet["confirmed_on_time_goals"]
    }
    qualifying_packet_ids = {packet["packet_id"] for packet in on_time}
    qualifying_packet_ids.update(f"ack:{goal_id}" for goal_id in on_time_goals)
    hop_samples = [
        attempt
        for attempt in attempts
        if attempt["outcome"] == "delivered" and attempt["packet_id"] in qualifying_packet_ids
    ]
    if hop_samples:
        margin_score = sum(
            max(0.0, min(1.0, (attempt["margin"] + 4.0) / 8.0))
            for attempt in hop_samples
        ) / len(hop_samples)
        wait_score = max(
            0.0,
            min(
                1.0,
                1.0
                - sum(attempt["queue_wait"] for attempt in hop_samples)
                / len(hop_samples)
                / 30.0,
            ),
        )
    else:
        margin_score = 0.0
        wait_score = 0.0
    if on_time_goals:
        ages = [
            goals[goal_id]["confirm_slot"] - goals[goal_id]["visit_slot"]
            for goal_id in on_time_goals
        ]
        age_score = max(0.0, min(1.0, 1.0 - sum(ages) / len(ages) / 30.0))
    else:
        age_score = 0.0
    auxiliary_2 = 0.4 * margin_score + 0.3 * wait_score + 0.3 * age_score

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
    provider: Callable[[int, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if name not in SCENARIO_NAMES or (decisions is None) == (provider is None):
        raise ValueError("provide exactly one valid scenario and decision source")
    positions = dict(INITIAL_POSITIONS)
    energies = {rover: INITIAL_ENERGY for rover in ROVERS}
    queues: dict[str, list[str]] = {node: [] for node in NODES}
    packets: dict[str, dict[str, Any]] = {}
    pending_arrival: dict[int, tuple[str, str, dict[str, Any]]] = {}
    ack_due: dict[int, list[str]] = {}
    goal_state = {
        goal_id: {
            "first_visitor": None,
            "visit_slot": None,
            "confirm_slot": None,
            "packetized": False,
        }
        for goal_id in GOAL_IDS
    }
    known_visited = {rover: set() for rover in ROVERS}
    known_confirmed = {rover: set() for rover in ROVERS}
    requested_packet_ids: set[str] = set()
    summary_feedback: list[dict[str, Any]] = []
    proposal_feedback: list[dict[str, Any]] = []
    collision_feedback = {rover: False for rover in ROVERS}
    duplicate_visits = 0
    selected_count = 0
    terrain_risk = 0.0
    collision_blocks = 0
    attempts: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    violations: dict[str, list[str]] = {
        "movement": [],
        "summary_identity": [],
        "summary_records": [],
        "proposal_fifo": [],
        "proposal_routes": [],
    }

    def fail(rule: str, detail: str) -> None:
        violations[rule].append(detail)

    def all_margins() -> dict[tuple[str, str], float]:
        return {
            pair: _margin(name, pair[0], pair[1], positions)["margin"] for pair in PAIRS
        }

    def record_checkpoint(boundary: int) -> None:
        checkpoints.append(
            {
                "boundary": boundary,
                **{
                    rover: {
                        "x": positions[rover][0],
                        "y": positions[rover][1],
                        "energy": energies[rover],
                        "known_visited_goal_ids": sorted(known_visited[rover]),
                        "known_confirmed_goal_ids": sorted(known_confirmed[rover]),
                    }
                    for rover in ROVERS
                },
                **{f"queue_{node}": list(queues[node]) for node in NODES},
                "visited_goal_ids": sorted(
                    goal_id
                    for goal_id, state in goal_state.items()
                    if state["visit_slot"] is not None
                ),
                "confirmed_goal_ids": sorted(
                    goal_id
                    for goal_id, state in goal_state.items()
                    if state["confirm_slot"] is not None
                ),
                "duplicate_visit_count": duplicate_visits,
                "selected_attempt_count": selected_count,
                "terrain_risk": terrain_risk,
            }
        )

    def process_arrival(boundary: int) -> None:
        arrival = pending_arrival.pop(boundary, None)
        if arrival is None:
            return
        packet_id, dst, attempt = arrival
        packet = packets[packet_id]
        if packet["type"] == "summary" and dst in ROVERS:
            before = set(known_visited[dst])
            known_visited[dst].update(packet["goal_ids"])
            attempt["new_visit_knowledge"] = bool(known_visited[dst] - before)
        elif packet["type"] == "ack" and dst in ROVERS:
            known_confirmed[dst].update(packet["goal_ids"])
        if dst == packet["destination"]:
            packet["status"] = "delivered"
            packet["arrival_slot"] = boundary
            if packet["type"] == "summary":
                for goal_id in packet["goal_ids"]:
                    state = goal_state[goal_id]
                    if state["confirm_slot"] is not None:
                        continue
                    state["confirm_slot"] = boundary
                    if boundary <= _confirm_deadline(goal_id):
                        packet["confirmed_on_time_goals"].append(goal_id)
                    if boundary + 1 <= HORIZON - 1:
                        ack_due.setdefault(boundary + 1, []).append(goal_id)
            return
        if len(queues[dst]) < QUEUE_CAPACITY:
            queues[dst].append(packet_id)
            packet["queue_enter_slot"] = boundary
            packet["status"] = "queued"
        else:
            packet["status"] = "dropped"

    def create_ack(goal_id: str, slot: int) -> None:
        packet_id = f"ack:{goal_id}"
        packet = {
            "packet_id": packet_id,
            "type": "ack",
            "origin": "L",
            "destination": goal_state[goal_id]["first_visitor"],
            "goal_ids": [goal_id],
            "bits": 64,
            "created_slot": slot,
            "queue_enter_slot": slot,
            "path": ["L"],
            "status": "queued",
            "arrival_slot": None,
            "confirmed_on_time_goals": [],
        }
        packets[packet_id] = packet
        if len(queues["L"]) < QUEUE_CAPACITY:
            queues["L"].append(packet_id)
        else:
            packet["status"] = "dropped"

    for slot in range(HORIZON):
        for node in NODES:
            remaining: list[str] = []
            for packet_id in queues[node]:
                packet = packets[packet_id]
                if slot > packet["created_slot"] + PACKET_AGE_LIMIT:
                    packet["status"] = "expired"
                else:
                    remaining.append(packet_id)
            queues[node] = remaining
        process_arrival(slot)
        for goal_id in sorted(ack_due.pop(slot, [])):
            create_ack(goal_id, slot)
        margins = all_margins()

        if slot in CHECKPOINT_BOUNDARIES:
            record_checkpoint(slot)
        view = _local_views(
            name,
            slot,
            positions,
            energies,
            queues,
            packets,
            known_visited,
            known_confirmed,
            goal_state,
            margins,
            selected_count,
            collision_feedback,
            summary_feedback,
            proposal_feedback,
        )
        if provider is not None:
            cycle_summaries = {}
            for rover in ROVERS:
                supplied = provider(slot, "summary_request", view["node_views"][rover])
                request = supplied.get("summary_request")
                if request is not None:
                    cycle_summaries[rover] = request
        else:
            assert decisions is not None
            cycle_summaries = {
                source: row
                for (row_slot, source), row in decisions["summaries"].items()
                if row_slot == slot
            }

        for source in ROVERS:
            supplied = cycle_summaries.get(source)
            if supplied is None:
                continue
            if provider is not None:
                row = {
                    "slot": slot,
                    "source": source,
                    "packet_id": supplied.get("packet_id"),
                    "goal_ids": list(supplied.get("goal_ids", [])),
                }
            else:
                row = dict(supplied)
            summary_rows.append(row)
            packet_id = row["packet_id"]
            goal_ids = list(row["goal_ids"])
            identity_ok = packet_id not in requested_packet_ids and not packet_id.startswith("ack:")
            if not identity_ok:
                fail(
                    "summary_identity",
                    f"{name}.summary_requests[{slot},{source}].packet_id: duplicate or reserved ID",
                )
            requested_packet_ids.add(packet_id)
            records_ok = (
                goal_ids == sorted(set(goal_ids))
                and all(
                    goal_state[goal_id]["first_visitor"] == source
                    and goal_state[goal_id]["visit_slot"] is not None
                    and goal_state[goal_id]["visit_slot"] < slot
                    and not goal_state[goal_id]["packetized"]
                    and goal_id not in known_confirmed[source]
                    for goal_id in goal_ids
                )
            )
            if not records_ok:
                fail(
                    "summary_records",
                    f"{name}.summary_requests[{slot},{source}]: "
                    "records must be eligible own records",
                )
            if not identity_ok or not records_ok:
                summary_feedback.append(
                    {"slot": slot, "source": source, "packet_id": packet_id, "outcome": "invalid"}
                )
                continue
            if len(queues[source]) >= QUEUE_CAPACITY:
                summary_feedback.append(
                    {"slot": slot, "source": source, "packet_id": packet_id, "outcome": "dropped"}
                )
                continue
            packet = {
                "packet_id": packet_id,
                "type": "summary",
                "origin": source,
                "destination": "L",
                "goal_ids": goal_ids,
                "bits": 96 if len(goal_ids) == 1 else 128,
                "created_slot": slot,
                "queue_enter_slot": slot,
                "path": [source],
                "status": "queued",
                "arrival_slot": None,
                "confirmed_on_time_goals": [],
            }
            packets[packet_id] = packet
            queues[source].append(packet_id)
            for goal_id in goal_ids:
                goal_state[goal_id]["packetized"] = True
            summary_feedback.append(
                {"slot": slot, "source": source, "packet_id": packet_id, "outcome": "admitted"}
            )

        updated_view = _local_views(
            name,
            slot,
            positions,
            energies,
            queues,
            packets,
            known_visited,
            known_confirmed,
            goal_state,
            margins,
            selected_count,
            collision_feedback,
            summary_feedback,
            proposal_feedback,
        )
        if provider is not None:
            chosen_moves = {}
            cycle_proposals = {}
            for node in NODES:
                supplied = provider(
                    slot, "move_and_proposal", updated_view["node_views"][node]
                )
                if node in ROVERS:
                    chosen_moves[node] = supplied.get("move", "stay")
                proposal = supplied.get("proposal")
                if proposal is not None:
                    cycle_proposals[node] = proposal
        else:
            assert decisions is not None
            chosen_moves = {
                rover: decisions["moves"][(slot, rover)]["action"] for rover in ROVERS
            }
            cycle_proposals = {
                src: row
                for (row_slot, src), row in decisions["proposals"].items()
                if row_slot == slot
            }

        for rover in ROVERS:
            move_rows.append({"slot": slot, "rover": rover, "action": chosen_moves[rover]})

        valid_proposals: list[dict[str, Any]] = []
        for src in NODES:
            supplied = cycle_proposals.get(src)
            if supplied is None:
                continue
            if provider is not None:
                row = {
                    "slot": slot,
                    "src": src,
                    "dst": supplied.get("dst"),
                    "packet_id": supplied.get("packet_id"),
                    "priority": supplied.get("priority"),
                }
            else:
                row = dict(supplied)
            proposal_rows.append(row)
            packet_id = row["packet_id"]
            fifo_ok = bool(queues[src]) and queues[src][0] == packet_id
            if not fifo_ok:
                fail(
                    "proposal_fifo",
                    f"{name}.transmission_proposals[{slot},{src}]: "
                    "packet must be current FIFO head",
                )
                proposal_feedback.append({**row, "outcome": "invalid"})
                continue
            packet = packets[packet_id]
            route_ok = (
                slot <= min(HORIZON - 1, packet["created_slot"] + PACKET_AGE_LIMIT)
                and row["dst"] not in packet["path"]
                and len(packet["path"]) - 1 < 3
                and (len(packet["path"]) - 1 < 2 or row["dst"] == packet["destination"])
            )
            if not route_ok:
                fail(
                    "proposal_routes",
                    f"{name}.transmission_proposals[{slot},{src}]: packet age or "
                    "destination violates the attempt, no-repeat, or three-hop rule",
                )
                proposal_feedback.append({**row, "outcome": "invalid"})
                continue
            valid_proposals.append(row)

        selected: dict[str, Any] | None = None
        if selected_count < ATTEMPT_LIMIT and valid_proposals:
            selected = min(
                valid_proposals,
                key=lambda row: (-row["priority"], row["src"], row["dst"], row["packet_id"]),
            )
            selected_count += 1
            packet = packets[selected["packet_id"]]
            pair = _pair_key(selected["src"], selected["dst"])
            margin = margins[pair]
            attempt = {
                "slot": slot,
                "src": selected["src"],
                "dst": selected["dst"],
                "packet_id": selected["packet_id"],
                "margin": margin,
                "queue_wait": slot - packet["queue_enter_slot"],
                "outcome": "delivered" if margin >= -EPS else "failed",
                "new_visit_knowledge": False,
            }
            attempts.append(attempt)
            if margin >= -EPS:
                if not queues[selected["src"]] or queues[selected["src"]][0] != packet["packet_id"]:
                    raise ValueError("selected packet is no longer FIFO head")
                queues[selected["src"]].pop(0)
                packet["path"].append(selected["dst"])
                packet["status"] = "in_flight"
                pending_arrival[slot + 1] = (packet["packet_id"], selected["dst"], attempt)
        for proposal in valid_proposals:
            outcome = (
                attempts[-1]["outcome"]
                if selected is proposal
                else "not_selected"
            )
            proposal_feedback.append({**proposal, "outcome": outcome})

        old_positions = dict(positions)
        proposed: dict[str, tuple[int, int]] = {}
        moving: dict[str, bool] = {}
        for rover in ROVERS:
            action = chosen_moves[rover]
            dx, dy = ACTION_DELTAS[action]
            old = old_positions[rover]
            candidate = (old[0] + dx, old[1] + dy)
            cost = 0.05
            legal = action == "stay"
            if action != "stay":
                in_grid = 0 <= candidate[0] < GRID_X and 0 <= candidate[1] < GRID_Y
                slope_ok = in_grid and abs(_elevation(name, candidate) - _elevation(name, old)) <= 2
                if in_grid:
                    cost = 1.0 + 0.25 * max(
                        0, _elevation(name, candidate) - _elevation(name, old)
                    )
                legal = in_grid and slope_ok
            legal = legal and energies[rover] + EPS >= cost
            if not legal:
                fail(
                    "movement",
                    f"{name}.moves[{slot},{rover}]: action violates grid, slope or energy",
                )
                proposed[rover] = old
                moving[rover] = False
                energies[rover] -= min(0.05, max(0.0, energies[rover]))
                continue
            energies[rover] = max(0.0, energies[rover] - cost)
            proposed[rover] = candidate
            moving[rover] = action != "stay"
            if action != "stay":
                terrain_risk += max(
                    0, _elevation(name, candidate) - _elevation(name, old)
                ) ** 2

        blocked: set[str] = set()
        for rover in ROVERS:
            if any(
                other != rover and proposed[other] == proposed[rover] for other in ROVERS
            ):
                blocked.add(rover)
            if any(
                other != rover
                and proposed[rover] == old_positions[other]
                and proposed[other] == old_positions[rover]
                and old_positions[rover] != old_positions[other]
                for other in ROVERS
            ):
                blocked.add(rover)
        changed = True
        while changed:
            changed = False
            retained_old = {old_positions[rover] for rover in blocked}
            for rover in ROVERS:
                if rover not in blocked and moving[rover] and proposed[rover] in retained_old:
                    blocked.add(rover)
                    changed = True
        positions = {
            rover: old_positions[rover] if rover in blocked else proposed[rover] for rover in ROVERS
        }
        collision_feedback = {rover: rover in blocked for rover in ROVERS}
        collision_blocks += len(blocked)

        entries: dict[str, list[str]] = {}
        for rover in ROVERS:
            goal_id = GOAL_BY_CELL.get(positions[rover])
            if goal_id is not None and old_positions[rover] != positions[rover]:
                entries.setdefault(goal_id, []).append(rover)
        for goal_id, visitors in entries.items():
            state = goal_state[goal_id]
            if state["visit_slot"] is None:
                first = min(visitors, key=ROVERS.index)
                state["first_visitor"] = first
                state["visit_slot"] = slot
                known_visited[first].add(goal_id)
                duplicate_visits += len(visitors) - 1
            else:
                duplicate_visits += len(visitors)

    process_arrival(HORIZON)
    record_checkpoint(HORIZON)
    replay = {
        "scenario": name,
        "positions": positions,
        "energies": energies,
        "queues": queues,
        "packets": packets,
        "attempts": attempts,
        "goal_state": goal_state,
        "known_visited": known_visited,
        "known_confirmed": known_confirmed,
        "duplicate_visits": duplicate_visits,
        "selected_attempts": selected_count,
        "terrain_risk": terrain_risk,
        "collision_blocks": collision_blocks,
        "checkpoints": checkpoints,
        "violations": violations,
        "summary_feedback": summary_feedback,
        "proposal_feedback": proposal_feedback,
    }
    replay["record"] = {
        "scenario": name,
        "moves": move_rows,
        "summary_requests": summary_rows,
        "transmission_proposals": proposal_rows,
        "replay_checkpoints": checkpoints,
    }
    return replay


def _dynamic_key_defect(records: Mapping[str, dict[str, Any]], field: str) -> str | None:
    key_fields = ("slot", "source") if field == "summary_requests" else ("slot", "src")
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None or not isinstance(record.get(field), list):
            continue
        seen: set[tuple[Any, Any]] = set()
        for index, row in enumerate(record[field]):
            if not isinstance(row, dict):
                continue
            key = (row.get(key_fields[0]), row.get(key_fields[1]))
            if key in seen:
                return f"{name}.{field}[{index}]: duplicate node-slot key {key!r}"
            seen.add(key)
    return None


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules: dict[str, tuple[bool, str]] = {
        "movement": (True, "moves satisfy grid, slope and energy constraints"),
        "summary_identity": (True, "summary node-slot and packet IDs are unique"),
        "summary_records": (True, "summary records are eligible original records"),
        "proposal_fifo": (True, "proposals use one local FIFO head per node-slot"),
        "proposal_routes": (True, "proposal destinations satisfy no-repeat and hop limits"),
    }
    summary_duplicate = _dynamic_key_defect(records, "summary_requests")
    proposal_duplicate = _dynamic_key_defect(records, "transmission_proposals")
    if summary_duplicate:
        rules["summary_identity"] = (False, summary_duplicate)
    if proposal_duplicate:
        rules["proposal_fifo"] = (False, proposal_duplicate)
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
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        for rule, violations in replay["violations"].items():
            if violations and rules[rule][0]:
                rules[rule] = (False, violations[0])
    return [
        AtomVerdict(
            f"decision_legality.{rule}",
            passed,
            detail,
            ATOM_SHARES["decision_legality"][f"decision_legality.{rule}"],
        )
        for rule, (passed, detail) in rules.items()
    ]


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

    def unordered(path: str, expected: Any, actual: Any) -> str | None:
        return exact(path, sorted(expected), sorted(actual))

    def real(path: str, expected: Any, actual: Any) -> str | None:
        if not _finite(actual) or not math.isclose(
            float(actual), float(expected), abs_tol=EPS, rel_tol=0.0
        ):
            return (
                f"{name}.replay_checkpoints[{path}]: expected={expected!r}, "
                f"actual={actual!r}, abs_tol={EPS}"
            )
        return None

    for boundary in CHECKPOINT_BOUNDARIES:
        expected = expected_by_boundary[boundary]
        actual = submitted[boundary]
        prefix = f"boundary={boundary}"
        if rule == "movement_energy":
            for rover in ROVERS:
                for field in ("x", "y", "energy"):
                    path = f"{prefix}.{rover}.{field}"
                    mismatch = (
                        real(path, expected[rover][field], actual[rover][field])
                        if field == "energy"
                        else exact(path, expected[rover][field], actual[rover][field])
                    )
                    if mismatch:
                        return mismatch
            mismatch = real(
                f"{prefix}.terrain_risk", expected["terrain_risk"], actual["terrain_risk"]
            )
            if mismatch:
                return mismatch
        elif rule == "packet_queues":
            for node in NODES:
                mismatch = exact(
                    f"{prefix}.queue_{node}", expected[f"queue_{node}"], actual[f"queue_{node}"]
                )
                if mismatch:
                    return mismatch
        elif rule == "radio_arbitration":
            mismatch = exact(
                f"{prefix}.selected_attempt_count",
                expected["selected_attempt_count"],
                actual["selected_attempt_count"],
            )
            if mismatch:
                return mismatch
        elif rule == "goal_knowledge":
            for rover in ROVERS:
                for field in ("known_visited_goal_ids", "known_confirmed_goal_ids"):
                    mismatch = unordered(
                        f"{prefix}.{rover}.{field}",
                        expected[rover][field],
                        actual[rover][field],
                    )
                    if mismatch:
                        return mismatch
        elif rule == "mission_state":
            for field in ("visited_goal_ids", "confirmed_goal_ids"):
                mismatch = unordered(
                    f"{prefix}.{field}", expected[field], actual[field]
                )
                if mismatch:
                    return mismatch
            mismatch = exact(
                f"{prefix}.duplicate_visit_count",
                expected["duplicate_visit_count"],
                actual["duplicate_visit_count"],
            )
            if mismatch:
                return mismatch
        else:
            raise ValueError(f"unknown replay checkpoint rule {rule!r}")
    return None


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules: dict[str, tuple[bool, str]] = {
        "movement_energy": (True, "movement, collision, energy and terrain risk replayed"),
        "packet_queues": (True, "packet creation, expiry, custody and FIFO queues replayed"),
        "radio_arbitration": (True, "radio margins, arbitration and attempts replayed"),
        "goal_knowledge": (True, "received visit and ACK knowledge replayed"),
        "mission_state": (True, "visits, confirmations and duplicates replayed"),
    }
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            detail = f"{name}: missing scenario record; checkpoints unavailable"
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        decisions = _decision_map(record)
        if decisions is None:
            detail = f"{name}: required decision or checkpoint map is unusable"
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
        if not all(
            0 <= position[0] < GRID_X
            and 0 <= position[1] < GRID_Y
            and -EPS <= replay["energies"][rover] <= INITIAL_ENERGY + EPS
            for rover, position in replay["positions"].items()
        ):
            rules["movement_energy"] = (False, f"{name}: derived movement invariant failed")
        if any(len(queue) > QUEUE_CAPACITY for queue in replay["queues"].values()):
            rules["packet_queues"] = (False, f"{name}: derived queue capacity invariant failed")
        if not 0 <= replay["selected_attempts"] <= ATTEMPT_LIMIT:
            rules["radio_arbitration"] = (False, f"{name}: attempt budget invariant failed")
        if any(
            not replay["known_confirmed"][rover].issubset(
                {
                    goal_id
                    for goal_id, state in replay["goal_state"].items()
                    if state["confirm_slot"] is not None
                }
            )
            for rover in ROVERS
        ):
            rules["goal_knowledge"] = (
                False,
                f"{name}: local confirmation knowledge invariant failed",
            )
        for rule in rules:
            mismatch = _checkpoint_mismatch(
                name, decisions["checkpoints"], replay["checkpoints"], rule
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
    scenario_metrics = {
        name: _scenario_metrics(name, replayed[name]) for name in SCENARIO_NAMES
    }
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
                headings.append(
                    (line_index, line_index + 1, len(match.group(1)), title)
                )
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
        title = previous_stripped
        if title:
            level = 1 if setext.group(1)[0] == "=" else 2
            headings.append((line_index - 1, line_index + 1, level, title))
    occurrences: dict[str, list[str]] = {}
    for index, (line_index, content_start, level, title) in enumerate(headings):
        end = len(lines)
        for next_line, _, next_level, _ in headings[index + 1 :]:
            if next_level <= level:
                end = next_line
                break
        occurrences.setdefault(title, []).append(
            "\n".join(lines[content_start:end]).strip()
        )
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
