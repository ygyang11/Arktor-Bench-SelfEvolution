from __future__ import annotations

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
GRID_X = 18
GRID_Y = 14
HORIZON = 120
NODES = ("R1", "R2", "R3", "E")
ROVERS = ("R1", "R2", "R3")
EDGE_POSITION = (16, 7)
INITIAL_ENERGY = 180.0
QUEUE_CAPACITY = 24
ATTEMPT_LIMIT = 160
AGE_LIMIT = 40
CHECKPOINT_BOUNDARIES = tuple(range(0, HORIZON + 1, 10))

SCENARIO_NAMES = (
    "relay_gap",
    "science_cluster",
    "compute_imbalance",
    "contact_outage",
    "deadline_chain",
)
SCENARIO_INDEX = {name: index for index, name in enumerate(SCENARIO_NAMES)}
STARTS = {
    "relay_gap": {"R1": (1, 2), "R2": (1, 7), "R3": (1, 12)},
    "science_cluster": {"R1": (2, 3), "R2": (2, 7), "R3": (2, 11)},
    "compute_imbalance": {"R1": (1, 1), "R2": (1, 6), "R3": (1, 11)},
    "contact_outage": {"R1": (2, 2), "R2": (2, 7), "R3": (2, 12)},
    "deadline_chain": {"R1": (1, 4), "R2": (1, 7), "R3": (1, 10)},
}
OBSTACLE_RECTS = {
    "relay_gap": ((6, 7, 0, 5), (6, 7, 9, 13)),
    "science_cluster": ((7, 10, 5, 8), (12, 13, 0, 4)),
    "compute_imbalance": ((5, 6, 0, 4), (5, 6, 9, 13)),
    "contact_outage": ((8, 9, 0, 5), (8, 9, 9, 13)),
    "deadline_chain": ((6, 8, 4, 9), (12, 13, 0, 3)),
}
ZONE_CENTERS = {
    "relay_gap": ((4, 10), (11, 3), (14, 10), (9, 7), (3, 5), (13, 5)),
    "science_cluster": ((5, 6), (5, 7), (5, 8), (14, 6), (14, 7), (14, 8)),
    "compute_imbalance": ((4, 4), (10, 10), (14, 4), (9, 7), (3, 11), (15, 11)),
    "contact_outage": ((4, 6), (4, 8), (13, 6), (13, 8), (10, 3), (10, 11)),
    "deadline_chain": ((4, 3), (4, 11), (10, 2), (10, 12), (15, 5), (15, 9)),
}
ZONE_IDS = tuple(f"Z{index:02d}" for index in range(6))
REQUIRED_RELEASES = (0, 30, 60, 90)
REQUIRED_TYPES = ("capture", "localize", "plan", "dispatch")
SCIENCE_TYPES = ("collect", "analyze", "store")
MIGRATABLE_TYPES = {"localize", "plan", "analyze"}
ACTION_DELTAS = {
    "stay": (0, 0),
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
TASK_COSTS = {
    "capture": {"R1": (3, 2.0), "R2": (3, 2.0), "R3": (3, 2.0)},
    "localize": {"R1": (6, 3.0), "R2": (1, 2.0), "R3": (5, 2.0), "E": (2, 3.0)},
    "plan": {"R1": (7, 4.0), "R2": (2, 3.0), "R3": (6, 3.0), "E": (3, 4.0)},
    "dispatch": {"R1": (2, 1.0), "R2": (2, 1.0), "R3": (2, 1.0)},
    "collect": {"R1": (4, 2.0), "R2": (4, 2.0), "R3": (4, 2.0)},
    "analyze": {"R1": (8, 4.0), "R2": (6, 3.0), "R3": (7, 3.0), "E": (3, 4.0)},
    "store": {"E": (2, 2.0)},
}
PRODUCT_SIZE_BY_TYPE = {
    "capture": 4,
    "localize": 3,
    "plan": 2,
    "collect": 5,
    "analyze": 3,
}
ZERO_RESIDUES = {
    "relay_gap": {0},
    "science_cluster": {0, 1},
    "compute_imbalance": {0},
    "contact_outage": {0, 1},
    "deadline_chain": {0},
}
HALF_RESIDUES = {
    "relay_gap": {1, 2},
    "science_cluster": {2, 3},
    "compute_imbalance": {1, 2, 3},
    "contact_outage": {2, 3, 4},
    "deadline_chain": {1},
}
LOSS_RESIDUES = {
    "relay_gap": {0, 1},
    "science_cluster": {0},
    "compute_imbalance": {0, 1},
    "contact_outage": {0, 1, 2, 3},
    "deadline_chain": {0, 1, 2},
}
DIRECTED_LINKS = tuple((src, dst) for src in NODES for dst in NODES if src != dst)
LINK_INDEX = {pair: index for index, pair in enumerate(DIRECTED_LINKS)}


def _required_chain_id(rover: str, release: int) -> str:
    return f"req_{rover}_{release:03d}"


def _science_chain_id(zone_id: str) -> str:
    return f"sci_{zone_id}"


def _make_task_ids() -> tuple[str, ...]:
    required = tuple(
        f"{_required_chain_id(rover, release)}_{task_type}"
        for rover in ROVERS
        for release in REQUIRED_RELEASES
        for task_type in REQUIRED_TYPES
    )
    science = tuple(
        f"{_science_chain_id(zone_id)}_{task_type}"
        for zone_id in ZONE_IDS
        for task_type in SCIENCE_TYPES
    )
    return required + science


TASK_IDS = _make_task_ids()
TASK_ID_SET = set(TASK_IDS)
PRODUCT_IDS = tuple(
    f"p:{task_id}" for task_id in TASK_IDS if not task_id.endswith(("_dispatch", "_store"))
)
PRODUCT_ID_SET = set(PRODUCT_IDS)

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.moves": 0.34,
        "result_evidence.task_starts": 0.33,
        "result_evidence.transmission_proposals": 0.33,
    },
    "decision_legality": {
        "decision_legality.movement_energy": 0.20,
        "decision_legality.task_identity": 0.10,
        "decision_legality.task_eligibility": 0.25,
        "decision_legality.proposal_identity": 0.10,
        "decision_legality.proposal_fifo": 0.20,
        "decision_legality.proposal_resources": 0.15,
    },
    "replay_validity": {
        "replay_validity.motion_energy": 0.20,
        "replay_validity.compute_activity": 0.20,
        "replay_validity.queue_products": 0.20,
        "replay_validity.transfer_fragments": 0.20,
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
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
        {"scenario", "moves", "task_starts", "transmission_proposals"},
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


def _task_start_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "node", "task_id"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["node"] not in NODES:
        return f"{path}.node: expected one of {NODES!r}"
    if not isinstance(value["task_id"], str) or value["task_id"] not in TASK_ID_SET:
        return f"{path}.task_id: expected a fixed public task ID"
    return None


def _proposal_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "src", "dst", "product_id", "priority"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1):
        return f"{path}.slot: expected integer 0..{HORIZON - 1}"
    if value["src"] not in NODES or value["dst"] not in NODES or value["src"] == value["dst"]:
        return f"{path}.src/dst: expected two different public nodes"
    if not isinstance(value["product_id"], str) or value["product_id"] not in PRODUCT_ID_SET:
        return f"{path}.product_id: expected p:<nonterminal-task-id>"
    if not _integer(value["priority"], 0, 9):
        return f"{path}.priority: expected integer 0..9"
    return None


def _distinct_ids(value: Any, allowed: set[str], path: str) -> str | None:
    if not isinstance(value, list) or any(item not in allowed for item in value):
        return f"{path}: expected an array of fixed IDs"
    if len(value) != len(set(value)):
        return f"{path}: IDs must be distinct"
    return None


def _product_size(product_id: str) -> int:
    task_type = product_id.rsplit("_", 1)[-1]
    return PRODUCT_SIZE_BY_TYPE[task_type]


def _queue_error(value: Any, path: str, owner: str | None = None) -> str | None:
    if not isinstance(value, list):
        return f"{path}: expected FIFO array"
    product_ids: list[str] = []
    total_units = 0
    for index, row in enumerate(value):
        item_path = f"{path}[{index}]"
        error = _shape(row, {"product_id", "session_dst", "sent_offsets"}, item_path)
        if error:
            return error
        product_id = row["product_id"]
        if not isinstance(product_id, str) or product_id not in PRODUCT_ID_SET:
            return f"{item_path}.product_id: expected fixed product ID"
        product_ids.append(product_id)
        total_units += _product_size(product_id)
        destination = row["session_dst"]
        if destination is not None and (
            destination not in NODES or (owner is not None and destination == owner)
        ):
            return f"{item_path}.session_dst: expected another node or null"
        offsets = row["sent_offsets"]
        size = _product_size(product_id)
        if (
            not isinstance(offsets, list)
            or any(not _integer(offset, 0, size - 1) for offset in offsets)
            or offsets != sorted(set(offsets))
        ):
            return f"{item_path}.sent_offsets: expected strictly increasing in-range integers"
        if destination is None and offsets:
            return f"{item_path}: sent offsets require a session destination"
    if len(product_ids) != len(set(product_ids)):
        return f"{path}: product IDs must be distinct"
    if total_units > QUEUE_CAPACITY:
        return f"{path}: product sizes exceed {QUEUE_CAPACITY} units"
    return None


def _fragments_error(value: Any, path: str) -> str | None:
    if not isinstance(value, list):
        return f"{path}: expected fragment array"
    product_ids: list[str] = []
    for index, row in enumerate(value):
        item_path = f"{path}[{index}]"
        error = _shape(row, {"product_id", "src", "offsets"}, item_path)
        if error:
            return error
        product_id = row["product_id"]
        if (
            not isinstance(product_id, str)
            or product_id not in PRODUCT_ID_SET
            or row["src"] not in NODES
        ):
            return f"{item_path}: invalid product_id or src"
        size = _product_size(product_id)
        offsets = row["offsets"]
        if (
            not isinstance(offsets, list)
            or any(not _integer(offset, 0, size - 1) for offset in offsets)
            or offsets != sorted(set(offsets))
        ):
            return f"{item_path}.offsets: expected strictly increasing in-range integers"
        product_ids.append(product_id)
    if product_ids != sorted(product_ids) or len(product_ids) != len(set(product_ids)):
        return f"{path}: fragments must have distinct product IDs in increasing order"
    return None


def _checkpoint_error(value: Any, path: str) -> str | None:
    error = _shape(
        value,
        {
            "boundary",
            "rovers",
            "nodes",
            "released_zones",
            "completed_task_ids",
            "on_time_required_chain_ids",
            "stored_science_zone_ids",
            "attempted_unit_count",
            "movement_count",
            "edge_compute_energy",
            "edge_communication_energy",
        },
        path,
    )
    if error:
        return error
    if (
        not _integer(value["boundary"], 0, HORIZON)
        or value["boundary"] not in CHECKPOINT_BOUNDARIES
    ):
        return f"{path}.boundary: expected one of {CHECKPOINT_BOUNDARIES!r}"
    if not isinstance(value["rovers"], dict) or set(value["rovers"]) != set(ROVERS):
        return f"{path}.rovers: expected exactly {ROVERS!r}"
    for rover in ROVERS:
        rover_path = f"{path}.rovers.{rover}"
        state = value["rovers"][rover]
        error = _shape(state, {"x", "y", "energy"}, rover_path)
        if error:
            return error
        if not _integer(state["x"], 0, GRID_X - 1) or not _integer(state["y"], 0, GRID_Y - 1):
            return f"{rover_path}.x/y: expected public grid coordinates"
        if (
            not _finite(state["energy"])
            or not -EPS <= float(state["energy"]) <= INITIAL_ENERGY + EPS
        ):
            return f"{rover_path}.energy: expected finite value in [0,{INITIAL_ENERGY:g}]"
    if not isinstance(value["nodes"], dict) or set(value["nodes"]) != set(NODES):
        return f"{path}.nodes: expected exactly {NODES!r}"
    for node in NODES:
        node_path = f"{path}.nodes.{node}"
        state = value["nodes"][node]
        error = _shape(
            state,
            {
                "activity_task_id",
                "remaining_compute_slots",
                "queue",
                "pending_product_ids",
                "fragments",
            },
            node_path,
        )
        if error:
            return error
        activity = state["activity_task_id"]
        remaining = state["remaining_compute_slots"]
        if activity is None:
            if remaining != 0 or isinstance(remaining, bool):
                return f"{node_path}: idle requires null activity and zero remaining"
        elif (
            not isinstance(activity, str)
            or activity not in TASK_ID_SET
            or not _integer(remaining, 1, 8)
        ):
            return f"{node_path}: compute activity requires task ID and remaining 1..8"
        error = _queue_error(state["queue"], f"{node_path}.queue", owner=node)
        if error:
            return error
        pending = state["pending_product_ids"]
        if (
            not isinstance(pending, list)
            or any(
                not isinstance(product_id, str) or product_id not in PRODUCT_ID_SET
                for product_id in pending
            )
            or len(pending) != len(set(pending))
        ):
            return f"{node_path}.pending_product_ids: expected distinct fixed product IDs"
        error = _fragments_error(state["fragments"], f"{node_path}.fragments")
        if error:
            return error
    zones = value["released_zones"]
    if not isinstance(zones, list):
        return f"{path}.released_zones: expected array"
    zone_ids: list[str] = []
    for index, row in enumerate(zones):
        item_path = f"{path}.released_zones[{index}]"
        error = _shape(row, {"zone_id", "winner", "entry_slot", "release_slot"}, item_path)
        if error:
            return error
        if row["zone_id"] not in ZONE_IDS or row["winner"] not in ROVERS:
            return f"{item_path}: invalid zone_id or winner"
        if (
            not _integer(row["entry_slot"], 0, HORIZON - 1)
            or row["release_slot"] != row["entry_slot"] + 1
        ):
            return f"{item_path}: release_slot must equal entry_slot+1"
        zone_ids.append(row["zone_id"])
    if zone_ids != sorted(set(zone_ids)):
        return f"{path}.released_zones: expected distinct rows in zone-ID order"
    for field, allowed in (
        ("completed_task_ids", TASK_ID_SET),
        (
            "on_time_required_chain_ids",
            {_required_chain_id(r, t) for r in ROVERS for t in REQUIRED_RELEASES},
        ),
        ("stored_science_zone_ids", set(ZONE_IDS)),
    ):
        error = _distinct_ids(value[field], allowed, f"{path}.{field}")
        if error:
            return error
    if not _integer(value["attempted_unit_count"], 0, ATTEMPT_LIMIT):
        return f"{path}.attempted_unit_count: expected integer 0..{ATTEMPT_LIMIT}"
    if not _integer(value["movement_count"], 0, len(ROVERS) * HORIZON):
        return f"{path}.movement_count: expected integer 0..{len(ROVERS) * HORIZON}"
    for field, upper in (("edge_compute_energy", 480.0), ("edge_communication_energy", 8.0)):
        if not _finite(value[field]) or not -EPS <= float(value[field]) <= upper + EPS:
            return f"{path}.{field}: expected finite value in [0,{upper:g}]"
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
        "task_starts": [True, "task-start fields are well formed"],
        "transmission_proposals": [True, "proposal fields are well formed"],
    }

    def fail(rule: str, detail: str) -> None:
        if rules[rule][0]:
            rules[rule] = [False, detail]

    validators = {
        "moves": _move_error,
        "task_starts": _task_start_error,
        "transmission_proposals": _proposal_error,
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
        record = matches[0]
        moves = record.get("moves")
        if isinstance(moves, list):
            defects.extend(
                _key_defects(
                    f"{name}.moves",
                    [
                        _hashable((row.get("slot"), row.get("rover")))
                        if isinstance(row, dict)
                        else _hashable(row)
                        for row in moves
                    ],
                    expected_moves,
                )
            )
        else:
            defects.append(f"{name}.moves: missing array")
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _decision_map(record: Mapping[str, Any]) -> dict[str, Any] | None:
    for field in ("moves", "task_starts", "transmission_proposals"):
        if not isinstance(record.get(field), list):
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
    starts: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["task_starts"]:
        if not isinstance(row, dict) or _task_start_error(row, "task_start"):
            return None
        key = (row["slot"], row["node"])
        if key in starts:
            return None
        starts[key] = row
    proposals: dict[tuple[int, str], dict[str, Any]] = {}
    for row in record["transmission_proposals"]:
        if not isinstance(row, dict) or _proposal_error(row, "proposal"):
            return None
        key = (row["slot"], row["src"])
        if key in proposals:
            return None
        proposals[key] = row
    return {"moves": moves, "starts": starts, "proposals": proposals}


def _obstacle(name: str, cell: tuple[int, int]) -> bool:
    x, y = cell
    return any(
        x_min <= x <= x_max and y_min <= y <= y_max
        for x_min, x_max, y_min, y_max in OBSTACLE_RECTS[name]
    )


def _round_half_down(numerator: int, denominator: int = 6) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient if 2 * remainder <= denominator else quotient + 1


def _node_position(node: str, positions: Mapping[str, tuple[int, int]]) -> tuple[int, int]:
    return EDGE_POSITION if node == "E" else positions[node]


def _capacity(
    name: str,
    slot: int,
    src: str,
    dst: str,
    positions: Mapping[str, tuple[int, int]],
) -> int:
    first = _node_position(src, positions)
    second = _node_position(dst, positions)
    if math.dist(first, second) > 6.0 + EPS:
        return 0
    for sample in range(1, 6):
        cell = (
            _round_half_down((6 - sample) * first[0] + sample * second[0]),
            _round_half_down((6 - sample) * first[1] + sample * second[1]),
        )
        if _obstacle(name, cell):
            return 0
    if name == "contact_outage" and "E" in {src, dst} and (35 <= slot <= 49 or 75 <= slot <= 89):
        return 0
    index = LINK_INDEX[(src, dst)]
    residue = (7 * slot + 3 * index + 5 * SCENARIO_INDEX[name]) % 12
    if residue in ZERO_RESIDUES[name]:
        return 0
    return 2 if residue in HALF_RESIDUES[name] else 4


def _unit_lost(name: str, slot: int, src: str, dst: str, offset: int) -> bool:
    index = LINK_INDEX[(src, dst)]
    residue = (11 * slot + 7 * index + 5 * offset + 3 * SCENARIO_INDEX[name]) % 17
    return residue in LOSS_RESIDUES[name]


def _zone_contains(name: str, zone_id: str, cell: tuple[int, int]) -> bool:
    center = ZONE_CENTERS[name][ZONE_IDS.index(zone_id)]
    return abs(cell[0] - center[0]) + abs(cell[1] - center[1]) <= 1


def _task_catalog() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for rover in ROVERS:
        for release in REQUIRED_RELEASES:
            chain_id = _required_chain_id(rover, release)
            for index, task_type in enumerate(REQUIRED_TYPES):
                task_id = f"{chain_id}_{task_type}"
                tasks[task_id] = {
                    "task_id": task_id,
                    "kind": "required",
                    "chain_id": chain_id,
                    "zone_id": None,
                    "task_type": task_type,
                    "source": rover,
                    "release": release,
                    "deadline": release + 17,
                    "predecessor": None
                    if index == 0
                    else f"p:{chain_id}_{REQUIRED_TYPES[index - 1]}",
                    "successor": None
                    if index == len(REQUIRED_TYPES) - 1
                    else f"{chain_id}_{REQUIRED_TYPES[index + 1]}",
                    "started": False,
                    "start": None,
                    "node": None,
                    "finish": None,
                    "completed": False,
                }
    for zone_id in ZONE_IDS:
        chain_id = _science_chain_id(zone_id)
        for index, task_type in enumerate(SCIENCE_TYPES):
            task_id = f"{chain_id}_{task_type}"
            tasks[task_id] = {
                "task_id": task_id,
                "kind": "science",
                "chain_id": chain_id,
                "zone_id": zone_id,
                "task_type": task_type,
                "source": None,
                "release": None,
                "deadline": None,
                "predecessor": None if index == 0 else f"p:{chain_id}_{SCIENCE_TYPES[index - 1]}",
                "successor": None
                if index == len(SCIENCE_TYPES) - 1
                else f"{chain_id}_{SCIENCE_TYPES[index + 1]}",
                "started": False,
                "start": None,
                "node": None,
                "finish": None,
                "completed": False,
            }
    return tasks


def _task_allowed_node(task: Mapping[str, Any], node: str) -> bool:
    task_type = task["task_type"]
    if node not in TASK_COSTS[task_type]:
        return False
    if task_type in {"capture", "dispatch", "collect"}:
        return node == task["source"]
    return task_type != "store" or node == "E"


def _queue_units(queue: Sequence[str], products: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(int(products[product_id]["size"]) for product_id in queue)


def _local_views(
    name: str,
    slot: int,
    positions: Mapping[str, tuple[int, int]],
    energies: Mapping[str, float],
    active: Mapping[str, str | None],
    queues: Mapping[str, list[str]],
    pending: Mapping[str, list[str]],
    fragments: Mapping[str, Mapping[str, Mapping[str, Any]]],
    products: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    zones: Mapping[str, Mapping[str, Any]],
    capacities: Mapping[tuple[str, str], int],
    attempted_units: int,
    known_science: Mapping[str, set[str]],
) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for node in NODES:
        visible_tasks = []
        for task in tasks.values():
            if task["kind"] == "required" or task["zone_id"] in known_science[node]:
                visible_tasks.append(
                    {
                        "task_id": task["task_id"],
                        "kind": task["kind"],
                        "zone_id": task["zone_id"],
                        "task_type": task["task_type"],
                        "source": task["source"],
                        "release": task["release"],
                        "deadline": task["deadline"],
                        "predecessor": task["predecessor"],
                        "started": bool(task["started"] and task["node"] == node),
                    }
                )
        view: dict[str, Any] = {
            "scenario": name,
            "slot": slot,
            "node": node,
            "activity_task_id": active[node],
            "queue": [
                {
                    "product_id": product_id,
                    "producer_task_id": products[product_id]["producer_task_id"],
                    "chain_id": products[product_id]["chain_id"],
                    "zone_id": products[product_id]["zone_id"],
                    "source": products[product_id]["source"],
                    "ready_boundary": products[product_id]["ready"],
                    "deadline": products[product_id]["deadline"],
                    "size": products[product_id]["size"],
                    "session_dst": products[product_id]["session_dst"],
                    "sent_offsets": sorted(products[product_id]["sent_offsets"]),
                    "successor": products[product_id]["successor"],
                }
                for product_id in queues[node]
            ],
            "pending_product_ids": list(pending[node]),
            "fragment_product_ids": sorted(fragments[node]),
            "visible_tasks": visible_tasks,
            "incident_capacities": {
                f"{src}->{dst}": capacity
                for (src, dst), capacity in capacities.items()
                if node in {src, dst}
            },
            "remaining_attempt_units": ATTEMPT_LIMIT - attempted_units,
        }
        if node in ROVERS:
            view.update({"position": positions[node], "energy": energies[node]})
        views[node] = view
    return {"scenario": name, "slot": slot, "node_views": views}


def _simulate(
    name: str,
    decisions: Mapping[str, Any] | None = None,
    provider: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if name not in SCENARIO_NAMES or (decisions is None) == (provider is None):
        raise ValueError("provide exactly one valid scenario and decision source")
    positions = dict(STARTS[name])
    energies = {rover: INITIAL_ENERGY for rover in ROVERS}
    active: dict[str, str | None] = {node: None for node in NODES}
    queues: dict[str, list[str]] = {node: [] for node in NODES}
    pending: dict[str, list[str]] = {node: [] for node in NODES}
    fragments: dict[str, dict[str, dict[str, Any]]] = {node: {} for node in NODES}
    products: dict[str, dict[str, Any]] = {}
    tasks = _task_catalog()
    zones: dict[str, dict[str, Any]] = {}
    known_science = {node: set() for node in NODES}
    arrivals: dict[int, list[tuple[str, str, str, int]]] = {}
    attempted_units = 0
    movement_count = 0
    edge_compute_energy = 0.0
    edge_communication_energy = 0.0
    attempts: list[dict[str, Any]] = []
    service_opportunities = 0
    served_opportunities = 0
    move_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    expiration_events: list[dict[str, Any]] = []
    violations: dict[str, list[str]] = {
        "movement_energy": [],
        "task_identity": [],
        "task_eligibility": [],
        "proposal_identity": [],
        "proposal_fifo": [],
        "proposal_resources": [],
    }

    def fail(rule: str, detail: str) -> None:
        violations[rule].append(detail)

    def remove_product(product_id: str, status: str) -> None:
        product = products.get(product_id)
        if product is not None:
            product["status"] = status
            product["holder"] = None
        for node in NODES:
            if product_id in queues[node]:
                queues[node].remove(product_id)
            if product_id in pending[node]:
                pending[node].remove(product_id)
            fragments[node].pop(product_id, None)

    def expire(boundary: int) -> None:
        for product_id, product in list(products.items()):
            if (
                product["status"] not in {"consumed", "expired"}
                and boundary > product["ready"] + AGE_LIMIT
            ):
                expiration_events.append({"boundary": boundary, "product_id": product_id})
                remove_product(product_id, "expired")

    def finish_computations(boundary: int) -> None:
        nonlocal edge_compute_energy
        for node in NODES:
            task_id = active[node]
            if task_id is None or tasks[task_id]["finish"] != boundary:
                continue
            task = tasks[task_id]
            task["completed"] = True
            active[node] = None
            if task["task_type"] in {"dispatch", "store"}:
                continue
            product_id = f"p:{task_id}"
            products[product_id] = {
                "product_id": product_id,
                "producer_task_id": task_id,
                "chain_id": task["chain_id"],
                "zone_id": task["zone_id"],
                "source": task["source"],
                "ready": boundary,
                "deadline": task["deadline"],
                "size": PRODUCT_SIZE_BY_TYPE[task["task_type"]],
                "successor": task["successor"],
                "holder": node,
                "queue_enter": boundary,
                "session_dst": None,
                "sent_offsets": set(),
                "status": "pending",
                "consumed_by": None,
            }
            pending[node].append(product_id)
            pending[node].sort(key=lambda item: (products[item]["ready"], item))

    def process_arrivals(boundary: int) -> None:
        for product_id, src, dst, offset in arrivals.pop(boundary, []):
            product = products.get(product_id)
            if product is None or product["status"] == "expired":
                continue
            buffer = fragments[dst].setdefault(product_id, {"src": src, "offsets": set()})
            buffer["offsets"].add(offset)
            if len(buffer["offsets"]) != product["size"]:
                continue
            fragments[dst].pop(product_id)
            product["holder"] = dst
            product["queue_enter"] = boundary
            product["session_dst"] = None
            product["sent_offsets"] = set()
            product["status"] = "pending"
            pending[dst].append(product_id)
            pending[dst].sort(key=lambda item: (products[item]["ready"], item))
            if product["zone_id"] is not None:
                known_science[dst].add(product["zone_id"])

    def apply_zone_releases(boundary: int) -> None:
        for zone_id, event in zones.items():
            if event["release_slot"] != boundary:
                continue
            winner = event["winner"]
            known_science[winner].add(zone_id)
            for task_type in SCIENCE_TYPES:
                task = tasks[f"{_science_chain_id(zone_id)}_{task_type}"]
                task["source"] = winner
                task["release"] = boundary
                task["deadline"] = boundary + 36

    def admit_pending() -> None:
        for node in NODES:
            while pending[node]:
                product_id = pending[node][0]
                if (
                    _queue_units(queues[node], products) + products[product_id]["size"]
                    > QUEUE_CAPACITY
                ):
                    break
                pending[node].pop(0)
                queues[node].append(product_id)
                products[product_id]["status"] = "queued"
                products[product_id]["holder"] = node
                products[product_id]["queue_enter"] = products[product_id].get("queue_enter", 0)

    def record_checkpoint(boundary: int) -> None:
        def activity(node: str) -> tuple[str | None, int]:
            task_id = active[node]
            return (None, 0) if task_id is None else (task_id, tasks[task_id]["finish"] - boundary)

        completed = sorted(task_id for task_id, task in tasks.items() if task["completed"])
        on_time_required = sorted(
            chain_id
            for rover in ROVERS
            for release in REQUIRED_RELEASES
            for chain_id in [_required_chain_id(rover, release)]
            if tasks[f"{chain_id}_dispatch"]["completed"]
            and tasks[f"{chain_id}_dispatch"]["finish"] <= release + 17
        )
        stored_science = sorted(
            zone_id
            for zone_id in ZONE_IDS
            if tasks[f"{_science_chain_id(zone_id)}_store"]["completed"]
        )
        checkpoints.append(
            {
                "boundary": boundary,
                "rovers": {
                    rover: {
                        "x": positions[rover][0],
                        "y": positions[rover][1],
                        "energy": energies[rover],
                    }
                    for rover in ROVERS
                },
                "nodes": {
                    node: {
                        "activity_task_id": activity(node)[0],
                        "remaining_compute_slots": activity(node)[1],
                        "queue": [
                            {
                                "product_id": product_id,
                                "session_dst": products[product_id]["session_dst"],
                                "sent_offsets": sorted(products[product_id]["sent_offsets"]),
                            }
                            for product_id in queues[node]
                        ],
                        "pending_product_ids": list(pending[node]),
                        "fragments": [
                            {
                                "product_id": product_id,
                                "src": fragments[node][product_id]["src"],
                                "offsets": sorted(fragments[node][product_id]["offsets"]),
                            }
                            for product_id in sorted(fragments[node])
                        ],
                    }
                    for node in NODES
                },
                "released_zones": [dict(zones[zone_id]) for zone_id in sorted(zones)],
                "completed_task_ids": completed,
                "on_time_required_chain_ids": on_time_required,
                "stored_science_zone_ids": stored_science,
                "attempted_unit_count": attempted_units,
                "movement_count": movement_count,
                "edge_compute_energy": edge_compute_energy,
                "edge_communication_energy": edge_communication_energy,
            }
        )

    def boundary_steps(boundary: int) -> dict[tuple[str, str], int]:
        expire(boundary)
        finish_computations(boundary)
        process_arrivals(boundary)
        apply_zone_releases(boundary)
        admit_pending()
        if boundary == HORIZON:
            return {}
        return {
            pair: _capacity(name, boundary, pair[0], pair[1], positions) for pair in DIRECTED_LINKS
        }

    for slot in range(HORIZON):
        capacities = boundary_steps(slot)
        if slot in CHECKPOINT_BOUNDARIES:
            record_checkpoint(slot)
        view = _local_views(
            name,
            slot,
            positions,
            energies,
            active,
            queues,
            pending,
            fragments,
            products,
            tasks,
            zones,
            capacities,
            attempted_units,
            known_science,
        )
        chosen_moves: dict[str, str] = {}
        cycle_starts: dict[str, dict[str, Any]] = {}
        cycle_proposals: dict[str, dict[str, Any]] = {}
        if provider is not None:
            for node in NODES:
                supplied = provider(slot, view["node_views"][node])
                if node in ROVERS:
                    chosen_moves[node] = str(supplied.get("move", "stay"))
                if supplied.get("task_start") is not None:
                    cycle_starts[node] = {
                        "slot": slot,
                        "node": node,
                        "task_id": supplied["task_start"],
                    }
                proposal = supplied.get("proposal")
                if proposal is not None:
                    cycle_proposals[node] = {
                        "slot": slot,
                        "src": node,
                        "dst": proposal.get("dst"),
                        "product_id": proposal.get("product_id"),
                        "priority": proposal.get("priority"),
                    }
        else:
            assert decisions is not None
            chosen_moves = {rover: decisions["moves"][(slot, rover)]["action"] for rover in ROVERS}
            cycle_starts = {
                node: row
                for (row_slot, node), row in decisions["starts"].items()
                if row_slot == slot
            }
            cycle_proposals = {
                node: row
                for (row_slot, node), row in decisions["proposals"].items()
                if row_slot == slot
            }

        for rover in ROVERS:
            move_rows.append({"slot": slot, "rover": rover, "action": chosen_moves[rover]})
        start_rows.extend(dict(cycle_starts[node]) for node in NODES if node in cycle_starts)
        proposal_rows.extend(
            dict(cycle_proposals[node]) for node in NODES if node in cycle_proposals
        )

        move_costs: dict[str, float] = {}
        move_candidates: dict[str, tuple[int, int]] = {}
        move_legal: dict[str, bool] = {}
        for rover in ROVERS:
            action = chosen_moves[rover]
            dx, dy = ACTION_DELTAS[action]
            old = positions[rover]
            candidate = (old[0] + dx, old[1] + dy)
            if action == "stay":
                move_costs[rover] = min(0.1, energies[rover])
                move_candidates[rover] = old
                move_legal[rover] = True
            else:
                legal = (
                    0 <= candidate[0] < GRID_X
                    and 0 <= candidate[1] < GRID_Y
                    and not _obstacle(name, candidate)
                )
                move_costs[rover] = 1.0
                move_candidates[rover] = candidate if legal else old
                move_legal[rover] = legal and energies[rover] + EPS >= 1.0

        accepted_starts: dict[str, str] = {}
        seen_started_ids: set[str] = set()
        for node in NODES:
            row = cycle_starts.get(node)
            if row is None:
                continue
            task_id = row["task_id"]
            if task_id in seen_started_ids:
                fail(
                    "task_identity", f"{name}.task_starts[{slot},{node}]: duplicate task ID in slot"
                )
                continue
            seen_started_ids.add(task_id)
            task = tasks[task_id]
            predecessor = task["predecessor"]
            predecessor_ok = predecessor is None or (
                predecessor in queues[node]
                and products[predecessor]["session_dst"] is None
                and not products[predecessor]["sent_offsets"]
            )
            valid = (
                task["release"] is not None
                and slot >= task["release"]
                and not task["started"]
                and active[node] is None
                and _task_allowed_node(task, node)
                and predecessor_ok
            )
            if not valid:
                fail(
                    "task_eligibility",
                    f"{name}.task_starts[{slot},{node}].task_id={task_id!r}: task must be "
                    "released, unstarted, supported, idle, and hold an unlocked predecessor",
                )
                continue
            duration, task_energy = TASK_COSTS[task["task_type"]][node]
            if node in ROVERS:
                if not move_legal[node] or energies[node] + EPS < task_energy + move_costs[node]:
                    fail(
                        "movement_energy",
                        f"{name}.slot={slot}.{node}: task plus requested movement exceeds "
                        "energy or movement boundary",
                    )
                    if energies[node] + EPS < task_energy + min(0.1, energies[node]):
                        continue
                    chosen_moves[node] = "stay"
                    move_candidates[node] = positions[node]
                    move_costs[node] = min(0.1, max(0.0, energies[node] - task_energy))
                    move_legal[node] = True
                energies[node] -= task_energy
            else:
                edge_compute_energy += task_energy
            if predecessor is not None:
                queues[node].remove(predecessor)
                products[predecessor]["status"] = "consumed"
                products[predecessor]["holder"] = None
                products[predecessor]["consumed_by"] = task_id
            task["started"] = True
            task["start"] = slot
            task["node"] = node
            task["finish"] = slot + duration
            active[node] = task_id
            accepted_starts[node] = task_id

        for rover in ROVERS:
            if not move_legal[rover]:
                fail(
                    "movement_energy",
                    f"{name}.moves[{slot},{rover}]: action violates grid, obstacle, or energy",
                )
                chosen_moves[rover] = "stay"
                move_candidates[rover] = positions[rover]
                move_costs[rover] = min(0.1, energies[rover])

        valid_proposals: list[dict[str, Any]] = []
        for src in NODES:
            row = cycle_proposals.get(src)
            if row is None:
                continue
            product_id = row["product_id"]
            if not queues[src] or queues[src][0] != product_id:
                fail(
                    "proposal_fifo",
                    f"{name}.transmission_proposals[{slot},{src}]: product must be current "
                    "FIFO head",
                )
                continue
            product = products[product_id]
            capacity = capacities[(src, row["dst"])]
            if (
                active[src] is not None
                or capacity <= 0
                or product["session_dst"] not in {None, row["dst"]}
                or slot >= product["ready"] + AGE_LIMIT
            ):
                fail(
                    "proposal_resources",
                    f"{name}.transmission_proposals[{slot},{src}]: source activity, capacity, "
                    "session destination, or product age is invalid",
                )
                continue
            remaining_offsets = product["size"] - len(product["sent_offsets"])
            reserve_units = min(capacity, remaining_offsets, ATTEMPT_LIMIT - attempted_units)
            if src in ROVERS and energies[src] + EPS < move_costs[src] + 0.05 * reserve_units:
                fail(
                    "proposal_resources",
                    f"{name}.transmission_proposals[{slot},{src}]: move plus maximum send "
                    "cost exceeds energy",
                )
                continue
            valid_proposals.append(row)

        # A rover's move must also leave room for the maximum unit charge of
        # its own otherwise well-formed proposal.  Revert the move in replay
        # while retaining the proposal-resource violation for the candidate.
        for rover in ROVERS:
            row = cycle_proposals.get(rover)
            if (
                row is None
                or chosen_moves[rover] == "stay"
                or rover in accepted_starts
                or not move_legal[rover]
            ):
                continue
            product = products.get(row.get("product_id"))
            dst = row.get("dst")
            if product is None or dst not in NODES or dst == rover:
                continue
            capacity = capacities.get((rover, dst), 0)
            remaining = product["size"] - len(product["sent_offsets"])
            reserve_units = min(capacity, remaining, ATTEMPT_LIMIT - attempted_units)
            if energies[rover] + EPS < move_costs[rover] + 0.05 * reserve_units:
                fail(
                    "movement_energy",
                    f"{name}.moves[{slot},{rover}]: move cannot reserve its proposal send cost",
                )
                chosen_moves[rover] = "stay"
                move_candidates[rover] = positions[rover]
                move_costs[rover] = min(0.1, energies[rover])
                move_legal[rover] = True

        occupied = {node for node in NODES if active[node] is not None}
        for src, dst in DIRECTED_LINKS:
            if src in occupied or dst in occupied or capacities[(src, dst)] <= 0 or not queues[src]:
                continue
            product = products[queues[src][0]]
            if (
                product["session_dst"] not in {None, dst}
                or slot >= product["ready"] + AGE_LIMIT
                or len(product["sent_offsets"]) >= product["size"]
            ):
                continue
            service_opportunities += 1

        selected: list[dict[str, Any]] = []
        used = set(occupied)
        for row in sorted(
            valid_proposals,
            key=lambda item: (-item["priority"], item["src"], item["dst"], item["product_id"]),
        ):
            if row["src"] in used or row["dst"] in used or attempted_units >= ATTEMPT_LIMIT:
                continue
            selected.append(row)
            used.update((row["src"], row["dst"]))
            if len(selected) == 2:
                break

        for row in selected:
            src = row["src"]
            dst = row["dst"]
            product = products[row["product_id"]]
            if product["session_dst"] is None:
                product["session_dst"] = dst
            offsets = [
                offset for offset in range(product["size"]) if offset not in product["sent_offsets"]
            ]
            offsets = offsets[: min(capacities[(src, dst)], ATTEMPT_LIMIT - attempted_units)]
            delivered_in_link = 0
            for offset in offsets:
                attempted_units += 1
                if src in ROVERS:
                    energies[src] -= 0.05
                else:
                    edge_communication_energy += 0.05
                lost = _unit_lost(name, slot, src, dst, offset)
                attempt = {
                    "slot": slot,
                    "src": src,
                    "dst": dst,
                    "product_id": product["product_id"],
                    "offset": offset,
                    "outcome": "lost" if lost else "delivered",
                    "arrival": None if lost else slot + 1,
                    "queue_wait": slot - product["queue_enter"],
                }
                attempts.append(attempt)
                if not lost:
                    delivered_in_link += 1
                    product["sent_offsets"].add(offset)
                    arrivals.setdefault(slot + 1, []).append(
                        (product["product_id"], src, dst, offset)
                    )
            if delivered_in_link:
                served_opportunities += 1
            if len(product["sent_offsets"]) == product["size"]:
                queues[src].pop(0)
                product["status"] = "in_flight"
                product["holder"] = None

        old_positions = dict(positions)
        for rover in ROVERS:
            energies[rover] = max(0.0, energies[rover] - move_costs[rover])
            positions[rover] = move_candidates[rover]
            if chosen_moves[rover] != "stay":
                movement_count += 1
        for zone_id in ZONE_IDS:
            if zone_id in zones:
                continue
            entrants = [
                rover
                for rover in ROVERS
                if not _zone_contains(name, zone_id, old_positions[rover])
                and _zone_contains(name, zone_id, positions[rover])
            ]
            if entrants:
                winner = min(entrants, key=ROVERS.index)
                zones[zone_id] = {
                    "zone_id": zone_id,
                    "winner": winner,
                    "entry_slot": slot,
                    "release_slot": slot + 1,
                }

    boundary_steps(HORIZON)
    record_checkpoint(HORIZON)
    return {
        "scenario": name,
        "positions": positions,
        "energies": energies,
        "active": active,
        "queues": queues,
        "pending": pending,
        "fragments": fragments,
        "products": products,
        "tasks": tasks,
        "zones": zones,
        "attempted_units": attempted_units,
        "movement_count": movement_count,
        "edge_compute_energy": edge_compute_energy,
        "edge_communication_energy": edge_communication_energy,
        "attempts": attempts,
        "service_opportunities": service_opportunities,
        "served_opportunities": served_opportunities,
        "checkpoints": checkpoints,
        "expiration_events": expiration_events,
        "violations": violations,
        "record": {
            "scenario": name,
            "moves": move_rows,
            "task_starts": start_rows,
            "transmission_proposals": proposal_rows,
        },
    }


def _dynamic_key_defect(records: Mapping[str, dict[str, Any]], field: str) -> str | None:
    key_fields = ("slot", "node") if field == "task_starts" else ("slot", "src")
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None or not isinstance(record.get(field), list):
            continue
        seen: set[Hashable] = set()
        for index, row in enumerate(record[field]):
            if not isinstance(row, dict):
                continue
            key = _hashable((row.get(key_fields[0]), row.get(key_fields[1])))
            if key in seen:
                return f"{name}.{field}[{index}]: duplicate node-slot key {key!r}"
            seen.add(key)
    return None


def _duplicate_task_id(records: Mapping[str, dict[str, Any]]) -> str | None:
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None or not isinstance(record.get("task_starts"), list):
            continue
        seen: set[Any] = set()
        for index, row in enumerate(record["task_starts"]):
            if not isinstance(row, dict):
                continue
            task_id = _hashable(row.get("task_id"))
            if task_id in seen:
                return f"{name}.task_starts[{index}].task_id: duplicate candidate task ID"
            seen.add(task_id)
    return None


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules: dict[str, tuple[bool, str]] = {
        "movement_energy": (True, "moves and combined rover energy are legal"),
        "task_identity": (True, "task node-slot keys and task IDs are unique"),
        "task_eligibility": (True, "task starts satisfy release, support, activity, and data"),
        "proposal_identity": (True, "proposal node-slot keys are unique"),
        "proposal_fifo": (True, "proposals use the local FIFO head"),
        "proposal_resources": (
            True,
            "proposals satisfy capacity, session, age, activity, and energy",
        ),
    }
    task_key = _dynamic_key_defect(records, "task_starts")
    task_id = _duplicate_task_id(records)
    proposal_key = _dynamic_key_defect(records, "transmission_proposals")
    if task_key or task_id:
        rules["task_identity"] = (False, task_key or task_id or "duplicate task decision")
    if proposal_key:
        rules["proposal_identity"] = (False, proposal_key)
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            continue
        decisions = _decision_map(record)
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


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules: dict[str, tuple[bool, str]] = {
        "motion_energy": (True, "movement and rover energy replayed"),
        "compute_activity": (True, "task starts, computation, and completion replayed"),
        "queue_products": (True, "product creation, consumption, pending, and FIFO replayed"),
        "transfer_fragments": (True, "contact, attempts, loss, handoff, and fragments replayed"),
        "mission_state": (True, "science releases and required/science results replayed"),
    }
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIO_NAMES:
        record = records.get(name)
        if record is None:
            detail = f"{name}: missing scenario record; independent replay unavailable"
            for rule in rules:
                if rules[rule][0]:
                    rules[rule] = (False, detail)
            continue
        decisions = _decision_map(record)
        if decisions is None:
            detail = f"{name}: required decision map is unusable"
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
        if (
            any(
                not (0 <= position[0] < GRID_X and 0 <= position[1] < GRID_Y)
                or not -EPS <= replay["energies"][rover] <= INITIAL_ENERGY + EPS
                for rover, position in replay["positions"].items()
            )
            or not 0 <= replay["movement_count"] <= len(ROVERS) * HORIZON
        ):
            rules["motion_energy"] = (False, f"{name}: derived motion invariant failed")
        if any(
            task["completed"] and (task["finish"] is None or task["finish"] > HORIZON)
            for task in replay["tasks"].values()
        ) or any(
            task_id is not None and replay["tasks"][task_id]["finish"] <= HORIZON
            for task_id in replay["active"].values()
        ):
            rules["compute_activity"] = (False, f"{name}: derived compute invariant failed")
        if any(
            _queue_units(queue, replay["products"]) > QUEUE_CAPACITY
            for queue in replay["queues"].values()
        ) or any(
            len(queue) != len(set(queue))
            or len(replay["pending"][node]) != len(set(replay["pending"][node]))
            for node, queue in replay["queues"].items()
        ):
            rules["queue_products"] = (False, f"{name}: derived FIFO capacity invariant failed")
        if (
            not 0 <= replay["attempted_units"] <= ATTEMPT_LIMIT
            or replay["attempted_units"] != len(replay["attempts"])
            or any(
                attempt["outcome"] not in {"lost", "delivered"}
                or not 0 <= attempt["offset"] < replay["products"][attempt["product_id"]]["size"]
                for attempt in replay["attempts"]
            )
        ):
            rules["transfer_fragments"] = (False, f"{name}: attempt budget invariant failed")
        if any(
            zone_id not in ZONE_IDS
            or event["winner"] not in ROVERS
            or event["release_slot"] != event["entry_slot"] + 1
            for zone_id, event in replay["zones"].items()
        ):
            rules["mission_state"] = (False, f"{name}: derived mission invariant failed")
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


def _chain_on_time(replay: Mapping[str, Any], chain_id: str) -> bool:
    tasks = replay["tasks"]
    terminal = f"{chain_id}_{'dispatch' if chain_id.startswith('req_') else 'store'}"
    task = tasks[terminal]
    return bool(task["completed"] and task["finish"] <= task["deadline"])


def _scenario_metrics(replay: Mapping[str, Any]) -> dict[str, float]:
    tasks = replay["tasks"]
    required_chains = [
        _required_chain_id(rover, release) for rover in ROVERS for release in REQUIRED_RELEASES
    ]
    on_time_required = [
        chain_id for chain_id in required_chains if _chain_on_time(replay, chain_id)
    ]
    on_time_required_tasks = sum(
        1
        for task in tasks.values()
        if task["kind"] == "required" and task["completed"] and task["finish"] <= task["deadline"]
    )
    primary_1 = 0.75 * len(on_time_required) / 12.0 + 0.25 * on_time_required_tasks / 48.0

    released_zones = set(replay["zones"])
    released_value = sum(2 + ((3 * ZONE_IDS.index(zone_id) + 1) % 5) for zone_id in released_zones)
    stored_value = sum(
        2 + ((3 * ZONE_IDS.index(zone_id) + 1) % 5)
        for zone_id in released_zones
        if _chain_on_time(replay, _science_chain_id(zone_id))
    )
    primary_2 = stored_value / released_value if released_value else 0.0

    if on_time_required:
        latency = sum(
            (
                tasks[f"{chain_id}_dispatch"]["deadline"]
                - tasks[f"{chain_id}_dispatch"]["finish"]
                + 1
            )
            / 18.0
            for chain_id in on_time_required
        ) / len(on_time_required)
        latency = max(0.0, min(1.0, latency))
    else:
        latency = 0.0
    rover_energy = len(ROVERS) * INITIAL_ENERGY - sum(replay["energies"].values())
    total_energy = (
        rover_energy + replay["edge_compute_energy"] + replay["edge_communication_energy"]
    )
    primary_3 = (len(on_time_required) / 12.0) * (
        0.4 * latency
        + 0.3 * max(0.0, min(1.0, 1.0 - replay["movement_count"] / 360.0))
        + 0.3 * max(0.0, min(1.0, 1.0 - total_energy / 1028.0))
    )

    qualifying_tasks: list[dict[str, Any]] = []
    for task in tasks.values():
        if task["task_type"] not in MIGRATABLE_TYPES or not task["completed"]:
            continue
        product = replay["products"].get(f"p:{task['task_id']}")
        if product is None or product["consumed_by"] is None:
            continue
        if _chain_on_time(replay, task["chain_id"]):
            qualifying_tasks.append(task)
    gain = 0.0
    max_gain = 0.0
    for task in qualifying_tasks:
        source = task["source"]
        source_duration = TASK_COSTS[task["task_type"]][source][0]
        actual_duration = TASK_COSTS[task["task_type"]][task["node"]][0]
        if task["node"] != source:
            gain += max(0.0, source_duration - actual_duration)
        max_gain += max(
            0.0,
            max(
                source_duration - duration for duration, _ in TASK_COSTS[task["task_type"]].values()
            ),
        )
    on_time_chain_ids = {
        chain_id for chain_id in required_chains if _chain_on_time(replay, chain_id)
    }
    on_time_chain_ids.update(
        _science_chain_id(zone_id)
        for zone_id in released_zones
        if _chain_on_time(replay, _science_chain_id(zone_id))
    )
    consumed_products = {
        product_id
        for product_id, product in replay["products"].items()
        if product["consumed_by"] is not None and product["chain_id"] in on_time_chain_ids
    }
    useful_units = sum(
        1
        for attempt in replay["attempts"]
        if attempt["outcome"] == "delivered" and attempt["product_id"] in consumed_products
    )
    attempts = len(replay["attempts"])
    gain_term = gain / max_gain if qualifying_tasks and max_gain > 0 else 0.0
    unit_term = useful_units / attempts if attempts else 0.0
    auxiliary_1 = 0.5 * gain_term + 0.5 * unit_term

    service = (
        replay["served_opportunities"] / replay["service_opportunities"]
        if replay["service_opportunities"]
        else 0.0
    )
    useful_attempts = [
        attempt
        for attempt in replay["attempts"]
        if attempt["outcome"] == "delivered" and attempt["product_id"] in consumed_products
    ]
    if useful_attempts:
        waits = [attempt["queue_wait"] for attempt in useful_attempts]
        ages = []
        for attempt in useful_attempts:
            product = replay["products"][attempt["product_id"]]
            consumer = tasks[product["consumed_by"]]
            ages.append(consumer["start"] - attempt["arrival"])
        wait_score = max(0.0, min(1.0, 1.0 - sum(waits) / len(waits) / 40.0))
        age_score = max(0.0, min(1.0, 1.0 - sum(ages) / len(ages) / 40.0))
    else:
        wait_score = 0.0
        age_score = 0.0
    auxiliary_2 = 0.4 * service + 0.3 * wait_score + 0.3 * age_score

    return {
        "primary_metrics_1": max(0.0, min(1.0, primary_1)),
        "primary_metrics_2": max(0.0, min(1.0, primary_2)),
        "primary_metrics_3": max(0.0, min(1.0, primary_3)),
        "auxiliary_metrics_1": max(0.0, min(1.0, auxiliary_1)),
        "auxiliary_metrics_2": max(0.0, min(1.0, auxiliary_2)),
    }


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all scenario replays are required for metrics")
    scenario_metrics = {name: _scenario_metrics(replayed[name]) for name in SCENARIO_NAMES}
    return {
        metric: sum(scenario_metrics[name][metric] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
        for metric in scenario_metrics[SCENARIO_NAMES[0]]
    }


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
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
        title: "\n\n".join(body for body in bodies if body) for title, bodies in occurrences.items()
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
            state.timed_out = state.exit_code is None and "timeout" in stderr.lower()
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
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
