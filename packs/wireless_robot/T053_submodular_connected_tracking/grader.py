from __future__ import annotations

import json
import math
import re
import stat
import time
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y = 20, 16
EPOCHS, EPOCH_SLOTS, MOVE_SLOTS = 12, 10, 8
HORIZON = EPOCHS * EPOCH_SLOTS
ENERGY = 110.0
QUEUE_CAPACITY = 12
PACKET_LIMIT = 24
HOP_LIMIT = 2
AGE_LIMIT = 20
ROBOTS = ("R1", "R2", "R3")
SCENARIOS = ("clustered_targets", "split_targets", "narrow_bridge", "crossing_tracks", "link_burst")
SCENARIO_INDEX = {name: i for i, name in enumerate(SCENARIOS)}
STARTS = {"R1": (2, 4), "R2": (2, 8), "R3": (2, 12)}
TARGET_IDS = tuple(f"T{i:02d}" for i in range(12))
PRIMITIVES: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((0, 0),) * 8,
    1: ((1, 0),) * 8,
    2: ((-1, 0),) * 8,
    3: ((0, 1),) * 8,
    4: ((0, -1),) * 8,
    5: ((1, 1),) * 8,
    6: ((1, -1),) * 8,
    7: ((-1, 1),) * 8,
    8: ((-1, -1),) * 8,
    9: ((1, 0),) * 4 + ((0, 1),) * 4,
    10: ((1, 0),) * 4 + ((0, -1),) * 4,
    11: ((-1, 0),) * 4 + ((0, 1),) * 4,
    12: ((-1, 0),) * 4 + ((0, -1),) * 4,
    13: ((0, 1),) * 4 + ((1, 0),) * 4,
    14: ((0, 1),) * 4 + ((-1, 0),) * 4,
    15: ((0, -1),) * 4 + ((1, 0),) * 4,
    16: ((0, -1),) * 4 + ((-1, 0),) * 4,
}
OBSTACLES = {
    "clustered_targets": (),
    "split_targets": ((7, 8, 4, 11),),
    "narrow_bridge": ((8, 11, 0, 6), (8, 11, 9, 15)),
    "crossing_tracks": ((9, 10, 0, 5), (9, 10, 10, 15)),
    "link_burst": (),
}
TARGET_STARTS = {
    "clustered_targets": (
        (7, 4),
        (8, 5),
        (9, 4),
        (10, 5),
        (11, 4),
        (12, 5),
        (7, 10),
        (8, 11),
        (9, 10),
        (10, 11),
        (11, 10),
        (12, 11),
    ),
    "split_targets": (
        (5, 2),
        (5, 4),
        (5, 6),
        (5, 8),
        (5, 10),
        (5, 12),
        (15, 2),
        (15, 4),
        (15, 6),
        (15, 8),
        (15, 10),
        (15, 12),
    ),
    "narrow_bridge": (
        (5, 3),
        (5, 5),
        (5, 7),
        (5, 9),
        (5, 11),
        (5, 13),
        (14, 3),
        (14, 5),
        (14, 7),
        (14, 9),
        (14, 11),
        (14, 13),
    ),
    "crossing_tracks": (
        (4, 2),
        (6, 3),
        (8, 4),
        (10, 5),
        (12, 6),
        (14, 7),
        (14, 13),
        (12, 12),
        (10, 11),
        (8, 10),
        (6, 9),
        (4, 8),
    ),
    "link_burst": (
        (6, 3),
        (7, 4),
        (8, 5),
        (9, 6),
        (10, 7),
        (11, 8),
        (12, 9),
        (13, 10),
        (14, 11),
        (15, 12),
        (8, 12),
        (12, 3),
    ),
}
TARGET_VALUES = {
    name: tuple(1 + ((i + SCENARIO_INDEX[name]) % 4) for i in range(12)) for name in SCENARIOS
}
PAIR_ORDER = (("R1", "R2"), ("R1", "R3"), ("R2", "R3"))
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.epoch_plans": 0.5,
        "result_evidence.transmission_proposals": 0.5,
    },
    "decision_legality": {
        "decision_legality.plan_keys": 0.25,
        "decision_legality.primitive": 0.25,
        "decision_legality.proposal_keys": 0.2,
        "decision_legality.proposal_shape": 0.3,
    },
    "replay_validity": {
        "replay_validity.motion": 0.2,
        "replay_validity.observations": 0.2,
        "replay_validity.links": 0.2,
        "replay_validity.custody": 0.2,
        "replay_validity.resources": 0.2,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, passed, detail, share


class EvalState:
    def __init__(self) -> None:
        self.run_started = False
        self.run_finished = False
        self.exit_code: int | None = None
        self.timed_out = False
        self.elapsed_s = 0.0
        self.result_file_is_new_regular = False
        self.official_ok = False
        self.official_message = ""
        self.data: Any = None
        self.parse_ok = False
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


def _integer(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


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


def _plan_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"robot", "epoch", "primitive_id"}, path)
    if error:
        return error
    if value["robot"] not in ROBOTS:
        return f"{path}.robot: expected rover"
    if not _integer(value["epoch"], 0, EPOCHS - 1):
        return f"{path}.epoch: expected integer 0..{EPOCHS - 1}"
    if not _integer(value["primitive_id"], 0, 16):
        return f"{path}.primitive_id: expected integer 0..16"
    return None


_OBS_ID = re.compile(r"^o_(R[123])_T(?:0[0-9]|1[01])_e(?:0[0-9]|1[01])_r[0-7]$")


def _proposal_error(value: Any, path: str) -> str | None:
    error = _shape(value, {"slot", "src", "dst", "observation_ids", "priority"}, path)
    if error:
        return error
    if not _integer(value["slot"], 0, HORIZON - 1) or value["slot"] % EPOCH_SLOTS not in (8, 9):
        return f"{path}.slot: expected exchange slot offset 8 or 9"
    if value["src"] not in ROBOTS or value["dst"] not in ROBOTS or value["src"] == value["dst"]:
        return f"{path}.src/dst: expected distinct rover endpoints"
    ids = value["observation_ids"]
    ids_are_strings = isinstance(ids, list) and all(isinstance(x, str) for x in ids)
    ids_valid = (
        isinstance(ids, list)
        and 1 <= len(ids) <= 4
        and ids_are_strings
        and len(ids) == len(set(ids))
        and all(_OBS_ID.fullmatch(x) for x in ids)
    )
    if not ids_valid:
        return f"{path}.observation_ids: expected one to four distinct observation IDs"
    if not _integer(value["priority"], 0, 9):
        return f"{path}.priority: expected integer 0..9"
    return None


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not _finite(value) or value <= 0 for value in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share differs from declaration")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[{}]; {}".format(
        ",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed)
    )


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    values = {
        "result_json.parse": (parse_ok, "strict JSON parse"),
        "result_json.root": (
            isinstance(data, dict) and set(data) == {"scenarios"},
            "root must contain exactly scenarios",
        ),
        "result_json.scenarios": (isinstance(scenarios, list), "scenarios must be an array"),
        "result_json.records": (
            isinstance(scenarios, list)
            and all(
                _shape(x, {"scenario", "epoch_plans", "transmission_proposals"}, f"scenarios[{i}]")
                is None
                for i, x in enumerate(scenarios)
            ),
            "scenario records must have exact keys",
        ),
    }
    return [
        AtomVerdict(i, ok, detail, ATOM_SHARES["result_json"][i])
        for i, (ok, detail) in values.items()
    ]


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in data["scenarios"]:
        if (
            isinstance(row, dict)
            and isinstance(row.get("scenario"), str)
            and row["scenario"] in SCENARIOS
        ):
            result.setdefault(row["scenario"], row)
    return result


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    good = {
        "epoch_plans": [True, "epoch plan rows are well formed"],
        "transmission_proposals": [True, "proposal rows are well formed"],
    }
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        for field, validator in (
            ("epoch_plans", _plan_error),
            ("transmission_proposals", _proposal_error),
        ):
            values = row.get(field)
            if not isinstance(values, list):
                good[field] = [False, f"{name}.{field}: expected array"]
                continue
            for i, item in enumerate(values):
                error = validator(item, f"{name}.{field}[{i}]")
                if error:
                    good[field] = [False, error]
                    break
    return [
        AtomVerdict(
            f"result_evidence.{field}",
            bool(value[0]),
            str(value[1]),
            ATOM_SHARES["result_evidence"][f"result_evidence.{field}"],
        )
        for field, value in good.items()
    ]


def _hashable(value: Any) -> Hashable:
    if isinstance(value, list):
        return tuple(_hashable(x) for x in value)
    try:
        hash(value)
    except Exception:  # noqa: BLE001
        return repr(value)
    return value


def _key_defects(label: str, actual: list[Hashable], expected: set[Hashable]) -> list[str]:
    seen: set[Hashable] = set()
    defects: list[str] = []
    for key in actual:
        if key in seen:
            defects.append(f"{label}: duplicate canonical key")
        seen.add(key)
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
        set(SCENARIOS),
    )
    expected_plans = {(robot, epoch) for robot in ROBOTS for epoch in range(EPOCHS)}
    for name in SCENARIOS:
        row = next((x for x in rows if isinstance(x, dict) and x.get("scenario") == name), None)
        if row is None:
            continue
        plans = row.get("epoch_plans")
        if not isinstance(plans, list):
            defects.append(f"{name}.epoch_plans: missing array")
            continue
        defects.extend(
            _key_defects(
                f"{name}.epoch_plans",
                [
                    _hashable((x.get("robot"), x.get("epoch")))
                    if isinstance(x, dict)
                    else _hashable(x)
                    for x in plans
                ],
                expected_plans,
            )
        )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _decision_map(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record.get("epoch_plans"), list) or not isinstance(
        record.get("transmission_proposals"), list
    ):
        return None
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    for row in record["epoch_plans"]:
        if isinstance(row, dict) and _plan_error(row, "plan") is None:
            plans.setdefault((row["robot"], row["epoch"]), row)
    proposals: dict[tuple[int, str], dict[str, Any]] = {}
    duplicates: list[tuple[int, str]] = []
    for row in record["transmission_proposals"]:
        if isinstance(row, dict) and _proposal_error(row, "proposal") is None:
            key = (row["slot"], row["src"])
            if key in proposals:
                duplicates.append(key)
            else:
                proposals[key] = row
    return {"plans": plans, "proposals": proposals, "duplicate_proposals": duplicates}


def _obstacle(name: str, cell: tuple[int, int]) -> bool:
    x, y = cell
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in OBSTACLES[name])


def _valid_cell(name: str, cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < GRID_X and 0 <= cell[1] < GRID_Y and not _obstacle(name, cell)


def _target_positions(name: str, epoch: int) -> tuple[tuple[int, int], ...]:
    positions = list(TARGET_STARTS[name])
    for e in range(epoch):
        for j, (x, y) in enumerate(positions):
            if name == "clustered_targets":
                dx = -1 if e % 4 == 1 and x > 9 else 1 if e % 4 == 3 and x < 12 else 0
                candidate = (x + dx, y)
            elif name == "split_targets":
                candidate = (x, y + (1 if (j + e) % 2 == 0 else -1))
            elif name == "narrow_bridge":
                direction = (
                    -1 if e % 4 in (0, 1) and x > 8 else 1 if e % 4 in (2, 3) and x < 14 else 0
                )
                candidate = (x + direction, y)
            elif name == "crossing_tracks":
                dirs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
                if j >= 6:
                    dirs = tuple(reversed(dirs))
                dx, dy = dirs[e % 4]
                candidate = (x + dx, y + dy)
            else:
                dirs = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
                dx, dy = dirs[(e // 3 + j) % 8]
                candidate = (x + dx, y + dy)
            positions[j] = candidate if _valid_cell(name, candidate) else (x, y)
    return tuple(positions)


def _line_clear(name: str, first: tuple[int, int], second: tuple[int, int]) -> bool:
    def round_half_down(value: float) -> int:
        # Explicitly resolve exact half-integers toward the lower integer.
        return math.floor(value + 0.5 - 1e-12)

    for step in range(1, 6):
        ratio = step / 6.0
        cell = (
            round_half_down(first[0] + ratio * (second[0] - first[0])),
            round_half_down(first[1] + ratio * (second[1] - first[1])),
        )
        if cell not in (first, second) and _obstacle(name, cell):
            return False
    return True


def _available(
    name: str,
    epoch: int,
    offset: int,
    first: tuple[int, int],
    second: tuple[int, int],
    pair_index: int,
) -> bool:
    if math.dist(first, second) > 5.0 + EPS or not _line_clear(name, first, second):
        return False
    s = SCENARIO_INDEX[name]
    gate = ((5 * (s + 1) + 7 * (epoch + 1) + 3 * pair_index + 2 * (offset - 8)) % 11) != 0
    if name == "link_burst" and epoch % 3 == 1 and offset == 9:
        gate = pair_index != 1
    return gate


def _loss(name: str, epoch: int, offset: int, pair_index: int) -> bool:
    residues = {
        "clustered_targets": {0},
        "split_targets": {0, 1},
        "narrow_bridge": {0, 1},
        "crossing_tracks": {0, 1, 2, 3},
        "link_burst": {0, 1, 2},
    }[name]
    return (
        3571 * (SCENARIO_INDEX[name] + 1)
        + 101 * (epoch + 1)
        + 17 * (offset - 8)
        + 13 * pair_index
        + 7
    ) % 19 in residues


def _probability(robot: tuple[int, int], target: tuple[int, int], name: str) -> float:
    if not _valid_cell(name, robot):
        return 0.0
    if not _line_clear(name, robot, target):
        return 0.0
    return max(0.0, 1.0 - math.dist(robot, target) / 6.0)


def _observation_id(robot: str, target: int, epoch: int, step: int) -> str:
    return f"o_{robot}_T{target:02d}_e{epoch:02d}_r{step}"


def _simulate(name: str, decisions: Mapping[str, Any]) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError("unknown scenario")
    plans = decisions.get("plans", {})
    proposals = decisions.get("proposals", {})
    positions = dict(STARTS)
    energies = {robot: ENERGY for robot in ROBOTS}
    queues: dict[str, list[str]] = {robot: [] for robot in ROBOTS}
    seen: dict[str, set[str]] = {robot: set() for robot in ROBOTS}
    observations: dict[str, dict[str, Any]] = {}
    arrivals: dict[int, list[tuple[dict[str, Any], bool]]] = {}
    replay_violations = {
        "motion": [],
        "observations": [],
        "links": [],
        "custody": [],
        "resources": [],
    }
    packets: list[dict[str, Any]] = []
    collision = 0
    move_count = 0
    selected_count = 0
    duplicate_deliveries = 0
    eligible_value = 0.0
    fused_value = 0.0
    fused_ids: set[str] = set()
    useful_packet_ids: set[str] = set()
    sent_bits = 0.0
    useful_bits = 0.0
    wait_samples: list[float] = []
    fresh_samples: list[float] = []
    coverage_values: list[float] = []
    connected_epochs = 0
    successful_hops = 0
    all_hops = 0

    def fail(rule: str, detail: str) -> None:
        replay_violations[rule].append(detail)

    def process_arrivals(slot: int, terminal: bool = False) -> None:
        nonlocal duplicate_deliveries, fused_value, successful_hops
        for packet, delivered in arrivals.pop(slot, []):
            if not delivered:
                for oid in reversed(packet["observation_ids"]):
                    queues[packet["src"]].insert(0, oid)
                continue
            dst = packet["dst"]
            for oid in packet["observation_ids"]:
                all_hops_local = 1
                del all_hops_local
                if oid in seen[dst]:
                    duplicate_deliveries += 1
                    continue
                seen[dst].add(oid)
                while len(queues[dst]) >= QUEUE_CAPACITY:
                    queues[dst].pop(0)
                queues[dst].append(oid)
                successful_hops += 1
                obs = observations.get(oid)
                if (
                    not terminal
                    and obs is not None
                    and dst != obs["robot"]
                    and slot - obs["slot"] <= AGE_LIMIT
                    and oid not in fused_ids
                ):
                    fused_ids.add(oid)
                    fused_value += obs["value"]
                    fresh_samples.append(max(0.0, min(1.0, 1.0 - (slot - obs["slot"]) / AGE_LIMIT)))
                    wait_samples.append(
                        max(0.0, min(1.0, 1.0 - (packet["service_slot"] - obs["slot"]) / AGE_LIMIT))
                    )
                    useful_packet_ids.add(packet["packet_id"])
                if obs is not None:
                    obs["hop_count"] = max(
                        obs.get("hop_count", 0), packet.get("hop_counts", {}).get(oid, 0)
                    )

    for epoch in range(EPOCHS):
        primitive_ids: dict[str, int] = {}
        for robot in ROBOTS:
            row = plans.get((robot, epoch))
            primitive_id = row.get("primitive_id", 0) if row is not None else 0
            if primitive_id not in PRIMITIVES:
                fail("motion", f"{name}:{robot},epoch={epoch}: unknown primitive")
                primitive_id = 0
            primitive_ids[robot] = primitive_id
        trajectory: dict[str, list[tuple[int, int]]] = {robot: [] for robot in ROBOTS}
        epoch_links: dict[tuple[str, str], bool] = {}
        for step in range(MOVE_SLOTS):
            slot = epoch * EPOCH_SLOTS + step
            process_arrivals(slot)
            old_positions = positions.copy()
            proposed = old_positions.copy()
            for robot in ROBOTS:
                dx, dy = PRIMITIVES[primitive_ids[robot]][step]
                candidate = (old_positions[robot][0] + dx, old_positions[robot][1] + dy)
                if not _valid_cell(name, candidate):
                    fail("motion", f"{name}:{robot},slot={slot}: primitive leaves free grid")
                    candidate = old_positions[robot]
                proposed[robot] = candidate
            involved: set[str] = set()
            for index, first in enumerate(ROBOTS):
                for second in ROBOTS[index + 1 :]:
                    same_cell = proposed[first] == proposed[second]
                    swap = (
                        proposed[first] == old_positions[second]
                        and proposed[second] == old_positions[first]
                        and proposed[first] != old_positions[first]
                    )
                    if same_cell or swap:
                        involved.update((first, second))
            positions = {
                robot: old_positions[robot] if robot in involved else proposed[robot]
                for robot in ROBOTS
            }
            if involved:
                collision += len(involved)
            move_count += sum(positions[r] != old_positions[r] for r in ROBOTS)
            for robot in ROBOTS:
                energies[robot] -= 1.0 if positions[robot] != old_positions[robot] else 0.05
            for robot in ROBOTS:
                trajectory[robot].append(positions[robot])
                for target_index, target in enumerate(_target_positions(name, epoch)):
                    p = _probability(positions[robot], target, name)
                    if p <= 0:
                        continue
                    h = (
                        7919 * (SCENARIO_INDEX[name] + 1)
                        + 104729 * (epoch + 1)
                        + 1009 * (step + 1)
                        + 31 * (target_index + 1)
                        + 17 * (ROBOTS.index(robot) + 1)
                    ) % 10000
                    if h < math.floor(10000 * p):
                        oid = _observation_id(robot, target_index, epoch, step)
                        if oid not in observations:
                            value = float(TARGET_VALUES[name][target_index])
                            observations[oid] = {
                                "robot": robot,
                                "target": target_index,
                                "epoch": epoch,
                                "slot": slot,
                                "value": value,
                                "hop_count": 0,
                            }
                            seen[robot].add(oid)
                            while len(queues[robot]) >= QUEUE_CAPACITY:
                                queues[robot].pop(0)
                            queues[robot].append(oid)

        # Compute official joint submodular coverage from all action outcomes.
        epoch_value = 0.0
        for target_index, target in enumerate(_target_positions(name, epoch)):
            factors = [
                _probability(cell, target, name) for robot in ROBOTS for cell in trajectory[robot]
            ]
            epoch_value += TARGET_VALUES[name][target_index] * (
                1.0 - math.prod(1.0 - p for p in factors)
            )
        denom = float(sum(TARGET_VALUES[name]))
        coverage_values.append(epoch_value / denom if denom else 0.0)

        for offset in (8, 9):
            slot = epoch * EPOCH_SLOTS + offset
            process_arrivals(slot)
            endpoint = positions.copy()
            avail: dict[tuple[str, str], bool] = {}
            for pair_index, (first, second) in enumerate(PAIR_ORDER):
                value = _available(
                    name, epoch, offset, endpoint[first], endpoint[second], pair_index
                )
                avail[(first, second)] = avail[(second, first)] = value
            for pair in PAIR_ORDER:
                epoch_links[pair] = epoch_links.get(pair, False) or avail.get(pair, False)
            valid: list[dict[str, Any]] = []
            for src in ROBOTS:
                row = proposals.get((slot, src))
                if row is None:
                    continue
                ids = row.get("observation_ids", [])
                dst = row.get("dst")
                if not avail.get((src, dst), False):
                    fail("links", f"{name}: unavailable link at slot {slot}")
                    continue
                if queues[src][: len(ids)] != ids:
                    fail("custody", f"{name}: proposal is not a FIFO prefix at slot {slot}")
                    continue
                if len(ids) > 4:
                    fail("links", f"{name}: proposal exceeds packet capacity")
                    continue
                hop_invalid = False
                for oid in ids:
                    obs = observations.get(oid)
                    if obs is None:
                        continue
                    if obs.get("hop_count", 0) >= HOP_LIMIT:
                        fail("resources", f"{name}: observation {oid} exceeds hop limit")
                        hop_invalid = True
                if hop_invalid:
                    continue
                valid.append(row)
            chosen: list[dict[str, Any]] = []
            used: set[str] = set()
            for row in sorted(
                valid, key=lambda x: (-x["priority"], x["src"], x["dst"], x["observation_ids"][0])
            ):
                if selected_count >= PACKET_LIMIT:
                    break
                if row["src"] in used or row["dst"] in used:
                    continue
                chosen.append(row)
                used.update((row["src"], row["dst"]))
            for row in chosen:
                src, dst = row["src"], row["dst"]
                ids = list(row["observation_ids"])
                for oid in ids:
                    if oid not in observations:
                        fail(
                            "observations", f"{name}: proposal references unknown observation {oid}"
                        )
                for oid in ids:
                    if oid in queues[src]:
                        queues[src].remove(oid)
                pair = tuple(sorted((src, dst)))
                pair_index = PAIR_ORDER.index(pair)
                delivered = not _loss(name, epoch, offset, pair_index)
                packet_id = f"pkt_{selected_count:03d}"
                packet = {
                    "packet_id": packet_id,
                    "src": src,
                    "dst": dst,
                    "observation_ids": ids,
                    "service_slot": slot,
                    "arrival_slot": slot + 1,
                    "hop_counts": {
                        oid: observations.get(oid, {}).get("hop_count", 0) + 1 for oid in ids
                    },
                }
                packets.append(packet)
                arrivals.setdefault(slot + 1, []).append((packet, delivered))
                selected_count += 1
                sent_bits += 256.0
                all_hops += len(ids)
                if delivered:
                    successful_hops += 0
        if _connected(ROBOTS, epoch_links):
            connected_epochs += 1

    process_arrivals(HORIZON, terminal=True)
    for oid, obs in observations.items():
        eligible_value += obs["value"]
    return {
        "scenario": name,
        "observations": observations,
        "packets": packets,
        "queues": queues,
        "coverage_values": coverage_values,
        "connected_epochs": connected_epochs,
        "fused_value": fused_value,
        "eligible_value": eligible_value,
        "fused_ids": fused_ids,
        "collision": collision,
        "move_count": move_count,
        "energy_used": sum(ENERGY - energy for energy in energies.values()),
        "selected_count": selected_count,
        "duplicate_deliveries": duplicate_deliveries,
        "useful_packet_count": len(useful_packet_ids),
        "sent_bits": sent_bits,
        "useful_bits": useful_bits + 48.0 * len(fused_ids),
        "fresh_samples": fresh_samples,
        "wait_samples": wait_samples,
        "all_hops": all_hops,
        "successful_hops": successful_hops,
        "replay_violations": replay_violations,
    }


def _connected(nodes: Sequence[str], links: Mapping[tuple[str, str], bool]) -> bool:
    reached = {nodes[0]}
    changed = True
    while changed:
        changed = False
        for first, second in PAIR_ORDER:
            if links.get((first, second), False) and ((first in reached) != (second in reached)):
                reached.update((first, second))
                changed = True
    return len(reached) == len(nodes)


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules = {
        "plan_keys": [True, "plan keys are unique"],
        "primitive": [True, "primitive IDs are legal"],
        "proposal_keys": [True, "proposal keys are unique"],
        "proposal_shape": [True, "proposal endpoints and priorities are legal"],
    }
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        decisions = _decision_map(row)
        if decisions is None:
            continue
        if len(decisions["plans"]) != len(row["epoch_plans"]):
            rules["plan_keys"] = [False, f"{name}: duplicate or malformed plan key"]
        if any(
            _plan_error(x, f"{name}.epoch_plans") is not None
            or x.get("primitive_id") not in PRIMITIVES
            for x in row["epoch_plans"]
        ):
            rules["primitive"] = [False, f"{name}: illegal primitive"]
        if decisions["duplicate_proposals"]:
            rules["proposal_keys"] = [
                False,
                f"{name}: duplicate proposal key {decisions['duplicate_proposals'][0]!r}",
            ]
        if any(
            _proposal_error(x, f"{name}.transmission_proposals") is not None
            for x in row["transmission_proposals"]
        ):
            rules["proposal_shape"] = [False, f"{name}: malformed proposal"]
    return [
        AtomVerdict(
            f"decision_legality.{key}",
            bool(value[0]),
            str(value[1]),
            ATOM_SHARES["decision_legality"][f"decision_legality.{key}"],
        )
        for key, value in rules.items()
    ]


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = {
        "motion": [True, "motion replay is consistent"],
        "observations": [True, "observation generation is consistent"],
        "links": [True, "link and arbitration replay is consistent"],
        "custody": [True, "FIFO custody is consistent"],
        "resources": [True, "packet, hop, and energy resources are consistent"],
    }
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        decisions = _decision_map(row)
        if decisions is None:
            continue
        state = _simulate(name, decisions)
        replayed[name] = state
        for rule, details in state["replay_violations"].items():
            if details:
                target = "motion" if rule == "motion" else rule
                if target in rules:
                    rules[target] = [False, details[0]]
        if state["selected_count"] > PACKET_LIMIT:
            rules["resources"] = [False, f"{name}: packet budget exceeded"]
    return [
        AtomVerdict(
            f"replay_validity.{key}",
            bool(value[0]),
            str(value[1]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{key}"],
        )
        for key, value in rules.items()
    ], replayed


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scenario_metrics(state: Mapping[str, Any]) -> dict[str, float]:
    coverage = sum(state["coverage_values"]) / EPOCHS
    connected = state["connected_epochs"] / EPOCHS
    fused = state["fused_value"] / state["eligible_value"] if state["eligible_value"] else 0.0
    on_time = fused
    primary_2 = 0.55 * connected + 0.45 * on_time
    primary_3 = coverage * (
        0.4 * _clip(1.0 - state["move_count"] / 288.0)
        + 0.3 * _clip(1.0 - state["collision"] / 288.0)
        + 0.3 * _clip(1.0 - state["energy_used"] / 330.0)
    )
    auxiliary_1 = 0.6 * state["useful_bits"] / max(1.0, state["sent_bits"]) + 0.4 * state[
        "useful_packet_count"
    ] / max(1.0, state["selected_count"])
    fresh = (
        sum(state["fresh_samples"]) / len(state["fresh_samples"]) if state["fresh_samples"] else 0.0
    )
    wait = sum(state["wait_samples"]) / len(state["wait_samples"]) if state["wait_samples"] else 0.0
    redundancy = 1.0 - state["duplicate_deliveries"] / max(1, state["selected_count"])
    auxiliary_2 = 0.4 * fresh + 0.3 * _clip(redundancy) + 0.3 * wait
    return {
        "primary_metrics_1": _clip(coverage),
        "primary_metrics_2": _clip(primary_2),
        "primary_metrics_3": _clip(primary_3),
        "auxiliary_metrics_1": _clip(auxiliary_1),
        "auxiliary_metrics_2": _clip(auxiliary_2),
    }


def _metrics(
    replayed: Mapping[str, dict[str, Any]],
    records: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    if set(replayed) != set(SCENARIOS):
        raise ValueError("all scenario replays are required")
    values = [_scenario_metrics(replayed[name]) for name in SCENARIOS]
    return {metric: sum(item[metric] for item in values) / len(values) for metric in values[0]}


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    result: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*$", line)
        if match:
            title = match.group(1)
            end = next(
                (
                    j
                    for j in range(index + 1, len(lines))
                    if re.match(r"^[ \t]{0,3}#{1,6}[ \t]+", lines[j])
                ),
                len(lines),
            )
            result[title] = lines[index + 1 : end] and "\n".join(lines[index + 1 : end]).strip()
    return result


def _report_score(text: str | None) -> tuple[float, str]:
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    found = _headings(text) if text else {}
    missing = [title for title in required if not found.get(title, "")]
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
    result_path = Path(ws.root, "results.json")
    cleanup_ok = False
    cleanup_detail = ""
    run_detail = ""
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            cleanup_ok = getattr(removed, "exit_code", None) == 0
            if not cleanup_ok:
                cleanup_detail = f"results cleanup exit={getattr(removed, 'exit_code', None)}"
        except Exception as exc:  # noqa: BLE001
            cleanup_detail = f"results cleanup exception: {type(exc).__name__}: {exc}"
        state.run_started = True
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
        except Exception as exc:  # noqa: BLE001
            state.timed_out = "timeout" in str(exc).lower()
            run_detail = f"run exception: {type(exc).__name__}: {exc}"
            result = None
        elapsed = time.monotonic() - started
        state.run_finished = result is not None
        state.elapsed_s = elapsed
        state.exit_code = getattr(result, "exit_code", None)
        try:
            file_stat = result_path.lstat()
            state.result_file_is_new_regular = (
                stat.S_ISREG(file_stat.st_mode) and not result_path.is_symlink()
            )
        except OSError:
            state.result_file_is_new_regular = False
        state.official_ok = (
            cleanup_ok
            and state.exit_code == 0
            and not state.timed_out
            and elapsed <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"cleanup_ok={cleanup_ok}, "
            f"exit={state.exit_code}, elapsed={elapsed:.2f}s, timed_out={state.timed_out}, "
            f"run_finished={state.run_finished}, "
            f"new_regular_results={state.result_file_is_new_regular}"
            + (f", {cleanup_detail}" if cleanup_detail else "")
            + (f", {run_detail}" if run_detail else "")
        )
        if state.official_ok:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except Exception as exc:  # noqa: BLE001
                state.diagnostics["result_json"] = (
                    f"strict parse failed: {type(exc).__name__}: {exc}"
                )
        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            for check_id in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = "official run or strict result parsing failed"
            _CACHE[key] = state
            return state
        state.records = _index_records(state.data)
        evidence = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence)
        coverage, defects = _coverage(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = (
            "failed_atoms=[]; defect_count=0"
            if not defects
            else f"defect_count={len(defects)}; " + " | ".join(defects[:8])
        )
        legality = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality)
        replay, replayed = _replay_atoms(state.records)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay)
        gate_names = (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        )
        state.validity_gate = state.official_ok and all(
            state.scores.get(x, 0.0) >= 1.0 - EPS for x in gate_names
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001
        message = f"grader analysis failed: {type(exc).__name__}: {exc}"
        state.official_message = state.official_message or message
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, message)
    _CACHE[key] = state
    return state


async def _metric(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{name}: validity gate not satisfied; metric not computed"
    return state.metrics[name], f"{name}: independently replayed value={state.metrics[name]:.9g}"


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "run.py: static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "run.py: static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected regular file"
    try:
        text = ws.read_text("report.md")
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"
    score, message = _report_score(text)
    return score, message


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
    return await _metric(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "primary_metrics_2")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "primary_metrics_3")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric(ws, "auxiliary_metrics_2")
