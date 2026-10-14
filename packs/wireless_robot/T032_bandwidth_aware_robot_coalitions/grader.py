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

HORIZON = 72
STAGE_LENGTH = 24
STAGES = 3
ROBOTS = 6
GRID_X = 22
GRID_Y = 16
ENERGY0 = 120.0
ATTEMPT_BUDGET = 216
SLOT_CAPACITY = 4
EPS = 1e-6

SCENARIO_NAMES = (
    "overlapping_targets",
    "corridor_conflict",
    "nonlocal_dependency",
    "value_shift",
    "bandwidth_saturation",
)
TARGET_POSITIONS = (
    (2, 2),
    (2, 13),
    (6, 3),
    (6, 12),
    (10, 2),
    (10, 13),
    (14, 3),
    (14, 12),
    (18, 2),
    (18, 13),
    (20, 5),
    (20, 10),
)
GROUPS = (
    (0, 4, 8),
    (1, 5, 9),
    (2, 6, 10),
    (3, 7, 11),
)
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
COMMS = ("none", "new", "retry")
SCENARIOS: dict[str, dict[str, Any]] = {
    "overlapping_targets": {
        "starts": ((0, 1), (0, 4), (0, 7), (0, 9), (0, 12), (0, 14)),
        "obstacles": ((8, 9, 4, 11),),
        "values": (2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5),
        "bonuses": (5, 5, 6, 6),
        "loss": 0.08,
    },
    "corridor_conflict": {
        "starts": ((1, 1), (1, 5), (1, 10), (1, 14), (8, 1), (8, 14)),
        "obstacles": ((7, 16, 5, 6), (7, 16, 9, 10)),
        "values": (2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5),
        "bonuses": (6, 6, 6, 6),
        "loss": 0.10,
    },
    "nonlocal_dependency": {
        "starts": ((0, 2), (0, 13), (5, 1), (5, 14), (11, 1), (11, 14)),
        "obstacles": ((9, 10, 3, 12),),
        "values": (1, 1, 3, 3, 2, 2, 5, 5, 3, 3, 6, 6),
        "bonuses": (8, 8, 5, 5),
        "loss": 0.12,
    },
    "value_shift": {
        "starts": ((1, 2), (1, 13), (7, 2), (7, 13), (13, 2), (13, 13)),
        "obstacles": ((4, 5, 5, 10), (15, 16, 5, 10)),
        "values": (4, 1, 4, 1, 1, 5, 1, 5, 6, 2, 6, 2),
        "bonuses": (5, 7, 5, 7),
        "loss": 0.10,
    },
    "bandwidth_saturation": {
        "starts": ((0, 1), (0, 4), (0, 7), (0, 9), (0, 12), (0, 14)),
        "obstacles": ((11, 12, 2, 13),),
        "values": (3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5),
        "bonuses": (7, 7, 7, 7),
        "loss": 0.18,
    },
}

SCENARIO_KEYS = frozenset({"scenario", "partitions", "actions"})
PARTITION_KEYS = frozenset({"stage", "coalitions"})
ACTION_KEYS = frozenset(
    {
        "t",
        "robot",
        "dx",
        "dy",
        "intent_target",
        "plan_ref",
        "comm",
        "packet_id",
        "recipient",
        "plan_target",
        "plan_dx1",
        "plan_dy1",
        "plan_dx2",
        "plan_dy2",
    }
)
OPTIONAL_ID_FIELDS = ("intent_target", "plan_ref", "packet_id", "recipient", "plan_target")
PLAN_MOVE_FIELDS = ("plan_dx1", "plan_dy1", "plan_dx2", "plan_dy2")

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.scenario_rows": 0.20,
        "result_evidence.partition_rows": 0.25,
        "result_evidence.action_rows": 0.30,
        "result_evidence.action_values": 0.25,
    },
    "decision_legality": {
        "decision_legality.partitions": 1.0 / 6.0,
        "decision_legality.motion": 1.0 / 6.0,
        "decision_legality.intents": 1.0 / 6.0,
        "decision_legality.packets": 1.0 / 6.0,
        "decision_legality.plan_refs": 1.0 / 6.0,
        "decision_legality.energy": 1.0 / 6.0,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.switches": 0.15,
        "replay_validity.packets": 0.25,
        "replay_validity.targets": 0.20,
        "replay_validity.metrics": 0.20,
    },
}
GATE_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)


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


def _brief(value: Any, limit: int = 120) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # huge/malformed values must stay diagnosable
        return f"<{type(value).__name__} repr failed: {type(exc).__name__}>"
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject,
        parse_float=finite_float,
        object_pairs_hook=unique,
    )


def _blocked(spec: Mapping[str, Any], position: tuple[int, int]) -> bool:
    x, y = position
    return any(
        xmin <= x <= xmax and ymin <= y <= ymax for xmin, xmax, ymin, ymax in spec["obstacles"]
    )


def _same_coalition(partition: tuple[tuple[int, ...], ...], first: int, second: int) -> bool:
    return any(first in coalition and second in coalition for coalition in partition)


def _coalition_of(partition: tuple[tuple[int, ...], ...], robot: int) -> tuple[int, ...]:
    for coalition in partition:
        if robot in coalition:
            return coalition
    return (robot,)


def _partners(partition: tuple[tuple[int, ...], ...], robot: int) -> frozenset[int]:
    return frozenset(member for member in _coalition_of(partition, robot) if member != robot)


def _partition_value(value: Any) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(value, list) or not value:
        return None
    coalitions: list[tuple[int, ...]] = []
    flat: list[int] = []
    for coalition in value:
        if not isinstance(coalition, list) or not coalition or len(coalition) > 4:
            return None
        if not all(_integer(member) for member in coalition):
            return None
        members = tuple(sorted(coalition))
        coalitions.append(members)
        flat.extend(members)
    if sorted(flat) != list(range(ROBOTS)) or len(set(flat)) != ROBOTS:
        return None
    return tuple(sorted(coalitions))


def _partition_typed(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and all(
            isinstance(coalition, list) and all(_integer(member) for member in coalition)
            for coalition in value
        )
    )


def _shape_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    root = isinstance(data, Mapping)
    root_keys = root and set(data) == {"scenarios"}
    scenarios = data.get("scenarios") if root else None
    scenario_list = isinstance(scenarios, list)
    records = bool(
        scenario_list is True
        and all(
            isinstance(record, Mapping)
            and set(record) == SCENARIO_KEYS
            and isinstance(record.get("partitions"), list)
            and isinstance(record.get("actions"), list)
            for record in scenarios
        )
    )
    return [
        _atom("result_json", "parse", parse_ok, "strict JSON parse"),
        _atom(
            "result_json",
            "root",
            bool(root_keys),
            f"expected exact root keys ['scenarios']; got {_brief(data)}",
        ),
        _atom("result_json", "scenarios", scenario_list, "scenarios must be a JSON array"),
        _atom(
            "result_json",
            "records",
            records,
            "every scenario row must have exact keys and array containers",
        ),
    ]


def _index_records(data: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return result
    for record in data["scenarios"]:
        if not isinstance(record, Mapping):
            continue
        name = record.get("scenario")
        if isinstance(name, str) and name not in result:
            result[name] = record
    return result


def _index(data: Any) -> dict[str, Mapping[str, Any]]:
    """Backward-compatible alias for the specification's canonical index."""
    return _index_records(data)


def _partition_rows(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("partitions")
    return value if isinstance(value, list) else []


def _action_rows(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("actions")
    return value if isinstance(value, list) else []


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    raw = data.get("scenarios") if isinstance(data, Mapping) else None
    records = raw if isinstance(raw, list) else []
    scenario_rows = True
    partition_rows = True
    action_rows = True
    action_values = True
    details = {
        "scenario_rows": "scenario rows and identifiers are typed",
        "partition_rows": "partition rows use exact keys and integer member arrays",
        "action_rows": "action rows use the exact public field set",
        "action_values": "action scalar and conditional-null fields are JSON integers, strings, or null as declared",
    }

    def fail(atom_name: str, path: str, message: str) -> None:
        nonlocal scenario_rows, partition_rows, action_rows, action_values
        if atom_name == "scenario_rows":
            scenario_rows = False
        elif atom_name == "partition_rows":
            partition_rows = False
        elif atom_name == "action_rows":
            action_rows = False
        else:
            action_values = False
        if details[atom_name].startswith(("scenario rows", "partition rows", "action rows", "action scalar")):
            details[atom_name] = f"{path}: {message}"

    for index, record in enumerate(records):
        path = f"scenarios[{index}]"
        if not isinstance(record, Mapping) or not isinstance(record.get("scenario"), str):
            fail("scenario_rows", path, "expected an object with a string scenario key")
            continue
        for partition_index, row in enumerate(_partition_rows(record)):
            row_path = f"{path}.partitions[{partition_index}]"
            if (
                not isinstance(row, Mapping)
                or set(row) != PARTITION_KEYS
                or not _integer(row.get("stage"))
                or not isinstance(row.get("coalitions"), list)
            ):
                fail("partition_rows", row_path, "invalid keys, stage, or coalition container")
                continue
            for coalition_index, coalition in enumerate(row["coalitions"]):
                if not isinstance(coalition, list) or not all(_integer(member) for member in coalition):
                    fail(
                        "partition_rows",
                        f"{row_path}.coalitions[{coalition_index}]",
                        "members must be an integer list",
                    )
        for action_index, row in enumerate(_action_rows(record)):
            row_path = f"{path}.actions[{action_index}]"
            if not isinstance(row, Mapping) or set(row) != ACTION_KEYS:
                fail("action_rows", row_path, "expected the exact public action key set")
                continue
            if not all(_integer(row.get(field)) for field in ("t", "robot", "dx", "dy")):
                fail("action_values", row_path, "t, robot, dx, and dy must be integers")
            elif not isinstance(row.get("comm"), str):
                fail("action_values", row_path, "comm must be a string")
            elif not all(_optional_integer(row.get(field)) for field in OPTIONAL_ID_FIELDS):
                fail("action_values", row_path, "optional ID fields must be integer or null")
            elif not all(
                row.get(field) is None or _integer(row.get(field)) for field in PLAN_MOVE_FIELDS
            ):
                fail("action_values", row_path, "plan move fields must be integer or null")
    return [
        _atom(
            "result_evidence",
            "scenario_rows",
            scenario_rows,
            details["scenario_rows"],
        ),
        _atom(
            "result_evidence",
            "partition_rows",
            partition_rows,
            details["partition_rows"],
        ),
        _atom(
            "result_evidence",
            "action_rows",
            action_rows,
            details["action_rows"],
        ),
        _atom(
            "result_evidence",
            "action_values",
            action_values,
            details["action_values"],
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
            defects.append(f"unknown scenario key {_brief(name)}")
    for name in SCENARIO_NAMES:
        count = scenario_counts.get(name, 0)
        if count == 0:
            defects.append(f"missing scenario {name}")
        elif count > 1:
            defects.extend([f"duplicate scenario {name}"] * (count - 1))

    indexed = _index_records(data)
    for name in SCENARIO_NAMES:
        record = indexed.get(name)
        if record is None:
            continue
        stage_counts: dict[int, int] = {}
        for index, row in enumerate(_partition_rows(record)):
            if not isinstance(row, Mapping) or not _integer(row.get("stage")):
                defects.append(f"{name}.partitions[{index}] has no integer stage key")
                continue
            stage = row["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if stage not in range(STAGES):
                defects.append(f"{name} has unknown stage {_brief(stage)}")
        for stage in range(STAGES):
            count = stage_counts.get(stage, 0)
            if count == 0:
                defects.append(f"{name} missing stage {stage}")
            elif count > 1:
                defects.extend([f"{name} duplicate stage {stage}"] * (count - 1))

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


def _partition_index(record: Mapping[str, Any]) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for row in _partition_rows(record):
        if isinstance(row, Mapping) and _integer(row.get("stage")) and row["stage"] not in result:
            result[row["stage"]] = row.get("coalitions")
    return result


def _action_index(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in _action_rows(record):
        if (
            not isinstance(row, Mapping)
            or not _integer(row.get("t"))
            or not _integer(row.get("robot"))
        ):
            continue
        key = (row["t"], row["robot"])
        if key not in result:
            result[key] = row
    return result


def _blank_action(t: int, robot: int) -> dict[str, Any]:
    return {
        "t": t,
        "robot": robot,
        "dx": 0,
        "dy": 0,
        "intent_target": None,
        "plan_ref": None,
        "comm": "none",
        "packet_id": None,
        "recipient": None,
        "plan_target": None,
        "plan_dx1": None,
        "plan_dy1": None,
        "plan_dx2": None,
        "plan_dy2": None,
    }


def _packet_success(
    scenario_index: int, loss: float, t: int, source: int, recipient: int, attempt: int
) -> bool:
    stage = t // STAGE_LENGTH
    u = (
        (
            97 * (t + 1)
            + 31 * (source + 1)
            + 17 * (recipient + 1)
            + 13 * (stage + 1)
            + 19 * (attempt + 1)
            + 29 * (scenario_index + 1)
        )
        % 1000
        / 1000.0
    )
    return u >= loss


def _replay(name: str, scenario: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    scenario_index = SCENARIO_NAMES.index(name)
    partition_rows = _partition_index(scenario)
    actions = _action_index(scenario)
    fallback_partition = tuple((robot,) for robot in range(ROBOTS))
    partitions: dict[int, tuple[tuple[int, ...], ...]] = {}
    errors: dict[str, list[str]] = {
        key: [] for key in ("partitions", "motion", "intents", "packets", "plan_refs", "energy")
    }
    for stage in range(STAGES):
        raw_partition = partition_rows.get(stage)
        canonical = _partition_value(raw_partition)
        if raw_partition is not None and _partition_typed(raw_partition) and canonical is None:
            errors["partitions"].append(
                f"{name}.partitions[stage={stage}].coalitions is not a size-1..4 partition of robots 0..5"
            )
        partitions[stage] = canonical or fallback_partition

    positions = [tuple(position) for position in spec["starts"]]
    energies = [ENERGY0] * ROBOTS
    position_history: list[tuple[tuple[int, int], ...]] = [tuple(positions)]
    energy_history: list[tuple[float, ...]] = [tuple(energies)]
    submitted_moves: dict[tuple[int, int], tuple[int, int]] = {}
    completions: dict[int, int] = {}
    packets: dict[int, dict[str, Any]] = {}
    scheduled: dict[int, list[tuple[int, int, bool]]] = {}
    accepted_attempts = 0
    accepted_per_slot = [0] * HORIZON
    safety_events = 0
    move_steps = 0
    repeats = 0
    switches = 0
    previous_partition: tuple[tuple[int, ...], ...] | None = None

    def process_feedback(boundary: int) -> None:
        for packet_id, attempt, success in scheduled.pop(boundary, []):
            packet = packets[packet_id]
            if success:
                packet["delivered"] = True
                packet["delivery_boundary"] = boundary
                packet["status"] = "delivered"
            elif (
                attempt == 0
                and boundary < HORIZON
                and packet["generation"] // STAGE_LENGTH == boundary // STAGE_LENGTH
            ):
                packet["retry_due"] = boundary
                packet["status"] = "first_failed"
            else:
                packet["status"] = "expired"

    for t in range(HORIZON):
        for packet in packets.values():
            if (
                packet.get("retry_due") is not None
                and packet["retry_due"] < t
                and not packet.get("retry_attempted")
            ):
                packet["status"] = "expired"
        process_feedback(t)

        stage = t // STAGE_LENGTH
        partition = partitions[stage]
        if t % STAGE_LENGTH == 0:
            if previous_partition is not None:
                for robot in range(ROBOTS):
                    if _partners(previous_partition, robot) != _partners(partition, robot):
                        switches += 1
                        energies[robot] -= 1.0
                        if energies[robot] < -EPS:
                            errors["energy"].append(
                                f"{name} robot {robot} is negative after stage-{stage} switch: {energies[robot]:.6g}"
                            )
            previous_partition = partition

        slot_actions: dict[int, Mapping[str, Any]] = {
            robot: actions.get((t, robot), _blank_action(t, robot)) for robot in range(ROBOTS)
        }
        intents: dict[int, int | None] = {}
        valid_plan_refs: list[tuple[int, int]] = []
        attempts: list[dict[str, Any]] = []

        for robot, row in slot_actions.items():
            dx = row.get("dx")
            dy = row.get("dy")
            move = (dx, dy) if _integer(dx) and _integer(dy) else (0, 0)
            if _integer(dx) and _integer(dy) and move not in MOVES:
                errors["motion"].append(
                    f"{name}.actions[{t},{robot}] move {_brief(move)} is outside the five-move domain"
                )
                move = (0, 0)
            submitted_moves[(t, robot)] = move

            raw_intent = row.get("intent_target")
            intent = _null_id(raw_intent) if _optional_integer(raw_intent) else None
            if _optional_integer(raw_intent) and raw_intent not in (None, -1):
                active = range(stage * 4, stage * 4 + 4)
                if intent not in active or intent in completions:
                    errors["intents"].append(
                        f"{name}.actions[{t},{robot}].intent_target={_brief(intent)} is not active and incomplete at action start"
                    )
                    intent = None
            intents[robot] = intent

            raw_ref = row.get("plan_ref")
            plan_ref = _null_id(raw_ref) if _optional_integer(raw_ref) else None
            if _optional_integer(raw_ref) and raw_ref not in (None, -1):
                packet = packets.get(plan_ref)
                valid = bool(
                    packet
                    and packet.get("delivered")
                    and packet["recipient"] == robot
                    and packet["generation"] // STAGE_LENGTH == stage
                    and t in (packet["generation"] + 1, packet["generation"] + 2)
                    and _same_coalition(partition, packet["source"], robot)
                )
                if not valid:
                    errors["plan_refs"].append(
                        f"{name}.actions[{t},{robot}].plan_ref={_brief(plan_ref)} is not a delivered valid plan for this recipient and slot"
                    )
                else:
                    valid_plan_refs.append((robot, int(plan_ref)))

            comm = row.get("comm")
            if not isinstance(comm, str):
                continue
            if comm not in COMMS:
                errors["packets"].append(f"{name}.actions[{t},{robot}].comm={comm!r} is unknown")
                continue
            raw_ids = {field: row.get(field) for field in ("packet_id", "recipient", "plan_target")}
            ids = {
                field: _null_id(value) if _optional_integer(value) else None
                for field, value in raw_ids.items()
            }
            plan_values = tuple(row.get(field) for field in PLAN_MOVE_FIELDS)
            if not all(_optional_integer(value) for value in raw_ids.values()) or not all(
                value is None or _integer(value) for value in plan_values
            ):
                continue
            if comm == "none":
                if any(value not in (None, -1) for value in raw_ids.values()) or any(
                    value is not None for value in plan_values
                ):
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] none must have null packet summary fields"
                    )
                continue
            if comm == "new":
                packet_id = ids["packet_id"]
                recipient = ids["recipient"]
                plan_target = ids["plan_target"]
                packet_ok = True
                if packet_id is None or packet_id < 0:
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] new packet_id must be nonnegative"
                    )
                    packet_ok = False
                elif packet_id in packets:
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] reuses scenario packet_id={_brief(packet_id)}"
                    )
                    packet_ok = False
                if (
                    recipient is None
                    or recipient not in range(ROBOTS)
                    or recipient == robot
                    or not _same_coalition(partition, robot, recipient)
                ):
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] new recipient={_brief(recipient)} is not a different current-coalition member"
                    )
                    packet_ok = False
                if plan_target is None or plan_target != intents[robot]:
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] plan_target={_brief(plan_target)} must equal the non-null legal intent_target"
                    )
                    packet_ok = False
                prediction1 = (plan_values[0], plan_values[1])
                prediction2 = (plan_values[2], plan_values[3])
                if (
                    not all(_integer(value) for value in plan_values)
                    or prediction1 not in MOVES
                    or prediction2 not in MOVES
                ):
                    errors["packets"].append(
                        f"{name}.actions[{t},{robot}] new plan predictions are outside the five-move domain"
                    )
                    packet_ok = False
                if packet_id is not None and packet_id >= 0 and packet_id not in packets:
                    packets[packet_id] = {
                        "source": robot,
                        "recipient": recipient,
                        "generation": t,
                        "target": plan_target,
                        "predictions": (prediction1, prediction2),
                        "delivered": False,
                        "delivery_boundary": None,
                        "retry_due": None,
                        "retry_attempted": False,
                        "coordinated": False,
                        "cited": False,
                        "status": "submitted",
                    }
                if packet_ok:
                    attempts.append(
                        {
                            "kind": "new",
                            "packet_id": packet_id,
                            "source": robot,
                            "recipient": recipient,
                            "generation": t,
                            "attempt": 0,
                        }
                    )
                continue

            packet_id = ids["packet_id"]
            if any(
                raw_ids[field] not in (None, -1) for field in ("recipient", "plan_target")
            ) or any(value is not None for value in plan_values):
                errors["packets"].append(
                    f"{name}.actions[{t},{robot}] retry must contain only packet_id"
                )
                continue
            packet = packets.get(packet_id) if packet_id is not None else None
            retry_ok = bool(
                packet_id is not None
                and packet_id >= 0
                and packet
                and packet["source"] == robot
                and packet.get("retry_due") == t
                and not packet.get("retry_attempted")
                and packet["generation"] // STAGE_LENGTH == stage
                and _same_coalition(partition, robot, packet["recipient"])
            )
            if not retry_ok:
                errors["packets"].append(
                    f"{name}.actions[{t},{robot}] retry packet_id={_brief(packet_id)} lacks the immediate legal retry opportunity"
                )
            else:
                attempts.append(
                    {
                        "kind": "retry",
                        "packet_id": packet_id,
                        "source": robot,
                        "recipient": packet["recipient"],
                        "generation": packet["generation"],
                        "attempt": 1,
                    }
                )

        attempts.sort(
            key=lambda attempt: (
                0 if attempt["kind"] == "retry" else 1,
                attempt["generation"],
                _coalition_of(partition, attempt["source"]),
                attempt["source"],
                attempt["recipient"],
                attempt["packet_id"],
            )
        )
        admitted_count = min(
            SLOT_CAPACITY, max(0, ATTEMPT_BUDGET - accepted_attempts), len(attempts)
        )
        admitted = attempts[:admitted_count]
        admitted_ids = {(attempt["kind"], attempt["packet_id"]) for attempt in admitted}
        for attempt in attempts:
            packet = packets[attempt["packet_id"]]
            if attempt["kind"] == "retry":
                packet["retry_attempted"] = True
            if (attempt["kind"], attempt["packet_id"]) not in admitted_ids:
                packet["status"] = "dropped"
                continue
            accepted_attempts += 1
            accepted_per_slot[t] += 1
            packet["status"] = "in_flight"
            success = _packet_success(
                scenario_index,
                spec["loss"],
                t,
                attempt["source"],
                attempt["recipient"],
                attempt["attempt"],
            )
            scheduled.setdefault(t + 1, []).append(
                (attempt["packet_id"], attempt["attempt"], success)
            )

        for packet in packets.values():
            if packet.get("retry_due") == t and not packet.get("retry_attempted"):
                packet["status"] = "expired"

        accepted_destinations: list[tuple[int, int]] = []
        next_positions: list[tuple[int, int]] = []
        for robot in range(ROBOTS):
            current = positions[robot]
            move = submitted_moves[(t, robot)]
            destination = (current[0] + move[0], current[1] + move[1])
            rejected = bool(
                destination[0] not in range(GRID_X)
                or destination[1] not in range(GRID_Y)
                or _blocked(spec, destination)
                or destination in accepted_destinations
                or destination in positions[robot + 1 :]
            )
            if rejected:
                destination = current
                if move != (0, 0):
                    safety_events += 1
            accepted_destinations.append(destination)
            next_positions.append(destination)

        admitted_by_robot = {robot: 0 for robot in range(ROBOTS)}
        for attempt in admitted:
            admitted_by_robot[attempt["source"]] += 1
        for robot in range(ROBOTS):
            moved = next_positions[robot] != positions[robot]
            movement_cost = 2.0 if moved else 0.2
            energies[robot] -= movement_cost + admitted_by_robot[robot]
            if energies[robot] < -EPS:
                errors["energy"].append(
                    f"{name}.actions[{t},{robot}] actual costs make energy negative: {energies[robot]:.6g} with tolerance {EPS}"
                )
            if moved:
                move_steps += 1
        positions = next_positions

        for robot in range(ROBOTS):
            for target in range(stage * 4, stage * 4 + 4):
                target_position = TARGET_POSITIONS[target]
                if (
                    abs(positions[robot][0] - target_position[0])
                    + abs(positions[robot][1] - target_position[1])
                    <= 1
                ):
                    if target not in completions:
                        completions[target] = t + 1
                    else:
                        repeats += 1

        for _recipient, packet_id in valid_plan_refs:
            packet = packets[packet_id]
            offset = t - packet["generation"]
            source_move = submitted_moves.get((t, packet["source"]))
            if not packet.get("cited"):
                packet["cited"] = True
                packet["citation_slot"] = t
                recipient = packet["recipient"]
                source_start = position_history[-1][packet["source"]]
                recipient_start = position_history[-1][recipient]
                source_destination = (
                    source_start[0] + source_move[0],
                    source_start[1] + source_move[1],
                )
                recipient_move = submitted_moves[(t, recipient)]
                recipient_destination = (
                    recipient_start[0] + recipient_move[0],
                    recipient_start[1] + recipient_move[1],
                )
                target_deconflicted = (
                    intents[recipient] is None or intents[recipient] != packet["target"]
                )
                destinations_deconflicted = source_destination != recipient_destination and not (
                    source_destination == recipient_start and recipient_destination == source_start
                )
                if (
                    offset in (1, 2)
                    and source_move == packet["predictions"][offset - 1]
                    and target_deconflicted
                    and destinations_deconflicted
                ):
                    packet["coordinated"] = True

        position_history.append(tuple(positions))
        energy_history.append(tuple(energies))

    process_feedback(HORIZON)
    for packet in packets.values():
        if packet.get("retry_due") == HORIZON:
            packet["status"] = "expired"

    return {
        "name": name,
        "partitions": partitions,
        "positions": tuple(position_history),
        "energies": tuple(energy_history),
        "submitted_moves": submitted_moves,
        "completions": completions,
        "packets": packets,
        "accepted_attempts": accepted_attempts,
        "accepted_per_slot": tuple(accepted_per_slot),
        "safety_events": safety_events,
        "move_steps": move_steps,
        "repeats": repeats,
        "switches": switches,
        "energy_used": ROBOTS * ENERGY0 - sum(energies),
        "errors": errors,
    }


def _legality(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    failures: dict[str, list[str]] = {
        key: [] for key in ("partitions", "motion", "intents", "packets", "plan_refs", "energy")
    }
    for name in SCENARIO_NAMES:
        record = scenarios.get(name)
        if not isinstance(record, Mapping):
            continue
        replayed = _replay(name, record)
        for key, details in replayed["errors"].items():
            failures[key].extend(details)

    def detail(key: str) -> str:
        values = failures[key]
        return "passed" if not values else " | ".join(values[:8])

    return [
        _atom("decision_legality", "partitions", not failures["partitions"], detail("partitions")),
        _atom("decision_legality", "motion", not failures["motion"], detail("motion")),
        _atom("decision_legality", "intents", not failures["intents"], detail("intents")),
        _atom("decision_legality", "packets", not failures["packets"], detail("packets")),
        _atom("decision_legality", "plan_refs", not failures["plan_refs"], detail("plan_refs")),
        _atom("decision_legality", "energy", not failures["energy"], detail("energy")),
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    totals = {key: 0.0 for key in METRIC_KEYS}
    for name in SCENARIO_NAMES:
        state = replayed[name]
        spec = SCENARIOS[name]
        completions = state["completions"]
        target_reward = sum(
            value for target, value in enumerate(spec["values"]) if target in completions
        )
        group_reward = sum(
            bonus
            for group, bonus in zip(GROUPS, spec["bonuses"])
            if all(target in completions for target in group)
        )
        maximum_reward = sum(spec["values"]) + sum(spec["bonuses"])
        totals["primary_metrics_1"] += _clip((target_reward + group_reward) / maximum_reward)

        response_numerator = 0.0
        for target, value in enumerate(spec["values"]):
            completion = completions.get(target)
            stage_start = STAGE_LENGTH * (target // 4)
            quality = (
                _clip(1.0 - (completion - stage_start) / STAGE_LENGTH)
                if completion is not None
                else 0.0
            )
            response_numerator += value * quality
        totals["primary_metrics_2"] += _clip(response_numerator / sum(spec["values"]))

        health = (
            0.5 * _clip(1.0 - state["safety_events"] / (HORIZON * ROBOTS))
            + 0.3 * _clip(1.0 - state["move_steps"] / (HORIZON * ROBOTS))
            + 0.2 * _clip(1.0 - state["energy_used"] / (ROBOTS * ENERGY0))
        )
        totals["primary_metrics_3"] += _clip(health)

        accepted = state["accepted_attempts"]
        coordinated_packets = [
            packet for packet in state["packets"].values() if packet.get("coordinated")
        ]
        coordinated = len(coordinated_packets)
        recipient_stage_coverage = len(
            {
                (packet["recipient"], packet["generation"] // STAGE_LENGTH)
                for packet in coordinated_packets
            }
        )
        communication = (
            0.0
            if accepted == 0
            else _clip(recipient_stage_coverage / (ROBOTS * STAGES))
            * (0.6 + 0.4 * _clip(coordinated / accepted))
        )
        totals["auxiliary_metrics_1"] += _clip(communication)

        accurate_cells: set[tuple[int, int, int]] = set()
        for packet in state["packets"].values():
            if not packet.get("delivered"):
                continue
            generation = packet["generation"]
            for offset in (1, 2):
                slot = generation + offset
                if slot < HORIZON and slot // STAGE_LENGTH == generation // STAGE_LENGTH:
                    if (
                        state["submitted_moves"].get((slot, packet["source"]))
                        == packet["predictions"][offset - 1]
                    ):
                        accurate_cells.add((packet["source"], generation // STAGE_LENGTH, offset))
        prediction_coverage = _clip(len(accurate_cells) / (ROBOTS * STAGES * 2))
        coalition_continuity = _clip(1.0 - state["switches"] / 12.0)
        stability = prediction_coverage * (0.7 + 0.3 * coalition_continuity)
        totals["auxiliary_metrics_2"] += _clip(stability)

    return {key: _clip(value / len(SCENARIO_NAMES)) for key, value in totals.items()}


def _replay_atoms(scenarios: Mapping[str, Any], replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    complete = set(replayed) == set(SCENARIO_NAMES)
    motion = complete and all(
        len(state["positions"]) == HORIZON + 1
        and len(state["energies"]) == HORIZON + 1
        and all(
            len(set(boundary)) == ROBOTS
            and all(
                position[0] in range(GRID_X)
                and position[1] in range(GRID_Y)
                and not _blocked(SCENARIOS[name], position)
                for position in boundary
            )
            for boundary in state["positions"]
        )
        for name, state in replayed.items()
    )
    switches = complete and all(0 <= state["switches"] <= 12 for state in replayed.values())
    packets = complete and all(
        0 <= state["accepted_attempts"] <= ATTEMPT_BUDGET
        and len(state["accepted_per_slot"]) == HORIZON
        and all(0 <= count <= SLOT_CAPACITY for count in state["accepted_per_slot"])
        and all(
            not packet.get("delivered") or packet.get("delivery_boundary") is not None
            for packet in state["packets"].values()
        )
        for state in replayed.values()
    )
    targets = complete and all(
        all(
            STAGE_LENGTH * (target // 4) + 1 <= boundary <= STAGE_LENGTH * (target // 4 + 1)
            for target, boundary in state["completions"].items()
        )
        and state["safety_events"] >= 0
        and state["repeats"] >= 0
        for state in replayed.values()
    )
    metric_values: dict[str, float] = {}
    if complete:
        try:
            metric_values = _metrics(replayed)
        except Exception:
            metric_values = {}
    metrics = set(metric_values) == set(METRIC_KEYS) and all(
        math.isfinite(value) and 0 <= value <= 1 for value in metric_values.values()
    )
    return [
        _atom(
            "replay_validity",
            "motion",
            motion,
            "73 valid collision-resolved position and energy boundaries per scenario",
        ),
        _atom(
            "replay_validity", "switches", switches, "partner-set switch replay at stages 1 and 2"
        ),
        _atom(
            "replay_validity",
            "packets",
            packets,
            "shared capacity, budget, loss, retry, delivery, and use replay",
        ),
        _atom(
            "replay_validity",
            "targets",
            targets,
            "stage observation, completion, dependency, repeat, and safety replay",
        ),
        _atom("replay_validity", "metrics", metrics, "five finite normalized replay metrics"),
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
        state.scenarios = _index_records(state.data)
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
