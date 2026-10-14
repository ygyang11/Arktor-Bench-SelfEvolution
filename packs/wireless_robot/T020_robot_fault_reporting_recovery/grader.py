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

HORIZON = 48
SCENES = (
    "pallet_drop",
    "sorting_shift",
    "blocked_pick",
    "multi_fault",
    "late_confirmation",
)
STAGE_WORK = (5, 6, 5, 5)
STAGE_DEADLINES = (10, 22, 35, 48)
STAGE_VALUES = (1, 1, 2, 2)
FIELDS = ("relation", "pose", "stage", "safety")
FIELD_BITS = {"relation": 24, "pose": 16, "stage": 8, "safety": 8}
REPORT_HEADER_BITS = 8
REPORT_BUDGET = 480
REPORT_QUEUE_CAPACITY = 2
COMMAND_QUEUE_CAPACITY = 2
ACTIONS = ("none", "stop", "regrasp", "reposition", "retry", "bypass")
ACTION_KIND = {
    "regrasp": "drop",
    "reposition": "pose",
    "retry": "placement",
    "bypass": "blockage",
}
ACTION_DURATION = {"stop": 1, "regrasp": 2, "reposition": 2, "retry": 1, "bypass": 3}
VALIDITY_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
METRIC_NAMES = (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
FIELD_KINDS = {
    "relation": frozenset({"drop", "placement"}),
    "pose": frozenset({"pose"}),
    "safety": frozenset({"blockage"}),
}
UPLINK_CAPACITY = {
    "pallet_drop": (48, 32, 0, 24, 40, 32, 0, 48),
    "sorting_shift": (32, 0, 24, 48, 32, 24, 40, 0),
    "blocked_pick": (24, 32, 0, 24, 48, 0, 32, 40),
    "multi_fault": (40, 24, 0, 32, 24, 48, 0, 32),
    "late_confirmation": (32, 24, 40, 0, 24, 32, 48, 0),
}
FAULT_DATA = {
    "pallet_drop": (
        ("F01", 0, 1, "drop", None, None),
        ("F02", 1, 3, "drop", None, None),
        ("F03", 2, 2, "placement", None, None),
        ("F04", 3, 2, "pose", 0.20, None),
    ),
    "sorting_shift": (
        ("F01", 0, 2, "pose", 0.25, None),
        ("F02", 1, 2, "blockage", None, True),
        ("F03", 2, 1, "placement", None, None),
        ("F04", 3, 2, "pose", 0.20, None),
    ),
    "blocked_pick": (
        ("F01", 0, 1, "drop", None, None),
        ("F02", 0, 1, "blockage", None, True),
        ("F03", 1, 4, "pose", 0.25, None),
        ("F04", 2, 2, "placement", None, None),
        ("F05", 3, 1, "blockage", None, True),
    ),
    "multi_fault": (
        ("F01", 0, 1, "drop", None, None),
        ("F02", 1, 2, "drop", None, None),
        ("F03", 1, 2, "blockage", None, True),
        ("F04", 2, 1, "placement", None, None),
        ("F05", 2, 1, "pose", 0.25, None),
        ("F06", 3, 2, "blockage", None, True),
    ),
    "late_confirmation": (
        ("F01", 0, 3, "pose", 0.20, None),
        ("F02", 1, 4, "blockage", None, True),
        ("F03", 2, 3, "placement", None, None),
        ("F04", 3, 3, "pose", 0.25, None),
    ),
}
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "report_fields", "command_action"}
EPS = 1e-9

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.34,
        "result_evidence.report_fields": 0.33,
        "result_evidence.command_actions": 0.33,
    },
    "decision_legality": {
        "decision_legality.report_field_set": 0.34,
        "decision_legality.report_budget": 0.33,
        "decision_legality.command_information": 0.33,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.fault_and_stage": 1 / 6,
        "replay_validity.uplink_and_processing": 1 / 6,
        "replay_validity.edge_knowledge": 1 / 6,
        "replay_validity.downlink_and_action": 1 / 6,
        "replay_validity.terminal_accounting": 1 / 6,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = bool(passed)
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
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _safe_repr(value: Any, limit: int = 180) -> str:
    try:
        rendered = repr(value)
    except (OverflowError, ValueError):
        if _integer(value):
            return f"<integer bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate object member {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[group][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    parse_ok = parsed
    root_ok = parse_ok and _shape(data, ROOT_KEYS) and isinstance(data["scenarios"], list)
    scenarios = data.get("scenarios") if parse_ok and isinstance(data, Mapping) else None
    containers_ok = isinstance(scenarios, list) and all(
        _shape(row, SCENE_KEYS)
        and isinstance(row["scenario"], str)
        and isinstance(row["slots"], list)
        for row in scenarios
    )
    return [
        _atom(
            "result_json",
            "parse",
            parse_ok,
            "strict JSON parsed" if parse_ok else "parse failed",
        ),
        _atom(
            "result_json",
            "root",
            root_ok,
            (
                "exact root keys and scenarios list"
                if root_ok
                else "expected exact root key scenarios"
            ),
        ),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            (
                "scene containers have exact keys"
                if containers_ok
                else "each scene needs exact keys scenario and slots with stated types"
            ),
        ),
    ]


def _scene_rows(data: Any) -> list[Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return []
    return [row for row in data["scenarios"] if isinstance(row, Mapping)]


def _slot_rows(data: Any) -> list[tuple[Any, Mapping[str, Any]]]:
    rows: list[tuple[Any, Mapping[str, Any]]] = []
    for scene_row in _scene_rows(data):
        slots = scene_row.get("slots")
        if not isinstance(slots, list):
            continue
        for slot in slots:
            if isinstance(slot, Mapping):
                rows.append((scene_row.get("scenario"), slot))
    return rows


def _raw_slot_entries(data: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return entries
    for scene_index, scene_row in enumerate(data["scenarios"]):
        if not isinstance(scene_row, Mapping) or not isinstance(scene_row.get("slots"), list):
            continue
        for slot_index, slot in enumerate(scene_row["slots"]):
            entries.append((f"scenarios[{scene_index}].slots[{slot_index}]", slot))
    return entries


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    entries = _raw_slot_entries(data)
    container_issue = next(
        (
            f"{path}: expected object with exact keys {sorted(SLOT_KEYS)}, "
            f"got {_safe_repr(slot)}"
            for path, slot in entries
            if not _shape(slot, SLOT_KEYS)
        ),
        None,
    )
    shaped = [(path, slot) for path, slot in entries if _shape(slot, SLOT_KEYS)]
    report_issue = next(
        (
            f"{path}.report_fields: expected a list of field enums, "
            f"got {_safe_repr(slot['report_fields'])}"
            for path, slot in shaped
            if not isinstance(slot["report_fields"], list)
            or any(
                not isinstance(field, str) or field not in FIELDS
                for field in slot["report_fields"]
            )
        ),
        None,
    )
    action_issue = next(
        (
            f"{path}.command_action: expected one of {ACTIONS}, "
            f"got {_safe_repr(slot['command_action'])}"
            for path, slot in shaped
            if not isinstance(slot["command_action"], str)
            or slot["command_action"] not in ACTIONS
        ),
        None,
    )
    return [
        _atom(
            "result_evidence",
            "slot_containers",
            container_issue is None,
            "slot objects have exact keys" if container_issue is None else container_issue,
        ),
        _atom(
            "result_evidence",
            "report_fields",
            report_issue is None,
            "report field values valid" if report_issue is None else report_issue,
        ),
        _atom(
            "result_evidence",
            "command_actions",
            action_issue is None,
            "command actions valid" if action_issue is None else action_issue,
        ),
    ]


def _key_defects(values: Sequence[Any], expected: set[Any], label: str) -> list[str]:
    counts: dict[Any, int] = {}
    defects: list[str] = []
    for value in values:
        try:
            counts[value] = counts.get(value, 0) + 1
        except TypeError:
            defects.append(f"{label} unhashable key={_safe_repr(value)}")
    for key in sorted(expected, key=str):
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"{label} missing key={key!r}")
        elif count > 1:
            defects.extend(f"{label} duplicate key={key!r}" for _ in range(count - 1))
    for key, count in counts.items():
        if key not in expected:
            defects.extend(f"{label} unknown key={_safe_repr(key)}" for _ in range(count))
    return defects


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    rows = _scene_rows(data)
    defects = _key_defects([row.get("scenario") for row in rows], set(SCENES), "scenario")
    by_scene: dict[str, list[Mapping[str, Any]]] = {scene: [] for scene in SCENES}
    for row in rows:
        scene = row.get("scenario")
        if isinstance(scene, str) and scene in by_scene:
            by_scene[scene].append(row)
    for scene, matches in by_scene.items():
        if len(matches) != 1 or not isinstance(matches[0].get("slots"), list):
            continue
        values = [
            slot.get("t")
            if _integer(slot.get("t"))
            else ("invalid_slot_key", _safe_repr(slot.get("t")))
            for slot in matches[0]["slots"]
            if isinstance(slot, Mapping)
        ]
        non_objects = sum(not isinstance(slot, Mapping) for slot in matches[0]["slots"])
        defects.extend(f"{scene} slot non-object" for _ in range(non_objects))
        defects.extend(_key_defects(values, set(range(HORIZON)), f"{scene} slot"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene_row in _scene_rows(data):
        scene = scene_row.get("scenario")
        slots = scene_row.get("slots")
        if not isinstance(scene, str) or scene not in SCENES or not isinstance(slots, list):
            continue
        for slot in slots:
            if not isinstance(slot, Mapping) or not _integer(slot.get("t")):
                continue
            key = (scene, slot["t"])
            if key not in records:
                records[key] = slot
    return records


def _raw_decision(records: Mapping[Hashable, Any], scene: str, t: int) -> dict[str, Any]:
    value = records.get((scene, t))
    if not isinstance(value, Mapping):
        return {"report_fields": [], "command_action": "none"}
    return {
        "report_fields": value.get("report_fields"),
        "command_action": value.get("command_action"),
    }


def _placement_label(scene: str) -> str:
    return "on" if scene in {"pallet_drop", "multi_fault"} else "inside"


def _relation_label(scene: str, stage: int) -> str:
    return ("holding", "holding", _placement_label(scene), "aligned")[stage]


def _report_lost(scene: str, generated: int, completed: int) -> bool:
    return (3 * SCENES.index(scene) + generated + completed) % 13 in {0, 1}


def _command_lost(scene: str, generated: int, served: int) -> bool:
    return (5 * SCENES.index(scene) + generated + served) % 17 == 0


def _payload(
    stage: int,
    progress: int,
    relation_label: str,
    relation_ok: bool,
    pose: float,
    path_clear: bool,
    alternate_path: bool,
    action: Mapping[str, Any] | None,
    failure_lock: bool,
) -> dict[str, Any]:
    return {
        "relation": (relation_label, relation_ok),
        "pose": pose,
        "stage": (
            stage,
            progress,
            None if action is None else action["action"],
            None if action is None else action["end"],
            failure_lock,
        ),
        "safety": (path_clear, alternate_path),
    }


def _local_view(
    t: int,
    stage: int,
    progress: int,
    relation_label: str,
    relation_ok: bool,
    pose: float,
    path_clear: bool,
    alternate_path: bool,
    action: Mapping[str, Any] | None,
    failure_lock: bool,
    queue: Sequence[Mapping[str, Any]],
    used_bits: int,
) -> dict[str, Any]:
    return {
        "t": t,
        "stage": stage,
        "progress": progress,
        "relation_label": relation_label,
        "relation_satisfied": relation_ok,
        "pose_deviation": pose,
        "path_clear": path_clear,
        "alternate_path": alternate_path,
        "active_action": None if action is None else action["action"],
        "action_completion": None if action is None else action["end"],
        "failure_lock": failure_lock,
        "queue": [
            {
                "generation_slot": report["generation_slot"],
                "fields": list(report["fields"]),
                "bits": report["bits"],
                "remaining_bits": report["remaining_bits"],
            }
            for report in queue
        ],
        "used_bits": used_bits,
        "remaining_budget": REPORT_BUDGET - used_bits,
    }


def _edge_view(
    t: int,
    processed: Mapping[int, Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "t": t,
        "processed_reports": [
            {
                "generation_slot": report["generation_slot"],
                "processing_boundary": report["processing_boundary"],
                "fields": list(report["fields"]),
                "payload": dict(report["payload"]),
            }
            for _, report in sorted(processed.items())
        ],
        "latest": {
            field: {"generation_slot": fact["generation_slot"], "value": fact["value"]}
            for field, fact in latest.items()
        },
        "queue": [
            {
                "generation_slot": command["generation_slot"],
                "action": command["action"],
                "evidence_slot": command["evidence_slot"],
            }
            for command in queue
        ],
    }


def _normalize_policy_decision(value: Any, local: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return (
            {"report_fields": []}
            if local
            else {"command_action": "none"}
        )
    if local:
        fields = value.get("report_fields", [])
        return {"report_fields": list(fields) if isinstance(fields, (list, tuple)) else fields}
    return {"command_action": value.get("command_action", "none")}


def _simulate_scene(
    scene: str,
    records: Mapping[Hashable, Any] | None = None,
    local_policy: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    edge_policy: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    stage = 0
    progress = 0
    relation_ok = True
    pose = 0.0
    path_clear = True
    alternate_path = False
    failure_lock = False
    action: dict[str, Any] | None = None
    stage_completions: list[int | None] = [None] * 4
    occurred: dict[str, int | None] = {row[0]: None for row in FAULT_DATA[scene]}
    recovered: dict[str, int | None] = {row[0]: None for row in FAULT_DATA[scene]}
    active_faults: set[str] = set()
    fault_rows = {row[0]: row for row in FAULT_DATA[scene]}
    report_queue: list[dict[str, Any]] = []
    report_arrivals: dict[int, list[dict[str, Any]]] = {}
    processing: dict[int, list[dict[str, Any]]] = {}
    processed_reports: dict[int, dict[str, Any]] = {}
    latest: dict[str, dict[str, Any]] = {}
    command_queue: list[dict[str, Any]] = []
    command_arrivals: dict[int, list[dict[str, Any]]] = {}
    used_bits = 0
    useful_bits = 0
    enqueued_reports: list[dict[str, Any]] = []
    serviced_commands: list[dict[str, Any]] = []
    actions_started: list[dict[str, Any]] = []
    slot_logs: list[dict[str, Any]] = []
    stage_history: list[int] = []
    decisions: list[dict[str, Any]] = []
    local_history: list[dict[str, Any]] = []
    edge_history: list[dict[str, Any]] = []
    legality_issues: dict[str, list[str]] = {
        "report_field_set": [],
        "report_budget": [],
        "command_information": [],
    }

    def complete_action(boundary: int) -> None:
        nonlocal action, relation_ok, pose, path_clear, alternate_path, failure_lock
        if action is None or action["end"] != boundary:
            return
        action["completed"] = True
        if action["action"] == "stop":
            action["success"] = True
            failure_lock = False
        elif action["planned_success"]:
            action["success"] = True
            kind = ACTION_KIND[action["action"]]
            for fault_id in list(active_faults):
                if fault_rows[fault_id][3] == kind:
                    active_faults.remove(fault_id)
                    recovered[fault_id] = boundary
            if kind in {"drop", "placement"}:
                relation_ok = not any(
                    fault_rows[fault_id][3] in {"drop", "placement"}
                    for fault_id in active_faults
                )
            elif kind == "pose":
                pose = 0.0
            else:
                path_clear = True
                alternate_path = False
        else:
            action["success"] = False
            failure_lock = True
        action = None

    def complete_stage(boundary: int) -> None:
        nonlocal stage, progress, relation_ok, pose, path_clear, alternate_path, failure_lock
        if stage >= 4 or action is not None or active_faults or progress < STAGE_WORK[stage]:
            return
        stage_completions[stage] = boundary
        stage += 1
        progress = 0
        relation_ok = True
        pose = 0.0
        path_clear = True
        alternate_path = False
        failure_lock = False

    def start_action(command: dict[str, Any], boundary: int) -> None:
        nonlocal action
        action_name = command["action"]
        if action_name == "stop":
            success = True
            duration = ACTION_DURATION[action_name]
            matching: tuple[str, ...] = ()
        else:
            kind = ACTION_KIND[action_name]
            matching = tuple(
                fault_id
                for fault_id in sorted(active_faults)
                if fault_rows[fault_id][3] == kind
            )
            extra = (
                path_clear
                if action_name in {"regrasp", "retry"}
                else 0.0 < pose <= 0.25 + EPS and path_clear
                if action_name == "reposition"
                else alternate_path
            )
            success = not failure_lock and bool(matching) and bool(extra)
            duration = ACTION_DURATION[action_name] + (0 if success else 2)
        action = {
            "action": action_name,
            "start": boundary,
            "end": boundary + duration,
            "evidence_slot": command["evidence_slot"],
            "planned_success": success,
            "success": False,
            "completed": False,
            "matching_faults": matching,
        }
        actions_started.append(action)

    for t in range(HORIZON):
        complete_action(t)
        complete_stage(t)
        stage_history.append(stage)
        task_complete = stage >= 4

        if not task_complete:
            for fault_id, fault_stage, trigger, kind, magnitude, alternate in FAULT_DATA[scene]:
                if fault_stage != stage or occurred[fault_id] is not None or progress < trigger:
                    continue
                occurred[fault_id] = t
                active_faults.add(fault_id)
                if kind in {"drop", "placement"}:
                    relation_ok = False
                elif kind == "pose":
                    pose = float(magnitude)
                else:
                    path_clear = False
                    alternate_path = bool(alternate)

        for report in processing.pop(t, []):
            report["processing_boundary"] = t
            report_useful = 0
            for field in report["fields"]:
                previous = latest.get(field)
                newer = previous is None or report["generation_slot"] > previous["generation_slot"]
                if newer:
                    if report["relevance"][field]:
                        report_useful += FIELD_BITS[field]
                    latest[field] = {
                        "generation_slot": report["generation_slot"],
                        "value": report["payload"][field],
                    }
            report["useful_bits"] = report_useful
            useful_bits += report_useful
            processed_reports[report["generation_slot"]] = report

        for report in report_arrivals.pop(t, []):
            report["arrival_boundary"] = t
            if t - report["generation_slot"] > 6:
                report["expired"] = True
            else:
                report["expired"] = False
                processing.setdefault(t + 1, []).append(report)

        for command in command_arrivals.pop(t, []):
            command["arrival_boundary"] = t
            if task_complete or t - command["evidence_slot"] > 12 or action is not None:
                command["arrival_outcome"] = "discarded"
            else:
                command["arrival_outcome"] = "started"
                start_action(command, t)

        if task_complete:
            raw = _raw_decision(records or {}, scene, t)
            raw_fields = raw["report_fields"]
            fields_valid = isinstance(raw_fields, list) and all(
                isinstance(field, str) and field in FIELDS for field in raw_fields
            )
            if fields_valid and len(raw_fields) != len(set(raw_fields)):
                legality_issues["report_field_set"].append(
                    f"t={t},duplicate fields={_safe_repr(raw_fields)}"
                )
            decisions.append({"t": t, **raw})
            slot_logs.append(
                {
                    "t": t,
                    "stage": stage,
                    "progress": progress,
                    "active_faults": tuple(),
                    "report_queue": len(report_queue),
                    "command_queue": len(command_queue),
                    "used_bits": used_bits,
                    "uplink_service": 0,
                    "task_complete": True,
                }
            )
            continue

        label = _relation_label(scene, stage)
        local_view = _local_view(
            t,
            stage,
            progress,
            label,
            relation_ok,
            pose,
            path_clear,
            alternate_path,
            action,
            failure_lock,
            report_queue,
            used_bits,
        )
        edge_view = _edge_view(t, processed_reports, latest, command_queue)
        local_view["history"] = tuple(local_history)
        edge_view["history"] = tuple(edge_history)
        if local_policy is not None:
            local_decision = _normalize_policy_decision(local_policy(local_view), True)
        else:
            raw = _raw_decision(records or {}, scene, t)
            local_decision = {"report_fields": raw["report_fields"]}
        if edge_policy is not None:
            edge_decision = _normalize_policy_decision(edge_policy(edge_view), False)
        else:
            raw = _raw_decision(records or {}, scene, t)
            edge_decision = {
                "command_action": raw["command_action"],
            }
        local_history.append({"view": local_view, "decision": dict(local_decision)})
        edge_history.append({"view": edge_view, "decision": dict(edge_decision)})
        raw_fields = local_decision.get("report_fields")
        raw_action = edge_decision.get("command_action")
        decisions.append(
            {
                "t": t,
                "report_fields": raw_fields,
                "command_action": raw_action,
            }
        )

        fields_valid = (
            isinstance(raw_fields, list)
            and all(isinstance(field, str) and field in FIELDS for field in raw_fields)
        )
        unique_fields = fields_valid and len(raw_fields) == len(set(raw_fields))
        if fields_valid and not unique_fields:
            legality_issues["report_field_set"].append(f"t={t},duplicate fields={raw_fields}")
        if communication_enabled and unique_fields and raw_fields:
            ordered_fields = tuple(field for field in FIELDS if field in raw_fields)
            bits = REPORT_HEADER_BITS + sum(FIELD_BITS[field] for field in ordered_fields)
            if len(report_queue) < REPORT_QUEUE_CAPACITY:
                if bits > REPORT_BUDGET - used_bits:
                    legality_issues["report_budget"].append(
                        f"t={t},requested={bits},remaining={REPORT_BUDGET - used_bits}"
                    )
                else:
                    active_kinds = {fault_rows[fault_id][3] for fault_id in active_faults}
                    relevance = {
                        field: (
                            bool(active_kinds & FIELD_KINDS[field])
                            if field in FIELD_KINDS
                            else bool(active_faults) or failure_lock
                        )
                        for field in ordered_fields
                    }
                    report = {
                        "generation_slot": t,
                        "fields": ordered_fields,
                        "bits": bits,
                        "remaining_bits": bits,
                        "payload": {
                            field: _payload(
                                stage,
                                progress,
                                label,
                                relation_ok,
                                pose,
                                path_clear,
                                alternate_path,
                                action,
                                failure_lock,
                            )[field]
                            for field in ordered_fields
                        },
                        "relevance": relevance,
                        "completion_slot": None,
                        "lost": None,
                        "arrival_boundary": None,
                        "expired": None,
                        "processing_boundary": None,
                        "useful_bits": 0,
                    }
                    report_queue.append(report)
                    enqueued_reports.append(report)
                    used_bits += bits

        action_valid = isinstance(raw_action, str) and raw_action in ACTIONS
        evidence_slot = max(processed_reports) if processed_reports else None
        information_valid = evidence_slot is not None and t - evidence_slot <= 10
        if action_valid and raw_action != "none":
            if not information_valid:
                legality_issues["command_information"].append(
                    f"t={t},latest_processed={evidence_slot} is absent or older than 10"
                )
        if (
            communication_enabled
            and action_valid
            and raw_action != "none"
            and information_valid
            and len(command_queue) < COMMAND_QUEUE_CAPACITY
        ):
            command_queue.append(
                {
                    "generation_slot": t,
                    "action": raw_action,
                    "evidence_slot": evidence_slot,
                    "service_slot": None,
                    "lost": None,
                    "arrival_boundary": None,
                    "arrival_outcome": None,
                }
            )

        uplink_service = 0
        if communication_enabled and report_queue:
            capacity = UPLINK_CAPACITY[scene][t % 8]
            report = report_queue[0]
            uplink_service = min(capacity, report["remaining_bits"])
            report["remaining_bits"] -= uplink_service
            if report["remaining_bits"] == 0:
                report_queue.pop(0)
                report["completion_slot"] = t
                report["lost"] = _report_lost(scene, report["generation_slot"], t)
                if not report["lost"] and t + 1 < HORIZON:
                    report_arrivals.setdefault(t + 1, []).append(report)

        if (
            communication_enabled
            and command_queue
            and (SCENES.index(scene) + t) % 5 != 0
        ):
            command = command_queue.pop(0)
            command["service_slot"] = t
            command["lost"] = _command_lost(scene, command["generation_slot"], t)
            serviced_commands.append(command)
            if not command["lost"] and t + 1 < HORIZON:
                command_arrivals.setdefault(t + 1, []).append(command)

        if action is None and not active_faults:
            progress += 1
        slot_logs.append(
            {
                "t": t,
                "stage": stage,
                "progress": progress,
                "active_faults": tuple(sorted(active_faults)),
                "report_queue": len(report_queue),
                "command_queue": len(command_queue),
                "used_bits": used_bits,
                "uplink_service": uplink_service,
                "task_complete": False,
            }
        )

    complete_action(HORIZON)
    complete_stage(HORIZON)
    stage_history.append(stage)
    faults = {
        fault_id: {
            "stage": row[1],
            "trigger": row[2],
            "kind": row[3],
            "occurrence": occurred[fault_id],
            "deadline": (
                None if occurred[fault_id] is None else min(occurred[fault_id] + 8, HORIZON)
            ),
            "recovery": recovered[fault_id],
        }
        for fault_id, row in fault_rows.items()
    }
    return {
        "scene": scene,
        "slots": decisions,
        "consumed_decisions": len(decisions),
        "stage": stage,
        "progress": progress,
        "stage_completions": stage_completions,
        "stage_history": stage_history,
        "faults": faults,
        "active_faults": tuple(sorted(active_faults)),
        "relation_satisfied": relation_ok,
        "pose_deviation": pose,
        "path_clear": path_clear,
        "alternate_path": alternate_path,
        "failure_lock": failure_lock,
        "active_action": action,
        "enqueued_reports": enqueued_reports,
        "processed_reports": processed_reports,
        "latest": latest,
        "serviced_commands": serviced_commands,
        "actions_started": actions_started,
        "slot_logs": slot_logs,
        "used_bits": used_bits,
        "useful_bits": useful_bits,
        "legality_issues": legality_issues,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    return _simulate_scene(scene, records=records)


def _online_scene(
    scene: str,
    local_policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    edge_policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    communication_enabled: bool = True,
) -> dict[str, Any]:
    return _simulate_scene(
        scene,
        local_policy=local_policy,
        edge_policy=edge_policy,
        communication_enabled=communication_enabled,
    )


def _result_from_online(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": [{"scenario": scene, "slots": replayed[scene]["slots"]} for scene in SCENES]
    }


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    atoms: list[AtomVerdict] = []
    for name in ("report_field_set", "report_budget", "command_information"):
        issues = [
            f"{scene}:{issue}"
            for scene in SCENES
            for issue in replayed[scene]["legality_issues"][name]
        ]
        atoms.append(
            _atom(
                "decision_legality",
                name,
                not issues,
                "no violations" if not issues else f"violations={issues[:10]}",
            )
        )
    return atoms


def _decision_consumption_issue(replay: Mapping[str, Any]) -> str | None:
    if replay.get("consumed_decisions") != HORIZON:
        return f"expected consumed_decisions={HORIZON}, got {replay.get('consumed_decisions')}"
    slots = replay.get("slots")
    if not isinstance(slots, list) or len(slots) != HORIZON:
        return f"expected {HORIZON} replay slots"
    for t, slot in enumerate(slots):
        if not isinstance(slot, Mapping) or slot.get("t") != t:
            return f"slot index mismatch at t={t}"
    return None


def _fault_stage_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    history = replay.get("stage_history")
    if not isinstance(history, list) or len(history) != HORIZON + 1:
        return "stage history must have 49 boundaries"
    if any(not _integer(value) or not 0 <= value <= 4 for value in history):
        return "stage history contains invalid stage"
    if any(new < old or new - old > 1 for old, new in zip(history, history[1:])):
        return "stage history is not monotone by single transitions"
    completions = replay.get("stage_completions")
    if not isinstance(completions, list) or len(completions) != 4:
        return "stage completion vector malformed"
    seen_none = False
    previous = -1
    for stage, boundary in enumerate(completions):
        if boundary is None:
            seen_none = True
            continue
        if (
            seen_none
            or not _integer(boundary)
            or not 0 <= boundary <= HORIZON
            or boundary <= previous
        ):
            return f"invalid stage completion stage={stage},boundary={boundary}"
        previous = boundary
    faults = replay.get("faults")
    expected = {row[0] for row in FAULT_DATA[scene]}
    if not isinstance(faults, Mapping) or set(faults) != expected:
        return "fault key set mismatch"
    for fault_id, fault in faults.items():
        if not isinstance(fault, Mapping):
            return f"fault={fault_id} record malformed"
        occurrence = fault.get("occurrence")
        recovery = fault.get("recovery")
        deadline = fault.get("deadline")
        if occurrence is None:
            if recovery is not None or deadline is not None:
                return f"fault={fault_id} recovered or dated before occurrence"
        elif (
            not _integer(occurrence)
            or not 0 <= occurrence < HORIZON
            or deadline != min(occurrence + 8, HORIZON)
            or (
                recovery is not None
                and (not _integer(recovery) or not occurrence < recovery <= HORIZON)
            )
        ):
            return f"fault={fault_id} timing malformed"
    return None


def _uplink_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    reports = replay.get("enqueued_reports")
    if not isinstance(reports, list):
        return "enqueued report log missing"
    generations: set[int] = set()
    total_bits = 0
    for report in reports:
        if not isinstance(report, Mapping):
            return "report log contains non-object"
        generated = report.get("generation_slot")
        fields = report.get("fields")
        bits = report.get("bits")
        if (
            not _integer(generated)
            or not 0 <= generated < HORIZON
            or generated in generations
            or not isinstance(fields, tuple)
            or not fields
            or any(field not in FIELDS for field in fields)
            or bits != REPORT_HEADER_BITS + sum(FIELD_BITS[field] for field in fields)
        ):
            return f"malformed report={_safe_repr(report)}"
        generations.add(generated)
        total_bits += bits
        completed = report.get("completion_slot")
        if completed is not None:
            if not _integer(completed) or not generated <= completed < HORIZON:
                return f"report={generated} completion invalid"
            if report.get("lost") is not _report_lost(scene, generated, completed):
                return f"report={generated} loss outcome invalid"
            arrival = report.get("arrival_boundary")
            if report["lost"] or completed + 1 >= HORIZON:
                if arrival is not None:
                    return f"report={generated} has impossible arrival"
            elif arrival != completed + 1:
                return f"report={generated} arrival boundary invalid"
    if total_bits != replay.get("used_bits") or total_bits > REPORT_BUDGET:
        return (
            f"report budget accounting mismatch total={total_bits},"
            f"used={replay.get('used_bits')}"
        )
    for log in replay.get("slot_logs", []):
        if (
            not isinstance(log, Mapping)
            or not _integer(log.get("report_queue"))
            or not 0 <= log["report_queue"] <= REPORT_QUEUE_CAPACITY
            or not _integer(log.get("uplink_service"))
            or not 0 <= log["uplink_service"] <= UPLINK_CAPACITY[scene][log.get("t", 0) % 8]
        ):
            return f"uplink slot accounting invalid at {_safe_repr(log)}"
    return None


def _edge_knowledge_issue(replay: Mapping[str, Any]) -> str | None:
    processed = replay.get("processed_reports")
    latest = replay.get("latest")
    if not isinstance(processed, Mapping) or not isinstance(latest, Mapping):
        return "edge knowledge containers missing"
    expected_latest: dict[str, tuple[int, Any]] = {}
    useful_total = 0
    for generation, report in sorted(processed.items()):
        if not _integer(generation) or not isinstance(report, Mapping):
            return "processed report key or record malformed"
        boundary = report.get("processing_boundary")
        arrival = report.get("arrival_boundary")
        if (
            report.get("lost") is not False
            or report.get("expired") is not False
            or not _integer(arrival)
            or boundary != arrival + 1
            or boundary > HORIZON
        ):
            return f"processed report={generation} timing invalid"
        report_useful = report.get("useful_bits")
        if not _integer(report_useful) or not 0 <= report_useful <= sum(
            FIELD_BITS[field] for field in report["fields"]
        ):
            return f"processed report={generation} useful bits invalid"
        useful_total += report_useful
        for field in report["fields"]:
            if field not in expected_latest or generation > expected_latest[field][0]:
                expected_latest[field] = (generation, report["payload"][field])
    actual_latest = {
        field: (fact.get("generation_slot"), fact.get("value"))
        for field, fact in latest.items()
        if isinstance(fact, Mapping)
    }
    if actual_latest != expected_latest:
        return "latest edge field values do not match processed reports"
    if useful_total != replay.get("useful_bits"):
        return (
            f"useful bit total mismatch expected={useful_total},"
            f"actual={replay.get('useful_bits')}"
        )
    return None


def _downlink_action_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    commands = replay.get("serviced_commands")
    actions = replay.get("actions_started")
    if not isinstance(commands, list) or not isinstance(actions, list):
        return "command or action log missing"
    for command in commands:
        if not isinstance(command, Mapping):
            return "command log contains non-object"
        generated = command.get("generation_slot")
        served = command.get("service_slot")
        if (
            not _integer(generated)
            or not _integer(served)
            or not 0 <= generated <= served < HORIZON
            or (SCENES.index(scene) + served) % 5 == 0
            or command.get("lost") is not _command_lost(scene, generated, served)
        ):
            return f"command service invalid={_safe_repr(command)}"
        arrival = command.get("arrival_boundary")
        if command["lost"] or served + 1 >= HORIZON:
            if arrival is not None:
                return f"command={generated} has impossible arrival"
        elif arrival != served + 1:
            return f"command={generated} arrival invalid"
    previous_end = -1
    for item in actions:
        if not isinstance(item, Mapping):
            return "action log contains non-object"
        start = item.get("start")
        end = item.get("end")
        action_name = item.get("action")
        if (
            action_name not in ACTION_DURATION
            or not _integer(start)
            or not _integer(end)
            or not 0 <= start < HORIZON
            or end <= start
            or start < previous_end
        ):
            return f"action interval invalid={_safe_repr(item)}"
        expected_duration = ACTION_DURATION[action_name] + (
            0 if action_name == "stop" or item.get("planned_success") else 2
        )
        if end - start != expected_duration:
            return f"action duration invalid action={action_name},start={start}"
        if item.get("completed") is not (end <= HORIZON):
            return f"action completion flag invalid action={action_name},start={start}"
        if item.get("success") is not (end <= HORIZON and bool(item.get("planned_success"))):
            return f"action success flag invalid action={action_name},start={start}"
        previous_end = end
    return None


def _terminal_issue(replay: Mapping[str, Any]) -> str | None:
    used = replay.get("used_bits")
    useful = replay.get("useful_bits")
    if not _integer(used) or not 0 <= used <= REPORT_BUDGET:
        return f"invalid terminal used bits={used}"
    if not _integer(useful) or not 0 <= useful <= used:
        return f"invalid terminal useful bits={useful}"
    stage = replay.get("stage")
    progress = replay.get("progress")
    if not _integer(stage) or not 0 <= stage <= 4 or not _integer(progress) or progress < 0:
        return f"invalid terminal stage/progress={stage}/{progress}"
    active = replay.get("active_faults")
    if not isinstance(active, tuple) or any(
        fault not in replay.get("faults", {}) for fault in active
    ):
        return "invalid active fault set"
    if not isinstance(replay.get("relation_satisfied"), bool):
        return "relation terminal state is not Boolean"
    if not _number(replay.get("pose_deviation")) or replay["pose_deviation"] < 0:
        return "pose terminal state invalid"
    if not isinstance(replay.get("path_clear"), bool) or not isinstance(
        replay.get("failure_lock"), bool
    ):
        return "terminal safety or failure-lock state invalid"
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    del records
    issues: dict[str, list[str]] = {
        "decision_consumption": [],
        "fault_and_stage": [],
        "uplink_and_processing": [],
        "edge_knowledge": [],
        "downlink_and_action": [],
        "terminal_accounting": [],
    }
    for scene in SCENES:
        replay = replayed.get(scene)
        if not isinstance(replay, Mapping):
            for values in issues.values():
                values.append(f"{scene}: replay missing")
            continue
        checks = {
            "decision_consumption": _decision_consumption_issue(replay),
            "fault_and_stage": _fault_stage_issue(scene, replay),
            "uplink_and_processing": _uplink_issue(scene, replay),
            "edge_knowledge": _edge_knowledge_issue(replay),
            "downlink_and_action": _downlink_action_issue(scene, replay),
            "terminal_accounting": _terminal_issue(replay),
        }
        for name, issue in checks.items():
            if issue is not None:
                issues[name].append(f"{scene}: {issue}")
    return [
        _atom(
            "replay_validity",
            name,
            not values,
            "replay invariant passed" if not values else f"issues={values[:8]}",
        )
        for name, values in issues.items()
    ]


def _scene_metrics(replay: Mapping[str, Any]) -> dict[str, float]:
    stage_score = sum(
        value
        for value, deadline, completion in zip(
            STAGE_VALUES, STAGE_DEADLINES, replay["stage_completions"], strict=True
        )
        if completion is not None and completion <= deadline
    ) / sum(STAGE_VALUES)
    timely: list[float] = []
    for fault in replay["faults"].values():
        occurrence = fault["occurrence"]
        recovery = fault["recovery"]
        deadline = fault["deadline"]
        if occurrence is None or recovery is None or deadline is None or recovery > deadline:
            timely.append(0.0)
        else:
            timely.append(
                max(0.0, min(1.0, 1.0 - (recovery - occurrence) / (deadline - occurrence)))
            )
    recovery_score = sum(timely) / len(timely)
    task_success = float(replay["stage"] == 4)
    communication = replay["useful_bits"] / replay["used_bits"] if replay["used_bits"] else 0.0
    action_samples = []
    for action in replay["actions_started"]:
        if action["action"] == "stop":
            continue
        success = float(action["completed"] and action["success"])
        freshness = max(
            0.0,
            min(1.0, 1.0 - (action["start"] - action["evidence_slot"]) / 12.0),
        )
        action_samples.append(0.6 * success + 0.4 * freshness)
    action_quality = sum(action_samples) / len(action_samples) if action_samples else 0.0
    return {
        "primary_metrics_1": stage_score,
        "primary_metrics_2": 0.7 * recovery_score + 0.3 * task_success,
        "auxiliary_metrics_1": max(0.0, min(1.0, communication)),
        "auxiliary_metrics_2": action_quality,
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values = {
        name: sum(_scene_metrics(replayed[scene])[name] for scene in SCENES) / len(SCENES)
        for name in METRIC_NAMES
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"metric out of range: {values}")
    return values


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8.0 * metrics["primary_metrics_1"]
        + 7.0 * metrics["primary_metrics_2"]
        + 3.0 * metrics["auxiliary_metrics_1"]
        + 2.0 * metrics["auxiliary_metrics_2"]
    ) / 20.0


def _gated_metric(
    scores: Mapping[str, float],
    official_ok: bool,
    metrics: Mapping[str, float] | None,
    name: str | None,
) -> tuple[float, str]:
    failed: list[str] = []
    if not official_ok:
        failed.append("official_run")
    failed.extend(
        f"{check_id}={scores.get(check_id, 0.0):.6f}"
        for check_id in VALIDITY_CHECKS
        if scores.get(check_id, 0.0) < 1.0 - EPS
    )
    if failed:
        return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
    if name is None:
        return 1.0, "validity gate passed"
    if name not in METRIC_NAMES:
        raise ValueError(f"unknown metric name {name!r}")
    if metrics is None or name not in metrics:
        return 0.0, f"validity gate passed but {name} is unavailable; metric=0"
    value = metrics[name]
    if not _number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"invalid gated metric {name}={value!r}")
    return float(value), "validity gate passed"


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
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
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    try:
        if not _regular(ws, "report.md"):
            return 0.0, "report.md missing or not a regular file"
        report = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(report)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


def _fail_run_checks(state: EvalState, detail: str) -> EvalState:
    state.official_ok = False
    state.official_message = detail
    for name in (
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    ):
        state.scores[name] = 0.0
        state.diagnostics[name] = f"official run failed; {detail}"
    return state


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        except TimeoutError:
            state.timed_out = True
            return _fail_run_checks(state, "pre-run results cleanup timed out after 30s")
        except Exception as exc:  # noqa: BLE001
            return _fail_run_checks(
                state, f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        started = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - started
            state.timed_out = True
            return _fail_run_checks(
                state, f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - started
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; elapsed={state.elapsed_s:.3f}s",
            )
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - started
        state.timed_out = state.exit_code is None and state.elapsed_s >= 600.0
        state.result_file_is_new_regular = getattr(removed, "exit_code", None) == 0 and _regular(
            ws, "results.json"
        )
        state.official_ok = (
            state.run_started and state.exit_code == 0 and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        )
        if not state.official_ok:
            return _fail_run_checks(state, state.official_message)
        try:
            state.data = _strict_json(ws.read_text("results.json") or "")
            state.parse_ok = True
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"

        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.parse_ok:
            state.records = _index_records(state.data)
            evidence_atoms = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(
                evidence_atoms, ATOM_SHARES["result_evidence"]
            )
            state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
            coverage, defects = _coverage_score(state.data)
            state.scores["canonical_coverage"] = coverage
            state.diagnostics["canonical_coverage"] = (
                f"defect_count={len(defects)}; defects={defects[:12]}"
            )
            state.replayed = {scene: _replay_scene(scene, state.records) for scene in SCENES}
            legality_atoms = _legality_atoms(state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                legality_atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
            replay_atoms = _replay_atoms(state.records, state.replayed)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
            gate_score, _ = _gated_metric(state.scores, state.official_ok, None, None)
            state.validity_gate = gate_score >= 1.0 - EPS
            if state.validity_gate:
                state.metrics = _metrics(state.replayed)
        else:
            for name in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[name] = 0.0
                state.diagnostics[name] = "strict JSON parse failed"
    except Exception as exc:  # noqa: BLE001
        detail = f"grader failure: {type(exc).__name__}: {exc}"
        state.official_message = (
            f"{state.official_message}; {detail}" if state.official_message else detail
        )
        for name in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
    return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    task = _INFLIGHT.get(key)
    if task is None:
        task = asyncio.create_task(_build_state_uncached(ws))
        _INFLIGHT[key] = task
    try:
        state = await asyncio.shield(task)
        _CACHE[key] = state
        return state
    finally:
        if task.done() and _INFLIGHT.get(key) is task:
            del _INFLIGHT[key]


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    try:
        if not _regular(ws, "run.py"):
            return 0.0, "run.py missing or not a regular file"
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except TimeoutError:
        return 0.0, "static compilation timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"static compilation raised {type(exc).__name__}: {exc}"
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    try:
        return _report_score(ws)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report contract check raised {type(exc).__name__}: {exc}"


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in METRIC_NAMES:

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        return _gated_metric(state.scores, state.official_ok, state.metrics, _id)

    globals()[_name] = check(id=_name)(_metric)
del _metric
