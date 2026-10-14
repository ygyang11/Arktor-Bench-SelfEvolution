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

SCENES = ("pick_place", "pick_toss", "push_pull", "contact_transition", "pause_dominant")
HORIZON = 40
FIELDS = ("joint_position", "gripper", "contact_force")
FIELD_BYTES = {"joint_position": 24, "gripper": 8, "contact_force": 12}
PHASES = (
    ("reach", 0, 9),
    ("grasp", 10, 14),
    ("transport", 15, 24),
    ("contact", 25, 29),
    ("release", 30, 39),
)
THRESH = {
    "reach": (0.08, 0.10, None, None),
    "transport": (0.08, 0.10, None, None),
    "grasp": (0.05, 0.08, 0.0, 0.12),
    "contact": (0.05, 0.08, 0.0, 0.08),
    "release": (0.06, 0.08, 0.0, 0.12),
}
CRITICAL_PHASES = {"grasp", "contact", "release"}
LOSS = {
    "pick_place": {3, 11, 23},
    "pick_toss": {5, 12, 26},
    "push_pull": {2, 16, 28},
    "contact_transition": {8, 18, 27},
    "pause_dominant": {6, 7, 21, 32},
}
EPS = 1e-6
TRACE_KEYS = {
    "t",
    "program_step",
    "q",
    "v",
    "gripper",
    "contact_force",
    "phase",
    "errors",
    "gate_passed",
    "safety_intervention",
    "task_aborted",
}
TWIN_KEYS = {"t", "q", "v", "gripper", "contact_force", "ages"}
UPDATE_KEYS = {"message_id", "send_t", "fields", "bytes", "arrival_t", "status", "ack_t", "payload"}
PHASE_KEYS = {"phase", "start_index", "end_index", "first_t", "last_t", "entered"}
OUTCOME_KEYS = {
    "final_program_step",
    "completed_fraction",
    "safety_interventions",
    "task_aborted",
    "completed",
}
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scenes": 0.33,
    },
    "result_evidence": {
        f"result_evidence.{x}": 0.1
        for x in (
            "scene",
            "physical",
            "twin",
            "updates",
            "phases",
            "outcome",
            "ordering",
            "fields",
            "finite",
            "links",
        )
    },
    "decision_legality": {
        f"decision_legality.{x}": 1 / 6
        for x in ("field_subset", "bytes", "spacing", "message_ids", "ack", "information")
    },
    "replay_validity": {
        f"replay_validity.{x}": 0.1
        for x in (
            "physical",
            "twin",
            "updates",
            "gates",
            "phases",
            "outcome",
            "errors",
            "ordering",
            "ages",
            "terminal",
        )
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, bool(passed), detail, share


class EvalState:
    def __init__(self) -> None:
        self.run_started = self.run_finished = self.official_ok = False
        self.exit_code = None
        self.timed_out = False
        self.elapsed_s = None
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.data = None
        self.parse_ok = False
        self.records: dict[Hashable, Any] = {}
        self.replayed = {}
        self.scores = {}
        self.diagnostics = {}
        self.metrics = None
        self.validity_gate = False


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _all_finite(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, Mapping):
        return all(_all_finite(x) for x in v.values())
    if isinstance(v, list):
        return all(_all_finite(x) for x in v)
    return not isinstance(v, (int, float)) or _finite(v)


def _strict_json(raw: str) -> Any:
    def reject(x: str) -> None:
        raise ValueError(x)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _shape(v: Any, keys: set[str]) -> bool:
    return isinstance(v, Mapping) and set(v) == keys


def _atom(group: str, name: str, ok: bool, detail: str) -> AtomVerdict:
    key = f"{group}.{name}"
    return AtomVerdict(key, ok, detail, ATOM_SHARES[group][key])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    if len({a.id for a in atoms}) != len(atoms) or {a.id for a in atoms} != set(expected):
        raise ValueError("atom IDs")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("atom shares")
    if any(not math.isclose(a.share, expected[a.id], abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share mismatch")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    bad = [a for a in atoms if not a.passed]
    return (
        "failed_atoms=[]"
        if not bad
        else "failed_atoms=["
        + ",".join(a.id for a in bad)
        + "]; "
        + " | ".join(f"{a.id}: {a.detail}" for a in bad)
    )


def _phase(k: int) -> str:
    return next(name for name, start, end in PHASES if start <= k <= end)


def _nominal(scene: str, k: int) -> dict[str, Any]:
    off = SCENES.index(scene) * 0.01
    q = [-0.65 + 0.03 * k + off, -0.35 + 0.02 * k - off]
    if scene == "push_pull":
        q[0] += 0.025 * math.sin(k / 3)
    if scene == "pick_toss":
        q[1] += 0.02 * math.sin(k / 2)
    q = [max(-1.0, min(1.0, x)) for x in q]
    prev = _nominal(scene, k - 1) if k else None
    v = [0.0, 0.0] if prev is None else [q[i] - prev["q"][i] for i in range(2)]
    return {
        "q": q,
        "v": v,
        "gripper": int(10 <= k < 30),
        "contact_force": 0.7 if 25 <= k < 30 else (0.45 if 10 <= k < 15 else 0.08),
    }


def _initial_twin(scene: str) -> dict[str, Any]:
    """Return the publicly specified no-update twin state."""
    nominal = _nominal(scene, 0)
    return {
        "q": list(nominal["q"]),
        "v": list(nominal["v"]),
        "gripper": nominal["gripper"],
        "contact_force": nominal["contact_force"],
        "ages": {field: 6 for field in FIELDS},
    }


def _apply_update(twin: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Apply one delivered update in message order.

    Velocity is not a transmitted field.  A successful joint-position update
    reconstructs it as the clipped difference from the previous twin position;
    other field updates leave the previous twin velocity unchanged.
    """
    for field in update.get("fields", []):
        payload = update.get("payload", {})
        if field == "joint_position":
            previous = list(twin["q"])
            current = list(payload[field])
            twin["q"] = current
            twin["v"] = [
                max(-0.25, min(0.25, current[i] - previous[i])) for i in range(2)
            ]
        elif field in {"gripper", "contact_force"}:
            twin[field] = payload[field]
        twin["ages"][field] = 0


def _authorization_phase(step: int) -> str:
    """Use the target phase only on a real phase-crossing transition."""
    current = _phase(step)
    if step < HORIZON - 1:
        target = _phase(step + 1)
        if target != current:
            return target
    return current


def _expected_phases(physical_trace: Any) -> list[dict[str, Any]]:
    """Derive the five canonical phase records from observed program steps."""
    rows = physical_trace if isinstance(physical_trace, list) else []
    out = []
    for name, start, end in PHASES:
        entered_slots = [
            row.get("t")
            for row in rows
            if isinstance(row, Mapping)
            and _integer(row.get("program_step"))
            and start <= row["program_step"] <= end
        ]
        out.append(
            {
                "phase": name,
                "start_index": start,
                "end_index": end,
                "first_t": min(entered_slots) if entered_slots else None,
                "last_t": max(entered_slots) if entered_slots else None,
                "entered": bool(entered_slots),
            }
        )
    return out


def _errors(physical: Mapping[str, Any], twin: Mapping[str, Any]) -> dict[str, float]:
    return {
        "e_q": sum(min(1.0, abs(a - b) / 2) for a, b in zip(physical["q"], twin["q"])) / 2,
        "e_v": sum(min(1.0, abs(a - b) / 0.5) for a, b in zip(physical["v"], twin["v"])) / 2,
        "e_g": float(physical["gripper"] != twin["gripper"]),
        "e_f": min(1.0, abs(physical["contact_force"] - twin["contact_force"])),
    }


def _gate(phase: str, errors: Mapping[str, float]) -> bool:
    return all(
        limit is None or value <= limit + EPS
        for value, limit in zip(
            (errors["e_q"], errors["e_v"], errors["e_g"], errors["e_f"]), THRESH[phase]
        )
    )


def _compare(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            errors.append(f"{path}: keys")
            return
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}", errors)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            errors.append(f"{path}: length")
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                _compare(a, e, f"{path}[{i}]", errors)
    elif _finite(actual) and _finite(expected):
        if not math.isclose(float(actual), float(expected), abs_tol=EPS, rel_tol=0):
            errors.append(f"{path}: numeric")
    elif actual != expected:
        errors.append(f"{path}: expected {expected!r}, actual {actual!r}")


def _rows(data: Any) -> list[Mapping[str, Any]]:
    return (
        data.get("scenarios", [])
        if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list)
        else []
    )


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = _shape(data, {"scenarios"})
    rows = data.get("scenarios") if root else None
    containers = isinstance(rows, list) and all(
        _shape(x, {"scenario", "physical_trace", "twin_trace", "updates", "phases", "task_outcome"})
        for x in rows
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON"),
        _atom("result_json", "root", root, "root keys"),
        _atom("result_json", "scenes", containers, "scene containers"),
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    flags = {
        x: True
        for x in (
            "scene",
            "physical",
            "twin",
            "updates",
            "phases",
            "outcome",
            "ordering",
            "fields",
            "finite",
            "links",
        )
    }
    rows = _rows(data)
    for row in rows:
        if not isinstance(row, Mapping):
            flags["scene"] = False
            continue
        phys, twin, updates, phases, outcome = (
            row.get(x)
            for x in ("physical_trace", "twin_trace", "updates", "phases", "task_outcome")
        )
        flags["physical"] &= isinstance(phys, list) and all(_shape(x, TRACE_KEYS) for x in phys)
        flags["twin"] &= isinstance(twin, list) and all(_shape(x, TWIN_KEYS) for x in twin)
        flags["updates"] &= isinstance(updates, list) and all(
            _shape(x, UPDATE_KEYS) for x in updates
        )
        flags["phases"] &= isinstance(phases, list) and len(phases) == len(PHASES) and all(
            _shape(x, PHASE_KEYS) for x in phases
        )
        flags["outcome"] &= _shape(outcome, OUTCOME_KEYS)
        flags["ordering"] &= isinstance(phys, list) and [
            x.get("t") for x in phys if isinstance(x, Mapping)
        ] == list(range(HORIZON))
        flags["ordering"] &= isinstance(twin, list) and [
            x.get("t") for x in twin if isinstance(x, Mapping)
        ] == list(range(HORIZON))
        flags["ordering"] &= isinstance(updates, list) and [
            x.get("message_id") for x in updates if isinstance(x, Mapping)
        ] == [f"m{i}" for i in range(len(updates))]
        flags["finite"] &= all(
            _all_finite(x)
            for seq in (phys, twin, updates, phases)
            if isinstance(seq, list)
            for x in seq
        )
        flags["finite"] &= _all_finite(outcome)
        for update in updates or []:
            if isinstance(update, Mapping):
                fields = update.get("fields")
                flags["fields"] &= (
                    isinstance(fields, list)
                    and all(isinstance(field, str) for field in fields)
                    and fields == sorted(set(fields))
                    and 1 <= len(fields) <= 3
                )
                flags["links"] &= (
                    isinstance(update.get("payload"), Mapping)
                    and isinstance(fields, list)
                    and all(isinstance(field, str) for field in fields)
                    and set(update["payload"]) == set(fields)
                )
                flags["links"] &= update.get("status") in {"delivered", "dropped", "late"}
    return [_atom("result_evidence", x, flags[x], "evidence rule") for x in flags]


def _index(data: Any) -> dict[Hashable, Any]:
    return {
        (row["scenario"], key): row.get(key)
        for row in _rows(data)
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str)
        for key in ("physical_trace", "twin_trace", "updates", "phases", "task_outcome")
    }


def _coverage(data: Any) -> tuple[float, list[str]]:
    names = [x.get("scenario") for x in _rows(data) if isinstance(x, Mapping)]
    defects = [] if names == list(SCENES) else ["scene order"]
    for scene in SCENES:
        if names.count(scene) != 1:
            defects.append(f"{scene} count")
        row = next(
            (x for x in _rows(data) if isinstance(x, Mapping) and x.get("scenario") == scene), None
        )
        for field in ("physical_trace", "twin_trace"):
            vals = [x.get("t") for x in row.get(field, []) if isinstance(x, Mapping)] if row else []
            if set(vals) != set(range(HORIZON)) or len(vals) != HORIZON:
                defects.append(f"{scene}.{field}")
    return max(0.0, (4 - 0.5 * len(defects)) / 4), defects


def _legality(records: Mapping[Hashable, Any]) -> list[AtomVerdict]:
    names = ("field_subset", "bytes", "spacing", "message_ids", "ack", "information")
    flags = {x: True for x in names}
    details = {x: "legal" for x in names}
    for scene in SCENES:
        updates = records.get((scene, "updates"), [])
        sends = []
        previous_send = None
        burst = 0
        physical_rows = records.get((scene, "physical_trace"), [])
        physical_by_t = {
            row.get("t"): row for row in physical_rows if isinstance(row, Mapping)
        } if isinstance(physical_rows, list) else {}
        for i, update in enumerate(updates if isinstance(updates, list) else []):
            fields = update.get("fields", []) if isinstance(update, Mapping) else []
            send_t = update.get("send_t") if isinstance(update, Mapping) else None
            sends.append(send_t)
            if (
                not isinstance(fields, list)
                or not fields
                or any(not isinstance(x, str) for x in fields)
                or fields != sorted(set(fields))
                or any(x not in FIELDS for x in fields)
            ):
                flags["field_subset"] = False
            if not isinstance(update, Mapping) or update.get("bytes") != sum(
                FIELD_BYTES.get(x, 0) for x in fields
            ):
                flags["bytes"] = False
            if not isinstance(update, Mapping) or update.get("message_id") != f"m{i}":
                flags["message_ids"] = False
            if isinstance(update, Mapping) and update.get("ack_t") is not None and not _integer(
                update.get("ack_t")
            ):
                flags["ack"] = False
            if (
                not isinstance(update, Mapping)
                or update.get("arrival_t") != update.get("send_t", -1) + 1
            ):
                flags["information"] = False
            if isinstance(update, Mapping):
                status = update.get("status")
                arrival = update.get("arrival_t")
                if not _integer(send_t) or not 0 <= send_t < HORIZON:
                    flags["information"] = False
                if _integer(send_t) and previous_send is not None and send_t == previous_send + 1:
                    burst += 1
                else:
                    burst = 1
                if burst > 2:
                    flags["spacing"] = False
                previous_send = send_t if _integer(send_t) else None
                expected_status = (
                    "dropped" if _integer(send_t) and send_t in LOSS[scene]
                    else ("late" if _integer(arrival) and arrival > 39 else "delivered")
                )
                if not _integer(arrival) or status != expected_status:
                    flags["information"] = False
                if status == "delivered" and update.get("ack_t") != arrival:
                    flags["ack"] = False
                if status in {"dropped", "late"} and update.get("ack_t") is not None:
                    flags["ack"] = False
                payload = update.get("payload")
                source = physical_by_t.get(send_t)
                if not isinstance(payload, Mapping) or not isinstance(source, Mapping):
                    flags["information"] = False
                else:
                    for field in fields:
                        value = payload.get(field)
                        expected = source.get("q") if field == "joint_position" else source.get(
                            "gripper" if field == "gripper" else "contact_force"
                        )
                        if field == "joint_position":
                            valid = (
                                isinstance(value, list)
                                and len(value) == 2
                                and all(_finite(x) and -1 <= float(x) <= 1 for x in value)
                                and isinstance(expected, list)
                                and all(
                                    math.isclose(float(a), float(b), abs_tol=EPS, rel_tol=0)
                                    for a, b in zip(value, expected)
                                )
                            )
                        elif field == "gripper":
                            valid = _integer(value) and value in {0, 1} and value == expected
                        else:
                            valid = (
                                _finite(value)
                                and _finite(expected)
                                and 0 <= float(value) <= 1
                                and math.isclose(
                                    float(value), float(expected), abs_tol=EPS, rel_tol=0
                                )
                            )
                        if not valid:
                            flags["information"] = False
        if any(not _integer(x) for x in sends) or any(
            _integer(a) and _integer(b) and b - a < 1 for a, b in zip(sends, sends[1:])
        ):
            flags["spacing"] = False
        if any(_integer(x) and not 0 <= x < HORIZON for x in sends):
            flags["information"] = False
        if sum(x.get("bytes", 0) for x in updates if isinstance(x, Mapping)) > 1760:
            flags["bytes"] = False
    return [_atom("decision_legality", x, flags[x], details[x]) for x in names]


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    phys = records.get((scene, "physical_trace"), [])
    twin_rows = records.get((scene, "twin_trace"), [])
    updates = records.get((scene, "updates"), [])
    outcome = records.get((scene, "task_outcome"), {})
    p_rows = (
        {x.get("t"): x for x in phys if isinstance(x, Mapping)} if isinstance(phys, list) else {}
    )
    t_rows = (
        {x.get("t"): x for x in twin_rows if isinstance(x, Mapping)}
        if isinstance(twin_rows, list)
        else {}
    )
    ups = [x for x in updates if isinstance(x, Mapping)] if isinstance(updates, list) else []
    arrivals: dict[int, list[Mapping[str, Any]]] = {}
    for update in ups:
        if update.get("status") == "delivered" and _integer(update.get("arrival_t")):
            arrivals.setdefault(update.get("arrival_t"), []).append(update)
    twin = _initial_twin(scene)
    step = 0
    interventions = 0
    aborted = False
    fails = 0
    fail_target = None
    errors: list[str] = []
    compliance_hits = compliance_total = 0
    age_total = quality_total = 0.0
    for t in range(HORIZON):
        for field in FIELDS:
            twin["ages"][field] = min(6, twin["ages"][field] + 1)
        for update in sorted(arrivals.get(t, []), key=lambda x: x.get("message_id", "")):
            fields = update.get("fields")
            payload = update.get("payload")
            safe_payload = isinstance(fields, list) and isinstance(payload, Mapping) and all(
                isinstance(field, str) and field in payload for field in fields
            )
            if safe_payload:
                for field in fields:
                    value = payload[field]
                    if field == "joint_position":
                        safe_payload = (
                            isinstance(value, list)
                            and len(value) == 2
                            and all(_finite(item) and -1 <= float(item) <= 1 for item in value)
                        )
                    elif field == "gripper":
                        safe_payload = _integer(value) and value in {0, 1}
                    elif field == "contact_force":
                        safe_payload = _finite(value) and 0 <= float(value) <= 1
                    if not safe_payload:
                        break
            if safe_payload:
                _apply_update(twin, update)
        physical = _nominal(scene, step)
        phase = _phase(step)
        auth_phase = _authorization_phase(step)
        err = _errors(physical, twin)
        active = not aborted and step < 39
        gate = _gate(auth_phase, err) if active else False
        quality_total += max(
            0.0,
            min(
                1.0,
                1.0
                - (0.35 * err["e_q"] + 0.25 * err["e_v"] + 0.25 * err["e_g"] + 0.15 * err["e_f"]),
            ),
        )
        if active and phase in CRITICAL_PHASES:
            compliance_total += 1
            compliance_hits += int(gate)
        old_step = step
        intervention = 0
        transition_target = (
            _phase(step + 1) if step < 39 and _phase(step + 1) != phase else None
        )
        if not active:
            gate = False
        elif gate and step < 39:
            step += 1
            fails = 0
            fail_target = None
        else:
            intervention = 1
            interventions += intervention
            if transition_target in CRITICAL_PHASES:
                if fail_target != transition_target:
                    fail_target = transition_target
                    fails = 0
                fails += 1
                if fails >= 3:
                    aborted = True
            else:
                fails = 0
                fail_target = None
        row, twin_row = p_rows.get(t), t_rows.get(t)
        if not isinstance(row, Mapping) or not isinstance(twin_row, Mapping):
            errors.append(f"{scene}.t={t}: missing trace")
            continue
        for key, expected in (
            ("t", t),
            ("program_step", old_step),
            ("phase", phase),
            ("errors", err),
            ("gate_passed", gate),
            ("safety_intervention", intervention),
            ("task_aborted", aborted),
        ):
            _compare(row.get(key), expected, f"{scene}.physical[{t}].{key}", errors)
        for key in ("q", "v", "gripper", "contact_force"):
            _compare(row.get(key), physical[key], f"{scene}.physical[{t}].{key}", errors)
        for key in ("t", "q", "v", "gripper", "contact_force", "ages"):
            _compare(
                twin_row.get(key),
                t if key == "t" else twin[key],
                f"{scene}.twin[{t}].{key}",
                errors,
            )
        age_total += sum(twin["ages"].values()) / 3
    expected_outcome = {
        "final_program_step": step,
        "completed_fraction": step / 39,
        "safety_interventions": interventions,
        "task_aborted": aborted,
        "completed": step >= 39 and not aborted,
    }
    _compare(outcome, expected_outcome, f"{scene}.task_outcome", errors)
    _compare(
        records.get((scene, "phases"), []),
        _expected_phases(phys),
        f"{scene}.phases",
        errors,
    )
    compliance = compliance_hits / compliance_total if compliance_total else 0.0
    physical_execution = (
        0.0 if aborted else 0.7 * step / 39 + 0.3 * max(0.0, min(1.0, 1.0 - interventions / 40))
    )
    return {
        "errors": errors,
        "compliance": compliance,
        "physical_execution": physical_execution,
        "bytes": sum(x.get("bytes", 0) for x in ups),
        "freshness": max(0.0, min(1.0, 1.0 - age_total / HORIZON / 6)),
        "quality": quality_total / HORIZON,
    }


def _replay_atoms(records: Mapping[Hashable, Any]):
    replayed = {scene: _replay_scene(scene, records) for scene in SCENES}
    names = (
        "physical",
        "twin",
        "updates",
        "gates",
        "phases",
        "outcome",
        "errors",
        "ordering",
        "ages",
        "terminal",
    )
    flags = {x: True for x in names}
    details = {x: "replay matched" for x in names}
    for scene, result in replayed.items():
        for error in result["errors"]:
            key = (
                "twin"
                if ".twin" in error
                else (
                    "outcome"
                    if "task_outcome" in error
                    else (
                        "phases"
                        if ".phases" in error
                        else ("physical" if ".physical" in error else "errors")
                    )
                )
            )
            flags[key] = False
            details[key] = error
            if "ordering" in error:
                flags["ordering"] = False
                details["ordering"] = error
        if result["errors"]:
            flags["terminal"] = False
    return [_atom("replay_validity", x, flags[x], details[x]) for x in names], replayed


def _metrics(replayed: Mapping[str, Any]):
    return {
        "primary_metrics_1": sum(
            0.5 * (x["compliance"] + x["physical_execution"]) for x in replayed.values()
        )
        / len(SCENES),
        "primary_metrics_2": sum(x["quality"] for x in replayed.values()) / len(SCENES),
        "auxiliary_metrics_1": sum(
            max(0.0, min(1.0, 1.0 - x["bytes"] / 1760)) for x in replayed.values()
        )
        / len(SCENES),
        "auxiliary_metrics_2": sum(x["freshness"] for x in replayed.values()) / len(SCENES),
    }


def _regular(ws: Workspace, rel: str) -> bool:
    p = Path(ws.root) / rel
    return p.is_file() and not p.is_symlink()


def _headings(text: str) -> dict[str, str]:
    lines = text.splitlines()
    parsed = []
    heading_lines = set()
    fence_lines = set()
    fence_char = None
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
            title = re.sub(r"[ \t]+#+[ \t]*$", "", (atx.group(2) or "").strip()).strip()
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
    found = {}
    for index, (start, end, level, title) in enumerate(parsed):
        stop = len(lines)
        for next_start, _, next_level, _ in parsed[index + 1 :]:
            if next_level <= level:
                stop = next_start
                break
        body = "\n".join(
            lines[j] for j in range(end + 1, stop)
            if j not in heading_lines and j not in fence_lines
        ).strip()
        if title not in found or not found[title]:
            found[title] = body
    return found


def _report_score(ws: Workspace):
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    if not _regular(ws, "report.md"):
        return 0.0, "report.md missing"
    missing = [x for x in required if not _headings(ws.read_text("report.md") or "").get(x)]
    return max(0.0, (2.0 - 0.5 * len(missing)) / 2.0), f"missing_or_empty={sorted(missing)}"


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace):
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    s = EvalState()
    start = time.monotonic()
    s.run_started = True
    try:
        rm = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        absent = await ws.run(
            "test ! -e results.json && test ! -L results.json", check=False, timeout=30.0
        )
        run = await ws.run("python run.py", check=False, timeout=600.0)
        s.run_finished = True
        s.exit_code = getattr(run, "exit_code", None)
        s.elapsed_s = time.monotonic() - start
        s.result_file_is_new_regular = _regular(ws, "results.json")
        remove_ok = (
            getattr(rm, "exit_code", None) == 0
            and getattr(absent, "exit_code", None) == 0
        )
        s.official_ok = (
            remove_ok
            and s.exit_code == 0
            and s.elapsed_s <= 600 + EPS
            and s.result_file_is_new_regular
        )
        s.official_message = (
            f"exit={s.exit_code}, elapsed={s.elapsed_s:.3f}s, "
            f"new_results={s.result_file_is_new_regular}"
        )
        if s.result_file_is_new_regular:
            try:
                s.data = _strict_json(ws.read_text("results.json") or "")
                s.parse_ok = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                s.diagnostics["result_json"] = f"parse failed: {exc}"
        ja = _result_json_atoms(s.data, s.parse_ok)
        s.scores["result_json"] = (
            _score_atoms(ja, ATOM_SHARES["result_json"]) if s.official_ok else 0.0
        )
        s.diagnostics["result_json"] = s.diagnostics.get("result_json", _diagnose(ja))
        if s.official_ok and s.parse_ok:
            s.records = _index(s.data)
            ea = _evidence_atoms(s.data)
            s.scores["result_evidence"] = _score_atoms(ea, ATOM_SHARES["result_evidence"])
            s.diagnostics["result_evidence"] = _diagnose(ea)
            cov, defects = _coverage(s.data)
            s.scores["canonical_coverage"] = cov
            s.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:8]}"
            la = _legality(s.records)
            s.scores["decision_legality"] = _score_atoms(la, ATOM_SHARES["decision_legality"])
            s.diagnostics["decision_legality"] = _diagnose(la)
            ra, s.replayed = _replay_atoms(s.records)
            s.scores["replay_validity"] = _score_atoms(ra, ATOM_SHARES["replay_validity"])
            s.diagnostics["replay_validity"] = _diagnose(ra)
            req = (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            )
            s.metrics = (
                _metrics(s.replayed) if all(s.scores.get(x, 0.0) >= 1 - EPS for x in req) else None
            )
            s.validity_gate = s.metrics is not None
        else:
            for x in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                s.scores[x] = 0.0
                s.diagnostics[x] = "official run or parse failed"
    except Exception as exc:
        s.official_message = f"grader failure: {type(exc).__name__}: {exc}"
    _CACHE[key] = s
    return s


@check(id="entry_point")
async def entry_point(ws: Workspace):
    if not _regular(ws, "run.py"):
        return 0.0, "run.py missing"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace):
    s = await _build_state(ws)
    return (1.0 if s.official_ok else 0.0), s.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace):
    return _report_score(ws)


async def _score(ws: Workspace, name: str):
    s = await _build_state(ws)
    return s.scores.get(name, 0.0), s.diagnostics.get(name, "")


for _name in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, _id: str = _name):
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, _id: str = _name):
        s = await _build_state(ws)
        return (s.metrics or {}).get(
            _id, 0.0
        ), "validity gate passed" if s.metrics else "validity gate failed; metric=0"

    globals()[_name] = check(id=_name)(_metric)
del _metric
