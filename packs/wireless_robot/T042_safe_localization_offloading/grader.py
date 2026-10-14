# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
HORIZON = 120
SLOT_DT = 0.1
SPEEDS = tuple(round(i * 0.1, 1) for i in range(11))
QUEUE_CAPACITY = 6
MAX_SENDS = 36
PACKET_BITS = 300
MAX_ENERGY = 80.0
RISK_CLEARANCE = 0.45
SCENARIO_NAMES = ("open_turn", "narrow_corridor", "moving_bottleneck", "handover_gap", "energy_limited")
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
METRIC_KEYS = ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")
GATE_CHECKS = ("official_run", "result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
ATOM_SHARES = {
    "result_json": {"result_json.parse": 0.20, "result_json.root": 0.30, "result_json.unit_shape": 0.50},
    "result_evidence": {"result_evidence.unit_fields": 0.35, "result_evidence.slot_fields": 0.65},
    "decision_legality": {"decision_legality.branch": 0.22, "decision_legality.speed": 0.24, "decision_legality.send": 0.22, "decision_legality.energy": 0.16, "decision_legality.queue": 0.16},
    "replay_validity": {"replay_validity.route": 0.20, "replay_validity.network": 0.22, "replay_validity.uncertainty": 0.20, "replay_validity.safety": 0.18, "replay_validity.metrics": 0.20},
}
UNIT_FIELDS = {"scenario", "variant", "slots"}
SLOT_FIELDS = {"t", "branch", "speed", "send"}
BRANCH_VALUES = frozenset(("undecided", "a", "b"))

SCENARIOS = {
    "open_turn": {"a": (7.8, 1.40, (3.6, 4.8)), "b": (6.9, 1.00, (3.2, 4.3)), "common_clearance": 1.40, "deadline": 108, "phase": 2, "affected": None, "loss": 1},
    "narrow_corridor": {"a": (8.2, 1.20, None), "b": (7.0, 0.62, None), "common_clearance": 1.20, "deadline": 112, "phase": 5, "affected": None, "loss": 2},
    "moving_bottleneck": {"a": (8.3, 1.10, (4.0, 5.0)), "b": (7.2, 0.76, (3.5, 4.5)), "common_clearance": 1.10, "deadline": 112, "phase": 8, "affected": "b", "affected_interval": (3.8, 5.0), "loss": 3},
    "handover_gap": {"a": (7.7, 1.00, (4.2, 5.1)), "b": (7.1, 0.78, (3.8, 4.6)), "common_clearance": 1.00, "deadline": 108, "phase": 11, "affected": None, "loss": 4},
    "energy_limited": {"a": (6.8, 0.86, None), "b": (8.1, 1.30, (4.4, 5.6)), "common_clearance": 0.86, "deadline": 110, "phase": 14, "affected": "a", "affected_interval": (4.4, 5.6), "loss": 2},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")
    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, bool(passed), str(detail), float(share)


class EvalState:
    def __init__(self) -> None:
        self.run_started = False; self.run_finished = False; self.exit_code = None; self.timed_out = False
        self.elapsed_s = None; self.official_ok = False; self.official_message = ""; self.result_file_is_new_regular = False
        self.parse_ok = False; self.data = None; self.records = {}; self.replayed = {}; self.scores = {}; self.diagnostics = {}; self.metrics = {}; self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _brief(value: Any, limit: int = 140) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # huge/malformed values must remain diagnosable
        return f"<{type(value).__name__} repr failed: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _branch_value(value: Any) -> bool:
    """Return whether a candidate branch field is one of the public strings."""
    return isinstance(value, str) and value in BRANCH_VALUES


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _strict_json(raw: str) -> Any:
    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number {value}")
        return number

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out
    return json.loads(
        raw,
        parse_constant=reject,
        parse_float=finite_float,
        object_pairs_hook=pairs,
    )


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected): raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(float(v)) or v <= 0 for v in expected.values()): raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0, abs_tol=1e-12) for a in atoms): raise ValueError("atom share mismatch")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0, abs_tol=1e-9): raise ValueError("atom shares do not sum to one")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed: return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(a.id for a in failed) + "]; " + " | ".join(f"{a.id}: {a.detail}" for a in failed)


def _shape_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, dict) and set(data) == {"scenarios"}
    unit = root and isinstance(data.get("scenarios"), list) and all(isinstance(row, dict) and set(row) == UNIT_FIELDS for row in data["scenarios"])
    return [_atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "strict JSON parse failed"), _atom("result_json", "root", root, f"root_keys={_brief(list(data) if isinstance(data, dict) else type(data).__name__)}"), _atom("result_json", "unit_shape", unit, "exact unit fields")]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, dict) else None
    unit_ok = isinstance(rows, list) and all(isinstance(r, dict) and isinstance(r.get("scenario"), str) and _integer(r.get("variant")) and isinstance(r.get("slots"), list) for r in rows)
    slot_ok = True; issue = "slot fields valid"
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or not isinstance(r.get("slots"), list): slot_ok = False; issue = "slots is not an array"; break
            for item in r["slots"]:
                if not isinstance(item, dict) or set(item) != SLOT_FIELDS: slot_ok = False; issue = "slot fields are not exact"; break
                if not _integer(item.get("t")) or not _branch_value(item.get("branch")) or not _finite(item.get("speed")) or not isinstance(item.get("send"), bool): slot_ok = False; issue = f"invalid slot={_brief(item.get('t'))} fields"; break
            if not slot_ok: break
    else: slot_ok = False; issue = "scenario array missing"
    return [_atom("result_evidence", "unit_fields", unit_ok, "unit fields valid" if unit_ok else "unit fields invalid"), _atom("result_evidence", "slot_fields", slot_ok, issue)]


def _index_records(data: Any) -> dict[tuple[str, int], Any]:
    out: dict[tuple[str, int], Any] = {}
    rows = data.get("scenarios") if isinstance(data, dict) else []
    if not isinstance(rows, list): return out
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("scenario"), str) and _integer(row.get("variant")):
            key = (row["scenario"], int(row["variant"]))
            if key not in out: out[key] = row
    return out


def _index(data: Any) -> dict[tuple[str, int], Any]:
    """Backward-compatible task-local alias for the required index entry point."""
    return _index_records(data)


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []; seen: list[Any] = []
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list): return 0.0, ["scenarios is not an array"]
    for row in rows:
        key = (row.get("scenario"), row.get("variant")) if isinstance(row, dict) else ("<invalid>", "<invalid>")
        seen.append(key)
    for key in SCENARIO_KEYS:
        if seen.count(key) == 0: defects.append(f"missing unit {_brief(key)}")
        if seen.count(key) > 1: defects.append(f"duplicate unit {_brief(key)}")
    for key in seen:
        if key not in SCENARIO_KEYS: defects.append(f"unknown unit {_brief(key)}")
    for key in SCENARIO_KEYS:
        rows_for = [r for r in rows if isinstance(r, dict) and (r.get("scenario"), r.get("variant")) == key]
        if len(rows_for) == 1 and isinstance(rows_for[0].get("slots"), list):
            ts = [x.get("t") for x in rows_for[0]["slots"] if isinstance(x, dict)]
            expected = list(range(HORIZON))
            for t in expected:
                if ts.count(t) == 0: defects.append(f"missing slot {_brief(key)}|{t}")
                if ts.count(t) > 1: defects.append(f"duplicate slot {_brief(key)}|{t}")
            for t in ts:
                if t not in expected: defects.append(f"unknown slot {_brief(key)}|{_brief(t)}")
    return max(0.0, min(1.0, (4.0 - 0.5 * len(defects)) / 4.0)), defects


def _branch_heading(info: Mapping[str, Any], branch: str, progress: float) -> float:
    turn = info[branch][2]
    if turn is None: return 0.0
    lo, hi = turn
    return 0.7 * _clip((progress - lo) / max(hi - lo, EPS))


def _clearance(info: Mapping[str, Any], branch: str, variant: int, slot: int, progress: float) -> float:
    nominal = float(info["common_clearance"] if branch == "undecided" else info[branch][1])
    if branch == "undecided":
        return nominal
    affected = info.get("affected") == branch and ((slot + int(info["phase"]) + 3 * variant) % 24) in {10, 11, 12, 13, 14, 15}
    interval = info.get("affected_interval")
    if interval is None:
        interval = (0.0, 0.0)
    return nominal - 0.35 if affected and interval[0] <= progress <= interval[1] else nominal


def _phase(name: str, variant: int) -> int:
    return int(SCENARIOS[name]["phase"] + 3 * variant)


def _loss(name: str, variant: int, generated: int, service: int) -> bool:
    info = SCENARIOS[name]
    return (17 * generated + 5 * service + 11 * variant + int(info["phase"])) % 19 < int(info["loss"])


def _valid_slot(item: Any) -> bool:
    return isinstance(item, dict) and set(item) == SLOT_FIELDS and _integer(item.get("t")) and _branch_value(item.get("branch")) and _finite(item.get("speed")) and isinstance(item.get("send"), bool)


def _replay_without_correction(name: str, variant: int, baseline: Mapping[str, Any], omit_generated: int) -> tuple[int, float]:
    """Recompute safety with the baseline event schedule, omitting one correction.

    The leave-one-out definition keeps the submitted action sequence and all
    other packet outcomes fixed.  Re-running FIFO admission after deleting a
    packet would shift service slots and losses for later packets, so this
    counterfactual reuses the baseline generated/service/response records and
    removes only the selected packet's uncertainty update.
    """
    info = SCENARIOS[name]
    uncertainty = 0.10
    progress = 0.0
    branch = "undecided"
    risk = 0
    excess = 0.0
    by_arrival: dict[int, list[dict[str, Any]]] = {}
    for event in baseline.get("measurements", []):
        arrival = event.get("arrival")
        if not _integer(arrival) or arrival > HORIZON or event.get("lost"):
            continue
        by_arrival.setdefault(int(arrival), []).append(event)
    for events in by_arrival.values():
        events.sort(key=lambda x: x["id"])
    raw_slots = {int(x["t"]): x for x in baseline.get("slots", []) if isinstance(x, dict) and _integer(x.get("t"))}
    for t in range(HORIZON):
        for event in by_arrival.get(t, []):
            if int(event["generated"]) == int(omit_generated):
                continue
            corr = _clip(0.35 * event["u_generated"] + 0.025 * event["age"])
            uncertainty = min(uncertainty, corr)
        slot = raw_slots.get(t, {})
        speed = float(slot.get("speed", 0.0)) if _finite(slot.get("speed")) else 0.0
        submitted_branch = slot.get("branch", "undecided")
        if branch == "undecided" and isinstance(submitted_branch, str) and submitted_branch in {"a", "b"} and progress >= 2.0 - EPS:
            branch = submitted_branch
        active = branch if branch in {"a", "b"} else "a"
        default_progress = min(2.0, progress + 0.1 * speed) if branch == "undecided" else progress + 0.1 * speed
        next_progress = float(slot.get("progress", default_progress))
        heading_before = _branch_heading(info, active, progress)
        heading_after = _branch_heading(info, active, next_progress)
        omega = (heading_after - heading_before) / SLOT_DT
        uncertainty = _clip(uncertainty + 0.015 * speed + 0.01 * abs(omega))
        clearance = float(slot.get("clearance", _clearance(info, branch, variant, t, next_progress)))
        if clearance < RISK_CLEARANCE - EPS:
            risk += 1
        required = _clip((clearance - RISK_CLEARANCE) / 1.0)
        excess += max(0.0, uncertainty - required)
        progress = next_progress
    # A response at the terminal boundary is still part of the fixed replay
    # evidence.  Apply it after the last action slot; no slot metric is added
    # because the public leave-one-out predicate only sums t=0..HORIZON-1.
    for event in by_arrival.get(HORIZON, []):
        if int(event["generated"]) == int(omit_generated):
            continue
        corr = _clip(0.35 * event["u_generated"] + 0.025 * event["age"])
        uncertainty = min(uncertainty, corr)
    return risk, excess


def _legality(records: Mapping[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures = {k: [] for k in ("branch", "speed", "send", "energy", "queue")}

    def slot_order(item: Mapping[str, Any]) -> int:
        value = item.get("t")
        return value if _integer(value) else -1

    for key, row in records.items():
        name, variant = key; info = SCENARIOS.get(name)
        if info is None or variant not in range(3) or not isinstance(row, dict) or not isinstance(row.get("slots"), list): continue
        previous_speed = 0.0; energy = 62.0 if name == "energy_limited" else 80.0; branch = "undecided"; progress = 0.0; sends = 0; q = 0; arrived = False
        for slot in sorted((x for x in row["slots"] if isinstance(x, dict)), key=slot_order):
            t = slot.get("t"); br = slot.get("branch"); speed = float(slot.get("speed", 0.0)) if _finite(slot.get("speed")) else float("nan"); send = bool(slot.get("send"))
            if not _integer(t) or not _valid_slot(slot): continue
            if branch == "undecided":
                if br != "undecided" and not (isinstance(br, str) and br in {"a", "b"}): failures["branch"].append(f"{key}|slot={_brief(t)}: invalid branch")
                if isinstance(br, str) and br in {"a", "b"}:
                    if progress + EPS < 2.0: failures["branch"].append(f"{key}|slot={_brief(t)}: branch selected before progress=2.0 (progress={progress:.9g})")
                    else: branch = br
                elif progress >= 2.0 - EPS:
                    failures["branch"].append(f"{key}|slot={_brief(t)}: branch must be chosen at progress={progress:.9g}")
            elif br != branch: failures["branch"].append(f"{key}|slot={_brief(t)}: branch changed")
            if branch == "undecided" and br != "undecided": failures["branch"].append(f"{key}|slot={_brief(t)}: branch before branch point")
            nearest = min(SPEEDS, key=lambda value: abs(speed - value)) if _finite(speed) else float("nan")
            if not _finite(speed) or abs(speed - nearest) > EPS:
                failures["speed"].append(f"{key}|slot={_brief(t)}: speed={_brief(speed)} not within EPS={EPS:.9g} of nearest public value={_brief(nearest)}")
            if abs(speed - previous_speed) > 0.1 + EPS: failures["speed"].append(f"{key}|slot={_brief(t)}: speed slew={abs(speed-previous_speed):.9g} exceeds bound=0.1 with tolerance={EPS}")
            if arrived and abs(speed) > EPS: failures["speed"].append(f"{key}|slot={_brief(t)}: nonzero speed after arrival")
            active = branch if branch != "undecided" else "a"
            if send:
                if sends >= MAX_SENDS: failures["send"].append(f"{key}|slot={t}: send budget exceeded")
                if q >= QUEUE_CAPACITY: failures["queue"].append(f"{key}|slot={t}: queue full")
            cost = 0.5 * speed * speed + (1.0 if send else 0.0)
            if cost > energy + EPS: failures["energy"].append(f"{key}|slot={t}: energy cost={cost:.9g} exceeds remaining={energy:.9g} with tolerance={EPS}")
            energy -= cost
            if send and sends < MAX_SENDS and q < QUEUE_CAPACITY and cost <= energy + cost + EPS: sends += 1; q += 1
            if q and not (name == "handover_gap" and 44 + 4 * variant <= t < 56 + 4 * variant): q -= 1
            common_limit = 2.0 if branch == "undecided" else float(info[branch][0])
            progress = min(common_limit, progress + 0.1 * speed)
            arrived = branch != "undecided" and progress >= float(info[branch][0]) - EPS
            previous_speed = speed
    return [_atom("decision_legality", name, not errors, "legal" if not errors else errors[0]) for name, errors in failures.items()]


def _replay(name: str, variant: int, row: Mapping[str, Any], *, omit_generated: int | None = None, calculate_useful: bool = True) -> dict[str, Any]:
    info = SCENARIOS[name]; energy = 62.0 if name == "energy_limited" else 80.0; progress = 0.0; branch = "undecided"; speed_prev = 0.0; uncertainty = 0.10; age = 6; sends = 0; queue: list[dict[str, Any]] = []; responses: dict[int, list[dict[str, Any]]] = {}; measurements: list[dict[str, Any]] = []; updates: list[dict[str, Any]] = []; slots_out: list[dict[str, Any]] = []; risk = 0; excess = 0.0; motion_energy = 0.0; arrival = None; useful_direct: set[int] = set()
    raw_slots = {int(x["t"]): x for x in row.get("slots", []) if isinstance(x, dict) and _integer(x.get("t"))}
    for t in range(HORIZON):
        for event in sorted(responses.pop(t, []), key=lambda x: x["id"]):
            if event["lost"]: continue
            before = uncertainty; corr = _clip(0.35 * event["u_generated"] + 0.025 * event["age"]); uncertainty = min(uncertainty, corr); age = event["age"]
            updates.append({"id": event["id"], "arrival": t, "before": before, "corr": corr, "after": uncertainty})
            if uncertainty < before - EPS: useful_direct.add(event["generated"])
        age_start = age
        slot = raw_slots.get(t, {"t": t, "branch": "undecided", "speed": 0.0, "send": False})
        br = slot.get("branch", "undecided"); speed = float(slot.get("speed", 0.0)) if _finite(slot.get("speed")) else 0.0; send = bool(slot.get("send"))
        if branch == "undecided" and isinstance(br, str) and br in {"a", "b"} and progress >= 2.0 - EPS: branch = br
        if branch == "undecided": active = "a"; length = float(info["a"][0])
        else: active = branch; length = float(info[branch][0])
        if branch != "undecided" and speed == 0.0 and progress >= length - EPS and arrival is None: arrival = t
        heading_before = _branch_heading(info, active, progress)
        admitted = send and sends < MAX_SENDS and len(queue) < QUEUE_CAPACITY and energy + EPS >= 0.5 * speed * speed + 1.0 and omit_generated != t
        if admitted:
            energy -= 1.0; sends += 1; item = {"generated": t, "id": f"measurement_id={name}|{variant}|g{t}|a0", "u_generated": uncertainty}; queue.append(item)
        energy -= 0.5 * speed * speed; motion_energy += 0.5 * speed * speed
        if queue and not (name == "handover_gap" and 44 + 4 * variant <= t < 56 + 4 * variant):
            item = queue.pop(0); q = t; age_resp = (item["generated"] + _phase(name, variant) + 4 * variant) % 3; at = q + 1 + age_resp; lost = _loss(name, variant, item["generated"], q); item.update({"service": q, "arrival": at, "age": at - item["generated"], "lost": lost}); measurements.append(item)
            # Boundary 120 is still a valid evidence boundary: it reveals due
            # responses after the final action, although it has no action slot.
            # Only arrivals strictly after the horizon are ignored.
            if at <= HORIZON: responses.setdefault(at, []).append(item)
        next_progress = min(length, progress + 0.1 * speed) if branch != "undecided" else min(2.0, progress + 0.1 * speed)
        heading_after = _branch_heading(info, active, next_progress); omega = (heading_after - heading_before) / SLOT_DT
        uncertainty = _clip(uncertainty + 0.015 * speed + 0.01 * abs(omega)); age = min(6, age + 1)
        clearance = _clearance(info, branch, variant, t, next_progress)
        if clearance < RISK_CLEARANCE - EPS: risk += 1
        required = _clip((clearance - RISK_CLEARANCE) / 1.0); excess += max(0.0, uncertainty - required)
        progress = next_progress
        if arrival is None and progress >= length - EPS: arrival = t + 1
        slots_out.append({"t": t, "branch": branch if branch != "undecided" else "undecided", "speed": speed, "send": send, "progress": progress, "clearance": clearance, "uncertainty": uncertainty, "energy": energy, "age": age_start})
        speed_prev = speed
    # Reveal responses due at terminal boundary 120.  There is no subsequent
    # action or slot metric, but the response remains valid replay evidence and
    # must be processed under the public event-order contract.
    for event in sorted(responses.pop(HORIZON, []), key=lambda x: x["id"]):
        if event["lost"]: continue
        before = uncertainty; corr = _clip(0.35 * event["u_generated"] + 0.025 * event["age"]); uncertainty = min(uncertainty, corr); age = event["age"]
        updates.append({"id": event["id"], "arrival": HORIZON, "before": before, "corr": corr, "after": uncertainty})
        if uncertainty < before - EPS: useful_direct.add(event["generated"])
    useful: set[int] = set()
    if calculate_useful:
        for item in measurements:
            generated = int(item["generated"])
            # A response revealed exactly at terminal boundary 120 is valid;
            # only arrivals strictly after that boundary are excluded.
            if item.get("lost") or item.get("arrival", HORIZON) > HORIZON or generated not in useful_direct:
                continue
            without_risk, without_excess = _replay_without_correction(name, variant, {**row, **{"_baseline": True}, "measurements": measurements, "slots": slots_out}, generated)
            if without_risk > risk or without_excess > excess + EPS:
                useful.add(generated)
    deadline = int(info["deadline"]); reached_on_time = arrival is not None and arrival <= deadline and risk == 0
    final_progress = progress; travel = sum(float(x["speed"]) * SLOT_DT for x in slots_out); length = float(info[branch][0]) if branch in {"a", "b"} else 2.0; lmin = min(float(info["a"][0]), float(info["b"][0]))
    p1 = 0.7 * (1.0 if reached_on_time else 0.5 * _clip(final_progress / max(length, EPS))) + 0.3 * (1.0 if arrival is not None and arrival <= deadline else (0.5 * _clip(1.0 - (arrival - deadline) / max(1, HORIZON - deadline)) if arrival is not None else 0.0))
    p2 = 0.6 * _clip(1.0 - risk / HORIZON) + 0.4 * _clip(1.0 - excess / HORIZON)
    route_eff = lmin / max(length, travel, EPS) if arrival is not None else 0.0
    p3 = 0.6 * _clip(route_eff) + 0.4 * _clip(1.0 - motion_energy / 60.0)
    sent_bits = sends * PACKET_BITS; useful_bits = len(useful) * PACKET_BITS
    a1 = 0.6 * useful_bits / max(sent_bits, 1) + 0.4 * useful_bits / (MAX_SENDS * PACKET_BITS)
    a2 = sum(_clip(1.0 - float(x["age"]) / 6.0) for x in slots_out) / HORIZON
    return {"slots": slots_out, "measurements": measurements, "updates": updates, "arrival": arrival, "risk": risk, "excess": excess, "motion_energy": motion_energy, "sent_bits": sent_bits, "useful_bits": useful_bits, "metrics": {"primary_metrics_1": _clip(p1), "primary_metrics_2": _clip(p2), "primary_metrics_3": _clip(p3), "auxiliary_metrics_1": _clip(a1), "auxiliary_metrics_2": _clip(a2)}}


def _replay_atoms(replayed: Mapping[tuple[str, int], Any], errors: Sequence[str]) -> list[AtomVerdict]:
    ok = not errors and set(replayed) == set(SCENARIO_KEYS)
    route = network = uncertainty = safety = metrics = ok
    for sim in replayed.values():
        route &= isinstance(sim.get("slots"), list) and len(sim["slots"]) == HORIZON and all(_finite(x.get("progress")) for x in sim["slots"])
        network &= all(x.get("service", 0) >= x.get("generated", 0) for x in sim.get("measurements", []))
        uncertainty &= all(_finite(x.get("uncertainty")) and 0 <= x.get("uncertainty", -1) <= 1 for x in sim.get("slots", []))
        safety &= sim.get("risk", -1) >= 0 and sim.get("excess", -1) >= 0
        metrics &= all(_finite(v) and 0 <= v <= 1 for v in sim.get("metrics", {}).values())
    return [_atom("replay_validity", "route", route, "route replay valid"), _atom("replay_validity", "network", network, "queue and response replay valid"), _atom("replay_validity", "uncertainty", uncertainty, "uncertainty replay valid"), _atom("replay_validity", "safety", safety, "safety replay valid"), _atom("replay_validity", "metrics", metrics, "metrics finite and bounded")]


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    return {key: sum(float(sim["metrics"][key]) for sim in replayed.values()) / max(len(replayed), 1) for key in METRIC_KEYS}


def _gated_metrics(scores: Mapping[str, float], replayed: Mapping[tuple[str, int], Any]) -> tuple[bool, dict[str, float]]:
    gate = all(scores.get(k, 0.0) >= 1.0 - EPS for k in GATE_CHECKS)
    return gate, _metrics(replayed) if gate else {key: 0.0 for key in METRIC_KEYS}


def _headings(text: str) -> dict[str, str]:
    """Parse visible Markdown headings and retain a non-empty body for each title."""
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


def _failed_atoms(group: str, detail: str) -> list[AtomVerdict]:
    return [
        _atom(group, ident.split(".", 1)[1], False, detail)
        for ident in ATOM_SHARES[group]
    ]


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    result_path = Path(ws.root) / "results.json"
    remove_ok = False
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
        remove_ok = getattr(removed, "exit_code", None) == 0
    except Exception as exc:
        state.official_message = f"old results removal failed: {type(exc).__name__}: {exc}"
        state.scores = {name: 0.0 for name in GATE_CHECKS}
        state.diagnostics = {name: state.official_message for name in GATE_CHECKS}
        _CACHE[key] = state
        return state

    state.run_started = True
    started = time.monotonic()
    try:
        result = await ws.run("python run.py", check=False, timeout=600)
        state.exit_code = getattr(result, "exit_code", None)
        stderr = str(getattr(result, "stderr", ""))
        state.timed_out = state.exit_code is None and (
            "timeout" in stderr.lower() or "timed out" in stderr.lower()
        )
    except Exception as exc:
        state.exit_code = None
        state.timed_out = "timeout" in f"{type(exc).__name__}: {exc}".lower()
        state.official_message = f"official run raised {type(exc).__name__}: {exc}"
    finally:
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
    state.result_file_is_new_regular = result_path.is_file() and not result_path.is_symlink()
    if state.exit_code is None and state.elapsed_s is not None and state.elapsed_s >= 600 - EPS:
        state.timed_out = True
    state.official_ok = bool(
        remove_ok
        and state.exit_code == 0
        and state.result_file_is_new_regular
        and state.elapsed_s <= 600 + EPS
    )
    status = (
        f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
        f"new_regular_results={state.result_file_is_new_regular}, timed_out={state.timed_out}"
    )
    state.official_message = f"{state.official_message}; {status}" if state.official_message else status
    state.scores["official_run"] = float(state.official_ok)
    state.diagnostics["official_run"] = state.official_message

    if state.result_file_is_new_regular:
        try:
            state.data = _strict_json(result_path.read_text(encoding="utf-8"))
            state.parse_ok = True
        except Exception as exc:
            state.diagnostics["result_json"] = f"strict parse failed: {type(exc).__name__}: {exc}"

    try:
        shape = _shape_atoms(state.data, state.parse_ok)
    except Exception as exc:
        shape = _failed_atoms("result_json", f"shape check raised {type(exc).__name__}: {exc}")
    try:
        evidence = _evidence_atoms(state.data)
    except Exception as exc:
        evidence = _failed_atoms("result_evidence", f"evidence check raised {type(exc).__name__}: {exc}")
    try:
        state.records = _index_records(state.data)
    except Exception as exc:
        state.records = {}
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", "") + f"; index failed: {type(exc).__name__}: {exc}"
    try:
        coverage, defects = _coverage(state.data)
    except Exception as exc:
        coverage, defects = 0.0, [f"coverage check raised {type(exc).__name__}: {exc}"]
    try:
        legality = _legality(state.records)
    except Exception as exc:
        legality = _failed_atoms("decision_legality", f"legality check raised {type(exc).__name__}: {exc}")

    state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
    state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
    state.scores["canonical_coverage"] = coverage
    state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])

    replay_errors: list[str] = []
    for unit in SCENARIO_KEYS:
        if unit not in state.records:
            continue
        try:
            state.replayed[unit] = _replay(unit[0], unit[1], state.records[unit])
        except Exception as exc:
            replay_errors.append(
                f"{unit}: replay raised {type(exc).__name__}: {exc}"
            )
    try:
        replay_atoms = _replay_atoms(state.replayed, replay_errors)
    except Exception as exc:
        replay_atoms = _failed_atoms("replay_validity", f"replay check raised {type(exc).__name__}: {exc}")
    state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
    state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose(shape))
    state.diagnostics["result_evidence"] = _diagnose(evidence)
    state.diagnostics["canonical_coverage"] = (
        f"defect_count={len(defects)}" + ("; " + " | ".join(defects[:12]) if defects else "")
    )
    state.diagnostics["decision_legality"] = _diagnose(legality)
    state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
    state.validity_gate = state.official_ok and all(
        state.scores.get(name, 0.0) >= 1.0 - EPS for name in GATE_CHECKS
    )
    if state.validity_gate:
        try:
            state.metrics = _metrics(state.replayed)
        except Exception as exc:
            state.validity_gate = False
            state.scores["replay_validity"] = 0.0
            state.diagnostics["replay_validity"] = (
                "failed_atoms=[replay_validity.metrics]; "
                f"replay_validity.metrics: metric calculation raised {type(exc).__name__}: {exc}"
            )
    if not state.validity_gate:
        state.metrics = {name: 0.0 for name in METRIC_KEYS}
    if not state.official_ok:
        for name in GATE_CHECKS:
            state.scores[name] = 0.0
            state.diagnostics[name] = f"official_run failed: {state.official_message}; {state.diagnostics.get(name, '')}"
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink(): return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (1.0, "static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "static compilation failed")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws); return float(state.official_ok), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]: return _report_score(ws)


for _id in GATE_CHECKS:
    if _id == "official_run": continue
    async def _gate(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws); return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")
    globals()[_id] = check(id=_id)(_gate)
for _id in METRIC_KEYS:
    async def _metric(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws); value = state.metrics.get(ident, 0.0) if state.validity_gate else 0.0; failed = [k for k in GATE_CHECKS if state.scores.get(k, 0.0) < 1.0 - EPS]; return value, f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}; failed_gate_checks={failed}"
    globals()[_id] = check(id=_id)(_metric)
del _gate, _metric
