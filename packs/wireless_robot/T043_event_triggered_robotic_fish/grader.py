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

EPS = 1e-6
HORIZON = 180
SLOT_DT = 0.1
MAX_TRIGGER = 36
MIN_TRIGGER_GAP = 3
MAX_IN_FLIGHT = 1
INITIAL_ENERGY = 72.0
OFFICIAL_TIMEOUT = 600.0
HEADING_ERROR_SCALE = 0.68
METRIC_KEYS = ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")
SCENARIO_NAMES = ("single_turn", "double_turn", "current_bias", "rate_drop", "narrow_channel")
CANONICAL = tuple(f"{name}_v{variant}" for name in SCENARIO_NAMES for variant in range(3))
GATE_CHECKS = ("official_run", "result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
ATOM_SHARES = {
    "result_json": {"result_json.parse": 0.20, "result_json.root": 0.30, "result_json.unit_shape": 0.50},
    "result_evidence": {"result_evidence.condition_fields": 0.35, "result_evidence.slot_fields": 0.65},
    "decision_legality": {"decision_legality.trigger": 0.30, "decision_legality.control": 0.25, "decision_legality.energy": 0.20, "decision_legality.in_flight": 0.25},
    "replay_validity": {"replay_validity.dynamics": 0.25, "replay_validity.network": 0.20, "replay_validity.segments": 0.20, "replay_validity.safety": 0.15, "replay_validity.metrics": 0.20},
}
CONDITION_FIELDS = {"condition_id", "slots"}
SLOT_FIELDS = {"slot", "trigger", "u"}

SCENARIOS: dict[str, dict[str, Any]] = {
    "single_turn": {"psi": (0.0, 0.0, 0.32, 0.58, 0.58), "q": (1.0, 2.0, 3.1, 4.2, 5.6), "dead": (34, 66, 102, 138, 170), "width": 1.30, "amp": 0.025, "bias": 0.0, "phase": 2, "loss": 1},
    "double_turn": {"psi": (0.0, 0.36, 0.68, 0.22, 0.0), "q": (1.0, 2.0, 3.1, 4.2, 5.6), "dead": (34, 68, 106, 142, 172), "width": 1.24, "amp": 0.030, "bias": 0.0, "phase": 5, "loss": 2},
    "current_bias": {"psi": (0.0, 0.24, 0.50, 0.24, 0.0), "q": (1.0, 2.1, 3.2, 4.4, 5.7), "dead": (36, 70, 108, 146, 174), "width": 1.22, "amp": 0.028, "bias": 0.020, "phase": 8, "loss": 2},
    "rate_drop": {"psi": (0.0, -0.30, -0.58, -0.30, 0.0), "q": (1.0, 2.0, 3.1, 4.2, 5.6), "dead": (34, 68, 106, 142, 172), "width": 1.28, "amp": 0.026, "bias": 0.0, "phase": 11, "loss": 4},
    "narrow_channel": {"psi": (0.0, 0.22, 0.46, 0.46, 0.12), "q": (1.0, 2.0, 3.0, 4.1, 5.5), "dead": (34, 66, 102, 138, 170), "width": 0.92, "amp": 0.024, "bias": 0.0, "phase": 14, "loss": 3},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, bool(passed), str(detail), float(share)


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
        self.records: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(float(value)) or value <= 0 for value in expected.values()):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share mismatch")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("atom shares do not sum to one")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(atom.id for atom in failed) + "]; " + " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)


def _condition_name(condition_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_v([0-2])", condition_id)
    if not match or match.group(1) not in SCENARIOS:
        return None
    return match.group(1), int(match.group(2))


def _shape_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, dict) and set(data) == {"conditions"}
    units = isinstance(data, dict) and isinstance(data.get("conditions"), list) and all(isinstance(row, dict) and set(row) == CONDITION_FIELDS for row in data["conditions"])
    return [_atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "strict JSON parse failed"), _atom("result_json", "root", root, "exact root key"), _atom("result_json", "unit_shape", units, "exact condition fields")]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("conditions") if isinstance(data, dict) else None
    condition_ok = isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("condition_id"), str) and isinstance(row.get("slots"), list) for row in rows)
    slot_ok = True
    issue = "slot fields valid"
    if not isinstance(rows, list):
        slot_ok, issue = False, "conditions is not an array"
    else:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("slots"), list):
                slot_ok, issue = False, "slots is not an array"
                break
            for item in row["slots"]:
                if not isinstance(item, dict) or set(item) != SLOT_FIELDS:
                    slot_ok, issue = False, "slot fields are not exact"
                    break
                if not _integer(item.get("slot")) or not isinstance(item.get("trigger"), bool) or not _finite(item.get("u")):
                    slot_ok, issue = False, f"invalid slot={item.get('slot')} types"
                    break
            if not slot_ok:
                break
    return [_atom("result_evidence", "condition_fields", condition_ok, "condition fields valid" if condition_ok else "condition fields invalid"), _atom("result_evidence", "slot_fields", slot_ok, issue)]


def _index(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rows = data.get("conditions") if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return result
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("condition_id"), str) and row["condition_id"] not in result:
            result[row["condition_id"]] = row
    return result


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = data.get("conditions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return 0.0, ["conditions is not an array"]
    seen = [row.get("condition_id") if isinstance(row, dict) else None for row in rows]
    for key in CANONICAL:
        count = sum(isinstance(value, str) and value == key for value in seen)
        if count == 0:
            defects.append(f"missing condition {key}")
        if count > 1:
            defects.append(f"duplicate condition {key}")
    for key in seen:
        if not isinstance(key, str) or key not in CANONICAL:
            defects.append(f"unknown condition {key!r}")
    for key in CANONICAL:
        candidates = [row for row in rows if isinstance(row, dict) and row.get("condition_id") == key]
        if len(candidates) != 1 or not isinstance(candidates[0].get("slots"), list):
            continue
        slots = [item.get("slot") for item in candidates[0]["slots"] if isinstance(item, dict)]
        for slot in range(HORIZON):
            count = sum(_integer(value) and value == slot for value in slots)
            if count == 0:
                defects.append(f"missing slot {key}|{slot}")
            if count > 1:
                defects.append(f"duplicate slot {key}|{slot}")
        for slot in slots:
            if not _integer(slot) or slot not in range(HORIZON):
                defects.append(f"unknown slot {key}|{slot!r}")
    return max(0.0, min(1.0, (4.0 - 0.5 * len(defects)) / 4.0)), defects


def _valid_slot(item: Any) -> bool:
    return isinstance(item, dict) and set(item) == SLOT_FIELDS and _integer(item.get("slot")) and isinstance(item.get("trigger"), bool) and _finite(item.get("u"))


def _phase(name: str, variant: int) -> int:
    return int(SCENARIOS[name]["phase"] + 5 * variant)


def _response_delay(name: str, variant: int, generated: int) -> int:
    return 1 + ((generated + _phase(name, variant)) % 4)


def _lost(name: str, variant: int, generated: int) -> bool:
    info = SCENARIOS[name]
    return (13 * generated + 7 * variant + int(info["phase"])) % 17 < int(info["loss"])


def _width(info: Mapping[str, Any], variant: int, segment: int) -> float:
    return float(info["width"]) - (0.06 * variant if info is SCENARIOS["narrow_channel"] and segment in {2, 3} else 0.0)


def _segment(info: Mapping[str, Any], q: float) -> int:
    for index, target in enumerate(info["q"]):
        if q < float(target) - EPS:
            return index
    return 4


def _truth_innovation(seed: int, slot: int) -> float:
    """Fixed environment-truth stream; never part of the controller input."""
    residue = (37 * (slot + 1) + 53 * seed + 11 * (slot + 1) * seed) % 101
    return (residue - 50) / 50.0


def _legality(records: Mapping[str, Any]) -> list[AtomVerdict]:
    failures = {name: [] for name in ("trigger", "control", "energy", "in_flight")}
    for condition_id, row in records.items():
        parsed = _condition_name(condition_id)
        if parsed is None or not isinstance(row, dict) or not isinstance(row.get("slots"), list):
            continue
        name, variant = parsed
        previous_trigger = -10**9
        trigger_count = 0
        energy = INITIAL_ENERGY
        pending: list[int] = []
        seen_slots: set[int] = set()
        valid_slot_keys = (x for x in row["slots"] if isinstance(x, dict) and _integer(x.get("slot")))
        for item in sorted(valid_slot_keys, key=lambda x: int(x["slot"])):
            slot = item.get("slot")
            if not _valid_slot(item) or not _integer(slot) or slot in seen_slots:
                continue
            seen_slots.add(slot)
            trigger = bool(item["trigger"])
            u = float(item["u"])
            pending = [arrival for arrival in pending if arrival > int(slot)]
            control_valid = _finite(u) and -1.0 - EPS <= u <= 1.0 + EPS
            control_cost = u * u if control_valid else 0.0
            if not control_valid:
                failures["control"].append(f"{condition_id}|slot={slot}: u={u:.9g} outside [-1,1]")
            if energy + EPS < control_cost:
                failures["energy"].append(f"{condition_id}|slot={slot}: control cost={control_cost:.9g} exceeds energy={energy:.9g}")
            spacing_ok = int(slot) - previous_trigger >= MIN_TRIGGER_GAP
            budget_ok = trigger_count < MAX_TRIGGER
            in_flight_ok = len(pending) < MAX_IN_FLIGHT
            trigger_energy_ok = energy + EPS >= control_cost + 1.0
            if trigger:
                if not spacing_ok:
                    failures["trigger"].append(f"{condition_id}|slot={slot}: trigger gap={int(slot)-previous_trigger} below {MIN_TRIGGER_GAP}")
                if not budget_ok:
                    failures["trigger"].append(f"{condition_id}|slot={slot}: trigger budget exceeded")
                if not trigger_energy_ok:
                    failures["energy"].append(f"{condition_id}|slot={slot}: cost={control_cost+1:.9g} exceeds energy={energy:.9g}")
                if not in_flight_ok:
                    failures["in_flight"].append(
                        f"{condition_id}|slot={slot}: in-flight={len(pending)} reaches capacity={MAX_IN_FLIGHT} before admission"
                    )
            admitted = trigger and control_valid and spacing_ok and budget_ok and in_flight_ok and trigger_energy_ok
            if admitted:
                previous_trigger = int(slot)
                trigger_count += 1
                pending.append(int(slot) + _response_delay(name, variant, int(slot)))
            energy -= control_cost + (1.0 if admitted else 0.0)
    return [_atom("decision_legality", name, not errors, "legal" if not errors else errors[0]) for name, errors in failures.items()]


def _replay(name: str, variant: int, row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    info = SCENARIOS[name]
    errors: list[str] = []
    raw_slots = {int(item["slot"]): item for item in row.get("slots", []) if isinstance(item, dict) and _integer(item.get("slot"))}
    q = e = psi = r = 0.0
    energy = INITIAL_ENERGY
    u_history = [0.0, 0.0]
    bq = be = bpsi = br = 0.0
    active_generation: int | None = None
    pending: dict[int, list[dict[str, Any]]] = {}
    measurements: list[dict[str, Any]] = []
    ages: list[float] = []
    state_slots: list[dict[str, Any]] = []
    heading_errors: list[float] = []
    normalized_cross_errors: list[float] = []
    completions: list[int | None] = [None] * 5
    unsafe_slots = 0
    control_energy = 0.0
    trigger_count = 0
    last_trigger = -10**9
    in_flight: list[int] = []
    deliveries = 0

    def process_arrivals(boundary: int) -> None:
        nonlocal active_generation, deliveries
        events = sorted(pending.pop(boundary, []), key=lambda event: event["generated"])
        for event in events:
            if not event["lost"] and boundary <= HORIZON:
                deliveries += 1
                age = boundary - int(event["generated"])
                if age <= 8 and (active_generation is None or int(event["generated"]) > active_generation):
                    active_generation = int(event["generated"])

    for t in range(HORIZON):
        process_arrivals(t)
        in_flight = [arrival for arrival in in_flight if arrival > t]
        if active_generation is None:
            ages.append(8.0)
        else:
            ages.append(float(min(8, max(0, t - active_generation))))
        item = raw_slots.get(t)
        if item is None:
            trigger, u = False, 0.0
        else:
            trigger = bool(item.get("trigger"))
            u = float(item.get("u", 0.0)) if _finite(item.get("u")) else 0.0
            u = max(-1.0, min(1.0, u))
        admitted = trigger and (t - last_trigger >= MIN_TRIGGER_GAP) and trigger_count < MAX_TRIGGER and len(in_flight) < MAX_IN_FLIGHT and energy + EPS >= u * u + 1.0
        if admitted:
            trigger_count += 1
            last_trigger = t
            arrival = t + _response_delay(name, variant, t)
            event = {"generated": t, "arrival": arrival, "lost": _lost(name, variant, t)}
            measurements.append(event)
            in_flight.append(arrival)
            if arrival <= HORIZON:
                pending.setdefault(arrival, []).append(event)
        segment = _segment(info, q)
        psi_ref = float(info["psi"][segment])
        c_parallel = 0.015 * math.sin(2.0 * math.pi * (t + 7 * variant) / 36.0)
        truth_seed = int(info["phase"]) + 17 * variant
        c_lateral = (
            float(info["bias"])
            + float(info["amp"])
            * math.sin(2.0 * math.pi * (t + int(info["phase"]) + 7 * variant) / 30.0)
            + 0.08 * _truth_innovation(truth_seed, t)
        )
        delayed = u_history[0] if t >= 2 else 0.0
        r_next = max(
            -1.2,
            min(
                1.2,
                0.82 * r
                + 0.35 * delayed
                + 0.05 * math.sin(2.0 * math.pi * (t + int(info["phase"])) / 27.0)
                + 0.08 * _truth_innovation(truth_seed + 19, t),
            ),
        )
        psi_next = _wrap(psi + 0.1 * r_next)
        q_next = max(0.0, min(float(info["q"][-1]), q + 0.1 * (0.35 * math.cos(psi - psi_ref) + c_parallel)))
        e_next = e + 0.1 * (0.35 * math.sin(psi - psi_ref) + c_lateral)
        bq = max(-0.12, min(0.12, 0.97 * bq + 0.001))
        be = max(
            -0.12,
            min(
                0.12,
                0.97 * be
                + 0.002 * (1 if c_lateral > 0 else -1 if c_lateral < 0 else 0)
                + 0.008 * _truth_innovation(truth_seed + 43, t),
            ),
        )
        bpsi = max(
            -0.20,
            min(
                0.20,
                0.96 * bpsi
                - 0.003 * u
                + 0.006 * _truth_innovation(truth_seed + 71, t),
            ),
        )
        br = 0.90 * br + 0.01 * u + 0.008 * _truth_innovation(truth_seed + 89, t)
        cost = u * u + (1.0 if admitted else 0.0)
        energy -= cost
        control_energy += u * u
        next_segment = _segment(info, q_next)
        next_psi_ref = float(info["psi"][next_segment])
        width = _width(info, variant, next_segment)
        unsafe = abs(e_next) > width / 2.0 - 0.18 + EPS
        if unsafe:
            unsafe_slots += 1
        current_completion = next((index for index, completion in enumerate(completions) if completion is None), None)
        if current_completion is not None and q_next >= float(info["q"][current_completion]) - EPS and abs(_wrap(psi_next - float(info["psi"][current_completion]))) <= 0.25 + EPS and not unsafe:
            completions[current_completion] = t + 1
        heading_errors.append(_wrap(psi_next - next_psi_ref))
        clearance = width / 2.0 - 0.18
        normalized_cross_errors.append(e_next / clearance)
        state_slots.append(
            {
                "slot": t,
                "q": q_next,
                "e": e_next,
                "psi": psi_next,
                "r": r_next,
                "bq": bq,
                "be": be,
                "bpsi": bpsi,
                "br": br,
                "energy": energy,
                "segment": next_segment,
                "age": ages[-1],
                "unsafe": unsafe,
            }
        )
        q, e, psi, r = q_next, e_next, psi_next, r_next
        u_history = [u_history[1], u]
    process_arrivals(HORIZON)
    if any(
        not _finite(value)
        for slot in state_slots
        for value in (
            slot["q"],
            slot["e"],
            slot["psi"],
            slot["r"],
            slot["bq"],
            slot["be"],
            slot["bpsi"],
            slot["br"],
            slot["energy"],
        )
    ):
        errors.append("non-finite replay state")
    def rms(values: Sequence[float]) -> float:
        return math.sqrt(sum(value * value for value in values) / max(1, len(values)))
    p1 = 0.6 * _clip(1.0 - rms(heading_errors) / HEADING_ERROR_SCALE) + 0.4 * _clip(1.0 - rms(normalized_cross_errors))
    timely: list[float] = []
    for completion, deadline in zip(completions, info["dead"]):
        if completion is None:
            timely.append(0.0)
        elif completion <= deadline:
            timely.append(1.0)
        else:
            timely.append(0.5 * _clip(1.0 - (completion - deadline) / max(1, HORIZON - deadline)))
    p2 = 0.7 * sum(completion is not None for completion in completions) / 5.0 + 0.3 * sum(timely) / 5.0
    p3 = 0.7 * _clip(1.0 - unsafe_slots / HORIZON) + 0.3 * _clip(1.0 - control_energy / HORIZON)
    delivery_fraction = deliveries / trigger_count if trigger_count else 0.0
    a1 = 0.6 * _clip(1.0 - trigger_count / MAX_TRIGGER) + 0.4 * _clip(delivery_fraction)
    a2 = sum(_clip(1.0 - age / 8.0) for age in ages) / HORIZON
    metrics = {"primary_metrics_1": _clip(p1), "primary_metrics_2": _clip(p2), "primary_metrics_3": _clip(p3), "auxiliary_metrics_1": _clip(a1), "auxiliary_metrics_2": _clip(a2)}
    return {"slots": state_slots, "measurements": measurements, "completions": completions, "unsafe_slots": unsafe_slots, "control_energy": control_energy, "trigger_count": trigger_count, "deliveries": deliveries, "metrics": metrics}, errors


def _replay_atoms(replayed: Mapping[str, Any], errors: Sequence[str]) -> list[AtomVerdict]:
    issues: dict[str, list[str]] = {name: [] for name in ("dynamics", "network", "segments", "safety", "metrics")}
    if errors:
        for name in issues:
            issues[name].append(errors[0])
    for condition_id, sim in replayed.items():
        slots = sim.get("slots")
        if not isinstance(slots, list) or len(slots) != HORIZON:
            issues["dynamics"].append(f"{condition_id}|slots: expected length={HORIZON}, actual={type(slots).__name__}")
        elif not all(
            isinstance(item, dict)
            and all(
                _finite(item.get(field))
                for field in ("q", "e", "psi", "r", "bq", "be", "bpsi", "br", "energy")
            )
            for item in slots
        ):
            issues["dynamics"].append(
                f"{condition_id}|slots: expected finite q/e/psi/r/bq/be/bpsi/br/energy at every boundary"
            )

        measurements = sim.get("measurements")
        if not isinstance(measurements, list):
            issues["network"].append(f"{condition_id}|measurements: expected list, actual={type(measurements).__name__}")
        else:
            for index, item in enumerate(measurements):
                if not isinstance(item, dict) or not _integer(item.get("generated")) or not _integer(item.get("arrival")) or item["arrival"] <= item["generated"]:
                    issues["network"].append(f"{condition_id}|measurements[{index}]: expected integer arrival > generated")
                    break

        completions = sim.get("completions")
        if not isinstance(completions, list) or len(completions) != 5 or not all(value is None or _integer(value) for value in completions):
            issues["segments"].append(f"{condition_id}|completions: expected five null-or-integer boundaries")

        unsafe_slots = sim.get("unsafe_slots")
        control_energy = sim.get("control_energy")
        if not _integer(unsafe_slots) or not 0 <= unsafe_slots <= HORIZON or not _finite(control_energy) or control_energy < 0.0:
            issues["safety"].append(f"{condition_id}|safety: expected unsafe_slots in [0,{HORIZON}] and finite control_energy >= 0")

        unit_metrics = sim.get("metrics")
        if not isinstance(unit_metrics, dict) or set(unit_metrics) != set(METRIC_KEYS) or not all(_finite(value) and 0.0 <= value <= 1.0 for value in unit_metrics.values()):
            issues["metrics"].append(f"{condition_id}|metrics: expected exact finite keys in [0,1]")

    labels = {
        "dynamics": "finite dynamics replay",
        "network": "measurement-event replay",
        "segments": "sequential segment replay",
        "safety": "safety and control-energy replay",
        "metrics": "finite bounded metric replay",
    }
    return [
        _atom(
            "replay_validity",
            name,
            not issues[name],
            f"{labels[name]} valid" if not issues[name] else issues[name][0],
        )
        for name in labels
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    return {key: sum(float(sim["metrics"][key]) for sim in replayed.values()) / max(1, len(replayed)) for key in METRIC_KEYS}


def _metrics_or_zero(replayed: Mapping[str, Any], validity_gate: bool) -> dict[str, float]:
    if not validity_gate:
        return {key: 0.0 for key in METRIC_KEYS}
    return _metrics(replayed)


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    required = {"Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations"}
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular file"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md unreadable: {type(exc).__name__}: {exc}"
    headings = [(i, len(match.group(1)), match.group(2).strip()) for i, line in enumerate(lines) if (match := re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line))]
    found: set[str] = set()
    for idx, level, title in headings:
        if title not in required:
            continue
        end = len(lines)
        for later_index, later_level, _ in headings:
            if later_index > idx and later_level <= level:
                end = later_index
                break
        if any(line.strip() for line in lines[idx + 1:end]):
            found.add(title)
    score = max(0.0, 2.0 - 0.5 * (len(required) - len(found))) / 2.0
    return score, f"missing_sections={sorted(required-found)}; normalized={score:.9g}"


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    result_path = Path(ws.root) / "results.json"
    run_started_at: float | None = None
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
        state.run_started = True
        run_started_at = time.monotonic()
        result = await ws.run("python run.py", check=False, timeout=OFFICIAL_TIMEOUT)
    except Exception as exc:
        state.run_finished = state.run_started
        state.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
        state.elapsed_s = time.monotonic() - run_started_at if run_started_at is not None else None
        message = f"official run raised {type(exc).__name__}: {exc}"
        state.official_message = message
        state.scores = {item: 0.0 for item in GATE_CHECKS}
        state.diagnostics = {item: message for item in GATE_CHECKS}
        _CACHE[key] = state
        return state
    state.run_finished = True
    state.elapsed_s = time.monotonic() - run_started_at
    state.exit_code = getattr(result, "exit_code", None)
    stderr = str(getattr(result, "stderr", ""))
    state.timed_out = state.exit_code is None and (
        "timeout" in stderr.lower()
        or state.elapsed_s >= OFFICIAL_TIMEOUT - EPS
    )
    state.result_file_is_new_regular = result_path.is_file() and not result_path.is_symlink()
    state.official_ok = getattr(removed, "exit_code", None) == 0 and state.exit_code == 0 and not state.timed_out and state.result_file_is_new_regular and state.elapsed_s <= OFFICIAL_TIMEOUT + EPS
    state.official_message = f"exit={state.exit_code}, timed_out={state.timed_out}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
    state.scores["official_run"] = float(state.official_ok)
    state.diagnostics["official_run"] = state.official_message
    if not state.official_ok:
        for item in GATE_CHECKS:
            if item != "official_run":
                state.scores[item] = 0.0
                state.diagnostics[item] = f"official_run failed; {state.official_message}"
        _CACHE[key] = state
        return state
    try:
        state.data = _strict_json(result_path.read_text(encoding="utf-8"))
        state.parse_ok = True
    except Exception as exc:
        state.diagnostics["result_json"] = f"strict parse failed: {type(exc).__name__}: {exc}"
    try:
        shape = _shape_atoms(state.data, state.parse_ok)
        evidence = _evidence_atoms(state.data)
        state.records = _index(state.data)
        coverage, defects = _coverage(state.data)
        legality = _legality(state.records)
        replay_errors: list[str] = []
        for condition_id in CANONICAL:
            if condition_id in state.records:
                parsed = _condition_name(condition_id)
                if parsed is not None:
                    try:
                        state.replayed[condition_id], errors = _replay(parsed[0], parsed[1], state.records[condition_id])
                        replay_errors.extend(f"{condition_id}: {error}" for error in errors)
                    except Exception as exc:
                        replay_errors.append(f"{condition_id}: replay raised {type(exc).__name__}: {exc}")
        replay_atoms = _replay_atoms(state.replayed, replay_errors)
        state.scores.update({"result_json": _score_atoms(shape, ATOM_SHARES["result_json"]), "result_evidence": _score_atoms(evidence, ATOM_SHARES["result_evidence"]), "canonical_coverage": coverage, "decision_legality": _score_atoms(legality, ATOM_SHARES["decision_legality"]), "replay_validity": _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])})
        state.diagnostics.update({"result_json": state.diagnostics.get("result_json", _diagnose(shape)), "result_evidence": _diagnose(evidence), "canonical_coverage": f"defect_count={len(defects)}" + (("; " + " | ".join(defects[:12])) if defects else ""), "decision_legality": _diagnose(legality), "replay_validity": _diagnose(replay_atoms)})
        state.validity_gate = all(state.scores.get(item, 0.0) >= 1.0 - EPS for item in GATE_CHECKS)
        state.metrics = _metrics_or_zero(state.replayed, state.validity_gate)
    except Exception as exc:
        message = f"grader evaluation failed: {type(exc).__name__}: {exc}"
        for item in GATE_CHECKS:
            if item != "official_run":
                state.scores[item] = 0.0
                state.diagnostics[item] = message
        state.metrics = {key: 0.0 for key in METRIC_KEYS}
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink():
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (1.0, "static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "static compilation failed")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return float(state.official_ok), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


for _id in GATE_CHECKS:
    if _id == "official_run":
        continue

    async def _gate(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    globals()[_id] = check(id=_id)(_gate)

for _id in METRIC_KEYS:

    async def _metric(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        value = state.metrics.get(ident, 0.0) if state.validity_gate else 0.0
        failed = [item for item in GATE_CHECKS if state.scores.get(item, 0.0) < 1.0 - EPS]
        return value, f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}; failed_gate_checks={failed}"

    globals()[_id] = check(id=_id)(_metric)

del _gate, _metric
