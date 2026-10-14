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
HORIZON = 90
DT = 0.25
MAX_SPEED = 0.8
TARGET_SLEW = 0.05
SAFETY_CLEARANCE = 0.35
RISK_CLEARANCE = 0.60
REQUEST_COUNT = 30
PATH_LIMIT = 2
PATH_BITS = 400
UPLINK_BUDGET = 16800
FALLBACK_COST = 3.0

SCENARIO_NAMES = (
    "balanced_links",
    "handover_gap",
    "correlated_outage",
    "server_burst",
    "deadline_tight",
)
SCENARIO_KEYS = tuple((name, variant) for name in SCENARIO_NAMES for variant in range(3))
REQUEST_SLOTS = lambda variant: [variant + 3 * k for k in range(REQUEST_COUNT)]
INTERFACES = {"I0": (2.0, 1.0), "I1": (7.0, 5.0), "I2": (12.0, 8.0)}
REPLICAS = ("C0", "C1", "C2")

UNIT_FIELDS = {"scenario", "variant", "requests", "retries", "controls"}
REQUEST_FIELDS = {"request_id", "generated_slot", "selected_paths", "fallback"}
RETRY_FIELDS = {"request_id", "interface_id", "replica_id", "send_slot"}
CONTROL_FIELDS = {"slot", "target_velocity"}

METRIC_KEYS = (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)
GATE_CHECKS = (
    "official_run",
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)
ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.30,
        "result_json.unit_shape": 0.50,
    },
    "result_evidence": {
        "result_evidence.unit_identifiers": 0.15,
        "result_evidence.request_fields": 0.35,
        "result_evidence.retry_fields": 0.20,
        "result_evidence.control_fields": 0.30,
    },
    "decision_legality": {
        "decision_legality.request_references": 0.30,
        "decision_legality.path_selection": 0.25,
        "decision_legality.retry_window": 0.25,
        "decision_legality.motion": 0.20,
    },
    "replay_validity": {
        "replay_validity.observations": 0.12,
        "replay_validity.network": 0.24,
        "replay_validity.responses": 0.18,
        "replay_validity.fallbacks": 0.10,
        "replay_validity.motion": 0.20,
        "replay_validity.mission": 0.16,
    },
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "balanced_links": {
        "robot": (1.0, 5.0), "target": (12.0, 5.0), "tv": (0.02, 0.0),
        "obstacles": (("circle", 6.0, 3.0, 0.8), ("circle", 6.0, 7.0, 0.8)), "deadline": 76,
    },
    "handover_gap": {
        "robot": (1.0, 2.0), "target": (12.0, 8.0), "tv": (0.0, 0.02),
        "obstacles": (("rect", 5.0, 6.0, 4.0, 6.0),), "deadline": 80,
    },
    "correlated_outage": {
        "robot": (1.0, 8.0), "target": (12.0, 2.0), "tv": (0.01, 0.01),
        "obstacles": (("circle", 6.0, 5.0, 0.9),), "deadline": 82,
    },
    "server_burst": {
        "robot": (1.0, 5.0), "target": (12.0, 5.0), "tv": (0.0, 0.02),
        "obstacles": (("circle", 4.0, 5.0, 0.7), ("circle", 8.0, 5.0, 0.7)), "deadline": 78,
    },
    "deadline_tight": {
        "robot": (1.0, 5.0), "target": (11.0, 5.0), "tv": (0.02, 0.0),
        "obstacles": (("rect", 5.0, 6.0, 0.0, 3.0), ("rect", 5.0, 6.0, 7.0, 9.0)), "deadline": 66,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
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
        self.records: dict[tuple[str, int], Any] = {}
        self.scenarios: dict[tuple[str, int], Any] = {}
        self.replayed: dict[tuple[str, int], Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.validity_gate = False


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


def _safe_text(value: Any) -> str:
    """Return bounded diagnostic text even for pathological scalar values."""
    return value if isinstance(value, str) else _brief(value)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_json(raw: str) -> Any:
    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number {value}")
        return number

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=reject_duplicate,
    )


def _atom(group: str, short_name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{short_name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared fixed set")
    if any(not math.isfinite(float(value)) or value <= 0.0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share differs from declared share")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _failed_atoms(group: str, detail: str) -> list[AtomVerdict]:
    return [
        _atom(group, key.rsplit(".", 1)[1], False, detail)
        for key in ATOM_SHARES[group]
    ]


def _shape_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = isinstance(data, dict) and set(data) == {"scenarios"}
    unit_ok = root_ok and isinstance(data.get("scenarios"), list) and all(
        isinstance(row, dict) and set(row) == UNIT_FIELDS for row in data["scenarios"]
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "strict JSON parse failed"),
        _atom("result_json", "root", root_ok, f"root_keys={_brief(list(data) if isinstance(data, dict) else type(data).__name__)}"),
        _atom("result_json", "unit_shape", unit_ok, "exact unit fields" if unit_ok else "unit fields are not exact"),
    ]


def _path_key(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, dict) or set(value) != {"interface_id", "replica_id"}:
        return None
    interface, replica = value.get("interface_id"), value.get("replica_id")
    if not isinstance(interface, str) or not isinstance(replica, str):
        return None
    return (interface, replica) if interface in INTERFACES and replica in REPLICAS else None


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("scenarios") if isinstance(data, dict) else None
    unit_issue: str | None = None
    request_issue: str | None = None
    retry_issue: str | None = None
    control_issue: str | None = None
    if not isinstance(rows, list):
        detail = f"scenarios: expected array, actual={type(rows).__name__}"
        return [
            _atom("result_evidence", name, False, detail)
            for name in ("unit_identifiers", "request_fields", "retry_fields", "control_fields")
        ]
    else:
        for unit_index, unit in enumerate(rows):
            if not isinstance(unit, dict):
                unit_issue = f"scenarios[{unit_index}]: expected object"
                break
            if not isinstance(unit.get("scenario"), str) or not _integer(unit.get("variant")):
                unit_issue = f"scenarios[{unit_index}]: scenario must be string and variant must be integer"
                break
            requests = unit.get("requests")
            if not isinstance(requests, list):
                request_issue = f"scenarios[{unit_index}].requests: expected array"
            else:
                for index, request in enumerate(requests):
                    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
                        request_issue = f"scenarios[{unit_index}].requests[{index}]: expected exact fields"
                        break
                    if not isinstance(request.get("request_id"), str) or not _integer(request.get("generated_slot")):
                        request_issue = f"scenarios[{unit_index}].requests[{index}]: invalid id/slot types"
                        break
                    if request.get("fallback") not in ("none", "start"):
                        request_issue = f"scenarios[{unit_index}].requests[{index}].fallback: expected none/start"
                        break
                    paths = request.get("selected_paths")
                    if not isinstance(paths, list) or any(_path_key(path) is None for path in paths):
                        request_issue = (
                            f"scenarios[{unit_index}].requests[{index}].selected_paths: "
                            "expected exact path objects with public interface and replica IDs"
                        )
                        break
            retries = unit.get("retries")
            if not isinstance(retries, list):
                retry_issue = f"scenarios[{unit_index}].retries: expected array"
            else:
                for index, retry in enumerate(retries):
                    if not isinstance(retry, dict) or set(retry) != RETRY_FIELDS:
                        retry_issue = f"scenarios[{unit_index}].retries[{index}]: expected exact fields"
                        break
                    if not isinstance(retry.get("request_id"), str) or not _integer(retry.get("send_slot")):
                        retry_issue = f"scenarios[{unit_index}].retries[{index}]: invalid id/slot types"
                        break
                    interface_id = retry.get("interface_id")
                    replica_id = retry.get("replica_id")
                    if (
                        not isinstance(interface_id, str)
                        or interface_id not in INTERFACES
                        or not isinstance(replica_id, str)
                        or replica_id not in REPLICAS
                    ):
                        retry_issue = f"scenarios[{unit_index}].retries[{index}]: invalid interface/replica"
                        break
            controls = unit.get("controls")
            if not isinstance(controls, list):
                control_issue = f"scenarios[{unit_index}].controls: expected array"
            else:
                for index, control in enumerate(controls):
                    if not isinstance(control, dict) or set(control) != CONTROL_FIELDS:
                        control_issue = f"scenarios[{unit_index}].controls[{index}]: expected exact fields"
                        break
                    slot = control.get("slot")
                    target = control.get("target_velocity")
                    if not _integer(slot):
                        control_issue = f"scenarios[{unit_index}].controls[{index}].slot: expected integer"
                        break
                    if not isinstance(target, list) or len(target) != 2 or not all(_finite(value) for value in target):
                        control_issue = f"scenarios[{unit_index}].controls[{index}].target_velocity: expected finite 2-vector"
                        break
    return [
        _atom("result_evidence", "unit_identifiers", unit_issue is None, unit_issue or "ok"),
        _atom("result_evidence", "request_fields", request_issue is None, request_issue or "ok"),
        _atom("result_evidence", "retry_fields", retry_issue is None, retry_issue or "ok"),
        _atom("result_evidence", "control_fields", control_issue is None, control_issue or "ok"),
    ]


def _index_records(data: Any) -> dict[tuple[str, int], Any]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return {}
    indexed: dict[tuple[str, int], Any] = {}
    for row in data["scenarios"]:
        if not isinstance(row, dict):
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            hash(key)
        except TypeError:
            continue
        indexed.setdefault(key, row)
    return indexed


# Compatibility alias for older local probes; the normative entry point is
# `_index_records`.
_index = _index_records


def _record_coverage_defects(
    rows: Any,
    field: str,
    expected_values: Sequence[int] | range,
    path: str,
) -> list[str]:
    if not isinstance(rows, list):
        return [f"{path}: expected array for coverage"]
    expected = tuple(expected_values)
    counts: dict[Any, int] = {}
    defects: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            defects.append(f"{path}[{index}]: expected object for coverage key")
            continue
        value = row.get(field)
        try:
            counts[value] = counts.get(value, 0) + 1
        except TypeError:
            defects.append(f"{path}[{index}].{field}: unhashable value={_brief(value)}")
    for value in expected:
        count = counts.get(value, 0)
        if count == 0:
            defects.append(f"{path}: missing {field}={value}")
        elif count > 1:
            defects.extend(
                [f"{path}: duplicate {field}={value}"] * (count - 1)
            )
    expected_set = set(expected)
    defects.extend(
        f"{path}: unknown {field}={_brief(value)}"
        for value in counts
        if value not in expected_set
    )
    return defects


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return 0.0, ["scenarios is not an array"]
    counts: dict[Any, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            defects.append("non-object scenario row")
            continue
        key = (row.get("scenario"), row.get("variant"))
        try:
            counts[key] = counts.get(key, 0) + 1
        except TypeError:
            defects.append(f"unhashable scenario key={_brief(key)}")
    for key in SCENARIO_KEYS:
        count = counts.get(key, 0)
        if count == 0:
            defects.append(f"missing scenario={key[0]}|variant={key[1]}")
        elif count > 1:
            defects.extend([f"duplicate scenario={_brief(key)}"] * (count - 1))
    defects.extend(f"unknown scenario key={_brief(key)}" for key in counts if key not in SCENARIO_KEYS)
    for key in SCENARIO_KEYS:
        matching = [
            row
            for row in rows
            if isinstance(row, dict)
            and (row.get("scenario"), row.get("variant")) == key
        ]
        if len(matching) != 1:
            continue
        unit = matching[0]
        defects.extend(
            _record_coverage_defects(
                unit.get("requests"),
                "generated_slot",
                REQUEST_SLOTS(key[1]),
                f"scenario={key[0]}|variant={key[1]}.requests",
            )
        )
        defects.extend(
            _record_coverage_defects(
                unit.get("controls"),
                "slot",
                range(HORIZON),
                f"scenario={key[0]}|variant={key[1]}.controls",
            )
        )
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _norm(a: tuple[float, float]) -> float:
    return math.hypot(a[0], a[1])


def _clip_vec(a: tuple[float, float], limit: float) -> tuple[float, float]:
    size = _norm(a)
    return a if size <= limit or size <= EPS else (a[0] * limit / size, a[1] * limit / size)


def _target(name: str, variant: int, boundary: int) -> tuple[float, float]:
    scene = SCENARIOS[name]
    return (
        min(14.0, max(0.0, scene["target"][0] + boundary * scene["tv"][0])),
        min(10.0, max(0.0, scene["target"][1] + boundary * scene["tv"][1])),
    )


def _clearance(name: str, point: tuple[float, float]) -> float:
    result = min(point[0], 14.0 - point[0], point[1], 10.0 - point[1])
    for obstacle in SCENARIOS[name]["obstacles"]:
        if obstacle[0] == "circle":
            result = min(result, math.hypot(point[0] - obstacle[1], point[1] - obstacle[2]) - obstacle[3])
        else:
            _, x0, x1, y0, y1 = obstacle
            dx = max(x0 - point[0], 0.0, point[0] - x1)
            dy = max(y0 - point[1], 0.0, point[1] - y1)
            result = min(result, math.hypot(dx, dy) if dx or dy else -min(point[0] - x0, x1 - point[0], point[1] - y0, y1 - point[1]))
    return result


def _segment_circle(a: tuple[float, float], b: tuple[float, float], cx: float, cy: float, radius: float) -> bool:
    ab = _sub(b, a)
    denominator = ab[0] ** 2 + ab[1] ** 2
    fraction = 0.0 if denominator <= EPS else max(0.0, min(1.0, ((cx - a[0]) * ab[0] + (cy - a[1]) * ab[1]) / denominator))
    closest = _add(a, (fraction * ab[0], fraction * ab[1]))
    return _norm(_sub(closest, (cx, cy))) <= radius + EPS


def _segment_rect(a: tuple[float, float], b: tuple[float, float], x0: float, x1: float, y0: float, y1: float) -> bool:
    if (x0 <= a[0] <= x1 and y0 <= a[1] <= y1) or (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
        return True
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) <= EPS:
            if q < -EPS:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            low = max(low, ratio)
        else:
            high = min(high, ratio)
        if low - high > EPS:
            return False
    return low <= high + EPS and high >= -EPS and low <= 1.0 + EPS


def _occluded(name: str, start: tuple[float, float], target: tuple[float, float]) -> int:
    for obstacle in SCENARIOS[name]["obstacles"]:
        if obstacle[0] == "circle" and _segment_circle(start, target, obstacle[1], obstacle[2], obstacle[3]):
            return 1
        if obstacle[0] == "rect" and _segment_rect(start, target, obstacle[1], obstacle[2], obstacle[3], obstacle[4]):
            return 1
    return 0


def _loss_rate(name: str, distance_norm: float, path_rank: int) -> float:
    return 0.08 + 0.03 * distance_norm + 0.04 * path_rank + (0.10 if name == "deadline_tight" else 0.0)


def _loss_sample(name: str, variant: int, request_id: str, interface: str, replica: str, attempt: int) -> float:
    text = f"{name}|{variant}|{request_id}|{interface}|{replica}|a{attempt}|salt={17 * variant}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") / 2**64


def _interface_up(name: str, variant: int, interface: str, slot: int) -> bool:
    if name == "handover_gap" and interface == "I1" and 30 + 4 * variant <= slot <= 38 + 4 * variant:
        return False
    if name == "correlated_outage" and interface in ("I0", "I1") and 42 + 4 * variant <= slot <= 50 + 4 * variant:
        return False
    return True


def _service_time(name: str, variant: int, service_start: int) -> int:
    if name == "server_burst" and 24 + 4 * variant <= service_start <= 35 + 4 * variant:
        return 4
    return 2


def _response_estimate(
    name: str,
    variant: int,
    generated_slot: int,
    interface_id: str,
    replica_id: str,
    attempt: int,
) -> list[float]:
    target = _target(name, variant, generated_slot)
    request_index = (generated_slot - variant) // 3
    interface = int(interface_id[1:])
    replica = int(replica_id[1:])
    phase = 2.0 * math.pi * ((request_index + 3 * interface + 5 * replica + 7 * attempt + variant) % 16) / 16.0
    return [target[0] + 0.08 * math.cos(phase), target[1] + 0.08 * math.sin(phase)]


def _fallback_estimate(
    name: str, variant: int, generated_slot: int
) -> list[float]:
    target = _target(name, variant, generated_slot)
    request_index = (generated_slot - variant) // 3
    angle = 2.0 * math.pi * ((request_index + variant) % 8) / 8.0
    error = 0.35 + 0.04 * 2
    return [
        min(14.0, max(0.0, target[0] + error * math.cos(angle))),
        min(10.0, max(0.0, target[1] + error * math.sin(angle))),
    ]


def _observation(name: str, variant: int, slot: int, position: tuple[float, float], request_index: int) -> dict[str, Any]:
    target = _target(name, variant, slot)
    distance = min(1.0, _norm(_sub(position, target)) / 3.0)
    occluded = _occluded(name, position, target)
    quality = max(0.0, min(1.0, 1.0 - 0.12 * distance - 0.15 * occluded))
    angle = 2.0 * math.pi * ((request_index + 2 * variant) % 12) / 12.0
    error = 1.20 + 0.80 * occluded + 0.40 * distance
    estimate = [
        min(14.0, max(0.0, target[0] + error * math.cos(angle))),
        min(10.0, max(0.0, target[1] + error * math.sin(angle))),
    ]
    return {
        "observation_id": f"obs={name}|variant={variant}|request={request_index}",
        "request_id": f"scenario={name}|variant={variant}|request={request_index}",
        "generated_slot": slot,
        "estimate": estimate,
        "quality": quality,
    }


def _requests_by_slot(unit: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    rows = unit.get("requests", [])
    if not isinstance(rows, list):
        return result
    for request in rows:
        if isinstance(request, Mapping) and _integer(request.get("generated_slot")):
            result.setdefault(int(request["generated_slot"]), request)
    return result


def _controls_by_slot(unit: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    rows = unit.get("controls", [])
    if not isinstance(rows, list):
        return result
    for control in rows:
        if isinstance(control, Mapping) and _integer(control.get("slot")):
            result.setdefault(int(control["slot"]), control)
    return result


def _replay(name: str, variant: int, unit: Mapping[str, Any]) -> dict[str, Any]:
    scene = SCENARIOS[name]
    request_slots = REQUEST_SLOTS(variant)
    requests = _requests_by_slot(unit)
    controls = _controls_by_slot(unit)
    retries_by_slot: dict[int, list[Mapping[str, Any]]] = {}
    retry_rows = unit.get("retries", [])
    if isinstance(retry_rows, list):
        for retry in retry_rows:
            if isinstance(retry, Mapping) and _integer(retry.get("send_slot")):
                retries_by_slot.setdefault(int(retry["send_slot"]), []).append(retry)

    position = tuple(scene["robot"])
    velocity = (0.0, 0.0)
    previous_target = (0.0, 0.0)
    interface_occupancy = {key: [] for key in INTERFACES}
    replica_occupancy = {key: [] for key in REPLICAS}
    admitted_bits = 0
    first_response: dict[str, str] = {}
    response_events: dict[int, list[dict[str, Any]]] = {}
    fallback_events: dict[int, list[dict[str, Any]]] = {}
    fallback_energy = 0.0
    attempt_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    fallback_records: list[dict[str, Any]] = []
    queue_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    mission_records: list[dict[str, Any]] = []
    observation_records: list[dict[str, Any]] = []
    robot_records: list[dict[str, Any]] = [
        {"boundary": 0, "position": list(position), "velocity": [0.0, 0.0]}
    ]
    risk_steps = 0
    collision_steps = 0
    motion_energy = 0.0
    legality = {key: [] for key in ATOM_SHARES["decision_legality"]}
    replay_defects = {key: [] for key in ATOM_SHARES["replay_validity"]}

    for slot in range(HORIZON):
        # Public event order: generate, reveal, retire, expose state, then act.
        if slot in request_slots:
            request_index = request_slots.index(slot)
            observation_records.append(_observation(name, variant, slot, position, request_index))

        for event in sorted(response_events.pop(slot, []), key=lambda item: item["attempt_id"]):
            request_id = event["request_id"]
            timely = event["completion_slot"] <= event["deadline"]
            accepted = timely and request_id not in first_response
            response_id = f"response={event['attempt_id']}"
            response_records.append(
                {
                    "response_id": response_id,
                    "request_id": request_id,
                    "attempt_id": event["attempt_id"],
                    "arrival_slot": event["completion_slot"],
                    "deadline": event["deadline"],
                    "age": event["completion_slot"] - event["generated_slot"],
                    "accepted": accepted,
                    "status": "accepted" if accepted else "late_or_duplicate",
                    "estimate": event["estimate"],
                }
            )
            if accepted:
                first_response[request_id] = response_id
        for values in interface_occupancy.values():
            values[:] = [release for release in values if release > slot]
        for values in replica_occupancy.values():
            values[:] = [release for release in values if release > slot]
        fallback_records.extend(fallback_events.pop(slot, []))

        queue_records.append(
            {
                "slot": slot,
                "interface_lengths": [len(interface_occupancy[key]) for key in ("I0", "I1", "I2")],
                "replica_lengths": [len(replica_occupancy[key]) for key in ("C0", "C1", "C2")],
            }
        )
        raw_control = controls.get(slot, {})
        raw_target = raw_control.get("target_velocity", [0.0, 0.0]) if isinstance(raw_control, Mapping) else [0.0, 0.0]
        if (
            isinstance(raw_target, list)
            and len(raw_target) == 2
            and all(_finite(value) for value in raw_target)
        ):
            world_target = float(raw_target[0]), float(raw_target[1])
        else:
            world_target = (0.0, 0.0)
        speed_norm = _norm(world_target)
        if speed_norm > MAX_SPEED + EPS:
            legality["decision_legality.motion"].append(
                f"slot={slot}: speed exceeds bound={MAX_SPEED:.9g} (norm={speed_norm:.9g}) "
                f"with tolerance={EPS:.9g}"
            )
        slew_norm = _norm(_sub(world_target, previous_target))
        if slew_norm > TARGET_SLEW + EPS:
            legality["decision_legality.motion"].append(
                f"slot={slot}: slew exceeds bound={TARGET_SLEW:.9g} (norm={slew_norm:.9g}) "
                f"with tolerance={EPS:.9g}"
            )

        attempts: list[tuple[Mapping[str, Any], int, tuple[str, str], int]] = []
        request = requests.get(slot, {})
        if slot in request_slots and isinstance(request, Mapping):
            if request.get("fallback") == "start":
                fallback_energy += FALLBACK_COST
                completion = slot + 2
                if completion <= HORIZON:
                    fallback_events.setdefault(completion, []).append(
                        {
                            "request_id": _safe_text(request.get("request_id")),
                            "generated_slot": slot,
                            "completion_slot": completion,
                            "estimate": _fallback_estimate(name, variant, slot),
                        }
                    )
            selected = request.get("selected_paths", [])
            if isinstance(selected, list):
                for rank, path in enumerate(selected):
                    path_key = _path_key(path)
                    if path_key is not None:
                        attempts.append((request, 0, path_key, rank))
        for retry in retries_by_slot.get(slot, []):
            request_id = _safe_text(retry.get("request_id"))
            request_slot = next(
                (candidate_slot for candidate_slot, candidate in requests.items() if str(candidate.get("request_id")) == request_id),
                None,
            )
            original = requests.get(request_slot, {}) if request_slot is not None else {}
            path_key = str(retry.get("interface_id")), str(retry.get("replica_id"))
            selected = original.get("selected_paths", []) if isinstance(original, Mapping) else []
            keys = [_path_key(path) for path in selected] if isinstance(selected, list) else []
            rank = keys.index(path_key) if path_key in keys else 0
            if isinstance(original, Mapping):
                attempts.append((original, 1, path_key, rank))

        attempts.sort(
            key=lambda item: (
                int(item[0].get("generated_slot", HORIZON + 1)) + 6,
                int(item[0].get("generated_slot", HORIZON + 1)),
                item[3], item[2][0], item[2][1], item[1],
            )
        )
        base_interfaces = {key: len(values) for key, values in interface_occupancy.items()}
        base_replicas = {key: len(values) for key, values in replica_occupancy.items()}
        reserved_interfaces = dict(base_interfaces)
        reserved_replicas = dict(base_replicas)
        for request, attempt, path_key, rank in attempts:
            request_id = _safe_text(request.get("request_id"))
            generated_slot = int(request.get("generated_slot", HORIZON + 1)) if _integer(request.get("generated_slot")) else HORIZON + 1
            deadline = generated_slot + 6
            interface, replica = path_key
            distance = _norm(_sub(position, INTERFACES.get(interface, (0.0, 0.0))))
            distance_norm = min(1.0, distance / 14.0)
            loss_rate = _loss_rate(name, distance_norm, rank)
            sample = _loss_sample(name, variant, request_id, interface, replica, attempt)
            up = interface in INTERFACES and _interface_up(name, variant, interface, slot)
            recovered = bool(up and not _interface_up(name, variant, interface, slot - 1))
            handover = int(recovered)
            queue_wait = base_interfaces.get(interface, 99) + base_replicas.get(replica, 99)
            path_delay = 1 + queue_wait + math.floor(2 * distance / 5) + handover
            service_start = slot + path_delay
            completion = service_start + _service_time(name, variant, service_start)
            budget_ok = admitted_bits + PATH_BITS <= UPLINK_BUDGET
            capacity_ok = reserved_interfaces.get(interface, 99) < 4 and reserved_replicas.get(replica, 99) < 6
            admitted = bool(up and budget_ok and capacity_ok)
            if admitted:
                admitted_bits += PATH_BITS
                reserved_interfaces[interface] += 1
                reserved_replicas[replica] += 1
                interface_occupancy[interface].append(service_start)
                replica_occupancy[replica].append(completion)
                status = "scheduled" if sample >= loss_rate else "lost"
            else:
                status = "interface_down" if not up else "budget_exhausted" if not budget_ok else "queue_rejected"
            attempt_id = f"{request_id}|{interface}|{replica}|a{attempt}"
            attempt_records.append(
                {
                    "attempt_id": attempt_id, "request_id": request_id, "attempt": attempt,
                    "interface_id": interface, "replica_id": replica, "send_slot": slot,
                    "reveal_slot": slot + 1,
                    "path_rank": rank, "queue_wait": queue_wait, "handover_penalty": handover,
                    "loss_rate": loss_rate, "sample_u": sample, "admitted": admitted,
                    "status": status, "service_start_slot": service_start if admitted else None,
                    "completion_slot": completion if admitted else None,
                }
            )
            if admitted and status == "scheduled":
                response_events.setdefault(completion, []).append(
                    {
                        "attempt_id": attempt_id, "request_id": request_id,
                        "generated_slot": generated_slot, "deadline": deadline,
                        "completion_slot": completion,
                        "estimate": _response_estimate(
                            name,
                            variant,
                            generated_slot,
                            interface,
                            replica,
                            attempt,
                        ),
                    }
                )

        actual = _clip_vec(
            _add((0.55 * velocity[0], 0.55 * velocity[1]), (0.45 * world_target[0], 0.45 * world_target[1])),
            MAX_SPEED,
        )
        proposed = _add(position, (DT * actual[0], DT * actual[1]))
        clearance = _clearance(name, proposed)
        collision = clearance < SAFETY_CLEARANCE - EPS or proposed[0] < -EPS or proposed[0] > 14.0 + EPS or proposed[1] < -EPS or proposed[1] > 10.0 + EPS
        if collision:
            next_position = position
            actual = (0.0, 0.0)
            collision_steps += 1
        else:
            next_position = proposed
        risk = clearance < RISK_CLEARANCE - EPS
        risk_steps += int(risk)
        slot_energy = 0.5 * _norm(actual) ** 2
        motion_energy += slot_energy
        control_records.append(
            {
                "slot": slot, "target_velocity": list(world_target), "executed_velocity": list(actual),
                "position_before": list(position), "position_after": list(next_position), "motion_energy": slot_energy,
            }
        )
        target = _target(name, variant, slot + 1)
        mission_records.append(
            {
                "boundary": slot + 1, "target_position": list(target),
                "tracking_error": _norm(_sub(next_position, target)), "risk": bool(risk),
                "collision": bool(collision), "energy": motion_energy, "fallback_energy": fallback_energy,
            }
        )
        robot_records.append({"boundary": slot + 1, "position": list(next_position), "velocity": list(actual)})
        position, velocity, previous_target = next_position, actual, world_target

    for completion in (HORIZON,):
        for event in sorted(response_events.get(completion, []), key=lambda item: item["attempt_id"]):
            request_id = event["request_id"]
            accepted = event["completion_slot"] <= event["deadline"] and request_id not in first_response
            response_id = f"response={event['attempt_id']}"
            response_records.append(
                {
                    "response_id": response_id, "request_id": request_id, "attempt_id": event["attempt_id"],
                    "arrival_slot": completion, "deadline": event["deadline"],
                    "age": completion - event["generated_slot"],
                    "accepted": accepted, "status": "accepted" if accepted else "late_or_duplicate",
                    "estimate": event["estimate"],
                }
            )
            if accepted:
                first_response[request_id] = response_id
        fallback_records.extend(fallback_events.get(completion, []))

    if len(observation_records) != REQUEST_COUNT or any(not _finite(value) for row in observation_records for value in row["estimate"]):
        replay_defects["replay_validity.observations"].append("observation replay is incomplete or non-finite")
    if len(queue_records) != HORIZON:
        replay_defects["replay_validity.network"].append("queue evidence length mismatch")
    if any(
        max(row["interface_lengths"], default=0) > 4
        or max(row["replica_lengths"], default=0) > 6
        for row in queue_records
    ) or admitted_bits > UPLINK_BUDGET:
        replay_defects["replay_validity.network"].append(
            "reservation capacity or uplink budget invariant failed"
        )
    expected_fallbacks = sum(
        1
        for request in requests.values()
        if request.get("fallback") == "start"
        and int(request.get("generated_slot", HORIZON + 1)) + 2 <= HORIZON
    )
    if len(fallback_records) != expected_fallbacks or any(
        not _finite(value)
        for row in fallback_records
        for value in row["estimate"]
    ):
        replay_defects["replay_validity.fallbacks"].append(
            "fallback completion replay is incomplete or non-finite"
        )
    if len(control_records) != HORIZON or any(not _finite(value) for row in control_records for value in row["position_after"] + row["executed_velocity"]):
        replay_defects["replay_validity.motion"].append("motion replay is incomplete or non-finite")
    if len(robot_records) != HORIZON + 1 or len(mission_records) != HORIZON or any(not _finite(row["tracking_error"]) for row in mission_records):
        replay_defects["replay_validity.mission"].append("mission replay is incomplete or non-finite")
    if any(row["completion_slot"] is not None and row["completion_slot"] < row["service_start_slot"] for row in attempt_records if row["admitted"]):
        replay_defects["replay_validity.network"].append("completion precedes service start")
    accepted_counts: dict[str, int] = {}
    for row in response_records:
        if row["accepted"]:
            accepted_counts[row["request_id"]] = accepted_counts.get(row["request_id"], 0) + 1
    if (
        len(first_response) > REQUEST_COUNT
        or any(count != 1 for count in accepted_counts.values())
        or any(
            not all(_finite(value) for value in row["estimate"])
            or row["arrival_slot"] > HORIZON
            or row["age"] < 0
            or (row["accepted"] and row["arrival_slot"] > min(row["deadline"], HORIZON))
            for row in response_records
        )
    ):
        replay_defects["replay_validity.responses"].append(
            "response finiteness, terminal, or first-acceptance invariant failed"
        )

    initial_target = _target(name, variant, 0)
    expected = {
        "scenario": name, "variant": variant, "attempts": attempt_records,
        "responses": response_records, "fallbacks": fallback_records,
        "queues": queue_records, "controls": control_records,
        "robot": robot_records, "observations": observation_records,
        "mission": [
            {
                "boundary": 0, "target_position": list(initial_target),
                "tracking_error": _norm(_sub(tuple(scene["robot"]), initial_target)),
                "risk": False, "collision": False, "energy": 0.0, "fallback_energy": 0.0,
            }
        ] + mission_records,
    }
    return {
        "expected": expected, "first_response": first_response, "admitted_bits": admitted_bits,
        "risk_steps": risk_steps, "collision_steps": collision_steps,
        "motion_energy": motion_energy, "fallback_energy": fallback_energy,
        "legality": legality, "replay_defects": replay_defects,
    }


def _metrics_for(sim: dict[str, Any]) -> dict[str, float]:
    expected = sim["expected"]
    mission = expected["mission"]
    deadline = int(SCENARIOS[expected["scenario"]]["deadline"])
    horizon = min(HORIZON, deadline)
    errors = [float(row["tracking_error"]) for row in mission[: horizon + 1]]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0
    final_error = float(mission[horizon]["tracking_error"])
    initial_error = float(mission[0]["tracking_error"])
    primary_1 = 0.6 * max(0.0, min(1.0, 1.0 - rmse / initial_error)) + 0.4 * max(0.0, min(1.0, 1.0 - final_error / initial_error))
    timely = [row for row in expected["responses"] if row["accepted"]]
    timely_ratio = len(sim["first_response"]) / REQUEST_COUNT
    freshness = sum(max(0.0, 1.0 - float(row["age"]) / 6.0) for row in timely) / len(timely) if timely else 0.0
    primary_2 = 0.7 * timely_ratio + 0.3 * freshness
    primary_3 = (
        0.7 * max(0.0, min(1.0, 1.0 - sim["risk_steps"] / HORIZON))
        + 0.3 * max(0.0, min(1.0, 1.0 - sim["motion_energy"] / (HORIZON * 0.5 * MAX_SPEED**2)))
    )
    admitted_attempts = [row for row in expected["attempts"] if row["admitted"]]
    sent_bits = len(admitted_attempts) * PATH_BITS
    useful_bits = len(sim["first_response"]) * PATH_BITS
    auxiliary_1 = 0.6 * useful_bits / max(sent_bits, 1) + 0.4 * max(0.0, min(1.0, 1.0 - sent_bits / UPLINK_BUDGET))
    mean_queue = sum(float(row["queue_wait"]) for row in admitted_attempts) / len(admitted_attempts) if admitted_attempts else 0.0
    auxiliary_2 = (
        0.5 * max(0.0, min(1.0, 1.0 - mean_queue / HORIZON))
        + 0.5 * max(0.0, min(1.0, 1.0 - sim["fallback_energy"] / 100.0))
    )
    return {
        "primary_metrics_1": primary_1,
        "primary_metrics_2": primary_2,
        "primary_metrics_3": primary_3,
        "auxiliary_metrics_1": auxiliary_1,
        "auxiliary_metrics_2": auxiliary_2,
    }


def _metrics(replayed: Mapping[tuple[str, int], Any]) -> dict[str, float]:
    """Aggregate the independently replayed unit metrics."""
    missing = [key for key in SCENARIO_KEYS if key not in replayed]
    if missing:
        raise ValueError(f"metric replay missing canonical units: {_brief(missing)}")
    unit_metrics = [_metrics_for(replayed[key]) for key in SCENARIO_KEYS]
    metrics = {
        name: sum(row[name] for row in unit_metrics) / len(unit_metrics)
        for name in METRIC_KEYS
    }
    if any(
        not _finite(value) or value < -EPS or value > 1.0 + EPS
        for value in metrics.values()
    ):
        raise ValueError(f"metric outside [0,1]: {_brief(metrics)}")
    return metrics


def _gated_metrics(
    scores: Mapping[str, float],
    replayed: Mapping[tuple[str, int], Any],
) -> tuple[bool, dict[str, float]]:
    gate = all(scores.get(key, 0.0) >= 1.0 - EPS for key in GATE_CHECKS)
    if not gate:
        return False, {name: 0.0 for name in METRIC_KEYS}
    return True, _metrics(replayed)


def _legality(index: dict[tuple[str, int], Any]) -> list[AtomVerdict]:
    failures = {key: [] for key in ATOM_SHARES["decision_legality"]}
    for (name, variant), unit in index.items():
        if name not in SCENARIOS or variant not in range(3) or not isinstance(unit, Mapping):
            continue
        try:
            sim = _replay(name, variant, unit)
        except Exception as exc:
            detail = f"{name}|{variant}: legality replay raised {type(exc).__name__}: {exc}"
            for atom_id in failures:
                failures[atom_id].append(detail)
            continue
        request_rows = unit.get("requests", [])
        safe_request_rows = request_rows if isinstance(request_rows, list) else []
        requests = {
            request.get("request_id"): request
            for request in safe_request_rows
            if isinstance(request, Mapping)
            and isinstance(request.get("request_id"), str)
        }
        request_slots = REQUEST_SLOTS(variant)
        seen_requests: set[str] = set()
        if isinstance(request_rows, list):
            for request in request_rows:
                if not isinstance(request, Mapping):
                    continue
                request_id = _safe_text(request.get("request_id"))
                generated = request.get("generated_slot")
                if not _integer(generated) or generated not in request_slots:
                    continue
                expected_id = f"scenario={name}|variant={variant}|request={request_slots.index(generated)}"
                if request_id != expected_id or request_id in seen_requests:
                    failures["decision_legality.request_references"].append(f"{name}|{variant}:{request_id}: request id mismatch or duplicate")
                seen_requests.add(request_id)
                paths = request.get("selected_paths")
                keys = [_path_key(path) for path in paths] if isinstance(paths, list) else []
                if isinstance(paths, list) and len(paths) > PATH_LIMIT:
                    failures["decision_legality.path_selection"].append(f"{name}|{variant}:{request_id}: at most two distinct valid paths required")
                valid_keys = [key for key in keys if key is not None]
                if len(valid_keys) != len(set(valid_keys)):
                    failures["decision_legality.path_selection"].append(f"{name}|{variant}:{request_id}: selected path pairs must be distinct")
        retry_rows = unit.get("retries", [])
        seen_retries: set[tuple[str, str, str]] = set()
        if isinstance(retry_rows, list):
            for retry in retry_rows:
                if not isinstance(retry, Mapping):
                    continue
                request_id = _safe_text(retry.get("request_id"))
                request = requests.get(request_id)
                path = retry.get("interface_id"), retry.get("replica_id")
                send_slot = retry.get("send_slot")
                retry_key = request_id, _safe_text(path[0]), _safe_text(path[1])
                if retry_key in seen_retries:
                    failures["decision_legality.retry_window"].append(f"{name}|{variant}:{request_id}: duplicate retry")
                seen_retries.add(retry_key)
                if request is None:
                    failures["decision_legality.retry_window"].append(f"{name}|{variant}:{request_id}: retry references unknown request")
                    continue
                selected = [_path_key(item) for item in request.get("selected_paths", [])] if isinstance(request.get("selected_paths"), list) else []
                generated = request.get("generated_slot")
                path_fields_valid = (
                    isinstance(path[0], str)
                    and path[0] in INTERFACES
                    and isinstance(path[1], str)
                    and path[1] in REPLICAS
                )
                if path_fields_valid and path not in selected:
                    failures["decision_legality.retry_window"].append(f"{name}|{variant}:{request_id}: retry path was not selected by attempt 0")
                if _integer(send_slot) and _integer(generated):
                    expected_send = generated + 1
                    if send_slot != expected_send or not 0 <= send_slot < HORIZON:
                        failures["decision_legality.retry_window"].append(
                            f"{name}|{variant}:{request_id}: retry send_slot actual={_brief(send_slot)} "
                            f"expected={_brief(expected_send)} within action slots 0..{HORIZON - 1} "
                            f"tolerance={EPS}"
                        )
                if not path_fields_valid or not _integer(send_slot) or not _integer(generated):
                    continue
                attempt_zero = next(
                    (
                        row for row in sim["expected"]["attempts"]
                        if row["request_id"] == request_id
                        and row["attempt"] == 0
                        and (row["interface_id"], row["replica_id"]) == path
                    ),
                    None,
                )
                if attempt_zero is None or attempt_zero["status"] == "scheduled":
                    failures["decision_legality.retry_window"].append(f"{name}|{variant}:{request_id}: retry requires visible failed attempt")
        failures["decision_legality.motion"].extend(sim["legality"]["decision_legality.motion"])
    return [
        _atom("decision_legality", key.rsplit(".", 1)[1], not failures[key], "; ".join(failures[key][:5]) if failures[key] else "ok")
        for key in ATOM_SHARES["decision_legality"]
    ]


def _replay_atoms(
    replayed: Mapping[tuple[str, int], Any], replay_errors: Sequence[str] = ()
) -> list[AtomVerdict]:
    failures = {key: [] for key in ATOM_SHARES["replay_validity"]}
    for detail in replay_errors:
        for atom in failures:
            failures[atom].append(detail)
    for key in SCENARIO_KEYS:
        sim = replayed.get(key)
        if not isinstance(sim, Mapping):
            for atom in failures:
                failures[atom].append(f"{key}: no replay state")
            continue
        for atom, defects in sim["replay_defects"].items():
            failures[atom].extend(f"{key}: {detail}" for detail in defects)
    return [
        _atom("replay_validity", key.rsplit(".", 1)[1], not failures[key], "; ".join(failures[key][:5]) if failures[key] else "ok")
        for key in ATOM_SHARES["replay_validity"]
    ]


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


async def _build_state(ws: Workspace) -> EvalState:
    cache_key = str(ws.root)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    state = EvalState()
    result_path = Path(ws.root) / "results.json"
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
    except Exception as exc:
        state.official_message = f"old results.json removal failed: {type(exc).__name__}: {exc}"
        state.scores = {key: 0.0 for key in GATE_CHECKS}
        state.diagnostics = {key: state.official_message for key in GATE_CHECKS}
        _CACHE[cache_key] = state
        return state

    state.run_started = True
    started = time.monotonic()
    try:
        result = await ws.run("python run.py", check=False, timeout=600)
    except Exception as exc:
        state.run_finished = True
        state.elapsed_s = time.monotonic() - started
        state.timed_out = "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
        state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.scores = {key: 0.0 for key in GATE_CHECKS}
        state.diagnostics = {key: state.official_message for key in GATE_CHECKS}
        _CACHE[cache_key] = state
        return state

    state.run_finished = True
    state.elapsed_s = time.monotonic() - started
    state.exit_code = getattr(result, "exit_code", None)
    state.result_file_is_new_regular = result_path.is_file() and not result_path.is_symlink()
    state.official_ok = (
        getattr(removed, "exit_code", None) == 0
        and state.exit_code == 0
        and state.result_file_is_new_regular
        and state.elapsed_s <= 600.0 + EPS
    )
    state.official_message = (
        f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
        f"new_regular_results={state.result_file_is_new_regular}"
    )
    state.scores["official_run"] = float(state.official_ok)
    state.diagnostics["official_run"] = state.official_message
    if not state.official_ok:
        for key in GATE_CHECKS:
            if key == "official_run":
                continue
            state.scores[key] = 0.0
            state.diagnostics[key] = f"official_run failed; {state.official_message}"
        _CACHE[cache_key] = state
        return state

    try:
        state.data = _strict_json(result_path.read_text(encoding="utf-8"))
        state.parse_ok = True
    except Exception as exc:
        state.diagnostics["result_json"] = (
            f"strict parse failed: {type(exc).__name__}: {exc}"
        )

    shape = _shape_atoms(state.data, state.parse_ok)
    try:
        evidence = _evidence_atoms(state.data)
    except Exception as exc:
        evidence = _failed_atoms(
            "result_evidence", f"evidence check raised {type(exc).__name__}: {exc}"
        )
    try:
        state.records = _index_records(state.data)
        state.scenarios = state.records
    except Exception as exc:
        state.records = {}
        state.scenarios = {}
        index_error = f"record index raised {type(exc).__name__}: {exc}"
    else:
        index_error = ""
    try:
        coverage, coverage_defects = _coverage(state.data)
    except Exception as exc:
        coverage, coverage_defects = 0.0, [
            f"coverage check raised {type(exc).__name__}: {exc}"
        ]
    try:
        legality = _legality(state.records)
    except Exception as exc:
        legality = _failed_atoms(
            "decision_legality", f"legality check raised {type(exc).__name__}: {exc}"
        )
    state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
    state.scores["result_evidence"] = _score_atoms(
        evidence, ATOM_SHARES["result_evidence"]
    )
    state.scores["canonical_coverage"] = coverage
    state.scores["decision_legality"] = _score_atoms(
        legality, ATOM_SHARES["decision_legality"]
    )
    replay_errors: list[str] = []
    for key in SCENARIO_KEYS:
        if key not in state.records:
            continue
        try:
            state.replayed[key] = _replay(key[0], key[1], state.records[key])
        except Exception as exc:
            replay_errors.append(
                f"scenario={key[0]}|variant={key[1]}: replay raised "
                f"{type(exc).__name__}: {exc}"
            )
    try:
        replay_atoms = _replay_atoms(state.replayed, replay_errors)
    except Exception as exc:
        replay_atoms = _failed_atoms(
            "replay_validity", f"replay atom check raised {type(exc).__name__}: {exc}"
        )
    state.scores["replay_validity"] = _score_atoms(
        replay_atoms, ATOM_SHARES["replay_validity"]
    )
    state.diagnostics["result_json"] = state.diagnostics.get(
        "result_json", _diagnose(shape)
    )
    state.diagnostics["result_evidence"] = _diagnose(evidence)
    state.diagnostics["canonical_coverage"] = (
        f"defect_count={len(coverage_defects)}"
        + ("; " + " | ".join(coverage_defects[:12]) if coverage_defects else "")
    )
    state.diagnostics["decision_legality"] = _diagnose(legality)
    state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
    if index_error:
        state.diagnostics["result_json"] += f"; {index_error}"
    try:
        state.validity_gate, state.metrics = _gated_metrics(
            state.scores, state.replayed
        )
    except Exception as exc:
        message = f"metric evaluation failed: {type(exc).__name__}: {exc}"
        state.scores["replay_validity"] = 0.0
        state.diagnostics["replay_validity"] = (
            f"failed_atoms=[replay_validity.metrics]; {message}"
        )
        state.validity_gate = False
        state.metrics = {name: 0.0 for name in METRIC_KEYS}
    _CACHE[cache_key] = state
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


for _check_id in GATE_CHECKS:
    if _check_id == "official_run":
        continue

    async def _gate(ws: Workspace, ident: str = _check_id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    globals()[_check_id] = check(id=_check_id)(_gate)


for _metric_id in METRIC_KEYS:

    async def _metric_check(ws: Workspace, ident: str = _metric_id) -> tuple[float, str]:
        state = await _build_state(ws)
        value = state.metrics.get(ident, 0.0) if state.validity_gate else 0.0
        failed = [key for key in GATE_CHECKS if state.scores.get(key, 0.0) < 1.0 - EPS]
        return value, (
            f"{ident}: replayed value={value:.9f}; gate={state.validity_gate}; "
            f"failed_gate_checks={failed}"
        )

    globals()[_metric_id] = check(id=_metric_id)(_metric_check)


del _gate, _metric_check
