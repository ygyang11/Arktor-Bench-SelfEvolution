"""Independent replay grader for process-state-aware MEC welding scheduling."""

from __future__ import annotations

import asyncio
import json
import math
import re
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
H = 240
ROBOTS = ("R0", "R1", "R2", "R3")
WELDS = tuple(f"P{i}" for i in range(6))
KINDS = ("scene", "start", "verify")
WORKERS = ("L0", "L1", "L2", "L3", "M0", "M1", "C0")
SCENARIOS = (
    "staggered_cells",
    "simultaneous_preheat",
    "uplink_jitter",
    "mec_worker_loss",
    "hardware_gap",
)
PLAN_KEYS = {
    "staggered_cells": (0, (0, 6, 12, 18), 68, 760.0),
    "simultaneous_preheat": (0, (0, 0, 0, 0), 60, 780.0),
    "uplink_jitter": (0, (0, 3, 6, 9), 64, 790.0),
    "mec_worker_loss": (0, (0, 4, 8, 12), 66, 800.0),
    "hardware_gap": (0, (0, 2, 4, 6), 62, 810.0),
}
REQUEST = {
    "scene": {
        "input": 2048,
        "output": 256,
        "offset": 22,
        "minimum": 0.80,
        "service": {"local": 12, "mec": 4, "cloud": 3},
        "quality": {"local": 0.84, "mec": 0.94, "cloud": 0.97},
        "energy": 6.0,
    },
    "start": {
        "input": 4096,
        "output": 384,
        "offset": 26,
        "minimum": 0.90,
        "service": {"local": 18, "mec": 6, "cloud": 4},
        "quality": {"local": 0.91, "mec": 0.96, "cloud": 0.99},
        "energy": 9.0,
    },
    "verify": {
        "input": 3072,
        "output": 256,
        "offset": 20,
        "minimum": 0.86,
        "service": {"local": 15, "mec": 5, "cloud": 3},
        "quality": {"local": 0.88, "mec": 0.95, "cloud": 0.98},
        "energy": 7.0,
    },
}
WORKER_CAPACITY = {"L0": 1, "L1": 1, "L2": 1, "L3": 1, "M0": 2, "M1": 2, "C0": 2}
CHANNEL_CAPACITY = 3
ROBOT_PACKET_CAPACITY = 1
COMMUNICATION_BUDGET = 32000
REMOTE_COMPUTE_BUDGET = 48
REPORT_SECTIONS = (
    "Problem Formulation",
    "Method Design",
    "Experimental Setup",
    "Results",
    "Research Analysis",
    "Conclusion and Limitations",
)
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.20,
        "result_json.scenarios": 0.25,
        "result_json.containers": 0.35,
    },
    "result_evidence": {
        "result_evidence.requests": 0.20,
        "result_evidence.attempts": 0.35,
        "result_evidence.cache": 0.20,
        "result_evidence.welds": 0.25,
    },
    "decision_legality": {
        "decision_legality.retries": 0.20,
        "decision_legality.placement": 0.20,
        "decision_legality.time_sequences": 0.20,
        "decision_legality.fallback_scope": 0.15,
        "decision_legality.dynamic_eligibility": 0.25,
    },
    "replay_validity": {
        "replay_validity.activation_deadlines": 0.25,
        "replay_validity.network_workers": 0.25,
        "replay_validity.process_workcells": 0.30,
        "replay_validity.resources": 0.20,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id = ident
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
        self.result_file_is_new_regular = False
        self.parse_ok = False
        self.data: Any = None
        self.records: dict[Any, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _integer(value: Any, lower: int, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper


def _attempt_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"retry_index", "start_slot", "path", "worker_id", "priority"}
        and _integer(value.get("retry_index"), 0, 2)
        and _integer(value.get("start_slot"), 0, H - 1)
        and isinstance(value.get("path"), str)
        and value["path"] in {"local", "mec", "cloud"}
        and isinstance(value.get("worker_id"), str)
        and value["worker_id"] in WORKERS
        and _integer(value.get("priority"), 0, 9)
    )


def _placement_consistent(robot: Any, path: Any, worker: Any) -> bool:
    if not isinstance(robot, str) or robot not in ROBOTS:
        return False
    if not isinstance(path, str) or not isinstance(worker, str):
        return False
    return (
        (path == "local" and worker == f"L{robot[1]}")
        or (path == "mec" and worker in {"M0", "M1"})
        or (path == "cloud" and worker == "C0")
    )


def _keys_error(value: Any, expected: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object, actual_type={type(value).__name__}"
    if set(value) != expected:
        return f"{path}: expected_keys={sorted(expected)!r}, actual_keys={sorted(value)!r}"
    return None


def _request_error(value: Any, path: str) -> str | None:
    error = _keys_error(
        value,
        {"robot_id", "weld_id", "request_type", "attempts", "cache_reuse_slots"},
        path,
    )
    if error:
        return error
    for field, domain in (
        ("robot_id", ROBOTS),
        ("weld_id", WELDS),
        ("request_type", KINDS),
    ):
        actual = value[field]
        if not isinstance(actual, str) or actual not in domain:
            return f"{path}.{field}: expected one of {domain!r}, actual={actual!r}"
    for field in ("attempts", "cache_reuse_slots"):
        if not isinstance(value[field], list):
            return f"{path}.{field}: expected array, actual_type={type(value[field]).__name__}"
    return None


def _attempt_error(value: Any, path: str) -> str | None:
    error = _keys_error(value, {"retry_index", "start_slot", "path", "worker_id", "priority"}, path)
    if error:
        return error
    for field, lower, upper in (
        ("retry_index", 0, 2),
        ("start_slot", 0, H - 1),
        ("priority", 0, 9),
    ):
        if not _integer(value[field], lower, upper):
            return (
                f"{path}.{field}: expected JSON integer in [{lower},{upper}], "
                f"actual={value[field]!r}"
            )
    if not isinstance(value["path"], str) or value["path"] not in {
        "local",
        "mec",
        "cloud",
    }:
        return f"{path}.path: expected local/mec/cloud, actual={value['path']!r}"
    if not isinstance(value["worker_id"], str) or value["worker_id"] not in WORKERS:
        return f"{path}.worker_id: expected one of {WORKERS!r}, actual={value['worker_id']!r}"
    return None


def _weld_error(value: Any, path: str) -> str | None:
    error = _keys_error(
        value, {"robot_id", "weld_id", "workcell_request_slots", "abandon_slot"}, path
    )
    if error:
        return error
    for field, domain in (("robot_id", ROBOTS), ("weld_id", WELDS)):
        actual = value[field]
        if not isinstance(actual, str) or actual not in domain:
            return f"{path}.{field}: expected one of {domain!r}, actual={actual!r}"
    slots = value["workcell_request_slots"]
    if not isinstance(slots, list):
        return f"{path}.workcell_request_slots: expected array, actual_type={type(slots).__name__}"
    for index, slot in enumerate(slots):
        if not _integer(slot, 0, H - 1):
            return (
                f"{path}.workcell_request_slots[{index}]: expected JSON integer in "
                f"[0,{H - 1}], actual={slot!r}"
            )
    abandon = value["abandon_slot"]
    if abandon is not None and not _integer(abandon, 0, H - 1):
        return (
            f"{path}.abandon_slot: expected null or JSON integer in [0,{H - 1}], actual={abandon!r}"
        )
    return None


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _result_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    verdict = {key: [True, "ok"] for key in ATOM_SHARES["result_json"]}
    if not parse_ok:
        return [
            AtomVerdict(key, False, "strict JSON parse failed", share)
            for key, share in ATOM_SHARES["result_json"].items()
        ]
    if not isinstance(data, dict) or set(data) != {"scenarios"}:
        actual = sorted(data) if isinstance(data, dict) else type(data).__name__
        verdict["result_json.root"] = [
            False,
            f"results.json: expected_keys=['scenarios'], actual={actual!r}",
        ]
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        actual_type = type(data.get("scenarios")).__name__ if isinstance(data, dict) else None
        verdict["result_json.scenarios"] = [
            False,
            f"results.json.scenarios: expected array, actual_type={actual_type}",
        ]
        verdict["result_json.containers"] = [
            False,
            f"results.json.scenarios[*]: unavailable because actual_type={actual_type}",
        ]
    else:
        for index, scenario in enumerate(data["scenarios"]):
            if not isinstance(scenario, dict) or set(scenario) != {
                "scenario",
                "request_decisions",
                "weld_decisions",
            }:
                actual = sorted(scenario) if isinstance(scenario, dict) else type(scenario).__name__
                verdict["result_json.containers"] = [
                    False,
                    f"scenarios[{index}]: expected keys scenario/request_decisions/"
                    f"weld_decisions, actual={actual!r}",
                ]
                break
            if not isinstance(scenario.get("request_decisions"), list) or not isinstance(
                scenario.get("weld_decisions"), list
            ):
                verdict["result_json.containers"] = [
                    False,
                    f"scenarios[{index}]: expected request_decisions and weld_decisions arrays, "
                    f"actual_types=({type(scenario.get('request_decisions')).__name__},"
                    f"{type(scenario.get('weld_decisions')).__name__})",
                ]
                break
    return [
        AtomVerdict(key, values[0], values[1], ATOM_SHARES["result_json"][key])
        for key, values in verdict.items()
    ]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    verdict = {key: [True, "ok"] for key in ATOM_SHARES["result_evidence"]}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return [
            AtomVerdict(key, False, "scenarios evidence unavailable", share)
            for key, share in ATOM_SHARES["result_evidence"].items()
        ]
    for si, scenario in enumerate(data["scenarios"]):
        if not isinstance(scenario, dict):
            for key in verdict:
                verdict[key] = [False, f"scenarios[{si}]: object unavailable"]
            continue
        requests = scenario.get("request_decisions")
        if not isinstance(requests, list):
            verdict["result_evidence.requests"] = [
                False,
                f"scenarios[{si}].request_decisions: array unavailable",
            ]
            verdict["result_evidence.attempts"] = [
                False,
                f"scenarios[{si}].request_decisions: attempt evidence unavailable",
            ]
            verdict["result_evidence.cache"] = [
                False,
                f"scenarios[{si}].request_decisions: cache evidence unavailable",
            ]
        else:
            for ri, request in enumerate(requests):
                path = f"scenarios[{si}].request_decisions[{ri}]"
                request_error = _request_error(request, path)
                if request_error:
                    verdict["result_evidence.requests"] = [False, request_error]
                    if not isinstance(request, dict) or not isinstance(
                        request.get("attempts"), list
                    ):
                        verdict["result_evidence.attempts"] = [
                            False,
                            f"{path}.attempts: array unavailable",
                        ]
                    if not isinstance(request, dict) or not isinstance(
                        request.get("cache_reuse_slots"), list
                    ):
                        verdict["result_evidence.cache"] = [
                            False,
                            f"{path}.cache_reuse_slots: array unavailable",
                        ]
                    continue
                attempts = request["attempts"]
                if len(attempts) > 3:
                    verdict["result_evidence.attempts"] = [
                        False,
                        f"{path}.attempts: expected length in [0,3], actual={len(attempts)}",
                    ]
                for ai, attempt in enumerate(attempts):
                    apath = f"{path}.attempts[{ai}]"
                    attempt_error = _attempt_error(attempt, apath)
                    if attempt_error:
                        verdict["result_evidence.attempts"] = [False, attempt_error]
                for ci, slot in enumerate(request["cache_reuse_slots"]):
                    if not _integer(slot, 0, H - 1):
                        verdict["result_evidence.cache"] = [
                            False,
                            f"{path}.cache_reuse_slots[{ci}]: expected JSON integer in "
                            f"[0,{H - 1}], actual={slot!r}",
                        ]
                        break
        welds = scenario.get("weld_decisions")
        if not isinstance(welds, list):
            verdict["result_evidence.welds"] = [
                False,
                f"scenarios[{si}].weld_decisions: array unavailable",
            ]
        else:
            for wi, weld in enumerate(welds):
                path = f"scenarios[{si}].weld_decisions[{wi}]"
                weld_error = _weld_error(weld, path)
                if weld_error:
                    verdict["result_evidence.welds"] = [False, weld_error]
    return [
        AtomVerdict(key, values[0], values[1], ATOM_SHARES["result_evidence"][key])
        for key, values in verdict.items()
    ]


def _key_defects(keys: Sequence[Any], expected: set[Any], label: str) -> list[str]:
    counts = Counter(keys)
    defects = [f"{label}: missing {key}" for key in sorted(expected - set(counts), key=str)]
    for key, count in counts.items():
        if key not in expected:
            defects.extend(f"{label}: unknown {key}" for _ in range(count))
        if count > 1:
            defects.extend(f"{label}: duplicate {key}" for _ in range(count - 1))
    return defects


def _coverage(data: Any) -> tuple[float, list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenario: missing all"]
    defects: list[str] = []
    scenario_keys = [
        row.get("scenario")
        for row in data["scenarios"]
        if isinstance(row, dict) and isinstance(row.get("scenario"), str)
    ]
    defects.extend(_key_defects(scenario_keys, set(SCENARIOS), "scenario"))
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict) or scenario.get("scenario") not in SCENARIOS:
            continue
        name = scenario["scenario"]
        request_keys = (
            [
                (row.get("robot_id"), row.get("weld_id"), row.get("request_type"))
                for row in scenario.get("request_decisions", [])
                if isinstance(row, dict)
                and isinstance(row.get("robot_id"), str)
                and isinstance(row.get("weld_id"), str)
                and isinstance(row.get("request_type"), str)
            ]
            if isinstance(scenario.get("request_decisions"), list)
            else []
        )
        weld_keys = (
            [
                (row.get("robot_id"), row.get("weld_id"))
                for row in scenario.get("weld_decisions", [])
                if isinstance(row, dict)
                and isinstance(row.get("robot_id"), str)
                and isinstance(row.get("weld_id"), str)
            ]
            if isinstance(scenario.get("weld_decisions"), list)
            else []
        )
        defects.extend(
            _key_defects(
                request_keys,
                {(robot, weld, kind) for robot in ROBOTS for weld in WELDS for kind in KINDS},
                f"{name}/request",
            )
        )
        defects.extend(
            _key_defects(
                weld_keys,
                {(robot, weld) for robot in ROBOTS for weld in WELDS},
                f"{name}/weld",
            )
        )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _legality_atoms(data: Any, replayed: Mapping[str, Any] | None = None) -> list[AtomVerdict]:
    verdict = {key: [True, "ok"] for key in ATOM_SHARES["decision_legality"]}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return [
            AtomVerdict(key, False, "decision data unavailable", share)
            for key, share in ATOM_SHARES["decision_legality"].items()
        ]
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict):
            continue
        name = scenario.get("scenario")
        requests = scenario.get("request_decisions", [])
        if isinstance(requests, list):
            for request in requests:
                if not isinstance(request, dict):
                    continue
                label = (
                    f"{name}/{request.get('robot_id')}/{request.get('weld_id')}/"
                    f"{request.get('request_type')}"
                )
                attempts = request.get("attempts")
                if isinstance(attempts, list):
                    usable = [a for a in attempts if isinstance(a, dict)]
                    retries = [a.get("retry_index") for a in usable]
                    if all(isinstance(x, int) and not isinstance(x, bool) for x in retries):
                        if retries != list(range(len(retries))):
                            verdict["decision_legality.retries"] = [
                                False,
                                f"{label}.attempts[*].retry_index: expected="
                                f"{list(range(len(retries)))!r}, actual={retries!r}",
                            ]
                    slots = [a.get("start_slot") for a in usable]
                    if all(isinstance(x, int) and not isinstance(x, bool) for x in slots):
                        if slots != sorted(set(slots)):
                            verdict["decision_legality.time_sequences"] = [
                                False,
                                f"{label}.attempts[*].start_slot: expected strictly increasing "
                                f"unique sequence, actual={slots!r}",
                            ]
                    for attempt in usable:
                        path, worker, robot = (
                            attempt.get("path"),
                            attempt.get("worker_id"),
                            request.get("robot_id"),
                        )
                        typed = (
                            isinstance(path, str)
                            and path in {"local", "mec", "cloud"}
                            and isinstance(worker, str)
                            and worker in WORKERS
                        )
                        if typed and not _placement_consistent(robot, path, worker):
                            verdict["decision_legality.placement"] = [
                                False,
                                f"{label}: expected worker Lr for local, M0/M1 for mec, or C0 "
                                f"for cloud; actual path={path!r}, worker_id={worker!r}",
                            ]
                cache = request.get("cache_reuse_slots")
                if isinstance(cache, list) and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in cache
                ):
                    if cache != sorted(set(cache)):
                        verdict["decision_legality.time_sequences"] = [
                            False,
                            f"{label}.cache_reuse_slots: expected strictly increasing unique "
                            f"sequence, actual={cache!r}",
                        ]
                    request_type = request.get("request_type")
                    if (
                        isinstance(request_type, str)
                        and request_type in {"scene", "verify"}
                        and cache
                    ):
                        verdict["decision_legality.fallback_scope"] = [
                            False,
                            f"{label}.cache_reuse_slots: expected=[] for request_type="
                            f"{request_type!r}, actual={cache!r}",
                        ]
        welds = scenario.get("weld_decisions", [])
        if isinstance(welds, list):
            for weld in welds:
                if not isinstance(weld, dict):
                    continue
                slots = weld.get("workcell_request_slots")
                if (
                    isinstance(slots, list)
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in slots)
                    and slots != sorted(set(slots))
                ):
                    verdict["decision_legality.time_sequences"] = [
                        False,
                        f"{name}/{weld.get('robot_id')}/{weld.get('weld_id')}: "
                        "expected strictly increasing unique workcell_request_slots, "
                        f"actual={slots!r}",
                    ]
    static_ready = all(
        values[0]
        for key, values in verdict.items()
        if key != "decision_legality.dynamic_eligibility"
    )
    evidence_ready = (
        _score_atoms(_evidence_atoms(data), ATOM_SHARES["result_evidence"]) >= 1.0 - EPS
    )
    coverage_ready = _coverage(data)[0] >= 1.0 - EPS
    if static_ready and evidence_ready and coverage_ready:
        if replayed is None:
            _, replayed = _replay(_records(data))
        dynamic_errors = [
            detail
            for result in replayed.values()
            for detail in result["errors"]["dynamic_eligibility"]
        ]
        if dynamic_errors:
            verdict["decision_legality.dynamic_eligibility"] = [
                False,
                " | ".join(dynamic_errors[:10]),
            ]
    return [
        AtomVerdict(key, values[0], values[1], ATOM_SHARES["decision_legality"][key])
        for key, values in verdict.items()
    ]


def _records(data: Any) -> dict[tuple[Any, ...], Any]:
    records: dict[tuple[Any, ...], Any] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return records
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict) or scenario.get("scenario") not in SCENARIOS:
            continue
        name = scenario["scenario"]
        requests = scenario.get("request_decisions", [])
        if isinstance(requests, list):
            for row in requests:
                if (
                    isinstance(row, dict)
                    and isinstance(row.get("robot_id"), str)
                    and isinstance(row.get("weld_id"), str)
                    and isinstance(row.get("request_type"), str)
                ):
                    records.setdefault(
                        (name, "request", row["robot_id"], row["weld_id"], row["request_type"]),
                        row,
                    )
        welds = scenario.get("weld_decisions", [])
        if isinstance(welds, list):
            for row in welds:
                if (
                    isinstance(row, dict)
                    and isinstance(row.get("robot_id"), str)
                    and isinstance(row.get("weld_id"), str)
                ):
                    records.setdefault((name, "weld", row["robot_id"], row["weld_id"]), row)
    return records


def _wireless_state(name: str, slot: int) -> str:
    if name == "simultaneous_preheat" and 20 <= slot <= 59:
        return "congested"
    if name == "hardware_gap" and 80 <= slot <= 119:
        return "congested"
    if name == "uplink_jitter" and (15 <= slot <= 34 or 70 <= slot <= 94 or 135 <= slot <= 159):
        return "jitter"
    return "normal"


def _link_capacity(name: str, channel: str, robot: str, slot: int) -> int:
    base = 1024 if channel == "uplink" else 512
    state = _wireless_state(name, slot)
    if state == "normal":
        multiplier = 1.0
    elif state == "congested":
        multiplier = 0.5
    else:
        multiplier = 0.5 + 0.5 * ((slot + int(robot[1])) % 3) / 2.0
    return math.floor(base * multiplier)


def _m1_available(name: str, slot: int) -> bool:
    return not (name == "mec_worker_loss" and 55 <= slot <= 124)


def _path_quality(name: str, robot: str, kind: str, path: str) -> float:
    quality = float(REQUEST[kind]["quality"][path])
    if name == "hardware_gap" and robot in {"R2", "R3"} and path == "local":
        quality -= 0.08
    return quality


def _service_units(name: str, robot: str, kind: str, path: str) -> int:
    units = int(REQUEST[kind]["service"][path])
    if name == "hardware_gap" and robot in {"R2", "R3"} and path == "local":
        return math.ceil(1.5 * units)
    return units


def _attempt_id(robot: str, weld: str, kind: str, retry: int) -> str:
    return f"A_{robot}_{weld}_{kind}_{retry}"


def _request_id(robot: str, weld: str, kind: str) -> str:
    return f"Q_{robot}_{weld}_{kind}"


def _workcell(robot: str, weld: str) -> str:
    return f"W{(int(robot[1]) + int(weld[1])) % 2}"


class Simulator:
    """Fixed deterministic event replay; submitted records contain decisions only."""

    def __init__(self, name: str, records: Mapping[tuple[Any, ...], Any]) -> None:
        self.name = name
        self.records = records
        base, offsets, span, energy_budget = PLAN_KEYS[name]
        self.base = base
        self.offsets = offsets
        self.span = span
        self.energy_budget = energy_budget
        self.current = {robot: 0 for robot in ROBOTS}
        self.welds = {
            (robot, weld): {
                "state": "future",
                "activation": None,
                "hard": None,
                "progress": 0.0,
                "ready_slot": None,
                "completed_slot": None,
                "failure": None,
            }
            for robot in ROBOTS
            for weld in WELDS
        }
        self.requests = {
            (robot, weld, kind): {
                "released": False,
                "release": None,
                "deadline": None,
                "status": "inactive",
                "active_attempt": None,
                "next_retry": 0,
                "retry_ready": 0,
                "adoption": None,
                "quality": None,
                "path": None,
                "attempt_id": None,
            }
            for robot in ROBOTS
            for weld in WELDS
            for kind in KINDS
        }
        self.attempts: dict[str, dict[str, Any]] = {}
        self.events: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self.worker_queues = {worker: [] for worker in WORKERS}
        self.worker_active: dict[str, str | None] = {worker: None for worker in WORKERS}
        self.packets: dict[str, dict[str, Any]] = {}
        self.channels = {"uplink": [], "downlink": []}
        self.workcells = {
            "W0": {"robot": None, "weld": None, "remaining": 0},
            "W1": {"robot": None, "weld": None, "remaining": 0},
        }
        self.cache = {robot: None for robot in ROBOTS}
        self.errors = {
            "dynamic_eligibility": [],
            "activation_deadlines": [],
            "network_workers": [],
            "process_workcells": [],
            "resources": [],
        }
        self.views: dict[int, dict[str, Any]] = {}
        self.safety = 0
        self.interventions = 0
        self.productive_slots = 0
        self.ready_wait_slots = 0
        self.energy = 0.0
        self.robot_energy = {robot: 0.0 for robot in ROBOTS}
        self.communication_bits = 0
        self.remote_compute_units = 0
        self.queue_rejections = 0
        self.remote_attempts = 0
        self.served_by_attempt: dict[str, int] = defaultdict(int)
        self.adopted_remote: set[str] = set()
        self.request_adoptions: list[dict[str, Any]] = []

        self.attempt_decisions: dict[int, list[tuple[tuple[str, str, str], dict[str, Any]]]] = (
            defaultdict(list)
        )
        self.cache_decisions: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self.workcell_decisions: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self.abandon_decisions: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self._index_decisions()

    def _index_decisions(self) -> None:
        for robot in ROBOTS:
            for weld in WELDS:
                for kind in KINDS:
                    row = self.records.get((self.name, "request", robot, weld, kind))
                    if not isinstance(row, Mapping):
                        continue
                    attempts = row.get("attempts", [])
                    if isinstance(attempts, list):
                        for attempt in attempts:
                            if _attempt_valid(attempt) and _placement_consistent(
                                robot, attempt["path"], attempt["worker_id"]
                            ):
                                self.attempt_decisions[attempt["start_slot"]].append(
                                    ((robot, weld, kind), attempt)
                                )
                    cache = row.get("cache_reuse_slots", [])
                    if kind == "start" and isinstance(cache, list):
                        for slot in cache:
                            if _integer(slot, 0, H - 1):
                                self.cache_decisions[slot].append((robot, weld))
                row = self.records.get((self.name, "weld", robot, weld))
                if not isinstance(row, Mapping):
                    continue
                slots = row.get("workcell_request_slots", [])
                if isinstance(slots, list):
                    for slot in slots:
                        if _integer(slot, 0, H - 1):
                            self.workcell_decisions[slot].append((robot, weld))
                abandon = row.get("abandon_slot")
                if _integer(abandon, 0, H - 1):
                    self.abandon_decisions[abandon].append((robot, weld))

    def _current_weld(self, robot: str) -> tuple[str, dict[str, Any]] | None:
        index = self.current[robot]
        return None if index >= len(WELDS) else (WELDS[index], self.welds[(robot, WELDS[index])])

    def _release(self, robot: str, weld: str, kind: str, slot: int) -> None:
        request = self.requests[(robot, weld, kind)]
        if request["released"]:
            self.errors["process_workcells"].append(
                f"{self.name}/{robot}/{weld}/{kind}/{slot}: request released twice"
            )
            return
        hard = self.welds[(robot, weld)]["hard"]
        request.update(
            released=True,
            release=slot,
            deadline=min(hard, slot + int(REQUEST[kind]["offset"])),
            status="waiting",
            retry_ready=slot,
        )

    def _activate(self, robot: str, slot: int) -> None:
        current = self._current_weld(robot)
        if current is None:
            return
        weld, state = current
        if state["state"] != "future":
            return
        earliest = self.base + 10 * int(weld[1]) + self.offsets[int(robot[1])]
        if slot < earliest:
            return
        state.update(state="inspect_wait", activation=slot, hard=min(H, slot + self.span))
        self._release(robot, weld, "scene", slot)

    def _cancel_attempt(self, attempt_id: str, status: str) -> None:
        attempt = self.attempts.get(attempt_id)
        if not attempt or attempt["status"] in {"accepted", "failed", "cancelled"}:
            return
        attempt["status"] = status
        worker = attempt.get("worker")
        if isinstance(worker, str) and worker in self.worker_active:
            if self.worker_active[worker] == attempt_id:
                self.worker_active[worker] = None
            self.worker_queues[worker] = [
                queued for queued in self.worker_queues[worker] if queued != attempt_id
            ]
        for packet in self.packets.values():
            if packet.get("attempt") == attempt_id and packet.get("status") == "queued":
                packet["status"] = "cancelled"
        key = attempt["request_key"]
        request = self.requests[key]
        if request["active_attempt"] == attempt_id:
            request["active_attempt"] = None

    def _finish_weld(self, robot: str, weld: str, slot: int, outcome: str) -> None:
        state = self.welds[(robot, weld)]
        if state["state"] in {"complete", "failed", "abandoned"}:
            return
        if state["state"] == "welding":
            cell = _workcell(robot, weld)
            if self.workcells[cell]["robot"] == robot and self.workcells[cell]["weld"] == weld:
                self.workcells[cell] = {"robot": None, "weld": None, "remaining": 0}
        state["state"] = outcome
        if outcome == "complete":
            state["completed_slot"] = slot
        else:
            state["failure"] = outcome
        for kind in KINDS:
            request = self.requests[(robot, weld, kind)]
            if request["active_attempt"]:
                self._cancel_attempt(request["active_attempt"], "cancelled")
            if request["status"] not in {"adopted", "inactive"}:
                request["status"] = "cancelled"
        self.current[robot] += 1

    def _adopt(self, attempt_id: str, slot: int) -> None:
        attempt = self.attempts.get(attempt_id)
        if not attempt or attempt["status"] in {"accepted", "failed", "cancelled"}:
            return
        robot, weld, kind = attempt["request_key"]
        request = self.requests[(robot, weld, kind)]
        quality = _path_quality(self.name, robot, kind, attempt["path"])
        if (
            request["status"] != "inflight"
            or slot > request["deadline"]
            or quality + EPS < REQUEST[kind]["minimum"]
        ):
            self._cancel_attempt(attempt_id, "failed")
            if request["status"] != "cancelled":
                request["status"] = "waiting"
                request["retry_ready"] = slot + 1
            return
        attempt["status"] = "accepted"
        request.update(
            status="adopted",
            active_attempt=None,
            adoption=slot,
            quality=quality,
            path=attempt["path"],
            attempt_id=attempt_id,
        )
        if attempt["path"] != "local":
            self.adopted_remote.add(attempt_id)
        self.request_adoptions.append(
            {
                "key": (robot, weld, kind),
                "release": request["release"],
                "deadline": request["deadline"],
                "adoption": slot,
                "quality": quality,
                "path": attempt["path"],
                "attempt_id": attempt_id,
            }
        )
        state = self.welds[(robot, weld)]
        if kind == "scene":
            state["progress"] = max(state["progress"], 0.15)
            state["state"] = "locate_wait"
            self._release(robot, weld, "start", slot)
        elif kind == "start":
            state["progress"] = max(state["progress"], 0.40)
            state["state"] = "ready_weld"
            state["ready_slot"] = slot
            self.cache[robot] = {"quality": quality, "adoption": slot, "attempt_id": attempt_id}
        else:
            state["progress"] = 1.0
            self._finish_weld(robot, weld, slot, "complete")

    def _attempt_failure(self, attempt_id: str, slot: int, reason: str) -> None:
        attempt = self.attempts.get(attempt_id)
        if not attempt or attempt["status"] in {"accepted", "failed", "cancelled"}:
            return
        attempt["status"] = "failed"
        attempt["failure"] = reason
        request = self.requests[attempt["request_key"]]
        if request["active_attempt"] == attempt_id:
            request["active_attempt"] = None
        if request["status"] not in {"adopted", "expired", "cancelled"}:
            request["status"] = "waiting"
            request["retry_ready"] = slot + 1

    def _worker_load(self, worker: str) -> int:
        return len(self.worker_queues[worker]) + (1 if self.worker_active[worker] else 0)

    def _enqueue_worker(self, attempt_id: str, slot: int) -> bool:
        attempt = self.attempts[attempt_id]
        worker = attempt["worker"]
        if worker == "M1" and not _m1_available(self.name, slot):
            self._attempt_failure(attempt_id, slot, "worker_unavailable")
            return False
        if self._worker_load(worker) >= WORKER_CAPACITY[worker]:
            self.queue_rejections += 1
            attempt["pending_failure"] = "worker_queue_rejected"
            attempt["status"] = "failure_pending"
            self.events[slot + 1].append(("attempt_fail", attempt_id))
            return False
        robot, _weld, kind = attempt["request_key"]
        attempt["remaining"] = _service_units(self.name, robot, kind, attempt["path"])
        attempt["admission"] = slot
        attempt["status"] = "worker_queue"
        self.worker_queues[worker].append(attempt_id)
        return True

    def _channel_has_capacity(self, channel: str, robot: str) -> bool:
        active = [pid for pid in self.channels[channel] if self.packets[pid]["status"] == "queued"]
        return (
            len(active) < CHANNEL_CAPACITY
            and sum(self.packets[pid]["robot"] == robot for pid in active) < ROBOT_PACKET_CAPACITY
        )

    def _enqueue_packet(self, attempt_id: str, direction: str, slot: int) -> None:
        attempt = self.attempts[attempt_id]
        robot, _weld, kind = attempt["request_key"]
        channel = "uplink" if direction == "input" else "downlink"
        if not self._channel_has_capacity(channel, robot):
            self.queue_rejections += 1
            self.events[slot + 1].append(("attempt_fail", attempt_id))
            attempt["pending_failure"] = "packet_queue_rejected"
            attempt["status"] = "failure_pending"
            return
        packet_id = f"{'I' if direction == 'input' else 'O'}_{attempt_id}"
        bits = int(REQUEST[kind][direction])
        self.packets[packet_id] = {
            "attempt": attempt_id,
            "robot": robot,
            "kind": kind,
            "path": attempt["path"],
            "priority": attempt["priority"],
            "deadline": self.requests[attempt["request_key"]]["deadline"],
            "admission": slot,
            "remaining": bits,
            "served": 0,
            "direction": direction,
            "status": "queued",
        }
        self.channels[channel].append(packet_id)
        attempt["status"] = "uplink" if direction == "input" else "downlink"

    def _process_events(self, slot: int) -> None:
        order = {
            "attempt_fail": 0,
            "compute_done": 1,
            "input_arrival": 2,
            "output_arrival": 3,
            "weld_done": 4,
        }
        events = sorted(self.events.pop(slot, []), key=lambda item: (order[item[0]], item[1]))
        for event, ident in events:
            if event == "attempt_fail":
                attempt = self.attempts.get(ident)
                reason = attempt.get("pending_failure", "link_loss") if attempt else "link_loss"
                self._attempt_failure(ident, slot, reason)
            elif event == "input_arrival":
                attempt = self.attempts.get(ident)
                if attempt and attempt["status"] == "in_transit_input":
                    self._enqueue_worker(ident, slot)
            elif event == "compute_done":
                attempt = self.attempts.get(ident)
                if not attempt or attempt["status"] != "compute_done":
                    continue
                if attempt["path"] == "local":
                    self._adopt(ident, slot)
                else:
                    self._enqueue_packet(ident, "output", slot)
            elif event == "output_arrival":
                attempt = self.attempts.get(ident)
                if attempt and attempt["status"] == "in_transit_output":
                    self._adopt(ident, slot)
            elif event == "weld_done":
                robot, weld = ident.split("/")
                state = self.welds[(robot, weld)]
                if state["state"] == "welding":
                    state["progress"] = max(state["progress"], 0.80)
                    state["state"] = "verify_wait"
                    self._release(robot, weld, "verify", slot)

    def _apply_m1_failure(self, slot: int) -> None:
        if self.name != "mec_worker_loss" or slot != 55:
            return
        affected = list(self.worker_queues["M1"])
        if self.worker_active["M1"]:
            affected.append(self.worker_active["M1"])
        self.worker_queues["M1"].clear()
        self.worker_active["M1"] = None
        for attempt_id in affected:
            self._attempt_failure(attempt_id, slot, "worker_unavailable")

    def _expire(self, slot: int) -> None:
        for key, request in self.requests.items():
            if (
                request["released"]
                and request["status"] not in {"adopted", "expired", "cancelled"}
                and slot >= request["deadline"]
            ):
                if request["active_attempt"]:
                    self._cancel_attempt(request["active_attempt"], "cancelled")
                request["status"] = "expired"
        for robot in ROBOTS:
            current = self._current_weld(robot)
            if current is None:
                continue
            weld, state = current
            if state["hard"] is not None and slot >= state["hard"] and state["state"] != "complete":
                self._finish_weld(robot, weld, slot, "failed")

    def _view(self, slot: int) -> dict[str, Any]:
        robots: dict[str, Any] = {}
        for robot in ROBOTS:
            current = self._current_weld(robot)
            if current is None:
                robots[robot] = {"current_weld": None, "state": "done", "request": None}
                continue
            weld, state = current
            kind = {"inspect_wait": "scene", "locate_wait": "start", "verify_wait": "verify"}.get(
                state["state"]
            )
            request = self.requests[(robot, weld, kind)] if kind else None
            cache = self.cache[robot]
            cache_view = None
            if cache:
                age = slot - cache["adoption"]
                cache_quality = cache["quality"] - 0.005 * age
                cache_view = {
                    "age": age,
                    "quality": cache_quality,
                    "qualified": age <= 40 and cache_quality + EPS >= 0.90,
                }
            robots[robot] = {
                "current_weld": weld,
                "state": state["state"],
                "hard_deadline": state["hard"],
                "ready_slot": state["ready_slot"],
                "request_type": kind,
                "request": None
                if request is None
                else {
                    "status": request["status"],
                    "release": request["release"],
                    "deadline": request["deadline"],
                    "next_retry": request["next_retry"],
                    "retry_ready": request["retry_ready"],
                },
                "cache": cache_view,
                "local_queue": self._worker_load(f"L{robot[1]}"),
                "energy": self.robot_energy[robot],
                "workcell": _workcell(robot, weld),
                "workcell_busy": self.workcells[_workcell(robot, weld)]["robot"] is not None,
                "link_state": _wireless_state(self.name, slot) if slot < H else None,
                "uplink_capacity": _link_capacity(self.name, "uplink", robot, slot)
                if slot < H
                else 0,
                "downlink_capacity": _link_capacity(self.name, "downlink", robot, slot)
                if slot < H
                else 0,
                "local_quality": None
                if kind is None
                else _path_quality(self.name, robot, kind, "local"),
                "local_service_units": None
                if kind is None
                else _service_units(self.name, robot, kind, "local"),
            }
        return {
            "slot": slot,
            "robots": robots,
            "workers": {
                worker: {
                    "load": self._worker_load(worker),
                    "available": worker != "M1" or _m1_available(self.name, slot),
                }
                for worker in WORKERS
            },
            "channels": {
                channel: sum(
                    self.packets.get(packet_id, {}).get("status") == "queued"
                    for packet_id in self.channels[channel]
                )
                for channel in self.channels
            },
            "communication_remaining": COMMUNICATION_BUDGET - self.communication_bits,
            "remote_compute_remaining": REMOTE_COMPUTE_BUDGET - self.remote_compute_units,
        }

    def _start_attempts(self, slot: int) -> None:
        decisions = list(self.attempt_decisions.get(slot, []))

        def rank(item: tuple[tuple[str, str, str], dict[str, Any]]) -> tuple[Any, ...]:
            key, decision = item
            deadline = self.requests[key]["deadline"]
            attempt_id = _attempt_id(*key, decision["retry_index"])
            return (
                -decision["priority"],
                H + 1 if deadline is None else deadline,
                attempt_id,
            )

        for key, decision in sorted(decisions, key=rank):
            robot, weld, kind = key
            request = self.requests[key]
            retry = decision.get("retry_index")
            label = f"{self.name}/{robot}/{weld}/{kind}/{slot}"
            current = self._current_weld(robot)
            reasons: list[str] = []
            if current is None or current[0] != weld:
                actual = None if current is None else current[0]
                reasons.append(f"current_weld expected={weld!r}, actual={actual!r}")
            if not request["released"]:
                reasons.append("released expected=True, actual=False")
            else:
                if request["status"] != "waiting":
                    reasons.append(
                        f"request_status expected='waiting', actual={request['status']!r}"
                    )
                if slot >= request["deadline"]:
                    reasons.append(f"start_slot expected < {request['deadline']}, actual={slot}")
                if slot < request["retry_ready"]:
                    reasons.append(
                        f"start_slot expected >= {request['retry_ready']}, actual={slot}"
                    )
            if reasons:
                self.errors["dynamic_eligibility"].append(
                    f"{label}: "
                    + "; ".join(reasons)
                    + "; context="
                    + repr(
                        {
                            "release": request["release"],
                            "deadline": request["deadline"],
                            "status": request["status"],
                            "active_attempt": request["active_attempt"],
                            "retry_ready": request["retry_ready"],
                            "expected_retry_index": request["next_retry"],
                            "submitted_retry_index": retry,
                            "submitted_start_slot": slot,
                        }
                    )
                )
                continue
            if retry != request["next_retry"]:
                continue
            attempt_id = _attempt_id(robot, weld, kind, retry)
            attempt = {
                "id": attempt_id,
                "request_key": key,
                "path": decision.get("path"),
                "worker": decision.get("worker_id"),
                "priority": decision.get("priority"),
                "start": slot,
                "status": "new",
                "remaining": None,
                "failure": None,
            }
            self.attempts[attempt_id] = attempt
            request["active_attempt"] = attempt_id
            request["status"] = "inflight"
            request["next_retry"] += 1
            if attempt["path"] == "local":
                if self._enqueue_worker(attempt_id, slot):
                    cost = float(REQUEST[kind]["energy"])
                    self.energy += cost
                    self.robot_energy[robot] += cost
            else:
                self.remote_attempts += 1
                self._enqueue_packet(attempt_id, "input", slot)

    def _cache_requests(self, slot: int) -> None:
        for robot, weld in self.cache_decisions.get(slot, []):
            current = self._current_weld(robot)
            state = self.welds[(robot, weld)]
            cache = self.cache[robot]
            valid = (
                current is not None
                and current[0] == weld
                and state["state"] == "locate_wait"
                and cache is not None
                and slot - cache["adoption"] <= 40
                and cache["quality"] - 0.005 * (slot - cache["adoption"]) + EPS >= 0.90
            )
            if not valid:
                self.safety += 1
                continue
            request = self.requests[(robot, weld, "start")]
            if request["active_attempt"]:
                self._cancel_attempt(request["active_attempt"], "cancelled")
            quality = cache["quality"] - 0.005 * (slot - cache["adoption"])
            request.update(
                status="adopted",
                active_attempt=None,
                adoption=slot,
                quality=quality,
                path="cache",
                attempt_id=cache["attempt_id"],
            )
            self.request_adoptions.append(
                {
                    "key": (robot, weld, "start"),
                    "release": request["release"],
                    "deadline": request["deadline"],
                    "adoption": slot,
                    "quality": quality,
                    "path": "cache",
                    "attempt_id": None,
                }
            )
            state["progress"] = max(state["progress"], 0.40)
            state["state"] = "ready_weld"
            state["ready_slot"] = slot

    def _abandon(self, slot: int) -> None:
        for robot, weld in self.abandon_decisions.get(slot, []):
            current = self._current_weld(robot)
            state = self.welds[(robot, weld)]
            if (
                current is None
                or current[0] != weld
                or state["state"] in {"future", "welding", "complete", "failed", "abandoned"}
            ):
                self.safety += 1
            else:
                self._finish_weld(robot, weld, slot, "abandoned")

    def _worker_service(self, slot: int) -> None:
        for worker in WORKERS:
            if worker == "M1" and not _m1_available(self.name, slot):
                continue
            active_id = self.worker_active[worker]
            if active_id is None:
                queue = [
                    aid
                    for aid in self.worker_queues[worker]
                    if self.attempts.get(aid, {}).get("status") == "worker_queue"
                ]
                self.worker_queues[worker] = queue
                if queue:

                    def rank(aid: str) -> tuple[Any, ...]:
                        attempt = self.attempts[aid]
                        request = self.requests[attempt["request_key"]]
                        return (
                            -attempt["priority"],
                            request["deadline"],
                            attempt["admission"],
                            aid,
                        )

                    active_id = min(queue, key=rank)
                    queue.remove(active_id)
                    self.worker_active[worker] = active_id
                    self.attempts[active_id]["status"] = "computing"
            if active_id is None or (worker == "C0" and slot % 2 == 1):
                continue
            attempt = self.attempts[active_id]
            if worker in {"M0", "M1", "C0"}:
                if self.remote_compute_units >= REMOTE_COMPUTE_BUDGET:
                    continue
                self.remote_compute_units += 1
            attempt["remaining"] -= 1
            if attempt["remaining"] == 0:
                attempt["status"] = "compute_done"
                self.worker_active[worker] = None
                self.events[slot + 1].append(("compute_done", active_id))

    def _channel_service(self, slot: int) -> None:
        for channel in ("uplink", "downlink"):
            queue = [
                pid
                for pid in self.channels[channel]
                if self.packets.get(pid, {}).get("status") == "queued"
            ]
            self.channels[channel] = queue
            if not queue or self.communication_bits >= COMMUNICATION_BUDGET:
                continue

            def rank(pid: str) -> tuple[Any, ...]:
                packet = self.packets[pid]
                return (-packet["priority"], packet["deadline"], packet["admission"], pid)

            packet_id = min(queue, key=rank)
            packet = self.packets[packet_id]
            capacity = _link_capacity(self.name, channel, packet["robot"], slot)
            served = min(
                packet["remaining"], capacity, COMMUNICATION_BUDGET - self.communication_bits
            )
            packet["remaining"] -= served
            packet["served"] += served
            self.communication_bits += served
            self.served_by_attempt[packet["attempt"]] += served
            if packet["remaining"] > 0:
                continue
            packet["status"] = "complete"
            queue.remove(packet_id)
            level = {"normal": 0, "congested": 1, "jitter": 2}[_wireless_state(self.name, slot)]
            robot_index = int(packet["robot"][1])
            kind_index = KINDS.index(packet["kind"])
            success = ((17 * robot_index + 7 * slot + kind_index) % 100) / 100.0 >= (
                0.04 + 0.12 * level
            )
            attempt = self.attempts[packet["attempt"]]
            if not success:
                attempt["pending_failure"] = "link_loss"
                attempt["status"] = "failure_pending"
                self.events[slot + 1].append(("attempt_fail", packet["attempt"]))
                continue
            delay = 1 + ((slot + robot_index) % 3)
            if packet["path"] == "cloud":
                delay += 8 + ((3 * slot + robot_index) % 11)
            if packet["direction"] == "input":
                attempt["status"] = "in_transit_input"
                self.events[slot + delay].append(("input_arrival", packet["attempt"]))
            else:
                attempt["status"] = "in_transit_output"
                self.events[slot + delay].append(("output_arrival", packet["attempt"]))

    def _workcell_service(self, slot: int) -> None:
        free_before = {cell for cell, value in self.workcells.items() if value["robot"] is None}
        contenders: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for robot, weld in self.workcell_decisions.get(slot, []):
            current = self._current_weld(robot)
            state = self.welds[(robot, weld)]
            if current is None or current[0] != weld or state["state"] != "ready_weld":
                self.safety += 1
                continue
            cell = _workcell(robot, weld)
            if cell in free_before:
                contenders[cell].append((robot, weld))
        for cell, options in contenders.items():

            def rank(item: tuple[str, str]) -> tuple[Any, ...]:
                robot, weld = item
                state = self.welds[item]
                return (state["ready_slot"], state["hard"] - slot, robot)

            winner = min(options, key=rank)
            self.interventions += len(options) - 1
            robot, weld = winner
            self.welds[(robot, weld)]["state"] = "welding"
            self.workcells[cell] = {"robot": robot, "weld": weld, "remaining": 12}
        for robot in ROBOTS:
            current = self._current_weld(robot)
            if current is None:
                continue
            weld, state = current
            if state["state"] == "ready_weld" and _workcell(robot, weld) in free_before:
                self.ready_wait_slots += 1
        for cell, value in self.workcells.items():
            if value["robot"] is None:
                continue
            self.productive_slots += 1
            value["remaining"] -= 1
            if value["remaining"] == 0:
                ident = f"{value['robot']}/{value['weld']}"
                self.events[slot + 1].append(("weld_done", ident))
                self.workcells[cell] = {"robot": None, "weld": None, "remaining": 0}

    def _audit_invariants(self) -> None:
        allowed_progress = (0.0, 0.15, 0.40, 0.80, 1.0)
        for (robot, weld), state in self.welds.items():
            activation = state["activation"]
            earliest = self.base + 10 * int(weld[1]) + self.offsets[int(robot[1])]
            if activation is None or activation < earliest:
                self.errors["activation_deadlines"].append(
                    f"{self.name}/{robot}/{weld}: activation expected >= {earliest}, "
                    f"actual={activation!r}"
                )
            elif state["hard"] != min(H, activation + self.span):
                self.errors["activation_deadlines"].append(
                    f"{self.name}/{robot}/{weld}: hard expected="
                    f"{min(H, activation + self.span)}, actual={state['hard']!r}"
                )
            if state["state"] not in {"complete", "failed", "abandoned"}:
                self.errors["process_workcells"].append(
                    f"{self.name}/{robot}/{weld}: terminal state expected, "
                    f"actual={state['state']!r}"
                )
            if not any(
                math.isclose(state["progress"], value, abs_tol=EPS) for value in allowed_progress
            ):
                self.errors["process_workcells"].append(
                    f"{self.name}/{robot}/{weld}: progress expected one of "
                    f"{allowed_progress!r}, actual={state['progress']!r}, tolerance={EPS}"
                )
        for (robot, weld, kind), request in self.requests.items():
            if not request["released"]:
                continue
            expected = min(
                self.welds[(robot, weld)]["hard"],
                request["release"] + int(REQUEST[kind]["offset"]),
            )
            if request["deadline"] != expected:
                self.errors["activation_deadlines"].append(
                    f"{self.name}/{robot}/{weld}/{kind}: deadline expected={expected}, "
                    f"actual={request['deadline']!r}"
                )
        active_workers = [worker for worker, attempt in self.worker_active.items() if attempt]
        queued_jobs = [
            attempt
            for worker in WORKERS
            for attempt in self.worker_queues[worker]
            if self.attempts.get(attempt, {}).get("status") == "worker_queue"
        ]
        queued_packets = [
            packet
            for channel in self.channels.values()
            for packet in channel
            if self.packets.get(packet, {}).get("status") == "queued"
        ]
        nonterminal_attempts = [
            attempt_id
            for attempt_id, attempt in self.attempts.items()
            if attempt["status"] not in {"accepted", "failed", "cancelled"}
        ]
        if active_workers or queued_jobs or queued_packets or nonterminal_attempts:
            self.errors["network_workers"].append(
                f"{self.name}: terminal resources expected empty; active_workers={active_workers}, "
                f"queued_jobs={queued_jobs}, queued_packets={queued_packets}, "
                f"nonterminal_attempts={nonterminal_attempts}"
            )
        busy_cells = [cell for cell, value in self.workcells.items() if value["robot"] is not None]
        if busy_cells:
            self.errors["process_workcells"].append(
                f"{self.name}: terminal workcells expected idle, actual_busy={busy_cells}"
            )

    def run(self) -> dict[str, Any]:
        for slot in range(H + 1):
            self._apply_m1_failure(slot)
            self._process_events(slot)
            for _ in range(len(WELDS) + 1):
                current_before = tuple(self.current.values())
                for robot in ROBOTS:
                    self._activate(robot, slot)
                self._expire(slot)
                if tuple(self.current.values()) == current_before:
                    break
            self.views[slot] = self._view(slot)
            if slot == H:
                break
            self._cache_requests(slot)
            self._start_attempts(slot)
            self._abandon(slot)
            self._channel_service(slot)
            self._worker_service(slot)
            self._workcell_service(slot)
            if any(len(self.channels[channel]) > CHANNEL_CAPACITY for channel in self.channels):
                self.errors["resources"].append(f"{self.name}/{slot}: channel capacity exceeded")
            if any(self._worker_load(worker) > WORKER_CAPACITY[worker] for worker in WORKERS):
                self.errors["resources"].append(f"{self.name}/{slot}: worker capacity exceeded")
            if (
                self.communication_bits > COMMUNICATION_BUDGET
                or self.remote_compute_units > REMOTE_COMPUTE_BUDGET
            ):
                self.errors["resources"].append(f"{self.name}/{slot}: service budget exceeded")
        self._audit_invariants()
        return self.result()

    def result(self) -> dict[str, Any]:
        useful_bits = sum(self.served_by_attempt[aid] for aid in self.adopted_remote)
        completed = sum(state["state"] == "complete" for state in self.welds.values())
        progress = sum(float(state["progress"]) for state in self.welds.values())
        return {
            "welds": self.welds,
            "requests": self.requests,
            "attempts": self.attempts,
            "views": self.views,
            "errors": self.errors,
            "completed_welds": completed,
            "stage_progress": progress,
            "safety_events": self.safety,
            "interventions": self.interventions,
            "productive_weld_slots": self.productive_slots,
            "ready_wait_slots": self.ready_wait_slots,
            "energy": self.energy,
            "energy_budget": self.energy_budget,
            "communication_bits": self.communication_bits,
            "remote_compute_units": self.remote_compute_units,
            "queue_rejections": self.queue_rejections,
            "remote_attempts": self.remote_attempts,
            "adopted_remote": len(self.adopted_remote),
            "useful_remote_bits": useful_bits,
            "served_by_attempt": dict(self.served_by_attempt),
            "request_adoptions": self.request_adoptions,
        }


def _replay(records: Mapping[tuple[Any, ...], Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    errors = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}
    replayed: dict[str, Any] = {}
    for name in SCENARIOS:
        result = Simulator(name, records).run()
        replayed[name] = result
        for category, details in result["errors"].items():
            if category in errors:
                errors[category].extend(details)
    atoms = [
        AtomVerdict(
            f"replay_validity.{category}",
            not details,
            "ok" if not details else " | ".join(details[:10]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{category}"],
        )
        for category, details in errors.items()
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    values = {
        key: []
        for key in (
            "primary_metrics_1",
            "primary_metrics_2",
            "primary_metrics_3",
            "auxiliary_metrics_1",
            "auxiliary_metrics_2",
        )
    }
    for result in replayed.values():
        completion = result["completed_welds"] / 24.0
        progress = result["stage_progress"] / 24.0
        p1 = _clip(0.65 * completion + 0.35 * progress)

        quality_sum = 0.0
        deadline_count = 0
        remote_freshness: list[float] = []
        for adoption in result["request_adoptions"]:
            age = adoption["adoption"] - adoption["release"]
            freshness = _clip(1.0 - age / 24.0)
            quality_sum += adoption["quality"] * freshness
            deadline_count += adoption["adoption"] <= adoption["deadline"]
            if adoption["path"] in {"mec", "cloud"}:
                remote_freshness.append(freshness)
        perception_quality = quality_sum / 72.0
        deadline_service = deadline_count / 72.0
        p2 = _clip(
            0.5 * perception_quality
            + 0.3 * deadline_service
            + 0.2 * _clip(1.0 - result["safety_events"] / (4.0 * H))
        )

        productive = result["productive_weld_slots"]
        waiting = result["ready_wait_slots"]
        flow = productive / (productive + waiting) if productive + waiting else 0.0
        p3 = _clip(
            0.45 * flow
            + 0.25 * _clip(1.0 - result["interventions"] / (4.0 * H))
            + 0.30 * _clip(1.0 - result["energy"] / result["energy_budget"])
        )

        remote_bits = result["communication_bits"]
        freshness = sum(remote_freshness) / len(remote_freshness) if remote_freshness else 0.0
        a1 = _clip(
            0.45 * result["useful_remote_bits"] / max(1, remote_bits)
            + 0.30 * result["adopted_remote"] / max(1, result["remote_attempts"])
            + 0.25 * freshness
        )
        a2 = _clip(
            0.45 * _clip(1.0 - remote_bits / COMMUNICATION_BUDGET)
            + 0.35 * _clip(1.0 - result["remote_compute_units"] / REMOTE_COMPUTE_BUDGET)
            + 0.20 * _clip(1.0 - result["queue_rejections"] / 72.0)
        )
        for key, value in zip(values, (p1, p2, p3, a1, a2)):
            values[key].append(value)
    return {key: _clip(sum(group) / len(group)) for key, group in values.items()}


def _normalized_task_performance(metrics: Mapping[str, float]) -> float:
    return _clip(
        (
            8 * metrics.get("primary_metrics_1", 0.0)
            + 4 * metrics.get("primary_metrics_2", 0.0)
            + 3 * metrics.get("primary_metrics_3", 0.0)
            + 3 * metrics.get("auxiliary_metrics_1", 0.0)
            + 2 * metrics.get("auxiliary_metrics_2", 0.0)
        )
        / 20.0
    )


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match fixed declaration")
    if any(not math.isfinite(value) or value <= 0.0 for value in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("atom shares must sum to one")
    if any(not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share mismatch")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diag(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return (
        "failed_atoms=["
        + ",".join(atom.id for atom in failed)
        + "]; "
        + " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    )


def _gated_metrics(
    scores: Mapping[str, float], replayed: Mapping[str, Any]
) -> dict[str, float] | None:
    required = (
        "official_run",
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    )
    return (
        _metrics(replayed) if all(scores.get(key, 0.0) >= 1.0 - EPS for key in required) else None
    )


RESULT_DEPENDENT = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
)


def _regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not path.is_symlink()


def _absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _zero_downstream(state: EvalState, reason: str) -> None:
    for key in RESULT_DEPENDENT:
        state.scores[key] = 0.0
        state.diagnostics[key] = f"{reason}; {key}=0"


async def _build_state(ws: Workspace) -> EvalState:
    root = str(Path(ws.root).resolve())
    if root in _CACHE:
        return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE:
            return _CACHE[root]
        state = EvalState()
        result_path = Path(ws.root, "results.json")
        deletion_ok = False
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
            remove_exit = getattr(removed, "exit_code", None)
            deletion_ok = remove_exit == 0 and _absent(result_path)
            if not deletion_ok:
                reason = (
                    f"old results deletion failed: exit={remove_exit}, "
                    f"absent={_absent(result_path)}"
                )
                state.scores["official_run"] = 0.0
                state.diagnostics["official"] = reason
                _zero_downstream(state, reason)
                _CACHE[root] = state
                return state
            state.run_started = True
            started = time.monotonic()
            try:
                result = await ws.run("python run.py", check=False, timeout=600)
            finally:
                state.run_finished = True
                state.elapsed_s = time.monotonic() - started
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = (
                state.exit_code is None
                or "timeout" in stderr.lower()
                or "timed out" in stderr.lower()
                or (state.elapsed_s is not None and state.elapsed_s >= 600.0 - EPS)
            )
            state.result_file_is_new_regular = deletion_ok and _regular_file(result_path)
            state.official_ok = bool(
                state.exit_code == 0
                and not state.timed_out
                and state.elapsed_s <= 600.0 + EPS
                and state.result_file_is_new_regular
            )
            if state.official_ok:
                try:
                    state.data = _strict_json(ws.read_text("results.json") or "")
                    state.parse_ok = True
                except Exception as exc:
                    state.diagnostics["parse"] = f"strict parse failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            state.timed_out = (
                isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
            state.diagnostics["official"] = f"official run raised {type(exc).__name__}: {exc}"
        elapsed = "none" if state.elapsed_s is None else f"{state.elapsed_s:.6f}"
        state.diagnostics.setdefault(
            "official",
            f"deletion_ok={deletion_ok}, exit={state.exit_code}, timed_out={state.timed_out}, "
            f"elapsed_s={elapsed}, new_regular_non_symlink={state.result_file_is_new_regular}",
        )
        state.scores["official_run"] = 1.0 if state.official_ok else 0.0
        if not state.official_ok:
            _zero_downstream(state, f"official run failed: {state.diagnostics['official']}")
            _CACHE[root] = state
            return state
        if not state.parse_ok:
            _zero_downstream(state, state.diagnostics.get("parse", "strict parse failed"))
            _CACHE[root] = state
            return state
        try:
            atoms = _result_atoms(state.data, True)
            state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"])
            state.diagnostics["result_json"] = _diag(atoms)
            atoms = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"])
            state.diagnostics["result_evidence"] = _diag(atoms)
            state.scores["canonical_coverage"], defects = _coverage(state.data)
            state.diagnostics["canonical_coverage"] = f"defects={defects}"
            state.records = _records(state.data)
            atoms, state.replayed = _replay(state.records)
            state.scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"])
            state.diagnostics["replay_validity"] = _diag(atoms)
            atoms = _legality_atoms(state.data, state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diag(atoms)
            state.metrics = _gated_metrics(state.scores, state.replayed)
            state.validity_gate = state.metrics is not None
            if state.metrics:
                for key, value in state.metrics.items():
                    state.scores[key] = value
                    state.diagnostics[key] = f"{key}={value:.9f}"
            else:
                for key in RESULT_DEPENDENT[5:]:
                    state.scores[key] = 0.0
                    state.diagnostics[key] = f"validity gate failed; {key}=0"
        except Exception as exc:
            _zero_downstream(
                state, f"deterministic result evaluation failed: {type(exc).__name__}: {exc}"
            )
        _CACHE[root] = state
        return state


def _report_score(text: str) -> tuple[float, list[str]]:
    missing: list[str] = []
    headings = "|".join(re.escape(title) for title in REPORT_SECTIONS)
    next_heading = re.compile(rf"^#{{1,6}}[ \t]+(?:{headings})[ \t]*$", re.M)
    for title in REPORT_SECTIONS:
        match = re.search(rf"^#{{1,6}}[ \t]+{re.escape(title)}[ \t]*$", text, re.M)
        if not match:
            missing.append(title)
            continue
        nxt = next_heading.search(text[match.end() :])
        body = text[match.end() : match.end() + (nxt.start() if nxt else len(text))]
        if not body.strip():
            missing.append(title)
    return max(0.0, (2.0 - 0.5 * len(missing)) / 2.0), missing


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
        ok = _regular_file(Path(ws.root, "run.py")) and getattr(result, "exit_code", 1) == 0
    except Exception as exc:
        return 0.0, f"run.py compile failed: {type(exc).__name__}: {exc}"
    return (1.0, "run.py exists and compiles") if ok else (0.0, "run.py missing or invalid")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("official_run", 0.0), state.diagnostics.get("official", "")


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "report.md")
    if not _regular_file(path):
        return 0.0, "report.md missing, non-regular, or symbolic link"
    try:
        text = ws.read_text("report.md")
        if not isinstance(text, str):
            return 0.0, "report.md read did not return text"
        score, missing = _report_score(text)
    except Exception as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    return score, f"missing_or_empty={missing}"


async def _gate(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(key, 0.0), state.diagnostics.get(key, "")


for _check_id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):
    globals()[_check_id] = check(id=_check_id)(lambda ws, key=_check_id: _gate(ws, key))


def _metric_check(key: str):
    @check(id=key)
    async def metric(ws: Workspace) -> tuple[float, str]:
        state = await _build_state(ws)
        if not state.validity_gate:
            return 0.0, state.diagnostics.get(key, f"{key}: validity gate failed")
        return state.metrics.get(key, 0.0), state.diagnostics.get(key, "")

    return metric


for _check_id in (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):
    globals()[_check_id] = _metric_check(_check_id)
