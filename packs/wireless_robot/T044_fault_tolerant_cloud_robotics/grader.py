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
HORIZON = 150
INITIAL_ENERGY = 145.0
MAX_DISPATCHES = 56
REPLICAS = ("R1", "R2", "R3")
OBJECT_IDS = tuple(f"o{i}" for i in range(1, 9))
STAGES = ("scan", "plan")
MOTIONS = ("N", "E", "S", "W", "wait")
MANIPULATIONS = ("none", "pick", "place")
KINDS = ("none", "dispatch", "cancel", "local_scan")
TERMINAL_STATUSES = {"preempted", "lost", "rejected", "cancelled", "late", "stale", "duplicate", "accepted"}
SLOT_FIELDS = {"slot", "motion", "service_action", "manipulation"}
SERVICE_FIELDS = {"kind", "object_id", "stage", "replicas"}
CANONICAL = tuple(f"{scene}_v{v}" for scene in (
    "stable_replicas", "spot_preemption", "regional_slowdown", "queue_contention", "recovery_overlap"
) for v in range(3))
METRIC_KEYS = ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")
GATE_CHECKS = ("official_run", "result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity")
ATOM_SHARES = {
    "result_json": {"result_json.parse": 0.20, "result_json.root": 0.25, "result_json.condition_shape": 0.25, "result_json.slot_shape": 0.30},
    "result_evidence": {"result_evidence.slot": 0.22, "result_evidence.motion": 0.16, "result_evidence.service": 0.38, "result_evidence.manipulation": 0.24},
    "decision_legality": {"decision_legality.motion": 0.12, "decision_legality.service": 0.18, "decision_legality.stage": 0.18, "decision_legality.group": 0.12, "decision_legality.replication": 0.15, "decision_legality.manipulation": 0.13, "decision_legality.energy": 0.12},
    "replay_validity": {"replay_validity.network": 0.22, "replay_validity.queue": 0.18, "replay_validity.version": 0.18, "replay_validity.robot": 0.22, "replay_validity.metrics": 0.20},
}

STATIC_BLOCKED = {(4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (7, 1), (7, 2), (7, 7), (7, 8)}
OBJECTS = {
    "o1": ((2, 2), 1.00, 0, 43), "o2": ((5, 1), 1.15, 0, 54),
    "o3": ((3, 6), 1.10, 10, 72), "o4": ((6, 4), 1.30, 16, 86),
    "o5": ((8, 8), 1.00, 22, 105), "o6": ((1, 8), 1.05, 30, 116),
    "o7": ((10, 6), 1.35, 38, 132), "o8": ((9, 2), 1.25, 45, 145),
}
SCENARIOS = {
    "stable_replicas": {"phase": 1, "preempt": {}, "slow": None, "bg": "sparse", "obstacle": ((5, 5), 46, 82)},
    "spot_preemption": {"phase": 4, "preempt": {"R1": (30, 38), "R3": (74, 82)}, "slow": None, "bg": "sparse", "obstacle": ((6, 5), 42, 90)},
    "regional_slowdown": {"phase": 7, "preempt": {}, "slow": (32, 70), "bg": "sparse", "obstacle": ((8, 3), 44, 88)},
    "queue_contention": {"phase": 10, "preempt": {}, "slow": None, "bg": "contention", "obstacle": ((3, 5), 52, 94)},
    "recovery_overlap": {"phase": 13, "preempt": {"R1": (36, 44), "R2": (40, 48)}, "slow": None, "bg": "recovery", "obstacle": ((6, 6), 48, 92)},
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _brief(value: Any, limit: int = 160) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        return f"<{type(value).__name__} repr failed: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


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
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(float(v)) or v <= 0 for v in expected.values()):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0.0, abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share mismatch")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("atom shares do not sum to one")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(a.id for a in failed) + "]; " + " | ".join(f"{a.id}: {a.detail}" for a in failed)


def _parts(condition_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_v([0-2])", condition_id)
    if not match or match.group(1) not in SCENARIOS:
        return None
    return match.group(1), int(match.group(2))


def _params(condition_id: str) -> dict[str, Any]:
    parsed = _parts(condition_id)
    if parsed is None:
        raise ValueError(f"unknown condition {condition_id}")
    scene, variant = parsed
    base = SCENARIOS[scene]
    preempt: dict[str, tuple[int, int]] = {}
    for replica, bounds in base["preempt"].items():
        if len(bounds) == 2:
            preempt[replica] = (bounds[0] + 2 * variant, bounds[1] + 2 * variant)
        else:
            preempt[replica] = (bounds[0] + 2 * variant, bounds[1] + 2 * variant)
    return {"scene": scene, "variant": variant, "phase": base["phase"] + 5 * variant, "preempt": preempt, "slow": None if base["slow"] is None else (base["slow"][0] + 2 * variant, base["slow"][1] + 2 * variant), "bg": base["bg"], "obstacle": (base["obstacle"][0], base["obstacle"][1] + 2 * variant, base["obstacle"][2] + 2 * variant)}


def _object_info(object_id: str, variant: int) -> tuple[tuple[int, int], float, int, int]:
    cell, value, release, deadline = OBJECTS[object_id]
    return cell, value, release + 2 * variant, deadline - 2 * variant


def _in_slow(params: Mapping[str, Any], slot: int) -> bool:
    interval = params["slow"]
    return interval is not None and interval[0] <= slot < interval[1]


def _replicas_valid(value: Any, *, allow_empty: bool = True) -> bool:
    """Validate replica arrays without ever hashing malformed JSON values."""
    if not isinstance(value, list):
        return False
    if not allow_empty and not value:
        return False
    if len(value) > 2:
        return False
    if not all(isinstance(replica, str) for replica in value):
        return False
    return len(set(value)) == len(value) and all(replica in REPLICAS for replica in value)


def _preempted(params: Mapping[str, Any], replica: str, slot: int) -> bool:
    bounds = params["preempt"].get(replica)
    return bounds is not None and bounds[0] <= slot < bounds[1]


def _target_cell(pos: tuple[int, int], motion: str) -> tuple[int, int]:
    dxdy = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0), "wait": (0, 0)}
    dx, dy = dxdy[motion]
    return pos[0] + dx, pos[1] + dy


def _record_action(row: Any, slot: int) -> dict[str, Any]:
    default = {"slot": slot, "motion": "wait", "service_action": {"kind": "none", "object_id": None, "stage": None, "replicas": []}, "manipulation": "none"}
    if not isinstance(row, dict) or set(row) != SLOT_FIELDS:
        return default
    service = row.get("service_action")
    if not _integer(row.get("slot")) or not isinstance(row.get("motion"), str) or not isinstance(row.get("manipulation"), str) or not isinstance(service, dict) or set(service) != SERVICE_FIELDS:
        return default
    if not isinstance(service.get("replicas"), list):
        return default
    return row


class CloudEnvironment:
    """Deterministic public-model replay, also reusable by author methods."""

    def __init__(self, condition_id: str):
        self.condition_id = condition_id
        self.params = _params(condition_id)
        self.variant = self.params["variant"]
        self.t = 0
        self.pos = (0, 0)
        self.energy = INITIAL_ENERGY
        self.carrying: str | None = None
        self.map_version = 0
        self.dynamic: set[tuple[int, int]] = set()
        self.objects: dict[str, dict[str, Any]] = {}
        for oid in OBJECT_IDS:
            cell, value, release, deadline = _object_info(oid, self.variant)
            self.objects[oid] = {"cell": cell, "value": value, "release": release, "deadline": deadline, "released": False, "expired": False, "info": 0, "scan_ready": False, "local_used": False, "local_scan_timeout": False, "plan_map": None, "picked": None, "placed": None}
        self.replicas: dict[str, dict[str, Any]] = {r: {"up": True, "queue": [], "current": None, "reports": [], "latest_report": None} for r in REPLICAS}
        self.groups: dict[str, dict[str, Any]] = {}
        self.current_groups: dict[tuple[str, str], str] = {}
        self.attempts: dict[str, dict[str, Any]] = {}
        self.responses: dict[int, list[str]] = {}
        self.status_due: dict[int, list[dict[str, Any]]] = {}
        self.local_remaining = 0
        self.local_object: str | None = None
        self.dispatches = 0
        self.sent_bits = 0
        self.useful_bits = 0
        self.first_stage_accept: dict[tuple[str, str], bool] = {}
        self.stage_delay: dict[tuple[str, str], int] = {}
        self.fallback_count = 0
        self.collisions = 0
        self.motion_energy = 0.0
        self.picked_by_deadline: dict[str, bool] = {oid: False for oid in OBJECT_IDS}
        self.placed_by_deadline: dict[str, bool] = {oid: False for oid in OBJECT_IDS}
        self.queue_waits: list[int] = []
        self.errors: dict[str, list[str]] = {name: [] for name in ("motion", "service", "stage", "group", "replication", "manipulation", "energy")}
        self.invariants: dict[str, list[str]] = {name: [] for name in ("network", "queue", "version", "robot", "metrics")}
        self.trace: list[dict[str, Any]] = []

    def _active_group(self, oid: str, stage: str) -> dict[str, Any] | None:
        gid = self.current_groups.get((oid, stage))
        group = self.groups.get(gid) if gid else None
        return group if group and group["active"] else None

    def _group_id(self, oid: str, stage: str, slot: int) -> str:
        return f"{self.condition_id}|{oid}|{stage}|g{slot}"

    def _attempt_id(self, gid: str, replica: str) -> str:
        return f"{gid}|{replica}"

    def _report_for(self, replica: str) -> dict[str, Any] | None:
        return self.replicas[replica]["latest_report"]

    def _observation(self) -> dict[str, Any]:
        local_blocked = []
        all_blocked = STATIC_BLOCKED | self.dynamic
        for cell in sorted(all_blocked):
            if abs(cell[0] - self.pos[0]) + abs(cell[1] - self.pos[1]) <= 2:
                local_blocked.append(cell)
        released = {oid: {"cell": obj["cell"], "value": obj["value"], "deadline": obj["deadline"], "info": obj["info"], "scan_ready": obj["scan_ready"], "plan_map": obj["plan_map"], "expired": obj["expired"], "picked": obj["picked"] is not None, "placed": obj["placed"] is not None} for oid, obj in self.objects.items() if obj["released"]}
        active = {f"{oid}|{stage}": {"dispatch_slot": group["dispatch"], "info": group["info"], "map": group["map"]} for (oid, stage), gid in self.current_groups.items() if (group := self.groups.get(gid)) and group["active"]}
        return {"slot": self.t, "position": self.pos, "energy": self.energy, "carrying": self.carrying, "map_version": self.map_version, "local_obstacles": tuple(local_blocked), "released": released, "replicas": {r: self._report_for(r) for r in REPLICAS}, "active_groups": active, "dispatches": self.dispatches, "local_scan_remaining": self.local_remaining, "local_scan_object": self.local_object}

    def _set_terminal(self, attempt: dict[str, Any], status: str, slot: int) -> None:
        if attempt["terminal"] is None:
            attempt["terminal"] = status
            attempt["terminal_slot"] = slot

    def _deliver_responses(self, slot: int) -> None:
        for aid in sorted(self.responses.pop(slot, [])):
            attempt = self.attempts.get(aid)
            if attempt is None or attempt["terminal"] is not None:
                continue
            if attempt["down_lost"]:
                self._set_terminal(attempt, "lost", slot)
                continue
            group = self.groups[attempt["group"]]
            obj = self.objects[attempt["object"]]
            if slot > group["dispatch"] + 12 or slot > obj["deadline"]:
                self._set_terminal(attempt, "late", slot)
            elif attempt["info"] != obj["info"] or (attempt["stage"] == "plan" and attempt["map"] != self.map_version):
                self._set_terminal(attempt, "stale", slot)
            elif group["accepted"] or (attempt["stage"] == "scan" and obj["scan_ready"]) or (attempt["stage"] == "plan" and obj["plan_map"] == self.map_version):
                self._set_terminal(attempt, "duplicate", slot)
            else:
                group["accepted"] = True
                group["active"] = False
                if self.current_groups.get((attempt["object"], attempt["stage"])) == attempt["group"]:
                    self.current_groups.pop((attempt["object"], attempt["stage"]), None)
                self._set_terminal(attempt, "accepted", slot)
                key = (attempt["object"], attempt["stage"])
                self.stage_delay.setdefault(key, slot - group["dispatch"])
                if key not in self.first_stage_accept:
                    self.first_stage_accept[key] = True
                    self.useful_bits += 360 if attempt["stage"] == "scan" else 520
                if attempt["stage"] == "scan":
                    obj["info"] = 1
                    obj["scan_ready"] = True
                else:
                    obj["plan_map"] = attempt["map"]

    def _deliver_reports(self, slot: int) -> None:
        for report in self.status_due.pop(slot, []):
            self.replicas[report["replica"]]["latest_report"] = report

    def _apply_events(self, slot: int) -> None:
        for replica in REPLICAS:
            bounds = self.params["preempt"].get(replica)
            if bounds and slot == bounds[0]:
                state = self.replicas[replica]
                state["up"] = False
                for job in list(state["queue"]):
                    if job["kind"] == "robot":
                        self._set_terminal(self.attempts[job["attempt"]], "preempted", slot)
                state["queue"] = []
                if state["current"] is not None and state["current"]["kind"] == "robot":
                    self._set_terminal(self.attempts[state["current"]["attempt"]], "preempted", slot)
                state["current"] = None
            if bounds and slot == bounds[1]:
                self.replicas[replica]["up"] = True
                self.replicas[replica]["queue"] = []
                self.replicas[replica]["current"] = None
        cell, add_slot, remove_slot = self.params["obstacle"]
        if slot == add_slot:
            self.dynamic.add(cell)
            self.map_version += 1
            for obj in self.objects.values():
                obj["plan_map"] = None
            for group in self.groups.values():
                if group["stage"] == "plan" and group["map"] != self.map_version:
                    group["active"] = False
            self.current_groups = {
                (oid, stage): gid
                for (oid, stage), gid in self.current_groups.items()
                if self.groups[gid]["active"] and (stage == "scan" or self.groups[gid]["map"] == self.map_version)
            }
        if slot == remove_slot:
            self.dynamic.discard(cell)
            self.map_version += 1
            for obj in self.objects.values():
                obj["plan_map"] = None
            for group in self.groups.values():
                if group["stage"] == "plan" and group["map"] != self.map_version:
                    group["active"] = False
            self.current_groups = {
                (oid, stage): gid
                for (oid, stage), gid in self.current_groups.items()
                if self.groups[gid]["active"] and (stage == "scan" or self.groups[gid]["map"] == self.map_version)
            }
        for oid, obj in self.objects.items():
            if slot == obj["release"]:
                obj["released"] = True
            if slot > obj["deadline"] and not obj["expired"] and obj["placed"] is None:
                obj["expired"] = True
                for key, gid in list(self.current_groups.items()):
                    if key[0] == oid:
                        self.groups[gid]["active"] = False
                        self.current_groups.pop(key, None)
                if self.carrying == oid:
                    self.carrying = None

    def _timeout_groups(self, slot: int) -> None:
        for gid, group in self.groups.items():
            if group["active"] and slot >= group["dispatch"] + 12:
                group["active"] = False
                if self.current_groups.get((group["object"], group["stage"])) == gid:
                    self.current_groups.pop((group["object"], group["stage"]), None)
                if group["stage"] == "scan":
                    self.objects[group["object"]]["local_scan_timeout"] = True

    def start_slot(self, slot: int) -> dict[str, Any]:
        self.t = slot
        self._deliver_responses(slot)
        self._deliver_reports(slot)
        self._apply_events(slot)
        self._timeout_groups(slot)
        return self._observation()

    def _service_ok(self, service: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
        kind = service.get("kind")
        oid, stage = service.get("object_id"), service.get("stage")
        reps = service.get("replicas")
        if kind == "none":
            return oid is None and stage is None and reps == []
        if kind == "dispatch":
            return isinstance(oid, str) and oid in OBJECT_IDS and stage in STAGES and _replicas_valid(reps, allow_empty=False)
        if kind == "cancel":
            return isinstance(oid, str) and oid in OBJECT_IDS and stage in STAGES and reps == []
        if kind == "local_scan":
            return isinstance(oid, str) and oid in OBJECT_IDS and stage == "scan" and reps == []
        return False

    def _legal_flags(self, action: Mapping[str, Any]) -> dict[str, bool]:
        raw_service = action.get("service_action", {})
        service = raw_service if isinstance(raw_service, Mapping) else {}
        kind = service.get("kind")
        oid, stage = service.get("object_id"), service.get("stage")
        reps = service.get("replicas")
        motion = action.get("motion")
        manipulation = action.get("manipulation")
        obj = self.objects.get(oid) if isinstance(oid, str) else None
        service_ok = self._service_ok(service, action)
        motion_ok = motion in MOTIONS and (self.local_remaining == 0 or motion == "wait")
        if self.carrying is not None and motion in ("N", "E", "S", "W"):
            motion_ok = motion_ok and self.objects[self.carrying]["plan_map"] == self.map_version
        if kind == "local_scan" and motion != "wait":
            motion_ok = False
        stage_ok = True
        if kind == "dispatch":
            stage_ok = obj is not None and obj["released"] and not obj["expired"] and self._active_group(oid, stage) is None
            if stage == "scan":
                stage_ok = stage_ok and obj["info"] == 0 and not obj["scan_ready"] and not obj["local_used"]
            elif stage == "plan":
                stage_ok = stage_ok and obj["info"] == 1 and obj["scan_ready"] and obj["plan_map"] != self.map_version
        elif kind == "local_scan":
            stage_ok = obj is not None and obj["released"] and not obj["expired"] and obj["info"] == 0 and not obj["scan_ready"] and not obj["local_used"] and obj["local_scan_timeout"] and self._active_group(oid, "scan") is None and self.local_remaining == 0 and self.pos == obj["cell"] and self.t + 2 <= obj["deadline"]
        group_ok = True
        if kind == "cancel":
            group_ok = self._active_group(oid, stage) is not None
        replication_ok = True
        if kind == "dispatch":
            replication_ok = _replicas_valid(reps, allow_empty=False) and self.dispatches + len(reps) <= MAX_DISPATCHES
        manipulation_ok = manipulation in MANIPULATIONS
        if manipulation in ("pick", "place") and kind != "none":
            manipulation_ok = False
        if manipulation in ("pick", "place") and motion != "wait":
            manipulation_ok = False
        if manipulation == "pick":
            candidates = [x for x in self.objects.values() if x["released"] and not x["expired"] and x["cell"] == self.pos and x["scan_ready"] and x["plan_map"] == self.map_version and x["picked"] is None and x["placed"] is None]
            manipulation_ok = manipulation_ok and self.carrying is None and len(candidates) == 1
        elif manipulation == "place":
            manipulation_ok = manipulation_ok and self.carrying is not None and self.pos == (11, 0)
        if self.local_remaining > 0 and manipulation != "none":
            manipulation_ok = False
        if kind == "local_scan" and manipulation != "none":
            service_ok = False
        service_effective = service_ok and stage_ok and group_ok and replication_ok
        energy_cost = (
            (4.0 if service_effective and kind == "local_scan" else 0.0)
            + (1.0 if motion_ok and motion in ("N", "E", "S", "W") else 0.0)
            + (2.0 if manipulation_ok and manipulation in ("pick", "place") else 0.0)
        )
        energy_ok = _finite(self.energy) and self.energy + EPS >= energy_cost
        return {"motion": motion_ok, "service": service_ok, "stage": stage_ok, "group": group_ok, "replication": replication_ok, "manipulation": manipulation_ok, "energy": energy_ok}

    def _cancel(self, oid: str, stage: str, slot: int) -> None:
        group = self._active_group(oid, stage)
        if group is None:
            return
        group["active"] = False
        self.current_groups.pop((oid, stage), None)
        for aid in group["attempts"]:
            attempt = self.attempts[aid]
            if attempt["started"] is None and attempt["response_generated"] is False:
                self._set_terminal(attempt, "cancelled", slot)
        for state in self.replicas.values():
            state["queue"] = [job for job in state["queue"] if job.get("attempt") not in group["attempts"]]

    def _create_dispatch(self, oid: str, stage: str, reps: list[str], slot: int) -> None:
        obj = self.objects[oid]
        gid = self._group_id(oid, stage, slot)
        group = {"id": gid, "object": oid, "stage": stage, "info": obj["info"], "map": self.map_version, "dispatch": slot, "active": True, "accepted": False, "attempts": []}
        self.groups[gid] = group
        self.current_groups[(oid, stage)] = gid
        payload = 360 if stage == "scan" else 520
        for rep in sorted(reps):
            aid = self._attempt_id(gid, rep)
            slow = 2 if rep == "R2" and _in_slow(self.params, slot) else 0
            arrival = slot + 1 + ((5 * slot + 2 * int(rep[1]) + self.params["phase"]) % 3) + slow
            lost = ((11 * slot + 5 * int(rep[1]) + 3 * self.params["phase"]) % 23) < (1 + (2 if rep == "R2" and _in_slow(self.params, slot) else 0))
            attempt = {"id": aid, "group": gid, "object": oid, "stage": stage, "info": obj["info"], "map": self.map_version, "replica": rep, "dispatch": slot, "arrival": arrival, "started": None, "end": None, "response_generated": False, "response_arrival": None, "down_lost": False, "terminal": None, "terminal_slot": None, "payload": payload}
            self.attempts[aid] = attempt
            group["attempts"].append(aid)
            self.responses.setdefault(arrival, [])
            attempt["uplink_lost"] = lost
        self.dispatches += len(reps)
        self.sent_bits += len(reps) * payload

    def _admit_network(self, slot: int) -> None:
        for rep in REPLICAS:
            state = self.replicas[rep]
            incoming: list[dict[str, Any]] = []
            for attempt in self.attempts.values():
                if attempt["replica"] == rep and attempt["arrival"] == slot and attempt["terminal"] is None:
                    if attempt["uplink_lost"]:
                        self._set_terminal(attempt, "lost", slot)
                    elif not state["up"]:
                        self._set_terminal(attempt, "rejected", slot)
                    else:
                        incoming.append({"kind": "robot", "attempt": attempt["id"], "id": f"r/{attempt['group']}/{rep}", "remaining": 1 if attempt["stage"] == "scan" else 2, "arrival": slot})
            if self.params["bg"] == "sparse":
                if (slot + 2 * int(rep[1]) + self.params["phase"]) % 9 == 0:
                    incoming.append({"kind": "background", "attempt": None, "id": f"b/{rep}/{slot}", "remaining": 1, "arrival": slot})
            elif self.params["bg"] == "recovery":
                if (slot + 2 * int(rep[1]) + self.params["phase"]) % 6 == 0:
                    incoming.append({"kind": "background", "attempt": None, "id": f"b/{rep}/{slot}", "remaining": 1, "arrival": slot})
            else:
                div = (3, 4, 5)[int(rep[1]) - 1]
                mul = (1, 2, 3)[int(rep[1]) - 1]
                if (slot + mul * int(rep[1]) + self.params["phase"]) % div == 0:
                    incoming.append({"kind": "background", "attempt": None, "id": f"b/{rep}/{slot}", "remaining": 1, "arrival": slot})
            if not state["up"]:
                continue
            incoming.sort(key=lambda item: item["id"])
            for job in incoming:
                occupancy = len(state["queue"]) + (1 if state["current"] is not None else 0)
                if occupancy >= 12:
                    if job["kind"] == "robot":
                        self._set_terminal(self.attempts[job["attempt"]], "rejected", slot)
                    continue
                state["queue"].append(job)

    def _service(self, slot: int) -> None:
        for rep in REPLICAS:
            state = self.replicas[rep]
            if not state["up"]:
                continue
            if state["current"] is None and state["queue"]:
                state["current"] = state["queue"].pop(0)
                job = state["current"]
                if job["kind"] == "robot":
                    attempt = self.attempts[job["attempt"]]
                    attempt["started"] = slot
                    self.queue_waits.append(slot - attempt["arrival"])
            if state["current"] is None:
                continue
            job = state["current"]
            job["remaining"] -= 1
            if job["remaining"] > 0:
                continue
            state["current"] = None
            if job["kind"] != "robot":
                continue
            attempt = self.attempts[job["attempt"]]
            attempt["end"] = slot
            attempt["response_generated"] = True
            slow = 2 if rep == "R2" and _in_slow(self.params, attempt["dispatch"]) else 0
            delay = 1 + ((3 * attempt["dispatch"] + int(rep[1]) + 2 * self.params["phase"]) % 2) + slow
            arrival = slot + delay
            attempt["response_arrival"] = arrival
            attempt["down_lost"] = ((7 * slot + 4 * int(rep[1]) + 5 * self.params["phase"]) % 29) < (1 + (2 if rep == "R2" and _in_slow(self.params, slot) else 0))
            self.responses.setdefault(arrival, []).append(attempt["id"])
        for rep in REPLICAS:
            state = self.replicas[rep]
            if state["up"]:
                report = {"replica": rep, "reported_slot": slot, "availability": "up", "queue_length": len(state["queue"]) + (1 if state["current"] is not None else 0), "map_version": self.map_version}
                self.status_due.setdefault(slot + 1, []).append(report)

    def _apply_service(self, service: Mapping[str, Any], slot: int) -> float:
        kind = service.get("kind")
        if kind == "local_scan":
            oid = service["object_id"]
            self.objects[oid]["local_used"] = True
            self.local_remaining = 3
            self.local_object = oid
            self.fallback_count += 1
            return 4.0
        if kind == "cancel":
            self._cancel(service["object_id"], service["stage"], slot)
        elif kind == "dispatch":
            self._create_dispatch(service["object_id"], service["stage"], list(service["replicas"]), slot)
        return 0.0

    def _robot(self, action: Mapping[str, Any], slot: int, flags: Mapping[str, bool], initial_cost: float = 0.0) -> None:
        motion = action.get("motion") if flags["motion"] else "wait"
        manipulation = action.get("manipulation") if flags["manipulation"] else "none"
        cost = initial_cost
        if motion in ("N", "E", "S", "W"):
            cost += 1.0
            self.motion_energy += 1.0
            proposed = _target_cell(self.pos, motion)
            if proposed[0] < 0 or proposed[0] > 11 or proposed[1] < 0 or proposed[1] > 9 or proposed in STATIC_BLOCKED or proposed in self.dynamic:
                self.collisions += 1
            else:
                self.pos = proposed
        if manipulation == "pick":
            candidates = [oid for oid, obj in self.objects.items() if obj["released"] and not obj["expired"] and obj["cell"] == self.pos and obj["scan_ready"] and obj["plan_map"] == self.map_version and obj["picked"] is None and obj["placed"] is None]
            if len(candidates) == 1:
                oid = candidates[0]
                self.carrying = oid
                self.objects[oid]["picked"] = slot
                if slot <= self.objects[oid]["deadline"]:
                    self.picked_by_deadline[oid] = True
                cost += 2.0
        elif manipulation == "place" and self.carrying is not None and self.pos == (11, 0):
            oid = self.carrying
            self.objects[oid]["placed"] = slot
            if slot <= self.objects[oid]["deadline"]:
                self.placed_by_deadline[oid] = True
            self.carrying = None
            cost += 2.0
        if self.local_remaining > 0:
            self.local_remaining -= 1
            if self.local_remaining == 0 and self.local_object is not None:
                obj = self.objects[self.local_object]
                if not obj["expired"]:
                    obj["info"] = 1
                    obj["scan_ready"] = True
                self.local_object = None
        self.energy -= cost
        if self.energy < -EPS:
            self.invariants["robot"].append(f"negative energy at slot {slot}")

    def apply_action(self, action: Mapping[str, Any], slot: int, collect_errors: bool = True) -> dict[str, Any]:
        flags = self._legal_flags(action)
        if collect_errors:
            for name, ok in flags.items():
                if not ok:
                    self.errors[name].append(f"slot={slot}")
        service = action.get("service_action", {})
        if not flags["service"] or not flags["stage"] or not flags["group"] or not flags["replication"]:
            service = {"kind": "none", "object_id": None, "stage": None, "replicas": []}
        action_effective = {"motion": action.get("motion"), "service_action": service, "manipulation": action.get("manipulation")}
        if not flags["motion"]:
            action_effective["motion"] = "wait"
        if not flags["manipulation"]:
            action_effective["manipulation"] = "none"
        if not flags["energy"]:
            action_effective = {"motion": "wait", "service_action": {"kind": "none", "object_id": None, "stage": None, "replicas": []}, "manipulation": "none"}
        service_cost = self._apply_service(service, slot)
        self._admit_network(slot)
        self._service(slot)
        self._robot(action_effective, slot, {"motion": True, "service": True, "stage": True, "group": True, "replication": True, "manipulation": True, "energy": True}, service_cost)
        self.trace.append({"slot": slot, "position": self.pos, "energy": self.energy, "carrying": self.carrying, "map_version": self.map_version, "collision": self.collisions, "dispatches": self.dispatches, "fallback_count": self.fallback_count})
        return flags

    def finish(self) -> None:
        self._deliver_responses(HORIZON)
        for attempt in self.attempts.values():
            if attempt["terminal"] is None:
                self._set_terminal(attempt, "late", HORIZON)
        for replica in REPLICAS:
            if len(self.replicas[replica]["queue"]) + (1 if self.replicas[replica]["current"] is not None else 0) > 12:
                self.invariants["queue"].append(f"capacity {replica}")

    def metrics(self) -> dict[str, float]:
        value_total = sum(float(OBJECTS[oid][1]) for oid in OBJECT_IDS)
        value_score = sum(float(OBJECTS[oid][1]) * (1.0 if self.placed_by_deadline[oid] else 0.25 if self.picked_by_deadline[oid] else 0.0) for oid in OBJECT_IDS) / value_total
        success = []
        timing = []
        for oid in OBJECT_IDS:
            for stage in STAGES:
                key = (oid, stage)
                ok = key in self.stage_delay
                success.append(1.0 if ok else 0.0)
                timing.append(_clip(1.0 - self.stage_delay[key] / 12.0) if ok else 0.0)
        p2 = 0.7 * (sum(success) / 16.0) + 0.3 * (sum(timing) / 16.0)
        p3 = 0.75 * _clip(1.0 - self.collisions / HORIZON) + 0.25 * _clip(1.0 - self.motion_energy / HORIZON)
        a1 = 0.6 * (self.useful_bits / max(self.sent_bits, 1)) + 0.4 * _clip(1.0 - self.dispatches / MAX_DISPATCHES)
        mean_wait = sum(self.queue_waits) / len(self.queue_waits) if self.queue_waits else 12.0
        a2 = 0.6 * _clip(1.0 - mean_wait / 12.0) + 0.4 * _clip(1.0 - self.fallback_count / 8.0)
        score = (8 * value_score + 4 * p2 + 3 * p3 + 3 * a1 + 2 * a2) / 20.0
        return {"primary_metrics_1": value_score, "primary_metrics_2": p2, "primary_metrics_3": p3, "auxiliary_metrics_1": a1, "auxiliary_metrics_2": a2, "normalized_performance": score, "collisions": float(self.collisions), "motion_energy": self.motion_energy, "dispatches": float(self.dispatches), "fallback_count": float(self.fallback_count), "accepted_stages": float(sum(1 for key in self.stage_delay if key[0] in OBJECT_IDS))}


def _shape_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    rows = data.get("conditions") if isinstance(data, dict) else None
    root = isinstance(data, dict) and set(data) == {"conditions"}
    condition_shape = isinstance(rows, list) and all(isinstance(row, dict) and set(row) == {"condition_id", "slots"} for row in rows)
    slot_shape = condition_shape and all(isinstance(row["slots"], list) and all(isinstance(item, dict) and set(item) == SLOT_FIELDS for item in row["slots"]) for row in rows)
    return [_atom("result_json", "parse", parsed, "strict JSON parsed" if parsed else "strict JSON parse failed"), _atom("result_json", "root", root, "exact root key"), _atom("result_json", "condition_shape", condition_shape, "exact condition object shape"), _atom("result_json", "slot_shape", slot_shape, "exact slot object shape")]


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    rows = data.get("conditions") if isinstance(data, dict) else None
    slot_ok = True
    motion_ok = True
    service_ok = True
    manipulation_ok = True
    detail = "valid"
    if not isinstance(rows, list):
        slot_ok = motion_ok = service_ok = manipulation_ok = False
        detail = "conditions is not an array"
    else:
        for row in rows:
            slots = row.get("slots") if isinstance(row, dict) else None
            if not isinstance(slots, list):
                slot_ok = False
                detail = "slots is not an array"
                continue
            for item in slots:
                if not isinstance(item, dict) or set(item) != SLOT_FIELDS or not _integer(item.get("slot")):
                    slot_ok = False
                    detail = f"invalid slot record {_brief(item)}"
                    continue
                if item.get("motion") not in MOTIONS:
                    motion_ok = False
                    detail = f"invalid motion at slot={_brief(item.get('slot'))}"
                if item.get("manipulation") not in MANIPULATIONS:
                    manipulation_ok = False
                    detail = f"invalid manipulation at slot={_brief(item.get('slot'))}"
                service = item.get("service_action")
                if not isinstance(service, dict) or set(service) != SERVICE_FIELDS:
                    service_ok = False
                    detail = f"invalid service shape at slot={_brief(item.get('slot'))}"
                    continue
                if service.get("kind") not in KINDS or not isinstance(service.get("replicas"), list):
                    service_ok = False
                    detail = f"invalid service enum at slot={_brief(item.get('slot'))}"
                if service.get("kind") == "dispatch" and (service.get("stage") not in STAGES or service.get("object_id") not in OBJECT_IDS or not _replicas_valid(service.get("replicas"), allow_empty=False)):
                    service_ok = False
                if service.get("kind") == "cancel" and (service.get("object_id") not in OBJECT_IDS or service.get("stage") not in STAGES or service.get("replicas") != []):
                    service_ok = False
                if service.get("kind") == "local_scan" and (service.get("object_id") not in OBJECT_IDS or service.get("stage") != "scan" or service.get("replicas") != []):
                    service_ok = False
                if service.get("kind") == "none" and (service.get("object_id") is not None or service.get("stage") is not None or service.get("replicas") != []):
                    service_ok = False
    return [_atom("result_evidence", "slot", slot_ok, detail if not slot_ok else "slot fields valid"), _atom("result_evidence", "motion", motion_ok, detail if not motion_ok else "motion values valid"), _atom("result_evidence", "service", service_ok, detail if not service_ok else "service actions valid"), _atom("result_evidence", "manipulation", manipulation_ok, detail if not manipulation_ok else "manipulation values valid")]


def _index(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rows = data.get("conditions") if isinstance(data, dict) else []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("condition_id"), str) and row["condition_id"] not in result:
                result[row["condition_id"]] = row
    return result


def _coverage(data: Any) -> tuple[float, list[str]]:
    rows = data.get("conditions") if isinstance(data, dict) and isinstance(data.get("conditions"), list) else []
    ids = [row.get("condition_id") for row in rows if isinstance(row, dict)]
    defects: list[str] = []
    for cid in CANONICAL:
        if ids.count(cid) == 0:
            defects.append(f"missing condition {cid}")
        elif ids.count(cid) > 1:
            defects.append(f"duplicate condition {cid}")
    for cid in ids:
        if cid not in CANONICAL:
            defects.append(f"unknown condition {_brief(cid)}")
    for row in rows:
        cid = row.get("condition_id") if isinstance(row, dict) else "<invalid>"
        if cid not in CANONICAL or not isinstance(row.get("slots"), list):
            continue
        slots = [item.get("slot") for item in row["slots"] if isinstance(item, dict)]
        for slot in range(HORIZON):
            if slots.count(slot) == 0:
                defects.append(f"{cid}: missing slot {slot}")
            elif slots.count(slot) > 1:
                defects.append(f"{cid}: duplicate slot {slot}")
        for slot in slots:
            if not _integer(slot) or slot < 0 or slot >= HORIZON:
                defects.append(f"{_brief(cid)}: unknown slot {_brief(slot)}")
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _replay(condition_id: str, row: Any) -> dict[str, Any]:
    env = CloudEnvironment(condition_id)
    slots = row.get("slots") if isinstance(row, dict) and isinstance(row.get("slots"), list) else []
    by_slot = {item.get("slot"): item for item in slots if isinstance(item, dict) and _integer(item.get("slot")) and 0 <= item.get("slot") < HORIZON}
    for t in range(HORIZON):
        env.start_slot(t)
        env.apply_action(_record_action(by_slot.get(t), t), t)
    env.finish()
    metrics = env.metrics()
    for attempt in env.attempts.values():
        if attempt["terminal"] is None:
            env.invariants["network"].append(f"unclosed attempt {attempt['id']}")
        elif attempt["terminal"] not in TERMINAL_STATUSES:
            env.invariants["network"].append(f"unknown terminal status for {attempt['id']}: {attempt['terminal']!r}")
        if attempt["response_generated"]:
            if attempt["started"] is None or attempt["end"] is None or attempt["response_arrival"] is None:
                env.invariants["network"].append(f"incomplete response lifecycle for {attempt['id']}")
            elif attempt["end"] < attempt["started"] or attempt["response_arrival"] < attempt["end"]:
                env.invariants["network"].append(f"non-monotone response lifecycle for {attempt['id']}")
        if attempt["terminal"] == "accepted":
            group = env.groups.get(attempt["group"])
            if group is None or not group["accepted"] or attempt["down_lost"]:
                env.invariants["network"].append(f"accepted attempt lacks accepted response state: {attempt['id']}")
    for state in env.replicas.values():
        if len(state["queue"]) + (1 if state["current"] is not None else 0) > 12:
            env.invariants["queue"].append("queue exceeds capacity")
        seen_jobs: set[str] = set()
        jobs = list(state["queue"])
        if state["current"] is not None:
            jobs.append(state["current"])
        for job in jobs:
            if job["kind"] == "robot":
                aid = job.get("attempt")
                if not isinstance(aid, str) or aid not in env.attempts:
                    env.invariants["queue"].append(f"unknown queued attempt {aid!r}")
                elif aid in seen_jobs:
                    env.invariants["queue"].append(f"duplicate queued attempt {aid}")
                else:
                    seen_jobs.add(aid)
                if job.get("remaining") not in (1, 2):
                    env.invariants["queue"].append(f"invalid remaining service for {aid!r}")
            elif job.get("kind") == "background":
                if job.get("attempt") is not None or job.get("remaining") != 1:
                    env.invariants["queue"].append("invalid background queue record")
            else:
                env.invariants["queue"].append("unknown queue job kind")
    if env.map_version not in (0, 1, 2):
        env.invariants["version"].append(f"invalid final map version {env.map_version}")
    for oid, obj in env.objects.items():
        if obj["info"] not in (0, 1) or (obj["plan_map"] is not None and obj["plan_map"] not in (0, 1, 2)):
            env.invariants["version"].append(f"invalid version state for {oid}")
    for attempt in env.attempts.values():
        if attempt["info"] not in (0, 1) or attempt["map"] not in (0, 1, 2):
            env.invariants["version"].append(f"invalid attempt version {attempt['id']}")
    if not _finite(env.energy) or env.energy < -EPS:
        env.invariants["robot"].append("invalid terminal energy")
    if any(not _finite(value) or value < -EPS for value in metrics.values() if isinstance(value, (int, float))):
        env.invariants["metrics"].append("non-finite metric")
    return {"env": env, "metrics": metrics, "invariants": env.invariants, "errors": env.errors}


def _legality_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    labels = {"motion": "motion and carrying-plan rules", "service": "service-action concurrency", "stage": "stage/version and release rules", "group": "active group and cancellation rules", "replication": "replication and dispatch budget", "manipulation": "pick/place and local-scan rules", "energy": "energy bounds"}
    return [_atom("decision_legality", name, not any(sim["errors"][name] for sim in replayed.values()), f"{labels[name]} valid" if not any(sim["errors"][name] for sim in replayed.values()) else next(sim["errors"][name][0] for sim in replayed.values() if sim["errors"][name])) for name in labels]


def _replay_atoms(replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    labels = {"network": "network and attempt lifecycles", "queue": "queue capacity and service", "version": "map and information versions", "robot": "robot state and energy", "metrics": "finite replay metrics"}
    return [_atom("replay_validity", name, not any(sim["invariants"][name] for sim in replayed.values()), f"{labels[name]} valid" if not any(sim["invariants"][name] for sim in replayed.values()) else next(sim["invariants"][name][0] for sim in replayed.values() if sim["invariants"][name])) for name in labels]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    return {key: sum(float(sim["metrics"][key]) for sim in replayed.values()) / max(1, len(replayed)) for key in METRIC_KEYS}


def _gated_metrics(replayed: Mapping[str, Any], validity_gate: bool) -> dict[str, float]:
    """Apply the public validity gate exactly once before metric credit."""
    return _metrics(replayed) if validity_gate else {key: 0.0 for key in METRIC_KEYS}


def _headings(text: str) -> dict[str, str]:
    """Parse visible Markdown headings and retain non-empty section bodies."""
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


def _failure_sim(detail: str) -> dict[str, Any]:
    return {
        "errors": {name.split(".", 1)[1]: [detail] for name in ATOM_SHARES["decision_legality"]},
        "invariants": {name.split(".", 1)[1]: [detail] for name in ATOM_SHARES["replay_validity"]},
        "metrics": {key: 0.0 for key in METRIC_KEYS},
    }


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
        message = f"old results cleanup raised {type(exc).__name__}: {exc}"
        state.scores = {item: 0.0 for item in GATE_CHECKS}
        state.diagnostics = {item: message for item in GATE_CHECKS}
        _CACHE[key] = state
        return state

    state.run_started = True
    started = time.monotonic()
    result = None
    try:
        result = await ws.run("python run.py", check=False, timeout=600.0)
        state.exit_code = getattr(result, "exit_code", None)
    except Exception as exc:
        state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
    finally:
        state.run_finished = True
        state.elapsed_s = time.monotonic() - started
    stderr = str(getattr(result, "stderr", ""))
    state.timed_out = state.timed_out or (
        state.exit_code is None
        and ("timeout" in stderr.lower() or state.elapsed_s >= 600.0 - EPS)
    )
    state.result_file_is_new_regular = result_path.is_file() and not result_path.is_symlink()
    state.official_ok = bool(
        remove_ok
        and state.exit_code == 0
        and not state.timed_out
        and state.result_file_is_new_regular
        and state.elapsed_s <= 600.0 + EPS
    )
    status = (
        f"exit={state.exit_code}, timed_out={state.timed_out}, "
        f"elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
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
        state.records = _index(state.data)
    except Exception as exc:
        state.records = {}
        state.diagnostics["result_json"] = state.diagnostics.get("result_json", "") + f"; index failed: {type(exc).__name__}: {exc}"
    try:
        coverage, defects = _coverage(state.data)
    except Exception as exc:
        coverage, defects = 0.0, [f"coverage check raised {type(exc).__name__}: {exc}"]

    state.replayed = {}
    for cid in CANONICAL:
        try:
            state.replayed[cid] = _replay(cid, state.records.get(cid, {"condition_id": cid, "slots": []}))
        except Exception as exc:
            state.replayed[cid] = _failure_sim(f"{cid}: replay raised {type(exc).__name__}: {exc}")
    try:
        legality = _legality_atoms(state.replayed)
    except Exception as exc:
        legality = _failed_atoms("decision_legality", f"legality check raised {type(exc).__name__}: {exc}")
    try:
        replay_atoms = _replay_atoms(state.replayed)
    except Exception as exc:
        replay_atoms = _failed_atoms("replay_validity", f"replay check raised {type(exc).__name__}: {exc}")

    state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
    state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
    state.scores["canonical_coverage"] = coverage
    state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
    state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
    state.diagnostics["result_json"] = state.diagnostics.get("result_json", _diagnose(shape))
    state.diagnostics["result_evidence"] = _diagnose(evidence)
    state.diagnostics["canonical_coverage"] = (
        f"defect_count={len(defects)}" + ("; " + " | ".join(defects[:12]) if defects else "")
    )
    state.diagnostics["decision_legality"] = _diagnose(legality)
    state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
    state.validity_gate = state.official_ok and all(
        state.scores.get(item, 0.0) >= 1.0 - EPS for item in GATE_CHECKS
    )
    if state.validity_gate:
        try:
            state.metrics = _gated_metrics(state.replayed, True)
        except Exception as exc:
            state.validity_gate = False
            state.scores["replay_validity"] = 0.0
            state.diagnostics["replay_validity"] = (
                "failed_atoms=[replay_validity.metrics]; "
                f"replay_validity.metrics: metric calculation raised {type(exc).__name__}: {exc}"
            )
    if not state.validity_gate:
        state.metrics = {key: 0.0 for key in METRIC_KEYS}
    if not state.official_ok:
        for item in GATE_CHECKS:
            state.scores[item] = 0.0
            state.diagnostics[item] = f"official_run failed; {state.official_message}; {state.diagnostics.get(item, '')}"
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
