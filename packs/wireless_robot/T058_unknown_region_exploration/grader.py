"""Independent replay grader for T058 unknown-region exploration."""

from __future__ import annotations

import asyncio
import json
import math
import re
import stat
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y, H = 24, 20, 180
ROBOTS = ("R0", "R1", "R2", "R3")
SCENARIOS = (
    "open_rooms",
    "narrow_tunnel",
    "deep_unknown",
    "uneven_traversability",
    "partition_burst",
)
ALLOC_SLOTS = tuple(range(0, 171, 10))
MESSAGE_SLOTS = tuple(range(0, 176, 5))
EIBS = tuple(f"E{i:02d}" for i in range(30))
EIB_SET = set(EIBS)
ACTIONS = {"stay": (0, 0), "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
STARTS = {
    "open_rooms": ((1, 1), (22, 1), (1, 18), (22, 18)),
    "narrow_tunnel": ((1, 9), (1, 10), (22, 9), (22, 10)),
    "deep_unknown": ((1, 1), (4, 1), (1, 4), (4, 4)),
    "uneven_traversability": ((1, 2), (22, 2), (1, 17), (22, 17)),
    "partition_burst": ((2, 2), (21, 2), (2, 17), (21, 17)),
}
DOORS = {
    "open_rooms": (((8, 8), 30), ((15, 10), 70)),
    "narrow_tunnel": (((6, 9), 24), ((12, 10), 62), ((18, 9), 104)),
    "deep_unknown": (((10, 15), 45), ((16, 4), 95)),
    "uneven_traversability": (),
    "partition_burst": (((11, 9), 50), ((12, 10), 110)),
}
OBSTACLES = {
    "open_rooms": ((8, 8, 0, 7), (8, 8, 10, 19), (15, 15, 0, 9), (15, 15, 12, 19)),
    "narrow_tunnel": (),
    "deep_unknown": ((10, 10, 0, 14), (16, 16, 5, 19)),
    "uneven_traversability": (),
    "partition_burst": ((11, 11, 0, 8), (11, 11, 11, 19), (12, 12, 0, 8), (12, 12, 11, 19)),
}
REPORT_SECTIONS = (
    "Problem Formulation",
    "Method Design",
    "Experimental Setup",
    "Results",
    "Research Analysis",
    "Conclusion and Limitations",
)
EIB_RE = re.compile(r"^E(?:0[0-9]|1[0-9]|2[0-9])$")

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.20,
        "result_json.scenarios": 0.25,
        "result_json.slots": 0.35,
    },
    "result_evidence": {
        "result_evidence.robot": 0.30,
        "result_evidence.allocations": 0.25,
        "result_evidence.messages": 0.25,
        "result_evidence.channel": 0.20,
    },
    "decision_legality": {
        "decision_legality.actions": 0.18,
        "decision_legality.routes": 0.22,
        "decision_legality.commitments": 0.25,
        "decision_legality.messages": 0.20,
        "decision_legality.channel": 0.15,
    },
    "replay_validity": {
        "replay_validity.snapshot": 0.40,
        "replay_validity.transport": 0.60,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = ident, bool(passed), detail, share


class EvalState:
    def __init__(self) -> None:
        self.run_started = False
        self.run_finished = False
        self.exit_code = None
        self.timed_out = False
        self.elapsed_s = None
        self.official_ok = False
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.parse_ok = False
        self.data = None
        self.records = {}
        self.replayed = {}
        self.scores = {}
        self.diagnostics = {}
        self.metrics = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value: Any, lo: int, hi: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate object key {key!r}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _obstacle(name: str, x: int, y: int) -> bool:
    if not (0 <= x < GRID_X and 0 <= y < GRID_Y):
        return True
    if name == "narrow_tunnel":
        side = (0 <= x <= 3 and 8 <= y <= 11) or (20 <= x <= 23 and 8 <= y <= 11)
        side = side or (0 <= x <= 3 and 0 <= y <= 3) or (20 <= x <= 23 and 16 <= y <= 19)
        return not (8 <= y <= 11 or side)
    if name == "uneven_traversability":
        if any(max(abs(x - sx), abs(y - sy)) <= 3 for sx, sy in STARTS[name]):
            return False
        return (7 * x + 5 * y) % 19 == 0
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in OBSTACLES[name])


def _true_state(name: str, x: int, y: int, slot: int) -> int:
    if not (0 <= x < GRID_X and 0 <= y < GRID_Y):
        return 2
    if name == "deep_unknown" and any(
        max(abs(x - sx), abs(y - sy)) <= 3 for sx, sy in STARTS[name]
    ):
        return 1
    if _obstacle(name, x, y):
        return 2
    for cell, opening in DOORS[name]:
        if (x, y) == cell and slot < opening:
            return 2
    return 1


def _eib_id(x: int, y: int) -> str:
    return f"E{(x // 4) + 6 * (y // 4):02d}"


def _eib_cells(eib: str) -> list[tuple[int, int]]:
    n = int(eib[1:])
    ox, oy = (n % 6) * 4, (n // 6) * 4
    return [(ox + dx, oy + dy) for dy in range(4) for dx in range(4)]


def _line_clear(name: str, a: tuple[int, int], b: tuple[int, int], slot: int) -> bool:
    steps = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    if steps == 0:
        return True
    for i in range(steps + 1):
        x = round(a[0] + (b[0] - a[0]) * i / steps)
        y = round(a[1] + (b[1] - a[1]) * i / steps)
        if _true_state(name, x, y, slot) == 2:
            return False
    return True


def _interference(name: str, slot: int) -> int:
    if name == "partition_burst" and (35 <= slot <= 74 or 120 <= slot <= 144):
        return 2
    if name == "uneven_traversability" and 60 <= slot <= 99:
        return 1
    if name == "open_rooms":
        return 0
    return 1


def _probe(
    name: str, sender: str, receiver: str, positions: Mapping[str, tuple[int, int]], slot: int
) -> tuple[bool, int, int]:
    if sender == receiver:
        return False, 0, _interference(name, slot)
    a, b = positions[sender], positions[receiver]
    distance = math.dist(a, b)
    level = _interference(name, slot)
    available = distance <= 10 and _line_clear(name, a, b, slot)
    return available, (192, 96, 64)[level] if available else 0, level


def _encode_states(states: Sequence[int]) -> int:
    out = 0
    for value in states:
        out = out * 3 + int(value)
    return out


def _decode_states(code: int) -> list[int]:
    vals = [0] * 16
    n = int(code)
    for i in range(15, -1, -1):
        vals[i] = n % 3
        n //= 3
    return vals


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _reachable_frontier(
    local: Mapping[tuple[int, int], int], start: tuple[int, int], eib: str
) -> tuple[bool, int | None]:
    q = deque([start])
    dist = {start: 0}
    targets: list[int] = []
    cells = set(_eib_cells(eib))
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur):
            if nxt in dist or local.get(nxt) != 1:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    for cell in cells:
        if local.get(cell, 0) != 0:
            continue
        if any(local.get(nxt) == 1 and nxt in dist for nxt in _neighbors(cell)):
            targets.append(
                min(dist[nxt] for nxt in _neighbors(cell) if local.get(nxt) == 1 and nxt in dist)
            )
    return bool(targets), (min(targets) if targets else None)


def _snapshot(
    local: Mapping[tuple[int, int], int],
    start: tuple[int, int],
    eib: str,
    versions: Mapping[str, int],
) -> dict[str, Any]:
    cells = _eib_cells(eib)
    vals = [local.get(cell, 0) for cell in cells]
    explorable, cost = _reachable_frontier(local, start, eib)
    known = [v for v in vals if v != 0]
    if explorable:
        state = "explorable"
    elif not any(v == 0 for v in vals) and any(v == 1 for v in vals):
        state = "traversable_explored"
    elif not any(v == 0 for v in vals):
        state = "blocked_explored"
    else:
        state = "merged"
    return {
        "eib_id": eib,
        "code": _encode_states(vals),
        "version": int(versions[eib]),
        "explorable": explorable,
        "estimated_cost": cost,
        "state": state,
        "known": known,
    }


def _records(data: Any) -> dict[tuple[str, int, str], dict[str, Any]]:
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return out
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict):
            continue
        name = scenario.get("scenario")
        if not isinstance(name, str):
            continue
        for slot in scenario.get("slots", []) if isinstance(scenario.get("slots"), list) else []:
            if not isinstance(slot, dict):
                continue
            t = slot.get("slot")
            if not isinstance(t, int) or isinstance(t, bool):
                continue
            for decision in (
                slot.get("robot_decisions", [])
                if isinstance(slot.get("robot_decisions"), list)
                else []
            ):
                if isinstance(decision, dict) and isinstance(decision.get("robot_id"), str):
                    out[(name, t, decision["robot_id"])] = {"slot": slot, "decision": decision}
            out[(name, t, "__slot__")] = {"slot": slot}
    return out


def _result_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    checks = {key: [True, "ok"] for key in ATOM_SHARES["result_json"]}
    if not parse_ok:
        checks["result_json.parse"] = [False, "strict JSON parse failed"]
    if not isinstance(data, dict) or set(data) != {"scenarios"}:
        checks["result_json.root"] = [False, "root must contain exactly scenarios"]
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        checks["result_json.scenarios"] = [False, "scenarios must be an array"]
    if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
        for item in data["scenarios"]:
            if not isinstance(item, dict) or set(item) != {"scenario", "slots"}:
                checks["result_json.scenarios"] = [False, "scenario object keys are invalid"]
                break
        if len(data["scenarios"]) != 5:
            checks["result_json.scenarios"] = [False, "expected five scenarios"]
        for item in data["scenarios"]:
            if isinstance(item, dict) and not isinstance(item.get("slots"), list):
                checks["result_json.slots"] = [False, "scenario slots must be an array"]
                break
            if isinstance(item, dict) and isinstance(item.get("slots"), list):
                for row in item["slots"]:
                    if not isinstance(row, dict) or set(row) != {
                        "slot",
                        "robot_decisions",
                        "allocations",
                        "messages",
                        "channel",
                    }:
                        checks["result_json.slots"] = [False, "slot object keys are invalid"]
                        break
                if not checks["result_json.slots"][0]:
                    break
    return [
        AtomVerdict(key, value[0], value[1], ATOM_SHARES["result_json"][key])
        for key, value in checks.items()
    ]


def _evidence_atoms(records: Mapping[tuple[str, int, str], dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key: [True, "ok"] for key in ATOM_SHARES["result_evidence"]}
    if not records:
        checks["result_evidence.robot"] = [False, "no indexed records"]
    for (name, t, suffix), row in records.items():
        if suffix == "__slot__":
            slot = row["slot"]
            if not isinstance(slot, dict) or set(slot) != {
                "slot",
                "robot_decisions",
                "allocations",
                "messages",
                "channel",
            }:
                checks["result_evidence.channel"] = [False, f"{name}/{t}: slot keys invalid"]
            continue
        d = row["decision"]
        if set(d) != {"robot_id", "action"} or not isinstance(d.get("action"), str):
            checks["result_evidence.robot"] = [
                False,
                f"{name}/{t}/{suffix}: robot decision keys or action type invalid",
            ]
        slot = row["slot"]
        for key in ("allocations", "messages"):
            if not isinstance(slot.get(key), list):
                checks[f"result_evidence.{key}"] = [False, f"{name}/{t}: {key} must be an array"]
    return [
        AtomVerdict(key, value[0], value[1], ATOM_SHARES["result_evidence"][key])
        for key, value in checks.items()
    ]


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["missing scenarios"]
    seen_s: set[str] = set()
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict):
            defects.append("invalid scenario object")
            continue
        name = scenario.get("scenario")
        if name not in SCENARIOS:
            defects.append(f"unknown scenario {name!r}")
            continue
        if name in seen_s:
            defects.append(f"duplicate scenario {name}")
        seen_s.add(name)
        slots = scenario.get("slots")
        if not isinstance(slots, list):
            defects.append(f"{name}: missing slots")
            continue
        seen_t: set[int] = set()
        for row in slots:
            t = row.get("slot") if isinstance(row, dict) else None
            if not _integer(t, 0, H - 1):
                defects.append(f"{name}: invalid slot {t!r}")
                continue
            if t in seen_t:
                defects.append(f"{name}: duplicate slot {t}")
            seen_t.add(t)
            decisions = row.get("robot_decisions") if isinstance(row, dict) else None
            if not isinstance(decisions, list):
                defects.append(f"{name}/{t}: missing robot decisions")
                continue
            ids = [d.get("robot_id") for d in decisions if isinstance(d, dict)]
            if set(ids) != set(ROBOTS) or len(ids) != 4:
                defects.append(f"{name}/{t}: robot key coverage")
        defects.extend(f"{name}: missing slot {t}" for t in set(range(H)) - seen_t)
    defects.extend(f"missing scenario {name}" for name in set(SCENARIOS) - seen_s)
    defects.extend(
        f"duplicate scenario {name}"
        for name in seen_s
        if sum(1 for s in data["scenarios"] if isinstance(s, dict) and s.get("scenario") == name)
        > 1
    )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _legality_atoms(records: Mapping[tuple[str, int, str], dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key: [True, "ok"] for key in ATOM_SHARES["decision_legality"]}
    for (name, t, suffix), row in records.items():
        if suffix == "__slot__":
            slot = row["slot"]
            if not isinstance(slot, dict):
                continue
            decisions = slot.get("robot_decisions", [])
            if isinstance(decisions, list):
                for d in decisions:
                    if (
                        not isinstance(d, dict)
                        or not isinstance(d.get("action"), str)
                        or d.get("action") not in ACTIONS
                    ):
                        checks["decision_legality.actions"] = [False, f"{name}/{t}: unknown action"]
            allocs = slot.get("allocations")
            if not isinstance(allocs, list):
                checks["decision_legality.routes"] = [False, f"{name}/{t}: allocations not array"]
            expected = 4 if t in ALLOC_SLOTS else 0
            if isinstance(allocs, list) and len(allocs) != expected:
                checks["decision_legality.routes"] = [
                    False,
                    f"{name}/{t}: expected {expected} allocation records",
                ]
            if isinstance(allocs, list):
                allocation_ids = [a.get("robot_id") for a in allocs if isinstance(a, dict)]
                if t in ALLOC_SLOTS and (
                    set(allocation_ids) != set(ROBOTS) or len(allocation_ids) != 4
                ):
                    checks["decision_legality.routes"] = [
                        False,
                        f"{name}/{t}: allocation robot coverage invalid",
                    ]
                for a in allocs:
                    if not isinstance(a, dict) or set(a) != {
                        "robot_id",
                        "route_eibs",
                        "commitments",
                    }:
                        checks["decision_legality.routes"] = [
                            False,
                            f"{name}/{t}: allocation keys invalid",
                        ]
                        continue
                    if (
                        not isinstance(a.get("robot_id"), str)
                        or a.get("robot_id") not in ROBOTS
                    ):
                        checks["decision_legality.routes"] = [
                            False,
                            f"{name}/{t}: invalid allocation robot",
                        ]
                    route = a.get("route_eibs")
                    if (
                        not isinstance(route, list)
                        or len(route) > 3
                        or any(not isinstance(x, str) or x not in EIB_SET for x in route)
                        or len(route) != len(set(route))
                    ):
                        checks["decision_legality.routes"] = [False, f"{name}/{t}: invalid route"]
                    commitments = a.get("commitments")
                    if not isinstance(commitments, list):
                        checks["decision_legality.commitments"] = [
                            False,
                            f"{name}/{t}: commitments not array",
                        ]
                        continue
                    commitment_ids = [
                        c.get("eib_id") for c in commitments if isinstance(c, dict)
                    ]
                    if len(commitment_ids) != len(
                        {x for x in commitment_ids if isinstance(x, str)}
                    ):
                        checks["decision_legality.commitments"] = [
                            False,
                            f"{name}/{t}: duplicate commitment EIB",
                        ]
                    for c in commitments:
                        if (
                            not isinstance(c, dict)
                            or set(c) != {"eib_id", "lease", "version", "cost_q"}
                            or c.get("eib_id") not in route
                            or not _integer(c.get("lease"), 1, 20)
                            or not _integer(c.get("version"), 0, 100000)
                            or not _integer(c.get("cost_q"), 0, 255)
                        ):
                            checks["decision_legality.commitments"] = [
                                False,
                                f"{name}/{t}: invalid commitment",
                            ]
            messages = slot.get("messages")
            if not isinstance(messages, list) or len(messages) != (4 if t in MESSAGE_SLOTS else 0):
                checks["decision_legality.messages"] = [
                    False,
                    f"{name}/{t}: message opportunity shape invalid",
                ]
            if isinstance(messages, list):
                message_ids = [m.get("robot_id") for m in messages if isinstance(m, dict)]
                if t in MESSAGE_SLOTS and (
                    set(message_ids) != set(ROBOTS) or len(message_ids) != 4
                ):
                    checks["decision_legality.messages"] = [
                        False,
                        f"{name}/{t}: message robot coverage invalid",
                    ]
                for m in messages:
                    if not isinstance(m, dict) or set(m) != {
                        "robot_id",
                        "eib_ids",
                        "commitment_eibs",
                    }:
                        checks["decision_legality.messages"] = [
                            False,
                            f"{name}/{t}: message keys invalid",
                        ]
                        continue
                    eids, cids = m.get("eib_ids"), m.get("commitment_eibs")
                    if (
                        not isinstance(eids, list)
                        or not isinstance(cids, list)
                        or any(
                            not isinstance(x, str) or x not in EIB_SET for x in eids + cids
                        )
                        or len(eids) != len(set(eids))
                        or len(cids) != len(set(cids))
                        or len(eids) + len(cids) > 5
                    ):
                        checks["decision_legality.messages"] = [
                            False,
                            f"{name}/{t}: invalid message sets",
                        ]
            channel = slot.get("channel")
            if (
                not isinstance(channel, dict)
                or set(channel) != {"sender"}
                or (channel.get("sender") is not None and channel.get("sender") not in ROBOTS)
            ):
                checks["decision_legality.channel"] = [False, f"{name}/{t}: invalid channel sender"]
    return [
        AtomVerdict(key, value[0], value[1], ATOM_SHARES["decision_legality"][key])
        for key, value in checks.items()
    ]


class Replay:
    def __init__(self, name: str, rows: Mapping[int, dict[str, Any]]) -> None:
        self.name, self.rows = name, rows
        self.pos = {r: STARTS[name][i] for i, r in enumerate(ROBOTS)}
        self.maps = {r: {} for r in ROBOTS}
        self.versions = {r: {e: 0 for e in EIBS} for r in ROBOTS}
        # Sender versions and receiver-local versions are different counters.
        # Keep a source watermark for each (sender, EIB) at every receiver.
        self.source_versions = {
            receiver: {
                sender: {eib: -1 for eib in EIBS}
                for sender in ROBOTS
                if sender != receiver
            }
            for receiver in ROBOTS
        }
        self.routes = {r: [] for r in ROBOTS}
        self.claims = {r: {} for r in ROBOTS}
        self.owners = {r: {} for r in ROBOTS}
        self.pending: dict[str, dict[str, Any]] = {}
        self.arrivals: dict[int, list[dict[str, Any]]] = {}
        self.served_bits = 0
        self.safety = 0
        self.executed = 0
        self.discovered: set[tuple[int, int]] = set()
        self.per_robot = {r: set() for r in ROBOTS}
        self.first_discovery = {r: 0 for r in ROBOTS}
        self.coverage_series: list[tuple[int, int]] = []
        self.all_open_series: list[int] = []
        self.route_pairs = [0, 0]
        self.useful_eib_bits = 0
        self.eib_bits = 0
        self.useful_commitments = 0
        self.commitment_records = 0
        self.accepted_eib_ages: list[int] = []
        self.accepted_commit_ages: list[int] = []
        self.productive_packets: set[str] = set()
        self.packet_bits: dict[str, int] = {}
        self.packet_useful: dict[str, bool] = {}
        self.local_commitments = {robot: {} for robot in ROBOTS}
        self.replay_errors = {"snapshot": [], "transport": []}
        self.prev_epoch_route = {r: None for r in ROBOTS}
        self.prev_epoch_explorable = {r: set() for r in ROBOTS}
        if name == "deep_unknown":
            for robot, start in zip(ROBOTS, STARTS[name]):
                for cell in self._radius(start, 3):
                    self.maps[robot][cell] = _true_state(name, cell[0], cell[1], 0)

    def _observe(self, robot: str, slot: int) -> None:
        x, y = self.pos[robot]
        for yy in range(max(0, y - 2), min(GRID_Y, y + 3)):
            for xx in range(max(0, x - 2), min(GRID_X, x + 3)):
                cell = (xx, yy)
                state = _true_state(self.name, xx, yy, slot)
                old = self.maps[robot].get(cell, 0)
                new = 2 if state == 2 or old == 2 else state
                if old != new:
                    self.maps[robot][cell] = new
                    self.versions[robot][_eib_id(xx, yy)] += 1
                if state == 1:
                    if cell not in self.per_robot[robot]:
                        self.per_robot[robot].add(cell)
                    if cell not in self.discovered:
                        self.discovered.add(cell)
                        self.first_discovery[robot] += 1

    def _expire(self, slot: int) -> None:
        for pid, packet in list(self.pending.items()):
            if slot - packet["created"] > 30:
                packet["expired"] = True
                del self.pending[pid]
        for claims in self.claims.values():
            for eib, items in list(claims.items()):
                claims[eib] = [item for item in items if slot < item["expiry"]]

    def _apply_arrivals(self, slot: int) -> None:
        for arrival in self.arrivals.pop(slot, []):
            if slot - arrival["created"] > 30:
                continue
            receiver, packet = arrival["receiver"], arrival["packet"]
            if packet.get("expired"):
                continue
            for eib, code, version in packet["eib_records"]:
                self.eib_bits += 32
                sender = packet["sender"]
                if version <= self.source_versions[receiver][sender][eib]:
                    continue
                self.source_versions[receiver][sender][eib] = version
                vals = _decode_states(code)
                changed = False
                for cell, incoming in zip(_eib_cells(eib), vals):
                    old = self.maps[receiver].get(cell, 0)
                    merged = 2 if old == 2 or incoming == 2 else max(old, incoming)
                    if merged != old:
                        self.maps[receiver][cell] = merged
                        self.versions[receiver][eib] += 1
                        changed = True
                self.accepted_eib_ages.append(slot - packet["created"])
                if changed:
                    self.useful_eib_bits += 32
                    self.packet_useful[packet["id"]] = True
                    self.productive_packets.add(packet["id"])
            for eib, lease, version, cost, epoch in packet["commitment_records"]:
                if slot >= epoch + lease:
                    continue
                self.commitment_records += 1
                sender = packet["sender"]
                item = {
                    "sender": sender,
                    "cost": cost,
                    "version": version,
                    "expiry": epoch + lease,
                    "created": packet["created"],
                }
                self.claims[receiver].setdefault(eib, []).append(item)
                before = self.owners[receiver].get(eib)
                active = [q for q in self.claims[receiver][eib] if slot < q["expiry"]]
                best = (
                    min(active, key=lambda q: (q["cost"], -q["version"], q["sender"]))
                    if active
                    else None
                )
                self.owners[receiver][eib] = best["sender"] if best else None
                if best and best["sender"] != before:
                    self.useful_commitments += 1
                    self.packet_useful[packet["id"]] = True
                    self.productive_packets.add(packet["id"])
                self.accepted_commit_ages.append(slot - packet["created"])

    def _generate(self, slot: int, rows: dict[str, Any]) -> None:
        for robot in ROBOTS:
            m = rows[robot].get("messages")
            if not isinstance(m, dict):
                continue
            eids, cids = m.get("eib_ids", []), m.get("commitment_eibs", [])
            if not eids and not cids:
                continue
            pid = f"P_{robot}_{slot}"
            records = []
            for eib in eids:
                snap = _snapshot(self.maps[robot], self.pos[robot], eib, self.versions[robot])
                records.append((eib, snap["code"], snap["version"]))
            commitments = []
            for eib in cids:
                claim = self.local_commitments[robot].get(eib)
                if claim is not None:
                    commitments.append(
                        (eib, claim["lease"], claim["version"], claim["cost_q"], claim["epoch"])
                    )
                else:
                    self.replay_errors["snapshot"].append(
                        f"{self.name}/{slot}/{robot}: message references a non-current commitment"
                    )
            bits = 32 + 32 * (len(records) + len(commitments))
            self.packet_bits[pid] = bits
            if sum(1 for packet in self.pending.values() if packet["sender"] == robot) >= 5:
                continue
            self.pending[pid] = {
                "id": pid,
                "sender": robot,
                "created": slot,
                "payload": bits,
                "remaining": bits,
                "eib_records": records,
                "commitment_records": commitments,
                "pending_receivers": set(ROBOTS) - {robot},
            }

    def _serve(self, slot: int, sender: str | None) -> None:
        if sender is None:
            return
        candidates = [
            p for p in self.pending.values() if p["sender"] == sender and not p.get("expired")
        ]
        if not candidates:
            self.replay_errors["transport"].append(
                f"{self.name}/{slot}: sender {sender} has no FIFO head"
            )
            return
        packet = min(candidates, key=lambda p: (p["created"], p["id"]))
        level = _interference(self.name, slot)
        capacity = (192, 96, 64)[level]
        if capacity <= 0:
            self.replay_errors["transport"].append(
                f"{self.name}/{slot}: sender {sender} has zero capacity"
            )
            return
        if self.served_bits >= 14400:
            self.replay_errors["transport"].append(
                f"{self.name}/{slot}: scene service cap exhausted"
            )
            return
        served = min(capacity, packet["remaining"], max(0, 14400 - self.served_bits))
        self.served_bits += served
        packet["remaining"] -= served
        if packet["remaining"] > 0:
            return
        for receiver in list(packet["pending_receivers"]):
            ok, _cap, lev = _probe(self.name, sender, receiver, self.pos, slot)
            if not ok:
                packet["pending_receivers"].remove(receiver)
                continue
            i, j = ROBOTS.index(sender), ROBOTS.index(receiver)
            success = ((13 * i + 7 * j + 5 * slot) % 100) / 100 >= 0.05 + 0.12 * lev
            if success:
                delay = min(
                    3, 1 + int(math.floor(math.dist(self.pos[sender], self.pos[receiver]) / 6))
                )
                self.arrivals.setdefault(slot + delay, []).append(
                    {"packet": packet, "receiver": receiver, "created": packet["created"]}
                )
                packet["pending_receivers"].remove(receiver)
        if not packet["pending_receivers"]:
            self.pending.pop(packet["id"], None)
        else:
            packet["remaining"] = packet["payload"]

    def _alloc(self, slot: int, rows: dict[str, Any]) -> None:
        for robot in ROBOTS:
            a = rows[robot].get("alloc")
            if not isinstance(a, dict):
                continue
            route = a.get("route_eibs", [])
            snapshots = {}
            for eib in EIBS:
                snap = _snapshot(self.maps[robot], self.pos[robot], eib, self.versions[robot])
                snapshots[eib] = snap
            valid_route = [e for e in route if e in EIB_SET]
            explorable = {
                eib for eib, snap in snapshots.items() if snap["explorable"]
            }
            old = self.prev_epoch_route[robot]
            if old is not None:
                self.route_pairs[1] += 1
                if valid_route and old and valid_route[0] == old[0]:
                    self.route_pairs[0] += 1
                elif old and old[0] not in self.prev_epoch_explorable[robot]:
                    self.route_pairs[0] += 1
            self.prev_epoch_route[robot] = valid_route
            self.prev_epoch_explorable[robot] = explorable
            self.routes[robot] = valid_route
            claims = {}
            for c in a.get("commitments", []) if isinstance(a.get("commitments"), list) else []:
                if c.get("eib_id") not in snapshots:
                    continue
                if (
                    c.get("eib_id") in valid_route
                    and c.get("version") == snapshots[c["eib_id"]]["version"]
                    and snapshots[c["eib_id"]]["estimated_cost"] is not None
                    and c.get("cost_q") == min(255, snapshots[c["eib_id"]]["estimated_cost"])
                ):
                    claims[c["eib_id"]] = {
                        "lease": c["lease"],
                        "version": c["version"],
                        "cost_q": c["cost_q"],
                        "epoch": slot,
                    }
                else:
                    self.replay_errors["snapshot"].append(
                        f"{self.name}/{slot}/{robot}: commitment does not match the local snapshot"
                    )
            rows[robot]["commitments"] = claims
            self.local_commitments[robot] = claims

    def _move(self, slot: int, rows: dict[str, Any]) -> None:
        requested = {r: rows[r].get("action", "stay") for r in ROBOTS}
        targets = {}
        blocked_requests: set[str] = set()
        for r, action in requested.items():
            dx, dy = ACTIONS.get(action, (0, 0))
            p = (self.pos[r][0] + dx, self.pos[r][1] + dy)
            if action != "stay" and _true_state(self.name, *p, slot) == 2:
                p = self.pos[r]
                blocked_requests.add(r)
            targets[r] = p
        rejected = set()
        for a in ROBOTS:
            for b in ROBOTS:
                if (
                    a < b
                    and targets[a] == self.pos[b]
                    and targets[b] == self.pos[a]
                    and targets[a] != self.pos[a]
                ):
                    rejected.update((a, b))
        for target in set(targets.values()):
            contenders = [
                r
                for r in ROBOTS
                if targets[r] == target and targets[r] != self.pos[r] and r not in rejected
            ]
            if len(contenders) > 1:
                winner = min(
                    contenders,
                    key=lambda r: (
                        sum(1 for cell in self._radius(target, 2) if cell not in self.discovered),
                        r,
                    ),
                )
                rejected.update(set(contenders) - {winner})
        changed = True
        while changed:
            changed = False
            for r in ROBOTS:
                if r in rejected or targets[r] == self.pos[r]:
                    continue
                if any(q != r and q not in rejected and targets[q] == self.pos[r] for q in ROBOTS):
                    continue
                if any(q != r and q in rejected and self.pos[q] == targets[r] for q in ROBOTS):
                    rejected.add(r)
                    changed = True
        for r in ROBOTS:
            if requested[r] != "stay" and (r in rejected or r in blocked_requests):
                self.safety += 1
            if requested[r] != "stay" and r not in rejected and targets[r] != self.pos[r]:
                self.pos[r] = targets[r]
                self.executed += 1

    @staticmethod
    def _radius(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(center[0] - radius, center[0] + radius + 1)
            for y in range(center[1] - radius, center[1] + radius + 1)
        ]

    def run(self) -> dict[str, Any]:
        for slot in range(H):
            self._apply_arrivals(slot)
            self._expire(slot)
            rows = self.rows.get(slot, {})
            for robot in ROBOTS:
                self._observe(robot, slot)
            if slot in ALLOC_SLOTS:
                self._alloc(slot, rows)
            if slot in MESSAGE_SLOTS:
                self._generate(slot, rows)
            sender = rows.get("__channel__")
            self._serve(slot, sender)
            self._move(slot, rows)
            open_count = sum(
                _true_state(self.name, x, y, slot) == 1
                for x in range(GRID_X)
                for y in range(GRID_Y)
            )
            self.all_open_series.append(open_count)
            self.coverage_series.append((len(self.discovered), open_count))
        self._apply_arrivals(H)
        self._expire(H)
        return {
            "discovered": self.discovered,
            "per_robot": self.per_robot,
            "first": self.first_discovery,
            "coverage": self.coverage_series,
            "open": self.all_open_series,
            "executed": self.executed,
            "safety": self.safety,
            "route_pairs": tuple(self.route_pairs),
            "useful_eib_bits": self.useful_eib_bits,
            "eib_bits": self.eib_bits,
            "useful_commitments": self.useful_commitments,
            "commitment_records": self.commitment_records,
            "eib_ages": self.accepted_eib_ages,
            "commit_ages": self.accepted_commit_ages,
            "productive_bits": sum(self.packet_bits.get(pid, 0) for pid in self.productive_packets),
            "served_bits": self.served_bits,
            "replay_errors": self.replay_errors,
        }


def _prepare_rows(
    records: Mapping[tuple[str, int, str], dict[str, Any]], name: str
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for (scenario, t, suffix), item in records.items():
        if scenario != name or suffix != "__slot__":
            continue
        slot = item["slot"]
        decisions = {}
        for d in (
            slot.get("robot_decisions", []) if isinstance(slot.get("robot_decisions"), list) else []
        ):
            if isinstance(d, dict) and d.get("robot_id") in ROBOTS:
                decisions[d["robot_id"]] = {"action": d.get("action")}
        for r in ROBOTS:
            decisions.setdefault(r, {"action": "stay"})
        if t in ALLOC_SLOTS:
            for a in (
                slot.get("allocations", []) if isinstance(slot.get("allocations"), list) else []
            ):
                if isinstance(a, dict) and a.get("robot_id") in ROBOTS:
                    decisions[a["robot_id"]]["alloc"] = a
        if t in MESSAGE_SLOTS:
            for m in slot.get("messages", []) if isinstance(slot.get("messages"), list) else []:
                if isinstance(m, dict) and m.get("robot_id") in ROBOTS:
                    decisions[m["robot_id"]]["messages"] = m
        channel = slot.get("channel", {})
        rows[t] = decisions
        rows[t]["__channel__"] = channel.get("sender") if isinstance(channel, dict) else None
    return rows


def _replay_atoms(
    records: Mapping[tuple[str, int, str], dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    errors = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}
    replayed: dict[str, dict[str, Any]] = {}
    for name in SCENARIOS:
        try:
            result = Replay(name, _prepare_rows(records, name)).run()
            replayed[name] = result
            for key, messages in result["replay_errors"].items():
                errors[key].extend(messages)
            # Invalid local commitments and unknown message IDs are diagnosed
            # against the replay snapshot.
            for (scenario, t, suffix), item in records.items():
                if scenario != name or suffix != "__slot__":
                    continue
                slot = item["slot"]
                if t in ALLOC_SLOTS:
                    for a in (
                        slot.get("allocations", [])
                        if isinstance(slot.get("allocations"), list)
                        else []
                    ):
                        if not isinstance(a, dict) or a.get("robot_id") not in ROBOTS:
                            continue
                        if any(
                            c.get("eib_id") not in a.get("route_eibs", [])
                            for c in a.get("commitments", [])
                            if isinstance(c, dict)
                        ):
                            errors["snapshot"].append(f"{name}/{t}: commitment not on route")
                if t in MESSAGE_SLOTS:
                    for m in (
                        slot.get("messages", []) if isinstance(slot.get("messages"), list) else []
                    ):
                        if (
                            isinstance(m, dict)
                            and len(m.get("eib_ids", [])) + len(m.get("commitment_eibs", [])) > 5
                        ):
                            errors["transport"].append(f"{name}/{t}: packet exceeds five records")
        except Exception as exc:
            errors["snapshot"].append(f"{name}: replay exception {type(exc).__name__}: {exc}")
    atoms = [
        AtomVerdict(
            f"replay_validity.{key}",
            not value,
            "ok" if not value else " | ".join(value[:8]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{key}"],
        )
        for key, value in errors.items()
    ]
    return atoms, replayed


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declaration")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("invalid atom shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0, abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share mismatch")
    return _clip(sum(a.share for a in atoms if a.passed))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    if not failed:
        return "failed_atoms=[]"
    return (
        "failed_atoms=["
        + ",".join(a.id for a in failed)
        + "]; "
        + " | ".join(f"{a.id}: {a.detail}" for a in failed)
    )


def _metrics(
    replayed: Mapping[str, dict[str, Any]],
    _records_arg: Mapping[tuple[str, int, str], dict[str, Any]] | None = None,
) -> dict[str, float]:
    vals = {
        k: []
        for k in (
            "primary_metrics_1",
            "primary_metrics_2",
            "primary_metrics_3",
            "auxiliary_metrics_1",
            "auxiliary_metrics_2",
        )
    }
    for name in SCENARIOS:
        r = replayed[name]
        final_cov = len(r["discovered"]) / max(1, r["open"][-1])
        auc = sum(c / max(1, n) for c, n in r["coverage"]) / H
        p1 = _clip(0.7 * final_cov + 0.3 * auc)
        union = set().union(*r["per_robot"].values())
        total = sum(len(v) for v in r["per_robot"].values())
        overlap = (total / max(1, len(union)) - 1) / 3 if union else 0.0
        shares = [r["first"][robot] / max(1, len(r["discovered"])) for robot in ROBOTS]
        mean = sum(shares) / 4
        sigma = math.sqrt(sum((x - mean) ** 2 for x in shares) / 4)
        balance = _clip(1 - sigma / math.sqrt(3)) if union else 0.0
        p2 = _clip(0.55 * (1 - overlap) + 0.45 * balance) if union else 0.0
        yield_score = _clip(len(r["discovered"]) / max(1, r["executed"]))
        route = r["route_pairs"][0] / r["route_pairs"][1] if r["route_pairs"][1] else 0.0
        p3 = _clip(0.4 * yield_score + 0.3 * route + 0.3 * _clip(1 - r["safety"] / (4 * H)))
        useful_ratio = r["useful_eib_bits"] / max(1, r["eib_bits"])
        commit_ratio = r["useful_commitments"] / max(1, r["commitment_records"])
        a1 = _clip(0.65 * useful_ratio + 0.35 * commit_ratio)
        ef = (
            sum(_clip(1 - age / 30) for age in r["eib_ages"]) / len(r["eib_ages"])
            if r["eib_ages"]
            else 0.0
        )
        cf = (
            sum(_clip(1 - age / 30) for age in r["commit_ages"]) / len(r["commit_ages"])
            if r["commit_ages"]
            else 0.0
        )
        a2 = _clip(0.45 * ef + 0.35 * cf + 0.2 * r["productive_bits"] / max(1, r["served_bits"]))
        for key, value in zip(vals, (p1, p2, p3, a1, a2)):
            vals[key].append(_clip(value))
    return {key: _clip(sum(value) / len(value)) for key, value in vals.items()}


async def _build_state(ws: Workspace) -> EvalState:
    root = str(ws.root)
    if root in _CACHE:
        return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE:
            return _CACHE[root]
        state, path = EvalState(), Path(ws.root, "results.json")
        try:
            await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            state.run_started = True
            started = time.monotonic()
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.run_finished = True
            state.elapsed_s = time.monotonic() - started
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = (
                state.exit_code is None
                or "timeout" in stderr.lower()
                or "timed out" in stderr.lower()
                or state.elapsed_s >= 600.0 - EPS
            )
            info = path.lstat()
            state.result_file_is_new_regular = stat.S_ISREG(info.st_mode) and not path.is_symlink()
            state.official_ok = (
                state.exit_code == 0
                and not state.timed_out
                and state.elapsed_s <= 600.0 + EPS
                and state.result_file_is_new_regular
            )
            state.official_message = (
                f"exit_code={state.exit_code}, elapsed_s={state.elapsed_s:.6f}, "
                f"fresh_regular_result={state.result_file_is_new_regular}"
            )
            if state.official_ok:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
        except Exception as exc:
            state.official_message = f"{type(exc).__name__}: {exc}"
            state.timed_out = (
                isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
        state.diagnostics["official"] = state.official_message or "official run failed"
        atoms = _result_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = _diagnose_atoms(atoms)
        if not state.official_ok or not state.parse_ok:
            for key in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[key] = 0.0
                state.diagnostics[key] = (
                    f"{key}: not evaluated because {state.diagnostics['official']}"
                )
            _CACHE[root] = state
            return state
        state.records = _records(state.data)
        atoms = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(atoms)
        state.scores["canonical_coverage"], defects = _coverage(state.data)
        state.diagnostics["canonical_coverage"] = f"defects={defects}"
        atoms = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(atoms)
        atoms, state.replayed = _replay_atoms(state.records)
        state.scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(k, 0.0) >= 1 - EPS
            for k in (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            )
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.records)
        _CACHE[root] = state
        return state


def _report_score(text: str) -> tuple[float, list[str]]:
    missing = []
    for section in REPORT_SECTIONS:
        match = re.search(rf"^# {re.escape(section)}\s*$", text, re.M)
        if not match:
            missing.append(section)
            continue
        end = re.search(r"^# ", text[match.end() :], re.M)
        body = text[match.end() : match.end() + (end.start() if end else len(text))]
        if not body.strip():
            missing.append(section)
    return max(0.0, (2.0 - 0.5 * len(missing)) / 2.0), missing


async def _gate(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(key, 0.0), state.diagnostics.get(key, "")


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "run.py")
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
        regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except Exception as exc:
        return 0.0, f"run.py compilation failed: {type(exc).__name__}: {exc}"
    return (
        (1.0, "run.py exists and compiles")
        if regular and getattr(result, "exit_code", 1) == 0
        else (0.0, "run.py is missing or does not compile")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (
        (1.0, "official run passed")
        if state.official_ok
        else (0.0, state.diagnostics.get("official", "official run failed"))
    )


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "report.md")
    try:
        regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
        text = ws.read_text("report.md") if regular else ""
    except OSError:
        regular, text = False, ""
    score, missing = _report_score(text or "")
    return score, f"missing_or_empty={missing}"


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]:
    return await _gate(ws, "result_json")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]:
    return await _gate(ws, "result_evidence")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace) -> tuple[float, str]:
    return await _gate(ws, "canonical_coverage")


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]:
    return await _gate(ws, "decision_legality")


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]:
    return await _gate(ws, "replay_validity")


def _metric_outcome(metrics: Mapping[str, float] | None, key: str) -> tuple[float, str]:
    return (
        (0.0, f"{key}: validity gate not satisfied")
        if metrics is None
        else (metrics[key], f"{key}: replayed={metrics[key]:.9g}")
    )


async def _metric(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return _metric_outcome(state.metrics, key)


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
