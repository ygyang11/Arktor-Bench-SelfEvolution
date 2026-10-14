# ruff: noqa
"""Independent replay grader for T055.

The replay deliberately treats candidate actions, operations and packet choices as the
submitted decisions.  Truth that is not observable to a robot (the field, future noise,
link outcomes and gradient sets) is generated only here.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import stat
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y, H = 20, 16, 120
ROBOTS = ("R0", "R1", "R2", "R3")
SCENARIOS = ("smooth_field", "sharp_frontier", "range_gap", "moving_hotspot", "bandwidth_burst")
SNAPSHOT_SLOTS = tuple(range(9, H, 10))
SNAPSHOT_SET = set(SNAPSHOT_SLOTS)
ROOTS = tuple((x, y) for y in range(0, GRID_Y, 4) for x in range(0, GRID_X, 4))
ENERGY_BUDGET = {
    "smooth_field": 480.0,
    "sharp_frontier": 500.0,
    "range_gap": 500.0,
    "moving_hotspot": 520.0,
    "bandwidth_burst": 490.0,
}
STARTS = {
    "smooth_field": ((1, 1), (18, 1), (1, 14), (18, 14)),
    "sharp_frontier": ((1, 2), (1, 13), (18, 2), (18, 13)),
    "range_gap": ((2, 2), (7, 2), (12, 13), (17, 13)),
    "moving_hotspot": ((1, 1), (18, 1), (1, 14), (18, 14)),
    "bandwidth_burst": ((2, 8), (7, 8), (12, 8), (17, 8)),
}
OBSTACLES = {
    "smooth_field": (),
    "sharp_frontier": ((9, 9, 0, 6), (9, 9, 9, 15)),
    "range_gap": ((8, 11, 5, 10),),
    "moving_hotspot": ((7, 12, 6, 9),),
    "bandwidth_burst": ((5, 5, 3, 12), (14, 14, 3, 12)),
}
RADIUS = {"smooth_field": 9.0, "sharp_frontier": 8.0, "range_gap": 6.0, "moving_hotspot": 8.0, "bandwidth_burst": 8.0}
INTERFERENCE = {
    "smooth_field": lambda t: 0,
    "sharp_frontier": lambda t: 2 if 40 <= t < 60 else 0,
    "range_gap": lambda t: 2 if 30 <= t < 70 else 0,
    "moving_hotspot": lambda t: 1 if 50 <= t < 80 else 0,
    "bandwidth_burst": lambda t: 2 if 24 <= t < 48 or 72 <= t < 96 else 0,
}
CAPACITY = (544, 288, 0)
DELAY = (0, 1, 2)
ACTION_DELTAS = {"stay": (0, 0), "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
NODE_COST = {"update": 1, "split": 4, "merge": 2, "fuse": 1}
PACKET_STATUS = {"queued", "in_service", "arrived", "expired", "rejected"}
PACKET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

ATOM_SHARES = {
    "result_json": {"parse": 0.25, "root": 0.25, "scenarios": 0.25, "records": 0.25},
    "result_evidence": {"robots": 0.18, "observations": 0.12, "maps": 0.16, "links": 0.10, "services": 0.12, "packets": 0.14, "queues": 0.08, "mission": 0.10},
    "decision_legality": {"actions": 0.16, "operations": 0.18, "services": 0.16, "packets": 0.18, "queues": 0.10, "resources": 0.12, "triggers": 0.10},
    "replay_validity": {"motion": 0.15, "field": 0.15, "maps": 0.20, "links": 0.10, "packets": 0.18, "resources": 0.12, "snapshots": 0.10},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float):
        self.id, self.passed, self.detail, self.share = ident, bool(passed), detail, share


# Backward-compatible alias used by the author-side boundary probe.
Atom = AtomVerdict


class EvalState:
    def __init__(self):
        self.run_started = False
        self.run_finished = False
        self.exit_code: int | None = None
        self.timed_out = False
        self.elapsed_s: float | None = None
        self.official_ok = False
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.data: Any = None
        self.parse_ok = False
        self.records: dict[str, dict[str, Any]] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


def _finite(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(v: Any, lo: int, hi: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _shape(v: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(v, dict):
        return f"{path}: expected object"
    if set(v) != keys:
        return f"{path}: expected keys={sorted(keys)!r}, actual={sorted(v)!r}"
    return None


def _strict_json(raw: str) -> Any:
    def reject(v: str):
        raise ValueError(v)

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _wall(name: str, cell: tuple[int, int]) -> bool:
    x, y = cell
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in OBSTACLES[name])


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """A deterministic supercover for integer cell centres."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = x1 - x0, y1 - y0
    out = [(x0, y0)]
    if not dx and not dy:
        return tuple(out)
    sx, sy = (1 if dx > 0 else -1 if dx < 0 else 0), (1 if dy > 0 else -1 if dy < 0 else 0)
    ax, ay = abs(dx), abs(dy)
    tx, ty = (0.5 / ax if ax else math.inf), (0.5 / ay if ay else math.inf)
    dtx, dty = (1 / ax if ax else math.inf), (1 / ay if ay else math.inf)
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if tx < ty - EPS:
            x += sx; tx += dtx; out.append((x, y))
        elif ty < tx - EPS:
            y += sy; ty += dty; out.append((x, y))
        else:
            out.extend(((x + sx, y), (x, y + sy)))
            x += sx; y += sy; tx += dtx; ty += dty; out.append((x, y))
    return tuple(dict.fromkeys(out))


def _node_id(x0: int, y0: int, size: int) -> int:
    return {4: 0, 2: 1, 1: 2}[size] * 10000 + 32 * y0 + x0


def _node_geom(node_id: Any) -> tuple[int, int, int] | None:
    if not isinstance(node_id, int) or isinstance(node_id, bool):
        return None
    depth, rem = divmod(node_id, 10000)
    if depth not in (0, 1, 2):
        return None
    y0, x0 = divmod(rem, 32)
    size = {0: 4, 1: 2, 2: 1}[depth]
    if x0 < 0 or y0 < 0 or x0 + size > GRID_X or y0 + size > GRID_Y or x0 % size or y0 % size:
        return None
    return x0, y0, size


def _field(name: str, x: int, y: int, t: int) -> float:
    if name == "smooth_field":
        v = 0.15 + 0.018 * x + 0.015 * y + 0.08 * math.sin((t + x) / 18)
    elif name == "sharp_frontier":
        v = 0.15 + 0.65 * (x >= 10) + 0.08 * math.sin((y + t) / 10)
    elif name == "range_gap":
        v = 0.2 + 0.5 * math.exp(-((x - 15) ** 2 + (y - 4) ** 2) / 30) + 0.1 * math.sin(t / 12)
    elif name == "moving_hotspot":
        hx = 3 + (t // 10 % 14); hy = 3 + (t // 15 % 10)
        v = 0.1 + 0.8 * math.exp(-((x - hx) ** 2 + (y - hy) ** 2) / 12)
    else:
        v = 0.2 + 0.35 * (y >= 8) + 0.25 * math.sin((x + t) / 8)
    return _clip(v)


def _measurement(name: str, x: int, y: int, t: int, i: int) -> float:
    noise = 0.04 * (((17 * x + 11 * y + 5 * t + 3 * i) % 9) - 4) / 4
    return _clip(_field(name, x, y, t) + noise)


def _link(name: str, positions: Mapping[str, tuple[int, int]], sender: str, receiver: str, slot: int) -> tuple[bool, int]:
    available = sender != receiver and math.dist(positions[sender], positions[receiver]) <= RADIUS[name] + EPS
    if available:
        available = not any(_wall(name, c) for c in _line_cells(positions[sender], positions[receiver])[1:-1])
    level = INTERFERENCE[name](slot)
    return bool(available), CAPACITY[level] if available else 0


def _record_shape(row: Any, path: str) -> str | None:
    return _shape(row, {"scenario", "robots", "observations", "map_operations", "map_snapshots", "links", "service_slots", "packets", "queues", "mission_state"}, path)


def _result_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    vals = {
        "parse": (parsed, "strict JSON parse"),
        "root": (isinstance(data, dict) and set(data) == {"scenarios"}, "exact root key"),
        "scenarios": (isinstance(scenarios, list), "scenarios array"),
        "records": (isinstance(scenarios, list) and all(_record_shape(r, f"scenarios[{i}]") is None for i, r in enumerate(scenarios)), "scenario record containers"),
    }
    return [AtomVerdict(f"result_json.{k}", bool(v[0]), str(v[1]), ATOM_SHARES["result_json"][k]) for k, v in vals.items()]


def _rows(row: Mapping[str, Any], field: str, keys: set[str], length: int) -> tuple[bool, str]:
    vals = row.get(field)
    if not isinstance(vals, list) or len(vals) != length:
        return False, f"{field}: expected {length} rows"
    for i, value in enumerate(vals):
        err = _shape(value, keys, f"{field}[{i}]")
        if err:
            return False, err
    return True, "ok"


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {k: [True, "ok"] for k in ATOM_SHARES["result_evidence"]}
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        robots = row.get("robots")
        if not isinstance(robots, list) or len(robots) != 4:
            checks["robots"] = [False, f"{name}: robots length"]
        else:
            seen = set()
            for robot in robots:
                if _shape(robot, {"robot_id", "state_trace", "intended_actions", "executed_actions"}, "robot") or robot.get("robot_id") not in ROBOTS:
                    checks["robots"] = [False, f"{name}: robot shape"]
                    continue
                seen.add(robot["robot_id"])
                for field, keys, length in (("state_trace", {"slot", "x", "y", "energy", "active_leaf_count", "compute_used"}, H + 1), ("intended_actions", {"slot", "action"}, H), ("executed_actions", {"slot", "action"}, H)):
                    ok, detail = _rows(robot, field, keys, length)
                    if not ok: checks["robots"] = [False, f"{name}: {detail}"]
                for s in robot.get("state_trace", []):
                    if not (_integer(s.get("slot"), 0, H) and _integer(s.get("x"), 0, GRID_X - 1) and _integer(s.get("y"), 0, GRID_Y - 1) and _finite(s.get("energy")) and s.get("energy") >= -EPS and _integer(s.get("active_leaf_count"), 1, 128) and _integer(s.get("compute_used"), 0, 48)):
                        checks["robots"] = [False, f"{name}: invalid state"]
                for a in robot.get("intended_actions", []) + robot.get("executed_actions", []):
                    if a.get("action") not in ACTION_DELTAS or not _integer(a.get("slot"), 0, H - 1): checks["robots"] = [False, f"{name}: invalid action"]
            if seen != set(ROBOTS): checks["robots"] = [False, f"{name}: robot ids"]
        obs = row.get("observations")
        ok, detail = _rows(row, "observations", {"robot_id", "slot", "x", "y", "measurement", "leaf_id"}, 4 * H)
        if not ok: checks["observations"] = [False, f"{name}: {detail}"]
        else:
            for o in obs:
                if o.get("robot_id") not in ROBOTS or not _integer(o.get("slot"), 0, H - 1) or not (_integer(o.get("x"), 0, GRID_X - 1) and _integer(o.get("y"), 0, GRID_Y - 1)) or not _finite(o.get("measurement")) or not 0 <= o.get("measurement") <= 1 or _node_geom(o.get("leaf_id")) is None:
                    checks["observations"] = [False, f"{name}: invalid observation"]
        snaps = row.get("map_snapshots")
        if not isinstance(snaps, list) or len(snaps) != 48: checks["maps"] = [False, f"{name}: snapshots length"]
        else:
            for s in snaps:
                if _shape(s, {"robot_id", "snapshot_slot", "cells", "active_nodes"}, "snapshot") or s.get("robot_id") not in ROBOTS or s.get("snapshot_slot") not in SNAPSHOT_SET or not isinstance(s.get("cells"), list) or len(s["cells"]) != 320 or not isinstance(s.get("active_nodes"), list):
                    checks["maps"] = [False, f"{name}: snapshot shape"]; continue
                for c in s["cells"]:
                    if _shape(c, {"x", "y", "mean", "precision", "version", "age"}, "cell") or not (_integer(c.get("x"), 0, GRID_X - 1) and _integer(c.get("y"), 0, GRID_Y - 1) and _finite(c.get("mean")) and 0 <= c.get("mean") <= 1 and _finite(c.get("precision")) and c.get("precision") > 0 and _integer(c.get("version"), -1, 10**9) and _integer(c.get("age"), 0, H)):
                        checks["maps"] = [False, f"{name}: invalid cell"]
                for n in s["active_nodes"]:
                    if _shape(n, {"node_id", "x0", "y0", "size", "mean", "precision", "version", "updated_slot", "source_versions"}, "active_node") or _node_geom(n.get("node_id")) != (n.get("x0"), n.get("y0"), n.get("size")) or not _finite(n.get("mean")) or not 0 <= n.get("mean") <= 1 or not _finite(n.get("precision")) or n.get("precision") <= 0 or not _integer(n.get("version"), -1, 10**9) or not _integer(n.get("updated_slot"), -1, H) or set(n.get("source_versions", {})) != set(ROBOTS) or any(not _integer(v, -1, 10**9) for v in n.get("source_versions", {}).values()):
                        checks["maps"] = [False, f"{name}: invalid active node"]
        ops = row.get("map_operations")
        if not isinstance(ops, list): checks["maps"] = [False, f"{name}: operations"]
        else:
            for op in ops:
                if _shape(op, {"operation_id", "robot_id", "slot", "kind", "node_id", "parent_id", "child_ids", "source_robot", "source_version", "cost"}, "operation") or not isinstance(op.get("operation_id"), int) or op.get("robot_id") not in ROBOTS or not _integer(op.get("slot"), 0, H - 1) or op.get("kind") not in NODE_COST or _node_geom(op.get("node_id")) is None or not isinstance(op.get("child_ids"), list) or op.get("source_robot") not in ROBOTS or not _integer(op.get("source_version"), -1, 10**9) or op.get("cost") != NODE_COST.get(op.get("kind")):
                    checks["maps"] = [False, f"{name}: invalid operation"]
        links = row.get("links")
        ok, detail = _rows(row, "links", {"sender", "receiver", "slot", "observed_available", "observed_capacity"}, 12 * H)
        if not ok: checks["links"] = [False, f"{name}: {detail}"]
        else:
            for l in links:
                if l.get("sender") not in ROBOTS or l.get("receiver") not in ROBOTS or l.get("sender") == l.get("receiver") or not _integer(l.get("slot"), 0, H - 1) or not isinstance(l.get("observed_available"), bool) or not _integer(l.get("observed_capacity"), 0, 544): checks["links"] = [False, f"{name}: invalid link"]
        services = row.get("service_slots")
        ok, detail = _rows(row, "service_slots", {"slot", "selected_links", "packet_ids", "served_bits"}, H)
        if not ok: checks["services"] = [False, f"{name}: {detail}"]
        else:
            for s in services:
                if not _integer(s.get("slot"), 0, H - 1) or not all(isinstance(x, str) and re.fullmatch(r"R[0-3]->R[0-3]", x) for x in s.get("selected_links", [])) or len(s.get("selected_links", [])) != len(s.get("packet_ids", [])) or len(s.get("selected_links", [])) != len(s.get("served_bits", [])) or any(not _integer(p, 0, 10**9) for p in s.get("packet_ids", [])) or any(not _integer(b, 1, 544) for b in s.get("served_bits", [])): checks["services"] = [False, f"{name}: invalid service"]
        packets = row.get("packets")
        if not isinstance(packets, list): checks["packets"] = [False, f"{name}: packets"]
        else:
            for p in packets:
                if _shape(p, {"packet_id", "sender", "receivers", "created_slot", "nodes", "payload_bits", "admitted_slot", "first_service_slot", "last_service_slot", "arrival_slot", "status", "retransmissions", "fused_node_ids"}, "packet") or not _integer(p.get("packet_id"), 0, 10**9) or p.get("sender") not in ROBOTS or not isinstance(p.get("receivers"), list) or len(p["receivers"]) != 1 or p["receivers"][0] not in ROBOTS or p["receivers"][0] == p["sender"] or not _integer(p.get("created_slot"), 0, H - 1) or not isinstance(p.get("nodes"), list) or not 1 <= len(p["nodes"]) <= 8 or p.get("payload_bits") != 32 + 64 * len(p["nodes"]) or not all(_shape(n, {"node_id", "source_robot", "version", "delta_tau", "delta_eta"}, "packet_node") is None and _node_geom(n.get("node_id")) is not None and n.get("source_robot") in ROBOTS and _integer(n.get("version"), 0, 10**9) and _finite(n.get("delta_tau")) and n.get("delta_tau") >= 0 and _finite(n.get("delta_eta")) for n in p["nodes"]) or any(v is not None and not _integer(v, 0, H + 2) for v in (p.get("admitted_slot"), p.get("first_service_slot"), p.get("last_service_slot"), p.get("arrival_slot"))) or p.get("status") not in PACKET_STATUS or not _integer(p.get("retransmissions"), 0, 10**6) or not isinstance(p.get("fused_node_ids"), list) or any(_node_geom(v) is None for v in p.get("fused_node_ids", [])):
                    checks["packets"] = [False, f"{name}: invalid packet"]
                elif len({(n["node_id"], n["version"]) for n in p["nodes"]}) != len(p["nodes"]) or len(set(p["fused_node_ids"])) != len(p["fused_node_ids"]):
                    checks["packets"] = [False, f"{name}: duplicate packet node evidence"]
        queues = row.get("queues")
        ok, detail = _rows(row, "queues", {"robot_id", "slot", "packet_ids", "served_packet_ids", "rejected_packet_ids"}, 4 * H)
        if not ok: checks["queues"] = [False, f"{name}: {detail}"]
        else:
            for q in queues:
                if q.get("robot_id") not in ROBOTS or not _integer(q.get("slot"), 0, H - 1) or not isinstance(q.get("packet_ids"), list) or len(q["packet_ids"]) > 16 or not isinstance(q.get("served_packet_ids"), list) or not isinstance(q.get("rejected_packet_ids"), list) or any(not _integer(x, 0, 10**9) for x in q["packet_ids"] + q["served_packet_ids"] + q["rejected_packet_ids"]): checks["queues"] = [False, f"{name}: invalid queue"]
        ok, detail = _rows(row, "mission_state", {"slot", "team_compute", "team_memory", "team_energy", "visited_cells", "expired_packets"}, H + 1)
        if not ok: checks["mission"] = [False, f"{name}: {detail}"]
        else:
            for m in row["mission_state"]:
                if not _integer(m.get("slot"), 0, H) or not _integer(m.get("team_compute"), 0, 3600) or not _integer(m.get("team_memory"), 0, 512) or not _finite(m.get("team_energy")) or m.get("team_energy") < -EPS or not _integer(m.get("visited_cells"), 0, GRID_X * GRID_Y) or not _integer(m.get("expired_packets"), 0, 10**6): checks["mission"] = [False, f"{name}: invalid mission"]
    return [AtomVerdict(f"result_evidence.{k}", bool(v[0]), str(v[1]), ATOM_SHARES["result_evidence"][k]) for k, v in checks.items()]


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list): return {}
    return {r.get("scenario"): r for r in data["scenarios"] if isinstance(r, dict) and r.get("scenario") in SCENARIOS}


# Legacy alias retained for older local tooling; the canonical coding-spec entry point is
# `_index_records`.
_records = _index_records


def _coverage_details(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list): return 0.0, ["scenarios.missing"]
    rows = data["scenarios"]
    names = [r.get("scenario") if isinstance(r, dict) else None for r in rows]
    for name in SCENARIOS:
        count = names.count(name)
        if count == 0:
            defects.append(f"scenario.missing:{name}")
        elif count > 1:
            defects.extend(f"scenario.duplicate:{name}:{i}" for i in range(count - 1))
    for i, name in enumerate(names):
        if name not in SCENARIOS:
            defects.append(f"scenario.unknown:{i}")
    expected_top = {"scenario", "robots", "observations", "map_operations", "map_snapshots", "links", "service_slots", "packets", "queues", "mission_state"}
    for name in SCENARIOS:
        row = next((r for r in rows if isinstance(r, dict) and r.get("scenario") == name), None)
        if row is None: continue
        if set(row) != expected_top: defects.append(f"{name}:top")
        if isinstance(row.get("robots"), list):
            keys = [r.get("robot_id") for r in row["robots"] if isinstance(r, dict)]
            if sorted(keys) != sorted(ROBOTS): defects.append(f"{name}:robot ids")
        canonical = {
            "observations": {(r, t) for r in ROBOTS for t in range(H)},
            "map_snapshots": {(r, t) for r in ROBOTS for t in SNAPSHOT_SLOTS},
            "links": {(a, b, t) for a in ROBOTS for b in ROBOTS if a != b for t in range(H)},
            "service_slots": {(t,) for t in range(H)},
            "queues": {(r, t) for r in ROBOTS for t in range(H)},
            "mission_state": {(t,) for t in range(H + 1)},
        }
        for field, keyfn in (("observations", lambda v: (v.get("robot_id"), v.get("slot"))), ("map_snapshots", lambda v: (v.get("robot_id"), v.get("snapshot_slot"))), ("links", lambda v: (v.get("sender"), v.get("receiver"), v.get("slot"))), ("service_slots", lambda v: (v.get("slot"),)), ("queues", lambda v: (v.get("robot_id"), v.get("slot"))), ("mission_state", lambda v: (v.get("slot"),))):
            vals = row.get(field) if isinstance(row.get(field), list) else []
            keys = [keyfn(v) if isinstance(v, dict) else ("<malformed>", i) for i, v in enumerate(vals)]
            expected = canonical[field]
            counts = defaultdict(int)
            for key in keys:
                counts[key] += 1
            for key in expected:
                if counts[key] == 0:
                    defects.append(f"{name}:{field}.missing:{key}")
                elif counts[key] > 1:
                    defects.extend(f"{name}:{field}.duplicate:{key}:{i}" for i in range(counts[key] - 1))
            for key in counts:
                if key not in expected:
                    defects.append(f"{name}:{field}.unknown:{key}")
        for snapshot in row.get("map_snapshots", []) if isinstance(row.get("map_snapshots"), list) else []:
            if not isinstance(snapshot, dict):
                defects.append(f"{name}:map_snapshots.malformed")
                continue
            cells = [(c.get("x"), c.get("y")) for c in snapshot.get("cells", []) if isinstance(c, dict)]
            if set(cells) != {(x, y) for x in range(GRID_X) for y in range(GRID_Y)} or len(cells) != GRID_X * GRID_Y:
                defects.append(f"{name}:snapshot-cells:{snapshot.get('robot_id')}/{snapshot.get('snapshot_slot')}")
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _coverage_score(data: Any) -> float:
    """Return only the fixed normalized canonical-coverage score."""
    return _coverage_details(data)[0]


# Legacy alias retained for older local tooling that also consumes diagnostics.
_coverage = _coverage_details


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {k: [True, "ok"] for k in ATOM_SHARES["decision_legality"]}
    for name in SCENARIOS:
        row = records.get(name)
        if row is None: continue
        for robot in row.get("robots", []) if isinstance(row.get("robots"), list) else []:
            if not isinstance(robot, dict):
                checks["actions"] = [False, f"{name}: malformed robot"]
                continue
            if any(a.get("action") not in ACTION_DELTAS for a in robot.get("intended_actions", []) + robot.get("executed_actions", [])): checks["actions"] = [False, f"{name}: action enum"]
        ops_by_slot = defaultdict(int)
        total = 0
        operation_ids = []
        for op in row.get("map_operations", []) if isinstance(row.get("map_operations"), list) else []:
            if not isinstance(op, dict):
                checks["operations"] = [False, f"{name}: malformed operation"]
                continue
            operation_ids.append(op.get("operation_id"))
            cost = op.get("cost")
            if not _integer(cost, 1, 4):
                checks["operations"] = [False, f"{name}: invalid operation cost"]
                continue
            ops_by_slot[(op.get("robot_id"), op.get("slot"))] += cost; total += cost
            if ops_by_slot[(op.get("robot_id"), op.get("slot"))] > 48: checks["operations"] = [False, f"{name}: per robot compute"]
            if op.get("kind") in ("split", "merge") and (op.get("parent_id") is not None and _node_geom(op.get("parent_id")) is None): checks["operations"] = [False, f"{name}: parent id"]
        if total > 3600: checks["resources"] = [False, f"{name}: total compute"]
        if len(operation_ids) != len(set(operation_ids)) or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in operation_ids): checks["operations"] = [False, f"{name}: operation ids"]
        slots = defaultdict(list)
        packet_ids = []
        for p in row.get("packets", []) if isinstance(row.get("packets"), list) else []:
            if not isinstance(p, dict):
                checks["packets"] = [False, f"{name}: malformed packet"]
                continue
            packet_ids.append(p.get("packet_id"))
            slots[p.get("sender")].append(p.get("created_slot"))
            if p.get("payload_bits", 0) > 544: checks["packets"] = [False, f"{name}: packet too large"]
        if len(packet_ids) != len(set(packet_ids)): checks["packets"] = [False, f"{name}: packet ids"]
        for sender, ts in slots.items():
            if any(not _integer(t, 0, H - 1) for t in ts) or len(ts) != len(set(ts)) or any(b - a < 2 for a, b in zip(sorted(ts), sorted(ts)[1:])): checks["triggers"] = [False, f"{name}: trigger spacing"]
        for s in row.get("service_slots", []) if isinstance(row.get("service_slots"), list) else []:
            if not isinstance(s, dict):
                checks["services"] = [False, f"{name}: malformed service"]
                continue
            links = s.get("selected_links", []); endpoints = []
            for link in links:
                try: a, b = link.split("->")
                except ValueError: continue
                endpoints.extend((a, b))
            if len(links) > 2 or len(endpoints) != len(set(endpoints)): checks["services"] = [False, f"{name}: link arbitration"]
        for q in row.get("queues", []) if isinstance(row.get("queues"), list) else []:
            if not isinstance(q, dict):
                checks["queues"] = [False, f"{name}: malformed queue"]
                continue
            if len(q.get("packet_ids", [])) > 16: checks["queues"] = [False, f"{name}: queue capacity"]
    return [AtomVerdict(f"decision_legality.{k}", bool(v[0]), str(v[1]), ATOM_SHARES["decision_legality"][k]) for k, v in checks.items()]


def _initial_map() -> dict[int, dict[str, Any]]:
    out = {}
    for x0, y0 in ROOTS:
        out[_node_id(x0, y0, 4)] = {"x0": x0, "y0": y0, "size": 4, "mu": 0.5, "tau": 4.0, "version": -1, "source_versions": {r: -1 for r in ROBOTS}, "source_updated": {r: -1 for r in ROBOTS}, "contrib": {}, "updated_slot": -1}
    return out


def _cover(nodes: Mapping[int, dict[str, Any]], x: int, y: int) -> tuple[int, dict[str, Any]] | None:
    candidates = [(nid, n) for nid, n in nodes.items() if n["x0"] <= x < n["x0"] + n["size"] and n["y0"] <= y < n["y0"] + n["size"]]
    return max(candidates, key=lambda p: p[1]["size"]) if candidates else None


def _overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    ax, ay, size_a = a
    bx, by, size_b = b
    width = max(0, min(ax + size_a, bx + size_b) - max(ax, bx))
    height = max(0, min(ay + size_a, by + size_b) - max(ay, by))
    return width * height


def _clone_nodes(nodes: Mapping[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        nid: {
            **node,
            "source_versions": dict(node["source_versions"]),
            "source_updated": dict(node["source_updated"]),
            "contrib": dict(node["contrib"]),
        }
        for nid, node in nodes.items()
    }


def _snapshot_cells(nodes: Mapping[int, dict[str, Any]], slot: int) -> list[dict[str, Any]]:
    out = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            found = _cover(nodes, x, y)
            if found is None: continue
            _, n = found
            out.append({"x": x, "y": y, "mean": n["mu"], "precision": n["tau"], "version": n["version"], "age": max(0, slot - n["updated_slot"])})
    return out


def _node_output(nodes: Mapping[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"node_id": nid, "x0": n["x0"], "y0": n["y0"], "size": n["size"], "mean": n["mu"], "precision": n["tau"], "version": n["version"], "updated_slot": n["updated_slot"], "source_versions": dict(n["source_versions"])} for nid, n in sorted(nodes.items())]


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try: return math.isclose(float(a), float(b), abs_tol=EPS, rel_tol=0.0)
        except Exception: return False
    return a == b


def _map_equal(candidate: Mapping[str, Any], nodes: Mapping[int, dict[str, Any]]) -> bool:
    actual = {n.get("node_id"): n for n in candidate.get("active_nodes", []) if isinstance(n, dict)}
    if set(actual) != set(nodes): return False
    for nid, n in nodes.items():
        c = actual[nid]
        for key, value in (("x0", n["x0"]), ("y0", n["y0"]), ("size", n["size"]), ("mean", n["mu"]), ("precision", n["tau"]), ("version", n["version"]), ("updated_slot", n["updated_slot"]), ("source_versions", n["source_versions"])):
            if not _eq(c.get(key), value): return False
    cells = {(c.get("x"), c.get("y")): c for c in candidate.get("cells", [])}
    expected = {(c["x"], c["y"]): c for c in _snapshot_cells(nodes, candidate.get("snapshot_slot", 0))}
    if set(cells) != set(expected): return False
    return all(all(_eq(cells[k].get(f), expected[k].get(f)) for f in ("mean", "precision", "version", "age")) for k in expected)


def _replay(name: str, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    errors = {k: "ok" for k in ("motion", "field", "maps", "links", "packets", "resources", "snapshots")}
    positions = {r: STARTS[name][i] for i, r in enumerate(ROBOTS)}
    energy = {r: ENERGY_BUDGET[name] / 4.0 for r in ROBOTS}
    maps = {r: _initial_map() for r in ROBOTS}
    obs = {(o.get("robot_id"), o.get("slot")): o for o in row.get("observations", []) if isinstance(o, dict)}
    robots = {r.get("robot_id"): r for r in row.get("robots", []) if isinstance(r, dict)}
    ops = sorted([o for o in row.get("map_operations", []) if isinstance(o, dict)], key=lambda x: (x.get("slot", -1), x.get("operation_id", -1)))
    ops_by = defaultdict(list)
    for op in ops: ops_by[(op.get("robot_id"), op.get("slot"))].append(op)
    snapshots = {(s.get("robot_id"), s.get("snapshot_slot")): s for s in row.get("map_snapshots", []) if isinstance(s, dict)}
    service_by = {s.get("slot"): s for s in row.get("service_slots", []) if isinstance(s, dict)}
    links_by = {(l.get("sender"), l.get("receiver"), l.get("slot")): l for l in row.get("links", []) if isinstance(l, dict)}
    packet_by = {p.get("packet_id"): p for p in row.get("packets", []) if isinstance(p, dict)}
    queues_by = {(q.get("robot_id"), q.get("slot")): q for q in row.get("queues", []) if isinstance(q, dict)}
    fifo = {r: [] for r in ROBOTS}; progress = defaultdict(int); retrans = defaultdict(int); completed = {}; arrival_events = defaultdict(list); expired = 0
    packet_list_by_slot = defaultdict(list)
    for p in packet_by.values(): packet_list_by_slot[(p.get("created_slot"), p.get("sender"))].append(p)
    expected_fused = defaultdict(set); expected_fuse_ops = defaultdict(list); local_updates = {}
    first_service = {}; last_service = {}; expired_ids = set(); rejected_ids = set(); successful = {}
    redundant_all = {}; truth_snapshots = {}; freshness_samples = []; qualified_visits = set()
    used_bits = 0; team_compute = 0; visited = set(); motion = 0; energy_used = 0.0; memory_peak = 80
    state_replay = {r: [dict(slot=0, x=positions[r][0], y=positions[r][1], energy=energy[r], active_leaf_count=20, compute_used=0)] for r in ROBOTS}
    mission_replay = [dict(slot=0, team_compute=0, team_memory=80, team_energy=sum(energy.values()), visited_cells=0, expired_packets=0)]
    for slot in range(H):
        # Arrivals are visible at the next slot boundary after completion plus delay.
        for pid in arrival_events.get(slot, []):
            p = packet_by.get(pid); completed[pid] = True
            if p is None: continue
            receiver = p["receivers"][0]
            for item_index, item in enumerate(p["nodes"]):
                source_geom = _node_geom(item["node_id"])
                if source_geom is None:
                    errors["maps"] = f"{name}: invalid arrival node"
                    continue
                changed = False
                for nid, node in maps[receiver].items():
                    overlap = _overlap(source_geom, (node["x0"], node["y0"], node["size"]))
                    source = item["source_robot"]
                    if overlap <= 0 or item["version"] <= node["source_versions"].get(source, -1):
                        continue
                    ratio = overlap / (source_geom[2] ** 2)
                    delta_tau = ratio * item["delta_tau"]
                    delta_eta = ratio * item["delta_eta"]
                    eta = node["mu"] * node["tau"] + delta_eta
                    node["tau"] += delta_tau
                    node["mu"] = _clip(eta / max(node["tau"], EPS))
                    node["source_versions"][source] = item["version"]
                    node["source_updated"][source] = slot
                    node["contrib"][(pid, item_index)] = (delta_tau, delta_eta)
                    node["updated_slot"] = slot
                    expected_fuse_ops[(receiver, slot)].append((nid, source, item["version"]))
                    changed = True
                if changed:
                    expected_fused[pid].add(item["node_id"])
        # Local observation and operations.
        for i, r in enumerate(ROBOTS):
            pxy = positions[r]; item = obs.get((r, slot)); expected_leaf = _cover(maps[r], *pxy)
            visited.add(pxy)
            if item is None or (item.get("x"), item.get("y")) != pxy or not _eq(item.get("measurement"), _measurement(name, pxy[0], pxy[1], slot, i)):
                errors["field"] = f"{name}: observation {r}/{slot}"
            if expected_leaf is None or item is None or item.get("leaf_id") != expected_leaf[0]: errors["maps"] = f"{name}: observation leaf {r}/{slot}"
            cost = 0
            # Fusion state was applied at the slot boundary; operation rows retain its evidence order.
            slot_ops = ops_by[(r, slot)]
            expected_arrivals = expected_fuse_ops[(r, slot)]
            fuse_rows = [op for op in slot_ops if op.get("kind") == "fuse"]
            update_rows = [op for op in slot_ops if op.get("kind") == "update"]
            actual_fuses = sorted((op.get("node_id"), op.get("source_robot"), op.get("source_version")) for op in fuse_rows)
            if actual_fuses != sorted(expected_arrivals):
                errors["maps"] = f"{name}: fuse evidence {r}/{slot}"
            if len(update_rows) != 1:
                errors["maps"] = f"{name}: update count {r}/{slot}"
            elif expected_leaf is not None:
                update = update_rows[0]
                version = update.get("source_version")
                if update.get("node_id") != expected_leaf[0] or update.get("source_robot") != r or not _integer(version, 0, 10**9) or version <= expected_leaf[1]["source_versions"][r] or update.get("parent_id") is not None or update.get("child_ids") != []:
                    errors["maps"] = f"{name}: update evidence {r}/{slot}"
                else:
                    node = expected_leaf[1]
                    z = _measurement(name, pxy[0], pxy[1], slot, i)
                    eta = node["mu"] * node["tau"] + 100.0 * z
                    node["tau"] += 100.0
                    node["mu"] = _clip(eta / node["tau"])
                    node["version"] = version
                    node["source_versions"][r] = version
                    node["source_updated"][r] = slot
                    node["updated_slot"] = slot
                    local_updates[(r, expected_leaf[0], version)] = (slot, z)
                    if abs(node["mu"] - _field(name, pxy[0], pxy[1], slot)) <= 0.15 and 1 / node["tau"] <= 0.05:
                        qualified_visits.add((r, pxy, slot))
            cost += 1
            kinds = [op.get("kind") for op in slot_ops]
            expected_prefix = ["fuse"] * len(fuse_rows) + ["update"]
            if kinds[: len(expected_prefix)] != expected_prefix or any(k in {"fuse", "update"} for k in kinds[len(expected_prefix):]):
                errors["maps"] = f"{name}: operation order {r}/{slot}"
            for op in slot_ops:
                kind, nid = op.get("kind"), op.get("node_id")
                if kind == "update": continue
                if kind == "split":
                    if op.get("parent_id") is not None or op.get("source_robot") != r or op.get("source_version") != -1:
                        errors["maps"] = f"{name}: split fields {r}/{slot}"
                    parent = maps[r].get(nid)
                    if parent is None or parent["size"] == 1: errors["maps"] = f"{name}: illegal split {r}/{slot}"; continue
                    h = parent["size"] // 2; expected = {_node_id(parent["x0"] + dx, parent["y0"] + dy, h) for dx in (0, h) for dy in (0, h)}
                    if set(op.get("child_ids", [])) != expected: errors["maps"] = f"{name}: split children {r}/{slot}"
                    del maps[r][nid]
                    for cid in expected:
                        x0, y0, size = _node_geom(cid); maps[r][cid] = {**parent, "x0": x0, "y0": y0, "size": size, "updated_slot": parent["updated_slot"], "source_versions": dict(parent["source_versions"]), "source_updated": dict(parent["source_updated"]), "contrib": dict(parent["contrib"])}
                elif kind == "merge":
                    if op.get("parent_id") is not None or op.get("source_robot") != r or op.get("source_version") != -1:
                        errors["maps"] = f"{name}: merge fields {r}/{slot}"
                    ids = op.get("child_ids", []); children = [maps[r].get(cid) for cid in ids]
                    geom = _node_geom(nid)
                    if geom is None or len(ids) != 4 or any(c is None for c in children): errors["maps"] = f"{name}: illegal merge {r}/{slot}"; continue
                    x0, y0, size = geom; expected = {_node_id(x0 + dx, y0 + dy, size // 2) for dx in (0, size // 2) for dy in (0, size // 2)} if size > 1 else set()
                    if size == 4 and set(ids) != expected or size == 2 and set(ids) != expected: errors["maps"] = f"{name}: merge children {r}/{slot}"
                    area = sum(c["size"] ** 2 for c in children); mu = sum(c["size"] ** 2 * c["mu"] for c in children) / area; var = sum(c["size"] ** 2 * (1 / c["tau"] + (c["mu"] - mu) ** 2) for c in children) / area
                    merged_tau = 1 / max(var, 0.0025); merged_eta = merged_tau * mu; contributions = {}
                    for key in set().union(*(set(c["contrib"]) for c in children)):
                        counter = []
                        for child in children:
                            dtau, deta = child["contrib"].get(key, (0.0, 0.0)); counter_tau = max(EPS, child["tau"] - dtau); counter_mu = (child["mu"] * child["tau"] - deta) / counter_tau; counter.append((child["size"] ** 2, counter_mu, counter_tau))
                        counter_mu = sum(a * m for a, m, _ in counter) / area; counter_var = sum(a * (1 / tau + (m - counter_mu) ** 2) for a, m, tau in counter) / area; counter_tau = 1 / max(counter_var, 0.0025)
                        contributions[key] = (merged_tau - counter_tau, merged_eta - counter_tau * counter_mu)
                    for cid in ids: maps[r].pop(cid, None)
                    maps[r][nid] = {"x0": x0, "y0": y0, "size": size, "mu": mu, "tau": merged_tau, "version": max(c["version"] for c in children), "source_versions": {src: max(c["source_versions"][src] for c in children) for src in ROBOTS}, "source_updated": {src: max(c["source_updated"][src] for c in children) for src in ROBOTS}, "contrib": contributions, "updated_slot": slot}
                elif kind == "fuse":
                    if op.get("parent_id") is not None or op.get("child_ids") != []:
                        errors["maps"] = f"{name}: fuse fields {r}/{slot}"
                cost += NODE_COST.get(kind, 0)
            if cost > 48: errors["resources"] = f"{name}: compute {r}/{slot}"
            if cost != sum(op.get("cost", 0) for op in slot_ops): errors["resources"] = f"{name}: cost evidence {r}/{slot}"
            if len(maps[r]) > 128:
                errors["resources"] = f"{name}: active leaf budget {r}/{slot}"
            state_replay[r].append(dict(slot=slot + 1, x=positions[r][0], y=positions[r][1], energy=energy[r], active_leaf_count=len(maps[r]), compute_used=cost))
            team_compute += cost
            memory_peak = max(memory_peak, sum(len(maps[q]) for q in ROBOTS))
        # Probes at pre-motion positions.
        for sender in ROBOTS:
            for receiver in ROBOTS:
                if sender == receiver: continue
                av, cap = _link(name, positions, sender, receiver, slot); row_link = links_by.get((sender, receiver, slot))
                if row_link is None or row_link.get("observed_available") != av or row_link.get("observed_capacity") != cap: errors["links"] = f"{name}: link {sender}->{receiver}/{slot}"
        # FIFO admissions and expiry.
        for r in ROBOTS:
            for pid in list(fifo[r]):
                p = packet_by.get(pid)
                if p and slot - p["created_slot"] > 16:
                    fifo[r].remove(pid); expired += 1; expired_ids.add(pid); progress[pid] = 0
            for p in packet_list_by_slot.get((slot, r), []):
                pid = p["packet_id"]
                nodes_valid = True
                for item in p["nodes"]:
                    truth = local_updates.get((r, item["node_id"], item["version"]))
                    if item["source_robot"] != r or truth is None or not _eq(item["delta_tau"], 100.0) or not _eq(item["delta_eta"], 100.0 * (truth[1] if truth else 0.0)):
                        nodes_valid = False
                if not nodes_valid:
                    errors["packets"] = f"{name}: invented packet increment {pid}"
                receiver = p["receivers"][0]
                redundant_all[pid] = all(
                    all(
                        node["source_versions"].get(item["source_robot"], -1) >= item["version"]
                        for node in maps[receiver].values()
                        if _overlap(_node_geom(item["node_id"]), (node["x0"], node["y0"], node["size"])) > 0
                    )
                    for item in p["nodes"]
                )
                if p.get("admitted_slot") == slot and len(fifo[r]) < 16:
                    fifo[r].append(pid)
                else:
                    rejected_ids.add(pid)
                    if p.get("admitted_slot") is not None:
                        errors["packets"] = f"{name}: admission {pid}"
        # At most two disjoint links, FIFO head service.
        service = service_by.get(slot, {"selected_links": [], "packet_ids": [], "served_bits": []}); selected = service.get("selected_links", [])
        for link, pid, bits in zip(selected, service.get("packet_ids", []), service.get("served_bits", [])):
            try: sender, receiver = link.split("->")
            except ValueError: errors["packets"] = f"{name}: bad link"; continue
            packet = packet_by.get(pid)
            if packet is None or sender != packet.get("sender") or packet.get("receivers") != [receiver]:
                errors["packets"] = f"{name}: service route {pid}/{slot}"
                continue
            av, cap = _link(name, positions, sender, receiver, slot)
            if not av or bits > cap or pid not in fifo.get(sender, []) or fifo[sender][0] != pid: errors["packets"] = f"{name}: FIFO/service {slot}"; continue
            remaining = packet_by[pid]["payload_bits"] - progress[pid]
            if bits > remaining:
                errors["packets"] = f"{name}: service exceeds remaining payload {pid}"
                continue
            used_bits += bits; energy[sender] -= bits * 0.001; energy_used += bits * 0.001
            first_service.setdefault(pid, slot); last_service[pid] = slot
            progress[pid] += bits
            if progress[pid] >= packet_by[pid]["payload_bits"]:
                level = INTERFERENCE[name](slot); good = ((19 * ROBOTS.index(sender) + 7 * ROBOTS.index(receiver) + 3 * slot) % 100) / 100 >= 0.05 + 0.15 * level
                if good:
                    fifo[sender].remove(pid); progress[pid] = 0; p = packet_by[pid]; arrival = slot + DELAY[level] + 1; arrival_events[arrival].append(pid); completed[pid] = arrival; successful[pid] = arrival
                else:
                    progress[pid] = 0; retrans[pid] += 1
        if used_bits > 48000:
            errors["packets"] = f"{name}: service bit budget"
        for r in ROBOTS:
            queue = queues_by.get((r, slot), {})
            served = [pid for link, pid, bits in zip(service.get("selected_links", []), service.get("packet_ids", []), service.get("served_bits", [])) if link.startswith(f"{r}->") and bits > 0]
            rejected = sorted(p["packet_id"] for p in packet_list_by_slot.get((slot, r), []) if p["packet_id"] in rejected_ids)
            if queue.get("packet_ids") != fifo[r] or queue.get("served_packet_ids") != served or sorted(queue.get("rejected_packet_ids", [])) != rejected:
                errors["packets"] = f"{name}: queue evidence {r}/{slot}"
        # Movement and resource boundary.
        for i, r in enumerate(ROBOTS):
            robot = robots.get(r, {}); acts = robot.get("intended_actions", []); exs = robot.get("executed_actions", []); intended = acts[slot].get("action", "stay") if slot < len(acts) else "stay"; submitted = exs[slot].get("action", "stay") if slot < len(exs) else "stay"; dx, dy = ACTION_DELTAS.get(intended, (0, 0)); candidate = (positions[r][0] + dx, positions[r][1] + dy); legal = 0 <= candidate[0] < GRID_X and 0 <= candidate[1] < GRID_Y and not _wall(name, candidate); actual = intended if legal else "stay"; target = candidate if legal else positions[r]
            if submitted != actual: errors["motion"] = f"{name}: executed action {r}/{slot}"
            if actual != "stay": positions[r] = target; motion += 1; energy[r] -= 1.0
            energy[r] -= 0.1
            energy_used += 0.1 + (1.0 if actual != "stay" else 0.0) + 0.02 * state_replay[r][-1]["compute_used"]
            energy[r] -= 0.02 * state_replay[r][-1]["compute_used"]
            if energy[r] < -EPS:
                errors["resources"] = f"{name}: exhausted energy {r}/{slot}"
            trace = state_replay[r][-1]; trace["x"], trace["y"], trace["energy"] = positions[r][0], positions[r][1], energy[r]
        # Snapshot consistency after slot state transition.
        if slot in SNAPSHOT_SET:
            for r in ROBOTS:
                truth_snapshots[(r, slot)] = _clone_nodes(maps[r])
                for source in ROBOTS:
                    if source == r:
                        continue
                    newest = max((node["source_updated"][source] for node in maps[r].values()), default=-1)
                    freshness_samples.append(0.0 if newest < 0 else _clip(1 - (slot - newest) / 16))
                cand = snapshots.get((r, slot))
                if cand is None or not _map_equal(cand, maps[r]): errors["snapshots"] = f"{name}: snapshot {r}/{slot}"
        mission_replay.append(dict(slot=slot + 1, team_compute=team_compute, team_memory=sum(len(maps[r]) for r in ROBOTS), team_energy=sum(energy.values()), visited_cells=len(visited), expired_packets=expired))
    # Compare state and mission traces.
    for r in ROBOTS:
        robot = robots.get(r, {})
        states = {s.get("slot"): s for s in robot.get("state_trace", [])}
        for t, expected in enumerate(state_replay[r]):
            got = states.get(t)
            if got is None or (got.get("x"), got.get("y")) != (expected["x"], expected["y"]) or not math.isclose(float(got.get("energy", math.nan)), expected["energy"], abs_tol=EPS) or got.get("active_leaf_count") != expected["active_leaf_count"] or got.get("compute_used") != expected["compute_used"]: errors["motion"] = f"{name}: state {r}/{t}"
    missions = {m.get("slot"): m for m in row.get("mission_state", []) if isinstance(m, dict)}
    for expected in mission_replay:
        got = missions.get(expected["slot"])
        if got is None or any(not _eq(got.get(k), expected[k]) for k in ("team_compute", "team_memory", "team_energy", "visited_cells", "expired_packets")): errors["resources"] = f"{name}: mission {expected['slot']}"
    # Packet terminal evidence follows the public end-of-horizon status definitions.
    for pid, p in packet_by.items():
        if p.get("retransmissions") != retrans.get(pid, 0): errors["packets"] = f"{name}: retransmissions {pid}"
        if p.get("first_service_slot") != first_service.get(pid) or p.get("last_service_slot") != last_service.get(pid): errors["packets"] = f"{name}: service slots {pid}"
        expected_arrival = successful.get(pid)
        if p.get("arrival_slot") != expected_arrival: errors["packets"] = f"{name}: arrival {pid}"
        if pid in rejected_ids:
            status = "rejected"
        elif pid in expired_ids:
            status = "expired"
        elif pid in successful:
            status = "arrived"
        elif progress.get(pid, 0) > 0:
            status = "in_service"
        else:
            status = "queued"
        if p.get("status") != status or set(p.get("fused_node_ids", [])) != expected_fused.get(pid, set()): errors["packets"] = f"{name}: terminal evidence {pid}"
    return {"maps": maps, "snapshots": snapshots, "truth_snapshots": truth_snapshots, "state": state_replay, "mission": mission_replay, "visited": visited, "qualified_visits": qualified_visits, "freshness": freshness_samples, "redundant_all": redundant_all, "energy_used": energy_used, "motion": motion, "compute": team_compute, "memory_peak": memory_peak, "sent_bits": used_bits, "packets": packet_by}, errors


def _replay_atoms(records: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    out = {}; passed = {k: [True, "ok"] for k in ATOM_SHARES["replay_validity"]}
    for name in SCENARIOS:
        row = records.get(name)
        if row is None: continue
        try: result, errors = _replay(name, row); out[name] = result
        except Exception as exc:
            errors = {k: f"{name}: replay exception {type(exc).__name__}: {exc}" for k in passed}; out[name] = {}
        for key, detail in errors.items():
            if detail != "ok": passed[key] = [False, detail]
    return [AtomVerdict(f"replay_validity.{k}", bool(v[0]), str(v[1]), ATOM_SHARES["replay_validity"][k]) for k, v in passed.items()], out


def _score_atoms(atoms: Sequence[AtomVerdict], shares: Mapping[str, float]) -> float:
    groups = [name for name, declared in ATOM_SHARES.items() if declared is shares or declared == shares]
    if len(groups) != 1:
        return 0.0
    prefix = groups[0]
    ids = [atom.id for atom in atoms]
    expected_ids = {f"{prefix}.{suffix}" for suffix in shares}
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        return 0.0
    try:
        declared = sum(float(value) for value in shares.values())
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(declared) or not math.isclose(declared, 1.0, rel_tol=0.0, abs_tol=EPS):
        return 0.0
    for atom in atoms:
        suffix = atom.id.split(".", 1)[1]
        try:
            if not math.isfinite(float(atom.share)) or float(atom.share) <= 0:
                return 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isclose(float(atom.share), float(shares[suffix]), rel_tol=0.0, abs_tol=EPS):
            return 0.0
    return _clip(sum(atom.share for atom in atoms if atom.passed), 0.0, 1.0)


# Legacy alias retained for the boundary probe and older local tooling.
_score = _score_atoms


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    return "failed_atoms=[]" if not failed else "failed_atoms=[{}]; {}".format(",".join(a.id for a in failed), " | ".join(f"{a.id}: {a.detail}" for a in failed))


# Legacy alias retained for the boundary probe and older local tooling.
_diagnose = _diagnose_atoms


def _gradient(name: str, cell: tuple[int, int], slot: int) -> float:
    x, y = cell
    return max(abs(_field(name, x, y, slot) - _field(name, nx, ny, slot)) for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if 0 <= nx < GRID_X and 0 <= ny < GRID_Y)


def _cell_node(nodes: Mapping[int, dict[str, Any]], cell: tuple[int, int]) -> dict[str, Any]:
    found = _cover(nodes, *cell)
    assert found is not None
    return found[1]


def _counter_mean(node: Mapping[str, Any], key: tuple[int, int] | None) -> tuple[float, float]:
    if key is None or key not in node["contrib"]:
        return node["mu"], node["tau"]
    delta_tau, delta_eta = node["contrib"][key]
    tau = max(EPS, node["tau"] - delta_tau)
    return _clip((node["mu"] * node["tau"] - delta_eta) / tau), tau


def _metrics(replayed: Mapping[str, dict[str, Any]], records: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    values = {key: [] for key in ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")}
    for name in SCENARIOS:
        rep = replayed[name]
        snapshot_scores = []
        for robot in ROBOTS:
            for slot in SNAPSHOT_SLOTS:
                nodes = rep["truth_snapshots"][(robot, slot)]
                weighted_error = 0.0
                weight_sum = 0.0
                variance_sum = 0.0
                for y in range(GRID_Y):
                    for x in range(GRID_X):
                        node = _cell_node(nodes, (x, y))
                        weight = 1 + 2 * _gradient(name, (x, y), slot)
                        weighted_error += weight * abs(node["mu"] - _field(name, x, y, slot))
                        weight_sum += weight
                        variance_sum += 1 / node["tau"]
                accuracy = 1 - weighted_error / weight_sum
                uncertainty = _clip(1 - (variance_sum / (GRID_X * GRID_Y)) / 0.25)
                snapshot_scores.append(0.65 * accuracy + 0.35 * uncertainty)
        values["primary_metrics_1"].append(sum(snapshot_scores) / len(snapshot_scores))

        eligible = covered = qualified = 0
        coverage_actual = {}
        for slot in SNAPSHOT_SLOTS:
            recent_slots = range(max(0, slot - 9), slot + 1)
            for y in range(GRID_Y):
                for x in range(GRID_X):
                    cell = (x, y)
                    if _gradient(name, cell, slot) < 0.25:
                        continue
                    eligible += 1
                    good = 0
                    for robot in ROBOTS:
                        node = _cell_node(rep["truth_snapshots"][(robot, slot)], cell)
                        good += int(abs(node["mu"] - _field(name, x, y, slot)) <= 0.15 and 1 / node["tau"] <= 0.05)
                    recent_visit = any((rep["state"][robot][t]["x"], rep["state"][robot][t]["y"]) == cell for robot in ROBOTS for t in recent_slots)
                    coverage_actual[(slot, cell)] = recent_visit and good >= 3
                    covered += int(coverage_actual[(slot, cell)])
                    qualified += int(any((robot, cell, t) in rep["qualified_visits"] for robot in ROBOTS for t in recent_slots))
        values["primary_metrics_2"].append(1.0 if not eligible else 0.6 * covered / eligible + 0.4 * qualified / eligible)
        values["primary_metrics_3"].append(0.3 * _clip(1 - rep["compute"] / 3600) + 0.25 * _clip(1 - rep["memory_peak"] / 512) + 0.2 * _clip(1 - rep["motion"] / (4 * H)) + 0.25 * _clip(1 - rep["energy_used"] / ENERGY_BUDGET[name]))

        packets = list(rep["packets"].values())
        useful_bits = 0
        packet_effects = {}
        for packet in packets:
            arrival = packet.get("arrival_slot")
            next_slot = next((slot for slot in SNAPSHOT_SLOTS if isinstance(arrival, int) and slot >= arrival), None)
            improvements = []
            coverage_changes = []
            for item_index, item in enumerate(packet["nodes"]):
                if item["node_id"] not in packet.get("fused_node_ids", []) or next_slot is None:
                    improvements.append(0.0); coverage_changes.append(False); continue
                receiver = packet["receivers"][0]
                nodes = rep["truth_snapshots"][(receiver, next_slot)]
                key = (packet["packet_id"], item_index)
                actual_error = counter_error = weight_sum = 0.0
                changed_coverage = False
                source_geom = _node_geom(item["node_id"])
                for y in range(GRID_Y):
                    for x in range(GRID_X):
                        cell = (x, y); node = _cell_node(nodes, cell); counter_mu, counter_tau = _counter_mean(node, key); truth = _field(name, x, y, next_slot); weight = 1 + 2 * _gradient(name, cell, next_slot)
                        actual_error += weight * abs(node["mu"] - truth); counter_error += weight * abs(counter_mu - truth); weight_sum += weight
                        if source_geom and _overlap(source_geom, (x, y, 1)) and coverage_actual.get((next_slot, cell), False):
                            actual_good = counter_good = 0
                            for robot in ROBOTS:
                                robot_node = _cell_node(rep["truth_snapshots"][(robot, next_slot)], cell)
                                mean, tau = _counter_mean(robot_node, key if robot == receiver else None)
                                actual_good += int(abs(robot_node["mu"] - truth) <= 0.15 and 1 / robot_node["tau"] <= 0.05)
                                counter_good += int(abs(mean - truth) <= 0.15 and 1 / tau <= 0.05)
                            changed_coverage = changed_coverage or actual_good >= 3 > counter_good
                improvement = (counter_error - actual_error) / weight_sum
                improvements.append(improvement); coverage_changes.append(changed_coverage)
                if improvement >= 0.01 - EPS or changed_coverage:
                    useful_bits += 64
            packet_effects[packet["packet_id"]] = (improvements, coverage_changes)
        generated = len(packets)
        on_time = sum(1 for packet in packets if isinstance(packet.get("arrival_slot"), int) and packet["arrival_slot"] - packet["created_slot"] <= 16)
        values["auxiliary_metrics_1"].append(0.0 if not generated else _clip(0.6 * useful_bits / max(1, rep["sent_bits"]) + 0.4 * on_time / generated))

        redundant = 0
        for packet in packets:
            effects, changes = packet_effects[packet["packet_id"]]
            if rep["redundant_all"].get(packet["packet_id"], False) or (packet.get("fused_node_ids") and all(value < 0.005 - EPS for value in effects) and not any(changes)):
                redundant += 1
        freshness = sum(rep["freshness"]) / len(rep["freshness"]) if rep["freshness"] else 0.0
        nonredundancy = 0.0 if not packets else 1 - redundant / generated
        admitted = [packet for packet in packets if isinstance(packet.get("admitted_slot"), int)]
        waits = [(packet.get("first_service_slot") if isinstance(packet.get("first_service_slot"), int) else H) - packet["admitted_slot"] for packet in admitted]
        queue_component = 1.0 if not admitted else _clip(1 - (sum(waits) / len(waits)) / 16)
        values["auxiliary_metrics_2"].append(_clip(0.4 * freshness + 0.3 * nonredundancy + 0.3 * queue_component))
    return {key: _clip(sum(items) / len(items)) for key, items in values.items()}


async def _build_state(ws: Workspace) -> EvalState:
    root = str(Path(ws.root).resolve())
    if root in _CACHE: return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE: return _CACHE[root]
        state = EvalState(); path = Path(ws.root, "results.json"); started = time.monotonic()
        state.run_started = True
        pre_signature: tuple[int, int, int, int] | None = None
        try:
            old_stat = path.lstat()
            pre_signature = (int(old_stat.st_dev), int(old_stat.st_ino), int(old_stat.st_mtime_ns), int(old_stat.st_size))
        except (FileNotFoundError, OSError):
            pre_signature = None
        exit_code: int | None = None
        timed_out = False
        remove_ok = False
        try:
            remove_result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(remove_result, "exit_code", None) == 0
            if remove_ok:
                try:
                    path.lstat()
                    remove_ok = False
                except FileNotFoundError:
                    pass
                except OSError:
                    remove_ok = False
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.run_finished = True
            exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            timed_out = exit_code is None
            elapsed_s = time.monotonic() - started
            timed_out = timed_out or elapsed_s >= 600.0 - EPS
            try:
                statv = path.lstat()
                post_signature = (int(statv.st_dev), int(statv.st_ino), int(statv.st_mtime_ns), int(statv.st_size))
                state.result_file_is_new_regular = bool(
                    remove_ok
                    and stat.S_ISREG(statv.st_mode)
                    and not path.is_symlink()
                    and (pre_signature is None or post_signature != pre_signature)
                )
            except (FileNotFoundError, OSError):
                state.result_file_is_new_regular = False
            state.official_ok = exit_code == 0 and not timed_out and elapsed_s <= 600.0 + EPS and state.result_file_is_new_regular
            state.official_message = f"exit_code={exit_code}, elapsed_s={elapsed_s:.6f}, removed_previous={remove_ok}, timed_out={timed_out}, fresh_regular_result={state.result_file_is_new_regular}"
            if not state.official_ok:
                state.diagnostics["official"] = f"official run failed: {state.official_message}; stderr={stderr[-400:]}"
            if state.official_ok:
                state.data = _strict_json(path.read_text(encoding="utf-8")); state.parse_ok = True
        except Exception as exc:
            elapsed_s = time.monotonic() - started
            timed_out = isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or elapsed_s >= 600.0 - EPS
            state.official_ok = False
            state.official_message = f"official run raised {type(exc).__name__}: {exc}; elapsed_s={elapsed_s:.6f}; timed_out={timed_out}"
            state.diagnostics["official"] = state.official_message
        finally:
            state.exit_code = exit_code
            state.timed_out = timed_out
            state.elapsed_s = elapsed_s if "elapsed_s" in locals() else time.monotonic() - started
            state.run_finished = True
        if "official" not in state.diagnostics:
            state.diagnostics["official"] = state.official_message or "official run did not complete"
        atoms = _result_atoms(state.data, state.parse_ok); state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"]); state.diagnostics["result_json"] = _diagnose_atoms(atoms)
        if not state.official_ok or not state.parse_ok:
            reason = state.diagnostics.get("official", "official run or strict results parse failed")
            for key in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[key] = 0.0
                state.diagnostics[key] = f"{key}: not evaluated because {reason}"
            _CACHE[root] = state
            return state
        state.records = _index_records(state.data); atoms = _evidence_atoms(state.records); state.scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"]); state.diagnostics["result_evidence"] = _diagnose_atoms(atoms)
        state.scores["canonical_coverage"], defects = _coverage_details(state.data); state.diagnostics["canonical_coverage"] = str(defects)
        atoms = _legality_atoms(state.records); state.scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"]); state.diagnostics["decision_legality"] = _diagnose_atoms(atoms)
        atoms, state.replayed = _replay_atoms(state.records); state.scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"]); state.diagnostics["replay_validity"] = _diagnose_atoms(atoms)
        state.validity_gate = all(state.scores.get(k, 0.0) >= 1 - EPS for k in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"))
        if state.validity_gate:
            try:
                metrics = _metrics(state.replayed, state.records)
                required = {"primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2"}
                if set(metrics) != required or any(not _finite(value) or not 0.0 <= float(value) <= 1.0 for value in metrics.values()):
                    state.validity_gate = False
                    state.diagnostics["metrics"] = "metrics rejected: expected five finite values in [0,1]"
                else:
                    state.metrics = {key: float(value) for key, value in metrics.items()}
            except Exception as exc:
                state.validity_gate = False
                state.diagnostics["metrics"] = f"metrics failed: {type(exc).__name__}: {exc}"
        else:
            state.diagnostics["metrics"] = "metrics not evaluated because validity gate failed"
        _CACHE[root] = state; return state


# Legacy alias retained for local callers from the earlier grader revision.
_build = _build_state


async def _gate(ws: Workspace, key: str):
    s = await _build_state(ws); return s.scores.get(key, 0.0), s.diagnostics.get(key, "")


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "run.py")
    try:
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except Exception as exc:
        return 0.0, f"run.py compilation failed: {type(exc).__name__}: {exc}"
    try:
        regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        regular = False
    if regular and getattr(result, "exit_code", 1) == 0:
        return 1.0, "run.py exists and compiles"
    return 0.0, "run.py is missing or does not compile"


@check(id="official_run")
async def official_run(ws: Workspace):
    s = await _build_state(ws); return (1.0, "official run passed") if s.official_ok else (0.0, s.diagnostics.get("official", "official run failed"))


def _report_missing_sections(text: str) -> list[str]:
    required = ("Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations")
    missing: list[str] = []
    for title in required:
        m = re.search(rf"(?m)^#{{1,6}}\s+{re.escape(title)}\s*$", text)
        if not m or not re.split(r"(?m)^#{{1,6}}\s+", text[m.end():], maxsplit=1)[0].strip():
            missing.append(title)
    return missing


def _report_score(text: str) -> float:
    """Return the fixed normalized report-contract score for the supplied text."""
    return max(0.0, 2.0 - 0.5 * len(_report_missing_sections(text))) / 2.0


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "report.md")
    try:
        regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        regular = False
    try:
        text = (ws.read_text("report.md") or "") if regular else ""
    except OSError:
        text = ""
    missing = _report_missing_sections(text)
    return _report_score(text), f"missing_or_empty={missing}"


@check(id="result_json")
async def result_json(ws: Workspace):
    return await _gate(ws, "result_json")


@check(id="result_evidence")
async def result_evidence(ws: Workspace):
    return await _gate(ws, "result_evidence")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace):
    return await _gate(ws, "canonical_coverage")


@check(id="decision_legality")
async def decision_legality(ws: Workspace):
    return await _gate(ws, "decision_legality")


@check(id="replay_validity")
async def replay_validity(ws: Workspace):
    return await _gate(ws, "replay_validity")


async def _metric(ws: Workspace, key: str):
    s = await _build_state(ws); return (s.metrics[key], f"{key}: replayed={s.metrics[key]:.9g}") if s.metrics else (0.0, f"{key}: validity gate not satisfied")


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace):
    return await _metric(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace):
    return await _metric(ws, "primary_metrics_2")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace):
    return await _metric(ws, "primary_metrics_3")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace):
    return await _metric(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace):
    return await _metric(ws, "auxiliary_metrics_2")
