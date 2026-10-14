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

HORIZON = 80
GRID = 18
ROBOTS = 4
REGIONS = 8
ENERGY0 = 120.0
COMM_RADIUS = 5
OBS_RADIUS = 2
EPS = 1e-6
SCENARIO_NAMES = (
    "metal_corridor",
    "split_weak_zone",
    "relay_bottleneck",
    "moving_interference",
    "coverage_deadline",
)
ROLES = ("explore", "relay", "return")
COMMS = ("none", "new", "retry", "forward")
ACTION_KEYS = frozenset({"t", "robot", "dx", "dy", "role", "comm", "packet_id", "cells", "relay"})

SCENARIOS: dict[str, dict[str, Any]] = {
    "metal_corridor": {
        "starts": ((1, 2), (1, 5), (1, 12), (1, 15)),
        "edge": (0, 8),
        "obstacles": ((5, 6, 0, 6), (5, 6, 10, 17), (11, 12, 3, 14)),
        "regions": (
            (3, 2, 1, 24),
            (3, 13, 1, 28),
            (7, 7, 2, 38),
            (8, 14, 2, 44),
            (13, 2, 3, 52),
            (13, 7, 3, 58),
            (15, 12, 4, 68),
            (15, 15, 4, 74),
        ),
        "weak": None,
    },
    "split_weak_zone": {
        "starts": ((2, 2), (2, 15), (8, 2), (8, 15)),
        "edge": (0, 8),
        "obstacles": ((8, 9, 4, 13),),
        "regions": (
            (4, 3, 1, 26),
            (4, 12, 1, 30),
            (10, 2, 2, 42),
            (10, 14, 2, 46),
            (13, 4, 3, 56),
            (13, 11, 3, 60),
            (15, 7, 4, 70),
            (15, 14, 4, 76),
        ),
        "weak": "x10",
    },
    "relay_bottleneck": {
        "starts": ((1, 8), (4, 2), (4, 15), (8, 8)),
        "edge": (0, 8),
        "obstacles": ((7, 11, 0, 6), (7, 11, 10, 17)),
        "regions": (
            (5, 4, 1, 24),
            (5, 12, 1, 28),
            (9, 7, 3, 40),
            (12, 3, 2, 50),
            (12, 13, 2, 54),
            (15, 2, 4, 66),
            (15, 8, 4, 70),
            (15, 14, 4, 74),
        ),
        "weak": None,
    },
    "moving_interference": {
        "starts": ((2, 2), (2, 15), (15, 2), (15, 15)),
        "edge": (9, 9),
        "obstacles": ((6, 7, 5, 12),),
        "regions": (
            (3, 6, 1, 26),
            (3, 13, 1, 30),
            (8, 2, 2, 40),
            (8, 14, 2, 44),
            (12, 4, 3, 54),
            (12, 11, 3, 60),
            (15, 7, 4, 70),
            (15, 14, 4, 76),
        ),
        "weak": "moving",
    },
    "coverage_deadline": {
        "starts": ((1, 1), (1, 16), (16, 1), (16, 16)),
        "edge": (9, 9),
        "obstacles": ((4, 5, 4, 13), (12, 13, 4, 13)),
        "regions": (
            (2, 7, 3, 20),
            (2, 13, 3, 24),
            (6, 2, 2, 34),
            (6, 14, 2, 38),
            (10, 2, 2, 44),
            (10, 14, 2, 48),
            (14, 7, 4, 60),
            (14, 13, 4, 66),
        ),
        "weak": "deadline",
    },
}

ATOM_SHARES = {
    "result_json": {f"result_json.{x}": 0.25 for x in ("parse", "root", "scenarios", "actions")},
    "result_evidence": {f"result_evidence.{x}": 0.25 for x in ("rows", "fields", "ids", "types")},
    "decision_legality": {
        f"decision_legality.{x}": 0.2
        for x in ("moves", "roles", "packets", "information", "channel")
    },
    "replay_validity": {
        f"replay_validity.{x}": 1 / 5 for x in ("motion", "radio", "packets", "reports", "metrics")
    },
}
GATE_CHECKS = (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
)


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = ident, bool(passed), detail, float(share)


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
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _brief(value: Any, limit: int = 120) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _norm(v: Any) -> int:
    return -1 if v is None or v == -1 else int(v)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _strict_json(raw: str) -> Any:
    def reject(v: str) -> None:
        raise ValueError(f"non-finite JSON constant {v}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON key {k}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def _cell(cid: int) -> tuple[int, int]:
    return int(cid) % GRID, int(cid) // GRID


def _cid(x: int, y: int) -> int:
    return GRID * int(y) + int(x)


def _region_cells(spec: Mapping[str, Any], rid: int) -> list[int]:
    x, y = spec["regions"][rid][:2]
    return [_cid(x, y), _cid(x + 1, y), _cid(x, y + 1), _cid(x + 1, y + 1)]


def _blocked(spec: Mapping[str, Any], x: int, y: int) -> bool:
    return any(a <= x <= b and c <= y <= d for a, b, c, d in spec["obstacles"])


def _line_hit(a: Sequence[int], b: Sequence[int], rect: Sequence[int]) -> bool:
    xmin, xmax, ymin, ymax = rect
    x0, y0 = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - x0, float(b[1]) - y0
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) <= EPS:
            if q < -EPS:
                return False
        else:
            u = q / p
            if p < 0:
                lo = max(lo, u)
            else:
                hi = min(hi, u)
            if lo - hi > EPS:
                return False
    return True


def _interference(name: str, t: int, endpoint: Sequence[int], spec: Mapping[str, Any]) -> int:
    if name == "moving_interference":
        cx = 4 + (t % 10)
        return int(abs(endpoint[0] - cx) + abs(endpoint[1] - 9) <= 2)
    if name == "coverage_deadline":
        return int(30 <= t <= 39)
    return 0


def _rssi(name: str, a: Sequence[int], b: Sequence[int], t: int, concurrent: int) -> float:
    spec = SCENARIOS[name]
    d = abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
    o = int(any(_line_hit(a, b, r) for r in spec["obstacles"]))
    h = max(_interference(name, t, a, spec), _interference(name, t, b, spec))
    w = 8 if spec["weak"] == "x10" and (a[0] >= 10 or b[0] >= 10) else 0
    return -35 - 4 * d - 12 * o - 8 * h - 3 * max(0, concurrent - 1) - w


def _action_shape(a: Any) -> bool:
    return (
        isinstance(a, Mapping)
        and set(a) == ACTION_KEYS
        and _integer(a.get("t"))
        and _integer(a.get("robot"))
        and _integer(a.get("dx"))
        and _integer(a.get("dy"))
        and isinstance(a.get("role"), str)
        and isinstance(a.get("comm"), str)
        and (_integer(a.get("packet_id")) or a.get("packet_id") is None)
        and (_integer(a.get("relay")) or a.get("relay") is None)
        and isinstance(a.get("cells"), list)
        and all(_integer(x) for x in a["cells"])
    )


def _shape_atoms(data: Any) -> list[AtomVerdict]:
    parse = isinstance(data, Mapping)
    root = parse and set(data) == {"scenarios"}
    scenarios = parse and isinstance(data.get("scenarios"), list)
    records = scenarios and all(
        isinstance(x, Mapping)
        and set(x) == {"scenario", "actions"}
        and isinstance(x.get("scenario"), str)
        and isinstance(x.get("actions"), list)
        for x in data["scenarios"]
    )
    return [
        _atom("result_json", "parse", parse, "root object"),
        _atom("result_json", "root", root, "exact root"),
        _atom("result_json", "scenarios", scenarios, "scenario list"),
        _atom("result_json", "actions", records, "scenario/action containers"),
    ]


def _index(data: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for x in data["scenarios"]:
        if isinstance(x, Mapping) and isinstance(x.get("scenario"), str):
            out[str(x["scenario"])] = x
    return out


def _evidence_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    rows = fields = ids = types = True
    row_defects: list[str] = []
    field_defects: list[str] = []
    id_defects: list[str] = []
    type_defects: list[str] = []

    def note(defects: list[str], path: str, actual: Any, expected: str) -> None:
        if len(defects) < 8:
            defects.append(f"{path}: actual={_brief(actual)}; expected={expected}")

    for name in SCENARIO_NAMES:
        sc = scenarios.get(name)
        acts = sc.get("actions") if isinstance(sc, Mapping) else None
        if not isinstance(acts, list):
            rows = False
            note(row_defects, f"{name}.actions", acts, "list of action objects")
            continue
        for i, a in enumerate(acts):
            path = f"{name}.actions[{i}]"
            if not isinstance(a, Mapping):
                rows = False
                note(row_defects, path, a, "action object")
                continue
            if set(a) != ACTION_KEYS:
                fields = False
                note(field_defects, path, sorted(map(str, a)), f"exact keys {sorted(ACTION_KEYS)}")
            for key in ("t", "robot"):
                if not _integer(a.get(key)):
                    ids = False
                    note(id_defects, f"{path}.{key}", a.get(key), "integer")
            type_checks = (
                ("dx", _integer(a.get("dx")), "finite integer"),
                ("dy", _integer(a.get("dy")), "finite integer"),
                ("role", isinstance(a.get("role"), str), "string"),
                ("comm", isinstance(a.get("comm"), str), "string"),
                (
                    "packet_id",
                    _integer(a.get("packet_id")) or a.get("packet_id") is None,
                    "integer or null",
                ),
                ("relay", _integer(a.get("relay")) or a.get("relay") is None, "integer or null"),
                (
                    "cells",
                    isinstance(a.get("cells"), list)
                    and all(_integer(x) for x in a.get("cells", [])),
                    "list of integers",
                ),
            )
            for key, valid, expected in type_checks:
                if not valid:
                    types = False
                    note(type_defects, f"{path}.{key}", a.get(key), expected)
    return [
        _atom("result_evidence", "rows", rows, f"defects={row_defects}"),
        _atom("result_evidence", "fields", fields, f"defects={field_defects}"),
        _atom("result_evidence", "ids", ids, f"defects={id_defects}"),
        _atom("result_evidence", "types", types, f"defects={type_defects}"),
    ]


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    records = data.get("scenarios", []) if isinstance(data, Mapping) else []
    if not isinstance(records, list):
        return 0.0, ["scenarios: not a list"]
    scenario_counts: dict[Any, int] = {}
    action_counts: dict[tuple[Any, Any, Any], int] = {}
    for sc in records:
        if not isinstance(sc, Mapping):
            defects.append("unknown scenario record")
            continue
        name = sc.get("scenario")
        if not isinstance(name, str):
            name = f"<invalid:{type(name).__name__}:{name!r}>"
        scenario_counts[name] = scenario_counts.get(name, 0) + 1
        acts = sc.get("actions", [])
        if isinstance(acts, list):
            for a in acts:
                if isinstance(a, Mapping):
                    t_value = a.get("t")
                    r_value = a.get("robot")
                    if not _integer(t_value):
                        t_value = f"<invalid:{type(t_value).__name__}:{t_value!r}>"
                    if not _integer(r_value):
                        r_value = f"<invalid:{type(r_value).__name__}:{r_value!r}>"
                    key = (name, t_value, r_value)
                    action_counts[key] = action_counts.get(key, 0) + 1
                else:
                    defects.append(f"{name}: unknown action record")
    for name in SCENARIO_NAMES:
        count = scenario_counts.get(name, 0)
        defects.extend([f"scenario missing:{name}"] * int(count == 0))
        defects.extend([f"scenario duplicate:{name}"] * max(0, count - 1))
    for name, count in scenario_counts.items():
        if name not in SCENARIO_NAMES:
            defects.extend([f"scenario unknown:{name}"] * count)
    expected = {
        (name, t, r) for name in SCENARIO_NAMES for t in range(HORIZON) for r in range(ROBOTS)
    }
    for key in expected:
        count = action_counts.get(key, 0)
        defects.extend([f"action missing:{key}"] * int(count == 0))
        defects.extend([f"action duplicate:{key}"] * max(0, count - 1))
    for key, count in action_counts.items():
        if key not in expected:
            defects.extend([f"action unknown:{key}"] * count)
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _replay(name: str, sc: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the public action-only protocol without trusting candidate-derived state."""
    spec = SCENARIOS[name]
    acts = sc.get("actions", []) if isinstance(sc, Mapping) else []
    by_t = {
        t: {int(a["robot"]): a for a in acts if isinstance(a, Mapping) and a.get("t") == t}
        for t in range(HORIZON)
    }
    pos = [list(p) for p in spec["starts"]]
    energy = [ENERGY0] * ROBOTS
    local: list[dict[int, tuple[int, int]]] = [dict() for _ in range(ROBOTS)]
    edge: dict[int, tuple[int, int, int, int]] = {}
    queues: list[list[int]] = [[] for _ in range(ROBOTS)]
    packets: dict[int, dict[str, Any]] = {}
    pending_arrivals: dict[int, list[tuple[int, int]]] = {}
    explored: set[int] = set()
    reports: list[dict[str, Any] | None] = [None] * REGIONS
    safety = 0
    accepted_units = 0
    raw_by_robot = [0] * ROBOTS
    obs_rows: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    report_sources: list[set[tuple[int, int]] | None] = [None] * REGIONS
    invalid: list[str] = []
    defaults = {
        "dx": 0,
        "dy": 0,
        "role": "explore",
        "comm": "none",
        "packet_id": None,
        "cells": [],
        "relay": None,
    }

    def packet_deadline(p: Mapping[str, Any]) -> int:
        return min(
            (
                spec["regions"][i][3]
                for i in range(REGIONS)
                if any(c in _region_cells(spec, i) for c in p["cells"])
            ),
            default=HORIZON,
        )

    for t in range(HORIZON):
        # Arrivals are fused before observations and decisions.
        relay_arrivals: list[list[int]] = [[] for _ in range(ROBOTS)]
        for pid, hi in pending_arrivals.pop(t, []):
            p = packets.get(pid)
            if not p or hi in p["applied_arrivals"]:
                continue
            p["applied_arrivals"].add(hi)
            if hi == 0 and p["relay"] is not None:
                relay = int(p["relay"])
                relay_arrivals[relay].append(pid)
            elif hi == 0 or hi == 1:
                for cid, ver, obs_t in zip(p["cells"], p["versions"], p["obs_states"]):
                    if ver > 0 and (cid not in edge or ver > edge[cid][0]):
                        edge[cid] = (ver, obs_t, pid, hi)
                p["stage"] = "delivered"
        for rid in range(REGIONS):
            if reports[rid] is None:
                cells = _region_cells(spec, rid)
                if all(c in edge and edge[c][0] > 0 for c in cells):
                    reports[rid] = {
                        "region_id": rid,
                        "cells": cells,
                        "versions": [edge[c][0] for c in cells],
                        "observation_states": [edge[c][1] for c in cells],
                        "arrival_state": t,
                        "deadline": spec["regions"][rid][3],
                        "on_time": t <= spec["regions"][rid][3],
                    }
                    report_sources[rid] = {(edge[c][2], edge[c][3]) for c in cells}
        for relay, arrivals in enumerate(relay_arrivals):
            arrivals.sort(
                key=lambda q: (
                    packet_deadline(packets[q]),
                    packets[q]["generation_slot"],
                    packets[q]["src"],
                    q,
                )
            )
            capacity = max(0, 4 - len(queues[relay]))
            queues[relay].extend(arrivals[:capacity])
            for queued in arrivals[:capacity]:
                packets[queued]["stage"] = "queued"
            if len(arrivals) > capacity:
                for dropped in arrivals[capacity:]:
                    packets[dropped]["stage"] = "dropped"
        # A failed retry is discarded when its failure feedback arrives.
        for pid, p in packets.items():
            hi = p.get("failed_hi")
            if p.get("failed_slot") == t - 1 and hi in (0, 1) and p["retry_counts"][hi] >= 1:
                p["stage"] = "discarded"
                if p["relay"] is not None and pid in queues[p["relay"]]:
                    queues[p["relay"]].remove(pid)
        # Local sensing creates the only source-side packet evidence.
        for r in range(ROBOTS):
            for y in range(max(0, pos[r][1] - OBS_RADIUS), min(GRID, pos[r][1] + OBS_RADIUS + 1)):
                for x in range(
                    max(0, pos[r][0] - OBS_RADIUS), min(GRID, pos[r][0] + OBS_RADIUS + 1)
                ):
                    if abs(x - pos[r][0]) + abs(y - pos[r][1]) > OBS_RADIUS:
                        continue
                    cid = _cid(x, y)
                    version = ROBOTS * t + r + 1
                    local[r][cid] = (version, t)
                    explored.add(cid)
                    obs_rows.append(
                        {
                            "robot": r,
                            "cell": cid,
                            "state_t": t,
                            "version": version,
                            "obstacle": bool(_blocked(spec, x, y)),
                            "rssi": _rssi(name, pos[r], spec["edge"], t, 1),
                        }
                    )
        cur = {r: {**defaults, **by_t[t].get(r, {})} for r in range(ROBOTS)}
        # Well-formed unsafe moves follow the published stay intervention and safety penalty.
        proposed = [list(pos[r]) for r in range(ROBOTS)]
        for r in range(ROBOTS):
            a = cur[r]
            nx, ny = pos[r][0] + int(a["dx"]), pos[r][1] + int(a["dy"])
            bad = not (0 <= nx < GRID and 0 <= ny < GRID) or _blocked(spec, nx, ny)
            if not bad:
                for j in range(r):
                    if math.hypot(nx - proposed[j][0], ny - proposed[j][1]) < 1 - EPS or (
                        proposed[j] == pos[r] and [nx, ny] == pos[j]
                    ):
                        bad = True
                for j in range(r + 1, ROBOTS):
                    if [nx, ny] == pos[j]:
                        bad = True
            if bad:
                safety += 1
                proposed[r] = list(pos[r])
            else:
                proposed[r] = [nx, ny]
        submissions: list[dict[str, Any]] = []
        for r, a in cur.items():
            comm = a["comm"]
            pid = _norm(a["packet_id"])
            relay = _norm(a["relay"])
            if comm == "new":
                cells = [int(x) for x in a["cells"]]
                packet_bad = (
                    pid < 0
                    or pid in packets
                    or not (1 <= len(cells) <= 4)
                    or len(cells) != len(set(cells))
                    or any(not (0 <= c < GRID * GRID) for c in cells)
                    or raw_by_robot[r] >= 30
                    or not (relay == -1 or 0 <= relay < ROBOTS and relay != r)
                    or (relay >= 0 and cur[relay]["role"] != "relay")
                )
                if packet_bad:
                    invalid.append(f"packet:{t}:{r}")
                    continue
                if any(c not in local[r] for c in cells):
                    invalid.append(f"information:{t}:{r}")
                    continue
                raw_by_robot[r] += 1
                vers = [local[r][c][0] for c in cells]
                obs_states = [local[r][c][1] for c in cells]
                packets[pid] = {
                    "packet_id": pid,
                    "src": r,
                    "relay": None if relay < 0 else relay,
                    "cells": cells,
                    "versions": vers,
                    "obs_states": obs_states,
                    "generation_slot": t,
                    "stage": "pending",
                    "failed_slot": None,
                    "failed_hi": None,
                    "retry_counts": [0, 0],
                    "applied_arrivals": set(),
                }
                submissions.append(
                    {
                        "r": r,
                        "pid": pid,
                        "hi": 0,
                        "relay": None if relay < 0 else relay,
                        "p": packets[pid],
                        "kind": "new",
                    }
                )
            elif comm in ("retry", "forward"):
                if a["cells"] != [] or relay != -1:
                    invalid.append(f"packet_shape:{t}:{r}")
                    continue
                p = packets.get(pid)
                if not p:
                    invalid.append(f"packet_ref:{t}:{r}")
                    continue
                if comm == "retry":
                    if (
                        p["failed_slot"] != t - 1
                        or p["retry_counts"][int(p["failed_hi"] or 0)] >= 1
                        or p["failed_hi"] not in (0, 1)
                        or (p["failed_hi"] == 0 and p["src"] != r)
                        or (p["failed_hi"] == 1 and p["relay"] != r)
                        or (
                            p["failed_hi"] == 0
                            and p["relay"] is not None
                            and cur[p["relay"]]["role"] != "relay"
                        )
                        or (
                            p["failed_hi"] == 1
                            and (
                                a["role"] != "relay"
                                or a["dx"] != 0
                                or a["dy"] != 0
                                or not queues[r]
                                or queues[r][0] != pid
                            )
                        )
                    ):
                        invalid.append(f"packet_retry:{t}:{r}")
                        continue
                    submissions.append(
                        {
                            "r": r,
                            "pid": pid,
                            "hi": p["failed_hi"],
                            "relay": p["relay"],
                            "p": p,
                            "kind": "retry",
                        }
                    )
                else:
                    if (
                        p["stage"] != "queued"
                        or p["relay"] != r
                        or a["role"] != "relay"
                        or a["dx"] != 0
                        or a["dy"] != 0
                        or (not queues[r] or queues[r][0] != pid)
                    ):
                        invalid.append(f"packet_forward:{t}:{r}")
                        continue
                    submissions.append(
                        {"r": r, "pid": pid, "hi": 1, "relay": r, "p": p, "kind": "forward"}
                    )
            elif comm == "none":
                if _norm(a["packet_id"]) != -1 or _norm(a["relay"]) != -1 or a["cells"] != []:
                    invalid.append(f"packet_shape:{t}:{r}")
            else:
                if _norm(a["packet_id"]) != -1 or _norm(a["relay"]) != -1 or a["cells"] != []:
                    invalid.append(f"packet_shape:{t}:{r}")
        submissions.sort(
            key=lambda s: (
                packet_deadline(s["p"]),
                s["p"]["generation_slot"],
                s["p"]["src"],
                s["pid"],
            )
        )
        accepted = submissions[: min(4, 120 - accepted_units)]
        retried = {s["pid"] for s in submissions if s["kind"] == "retry"}
        if len(submissions) > len(accepted):
            for s in submissions[len(accepted) :]:
                invalid.append(f"channel:{t}:{s['r']}")
        accepted_units += len(accepted)
        for s in submissions:
            p, r, hi, relay = s["p"], s["r"], s["hi"], s["relay"]
            ok_accept = s in accepted
            if hi == 0:
                a, b = pos[r], pos[relay] if relay is not None else list(spec["edge"])
            else:
                a, b = pos[r], list(spec["edge"])
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            rss = _rssi(name, a, b, t, len(accepted))
            move_cost = int(proposed[r] != pos[r])
            if ok_accept and energy[r] < 1 + move_cost:
                invalid.append(f"energy:{t}:{r}")
            delivered = bool(ok_accept and d <= COMM_RADIUS and rss >= -75)
            if ok_accept:
                energy[r] -= 1
                if s["kind"] == "retry":
                    p["retry_counts"][hi] += 1
                if delivered:
                    p["stage"] = "pending_arrival"
                    p["failed_slot"] = None
                    p["failed_hi"] = None
                    pending_arrivals.setdefault(t + 1, []).append((p["packet_id"], hi))
                    if hi == 1 and p["packet_id"] in queues[r]:
                        queues[r].remove(p["packet_id"])
                else:
                    p["stage"] = "failed"
                    p["failed_slot"] = t
                    p["failed_hi"] = hi
            links.append(
                {
                    "slot": t,
                    "packet_id": p["packet_id"],
                    "hop_index": hi,
                    "src": r,
                    "dst": relay if hi == 0 and relay is not None else -1,
                    "relay": relay if relay is not None else -1,
                    "distance": d,
                    "rssi": rss,
                    "accepted": ok_accept,
                    "delivered": delivered,
                }
            )
        # An unused first-failure retry opportunity expires after this slot.
        for pid, p in packets.items():
            hi = p.get("failed_hi")
            if (
                p.get("failed_slot") == t - 1
                and hi in (0, 1)
                and p["retry_counts"][hi] == 0
                and pid not in retried
            ):
                p["stage"] = "discarded"
                if p["relay"] is not None and pid in queues[p["relay"]]:
                    queues[p["relay"]].remove(pid)
        for r in range(ROBOTS):
            if proposed[r] != pos[r]:
                if energy[r] < 1:
                    invalid.append(f"energy:{t}:{r}")
                else:
                    energy[r] -= 1
                    pos[r] = proposed[r]
            state_rows.append(
                {
                    "state_t": t + 1,
                    "robot": r,
                    "x": pos[r][0],
                    "y": pos[r][1],
                    "energy": energy[r],
                    "role": cur[r]["role"],
                }
            )
    # Slot-79 hops arrive at boundary 80; only final edge hops can affect reports.
    final_relay_arrivals: list[list[int]] = [[] for _ in range(ROBOTS)]
    for pid, hi in pending_arrivals.pop(HORIZON, []):
        p = packets.get(pid)
        if not p or hi in p["applied_arrivals"]:
            continue
        p["applied_arrivals"].add(hi)
        if hi == 0 and p["relay"] is not None:
            final_relay_arrivals[p["relay"]].append(pid)
            continue
        for cid, ver, obs_t in zip(p["cells"], p["versions"], p["obs_states"]):
            if ver > 0 and (cid not in edge or ver > edge[cid][0]):
                edge[cid] = (ver, obs_t, pid, hi)
    for relay, arrivals in enumerate(final_relay_arrivals):
        arrivals.sort(
            key=lambda q: (
                packet_deadline(packets[q]),
                packets[q]["generation_slot"],
                packets[q]["src"],
                q,
            )
        )
        capacity = max(0, 4 - len(queues[relay]))
        queues[relay].extend(arrivals[:capacity])
        for queued in arrivals[:capacity]:
            packets[queued]["stage"] = "queued"
        for dropped in arrivals[capacity:]:
            packets[dropped]["stage"] = "dropped"
    for rid in range(REGIONS):
        if reports[rid] is None:
            cells = _region_cells(spec, rid)
            if all(c in edge and edge[c][0] > 0 for c in cells):
                reports[rid] = {
                    "region_id": rid,
                    "cells": cells,
                    "versions": [edge[c][0] for c in cells],
                    "observation_states": [edge[c][1] for c in cells],
                    "arrival_state": HORIZON,
                    "deadline": spec["regions"][rid][3],
                    "on_time": HORIZON <= spec["regions"][rid][3],
                }
                report_sources[rid] = {(edge[c][2], edge[c][3]) for c in cells}
    for rid in range(REGIONS):
        if reports[rid] is None:
            cells = _region_cells(spec, rid)
            reports[rid] = {
                "region_id": rid,
                "cells": cells,
                "versions": [],
                "observation_states": [],
                "arrival_state": None,
                "deadline": spec["regions"][rid][3],
                "on_time": False,
            }
    useful = len(set().union(*(x for x in report_sources if x is not None)))
    free = [_cid(x, y) for y in range(GRID) for x in range(GRID) if not _blocked(spec, x, y)]
    return {
        "states": state_rows,
        "observations": obs_rows,
        "links": links,
        "packets": [],
        "reports": reports,
        "edge": edge,
        "explored": explored,
        "covered_value": sum(1 for c in explored if c in free),
        "value_total": len(free),
        "safety": safety,
        "energy_used": sum(ENERGY0 - e for e in energy),
        "accepted": accepted_units,
        "useful": useful,
        "packet_count": len(packets),
        "raw_by_robot": raw_by_robot,
        "queue_lengths": [len(q) for q in queues],
        "reports_raw": [r if r and r["arrival_state"] is not None else None for r in reports],
        "invalid": invalid,
    }


def _legality(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    moves = roles = packets = information = channel = True
    move_defects: list[str] = []
    role_defects: list[str] = []
    packet_defects: list[str] = []
    information_defects: list[str] = []
    channel_defects: list[str] = []
    for name in SCENARIO_NAMES:
        sc = scenarios.get(name, {})
        acts = sc.get("actions", []) if isinstance(sc, Mapping) else []
        model_ready = isinstance(acts, list)
        for i, a in enumerate(acts if isinstance(acts, list) else []):
            path = f"{name}.actions[{i}]"
            if not _action_shape(a):
                moves = roles = packets = information = channel = False
                model_ready = False
                continue
            move_valid = (
                a["dx"] in (-1, 0, 1) and a["dy"] in (-1, 0, 1) and (a["dx"] == 0 or a["dy"] == 0)
            )
            if not move_valid and len(move_defects) < 8:
                move_defects.append(
                    f"{path}.(dx,dy): actual={_brief((a['dx'], a['dy']))}; "
                    "expected one of ((0,0),(1,0),(-1,0),(0,1),(0,-1))"
                )
            moves &= move_valid
            role_valid = a.get("role") in ROLES
            if not role_valid and len(role_defects) < 8:
                role_defects.append(
                    f"{path}.role: actual={_brief(a.get('role'))}; expected one of {ROLES}"
                )
            roles &= role_valid
            packets &= (
                a["comm"] in COMMS
                and all(0 <= c < GRID * GRID for c in a["cells"])
                and len(a["cells"]) <= 4
                and len(a["cells"]) == len(set(a["cells"]))
            )
        if model_ready:
            replay = _replay(name, sc)
            invalid = replay["invalid"]
            packet_defects.extend(f"{name}:{x}" for x in invalid if x.startswith("packet"))
            information_defects.extend(
                f"{name}:{x}" for x in invalid if x.startswith("information:")
            )
            channel_defects.extend(
                f"{name}:{x}" for x in invalid if x.startswith(("energy:", "channel:"))
            )
    packets &= not packet_defects
    information &= not information_defects
    channel &= not channel_defects
    return [
        _atom("decision_legality", "moves", moves, f"defects={move_defects}"),
        _atom("decision_legality", "roles", roles, f"defects={role_defects}"),
        _atom(
            "decision_legality",
            "packets",
            packets,
            f"packet_defects={packet_defects[:8]}",
        ),
        _atom(
            "decision_legality",
            "information",
            information,
            f"information_defects={information_defects[:8]}",
        ),
        _atom(
            "decision_legality",
            "channel",
            channel,
            f"resource_defects={channel_defects[:8]}",
        ),
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    p1s = []
    p2s = []
    p3s = []
    a1s = []
    a2s = []
    for name in SCENARIO_NAMES:
        spec = SCENARIOS[name]
        d = replayed[name]
        vals = [x[2] for x in spec["regions"]]
        den = sum(vals)
        p1 = p2 = a1 = 0.0
        completed_value = 0.0
        for i, rep in enumerate(d["reports_raw"]):
            v = spec["regions"][i][2]
            if rep:
                p1 += v
                p2 += v * (
                    _clip(1 - rep["arrival_state"] / max(1, spec["regions"][i][3]))
                    if rep["on_time"]
                    else 0
                )
                freshness = (
                    sum(
                        _clip(1 - (rep["arrival_state"] - x) / 8) for x in rep["observation_states"]
                    )
                    / 4
                )
                a1 += v * freshness
                completed_value += v
            else:
                received = sum(c in d["edge"] for c in _region_cells(spec, i))
                p1 += v * 0.5 * received / 4
        p1s.append(p1 / den)
        p2s.append(p2 / den)
        a1s.append(a1 / completed_value if completed_value else 0)
        p3s.append(
            _clip(
                0.5 * d["covered_value"] / max(1, d["value_total"])
                + 0.3 * _clip(1 - d["safety"] / (HORIZON * 4))
                + 0.2 * _clip(1 - d["energy_used"] / (4 * ENERGY0))
            )
        )
        efficiency = d["useful"] / d["accepted"] if d["accepted"] else 0.0
        a2s.append(0.5 * _clip(1 - d["accepted"] / 120) + 0.5 * efficiency)
    return {
        "primary_metrics_1": _clip(sum(p1s) / 5),
        "primary_metrics_2": _clip(sum(p2s) / 5),
        "primary_metrics_3": _clip(sum(p3s) / 5),
        "auxiliary_metrics_1": _clip(sum(a1s) / 5),
        "auxiliary_metrics_2": _clip(sum(a2s) / 5),
    }


def _replay_atoms(scenarios: Mapping[str, Any], replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    ok = all(name in replayed for name in SCENARIO_NAMES)
    motion = ok and all(len(replayed[n]["states"]) == HORIZON * ROBOTS for n in SCENARIO_NAMES)
    radio = ok and all(
        all(_finite(x["rssi"]) for x in replayed[n]["links"]) for n in SCENARIO_NAMES
    )
    packets = ok and all(
        replayed[n]["accepted"] <= 120
        and max(replayed[n]["raw_by_robot"], default=0) <= 30
        and max(replayed[n]["queue_lengths"], default=0) <= 4
        for n in SCENARIO_NAMES
    )
    reports = ok and all(len(replayed[n]["reports"]) == REGIONS for n in SCENARIO_NAMES)
    metrics = False
    if ok:
        try:
            metrics = all(_finite(v) and 0 <= v <= 1 for v in _metrics(replayed).values())
        except Exception:
            metrics = False
    return [
        _atom("replay_validity", "motion", motion, "320 deterministic state transitions"),
        _atom("replay_validity", "radio", radio, "finite deterministic link outcomes"),
        _atom("replay_validity", "packets", packets, "packet budgets and queue bounds"),
        _atom("replay_validity", "reports", reports, "eight deterministic region outcomes"),
        _atom("replay_validity", "metrics", metrics, "finite normalized replay metrics"),
    ]


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(atom.share) or atom.share <= 0 for atom in atoms):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("declared atom shares must sum to 1.0")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    st = EvalState()
    remove_ok = False
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
        remove_ok = getattr(removed, "exit_code", None) == 0
        if not remove_ok:
            st.official_message = "could not remove old results.json"
        st.run_started = True
        start = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600)
            st.exit_code = getattr(result, "exit_code", None)
            run_text = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".lower()
            st.timed_out = st.exit_code is None and any(
                token in run_text for token in ("timeout", "timed out")
            )
        except Exception as exc:
            st.official_message = f"official run raised {type(exc).__name__}: {exc}"
            st.timed_out = isinstance(exc, TimeoutError) or any(
                token in str(exc).lower() for token in ("timeout", "timed out")
            )
        finally:
            st.elapsed_s = time.monotonic() - start
            st.run_finished = True
        path = Path(ws.root) / "results.json"
        st.result_file_is_new_regular = remove_ok and path.is_file() and not path.is_symlink()
        st.official_ok = (
            remove_ok
            and st.exit_code == 0
            and not st.timed_out
            and st.elapsed_s <= 600 + EPS
            and st.result_file_is_new_regular
        )
        run_status = (
            f"exit={st.exit_code}, elapsed={st.elapsed_s:.2f}s, timed_out={st.timed_out}, "
            f"new_regular_results={st.result_file_is_new_regular}"
        )
        st.official_message = (
            f"{st.official_message}; {run_status}" if st.official_message else run_status
        )
        if st.result_file_is_new_regular:
            try:
                st.data = _strict_json(path.read_text())
                st.parse_ok = True
            except Exception as exc:
                st.diagnostics["result_json"] = f"strict parse failed: {exc}"
        shape = _shape_atoms(st.data)
        st.scenarios = _index(st.data)
        evidence = _evidence_atoms(st.scenarios)
        cov, defects = _coverage(st.data)
        legal = _legality(st.scenarios)
        st.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        st.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        st.scores["canonical_coverage"] = cov
        st.scores["decision_legality"] = _score_atoms(legal, ATOM_SHARES["decision_legality"])
        can_replay = all(
            st.scores[k] >= 1 - EPS
            for k in ("result_json", "result_evidence", "canonical_coverage", "decision_legality")
        )
        replay = (
            {name: _replay(name, st.scenarios[name]) for name in SCENARIO_NAMES}
            if can_replay
            else {}
        )
        ratoms = _replay_atoms(st.scenarios, replay)
        st.replayed = replay
        st.scores["replay_validity"] = _score_atoms(ratoms, ATOM_SHARES["replay_validity"])
        parse_failure = st.diagnostics.get("result_json", "")
        st.diagnostics["result_json"] = _diagnose_atoms(shape)
        if parse_failure:
            st.diagnostics["result_json"] = f"{parse_failure}; {st.diagnostics['result_json']}"
        st.diagnostics["result_evidence"] = _diagnose_atoms(evidence)
        st.diagnostics["canonical_coverage"] = (
            f"defect_count={len(defects)}; " + " | ".join(defects[:8])
            if defects
            else "defect_count=0"
        )
        st.diagnostics["decision_legality"] = _diagnose_atoms(legal)
        st.diagnostics["replay_validity"] = _diagnose_atoms(ratoms)
        st.validity_gate = st.official_ok and all(
            st.scores.get(k, 0) >= 1 - EPS for k in GATE_CHECKS
        )
        if st.validity_gate:
            st.metrics = _metrics(replay)
        if not st.official_ok:
            for k in GATE_CHECKS:
                st.scores[k] = 0.0
                st.diagnostics[k] = (
                    f"official_run failed: {st.official_message}; {st.diagnostics[k]}"
                )
    except Exception as exc:
        st.run_finished = st.run_finished or st.run_started
        st.official_message = f"grader state failed: {type(exc).__name__}: {exc}"
        for k in GATE_CHECKS:
            st.scores[k] = 0.0
            st.diagnostics[k] = st.official_message
    _CACHE[key] = st
    return st


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular file"
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    missing: list[str] = []
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    for title in required:
        found = False
        for i, heading in enumerate(headings):
            heading_title = re.sub(r"\s+#+\s*$", "", heading.group(2)).strip()
            if heading_title != title:
                continue
            level = len(heading.group(1))
            end = len(text)
            for later in headings[i + 1 :]:
                if len(later.group(1)) <= level:
                    end = later.start()
                    break
            if text[heading.end() : end].strip():
                found = True
                break
        if not found:
            missing.append(title)
    score = max(0.0, 2.0 - 0.5 * len(missing)) / 2.0
    return score, f"missing_or_empty={missing}; normalized={score:.9g}"


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not (Path(ws.root) / "run.py").is_file() or (Path(ws.root) / "run.py").is_symlink():
        return 0.0, "run.py is not a regular file"
    r = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (
        (1.0, "static compilation passed")
        if getattr(r, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    st = await _build_state(ws)
    return (1.0, st.official_message) if st.official_ok else (0.0, st.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _simple(ws: Workspace, ident: str) -> tuple[float, str]:
    st = await _build_state(ws)
    return st.scores.get(ident, 0.0), st.diagnostics.get(ident, "")


for _id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        return await _simple(ws, ident)

    globals()[_id] = check(id=_id)(_check)


async def _metric(ws: Workspace, ident: str) -> tuple[float, str]:
    st = await _build_state(ws)
    return (
        (st.metrics or {}).get(ident, 0.0),
        f"{ident}: validity gate not satisfied"
        if st.metrics is None
        else f"{ident}: replayed value={(st.metrics or {})[ident]:.9g}",
    )


for _id in (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric_check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        return await _metric(ws, ident)

    globals()[_id] = check(id=_id)(_metric_check)
del _check, _metric_check
