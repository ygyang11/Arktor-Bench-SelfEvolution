"""Independent replay grader for signal-only decentralized exploration."""

from __future__ import annotations

import asyncio
import json
import math
import re
import stat
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
W, H, T = 24, 20, 180
ROBOTS = ("R0", "R1", "R2")
SCENARIOS = (
    "homogeneous_clutter",
    "slow_robot",
    "signal_multipath",
    "robot_failure",
    "narrow_partition",
)
PLAN = tuple(range(0, 176, 5))
ACTIONS = {"stay": (0, 0), "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
REPORT_SECTIONS = (
    "Problem Formulation",
    "Method Design",
    "Experimental Setup",
    "Results",
    "Research Analysis",
    "Conclusion and Limitations",
)
PAIR_KEYS = tuple((a, b) for a in ROBOTS for b in ROBOTS if a != b)
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.20,
        "result_json.root": 0.20,
        "result_json.scenarios": 0.25,
        "result_json.containers": 0.35,
    },
    "result_evidence": {
        "result_evidence.actions": 0.25,
        "result_evidence.planning": 0.25,
        "result_evidence.ping": 0.20,
        "result_evidence.estimates": 0.30,
    },
    "decision_legality": {
        "decision_legality.actions": 0.25,
        "decision_legality.planning": 0.25,
        "decision_legality.ping": 0.25,
        "decision_legality.estimates": 0.25,
    },
    "replay_validity": {
        "replay_validity.observation": 0.25,
        "replay_validity.signal": 0.25,
        "replay_validity.estimates": 0.25,
        "replay_validity.motion": 0.25,
    },
}

STARTS = {
    "homogeneous_clutter": ((1, 1), (22, 1), (12, 18)),
    "slow_robot": ((1, 2), (22, 2), (12, 18)),
    "signal_multipath": ((2, 2), (21, 2), (12, 17)),
    "robot_failure": ((1, 1), (22, 1), (12, 18)),
    "narrow_partition": ((1, 9), (22, 9), (12, 18)),
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


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _finite(x: Any) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(x: Any, lo: int, hi: int) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and lo <= x <= hi


def _strict_json(raw: str) -> Any:
    def reject(x: str) -> None:
        raise ValueError(f"non-finite JSON constant {x}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _rect_block(name: str, x: int, y: int) -> bool:
    if not (0 <= x < W and 0 <= y < 20):
        return True
    if name == "homogeneous_clutter":
        return (6 <= x <= 7 and 3 <= y <= 14 and y not in (8, 9)) or (
            15 <= x <= 16 and 5 <= y <= 16 and y not in (11, 12)
        )
    if name == "slow_robot":
        return (9 <= x <= 10 and 0 <= y <= 7) or (9 <= x <= 10 and 10 <= y <= 19)
    if name == "signal_multipath":
        return (7 <= x <= 9 and 5 <= y <= 14) or (15 <= x <= 17 and 5 <= y <= 14)
    if name == "robot_failure":
        return (
            10 <= x <= 13 and 4 <= y <= 15 and (x, y) not in {(10, 4), (13, 4), (10, 15), (13, 15)}
        )
    if name == "narrow_partition":
        room = (
            (0 <= x <= 3 and 8 <= y <= 11)
            or (20 <= x <= 23 and 8 <= y <= 11)
            or (0 <= x <= 3 and 16 <= y <= 19)
            or (20 <= x <= 23 and 0 <= y <= 3)
        )
        return not (8 <= y <= 11 or 10 <= x <= 13 or room)
    return False


def _true(name: str, x: int, y: int, slot: int) -> bool:
    if _rect_block(name, x, y):
        return False
    return True


def _noise_level(name: str, slot: int) -> int:
    if name == "signal_multipath" and 40 <= slot <= 119:
        return 2
    if name in {"robot_failure", "narrow_partition"}:
        return 1
    return 0


def _is_live(name: str, robot: str, slot: int) -> bool:
    return not (name == "robot_failure" and robot == "R2" and slot >= 85)


def _round_grid_coordinate(value: float) -> int:
    """Public DDA convention: non-negative half-integers round upward."""
    return math.floor(value + 0.5)


def _dda_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    steps = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    cells = []
    for i in range(steps + 1):
        x = _round_grid_coordinate(a[0] + (b[0] - a[0]) * i / max(1, steps))
        y = _round_grid_coordinate(a[1] + (b[1] - a[1]) * i / max(1, steps))
        cell = (x, y)
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _line_blocked(name: str, a: tuple[int, int], b: tuple[int, int]) -> int:
    return len({cell for cell in _dda_cells(a, b) if not _true(name, *cell, 0)})


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _signal_measurement(
    name: str,
    a: str,
    b: str,
    positions: Mapping[str, tuple[int, int]],
    slot: int,
) -> dict[str, Any]:
    """Return the public environment's two endpoint observations for one pilot."""
    if a not in ROBOTS or b not in ROBOTS or a == b or a not in positions or b not in positions:
        return {"success": False, "endpoints": {}}
    start, end = positions[a], positions[b]
    distance = math.dist(start, end)
    cells = _dda_cells(start, end)
    blocked = []
    for index, cell in enumerate(cells):
        if not _true(name, *cell, 0) and cell not in {item[1] for item in blocked}:
            blocked.append((index, cell))
    i, j = ROBOTS.index(a), ROBOTS.index(b)
    threshold = 0.04 + 0.08 * len(blocked) + 0.12 * _noise_level(name, slot)
    success = (
        distance <= 14
        and len(blocked) <= 2
        and ((17 * i + 11 * j + 5 * slot) % 100) / 100 >= threshold
    )
    if not success:
        return {"success": False, "endpoints": {}}
    measured_range = distance + 0.08 * (((7 * i + 3 * j + slot) % 5) - 2)
    direct = _normalize_angle(
        math.atan2(end[1] - start[1], end[0] - start[0]) + 0.03 * (((5 * i + 7 * j + slot) % 5) - 2)
    )
    strength = 1.0 / (1.0 + distance)
    candidates: list[tuple[float, float, tuple[int, int] | None]] = [(direct, strength, None)]
    for index, cell in blocked:
        previous = cells[max(0, index - 1)]
        dx = cell[0] - previous[0]
        if dx != 0:
            reflected = _normalize_angle(-direct)
        else:
            reflected = _normalize_angle(math.pi - direct)
        candidates.append((reflected, strength, cell))
    candidates.sort(key=lambda item: (-item[1], item[0], item[2] or (-1, -1)))
    sender = [(angle, power) for angle, power, _ in candidates]
    receiver_candidates = [
        (_normalize_angle(angle + math.pi), power, cell) for angle, power, cell in candidates
    ]
    receiver_candidates.sort(key=lambda item: (-item[1], item[0], item[2] or (-1, -1)))
    receiver = [(angle, power) for angle, power, _ in receiver_candidates]
    return {
        "success": True,
        "endpoints": {
            a: {"range": measured_range, "bearing_candidates": sender},
            b: {"range": measured_range, "bearing_candidates": receiver},
        },
    }


def _signal_ok(
    name: str, a: str, b: str, positions: Mapping[str, tuple[int, int]], slot: int
) -> bool:
    return bool(_signal_measurement(name, a, b, positions, slot)["success"])


def _neighbors(c: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = c
    return [
        (nx, ny)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if 0 <= nx < W and 0 <= ny < H
    ]


def _frontiers(local: Mapping[tuple[int, int], int]) -> list[tuple[str, list[tuple[int, int]]]]:
    cells = {
        c for c, v in local.items() if v == 1 and any(local.get(n, 0) == 0 for n in _neighbors(c))
    }
    comps = []
    while cells:
        seed = min(cells)
        q = [seed]
        cells.remove(seed)
        comp = [seed]
        while q:
            c = q.pop()
            for x in (
                (c[0] - 1, c[1]),
                (c[0] + 1, c[1]),
                (c[0], c[1] - 1),
                (c[0], c[1] + 1),
                (c[0] - 1, c[1] - 1),
                (c[0] - 1, c[1] + 1),
                (c[0] + 1, c[1] - 1),
                (c[0] + 1, c[1] + 1),
            ):
                if x in cells:
                    cells.remove(x)
                    q.append(x)
                    comp.append(x)
        comps.append((f"F{seed[0]}_{seed[1]}", sorted(comp)))
    return sorted(comps)


def _reachable(
    local: Mapping[tuple[int, int], int], start: tuple[int, int], target: tuple[int, int]
) -> bool:
    q = deque([start])
    seen = {start}
    while q:
        c = q.popleft()
        if c == target:
            return True
        for n in _neighbors(c):
            if n not in seen and local.get(n) == 1:
                seen.add(n)
                q.append(n)
    return False


def _viewpoint(comp: Sequence[tuple[int, int]]) -> tuple[int, int]:
    mx = sum(x for x, _ in comp) / len(comp)
    my = sum(y for _, y in comp) / len(comp)
    ordered = sorted(comp, key=lambda c: (abs(c[0] - mx) + abs(c[1] - my), c[0], c[1]))
    # The public rule retains the nearest cell and the two farthest cells,
    # with de-duplication for one/two-cell components.  The caller uses this
    # helper for the canonical nearest candidate; `_viewpoint_candidates`
    # exposes the complete set for legality checks.
    return ordered[0]


def _viewpoint_candidates(comp: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not comp:
        return []
    mx = sum(x for x, _ in comp) / len(comp)
    my = sum(y for _, y in comp) / len(comp)
    near = sorted(comp, key=lambda c: (abs(c[0] - mx) + abs(c[1] - my), c[0], c[1]))[0]
    far = sorted(comp, key=lambda c: (-abs(c[0] - mx) - abs(c[1] - my), c[0], c[1]))
    out = [near]
    for c in far:
        if c not in out:
            out.append(c)
        if len(out) == 3:
            break
    return out


def _records(data: Any) -> dict[tuple[str, str, int, str], Any]:
    out = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return out
    for sc in data["scenarios"]:
        if not isinstance(sc, dict):
            continue
        name = sc.get("scenario")
        if not isinstance(name, str) or name not in SCENARIOS:
            continue
        for rob in sc.get("robots", []) if isinstance(sc.get("robots"), list) else []:
            if isinstance(rob, dict) and isinstance(rob.get("robot_id"), str):
                for a in (
                    rob.get("intended_actions", [])
                    if isinstance(rob.get("intended_actions"), list)
                    else []
                ):
                    if (
                        isinstance(a, dict)
                        and isinstance(a.get("slot"), int)
                        and not isinstance(a.get("slot"), bool)
                    ):
                        key = (name, "action", a["slot"], rob["robot_id"])
                        if key not in out:
                            out[key] = a
        for p in (
            sc.get("planning_opportunities", [])
            if isinstance(sc.get("planning_opportunities"), list)
            else []
        ):
            if (
                isinstance(p, dict)
                and isinstance(p.get("slot"), int)
                and not isinstance(p.get("slot"), bool)
                and isinstance(p.get("robot_id"), str)
            ):
                key = (name, "plan", p["slot"], p.get("robot_id"))
                if key not in out:
                    out[key] = p
        for p in (
            sc.get("ping_opportunities", [])
            if isinstance(sc.get("ping_opportunities"), list)
            else []
        ):
            if (
                isinstance(p, dict)
                and isinstance(p.get("slot"), int)
                and not isinstance(p.get("slot"), bool)
            ):
                key = (name, "ping", p["slot"], "__slot__")
                if key not in out:
                    out[key] = p
        for e in (
            sc.get("relative_estimates", [])
            if isinstance(sc.get("relative_estimates"), list)
            else []
        ):
            if (
                isinstance(e, dict)
                and isinstance(e.get("planning_slot"), int)
                and not isinstance(e.get("planning_slot"), bool)
                and isinstance(e.get("observer"), str)
                and isinstance(e.get("neighbor"), str)
            ):
                key = (
                    name,
                    "estimate",
                    e["planning_slot"],
                    f"{e.get('observer')}->{e.get('neighbor')}",
                )
                if key not in out:
                    out[key] = e
    return out


def _result_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    c = {k: [True, "ok"] for k in ATOM_SHARES["result_json"]}
    if not parse_ok:
        return [
            AtomVerdict(k, False, "strict JSON parse failed", share)
            for k, share in ATOM_SHARES["result_json"].items()
        ]
    if not isinstance(data, dict) or set(data) != {"scenarios"}:
        c["result_json.root"] = [False, "root keys must be exactly scenarios"]
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        c["result_json.scenarios"] = [False, "scenarios must be an array"]
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        c["result_json.containers"] = [False, "scenario containers unavailable"]
    else:
        for s in data["scenarios"]:
            if not isinstance(s, dict) or set(s) != {
                "scenario",
                "robots",
                "planning_opportunities",
                "ping_opportunities",
                "relative_estimates",
            }:
                c["result_json.containers"] = [False, "scenario container keys invalid"]
                break
            if any(
                not isinstance(s.get(key), list)
                for key in (
                    "robots",
                    "planning_opportunities",
                    "ping_opportunities",
                    "relative_estimates",
                )
            ):
                c["result_json.containers"] = [False, "scenario child container must be an array"]
                break
    return [AtomVerdict(k, v[0], v[1], ATOM_SHARES["result_json"][k]) for k, v in c.items()]


def _positive_definite(cxx: Any, cxy: Any, cyy: Any) -> bool:
    if not all(_finite(value) for value in (cxx, cxy, cyy)):
        return False
    x, y, z = float(cxx), float(cxy), float(cyy)
    if x <= 0.0 or z <= 0.0:
        return False
    scale = max(abs(x), abs(y), abs(z))
    if scale <= 0.0 or not math.isfinite(scale):
        return False
    a, b, c = x / scale, y / scale, z / scale
    return a * c - b * b > 0.0


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    c = {k: [True, "ok"] for k in ATOM_SHARES["result_evidence"]}
    if not isinstance(data, dict):
        return [
            AtomVerdict(k, False, "no result data", v)
            for k, v in ATOM_SHARES["result_evidence"].items()
        ]
    for s in data.get("scenarios", []) if isinstance(data.get("scenarios"), list) else []:
        if not isinstance(s, dict):
            continue
        for r in s.get("robots", []) if isinstance(s.get("robots"), list) else []:
            if (
                not isinstance(r, dict)
                or set(r) != {"robot_id", "intended_actions"}
                or r.get("robot_id") not in ROBOTS
                or not isinstance(r.get("intended_actions"), list)
            ):
                c["result_evidence.actions"] = [False, "robot action record keys or list invalid"]
            elif any(
                not isinstance(a, dict)
                or set(a) != {"slot", "action"}
                or not _integer(a.get("slot"), 0, T - 1)
                or not isinstance(a.get("action"), str)
                or a.get("action") not in ACTIONS
                for a in r["intended_actions"]
            ):
                c["result_evidence.actions"] = [False, "action evidence has invalid fields"]
        for p in (
            s.get("planning_opportunities", [])
            if isinstance(s.get("planning_opportunities"), list)
            else []
        ):
            if (
                not isinstance(p, dict)
                or set(p)
                != {"robot_id", "slot", "selected_frontier_id", "selected_viewpoint", "standby"}
                or p.get("robot_id") not in ROBOTS
                or p.get("slot") not in PLAN
            ):
                c["result_evidence.planning"] = [False, "planning record keys invalid"]
                continue
            standby = p.get("standby")
            frontier_id = p.get("selected_frontier_id")
            viewpoint = p.get("selected_viewpoint")
            if not isinstance(standby, bool):
                c["result_evidence.planning"] = [False, "planning standby type invalid"]
            elif frontier_id is not None and not (
                isinstance(frontier_id, str) and re.fullmatch(r"F\d+_\d+", frontier_id)
            ):
                c["result_evidence.planning"] = [False, "planning frontier ID format invalid"]
            elif viewpoint is not None and (
                not isinstance(viewpoint, dict)
                or set(viewpoint) != {"x", "y"}
                or not _integer(viewpoint.get("x"), 0, W - 1)
                or not _integer(viewpoint.get("y"), 0, H - 1)
            ):
                c["result_evidence.planning"] = [False, "planning viewpoint fields invalid"]
            elif standby and (frontier_id is not None or viewpoint is not None):
                c["result_evidence.planning"] = [False, "standby requires both target fields null"]
            elif not standby and (frontier_id is None or viewpoint is None):
                c["result_evidence.planning"] = [False, "non-standby requires both target fields"]
        for p in (
            s.get("ping_opportunities", []) if isinstance(s.get("ping_opportunities"), list) else []
        ):
            if (
                not isinstance(p, dict)
                or set(p) != {"slot", "robot_a", "robot_b"}
                or not _integer(p.get("slot"), 0, T - 1)
            ):
                c["result_evidence.ping"] = [False, "ping record keys invalid"]
            elif not (
                (p.get("robot_a") is None and p.get("robot_b") is None)
                or (
                    p.get("robot_a") in ROBOTS
                    and p.get("robot_b") in ROBOTS
                    and p.get("robot_a") != p.get("robot_b")
                )
            ):
                c["result_evidence.ping"] = [False, "ping pair types invalid"]
        for e in (
            s.get("relative_estimates", []) if isinstance(s.get("relative_estimates"), list) else []
        ):
            if not isinstance(e, dict) or set(e) != {
                "observer",
                "neighbor",
                "planning_slot",
                "estimate_x",
                "estimate_y",
                "cov_xx",
                "cov_xy",
                "cov_yy",
                "age",
                "valid",
                "source_measurement_ids",
            }:
                c["result_evidence.estimates"] = [False, "estimate record keys invalid"]
            elif (
                e.get("observer") not in ROBOTS
                or e.get("neighbor") not in ROBOTS
                or e.get("observer") == e.get("neighbor")
                or not _integer(e.get("planning_slot"), 0, T - 5)
                or not isinstance(e.get("valid"), bool)
                or not isinstance(e.get("source_measurement_ids"), list)
                or any(not isinstance(x, str) for x in e["source_measurement_ids"])
            ):
                c["result_evidence.estimates"] = [False, "estimate evidence types invalid"]
                continue
            valid = e["valid"]
            mids = e["source_measurement_ids"]
            if valid:
                if (
                    not all(
                        _finite(e.get(key))
                        for key in ("estimate_x", "estimate_y", "cov_xx", "cov_xy", "cov_yy")
                    )
                    or not _positive_definite(e.get("cov_xx"), e.get("cov_xy"), e.get("cov_yy"))
                    or not _integer(e.get("age"), 0, 12)
                    or not mids
                ):
                    c["result_evidence.estimates"] = [False, "valid estimate fields invalid"]
            elif (
                any(
                    e.get(key) is not None
                    for key in ("estimate_x", "estimate_y", "cov_xx", "cov_xy", "cov_yy", "age")
                )
                or mids != []
            ):
                c["result_evidence.estimates"] = [False, "invalid estimate must use null fields"]
            if len(set(mids)) != len(mids) or any(
                re.fullmatch(r"M\d+_R[0-2]_R[0-2]", mid) is None for mid in mids
            ):
                c["result_evidence.estimates"] = [
                    False,
                    "estimate source ID format or uniqueness invalid",
                ]
    return [AtomVerdict(k, v[0], v[1], ATOM_SHARES["result_evidence"][k]) for k, v in c.items()]


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
        return 0.0, ["missing scenarios"]
    defects: list[str] = []
    seen = [
        scenario.get("scenario")
        for scenario in data["scenarios"]
        if isinstance(scenario, dict) and isinstance(scenario.get("scenario"), str)
    ]
    defects.extend(_key_defects(seen, set(SCENARIOS), "scenario"))
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict):
            continue
        name = scenario.get("scenario")
        if name not in SCENARIOS:
            continue
        robots = scenario.get("robots", []) if isinstance(scenario.get("robots"), list) else []
        robot_ids = [
            robot.get("robot_id")
            for robot in robots
            if isinstance(robot, dict) and isinstance(robot.get("robot_id"), str)
        ]
        defects.extend(_key_defects(robot_ids, set(ROBOTS), f"{name}/robot"))
        for robot in robots:
            if not isinstance(robot, dict) or robot.get("robot_id") not in ROBOTS:
                continue
            actions = (
                robot.get("intended_actions", [])
                if isinstance(robot.get("intended_actions"), list)
                else []
            )
            action_keys = [
                action.get("slot")
                for action in actions
                if isinstance(action, dict)
                and isinstance(action.get("slot"), int)
                and not isinstance(action.get("slot"), bool)
            ]
            defects.extend(
                _key_defects(action_keys, set(range(T)), f"{name}/{robot.get('robot_id')}/action")
            )
        plans = (
            scenario.get("planning_opportunities", [])
            if isinstance(scenario.get("planning_opportunities"), list)
            else []
        )
        plan_keys = [
            (plan.get("robot_id"), plan.get("slot"))
            for plan in plans
            if isinstance(plan, dict)
            and isinstance(plan.get("robot_id"), str)
            and isinstance(plan.get("slot"), int)
            and not isinstance(plan.get("slot"), bool)
        ]
        defects.extend(
            _key_defects(
                plan_keys, {(robot, slot) for robot in ROBOTS for slot in PLAN}, f"{name}/plan"
            )
        )
        pings = (
            scenario.get("ping_opportunities", [])
            if isinstance(scenario.get("ping_opportunities"), list)
            else []
        )
        ping_keys = [
            ping.get("slot")
            for ping in pings
            if isinstance(ping, dict)
            and isinstance(ping.get("slot"), int)
            and not isinstance(ping.get("slot"), bool)
        ]
        defects.extend(_key_defects(ping_keys, set(range(T)), f"{name}/ping"))
        estimates = (
            scenario.get("relative_estimates", [])
            if isinstance(scenario.get("relative_estimates"), list)
            else []
        )
        estimate_keys = [
            (estimate.get("observer"), estimate.get("neighbor"), estimate.get("planning_slot"))
            for estimate in estimates
            if isinstance(estimate, dict)
            and isinstance(estimate.get("observer"), str)
            and isinstance(estimate.get("neighbor"), str)
            and isinstance(estimate.get("planning_slot"), int)
            and not isinstance(estimate.get("planning_slot"), bool)
        ]
        defects.extend(
            _key_defects(
                estimate_keys,
                {(observer, neighbour, slot) for observer, neighbour in PAIR_KEYS for slot in PLAN},
                f"{name}/estimate",
            )
        )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _legality(data: Any) -> list[AtomVerdict]:
    c = {k: [True, "ok"] for k in ATOM_SHARES["decision_legality"]}
    if not isinstance(data, dict):
        return [
            AtomVerdict(k, False, "no data", v) for k, v in ATOM_SHARES["decision_legality"].items()
        ]
    for s in data.get("scenarios", []) if isinstance(data.get("scenarios"), list) else []:
        if not isinstance(s, dict):
            continue
        name = s.get("scenario")
        if name not in SCENARIOS:
            continue
        action_rows: dict[tuple[str, int], str] = {}
        for robot_row in s.get("robots", []) if isinstance(s.get("robots"), list) else []:
            if not isinstance(robot_row, dict) or robot_row.get("robot_id") not in ROBOTS:
                continue
            robot = robot_row["robot_id"]
            actions = (
                robot_row.get("intended_actions", [])
                if isinstance(robot_row.get("intended_actions"), list)
                else []
            )
            for action_row in actions:
                if (
                    isinstance(action_row, dict)
                    and _integer(action_row.get("slot"), 0, T - 1)
                    and isinstance(action_row.get("action"), str)
                    and action_row.get("action") in ACTIONS
                ):
                    action_rows.setdefault((robot, action_row["slot"]), action_row["action"])
        for (robot, slot), action in action_rows.items():
            if name == "robot_failure" and robot == "R2" and slot >= 85 and action != "stay":
                c["decision_legality.actions"] = [
                    False,
                    f"{name}/{robot}/{slot}: failed robot must request stay",
                ]

        plans = (
            s.get("planning_opportunities", [])
            if isinstance(s.get("planning_opportunities"), list)
            else []
        )
        for plan in plans:
            if (
                not isinstance(plan, dict)
                or plan.get("robot_id") not in ROBOTS
                or plan.get("slot") not in PLAN
                or not isinstance(plan.get("standby"), bool)
            ):
                continue
            robot, slot = plan["robot_id"], plan["slot"]
            if plan["standby"]:
                for action_slot in range(slot, min(slot + 5, T)):
                    action = action_rows.get((robot, action_slot))
                    if (
                        action is not None
                        and action != "stay"
                        and c["decision_legality.actions"][0]
                    ):
                        c["decision_legality.actions"] = [
                            False,
                            f"{name}/{robot}/{action_slot}: active standby requires stay",
                        ]
            if name == "robot_failure" and robot == "R2" and slot >= 85:
                if (
                    plan["standby"] is not True
                    or plan.get("selected_frontier_id") is not None
                    or plan.get("selected_viewpoint") is not None
                ):
                    c["decision_legality.planning"] = [
                        False,
                        f"{name}/{robot}/{slot}: failed robot requires null standby plan",
                    ]

        pings = (
            s.get("ping_opportunities", []) if isinstance(s.get("ping_opportunities"), list) else []
        )
        nonidle = 0
        for ping in pings:
            if not isinstance(ping, dict) or not _integer(ping.get("slot"), 0, T - 1):
                continue
            a, b, slot = ping.get("robot_a"), ping.get("robot_b"), ping["slot"]
            valid_pair = (a is None and b is None) or (a in ROBOTS and b in ROBOTS and a != b)
            if not valid_pair:
                continue
            if a is not None:
                nonidle += 1
                if name == "robot_failure" and slot >= 85 and "R2" in (a, b):
                    c["decision_legality.ping"] = [
                        False,
                        f"{name}/{slot}: failed robot cannot participate in a pilot",
                    ]
        if nonidle > 90:
            c["decision_legality.ping"] = [
                False,
                f"{name}: non-idle pilot count {nonidle} exceeds 90",
            ]

        estimates = (
            s.get("relative_estimates", []) if isinstance(s.get("relative_estimates"), list) else []
        )
        for estimate in estimates:
            if (
                not isinstance(estimate, dict)
                or estimate.get("planning_slot") not in PLAN
                or not isinstance(estimate.get("valid"), bool)
            ):
                continue
            if (
                name == "robot_failure"
                and estimate["planning_slot"] >= 85
                and "R2" in (estimate.get("observer"), estimate.get("neighbor"))
                and estimate["valid"]
            ):
                c["decision_legality.estimates"] = [
                    False,
                    f"{name}/{estimate.get('observer')}->{estimate.get('neighbor')}/"
                    f"{estimate['planning_slot']}: failed-robot estimate must be invalid",
                ]
    return [AtomVerdict(k, v[0], v[1], ATOM_SHARES["decision_legality"][k]) for k, v in c.items()]


class Replay:
    def __init__(self, name: str, rows: Mapping[str, Any]) -> None:
        self.name = name
        self.rows = rows
        self.pos = {r: STARTS[name][i] for i, r in enumerate(ROBOTS)}
        self.local = {r: {} for r in ROBOTS}
        self.discovered = set()
        self.per_robot = {r: set() for r in ROBOTS}
        self.first_discoverer = {}
        self.usable_coverage = []
        self.usable_sets = []
        self.measurements = {}
        self.arrivals = {}
        self.errors = {"observation": [], "signal": [], "estimates": [], "motion": []}
        self.safety = 0
        self.moves = 0
        self.pings = 0
        self.success = 0
        self.coverage = []
        self.open_counts = []
        self.estimate_errors = []
        self.plan_snapshots = {}
        self.positions_by_slot = {}
        self.hgrid = {r: [[0.0 for _ in range(6)] for _ in range(5)] for r in ROBOTS}
        self.hgrid_history = []
        self.per_robot_history = []
        self.local_history = []
        self.standby_by_slot = []
        self.termination_slot = None
        self.t95_slot = None

    def observe(self, r: str, slot: int) -> None:
        x, y = self.pos[r]
        for yy in range(max(0, y - 2), min(20, y + 3)):
            for xx in range(max(0, x - 2), min(24, x + 3)):
                c = (xx, yy)
                val = 1 if _true(self.name, xx, yy, slot) else 2
                old = self.local[r].get(c, 0)
                self.local[r][c] = 2 if old == 2 or val == 2 else val
                if val == 1:
                    self.per_robot[r].add(c)
                    self.discovered.add(c)
                    self.first_discoverer.setdefault(c, r)

    def process_arrivals(self, slot: int) -> None:
        # Measurements are materialized at ping time with arrival=ping+1;
        # this hook makes the event boundary explicit for replay diagnostics.
        for mid, m in self.measurements.items():
            if m.get("arrival") == slot and _is_live(self.name, m.get("receiver"), slot):
                self.arrivals[mid] = m

    def update_hgrid_and_check_plan(self, slot: int) -> None:
        if slot not in PLAN:
            return
        for r in ROBOTS:
            if not _is_live(self.name, r, slot):
                continue
            grid = self.hgrid[r]
            x, y = self.pos[r]
            grid[min(4, y // 4)][min(5, x // 4)] += 1.0
            for n in ROBOTS:
                if n == r or not _is_live(self.name, n, slot):
                    continue
                e = self.rows.get(("estimate", slot, f"{r}->{n}"))
                if not isinstance(e, Mapping) or e.get("valid") is not True:
                    continue
                mids = e.get("source_measurement_ids")
                if (
                    not isinstance(mids, list)
                    or not mids
                    or any(mid not in self.arrivals for mid in mids)
                ):
                    continue
                if not _integer(e.get("age"), 0, 12) or e.get("age") > 12:
                    continue
                if (
                    not _finite(e.get("estimate_x"))
                    or not _finite(e.get("estimate_y"))
                    or not _finite(e.get("cov_xx"))
                    or not _finite(e.get("cov_yy"))
                ):
                    continue
                w = _clip(1.0 - (float(e["cov_xx"]) + float(e["cov_yy"])) / 8.0)
                ex, ey = x + float(e["estimate_x"]), y + float(e["estimate_y"])
                if 0 <= ex < W and 0 <= ey < H:
                    grid[min(4, int(ey) // 4)][min(5, int(ex) // 4)] += w
            self.hgrid[r] = grid
            fronts = self.plan_snapshots.get(slot, {}).get(r, {}).get("frontiers", [])
            reachable_low = False
            local = self.plan_snapshots.get(slot, {}).get(r, {}).get("local", {})
            pos = self.plan_snapshots.get(slot, {}).get(r, {}).get("position", self.pos[r])
            for _, comp in fronts:
                candidates = _viewpoint_candidates(comp)
                if not candidates or not any(_reachable(local, pos, point) for point in candidates):
                    continue
                ov = 0.0
                for cx, cy in comp:
                    ov += _clip(grid[min(4, cy // 4)][min(5, cx // 4)] / 3.0)
                if ov / len(comp) < 0.5:
                    reachable_low = True
                    break
            filled = sum(v >= 3.0 for row in grid for v in row) / 30.0
            p = self.rows.get(("plan", slot, r))
            has_reachable = any(
                any(_reachable(local, pos, point) for point in _viewpoint_candidates(comp))
                for _, comp in fronts
            )
            if (
                isinstance(p, Mapping)
                and p.get("standby") is True
                and has_reachable
                and (filled < 0.65 or reachable_low)
            ):
                self.errors["motion"].append(f"{self.name}/{r}/{slot}: standby eligibility failed")
        self.hgrid_history.append({r: [row[:] for row in self.hgrid[r]] for r in ROBOTS})

    def _standby_flags(self, slot: int) -> dict[str, bool]:
        flags = {}
        for r in ROBOTS:
            latest = None
            for t in PLAN:
                if t <= slot:
                    row = self.rows.get(("plan", t, r))
                    if isinstance(row, Mapping):
                        latest = row.get("standby")
            flags[r] = latest is True
        if self.name == "robot_failure" and slot >= 85:
            flags.pop("R2", None)
        return flags

    def _evaluate_termination(self, boundary: int, usable_count: int, open_count: int) -> None:
        if self.t95_slot is None and usable_count >= 0.95 * max(1, open_count):
            self.t95_slot = boundary
        if self.termination_slot is not None:
            return
        if self.t95_slot is not None:
            self.termination_slot = boundary
            return
        if len(self.standby_by_slot) >= 12:
            recent = self.standby_by_slot[-12:]
            if recent and all(all(v for v in x.values()) for x in recent):
                self.termination_slot = boundary

    def make_measurements(self, slot: int, a: str | None, b: str | None) -> None:
        if a is None or b is None or a not in ROBOTS or b not in ROBOTS or a == b:
            return
        self.pings += 1
        signal = _signal_measurement(self.name, a, b, self.pos, slot)
        ok = bool(signal["success"])
        if ok:
            self.success += 1
        for receiver, other in ((a, b), (b, a)):
            mid = f"M{slot}_{receiver}_{other}"
            endpoint = signal.get("endpoints", {}).get(receiver, {})
            self.measurements[mid] = {
                "slot": slot,
                "receiver": receiver,
                "other": other,
                "success": ok,
                "arrival": slot + 1 if ok else None,
                "range": endpoint.get("range") if ok else None,
                "bearing_candidates": endpoint.get("bearing_candidates", []) if ok else [],
            }

    def move(self, slot: int) -> None:
        req = {r: self.rows.get(("action", slot, r), {}).get("action", "stay") for r in ROBOTS}
        target = {}
        if self.name == "robot_failure" and slot >= 85:
            req["R2"] = "stay"
        if self.name == "slow_robot" and slot % 2 == 1:
            req["R1"] = "stay"
        for r, a in req.items():
            if not isinstance(a, str) or a not in ACTIONS:
                self.errors["motion"].append(f"{self.name}/{slot}/{r}: invalid action")
                a = "stay"
                req[r] = a
            dx, dy = ACTIONS[a]
            p = (self.pos[r][0] + dx, self.pos[r][1] + dy)
            if a != "stay" and not _true(self.name, *p, slot):
                self.safety += 1
                req[r] = "stay"
                target[r] = self.pos[r]
            else:
                target[r] = p if a != "stay" else self.pos[r]
        rejected = set()
        for a, b in PAIR_KEYS:
            if (
                a < b
                and target[a] == self.pos[b]
                and target[b] == self.pos[a]
                and target[a] != self.pos[a]
            ):
                rejected.update((a, b))
        for cell in set(target.values()):
            cs = [r for r in ROBOTS if target[r] == cell and target[r] != self.pos[r]]
            if len(cs) > 1:

                def priority(r: str) -> tuple[int, int]:
                    tx, ty = target[r]
                    unknown = 0
                    for yy in range(max(0, ty - 2), min(H, ty + 3)):
                        for xx in range(max(0, tx - 2), min(W, tx + 3)):
                            if self.local[r].get((xx, yy), 0) == 0:
                                unknown += 1
                    return (unknown, ROBOTS.index(r))

                winner = min(cs, key=priority)
                rejected.update(r for r in cs if r != winner)
        # A move into a cell occupied by a robot whose request remains still is rejected;
        # repeat until no newly rejected move changes occupancy.
        changed = True
        while changed:
            changed = False
            for r in ROBOTS:
                if r in rejected or req[r] == "stay" or target[r] == self.pos[r]:
                    continue
                occupant = next((q for q in ROBOTS if q != r and self.pos[q] == target[r]), None)
                if occupant is not None and (req[occupant] == "stay" or occupant in rejected):
                    rejected.add(r)
                    changed = True
        for r in ROBOTS:
            if req[r] != "stay" and r not in rejected and target[r] != self.pos[r]:
                self.pos[r] = target[r]
                self.moves += 1
            elif req[r] != "stay" and r in rejected:
                self.safety += 1

    def run(self) -> dict[str, Any]:
        standby_slots = 0
        active_slots = 0
        for (kind, t, key), row in self.rows.items():
            if kind == "plan" and isinstance(row, Mapping):
                if row.get("standby") is True:
                    standby_slots += 1
                else:
                    active_slots += 1
        for slot in range(T):
            self.process_arrivals(slot)
            for r in ROBOTS:
                if _is_live(self.name, r, slot):
                    self.observe(r, slot)
            self.local_history.append({r: dict(self.local[r]) for r in ROBOTS})
            self.positions_by_slot[slot] = dict(self.pos)
            if slot in PLAN:
                open_count = sum(_true(self.name, x, y, slot) for x in range(W) for y in range(H))
                usable_now = set(self.discovered)
                if self.name == "robot_failure" and slot >= 85:
                    survivors = set().union(self.per_robot["R0"], self.per_robot["R1"])
                    usable_now = {
                        c
                        for c in usable_now
                        if self.first_discoverer.get(c) != "R2" or c in survivors
                    }
                self._evaluate_termination(slot, len(usable_now), open_count)
                self.plan_snapshots[slot] = {
                    r: {
                        "frontiers": _frontiers(self.local[r]),
                        "position": self.pos[r],
                        "local": dict(self.local[r]),
                    }
                    for r in ROBOTS
                }
                self.update_hgrid_and_check_plan(slot)
            ping = self.rows.get(("ping", slot, "__slot__"), {})
            a = ping.get("robot_a") if isinstance(ping, dict) else None
            b = ping.get("robot_b") if isinstance(ping, dict) else None
            if a is not None and not (
                self.name == "robot_failure" and slot >= 85 and "R2" in (a, b)
            ):
                self.make_measurements(slot, a, b)
            self.move(slot)
            op = sum(_true(self.name, x, y, slot) for x in range(W) for y in range(20))
            usable = set(self.discovered)
            if self.name == "robot_failure" and slot >= 85:
                survivors = set().union(self.per_robot["R0"], self.per_robot["R1"])
                usable = {
                    c for c in usable if self.first_discoverer.get(c) != "R2" or c in survivors
                }
            self.open_counts.append(op)
            self.coverage.append(len(self.discovered))
            self.usable_sets.append(usable)
            self.usable_coverage.append(len(usable))
            self.per_robot_history.append({r: set(v) for r, v in self.per_robot.items()})
            self.standby_by_slot.append(self._standby_flags(slot))
        # The terminal boundary has no decisions or observations.  It only
        # materializes the result of a successful slot-179 pilot and applies
        # the public age-expiry boundary.
        self.process_arrivals(T)
        final_open = self.open_counts[-1] if self.open_counts else 0
        final_usable = self.usable_coverage[-1] if self.usable_coverage else 0
        self._evaluate_termination(T, final_usable, final_open)
        for r in ROBOTS:
            x, y = self.pos[r]
            if not (0 <= x < W and 0 <= y < H):
                self.errors["observation"].append(f"{self.name}/{r}: final position out of bounds")
            local_traversable = {c for c, value in self.local[r].items() if value == 1}
            if local_traversable != self.per_robot[r]:
                self.errors["observation"].append(
                    f"{self.name}/{r}: local traversable/discovery mismatch"
                )
            if any(_rect_block(self.name, x, y) for x, y in local_traversable):
                self.errors["observation"].append(
                    f"{self.name}/{r}: blocked cell marked traversable"
                )
        for slot, positions in self.positions_by_slot.items():
            for r, (x, y) in positions.items():
                if not (0 <= x < W and 0 <= y < H) or _rect_block(self.name, x, y):
                    self.errors["observation"].append(
                        f"{self.name}/{slot}/{r}: illegal replay position"
                    )
        return {
            "coverage": self.coverage,
            "usable_coverage": self.usable_coverage,
            "usable_sets": self.usable_sets,
            "open": self.open_counts,
            "per_robot": self.per_robot,
            "per_robot_history": self.per_robot_history,
            "first_discoverer": self.first_discoverer,
            "hgrid_history": self.hgrid_history,
            "moves": self.moves,
            "safety": self.safety,
            "pings": self.pings,
            "success": self.success,
            "measurements": self.measurements,
            "arrivals": self.arrivals,
            "plan_snapshots": self.plan_snapshots,
            "positions_by_slot": self.positions_by_slot,
            "local_history": self.local_history,
            "errors": self.errors,
            "standby_slots": standby_slots,
            "active_slots": active_slots,
            "standby_by_slot": self.standby_by_slot,
            "termination_slot": self.termination_slot,
            "t95_slot": self.t95_slot,
        }


def _replay(
    records: Mapping[tuple[str, str, int, str], Any],
) -> tuple[list[AtomVerdict], dict[str, Any]]:
    errors = {k.split(".", 1)[1]: [] for k in ATOM_SHARES["replay_validity"]}
    out = {}
    for name in SCENARIOS:
        rows = {(kind, t, key): v for (sc, kind, t, key), v in records.items() if sc == name}
        result = Replay(name, rows).run()
        out[name] = result
        for k, vals in result.get("errors", {}).items():
            if k in errors:
                errors[k].extend(vals)
        for r in ROBOTS:
            for t in PLAN:
                p = records.get((name, "plan", t, r))
                snap = result["plan_snapshots"].get(t, {}).get(r, {})
                fronts = {fid: comp for fid, comp in snap.get("frontiers", [])}
                if not isinstance(p, Mapping):
                    errors["estimates"].append(f"{name}/{r}/{t}: missing planning record")
                    continue
                standby = p.get("standby")
                fid = p.get("selected_frontier_id")
                vp = p.get("selected_viewpoint")
                if not _is_live(name, r, t):
                    if standby is not True or fid is not None or vp is not None:
                        errors["motion"].append(
                            f"{name}/{r}/{t}: failed robot requires null standby plan"
                        )
                    continue
                if standby:
                    if fid is not None or vp is not None:
                        errors["motion"].append(f"{name}/{r}/{t}: standby target not null")
                else:
                    if not isinstance(fid, str) or fid not in fronts or not isinstance(vp, Mapping):
                        errors["motion"].append(f"{name}/{r}/{t}: frontier/viewpoint unavailable")
                    else:
                        comp = fronts[fid]
                        candidates = _viewpoint_candidates(comp)
                        point = (vp.get("x"), vp.get("y"))
                        if point not in candidates or not _reachable(
                            snap.get("local", {}), snap.get("position"), point
                        ):
                            errors["motion"].append(f"{name}/{r}/{t}: illegal viewpoint")
        for (sc, kind, t, key), e in records.items():
            if sc != name or kind != "estimate":
                continue
            if (
                not _is_live(name, e.get("observer"), t) or not _is_live(name, e.get("neighbor"), t)
            ) and e.get("valid") is True:
                errors["estimates"].append(
                    f"{name}/{t}: estimate involving failed robot must be invalid"
                )
                continue
            if e.get("valid"):
                mids = e.get("source_measurement_ids", [])
                if not isinstance(mids, list):
                    errors["estimates"].append(f"{name}/{t}: source list is not an array")
                    continue
                source_slots = []
                for mid in mids:
                    m = result["measurements"].get(mid)
                    if (
                        m is None
                        or not m["success"]
                        or m["arrival"] is None
                        or m["arrival"] > t
                        or mid not in result.get("arrivals", {})
                    ):
                        errors["signal"].append(
                            f"{name}/{e.get('observer')}->{e.get('neighbor')}/{t}: "
                            f"unavailable source {mid}"
                        )
                    if m is not None:
                        source_slots.append(m.get("slot"))
                        expected = f"M{m['slot']}_{e.get('observer')}_{e.get('neighbor')}"
                        if (
                            mid != expected
                            or m.get("receiver") != e.get("observer")
                            or m.get("other") != e.get("neighbor")
                        ):
                            errors["signal"].append(f"{name}/{t}: source direction mismatch {mid}")
                if source_slots:
                    expected_age = t - max(source_slots)
                    if e.get("age") != expected_age:
                        errors["estimates"].append(
                            f"{name}/{e.get('observer')}->{e.get('neighbor')}/{t}: "
                            f"source age mismatch expected={expected_age}, actual={e.get('age')}, "
                            f"sources={mids}"
                        )
                if len(set(mids)) != len(mids):
                    errors["estimates"].append(f"{name}/{t}: duplicate source IDs")
                if e.get("age") is not None and (
                    not _integer(e.get("age"), 0, 12) or e.get("age") > 12
                ):
                    errors["estimates"].append(f"{name}/{t}: estimate age invalid or too old")
            elif e.get("source_measurement_ids") not in ([], None):
                errors["estimates"].append(f"{name}/{t}: invalid estimate has sources")
    atoms = [
        AtomVerdict(
            f"replay_validity.{k}",
            not v,
            "ok" if not v else " | ".join(v[:8]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{k}"],
        )
        for k, v in errors.items()
    ]
    return atoms, out


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs mismatch")
    if any(not _finite(share) or float(share) <= 0.0 for share in expected.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("atom shares")
    if any(not math.isclose(a.share, expected[a.id], abs_tol=1e-12, rel_tol=0) for a in atoms):
        raise ValueError("atom share mismatch")
    return _clip(sum(a.share for a in atoms if a.passed))


def _diag(atoms: Sequence[AtomVerdict]) -> str:
    bad = [a for a in atoms if not a.passed]
    return (
        "failed_atoms=[]"
        if not bad
        else "failed_atoms=["
        + ",".join(a.id for a in bad)
        + "]; "
        + " | ".join(f"{a.id}: {a.detail}" for a in bad)
    )


def _nis_value(errx: float, erry: float, cxx: float, cxy: float, cyy: float) -> float:
    """Evaluate a 2-D quadratic form without overflowing on finite JSON numbers."""
    if not _positive_definite(cxx, cxy, cyy):
        return math.inf
    covariance_scale = max(abs(cxx), abs(cxy), abs(cyy))
    a, b, c = cxx / covariance_scale, cxy / covariance_scale, cyy / covariance_scale
    determinant = a * c - b * b
    error_scale = max(abs(errx), abs(erry))
    if error_scale == 0.0:
        return 0.0
    x, y = errx / error_scale, erry / error_scale
    normalized = max(0.0, (c * x * x - 2.0 * b * x * y + a * y * y) / determinant)
    value = normalized * error_scale * (error_scale / covariance_scale)
    return value if math.isfinite(value) else math.inf


def _metrics(
    replayed: Mapping[str, Any], records: Mapping[tuple[str, str, int, str], Any] | None = None
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
    for n, r in replayed.items():
        usable = r.get("usable_coverage", r["coverage"])
        final = usable[-1] / max(1, r["open"][-1])
        auc = sum(c / max(1, o) for c, o in zip(usable, r["open"])) / T
        p1 = _clip(0.7 * final + 0.3 * auc)

        # Overlap is evaluated on the final per-robot discovery sets.  The
        # public formula is defined only when there is coverage; no coverage
        # therefore contributes zero suppression rather than an artificial one.
        sets = [set(r["per_robot"].get(robot, set())) for robot in ROBOTS]
        union_set = set().union(*sets)
        if union_set:
            overlap_suppression = _clip(
                1.0 - ((sum(len(s) for s in sets) / len(union_set)) - 1.0) / 2.0
            )
        else:
            overlap_suppression = 0.0

        t_stop = r.get("termination_slot") if r.get("termination_slot") is not None else T
        t95 = r.get("t95_slot") if r.get("t95_slot") is not None else T
        term_quality = final * _clip(1.0 - abs(t_stop - t95) / T)

        failure_recovery = 1.0
        if n == "robot_failure":
            # Ownership is the pre-failure exclusive set, not simply the
            # first-discoverer set: a survivor may already have visited a cell
            # before R2 fails.  Recovery requires a survivor observation at or
            # after the fixed failure boundary.
            history = r.get("per_robot_history", [])
            pre = history[84] if len(history) > 84 and isinstance(history[84], Mapping) else {}
            r2_pre = set(pre.get("R2", set()))
            survivors_pre = set(pre.get("R0", set())) | set(pre.get("R1", set()))
            owned = r2_pre - survivors_pre
            recovered = set()
            for slot, hist in enumerate(r.get("per_robot_history", [])):
                if slot < 85 or not isinstance(hist, Mapping):
                    continue
                recovered.update(set(hist.get("R0", set())) | set(hist.get("R1", set())))
            failure_recovery = 1.0 if not owned else len(owned & recovered) / len(owned)
        p2 = _clip(0.5 * overlap_suppression + 0.3 * term_quality + 0.2 * failure_recovery)

        # Discovery yield uses unique usable cells; standby efficiency uses
        # active robot-slot occupancy over the full horizon (not planning
        # boundaries only), as specified by the result contract.
        unique_usable = len(r.get("usable_sets", [set()])[-1])
        active_slots = sum(
            1 for flags in r.get("standby_by_slot", []) for value in flags.values() if not value
        )
        p3 = _clip(
            0.45 * _clip(unique_usable / max(1, r["moves"]))
            + 0.3 * _clip(1 - r["safety"] / (3 * T))
            + 0.25 * final * _clip(1 - active_slots / (3 * T))
        )

        used_source_ids = set()
        if records is not None:
            for (sc, kind, _t, _key), e in records.items():
                if (
                    sc == n
                    and kind == "estimate"
                    and isinstance(e, Mapping)
                    and e.get("valid") is True
                ):
                    used_source_ids.update(
                        x for x in e.get("source_measurement_ids", []) if isinstance(x, str)
                    )
        executed_ids = {
            mid for mid, m in r.get("measurements", {}).items() if isinstance(m, Mapping)
        }
        useful_ping_slots = {
            mid.split("_", 1)[0]
            for mid in (executed_ids & used_source_ids)
            if re.fullmatch(r"M\d+_R[0-2]_R[0-2]", mid)
        }
        decision_used_pings = len(useful_ping_slots)
        # Each successful measurement arrives exactly one slot later under the
        # public generator; count all such arrivals as on-time and all failed
        # or missing results as late.
        on_time = r.get("success", 0)
        a1 = decision_used_pings / max(1, r.get("pings", 0))
        ontime_fraction = on_time / max(1, r.get("pings", 0))
        auxiliary_1 = _clip(0.65 * a1 + 0.35 * ontime_fraction)

        quality = []
        freshness = []
        nis = []
        if records is not None:
            for (sc, kind, t, _key), e in records.items():
                if (
                    sc != n
                    or kind != "estimate"
                    or not isinstance(e, Mapping)
                    or e.get("valid") is not True
                ):
                    continue
                obs, nei = e.get("observer"), e.get("neighbor")
                positions = r["positions_by_slot"].get(t, {})
                if obs not in positions or nei not in positions:
                    continue
                if not all(
                    _finite(e.get(k))
                    for k in ("estimate_x", "estimate_y", "cov_xx", "cov_xy", "cov_yy")
                ):
                    continue
                dx = positions[nei][0] - positions[obs][0]
                dy = positions[nei][1] - positions[obs][1]
                ex, ey = float(e["estimate_x"]), float(e["estimate_y"])
                errx, erry = ex - dx, ey - dy
                err = math.hypot(errx, erry)
                quality.append(_clip(1 - err / 20))
                if _integer(e.get("age"), 0, 12):
                    freshness.append(_clip(1 - e["age"] / 12))
                cxx, cxy, cyy = float(e["cov_xx"]), float(e["cov_xy"]), float(e["cov_yy"])
                if _positive_definite(cxx, cxy, cyy):
                    nis.append(_nis_value(errx, erry, cxx, cxy, cyy))
        if quality:
            rel_quality = sum(quality) / len(quality)
            fresh = sum(freshness) / len(freshness) if freshness else 0.0
            coverage95 = sum(v <= 5.991 for v in nis) / len(nis) if nis else 0.0
            mean_nis = sum(nis) / len(nis) if nis else 0.0
            calibration = (
                0.5 * _clip(1 - abs(coverage95 - 0.95) / 0.25)
                + 0.5 * _clip(1 - abs(mean_nis - 2) / 4)
                if nis
                else 0.0
            )
            auxiliary_2 = _clip(0.45 * rel_quality + 0.3 * fresh + 0.25 * calibration)
        else:
            auxiliary_2 = 0.0
        for k, v in zip(vals, (p1, p2, p3, auxiliary_1, auxiliary_2)):
            vals[k].append(v)
    return {k: _clip(sum(v) / len(v)) for k, v in vals.items()}


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


def _gated_metrics(
    scores: Mapping[str, float],
    replayed: Mapping[str, Any],
    records: Mapping[tuple[str, str, int, str], Any],
) -> dict[str, float] | None:
    required = (
        "official_run",
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    )
    if not all(scores.get(key, 0.0) >= 1.0 - EPS for key in required):
        return None
    return _metrics(replayed, records)


RESULT_DEPENDENT_CHECKS = (
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


def _zero_result_dependent(state: EvalState, reason: str) -> None:
    for key in RESULT_DEPENDENT_CHECKS:
        state.scores[key] = 0.0
        state.diagnostics[key] = f"{reason}; {key}=0"
    state.metrics = None
    state.validity_gate = False


async def _build_state(ws: Workspace) -> EvalState:
    root = str(ws.root)
    if root in _CACHE:
        return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE:
            return _CACHE[root]
        st = EvalState()
        path = Path(ws.root, "results.json")
        deletion_ok = False
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
            remove_exit = getattr(removed, "exit_code", None)
            deletion_ok = remove_exit == 0 and _absent(path)
            if not deletion_ok:
                st.official_message = (
                    f"old results deletion failed: exit={remove_exit}, "
                    f"absent_after_delete={_absent(path)}"
                )
                st.diagnostics["official"] = st.official_message
                st.scores["official_run"] = 0.0
                _zero_result_dependent(st, st.official_message)
                _CACHE[root] = st
                return st
            st.run_started = True
            start = time.monotonic()
            try:
                res = await ws.run("python run.py", check=False, timeout=600)
            finally:
                st.run_finished = True
                st.elapsed_s = time.monotonic() - start
            st.exit_code = getattr(res, "exit_code", None)
            st.result_file_is_new_regular = deletion_ok and _regular_file(path)
            st.official_ok = (
                deletion_ok
                and st.exit_code == 0
                and st.elapsed_s <= 600 + EPS
                and st.result_file_is_new_regular
            )
            if st.official_ok:
                try:
                    st.data = _strict_json(ws.read_text("results.json") or "")
                    st.parse_ok = True
                except Exception as exc:
                    st.parse_ok = False
                    st.data = None
                    st.diagnostics["parse"] = f"strict parse failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            st.timed_out = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
            st.official_ok = False
            st.parse_ok = False
            st.data = None
            st.official_message = f"{type(exc).__name__}: {exc}"
        elapsed = "none" if st.elapsed_s is None else f"{st.elapsed_s:.6f}"
        st.diagnostics["official"] = st.official_message or (
            f"deletion_ok={deletion_ok}, exit_code={st.exit_code}, timed_out={st.timed_out}, "
            f"elapsed_s={elapsed}, new_regular_non_symlink={st.result_file_is_new_regular}"
        )
        st.scores["official_run"] = 1.0 if st.official_ok else 0.0
        if not st.official_ok:
            _zero_result_dependent(st, f"official run failed: {st.diagnostics['official']}")
            _CACHE[root] = st
            return st
        if not st.parse_ok:
            _zero_result_dependent(st, st.diagnostics.get("parse", "strict JSON parse failed"))
            _CACHE[root] = st
            return st
        try:
            a = _result_atoms(st.data, st.parse_ok)
            st.scores["result_json"] = _score_atoms(a, ATOM_SHARES["result_json"])
            st.diagnostics["result_json"] = _diag(a)
            st.records = _records(st.data)
            a = _evidence_atoms(st.data)
            st.scores["result_evidence"] = _score_atoms(a, ATOM_SHARES["result_evidence"])
            st.diagnostics["result_evidence"] = _diag(a)
            st.scores["canonical_coverage"], defects = _coverage(st.data)
            st.diagnostics["canonical_coverage"] = f"defects={defects}"
            a = _legality(st.data)
            st.scores["decision_legality"] = _score_atoms(a, ATOM_SHARES["decision_legality"])
            st.diagnostics["decision_legality"] = _diag(a)
            a, st.replayed = _replay(st.records)
            st.scores["replay_validity"] = _score_atoms(a, ATOM_SHARES["replay_validity"])
            st.diagnostics["replay_validity"] = _diag(a)
            st.metrics = _gated_metrics(st.scores, st.replayed, st.records)
            st.validity_gate = st.metrics is not None
            if st.metrics is not None:
                for key, value in st.metrics.items():
                    st.scores[key] = value
                    st.diagnostics[key] = f"{key}={value:.9f}"
            else:
                for key in RESULT_DEPENDENT_CHECKS[5:]:
                    st.scores[key] = 0.0
                    st.diagnostics[key] = f"{key}: validity gate not satisfied; {key}=0"
        except Exception as exc:
            reason = f"deterministic result evaluation failed: {type(exc).__name__}: {exc}"
            _zero_result_dependent(st, reason)
        _CACHE[root] = st
        return st


def _report_score(text: str) -> tuple[float, list[str]]:
    missing = []
    section_names = "|".join(re.escape(name) for name in REPORT_SECTIONS)
    next_section = re.compile(rf"^#{{1,6}}[ \t]+(?:{section_names})[ \t]*$", re.M)
    for s in REPORT_SECTIONS:
        m = re.search(rf"^#{{1,6}}[ \t]+{re.escape(s)}[ \t]*$", text, re.M)
        if not m:
            missing.append(s)
            continue
        nxt = next_section.search(text[m.end() :])
        body = text[m.end() : m.end() + (nxt.start() if nxt else len(text))]
        if not body.strip():
            missing.append(s)
    return max(0, (2 - 0.5 * len(missing)) / 2), missing


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    try:
        r = await ws.run("python -m py_compile run.py", check=False, timeout=30)
        ok = _regular_file(Path(ws.root, "run.py")) and getattr(r, "exit_code", 1) == 0
    except Exception as exc:
        return 0.0, f"compile failed: {exc}"
    return (1.0, "run.py exists and compiles") if ok else (0.0, "run.py missing or compile failed")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    s = await _build_state(ws)
    return (
        (1.0, "official run passed")
        if s.official_ok
        else (0.0, s.diagnostics.get("official", "official run failed"))
    )


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "report.md")
    if not _regular_file(path):
        return 0.0, "report.md is missing, non-regular, or a symbolic link"
    try:
        text = ws.read_text("report.md")
        if not isinstance(text, str):
            return 0.0, "report.md read did not return text"
        score, missing = _report_score(text)
    except Exception as exc:
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    return score, f"missing_or_empty={missing}"


async def _gate(ws: Workspace, key: str) -> tuple[float, str]:
    s = await _build_state(ws)
    return s.scores.get(key, 0.0), s.diagnostics.get(key, "")


for _id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):
    globals()[_id] = check(id=_id)(lambda ws, key=_id: _gate(ws, key))


def _metric_check(key: str):
    @check(id=key)
    async def fn(ws: Workspace) -> tuple[float, str]:
        s = await _build_state(ws)
        if not s.validity_gate:
            return 0.0, s.diagnostics.get(key, f"{key}: validity gate not satisfied")
        return s.metrics.get(key, 0.0), f"{key}={s.metrics.get(key, 0.0):.9f}"

    return fn


for _id in (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):
    globals()[_id] = _metric_check(_id)
