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

HORIZON = 60
DT = 0.2
EPS = 1e-6
DOMAIN = (0.0, 20.0, 0.0, 12.0)
ROBOT_RADIUS = 0.3
CLEARANCE = 0.45
LOCAL_RANGE = {
    "open_collection": 1.5,
    "occluded_crossing": 1.2,
    "far_sensor": 1.4,
    "power_burst": 1.1,
    "fairness_route": 1.3,
}
SCENARIO_NAMES = (
    "open_collection",
    "occluded_crossing",
    "far_sensor",
    "power_burst",
    "fairness_route",
)
POWER_LEVELS = (0.0, 2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3)
P_TOTAL = 2.0e-3
ENERGY_BUDGET = 0.012
RATE_THRESHOLD = 700000.0
SENSOR_BW = 100000.0
RHO0 = 1.0e-3
NOISE = 1.0e-9
GOAL_RADIUS = 0.35
ROUTE_ALERT = 1.2
SCENARIOS: dict[str, dict[str, Any]] = {
    "open_collection": {
        "start": (2.0, 2.0, 0.0),
        "goal": (7.2, 2.0),
        "deadline": 60,
        "sensors": ((2.8, 3.0), (4.1, 1.1), (5.4, 3.0), (6.7, 1.1)),
        "route": ((2.0, 2.0), (4.6, 2.0), (7.2, 2.0)),
        "obstacles": (),
    },
    "occluded_crossing": {
        "start": (2.0, 3.0, 0.0),
        "goal": (7.0, 3.0),
        "deadline": 60,
        "sensors": ((2.8, 4.0), (4.0, 2.0), (5.2, 4.0), (6.4, 2.0)),
        "route": ((2.0, 3.0), (4.5, 3.0), (7.0, 3.0)),
        "obstacles": ((4.05, 4.55, 2.55, 3.45),),
    },
    "far_sensor": {
        "start": (2.0, 6.0, 0.0),
        "goal": (7.0, 6.0),
        "deadline": 58,
        "sensors": ((2.5, 7.0), (3.8, 5.0), (5.2, 8.0), (6.6, 5.0)),
        "route": ((2.0, 6.0), (4.5, 6.0), (7.0, 6.0)),
        "obstacles": ((4.5, 5.1, 6.2, 7.4),),
    },
    "power_burst": {
        "start": (3.0, 9.0, 0.0),
        "goal": (8.0, 9.0),
        "deadline": 56,
        "sensors": ((3.5, 10.0), (4.8, 8.0), (6.2, 10.0), (7.5, 8.0)),
        "route": ((3.0, 9.0), (5.5, 9.0), (8.0, 9.0)),
        "obstacles": ((5.2, 5.8, 8.55, 9.45),),
    },
    "fairness_route": {
        "start": (10.0, 3.0, 0.0),
        "goal": (15.0, 3.0),
        "deadline": 60,
        "sensors": ((10.6, 4.0), (11.8, 2.0), (13.6, 5.0), (14.4, 2.0)),
        "route": ((10.0, 3.0), (12.5, 3.0), (15.0, 3.0)),
        "obstacles": ((12.6, 13.2, 3.3, 4.5),),
    },
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.scenario_keys": 0.25,
    },
    "result_evidence": {
        "result_evidence.decisions": 1 / 6,
        "result_evidence.states": 1 / 6,
        "result_evidence.links": 1 / 6,
        "result_evidence.reports": 1 / 6,
        "result_evidence.obstacles": 1 / 6,
        "result_evidence.collection": 1 / 6,
    },
    "decision_legality": {
        "decision_legality.motion": 1 / 7,
        "decision_legality.powers": 1 / 7,
        "decision_legality.total_power": 1 / 7,
        "decision_legality.continuity": 1 / 7,
        "decision_legality.transmit_continuity": 1 / 7,
        "decision_legality.energy": 1 / 7,
        "decision_legality.power_domain": 1 / 7,
    },
    "replay_validity": {
        "replay_validity.motion": 1 / 8,
        "replay_validity.links": 1 / 8,
        "replay_validity.reports": 1 / 8,
        "replay_validity.obstacles": 1 / 8,
        "replay_validity.states": 1 / 8,
        "replay_validity.collection": 1 / 8,
        "replay_validity.goal": 1 / 8,
        "replay_validity.report_evidence": 1 / 8,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, passed, detail, share


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
        self.records: dict[Hashable, Any] = {}
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    return AtomVerdict(
        f"{group}.{name}", bool(passed), detail, ATOM_SHARES[group][f"{group}.{name}"]
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _close(a: Any, b: Any, tol: float = EPS) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tol


def _deep_close(actual: Any, expected: Any) -> bool:
    if _finite(actual) and _finite(expected):
        return _close(actual, expected)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _deep_close(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _deep_close(actual[key], expected[key]) for key in expected
        )
    return actual == expected


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def _point_rect_distance(point: Sequence[float], rect: Sequence[float]) -> float:
    x, y = float(point[0]), float(point[1])
    xmin, xmax, ymin, ymax = map(float, rect)
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def _segment_rect_intersects(
    a: Sequence[float], b: Sequence[float], rect: Sequence[float], pad: float = 0.0
) -> bool:
    xmin, xmax, ymin, ymax = (
        float(rect[0]) - pad,
        float(rect[1]) + pad,
        float(rect[2]) - pad,
        float(rect[3]) + pad,
    )
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    p = (-dx, dx, -dy, dy)
    q = (float(a[0]) - xmin, xmax - float(a[0]), float(a[1]) - ymin, ymax - float(a[1]))
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= EPS:
            if qi < -EPS:
                return False
        elif pi < 0:
            u0 = max(u0, qi / pi)
        else:
            u1 = min(u1, qi / pi)
        if u0 - u1 > EPS:
            return False
    return True


def _distance_to_route(point: Sequence[float], route: Sequence[Sequence[float]]) -> float:
    best = float("inf")
    for a, b in zip(route, route[1:]):
        vx, vy = b[0] - a[0], b[1] - a[1]
        wx, wy = point[0] - a[0], point[1] - a[1]
        den = vx * vx + vy * vy
        u = min(1.0, max(0.0, (wx * vx + wy * vy) / den)) if den else 0.0
        best = min(best, math.hypot(point[0] - (a[0] + u * vx), point[1] - (a[1] + u * vy)))
    return best


def _region(name: str, sensor: int) -> list[float]:
    sx, sy = SCENARIOS[name]["sensors"][sensor]
    return [max(0.0, sx - 1.0), max(0.0, sy - 1.0), min(20.0, sx + 1.0), min(12.0, sy + 1.0)]


def _intersections(name: str, sensor: int) -> list[list[float]]:
    rx0, ry0, rx1, ry1 = _region(name, sensor)
    out: list[list[float]] = []
    for xmin, xmax, ymin, ymax in SCENARIOS[name]["obstacles"]:
        x0, x1, y0, y1 = max(rx0, xmin), min(rx1, xmax), max(ry0, ymin), min(ry1, ymax)
        if x0 <= x1 + EPS and y0 <= y1 + EPS:
            out.append([x0, y0, x1, y1])
    return sorted(out)


def _rate(distance: float, power: float) -> float:
    d = max(0.5, float(distance))
    return SENSOR_BW * math.log2(1.0 + RHO0 * float(power) / (NOISE * d * d))


def _replay_scenario(name: str, decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    p = [spec["start"][0], spec["start"][1]]
    theta = spec["start"][2]
    energy = ENERGY_BUDGET
    done = collision = out = goal = False
    goal_arrival: int | None = None
    qual = [0, 0, 0, 0]
    versions = [0, 0, 0, 0]
    cache: list[dict[str, Any] | None] = [None] * 4
    pending: dict[int, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    obstacles: list[dict[str, Any]] = []
    collected = [0.0] * 4
    report_counts = [0] * 4
    first = [None] * 4
    last = [None] * 4

    def process_boundary(state_t: int) -> tuple[list[list[float]], list[list[Any]], list[int]]:
        for report in pending.pop(state_t, []):
            sensor = report["sensor"]
            old = cache[sensor]
            if old is None or report["report_version"] > old["report_version"]:
                report["fusion_status"] = "fused"
                cache[sensor] = report
                report_counts[sensor] += 1
                if first[sensor] is None:
                    first[sensor] = state_t
                last[sensor] = state_t
            else:
                report["fusion_status"] = "duplicate"
        for sensor, report in enumerate(cache):
            if report is not None and state_t > report["valid_until_state"]:
                cache[sensor] = None
        local = [
            list(r)
            for r in spec["obstacles"]
            if _point_rect_distance(p, r) <= LOCAL_RANGE[name] + EPS
        ]
        external: list[list[Any]] = []
        active_ids: list[int] = []
        for sensor, report in enumerate(cache):
            if report is not None:
                active_ids.append(report["report_id"])
                for rect in report["obstacle_intersections"]:
                    external.append([sensor, report["report_version"], list(rect)])
        external.sort(key=lambda x: (x[0], x[1], x[2]))
        # Internal obstacles use (xmin, xmax, ymin, ymax), while the public
        # contract uses [xmin, ymin, xmax, ymax]. Sort after serialization so
        # replay ordering follows the documented public representation.
        local_payload = sorted(
            ([r[0], r[2], r[1], r[3]] for r in local),
            key=lambda r: r,
        )
        all_rects = [[rect[0], rect[2], rect[1], rect[3]] for rect in local_payload] + [
            [item[2][0], item[2][2], item[2][1], item[2][3]] for item in external
        ]
        min_dist = min((_point_rect_distance(p, r) for r in all_rects), default=20.0)
        obstacles.append(
            {
                "state_t": state_t,
                "local_visible": local_payload,
                "external_fused": external,
                "active_report_ids": sorted(active_ids),
                "min_distance": min_dist,
                "collision": collision,
                "out_of_bounds": out,
                "route_alert": _distance_to_route(p, spec["route"]) > ROUTE_ALERT + EPS,
            }
        )
        return local, external, active_ids

    process_boundary(0)
    states.append(
        {
            "state_t": 0,
            "x": p[0],
            "y": p[1],
            "theta": theta,
            "v": 0.0,
            "omega": 0.0,
            "remaining_tx_energy": energy,
            "route_deviation": _distance_to_route(p, spec["route"]),
            "done": done,
            "collision": collision,
            "out_of_bounds": out,
            "goal_reached": goal,
            "goal_arrival_state": goal_arrival,
        }
    )
    for t in range(HORIZON):
        decision = decisions[t]
        powers = [float(v) for v in decision["powers"]]
        for sensor, power in enumerate(powers):
            sx, sy = spec["sensors"][sensor]
            distance = math.hypot(p[0] - sx, p[1] - sy)
            rate = _rate(distance, power)
            threshold = rate >= RATE_THRESHOLD - EPS
            bits = rate * DT
            collected[sensor] += bits
            qual[sensor] = qual[sensor] + 1 if threshold else 0
            links.append(
                {
                    "sensor": sensor,
                    "t": t,
                    "distance": distance,
                    "power": power,
                    "rate": rate,
                    "bits": bits,
                    "threshold_met": threshold,
                }
            )
        submitted_v, submitted_w = float(decision["v"]), float(decision["omega"])
        actual_v, actual_w = (0.0, 0.0) if done else (submitted_v, submitted_w)
        old_p = tuple(p)
        new_p = [p[0] + DT * actual_v * math.cos(theta), p[1] + DT * actual_v * math.sin(theta)]
        collision_now = any(
            _segment_rect_intersects(old_p, new_p, r, CLEARANCE)
            or _point_rect_distance(new_p, r) <= CLEARANCE + EPS
            for r in spec["obstacles"]
        )
        out_now = not (
            DOMAIN[0] - EPS <= new_p[0] <= DOMAIN[1] + EPS
            and DOMAIN[2] - EPS <= new_p[1] <= DOMAIN[3] + EPS
        )
        if collision_now:
            collision = True
            done = True
        if out_now:
            out = True
            done = True
        p = new_p
        theta = ((theta + DT * actual_w + math.pi) % (2 * math.pi)) - math.pi
        energy = max(0.0, energy - DT * sum(powers))
        if (
            not goal
            and math.hypot(p[0] - spec["goal"][0], p[1] - spec["goal"][1]) <= GOAL_RADIUS + EPS
        ):
            goal = True
            goal_arrival = t + 1
        if any(qv >= 2 for qv in qual):
            for sensor, qv in enumerate(qual):
                if qv == 2:
                    versions[sensor] += 1
                    arrival = t + 1
                    valid = min(HORIZON, arrival + 3)
                    report = {
                        "report_id": len(reports),
                        "sensor": sensor,
                        "report_version": versions[sensor],
                        "generated_slot": t,
                        "arrival_state": arrival,
                        "trigger_slots": [t - 1, t],
                        "region": _region(name, sensor),
                        "obstacle_intersections": _intersections(name, sensor),
                        "valid_until_state": valid,
                        "fusion_status": "late",
                        "used_states": list(range(arrival, valid + 1)),
                    }
                    reports.append(report)
                    pending.setdefault(arrival, []).append(report)
        states.append(
            {
                "state_t": t + 1,
                "x": p[0],
                "y": p[1],
                "theta": theta,
                "v": actual_v,
                "omega": actual_w,
                "remaining_tx_energy": energy,
                "route_deviation": _distance_to_route(p, spec["route"]),
                "done": done,
                "collision": collision,
                "out_of_bounds": out,
                "goal_reached": goal,
                "goal_arrival_state": goal_arrival,
            }
        )
        if t < HORIZON - 1:
            process_boundary(t + 1)
    process_boundary(HORIZON)
    # A report that arrived at the terminal boundary is still valid but cannot affect later motion.
    collection = []
    for sensor in range(4):
        fused = [
            report
            for report in reports
            if report["sensor"] == sensor and report["fusion_status"] == "fused"
        ]
        first_valid = min((int(report["arrival_state"]) for report in fused), default=None)
        last_valid = max((int(report["valid_until_state"]) for report in fused), default=None)
        collection.append(
            {
                "sensor": sensor,
                "total_bits": collected[sensor],
                "valid_report_count": report_counts[sensor],
                "first_valid_state": first_valid,
                "last_valid_state": last_valid,
                "fairness_component": collected[sensor],
                "deadline_met": bool(first_valid is not None and first_valid <= spec["deadline"]),
            }
        )
    return {
        "states": states,
        "links": links,
        "reports": reports,
        "obstacles": obstacles,
        "collection": collection,
    }


def _scenario_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {
        "scenario",
        "decisions",
        "states",
        "links",
        "reports",
        "obstacles",
        "collection",
    }


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return out
    for row in data["scenarios"]:
        if not isinstance(row, dict) or not isinstance(row.get("scenario"), str):
            continue
        name = row["scenario"]
        if name in out:
            continue

        def keyed(field: str, key: str) -> dict[Any, Any]:
            rows = row.get(field) if isinstance(row.get(field), list) else []
            result: dict[Any, Any] = {}
            for item in rows:
                if isinstance(item, dict) and _integer(item.get(key)):
                    result.setdefault(item[key], item)
            return result

        out[name] = {
            "record": row,
            "decisions_rows": row.get("decisions", []),
            "states_rows": row.get("states", []),
            "links_rows": row.get("links", []),
            "reports_rows": row.get("reports", []),
            "obstacles_rows": row.get("obstacles", []),
            "collection_rows": row.get("collection", []),
            "decisions": keyed("decisions", "t"),
            "states": keyed("states", "state_t"),
            "obstacles": keyed("obstacles", "state_t"),
            "collection": keyed("collection", "sensor"),
        }
    return out


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, dict)
    scenarios = root and isinstance(data.get("scenarios"), list)
    names = scenarios and all(
        isinstance(r, dict) and isinstance(r.get("scenario"), str) for r in data["scenarios"]
    )
    keys = scenarios and all(_scenario_shape(r) for r in data["scenarios"])
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom(
            "result_json",
            "root",
            root and set(data) == {"scenarios"},
            "root must contain only scenarios",
        ),
        _atom("result_json", "scenarios", scenarios, "scenarios must be a list"),
        _atom(
            "result_json",
            "scenario_keys",
            bool(names and keys),
            "scenario records have declared exact keys",
        ),
    ]


def _valid_decision(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row) == {"t", "v", "omega", "powers", "total_power"}
        and _integer(row.get("t"))
        and 0 <= row["t"] < HORIZON
        and _finite(row.get("v"))
        and 0.0 - EPS <= float(row["v"]) <= 0.5 + EPS
        and _finite(row.get("omega"))
        and -1.0 - EPS <= float(row["omega"]) <= 1.0 + EPS
        and isinstance(row.get("powers"), list)
        and len(row["powers"]) == 4
        and all(
            _finite(v) and any(abs(float(v) - q) <= EPS for q in POWER_LEVELS)
            for v in row["powers"]
        )
        and _finite(row.get("total_power"))
        and 0.0 - EPS <= float(row["total_power"]) <= P_TOTAL + EPS
    )


def _decision_evidence_valid(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row) == {"t", "v", "omega", "powers", "total_power"}
        and _integer(row.get("t"))
        and _finite(row.get("v"))
        and _finite(row.get("omega"))
        and isinstance(row.get("powers"), list)
        and len(row["powers"]) == 4
        and all(_finite(value) for value in row["powers"])
        and _finite(row.get("total_power"))
    )


def _valid_state(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row)
        == {
            "state_t",
            "x",
            "y",
            "theta",
            "v",
            "omega",
            "remaining_tx_energy",
            "route_deviation",
            "done",
            "collision",
            "out_of_bounds",
            "goal_reached",
            "goal_arrival_state",
        }
        and _integer(row.get("state_t"))
        and 0 <= row["state_t"] <= HORIZON
        and all(
            _finite(row.get(k))
            for k in ("x", "y", "theta", "v", "omega", "remaining_tx_energy", "route_deviation")
        )
        and float(row["remaining_tx_energy"]) >= -EPS
        and float(row["route_deviation"]) >= -EPS
        and all(
            isinstance(row.get(k), bool)
            for k in ("done", "collision", "out_of_bounds", "goal_reached")
        )
        and (
            row.get("goal_arrival_state") is None
            or (_integer(row["goal_arrival_state"]) and 1 <= row["goal_arrival_state"] <= HORIZON)
        )
    )


def _valid_link(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row) == {"sensor", "t", "distance", "power", "rate", "bits", "threshold_met"}
        and _integer(row.get("sensor"))
        and 0 <= row["sensor"] < 4
        and _integer(row.get("t"))
        and 0 <= row["t"] < HORIZON
        and all(
            _finite(row.get(k)) and float(row[k]) >= -EPS
            for k in ("distance", "power", "rate", "bits")
        )
        and isinstance(row.get("threshold_met"), bool)
    )


def _rectangles_valid(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(rect, list)
        and len(rect) == 4
        and all(_finite(v) for v in rect)
        and rect[0] <= rect[2] + EPS
        and rect[1] <= rect[3] + EPS
        for rect in value
    )


def _valid_report(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row)
        == {
            "report_id",
            "sensor",
            "report_version",
            "generated_slot",
            "arrival_state",
            "trigger_slots",
            "region",
            "obstacle_intersections",
            "valid_until_state",
            "fusion_status",
            "used_states",
        }
        and _integer(row.get("report_id"))
        and row["report_id"] >= 0
        and _integer(row.get("sensor"))
        and 0 <= row["sensor"] < 4
        and _integer(row.get("report_version"))
        and row["report_version"] > 0
        and _integer(row.get("generated_slot"))
        and 0 <= row["generated_slot"] < HORIZON
        and _integer(row.get("arrival_state"))
        and 1 <= row["arrival_state"] <= HORIZON
        and isinstance(row.get("trigger_slots"), list)
        and len(row["trigger_slots"]) == 2
        and all(_integer(v) and 0 <= v < HORIZON for v in row["trigger_slots"])
        and row["trigger_slots"][1] == row["trigger_slots"][0] + 1
        and isinstance(row.get("region"), list)
        and len(row["region"]) == 4
        and all(_finite(v) for v in row["region"])
        and row["region"][0] <= row["region"][2] + EPS
        and row["region"][1] <= row["region"][3] + EPS
        and _rectangles_valid(row.get("obstacle_intersections"))
        and isinstance(row.get("used_states"), list)
        and all(_integer(v) and 0 <= v <= HORIZON for v in row["used_states"])
        and _integer(row.get("valid_until_state"))
        and 0 <= row["valid_until_state"] <= HORIZON
        and row.get("fusion_status") == "fused"
    )


def _valid_obstacle(row: Any) -> bool:
    def external_valid(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, list)
            and len(item) == 3
            and _integer(item[0])
            and 0 <= item[0] < 4
            and _integer(item[1])
            and item[1] > 0
            and _rectangles_valid([item[2]])
            for item in value
        )

    return (
        isinstance(row, dict)
        and set(row)
        == {
            "state_t",
            "local_visible",
            "external_fused",
            "active_report_ids",
            "min_distance",
            "collision",
            "out_of_bounds",
            "route_alert",
        }
        and _integer(row.get("state_t"))
        and 0 <= row["state_t"] <= HORIZON
        and _rectangles_valid(row.get("local_visible"))
        and external_valid(row.get("external_fused"))
        and isinstance(row.get("active_report_ids"), list)
        and all(_integer(v) and v >= 0 for v in row["active_report_ids"])
        and _finite(row.get("min_distance"))
        and float(row["min_distance"]) >= -EPS
        and all(isinstance(row.get(k), bool) for k in ("collision", "out_of_bounds", "route_alert"))
    )


def _valid_collection(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and set(row)
        == {
            "sensor",
            "total_bits",
            "valid_report_count",
            "first_valid_state",
            "last_valid_state",
            "fairness_component",
            "deadline_met",
        }
        and _integer(row.get("sensor"))
        and 0 <= row["sensor"] < 4
        and all(
            _finite(row.get(k)) and float(row[k]) >= -EPS
            for k in ("total_bits", "fairness_component")
        )
        and _integer(row.get("valid_report_count"))
        and row["valid_report_count"] >= 0
        and (
            row.get("first_valid_state") is None
            or (_integer(row["first_valid_state"]) and 0 <= row["first_valid_state"] <= HORIZON)
        )
        and (
            row.get("last_valid_state") is None
            or (_integer(row["last_valid_state"]) and 0 <= row["last_valid_state"] <= HORIZON)
        )
        and isinstance(row.get("deadline_met"), bool)
    )


def _evidence_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    def rows_ok(name: str, field: str, validator: Any, key: Any) -> bool:
        if name not in scenarios:
            return False
        rows = scenarios[name].get(f"{field}_rows")
        if not isinstance(rows, list) or not all(validator(row) for row in rows):
            return False
        keys = [key(row) for row in rows]
        return keys == sorted(keys)

    checks = [
        (
            "decisions",
            all(
                rows_ok(n, "decisions", _decision_evidence_valid, lambda r: r["t"])
                for n in SCENARIO_NAMES
            ),
            "typed decisions in increasing t order",
        ),
        (
            "states",
            all(rows_ok(n, "states", _valid_state, lambda r: r["state_t"]) for n in SCENARIO_NAMES),
            "typed states in increasing state_t order",
        ),
        (
            "links",
            all(
                rows_ok(n, "links", _valid_link, lambda r: (r["t"], r["sensor"]))
                for n in SCENARIO_NAMES
            ),
            "typed links in increasing t then sensor order",
        ),
        (
            "reports",
            all(
                n in scenarios
                and rows_ok(n, "reports", _valid_report, lambda r: r["report_id"])
                and [r["report_id"] for r in scenarios[n]["reports_rows"]]
                == list(range(len(scenarios[n]["reports_rows"])))
                for n in SCENARIO_NAMES
            ),
            "typed reports in generation/report_id order",
        ),
        (
            "obstacles",
            all(
                rows_ok(n, "obstacles", _valid_obstacle, lambda r: r["state_t"])
                for n in SCENARIO_NAMES
            ),
            "typed obstacle snapshots in increasing state_t order",
        ),
        (
            "collection",
            all(
                rows_ok(n, "collection", _valid_collection, lambda r: r["sensor"])
                for n in SCENARIO_NAMES
            ),
            "typed collection records in increasing sensor order",
        ),
    ]
    return [_atom("result_evidence", k, ok, d) for k, ok, d in checks]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios missing"]
    groups: dict[str, list[Any]] = {n: [] for n in SCENARIO_NAMES}
    for i, row in enumerate(scenarios):
        name = row.get("scenario") if isinstance(row, dict) else None
        if name in groups:
            groups[name].append(row)
        else:
            defects.append(f"scenarios[{i}]: unknown scenario")

    def count(rows: Any, keys: set[Any], field: str, label: str, key_fn: Any) -> None:
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing {k}" for k in keys)
            return
        seen: dict[Any, int] = {k: 0 for k in keys}
        for i, row in enumerate(rows):
            key = key_fn(row) if isinstance(row, dict) else None
            if key in seen:
                seen[key] += 1
            else:
                defects.append(f"{label}[{i}]: unknown key")
        defects.extend(f"{label}: missing {k}" for k, v in seen.items() if v == 0)
        defects.extend(
            f"{label}: duplicate {k}" for k, v in seen.items() if v > 1 for _ in range(v - 1)
        )

    for name in SCENARIO_NAMES:
        if not groups[name]:
            defects.append(f"scenario {name}: missing record")
            continue
        if len(groups[name]) > 1:
            defects.extend(f"scenario {name}: duplicate record" for _ in groups[name][1:])
            continue
        row = groups[name][0]
        count(
            row.get("decisions"),
            set(range(HORIZON)),
            "t",
            f"{name}.decisions",
            lambda x: x.get("t"),
        )
        count(
            row.get("states"),
            set(range(HORIZON + 1)),
            "state_t",
            f"{name}.states",
            lambda x: x.get("state_t"),
        )
        count(
            row.get("links"),
            {(s, t) for s in range(4) for t in range(HORIZON)},
            "",
            f"{name}.links",
            lambda x: (x.get("sensor"), x.get("t")),
        )
        count(
            row.get("obstacles"),
            set(range(HORIZON + 1)),
            "state_t",
            f"{name}.obstacles",
            lambda x: x.get("state_t"),
        )
        count(
            row.get("collection"),
            set(range(4)),
            "sensor",
            f"{name}.collection",
            lambda x: x.get("sensor"),
        )
    score = max(0.0, 1.0 - len(defects) / 8.0)
    return score, defects


def _legality_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    complete = all(n in scenarios for n in SCENARIO_NAMES)
    checks: dict[str, tuple[bool, str]] = {}

    def drows(name: str) -> list[Any]:
        rows = scenarios[name].get("decisions_rows", [])
        return rows if isinstance(rows, list) else []

    evidence_ok = complete and all(
        all(_decision_evidence_valid(row) for row in drows(name)) for name in SCENARIO_NAMES
    )
    checks["motion"] = (
        evidence_ok
        and all(
            all(
                -EPS <= float(row["v"]) <= 0.5 + EPS and -1 - EPS <= float(row["omega"]) <= 1 + EPS
                for row in drows(name)
            )
            for name in SCENARIO_NAMES
        ),
        "v and omega bounds",
    )
    checks["powers"] = (
        evidence_ok
        and all(
            all(
                all(
                    any(abs(float(power) - level) <= EPS for level in POWER_LEVELS)
                    for power in row["powers"]
                )
                for row in drows(name)
            )
            for name in SCENARIO_NAMES
        ),
        "fixed power levels",
    )
    checks["power_domain"] = (
        evidence_ok
        and all(
            all(
                sum(float(power) for power in row["powers"]) <= P_TOTAL + EPS for row in drows(name)
            )
            for name in SCENARIO_NAMES
        ),
        "per-slot total power",
    )
    checks["total_power"] = (
        evidence_ok
        and all(
            all(
                _close(row["total_power"], sum(float(power) for power in row["powers"]))
                for row in drows(name)
            )
            for name in SCENARIO_NAMES
        ),
        "total_power equals powers",
    )
    checks["energy"] = (
        evidence_ok
        and all(
            sum(sum(float(power) for power in row["powers"]) * DT for row in drows(name))
            <= ENERGY_BUDGET + EPS
            for name in SCENARIO_NAMES
        ),
        "cumulative energy budget from submitted powers",
    )

    def transitions(rows: Any) -> tuple[bool, bool]:
        if not isinstance(rows, list) or not all(_decision_evidence_valid(row) for row in rows):
            return False, False
        ordered = sorted(rows, key=lambda row: row["t"])
        if [row["t"] for row in ordered] != list(range(HORIZON)):
            return True, True
        previous = {"v": 0.0, "omega": 0.0}
        adjacent = True
        for current in ordered:
            adjacent = (
                adjacent
                and abs(float(current["v"]) - float(previous["v"])) <= 0.2 + EPS
                and abs(float(current["omega"]) - float(previous["omega"])) <= 0.5 + EPS
            )
            previous = current
        tx = [row for row in ordered if sum(float(power) for power in row["powers"]) > EPS]
        transmitted = all(
            math.hypot(float(a["v"]) - float(b["v"]), float(a["omega"]) - float(b["omega"]))
            <= 0.75 + EPS
            for a, b in zip(tx, tx[1:])
        )
        return adjacent, transmitted

    transition_results = (
        {name: transitions(scenarios[name]["decisions_rows"]) for name in SCENARIO_NAMES}
        if complete
        else {}
    )
    checks["continuity"] = (
        complete and all(result[0] for result in transition_results.values()),
        "adjacent action changes, including the zero action at slot -1",
    )
    checks["transmit_continuity"] = (
        complete and all(result[1] for result in transition_results.values()),
        "successive nonzero-power transmitted command distance, including "
        "transmissions separated by zero-power slots",
    )
    return [_atom("decision_legality", k, v[0], v[1]) for k, v in checks.items()]


def _replay_atoms(scenarios: Mapping[str, Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    names = [k.removeprefix("replay_validity.") for k in ATOM_SHARES["replay_validity"]]
    ok = {k: True for k in names}
    detail = {k: "" for k in names}
    replayed: dict[str, Any] = {}

    def fail(rule: str, msg: str) -> None:
        if ok[rule]:
            ok[rule] = False
            detail[rule] = msg

    for name in SCENARIO_NAMES:
        if (
            name not in scenarios
            or not isinstance(scenarios[name].get("decisions_rows"), list)
            or len(scenarios[name]["decisions_rows"]) != HORIZON
        ):
            for rule in names:
                fail(rule, f"{name}: complete decisions required")
            continue
        decisions = {
            row["t"]: row
            for row in scenarios[name]["decisions_rows"]
            if isinstance(row, dict) and _valid_decision(row)
        }
        if set(decisions) != set(range(HORIZON)):
            for rule in names:
                fail(rule, f"{name}: decision coverage invalid")
            continue
        expected = _replay_scenario(name, decisions)
        replayed[name] = expected
        row = scenarios[name]
        state_rows = row["states_rows"] if isinstance(row.get("states_rows"), list) else []
        if len(state_rows) != HORIZON + 1:
            fail("motion", f"{name}.states: complete state evidence required")
            fail("states", f"{name}.states: complete state evidence required")
            fail("goal", f"{name}.states: complete state evidence required")
        elif not all(_valid_state(r) for r in state_rows):
            fail("motion", f"{name}.states: typed state evidence required")
            fail("states", f"{name}.states: typed state evidence required")
            fail("goal", f"{name}.states: typed state evidence required")
        state_map = {r.get("state_t"): r for r in state_rows if isinstance(r, dict)}
        for t, wanted in enumerate(expected["states"]):
            actual = state_map.get(t)
            if actual is None:
                continue
            if not all(
                _close(actual.get(field), wanted[field])
                for field in ("x", "y", "theta", "v", "omega")
            ):
                fail("motion", f"{name}.states[{t}] motion state mismatch")
            if not all(
                _close(actual.get(field), wanted[field])
                for field in ("remaining_tx_energy", "route_deviation")
            ) or any(
                actual.get(field) != wanted[field]
                for field in ("done", "collision", "out_of_bounds")
            ):
                fail("states", f"{name}.states[{t}] resource, route, or terminal flag mismatch")
            if any(
                actual.get(field) != wanted[field]
                for field in ("goal_reached", "goal_arrival_state")
            ):
                fail("goal", f"{name}.states[{t}] goal evidence mismatch")
        link_rows = row["links_rows"] if isinstance(row.get("links_rows"), list) else []
        if len(link_rows) != 4 * HORIZON:
            fail("links", f"{name}.links: complete link evidence required")
        elif not all(_valid_link(r) for r in link_rows):
            fail("links", f"{name}.links: typed link evidence required")
        for actual, wanted in zip(link_rows, expected["links"]):
            if not isinstance(actual, dict):
                continue
            for field in ("sensor", "t", "threshold_met"):
                if actual.get(field) != wanted[field]:
                    fail("links", f"{name}.links key mismatch")
            if not all(
                _close(actual.get(field), wanted[field])
                for field in ("distance", "power", "rate", "bits")
            ):
                fail("links", f"{name}.links numeric mismatch")
        report_rows = row["reports_rows"] if isinstance(row.get("reports_rows"), list) else []
        if not isinstance(row.get("reports_rows"), list):
            fail("reports", f"{name}.reports: report list required")
            fail("report_evidence", f"{name}.reports: report list required")
        actual_reports = {r.get("report_id"): r for r in report_rows if isinstance(r, dict)}
        if set(actual_reports) != set(range(len(expected["reports"]))):
            fail("reports", f"{name}.reports IDs mismatch")
        for wanted in expected["reports"]:
            actual = actual_reports.get(wanted["report_id"])
            if actual is None:
                continue
            for field in (
                "sensor",
                "report_version",
                "generated_slot",
                "arrival_state",
                "trigger_slots",
            ):
                if not _deep_close(actual.get(field), wanted[field]):
                    fail("reports", f"{name}.reports[{wanted['report_id']}].{field} mismatch")
            for field in (
                "region",
                "obstacle_intersections",
                "valid_until_state",
                "fusion_status",
                "used_states",
            ):
                if not _deep_close(actual.get(field), wanted[field]):
                    fail(
                        "report_evidence", f"{name}.reports[{wanted['report_id']}].{field} mismatch"
                    )
        obstacle_rows = row["obstacles_rows"] if isinstance(row.get("obstacles_rows"), list) else []
        if len(obstacle_rows) != HORIZON + 1:
            fail("obstacles", f"{name}.obstacles: complete obstacle evidence required")
        elif not all(_valid_obstacle(r) for r in obstacle_rows):
            fail("obstacles", f"{name}.obstacles: typed obstacle evidence required")
        obstacle_map = {r.get("state_t"): r for r in obstacle_rows if isinstance(r, dict)}
        for wanted in expected["obstacles"]:
            actual = obstacle_map.get(wanted["state_t"])
            if actual is None:
                continue
            if (
                not _deep_close(actual.get("local_visible"), wanted["local_visible"])
                or not _deep_close(actual.get("external_fused"), wanted["external_fused"])
                or actual.get("active_report_ids") != wanted["active_report_ids"]
                or not _close(actual.get("min_distance"), wanted["min_distance"])
            ):
                fail("obstacles", f"{name}.obstacles[{wanted['state_t']}] geometry mismatch")
            if any(
                actual.get(field) != wanted[field]
                for field in ("collision", "out_of_bounds", "route_alert")
            ):
                fail("obstacles", f"{name}.obstacles[{wanted['state_t']}] flags mismatch")
        collection_rows = (
            row["collection_rows"] if isinstance(row.get("collection_rows"), list) else []
        )
        if len(collection_rows) != 4:
            fail("collection", f"{name}.collection: complete collection evidence required")
        elif not all(_valid_collection(r) for r in collection_rows):
            fail("collection", f"{name}.collection: typed collection evidence required")
        collection_map = {r.get("sensor"): r for r in collection_rows if isinstance(r, dict)}
        for wanted in expected["collection"]:
            actual = collection_map.get(wanted["sensor"])
            if actual is None:
                continue
            if not _close(actual.get("total_bits"), wanted["total_bits"]) or not _close(
                actual.get("fairness_component"), wanted["fairness_component"]
            ):
                fail("collection", f"{name}.collection[{wanted['sensor']}] numeric mismatch")
            if any(
                actual.get(field) != wanted[field]
                for field in (
                    "valid_report_count",
                    "first_valid_state",
                    "last_valid_state",
                    "deadline_met",
                )
            ):
                fail(
                    "collection", f"{name}.collection[{wanted['sensor']}] report/deadline mismatch"
                )
    atoms = [
        _atom("replay_validity", k, ok[k], detail[k] or "fixed replay comparison passed")
        for k in names
    ]
    return atoms, replayed


def _references() -> tuple[float, float]:
    best_total = 0.0
    best_one = 0.0
    import itertools

    for powers in itertools.product(POWER_LEVELS, repeat=4):
        if sum(powers) <= P_TOTAL + EPS:
            rates = [_rate(0.5, p) for p in powers]
            best_total = max(best_total, sum(rates) * DT * HORIZON)
            best_one = max(best_one, max(rates) * DT * HORIZON)
    return best_total, best_one


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    total_ref, fair_ref = _references()
    arrival_scores = []
    collection_scores = []
    power_scores = []
    freshness_scores = []
    for name in SCENARIO_NAMES:
        exp = replayed[name]
        spec = SCENARIOS[name]
        states = exp["states"]
        links = exp["links"]
        final = states[-1]
        safe = bool(
            final["goal_reached"]
            and final["goal_arrival_state"] is not None
            and final["goal_arrival_state"] <= spec["deadline"]
            and not final["collision"]
            and not final["out_of_bounds"]
        )
        arrival_scores.append(1.0 if safe else 0.0)
        bits = [0.0] * 4
        for item in links:
            bits[item["sensor"]] += item["bits"]
        collection_scores.append(
            0.6 * min(1.0, sum(bits) / total_ref) + 0.4 * min(1.0, min(bits) / fair_ref)
        )
        total_power = sum(item["power"] for item in links)
        power_scores.append(max(0.0, min(1.0, 1.0 - total_power / (HORIZON * P_TOTAL))))
        first_arrivals = [
            r["arrival_state"] for r in exp["reports"] if r.get("fusion_status") == "fused"
        ]
        aq = max(0.0, min(1.0, 1.0 - min(first_arrivals) / HORIZON)) if first_arrivals else 0.0
        fresh = 0.0
        for sensor in range(4):
            for st in range(1, HORIZON + 1):
                active = [
                    r
                    for r in exp["reports"]
                    if r["sensor"] == sensor and r["arrival_state"] <= st <= r["valid_until_state"]
                ]
                fresh += max(
                    (max(0.0, min(1.0, 1.0 - (st - r["arrival_state"]) / 4.0)) for r in active),
                    default=0.0,
                )
        freshness_scores.append(0.6 * aq + 0.4 * fresh / (4 * HORIZON))
    return {
        "primary_metrics_1": sum(arrival_scores) / 5,
        "primary_metrics_2": sum(collection_scores) / 5,
        "auxiliary_metrics_1": sum(power_scores) / 5,
        "auxiliary_metrics_2": sum(freshness_scores) / 5,
    }


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs mismatch")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("invalid shares")
    if any(not math.isclose(a.share, expected[a.id], rel_tol=0.0, abs_tol=1e-12) for a in atoms):
        raise ValueError("atom share mismatch")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    return (
        "failed_atoms=[]"
        if not failed
        else "failed_atoms=["
        + ",".join(a.id for a in failed)
        + "]; "
        + " | ".join(f"{a.id}: {a.detail}" for a in failed)
    )


def _report_score(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "report.md"
    if not path.is_file() or path.is_symlink():
        return 0.0, "report.md is not a regular non-symlink file"
    text = path.read_text(encoding="utf-8")
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    found: set[str] = set()
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.MULTILINE))
    for i, heading in enumerate(headings):
        # Markdown permits optional closing hashes (for example, `## Results ##`).
        title = re.sub(r"\s+#+\s*$", "", heading.group(2)).strip()
        level = len(heading.group(1))
        if title not in required:
            continue
        end = len(text)
        for later in headings[i + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        if text[heading.end() : end].strip():
            found.add(title)
    score = max(0.0, min(1.0, 1.0 - (len(required) - len(found)) / 4.0))
    return (
        score,
        f"missing_sections={[x for x in required if x not in found]}; normalized={score:.9g}",
    )


def _regular_file(ws: Workspace, rel: str) -> bool:
    p = Path(ws.root) / rel
    return p.is_file() and not p.is_symlink()


def _evaluate_payload(data: Any, official_ok: bool = True, parse_ok: bool = True) -> EvalState:
    state = EvalState()
    state.data = data
    state.official_ok = official_ok
    state.parse_ok = parse_ok
    gate_ids = (
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    )
    if not official_ok or not parse_ok:
        reason = "official run failed" if not official_ok else "strict parse failed"
        for check_id in gate_ids:
            state.scores[check_id] = 0.0
            state.diagnostics[check_id] = reason
        return state
    try:
        result_atoms = _result_json_atoms(data, parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = _diagnose(result_atoms)
        state.scenarios = _index_records(data)
        state.records = state.scenarios
        evidence_atoms = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(
            evidence_atoms, ATOM_SHARES["result_evidence"]
        )
        state.diagnostics["result_evidence"] = _diagnose(evidence_atoms)
        coverage_score, coverage_defects = _coverage_analysis(data)
        state.scores["canonical_coverage"] = coverage_score
        state.diagnostics["canonical_coverage"] = (
            f"defect_count={len(coverage_defects)}; {coverage_defects[:20]}"
        )
        legality_atoms = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(
            legality_atoms, ATOM_SHARES["decision_legality"]
        )
        state.diagnostics["decision_legality"] = _diagnose(legality_atoms)
        replay_atoms, state.replayed = _replay_atoms(state.scenarios)
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in gate_ids
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:
        state.official_ok = False
        state.official_message = f"grader state failed: {type(exc).__name__}: {exc}"
        state.metrics = None
        state.validity_gate = False
        for check_id in gate_ids:
            state.scores[check_id] = 0.0
            state.diagnostics[check_id] = state.official_message
    return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    started = time.monotonic()
    remove_ok = False
    try:
        state.run_started = True
        try:
            remove_ok = (
                getattr(
                    await ws.run("rm -f -- results.json", check=False, timeout=30.0),
                    "exit_code",
                    None,
                )
                == 0
            )
        except Exception as exc:
            state.official_message = f"could not remove old results.json: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            run_text = str(getattr(result, "stderr", "")) + " " + str(getattr(result, "stdout", ""))
            state.timed_out = state.exit_code is None and any(
                token in run_text.lower() for token in ("timeout", "timed out")
            )
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or any(
                token in str(exc).lower() for token in ("timeout", "timed out")
            )
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = (
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600 + EPS
            and state.result_file_is_new_regular
        )
        state.official_message = (
            state.official_message
            or f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
            f"new_regular_results={state.result_file_is_new_regular}"
        )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except Exception as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        if not state.official_ok or not state.parse_ok:
            for n in (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[n] = 0.0
                state.diagnostics[n] = "official run or strict parse failed"
            _CACHE[key] = state
            return state
        evaluated = _evaluate_payload(state.data)
        state.official_ok = evaluated.official_ok
        state.records = evaluated.records
        state.scenarios = evaluated.scenarios
        state.replayed = evaluated.replayed
        state.scores = evaluated.scores
        state.diagnostics.update(evaluated.diagnostics)
        state.metrics = evaluated.metrics
        state.validity_gate = evaluated.validity_gate
        if evaluated.official_message:
            state.official_message = evaluated.official_message
    except Exception as exc:
        state.official_ok = False
        state.official_message = (
            state.official_message or f"grader state failed: {type(exc).__name__}: {exc}"
        )
        for n in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores[n] = 0.0
            state.diagnostics[n] = state.official_message
    _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    s = await _build_state(ws)
    return (1.0, s.official_message) if s.official_ok else (0.0, s.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    s = await _build_state(ws)
    if name.startswith(("primary_metrics_", "auxiliary_metrics_")) and s.metrics is None:
        return 0.0, f"{name}: validity gate not satisfied"
    return s.scores.get(name, s.metrics.get(name, 0.0) if s.metrics else 0.0), s.diagnostics.get(
        name,
        f"{name}: independently replayed "
        f"value={s.metrics.get(name, 0.0) if s.metrics else 0.0:.9g}",
    )


for _id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _check(ws: Workspace, _name: str = _id) -> tuple[float, str]:
        return await _score(ws, _name)

    globals()[_id] = check(id=_id)(_check)
del _check, _id
