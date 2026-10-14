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
DT = 0.1
EPS = 1e-6
MASS = 10.0
INERTIA = 0.06
DOMAIN = (0.0, 8.0, 0.0, 8.0)
ROBOT_COUNT = 4
ROBOT_RADIUS = 0.1
CONTACT_RADIUS = 0.35
ROBOT_SEPARATION = 0.2
OBSTACLE_SENSING_RANGE = 1.0
PAYLOAD_HALF_DIAG = math.hypot(0.4, 0.2)
SCENARIO_NAMES = ("straight_transport", "rotation_balance", "narrow_passage", "link_partition", "robot_dropout")
CONTACT_OFFSETS = ((-0.45, -0.25), (-0.45, 0.25), (0.45, -0.25), (0.45, 0.25))
SCENARIOS: dict[str, dict[str, Any]] = {
    "straight_transport": {"start": (1.5, 1.5, 0.0), "goal": (3.0, 1.5, 0.0), "obstacles": (), "radius": 2.0, "q": 0.10, "ell": 0.02, "c": 11, "failure": None},
    "rotation_balance": {"start": (1.5, 5.8, 0.0), "goal": (2.6, 5.8, math.pi / 2), "obstacles": (), "radius": 1.4, "q": 0.15, "ell": 0.04, "c": 23, "failure": None},
    "narrow_passage": {"start": (4.0, 2.0, 0.0), "goal": (5.4, 2.0, 0.0), "obstacles": ((4.5, 5.0, 0.0, 1.15), (4.5, 5.0, 2.85, 4.0)), "radius": 1.2, "q": 0.20, "ell": 0.06, "c": 37, "failure": None},
    "link_partition": {"start": (2.0, 4.0, 0.0), "goal": (3.4, 4.6, math.pi / 4), "obstacles": ((2.5, 2.9, 4.9, 6.0),), "radius": 0.85, "q": 0.30, "ell": 0.10, "c": 53, "failure": None},
    "robot_dropout": {"start": (5.5, 5.5, 0.0), "goal": (6.7, 5.0, -math.pi / 6), "obstacles": ((6.0, 6.4, 5.8, 7.0),), "radius": 1.3, "q": 0.20, "ell": 0.08, "c": 71, "failure": (2, 16, 31)},
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {"result_json.parse": 0.25, "result_json.root": 0.25, "result_json.scenarios": 0.25, "result_json.scenario_keys": 0.25},
    "result_evidence": {"result_evidence.payload": 1 / 7, "result_evidence.robots": 1 / 7, "result_evidence.estimates": 1 / 7, "result_evidence.messages": 1 / 7, "result_evidence.contacts": 1 / 7, "result_evidence.safety": 1 / 7, "result_evidence.types": 1 / 7},
    "decision_legality": {"decision_legality.time": 1 / 7, "decision_legality.velocity": 1 / 7, "decision_legality.force": 1 / 7, "decision_legality.failure": 1 / 7, "decision_legality.channel": 1 / 7, "decision_legality.burst": 1 / 7, "decision_legality.total": 1 / 7},
    "replay_validity": {"replay_validity.payload": 1 / 7, "replay_validity.robots": 1 / 7, "replay_validity.communication": 1 / 7, "replay_validity.estimates": 1 / 7, "replay_validity.contacts": 1 / 7, "replay_validity.safety": 1 / 7, "replay_validity.messages": 1 / 7},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = id, bool(passed), detail, float(share)


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
    return AtomVerdict(f"{group}.{name}", passed, detail, ATOM_SHARES[group][f"{group}.{name}"])


def _finite(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _close(a: Any, b: Any, tol: float = EPS) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tol


def _same(actual: Any, expected: Any) -> bool:
    if _finite(expected):
        return _close(actual, expected)
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(_same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _difference(actual: Any, expected: Any, path: str) -> str | None:
    """Return a deterministic, actionable description of the first mismatch."""
    if _finite(expected):
        if not _close(actual, expected):
            return (
                f"{path}: expected={expected!r}, actual={actual!r}, "
                f"tolerance={EPS}"
            )
        return None
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path}: expected=object, actual={type(actual).__name__}"
        if set(actual) != set(expected):
            return f"{path}: expected_keys={sorted(expected, key=repr)!r}, actual_keys={sorted(actual, key=repr)!r}"
        for key, value in expected.items():
            difference = _difference(actual[key], value, f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected=list[{len(expected)}], actual={type(actual).__name__}"
        if len(actual) != len(expected):
            return f"{path}: expected_length={len(expected)}, actual_length={len(actual)}"
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            difference = _difference(actual_value, expected_value, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if actual != expected:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def _contact_points(p: Sequence[float], theta: float) -> list[list[float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [[p[0] + c * ox - s * oy, p[1] + s * ox + c * oy] for ox, oy in CONTACT_OFFSETS]


def _point_rect_distance(point: Sequence[float], rect: Sequence[float]) -> float:
    x, y = float(point[0]), float(point[1]); xmin, xmax, ymin, ymax = map(float, rect)
    return math.hypot(max(xmin - x, 0.0, x - xmax), max(ymin - y, 0.0, y - ymax))


def _segment_rect_intersects(a: Sequence[float], b: Sequence[float], rect: Sequence[float], pad: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = float(rect[0]) - pad, float(rect[1]) + pad, float(rect[2]) - pad, float(rect[3]) + pad
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    u0, u1 = 0.0, 1.0
    for p, q in ((-dx, float(a[0]) - xmin), (dx, xmax - float(a[0])), (-dy, float(a[1]) - ymin), (dy, ymax - float(a[1]))):
        if abs(p) <= EPS:
            if q < -EPS:
                return False
        elif p < 0:
            u0 = max(u0, q / p)
        else:
            u1 = min(u1, q / p)
        if u0 - u1 > EPS:
            return False
    return True


def _payload_obstacle_distance(p: Sequence[float], obstacles: Sequence[Sequence[float]]) -> float:
    # Safety uses the explicitly recorded payload-center distance to each closed obstacle.
    return min((_point_rect_distance(p, rect) for rect in obstacles), default=8.0)


def _payload_safe(p: Sequence[float], obstacles: Sequence[Sequence[float]]) -> bool:
    if not (DOMAIN[0] - EPS <= p[0] <= DOMAIN[1] + EPS and DOMAIN[2] - EPS <= p[1] <= DOMAIN[3] + EPS):
        return False
    return _payload_obstacle_distance(p, obstacles) >= 0.15 - EPS


def _active(name: str, robot: int, t: int) -> bool:
    failure = SCENARIOS[name]["failure"]
    return failure is None or robot != failure[0] or not (failure[1] <= t <= failure[2])


def _hash(name: str, t: int, sender: int, receiver: int, k: int) -> float:
    c_s = SCENARIOS[name]["c"]
    return ((1103515245 * (t + 1) + 12345 * (sender + 1) + 2654435761 * (receiver + 1) + 97 * c_s + 53 * k) % 1000) / 1000.0


def _neighbor_graph(name: str, positions: Sequence[Sequence[float]], active: Sequence[bool]) -> list[list[int]]:
    radius = SCENARIOS[name]["radius"]
    out = [[] for _ in range(ROBOT_COUNT)]
    for i in range(ROBOT_COUNT):
        if not active[i]:
            continue
        for j in range(ROBOT_COUNT):
            if i != j and active[j] and math.dist(positions[i], positions[j]) <= radius + EPS:
                out[i].append(j)
        out[i].sort()
    return out


def _connected(graph: Sequence[Sequence[int]], active: Sequence[bool]) -> bool:
    nodes = [i for i, ok in enumerate(active) if ok]
    if len(nodes) <= 1:
        return True
    seen = {nodes[0]}; stack = [nodes[0]]
    while stack:
        i = stack.pop()
        for j in graph[i]:
            if j not in seen:
                seen.add(j); stack.append(j)
    return len(seen) == len(nodes)


def _obs(name: str, t: int, robot: int, p: Sequence[float], theta: float, payload_v: Sequence[float], omega: float, positions: Sequence[Sequence[float]], prev_forces: Sequence[Sequence[float]], prev_torques: Sequence[float]) -> dict[str, Any]:
    contacts = _contact_points(p, theta)
    d = math.dist(positions[robot], contacts[robot])
    graph = _neighbor_graph(name, positions, [_active(name, k, t) for k in range(ROBOT_COUNT)])
    visible = [list(rect) for rect in SCENARIOS[name]["obstacles"] if _point_rect_distance(positions[robot], rect) <= OBSTACLE_SENSING_RANGE + EPS]
    visible.sort()
    return {"payload_x": p[0], "payload_y": p[1], "payload_theta": theta, "payload_vx": payload_v[0], "payload_vy": payload_v[1], "payload_omega": omega, "contact_x": contacts[robot][0], "contact_y": contacts[robot][1], "contact_distance": d, "measured_force_x": prev_forces[robot][0], "measured_force_y": prev_forces[robot][1], "measured_torque": prev_torques[robot], "neighbor_ids": list(graph[robot]), "visible_obstacles": visible}


def _initial_positions(spec: Mapping[str, Any]) -> list[list[float]]:
    return _contact_points(spec["start"][:2], spec["start"][2])


def _replay_scenario(name: str, rows: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    p = [float(spec["start"][0]), float(spec["start"][1])]
    theta = float(spec["start"][2])
    v = [0.0, 0.0]
    omega = 0.0
    positions = _initial_positions(spec)
    prev_forces = [[0.0, 0.0] for _ in range(ROBOT_COUNT)]
    prev_torques = [0.0] * ROBOT_COUNT
    pending: dict[int, list[dict[str, Any]]] = {}
    cache: list[dict[int, dict[str, Any]]] = [dict() for _ in range(ROBOT_COUNT)]
    broadcast_id = 0
    payload: list[dict[str, Any]] = []
    robots: list[dict[str, Any]] = []
    estimates: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    safety: list[dict[str, Any]] = []
    goal = False
    violation = False
    goal_p = spec["goal"][:2]
    goal_theta = spec["goal"][2]

    robot_rows = rows.get("robots", {}) if isinstance(rows.get("robots"), Mapping) else {}

    def record_boundary(state_t: int) -> None:
        nonlocal goal, violation
        active = [_active(name, i, state_t) for i in range(ROBOT_COUNT)]
        graph = _neighbor_graph(name, positions, active)
        obstacle_distance = _payload_obstacle_distance(p, spec["obstacles"])
        min_robot_distance = min(math.dist(positions[i], positions[j]) for i in range(ROBOT_COUNT) for j in range(i + 1, ROBOT_COUNT))
        collision = obstacle_distance <= EPS or min_robot_distance < ROBOT_SEPARATION - EPS
        payload_safe = _payload_safe(p, spec["obstacles"])
        violation = violation or collision or not payload_safe
        position_error = math.dist(p, goal_p)
        heading_error = abs(_wrap(theta - goal_theta))
        goal = goal or (position_error <= 0.1 + EPS and heading_error <= math.radians(10.0) + EPS)
        payload.append({"state_t": state_t, "x": p[0], "y": p[1], "theta": theta, "vx": v[0], "vy": v[1], "omega": omega, "position_error": position_error, "heading_error": heading_error, "goal_reached": goal, "safety_violation": violation})
        safety.append({"state_t": state_t, "obstacle_distance": obstacle_distance, "min_robot_distance": min_robot_distance, "collision": collision, "payload_safe": payload_safe, "connected": _connected(graph, active), "goal_reached": goal})

    def recent_count(robot: int, state_t: int) -> int:
        return sum(bool(robot_rows.get((robot, old_t), {}).get("broadcast", False)) for old_t in range(max(0, state_t - 3), state_t))

    record_boundary(0)
    for t in range(HORIZON):
        arrivals = [m for m in pending.pop(t, []) if m["delivered"]]
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for message in arrivals:
            grouped.setdefault((message["receiver"], message["sender"]), []).append(message)
        for (receiver, sender), candidates in grouped.items():
            newest = max(candidates, key=lambda m: (m["generation_slot"], m["broadcast_id"]))
            old = cache[receiver].get(sender)
            if old is None or newest["generation_slot"] > old["generation_slot"]:
                newest["used"] = True
                cache[receiver][sender] = newest
        for receiver in range(ROBOT_COUNT):
            cache[receiver] = {sender: message for sender, message in cache[receiver].items() if t - message["generation_slot"] <= 4}

        for robot in range(ROBOT_COUNT):
            for neighbor in range(ROBOT_COUNT):
                if neighbor == robot:
                    continue
                message = cache[robot].get(neighbor)
                available = message is not None
                age = t - message["generation_slot"] if message is not None else None
                force_x = float(message["content"]["force_x"]) if message is not None else 0.0
                force_y = float(message["content"]["force_y"]) if message is not None else 0.0
                torque = float(message["content"]["torque"]) if message is not None else 0.0
                fusion_error = math.sqrt((force_x - prev_forces[neighbor][0]) ** 2 + (force_y - prev_forces[neighbor][1]) ** 2 + (torque - prev_torques[neighbor]) ** 2)
                estimates.append({"robot": robot, "state_t": t, "neighbor": neighbor, "force_x": force_x, "force_y": force_y, "torque": torque, "age": age, "available": available, "fusion_error": fusion_error})

        start_active = [_active(name, i, t) for i in range(ROBOT_COUNT)]
        start_graph = _neighbor_graph(name, positions, start_active)
        contact_points = _contact_points(p, theta)
        valid_forces: list[list[float]] = []
        valid_torques: list[float] = []
        executed_velocities: list[list[float]] = []
        for robot in range(ROBOT_COUNT):
            rr = robot_rows.get((robot, t), {})
            cmd_vx = float(rr.get("cmd_vx", 0.0))
            cmd_vy = float(rr.get("cmd_vy", 0.0))
            force_x = float(rr.get("force_x", 0.0))
            force_y = float(rr.get("force_y", 0.0))
            torque = float(rr.get("torque", 0.0))
            broadcast = bool(rr.get("broadcast", False))
            content = rr.get("broadcast_content", {})
            broadcast_content = {"force_x": float(content.get("force_x", 0.0)), "force_y": float(content.get("force_y", 0.0)), "torque": float(content.get("torque", 0.0))}
            active = start_active[robot]
            executed_vx, executed_vy = (cmd_vx, cmd_vy) if active else (0.0, 0.0)
            distance = math.dist(positions[robot], contact_points[robot])
            valid = active and distance <= CONTACT_RADIUS + EPS
            applied_force_x, applied_force_y, applied_torque = (force_x, force_y, torque) if valid else (0.0, 0.0, 0.0)
            observation = _obs(name, t, robot, p, theta, v, omega, positions, prev_forces, prev_torques)
            robots.append({"robot": robot, "state_t": t, "x": positions[robot][0], "y": positions[robot][1], "executed_vx": executed_vx, "executed_vy": executed_vy, "applied_force_x": applied_force_x, "applied_force_y": applied_force_y, "applied_torque": applied_torque, "active": active, "observation": observation, "recent_broadcast_count": recent_count(robot, t), "cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "force_x": force_x, "force_y": force_y, "torque": torque, "broadcast": broadcast, "broadcast_content": broadcast_content})
            contacts.append({"robot": robot, "t": t, "contact_distance": distance, "valid_force_x": applied_force_x, "valid_force_y": applied_force_y, "valid_torque": applied_torque, "contact_degree": sum(1 for other in range(ROBOT_COUNT) if other != robot and start_active[other] and math.dist(positions[robot], positions[other]) <= 0.55 + EPS)})
            executed_velocities.append([executed_vx, executed_vy])
            valid_forces.append([applied_force_x, applied_force_y])
            valid_torques.append(applied_torque)

        total_force = [sum(force[0] for force in valid_forces), sum(force[1] for force in valid_forces)]
        total_torque = sum(_cross((contact_points[i][0] - p[0], contact_points[i][1] - p[1]), valid_forces[i]) + valid_torques[i] for i in range(ROBOT_COUNT))
        acceleration = [(total_force[0] - 0.2 * v[0]) / MASS, (total_force[1] - 0.2 * v[1]) / MASS]
        angular_acceleration = (total_torque - 0.1 * omega) / INERTIA
        v_new = [max(-1.0, min(1.0, v[k] + DT * acceleration[k])) for k in (0, 1)]
        omega_new = max(-2.0, min(2.0, omega + DT * angular_acceleration))
        p_new = [p[0] + DT * v_new[0], p[1] + DT * v_new[1]]
        theta_new = _wrap(theta + DT * omega_new)
        for robot in range(ROBOT_COUNT):
            positions[robot][0] += DT * executed_velocities[robot][0]
            positions[robot][1] += DT * executed_velocities[robot][1]

        for sender in range(ROBOT_COUNT):
            rr = robot_rows.get((sender, t), {})
            if bool(rr.get("broadcast", False)) and start_active[sender]:
                receivers = list(start_graph[sender])
                content = rr["broadcast_content"]
                for receiver in receivers:
                    delay = 2 if _hash(name, t, sender, receiver, 0) < spec["q"] else 1
                    arrival_state = t + delay
                    delivered = arrival_state <= HORIZON and _hash(name, t, sender, receiver, 1) >= spec["ell"]
                    message = {"broadcast_id": broadcast_id, "receiver": receiver, "sender": sender, "generation_slot": t, "arrival_state": arrival_state, "receivers": receivers, "content": {"force_x": float(content["force_x"]), "force_y": float(content["force_y"]), "torque": float(content["torque"]), "generation_slot": t, "sender": sender}, "delivered": delivered, "used": False, "channel_unit": True}
                    messages.append(message)
                    pending.setdefault(arrival_state, []).append(message)
                broadcast_id += 1

        p, theta, v, omega = p_new, theta_new, v_new, omega_new
        prev_forces, prev_torques = valid_forces, valid_torques
        record_boundary(t + 1)

    terminal_active = [_active(name, i, HORIZON) for i in range(ROBOT_COUNT)]
    zero_content = {"force_x": 0.0, "force_y": 0.0, "torque": 0.0}
    for robot in range(ROBOT_COUNT):
        robots.append({"robot": robot, "state_t": HORIZON, "x": positions[robot][0], "y": positions[robot][1], "executed_vx": 0.0, "executed_vy": 0.0, "applied_force_x": 0.0, "applied_force_y": 0.0, "applied_torque": 0.0, "active": terminal_active[robot], "observation": _obs(name, HORIZON, robot, p, theta, v, omega, positions, prev_forces, prev_torques), "recent_broadcast_count": recent_count(robot, HORIZON), "cmd_vx": 0.0, "cmd_vy": 0.0, "force_x": 0.0, "force_y": 0.0, "torque": 0.0, "broadcast": False, "broadcast_content": dict(zero_content)})
    return {"payload": payload, "robots": robots, "estimates": estimates, "messages": sorted(messages, key=lambda r: (r["broadcast_id"], r["receiver"])), "contacts": contacts, "safety": safety}


def _scenario_shape(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"scenario", "payload", "robots", "estimates", "messages", "contacts", "safety"}


def _index_records(data: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return out
    for row in data["scenarios"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str) or row["scenario"] in out:
            continue
        def keyed(field: str, key: Any) -> dict[Any, Any]:
            rows = row.get(field) if isinstance(row.get(field), list) else []; result: dict[Any, Any] = {}
            for item in rows:
                if isinstance(item, Mapping):
                    value = key(item)
                    try:
                        hash(value)
                    except TypeError:
                        continue
                    if value not in result:
                        result[value] = item
            return result
        out[row["scenario"]] = {"record": row, "payload_rows": row.get("payload", []), "robots_rows": row.get("robots", []), "estimates_rows": row.get("estimates", []), "messages_rows": row.get("messages", []), "contacts_rows": row.get("contacts", []), "safety_rows": row.get("safety", []), "payload": keyed("payload", lambda x: x.get("state_t")), "robots": keyed("robots", lambda x: (x.get("robot"), x.get("state_t"))), "estimates": keyed("estimates", lambda x: (x.get("robot"), x.get("state_t"), x.get("neighbor"))), "contacts": keyed("contacts", lambda x: (x.get("robot"), x.get("t"))), "safety": keyed("safety", lambda x: x.get("state_t"))}
    return out


def _rects(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(r, list) and len(r) == 4 and all(_finite(x) for x in r) and r[0] <= r[1] + EPS and r[2] <= r[3] + EPS for r in v)


def _sorted_unique_ints(v: Any) -> bool:
    """Check a JSON list before sorting it, so malformed types never raise."""
    return isinstance(v, list) and all(_integer(item) for item in v) and v == sorted(set(v))


def _valid_payload(r: Any) -> bool:
    return isinstance(r, Mapping) and set(r) == {"state_t", "x", "y", "theta", "vx", "vy", "omega", "position_error", "heading_error", "goal_reached", "safety_violation"} and _integer(r.get("state_t")) and 0 <= r["state_t"] <= HORIZON and all(_finite(r.get(k)) for k in ("x", "y", "theta", "vx", "vy", "omega", "position_error", "heading_error")) and all(isinstance(r.get(k), bool) for k in ("goal_reached", "safety_violation"))


def _valid_robot(r: Any) -> bool:
    keys = {"robot", "state_t", "x", "y", "executed_vx", "executed_vy", "applied_force_x", "applied_force_y", "applied_torque", "active", "observation", "recent_broadcast_count", "cmd_vx", "cmd_vy", "force_x", "force_y", "torque", "broadcast", "broadcast_content"}
    obs = r.get("observation") if isinstance(r, Mapping) else None
    obs_keys = {"payload_x", "payload_y", "payload_theta", "payload_vx", "payload_vy", "payload_omega", "contact_x", "contact_y", "contact_distance", "measured_force_x", "measured_force_y", "measured_torque", "neighbor_ids", "visible_obstacles"}
    obs_ok = isinstance(obs, Mapping) and set(obs) == obs_keys and all(_finite(obs.get(k)) for k in obs_keys - {"neighbor_ids", "visible_obstacles"}) and _sorted_unique_ints(obs.get("neighbor_ids")) and all(0 <= x < ROBOT_COUNT for x in obs["neighbor_ids"]) and _rects(obs.get("visible_obstacles")) and obs["visible_obstacles"] == sorted(obs["visible_obstacles"])
    content = r.get("broadcast_content") if isinstance(r, Mapping) else None
    content_ok = isinstance(content, Mapping) and set(content) == {"force_x", "force_y", "torque"} and all(_finite(content.get(k)) for k in ("force_x", "force_y", "torque")) and all(abs(float(content[k])) <= 6.0 + EPS for k in ("force_x", "force_y")) and abs(float(content["torque"])) <= 8.0 + EPS
    return isinstance(r, Mapping) and set(r) == keys and _integer(r.get("robot")) and 0 <= r["robot"] < ROBOT_COUNT and _integer(r.get("state_t")) and 0 <= r["state_t"] <= HORIZON and all(_finite(r.get(k)) for k in ("x", "y", "executed_vx", "executed_vy", "applied_force_x", "applied_force_y", "applied_torque", "cmd_vx", "cmd_vy", "force_x", "force_y", "torque")) and _integer(r.get("recent_broadcast_count")) and 0 <= r["recent_broadcast_count"] <= 3 and isinstance(r.get("active"), bool) and isinstance(r.get("broadcast"), bool) and obs_ok and content_ok


def _valid_estimate(r: Any) -> bool:
    keys = {"robot", "state_t", "neighbor", "force_x", "force_y", "torque", "age", "available", "fusion_error"}
    age_ok = r.get("age") is None or (_integer(r.get("age")) and 0 <= r["age"] <= 4)
    return isinstance(r, Mapping) and set(r) == keys and _integer(r.get("robot")) and 0 <= r["robot"] < ROBOT_COUNT and _integer(r.get("neighbor")) and 0 <= r["neighbor"] < ROBOT_COUNT and r["robot"] != r["neighbor"] and _integer(r.get("state_t")) and 0 <= r["state_t"] < HORIZON and all(_finite(r.get(k)) for k in ("force_x", "force_y", "torque", "fusion_error")) and float(r["fusion_error"]) >= -EPS and age_ok and isinstance(r.get("available"), bool)


def _valid_message(r: Any) -> bool:
    keys = {"broadcast_id", "receiver", "sender", "generation_slot", "arrival_state", "receivers", "content", "delivered", "used", "channel_unit"}
    ckeys = {"force_x", "force_y", "torque", "generation_slot", "sender"}; c = r.get("content") if isinstance(r, Mapping) else None
    return isinstance(r, Mapping) and set(r) == keys and _integer(r.get("broadcast_id")) and r["broadcast_id"] >= 0 and _integer(r.get("receiver")) and 0 <= r["receiver"] < ROBOT_COUNT and _integer(r.get("sender")) and 0 <= r["sender"] < ROBOT_COUNT and r["sender"] != r["receiver"] and _integer(r.get("generation_slot")) and 0 <= r["generation_slot"] < HORIZON and _integer(r.get("arrival_state")) and 1 <= r["arrival_state"] <= HORIZON + 1 and _sorted_unique_ints(r.get("receivers")) and all(0 <= x < ROBOT_COUNT for x in r["receivers"]) and isinstance(c, Mapping) and set(c) == ckeys and all(_finite(c.get(k)) for k in ("force_x", "force_y", "torque")) and _integer(c.get("generation_slot")) and c["generation_slot"] == r["generation_slot"] and _integer(c.get("sender")) and c["sender"] == r["sender"] and isinstance(r.get("delivered"), bool) and isinstance(r.get("used"), bool) and r.get("channel_unit") is True


def _valid_contact(r: Any) -> bool:
    keys = {"robot", "t", "contact_distance", "valid_force_x", "valid_force_y", "valid_torque", "contact_degree"}
    return isinstance(r, Mapping) and set(r) == keys and _integer(r.get("robot")) and 0 <= r["robot"] < ROBOT_COUNT and _integer(r.get("t")) and 0 <= r["t"] < HORIZON and all(_finite(r.get(k)) for k in ("contact_distance", "valid_force_x", "valid_force_y", "valid_torque")) and _integer(r.get("contact_degree")) and 0 <= r["contact_degree"] <= 3


def _valid_safety(r: Any) -> bool:
    keys = {"state_t", "obstacle_distance", "min_robot_distance", "collision", "payload_safe", "connected", "goal_reached"}
    return isinstance(r, Mapping) and set(r) == keys and _integer(r.get("state_t")) and 0 <= r["state_t"] <= HORIZON and all(_finite(r.get(k)) for k in ("obstacle_distance", "min_robot_distance")) and all(isinstance(r.get(k), bool) for k in ("collision", "payload_safe", "connected", "goal_reached"))


def _evidence_atoms(sc: Mapping[str, Any]) -> list[AtomVerdict]:
    names = ("payload", "robots", "estimates", "messages", "contacts", "safety")
    fields = {
        "payload": ("payload_rows", HORIZON + 1, _valid_payload),
        "robots": ("robots_rows", ROBOT_COUNT * (HORIZON + 1), _valid_robot),
        "estimates": ("estimates_rows", ROBOT_COUNT * (ROBOT_COUNT - 1) * HORIZON, _valid_estimate),
        "messages": ("messages_rows", None, _valid_message),
        "contacts": ("contacts_rows", ROBOT_COUNT * HORIZON, _valid_contact),
        "safety": ("safety_rows", HORIZON + 1, _valid_safety),
    }
    missing = [name for name in SCENARIO_NAMES if name not in sc]
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    for label in names:
        checks[label] = not missing
        details[label] = f"{label}: missing scenarios={missing!r}" if missing else f"{label} evidence passed"
        if missing:
            continue
        field, expected_count, validator = fields[label]
        for scenario in SCENARIO_NAMES:
            rows = sc[scenario].get(field) if isinstance(sc[scenario], Mapping) else None
            if not isinstance(rows, list):
                checks[label] = False
                details[label] = f"{scenario}.{label}: expected list, actual={type(rows).__name__}"
                break
            if expected_count is not None and len(rows) != expected_count:
                checks[label] = False
                details[label] = f"{scenario}.{label}: expected_length={expected_count}, actual_length={len(rows)}"
                break
            for index, row in enumerate(rows):
                try:
                    valid = validator(row)
                except Exception as exc:  # noqa: BLE001
                    valid = False
                    details[label] = f"{scenario}.{label}[{index}]: validator raised {type(exc).__name__}: {exc}"
                if not valid:
                    checks[label] = False
                    if label not in details or details[label].endswith("evidence passed"):
                        details[label] = f"{scenario}.{label}[{index}]: invalid type, range, or field set"
                    break
            if not checks[label]:
                break

    checks["types"] = not missing and all(
        isinstance(sc[name].get("record"), Mapping) and sc[name]["record"].get("scenario") == name
        for name in SCENARIO_NAMES
    )
    details["types"] = (
        f"scenario record type/path mismatch; missing scenarios={missing!r}"
        if not checks["types"]
        else "scenario record types and names passed"
    )
    return [_atom("result_evidence", key, checks[key], details[key]) for key in (*names, "types")]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []; scenarios = data.get("scenarios") if isinstance(data, Mapping) else None
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios missing"]
    groups = {n: [] for n in SCENARIO_NAMES}
    for row in scenarios:
        name = row.get("scenario") if isinstance(row, Mapping) else None
        if name in groups: groups[name].append(row)
        else: defects.append(f"unknown scenario {name}")
    def exact(rows: Any, wanted: set[Any], key_fn: Any, label: str) -> None:
        if not isinstance(rows, list): defects.append(f"{label}: missing"); return
        seen = []
        for row in rows:
            key = key_fn(row) if isinstance(row, Mapping) else None
            try:
                hash(key)
            except TypeError:
                key = None
            seen.append(key)
        defects.extend(f"{label}: missing {k}" for k in wanted if seen.count(k) == 0); defects.extend(f"{label}: duplicate {k}" for k in wanted if seen.count(k) > 1); defects.extend(f"{label}: unknown key" for k in seen if k not in wanted)
    for name in SCENARIO_NAMES:
        if len(groups[name]) != 1: defects.append(f"scenario {name}: cardinality"); continue
        row = groups[name]
        exact(row[0].get("payload"), set(range(HORIZON + 1)), lambda x: x.get("state_t"), f"{name}.payload")
        exact(row[0].get("robots"), {(i, t) for i in range(ROBOT_COUNT) for t in range(HORIZON + 1)}, lambda x: (x.get("robot"), x.get("state_t")), f"{name}.robots")
        exact(row[0].get("estimates"), {(i, t, j) for i in range(ROBOT_COUNT) for t in range(HORIZON) for j in range(ROBOT_COUNT) if i != j}, lambda x: (x.get("robot"), x.get("state_t"), x.get("neighbor")), f"{name}.estimates")
        exact(row[0].get("contacts"), {(i, t) for i in range(ROBOT_COUNT) for t in range(HORIZON)}, lambda x: (x.get("robot"), x.get("t")), f"{name}.contacts")
        exact(row[0].get("safety"), set(range(HORIZON + 1)), lambda x: x.get("state_t"), f"{name}.safety")
        message_rows = row[0].get("messages")
        if not isinstance(message_rows, list):
            defects.append(f"{name}.messages: missing")
        else:
            message_keys = []
            for index, message in enumerate(message_rows):
                if isinstance(message, Mapping) and _integer(message.get("broadcast_id")) and _integer(message.get("receiver")):
                    message_keys.append((message["broadcast_id"], message["receiver"]))
                else:
                    defects.append(f"{name}.messages[{index}]: unknown key")
            if len(message_keys) != len(set(message_keys)):
                defects.append(f"{name}.messages: duplicate key")
            if message_keys != sorted(message_keys):
                defects.append(f"{name}.messages: order")
        for field, expected, key_fn in (("payload", list(range(HORIZON + 1)), lambda x: x.get("state_t")), ("robots", [(i, t) for i in range(ROBOT_COUNT) for t in range(HORIZON + 1)], lambda x: (x.get("robot"), x.get("state_t"))), ("estimates", [(i, t, j) for i in range(ROBOT_COUNT) for t in range(HORIZON) for j in range(ROBOT_COUNT) if i != j], lambda x: (x.get("robot"), x.get("state_t"), x.get("neighbor"))), ("contacts", [(i, t) for i in range(ROBOT_COUNT) for t in range(HORIZON)], lambda x: (x.get("robot"), x.get("t"))), ("safety", list(range(HORIZON + 1)), lambda x: x.get("state_t"))):
            rows = row[0].get(field); actual = [key_fn(x) for x in rows] if isinstance(rows, list) and all(isinstance(x, Mapping) for x in rows) else []
            if actual != expected: defects.append(f"{name}.{field}: order")
    return max(0.0, 1.0 - len(defects) / 8.0), defects


def _legality_atoms(sc: Mapping[str, Any]) -> list[AtomVerdict]:
    complete = all(n in sc for n in SCENARIO_NAMES)
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    def robot_rows(name: str) -> list[Any]:
        value = sc[name].get("robots_rows", []) if name in sc and isinstance(sc[name], Mapping) else []
        return value if isinstance(value, list) else []

    rows_complete = complete and all(
        len(robot_rows(name)) == ROBOT_COUNT * (HORIZON + 1)
        and all(
            isinstance(row, Mapping)
            and _integer(row.get("robot"))
            and _integer(row.get("state_t"))
            for row in robot_rows(name)
        )
        for name in SCENARIO_NAMES
    )

    if not rows_complete:
        details["time"] = "robots rows: expected 244 typed records per scenario"
        checks["time"] = False
    else:
        checks["time"] = True
        details["time"] = "state_t domain passed"
        for name in SCENARIO_NAMES:
            for index, row in enumerate(robot_rows(name)):
                if not 0 <= row["state_t"] <= HORIZON:
                    checks["time"] = False
                    details["time"] = (
                        f"{name}.robots[{index}].state_t: expected integer in [0,{HORIZON}], "
                        f"actual={row['state_t']!r}, tolerance={EPS}"
                    )
                    break
            if not checks["time"]:
                break

    if not rows_complete:
        checks["velocity"] = False
        details["velocity"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["velocity"] = True
        details["velocity"] = "velocity norm bound passed"
        for name in SCENARIO_NAMES:
            for index, row in enumerate(robot_rows(name)):
                if row["state_t"] >= HORIZON:
                    continue
                if not (_finite(row.get("cmd_vx")) and _finite(row.get("cmd_vy"))):
                    checks["velocity"] = False
                    details["velocity"] = f"{name}.robots[{index}].cmd_vx/cmd_vy: expected finite numbers"
                    break
                norm = math.hypot(float(row["cmd_vx"]), float(row["cmd_vy"]))
                if norm > 0.3 + EPS:
                    checks["velocity"] = False
                    details["velocity"] = (
                        f"{name}.robots[{index}].cmd_vx/cmd_vy: expected_norm<=0.3, "
                        f"actual_norm={norm!r}, tolerance={EPS}"
                    )
                    break
            if not checks["velocity"]:
                break

    if not rows_complete:
        checks["force"] = False
        details["force"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["force"] = True
        details["force"] = "force and torque bounds passed"
        for name in SCENARIO_NAMES:
            for index, row in enumerate(robot_rows(name)):
                if row["state_t"] >= HORIZON:
                    continue
                if not all(_finite(row.get(field)) for field in ("force_x", "force_y", "torque")):
                    checks["force"] = False
                    details["force"] = f"{name}.robots[{index}].force_x/force_y/torque: expected finite numbers"
                    break
                force_norm = math.hypot(float(row["force_x"]), float(row["force_y"]))
                torque = abs(float(row["torque"]))
                if force_norm > 1.5 + EPS:
                    checks["force"] = False
                    details["force"] = (
                        f"{name}.robots[{index}].force_x/force_y: expected_norm<=1.5, "
                        f"actual_norm={force_norm!r}, tolerance={EPS}"
                    )
                    break
                if torque > 0.8 + EPS:
                    checks["force"] = False
                    details["force"] = (
                        f"{name}.robots[{index}].torque: expected_abs<=0.8, "
                        f"actual_abs={torque!r}, tolerance={EPS}"
                    )
                    break
            if not checks["force"]:
                break

    if not rows_complete:
        checks["failure"] = False
        details["failure"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["failure"] = True
        details["failure"] = "failure activity and zero actions passed"
        for name in SCENARIO_NAMES:
            for index, row in enumerate(robot_rows(name)):
                active = _active(name, row["robot"], row["state_t"])
                if row.get("active") != active:
                    checks["failure"] = False
                    details["failure"] = (
                        f"{name}.robots[{index}].active: expected={active!r}, "
                        f"actual={row.get('active')!r}"
                    )
                    break
                if not active:
                    for field in ("cmd_vx", "cmd_vy", "force_x", "force_y", "torque"):
                        if not _close(row.get(field), 0.0):
                            checks["failure"] = False
                            details["failure"] = (
                                f"{name}.robots[{index}].{field}: expected=0.0, "
                                f"actual={row.get(field)!r}, tolerance={EPS}"
                            )
                            break
                    if not checks["failure"]:
                        break
                    if row.get("broadcast") is not False:
                        checks["failure"] = False
                        details["failure"] = (
                            f"{name}.robots[{index}].broadcast: expected=False, "
                            f"actual={row.get('broadcast')!r}"
                        )
                        break
            if not checks["failure"]:
                break

    def channels(name: str) -> tuple[bool, str]:
        rows = robot_rows(name)
        for t in range(HORIZON):
            count = sum(
                1
                for row in rows
                if isinstance(row, Mapping) and row.get("state_t") == t and row.get("broadcast") is True
            )
            if count > 2:
                return False, f"{name}.robots[state_t={t}].broadcast: expected_count<=2, actual_count={count}, tolerance={EPS}"
        for index, row in enumerate(rows):
            if isinstance(row, Mapping) and row.get("broadcast") and row.get("active") is False:
                return False, f"{name}.robots[{index}].broadcast: expected=False for inactive robot, actual=True"
        return True, "at most two active broadcasts per slot passed"

    if not rows_complete:
        checks["channel"] = False
        details["channel"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["channel"] = True
        details["channel"] = "at most two active broadcasts per slot passed"
        for name in SCENARIO_NAMES:
            checks["channel"], details["channel"] = channels(name)
            if not checks["channel"]:
                break

    def burst(name: str) -> tuple[bool, str]:
        rows = robot_rows(name)
        for robot in range(ROBOT_COUNT):
            for start in range(HORIZON - 3):
                count = sum(
                    1
                    for row in rows
                    if isinstance(row, Mapping)
                    and row.get("robot") == robot
                    and row.get("broadcast")
                    and _integer(row.get("state_t"))
                    and start <= row["state_t"] < start + 4
                )
                if count > 3:
                    return False, (
                        f"{name}.robot={robot},window={start}..{start + 3}: "
                        f"expected_broadcast_count<=3, actual_count={count}, tolerance={EPS}"
                    )
        return True, "four-slot per-robot burst bound passed"

    if not rows_complete:
        checks["burst"] = False
        details["burst"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["burst"] = True
        details["burst"] = "four-slot per-robot burst bound passed"
        for name in SCENARIO_NAMES:
            checks["burst"], details["burst"] = burst(name)
            if not checks["burst"]:
                break

    if not rows_complete:
        checks["total"] = False
        details["total"] = "robots rows: expected 244 typed records per scenario"
    else:
        checks["total"] = True
        details["total"] = "terminal commands and executed/applied values are zero"
        terminal_fields = (
            "cmd_vx", "cmd_vy", "force_x", "force_y", "torque",
            "executed_vx", "executed_vy", "applied_force_x", "applied_force_y", "applied_torque",
        )
        for name in SCENARIO_NAMES:
            terminal = [
                row for row in robot_rows(name)
                if isinstance(row, Mapping) and row.get("state_t") == HORIZON
            ]
            if len(terminal) != ROBOT_COUNT:
                checks["total"] = False
                details["total"] = f"{name}.robots[state_t={HORIZON}]: expected_count={ROBOT_COUNT}, actual_count={len(terminal)}"
                break
            for index, row in enumerate(terminal):
                for field in terminal_fields:
                    if not _close(row.get(field), 0.0):
                        checks["total"] = False
                        details["total"] = (
                            f"{name}.robots[state_t={HORIZON}][{index}].{field}: expected=0.0, "
                            f"actual={row.get(field)!r}, tolerance={EPS}"
                        )
                        break
                if not checks["total"]:
                    break
                if row.get("broadcast") is not False:
                    checks["total"] = False
                    details["total"] = (
                        f"{name}.robots[state_t={HORIZON}][{index}].broadcast: expected=False, "
                        f"actual={row.get('broadcast')!r}"
                    )
                    break
            if not checks["total"]:
                break

    order = ("time", "velocity", "force", "failure", "channel", "burst", "total")
    return [_atom("decision_legality", key, checks[key], details[key]) for key in order]


def _replay_atoms(sc: Mapping[str, Any]) -> tuple[list[AtomVerdict], dict[str, Any]]:
    names = [key.split(".", 1)[1] for key in ATOM_SHARES["replay_validity"]]; ok = {n: True for n in names}; detail = {n: "" for n in names}; replayed: dict[str, Any] = {}
    def fail(rule: str, msg: str) -> None:
        if ok[rule]: ok[rule] = False; detail[rule] = msg
    for name in SCENARIO_NAMES:
        if name not in sc: [fail(r, f"{name}: missing scenario") for r in names]; continue
        data = sc[name]
        typed = (
            isinstance(data.get("payload_rows"), list) and len(data["payload_rows"]) == HORIZON + 1 and all(_valid_payload(r) for r in data["payload_rows"])
            and isinstance(data.get("robots_rows"), list) and len(data["robots_rows"]) == ROBOT_COUNT * (HORIZON + 1) and all(_valid_robot(r) for r in data["robots_rows"])
            and isinstance(data.get("estimates_rows"), list) and len(data["estimates_rows"]) == ROBOT_COUNT * (ROBOT_COUNT - 1) * HORIZON and all(_valid_estimate(r) for r in data["estimates_rows"])
            and isinstance(data.get("messages_rows"), list) and all(_valid_message(r) for r in data["messages_rows"])
            and isinstance(data.get("contacts_rows"), list) and len(data["contacts_rows"]) == ROBOT_COUNT * HORIZON and all(_valid_contact(r) for r in data["contacts_rows"])
            and isinstance(data.get("safety_rows"), list) and len(data["safety_rows"]) == HORIZON + 1 and all(_valid_safety(r) for r in data["safety_rows"])
        )
        if not typed:
            for rule in names:
                fail(rule, f"{name}: complete typed evidence required")
            continue
        expected = _replay_scenario(name, data); replayed[name] = expected
        for t, wanted in enumerate(expected["payload"]):
            actual = data["payload"].get(t)
            if actual is None: fail("payload", f"{name}.payload[{t}] missing"); continue
            for field in ("x", "y", "theta", "vx", "vy", "omega", "position_error", "heading_error"):
                if not _close(actual.get(field), wanted[field]):
                    fail("payload", _difference(actual.get(field), wanted[field], f"{name}.payload[{t}].{field}") or f"{name}.payload[{t}].{field}: mismatch")
                    break
            else:
                for field in ("goal_reached", "safety_violation"):
                    if actual.get(field) != wanted[field]:
                        fail("payload", f"{name}.payload[{t}].{field}: expected={wanted[field]!r}, actual={actual.get(field)!r}")
                        break
        expected_robot_map = {(r["robot"], r["state_t"]): r for r in expected["robots"]}
        for key, wanted in expected_robot_map.items():
            actual = data["robots"].get(key)
            if actual is None:
                fail("robots", f"{name}.robots{key} missing")
                continue
            for field in ("x", "y", "executed_vx", "executed_vy", "applied_force_x", "applied_force_y", "applied_torque", "cmd_vx", "cmd_vy", "force_x", "force_y", "torque"):
                if not _close(actual.get(field), wanted[field]):
                    fail("robots", _difference(actual.get(field), wanted[field], f"{name}.robots{key}.{field}") or f"{name}.robots{key}.{field}: mismatch")
                    break
            for field in ("active", "broadcast"):
                if actual.get(field) != wanted[field]:
                    fail("robots", f"{name}.robots{key}.{field}: expected={wanted[field]!r}, actual={actual.get(field)!r}")
                    break
            if actual.get("recent_broadcast_count") != wanted["recent_broadcast_count"]:
                fail("robots", f"{name}.robots{key}.recent_broadcast_count: expected={wanted['recent_broadcast_count']!r}, actual={actual.get('recent_broadcast_count')!r}")
            difference = _difference(actual.get("observation"), wanted["observation"], f"{name}.robots{key}.observation")
            if difference:
                fail("robots", difference)
            difference = _difference(actual.get("broadcast_content"), wanted["broadcast_content"], f"{name}.robots{key}.broadcast_content")
            if difference:
                fail("robots", difference)
        # Compare the independently recomputed action-start positions and applied contacts.
        for wanted in expected["contacts"]:
            actual = data["contacts"].get((wanted["robot"], wanted["t"]))
            if actual is None:
                fail("contacts", f"{name}.contacts[{wanted['robot']},{wanted['t']}]: missing")
                continue
            for field in ("contact_distance", "valid_force_x", "valid_force_y", "valid_torque"):
                if not _close(actual.get(field), wanted[field]):
                    fail("contacts", _difference(actual.get(field), wanted[field], f"{name}.contacts[{wanted['robot']},{wanted['t']}].{field}") or f"{name}.contacts[{wanted['robot']},{wanted['t']}].{field}: mismatch")
                    break
            else:
                if actual.get("contact_degree") != wanted["contact_degree"]:
                    fail("contacts", f"{name}.contacts[{wanted['robot']},{wanted['t']}].contact_degree: expected={wanted['contact_degree']!r}, actual={actual.get('contact_degree')!r}")
        for t, wanted in enumerate(expected["safety"]):
            actual = data["safety"].get(t)
            if actual is None:
                fail("safety", f"{name}.safety[{t}]: missing")
                continue
            for field in ("obstacle_distance", "min_robot_distance"):
                if not _close(actual.get(field), wanted[field]):
                    fail("safety", _difference(actual.get(field), wanted[field], f"{name}.safety[{t}].{field}") or f"{name}.safety[{t}].{field}: mismatch")
                    break
            else:
                for field in ("collision", "payload_safe", "connected", "goal_reached"):
                    if actual.get(field) != wanted[field]:
                        fail("safety", f"{name}.safety[{t}].{field}: expected={wanted[field]!r}, actual={actual.get(field)!r}")
                        break
        # Communication and estimate cache semantics are checked against each generated message.
        msg_rows = data["messages_rows"]
        expected_messages = {(m["broadcast_id"], m["receiver"]): m for m in expected["messages"]}
        actual_messages = {(m.get("broadcast_id"), m.get("receiver")): m for m in msg_rows if isinstance(m, Mapping)}
        if set(actual_messages) != set(expected_messages):
            fail("messages", f"{name}.messages: expected_keys={sorted(expected_messages, key=repr)!r}, actual_keys={sorted(actual_messages, key=repr)!r}")
        for key, wanted in expected_messages.items():
            m = actual_messages.get(key)
            if m is None:
                continue
            for field in ("sender", "generation_slot", "arrival_state", "receivers", "delivered", "used", "channel_unit"):
                if m.get(field) != wanted[field]:
                    if _finite(wanted[field]):
                        fail("messages", _difference(m.get(field), wanted[field], f"{name}.message{key}.{field}") or f"{name}.message{key}.{field}: mismatch")
                    else:
                        fail("messages", f"{name}.message{key}.{field}: expected={wanted[field]!r}, actual={m.get(field)!r}")
                    break
            difference = _difference(m.get("content"), wanted["content"], f"{name}.message{key}.content")
            if difference:
                fail("messages", difference)
        for m in msg_rows:
            if not _valid_message(m): fail("messages", f"{name}.messages: invalid message fields/type at broadcast_id={m.get('broadcast_id') if isinstance(m, Mapping) else None!r}"); continue
            expected_delay = 2 if _hash(name, m["generation_slot"], m["sender"], m["receiver"], 0) < SCENARIOS[name]["q"] else 1
            expected_delivered = m["arrival_state"] <= HORIZON and _hash(name, m["generation_slot"], m["sender"], m["receiver"], 1) >= SCENARIOS[name]["ell"]
            expected_arrival = m["generation_slot"] + expected_delay
            if m["arrival_state"] != expected_arrival:
                fail("communication", f"{name}.message[{m.get('broadcast_id')},{m.get('receiver')}].arrival_state: expected={expected_arrival!r}, actual={m['arrival_state']!r}, tolerance={EPS}")
            elif m["delivered"] != expected_delivered:
                fail("communication", f"{name}.message[{m.get('broadcast_id')},{m.get('receiver')}].delivered: expected={expected_delivered!r}, actual={m['delivered']!r}")
            elif m["receiver"] not in m["receivers"]:
                fail("communication", f"{name}.message[{m.get('broadcast_id')},{m.get('receiver')}].receivers: expected receiver membership, actual={m['receivers']!r}")
        for er in data["estimates_rows"]:
            if not _valid_estimate(er): fail("estimates", f"{name}.estimates: invalid estimate fields/type at key={er.get('robot'), er.get('state_t'), er.get('neighbor') if isinstance(er, Mapping) else None!r}"); continue
            if er["age"] is not None and not er["available"]: fail("estimates", f"{name}.estimate{(er['robot'], er['state_t'], er['neighbor'])}.age: expected=None when available=False, actual={er['age']!r}")
            if er["age"] is None and er["available"]: fail("estimates", f"{name}.estimate{(er['robot'], er['state_t'], er['neighbor'])}.age: expected integer when available=True, actual=None")
        expected_estimates = {(r["robot"], r["state_t"], r["neighbor"]): r for r in expected["estimates"]}
        actual_estimates = {(r.get("robot"), r.get("state_t"), r.get("neighbor")): r for r in data["estimates_rows"] if isinstance(r, Mapping)}
        if set(actual_estimates) != set(expected_estimates):
            fail("estimates", f"{name}.estimates: expected_keys={sorted(expected_estimates, key=repr)!r}, actual_keys={sorted(actual_estimates, key=repr)!r}")
        for key, wanted in expected_estimates.items():
            actual = actual_estimates.get(key)
            if actual is None:
                continue
            difference = _difference(actual, wanted, f"{name}.estimate{key}")
            if difference:
                fail("estimates", difference)
        replayed[name]["robots"] = expected.get("robots", [])
    return [_atom("replay_validity", n, ok[n], detail[n] or "independent replay passed") for n in names], replayed


def _metrics(replayed: Mapping[str, Any], sc: Mapping[str, Any]) -> dict[str, float]:
    p_scores = []; completion = []; comm = []; freshness = []
    for name in SCENARIO_NAMES:
        exp = replayed[name]; payload = exp["payload"]; final = payload[-1]; safe = bool(final["goal_reached"] and not final["safety_violation"]); completion.append(1.0 if safe else 0.0)
        apos = sum(float(r["position_error"]) ** 2 for r in payload[1:]); ayaw = sum(float(r["heading_error"]) ** 2 for r in payload[1:]); p_scores.append(0.7 * max(0.0, min(1.0, 1.0 - apos / (HORIZON * (8 * math.sqrt(2)) ** 2))) + 0.3 * max(0.0, min(1.0, 1.0 - ayaw / (HORIZON * math.pi ** 2))))
        broadcast_count = sum(1 for row in sc[name]["robots_rows"] if row["state_t"] < HORIZON and row["broadcast"])
        comm.append(max(0.0, min(1.0, 1.0 - broadcast_count / (HORIZON * 2))))
        age_scores = [max(0.0, min(1.0, 1.0 - float(row["age"]) / 4.0)) if row["available"] else 0.0 for row in exp["estimates"]]
        graph = [float(s["connected"]) for s in exp["safety"][:-1]]
        freshness.append(0.6 * (sum(age_scores) / (ROBOT_COUNT * (ROBOT_COUNT - 1) * HORIZON)) + 0.4 * (sum(graph) / HORIZON))
    return {"primary_metrics_1": sum(p_scores) / 5.0, "primary_metrics_2": sum(completion) / 5.0, "auxiliary_metrics_1": sum(comm) / 5.0, "auxiliary_metrics_2": sum(freshness) / 5.0}


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected): raise ValueError("atom IDs mismatch")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9): raise ValueError("invalid shares")
    if any(not math.isclose(a.share, expected[a.id], abs_tol=1e-12) for a in atoms): raise ValueError("atom share mismatch")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [a for a in atoms if not a.passed]
    return "failed_atoms=[]" if not failed else "failed_atoms=[" + ",".join(a.id for a in failed) + "]; " + " | ".join(f"{a.id}: {a.detail}" for a in failed)


def _headings(text: str) -> dict[str, str]:
    """Parse visible Markdown sections without treating examples as headings.

    Reports are candidate-authored Markdown, so headings may be indented by the
    usual Markdown allowance, use closing ``#`` markers, or use setext syntax.
    Fenced examples and HTML comments are not report sections.  The returned
    mapping keeps the first non-empty body when a title is repeated, which makes
    duplicate headings unable to inflate the normalized score.
    """
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
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    try:
        if not _regular_file(ws, "report.md"):
            return 0.0, "report.md missing or not a regular file"
        text = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(text)
    missing = [title for title in required if not headings.get(title)]
    score = max(0.0, 2.0 - 0.5 * len(missing)) / 2.0
    return score, f"missing_sections={missing}; normalized={score:.9g}"


def _regular_file(ws: Workspace, rel: str) -> bool:
    p = Path(ws.root) / rel; return p.is_file() and not p.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE: return _CACHE[key]
    state = EvalState(); started = time.monotonic(); remove_ok = False
    try:
        state.run_started = True
        try: remove_ok = getattr(await ws.run("rm -f -- results.json", check=False, timeout=30.0), "exit_code", None) == 0
        except Exception as exc: state.official_message = f"could not remove old results.json: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0); state.exit_code = getattr(result, "exit_code", None); state.timed_out = state.exit_code is None
        except Exception as exc: state.official_message = f"official run raised {type(exc).__name__}: {exc}"; state.timed_out = isinstance(exc, TimeoutError)
        state.elapsed_s = time.monotonic() - started; state.run_finished = True; state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json"); state.official_ok = remove_ok and state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600 + EPS and state.result_file_is_new_regular; state.official_message = state.official_message or f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, new_regular_results={state.result_file_is_new_regular}"
        if state.result_file_is_new_regular:
            try: state.data = _strict_json(ws.read_text("results.json") or ""); state.parse_ok = True
            except Exception as exc: state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        ra = _result_json_atoms(state.data, state.parse_ok); state.scores["result_json"] = _score_atoms(ra, ATOM_SHARES["result_json"]); state.diagnostics["result_json"] = _diagnose(ra)
        if not state.official_ok or not state.parse_ok:
            for n in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"): state.scores[n] = 0.0; state.diagnostics[n] = "official run or strict parse failed"
            _CACHE[key] = state; return state
        state.scenarios = _index_records(state.data); ea = _evidence_atoms(state.scenarios); state.scores["result_evidence"] = _score_atoms(ea, ATOM_SHARES["result_evidence"]); state.diagnostics["result_evidence"] = _diagnose(ea); cov, defects = _coverage_analysis(state.data); state.scores["canonical_coverage"] = cov; state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:20]}"; la = _legality_atoms(state.scenarios); state.scores["decision_legality"] = _score_atoms(la, ATOM_SHARES["decision_legality"]); state.diagnostics["decision_legality"] = _diagnose(la); rp, replayed = _replay_atoms(state.scenarios); state.replayed = replayed; state.scores["replay_validity"] = _score_atoms(rp, ATOM_SHARES["replay_validity"]); state.diagnostics["replay_validity"] = _diagnose(rp)
        gate = ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"); state.validity_gate = state.official_ok and all(state.scores.get(n, 0.0) >= 1 - EPS for n in gate)
        if state.validity_gate: state.metrics = _metrics(state.replayed, state.scenarios)
    except Exception as exc:
        state.official_ok = False; state.official_message = state.official_message or f"grader state failed: {type(exc).__name__}: {exc}"
        for n in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"): state.scores.setdefault(n, 0.0); state.diagnostics.setdefault(n, state.official_message)
    _CACHE[key] = state; return state


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root = isinstance(data, Mapping); scenarios = root and isinstance(data.get("scenarios"), list); names = scenarios and all(isinstance(r, Mapping) and isinstance(r.get("scenario"), str) for r in data["scenarios"]); keys = scenarios and all(_scenario_shape(r) for r in data["scenarios"])
    return [_atom("result_json", "parse", parsed, "strict JSON parse"), _atom("result_json", "root", root and set(data) == {"scenarios"}, "root key set"), _atom("result_json", "scenarios", scenarios, "scenarios list"), _atom("result_json", "scenario_keys", bool(names and keys), "scenario record keys")]


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"): return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0); return (1.0, "static compilation passed") if getattr(result, "exit_code", None) == 0 else (0.0, "static compilation failed")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws); return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if name.startswith(("primary_metrics_", "auxiliary_metrics_")) and state.metrics is None: return 0.0, f"{name}: validity gate not satisfied"
    return state.scores.get(name, state.metrics.get(name, 0.0) if state.metrics else 0.0), state.diagnostics.get(name, f"{name}: independently replayed value={state.metrics.get(name, 0.0) if state.metrics else 0.0:.9g}")


for _id in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity", "primary_metrics_1", "primary_metrics_2", "auxiliary_metrics_1", "auxiliary_metrics_2"):
    async def _check(ws: Workspace, _name: str = _id) -> tuple[float, str]:
        return await _score(ws, _name)
    globals()[_id] = check(id=_id)(_check)
del _check, _id
