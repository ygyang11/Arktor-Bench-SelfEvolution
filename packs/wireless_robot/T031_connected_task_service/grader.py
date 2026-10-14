# ruff: noqa: E501
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

HORIZON = 100
GRID_X = 20
GRID_Y = 14
ROBOTS = 4
REQUESTS = 8
ENERGY0 = 120.0
TEAM_BUDGET = 160
RELAY_CAPACITY = 4
COMM_RADIUS = 4
EPS = 1e-6
SCENARIO_NAMES = (
    "staggered_requests",
    "narrow_bridge",
    "late_priority",
    "moving_obstacles",
    "range_partition",
)
COMMS = ("none", "new", "retry", "forward")
ACTION_KEYS = frozenset(
    {
        "t",
        "robot",
        "dx",
        "dy",
        "release_request",
        "comm",
        "packet_id",
        "proposal_request",
        "proposal_cost",
        "relay",
    }
)
MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
REQUESTS_TEMPLATE = (
    (3, 2, 4, 4, 34),
    (6, 11, 6, 3, 46),
    (11, 3, 5, 5, 52),
    (15, 10, 7, 2, 60),
    (4, 8, 4, 6, 68),
    (8, 12, 6, 4, 76),
    (14, 4, 5, 7, 84),
    (17, 11, 8, 3, 94),
)
SCENARIOS: dict[str, dict[str, Any]] = {
    "staggered_requests": {
        "starts": ((1, 1), (3, 1), (5, 1), (7, 1)),
        "dispatcher": (0, 1),
        "obstacles": ((8, 9, 3, 10),),
        "releases": (0, 8, 16, 24, 32, 40, 48, 56),
        "perm": (0, 1, 2, 3, 4, 5, 6, 7),
        "moving": False,
    },
    "narrow_bridge": {
        "starts": ((1, 6), (3, 6), (5, 6), (7, 6)),
        "dispatcher": (0, 6),
        "obstacles": ((9, 10, 0, 5), (9, 10, 8, 13)),
        "releases": (0, 4, 8, 12, 32, 36, 40, 44),
        "perm": (4, 5, 6, 7, 0, 1, 2, 3),
        "moving": False,
    },
    "late_priority": {
        "starts": ((6, 3), (8, 3), (10, 3), (12, 3)),
        "dispatcher": (5, 3),
        "obstacles": ((6, 13, 6, 7),),
        "releases": (0, 2, 4, 6, 54, 60, 66, 72),
        "perm": (0, 4, 1, 5, 2, 6, 3, 7),
        "moving": False,
    },
    "moving_obstacles": {
        "starts": ((2, 6), (4, 6), (6, 6), (8, 6)),
        "dispatcher": (0, 6),
        "obstacles": ((7, 8, 3, 5), (11, 12, 8, 10)),
        "releases": (0, 10, 20, 30, 40, 50, 60, 70),
        "perm": (1, 2, 3, 4, 5, 6, 7, 0),
        "moving": True,
    },
    "range_partition": {
        "starts": ((2, 6), (4, 6), (6, 6), (8, 6)),
        "dispatcher": (0, 6),
        "obstacles": ((9, 10, 5, 8),),
        "releases": (0, 6, 12, 18, 30, 36, 42, 48),
        "perm": (0, 2, 4, 6, 1, 3, 5, 7),
        "moving": False,
    },
}
ATOM_SHARES = {
    "result_json": {f"result_json.{x}": 0.25 for x in ("parse", "root", "scenarios", "actions")},
    "result_evidence": {f"result_evidence.{x}": 0.25 for x in ("rows", "fields", "ids", "types")},
    "decision_legality": {
        f"decision_legality.{x}": 0.2
        for x in ("motion", "release", "packets", "proposal_cost", "resources")
    },
    "replay_validity": {
        f"replay_validity.{x}": 0.2
        for x in ("motion", "requests", "packets", "services", "metrics")
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
        self.id = ident
        self.passed = bool(passed)
        self.detail = str(detail)
        self.share = float(share)


class EvalState:
    def __init__(self) -> None:
        self.run_started = False
        self.run_finished = False
        self.timed_out = False
        self.exit_code: int | None = None
        self.elapsed_s: float | None = None
        self.official_ok = False
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.parse_ok = False
        self.data: Any = None
        self.scenarios: dict[str, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _brief(value: Any, limit: int = 120) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _norm(value: Any) -> int:
    return -1 if value is None or value == -1 else int(value)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    ident = f"{group}.{name}"
    return AtomVerdict(ident, passed, detail, ATOM_SHARES[group][ident])


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique)


def _blocked(
    spec: Mapping[str, Any],
    x: int,
    y: int,
    t: int,
    obstacles: Sequence[Sequence[int]] | None = None,
) -> bool:
    rects = obstacles if obstacles is not None else _obstacles(spec, t)
    for xmin, xmax, ymin, ymax in rects:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False


def _obstacles(spec: Mapping[str, Any], t: int) -> tuple[tuple[int, int, int, int], ...]:
    obstacles = list(spec["obstacles"])
    if spec.get("moving"):
        shifts = min(4, max(0, (t - 40) // 4 + 1)) if t >= 40 else 0
        xmin, xmax, ymin, ymax = obstacles[1]
        obstacles[1] = (xmin + shifts, xmax + shifts, ymin, ymax)
    return tuple(obstacles)


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


def _edge(
    spec: Mapping[str, Any],
    a: Sequence[int],
    b: Sequence[int],
    t: int,
    obstacles: Sequence[Sequence[int]] | None = None,
) -> bool:
    d = abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
    rects = obstacles if obstacles is not None else _obstacles(spec, t)
    return d <= COMM_RADIUS and not any(_line_hit(a, b, rect) for rect in rects)


def _graph_connected(
    spec: Mapping[str, Any],
    positions: Sequence[Sequence[int]],
    t: int,
    obstacles: Sequence[Sequence[int]] | None = None,
) -> tuple[bool, float]:
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(ROBOTS):
            if j not in seen and _edge(spec, positions[i], positions[j], t, obstacles):
                seen.add(j)
                stack.append(j)
    margins = []
    for i in range(ROBOTS):
        for j in range(i + 1, ROBOTS):
            if _edge(spec, positions[i], positions[j], t, obstacles):
                d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                margins.append((COMM_RADIUS - d) / COMM_RADIUS)
    return len(seen) == ROBOTS, min(margins, default=0.0) if len(seen) == ROBOTS else 0.0


def _request_spec(name: str, request: int) -> dict[str, Any]:
    spec = SCENARIOS[name]
    x, y, service, value, deadline = REQUESTS_TEMPLATE[request]
    p = spec["perm"][request]
    px, py = REQUESTS_TEMPLATE[p][:2]
    return {
        "request": request,
        "x": px,
        "y": py,
        "service": service,
        "value": value,
        "deadline": deadline,
        "release": spec["releases"][request],
    }


def _action_shape(action: Any) -> bool:
    return (
        isinstance(action, Mapping)
        and set(action) == ACTION_KEYS
        and _integer(action.get("t"))
        and _integer(action.get("robot"))
        and _integer(action.get("dx"))
        and _integer(action.get("dy"))
        and (_integer(action.get("release_request")) or action.get("release_request") is None)
        and isinstance(action.get("comm"), str)
        and (_integer(action.get("packet_id")) or action.get("packet_id") is None)
        and (_integer(action.get("proposal_request")) or action.get("proposal_request") is None)
        and (_finite(action.get("proposal_cost")) or action.get("proposal_cost") is None)
        and (_integer(action.get("relay")) or action.get("relay") is None)
    )


def _shape_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    root = isinstance(data, Mapping) and set(data) == {"scenarios"}
    scenarios = isinstance(data, Mapping) and isinstance(data.get("scenarios"), list)
    records = scenarios and all(
        isinstance(item, Mapping)
        and set(item) == {"scenario", "actions"}
        and isinstance(item.get("scenario"), str)
        and isinstance(item.get("actions"), list)
        for item in data["scenarios"]
    )
    return [
        _atom("result_json", "parse", parse_ok, f"strict_parse_ok={parse_ok}"),
        _atom(
            "result_json",
            "root",
            root,
            f"actual_keys={sorted(data) if isinstance(data, Mapping) else None}; expected=['scenarios']",
        ),
        _atom("result_json", "scenarios", scenarios, "scenarios is a list"),
        _atom("result_json", "actions", records, "scenario records have exact containers"),
    ]


def _index(data: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return {}
    return {
        str(item["scenario"]): item
        for item in data["scenarios"]
        if isinstance(item, Mapping) and isinstance(item.get("scenario"), str)
    }


def _evidence_atoms(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    rows = fields = ids = types = True
    defects: dict[str, list[str]] = {key: [] for key in ("rows", "fields", "ids", "types")}

    def note(group: str, path: str, actual: Any, expected: str) -> None:
        if len(defects[group]) < 8:
            defects[group].append(f"{path}: actual={_brief(actual)}; expected={expected}")

    for name in SCENARIO_NAMES:
        acts = (
            scenarios.get(name, {}).get("actions")
            if isinstance(scenarios.get(name), Mapping)
            else None
        )
        if not isinstance(acts, list):
            rows = False
            note("rows", f"{name}.actions", acts, "list")
            continue
        for index, action in enumerate(acts):
            path = f"{name}.actions[{index}]"
            if not isinstance(action, Mapping):
                rows = False
                note("rows", path, action, "object")
                continue
            if set(action) != ACTION_KEYS:
                fields = False
                note("fields", path, sorted(action), f"exact keys {sorted(ACTION_KEYS)}")
            for key in ("t", "robot"):
                if key not in action:
                    continue
                if not _integer(action.get(key)):
                    ids = False
                    note("ids", f"{path}.{key}", action.get(key), "integer")
            checks = (
                ("dx", _integer(action.get("dx")), "integer"),
                ("dy", _integer(action.get("dy")), "integer"),
                (
                    "release_request",
                    _integer(action.get("release_request"))
                    or action.get("release_request") is None,
                    "integer or null",
                ),
                ("comm", isinstance(action.get("comm"), str), "string"),
                (
                    "packet_id",
                    _integer(action.get("packet_id")) or action.get("packet_id") is None,
                    "integer or null",
                ),
                (
                    "proposal_request",
                    _integer(action.get("proposal_request"))
                    or action.get("proposal_request") is None,
                    "integer or null",
                ),
                (
                    "proposal_cost",
                    _finite(action.get("proposal_cost")) or action.get("proposal_cost") is None,
                    "finite number or null",
                ),
                (
                    "relay",
                    _integer(action.get("relay")) or action.get("relay") is None,
                    "integer or null",
                ),
            )
            for key, valid, expected in checks:
                if key not in action:
                    continue
                if not valid:
                    types = False
                    note("types", f"{path}.{key}", action.get(key), expected)
    return [
        _atom("result_evidence", "rows", rows, f"defects={defects['rows']}"),
        _atom("result_evidence", "fields", fields, f"defects={defects['fields']}"),
        _atom("result_evidence", "ids", ids, f"defects={defects['ids']}"),
        _atom("result_evidence", "types", types, f"defects={defects['types']}"),
    ]


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    records = data.get("scenarios", []) if isinstance(data, Mapping) else []
    if not isinstance(records, list):
        return 0.0, ["scenarios: not a list"]
    scenario_counts: dict[Any, int] = {}
    actions: dict[tuple[Any, Any, Any], int] = {}
    for item in records:
        if not isinstance(item, Mapping):
            defects.append("unknown scenario record")
            continue
        name = item.get("scenario")
        if not isinstance(name, str):
            name = f"<invalid:{type(name).__name__}:{name!r}>"
        scenario_counts[name] = scenario_counts.get(name, 0) + 1
        for action in item.get("actions", []) if isinstance(item.get("actions"), list) else []:
            if isinstance(action, Mapping):
                t_value = action.get("t")
                robot_value = action.get("robot")
                if not _integer(t_value):
                    t_value = f"<invalid:{type(t_value).__name__}:{_brief(t_value)}>"
                if not _integer(robot_value):
                    robot_value = f"<invalid:{type(robot_value).__name__}:{_brief(robot_value)}>"
                key = (name, t_value, robot_value)
                actions[key] = actions.get(key, 0) + 1
    for name in SCENARIO_NAMES:
        count = scenario_counts.get(name, 0)
        if count == 0:
            defects.append(f"scenario missing:{name}")
        defects.extend([f"scenario duplicate:{name}"] * max(0, count - 1))
    for name, count in scenario_counts.items():
        if name not in SCENARIO_NAMES:
            defects.extend([f"scenario unknown:{name}"] * count)
    expected = {
        (name, t, robot)
        for name in SCENARIO_NAMES
        for t in range(HORIZON)
        for robot in range(ROBOTS)
    }
    for key in expected:
        count = actions.get(key, 0)
        if count == 0:
            defects.append(f"action missing:{key}")
        defects.extend([f"action duplicate:{key}"] * max(0, count - 1))
    for key, count in actions.items():
        if key not in expected:
            defects.extend([f"action unknown:{key}"] * count)
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _packet_probability(
    name: str,
    spec: Mapping[str, Any],
    a: Sequence[int],
    b: Sequence[int],
    t: int,
    n: int,
    obstacles: Sequence[Sequence[int]] | None = None,
) -> float:
    d = abs(a[0] - b[0]) + abs(a[1] - b[1])
    rects = obstacles if obstacles is not None else _obstacles(spec, t)
    obstacle = int(any(_line_hit(a, b, rect) for rect in rects))
    return _clip(
        max(0.1, min(0.92, 0.92 - 0.06 * (d / 4) - 0.08 * obstacle - 0.04 * max(0, n - 1)))
    )


def _hop_eligible(a: Sequence[int], b: Sequence[int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= COMM_RADIUS


def _packet_success(
    name: str,
    t: int,
    source: int,
    request: int,
    epoch: int,
    hop: int,
    attempt: int,
    p: float,
) -> bool:
    key = f"{name}|{t}|{source}|{request}|{epoch}|{hop}|{attempt}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    u = int(digest, 16) / 0xFFFFFFFF
    return u <= p


def _replay(name: str, scenario: Mapping[str, Any]) -> dict[str, Any]:
    spec = SCENARIOS[name]
    actions = scenario.get("actions", []) if isinstance(scenario, Mapping) else []
    by_t: dict[int, dict[int, Mapping[str, Any]]] = {
        t: {
            int(action["robot"]): action
            for action in actions
            if isinstance(action, Mapping)
            and action.get("t") == t
            and _integer(action.get("robot"))
        }
        for t in range(HORIZON)
    }
    positions = [list(item) for item in spec["starts"]]
    obstacles = [tuple(item) for item in spec["obstacles"]]
    energy = [ENERGY0] * ROBOTS
    assignments: list[int | None] = [None] * REQUESTS
    request_state = ["hidden"] * REQUESTS
    release_epochs = [0] * REQUESTS
    epoch_started = [None] * REQUESTS
    proposals: dict[tuple[int, int, int], dict[str, Any]] = {}
    packets: dict[int, dict[str, Any]] = {}
    queues: list[list[int]] = [[] for _ in range(ROBOTS)]
    pending: dict[int, list[tuple[int, int]]] = {}
    failed: dict[int, tuple[int, int, int]] = {}
    retry_used: set[tuple[int, int]] = set()
    services: list[dict[str, Any] | None] = [None] * REQUESTS
    service_progress = [0] * REQUESTS
    blocked_count = 0
    deadlock_events = 0
    last_deadlock_slot: int | None = None
    safety_events = 0
    connected_slots = 0
    margins: list[float] = []
    accepted_units = 0
    useful_units = 0
    complete_values = 0.0
    action_rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    packet_arrivals: list[dict[str, Any]] = []
    assignment_events: list[dict[str, int]] = []
    deadlock_slots: list[int] = []
    blocked_trace: list[dict[str, int | bool]] = []
    states: list[dict[str, Any]] = []
    decision_states: list[dict[str, Any]] = []
    defaults = {
        "dx": 0,
        "dy": 0,
        "release_request": None,
        "comm": "none",
        "packet_id": None,
        "proposal_request": None,
        "proposal_cost": None,
        "relay": None,
    }

    def req_key(pid: int) -> tuple[int, int, int, int, int]:
        packet = packets[pid]
        req = _request_spec(name, packet["request"])
        return (
            req["deadline"],
            packet["epoch_start"],
            packet["generation_slot"],
            packet["src"],
            pid,
        )

    def apply_arrivals(t: int) -> None:
        nonlocal useful_units
        relay_arrivals: list[list[int]] = [[] for _ in range(ROBOTS)]
        for pid, hop in pending.pop(t, []):
            packet = packets.get(pid)
            if not packet or (hop, t) in packet["applied"]:
                continue
            packet["applied"].add((hop, t))
            if hop == 0 and packet["relay"] is not None:
                relay_arrivals[packet["relay"]].append(pid)
                packet["stage"] = "queued"
                continue
            packet["stage"] = "delivered"
            packet["arrival_state"] = t
            packet_arrivals.append({"pid": pid, "state": t})
            if packet["class"] == "proposal":
                req = packet["request"]
                if (
                    req < REQUESTS
                    and packet["epoch"] == release_epochs[req]
                    and assignments[req] is None
                ):
                    key = (packet["src"], req, packet["epoch"])
                    previous = proposals.get(key)
                    if previous is None or (packet["generation_slot"], packet["packet_id"]) > (
                        previous["generation_slot"],
                        previous["packet_id"],
                    ):
                        proposals[key] = packet
        for relay, arrivals in enumerate(relay_arrivals):
            arrivals.sort(key=req_key)
            room = max(0, RELAY_CAPACITY - len(queues[relay]))
            queues[relay].extend(arrivals[:room])
            for pid in arrivals[room:]:
                packets[pid]["stage"] = "dropped"

    def discard_expired_failures(t: int) -> None:
        for pid, (failed_slot, hop, _holder) in list(failed.items()):
            retried = (pid, hop) in retry_used
            if failed_slot < t - 1 or (retried and failed_slot == t - 1):
                packets[pid]["stage"] = "discarded"
                relay = packets[pid]["relay"]
                if relay is not None and pid in queues[relay]:
                    queues[relay].remove(pid)
                failed.pop(pid, None)

    def robot_busy(robot: int) -> bool:
        return any(
            assignments[req] == robot and request_state[req] in ("assigned", "serving")
            for req in range(REQUESTS)
        )

    def form_assignments(t: int) -> None:
        nonlocal useful_units
        for req in range(REQUESTS):
            if (
                spec["releases"][req] > t
                or assignments[req] is not None
                or request_state[req] == "completed"
            ):
                continue
            if epoch_started[req] is None:
                epoch_started[req] = spec["releases"][req]
            candidates = [
                packet
                for (robot, r, epoch), packet in proposals.items()
                if r == req
                and epoch == release_epochs[req]
                and packet["stage"] == "delivered"
                and not robot_busy(robot)
            ]
            if not candidates:
                continue
            if (
                len({packet["src"] for packet in candidates}) < ROBOTS
                and t < epoch_started[req] + 4
            ):
                continue
            winner = min(
                candidates,
                key=lambda packet: (
                    packet["cost"],
                    packet["generation_slot"],
                    packet["src"],
                    packet["packet_id"],
                ),
            )
            assignments[req] = winner["src"]
            request_state[req] = "assigned"
            useful_units += 1
            assignment_events.append(
                {
                    "state": t,
                    "request": req,
                    "robot": winner["src"],
                    "packet_id": winner["packet_id"],
                }
            )

    for t in range(HORIZON):
        completed_this_slot = False
        release_window = last_deadlock_slot == t - 1
        discard_expired_failures(t)
        apply_arrivals(t)
        form_assignments(t)
        for req in range(REQUESTS):
            if spec["releases"][req] == t:
                request_state[req] = "released"
                epoch_started[req] = t
        decision_states.append(
            {
                "positions": [tuple(item) for item in positions],
                "energy": list(energy),
                "obstacles": [tuple(item) for item in obstacles],
                "assignments": list(assignments),
                "request_state": list(request_state),
                "release_epochs": list(release_epochs),
                "epoch_started": list(epoch_started),
                "queues": [list(queue) for queue in queues],
                "packets": {
                    pid: {
                        "src": packet["src"],
                        "relay": packet["relay"],
                        "request": packet["request"],
                        "epoch": packet["epoch"],
                        "stage": packet["stage"],
                        "failure": failed.get(pid),
                        "first_attempt": packet["first_attempt"],
                        "retry_used": failed.get(pid) is not None
                        and (pid, failed[pid][1]) in retry_used,
                    }
                    for pid, packet in packets.items()
                },
                "services": [dict(item) if item is not None else None for item in services],
                "last_deadlock_slot": last_deadlock_slot,
            }
        )
        current = {r: {**defaults, **by_t[t].get(r, {})} for r in range(ROBOTS)}
        for r in range(ROBOTS):
            if not _action_shape(current[r]):
                invalid.append(f"shape:{t}:{r}")
        proposed = [list(position) for position in positions]
        for r in range(ROBOTS):
            action = current[r]
            nx, ny = positions[r][0] + action["dx"], positions[r][1] + action["dy"]
            bad = not (0 <= nx < GRID_X and 0 <= ny < GRID_Y) or _blocked(
                spec, nx, ny, t, obstacles
            )
            if not bad:
                for j in range(r):
                    if [nx, ny] == proposed[j] or (
                        [nx, ny] == positions[r] and proposed[j] == positions[r]
                    ):
                        bad = True
                for j in range(r + 1, ROBOTS):
                    if [nx, ny] == positions[j]:
                        bad = True
            if (
                robot_busy(r)
                and any(
                    assignments[req] == r and request_state[req] == "serving"
                    for req in range(REQUESTS)
                )
                and (nx, ny) != tuple(positions[r])
            ):
                invalid.append(f"serving_move:{t}:{r}")
                bad = True
            if bad:
                if (action["dx"], action["dy"]) != (0, 0):
                    safety_events += 1
                proposed[r] = list(positions[r])
            else:
                proposed[r] = [nx, ny]
        connected, margin = _graph_connected(spec, proposed, t, obstacles)
        if not connected:
            for r in range(ROBOTS):
                if proposed[r] != positions[r]:
                    proposed[r] = list(positions[r])
                    safety_events += 1
            connected, margin = _graph_connected(spec, proposed, t, obstacles)
        submissions: list[dict[str, Any]] = []
        for r, action in current.items():
            release = _norm(action["release_request"])
            if release >= 0:
                legal_release = (
                    release_window
                    and release < REQUESTS
                    and assignments[release] == r
                    and request_state[release] == "assigned"
                )
                if not legal_release:
                    invalid.append(f"release:{t}:{r}")
            comm = action["comm"]
            pid = _norm(action["packet_id"])
            relay = _norm(action["relay"])
            proposal_req = _norm(action["proposal_request"])
            if comm == "none":
                if (
                    pid >= 0
                    or relay >= 0
                    or proposal_req >= 0
                    or action["proposal_cost"] is not None
                ):
                    invalid.append(f"packet_shape:{t}:{r}")
            elif comm == "new":
                cost = action["proposal_cost"]
                packet_bad = (
                    pid < 0
                    or pid in packets
                    or proposal_req < 0
                    or proposal_req >= REQUESTS
                    or spec["releases"][proposal_req] > t
                    or request_state[proposal_req] != "released"
                    or assignments[proposal_req] is not None
                    or robot_busy(r)
                    or (relay >= 0 and (relay == r or relay >= ROBOTS))
                )
                cost_bad = not _finite(cost) or not 0 <= float(cost) <= 1
                if packet_bad:
                    invalid.append(f"packet:{t}:{r}")
                if cost_bad:
                    invalid.append(f"proposal_cost:{t}:{r}")
                if not packet_bad and not cost_bad:
                    packet = {
                        "packet_id": pid,
                        "class": "proposal",
                        "request": proposal_req,
                        "src": r,
                        "relay": None if relay < 0 else relay,
                        "generation_slot": t,
                        "epoch": release_epochs[proposal_req],
                        "epoch_start": epoch_started[proposal_req],
                        "cost": float(cost),
                        "stage": "pending",
                        "applied": set(),
                        "arrival_state": None,
                        "first_attempt": True,
                    }
                    packets[pid] = packet
                    submissions.append(
                        {"r": r, "pid": pid, "hop": 0, "relay": packet["relay"], "packet": packet}
                    )
            elif comm in ("retry", "forward"):
                if (
                    proposal_req >= 0
                    or action["proposal_cost"] is not None
                    or relay >= 0
                    or pid < 0
                ):
                    invalid.append(f"packet_shape:{t}:{r}")
                    continue
                packet = packets.get(pid)
                if packet is None:
                    invalid.append(f"packet_ref:{t}:{r}")
                    continue
                failure = failed.get(pid)
                if comm == "retry":
                    if (
                        failure is None
                        or failure[0] != t - 1
                        or (pid, failure[1]) in retry_used
                        or failure[2] != r
                        or (
                            failure[1] == 1
                            and (
                                action["dx"] != 0
                                or action["dy"] != 0
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
                            "hop": failure[1],
                            "relay": packet["relay"],
                            "packet": packet,
                        }
                    )
                    retry_used.add((pid, failure[1]))
                    packet["first_attempt"] = False
                else:
                    if (
                        packet["relay"] != r
                        or packet["stage"] != "queued"
                        or failure is not None
                        or not queues[r]
                        or queues[r][0] != pid
                        or action["dx"] != 0
                        or action["dy"] != 0
                    ):
                        invalid.append(f"packet_forward:{t}:{r}")
                        continue
                    submissions.append({"r": r, "pid": pid, "hop": 1, "relay": r, "packet": packet})
            else:
                invalid.append(f"packet_comm:{t}:{r}")
        submissions.sort(key=lambda item: req_key(item["pid"]))
        for submission in submissions:
            r, hop, relay = submission["r"], submission["hop"], submission["relay"]
            destination = spec["dispatcher"] if hop == 1 or relay is None else positions[relay]
            if not _hop_eligible(positions[r], destination):
                invalid.append(f"packet_range:{t}:{r}")
        accepted = submissions[: min(4, max(0, TEAM_BUDGET - accepted_units))]
        accepted_units += len(accepted)
        for submission in submissions[len(accepted) :]:
            invalid.append(f"channel:{t}:{submission['r']}")
        accepted_by_robot = {r: 0 for r in range(ROBOTS)}
        for submission in accepted:
            accepted_by_robot[submission["r"]] += 1
        for r in range(ROBOTS):
            motion_cost = 0.2 if proposed[r] == positions[r] else 2.0
            if energy[r] + EPS < motion_cost + accepted_by_robot[r]:
                invalid.append(f"energy:{t}:{r}")
        for submission in accepted:
            packet = submission["packet"]
            r, hop, relay = submission["r"], submission["hop"], submission["relay"]
            a = positions[r]
            b = list(spec["dispatcher"] if hop == 1 or relay is None else positions[relay])
            p = _packet_probability(name, spec, a, b, t, len(accepted), obstacles)
            if not _hop_eligible(a, b):
                p = 0.0
            energy[r] -= 1
            if _packet_success(
                name,
                t,
                packet["src"],
                packet["request"],
                packet["epoch"],
                hop,
                1 if (packet["packet_id"], hop) in retry_used else 0,
                p,
            ):
                packet["stage"] = "pending_arrival"
                failed.pop(packet["packet_id"], None)
                pending.setdefault(t + 1, []).append((packet["packet_id"], hop))
                if hop == 1 and packet["packet_id"] in queues[r]:
                    queues[r].remove(packet["packet_id"])
            else:
                packet["stage"] = "failed"
                failed[packet["packet_id"]] = (t, hop, r)
        for r in range(ROBOTS):
            if proposed[r] != positions[r]:
                energy[r] -= 2.0
                positions[r] = proposed[r]
            else:
                energy[r] -= 0.2
            if -EPS <= energy[r] < 0:
                energy[r] = 0.0
        for req in range(REQUESTS):
            robot = assignments[req]
            if robot is None:
                continue
            rs = _request_spec(name, req)
            if request_state[req] == "assigned" and tuple(positions[robot]) == (rs["x"], rs["y"]):
                request_state[req] = "serving"
                services[req] = {"start": t, "completion": None}
                service_progress[req] = 1
            elif request_state[req] == "serving":
                service_progress[req] += 1
            if request_state[req] == "serving" and service_progress[req] >= rs["service"]:
                request_state[req] = "completed"
                services[req]["completion"] = t + 1
                complete_values += rs["value"] if t + 1 <= rs["deadline"] else 0.0
                completed_this_slot = True
                assignments[req] = None
        valid_releases = [
            (
                _request_spec(name, _norm(current[r]["release_request"]))["value"],
                _norm(current[r]["release_request"]),
                r,
            )
            for r in range(ROBOTS)
            if _norm(current[r]["release_request"]) >= 0
            and _norm(current[r]["release_request"]) < REQUESTS
            and assignments[_norm(current[r]["release_request"])] == r
            and request_state[_norm(current[r]["release_request"])] == "assigned"
            and release_window
        ]
        released_this_slot = bool(valid_releases)
        if valid_releases:
            _, req, robot = min(valid_releases)
            assignments[req] = None
            request_state[req] = "released"
            release_epochs[req] += 1
            epoch_started[req] = t + 1
            proposals = {key: packet for key, packet in proposals.items() if key[1] != req}

        progress_options: list[bool] = []
        for req, robot in enumerate(assignments):
            if robot is not None and request_state[req] == "assigned":
                rs = _request_spec(name, req)
                possible = any(
                    0 <= positions[robot][0] + move[0] < GRID_X
                    and 0 <= positions[robot][1] + move[1] < GRID_Y
                    and _graph_connected(
                        spec,
                        [
                            [
                                positions[j][0] + (move[0] if j == robot else 0),
                                positions[j][1] + (move[1] if j == robot else 0),
                            ]
                            for j in range(ROBOTS)
                        ],
                        t,
                        obstacles,
                    )[0]
                    and not _blocked(
                        spec,
                        positions[robot][0] + move[0],
                        positions[robot][1] + move[1],
                        t,
                        obstacles,
                    )
                    and [positions[robot][0] + move[0], positions[robot][1] + move[1]]
                    not in [positions[j] for j in range(ROBOTS) if j != robot]
                    and abs(positions[robot][0] + move[0] - rs["x"])
                    + abs(positions[robot][1] + move[1] - rs["y"])
                    < abs(positions[robot][0] - rs["x"]) + abs(positions[robot][1] - rs["y"])
                    for move in MOVES[1:]
                )
                progress_options.append(possible)
        blocked = bool(progress_options) and not any(progress_options)
        blocked_count = (
            blocked_count + 1
            if blocked and not completed_this_slot and not released_this_slot
            else 0
        )
        blocked_trace.append({"slot": t, "blocked": blocked, "run": blocked_count})
        if blocked_count >= 3:
            deadlock_events += 1
            last_deadlock_slot = t
            deadlock_slots.append(t)
            blocked_count = 0
        if spec.get("moving") and t in (39, 43, 47, 51):
            xmin, xmax, ymin, ymax = obstacles[1]
            shifted = (xmin + 1, xmax + 1, ymin, ymax)
            if not any(
                shifted[0] <= position[0] <= shifted[1] and shifted[2] <= position[1] <= shifted[3]
                for position in positions
            ):
                obstacles[1] = shifted
        connected, margin = _graph_connected(spec, positions, t + 1, obstacles)
        connected_slots += int(connected)
        margins.append(margin)
        action_rows.extend(dict(current[r], t=t, robot=r) for r in range(ROBOTS))
        states.append(
            {
                "positions": [tuple(item) for item in positions],
                "energy": list(energy),
                "connected": connected,
                "margin": margin,
            }
        )
    discard_expired_failures(HORIZON)
    apply_arrivals(HORIZON)
    form_assignments(HORIZON)
    for req in range(REQUESTS):
        if services[req] is None:
            services[req] = {"start": None, "completion": None}
    return {
        "states": states,
        "decision_states": decision_states,
        "packets": packets,
        "services": services,
        "assignments": assignments,
        "accepted": accepted_units,
        "useful": useful_units,
        "complete_value": complete_values,
        "connected_slots": connected_slots,
        "margins": margins,
        "safety": safety_events,
        "energy_used": sum(ENERGY0 - value for value in energy),
        "invalid": invalid,
        "actions": action_rows,
        "packet_arrivals": packet_arrivals,
        "assignment_events": assignment_events,
        "deadlock_events": deadlock_events,
        "deadlock_slots": deadlock_slots,
        "blocked_trace": blocked_trace,
        "queues": [list(queue) for queue in queues],
        "obstacles": [tuple(item) for item in obstacles],
        "request_state": list(request_state),
    }


def _legality(scenarios: Mapping[str, Any]) -> list[AtomVerdict]:
    motion = release = packets = proposal_cost = resources = True
    defects = {key: [] for key in ("motion", "release", "packets", "proposal_cost", "resources")}
    for name in SCENARIO_NAMES:
        sc = scenarios.get(name, {})
        acts = sc.get("actions", []) if isinstance(sc, Mapping) else []
        replayable = True
        if not isinstance(acts, list):
            continue
        for i, action in enumerate(acts):
            path = f"{name}.actions[{i}]"
            if not _action_shape(action):
                replayable = False
                continue
            move_valid = (action["dx"], action["dy"]) in MOVES
            motion &= move_valid
            if not move_valid and len(defects["motion"]) < 8:
                defects["motion"].append(
                    f"{path}.(dx,dy): actual={(action['dx'], action['dy'])}; expected={MOVES}"
                )
            release_value = _norm(action["release_request"])
            release_valid = action["release_request"] in (None, -1) or 0 <= release_value < REQUESTS
            release &= release_valid
            if not release_valid and len(defects["release"]) < 8:
                defects["release"].append(
                    f"{path}.release_request: actual={release_value}; expected=-1 or 0..7"
                )
            packets &= action["comm"] in COMMS
            if action["comm"] not in COMMS and len(defects["packets"]) < 8:
                defects["packets"].append(
                    f"{path}.comm: actual={action['comm']!r}; expected={COMMS}"
                )
            optional_domains = (
                ("packet_id", None),
                ("proposal_request", REQUESTS),
                ("relay", ROBOTS),
            )
            for key, upper in optional_domains:
                raw = action[key]
                domain_valid = raw in (None, -1) or (
                    _integer(raw) and raw >= 0 and (upper is None or raw < upper)
                )
                packets &= domain_valid
                if not domain_valid and len(defects["packets"]) < 8:
                    expected = "null, -1, or nonnegative integer"
                    if upper is not None:
                        expected = f"null, -1, or integer 0..{upper - 1}"
                    defects["packets"].append(f"{path}.{key}: actual={raw!r}; expected={expected}")
            cost = action["proposal_cost"]
            cost_valid = action["comm"] != "new" or (_finite(cost) and 0 <= float(cost) <= 1)
            proposal_cost &= cost_valid
            if not cost_valid and len(defects["proposal_cost"]) < 8:
                defects["proposal_cost"].append(
                    f"{path}.proposal_cost: actual={cost!r}; expected=null or finite [0,1]"
                )
        if not replayable:
            continue
        replay = _replay(name, sc)
        motion_bad = [item for item in replay["invalid"] if item.startswith("serving_move")]
        packet_bad = [item for item in replay["invalid"] if item.startswith("packet")]
        cost_bad = [item for item in replay["invalid"] if item.startswith("proposal_cost")]
        resource_bad = [
            item for item in replay["invalid"] if item.startswith(("energy", "channel"))
        ]
        release_bad = [item for item in replay["invalid"] if item.startswith("release")]
        packets &= not packet_bad
        proposal_cost &= not cost_bad
        resources &= not resource_bad
        release &= not release_bad
        motion &= not motion_bad
        for group, values in (
            ("motion", motion_bad),
            ("packets", packet_bad),
            ("proposal_cost", cost_bad),
            ("resources", resource_bad),
            ("release", release_bad),
        ):
            defects[group].extend(f"{name}:{item}" for item in values[:8])
    return [
        _atom("decision_legality", "motion", motion, f"defects={defects['motion']}"),
        _atom("decision_legality", "release", release, f"defects={defects['release']}"),
        _atom("decision_legality", "packets", packets, f"defects={defects['packets']}"),
        _atom(
            "decision_legality",
            "proposal_cost",
            proposal_cost,
            f"defects={defects['proposal_cost']}",
        ),
        _atom("decision_legality", "resources", resources, f"defects={defects['resources']}"),
    ]


def _metrics(replayed: Mapping[str, Any]) -> dict[str, float]:
    p1 = p2 = p3 = a1 = a2 = 0.0
    for name in SCENARIO_NAMES:
        d = replayed[name]
        value_total = sum(item[3] for item in REQUESTS_TEMPLATE)
        completion_value = 0.0
        response = 0.0
        for req, service in enumerate(d["services"]):
            spec = _request_spec(name, req)
            if service["completion"] is not None:
                completion = service["completion"]
                completion_value += spec["value"] * (
                    1.0
                    if completion <= spec["deadline"]
                    else 0.5 * _clip(1 - (completion - spec["deadline"]) / HORIZON)
                )
            if service["start"] is not None:
                response += spec["value"] * _clip(
                    1 - (service["start"] - spec["release"]) / HORIZON
                )
        p1 += completion_value / value_total
        p2 += response / value_total
        connected_fraction = d["connected_slots"] / HORIZON
        margin = sum(d["margins"]) / HORIZON
        p3 += (
            0.4 * connected_fraction
            + 0.2 * _clip(margin)
            + 0.4 * _clip(1 - d["safety"] / (HORIZON * ROBOTS))
        )
        a1 += (
            (
                0.5 * _clip(1 - d["accepted"] / TEAM_BUDGET)
                + 0.5 * _clip(d["useful"] / max(1, d["accepted"]))
            )
            if d["accepted"]
            else 0.0
        )
        arrivals = d["packet_arrivals"]
        if arrivals:
            freshness = sum(
                _clip(1 - (item["state"] - d["packets"][item["pid"]]["generation_slot"]) / HORIZON)
                for item in arrivals
            ) / len(arrivals)
            first = sum(
                1 for item in arrivals if d["packets"][item["pid"]].get("first_attempt", True)
            ) / len(arrivals)
            a2 += 0.6 * freshness + 0.4 * first
    return {
        "primary_metrics_1": _clip(p1 / len(SCENARIO_NAMES)),
        "primary_metrics_2": _clip(p2 / len(SCENARIO_NAMES)),
        "primary_metrics_3": _clip(p3 / len(SCENARIO_NAMES)),
        "auxiliary_metrics_1": _clip(a1 / len(SCENARIO_NAMES)),
        "auxiliary_metrics_2": _clip(a2 / len(SCENARIO_NAMES)),
    }


def _replay_atoms(scenarios: Mapping[str, Any], replayed: Mapping[str, Any]) -> list[AtomVerdict]:
    ok = all(name in replayed for name in SCENARIO_NAMES)
    motion = ok and all(len(replayed[name]["states"]) == HORIZON for name in SCENARIO_NAMES)
    requests = ok and all(len(replayed[name]["services"]) == REQUESTS for name in SCENARIO_NAMES)
    packets = ok and all(replayed[name]["accepted"] <= TEAM_BUDGET for name in SCENARIO_NAMES)
    services = ok and all(
        all(item is not None for item in replayed[name]["services"]) for name in SCENARIO_NAMES
    )
    metrics = False
    if ok:
        try:
            metrics = all(
                _finite(value) and 0 <= value <= 1 for value in _metrics(replayed).values()
            )
        except Exception:
            metrics = False
    return [
        _atom("replay_validity", "motion", motion, "100 post-event boundaries per scenario"),
        _atom("replay_validity", "requests", requests, "eight request service states per scenario"),
        _atom("replay_validity", "packets", packets, "budget and packet replay"),
        _atom("replay_validity", "services", services, "service and deadlock replay"),
        _atom("replay_validity", "metrics", metrics, "finite normalized metrics"),
    ]


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()):
        raise ValueError("atom shares must be positive finite")
    if any(
        not math.isclose(atom.share, expected[atom.id], abs_tol=1e-12, rel_tol=0) for atom in atoms
    ):
        raise ValueError("atom share differs")
    if not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("shares must sum to one")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    return (
        "failed_atoms=[]"
        if not failed
        else "failed_atoms=[{}]; {}".format(
            ",".join(atom.id for atom in failed),
            " | ".join(f"{atom.id}: {atom.detail}" for atom in failed),
        )
    )


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = False
    try:
        removed = await ws.run("rm -f -- results.json", check=False, timeout=30)
        remove_ok = getattr(removed, "exit_code", None) == 0
        state.run_started = True
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600)
            state.exit_code = getattr(result, "exit_code", None)
            output = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".lower()
            state.timed_out = state.exit_code is None and any(
                word in output for word in ("timeout", "timed out")
            )
        except Exception as exc:
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
            state.timed_out = isinstance(exc, TimeoutError) or any(
                word in str(exc).lower() for word in ("timeout", "timed out")
            )
        finally:
            state.elapsed_s = time.monotonic() - started
            state.run_finished = True
        path = Path(ws.root) / "results.json"
        state.result_file_is_new_regular = remove_ok and path.is_file() and not path.is_symlink()
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600 + EPS
            and state.result_file_is_new_regular
        )
        status = f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
        state.official_message = (
            f"{state.official_message}; {status}" if state.official_message else status
        )
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(path.read_text(encoding="utf-8"))
                state.parse_ok = True
            except Exception as exc:
                state.diagnostics["result_json"] = f"strict parse failed: {exc}"
        parse_failure = state.diagnostics.get("result_json", "")
        shape = _shape_atoms(state.data, state.parse_ok)
        state.scenarios = _index(state.data)
        evidence = _evidence_atoms(state.scenarios)
        coverage, defects = _coverage(state.data)
        legality = _legality(state.scenarios)
        state.scores["result_json"] = _score_atoms(shape, ATOM_SHARES["result_json"])
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.scores["canonical_coverage"] = coverage
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        can_replay = all(
            state.scores[key] >= 1 - EPS
            for key in ("result_json", "result_evidence", "canonical_coverage", "decision_legality")
        )
        replayed = (
            {name: _replay(name, state.scenarios[name]) for name in SCENARIO_NAMES}
            if can_replay
            else {}
        )
        replay_atoms = _replay_atoms(state.scenarios, replayed)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["result_json"] = _diagnose(shape)
        if parse_failure:
            state.diagnostics["result_json"] = (
                f"{parse_failure}; {state.diagnostics['result_json']}"
            )
        state.diagnostics["result_evidence"] = _diagnose(evidence)
        state.diagnostics["canonical_coverage"] = (
            f"defect_count={len(defects)}; " + " | ".join(defects[:8])
            if defects
            else "defect_count=0"
        )
        state.diagnostics["decision_legality"] = _diagnose(legality)
        state.diagnostics["replay_validity"] = _diagnose(replay_atoms)
        state.validity_gate = state.official_ok and all(
            state.scores.get(key, 0) >= 1 - EPS for key in GATE_CHECKS
        )
        if state.validity_gate:
            state.metrics = _metrics(replayed)
        if not state.official_ok:
            for check_id in GATE_CHECKS:
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = (
                    f"official_run failed: {state.official_message}; {state.diagnostics.get(check_id, '')}"
                )
    except Exception as exc:
        state.run_finished = state.run_finished or state.run_started
        state.official_message = f"grader state failed: {type(exc).__name__}: {exc}"
        for check_id in GATE_CHECKS:
            state.scores[check_id] = 0.0
            state.diagnostics[check_id] = state.official_message
    _CACHE[key] = state
    return state


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
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    found: set[str] = set()
    for index, heading in enumerate(headings):
        title = re.sub(r"\s+#+\s*$", "", heading.group(2)).strip()
        if title not in required:
            continue
        level = len(heading.group(1))
        end = len(text)
        for later in headings[index + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        if text[heading.end() : end].strip():
            found.add(title)
    score = _clip(1 - (len(required) - len(found)) / 4)
    return (
        score,
        f"missing_sections={[title for title in required if title not in found]}; normalized={score:.9g}",
    )


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root) / "run.py"
    if not path.is_file() or path.is_symlink():
        return 0.0, "run.py is not a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30)
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


for _id in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        return state.scores.get(ident, 0.0), state.diagnostics.get(ident, "")

    globals()[_id] = check(id=_id)(_check)


for _id in (
    "primary_metrics_1",
    "primary_metrics_2",
    "primary_metrics_3",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, ident: str = _id) -> tuple[float, str]:
        state = await _build_state(ws)
        value = (state.metrics or {}).get(ident, 0.0)
        return (
            value,
            f"{ident}: replayed value={value:.9g}"
            if state.metrics is not None
            else f"{ident}: validity gate not satisfied",
        )

    globals()[_id] = check(id=_id)(_metric)

del _check, _metric
