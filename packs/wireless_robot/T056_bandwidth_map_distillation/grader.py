# ruff: noqa
"""Independent decision replay grader for T056."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import stat
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y, H = 22, 18, 150
ROBOTS = ("R0", "R1", "R2", "R3")
SCENARIOS = (
    "texture_transition",
    "loop_corridor",
    "bandwidth_surge",
    "edge_outage",
    "crossing_routes",
)
EDGE = (11, 9)
ACTION_DELTAS = {
    "stay": (0, 0),
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}
STARTS = {
    "texture_transition": ((1, 1), (20, 1), (1, 16), (20, 16)),
    "loop_corridor": ((1, 8), (20, 8), (11, 1), (11, 16)),
    "bandwidth_surge": ((1, 1), (20, 1), (1, 16), (20, 16)),
    "edge_outage": ((2, 2), (19, 2), (2, 15), (19, 15)),
    "crossing_routes": ((1, 8), (20, 8), (11, 1), (11, 16)),
}
OBSTACLES = {
    "texture_transition": ((10, 10, 0, 7), (10, 10, 10, 17)),
    "loop_corridor": ((5, 16, 3, 3), (5, 16, 14, 14)),
    "bandwidth_surge": ((7, 14, 5, 12),),
    "edge_outage": ((9, 12, 4, 13),),
    "crossing_routes": ((7, 9, 7, 10), (13, 15, 7, 10)),
}
BIAS_SCALE = {
    "texture_transition": 0.035,
    "loop_corridor": 0.045,
    "bandwidth_surge": 0.040,
    "edge_outage": 0.050,
    "crossing_routes": 0.055,
}
ENERGY_BUDGET = {
    "texture_transition": 620.0,
    "loop_corridor": 640.0,
    "bandwidth_surge": 630.0,
    "edge_outage": 650.0,
    "crossing_routes": 660.0,
}
REGIONS = {
    "texture_transition": (
        ((6, 14, .55, 80), (18, 14, .45, 140)),
        ((15, 6, .55, 80), (3, 6, .45, 140)),
        ((6, 3, .55, 80), (18, 3, .45, 140)),
        ((15, 11, .55, 80), (3, 11, .45, 140)),
    ),
    "loop_corridor": (
        ((19, 8, .55, 72), (2, 8, .45, 142)),
        ((2, 8, .55, 72), (19, 8, .45, 142)),
        ((11, 16, .55, 72), (11, 1, .45, 142)),
        ((11, 1, .55, 72), (11, 16, .45, 142)),
    ),
    "bandwidth_surge": (
        ((5, 14, .55, 70), (18, 14, .45, 142)),
        ((16, 3, .55, 70), (3, 3, .45, 142)),
        ((5, 3, .55, 70), (18, 3, .45, 142)),
        ((16, 14, .55, 70), (3, 14, .45, 142)),
    ),
    "edge_outage": (
        ((6, 13, .55, 76), (18, 13, .45, 142)),
        ((15, 4, .55, 76), (3, 4, .45, 142)),
        ((6, 4, .55, 76), (18, 4, .45, 142)),
        ((15, 13, .55, 76), (3, 13, .45, 142)),
    ),
    "crossing_routes": (
        ((5, 8, .55, 65), (18, 8, .45, 140)),
        ((17, 8, .55, 65), (4, 8, .45, 140)),
        ((11, 5, .55, 65), (11, 13, .45, 140)),
        ((11, 13, .55, 65), (11, 5, .45, 140)),
    ),
}
FRAME_RE = re.compile(r"^F_R[0-3]_(?:0|[1-9][0-9]{0,2})$")
LANDMARK_RE = re.compile(r"^L_(?:0|[1-9][0-9]*)_(?:0|[1-9][0-9]*)$")
CORRECTION_RE = re.compile(r"^C_F_R[0-3]_(?:0|[1-9][0-9]{0,2})__F_R[0-3]_(?:0|[1-9][0-9]{0,2})$")
REPORT_SECTIONS = (
    "Problem Formulation",
    "Method Design",
    "Experimental Setup",
    "Results",
    "Research Analysis",
    "Conclusion and Limitations",
)

ATOM_SHARES = {
    "result_json": {"result_json.parse": .20, "result_json.root": .20, "result_json.scenarios": .25, "result_json.records": .35},
    "result_evidence": {"result_evidence.slots": .25, "result_evidence.robots": .35, "result_evidence.sets": .20, "result_evidence.schedules": .20},
    "decision_legality": {"decision_legality.actions": .20, "decision_legality.sets": .25, "decision_legality.opportunities": .15, "decision_legality.schedules": .20, "decision_legality.edge_null": .20},
    "replay_validity": {"replay_validity.frames": .20, "replay_validity.transport": .20, "replay_validity.edge": .18, "replay_validity.corrections": .17, "replay_validity.motion": .10, "replay_validity.resources": .15},
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
        self.official_message = ""
        self.result_file_is_new_regular = False
        self.parse_ok = False
        self.data: Any = None
        self.records: dict[str, dict[str, Any]] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _integer(value: Any, lo: int, hi: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi


def _shape(value: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object"
    if set(value) != keys:
        return f"{path}: expected keys={sorted(keys)!r}, actual={sorted(value)!r}"
    return None


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


def _wall(name: str, cell: tuple[int, int]) -> bool:
    x, y = cell
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in OBSTACLES[name])


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x0, y0 = a
    x1, y1 = b
    dx, dy = x1 - x0, y1 - y0
    out = [(x0, y0)]
    if not dx and not dy:
        return tuple(out)
    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0
    ax, ay = abs(dx), abs(dy)
    tx = .5 / ax if ax else math.inf
    ty = .5 / ay if ay else math.inf
    dtx = 1 / ax if ax else math.inf
    dty = 1 / ay if ay else math.inf
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if tx < ty - EPS:
            x += sx
            tx += dtx
            out.append((x, y))
        elif ty < tx - EPS:
            y += sy
            ty += dty
            out.append((x, y))
        else:
            out.extend(((x + sx, y), (x, y + sy)))
            x += sx
            y += sy
            tx += dtx
            ty += dty
            out.append((x, y))
    return tuple(dict.fromkeys(out))


def _interference(name: str, slot: int) -> int:
    if name in {"loop_corridor", "edge_outage"}:
        return 1
    if name == "bandwidth_surge" and 35 <= slot <= 69:
        return 2
    if name == "crossing_routes" and 50 <= slot <= 89:
        return 1
    return 0


def _edge_offline(name: str, slot: int) -> bool:
    return name == "edge_outage" and 45 <= slot <= 74


def _quality(name: str, position: tuple[int, int], slot: int) -> float:
    blocked = any(_wall(name, cell) for cell in _line_cells(position, EDGE)[1:-1])
    return _clip(1 - math.dist(position, EDGE) / 30 - .15 * blocked - .08 * _interference(name, slot))


def _landmarks(name: str) -> dict[str, tuple[int, int, float]]:
    out = {}
    for y in range(GRID_Y):
        for x in range(GRID_X):
            if not _wall(name, (x, y)) and (7 * x + 11 * y) % 9 == 0:
                out[f"L_{x}_{y}"] = (x, y, .4 + .1 * ((3 * x + 5 * y) % 6))
    return out


def _visible(name: str, position: tuple[int, int], slot: int, robot: str) -> list[dict[str, Any]]:
    if slot % 5:
        return []
    i = ROBOTS.index(robot)
    out = []
    for ident, (x, y, stability) in _landmarks(name).items():
        if max(abs(x - position[0]), abs(y - position[1])) > 3:
            continue
        if any(_wall(name, cell) for cell in _line_cells(position, (x, y))[1:-1]):
            continue
        noise = .05 * (((13 * x + 7 * y + 3 * slot + i) % 5) - 2)
        out.append({
            "landmark_id": ident,
            "local_x": x - position[0] + noise,
            "local_y": y - position[1] + noise,
            "stability": stability,
        })
    return sorted(out, key=lambda item: item["landmark_id"])


def _frame_id(robot: str, slot: int) -> str:
    return f"F_{robot}_{slot}"


def _frame_key(frame: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(frame.get("processed_slot", H + 1)), ROBOTS.index(frame["robot"]), int(frame["created_slot"]))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


class ScenarioEngine:
    """Pure public-model environment used by replay and author-side policies."""

    def __init__(self, name: str) -> None:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}")
        self.name = name
        self.slot = 0
        self.prepared = False
        self.true = {robot: STARTS[name][i] for i, robot in enumerate(ROBOTS)}
        self.estimate = {robot: tuple(map(float, STARTS[name][i])) for i, robot in enumerate(ROBOTS)}
        self.upload = {robot: [] for robot in ROBOTS}
        self.downlink = {robot: [] for robot in ROBOTS}
        self.pending = {robot: {} for robot in ROBOTS}
        self.frame_transit: list[dict[str, Any]] = []
        self.correction_transit: list[dict[str, Any]] = []
        self.edge_queue: list[dict[str, Any]] = []
        self.processed: list[dict[str, Any]] = []
        self.frames: dict[str, dict[str, Any]] = {}
        self.corrections: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.completed = {robot: set() for robot in ROBOTS}
        self.errors = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}
        self.energy = 0.0
        self.uplink_bits = 0
        self.downlink_bits = 0
        self.safety_rejections = 0
        self.executed_moves = 0
        self.generated_frames = 0
        self.successful_uploads = 0
        self.pose_errors = [math.dist(self.true[r], self.estimate[r]) for r in ROBOTS]
        self.applied: list[dict[str, Any]] = []

    def _error(self, key: str, detail: str) -> None:
        if len(self.errors[key]) < 8:
            self.errors[key].append(detail)

    def _prepare(self) -> None:
        if self.prepared:
            return
        t = self.slot
        arrivals = [item for item in self.frame_transit if item["arrival_slot"] == t]
        self.frame_transit = [item for item in self.frame_transit if item["arrival_slot"] != t]
        arrivals.sort(key=lambda frame: (frame["completion_slot"], ROBOTS.index(frame["robot"]), frame["created_slot"]))
        for frame in arrivals:
            frame["arrival_slot"] = t
            if len(self.edge_queue) < 16:
                self.edge_queue.append(frame)
            else:
                frame["edge_overflow"] = True
        corr_arrivals = [item for item in self.correction_transit if item["arrival_slot"] == t]
        self.correction_transit = [item for item in self.correction_transit if item["arrival_slot"] != t]
        for corr in sorted(corr_arrivals, key=lambda item: item["id"]):
            if t - corr["created_slot"] <= 30:
                self.pending[corr["robot"]][corr["id"]] = corr
            else:
                corr["expired"] = True
        for robot in ROBOTS:
            kept = []
            for corr in self.downlink[robot]:
                if t - corr["created_slot"] > 30:
                    corr["expired"] = True
                else:
                    kept.append(corr)
            self.downlink[robot] = kept
            for ident, corr in list(self.pending[robot].items()):
                if t - corr["created_slot"] > 30:
                    corr["expired"] = True
                    del self.pending[robot][ident]
        self.prepared = True

    def context(self) -> dict[str, Any]:
        self._prepare()
        robots = {}
        for robot in ROBOTS:
            robots[robot] = {
                "estimate": self.estimate[robot],
                "visible_landmarks": _visible(self.name, self.true[robot], self.slot, robot),
                "link_quality": _quality(self.name, self.true[robot], self.slot),
                "upload_queue": [frame["id"] for frame in self.upload[robot]],
                "downlink_queue": [corr["id"] for corr in self.downlink[robot]],
                "pending_corrections": [
                    {"correction_id": corr["id"], "dx": corr["dx"], "dy": corr["dy"], "age": self.slot - corr["created_slot"]}
                    for corr in sorted(self.pending[robot].values(), key=lambda item: item["id"])
                ],
                "completed_regions": sorted(self.completed[robot]),
                "regions": REGIONS[self.name][ROBOTS.index(robot)],
            }
        return copy.deepcopy({
            "scenario": self.name,
            "slot": self.slot,
            "robots": robots,
            "edge_queue": [
                {"frame_id": frame["id"], "robot_id": frame["robot"], "landmark_ids": list(frame["landmark_ids"]), "arrival_slot": frame["arrival_slot"]}
                for frame in self.edge_queue
            ],
            "processed_frames": [
                {"frame_id": frame["id"], "robot_id": frame["robot"], "landmark_ids": list(frame["landmark_ids"]), "processed_slot": frame["processed_slot"]}
                for frame in self.processed
            ],
            "energy_used": self.energy,
            "uplink_bits": self.uplink_bits,
        })

    def _serve(self, order: Sequence[str], kind: str) -> None:
        shared = 1024 if kind == "frame" else 256
        queues = self.upload if kind == "frame" else self.downlink
        for robot in order:
            if shared <= 0 or robot not in queues or not queues[robot]:
                continue
            packet = queues[robot][0]
            q = _quality(self.name, self.true[robot], self.slot)
            base = 1024 if kind == "frame" else 256
            link_limit = math.floor(base * q / 32) * 32
            served = min(int(packet["remaining_bits"]), int(link_limit), shared)
            packet["remaining_bits"] -= served
            shared -= served
            self.energy += .0005 * served
            if kind == "frame":
                self.uplink_bits += served
            else:
                self.downlink_bits += served
            if served <= 0 or packet["remaining_bits"]:
                continue
            i = ROBOTS.index(robot)
            success = ((17 * i + 5 * self.slot + (0 if kind == "frame" else 1)) % 100) / 100 >= .05 + .2 * (1 - q)
            if not success:
                packet["remaining_bits"] = packet["payload_bits"]
                packet["retries"] += 1
                continue
            queues[robot].pop(0)
            packet["completion_slot"] = self.slot
            packet["arrival_slot"] = self.slot + 1 + math.floor(3 * (1 - q))
            if kind == "frame":
                packet["successful_upload"] = True
                self.successful_uploads += 1
                self.frame_transit.append(packet)
            else:
                self.correction_transit.append(packet)

    def _process_edge(self, selected: Any) -> None:
        if selected is None:
            return
        matches = [frame for frame in self.edge_queue if frame["id"] == selected]
        if not matches:
            self._error("edge", f"slot {self.slot}: frame {selected!r} is not in edge queue")
            return
        current = matches[0]
        self.edge_queue.remove(current)
        current["processed_slot"] = self.slot
        productive = False
        for previous in sorted(self.processed, key=_frame_key):
            if previous["robot"] == current["robot"]:
                continue
            shared = sorted(set(current["landmark_ids"]) & set(previous["landmark_ids"]))
            if len(shared) < 3:
                continue
            dxs, dys = [], []
            for ident in shared:
                co = current["observations"][ident]
                po = previous["observations"][ident]
                dxs.append(current["estimate"][0] + co[0] - previous["estimate"][0] - po[0])
                dys.append(current["estimate"][1] + co[1] - previous["estimate"][1] - po[1])
            rx, ry = _median(dxs), _median(dys)
            if abs(rx) > 2.5 + EPS or abs(ry) > 2.5 + EPS:
                continue
            productive = True
            current["useful_landmarks"].update(shared)
            previous["useful_landmarks"].update(shared)
            edge = {"current": current["id"], "previous": previous["id"], "shared": shared, "rx": rx, "ry": ry, "slot": self.slot}
            self.edges.append(edge)
            ident = f"C_{current['id']}__{previous['id']}"
            corr = {
                "id": ident,
                "robot": current["robot"],
                "created_slot": self.slot,
                "dx": _clip(-.5 * rx, -4, 4),
                "dy": _clip(-.5 * ry, -4, 4),
                "payload_bits": 128,
                "remaining_bits": 128,
                "retries": 0,
                "applied": False,
            }
            self.corrections[ident] = corr
            if len(self.downlink[corr["robot"]]) < 16:
                self.downlink[corr["robot"]].append(corr)
            else:
                corr["downlink_overflow"] = True
        current["productive"] = productive
        self.processed.append(current)
        self.energy += .1

    def step(self, row: Mapping[str, Any]) -> None:
        self._prepare()
        t = self.slot
        raw_decisions = row.get("robot_decisions", []) if isinstance(row, Mapping) else []
        decisions = {item.get("robot_id"): item for item in raw_decisions if isinstance(item, Mapping) and item.get("robot_id") in ROBOTS}
        for robot in ROBOTS:
            decision = decisions.get(robot, {"action": "stay", "landmark_ids": [], "apply_correction_ids": []})
            apply_ids = decision.get("apply_correction_ids", [])
            if not isinstance(apply_ids, list) or not all(isinstance(item, str) for item in apply_ids):
                apply_ids = []
            else:
                apply_ids = [item for item in apply_ids if CORRECTION_RE.fullmatch(item)]
            chosen = []
            for ident in apply_ids:
                corr = self.pending[robot].get(ident)
                if corr is None or corr.get("applied") or t - corr["created_slot"] > 30:
                    self._error("corrections", f"slot {t} {robot}: correction {ident!r} is not eligible")
                else:
                    chosen.append(corr)
            if chosen:
                ex, ey = self.estimate[robot]
                ex = _clip(ex + sum(corr["dx"] for corr in chosen), -8, 29)
                ey = _clip(ey + sum(corr["dy"] for corr in chosen), -8, 25)
                self.estimate[robot] = (ex, ey)
                for corr in chosen:
                    corr["applied"] = True
                    corr["applied_slot"] = t
                    self.applied.append(corr)
                    del self.pending[robot][corr["id"]]
                self.energy += .05 * len(chosen)
        for robot in ROBOTS:
            decision = decisions.get(robot, {"landmark_ids": []})
            selected = decision.get("landmark_ids", [])
            if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
                selected = []
            elif t % 5:
                selected = []
            else:
                selected = [item for item in selected if LANDMARK_RE.fullmatch(item)]
            visible = {item["landmark_id"]: item for item in _visible(self.name, self.true[robot], t, robot)}
            invalid = sorted(set(selected) - set(visible))
            if invalid:
                self._error("frames", f"slot {t} {robot}: non-visible landmarks {invalid!r}")
                continue
            if selected:
                if len(self.upload[robot]) >= 8:
                    self._error("frames", f"slot {t} {robot}: upload FIFO is full")
                    continue
                canonical = sorted(set(selected))
                frame = {
                    "id": _frame_id(robot, t),
                    "robot": robot,
                    "created_slot": t,
                    "landmark_ids": canonical,
                    "observations": {ident: (visible[ident]["local_x"], visible[ident]["local_y"]) for ident in canonical},
                    "estimate": self.estimate[robot],
                    "payload_bits": 96 + 32 * len(canonical),
                    "remaining_bits": 96 + 32 * len(canonical),
                    "retries": 0,
                    "successful_upload": False,
                    "useful_landmarks": set(),
                }
                self.frames[frame["id"]] = frame
                self.upload[robot].append(frame)
                self.generated_frames += 1
                self.energy += .2
        up = row.get("uplink_order", []) if isinstance(row, Mapping) else []
        down = row.get("downlink_order", []) if isinstance(row, Mapping) else []
        self._serve(up if isinstance(up, list) and all(isinstance(item, str) for item in up) else [], "frame")
        self._serve(down if isinstance(down, list) and all(isinstance(item, str) for item in down) else [], "correction")
        selected_edge = row.get("edge_frame_id") if isinstance(row, Mapping) else None
        if selected_edge is not None and (not isinstance(selected_edge, str) or not FRAME_RE.fullmatch(selected_edge)):
            selected_edge = None
        if t % 3 == 0 and not _edge_offline(self.name, t):
            self._process_edge(selected_edge)
        requested = {}
        for robot in ROBOTS:
            action = decisions.get(robot, {}).get("action", "stay")
            requested[robot] = action if isinstance(action, str) and action in ACTION_DELTAS else "stay"
        tentative = {}
        rejected = set()
        for robot, action in requested.items():
            dx, dy = ACTION_DELTAS.get(action, (0, 0))
            target = (self.true[robot][0] + dx, self.true[robot][1] + dy)
            if action != "stay" and (not (0 <= target[0] < GRID_X and 0 <= target[1] < GRID_Y) or _wall(self.name, target)):
                target = self.true[robot]
                rejected.add(robot)
            tentative[robot] = target
        current_cells = {robot: self.true[robot] for robot in ROBOTS}
        moving_targets = Counter(tentative[r] for r in ROBOTS if tentative[r] != current_cells[r])
        for robot in ROBOTS:
            if tentative[robot] == current_cells[robot]:
                continue
            if tentative[robot] in {cell for other, cell in current_cells.items() if other != robot} or moving_targets[tentative[robot]] > 1:
                rejected.add(robot)
        for robot in ROBOTS:
            if robot in rejected:
                tentative[robot] = current_cells[robot]
                if requested[robot] != "stay":
                    self.safety_rejections += 1
            moved = tentative[robot] != current_cells[robot]
            self.true[robot] = tentative[robot]
            if moved:
                self.executed_moves += 1
                self.energy += 1
                dx = tentative[robot][0] - current_cells[robot][0]
                dy = tentative[robot][1] - current_cells[robot][1]
                i = ROBOTS.index(robot)
                bias = BIAS_SCALE[self.name] / math.sqrt(2)
                ex, ey = self.estimate[robot]
                self.estimate[robot] = (
                    _clip(ex + dx + bias * ((-1) ** i), -8, 29),
                    _clip(ey + dy + bias * ((-1) ** (i + 1)), -8, 25),
                )
            self.pose_errors.append(math.dist(self.true[robot], self.estimate[robot]))
            next_index = len(self.completed[robot])
            if next_index < 2:
                x, y, _value, deadline = REGIONS[self.name][ROBOTS.index(robot)][next_index]
                if t <= deadline and max(abs(self.true[robot][0] - x), abs(self.true[robot][1] - y)) <= 1 and math.dist(self.true[robot], self.estimate[robot]) <= 2 + EPS:
                    self.completed[robot].add(next_index)
        if self.energy > ENERGY_BUDGET[self.name] + EPS:
            self._error("resources", f"slot {t}: energy {self.energy:.9g} exceeds {ENERGY_BUDGET[self.name]:.9g}")
        if self.uplink_bits > 80000:
            self._error("transport", f"slot {t}: uplink bits {self.uplink_bits} exceed 80000")
        if any(len(self.upload[r]) > 8 or len(self.downlink[r]) > 16 for r in ROBOTS) or len(self.edge_queue) > 16:
            self._error("resources", f"slot {t}: queue capacity invariant failed")
        if any(not all(math.isfinite(v) for v in self.estimate[r]) for r in ROBOTS):
            self._error("motion", f"slot {t}: non-finite estimated state")
        self.slot += 1
        self.prepared = False

    def result(self) -> dict[str, Any]:
        return {
            "engine": self,
            "errors": copy.deepcopy(self.errors),
            "frames": self.frames,
            "processed": self.processed,
            "edges": self.edges,
            "applied": self.applied,
            "pose_errors": self.pose_errors,
            "completed": self.completed,
            "energy": self.energy,
            "uplink_bits": self.uplink_bits,
            "generated_frames": self.generated_frames,
            "successful_uploads": self.successful_uploads,
            "safety_rejections": self.safety_rejections,
            "executed_moves": self.executed_moves,
        }


def generate_scenario(name: str, policy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    engine = ScenarioEngine(name)
    slots = []
    while engine.slot < H:
        context = engine.context()
        row = policy(context)
        slots.append(copy.deepcopy(row))
        engine.step(row)
    return {"scenario": name, "slots": slots}, engine.result()


def _result_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    record_errors = []
    if isinstance(scenarios, list):
        for i, row in enumerate(scenarios):
            issue = _shape(row, {"scenario", "slots"}, f"scenarios[{i}]")
            if issue:
                record_errors.append(issue)
    checks = {
        "parse": (parsed, "strict JSON parse"),
        "root": (isinstance(data, dict) and set(data) == {"scenarios"}, "root keys must equal ['scenarios']"),
        "scenarios": (isinstance(scenarios, list), "scenarios must be an array"),
        "records": (isinstance(scenarios, list) and not record_errors, record_errors[0] if record_errors else "scenario records have exact scenario,slots keys"),
    }
    return [AtomVerdict(f"result_json.{key}", value[0], value[1], ATOM_SHARES["result_json"][f"result_json.{key}"]) for key, value in checks.items()]


def _records(data: Any) -> dict[str, dict[str, Any]]:
    out = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return out
    for row in data["scenarios"]:
        if isinstance(row, dict) and isinstance(row.get("scenario"), str):
            out[row["scenario"]] = row
    return out


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key.split(".", 1)[1]: [True, "ok"] for key in ATOM_SHARES["result_evidence"]}
    for name, record in records.items():
        slots = record.get("slots")
        if not isinstance(slots, list):
            checks["slots"] = [False, f"{name}.slots must be an array"]
            continue
        for i, row in enumerate(slots):
            path = f"{name}.slots[{i}]"
            issue = _shape(row, {"slot", "robot_decisions", "uplink_order", "downlink_order", "edge_frame_id"}, path)
            if issue:
                checks["slots"] = [False, issue]
                continue
            if not isinstance(row["slot"], int) or isinstance(row["slot"], bool):
                checks["slots"] = [False, f"{path}.slot must be a non-boolean integer"]
            if not isinstance(row["robot_decisions"], list):
                checks["robots"] = [False, f"{path}.robot_decisions must be an array"]
                continue
            for j, decision in enumerate(row["robot_decisions"]):
                dpath = f"{path}.robot_decisions[{j}]"
                issue = _shape(decision, {"robot_id", "action", "landmark_ids", "apply_correction_ids"}, dpath)
                if issue:
                    checks["robots"] = [False, issue]
                    continue
                if not isinstance(decision["robot_id"], str) or not isinstance(decision["action"], str):
                    checks["robots"] = [False, f"{dpath}: robot_id and action must be strings"]
                for field in ("landmark_ids", "apply_correction_ids"):
                    value = decision[field]
                    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                        checks["sets"] = [False, f"{dpath}.{field} must be an array of strings"]
            for field in ("uplink_order", "downlink_order"):
                value = row[field]
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    checks["schedules"] = [False, f"{path}.{field} must be an array of strings"]
            if row["edge_frame_id"] is not None and not isinstance(row["edge_frame_id"], str):
                checks["schedules"] = [False, f"{path}.edge_frame_id must be string or null"]
    return [AtomVerdict(f"result_evidence.{key}", value[0], value[1], ATOM_SHARES["result_evidence"][f"result_evidence.{key}"]) for key, value in checks.items()]


def _typed_token(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios is not an array"]
    expected_scenarios = {_typed_token(name) for name in SCENARIOS}
    actual_scenarios = [_typed_token(row.get("scenario")) for row in scenarios if isinstance(row, dict)]
    counts = Counter(actual_scenarios)
    for token in sorted(expected_scenarios - set(counts)):
        defects.append(f"missing scenario {token[1]}")
    for token in sorted(set(counts) - expected_scenarios):
        defects.append(f"unknown scenario {token[1]}")
    for token, count in counts.items():
        if token in expected_scenarios and count > 1:
            defects.append(f"duplicate scenario {token[1]} count={count}")
    for name in SCENARIOS:
        matches = [row for row in scenarios if isinstance(row, dict) and type(row.get("scenario")) is str and row.get("scenario") == name]
        if len(matches) != 1 or not isinstance(matches[0].get("slots"), list):
            continue
        slot_counts = Counter(_typed_token(row.get("slot")) for row in matches[0]["slots"] if isinstance(row, dict))
        expected_slots = {_typed_token(slot) for slot in range(H)}
        for token in expected_slots - set(slot_counts):
            defects.append(f"{name}: missing slot {token[1]}")
        for token in set(slot_counts) - expected_slots:
            defects.append(f"{name}: unknown slot {token[1]}")
        for token, count in slot_counts.items():
            if token in expected_slots and count > 1:
                defects.append(f"{name}: duplicate slot {token[1]} count={count}")
        for row in matches[0]["slots"]:
            if not isinstance(row, dict) or not _integer(row.get("slot"), 0, H - 1) or not isinstance(row.get("robot_decisions"), list):
                continue
            robot_counts = Counter(_typed_token(item.get("robot_id")) for item in row["robot_decisions"] if isinstance(item, dict))
            expected_robots = {_typed_token(robot) for robot in ROBOTS}
            for token in expected_robots - set(robot_counts):
                defects.append(f"{name} slot {row['slot']}: missing robot {token[1]}")
            for token in set(robot_counts) - expected_robots:
                defects.append(f"{name} slot {row['slot']}: unknown robot {token[1]}")
            for token, count in robot_counts.items():
                if token in expected_robots and count > 1:
                    defects.append(f"{name} slot {row['slot']}: duplicate robot {token[1]} count={count}")
    return max(0.0, 4.0 - .5 * len(defects)) / 4.0, defects[:40]


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key.split(".", 1)[1]: [True, "ok"] for key in ATOM_SHARES["decision_legality"]}
    for name, record in records.items():
        for row in record.get("slots", []) if isinstance(record.get("slots"), list) else []:
            if not isinstance(row, dict):
                continue
            slot = row.get("slot")
            decisions = row.get("robot_decisions", [])
            if isinstance(decisions, list):
                for decision in decisions:
                    if not isinstance(decision, dict):
                        continue
                    robot = decision.get("robot_id")
                    action = decision.get("action")
                    if isinstance(action, str) and action not in ACTION_DELTAS:
                        checks["actions"] = [False, f"{name} slot {slot} {robot}: unknown action {action!r}"]
                    landmarks = decision.get("landmark_ids")
                    corrections = decision.get("apply_correction_ids")
                    if isinstance(landmarks, list):
                        landmark_strings = all(isinstance(item, str) for item in landmarks)
                        if landmark_strings and (len(landmarks) != len(set(landmarks)) or any(not LANDMARK_RE.fullmatch(item) for item in landmarks)):
                            checks["sets"] = [False, f"{name} slot {slot} {robot}: invalid or duplicate landmark IDs"]
                        if landmark_strings and len(landmarks) > 12:
                            checks["sets"] = [False, f"{name} slot {slot} {robot}: more than 12 landmarks"]
                        if landmark_strings and _integer(slot, 0, H - 1) and slot % 5 and landmarks:
                            checks["opportunities"] = [False, f"{name} slot {slot} {robot}: landmarks outside opportunity"]
                    if isinstance(corrections, list):
                        correction_strings = all(isinstance(item, str) for item in corrections)
                        if correction_strings and (len(corrections) != len(set(corrections)) or any(not CORRECTION_RE.fullmatch(item) for item in corrections)):
                            checks["sets"] = [False, f"{name} slot {slot} {robot}: invalid or duplicate correction IDs"]
            for field in ("uplink_order", "downlink_order"):
                order = row.get(field)
                if isinstance(order, list):
                    robot_strings = all(isinstance(robot, str) for robot in order)
                    if robot_strings and (len(order) != len(set(order)) or any(robot not in ROBOTS for robot in order)):
                        checks["schedules"] = [False, f"{name} slot {slot}: {field} must be a unique robot subset"]
            edge = row.get("edge_frame_id")
            if isinstance(edge, str) and not FRAME_RE.fullmatch(edge):
                checks["edge_null"] = [False, f"{name} slot {slot}: malformed edge frame ID"]
            if _integer(slot, 0, H - 1) and isinstance(edge, str) and (slot % 3 or _edge_offline(name, slot)):
                checks["edge_null"] = [False, f"{name} slot {slot}: edge frame must be null"]
    return [AtomVerdict(f"decision_legality.{key}", value[0], value[1], ATOM_SHARES["decision_legality"][f"decision_legality.{key}"]) for key, value in checks.items()]


def _canonical_slot_rows(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in record.get("slots", []) if isinstance(record.get("slots"), list) else []:
        if isinstance(row, dict) and _integer(row.get("slot"), 0, H - 1) and row["slot"] not in out:
            out[row["slot"]] = row
    return out


def _replay_atoms(records: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    aggregate = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}
    replayed = {}
    for name in SCENARIOS:
        record = records.get(name)
        rows = _canonical_slot_rows(record) if isinstance(record, Mapping) else {}
        engine = ScenarioEngine(name)
        for slot in range(H):
            row = rows.get(slot)
            if row is None:
                for key in aggregate:
                    aggregate[key].append(f"{name}: missing replay row for slot {slot}")
                row = {
                    "slot": slot,
                    "robot_decisions": [{"robot_id": robot, "action": "stay", "landmark_ids": [], "apply_correction_ids": []} for robot in ROBOTS],
                    "uplink_order": [],
                    "downlink_order": [],
                    "edge_frame_id": None,
                }
            try:
                engine.step(row)
            except Exception as exc:
                for key in aggregate:
                    aggregate[key].append(f"{name} slot {slot}: replay exception {type(exc).__name__}: {exc}")
                break
        replayed[name] = engine.result()
        for key, errors in engine.errors.items():
            aggregate[key].extend(f"{name}: {detail}" for detail in errors)
    atoms = []
    for key, errors in aggregate.items():
        detail = "ok" if not errors else " | ".join(errors[:8])
        atoms.append(AtomVerdict(f"replay_validity.{key}", not errors, detail, ATOM_SHARES["replay_validity"][f"replay_validity.{key}"]))
    return atoms, replayed


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected.values()) or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("atom shares must be finite, positive and sum to one")
    if any(not math.isclose(atom.share, expected[atom.id], rel_tol=0.0, abs_tol=1e-12) for atom in atoms):
        raise ValueError("atom share differs from declared share")
    return _clip(sum(atom.share for atom in atoms if atom.passed))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(atom.id for atom in failed) + "]; " + " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)


def _report_score(text: str) -> tuple[float, list[str]]:
    missing = []
    for title in REPORT_SECTIONS:
        match = re.search(rf"(?m)^(#{{1,6}})\s+{re.escape(title)}\s*$", text)
        if not match:
            missing.append(title)
            continue
        level = len(match.group(1))
        body = []
        for line in text[match.end():].splitlines():
            heading = re.match(r"^(#{1,6})\s+", line)
            if heading and len(heading.group(1)) <= level:
                break
            if not heading:
                body.append(line)
        if not "\n".join(body).strip():
            missing.append(title)
    return max(0.0, 2.0 - .5 * len(missing)) / 2.0, missing


def _scenario_metrics(name: str, replay: Mapping[str, Any]) -> dict[str, float]:
    completed_value = 0.0
    for robot in ROBOTS:
        for index in replay["completed"][robot]:
            completed_value += REGIONS[name][ROBOTS.index(robot)][index][2]
    goal = completed_value / 4
    rmse_pose = math.sqrt(sum(error * error for error in replay["pose_errors"]) / len(replay["pose_errors"]))
    p1 = .6 * goal + .4 * _clip(1 - rmse_pose / 8)
    contributors: dict[str, set[str]] = defaultdict(set)
    observations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for frame in replay["processed"]:
        for ident, local in frame["observations"].items():
            contributors[ident].add(frame["robot"])
            observations[ident].append((frame["estimate"][0] + local[0], frame["estimate"][1] + local[1]))
    collaborative = {ident for ident, robots in contributors.items() if len(robots) >= 2}
    errors = []
    fixed = _landmarks(name)
    for ident in collaborative:
        x, y, _stability = fixed[ident]
        errors.extend(math.dist(point, (x, y)) for point in observations[ident])
    consistency = 0.0 if not errors else _clip(1 - math.sqrt(sum(value * value for value in errors) / len(errors)) / 6)
    coverage = len(collaborative) / len(fixed)
    p2 = .6 * consistency + .4 * coverage
    p3 = .45 * _clip(1 - replay["safety_rejections"] / (4 * H)) + .30 * _clip(1 - replay["executed_moves"] / (4 * H)) + .25 * _clip(1 - replay["energy"] / ENERGY_BUDGET[name])
    useful_bits = 0
    for frame in replay["processed"]:
        if frame.get("successful_upload") and frame["useful_landmarks"]:
            useful_bits += 96 + 32 * len(frame["useful_landmarks"])
    upload_ratio = 0.0 if replay["generated_frames"] == 0 else replay["successful_uploads"] / replay["generated_frames"]
    a1 = _clip(.65 * useful_bits / max(1, replay["uplink_bits"]) + .35 * upload_ratio)
    freshness = [
        _clip(1 - (corr["applied_slot"] - corr["created_slot"]) / 30)
        for corr in replay["applied"]
    ]
    waits = [_clip(1 - (frame["processed_slot"] - frame["arrival_slot"]) / 30) for frame in replay["processed"]]
    productive = [bool(frame.get("productive")) for frame in replay["processed"]]
    a2 = .4 * (sum(freshness) / len(freshness) if freshness else 0.0) + .3 * (sum(waits) / len(waits) if waits else 0.0) + .3 * (sum(productive) / len(productive) if productive else 0.0)
    return {"primary_metrics_1": _clip(p1), "primary_metrics_2": _clip(p2), "primary_metrics_3": _clip(p3), "auxiliary_metrics_1": _clip(a1), "auxiliary_metrics_2": _clip(a2)}


def _metrics(replayed: Mapping[str, dict[str, Any]], _records_arg: Mapping[str, dict[str, Any]] | None = None) -> dict[str, float]:
    values = {key: [] for key in ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")}
    for name in SCENARIOS:
        metrics = _scenario_metrics(name, replayed[name])
        for key, value in metrics.items():
            values[key].append(value)
    return {key: _clip(sum(items) / len(items)) for key, items in values.items()}


async def _build_state(ws: Workspace) -> EvalState:
    root = str(ws.root)
    if root in _CACHE:
        return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE:
            return _CACHE[root]
        state = EvalState()
        path = Path(ws.root, "results.json")
        try:
            await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            started = time.monotonic()
            state.run_started = True
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
            state.official_ok = state.exit_code == 0 and not state.timed_out and state.elapsed_s <= 600.0 + EPS and state.result_file_is_new_regular
            state.official_message = f"exit_code={state.exit_code}, elapsed_s={state.elapsed_s:.6f}, fresh_regular_result={state.result_file_is_new_regular}"
            if state.official_ok:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
        except Exception as exc:
            state.timed_out = (
                isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
            state.official_message = f"{type(exc).__name__}: {exc}"
        state.diagnostics["official"] = state.official_message or "official run did not complete"
        atoms = _result_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = _diagnose_atoms(atoms)
        if not state.official_ok or not state.parse_ok:
            reason = state.diagnostics.get("official", "official run or strict JSON parse failed")
            for key in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[key] = 0.0
                state.diagnostics[key] = f"{key}: not evaluated because {reason}"
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
        state.validity_gate = state.official_ok and all(state.scores.get(key, 0.0) >= 1 - EPS for key in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"))
        if state.validity_gate:
            state.metrics = _metrics(state.replayed, state.records)
        _CACHE[root] = state
        return state


async def _gate(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(key, 0.0), state.diagnostics.get(key, "")


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
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, "official run passed") if state.official_ok else (0.0, state.diagnostics.get("official", "official run failed"))


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
    score, missing = _report_score(text)
    return score, f"missing_or_empty={missing}"


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]: return await _gate(ws, "result_json")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]: return await _gate(ws, "result_evidence")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace) -> tuple[float, str]: return await _gate(ws, "canonical_coverage")


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]: return await _gate(ws, "decision_legality")


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]: return await _gate(ws, "replay_validity")


async def _metric(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{key}: validity gate not satisfied"
    return state.metrics[key], f"{key}: replayed={state.metrics[key]:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]: return await _metric(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]: return await _metric(ws, "primary_metrics_2")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]: return await _metric(ws, "primary_metrics_3")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]: return await _metric(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]: return await _metric(ws, "auxiliary_metrics_2")
