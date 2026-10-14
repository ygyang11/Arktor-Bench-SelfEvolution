# ruff: noqa
"""Independent decision replay grader for T057."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y, H = 24, 18, 96
AUVS = ("A0", "A1", "A2")
SCENARIOS = ("flat_ridge", "steep_ravine", "orientation_shift", "acoustic_gap", "overlap_chain")
GATEWAY = (12, 0)
OPPORTUNITY_SLOTS = tuple(range(7, 88, 8))
OPPORTUNITY_SET = set(OPPORTUNITY_SLOTS)
BLOCK_IDS = tuple(f"B{i:02d}" for i in range(16))
BLOCK_SET = set(BLOCK_IDS)
ACTION_DELTAS = {"stay": (0, 0), "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
STARTS = {
    "flat_ridge": ((1, 2), (22, 2), (1, 15)),
    "steep_ravine": ((1, 1), (22, 1), (1, 16)),
    "orientation_shift": ((2, 2), (21, 2), (2, 15)),
    "acoustic_gap": ((1, 3), (22, 3), (1, 14)),
    "overlap_chain": ((1, 1), (22, 1), (1, 16)),
}
OBSTACLES = {
    "flat_ridge": (),
    "steep_ravine": ((10, 13, 7, 10),),
    "orientation_shift": ((7, 9, 7, 10),),
    "acoustic_gap": ((9, 14, 5, 12),),
    "overlap_chain": ((11, 12, 4, 13),),
}
BASE_GATES = {
    "flat_ridge": ((4, 4), (8, 8), (12, 8), (16, 12), (20, 14), (22, 16)),
    "steep_ravine": ((4, 3), (8, 6), (9, 13), (15, 13), (19, 8), (22, 15)),
    "orientation_shift": ((5, 3), (6, 12), (11, 14), (15, 10), (19, 5), (22, 14)),
    "acoustic_gap": ((5, 4), (7, 13), (12, 14), (16, 4), (20, 8), (22, 15)),
    "overlap_chain": ((4, 5), (8, 5), (10, 14), (15, 14), (19, 9), (22, 16)),
}
GATE_DEADLINES = (20, 34, 48, 62, 76, 88)
ENERGY_BUDGET = {"flat_ridge": 115.0, "steep_ravine": 120.0, "orientation_shift": 122.0, "acoustic_gap": 118.0, "overlap_chain": 125.0}
REPORT_SECTIONS = ("Problem Formulation", "Method Design", "Experimental Setup", "Results", "Research Analysis", "Conclusion and Limitations")
SUBMAP_RE = re.compile(r"^S_A[0-2]_(?:7|15|23|31|39|47|55|63|71|79|87)$")

ATOM_SHARES = {
    "result_json": {"result_json.parse": .20, "result_json.root": .20, "result_json.scenarios": .25, "result_json.records": .35},
    "result_evidence": {"result_evidence.slots": .20, "result_evidence.auvs": .30, "result_evidence.sets": .20, "result_evidence.channel": .18, "result_evidence.gateway": .12},
    "decision_legality": {"decision_legality.actions": .18, "decision_legality.block_ids": .22, "decision_legality.opportunities": .25, "decision_legality.channel": .20, "decision_legality.gateway_id": .15},
    "replay_validity": {"replay_validity.submaps": .34, "replay_validity.transport": .36, "replay_validity.gateway": .30},
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, ident: str, passed: bool, detail: str, share: float) -> None:
        self.id, self.passed, self.detail, self.share = ident, bool(passed), detail, share


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
    return not (0 <= x < GRID_X and 0 <= y < GRID_Y) or any(
        x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in OBSTACLES[name]
    )


def _depth(name: str, x: int, y: int, slot: int) -> float:
    if name == "flat_ridge":
        return 35 + .25 * x + .15 * y + (3 if x >= 12 else 0)
    if name == "steep_ravine":
        return 30 + .2 * y + 8 * abs(x - 12) / 12
    if name == "orientation_shift":
        return 38 + 4 * math.sin((x + y) / 3) + 4 * math.cos((x - y + slot / 24) / 4)
    if name == "acoustic_gap":
        return 32 + .18 * x + .22 * y + (3 if y >= 9 else 0)
    return 36 + 3 * math.sin(x / 3) + 3 * math.cos(y / 2)


def _gates(name: str, auv: str) -> tuple[tuple[int, int], ...]:
    base = BASE_GATES[name]
    if auv == "A1":
        return tuple((23 - x, y) for x, y in base)
    if auv == "A2":
        return tuple((x, 17 - y) for x, y in base)
    return base


def _gap(name: str, slot: int) -> tuple[float, bool]:
    if name == "orientation_shift" and 32 <= slot <= 55:
        return .25, False
    if name == "acoustic_gap" and 28 <= slot <= 51:
        return 1.0, True
    if name == "overlap_chain" and 40 <= slot <= 71:
        return .25, False
    return 0.0, False


def _probe(name: str, position: tuple[int, int], slot: int) -> dict[str, Any]:
    z = _depth(name, position[0], position[1], slot)
    level = int(_clip(math.floor((z - 30) / 5), 0, 3))
    gap, interrupted = _gap(name, slot)
    q = _clip(1 - math.dist(position, GATEWAY) / 32 - .04 * level - .15 * gap)
    cap = 0
    if not interrupted:
        for value in (64, 96, 128):
            if value <= 128 * q + EPS:
                cap = value
    return {"q": q, "available": cap > 0, "capacity": cap}


COORDS = ((0, 0), (1, 0), (0, 1), (1, 1))
COORD_INDEX = {coord: i for i, coord in enumerate(COORDS)}


def _transform(coord: tuple[int, int], orientation: int) -> tuple[int, int]:
    u, v = coord
    for _ in range(orientation % 4):
        u, v = v, 1 - u
    if orientation >= 4:
        u, v = 1 - u, v
    return u, v


def _reconstruct(center: int, index: int, orientation: int) -> tuple[float, ...]:
    code = [(2 * ((index >> j) & 1) - 1) / 2 for j in range(4)]
    permuted = [0.0] * 4
    for j, coord in enumerate(COORDS):
        permuted[COORD_INDEX[_transform(coord, orientation)]] = code[j]
    return tuple(center / 10 + 2 * value for value in permuted)


@lru_cache(maxsize=8192)
def _encode(values: tuple[float, ...]) -> tuple[int, int, int]:
    center = int(_clip(math.floor(10 * (sum(values) / 4) + .5), 0, 1023))
    best: tuple[float, int, int] | None = None
    for index in range(16):
        for orientation in range(8):
            rec = _reconstruct(center, index, orientation)
            mse = sum((float(values[j]) - rec[j]) ** 2 for j in range(4)) / 4
            candidate = (mse, index, orientation)
            if best is None or mse < best[0] - EPS:
                best = candidate
    assert best is not None
    return center, best[1], best[2]


def _submap(name: str, auv: str, slot: int, position: tuple[int, int], estimate: tuple[float, float]) -> dict[str, Any]:
    ox = int(_clip(position[0] - 3, 0, 16))
    oy = int(_clip(position[1] - 3, 0, 10))
    blocks: dict[str, dict[str, Any]] = {}
    for by in range(4):
        for bx in range(4):
            ident = f"B{4 * by + bx:02d}"
            cells = ((ox + 2 * bx, oy + 2 * by), (ox + 2 * bx + 1, oy + 2 * by), (ox + 2 * bx, oy + 2 * by + 1), (ox + 2 * bx + 1, oy + 2 * by + 1))
            values = tuple(_depth(name, x, y, slot) for x, y in cells)
            center, index, orientation = _encode(values)
            blocks[ident] = {"id": ident, "cells": cells, "values": values, "center": center, "index": index, "orientation": orientation}
    return {
        "id": f"S_{auv}_{slot}", "auv": auv, "slot": slot, "origin": (ox, oy),
        "cells": {(x, y) for x in range(ox, ox + 8) for y in range(oy, oy + 8)},
        "true_capture": position, "estimate_capture": estimate, "blocks": blocks,
        "selected_blocks": [], "orientation_blocks": [],
    }


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _sign(value: float) -> int:
    return 0 if abs(value) <= EPS else 1 if value > 0 else -1


class ScenarioEngine:
    """Pure public-model environment shared by replay and author-side policies."""

    def __init__(self, name: str) -> None:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}")
        self.name, self.slot, self.prepared = name, 0, False
        self.true = {auv: STARTS[name][i] for i, auv in enumerate(AUVS)}
        self.estimate = {auv: tuple(map(float, STARTS[name][i])) for i, auv in enumerate(AUVS)}
        self.gate_index = {auv: 0 for auv in AUVS}
        self.energy = {auv: 0.0 for auv in AUVS}
        self.gateway_energy = 0.0
        self.upload = {auv: [] for auv in AUVS}
        self.downlink = {auv: [] for auv in AUVS}
        self.uplink_transit: list[dict[str, Any]] = []
        self.downlink_transit: list[dict[str, Any]] = []
        self.gateway_queue: list[dict[str, Any]] = []
        self.active_job: dict[str, Any] | None = None
        self.opportunities: list[dict[str, Any]] = []
        self.current_opportunity: dict[str, dict[str, Any]] = {}
        self.submap_packets: dict[str, dict[str, Any]] = {}
        self.correction_packets: dict[str, dict[str, Any]] = {}
        self.restored: dict[str, dict[str, Any]] = {}
        self.matches: dict[str, dict[str, Any]] = {}
        self.accepted_matches: list[dict[str, Any]] = []
        self.applied_corrections: list[dict[str, Any]] = []
        self.service_events: list[dict[str, Any]] = []
        self.compute_events: list[dict[str, Any]] = []
        self.uplink_bits = 0
        self.safety_events = 0
        self.executed_moves = 0
        self.pose_errors = [math.dist(self.true[auv], self.estimate[auv]) for auv in AUVS]
        self.previous_action = {auv: {"requested": "stay", "executed": "stay", "rejected": False} for auv in AUVS}
        self.errors = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}

    def _error(self, key: str, detail: str) -> None:
        if len(self.errors[key]) < 8:
            self.errors[key].append(detail)

    def _add_match_opportunities(self) -> None:
        for i, first in enumerate(self.opportunities):
            for second in self.opportunities[i + 1:]:
                if first["auv"] == second["auv"] or abs(first["slot"] - second["slot"]) > 16:
                    continue
                overlap = first["cells"] & second["cells"]
                if len(overlap) < 8:
                    continue
                earlier, later = sorted((first, second), key=lambda item: (item["slot"], item["auv"], item["id"]))
                ident = f"M_{later['id']}__{earlier['id']}"
                if ident not in self.matches:
                    self.matches[ident] = {"id": ident, "earlier": earlier, "later": later, "overlap": set(overlap), "deadline": later["slot"] + 24, "tested": False, "accepted": False}

    def _prepare(self, terminal: bool = False) -> None:
        if self.prepared:
            return
        t = self.slot
        arrivals = sorted((p for p in self.uplink_transit if p.get("arrival_slot") == t), key=lambda p: (p["arrival_slot"], p["source"], p["submap_id"]))
        self.uplink_transit = [p for p in self.uplink_transit if p.get("arrival_slot") != t]
        for packet in arrivals:
            if t - packet["created_slot"] > 24:
                packet["status"] = "expired_in_transit"
                continue
            if len(self.gateway_queue) + (1 if self.active_job is not None else 0) >= 8:
                packet["status"] = "gateway_overflow"
                continue
            packet["status"] = "gateway_waiting"
            self.gateway_queue.append(packet)
        self.gateway_queue.sort(key=lambda p: (p["arrival_slot"], p["source"], p["submap_id"]))

        corr_arrivals = sorted((p for p in self.downlink_transit if p.get("arrival_slot") == t), key=lambda p: p["id"])
        self.downlink_transit = [p for p in self.downlink_transit if p.get("arrival_slot") != t]
        for packet in corr_arrivals:
            if t - packet["created_slot"] > 24:
                packet["status"] = "expired_in_transit"
                continue
            auv = packet["target"]
            self.estimate[auv] = (self.estimate[auv][0] + packet["dx"], self.estimate[auv][1] + packet["dy"])
            packet["status"] = "applied"
            packet["applied_slot"] = t
            self.applied_corrections.append(packet)

        for auv in AUVS:
            kept = []
            for packet in self.upload[auv]:
                if t - packet["created_slot"] > 24:
                    packet["status"] = "expired_uplink"
                else:
                    kept.append(packet)
            self.upload[auv] = kept
            kept = []
            for packet in self.downlink[auv]:
                if t - packet["created_slot"] > 24:
                    packet["status"] = "expired_downlink"
                else:
                    kept.append(packet)
            self.downlink[auv] = kept
        self.gateway_queue = [p for p in self.gateway_queue if t - p["created_slot"] <= 24]
        if self.active_job is not None and t - self.active_job["created_slot"] > 24:
            self.active_job["status"] = "expired_gateway"
            self.active_job = None

        self.current_opportunity = {}
        if not terminal and t in OPPORTUNITY_SET:
            for auv in AUVS:
                opportunity = _submap(self.name, auv, t, self.true[auv], self.estimate[auv])
                self.current_opportunity[auv] = opportunity
                self.opportunities.append(opportunity)
            self._add_match_opportunities()
        self.prepared = True

    def context(self) -> dict[str, Any]:
        self._prepare()
        robots = {}
        for auv in AUVS:
            probe = _probe(self.name, self.true[auv], self.slot)
            opportunity = self.current_opportunity.get(auv)
            robots[auv] = {
                "estimate": self.estimate[auv], "gate_index": self.gate_index[auv],
                "gates": tuple(zip(_gates(self.name, auv), GATE_DEADLINES)),
                "energy": self.energy[auv], "previous_action": self.previous_action[auv],
                "upload_queue": [{k: p[k] for k in ("id", "payload_bits", "remaining_bits", "created_slot", "retries", "status")} for p in self.upload[auv]],
                "applied_corrections": [{k: p[k] for k in ("id", "created_slot", "applied_slot", "dx", "dy")} for p in self.applied_corrections if p["target"] == auv],
                "link_probe": copy.deepcopy(probe),
                "opportunity": None if opportunity is None else {
                    "submap_id": opportunity["id"],
                    "blocks": [{"block_id": bid, "cell_ids": [f"C_{x}_{y}" for x, y in opportunity["blocks"][bid]["cells"]], "values": list(opportunity["blocks"][bid]["values"]), "center_code": opportunity["blocks"][bid]["center"], "index_code": opportunity["blocks"][bid]["index"], "direction_code": opportunity["blocks"][bid]["orientation"]} for bid in BLOCK_IDS],
                },
            }
        channel_queues = []
        for direction, queues in (("uplink", self.upload), ("downlink", self.downlink)):
            for auv in AUVS:
                head = queues[auv][0] if queues[auv] else None
                channel_queues.append({"direction": direction, "auv_id": auv, "queue_length": len(queues[auv]), "head": None if head is None else {k: head[k] for k in ("id", "payload_bits", "remaining_bits", "created_slot", "retries", "status")}})
        return copy.deepcopy({
            "scenario": self.name, "slot": self.slot, "robots": robots,
            "channel": {"uplink_bits": self.uplink_bits, "queues": channel_queues},
            "gateway": {
                "active": None if self.active_job is None else {"submap_id": self.active_job["submap_id"], "source": self.active_job["source"], "origin": self.active_job["opportunity"]["origin"], "selected_block_ids": list(self.active_job["selected_blocks"]), "encoded_blocks": [{"block_id": bid, "center_code": self.active_job["opportunity"]["blocks"][bid]["center"], "index_code": self.active_job["opportunity"]["blocks"][bid]["index"], "direction_code": self.active_job["opportunity"]["blocks"][bid]["orientation"] if bid in self.active_job["orientation_blocks"] else None} for bid in self.active_job["selected_blocks"]], "block_id": self.active_job["selected_blocks"][self.active_job["block_cursor"]] if self.active_job["block_cursor"] < len(self.active_job["selected_blocks"]) else None, "remaining_block_units": self.active_job.get("block_remaining", 0)},
                "waiting": [{"submap_id": p["submap_id"], "source": p["source"], "arrival_slot": p["arrival_slot"], "created_slot": p["created_slot"], "remaining_units": p["remaining_work"], "origin": p["opportunity"]["origin"], "selected_block_ids": list(p["selected_blocks"]), "encoded_blocks": [{"block_id": bid, "center_code": p["opportunity"]["blocks"][bid]["center"], "index_code": p["opportunity"]["blocks"][bid]["index"], "direction_code": p["opportunity"]["blocks"][bid]["orientation"] if bid in p["orientation_blocks"] else None} for bid in p["selected_blocks"]]} for p in self.gateway_queue],
                "restored": [{"submap_id": sid, "source": p["source"], "origin": p["opportunity"]["origin"], "selected_block_ids": list(p["selected_blocks"]), "encoded_blocks": [{"block_id": bid, "center_code": p["opportunity"]["blocks"][bid]["center"], "index_code": p["opportunity"]["blocks"][bid]["index"], "direction_code": p["opportunity"]["blocks"][bid]["orientation"] if bid in p["orientation_blocks"] else None} for bid in p["selected_blocks"]], "restored_blocks": [{"block_id": bid, "values": list(p["restored_blocks"][bid]["values"]), "used_direction_code": p["restored_blocks"][bid]["used_orientation"]} for bid in p["selected_blocks"]]} for sid, p in sorted(self.restored.items())],
                "completed_matches": [{"match_id": m["id"], "tested_slot": m.get("tested_slot"), "accepted": m["accepted"], "depth_rmse": m.get("depth_rmse"), "gradient_agreement": m.get("gradient_agreement")} for m in sorted(self.matches.values(), key=lambda item: item["id"]) if m["tested"]],
                "compute_units": sum(e["units"] for e in self.compute_events),
            },
        })

    def _serve(self, direction: str, auv: str) -> None:
        queues = self.upload if direction == "uplink" else self.downlink
        probe = _probe(self.name, self.true[auv], self.slot)
        cap = int(probe["capacity"])
        if direction == "uplink":
            cap = min(cap, 7200 - self.uplink_bits)
        if not queues[auv] or cap <= 0:
            self._error("transport", f"slot {self.slot}: {direction} {auv} has no serviceable FIFO head")
            return
        packet = queues[auv][0]
        served = min(cap, packet["remaining_bits"])
        packet["remaining_bits"] -= served
        packet["served_bits_total"] += served
        self.service_events.append({"packet_id": packet["id"], "kind": direction, "bits": served})
        if direction == "uplink":
            self.uplink_bits += served
            self.energy[auv] += .0005 * served
        if packet["remaining_bits"]:
            return
        index = AUVS.index(auv)
        kind = 0 if direction == "uplink" else 1
        success = ((13 * index + 7 * self.slot + kind) % 100) / 100 >= .08 + .2 * (1 - probe["q"])
        if not success:
            packet["remaining_bits"] = packet["payload_bits"]
            packet["retries"] += 1
            packet["status"] = "queued_after_failure"
            return
        queues[auv].pop(0)
        packet["completion_slot"] = self.slot
        packet["arrival_slot"] = self.slot + 2 + math.floor(4 * (1 - probe["q"]))
        packet["status"] = "in_transit"
        (self.uplink_transit if direction == "uplink" else self.downlink_transit).append(packet)

    @staticmethod
    def _edge_cost(values: Sequence[float], neighbour: Sequence[float], side: str) -> float:
        if side == "left":
            pairs = ((0, 1), (2, 3))
        elif side == "right":
            pairs = ((1, 0), (3, 2))
        elif side == "lower":
            pairs = ((0, 2), (1, 3))
        else:
            pairs = ((2, 0), (3, 1))
        return sum((values[a] - neighbour[b]) ** 2 for a, b in pairs)

    def _infer_orientation(self, packet: Mapping[str, Any], block_id: str) -> int:
        index = int(block_id[1:])
        bx, by = index % 4, index // 4
        neighbours = []
        for dx, dy, side in ((-1, 0, "left"), (1, 0, "right"), (0, -1, "lower"), (0, 1, "upper")):
            nx, ny = bx + dx, by + dy
            ident = f"B{4 * ny + nx:02d}" if 0 <= nx < 4 and 0 <= ny < 4 else ""
            if ident in packet["restored_blocks"]:
                neighbours.append((packet["restored_blocks"][ident]["values"], side))
        if not neighbours:
            return 0
        block = packet["opportunity"]["blocks"][block_id]
        best_o, best_cost = 0, math.inf
        for orientation in range(8):
            values = _reconstruct(block["center"], block["index"], orientation)
            cost = sum(self._edge_cost(values, neighbour, side) for neighbour, side in neighbours)
            if cost < best_cost - EPS:
                best_o, best_cost = orientation, cost
        return best_o

    def _compute(self) -> None:
        packet = self.active_job
        if packet is None:
            return
        budget = 48
        while budget > 0 and packet["block_cursor"] < len(packet["selected_blocks"]):
            block_id = packet["selected_blocks"][packet["block_cursor"]]
            if packet.get("block_remaining", 0) <= 0:
                packet["block_remaining"] = 2 + (0 if block_id in packet["orientation_blocks"] else 6)
            used = min(budget, packet["block_remaining"])
            packet["block_remaining"] -= used
            packet["remaining_work"] -= used
            budget -= used
            self.gateway_energy += .01 * used
            self.compute_events.append({"submap_id": packet["submap_id"], "block_id": block_id, "units": used})
            if packet["block_remaining"] > 0:
                break
            block = packet["opportunity"]["blocks"][block_id]
            fallback = self._infer_orientation(packet, block_id)
            used_orientation = block["orientation"] if block_id in packet["orientation_blocks"] else fallback
            values = _reconstruct(block["center"], block["index"], used_orientation)
            packet["restored_blocks"][block_id] = {"values": values, "fallback_orientation": fallback, "used_orientation": used_orientation, "direction_sent": block_id in packet["orientation_blocks"], "workload": 2 + (0 if block_id in packet["orientation_blocks"] else 6)}
            packet["block_cursor"] += 1
        if packet["block_cursor"] == len(packet["selected_blocks"]):
            packet["status"] = "restored"
            packet["restored_slot"] = self.slot
            self.restored[packet["submap_id"]] = packet
            self.active_job = None
            self._evaluate_matches()

    @staticmethod
    def _cell_map(packet: Mapping[str, Any]) -> dict[tuple[int, int], float]:
        out = {}
        for block_id, restored in packet["restored_blocks"].items():
            block = packet["opportunity"]["blocks"][block_id]
            out.update(zip(block["cells"], restored["values"]))
        return out

    def _evaluate_matches(self) -> None:
        for match in sorted(self.matches.values(), key=lambda item: item["id"]):
            if match["tested"]:
                continue
            earlier_id, later_id = match["earlier"]["id"], match["later"]["id"]
            if earlier_id not in self.restored or later_id not in self.restored:
                continue
            match["tested"] = True
            match["tested_slot"] = self.slot
            if self.slot > match["deadline"]:
                continue
            first, second = self._cell_map(self.restored[earlier_id]), self._cell_map(self.restored[later_id])
            overlap = sorted(match["overlap"] & set(first) & set(second))
            if len(overlap) < 8:
                continue
            depth_rmse = _rmse([first[cell] - second[cell] for cell in overlap])
            signs = []
            overlap_set = set(overlap)
            for x, y in overlap:
                for neighbour in ((x + 1, y), (x, y + 1)):
                    if neighbour in overlap_set:
                        signs.append(_sign(first[neighbour] - first[(x, y)]) == _sign(second[neighbour] - second[(x, y)]))
            agreement = sum(signs) / len(signs) if signs else 0.0
            match["depth_rmse"], match["gradient_agreement"] = depth_rmse, agreement
            if depth_rmse > 3 + EPS or agreement < .7 - EPS:
                continue
            match["accepted"] = True
            self.accepted_matches.append(match)
            earlier, later = match["earlier"], match["later"]
            rx = (later["estimate_capture"][0] - earlier["estimate_capture"][0]) - (later["true_capture"][0] - earlier["true_capture"][0])
            ry = (later["estimate_capture"][1] - earlier["estimate_capture"][1]) - (later["true_capture"][1] - earlier["true_capture"][1])
            target = later["auv"]
            ident = f"C_{later['id']}__{earlier['id']}"
            correction = {"id": ident, "kind": "downlink", "target": target, "created_slot": self.slot, "payload_bits": 96, "remaining_bits": 96, "retries": 0, "served_bits_total": 0, "dx": _clip(-.5 * rx, -3, 3), "dy": _clip(-.5 * ry, -3, 3), "status": "queued"}
            self.correction_packets[ident] = correction
            if len(self.downlink[target]) < 6:
                self.downlink[target].append(correction)
            else:
                correction["status"] = "downlink_overflow"

    def _motion(self, decisions: Mapping[str, Mapping[str, Any]]) -> None:
        requested = {auv: decisions.get(auv, {}).get("action", "stay") for auv in AUVS}
        requested = {auv: action if isinstance(action, str) and action in ACTION_DELTAS else "stay" for auv, action in requested.items()}
        proposed: dict[str, tuple[int, int]] = {}
        moving: set[str] = set()
        for auv in AUVS:
            action = requested[auv]
            dx, dy = ACTION_DELTAS[action]
            target = (self.true[auv][0] + dx, self.true[auv][1] + dy)
            if action != "stay" and not _wall(self.name, target) and abs(_depth(self.name, target[0], target[1], self.slot) - _depth(self.name, self.true[auv][0], self.true[auv][1], self.slot)) <= 4 + EPS:
                proposed[auv] = target
                moving.add(auv)
            else:
                proposed[auv] = self.true[auv]
        for i, first in enumerate(AUVS):
            for second in AUVS[i + 1:]:
                if first in moving and second in moving and proposed[first] == self.true[second] and proposed[second] == self.true[first]:
                    moving.discard(first)
                    moving.discard(second)
        groups: dict[tuple[int, int], list[str]] = defaultdict(list)
        for auv in moving:
            groups[proposed[auv]].append(auv)
        for ids in groups.values():
            for auv in sorted(ids)[1:]:
                moving.discard(auv)
        changed = True
        while changed:
            changed = False
            occupied = {self.true[auv] for auv in AUVS if auv not in moving}
            for auv in tuple(moving):
                if proposed[auv] in occupied:
                    moving.remove(auv)
                    changed = True
        for auv in AUVS:
            executed = requested[auv] if auv in moving else "stay"
            if requested[auv] != "stay" and executed == "stay":
                self.safety_events += 1
            if auv in moving:
                old = self.true[auv]
                self.true[auv] = proposed[auv]
                dx, dy = self.true[auv][0] - old[0], self.true[auv][1] - old[1]
                index = AUVS.index(auv)
                self.estimate[auv] = (self.estimate[auv][0] + dx + .04 * ((-1) ** index), self.estimate[auv][1] + dy + .04 * ((-1) ** (index + 1)))
                self.energy[auv] += 1
                self.executed_moves += 1
            self.previous_action[auv] = {"requested": requested[auv], "executed": executed, "rejected": requested[auv] != "stay" and executed == "stay"}

    def step(self, row: Mapping[str, Any]) -> None:
        self._prepare()
        t = self.slot
        raw = row.get("auv_decisions", []) if isinstance(row, Mapping) else []
        decisions = {item.get("auv_id"): item for item in raw if isinstance(item, Mapping) and item.get("auv_id") in AUVS}
        start_upload_lengths = {auv: len(self.upload[auv]) for auv in AUVS}

        channel = row.get("channel", {}) if isinstance(row, Mapping) else {}
        direction = channel.get("direction") if isinstance(channel, Mapping) else None
        channel_auv = channel.get("auv_id") if isinstance(channel, Mapping) else None
        if direction in {"uplink", "downlink"} and channel_auv in AUVS:
            self._serve(direction, channel_auv)

        selected_job = row.get("gateway_submap_id") if isinstance(row, Mapping) else None
        if self.active_job is not None:
            if selected_job is not None:
                self._error("gateway", f"slot {t}: gateway selection must be null while a job is active")
        elif isinstance(selected_job, str):
            matches = [packet for packet in self.gateway_queue if packet["submap_id"] == selected_job]
            if len(matches) != 1:
                self._error("gateway", f"slot {t}: submap {selected_job!r} is not in the gateway waiting queue")
            else:
                self.active_job = matches[0]
                self.gateway_queue.remove(matches[0])
                self.active_job["status"] = "gateway_active"
        self._compute()

        self._motion(decisions)

        for auv in AUVS:
            opportunity = self.current_opportunity.get(auv)
            if opportunity is None:
                continue
            self.energy[auv] += .1
            decision = decisions.get(auv, {})
            selected = decision.get("selected_block_ids", [])
            orientations = decision.get("orientation_block_ids", [])
            if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
                selected = []
            if not isinstance(orientations, list) or not all(isinstance(item, str) for item in orientations):
                orientations = []
            selected = sorted(set(selected) & BLOCK_SET)
            orientations = sorted(set(orientations) & set(selected))
            opportunity["selected_blocks"], opportunity["orientation_blocks"] = selected, orientations
            if not selected:
                continue
            if start_upload_lengths[auv] >= 4:
                self._error("submaps", f"slot {t} {auv}: nonempty submap selection with full decision-boundary upload FIFO")
                continue
            payload = 48 + 14 * len(selected) + 3 * len(orientations)
            work = sum(2 + (0 if block in orientations else 6) for block in selected)
            packet = {
                "id": f"U_{auv}_{t}", "kind": "uplink", "source": auv, "submap_id": opportunity["id"],
                "created_slot": t, "payload_bits": payload, "remaining_bits": payload, "retries": 0,
                "served_bits_total": 0, "status": "queued", "opportunity": opportunity,
                "selected_blocks": selected, "orientation_blocks": orientations, "remaining_work": work,
                "block_cursor": 0, "block_remaining": 0, "restored_blocks": {},
            }
            self.upload[auv].append(packet)
            self.submap_packets[opportunity["id"]] = packet

        boundary = t + 1
        for auv in AUVS:
            index = self.gate_index[auv]
            if index >= 6 or boundary > GATE_DEADLINES[index]:
                continue
            target = _gates(self.name, auv)[index]
            if max(abs(self.true[auv][0] - target[0]), abs(self.true[auv][1] - target[1])) <= 1 and math.dist(self.true[auv], self.estimate[auv]) <= 2 + EPS:
                self.gate_index[auv] += 1
        self.pose_errors.extend(math.dist(self.true[auv], self.estimate[auv]) for auv in AUVS)
        self.slot += 1
        self.prepared = False

    def result(self) -> dict[str, Any]:
        if self.slot == H:
            self._prepare(terminal=True)
            self.pose_errors[-len(AUVS):] = [math.dist(self.true[auv], self.estimate[auv]) for auv in AUVS]
        if self.uplink_bits > 7200:
            self._error("transport", f"uplink budget exceeded: {self.uplink_bits}")
        if any(len(self.upload[auv]) > 4 or len(self.downlink[auv]) > 6 for auv in AUVS):
            self._error("transport", "a packet FIFO exceeds its public capacity")
        if len(self.gateway_queue) + (1 if self.active_job is not None else 0) > 8:
            self._error("gateway", "gateway system exceeds capacity eight")
        return {
            "true": self.true, "estimate": self.estimate, "gate_index": self.gate_index,
            "energy": sum(self.energy.values()) + self.gateway_energy, "safety_events": self.safety_events,
            "executed_moves": self.executed_moves, "pose_errors": self.pose_errors,
            "opportunities": self.opportunities, "submap_packets": self.submap_packets,
            "correction_packets": self.correction_packets, "restored": self.restored,
            "matches": self.matches, "accepted_matches": self.accepted_matches,
            "applied_corrections": self.applied_corrections, "service_events": self.service_events,
            "compute_events": self.compute_events, "uplink_bits": self.uplink_bits,
            "errors": self.errors,
        }


def generate_scenario(name: str, policy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    engine = ScenarioEngine(name)
    slots = []
    for _ in range(H):
        row = policy(engine.context())
        engine.step(row)
        slots.append(row)
    return {"scenario": name, "slots": slots}, engine.result()


def _result_atoms(data: Any, parse_ok: bool = True) -> list[AtomVerdict]:
    checks: dict[str, tuple[bool, str]] = {}
    checks["parse"] = (parse_ok, "ok" if parse_ok else "strict JSON parsing failed")
    checks["root"] = (isinstance(data, dict) and set(data) == {"scenarios"}, "root must be exactly {'scenarios'}")
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    scenario_ok = isinstance(scenarios, list)
    scenario_detail = "ok"
    if scenario_ok:
        for index, record in enumerate(scenarios):
            err = _shape(record, {"scenario", "slots"}, f"scenarios[{index}]")
            if err or not isinstance(record.get("slots"), list):
                scenario_ok, scenario_detail = False, err or f"scenarios[{index}].slots: expected array"
                break
    else:
        scenario_detail = "scenarios: expected array"
    checks["scenarios"] = (scenario_ok, scenario_detail)
    records_ok = scenario_ok
    records_detail = "ok"
    if scenario_ok:
        for si, scenario in enumerate(scenarios):
            for ri, row in enumerate(scenario["slots"]):
                if not isinstance(row, dict):
                    records_ok, records_detail = False, f"scenarios[{si}].slots[{ri}]: expected object"
                    break
            if not records_ok:
                break
    checks["records"] = (records_ok, records_detail)
    return [AtomVerdict(f"result_json.{key}", passed, detail, ATOM_SHARES["result_json"][f"result_json.{key}"]) for key, (passed, detail) in checks.items()]


def _records(data: Any) -> dict[str, dict[str, Any]]:
    out = {}
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []
    if not isinstance(scenarios, list):
        return out
    for record in scenarios:
        if isinstance(record, dict) and isinstance(record.get("scenario"), str) and record["scenario"] not in out:
            out[record["scenario"]] = record
    return out


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key.split(".", 1)[1]: [True, "ok"] for key in ATOM_SHARES["result_evidence"]}
    for name, record in records.items():
        slots = record.get("slots", []) if isinstance(record, Mapping) else []
        if not isinstance(slots, list):
            checks["slots"] = [False, f"{name}.slots: expected array"]
            continue
        for ri, row in enumerate(slots):
            path = f"{name}.slots[{ri}]"
            err = _shape(row, {"slot", "auv_decisions", "channel", "gateway_submap_id"}, path)
            if err:
                checks["slots"] = [False, err]
                continue
            if not isinstance(row.get("slot"), int) or isinstance(row.get("slot"), bool) or not isinstance(row.get("auv_decisions"), list):
                checks["slots"] = [False, f"{path}: slot must be a non-boolean integer and auv_decisions an array"]
            for ai, decision in enumerate(row.get("auv_decisions", []) if isinstance(row.get("auv_decisions"), list) else []):
                dpath = f"{path}.auv_decisions[{ai}]"
                derr = _shape(decision, {"auv_id", "action", "selected_block_ids", "orientation_block_ids"}, dpath)
                if derr:
                    checks["auvs"] = [False, derr]
                    continue
                if not isinstance(decision.get("auv_id"), str) or not isinstance(decision.get("action"), str):
                    checks["auvs"] = [False, f"{dpath}: auv_id and action must be strings"]
                for field in ("selected_block_ids", "orientation_block_ids"):
                    value = decision.get(field)
                    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                        checks["sets"] = [False, f"{dpath}.{field}: expected an array of strings"]
            channel = row.get("channel")
            cerr = _shape(channel, {"direction", "auv_id"}, f"{path}.channel")
            if cerr:
                checks["channel"] = [False, cerr]
            elif not (channel.get("direction") is None or isinstance(channel.get("direction"), str)) or not (channel.get("auv_id") is None or isinstance(channel.get("auv_id"), str)):
                checks["channel"] = [False, f"{path}.channel: fields must be strings or null"]
            gateway = row.get("gateway_submap_id")
            if gateway is not None and not isinstance(gateway, str):
                checks["gateway"] = [False, f"{path}.gateway_submap_id: expected string or null"]
    return [AtomVerdict(f"result_evidence.{key}", value[0], value[1], ATOM_SHARES["result_evidence"][f"result_evidence.{key}"]) for key, value in checks.items()]


def _typed_token(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _coverage(data: Any) -> tuple[float, list[str]]:
    defects = []
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []
    if not isinstance(scenarios, list):
        return 0.0, ["scenarios is not an array"]
    expected = {_typed_token(name) for name in SCENARIOS}
    actual = [_typed_token(row.get("scenario")) for row in scenarios if isinstance(row, dict)]
    counts = Counter(actual)
    for token in sorted(expected - set(counts)): defects.append(f"missing scenario {token[1]}")
    for token in sorted(set(counts) - expected): defects.append(f"unknown scenario {token[1]}")
    for token, count in counts.items():
        if token in expected and count > 1: defects.append(f"duplicate scenario {token[1]} count={count}")
    for name in SCENARIOS:
        matches = [r for r in scenarios if isinstance(r, dict) and type(r.get("scenario")) is str and r.get("scenario") == name]
        if len(matches) != 1 or not isinstance(matches[0].get("slots"), list): continue
        slot_counts = Counter(_typed_token(r.get("slot")) for r in matches[0]["slots"] if isinstance(r, dict))
        expected_slots = {_typed_token(slot) for slot in range(H)}
        for token in expected_slots - set(slot_counts): defects.append(f"{name}: missing slot {token[1]}")
        for token in set(slot_counts) - expected_slots: defects.append(f"{name}: unknown slot {token[1]}")
        for token, count in slot_counts.items():
            if token in expected_slots and count > 1: defects.append(f"{name}: duplicate slot {token[1]} count={count}")
        for row in matches[0]["slots"]:
            if not isinstance(row, dict) or not _integer(row.get("slot"), 0, H - 1) or not isinstance(row.get("auv_decisions"), list): continue
            auv_counts = Counter(_typed_token(d.get("auv_id")) for d in row["auv_decisions"] if isinstance(d, dict))
            expected_auvs = {_typed_token(auv) for auv in AUVS}
            for token in expected_auvs - set(auv_counts): defects.append(f"{name} slot {row['slot']}: missing AUV {token[1]}")
            for token in set(auv_counts) - expected_auvs: defects.append(f"{name} slot {row['slot']}: unknown AUV {token[1]}")
            for token, count in auv_counts.items():
                if token in expected_auvs and count > 1: defects.append(f"{name} slot {row['slot']}: duplicate AUV {token[1]} count={count}")
    return max(0.0, 4.0 - .5 * len(defects)) / 4.0, defects[:40]


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {key.split(".", 1)[1]: [True, "ok"] for key in ATOM_SHARES["decision_legality"]}
    for name, record in records.items():
        for row in record.get("slots", []) if isinstance(record.get("slots"), list) else []:
            if not isinstance(row, dict): continue
            slot = row.get("slot")
            for decision in row.get("auv_decisions", []) if isinstance(row.get("auv_decisions"), list) else []:
                if not isinstance(decision, dict): continue
                auv, action = decision.get("auv_id"), decision.get("action")
                if isinstance(action, str) and action not in ACTION_DELTAS:
                    checks["actions"] = [False, f"{name} slot {slot} {auv}: unknown action {action!r}"]
                selected, orientations = decision.get("selected_block_ids"), decision.get("orientation_block_ids")
                if isinstance(selected, list) and all(isinstance(x, str) for x in selected):
                    if len(selected) != len(set(selected)) or any(x not in BLOCK_SET for x in selected):
                        checks["block_ids"] = [False, f"{name} slot {slot} {auv}: invalid or duplicate selected block IDs"]
                if isinstance(orientations, list) and all(isinstance(x, str) for x in orientations):
                    if len(orientations) != len(set(orientations)) or any(x not in BLOCK_SET for x in orientations):
                        checks["block_ids"] = [False, f"{name} slot {slot} {auv}: invalid or duplicate orientation block IDs"]
                if _integer(slot, 0, H - 1) and isinstance(selected, list) and all(isinstance(x, str) for x in selected) and isinstance(orientations, list) and all(isinstance(x, str) for x in orientations):
                    if slot not in OPPORTUNITY_SET and (selected or orientations):
                        checks["opportunities"] = [False, f"{name} slot {slot} {auv}: block sets must be empty outside opportunities"]
                    elif slot in OPPORTUNITY_SET and selected and not 4 <= len(selected) <= 16:
                        checks["opportunities"] = [False, f"{name} slot {slot} {auv}: selected block count must be zero or 4..16"]
                    elif not set(orientations) <= set(selected):
                        checks["opportunities"] = [False, f"{name} slot {slot} {auv}: orientation blocks must be selected"]
            channel = row.get("channel")
            if isinstance(channel, dict):
                direction, auv = channel.get("direction"), channel.get("auv_id")
                if not ((direction is None and auv is None) or (direction in {"uplink", "downlink"} and auv in AUVS)):
                    checks["channel"] = [False, f"{name} slot {slot}: channel fields must be null together or a legal direction/AUV pair"]
            gateway = row.get("gateway_submap_id")
            if isinstance(gateway, str) and not SUBMAP_RE.fullmatch(gateway):
                checks["gateway_id"] = [False, f"{name} slot {slot}: malformed gateway submap ID {gateway!r}"]
    return [AtomVerdict(f"decision_legality.{key}", value[0], value[1], ATOM_SHARES["decision_legality"][f"decision_legality.{key}"]) for key, value in checks.items()]


def _canonical_rows(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in record.get("slots", []) if isinstance(record.get("slots"), list) else []:
        if isinstance(row, dict) and _integer(row.get("slot"), 0, H - 1) and row["slot"] not in out:
            out[row["slot"]] = row
    return out


def _blank_row(slot: int) -> dict[str, Any]:
    return {"slot": slot, "auv_decisions": [{"auv_id": auv, "action": "stay", "selected_block_ids": [], "orientation_block_ids": []} for auv in AUVS], "channel": {"direction": None, "auv_id": None}, "gateway_submap_id": None}


def _replay_atoms(records: Mapping[str, dict[str, Any]]) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    aggregate = {key.split(".", 1)[1]: [] for key in ATOM_SHARES["replay_validity"]}
    replayed = {}
    for name in SCENARIOS:
        rows = _canonical_rows(records[name]) if name in records else {}
        engine = ScenarioEngine(name)
        for slot in range(H):
            row = rows.get(slot)
            if row is None:
                for key in aggregate: aggregate[key].append(f"{name}: missing replay row for slot {slot}")
                row = _blank_row(slot)
            try:
                engine.step(row)
            except Exception as exc:
                for key in aggregate: aggregate[key].append(f"{name} slot {slot}: replay exception {type(exc).__name__}: {exc}")
                break
        replayed[name] = engine.result()
        for key, errors in engine.errors.items():
            aggregate[key].extend(f"{name}: {detail}" for detail in errors)
    atoms = [AtomVerdict(f"replay_validity.{key}", not errors, "ok" if not errors else " | ".join(errors[:8]), ATOM_SHARES["replay_validity"][f"replay_validity.{key}"]) for key, errors in aggregate.items()]
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
    if not failed: return "failed_atoms=[]"
    return "failed_atoms=[" + ",".join(atom.id for atom in failed) + "]; " + " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)


def _report_score(text: str) -> tuple[float, list[str]]:
    missing = []
    for title in REPORT_SECTIONS:
        match = re.search(rf"(?m)^(#{{1,6}})\s+{re.escape(title)}\s*$", text)
        if not match:
            missing.append(title); continue
        level, body = len(match.group(1)), []
        for line in text[match.end():].splitlines():
            heading = re.match(r"^(#{1,6})\s+", line)
            if heading and len(heading.group(1)) <= level: break
            if not heading: body.append(line)
        if not "\n".join(body).strip(): missing.append(title)
    return max(0.0, 2.0 - .5 * len(missing)) / 2.0, missing


def _block_for_cell(packet: Mapping[str, Any], cell: tuple[int, int]) -> str | None:
    for block_id, block in packet["opportunity"]["blocks"].items():
        if block_id in packet["restored_blocks"] and cell in block["cells"]:
            return block_id
    return None


def _scenario_metrics(name: str, replay: Mapping[str, Any]) -> dict[str, float]:
    gate_completion = sum(replay["gate_index"].values()) / 18
    pose_quality = _clip(1 - _rmse(replay["pose_errors"]) / 6)
    p1 = .6 * gate_completion + .4 * pose_quality

    latest: dict[tuple[int, int], tuple[tuple[Any, ...], float, float, str, str]] = {}
    productive_blocks: set[tuple[str, str]] = set()
    for submap_id, packet in replay["restored"].items():
        opportunity = packet["opportunity"]
        for block_id, restored in packet["restored_blocks"].items():
            block = opportunity["blocks"][block_id]
            for cell, value, truth in zip(block["cells"], restored["values"], block["values"]):
                # Forbidden cells are outside the public map-quality denominator.
                if _wall(name, cell):
                    continue
                key = (opportunity["slot"], submap_id, block_id)
                if cell not in latest or key > latest[cell][0]:
                    latest[cell] = (key, value, truth, submap_id, block_id)
    for item in latest.values(): productive_blocks.add((item[3], item[4]))
    target_cells = [(x, y) for x in range(GRID_X) for y in range(GRID_Y) if not _wall(name, (x, y))]
    map_fraction = len(latest) / len(target_cells)
    depth_rmse = _rmse([item[1] - item[2] for item in latest.values()]) if latest else 0.0
    map_quality = map_fraction * (_clip(1 - depth_rmse / 4) if latest else 0.0)
    for match in replay["accepted_matches"]:
        for side in (match["earlier"], match["later"]):
            packet = replay["restored"].get(side["id"])
            if packet is None: continue
            for cell in match["overlap"]:
                block_id = _block_for_cell(packet, cell)
                if block_id is not None: productive_blocks.add((side["id"], block_id))
    loop_coverage = len(replay["accepted_matches"]) / len(replay["matches"]) if replay["matches"] else 0.0
    p2 = .65 * map_quality + .35 * loop_coverage

    returned = sum(max(abs(replay["true"][auv][0] - STARTS[name][i][0]), abs(replay["true"][auv][1] - STARTS[name][i][1])) <= 1 for i, auv in enumerate(AUVS)) / 3
    p3 = .45 * returned + .35 * _clip(1 - replay["safety_events"] / (3 * H)) + .2 * _clip(1 - replay["energy"] / (3 * ENERGY_BUDGET[name]))

    by_submap: dict[str, set[str]] = defaultdict(set)
    for submap_id, block_id in productive_blocks: by_submap[submap_id].add(block_id)
    useful_bits = 0
    for submap_id, blocks in by_submap.items():
        packet = replay["submap_packets"].get(submap_id)
        if packet is None: continue
        useful_bits += 48 + 14 * len(blocks) + 3 * len(blocks & set(packet["orientation_blocks"]))
    essential, sent = 0, 0
    for packet in replay["restored"].values():
        for block_id, restored in packet["restored_blocks"].items():
            if not restored["direction_sent"]: continue
            sent += 1
            block = packet["opportunity"]["blocks"][block_id]
            actual = _reconstruct(block["center"], block["index"], block["orientation"])
            fallback = _reconstruct(block["center"], block["index"], restored["fallback_orientation"])
            actual_rmse = _rmse([actual[j] - block["values"][j] for j in range(4)])
            fallback_rmse = _rmse([fallback[j] - block["values"][j] for j in range(4)])
            if fallback_rmse - actual_rmse >= .5 - EPS: essential += 1
    orientation_yield = essential / sent if sent else 0.0
    a1 = .6 * _clip(useful_bits / max(1, replay["uplink_bits"])) + .4 * orientation_yield

    on_time = len(replay["restored"]) / 33
    freshness = [_clip(1 - (p["applied_slot"] - p["created_slot"]) / 24) for p in replay["applied_corrections"]]
    productive_packet_ids = {replay["submap_packets"][sid]["id"] for sid in by_submap if sid in replay["submap_packets"]}
    productive_packet_ids.update(p["id"] for p in replay["applied_corrections"])
    total_comm = sum(event["bits"] for event in replay["service_events"])
    productive_comm = sum(event["bits"] for event in replay["service_events"] if event["packet_id"] in productive_packet_ids)
    total_compute = sum(event["units"] for event in replay["compute_events"])
    productive_compute = sum(event["units"] for event in replay["compute_events"] if (event["submap_id"], event["block_id"]) in productive_blocks)
    productive_service = .5 * (productive_comm / total_comm if total_comm else 0.0) + .5 * (productive_compute / total_compute if total_compute else 0.0)
    a2 = .4 * on_time + .35 * (sum(freshness) / len(freshness) if freshness else 0.0) + .25 * productive_service
    return {"primary_metrics_1": _clip(p1), "primary_metrics_2": _clip(p2), "primary_metrics_3": _clip(p3), "auxiliary_metrics_1": _clip(a1), "auxiliary_metrics_2": _clip(a2)}


def _metrics(replayed: Mapping[str, dict[str, Any]], _records_arg: Mapping[str, dict[str, Any]] | None = None) -> dict[str, float]:
    values = {key: [] for key in ("primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2")}
    for name in SCENARIOS:
        metrics = _scenario_metrics(name, replayed[name])
        for key, value in metrics.items(): values[key].append(value)
    return {key: _clip(sum(items) / len(items)) for key, items in values.items()}


async def _build_state(ws: Workspace) -> EvalState:
    root = str(ws.root)
    if root in _CACHE: return _CACHE[root]
    lock = _LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _CACHE: return _CACHE[root]
        state, path = EvalState(), Path(ws.root, "results.json")
        try:
            await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            started = time.monotonic(); state.run_started = True
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.run_finished = True; state.elapsed_s = time.monotonic() - started
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
                state.data = _strict_json(ws.read_text("results.json") or ""); state.parse_ok = True
        except Exception as exc:
            state.timed_out = (isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower())
            state.official_message = f"{type(exc).__name__}: {exc}"
        state.diagnostics["official"] = state.official_message or "official run did not complete"
        atoms = _result_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(atoms, ATOM_SHARES["result_json"]); state.diagnostics["result_json"] = _diagnose_atoms(atoms)
        if not state.official_ok or not state.parse_ok:
            reason = state.diagnostics["official"]
            for key in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
                state.scores[key] = 0.0; state.diagnostics[key] = f"{key}: not evaluated because {reason}"
            _CACHE[root] = state; return state
        state.records = _records(state.data)
        atoms = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"]); state.diagnostics["result_evidence"] = _diagnose_atoms(atoms)
        state.scores["canonical_coverage"], defects = _coverage(state.data); state.diagnostics["canonical_coverage"] = f"defects={defects}"
        atoms = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"]); state.diagnostics["decision_legality"] = _diagnose_atoms(atoms)
        atoms, state.replayed = _replay_atoms(state.records)
        state.scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"]); state.diagnostics["replay_validity"] = _diagnose_atoms(atoms)
        state.validity_gate = state.official_ok and all(state.scores.get(key, 0.0) >= 1 - EPS for key in ("result_json", "result_evidence", "canonical_coverage", "decision_legality", "replay_validity"))
        if state.validity_gate: state.metrics = _metrics(state.replayed, state.records)
        _CACHE[root] = state; return state


async def _gate(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws); return state.scores.get(key, 0.0), state.diagnostics.get(key, "")


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "run.py")
    try: result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except Exception as exc: return 0.0, f"run.py compilation failed: {type(exc).__name__}: {exc}"
    try: regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError: regular = False
    return (1.0, "run.py exists and compiles") if regular and getattr(result, "exit_code", 1) == 0 else (0.0, "run.py is missing or does not compile")


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws); return (1.0, "official run passed") if state.official_ok else (0.0, state.diagnostics.get("official", "official run failed"))


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    path = Path(ws.root, "report.md")
    try: regular = stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError: regular = False
    try: text = (ws.read_text("report.md") or "") if regular else ""
    except OSError: text = ""
    score, missing = _report_score(text); return score, f"missing_or_empty={missing}"


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


def _metric_outcome(metrics: Mapping[str, float] | None, key: str) -> tuple[float, str]:
    if metrics is None: return 0.0, f"{key}: validity gate not satisfied"
    return metrics[key], f"{key}: replayed={metrics[key]:.9g}"


async def _metric(ws: Workspace, key: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return _metric_outcome(state.metrics, key)


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
