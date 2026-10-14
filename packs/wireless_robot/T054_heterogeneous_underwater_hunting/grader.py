# ruff: noqa: E501
from __future__ import annotations

import asyncio
import json
import math
import re
import stat
import time
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

EPS = 1e-6
GRID_X, GRID_Y, H = 24, 20, 96
VEHICLES = ("A0", "S0", "D0", "D1", "D2")
UUVS = ("D0", "D1", "D2")
SCENARIOS = ("open_basin", "island_shadow", "shear_current", "split_escape", "deadline_burst")
STARTS = {
    "open_basin": {
        "A0": (1, 1),
        "S0": (12, 1),
        "D0": (2, 17),
        "D1": (5, 17),
        "D2": (8, 17),
        "target": (20, 16),
    },
    "island_shadow": {
        "A0": (1, 18),
        "S0": (2, 2),
        "D0": (2, 16),
        "D1": (5, 18),
        "D2": (8, 16),
        "target": (20, 3),
    },
    "shear_current": {
        "A0": (1, 1),
        "S0": (12, 2),
        "D0": (2, 18),
        "D1": (6, 18),
        "D2": (10, 18),
        "target": (21, 15),
    },
    "split_escape": {
        "A0": (2, 10),
        "S0": (12, 1),
        "D0": (2, 2),
        "D1": (2, 10),
        "D2": (2, 18),
        "target": (21, 10),
    },
    "deadline_burst": {
        "A0": (1, 1),
        "S0": (12, 1),
        "D0": (1, 18),
        "D1": (5, 18),
        "D2": (9, 18),
        "target": (21, 17),
    },
}
OBSTACLES = {
    "open_basin": (),
    "island_shadow": ((9, 14, 6, 13),),
    "shear_current": ((10, 13, 8, 11),),
    "split_escape": ((10, 13, 7, 12),),
    "deadline_burst": ((7, 9, 8, 12), (15, 17, 5, 9)),
}
DEADLINES = {
    "open_basin": (32, 90),
    "island_shadow": (38, 92),
    "shear_current": (34, 92),
    "split_escape": (30, 88),
    "deadline_burst": (26, 78),
}
ENERGY = {
    "open_basin": {"A0": 150.0, "S0": 110.0, "D0": 100.0, "D1": 100.0, "D2": 100.0},
    "island_shadow": {"A0": 155.0, "S0": 115.0, "D0": 105.0, "D1": 105.0, "D2": 105.0},
    "shear_current": {"A0": 155.0, "S0": 120.0, "D0": 110.0, "D1": 110.0, "D2": 110.0},
    "split_escape": {"A0": 150.0, "S0": 115.0, "D0": 105.0, "D1": 105.0, "D2": 105.0},
    "deadline_burst": {"A0": 145.0, "S0": 105.0, "D0": 95.0, "D1": 95.0, "D2": 95.0},
}
ACTION_DELTAS = {
    "stay": (0, 0),
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
    "N1": (0, 1),
    "S1": (0, -1),
    "E1": (1, 0),
    "W1": (-1, 0),
    "N2": (0, 2),
    "S2": (0, -2),
    "E2": (2, 0),
    "W2": (-2, 0),
}
ACTION_SET = frozenset(ACTION_DELTAS)
AERIAL_ACTIONS = frozenset(("stay", "N1", "S1", "E1", "W1", "N2", "S2", "E2", "W2"))
MARINE_ACTIONS = frozenset(("stay", "N", "S", "E", "W"))
LINKS = ("A0-S0", "S0-D0", "S0-D1", "S0-D2")
LINK_ENDPOINTS = {
    "A0-S0": ("A0", "S0"),
    "S0-D0": ("S0", "D0"),
    "S0-D1": ("S0", "D1"),
    "S0-D2": ("S0", "D2"),
}
RECEIVER_INDEX = {"S0": 0, "D0": 1, "D1": 2, "D2": 3}
PACKET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PACKET_STATUSES = {"queued", "in_service", "arrived", "lost", "expired", "rejected"}
ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.25,
        "result_json.root": 0.25,
        "result_json.scenarios": 0.25,
        "result_json.records": 0.25,
    },
    "result_evidence": {
        "result_evidence.vehicles": 0.20,
        "result_evidence.observations": 0.15,
        "result_evidence.links": 0.12,
        "result_evidence.services": 0.12,
        "result_evidence.packets": 0.16,
        "result_evidence.queues": 0.10,
        "result_evidence.mission": 0.15,
    },
    "decision_legality": {
        "decision_legality.actions": 0.20,
        "decision_legality.observations": 0.12,
        "decision_legality.services": 0.18,
        "decision_legality.packets": 0.20,
        "decision_legality.queues": 0.10,
        "decision_legality.budget": 0.20,
    },
    "replay_validity": {
        "replay_validity.motion": 0.20,
        "replay_validity.target": 0.15,
        "replay_validity.observations": 0.15,
        "replay_validity.packets": 0.15,
        "replay_validity.energy": 0.15,
        "replay_validity.capture": 0.20,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float):
        self.id, self.passed, self.detail, self.share = id, passed, detail, share


class EvalState:
    def __init__(self):
        # Official-run bookkeeping is kept separately from parsed candidate data so
        # every public check can report the same deterministic failure reason.
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
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[Hashable, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_BUILD_CACHE: dict[str, EvalState] = {}
_BUILD_LOCKS: dict[str, asyncio.Lock] = {}


def _finite(v: Any) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, TypeError, ValueError):
        return False


def _safe_float(v: Any) -> float | None:
    """Return a finite float or ``None`` without leaking conversion errors."""
    if not _finite(v):
        return None
    try:
        return float(v)
    except (OverflowError, TypeError, ValueError):
        return None


def _integer(v: Any, lo: int, hi: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _shape(v: Any, keys: set[str], path: str) -> str | None:
    if not isinstance(v, dict):
        return f"{path}: expected object"
    if set(v) == keys:
        return None
    # JSON object keys are normally strings, but boundary probes may pass arbitrary
    # Python mappings.  Stringifying only for diagnostics keeps malformed input safe.
    actual = sorted((repr(k) for k in v), key=str)
    expected = sorted((repr(k) for k in keys), key=str)
    return f"{path}: expected keys={expected!r}, actual={actual!r}"


def _strict_json(raw: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError("strict JSON input must be text")

    def reject(v: str):
        raise ValueError(v)

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate JSON key: {k}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _strict_json_bytes(raw: bytes) -> Any:
    """Decode UTF-8 strictly, then apply duplicate-key/non-finite JSON checks."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("strict JSON input must be bytes")
    try:
        text = bytes(raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8: {exc}") from exc
    return _strict_json(text)


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
    sx, sy = (1 if dx > 0 else -1 if dx < 0 else 0), (1 if dy > 0 else -1 if dy < 0 else 0)
    adx, ady = abs(dx), abs(dy)
    tx = 0.5 / adx if adx else math.inf
    ty = 0.5 / ady if ady else math.inf
    dtx, dty = 1 / adx if adx else math.inf, 1 / ady if ady else math.inf
    x, y, t = x0, y0, 0.0
    while (x, y) != (x1, y1):
        if tx < ty - EPS:
            x += sx
            t = tx
            tx += dtx
            out.append((x, y))
        elif ty < tx - EPS:
            y += sy
            t = ty
            ty += dty
            out.append((x, y))
        else:
            sxcell, sycell = (x + sx, y), (x, y + sy)
            if sxcell not in out:
                out.append(sxcell)
            if sycell not in out:
                out.append(sycell)
            x += sx
            y += sy
            tx += dtx
            ty += dty
            t = min(tx, ty)
            out.append((x, y))
        if t >= 1 - EPS:
            break
    return tuple(dict.fromkeys(out))


def _current(name: str, slot: int) -> tuple[int, int]:
    if name == "shear_current":
        return (1, 0) if slot < 48 else (-1, 0)
    return (0, 0)


def _interference(name: str, slot: int) -> int:
    return (
        5
        if name == "deadline_burst" and 24 <= slot < 48
        else 2
        if name == "deadline_burst"
        else 1 + slot // 24
        if name == "shear_current"
        else 0
        if name == "open_basin"
        else 2
        if name == "island_shadow"
        else 3
    )


def _shadow(name: str, a: tuple[int, int], b: tuple[int, int]) -> int:
    """Deterministic supercover obstruction test for the public island cores."""
    if name == "open_basin":
        return 0
    return int(any(_wall(name, cell) for cell in _line_cells(a, b)[1:-1]))


def _link_quality(
    name: str, link_id: str, slot: int, positions: Mapping[str, tuple[int, int]]
) -> float:
    sender, receiver = LINK_ENDPOINTS[link_id]
    d = math.dist(positions[sender], positions[receiver])
    interference = _interference(name, slot)
    if link_id == "A0-S0":
        q = (
            1.0
            - d / 30.0
            - 0.12 * _shadow(name, positions[sender], positions[receiver])
            - 0.03 * interference
        )
    else:
        depth = {"D0": 1, "D1": 2, "D2": 3}[receiver]
        q = 1.0 - d / 22.0 - 0.08 * depth - 0.025 * interference
    return max(0.0, min(1.0, q))


def _link_available(
    name: str, link_id: str, slot: int, positions: Mapping[str, tuple[int, int]]
) -> bool:
    return _link_quality(name, link_id, slot, positions) >= 0.2 - EPS


def _success(name: str, link_id: str, slot: int, positions: Mapping[str, tuple[int, int]]) -> bool:
    sender, receiver = LINK_ENDPOINTS[link_id]
    q = _link_quality(name, link_id, slot, positions)
    receiver_index = RECEIVER_INDEX[receiver]
    value = (
        (13 * positions[sender][0] + 7 * positions[sender][1] + 5 * slot + 3 * receiver_index) % 100
    ) / 100
    return value >= 0.08 + 0.25 * (1 - q)


def _link_delay(
    name: str, link_id: str, slot: int, positions: Mapping[str, tuple[int, int]]
) -> int:
    q = _link_quality(name, link_id, slot, positions)
    return (1 if link_id == "A0-S0" else 3) + math.floor(3 * (1 - q))


def _project_move(
    name: str,
    position: tuple[int, int],
    action: str,
    slot: int,
    *,
    underwater: bool,
) -> tuple[str, tuple[int, int]]:
    dx, dy = ACTION_DELTAS.get(action, (0, 0))
    cx, cy = _current(name, slot)
    scale = 0.5 if underwater else 1.0
    desired = (dx + scale * cx, dy + scale * cy)
    choices: list[tuple[float, int, str, tuple[int, int]]] = []
    for rank, candidate_action in enumerate(("stay", "N", "E", "S", "W")):
        px, py = ACTION_DELTAS[candidate_action]
        cell = (position[0] + px, position[1] + py)
        if not (0 <= cell[0] < GRID_X and 0 <= cell[1] < GRID_Y) or _wall(name, cell):
            continue
        error = (px - desired[0]) ** 2 + (py - desired[1]) ** 2
        choices.append((error, rank, candidate_action, cell))
    if not choices:
        return "stay", position
    _, _, selected, cell = min(choices)
    return selected, cell


def _shortest_legal_tracking_step(
    name: str,
    position: tuple[int, int],
    action: Any,
    slot: int,
    target: tuple[int, int],
) -> bool:
    """Whether an intended UUV action projects to a shortest legal step.

    The comparison is made on the actual projected one-cell destination, so
    currents and obstacle projection are handled exactly as in the fixed model.
    """
    if action not in MARINE_ACTIONS or action == "stay":
        return False
    try:
        projected, destination = _project_move(name, position, action, slot, underwater=True)
        if projected == "stay":
            return False
        current_distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
        legal_distances: list[int] = []
        for candidate in MARINE_ACTIONS:
            selected, cell = _project_move(name, position, candidate, slot, underwater=True)
            if selected != "stay":
                legal_distances.append(abs(cell[0] - target[0]) + abs(cell[1] - target[1]))
        if not legal_distances:
            return False
        destination_distance = abs(destination[0] - target[0]) + abs(destination[1] - target[1])
        return destination_distance < current_distance and destination_distance == min(legal_distances)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False


def _target_step(
    name: str,
    target: tuple[int, int],
    vehicles: Mapping[str, tuple[int, int]],
    phase: str,
    slot: int,
) -> tuple[int, int]:
    anchor = (
        vehicles["A0"]
        if phase == "search"
        else (
            sum(vehicles[d][0] for d in UUVS) / 3,
            sum(vehicles[d][1] for d in UUVS) / 3,
        )
    )
    best_action = "stay"
    best_score = -1.0
    for action in ("E", "N", "W", "S", "stay"):
        dx, dy = ACTION_DELTAS[action]
        cell = (target[0] + dx, target[1] + dy)
        if not (0 <= cell[0] < GRID_X and 0 <= cell[1] < GRID_Y) or _wall(name, cell):
            continue
        score = (
            abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
            if phase == "search"
            else math.dist(cell, anchor)
        )
        if score > best_score:
            best_action, best_score = action, score
    return _project_move(name, target, best_action, slot, underwater=False)[1]


def _record_error(row: Any, path: str) -> str | None:
    return _shape(
        row,
        {
            "scenario",
            "vehicles",
            "observations",
            "links",
            "service_slots",
            "packets",
            "queues",
            "mission_state",
        },
        path,
    )


def _index_records(data: Any) -> dict[Hashable, Any]:
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return {}
    indexed: dict[Hashable, Any] = {}
    for row in data["scenarios"]:
        if not isinstance(row, Mapping):
            continue
        key = row.get("scenario")
        try:
            if key in SCENARIOS and key not in indexed:
                indexed[key] = row
        except (TypeError, ValueError):
            # Malformed/unhashable canonical keys are reported by coverage; the
            # index must remain total so later atoms can fail safely.
            continue
    return indexed


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    vals = {
        "result_json.parse": (parse_ok, "strict JSON parse"),
        "result_json.root": (isinstance(data, dict) and set(data) == {"scenarios"}, "root"),
        "result_json.scenarios": (isinstance(scenarios, list), "scenarios"),
        "result_json.records": (
            isinstance(scenarios, list)
            and all(_record_error(v, f"scenarios[{i}]") is None for i, v in enumerate(scenarios)),
            "records",
        ),
    }
    return [
        AtomVerdict(k, bool(v[0]), str(v[1]), ATOM_SHARES["result_json"][k])
        for k, v in vals.items()
    ]


def _rows(row: Mapping[str, Any], field: str, keys: set[str], length: int) -> tuple[bool, str]:
    if not isinstance(row, Mapping):
        return False, f"{field}: parent is not an object"
    vals = row.get(field)
    if not isinstance(vals, list) or len(vals) != length:
        return False, f"{field}: expected {length} rows"
    for i, v in enumerate(vals):
        if not isinstance(v, dict):
            return False, f"{field}[{i}]: expected object"
        e = _shape(v, keys, f"{field}[{i}]")
        if e:
            return False, e
    return True, "ok"


def _slot_map(value: Any, limit: int) -> tuple[dict[int, Mapping[str, Any]], bool]:
    """Index slot-keyed rows without giving their physical JSON order meaning.

    The public contract makes ``slot`` the canonical key for state/action rows.
    Returning a validity flag lets callers report duplicate/out-of-range rows while
    still safely indexing any well-formed subset during diagnostic paths.
    """
    if not isinstance(value, list):
        return {}, False
    result: dict[int, Mapping[str, Any]] = {}
    valid = True
    for item in value:
        if not isinstance(item, Mapping):
            valid = False
            continue
        slot = item.get("slot")
        if not _integer(slot, 0, limit) or slot in result:
            valid = False
            continue
        result[slot] = item
    return result, valid


def _canonical_cells(value: Any) -> tuple[tuple[int, int], ...] | None:
    """Normalize a scan cell list without assigning meaning to its order."""
    if not isinstance(value, list):
        return None
    cells: list[tuple[int, int]] = []
    for cell in value:
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or not _integer(cell[0], 0, GRID_X - 1)
            or not _integer(cell[1], 0, GRID_Y - 1)
        ):
            return None
        cells.append((cell[0], cell[1]))
    if len(cells) != len(set(cells)):
        return None
    return tuple(sorted(cells))


def _packet_id_sort_key(value: Any) -> str:
    """Stable public tie key for packet IDs of either permitted JSON type."""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _packet_link(packet: Mapping[str, Any]) -> str | None:
    """Return the canonical FIFO link for a packet without trusting its shape."""
    if not isinstance(packet, Mapping):
        return None
    source, destination = packet.get("source"), packet.get("destination")
    if (source, destination) == ("A0", "S0"):
        return "A0-S0"
    if source == "S0" and destination in UUVS:
        return f"S0-{destination}"
    return None


def _estimate_option_key(packet: Mapping[str, Any]) -> tuple[int, str]:
    """Canonical tie key for same-boundary packet updates."""
    link = _packet_link(packet)
    return (LINKS.index(link) if link in LINKS else len(LINKS), _packet_id_sort_key(packet.get("packet_id")))


def _evidence_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    checks = {
        k: [True, "ok"]
        for k in ("vehicles", "observations", "links", "services", "packets", "queues", "mission")
    }
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        if not isinstance(row.get("vehicles"), list) or len(row["vehicles"]) != 5:
            checks["vehicles"] = [False, f"{name}: vehicles"]
        else:
            seen_vehicle_ids: set[Any] = set()
            for vehicle in row["vehicles"]:
                if not isinstance(vehicle, dict):
                    checks["vehicles"] = [False, f"{name}: vehicle is not an object"]
                    continue
                if _shape(
                    vehicle,
                    {"vehicle_id", "state_trace", "intended_actions", "executed_actions"},
                    "vehicle",
                ):
                    checks["vehicles"] = [False, f"{name}: vehicle shape"]
                    continue
                vehicle_id = vehicle.get("vehicle_id")
                if vehicle_id not in VEHICLES:
                    checks["vehicles"] = [False, f"{name}: unknown vehicle"]
                elif vehicle_id in seen_vehicle_ids:
                    checks["vehicles"] = [False, f"{name}: duplicate vehicle"]
                else:
                    seen_vehicle_ids.add(vehicle_id)
                for field, keys, length in (
                    (
                        "state_trace",
                        {"slot", "x", "y", "energy", "estimate_x", "estimate_y", "estimate_age"},
                        H + 1,
                    ),
                    ("intended_actions", {"slot", "action"}, H),
                    ("executed_actions", {"slot", "action"}, H),
                ):
                    ok, detail = _rows(vehicle, field, keys, length)
                    if not ok:
                        checks["vehicles"] = [False, f"{name}: {detail}"]
                state_rows = vehicle.get("state_trace", [])
                if not isinstance(state_rows, list):
                    state_rows = []
                for s in state_rows:
                    if not isinstance(s, dict):
                        checks["vehicles"] = [False, f"{name}: invalid state row"]
                        continue
                    if not (
                        _integer(s.get("slot"), 0, H)
                        and _integer(s.get("x"), 0, GRID_X - 1)
                        and _integer(s.get("y"), 0, GRID_Y - 1)
                        and _finite(s.get("energy"))
                        and s.get("energy") >= -EPS
                    ):
                        checks["vehicles"] = [False, f"{name}: invalid state"]
                    estimates = (s.get("estimate_x"), s.get("estimate_y"), s.get("estimate_age"))
                    if estimates != (None, None, None) and not (
                        _integer(estimates[0], 0, GRID_X - 1)
                        and _integer(estimates[1], 0, GRID_Y - 1)
                        and _integer(estimates[2], 0, 12)
                    ):
                        checks["vehicles"] = [False, f"{name}: invalid local estimate"]
                intended_rows = vehicle.get("intended_actions", [])
                executed_rows = vehicle.get("executed_actions", [])
                if not isinstance(intended_rows, list):
                    intended_rows = []
                if not isinstance(executed_rows, list):
                    executed_rows = []
                for rows_value, row_limit, field_name in (
                    (state_rows, H, "state_trace"),
                    (intended_rows, H - 1, "intended_actions"),
                    (executed_rows, H - 1, "executed_actions"),
                ):
                    _, slot_ok = _slot_map(rows_value, row_limit)
                    if not slot_ok:
                        checks["vehicles"] = [False, f"{name}: duplicate or invalid {field_name} slot"]
                for action_row in intended_rows + executed_rows:
                    if not isinstance(action_row, dict):
                        checks["vehicles"] = [False, f"{name}: invalid action row"]
                        continue
                    if not _integer(action_row.get("slot"), 0, H - 1) or not isinstance(
                        action_row.get("action"), str
                    ):
                        checks["vehicles"] = [False, f"{name}: invalid action row"]
            if seen_vehicle_ids != set(VEHICLES):
                checks["vehicles"] = [False, f"{name}: vehicle IDs do not cover canonical set"]
        ok, detail = _rows(
            row,
            "observations",
            {"vehicle_id", "slot", "kind", "cells", "hit", "observed_x", "observed_y"},
            len(VEHICLES) * H,
        )
        if not ok:
            checks["observations"] = [False, f"{name}: {detail}"]
        for observation in (
            row.get("observations", []) if isinstance(row.get("observations"), list) else []
        ):
            if not isinstance(observation, dict):
                checks["observations"] = [False, f"{name}: observation is not an object"]
                continue
            if (
                observation.get("vehicle_id") not in VEHICLES
                or not _integer(observation.get("slot"), 0, H - 1)
                or observation.get("kind") not in ("scan", "sonar", "none")
                or not isinstance(observation.get("hit"), bool)
                or _canonical_cells(observation.get("cells")) is None
                or (
                    observation.get("observed_x") is not None
                    and not _integer(observation.get("observed_x"), 0, GRID_X - 1)
                )
                or (
                    observation.get("observed_y") is not None
                    and not _integer(observation.get("observed_y"), 0, GRID_Y - 1)
                )
                or (observation.get("hit") and observation.get("kind") not in ("scan", "sonar"))
                or (observation.get("hit") and (observation.get("observed_x") is None or observation.get("observed_y") is None))
                or (not observation.get("hit") and (observation.get("observed_x") is not None or observation.get("observed_y") is not None))
            ):
                checks["observations"] = [False, f"{name}: invalid observation row"]
        ok, detail = _rows(
            row,
            "links",
            {"link_id", "slot", "sender", "receiver", "observed_available", "observed_quality"},
            len(LINKS) * H,
        )
        if not ok:
            checks["links"] = [False, f"{name}: {detail}"]
        for link in row.get("links", []) if isinstance(row.get("links"), list) else []:
            if not isinstance(link, dict):
                checks["links"] = [False, f"{name}: link is not an object"]
                continue
            try:
                endpoints = LINK_ENDPOINTS.get(link.get("link_id"))
            except TypeError:
                endpoints = None
            quality = link.get("observed_quality")
            if (
                endpoints != (link.get("sender"), link.get("receiver"))
                or not _integer(link.get("slot"), 0, H - 1)
                or not isinstance(link.get("observed_available"), bool)
                or (quality is not None and (not _finite(quality) or not 0 <= quality <= 1))
                or (link.get("observed_available") and (quality is None or not _finite(quality)))
                or (not link.get("observed_available") and quality is not None)
            ):
                checks["links"] = [False, f"{name}: invalid link row"]
        ok, detail = _rows(
            row,
            "service_slots",
            {"channel", "slot", "selected_link", "packet_id", "served_units"},
            2 * H,
        )
        if not ok:
            checks["services"] = [False, f"{name}: {detail}"]
        for service in (
            row.get("service_slots", []) if isinstance(row.get("service_slots"), list) else []
        ):
            if not isinstance(service, dict):
                checks["services"] = [False, f"{name}: service is not an object"]
                continue
            packet_id = service.get("packet_id")
            valid_id = (
                packet_id is None
                or (
                    isinstance(packet_id, int)
                    and not isinstance(packet_id, bool)
                    and packet_id >= 0
                )
                or (isinstance(packet_id, str) and PACKET_ID_RE.fullmatch(packet_id))
            )
            if (
                service.get("channel") not in ("wireless", "acoustic")
                or not _integer(service.get("slot"), 0, H - 1)
                or service.get("selected_link") is not None
                and service.get("selected_link") not in LINKS
                or not valid_id
                or not _integer(service.get("served_units"), 0, 4)
            ):
                checks["services"] = [False, f"{name}: invalid service row"]
        ok, detail = _rows(
            row,
            "queues",
            {
                "link_id",
                "slot",
                "packet_ids",
                "occupied_units",
                "served_packet_id",
                "rejected_packet_ids",
            },
            len(LINKS) * H,
        )
        if not ok:
            checks["queues"] = [False, f"{name}: {detail}"]
        for queue in row.get("queues", []) if isinstance(row.get("queues"), list) else []:
            if not isinstance(queue, dict):
                checks["queues"] = [False, f"{name}: queue is not an object"]
                continue
            if (
                queue.get("link_id") not in LINKS
                or not _integer(queue.get("slot"), 0, H - 1)
                or not isinstance(queue.get("packet_ids"), list)
                or not _integer(queue.get("occupied_units"), 0, 12)
                or queue.get("served_packet_id") is not None
                and (
                    not isinstance(queue.get("served_packet_id"), (int, str))
                    or isinstance(queue.get("served_packet_id"), bool)
                )
                or not isinstance(queue.get("rejected_packet_ids"), list)
            ):
                checks["queues"] = [False, f"{name}: invalid queue row"]
        ok, detail = _rows(
            row,
            "mission_state",
            {
                "slot",
                "observed_target_x",
                "observed_target_y",
                "phase",
                "first_detection_slot",
                "capture_condition",
                "capture_run",
                "capture_complete",
                "safety_events",
                "total_energy",
            },
            H + 1,
        )
        if not ok:
            checks["mission"] = [False, f"{name}: {detail}"]
        for mission in (
            row.get("mission_state", []) if isinstance(row.get("mission_state"), list) else []
        ):
            if not isinstance(mission, dict):
                checks["mission"] = [False, f"{name}: mission is not an object"]
                continue
            if (
                not _integer(mission.get("slot"), 0, H)
                or mission.get("phase") not in ("search", "hunting")
                or not isinstance(mission.get("capture_condition"), bool)
                or not isinstance(mission.get("capture_complete"), bool)
                or not _integer(mission.get("capture_run"), 0, H)
                or not _integer(mission.get("safety_events"), 0, 10**9)
                or not _finite(mission.get("total_energy"))
                or mission.get("total_energy") < -EPS
                or (
                    mission.get("observed_target_x") is not None
                    and not _integer(mission.get("observed_target_x"), 0, GRID_X - 1)
                )
                or (
                    mission.get("observed_target_y") is not None
                    and not _integer(mission.get("observed_target_y"), 0, GRID_Y - 1)
                )
                or (
                    mission.get("first_detection_slot") is not None
                    and not _integer(mission.get("first_detection_slot"), 0, H - 1)
                )
            ):
                checks["mission"] = [False, f"{name}: invalid mission row"]
        packets = row.get("packets")
        if not isinstance(packets, list):
            checks["packets"] = [False, f"{name}: packets"]
        else:
            for p in packets:
                if not isinstance(p, dict):
                    checks["packets"] = [False, f"{name}: packet is not an object"]
                    continue
                keys = {
                    "packet_id",
                    "parent_id",
                    "source",
                    "destination",
                    "granularity",
                    "estimate_x",
                    "estimate_y",
                    "created_slot",
                    "admitted_slot",
                    "first_service_slot",
                    "last_service_slot",
                    "arrival_slot",
                    "status",
                    "payload_units",
                    "retransmissions",
                    "used_slots",
                }
                if (
                    _shape(p, keys, "packet")
                    or not (
                        (
                            isinstance(p.get("packet_id"), int)
                            and not isinstance(p.get("packet_id"), bool)
                            and p.get("packet_id") >= 0
                        )
                        or (
                            isinstance(p.get("packet_id"), str)
                            and PACKET_ID_RE.fullmatch(p.get("packet_id"))
                        )
                    )
                    or not (
                        p.get("parent_id") is None
                        or (
                            isinstance(p.get("parent_id"), (int, str))
                            and not isinstance(p.get("parent_id"), bool)
                        )
                    )
                    or p.get("source") not in VEHICLES
                    or p.get("destination") not in VEHICLES
                    or (p.get("source"), p.get("destination"))
                    not in (("A0", "S0"), ("S0", "D0"), ("S0", "D1"), ("S0", "D2"))
                    or not (
                        p.get("granularity") in ("coarse", "fine")
                        and _integer(p.get("estimate_x"), 0, GRID_X - 1)
                        and _integer(p.get("estimate_y"), 0, GRID_Y - 1)
                        and _integer(p.get("created_slot"), 0, H - 1)
                        and all(
                            value is None or _integer(value, 0, H)
                            for value in (
                                p.get("admitted_slot"),
                                p.get("first_service_slot"),
                                p.get("last_service_slot"),
                                p.get("arrival_slot"),
                            )
                        )
                        and _integer(p.get("payload_units"), 1, 3)
                        and p.get("payload_units") == (1 if p.get("granularity") == "coarse" else 3)
                        and _integer(p.get("retransmissions"), 0, 180)
                        and isinstance(p.get("used_slots"), list)
                        and all(_integer(value, 0, H - 1) for value in p.get("used_slots", []))
                        and len(p.get("used_slots", [])) == len(set(p.get("used_slots", [])))
                        and p.get("status") in PACKET_STATUSES
                    )
                ):
                    checks["packets"] = [False, f"{name}: packet shape"]
    return [
        AtomVerdict(
            f"result_evidence.{k}",
            bool(v[0]),
            str(v[1]),
            ATOM_SHARES["result_evidence"][f"result_evidence.{k}"],
        )
        for k, v in checks.items()
    ]


def _hashable(v: Any) -> Hashable:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    try:
        hash(v)
        return v
    except Exception:
        return repr(v)


def _coverage(data: Any) -> tuple[float, list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios.missing"]
    defects: list[str] = []
    rows = data["scenarios"]
    actual = [_hashable(r.get("scenario")) if isinstance(r, dict) else _hashable(r) for r in rows]
    for key in SCENARIOS:
        if actual.count(key) == 0:
            defects.append(f"scenario.missing:{key}")
        elif actual.count(key) > 1:
            defects.extend(f"scenario.duplicate:{key}:{i}" for i in range(actual.count(key) - 1))
    for i, key in enumerate(actual):
        if key not in SCENARIOS:
            defects.append(f"scenario.unknown:{i}")
    for name in SCENARIOS:
        row = next((r for r in rows if isinstance(r, dict) and r.get("scenario") == name), None)
        if row is None:
            continue
        expected = {
            "vehicles": set(VEHICLES),
            "vehicle_state": {(v, t) for v in VEHICLES for t in range(H + 1)},
            "vehicle_intended_actions": {(v, t) for v in VEHICLES for t in range(H)},
            "vehicle_executed_actions": {(v, t) for v in VEHICLES for t in range(H)},
            "observations": {(v, t) for v in VEHICLES for t in range(H)},
            "links": {(link_id, t) for link_id in LINKS for t in range(H)},
            "service_slots": {(c, t) for c in ("wireless", "acoustic") for t in range(H)},
            "queues": {(link_id, t) for link_id in LINKS for t in range(H)},
            "mission_state": {(t,) for t in range(H + 1)},
        }
        for field, keys in expected.items():
            actual_keys = []
            if field == "vehicles":
                vals = row.get("vehicles", [])
                for value in vals if isinstance(vals, list) else []:
                    if isinstance(value, Mapping):
                        actual_keys.append(_hashable(value.get("vehicle_id")))
                    else:
                        actual_keys.append(("<malformed>", len(actual_keys)))
            elif field in {
                "vehicle_state",
                "vehicle_intended_actions",
                "vehicle_executed_actions",
            }:
                # Nested vehicle rows are canonicalized by the declared IDs and
                # explicit slot fields; physical list order is not semantic.
                nested_field = {
                    "vehicle_state": "state_trace",
                    "vehicle_intended_actions": "intended_actions",
                    "vehicle_executed_actions": "executed_actions",
                }[field]
                vehicles = row.get("vehicles", [])
                for vehicle in vehicles if isinstance(vehicles, list) else []:
                    if not isinstance(vehicle, Mapping):
                        actual_keys.append(("<malformed>", len(actual_keys)))
                        continue
                    parent_id = _hashable(vehicle.get("vehicle_id"))
                    nested = vehicle.get(nested_field)
                    if not isinstance(nested, list):
                        actual_keys.append((parent_id, "<missing>"))
                        continue
                    for item in nested:
                        if isinstance(item, Mapping):
                            actual_keys.append((parent_id, _hashable(item.get("slot"))))
                        else:
                            actual_keys.append(("<malformed>", len(actual_keys)))
            else:
                vals = row.get(field, [])
                for value in vals if isinstance(vals, list) else []:
                    if not isinstance(value, Mapping):
                        actual_keys.append(("<malformed>", len(actual_keys)))
                        continue
                    if field in {"observations", "links"}:
                        actual_keys.append(
                            (
                                _hashable(value.get("vehicle_id") if field == "observations" else value.get("link_id")),
                                _hashable(value.get("slot")),
                            )
                        )
                    elif field == "service_slots":
                        actual_keys.append((_hashable(value.get("channel")), _hashable(value.get("slot"))))
                    elif field == "queues":
                        actual_keys.append((_hashable(value.get("link_id")), _hashable(value.get("slot"))))
                    else:
                        actual_keys.append((_hashable(value.get("slot")),))
            for expected_key in keys:
                count = actual_keys.count(expected_key)
                if count == 0:
                    defects.append(f"{name}.{field}.missing:{expected_key}")
                elif count > 1:
                    defects.extend(
                        f"{name}.{field}.duplicate:{expected_key}:{i}" for i in range(count - 1)
                    )
            for i, actual_key in enumerate(actual_keys):
                if actual_key not in keys:
                    defects.append(f"{name}.{field}.unknown:{i}")
    # One raw half-point is charged for every concrete defect, with four raw points total.
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


def _maps(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        row = {}

    def rows(field: str) -> list[Any]:
        value = row.get(field)
        return value if isinstance(value, list) else []

    def keyed(field: str, key_fn) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for value in rows(field):
            if not isinstance(value, Mapping):
                continue
            try:
                key = key_fn(value)
                hash(key)
            except Exception:
                # Keep malformed rows addressable without allowing a hostile key to
                # abort the remaining atom checks.
                try:
                    raw_key = key_fn(value)
                except Exception:
                    raw_key = "<malformed-key>"
                key = repr(raw_key)
            result.setdefault(key, value)
        return result

    return {
        "vehicles": keyed("vehicles", lambda v: v.get("vehicle_id")),
        "observations": keyed(
            "observations", lambda v: (v.get("vehicle_id"), v.get("slot"))
        ),
        "links": keyed("links", lambda v: (v.get("link_id"), v.get("slot"))),
        "services": keyed("service_slots", lambda v: (v.get("channel"), v.get("slot"))),
        "packets": keyed("packets", lambda v: v.get("packet_id")),
    }


def _packet_origin_slots(packet_rows: Any) -> dict[Hashable, int]:
    """Derive each packet's root A0 creation slot without trusting a hidden field."""
    if not isinstance(packet_rows, list):
        return {}
    by_id: dict[Hashable, Mapping[str, Any]] = {}
    for packet in packet_rows:
        if not isinstance(packet, Mapping):
            continue
        pid = packet.get("packet_id")
        if isinstance(pid, bool):
            continue
        try:
            hash(pid)
        except Exception:
            continue
        by_id.setdefault(pid, packet)
    cache: dict[Hashable, int] = {}
    visiting: set[Hashable] = set()

    def visit(pid: Hashable) -> int | None:
        try:
            hash(pid)
        except Exception:
            return None
        if pid in cache:
            return cache[pid]
        if pid in visiting:
            return None
        packet = by_id.get(pid)
        if packet is None:
            return None
        visiting.add(pid)
        if packet.get("source") == "A0":
            value = packet.get("created_slot")
        else:
            parent = packet.get("parent_id")
            value = visit(parent) if parent is not None else None
        visiting.discard(pid)
        if _integer(value, 0, H - 1):
            cache[pid] = value
            return value
        return None

    for pid in by_id:
        visit(pid)
    return cache


def _safe_origin_lookup(origins: Mapping[Any, int], packet_id: Any) -> int | None:
    """Look up a derived origin without allowing an unhashable hostile ID to raise."""
    try:
        return origins.get(packet_id)
    except (TypeError, ValueError, AttributeError):
        return None


def _estimate_options(
    row: Mapping[str, Any], vehicle_id: str, slot: int, *, exclude_packet: Hashable | None = None
) -> list[tuple[int, int, int, str, str]]:
    """Return estimates in the public decision-boundary update order.

    The first item is the fresh estimate carried into the boundary from causal
    history.  Arrivals at this boundary then replace it in canonical link/packet
    order, and a local hit is processed last.  Callers that need the *latest*
    estimate must use the final item rather than accepting an arbitrary member.
    """
    if not isinstance(row, Mapping) or vehicle_id not in UUVS:
        return []
    options: list[tuple[int, int, int, str, str]] = []
    maps = _maps(row)
    # Reconstruct the carried estimate from a causal source instead of trusting
    # a clipped age value of 12 in the candidate trace.  A source older than the
    # control horizon is therefore not an action option even though its displayed
    # age may remain clipped at 12.
    prior_events = _historical_estimate_events(row, vehicle_id, slot, control_only=True)
    if prior_events:
        _, ex, ey, age, _, _, granularity = prior_events[-1]
        options.append((ex, ey, age, "trace", granularity))

    packets = row.get("packets")
    origins = _packet_origin_slots(packets)
    eligible_packets: list[Mapping[str, Any]] = []
    if isinstance(packets, list):
        for packet in packets:
            if not isinstance(packet, Mapping) or packet.get("destination") != vehicle_id:
                continue
            pid = packet.get("packet_id")
            if exclude_packet is not None:
                try:
                    if pid == exclude_packet:
                        continue
                except Exception:
                    pass
            if packet.get("status") != "arrived" or packet.get("arrival_slot") != slot:
                continue
            root_slot = _safe_origin_lookup(origins, pid)
            if root_slot is None or slot - root_slot > 12:
                continue
            if slot < root_slot:
                continue
            ex, ey = packet.get("estimate_x"), packet.get("estimate_y")
            if _integer(ex, 0, GRID_X - 1) and _integer(ey, 0, GRID_Y - 1):
                eligible_packets.append(packet)
    for packet in sorted(eligible_packets, key=_estimate_option_key):
        root_slot = _safe_origin_lookup(origins, packet.get("packet_id"))
        if root_slot is None:
            continue
        granularity = packet.get("granularity")
        if granularity not in ("coarse", "fine"):
            # Evidence/legality atoms report malformed packet shape.  The
            # provenance helper must nevertheless stay total while diagnosing
            # that malformed row and must not raise on a missing field.
            continue
        options.append(
            (
                packet["estimate_x"],
                packet["estimate_y"],
                min(12, slot - root_slot),
                "packet",
                granularity,
            )
        )
    # A local scan/sonar hit is processed after all packet arrivals at the
    # boundary, so it wins the deterministic tie order.
    observations = maps["observations"]
    obs = observations.get((vehicle_id, slot))
    if isinstance(obs, Mapping) and obs.get("hit"):
        ex, ey = obs.get("observed_x"), obs.get("observed_y")
        if _integer(ex, 0, GRID_X - 1) and _integer(ey, 0, GRID_Y - 1):
            options.append((ex, ey, 0, "local_hit", "fine"))
    return options


def _historical_estimate_events(
    row: Mapping[str, Any],
    vehicle_id: str,
    boundary_slot: int,
    *,
    exclude_packet: Hashable | None = None,
    control_only: bool = True,
) -> list[tuple[int, int, int, int, str, Hashable | None, str]]:
    """List estimate updates strictly before a boundary in canonical order.

    This is used to audit the candidate's recorded state trace and aggregate
    evidence.  It never consults hidden target state.  ``control_only`` filters
    sources that are too old to drive a decision; aggregate evidence can request
    the complete (including over-age) arrival history.
    """
    if not isinstance(row, Mapping) or vehicle_id not in VEHICLES or not _integer(boundary_slot, 0, H):
        return []
    maps = _maps(row)
    origins = _packet_origin_slots(row.get("packets"))
    events: list[tuple[int, int, int, int, str, Hashable | None, str]] = []
    observations = maps.get("observations", {})
    packets = row.get("packets")
    for event_slot in range(boundary_slot):
        packet_events: list[Mapping[str, Any]] = []
        if isinstance(packets, list):
            for packet in packets:
                if not isinstance(packet, Mapping) or packet.get("destination") != vehicle_id:
                    continue
                pid = packet.get("packet_id")
                if exclude_packet is not None:
                    try:
                        if pid == exclude_packet:
                            continue
                    except Exception:
                        pass
                root_slot = _safe_origin_lookup(origins, pid)
                arrival = packet.get("arrival_slot")
                if (
                    packet.get("status") == "arrived"
                    and _integer(arrival, 0, H - 1)
                    and arrival == event_slot
                    and root_slot is not None
                    and event_slot >= root_slot
                    and (not control_only or boundary_slot - root_slot <= 12)
                    and _integer(packet.get("estimate_x"), 0, GRID_X - 1)
                    and _integer(packet.get("estimate_y"), 0, GRID_Y - 1)
                ):
                    packet_events.append(packet)
        for packet in sorted(packet_events, key=_estimate_option_key):
            root_slot = _safe_origin_lookup(origins, packet.get("packet_id"))
            if root_slot is None:
                continue
            events.append(
                (
                    event_slot,
                    packet["estimate_x"],
                    packet["estimate_y"],
                    min(12, boundary_slot - root_slot),
                    "packet",
                    packet.get("packet_id"),
                    packet.get("granularity"),
                )
            )
        observation = observations.get((vehicle_id, event_slot))
        if (
            isinstance(observation, Mapping)
            and observation.get("hit") is True
            and (not control_only or boundary_slot - event_slot <= 12)
            and _integer(observation.get("observed_x"), 0, GRID_X - 1)
            and _integer(observation.get("observed_y"), 0, GRID_Y - 1)
        ):
            events.append(
                (
                    event_slot,
                    observation["observed_x"],
                    observation["observed_y"],
                    min(12, boundary_slot - event_slot),
                    "local_hit",
                    None,
                    "fine",
                )
            )
    return events


def _effective_estimate_event(
    row: Mapping[str, Any], vehicle_id: str, boundary_slot: int, *, exclude_packet: Hashable | None = None
) -> tuple[int, int, int, int, str, Hashable | None, str] | None:
    """Return the estimate memory visible at a boundary.

    Fresh events are the control memory.  An over-age arrival remains auditable
    evidence, but cannot displace a still-fresh memory.  Once no fresh event
    remains, the latest stale event is the only possible stale representation;
    callers may still allow the candidate to clear that representation in the
    result trace.
    """
    all_events = _historical_estimate_events(
        row, vehicle_id, boundary_slot, exclude_packet=exclude_packet, control_only=False
    )
    if not all_events:
        return None
    fresh_events = _historical_estimate_events(
        row, vehicle_id, boundary_slot, exclude_packet=exclude_packet, control_only=True
    )
    return fresh_events[-1] if fresh_events else all_events[-1]


def _latest_aggregate_estimate(
    row: Mapping[str, Any], boundary_slot: int
) -> tuple[int, int] | None:
    """Return the most recent public hit/arrival available at a mission boundary."""
    candidates: list[tuple[tuple[int, int, int, str], tuple[int, int]]] = []
    for vehicle_index, vehicle_id in enumerate(VEHICLES):
        for source_slot, ex, ey, _age, source, pid, _granularity in _historical_estimate_events(
            row, vehicle_id, boundary_slot, control_only=False
        ):
            packet_rank = 0
            tie_id = ""
            if source == "packet":
                packet = _maps(row)["packets"].get(pid, {})
                packet_rank = LINKS.index(_packet_link(packet)) if _packet_link(packet) in LINKS else len(LINKS)
                tie_id = _packet_id_sort_key(pid)
            else:
                # Local hits are processed after packets; vehicle order makes
                # simultaneous local hits deterministic for the aggregate field.
                packet_rank = len(LINKS) + 1
                tie_id = str(vehicle_index)
            candidates.append(((source_slot, packet_rank, vehicle_index, tie_id), (ex, ey)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _legality_atoms(records: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules = {
        k: [True, msg]
        for k, msg in (
            ("actions", "actions legal"),
            ("observations", "observations legal"),
            ("services", "services legal"),
            ("packets", "packets legal"),
            ("queues", "queues legal"),
            ("budget", "budgets legal"),
        )
    }

    def fail(key: str, detail: str) -> None:
        # Keep the first concrete diagnostic: it is generally the nearest cause
        # and avoids replacing a useful field path with a later cascade.
        if rules[key][0]:
            rules[key] = [False, detail]

    for name in SCENARIOS:
        row = records.get(name)
        if not isinstance(row, Mapping):
            for key in rules:
                fail(key, f"{name}: missing scenario record")
            continue
        maps = _maps(row)
        total = 0
        origins = _packet_origin_slots(row.get("packets"))

        vehicles = row.get("vehicles")
        if not isinstance(vehicles, list):
            fail("actions", f"{name}: vehicles must be a list")
        for vehicle_id, vehicle in maps["vehicles"].items():
            if vehicle_id not in VEHICLES or not isinstance(vehicle, Mapping):
                fail("actions", f"{name}: unknown or malformed vehicle")
                continue
            allowed = AERIAL_ACTIONS if vehicle_id == "A0" else MARINE_ACTIONS
            intended = vehicle.get("intended_actions")
            executed = vehicle.get("executed_actions")
            for field, action_rows in (("intended_actions", intended), ("executed_actions", executed)):
                if not isinstance(action_rows, list):
                    fail("actions", f"{name}: {vehicle_id}.{field} is not a list")
                    continue
                for action_row in action_rows:
                    if not isinstance(action_row, Mapping) or action_row.get("action") not in allowed:
                        fail("actions", f"{name}: invalid {field} action")

            # The intended-action restriction for UUVs is based on the estimate
            # available at the decision boundary, including slot-t arrivals/hits.
            if vehicle_id in UUVS:
                intended_by_slot, _ = _slot_map(intended, H - 1)
                trace_by_slot, _ = _slot_map(vehicle.get("state_trace"), H)
                for slot in range(H):
                    action_row = intended_by_slot.get(slot)
                    if not isinstance(action_row, Mapping):
                        continue
                    action = action_row.get("action")
                    options = _estimate_options(row, vehicle_id, slot)
                    if not options:
                        if action != "stay":
                            fail("actions", f"{name}: {vehicle_id} non-stay without fresh estimate at {slot}")
                        continue
                    if action == "stay":
                        continue
                    # Only the final update in the deterministic boundary order
                    # is the latest estimate available to this decision.  Earlier
                    # same-slot packets must not make an action toward a stale
                    # estimate look legal.
                    estimate_x, estimate_y, _, _, _ = options[-1]
                    boundary = trace_by_slot.get(slot)
                    if not isinstance(boundary, Mapping) or not (
                        _integer(boundary.get("x"), 0, GRID_X - 1)
                        and _integer(boundary.get("y"), 0, GRID_Y - 1)
                    ):
                        fail("actions", f"{name}: {vehicle_id} missing position at {slot}")
                        continue
                    position = (boundary["x"], boundary["y"])
                    projected, destination = _project_move(
                        name, position, action, slot, underwater=True
                    )
                    if projected == "stay" or not (
                        abs(destination[0] - estimate_x) + abs(destination[1] - estimate_y)
                        < abs(position[0] - estimate_x) + abs(position[1] - estimate_y)
                    ):
                        fail("actions", f"{name}: {vehicle_id} action not toward fresh estimate at {slot}")

        observations = row.get("observations")
        if not isinstance(observations, list):
            fail("observations", f"{name}: observations must be a list")
        for obs in maps["observations"].values():
            if not isinstance(obs, Mapping) or (
                obs.get("kind") not in ("scan", "sonar", "none")
                or not isinstance(obs.get("cells"), list)
                or not isinstance(obs.get("hit"), bool)
                or (obs.get("hit") and (obs.get("observed_x") is None or obs.get("observed_y") is None))
                or (not obs.get("hit") and (obs.get("observed_x") is not None or obs.get("observed_y") is not None))
            ):
                fail("observations", f"{name}: invalid observation semantics")

        services = row.get("service_slots")
        if not isinstance(services, list):
            fail("services", f"{name}: service_slots must be a list")
        for service in maps["services"].values():
            if not isinstance(service, Mapping):
                fail("services", f"{name}: malformed service row")
                continue
            channel = service.get("channel")
            units = service.get("served_units")
            selected = service.get("selected_link")
            packet_id = service.get("packet_id")
            slot = service.get("slot")
            cap = 4 if channel == "wireless" else 2 if channel == "acoustic" else 0
            if channel not in ("wireless", "acoustic") or not _integer(slot, 0, H - 1) or not _integer(units, 0, cap):
                fail("services", f"{name}: invalid service {channel!r}/{slot!r}")
                continue
            total += units
            if units == 0:
                if selected is not None or packet_id is not None:
                    fail("services", f"{name}: idle service fields")
                continue
            if selected not in LINKS:
                fail("services", f"{name}: selected link")
                continue
            if (channel == "wireless" and selected != "A0-S0") or (
                channel == "acoustic" and selected == "A0-S0"
            ):
                fail("services", f"{name}: channel/link mismatch")
            try:
                packet = maps["packets"].get(packet_id)
            except (TypeError, AttributeError):
                packet = None
            if not isinstance(packet, Mapping) or packet.get("admitted_slot") is None:
                fail("services", f"{name}: service names unknown or rejected packet")
            elif packet.get("destination") not in VEHICLES:
                fail("services", f"{name}: service packet endpoint")

            # Service follows the automatic scan and precedes movement.  Use the boundary state
            # and the automatic scan, without trusting candidate totals.  The
            # public slot order charges service before the selected move.
            sender = LINK_ENDPOINTS[selected][0]
            vehicle = maps["vehicles"].get(sender)
            trace_by_slot, _ = _slot_map(
                vehicle.get("state_trace", []) if isinstance(vehicle, Mapping) else [], H
            )
            start = trace_by_slot.get(slot)
            if not isinstance(start, Mapping):
                fail("services", f"{name}: missing sender energy trace at {slot}")
            else:
                start_energy = _safe_float(start.get("energy"))
                if start_energy is None:
                    fail("services", f"{name}: malformed sender state at {slot}")
                else:
                    available_energy = start_energy - (0.25 if sender == "A0" else 0.0)
                    if units * 0.05 > available_energy + EPS:
                        fail("services", f"{name}: unaffordable service must be idle")

        packets = row.get("packets")
        if not isinstance(packets, list):
            fail("packets", f"{name}: packets must be a list")
        for p in maps["packets"].values():
            pid = p.get("packet_id") if isinstance(p, Mapping) else None
            valid_id = (
                isinstance(pid, int) and not isinstance(pid, bool) and pid >= 0
            ) or (isinstance(pid, str) and PACKET_ID_RE.fullmatch(pid) is not None)
            route = (p.get("source"), p.get("destination")) if isinstance(p, Mapping) else None
            if not isinstance(p, Mapping) or not valid_id or route not in (
                ("A0", "S0"), ("S0", "D0"), ("S0", "D1"), ("S0", "D2")
            ):
                fail("packets", f"{name}: invalid packet identity or route")
                continue
            granularity = p.get("granularity")
            ex, ey = p.get("estimate_x"), p.get("estimate_y")
            payload = p.get("payload_units")
            if (
                granularity not in ("coarse", "fine")
                or not _integer(ex, 0, GRID_X - 1)
                or not _integer(ey, 0, GRID_Y - 1)
                or (granularity == "coarse" and (ex % 4 or ey % 4))
                or payload != (1 if granularity == "coarse" else 3)
                or not _integer(p.get("created_slot"), 0, H - 1)
                or p.get("admitted_slot") is not None
                and p.get("admitted_slot") != p.get("created_slot")
                or p.get("status") not in PACKET_STATUSES
            ):
                fail("packets", f"{name}: invalid packet bounds or payload {pid!r}")
            if p.get("source") == "A0" and p.get("parent_id") is not None:
                fail("packets", f"{name}: root packet has parent {pid!r}")
            if p.get("source") == "S0" and p.get("parent_id") is None:
                fail("packets", f"{name}: copy packet lacks parent {pid!r}")
            if p.get("admitted_slot") is None and (
                p.get("status") != "rejected"
                or p.get("first_service_slot") is not None
                or p.get("last_service_slot") is not None
                or p.get("arrival_slot") is not None
                or p.get("used_slots") != []
            ):
                fail("packets", f"{name}: rejected packet lifecycle {pid!r}")

        queues = row.get("queues")
        if not isinstance(queues, list):
            fail("queues", f"{name}: queues must be a list")
        for q in queues if isinstance(queues, list) else []:
            if not isinstance(q, Mapping):
                fail("queues", f"{name}: malformed queue")
                continue
            packet_ids = q.get("packet_ids")
            rejected_ids = q.get("rejected_packet_ids")
            try:
                unique_packet_ids = len(packet_ids) == len({_hashable(pid) for pid in packet_ids}) if isinstance(packet_ids, list) else False
                known_packets = all(pid in maps["packets"] for pid in packet_ids) if isinstance(packet_ids, list) else False
                unique_rejected = len(rejected_ids) == len({_hashable(pid) for pid in rejected_ids}) if isinstance(rejected_ids, list) else False
            except Exception:
                unique_packet_ids = known_packets = unique_rejected = False
            if (
                not isinstance(packet_ids, list)
                or not unique_packet_ids
                or not known_packets
                or not _integer(q.get("occupied_units"), 0, 12)
                or not isinstance(rejected_ids, list)
                or not unique_rejected
                or any(pid in packet_ids for pid in rejected_ids)
            ):
                fail("queues", f"{name}: queue capacity or identity")
        if total > 180:
            fail("budget", f"{name}: report budget")
    return [
        AtomVerdict(
            f"decision_legality.{k}",
            bool(v[0]),
            str(v[1]),
            ATOM_SHARES["decision_legality"][f"decision_legality.{k}"],
        )
        for k, v in rules.items()
    ]


def _danger(name: str, cell: tuple[int, int]) -> bool:
    return any(
        x0 - 1 <= cell[0] <= x1 + 1 and y0 - 1 <= cell[1] <= y1 + 1
        for x0, x1, y0, y1 in OBSTACLES[name]
    ) and not _wall(name, cell)


def _simulate(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Replay public motion, automatic sensing, energy and capture semantics."""
    maps = _maps(row)
    positions = {v: STARTS[name][v] for v in VEHICLES}
    energies = {v: float(ENERGY[name][v]) for v in VEHICLES}
    target = STARTS[name]["target"]
    phase = "search"
    first_detection = None
    capture_run = 0
    capture_complete = False
    position_trace = {v: [positions[v]] for v in VEHICLES}
    energy_trace = {v: [energies[v]] for v in VEHICLES}
    target_trace = [target]
    conditions: list[bool] = []
    executed: dict[str, list[str]] = {v: [] for v in VEHICLES}
    observation_truth: dict[
        tuple[str, int], tuple[str, list[list[int]], bool, int | None, int | None]
    ] = {}
    phase_trace = [phase]
    capture_runs = [0]
    completion_trace = [False]
    total_energy = [sum(energies.values())]
    safety_events = 0
    safety_trace = [0]
    safety_events_by_vehicle = {v: 0 for v in VEHICLES}

    def mark_safety(vehicle_id: str) -> None:
        nonlocal safety_events
        safety_events += 1
        if vehicle_id in safety_events_by_vehicle:
            safety_events_by_vehicle[vehicle_id] += 1

    aoi = [[24 for _ in range(GRID_Y)] for _ in range(GRID_X)]
    search_aoi_samples: list[float] = []
    for slot in range(H):
        slot_phase = phase
        hit_any = False
        for x in range(GRID_X):
            for y in range(GRID_Y):
                aoi[x][y] = min(24, aoi[x][y] + 1)
        # Automatic probes and local sensing occur at the boundary before decisions.
        for vehicle_id in VEHICLES:
            if vehicle_id == "A0":
                x0, y0 = positions[vehicle_id]
                cells = [
                    [x, y]
                    for x in range(max(0, x0 - 2), min(GRID_X, x0 + 3))
                    for y in range(max(0, y0 - 2), min(GRID_Y, y0 + 3))
                ]
                hit = [target[0], target[1]] in cells
                observation_truth[(vehicle_id, slot)] = (
                    "scan",
                    cells,
                    hit,
                    target[0] if hit else None,
                    target[1] if hit else None,
                )
            elif vehicle_id in UUVS:
                hit = (
                    max(
                        abs(positions[vehicle_id][0] - target[0]),
                        abs(positions[vehicle_id][1] - target[1]),
                    )
                    <= 2
                )
                observation_truth[(vehicle_id, slot)] = (
                    "sonar",
                    [],
                    hit,
                    target[0] if hit else None,
                    target[1] if hit else None,
                )
            else:
                hit = False
                observation_truth[(vehicle_id, slot)] = ("none", [], False, None, None)
            hit_any = hit_any or hit
        # Target action is selected at the same decision boundary as vehicle actions,
        # before any of those actions are executed.
        target_next = _target_step(name, target, positions, slot_phase, slot)
        # Automatic A0 scanning is charged before service.  This is the explicit
        # slot order used by the task: arrivals/observations, decisions, scan,
        # service, then the already-selected vehicle and target moves.
        if energies["A0"] < 0.25 - EPS:
            mark_safety("A0")
        energies["A0"] = max(0.0, energies["A0"] - 0.25)
        for x, y in observation_truth[("A0", slot)][1]:
            aoi[x][y] = 0
        if slot_phase == "search":
            search_aoi_samples.append(sum(sum(column) for column in aoi) / (GRID_X * GRID_Y))
        for channel in ("wireless", "acoustic"):
            service = maps["services"].get((channel, slot), {})
            units = service.get("served_units", 0)
            selected_link = service.get("selected_link")
            if _integer(units, 1, 4) and selected_link in LINK_ENDPOINTS:
                sender = LINK_ENDPOINTS[selected_link][0]
                charge = 0.05 * units
                if charge > energies[sender] + EPS:
                    mark_safety(sender)
                else:
                    energies[sender] -= charge
        # Execute the submitted motion decision, projecting current and obstacle legality.
        for v in VEHICLES:
            vehicle_row = maps["vehicles"].get(v, {})
            actions = vehicle_row.get("intended_actions", []) if isinstance(vehicle_row, Mapping) else []
            action_rows, _ = _slot_map(actions, H - 1)
            action_row = action_rows.get(slot)
            action = action_row.get("action", "stay") if isinstance(action_row, Mapping) else "stay"
            if v == "A0":
                dx, dy = ACTION_DELTAS.get(action, (0, 0))
                candidate = (positions[v][0] + dx, positions[v][1] + dy)
                selected = (
                    action
                    if action in AERIAL_ACTIONS
                    and 0 <= candidate[0] < GRID_X
                    and 0 <= candidate[1] < GRID_Y
                    else "stay"
                )
                if selected == "stay":
                    candidate = positions[v]
                cost = float(abs(dx) + abs(dy)) if selected != "stay" else 0.0
            else:
                selected, candidate = _project_move(
                    name, positions[v], action, slot, underwater=v in UUVS
                )
                cost = (1.2 if v == "S0" else 1.5) if selected != "stay" else 0.0
            if cost > energies[v] + EPS:
                if action != "stay":
                    mark_safety(v)
                selected, candidate, cost = "stay", positions[v], 0.0
            if candidate in [positions[o] for o in VEHICLES if o != v]:
                mark_safety(v)
            if _danger(name, candidate) or (v in UUVS and _wall(name, candidate)):
                mark_safety(v)
            positions[v] = candidate
            energies[v] = max(0.0, energies[v] - cost)
            executed[v].append(selected)
            position_trace[v].append(candidate)
        target = target_next
        if hit_any and phase == "search":
            first_detection = slot if first_detection is None else first_detection
            phase = "hunting"
        condition = all(
            max(abs(positions[d][0] - target[0]), abs(positions[d][1] - target[1])) <= 2
            for d in UUVS
        )
        condition = condition and all(
            max(abs(positions[a][0] - positions[b][0]), abs(positions[a][1] - positions[b][1])) >= 2
            for i, a in enumerate(UUVS)
            for b in UUVS[i + 1 :]
        )
        conditions.append(condition)
        capture_run = capture_run + 1 if condition else 0
        if capture_run >= 4:
            capture_complete = True
        target_trace.append(target)
        phase_trace.append(phase)
        capture_runs.append(capture_run)
        completion_trace.append(capture_complete)
        # A boundary row is post-event evidence: movement, the automatic scan and
        # any affordable service in the slot have all been charged before the next
        # boundary is recorded.
        for v in VEHICLES:
            energy_trace[v].append(energies[v])
        total_energy.append(sum(energies.values()))
        safety_trace.append(safety_events)
    return {
        "positions": position_trace,
        "energies": energy_trace,
        "target": target_trace,
        "conditions": conditions,
        "capture_run": capture_run,
        "capture_complete": capture_complete,
        "phase": phase,
        "first_detection": first_detection,
        "executed": executed,
        "observations": observation_truth,
        "phase_trace": phase_trace,
        "capture_runs": capture_runs,
        "capture_complete_trace": completion_trace,
        "total_energy": total_energy,
        "safety_events": safety_trace,
        "safety_events_by_vehicle": safety_events_by_vehicle,
        "search_aoi_samples": search_aoi_samples,
    }


def _replay_packets(
    name: str, row: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[bool, str]:
    """Replay packet creation, FIFO evolution, service outcomes and arrivals.

    Packet/queue rows are post-event evidence, so this routine reconstructs the
    queue at every slot instead of trusting a candidate's ``served_packet_id`` or
    packet status.  It deliberately returns a diagnostic rather than propagating
    malformed-input exceptions; shape/type atoms remain responsible for reporting
    the field-level defect.
    """
    try:
        if not isinstance(row, Mapping) or not isinstance(state, Mapping):
            return False, f"{name}: malformed scenario/state container"
        packets_raw = row.get("packets")
        services_raw = row.get("service_slots")
        queues_raw = row.get("queues")
        if not isinstance(packets_raw, list) or not isinstance(services_raw, list) or not isinstance(queues_raw, list):
            return False, f"{name}: packet/service/queue arrays must be lists"

        packets: list[Mapping[str, Any]] = []
        by_id: dict[Hashable, Mapping[str, Any]] = {}
        for index, packet in enumerate(packets_raw):
            if not isinstance(packet, Mapping):
                return False, f"{name}: malformed packet row {index}"
            pid = packet.get("packet_id")
            if isinstance(pid, bool):
                return False, f"{name}: boolean packet id at {index}"
            try:
                hash(pid)
            except Exception:
                return False, f"{name}: unhashable packet id at {index}"
            if pid in by_id:
                return False, f"{name}: duplicate packet id {pid!r}"
            by_id[pid] = packet
            packets.append(packet)
            route = (packet.get("source"), packet.get("destination"))
            if route not in (
                ("A0", "S0"),
                ("S0", "D0"),
                ("S0", "D1"),
                ("S0", "D2"),
            ):
                return False, f"{name}: invalid packet route {pid!r}"
            if not _integer(packet.get("created_slot"), 0, H - 1):
                return False, f"{name}: invalid creation slot {pid!r}"
            if not _integer(packet.get("payload_units"), 1, 3):
                return False, f"{name}: invalid payload units {pid!r}"
            granularity = packet.get("granularity")
            if granularity not in ("coarse", "fine"):
                return False, f"{name}: invalid granularity {pid!r}"
            if granularity == "coarse" and (
                not _integer(packet.get("estimate_x"), 0, GRID_X - 1)
                or not _integer(packet.get("estimate_y"), 0, GRID_Y - 1)
                or packet.get("estimate_x") % 4 != 0
                or packet.get("estimate_y") % 4 != 0
            ):
                return False, f"{name}: coarse estimate is not block lower-left {pid!r}"
            if granularity == "fine" and not (
                _integer(packet.get("estimate_x"), 0, GRID_X - 1)
                and _integer(packet.get("estimate_y"), 0, GRID_Y - 1)
            ):
                return False, f"{name}: fine estimate is out of bounds {pid!r}"
            expected_payload = 1 if granularity == "coarse" else 3
            if packet.get("payload_units") != expected_payload:
                return False, f"{name}: granularity/payload mismatch {pid!r}"
            if packet.get("admitted_slot") is not None and not _integer(
                packet.get("admitted_slot"), 0, H - 1
            ):
                return False, f"{name}: invalid admission slot {pid!r}"
            for field in ("first_service_slot", "last_service_slot", "arrival_slot"):
                value = packet.get(field)
                if value is not None and not _integer(value, 0, H - 1):
                    return False, f"{name}: invalid {field} {pid!r}"
            if not _integer(packet.get("retransmissions"), 0, 180):
                return False, f"{name}: invalid retransmission count {pid!r}"
            used_slots = packet.get("used_slots")
            if not isinstance(used_slots, list) or any(
                not _integer(value, 0, H - 1) for value in used_slots
            ) or len(used_slots) != len(set(used_slots)):
                return False, f"{name}: invalid used slots {pid!r}"
            if packet.get("status") not in PACKET_STATUSES:
                return False, f"{name}: invalid packet status {pid!r}"
            if packet.get("source") == "A0" and packet.get("parent_id") is not None:
                return False, f"{name}: root report has parent {pid!r}"

        # Parent links are a causal relation: a copy can only be created after its
        # A0->S0 parent has arrived, and it carries the same estimate/granularity.
        origin_cache: dict[Hashable, int] = {}
        visiting: set[Hashable] = set()

        def origin(pid: Hashable) -> int | None:
            try:
                hash(pid)
            except Exception:
                return None
            if pid in origin_cache:
                return origin_cache[pid]
            if pid in visiting:
                return None
            packet = by_id.get(pid)
            if packet is None:
                return None
            visiting.add(pid)
            if packet.get("source") == "A0":
                value = packet.get("created_slot")
            else:
                parent_id = packet.get("parent_id")
                parent = by_id.get(parent_id)
                value = origin(parent_id) if parent is not None else None
            visiting.discard(pid)
            if _integer(value, 0, H - 1):
                origin_cache[pid] = value
                return value
            return None

        for packet in packets:
            pid = packet.get("packet_id")
            if packet.get("source") == "S0":
                parent_id = packet.get("parent_id")
                parent = by_id.get(parent_id)
                if parent is None or parent.get("source") != "A0" or parent.get("destination") != "S0":
                    return False, f"{name}: copy parent missing or wrong route {pid!r}"
                if not _integer(parent.get("arrival_slot"), 0, H - 1):
                    return False, f"{name}: copy before parent arrival {pid!r}"
                if packet.get("created_slot") < parent.get("arrival_slot"):
                    return False, f"{name}: copy created before parent arrival {pid!r}"
                if (packet.get("granularity"), packet.get("estimate_x"), packet.get("estimate_y")) != (
                    parent.get("granularity"), parent.get("estimate_x"), parent.get("estimate_y")
                ):
                    return False, f"{name}: copy payload differs from parent {pid!r}"
                if origin(pid) is None:
                    return False, f"{name}: invalid parent chain {pid!r}"
            elif packet.get("parent_id") is not None:
                return False, f"{name}: root parent relation {pid!r}"

        # A root report must be grounded in an actually observed A0 hit.  The
        # A packet may be generated arbitrarily later; an over-age report simply
        # cannot control a UUV.  Hidden target coordinates still cannot create a
        # report by themselves.
        observation_map = _maps(row)["observations"]
        for packet in packets:
            if packet.get("source") != "A0":
                continue
            created = packet.get("created_slot")
            ex, ey = packet.get("estimate_x"), packet.get("estimate_y")
            grounded = False
            for hit_slot in range(H):
                obs = observation_map.get(("A0", hit_slot))
                if not isinstance(obs, Mapping) or not obs.get("hit") or hit_slot > created:
                    continue
                ox, oy = obs.get("observed_x"), obs.get("observed_y")
                if packet.get("granularity") == "fine" and (ex, ey) == (ox, oy):
                    grounded = True
                if packet.get("granularity") == "coarse" and (ex, ey) == (
                    (ox // 4) * 4,
                    (oy // 4) * 4,
                ):
                    grounded = True
                if grounded:
                    break
            if not grounded:
                return False, f"{name}: root report lacks local A0 hit {packet.get('packet_id')!r}"

        service_by_key: dict[tuple[Any, Any], Mapping[str, Any]] = {}
        for index, service in enumerate(services_raw):
            if not isinstance(service, Mapping):
                return False, f"{name}: malformed service row {index}"
            key = (service.get("channel"), service.get("slot"))
            if key in service_by_key:
                return False, f"{name}: duplicate service key {key!r}"
            service_by_key[key] = service
        queue_by_key: dict[tuple[Any, Any], Mapping[str, Any]] = {}
        for index, queue in enumerate(queues_raw):
            if not isinstance(queue, Mapping):
                return False, f"{name}: malformed queue row {index}"
            key = (queue.get("link_id"), queue.get("slot"))
            if key in queue_by_key:
                return False, f"{name}: duplicate queue key {key!r}"
            queue_by_key[key] = queue

        queues: dict[str, list[Hashable]] = {link: [] for link in LINKS}
        remaining = {pid: int(packet.get("payload_units", 0)) for pid, packet in by_id.items()}
        used: dict[Hashable, list[int]] = {pid: [] for pid in by_id}
        failures: dict[Hashable, int] = {pid: 0 for pid in by_id}
        first_service: dict[Hashable, int | None] = {pid: None for pid in by_id}
        last_service: dict[Hashable, int | None] = {pid: None for pid in by_id}
        expected_arrival: dict[Hashable, int | None] = {pid: None for pid in by_id}
        status: dict[Hashable, str] = {
            pid: ("rejected" if packet.get("admitted_slot") is None else "queued")
            for pid, packet in by_id.items()
        }
        pending: dict[int, list[Hashable]] = {}
        rejected_at: dict[tuple[str, int], list[Hashable]] = {}
        total_attempted = 0

        created_by_slot: dict[int, list[Hashable]] = {slot: [] for slot in range(H)}
        for packet in packets:
            created_by_slot[packet["created_slot"]].append(packet["packet_id"])

        def service_energy(name_: str, link_: str, slot_: int) -> float | None:
            """Energy available after the automatic scan and before movement."""
            try:
                sender = LINK_ENDPOINTS[link_][0]
                traces = state.get("energies", {}).get(sender)
                if not isinstance(traces, list):
                    return None
                if slot_ >= len(traces):
                    return None
                start_energy = traces[slot_]
                if not _finite(start_energy):
                    return None
                before = float(start_energy)
                return before - (0.25 if sender == "A0" else 0.0)
            except (KeyError, TypeError, ValueError, OverflowError):
                return None

        for slot in range(H):
            # Deliver completed hops at the boundary before admission/service.
            for pid in pending.pop(slot, []):
                packet = by_id[pid]
                root_slot = origin(pid)
                if root_slot is None:
                    return False, f"{name}: missing origin for pending packet {pid!r}"
                # Age limits control whether an arrival may update a vehicle; they
                # do not erase a physically completed hop.  Only packets still in
                # a FIFO are expired before service.  Every completed hop arriving
                # before boundary H is therefore retained as arrival evidence,
                # even when its clipped age is 12 and it cannot control.
                status[pid] = "arrived"
                expected_arrival[pid] = slot

            # Expire queued packets before offering new packets or serving the head.
            for link in LINKS:
                retained: list[Hashable] = []
                for pid in queues[link]:
                    root_slot = origin(pid)
                    if root_slot is None or slot - root_slot > 12:
                        status[pid] = "expired"
                    else:
                        retained.append(pid)
                queues[link] = retained

            # Offer all packets created at this slot.  The physical order of the
            # dynamic packet array is immaterial; packets offered on one link at
            # the same boundary are appended in packet-ID JSON order.  Existing
            # FIFO members always stay ahead of newly admitted packets.
            for link in LINKS:
                created = [
                    pid
                    for pid in created_by_slot[slot]
                    if ("A0-S0" if by_id[pid].get("destination") == "S0" else f"S0-{by_id[pid].get('destination')}") == link
                ]
                if any(
                    by_id[pid].get("admitted_slot") not in (None, slot) for pid in created
                ):
                    return False, f"{name}: packet admitted outside creation slot {link}/{slot}"
                qrow = queue_by_key.get((link, slot))
                if qrow is None or not isinstance(qrow.get("packet_ids"), list):
                    return False, f"{name}: missing FIFO evidence {link}/{slot}"
                snapshot_ids = qrow.get("packet_ids", [])
                try:
                    snapshot_keys = [_hashable(pid) for pid in snapshot_ids]
                    if len(snapshot_keys) != len(set(snapshot_keys)):
                        return False, f"{name}: duplicate FIFO IDs {link}/{slot}"
                except Exception:
                    return False, f"{name}: malformed FIFO IDs {link}/{slot}"
                if any(pid not in by_id for pid in snapshot_ids):
                    return False, f"{name}: unknown FIFO ID {link}/{slot}"
                if any(
                    ("A0-S0" if by_id[pid].get("destination") == "S0" else f"S0-{by_id[pid].get('destination')}")
                    != link
                    for pid in snapshot_ids
                ):
                    return False, f"{name}: FIFO route mismatch {link}/{slot}"

                prior_queue = list(queues[link])
                capacity_left = 12 - sum(remaining[pid] for pid in prior_queue)
                if capacity_left < 0:
                    return False, f"{name}: prior FIFO over capacity {link}/{slot}"
                active_accepted: list[Hashable] = []
                expected_rejected: list[Hashable] = []
                # Admission is deterministic for ordinary packets: the first
                # packets in canonical ID order that fit are accepted and later
                # ones are rejected.  A packet already beyond the age horizon is
                # recorded as an admission event and expires immediately, so it
                # never consumes active FIFO capacity.
                for pid in sorted(created, key=_packet_id_sort_key):
                    packet = by_id[pid]
                    root_slot = origin(pid)
                    if root_slot is None:
                        return False, f"{name}: missing root for packet {pid!r}"
                    if slot - root_slot > 12:
                        if packet.get("admitted_slot") != slot:
                            return False, f"{name}: over-age packet lacks admission event {pid!r}"
                        status[pid] = "expired"
                    elif remaining[pid] <= capacity_left:
                        if packet.get("admitted_slot") != slot:
                            return False, f"{name}: fitting packet was rejected {pid!r}"
                        active_accepted.append(pid)
                        capacity_left -= remaining[pid]
                    else:
                        if packet.get("admitted_slot") is not None:
                            return False, f"{name}: over-capacity packet was admitted {pid!r}"
                        expected_rejected.append(pid)
                rejected = sorted(
                    (pid for pid in created if by_id[pid].get("admitted_slot") is None),
                    key=_packet_id_sort_key,
                )
                if rejected != expected_rejected:
                    return False, f"{name}: non-canonical rejection order {link}/{slot}"
                rejected_at[(link, slot)] = list(expected_rejected)
                for pid in active_accepted:
                    status[pid] = "queued"
                for pid in rejected:
                    status[pid] = "rejected"
                # No candidate-provided queue row is used to reconstruct the
                # pre-service order: doing so would permit a new packet to jump
                # ahead of an older head.  The post-service row is checked below
                # against this replayed queue after the service outcome.
                queues[link] = prior_queue + active_accepted

            # Serve the current FIFO head for each channel.  The canonical service
            # rows make acoustic sharing explicit (one row, capacity two).
            for channel, capacity in (("wireless", 4), ("acoustic", 2)):
                service = service_by_key.get((channel, slot))
                if service is None:
                    return False, f"{name}: missing service row {channel}/{slot}"
                units = service.get("served_units")
                if not _integer(units, 0, capacity):
                    return False, f"{name}: service capacity {channel}/{slot}"
                selected = service.get("selected_link")
                pid = service.get("packet_id")
                if units == 0:
                    if selected is not None or pid is not None:
                        return False, f"{name}: idle service names packet {channel}/{slot}"
                    continue
                if selected not in LINK_ENDPOINTS:
                    return False, f"{name}: service identity {channel}/{slot}"
                try:
                    serviced_packet = by_id.get(pid)
                except (TypeError, AttributeError):
                    serviced_packet = None
                if serviced_packet is None:
                    return False, f"{name}: service identity {channel}/{slot}"
                link = str(selected)
                expected_link = "A0-S0" if serviced_packet.get("destination") == "S0" else f"S0-{serviced_packet.get('destination')}"
                if link != expected_link or (channel == "wireless") != (link == "A0-S0"):
                    return False, f"{name}: service route {channel}/{slot}"
                if not queues[link] or queues[link][0] != pid:
                    return False, f"{name}: service is not current FIFO head {link}/{slot}"
                positions = {
                    v: state.get("positions", {}).get(v, [])[slot] for v in VEHICLES
                }
                if any(not isinstance(value, (tuple, list)) or len(value) != 2 for value in positions.values()):
                    return False, f"{name}: missing replay positions {link}/{slot}"
                if not _link_available(name, link, slot, positions):
                    return False, f"{name}: unavailable service {link}/{slot}"
                if units > remaining[pid]:
                    return False, f"{name}: service exceeds remaining payload {pid!r}"
                available_energy = service_energy(name, link, slot)
                if available_energy is None:
                    return False, f"{name}: missing service energy {link}/{slot}"
                if units * 0.05 > available_energy + EPS:
                    return False, f"{name}: unaffordable non-idle service {link}/{slot}"
                total_attempted += units
                used[pid].append(slot)
                if first_service[pid] is None:
                    first_service[pid] = slot
                last_service[pid] = slot
                status[pid] = "in_service"
                if _success(name, link, slot, positions):
                    remaining[pid] -= units
                    if remaining[pid] == 0:
                        queues[link].pop(0)
                        delay = _link_delay(name, link, slot, positions)
                        arrival = slot + delay
                        if arrival < H:
                            pending.setdefault(arrival, []).append(pid)
                        else:
                            status[pid] = "lost"
                else:
                    failures[pid] += units

            # Queue rows are end-of-slot snapshots, including the current service
            # result and IDs rejected at this creation boundary.
            for link in LINKS:
                q = queue_by_key.get((link, slot))
                if q is None:
                    return False, f"{name}: missing queue row {link}/{slot}"
                expected_rejected = rejected_at.get((link, slot), [])
                if q.get("packet_ids") != list(queues[link]):
                    return False, f"{name}: FIFO snapshot {link}/{slot}"
                occupied = sum(remaining[pid] for pid in queues[link])
                if q.get("occupied_units") != occupied:
                    return False, f"{name}: FIFO occupancy {link}/{slot}"
                channel = "wireless" if link == "A0-S0" else "acoustic"
                service = service_by_key[(channel, slot)]
                expected_served = (
                    service.get("packet_id")
                    if service.get("served_units", 0) and service.get("selected_link") == link
                    else None
                )
                if q.get("served_packet_id") != expected_served:
                    return False, f"{name}: queue served head {link}/{slot}"
                rejected_evidence = q.get("rejected_packet_ids")
                if (
                    not isinstance(rejected_evidence, list)
                    or len(rejected_evidence) != len(set(_hashable(pid) for pid in rejected_evidence))
                    or set(_hashable(pid) for pid in rejected_evidence)
                    != set(_hashable(pid) for pid in expected_rejected)
                ):
                    return False, f"{name}: rejected packet evidence {link}/{slot}"

        if total_attempted > 180:
            return False, f"{name}: served-unit budget"

        # A completed in-horizon hop is the only source of an arrival.  Packets
        # still queued/in service at H retain their derived status; no hidden
        # candidate fields are accepted as a substitute for this replay.
        for pid, packet in by_id.items():
            expected_status = status[pid]
            packet_used = packet.get("used_slots")
            if (
                not isinstance(packet_used, list)
                or len(packet_used) != len(set(packet_used))
                or sorted(packet_used) != sorted(used[pid])
            ):
                return False, f"{name}: used_slots {pid!r}"
            if packet.get("retransmissions") != failures[pid]:
                return False, f"{name}: retransmissions {pid!r}"
            if packet.get("first_service_slot") != first_service[pid] or packet.get("last_service_slot") != last_service[pid]:
                return False, f"{name}: service bounds {pid!r}"
            if packet.get("arrival_slot") != expected_arrival[pid]:
                return False, f"{name}: arrival evidence {pid!r}"
            if packet.get("status") != expected_status:
                return False, f"{name}: packet status {pid!r}: expected {expected_status!r}, actual {packet.get('status')!r}"
            if packet.get("admitted_slot") is None and any(used[pid]):
                return False, f"{name}: rejected packet was serviced {pid!r}"
            if packet.get("arrival_slot") is not None:
                final_slot = used[pid][-1] if used[pid] else -1
                link = "A0-S0" if packet.get("destination") == "S0" else f"S0-{packet.get('destination')}"
                positions = {v: state["positions"][v][final_slot] for v in VEHICLES}
                expected = final_slot + _link_delay(name, link, final_slot, positions)
                if packet["arrival_slot"] != expected or packet["arrival_slot"] >= H:
                    return False, f"{name}: arrival timing {pid!r}"
        return True, "packet replay consistent"
    except Exception as exc:
        return False, f"{name}: packet replay {type(exc).__name__}: {exc}"


def _replay_atoms(
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = {
        k: [True, msg]
        for k, msg in (
            ("motion", "motion consistent"),
            ("target", "target replay consistent"),
            ("observations", "observations consistent"),
            ("packets", "packet replay consistent"),
            ("energy", "energy replay consistent"),
            ("capture", "capture replay consistent"),
        )
    }
    states = {}
    for name in SCENARIOS:
        row = records.get(name)
        if row is None:
            continue
        try:
            state = _simulate(name, row)
            states[name] = state
        except Exception as exc:
            rules["motion"] = [False, f"{name}: {type(exc).__name__}: {exc}"]
            continue
        for v in VEHICLES:
            vehicle = _maps(row)["vehicles"].get(v, {})
            if not isinstance(vehicle, Mapping):
                rules["motion"] = [False, f"{name}: missing vehicle {v}"]
                continue
            actual = vehicle.get("executed_actions", [])
            if not isinstance(actual, list):
                actual = []
            actual_by_slot, actual_slot_ok = _slot_map(actual, H - 1)
            allowed = AERIAL_ACTIONS if v == "A0" else MARINE_ACTIONS
            if len(actual) != H or not actual_slot_ok or any(
                not isinstance(a, Mapping) or a.get("action") not in allowed for a in actual_by_slot.values()
            ):
                rules["motion"] = [False, f"{name}: executed action evidence"]
            for slot in range(H):
                action_row = actual_by_slot.get(slot)
                if not isinstance(action_row, Mapping):
                    continue
                if action_row.get("action") != state["executed"][v][slot]:
                    rules["motion"] = [False, f"{name}: {v} action {slot}"]
            trace_rows = vehicle.get("state_trace", [])
            if not isinstance(trace_rows, list):
                trace_rows = []
            trace_by_slot, trace_slot_ok = _slot_map(trace_rows, H)
            if len(trace_rows) != H + 1 or not trace_slot_ok:
                rules["motion"] = [False, f"{name}: {v} state slot coverage"]
            for t in range(H + 1):
                s = trace_by_slot.get(t)
                if not isinstance(s, Mapping):
                    rules["motion"] = [False, f"{name}: {v} malformed state {t}"]
                    continue
                if (s.get("x"), s.get("y")) != state["positions"][v][t]:
                    rules["motion"] = [False, f"{name}: {v} position {t}"]
                actual_energy = _safe_float(s.get("energy"))
                if actual_energy is None or not math.isclose(
                    actual_energy, state["energies"][v][t], abs_tol=EPS
                ):
                    rules["energy"] = [False, f"{name}: {v} energy {t}"]
            # A non-null estimate must be exactly the latest causally available
            # update at that boundary, with the clipped age implied by its source.
            # Once that source is over-age, the public contract permits either
            # retaining the clipped stale value or clearing all three fields.
            for t in range(H + 1):
                trace = trace_by_slot.get(t)
                if not isinstance(trace, Mapping):
                    continue
                estimate = (
                    trace.get("estimate_x"),
                    trace.get("estimate_y"),
                    trace.get("estimate_age"),
                )
                # A still-fresh control memory is not displaced by a later
                # over-age arrival.  If no fresh event remains, the latest
                # stale event supplies the optional stale representation.
                all_events = _historical_estimate_events(row, v, t, control_only=False)
                fresh_events = _historical_estimate_events(row, v, t, control_only=True)
                expected = fresh_events[-1] if fresh_events else (all_events[-1] if all_events else None)
                expected_stale = bool(expected is not None and not fresh_events)
                if expected is not None:
                    source_slot, _, _, _, source, packet_id, _granularity = expected
                    root_slot = source_slot
                    if source == "packet":
                        try:
                            root_slot = _packet_origin_slots(row.get("packets")).get(packet_id)
                        except (TypeError, ValueError):
                            root_slot = None
                    expected_stale = root_slot is not None and t - root_slot > 12
                if estimate == (None, None, None):
                    if expected is not None and not expected_stale:
                        rules["packets"] = [False, f"{name}: missing estimate provenance {v}/{t}"]
                    continue
                if (
                    estimate[0] is None
                    or estimate[1] is None
                    or estimate[2] is None
                    or expected is None
                    or (estimate[0], estimate[1]) != (expected[1], expected[2])
                    or estimate[2] != expected[3]
                ):
                    rules["packets"] = [False, f"{name}: stale or mis-aged estimate {v}/{t}"]
        mission_rows = row.get("mission_state", [])
        if not isinstance(mission_rows, list):
            mission_rows = []
        mission, mission_slot_ok = _slot_map(mission_rows, H)
        if len(mission_rows) != H + 1 or not mission_slot_ok:
            rules["capture"] = [False, f"{name}: mission slot coverage"]
        for t in range(H + 1):
            item = mission.get(t, {})
            if not isinstance(item, Mapping):
                rules["capture"] = [False, f"{name}: malformed mission {t}"]
                continue
            observed_xy = (item.get("observed_target_x"), item.get("observed_target_y"))
            latest_aggregate = _latest_aggregate_estimate(row, t)
            # Once a local hit or arrived estimate exists, the public aggregate
            # evidence is persistent.  Silently accepting null here would make
            # a required replay fact optional and disagree with task/Card.
            if observed_xy != (None, None) or latest_aggregate is not None:
                if observed_xy != latest_aggregate:
                    rules["capture"] = [False, f"{name}: aggregate observation {t}"]
            if item.get("phase") != state["phase_trace"][t]:
                rules["capture"] = [False, f"{name}: phase {t}"]
            expected_detection = (
                state["first_detection"]
                if state["first_detection"] is not None and t > state["first_detection"]
                else None
            )
            if item.get("first_detection_slot") != expected_detection:
                rules["capture"] = [False, f"{name}: first detection {t}"]
            expected_condition = state["conditions"][t - 1] if t > 0 else False
            if (
                item.get("capture_condition") != expected_condition
                or item.get("capture_run") != state["capture_runs"][t]
            ):
                rules["capture"] = [False, f"{name}: capture evidence {t}"]
            if item.get("capture_complete") != state["capture_complete_trace"][t]:
                rules["capture"] = [False, f"{name}: capture completion {t}"]
            if (
                not _integer(item.get("safety_events"), 0, 10**9)
                or item.get("safety_events") != state["safety_events"][t]
            ):
                rules["capture"] = [False, f"{name}: safety count {t}"]
            actual_total_energy = _safe_float(item.get("total_energy"))
            if actual_total_energy is None or not math.isclose(
                actual_total_energy, state["total_energy"][t], abs_tol=EPS
            ):
                rules["energy"] = [False, f"{name}: total energy {t}"]
        for key, truth in state["observations"].items():
            item = _maps(row)["observations"].get(key, {})
            expected = {
                "kind": truth[0],
                "cells": truth[1],
                "hit": truth[2],
                "observed_x": truth[3],
                "observed_y": truth[4],
            }
            if (
                item.get("kind") != expected["kind"]
                or _canonical_cells(item.get("cells")) != _canonical_cells(expected["cells"])
                or item.get("hit") != expected["hit"]
            ):
                rules["observations"] = [False, f"{name}: observation {key}"]
            if any(item.get(field) != expected[field] for field in ("observed_x", "observed_y")):
                rules["target"] = [False, f"{name}: target observation {key}"]
        # Probe rows are deterministic public observations of the replayed link quality.
        for link in LINKS:
            sender, receiver = LINK_ENDPOINTS[link]
            for slot in range(H):
                probe = _maps(row)["links"].get((link, slot), {})
                q = _link_quality(
                    name, link, slot, {v: state["positions"][v][slot] for v in VEHICLES}
                )
                available = q >= 0.2 - EPS
                expected_q = q if available else None
                actual_q = _safe_float(probe.get("observed_quality")) if isinstance(probe, Mapping) else None
                if (
                    (probe.get("sender"), probe.get("receiver")) != (sender, receiver)
                    or probe.get("observed_available") != available
                    or (expected_q is None and probe.get("observed_quality") is not None)
                    or (
                        expected_q is not None
                        and (actual_q is None or not math.isclose(actual_q, expected_q, abs_tol=EPS))
                    )
                ):
                    rules["packets"] = [False, f"{name}: link probe {link}/{slot}"]
        packet_ok, packet_detail = _replay_packets(name, row, state)
        if not packet_ok:
            rules["packets"] = [False, packet_detail]
    return [
        AtomVerdict(
            f"replay_validity.{k}",
            bool(v[0]),
            str(v[1]),
            ATOM_SHARES["replay_validity"][f"replay_validity.{k}"],
        )
        for k, v in rules.items()
    ], states


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def _quantized_contains(
    ex: Any, ey: Any, tx: Any, ty: Any, granularity: Any = None
) -> bool:
    """Whether an estimate of the declared granularity contains a target cell.

    Coordinates alone are insufficient because a fine report may happen to lie
    on a four-cell boundary.  The packet/local-event provenance therefore has to
    supply ``fine`` (exact cell) or ``coarse`` (4x4 block lower-left).
    """
    if not (
        _integer(ex, 0, GRID_X - 1)
        and _integer(ey, 0, GRID_Y - 1)
        and _integer(tx, 0, GRID_X - 1)
        and _integer(ty, 0, GRID_Y - 1)
    ):
        return False
    if granularity == "fine":
        return (ex, ey) == (tx, ty)
    if granularity == "coarse":
        return (
            ex % 4 == 0
            and ey % 4 == 0
            and ex <= tx < ex + 4
            and ey <= ty < ey + 4
        )
    return False


def _queue_wait_samples(packet_rows: Any) -> list[int]:
    """Return the contractual wait endpoint for every admitted packet.

    The endpoint is first service for a served packet, the first over-age
    boundary for an unserved packet that expires, and ``H`` for a packet that
    remains active at the horizon.  A packet admitted after its root has
    already aged out expires at its admission boundary and contributes zero.
    Keeping this rule in one helper makes the metric and author-side probes use
    exactly the same lifecycle interpretation.
    """
    if not isinstance(packet_rows, list):
        return []
    origins = _packet_origin_slots(packet_rows)
    waits: list[int] = []
    for packet in packet_rows:
        if not isinstance(packet, Mapping):
            continue
        admitted = packet.get("admitted_slot")
        if not _integer(admitted, 0, H - 1):
            continue
        first = packet.get("first_service_slot")
        if _integer(first, 0, H - 1) and first >= admitted:
            endpoint = first
        else:
            root_slot = _safe_origin_lookup(origins, packet.get("packet_id"))
            if root_slot is not None and admitted - root_slot > 12:
                endpoint = admitted
            elif packet.get("status") == "expired" and root_slot is not None:
                endpoint = min(H, root_slot + 13)
            else:
                endpoint = H
        waits.append(max(0, int(endpoint) - int(admitted)))
    return waits


def _metrics(
    states: Mapping[str, dict[str, Any]], records: Mapping[str, dict[str, Any]]
) -> dict[str, float]:
    vals = {
        "primary_metrics_1": [],
        "primary_metrics_2": [],
        "primary_metrics_3": [],
        "auxiliary_metrics_1": [],
        "auxiliary_metrics_2": [],
    }
    for name in SCENARIOS:
        s = states[name]
        deadline = DEADLINES[name][1]
        found = s["first_detection"]
        search_deadline = DEADLINES[name][0]
        timing = (
            _clip((search_deadline - found) / max(1, search_deadline)) if found is not None else 0.0
        )
        end_target = s["target"][deadline if deadline < len(s["target"]) else -1]
        end_dist = [
            max(
                abs(s["positions"][d][deadline][0] - end_target[0]),
                abs(s["positions"][d][deadline][1] - end_target[1]),
            )
            for d in UUVS
        ]
        capture_success = bool(
            any(s["capture_complete_trace"][t] for t in range(min(H, deadline) + 1))
        )
        # The incomplete-case run is the consecutive suffix ending at the last
        # slot-end sample strictly before the inclusive capture deadline.
        longest_run = 0
        for condition in reversed(s["conditions"][: min(H, deadline)]):
            if not condition:
                break
            longest_run += 1
        progress = (
            1.0
            if capture_success
            else 0.7 * sum(_clip(1 - d / 23) for d in end_dist) / 3
            + 0.3 * min(longest_run, 4) / 4
        )
        vals["primary_metrics_1"].append(
            _clip(
                0.55 * float(capture_success)
                + 0.25 * progress
                + 0.20 * timing
            )
        )
        row = records[name]
        termination = next(
            (t for t, complete in enumerate(s["capture_complete_trace"]) if complete), H
        )
        hunting_slots = [
            t for t in range(min(H, termination) + 1) if s["phase_trace"][t] == "hunting"
        ]
        fresh = 0
        shared = 0
        for t in hunting_slots:
            valid = 0
            for d in UUVS:
                options = _estimate_options(row, d, t)
                if options:
                    ex, ey, age, _, granularity = options[-1]
                else:
                    ex = ey = age = granularity = None
                tx, ty = s["target"][t]
                quantized = (
                    ex is not None
                    and ey is not None
                    and (
                        _quantized_contains(ex, ey, tx, ty, granularity)
                    )
                )
                if quantized and _integer(age, 0, 12):
                    valid += 1
            fresh += valid / 3
            shared += float(valid >= 2)
        vals["primary_metrics_2"].append(
            0.6 * fresh / max(1, len(hunting_slots)) + 0.4 * shared / max(1, len(hunting_slots))
        )
        motion_cost = sum(
            abs(s["positions"][v][t + 1][0] - s["positions"][v][t][0])
            + abs(s["positions"][v][t + 1][1] - s["positions"][v][t][1])
            for v in VEHICLES
            for t in range(H)
        )
        safety_by_vehicle = s.get("safety_events_by_vehicle", {})
        safety = sum(
            int(safety_by_vehicle.get(v, 0))
            for v in ("S0",) + UUVS
            if isinstance(safety_by_vehicle.get(v, 0), int)
            and not isinstance(safety_by_vehicle.get(v, 0), bool)
        )
        energy_used = sum(ENERGY[name][v] - s["energies"][v][-1] for v in VEHICLES)
        vals["primary_metrics_3"].append(
            _clip(
                0.4 * _clip(1 - safety / (4 * H))
                + 0.3 * _clip(1 - motion_cost / (6 * H))
                + 0.3 * _clip(1 - energy_used / sum(ENERGY[name].values()))
            )
        )
        service_rows = row.get("service_slots", [])
        sent = sum(
            int(x.get("served_units", 0))
            for x in service_rows
            if isinstance(x, Mapping) and _integer(x.get("served_units"), 0, 4)
        ) if isinstance(service_rows, list) else 0
        useful_units = 0
        on_time = 0
        packet_rows = row.get("packets", [])
        packet_rows = packet_rows if isinstance(packet_rows, list) else []
        generated = sum(
            1 for p in packet_rows if isinstance(p, Mapping) and p.get("source") == "A0"
        )
        origins = _packet_origin_slots(packet_rows)
        vehicle_maps = _maps(row)["vehicles"]
        for packet in packet_rows:
            if not isinstance(packet, Mapping) or packet.get("source") != "S0" or packet.get("destination") not in UUVS:
                continue
            arrival = packet.get("arrival_slot")
            pid = packet.get("packet_id")
            root_slot = _safe_origin_lookup(origins, pid)
            if (
                packet.get("status") != "arrived"
                or not _integer(arrival, 0, H - 1)
                or root_slot is None
                or arrival < root_slot
                or arrival - root_slot > 12
            ):
                continue
            target_at_arrival = s["target"][arrival]
            ex, ey = packet.get("estimate_x"), packet.get("estimate_y")
            granularity = packet.get("granularity")
            if not _quantized_contains(
                ex, ey, target_at_arrival[0], target_at_arrival[1], granularity
            ):
                continue
            # Every timely, correctly quantized copy contributes to on-time
            # reports, independently of whether it is later useful.
            on_time += 1
            destination = packet["destination"]
            vehicle_row = vehicle_maps.get(destination, {})
            traces = vehicle_row.get("state_trace", []) if isinstance(vehicle_row, Mapping) else []
            trace_by_slot, _ = _slot_map(traces, H)
            previous = trace_by_slot.get(arrival)
            current = trace_by_slot.get(arrival + 1)
            actions = vehicle_row.get("intended_actions", []) if isinstance(vehicle_row, Mapping) else []
            action_by_slot, _ = _slot_map(actions, H - 1)
            action_row = action_by_slot.get(arrival)
            if not isinstance(previous, Mapping) or not isinstance(current, Mapping) or not isinstance(action_row, Mapping):
                continue
            actual_xy = (current.get("estimate_x"), current.get("estimate_y"))
            focal_key = (ex, ey, granularity)
            actual_options = _estimate_options(row, destination, arrival)
            actual_key = (
                (actual_options[-1][0], actual_options[-1][1], actual_options[-1][4])
                if actual_options
                else None
            )
            counterfactual_options = _estimate_options(
                row, destination, arrival, exclude_packet=pid
            )
            counterfactual_key = (
                (
                    counterfactual_options[-1][0],
                    counterfactual_options[-1][1],
                    counterfactual_options[-1][4],
                )
                if counterfactual_options
                else None
            )
            changed = (
                actual_key == focal_key
                and actual_xy == (ex, ey)
                and actual_key != counterfactual_key
            )
            position = (previous.get("x"), previous.get("y"))
            if not (
                _integer(position[0], 0, GRID_X - 1)
                and _integer(position[1], 0, GRID_Y - 1)
            ):
                continue
            directed = _shortest_legal_tracking_step(
                name,
                (position[0], position[1]),
                action_row.get("action"),
                arrival,
                (ex, ey),
            )
            if changed and directed:
                useful_units += int(packet.get("payload_units", 0))
        vals["auxiliary_metrics_1"].append(
            _clip(0.6 * useful_units / max(1, sent) + 0.4 * on_time / max(1, generated))
        )
        aoi_mean = sum(s.get("search_aoi_samples", [])) / max(
            1, len(s.get("search_aoi_samples", []))
        )
        # Queue wait is defined for every *admitted* packet, including packets
        # that never receive a service attempt.  A served packet ends its wait
        # at first service; an accepted packet that expires before service ends
        # at the first over-age boundary; an accepted packet still active at H
        # ends at H.  A packet admitted already over-age has an immediate expiry
        # event and therefore contributes zero.  The validity/replay atoms have
        # already established the lifecycle for metric evaluation, but keep this
        # calculation defensive for malformed author-side probes.
        waits = _queue_wait_samples(packet_rows)
        mean_wait = sum(waits) / max(1, len(waits))
        demand_total = demand_available = 0
        demand_keys: set[tuple[str, int]] = set()
        for packet in packet_rows:
            if not isinstance(packet, Mapping) or packet.get("source") not in VEHICLES:
                continue
            link = _packet_link(packet)
            start = packet.get("admitted_slot")
            root_slot = _safe_origin_lookup(origins, packet.get("packet_id"))
            if link not in LINKS or not _integer(start, 0, H - 1) or root_slot is None:
                continue
            # A packet is a demand until the slot in which it completes/leaves,
            # or through the last slot before its age-based expiry.  Accepted
            # packets created already over-age have no demand sample.
            if start - root_slot > 12 or packet.get("status") == "rejected":
                continue
            status = packet.get("status")
            if status == "expired":
                end = min(H - 1, root_slot + 12)
            elif status in {"arrived", "lost"} and _integer(
                packet.get("last_service_slot"), 0, H - 1
            ):
                end = packet["last_service_slot"]
            else:
                end = H - 1
            if end < start:
                continue
            for slot in range(start, end + 1):
                demand_keys.add((link, slot))
        for link, slot in sorted(demand_keys, key=lambda item: (LINKS.index(item[0]), item[1])):
            demand_total += 1
            if _link_available(
                name, link, slot, {v: s["positions"][v][slot] for v in VEHICLES}
            ):
                demand_available += 1
        availability = demand_available / max(1, demand_total)
        vals["auxiliary_metrics_2"].append(
            _clip(
                0.4 * _clip(1 - aoi_mean / 24) + 0.3 * _clip(1 - mean_wait / H) + 0.3 * availability
            )
        )
    return {k: sum(v) / len(v) for k, v in vals.items()}


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    """Score exactly the statically declared atom set and relative shares."""
    atom_list = list(atoms)
    ids = [getattr(atom, "id", None) for atom in atom_list]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match the declared fixed set")
    shares = list(expected.values())
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0 for value in shares):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(float(value) for value in shares), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    for atom in atom_list:
        share = getattr(atom, "share", None)
        if not _finite(share) or not math.isclose(float(share), float(expected[atom.id]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"atom share differs for {atom.id}")
    return max(0.0, min(1.0, sum(float(atom.share) for atom in atom_list if bool(atom.passed))))


def _score(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    """Backward-compatible wrapper used by the author-side boundary script."""
    try:
        return _score_atoms(atoms, expected)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    return (
        "failed_atoms=[]"
        if all(a.passed for a in atoms)
        else "failed_atoms=[{}]; {}".format(
            ",".join(a.id for a in atoms if not a.passed),
            " | ".join(f"{a.id}: {a.detail}" for a in atoms if not a.passed),
        )
    )


async def _build_state(ws: Workspace) -> EvalState:
    """Run the candidate once and cache the complete deterministic evaluation state."""
    root = str(Path(ws.root).resolve())
    if root in _BUILD_CACHE:
        return _BUILD_CACHE[root]
    lock = _BUILD_LOCKS.setdefault(root, asyncio.Lock())
    async with lock:
        if root in _BUILD_CACHE:
            return _BUILD_CACHE[root]
        result_state = await _build_once(ws)
        _BUILD_CACHE[root] = result_state
        return result_state


async def _build_once(ws: Workspace) -> EvalState:
    state = EvalState()
    path = Path(ws.root, "results.json")
    started = time.monotonic()
    state.run_started = True
    pre_signature: tuple[int, int, int, int] | None = None
    try:
        old_stat = path.lstat()
        pre_signature = (
            int(old_stat.st_dev),
            int(old_stat.st_ino),
            int(old_stat.st_mtime_ns),
            int(old_stat.st_size),
        )
    except (FileNotFoundError, OSError):
        pre_signature = None
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
        state.exit_code = getattr(result, "exit_code", None)
        stderr = str(getattr(result, "stderr", ""))
        state.elapsed_s = time.monotonic() - started
        state.timed_out = (
            state.exit_code is None
            or "timeout" in stderr.lower()
            or "timed out" in stderr.lower()
            or state.elapsed_s >= 600.0 - EPS
        )
        try:
            statv = path.lstat()
            post_signature = (
                int(statv.st_dev),
                int(statv.st_ino),
                int(statv.st_mtime_ns),
                int(statv.st_size),
            )
            state.result_file_is_new_regular = bool(
                remove_ok
                and stat.S_ISREG(statv.st_mode)
                and not path.is_symlink()
                and (pre_signature is None or post_signature != pre_signature)
            )
        except (FileNotFoundError, OSError):
            state.result_file_is_new_regular = False
        state.official_ok = (
            state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit_code={state.exit_code}, elapsed_s={state.elapsed_s:.6f}, "
            f"removed_previous={remove_ok}, timed_out={state.timed_out}, "
            f"fresh_regular_result={state.result_file_is_new_regular}"
        )
        if state.official_ok:
            state.data = _strict_json_bytes(path.read_bytes())
            state.parse_ok = True
    except Exception as exc:
        state.elapsed_s = time.monotonic() - started
        state.timed_out = (
            isinstance(exc, (asyncio.TimeoutError, TimeoutError))
            or "timeout" in str(exc).lower()
            or "timed out" in str(exc).lower()
            or state.elapsed_s >= 600.0 - EPS
        )
        state.official_ok = False
        state.official_message = (
            f"official run raised {type(exc).__name__}: {exc}; "
            f"elapsed_s={state.elapsed_s:.6f}; timed_out={state.timed_out}"
        )
    finally:
        state.run_finished = True
        if state.elapsed_s is None:
            state.elapsed_s = time.monotonic() - started
    state.diagnostics["official"] = state.official_message or "official run did not complete"

    result_atoms = _result_json_atoms(state.data, state.parse_ok)
    try:
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
    except (TypeError, ValueError, OverflowError) as exc:
        state.scores["result_json"] = 0.0
        state.diagnostics["result_json"] = f"atom declaration error: {exc}"
    else:
        state.diagnostics["result_json"] = _diagnose(result_atoms)
    if not state.official_ok or not state.parse_ok:
        reason = state.diagnostics["official"] if not state.official_ok else "strict JSON parse failed"
        for key in ("result_evidence", "canonical_coverage", "decision_legality", "replay_validity"):
            state.scores[key] = 0.0
            state.diagnostics[key] = f"{key}: not evaluated because {reason}"
        state.validity_gate = False
        return state
    state.records = _index_records(state.data)
    atoms = _evidence_atoms(state.records)
    state.scores["result_evidence"] = _score_atoms(atoms, ATOM_SHARES["result_evidence"])
    state.diagnostics["result_evidence"] = _diagnose(atoms)
    state.scores["canonical_coverage"], defects = _coverage(state.data)
    state.diagnostics["canonical_coverage"] = f"defects={defects}"
    atoms = _legality_atoms(state.records)
    state.scores["decision_legality"] = _score_atoms(atoms, ATOM_SHARES["decision_legality"])
    state.diagnostics["decision_legality"] = _diagnose(atoms)
    atoms, state.replayed = _replay_atoms(state.records)
    state.scores["replay_validity"] = _score_atoms(atoms, ATOM_SHARES["replay_validity"])
    state.diagnostics["replay_validity"] = _diagnose(atoms)
    state.validity_gate = all(
        state.scores.get(k, 0) >= 1 - EPS
        for k in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        )
    )
    if state.validity_gate:
        try:
            metrics = _metrics(state.replayed, state.records)
            state.metrics = {
                key: value
                for key, value in metrics.items()
                if key in {"primary_metrics_1", "primary_metrics_2", "primary_metrics_3", "auxiliary_metrics_1", "auxiliary_metrics_2"}
                and _finite(value)
                and 0.0 <= float(value) <= 1.0
            }
            if len(state.metrics) != 5:
                state.metrics = None
                state.validity_gate = False
                state.diagnostics["metrics"] = "metrics rejected: non-finite or out-of-range value"
        except Exception as exc:
            state.metrics = None
            state.validity_gate = False
            state.diagnostics["metrics"] = f"metrics failed: {type(exc).__name__}: {exc}"
    elif not state.validity_gate:
        state.diagnostics["metrics"] = "metrics not evaluated because validity gate failed"
    return state


# Compatibility name for author-side tools; the assignment keeps one cache/run
# implementation and all public checks call ``_build_state`` directly.
_build = _build_state


async def _metric(ws: Workspace, name: str):
    s = await _build_state(ws)
    return (
        (s.metrics[name], f"{name}: replayed={s.metrics[name]:.9g}")
        if s.metrics
        else (0.0, f"{name}: validity gate not satisfied")
    )


@check(id="entry_point")
async def entry_point(ws: Workspace):
    p = Path(ws.root, "run.py")
    r = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "run.py compiled")
        if p.is_file() and getattr(r, "exit_code", 1) == 0
        else (0.0, "run.py missing or compile failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace):
    s = await _build_state(ws)
    return (
        (1.0, "official run passed")
        if s.official_ok
        else (0.0, s.diagnostics.get("official", "official run failed"))
    )


@check(id="report_contract")
async def report_contract(ws: Workspace):
    text = ws.read_text("report.md") or ""
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    missing = []
    for title in required:
        match = re.search(rf"(?m)^#{{1,6}}\s+{re.escape(title)}\s*$", text)
        if not match:
            missing.append(title)
            continue
        body = text[match.end() :]
        body = re.split(r"(?m)^#{1,6}\s+", body, maxsplit=1)[0]
        if not body.strip():
            missing.append(f"{title}:empty")
    return (max(0.0, 1.0 - 0.25 * len(missing)), f"missing_or_empty={missing}")


async def _gate(ws: Workspace, key: str):
    s = await _build_state(ws)
    return s.scores.get(key, 0.0), s.diagnostics.get(key, "")


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
