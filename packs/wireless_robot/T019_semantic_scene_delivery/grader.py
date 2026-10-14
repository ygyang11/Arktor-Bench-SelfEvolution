from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

WIDTH = 12
HEIGHT = 10
HORIZON = 60
ROBOTS = (0, 1, 2, 3)
TARGET_IDS = tuple(f"g{index}" for index in range(8))
SCENES = (
    "open_layout",
    "blocked_rooms",
    "target_burst",
    "status_conflict",
    "long_corridor",
)
MOVES = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "stay": (0, 0),
}
PACKET_BITS = {"structure": 32, "anchor": 16, "object": 24}
KIND_PRIORITY = {"object": 0, "structure": 1, "anchor": 2}
STATUS_ORDER = {
    "unreleased": 0,
    "unassigned": 1,
    "claimed": 2,
    "carried": 3,
    "delivered": 4,
}
LOSS_SETS = {
    "open_layout": frozenset({0}),
    "blocked_rooms": frozenset({0, 1, 2}),
    "target_burst": frozenset({0, 4}),
    "status_conflict": frozenset({0, 1, 7}),
    "long_corridor": frozenset({0, 3}),
}
ROOT_KEYS = {"scenarios"}
SCENE_KEYS = {"scenario", "slots"}
SLOT_KEYS = {"t", "actions"}
ACTION_KEYS = {"robot", "move", "claim_target", "message_kind", "message_subject"}
ACTION_NONKEY_KEYS = ACTION_KEYS - {"robot"}
MESSAGE_CELL_RE = re.compile(r"^c_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)$")
QUEUE_CAPACITY = 2
ENQUEUE_BUDGET = 960
CHANNEL_BITS = 96
EPS = 1e-9

SCENE_DATA: dict[str, dict[str, Any]] = {
    "open_layout": {
        "depot": (1, 1),
        "starts": ((0, 0), (0, 2), (2, 0), (2, 2)),
        "obstacles": ((5, 5, 4, 5),),
        "targets": (
            ((3, 1), 6, 0, 15),
            ((2, 6), 5, 0, 22),
            ((6, 2), 8, 3, 28),
            ((7, 7), 7, 5, 36),
            ((10, 1), 9, 9, 40),
            ((9, 5), 6, 12, 45),
            ((4, 8), 4, 15, 50),
            ((10, 8), 8, 18, 58),
        ),
    },
    "blocked_rooms": {
        "depot": (1, 1),
        "starts": ((0, 0), (0, 2), (2, 0), (2, 2)),
        "obstacles": ((4, 4, 0, 3), (4, 4, 5, 9), (8, 8, 0, 5), (8, 8, 7, 9)),
        "targets": (
            ((3, 7), 5, 0, 20),
            ((6, 2), 7, 0, 28),
            ((6, 7), 6, 4, 32),
            ((9, 6), 8, 6, 38),
            ((10, 2), 10, 8, 42),
            ((10, 8), 9, 10, 48),
            ((7, 4), 4, 14, 46),
            ((11, 5), 7, 18, 58),
        ),
    },
    "target_burst": {
        "depot": (1, 1),
        "starts": ((0, 1), (1, 0), (1, 2), (2, 1)),
        "obstacles": ((3, 5, 4, 4), (7, 8, 4, 4)),
        "targets": (
            ((4, 2), 7, 8, 24),
            ((6, 2), 8, 8, 26),
            ((9, 2), 9, 8, 32),
            ((4, 7), 6, 10, 30),
            ((6, 7), 10, 10, 34),
            ((9, 7), 8, 10, 38),
            ((10, 5), 5, 12, 42),
            ((2, 8), 4, 12, 36),
        ),
    },
    "status_conflict": {
        "depot": (1, 1),
        "starts": ((0, 0), (0, 2), (2, 0), (2, 2)),
        "obstacles": ((5, 6, 0, 3), (5, 6, 5, 9)),
        "targets": (
            ((4, 4), 6, 0, 20),
            ((7, 4), 10, 0, 28),
            ((8, 2), 8, 2, 34),
            ((8, 7), 8, 2, 36),
            ((3, 7), 5, 6, 30),
            ((10, 4), 9, 8, 44),
            ((4, 8), 4, 10, 40),
            ((10, 8), 7, 12, 52),
        ),
    },
    "long_corridor": {
        "depot": (1, 4),
        "starts": ((0, 3), (0, 4), (0, 5), (0, 6)),
        "obstacles": ((3, 10, 0, 3), (3, 10, 6, 9)),
        "targets": (
            ((4, 4), 4, 0, 18),
            ((5, 5), 5, 0, 22),
            ((7, 4), 7, 3, 30),
            ((8, 5), 8, 5, 34),
            ((11, 4), 10, 8, 42),
            ((11, 5), 9, 10, 44),
            ((10, 4), 6, 15, 50),
            ((9, 5), 6, 18, 54),
        ),
    },
}

ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scene_containers": 0.33,
    },
    "result_evidence": {
        "result_evidence.slot_containers": 0.15,
        "result_evidence.action_containers": 0.15,
        "result_evidence.move_fields": 0.20,
        "result_evidence.claim_fields": 0.20,
        "result_evidence.message_fields": 0.30,
    },
    "decision_legality": {
        "decision_legality.grid_boundary": 0.20,
        "decision_legality.obstacle_entry": 0.20,
        "decision_legality.claim_knowledge": 0.20,
        "decision_legality.message_knowledge": 0.20,
        "decision_legality.enqueue_budget": 0.20,
    },
    "replay_validity": {
        "replay_validity.decision_consumption": 1 / 6,
        "replay_validity.observation_and_knowledge": 1 / 6,
        "replay_validity.motion_and_transport": 1 / 6,
        "replay_validity.queue_and_service": 1 / 6,
        "replay_validity.arrival_and_fusion": 1 / 6,
        "replay_validity.terminal_accounting": 1 / 6,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
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
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[str, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _safe_repr(value: Any, limit: int = 180) -> str:
    try:
        rendered = repr(value)
    except (OverflowError, ValueError):
        if _integer(value):
            return f"<integer bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate object member {key!r}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _atom(group: str, name: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{group}.{name}"
    return AtomVerdict(atom_id, passed, detail, ATOM_SHARES[group][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if any(
        not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12)
        for atom in atoms
    ):
        raise ValueError("atom share differs from the declared share")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    score = sum(atom.share for atom in atoms if atom.passed)
    return max(0.0, min(1.0, score))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _rows(data: Any) -> list[Any]:
    if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list):
        return data["scenarios"]
    return []


def _result_json_atoms(data: Any, parsed: bool) -> list[AtomVerdict]:
    root_ok = _shape(data, ROOT_KEYS)
    scenes = data.get("scenarios") if isinstance(data, Mapping) else None
    containers_ok = isinstance(scenes, list) and all(
        isinstance(row, Mapping)
        and set(row) == {"scenario", "slots"}
        and isinstance(row.get("slots"), list)
        for row in scenes
    )
    return [
        _atom("result_json", "parse", parsed, "strict JSON parse"),
        _atom("result_json", "root", root_ok, "expected exact root keys={'scenarios'}"),
        _atom(
            "result_json",
            "scene_containers",
            containers_ok,
            "each submitted scene requires only its canonical key and a slots list",
        ),
    ]


def _canonical_defects(
    actual: Sequence[Hashable], expected: Sequence[Hashable], label: str
) -> list[str]:
    defects: list[str] = []
    for key in expected:
        count = actual.count(key)
        if count == 0:
            defects.append(f"{label}: missing key={_safe_repr(key)}")
        elif count > 1:
            defects.extend(f"{label}: duplicate key={_safe_repr(key)}" for _ in range(count - 1))
    expected_set = set(expected)
    defects.extend(
        f"{label}: unknown key={_safe_repr(key)}" for key in actual if key not in expected_set
    )
    return defects


def _first_scene(data: Any, scene: str) -> Mapping[str, Any] | None:
    for row in _rows(data):
        if isinstance(row, Mapping) and row.get("scenario") == scene:
            return row
    return None


def _first_slot(scene_row: Any, t: int) -> Mapping[str, Any] | None:
    slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if isinstance(slot, Mapping) and _integer(slot.get("t")) and slot.get("t") == t:
            return slot
    return None


def _first_action(slot: Any, robot: int) -> Mapping[str, Any] | None:
    actions = slot.get("actions") if isinstance(slot, Mapping) else None
    if not isinstance(actions, list):
        return None
    for action in actions:
        if (
            isinstance(action, Mapping)
            and _integer(action.get("robot"))
            and action.get("robot") == robot
        ):
            return action
    return None


def _coverage_score(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    rows = _rows(data)
    scene_keys: list[Hashable] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping) and isinstance(row.get("scenario"), str):
            scene_keys.append(row["scenario"])
        else:
            scene_keys.append(("invalid_scene", index))
    defects.extend(_canonical_defects(scene_keys, SCENES, "scenarios"))

    for scene in SCENES:
        scene_row = _first_scene(data, scene)
        slots = scene_row.get("slots") if isinstance(scene_row, Mapping) else None
        if not isinstance(slots, list):
            continue
        slot_keys: list[Hashable] = []
        for index, slot in enumerate(slots):
            if isinstance(slot, Mapping) and _integer(slot.get("t")):
                slot_keys.append(slot["t"])
            else:
                slot_keys.append(("invalid_slot", index))
        defects.extend(_canonical_defects(slot_keys, list(range(HORIZON)), f"{scene}.slots"))
        for t in range(HORIZON):
            slot = _first_slot(scene_row, t)
            actions = slot.get("actions") if isinstance(slot, Mapping) else None
            if not isinstance(actions, list):
                continue
            action_keys: list[Hashable] = []
            for index, action in enumerate(actions):
                if isinstance(action, Mapping) and _integer(action.get("robot")):
                    action_keys.append(action["robot"])
                else:
                    action_keys.append(("invalid_robot", index))
            defects.extend(_canonical_defects(action_keys, ROBOTS, f"{scene}.slots[{t}].actions"))
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _index_records(data: Any) -> dict[Hashable, Any]:
    records: dict[Hashable, Any] = {}
    for scene in SCENES:
        scene_row = _first_scene(data, scene)
        if scene_row is None:
            continue
        records[(scene,)] = scene_row
        for t in range(HORIZON):
            slot = _first_slot(scene_row, t)
            if slot is None:
                continue
            records[(scene, t)] = slot
            for robot in ROBOTS:
                action = _first_action(slot, robot)
                if action is not None:
                    records[(scene, t, robot)] = action
    return records


def _cell_subject(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    match = MESSAGE_CELL_RE.fullmatch(value)
    if match is None:
        return None
    try:
        cell = (int(match.group(1)), int(match.group(2)))
    except ValueError:
        return None
    return cell if _in_bounds(cell) else None


def _claim_field_valid(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value in TARGET_IDS)


def _message_fields_valid(kind: Any, subject: Any) -> bool:
    if kind is None:
        return subject is None
    if not isinstance(kind, str) or kind not in PACKET_BITS or not isinstance(subject, str):
        return False
    if kind == "object":
        return subject in TARGET_IDS
    return _cell_subject(subject) is not None


def _action_local_valid(action: Any) -> bool:
    return (
        isinstance(action, Mapping)
        and set(action) == ACTION_KEYS
        and isinstance(action.get("move"), str)
        and action.get("move") in MOVES
        and _claim_field_valid(action.get("claim_target"))
        and _message_fields_valid(action.get("message_kind"), action.get("message_subject"))
    )


def _evidence_atoms(data: Any) -> list[AtomVerdict]:
    names = (
        "slot_containers",
        "action_containers",
        "move_fields",
        "claim_fields",
        "message_fields",
    )
    flags = {name: True for name in names}
    details = {name: "all retained local records satisfy the declared format" for name in names}
    records = _index_records(data)
    for scene in SCENES:
        for t in range(HORIZON):
            slot = records.get((scene, t))
            if slot is None:
                continue
            slot_ok = (
                isinstance(slot, Mapping)
                and set(slot) - {"t"} == {"actions"}
                and isinstance(slot.get("actions"), list)
            )
            if not slot_ok and flags["slot_containers"]:
                details["slot_containers"] = (
                    f"{scene}.slots[{t}]: expected only t and an actions list"
                )
            flags["slot_containers"] &= slot_ok
            if not isinstance(slot.get("actions"), list):
                continue
            for robot in ROBOTS:
                action = records.get((scene, t, robot))
                if action is None:
                    continue
                path = f"{scene}.slots[{t}].actions[{robot}]"
                container_ok = (
                    isinstance(action, Mapping) and set(action) - {"robot"} <= ACTION_NONKEY_KEYS
                )
                if not container_ok and flags["action_containers"]:
                    details["action_containers"] = f"{path}: unexpected non-key decision fields"
                flags["action_containers"] &= container_ok
                move_ok = (
                    isinstance(action, Mapping)
                    and isinstance(action.get("move"), str)
                    and action.get("move") in MOVES
                )
                if not move_ok and flags["move_fields"]:
                    details["move_fields"] = f"{path}.move: expected one of {tuple(MOVES)}"
                flags["move_fields"] &= move_ok
                claim_ok = (
                    isinstance(action, Mapping)
                    and "claim_target" in action
                    and _claim_field_valid(action.get("claim_target"))
                )
                if not claim_ok and flags["claim_fields"]:
                    details["claim_fields"] = f"{path}.claim_target: expected null or g0,...,g7"
                flags["claim_fields"] &= claim_ok
                message_ok = (
                    isinstance(action, Mapping)
                    and "message_kind" in action
                    and "message_subject" in action
                    and _message_fields_valid(
                        action.get("message_kind"), action.get("message_subject")
                    )
                )
                if not message_ok and flags["message_fields"]:
                    details["message_fields"] = (
                        f"{path}: invalid message kind, subject format, or null relation"
                    )
                flags["message_fields"] &= message_ok
    return [_atom("result_evidence", name, flags[name], details[name]) for name in names]


def _default_action(robot: int) -> dict[str, Any]:
    return {
        "robot": robot,
        "move": "stay",
        "claim_target": None,
        "message_kind": None,
        "message_subject": None,
    }


def _submitted_action(
    records: Mapping[Hashable, Any], scene: str, t: int, robot: int
) -> tuple[dict[str, Any], bool]:
    slot = records.get((scene, t))
    if not (
        isinstance(slot, Mapping)
        and set(slot) == SLOT_KEYS
        and isinstance(slot.get("actions"), list)
    ):
        return _default_action(robot), False
    action = records.get((scene, t, robot))
    if not _action_local_valid(action):
        return _default_action(robot), False
    return dict(action), True


def _in_bounds(cell: Sequence[int]) -> bool:
    return 0 <= cell[0] < WIDTH and 0 <= cell[1] < HEIGHT


def _obstacle_cells(scene: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for xmin, xmax, ymin, ymax in SCENE_DATA[scene]["obstacles"]:
        cells.update((x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1))
    return cells


def _neighbours(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = cell
    return tuple(
        candidate
        for candidate in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        if _in_bounds(candidate)
    )


def _target_fact(
    target: Mapping[str, Any], timestamp: int, sender: int, local_phase: int
) -> dict[str, Any]:
    return {
        "position": tuple(target["position"]),
        "value": target["value"],
        "deadline": target["deadline"],
        "status": target["status"],
        "claimant": target["claimant"],
        "observation_timestamp": timestamp,
        "source": (timestamp, sender, local_phase),
    }


def _target_fact_better(incoming: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    incoming_rank = STATUS_ORDER[incoming["status"]]
    current_rank = STATUS_ORDER[current["status"]]
    return incoming_rank > current_rank or (
        incoming_rank == current_rank and tuple(incoming["source"]) > tuple(current["source"])
    )


def _merge_target_fact(store: dict[str, Any], target_id: str, incoming: dict[str, Any]) -> bool:
    current = store.get(target_id)
    if current is None or _target_fact_better(incoming, current):
        store[target_id] = incoming
        return True
    return False


def _initial_targets(scene: str) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for target_id, (position, value, release, deadline) in zip(
        TARGET_IDS, SCENE_DATA[scene]["targets"], strict=True
    ):
        targets[target_id] = {
            "position": tuple(position),
            "value": value,
            "release": release,
            "deadline": deadline,
            "status": "unreleased",
            "claimant": None,
            "carrier": None,
            "discovery_slot": None,
            "delivery_slot": None,
        }
    return targets


def _initial_robots(scene: str) -> list[dict[str, Any]]:
    return [
        {
            "position": tuple(start),
            "cells": {},
            "objects": {},
            "queue": [],
            "used_bits": 0,
            "assignment": None,
            "carrying": None,
            "visited": [tuple(start)],
            "shared_timestamp": None,
        }
        for start in SCENE_DATA[scene]["starts"]
    ]


def _policy_view(
    t: int, robot: int, state: Mapping[str, Any], depot: tuple[int, int]
) -> dict[str, Any]:
    cells = [
        {
            "cell": list(cell),
            "occupied": fact["occupied"],
            "observation_timestamp": fact["observation_timestamp"],
        }
        for cell, fact in sorted(state["cells"].items())
    ]
    targets = [
        {
            "target": target_id,
            "position": list(fact["position"]),
            "value": fact["value"],
            "deadline": fact["deadline"],
            "status": fact["status"],
            "claimant": fact["claimant"],
            "observation_timestamp": fact["observation_timestamp"],
        }
        for target_id, fact in sorted(state["objects"].items())
    ]
    queue = [
        {
            "generation_slot": packet["generation_slot"],
            "message_id": packet["message_id"],
            "kind": packet["kind"],
            "subject": packet["subject"],
            "bits": packet["bits"],
        }
        for packet in state["queue"]
    ]
    return {
        "t": t,
        "robot": robot,
        "position": list(state["position"]),
        "visited": [list(cell) for cell in state["visited"]],
        "grid": {"width": WIDTH, "height": HEIGHT},
        "depot": list(depot),
        "target_ids": list(TARGET_IDS),
        "cells": cells,
        "targets": targets,
        "assignment": state["assignment"],
        "carrying": state["carrying"],
        "queue": queue,
        "remaining_enqueue_bits": ENQUEUE_BUDGET - state["used_bits"],
    }


def _packet_lost(scene: str, packet: Mapping[str, Any], receiver: int) -> bool:
    scene_index = SCENES.index(scene)
    value = (
        11 * scene_index
        + 5 * packet["generation_slot"]
        + 3 * packet["sender"]
        + 7 * receiver
        + packet["message_id"]
    ) % 19
    return value in LOSS_SETS[scene]


def _packet_expired(scene: str, packet: Mapping[str, Any], boundary: int) -> bool:
    if packet["kind"] == "structure":
        return boundary > packet["generation_slot"] + 5
    if packet["kind"] == "anchor":
        return boundary > packet["generation_slot"] + 8
    target_index = TARGET_IDS.index(packet["subject"])
    deadline = SCENE_DATA[scene]["targets"][target_index][3]
    return boundary > deadline


def _packet_timestamps(packet: Mapping[str, Any]) -> list[int]:
    if packet["kind"] == "object":
        return [packet["object"]["observation_timestamp"]]
    return [fact["observation_timestamp"] for fact in packet["cells"]]


def _process_arrivals(
    scene: str,
    boundary: int,
    scheduled: Mapping[int, list[dict[str, Any]]],
    robots: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    events = sorted(
        scheduled.get(boundary, []),
        key=lambda event: (
            event["packet"]["sender"],
            event["packet"]["message_id"],
            event["receiver"],
        ),
    )
    for event in events:
        packet = event["packet"]
        receiver = event["receiver"]
        outcome = "valid"
        if _packet_lost(scene, packet, receiver):
            outcome = "lost"
        elif _packet_expired(scene, packet, boundary):
            outcome = "expired"
        if outcome != "valid":
            stats["arrival_log"].append(
                {
                    "boundary": boundary,
                    "receiver": receiver,
                    "sender": packet["sender"],
                    "message_id": packet["message_id"],
                    "outcome": outcome,
                }
            )
            continue

        receiver_state = robots[receiver]
        timestamps = _packet_timestamps(packet)
        latest = max(timestamps)
        previous_shared = receiver_state["shared_timestamp"]
        receiver_state["shared_timestamp"] = (
            latest if previous_shared is None else max(previous_shared, latest)
        )
        improved = False
        if packet["kind"] in {"structure", "anchor"}:
            for payload_fact in packet["cells"]:
                cell = tuple(payload_fact["cell"])
                current = receiver_state["cells"].get(cell)
                if current is None:
                    improved = True
                if (
                    current is None
                    or payload_fact["observation_timestamp"] > current["observation_timestamp"]
                ):
                    receiver_state["cells"][cell] = {
                        "occupied": payload_fact["occupied"],
                        "observation_timestamp": payload_fact["observation_timestamp"],
                    }
        else:
            target_id = packet["subject"]
            payload_fact = packet["object"]
            current = receiver_state["objects"].get(target_id)
            if current is None:
                improved = True
            elif STATUS_ORDER[payload_fact["status"]] > STATUS_ORDER[current["status"]]:
                improved = True
            elif (
                STATUS_ORDER[payload_fact["status"]] == STATUS_ORDER[current["status"]]
                and payload_fact["observation_timestamp"] > current["observation_timestamp"]
            ):
                improved = True
            incoming = {
                "position": tuple(payload_fact["position"]),
                "value": payload_fact["value"],
                "deadline": payload_fact["deadline"],
                "status": payload_fact["status"],
                "claimant": payload_fact["claimant"],
                "observation_timestamp": payload_fact["observation_timestamp"],
                "source": (
                    payload_fact["observation_timestamp"],
                    packet["sender"],
                    packet["message_id"],
                ),
            }
            _merge_target_fact(receiver_state["objects"], target_id, incoming)
        if improved:
            packet["useful"] = True
        stats["arrival_log"].append(
            {
                "boundary": boundary,
                "receiver": receiver,
                "sender": packet["sender"],
                "message_id": packet["message_id"],
                "outcome": "valid",
                "improved": improved,
            }
        )


def _make_packet(
    t: int,
    sender: int,
    kind: str,
    subject: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "generation_slot": t,
        "message_id": 4 * t + sender,
        "sender": sender,
        "kind": kind,
        "subject": subject,
        "bits": PACKET_BITS[kind],
        "useful": False,
    }
    if kind in {"structure", "anchor"}:
        subject_cell = _cell_subject(subject)
        if subject_cell is None:
            raise ValueError("cannot construct packet for malformed cell subject")
        selected = (subject_cell,) if kind == "anchor" else _neighbours(subject_cell)
        packet["cells"] = [
            {
                "cell": cell,
                "occupied": state["cells"][cell]["occupied"],
                "observation_timestamp": state["cells"][cell]["observation_timestamp"],
            }
            for cell in sorted(selected)
            if cell in state["cells"]
        ]
    else:
        fact = state["objects"][subject]
        packet["object"] = {
            "position": tuple(fact["position"]),
            "value": fact["value"],
            "deadline": fact["deadline"],
            "status": fact["status"],
            "claimant": fact["claimant"],
            "observation_timestamp": fact["observation_timestamp"],
        }
    return packet


def _action_from_policy(
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    view: Mapping[str, Any],
    robot: int,
    communication_enabled: bool,
) -> dict[str, Any]:
    chosen = policy(view)
    if not isinstance(chosen, Mapping):
        chosen = {}
    kind = chosen.get("message_kind") if communication_enabled else None
    subject = chosen.get("message_subject") if communication_enabled else None
    return {
        "robot": robot,
        "move": chosen.get("move", "stay"),
        "claim_target": chosen.get("claim_target"),
        "message_kind": kind,
        "message_subject": subject,
    }


def _simulate_scene(
    scene: str,
    records: Mapping[Hashable, Any] | None = None,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    communication_enabled: bool = True,
) -> dict[str, Any]:
    if (records is None) == (policy is None):
        raise ValueError("provide exactly one of records or policy")
    data = SCENE_DATA[scene]
    depot = tuple(data["depot"])
    obstacles = _obstacle_cells(scene)
    robots = _initial_robots(scene)
    targets = _initial_targets(scene)
    scheduled: dict[int, list[dict[str, Any]]] = {}
    generated_slots: list[dict[str, Any]] = []
    positions_history: list[tuple[tuple[int, int], ...]] = [
        tuple(tuple(robot["position"]) for robot in robots)
    ]
    status_history: list[tuple[str, ...]] = [
        tuple(targets[target_id]["status"] for target_id in TARGET_IDS)
    ]
    seen_positions = set(positions_history[0])
    trusted_prefix = True
    stats: dict[str, Any] = {
        "arrival_log": [],
        "serviced_packets": [],
        "slot_logs": [],
        "legality_issues": {
            "grid_boundary": [],
            "obstacle_entry": [],
            "claim_knowledge": [],
            "message_knowledge": [],
            "enqueue_budget": [],
        },
        "freshness_sum": 0.0,
        "freshness_samples": 0,
        "redundant_visits": 0,
        "consumed_decisions": 0,
        "decision_views": [],
    }

    for t in range(HORIZON):
        _process_arrivals(scene, t, scheduled, robots, stats)
        if t >= 1:
            for robot_state in robots:
                shared = robot_state["shared_timestamp"]
                age = 4 if shared is None else t - shared
                stats["freshness_sum"] += max(0.0, min(1.0, 1.0 - age / 4.0))
                stats["freshness_samples"] += 1

        for target in targets.values():
            if target["release"] == t:
                target["status"] = "unassigned"

        for robot_id, robot_state in enumerate(robots):
            x, y = robot_state["position"]
            for cell in (
                (cx, cy)
                for cx in range(max(0, x - 1), min(WIDTH, x + 2))
                for cy in range(max(0, y - 1), min(HEIGHT, y + 2))
                if abs(cx - x) + abs(cy - y) <= 1
            ):
                robot_state["cells"][cell] = {
                    "occupied": cell in obstacles,
                    "observation_timestamp": t,
                }
            for target_id, target in targets.items():
                tx, ty = target["position"]
                if target["status"] != "unreleased" and abs(tx - x) + abs(ty - y) <= 1:
                    if target["discovery_slot"] is None:
                        target["discovery_slot"] = t
                    _merge_target_fact(
                        robot_state["objects"],
                        target_id,
                        _target_fact(target, t, robot_id, -1),
                    )

        views = [_policy_view(t, robot, robots[robot], depot) for robot in ROBOTS]
        stats["decision_views"].append(views)
        snapshot_cells = [dict(robot["cells"]) for robot in robots]
        snapshot_objects = [dict(robot["objects"]) for robot in robots]
        actions: list[dict[str, Any]] = []
        locally_valid: list[bool] = []
        for robot in ROBOTS:
            if policy is not None:
                action = _action_from_policy(policy, views[robot], robot, communication_enabled)
                valid = _action_local_valid(action)
            else:
                action, valid = _submitted_action(records or {}, scene, t, robot)
            actions.append(action)
            locally_valid.append(valid)
        generated_slots.append({"t": t, "actions": actions})
        stats["consumed_decisions"] += len(actions)
        score_dynamic = trusted_prefix and all(locally_valid)

        known_claims: list[str | None] = []
        for robot, action in enumerate(actions):
            target_id = action["claim_target"]
            known = target_id is None or target_id in snapshot_objects[robot]
            if target_id is not None and not known and score_dynamic:
                stats["legality_issues"]["claim_knowledge"].append(
                    f"t={t},robot={robot},target={target_id}"
                )
            known_claims.append(target_id if known else None)

        owned_before = [robot["assignment"] for robot in robots]
        for robot in ROBOTS:
            target_id = known_claims[robot]
            if target_id is None:
                continue
            target = targets[target_id]
            if target["status"] == "unassigned" and robots[robot]["assignment"] is None:
                target["status"] = "claimed"
                target["claimant"] = robot
                robots[robot]["assignment"] = target_id

        starts = [tuple(robot["position"]) for robot in robots]
        endpoints: list[tuple[int, int]] = []
        move_events: list[dict[str, Any]] = []
        for robot, action in enumerate(actions):
            dx, dy = MOVES[action["move"]]
            requested = (starts[robot][0] + dx, starts[robot][1] + dy)
            executed = requested
            reason = "accepted"
            if not _in_bounds(requested):
                if score_dynamic:
                    stats["legality_issues"]["grid_boundary"].append(
                        f"t={t},robot={robot},endpoint={requested}"
                    )
                executed = starts[robot]
                reason = "grid_boundary"
            elif requested in obstacles:
                if score_dynamic:
                    stats["legality_issues"]["obstacle_entry"].append(
                        f"t={t},robot={robot},endpoint={requested}"
                    )
                executed = starts[robot]
                reason = "obstacle_entry"

            reserved = {cell for cell in endpoints if cell != depot}
            reserved.update(
                starts[other] for other in range(robot + 1, len(ROBOTS)) if starts[other] != depot
            )
            swaps = any(
                executed == starts[other] and endpoints[other] == starts[robot]
                for other in range(robot)
                if executed != depot and starts[robot] != depot
            )
            if executed != depot and (executed in reserved or swaps):
                executed = starts[robot]
                reason = "robot_conflict"
            endpoints.append(executed)
            move_events.append(
                {
                    "robot": robot,
                    "submitted": action["move"],
                    "requested": requested,
                    "executed": executed,
                    "reason": reason,
                }
            )

        for robot, endpoint in enumerate(endpoints):
            robots[robot]["position"] = endpoint
            robots[robot]["visited"].append(endpoint)
            carried = robots[robot]["carrying"]
            if carried is not None:
                targets[carried]["position"] = endpoint

        for robot in ROBOTS:
            assigned = robots[robot]["assignment"]
            if assigned is None or robots[robot]["carrying"] is not None:
                continue
            target = targets[assigned]
            if (
                target["status"] == "claimed"
                and target["claimant"] == robot
                and tuple(target["position"]) == tuple(robots[robot]["position"])
            ):
                target["status"] = "carried"
                target["carrier"] = robot
                target["position"] = tuple(robots[robot]["position"])
                robots[robot]["carrying"] = assigned

        delivered_by: dict[int, str] = {}
        for robot in ROBOTS:
            carried = robots[robot]["carrying"]
            if carried is not None and tuple(robots[robot]["position"]) == depot:
                target = targets[carried]
                target["status"] = "delivered"
                target["position"] = depot
                target["carrier"] = None
                target["delivery_slot"] = t
                delivered_by[robot] = carried
                robots[robot]["assignment"] = None
                robots[robot]["carrying"] = None

        for robot in ROBOTS:
            feedback_targets = {
                target_id
                for target_id in (
                    owned_before[robot],
                    robots[robot]["assignment"],
                    delivered_by.get(robot),
                )
                if target_id is not None
            }
            for target_id in feedback_targets:
                _merge_target_fact(
                    robots[robot]["objects"],
                    target_id,
                    _target_fact(targets[target_id], t, robot, 0),
                )

        pre_queue_sizes = tuple(len(robot["queue"]) for robot in robots)
        pre_used_bits = tuple(robot["used_bits"] for robot in robots)
        enqueue_events: list[dict[str, Any]] = []
        for robot, action in enumerate(actions):
            kind = action["message_kind"]
            subject = action["message_subject"]
            if kind is None:
                continue
            if kind == "object":
                message_known = subject in snapshot_objects[robot]
            else:
                cell = _cell_subject(subject)
                message_known = cell in snapshot_cells[robot]
                if kind == "anchor" and message_known:
                    message_known = not snapshot_cells[robot][cell]["occupied"]
            if not message_known:
                if score_dynamic:
                    stats["legality_issues"]["message_knowledge"].append(
                        f"t={t},robot={robot},kind={kind},subject={subject}"
                    )
                enqueue_events.append({"robot": robot, "outcome": "unknown_subject"})
                continue
            bits = PACKET_BITS[kind]
            if len(robots[robot]["queue"]) >= QUEUE_CAPACITY:
                enqueue_events.append({"robot": robot, "outcome": "queue_full"})
                continue
            if robots[robot]["used_bits"] + bits > ENQUEUE_BUDGET:
                if score_dynamic:
                    stats["legality_issues"]["enqueue_budget"].append(
                        f"t={t},robot={robot},used={robots[robot]['used_bits']},request={bits}"
                    )
                enqueue_events.append({"robot": robot, "outcome": "over_budget"})
                continue
            packet = _make_packet(t, robot, kind, subject, robots[robot])
            robots[robot]["queue"].append(packet)
            robots[robot]["used_bits"] += bits
            enqueue_events.append(
                {"robot": robot, "outcome": "enqueued", "message_id": packet["message_id"]}
            )

        queued_packets = sorted(
            (packet for robot in robots for packet in robot["queue"]),
            key=lambda packet: (
                packet["generation_slot"],
                KIND_PRIORITY[packet["kind"]],
                packet["sender"],
                packet["message_id"],
            ),
        )
        remaining = CHANNEL_BITS
        serviced_this_slot: list[dict[str, Any]] = []
        for packet in queued_packets:
            if packet["bits"] > remaining:
                continue
            remaining -= packet["bits"]
            sender = packet["sender"]
            robots[sender]["queue"] = [
                queued for queued in robots[sender]["queue"] if queued is not packet
            ]
            packet["service_slot"] = t
            serviced_this_slot.append(packet)
            stats["serviced_packets"].append(packet)
            for receiver in ROBOTS:
                if receiver != sender:
                    scheduled.setdefault(t + 1, []).append({"receiver": receiver, "packet": packet})

        for endpoint in endpoints:
            if endpoint != depot and endpoint in seen_positions:
                stats["redundant_visits"] += 1
        seen_positions.update(endpoints)
        positions_history.append(tuple(endpoints))
        status_history.append(tuple(targets[target_id]["status"] for target_id in TARGET_IDS))
        stats["slot_logs"].append(
            {
                "t": t,
                "locally_valid": tuple(locally_valid),
                "dynamic_scored": score_dynamic,
                "moves": move_events,
                "enqueue": enqueue_events,
                "service_bits": CHANNEL_BITS - remaining,
                "serviced_ids": tuple(packet["message_id"] for packet in serviced_this_slot),
                "queue_sizes": tuple(len(robot["queue"]) for robot in robots),
                "used_bits": tuple(robot["used_bits"] for robot in robots),
                "pre_queue_sizes": pre_queue_sizes,
                "pre_used_bits": pre_used_bits,
            }
        )
        trusted_prefix = trusted_prefix and all(locally_valid)

    _process_arrivals(scene, HORIZON, scheduled, robots, stats)
    for robot_state in robots:
        shared = robot_state["shared_timestamp"]
        age = 4 if shared is None else HORIZON - shared
        stats["freshness_sum"] += max(0.0, min(1.0, 1.0 - age / 4.0))
        stats["freshness_samples"] += 1

    return {
        "scene": scene,
        "slots": generated_slots,
        "positions": positions_history,
        "status_history": status_history,
        "robots": robots,
        "targets": targets,
        **stats,
    }


def _replay_scene(scene: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    return _simulate_scene(scene, records=records)


def _online_scene(
    scene: str,
    policy: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    communication_enabled: bool = True,
) -> dict[str, Any]:
    return _simulate_scene(
        scene,
        policy=policy,
        communication_enabled=communication_enabled,
    )


def _result_from_online(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": [{"scenario": scene, "slots": replayed[scene]["slots"]} for scene in SCENES]
    }


def _legality_atoms(replayed: Mapping[str, Mapping[str, Any]]) -> list[AtomVerdict]:
    names = (
        "grid_boundary",
        "obstacle_entry",
        "claim_knowledge",
        "message_knowledge",
        "enqueue_budget",
    )
    atoms: list[AtomVerdict] = []
    for name in names:
        issues = [
            f"{scene}:{issue}"
            for scene in SCENES
            for issue in replayed[scene]["legality_issues"][name]
        ]
        detail = "no violations" if not issues else f"violations={issues[:10]}"
        atoms.append(_atom("decision_legality", name, not issues, detail))
    return atoms


def _decision_consumption_issue(replay: Mapping[str, Any]) -> str | None:
    if replay.get("consumed_decisions") != HORIZON * len(ROBOTS):
        return (
            f"expected consumed_decisions={HORIZON * len(ROBOTS)}, "
            f"got {replay.get('consumed_decisions')}"
        )
    slots = replay.get("slots")
    if not isinstance(slots, list) or len(slots) != HORIZON:
        return f"expected {HORIZON} replay slots"
    for t, slot in enumerate(slots):
        if not isinstance(slot, Mapping) or slot.get("t") != t:
            return f"slot index mismatch at t={t}"
        actions = slot.get("actions")
        if not isinstance(actions, list) or len(actions) != len(ROBOTS):
            return f"expected four consumed actions at t={t}"
    return None


def _observation_knowledge_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    obstacles = _obstacle_cells(scene)
    robots = replay.get("robots")
    if not isinstance(robots, list) or len(robots) != len(ROBOTS):
        return "terminal robot state count mismatch"
    for robot, state in enumerate(robots):
        cells = state.get("cells") if isinstance(state, Mapping) else None
        objects = state.get("objects") if isinstance(state, Mapping) else None
        if not isinstance(cells, Mapping) or not isinstance(objects, Mapping):
            return f"robot={robot} knowledge containers missing"
        for cell, fact in cells.items():
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or not _in_bounds(cell)
                or not isinstance(fact, Mapping)
                or fact.get("occupied") is not (cell in obstacles)
                or not _integer(fact.get("observation_timestamp"))
                or not 0 <= fact["observation_timestamp"] < HORIZON
            ):
                return f"robot={robot} invalid cell fact at {_safe_repr(cell)}"
        for target_id, fact in objects.items():
            if target_id not in TARGET_IDS or not isinstance(fact, Mapping):
                return f"robot={robot} invalid target fact key={_safe_repr(target_id)}"
            index = TARGET_IDS.index(target_id)
            _, value, _, deadline = SCENE_DATA[scene]["targets"][index]
            source = fact.get("source")
            position = fact.get("position")
            if (
                fact.get("value") != value
                or fact.get("deadline") != deadline
                or fact.get("status") not in STATUS_ORDER
                or not isinstance(position, tuple)
                or len(position) != 2
                or not _in_bounds(position)
                or not _integer(fact.get("observation_timestamp"))
                or not 0 <= fact["observation_timestamp"] < HORIZON
                or not isinstance(source, tuple)
                or len(source) != 3
                or source[0] != fact["observation_timestamp"]
            ):
                return f"robot={robot} malformed target fact target={target_id}"
    return None


def _motion_transport_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    obstacles = _obstacle_cells(scene)
    positions = replay.get("positions")
    if not isinstance(positions, list) or len(positions) != HORIZON + 1:
        return f"expected {HORIZON + 1} position boundaries"
    for boundary, joint in enumerate(positions):
        if not isinstance(joint, tuple) or len(joint) != len(ROBOTS):
            return f"joint position shape mismatch at boundary={boundary}"
        non_depot = [cell for cell in joint if cell != tuple(SCENE_DATA[scene]["depot"])]
        if len(set(non_depot)) != len(non_depot):
            return f"robot endpoints overlap outside depot at boundary={boundary}"
        for robot, cell in enumerate(joint):
            if not _in_bounds(cell) or cell in obstacles:
                return f"invalid endpoint boundary={boundary},robot={robot},cell={cell}"
    history = replay.get("status_history")
    if not isinstance(history, list) or len(history) != HORIZON + 1:
        return "target status history length mismatch"
    previous = tuple(STATUS_ORDER["unreleased"] for _ in TARGET_IDS)
    for boundary, statuses in enumerate(history):
        if not isinstance(statuses, tuple) or len(statuses) != len(TARGET_IDS):
            return f"target status shape mismatch at boundary={boundary}"
        ranks = tuple(STATUS_ORDER.get(status, -1) for status in statuses)
        if any(rank < old for rank, old in zip(ranks, previous, strict=True)):
            return f"target status regressed at boundary={boundary}"
        previous = ranks
    targets = replay.get("targets")
    robots = replay.get("robots")
    if not isinstance(targets, Mapping) or not isinstance(robots, list):
        return "terminal transport state missing"
    for robot, state in enumerate(robots):
        assignment = state.get("assignment")
        carrying = state.get("carrying")
        if carrying is not None and carrying != assignment:
            return f"robot={robot} carrying target differs from assignment"
        if assignment is not None:
            target = targets.get(assignment)
            if not isinstance(target, Mapping) or target.get("claimant") != robot:
                return f"robot={robot} assignment claimant mismatch"
            if target.get("status") not in {"claimed", "carried"}:
                return f"robot={robot} assignment has status={target.get('status')}"
    return None


def _queue_service_issue(replay: Mapping[str, Any]) -> str | None:
    logs = replay.get("slot_logs")
    if not isinstance(logs, list) or len(logs) != HORIZON:
        return "slot audit log length mismatch"
    for t, log in enumerate(logs):
        if not isinstance(log, Mapping) or log.get("t") != t:
            return f"slot audit index mismatch at t={t}"
        service_bits = log.get("service_bits")
        queue_sizes = log.get("queue_sizes")
        used_bits = log.get("used_bits")
        if not _integer(service_bits) or not 0 <= service_bits <= CHANNEL_BITS:
            return f"service capacity violated at t={t},bits={service_bits}"
        if (
            not isinstance(queue_sizes, tuple)
            or len(queue_sizes) != len(ROBOTS)
            or any(not _integer(value) or not 0 <= value <= QUEUE_CAPACITY for value in queue_sizes)
        ):
            return f"queue capacity audit failed at t={t}"
        if (
            not isinstance(used_bits, tuple)
            or len(used_bits) != len(ROBOTS)
            or any(not _integer(value) or not 0 <= value <= ENQUEUE_BUDGET for value in used_bits)
        ):
            return f"enqueue budget audit failed at t={t}"
    for packet in replay.get("serviced_packets", []):
        if (
            not isinstance(packet, Mapping)
            or packet.get("message_id")
            != 4 * packet.get("generation_slot", -HORIZON) + packet.get("sender", -1)
            or packet.get("kind") not in PACKET_BITS
            or packet.get("bits") != PACKET_BITS.get(packet.get("kind"))
            or not _integer(packet.get("service_slot"))
            or packet["service_slot"] < packet["generation_slot"]
            or packet["service_slot"] >= HORIZON
        ):
            return f"malformed serviced packet={_safe_repr(packet)}"
    robots = replay.get("robots", [])
    if any(len(state.get("queue", [])) > QUEUE_CAPACITY for state in robots):
        return "terminal queue capacity exceeded"
    return None


def _arrival_fusion_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    packets = {
        packet["message_id"]: packet
        for packet in replay.get("serviced_packets", [])
        if isinstance(packet, Mapping) and "message_id" in packet
    }
    arrival_log = replay.get("arrival_log", [])
    if not isinstance(arrival_log, list) or len(arrival_log) != 3 * len(packets):
        actual = len(arrival_log) if isinstance(arrival_log, list) else "non-list"
        return f"expected {3 * len(packets)} packet-receiver arrivals, got {actual}"
    seen_events: set[tuple[int, int]] = set()
    for event in arrival_log:
        if not isinstance(event, Mapping):
            return "arrival log contains a non-object"
        packet = packets.get(event.get("message_id"))
        receiver = event.get("receiver")
        boundary = event.get("boundary")
        if packet is None or receiver not in ROBOTS or not _integer(boundary):
            return f"arrival cannot be associated: {_safe_repr(event)}"
        event_key = (packet["message_id"], receiver)
        if receiver == packet["sender"] or event_key in seen_events:
            return (
                f"duplicate or self arrival message_id={packet['message_id']},receiver={receiver}"
            )
        seen_events.add(event_key)
        if boundary != packet["service_slot"] + 1:
            return f"arrival boundary mismatch message_id={packet['message_id']}"
        expected = (
            "lost"
            if _packet_lost(scene, packet, receiver)
            else "expired"
            if _packet_expired(scene, packet, boundary)
            else "valid"
        )
        if event.get("outcome") != expected:
            return (
                f"arrival outcome mismatch message_id={packet['message_id']},"
                f"receiver={receiver},expected={expected},actual={event.get('outcome')}"
            )
        if expected == "valid" and not isinstance(event.get("improved"), bool):
            return f"valid arrival lacks improvement flag message_id={packet['message_id']}"
    for robot, state in enumerate(replay.get("robots", [])):
        shared = state.get("shared_timestamp")
        if shared is not None and (not _integer(shared) or not 0 <= shared < HORIZON):
            return f"robot={robot} invalid shared_timestamp={shared}"
    return None


def _terminal_accounting_issue(scene: str, replay: Mapping[str, Any]) -> str | None:
    if replay.get("freshness_samples") != len(ROBOTS) * HORIZON:
        return f"expected {len(ROBOTS) * HORIZON} freshness samples"
    freshness_sum = replay.get("freshness_sum")
    redundant = replay.get("redundant_visits")
    if (
        not isinstance(freshness_sum, (int, float))
        or isinstance(freshness_sum, bool)
        or not math.isfinite(float(freshness_sum))
        or not 0 <= freshness_sum <= len(ROBOTS) * HORIZON
    ):
        return f"invalid freshness sum={freshness_sum}"
    if not _integer(redundant) or not 0 <= redundant <= len(ROBOTS) * HORIZON:
        return f"invalid redundant visit count={redundant}"
    targets = replay.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(TARGET_IDS):
        return "terminal target set mismatch"
    for target_id, target in targets.items():
        discovery = target.get("discovery_slot")
        delivery = target.get("delivery_slot")
        if discovery is not None and (
            not _integer(discovery) or not target["release"] <= discovery < HORIZON
        ):
            return f"target={target_id} invalid discovery_slot={discovery}"
        if delivery is not None and (
            not _integer(delivery)
            or not 0 <= delivery < HORIZON
            or discovery is None
            or delivery < discovery
            or target.get("status") != "delivered"
        ):
            return f"target={target_id} invalid delivery_slot={delivery}"
        if target.get("status") == "delivered" and delivery is None:
            return f"target={target_id} delivered without a delivery slot"
    if any(
        not isinstance(packet.get("useful"), bool) for packet in replay.get("serviced_packets", [])
    ):
        return "serviced packet lacks Boolean useful accounting"
    return None


def _replay_atoms(
    records: Mapping[Hashable, Any], replayed: Mapping[str, Mapping[str, Any]]
) -> list[AtomVerdict]:
    del records
    issue_maps: dict[str, list[str]] = {
        "decision_consumption": [],
        "observation_and_knowledge": [],
        "motion_and_transport": [],
        "queue_and_service": [],
        "arrival_and_fusion": [],
        "terminal_accounting": [],
    }
    for scene in SCENES:
        replay = replayed.get(scene)
        if not isinstance(replay, Mapping):
            for issues in issue_maps.values():
                issues.append(f"{scene}: replay missing")
            continue
        checks = {
            "decision_consumption": _decision_consumption_issue(replay),
            "observation_and_knowledge": _observation_knowledge_issue(scene, replay),
            "motion_and_transport": _motion_transport_issue(scene, replay),
            "queue_and_service": _queue_service_issue(replay),
            "arrival_and_fusion": _arrival_fusion_issue(scene, replay),
            "terminal_accounting": _terminal_accounting_issue(scene, replay),
        }
        for name, issue in checks.items():
            if issue is not None:
                issue_maps[name].append(f"{scene}: {issue}")
    return [
        _atom(
            "replay_validity",
            name,
            not issues,
            "replay invariant passed" if not issues else f"issues={issues[:8]}",
        )
        for name, issues in issue_maps.items()
    ]


def _scene_metrics(scene: str, replay: Mapping[str, Any]) -> dict[str, float]:
    targets = replay["targets"]
    total_value = sum(targets[target_id]["value"] for target_id in TARGET_IDS)
    delivered_value = sum(
        target["value"]
        for target in targets.values()
        if target["delivery_slot"] is not None and target["delivery_slot"] <= target["deadline"]
    )
    timely_value = 0.0
    for target in targets.values():
        discovery = target["discovery_slot"]
        delivery = target["delivery_slot"]
        if (
            discovery is None
            or discovery >= target["deadline"]
            or delivery is None
            or delivery > target["deadline"]
        ):
            utility = 0.0
        else:
            utility = 1.0 - (delivery - discovery) / (target["deadline"] - discovery)
            utility = max(0.0, min(1.0, utility))
        timely_value += target["value"] * utility

    serviced = replay["serviced_packets"]
    total_sent_bits = sum(packet["bits"] for packet in serviced)
    useful_bits = sum(packet["bits"] for packet in serviced if packet["useful"])
    communication_efficiency = useful_bits / total_sent_bits if total_sent_bits > 0 else 0.0
    freshness = replay["freshness_sum"] / (len(ROBOTS) * HORIZON)
    nonredundancy = 1.0 - replay["redundant_visits"] / (len(ROBOTS) * HORIZON)
    return {
        "primary_metrics_1": delivered_value / total_value,
        "primary_metrics_2": timely_value / total_value,
        "auxiliary_metrics_1": max(0.0, min(1.0, communication_efficiency)),
        "auxiliary_metrics_2": 0.6 * freshness + 0.4 * max(0.0, min(1.0, nonredundancy)),
    }


def _metrics(replayed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    names = (
        "primary_metrics_1",
        "primary_metrics_2",
        "auxiliary_metrics_1",
        "auxiliary_metrics_2",
    )
    values = {
        name: sum(_scene_metrics(scene, replayed[scene])[name] for scene in SCENES) / len(SCENES)
        for name in names
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"metric out of range: {values}")
    return values


def _task_performance(metrics: Mapping[str, float]) -> float:
    return (
        8.0 * metrics["primary_metrics_1"]
        + 7.0 * metrics["primary_metrics_2"]
        + 3.0 * metrics["auxiliary_metrics_1"]
        + 2.0 * metrics["auxiliary_metrics_2"]
    ) / 20.0


def _regular(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _headings(text: str) -> dict[str, str]:
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
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    try:
        if not _regular(ws, "report.md"):
            return 0.0, "report.md missing or not a regular file"
        report = ws.read_text("report.md") or ""
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report.md read failed: {type(exc).__name__}: {exc}"
    headings = _headings(report)
    missing = sorted(title for title in required if not headings.get(title))
    return max(0.0, 2.0 - 0.5 * len(missing)) / 2.0, f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}
_INFLIGHT: dict[str, asyncio.Task[EvalState]] = {}


def _fail_run_checks(state: EvalState, detail: str) -> EvalState:
    state.official_ok = False
    state.official_message = detail
    for name in (
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    ):
        state.scores[name] = 0.0
        state.diagnostics[name] = f"official run failed; {detail}"
    return state


async def _build_state_uncached(ws: Workspace) -> EvalState:
    state = EvalState()
    try:
        try:
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
        except TimeoutError:
            state.timed_out = True
            return _fail_run_checks(state, "pre-run results cleanup timed out after 30s")
        except Exception as exc:  # noqa: BLE001
            return _fail_run_checks(
                state, f"pre-run results cleanup raised {type(exc).__name__}: {exc}"
            )
        state.run_started = True
        started = time.monotonic()
        try:
            run = await ws.run("python run.py", check=False, timeout=600.0)
        except TimeoutError:
            state.elapsed_s = time.monotonic() - started
            state.timed_out = True
            return _fail_run_checks(
                state, f"official run timed out after 600s; elapsed={state.elapsed_s:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            state.elapsed_s = time.monotonic() - started
            return _fail_run_checks(
                state,
                f"official run raised {type(exc).__name__}: {exc}; elapsed={state.elapsed_s:.3f}s",
            )
        state.run_finished = True
        state.exit_code = getattr(run, "exit_code", None)
        state.elapsed_s = time.monotonic() - started
        state.timed_out = state.exit_code is None and state.elapsed_s >= 600.0
        state.result_file_is_new_regular = getattr(removed, "exit_code", None) == 0 and _regular(
            ws, "results.json"
        )
        state.official_ok = (
            state.run_started and state.exit_code == 0 and state.result_file_is_new_regular
        )
        state.official_message = (
            f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
            f"timed_out={state.timed_out}, new_results={state.result_file_is_new_regular}"
        )
        if not state.official_ok:
            return _fail_run_checks(state, state.official_message)
        try:
            state.data = _strict_json(ws.read_text("results.json") or "")
            state.parse_ok = True
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            state.diagnostics["result_json"] = f"strict JSON parse failed: {exc}"

        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if state.parse_ok:
            state.records = _index_records(state.data)
            evidence_atoms = _evidence_atoms(state.data)
            state.scores["result_evidence"] = _score_atoms(
                evidence_atoms, ATOM_SHARES["result_evidence"]
            )
            state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)
            coverage, defects = _coverage_score(state.data)
            state.scores["canonical_coverage"] = coverage
            state.diagnostics["canonical_coverage"] = (
                f"defect_count={len(defects)}; defects={defects[:12]}"
            )
            state.replayed = {scene: _replay_scene(scene, state.records) for scene in SCENES}
            legality_atoms = _legality_atoms(state.replayed)
            state.scores["decision_legality"] = _score_atoms(
                legality_atoms, ATOM_SHARES["decision_legality"]
            )
            state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)
            replay_atoms = _replay_atoms(state.records, state.replayed)
            state.scores["replay_validity"] = _score_atoms(
                replay_atoms, ATOM_SHARES["replay_validity"]
            )
            state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)
            required = (
                "result_json",
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            )
            if all(state.scores.get(name, 0.0) >= 1.0 - EPS for name in required):
                state.metrics = _metrics(state.replayed)
                state.validity_gate = True
        else:
            for name in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[name] = 0.0
                state.diagnostics[name] = "strict JSON parse failed"
    except Exception as exc:  # noqa: BLE001
        detail = f"grader failure: {type(exc).__name__}: {exc}"
        state.official_message = (
            f"{state.official_message}; {detail}" if state.official_message else detail
        )
        for name in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores[name] = 0.0
            state.diagnostics[name] = detail
    return state


async def _build_state(ws: Workspace) -> EvalState:
    key = str(ws.root)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    task = _INFLIGHT.get(key)
    if task is None:
        task = asyncio.create_task(_build_state_uncached(ws))
        _INFLIGHT[key] = task
    try:
        state = await asyncio.shield(task)
        _CACHE[key] = state
        return state
    finally:
        if task.done() and _INFLIGHT.get(key) is task:
            del _INFLIGHT[key]


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    try:
        if not _regular(ws, "run.py"):
            return 0.0, "run.py missing or not a regular file"
        result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    except TimeoutError:
        return 0.0, "static compilation timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"static compilation raised {type(exc).__name__}: {exc}"
    return (
        (1.0, "static compilation passed")
        if getattr(result, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    try:
        return _report_score(ws)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"report contract check raised {type(exc).__name__}: {exc}"


async def _score(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get(name, 0.0), state.diagnostics.get(name, "missing score")


for _name in (
    "result_json",
    "result_evidence",
    "canonical_coverage",
    "decision_legality",
    "replay_validity",
):

    async def _check(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        return await _score(ws, _id)

    globals()[_name] = check(id=_name)(_check)
del _check


for _name in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric(ws: Workspace, _id: str = _name) -> tuple[float, str]:
        state = await _build_state(ws)
        if state.metrics is None:
            failed: list[str] = []
            if not state.official_ok:
                failed.append("official_run")
            failed.extend(
                f"{name}={state.scores.get(name, 0.0):.6f}"
                for name in (
                    "result_json",
                    "result_evidence",
                    "canonical_coverage",
                    "decision_legality",
                    "replay_validity",
                )
                if state.scores.get(name, 0.0) < 1.0 - EPS
            )
            return 0.0, f"validity gate failed; failed_checks=[{','.join(failed)}]; metric=0"
        return state.metrics[_id], "validity gate passed"

    globals()[_name] = check(id=_name)(_metric)
del _metric
