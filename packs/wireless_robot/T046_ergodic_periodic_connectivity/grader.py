from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

HORIZON = 120
PERIOD = 24
WINDOW_START = 12
CONTACT_RADIUS = 3
QUEUE_CAPACITY = 12
PERIOD_ATTEMPT_LIMIT = 8
SCENE_UNIT_BUDGET = 120
INITIAL_ENERGY = 120.0
EPS = 1e-6

ROBOT_IDS = ("A", "B", "C")
ROBOT_RANK = {robot: index for index, robot in enumerate(ROBOT_IDS)}
UNIT_IDS = tuple(f"u{index}" for index in range(1, 13))
PAIR_ROWS = (("AB", "A", "B"), ("AC", "A", "C"), ("BC", "B", "C"))
PAIR_IDS = tuple(row[0] for row in PAIR_ROWS)
MOVES = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0), "stay": (0, 0)}

UNIT_BASE = (
    ((3, 3), 1.0),
    ((5, 5), 1.1),
    ((7, 9), 1.2),
    ((9, 4), 1.0),
    ((11, 12), 1.3),
    ((14, 14), 1.1),
    ((4, 13), 1.2),
    ((13, 5), 1.0),
    ((2, 9), 0.9),
    ((16, 9), 1.1),
    ((9, 15), 1.2),
    ((9, 2), 0.9),
)


def _owner(unit_id: str) -> str:
    index = UNIT_IDS.index(unit_id)
    return ROBOT_IDS[index // 4]


SCENARIOS: dict[str, dict[str, Any]] = {
    "clustered_sources": {
        "phase": 1,
        "starts": {"A": (2, 2), "B": (2, 15), "C": (15, 2)},
        "obstacles": frozenset((8, y) for y in range(3, 15) if y not in {8, 9}),
        "shift": (0, 0),
    },
    "split_sources": {
        "phase": 4,
        "starts": {"A": (2, 2), "B": (15, 15), "C": (2, 15)},
        "obstacles": frozenset((8, y) for y in (*range(0, 6), *range(7, 11), *range(12, 18))),
        "shift": (1, 0),
    },
    "corridor_rendezvous": {
        "phase": 7,
        "starts": {"A": (1, 9), "B": (16, 3), "C": (16, 15)},
        "obstacles": frozenset((x, y) for x in (6, 11) for y in (*range(0, 8), *range(10, 18))),
        "shift": (0, 1),
    },
    "shifting_sources": {
        "phase": 10,
        "starts": {"A": (2, 9), "B": (15, 3), "C": (15, 15)},
        "obstacles": frozenset((x, 8) for x in (*range(0, 6), *range(7, 11), *range(12, 18))),
        "shift": (-1, 0),
    },
    "uneven_private_maps": {
        "phase": 13,
        "starts": {"A": (2, 3), "B": (15, 9), "C": (2, 15)},
        "obstacles": frozenset(
            {(7, y) for y in (*range(2, 7), *range(11, 16))}
            | {(11, y) for y in (*range(4, 9), *range(12, 17))}
        ),
        "shift": (0, -1),
    },
}
SCENARIO_NAMES = tuple(SCENARIOS)

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 1.0 / 3.0,
        "result_json.root": 1.0 / 3.0,
        "result_json.scenarios": 1.0 / 3.0,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 1.0 / 9.0,
        "result_evidence.slot_request_types": 1.0 / 9.0,
        "result_evidence.robot_types": 1.0 / 9.0,
        "result_evidence.unit_types": 1.0 / 9.0,
        "result_evidence.pair_period_types": 1.0 / 9.0,
        "result_evidence.observation_types": 1.0 / 9.0,
        "result_evidence.packet_types": 1.0 / 9.0,
        "result_evidence.delivery_types": 1.0 / 9.0,
        "result_evidence.mission_types": 1.0 / 9.0,
    },
    "decision_legality": {
        "decision_legality.request_domains": 1.0 / 6.0,
        "decision_legality.motion": 1.0 / 6.0,
        "decision_legality.packet_creation": 1.0 / 6.0,
        "decision_legality.local_content": 1.0 / 6.0,
        "decision_legality.fifo_contact": 1.0 / 6.0,
        "decision_legality.attempt_limits": 1.0 / 6.0,
    },
    "replay_validity": {
        "replay_validity.robot_motion": 1.0 / 6.0,
        "replay_validity.observations_units": 1.0 / 6.0,
        "replay_validity.contact_pair_periods": 1.0 / 6.0,
        "replay_validity.packet_attempts": 1.0 / 6.0,
        "replay_validity.delivery_fusion": 1.0 / 6.0,
        "replay_validity.queues_mission": 1.0 / 6.0,
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float) -> None:
        self.id = id
        self.passed = passed
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
        self.scenarios: dict[str, dict[str, Any]] = {}
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


_CACHE: dict[str, EvalState] = {}
DecisionProvider = Callable[[int, Mapping[str, Any]], Mapping[str, Any]]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_integer(component) for component in value)
    )


def _close(actual: Any, expected: Any) -> bool:
    return (
        _finite_number(actual)
        and _finite_number(expected)
        and abs(float(actual) - float(expected)) <= EPS
    )


def _strict_json(raw: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject, object_pairs_hook=unique_pairs)


def _nullable_integer(value: Any, low: int, high: int) -> bool:
    return value is None or (_integer(value) and low <= value <= high)


def _nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _scenario_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "scenario",
        "slots",
        "robots",
        "units",
        "pair_periods",
        "observations",
        "packets",
        "deliveries",
        "mission",
    }


def _slot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "slot",
        "requests",
        "served_packet_ids",
        "state",
    }


def _request_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "robot",
        "move",
        "packet_id",
        "destination",
    }


def _slot_state_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "collision_robots",
        "send_attempts",
        "contact_pairs",
        "period",
    }


def _robot_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "robot",
        "positions",
        "energy",
        "queue_lengths",
        "known_unit_versions",
    }


def _unit_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "unit_id",
        "initial_owner",
        "current_version",
        "last_observed_slot",
        "value",
        "deadline_slot",
    }


def _pair_period_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "period",
        "pair",
        "window_contact_slots",
        "successful_delivery_id",
        "satisfied",
    }


def _observation_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "observation_id",
        "robot",
        "unit_id",
        "version",
        "slot",
        "value",
    }


def _packet_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "packet_id",
        "source",
        "destination",
        "created_slot",
        "unit_versions",
        "attempt_slots",
        "attempt_nos",
        "arrival_slot",
        "outcome",
    }


def _delivery_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "delivery_id",
        "packet_id",
        "source",
        "destination",
        "slot",
        "unit_versions",
        "useful",
        "within_deadline",
    }


def _mission_shape(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {
        "collision_events",
        "motion_cost",
        "energy_used",
        "sent_units",
        "useful_summary_units",
        "retransmitted_units",
        "contact_wait_slots",
        "timely_fused_value",
    }


def _request_valid(value: Any) -> bool:
    return (
        _request_shape(value)
        and isinstance(value["robot"], str)
        and isinstance(value["move"], str)
        and _nullable_string(value["packet_id"])
        and _nullable_string(value["destination"])
    )


def _slot_valid(value: Any) -> bool:
    if not _slot_shape(value) or not _integer(value["slot"]):
        return False
    requests = value["requests"]
    served = value["served_packet_ids"]
    state = value["state"]
    return (
        isinstance(requests, list)
        and len(requests) == len(ROBOT_IDS)
        and all(_request_valid(row) for row in requests)
        and isinstance(served, list)
        and len(served) <= 2
        and all(isinstance(packet_id, str) for packet_id in served)
        and _slot_state_shape(state)
        and isinstance(state["collision_robots"], list)
        and all(isinstance(robot, str) for robot in state["collision_robots"])
        and _integer(state["send_attempts"])
        and state["send_attempts"] >= 0
        and isinstance(state["contact_pairs"], list)
        and all(isinstance(pair, str) for pair in state["contact_pairs"])
        and _integer(state["period"])
        and 0 <= state["period"] < HORIZON // PERIOD
    )


def _bounded_position(value: Any) -> bool:
    return _position(value) and 0 <= value[0] < 18 and 0 <= value[1] < 18


def _robot_valid(value: Any) -> bool:
    return (
        _robot_shape(value)
        and isinstance(value["robot"], str)
        and isinstance(value["positions"], list)
        and len(value["positions"]) == HORIZON + 1
        and all(_bounded_position(position) for position in value["positions"])
        and isinstance(value["energy"], list)
        and len(value["energy"]) == HORIZON + 1
        and all(
            _finite_number(energy) and -EPS <= float(energy) <= INITIAL_ENERGY + EPS
            for energy in value["energy"]
        )
        and isinstance(value["queue_lengths"], list)
        and len(value["queue_lengths"]) == HORIZON
        and all(
            _integer(length) and 0 <= length <= QUEUE_CAPACITY for length in value["queue_lengths"]
        )
        and isinstance(value["known_unit_versions"], list)
        and len(value["known_unit_versions"]) == HORIZON
        and all(
            isinstance(versions, list)
            and len(versions) == len(UNIT_IDS)
            and all(_integer(version) and -1 <= version <= 1 for version in versions)
            for versions in value["known_unit_versions"]
        )
    )


def _unit_valid(value: Any) -> bool:
    return (
        _unit_shape(value)
        and isinstance(value["unit_id"], str)
        and isinstance(value["initial_owner"], str)
        and _integer(value["current_version"])
        and 0 <= value["current_version"] <= 1
        and _nullable_integer(value["last_observed_slot"], 0, HORIZON - 1)
        and _finite_number(value["value"])
        and float(value["value"]) > 0.0
        and _nullable_integer(value["deadline_slot"], 0, HORIZON - 1)
    )


def _pair_period_valid(value: Any) -> bool:
    return (
        _pair_period_shape(value)
        and _integer(value["period"])
        and 0 <= value["period"] < HORIZON // PERIOD
        and isinstance(value["pair"], str)
        and isinstance(value["window_contact_slots"], list)
        and all(_integer(slot) and 0 <= slot < HORIZON for slot in value["window_contact_slots"])
        and _nullable_string(value["successful_delivery_id"])
        and isinstance(value["satisfied"], bool)
    )


def _observation_valid(value: Any) -> bool:
    return (
        _observation_shape(value)
        and isinstance(value["observation_id"], str)
        and isinstance(value["robot"], str)
        and isinstance(value["unit_id"], str)
        and _integer(value["version"])
        and 0 <= value["version"] <= 1
        and _integer(value["slot"])
        and 0 <= value["slot"] < HORIZON
        and _finite_number(value["value"])
        and float(value["value"]) > 0.0
    )


def _unit_version_items(value: Any) -> bool:
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 3
        and all(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and _integer(item[1])
            and 0 <= item[1] <= 1
            for item in value
        )
    )


def _packet_valid(value: Any) -> bool:
    return (
        _packet_shape(value)
        and all(isinstance(value[key], str) for key in ("packet_id", "source", "destination"))
        and _integer(value["created_slot"])
        and 0 <= value["created_slot"] < HORIZON
        and _unit_version_items(value["unit_versions"])
        and isinstance(value["attempt_slots"], list)
        and all(_integer(slot) and 0 <= slot < HORIZON for slot in value["attempt_slots"])
        and isinstance(value["attempt_nos"], list)
        and all(_integer(number) and number >= 0 for number in value["attempt_nos"])
        and len(value["attempt_slots"]) == len(value["attempt_nos"])
        and _nullable_integer(value["arrival_slot"], 1, HORIZON)
        and isinstance(value["outcome"], str)
        and value["outcome"] in {"delivered", "expired", "dropped", "unfinished"}
    )


def _delivery_valid(value: Any) -> bool:
    return (
        _delivery_shape(value)
        and all(
            isinstance(value[key], str)
            for key in ("delivery_id", "packet_id", "source", "destination")
        )
        and _integer(value["slot"])
        and 1 <= value["slot"] <= HORIZON
        and _unit_version_items(value["unit_versions"])
        and isinstance(value["useful"], bool)
        and isinstance(value["within_deadline"], bool)
    )


def _mission_valid(value: Any) -> bool:
    if not _mission_shape(value):
        return False
    integer_fields = (
        "collision_events",
        "motion_cost",
        "sent_units",
        "useful_summary_units",
        "retransmitted_units",
    )
    number_fields = ("energy_used", "contact_wait_slots", "timely_fused_value")
    return all(_integer(value[key]) and value[key] >= 0 for key in integer_fields) and all(
        _finite_number(value[key]) and float(value[key]) >= 0.0 for key in number_fields
    )


def _ids_unique(rows: Sequence[Any], key: str) -> bool:
    ids = [row.get(key) for row in rows if isinstance(row, dict)]
    return (
        len(ids) == len(rows)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(set(ids))
    )


def _canonical_subset(values: Any, canonical: Sequence[str]) -> bool:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return False
    selected = set(values)
    return len(selected) == len(values) and values == [
        value for value in canonical if value in selected
    ]


def _shape_error(value: Any, expected: set[str], path: str) -> str | None:
    if not isinstance(value, dict):
        return f"{path}: expected object; actual_type={type(value).__name__}"
    if set(value) != expected:
        return f"{path}: expected keys={sorted(expected)}; actual_keys={sorted(value)}"
    return None


def _slot_error(value: Any, path: str) -> str | None:
    error = _shape_error(value, {"slot", "requests", "served_packet_ids", "state"}, path)
    if error:
        return error
    if not _integer(value["slot"]):
        return f"{path}.slot: expected integer; actual={value['slot']!r}"
    requests = value["requests"]
    if not isinstance(requests, list) or len(requests) != len(ROBOT_IDS):
        length = len(requests) if isinstance(requests, list) else None
        return f"{path}.requests: expected array length=3; actual_length={length}"
    for index, request in enumerate(requests):
        request_path = f"{path}.requests[{index}]"
        error = _shape_error(request, {"robot", "move", "packet_id", "destination"}, request_path)
        if error:
            return error
        for key in ("robot", "move"):
            if not isinstance(request[key], str):
                return f"{request_path}.{key}: expected string; actual={request[key]!r}"
        for key in ("packet_id", "destination"):
            if not _nullable_string(request[key]):
                return f"{request_path}.{key}: expected string or null; actual={request[key]!r}"
    served = value["served_packet_ids"]
    if (
        not isinstance(served, list)
        or len(served) > 2
        or not all(isinstance(packet_id, str) for packet_id in served)
    ):
        return f"{path}.served_packet_ids: expected at most two strings; actual={served!r}"
    state = value["state"]
    error = _shape_error(
        state, {"collision_robots", "send_attempts", "contact_pairs", "period"}, f"{path}.state"
    )
    if error:
        return error
    if not isinstance(state["collision_robots"], list) or not all(
        isinstance(robot, str) for robot in state["collision_robots"]
    ):
        return (
            f"{path}.state.collision_robots: expected string array; "
            f"actual={state['collision_robots']!r}"
        )
    if not _integer(state["send_attempts"]) or state["send_attempts"] < 0:
        return (
            f"{path}.state.send_attempts: expected nonnegative integer; "
            f"actual={state['send_attempts']!r}"
        )
    if not isinstance(state["contact_pairs"], list) or not all(
        isinstance(pair, str) for pair in state["contact_pairs"]
    ):
        return (
            f"{path}.state.contact_pairs: expected string array; actual={state['contact_pairs']!r}"
        )
    if not _integer(state["period"]) or not 0 <= state["period"] < HORIZON // PERIOD:
        return f"{path}.state.period: expected integer in [0,4]; actual={state['period']!r}"
    return None


def _robot_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {"robot", "positions", "energy", "queue_lengths", "known_unit_versions"},
        path,
    )
    if error:
        return error
    if not isinstance(value["robot"], str):
        return f"{path}.robot: expected string; actual={value['robot']!r}"
    positions = value["positions"]
    if not isinstance(positions, list) or len(positions) != HORIZON + 1:
        length = len(positions) if isinstance(positions, list) else None
        return f"{path}.positions: expected array length=121; actual_length={length}"
    for index, position in enumerate(positions):
        if not _bounded_position(position):
            return (
                f"{path}.positions[{index}]: expected two integers in grid [0,17]^2; "
                f"actual={position!r}"
            )
    energies = value["energy"]
    if not isinstance(energies, list) or len(energies) != HORIZON + 1:
        length = len(energies) if isinstance(energies, list) else None
        return f"{path}.energy: expected array length=121; actual_length={length}"
    for index, energy in enumerate(energies):
        if not _finite_number(energy) or not -EPS <= float(energy) <= INITIAL_ENERGY + EPS:
            return (
                f"{path}.energy[{index}]: expected finite value in [0,120]; "
                f"actual={energy!r}; tolerance={EPS}"
            )
    queue_lengths = value["queue_lengths"]
    if not isinstance(queue_lengths, list) or len(queue_lengths) != HORIZON:
        length = len(queue_lengths) if isinstance(queue_lengths, list) else None
        return f"{path}.queue_lengths: expected array length=120; actual_length={length}"
    for index, length in enumerate(queue_lengths):
        if not _integer(length) or not 0 <= length <= QUEUE_CAPACITY:
            return (
                f"{path}.queue_lengths[{index}]: expected integer in [0,{QUEUE_CAPACITY}]; "
                f"actual={length!r}"
            )
    tracks = value["known_unit_versions"]
    if not isinstance(tracks, list) or len(tracks) != HORIZON:
        length = len(tracks) if isinstance(tracks, list) else None
        return f"{path}.known_unit_versions: expected array length=120; actual_length={length}"
    for slot, versions in enumerate(tracks):
        if not isinstance(versions, list) or len(versions) != len(UNIT_IDS):
            length = len(versions) if isinstance(versions, list) else None
            return (
                f"{path}.known_unit_versions[{slot}]: expected array length=12; "
                f"actual_length={length}"
            )
        for index, version in enumerate(versions):
            if not _integer(version) or not -1 <= version <= 1:
                return (
                    f"{path}.known_unit_versions[{slot}][{index}]: expected integer in [-1,1]; "
                    f"actual={version!r}"
                )
    return None


def _unit_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "unit_id",
            "initial_owner",
            "current_version",
            "last_observed_slot",
            "value",
            "deadline_slot",
        },
        path,
    )
    if error:
        return error
    for key in ("unit_id", "initial_owner"):
        if not isinstance(value[key], str):
            return f"{path}.{key}: expected string; actual={value[key]!r}"
    if not _integer(value["current_version"]) or not 0 <= value["current_version"] <= 1:
        return (
            f"{path}.current_version: expected integer in [0,1]; "
            f"actual={value['current_version']!r}"
        )
    if not _nullable_integer(value["last_observed_slot"], 0, HORIZON - 1):
        return (
            f"{path}.last_observed_slot: expected null or integer in [0,119]; "
            f"actual={value['last_observed_slot']!r}"
        )
    if not _finite_number(value["value"]) or float(value["value"]) <= 0.0:
        return (
            f"{path}.value: expected finite value > 0; actual={value['value']!r}; tolerance={EPS}"
        )
    if not _nullable_integer(value["deadline_slot"], 0, HORIZON - 1):
        return (
            f"{path}.deadline_slot: expected null or integer in [0,119]; "
            f"actual={value['deadline_slot']!r}"
        )
    return None


def _pair_period_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {"period", "pair", "window_contact_slots", "successful_delivery_id", "satisfied"},
        path,
    )
    if error:
        return error
    if not _integer(value["period"]) or not 0 <= value["period"] < HORIZON // PERIOD:
        return f"{path}.period: expected integer in [0,4]; actual={value['period']!r}"
    if not isinstance(value["pair"], str):
        return f"{path}.pair: expected string; actual={value['pair']!r}"
    slots = value["window_contact_slots"]
    if not isinstance(slots, list):
        return f"{path}.window_contact_slots: expected array; actual_type={type(slots).__name__}"
    for index, slot in enumerate(slots):
        if not _integer(slot) or not 0 <= slot < HORIZON:
            return (
                f"{path}.window_contact_slots[{index}]: expected integer in [0,119]; "
                f"actual={slot!r}"
            )
    if not _nullable_string(value["successful_delivery_id"]):
        return (
            f"{path}.successful_delivery_id: expected string or null; "
            f"actual={value['successful_delivery_id']!r}"
        )
    if not isinstance(value["satisfied"], bool):
        return f"{path}.satisfied: expected boolean; actual={value['satisfied']!r}"
    return None


def _observation_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value, {"observation_id", "robot", "unit_id", "version", "slot", "value"}, path
    )
    if error:
        return error
    for key in ("observation_id", "robot", "unit_id"):
        if not isinstance(value[key], str):
            return f"{path}.{key}: expected string; actual={value[key]!r}"
    if not _integer(value["version"]) or not 0 <= value["version"] <= 1:
        return f"{path}.version: expected integer in [0,1]; actual={value['version']!r}"
    if not _integer(value["slot"]) or not 0 <= value["slot"] < HORIZON:
        return f"{path}.slot: expected integer in [0,119]; actual={value['slot']!r}"
    if not _finite_number(value["value"]) or float(value["value"]) <= 0.0:
        return (
            f"{path}.value: expected finite value > 0; actual={value['value']!r}; tolerance={EPS}"
        )
    return None


def _unit_versions_error(value: Any, path: str) -> str | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        length = len(value) if isinstance(value, list) else None
        return f"{path}: expected array length in [1,3]; actual_length={length}"
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            return f"{path}[{index}]: expected [unit_id,version]; actual={item!r}"
        if not isinstance(item[0], str):
            return f"{path}[{index}][0]: expected string; actual={item[0]!r}"
        if not _integer(item[1]) or not 0 <= item[1] <= 1:
            return f"{path}[{index}][1]: expected integer in [0,1]; actual={item[1]!r}"
    return None


def _packet_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "packet_id",
            "source",
            "destination",
            "created_slot",
            "unit_versions",
            "attempt_slots",
            "attempt_nos",
            "arrival_slot",
            "outcome",
        },
        path,
    )
    if error:
        return error
    for key in ("packet_id", "source", "destination"):
        if not isinstance(value[key], str):
            return f"{path}.{key}: expected string; actual={value[key]!r}"
    if not _integer(value["created_slot"]) or not 0 <= value["created_slot"] < HORIZON:
        return f"{path}.created_slot: expected integer in [0,119]; actual={value['created_slot']!r}"
    error = _unit_versions_error(value["unit_versions"], f"{path}.unit_versions")
    if error:
        return error
    attempt_slots = value["attempt_slots"]
    if not isinstance(attempt_slots, list):
        return f"{path}.attempt_slots: expected array; actual_type={type(attempt_slots).__name__}"
    for index, slot in enumerate(attempt_slots):
        if not _integer(slot) or not 0 <= slot < HORIZON:
            return f"{path}.attempt_slots[{index}]: expected integer in [0,119]; actual={slot!r}"
    attempt_nos = value["attempt_nos"]
    if not isinstance(attempt_nos, list):
        return f"{path}.attempt_nos: expected array; actual_type={type(attempt_nos).__name__}"
    for index, number in enumerate(attempt_nos):
        if not _integer(number) or number < 0:
            return f"{path}.attempt_nos[{index}]: expected nonnegative integer; actual={number!r}"
    if len(attempt_slots) != len(attempt_nos):
        return (
            f"{path}.attempt_nos: expected length={len(attempt_slots)}; "
            f"actual_length={len(attempt_nos)}"
        )
    if not _nullable_integer(value["arrival_slot"], 1, HORIZON):
        return (
            f"{path}.arrival_slot: expected null or integer in [1,120]; "
            f"actual={value['arrival_slot']!r}"
        )
    if not isinstance(value["outcome"], str) or value["outcome"] not in {
        "delivered",
        "expired",
        "dropped",
        "unfinished",
    }:
        return f"{path}.outcome: expected declared enum; actual={value['outcome']!r}"
    return None


def _packet_contract_error(value: Any, path: str) -> str | None:
    error = _packet_error(value, path)
    if error:
        return error
    if value["source"] not in ROBOT_IDS:
        return f"{path}.source: expected one of {ROBOT_IDS}; actual={value['source']!r}"
    if value["destination"] not in ROBOT_IDS:
        return f"{path}.destination: expected one of {ROBOT_IDS}; actual={value['destination']!r}"
    expected_id = f"sum-{value['created_slot']:03d}-{value['source']}"
    if value["packet_id"] != expected_id:
        return f"{path}.packet_id: expected={expected_id!r}; actual={value['packet_id']!r}"
    unit_ids = [item[0] for item in value["unit_versions"]]
    if any(unit_id not in UNIT_IDS for unit_id in unit_ids):
        return f"{path}.unit_versions: expected unit references in {UNIT_IDS}; actual={unit_ids!r}"
    if len(unit_ids) != len(set(unit_ids)):
        return f"{path}.unit_versions: expected distinct unit references; actual={unit_ids!r}"
    if unit_ids != sorted(unit_ids, key=UNIT_IDS.index):
        return f"{path}.unit_versions: expected u1..u12 canonical order; actual={unit_ids!r}"
    if value["attempt_slots"] != sorted(value["attempt_slots"]):
        return (
            f"{path}.attempt_slots: expected nondecreasing order; actual={value['attempt_slots']!r}"
        )
    expected_nos = list(range(len(value["attempt_nos"])))
    if value["attempt_nos"] != expected_nos:
        return f"{path}.attempt_nos: expected={expected_nos!r}; actual={value['attempt_nos']!r}"
    delivered = value["outcome"] == "delivered"
    if delivered != (value["arrival_slot"] is not None):
        return (
            f"{path}.arrival_slot/outcome: expected non-null arrival iff delivered; "
            f"actual_arrival={value['arrival_slot']!r}; actual_outcome={value['outcome']!r}"
        )
    return None


def _delivery_error(value: Any, path: str) -> str | None:
    error = _shape_error(
        value,
        {
            "delivery_id",
            "packet_id",
            "source",
            "destination",
            "slot",
            "unit_versions",
            "useful",
            "within_deadline",
        },
        path,
    )
    if error:
        return error
    for key in ("delivery_id", "packet_id", "source", "destination"):
        if not isinstance(value[key], str):
            return f"{path}.{key}: expected string; actual={value[key]!r}"
    if not _integer(value["slot"]) or not 1 <= value["slot"] <= HORIZON:
        return f"{path}.slot: expected integer in [1,120]; actual={value['slot']!r}"
    error = _unit_versions_error(value["unit_versions"], f"{path}.unit_versions")
    if error:
        return error
    for key in ("useful", "within_deadline"):
        if not isinstance(value[key], bool):
            return f"{path}.{key}: expected boolean; actual={value[key]!r}"
    return None


def _mission_error(value: Any, path: str) -> str | None:
    expected = {
        "collision_events",
        "motion_cost",
        "energy_used",
        "sent_units",
        "useful_summary_units",
        "retransmitted_units",
        "contact_wait_slots",
        "timely_fused_value",
    }
    error = _shape_error(value, expected, path)
    if error:
        return error
    for key in (
        "collision_events",
        "motion_cost",
        "sent_units",
        "useful_summary_units",
        "retransmitted_units",
    ):
        if not _integer(value[key]) or value[key] < 0:
            return f"{path}.{key}: expected nonnegative integer; actual={value[key]!r}"
    for key in ("energy_used", "contact_wait_slots", "timely_fused_value"):
        if not _finite_number(value[key]) or float(value[key]) < 0.0:
            return (
                f"{path}.{key}: expected finite nonnegative number; "
                f"actual={value[key]!r}; tolerance={EPS}"
            )
    return None


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return indexed
    for root_index, record in enumerate(data["scenarios"]):
        if not isinstance(record, dict) or not isinstance(record.get("scenario"), str):
            continue
        name = record["scenario"]
        if name in indexed:
            continue

        def rows_and_index(field: str, key: str) -> tuple[list[Any], dict[Any, Any]]:
            rows = record.get(field) if isinstance(record.get(field), list) else []
            row_index: dict[Any, Any] = {}
            for row in rows:
                if isinstance(row, dict) and key in row and isinstance(row[key], Hashable):
                    row_index.setdefault(row[key], row)
            return rows, row_index

        slot_rows, slots = rows_and_index("slots", "slot")
        robot_rows, robots = rows_and_index("robots", "robot")
        unit_rows, units = rows_and_index("units", "unit_id")
        observation_rows, observations = rows_and_index("observations", "observation_id")
        packet_rows, packets = rows_and_index("packets", "packet_id")
        delivery_rows, deliveries = rows_and_index("deliveries", "delivery_id")
        pair_rows = (
            record.get("pair_periods") if isinstance(record.get("pair_periods"), list) else []
        )
        pair_periods: dict[tuple[Any, Any], Any] = {}
        for row in pair_rows:
            if not isinstance(row, dict):
                continue
            period = row.get("period")
            pair = row.get("pair")
            if isinstance(period, Hashable) and isinstance(pair, Hashable):
                pair_periods.setdefault((period, pair), row)
        indexed[name] = {
            "record": record,
            "root_index": root_index,
            "slot_rows": slot_rows,
            "slots": slots,
            "robot_rows": robot_rows,
            "robots": robots,
            "unit_rows": unit_rows,
            "units": units,
            "pair_period_rows": pair_rows,
            "pair_periods": pair_periods,
            "observation_rows": observation_rows,
            "observations": observations,
            "packet_rows": packet_rows,
            "packets": packets,
            "delivery_rows": delivery_rows,
            "deliveries": deliveries,
        }
    return indexed


def _in_grid(cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < 18 and 0 <= cell[1] < 18


def _open_cell(name: str, cell: tuple[int, int]) -> bool:
    return _in_grid(cell) and cell not in SCENARIOS[name]["obstacles"]


def _unit_models(name: str) -> dict[str, dict[str, Any]]:
    shift_x, shift_y = SCENARIOS[name]["shift"]
    models: dict[str, dict[str, Any]] = {}
    for unit_id, (base_cell, value) in zip(UNIT_IDS, UNIT_BASE):
        cell = (base_cell[0] + shift_x, base_cell[1] + shift_y)
        if not _open_cell(name, cell):
            for move in ("N", "E", "S", "W"):
                dx, dy = MOVES[move]
                neighbor = (cell[0] + dx, cell[1] + dy)
                if _open_cell(name, neighbor):
                    cell = neighbor
                    break
            else:
                raise ValueError(f"{name}.{unit_id}: no open transformed cell")
        models[unit_id] = {
            "unit_id": unit_id,
            "cell": cell,
            "value": float(value),
            "initial_owner": _owner(unit_id),
        }
    if len({model["cell"] for model in models.values()}) != len(models):
        raise ValueError(f"{name}: transformed unit cells must be unique")
    return models


def _update_events(name: str) -> dict[int, tuple[str, ...]]:
    if name == "shifting_sources":
        return {36: ("u2", "u6", "u10"), 72: ("u4", "u8", "u12")}
    return {48: ("u3", "u7", "u11")}


def _shortest_path(
    name: str, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    frontier: deque[tuple[int, int]] = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        cell = frontier.popleft()
        for move in ("N", "E", "S", "W"):
            dx, dy = MOVES[move]
            candidate = (cell[0] + dx, cell[1] + dy)
            if not _open_cell(name, candidate) or candidate in previous:
                continue
            previous[candidate] = cell
            if candidate == goal:
                path = [candidate]
                while path[-1] != start:
                    parent = previous[path[-1]]
                    if parent is None:
                        break
                    path.append(parent)
                return list(reversed(path))
            frontier.append(candidate)
    return [start]


def _next_move(name: str, start: tuple[int, int], goal: tuple[int, int]) -> str:
    path = _shortest_path(name, start, goal)
    if len(path) < 2:
        return "stay"
    delta = (path[1][0] - start[0], path[1][1] - start[1])
    return next(move for move, vector in MOVES.items() if vector == delta)


def _resolve_collisions(
    starts: Mapping[str, tuple[int, int]], moves: Mapping[str, str]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    destinations = {
        robot: (
            starts[robot][0] + MOVES[moves[robot]][0],
            starts[robot][1] + MOVES[moves[robot]][1],
        )
        for robot in ROBOT_IDS
    }
    collision: set[str] = set()
    while True:
        implicated: set[str] = set()
        by_destination: dict[tuple[int, int], list[str]] = {}
        for robot in ROBOT_IDS:
            by_destination.setdefault(destinations[robot], []).append(robot)
        for group in by_destination.values():
            if len(group) > 1:
                implicated.update(group)
        stopped = {robot for robot in ROBOT_IDS if moves[robot] == "stay"} | collision
        for moving in ROBOT_IDS:
            if moving in stopped:
                continue
            for occupying in stopped:
                if moving != occupying and destinations[moving] == starts[occupying]:
                    implicated.update((moving, occupying))
        newly_implicated = implicated - collision
        if not newly_implicated:
            break
        collision.update(newly_implicated)
        for robot in newly_implicated:
            destinations[robot] = starts[robot]
    return destinations, [robot for robot in ROBOT_IDS if robot in collision]


def _default_request(robot: str) -> dict[str, Any]:
    return {"robot": robot, "move": "stay", "packet_id": None, "destination": None}


def _pair_label(left: str, right: str) -> str:
    return "".join(sorted((left, right), key=ROBOT_IDS.index))


def _packet_public(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "source": packet["source"],
        "destination": packet["destination"],
        "created_slot": packet["created_slot"],
        "unit_versions": [list(item) for item in packet["unit_versions"]],
        "deadlines": [int(item["deadline"]) for item in packet["provenance"]],
    }


def _packet_record(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "source": packet["source"],
        "destination": packet["destination"],
        "created_slot": packet["created_slot"],
        "unit_versions": [list(item) for item in packet["unit_versions"]],
        "attempt_slots": list(packet["attempt_slots"]),
        "attempt_nos": list(packet["attempt_nos"]),
        "arrival_slot": packet["arrival_slot"],
        "outcome": packet["outcome"],
    }


def _simulate_scenario(
    name: str,
    decisions: Mapping[int, Mapping[str, Any]] | None = None,
    decision_provider: DecisionProvider | None = None,
) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    if (decisions is None) == (decision_provider is None):
        raise ValueError("provide exactly one of decisions or decision_provider")

    units = _unit_models(name)
    positions = {robot: tuple(SCENARIOS[name]["starts"][robot]) for robot in ROBOT_IDS}
    energy = {robot: INITIAL_ENERGY for robot in ROBOT_IDS}
    environment_versions = {unit_id: 0 for unit_id in UNIT_IDS}
    local_versions = {
        robot: {
            unit_id: 0 if units[unit_id]["initial_owner"] == robot else -1 for unit_id in UNIT_IDS
        }
        for robot in ROBOT_IDS
    }
    local_provenance: dict[str, dict[str, dict[str, int] | None]] = {
        robot: {
            unit_id: (
                {"version": 0, "slot": 0, "deadline": 23}
                if units[unit_id]["initial_owner"] == robot
                else None
            )
            for unit_id in UNIT_IDS
        }
        for robot in ROBOT_IDS
    }
    queues: dict[str, list[str]] = {robot: [] for robot in ROBOT_IDS}
    packets: dict[str, dict[str, Any]] = {}
    packet_order: list[str] = []
    arrivals: dict[int, list[str]] = {}
    arrived_this_slot: dict[str, list[dict[str, Any]]] = {robot: [] for robot in ROBOT_IDS}

    slots_out: list[dict[str, Any]] = []
    observations_out: list[dict[str, Any]] = []
    deliveries_out: list[dict[str, Any]] = []
    position_tracks = {robot: [[*positions[robot]]] for robot in ROBOT_IDS}
    energy_tracks = {robot: [energy[robot]] for robot in ROBOT_IDS}
    queue_tracks: dict[str, list[int]] = {robot: [] for robot in ROBOT_IDS}
    knowledge_tracks: dict[str, list[list[int]]] = {robot: [] for robot in ROBOT_IDS}
    movement_end_positions: list[tuple[int, int]] = []
    physical_observed_units: set[str] = set()
    observation_slots: dict[str, list[int]] = {unit_id: [] for unit_id in UNIT_IDS}

    pair_periods: dict[tuple[int, str], dict[str, Any]] = {
        (period, pair): {
            "period": period,
            "pair": pair,
            "window_contact_slots": [],
            "successful_delivery_id": None,
            "satisfied": False,
        }
        for period in range(HORIZON // PERIOD)
        for pair in PAIR_IDS
    }
    attempts_by_period = {
        (robot, period): 0 for robot in ROBOT_IDS for period in range(HORIZON // PERIOD)
    }
    send_attempts = 0
    sent_units = 0
    retransmitted_units = 0
    useful_summary_units = 0
    contact_wait_slots = 0
    collision_events = 0
    motion_cost = 0
    fusion_ages: list[int] = []
    timely_versions: dict[str, set[int]] = {unit_id: set() for unit_id in UNIT_IDS}

    legality: dict[str, list[Any]] = {
        rule.removeprefix("decision_legality."): [True, ""]
        for rule in ATOM_SHARES["decision_legality"]
    }

    def fail(rule: str, detail: str) -> None:
        if legality[rule][0]:
            legality[rule] = [False, detail]

    def clean_expired_heads(slot: int) -> None:
        for robot in ROBOT_IDS:
            while queues[robot]:
                packet = packets[queues[robot][0]]
                if not all(int(item["deadline"]) < slot for item in packet["provenance"]):
                    break
                queues[robot].pop(0)
                packet["outcome"] = "expired"

    def process_arrivals(slot: int, *, terminal: bool) -> None:
        nonlocal useful_summary_units, contact_wait_slots
        arrived_this_slot.clear()
        arrived_this_slot.update({robot: [] for robot in ROBOT_IDS})
        for packet_id in sorted(arrivals.pop(slot, [])):
            packet = packets[packet_id]
            packet["arrival_slot"] = slot
            packet["outcome"] = "delivered"
            destination = str(packet["destination"])
            useful_items = 0
            within_deadline = False
            if not terminal:
                for unit_version, provenance in zip(packet["unit_versions"], packet["provenance"]):
                    unit_id, version = str(unit_version[0]), int(unit_version[1])
                    if version <= local_versions[destination][unit_id]:
                        continue
                    local_versions[destination][unit_id] = version
                    local_provenance[destination][unit_id] = dict(provenance)
                    useful_items += 1
                    fusion_ages.append(slot - int(provenance["slot"]))
                    if slot <= int(provenance["deadline"]):
                        within_deadline = True
                        if destination != units[unit_id]["initial_owner"]:
                            timely_versions[unit_id].add(version)
                useful_summary_units += useful_items
                if useful_items:
                    contact_wait_slots += int(packet["service_slot"]) - int(packet["created_slot"])
            delivery_id = f"del-{packet_id}"
            delivery = {
                "delivery_id": delivery_id,
                "packet_id": packet_id,
                "source": packet["source"],
                "destination": packet["destination"],
                "slot": slot,
                "unit_versions": [list(item) for item in packet["unit_versions"]],
                "useful": bool(useful_items),
                "within_deadline": bool(within_deadline),
            }
            deliveries_out.append(delivery)
            arrived_this_slot[destination].append(_packet_public(packet))
            if terminal or slot % PERIOD < WINDOW_START:
                continue
            period = slot // PERIOD
            pair = _pair_label(str(packet["source"]), destination)
            pair_record = pair_periods[(period, pair)]
            if not pair_record["satisfied"]:
                pair_record["satisfied"] = True
                pair_record["successful_delivery_id"] = delivery_id

    for slot in range(HORIZON):
        process_arrivals(slot, terminal=False)
        clean_expired_heads(slot)
        for unit_id in _update_events(name).get(slot, ()):
            environment_versions[unit_id] = 1

        contact_pairs: list[str] = []
        contact_neighbors: dict[str, list[str]] = {robot: [] for robot in ROBOT_IDS}
        for pair, left, right in PAIR_ROWS:
            distance = abs(positions[left][0] - positions[right][0]) + abs(
                positions[left][1] - positions[right][1]
            )
            if distance <= CONTACT_RADIUS:
                contact_pairs.append(pair)
                contact_neighbors[left].append(right)
                contact_neighbors[right].append(left)
                if slot % PERIOD >= WINDOW_START:
                    pair_periods[(slot // PERIOD, pair)]["window_contact_slots"].append(slot)

        static_view = {
            "grid": [18, 18],
            "obstacles": [list(cell) for cell in sorted(SCENARIOS[name]["obstacles"])],
            "initial_positions": {
                robot: list(SCENARIOS[name]["starts"][robot]) for robot in ROBOT_IDS
            },
            "units": [
                {
                    "unit_id": unit_id,
                    "target": list(units[unit_id]["cell"]),
                    "value": units[unit_id]["value"],
                    "initial_owner": units[unit_id]["initial_owner"],
                }
                for unit_id in UNIT_IDS
            ],
            "period": PERIOD,
            "window_start": WINDOW_START,
            "contact_radius": CONTACT_RADIUS,
        }
        role_views: dict[str, dict[str, Any]] = {}
        for robot in ROBOT_IDS:
            role_views[robot] = {
                "robot": robot,
                "slot": slot,
                "position": list(positions[robot]),
                "energy": energy[robot],
                "trajectory": [list(cell) for cell in position_tracks[robot]],
                "known_units": [
                    {
                        "unit_id": unit_id,
                        "version": local_versions[robot][unit_id],
                        "provenance_slot": (
                            local_provenance[robot][unit_id]["slot"]
                            if local_provenance[robot][unit_id] is not None
                            else None
                        ),
                        "deadline": (
                            local_provenance[robot][unit_id]["deadline"]
                            if local_provenance[robot][unit_id] is not None
                            else None
                        ),
                    }
                    for unit_id in UNIT_IDS
                ],
                "queue": [_packet_public(packets[packet_id]) for packet_id in queues[robot]],
                "arrived_summaries": list(arrived_this_slot[robot]),
                "contact_neighbors": list(contact_neighbors[robot]),
                "static": static_view,
            }
        views = {"roles": role_views}

        supplied: Mapping[str, Any]
        if decision_provider is not None:
            candidate = decision_provider(slot, views)
            supplied = candidate if isinstance(candidate, Mapping) else {}
        else:
            candidate = decisions.get(slot, {}) if decisions is not None else {}
            supplied = candidate if isinstance(candidate, Mapping) else {}
        if not isinstance(candidate, Mapping):
            fail("request_domains", f"{name}.slots[{slot}]: decision is not an object")

        raw_creations = supplied.get("creations", [])
        if isinstance(raw_creations, Mapping):
            creation_rows = [
                raw_creations[robot] for robot in ROBOT_IDS if raw_creations.get(robot) is not None
            ]
        elif isinstance(raw_creations, list):
            creation_rows = list(raw_creations)
        else:
            creation_rows = []
            fail("packet_creation", f"{name}.slot[{slot}].creations: expected array or mapping")

        creations_seen: Counter[str] = Counter()
        for creation_index, raw_creation in enumerate(creation_rows):
            if not isinstance(raw_creation, Mapping):
                fail(
                    "packet_creation",
                    f"{name}.slot[{slot}].creations[{creation_index}]: expected object",
                )
                continue
            source = raw_creation.get("source")
            created_slot = raw_creation.get("created_slot")
            packet_id = raw_creation.get("packet_id")
            destination = raw_creation.get("destination")
            content = raw_creation.get("unit_versions")
            if source not in ROBOT_IDS:
                fail(
                    "packet_creation",
                    f"{name}.slot[{slot}].creations[{creation_index}].source: {source!r}",
                )
                continue
            source = str(source)
            creations_seen[source] += 1
            expected_id = f"sum-{slot:03d}-{source}"
            creation_ok = (
                creations_seen[source] == 1
                and created_slot == slot
                and packet_id == expected_id
                and destination in ROBOT_IDS
                and destination != source
                and packet_id not in packets
            )
            if not creation_ok:
                fail(
                    "packet_creation",
                    f"{name}.slot[{slot}].creation[{source}]: expected id={expected_id!r}, "
                    "one packet, current slot, unique ID, and different known destination",
                )
                continue
            content_ok = (
                isinstance(content, list)
                and 1 <= len(content) <= 3
                and all(
                    isinstance(item, list)
                    and len(item) == 2
                    and item[0] in UNIT_IDS
                    and _integer(item[1])
                    for item in content
                )
            )
            if content_ok:
                content_ids = [str(item[0]) for item in content]
                content_ok = len(content_ids) == len(set(content_ids)) and content_ids == sorted(
                    content_ids, key=UNIT_IDS.index
                )
            if not content_ok:
                fail(
                    "local_content",
                    f"{name}.slot[{slot}].creation[{source}].unit_versions: expected 1..3 "
                    "distinct canonical-order pairs",
                )
                continue
            provenance: list[dict[str, int]] = []
            locally_known = True
            for item in content:
                unit_id, version = str(item[0]), int(item[1])
                local = local_provenance[source][unit_id]
                if local is None or local_versions[source][unit_id] != version:
                    locally_known = False
                    break
                provenance.append(dict(local))
            if not locally_known:
                fail(
                    "local_content",
                    f"{name}.slot[{slot}].creation[{source}]: content is not the source's "
                    "current local version",
                )
                continue
            packet = {
                "packet_id": packet_id,
                "source": source,
                "destination": destination,
                "created_slot": slot,
                "unit_versions": [list(item) for item in content],
                "provenance": provenance,
                "attempt_slots": [],
                "attempt_nos": [],
                "arrival_slot": None,
                "outcome": "unfinished",
                "service_slot": None,
            }
            packets[str(packet_id)] = packet
            packet_order.append(str(packet_id))
            if len(queues[source]) >= QUEUE_CAPACITY:
                packet["outcome"] = "dropped"
            elif all(int(item["deadline"]) < slot for item in provenance):
                packet["outcome"] = "expired"
            else:
                queues[source].append(str(packet_id))
                queues[source].sort(key=lambda value: (int(packets[value]["created_slot"]), value))

        supplied_requests = supplied.get("requests")
        if isinstance(supplied_requests, Mapping):
            request_rows = [
                supplied_requests.get(robot, _default_request(robot)) for robot in ROBOT_IDS
            ]
        elif isinstance(supplied_requests, list):
            request_rows = list(supplied_requests)
        else:
            request_rows = [_default_request(robot) for robot in ROBOT_IDS]
            fail("request_domains", f"{name}.slot[{slot}].requests: expected three requests")
        if len(request_rows) != len(ROBOT_IDS):
            fail(
                "request_domains",
                f"{name}.slot[{slot}].requests: expected length=3; actual={len(request_rows)}",
            )

        normalized_requests: list[dict[str, Any]] = []
        move_names: dict[str, str] = {}
        eligible: list[dict[str, Any]] = []
        for robot_index, robot in enumerate(ROBOT_IDS):
            raw_request = (
                request_rows[robot_index]
                if robot_index < len(request_rows)
                else _default_request(robot)
            )
            if not isinstance(raw_request, Mapping):
                raw_request = _default_request(robot)
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].requests[{robot_index}]: expected object",
                )
            row = {
                "robot": raw_request.get("robot"),
                "move": raw_request.get("move"),
                "packet_id": raw_request.get("packet_id"),
                "destination": raw_request.get("destination"),
            }
            normalized_requests.append(row)
            if row["robot"] != robot:
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].requests[{robot_index}].robot: "
                    f"expected={robot!r}; actual={row['robot']!r}",
                )

            move = row["move"]
            if not isinstance(move, str) or move not in MOVES:
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].request[{robot}].move: invalid enum={move!r}",
                )
                move = "stay"
            dx, dy = MOVES[str(move)]
            destination_cell = (positions[robot][0] + dx, positions[robot][1] + dy)
            if move != "stay" and (
                not _open_cell(name, destination_cell) or energy[robot] + EPS < 1.0
            ):
                fail(
                    "motion",
                    f"{name}.slot[{slot}].request[{robot}].move: {move!r} violates map or energy",
                )
                move = "stay"
            move_names[robot] = str(move)

            packet_id = row["packet_id"]
            destination = row["destination"]
            no_send = packet_id is None and destination is None
            send = isinstance(packet_id, str) and isinstance(destination, str)
            if not (no_send or send):
                fail(
                    "request_domains",
                    f"{name}.slot[{slot}].request[{robot}]: packet_id and destination "
                    "must both be strings or both null",
                )
                continue
            if no_send:
                continue
            head = queues[robot][0] if queues[robot] else None
            if (
                packet_id != head
                or packet_id not in packets
                or packets[str(packet_id)]["destination"] != destination
                or destination not in contact_neighbors[robot]
            ):
                fail(
                    "fifo_contact",
                    f"{name}.slot[{slot}].request[{robot}]: expected head={head!r}, fixed "
                    f"destination, and current contact; actual packet={packet_id!r}, "
                    f"destination={destination!r}",
                )
                continue
            packet = packets[str(packet_id)]
            units_in_packet = len(packet["unit_versions"])
            period = slot // PERIOD
            if (
                attempts_by_period[(robot, period)] >= PERIOD_ATTEMPT_LIMIT
                or sent_units + units_in_packet > SCENE_UNIT_BUDGET
            ):
                fail(
                    "attempt_limits",
                    f"{name}.slot[{slot}].request[{robot}]: period attempts or scene "
                    "sent-unit budget exceeded",
                )
                continue
            attempt_no = len(packet["attempt_slots"])
            packet["attempt_slots"].append(slot)
            packet["attempt_nos"].append(attempt_no)
            attempts_by_period[(robot, period)] += 1
            sent_units += units_in_packet
            send_attempts += 1
            if attempt_no >= 1:
                retransmitted_units += units_in_packet
            eligible.append(
                {
                    "packet": packet,
                    "packet_id": packet_id,
                    "source": robot,
                    "destination": destination,
                    "created_slot": packet["created_slot"],
                }
            )

        selected = sorted(
            eligible,
            key=lambda item: (
                int(item["created_slot"]),
                ROBOT_RANK[str(item["source"])],
                str(item["packet_id"]),
            ),
        )[:2]
        served_packet_ids = [str(item["packet_id"]) for item in selected]
        for item in selected:
            source = str(item["source"])
            destination = str(item["destination"])
            packet = item["packet"]
            success = (
                7 * slot
                + 3 * int(SCENARIOS[name]["phase"])
                + 2 * ROBOT_RANK[source]
                + ROBOT_RANK[destination]
            ) % 19 > 1
            if not success:
                continue
            packet_id = str(item["packet_id"])
            if queues[source] and queues[source][0] == packet_id:
                queues[source].pop(0)
            packet["service_slot"] = slot
            arrivals.setdefault(slot + 1, []).append(packet_id)

        destinations, collision_robots = _resolve_collisions(positions, move_names)
        collision_events += len(collision_robots)
        for robot in ROBOT_IDS:
            successful_move = robot not in collision_robots and move_names[robot] != "stay"
            if successful_move:
                energy[robot] -= 1.0
                motion_cost += 1
            positions[robot] = destinations[robot]
            position_tracks[robot].append([*positions[robot]])
            energy_tracks[robot].append(energy[robot])
            movement_end_positions.append(positions[robot])

        for robot in ROBOT_IDS:
            for unit_id in UNIT_IDS:
                target = units[unit_id]["cell"]
                distance = abs(positions[robot][0] - target[0]) + abs(
                    positions[robot][1] - target[1]
                )
                if distance > 1:
                    continue
                version = environment_versions[unit_id]
                observations_out.append(
                    {
                        "observation_id": f"obs-{slot:03d}-{robot}-{unit_id}",
                        "robot": robot,
                        "unit_id": unit_id,
                        "version": version,
                        "slot": slot,
                        "value": units[unit_id]["value"],
                    }
                )
                physical_observed_units.add(unit_id)
                observation_slots[unit_id].append(slot)
                if version >= local_versions[robot][unit_id]:
                    local_versions[robot][unit_id] = version
                    local_provenance[robot][unit_id] = {
                        "version": version,
                        "slot": slot,
                        "deadline": (slot // PERIOD + 1) * PERIOD - 1,
                    }

        for robot in ROBOT_IDS:
            queue_tracks[robot].append(len(queues[robot]))
            knowledge_tracks[robot].append([local_versions[robot][unit_id] for unit_id in UNIT_IDS])
        slots_out.append(
            {
                "slot": slot,
                "requests": normalized_requests,
                "served_packet_ids": served_packet_ids,
                "state": {
                    "collision_robots": collision_robots,
                    "send_attempts": send_attempts,
                    "contact_pairs": contact_pairs,
                    "period": slot // PERIOD,
                },
            }
        )

    process_arrivals(HORIZON, terminal=True)
    timely_fused_value = sum(
        units[unit_id]["value"] for unit_id in UNIT_IDS if timely_versions[unit_id]
    )
    energy_used = sum(INITIAL_ENERGY - energy[robot] for robot in ROBOT_IDS)
    unit_records = []
    for unit_id in UNIT_IDS:
        last_observed = max(observation_slots[unit_id], default=None)
        unit_records.append(
            {
                "unit_id": unit_id,
                "initial_owner": units[unit_id]["initial_owner"],
                "current_version": environment_versions[unit_id],
                "last_observed_slot": last_observed,
                "value": units[unit_id]["value"],
                "deadline_slot": (
                    (last_observed // PERIOD + 1) * PERIOD - 1
                    if last_observed is not None
                    else None
                ),
            }
        )
    pair_records = [
        pair_periods[(period, pair)] for period in range(HORIZON // PERIOD) for pair in PAIR_IDS
    ]
    mission = {
        "collision_events": collision_events,
        "motion_cost": motion_cost,
        "energy_used": energy_used,
        "sent_units": sent_units,
        "useful_summary_units": useful_summary_units,
        "retransmitted_units": retransmitted_units,
        "contact_wait_slots": contact_wait_slots,
        "timely_fused_value": timely_fused_value,
    }
    record = {
        "scenario": name,
        "slots": slots_out,
        "robots": [
            {
                "robot": robot,
                "positions": position_tracks[robot],
                "energy": energy_tracks[robot],
                "queue_lengths": queue_tracks[robot],
                "known_unit_versions": knowledge_tracks[robot],
            }
            for robot in ROBOT_IDS
        ],
        "units": unit_records,
        "pair_periods": pair_records,
        "observations": observations_out,
        "packets": [_packet_record(packets[packet_id]) for packet_id in packet_order],
        "deliveries": sorted(
            deliveries_out, key=lambda row: (int(row["slot"]), str(row["delivery_id"]))
        ),
        "mission": mission,
    }
    return {
        "record": record,
        "legality": legality,
        "fusion_ages": fusion_ages,
        "movement_end_positions": movement_end_positions,
        "physical_observed_units": physical_observed_units,
    }


def _simulate_policy(name: str, provider: DecisionProvider) -> dict[str, Any]:
    return _simulate_scenario(name, decision_provider=provider)


def _atom(check_id: str, rule: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{rule}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


def _score_atoms(atoms: Sequence[AtomVerdict], expected_share: Mapping[str, float]) -> float:
    ids = [atom.id for atom in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_share):
        raise ValueError("atom IDs do not match the declared fixed set")
    if any(not math.isfinite(value) or value <= 0 for value in expected_share.values()):
        raise ValueError("atom shares must be finite and positive")
    if not math.isclose(sum(expected_share.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("declared atom shares must sum to 1.0")
    for atom in atoms:
        if not math.isclose(atom.share, expected_share[atom.id], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"atom share differs for {atom.id}")
    score = sum(atom.share for atom in atoms if atom.passed)
    return (
        1.0 if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-12) else max(0.0, min(1.0, score))
    )


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    root_object = parse_ok and isinstance(data, dict)
    root_shape = root_object and set(data) == {"scenarios"}
    scenarios_list = root_object and isinstance(data.get("scenarios"), list)
    return [
        _atom(
            "result_json",
            "parse",
            bool(parse_ok),
            f"results.json: expected strict JSON; actual_type={type(data).__name__}",
        ),
        _atom(
            "result_json",
            "root",
            bool(root_shape),
            "results.json: expected root keys={'scenarios'}; "
            f"actual_keys={sorted(data) if isinstance(data, dict) else None}",
        ),
        _atom(
            "result_json",
            "scenarios",
            bool(scenarios_list),
            "results.json.scenarios: expected an array",
        ),
    ]


def _known_scenarios(scenarios: Mapping[str, dict[str, Any]]) -> tuple[str, ...]:
    return tuple(name for name in SCENARIO_NAMES if name in scenarios)


def _evidence_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    known = _known_scenarios(scenarios)
    scenario_ok = True
    scenario_detail = "known scenario records have the exact declared key set"
    checks: dict[str, tuple[bool, str]] = {
        "slot_request_types": (True, "slot and request records satisfy declared types"),
        "robot_types": (True, "robot records satisfy declared types and lengths"),
        "unit_types": (True, "unit records satisfy declared types"),
        "pair_period_types": (True, "pair-period records satisfy declared types"),
        "observation_types": (True, "observation records satisfy declared types and unique IDs"),
        "packet_types": (True, "packet records satisfy declared types and unique IDs"),
        "delivery_types": (True, "delivery records satisfy declared types and unique IDs"),
        "mission_types": (True, "mission records satisfy declared summary types"),
    }

    def set_failure(rule: str, detail: str) -> None:
        if checks[rule][0]:
            checks[rule] = (False, detail)

    for name in known:
        record = scenarios[name]["record"]
        error = _shape_error(
            record,
            {
                "scenario",
                "slots",
                "robots",
                "units",
                "pair_periods",
                "observations",
                "packets",
                "deliveries",
                "mission",
            },
            name,
        )
        if error and scenario_ok:
            scenario_ok = False
            scenario_detail = error

        packet_rows = record.get("packets")
        packet_index: dict[str, Mapping[str, Any]] = {}
        if isinstance(packet_rows, list):
            for row in packet_rows:
                if isinstance(row, Mapping) and isinstance(row.get("packet_id"), str):
                    packet_index.setdefault(str(row["packet_id"]), row)
        delivery_rows = record.get("deliveries")
        delivery_ids = (
            {
                row["delivery_id"]
                for row in delivery_rows
                if isinstance(row, Mapping) and isinstance(row.get("delivery_id"), str)
            }
            if isinstance(delivery_rows, list)
            else set()
        )

        slot_rows = record.get("slots")
        if not isinstance(slot_rows, list):
            set_failure("slot_request_types", f"{name}.slots: expected array")
        else:
            for index, row in enumerate(slot_rows):
                error = _slot_error(row, f"{name}.slots[{index}]")
                if error:
                    set_failure("slot_request_types", error)
                    break
                actual_order = [request["robot"] for request in row["requests"]]
                if actual_order != list(ROBOT_IDS):
                    set_failure(
                        "slot_request_types",
                        f"{name}.slots[{index}].requests: expected robot "
                        f"order={list(ROBOT_IDS)!r}; "
                        f"actual={actual_order!r}",
                    )
                    break
                collision_robots = row["state"]["collision_robots"]
                if not _canonical_subset(collision_robots, ROBOT_IDS):
                    set_failure(
                        "slot_request_types",
                        f"{name}.slots[{index}].state.collision_robots: expected unique "
                        f"canonical robot subset; actual={collision_robots!r}",
                    )
                    break
                contact_pairs = row["state"]["contact_pairs"]
                if not _canonical_subset(contact_pairs, PAIR_IDS):
                    set_failure(
                        "slot_request_types",
                        f"{name}.slots[{index}].state.contact_pairs: expected unique "
                        f"canonical pair subset; actual={contact_pairs!r}",
                    )
                    break
                unknown_served = [
                    packet_id
                    for packet_id in row["served_packet_ids"]
                    if packet_id not in packet_index
                ]
                if unknown_served:
                    set_failure(
                        "slot_request_types",
                        f"{name}.slots[{index}].served_packet_ids: unknown packet "
                        f"references={unknown_served!r}",
                    )
                    break

        robot_rows = record.get("robots")
        if not isinstance(robot_rows, list):
            set_failure("robot_types", f"{name}.robots: expected array")
        else:
            for index, row in enumerate(robot_rows):
                error = _robot_error(row, f"{name}.robots[{index}]")
                if error:
                    set_failure("robot_types", error)
                    break

        unit_rows = record.get("units")
        if not isinstance(unit_rows, list):
            set_failure("unit_types", f"{name}.units: expected array")
        else:
            for index, row in enumerate(unit_rows):
                error = _unit_error(row, f"{name}.units[{index}]")
                if error:
                    set_failure("unit_types", error)
                    break

        pair_rows = record.get("pair_periods")
        if not isinstance(pair_rows, list):
            set_failure("pair_period_types", f"{name}.pair_periods: expected array")
        else:
            for index, row in enumerate(pair_rows):
                error = _pair_period_error(row, f"{name}.pair_periods[{index}]")
                if error:
                    set_failure("pair_period_types", error)
                    break
                delivery_id = row["successful_delivery_id"]
                if delivery_id is not None and delivery_id not in delivery_ids:
                    set_failure(
                        "pair_period_types",
                        f"{name}.pair_periods[{index}].successful_delivery_id: "
                        f"unknown delivery reference={delivery_id!r}",
                    )
                    break

        observation_rows = record.get("observations")
        if not isinstance(observation_rows, list):
            set_failure("observation_types", f"{name}.observations: expected array")
        else:
            for index, row in enumerate(observation_rows):
                error = _observation_error(row, f"{name}.observations[{index}]")
                if error:
                    set_failure("observation_types", error)
                    break
                if row["robot"] not in ROBOT_IDS or row["unit_id"] not in UNIT_IDS:
                    set_failure(
                        "observation_types",
                        f"{name}.observations[{index}]: unknown robot/unit reference "
                        f"robot={row['robot']!r}, unit_id={row['unit_id']!r}",
                    )
                    break
                expected_id = f"obs-{row['slot']:03d}-{row['robot']}-{row['unit_id']}"
                if row["observation_id"] != expected_id:
                    set_failure(
                        "observation_types",
                        f"{name}.observations[{index}].observation_id: "
                        f"expected={expected_id!r}; actual={row['observation_id']!r}",
                    )
                    break
            if checks["observation_types"][0] and not _ids_unique(
                observation_rows, "observation_id"
            ):
                set_failure(
                    "observation_types", f"{name}.observations.observation_id: expected unique IDs"
                )
            if checks["observation_types"][0]:
                expected_order = sorted(
                    observation_rows,
                    key=lambda row: (
                        row["slot"],
                        ROBOT_RANK[row["robot"]],
                        UNIT_IDS.index(row["unit_id"]),
                    ),
                )
                if observation_rows != expected_order:
                    set_failure(
                        "observation_types",
                        f"{name}.observations: expected (slot,robot-rank,unit-rank) order",
                    )

        if not isinstance(packet_rows, list):
            set_failure("packet_types", f"{name}.packets: expected array")
        else:
            for index, row in enumerate(packet_rows):
                error = _packet_contract_error(row, f"{name}.packets[{index}]")
                if error:
                    set_failure("packet_types", error)
                    break
            if checks["packet_types"][0] and not _ids_unique(packet_rows, "packet_id"):
                set_failure("packet_types", f"{name}.packets.packet_id: expected unique IDs")
            if checks["packet_types"][0]:
                expected_order = sorted(
                    packet_rows,
                    key=lambda row: (
                        row["created_slot"],
                        ROBOT_RANK[row["source"]],
                        row["packet_id"],
                    ),
                )
                if packet_rows != expected_order:
                    set_failure(
                        "packet_types",
                        f"{name}.packets: expected (created_slot,source-rank,packet_id) order",
                    )

        if not isinstance(delivery_rows, list):
            set_failure("delivery_types", f"{name}.deliveries: expected array")
        else:
            for index, row in enumerate(delivery_rows):
                error = _delivery_error(row, f"{name}.deliveries[{index}]")
                if error:
                    set_failure("delivery_types", error)
                    break
                packet = packet_index.get(row["packet_id"])
                if packet is None:
                    set_failure(
                        "delivery_types",
                        f"{name}.deliveries[{index}].packet_id: "
                        f"unknown packet reference={row['packet_id']!r}",
                    )
                    break
                expected_id = f"del-{row['packet_id']}"
                if row["delivery_id"] != expected_id:
                    set_failure(
                        "delivery_types",
                        f"{name}.deliveries[{index}].delivery_id: "
                        f"expected={expected_id!r}; actual={row['delivery_id']!r}",
                    )
                    break
                for key in ("source", "destination", "unit_versions"):
                    if row[key] != packet.get(key):
                        set_failure(
                            "delivery_types",
                            f"{name}.deliveries[{index}].{key}: expected packet value="
                            f"{packet.get(key)!r}; actual={row[key]!r}",
                        )
                        break
                if not checks["delivery_types"][0]:
                    break
            if checks["delivery_types"][0] and not _ids_unique(delivery_rows, "delivery_id"):
                set_failure("delivery_types", f"{name}.deliveries.delivery_id: expected unique IDs")
            if checks["delivery_types"][0]:
                expected_order = sorted(
                    delivery_rows, key=lambda row: (row["slot"], row["delivery_id"])
                )
                if delivery_rows != expected_order:
                    set_failure(
                        "delivery_types",
                        f"{name}.deliveries: expected (slot,delivery_id) order",
                    )

        error = _mission_error(record.get("mission"), f"{name}.mission")
        if error:
            set_failure("mission_types", error)

    return [
        _atom(
            "result_evidence",
            "scenario_shape",
            scenario_ok,
            scenario_detail,
        ),
        *[
            _atom("result_evidence", rule, checks[rule][0], checks[rule][1])
            for rule in (
                "slot_request_types",
                "robot_types",
                "unit_types",
                "pair_period_types",
                "observation_types",
                "packet_types",
                "delivery_types",
                "mission_types",
            )
        ],
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["results.scenarios: missing or not a list"] * 8
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_NAMES}
    supplied_scenarios: list[Any] = []
    for index, row in enumerate(data["scenarios"]):
        name = row.get("scenario") if isinstance(row, dict) else None
        supplied_scenarios.append(name)
        if not isinstance(name, str) or name not in by_name:
            defects.append(f"results.scenarios[{index}]: unknown scenario key={name!r}")
        else:
            by_name[name].append(row)
    if (
        len(supplied_scenarios) == len(SCENARIO_NAMES)
        and all(isinstance(name, str) for name in supplied_scenarios)
        and set(supplied_scenarios) == set(SCENARIO_NAMES)
        and supplied_scenarios != list(SCENARIO_NAMES)
    ):
        defects.append("results.scenarios: canonical order mismatch")

    def count_keys(
        record: Mapping[str, Any],
        field: str,
        key_value: Callable[[Mapping[str, Any]], Any],
        expected: Sequence[Any],
        label: str,
    ) -> None:
        rows = record.get(field)
        if not isinstance(rows, list):
            defects.extend(f"{label}: missing key {value!r}" for value in expected)
            return
        counts = {value: 0 for value in expected}
        supplied: list[Any] = []
        for index, row in enumerate(rows):
            value = key_value(row) if isinstance(row, Mapping) else None
            supplied.append(value)
            try:
                known = value in counts
            except TypeError:
                known = False
            if known and type(value) is type(expected[0]):
                counts[value] += 1
            else:
                defects.append(f"{label}[{index}]: unknown key={value!r}")
        for value, count in counts.items():
            if count == 0:
                defects.append(f"{label}: missing key {value!r}")
            defects.extend(f"{label}: duplicate key {value!r}" for _ in range(max(0, count - 1)))
        if (
            len(supplied) == len(expected)
            and all(count == 1 for count in counts.values())
            and supplied != list(expected)
        ):
            defects.append(f"{label}: canonical order mismatch")

    expected_pairs = [(period, pair) for period in range(HORIZON // PERIOD) for pair in PAIR_IDS]
    for name in SCENARIO_NAMES:
        matches = by_name[name]
        if not matches:
            defects.append(f"scenario {name}: missing record")
            defects.extend(f"{name}.slots: missing key {slot!r}" for slot in range(HORIZON))
            defects.extend(f"{name}.robots: missing key {robot!r}" for robot in ROBOT_IDS)
            defects.extend(f"{name}.units: missing key {unit!r}" for unit in UNIT_IDS)
            defects.extend(f"{name}.pair_periods: missing key {pair!r}" for pair in expected_pairs)
            continue
        defects.extend(f"scenario {name}: duplicate record" for _ in range(len(matches) - 1))
        record = matches[0]
        count_keys(
            record, "slots", lambda row: row.get("slot"), list(range(HORIZON)), f"{name}.slots"
        )
        count_keys(record, "robots", lambda row: row.get("robot"), ROBOT_IDS, f"{name}.robots")
        count_keys(record, "units", lambda row: row.get("unit_id"), UNIT_IDS, f"{name}.units")
        count_keys(
            record,
            "pair_periods",
            lambda row: (row.get("period"), row.get("pair")),
            expected_pairs,
            f"{name}.pair_periods",
        )
    score = max(0.0, 4.0 - 0.5 * len(defects)) / 4.0
    return score, defects


def _coverage_diagnostic(score: float, defects: Sequence[str]) -> str:
    if not defects:
        return "failed_atoms=[]; defect_count=0"
    shown = " | ".join(defects[:24])
    if len(defects) > 24:
        shown += f" | ... {len(defects) - 24} more"
    return (
        "failed_atoms=[canonical_coverage.defect]; "
        f"defect_count={len(defects)}; {shown}; normalized={score:.9g}"
    )


def _decision_map(item: Mapping[str, Any]) -> dict[int, Mapping[str, Any]] | None:
    slots = item.get("slots")
    if not isinstance(slots, Mapping) or any(slot not in slots for slot in range(HORIZON)):
        return None
    record = item.get("record")
    if not isinstance(record, Mapping):
        return None
    packet_rows = record.get("packets")
    if not isinstance(packet_rows, list) or not all(
        isinstance(row, Mapping) for row in packet_rows
    ):
        return None
    if any(_packet_contract_error(row, "packet") is not None for row in packet_rows):
        return None
    if not _ids_unique(packet_rows, "packet_id"):
        return None
    expected_packet_order = sorted(
        packet_rows,
        key=lambda row: (
            row["created_slot"],
            ROBOT_RANK[row["source"]],
            row["packet_id"],
        ),
    )
    if packet_rows != expected_packet_order:
        return None
    decisions: dict[int, Mapping[str, Any]] = {}
    for slot in range(HORIZON):
        slot_record = slots[slot]
        if not isinstance(slot_record, Mapping):
            return None
        requests = slot_record.get("requests")
        if (
            not isinstance(requests, list)
            or len(requests) != len(ROBOT_IDS)
            or not all(_request_valid(request) for request in requests)
            or [request["robot"] for request in requests] != list(ROBOT_IDS)
        ):
            return None
        creations = [
            row
            for row in packet_rows
            if _integer(row.get("created_slot")) and row.get("created_slot") == slot
        ]
        creations.sort(
            key=lambda row: (
                ROBOT_RANK.get(row.get("source"), len(ROBOT_IDS))
                if isinstance(row.get("source"), str)
                else len(ROBOT_IDS),
                str(row.get("packet_id")),
            )
        )
        decisions[slot] = {"creations": creations, "requests": requests}
    return decisions


def _legality_atoms(scenarios: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    rules = [name.removeprefix("decision_legality.") for name in ATOM_SHARES["decision_legality"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(scenarios):
        decisions = _decision_map(scenarios[name])
        if decisions is None:
            continue
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception as exc:  # malformed evidence remains a deterministic failed decision trace
            fail("request_domains", f"{name}: decision replay raised {type(exc).__name__}: {exc}")
            continue
        for rule in rules:
            ok, detail = replay["legality"][rule]
            if not ok:
                fail(rule, str(detail))
    return [
        _atom(
            "decision_legality",
            rule,
            passed[rule],
            details[rule] or "all decisions satisfy this rule",
        )
        for rule in rules
    ]


def _compare(actual: Any, expected: Any, path: str) -> tuple[bool, str]:
    if isinstance(expected, bool):
        ok = isinstance(actual, bool) and actual is expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if _finite_number(expected):
        ok = _close(actual, expected)
        return (
            ok,
            "" if ok else f"{path}: expected={expected!r}; actual={actual!r}; tolerance={EPS}",
        )
    if isinstance(expected, str) or expected is None:
        ok = actual == expected
        return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            actual_length = len(actual) if isinstance(actual, list) else None
            return (
                False,
                f"{path}: expected list length={len(expected)}; "
                f"actual_type={type(actual).__name__}; actual_length={actual_length}",
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            ok, detail = _compare(actual_item, expected_item, f"{path}[{index}]")
            if not ok:
                return False, detail
        return True, ""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            actual_keys = sorted(actual) if isinstance(actual, dict) else None
            return False, f"{path}: expected keys={sorted(expected)}; actual_keys={actual_keys}"
        for key in expected:
            ok, detail = _compare(actual[key], expected[key], f"{path}.{key}")
            if not ok:
                return False, detail
        return True, ""
    ok = actual == expected
    return ok, "" if ok else f"{path}: expected={expected!r}; actual={actual!r}"


def _robot_motion_component_valid(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and isinstance(row.get("robot"), str)
        and isinstance(row.get("positions"), list)
        and len(row["positions"]) == HORIZON + 1
        and all(_bounded_position(position) for position in row["positions"])
        and isinstance(row.get("energy"), list)
        and len(row["energy"]) == HORIZON + 1
        and all(
            _finite_number(energy) and -EPS <= float(energy) <= INITIAL_ENERGY + EPS
            for energy in row["energy"]
        )
    )


def _robot_versions_component_valid(row: Any) -> bool:
    tracks = row.get("known_unit_versions") if isinstance(row, Mapping) else None
    return (
        isinstance(tracks, list)
        and len(tracks) == HORIZON
        and all(
            isinstance(versions, list)
            and len(versions) == len(UNIT_IDS)
            and all(_integer(version) and -1 <= version <= 1 for version in versions)
            for versions in tracks
        )
    )


def _robot_queues_component_valid(row: Any) -> bool:
    lengths = row.get("queue_lengths") if isinstance(row, Mapping) else None
    return (
        isinstance(lengths, list)
        and len(lengths) == HORIZON
        and all(_integer(length) and 0 <= length <= QUEUE_CAPACITY for length in lengths)
    )


def _slot_collisions_component_valid(row: Any) -> bool:
    state = row.get("state") if isinstance(row, Mapping) else None
    collisions = state.get("collision_robots") if isinstance(state, Mapping) else None
    return _canonical_subset(collisions, ROBOT_IDS)


def _slot_contacts_component_valid(row: Any) -> bool:
    state = row.get("state") if isinstance(row, Mapping) else None
    if not isinstance(state, Mapping):
        return False
    period = state.get("period")
    return (
        _integer(period)
        and 0 <= period < HORIZON // PERIOD
        and _canonical_subset(state.get("contact_pairs"), PAIR_IDS)
    )


def _slot_service_component_valid(row: Any, packet_ids: set[str]) -> bool:
    if not isinstance(row, Mapping):
        return False
    served = row.get("served_packet_ids")
    state = row.get("state")
    return (
        isinstance(served, list)
        and len(served) <= 2
        and all(isinstance(packet_id, str) and packet_id in packet_ids for packet_id in served)
        and isinstance(state, Mapping)
        and _integer(state.get("send_attempts"))
        and state["send_attempts"] >= 0
    )


def _slot_state_value(row: Any, key: str) -> Any:
    state = row.get("state") if isinstance(row, Mapping) else None
    return state.get(key) if isinstance(state, Mapping) else None


def _observations_component_valid(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    for index, row in enumerate(rows):
        if _observation_error(row, f"observations[{index}]") is not None:
            return False
        if row["robot"] not in ROBOT_IDS or row["unit_id"] not in UNIT_IDS:
            return False
        if row["observation_id"] != f"obs-{row['slot']:03d}-{row['robot']}-{row['unit_id']}":
            return False
    if not _ids_unique(rows, "observation_id"):
        return False
    expected = sorted(
        rows,
        key=lambda row: (
            row["slot"],
            ROBOT_RANK[row["robot"]],
            UNIT_IDS.index(row["unit_id"]),
        ),
    )
    return rows == expected


def _deliveries_component_valid(rows: Any, packets: Mapping[str, Any]) -> bool:
    if not isinstance(rows, list):
        return False
    for index, row in enumerate(rows):
        if _delivery_error(row, f"deliveries[{index}]") is not None:
            return False
        packet = packets.get(row["packet_id"])
        if packet is None or row["delivery_id"] != f"del-{row['packet_id']}":
            return False
        if any(row[key] != packet.get(key) for key in ("source", "destination", "unit_versions")):
            return False
    if not _ids_unique(rows, "delivery_id"):
        return False
    return rows == sorted(rows, key=lambda row: (row["slot"], row["delivery_id"]))


def _replay_atoms(
    scenarios: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    rules = [name.removeprefix("replay_validity.") for name in ATOM_SHARES["replay_validity"]]
    passed = {rule: True for rule in rules}
    details = {rule: "" for rule in rules}
    replayed: dict[str, dict[str, Any]] = {}

    def fail(rule: str, detail: str) -> None:
        if passed[rule]:
            passed[rule] = False
            details[rule] = detail

    for name in _known_scenarios(scenarios):
        item = scenarios[name]
        decisions = _decision_map(item)
        if decisions is None:
            continue
        try:
            replay = _simulate_scenario(name, decisions=decisions)
        except Exception:
            # Parsing and decision-domain checks own malformed traces. Replaying them again
            # would turn one defect into several unrelated state mismatches.
            continue
        replayed[name] = replay
        actual = item["record"]
        expected = replay["record"]
        actual_slots = [item["slots"][slot] for slot in range(HORIZON)]
        expected_slots = expected["slots"]
        robot_keys = [robot for robot in ROBOT_IDS if robot in item["robots"]]
        actual_robots = [item["robots"][robot] for robot in robot_keys]
        expected_robot_index = {row["robot"]: row for row in expected["robots"]}
        expected_robots = [expected_robot_index[robot] for robot in robot_keys]
        unit_keys = [unit_id for unit_id in UNIT_IDS if unit_id in item["units"]]
        actual_units = [item["units"][unit_id] for unit_id in unit_keys]
        expected_unit_index = {row["unit_id"]: row for row in expected["units"]}
        expected_units = [expected_unit_index[unit_id] for unit_id in unit_keys]
        pair_keys = [
            (period, pair)
            for period in range(HORIZON // PERIOD)
            for pair in PAIR_IDS
            if (period, pair) in item["pair_periods"]
        ]
        actual_pair_periods = [item["pair_periods"][key] for key in pair_keys]
        expected_pair_index = {
            (row["period"], row["pair"]): row for row in expected["pair_periods"]
        }
        expected_pair_periods = [expected_pair_index[key] for key in pair_keys]

        groups: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            rule: ({}, {}) for rule in rules
        }

        def add_component(
            rule: str, key: str, actual_value: Any, expected_value: Any, valid: bool
        ) -> None:
            if valid:
                groups[rule][0][key] = actual_value
                groups[rule][1][key] = expected_value

        robot_motion_valid = all(_robot_motion_component_valid(row) for row in actual_robots)
        add_component(
            "robot_motion",
            "robots",
            [
                {
                    "robot": row.get("robot"),
                    "positions": row.get("positions"),
                    "energy": row.get("energy"),
                }
                for row in actual_robots
            ],
            [
                {"robot": row["robot"], "positions": row["positions"], "energy": row["energy"]}
                for row in expected_robots
            ],
            robot_motion_valid,
        )
        slot_collisions_valid = all(_slot_collisions_component_valid(row) for row in actual_slots)
        add_component(
            "robot_motion",
            "collisions",
            [_slot_state_value(row, "collision_robots") for row in actual_slots],
            [row["state"]["collision_robots"] for row in expected_slots],
            slot_collisions_valid,
        )

        observations = actual.get("observations")
        add_component(
            "observations_units",
            "observations",
            observations,
            expected["observations"],
            _observations_component_valid(observations),
        )
        units_valid = all(
            _unit_error(row, f"units[{index}]") is None for index, row in enumerate(actual_units)
        )
        add_component("observations_units", "units", actual_units, expected_units, units_valid)

        slot_contacts_valid = all(_slot_contacts_component_valid(row) for row in actual_slots)
        add_component(
            "contact_pair_periods",
            "slot_contacts",
            [
                {
                    "period": _slot_state_value(row, "period"),
                    "contact_pairs": _slot_state_value(row, "contact_pairs"),
                }
                for row in actual_slots
            ],
            [
                {
                    "period": row["state"]["period"],
                    "contact_pairs": row["state"]["contact_pairs"],
                }
                for row in expected_slots
            ],
            slot_contacts_valid,
        )
        delivery_ids = set(item["deliveries"])
        pair_periods_valid = all(
            _pair_period_error(row, f"pair_periods[{index}]") is None
            and (
                row["successful_delivery_id"] is None
                or row["successful_delivery_id"] in delivery_ids
            )
            for index, row in enumerate(actual_pair_periods)
        )
        add_component(
            "contact_pair_periods",
            "pair_periods",
            actual_pair_periods,
            expected_pair_periods,
            pair_periods_valid,
        )

        add_component("packet_attempts", "packets", actual["packets"], expected["packets"], True)
        packet_ids = set(item["packets"])
        slot_service_valid = all(
            _slot_service_component_valid(row, packet_ids) for row in actual_slots
        )
        add_component(
            "packet_attempts",
            "slot_service",
            [
                {
                    "served_packet_ids": row.get("served_packet_ids")
                    if isinstance(row, Mapping)
                    else None,
                    "send_attempts": _slot_state_value(row, "send_attempts"),
                }
                for row in actual_slots
            ],
            [
                {
                    "served_packet_ids": row["served_packet_ids"],
                    "send_attempts": row["state"]["send_attempts"],
                }
                for row in expected_slots
            ],
            slot_service_valid,
        )

        deliveries = actual.get("deliveries")
        add_component(
            "delivery_fusion",
            "deliveries",
            deliveries,
            expected["deliveries"],
            _deliveries_component_valid(deliveries, item["packets"]),
        )
        versions_valid = all(_robot_versions_component_valid(row) for row in actual_robots)
        add_component(
            "delivery_fusion",
            "known_unit_versions",
            [row.get("known_unit_versions") for row in actual_robots],
            [row["known_unit_versions"] for row in expected_robots],
            versions_valid,
        )

        queues_valid = all(_robot_queues_component_valid(row) for row in actual_robots)
        add_component(
            "queues_mission",
            "queue_lengths",
            [row.get("queue_lengths") for row in actual_robots],
            [row["queue_lengths"] for row in expected_robots],
            queues_valid,
        )
        mission = actual.get("mission")
        add_component(
            "queues_mission",
            "mission",
            mission,
            expected["mission"],
            _mission_error(mission, "mission") is None,
        )
        legal = {rule: bool(value[0]) for rule, value in replay["legality"].items()}
        dependencies = {
            "robot_motion": ("request_domains", "motion"),
            "observations_units": ("request_domains", "motion"),
            "contact_pair_periods": (
                "request_domains",
                "motion",
                "packet_creation",
                "local_content",
                "fifo_contact",
                "attempt_limits",
            ),
            "packet_attempts": (
                "request_domains",
                "packet_creation",
                "local_content",
                "fifo_contact",
                "attempt_limits",
            ),
            "delivery_fusion": tuple(legal),
            "queues_mission": tuple(legal),
        }
        for rule, (actual_group, expected_group) in groups.items():
            if any(not legal.get(dependency, False) for dependency in dependencies[rule]):
                continue
            ok, detail = _compare(actual_group, expected_group, f"{name}.{rule}")
            if not ok:
                fail(rule, detail)

    atoms = [
        _atom(
            "replay_validity",
            rule,
            passed[rule],
            details[rule] or "fixed-model evidence comparison passed",
        )
        for rule in rules
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    if set(replayed) != set(SCENARIO_NAMES):
        raise ValueError("all canonical scenario replays are required for metrics")
    totals = {
        "goal_coverage": 0.0,
        "periodic_fusion": 0.0,
        "safe_motion_efficiency": 0.0,
        "summary_freshness_efficiency": 0.0,
        "contact_load_efficiency": 0.0,
    }
    total_value = sum(float(value) for _, value in UNIT_BASE)
    grid = [(x, y) for x in range(18) for y in range(18)]
    for name in SCENARIO_NAMES:
        replay = replayed[name]
        record = replay["record"]
        mission = record["mission"]
        units = _unit_models(name)

        counts = Counter(replay["movement_end_positions"])
        information_mass = {
            units[unit_id]["cell"]: float(units[unit_id]["value"]) / total_value
            for unit_id in UNIT_IDS
        }
        l1_distance = sum(
            abs(counts.get(cell, 0) / (len(ROBOT_IDS) * HORIZON) - information_mass.get(cell, 0.0))
            for cell in grid
        )
        ergodic_score = max(0.0, min(1.0, 1.0 - 0.5 * l1_distance))
        covered_value = sum(
            float(units[unit_id]["value"]) for unit_id in replay["physical_observed_units"]
        )
        totals["goal_coverage"] += 0.6 * ergodic_score + 0.4 * covered_value / total_value

        satisfied = sum(int(row["satisfied"]) for row in record["pair_periods"])
        totals["periodic_fusion"] += (
            0.6 * satisfied / len(record["pair_periods"])
            + 0.4 * float(mission["timely_fused_value"]) / total_value
        )

        collision_score = max(
            0.0,
            min(1.0, 1.0 - int(mission["collision_events"]) / (len(ROBOT_IDS) * HORIZON)),
        )
        motion_score = max(
            0.0,
            min(1.0, 1.0 - int(mission["motion_cost"]) / (len(ROBOT_IDS) * HORIZON)),
        )
        energy_score = max(0.0, min(1.0, 1.0 - float(mission["energy_used"]) / 360.0))
        totals["safe_motion_efficiency"] += (
            0.4 * collision_score + 0.3 * motion_score + 0.3 * energy_score
        )

        sent_units = int(mission["sent_units"])
        useful_units = int(mission["useful_summary_units"])
        fusion_ages = replay["fusion_ages"]
        if sent_units > 0 and fusion_ages:
            mean_age = sum(float(age) for age in fusion_ages) / len(fusion_ages)
            freshness = 0.6 * max(0.0, min(1.0, 1.0 - mean_age / PERIOD))
            freshness += 0.4 * useful_units / sent_units
        else:
            freshness = 0.0
        totals["summary_freshness_efficiency"] += max(0.0, min(1.0, freshness))

        if sent_units > 0 and useful_units > 0:
            load = 0.5 * max(
                0.0,
                min(1.0, 1.0 - float(mission["contact_wait_slots"]) / (PERIOD * 3)),
            )
            load += 0.25 * max(
                0.0,
                min(1.0, 1.0 - int(mission["retransmitted_units"]) / sent_units),
            )
            load += 0.25 * satisfied / len(record["pair_periods"])
        else:
            load = 0.0
        totals["contact_load_efficiency"] += max(0.0, min(1.0, load))

    metrics = {
        metric: max(0.0, min(1.0, total / len(SCENARIO_NAMES))) for metric, total in totals.items()
    }
    if not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
        raise ValueError("task metrics must be finite values in [0,1]")
    return metrics


def _gated_metric(metrics: Mapping[str, float] | None, name: str) -> float:
    if metrics is None:
        return 0.0
    value = metrics.get(name, 0.0)
    return float(value) if _finite_number(value) and 0.0 <= float(value) <= 1.0 else 0.0


def _headings(text: str) -> dict[str, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    lines = text.splitlines()
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match or match.group(2).strip() not in required:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            next_heading = re.match(r"^(#{1,6})\s+.+", lines[cursor])
            if next_heading and len(next_heading.group(1)) <= level:
                break
            body.append(lines[cursor])
            cursor += 1
        found.setdefault(title, "\n".join(body).strip())
    return found


def _report_score(text: str | None) -> tuple[float, str]:
    required = (
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    )
    found = _headings(text) if text and text.strip() else {}
    missing = [title for title in required if not found.get(title, "").strip()]
    score = max(0.0, 1.0 - len(missing) / 4.0)
    return (
        score,
        f"missing_sections={missing}; normalized={score:.9g}" if missing else "missing_sections=[]",
    )


def _regular_file(ws: Workspace, rel: str) -> bool:
    path = Path(ws.root) / rel
    return path.is_file() and not path.is_symlink()


async def _build_state(ws: Workspace) -> EvalState:
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    state = EvalState()
    remove_ok = True
    try:
        state.run_started = True
        try:
            remove_result = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(remove_result, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001 - retain failure in the cached state
            remove_ok = False
            state.official_message = f"could not remove previous results.json: {exc}"
        started = time.monotonic()
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr = str(getattr(result, "stderr", ""))
            state.timed_out = state.exit_code is None and "timeout" in stderr.lower()
        except Exception as exc:  # noqa: BLE001 - every execution path is cached
            state.exit_code = None
            state.timed_out = "timeout" in str(exc).lower()
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = (
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        if not state.official_message:
            run_summary = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"new_regular_results={state.result_file_is_new_regular}"
            )
            failure_summary = (
                f"official run failed: exit={state.exit_code}, elapsed={state.elapsed_s:.2f}s, "
                f"timed_out={state.timed_out}, "
                f"new_regular_results={state.result_file_is_new_regular}"
            )
            state.official_message = run_summary if state.official_ok else failure_summary
        if state.result_file_is_new_regular:
            try:
                state.data = _strict_json(ws.read_text("results.json") or "")
                state.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                state.diagnostics["result_json"] = f"results.json: strict parse failed: {exc}"

        result_atoms = _result_json_atoms(state.data, state.parse_ok)
        state.scores["result_json"] = _score_atoms(result_atoms, ATOM_SHARES["result_json"])
        state.diagnostics["result_json"] = state.diagnostics.get(
            "result_json", _diagnose_atoms(result_atoms)
        )
        if not state.official_ok or not state.parse_ok:
            reason = "official run or strict result parsing failed"
            state.scores["result_json"] = 0.0
            state.diagnostics["result_json"] = f"{reason}; {state.diagnostics['result_json']}"
            for check_id in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = reason
            _CACHE[key] = state
            return state

        state.scenarios = _index_records(state.data)
        state.records = state.scenarios
        evidence_atoms = _evidence_atoms(state.scenarios)
        state.scores["result_evidence"] = _score_atoms(
            evidence_atoms, ATOM_SHARES["result_evidence"]
        )
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence_atoms)

        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = _coverage_diagnostic(coverage, defects)

        legality_atoms = _legality_atoms(state.scenarios)
        state.scores["decision_legality"] = _score_atoms(
            legality_atoms, ATOM_SHARES["decision_legality"]
        )
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality_atoms)

        replay_atoms, replayed = _replay_atoms(state.scenarios)
        state.replayed = replayed
        state.scores["replay_validity"] = _score_atoms(replay_atoms, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay_atoms)

        gate_names = (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        )
        state.validity_gate = state.official_ok and all(
            state.scores.get(check_id, 0.0) >= 1.0 - EPS for check_id in gate_names
        )
        if state.validity_gate:
            state.metrics = _metrics(state.replayed)
    except Exception as exc:  # noqa: BLE001 - cache a deterministic failed state
        analysis_message = f"grader analysis failed: {type(exc).__name__}: {exc}"
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, analysis_message)
    finally:
        _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected a regular file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    if getattr(result, "exit_code", None) == 0:
        return 1.0, "run.py: static compilation passed"
    return 0.0, "run.py: static compilation failed"


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0, state.official_message) if state.official_ok else (0.0, state.official_message)


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a readable nonempty regular file"
    try:
        text = ws.read_text("report.md")
    except Exception as exc:  # noqa: BLE001 - report failure is a deterministic check result
        return 0.0, f"report.md: read failed: {type(exc).__name__}: {exc}"
    return _report_score(text)


@check(id="result_json")
async def result_json(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_json", 0.0), state.diagnostics.get("result_json", "")


@check(id="result_evidence")
async def result_evidence(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("result_evidence", 0.0), state.diagnostics.get("result_evidence", "")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("canonical_coverage", 0.0), state.diagnostics.get(
        "canonical_coverage", ""
    )


@check(id="decision_legality")
async def decision_legality(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("decision_legality", 0.0), state.diagnostics.get(
        "decision_legality", ""
    )


@check(id="replay_validity")
async def replay_validity(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return state.scores.get("replay_validity", 0.0), state.diagnostics.get("replay_validity", "")


async def _metric_check(ws: Workspace, metric: str) -> tuple[float, str]:
    state = await _build_state(ws)
    if state.metrics is None:
        return 0.0, f"{metric}: validity gate not satisfied; metric not computed"
    value = _gated_metric(state.metrics, metric)
    return value, f"{metric}: independently replayed value={value:.9g}"


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "goal_coverage")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "periodic_fusion")


@check(id="primary_metrics_3")
async def primary_metrics_3(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "safe_motion_efficiency")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "summary_freshness_efficiency")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "contact_load_efficiency")
