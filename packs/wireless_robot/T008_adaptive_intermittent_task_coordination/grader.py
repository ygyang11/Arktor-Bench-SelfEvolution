from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from arktor_bench.grading import check
from arktor_bench.sandbox.workspace import Workspace

GRID_W = 14
GRID_H = 10
HORIZON = 36
N_ROBOTS = 4
N_TASKS = 10
MOVE_BUDGET = 28
SENSE_RANGE = 2
RADIO_RANGE = 5
EVENT_BUDGET = 6
MIN_EVENT_LAG = 2
MAX_EVENT_LAG = 7
MAX_ATTEMPT_T = 34
CONCURRENT_GROUP_SIZE = 2
EPS = 1e-6
SCENARIO_NAMES = (
    "sparse_sequence",
    "precedence_chain",
    "parallel_team",
    "connectivity_bottleneck",
    "bursty_arrivals",
)
MOVES = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0), "stay": (0, 0)}
ACTIONS = frozenset({"move", "work", "wait"})
FACT_STATES = frozenset({"available", "active", "completed", "expired"})
TASK_STATES = frozenset({"unreleased", *FACT_STATES})
ACTION_RESULTS = frozenset({"none", "moved", "working", "completed", "invalid"})
EVENT_STATUSES = frozenset({"success", "missed", "disconnected"})

STARTS: dict[str, tuple[tuple[int, int], ...]] = {
    "sparse_sequence": ((0, 0), (13, 0), (0, 9), (13, 9)),
    "precedence_chain": ((1, 1), (12, 1), (1, 8), (12, 8)),
    "parallel_team": ((1, 2), (12, 2), (1, 7), (12, 7)),
    "connectivity_bottleneck": ((1, 1), (3, 8), (10, 1), (12, 8)),
    "bursty_arrivals": ((1, 1), (12, 1), (1, 8), (12, 8)),
}

OBSTACLES: dict[str, frozenset[tuple[int, int]]] = {
    "sparse_sequence": frozenset({(6, 3), (6, 4), (7, 4), (7, 5)}),
    "precedence_chain": frozenset({(6, 0), (6, 1), (6, 3), (6, 4), (6, 5), (6, 6), (6, 8), (6, 9)}),
    "parallel_team": frozenset({(2, 4), (3, 4), (5, 4), (6, 4), (7, 4), (8, 4), (10, 4), (11, 4)}),
    "connectivity_bottleneck": frozenset(
        {
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 7),
            (9, 8),
            (9, 9),
        }
    ),
    "bursty_arrivals": frozenset({(5, 2), (5, 3), (5, 6), (5, 7), (8, 2), (8, 3), (8, 6), (8, 7)}),
}


def _task(
    position: tuple[int, int],
    release_t: int,
    deadline_t: int,
    duration: int,
    team_size: int,
    predecessors: tuple[int, ...] = (),
    mutex_group: str | None = None,
    concurrent_group: str | None = None,
) -> dict[str, Any]:
    return {
        "position": position,
        "release_t": release_t,
        "deadline_t": deadline_t,
        "duration": duration,
        "team_size": team_size,
        "predecessors": predecessors,
        "mutex_group": mutex_group,
        "concurrent_group": concurrent_group,
    }


TASKS: dict[str, tuple[dict[str, Any], ...]] = {
    "sparse_sequence": (
        _task((2, 1), 0, 9, 2, 1),
        _task((11, 1), 1, 11, 2, 1),
        _task((2, 8), 4, 17, 2, 2),
        _task((11, 8), 6, 18, 2, 1, (0,)),
        _task((5, 2), 9, 22, 2, 1, (1,), "m0"),
        _task((8, 7), 10, 24, 2, 2, (), "m0"),
        _task((1, 5), 14, 28, 1, 1, (2,)),
        _task((12, 4), 15, 29, 1, 1, (3,)),
        _task((5, 7), 19, 33, 1, 1, (), None, "c0"),
        _task((8, 2), 19, 33, 1, 1, (), None, "c0"),
    ),
    "precedence_chain": (
        _task((2, 2), 0, 10, 2, 1),
        _task((11, 2), 0, 11, 2, 1),
        _task((4, 7), 3, 18, 2, 2, (0,)),
        _task((9, 7), 4, 19, 2, 1, (1,)),
        _task((2, 7), 7, 22, 1, 1, (0, 1)),
        _task((6, 2), 9, 26, 2, 2, (2,)),
        _task((11, 7), 11, 27, 1, 1, (3,), "m1"),
        _task((4, 2), 14, 30, 2, 1, (4,), "m1"),
        _task((5, 7), 18, 34, 1, 1, (5,), None, "c1"),
        _task((8, 7), 18, 34, 1, 1, (6,), None, "c1"),
    ),
    "parallel_team": (
        _task((2, 2), 0, 12, 2, 2),
        _task((11, 2), 0, 12, 2, 2),
        _task((2, 7), 3, 16, 1, 1, (), None, "c2"),
        _task((11, 7), 3, 16, 1, 1, (), None, "c2"),
        _task((4, 4), 6, 21, 2, 1, (), "m2"),
        _task((9, 4), 6, 21, 2, 1, (), "m2"),
        _task((4, 7), 10, 26, 2, 2, (2,)),
        _task((9, 7), 10, 26, 2, 2, (3,)),
        _task((4, 2), 16, 32, 1, 1, (4,), None, "c3"),
        _task((9, 2), 16, 32, 1, 1, (5,), None, "c3"),
    ),
    "connectivity_bottleneck": (
        _task((2, 2), 0, 11, 2, 1),
        _task((11, 2), 0, 11, 2, 1),
        _task((6, 2), 3, 17, 2, 2),
        _task((7, 7), 4, 19, 2, 2),
        _task((2, 7), 7, 22, 1, 1, (0,)),
        _task((11, 7), 8, 23, 1, 1, (1,)),
        _task((6, 5), 11, 27, 2, 2, (2,), "m3"),
        _task((7, 4), 12, 28, 2, 2, (3,), "m3"),
        _task((3, 3), 17, 33, 1, 1, (4,), None, "c4"),
        _task((10, 6), 17, 33, 1, 1, (5,), None, "c4"),
    ),
    "bursty_arrivals": (
        _task((2, 2), 0, 11, 2, 1),
        _task((11, 2), 0, 12, 2, 1),
        _task((2, 7), 1, 14, 2, 2),
        _task((11, 7), 1, 15, 2, 2),
        _task((6, 1), 8, 22, 1, 1, (0,), None, "c5"),
        _task((7, 8), 8, 22, 1, 1, (1,), None, "c5"),
        _task((3, 5), 9, 25, 2, 2, (), "m4"),
        _task((10, 4), 9, 25, 2, 1, (), "m4"),
        _task((6, 5), 17, 33, 2, 2, (2,)),
        _task((7, 4), 17, 34, 2, 2, (3,)),
    ),
}

BLACKOUTS: dict[str, dict[int, frozenset[tuple[int, int]]]] = {
    "sparse_sequence": {
        9: frozenset({(0, 2)}),
        10: frozenset({(1, 3)}),
        21: frozenset({(0, 1), (2, 3)}),
    },
    "precedence_chain": {
        8: frozenset({(0, 1)}),
        12: frozenset({(0, 2), (1, 3)}),
        20: frozenset({(0, 1), (2, 3)}),
        28: frozenset({(0, 3)}),
    },
    "parallel_team": {
        7: frozenset({(0, 1), (2, 3)}),
        15: frozenset({(0, 2), (1, 3)}),
        24: frozenset({(0, 3), (1, 2)}),
    },
    "connectivity_bottleneck": {
        6: frozenset({(0, 2), (1, 3)}),
        7: frozenset({(0, 2), (1, 3)}),
        14: frozenset({(0, 1), (2, 3)}),
        15: frozenset({(0, 1), (2, 3)}),
        22: frozenset({(0, 2), (0, 3), (1, 3)}),
        30: frozenset({(1, 2)}),
    },
    "bursty_arrivals": {
        5: frozenset({(0, 1), (2, 3)}),
        10: frozenset({(0, 2), (1, 3)}),
        11: frozenset({(0, 2), (1, 3)}),
        18: frozenset({(0, 3), (1, 2)}),
        26: frozenset({(0, 1)}),
    },
}

ATOM_SHARES: dict[str, dict[str, float]] = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scenarios": 0.33,
    },
    "result_evidence": {
        "result_evidence.scenario_shape": 0.08333333333333333,
        "result_evidence.robot_shape": 0.08333333333333333,
        "result_evidence.robot_fields": 0.08333333333333333,
        "result_evidence.robot_order": 0.08333333333333333,
        "result_evidence.task_shape": 0.08333333333333333,
        "result_evidence.task_fields": 0.08333333333333333,
        "result_evidence.task_order": 0.08333333333333333,
        "result_evidence.timeline_fields": 0.08333333333333333,
        "result_evidence.event_shape": 0.08333333333333333,
        "result_evidence.event_fields": 0.08333333333333333,
        "result_evidence.event_order": 0.08333333333333333,
        "result_evidence.fact_fields": 0.08333333333333333,
    },
    "decision_legality": {
        "decision_legality.action_relations": 0.16666666666666666,
        "decision_legality.motion_bounds": 0.16666666666666666,
        "decision_legality.work_local_state": 0.16666666666666666,
        "decision_legality.event_request_links": 0.16666666666666666,
        "decision_legality.event_resources": 0.16666666666666666,
        "decision_legality.event_positions": 0.16666666666666666,
    },
    "replay_validity": {
        "replay_validity.positions_budget": 0.1,
        "replay_validity.knowledge": 0.1,
        "replay_validity.observations": 0.1,
        "replay_validity.active_events": 0.1,
        "replay_validity.movement_results": 0.1,
        "replay_validity.work_results": 0.1,
        "replay_validity.task_timelines": 0.1,
        "replay_validity.task_records": 0.1,
        "replay_validity.event_basis": 0.1,
        "replay_validity.events": 0.1,
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
        self.records: dict[Hashable, Any] = {}
        self.replayed: dict[Hashable, Any] = {}
        self.scores: dict[str, float] = {}
        self.diagnostics: dict[str, str] = {}
        self.metrics: dict[str, float] | None = None
        self.validity_gate = False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _coordinate(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_integer(v) for v in value)


def _in_grid(position: tuple[int, int]) -> bool:
    return 0 <= position[0] < GRID_W and 0 <= position[1] < GRID_H


def _free(scenario: str, position: tuple[int, int]) -> bool:
    return _in_grid(position) and position not in OBSTACLES[scenario]


def _task_truth(scenario: str, task: int) -> dict[str, Any]:
    return TASKS[scenario][task]


def _l_path(scenario: str, start: tuple[int, int], end: tuple[int, int], x_first: bool) -> bool:
    x, y = start
    cells: list[tuple[int, int]] = []
    axes = ("x", "y") if x_first else ("y", "x")
    for axis in axes:
        if axis == "x":
            while x != end[0]:
                x += 1 if end[0] > x else -1
                cells.append((x, y))
        else:
            while y != end[1]:
                y += 1 if end[1] > y else -1
                cells.append((x, y))
    interior = cells[:-1] if cells and cells[-1] == end else cells
    return not any(cell in OBSTACLES[scenario] for cell in interior)


def _visible(scenario: str, position: tuple[int, int], task: int) -> bool:
    target = _task_truth(scenario, task)["position"]
    distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
    return distance <= SENSE_RANGE and (
        _l_path(scenario, position, target, True) or _l_path(scenario, position, target, False)
    )


def _connected(scenario: str, positions: Sequence[tuple[int, int]], t: int) -> bool:
    graph = {robot: set() for robot in range(N_ROBOTS)}
    blocked = BLACKOUTS[scenario].get(t, frozenset())
    for first in range(N_ROBOTS):
        for second in range(first + 1, N_ROBOTS):
            p, q = positions[first], positions[second]
            distance = abs(p[0] - q[0]) + abs(p[1] - q[1])
            if (
                distance <= RADIO_RANGE
                and (first, second) not in blocked
                and (_l_path(scenario, p, q, True) or _l_path(scenario, p, q, False))
            ):
                graph[first].add(second)
                graph[second].add(first)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in graph[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == N_ROBOTS


def _fact(
    task: int, state: str, progress: int, team: Sequence[int], version: int
) -> dict[str, Any]:
    return {
        "task": task,
        "state": state,
        "progress": progress,
        "team": list(team),
        "version": version,
    }


def _facts_list(cache: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(dict(cache[task])) for task in sorted(cache)]


def _same_fact_content(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(first.get(key) == second.get(key) for key in ("task", "state", "progress", "team"))


def _merged_facts(caches: Sequence[Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[int, tuple[int, dict[str, Any]]] = {}
    for robot, cache in enumerate(caches):
        for task, raw in cache.items():
            value = deepcopy(dict(raw))
            old = merged.get(task)
            if (
                old is None
                or value["version"] > old[1]["version"]
                or (value["version"] == old[1]["version"] and robot < old[0])
            ):
                merged[task] = (robot, value)
    return [merged[task][1] for task in sorted(merged)]


def _common_facts(caches: Sequence[Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    common: list[dict[str, Any]] = []
    for task in sorted(set.intersection(*(set(cache) for cache in caches)) if caches else set()):
        values = [cache[task] for cache in caches]
        if all(dict(value) == dict(values[0]) for value in values[1:]):
            common.append(deepcopy(dict(values[0])))
    return common


def _strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )


class _Model:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.positions = list(STARTS[scenario])
        self.move_used = [0] * N_ROBOTS
        self.local: list[dict[int, dict[str, Any]]] = [dict() for _ in range(N_ROBOTS)]
        self.states = ["unreleased"] * N_TASKS
        self.progress = [0] * N_TASKS
        self.teams: list[tuple[int, ...]] = [tuple() for _ in range(N_TASKS)]
        self.active_start: list[int | None] = [None] * N_TASKS
        self.has_started = [False] * N_TASKS
        self.first_discover: list[int | None] = [None] * N_TASKS
        self.completed_start: list[int | None] = [None] * N_TASKS
        self.finish: list[int | None] = [None] * N_TASKS
        self.completion_team: list[tuple[int, ...]] = [tuple() for _ in range(N_TASKS)]
        self.timelines: list[list[dict[str, Any]]] = [[] for _ in range(N_TASKS)]
        self.robot_rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.active_event: dict[str, Any] | None = None
        self.pending_delivery: dict[int, list[dict[str, Any]]] = {}
        self.synchronized_facts: dict[int, dict[str, Any]] = {}
        self.last_attempt = -100
        self.end_positions: list[dict[int, tuple[int, int]]] = [dict() for _ in range(N_ROBOTS)]
        self._pre: list[dict[str, Any]] = []
        self._common_pre: list[dict[str, Any]] = []

    def begin(self, t: int) -> list[dict[str, Any]]:
        delivered = self.pending_delivery.pop(t, [])
        if delivered:
            for value in delivered:
                self.synchronized_facts[value["task"]] = deepcopy(value)
            for cache in self.local:
                for value in delivered:
                    old = cache.get(value["task"])
                    if old is None or value["version"] >= old["version"]:
                        cache[value["task"]] = deepcopy(value)
        for task in range(N_TASKS):
            truth = _task_truth(self.scenario, task)
            if self.states[task] == "unreleased" and t >= truth["release_t"]:
                self.states[task] = "available"
        active_id = self.active_event["event_id"] if self.active_event is not None else None
        active_plan = None
        if self.active_event is not None:
            active_plan = {
                key: deepcopy(self.active_event[key])
                for key in (
                    "event_id",
                    "propose_t",
                    "requesters",
                    "attempt_t",
                    "meeting_positions",
                )
            }
        self._common_pre = _common_facts(self.local)
        self._pre = []
        for robot in range(N_ROBOTS):
            self._pre.append(
                {
                    "robot": robot,
                    "t": t,
                    "position": list(self.positions[robot]),
                    "all_positions": [list(value) for value in self.positions],
                    "move_steps_used": self.move_used[robot],
                    "known_tasks": _facts_list(self.local[robot]),
                    "active_event": active_id,
                    "active_plan": active_plan,
                }
            )
        return deepcopy(self._pre)

    def policy_public_state(self, t: int) -> dict[str, Any]:
        completed_events = []
        for event in self.events:
            if event["attempt_t"] >= t:
                continue
            completed_events.append(
                {
                    key: deepcopy(event[key])
                    for key in (
                        "event_id",
                        "propose_t",
                        "requesters",
                        "attempt_t",
                        "meeting_positions",
                        "arrival_t",
                        "connected",
                        "confirm_t",
                        "status",
                    )
                }
            )
        return {
            "positions": [list(position) for position in self.positions],
            "move_steps_used": list(self.move_used),
            "active_plan": deepcopy(self._pre[0]["active_plan"]),
            "event_count": len(self.events),
            "last_attempt_t": None if self.last_attempt < 0 else self.last_attempt,
            "event_history": completed_events,
            "synchronized_facts": _facts_list(self.synchronized_facts),
        }

    def _write_local(
        self, robot: int, task: int, t: int, observed: dict[int, dict[str, Any]]
    ) -> None:
        value = _fact(task, self.states[task], self.progress[task], self.teams[task], t)
        old = self.local[robot].get(task)
        if old is None or not _same_fact_content(old, value):
            self.local[robot][task] = value
            observed[task] = deepcopy(value)

    def _proposal_valid(self, t: int, requesters: list[int], proposal: Any) -> bool:
        if not requesters or self.active_event is not None or len(self.events) >= EVENT_BUDGET:
            return False
        if self.last_attempt > -100 and t - self.last_attempt < 2:
            return False
        if not isinstance(proposal, Mapping):
            return False
        attempt = proposal.get("attempt_t")
        meetings = proposal.get("meeting_positions")
        if not _integer(attempt) or not (
            t + MIN_EVENT_LAG <= attempt <= min(t + MAX_EVENT_LAG, MAX_ATTEMPT_T)
        ):
            return False
        if (
            not isinstance(meetings, list)
            or len(meetings) != N_ROBOTS
            or not all(_coordinate(item) for item in meetings)
        ):
            return False
        return all(_free(self.scenario, tuple(item)) for item in meetings)

    def resolve(
        self, t: int, decisions: Sequence[Mapping[str, Any]], proposal: Any = None
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for robot in range(N_ROBOTS):
            raw = (
                decisions[robot]
                if robot < len(decisions) and isinstance(decisions[robot], Mapping)
                else {}
            )
            normalized.append(
                {
                    "action": raw.get("action", "wait"),
                    "move": raw.get("move", "stay"),
                    "task": raw.get("task"),
                    "event_request": raw.get("event_request", False),
                }
            )
        start_positions = list(self.positions)
        results = [{"status": "none", "task": None} for _ in range(N_ROBOTS)]

        for robot, choice in enumerate(normalized):
            if choice["action"] != "move":
                continue
            move = choice["move"]
            if not isinstance(move, str) or move not in MOVES or choice["task"] is not None:
                results[robot] = {"status": "invalid", "task": None}
                continue
            dx, dy = MOVES[move]
            target = (start_positions[robot][0] + dx, start_positions[robot][1] + dy)
            if move == "stay":
                continue
            if self.move_used[robot] >= MOVE_BUDGET or not _free(self.scenario, target):
                results[robot] = {"status": "invalid", "task": None}
                continue
            self.positions[robot] = target
            self.move_used[robot] += 1
            results[robot] = {"status": "moved", "task": None}

        state_start = list(self.states)
        progress_start = list(self.progress)
        team_start = list(self.teams)
        workers: dict[int, list[int]] = defaultdict(list)
        for robot, choice in enumerate(normalized):
            if choice["action"] != "work":
                if choice["action"] == "wait" and (
                    choice["move"] != "stay" or choice["task"] is not None
                ):
                    results[robot] = {"status": "invalid", "task": None}
                continue
            task = choice["task"]
            results[robot] = {"status": "invalid", "task": task if _integer(task) else None}
            if choice["move"] != "stay" or not _integer(task) or not (0 <= task < N_TASKS):
                continue
            truth = _task_truth(self.scenario, task)
            fact = self.local[robot].get(task)
            if (
                fact is None
                or t > truth["deadline_t"]
                or start_positions[robot] != truth["position"]
            ):
                continue
            if fact["state"] in {"completed", "expired"}:
                continue
            workers[task].append(robot)

        can_progress = [False] * N_TASKS
        for task in range(N_TASKS):
            truth = _task_truth(self.scenario, task)
            assigned = tuple(sorted(workers.get(task, [])))
            if state_start[task] == "active":
                can_progress[task] = assigned == team_start[task]
            elif state_start[task] == "available":
                predecessors_done = all(
                    state_start[pred] == "completed" for pred in truth["predecessors"]
                )
                can_progress[task] = len(assigned) == truth["team_size"] and predecessors_done

        concurrent: dict[str, list[int]] = defaultdict(list)
        for task in range(N_TASKS):
            group = _task_truth(self.scenario, task)["concurrent_group"]
            if group is not None:
                concurrent[group].append(task)
        for group_tasks in concurrent.values():
            starting = [
                task
                for task in group_tasks
                if state_start[task] == "available" and not self.has_started[task]
            ]
            if starting and not all(
                can_progress[task]
                and state_start[task] == "available"
                and not self.has_started[task]
                for task in group_tasks
            ):
                for task in starting:
                    can_progress[task] = False

        mutex: dict[str, list[int]] = defaultdict(list)
        for task in range(N_TASKS):
            group = _task_truth(self.scenario, task)["mutex_group"]
            if group is not None and can_progress[task]:
                mutex[group].append(task)
        for group_tasks in mutex.values():
            if len(group_tasks) > 1:
                for task in group_tasks:
                    can_progress[task] = False

        for task in range(N_TASKS):
            truth = _task_truth(self.scenario, task)
            assigned = tuple(sorted(workers.get(task, [])))
            if state_start[task] == "active" and not can_progress[task]:
                self.states[task] = "available"
                self.progress[task] = 0
                self.teams[task] = tuple()
                self.active_start[task] = None
            if not can_progress[task]:
                continue
            if state_start[task] == "available":
                self.states[task] = "active"
                self.progress[task] = 1
                self.teams[task] = assigned
                self.active_start[task] = t
                self.has_started[task] = True
            else:
                self.progress[task] = progress_start[task] + 1
            completed = self.progress[task] >= truth["duration"]
            if completed:
                self.states[task] = "completed"
                self.progress[task] = truth["duration"]
                self.finish[task] = t
                self.completed_start[task] = self.active_start[task]
                self.completion_team[task] = self.teams[task]
            for robot in assigned:
                results[robot] = {"status": "completed" if completed else "working", "task": task}

        for task in range(N_TASKS):
            truth = _task_truth(self.scenario, task)
            if self.states[task] in {"available", "active"} and t >= truth["deadline_t"]:
                self.states[task] = "expired"
                self.progress[task] = 0
                self.teams[task] = tuple()
                self.active_start[task] = None

        observed: list[dict[int, dict[str, Any]]] = [dict() for _ in range(N_ROBOTS)]
        for robot in range(N_ROBOTS):
            for task in range(N_TASKS):
                truth = _task_truth(self.scenario, task)
                if t >= truth["release_t"] and _visible(self.scenario, self.positions[robot], task):
                    if self.first_discover[task] is None:
                        self.first_discover[task] = t
                    self._write_local(robot, task, t, observed[robot])

        for robot in range(N_ROBOTS):
            self.end_positions[robot][t] = self.positions[robot]

        if self.active_event is not None and self.active_event["attempt_t"] == t:
            event = self.active_event
            actual = list(self.positions)
            planned = [tuple(value) for value in event["meeting_positions"]]
            met = actual == planned
            connected = _connected(self.scenario, actual, t)
            status = "missed" if not met else ("success" if connected else "disconnected")
            arrivals: list[int | None] = []
            for robot in range(N_ROBOTS):
                arrivals.append(
                    next(
                        (
                            slot
                            for slot in range(event["propose_t"] + 1, t + 1)
                            if self.end_positions[robot].get(slot) == planned[robot]
                        ),
                        None,
                    )
                )
            event["arrival_t"] = arrivals
            event["connected"] = connected
            event["status"] = status
            event["confirm_t"] = t + 1 if status == "success" else None
            event["shared_facts"] = _merged_facts(self.local) if status == "success" else []
            if status == "success":
                self.pending_delivery[t + 1] = deepcopy(event["shared_facts"])
            self.last_attempt = t
            self.active_event = None

        requesters = [
            robot for robot, choice in enumerate(normalized) if choice["event_request"] is True
        ]
        if self._proposal_valid(t, requesters, proposal):
            event = {
                "event_id": f"e{len(self.events)}",
                "propose_t": t,
                "requesters": requesters,
                "attempt_t": proposal["attempt_t"],
                "meeting_positions": [list(value) for value in proposal["meeting_positions"]],
                "arrival_t": [None] * N_ROBOTS,
                "basis_facts": deepcopy(self._common_pre),
                "connected": False,
                "confirm_t": None,
                "shared_facts": [],
                "status": "missed",
            }
            self.events.append(event)
            self.active_event = event

        rows: list[dict[str, Any]] = []
        for robot, choice in enumerate(normalized):
            row = {
                "robot": robot,
                "t": t,
                "position": self._pre[robot]["position"],
                "move_steps_used": self._pre[robot]["move_steps_used"],
                "action": choice["action"],
                "move": choice["move"],
                "task": choice["task"],
                "known_tasks": self._pre[robot]["known_tasks"],
                "active_event": self._pre[robot]["active_event"],
                "event_request": choice["event_request"],
                "observed_facts": _facts_list(observed[robot]),
                "action_result": results[robot],
            }
            rows.append(row)
            self.robot_rows.append(row)
        for task in range(N_TASKS):
            self.timelines[task].append(
                {
                    "t": t,
                    "state": self.states[task],
                    "progress": self.progress[task],
                    "assigned_robots": list(self.teams[task]),
                }
            )
        return deepcopy(rows)

    def result(self) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        for task in range(N_TASKS):
            truth = _task_truth(self.scenario, task)
            completed = self.states[task] == "completed"
            tasks.append(
                {
                    "task": task,
                    "position": list(truth["position"]),
                    "release_t": truth["release_t"],
                    "deadline_t": truth["deadline_t"],
                    "duration": truth["duration"],
                    "team_size": truth["team_size"],
                    "predecessors": list(truth["predecessors"]),
                    "mutex_group": truth["mutex_group"],
                    "concurrent_group": truth["concurrent_group"],
                    "first_discover_t": self.first_discover[task],
                    "start_t": self.completed_start[task] if completed else None,
                    "finish_t": self.finish[task] if completed else None,
                    "assigned_robots": list(self.completion_team[task]) if completed else [],
                    "timeline": deepcopy(self.timelines[task]),
                    "completion_reason": "completed" if completed else "expired",
                }
            )
        return {
            "scenario": self.scenario,
            "robots": deepcopy(self.robot_rows),
            "tasks": tasks,
            "communication_events": deepcopy(self.events),
        }


DecisionProvider = Callable[
    [Mapping[str, Any]],
    tuple[Sequence[Mapping[str, Any]], Mapping[str, Any] | None],
]


def _simulate_policy(scenario: str, provider: DecisionProvider) -> dict[str, Any]:
    """Run one deterministic scenario while exposing only policy-facing state."""
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario: {scenario!r}")
    model = _Model(scenario)
    for t in range(HORIZON):
        robot_views = model.begin(t)
        decisions, proposal = provider(
            {
                "scenario": scenario,
                "t": t,
                "robots": robot_views,
                "public": model.policy_public_state(t),
            }
        )
        model.resolve(t, decisions, proposal)
    return model.result()


SCENARIO_KEYS = {"scenario", "robots", "tasks", "communication_events"}
ROBOT_KEYS = {
    "robot",
    "t",
    "position",
    "move_steps_used",
    "action",
    "move",
    "task",
    "known_tasks",
    "active_event",
    "event_request",
    "observed_facts",
    "action_result",
}
FACT_KEYS = {"task", "state", "progress", "team", "version"}
TASK_KEYS = {
    "task",
    "position",
    "release_t",
    "deadline_t",
    "duration",
    "team_size",
    "predecessors",
    "mutex_group",
    "concurrent_group",
    "first_discover_t",
    "start_t",
    "finish_t",
    "assigned_robots",
    "timeline",
    "completion_reason",
}
TIMELINE_KEYS = {"t", "state", "progress", "assigned_robots"}
EVENT_KEYS = {
    "event_id",
    "propose_t",
    "requesters",
    "attempt_t",
    "meeting_positions",
    "arrival_t",
    "basis_facts",
    "connected",
    "confirm_t",
    "shared_facts",
    "status",
}
ACTION_RESULT_KEYS = {"status", "task"}


def _exact_mapping(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _fact_issue(scenario: str, value: Any, path: str) -> str | None:
    if not _exact_mapping(value, FACT_KEYS):
        return _field_issue(path, value, f"an object with keys {sorted(FACT_KEYS)}")
    task = value["task"]
    if not _integer(task) or not (0 <= task < N_TASKS):
        return _field_issue(f"{path}.task", task, "integer 0..9")
    truth = _task_truth(scenario, task)
    state, progress, team, version = (
        value["state"],
        value["progress"],
        value["team"],
        value["version"],
    )
    if not isinstance(state, str) or state not in FACT_STATES:
        return _field_issue(f"{path}.state", state, f"one of {sorted(FACT_STATES)}")
    if not _integer(progress) or not (0 <= progress <= truth["duration"]):
        return _field_issue(f"{path}.progress", progress, f"integer 0..{truth['duration']}")
    if (
        not isinstance(team, list)
        or any(not _integer(robot) or not (0 <= robot < N_ROBOTS) for robot in team)
        or team != sorted(set(team))
    ):
        return _field_issue(f"{path}.team", team, "sorted distinct robot IDs 0..3")
    if not _integer(version) or not (0 <= version < HORIZON):
        return _field_issue(f"{path}.version", version, "integer 0..35")
    if state in {"available", "expired"}:
        if progress != 0 or team != []:
            return _field_issue(path, value, f"{state} fact with progress=0 and an empty team")
        return None
    if len(team) != truth["team_size"]:
        return _field_issue(f"{path}.team", team, f"exactly {truth['team_size']} robot IDs")
    if state == "active":
        if not 1 <= progress < truth["duration"]:
            return _field_issue(f"{path}.progress", progress, f"integer 1..{truth['duration'] - 1}")
        return None
    if progress != truth["duration"]:
        return _field_issue(f"{path}.progress", progress, f"completed progress={truth['duration']}")
    return None


def _timeline_valid(scenario: str, task: int, value: Any) -> bool:
    if not _exact_mapping(value, TIMELINE_KEYS):
        return False
    t, state, progress, team = (
        value["t"],
        value["state"],
        value["progress"],
        value["assigned_robots"],
    )
    truth = _task_truth(scenario, task)
    if (
        not _integer(t)
        or not (0 <= t < HORIZON)
        or not isinstance(state, str)
        or state not in TASK_STATES
    ):
        return False
    if not _integer(progress) or not (0 <= progress <= truth["duration"]):
        return False
    if not isinstance(team, list) or any(
        not _integer(robot) or not (0 <= robot < N_ROBOTS) for robot in team
    ):
        return False
    if team != sorted(set(team)):
        return False
    if state in {"unreleased", "available", "expired"}:
        return progress == 0 and team == []
    return len(team) == truth["team_size"] and (
        (state == "active" and 1 <= progress < truth["duration"])
        or (state == "completed" and progress == truth["duration"])
    )


def _field_issue(path: str, actual: Any, expected: str) -> str:
    return f"{path}: actual={actual!r}; expected {expected}"


def _robot_field_issue(value: Any, path: str) -> str | None:
    if not isinstance(value, Mapping):
        return _field_issue(path, value, "an object")
    checks = (
        ("robot", _integer(value.get("robot")) and 0 <= value["robot"] < N_ROBOTS, "integer 0..3"),
        ("t", _integer(value.get("t")) and 0 <= value["t"] < HORIZON, "integer 0..35"),
        (
            "move_steps_used",
            _integer(value.get("move_steps_used")) and 0 <= value["move_steps_used"] <= MOVE_BUDGET,
            f"integer 0..{MOVE_BUDGET}",
        ),
        (
            "action",
            isinstance(value.get("action"), str) and value["action"] in ACTIONS,
            f"one of {sorted(ACTIONS)}",
        ),
        (
            "move",
            isinstance(value.get("move"), str) and value["move"] in MOVES,
            f"one of {sorted(MOVES)}",
        ),
        (
            "task",
            value.get("task") is None
            or (_integer(value.get("task")) and 0 <= value["task"] < N_TASKS),
            "null or integer 0..9",
        ),
        (
            "active_event",
            value.get("active_event") is None
            or (
                isinstance(value.get("active_event"), str)
                and re.fullmatch(r"e(?:0|[1-9][0-9]*)", value["active_event"]) is not None
            ),
            "null or an e<integer> ID",
        ),
        ("event_request", isinstance(value.get("event_request"), bool), "a Boolean"),
    )
    for field, valid, expected in checks:
        if not valid:
            return _field_issue(f"{path}.{field}", value.get(field), expected)
    action_result = value.get("action_result")
    if not _exact_mapping(action_result, ACTION_RESULT_KEYS):
        return _field_issue(
            f"{path}.action_result",
            action_result,
            f"an object with keys {sorted(ACTION_RESULT_KEYS)}",
        )
    if (
        not isinstance(action_result["status"], str)
        or action_result["status"] not in ACTION_RESULTS
    ):
        return _field_issue(
            f"{path}.action_result.status",
            action_result["status"],
            f"one of {sorted(ACTION_RESULTS)}",
        )
    if not (
        action_result["task"] is None
        or (_integer(action_result["task"]) and 0 <= action_result["task"] < N_TASKS)
    ):
        return _field_issue(
            f"{path}.action_result.task", action_result["task"], "null or integer 0..9"
        )
    position = value.get("position")
    if not _coordinate(position) or not _in_grid(tuple(position)):
        return _field_issue(f"{path}.position", position, "an in-grid integer pair")
    return None


def _task_field_issue(value: Any, path: str) -> str | None:
    if not isinstance(value, Mapping):
        return _field_issue(path, value, "an object")
    scalar_checks = (
        ("task", _integer(value.get("task")) and 0 <= value["task"] < N_TASKS, "integer 0..9"),
        (
            "release_t",
            _integer(value.get("release_t")) and 0 <= value["release_t"] < HORIZON,
            "integer 0..35",
        ),
        (
            "deadline_t",
            _integer(value.get("deadline_t")) and 0 <= value["deadline_t"] < HORIZON,
            "integer 0..35",
        ),
        (
            "duration",
            _integer(value.get("duration")) and value["duration"] >= 1,
            "positive integer",
        ),
        (
            "team_size",
            _integer(value.get("team_size")) and 1 <= value["team_size"] <= N_ROBOTS,
            f"integer 1..{N_ROBOTS}",
        ),
    )
    for field, valid, expected in scalar_checks:
        if not valid:
            return _field_issue(f"{path}.{field}", value.get(field), expected)
    for field in ("first_discover_t", "start_t", "finish_t"):
        actual = value.get(field)
        if actual is not None and not (_integer(actual) and 0 <= actual < HORIZON):
            return _field_issue(f"{path}.{field}", actual, "null or integer 0..35")
    for field in ("mutex_group", "concurrent_group"):
        actual = value.get(field)
        if actual is not None and not isinstance(actual, str):
            return _field_issue(f"{path}.{field}", actual, "null or a string")
    position = value.get("position")
    if not _coordinate(position) or not _in_grid(tuple(position)):
        return _field_issue(f"{path}.position", position, "an in-grid integer pair")
    predecessors = value.get("predecessors")
    if (
        not isinstance(predecessors, list)
        or not all(_integer(task) and 0 <= task < N_TASKS for task in predecessors)
        or predecessors != sorted(set(predecessors))
    ):
        return _field_issue(f"{path}.predecessors", predecessors, "sorted distinct task IDs 0..9")
    assigned = value.get("assigned_robots")
    if (
        not isinstance(assigned, list)
        or not all(_integer(robot) and 0 <= robot < N_ROBOTS for robot in assigned)
        or assigned != sorted(set(assigned))
    ):
        return _field_issue(f"{path}.assigned_robots", assigned, "sorted distinct robot IDs 0..3")
    if not isinstance(value.get("completion_reason"), str) or value.get(
        "completion_reason"
    ) not in {"completed", "expired"}:
        return _field_issue(
            f"{path}.completion_reason", value.get("completion_reason"), "completed or expired"
        )
    return None


def _event_field_issue(value: Any, path: str) -> str | None:
    if not isinstance(value, Mapping):
        return _field_issue(path, value, "an object")
    event_id = value.get("event_id")
    if not isinstance(event_id, str) or re.fullmatch(r"e(?:0|[1-9][0-9]*)", event_id) is None:
        return _field_issue(f"{path}.event_id", event_id, "an e<integer> ID")
    for field, upper in (("propose_t", HORIZON - 1), ("attempt_t", MAX_ATTEMPT_T)):
        actual = value.get(field)
        if not _integer(actual) or not (0 <= actual <= upper):
            return _field_issue(f"{path}.{field}", actual, f"integer 0..{upper}")
    if not isinstance(value.get("connected"), bool):
        return _field_issue(f"{path}.connected", value.get("connected"), "a Boolean")
    confirm_t = value.get("confirm_t")
    if confirm_t is not None and not (_integer(confirm_t) and 0 <= confirm_t < HORIZON):
        return _field_issue(f"{path}.confirm_t", confirm_t, "null or integer 0..35")
    if not isinstance(value.get("status"), str) or value.get("status") not in EVENT_STATUSES:
        return _field_issue(
            f"{path}.status", value.get("status"), f"one of {sorted(EVENT_STATUSES)}"
        )
    requesters = value.get("requesters")
    if (
        not isinstance(requesters, list)
        or not requesters
        or not all(_integer(robot) and 0 <= robot < N_ROBOTS for robot in requesters)
        or requesters != sorted(set(requesters))
    ):
        return _field_issue(
            f"{path}.requesters", requesters, "a nonempty sorted set of robot IDs 0..3"
        )
    meetings = value.get("meeting_positions")
    if (
        not isinstance(meetings, list)
        or len(meetings) != N_ROBOTS
        or not all(_coordinate(position) and _in_grid(tuple(position)) for position in meetings)
    ):
        return _field_issue(
            f"{path}.meeting_positions", meetings, f"{N_ROBOTS} in-grid integer pairs"
        )
    arrivals = value.get("arrival_t")
    if not isinstance(arrivals, list) or len(arrivals) != N_ROBOTS:
        return _field_issue(f"{path}.arrival_t", arrivals, f"an array of length {N_ROBOTS}")
    for index, arrival in enumerate(arrivals):
        if arrival is not None and not (
            _integer(arrival) and value["propose_t"] + 1 <= arrival <= value["attempt_t"]
        ):
            return _field_issue(
                f"{path}.arrival_t[{index}]",
                arrival,
                f"null or integer {value['propose_t'] + 1}..{value['attempt_t']}",
            )
    for field in ("basis_facts", "shared_facts"):
        if not isinstance(value.get(field), list):
            return _field_issue(f"{path}.{field}", value.get(field), "an array")
    return None


def _scenario_shape(value: Any) -> bool:
    return _exact_mapping(value, SCENARIO_KEYS)


def _robot_shape(value: Any) -> bool:
    return _exact_mapping(value, ROBOT_KEYS)


def _task_shape(value: Any) -> bool:
    return _exact_mapping(value, TASK_KEYS)


def _event_shape(value: Any) -> bool:
    return _exact_mapping(value, EVENT_KEYS)


def _index_records(data: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return indexed
    for item in data["scenarios"]:
        if (
            isinstance(item, Mapping)
            and item.get("scenario") in SCENARIO_NAMES
            and item["scenario"] not in indexed
        ):
            indexed[item["scenario"]] = dict(item)
    return indexed


def _atom(check_id: str, rule: str, passed: bool, detail: str) -> AtomVerdict:
    atom_id = f"{check_id}.{rule}"
    return AtomVerdict(atom_id, bool(passed), detail, ATOM_SHARES[check_id][atom_id])


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
    return max(0.0, min(1.0, sum(atom.share for atom in atoms if atom.passed)))


def _diagnose_atoms(atoms: Sequence[AtomVerdict]) -> str:
    failed = [atom for atom in atoms if not atom.passed]
    if not failed:
        return "failed_atoms=[]"
    ids = ",".join(atom.id for atom in failed)
    details = " | ".join(f"{atom.id}: {atom.detail}" for atom in failed)
    return f"failed_atoms=[{ids}]; {details}"


def _result_json_atoms(data: Any, parse_ok: bool) -> list[AtomVerdict]:
    return [
        _atom("result_json", "parse", parse_ok, "results.json: strict JSON parse required"),
        _atom(
            "result_json",
            "root",
            isinstance(data, Mapping) and set(data) == {"scenarios"},
            "results.json: exact root key set",
        ),
        _atom(
            "result_json",
            "scenarios",
            isinstance(data, Mapping) and isinstance(data.get("scenarios"), list),
            "results.json.scenarios: array required",
        ),
    ]


def _facts_ordered(values: Any) -> bool:
    if not isinstance(values, list) or not all(
        isinstance(item, Mapping) and _integer(item.get("task")) for item in values
    ):
        return False
    tasks = [item["task"] for item in values]
    return tasks == sorted(set(tasks))


def _fail_once(flags: dict[str, bool], details: dict[str, str], rule: str, detail: str) -> None:
    flags[rule] = False
    if not details[rule]:
        details[rule] = detail


def _evidence_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    names = [key.removeprefix("result_evidence.") for key in ATOM_SHARES["result_evidence"]]
    flags = {name: True for name in names}
    details = {name: "" for name in names}
    if set(indexed) != set(SCENARIO_NAMES):
        missing = sorted(set(SCENARIO_NAMES) - set(indexed))
        for name in names:
            _fail_once(flags, details, name, f"scenarios: missing canonical records {missing}")
    if list(indexed) != list(SCENARIO_NAMES):
        _fail_once(
            flags,
            details,
            "scenario_shape",
            f"scenarios: expected order={list(SCENARIO_NAMES)}, actual={list(indexed)}",
        )
    for scenario, item in indexed.items():
        if not _scenario_shape(item):
            _fail_once(
                flags,
                details,
                "scenario_shape",
                f"{scenario}: expected keys={sorted(SCENARIO_KEYS)}, actual={sorted(item)}",
            )
        robots = item.get("robots") if isinstance(item, Mapping) else None
        tasks = item.get("tasks") if isinstance(item, Mapping) else None
        events = item.get("communication_events") if isinstance(item, Mapping) else None
        if not isinstance(robots, list):
            for name in ("robot_shape", "robot_fields", "robot_order", "fact_fields"):
                _fail_once(
                    flags, details, name, f"{scenario}.robots: expected array, actual={robots!r}"
                )
        else:
            robot_keys: list[tuple[Any, Any]] = []
            for index, row in enumerate(robots):
                path = f"{scenario}.robots[{index}]"
                if not _robot_shape(row):
                    actual = sorted(row) if isinstance(row, Mapping) else type(row).__name__
                    _fail_once(
                        flags,
                        details,
                        "robot_shape",
                        f"{path}: expected keys={sorted(ROBOT_KEYS)}, actual={actual}",
                    )
                issue = _robot_field_issue(row, path)
                if issue is not None:
                    _fail_once(
                        flags,
                        details,
                        "robot_fields",
                        issue,
                    )
                if not isinstance(row, Mapping):
                    _fail_once(flags, details, "fact_fields", f"{path}: mapping required")
                    continue
                robot_keys.append((row.get("robot"), row.get("t")))
                known, observed = row.get("known_tasks"), row.get("observed_facts")
                for field, values in (("known_tasks", known), ("observed_facts", observed)):
                    if not _facts_ordered(values):
                        _fail_once(
                            flags,
                            details,
                            "fact_fields",
                            f"{path}.{field}: expected facts sorted by distinct task ID, "
                            f"actual={values!r}",
                        )
                    else:
                        for fact_index, fact in enumerate(values):
                            issue = _fact_issue(scenario, fact, f"{path}.{field}[{fact_index}]")
                            if issue is not None:
                                _fail_once(flags, details, "fact_fields", issue)
                                break
            expected_robot_keys = [(robot, t) for t in range(HORIZON) for robot in range(N_ROBOTS)]
            if robot_keys != expected_robot_keys:
                mismatch = next(
                    (
                        (index, expected_robot_keys[index], robot_keys[index])
                        for index in range(min(len(robot_keys), len(expected_robot_keys)))
                        if robot_keys[index] != expected_robot_keys[index]
                    ),
                    None,
                )
                _fail_once(
                    flags,
                    details,
                    "robot_order",
                    f"{scenario}.robots: expected t-then-robot order; first_mismatch={mismatch}; "
                    f"actual_length={len(robot_keys)}, expected_length={len(expected_robot_keys)}",
                )
        if not isinstance(tasks, list):
            for name in ("task_shape", "task_fields", "task_order", "timeline_fields"):
                _fail_once(
                    flags, details, name, f"{scenario}.tasks: expected array, actual={tasks!r}"
                )
        else:
            task_ids: list[Any] = []
            for index, record in enumerate(tasks):
                path = f"{scenario}.tasks[{index}]"
                if not _task_shape(record):
                    actual = (
                        sorted(record) if isinstance(record, Mapping) else type(record).__name__
                    )
                    _fail_once(
                        flags,
                        details,
                        "task_shape",
                        f"{path}: expected keys={sorted(TASK_KEYS)}, actual={actual}",
                    )
                issue = _task_field_issue(record, path)
                if issue is not None:
                    _fail_once(
                        flags,
                        details,
                        "task_fields",
                        issue,
                    )
                if not isinstance(record, Mapping):
                    _fail_once(flags, details, "timeline_fields", f"{path}: mapping required")
                    continue
                task_ids.append(record.get("task"))
                task = record.get("task")
                timeline = record.get("timeline")
                if (
                    not _integer(task)
                    or not (0 <= task < N_TASKS)
                    or not isinstance(timeline, list)
                    or len(timeline) != HORIZON
                ):
                    _fail_once(
                        flags,
                        details,
                        "timeline_fields",
                        f"{path}.timeline: expected {HORIZON} records for a valid task ID, "
                        f"actual_length={len(timeline) if isinstance(timeline, list) else None}",
                    )
                else:
                    actual_times = [
                        value.get("t") if isinstance(value, Mapping) else None for value in timeline
                    ]
                    if actual_times != list(range(HORIZON)):
                        _fail_once(
                            flags,
                            details,
                            "timeline_fields",
                            f"{path}.timeline[*].t: actual={actual_times}; expected ordered "
                            f"integers 0..{HORIZON - 1}",
                        )
                    else:
                        for timeline_index, value in enumerate(timeline):
                            if not _timeline_valid(scenario, task, value):
                                _fail_once(
                                    flags,
                                    details,
                                    "timeline_fields",
                                    _field_issue(
                                        f"{path}.timeline[{timeline_index}]",
                                        value,
                                        "the declared timeline field and state domain",
                                    ),
                                )
                                break
            if task_ids != list(range(N_TASKS)):
                _fail_once(
                    flags,
                    details,
                    "task_order",
                    f"{scenario}.tasks: expected task order={list(range(N_TASKS))}, "
                    f"actual={task_ids}",
                )
        if not isinstance(events, list):
            for name in ("event_shape", "event_fields", "event_order", "fact_fields"):
                _fail_once(
                    flags,
                    details,
                    name,
                    f"{scenario}.communication_events: expected array, actual={events!r}",
                )
        else:
            event_order: list[tuple[Any, Any]] = []
            for index, event in enumerate(events):
                path = f"{scenario}.communication_events[{index}]"
                if not _event_shape(event):
                    actual = sorted(event) if isinstance(event, Mapping) else type(event).__name__
                    _fail_once(
                        flags,
                        details,
                        "event_shape",
                        f"{path}: expected keys={sorted(EVENT_KEYS)}, actual={actual}",
                    )
                issue = _event_field_issue(event, path)
                if issue is not None:
                    _fail_once(
                        flags,
                        details,
                        "event_fields",
                        issue,
                    )
                if not isinstance(event, Mapping):
                    _fail_once(flags, details, "fact_fields", f"{path}: mapping required")
                    continue
                event_order.append((event.get("event_id"), event.get("propose_t")))
                basis, shared = event.get("basis_facts"), event.get("shared_facts")
                for field, values in (("basis_facts", basis), ("shared_facts", shared)):
                    if not _facts_ordered(values):
                        _fail_once(
                            flags,
                            details,
                            "fact_fields",
                            f"{path}.{field}: expected facts sorted by distinct task ID, "
                            f"actual={values!r}",
                        )
                    else:
                        for fact_index, fact in enumerate(values):
                            issue = _fact_issue(scenario, fact, f"{path}.{field}[{fact_index}]")
                            if issue is not None:
                                _fail_once(flags, details, "fact_fields", issue)
                                break
            expected_ids = [f"e{index}" for index in range(len(events))]
            actual_ids = [value[0] for value in event_order]
            propose_times = [value[1] for value in event_order]
            proposal_ordered = all(
                _integer(value) for value in propose_times
            ) and propose_times == sorted(propose_times)
            if actual_ids != expected_ids or not proposal_ordered:
                _fail_once(
                    flags,
                    details,
                    "event_order",
                    f"{scenario}.communication_events: expected IDs={expected_ids} in proposal "
                    f"order, actual_ids={actual_ids}, propose_t={propose_times}",
                )
    return [
        _atom(
            "result_evidence",
            name,
            flags[name],
            details[name] or "declared evidence structure, domain, and order satisfied",
        )
        for name in names
    ]


def _coverage_analysis(data: Any) -> tuple[float, list[str]]:
    defects: list[str] = []
    if not isinstance(data, Mapping) or not isinstance(data.get("scenarios"), list):
        return 0.0, ["scenarios array unavailable"]
    items = data["scenarios"]
    names = [item.get("scenario") if isinstance(item, Mapping) else None for item in items]
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        if name in SCENARIO_NAMES:
            counts[name] += 1
        else:
            defects.append(f"scenarios: unknown key {name!r}")
    for scenario in SCENARIO_NAMES:
        if counts[scenario] == 0:
            defects.append(f"{scenario}: missing scenario")
        elif counts[scenario] > 1:
            defects.append(f"{scenario}: duplicate scenario count={counts[scenario]}")
    for item in items:
        if not isinstance(item, Mapping) or item.get("scenario") not in SCENARIO_NAMES:
            continue
        scenario = item["scenario"]
        robots = item.get("robots")
        if not isinstance(robots, list):
            defects.append(f"{scenario}.robots: array required")
        else:
            robot_counts: dict[tuple[int, int], int] = defaultdict(int)
            for row in robots:
                if (
                    isinstance(row, Mapping)
                    and _integer(row.get("robot"))
                    and _integer(row.get("t"))
                ):
                    key = (row["robot"], row["t"])
                    if 0 <= key[0] < N_ROBOTS and 0 <= key[1] < HORIZON:
                        robot_counts[key] += 1
                    else:
                        defects.append(f"{scenario}.robots: unknown key {key}")
                else:
                    defects.append(f"{scenario}.robots: malformed canonical key")
            for key in ((robot, t) for t in range(HORIZON) for robot in range(N_ROBOTS)):
                if robot_counts[key] == 0:
                    defects.append(f"{scenario}.robots: missing key {key}")
                elif robot_counts[key] > 1:
                    defects.append(f"{scenario}.robots: duplicate key {key}")
        tasks = item.get("tasks")
        if not isinstance(tasks, list):
            defects.append(f"{scenario}.tasks: array required")
        else:
            task_counts: dict[int, int] = defaultdict(int)
            for record in tasks:
                if (
                    isinstance(record, Mapping)
                    and _integer(record.get("task"))
                    and 0 <= record["task"] < N_TASKS
                ):
                    task_counts[record["task"]] += 1
                else:
                    defects.append(f"{scenario}.tasks: unknown or malformed key")
            for task in range(N_TASKS):
                if task_counts[task] == 0:
                    defects.append(f"{scenario}.tasks: missing key {task}")
                elif task_counts[task] > 1:
                    defects.append(f"{scenario}.tasks: duplicate key {task}")
    return max(0.0, 4.0 - 0.5 * len(defects)) / 4.0, defects


def _legality_atoms(indexed: Mapping[str, dict[str, Any]]) -> list[AtomVerdict]:
    names = [key.removeprefix("decision_legality.") for key in ATOM_SHARES["decision_legality"]]
    flags = {name: True for name in names}
    details = {name: "" for name in flags}
    for scenario in SCENARIO_NAMES:
        item = indexed.get(scenario)
        if item is None:
            for name in names:
                _fail_once(flags, details, name, f"{scenario}: canonical scenario record required")
            continue
        robots = item.get("robots", [])
        events = item.get("communication_events", [])
        if not isinstance(robots, list) or not isinstance(events, list):
            for name in names:
                _fail_once(
                    flags,
                    details,
                    name,
                    f"{scenario}: robot and communication-event arrays required",
                )
            continue
        mappings = [row for row in robots if isinstance(row, Mapping)]
        row_index = {(row.get("robot"), row.get("t")): row for row in mappings}
        for row in mappings:
            action, move, task = row.get("action"), row.get("move"), row.get("task")
            relation_ok = (
                (action == "move" and isinstance(move, str) and move in MOVES and task is None)
                or (action == "work" and move == "stay" and _integer(task) and 0 <= task < N_TASKS)
                or (action == "wait" and move == "stay" and task is None)
            )
            if not relation_ok:
                _fail_once(
                    flags,
                    details,
                    "action_relations",
                    f"{scenario}.robots[{row.get('robot')},{row.get('t')}]: "
                    f"expected declared action/move/task relation, actual="
                    f"{action!r}/{move!r}/{task!r}",
                )
            if (
                action == "move"
                and isinstance(move, str)
                and move in MOVES
                and move != "stay"
                and _coordinate(row.get("position"))
            ):
                dx, dy = MOVES[move]
                target = (row["position"][0] + dx, row["position"][1] + dy)
                if (
                    not _free(scenario, target)
                    or not _integer(row.get("move_steps_used"))
                    or row["move_steps_used"] >= MOVE_BUDGET
                ):
                    _fail_once(
                        flags,
                        details,
                        "motion_bounds",
                        f"{scenario}.robots[{row.get('robot')},{row.get('t')}]: "
                        f"target={target}, move_steps_used={row.get('move_steps_used')}, "
                        f"required free target and budget<{MOVE_BUDGET}",
                    )
            if action == "work" and _integer(task) and 0 <= task < N_TASKS:
                known_tasks = row.get("known_tasks")
                facts = (
                    {
                        fact.get("task"): fact
                        for fact in known_tasks
                        if isinstance(fact, Mapping) and _integer(fact.get("task"))
                    }
                    if isinstance(known_tasks, list)
                    else {}
                )
                truth = _task_truth(scenario, task)
                fact = facts.get(task)
                if (
                    fact is None
                    or row.get("position") != list(truth["position"])
                    or not _integer(row.get("t"))
                    or row["t"] > truth["deadline_t"]
                    or fact.get("state") == "completed"
                    or fact.get("state") == "expired"
                ):
                    _fail_once(
                        flags,
                        details,
                        "work_local_state",
                        f"{scenario}.robots[{row.get('robot')},{row.get('t')}]: "
                        f"task={task}, expected known nonterminal fact, position="
                        f"{list(truth['position'])}, t<={truth['deadline_t']}; actual fact="
                        f"{fact!r}, position={row.get('position')!r}, t={row.get('t')!r}",
                    )
        event_maps = [event for event in events if isinstance(event, Mapping)]
        by_propose: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for event in event_maps:
            if _integer(event.get("propose_t")):
                by_propose[event["propose_t"]].append(event)
        for t in range(HORIZON):
            requesters = [
                robot
                for robot in range(N_ROBOTS)
                if row_index.get((robot, t), {}).get("event_request") is True
            ]
            proposals = by_propose.get(t, [])
            if bool(requesters) != (len(proposals) == 1):
                _fail_once(
                    flags,
                    details,
                    "event_request_links",
                    f"{scenario}.t={t}: requesters={requesters}, matching_event_count="
                    f"{len(proposals)}; expected one event iff requesters are nonempty",
                )
            if len(proposals) == 1 and proposals[0].get("requesters") != requesters:
                _fail_once(
                    flags,
                    details,
                    "event_request_links",
                    f"{scenario}.t={t}: expected requesters={requesters}, "
                    f"actual={proposals[0].get('requesters')!r}",
                )
            if any(
                row_index.get((robot, t), {}).get("active_event") is not None
                and robot in requesters
                for robot in range(N_ROBOTS)
            ):
                _fail_once(
                    flags,
                    details,
                    "event_resources",
                    f"{scenario}.t={t}: requesters={requesters} while an active event is recorded",
                )
        if len(event_maps) > EVENT_BUDGET:
            _fail_once(
                flags,
                details,
                "event_resources",
                f"{scenario}: expected at most {EVENT_BUDGET} events, actual={len(event_maps)}",
            )
        previous_attempt: int | None = None
        for index, event in enumerate(event_maps):
            propose, attempt = event.get("propose_t"), event.get("attempt_t")
            meetings = event.get("meeting_positions")
            valid_resource = (
                _integer(propose)
                and _integer(attempt)
                and propose + MIN_EVENT_LAG
                <= attempt
                <= min(propose + MAX_EVENT_LAG, MAX_ATTEMPT_T)
                and (previous_attempt is None or propose - previous_attempt >= 2)
            )
            if not valid_resource:
                _fail_once(
                    flags,
                    details,
                    "event_resources",
                    f"{scenario}.communication_events[{index}]: "
                    f"expected attempt in [propose+{MIN_EVENT_LAG}, "
                    f"min(propose+{MAX_EVENT_LAG},{MAX_ATTEMPT_T})] and proposal gap>=2; "
                    f"actual propose_t={propose!r}, attempt_t={attempt!r}, "
                    f"previous_attempt_t={previous_attempt!r}",
                )
            valid_positions = (
                isinstance(meetings, list)
                and len(meetings) == N_ROBOTS
                and all(
                    _coordinate(position) and _free(scenario, tuple(position))
                    for position in meetings
                )
            )
            if not valid_positions:
                _fail_once(
                    flags,
                    details,
                    "event_positions",
                    f"{scenario}.communication_events[{index}].meeting_positions: expected "
                    f"{N_ROBOTS} free grid cells, actual={meetings!r}",
                )
            if _integer(attempt):
                previous_attempt = attempt
        for event in event_maps:
            if not _integer(event.get("propose_t")) or not _integer(event.get("attempt_t")):
                continue
            for t in range(event["propose_t"] + 1, min(event["attempt_t"], HORIZON - 1) + 1):
                if any(
                    row_index.get((robot, t), {}).get("event_request") is True
                    for robot in range(N_ROBOTS)
                ):
                    _fail_once(
                        flags,
                        details,
                        "event_resources",
                        f"{scenario}.t={t}: request overlaps an active plan",
                    )
    return [
        _atom(
            "decision_legality", name, flags[name], details[name] or "decision constraint satisfied"
        )
        for name in names
    ]


def _candidate_rows(item: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    raw = item.get("robots")
    if not isinstance(raw, list):
        return rows
    for row in raw:
        if (
            not isinstance(row, Mapping)
            or not _integer(row.get("robot"))
            or not _integer(row.get("t"))
        ):
            continue
        key = (row["robot"], row["t"])
        if key not in rows:
            rows[key] = row
    return rows


def _candidate_events(
    item: Mapping[str, Any],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    by_propose: dict[int, Mapping[str, Any]] = {}
    ordered: list[Mapping[str, Any]] = []
    raw = item.get("communication_events")
    if not isinstance(raw, list):
        return by_propose, ordered
    for event in raw:
        if not isinstance(event, Mapping):
            continue
        ordered.append(event)
        propose = event.get("propose_t")
        if _integer(propose) and propose not in by_propose:
            by_propose[propose] = event
    return by_propose, ordered


def _first_difference(expected: Any, actual: Any, path: str) -> str | None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: expected keys={sorted(expected_keys)}, actual keys={sorted(actual_keys)}"
            )
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: expected length={len(expected)}, actual length={len(actual)}"
        for index, (wanted, observed) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(wanted, observed, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


def _replay_scenario(
    scenario: str, item: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool], dict[str, str]]:
    model = _Model(scenario)
    rows = _candidate_rows(item)
    events_by_propose, _ = _candidate_events(item)
    for t in range(HORIZON):
        model.begin(t)
        decisions: list[Mapping[str, Any]] = []
        for robot in range(N_ROBOTS):
            row = rows.get((robot, t), {})
            decisions.append(
                {
                    "action": row.get("action", "wait"),
                    "move": row.get("move", "stay"),
                    "task": row.get("task"),
                    "event_request": row.get("event_request", False),
                }
            )
        model.resolve(t, decisions, events_by_propose.get(t))
    expected = model.result()
    expected_rows = {(row["robot"], row["t"]): row for row in expected["robots"]}
    checks = {
        name: True
        for name in (
            "positions_budget",
            "knowledge",
            "observations",
            "active_events",
            "movement_results",
            "work_results",
            "task_timelines",
            "task_records",
            "event_basis",
            "events",
        )
    }
    details = {name: "" for name in checks}

    for key in ((robot, t) for t in range(HORIZON) for robot in range(N_ROBOTS)):
        actual = rows.get(key)
        wanted = expected_rows[key]
        if actual is None:
            for name in (
                "positions_budget",
                "knowledge",
                "observations",
                "active_events",
                "movement_results",
                "work_results",
            ):
                checks[name] = False
                details[name] = f"{scenario}.robots{key}: missing canonical record"
            continue
        if (
            actual.get("position") != wanted["position"]
            or actual.get("move_steps_used") != wanted["move_steps_used"]
        ):
            checks["positions_budget"] = False
            if not details["positions_budget"]:
                details["positions_budget"] = (
                    f"{scenario}.robots{key}: expected position/budget "
                    f"{wanted['position']}/{wanted['move_steps_used']}, "
                    f"actual={actual.get('position')}/{actual.get('move_steps_used')}"
                )
        if actual.get("known_tasks") != wanted["known_tasks"]:
            checks["knowledge"] = False
            if not details["knowledge"]:
                details["knowledge"] = (
                    _first_difference(
                        wanted["known_tasks"],
                        actual.get("known_tasks"),
                        f"{scenario}.robots{key}.known_tasks",
                    )
                    or f"{scenario}.robots{key}.known_tasks: mismatch"
                )
        if actual.get("observed_facts") != wanted["observed_facts"]:
            checks["observations"] = False
            if not details["observations"]:
                details["observations"] = (
                    _first_difference(
                        wanted["observed_facts"],
                        actual.get("observed_facts"),
                        f"{scenario}.robots{key}.observed_facts",
                    )
                    or f"{scenario}.robots{key}.observed_facts: mismatch"
                )
        if actual.get("active_event") != wanted["active_event"]:
            checks["active_events"] = False
            if not details["active_events"]:
                details["active_events"] = (
                    f"{scenario}.robots{key}.active_event: "
                    f"expected {wanted['active_event']!r}, "
                    f"actual={actual.get('active_event')!r}"
                )
        action = actual.get("action")
        if action in {"move", "wait"} and actual.get("action_result") != wanted["action_result"]:
            checks["movement_results"] = False
            if not details["movement_results"]:
                details["movement_results"] = (
                    f"{scenario}.robots{key}.action_result: "
                    f"expected {wanted['action_result']}, "
                    f"actual={actual.get('action_result')}"
                )
        if action == "work" and actual.get("action_result") != wanted["action_result"]:
            checks["work_results"] = False
            if not details["work_results"]:
                details["work_results"] = (
                    f"{scenario}.robots{key}.action_result: "
                    f"expected {wanted['action_result']}, "
                    f"actual={actual.get('action_result')}"
                )

    actual_tasks = item.get("tasks") if isinstance(item.get("tasks"), list) else []
    actual_task_index = {
        record.get("task"): record
        for record in actual_tasks
        if isinstance(record, Mapping) and _integer(record.get("task"))
    }
    for wanted in expected["tasks"]:
        actual = actual_task_index.get(wanted["task"])
        if actual is None or actual.get("timeline") != wanted["timeline"]:
            checks["task_timelines"] = False
            if not details["task_timelines"]:
                details["task_timelines"] = (
                    f"{scenario}.tasks[{wanted['task']}]: missing task record"
                    if actual is None
                    else _first_difference(
                        wanted["timeline"],
                        actual.get("timeline"),
                        f"{scenario}.tasks[{wanted['task']}].timeline",
                    )
                    or f"{scenario}.tasks[{wanted['task']}].timeline: mismatch"
                )
        if actual is None or {key: actual.get(key) for key in TASK_KEYS - {"timeline"}} != {
            key: wanted[key] for key in TASK_KEYS - {"timeline"}
        }:
            checks["task_records"] = False
            if not details["task_records"]:
                expected_record = {key: wanted[key] for key in TASK_KEYS - {"timeline"}}
                actual_record = (
                    None
                    if actual is None
                    else {key: actual.get(key) for key in TASK_KEYS - {"timeline"}}
                )
                details["task_records"] = (
                    _first_difference(
                        expected_record,
                        actual_record,
                        f"{scenario}.tasks[{wanted['task']}]",
                    )
                    or f"{scenario}.tasks[{wanted['task']}]: terminal/static record mismatch"
                )
    actual_events = item.get("communication_events")
    expected_events = expected["communication_events"]
    actual_basis = (
        [event.get("basis_facts") for event in actual_events if isinstance(event, Mapping)]
        if isinstance(actual_events, list)
        else actual_events
    )
    expected_basis = [event["basis_facts"] for event in expected_events]
    if actual_basis != expected_basis:
        checks["event_basis"] = False
        details["event_basis"] = (
            _first_difference(
                expected_basis,
                actual_basis,
                f"{scenario}.communication_events[*].basis_facts",
            )
            or f"{scenario}.communication_events[*].basis_facts: mismatch"
        )
    actual_without_basis = (
        [
            {key: value for key, value in event.items() if key != "basis_facts"}
            for event in actual_events
            if isinstance(event, Mapping)
        ]
        if isinstance(actual_events, list)
        else actual_events
    )
    expected_without_basis = [
        {key: value for key, value in event.items() if key != "basis_facts"}
        for event in expected_events
    ]
    if actual_without_basis != expected_without_basis:
        checks["events"] = False
        details["events"] = (
            _first_difference(
                expected_without_basis,
                actual_without_basis,
                f"{scenario}.communication_events",
            )
            or f"{scenario}.communication_events: fixed event replay mismatch"
        )

    completed = sum(record["completion_reason"] == "completed" for record in expected["tasks"])
    utility = 0.0
    for record in expected["tasks"]:
        if record["finish_t"] is None:
            continue
        utility += max(
            0.0,
            min(
                1.0,
                1.0
                - (record["finish_t"] - record["release_t"])
                / (record["deadline_t"] - record["release_t"]),
            ),
        )
    attempt_by_id = {
        event["event_id"]: event["attempt_t"] for event in expected["communication_events"]
    }
    wait_slots = sum(
        row["active_event"] is not None
        and row["t"] < attempt_by_id.get(row["active_event"], -1)
        and (row["action"] == "wait" or (row["action"] == "move" and row["move"] == "stay"))
        for row in expected["robots"]
    )
    replayed = {
        "completed": completed / N_TASKS,
        "utility": utility / N_TASKS,
        "event_attempts": len(expected["communication_events"]),
        "rendezvous_wait": wait_slots,
        "expected": expected,
    }
    return replayed, checks, details


def _replay_atoms(
    indexed: Mapping[str, dict[str, Any]],
) -> tuple[list[AtomVerdict], dict[str, dict[str, Any]]]:
    names = [key.removeprefix("replay_validity.") for key in ATOM_SHARES["replay_validity"]]
    aggregate = {name: True for name in names}
    details = {name: "" for name in names}
    replayed: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIO_NAMES:
        item = indexed.get(scenario)
        if item is None:
            for name in names:
                aggregate[name] = False
                details[name] = "all canonical scenarios are required"
            continue
        try:
            result, checks, local = _replay_scenario(scenario, item)
            replayed[scenario] = result
        except Exception as exc:  # noqa: BLE001
            checks = {name: False for name in names}
            local = {name: f"replay raised {type(exc).__name__}: {exc}" for name in names}
        for name in names:
            if not checks.get(name, False):
                aggregate[name] = False
                if not details[name]:
                    details[name] = local.get(name, "replay mismatch")
    atoms = [
        _atom("replay_validity", name, aggregate[name], details[name] or "replay matched")
        for name in names
    ]
    return atoms, replayed


def _metrics(replayed: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    completion = sum(replayed[name]["completed"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    utility = sum(replayed[name]["utility"] for name in SCENARIO_NAMES) / len(SCENARIO_NAMES)
    event_efficiency = sum(
        max(0.0, min(1.0, 1.0 - replayed[name]["event_attempts"] / EVENT_BUDGET))
        for name in SCENARIO_NAMES
    ) / len(SCENARIO_NAMES)
    waiting = sum(
        max(0.0, min(1.0, 1.0 - replayed[name]["rendezvous_wait"] / (N_ROBOTS * HORIZON)))
        for name in SCENARIO_NAMES
    ) / len(SCENARIO_NAMES)
    return {
        "primary_metrics_1": completion,
        "primary_metrics_2": utility,
        "auxiliary_metrics_1": event_efficiency,
        "auxiliary_metrics_2": waiting,
        "normalized_task_performance": (
            8 * completion + 7 * utility + 3 * event_efficiency + 2 * waiting
        )
        / 20,
    }


def _gated_metrics(
    official_ok: bool,
    scores: Mapping[str, float],
    replayed: Mapping[str, dict[str, Any]],
) -> dict[str, float] | None:
    required = (
        "result_json",
        "result_evidence",
        "canonical_coverage",
        "decision_legality",
        "replay_validity",
    )
    if not official_ok or not all(scores.get(name, 0.0) >= 1.0 - EPS for name in required):
        return None
    return _metrics(replayed)


def _gated_metric_result(
    name: str,
    official_ok: bool,
    scores: Mapping[str, float],
    replayed: Mapping[str, dict[str, Any]],
) -> tuple[float, str]:
    metrics = _gated_metrics(official_ok, scores, replayed)
    if metrics is None:
        return 0.0, f"{name}: validity gate not satisfied; metric not computed"
    value = metrics[name]
    return value, f"{name}: independently replayed value={value:.9g}"


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
    parsed: list[tuple[int, int, int, str]] = []
    heading_lines: set[int] = set()
    fence_lines: set[int] = set()
    fence_char: str | None = None
    fence_length = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_char is not None:
            if fence:
                marker = fence.group(1)
                if (
                    marker[0] == fence_char
                    and len(marker) >= fence_length
                    and not fence.group(2).strip()
                ):
                    fence_char = None
                    fence_length = 0
                fence_lines.add(i)
            i += 1
            continue
        if fence:
            marker = fence.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            fence_lines.add(i)
            i += 1
            continue
        atx = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", line)
        if atx:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", (atx.group(2) or "").strip()).strip()
            parsed.append((i, i, len(atx.group(1)), title))
            heading_lines.add(i)
            i += 1
            continue
        if i + 1 < len(lines):
            title_match = re.match(r"^ {0,3}(\S.*?)\s*$", line)
            underline = re.match(r"^ {0,3}(=+|-+)[ \t]*$", lines[i + 1])
            if title_match and underline:
                parsed.append(
                    (
                        i,
                        i + 1,
                        1 if underline.group(1)[0] == "=" else 2,
                        title_match.group(1).strip(),
                    )
                )
                heading_lines.update((i, i + 1))
                i += 2
                continue
        i += 1
    found: dict[str, str] = {}
    for index, (start, end, level, title) in enumerate(parsed):
        if title not in required:
            continue
        stop = len(lines)
        for next_start, _, next_level, _ in parsed[index + 1 :]:
            if next_level <= level:
                stop = next_start
                break
        body = "\n".join(
            lines[line]
            for line in range(end + 1, stop)
            if line not in heading_lines and line not in fence_lines
        ).strip()
        if title not in found or not found[title]:
            found[title] = body
    return found


def _regular_file(ws: Workspace, relative: str) -> bool:
    path = Path(ws.root) / relative
    return path.is_file() and not path.is_symlink()


def _report_score(ws: Workspace) -> tuple[float, str]:
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected a regular non-symlink file"
    try:
        headings = _headings(ws.read_text("report.md") or "")
    except (OSError, UnicodeError) as exc:
        return 0.0, f"report.md: read failed: {exc}"
    missing = sorted(name for name in required if not headings.get(name))
    score = max(0.0, (2.0 - 0.5 * len(missing)) / 2.0)
    return score, f"report.md: missing_or_empty={missing}; normalized={score:.9g}"


_CACHE: dict[str, EvalState] = {}


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
            removed = await ws.run("rm -f -- results.json", check=False, timeout=30.0)
            remove_ok = getattr(removed, "exit_code", None) == 0
        except Exception as exc:  # noqa: BLE001
            state.official_message = f"could not remove previous results.json: {exc}"
        try:
            result = await ws.run("python run.py", check=False, timeout=600.0)
            state.exit_code = getattr(result, "exit_code", None)
            stderr_text = str(getattr(result, "stderr", "")).lower()
            state.timed_out = "timeout" in stderr_text or "timed out" in stderr_text
        except Exception as exc:  # noqa: BLE001
            state.exit_code = None
            state.timed_out = (
                isinstance(exc, TimeoutError)
                or "timeout" in str(exc).lower()
                or "timed out" in str(exc).lower()
            )
            state.official_message = f"official run raised {type(exc).__name__}: {exc}"
        state.elapsed_s = time.monotonic() - started
        if state.exit_code is None and state.elapsed_s >= 600.0 - EPS:
            state.timed_out = True
        state.run_finished = True
        state.result_file_is_new_regular = remove_ok and _regular_file(ws, "results.json")
        state.official_ok = bool(
            remove_ok
            and state.exit_code == 0
            and not state.timed_out
            and state.elapsed_s <= 600.0 + EPS
            and state.result_file_is_new_regular
        )
        if not state.official_message:
            state.official_message = (
                f"exit={state.exit_code}, elapsed={state.elapsed_s:.3f}s, "
                f"timed_out={state.timed_out}, new_regular_results={state.result_file_is_new_regular}"
            )
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
            if not state.official_ok:
                state.scores["result_json"] = 0.0
                state.diagnostics["result_json"] = reason
            for check_id in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                state.scores[check_id] = 0.0
                state.diagnostics[check_id] = reason
            return state
        state.records = _index_records(state.data)
        evidence = _evidence_atoms(state.records)
        state.scores["result_evidence"] = _score_atoms(evidence, ATOM_SHARES["result_evidence"])
        state.diagnostics["result_evidence"] = _diagnose_atoms(evidence)
        coverage, defects = _coverage_analysis(state.data)
        state.scores["canonical_coverage"] = coverage
        state.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:20]}"
        legality = _legality_atoms(state.records)
        state.scores["decision_legality"] = _score_atoms(legality, ATOM_SHARES["decision_legality"])
        state.diagnostics["decision_legality"] = _diagnose_atoms(legality)
        replay, state.replayed = _replay_atoms(state.records)
        state.scores["replay_validity"] = _score_atoms(replay, ATOM_SHARES["replay_validity"])
        state.diagnostics["replay_validity"] = _diagnose_atoms(replay)
        state.metrics = _gated_metrics(state.official_ok, state.scores, state.replayed)
        state.validity_gate = state.metrics is not None
    except Exception as exc:  # noqa: BLE001
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        state.validity_gate = False
        state.metrics = None
        if not state.run_finished:
            state.official_ok = False
            state.official_message = state.official_message or detail
        for check_id in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            state.scores.setdefault(check_id, 0.0)
            state.diagnostics.setdefault(check_id, detail)
    finally:
        _CACHE[key] = state
    return state


@check(id="entry_point")
async def entry_point(ws: Workspace) -> tuple[float, str]:
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected a regular non-symlink file"
    result = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    if getattr(result, "exit_code", None) == 0:
        return 1.0, "run.py: static compilation passed"
    return 0.0, "run.py: static compilation failed"


@check(id="official_run")
async def official_run(ws: Workspace) -> tuple[float, str]:
    state = await _build_state(ws)
    return (1.0 if state.official_ok else 0.0), state.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace) -> tuple[float, str]:
    return _report_score(ws)


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


async def _metric_check(ws: Workspace, name: str) -> tuple[float, str]:
    state = await _build_state(ws)
    return _gated_metric_result(name, state.official_ok, state.scores, state.replayed)


@check(id="primary_metrics_1")
async def primary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_1")


@check(id="primary_metrics_2")
async def primary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "primary_metrics_2")


@check(id="auxiliary_metrics_1")
async def auxiliary_metrics_1(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_1")


@check(id="auxiliary_metrics_2")
async def auxiliary_metrics_2(ws: Workspace) -> tuple[float, str]:
    return await _metric_check(ws, "auxiliary_metrics_2")
