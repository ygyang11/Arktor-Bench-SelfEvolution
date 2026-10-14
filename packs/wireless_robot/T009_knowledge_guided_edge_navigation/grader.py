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

GRID_W = GRID_H = 12
HORIZON = 36
N_OBJECTS = 8
MAX_OBSERVATIONS = 4
REPORT_BUDGET = 24
REPORT_WINDOW = 4
REPORT_WINDOW_LIMIT = 2
EPS = 1e-6
SCENARIO_NAMES = ("short_path", "overlap_paths", "weak_uplink", "late_feedback", "alias_ambiguity")
ACTIONS = frozenset({"move", "observe", "report", "wait"})
MOVES = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0), "stay": (0, 0)}
LINKS = frozenset({"strong", "weak"})
UPLINK_STATUSES = frozenset({"delivered", "dropped", "late"})
POSITIONS = ((2, 1), (4, 1), (7, 1), (10, 1), (10, 4), (7, 4), (4, 4), (1, 4))
CATEGORIES = ("key", "panel", "valve", "parcel", "dock", "switch")
OBJECT_CATEGORIES = {
    "short_path": ("key", "panel", "valve", "parcel", "dock", "switch", "panel", "dock"),
    "overlap_paths": ("key", "panel", "dock", "valve", "switch", "parcel", "panel", "dock"),
    "weak_uplink": ("key", "panel", "valve", "parcel", "dock", "switch", "panel", "dock"),
    "late_feedback": ("key", "panel", "valve", "parcel", "dock", "switch", "panel", "dock"),
    "alias_ambiguity": ("key", "panel", "valve", "parcel", "dock", "switch", "panel", "dock"),
}
ALIASES = {
    "key": {"key"},
    "panel": {"panel"},
    "valve": {"valve"},
    "parcel": {"parcel"},
    "dock": {"dock"},
    "switch": {"switch"},
    "access": {"key", "switch"},
    "control": {"panel", "valve"},
    "rotary": {"valve", "switch"},
    "station": {"panel", "dock"},
    "cargo": {"parcel", "dock"},
}
SCENARIOS = {
    "short_path": {
        "endpoint": (11, 10),
        "access": (6, 2),
        "deadline": 34,
        "downlink": 1,
        "outages": {5, 12, 20},
        "spikes": {9, 18},
    },
    "overlap_paths": {
        "endpoint": (11, 10),
        "access": (6, 3),
        "deadline": 35,
        "downlink": 2,
        "outages": {6, 13, 21},
        "spikes": {8, 17},
    },
    "weak_uplink": {
        "endpoint": (11, 10),
        "access": (6, 6),
        "deadline": 35,
        "downlink": 2,
        "outages": {1, 6, 14, 22},
        "spikes": {5, 16, 25},
    },
    "late_feedback": {
        "endpoint": (10, 8),
        "access": (5, 2),
        "deadline": 31,
        "downlink": 5,
        "outages": {3, 11, 19},
        "spikes": {6, 14},
    },
    "alias_ambiguity": {
        "endpoint": (11, 10),
        "access": (6, 3),
        "deadline": 35,
        "downlink": 2,
        "outages": {4, 12, 20},
        "spikes": {9, 18},
    },
}
PATHS = {
    "short_path": (("key", "panel"), ((5, 2), (8, 5), (11, 10))),
    "overlap_paths": (("key", "panel", "dock"), ((6, 2), (8, 6), (11, 10))),
    "weak_uplink": (("key", "panel"), ((6, 4), (9, 7), (11, 10))),
    "late_feedback": (("key", "panel"), ((6, 2), (8, 5), (10, 8))),
    "alias_ambiguity": (("key", "panel"), ((6, 3), (9, 7), (11, 10))),
}
PATH_ALTERNATES = {
    "short_path": (
        (("valve", "parcel"), ((8, 2), (9, 6), (11, 10))),
        (("switch", "dock"), ((7, 5), (9, 8), (11, 10))),
    ),
    "overlap_paths": (
        (("key", "valve", "dock"), ((7, 2), (9, 6), (11, 10))),
        (("switch", "panel", "dock"), ((6, 4), (8, 7), (11, 10))),
    ),
    "weak_uplink": (
        (("valve", "parcel"), ((7, 4), (10, 7), (11, 10))),
        (("switch", "dock"), ((6, 6), (9, 8), (11, 10))),
    ),
    "late_feedback": (
        (("valve", "parcel"), ((7, 2), (9, 5), (10, 8))),
        (("switch", "dock"), ((6, 4), (8, 6), (10, 8))),
    ),
    "alias_ambiguity": (
        (("valve", "parcel"), ((7, 3), (10, 7), (11, 10))),
        (("switch", "dock"), ((6, 5), (9, 8), (11, 10))),
    ),
}


def _paths(name: str):
    return (PATHS[name],) + PATH_ALTERNATES[name]


def _labels(name: str) -> tuple[str, ...]:
    if name == "overlap_paths":
        return ("access", "panel", "station", "valve", "switch", "parcel", "panel", "dock")
    if name == "alias_ambiguity":
        return ("access", "control", "control", "cargo", "station", "access", "panel", "dock")
    return ("key", "panel", "valve", "parcel", "dock", "switch", "panel", "dock")


def _label_table(name: str) -> tuple[tuple[str, ...], ...]:
    labels, cats = _labels(name), OBJECT_CATEGORIES[name]
    return tuple(tuple((labels[i],) + (cats[i],) * 3) for i in range(N_OBJECTS))


ATOM_SHARES = {
    "result_json": {
        "result_json.parse": 0.34,
        "result_json.root": 0.33,
        "result_json.scenarios": 0.33,
    },
    "result_evidence": {
        f"result_evidence.{x}": 0.1
        for x in (
            "scenario_shape",
            "trace_shape",
            "trace_fields",
            "object_fields",
            "uplink_fields",
            "edge_fields",
            "downlink_fields",
            "ordering",
            "message_links",
            "finite_values",
        )
    },
    "decision_legality": {
        f"decision_legality.{x}": 1 / 6
        for x in (
            "action_relations",
            "movement",
            "observation",
            "report_payload",
            "report_budget",
            "feedback_execution",
        )
    },
    "replay_validity": {
        f"replay_validity.{x}": 0.1
        for x in (
            "trace_state",
            "observations",
            "uplink_outcomes",
            "edge_knowledge",
            "edge_versions",
            "path_states",
            "downlink",
            "execution",
            "objects",
            "terminal",
        )
    },
}


class AtomVerdict:
    __slots__ = ("id", "passed", "detail", "share")

    def __init__(self, id: str, passed: bool, detail: str, share: float):
        self.id, self.passed, self.detail, self.share = id, bool(passed), detail, share


class EvalState:
    def __init__(self):
        self.official_ok = False
        self.official_message = ""
        self.data: Any = None
        self.parse_ok = False
        self.records = {}
        self.replayed = {}
        self.scores = {}
        self.diagnostics = {}
        self.metrics = None


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _all_finite(v: Any) -> bool:
    if isinstance(v, Mapping):
        return all(_all_finite(x) for x in v.values())
    if isinstance(v, list):
        return all(_all_finite(x) for x in v)
    return not isinstance(v, (int, float)) or _finite(v)


def _strict_json(raw: str) -> Any:
    def reject(x: str):
        raise ValueError(f"non-standard JSON constant {x}")

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out

    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _shape(v: Any, keys: set[str]) -> bool:
    return isinstance(v, Mapping) and set(v) == keys


def _id_num(v: Any, prefix: str) -> int | None:
    return (
        int(v[len(prefix) :])
        if isinstance(v, str) and re.fullmatch(prefix + r"(?:0|[1-9][0-9]*)", v)
        else None
    )


def _atom(group: str, name: str, ok: bool, detail: str):
    key = f"{group}.{name}"
    return AtomVerdict(key, ok, detail, ATOM_SHARES[group][key])


def _score_atoms(atoms: Sequence[AtomVerdict], expected: Mapping[str, float]) -> float:
    ids = [a.id for a in atoms]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("atom IDs do not match declared set")
    if any(not math.isfinite(v) or v <= 0 for v in expected.values()) or not math.isclose(
        sum(expected.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("invalid atom shares")
    return max(0.0, min(1.0, sum(a.share for a in atoms if a.passed)))


def _diagnose(atoms: Sequence[AtomVerdict]) -> str:
    bad = [a for a in atoms if not a.passed]
    return (
        "failed_atoms=[]"
        if not bad
        else "failed_atoms=["
        + ",".join(a.id for a in bad)
        + "]; "
        + " | ".join(f"{a.id}: {a.detail}" for a in bad)
    )


_diagnose_atoms = _diagnose


def _scenario_rows(data: Any) -> list[Mapping[str, Any]]:
    return (
        data.get("scenarios", [])
        if isinstance(data, Mapping) and isinstance(data.get("scenarios"), list)
        else []
    )


def _result_json_atoms(data: Any, parsed: bool):
    root = _shape(data, {"scenarios"})
    rows = data.get("scenarios") if root else None
    ok = isinstance(rows, list) and all(
        _shape(x, {"scenario", "robot_trace", "objects", "uplink", "edge_state", "downlink"})
        for x in rows
    )
    return [
        _atom("result_json", "parse", parsed, "results.json must be strict JSON"),
        _atom("result_json", "root", root, "root keys must be exactly ['scenarios']"),
        _atom("result_json", "scenarios", ok, "scenarios must contain declared containers"),
    ]


def _evidence_atoms(data: Any):
    rows = _scenario_rows(data)
    scenario_ok = len(rows) == 5 and all(isinstance(x, Mapping) for x in rows)
    trace_shape = scenario_ok and all(
        isinstance(row.get("robot_trace"), list) for row in rows if isinstance(row, Mapping)
    )
    trace_fields = objects_ok = uplink_ok = edge_ok = down_ok = order_ok = links_ok = finite_ok = True
    trace_keys = {
        "t",
        "position",
        "nearby_objects",
        "known_observations",
        "link_indicator",
        "report_units_used",
        "edge_version",
        "processed_uplink",
        "processed_downlink",
        "active_feedback",
        "execution_index",
        "action",
        "move",
        "target_object",
        "report_observations",
        "observation_result",
        "action_result",
    }
    object_keys = {
        "object",
        "position",
        "category",
        "view_labels",
        "first_observe_t",
        "observations",
        "delivered_observations",
        "confirmed_t",
    }
    uplink_keys = {
        "message_id",
        "send_t",
        "observation_ids",
        "facts",
        "units",
        "link_indicator",
        "status",
        "arrival_t",
        "retry_of",
    }
    edge_keys = {"t", "version", "confirmed_facts", "path_states", "selected_path", "feedback_id"}
    down_keys = {
        "feedback_id",
        "send_t",
        "arrival_t",
        "path_id",
        "required_categories",
        "route",
        "valid_until",
        "status",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        trace, objs, uplink, edge, down = (
            row.get(x) for x in ("robot_trace", "objects", "uplink", "edge_state", "downlink")
        )
        trace_fields &= isinstance(trace, list) and all(_shape(x, trace_keys) for x in trace)
        objects_ok &= isinstance(objs, list) and all(_shape(x, object_keys) for x in objs)
        uplink_ok &= isinstance(uplink, list) and all(_shape(x, uplink_keys) for x in uplink)
        edge_ok &= isinstance(edge, list) and all(_shape(x, edge_keys) for x in edge)
        down_ok &= isinstance(down, list) and all(_shape(x, down_keys) for x in down)
        if isinstance(trace, list):
            order_ok &= [x.get("t") for x in trace if isinstance(x, Mapping)] == list(
                range(HORIZON)
            )
        if isinstance(objs, list):
            order_ok &= [x.get("object") for x in objs if isinstance(x, Mapping)] == list(
                range(N_OBJECTS)
            )
        if isinstance(uplink, list):
            order_ok &= [x.get("message_id") for x in uplink if isinstance(x, Mapping)] == [
                f"u{i}" for i in range(len(uplink))
            ]
        if isinstance(down, list):
            order_ok &= len(down) <= 1
        for seq in (trace or [], objs or [], uplink or [], edge or [], down or []):
            if isinstance(seq, Mapping):
                seq = [seq]
            for x in seq:
                if isinstance(x, Mapping):
                    finite_ok &= _all_finite(x)
        for msg in uplink or []:
            if isinstance(msg, Mapping):
                ids, facts = msg.get("observation_ids"), msg.get("facts")
                links_ok &= (
                    isinstance(ids, list)
                    and isinstance(facts, list)
                    and 1 <= len(ids) <= 3
                    and msg.get("units") == len(ids)
                    and len(facts) == len(ids)
                )
    return [
        _atom("result_evidence", "scenario_shape", scenario_ok, "exactly five scenario containers"),
        _atom("result_evidence", "trace_shape", trace_shape, "trace arrays are present"),
        _atom("result_evidence", "trace_fields", trace_fields, "trace records have exact keys"),
        _atom("result_evidence", "object_fields", objects_ok, "object record keys"),
        _atom("result_evidence", "uplink_fields", uplink_ok, "uplink record keys"),
        _atom("result_evidence", "edge_fields", edge_ok, "edge record keys"),
        _atom("result_evidence", "downlink_fields", down_ok, "downlink record keys"),
        _atom("result_evidence", "ordering", order_ok, "canonical records and IDs ordered"),
        _atom("result_evidence", "message_links", links_ok, "message payload links and units"),
        _atom("result_evidence", "finite_values", finite_ok, "numeric fields finite"),
    ]


def _index_records(data: Any) -> dict[Hashable, Any]:
    out = {}
    for row in _scenario_rows(data):
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            continue
        n = row["scenario"]
        for k in ("trace", "objects", "uplink", "edge", "downlink"):
            out[(n, k)] = row.get(
                {
                    "trace": "robot_trace",
                    "objects": "objects",
                    "uplink": "uplink",
                    "edge": "edge_state",
                    "downlink": "downlink",
                }[k],
                [],
            )
    return out


def _nearby(pos: tuple[int, int], obj: int) -> bool:
    return abs(pos[0] - POSITIONS[obj][0]) + abs(pos[1] - POSITIONS[obj][1]) <= 1


def _link(name: str, pos: tuple[int, int]) -> str:
    a = SCENARIOS[name]["access"]
    return "strong" if abs(pos[0] - a[0]) + abs(pos[1] - a[1]) <= 4 else "weak"


def _message_outcome(name: str, send_t: int, link: str) -> tuple[str, int | None]:
    if link == "weak" and send_t in SCENARIOS[name]["outages"]:
        return "dropped", None
    arrival = (
        send_t + (1 if link == "strong" else 3) + (2 if send_t in SCENARIOS[name]["spikes"] else 0)
    )
    return ("delivered", arrival) if arrival <= 35 else ("late", arrival)


def _path_status(name: str, confirmed: set[str]):
    states, selected = [], None
    for i, (cats, _) in enumerate(_paths(name)):
        ok = all(c in confirmed for c in cats)
        states.append({"path_id": f"p{i}", "status": "confirmed" if ok else "open"})
        if ok and selected is None:
            selected = f"p{i}"
    return states, selected


def _replay_scenario(name: str, records: Mapping[Hashable, Any]) -> dict[str, Any]:
    trace = records.get((name, "trace"), [])
    messages = records.get((name, "uplink"), [])
    edges = records.get((name, "edge"), [])
    down = records.get((name, "downlink"), [])
    tb = {r.get("t"): r for r in trace if isinstance(r, Mapping)}
    mb = {m.get("message_id"): m for m in messages if isinstance(m, Mapping)}
    arrivals: dict[int, list[Mapping[str, Any]]] = {}
    for m in messages:
        if (
            isinstance(m, Mapping)
            and m.get("status") == "delivered"
            and _integer(m.get("arrival_t"))
        ):
            arrivals.setdefault(m["arrival_t"], []).append(m)
    pos = (1, 1)
    observed: dict[str, dict[str, Any]] = {}
    object_counts = [0] * N_OBJECTS
    candidates = {i: set(CATEGORIES) for i in range(N_OBJECTS)}
    processed: set[str] = set()
    duplicate_units = 0
    edge_version = 0
    confirmed: set[int] = set()
    selected = None
    feedback_arrival = None
    execution = 0
    complete_t = None
    errors: list[str] = []
    expected_edge = []
    expected_objects = [
        {"observations": [], "delivered_observations": [], "confirmed_t": None}
        for _ in range(N_OBJECTS)
    ]
    expected_messages = []
    for t in range(HORIZON):
        proc_ids = []
        for m in sorted(
            arrivals.get(t, []),
            key=lambda x: (
                _id_num(x.get("message_id"), "u")
                if _id_num(x.get("message_id"), "u") is not None
                else 10**9
            ),
        ):
            mid = m.get("message_id")
            proc_ids.append(mid)
            new = 0
            ids = m.get("observation_ids", []) if isinstance(m.get("observation_ids"), list) else []
            for oid in ids:
                if oid in processed:
                    duplicate_units += 1
                    continue
                obs = observed.get(oid)
                if obs is None:
                    errors.append(f"{name}.uplink[{mid}].observation_ids: unknown {oid}")
                    continue
                processed.add(oid)
                new += 1
                expected_objects[obs["object"]]["delivered_observations"].append(oid)
                candidates[obs["object"]] &= ALIASES.get(obs["raw_label"], set())
            if new:
                edge_version += 1
        for obj, cand in candidates.items():
            if len(cand) == 1 and obj not in confirmed:
                confirmed.add(obj)
                expected_objects[obj]["confirmed_t"] = t
        confirmed_cats = {next(iter(candidates[i])) for i in confirmed if len(candidates[i]) == 1}
        states, newly = _path_status(name, confirmed_cats)
        if selected is None and newly is not None:
            selected = newly
            feedback_arrival = t + SCENARIOS[name]["downlink"]
        row = tb.get(t)
        edge_row = {
            "t": t,
            "version": edge_version,
            "confirmed_facts": [
                {
                    "object": i,
                    "category": next(iter(candidates[i])),
                    "position": list(POSITIONS[i]),
                    "confirmed_t": expected_objects[i]["confirmed_t"],
                }
                for i in sorted(confirmed)
                if len(candidates[i]) == 1
            ],
            "path_states": states,
            "selected_path": selected,
            "feedback_id": "d0" if selected is not None else None,
        }
        expected_edge.append(edge_row)
        active = (
            "d0"
            if selected is not None
            and feedback_arrival is not None
            and feedback_arrival <= t <= SCENARIOS[name]["deadline"]
            and complete_t is None
            else None
        )
        known_before = list(observed)
        before = pos
        nearby = [i for i in range(N_OBJECTS) if _nearby(pos, i)]
        link = _link(name, pos)
        pre_exec = execution
        units_before = sum(
            len(m.get("observation_ids", []))
            for m in messages
            if isinstance(m, Mapping) and _integer(m.get("send_t")) and m.get("send_t") < t
        )
        if not isinstance(row, Mapping):
            errors.append(f"{name}.robot_trace[{t}]: missing")
            continue
        expected = {
            "t": t,
            "position": list(before),
            "nearby_objects": nearby,
            "known_observations": known_before,
            "link_indicator": link,
            "report_units_used": units_before,
            "edge_version": edge_version,
            "processed_uplink": proc_ids,
            "processed_downlink": ["d0"] if feedback_arrival == t and selected is not None else [],
            "active_feedback": active,
            "execution_index": pre_exec,
        }
        for k, want in expected.items():
            if row.get(k) != want:
                errors.append(
                    f"{name}.robot_trace[{t}].{k}: expected {want!r}, actual {row.get(k)!r}"
                )
        action, move, target, payload = (
            row.get("action"),
            row.get("move"),
            row.get("target_object"),
            row.get("report_observations"),
        )
        result = {"status": "invalid", "reference": None}
        obs_result = None
        legal = True
        if action not in ACTIONS or move not in MOVES:
            legal = False
        if action != "move" and move != "stay":
            legal = False
        if action == "move" and (target is not None or payload != []):
            legal = False
        if action == "observe" and payload != []:
            legal = False
        if action == "wait" and (target is not None or payload != []):
            legal = False
        if action == "report" and target is not None:
            legal = False
        new_pos = before
        if action == "move" and legal:
            d = MOVES[move]
            candidate = (before[0] + d[0], before[1] + d[1])
            if not (0 <= candidate[0] < GRID_W and 0 <= candidate[1] < GRID_H):
                legal = False
            else:
                new_pos = candidate
                result = {"status": "moved" if move != "stay" else "none", "reference": None}
        elif action == "observe" and legal:
            if not (
                _integer(target)
                and 0 <= target < N_OBJECTS
                and _nearby(before, target)
                and object_counts[target] < MAX_OBSERVATIONS
            ):
                legal = False
            else:
                oid = f"o{len(observed)}"
                raw = _label_table(name)[target][object_counts[target]]
                object_counts[target] += 1
                obs_result = {
                    "observation_id": oid,
                    "object": target,
                    "t": t,
                    "position": list(POSITIONS[target]),
                    "raw_label": raw,
                }
                observed[oid] = obs_result
                expected_objects[target]["observations"].append(obs_result)
                result = {"status": "observed", "reference": oid}
        elif action == "report" and legal:
            legal = (
                isinstance(payload, list)
                and 1 <= len(payload) <= 3
                and payload
                == sorted(
                    set(payload),
                    key=lambda x: _id_num(x, "o") if _id_num(x, "o") is not None else 10**9,
                )
                and all(x in known_before for x in payload)
                and units_before + len(payload) <= REPORT_BUDGET
                and sum(
                    1
                    for old in messages
                    if isinstance(old, Mapping)
                    and old.get("send_t") is not None
                    and t - REPORT_WINDOW + 1 <= old.get("send_t", -999) < t
                )
                < REPORT_WINDOW_LIMIT
            )
            if legal:
                mid = f"u{len(expected_messages)}"
                status, arrival = _message_outcome(name, t, link)
                same = [
                    m["message_id"] for m in expected_messages if m["observation_ids"] == payload
                ]
                retry = same[-1] if same else None
                msg = {
                    "message_id": mid,
                    "send_t": t,
                    "observation_ids": list(payload),
                    "facts": [observed[x] for x in payload],
                    "units": len(payload),
                    "link_indicator": link,
                    "status": status,
                    "arrival_t": arrival,
                    "retry_of": retry,
                }
                expected_messages.append(msg)
                result = {"status": "reported", "reference": mid}
        elif action == "wait" and legal:
            result = {"status": "none", "reference": None}
        if not legal:
            errors.append(
                f"{name}.robot_trace[{t}].action: illegal action fields or decision "
                f"action={action!r}, move={move!r}, target_object={target!r}, "
                f"report_observations={payload!r}"
            )
            result = {"status": "invalid", "reference": None}
            obs_result = None
        pos = new_pos
        if active and selected is not None:
            route = _paths(name)[int(selected[1:])][1]
            if execution < len(route) and pos == route[execution]:
                execution += 1
                complete_t = t if execution == len(route) else complete_t
        if row.get("observation_result") != obs_result:
            errors.append(
                f"{name}.robot_trace[{t}].observation_result: expected {obs_result!r}, "
                f"actual {row.get('observation_result')!r}"
            )
        if row.get("action_result") != result:
            errors.append(
                f"{name}.robot_trace[{t}].action_result: expected {result!r}, "
                f"actual {row.get('action_result')!r}"
            )
        if (
            action == "report"
            and legal
            and row.get("report_observations") != expected_messages[-1]["observation_ids"]
        ):
            errors.append(f"{name}.robot_trace[{t}].report_observations: payload mismatch")
        if edges and isinstance(edges, list) and t < len(edges) and edges[t] != edge_row:
            errors.append(f"{name}.edge_state[{t}]: expected {edge_row!r}, actual {edges[t]!r}")
    expected_down = []
    if selected is not None and feedback_arrival is not None:
        cats, route = _paths(name)[int(selected[1:])]
        status = (
            "completed"
            if complete_t is not None and complete_t <= SCENARIOS[name]["deadline"]
            else ("late" if feedback_arrival > SCENARIOS[name]["deadline"] else "expired")
        )
        expected_down = [
            {
                "feedback_id": "d0",
                "send_t": feedback_arrival - SCENARIOS[name]["downlink"],
                "arrival_t": feedback_arrival,
                "path_id": selected,
                "required_categories": list(cats),
                "route": [list(p) for p in route],
                "valid_until": SCENARIOS[name]["deadline"],
                "status": status,
            }
        ]
    if down != expected_down:
        errors.append(f"{name}.downlink: expected {expected_down!r}, actual {down!r}")
    actual_msgs = [mb.get(f"u{i}") for i in range(len(messages))]
    if actual_msgs != messages:
        errors.append(f"{name}.uplink: IDs/order malformed")
    if len(messages) != len(expected_messages):
        errors.append(
            f"{name}.uplink: expected {len(expected_messages)} report messages, "
            f"actual {len(messages)}"
        )
    for i, want in enumerate(expected_messages):
        if i >= len(messages) or messages[i] != want:
            errors.append(
                f"{name}.uplink[u{i}]: expected {want!r}, "
                f"actual {messages[i] if i < len(messages) else None!r}"
            )
    for i in range(N_OBJECTS):
        expected_objects[i]["observations"] = expected_objects[i]["observations"]
        expected_objects[i]["delivered_observations"].sort(
            key=lambda oid: _id_num(oid, "o") if _id_num(oid, "o") is not None else 10**9
        )
    return {
        "errors": errors,
        "complete_t": complete_t,
        "finish_t": complete_t,
        "edge": expected_edge,
        "objects": expected_objects,
        "downlink": expected_down,
        "messages": expected_messages,
        "trace": tb,
        "report_units": sum(
            len(m.get("observation_ids", [])) for m in messages if isinstance(m, Mapping)
        ),
        "repeat_obs": sum(max(0, len(x["observations"]) - 1) for x in expected_objects),
        "duplicate_units": duplicate_units,
    }


def _coverage_analysis(data: Any):
    defects = []
    rows = _scenario_rows(data)
    names = [x.get("scenario") for x in rows if isinstance(x, Mapping)]
    if names != list(SCENARIO_NAMES):
        defects.append(f"scenario order: expected {list(SCENARIO_NAMES)!r}, actual {names!r}")
    for n in SCENARIO_NAMES:
        if names.count(n) != 1:
            defects.append(f"scenario {n}: count={names.count(n)}")
    for n in names:
        if n not in SCENARIO_NAMES:
            defects.append(f"unknown scenario key {n!r}")
    for n in SCENARIO_NAMES:
        row = next((x for x in rows if isinstance(x, Mapping) and x.get("scenario") == n), None)
        if row is None:
            continue
        for field, key, expected in (
            ("robot_trace", "t", set(range(HORIZON))),
            ("objects", "object", set(range(N_OBJECTS))),
            ("edge_state", "t", set(range(HORIZON))),
        ):
            seq = row.get(field)
            vals = (
                [x.get(key) for x in seq if isinstance(x, Mapping)] if isinstance(seq, list) else []
            )
            if set(vals) != expected or len(vals) != len(expected):
                defects.append(
                    f"{n}.{field}: expected unique keys {sorted(expected)}, actual {vals}"
                )
    return max(0.0, (4.0 - 0.5 * len(defects)) / 4.0), defects


_coverage = _coverage_analysis


def _legality_atoms(records: Mapping[Hashable, Any]):
    flags = {
        x: True
        for x in (
            "action_relations",
            "movement",
            "observation",
            "report_payload",
            "report_budget",
            "feedback_execution",
        )
    }
    details = {x: "constraint satisfied" for x in flags}
    for n in SCENARIO_NAMES:
        trace = records.get((n, "trace"), [])
        pos = (1, 1)
        known = []
        units = 0
        report_slots = []
        previous_execution = 0
        previous_active = None
        for row in trace if isinstance(trace, list) else []:
            if not isinstance(row, Mapping):
                continue
            t, action, move, target, payload = (
                row.get("t"),
                row.get("action"),
                row.get("move"),
                row.get("target_object"),
                row.get("report_observations"),
            )
            if action not in ACTIONS or move not in MOVES:
                flags["action_relations"] = False
                details["action_relations"] = f"{n}.t={t}: invalid action/move"
            if (
                (action != "move" and move != "stay")
                or (action == "move" and target is not None)
                or (action != "report" and payload != [])
                or (action == "wait" and target is not None)
                or (action == "report" and target is not None)
            ):
                flags["action_relations"] = False
                details["action_relations"] = f"{n}.t={t}: action fields disagree"
            if action == "move" and move in MOVES and move != "stay":
                p = (pos[0] + MOVES[move][0], pos[1] + MOVES[move][1])
                if not (0 <= p[0] < GRID_W and 0 <= p[1] < GRID_H):
                    flags["movement"] = False
                    details["movement"] = f"{n}.t={t}: move leaves grid"
                else:
                    pos = p
            if action == "observe" and not (
                _integer(target) and 0 <= target < N_OBJECTS and _nearby(pos, target)
            ):
                flags["observation"] = False
                details["observation"] = f"{n}.t={t}: target {target!r} not nearby at {pos}"
            if action == "report":
                valid = (
                    isinstance(payload, list)
                    and 1 <= len(payload) <= 3
                    and payload
                    == sorted(
                        set(payload),
                        key=lambda x: _id_num(x, "o") if _id_num(x, "o") is not None else 10**9,
                    )
                    and all(x in known for x in payload)
                )
                if not valid:
                    flags["report_payload"] = False
                    details["report_payload"] = f"{n}.t={t}: payload {payload!r}, known={known!r}"
                units += len(payload) if isinstance(payload, list) else 0
                report_slots.append(t)
            if units > REPORT_BUDGET:
                flags["report_budget"] = False
                details["report_budget"] = f"{n}.t={t}: units={units} > {REPORT_BUDGET}"
            if (
                _integer(t)
                and sum(1 for x in report_slots if t - 3 <= x <= t) > REPORT_WINDOW_LIMIT
            ):
                flags["report_budget"] = False
                details["report_budget"] = (
                    f"{n}.t={t}: rolling reports exceed {REPORT_WINDOW_LIMIT}"
                )
            execution_index = row.get("execution_index")
            active_feedback = row.get("active_feedback")
            if (
                not _integer(execution_index)
                or execution_index < previous_execution
                or execution_index - previous_execution > 1
                or execution_index > 3
                or (
                    execution_index > previous_execution
                    and active_feedback != "d0"
                    and previous_active != "d0"
                )
            ):
                flags["feedback_execution"] = False
                details["feedback_execution"] = (
                    f"{n}.t={t}: execution_index={execution_index!r}, "
                    f"previous={previous_execution}, active={active_feedback!r}"
                )
            if _integer(execution_index):
                previous_execution = execution_index
            previous_active = active_feedback
            obs = row.get("observation_result")
            if isinstance(obs, Mapping) and obs.get("observation_id") not in known:
                known.append(obs.get("observation_id"))
    return [_atom("decision_legality", k, flags[k], details[k]) for k in flags]


def _replay_atoms(records: Mapping[Hashable, Any]):
    replayed = {n: _replay_scenario(n, records) for n in SCENARIO_NAMES}
    flags = {
        x: True
        for x in (
            "trace_state",
            "observations",
            "uplink_outcomes",
            "edge_knowledge",
            "edge_versions",
            "path_states",
            "downlink",
            "execution",
            "objects",
            "terminal",
        )
    }
    details = {x: "replay matched" for x in flags}
    for n, rep in replayed.items():
        if rep["errors"]:
            for error in rep["errors"]:
                affected = {"trace_state"}
                if ".observation_result" in error or ".action" in error:
                    affected.add("observations")
                if ".execution_index" in error or ".active_feedback" in error:
                    affected.add("execution")
                if ".uplink" in error:
                    affected.update({"uplink_outcomes", "edge_knowledge"})
                if ".edge_state" in error:
                    affected.update({"edge_versions", "path_states"})
                if ".downlink" in error:
                    affected.add("downlink")
                if ".objects" in error:
                    affected.add("objects")
                for flag in affected:
                    flags[flag] = False
                    if details[flag] == "replay matched":
                        details[flag] = error
        if records.get((n, "edge"), []) != rep["edge"]:
            flags["edge_versions"] = flags["path_states"] = False
            details["edge_versions"] = f"{n}.edge_state: full snapshot mismatch"
        if records.get((n, "objects"), []) != [
            {
                "object": i,
                "position": list(POSITIONS[i]),
                "category": OBJECT_CATEGORIES[n][i],
                "view_labels": list(_label_table(n)[i]),
                "first_observe_t": (x["observations"][0]["t"] if x["observations"] else None),
                "observations": x["observations"],
                "delivered_observations": x["delivered_observations"],
                "confirmed_t": x["confirmed_t"],
            }
            for i, x in enumerate(rep["objects"])
        ]:
            flags["objects"] = False
            details["objects"] = f"{n}.objects: static or observation fields mismatch"
        if records.get((n, "downlink"), []) != rep["downlink"]:
            flags["downlink"] = False
            details["downlink"] = (
                f"{n}.downlink: expected {rep['downlink']!r}, "
                f"actual {records.get((n, 'downlink'), [])!r}"
            )
        if rep["complete_t"] is not None and rep["complete_t"] > SCENARIOS[n]["deadline"]:
            flags["terminal"] = False
            details["terminal"] = f"{n}: completion {rep['complete_t']} after deadline"
    return [_atom("replay_validity", k, flags[k], details[k]) for k in flags], replayed


def _metrics(replayed: Mapping[str, Any]):
    return {
        "primary_metrics_1": sum(
            r["complete_t"] is not None and r["complete_t"] <= SCENARIOS[n]["deadline"]
            for n, r in replayed.items()
        )
        / 5,
        "primary_metrics_2": sum(
            max(0.0, 1.0 - r["complete_t"] / SCENARIOS[n]["deadline"])
            if r["complete_t"] is not None and r["complete_t"] <= SCENARIOS[n]["deadline"]
            else 0.0
            for n, r in replayed.items()
        )
        / 5,
        "auxiliary_metrics_1": sum(
            max(0.0, min(1.0, 1.0 - r["report_units"] / REPORT_BUDGET)) for r in replayed.values()
        )
        / 5,
        "auxiliary_metrics_2": sum(
            0.5 * max(0.0, min(1.0, 1.0 - r["repeat_obs"] / 24))
            + 0.5 * max(0.0, min(1.0, 1.0 - r["duplicate_units"] / 24))
            for r in replayed.values()
        )
        / 5,
    }


def _gated_metrics(state: EvalState):
    if not state.official_ok or any(
        state.scores.get(k, 0.0) < 1.0 - 1e-9
        for k in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        )
    ):
        return None
    return _metrics(state.replayed)


def _regular_file(ws: Workspace, rel: str) -> bool:
    p = Path(ws.root) / rel
    return p.is_file() and not p.is_symlink()


def _headings(text: str):
    lines = text.splitlines()
    parsed = []
    heading_lines = set()
    fence_lines = set()
    fence_char = None
    fence_length = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_char is not None:
            if fence:
                marker = fence.group(1)
                if marker[0] == fence_char and len(marker) >= fence_length and not fence.group(2).strip():
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
                level = 1 if underline.group(1)[0] == "=" else 2
                parsed.append((i, i + 1, level, title_match.group(1).strip()))
                heading_lines.update((i, i + 1))
                i += 2
                continue
        i += 1
    found = {}
    for index, (start, end, level, title) in enumerate(parsed):
        stop = len(lines)
        for next_start, _, next_level, _ in parsed[index + 1 :]:
            if next_level <= level:
                stop = next_start
                break
        body = "\n".join(
            lines[j] for j in range(end + 1, stop)
            if j not in heading_lines and j not in fence_lines
        ).strip()
        if title not in found or not found[title]:
            found[title] = body
    return found


def _report_score(ws: Workspace):
    required = {
        "Problem Formulation",
        "Method Design",
        "Experimental Setup",
        "Results",
        "Research Analysis",
        "Conclusion and Limitations",
    }
    if not _regular_file(ws, "report.md"):
        return 0.0, "report.md: expected regular file"
    try:
        found = _headings(ws.read_text("report.md") or "")
    except OSError as exc:
        return 0.0, f"report.md: read failed: {exc}"
    missing = sorted(x for x in required if not found.get(x))
    return max(0.0, (2 - 0.5 * len(missing)) / 2), f"missing_or_empty={missing}"


_CACHE: dict[str, EvalState] = {}


async def _build_state(ws: Workspace):
    key = str(Path(ws.root).resolve())
    if key in _CACHE:
        return _CACHE[key]
    s = EvalState()
    start = time.monotonic()
    remove_ok = False
    try:
        removed_ok = (
            getattr(
                await ws.run("rm -f -- results.json", check=False, timeout=30.0), "exit_code", None
            )
            == 0
        )
        absent = await ws.run(
            "test ! -e results.json && test ! -L results.json", check=False, timeout=30.0
        )
        remove_ok = removed_ok and getattr(absent, "exit_code", None) == 0
        result = await ws.run("python run.py", check=False, timeout=600.0)
        exit_code = getattr(result, "exit_code", None)
        elapsed = time.monotonic() - start
        s.official_ok = (
            remove_ok
            and exit_code == 0
            and elapsed <= 600 + EPS
            and _regular_file(ws, "results.json")
        )
        s.official_message = (
            f"exit={exit_code}, elapsed={elapsed:.3f}s, "
            f"new_results={_regular_file(ws, 'results.json')}"
        )
        if _regular_file(ws, "results.json"):
            try:
                s.data = _strict_json(ws.read_text("results.json") or "")
                s.parse_ok = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                s.diagnostics["result_json"] = f"strict parse failed: {exc}"
        ja = _result_json_atoms(s.data, s.parse_ok)
        s.scores["result_json"] = _score_atoms(ja, ATOM_SHARES["result_json"])
        s.diagnostics["result_json"] = s.diagnostics.get("result_json", _diagnose(ja))
        if not s.official_ok or not s.parse_ok:
            for cid in (
                "result_evidence",
                "canonical_coverage",
                "decision_legality",
                "replay_validity",
            ):
                s.scores[cid] = 0.0
                s.diagnostics[cid] = "official run or strict parse failed"
        else:
            s.records = _index_records(s.data)
            ea = _evidence_atoms(s.data)
            s.scores["result_evidence"] = _score_atoms(ea, ATOM_SHARES["result_evidence"])
            s.diagnostics["result_evidence"] = _diagnose(ea)
            cov, defects = _coverage_analysis(s.data)
            s.scores["canonical_coverage"] = cov
            s.diagnostics["canonical_coverage"] = f"defect_count={len(defects)}; {defects[:8]}"
            la = _legality_atoms(s.records)
            s.scores["decision_legality"] = _score_atoms(la, ATOM_SHARES["decision_legality"])
            s.diagnostics["decision_legality"] = _diagnose(la)
            ra, s.replayed = _replay_atoms(s.records)
            s.scores["replay_validity"] = _score_atoms(ra, ATOM_SHARES["replay_validity"])
            s.diagnostics["replay_validity"] = _diagnose(ra)
            s.metrics = _gated_metrics(s)
    except Exception as exc:
        detail = f"grader validation failed: {type(exc).__name__}: {exc}"
        s.official_ok = False
        for cid in (
            "result_json",
            "result_evidence",
            "canonical_coverage",
            "decision_legality",
            "replay_validity",
        ):
            s.scores.setdefault(cid, 0.0)
            s.diagnostics.setdefault(cid, detail)
    _CACHE[key] = s
    return s


@check(id="entry_point")
async def entry_point(ws: Workspace):
    if not _regular_file(ws, "run.py"):
        return 0.0, "run.py: expected regular file"
    r = await ws.run("python -m py_compile run.py", check=False, timeout=30.0)
    return (
        (1.0, "static compilation passed")
        if getattr(r, "exit_code", None) == 0
        else (0.0, "static compilation failed")
    )


@check(id="official_run")
async def official_run(ws: Workspace):
    s = await _build_state(ws)
    return (1.0 if s.official_ok else 0.0), s.official_message


@check(id="report_contract")
async def report_contract(ws: Workspace):
    return _report_score(ws)


async def _score(ws: Workspace, cid: str):
    s = await _build_state(ws)
    return s.scores.get(cid, 0.0), s.diagnostics.get(cid, "")


@check(id="result_json")
async def result_json(ws: Workspace):
    return await _score(ws, "result_json")


@check(id="result_evidence")
async def result_evidence(ws: Workspace):
    return await _score(ws, "result_evidence")


@check(id="canonical_coverage")
async def canonical_coverage(ws: Workspace):
    return await _score(ws, "canonical_coverage")


@check(id="decision_legality")
async def decision_legality(ws: Workspace):
    return await _score(ws, "decision_legality")


@check(id="replay_validity")
async def replay_validity(ws: Workspace):
    return await _score(ws, "replay_validity")


for _metric in (
    "primary_metrics_1",
    "primary_metrics_2",
    "auxiliary_metrics_1",
    "auxiliary_metrics_2",
):

    async def _metric_check(ws: Workspace, _name=_metric):
        s = await _build_state(ws)
        return (s.metrics or {}).get(
            _name, 0.0
        ), "validity gate passed" if s.metrics else "validity gate failed; metric=0"

    globals()[_metric] = check(id=_metric)(_metric_check)
del _metric_check
