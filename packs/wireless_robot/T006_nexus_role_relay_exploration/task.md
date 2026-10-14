---
id: T006_nexus_role_relay_exploration
name: Adaptive Explorer–Messenger Coordination for Communication-Constrained Grid Exploration
labels:
  domain: research
  subdomain: wireless_robot
  model_capability: [reasoning, code, math, domain_knowledge, planning]
  harness_focus: [instructions, loop, context]
  complexity: long_horizon
  multimodal: false
  online: true
---

## Prompt

### Scenario

Four ground robots explore an unknown ten-by-ten indoor grid without fixed wireless infrastructure. Robots expand local coverage, but knowledge of newly observed cells must travel over short-range intermittent links before it can prevent another robot from repeating the same work. A robot can act as an `explorer`, which observes a small neighbourhood while moving, or as a `messenger`, which observes only its final cell but can forward a bounded summary of known facts. Changing roles consumes a slot, and a role cannot be changed again immediately.

In each discrete slot, messages that arrived from the previous slot are merged, every robot chooses its role and a move, eligible messengers choose a bounded fact summary to broadcast, and all moves then occur simultaneously. Obstacles and distance determine which broadcasts reach another robot. Messages can be lost because no usable link exists, and a successful message is available only at the next slot. The resulting positions, information freshness and explored area depend jointly on movement, role opportunity cost, link geometry, transmission limits and the history of received facts. A poor policy can leave regions unexplored, repeatedly explore already known cells, or spend its communication budget without restoring useful knowledge.

### Research Problem

Develop a self-contained, non-training simulation and a unified state-driven policy for the coupled exploration and communication problem. At every slot, for each robot, decide its role, one four-neighbour move or stay action, whether to broadcast, and which currently known facts to put in the broadcast summary. The policy must use only the robot's position, current role and role age, local knowledge, messages that have arrived, link outcomes already observed and remaining resource state. It must respect role-switch occupancy, role dwell time, collision rules, per-robot broadcast budget and sliding-window limit.

The main objective is to maximize final exploration coverage while keeping shared knowledge fresh. Communication is not free: a messenger may spend a slot forwarding facts instead of exploring, and role changes and broadcasts are limited. Explain how the policy balances marginal coverage, the need to reconnect separated robots, summary selection, stale knowledge and switching cost. Use one reusable decision procedure for all canonical maps; do not supply actions or result tables keyed to map names or absolute slots.

### Experimental Setting and Constraints

#### Canonical evaluation units and grid model

There are five canonical scenarios, each evaluated for slots `t=0,...,31`. Coordinates are integer `(x,y)` with `x` increasing eastward and `y` increasing northward, both in `0,...,9`. The map rows below are listed from `y=9` down to `y=0`; `.` is a free cell and `#` is an obstacle. A cell's canonical integer ID is `cell = x + 10*y`. The four initial positions are robot `0..3` at `[[1,1],[1,8],[8,1],[8,8]]` in every scenario.

| scenario | rows from `y=9` to `y=0` |
|---|---|
| `corridor` | `....##....` / `....##....` / `....##....` / `....##....` / `..........` / `....##....` / `....##....` / `....##....` / `....##....` / `....##....` |
| `rooms` | `...#..#...` / `...#..#...` / `..........` / `...#..#...` / `...#..#...` / `...#..#...` / `...#..#...` / `..........` / `...#..#...` / `...#..#...` |
| `crossing` | `....##....` / `....##....` / `....##....` / `....##....` / `..........` / `..........` / `....##....` / `....##....` / `....##....` / `....##....` |
| `dead_end` | `..........` / `..........` / `....##....` / `....##....` / `....##....` / `.#..#...#.` / `....##....` / `....##....` / `....##....` / `....##....` |
| `open_field` | `..........` / `..#.......` / `.......#..` / `..........` / `....#.....` / `.....#....` / `..........` / `..#.......` / `.......#..` / `..........` |

The fixed map is environment truth. A policy may not inspect obstacle cells or future link outcomes unless it has obtained the corresponding fact through an observation or an arrived message.

#### State, actions and observations

Every robot starts as an `explorer` with role age `1` and knows only its own initial free cell as a fact with version `0`. A fact is an object `{cell,state,version}`, where `state` is exactly `free` or `obstacle`, and `version` is the slot in which that observation was made. An explorer may choose `N`, `S`, `E`, `W` or `stay`; after its move it observes every grid cell at Manhattan distance at most `1` from its final position, including obstacle cells. A messenger may choose the same move alphabet but observes only its final cell. Observations update the robot's local fact map with the new version. The global explored set contains every free cell observed by at least one robot.

#### Information and communication event order

At the beginning of slot `t`, all messages scheduled for `t` are delivered and merged before any new decision. A receiver keeps, for each cell, the fact with the larger version; equal versions do not replace an existing fact. A link from sender `i` to receiver `j` at slot `t` exists when their Manhattan distance is at most `3` and at least one shortest Manhattan path between them contains no obstacle. The sender is not its own receiver. A broadcast creates one event with all linked receiver IDs as `targets`; delivery occurs at `t+1`, so events with `t+1>31` have no effect on the task. Receivers not in `targets` do not receive that event. A `delivered` event has a non-empty `delivered_to` list exactly equal to `targets`; an event with no target is `lost`. The event order is: arrivals and knowledge merge, role and move choice, role-switch and broadcast processing, simultaneous move, role-specific observation, then coverage accounting. A policy cannot use the current slot's observations or a message scheduled for `t+1` to choose the same slot's action. Link targets are evaluated during the broadcast-processing step from the positions at the beginning of slot `t` (the same positions from which simultaneous moves are proposed); the move in slot `t` does not change the targets of that event.

Within the records, `role`, `move`, `broadcast`, and `broadcast_facts` are decisions submitted at the slot boundary; `received_event_ids` records arrivals available before that decision and `observed_facts` records observations produced after movement. The `events`, `sync`, and `coverage` arrays are post-event evidence. This categorization is only a record interpretation; the event order above is the authoritative definition whenever a field's timing is relevant.

#### Legal actions and boundary conditions

At a slot, a role change is legal only if the previous role has been held for at least four slots. The changing robot must use `stay` and must not broadcast in that slot. Its role age is `1`; if the role does not change, the age increases by one. A messenger that does not change role may broadcast at most twelve facts from its current local map. An explorer cannot broadcast. Each robot may broadcast at most twenty times over the horizon, and in every consecutive four-slot window it may broadcast at most twice. A broadcast summary must contain distinct currently known facts and its length is between zero and twelve. A false `broadcast` flag must have an empty summary. The `role_age` recorded for slot `t` is the age of the role used in that slot after applying any role-change decision: it is `1` in the initial `t=0` record and in a switch record, and otherwise is the previous record's age plus one.

Moves are proposed simultaneously from the positions at the beginning of the slot. A move outside the grid or into an obstacle is invalid. If two robots propose the same destination, or two robots propose to exchange positions, all robots in that conflict stay in place; other legal moves occur. The position recorded for slot `t` is the position before that slot's move.

#### Task metrics

The official metrics are computed from independent replay across the five scenarios:

* `final_coverage` is the mean, over scenarios, of final explored free cells divided by the number of  free cells in that map.
* `knowledge_freshness` is the mean over all scenario-slot-robot pairs and all free cells in the  global explored set at that slot. If robot `r` acquired the latest fact for cell `c` at slot `k`,  its contribution at slot `t` is `max(0, 1-(t-k)/4)`; an unknown cell contributes `0`.
* `communication_efficiency` is the mean over scenarios of `clip(1-n_tx/80,0,1)`, where `n_tx` is  the number of broadcast slots and `80` is the four-robot total budget.
* `role_stability` is the mean over scenarios of `clip(1-n_switch/32,0,1)`, where `n_switch` is the  number of role changes and `32` is the fixed horizon-based upper bound.

The normalized task performance is `(8*final_coverage + 7*knowledge_freshness + 3*communication_efficiency + 2*role_stability)/20`. All four values are computed from replay and lie in `[0,1]`; higher is better. For author-side calibration records, this aggregate is named `normalized_task_performance`; it is reported from the four independently replayed metrics and is not an additional Auto Check item.

#### Result contract

Running `python run.py` must create or overwrite UTF-8 `results.json` in the project root. Its root object has exactly one key, `scenarios`, whose value is an array in the canonical order `["corridor","rooms","crossing","dead_end","open_field"]`. Each scenario object has exactly the keys `scenario`, `robots`, `events`, `coverage` and `sync`.

`robots` is a flat array with exactly `4*32` records, one for every `(robot,t)` key. Each record has exactly these keys: `robot` (integer `0..3`), `t` (integer `0..31`), `position` (two integer coordinates), `role` (`explorer` or `messenger`), `role_age` (positive integer), `move` (`N`, `S`, `E`, `W` or `stay`), `broadcast` (boolean), `broadcast_facts` (a list of at most twelve distinct fact objects), `observed_facts` (a list of fact objects sorted by increasing `cell`) and `received_event_ids` (a lexicographically sorted list of strings). The `broadcast_facts` order is preserved as the sender's chosen summary order. The position is the beginning-of-slot position. Fact objects have exactly `cell` (integer `0..99`), `state` (`free` or `obstacle`) and `version` (integer `0..31`).

`events` is a list with at most one event per robot and slot. Each event has exactly `event_id` (the string `"r{robot}_t{t}"`), `sender` (integer), `send_t` (integer), `facts` (the exact broadcast summary), `targets` (a list of distinct receiver IDs in increasing numeric order), `delivered_to` (the same increasing list as `targets`), `arrival_t` (integer `send_t+1`) and `status` (`delivered` iff `delivered_to` is non-empty, otherwise `lost`). Because link evaluation is deterministic, `delivered_to` must equal `targets` exactly; there is no partial delivery. `sync` contains one record for each delivered receiver and has exactly `event_id`, `sender`, `receiver`, `arrival_t` and `facts`, where `facts` are the facts actually adopted from that event in event-summary order. `coverage` contains exactly 32 records `{t,cells}`, with sorted distinct free-cell IDs known to at least one robot after observations in slot `t`.

All arrays and keys described above are exact. Numeric comparisons use absolute tolerance `1e-6`. The grader independently derives link targets, message arrival, knowledge merge, role ages, movement conflicts, observations, coverage and resource use from the role/move/broadcast records; self-reported aggregate values or report numbers are never scoring inputs.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md` and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` without interactive input. `report.md` must contain non-empty Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. The report must describe the actual implementation, results, communication-control trade-offs and evidence-supported limitations.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json is strict JSON with the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "robot records, events, coverage and sync provide typed evidence for the replay", weight: 4}
- {id: canonical_coverage, desc: "all scenarios, robot-slot records and coverage slots are present exactly once", weight: 4}
- {id: decision_legality, desc: "roles, moves, broadcasts, budgets, windows and information-bound summaries are legal", weight: 4}
- {id: replay_validity, desc: "independent event, movement, observation, merge and coverage replay matches the evidence", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed final exploration coverage", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed shared-knowledge freshness", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed communication efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed role stability", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the grid, role, communication, information-boundary, state-transition and objective model"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given or the role, grid or message model is materially wrong; Implementation/evidence: no executable representation or recorded evidence corresponds to the claimed process."}
    - {score: 0.4, desc: "Report: only part of the robots, observations, links, role constraints or causal relations is defined, with a decisive omission; Implementation/evidence: the code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: the core grid, role abilities, information boundary, event order, movement constraints and coverage/freshness objectives are accurate; Implementation/evidence: code and records implement the core model across all canonical units, with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely specifies role opportunity cost, shortest-path link condition, message delay, merging, collision resolution, budgets and coverage-freshness trade-off; Implementation/evidence: code and records faithfully realize these relations with traceable condition-level evidence."}
    
- id: method_design
  desc: "quality and task relevance of the unified non-training role, movement and summary policy"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, or it violates the information boundary or hard constraints; Implementation/evidence: no runnable unified procedure or supporting decisions are present."}
    - {score: 0.4, desc: "Report: the method ignores decisive role, link, freshness or budget coupling, or uses map/slot-specific action tables; Implementation/evidence: submitted decisions rely on pre-filled schedules or materially differ from the described method."}
    - {score: 0.7, desc: "Report: a legal reusable policy derives role, move and summary choices from local state and history across all units; Implementation/evidence: one procedure executes everywhere and supports its choices, with limited planning depth."}
    - {score: 1.0, desc: "Report: a justified reproducible policy addresses coverage marginal value, reconnection, summary value, role dwell, freshness and resource limits; Implementation/evidence: decisive choices arise from evolving state and history without scenario-specific tables or material divergence."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of canonical and controlled experiments"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment is provided; Implementation/evidence: no checkable execution exists."}
    - {score: 0.4, desc: "Report: only isolated maps or aggregate numbers are shown and the communication mechanism is not tested; Implementation/evidence: recorded outputs do not support the main claims."}
    - {score: 0.7, desc: "Report: all canonical maps are covered and at least one controlled role, link or budget comparison tests a central mechanism; Implementation/evidence: comparable conditions are actually executed with supporting records."}
    - {score: 1.0, desc: "Report: complementary role-mechanism and link, budget or failure-boundary experiments test distinct claims; Implementation/evidence: condition-level coverage, freshness and communication evidence distinguish those claims beyond canonical aggregates."}

- id: research_analysis
  desc: "evidence-grounded interpretation, conclusions and limitations"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the recorded findings; Implementation/evidence: code and results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats coverage or message counts, makes unsupported causal claims and omits material limitations; Implementation/evidence: records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: conclusions use actual records to explain coverage-freshness-resource trends, distinguish observation from inference and state limitations; Implementation/evidence: implementation and outputs support those claims."}
    - {score: 1.0, desc: "Report: controlled evidence explains role opportunity cost, stale knowledge and link geometry, quantifies sensitivities, tests a failure boundary and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central analysis."}
