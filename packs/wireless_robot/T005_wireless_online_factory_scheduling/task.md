---
id: T005_wireless_online_factory_scheduling
name: Wireless Intention Exchange for Online Multi-Robot Factory Scheduling
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

An automated factory uses three mobile transport robots (AGVs) to move material between production stations. Jobs appear while the line is operating. Each job requires an AGV to visit a pickup station and then a delivery station before a deadline; some jobs cannot be released until an earlier job in the same production flow has been delivered. The AGVs share narrow aisles and can block or collide when they choose routes without knowing one another's intentions.

Each AGV has only a local view of the floor and nearby robots. It may send a short wireless intention report containing its post-move position, current task state and a committed move for the next slot. Reports use a shared, contention-prone radio resource and arrive with a delay, so a robot may have to act on stale or missing information. The transport, communication and scheduling decisions form one online closed loop: a route choice changes future task availability and encounters, while a report can change another robot's route and collision risk.

### Research Problem

Build a self-contained numerical simulation and a non-training coordination policy for this factory. At every time slot, each AGV chooses one grid move, may claim one eligible job, and may send one directed intention report. The policy should assign jobs, carry them through pickup and delivery, avoid collisions and unnecessary blocking, and decide when communication is worth its shared cost. Each robot policy may use only that robot's state and history, the static map and job table, the currently announced eligible jobs, its local observations and reports already admitted to it. Global simulator state is used only to resolve communication, simultaneous motion and task events; it must not be supplied to a robot policy as coordination information.

Study how intention exchange changes online scheduling under partial observability. Report the effect of communication and coordination choices on deadline service, completed work, movement safety and radio use. Use one reusable, state-driven procedure for all canonical scenarios. Do not provide pre-filled actions or result tables for individual scenarios.

### Experimental Setting and Constraints

#### Factory grids and time

There are five canonical scenarios. Each factory is a `9x9` grid with integer coordinates `(x,y)`, `0 <= x,y <= 8`; `x` increases east and `y` increases north. Rows below are listed from `y=8` to `y=0`. `#` is an occupied cell and `.` is traversable. A robot occupies one traversable cell. A move changes one coordinate by one, or is `stay`.

```text
cross_aisles
#########
#.......#
#..###..#
#.......#
#..#....#
#.......#
#.......#
#.......#
#########

dual_bottleneck
#########
#...#...#
#...#...#
#.......#
###.#####
#.......#
#...#...#
#...#...#
#########

ring_aisle
#########
#.......#
#.#####.#
#.......#
#...#...#
#.......#
#.#####.#
#.......#
#########

staggered
#########
#..#....#
#..#....#
#.......#
#....#..#
#....#..#
#.......#
#.......#
#########

crossing
#########
#.......#
#...#...#
#...#...#
#.......#
#...#...#
#...#...#
#.......#
#########
```

The initial positions `(robot 0, robot 1, robot 2)` are:

```text
cross_aisles:     (1,1), (7,1), (1,7)
dual_bottleneck:  (1,1), (7,1), (7,7)
ring_aisle:       (1,1), (7,1), (1,7)
staggered:        (1,1), (7,1), (7,7)
crossing:         (1,1), (7,1), (1,7)
```

Each mission has `40` slots, `t=0,...,39`, and ends after movement and task events in slot `39`. The state at post-slot index `t+1` is the state after slot `t`.

#### Online jobs

Each scenario has eight jobs with integer IDs `0,...,7`. A job record is `(release, pickup, delivery, deadline, predecessor)`. A job is eligible at slot `t` exactly when it is unassigned, `release <= t`, and its predecessor is `null` or has already been delivered. The following tables give the complete fixed job input; a predecessor is written as `-`.

| scenario | job | release | pickup | delivery | deadline | predecessor |
|---|---:|---:|---|---|---:|---|
| `cross_aisles` | 0 | 0 | `[2,1]` | `[6,1]` | 14 | - |
|  | 1 | 0 | `[6,7]` | `[2,7]` | 16 | - |
|  | 2 | 4 | `[1,3]` | `[7,3]` | 23 | 0 |
|  | 3 | 5 | `[7,5]` | `[1,5]` | 25 | 1 |
|  | 4 | 9 | `[3,1]` | `[3,7]` | 31 | 2 |
|  | 5 | 11 | `[5,7]` | `[5,1]` | 33 | 3 |
|  | 6 | 17 | `[1,4]` | `[7,4]` | 37 | 4 |
|  | 7 | 19 | `[7,6]` | `[1,6]` | 39 | 5 |
| `dual_bottleneck` | 0 | 0 | `[2,1]` | `[6,1]` | 15 | - |
|  | 1 | 0 | `[6,7]` | `[2,7]` | 17 | - |
|  | 2 | 4 | `[1,3]` | `[7,3]` | 24 | 0 |
|  | 3 | 5 | `[7,5]` | `[1,5]` | 26 | 1 |
|  | 4 | 9 | `[3,2]` | `[3,6]` | 32 | 2 |
|  | 5 | 11 | `[5,6]` | `[5,2]` | 34 | 3 |
|  | 6 | 17 | `[3,5]` | `[7,5]` | 38 | 4 |
|  | 7 | 19 | `[5,3]` | `[1,3]` | 39 | 5 |
| `ring_aisle` | 0 | 0 | `[2,1]` | `[6,1]` | 14 | - |
|  | 1 | 0 | `[6,7]` | `[2,7]` | 16 | - |
|  | 2 | 4 | `[1,5]` | `[7,5]` | 23 | 0 |
|  | 3 | 5 | `[7,3]` | `[1,3]` | 25 | 1 |
|  | 4 | 9 | `[3,5]` | `[3,3]` | 31 | 2 |
|  | 5 | 11 | `[5,3]` | `[5,5]` | 33 | 3 |
|  | 6 | 17 | `[1,4]` | `[7,4]` | 37 | 4 |
|  | 7 | 19 | `[7,6]` | `[1,6]` | 39 | 5 |
| `staggered` | 0 | 0 | `[2,1]` | `[6,1]` | 14 | - |
|  | 1 | 0 | `[6,7]` | `[2,7]` | 16 | - |
|  | 2 | 4 | `[1,5]` | `[7,5]` | 23 | 0 |
|  | 3 | 5 | `[7,3]` | `[1,3]` | 25 | 1 |
|  | 4 | 9 | `[2,2]` | `[6,2]` | 31 | 2 |
|  | 5 | 11 | `[6,6]` | `[2,6]` | 33 | 3 |
|  | 6 | 17 | `[1,4]` | `[7,4]` | 37 | 4 |
|  | 7 | 19 | `[7,5]` | `[1,5]` | 39 | 5 |
| `crossing` | 0 | 0 | `[2,1]` | `[6,1]` | 14 | - |
|  | 1 | 0 | `[6,7]` | `[2,7]` | 16 | - |
|  | 2 | 4 | `[1,4]` | `[7,4]` | 23 | 0 |
|  | 3 | 5 | `[7,4]` | `[1,4]` | 25 | 1 |
|  | 4 | 9 | `[3,1]` | `[3,7]` | 31 | 2 |
|  | 5 | 11 | `[5,7]` | `[5,1]` | 33 | 3 |
|  | 6 | 17 | `[1,6]` | `[7,6]` | 37 | 4 |
|  | 7 | 19 | `[7,2]` | `[1,2]` | 39 | 5 |

The task table is the input specification, not a schedule. At the beginning of a slot, all jobs whose release and predecessor conditions are satisfied are announced by the factory controller and may be claimed by an idle AGV. A job can be claimed once; if several AGVs claim the same eligible job in the same slot, the lowest robot index receives it and the other claims have no effect. An AGV can carry at most one job. It first travels to pickup, becomes loaded on the post-move state when it reaches pickup, and then travels to delivery. Delivery is recorded at the post-slot index when the loaded AGV reaches the delivery cell. A late delivery remains a completion but is not an on-time delivery. An assigned job is never reassigned.

#### Local information and intention reports

The sensing range is Manhattan distance `1`. At the start of a slot, an AGV observes traversable and blocked cells within that range, its own position and task state, and the positions of other AGVs within that range. The static map and complete job table are public. The factory controller announces the IDs of jobs that are eligible and unassigned at the start of the slot. A report admitted at the start of a slot adds the sender's most recent reported position, task state and committed move to the receiver's information. An unavailable or not-yet-admitted report adds nothing. Information about an AGV outside direct sensing range is unavailable unless it appears in an admitted report to the receiver.

At each slot `t<39`, an AGV may set `announce_to` to `null` or one distinct robot index. If `announce_to` is not `null`, `committed_move` must be one of `north/east/south/west/stay`; otherwise `committed_move` must be `null`. A report is generated after movement and task events. It contains the sender's post-move position, active task ID (or `null`), phase (`idle`, `to_pickup`, or `to_delivery`), loaded flag and committed move for slot `t+1`. The sender must propose that committed move at `t+1`, whether or not the report is delivered. At `t=39`, both report fields must be `null`. Reports generated at the same slot use two shared channels. For a report from `i` to `j`,

```text
channel = (t + i + j) mod 2
in_range = Manhattan distance between the post-move positions <= 3
delivered = in_range and no other generated report uses the same channel at t
            and (t + 3*i + 5*j + scenario_index) mod 7 != 0
```

`scenario_index` is the zero-based order in the scenario list above. `delivered` records the link outcome. A delivered report has `arrival=t+1` and is admitted at the beginning of that slot; an undelivered report has `arrival=null`. Every generated report consumes one unit of a shared budget, including an out-of-range or lost report. There are at most `24` reports per scenario, and every consecutive block of four slots contains at most `3` generated reports across all senders. Events arriving at the same slot are admitted in increasing `(sender, receiver)` order; the receiver keeps the newest admitted report from each sender.

#### Event order and movement

For every slot, use this order: (1) admit reports arriving at `t`; (2) activate and announce jobs whose release, predecessor and assignment conditions now hold; (3) update each AGV's local observation; (4) apply the three claim decisions simultaneously; (5) resolve all proposed moves simultaneously; (6) apply pickup and delivery events at post-move positions; (7) choose report targets and next-slot commitments from each sender's available information and post-move task state, then resolve report outcomes; and (8) record the trace. A claim decision is made against the state before claims in that slot. A move is proposed from the pre-move position, regardless of the assigned job target. The fixed simulator may access global state for event resolution, but the policy producing an individual AGV's decisions remains subject to the information boundary above.

A proposal outside the grid, into a blocked cell, or into another AGV's pre-move cell is rejected and that AGV stays with `blocked=true`, except an exchange of two pre-move cells, which is a `collision` for both AGVs. If two or more non-blocked proposals select the same free target, all those AGVs stay and have `collision=true`. These tests are simultaneous; a rejected proposal does not free a cell for another proposal. Other valid proposals execute simultaneously. A slot is collision-free when no AGV has `collision=true`.

The `move` and `claim` fields are submitted from the pre-move observation in step (3). The `announce_to` and `committed_move` fields are selected in step (7) using that same observation and history, reports admitted at the start of the slot, and the sender's own post-move task state. No new neighbor sensing, report admission, or other external information occurs between steps (3) and (7); a commitment selected at slot `t` is the required move at slot `t+1`. A task delivered at the end of slot `t` becomes a predecessor fact for eligibility only at the beginning of slot `t+1`. Decision fields therefore represent the claims and move submitted before simultaneous execution and the post-event report commitment, while `report_ages` is the start-of-slot observation after arrival, `trace` and `packets` are post-event evidence, and `tasks` are final replay evidence. No field may use future releases, arrivals, movement outcomes, or task outcomes.

#### Task metrics

All metrics are computed only after the official run, strict result structure, evidence, canonical coverage, decision legality and replay checks pass.

- `deadline_service`: the number of jobs with a delivery time no greater than its deadline divided  by `5*8`.
- `completion_rate`: the number of jobs delivered by post-slot index `40` divided by `5*8`.
- `collision_free`: one minus the number of slots with any collision divided by `5*40`.
- `communication_efficiency`: `clip(1 - n_report/120, 0, 1)`, where `n_report` is the total number  of generated reports over all scenarios.
- `intent_freshness`: count ordered receiver/sender pair-slots whose newest admitted report at the  start of the slot has age at most `3`, divide by `5*24*3=360`, and clip to `[0,1]`. The denominator  is the maximum fresh pair-slot contribution of the five per-scenario report budgets.

Here `clip(x,0,1)=min(1,max(0,x))`; higher values are better. Numeric comparisons use absolute tolerance `1e-6`.

#### Result contract

`python run.py` must create or overwrite a UTF-8 `results.json` in the project root. This file records the canonical radio-enabled condition, in which the policy may generate and use reports. Its root is exactly an object with one key, `scenarios`; `scenarios` is an array containing exactly one record for each of the five scenario names.

Each scenario record has exactly `scenario`, `decisions`, `trace` and `tasks`.

`decisions` contains exactly 40 records with unique `t=0,...,39`. Each record has exactly `t` and `robots`; `robots` contains exactly three records with unique `robot=0,1,2`, `move` in `north/east/south/west/stay`, `claim` as `null` or an integer in `0,...,7`, `announce_to` as `null` or an integer in `0,1,2` different from `robot`, and `committed_move` as specified by the reporting and next-slot commitment rules.

`trace` contains exactly 40 records with unique `t=0,...,39`. Each record has exactly `t`, `positions`, `post_positions`, `active_tasks`, `phases`, `loaded`, `blocked`, `collision`, `report_ages` and `packets`. `positions` and `post_positions` are arrays of three coordinate arrays with two integers in `0,...,8`; `active_tasks` contains three integers or `null` describing the state after pickup and delivery in that slot; `phases` contains three values from `idle`, `to_pickup`, `to_delivery`; `loaded`, `blocked` and `collision` contain three booleans. `report_ages` is a `3x3` array sampled at the start of the slot after arrivals: row `r`, column `s` is `t - generation_slot` for the newest report from sender `s` that was generated at `generation_slot` and admitted to receiver `r`; thus a report generated at slot `t-1` and admitted at the start of slot `t` has age `1`, not `0`. The age is not reset by admission. It is `null` when no such report exists, including every self-pair. `packets` contains exactly the reports generated in that slot, ordered by `(sender,receiver)`. Each packet has exactly `sender`, `receiver`, `channel`, `in_range`, `delivered`, `arrival`, `position`, `committed_move`, `task`, `phase` and `loaded`; `sender` and `receiver` are distinct integers in `0,...,2`, `channel` is an integer in `0,1`, `in_range` and `delivered` are booleans, `arrival` is `null` or an integer in `1,...,39`, `task` is `null` or an integer in `0,...,7`, `position` is a two-integer grid coordinate, and the remaining values use the stated move and phase enums.

`tasks` contains exactly eight records with unique `task=0,...,7`. Each record has exactly `task`, `assigned_robot`, `pickup_time`, `delivery_time` and `on_time`; `assigned_robot` is `null` or an integer in `0,...,2`, `pickup_time` and `delivery_time` are `null` or integers in `1,...,40`, and `on_time` is boolean. The trace and task records are evidence of the fixed replay; the decisions are the only candidate-controlled input. The grader derives all metrics from replayed state and verified decisions, not from candidate-reported aggregates.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root of your project must contain `run.py`, `report.md` and `results.json`. `run.py` is the only official entry point and should start the simulation without interactive input. `report.md` must contain six non-empty Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`. The project must also execute a controlled radio-enabled/radio-disabled comparison using the same policy, task inputs and parameters; the disabled condition removes report generation and admission only. Retain one record for every scenario under both `radio_enabled` and `radio_disabled` in a supplementary UTF-8 JSON file. Each record must contain its condition, decisions, task outcomes and derived metrics; state the file name and schema in the report.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json parses as strict JSON and matches the declared root and top-level structure", weight: 3}
- {id: result_evidence, desc: "each scenario provides typed decisions, movement state, task state, report-age, commitment and packet evidence required by the result contract", weight: 4}
- {id: canonical_coverage, desc: "all five scenarios and every declared decision, trace and task key occur exactly once", weight: 4}
- {id: decision_legality, desc: "claims, moves, report targets, next-slot commitments, report budget and burst constraints satisfy the public rules", weight: 4}
- {id: replay_validity, desc: "job releases, predecessor state, report payloads, outcomes and ages, positions, conflicts and task records match the fixed replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline service rate", weight: 7}
- {id: primary_metrics_2, desc: "independently replayed completed-task rate", weight: 5}
- {id: primary_metrics_3, desc: "independently replayed collision-free movement rate", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed intention communication efficiency", weight: 2}
- {id: auxiliary_metrics_2, desc: "independently replayed intention-report freshness", weight: 3}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the online task, partial-observation, intention-communication and multi-robot movement formulation"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent model is given, or task state, communication timing or movement semantics are materially wrong; Implementation/evidence: no runnable representation or corresponding records are present."}
    - {score: 0.4, desc: "Report: only part of task release, precedence, per-robot information, report commitments, routes or conflicts is defined and a decisive relation is omitted; Implementation/evidence: code or records cover only a partial or materially different process."}
    - {score: 0.7, desc: "Report: accurately defines the principal states, per-robot information boundary, event order, task constraints, communication process, movement rules and objectives; Implementation/evidence: code separates policy observations from simulator truth and instantiates the core model across all scenarios, with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains how online precedence, persistent carrying, delayed committed intentions, contention and simultaneous movement jointly determine outcomes; Implementation/evidence: the implementation and records faithfully realize these relations and information boundaries without material divergence."}

- id: method_design
  desc: "quality and task relevance of the unified non-training scheduling and communication policy"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, or the method is prohibited or incompatible with the task model; Implementation/evidence: no runnable procedure or supporting decisions are present."}
    - {score: 0.4, desc: "Report: the method ignores a decisive temporal or informational coupling, uses remote ground truth, or presents scenario-specific schedules as the solution; Implementation/evidence: policy decisions use out-of-range global state, pre-filled actions or a procedure materially different from the report."}
    - {score: 0.7, desc: "Report: presents a legal reusable policy that assigns and routes jobs from per-robot observations while using report history and resource limits to address deadlines and conflicts; Implementation/evidence: one locally informed procedure generates all canonical decisions and records support its operation, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: justifies a unified reproducible policy that jointly treats dependent jobs, persistent carrying, route conflicts, committed intentions and communication cost; Implementation/evidence: code derives each robot's decisions only from its permitted evolving information, and records substantiate the decisive choices without action tables."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of experiments for communication-assisted online scheduling"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: code and records contain no checkable experimental execution."}
    - {score: 0.4, desc: "Report: experiments are incomplete or limited to canonical aggregates without a valid communication comparison; Implementation/evidence: condition-level outputs are absent or do not support the stated claims."}
    - {score: 0.7, desc: "Report: all canonical scenarios and the same-policy radio-enabled/radio-disabled comparison test communication effects on scheduling and safety; Implementation/evidence: both conditions are executed and retained with comparable decisions, task outcomes and metrics."}
    - {score: 1.0, desc: "Report: complementary experiments add a controlled sensitivity or failure-boundary test that explains when committed intentions help or fail; Implementation/evidence: reproducible condition-level records support both the counterfactual and the additional test."}

- id: research_analysis
  desc: "evidence-grounded interpretation of communication, scheduling and movement trade-offs"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent, materially contradicts the records or treats unexecuted experiments as facts; Implementation/evidence: code and results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly lists completion or collision values, makes unsupported causal claims or omits material limits; Implementation/evidence: records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: conclusions follow the recorded counterfactual and trajectory evidence, explain important report, route and deadline trends, distinguish observation from inference and state material limitations; Implementation/evidence: code and condition records support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains how information range, commitment delivery and communication cost produce the observed scheduling outcomes, quantifies the main trade-offs and tests a sensitivity or failure boundary; Implementation/evidence: reproducible condition-level comparisons support the central conclusions."}
