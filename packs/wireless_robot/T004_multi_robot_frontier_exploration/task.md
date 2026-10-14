---
id: T004_multi_robot_frontier_exploration
name: Communication-Constrained Multi-Robot Frontier Exploration
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

Three mobile robots explore an unfamiliar indoor area represented by a small occupancy grid. Each robot has a short-range sensor and maintains its own map, so different robots can have different knowledge while they are separated. Low-rate position messages are always available with a one-slot delay. At the beginning of a slot, a robot may send a directed map packet to another robot; a successful packet becomes usable at the beginning of the next slot. A higher-rate wireless link can carry a map snapshot only when two robots are sufficiently close; those packets consume a shared communication budget and can be lost. A map merge can reveal frontiers that were invisible to another robot and can therefore change both its next movement and the time at which the team completes exploration. Robots must also avoid walls and one another while moving simultaneously.

The study is a repeatable numerical model of collaborative search. It requires no images, map files, robot middleware, radio stack or external service. The practical objective is to discover the area quickly while keeping the team safe and using map communication selectively.

### Research Problem

Build a self-contained simulation and a non-training coordination policy. At every slot, each robot chooses a grid move and may request one directed map synchronization with another robot. The policy must use the robot's current local map, its own state, delayed peer positions, received snapshots, communication history and remaining budget to coordinate frontier exploration and rendezvous.

The study concerns the trade-off between moving to locally valuable frontiers and spending time and packets to exchange information. A synchronization decision affects later map knowledge rather than only the current slot, while movement choices affect future communication opportunities and collision risk. Evaluate the resulting coverage, completion speed, safe movement and map agreement, and explain which communication and exploration choices produce the observed behavior.

Use one unified, reusable procedure for all canonical environments. Do not supply pre-filled action tables or branches keyed to a particular environment name or time sequence.

### Experimental Setting and Constraints

#### Canonical environments and time

There are five canonical environments, each an `8x8` grid with coordinates `(x,y)`, where `x` grows east and `y` grows north. The following rows are listed from `y=7` down to `y=0`; `#` is an occupied cell and `.` is a free cell.

```text
split_corridor
########
#...#..#
#...#..#
#......#
#...#..#
#...#..#
#...#..#
########

central_ring
########
#......#
#..##..#
#..##..#
#..##..#
#..##..#
#......#
########

branching_rooms
########
#...#..#
#...#..#
#......#
#..#...#
#...#..#
#...#..#
########

offset_cross
########
#......#
#..##..#
#......#
#..##..#
#......#
#......#
########

gated_loop
########
#..#...#
#..#...#
#......#
#...#..#
#...#..#
#......#
########
```

The three initial positions are, in order `(robot 0, robot 1, robot 2)`:

```text
split_corridor:  (1,1), (6,1), (1,6)
central_ring:    (1,1), (6,1), (1,6)
branching_rooms: (1,1), (6,1), (6,6)
offset_cross:    (1,1), (6,1), (3,6)
gated_loop:      (1,1), (6,1), (1,6)
```

Each mission has 32 slots, indexed `t=0,...,31`, and ends after the movement in slot 31. A post-slot state at index `t+1` is the state after slot `t`. The fixed environment layout, starts and all rules below are the complete canonical input; no random seed is used.

#### Local maps and observations

Every robot begins with an unknown map (`?`) except that its starting cell is known free. At the start of slot `t`, after packet admissions, robot `i` observes every grid cell with Manhattan distance at most 1 from its current position, including the current cell. An observed cell is recorded as `.` or `#` according to the fixed environment and receives observation timestamp `t` and source `i`. Cells outside the grid are not map cells and are never observed. A robot cannot use another robot's map until a synchronization packet containing that map has been admitted.

The low-rate position view available at slot `t` contains the positions at the end of slot `t-1` (and the initial positions at `t=0`); it is delivered for every robot without consuming the map-packet budget. No future positions, map observations or packet outcomes are available to a decision.

#### Decisions, movement and collision resolution

At each slot, robot `i` emits exactly one `move` from `{north,east,south,west,stay}` and either `sync_to=null` or the index of one other robot. Moves are proposed from the positions at the start of the slot. A proposal outside the grid, into an occupied cell, or into another robot's current cell is rejected and that robot stays with `blocked=true`. If two or more robots propose the same free target, all of them stay and have `collision=true`. If two robots exchange their current cells, both stay and have `collision=true`. These checks are simultaneous; a rejected move does not free a cell for another proposal. Other valid proposals execute simultaneously. A slot is collision-free when no same-target or exchange collision occurs.

#### Map synchronization and communication

The high-rate communication range is Manhattan distance at most `2` at the generation instant. A `sync_to=j` decision generates one packet even when the pair is out of range or the packet is lost; every generated packet consumes one unit of the per-environment budget. The budget is `24` packets per environment, and every consecutive block of four slots contains at most `3` generated packets across all senders and receivers. A packet generated at slot `t` is scheduled for arrival at `t+1`; arrivals at or after slot 32 are ignored. The packet is available only if the pair was in range and the fixed delivery rule below returns success. At the generation instant it carries the sender's complete map, including the observations made at the start of that slot.

For environment index `e` in the order shown above (`0` through `4`), a generated packet from sender `i` to receiver `j` is delivered exactly when the pair is in range and `(t + 3*i + 5*j + e) mod 7 != 0`. The result is determined at generation and does not depend on later positions. At the beginning of slot `t`, all packets arriving at `t` are admitted in increasing `(sender, receiver)` order. For each cell, the receiver keeps the value with the larger observation timestamp; if timestamps tie, the lower source index wins. Unknown cells have timestamp `-1` and source index `3`. A packet that is unavailable, lost or outside the mission has no map effect.

The trace must report each generated synchronization event, including its range test, delivery result and arrival index. `arrival=null` is used when no packet will be admitted.

#### Initial event order and trace state

For each slot, process events in this order: (1) admit all map packets arriving at `t`; (2) apply the local sensor observations at the three current positions; (3) make the three movement and synchronization decisions; (4) generate synchronization packets from the maps now held; (5) resolve all proposed moves simultaneously; and (6) record the pre-move positions, post-move positions, maps after observation and admission, known-cell counts, blocked flags, collision flags and generated synchronization events. The maps do not change during steps 4--6. At `t=0`, there are no arriving packets and the delayed position view is the initial-position tuple.

#### Task metrics

All metrics are computed only after the result structure, coverage, legality and replay checks pass. Unknown cells are those encoded as `?`; a cell is known when it is `.` or `#`.

- `coverage_quality`: for each environment, the number of grid cells known by at least one robot in  the final recorded map divided by 64; average over five environments.
- `completion_speed`: for each environment, let `t90` be the smallest post-slot index in `1,...,32`  at which the union of the three known-cell sets contains at least `ceil(0.90*64)=58` cells. If no  such index exists, that environment contributes `0`; otherwise it contributes `max(0,(32-t90)/31)`.  Average the five contributions.
- `safe_movement`: one minus the number of slots with any same-target or exchange collision divided by  `5*32`.
- `map_consensus`: at the final state of each environment, for each of the three robot pairs compute  the fraction of cells that are known by at least one member and have equal encoded values in both  maps. Average the defined pair fractions over pairs and environments; a pair with no known cell  contributes `0`.
- `sync_efficiency`: `clip(1-n_sync/120,0,1)`, where `n_sync` is the total number of generated  synchronization packets over all five environments and `clip(x,0,1)=min(1,max(0,x))`.

Higher values are better. Numeric comparisons use absolute tolerance `1e-6`.

#### Result contract

`python run.py` must create or overwrite a UTF-8 `results.json` in the project root. Its root is exactly an object with one key, `scenarios`, whose value has exactly one record for each of the five environment names. A scenario record has exactly `scenario`, `decisions` and `trace`.

`decisions` contains exactly 32 records with unique `t=0,...,31`. Each record has exactly `t` and `robots`; `robots` contains exactly three records with unique `robot=0,1,2`, `move` as one of the five enumerated strings and `sync_to` as `null` or an integer in `{0,1,2}` different from `robot`.

`trace` contains exactly 32 records with unique `t=0,...,31`. Each record has exactly `t`, `positions`, `post_positions`, `map_codes`, `known_counts`, `blocked`, `collision` and `sync_events`. `positions` and `post_positions` are arrays of three coordinate arrays, each containing two integers in `0,...,7`. `map_codes` is an array of three strings, each exactly 64 characters in row-major order from `y=0` to `y=7`, with only `?`, `.` and `#`. `known_counts` contains three integers in `[0,64]`; `blocked` and `collision` contain three booleans. `sync_events` contains exactly the generated packets for that slot, each with exactly `sender`, `receiver`, `in_range`, `delivered` and `arrival`; sender and receiver are distinct robot indices, `in_range` and `delivered` are booleans, and `arrival` is `null` or an integer in `1,...,32`. The event list is ordered by `(sender,receiver)`.

The decisions are the only candidate-controlled input to independent replay. Trace fields are evidence of the fixed observations, packet events, movement and map state; the grader derives all metrics from the replay and does not use candidate-reported aggregate values.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root of your project must contain `run.py`, `report.md` and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain six non-empty Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json parses as strict JSON and matches the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "each canonical environment provides typed decision, map, movement and synchronization evidence", weight: 4}
- {id: canonical_coverage, desc: "all five environments and every declared decision and trace time key occur exactly once", weight: 4}
- {id: decision_legality, desc: "moves, synchronization targets, packet budget and simultaneous movement constraints satisfy the public rules", weight: 4}
- {id: replay_validity, desc: "observations, packet outcomes, map merges, positions, conflicts and trace maps match the fixed replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed final exploration coverage", weight: 6}
- {id: primary_metrics_2, desc: "independently replayed time to reach ninety-percent team coverage", weight: 5}
- {id: primary_metrics_3, desc: "independently replayed safe movement rate", weight: 4}
- {id: auxiliary_metrics_1, desc: "independently replayed final pairwise map consensus", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed synchronization communication efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the local-map, communication, movement, collision and exploration formulation"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent exploration formulation is given, or the stated state and communication process is materially wrong; Implementation/evidence: no executable representation or recorded evidence corresponds to the claimed model."}
    - {score: 0.4, desc: "Report: only part of local knowledge, packet timing, movement, collision or objective relations is defined and a decisive relation is omitted; Implementation/evidence: code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: accurately defines the robots, local maps, observations, synchronization, movement rules, constraints and objectives; Implementation/evidence: code and records instantiate the core model across all canonical environments, with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely specifies information boundaries, event order, timestamped map merging, communication-to-exploration consequences and collision semantics; Implementation/evidence: code and records faithfully realize these relations without material divergence."}

- id: method_design
  desc: "quality and task relevance of the unified non-training exploration and synchronization policy"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, a prohibited training method is used, or the policy conflicts with the fixed information and action model; Implementation/evidence: no runnable procedure or supporting decisions are present."}
    - {score: 0.4, desc: "Report: the method is partial, ignores a decisive map, rendezvous, resource or collision trade-off, or presents environment-specific schedules as the primary solution; Implementation/evidence: the procedure relies on pre-filled actions or materially differs from the report."}
    - {score: 0.7, desc: "Report: presents a legal reusable policy that uses local maps, peer positions and synchronization history to address frontier progress and communication cost; Implementation/evidence: one procedure generates all canonical decisions and records support its operation, with limited look-ahead or robustness."}
    - {score: 1.0, desc: "Report: justifies a unified reproducible policy that plans exploration and rendezvous from evolving map state, delayed positions, packet outcomes, budget and collision consequences; Implementation/evidence: the implementation applies that policy consistently and records support its decisive choices without environment-specific action tables or material divergence."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of experiments for communication-constrained exploration"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: code and records contain no checkable experimental execution."}
    - {score: 0.4, desc: "Report: experiments are isolated, incomplete or limited to aggregate coverage and do not examine map exchange, movement safety or communication cost; Implementation/evidence: recorded outputs do not execute or support the principal claims."}
    - {score: 0.7, desc: "Report: all canonical environments are covered and at least one explicitly controlled comparison or condition-specific experiment beyond canonical aggregates tests a central synchronization, frontier or safety trade-off; Implementation/evidence: the compared conditions are actually executed with records and metrics supporting the comparison."}
    - {score: 1.0, desc: "Report: complementary experiments examine at least two meaningful communication or coordination regimes, including a mechanism-focused comparison and a sensitivity or failure-boundary check; Implementation/evidence: condition-level trajectories, map states and metrics are executed and distinguish the stated claims beyond canonical aggregates."}

- id: research_analysis
  desc: "evidence-grounded interpretation of map sharing, exploration and safety trade-offs"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent, materially contradicts the records, or treats unimplemented experiments as facts; Implementation/evidence: code and results cannot support the reported conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats coverage values or makes unsupported causal claims and does not identify material limits; Implementation/evidence: records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: conclusions follow the recorded maps and trajectories, explain important condition-dependent communication or movement trends, distinguish observation from inference and state material limitations; Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains how local uncertainty, rendezvous, packet loss, budget and conflicts produce outcomes, quantifies principal trade-offs or sensitivities, tests the explanation against contrasting evidence or a failure boundary, and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
