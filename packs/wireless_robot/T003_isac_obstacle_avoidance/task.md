---
id: T003_isac_obstacle_avoidance
name: ISAC UAV Obstacle Avoidance with Sensing and Command Scheduling
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

A ground station supervises a planar unmanned aerial vehicle (UAV) flying through a time-sensitive inspection route. The station can send two kinds of downlink packet over the same wireless system: an environmental-sensing packet and a command-and-control (C&C) packet. A sensing packet provides the station and the UAV with a snapshot of the moving obstacles. A C&C packet carries a planar velocity command. Packets can be lost or delayed. Consequently, the UAV may execute an older command while its most recent obstacle map is already stale. The vehicle has a simple safety monitor: when it has a sufficiently fresh map and the next commanded position would enter an obstacle's safety radius, it stops for that slot. With no fresh map, that protective stop is unavailable and a collision can occur. Selective sensing and command transmission therefore changes both communication use and the physical mission outcome.

The ground station observes the UAV position after each slot and knows the route and the fixed packet delivery rules for the current operating condition. The study is a repeatable numerical model; it does not require a radio stack, a perception library, a hardware interface, or external data.

### Research Problem

Develop a self-contained simulation and a non-training policy for the closed-loop mission. At each control slot, decide independently whether to send a sensing packet and whether to send a new C&C velocity command. The policy must coordinate the shared signal budget and burst limit with different sensing and command delivery patterns, sensing-map age, command age, velocity continuity, moving obstacles, safety stops, route tracking, and waypoint deadlines.

The objective is to keep the UAV collision-free and close to its scheduled route, reach the prescribed waypoints on time, and use sensing and C&C packets selectively. Decisions have consequences beyond the current slot: a packet may arrive later, a map may remain useful for several slots, a command may become the active command or expire, and a transmission consumes budget even when it is lost. Explain the communication--control trade-off and support the method with reproducible experiments and analysis.

Use one state-driven, reusable procedure across all canonical scenarios. The procedure must derive actions from the scenario inputs, current UAV state, available map and command state, communication history, and remaining resources; do not provide pre-filled action tables or branches keyed to individual scenario names and slot sequences. For fixed inputs and history, use a deterministic procedure; do not introduce unrecorded randomness into the submitted decisions.

### Experimental Setting and Constraints

#### Canonical scenarios and route

There are five canonical scenarios. Each has four physical waypoints after the common start. The waypoint deadlines are slots `0, 6, 12, 18, 24`; waypoint 0 is the initial position. For scenario `s`, let `w[s,k]` be waypoint `k` (`k=0,...,4`). The scheduled reference point at every integer `t=0,...,24` is the linear interpolation between the adjacent waypoints whose deadlines surround `t`:

```text
deadline = [0, 6, 12, 18, 24]
g[s,t] = (1-a) * w[s,k] + a * w[s,k+1]
a = (t - deadline[k]) / 6
```

| scenario | `w0` | `w1` | `w2` | `w3` | `w4` |
|---|---|---|---|---|---|
| `crossing_corridor` | `[0,0]` | `[4,0]` | `[4,4]` | `[8,4]` | `[8,8]` |
| `north_gate` | `[0,0]` | `[0,4]` | `[4,4]` | `[4,8]` | `[8,8]` |
| `switchback` | `[0,0]` | `[3,1]` | `[1,5]` | `[6,6]` | `[8,8]` |
| `diagonal_drift` | `[0,0]` | `[2,3]` | `[5,1]` | `[6,6]` | `[8,8]` |
| `return_arc` | `[0,0]` | `[4,2]` | `[2,6]` | `[6,7]` | `[8,8]` |

The UAV starts at `p[s,0]=[0,0]` metres. Slot `t` is `[t,t+1)` seconds for `t=0,...,23`. The position at the beginning of slot `t` is `p[s,t]`. A velocity is measured in metres per second.

#### Moving obstacles and fixed packet links

Each scenario has two obstacles. Obstacle `j` has centre `o[s,j,t] = o0[s,j] + t * v[s,j]` at the beginning of slot `t`, with the following fixed values in metres and metres per second:

| scenario | `o0[0]`, `v[0]` | `o0[1]`, `v[1]` |
|---|---|---|
| `crossing_corridor` | `[2.0,-0.8]`, `[0,0.18]` | `[4.4,2.0]`, `[-0.12,0]` |
| `north_gate` | `[-0.9,2.3]`, `[0.18,0]` | `[3.2,4.4]`, `[0,0.18]` |
| `switchback` | `[2.0,2.0]`, `[0.12,0.12]` | `[3.2,5.2]`, `[-0.20,0]` |
| `diagonal_drift` | `[2.0,1.8]`, `[0.15,-0.05]` | `[5.4,3.1]`, `[-0.10,0.16]` |
| `return_arc` | `[3.5,1.8]`, `[-0.10,0.14]` | `[4.2,6.0]`, `[0.16,-0.08]` |

For a packet generated at slot `t`, the sensing link and command link have the following outcomes. `lost` is the set of generation slots whose packet is not delivered. For every other slot, the delay is the value returned by the stated rule.

| scenario | sensing `lost`; delay | command `lost`; delay |
|---|---|---|
| `crossing_corridor` | `{5,6,7}`; `1` if `t mod 3 = 0`, otherwise `0` | `{4,5,6,15}`; `1` for `t` in `{3,4,11,12,19,20}`, otherwise `0` |
| `north_gate` | `{2,3,4,14}`; `2` for `t` in `{1,8,15,22}`, otherwise `1` | `{0,1,10,11,12}`; `2` if `t` is even, otherwise `1` |
| `switchback` | `{8,9,10,17}`; `1` if `t` is even, otherwise `0` | `{3,4,5,16,17}`; `1` if `t mod 3 = 0`, otherwise `0` |
| `diagonal_drift` | `{0,1,12,13,20}`; `2` if `t mod 4 = 0`, otherwise `1` | `{7,8,9,18}`; `2` for `t` in `{6,14,22}`, otherwise `0` |
| `return_arc` | `{4,5,6,18,19}`; `1` if `t` is odd, otherwise `0` | `{2,3,14,15,16}`; `1` if `t` is odd, otherwise `0` |

The station knows all fixed route, obstacle and link definitions at the start of the mission. It may use the current position, arrived packets and previous observations. A sensing packet generated at `t` contains the two obstacle centres at `t`; it does not contain future centres. A command packet contains only its generated velocity. The station may derive obstacle centres from the published obstacle initial positions and velocities when planning, but the UAV safety monitor may use only the most recently admitted sensing map and the fixed obstacle velocities; model knowledge at the station does not itself enable the monitor.

#### Event order, caches and motion

At the beginning of slot `t`, pending packets with arrival time `t` are admitted. Among packets of the same type, the one with the largest generation slot becomes current; an older packet never replaces a newer one. The station then makes the two transmission decisions. A successfully delivered zero-delay packet is admitted immediately before the slot is executed. A packet whose arrival time is `24` or later is outside the mission and is ignored. The delivery result and delay depend only on the scenario and generation slot. The policy may use these published deterministic rules for planning, but packet contents become available only when the corresponding packet is admitted.

The active command has age `t-generation_slot`. Before movement, an active command older than `3` slots is discarded; with no active command the commanded velocity is `[0,0]`. A sensing map is retained after it becomes old, but the safety monitor is enabled only when its age is at most `2` slots. If a map generated at `q` is available at `t`, the monitor predicts obstacle `j` at `o0[s,j] + (t+1)*v[s,j]`. If the position produced by the active command in slot `t` would be within `1.0` metre of any predicted centre, the executed velocity is replaced by `[0,0]` for that slot. Otherwise the active command is executed. The post-slot position is

```text
p[s,t+1] = p[s,t] + u_exec[s,t]
```

The physical collision flag for slot `t` is true when the post-slot position is at distance at most `0.75` metre from any true centre `o[s,j,t+1]`. A safety stop prevents movement but is not itself a collision. The map monitor uses the most recent accepted sensing packet and the fixed obstacle velocities; it does not reveal any additional information.

In the result trace, `position` is the pre-movement position `p[s,t]` at the execution point described above, and `map_age` and `command_age` are sampled at that same point. Decision fields represent the actions submitted at the decision boundary; trace cache fields represent information available or state established at that execution point; executed velocity, safety, collision and waypoint fields represent evidence after the fixed event processing and motion rules. The detailed event order above is the authoritative definition of these semantics.

#### Signal and command constraints

The total number of generated sensing and command packets in one scenario is at most `18`. Each packet consumes one unit even if it is lost or arrives after the mission. In every consecutive block of four control slots, at most `3` packets of either type may be generated. For two transmitted commands in chronological generation order, the Euclidean difference between their velocities is at most `0.8` metres per second; a lost command still participates in this rule. Every transmitted velocity satisfies `-1 <= u_x,u_y <= 1` and `u_x^2+u_y^2 <= 1`. A command decision with no transmission has a `null` velocity.

#### Waypoint and metric definitions

The four non-initial waypoints are evaluated at `t=6,12,18,24`. A waypoint is successful when the independently replayed position is at most `1.0` metre from its target. Trajectory error uses the 120 post-slot positions `p[s,t]` for `s` in the five scenarios and `t=1,...,24`.

The official metrics are computed only after all validity checks pass:

- `collision_free_rate`: fraction of the 120 slots whose collision flag is false.
- `waypoint_completion`: fraction of the 20 waypoint deadlines that are successful.
- `tracking_quality`: `max(0, 1 - mse/64)`, where `mse` is the mean squared distance between  `p[s,t]` and `g[s,t]` over the 120 post-slot positions.
- `signal_efficiency`: `clip(1 - n_signal/90, 0, 1)`, where `n_signal` is the total number of  sensing and command packets generated over all scenarios.
- `map_freshness`: the mean over the 120 slots of `max(0,1-map_age/2)` when a map is available,  and `0` otherwise. `map_age` is the value recorded at execution time, after pending and zero-delay  admissions and before the safety check and movement.

Here `clip(x,0,1)=min(1,max(0,x))`; higher values are better. Numeric comparisons use absolute tolerance `1e-6`. The grader computes every value from the fixed replay and verified decisions.

#### Result contract

Running `python run.py` must create or overwrite a UTF-8 `results.json` in the project root. Its root is exactly an object with one key, `scenarios`; `scenarios` is an array containing exactly one record for each of the five scenario names. A scenario record has exactly the keys `scenario`, `decisions`, `trace`, and `waypoints`.

`decisions` contains exactly 24 records for `t=0,...,23` (order is not significant). Each record has exactly `t`, `sense`, `transmit`, and `velocity`; `sense` and `transmit` are booleans. When `transmit` is true, `velocity` is a finite two-number JSON array satisfying the bounds above; otherwise it is `null`. `trace` contains exactly 24 records for the same `t` values and exactly these keys: `t`, `position`, `active_sequence`, `command_age`, `map_sequence`, `map_age`, `executed_velocity`, `safety_stop`, and `collision`. `position` and `executed_velocity` are finite two-number arrays; each sequence is either `null` or an integer generation slot in `[0,23]`; each age is either `null` or a non-negative integer; `safety_stop` and `collision` are booleans. `waypoints` contains exactly four records with keys `waypoint`, `t`, `position`, `distance`, and `success`, for the four `(waypoint,t)` pairs specified above. Positions and distances are finite, and `success` is boolean. Numeric comparisons use absolute tolerance `1e-6`.

The decisions are the only input to independent replay. Trace and waypoint values are evidence of that replay; the grader derives packet outcomes, cache states, safety stops, positions, collisions, distances and all metrics from the decisions and fixed definitions.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root of your project must contain `run.py`, `report.md` and `results.json`. You may organize the implementation across multiple files; `run.py` is the only official entry point and should start the simulation without interactive input. `report.md` must contain six non-empty Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`. The report must describe the actual implementation and results, including assumptions, sensing/command scheduling, the communication-control trade-off, and limitations supported by the experiments.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json parses as strict JSON and matches the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "each scenario provides typed decision, cache-state trace, collision and waypoint evidence required by the result contract", weight: 4}
- {id: canonical_coverage, desc: "the five canonical scenarios and every declared decision, trace and waypoint key occur exactly once", weight: 4}
- {id: decision_legality, desc: "sensing and command decisions satisfy packet, burst, velocity and continuity constraints", weight: 4}
- {id: replay_validity, desc: "recorded caches, safety stops, motion, collision and waypoint fields match the fixed independent replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed collision-free rate", weight: 7}
- {id: primary_metrics_2, desc: "independently replayed waypoint completion rate", weight: 5}
- {id: primary_metrics_3, desc: "independently replayed trajectory tracking quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed signal efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed sensing-map freshness", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the sensing, command, cache, motion, obstacle, resource and deadline model in the report and implementation"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent ISAC obstacle-avoidance formulation is given, or the stated process is materially wrong; Implementation/evidence: no executable representation or recorded evidence corresponds to the claimed model."}
    - {score: 0.4, desc: "Report: only part of the sensing, command, obstacle, timing, state or objective model is defined and a decisive relation is omitted; Implementation/evidence: the code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: the principal caches, packet outcomes, UAV state, obstacle motion, safety condition, constraints, deadlines and objectives are accurately defined; Implementation/evidence: the code and records instantiate the core model across all canonical scenarios, with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely specifies event order, sensing and command availability, map and command ageing, safety intervention, motion, collision and communication-task trade-offs; Implementation/evidence: the code and records faithfully realize these relations without material divergence."}

- id: method_design
  desc: "quality and task relevance of the non-training policy for sensing and command scheduling"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, a prohibited training method is used, or the policy conflicts with the fixed information, action or safety model; Implementation/evidence: no runnable procedure or supporting decisions are present."}
    - {score: 0.4, desc: "Report: the method is partial, ignores a decisive cache, delivery, resource or obstacle trade-off, or presents scenario-specific schedules as the primary solution; Implementation/evidence: the procedure relies on pre-filled actions or materially differs from the report. A fixed schedule used only as an identified comparison is assessed under experimental coverage."}
    - {score: 0.7, desc: "Report: a legal, reusable state-driven policy coordinates sensing, command timing and velocity choices using current state, available information and history; Implementation/evidence: the same procedure generates all canonical decisions and the records support its operation, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified procedure anticipates delayed or lost packets, map and command ageing, shared resources, safety stops and waypoint consequences; Implementation/evidence: the implementation applies that procedure consistently and records support its decisive choices without scenario-specific action tables or material divergence."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of experiments for sensing, communication and obstacle-avoidance mechanisms"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: the code and records contain no checkable experimental execution."}
    - {score: 0.4, desc: "Report: experiments are isolated, incomplete or limited to aggregate outputs and do not examine the sensing-to-safety or command-to-motion relation; Implementation/evidence: recorded outputs do not execute or support the principal claims."}
    - {score: 0.7, desc: "Report: all canonical scenarios are covered and at least one explicitly controlled comparison or condition-specific experiment beyond aggregate scenario totals tests a central sensing, delivery, safety or resource trade-off; Implementation/evidence: the compared conditions are actually executed with records and metrics supporting the comparison."}
    - {score: 1.0, desc: "Report: complementary experiments examine multiple meaningful communication or control regimes, including a mechanism comparison and a sensitivity or failure-boundary check; Implementation/evidence: condition-level decisions, trajectories and outcomes are executed and distinguish the stated claims beyond canonical aggregates."}

- id: research_analysis
  desc: "evidence-grounded interpretation, conclusions and limitations"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent, materially contradicts the records, or treats unimplemented experiments as facts; Implementation/evidence: code and results cannot support the reported conclusions."}
    - {score: 0.4, desc: "Report: the discussion mainly repeats values or makes unsupported claims and does not explain how map freshness, packet timing, safety stops and control choices affect outcomes or identify material limits; Implementation/evidence: records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: conclusions follow the recorded trajectories and metrics, explain important condition-dependent trends and trade-offs, distinguish observation from inference and state material limitations; Implementation/evidence: code, trace and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence is used to explain mechanisms, quantify principal trade-offs or sensitivities, test the explanation against contrasting conditions or a failure boundary, and state scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
