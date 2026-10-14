---
id: T001_uav_waypoint_delivery
name: Budgeted UAV Waypoint Control under Delayed Downlink Commands
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

A ground station supervises a small unmanned aerial vehicle (UAV) during a time-sensitive inspection mission. The station sends planar command-and-control (C&C) velocity commands over a wireless downlink. The UAV has no independent route planner in this study: in each one-second control slot it executes the most recent command that has reached it. A packet may be lost or may arrive after a delay, so the UAV can continue executing an older command. A command that becomes too old is discarded and the UAV then hovers until another command arrives. The timing and content of downlink decisions therefore affect both the continuous flight trajectory and whether the UAV reaches prescribed locations by their deadlines.

The station receives the UAV position after each slot and knows the prescribed route and the fixed downlink rule for the operating condition. A transmission consumes a limited communication budget, and commands cannot change arbitrarily fast. The task is a repeatable numerical model of this closed loop; it does not require a radio implementation, a perception system or a hardware interface.

### Research Problem

Develop a self-contained simulation and a non-training control and communication policy for the closed-loop waypoint mission. At every control slot, decide whether to transmit a new planar velocity command and, when transmitting, choose its value. The policy must coordinate a finite transmission budget, a local limit on transmission bursts, delayed or lost command delivery, the age of the command currently available to the UAV and a limit on changes between transmitted velocities.

The research objective is to keep the UAV close to its scheduled route and reach the waypoints on time while using transmissions selectively and preserving usable command freshness. A decision can affect later slots through the communication budget, the burst constraint, the command-age state, pending deliveries and the velocity-change constraint. Explain the resulting control--communication trade-off and provide evidence for the design choices in the report. The policy and numerical procedure are your decisions, subject to the fixed model and result contract below.

Use one state-driven, reusable policy procedure across all canonical scenarios. The procedure must derive decisions from the route, current state, communication history and fixed task inputs; do not provide pre-filled action tables or branches keyed to individual scenario names and slot sequences.

### Experimental Setting and Constraints

#### Canonical scenarios and reference route

There are five canonical scenarios. Each scenario has five physical waypoints listed in order. The waypoint deadlines are slots `0, 6, 12, 18, 24`; waypoint `0` is the common initial location. For scenario `s`, let `w[s,k]` be waypoint `k` (`k = 0,...,4`). The scheduled reference point `g[s,t]` for every integer `t` in `0,...,24` is the linear interpolation between the adjacent waypoints whose deadlines surround `t`:

```text
deadline = [0, 6, 12, 18, 24]
g[s,t] = (1-a) * w[s,k] + a * w[s,k+1]
a = (t - deadline[k]) / 6
```

| scenario | `w0` | `w1` | `w2` | `w3` | `w4` |
|---|---|---|---|---|---|
| `steady_square` | `[0,0]` | `[4,0]` | `[4,4]` | `[0,4]` | `[0,0]` |
| `crossing_route` | `[0,0]` | `[3,3]` | `[-3,3]` | `[-3,-3]` | `[3,-3]` |
| `zigzag_route` | `[0,0]` | `[-3,2]` | `[2,4]` | `[-2,5]` | `[0,0]` |
| `late_turn` | `[0,0]` | `[1,0]` | `[2,0]` | `[2,5]` | `[-3,5]` |
| `radial_return` | `[0,0]` | `[-2,-2]` | `[0,-4]` | `[4,-4]` | `[0,0]` |

The UAV starts at `p[s,0] = [0,0]` metres. Slot `t` is the interval `[t,t+1)` seconds for `t = 0,...,23`. A transmitted command is a planar velocity `u = [u_x,u_y]` in metres per second. It is legal when `-1 <= u_x <= 1`, `-1 <= u_y <= 1`, and `u_x^2 + u_y^2 <= 1`. The executed velocity is held constant during the slot and

```text
p[s,t+1] = p[s,t] + u_exec[s,t].
```

#### Downlink event model

At the beginning of slot `t`, packets scheduled to arrive at `t` from earlier slots are admitted. Among admitted packets, the packet with the largest generation slot becomes active; a packet never replaces a newer active packet. The station then chooses at most one packet for slot `t`. A transmitted packet is generated at `t`, and its delivery outcome and delay are given by the scenario rule below. A successfully delivered zero-delay packet is admitted immediately and can control slot `t`. Packets with arrival slot `24` or later are outside the mission and are ignored.

The active packet has age `t - generation_slot`. Before executing the slot, an active packet whose age is greater than `4` is discarded. An active packet of age `0,...,4` supplies its velocity; with no active packet the UAV executes `[0,0]`.

| scenario | slots that are lost (`delivery = 0`) | delay in slots for every other slot |
|---|---|---|
| `steady_square` | none | `0` |
| `crossing_route` | `5,6,7,8` | `2` for `9,10,11,12`; `0` otherwise |
| `zigzag_route` | `2,3,4,14,15,16,22` | `1` for even `t`; `0` for odd `t` |
| `late_turn` | `0,1,12,13,14,20` | `3` for `4,5,6,7,16,17`; `1` otherwise |
| `radial_return` | `3,4,5,18,19` | `2` for even `t`; `0` for odd `t` |

The station knows the route and the fixed delivery and delay rules at the beginning of the mission. It may use the current position and all observations from previous slots. The submitted record must still contain every slot-level transmission and command decision.

#### Communication and command constraints

Each scenario has a transmission budget of `12`. A transmission consumes one unit when it is scheduled, regardless of whether its packet is delivered. For every consecutive block of four control slots, at most `3` transmissions may be scheduled. For two successive transmitted commands in chronological generation order, their Euclidean velocity difference must be at most `0.75` metres per second; the first transmitted command has no predecessor. A lost or delayed packet still counts toward the budget and the velocity-difference rule because the command was generated.

These constraints are applied to the slot-level decisions before delivery outcomes are evaluated.

#### Waypoint and trajectory evaluation

The four non-initial waypoints are evaluated at `t = 6,12,18,24`. A waypoint is successful when the Euclidean distance between the independently replayed UAV position and its waypoint is at most `1.0` metre. The trajectory error is evaluated at the 24 post-slot positions `p[s,t]`, `t = 1,...,24`, against `g[s,t]`.

#### Result contract

Running `python run.py` must create or overwrite a UTF-8 `results.json` in the project root. Its root is exactly an object with one required key, `scenarios`; no other root keys are used. The value is an array containing exactly one record for each of the five scenario names. Every scenario record has exactly the keys `scenario`, `decisions`, `trace` and `waypoints`.

`decisions` contains exactly 24 records, one for each `t = 0,...,23`, in any order. Each decision has the keys `t`, `transmit` and `velocity`. `transmit` is a boolean. When it is `true`, `velocity` is a two-element finite JSON number array satisfying the velocity bounds, the per-scenario budget, the four-slot transmission limit and the successive-command difference limit above. When it is `false`, `velocity` is `null`.

`trace` contains exactly 24 records for `t = 0,...,23`. Its `position` is the position at the beginning of the slot after arrival handling and before movement; `active_sequence` is the generation slot of the command selected for that slot or `null`; `command_age` is `t - active_sequence` or `null`; and `executed_velocity` is the velocity actually used during the slot (zero when no command is active).

`waypoints` contains exactly four records for `(waypoint, t) = (1,6),(2,12),(3,18),(4,24)`. Each record gives the independently replayable endpoint `position`, its `distance` from the target, and a boolean `success`. Positions and distances are finite numbers. The record order is not semantically significant, but every required key, type, index and scenario identifier must occur exactly once. Numeric comparisons use an absolute tolerance of `1e-6` metres.

The decisions are the input to independent replay. The trace and waypoint fields are evidence of that replay; the grader derives delivery outcomes, active commands, positions, distances and all metrics from the decisions and the fixed model.

#### Task metrics

The official metrics are computed from the independent replay over all five scenarios:

* `completion_rate` is the fraction of the 20 waypoint deadlines that are successful.
* `tracking_quality` is `max(0, 1 - mse/1800)`, where `mse` is the mean of  `||p[s,t]-g[s,t]||^2` over the 120 pairs `(s,t)` with `t=1,...,24`.
* `transmission_efficiency` is `clip(1 - n_tx/60, 0, 1)`, where `n_tx` is the total number of  transmitted packets over the five scenarios.
* `freshness_quality` is the mean over the 120 slots of `max(0,1-age/4)` for slots with an active  command, and `0` for slots without one.

Here `clip(x,0,1) = min(1,max(0,x))`. Higher values are better. The metric formulas and the fixed `1800` tracking scale are the complete performance definition; numbers stated in the report are not used as scoring inputs.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root of your project must contain `run.py`, `report.md` and `results.json`. You may organize the implementation across multiple files; `run.py` is the only official entry point and should start the simulation without interactive input. `report.md` must contain six substantive Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`. The report must describe the actual implementation and results, including the assumptions, the communication-control trade-off and the limitations supported by the experiments.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json parses as strict JSON and matches the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "each scenario provides typed decision, trace-state and waypoint evidence required by the result contract", weight: 4}
- {id: canonical_coverage, desc: "the five canonical scenarios and every declared decision, trace and waypoint key occur exactly once", weight: 4}
- {id: decision_legality, desc: "each transmitted command satisfies its declared velocity, budget, burst and continuity constraints", weight: 4}
- {id: replay_validity, desc: "recorded trace and waypoint fields match the independently replayed downlink and motion transitions", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed waypoint completion rate", weight: 9}
- {id: primary_metrics_2, desc: "independently replayed trajectory tracking quality", weight: 6}
- {id: auxiliary_metrics_1, desc: "independently replayed transmission efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed command freshness quality", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the communication, command-availability, motion, budget and deadline model in the report and implementation"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given, or the stated UAV, downlink, timing or waypoint model is materially wrong; Implementation/evidence: no executable representation or recorded evidence corresponds to the claimed process."}
    - {score: 0.4, desc: "Report: only part of the route, command, communication, information or constraint model is defined and a decisive relation is omitted; Implementation/evidence: the code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: the principal states, actions, fixed link events, information boundary, constraints, deadlines, objectives and causal relations are accurately defined; Implementation/evidence: the code and records instantiate that core model across all canonical scenarios, with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely specifies event order, command delivery and ageing, motion, budgets, deadlines and the communication-to-task trade-off; Implementation/evidence: the code and records faithfully realize these relations and link decisions to outcomes without material divergence."}

- id: method_design
  desc: "quality and task relevance of the non-training policy for transmission timing and velocity control"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, a prohibited training method is used, or the proposed policy conflicts with the fixed information, action or hard constraints; Implementation/evidence: no runnable procedure or supporting decision record is present."}
    - {score: 0.4, desc: "Report: the method is partial, ignores a decisive timing, freshness or communication trade-off, or presents scenario- or slot-specific schedules as the primary solution; Implementation/evidence: the procedure relies on pre-filled actions or materially differs from the report. A fixed schedule explicitly used as a comparison is assessed under experimental coverage."}
    - {score: 0.7, desc: "Report: a legal, reproducible and reusable state-driven policy derives transmission timing and velocity from route, command state and communication history while addressing budget, burst, continuity and tracking trade-offs; Implementation/evidence: the same procedure generates decisions across all canonical scenarios and the records support its operation, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified and reproducible procedure accounts for deadline risk, delivery delay, freshness, budget, burst history and velocity continuity; Implementation/evidence: the implementation derives decisions from route and evolving state without scenario-specific action tables, with records supporting the decisive choices."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of the implemented experiments for the five routes and their communication conditions"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose is provided; Implementation/evidence: the code and records contain no checkable experimental execution."}
    - {score: 0.4, desc: "Report: the work shows only isolated examples, fixed action schedules or incomplete canonical coverage and does not examine the central communication-control relation; Implementation/evidence: the recorded outputs do not execute or support the principal experimental claims."}
    - {score: 0.7, desc: "Report: all five canonical scenarios are covered and at least one explicitly identified controlled comparison or condition-specific experiment beyond merely contrasting canonical aggregates directly tests a central mechanism or trade-off among delivery, command freshness, communication limits and waypoint performance; Implementation/evidence: both sides or conditions are actually executed with comparable inputs and records that support the stated comparison."}
    - {score: 1.0, desc: "Report: complementary, non-redundant experiments cover at least two meaningful controlled communication or control conditions or decision regimes, including a mechanism-focused comparison and a sensitivity or failure-boundary check; Implementation/evidence: they are executed with condition-level decisions, trajectories, outcomes and metrics that distinguish the claims beyond aggregate canonical scores."}

- id: research_analysis
  desc: "evidence-grounded interpretation, conclusions and limitations"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the recorded findings, or presents unimplemented experiments as facts; Implementation/evidence: the code and records cannot support the reported observations or conclusions."}
    - {score: 0.4, desc: "Report: the discussion mainly repeats values or makes unsupported causal claims and does not explain how delivery timing, command age or communication limits alter trajectory and deadlines or identify material limits; Implementation/evidence: the records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: conclusions are grounded in the actual records, explain important condition-dependent trends and trade-offs, distinguish observation from inference and state material limitations; Implementation/evidence: the implementation, traces and metrics support those observations and explanations."}
    - {score: 1.0, desc: "Report: the analysis uses controlled evidence to explain how delivery timing, command age, communication limits and velocity choices shape outcomes, quantify the principal trade-offs or sensitivities, test the explanation against contrasting evidence or a failure boundary, and state scope and unknowns; Implementation/evidence: reproducible condition-level decisions, trajectories and metrics support the central conclusions."}
