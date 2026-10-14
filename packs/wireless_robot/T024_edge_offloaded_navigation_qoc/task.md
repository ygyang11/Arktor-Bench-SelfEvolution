---
id: T024_edge_offloaded_navigation_qoc
name: Quality-Aware Non-Holonomic Navigation under Edge-Controlled Wireless Delays
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

Three automated guided vehicles receive navigation commands from an edge controller over a shared wireless link. A command that is delayed, lost, or waiting behind another command leaves the vehicle executing an older command. The resulting command age changes the conservative separation distance used for safety, while the non-holonomic heading constraint makes stale commands especially costly in crossings and turns. The vehicles must reach their targets before their deadlines while keeping the communication load and command staleness under control.

### Research Problem

Build one deterministic, non-learning controller and simulator for all five canonical scenarios. At every half-second slot, each vehicle generates a candidate linear/angular command and decides whether to enqueue it using `best_effort` or `reliable` delivery. The same reusable procedure must derive all decisions from the public scenario inputs, the complete frozen joint state at the decision boundary, and prior history; its three action components are committed simultaneously. Design choices include the feedback law, collision avoidance, send-trigger policy, reliability mode, finite prediction, and experiment code. Study the trade-off between trajectory quality, timely safe arrival, effective control updates, and command freshness.

### Experimental Setting and Constraints

#### Canonical Evaluation Units

There are five scenarios, each with slots `t=0,...,59` and state boundaries `0,...,60`. One slot is `0.5` seconds. The workspace is `[0,12] x [0,8]` metres. Each row lists `(x,y,theta)` starts, `(x,y)` goals, inclusive goal deadlines, and closed rectangular obstacles. Angles are radians.

| scenario | starts for robots `0,1,2` | goals for robots `0,1,2` | deadlines | obstacles |
|---|---|---|---|---|
| `open_crossing` | `(1,1,0),(1,7,0),(10,4,3.141593)` | `(11,7),(11,1),(1,4)` | `42,42,42` | none |
| `narrow_merge` | `(1,1,0),(1,7,0),(2,4,0)` | `(11,4),(10,6),(10,2)` | `54,54,54` | `[5,0,7,3]`, `[5,5,7,8]` |
| `delay_corridor` | `(1,4,0),(2,1,1.570796),(2,7,-1.570796)` | `(11,4),(10,7),(10,1)` | `52,52,52` | `[4,0,8,2.5]`, `[4,5.5,8,8]` |
| `shared_bottleneck` | `(1,1,0),(1,7,0),(10,4,3.141593)` | `(11,7),(11,1),(1,4)` | `48,48,48` | `[5,3,7,5]` |
| `recovery_turn` | `(1,1,0),(1,7,0),(10,4,3.141593)` | `(10,4),(10,6),(2,4)` | `50,50,50` | `[4,0,6,3]`, `[4,5,6,8]` |

The public link level at decision slot `t` is determined by the following rule; it is visible before the three decisions. `good` applies to `open_crossing` when `t mod 12 < 8`, to `narrow_merge` when `t < 14` or `t >= 42`, and to `recovery_turn` when `t < 16` or `t >= 36`. `delay_corridor` is `degraded` for `t<18`, `outage` for `18<=t<=27`, `degraded` for `28<=t<=39`, and `good` otherwise. `shared_bottleneck` is `outage` for `30<=t<=35`, `good` when `t mod 10 < 4`, and `degraded` otherwise. For `narrow_merge`, slots not covered by its `good` rule are `outage` for `22<=t<=29` and `degraded` otherwise. For `recovery_turn`, slots not covered by its `good` rule are `outage` for `16<=t<=22` and `degraded` otherwise. The level is common to all packets in a slot. For `open_crossing`, every slot not covered by its `good` rule is `degraded`.

#### State, Actions, and Task Model

Each robot is a disc of radius `0.25` metres, with state `(x,y,theta)`. An active command `(v,omega)` is held for the whole slot and updates the state by `x'=x+0.5*v*cos(theta)`, `y'=y+0.5*v*sin(theta)`, and `theta'=wrap(theta+0.5*omega)`, where `wrap` returns an angle in `[-pi,pi)`. Commands satisfy `0<=v<=0.6` and `-1.2<=omega<=1.2`. Every generated command, whether sent or not, must differ from the preceding generated command of that robot by at most `0.2` in `v` and `0.6` in `omega`; the preceding command at slot zero is `(0,0)`.

Each robot has a FIFO command queue of capacity `4`. A decision with `send=true` enqueues its generated command only if the queue is not full, so `send=true` is an enqueue request rather than a guaranteed admission. An admitted packet receives a monotonically increasing per-robot sequence number starting at `0`. The shared channel freezes at most two selected packets per slot, ordered by original enqueue slot, robot ID, and sequence number. Each selected packet leaves its FIFO immediately before the attempt. A successful packet is then in flight, outside every FIFO and its capacity, until arrival. A failed `reliable` packet with fewer than three attempts is reinserted with its original ordering key; a failed `best_effort` packet and a third failed `reliable` attempt are discarded. A packet can be selected at most once in one slot. The first attempt number is `1`; retries have attempt numbers `2` and `3`. The candidate submits only `send` and `mode`; it never submits attempts or arrival events.

For the current public level, a `good` attempt succeeds and arrives at the next boundary; a `degraded` attempt succeeds iff the environment residual `u(s,t,robot,sequence,attempt)<=0.65` and then arrives two boundaries later; an `outage` attempt fails. The residual is fixed by `u=((37*(scenario_index+1)+19*(t+1)+11*(robot+1)+7*(sequence+1)+5*(attempt+1)) mod 997 + 0.5)/997`. Here `scenario_index` is the scenario's zero-based position in the canonical table, `t` is the zero-based slot in which this attempt is made, `sequence` is zero-based, and `attempt` is the one-based attempt number. The formula is published only to make the deterministic environment reproducible: the environment evaluates it after all three actions for the slot are frozen, and a decision procedure must not evaluate or receive the current or future residual. At an arrival boundary, delivered packets for each robot are ordered by generation slot and sequence number. If one or more are newer than the pre-boundary active command, only the newest becomes active and counts as one effective update; all delivered packets still leave the in-flight set. Packets arriving after boundary `60` are discarded. Initial active commands are `(0,0)` with age `8` at boundary zero. If no newer command becomes active, age increases by one per boundary and is clipped at `8`.

Workspace and closed-obstacle membership are tested on robot centres; the `0.25`-metre radius is represented by the `0.50`-metre base centre-to-centre separation rather than by dilating workspace or obstacles. First compute all three simultaneous proposed full states `(x',y',theta')`. A proposed state whose centre is outside the workspace or inside a closed obstacle is unsafe, and both robots in a pair whose proposed centre distance is below `0.50 + 0.05*max(age_i,age_j)` are unsafe. These flags are computed once from the complete proposed joint state. Every flagged robot restores its complete previous `(x,y,theta)`; unflagged robots take their proposed states, and no second safety test is made after any restoration. The proposed-state test records a safety violation and makes the canonical trajectory physically illegal. A first goal entry within `0.3` at boundary `b` is timely and safe only if `b<=deadline` and that robot had no safety violation in execution slots `0,...,b-1`. The simulator continues after arrival.

#### Information Availability and Event Order

At boundary `0`, each robot starts at its tabulated state with generated and active command `(0,0)`, active generation `-1`, and age `8`; every FIFO, in-flight set, pending-arrival list, goal history, safety history, and packet counter is empty, and each robot's next sequence number is `0`. The initial `attempts`, `admitted`, and `effective_updates` counters are all zero. The geometry, starts, goals, deadlines, public link-level rule, command limits, and queue limits are fixed decision inputs. The residual formula belongs to the environment transition, not to the decision view. At boundary `t`, the controller sees all robots' current states, active commands, ages, FIFO entries and retry counts, prior goal and safety history, arrivals already processed at that boundary, and packets requeued after failed reliable attempts in earlier slots. A successfully attempted packet that is in flight is not exposed until its arrival boundary. The controller cannot query current or future residuals, current-slot attempts, future arrivals, or post-motion safety events. Candidate code must therefore isolate residual evaluation in the simulator transition that runs only after the simultaneous joint action has returned; no component may use a same-slot effect of another component. This procedural information boundary must also be explicit in the report and implementation.

Each slot follows this order: (1) process arrivals scheduled for the boundary and update each active command and age; (2) expose the frozen state and public link level and read all three actions; (3) check local command domains, generation deltas, and queue admission; (4) execute the active command for motion, apply the one-pass full-state restoration rule, and test goals; (5) enqueue legal new packets and schedule up to two attempts in the stated order, using the environment residual and fixed delay; and (6) advance to the next boundary. At that boundary, the next occurrence of step (1) processes its arrivals exactly once before the state is exposed or recorded as a checkpoint. A generated command and its `send/mode` are decision fields. Sparse audit checkpoints are candidate-produced execution-after evidence; all other position, command, age, packet, safety, goal, reference, and metric facts are grader-derived.

#### Legal Actions and Boundaries

An action is locally legal only if its velocity is finite and within bounds, its generation deltas respect the preceding generated command, `send=false` has `mode=null`, and `send=true` has exactly one of `best_effort` or `reliable`. A send whose queue is full is illegal and is not enqueued; replay continues. A finite in-domain command that violates a generation slew is still the submitted generated command for subsequent slots and may be admitted if its send request is otherwise valid. A `send=true` action with a null or invalid mode produces no packet; `send=false` with a non-null mode also produces no packet. A queue admission is not itself a channel attempt. Physical safety violations are illegal for the canonical trajectory but use the deterministic full-state stay replacement. Duplicate, missing, or unknown canonical keys fail coverage. All numeric comparisons use absolute tolerance `1e-6`; integer fields reject Boolean values.

#### Task Metrics

The fixed replay independently computes four values in `[0,1]`, all higher-is-better, with equal weight over robots and scenarios as stated here. The reference trajectory is generated once per scenario and robot from the same kinematic update, starting at the public start state, using an obstacle-independent nominal free-space controller toward that robot's goal: `v_ref=clip(0.45*distance,0,0.6)` and `omega_ref=clip(2*wrap(atan2(goal_y-y,goal_x-x)-theta),-1.2,1.2)`. It is not affected by candidate packets or obstacles. `primary_metrics_1` is the mean of `clip(1-sum_{b=1..60}((x_b-xref_b)^2+(y_b-yref_b)^2)/(60*5.76),0,1)`. The `5.76 m^2` scale is derived from the maximum open-loop translation over the capped initial age window: `8 slots * 0.5 s * 0.6 m/s = 2.4 m`, whose squared displacement is `5.76 m^2`. `primary_metrics_2` is the mean over scenarios of the fraction of robots with timely safe first goal entries. For each scenario, `effective_updates` counts boundary-level active-command replacements by packets that arrive by boundary 60; at most one update per robot per boundary is effective. `total_attempts` counts all initial and reliable retry attempts. `auxiliary_metrics_1` is the scenario mean of `0` when no effective update exists, otherwise `effective_updates/total_attempts`. `auxiliary_metrics_2` is the scenario mean of `0.6*mean_{robots,t=0..59}(clip(1-age_t/8,0,1))+0.4*effective_update_admission_ratio`, where `age_t` is the active-command age at the start of execution slot `t`, and the ratio is effective updates divided by admitted sent packets and is `0` when no packet is admitted.

The normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result Record Contract

`results.json` is a strict object with sole key `scenarios`. It contains one object per scenario with exact keys `scenario`, `slots`, and `audit`. `slots` contains exactly one record for each `t=0,...,59` with exact keys `t` and `actions`; each `actions` list contains exactly one record for robots `0,1,2`. Each action has exactly `robot`, `velocity`, `send`, and `mode`. `velocity` is a two-number list `[v,omega]`; `send` is Boolean; `mode` is `null` when `send=false`, otherwise `best_effort` or `reliable`. The canonical action key is `(scenario,t,robot)`, for 900 records.

Each scene's `audit` has exact keys `checkpoints`, `packet_summary`, and `goal_arrivals`. It is a sparse audit projection produced by the candidate's simulator and retained only for comparison with the grader's independent replay; it is not a second source for legality or task metrics.

- `checkpoints` contains boundaries `20`, `40`, and `60` exactly once. Each checkpoint has exact  keys `boundary` and `robots`, and contains robots `0`, `1`, and `2` exactly once. Each robot object  has exact keys `robot`, `position`, `heading`, `active_velocity`, `active_generation`, `age`, and  `unsafe`. At boundary `b`, `position=[x,y]` and `heading` are the state after execution slot  `b-1`; `active_velocity`, `active_generation`, and `age` are the active-command state after arrivals  at boundary `b`; and `unsafe` is the Boolean proposed-state safety flag from execution slot `b-1`.  Positions are finite workspace coordinates, `heading` is finite in `[-pi,pi)`,  `active_velocity=[v,omega]` obeys the command bounds, `active_generation` is an integer in  `[-1,59]`, and `age` is an integer in `[0,8]`.
- `packet_summary` has exact integer fields `attempts`, `admitted`, and `effective_updates` over all  60 slots. Their respective ranges are `[0,120]`, `[0,180]`, and `[0,120]`, with meanings identical  to the metric definitions.
- `goal_arrivals` is a length-three list indexed by robot ID. Each item is `null` if that robot never  first enters its goal by boundary `60`, otherwise the integer first-entry boundary in `[1,60]`.

Scenario, slot-action, checkpoint, and checkpoint-robot array order has no meaning; `goal_arrivals` is robot-indexed. Floating audit values are compared with absolute tolerance `1e-6`, while IDs, integers, Booleans, and `null` are exact. Full trajectories, in-flight packets, FIFO states, individual attempts, residual values, references, and metrics must not be submitted. They are derived by the grader and have no independent audit role. Duplicate JSON members, NaN, and infinities are invalid.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must generate or overwrite `results.json`. The report must contain non-empty sections with exact headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Controlled comparisons are optional to the canonical result contract. If reported, a comparison must be generated by reusable code that emits a clearly named JSON artifact in addition to canonical `results.json`; each condition must state exactly one changed factor and value, keep all other task inputs and the decision procedure fixed, and include the 900 canonical actions, sparse audits in the schema above, and all four replayed metric values. Permitted one-factor changes are the public link schedule, good/degraded delay, degraded success threshold, queue capacity, channel capacity, age-separation coefficient, or a stated controller parameter. Such artifacts are evidence for the report and are not part of the canonical submission.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official Python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the new results.json is strict JSON with the declared root and scenario containers", weight: 3}
- {id: result_evidence, desc: "canonical actions and sparse scene audits have the declared typed fields", weight: 4}
- {id: canonical_coverage, desc: "all 900 scenario-slot-robot action keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "generation deltas, queue admission, send-mode relation, workspace/obstacle motion, and separation safety follow the public rules", weight: 4}
- {id: replay_validity, desc: "sparse motion, command, channel, goal, and safety audits match fixed-model replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed reference-trajectory QoC", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed timely safe goal arrival", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed effective-update communication efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed command freshness and delivery efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the non-holonomic edge-control, queue, delay, age, safety, and QoC model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent non-holonomic communication model is given or the stated dynamics and event order are materially wrong; Implementation/evidence: no executable state process corresponds to the claim."}
    - {score: 0.4, desc: "Report: only part of motion, command holding, queue, link, age, safety, or objective relations is defined; Implementation/evidence: the code or records cover only a partial model or depart materially."}
    - {score: 0.7, desc: "Report: public states, actions, information boundary, event order, queue, link, age, safety, deadlines, and QoC are accurately formulated; Implementation/evidence: one procedure realizes the core model across all scenarios."}
    - {score: 1.0, desc: "Report: the formulation precisely explains how delay, loss, retries, stale commands, non-holonomic motion, and safety change QoC and arrival; Implementation/evidence: transitions and derived outcomes are faithfully realized without material divergence."}
- id: method_design
  desc: "quality of the unified non-training control and send-trigger procedure"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, prohibited training, or information violation; Implementation/evidence: no runnable unified procedure, or decisions query the current/future residual or another unavailable effect."}
    - {score: 0.4, desc: "Report: fixed periodic or scenario-specific commands ignore queue, age, reliability, or safety coupling; Implementation/evidence: primary decisions use pre-filled schedules or materially differ from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method addresses command generation, send timing, reliability, age, motion, deadlines, and the residual information boundary; Implementation/evidence: the same method produces all canonical actions without querying current/future residuals, with limited planning depth."}
    - {score: 1.0, desc: "Report: a justified state-driven method jointly reasons over non-holonomic control, queue/retry consequences, age-dependent safety, communication load, goal deadlines, and observable feedback; Implementation/evidence: decisive actions are reproducibly derived from evolving visible state, with residual evaluation isolated after joint-action commitment and no unit tables."}
- id: experimental_coverage
  desc: "coverage and diagnostic value of controlled QoC and communication experiments"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable canonical or controlled execution."}
    - {score: 0.4, desc: "Report: only isolated scenarios or aggregate QoC values are shown without testing a communication-motion relation; Implementation/evidence: condition evidence does not support the claim."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one strict single-factor comparison of the public link schedule, delay, degraded threshold, queue/channel capacity, age-separation coefficient, or a stated controller parameter are covered; Implementation/evidence: reusable code emits both condition records with the canonical action/audit schema and four replayed metrics."}
    - {score: 1.0, desc: "Report: complementary comparisons cover a mechanism and a failure boundary, quantifying freshness, retries, safety, QoC, and arrival effects; Implementation/evidence: every condition has reproducible action-level outputs and replayed metrics beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-grounded interpretation of communication-aware non-holonomic navigation"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts recorded outcomes; Implementation/evidence: code and outputs cannot support the conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats QoC or arrival values, makes unsupported causal claims, or omits material limits; Implementation/evidence: records do not support the interpretation."}
    - {score: 0.7, desc: "Report: actual records explain conditional trends in delay, retries, age, safety, and motion, distinguish observation from inference, and state limitations; Implementation/evidence: canonical and controlled outputs support those findings."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies the central communication-motion trade-offs, tests a causal explanation against a contrasting condition or failure boundary, and limits scope; Implementation/evidence: reproducible condition-level traces and metrics support every central conclusion."}
