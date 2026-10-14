---
id: T040_async_cloud_navigation
name: Asynchronous Cloud-Intent Alignment for Safe Mobile Robot Navigation
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

A mobile robot receives low-rate cloud navigation intents while moving continuously among static and dynamic obstacles. Each cloud packet contains five waypoints expressed in the robot's ego frame at the packet's generation pose. Network loss, delay, and jitter mean that a delivered packet may refer to a pose the robot left several control cycles earlier. Directly treating those stale coordinates as current can misalign the cloud intent; aggressive correction can instead violate local clearance or excite a lagged actuator. The edge controller has a pose history, current execution state, and sixteen local range rays, but it does not receive the mission goal or future network and obstacle events.

### Research Problem

Design and evaluate one deterministic, non-learning, state-driven controller that selects among arrived cloud packets, aligns an accepted packet's anchor-frame waypoints with the current robot frame, and produces locally safe velocity targets. The method should balance timely goal arrival, obstacle clearance, cloud-intent tracking, command freshness, acceleration, and motion energy under loss, delay bursts, moving obstacles, narrow passages, and different actuator lag. One reusable procedure must handle all fifteen public scenario variants from its action-start information and history; it must not supply scenario-, variant-, packet-, route-stage-, or absolute-slot-indexed command tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

The workspace is `[0,12] x [0,12] m`. Every unit has one holonomic robot, `H=100` action slots `t=0..99` of `dt=0.25 s`, and state boundaries `b=0..100`. Initial actual world velocity and the previous world target velocity are `(0,0) m/s`. The safety clearance is `0.35 m`, the risk clearance is `0.60 m`, the maximum target and actual speed is `0.8 m/s`, and the world-target slew limit is `0.05 m/s` per slot. Each of the five base scenarios has `variant v` in `{0,1,2}`. The fifteen `(scenario,variant)` pairs are the canonical units, with key `scenario=<name>|variant=<v>`, and all terminate at boundary `100`.

The goal and obstacle columns below are fixed environment truth used by cloud generation and replay. They are not policy-visible coordinates: the controller may use the goal only through packets that have actually arrived, and dynamic obstacles only through the current LiDAR rays. A static entry `(x,y)/r` is a circle with center `(x,y)` and radius `r`. Define `tri_m(n)=m-|((n mod (2m))-m)|` for integer boundary `n`.

| scenario | initial pose `(x,y,heading)` | goal truth | static circles | dynamic-circle center/radius at boundary `b` | deadline | base delay; loss |
|---|---|---|---|---|---:|---|
| `static_chicane` | `(1,1,pi/4)` | `(10,10)` | `(4,3)/0.8`, `(4,7)/0.8`, `(8,5)/0.8` | none | `82` | `1+(g mod 2)`; `0.08` |
| `dynamic_crossing` | `(1,10,-pi/4)` | `(10,1)` | `(5,5)/0.7` | `(2+tri_7(b+3v),6)/0.30` | `86` | `1`; `0.10` |
| `latency_spike` | `(1,6,0)` | `(10,6)` | `(5,3)/0.7`, `(5,9)/0.7` | none | `88` | `1`, except generation boundaries `36+3v..48+3v` use `4`; `0.08` |
| `narrow_doorway` | `(1,1,pi/4)` | `(10,10)` | `(6,4)/1.0`, `(6,8)/1.0` | none | `84` | `2`; `0.14` |
| `actuator_lag` | `(1,10,-pi/4)` | `(10,2)` | `(4,6)/0.7`, `(8,6)/0.7` | `(6,2+tri_5(b+3v))/0.30` | `90` | `2`; `0.10` |

Angles use radians. Heading zero points along world `+x`; positive rotation is counter-clockwise. A pose is `(x,y,heading)`. `R(theta)` is the standard two-dimensional counter-clockwise rotation matrix. Euclidean vector norms are denoted by `||.||`.

#### Cloud packets and fixed task model

The generation-boundary set is

`G_v={1+5k+v | k is a nonnegative integer and 1+5k+v<100}`.

At each `g` in `G_v`, the cloud anchors a packet to the robot pose at boundary `g` and freezes any dynamic circle at that boundary. Starting from the anchor position, it constructs five world waypoints. At each step it considers the sixteen world unit directions

`d_j=(cos(2*pi*j/16),sin(2*pi*j/16)), j=0..15`.

A candidate endpoint is the previous waypoint plus `0.4*d_j`. It is admissible exactly when its center clearance to each workspace boundary, static-circle boundary, and frozen dynamic-circle boundary is at least `0.60 m`. Among admissible directions, choose the one maximizing the inner product between `d_j` and the unit vector from the previous waypoint to the goal truth; ties use smaller `j`. If the previous waypoint equals the goal, use zero as that unit vector. If no direction is admissible, repeat the previous waypoint. The packet payload is the five offsets `R(-heading_g)*(waypoint-anchor_position)` in the anchor ego frame. It does not contain the anchor pose; the controller can recover that pose only from its own history at `g`.

Packet ID is `packet=<scenario>|variant=<v>|g=<g>`. Let `D_s(g)` be the scenario's base delay from the table and

`D_(s,v)(g)=D_s(g)+((g+v) mod 2)`.

The arrival boundary is `g+D_(s,v)(g)`. Define

`z=uint64(SHA256(UTF-8(packet_id+"|loss_phase="+str(4v)))[:16])/2^64`.

The first sixteen hexadecimal SHA-256 characters are interpreted as an unsigned big-endian integer. The packet is lost exactly when `z` is below the scenario loss value; loss has no retransmission. This hash is an environment/replay mechanism only: a policy must not compute or query future hash outcomes, and the controller receives only delivery events that have already been revealed. Generation alone reveals neither loss nor future arrival. A non-lost packet becomes visible only at its arrival boundary. A packet arriving at boundary `100` is recorded by the environment but cannot be used by an action.

For `clip_vec(q,L)`, return `q` when `||q||<=L`, otherwise return `L*q/||q||`. Each submitted ego target `u_t=(u_x,u_y)` becomes world target

`w_t=R(heading_t)*u_t`.

Actual world velocity is

`v_t=clip_vec(lambda_(s,v)*v_(t-1)+(1-lambda_(s,v))*w_t,0.8)`,

where `v_-1=(0,0)`, `lambda=0.65` except that `actuator_lag` uses `0.65+0.08v`. The proposed position is `p*=p_t+0.25*v_t`.

For a point `p`, workspace clearance is `min(p_x,12-p_x,p_y,12-p_y)`. Clearance to a circle with center `c` and radius `r` is `||p-c||-r`. After computing `p*`, advance dynamic circles to boundary `t+1`. If the minimum workspace/circle clearance of `p*` is below `0.35 m`, the fixed safety executor records one safety event for that slot, sets `p_(t+1)=p_t`, and sets executed `v_t=(0,0)`. Multiple simultaneous conflicts still produce one event. Otherwise it accepts `p_(t+1)=p*`. If executed speed is nonzero, `heading_(t+1)=atan2(v_y,v_x)`; otherwise heading is unchanged. A risk slot is an action slot whose accepted or filtered position at boundary `t+1` has minimum clearance below `0.60 m`. The first boundary at which the position is within `0.4 m` of the goal truth is the arrival boundary; late arrival is still recorded.

#### Information availability and event order

At action start `t`, the controller receives exactly:

- the current unit's string `scenario`, integer `variant`, and integer deadline as static context; the public packet-generation, base-delay, loss-threshold, command-age, and actuator-lag laws are also known, but only packet deliveries already revealed by this boundary are realized link outcomes;
- its poses at boundaries `0..t`, its previous executed world velocity, and its previous world target velocity;
- every packet already delivered, each exposing string `packet_id`, integer `generated_boundary`, integer `arrival_boundary`, and five length-two finite ego-frame waypoint vectors; collection order has no semantic effect;
- the current sixteen LiDAR distances indexed by `j=0..15`, its prior packet/velocity decisions, and its first-arrival boundary or null.

LiDAR ray `j` starts at the current robot center in world direction `heading_t+2*pi*j/16`. Its value is the smallest nonnegative distance to the workspace boundary or to a static/dynamic circle boundary at boundary `t`, truncated to `[0,2.5] m`. For a circle, use the smallest nonnegative `rho` satisfying `||p_t+rho*d_j-c||=r`; use zero if the origin is on or inside the circle. Static context permits deadline- and declared-link-aware control, but `scenario`, `variant`, or packet IDs do not authorize a policy to look up the goal truth or analytic obstacle table. The controller cannot use those truths, undelivered or lost packets, computed future hash outcomes, future dynamic positions, or terminal state. The declared link and actuator laws describe future environment behavior but do not reveal future delivery outcomes.

At every nonterminal boundary, events occur in this order:

1. generate and send the packet scheduled at this boundary using the boundary-start pose;
2. reveal every non-lost packet whose arrival boundary is now;
3. update pose history and the prior arrival signal, then generate LiDAR from boundary-`t` geometry;
4. expose the action-start information and read the command;
5. validate the packet reference, null-command braking, speed, and slew;
6. compute the lagged actual velocity and proposed movement;
7. advance the dynamic circle, apply the safety executor, and update pose, risk, arrival, acceleration, and energy evidence for boundary `t+1`.

Boundary `100` only reveals due packet arrivals and has no command. Submitted fields are only the action-start packet selection and target velocity. Packets, pose history, LiDAR, actual velocity, obstacles, safety/risk events, arrival, acceleration, motion energy, and all metrics are fixed-replay evidence, not submitted fields.

#### Legal actions and boundaries

Every slot selects either one currently usable packet or null. A non-null `packet_id` must refer to a packet visible at action start with command age `t-arrival_boundary` in `0..10`; a delivered packet may be reused or replaced while usable. If `packet_id` is null, the world target must equal the unique deterministic braking target

`brake(w_(t-1)) = 0` when `||w_(t-1)||<=0.05`, otherwise `(1-0.05/||w_(t-1)||)*w_(t-1)`.

Thus a gap between packets can decelerate safely but cannot introduce a new direction or acceleration. In all cases, `||w_t||<=0.8` and `||w_t-w_(t-1)||<=0.05`. All submitted numbers must be finite; Boolean values are not numbers. The global comparison tolerance is `EPS=1e-6`: it applies to legality, clearance/admissibility, goal-radius equality, vector equality, and waypoint-direction tie handling. These action-start rules are the entire legality gate. Safety filtering, risk, loss, delay, and failure to arrive are execution outcomes and do not become hidden legality requirements.

#### Task metrics

Compute each metric separately per canonical unit, then take the arithmetic mean over all fifteen units.

- `primary_metrics_1`: deadline-arrival quality is `1` when first arrival is no later than the deadline, `0.5*clip(1-(arrival-deadline)/max(1,100-deadline),0,1)` for late arrival, and `0` when no arrival occurs. This metric does not include safety.
- `primary_metrics_2`: `0.7*clip(1-risk_steps/100,0,1)+0.3*clip(1-safety_events/100,0,1)`. It is the only metric that uses risk or safety events.
- `primary_metrics_3`: for a non-null command, recover the selected packet's anchor pose from pose history, transform its five ego waypoints to world points, prepend the anchor position, and measure the action-start position's shortest Euclidean distance to this polyline. For a null command, error is `0.8 m`. The metric is `clip(1-mean(error_t)/0.8,0,1)`.
- `auxiliary_metrics_1`: `0.6*useful_packet_fraction+0.4*command_freshness`. The useful fraction is the number of unique packets selected at least once divided by the number delivered by boundaries `0..99`, or zero for an empty denominator. Freshness is the mean of `clip(1-(t-arrival_boundary)/10,0,1)` over non-null command slots, or zero for an empty set.
- `auxiliary_metrics_2`: let `valid_command_fraction` be the number of non-null command slots divided by `100`. Let actual acceleration be `||v_t-v_(t-1)||/0.25 m/s^2`, including a safety-filtered zero velocity, and motion energy be `sum_t 0.5*||v_t||^2`. The metric is `valid_command_fraction*(0.5*clip(1-mean_acceleration/0.20,0,1)+0.5*clip(1-motion_energy/32,0,1))`.

All metric values lie in `[0,1]`. Normalized task performance is

`(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result record contract

`results.json` must be strict finite JSON with exactly the root key `scenarios`. Its array contains exactly one record for every canonical `(scenario,variant)` unit, in any order. Each unit record has exactly string `scenario`, integer `variant`, and array `commands`. Every commands array contains exactly one record for each `slot=0..99`, in any order.

Every command has exactly `slot`, `packet_id`, `u_x`, and `u_y`. `slot` is a JSON integer in `0..99`. `packet_id` is either a non-empty string or JSON null. `u_x` and `u_y` are finite JSON numbers; Boolean values are invalid. Only these minimal decisions are submitted. The fixed model derives every packet, waypoint, pose, LiDAR sample, actual velocity, obstacle state, mission event, and metric.

Validation responsibilities do not overlap. `result_evidence` checks exact unit/command fields, primitive types, nullability, and finite numbers. `canonical_coverage` checks the fifteen unit keys and all one hundred slot keys. `decision_legality` checks action-start packet visibility and age, the exact null-command brake, target-speed norm, and world-target slew. `replay_validity` checks deterministic packet, sensor, dynamics, safety, and mission reconstruction. Any representation that produces commands satisfying these public semantics is accepted; no private policy or internal data layout is required.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain three readable ordinary files, not symbolic links: `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the causal information boundary, the unified alignment and safety method, all canonical units, controlled evidence, and supported limitations. The code and report must expose whether policy decisions use only arrived packets and declared action-start state rather than goal truth or future environment outcomes. Additional project files and report sections are allowed, but external services, hidden files, and unpublished reference code are not.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains six non-empty required headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenario container", weight: 3}
- {id: result_evidence, desc: "unit and command records contain the exact finite minimal decision fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen variants and each unit's one hundred slot keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "commands obey arrived-packet, age, braking, speed, and world-target slew rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives packets, LiDAR, lagged motion, safety, and arrival", weight: 6}
- {id: primary_metrics_1, desc: "deadline-sensitive mission arrival quality", weight: 8}
- {id: primary_metrics_2, desc: "obstacle-clearance and safety-executor quality", weight: 4}
- {id: primary_metrics_3, desc: "aligned cloud-intent tracking quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "delivered-packet usefulness and selected-command freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "communication-conditioned acceleration and motion-energy efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the asynchronous packet, pose-history, LiDAR, lagged motion, and safety model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent asynchronous navigation formulation; Implementation/evidence: no executable packet-to-motion process corresponds to the task."}
    - {score: 0.4, desc: "Report: covers only waypoint following or obstacle avoidance and omits a decisive anchor, arrival, information, or actuator relation; Implementation/evidence: the executable state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: accurately defines anchor-frame packets, pose history, arrived information, LiDAR, lagged velocity, clearance, and deadlines; Implementation/evidence: one reusable process implements the core model across all units with only limited omissions."}
    - {score: 1.0, desc: "Report: precisely closes event order, information boundaries, spatial alignment, execution lag, and arrival-safety trade-offs; Implementation/evidence: all variants faithfully realize those relations with reconstructible evidence and no material divergence."}
- id: method_design
  desc: "quality of the unified cloud-intent alignment, packet-selection, local safety, and velocity controller"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or uses prohibited training; Implementation/evidence: no legal unified controller is present."}
    - {score: 0.4, desc: "Report: directly treats stale ego waypoints as current, relies on fixed smoothing or velocity tables, or ignores a decisive clearance, age, or lag consequence; Implementation/evidence: decisions are pre-filled by unit or materially differ from the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning packet, alignment, local-safety, and velocity method that treats the principal temporal and dynamic constraints; Implementation/evidence: the same causal procedure covers all variants using permitted current state and history."}
    - {score: 1.0, desc: "Report: justifies joint use of anchor pose, command freshness, local range geometry, actuator response, target slew, clearance, and deadline pressure; Implementation/evidence: current action-start state drives decisions without unit or slot command tables or forbidden future inputs."}
- id: experimental_coverage
  desc: "coverage of canonical variants, alignment mechanisms, network conditions, obstacles, and execution boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only a single link or obstacle condition is summarized and alignment is not tested; Implementation/evidence: condition evidence cannot support the main claims."}
    - {score: 0.7, desc: "Report: covers every canonical variant and one controlled two-sided alignment, delay, loss, obstacle, lag, or safety comparison; Implementation/evidence: both conditions execute with attributable inputs and outputs and enough code or records to reconstruct the comparison."}
    - {score: 1.0, desc: "Report: complementary non-redundant controls separately test the alignment mechanism and a network, obstacle, actuator, or safety failure boundary; Implementation/evidence: each condition is actually executed with reconstructible condition-level evidence beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based interpretation of alignment, network, clearance, execution, and arrival trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: conclusions are unsupported."}
    - {score: 0.4, desc: "Report: mainly lists arrival, error, or delay without explaining spatial misalignment, local safety, or limitations; Implementation/evidence: recorded evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: uses actual results to explain command age, anchor alignment, obstacle response, execution lag, and arrival trends while distinguishing observations, inferences, and limitations; Implementation/evidence: the executed conditions support those claims."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify interactions among network degradation, spatial alignment, clearance, actuator response, and deadline outcome and states scope and unknowns; Implementation/evidence: reproducible condition records support the central conclusions."}
