---
id: T027_event_triggered_cooperative_transport
name: Event-Triggered Distributed Control for Cooperative Payload Transport
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

Four small mobile robots push a rigid rectangular payload across a bounded workspace. Each robot observes the payload and its own contact wrench, and can broadcast a local estimate of the team's resultant force and torque to robots within a wireless radius. Broadcasts share a two-packet-per- slot channel and may be delayed or lost. A robot therefore acts using a local estimate that can be stale or inconsistent with its neighbours. The payload must reach a target pose without collision; temporary robot failure and a narrow passage make communication and contact geometry consequential.

The uplink is robot-to-robot. A broadcast generated in slot `t` becomes usable by a receiver at its arrival boundary, and the receiver uses it only in later slot decisions. The payload and robot states are updated after the action in the same slot. A failed robot executes zero motion and zero wrench during its fixed failure interval.

### Research Problem

Implement a self-contained, non-training distributed controller and event-trigger rule. At every slot choose each active robot's planar velocity, bounded planar force, bounded torque, and whether to broadcast one estimate. Use only the robot's own observation, past actions, and arrived neighbour messages. Coordinate contact forces and torques while respecting message age, channel capacity, communication bursts, changing topology, safety, and a robot dropout. Study the trade-off between pose accuracy, safe completion, and sparse fresh communication. Use one reusable state-driven procedure for every canonical scenario; do not provide scenario- or slot-indexed action tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios with horizon `H=60`, slot length `Delta t=0.1 s`, square domain `[0,8]x[0,8]`, payload mass `M=10`, inertia `J=0.06`, payload size `0.8x0.4`, four robots of radius `0.1`, and state boundaries `0,...,60`. The payload starts at the listed pose with zero linear/angular velocity. Body-frame contact points, in order for robots `0..3`, are `(-0.45,-0.25),(-0.45,0.25),(0.45,-0.25),(0.45,0.25)`; initial robot positions are their rigid transforms. The scenario table fixes goals, closed obstacle rectangles, communication radius, loss parameters `(q_s, ell_s, c_s)`, and any failure interval.

| scenario | initial payload | goal payload | obstacles | radius | q / ell / c | failure |
|---|---|---|---|---:|---|---|
| `straight_transport` | `(1.5,1.5,0)` | `(3.0,1.5,0)` | none | `2.0` | `0.10/0.02/11` | none |
| `rotation_balance` | `(1.5,5.8,0)` | `(2.6,5.8,pi/2)` | none | `1.4` | `0.15/0.04/23` | none |
| `narrow_passage` | `(4.0,2.0,0)` | `(5.4,2.0,0)` | `[4.5,5.0]x[0,1.15]`, `[4.5,5.0]x[2.85,4.0]` | `1.2` | `0.20/0.06/37` | none |
| `link_partition` | `(2.0,4.0,0)` | `(3.4,4.6,pi/4)` | `[2.5,2.9]x[4.9,6.0]` | `0.85` | `0.30/0.10/53` | none |
| `robot_dropout` | `(5.5,5.5,0)` | `(6.7,5.0,-pi/6)` | `[6.0,6.4]x[5.8,7.0]` | `1.3` | `0.20/0.08/71` | robot 2 in slots `16..31` |

#### State, action and task model

At slot `t`, each active robot selects `u_i=(ux,uy)` with `||u_i||2<=0.3`, force `f_i=(fx,fy)` with `||f_i||2<=1.5`, torque `tau_i` with `|tau_i|<=0.8`, and a broadcast flag. At the action-start pose, the force and torque are valid only when robot `i` is within `0.35 m` of its transformed contact point. Let `F=sum(valid f_i)` and `T=sum((c_i-p) cross valid_f_i + valid_tau_i)`. The payload update is `a=(F-0.2*v)/M`, `alpha=(T-0.1*r)/J`, `v'=clip(v+0.1*a,[-1,1]^2)`, `r'=clip(r+0.1*alpha,-2,2)`, `p'=p+0.1*v'`, and `theta'=wrap(theta+0.1*r')`. Robot positions update as `q_i'=q_i+0.1*u_i`. In failure slots the robot must use `u_i=(0,0)`, `f_i=(0,0)`, `tau_i=0`, and `broadcast=false`.

#### Information availability and event order

At a boundary, first admit messages arriving there and discard cached estimates older than four slots. Each receiver keeps one cache entry per sender and replaces it only with a strictly newer generation slot. Each receiver then forms its estimate from its own measured resultant and available cached neighbour estimates. It chooses actions and a broadcast using only this information, executes the payload/robot update, checks safety and goal, registers the broadcast's fixed delay/loss outcome, and records the next boundary. The link hash is `h_k=((1103515245(t+1)+12345(i+1)+2654435761(j+1)+97c_s+53k) mod 1000)/1000`. Delay is two slots when `h_0<q_s`, otherwise one; the packet is lost when `h_1<ell_s`. A broadcast is copied to all neighbours at generation, but counts once toward the channel. A slot serves at most two broadcasts and each robot has at most three broadcasts in every consecutive four-slot window. Each robot also knows its own action-start position and activity state. Every robot observes all closed obstacle rectangles whose point-to-rectangle distance from its action-start position is at most `1.0 m`; these lexicographically sorted rectangles form its only obstacle observation and are serialized as `[xmin,xmax,ymin,ymax]`. The action-start communication graph contains active robots only: an inactive robot has no graph neighbours, has an empty `neighbor_ids` list, and neither sends nor receives a packet in that slot. Future link outcomes, future states, other robots' unshared current actions, and obstacle truth outside this local observation are unavailable to the controller.

#### Legal actions and boundary conditions

Velocity, force, torque, contact radius, failure, channel, age, burst, domain and safety bounds are hard constraints. A safety violation occurs when the payload center enters a closed obstacle rectangle, leaves the domain, has center-to-rectangle distance below `0.15 m`, or any pair of robots is closer than `0.2 m`. `collision` is true exactly when the center is in a closed obstacle or a robot pair is closer than `0.2 m`; `payload_safe` tests only the payload center's domain and `0.15 m` obstacle clearance. Goal arrival is the first boundary with payload position error at most `0.1 m` and wrapped heading error at most `10 degrees`; the episode still records all 60 slots.

#### Task metrics

`primary_metrics_1` is the mean over scenarios of `0.7*clip(1-A_pos/(60*(8*sqrt(2))^2),0,1)+0.3*clip(1-A_yaw/(60*pi^2),0,1)`, where the errors are computed from independently replayed payload boundaries `1..60`. `primary_metrics_2` is the mean indicator of reaching the goal by boundary 60 with no safety violation. `auxiliary_metrics_1` is the mean `clip(1-broadcast_count/(60*2),0,1)`. `auxiliary_metrics_2` is the mean of `0.6*estimate_freshness+0.4*connectivity_ratio`, where freshness averages the `4*3*60` receiver/sender action-start samples at boundaries `0..59` as `clip(1-age/4,0,1)` for a usable cache and zero otherwise, and connectivity is the start-of-slot connected-graph indicator. The weighted normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result record contract

`results.json` is strict JSON with only `scenarios`, one record per canonical name. A scenario record has exactly `scenario`, `payload`, `robots`, `estimates`, `messages`, `contacts`, and `safety`. `payload` contains 61 ordered records with exactly `state_t,x,y,theta,vx,vy,omega,position_error,heading_error,goal_reached,safety_violation`; the two errors are current-boundary Euclidean position error and absolute wrapped heading error, while `goal_reached` and `safety_violation` remain true after their first occurrence.

`robots` contains 244 records with exactly `robot,state_t,x,y,executed_vx,executed_vy,applied_force_x,applied_force_y,applied_torque,active,observation,recent_broadcast_count,cmd_vx,cmd_vy,force_x,force_y,torque,broadcast,broadcast_content`. `broadcast_content` has exactly `force_x,force_y,torque`, is the sender's chosen local team-resultant estimate, and is bounded to force components in `[-6,6]` and torque in `[-8,8]`. `observation` has exactly `payload_x,payload_y,payload_theta,payload_vx,payload_vy,payload_omega,contact_x,contact_y,contact_distance,measured_force_x,measured_force_y,measured_torque,neighbor_ids,visible_obstacles`; for an active row robot, `neighbor_ids` is the sorted list of other active robots within the radio radius, while it is the empty list for an inactive row robot. `visible_obstacles` follows the `1.0 m` rule. At boundary `state_t`, measured wrench is the robot's applied wrench from slot `state_t-1`, or zero at state `0`. `recent_broadcast_count` is the number of this robot's broadcasts in the preceding three slots before the current decision. Records `0..59` hold submitted commands and independently derived executed/applied values. The terminal record at state `60` uses the terminal position, observation and activity state, preserves the preceding-three-slot count, and has zero command, broadcast content, executed and applied values with `broadcast=false`.

`estimates` contains 720 records with exactly `robot,state_t,neighbor,force_x,force_y,torque,age,available,fusion_error` for distinct robots and action-start boundaries `state_t=0,...,59`. For an available sender cache, force/torque equal its message content and integer age is `state_t-generation_slot` in `0..4`; otherwise force/torque are zero, age is `null`, and `available=false`. `fusion_error` is the Euclidean error of this cached triplet, using zeros when unavailable, from that sender's applied wrench in the preceding slot, with a zero reference at state `0`.

`messages` is dynamic and each row has exactly `broadcast_id,receiver,sender,generation_slot,arrival_state,receivers,content,delivered,used,channel_unit`. One broadcast receives the next integer ID even if its receiver set is empty; it produces one row per action-start neighbour. `content` has exactly `force_x,force_y,torque,generation_slot,sender` and equals the sender's robot-row `broadcast_content` plus its key fields. `arrival_state` is integer `generation_slot+1` or `+2`, hence may be `61`; `delivered` is true only if the hash does not lose it and arrival is at most `60`. At each action-start arrival boundary `0..59`, only the greatest-generation delivered packet per receiver-sender pair may strictly replace an older cache entry and receive `used=true`; all other packets, including arrivals at boundary `60` or later, have `used=false`. `receivers` is the sorted action-start receiver set and `channel_unit` is JSON boolean `true`; receiver rows for one ID still count as one broadcast.

`contacts` contains 240 records with exactly `robot,t,contact_distance,valid_force_x,valid_force_y,valid_torque,contact_degree`; degree counts other active robots within `0.55 m` at action start. `safety` contains 61 records with exactly `state_t,obstacle_distance,min_robot_distance,collision,payload_safe,connected,goal_reached`; when no obstacles exist, obstacle distance is `8.0`. Numeric comparisons use absolute tolerance `1e-6`. Payload and safety are ascending by `state_t`, robots by `(robot,state_t)`, estimates by `(robot,state_t,neighbor)`, contacts by `(robot,t)`, and messages by `(broadcast_id,receiver)`; all numeric values are finite and all flags are JSON booleans.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root directory must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must overwrite `results.json`. `report.md` must have non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact scenario structures", weight: 3}
- {id: result_evidence, desc: "payload, robot, estimate, message, contact and safety evidence", weight: 4}
- {id: canonical_coverage, desc: "all fixed scenario, state, robot and neighbour keys", weight: 4}
- {id: decision_legality, desc: "action, contact, failure, channel and information constraints", weight: 4}
- {id: replay_validity, desc: "independent dynamics, communication and safety replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed payload pose quality", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed safe completion quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed broadcast efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed estimate freshness and connectivity", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of payload dynamics, contact, local information, communication and safety model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent transport model; Implementation/evidence: no executable payload or communication process."}
    - {score: 0.4, desc: "Report: omits a decisive contact, message-age, topology or safety relation; Implementation/evidence: only part of the fixed process is represented."}
    - {score: 0.7, desc: "Report: accurately defines local observations, contact dynamics, link timing, constraints and objectives; Implementation/evidence: core model runs across all units with minor omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order and how estimates, topology, contact torque and dropout affect pose; Implementation/evidence: records faithfully realize these relations."}
- id: method_design
  desc: "quality of reusable distributed control and event triggering"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: fixed actions/full communication or local-only control ignores age, contact and capacity; Implementation/evidence: scenario tables or material divergence."}
    - {score: 0.7, desc: "Report: legal unified method handles force/torque, contacts, message age and capacity; Implementation/evidence: one procedure covers all scenes."}
    - {score: 1.0, desc: "Report: justified reproducible method jointly handles consensus estimates, event triggers, topology, dropout and pose trade-offs; Implementation/evidence: local evolving state drives all actions."}
- id: experimental_coverage
  desc: "coverage and controlled communication/transport evidence"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one team or communication condition; Implementation/evidence: no evidence for event-triggered coupling."}
    - {score: 0.7, desc: "Report: all canonical scenes plus a controlled capacity, topology, dropout or target comparison; Implementation/evidence: condition-level records support it."}
    - {score: 1.0, desc: "Report: complementary single-factor comparisons quantify sparse communication, topology failure, contact geometry and dropout boundaries; Implementation/evidence: records distinguish mechanisms."}
- id: research_analysis
  desc: "evidence-grounded interpretation and limitations"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradictory analysis; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: repeats pose or message counts without causal age/topology/contact analysis or limitations; Implementation/evidence: claims unsupported."}
    - {score: 0.7, desc: "Report: replay evidence supports conditional trends and trade-offs with limitations; Implementation/evidence: code and records agree."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies event-trigger, connectivity, torque and dropout interactions and failure boundaries with scope; Implementation/evidence: condition-level records reproduce claims."}
