---
id: T056_bandwidth_map_distillation
name: Bandwidth-Aware Keyframe Distillation and Edge Correction for Collaborative Navigation
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

Four indoor robots navigate through assigned regions while building local landmark maps. Their odometry accumulates bias, so selected keyframes and landmarks must cross a time-varying shared uplink to an edge server. The server compares maps from different robots and returns geometric corrections over a slower shared downlink. A large frame may preserve useful overlap but delay other traffic; a stale or lost correction leaves later navigation based on drifted estimates. Frame distillation, wireless service, edge merging and physical progress therefore form one closed loop.

### Research Problem

Design a deterministic, non-training, state-driven procedure that jointly chooses motion, keyframe retention and landmark content, uplink and downlink priorities, edge processing order, and which arrived corrections to apply. One reusable procedure must operate across all scenarios from the information available at each public decision boundary. The objective is to complete the navigation regions with accurate localization, construct a consistent cross-robot edge map, and use bandwidth, edge service and energy efficiently. The method must confront the temporal trade-off between richer frames, queue delay, loop opportunities, correction freshness and remaining task deadlines.

### Experimental Setting and Constraints

#### Canonical scenarios and task model

The grid consists of integer cells `x=0..21,y=0..17`; a cell has side length `1 m`. Robots are `R0,R1,R2,R3`, the edge is at `(11,9)`, and each scenario has `H=150` one-second slots indexed `0..149`, with state boundaries `0..150`. Rectangles below are inclusive obstacle cells. The interference level is low unless the table states a baseline or interval; low, medium and high have numeric levels `0,1,2`. An edge outage disables only edge processing, not radio transfer.

| scenario | initial `R0,R1,R2,R3` | obstacle rectangles | bias scale | interference | edge outage | energy budget |
|---|---|---|---:|---|---|---:|
| `texture_transition` | `(1,1),(20,1),(1,16),(20,16)` | `[10,10]x[0,7]`, `[10,10]x[10,17]` | `0.035` | low | none | `620` |
| `loop_corridor` | `(1,8),(20,8),(11,1),(11,16)` | `[5,16]x[3,3]`, `[5,16]x[14,14]` | `0.045` | medium throughout | none | `640` |
| `bandwidth_surge` | `(1,1),(20,1),(1,16),(20,16)` | `[7,14]x[5,12]` | `0.040` | high at `35..69`, low otherwise | none | `630` |
| `edge_outage` | `(2,2),(19,2),(2,15),(19,15)` | `[9,12]x[4,13]` | `0.050` | medium throughout | `45..74` | `650` |
| `crossing_routes` | `(1,8),(20,8),(11,1),(11,16)` | `[7,9]x[7,10]`, `[13,15]x[7,10]` | `0.055` | medium at `50..89`, low otherwise | none | `660` |

Each robot has two regions that must be completed in the table order. A tuple is `(center_x,center_y,value,deadline)`; the region is the Chebyshev-radius-one set around its center, and the deadline is inclusive. At a post-motion boundary, only the lowest-index incomplete region is eligible for completion. Occupying the second region before the first is complete earns no value and does not consume the second region's later eligibility.

| scenario | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| `texture_transition` | `(6,14,.55,80),(18,14,.45,140)` | `(15,6,.55,80),(3,6,.45,140)` | `(6,3,.55,80),(18,3,.45,140)` | `(15,11,.55,80),(3,11,.45,140)` |
| `loop_corridor` | `(19,8,.55,72),(2,8,.45,142)` | `(2,8,.55,72),(19,8,.45,142)` | `(11,16,.55,72),(11,1,.45,142)` | `(11,1,.55,72),(11,16,.45,142)` |
| `bandwidth_surge` | `(5,14,.55,70),(18,14,.45,142)` | `(16,3,.55,70),(3,3,.45,142)` | `(5,3,.55,70),(18,3,.45,142)` | `(16,14,.55,70),(3,14,.45,142)` |
| `edge_outage` | `(6,13,.55,76),(18,13,.45,142)` | `(15,4,.55,76),(3,4,.45,142)` | `(6,4,.55,76),(18,4,.45,142)` | `(15,13,.55,76),(3,13,.45,142)` |
| `crossing_routes` | `(5,8,.55,65),(18,8,.45,140)` | `(17,8,.55,65),(4,8,.45,140)` | `(11,5,.55,65),(11,13,.45,140)` | `(11,13,.55,65),(11,5,.45,140)` |

A requested action is `stay,N,S,E,W`. First replace an out-of-grid or obstacle-targeting request by `stay`. A requested move is then rejected if its target is any other robot's boundary position or if two or more robots have the same target; all such rejected robots stay. Thus entering a cell occupied at the boundary and swapping cells are forbidden even when the other robot requested a move. All remaining moves execute simultaneously. The real position follows the executed move. The initial estimate equals the initial real position. For robot index `i`, an executed move adds its unit displacement and `b_i=a_s*((-1)^i,(-1)^(i+1))/sqrt(2)` to its estimate; `stay` adds neither. Estimate components are clipped to `[-8,29]` and `[-8,25]`.

The scenario landmark set contains every non-obstacle cell satisfying `(7*x+11*y) mod 9=0`. Its ID is `L_<x>_<y>` and stability is `0.4+0.1*((3*x+5*y) mod 6)`. Keyframe opportunities occur for every robot at slots `0,5,...,145`. A landmark is visible when its Chebyshev distance from the robot's pre-motion real position is at most three and its center-to-center segment does not intersect an obstacle cell. Cells are closed unit squares; every cell whose square intersects the segment, including only at an edge or corner, belongs to the supercover, and the two endpoint cells are removed before the obstacle test. For landmark `(x,y)`, slot `t`, and robot index `i`, its local coordinate is the landmark minus the real position plus the same offset in both components, `0.05*(((13*x+7*y+3*t+i) mod 5)-2)`.

At an opportunity the candidate either submits no landmarks and drops the frame, or selects one to twelve distinct visible landmark IDs. A nonempty choice creates `F_<robot_id>_<slot>`, captures the robot estimate after this slot's selected corrections, and has `96+32*n` bits. It is admitted to the tail of that robot's upload FIFO, whose capacity is eight frames; choosing a nonempty frame when the FIFO is already full is invalid. Successful upload removes a frame. Failed completion resets the whole frame at the FIFO head for automatic retransmission; there is no separate retry decision.

#### Wireless transfer, edge processing and corrections

For the pre-motion robot position and the edge, let `d` be Euclidean distance and `blocked` be one when their supercover line, excluding endpoints, intersects an obstacle. The current link quality is `q=clip(1-d/30-0.15*blocked-0.08*level,0,1)`. At decision boundary `t`, robot `Ri` receives exactly this scalar evaluated from its boundary-`t` real position and the slot-`t` interference level. The same revealed value, without resampling, is used for both directions of `Ri`'s service in slot `t`. The shared uplink and downlink have respectively `1024` and `256` bits per slot and operate independently and simultaneously. A listed robot can use at most `floor(1024*q/32)*32` uplink bits or `floor(256*q/32)*32` downlink bits. Within each direction, robots are visited in the submitted priority order and each receives the minimum of its head packet's remaining bits, its link limit and the shared bits left. Listing an empty queue simply serves zero bits. Partial packets retain their remaining bits at the FIFO head.

When a packet finishes in slot `c`, it succeeds exactly when `((17*i+5*c+kind) mod 100)/100 >= 0.05+0.2*(1-q)`, where `i` is the frame source or correction target robot index and `kind` is zero for a frame and one for a correction. Failure resets the packet's remaining bits to its full payload size and increments the retry count. Success schedules arrival at the start of `c+1+floor(3*(1-q))`. Uplink service, including failed attempts and retries, may not exceed `80000` bits per scenario. Multiple frame arrivals at one boundary are ordered by completion slot, then source robot index, then frame opportunity slot, and are admitted in that order to a capacity-16 edge FIFO; overflow frames are discarded.

Edge service opportunities are slots `0,3,...,147`; outside an outage the candidate may process one frame currently in the edge FIFO or select `null`. The processed frame is compared with every earlier processed frame from another robot, ordered by that earlier frame's processing slot, robot index and opportunity slot. For each shared landmark, form `current.estimate_at_capture+current.local_coordinate - previous.estimate_at_capture-previous.local_coordinate`. Sort the residual values separately for each component. For odd count `n`, that component's median is sorted value `(n-1)/2`; for even `n`, it is the arithmetic mean of sorted values `n/2-1` and `n/2`. These two component medians form the residual. At least three shared IDs and absolute residual components at most `2.5` produce a passed edge. It creates a correction for the current frame's robot with components `clip(-0.5*residual,-4,4)` and ID `C_<current_frame_id>__<previous_frame_id>`.

Passed corrections are appended in the comparison order to the target robot's capacity-16 downlink FIFO; every correction packet has `128` bits and an overflow correction is discarded. A correction's age at boundary `t` is `t-created_slot`. Before the decision, queued and arrived corrections with age greater than 30 expire. A successful downlink arrival joins the target's pending set. The target may apply any subset of its pending, unexpired corrections at most once; all selected vectors are summed, added to the current estimate and clipped once to the estimate bounds. Since summation is commutative, submitted correction ID order has no meaning.

Energy costs are `1` per executed move, `0.2` per retained frame, `0.0005` per served uplink or downlink bit, `0.1` per processed frame and `0.05` per applied correction. Cumulative team energy may not exceed the scenario budget after any slot.

#### Information boundary and slot order

At the start of slot `t`, scheduled arrivals are processed and corrections older than 30 are removed. The fixed simulator maintains real positions solely to generate observations and replay consequences; real positions are not policy inputs. Each robot's local motion, landmark and correction decisions may use the public static model and only these current or historical local fields: its estimate; region table and completed-region indices; cumulative team energy and uplink bits; its prior requested and executed/rejected actions; its upload FIFO packets and already revealed transfer outcomes; its pending correction IDs, vectors, creation slots and ages; its current keyframe observation, if one exists; and its current scalar link probe. A current keyframe observation is precisely the list of visible `(landmark_id,local_x,local_y,stability)` tuples defined above. A local decision may not inspect another robot's estimate, action feedback, region progress, pending corrections, unsent landmark observation, or queued upload-frame content.

The shared uplink/downlink scheduler may use cumulative team energy and uplink bits, all four current scalar link probes and, for each direction, robot ID, queue length, and the head packet's ID, payload bits, remaining bits, creation slot and retry count, plus already revealed transfer outcomes. These are the queue/link metadata visible before the same-slot decisions; the scheduler may not inspect the landmark contents of queued or newly selected upload frames. The edge processor may inspect all arrived and processed frame contents, their public timing metadata, cumulative resource values and the same downlink queue metadata. A unified candidate procedure may coordinate the components only through these interfaces. No decision component may use real robot positions, future landmark observations, future link probes, or a packet completion result before that result is produced. The deterministic success formula defines environment replay and is not an advance outcome oracle for a decision at the same or an earlier boundary.

The candidate submits all decisions for the slot at this boundary. The fixed replay then applies the selected corrections; creates and admits selected frames using the corrected estimate; serves the uplink and downlink independently; executes the selected edge processing and creates corrections; filters and executes motion; updates odometry, region completion and energy; and advances to the next boundary. Radio probes and keyframe visibility use the pre-motion real position. At the post-motion boundary, the current lowest-index incomplete region earns its value when the real position is in that region, the slot index is no later than its inclusive deadline, and the Euclidean estimate error is at most `2 m`; at most one region per robot can complete in a slot.

Arrival and expiration processing occurs only at the starts of decision slots `0..149`. A transfer that completes in slot `149` still counts as a successful transfer, but any arrival scheduled for boundary `150` or later remains in flight and cannot be processed, applied or used in map metrics.

Decision fields therefore mean requests submitted at the stated boundary. The keyframe observation, link quality, pending corrections and queue contents described above are observations or feedback already visible at that boundary. Executed motion, transfer outcomes, arrivals, edge graph, corrected state, resource use and task completion are post-decision facts generated by replay; they are not submitted and cannot be used early.

#### Metrics

All metrics are clipped to `[0,1]` and then averaged equally over the five scenarios. `primary_metrics_1` is `0.6*goal_completion+0.4*localization_quality`. Region values are `0.55,0.45` per robot and are earned only in their required order, so `goal_completion` is earned team value divided by four. `localization_quality=clip(1-RMSE_pose/8,0,1)`, where `RMSE_pose=sqrt(sum_i sum_k ||p_i,k-p_hat_i,k||^2/(4*151))` pools all four robots at the initial boundary `k=0` and all 150 post-slot boundaries `k=1..150`.

`primary_metrics_2=0.6*map_consistency+0.4*collaborative_coverage`. Among processed frames, a landmark is collaborative when at least two robot IDs contributed it. For all observations of collaborative landmarks, compute error between `estimate_at_capture+local_coordinate` and the landmark's fixed coordinate; `map_consistency=clip(1-RMSE/6,0,1)`, or zero when no collaborative landmark exists. `collaborative_coverage` is the fraction of all scenario landmarks that are collaborative.

`primary_metrics_3` is `0.45*clip(1-safety_rejections/(4*H),0,1)+0.30*clip(1-executed_moves/(4*H),0,1)+0.25*clip(1-energy_used/energy_budget,0,1)`. A safety rejection is a requested non-`stay` move filtered by the public obstacle, occupancy or same-target rules, including an out-of-grid target in the obstacle/boundary filtering stage.

`auxiliary_metrics_1` is `0.65*useful_bits/max(1,uplink_served_bits)+0.35*successful_uploads/max(1,generated_frames)`. For each successfully uploaded and processed frame participating in at least one passed edge, its 96-bit header is useful, and each selected landmark appearing in at least one of that frame's passed edges contributes 32 useful bits once. Uplink served bits include failed completions and retries. The upload ratio is zero if no frame is generated.

`auxiliary_metrics_2=0.4*correction_freshness+0.3*edge_wait_score+0.3*productive_service_fraction`. Correction freshness averages `clip(1-(applied_slot-created_slot)/30,0,1)` over applied corrections, or is zero with none. Edge wait score averages `clip(1-(processed_slot-arrival_slot)/30,0,1)` over processed frames, or is zero with none. A processed frame is productive when it creates at least one passed edge; the fraction is zero with no processing. The formal task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20` after all validity gates pass.

#### Result contract

`results.json` is strict JSON with root exactly `{"scenarios":[...]}`. There is exactly one scenario object for each canonical scenario; its keys are exactly `scenario,slots`. Scenario array order is irrelevant. Each `slots` array contains exactly one object for every slot `0..149`, in any order, with exact keys `slot,robot_decisions,uplink_order,downlink_order,edge_frame_id`. Each `robot_decisions` array contains exactly one object for every `R0..R3`, in any order, with exact keys `robot_id,action,landmark_ids,apply_correction_ids`.

`slot` is a non-boolean integer. IDs and actions are strings. `edge_frame_id` is a string or `null`. `action` belongs to `stay,N,S,E,W`. `landmark_ids` and `apply_correction_ids` are arrays of distinct strings interpreted as unordered sets. `landmark_ids` is empty outside a keyframe opportunity; at an opportunity it is empty or contains one to twelve currently visible IDs. `uplink_order` and `downlink_order` are arrays containing any distinct subset of `R0..R3`; their array order is the submitted service priority and is semantically significant. `edge_frame_id` is `null` or a currently queued frame ID; it must be `null` outside edge opportunities or during an outage.

All objects reject undeclared keys. The canonical keys are scenario name, slot, and robot ID within a slot. Frame and correction IDs have the exact public forms defined above; there are no candidate- assigned dynamic IDs. The grader accepts every ordering permitted above and does not compare against a reference policy or representation. All trajectories, visible observations, link values, packet states, graph edges, corrections, mission outcomes and metric summaries are derived by the grader and must not be added to this minimal result object. Replay uses absolute tolerance `1e-6` for finite floating-point state.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite the root `results.json` from the project code and fixed conditions. `report.md` must contain non-empty headings named exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The executed project and the `Results` and `Research Analysis` sections must include condition-level evidence for at least one controlled mechanism comparison and one communication sensitivity or failure-boundary comparison, using comparable fixed inputs. These experiments may write additional candidate-defined UTF-8 JSON or CSV artifacts or may report their condition records directly in `report.md`; they do not change the exact canonical `results.json` contract. The report must also identify the policy inputs used by each decision component and explain how the stated information interfaces are respected.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root, scenario and slot containers", weight: 3}
- {id: result_evidence, desc: "slot and robot decision fields have exact typed schemas", weight: 4}
- {id: canonical_coverage, desc: "every scenario, slot and robot key appears exactly once", weight: 4}
- {id: decision_legality, desc: "local action, set, service-order and null boundaries are legal", weight: 4}
- {id: replay_validity, desc: "visibility, queues, transfer, edge, correction, motion, energy and task dynamics replay", weight: 6}
- {id: primary_metrics_1, desc: "deadline region completion and localization quality", weight: 8}
- {id: primary_metrics_2, desc: "cross-robot map consistency and collaborative landmark coverage", weight: 4}
- {id: primary_metrics_3, desc: "safety, movement and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful landmark bits and successful upload efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "correction freshness, edge wait and productive service", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the collaborative mapping, communication and navigation model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: the formulation is absent or materially wrong; Implementation/evidence: no executable mapping-communication process is present."}
    - {score: 0.4, desc: "Report: keyframes, shared service, edge matching or correction timing has a decisive omission; Implementation/evidence: only a partial model is implemented or it materially departs from the report."}
    - {score: 0.7, desc: "Report: observations, distillation, FIFOs, edge merging, correction and navigation relations are accurately defined; Implementation/evidence: the core model runs across every scenario."}
    - {score: 1.0, desc: "Report: information timing, causal couplings, resource trade-offs and boundaries are precise; Implementation/evidence: code and records faithfully realize every central relation without material divergence."}
- id: method_design
  desc: "quality of the unified state-driven distillation and scheduling method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal method is specified or a prohibited method is used; Implementation/evidence: no runnable reusable procedure supports the claim."}
    - {score: 0.4, desc: "Report: a decisive landmark-bandwidth-correction coupling is ignored or unit schedules dominate; Implementation/evidence: decisions are pre-filled or materially differ from the report."}
    - {score: 0.7, desc: "Report: a reproducible unified method uses state to address navigation, distillation, transfer, edge service and correction; Implementation/evidence: one procedure runs across all units with limited planning depth."}
    - {score: 1.0, desc: "Report: the justified method jointly handles information value, cross-robot overlap, link and FIFO age, correction suitability and task slack; Implementation/evidence: evolving state drives decisive choices without scenario action tables."}
- id: experimental_coverage
  desc: "coverage of distillation, communication, edge and correction mechanisms"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment is provided; Implementation/evidence: no checkable experimental execution exists."}
    - {score: 0.4, desc: "Report: experiments are partial or aggregate-only and do not test the central relation; Implementation/evidence: outputs cannot support the experimental claims."}
    - {score: 0.7, desc: "Report: all scenarios run and one comparable executed condition tests a distillation, communication, edge or correction mechanism; Implementation/evidence: both conditions use comparable inputs and recorded results support the comparison."}
    - {score: 1.0, desc: "Report: complementary mechanism and sensitivity or failure-boundary comparisons test frames, landmarks, congestion and correction age; Implementation/evidence: reproducible condition-level code, records and metrics distinguish the claims."}
- id: research_analysis
  desc: "quality of evidence-based analysis of map and navigation trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts results; Implementation/evidence: the records cannot support the conclusions."}
    - {score: 0.4, desc: "Report: values are repeated without grounded mechanism or material limitations; Implementation/evidence: the asserted causal analysis lacks support."}
    - {score: 0.7, desc: "Report: actual evidence explains important distillation, congestion, loop, correction and task trends while separating observation, inference and limits; Implementation/evidence: code and results support those conclusions."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies central interactions, tests explanations against a contrast or failure boundary, and limits scope; Implementation/evidence: reproducible condition records fully support the central claims."}
