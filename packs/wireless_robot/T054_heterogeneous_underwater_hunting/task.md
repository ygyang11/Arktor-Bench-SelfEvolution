---
id: T054_heterogeneous_underwater_hunting
name: Age-Aware UAV-USV-UUV Target Hunting with Delayed Relay Information
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

An aerial UAV searches a bounded water surface while a USV relays reports over the directed `A0->S0` wireless hop and then one of the directed `S0->Di` acoustic hops. The UUV team must use only locally obtained observations and delayed reports to approach a moving target and hold a separated formation. A report or sonar hit that has arrived at a slot boundary can affect that slot's intended move; service outcomes produced later in the slot first affect the next boundary. Limited capacity, packet loss, queueing and report age therefore determine whether information reaches a UUV in time to affect its next move.

### Research Problem

Design a deterministic, non-training online method for one UAV (`A0`), one USV (`S0`), and three UUVs (`D0,D1,D2`) that searches for and continuously captures a moving target. The method must cover all five canonical scenarios through one state-driven procedure. It may use the public map, its own local state, revealed observations, link probes, acknowledgements and history, but may not read true target coordinates, another vehicle's hidden state, future interference or future packet outcomes. Do not use machine learning, training, reinforcement learning or pre-filled scenario routes.

### Experimental Setting and Constraints

#### Canonical evaluation units and environment

The grid is `0<=x<24, 0<=y<20`, `H=96` slots of one second. `A0` uses `stay,N1,S1,E1,W1,N2,S2,E2,W2` and may fly over islands. `S0` and each UUV use `stay,N,S,E,W`; underwater vehicles may not enter an obstacle core. The canonical scenarios are:

|scenario|A0,S0,D0,D1,D2,target|obstacle core|current `(c_t)` / interference|deadlines|energy A/S/D|
|---|---|---|---|---|---|
|open_basin|`(1,1),(12,1),(2,17),(5,17),(8,17),(20,16)`|none|`(0,0)`, `I=0`|search 32, capture 90|150/110/100|
|island_shadow|`(1,18),(2,2),(2,16),(5,18),(8,16),(20,3)`|`[9,14]x[6,13]`|`(0,0)`, `I=2`|38,92|155/115/105|
|shear_current|`(1,1),(12,2),(2,18),(6,18),(10,18),(21,15)`|`[10,13]x[8,11]`|`t<48:(1,0)`, else `(-1,0)`, `I=1+floor(t/24)`|34,92|155/120/110|
|split_escape|`(2,10),(12,1),(2,2),(2,10),(2,18),(21,10)`|`[10,13]x[7,12]`|`(0,0)`, `I=3`|30,88|150/115/105|
|deadline_burst|`(1,1),(12,1),(1,18),(5,18),(9,18),(21,17)`|`[7,9]x[8,12]` and `[15,17]x[5,9]`|`(0,0)`, `I=5` for slots 24..47, else 2|26,78|145/105/95|

#### State, actions and task model

For `S0`, the UUVs and the target, add the current vector to the intended displacement and choose the legal action with minimum Euclidean error; ties are `stay,N,E,S,W`. For UUVs scale the current by `0.5` first. Out-of-grid, obstacle-core or insufficient-energy moves execute `stay`. Move cost per traversed cell is `1.0` for A0 (`N2/S2/E2/W2` therefore cost `2.0`), `1.2` for S0, and `1.5` for each UUV; an A0 scan costs `0.25`, and each served payload unit costs `0.05` at its sender.

At slot start A0 scans all cells within Chebyshev distance 2 and UUV sonar is exact when within Chebyshev distance 2 of the target. Every search cell starts with AoI 24; each slot first increments and clips AoI to 24, then scanned cells reset to zero at slot end. The task starts in `search` and enters `hunting` in the next slot after a scan or sonar hit; `first_detection_slot` is the slot of that first hit, not the following boundary. In search the target maximizes Manhattan distance from A0; in hunting it maximizes Euclidean distance from the arithmetic UUV centroid. Target tie order is `E,N,W,S,stay`, then the same projection rule is applied. The true target is used only by the environment and independent replay.

The target and every vehicle choose their intended action from the slot-start state. The target first selects the phase-specific maximizing action and then applies the stated current projection; vehicle actions are projected in the same way when required. Packet arrivals and the automatic scan/sonar update are processed at the slot-start boundary and may inform that slot's decision; a decision cannot use a service outcome or any event produced later in the same slot.

#### Information and communication event order

For the deterministic supercover, a segment crossing a grid corner includes both side-touching cells before the diagonal corner cell; an obstacle in either side cell therefore sets `shadow=1`.

Reports are `coarse` (one unit, lower-left of the target's 4x4 block) or `fine` (three units, exact cell). A0-S0 wireless capacity is 4 units/slot. S0-Di acoustic links share capacity 2 and only one acoustic link is served per slot. Let `d` be the Euclidean distance between the link endpoints. Quality is `q_w=clip(1-d/30-0.12*shadow-0.03*I,0,1)` and `q_a=clip(1-d/22-0.08*depth_i-0.025*I,0,1)`, with `depth_i=1,2,3` for `D0,D1,D2`, respectively. In the successful-hop expression, the zero-based `receiver_index` is `0` for `S0`, `1` for `D0`, `2` for `D1`, and `3` for `D2`. Quality below 0.2 is unavailable; `shadow` is `1` exactly when the deterministic supercover grid cells strictly between the two endpoints intersect an obstacle core, and `0` otherwise; it is used only by `q_w`. Delay is `1+floor(3*(1-q))` for wireless and `3+floor(3*(1-q))` for acoustic. A successful hop uses `((13*x_s+7*y_s+5*t+3*receiver_index) mod 100)/100 >= 0.08+0.25*(1-q)`. Four directed FIFOs have capacity 12 units. A report may be copied at S0; each copy is an independent queued packet. First transmissions plus retransmissions are at most 180 units per scenario. Report age starts at the UAV creation slot and is clipped at 12 for recorded fields; control eligibility uses the underlying age, so age over 12 or arrival after H cannot control a UUV.

The four link probes are automatic sender-side observations made at the slot boundary before service. Every `(link_id,slot)` row is mandatory: `observed_available` is true exactly when the public link quality is at least 0.2, and `observed_quality` is that quality when available and null otherwise. Probes do not consume energy and do not reveal future quality or delivery. Service is attempted only on an available link and on the current FIFO head. A service unit is affordable only when the sender has at least `0.05` energy per unit at service time; otherwise the channel must be idle. The automatic scan is charged first, then service energy, then the already-selected movement energy; a move that cannot pay its complete move cost executes `stay`. The danger area is the one-cell Chebyshev outer ring around each obstacle core (the core itself remains an obstacle); entering either an obstacle core or this danger ring is recorded. `safety_events` includes collisions, danger/obstacle entry, and energy exhaustion caused by an unaffordable action; it is a cumulative integer count.

The fixed slot order is: process prior arrivals and update local observations and AoI; choose vehicle motion, target motion, report generation, forwarding and service order from the slot-start state (the A0 scan and all link probes are automatic); serve links and schedule outcomes; execute the already-selected vehicle and target moves; update estimates, age, capture and energy; record post-event evidence. `state_trace[0]` and `mission_state[0]` are slot-0 boundary states; row `t+1` is after slot `t`. Within the records, `intended_actions` and nonzero service requests are decisions submitted at the decision boundary; observation and probe fields are information available at that boundary; executed actions, state, packet, queue and mission fields are post-event evidence produced by the fixed replay order. The detailed event order is authoritative whenever a field's timing is relevant. At a boundary, prior packet arrivals are processed in deterministic link/queue replay order, then the local scan or sonar update is processed; a slot-`t` action may use either update, while `state_trace[t+1]` records the resulting estimate. If multiple eligible packet updates have the same boundary, their order is the fixed link order `A0-S0,S0-D0,S0-D1,S0-D2` and then ascending packet identifier using its JSON text representation; a local hit is processed last. Fresh arrivals and local hits update the control memory in that order; an over-age arrival remains independent arrival evidence and does not displace a still-fresh control memory. If no fresh control memory exists, the latest over-age event may be represented as stale evidence. This tie rule is used for the counterfactual report calculation as well.

Vehicles are projected and checked in the fixed order `A0,S0,D0,D1,D2`; each candidate is compared with the current position map at that instant (earlier vehicles have already been updated and later vehicles remain at their slot-start positions). A match records one collision event for the moving vehicle, even if the other vehicle later moves away.

For evidence indexing, boundary row `t` is the pre-event snapshot for slot `t`, while row `t+1` contains the effects of slot `t`. A scan/sonar hit or an arrival labelled `arrival_slot=t` may be used internally when choosing slot-`t` decisions, but its resulting estimate is first represented in `state_trace[t+1]`; therefore a non-null estimate in row `t` must be traceable to an event with slot `<t`. A fresh control memory is the latest packet/local-hit event whose root report age is at most `12`; a later over-age arrival remains independent arrival evidence and does not displace that fresh control memory. If no fresh control memory exists, the latest over-age event is stale evidence. For a fresh memory, coordinates and clipped age must match that event. For stale evidence, a candidate may either retain its coordinates with `estimate_age=12` or clear all three estimate fields to `null`; these are equivalent non-control representations, and neither may authorize a non-stay UUV action. This convention keeps decision-time information and post-event evidence distinct.

A local scan or sonar hit has underlying age `0` at its event slot and age `boundary_slot-event_slot` at a later boundary (so a hit in slot `t` is recorded with age `1` in `state_trace[t+1]`); the twelve-slot freshness rule applies to this local memory as well as to packet-root age.

#### Legal actions and boundary conditions

An estimate is a control input only when it was obtained by that vehicle from its own scan/sonar or from an arrived packet whose underlying age at the decision boundary is at most 12. A UUV with no such fresh estimate may autonomously choose only `stay`; a UUV may choose a non-stay tracking action only when its recorded estimate is the latest locally available fresh estimate and the action is a legal step toward that estimate. A `coarse` estimate's tracking coordinate is its reported block lower-left corner, while a `fine` estimate's tracking coordinate is its exact cell. After an estimate becomes over-age, retaining it with clipped age 12 or clearing it to three nulls is allowed, but neither representation is a control input; an over-age arrival cannot displace a still-fresh control memory. The grader checks provenance and age against independent observation and packet replay.

The report budget is the sum of all served units, including units that fail and are retransmitted. Packets older than 12 slots expire before service; a packet or copy that is already over-age when offered at a creation boundary is recorded as admitted evidence but expires before that slot's service. Rejected packets never enter a FIFO, and an admitted packet remains at the head until its payload is completed or it expires. A completed packet whose arrival would be at or after boundary `H` does not control a vehicle and has no in-horizon arrival record. Queue rows are end-of-slot snapshots after admission, expiry and service, with packet IDs in FIFO order. The `service_slots` rows are keyed by `(channel,slot)` and their physical array order is not semantic; an acoustic row can name at most one link.

Capture holds when every UUV is within Chebyshev distance 2 of the target and every pair of UUVs is at least 2 apart; four consecutive slot-end holds complete capture. `capture_complete` is an absorbing event: it becomes true at the first boundary after the fourth consecutive hold and remains true thereafter. `capture_run` continues to report the current consecutive suffix and may reset after completion; the completion event is not revoked. Safety events include collisions, obstacle/danger entry and energy exhaustion.

#### Task metrics

All metrics are computed from independent replay and averaged equally over the five scenarios. `primary_metrics_1` is `0.55*capture_success + 0.25*capture_progress + 0.20*detection_timeliness`. `capture_success` is completion by the inclusive capture deadline. If incomplete, let `d_i` be the UUV-to-target Chebyshev distances at that deadline and `r` the length of the consecutive suffix of true `capture_condition` ending at the last evaluated slot before that deadline; then `capture_progress = 0.7*mean_i clip(1-d_i/23,0,1) + 0.3*min(r,4)/4`. `detection_timeliness = clip((search_deadline-first_detection_slot)/search_deadline,0,1)` and is zero when no detection occurs.

`primary_metrics_2` is `0.6*fresh_control_fraction + 0.4*shared_valid_estimate_fraction` over all UUV/slot samples from the first hunting boundary through termination. A fresh sample uses the latest fresh estimate available at that slot boundary (including an arrival or local hit labelled by that slot), has age at most 12, and has a quantized region containing the replayed target; a `fine` estimate contains only its exact cell, while a `coarse` estimate contains its 4 x 4 block. A later over-age arrival is evidence but does not displace a still-fresh control memory. The shared fraction is the proportion of hunting slots with at least two such UUVs. It is zero when the mission never enters hunting.

`primary_metrics_3` is `0.4*clip(1-safety_events/(4H),0,1) + 0.3*clip(1-motion_cost/(6H),0,1) + 0.3*clip(1-energy_used/energy_budget,0,1)`. Safety samples are S0 and the three UUVs (the mission evidence may also count A0 events, but those are excluded from this metric); motion cost is UAV displacement cells plus other vehicle displacement cells; energy budget is the sum of the five individual budgets.

`auxiliary_metrics_1` is `0.6*useful_delivered_units/max(1,sent_units) + 0.4*on_time_reports/max(1,generated_reports)`. A copy is timely when it arrives at its UUV before boundary `H`, its report age from the root A0 packet's `created_slot` is at most `12`, and its declared quantized region contains the replayed target at the arrival boundary (`fine` is exact and `coarse` is its 4 x 4 block); each such copy contributes one to `on_time_reports`. A timely copy is useful only when removing that copy and replaying the other eligible updates leaves a latest estimate different from the recorded next-boundary estimate, and the next action is a shortest legal step toward the copy's estimate. The counterfactual baseline uses the latest estimate that would remain without the focal copy, including the pre-arrival boundary estimate and other same-boundary eligible events. With no generated report both terms are zero.

`auxiliary_metrics_2` is `0.4*clip(1-mean_search_AoI/24,0,1) + 0.3*clip(1-mean_queue_wait/H,0,1) + 0.3*demand_link_availability`. AoI is sampled over all cells and search slots immediately after that slot's scan reset (the post-scan slot-end value); queue wait is sampled over every admitted packet (no packets gives one). A packet that receives service contributes `first_service_slot-admitted_slot`; one that expires before any service contributes its expiry boundary minus `admitted_slot` (zero when it is over-age at creation), and one still active at boundary `H` contributes `H-admitted_slot`. Availability is sampled only for links with an unexpired queued packet (no demand gives zero). The formal score is `(8*primary_metrics_1 + 4*primary_metrics_2 + 3*primary_metrics_3 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`, clipped to `[0,1]`.

#### Result contract

The project root must contain `run.py`, `report.md` and a freshly written `results.json`. The JSON root is exactly `{"scenarios":[...]}`. It contains exactly one record for each of the five canonical scenario names; physical array order is not semantic. Each record has exactly `scenario,vehicles,observations,links,service_slots,packets,queues,mission_state`.

`vehicles` has five objects `{vehicle_id,state_trace,intended_actions,executed_actions}`. State has 97 rows `{slot,x,y,energy,estimate_x,estimate_y,estimate_age}` and each action array has 96 `{slot,action}` rows; A0 actions must be from its aerial set and S0/UUV actions from the marine cardinal set exactly as listed above. The trace records the latest local estimate when it is retained; at an underlying age greater than `12`, either that estimate with clipped `estimate_age=12` or three null estimate fields is valid, but the stale value is never control-eligible. `observations` has 480 unique `(vehicle_id,slot)` rows `{vehicle_id,slot,kind,cells,hit,observed_x,observed_y}`, where kind is `scan,sonar,none` and coordinates are null unless a target position was locally obtained. `links` has 384 unique `(link_id,slot)` rows `{link_id,slot,sender,receiver,observed_available,observed_quality}`; these are only sender-side probes. `service_slots` has 192 unique `(channel,slot)` rows `{channel,slot,selected_link,packet_id,served_units}`, channel `wireless` or `acoustic`.

Packets have `{packet_id,parent_id,source,destination,granularity,estimate_x,estimate_y,created_slot,admitted_slot,first_service_slot,last_service_slot,arrival_slot,status,payload_units,retransmissions,used_slots}`; packet IDs are unique nonnegative integers or ASCII identifiers matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` (integers are nonnegative). An `A0->S0` report has `parent_id=null`; every `S0->Di` copy is its own packet row and carries the received report's ID in `parent_id`. Packet statuses are `queued,in_service,arrived,lost,expired,rejected`. `queues` has 384 rows `{link_id,slot,packet_ids,occupied_units,served_packet_id,rejected_packet_ids}`. `mission_state` has 97 rows `{slot,observed_target_x,observed_target_y,phase,first_detection_slot,capture_condition,capture_run,capture_complete,safety_events,total_energy}`; true target coordinates must never be written. All indices, coordinates and counters are integral and all numeric values finite. `safety_events` is the nonnegative integer count accumulated through that boundary, not an event-name list. `observed_target_x/y` are optional aggregate evidence: at a boundary they may contain the most recent locally obtained hit or arrived estimate, and otherwise must be null; if several updates share the same source slot, use the packet link/identifier order above and then vehicle order `A0,S0,D0,D1,D2` for local hits. They must never be filled from hidden truth.

The aggregate fields may persist the latest such event at later boundaries, including a physically completed but over-age arrival; they are `null` only before any local hit or arrival exists and do not themselves grant control eligibility.

A packet is created and offered to its link FIFO at `created_slot`; if accepted, `admitted_slot=created_slot`, otherwise it is `rejected` with null service and arrival fields. FIFO capacity is the sum of remaining payload units. At the start of a slot, packets older than 12 slots expire before service; an accepted packet whose root age already exceeds 12 at that boundary is immediately `expired` and is not placed in the active FIFO. A nonzero service row must name the available link and its current FIFO head. Successful served units accumulate toward the payload; failed units do not and increment `retransmissions`. `used_slots` lists every nonzero service slot once. Status is a post-event lifecycle: an admitted packet is `queued` until its first nonzero service attempt, becomes `in_service` after any such attempt while incomplete or awaiting its hop arrival, becomes `arrived` when a completed hop arrives before `H`, and becomes `lost` when completion would arrive at or after `H`; `expired` is assigned before service to an over-age FIFO packet and `rejected` is never admitted. When the payload completes, arrival is scheduled from that final service slot and is written only when it occurs before boundary `H`. `first_service_slot`, `last_service_slot`, statuses, arrivals, retransmissions and queue rows are post-event evidence derived from `service_slots`, not additional candidate decisions. The 180-unit budget is exactly the sum of `served_units` over both channel rows. The physical order of the dynamic `packets` array has no meaning; packets offered to the same link at one creation boundary are appended in ascending packet-ID JSON-text order, existing FIFO members remain ahead of them, and FIFO order is carried by each queue row's `packet_ids` list. `used_slots` is a unique set of attempted slots (chronological order is conventional but not required).

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

`report.md` must contain non-empty sections titled `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "vehicle, observation, link, service, packet, queue and mission evidence are typed and finite", weight: 4}
- {id: canonical_coverage, desc: "all fixed vehicle, observation, link, channel, queue and mission keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "actions, reports, service order, FIFO and budgets are legal", weight: 4}
- {id: replay_validity, desc: "target, observations, links, packets, motion, energy and capture replay consistently", weight: 6}
- {id: primary_metrics_1, desc: "deadline discovery and sustained capture quality from replay", weight: 8}
- {id: primary_metrics_2, desc: "fresh control and shared valid estimate quality", weight: 4}
- {id: primary_metrics_3, desc: "safety, motion and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful delivered report efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "search AoI, queue wait and demand-link availability", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the online search, relay and capture model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no runnable search/capture formulation; Implementation/evidence: no executable model or corresponding records."}
    - {score: 0.4, desc: "Report: AoI, relay or target response is materially incomplete; Implementation/evidence: the same missing mechanism is absent or inconsistent in the generated records."}
    - {score: 0.7, desc: "Report: local observation, age, queues, target response and capture are modeled across scenarios; Implementation/evidence: runnable records expose those mechanisms with only limited omissions."}
    - {score: 1.0, desc: "Report: precisely connects hidden target dynamics, two-hop age, formation and sustained capture; Implementation/evidence: replay-consistent records demonstrate each connection across the canonical scenarios."}
- id: method_design
  desc: "quality of the unified state-driven method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal reusable method is described; Implementation/evidence: no runnable state-driven procedure is present."}
    - {score: 0.4, desc: "Report: the design relies on fixed routes or schedules and ignores age, queues or target feedback; Implementation/evidence: generated actions show those omissions."}
    - {score: 0.7, desc: "Report: a deterministic method handles search, reporting, relay and tracking with limited long-horizon handling; Implementation/evidence: one reusable procedure runs legally across all scenarios."}
    - {score: 1.0, desc: "Report: a reproducible method jointly handles information value, AoI, granularity, links, target feedback, formation, safety and energy; Implementation/evidence: the same procedure and records substantiate every element."}
- id: experimental_coverage
  desc: "coverage of search, relay, age, dynamics and capture comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experimental design; Implementation/evidence: no reproducible run or usable result records."}
    - {score: 0.4, desc: "Report: only one condition is described; Implementation/evidence: only one condition is actually run or evidenced."}
    - {score: 0.7, desc: "Report: all scenarios and at least one communication or deadline comparison are specified; Implementation/evidence: the canonical runs and comparison records are present."}
    - {score: 1.0, desc: "Report: controlled single-factor comparisons test search timing, relay, granularity, queue/loss, target feedback and failure boundaries; Implementation/evidence: each comparison is reproducibly executed and supported by records."}
- id: research_analysis
  desc: "evidence-based interpretation of age, relay and capture trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records do not support the claims or are missing."}
    - {score: 0.4, desc: "Report: results are listed without mechanism or limits; Implementation/evidence: records provide numbers but no traceable explanatory evidence."}
    - {score: 0.7, desc: "Report: age, links, queues, target response, safety and energy trends are explained from records; Implementation/evidence: cited traces and metrics support those explanations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies interactions and failure boundaries without causal overclaiming; Implementation/evidence: comparisons, traces and metrics make each conclusion reproducible."}
