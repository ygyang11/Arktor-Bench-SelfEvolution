---
id: T019_semantic_scene_delivery
name: Task-Oriented Semantic Updates for Multi-Robot Search and Transport
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

Four ground robots must search initially unknown indoor grids, claim discovered targets, and transport them to a shared depot before target-specific deadlines. Each robot observes only nearby cells and targets. It can share compact structural, geometric-anchor, or object-state semantics over a common wireless channel, but successful enqueues consume a finite budget and queued packets compete for service. Delayed, expired, or deterministically lost packets leave the robots with different scene and task knowledge. Those differences can cause repeated exploration, stale target pursuit, and late deliveries even though collision conflicts are resolved by the environment.

### Research Problem

Build a deterministic, non-learning simulator and a unified local-state-driven method for all five canonical scenes. At every decision slot, choose one grid motion, an optional target claim, and an optional semantic message subject for each robot. The method should jointly reason about partial scene knowledge, message value and age, queue competition, remaining communication budget, target value and deadline, claim consistency, path consequences, and carrying capacity. It may use deterministic planning, search, optimization, or heuristics. Study when the three semantic types change discovery and delivery outcomes, and when communication cost, expiry, congestion, or stale status limits their value.

### Experimental Setting and Constraints

#### Canonical scenes and initial state

Every scene is the integer grid `x=0,...,11`, `y=0,...,9`. A cell is represented as `[x,y]`. Each listed closed integer rectangle `[xmin,xmax,ymin,ymax]` makes every included cell an obstacle for the entire scene. Robots are indexed `0,...,3`; their starts are listed in robot-ID order. Target IDs are exactly `g0,...,g7` in each scene. A target row has the form `target:([x,y],value,release_slot,deadline)`.

| scene | depot | starts | obstacle rectangles |
|---|---|---|---|
| `open_layout` | `[1,1]` | `[[0,0],[0,2],[2,0],[2,2]]` | `[[5,5,4,5]]` |
| `blocked_rooms` | `[1,1]` | `[[0,0],[0,2],[2,0],[2,2]]` | `[[4,4,0,3],[4,4,5,9],[8,8,0,5],[8,8,7,9]]` |
| `target_burst` | `[1,1]` | `[[0,1],[1,0],[1,2],[2,1]]` | `[[3,5,4,4],[7,8,4,4]]` |
| `status_conflict` | `[1,1]` | `[[0,0],[0,2],[2,0],[2,2]]` | `[[5,6,0,3],[5,6,5,9]]` |
| `long_corridor` | `[1,4]` | `[[0,3],[0,4],[0,5],[0,6]]` | `[[3,10,0,3],[3,10,6,9]]` |

| scene | target truth `(position,value,release_slot,deadline)` |
|---|---|
| `open_layout` | `g0:([3,1],6,0,15); g1:([2,6],5,0,22); g2:([6,2],8,3,28); g3:([7,7],7,5,36); g4:([10,1],9,9,40); g5:([9,5],6,12,45); g6:([4,8],4,15,50); g7:([10,8],8,18,58)` |
| `blocked_rooms` | `g0:([3,7],5,0,20); g1:([6,2],7,0,28); g2:([6,7],6,4,32); g3:([9,6],8,6,38); g4:([10,2],10,8,42); g5:([10,8],9,10,48); g6:([7,4],4,14,46); g7:([11,5],7,18,58)` |
| `target_burst` | `g0:([4,2],7,8,24); g1:([6,2],8,8,26); g2:([9,2],9,8,32); g3:([4,7],6,10,30); g4:([6,7],10,10,34); g5:([9,7],8,10,38); g6:([10,5],5,12,42); g7:([2,8],4,12,36)` |
| `status_conflict` | `g0:([4,4],6,0,20); g1:([7,4],10,0,28); g2:([8,2],8,2,34); g3:([8,7],8,2,36); g4:([3,7],5,6,30); g5:([10,4],9,8,44); g6:([4,8],4,10,40); g7:([10,8],7,12,52)` |
| `long_corridor` | `g0:([4,4],4,0,18); g1:([5,5],5,0,22); g2:([7,4],7,3,30); g3:([8,5],8,5,34); g4:([11,4],10,8,42); g5:([11,5],9,10,44); g6:([10,4],6,15,50); g7:([9,5],6,18,54)` |

The complete tables are public simulator truth, but not policy-visible truth. A robot policy may use the grid dimensions, depot, target-ID set, and all fixed rules. It must not inspect an obstacle fact or a target's position, value, release slot, or deadline until that fact is locally observed or received in a valid packet. At scene initialization, robots are at their starts; all four packet queues and all local knowledge stores are empty; zero enqueue bits have been used; and every target is unreleased, unclaimed, and not carried. A start position is known as the robot's own state but does not by itself create an occupancy fact.

Cell occupancy is true exactly for cells in the listed static obstacle rectangles. Robots, targets, and the depot do not change a cell's occupancy value. An uncarried target remains at its listed cell, a carried target is located at its carrier's current cell, and a delivered target is located at the depot. There is no randomness and no early termination. Decision slots are `t=0,...,59`; boundary 60 has no decision.

#### Local knowledge, decisions, and authoritative event order

At each slot, the following order is authoritative.

1. At the slot boundary, admit packets served in earlier slots and scheduled for this boundary. Drop   packets that are lost or expired, then fuse the other arrivals as specified below.
2. Release every target whose release slot is `t`. Each robot then observes all in-bounds cells at   Manhattan distance at most 1 from its pre-move position. Every observed cell yields its occupancy   value with observation timestamp `t`. Every released target currently in that range yields its   position, value, deadline, global status, claimant, and timestamp `t`. A target's discovery slot is   the first slot in which any robot makes such a local observation.
3. Freeze one private decision view per robot after steps 1 and 2, and call the same policy procedure   separately on those views. A view may contain the robot ID and current slot; its own exact position   and visited history; its acquired cell and target facts; its own current claim or carried target;   its own packet queue and remaining enqueue budget; the public grid, depot, target IDs, and rules; and   its own past views and decisions. It may also reveal whether an old packet has left that robot's   queue. It must not contain unobserved simulator truth, another robot's private knowledge or queue,   another robot's current decision, a future loss, a future arrival, or receiver-side loss feedback.   The motion, claim, message kind, and message subject are simultaneous decisions from this frozen   view. Robot ID may be used as a uniform priority or tie-break.
4. Resolve claim requests. For a globally unassigned target, the lowest-ID requester among robots that   are free at this step establishes the unique claim; a busy requester does not block a higher-ID free   requester. A request is legal if the target occurred in that robot's frozen target knowledge. A   legal request for a target that is already claimed, carried, or delivered, or from a busy robot,   simply has no effect.
5. Resolve and execute motions in robot-ID order as specified below. Then apply automatic pickup and   automatic depot delivery, defining the slot's endpoint and any delivery slot.
6. Privately report to each robot the actual result of its own claim request and the current fact only   for a target that it itself has claimed, is carrying, or has just delivered, after pickup, carried   motion, or delivery. A rejected claim does not reveal the rejected target's complete current fact.   These feedback facts and the robot's exact endpoint receive observation timestamp `t`; no other robot   receives them directly.
7. Generate a requested packet from the sender's step-6 knowledge, enqueue it, and run this slot's   channel service. Its subject must already have been legal in the frozen step-3 view. Action feedback   may update the fixed payload for that already selected subject, but cannot legalize a subject that   was unknown at the decision boundary.
8. Save the derived state and advance to the next slot.

Boundary 60 performs only step 1 for packets served in slot 59. It performs no release, observation, decision, motion, private feedback, enqueue, or channel service. Submitted fields are decisions at step
3. Positions, observations, knowledge, claims, pickups, deliveries, packet IDs and payloads, queues, service, losses, arrivals, fusion, visits, and all metrics are deterministic consequences derived by the evaluator.

Every cell fact stores its latest observation timestamp. Every target fact stores the fields above and a deterministic source order. A local pre-move observation at robot `i` has source order `(t,i,-1)`, and its step-6 private feedback has source order `(t,i,0)`, so same-slot feedback supersedes its pre-move observation. An arrived target fact has source order `(observation_timestamp,sender,message_id)`.

#### Motion, claims, and target transport

The legal motion strings are `{N,E,S,W,stay}` with displacements `(0,1)`, `(1,0)`, `(0,-1)`, `(-1,0)`, and `(0,0)`, respectively. A request whose endpoint is outside the grid or is an obstacle is dynamically illegal; it is replaced by `stay` for diagnostic continuation. The depot is a service cell that any number of robots may occupy simultaneously, so it creates no robot-cell reservation or swap conflict. For every other cell, the environment processes robots in increasing ID order. Processed robots reserve their executed endpoints, while every unprocessed robot continues to reserve its start cell. A request that conflicts with any reserved non-depot cell or swaps with a lower-ID robot outside the depot is replaced by `stay`. This deterministic inter-robot conflict resolution is not a validity defect.

Each robot can hold at most one unresolved claim or carried target. Target status evolves monotonically through `unreleased`, `unassigned`, `claimed`, `carried`, and `delivered`. After motion, a claimant on its target cell automatically picks it up. A carried target moves with that robot. Reaching the depot automatically delivers it and frees the robot. The delivered target retains the ID of its last successful claimant even though that robot is free. A delivery in slot `t` has delivery slot `t`. Deadline endpoints are inclusive: delivery is on time exactly when `delivery_slot <= deadline`.

#### Semantic packets, queues, loss, and fusion

Each robot requests at most one packet per slot using `message_kind` and `message_subject`. For no request, both are `null`; for a request, both are strings. Packet kinds and fixed payloads are:

| kind | bits | legal subject in the frozen decision view | fixed payload from step-6 knowledge |
|---|---:|---|---|
| `structure` | 32 | `c_x_y` for a known in-bounds cell `(x,y)` | that cell and every known four-neighbour occupancy value, each with its observation timestamp |
| `anchor` | 16 | `c_x_y` for a cell known to be free | that free-cell fact and its observation timestamp |
| `object` | 24 | a known target ID `g0,...,g7` | the latest known position, value, deadline, status, claimant, and observation timestamp |

Here `c_x_y` is literal decimal notation with `0 <= x <= 11` and `0 <= y <= 9`. The candidate does not submit packet IDs, payloads, or communication outcomes.

Every sender has a queue capacity of two whole packets and a per-scene budget of 960 successfully enqueued bits. First test the subject against the frozen decision view; suppress an illegal subject and record a validity defect. For a subject-legal request, process queue capacity before budget. A request to a full queue is dropped without budget use or a validity defect, and its budget condition is not examined. If the queue has space, suppress a request as dynamically illegal when its packet would make used bits exceed 960; otherwise enqueue it and consume its packet bits. Each admitted packet has generation slot `t` and unique message ID `q=4*t+sender`.

After all enqueues, sort all packets then present in the four sender queues by generation slot, kind priority `object` before `structure` before `anchor`, sender ID, and message ID. Starting with 96 service bits, scan that sorted list exactly once. Service and remove a packet if its whole size fits. Otherwise leave it queued and continue scanning, so a later smaller packet may fit. Partial service is forbidden. Each serviced packet is broadcast to the other three robots, never to its sender, and is scheduled for the next boundary. Senders receive no receiver loss feedback.

Let scene indices `k=0,...,4` follow the table order and receiver ID be `j`. A packet is lost for that receiver when

`(11*k + 5*t + 3*sender + 7*j + q) mod 19`

belongs to the scene's loss set below. In this expression, `t` is the generation slot, not the service slot.

| scene | loss set |
|---|---|
| `open_layout` | `{0}` |
| `blocked_rooms` | `{0,1,2}` |
| `target_burst` | `{0,4}` |
| `status_conflict` | `{0,1,7}` |
| `long_corridor` | `{0,3}` |

A non-lost `structure` packet is valid at arrival boundary `b` when `b <= generation_slot+5`; an `anchor` packet is valid when `b <= generation_slot+8`; and an `object` packet is valid when `b <= target_deadline`. Expired packets do not fuse. At one boundary, fuse valid arrivals in sender-ID and then message-ID order. For a static cell, retain the fact with greater observation timestamp; equal timestamps necessarily have equal occupancy, so retain the current fact. For a target, first retain the higher status. At equal status, retain the fact with lexicographically greater source order, with local and arrived source orders defined above. The latter two source-order components only make equal-time fusion deterministic; only a greater observation timestamp is a newer task fact.

#### Result contract, diagnostic continuation, and metrics

`results.json` must be strict JSON with exactly `{"scenarios":[...]}`. Object member names must be unique; `NaN`, infinities, and other non-JSON constants are invalid. `scenarios` uniquely covers the five scene names in any order. A scene object has exactly `scenario` and `slots`. `slots` uniquely covers integer `t=0,...,59` in any order, and a slot object has exactly `t` and `actions`. `actions` uniquely covers integer robot IDs `0,...,3` in any order. An action object has exactly:

- integer `robot`;
- string `move` in `{N,E,S,W,stay}`;
- string target ID `claim_target` in `g0,...,g7`, or `null`;
- string `message_kind` in `{structure,anchor,object}`, or `null`;
- `message_subject`, which is `null` exactly when `message_kind` is `null`, otherwise a string with the  kind-specific local format above.

Booleans are not integers. These 1,200 robot-slot decisions are all required replay evidence. Derived state and aggregate fields are neither required nor accepted at these record levels.

Strict parsing, the root keys, the `scenarios` container, and scene-object top-level shape belong only to result structure. Canonical scene, slot, and robot keys, including missing, duplicate, and unknown keys, belong only to coverage. Slot/action containers, action-object exact keys, and the four non-key decision fields' types, enums, null relation, and local formats belong only to evidence. Dynamic feasibility relative to the frozen private view, map boundary, obstacles, and enqueue budget belongs only to legality. One fact is not scored in multiple categories.

At each level, indexing retains only the first record with an expected explicit canonical key; a later duplicate cannot supply missing descendants. Every missing expected key, extra duplicate record, and unknown key record counts as one coverage defect. Child coverage is examined only for an indexed parent whose child container is a list, so a parent defect does not expand into all descendant defects. A missing or unrelatable robot-slot decision becomes `stay,null,null,null` for diagnostic replay and fails coverage only. A retained malformed container or non-key decision becomes the same default and fails evidence only. An out-of-enum move becomes `stay`; a dynamically illegal move becomes `stay`; and an illegal claim or message request is suppressed. In each scene, dynamic legality is not inferred in or after the first slot that is missing, unrelatable, or has a malformed container or non-key decision. Fixed-model replay and diagnostics still continue with the stated defaults. These rules prevent one replacement from creating secondary legality failures; every task metric is zero unless all validity checks are full.

For `primary_metrics_1`, sum the values of targets delivered on time in each scene, divide by that scene's total target value, and average the five fractions.

For target `g`, temporal utility is zero when it is undiscovered, discovered at or after its deadline, undelivered, or delivered after its deadline. Otherwise it is

`clip(1-(delivery_slot-discovery_slot)/(deadline-discovery_slot),0,1)`.

Within a scene, weight each target's temporal utility by its value divided by total scene value. `primary_metrics_2` is the mean of the five scene values.

For `auxiliary_metrics_1`, process served packets in their arrival and fusion order. A packet contributes its complete size once to `useful_bits` if at least one receiver uses it to obtain an unknown cell for the first time, or a target fact with higher status, or an equal-status target fact with greater observation timestamp. A source-order-only replacement is not useful. Lost, expired, or non-improving packets are not useful, while every serviced packet contributes its size to `total_sent_bits`. The scene score is zero if `total_sent_bits=0`; otherwise it is `clip(useful_bits/total_sent_bits,0,1)`. The metric is the five-scene mean. A useful packet is not counted again for additional receivers.

For `auxiliary_metrics_2`, each robot has `shared_timestamp`, the greatest observation timestamp among all facts in every valid packet that has reached it by the current boundary, whether or not that fact wins fusion. At boundaries `b=1,...,60`, set `age=4` if no such fact has arrived, and otherwise set `age=b-shared_timestamp`. Freshness is the mean of `clip(1-age/4,0,1)` over all robots and these boundaries. Starts are positions at boundary 0; a slot-`t` endpoint is at boundary `t+1`. A robot makes one redundant visit in slot `t` when its endpoint has appeared as any robot's position at a boundary in `0,...,t`; depot endpoints are excluded. Thus staying in a non-depot cell is redundant. Let the scene's count be `redundant_visits`. Its scene score is

`0.6*freshness + 0.4*clip(1-redundant_visits/(4*60),0,1)`.

The metric is the five-scene mean. Here `clip(x,0,1)=min(1,max(0,x))`. All metrics are in `[0,1]` and higher is better. Normalized task performance is

`(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`.

All inputs are derived by deterministic replay. Metrics are scored only when the official run, strict result structure, evidence, canonical coverage, decision legality, and replay validity are all full.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite the root-level `results.json` from project code and the fixed conditions above. The evaluator deletes an old result before the one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations. Controlled experiments used as Judge evidence may produce optional text or JSON artifacts outside `results.json`; for every claimed comparison, identify the conditions, shared inputs, changed factor, artifact path, and condition-level outcomes.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "slot, motion, claim, and semantic-request fields satisfy local contracts", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "motions, private-view claims, semantic subjects, and enqueue budgets are legal", weight: 4}
- {id: replay_validity, desc: "observations, transport, queues, delivery, and semantic fusion replay consistently", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed on-time delivered target value", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed discovery-to-delivery temporal utility", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed useful semantic bits per serviced bit", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed shared freshness and nonredundant exploration", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of local semantics, wireless timing, target transport, and information boundaries"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent local-observation, semantic-communication, or transport formulation, or a materially wrong model. Implementation/evidence: no executable states or decisions correspond to the claimed process."}
    - {score: 0.4, desc: "Report: defines only part of the semantic types, queues, target states, timing, or task relation and omits a decisive coupling. Implementation/evidence: code realizes only a partial model or materially departs from the report."}
    - {score: 0.7, desc: "Report: accurately defines local observations, the three semantic types, capacity and delay, fusion, claims, carrying, and delivery. Implementation/evidence: code and records implement the core model across every scene with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains the causal relations among semantic value, fact age, queue competition, target-state consistency, path choice, and delivery outcome, including event and boundary cases. Implementation/evidence: code faithfully realizes these relations with traceable observation, packet, fusion, and transport evidence."}
- id: method_design
  desc: "design of unified semantic selection, communication scheduling, exploration, and transport"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, a prohibited learning method, or a method outside the private information boundary. Implementation/evidence: no legal unified procedure is present."}
    - {score: 0.4, desc: "Report: relies mainly on periodic or indiscriminate sending, nearest-target pursuit, or scene-specific schedules while ignoring decisive message or task consequences. Implementation/evidence: decisions are pre-filled or materially inconsistent with the described method."}
    - {score: 0.7, desc: "Report: gives a legal, reproducible, unified method for exploration, semantic selection, channel use, claiming, and transport. Implementation/evidence: the same state-driven procedure supports its main choices across scenes, with limited anticipation or robustness."}
    - {score: 1.0, desc: "Report: justifies a unified and reproducible method that jointly treats semantic utility and age, queue and budget pressure, stale-status conflict, deadlines, carrying capacity, and path consequences. Implementation/evidence: evolving permitted state drives the decisive choices without scene or slot action tables, and records agree with the method."}
- id: experimental_coverage
  desc: "coverage of semantic mechanisms, communication pressure, task outcomes, and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose. Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: presents only isolated scene aggregates or delivery totals and does not test the semantic-task relation. Implementation/evidence: saved conditions do not support the principal experimental claims."}
    - {score: 0.7, desc: "Report: covers every canonical scene and at least one controlled comparison of semantic type, triggering, capacity, delay, or deadline effects. Implementation/evidence: both comparable conditions are actually executed with condition-level packet, target, and delivery records."}
    - {score: 1.0, desc: "Report: uses complementary non-redundant controlled conditions to test a semantic mechanism and a bandwidth, staleness, conflict, or deadline sensitivity or failure boundary. Implementation/evidence: executed condition-level records and metrics reconstruct messages, ages, fusion, claims, paths, and deliveries beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based analysis of semantic utility, freshness, coordination, and delivery"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is missing or materially contradicts the recorded findings. Implementation/evidence: code and records cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mainly repeats bits or delivery values, makes unsupported causal claims, or omits material limitations. Implementation/evidence: records do not support the stated interpretation."}
    - {score: 0.7, desc: "Report: uses actual records to explain conditional trends in semantic selection, overhead, freshness, repeated exploration, and delivery timing, distinguishes observation from inference, and states material limits. Implementation/evidence: saved comparisons support those claims."}
    - {score: 1.0, desc: "Report: uses controlled contrasts and a failure boundary to explain which semantics alter claims and deliveries under which channel and deadline conditions, quantifies the principal sensitivities, and limits the scope of inference. Implementation/evidence: reproducible condition-level evidence supports the mechanism and conclusions."}
