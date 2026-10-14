---
id: T046_ergodic_periodic_connectivity
name: Ergodic Exploration with Flexible Periodic Connectivity
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

Three ground robots survey spatially distributed information sources. Each robot initially owns a different private part of the information map, and later source updates are learned only by robots that physically sample the affected cells. The robots should spend their motion where information value is concentrated, but their local maps become useful to the team only after summaries are delivered to another robot.

Continuous connectivity is unnecessary and would keep the robots from exploring. Instead, every robot pair should exchange at least one summary during the second half of each recurring period. Different pairs may exchange at different places and times. Range-limited contacts, a two-request shared channel, deterministic losses, finite FIFO queues, version deadlines, collision resolution, and motion energy couple exploration to these flexible rendezvous decisions.

### Research Problem

Build a self-contained deterministic simulation and design a non-training online procedure that jointly chooses robot motion, summary creation, destination, and FIFO-head transmission. The method should match the team's occupancy to the information distribution, physically sample valuable cells, create useful cross-robot fusion, satisfy pair-period exchanges, and control collision, motion, retransmission, and waiting costs.

Use one reusable state-driven procedure across every canonical scenario and role. A role may use only its local view and public static inputs defined below. Current time may be used to calculate period phase, window slack, and provenance age, but scenario names, phases, period numbers, or absolute slots must not index pre-filled trajectories, rendezvous, packet contents, or send tables. Deterministic heuristics, estimation, search, planning, and optimization are allowed; the method and project organization are otherwise your decisions.

### Experimental Setting and Constraints

#### Canonical scenarios, grid, and periods

The integer grid is `x=0,...,17`, `y=0,...,17`. There are `H=120` half-second decision slots `t=0,...,119` and a non-decision terminal boundary at `t=120`. Robots are `A,B,C`, in that rank order, and each starts with `120` energy units. The five scenario records must occur in the following table order. Every listed obstacle cell is closed.

| scenario | phase `p` | starts `(A,B,C)` | closed obstacle cells | information-cell transform |
|---|---:|---|---|---|
| `clustered_sources` | 1 | `(2,2),(2,15),(15,2)` | `x=8, y in {3,...,14} except y=8,9` | `(x,y)` |
| `split_sources` | 4 | `(2,2),(15,15),(2,15)` | `x=8, y in {0,...,5,7,...,10,12,...,17}` | `(x+1,y)` |
| `corridor_rendezvous` | 7 | `(1,9),(16,3),(16,15)` | `x in {6,11}, y in {0,...,7,10,...,17}` | `(x,y+1)` |
| `shifting_sources` | 10 | `(2,9),(15,3),(15,15)` | `y=8, x in {0,...,5,7,...,10,12,...,17}` | `(x-1,y)` |
| `uneven_private_maps` | 13 | `(2,3),(15,9),(2,15)` | `x=7, y in {2,...,6,11,...,15}` and `x=11, y in {4,...,8,12,...,16}` | `(x,y-1)` |

There are five periods of length `P=24`. Period `k in {0,...,4}` contains slots `24k,...,24k+23`, and its exchange window is exactly `24k+12,...,24k+23`. A pair-period is a scored objective, not a legality gate. Pair labels are ordered `AB,AC,BC`.

The simulator uses the phase `p` for fixed communication outcomes. A role does not receive the scenario name or phase. It does receive the grid, obstacles, all initial positions, the transformed information-cell coordinates and values, period constants, and its own initial private knowledge as public or local inputs.

#### Information cells, versions, and observation

The untransformed information cells `u1,...,u12`, in canonical order, are:

```text
u1=(3,3,1.0)    u2=(5,5,1.1)    u3=(7,9,1.2)    u4=(9,4,1.0)
u5=(11,12,1.3)  u6=(14,14,1.1)  u7=(4,13,1.2)   u8=(13,5,1.0)
u9=(2,9,0.9)    u10=(16,9,1.1)  u11=(9,15,1.2)  u12=(9,2,0.9)
```

Each tuple is `(x,y,value)`. Apply the scenario transform. If the result is closed or outside the grid, replace it with the first open in-grid neighbor in `N,E,S,W` order. Values do not change. Let `V` be the sum of all twelve values and define `phi(c)=value(uj)/V` at each transformed information cell and `phi(c)=0` at every other cell of the full 18 by 18 grid. Thus `sum_c phi(c)=1`.

Robot `A` initially knows version `0` of `u1,...,u4`, `B` knows version `0` of `u5,...,u8`, and `C` knows version `0` of `u9,...,u12`; every other local version is unknown. Initial private versions are available for packet creation at slot `0`, with provenance slot `0` and sharing deadline `23`. Initial ownership does not count as a physical observation or physical coverage.

Environment versions change at the start of a slot. In `shifting_sources`, `u2,u6,u10` change from version `0` to `1` at slot `36`, and `u4,u8,u12` change at slot `72`. In each other scenario, `u3,u7,u11` change from version `0` to `1` at slot `48`. These future updates are environment truth and are not included in role views.

After movement in every decision slot, a robot automatically observes every information cell at Manhattan distance at most `1` from its new position. The observation uses the environment version effective in that slot and has ID `obs-{slot:03d}-{robot}-{unit_id}`. Every such robot-cell combination is recorded, including repeated observations. An observation at slot `o` becomes local knowledge only at the end of `o`, so it first may enter a packet created in slot `o+1`. Its sharing deadline is the end of its observation period, `24*floor(o/24)+23`. Observing the same version again refreshes that robot's provenance slot and deadline. A useful received item inherits the provenance stored in its packet.

#### Local information and event order

At a decision boundary, a role sees only its own position, energy, trajectory, current local version and provenance for each unit, FIFO contents, summaries delivered to itself at that boundary, and the identities of its current contact neighbors. It never sees another robot's current position, energy, trajectory, private knowledge, or queue. It also cannot see undelivered summaries, future source updates, the phase, or future communication outcomes.

Each decision slot uses this exact order:

1. Process successful services from slot `t-1`; for `t<=119`, apply strict version improvements to   the receiver's local fusion state.
2. Repeatedly remove expired FIFO heads, then apply any environment version update beginning at `t`.
3. Form the contact graph from slot-start positions and expose a separate local view to each role.
4. Each role jointly chooses at most one new summary packet, one move, and either a FIFO-head send   request or no send. New packets are admitted in robot rank order and may be sent in the same slot.
5. Sanitize illegal components, register eligible attempts, select at most two channel services, and   evaluate their fixed outcomes.
6. Execute movement simultaneously and resolve collisions and energy.
7. Automatically sample information cells from movement-end positions and update local provenance.
8. Record slot-end queues, local versions, communication counts, contacts, and motion state.

A successful service in slot `119` arrives at terminal boundary `120`. That boundary records the packet and delivery only; it does not create a decision, fusion update, observation, pair-period credit, or task-metric credit.

#### Movement, collision, and contact

A requested move is one of `N=(0,1)`, `E=(1,0)`, `S=(0,-1)`, `W=(-1,0)`, or `stay=(0,0)`. A successful non-stay move costs one energy unit. An out-of-grid, closed-cell, or energy-insufficient move is illegal and is treated as `stay` for deterministic replay.

Other moves use a fixed-point simultaneous rule. Begin with each sanitized tentative destination. Repeatedly stop all robots sharing a tentative destination and every mover entering the slot-start cell of a robot that stayed or was stopped in an earlier iteration; include the occupying robot in the stopped set. Newly stopped robots revert to their slot-start cells before the next iteration. Continue until no robot is newly stopped. Pairwise swaps and longer rotations with distinct tentative destinations are allowed. `collision_robots` contains the stopped robots in `A,B,C` order. Each stopped robot contributes one `collision_event`; only successful non-stay moves count toward motion cost or energy use.

Two robots are in contact when their slot-start Manhattan distance is at most `3`. Contacts are undirected. A role sees the identities of its incident contact neighbors, but not their positions.

#### Packet creation, FIFO service, and delivery

A summary packet has one source, one fixed destination different from its source, and `1..3` distinct `[unit_id,version]` items in canonical unit order. Every item must equal the source's current locally known version when the packet is created. Its replayed provenance is the source's most recent provenance for that version at creation; the packet content and provenance then remain fixed.

Each source may create at most one packet in a slot. Its ID is exactly `sum-{created_slot:03d}-{source}`. Packet creation is represented by `packets.created_slot`, `source`, `destination`, and `unit_versions` in the result record. Creation occurs after expiry cleanup and local-view exposure. A packet whose source FIFO already contains 12 packets is recorded with outcome `dropped` and never admitted. If every content item's deadline is less than the creation slot, it is recorded as `expired` and never admitted. Otherwise it enters the source FIFO, ordered by `(created_slot,packet_id)`, and can be requested immediately if it is the head.

A send request must cite the local FIFO head, its fixed destination, and a destination currently in contact. A request satisfying those conditions, the source's per-period attempt cap, and the scene unit budget is eligible. Each robot has at most `8` eligible attempts per period. Every eligible request appends its request slot and next zero-based attempt number to the packet. Its number of content items is added to `sent_units`; if its attempt number is at least one, those items are also added to `retransmitted_units`. The scene permits at most `120` sent units. Requests are processed in robot rank order for these limits; an over-limit request is illegal and produces no attempt.

The shared channel serves at most two eligible requests per slot, selecting the smallest `(created_slot,rank(source),packet_id)` values. All eligible requests, including capacity-unselected ones, count as attempts and remain recorded. A capacity-unselected packet stays at its FIFO head. For a served packet from `source` to `destination` in slot `t`, physical transmission succeeds iff

```text
(7*t + 3*p + 2*rank(source) + rank(destination)) mod 19 > 1.
```

A physical loss leaves the packet at the head. A success immediately removes it and schedules a delivery at `t+1`. A packet can be delivered only once. At the start of each decision slot, FIFO heads for which every content provenance deadline is less than the current slot are repeatedly removed as `expired`. Physical loss is not a packet terminal outcome; packets remaining queued at the horizon are `unfinished`. The other outcomes are `delivered`, `expired`, and `dropped`.

For arrival slots `1..119`, every success creates one delivery. `useful=true` iff at least one item strictly increases the receiver's local version. `within_deadline=true` iff at least one such useful item arrives no later than its fixed provenance deadline. Useful items update the receiver and keep their packet provenance. Duplicate or older items do not update it. A terminal-slot delivery has both booleans false and does not update fusion state.

For each pair-period, `window_contact_slots` contains every decision slot in that period's window where the pair is in contact. `satisfied=true` iff a successful non-terminal delivery between the pair has an arrival slot in that window. `successful_delivery_id` is the first such delivery in arrival and delivery-ID order. Contact alone does not satisfy a pair-period.

#### Task metrics

Let `n(c)` count all robot-slot samples whose movement-end position is cell `c`; stays and repeated visits count. Define `rho(c)=n(c)/(3H)` over all 324 grid cells. Let `covered_value` be the sum of values of distinct units physically observed at least once in slots `0..119`.

- `goal_coverage = 0.6*clip(1-0.5*sum_c |rho(c)-phi(c)|,0,1)
  + 0.4*covered_value/V`.
- `periodic_fusion = 0.6*satisfied_pair_periods/15 + 0.4*timely_fused_value/V`.  For each unit, inspect versions from high to low. Its value enters `timely_fused_value` once if a  non-terminal delivery to a robot other than that unit's initial owner strictly improves the  receiver with that version and arrives by the item's provenance deadline. If the highest version  has no qualifying delivery, fall back to lower versions.
- `safe_motion_efficiency = 0.4*clip(1-collision_events/(3H),0,1)
  + 0.3*clip(1-motion_cost/(3H),0,1) + 0.3*clip(1-energy_used/360,0,1)`.
- `summary_freshness_efficiency` is zero unless `sent_units>0` and at least one item is useful.  Otherwise it is `0.6*clip(1-mean_fusion_age/24,0,1)
  + 0.4*useful_summary_units/sent_units`. For every item that strictly improves a receiver in a  non-terminal delivery, fusion age is `arrival_slot-provenance_slot`; take their mean.
- `contact_load_efficiency` is zero unless `sent_units>0` and `useful_summary_units>0`.  Otherwise it is `0.5*clip(1-contact_wait_slots/(24*3),0,1)
  + 0.25*clip(1-retransmitted_units/sent_units,0,1)
  + 0.25*satisfied_pair_periods/15`. For each non-terminal delivery containing at least one useful  item, add `successful_service_slot-created_slot` once to `contact_wait_slots`.

All five values are averaged equally over the five scenarios and clipped to `[0,1]`. Formal task performance is

```text
(8*goal_coverage + 4*periodic_fusion + 3*safe_motion_efficiency
 + 3*summary_freshness_efficiency + 2*contact_load_efficiency) / 20.
```

#### `results.json` contract

The root object has exactly `scenarios`, an array of five records in canonical table order. Each scenario has exactly `scenario`, `slots`, `robots`, `units`, `pair_periods`, `observations`, `packets`, `deliveries`, and `mission`.

`slots` contains exactly 120 entries in `slot=0,...,119` order. Each has exactly `slot`, `requests`, `served_packet_ids`, and `state`. `requests` has three entries in `A,B,C` order, each with exactly `robot`, `move`, `packet_id`, and `destination`; the last two are both null for no send and non-null strings for a send. `served_packet_ids` contains at most two strings in arbitration order. `state` has exactly `collision_robots`, cumulative nonnegative integer `send_attempts`, `contact_pairs`, and integer `period`. Robot and pair arrays use their canonical orders.

`robots` contains exactly `A,B,C` in order. Each record has exactly `robot`, `positions`, `energy`, `queue_lengths`, and `known_unit_versions`. Positions and finite energy arrays have length 121; each position is a two-integer `[x,y]` open-grid cell and each energy value is in `[0,120]`. Index `t` is the slot-start value and `t+1` is the slot-end value. Queue lengths have length 120 and integer range `0..12`. `known_unit_versions` has 120 entries, each a 12-integer array in `u1,...,u12` order, using `-1` for unknown and `0..1` for known versions; index `t` is the slot-end state.

`units` contains exactly `u1,...,u12` in order. Each has exactly `unit_id`, `initial_owner`, terminal environment `current_version`, `last_observed_slot`, `value`, and `deadline_slot`. The last observed slot is the maximum physical observation slot for that unit, or null; its deadline is that period's end, or null when no observation occurred. Values are finite and positive.

`pair_periods` has 15 entries in period-major order and `AB,AC,BC` order within each period. Each has exactly integer `period`, string `pair`, sorted integer `window_contact_slots`, nullable `successful_delivery_id`, and boolean `satisfied`.

`observations` is sorted by `(slot,rank(robot),rank(unit_id))`, where unit rank is its position in `u1,...,u12`. Each entry has exactly `observation_id`, `robot`, `unit_id`, integer `version`, integer `slot`, and finite positive `value`. IDs use the fixed format defined above and are unique.

`packets` is sorted by `(created_slot,source,packet_id)`. Each entry has exactly `packet_id`, `source`, `destination`, `created_slot`, `unit_versions`, `attempt_slots`, `attempt_nos`, `arrival_slot`, and `outcome`. Unit-version items are two-element arrays, ordered and unique as defined above. Attempt arrays have equal length; `attempt_nos` is exactly `0,...,length-1`, and attempt slots are nondecreasing integers in `0..119`. Arrival is null or an integer in `1..120`. Outcome is one of `delivered,expired,dropped,unfinished`; only a delivered packet has a non-null arrival.

`deliveries` is sorted by `(slot,delivery_id)`. Each entry has exactly `delivery_id`, `packet_id`, `source`, `destination`, `slot`, `unit_versions`, boolean `useful`, and boolean `within_deadline`. Its ID is `del-{packet_id}`, and its endpoints and content equal the packet's.

`mission` has exactly `collision_events`, `motion_cost`, `energy_used`, `sent_units`, `useful_summary_units`, `retransmitted_units`, `contact_wait_slots`, and `timely_fused_value`. Counts are nonnegative integers except finite nonnegative `energy_used`, `contact_wait_slots`, and `timely_fused_value`. Every summary is independently replayed; integer, ID, enum, and array-order facts use exact equality, and finite real values use absolute tolerance `1e-6`.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite root-level `results.json` from the project code and fixed conditions above. The evaluator deletes an old result before one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations.

Controlled experiments beyond the canonical run may be documented in `report.md` and in optional project-generated text or JSON artifacts. They must not add fields to or replace the exact canonical `results.json` contract above.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static Python compilation", weight: 2}
- {id: official_run, desc: "one official run succeeds within the limit and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the declared root and scenario container structure", weight: 3}
- {id: result_evidence, desc: "slot, robot, unit, pair-period, observation, packet, delivery, and mission evidence has the declared form", weight: 4}
- {id: canonical_coverage, desc: "all fixed scenarios, slots, robots, units, and pair-periods occur exactly once in canonical order", weight: 4}
- {id: decision_legality, desc: "motion, packet creation, local-version content, FIFO contact requests, and attempt budgets are legal", weight: 4}
- {id: replay_validity, desc: "fixed observation, contact, arbitration, loss, delivery, fusion, movement, and terminal replay matches the records", weight: 6}
- {id: primary_metrics_1, desc: "replayed occupancy distribution and physical information coverage", weight: 8}
- {id: primary_metrics_2, desc: "replayed pair-period delivery satisfaction and timely cross-robot fusion", weight: 4}
- {id: primary_metrics_3, desc: "replayed collision, successful motion, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "replayed useful-summary efficiency and provenance freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "replayed contact waiting, retransmission load, and pair-period efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the periodic exploration, private-version, contact, queue, and fusion formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given or the model is materially wrong; Implementation/evidence: no executable information-sampling and periodic communication process corresponds to it."}
    - {score: 0.4, desc: "Report: only exploration or contact is described and a decisive private-version, provenance, FIFO, window, or fusion relation is omitted; Implementation/evidence: state transitions or event order are partial or materially inconsistent."}
    - {score: 0.7, desc: "Report: occupancy, local observation, private versions, contact windows, packet service, motion, and fusion are accurately defined; Implementation/evidence: all canonical scenarios implement the core model with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: event order and the causal interaction of flexible rendezvous, provenance deadlines, coverage, and fusion are precise, including boundary cases; Implementation/evidence: traceable trajectories, observations, attempts, pair-periods, and deliveries faithfully realize those relations."}
- id: method_design
  desc: "quality of the unified non-training exploration, rendezvous, packet, and transmission method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified or it uses prohibited training or private truth; Implementation/evidence: no legal unified procedure supports the claimed decisions."}
    - {score: 0.4, desc: "Report: the method does not respond coherently to evolving local state, window slack, queues, version deadlines, or motion consequences; Implementation/evidence: decisions are pre-filled, remain insensitive to decisive state changes, or materially diverge from the description."}
    - {score: 0.7, desc: "Report: a legal, reusable method handles information sampling, rendezvous, summary content, and FIFO transmission under the main periodic constraints; Implementation/evidence: the same state-driven procedure runs across all roles and scenarios, with limited planning depth."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly uses information marginal value, contact reachability, window slack, provenance deadlines, capacity, and movement cost; Implementation/evidence: contrasting local states cause corresponding trajectory and packet choices without scenario, period, or slot action tables."}
- id: experimental_coverage
  desc: "coverage of canonical conditions and controlled tests of periodic communication and exploration"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: there is no checkable experimental execution."}
    - {score: 0.4, desc: "Report: experiments cover only isolated information or contact summaries and do not test the central trade-off; Implementation/evidence: condition records are incomplete or uninformative."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one explicitly identified controlled comparison beyond canonical aggregates test a central exploration or sharing mechanism; Implementation/evidence: both conditions execute with comparable inputs and condition-level trajectory, packet, and fusion evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant comparisons include a rendezvous or sharing mechanism test and a window, capacity, or version-deadline sensitivity or failure boundary; Implementation/evidence: comparable condition-level records and metrics distinguish both claims."}
- id: research_analysis
  desc: "evidence-based analysis of coverage, rendezvous, version fusion, and resource trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the records; Implementation/evidence: the project cannot support its conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats coverage or contact values, makes unsupported causal claims, or omits material limits; Implementation/evidence: records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: conclusions use actual records to explain conditional coverage, window, queue, motion, and fusion trends, distinguish observation from inference, and state limitations; Implementation/evidence: the reported comparisons are supported."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies the interaction of occupancy, rendezvous timing, provenance deadlines, and capacity, tests the explanation against a contrasting condition or failure boundary, and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
