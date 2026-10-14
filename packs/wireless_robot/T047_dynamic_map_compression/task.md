---
id: T047_dynamic_map_compression
name: Communication-Aware Dynamic Map Compression for Robot Navigation
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

A sender robot observes a complete time-varying occupancy map while a navigator crosses narrow passages using only its local sensor and map updates received over a constrained wireless link. The sender cannot transmit the full 16 by 16 map whenever it changes. It must choose quadtree regions and spatial resolutions whose quantized values fit a finite FIFO, link capacity, and energy budget. Coarse or stale updates can hide moving obstacles, while detailed updates may wait in the queue until they no longer help navigation.

The receiver combines delivered regions with its own radius-one observations and uses that map to choose motion. Link outages, partial packet service, completion-time loss, local sensing, dynamic obstacle motion, and navigation therefore form one closed temporal process. A useful method must balance compression load and freshness against progress, collision risk, route length, and energy.

### Research Problem

Build a self-contained deterministic simulation and design a non-training online procedure that jointly selects at most one quadtree update packet and one navigator move in each slot. The sender should prioritize map innovation that can arrive while it is safety-relevant; the navigator should plan from its received map and current local observations without using hidden future occupancy.

Use one reusable state-driven procedure across all canonical scenarios. The environment tables below are public so the simulator and independent replay can be implemented, but scenario names, phases, future dynamic-path positions, future capacities, and future completion losses must not be passed into a role's decision snapshot or used to index pre-filled node sets, send slots, paths, or move fragments. Deterministic rules, search, planning, and optimization are allowed. Algorithmic approximations, parameters, project organization, and additional experiments are otherwise your choices.

### Experimental Setting and Constraints

#### Canonical scenarios and occupancy

The grid is `G={0,...,15} x {0,...,15}`, with `(0,0)` at the lower left. Each scenario has `H=100` half-second decision slots `t=0,...,99` and a terminal boundary `t=100`. The sender is `S` and the navigator is `N`. Every interval shown as `a:b` includes both endpoints, and each dynamic path cycles in its listed order.

| scenario | phase `p` | start | goal | deadline `d` | static obstacles `W` | dynamic path `P1` | dynamic path `P2` | base capacity `b` |
|---|---:|---|---|---:|---|---|---|---:|
| `static_narrow_passage` | 1 | `(1,1)` | `(14,14)` | 64 | `(7,y), y in 0:5 or 7:15`; `(11,y), y in 9:11` | `(6,6),(7,6),(8,6),(8,7),(8,8),(8,7)` | `(10,12),(11,12),(12,12),(13,12),(12,12),(11,12)` | 48 |
| `moving_crossing` | 4 | `(1,13)` | `(14,2)` | 72 | `(5,y), y in 3:11`; `(10,y), y in 4:12`; `(x,7), x in 6:9` | `(7,11),(8,11),(9,11),(9,10),(9,9),(8,9),(7,9),(6,9),(6,10),(6,11)` | `(11,3),(12,3),(13,3),(12,3)` | 48 |
| `bandwidth_burst` | 7 | `(1,8)` | `(14,8)` | 68 | `(5,y), y in 0:6 or 9:15`; `(10,y), y in 0:5 or 8:15` | `(4,7),(5,7),(6,7),(6,8),(6,9),(6,8)` | `(9,6),(10,6),(11,6),(11,7),(11,8),(11,7)` | 48 |
| `innovation_shift` | 10 | `(2,2)` | `(13,13)` | 78 | `(3,y), y in 5:13`; `(9,y), y in 1:9`; `(x,10), x in 10:14` | `(4,4),(5,4),(6,4),(6,5),(6,6),(6,7),(5,7),(4,7),(4,6),(4,5)` | `(10,11),(11,11),(12,11),(12,12),(12,13),(11,13),(10,13),(10,12)` | 48 |
| `long_range_update` | 13 | `(1,14)` | `(14,1)` | 86 | `(6,y), y in 0:4 or 6:15`; `(11,y), y in 2:13` | `(5,5),(6,5),(7,5),(7,6),(7,7),(7,8),(6,8),(5,8),(5,7),(5,6)` | `(10,1),(11,1),(12,1),(13,1),(13,2),(13,3),(12,3),(12,2)` | 36 |

At slot `t`, dynamic obstacle `i` occupies element `(t+p) mod |Pi|` of its path. For a cell `c`, the true occupancy is

```text
p_true(c,t) = 0.9  if c is static or is occupied by either dynamic obstacle at t,
              0.1  otherwise.
```

The receiver's initial estimate is `0.9` on static obstacles and `0.1` elsewhere, with source slot `-1` and source kind `initial`. Every initial, local, or remote probability stored in a map-cell `code` uses `floor(4095*p+0.5)`, so `0.1` and `0.9` are stored as `410` and `3686` respectively. Dynamic obstacles advance to their next positions at the end of each slot.

Both roles know the grid bounds, static obstacles, start, goal, deadline, horizon, and all fixed encoding, queue, energy, and motion rules. They are not given the scenario name, phase, dynamic paths, or future link outcomes in a decision snapshot.

For each scenario, `C_safe` is the union of every cell on either dynamic path, all in-grid cells at Manhattan distance at most one from those path cells, the start, and the goal. Its cell weight is `w(c)=2` on a dynamic-path cell and `w(c)=1` otherwise. This fixed nonempty set determines the required receiver-map records and the fidelity metric. The path and phase columns are environment truth for simulation and replay; a decision role is not given them or the scenario name.

#### Quadtree regions and packet values

The root covers the full grid. Recursively split each even-sided square into `0=NW`, `1=NE`, `2=SW`, and `3=SE`, where west/east are the lower/higher `x` halves and south/north are the lower/higher `y` halves. Thus root child `NW` covers `x=0,...,7`, `y=8,...,15`. The maximum depth is four.

A node is written `d:q`, where `d in {0,...,4}` and `q` has exactly `d` digits from `0,...,3`. The root is `0:`. Let

```text
start(d) = (4^d - 1)/3
node_id(d:q) = start(d) + value_4(q),
```

where the empty path has base-four value zero. The depth ranges are `0`, `1..4`, `5..20`, `21..84`, and `85..340`. This is a bijection between all 341 nodes and unsigned 9-bit IDs.

A packet contains `1..16` distinct nodes in strictly increasing `node_id` order. No two selected nodes may be equal or have an ancestor-descendant prefix relation. For a node `n` created in slot `t`, its only legal payload is

```text
code(n,t) = floor(4095 * mean_{c in R(n)} p_true(c,t) + 0.5)
value(n,t) = code(n,t) / 4095.
```

Each entry costs 9 node-ID bits and 12 code bits. A packet has a 16-bit header, so `bits=16+21*entry_count`. At most one packet is created in a slot. Its ID is exactly `pkt-{created_slot:03d}`, and `source_slot=queue_enter_slot=created_slot`. All entries use the sender's current full map at that source slot.

On arrival, packets are applied in lexicographic packet-ID order. A remote value covers every cell in its node region. It replaces a cell only if its source slot is newer than the cell's saved source slot, or if the source slots tie and its packet ID is lexicographically greater than the previously saved remote packet ID. Local sensing later in the same slot wins a source-slot tie. Source kinds are `initial`, `remote`, and `local`.

#### Link, FIFO, and sender energy

Current slot capacity is visible to the sender and equals

```text
B(t) = 0,
       if (11*t + 5*p) mod 23 = 0,
       or if scenario=bandwidth_burst and 36 <= t <= 48;
B(t) = b * (1 + ((7*t + 3*p) mod 4)) bits otherwise.
```

A packet that completes service in slot `t` is physically lost iff

```text
(13*t + 4*p) mod 31 = 0.
```

A completed non-lost packet arrives at the start of `t+1`. Loss is revealed only after completion.

The FIFO capacity is 1024 remaining bits. After the decision, a new packet is admitted before service, so it may start service in its creation slot. A creation is legal only if existing queued remaining bits plus its full size do not exceed 1024, cumulative admitted bits plus its size do not exceed 30720, and sender energy is sufficient. Legal result records contain admitted packets only. The sender begins with 80 energy units and admission immediately costs `0.002*bits`.

After admission, the current capacity is automatically allocated only to the FIFO head. Actual service is the smaller of `B(t)` and that packet's remaining bits. Zero capacity does not start service. `service_start_slot` is the first positive-service slot, and `service_complete_slot` is the slot where cumulative service reaches `bits`. The completed packet is immediately removed; unused capacity never transfers to another packet in that slot. A completed lost packet is not requeued. Retransmission requires creating a new packet in a later slot with its new fixed ID. Packets are ordered by queue-enter slot and then packet ID.

The terminal outcomes are `delivered` for a completed non-lost packet, `lost` for a completed lost packet, and `unfinished` for an admitted packet not completed by the horizon. A delivered packet has `arrival_slot=service_complete_slot+1`; a packet completing in slot 99 may arrive at terminal boundary 100. That arrival is recorded but does not update the receiver map or any task metric.

#### Navigator motion, sensing, and collision

The navigator starts with 120 energy units. A requested move is `N=(0,1)`, `E=(1,0)`, `S=(0,-1)`, `W=(-1,0)`, or `stay=(0,0)`. Every non-stay request made with at least one energy unit costs one unit, even when the target is blocked and no displacement occurs. A non-stay request with insufficient energy is illegal and is replayed as `stay`.

An out-of-grid target or a target occupied by a static or current-slot dynamic obstacle leaves the navigator in place and marks a collision for that slot. After the action, the dynamic obstacles advance to `t+1`; if either new obstacle position equals the navigator's action-end position, that slot is also a collision. Multiple causes in one slot count once. The navigator may move away at the next slot. Goal arrival is the first slot whose action-end position equals the goal. The `goal_reached` state then remains true, although subsequent legal moves are still allowed. Deadline arrival includes the action in slot `d`.

At the start of every decision slot, after remote arrivals, the navigator senses all in-grid cells at Manhattan distance at most one from its current position. Those true values overwrite its map with source slot `t` and source kind `local`. The sender sees the complete true map at `t`; the navigator sees only its current receiver map, current local sensor result, position, energy, and arrivals. It does not see the sender queue or energy. The sender sees its energy, the current capacity, all queued packets with remaining bits, and prior completion/loss feedback. Neither role sees future occupancy, capacity, loss, or information held only by the other role. Apart from the fixed task facts explicitly shared above, these lists are the complete role-specific information available for a decision.

#### Event order

`states[0]` is recorded before any slot event. Each decision slot uses this order:

1. Process deliveries arriving at boundary `t` in packet-ID order and apply remote map updates.
2. Observe the current full map at `S`; sense radius-one cells at `N` and apply the local updates.
3. Record decision-boundary `map_estimates[t]` and expose separate legal role snapshots.
4. Choose at most one admitted packet and one navigator move.
5. Admit the packet and debit sender energy, then perform one automatic FIFO-head partial service;   remove a completion and determine loss or next-boundary arrival.
6. Execute and charge the navigator action, resolve current-obstacle collision, advance dynamic   obstacles, check next-obstacle collision, and record `states[t+1]`.

`states[t].collision` reports whether slot `t-1` collided, so `states[0].collision=false` and `states[100].collision` reports slot 99. A terminal delivery at 100 is recorded only in its packet.

#### Task metrics

Let `z(t)` be the position in `states[t]`, `D0` the start-goal Manhattan distance, `g` the first action-end goal-arrival slot or null, and `moves` the number of recorded non-stay actions. Define

```text
progress = 1,                                                   if g <= d;
progress = clip(1 - ||z(d+1)-goal||_1 / D0, 0, 1),             otherwise.
safe = clip(1 - collision_slots/H, 0, 1).
```

- `navigation_safety = mean_s(0.75*progress + 0.25*safe)`.
- For `t=0,...,d`, let `p_est(c,t)=map_estimates[t].code/4095`. Define  `fidelity` as the weighted mean of `1-|p_est(c,t)-p_true(c,t)|` over  `C_safe x {0,...,d}`, using `w(c)`.
- For every true occupancy change `e=(c,t)` with `c in C_safe` and `1<=t<=d`, let  `v_e=w(c)*|p_true(c,t)-p_true(c,t-1)|`. Let `t_next` be that cell's next change slot, or  `d+1` if none. Set `h_e=1` iff a remote packet writes that cell during some  `u in [t,min(d,t+6,t_next-1)]`, has source slot at least `t`, and its written value differs from  `p_true(c,u)` by at most `1/4095`; otherwise set it to zero. A later local overwrite in the same  slot does not erase a remote hit. Define `timely=sum(v_e*h_e)/sum(v_e)`; each scenario has a  positive denominator. Then  `map_fidelity = mean_s(0.65*fidelity + 0.35*timely)`.
- Let `route_eff=clip(1-max(0,moves-D0)/max(1,H-D0),0,1)`,  `energy=0.002*sent_bits+moves`, and `energy_eff=clip(1-energy/200,0,1)`. Then  `route_energy_efficiency=mean_s(progress*(0.6*route_eff+0.4*energy_eff))`.

If multiple packets satisfy one event, assign it to the earliest arrival and then smallest packet ID. Let `R` be packets assigned at least one event, `age_r=arrival_slot-created_slot`, and `wait_r=service_start_slot-queue_enter_slot`.

- `compression_load_efficiency` is the five-scenario mean of  `timely*(0.5+0.5*clip(1-sent_bits/30720,0,1))`.
- `update_freshness_efficiency` is zero for a scenario with empty `R`; otherwise it is the  five-scenario mean of  `timely*(0.5*clip(1-mean_R(age_r)/6,0,1)+0.5*clip(1-mean_R(wait_r)/H,0,1))`.

Every metric is clipped to `[0,1]`. Formal task performance is

```text
(8*navigation_safety + 4*map_fidelity + 3*route_energy_efficiency
 + 3*compression_load_efficiency + 2*update_freshness_efficiency) / 20.
```

#### `results.json` contract

The root object has exactly `scenarios`, an array of five scenario records in the canonical table order. Each scenario has exactly `scenario`, `states`, `slots`, `packets`, `map_estimates`, and `mission`.

`states` contains exactly 101 entries in `slot=0,...,100` order. Each has exactly `slot`, integer `x,y` in `0..15`, finite `sender_energy,navigator_energy` in `[0,120]`, boolean `goal_reached`, and boolean `collision`. Entry `t` is the decision-preceding state; entry 100 is terminal.

`slots` contains exactly 100 entries in `slot=0,...,99` order. Each has exactly `slot`, `move`, and `packet_id`. The move uses the declared enum. Packet ID is non-null if and only if a packet is admitted in that slot, and then it uniquely references that same-slot packet record; otherwise it is null.

`packets` is in strictly increasing creation-slot order. Each entry has exactly `packet_id`, `created_slot`, `entries`, `bits`, `queue_enter_slot`, `service_start_slot`, `service_complete_slot`, `arrival_slot`, and `outcome`. IDs use `pkt-{created_slot:03d}`; creation and event slots are integers in `0..99` or null where inapplicable. Queue-enter slot equals creation slot. Entries number `1..16`; each has exactly string `node`, integer `node_id` in `0..340`, and integer `code` in `0..4095`. Nodes obey the bijection, strict order, prefix rule, current-map code, and bit formula above.

`service_start_slot` is null exactly when no positive service occurred. Delivered and lost packets have non-null completion; unfinished packets have null completion and arrival. Only delivered packets have a non-null arrival equal to completion plus one in `1..100`.

`map_estimates` contains exactly 100 entries in `slot=0,...,99` order. Each has exactly `slot` and `cells`. Cells strictly cover that scenario's `C_safe` in increasing `(y,x)` order. Every cell has exactly integer `x,y`, integer `code` in `0..4095`, integer `source_slot` in `-1..slot`, and `source_kind` in `initial,local,remote`. It represents the receiver map after that boundary's arrivals and local sensing, immediately before the decision.

`mission` has exactly `goal_arrival_slot`, `collision_slots`, `sent_bits`, `energy_used`, `remote_event_value`, and `mean_useful_age`. Arrival is null or an integer in `0..99`; collision slots is a nonnegative integer; sent bits is an integer in `0..30720`; energy used is finite in `[0,200]`; remote event value is finite and nonnegative; mean useful age is finite and nonnegative or null. `sent_bits` is cumulative admitted bits, `energy_used=0.002*sent_bits+moves`, `remote_event_value=sum(v_e)` over hit events, and mean useful age is `mean_R(age_r)` or null for empty `R`.

The fixed states, slots, safety-cell snapshots, packets, queue, maps, motion, collision, and metric events are independently replayed. Integers, IDs, enums, booleans, and array order use exact equality; finite real values use absolute tolerance `1e-6`.

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
- {id: result_evidence, desc: "state, action, packet, quadtree, receiver-map, and mission evidence has the declared form", weight: 4}
- {id: canonical_coverage, desc: "all scenarios, state slots, action slots, and safety-cell map records have exact canonical coverage", weight: 4}
- {id: decision_legality, desc: "moves, same-slot packet references, node prefixes, codes, bits, FIFO admission, budgets, and energy are legal", weight: 4}
- {id: replay_validity, desc: "fixed occupancy, sensing, queue service, loss, decoding, motion, collision, energy, and event order match the records", weight: 6}
- {id: primary_metrics_1, desc: "replayed deadline navigation progress and dynamic collision safety", weight: 8}
- {id: primary_metrics_2, desc: "replayed safety-region map fidelity and timely remote dynamic updates", weight: 4}
- {id: primary_metrics_3, desc: "replayed progress-gated route and total energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "timely dynamic-event coverage relative to admitted compression bits", weight: 3}
- {id: auxiliary_metrics_2, desc: "arrival age and FIFO wait of packets that cover dynamic events", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the dynamic occupancy, hierarchical update, receiver-map, and navigation formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given or the model is materially wrong; Implementation/evidence: no executable dynamic-map communication and navigation process corresponds to it."}
    - {score: 0.4, desc: "Report: only compression or navigation is described and a decisive tree, receiver-map, queue, information, or event-order relation is omitted; Implementation/evidence: the implemented state or timing model is partial or materially inconsistent."}
    - {score: 0.7, desc: "Report: local observation, quadtree encoding, link, receiver map, motion, and navigation consequences are accurately defined; Implementation/evidence: all canonical scenarios realize the core model with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: resolution, service and arrival timing, map-source precedence, and dynamic collision consequences are precise, including boundary cases; Implementation/evidence: traceable nodes, packets, map cells, queue events, and trajectories faithfully realize those relations."}
- id: method_design
  desc: "quality of the unified non-training compression, transmission, receiver-map, and navigation method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified or it uses prohibited training or hidden future truth; Implementation/evidence: no legal unified procedure supports the claimed decisions."}
    - {score: 0.4, desc: "Report: the method does not respond coherently to changing innovation, link, queue, map age, or navigation risk; Implementation/evidence: decisions collapse to pre-filled scenario, node, send-slot, or path tables or materially diverge from the description."}
    - {score: 0.7, desc: "Report: a legal, reusable method handles region selection, transmission, map use, and navigation under the main temporal constraints; Implementation/evidence: the same state-driven procedure runs across all scenarios with limited planning depth or resource coupling."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly treats resolution, change persistence, queue slack, current service, map-source age, and goal-directed safety; Implementation/evidence: contrasting visible link or map-risk states cause corresponding node, admission, and motion choices without unit-specific tables."}
- id: experimental_coverage
  desc: "coverage of canonical conditions and controlled tests of map updates and navigation"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: there is no checkable experimental execution."}
    - {score: 0.4, desc: "Report: experiments provide only isolated map, capacity, or navigation summaries and do not test the central communication-navigation relation; Implementation/evidence: condition records are incomplete or uninformative."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one explicitly identified controlled comparison beyond canonical aggregates test a compression or update-timing mechanism; Implementation/evidence: both conditions execute with comparable inputs and condition-level node, packet, map, trajectory, and metric evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant comparisons test a resolution or update-timing mechanism and a capacity-burst, dynamic-obstacle, or deadline boundary; Implementation/evidence: comparable condition-level records can reconstruct both the central mechanism and failure boundary."}
- id: research_analysis
  desc: "evidence-based analysis of resolution, freshness, queue, dynamic safety, and navigation trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the records; Implementation/evidence: the project cannot support its conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats compression or navigation values, makes unsupported causal claims, or omits material limits; Implementation/evidence: records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: conclusions use actual records to explain conditional bandwidth, obstacle, map-age, and navigation-cost trends, distinguish observation from inference, and state limitations; Implementation/evidence: the reported comparisons are supported."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies the interaction of spatial resolution, update age, queue service, and navigation risk, tests the explanation against contrasting evidence or a failure boundary, and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
