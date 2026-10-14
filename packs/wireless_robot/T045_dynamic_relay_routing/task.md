---
id: T045_dynamic_relay_routing
name: Online Relay Routing for Dynamic Indoor Robot Missions
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

An indoor service robot receives a sequence of mission goals from a fixed ground station and must return progress before short confirmation windows close. Walls and distance make the direct radio link unreliable, so two mobile robots can reposition as relays and forward messages in either direction. A command is unusable until it reaches the client, and completing a physical goal is only partly useful if the corresponding progress message does not return on time.

Relay motion and communication are coupled. Moving a relay can improve a future hop but consumes energy and may create a collision; probing a link consumes the node's communication request for a slot; unsuccessful or competing transmissions leave packets in finite FIFO queues; and a packet can become too old or exhaust its hop limit. The robots must make online decisions from local packet and probe history while interference and deterministic packet losses remain future environment events.

### Research Problem

Build a self-contained deterministic simulation and design a non-training online procedure that jointly selects client and relay motion, link probes, next-hop forwarding, retransmission, and the client's active command. The procedure should deliver useful commands and progress, move the client toward revealed goals, preserve viable multi-hop routes, and control collision, motion, energy, queue-wait, probe, and retransmission costs.

Use one reusable state-driven procedure across all canonical scenarios and roles. A role decision may use only that role's view defined below, together with public static inputs and its own history. Current time may be used to compute release, deadline, and age slack, but scenario names, phases, goal IDs, release slots, or absolute slots must not index pre-filled routes, relay positions, forwarding schedules, or goal plans. Analytical rules, deterministic estimation, search, planning, optimization, and other non-training methods are allowed; the method and project organization are otherwise your decisions.

### Experimental Setting and Constraints

#### Canonical scenarios, grid, and time

The integer grid is `x=0,...,19`, `y=0,...,13`. There are `H=96` one-second decision slots `t=0,...,95` and a non-decision terminal boundary at `t=96`. Nodes are the fixed ground station `G`, client `C`, and relays `R1,R2`, with rank order `G,C,R1,R2` equal to `0,1,2,3`. All four nodes start with `120` energy units. The five canonical scenario records must occur in this table order.

| scenario | phase `p` | starts `(G,C,R1,R2)` | closed obstacle cells | interference windows |
|---|---:|---|---|---|
| `cross_corridor` | 1 | `(1,7),(18,7),(6,5),(12,9)` | `x=9, y in {0,...,5,8,...,13}` | `[34,46)` |
| `doorway_shift` | 4 | `(1,2),(18,11),(6,3),(13,10)` | `x=8, y in {0,...,4,6,...,13}` | `[42,54)` |
| `branching_mission` | 7 | `(1,11),(18,2),(6,10),(13,3)` | `(7,y), y=3,...,8` and `(12,y), y=5,...,10` | `[20,32)` and `[62,74)` |
| `moving_blocker` | 10 | `(1,7),(18,7),(5,6),(14,8)` | `x=10, y in {0,...,3,5,...,13}` | `[28,40)` |
| `relay_bottleneck` | 13 | `(1,6),(18,6),(6,4),(13,9)` | `x=9, y in {0,...,2,4,...,10,12,13}` | `[48,66)` |

The fixed simulator uses `p` and the interference-window column as environment truth. They must not be included in a role's decision view or used indirectly as scenario lookup keys. The map, all initial positions, the unshifted goal table, and release slots are public static inputs.

#### Goal release and mission state

Goals `g1,...,g6` are released to `G` at slots `(0,16,32,48,64,80)`. Their unshifted `(target,value,deadline offset)` values are

```text
g1=((4,3), 1.00, 24)   g2=((15,10), 1.10, 24)
g3=((7,11),1.15, 25)   g4=((13,2),  1.20, 24)
g5=((5,8), 1.25, 23)   g6=((17,6),  1.30, 15).
```

For a scenario, set `delta_x=(floor(p/3) mod 3)-1` and shift each target by `(delta_x,0)`. If the shifted cell is closed or outside the grid, replace it by the first open, in-grid neighbor in the order `N,E,S,W`, where `N=(0,1), E=(1,0), S=(0,-1), W=(-1,0)`. A deadline is release plus its offset, so `g6` has deadline `95`. The shifted target and unreleased-goal contents are environment truth until the command is revealed to a role through its local queue or delivery.

After processing start-of-slot arrivals in a release slot, `G` automatically creates the unique command `cmd-{goal_id}` and attempts to admit it to `G`'s queue. It is available for a send request in that same slot. When `C` first completes a goal on time, it creates the unique progress packet `prog-{goal_id}` at the end of that slot; it can first be sent in the next decision slot. An automatically created packet is still recorded if its source queue is full, but its outcome is immediately `dropped`.

For goal `g`, define

```text
W_g = {integer t : deadline_g < t <= min(deadline_g+8,H)}.
```

Thus `W_g6={96}`. Once a command has reached `C`, the client may select its unfinished goal as `active_command_goal` in any slot through the deadline, inclusive. The goal completes only if that goal is selected and `C` is at its target after movement in that slot. A progress packet confirms the goal only when it is finally delivered to `G` in `W_g`; an earlier or later final delivery does not confirm it.

#### Movement, energy, and simultaneous collision resolution

Every node requests one move from `N,E,S,W,stay`, with `stay=(0,0)`. `G` must request `stay`. A request outside the grid, into an obstacle, by `G` for a non-`stay` move, or without enough slot-start energy is illegal and is replaced by `stay` for diagnostic replay. A successfully executed non-`stay` move costs `1` for `C` and `1.25` for either relay. Communication has no direct energy cost.

Resolve all remaining moves simultaneously by this fixed-point rule. Initially, each tentative destination is its start-of-slot position plus its sanitized move. Repeatedly add all of the following nodes to the collision set, replacing every newly added node's destination by its start-of-slot position, until no node is added:

* every node in a group with the same tentative destination;
* both nodes in any two-node exchange of start-of-slot positions; and
* a moving node entering the start-of-slot cell of a node that requested or was sanitized to  `stay`, or that was stopped in an earlier iteration, together with that occupying node.

A cycle of length greater than two is allowed when its tentative destinations are distinct. Every node in the final collision set stays, and `collision_nodes` lists the set in `G,C,R1,R2` order. Only a non-colliding, non-`stay` move spends energy.

#### Link model, packets, and FIFO queues

For distinct nodes `u,v`, let `d_uv(t)` be their shortest four-neighbor path distance through open cells using their start-of-slot positions. It is infinity if no path exists. Define

```text
i_s(t) = 0.15  if t is in a scenario interference window
                    or (5*t+2*p) mod 29 < 3,
         0     otherwise,
q_uv(t) = clip(1 - 0.045*d_uv(t) - 0.18*i_s(t), 0, 1),
clip(z,0,1) = min(1,max(0,z)).
```

Use `q_uv(t)=0` when the nodes are disconnected. A selected transmission is physically successful exactly when

```text
q_uv(t) >= 0.65
and (13*t + 3*rank(u) + 5*rank(v) + p) mod 17 > 1.
```

A physical success arrives at the start of slot `t+1`; other attempts produce no arrival. Links are undirected for quality, but the loss expression uses the ordered sender `u` and receiver `v`.

Commands and progress packets each have one payload unit and initial TTL `4`. Every physical success decrements TTL at arrival, so at most four successful hops are possible. Final delivery to `C` for a command or `G` for progress is processed before TTL and age tests and remains a delivery even when late. At a non-final receiver, process a physical arrival in this order:

1. if post-hop TTL is zero, mark the packet `expired`;
2. if it is a command arriving at `t>=deadline`, or progress arriving at   `t>=confirmation_window_end`, mark it `expired`;
3. if the receiver queue is full, mark it `dropped`;
4. otherwise enqueue it with entry slot `t`.

Each node has a capacity-`12` FIFO ordered by entry slot and then lexicographic `packet_id`. Only the head may be requested. At the start of every decision slot, after arrivals, repeatedly remove an age-expired head until the queue is empty or its head is not expired. A packet behind an unexpired head is not examined early; if a send later exposes it, it is first eligible for cleanup in the next decision slot. Therefore a command's latest final hop starts at `deadline-1`, and a progress packet's latest final hop starts at `confirmation_window_end-1`. Failed and non-selected attempts leave the packet at the same queue entry slot.

#### Requests, probes, attempts, and arbitration

In every decision slot, each node submits exactly one request in `G,C,R1,R2` order. Its derived ID is `req-{slot:02d}-{node}`. The request contains its move and one communication `kind` from `send,probe,none`:

* `send` supplies non-null string `packet_id` and `next_hop`, where the packet is the exposed local  FIFO head and `next_hop` is one of the other three nodes;
* `probe` and `none` have null `packet_id` and `next_hop`.

Every `kind=send` entry counts toward a scenario's cumulative `240`-request limit, whether or not its reference is legal. Requests after number `240`, unknown or non-head packet references, and self-next-hops are illegal; their communication component is replaced by `none` and creates no hop attempt. A request within budget whose fields, reference, and next hop are legal is an eligible send. Every eligible send creates one hop attempt and increments `send_attempts`, including attempts not selected by the shared channel.

The channel selects at most one eligible send using the lexicographically smallest tuple

```text
(class_priority, packet.created_slot, rank(current_sender), packet_id, attempt_no),
```

where command priority is `0` and progress priority is `1`. The selected request's derived ID is `arbitrated_request_id`; it is null when there is no eligible send. Every other eligible attempt has `success=false`, no arrival, and reason `not_selected`. For the selected attempt, quality below `0.65` gives reason `quality`; otherwise a failed loss expression gives reason `loss`. Both have `success=false` and no arrival. A physical success has `success=true`, arrival `t+1`, and reason `ttl` only when that arrival is to a non-final node with post-hop TTL zero; every other physical success has reason `delivered`.

For each logical packet, `attempt_no` starts at `0` and is consecutive across all senders and hops; `hop_id` is `{packet_id}#a{attempt_no}`. `retransmission` is true exactly when the immediately preceding attempt was unsuccessful, including `not_selected`, and the packet has remained at the same sender until this request. The first request after a successful forwarding hop is not a retransmission. `queue_wait_sum` adds `request_slot-current_entry_slot` for selected attempts only.

A legal probe records ID `probe-{slot:02d}-{node}`, measures that node's three incident qualities at the request slot before movement, and makes that probe ID and measurements visible to the same node at the start of the next slot. Its `neighbors` use `G,C,R1,R2` order with the probing node removed. A probe does not enter send arbitration but consumes the node's one communication request. At most `96` probes may be requested per scenario; later probes are illegal and treated as `none`. Feedback from a slot-95 probe reaches terminal boundary `96` and cannot affect a decision. No other link, queue, or delivery feedback is shared between roles; a sender can observe its own queue retention or removal in its subsequent local state.

#### Role information and event order

`G` observes released goal contents, finally delivered progress, its own queue, and its own arrived probe feedback. `C` observes finally delivered command contents, its own position and energy, its own queue, and its own arrived probe feedback. Each relay observes its own position and energy, packets admitted to its own FIFO, and its own arrived probe feedback. Every role observes packets admitted to its local FIFO and retains its applicable class-final deliveries; a non-final physical arrival rejected by TTL, age, or FIFO capacity is an environment-only event and is not exposed to that role. A role also knows the public static inputs and its own action and queue history, but not another node's current position, energy, queue, or probe feedback.

The fixed simulator may use phase `p`, all current positions and queues, actual targets, future interference and loss events, and scheduled arrivals to update environment truth. Those values do not enter a role's decision view unless revealed by the rules above. In particular, future link quality, unreleased shifted targets, and undelivered packet contents are unavailable.

Each decision slot uses this order:

1. process prior physical arrivals and probe feedback; apply TTL, age, and capacity to non-final   arrivals; repeatedly clean age-expired FIFO heads;
2. release the current goal, if any, and create/admit its command at `G`;
3. expose the four separate local role views;
4. let `C` request an `active_command_goal` or null, and let every role request its move and   communication kind;
5. sanitize illegal components, create eligible hop attempts, arbitrate the shared channel, and   schedule any physical success;
6. execute movement by the collision fixed point and update energy;
7. complete the selected goal if eligible and create/admit its progress packet;
8. record the end-of-slot state.

An invalid active goal is replaced by null; it is legal only if its command has reached `C`, the goal is unfinished, and the current slot is no later than its deadline. Sanitization makes replay deterministic but does not make an invalid request legal: it still fails decision legality and gates all task metrics.

At terminal boundary `96`, process physical arrivals from slot `95` using the same final-delivery or TTL--age--capacity rules and confirm any progress arriving in its window. Then freeze packets and queues. Do not clean existing queue heads, release a goal, create a packet, move, transmit, or probe at the terminal boundary.

#### Task metrics

All metrics are independently replayed and clipped to `[0,1]`; the five scenarios are averaged equally. Empty-set means and ratios with zero denominator are zero unless stated otherwise.

* `goal_progress`: within a scenario, give a goal value `1` when it completes by its deadline and  its progress arrives in `W_g`, `0.5` when it completes by its deadline without such confirmation,  and `0` otherwise. Take the goal-value-weighted mean.
* `command_connectivity`: within a scenario, compute  `0.6*mean(A_g)+0.4*mean(S_g/(deadline_g-release_g+1))` over the six goals. `A_g=1` exactly when  command arrival is no later than `release_g+8`; otherwise it is zero. `S_g` counts slots in the  inclusive release-to-deadline interval whose recorded `active_command_goal` is `g`.
* `safe_motion_efficiency`: let `collision_slots` count slots with nonempty `collision_nodes`,  `motion_cost` count successful non-`stay` node-slots for `C,R1,R2`, and  `energy_used=sum(120-energy_node_at_96)` over those three nodes. Compute  `0.4*clip(1-collision_slots/96,0,1) + 0.3*clip(1-motion_cost/288,0,1)
  + 0.3*clip(1-energy_used/360,0,1)`.
* `delivery_efficiency`: when `send_attempts>0`, compute  `0.5*useful_deliveries/send_attempts + 0.5*confirmed_progress/6`; otherwise zero.  A useful delivery is the first final command delivery no later than its deadline or the unique  progress final delivery in its confirmation window.
* `communication_overhead_efficiency`: this is zero when `useful_deliveries=0`. Otherwise let  `mean_queue_wait=queue_wait_sum/selected_send_attempts`, using zero when no attempt was selected,  and compute `0.4*clip(1-mean_queue_wait/12,0,1)
  + 0.3*clip(1-probe_count/96,0,1)
  + 0.3*clip(1-retransmissions/send_attempts,0,1)`.

The normalized task performance is

```text
(8*goal_progress + 4*command_connectivity + 3*safe_motion_efficiency
 + 3*delivery_efficiency + 2*communication_overhead_efficiency) / 20.
```

#### Result contract

`python run.py` must create or overwrite a UTF-8 `results.json` in the project root. It is strict JSON: duplicate object keys, `NaN`, and infinities are invalid. The root object has exactly `scenarios`. That array contains one record for each canonical scenario in table order. Every scenario record has exactly `scenario`, `slots`, `robots`, `goals`, `packets`, `hops`, `probes`, and `mission`.

`slots` contains 96 records ordered by integer `slot=0,...,95`. Each has exactly `slot`, `requests`, `arbitrated_request_id`, `active_command_goal`, and `state`:

* `requests` contains four records in `G,C,R1,R2` order. Each has exactly `node`, `move`, `kind`,  `packet_id`, and `next_hop`, using the request domains above. Only `send` has two non-null strings.
* `arbitrated_request_id` is the selected derived request ID or null. `active_command_goal` is one of  `g1,...,g6` or null.
* `state` has exactly boolean `interference`, nonnegative integer `budget_used`, `collision_nodes`  in node-rank order, and `client_position` as two grid integers. The last three are end-of-slot  values. `budget_used` is the cumulative count of `kind=send` entries through this slot, including  an over-budget entry in an invalid result.

`robots` contains four records in `G,C,R1,R2` order. Each has exactly `node`, `positions`, `energy`, `queue_lengths`, and `local_feedback`. `positions` and `energy` have length 97: index `t` is the start of slot `t` and index `t+1` is its end; index 96 is also the terminal position/energy. A position is two grid integers, and energy is finite and in `[0,120]`. `queue_lengths` and `local_feedback` have length 96. Their index `t` gives the queue length at the end of slot `t`, after progress admission and before any terminal arrival, and the lexicographically sorted probe IDs visible at the start of slot `t`, respectively. Queue lengths are integers in `[0,12]`.

`goals` contains six records in `g1,...,g6` order. Each has exactly `goal_id`, `release_slot`, `target_x`, `target_y`, `value`, `deadline`, `confirmation_window_end`, `command_packet_id`, `command_arrival_slot`, `completion_slot`, `progress_packet_id`, `confirmation_slot`, and `status`. Fixed fields equal the released goal model. Event times and progress ID are null when the event did not occur; command arrival includes a late final delivery, and confirmation may be at `96`. Status is:

* `confirmed` when the goal completed on time and progress arrived in `W_g`;
* `completed` when it completed on time but was not confirmed;
* `commanded` when its command arrived by the deadline but it did not complete; and
* `expired` otherwise.

`packets` is ordered by `(created_slot,packet_id)`. Each record has exactly unique `packet_id`, `class`, `goal_id`, `source`, `created_slot`, `ttl`, `initial_node`, `final_node`, and `outcome`. `class` is `command` or `progress`; commands have `source=initial_node=G, final_node=C`, and progress has `source=initial_node=C, final_node=G`. `ttl` is the integer remaining after terminal processing in `[0,4]`. `outcome` is `delivered,expired,dropped,unfinished`.

`hops` contains one record per eligible send, ordered by request slot and then sender rank. Each has exactly unique `hop_id`, `packet_id`, integer `attempt_no`, boolean `retransmission`, `from`, `to`, integer `request_slot`, `arrival_slot`, finite `quality` in `[0,1]`, boolean `success`, and `reason` in `delivered,quality,loss,ttl,not_selected`. IDs and attempt numbers follow the definitions above. For `quality,loss,not_selected`, success is false and arrival is null. For `delivered,ttl`, success is true and arrival equals `request_slot+1`.

`probes` contains legal, within-budget probes ordered by request slot and node rank. Each has exactly unique `probe_id`, `node`, integer `request_slot`, integer `feedback_slot=request_slot+1`, `neighbors`, and `qualities`. `neighbors` is the three-node rank-ordered array excluding the probe node; `qualities` is an equal-length array of finite values in `[0,1]` measured before movement.

`mission` has exactly `send_attempts`, `useful_deliveries`, `queue_wait_sum`, `probe_count`, `retransmissions`, `collision_slots`, `motion_cost`, `energy_used`, `completed_value`, and `terminal_arrival_packet_ids`. `send_attempts` is the number of eligible-hop records; `useful_deliveries` uses the metric definition; `queue_wait_sum` uses selected attempts only; `probe_count` is the number of legal, within-budget probes; `retransmissions` counts true hop flags; `collision_slots` and `motion_cost` use the metric definitions; and `completed_value` sums values of goals completed on time. These are nonnegative integers except finite nonnegative `energy_used` and `completed_value`. `terminal_arrival_packet_ids` lists in lexicographic order every packet physically arriving at `96`, including a non-final arrival then expired or dropped.

`slots`, `robots`, and `goals` are immutable canonical records. Packet, hop, probe, and queue events are dynamic evidence. All IDs are unique in a scenario and references stay in that scenario. Integer, Boolean, enum, key-set, length, and ordering requirements are exact. Finite numeric replay comparisons use absolute tolerance `1e-6`. Requests and active-goal choices are replay inputs; link qualities and outcomes, packet state, robot trajectories, collisions, feedback, goal events, and mission summaries are replay evidence.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` from the project code and fixed conditions without interactive input. The implementation may use a reasonable multi-file organization.

`report.md` must contain nonempty sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must describe the implemented event model and local-information method, actual canonical results, relay--routing and queue--deadline evidence, safety and energy outcomes, and evidence-supported limitations. Any additional controlled comparison should preserve comparable inputs and retain condition-level evidence in the project.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "the new results.json is strict JSON with the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "scenario records provide typed slot, robot, goal, packet, hop, probe, and mission evidence", weight: 4}
- {id: canonical_coverage, desc: "all scenario, slot, robot, and goal canonical keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "moves, active commands, FIFO sends, next hops, probes, energy, and request budgets are legal", weight: 4}
- {id: replay_validity, desc: "fixed arrival, queue, link, arbitration, movement, collision, goal, and terminal replay matches the records", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-weighted goal completion and confirmation", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed command arrival and active-command connectivity", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed collision, motion, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed useful message delivery efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed queue-wait, probe, and retransmission efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the local-information relay, queue, motion, and mission model"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given, or the relay, packet, motion, or mission model is materially wrong; Implementation/evidence: no executable client, relay, and multi-hop process or corresponding records are present."}
    - {score: 0.4, desc: "Report: only movement or link quality is treated and a decisive local-information, FIFO, TTL, arbitration, deadline, or collision relation is omitted; Implementation/evidence: state transitions or event order are partial or materially depart from the report."}
    - {score: 0.7, desc: "Report: release, local views, link quality, FIFO service, arbitration, TTL, movement, energy, completion, and confirmation are accurately defined; Implementation/evidence: all canonical scenarios implement the core model with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely specifies event order, boundary handling, and how local multi-hop delivery changes command use and progress confirmation; Implementation/evidence: traceable requests, hops, queues, robot states, and goal events faithfully realize those relations without material divergence."}

- id: method_design
  desc: "quality of the unified local state-driven relay-routing procedure"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, or it is prohibited or violates role information boundaries; Implementation/evidence: no legal reusable decision procedure is present."}
    - {score: 0.4, desc: "Report: the method relies on fixed relay positions, scenario action tables, or one-step current-link greed and ignores a decisive queue, deadline, or motion consequence; Implementation/evidence: decisions are pre-filled or do not realize the described procedure."}
    - {score: 0.7, desc: "Report: a legal, reproducible, reusable online procedure addresses relay motion, probing, next-hop forwarding, retransmission, and client progress under the principal temporal constraints; Implementation/evidence: one state-driven procedure generates decisions across all scenarios and roles, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: the justified unified method uses local link evidence, queue age, TTL, deadline slack, reachability, and energy to coordinate relay reconfiguration and forwarding; Implementation/evidence: contrasting local congestion or link states induce corresponding decisions without scenario-, goal-, phase-, or slot-indexed schedules or material report divergence."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of relay, routing, and failure-boundary experiments"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose is provided; Implementation/evidence: the project contains no checkable experimental execution."}
    - {score: 0.4, desc: "Report: only one map or aggregate condition is discussed and the central relay-to-mission relation is not tested; Implementation/evidence: condition-level records do not support the principal claims."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one explicitly identified comparable controlled condition beyond canonical aggregate contrasts test a central relay or routing mechanism; Implementation/evidence: both conditions execute with comparable inputs and preserve condition-level trajectory, hop, queue, and goal evidence."}
    - {score: 1.0, desc: "Report: non-redundant single-factor comparisons test the relay-routing mechanism and an additional interference, capacity, TTL, or deadline boundary; Implementation/evidence: reconstructable condition-level evidence and metrics distinguish both claims rather than merely restating canonical totals."}

- id: research_analysis
  desc: "evidence-grounded interpretation of relay--routing and queue--deadline behavior"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts recorded results; Implementation/evidence: the project cannot support its conclusions."}
    - {score: 0.4, desc: "Report: the discussion lists delivery, collision, or goal totals without a supported causal explanation or material limitation; Implementation/evidence: records do not support the asserted inference."}
    - {score: 0.7, desc: "Report: conclusions use actual records to explain condition-dependent relay position, interference, queue, energy, and deadline trends, distinguish observation from inference, and state limitations; Implementation/evidence: implemented comparisons support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies how relay reconfiguration, queue age, and mission deadlines interact, tests the explanation against contrasting evidence or a failure boundary, and limits its scope; Implementation/evidence: reproducible condition-level traces and comparisons fully support the central conclusions."}
