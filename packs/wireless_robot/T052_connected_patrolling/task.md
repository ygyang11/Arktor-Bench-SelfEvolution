---
id: T052_connected_patrolling
name: Connectivity-Constrained Multi-Robot Patrolling with Timely Data Delivery
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

Three robots repeatedly patrol sensing locations on a road graph. A visit serves two purposes: it reduces the location's current idleness, and it creates a monitoring record that must reach a fixed base before its deadline. Robots may carry records, exchange them during short wireless contacts, and forward them toward the base. Slow bridge edges, opposing bridge traffic, changing sensing priorities, finite queues, and periodic interference make a short patrol route insufficient by itself. A robot that remains far from its teammates may cover locations well while its records expire; a robot that waits only for communication can deliver fresh data while allowing important locations to become idle.

### Research Problem

Build a deterministic simulator and a non-training online method that jointly chooses recurrent patrol motion and local store-and-forward proposals. The method should control priority-weighted worst idleness, revisit all sensing locations in each period, and deliver generated records to the base on time without wasting the shared packet budget, queue capacity, energy, or narrow bridge access. Use one reusable state-driven procedure across all canonical scenarios. Each robot's decision must use only its static graph input, current local state, revealed priorities and contacts, and locally received history; it must not inspect another robot's private state or any future priority, interference, arrival, or arbitration outcome. Deterministic search, graph planning, optimization, scheduling, and analytical heuristics are allowed.

### Experimental Setting and Constraints

#### Canonical scenarios and graph

The canonical scenario IDs are `balanced_loop`, `long_bridge`, `relay_gap`, `priority_shift`, and `intermittent_base`. Their order in `results.json` has no meaning. Every scenario has 180 one-second decision slots `t=0,...,179`; boundary 180 is terminal. A patrol period has 30 slots, so period `p=floor(t/30)` ranges from 0 through 5.

The public undirected road graph contains

```text
V = {v_xy : x=0,...,5 and y=0,...,3}.
```

Vertex `v_xy` has physical coordinate `(2x,2y)`. Two vertices have a movement edge exactly when their grid indices are horizontally or vertically adjacent. The base `B` is fixed at `v_00`. `R1`, `R2`, and `R3` start at `v_01`, `v_02`, and `v_03`, respectively, each with energy 220. The sensing set contains the 18 vertices with `y in {1,2,3}`; the six `y=0` vertices are relay-only. Define `index(v_xy)=4x+y`.

Let `phi` equal 0, 1, 2, 0, and 1 in the scenario order above. For an edge between `v_xy` and `v_x'y'`, the default integer travel duration is

```text
tau(e) = 1 + ((x+y+x'+y'+phi) mod 3).
```

In `long_bridge`, each of the four edges between columns 2 and 3 instead has duration 5. In `relay_gap`, each such edge instead has duration 4. In `intermittent_base`, the duration of every edge incident to `v_00` is increased by one after applying the default formula. The column-2/3 edges form the narrow-edge set.

A robot at a vertex may stay or request entry into an adjacent edge. Entry at slot `t` consumes the first of `tau(e)` travel slots; a duration-one edge therefore arrives at the destination at the end of the same slot. A robot already in transit has no vertex contact and must continue on its fixed edge; it arrives at the end of the slot in which its remaining travel becomes zero. Accepted entry and continued travel cost 1 energy per occupied robot-slot. Actual stay costs 0.05.

An entry request on a narrow edge is rejected if another robot is traversing that edge in the opposite direction at the slot start. If new requests enter the same narrow edge in both directions, all new requests in those opposing directions are rejected. A rejected requester remains at its source vertex, spends 0.05 energy, and incurs one collision robot-slot; an already travelling robot is never displaced. Same-direction travel and vertex co-location are allowed. On requesting entry, a robot must have enough current energy to finish the entire edge, plus 0.2 if it also submits a transmission proposal in that slot. A staying robot must analogously reserve 0.05 plus that possible send cost.

#### Priorities, idleness, and visits

Before slot 0, every sensing location has `last_visit=-1`. Initial occupancy is not a visit. At the decision boundary of each slot `t`, its idleness sample is

```text
idle(v,t) = t - last_visit(v).
```

Only an edge arrival at the end of a slot creates a visit. If multiple robots arrive at the same sensing vertex in the same slot, the first robot in `R1,R2,R3` order is the sole visit winner. That visit sets `last_visit(v)=t`, affects idleness beginning with the next slot, and belongs to period `floor(t/30)`.

Each sensing priority is in `{1,2,3}`. The complete vector for period `p` is revealed at boundary `30p` and remains fixed for that period. In scenarios other than the two exceptions below, it is

```text
q_s(v_xy,p) = 1 + ((index(v_xy)+p+phi_s) mod 3).
```

`long_bridge` adds one to this value for `x>=3` and clips it to 3. For `priority_shift` and `p>=3`, `v_xy` instead takes the default value of its mirror `v_(5-x)y` in the same period. This exchanges the two sides' priority distributions. The environment component of a project may contain these rules to generate canonical truth, but the online policy interface may receive only the current revealed vector, not the scenario ID or any later vector.

#### Records, queues, and forwarding

Each effective visit creates one 32-bit record with canonical string ID `<location>@<visit_slot:03d>`, for example `v_31@007`. Its fixed metadata are the winning robot, location, visit slot, priority at that slot, and inclusive base deadline `visit_slot+45`.

Each rover has a FIFO queue holding at most 12 record tokens and a persistent `seen` set. A token also has a hop count, initially zero. A locally generated record is added to its winner's `seen` set and FIFO at the visit-slot end and first becomes sendable at the next boundary. A novel record that arrives from another rover at boundary `t` becomes sendable in slot `t`. Records in a packet retain their listed order. Before admitting a novel batch, the receiver discards duplicates already in its `seen` set. If the whole remaining batch would exceed capacity, it discards existing FIFO heads until the whole batch fits, then appends the batch in packet order. A new local visit uses the same capacity rule with a one-record batch.

A packet is always 160 bits: a 64-bit header and one, two, or three 32-bit records. Unused payload space cannot be combined with another packet. A selected packet immediately removes its listed tokens from the source FIFO and increments each token's hop count; it arrives at the next boundary. Arrival processing handles each record in packet order:

- if the arrival boundary is greater than its deadline, the token terminates as late;
- at `B`, a previously unconfirmed ID is confirmed and a duplicate ID terminates;
- at a rover, an ID already in its persistent `seen` set terminates as a duplicate;
- a novel third-hop arrival at a rover is added to `seen` and terminates at the hop limit;
- every other novel rover arrival is added to `seen` and admitted to that rover's FIFO.

At each nonterminal boundary, after arrivals and before decisions, a FIFO token whose deadline is strictly less than the current slot expires. Overflow, late, duplicate, hop-limit, and expired tokens are terminal and cannot be sent again. The base never forwards.

#### Wireless contacts and arbitration

The fixed unordered pair order is

```text
R1-R2, R1-R3, R2-R3, R1-B, R2-B, R3-B.
```

Two nodes have base contact when both occupy vertices and either share a vertex or their Euclidean distance is at most 3.0. `B` always occupies `v_00`. In `relay_gap`, if two endpoints lie on opposite sides of the column-2/3 boundary, contact is allowed only when their vertices are exactly `v_20` and `v_30`; all same-side contacts use the normal rule.

For pair `pair` in slot `t`, interference `a_s(pair,t)` equals one when `t mod 30` is in the table below and zero otherwise. In `intermittent_base`, every rover-base pair is additionally blocked at offsets `{8,9,10,11,12,13}`; this union takes precedence over the sparse row. A pair is available exactly when it has base contact and `a=0`. No other blocking exists.

| Scenario | Pair | Blocked offsets |
|---|---|---|
| `balanced_loop` | `R1-R2` | `{5,17}` |
| `balanced_loop` | `R1-R3` | `{11}` |
| `balanced_loop` | `R2-R3` | `{23}` |
| `balanced_loop` | `R1-B` | `{2,20}` |
| `balanced_loop` | `R2-B` | `{14}` |
| `balanced_loop` | `R3-B` | `{8,26}` |
| `long_bridge` | `R1-R2` | `{3,19}` |
| `long_bridge` | `R1-R3` | `{7}` |
| `long_bridge` | `R2-R3` | `{13,28}` |
| `long_bridge` | `R1-B` | `{1,16}` |
| `long_bridge` | `R2-B` | `{9}` |
| `long_bridge` | `R3-B` | `{21,27}` |
| `relay_gap` | `R1-R2` | `{4}` |
| `relay_gap` | `R1-R3` | `{10,24}` |
| `relay_gap` | `R2-R3` | `{18}` |
| `relay_gap` | `R1-B` | `{6,22}` |
| `relay_gap` | `R2-B` | `{15}` |
| `relay_gap` | `R3-B` | `{0,25}` |
| `priority_shift` | `R1-R2` | `{1,14}` |
| `priority_shift` | `R1-R3` | `{6}` |
| `priority_shift` | `R2-R3` | `{20,29}` |
| `priority_shift` | `R1-B` | `{5}` |
| `priority_shift` | `R2-B` | `{12,24}` |
| `priority_shift` | `R3-B` | `{17}` |
| `intermittent_base` | `R1-R2` | `{2,16}` |
| `intermittent_base` | `R1-R3` | `{9}` |
| `intermittent_base` | `R2-R3` | `{21,27}` |
| `intermittent_base` | `R1-B` | `{4,19}` |
| `intermittent_base` | `R2-B` | `{1,14}` |
| `intermittent_base` | `R3-B` | `{6,23}` |

At most one packet is selected per slot and at most 120 packets are selected in a scenario. Each rover may submit at most one proposal in a slot. It must be at a vertex, its destination must be a currently available contact neighbor, and its one-to-three `record_ids` must be an exact nonempty prefix of its current FIFO. Every listed token must be unexpired and have hop count below three. Proposal priority is an integer from 0 through 9. A proposal must reserve its possible 0.2 sender energy in addition to the requested movement reserve.

While packet budget remains, the arbiter selects the first legal proposal ordered by `(-priority,src,dst,first_record_id)`, using Unicode code-point ordering for strings. Other legal proposals are not sent and spend no energy or packet budget. Once 120 packets have been selected, later well-formed proposals remain permissible but none is selected. A slot-179 selection arrives at terminal boundary 180 and uses the same deadline, duplicate, confirmation, and hop-limit rules.

#### Local information and event order

Each robot's online method may read the complete static graph and its scenario's complete travel-duration map, its own current vertex or transit state, energy, local visit history, FIFO and persistent `seen` records, the current revealed priority vector, current available incident contacts, and its own prior transmission feedback. It may not read another robot's untransmitted visits, position, transit state, queue, proposal, or energy, nor any future priority, interference, arrival, or arbitration result. The environment simulator necessarily evaluates the public canonical formulas, but it must expose only the allowed current snapshot to the reusable decision procedure. The procedure must not branch on a scenario ID or use scenario-, period-, priority-, location-, or absolute-slot action tables.

Each slot `t=0,...,179` has this order:

1. Process the packet selected in `t-1`, including confirmation, duplicate/late/hop-limit termination,   and rover admission; then expire queued tokens with `deadline<t`.
2. At a period boundary, reveal that period's priority vector.
3. Compute current contacts from slot-start vertex states and current interference, then sample   decision-before idleness.
4. Every rover records one nullable movement destination and may submit one local proposal.
5. Validate the local decisions, select at most one packet by the fixed arbiter, remove its FIFO   prefix, and charge its sender.
6. Continue existing transit and resolve new entries and narrow-edge conflicts; charge actual motion   or stay, advance travel, and process slot-end arrivals at vertices.
7. Resolve one visit winner per arrived sensing vertex, update `last_visit`, and append its record for   availability at the next boundary.

Boundary 180 performs only step 1 for the slot-179 packet. It has no priority reveal, contact, idleness sample, movement, or proposal.

The movement destination and proposal rows in `results.json` are decisions submitted in step 4. Visits, priority state, contacts, selected packets, queue state, idleness, collisions, energy, and confirmations are execution facts derived by the fixed replay; candidates do not submit them.

#### Metrics

Let `clip(x,0,1)=min(1,max(0,x))`. All metric values are averaged over the five scenarios.

For scenario `s`, define

```text
weighted_idle(v,t) = (priority(v,floor(t/30))/3) * idle(v,t)
I_s = max over sensing v and t=0,...,179 of weighted_idle(v,t)
C_s = fraction of the 18*6 location-period pairs having at least one effective visit.
```

`primary_metrics_1` is the priority-weighted idleness and recurrent-coverage score

```text
0.6*clip(1-I_s/180,0,1) + 0.4*C_s.
```

For every effective visit record `r`, let `base_arrival_r` be its first confirmation boundary or null. Let `W_s` be the sum of all visit priorities, `D_s` the priority sum for records confirmed no later than their inclusive deadline, and

```text
F_s = sum over on-time r of priority_r *
      clip(1-(base_arrival_r-visit_slot_r)/45,0,1).
```

Both delivery terms are zero when there are no visits. `primary_metrics_2` is

```text
0.65*D_s/max(1,W_s) + 0.35*F_s/max(1,W_s).
```

Let `motion_s` count accepted entry and continue robot-slots; let `unproductive_s` count physically stationary robot-slots, including rejected entries, in which the robot neither creates a visit, sends a selected packet, nor receives a novel accepted record; let `collision_s` count rejected-entry robot-slots. Let `energy_used_s` include all rover travel, stay, and selected-send energy. `primary_metrics_3` is

```text
C_s * (0.25*clip(1-motion_s/540,0,1)
     + 0.25*clip(1-unproductive_s/540,0,1)
     + 0.25*clip(1-collision_s/540,0,1)
     + 0.25*clip(1-energy_used_s/660,0,1)).
```

For each record hop in a selected packet of `k` records, count `32+64/k` useful bits exactly when that record is eventually first confirmed on time. Let `sent_bits=160*selected_packet_count`. An eligible contact slot is a directed `(slot,src,dst)` while packet budget remains where `src` has a nonexpired FIFO head, `dst` is currently available, and `dst` has not previously seen that head ID. It is productive when the selected packet on that directed contact causes at least one record to be accepted by a rover or first confirmed on time by `B` at the next boundary. Empty sent or eligible samples contribute zero. `auxiliary_metrics_1` is

```text
0.6*useful_bits/max(1,sent_bits)
+ 0.4*productive_contact_slots/max(1,eligible_contact_slots).
```

For each record first confirmed on time, sum `send_slot-queue_available_boundary` over all of its custody queues and call this `total_queue_wait`. With no such record, `wait_s=0`; otherwise `wait_s=clip(1-mean(total_queue_wait)/45,0,1)`. Let

```text
hop_s = successful_novel_record_hops / max(1,all_selected_record_hops)
redundancy_s = 1 - duplicate_record_arrivals/all_record_arrivals.
```

A successful novel hop is one accepted by a rover or first confirmed on time by `B`. `hop_s` is zero with no selected record hop, and `redundancy_s` is zero with no record arrival. Then `auxiliary_metrics_2 = 0.4*wait_s + 0.3*hop_s + 0.3*redundancy_s`.

The normalized task performance is

```text
(8*primary_metrics_1 + 4*primary_metrics_2 + 3*primary_metrics_3
 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2) / 20.
```

All five metrics lie in `[0,1]` and are computed from independent replay.

#### Results contract

`results.json` has exactly one root key, `scenarios`, whose value is an array. It contains exactly one record for each canonical scenario. Each scenario record has exactly the keys `scenario`, `moves`, and `transmission_proposals`:

- `scenario` is the canonical scenario string.
- `moves` is an array of exactly 540 rows, one for every `(slot,robot)`. Each row has exactly integer  `slot` in `0..179`, string `robot` in `{R1,R2,R3}`, and `destination`, which is either null or a  vertex string. At a vertex, null requests stay and a non-null value requests an adjacent edge. In  transit, destination must be null and denotes forced continuation. Row order is arbitrary.
- `transmission_proposals` is a dynamic array. Each row has exactly integer `slot` in `0..179`,  `src` in `{R1,R2,R3}`, different `dst` in `{R1,R2,R3,B}`, a `record_ids` array of one to three  canonical record-ID strings, and integer `priority` in `0..9`. Each `(slot,src)` occurs at most  once. Absence means no proposal. Row order is arbitrary.

The arrays are indexed by their declared public keys; no reference row order, route, proposal, or priority is used. JSON must be strict: duplicate object keys, `NaN`, and `Infinity` are invalid, and booleans are not integers. The grader derives all visits, links, packets, queues, states, and metrics and does not require checkpoints or other redundant state tables.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain the six non-empty Markdown sections `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The exact `results.json` contract above is reserved for canonical evaluation. Controlled comparisons are optional for the validity gate but support higher Judge levels. Record them in `report.md`, reproducible project code, or optional additional text/JSON project artifacts; any clear condition-level representation is accepted, and no extra filename or schema is required.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "movement destinations and transmission proposals have the declared forms", weight: 4}
- {id: canonical_coverage, desc: "every scenario and rover-slot movement key occurs exactly once", weight: 4}
- {id: decision_legality, desc: "movement, energy reserves, proposal keys, and current contacts are legal", weight: 4}
- {id: replay_validity, desc: "independent replay validates visits, FIFO custody, deadlines, hops, delivery, and resource invariants", weight: 6}
- {id: primary_metrics_1, desc: "priority-weighted worst idleness and recurrent period coverage", weight: 8}
- {id: primary_metrics_2, desc: "on-time base delivery and delivered-record freshness", weight: 4}
- {id: primary_metrics_3, desc: "coverage-conditioned motion, waiting, collision, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful packet bits and productive eligible-contact use", weight: 3}
- {id: auxiliary_metrics_2, desc: "queue wait, novel-hop success, and duplicate suppression", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the recurrent patrol, intermittent forwarding, and deadline formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent recurrent-patrol and delivery model; Implementation/evidence: no executable patrol or record process."}
    - {score: 0.4, desc: "Report: idleness, priority, transit, contact, custody, or deadline relations are decisively incomplete; Implementation/evidence: state or event timing materially diverges."}
    - {score: 0.7, desc: "Report: graph motion, periodic visits, revealed priority, intermittent contacts, FIFO forwarding, and deadlines are accurate; Implementation/evidence: all scenarios implement the core model with limited noncritical omissions."}
    - {score: 1.0, desc: "Report: movement conflicts, visit timing, local information, custody transitions, delivery freshness, and objective trade-offs are precisely connected; Implementation/evidence: event order and auditable behavior are faithfully realized without material divergence."}
- id: method_design
  desc: "quality of the unified online patrol, rendezvous, and forwarding method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited method is used; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: a fixed route or direct-only policy ignores decisive priority, contact, queue, or deadline effects; Implementation/evidence: decisions are unit-specific or inconsistent."}
    - {score: 0.7, desc: "Report: a legal unified method coordinates recurrent motion, meetings, routing, and revealed-priority response; Implementation/evidence: one procedure covers all units with limited long-horizon or delivery trade-off handling."}
    - {score: 1.0, desc: "Report: worst idleness, priority shifts, transit, meeting reachability, record age, FIFO pressure, interference, and energy are jointly justified; Implementation/evidence: evolving allowed state drives choices without scenario, period, or slot action tables."}
- id: experimental_coverage
  desc: "coverage of canonical patrols and controlled meeting, routing, or sensitivity comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only an incomplete condition is shown and the patrol-delivery relation is not tested; Implementation/evidence: condition evidence is insufficient."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled comparison directly test a meeting, patrol-allocation, routing, or related central trade-off; Implementation/evidence: both comparable conditions are executed with checkable records."}
    - {score: 1.0, desc: "Report: two complementary conditions include one path, meeting, or routing mechanism comparison and one graph, interference, priority, deadline, or queue sensitivity or failure boundary; Implementation/evidence: reproducible condition records and metrics support both conclusions beyond canonical aggregates."}
- id: research_analysis
  desc: "evidence-based analysis of idleness, meetings, record age, and timely delivery"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: claims are unsupported."}
    - {score: 0.4, desc: "Report: idleness or delivery values are repeated without causal path, meeting, queue, or deadline explanation and limits; Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: graph structure, revealed priority, meetings, queue age, energy, delivery, and limitations are interpreted from actual evidence; Implementation/evidence: records support the stated conditional trends."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies patrol-routing interactions and a sensitivity or failure boundary while testing alternatives and scope; Implementation/evidence: reconstructible condition records support the central conclusions."}
