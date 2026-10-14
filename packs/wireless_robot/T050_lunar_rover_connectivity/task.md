---
id: T050_lunar_rover_connectivity
name: Connectivity-Constrained Multi-Rover Exploration and Goal Assignment
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

Three lunar rovers leave a fixed lander to visit geographically dispersed science targets. Terrain elevation affects both traversability and radio line of sight. A target contributes its full mission value only after a science summary reaches the lander, possibly through other rovers, before its confirmation deadline. The same shared channel also carries acknowledgements back toward the first visitor. Long excursions can obtain valuable samples but create blocked links, stale packets, competing FIFO queues and costly duplicate visits.

### Research Problem

Build a deterministic simulator and a non-training online method that jointly chooses each rover's movement, the formation of one- or two-record science summaries, and local store-and-forward transmission proposals. The method should allocate targets without fixed target sequences, use terrain-dependent communication opportunities, return science value before deadlines, and manage collision, energy, queue age and a shared attempt budget. Use one reusable state-driven procedure for all five canonical scenarios. Each node's decisions may use only its current local snapshot and shared budget feedback; they must not use another rover's current position or queue, future link margins, future motion, future arrivals, or a scenario/target/slot answer table. Deterministic search, planning, optimization and analytical heuristics are allowed.

### Experimental Setting and Constraints

#### Canonical scenarios, terrain and goals

There are five canonical scenarios in the order below. Every scenario uses integer cells `0<=x<24`, `0<=y<20`, a fixed lander `L` at `(1,9)`, and rovers `R1,R2,R3` starting at `(2,8),(2,9),(2,10)`. Slots are one second long. Decisions and service occur at slots `t=0,...,149`; boundary 150 is terminal.

Let `phi_s` be the tabled scenario index parameter. Base elevation is `b_s(x,y)=(x+2*y+phi_s) mod 3`. Add `q_s(x,y)` as follows, and set `z_s=b_s+q_s`:

| scenario | `phi_s` | positive additional elevation `q_s` | radio dead-zone cells |
|---|---:|---|---|
| `ridge_shadow` | 0 | 4 if `x in {10,11}` and `y not in {8,9,10}`, else 0 | none |
| `distributed_ice` | 1 | 0 | none |
| `relay_saddle` | 2 | 4 if `x in {10,11,12,13}` and `y not in {8,9,10,11}`, else 0 | none |
| `narrow_crater` | 0 | 3 if `16<=(x-12)^2+(y-10)^2<=36` and `y not in {9,10}`, else 0 | none |
| `radio_dead_zone` | 1 | 0 | `D={(x,y):14<=x<=19,4<=y<=15}` |

The twenty fixed goals, in goal-ID order `G00,...,G19`, are `(4,3),(4,7),(4,12),(4,16),(8,2),(8,6),(8,10),(8,14),(8,18),(12,3),` `(12,7),(12,12),(12,16),(16,2),(16,6),(16,10),(16,14),(16,18),(21,5),(21,15)`. For goal index `j`, its value is `v_s(j)=1+((j+phi_s) mod 5)`, its visit deadline is `d_s(j)=70+4*(j mod 10)`, and its confirmation deadline is `c_s(j)=min(149,d_s(j)+24)`. Deadline comparisons include equality. Each rover starts with `185` energy units. Terrain, goals, values and deadlines are static information visible to every node.

#### Movement, collisions and visits

The action enumeration and coordinate deltas are fixed:

| action | delta `(dx,dy)` |
|---|---|
| `stay` | `(0,0)` |
| `north` | `(0,+1)` |
| `northeast` | `(+1,+1)` |
| `east` | `(+1,0)` |
| `southeast` | `(+1,-1)` |
| `south` | `(0,-1)` |
| `southwest` | `(-1,-1)` |
| `west` | `(-1,0)` |
| `northwest` | `(-1,+1)` |

A non-`stay` action is legal only if its proposed cell is in the grid, `|z_s(proposed)-z_s(old)|<=2`, and the rover can pay `1+0.25*max(0,z_s(proposed)-z_s(old))`. A legal `stay` costs `0.05` and also requires enough energy. Legal action cost is charged even if a collision later blocks motion. An illegal action closes the decision-legality gate; for deterministic downstream replay it becomes `stay` and spends `min(0.05,current_energy)`.

Use simultaneous proposed cells for three-rover collision resolution. First block every rover in a group with the same proposed cell and every rover in a pair that directly swaps old cells. Blocked rovers retain their old cells. Then repeatedly block any still-unblocked moving rover whose proposed cell equals the old cell retained by any blocked rover, until the blocked set no longer changes. All remaining rovers reach their proposed cells. Each blocked rover-slot adds one collision block. For every legal requested non-`stay` action, add `max(0,z_s(proposed)-z_s(old))^2` to terrain risk even if collision-blocked.

A goal-entry event occurs only when a rover's actual end cell is a goal and its old cell was not that goal; staying on a goal is not another entry. The first entry globally creates the goal's sole visit record. If several rovers first enter it in the same slot, `R1,R2,R3` order selects the `first_visitor` and every other simultaneous entry is a duplicate. Every later entry is also a duplicate and creates no record. At the next boundary each entering rover observes whether its entry created the first-visit record or was a duplicate. A first visit at any slot is retained, while visit value is on time only when `visit_slot<=d_s(j)`.

#### Slot-start radio model

Node positions at a slot start determine all six undirected pairs in this exact pair order: `L-R1,L-R2,L-R3,R1-R2,R1-R3,R2-R3`. For endpoints `u,v`, let horizontal distance be `d=||p_u-p_v||_2`. For `j=1,...,7`, form the fractional point `p_j=(1-j/8)*p_u+(j/8)*p_v`. Round each coordinate to its nearest integer, with an exact half rounded to the smaller integer. Let the endpoint-interpolated elevation at that sample be `ell_j=(1-j/8)*z_s(p_u)+(j/8)*z_s(p_v)`, and let `e=max(0,max_j(z_s(round(p_j))-ell_j))`. In `radio_dead_zone`, let `n_D` be the number of the seven rounded sample cells in `D`; otherwise set `n_D=0`.

The symmetric margin is `m_s(u,v)=8-0.45*d-1.25*e-0.6*n_D`. Both directed uses of the pair are available exactly when `m_s>=0`. Margins use slot-start positions, so a movement changes them from the next slot. The fixed simulator reveals to each node only its current incident margins. All fixed-model real comparisons use absolute tolerance `1e-6`.

#### Summaries, ACKs and FIFO custody

Every node has one FIFO of capacity eight. A first-visitor record is available at its rover from the slot after entry. At each nonterminal slot, an original visitor may request at most one new science summary containing one or two of its own first-visit records. Goal IDs must be distinct and strictly increasing; each must be locally unconfirmed and must not have entered a previously admitted science summary. A full local FIFO drops the request without creating a logical packet, so its records may be requested later. An admitted request creates the packet and permanently marks those records as used for summary creation. Received records may inform a rover but cannot be regrouped; relays forward the same logical packet.

A candidate summary packet ID is a nonempty scenario-local string, unique across all summary requests including dropped requests, and cannot start with reserved prefix `ack:`. A one-record summary is 96 bits and a two-record summary is 128 bits. Its origin is the requesting rover and final destination is `L`.

When a science summary first reaches `L` at boundary `t`, every goal it carries that is not already globally confirmed is confirmed at `t`, including late confirmation. For each newly confirmed goal, if `t+1<=149`, the simulator creates exactly one 64-bit ACK at slot `t+1`. Its fixed ID is `ack:<goal_id>`, origin is `L`, and final destination is that goal's first visitor. A boundary-150 confirmation creates no ACK. An ACK received by any rover adds its goal to that rover's locally known confirmed set; arrival at its final visitor terminates it. Losing or dropping an ACK never reverses the lander's confirmation.

A successful hop removes its packet from the source FIFO at slot end and makes it arrive at `dst` at the next boundary. A final destination consumes it. A nonfinal rover reads the payload first, adding science records or confirmation knowledge, and then admits the same packet if its FIFO has room; otherwise the packet terminates as dropped while the decoded knowledge remains. A packet's visited node path initially contains its origin and appends every successful-hop destination. A proposal may not revisit a node. Each packet permits at most three successful hops; after two successes it may be proposed only directly to its final destination.

A packet created at slot `a` may start attempts in slots `a,...,min(149,a+30)`. At a later slot boundary, a still-queued packet expires and is removed. A successful attempt in its last eligible slot is still processed at the next boundary. Queue expiration never removes an in-flight packet because service lasts exactly one slot. If such a last-eligible-slot arrival is nonfinal, the receiver still decodes it and admits it when capacity allows, but the packet cannot be proposed after slot `a+30` and is removed by the next slot's expiration step if it remains queued.

#### Boundary processing, proposals and information

For each slot `t=0,...,149`, use this exact order:

1. In node order `L,R1,R2,R3`, remove every queued packet with `t>created_slot+30`, preserving   the relative order of all remaining packets.
2. Process the sole successful-hop arrival, if any, from slot `t-1`, including payload knowledge,   final delivery, relay admission or relay overflow.
3. Enqueue ACKs due at `t` into `L` in increasing goal-ID order; an ACK is dropped if the FIFO is   full.
4. Compute current margins from slot-start positions and reveal each node's incident values.
5. In `R1,R2,R3` order, process each candidate summary request, admitting or dropping it by that   rover's current capacity.
6. Each rover chooses one movement action. Each of `L,R1,R2,R3` may submit at most one transmission   proposal from its now-current local FIFO.
7. If fewer than 120 attempts have previously been selected, arbitrate one proposal by greatest   integer priority and then ascending `(src,dst,packet_id)` Unicode code-point order. Other proposals   do nothing. Once 120 attempts have been selected, all proposals do nothing.
8. The selected attempt consumes one attempt whether it succeeds or fails. If its current margin is   nonnegative, remove the queue head and schedule arrival at boundary `t+1`; otherwise retain it at   the source head. Then execute simultaneous rover movement, energy, collision and goal-entry rules.

At boundary 150, process only a successful arrival from slot 149, including final confirmation or relay payload knowledge. The usual destination consumption, relay admission, or relay-overflow rule still applies to that arrival. Do not expire queues, generate ACKs, create summaries, propose service, move, or visit a goal. Then freeze the terminal state.

A transmission proposal has its submitting node as `src`, a different node as `dst`, the exact current FIFO-head packet ID, and priority `0..9`. Its `dst` must obey the packet's no-repeat and three-hop rules, and the packet must still be within its attempt window. A semantically invalid summary request is ignored and creates no packet; a semantically invalid transmission proposal is excluded from arbitration. Either defect closes the decision-legality gate. Structurally malformed records are handled by their evidence atom. A rover observes whether its step-5 summary request was admitted, dropped, or invalid before choosing its step-6 move and transmission proposal.

Each rover's local decision snapshot consists only of the static terrain and goals; its own position, energy and collision feedback; its own first-visit records and known visited/confirmed goal sets; its FIFO and packet contents; outcomes of its own earlier summary requests and transmission proposals; current incident margins; the globally shared remaining-attempt count; and summaries or ACKs just received. It cannot inspect another rover's position, energy, queue, records, proposal or unsent state. The lander process sees only its FIFO, globally confirmed goals, due ACKs, incident margins, remaining attempts and its own proposal feedback. The environment and fixed arbiter use global state only to execute the published transition and do not expose it to local policies. Current decisions cannot use simultaneous proposals or moves, future margins, future movements, future arrivals or future arbitration. The shared static terrain, goals, values, and deadlines stated above remain available to the lander as well as the rovers. A node may evaluate the published deterministic terrain and margin formula for its own hypothetical position relative to fixed `L`; this is planning from static inputs. It may not obtain an actual future incident margin whose other endpoint depends on another rover's unobserved current or future position.

Decision records represent candidate choices at steps 5-7. Checkpoints represent state after steps 1-4 at boundaries divisible by ten; boundary 150 uses its terminal rule. They are post-transition audit evidence, never policy input. The grader reconstructs all terrain, link, packet, queue, ACK, visit, confirmation and metric state from the three decision arrays, then compares checkpoint semantic values without using them to drive replay or metrics.

#### Metrics

For scenario `s`, total target value is `V=sum_j v_s(j)=60`. Let `V_confirm` be the value of goals with `confirm_slot<=c_s(j)` and `V_visit` the value of goals with `visit_slot<=d_s(j)`. `primary_metrics_1` is the five-scenario mean of `0.75*V_confirm/V+0.25*V_visit/V`.

Let `J_v` be the set of visited goals and `W_v=sum_(j in J_v) v_s(j)`. If `J_v` is empty, the next two quantities are zero. Otherwise define `complete_s=sum_(j in J_v) v_s(j)*I[confirm_slot_j<=c_s(j)]/W_v` and `fresh_s=sum_(j in J_v) v_s(j)*clip(1-(confirm_slot_j-visit_slot_j)/30,0,1)` `*I[confirm_slot_j<=c_s(j)]/W_v`. `primary_metrics_2` is the five-scenario mean of `0.6*complete_s+0.4*fresh_s`.

Let `terrain_risk` be the sum defined under movement, `duplicate` the number of duplicate entry events, and `visits=|J_v|+duplicate`. Let `energy_used` be the three initial budgets minus terminal energies and `progress_s=|J_v|/20`. `primary_metrics_3` is the five-scenario mean of `progress_s*(0.4*clip(1-terrain_risk/1800,0,1)` `+0.3*(1-duplicate/max(1,visits))+0.3*clip(1-energy_used/555,0,1))`; it is zero with no visit.

`selected_attempts` counts selected nonidle attempts, successful or failed. A successful summary hop is useful if either its logical packet later first confirms at least one goal on time at `L`, or its receiver is a rover that did not know at least one carried visit record before that arrival. Each successful hop counts at most once; ACK hops do not enter this numerator. `generated_summaries` counts admitted candidate summary requests, and `on_time_summaries` counts those whose first lander arrival newly confirms at least one carried goal on time. Define `auxiliary_metrics_1` as the five-scenario mean of `0.6*useful_hops/max(1,selected_attempts)+0.4*on_time_summaries/max(1,generated_summaries)`; the corresponding term is zero when its actual denominator is empty.

For `auxiliary_metrics_2`, use successful hops of each summary that newly confirms an on-time goal, plus successful hops of the ACKs generated for those goals. For every included hop, use its service margin and queue wait `service_slot-most_recent_queue_enter_slot`. Let `margin_s` be the mean `clip((margin-(-4))/(4-(-4)),0,1)`, `wait_s=clip(1-mean(queue_wait)/30,0,1)`, and let `age_s=clip(1-mean(confirm_slot-visit_slot)/30,0,1)` over the on-time goals. A component with no sample is zero. `auxiliary_metrics_2` is the five-scenario mean of `0.4*margin_s+0.3*wait_s+0.3*age_s`. All metric values are clipped to `[0,1]` and computed only from independent replay.

#### Result contract

`results.json` is a JSON object with exactly `scenarios`, an array containing the five scenario records in any order. `scenario` is the unique canonical key. Each record has exactly `scenario`, `moves`, `summary_requests`, `transmission_proposals`, and `replay_checkpoints`.

`moves` has exactly one row for every `(slot,rover)` in `{0,...,149} x {R1,R2,R3}`. A row has exactly `slot`, `rover`, and `action`, using the integer/string domains and action enumeration above. Row order is arbitrary.

`summary_requests` is a dynamic array. Every row has exactly `slot`, `source`, `packet_id`, and `goal_ids`. `slot` is an integer `0..149`, `source` is `R1`,`R2`, or `R3`, and at most one row may use the same `(slot,source)`. `packet_id` obeys the uniqueness and reserved-prefix rule above. `goal_ids` is an array of one or two distinct valid goal IDs in strictly increasing order. Absence of a node-slot row means no request. Array order is arbitrary.

`transmission_proposals` is a dynamic array. Every row has exactly `slot`, `src`, `dst`, `packet_id`, and `priority`. `slot` is an integer `0..149`; `src,dst` are different members of `{L,R1,R2,R3}`; at most one row may use the same `(slot,src)`; `packet_id` is a nonempty string; and `priority` is an integer `0..9`. Absence of a node-slot row means no proposal. Array order is arbitrary.

`replay_checkpoints` has exactly one row for every boundary in `{0,10,20,...,140,150}`, in any order. Each row has exactly `boundary`, `R1`, `R2`, `R3`, `queue_L`, `queue_R1`, `queue_R2`, `queue_R3`, `visited_goal_ids`, `confirmed_goal_ids`, `duplicate_visit_count`, `selected_attempt_count`, and `terrain_risk`. Each rover object has exactly integer `x` in `0..23`, integer `y` in `0..19`, finite `energy` in `[0,185]`, and arrays `known_visited_goal_ids`, `known_confirmed_goal_ids` containing distinct valid goal IDs; those set arrays may use any order. Each queue is an array of at most eight distinct nonempty packet IDs in semantic FIFO order. The global visited and confirmed arrays contain distinct valid goal IDs and may use any order. `duplicate_visit_count` is an integer `0..450`, `selected_attempt_count` is an integer `0..120`, and `terrain_risk` is finite in `[0,1800]`.

For a rover, `known_visited_goal_ids` is exactly the set of sole first-visit records that it created as `first_visitor` or decoded from an arriving summary; a duplicate physical entry alone does not add the goal. `known_confirmed_goal_ids` is exactly the set learned from ACKs that have arrived at that rover. The global arrays instead contain all goals with a first visit and all goals confirmed at the lander, respectively.

All scenario and record arrays are order-insensitive except strictly increasing summary goal IDs and FIFO packet order. The grader indexes them by the public keys. JSON numbers must be finite, duplicate JSON object keys are invalid, and booleans are not integers. Checkpoint real values use absolute tolerance `1e-6`; other checkpoint values are exact. The grader never compares candidate movement, grouping, routing or priority with a reference policy and does not require link, selected-attempt, packet-hop, per-slot queue, goal-event or metric summaries.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain the six non-empty Markdown sections `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The exact `results.json` contract above is reserved for the canonical evaluation. Controlled comparisons are optional for the validity gate but provide evidence for higher Judge levels. Record them in `report.md`, reproducible project code, or optional additional text/JSON project artifacts; any clear condition-level representation is accepted and no extra filename or schema is required.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "move, summary-request, proposal, and checkpoint fields have the declared forms", weight: 4}
- {id: canonical_coverage, desc: "every scenario, rover-slot move, and checkpoint boundary occurs exactly once", weight: 4}
- {id: decision_legality, desc: "movement, grouping, FIFO-head proposals, paths, priorities, and local timing are legal", weight: 4}
- {id: replay_validity, desc: "checkpoints match independently replayed motion, radio, queue, packet, ACK, and goal state", weight: 6}
- {id: primary_metrics_1, desc: "on-time confirmed and visited science value", weight: 8}
- {id: primary_metrics_2, desc: "visited-goal confirmation completeness and freshness", weight: 4}
- {id: primary_metrics_3, desc: "progress-conditioned terrain, duplicate-visit, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful successful hops and on-time summary efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "useful-path margin, queue wait, and confirmation age", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the terrain, local-information, store-and-forward, and science-confirmation formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent rover exploration and return model; Implementation/evidence: no executable movement or packet process."}
    - {score: 0.4, desc: "Report: terrain, shared-channel, target-confirmation, or information constraints are decisively incomplete; Implementation/evidence: state or event timing materially diverges."}
    - {score: 0.7, desc: "Report: values, terrain radio, FIFO custody, proposals, ACKs, duplicates, and deadlines are accurate; Implementation/evidence: all scenarios implement the core model with limited noncritical omissions."}
    - {score: 1.0, desc: "Report: allocation, movement and multihop return are precisely connected to science value and risk; Implementation/evidence: local information and event order are faithfully realized with traceable decision and checkpoint evidence."}
- id: method_design
  desc: "quality of the unified non-training target, path, summary, and forwarding policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited method is used; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: a fixed partition or distance-only rule omits decisive terrain, return, queue, or energy effects; Implementation/evidence: decisions are unit-specific or inconsistent."}
    - {score: 0.7, desc: "Report: a legal unified allocation, path and forwarding method treats the principal temporal constraints; Implementation/evidence: one procedure covers all units with limited horizon or trade-off handling."}
    - {score: 1.0, desc: "Report: value, terrain risk, changing return opportunities, queue age, duplicates and energy are jointly justified; Implementation/evidence: local evolving state drives consistent moves, grouping and proposals without scenario or target answer tables."}
- id: experimental_coverage
  desc: "coverage of canonical terrain searches and controlled return-path comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one incomplete condition is shown; Implementation/evidence: return-path claims lack condition evidence."}
    - {score: 0.7, desc: "Report: every canonical scenario and one controlled value, margin, energy, deadline, or rover-count comparison are covered; Implementation/evidence: comparable condition-level decisions and outcomes are retained."}
    - {score: 1.0, desc: "Report: complementary controlled experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check for terrain shadow, return capacity, deadlines, or allocation effects; Implementation/evidence: reproducible condition records support both experiments beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based analysis of terrain-aware allocation, packet return, and science value"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: claims are unsupported."}
    - {score: 0.4, desc: "Report: coverage or link counts are repeated without causal explanation or limitations; Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: conditional value, terrain, margin, queue, energy and deadline trends and limits are explained; Implementation/evidence: actual comparisons support the analysis."}
    - {score: 1.0, desc: "Report: allocation and return mechanisms are quantified and tested against contrasting evidence or a failure boundary; Implementation/evidence: reconstructible condition records support scoped conclusions."}
