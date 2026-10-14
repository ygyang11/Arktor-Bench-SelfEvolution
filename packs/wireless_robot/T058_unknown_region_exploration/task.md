---
id: T058_unknown_region_exploration
name: Unknown-Region-Guided Exploration with Communication-Limited Task Allocation
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

Four ground robots explore a partially known grid while exchanging compact regional summaries over a shared half-duplex broadcast channel. Each robot observes only a local neighborhood. A summary can carry the state of several exploration-information blocks (EIBs) and short-lived route commitments. Delayed or lost broadcasts make ownership stale, while narrow passages, doors and uneven traversability make a fixed spatial partition unreliable. The team must discover open cells before the mission horizon while avoiding repeated visits, collisions and communication waste.

### Research Problem

Build a deterministic, non-training, state-driven procedure that jointly chooses each robot's motion, the EIBs and commitments it proposes at allocation epochs, the EIB and commitment records it broadcasts, and which robot's FIFO is served by the shared channel. One reusable procedure must run over every canonical scenario. It should use unknown regions as well as nearby frontiers, adapt to stale summaries and lease conflicts, and balance new coverage, duplicate exploration, route stability, safety and communication cost.

### Experimental Setting and Constraints

#### Canonical units and environment

The workspace is a `24 x 20` integer grid with cells `x=0..23,y=0..19` and one-second slots `t=0..179`; state boundaries are `0..180`. Robots are `R0,R1,R2,R3`. Their starts, permanent blocked rectangles, doors and interference intervals are:

| scenario | starts `R0..R3` | permanent blocked cells | doors `(cell,open_slot)` | interference |
|---|---|---|---|
| `open_rooms` | `(1,1),(22,1),(1,18),(22,18)` | `(8,0..7),(8,10..19),(15,0..9),(15,12..19)` | `((8,8),30),((15,10),70)` | low |
| `narrow_tunnel` | `(1,9),(1,10),(22,9),(22,10)` | all cells outside `y=8..11` and four `4x4` side rooms are blocked | `((6,9),24),((12,10),62),((18,9),104)` | medium |
| `deep_unknown` | `(1,1),(4,1),(1,4),(4,4)` | `(10,0..14),(16,5..19)` | `((10,15),45),((16,4),95)` | low |
| `uneven_traversability` | `(1,2),(22,2),(1,17),(22,17)` | `(7x+5y) mod 19 = 0`, except the initial Chebyshev-radius-three neighborhoods | none | medium at `60..99` |
| `partition_burst` | `(2,2),(21,2),(2,17),(21,17)` | `(11,0..8),(11,11..19),(12,0..8),(12,11..19)` | `((11,9),50),((12,10),110)` | high at `35..74` and `120..144` |

For `narrow_tunnel`, the four side rooms are `[0,3]x[8,11]`, `[20,23]x[8,11]`, `[0,3]x[0,3]`, and `[20,23]x[16,19]`; all other cells are blocked. For `deep_unknown`, cells within Chebyshev distance three of any listed start are traversable at every slot even if covered by a listed rectangle; this is an environment exception and the corresponding states are initialized in that robot's local map before slot `0`. It is initial local knowledge, not a global-map reveal. Doors are blocked before their inclusive opening slot and traversable at and after that slot. These rules define environment truth; they do not reveal the global map to a policy.

#### State, observations, actions and EIBs

At the start of each slot, a robot observes every cell in Chebyshev radius `2` of its current real position and updates its local three-valued map (`unknown`, `traversable`, `blocked`). A first physical observation of a traversable cell is a team discovery; later observations are repeats. The EIB grid has `ex=0..5, ey=0..4`, ID `E<ex+6*ey>` written with two digits, and contains its `4x4` cells in row-major order. A local EIB version starts at zero and increases by one whenever any of its sixteen local states changes, whether the change comes from local observation or an accepted summary merge. The version belongs to that robot and is never copied from another robot. At an allocation epoch `0,10,...,170`, an EIB is `explorable` exactly when it contains an unknown cell adjacent (4-neighbor) to a known traversable cell connected to the robot through known traversable cells. It is `traversable_explored` when it has no unknown and at least one traversable cell, `blocked_explored` when it has no unknown and no traversable cell, and `merged` otherwise. `estimated_cost` is the shortest known-traversable path length from the robot to the nearest frontier cell in that EIB, or JSON `null` when no such path exists. `cost_q=min(255,estimated_cost)` is an integer; a commitment cannot use a null cost.

At every slot each robot requests one action from `stay,N,S,E,W`, with displacements `stay=(0,0), N=(0,+1), S=(0,-1), E=(+1,0), W=(-1,0)`. A target outside the grid or blocked in environment truth is replaced by `stay`. Two-way swaps are rejected; if several moves target one cell, the move with fewer currently unknown radius-two cells at its target wins, then smaller robot ID. Here "currently unknown" is computed by the fixed execution arbiter from the team discovery set immediately before motion; it is not an additional policy observation. A move into a cell occupied by a robot that remains still is then rejected, repeatedly until stable. Every requested non-stay move that does not execute is a safety event.

#### Allocation and commitment decisions

At each allocation epoch every robot submits up to three distinct EIB IDs in an ordered route. It may attach a commitment to any route EIB. A commitment is `{eib_id,lease,version,cost_q}` with `lease` in `1..20`; `version` and `cost_q` must equal that robot's current local EIB snapshot and `min(255,estimated_cost)`. It is active at slots `epoch <= t < epoch+lease` and expires at the start of slot `epoch+lease`; a commitment arriving after that boundary is ignored for ownership. A receiver merges commitments by choosing the unexpired claim with smallest `(cost_q,-version,robot_id)`; a changed owner is a commitment conflict resolution. Route and commitment choices use only the robot's current local map, history and received summaries.

#### Broadcast packets, channel and timing

Message opportunities are slots `0,5,...,175`. At an opportunity a robot may list up to five distinct current EIB IDs and up to five distinct EIB IDs with current commitments. The packet has a `32 bit` header plus `32 bit` per listed EIB or commitment record. An empty pair emits no packet. Each robot has a FIFO capacity of five packets; packets older than `30` slots expire. The shared channel serves at most one sender FIFO head per slot. Capacity is `192`, `96` or `64 bit` under low, medium or high interference. A served packet broadcasts independently to the other three robots. A receiver is unavailable when distance is greater than `10` or the straight grid segment crosses a blocked cell. For an available receiver, propagation delay is `1+floor(distance/6)`, capped at `3`, and completion succeeds when `((13*i+7*j+5*t) mod 100)/100 >= 0.05+0.12*interference_level`, where `i,j` are sender and receiver indices and the level is `0,1,2` for low, medium, high. A lost receiver remains pending and the packet retries from its full payload at a later service opportunity; a delivered or unavailable receiver is removed from the pending set. A packet leaves the FIFO after all receivers are removed. All served bits count toward a scene cap of `14400` bits. A successful arrival is processed at the start of its scheduled boundary only when its age is at most `30`. For EIB records, each receiver keeps a highest accepted sender version for every `(sender,EIB)`, initially `-1`; a record with a version no greater than that watermark is stale and ignored. A newer record advances only that sender watermark, then merges each cell into the receiver map: `blocked` dominates either other state, otherwise `traversable` dominates `unknown`. Every changed receiver cell advances the receiver's own local EIB version; the sender version is not assigned to it. A newer record that changes no receiver cell is accepted for freshness but is not useful payload.

At each slot the event order is: process arrivals and expire old packets/claims; perform local observation; submit allocation decisions at an allocation epoch; generate message packets at a message opportunity; submit the channel sender; serve one FIFO head and schedule receiver outcomes; filter and execute simultaneous motion; update discovery and allocation history. Boundary `180` processes only already scheduled arrivals and expirations; no new observation, packet or motion occurs.

The local actor sees the public static model, its own position and map, current EIB table, received records, own route/commitment history, own FIFO metadata and current interference/capacity probe. It does not see other robots' maps, positions, queues, future doors or future channel outcomes. The shared channel actor sees current sender positions only through the six public availability/capacity probes, queue heads and interference level; it cannot inspect packet contents. A unified implementation may coordinate these roles only through these interfaces. Model-based prediction from already visible public formulas is allowed, but later realized delivery, map, ownership, motion and coverage results may not be used as current input.

#### Record semantics and result contract

`results.json` has root exactly `{"scenarios":[...]}`. Each scenario object has exactly `scenario,slots`; each has one slot object for every integer `0..179`, with exact keys `slot,robot_decisions,allocations,messages,channel`. `robot_decisions` has one object for each robot, exactly `{robot_id,action}`. `allocations` is an array of four objects only at allocation epochs and an empty array otherwise; each object is exactly `{robot_id,route_eibs,commitments}`. `route_eibs` is an ordered list of at most three distinct valid EIB IDs. `commitments` is an unordered list of exact `{eib_id,lease,version,cost_q}` objects referencing that route. `messages` is an array of four exact objects only at message opportunities and empty otherwise; each is exactly `{robot_id,eib_ids,commitment_eibs}`. Both lists are unordered sets, contain distinct valid EIB IDs, and their combined length is at most five. The grader derives the current EIB codes, versions, commitment costs and packet bits from the sender's public local snapshot; candidates do not submit redundant codes, versions or packet IDs. `channel` is exactly `{sender}` with `R0..R3` or JSON `null`; a non-null sender must have a nonempty, unexpired FIFO head and positive current capacity.

Decision arrays are requests at the stated boundary. Real trajectories, local maps, EIB counts and versions, packet IDs, service outcomes, arrivals, ownership, queues, coverage and conflicts and mission summaries are post-decision evidence reconstructed by the grader and must not be submitted. All objects reject undeclared fields; integers exclude booleans, null is the only empty value, and finite floating-point comparisons use absolute tolerance `1e-6`. Container order is irrelevant except for `route_eibs`.

#### Metrics

All metrics are clipped to `[0,1]` per scenario and then averaged equally. `primary_metrics_1` is `0.7*final_coverage+0.3*coverage_auc`, where `C_t` is the number of distinct traversable cells physically observed by boundary `t`, `N_t` is the number of cells traversable at slot `t`, `coverage_auc=H^{-1} sum_{t=0}^{H-1} C_t/max(1,N_t)`, and final coverage uses boundary `180`. `primary_metrics_2=0.55*(1-overlap_ratio)+0.45*load_balance`, where `F_r` is robot `r`'s set of physically observed traversable cells, `overlap_ratio=(sum_r |F_r|/|union_r F_r|-1)/3`, and `load_balance=clip(1-sigma/sqrt(3),0,1)` for the population standard deviation of first-discovery shares; with no discoveries both terms are zero. `primary_metrics_3=0.4*exploration_yield+0.3*route_consistency+0.3*clip(1-safety_events/(4H),0,1)`, where yield is unique first discoveries divided by `max(1,executed_moves)`. Route consistency examines every adjacent allocation pair for each robot, including empty routes: a pair contributes one numerator point when both heads are present and equal, or when the prior head is no longer locally explorable at the later epoch; otherwise it contributes zero. The denominator is the number of adjacent epoch pairs, and is zero only when there is no pair.

`auxiliary_metrics_1=0.65*useful_eib_payload_ratio+0.35*useful_commitment_fraction`. An EIB record is useful when accepted at a receiver and changing at least one receiver cell state/version; a commitment record is useful when it changes the receiver's selected owner. Each useful record contributes `32 bit`, and denominators are served bits attributable to the corresponding record types. `auxiliary_metrics_2=0.45*eib_freshness+0.35*commitment_freshness+0.2*productive_channel_fraction`: freshness averages `clip(1-age/30,0,1)` over accepted records, and productive bits belong to packets containing at least one useful record. Empty samples contribute zero.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite root `results.json` from project code and fixed conditions. `report.md` must contain non-empty headings named exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. The report must include at least one executed controlled comparison of communication age/capacity, EIB/commitment selection or unknown-region routing, and one comparable failure-boundary or interference condition; condition-level evidence may be stored in candidate-defined UTF-8 JSON/CSV files and does not alter the canonical result contract. The report must state the policy inputs for each decision component and its compliance with the actor information boundaries.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact scenario and slot containers", weight: 3}
- {id: result_evidence, desc: "robot, allocation, commitment, message and channel fields are exactly typed", weight: 4}
- {id: canonical_coverage, desc: "every scenario, slot, robot and epoch/opportunity key appears exactly once", weight: 4}
- {id: decision_legality, desc: "actions, routes, commitments, message sets and channel sender requests are legal", weight: 4}
- {id: replay_validity, desc: "local snapshot, packet service, receiver arrival and expiration replay is consistent", weight: 6}
- {id: primary_metrics_1, desc: "final and time-integrated physical exploration coverage", weight: 8}
- {id: primary_metrics_2, desc: "duplicate exploration suppression and discovery load balance", weight: 4}
- {id: primary_metrics_3, desc: "exploration yield, route stability and motion safety", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful EIB and commitment payload efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "summary freshness and productive broadcast service", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the local exploration, EIB, lease and broadcast model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent local exploration or communication formulation; Implementation/evidence: no executable model exists."}
    - {score: 0.4, desc: "Report: EIB evolution, stale summaries, lease ownership or slot timing has a decisive omission; Implementation/evidence: state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: local maps, EIBs, commitments, broadcast, routes and coverage are accurately related; Implementation/evidence: the core model runs across all scenarios."}
    - {score: 1.0, desc: "Report: observation, information boundaries, packet timing, ownership conflicts and unknown-region consequences are precise; Implementation/evidence: code and condition records trace every central relation without material divergence."}
- id: method_design
  desc: "quality of the unified state-driven allocation and exploration method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal method or a prohibited method; Implementation/evidence: no reusable procedure supports it."}
    - {score: 0.4, desc: "Report: fixed partitions, nearest-frontier rules or unit-specific schedules dominate and ignore stale ownership or load; Implementation/evidence: decisions are not state-driven or differ materially from the report."}
    - {score: 0.7, desc: "Report: one reproducible method uses local state for routes, commitments, summaries, service and motion; Implementation/evidence: the same procedure runs across all units with limited long-horizon trade-offs."}
    - {score: 1.0, desc: "Report: a justified unified method jointly handles unknown reachability, age, quantized cost, lease conflicts, route stability, load and remaining time; Implementation/evidence: evolving allowed state drives choices without scenario or slot action tables."}
- id: experimental_coverage
  desc: "coverage of unknown-region guidance, communication and ownership mechanisms"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution exists."}
    - {score: 0.4, desc: "Report: only one canonical aggregate or an unexecuted condition claim; Implementation/evidence: records cannot test the central summary and allocation relation."}
    - {score: 0.7, desc: "Report: all scenarios and at least one executed comparable age, capacity, EIB, lease, load or unknown-region condition are covered; Implementation/evidence: condition records support the comparison."}
    - {score: 1.0, desc: "Report: complementary controlled conditions test unknown guidance, stale commitments and a communication or geometric failure boundary; Implementation/evidence: actual condition-level map, ownership, route and metric records support each claim."}
- id: research_analysis
  desc: "quality of evidence-based exploration and communication analysis"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: conclusions cannot be supported."}
    - {score: 0.4, desc: "Report: coverage or bit counts are listed without mechanisms or limits; Implementation/evidence: records do not support the assertions."}
    - {score: 0.7, desc: "Report: actual records explain EIB, age, ownership, load, obstacles and route trends while separating evidence, inference and limitations; Implementation/evidence: code supports the comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies unknown information, stale claims, load and failure boundaries, tests the explanation and limits scope; Implementation/evidence: reproducible condition records support the conclusions."}
