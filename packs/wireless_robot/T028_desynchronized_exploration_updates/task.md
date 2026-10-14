---
id: T028_desynchronized_exploration_updates
name: Desynchronised Communication Scheduling for Bandwidth-Limited Multi-Robot Exploration
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

Four mobile robots explore an initially unknown grid using separate local maps. A robot can share a compact pose and frontier summary over a range-limited broadcast channel, but only two transmissions can be served in a slot. Concurrent transmissions can fail during interference bursts. When updates are synchronised, several robots may act on equally stale summaries and revisit the same region; when updates are too sparse, claims and frontier locations become obsolete. The team must therefore coordinate exploration and transmission phases without a central map or future link knowledge.

A successful physical delivery arrives at the next state boundary. The receiver atomically replaces only an older sender summary, while duplicate or stale retries leave both the cache and its age unchanged. Locally observed cells expand after motion, and communication changes only the shared frontier candidates and cached poses, not the receiver's local obstacle map.

### Research Problem

Implement a self-contained, non-training method that jointly selects each robot's local grid move, frontier claim request, and event-triggered new or retry transmission. Use the robot's accumulated local map, current observation, arrived summaries, own history, and remaining budget. Coordinate transmission phase, message age, interference, expiring claims, high-value cells, travel, and repeated visits. Use one reusable state-driven procedure for every canonical scenario; do not provide scenario-, cell-, or slot-indexed action or transmission tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios on cells `(x,y) in {0,...,11}^2`. The table below is complete simulator and independent-replay truth; it is not a future-truth input to the decision procedure. A cell has `cell_id=12*y+x`. Closed integer rectangles include every cell in their inclusive coordinate ranges; all other cells form the explorable set `E_s`. Every explorable cell has value `1` except the listed high-value cells. Four robots of radius `0.25` start at the listed distinct explorable cells. The horizon has action slots `t=0,...,79`, one second per slot, and state boundaries `0,...,80`. Initial local maps and visits are formed at boundary `0`. The scenario deadline `D_s` is the last boundary included in task metrics.

| scenario | starts `(r0,r1,r2,r3)` | obstacle cells | high-value cells | radio radius | interference | `D_s` |
|---|---|---|---|---:|---|---:|
| `synchrony_tax` | `(1,1),(1,10),(10,1),(10,10)` | `[5,6]x[1,10]` | `(3,3):3,(8,8):3` | `6` | none | `80` |
| `range_limited` | `(1,1),(1,10),(10,1),(10,10)` | `[3,4]x[2,9]`, `[8,9]x[3,10]` | `(2,6):4,(10,6):4` | `4` | none | `80` |
| `clustered_frontiers` | `(2,2),(2,9),(9,2),(9,9)` | `[5,7]x[5,7]` | `(4,4):5,(4,5):5,(8,8):2` | `5` | none | `80` |
| `interference_burst` | `(1,6),(6,1),(6,10),(10,6)` | `[5,6]x[2,4]`, `[5,6]x[8,10]` | `(3,3):3,(9,9):3` | `5` | slots `18..31`: `0.8`; `52..59`: `0.6` | `80` |
| `heterogeneous_value` | `(1,1),(1,10),(10,1),(10,10)` | `[2,3]x[5,9]`, `[9,10]x[2,7]` | `(4,9):8,(8,3):2,(6,6):6` | `5` | none | `72` |

#### State, action, exploration, and claim model

At every boundary a robot adds all in-domain cells within Manhattan distance `2` of its current position, including their obstacle/free class, to its accumulated local map. Its local frontier is the sorted set of in-domain cells not yet in that local map that are Manhattan-adjacent to at least one locally observed free cell. This definition uses no unknown cell class; an unknown obstacle can remain a frontier until sensed. The shared frontier is the union of frontier IDs in currently usable sender caches, minus cells already in the receiver's local map, sorted and deduplicated.

In slot `t`, a robot chooses integer `(move_dx,move_dy)` from `{(0,0),(1,0),(-1,0),(0,1),(0,-1)}`. A nonzero move is legal only when its destination is an action-start locally observed free cell. A legal nonzero move adds one unit of travel. The four start cells count as unique visits at boundary `0`. Later, entering an unvisited free cell adds its fixed value to unique coverage; each nonzero move into an already visited cell adds one redundant visit. Staying adds neither. If several robots first enter the same cell in one slot, the lowest robot ID gets the unique visit and each other entry is redundant.

Each robot may also request one `claim_cell_id` from its current local or shared frontier, or `null`. At action start, claims whose expiry boundary has arrived and claims whose target was visited in the preceding slot are released. All requests in a slot are evaluated against one immutable snapshot of the claims that remain at action start. A valid request for a cell unclaimed in that snapshot wins; simultaneous requests for that cell are resolved by lowest robot ID. Only after all outcomes have been determined are winning replacements applied atomically. A winning request replaces that robot's previous claim after the slot, is available from boundary `t+1`, and has expiry boundary `t+6`, which may be as large as `85`. Requests for a cell claimed in the action-start snapshot lose and add one claim conflict. Any legal nonzero entry into a claim target, including a redundant entry, releases its claim at the following boundary. In the post-action transition, winning replacements are applied before visit effects: a target newly claimed and legally entered in the same slot is therefore closed at boundary `t+1` with reason `visited`, while an old claim replaced in that slot retains reason `replaced`. Visit-triggered release from slot `t` is applied before the expiry check at boundary `t+1`, so `visited` also wins when `t+1` equals that claim's `expiry_state`. Claims affect policy information and conflicts but do not reserve motion cells. At boundary `80`, after any slot-`79` visit-triggered releases, expiry releases are applied first and every other still-active claim is closed with reason `horizon`.

#### Information availability, messages, and event order

At boundary `0`, every ordered receiver-sender cache is absent with age `8`; each sender's logical version is `0`. From one boundary to the next, all ages first increase by one and clip at `8`. Physical deliveries arriving at the boundary are then processed by receiver-sender pair. A payload with a strictly larger version atomically replaces the cached sender pose, source slot, version, complete frontier set, and logical broadcast ID and sets age to `0`. An equal version is `duplicate`; a smaller version is `stale`. Neither changes the cache or age. A cache is usable exactly when present and its age is below `8`.

Each slot's order is: age increment and arrival fusion; sensing, local/shared frontier derivation and claim expiry; action-start snapshot recording and local move/claim/transmission decisions; action-start legality and simultaneous motion; atomic claim resolution, visits and safety update; transmission outcome registration for boundary `t+1`; then advancement to the next boundary.

The simulator holds the scenario name, full obstacle rectangles, all robots' true states, and the complete interference schedule so that it can produce observations and execute/replay actions. It must invoke the reusable decision procedure separately for each robot with only: robot ID and current slot `t`; grid size, radio radius, deadline, public cell values, and current interference `h_s,t`; that robot's current position and accumulated observed cells with their sensed free/obstacle classes; its local and shared frontiers; action-start positions of robots currently within sensing radius `2`; its active claim; its arrived per-sender caches and ages; its own prior actions and transmission requests; and its remaining budget. The policy interface must not receive the scenario name, full obstacle rectangles, unobserved cell classes, non-neighbour current robot states, packets not yet fused, future interference values or link outcomes, or pre-filled routes, claims, or transmission phases. The public scenario table may be used by the simulator and independent replay, but not to bypass this policy interface.

A new payload uses the current action-start pose and frontier and receives the sender's next version and next scenario-wide sequential logical `broadcast_id`. Within each scenario, `broadcast_id` starts at `0` and is assigned to new payloads consecutively by slot and then sender ID. The fixed unsigned big-endian bit layout is `sender(2)|source_slot(7)|version(7)|x(4)|y(4)|frontier_count(4)|frontier_id[12](8 each)|reserved(4)`. The sender ranks its current local frontier by descending public cell value then ascending `cell_id`, takes at most 12, and writes the selected IDs in ascending order. Unused entries are `255`, `frontier_count` is the number of real IDs, and `reserved=0`. The 128 bits serialize as 16 bytes and exactly 32 lowercase hexadecimal characters. Versions never wrap. A retry names one earlier payload created by the same sender and preserves its logical ID, source slot, version, pose, frontier set, and all bits.

Every non-`none` transmission request gets the next scenario-wide `attempt_id`, starting at `0` and assigned consecutively by slot and then sender ID; new and retry requests both consume one of that robot's 20 total attempts. At most two requests in a slot are legal. For deterministic replay of an illegal excess, the two lowest sender IDs are served and the rest are `capacity_lost`. Each attempt has three receiver diagnostics, one for every other robot. Let `d_ij,t` be action-start Manhattan distance, `c_t` the number of served attempts, and `h_s,t` the listed interference, zero outside listed intervals. A served attempt is physically delivered to a receiver at boundary `t+1` exactly when `d_ij,t<=r_s` and `h_s,t+0.2*(c_t-1)<=0.8`. Otherwise its status is, in priority order, `capacity_lost`, `out_of_range`, or `interference_lost`. A delivered row becomes `fused`, `duplicate`, or `stale` at arrival according to the version rule. Packets are not automatically retried.

#### Legal actions and boundary conditions

Moves, known-free destinations, claims, two-request capacity, per-robot 20-attempt budget, retry ownership, payload encoding, and information timing are hard constraints. An illegal move leaves the robot at its old cell and latches `safety_violation`. Two robots occupying the same cell at a post- action boundary also latch it; pairwise swaps are evaluated only at boundaries and are allowed. All 80 slots are recorded even after the deadline or a safety violation. At terminal boundary `80`, all action fields are zero/`null`/`none`; maps, caches, positions, budgets, and cumulative evidence retain their terminal values, and `active_claim_cell_id` is `null` after the terminal claim closures.

#### Task metrics

For scenario `s`, use the independently replayed coverage record at boundary `D_s`. `primary_metrics_1` is visited unique value divided by the total value of `E_s`, clipped to `[0,1]`, then averaged over scenarios. `primary_metrics_2` is the scenario mean of `0.6*clip(1-total_travel/(4*D_s),0,1)+0.4*clip(1-redundant_visits/(4*D_s),0,1)`.

For each boundary `t=1,...,D_s`, define the team peak age as the maximum over six unordered pairs `{i,j}` of `(age_(i<-j,t)+age_(j<-i,t))/2`; absent and unusable entries retain clipped age `8`. Let `P_s` be the mean of these `D_s` boundary peaks. `auxiliary_metrics_1` is the scenario mean of `clip(1-P_s/8,0,1)`. `auxiliary_metrics_2` is the mean `clip(1-transmission_attempts_before_D_s/80,0,1)`, where new and retry attempts both count and 80 is the four-robot aggregate budget. The normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result record contract

`results.json` is strict JSON with only `scenarios`, one record per canonical name. Each scenario record has exactly `scenario,robots,belief,frontiers,messages,channels,coverage`.

`robots` has 324 records ordered by `(robot,state_t)`, with exactly `robot,state_t,x,y,observed_cell_ids,local_frontier_cell_ids,shared_frontier_cell_ids,nearby_robots,active_claim_cell_id,remaining_broadcast_budget,move_dx,move_dy,claim_cell_id,tx_kind,retry_broadcast_id,emitted_broadcast_id,safe,deadline_state`. For `state_t<80`, all state fields in the row are the action-start boundary snapshot after arrival fusion, sensing, and claim expiry, while the action fields are that robot's decisions for slot `state_t`. Thus `x,y`, all three frontier/map lists, `nearby_robots`, `active_claim_cell_id`, `remaining_broadcast_budget`, and `safe` do not include effects of that row's action. `safe` is the negation of the scenario-wide latched safety violation at that boundary. `deadline_state` is the scenario's integer deadline `D_s` in every row; it is not a Boolean deadline indicator. Cell-ID lists are sorted unique integers in `0..143`; `nearby_robots` is sorted by robot and each item has exactly `robot,x,y` for another robot within sensing distance `2` at action start. Nullable claim/broadcast IDs are nonnegative integers when present. `tx_kind` is `none`, `new`, or `retry`; only retry has a non-null `retry_broadcast_id`. `emitted_broadcast_id` is the new logical ID for `new`, the referenced logical ID for `retry`, and `null` for `none`. The boundary-`80` row is the terminal snapshot described above.

`belief` has 972 records ordered by `(receiver,sender,state_t)`, with exactly `receiver,sender,state_t,summary_version,source_slot,cached_x,cached_y,frontier_cell_ids,age,available,last_broadcast_id`. For an absent cache the nullable fields are `null`, frontier IDs are empty, age is `8`, and available is false. A present cache retains its finite integer fields even at age `8`; available is exactly `age<8`.

`frontiers` is dynamic and ordered by `claim_sequence`, with one record per non-null request. Within each scenario, `claim_sequence` starts at `0` and is assigned consecutively by request slot and then robot ID. Each record has exactly `claim_sequence,robot,cell_id,request_slot,outcome,owner_robot,expiry_state,release_state,release_reason,visited_state`. Outcome is `won`, `conflict`, or `invalid`. Only a winning row has a later release state and reason in `visited,expired,replaced,horizon`; conflict rows name the action-start winner and expiry, while invalid rows have null owner/expiry. `expiry_state` is an integer in `0..85` when present; `release_state` and `visited_state` are integer boundaries in `0..80` when present.

`messages` is dynamic and ordered by `(attempt_id,receiver)`, with three rows per attempt and exactly `attempt_id,broadcast_id,receiver,sender,transmit_slot,tx_kind,arrival_state,receiver_distance,in_range,interference,concurrency,payload_hex,decoded,delivered,fusion_status`. `tx_kind` is `new` or `retry`; arrival is `transmit_slot+1`; distance, interference and concurrency are action-start values. `decoded` has exactly `sender,source_slot,version,x,y,frontier_count,frontier_cell_ids,reserved` and must match the 32-digit payload. Status is `capacity_lost,out_of_range,interference_lost,fused,duplicate`, or `stale`; `delivered` is true exactly for the final three fusion statuses.

`channels` has 80 records ordered by `t`, with exactly `t,interference,requested_sender_ids,served_sender_ids,attempt_ids,served_attempt_ids,concurrency,capacity_ok`. Sender lists are in ascending sender ID. Attempt lists follow that same sender order and therefore contain increasing consecutive IDs. Concurrency is the served count, and capacity is true exactly when at most two requests were made.

`coverage` has 81 records ordered by `state_t`, with exactly `state_t,newly_visited_cell_ids,cumulative_visited_cell_ids,unique_value,redundant_visits,claim_conflicts,total_travel,safety_violation,deadline_state`. At boundary `0`, `newly_visited_cell_ids` contains the four starts; at every later boundary it contains the cells first entered in the immediately preceding slot. Lists are sorted unique IDs; counts and value are nonnegative integers and cumulative. `deadline_state` is the scenario's integer `D_s` in every coverage row, not a Boolean. All numeric values in the file are finite; booleans are JSON booleans. Numeric comparisons use absolute tolerance `1e-6`.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root directory must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must overwrite `results.json`. `report.md` must have non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

Controlled comparisons used as Judge evidence must be executed by project code and reported with condition-level metrics in `report.md`. They are supported by the canonical records, implementation, and report and must not be inserted as non-canonical scenario records in `results.json`; no additional artifact is required.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact scenario-level structures", weight: 3}
- {id: result_evidence, desc: "robot, belief, claim, message, channel and coverage evidence", weight: 4}
- {id: canonical_coverage, desc: "all fixed scenario, boundary, robot and ordered-pair keys", weight: 4}
- {id: decision_legality, desc: "move, claim, safety, capacity, budget, retry and terminal constraints", weight: 4}
- {id: replay_validity, desc: "independent sensing, motion, claim, cache, channel and coverage replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed value-weighted unique coverage at the deadline", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed travel and repeated-visit efficiency", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed mean team peak-age freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed transmission-attempt efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of local exploration, claims, communication age, interference and safety model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent exploration model; Implementation/evidence: no executable local-map, belief, or communication process."}
    - {score: 0.4, desc: "Report: omits a decisive sensing, age, interference, claim, or visit relation; Implementation/evidence: only a partial or materially divergent process is present."}
    - {score: 0.7, desc: "Report: accurately defines the core local maps, frontiers, channel, caches, claims, motion, and objectives; Implementation/evidence: all canonical units implement them with only limited omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order and how transmission phase and peak age cause frontier conflict and repeated travel; Implementation/evidence: records faithfully realize every central relation."}
- id: method_design
  desc: "quality of the reusable local exploration and transmission scheduler"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training; Implementation/evidence: no reusable state-driven procedure."}
    - {score: 0.4, desc: "Report: uses a fixed synchronised schedule, scenario route, or partial rule that ignores decisive age, interference, or claim effects; Implementation/evidence: decisions are pre-filled or materially diverge."}
    - {score: 0.7, desc: "Report: gives a legal unified method for local motion, claims, age-aware sending, and budgets; Implementation/evidence: one procedure covers all scenarios with limited predictive or value reasoning."}
    - {score: 1.0, desc: "Report: justifies a reproducible method that jointly handles marginal frontier value, peak age, range/interference, claim expiry, motion safety, and remaining budget; Implementation/evidence: local evolving state drives all decisions without scenario tables."}
- id: experimental_coverage
  desc: "coverage and controlled phase/interference/exploration evidence"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only canonical aggregates or one schedule are discussed; Implementation/evidence: no controlled evidence tests the communication-exploration relation."}
    - {score: 0.7, desc: "Report: all canonical scenarios plus one controlled phase, range, interference, frontier-value, or budget comparison with condition-level metrics; Implementation/evidence: project code executes both conditions, and canonical records plus reported outcomes support the comparison."}
    - {score: 1.0, desc: "Report: at least two complementary single-factor experiments include a mechanism comparison and a sensitivity or failure boundary; Implementation/evidence: project code executes the comparisons, and canonical records plus condition-level age, claim, coverage, and revisit outcomes support the claims."}
- id: research_analysis
  desc: "evidence-grounded interpretation of age, phase, conflicts, coverage and limitations"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: conclusions cannot be supported."}
    - {score: 0.4, desc: "Report: repeats coverage or message counts without causal age/phase/conflict reasoning or material limitations; Implementation/evidence: central claims lack recorded support."}
    - {score: 0.7, desc: "Report: actual records support conditional trends among phase, interference, age, claims, travel, and coverage, with observation separated from inference and limitations; Implementation/evidence: supports those comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies synchrony tax and tests its explanation against range, burst, value, or budget boundaries with scope and unknowns; Implementation/evidence: reproducible condition-level comparisons support the conclusions."}
