---
id: T029_aerial_map_support
name: Communication-Aware Aerial Map Support for Multi-UGV Goal Navigation
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

A supporter UAV assists three ground UGVs that navigate to separate goals through an initially unknown grid. Each UGV has only a small local sensor and plans with an optimistic belief about cells that it has not confirmed. The UAV has a larger sensor and can send single-cell occupancy and cost updates over a shared, capacity-limited downlink. UGVs periodically request map information along their current belief paths. A request or map packet can be delayed or lost, and packets compete for the same slot capacity. A useful update can remove an unnecessary detour or reveal a blocked route, while a late or redundant update consumes the map-attempt budget without helping navigation.

The UAV must choose where to fly, which requested cells to place in its per-UGV queues, and which queued protocol units to submit. Each UGV must choose a legal local move and when and what request to generate. Successful requests and map packets take effect only at their arrival boundary. The team is evaluated on goal progress, deadlines, safe motion, path cost, information freshness, and communication load under the same deterministic environments.

### Research Problem

Implement a self-contained, non-training distributed policy for the supporter UAV and the three UGVs. The policy should use only each agent's local observations, arrived messages, current request or map queues, information ages, action history, and remaining budgets. It must jointly handle UAV exploration, UGV belief-space navigation, request timing and content, map-cell prioritization, queue contention, retries, and safety. Study how supporter position, map value, message freshness, shared capacity, and heterogeneous deadlines trade off against travel and successful goal arrival. Use one reusable state-driven procedure for all canonical scenarios; do not provide scenario-, cell-, robot-, request-, or absolute-slot-indexed action tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios on grid cells `(x,y) in {0,...,15}^2`, with `cell_id=16*y+x`. Each scenario has one supporter UAV (ID `3`) and three UGVs (IDs `0,1,2`). Action slots are `t=0,...,71`, each of length `1` second; state boundaries are `state_t=0,...,72`, and slot `t` updates boundary `t` to boundary `t+1`. The fixed scenario deadline for each UGV is `D_r`; the mission horizon is `H=72`. Rectangles are closed and inclusive. Every non-obstacle cell has movement cost `1`; this task has no additional high-cost cells. The table is simulator and independent-replay truth, not future-truth input to any policy.

All published start and goal cells are guaranteed to be free. If a closed obstacle rectangle covers one of those cells, that start/goal cell takes precedence and is removed from the obstacle set. No rectangle in the current canonical table covers a published start or goal; this precedence rule defines the interpretation defensively and is part of the independent-replay truth.

| scenario | UAV start | UGV start -> goal (deadline) | obstacle rectangles | radio radius |
|---|---|---|---|---:|
| `forked_corridors` | `(8,8)` | `(2,2)->(5,13) (48)`; `(2,13)->(13,13) (60)`; `(13,2)->(13,8) (56)` | `[7,8]x[1,10]`, `[7,8]x[12,15]` | `6` |
| `shared_bottleneck` | `(8,8)` | `(2,2)->(14,2) (58)`; `(2,14)->(14,14) (62)`; `(14,8)->(8,8) (50)` | `[6,10]x[7,7]`, `[6,10]x[9,9]` | `5` |
| `supporter_detour` | `(2,14)` | `(2,2)->(14,4) (64)`; `(14,2)->(4,14) (66)`; `(14,14)->(8,8) (54)` | `[4,5]x[4,12]`, `[11,12]x[5,13]` | `6` |
| `late_hazard` | `(14,8)` | `(2,2)->(5,5) (40)`; `(2,14)->(12,12) (58)`; `(14,2)->(9,6) (52)` | `[5,6]x[6,10]`, `[10,11]x[2,5]` | `5` |
| `asymmetric_deadlines` | `(8,2)` | `(2,2)->(4,14) (36)`; `(14,2)->(12,12) (54)`; `(2,14)->(14,14) (70)` | `[7,9]x[5,6]` | `6` |

#### State, action, local maps, requests, queues, and packet model

At every boundary, a UGV senses every in-domain cell with Manhattan distance at most `2` from its current position and records its true class (`free` or `obstacle`) in its local map. The UAV senses every in-domain cell with Chebyshev distance at most `4` from its position and records the true class and cost. A UGV may plan through unknown cells as free cells of cost `1`; a confirmed obstacle is impassable. A request path is an ordered list of at most six cell IDs taken from the UGV's current belief path. It starts at the UGV's current cell, uses four-neighbour steps, and is the next at-most-six-cell prefix of that path; it ends at the goal only when the goal falls inside the prefix (otherwise the last cell is a local waypoint). Unknown cells may occur in the path. The request uncertainty is a finite `u` in `[0,1]`, chosen by the candidate and recorded with the path. A candidate may omit a request at an opportunity.

UGV moves are `move_dx,move_dy` in `{(0,0),(1,0),(-1,0),(0,1),(0,-1)}`. A move must be in-domain and one four-neighbour step. A move into a currently known obstacle is an illegal decision. A move into an unknown true obstacle is a legal optimistic decision but is changed by the safety executor to a stay, increments `blocked_moves`, and records a blocked event. A UGV that has reached its goal stays there. The UAV moves by `move_dx,move_dy` in `{-1,0,1}^2`; it may fly over ground obstacles but must remain in-domain. A nonzero UGV move costs `1`; a nonzero UAV move costs `1.5`. A goal is reached at the first boundary at which the UGV position equals its goal, and remains reached.

The minimum Euclidean separation is `1` between two UGVs and `0.5` between the UAV and any UGV. The fixed safety executor processes proposed UGV moves in increasing robot ID, accepting a move only if the resulting position remains separated from all already accepted positions and the current UAV position; a rejected separation move becomes a stay and increments `separation_interventions`. The UAV move is then accepted only if it remains separated from all UGVs, otherwise it becomes a stay and increments the same counter. Pairwise UGV swaps are not accepted. A true obstacle or separation event is recorded in `mission_trace`; it does not silently change the proposed action.

At each request opportunity `t in {0,6,...,66}`, every UGV not yet at its goal may submit at most one request. A generated request object has exactly `request_id`, `request_version`, `path_cell_ids`, and `uncertainty`; its `request_id` must equal the next dense scenario-local ID assigned in opportunity then UGV-ID order (omitted opportunities do not consume an ID), and its `request_version` is strictly increasing per UGV. A request path is an ordered list of at most six cell IDs taken from the UGV's current belief path. It starts at the UGV's current cell, uses four-neighbour steps, and is the next at-most-six-cell prefix of that path; it ends at the goal only when the goal falls inside the prefix (otherwise the last cell is a local waypoint). This bounded-prefix rule is necessary because several published start-to-goal distances exceed five moves. Unknown cells may occur in the path. A request enters that UGV's FIFO request queue with capacity `2`; if full it is dropped and the opportunity records `queue_full`. A request generated in slot `g` is eligible for admission in slots `g,...,g+7`; at the start of boundary `g+8` an unsent queue entry expires. A request attempt that arrives at boundary `g+8` or later is recorded as `expired` and is not exposed to the UAV. A successful arrival before that boundary is recorded as `delivered` and is appended to the UAV's arrived-request set. If a successful attempt's scheduled arrival is beyond boundary `72` but still before its expiry boundary, it remains `sent` with that out-of-horizon `arrival_state` and has no effect. When a newer version from the same UGV is delivered, older delivered requests from that UGV are removed from the arrived-request set; their recorded status remains `delivered` and they cannot be submitted again. Each UGV has a request-generation budget of `12`, initially; a generated request consumes one unit even when its outgoing queue is full, while an omitted opportunity does not consume budget. `remaining_request_budget` reports this budget. A request protocol unit is addressed from the UGV to the UAV and consumes one shared-channel unit when admitted.

The UAV has one FIFO map-packet queue of capacity `6` for each UGV. A map packet contains exactly one cell ID, its true class, its fixed cost, and the latest positive UAV map version for that cell. The UAV may enqueue only a cell it has already sensed, using the latest positive version for that cell. A new logical map packet has a scenario-local nonnegative `packet_id` and attempt `0`; its `retry_of` is null. A failed attempt is removed from its queue and can be retried only by explicitly creating a new attempt with the same `packet_id`, the same receiver and cell payload (`cell_id`, `cell_state`, `cell_cost`, and `cell_version`), a larger attempt number, and integer `retry_of` equal to the immediately preceding failed attempt number. Each `(packet_id,attempt)` pair is unique. Each slot may enqueue at most three map attempts. Each attempt consumes one from the scenario-wide map-attempt budget `120` when it is enqueued. A packet queue entry is removed when its attempt is admitted, whether delivery succeeds or fails. A successful map packet strictly newer than the receiver's cell version replaces that cell's belief; equal or older versions are delivered but not fused. A successful attempt whose arrival is beyond boundary `72` remains a delivered pending event in the record but has no later fusion.

#### Information availability, channel, and event order

The policy interfaces are isolated. A UGV receives only its ID, current slot, grid and goal, public deadline and costs, its position, accumulated classified local map, current belief path, arrived map packets and ages, its request queue, nearby UGV positions within Manhattan distance `2`, its own history, and remaining budgets. It does not receive the scenario name, full obstacles, unobserved classes, UAV state, other UGV maps, future link outcomes, or future interference. The UAV receives only its position and classified sensed map, arrived unexpired requests, its three map queues, the current interference value, current slot, action history, and remaining map-attempt budget. It does not receive unobserved map truth, future link samples/delays, or undelivered requests.

At boundary `0`, queues are empty, all receiver map versions are `0`, and all information ages are `8`. UAV sensing at boundary `0` creates positive per-cell UAV versions. Versions are independent per cell: the first UAV observation of a cell is version `1`, and a static truth cell remains at version `1` on later sensing; UGV local sensing creates local observations but does not create receiver map versions. `observed_cells.version` is the receiver's current evidence version (`0` for local-only evidence and the fused UAV version for a cell learned from a map packet); `map_versions` and `ages` contain exactly the receiver cells learned from UAV packets. A successful fusion sets that cell's age to `0`; each later boundary without a newer fusion increments its age, capped at `8`. Each slot executes exactly this order: (1) process packets arriving at the boundary, expose only requests whose arrival is before their expiry boundary, and discard expired requests; (2) update local maps, beliefs, versions, and ages; (3) at a request opportunity, create any submitted UGV requests and enqueue them; (4) obtain action decisions and packet-enqueue decisions from the local interfaces; (5) read the candidate's ordered `submitted_keys` for the shared channel and admit only legal queue-head units until four units are admitted; (6) schedule a deterministic delivery for each admitted unit; (7) execute the UAV and UGV motion/safety executor and perform the next sensing; (8) update goal, costs, counters, budgets, and record the next-boundary state. State rows are action-start snapshots after steps (1)--(3) and before that slot's action; mission rows are post-transition boundary records. A delivered request uses `request_packet_id=request_id`. In `channel_slots`, both queue-size arrays are action-start snapshots after request generation and packet enqueue decisions but before any submitted key is admitted; `admitted_keys` and `units` describe the subsequent service.

`belief_path_unknown` and `belief_path_cost` have two equivalent encodings. The expanded encoding uses a list with one entry per `belief_path_cell_ids` cell: each unknown entry is a boolean and each cost entry is that cell's optimistic traversal cost (the start-cell entry is retained for alignment). The compact encoding uses one boolean for whether any path cell is unknown and one finite number for the total optimistic movement cost along the path (the sum of entries after the start cell, or `0` for a path of length `0` or `1`). A result may use either encoding, and the grader compares the corresponding semantics rather than the JSON representation.

In each `channel_slots` row, `request_queue_sizes` and `map_queue_sizes` may independently use either the compact list `[size0,size1,size2]` or the explicit list `[{'robot':0,'size':size0},{'robot':1,'size':size1},{'robot':2,'size':size2}]`. The explicit entries are keyed by `robot` and may appear in any order. Both forms encode the same queue sizes; the grader canonicalizes them before checking the action-start snapshot and does not penalize a representation choice.

For a request or map attempt sent by sender at slot `t` to receiver, Manhattan distance is evaluated against the scenario radio radius. An admitted unit outside that radius is recorded and fails with `out_of_range`; candidates may prefilter such units but are not required to do so. The interference is `h_{s,t}=min(1,b_s+g_s(t))`, where `b_s=(0.10,0.20,0.15,0.25,0.20)` in the scenario table order; `g_s(t)` is respectively `0.35` on `24..29`, `0.30` on `30..39`, `0.25` on `18..27`, `0.40` on `36..47`, or `0.30` on `12..23`, and zero otherwise. For the accepted service order `m=0..n_t-1`, `u_{s,t,m}=((37(t+1)+101(m+1)+c_s) mod 997)/996`, and `delta_{s,t,m}=1+I(((53(t+1)+89(m+1)+c_s) mod 10)<d_s)`, with `(c_s)=(11,23,41,59,73)` and `(d_s)=(1,2,2,3,2)`. An admitted unit succeeds exactly when `h_{s,t}+0.1*n_t<=0.9` and `u_{s,t,m}<=0.85-0.05*h_{s,t}`. A successful unit arrives at `t+delta`; arrival beyond boundary `72` is not effective. A failed attempt is never retried unless the candidate explicitly creates a later attempt.

#### Legal actions and boundaries

All moves, request paths, request versions, queue heads, map-cell ownership, channel capacity, packet attempts, budgets, distance checks, information restrictions, safety-executor outcomes, and terminal states are hard constraints. Request opportunities, channel slots, and all state/action arrays are recorded even after a UGV reaches its goal. At boundary `72`, no new action is executed; terminal state rows preserve the final positions, maps, queues, versions, ages, goals, budgets, and safety flags. The `t=71` action is an ordinary executable slot; there is no synthetic terminal action row. A packet arriving at boundary `72` is recorded and fused at that terminal boundary, but is not used in any later decision.

#### Task metrics

For each UGV, if it first reaches its goal at `arrival_state<=D_r`, its completion score is `0.7+0.3*(1-arrival_state/D_r)`. If it first reaches after its deadline but by `H=72`, the score is `0.4*(H-arrival_state)/max(1,H-D_r)`. Otherwise it is `0`. `primary_metrics_1` is the mean over all fifteen UGV-scenario units. At boundary `D_r`, let `d_true` denote the true Manhattan distance, `progress=clip(1-d_true(final_position,goal)/d_true(start,goal),0,1)`, and `safety=clip(1-(blocked_moves+separation_interventions)/H,0,1)`; `primary_metrics_2` is the scenario mean of `0.7*mean(progress*safety)+0.3*min(progress*safety)` then averaged over scenarios. For each UGV that reaches by its deadline, let `C*_true` be its shortest path cost under fixed obstacles and `C_actual` its independently replayed ground movement cost; its path score is `clip(C*_true/max(C_actual,1e-6),0,1)`, otherwise `0`. `primary_metrics_3` is the mean of per-scenario UGV path scores.

For each UGV-slot whose current belief path contains an unconfirmed cell, freshness is `clip(1-age/8,0,1)` using the latest fusion that changed a path cell; no such fusion contributes `0`, and paths without an unconfirmed cell are omitted. `auxiliary_metrics_1` averages these values by UGV then scenario, using `0.5` for a scenario with no samples. For each scenario, `auxiliary_metrics_2` is `0.5*clip(1-admitted_map_attempts/120,0,1)+0.5*useful_delivered_cells/max(1,admitted_map_attempts)`, where a useful delivery strictly changes the receiver belief and confirms a cell that was unknown on its pre-delivery path; repeated receiver-cell versions count at most once. The normalized task performance is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. All metrics are finite and clipped to `[0,1]` and come from independent replay.

#### Result record contract

`results.json` is strict JSON with only a `scenarios` array containing the five scenario records in the table order. Each record has exactly `scenario`, `supporter_trace`, `seeker_traces`, `request_opportunities`, `map_packets`, `channel_slots`, and `mission_trace`.

`supporter_trace` is an object with `states` (73 rows, ordered by `state_t=0..72`) and `actions` (72 rows, ordered by `t=0..71`). State rows have exactly `state_t,x,y,observed_cell_ids,map_versions,map_queue_sizes,remaining_map_attempts,safe`; action rows have exactly `t,move_dx,move_dy,enqueue_packets`, where each enqueue item has `receiver,packet_id,attempt,cell_id,cell_state,cell_cost,cell_version,retry_of`. `observed_cell_ids` is sorted; each `map_versions` entry is `{cell_id,version}` and entries are sorted by `cell_id`; `map_queue_sizes` is `{robot,size}` for robots `0,1,2`; `safe` is the latched UAV safety flag at the action-start boundary.

Each `seeker_traces` item has `robot`, `states`, and `actions`. States have 73 rows exactly with `state_t,x,y,observed_cells,belief_path_cell_ids,belief_path_unknown,belief_path_cost,request_queue,request_versions,map_versions,ages,goal_reached,safe,remaining_request_budget`; `observed_cells` is a sorted list of `{cell_id,state,cost,version}`, `request_queue` is a sorted FIFO list of request IDs, `request_versions` is the current integer version for that UGV, `map_versions` entries are `{cell_id,version}` sorted by `cell_id`, `ages` entries are `{cell_id,age}` sorted by `cell_id`, and all ages are integer `0..8`. `belief_path_unknown` is either one boolean per path cell or one boolean equal to their logical OR. `belief_path_cost` is either one finite optimistic cost per path cell or one finite total equal to the sum after the aligned start-cell entry. Actions have 72 rows exactly with `t,move_dx,move_dy,request`, where `request` is `null` or an object with `request_id,request_version,path_cell_ids,uncertainty`; actions cover executable slots `t=0..71`, including slot `71`, and there is no synthetic terminal action at boundary `72`.

`request_opportunities` has exactly 36 rows per scenario, ordered by `(opportunity_t,robot)`, with exactly `robot,opportunity_t,active,request_id,request_version,path_cell_ids,uncertainty,queue_before,queue_after,generated,request_packet_id,send_slot,arrival_state,status`. When no request is generated, request-specific fields (`request_id`, `request_version`, `path_cell_ids`, `uncertainty`, `request_packet_id`, `send_slot`, and `arrival_state`) are null; `queue_before` and `queue_after` may be null or may report their integer action-start and post-generation snapshots. When a request is generated, the queue snapshots are integers in `0..2`. Status is `none`, `queued`, `queue_full`, `sent`, `delivered`, `lost`, or `expired`; request IDs and versions are scenario-local nonnegative integers.

`map_packets` is dynamic and ordered by `(packet_id,attempt)`. Each row has exactly `packet_id,attempt,sender,receiver,cell_id,cell_state,cell_cost,cell_version,enqueue_slot,send_slot,arrival_state,retry_of,receiver_distance,interference,concurrency,delivered,fusion_status`. A packet attempt is one channel unit; before admission, `send_slot`, `arrival_state`, `receiver_distance`, `interference`, and `concurrency` are null. After admission, `send_slot`, `receiver_distance`, `interference`, and `concurrency` are the service slot, action-start Manhattan distance, slot interference, and admitted-unit count; `arrival_state` is the scheduled boundary for a successful attempt and remains null for an admitted attempt that fails. Fusion status is `pending`, `out_of_range`, `interference_lost`, `fused`, `duplicate`, or `stale`.

`channel_slots` has exactly 72 rows ordered by `t`, each with `t,interference,request_queue_sizes,map_queue_sizes,submitted_keys,admitted_keys,units,capacity_remaining`. Each submitted/admitted key has `kind` (`request` or `map`), and request keys additionally have exactly `robot,opportunity_t`; map keys have `packet_id,attempt` and may also include the redundant `receiver`. If present, `receiver` must be the integer ID of the packet's actual destination queue; the forms with and without it are semantically identical. `units` is the admitted count and `capacity_remaining=4-units`. Each queue-size field is either the compact integer list `[size0,size1,size2]` or the explicit object list `[{'robot':0,'size':size0},{'robot':1,'size':size1},{'robot':2,'size':size2}]`; these forms are semantically identical and are canonicalized by the grader, and the explicitly keyed objects may appear in any order.

`mission_trace` has 73 rows ordered by `state_t`, each with exactly `state_t,goal_reached,arrival_states,ground_move_cost,uav_move_cost,blocked_moves,separation_interventions,redundant_detours,remaining_map_attempts,safety_violation,terminal`. `redundant_detours` is the cumulative count of executed nonzero UGV moves whose true Manhattan distance to that UGV's goal strictly increases. Ground and UAV move costs and the other counters in `mission_trace` are cumulative post-transition values; `safety_violation` is a latched boolean. All numeric values are finite, IDs and lists use exact integer ranges, lists follow the ordering above except for the explicitly keyed queue-size object lists, and numeric comparisons use absolute tolerance `1e-6`. Judge controlled experiments are executed by project code and reported in `report.md`; they are not extra scenario records or required artifacts.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root directory must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must overwrite `results.json`. `report.md` must contain non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact scenario-level structures", weight: 3}
- {id: result_evidence, desc: "supporter, seeker, request, packet, channel and mission evidence", weight: 4}
- {id: canonical_coverage, desc: "all scenario, state, action, request and channel keys are complete", weight: 4}
- {id: decision_legality, desc: "moves, requests, queues, capacity, retries, budgets and information are legal", weight: 4}
- {id: replay_validity, desc: "independent map, queue, delivery, motion and safety replay is consistent", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-aware goal completion", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed safe team progress", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed true-map path efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed path-related map freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed map-attempt and useful-delivery efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of UAV exploration, UGV belief navigation, queues, delivery, versions, deadlines and safety"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent UAV-UGV map-support model; Implementation/evidence: no executable navigation or packet process."}
    - {score: 0.4, desc: "Report: omits a decisive local-information, queue, version, timing or deadline relation; Implementation/evidence: only part of the fixed process is represented."}
    - {score: 0.7, desc: "Report: accurately defines local maps, supporter exploration, requests, queues, delivery, safety and deadlines; Implementation/evidence: core model runs across all scenarios with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains how supporter position, map age, capacity and request competition alter UGV paths and completion; Implementation/evidence: records faithfully realize these relations."}
- id: method_design
  desc: "quality of reusable aerial exploration, map selection and local navigation"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method, prohibited training or information-boundary violation; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: fixed flight/send tables or a single distance rule ignore request competition, deadlines or budget; Implementation/evidence: unit-specific actions or material divergence."}
    - {score: 0.7, desc: "Report: legal unified method handles UAV motion, UGV paths, request timing, map selection, queues and budgets; Implementation/evidence: one procedure covers all scenarios with limited lookahead."}
    - {score: 1.0, desc: "Report: justified reproducible method jointly handles supporter reachability, marginal map value, age, deadline, safety and shared capacity; Implementation/evidence: evolving local state drives all decisions without tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for aerial support and communication trade-offs"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one map or communication condition is discussed; Implementation/evidence: no evidence for map-support effects."}
    - {score: 0.7, desc: "Report: all canonical scenarios plus a controlled capacity, delivery, supporter, request-priority or deadline comparison; Implementation/evidence: condition-level records support it."}
    - {score: 1.0, desc: "Report: complementary controlled comparisons isolate supporter detour, map value, capacity/delay and multi-UGV competition with a failure boundary; Implementation/evidence: project code executes them and records distinguish mechanisms."}
- id: research_analysis
  desc: "evidence-grounded interpretation of map support, age, queues, deadlines and navigation"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradictory analysis; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: repeats arrival or packet counts without causal local-information, queue or age analysis and limitations; Implementation/evidence: claims unsupported."}
    - {score: 0.7, desc: "Report: replay evidence explains conditional trends among supporter position, delivery, deadlines and path efficiency and states limitations; Implementation/evidence: code and records agree."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies map value, capacity/delay, supporter trajectory and multi-UGV competition interactions and failure boundaries with scope; Implementation/evidence: condition-level records reproduce claims."}
