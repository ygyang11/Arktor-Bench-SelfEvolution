---
id: T009_knowledge_guided_edge_navigation
name: Knowledge-Guided Edge Feedback for Latency-Constrained Robot Navigation
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

A mobile robot performs a time-critical indoor service task with guidance from an edge server. The robot does not initially know which objects occupy the workspace. When it reaches an object and observes it, it obtains a local semantic label. Some labels identify a category immediately, while others remain ambiguous until another view is observed. The edge server maintains three candidate knowledge paths for the task and can select a feasible path only after enough observations have arrived to confirm every category required by that path.

The uplink is intermittent. Reports sent far from the access point take longer and can be lost, and the robot cannot know the current hidden outage before transmitting. Repeated observations and retransmissions may resolve ambiguity or loss, but consume slots and a finite reporting budget. After the edge confirms a path, a delayed feedback message supplies the ordered route that the robot must execute. Late reporting therefore delays both knowledge confirmation and physical navigation.

### Research Problem

Build a deterministic simulation and design one unified, non-learning online policy for exploration, observation, semantic reporting, retransmission, and execution of delayed edge feedback. In every slot, the policy chooses whether to move, observe one nearby object, report one to three previously observed facts, or wait. It must use only the current local observations, acknowledged edge knowledge, current link-strength indicator, arrived feedback, public map inputs, and its history.

The policy should balance information gain, report aggregation, delivery risk, and the remaining time needed to execute a confirmed route. The same state-driven procedure must handle all five canonical scenarios. It must not use scenario-specific action schedules, object-ID-specific plans, future link outcomes, or object positions and categories before they are observed.

### Experimental Setting and Constraints

#### 1. Canonical episodes and public task inputs

The five canonical scenarios, in required order, are `short_path`, `overlap_paths`, `weak_uplink`, `late_feedback`, and `alias_ambiguity`. Each is one deterministic episode with one robot, eight objects, and 36 slots `t=0,...,35` on a free `12 x 12` integer grid. Coordinates satisfy `x=0,...,11` and `y=0,...,11`; there are no obstacle or collision rules. The robot starts at `[1,1]`. The scenario endpoint, access point, inclusive task deadline, fixed downlink delay, hidden weak-link outage slots, and hidden uplink-delay-spike slots are:

| scenario | endpoint | access point | deadline | downlink delay | weak-link outage slots | uplink delay +2 slots |
|---|---|---|---:|---:|---|---|
| `short_path` | `[11,10]` | `[6,2]` | 34 | 1 | `5,12,20` | `9,18` |
| `overlap_paths` | `[11,10]` | `[6,3]` | 35 | 2 | `6,13,21` | `8,17` |
| `weak_uplink` | `[11,10]` | `[6,6]` | 35 | 2 | `1,6,14,22` | `5,16,25` |
| `late_feedback` | `[10,8]` | `[5,2]` | 31 | 5 | `3,11,19` | `6,14` |
| `alias_ambiguity` | `[11,10]` | `[6,3]` | 35 | 2 | `4,12,20` | `9,18` |

The outage and delay-spike columns are environment truth for simulation and independent replay. They are not policy inputs. The remaining columns, the empty grid, and the candidate paths below are public policy inputs. Each path has an ordered required-category sequence and an ordered route. Its last route coordinate is the scenario endpoint.

| scenario | `p0` | `p1` | `p2` |
|---|---|---|---|
| `short_path` | `key,panel -> [5,2],[8,5],[11,10]` | `valve,parcel -> [8,2],[9,6],[11,10]` | `switch,dock -> [7,5],[9,8],[11,10]` |
| `overlap_paths` | `key,panel,dock -> [6,2],[8,6],[11,10]` | `key,valve,dock -> [7,2],[9,6],[11,10]` | `switch,panel,dock -> [6,4],[8,7],[11,10]` |
| `weak_uplink` | `key,panel -> [6,4],[9,7],[11,10]` | `valve,parcel -> [7,4],[10,7],[11,10]` | `switch,dock -> [6,6],[9,8],[11,10]` |
| `late_feedback` | `key,panel -> [6,2],[8,5],[10,8]` | `valve,parcel -> [7,2],[9,5],[10,8]` | `switch,dock -> [6,4],[8,6],[10,8]` |
| `alias_ambiguity` | `key,panel -> [6,3],[9,7],[11,10]` | `valve,parcel -> [7,3],[10,7],[11,10]` | `switch,dock -> [6,5],[9,8],[11,10]` |

Path order in this table is selection priority. A path is confirmed when the edge has at least one confirmed object in every required category. When multiple paths first become confirmed together, the edge selects the first one in table order.

#### 2. Hidden objects, sensing, and semantic aliases

The following object table is fixed environment truth needed to build the simulator and replay results. A policy must not receive an object's exact position, category, or view-label sequence until the corresponding facts are exposed by the observation process. All scenarios use positions `0:[2,1]`, `1:[4,1]`, `2:[7,1]`, `3:[10,1]`, `4:[10,4]`, `5:[7,4]`, `6:[4,4]`, and `7:[1,4]`. Each object row below is `object:category:[labels for views 1..4]`.

| scenario | object rows in increasing object order |
|---|---|
| `short_path` | `0:key:[key,key,key,key]`; `1:panel:[panel,panel,panel,panel]`; `2:valve:[valve,valve,valve,valve]`; `3:parcel:[parcel,parcel,parcel,parcel]`; `4:dock:[dock,dock,dock,dock]`; `5:switch:[switch,switch,switch,switch]`; `6:panel:[panel,panel,panel,panel]`; `7:dock:[dock,dock,dock,dock]` |
| `overlap_paths` | `0:key:[access,key,key,key]`; `1:panel:[panel,panel,panel,panel]`; `2:dock:[station,dock,dock,dock]`; `3:valve:[valve,valve,valve,valve]`; `4:switch:[switch,switch,switch,switch]`; `5:parcel:[parcel,parcel,parcel,parcel]`; `6:panel:[panel,panel,panel,panel]`; `7:dock:[dock,dock,dock,dock]` |
| `weak_uplink` | same rows as `short_path` |
| `late_feedback` | same rows as `short_path` |
| `alias_ambiguity` | `0:key:[access,key,key,key]`; `1:panel:[control,panel,panel,panel]`; `2:valve:[control,valve,valve,valve]`; `3:parcel:[cargo,parcel,parcel,parcel]`; `4:dock:[station,dock,dock,dock]`; `5:switch:[access,switch,switch,switch]`; `6:panel:[panel,panel,panel,panel]`; `7:dock:[dock,dock,dock,dock]` |

The public alias map is:

```text
key->{key}          panel->{panel}       valve->{valve}
parcel->{parcel}    dock->{dock}         switch->{switch}
access->{key,switch}
control->{panel,valve}
rotary->{valve,switch}
station->{panel,dock}
cargo->{parcel,dock}
```

At the beginning of a slot, the robot's proximity sensor exposes only the sorted IDs of objects at Manhattan distance at most 1 from its current position. This does not expose an exact position, category, or label. An `observe` action names one of these nearby IDs. If the object has been observed fewer than four times, the action creates the next label in its fixed sequence and reveals the exact fact `{observation_id,object,t,position,raw_label}`. Observation IDs are `o0`, `o1`, ... in creation order. The robot retains all observations. A fifth observation of an object, or observing a non-nearby object, is illegal.

For each object, the edge begins with all six categories as candidates. It intersects that set with the alias set of each newly delivered observation. The category becomes confirmed when the intersection first has one element. Fixed observations never produce an empty intersection.

#### 3. Robot actions and uplink reports

Each slot has exactly one action from `move`, `observe`, `report`, and `wait`. A move direction is `N`, `S`, `E`, `W`, or `stay`, with deltas `(0,1)`, `(0,-1)`, `(1,0)`, `(-1,0)`, and `(0,0)`. A non-stationary move must remain inside the grid. `move` uses one direction, a null target object, and an empty report list. `observe` uses `stay`, one integer target object, and an empty report list. `report` uses `stay`, a null target, and a list of one to three distinct locally known observation IDs in increasing numeric-ID order. `wait` uses `stay`, a null target, and an empty report list.

Each listed observation consumes one report unit. At most 24 units may be sent in an episode. In every inclusive four-slot window `[max(0,t-3),t]`, at most two slots may contain a report action. Violating a field relation, movement bound, observation condition, report payload condition, unit budget, or rolling report limit is an illegal decision. An illegal action does not move, observe, or create a message.

A report creates message `u0`, `u1`, ... in sending order. Its payload contains the selected observations exactly. `retry_of` is the latest earlier message with the same ordered observation-ID list, or null if none exists. The pre-decision link indicator is `strong` when the robot's Manhattan distance to the access point is at most 4 and `weak` otherwise. Its base delay is 1 slot for `strong` and 3 slots for `weak`; a send in a listed delay-spike slot adds 2 slots. A send is dropped exactly when its indicator is `weak` and its send slot is a listed outage slot. A dropped message has null `arrival_t` and status `dropped`. Otherwise `arrival_t=send_t+delay`; status is `delivered` when arrival is at most 35 and `late` when it is greater than 35. A late message is not processed. The policy sees the current `strong`/`weak` indicator before deciding, but not the current or future outage or delay-spike outcome.

#### 4. Edge processing, feedback, execution, and slot order

At a slot start, delivered uplinks are processed in increasing numeric message-ID order. An observation ID already processed by the edge has no further effect. A delivered message that adds at least one new observation ID increments the edge knowledge version by one after all its new facts are merged. The robot receives reliable acknowledgements listing the uplink IDs processed in this slot. Because it knows those payloads and the alias rule, it can reconstruct confirmed edge facts; no hidden object fact is added by the acknowledgement.

When one or more paths become confirmed and no feedback has previously been sent, the edge selects the first confirmed path and sends `d0` in that same slot. Its arrival is `send_t+downlink_delay`; downlink is reliable. The feedback contains the selected path ID, required categories, ordered route, and `valid_until=deadline`. It becomes usable only at the beginning of its arrival slot and remains active through the inclusive deadline unless already completed. A feedback arriving after the deadline is never active.

Execution progress is an integer route index, initially zero. After each action with active feedback, if the robot's resulting position equals the next route coordinate, the index advances by exactly one; it cannot skip or consume multiple coordinates in one slot. Reaching the last coordinate completes the episode at the end of that slot. Progress is allowed in the deadline slot. If the route is not complete after that slot, the task expires. Movement before feedback remains legal exploration but cannot advance route progress.

Each slot follows this exact order:

1. Process uplinks whose `arrival_t` is the current slot, update edge candidates, confirmed facts,   knowledge version, and path states, and send the first feedback if this is the first path   confirmation.
2. Deliver a previously sent feedback whose arrival is the current slot.
3. Record the pre-decision position, nearby IDs, local observation IDs, link indicator, used report   units, edge version, processed message IDs, active feedback, and execution index. Choose one   action.
4. Resolve that action. A legal observation creates its fact; a legal report creates its uplink.
5. If feedback is active, compare the post-action position with the next route coordinate and advance   at most once. Mark completion on the final coordinate; otherwise expire after the deadline-slot   action. Record the trace and the pre-decision edge snapshot for this slot.

#### 5. Task metrics

All metrics are computed by independent replay and averaged over the five scenarios. Define `clip(x,0,1)=min(1,max(0,x))`.

- `primary_metrics_1` is the fraction of scenarios completed by the inclusive task deadline.
- `primary_metrics_2` is `max(0,1-finish_t/deadline_t)` for a completed scenario and zero otherwise,  averaged over scenarios.
- `auxiliary_metrics_1` is `clip(1-n_report_units/24,0,1)` per scenario, averaged over scenarios.
- `auxiliary_metrics_2` is the scenario average of  `0.5*clip(1-n_repeat_observations/24,0,1) +  0.5*clip(1-n_duplicate_report_units/24,0,1)`. A repeat observation is any legal observation after  the first one of the same object. A duplicate report unit is a delivered observation ID that the  edge had already processed before processing that message. Both counts have fixed upper bound 24.

The normalized task performance is `(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`. Higher is better for every metric.

#### 6. `results.json` contract

The root object has exactly `scenarios`. Its array contains the five scenario objects in canonical order. Each scenario object has exactly `scenario`, `robot_trace`, `objects`, `uplink`, `edge_state`, and `downlink`. All integer fields reject Booleans. Numeric comparisons use absolute tolerance `1e-6`; all numeric values must be finite.

`robot_trace` contains exactly 36 records in increasing `t` order. Each record has exactly:

- `t`: integer `0..35`; `position`: two grid integers representing the pre-decision position.
- `nearby_objects`: sorted distinct object IDs; `known_observations`: observation IDs in creation  order; `link_indicator`: `strong` or `weak`; `report_units_used`: integer `0..24` before the  action; `edge_version`: a nonnegative integer.
- `processed_uplink`: uplink IDs processed at this slot start in numeric order;  `processed_downlink`: either `[]` or `["d0"]` when feedback arrives in this slot.
- `active_feedback`: `d0` or null; `execution_index`: a nonnegative integer before the action.
- `action`: `move`, `observe`, `report`, or `wait`; `move`: `N`, `S`, `E`, `W`, or `stay`;  `target_object`: object ID or null; `report_observations`: an array of observation IDs.
- `observation_result`: null or exactly  `{observation_id,object,t,position,raw_label}` for the observation created by this action.
- `action_result`: exactly `{status,reference}`. Status is `moved`, `none`, `observed`, `reported`,  or `invalid`. A non-stationary accepted move is `moved`; stationary move and wait are `none`;  successful observe and report use `observed` and `reported`. `reference` is the created  observation or uplink ID for those two statuses and null otherwise.

`objects` contains exactly eight records in increasing object order. Each record has exactly `object`, `position`, `category`, `view_labels`, `first_observe_t`, `observations`, `delivered_observations`, and `confirmed_t`. Static fields match the environment table; `view_labels` has four strings. The nullable time fields are integers `0..35`. `observations` is the creation-ordered list of exact `{observation_id,object,t,position,raw_label}` facts for that object. `delivered_observations` contains those observation IDs processed by the edge, in observation-ID order. `confirmed_t` is the first slot at which the edge candidate set became a singleton.

`uplink` is in increasing message-ID and send order. Each record has exactly `message_id`, `send_t`, `observation_ids`, `facts`, `units`, `link_indicator`, `status`, `arrival_t`, and `retry_of`. IDs are `u0,u1,...`; `facts` is the ordered list of exact observation facts named by `observation_ids`; `units` equals its length. Status is `delivered`, `dropped`, or `late`. `arrival_t` is null only for `dropped`, and otherwise is the integer computed above in `1..40`. `retry_of` is a prior uplink ID or null under the rule above.

`edge_state` contains exactly 36 records in increasing `t` order. Each has exactly `t`, `version`, `confirmed_facts`, `path_states`, `selected_path`, and `feedback_id`. A confirmed fact is exactly `{object,category,position,confirmed_t}` and facts are in increasing object order. `path_states` contains exactly `p0`, `p1`, and `p2` in path order, each as `{path_id,status}` with status `open` or `confirmed`. `selected_path` and `feedback_id` are both null before selection and then retain the selected path ID and `d0`.

`downlink` is empty if no path is confirmed; otherwise it contains exactly one record with exactly `feedback_id`, `send_t`, `arrival_t`, `path_id`, `required_categories`, `route`, `valid_until`, and `status`. Its static path fields match the selected public path. Status is `completed` if the route finishes by the deadline, `expired` if feedback arrives by the deadline but the route does not finish, and `late` if feedback arrives after the deadline.

The scenario, trace-slot, object, and edge-slot records form the fixed canonical key sets. Uplink and downlink are variable event records, but their IDs, order, payload links, and derived fields must be complete. The process is deterministic; do not include random seeds or self-reported metric fields.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` from the project code and fixed conditions. The official evaluator deletes an old result, runs `python run.py` once from the project root, and then reads the newly generated result.

`report.md` must contain nonempty sections titled exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified policy, experiments, actual results, evidence-based mechanisms, and limitations.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes Python static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scenario containers", weight: 3}
- {id: result_evidence, desc: "trace, observation, uplink, edge-state, and feedback records have the declared fields and domains", weight: 4}
- {id: canonical_coverage, desc: "all scenario, trace-slot, object, and edge-slot canonical keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "action relations, movement, observation, report payload, budget, and rolling-rate decisions obey the public constraints", weight: 4}
- {id: replay_validity, desc: "independent replay matches sensing, uplink, edge knowledge, feedback, route progress, and terminal records", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-valid route completion fraction", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed route-completion temporal utility", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed semantic-report unit efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed repeat-observation and duplicate-report control", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the semantic observation, intermittent uplink, edge feedback, and navigation formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent observation-to-feedback task model, or a materially wrong model; Implementation/evidence: no corresponding executable state, message, or route records."}
    - {score: 0.4, desc: "Report: covers only part of sensing, aliases, uplink timing, edge confirmation, feedback validity, or route execution; Implementation/evidence: realizes only a partial or divergent model."}
    - {score: 0.7, desc: "Report: accurately defines local observations, semantic aliases, message timing, edge knowledge, feedback, route progress, constraints, and objectives; Implementation/evidence: implements the core model across all canonical scenarios with only limited omissions."}
    - {score: 1.0, desc: "Report: precisely connects multi-view evidence, delivery uncertainty, knowledge-path confirmation, delayed feedback, navigation time, budgets, and objective trade-offs; Implementation/evidence: faithfully exposes these causal relations and boundary states without material divergence."}
- id: method_design
  desc: "quality of unified exploration, semantic reporting, retransmission, and feedback execution decisions"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, or a prohibited or incompatible method; Implementation/evidence: no runnable unified decision procedure."}
    - {score: 0.4, desc: "Report: uses isolated geometric exploration or fixed reporting while ignoring a decisive ambiguity, delivery, budget, feedback, or remaining-route consequence; Implementation/evidence: relies on pre-filled scenario schedules or materially differs from the report."}
    - {score: 0.7, desc: "Report: gives a legal, coherent, reusable online policy for the principal observation, reporting, and navigation decisions; Implementation/evidence: applies it across scenarios with state-driven records but limited lookahead or robustness."}
    - {score: 1.0, desc: "Report: justifies a unified reproducible method that couples path information value, multi-view evidence, report aggregation or retransmission, link indication, feedback delay, and remaining route feasibility; Implementation/evidence: decisive choices follow evolving visible state and history without scenario-, object-, or slot-specific action tables."}
- id: experimental_coverage
  desc: "coverage of semantic-report mechanisms, uplink pressure, feedback timing, and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only isolated scenarios or aggregate values and no direct test of semantic reporting or edge feedback; Implementation/evidence: records do not support the main experimental claims."}
    - {score: 0.7, desc: "Report: covers all canonical scenarios and at least one controlled comparison of report aggregation, retransmission, multi-view observation, or feedback timing; Implementation/evidence: both conditions are executed on comparable inputs with condition-level evidence."}
    - {score: 1.0, desc: "Report: uses complementary non-redundant conditions including a mechanism-focused semantic-report comparison and an uplink, ambiguity, deadline, or feedback failure boundary; Implementation/evidence: condition-level observation, message, edge-state, route, and completion records distinguish both claims beyond canonical aggregates."}
- id: research_analysis
  desc: "evidence-based analysis of semantic information value, delivery cost, and navigation consequences"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts recorded results; Implementation/evidence: cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mainly repeats completion or message counts, makes unsupported causal claims, or omits material limitations; Implementation/evidence: lacks the records needed for those claims."}
    - {score: 0.7, desc: "Report: uses actual records to explain conditional trends among ambiguity, report delay or loss, edge confirmation, feedback arrival, and route completion, distinguishes observation from inference, and states limitations; Implementation/evidence: supports those comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence explains when added views, aggregation, or retransmission changes path feasibility, quantifies its reporting and route-time trade-offs, tests the explanation against a contrasting failure boundary, and limits scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
