---
id: T008_adaptive_intermittent_task_coordination
name: Adaptive Team-Wise Rendezvous Scheduling for Intermittently Connected Multi-Robot Tasks
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

A team of four ground robots handles time-critical inspection and service requests in a partitioned industrial site. Requests are released while the robots are already moving. Some requests need two robots to work together, while precedence, mutual-exclusion, and concurrent-start relations couple the order in which other requests can be served.

Task data are local. A robot learns a released request only after sensing it nearby, and another robot cannot use that information until a successful team-wise rendezvous. A one-bit request can activate a task-free rendezvous plan, but exchanging task facts requires all four robots to reach their planned meeting positions and form a connected wireless graph. Rendezvous travel and early waiting consume time that could otherwise be spent on tasks; delaying communication leaves the team with stale and inconsistent task knowledge.

### Research Problem

Build a deterministic simulation and design one unified, non-learning online policy that jointly chooses each robot's movement and work actions, the robots assigned to cooperative tasks, one-bit rendezvous requests, and the time and four meeting positions of each team-wise communication event. The policy must respond to locally discovered task facts, synchronized facts, robot positions, task relations, remaining deadlines, active plans, and movement and event budgets.

The central research trade-off is between timely information propagation and productive task execution. Frequent or conservative rendezvous can expose cooperative and related tasks early but consume travel and idle slots. Sparse rendezvous preserve task time but can prevent compatible teams from forming before deadlines. The same state-driven procedure must handle all canonical scenarios without scenario-specific action schedules or pre-filled rendezvous plans.

### Experimental Setting and Constraints

#### 1. Canonical evaluation units

The five canonical scenarios, in required order, are `sparse_sequence`, `precedence_chain`, `parallel_team`, `connectivity_bottleneck`, and `bursty_arrivals`. Each scenario is one deterministic episode with four robots, ten tasks, and 36 slots `t=0,...,35`. Coordinates are integer cells in a `14 x 10` grid: `x=0,...,13`, `y=0,...,9`. The state after slot 35 is the terminal evaluated state.

Robots are indexed `0,...,3`. Their starting positions and the blocked movement/radio cells are:

| scenario | robot positions `0..3` | obstacle cells |
|---|---|---|
| `sparse_sequence` | `[(0,0),(13,0),(0,9),(13,9)]` | `[(6,3),(6,4),(7,4),(7,5)]` |
| `precedence_chain` | `[(1,1),(12,1),(1,8),(12,8)]` | `[(6,0),(6,1),(6,3),(6,4),(6,5),(6,6),(6,8),(6,9)]` |
| `parallel_team` | `[(1,2),(12,2),(1,7),(12,7)]` | `[(2,4),(3,4),(5,4),(6,4),(7,4),(8,4),(10,4),(11,4)]` |
| `connectivity_bottleneck` | `[(1,1),(3,8),(10,1),(12,8)]` | `[(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,7),(9,8),(9,9)]` |
| `bursty_arrivals` | `[(1,1),(12,1),(1,8),(12,8)]` | `[(5,2),(5,3),(5,6),(5,7),(8,2),(8,3),(8,6),(8,7)]` |

Each task row below has the form `task:position:release_t:deadline_t:duration:team_size:predecessors:mutex_group:concurrent_group`. `-` is null and square brackets hold predecessor task IDs.

| scenario | tasks in increasing task order |
|---|---|
| `sparse_sequence` | `0:[2,1]:0:9:2:1:[]:-:-`; `1:[11,1]:1:11:2:1:[]:-:-`; `2:[2,8]:4:17:2:2:[]:-:-`; `3:[11,8]:6:18:2:1:[0]:-:-`; `4:[5,2]:9:22:2:1:[1]:m0:-`; `5:[8,7]:10:24:2:2:[]:m0:-`; `6:[1,5]:14:28:1:1:[2]:-:-`; `7:[12,4]:15:29:1:1:[3]:-:-`; `8:[5,7]:19:33:1:1:[]:-:c0`; `9:[8,2]:19:33:1:1:[]:-:c0` |
| `precedence_chain` | `0:[2,2]:0:10:2:1:[]:-:-`; `1:[11,2]:0:11:2:1:[]:-:-`; `2:[4,7]:3:18:2:2:[0]:-:-`; `3:[9,7]:4:19:2:1:[1]:-:-`; `4:[2,7]:7:22:1:1:[0,1]:-:-`; `5:[6,2]:9:26:2:2:[2]:-:-`; `6:[11,7]:11:27:1:1:[3]:m1:-`; `7:[4,2]:14:30:2:1:[4]:m1:-`; `8:[5,7]:18:34:1:1:[5]:-:c1`; `9:[8,7]:18:34:1:1:[6]:-:c1` |
| `parallel_team` | `0:[2,2]:0:12:2:2:[]:-:-`; `1:[11,2]:0:12:2:2:[]:-:-`; `2:[2,7]:3:16:1:1:[]:-:c2`; `3:[11,7]:3:16:1:1:[]:-:c2`; `4:[4,4]:6:21:2:1:[]:m2:-`; `5:[9,4]:6:21:2:1:[]:m2:-`; `6:[4,7]:10:26:2:2:[2]:-:-`; `7:[9,7]:10:26:2:2:[3]:-:-`; `8:[4,2]:16:32:1:1:[4]:-:c3`; `9:[9,2]:16:32:1:1:[5]:-:c3` |
| `connectivity_bottleneck` | `0:[2,2]:0:11:2:1:[]:-:-`; `1:[11,2]:0:11:2:1:[]:-:-`; `2:[6,2]:3:17:2:2:[]:-:-`; `3:[7,7]:4:19:2:2:[]:-:-`; `4:[2,7]:7:22:1:1:[0]:-:-`; `5:[11,7]:8:23:1:1:[1]:-:-`; `6:[6,5]:11:27:2:2:[2]:m3:-`; `7:[7,4]:12:28:2:2:[3]:m3:-`; `8:[3,3]:17:33:1:1:[4]:-:c4`; `9:[10,6]:17:33:1:1:[5]:-:c4` |
| `bursty_arrivals` | `0:[2,2]:0:11:2:1:[]:-:-`; `1:[11,2]:0:12:2:1:[]:-:-`; `2:[2,7]:1:14:2:2:[]:-:-`; `3:[11,7]:1:15:2:2:[]:-:-`; `4:[6,1]:8:22:1:1:[0]:-:c5`; `5:[7,8]:8:22:1:1:[1]:-:c5`; `6:[3,5]:9:25:2:2:[]:m4:-`; `7:[10,4]:9:25:2:1:[]:m4:-`; `8:[6,5]:17:33:2:2:[2]:-:-`; `9:[7,4]:17:34:2:2:[3]:-:-` |

The scenario and task tables belong to the environment simulator. Before a task is locally known, the policy interface must not expose that row, its release, position, deadline, team size, duration, or relations. Future releases and undiscovered task rows therefore cannot be used to choose policy actions or rendezvous requests.

#### 2. State, actions, task execution, and sensing

Each robot chooses exactly one `action` per slot from `move`, `work`, and `wait`. A move is one of `N`, `S`, `E`, `W`, and `stay`, with deltas `(0,1)`, `(0,-1)`, `(1,0)`, `(-1,0)`, and `(0,0)` respectively. `move` uses a null task; `work` uses an integer task ID and `stay`; `wait` uses a null task and `stay`. A non-stationary move is accepted only if it stays in the grid, avoids an obstacle, and the robot has used fewer than 28 movement steps. An accepted non-stationary move consumes one step. Submitting a non-stationary move that violates one of these bounds is an illegal decision. Robots may occupy the same cell, and there is no collision rule. A legal non-stationary move records `action_result={"status":"moved","task":null}`; a legal `move` with `stay` or a legal `wait` records `{"status":"none","task":null}`; a rejected move records `{"status":"invalid","task":null}`.

A task is `unreleased` before `release_t`, becomes `available` at the start of its release slot, and becomes `expired` after work but before sensing in its deadline slot if it has not completed. Work in the deadline slot is allowed. A robot may submit `work` only when its pre-decision local cache contains the task in a nonterminal state (`available` or `active`), `t <= deadline_t`, and its beginning-of-slot position equals the task position. A cached `completed` or `expired` fact therefore makes the work submission locally inadmissible. Violating one of these locally knowable conditions is an illegal decision. Every work submission records the selected task in `action_result.task`. Illegal work and work that is locally admissible but cannot advance under the true team or relation state return `invalid` without advancing the task; progressing work returns `working`, or `completed` in the slot that finishes the task.

An available task starts only when exactly `team_size` robots submit admissible work for it in the same slot. That sorted team remains fixed. During every subsequent slot, exactly the complete fixed team must work on the task; otherwise its progress resets to zero, its team is cleared, it returns to `available`, and it cannot restart in that same slot. Each successful work slot adds one unit of progress. The task completes after its progress reaches `duration`, at the end of that slot.

All predecessors must already be completed at the beginning of a task's first work slot. At most one task in a non-null `mutex_group` may advance in a slot; if multiple such tasks would otherwise advance, none advances. Every non-null `concurrent_group` contains exactly two tasks, so a policy that knows two task rows with the same group label knows that group's complete membership. Both tasks must satisfy their own start conditions and first start in the same slot. Concurrent-start coupling applies only to those first starts; if later interrupted, either task may restart without repeating the group start, while continuation still follows each task's fixed team. A blocked start or continuation gives all submitted workers for the affected task an `invalid` result.

After movement and work, a robot senses a released task when its end-of-slot position has Manhattan distance at most 2 from the task and at least one of the two canonical shortest L-shaped paths has no obstacle in its interior. One canonical path moves in `x` first and then `y`; the other moves in `y` first and then `x`. Sensing or participating in work writes a fact only if that task is absent from the local cache or its visible state, progress, or team changed. A fact is exactly `{task,state,progress,team,version}`; `team` is sorted and `version` is the write slot. Available and expired facts have zero progress and an empty team; active and completed facts retain their current progress and team.

#### 3. Information and slot order

A task's static row may be supplied to a robot's policy only after that robot has a local fact for the task. Current robot positions, the obstacle map, movement counters, active rendezvous metadata, and previous event outcomes are available through a task-free positioning/control beacon. The beacon never carries a task ID, task property, task state, assignment, or progress. A robot action may use its own local facts and facts previously synchronized to all robots; it may not use another robot's unsynchronized cache. A one-bit event request may depend on the requesting robot's local state but contains no task data.

When an eligible nonempty request set exists, a deterministic team-wise plan selector chooses the attempt slot and meeting positions. Its input is limited to the proposal slot's pre-decision control-beacon snapshot (scenario obstacle map, all current positions, movement counters, active plan metadata, and completed event outcomes), the sorted requester IDs, and task facts delivered to all robots by earlier successful events. It does not receive any robot's unsynchronized cache, the robot actions selected in the proposal slot, observations produced in that slot, or current or future blackout outcomes. The selector output is therefore not an input to proposal-slot actions.

Each slot uses this exact order:

1. Deliver facts from a successful event attempted in the preceding slot, then release tasks whose   `release_t` equals the current slot.
2. Record the pre-decision positions, movement counters, local facts, and active event. Choose all   robot actions and one-bit event requests. If requests are eligible, choose the new event's attempt   slot and four meeting positions; the plan is not visible to actions in this proposal slot.
3. Resolve movement, then task work, relation checks, progress, completion, and deadline-slot expiry.
4. Apply work observations and end-of-slot sensing to local caches.
5. Attempt an event already scheduled for this slot, using the end-of-slot positions and facts.
6. If an eligible new event was requested in step 2, announce its metadata for use from the next   slot. Record robot results and post-action task timelines.

At a successful event, merge the four post-observation caches separately for each task. A larger `version` wins; at equal version the fact from the lower robot ID wins. The sorted merged facts are delivered to all four robots at the beginning of the next slot. A failed event shares no facts.

#### 4. Rendezvous protocol and boundaries

A robot records `event_request=true` to submit its one-bit request. One event is created from the sorted set of all requesters only if there is no active event, fewer than six events have already been attempted or planned, and `propose_t - previous_attempt_t >= 2`; the first event has no prior gap restriction. Requests while an event is active or resources are unavailable are illegal. Every request set must correspond to exactly one event record, and an event cannot exist without at least one requester.

Event IDs are `e0`, `e1`, ... in proposal order. `attempt_t` must be from `propose_t+2` through `propose_t+7` and no later than 34. The four meeting positions correspond to robots `0..3` and must be free grid cells. The plan is active from `propose_t+1` through `attempt_t`, inclusive. No new request is allowed while it is active.

At the end of `attempt_t`, robots `0..3` must be at their respective planned meeting positions. For robot positions `p` and `q`, a radio edge exists when their Manhattan distance is at most 5, at least one canonical L-shaped path defined above has no obstacle in its interior, and their unordered robot pair is not blacked out at that scenario and slot. Unlisted scenario-slot pairs have no extra blackout:

| scenario | blacked-out slot and unordered robot pairs |
|---|---|
| `sparse_sequence` | `9:{0-2}`; `10:{1-3}`; `21:{0-1,2-3}` |
| `precedence_chain` | `8:{0-1}`; `12:{0-2,1-3}`; `20:{0-1,2-3}`; `28:{0-3}` |
| `parallel_team` | `7:{0-1,2-3}`; `15:{0-2,1-3}`; `24:{0-3,1-2}` |
| `connectivity_bottleneck` | `6:{0-2,1-3}`; `7:{0-2,1-3}`; `14:{0-1,2-3}`; `15:{0-1,2-3}`; `22:{0-2,0-3,1-3}`; `30:{1-2}` |
| `bursty_arrivals` | `5:{0-1,2-3}`; `10:{0-2,1-3}`; `11:{0-2,1-3}`; `18:{0-3,1-2}`; `26:{0-1}` |

The blackout outcome for the current or a future slot is environment truth and is unavailable to the policy before the attempt. The four-node graph may use multi-hop paths. If any actual position differs from its plan, status is `missed`, even if the actual graph is connected. Otherwise status is `success` when the graph is connected and `disconnected` when it is not. A successful event has `confirm_t=attempt_t+1`; a failed event has null `confirm_t` and an empty fact exchange.

#### 5. Task metrics

All metrics are independently replayed and averaged over the five scenarios.

- `primary_metrics_1` is the fraction of the ten tasks that complete by their deadline while  satisfying their team and temporal relations. Its physical bounds are 0 and 1; higher is better.
- `primary_metrics_2` assigns each completed task  `max(0, 1-(finish_t-release_t)/(deadline_t-release_t))`; an incomplete task receives zero. All  task windows are positive. Average over ten tasks and then scenarios; bounds are 0 and 1 and  higher is better.
- `auxiliary_metrics_1` is `clip(1-n_event_attempts/6,0,1)` per scenario. Every planned event,  whether successful, missed, or disconnected, counts once. Six is the fixed attempt budget.
- `auxiliary_metrics_2` is `clip(1-n_rendezvous_wait/144,0,1)` per scenario. A rendezvous-wait slot  is a robot slot before the active event's attempt in which the action is `wait` or `move` with  `stay`. The model upper bound is all four robots idle over all 36 slots, or 144 slots.

The normalized task performance is `(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`.

#### 6. `results.json` contract

The root object has exactly `scenarios`. Its array contains the five scenario objects in canonical order. Each scenario object has exactly `scenario`, `robots`, `tasks`, and `communication_events`. All integer fields reject booleans. All numeric values must be finite; numeric comparisons use absolute tolerance `1e-6`.

`robots` contains exactly one record for each `(robot,t)` pair, ordered by `t` and then `robot`. Each record has exactly:

- `robot`: integer `0..3`; `t`: integer `0..35`.
- `position`: two integers within the grid, representing the beginning-of-slot position.
- `move_steps_used`: integer `0..28`, before this slot's action.
- `action`: `move`, `work`, or `wait`; `move`: `N`, `S`, `E`, `W`, or `stay`; `task`: integer  `0..9` or null, with the action-field relations defined above.
- `known_tasks`: the pre-decision facts, sorted by increasing distinct task ID.
- `active_event`: the current `e<integer>` ID or null; `event_request`: Boolean.
- `observed_facts`: facts actually written after work or sensing in this slot, sorted by increasing  distinct task ID. Delivery at slot start is represented in `known_tasks`, not in this array.
- `action_result`: exactly `{status,task}`. Status is `none`, `moved`, `working`, `completed`, or  `invalid`. `none` and `moved` use a null task; every work result uses the selected task.

Every fact in `known_tasks`, `observed_facts`, `basis_facts`, or `shared_facts` has exactly `{task,state,progress,team,version}`. `task` is `0..9`; `state` is `available`, `active`, `completed`, or `expired`; `progress` is an integer `0..duration`; `team` is a sorted distinct array of robot IDs; `version` is `0..35`. Available and expired facts have progress zero and an empty team. Active facts have progress `1..duration-1` and exactly `team_size` robots. Completed facts have progress equal to duration and exactly `team_size` robots.

`tasks` contains exactly ten records in increasing task order. Each has exactly `task`, `position`, `release_t`, `deadline_t`, `duration`, `team_size`, `predecessors`, `mutex_group`, `concurrent_group`, `first_discover_t`, `start_t`, `finish_t`, `assigned_robots`, `timeline`, and `completion_reason`. Static fields exactly match the scenario table. Discovery/start/finish fields are integers `0..35` or null. `first_discover_t` is the earliest slot in which any robot first senses the task at slot end; it is null if no robot ever senses the task. `predecessors` and `assigned_robots` are sorted arrays of distinct task IDs `0..9` and robot IDs `0..3`, respectively. `timeline` has exactly 36 records `{t,state,progress,assigned_robots}` in time order, with `t` equal to `0,...,35`; every timeline assignment is also a sorted distinct robot-ID array. Timeline state is `unreleased`, `available`, `active`, `completed`, or `expired`; team and progress obey the state rules above. `start_t` and the task-level `assigned_robots` describe the uninterrupted execution segment that completed the task; for an incomplete task they are null and empty. `completion_reason` is `completed` or `expired`.

`communication_events` is in proposal order. Every record has exactly `event_id`, `propose_t`, `requesters`, `attempt_t`, `meeting_positions`, `arrival_t`, `basis_facts`, `connected`, `confirm_t`, `shared_facts`, and `status`.

- `event_id` follows `e0,e1,...`; proposal and attempt slots are integers in their stated bounds.
- `requesters` is the sorted nonempty set of robot IDs that requested at `propose_t`.
- `meeting_positions` has four valid free coordinates in robot order. `arrival_t` has four entries;  each is the first end-of-slot time in `propose_t+1,...,attempt_t` when that robot reached its  planned position, or null if it never did.
- `basis_facts` is the sorted exact intersection of all four pre-decision caches at proposal time.
- `connected` is Boolean and reports the actual graph at the attempt. `status` is `success`,  `missed`, or `disconnected`. `confirm_t` is `attempt_t+1` only for success and null otherwise.
- `shared_facts` is the sorted merged post-observation cache for success and is empty for failure.

The process is deterministic. Do not include random seeds or self-reported metric fields.

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
- {id: result_evidence, desc: "robot, local-fact, task-timeline, and rendezvous records have the declared evidence fields", weight: 4}
- {id: canonical_coverage, desc: "all scenario, robot-slot, and task canonical keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "action relations, movement and work preconditions, event requests, resources, and meeting positions obey the public constraints", weight: 4}
- {id: replay_validity, desc: "independent replay matches movement, discovery, work, temporal relations, communication, and terminal task records", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-valid task completion fraction", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed release-to-deadline temporal utility", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed rendezvous-attempt efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed rendezvous waiting control", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the intermittently connected cooperative-task formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent dynamic-task and rendezvous formulation, or a materially wrong model; Implementation/evidence: no corresponding executable state or event records."}
    - {score: 0.4, desc: "Report: covers only part of discovery, continuous team work, temporal relations, information boundaries, or rendezvous effects; Implementation/evidence: realizes only a partial or divergent model."}
    - {score: 0.7, desc: "Report: accurately defines task lifecycle, local facts, team work, temporal relations, event timing, constraints, and objectives; Implementation/evidence: implements the core model across all canonical scenarios with only limited omissions."}
    - {score: 1.0, desc: "Report: precisely connects discovery, fact synchronization, fixed-team progress, relation feasibility, meeting travel/wait, budgets, and objective trade-offs; Implementation/evidence: faithfully exposes these causal relations and boundary states without material divergence."}
- id: method_design
  desc: "quality of unified task, team, movement, and rendezvous decisions"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, or a prohibited or incompatible method; Implementation/evidence: no runnable unified decision procedure."}
    - {score: 0.4, desc: "Report: uses isolated nearest-task or fixed-rendezvous rules while ignoring a decisive team, relation, knowledge, or waiting consequence; Implementation/evidence: relies on pre-filled scenario schedules or materially differs from the report."}
    - {score: 0.7, desc: "Report: gives a legal, coherent, reusable online policy for the principal task and event decisions; Implementation/evidence: applies it across scenarios with state-driven records but limited lookahead or robustness."}
    - {score: 1.0, desc: "Report: justifies a unified reproducible method that couples local knowledge, task relations, team commitments, deadlines, positions, event value, and remaining resources; Implementation/evidence: decisive choices follow evolving state and history without scenario-specific action or rendezvous tables."}
- id: experimental_coverage
  desc: "coverage of rendezvous mechanisms, task relations, and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only isolated scenarios or aggregate values and no direct test of intermittent synchronization or cooperative timing; Implementation/evidence: records do not support the main experimental claims."}
    - {score: 0.7, desc: "Report: covers all canonical scenarios and at least one controlled comparison of rendezvous timing, communication availability, or task-relation handling; Implementation/evidence: both conditions are executed on comparable inputs with condition-level evidence."}
    - {score: 1.0, desc: "Report: uses complementary non-redundant conditions including a mechanism-focused communication comparison and a connectivity, deadline, or team-formation failure boundary; Implementation/evidence: condition-level task, event, waiting, and completion records distinguish both claims beyond canonical aggregates."}
- id: research_analysis
  desc: "evidence-based analysis of synchronization value and execution cost"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts recorded results; Implementation/evidence: cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mainly repeats completion or event counts, makes unsupported causal claims, or omits material limitations; Implementation/evidence: lacks the records needed for those claims."}
    - {score: 0.7, desc: "Report: uses actual records to explain conditional trends among stale facts, team formation, deadlines, event cost, and waiting, distinguishes observation from inference, and states limitations; Implementation/evidence: supports those comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence explains when synchronization changes task feasibility, quantifies its travel/wait trade-offs, tests the explanation against a contrasting failure boundary, and limits scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
