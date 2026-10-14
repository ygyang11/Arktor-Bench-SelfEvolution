---
id: T007_event_triggered_task_allocation
name: Event-Triggered Cooperative Task Allocation under Delayed Capability Information
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

Four heterogeneous ground robots search a flat `12 x 12` operating area for short-lived rescue tasks. A task can be completed only by a robot with the matching capability, but its requirement is not visible until a robot reaches the task and spends a slot inspecting it. A robot may then claim and execute the task, while nearby robots can learn the inspected requirement and claim through delayed wireless summaries. A claim made from stale local information can compete with another claim, and a late or failed execution leaves the task unfinished.

The robots have no central coordinator. Each slot begins with delayed summaries from the previous slot and newly released tasks, then every robot chooses one local action and whether to send a bounded summary. Movement, inspection, claiming, execution, abandonment, and communication all consume the same finite horizon. Wireless links depend on distance and deterministic blackout events; a successful summary is usable only in the next slot. Unnecessary periodic broadcasts consume scarce opportunities, while too little communication causes duplicate inspection, stale claims, and missed deadlines.

### Research Problem

Develop a self-contained, non-training simulation and one reusable decentralized policy for the dynamic multi-robot task allocation problem. At every slot, each robot must choose a movement or task action, a target task when needed, whether to broadcast, and which currently known task facts to include. The policy must use only the robot's current position, capability, locally known requirements and claims, arrived messages, action outcomes already observed, and remaining communication history. It must balance early completion and deadline utility against broadcast cost, avoidable inspections, claim conflicts, delayed capability propagation, and execution failures. The same procedure must generate all canonical scenarios; do not provide scenario- or slot-specific action tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios, evaluated in this order: `dense_arrival`, `capability_bottleneck`, `relay_gap`, `late_urgent`, and `balanced`. Coordinates are integer ((x,y)) with (0\le x,y<12), and the grid has no obstacles. Robots `0..3` start at `[[0,0],[11,0],[0,11],[11,11]]` with capabilities `red`, `green`, `blue`, and `blue`. The horizon is `t=0,...,39`; a robot record at slot `t` describes the state before that slot's action. Each scenario has the following twelve tasks. Every task's position, release time, and deadline are policy-visible inputs from `t=0`, including for tasks that have not yet been released. The requirement is environment truth and is not available to a policy before a successful local inspection or an arrived message.

| scenario | task records `(task:position:release_t:deadline_t:requirement)` |
|---|---|
| `dense_arrival` | `0:[3,3]:0:16:red`; `1:[4,3]:1:18:green`; `2:[3,4]:2:20:blue`; `3:[4,4]:3:22:blue`; `4:[5,4]:4:24:red`; `5:[6,4]:5:26:green`; `6:[5,5]:6:27:blue`; `7:[6,5]:8:29:red`; `8:[5,6]:10:31:green`; `9:[6,6]:12:33:blue`; `10:[4,6]:14:35:red`; `11:[7,6]:16:37:blue` |
| `capability_bottleneck` | `0:[2,9]:0:18:red`; `1:[3,8]:0:20:red`; `2:[4,7]:1:22:red`; `3:[8,9]:1:19:green`; `4:[9,8]:2:23:green`; `5:[8,3]:3:20:blue`; `6:[9,4]:4:24:blue`; `7:[6,6]:4:25:red`; `8:[5,5]:6:27:red`; `9:[6,3]:8:29:green`; `10:[3,2]:10:31:blue`; `11:[7,1]:12:34:red` |
| `relay_gap` | `0:[1,5]:0:20:red`; `1:[2,6]:1:22:green`; `2:[1,7]:2:24:blue`; `3:[2,8]:3:25:blue`; `4:[10,5]:0:20:red`; `5:[9,6]:1:22:green`; `6:[10,7]:2:24:blue`; `7:[9,8]:3:25:blue`; `8:[5,1]:4:27:red`; `9:[6,2]:6:29:green`; `10:[5,10]:8:31:blue`; `11:[6,9]:10:33:red` |
| `late_urgent` | `0:[2,2]:10:18:red`; `1:[9,2]:10:19:green`; `2:[2,9]:11:21:blue`; `3:[9,9]:11:22:blue`; `4:[4,2]:12:22:red`; `5:[7,2]:13:23:green`; `6:[4,9]:14:25:blue`; `7:[7,9]:14:26:red`; `8:[5,5]:16:28:green`; `9:[6,5]:17:29:blue`; `10:[5,6]:18:30:red`; `11:[6,6]:20:32:blue` |
| `balanced` | `0:[2,2]:0:20:red`; `1:[9,2]:0:21:green`; `2:[2,9]:1:22:blue`; `3:[9,9]:1:23:blue`; `4:[4,4]:3:24:red`; `5:[7,4]:4:25:green`; `6:[4,7]:5:26:blue`; `7:[7,7]:6:27:red`; `8:[5,3]:8:29:green`; `9:[6,3]:9:30:blue`; `10:[5,8]:10:32:red`; `11:[6,8]:11:34:blue` |

#### State, actions and task model

All task requirements, release times, deadlines, and positions above are fixed environment inputs. At the beginning of a slot, tasks already past their deadlines expire, messages scheduled for that slot are merged, and tasks released in that slot become active, in that order. Simultaneously arriving messages are merged by increasing sender ID; entries retain their recorded order. A local fact has the form `{task,requirement,claim_robot,version}`. A larger `version` replaces an older fact, while an equal version does not replace the fact already held, including one accepted from an earlier sender in the same merge. The initial local fact sets are empty. A robot's pre-decision state contains only its position, capability, the public task geometry and timing, local facts, arrived message IDs, observed action history, and broadcast history. It cannot use another robot's current position or knowledge, a current or future link outcome, a future message, a current or future execution-failure outcome, or a task requirement that it has not learned.

Each slot a robot chooses exactly one `action` from `move`, `inspect`, `claim`, `execute`, `abandon`, and `wait`, a `move` from `N`, `S`, `E`, `W`, and `stay`, and a task ID exactly for `inspect`, `claim`, `execute`, and `abandon`; `move` and `wait` use a null task. Only `move` may use a non-`stay` movement. Every robot has a proposed destination: a valid move endpoint for a moving robot, and its current cell for `move/stay` or any non-move action. Moving outside the grid is invalid and uses the robot's current cell as its proposal. The conflict set is computed once from all robots' proposals, including each stationary robot's current cell. If two or more robots have the same proposed destination, every involved moving robot remains at its prior position and each stationary robot remains stationary. Two robots proposing to exchange positions also both remain at their prior positions. Blocking is not propagated into a second conflict-resolution pass. A legal non-`stay` move outside the conflict set has status `moved`; `move/stay` has status `none`; an out-of-grid or conflict-blocked non-`stay` move has status `invalid`.

`inspect`, `claim`, `execute`, and `abandon` use the beginning-of-slot position and remain in place. An inspection is legal only at the cell of a task whose state is `available` or `claimed`. Each robot-task pair permits one inspection attempt in the entire scenario, and an invalid attempt also uses that allowance. A successful inspection has status `inspected` and reveals the requirement after the action. A claim may be submitted only when the robot knows the requirement, its capability matches, and it has not made a claim attempt for that task before; an invalid attempt also uses that allowance. Claim submission does not require the robot to be at the task cell. The environment accepts a submitted claim only if the task is actually available and unowned at claim resolution. A claim based on a stale null-claim fact therefore has an `invalid` result when another robot already owns the task, and the existing owner is never replaced. A robot may still submit a claim when its local fact names another claimant; that submission is permitted, counted as a conflict cost, and accepted only if the actual task has since become available. If an unowned task receives multiple acceptable same-slot claims, the lowest robot ID wins and all other claim results are `invalid`. After movement resolution, task actions are resolved in this fixed order: all inspections, then all claims, then executions and abandonments. An execution is legal for the current owner at the task cell and completes the task only if the capability matches and the action does not fail. A valid execution has status `executed`, and a deterministic execution failure has status `failed`. An abandonment is legal for the current owner regardless of its position, releases the claim, and has status `abandoned`. Any task action that does not meet its stated conditions has status `invalid`; `wait` has status `none`. A task completed at or before its deadline succeeds; active tasks not completed after their deadline become expired. Invalid actions do not change task state or local facts. A choice that violates a precondition available in the robot's pre-decision state is an illegal decision. When the available preconditions are satisfied but hidden environment state, such as an unobserved completion or stale owner, rejects the action, the decision is still an admissible submission and its recorded action result is `invalid`. The environment and independent replay are deterministic; for fixed scenario inputs and observed history, the submitted procedure must make deterministic choices and must not use unrecorded randomness.

#### Information availability and event order

Local facts change only through message merging and the acting robot's successful task-action outcome. A successful inspection at slot `t` writes `{task,requirement,claim_robot,version:t}`, where `claim_robot` is the owner before same-slot claim resolution. An accepted claim writes the winner's local fact with `claim_robot` equal to that winner and `version=t`. A successful execution, a failed execution, or a successful abandonment writes the acting robot's local fact with `claim_robot=null` and `version=t`; invalid actions write no fact. Completion and expiration do not otherwise create facts. These post-action updates can be used from the next slot and cannot enter the current slot's pre-decision record or broadcast.

The complete slot order is: expire tasks whose deadline was passed, deliver and merge messages, release tasks, record pre-decision state and choose actions/summaries, resolve movement, inspections, claims, executions and abandonments in the order above, expire unfinished tasks at their inclusive deadline, create the already-selected broadcast events from beginning-of-slot positions, and record the post-action task timeline. Record semantics are fixed as follows: `action`, `move`, `task`, `broadcast`, and `broadcast_entries` are the candidate submission at the decision boundary; `position`, `known_facts`, and `received_message_ids` are the information visible at that boundary; `inspection_result` and `action_result` are post-event outcomes; and task timelines and message records are post-resolution evidence. The detailed event order above remains authoritative.

#### Legal actions and resource bounds

Communication is independent of the task action. A broadcast can contain at most four distinct currently known facts. Each robot may broadcast at most 18 times, in any five consecutive slots at most twice, and may not broadcast in adjacent slots. A link at send slot `t` exists when the Manhattan distance between the sender and receiver at the beginning of the slot is at most 5 and `t` is not in that scenario's blackout set. The blackout sets are `dense_arrival:{7,8}`, `capability_bottleneck:{12,13,14}`, `relay_gap:{4,5,6,20,21}`, `late_urgent:{0,1,2,15,16}`, and `balanced:{10,11,24}`. The blackout table and other robots' current positions are available to the environment replay but not to the sending policy; the current target set is determined by the environment and recorded only in the resulting event, rather than supplied as a pre-decision policy input. One broadcast creates one event with all currently linked receivers as `targets`; delivery is at the next slot, with no partial delivery. `delivered_to` must equal `targets` in increasing robot order. An event with no target is `lost`; otherwise it is `delivered`. Events arriving after slot 39 have no effect. A broadcast summary is selected before the current action, so a fact revealed by the same slot's inspection cannot be included in that slot's summary.

The environment has deterministic action-failure events that are not available to policy state. An `execute` at slot `t` fails exactly when `t` belongs to the scenario set below and `(t+robot+task) mod 4 = 0`; a failed execution releases the claim and leaves the task active: `dense_arrival:{9,21}`, `capability_bottleneck:{8,17,28}`, `relay_gap:{11,22}`, `late_urgent:{18,27}`, and `balanced:{13,25}`.

#### Task metrics

The official metrics are computed only after all validity checks pass. `primary_metrics_1` is the mean across scenarios of completed tasks before or at their deadlines divided by 12. `primary_metrics_2` is the mean over all tasks and scenarios of `clip(1-(complete_t-release_t)/(deadline_t-release_t),0,1)` for completed tasks with a positive window, and zero otherwise. `auxiliary_metrics_1` is the mean over scenarios of `clip(1-n_broadcast/72,0,1)`. `auxiliary_metrics_2` is the mean over scenarios of `0.5*clip(1-n_avoidable_inspect/48,0,1) + 0.5*clip(1-n_conflicting_claim/48,0,1)`; an avoidable inspection is one performed when the robot's pre-decision local facts already contain that task or when the public deadline is already past. A task completed by another robot is not counted solely from hidden environment state when the acting robot has received no fact for it. A conflicting claim is one whose pre-decision local fact names a different robot in `claim_robot`, whether or not that fact is stale relative to environment truth. The normalized task performance is `(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`.

#### Result recording contract

The root object of `results.json` has exactly `scenarios`. Each scenario object has exactly `scenario`, `robots`, `tasks`, and `messages`, in the canonical scenario order above. `robots` contains exactly 160 records, one per `(robot,t)`, each with exactly: `robot` (integer `0..3`), `t` (integer `0..39`), `position` (two integer coordinates), `capability` (`red`, `green`, or `blue`), `action` (the six action values above), `move` (the five movement values), `task` (integer `0..11` or `null`), `known_facts` (sorted distinct facts), `received_message_ids` (sorted strings for messages arriving in this slot), `broadcast` (boolean), `broadcast_entries` (at most four distinct facts), `inspection_result` (the successful action's `{task,requirement,version}` or `null`), and `action_result` (`{status,task}` with status `none`, `moved`, `inspected`, `claimed`, `executed`, `failed`, `abandoned`, or `invalid`). In `action_result`, `task` is null for `none` and `moved`; for every task action it is the selected task ID even when the resulting status is `failed` or `invalid`. Facts and entries have exactly `{task,requirement,claim_robot,version}`, with `claim_robot` an integer `0..3` or `null`, and `version` an integer `0..39`. `known_facts` is sorted by increasing task ID; `received_message_ids` is sorted lexicographically.

`tasks` contains exactly 12 records with exactly `task`, `position`, `release_t`, `deadline_t`, `requirement`, `timeline`, `claimed_by`, `inspect_t`, `complete_t`, and `completion_reason`. `timeline` contains exactly 40 `{t,state,claimed_by}` records, where state is one of `unreleased`, `available`, `claimed`, `completed`, and `expired`; terminal fields are integers or `null`, and `completion_reason` is `completed` or `expired`. `inspect_t` is the first slot with a successful inspection by any robot; `claimed_by` is the owner after the final slot (and is `null` after completion or expiry); `complete_t` is the successful execution slot, or `null`; and `completion_reason` records the terminal outcome. The requirement in this post-action task evidence is environment truth and is never a pre-decision input.

`messages` is a variable-length list with at most one record per robot and send slot. Each record has exactly `message_id` (`r{robot}_t{t}`), `sender`, `send_t`, `targets`, `delivered_to`, `entries`, `arrival_t=send_t+1`, and `status`. `entries` have the same fact structure as above and preserve sender order. All arrays, keys, domains, finite values, and `1e-6` comparisons are checked from the records; the grader independently derives task state, link targets, message arrival, claim resolution, action effects, timelines, and metrics.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must create or overwrite UTF-8 `results.json` without interactive input. `report.md` must contain non-empty Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. The report must describe the actual implementation, results, communication-control trade-offs, and evidence-supported limitations.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable" , weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json has the declared strict root and scenario structure", weight: 3}
- {id: result_evidence, desc: "robot, task, action-result, timeline, and message records provide typed replay evidence", weight: 4}
- {id: canonical_coverage, desc: "all scenarios, robot-slot records, task records, and timeline slots are uniquely covered", weight: 4}
- {id: decision_legality, desc: "actions, movement, information, task-pair, and broadcast constraints are legal", weight: 4}
- {id: replay_validity, desc: "independent replay matches arrivals, knowledge, actions, claims, task state, and messages", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed on-time task completion rate", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed deadline-weighted task utility", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed broadcast communication efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed avoidable inspection and conflicting-claim control", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the heterogeneous robot, hidden requirement, delayed message, task lifecycle and objective model"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent task-allocation model or materially wrong process; Implementation/evidence: no corresponding executable process."}
    - {score: 0.4, desc: "Report: omits a decisive capability, deadline, information-boundary or communication relation; Implementation/evidence: partial or inconsistent process."}
    - {score: 0.7, desc: "Report: accurately defines heterogeneous robots, hidden requirements, task lifecycle, delayed summaries, constraints and objectives; Implementation/evidence: core model runs across all scenarios."}
    - {score: 1.0, desc: "Report: precisely explains requirement propagation, stale claim conflicts, event order, failures, deadlines and communication trade-offs; Implementation/evidence: records faithfully expose these relations."}
- id: method_design
  desc: "quality and reproducibility of the unified non-training allocation and communication-trigger policy"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable or legal allocation method; Implementation/evidence: no unified procedure."}
    - {score: 0.4, desc: "Report: mainly uses nearest-task or fixed-period rules while ignoring capability, deadline or coordination coupling; Implementation/evidence: scenario-specific tables or divergence from the report."}
    - {score: 0.7, desc: "Report: gives a legal reusable state-driven allocation and trigger policy; Implementation/evidence: one procedure runs all scenarios and supports key decisions."}
    - {score: 1.0, desc: "Report: justifies a unified policy maintaining commitments, capability knowledge, urgency, conflict risk and communication budget; Implementation/evidence: actions are history-driven without prefilled schedules."}
- id: experimental_coverage
  desc: "coverage of canonical scenarios and controlled communication, delay, capability and deadline comparisons"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only isolated or aggregate results without testing communication or hidden capability propagation; Implementation/evidence: records do not support claims."}
    - {score: 0.7, desc: "Report: covers all canonical scenarios and at least one controlled trigger, delay, capability or budget comparison; Implementation/evidence: comparable records support both."}
    - {score: 1.0, desc: "Report: uses complementary mechanism, communication-pressure and deadline/capability boundary conditions; Implementation/evidence: condition-level task, message and failure records distinguish the claims."}
- id: research_analysis
  desc: "evidence-supported explanation of capability, delay, deadline and communication trade-offs and limitations"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: absent or contradicts records; Implementation/evidence: conclusions unsupported."}
    - {score: 0.4, desc: "Report: repeats counts or makes unsupported causal claims and omits important limitations; Implementation/evidence: insufficient evidence."}
    - {score: 0.7, desc: "Report: explains capability, delay, deadline and communication trends from records, separates observation from inference and states limitations; Implementation/evidence: supports the trends."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies sensitivities and failure boundaries for event triggers and stale claims with scope and unknowns; Implementation/evidence: reproducible condition-level records support the conclusions."}
