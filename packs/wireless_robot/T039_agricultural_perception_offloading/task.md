---
id: T039_agricultural_perception_offloading
name: Deadline-Aware Edge Offloading for Battery-Limited Robotic Harvesting
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

Two battery-limited agricultural robots harvest targets in a bounded field. A target can be harvested only after one robot has observed it and obtained a sufficiently confident perception result. Processing an observation locally is prompt but energy-intensive and may not be accurate enough under age or occlusion. Compact and full edge requests can improve confidence, but their fragments share a lossy uplink and their completed uploads compete for one FIFO edge service. Queue bursts, weak-link positions, retransmissions, target deadlines, and movement therefore determine whether a perception result remains useful when a robot reaches its target.

### Research Problem

Design and evaluate one deterministic, non-learning, state-driven method that jointly selects targets and paths, chooses local or edge processing, schedules submissions and one-shot retransmissions, cancels obsolete work, and decides when to harvest. The method must use only each robot's available local information and the public edge-load feedback at the action boundary. It should balance deadline-qualified harvest value, perception latency and confidence, safe and efficient motion, transmitted load, queue delay, and onboard energy. One reusable procedure must handle all fifteen public scenario variants; it may read their task inputs, but it must not use scenario-, variant-, robot-, target-stage-, or absolute-slot-indexed action tables.

### Experimental Setting and Constraints

#### Canonical evaluation units

The field is the integer grid `x=0..17`, `y=0..13`. Each unit has two robots, `r0` and `r1`, one edge server at `E=(9,7)`, eight targets, `H=72` one-second action slots `0..71`, and state boundaries `0..72`. The five base scenarios are `clear_rows`, `occluded_cluster`, `edge_queue_burst`, `weak_link_turn`, and `deadline_competition`; each has `variant v` in `{0,1,2}`. These fifteen `(scenario,variant)` pairs are the canonical evaluation units, with canonical key `scenario=<name>|variant=<v>`. Every unit starts from the table entry and terminates at boundary `72`.

Target IDs `T0..T7` have fixed `(value,deadline_state)` values `(1,24),(1,30),(2,36),(2,42),(3,50),(3,56),(4,64),(4,70)`. The coordinate lists below are in target-ID order. A closed obstacle rectangle includes its boundary and every integer cell in the listed Cartesian product.

| scenario | starts `(r0,r1)` | closed obstacle rectangles | coordinates of `T0..T7` | additional condition |
|---|---|---|---|---|
| `clear_rows` | `(1,2),(1,11)` | none | `(3,2),(3,11),(7,3),(7,10),(12,2),(12,11),(16,4),(16,9)` | none |
| `occluded_cluster` | `(2,7),(15,7)` | `[8,9]x[4,10]` | `(3,2),(3,11),(12,2),(12,11),(7,3),(7,10),(16,4),(16,9)` | occlusion enters observation and link quality |
| `edge_queue_burst` | `(2,2),(2,11)` | `[6,7]x[5,9]` | `(3,11),(7,3),(7,10),(12,2),(12,11),(16,4),(16,9),(3,2)` | edge service is `240 bit/slot` in slots `24+3v..35+3v`, inclusive |
| `weak_link_turn` | `(1,6),(16,6)` | `[8,9]x[0,4]`; `[8,9]x[9,13]` | `(3,11),(3,2),(7,10),(7,3),(12,2),(12,11),(16,4),(16,9)` | link success probability loses `0.15` when the sender has `7<=x<=10` |
| `deadline_competition` | `(2,3),(2,10)` | `[5,6]x[5,8]` | `(3,2),(7,3),(12,2),(16,4),(3,11),(7,10),(12,11),(16,9)` | link success probability loses `0.20` in slots `12+3v..19+3v`, inclusive |

#### State, actions, and task model

At each action boundary, each robot observes every unharvested target within Euclidean distance `3`. Observation ID is `obs=<scenario>|variant=<v>|robot=<i>|target=<Tj>|slot=<t>`, where `<i>` is the numeric character `0` or `1`, not `r0` or `r1`; for example, `obs=clear_rows|variant=0|robot=1|target=T3|slot=12`. An observation fixes its generation boundary, robot and target positions, `d_o=clip(||p_robot-p_target||_2/3,0,1)`, and an occlusion indicator `o`. Here `o=1` exactly when the closed line segment between the robot and target centers intersects any closed obstacle rectangle, including its boundary; otherwise `o=0`. A robot retains all of its own observations. If it references an observation at slot `t`, its age is `a=t-observation_slot` and its current quality is

`q=clip(1-0.12*d_o-0.20*o-0.04*a-0.04*v,0,1)`.

Each `(slot,robot)` has one mutually exclusive action kind: `idle`, `move`, `submit`, `cancel`, `retry`, or `harvest`. The move deltas are exactly `N=(0,+1)`, `E=(+1,0)`, `S=(0,-1)`, and `W=(-1,0)`. Each attempted move consumes `1` energy. The fixed safety executor processes `r0` then `r1`. A proposed destination is unsafe if it is outside the grid, is an obstacle cell, equals the executor-selected final cell of a lower-ID robot in this slot whether that robot moved or stayed, or equals the action-start cell of a higher-ID robot. An unsafe move becomes no movement and creates one safety event; the same attempted move cannot create multiple safety events.

A harvest action consumes `3` energy. At action start, the robot must be at the target coordinate and have at least one result for that target with confidence at least `0.75` and completion age at most `4`. The executor selects the eligible result with greatest confidence, then earliest completion boundary, then lexicographically smallest job ID. If both robots legally harvest the same still-unharvested target in one slot, the lower robot ID succeeds and the other attempt fails without becoming a safety event. Harvest occurs at boundary `t+1`; it is on time exactly when `t+1<=deadline_state`. Both robots start with `90` energy units.

A `submit` action must reference an observation already cached by that robot at action start, name the same target as the observation, occur at a slot strictly below the target deadline, and leave at most two unfinished jobs owned by that robot. Its fixed ID is `job=<scenario>|variant=<v>|robot=<i>|slot=<t>`, with the same numeric `0/1` convention for `<i>` as observation IDs. The three modes are:

| mode | upload fragments | edge service | confidence | onboard energy |
|---|---:|---:|---|---:|
| `local` | none | none | `clip(0.40+0.36*q,0,1)` | `5` at submission |
| `edge_compact` | one `480 bit` fragment | `480 bit` | `clip(0.55+0.28*q,0,1)` | `1` per admitted fragment attempt |
| `edge_full` | two `480 bit` fragments | `960 bit` | `clip(0.72+0.22*q,0,1)` | `1` per admitted fragment attempt |

A local job completes at the next boundary. A cancel action may reference any job of the acting robot whose action-start status is `local_pending`, `ready`, `await_retry`, `queued`, or `active`. Cancellation returns no bits or energy; canceling an active job does not start a replacement until the next boundary.

#### Information availability and event order

At action start, robot `ri` may use the public scenario and target tables; its own current position and energy; its own cached observations, unfinished-job states, fragment-failure feedback, queue feedback, completed results, and harvest history; targets within its current sensing radius that are already harvested; the edge service's current broadcast `queue_load=active+waiting`; and its past local history. It may not use the other robot's post-start dynamic position, energy, observations, jobs, results, or actions, nor any future hash outcome, queue event, completion, harvest, or terminal state. The fixed inputs are public, but a policy cannot treat a future fixed hash as an already observed link outcome.

The action-start information has the following exact public schema; collection order has no meaning. Each current or cached owned observation exposes `observation_id,target,observation_slot,robot_position,target_position,d_o,o`. IDs and targets are strings; the slot is an integer in `0..t`; each position is a length-two integer coordinate; `d_o` is finite in `[0,1]`; and `o` is integer `0` or `1`. Current observations are the subset generated at this boundary. Each owned job exposes `job_id,target,mode,observation_id,submit_slot,deadline,status,next_fragment,failure_due,queue_entry,service_start`. Its IDs, target, mode, and status are strings; slots and deadlines are integers. Action-start `status` is one of `local_pending,ready,await_retry,queued,active,completed,canceled,failed,rejected`; the first five are unfinished. `next_fragment` is null for local jobs and otherwise an integer from `0` through the mode's fragment count, equaling that count after the final upload. `failure_due` is null until an attempt-0 failure and otherwise gives its feedback boundary. `queue_entry` and `service_start` are null until those events occur and otherwise give their integer boundaries. A current-boundary attempt-0 failure exposes `job_id,fragment_id,fragment_index,attempt,outcome,feedback_boundary`: its two IDs and outcome are strings, the other fields are integers, `attempt=0`, and `outcome` is one of `capacity,budget,energy,loss`. This feedback is visible only at its stated boundary. Each owned completed result exposes string `job_id,target,mode,observation_id`, integer `observation_slot,completion_boundary`, and finite `confidence` in `[0,1]`. `known_harvested` is the set of target-ID strings already known to that robot, and `queue_load` is the integer number of active plus waiting edge jobs in `0..3`. These records define policy inputs at action start and are not fields of `results.json`.

At each nonterminal boundary `t`, the fixed process performs these steps in order:

1. complete local or edge work due at `t` and expose its result;
2. expose outcomes for fragments sent in slot `t-1`, advance successful jobs, and admit fully uploaded jobs to the edge FIFO;
3. start the FIFO head if the service is idle;
4. update locally known harvest state, result ages, own queue feedback, the queue-load broadcast, and current observations;
5. expose this action-start state and read both actions simultaneously;
6. validate references and reserve explicit move, harvest, or local-compute energy, then apply cancellations;
7. form new, automatic-next, and requested-retry fragments; order, admit, charge, and sample them;
8. advance the active edge service for this slot;
9. execute harvests and safety-filtered moves in robot-ID order; and
10. update target, energy, safety, and task state and record boundary `t+1`.

Boundary `72` performs only completions, previous-fragment outcomes, and FIFO admission due from slot `71`; it has no action or service progress, and later events provide no benefit. Submitted fields are action-start decisions. The observation, feedback, job, result, known-harvest, and queue-load views above contain only information already visible after steps 1--4 of that boundary. Environment transitions, fragment events, full queue state, positions, harvests, and metrics are fixed-replay evidence produced during or after execution; none of these derived records is submitted in `results.json`.

#### Legal actions and resource boundaries

A new edge job sends fragment `f0` in its submission slot. After a successful nonfinal fragment, the next fragment is offered automatically in the following slot. Failure feedback is visible at the next boundary. Only at that boundary may the robot use `retry(job_id)` to offer the same fragment with `attempt=1`; omitting that retry or failing attempt 1 terminates the job. Retry is legal only for the acting robot's attempt-0 failure, in that one-slot window, and before the target deadline. Fragment ID is `<job_id>|f<index>`.

All fragment offers in a slot are ordered by `(submit_slot,deadline,robot_id,job_id,fragment_index,attempt)`. Shared uplink capacity is `960 bit/slot`, and each canonical unit has a total admitted-bit budget of `9600 bit`. Capacity, budget, or remaining transmission energy can reject an offer and produces failure feedback; those fixed outcomes are not illegal decisions. Every admitted attempt consumes `480 bit` and `1` energy even if its link sample fails. Explicit action energy is reserved before fragment admission. An edge submit or retry therefore requires at least `1` energy at action start; an automatically offered fragment with insufficient post-reservation energy receives an energy failure rather than making the current action illegal.

For the zero-based `n_t`-th admitted fragment in a slot, let `d_n=clip(||p_robot-E||_2/18,0,1)` and use the submitted observation's fixed occlusion `o`. Then

`R_t=clip(960/(1+2*d_n+0.2*n_t+0.15*o),120,960)` and `p=clip(R_t/960-b_t,0.05,1)`,

where `b_t` is `0.15` only for a `weak_link_turn` sender with `7<=x<=10`, `0.20` only in the public `deadline_competition` interval, and `0` otherwise. The sample is

`u=uint64(SHA256(UTF-8("<scenario>|<variant>|<fragment_id>|a<attempt>|salt=<5v>"))[:16])/2^64`.

The whole fragment succeeds exactly when `u<p`. The first 16 hexadecimal SHA-256 digits are the first 64 bits, interpreted as an unsigned big-endian integer.

On the boundary after its final fragment succeeds, an edge job is considered for a FIFO whose capacity `3` includes the active and waiting jobs. Simultaneous completed uploads are ordered by `(submit_slot,deadline,robot_id,job_id)`; excess jobs are rejected and terminated. At a boundary, the previous active service completes first, uploads are then admitted, and an idle server starts the FIFO head. Service subtracts `480 bit` in a normal action slot and `240 bit` in the public queue-burst interval. A result appears on the boundary after remaining service first becomes nonpositive. Completed results remain recorded, but a harvest cannot use one when its age exceeds `4`, the target is locally known as harvested, or the robot is not at that target.

The legality gate checks only the declared action shape, action-start local references, unfinished-job capacity, retry window, harvest eligibility, and explicit energy requirements. A legal action may still experience uplink or FIFO rejection, a same-slot harvest conflict, an expired result later, or safety filtering; these are fixed execution outcomes and never become hidden legality requirements.

#### Task metrics

Compute every metric per canonical unit and then take the arithmetic mean over all fifteen units. For `primary_metrics_1`, each target earns its value times `1` if harvested by its deadline, `0.5` if it is not harvested on time but some result of confidence at least `0.75` completes by its deadline, and `0` otherwise. Divide by total target value `20`.

For `primary_metrics_2`, use the result actually selected by an on-time harvest. Otherwise use the earliest result that completes by the deadline with confidence at least `0.75`, breaking a completion tie by greater confidence and then smaller job ID. Let `delay=completion_boundary-observation_slot`. The use boundary is the harvest boundary for an on-time harvest and the completion boundary otherwise; `age_at_use=use_boundary-completion_boundary`. A target contributes

`0.6*clip(1-delay/max(1,deadline-observation_slot),0,1) + 0.4*clip((confidence-0.75)/0.25,0,1)*clip(1-age_at_use/4,0,1)`.

A target with no qualifying result contributes zero. Weight targets by value and divide by `20`.

For `primary_metrics_3`, let

`safe_harvest_ratio=(successfully_harvested_targets/8)*clip(1-safety_events/(2*72),0,1)`.

For each robot, follow its actual successful-harvest order and sum the four-neighbor shortest path distances, around closed obstacle cells, from its start to its first harvested target and between successive harvested targets. Sum these values as `shortest_cost`. Let `successful_moves` be the number of executed one-cell moves over the whole unit. `motion_efficiency=clip(shortest_cost/max(1,successful_moves),0,1)`, or `0` when no target is harvested. The metric is `0.6*safe_harvest_ratio+0.4*motion_efficiency`.

For `auxiliary_metrics_1`, `total_edge_bits` is `480` times every admitted edge-fragment attempt, including failed samples and retries. For each target, identify its first result of confidence at least `0.75` that completes by its deadline using the same completion/confidence/job-ID tie order. If that result is an edge job, its successfully delivered original payload, `480` or `960` bits, contributes to `useful_edge_bits`; retry overhead is not useful a second time. A local or absent first result contributes zero. The metric is

`0.6*useful_edge_bits/max(total_edge_bits,1)+0.4*clip(1-total_edge_bits/9600,0,1)`;

the useful fraction is zero when no edge bits are sent.

For `auxiliary_metrics_2`, `total_local_compute_energy` is `5` per local submission. For each job admitted to the edge FIFO, queue wait is `service_start-queue_entry`; a job not started by termination uses `72-queue_entry`. The empty mean is zero. The metric is

`0.5*clip(1-total_local_compute_energy/(2*90),0,1)+0.5*clip(1-mean(queue_wait_slots)/72,0,1)`.

All metrics lie in `[0,1]`. Normalized task performance is the weighted mean of the five metrics with weights `8,4,3,3,2`. Floating comparisons use absolute tolerance `1e-6`; scores come only from fixed replay, not reported aggregates.

#### Result record contract

`results.json` must be strict finite JSON with exactly the root key `scenarios`. Its array must contain exactly one record for each canonical unit. Every unit record has exactly `scenario` (string), `variant` (integer), and `actions` (array). Each actions array has exactly one record for every `(slot,robot)`, hence `72*2=144` records. Record order and JSON whitespace are immaterial.

Every action has exactly these fields: `slot`, `robot`, `kind`, `direction`, `target`, `mode`, `observation_id`, and `job_id`. `slot` is an integer in `0..71`; `robot` is `r0` or `r1`; `kind` is one of the six action kinds. The action-specific non-null fields are exact:

| kind | required non-null fields | fields that must be null |
|---|---|---|
| `idle` | none | `direction,target,mode,observation_id,job_id` |
| `move` | `direction` in `N,E,S,W` | `target,mode,observation_id,job_id` |
| `submit` | `target` in `T0..T7`; `mode` in `local,edge_compact,edge_full`; `observation_id` | `direction,job_id` |
| `cancel` | `job_id` | `direction,target,mode,observation_id` |
| `retry` | `job_id` | `direction,target,mode,observation_id` |
| `harvest` | `target` in `T0..T7` | `direction,mode,observation_id,job_id` |

Only these action decisions are submitted. Observations, feedback, jobs, fragments, queues, results, target states, robot states, and metric summaries are fully derived from the public model and must not be added as required result fields.

Validation responsibilities are non-overlapping. `result_evidence` checks only the exact unit/action fields, primitive types, action-kind enums, and action-specific nullability in this contract. `canonical_coverage` checks the fifteen unit keys and all `72*2` `(slot,robot)` keys. `decision_legality` checks action-start observation and job references, unfinished-job capacity, retry timing, harvest eligibility, and explicit action energy. `replay_validity` checks the deterministic state and event reconstruction. A representation that satisfies these public semantics is not required to match any private policy or internal data layout.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the unified policy, all canonical units, controlled processing-path or communication evidence, and supported limitations. The code and report must make it possible to inspect whether decisions are produced causally from the declared action-start information rather than from another robot's private state or future fixed outcomes. Additional project files and report sections are allowed, but external services, hidden files, and unpublished reference code are not.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains six non-empty required headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenario container", weight: 3}
- {id: result_evidence, desc: "scenario and action records contain the declared minimal decision fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen variants and slot-robot action keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "actions obey local visibility, job, retry, result and explicit-energy rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives observations, processing, uplink, queue, harvest and motion", weight: 6}
- {id: primary_metrics_1, desc: "deadline-qualified weighted harvest completion", weight: 8}
- {id: primary_metrics_2, desc: "perception-to-harvest latency and confidence quality", weight: 4}
- {id: primary_metrics_3, desc: "safe harvest and motion efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful edge payload and admitted-uplink efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "onboard compute-energy and edge-queue efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the harvesting, observation, processing, uplink, queue and energy model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent coupled formulation; Implementation/evidence: no executable harvesting and processing process corresponds to the task."}
    - {score: 0.4, desc: "Report: covers only motion or offloading; Implementation/evidence: omits a decisive observation-age, queue, result-validity, deadline, or energy relation."}
    - {score: 0.7, desc: "Report: accurately states the variants, local information, processing paths, uplink, FIFO, energy, and deadlines; Implementation/evidence: one reusable process implements the core model with limited omissions."}
    - {score: 1.0, desc: "Report: precisely closes the observation, decision, fragment, service, result, motion, and harvest order and its trade-offs; Implementation/evidence: all variants are faithfully realized with traceable actions and no material divergence."}
- id: method_design
  desc: "quality of the unified target, processing, retransmission, cancellation, harvest and motion policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or uses prohibited training; Implementation/evidence: no legal unified procedure."}
    - {score: 0.4, desc: "Report: relies on an always-local, always-edge, fixed target, or fixed-slot rule and ignores a decisive queue, age, deadline, or energy consequence; Implementation/evidence: actions are pre-filled by unit or materially differ from the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning target, path, processing, retry, cancellation, and harvest policy; Implementation/evidence: one procedure covers all variants using permitted current state and history."}
    - {score: 1.0, desc: "Report: justifies joint decisions using observation value, processing confidence and delay, link load, queue state, result age, remaining energy, reachability, and deadlines; Implementation/evidence: current local state drives decisive actions without unit or slot action tables."}
- id: experimental_coverage
  desc: "coverage of canonical variants, processing mechanisms, communication conditions and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only a single condition or aggregate is shown; Implementation/evidence: no evidence that processing or communication changes harvest outcomes."}
    - {score: 0.7, desc: "Report: covers all variants and one controlled processing-path, link, queue, occlusion, deadline, or energy comparison; Implementation/evidence: both comparable conditions are executed with condition-level evidence."}
    - {score: 1.0, desc: "Report: complementary single-factor controls separately test the processing-path mechanism and a communication, deadline, or energy failure boundary; Implementation/evidence: each condition is executed and reconstructible beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based interpretation of processing, communication, queue, energy and harvest trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: conclusions are unsupported."}
    - {score: 0.4, desc: "Report: mainly lists completion, delay, or energy values without causal explanation or material limits; Implementation/evidence: condition evidence is insufficient."}
    - {score: 0.7, desc: "Report: explains observed processing, link, queue, result-age, movement, energy, and deadline trends and states limitations; Implementation/evidence: actual records support those claims."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify the interaction among confidence, latency, queue competition, reachability, and energy while separating observation, inference, and unknowns; Implementation/evidence: reproducible condition records support the central conclusions."}
