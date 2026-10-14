---
id: T022_dynamic_delivery_task_allocation
name: Communication-Limited Online Task Allocation with Delayed Travel Outcomes
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

Three delivery hubs coordinate six robots that serve requests with release times and hard service windows. A hub senses only requests inside its fixed coverage region. Overlapping regions create competition, while non-overlapping regions leave some requests private. Directed hub links carry claim, cancellation, and completion intentions through small FIFO queues; delayed or lost intentions can leave hubs with different beliefs about the same request.

Each robot must return to its home hub before accepting another request. Its outbound and return durations include an execution residual that is not known until that leg ends. A locally reasonable claim can therefore occupy a robot longer than expected, miss a window, or duplicate a commitment at another hub. Successful coordination must connect online allocation, message timing, uncertain robot occupancy, service windows, and return availability rather than treating communication as a detached score.

### Research Problem

Build the deterministic simulator below and design one unified, non-learning online procedure for all five canonical scenarios. At each decision boundary, each hub chooses claims for its currently idle robots and may enqueue intentions about already observed events on its currently available outgoing links. The procedure receives only that hub's released and sensed tasks, home-robot states, arrived intentions, outgoing queues, current links, and history. Claims at different hubs are simultaneous and cannot inspect one another.

The method should select valuable tasks that can finish within their windows, account for delayed outbound and return occupancy, and communicate intentions early enough to suppress avoidable duplicate claims without flooding the queues. Evaluate how task overlap, directed information flow, bursty releases, tight windows, and travel uncertainty change timely value and communication efficiency. Deterministic rules, heuristics, search, planning, or optimization are permitted; algorithm choice and parameters remain yours.

### Experimental Setting and Constraints

#### Canonical scenarios, hubs, and task instances

The five scenarios, in fixed zero-based `scenario_index` order, are `balanced_stream`, `overlap_hotspot`, `sparse_link`, `tight_windows`, and `burst_arrivals`. There are decision slots `t=0,...,59` and boundaries `0,...,60`; one slot is one minute.

Hub positions are `H0=(1,1)`, `H1=(11,1)`, and `H2=(6,10)`. Robots `0,1` belong to `H0`, robots `2,3` to `H1`, and robots `4,5` to `H2`. All hubs have Euclidean sensing radius `7.25`. A task is sensed by a hub exactly when its position is within that closed radius. The following position codes are used by the scenario table; the final column states the resulting hub set and is normative.

| code | position | sensing hubs |
|---|---|---|
| `A` | `(2,2)` | `0` |
| `B` | `(3,2)` | `0` |
| `C` | `(4,4)` | `0,2` |
| `D` | `(5,4)` | `0,1,2` |
| `E` | `(6,4)` | `0,1,2` |
| `F` | `(7,4)` | `0,1,2` |
| `G` | `(8,4)` | `1,2` |
| `H` | `(9,2)` | `1` |
| `I` | `(10,2)` | `1` |
| `J` | `(5,8)` | `2` |
| `K` | `(6,9)` | `2` |
| `L` | `(7,8)` | `2` |
| `M` | `(3.5,5)` | `0,2` |
| `N` | `(8.5,5)` | `1,2` |
| `O` | `(6,2)` | `0,1` |
| `P` | `(5,5)` | `0,1,2` |
| `Q` | `(6,5)` | `0,1,2` |
| `R` | `(7,5)` | `0,1,2` |

Every scenario has tasks `K00,...,K17`. For task index `k=0,...,17`, the value is respectively `[4,4,3,5,5,5,3,4,4,3,5,4,2,2,4,5,3,5]`; the total task value is `70` in every scenario. Each row below contains four parallel arrays of length 18, ordered by task index: position code, release boundary, inclusive service-window start, and inclusive latest completion boundary.

| scenario | position codes | release | window start | window end |
|---|---|---|---|---|
| `balanced_stream` | `A,I,K,C,G,E,M,N,J,L,O,P,D,F,B,H,Q,R` | `0,0,0,4,5,7,10,11,13,14,17,19,22,23,26,28,32,36` | `1,1,2,6,7,9,12,13,15,16,19,21,24,25,28,30,34,38` | `10,10,14,18,19,22,26,27,30,31,33,35,39,40,43,45,50,56` |
| `overlap_hotspot` | `D,E,F,P,Q,R,D,E,F,P,Q,R,D,E,F,P,Q,R` | `0,0,0,0,1,1,10,10,10,11,11,11,22,22,23,23,24,24` | `2,2,2,3,3,3,12,12,12,13,13,13,24,24,25,25,26,26` | `14,14,15,15,16,16,25,25,25,26,26,27,38,38,39,39,40,40` |
| `sparse_link` | `D,E,F,C,G,P,Q,R,O,M,N,D,E,F,G,P,Q,R` | `0,0,2,3,6,8,12,13,16,18,22,24,28,30,34,36,40,43` | `1,1,4,5,8,10,14,15,18,20,24,26,30,32,36,38,42,45` | `15,15,17,18,21,23,27,28,31,33,37,39,43,45,49,51,55,58` |
| `tight_windows` | `A,B,H,I,J,K,L,C,G,O,M,N,D,E,P,Q,R,F` | `0,0,0,1,4,5,6,8,10,12,14,17,20,23,27,31,35,40` | `0,1,1,2,5,6,7,9,11,13,15,18,21,24,28,32,36,41` | `6,7,7,7,11,10,13,18,20,22,25,28,31,33,37,40,45,51` |
| `burst_arrivals` | `A,B,H,I,K,P,D,E,F,P,Q,R,J,L,M,N,G,O` | `0,0,0,0,0,0,18,18,18,18,18,18,36,36,36,36,36,36` | `1,1,1,2,2,2,20,20,20,21,21,21,38,38,38,39,39,39` | `13,13,13,15,15,16,32,32,33,33,34,34,51,51,52,53,54,55` |

At its release boundary, a task's ID, position, value, release, and window become visible to every sensing hub and remain locally known. Before release, the row is environment truth and cannot be used by the policy. An intention about a task outside a receiver's sensing region may be stored as remote robot status but does not reveal that task's metadata or make it claimable.

#### Directed links and intention queues

The currently usable directed hub edges are:

| scenario and slots | directed edges |
|---|---|
| `balanced_stream`, all slots | all six ordered pairs of distinct hubs |
| `overlap_hotspot`, all slots | `(0,1),(1,2),(2,0)` |
| `sparse_link`, `0..19` | `(0,1),(1,2)` |
| `sparse_link`, `20..39` | `(2,1),(1,0)` |
| `sparse_link`, `40..59` | `(0,2),(2,0)` |
| `tight_windows`, all slots | `(0,2),(2,0),(1,2),(2,1)` |
| `burst_arrivals`, all slots | `(0,1),(1,2),(2,0)` |

Only the current row is visible at a decision boundary; later `sparse_link` rows are future topology and may not be used early. Each directed edge owns an independent FIFO queue of capacity two. In one slot, its sender may request at most one new intention for that receiver. A dynamically legal request is assigned the next zero-based environment `message_seq`. At one boundary, all dynamically legal requests are ordered by `(sender_hub,receiver_hub)` ascending before consecutive sequences are assigned; the one-request-per-receiver rule makes this order unique and independent of JSON array order. If that edge queue already contains two messages, the request is dropped but still counts as a generated intention. Otherwise it joins the tail.

After all enqueues in a slot, each currently usable nonempty edge exposes its head. At most four heads are served across the scene, selected by `(generated_boundary,sender_hub,message_seq)` ascending; one edge cannot serve more than its one head. Unselected heads and queues on currently unavailable edges remain FIFO. Every message is `32 bits`. A served message is lost exactly when

```text
(scenario_index + generated_boundary + message_seq) mod 9 = 0.
```

Otherwise it arrives at the receiving hub at boundary `t+1`, where `t` is its service slot. The current `message_seq`, service selection, and loss outcome are unavailable when enqueue and claim decisions are made. A message carries exact fields `(robot,task,status,event_boundary)` and no task metadata. `status` is `claim`, `cancel`, or `complete`. A hub may enqueue only an event already observed for one of its own robots before the current views were frozen; an earlier authentic event may be retransmitted. An accepted local claim at boundary `t` becomes available for enqueue decisions at boundary `t+1`, whereas cancellation and completion events processed before the views at boundary `t` may be enqueued at `t`.

Arrived events update the receiver's latest remote status for the stated `(robot,task)` only when their `event_boundary` is newer than its stored event, or when the boundary is equal and status precedence is `claim < cancel < complete`. A remote `claim` marks that task occupied at the receiver; a later `cancel` clears that robot's occupation; `complete` permanently marks the task complete. Locally observed events and task metadata never depend on message delivery.

#### Claims, delayed travel, service, cancellation, and return

At a decision boundary, an idle robot may submit `null` or claim one task that its home hub has sensed and released. The boundary must satisfy `t < window_end`. The hub must not have attempted that task before with either home robot, and its frozen local knowledge must contain no active remote claim or remote completion for it. A busy robot must submit `null`. A task that another home robot already claimed is locally attempted and unavailable. These are legality conditions based only on the frozen home-hub view, never on hidden commitments at other hubs.

Claims are simultaneous. If both home robots legally request the same locally available task in one slot, the lower robot ID receives the home-hub commitment; the other remains idle. Both requests are legal, and the losing request is a derived duplicate attempt. Different hubs may legally commit to the same task while their local information is split. Such duplication is not a legality failure.

For hub `h` and task position `(x,y)`, define the integer base travel duration

```text
base_h,k = ceil(|x-H_h.x| + |y-H_h.y|).
```

An outbound leg departing at boundary `b` and a return leg departing at boundary `b` each last `base_h,k + delta` slots, with `leg=0` outbound and `leg=1` return:

```text
delta = (17*(scenario_index+1) + 11*(robot+1) + 7*(task_index+1)
         + 5*(b+1) + 3*leg) mod 2.
```

The formula is fixed simulator truth, but `delta` for an uncompleted leg and its resulting arrival boundary are unavailable to the policy until that leg ends. The policy may use the base duration, elapsed time, and past observed leg outcomes, but not evaluate future residuals to choose a claim.

On outbound arrival, a robot may acquire service ownership only if two service slots can finish no later than `window_end`. The earliest eligible arrival immediately obtains the task's single service ownership and waits while retaining it if the window has not started; arrivals tied at one boundary use lower robot ID. A later or tied losing robot observes the conflict at its arrival, emits `cancel`, spends two complete cancellation slots, and then starts its return leg. The service owner begins at `max(arrival,window_start)`, completes at that boundary plus two, emits `complete`, and immediately starts its return leg. Completion cancels every other active commitment to that task at the same boundary; each canceled robot emits `cancel`, spends two cancellation slots, and returns. An arrival that cannot complete two service slots by `window_end` is likewise canceled immediately at its arrival boundary rather than waiting for window expiry.

At `window_end`, a service completion due exactly then is processed first and succeeds. If the task is still incomplete, all of its outbound, waiting, or service commitments are canceled before that boundary's views. A robot canceled by window expiry also spends two cancellation slots and then returns. A completed or canceled robot is not idle while returning; it becomes idle only when the return leg arrives at its home hub. Commitment, service ownership, conflict, travel residual, task completion, cancellation, and return events are all derived and are not submitted.

#### Authoritative boundary order and information

Immediately before boundary `0`, every robot is idle at its stated home hub with no task. Every task is incomplete, uncommitted, and unreleased until its listed release step. Every hub's attempted set, own event history, arrived remote-status store, policy history, and every directed-edge FIFO queue are empty. There are no pending message deliveries, and the next environment `message_seq` is `0`.

For each boundary `t=0,...,59`, the following order is authoritative:

1. Deliver successful messages served in slot `t-1` and apply their remote events. At boundary `0`, there are no messages.
2. Process return arrivals, service completions, outbound arrivals, waiting-to-service transitions, two-slot cancellation delays, and then window expiry. Record all resulting own-robot events.
3. Reveal tasks whose `release=t`, reveal each hub's current outgoing directed edges, and freeze one local view per hub. A hub sees its released sensed task metadata, permanent own attempted set, two home-robot states and observed events, latest arrived remote statuses, every outgoing queue, its current outgoing directed edges, and its own prior views and decisions.
4. Obtain the six robot claims and three hub enqueue actions. The three hub action tuples and all robot claims are simultaneous. Validate each claim only against its frozen home view, and each enqueue action only against its sender's frozen view.
5. Every dynamically legal non-null request, including a same-hub losing request, permanently marks that task attempted at its home hub. Resolve same-hub claim competition, launch accepted outbound legs, create claim events, then assign sequences to dynamically legal enqueue requests. For the derived duplicate count only, accepted claims are considered in ascending robot-ID order; an accepted claim is duplicate when an earlier active commitment from another hub or a completed task already exists, so simultaneous cross-hub commitments have one first claim and every additional claim is counted. This ordering does not change execution. A dynamically illegal robot claim is replaced by `null`; an enqueue action containing any illegal request is replaced by an empty array. These replacements support deterministic diagnostics, while the corresponding legality atom still fails.
6. Append requests or record queue-full drops, select and serve up to four FIFO heads, schedule their loss or boundary-`t+1` delivery, and advance one slot. Task and robot transitions due at `t+1` are processed only at the next boundary.

Boundary `60` performs steps 1 and 2 only, including completions and returns due exactly at 60; it has no decision record. No event after step 3 can change a same-boundary frozen view. Current other-hub decisions, future tasks, future topology, unarrived messages, hidden global commitments, service ownership, unobserved travel residuals, and current message loss outcomes are not policy inputs.

The submitted fields below are decision fields from step 4. Local views, task knowledge, messages, queues, commitments, travel, service, cancellation, returns, task outcomes, and metrics are execution or derived evidence reconstructed by the grader. This order is the sole timing authority. The process has no randomness. Integers reject Boolean values, and floating comparisons use absolute tolerance `1e-6`.

#### Task metrics

All four metrics are independently replayed, lie in `[0,1]`, and are higher-is-better. For each scenario, the denominator for task value is the fixed total `70`.

`primary_metrics_1` is completed task value divided by `70`, where a task contributes only when its service completion boundary is no later than its `window_end`; it is then averaged equally over the five scenarios.

For an on-time task with value `v`, start `a`, end `z`, and completion boundary `c`, its temporal value is `v*clip(1-(c-a)/(z-a),0,1)`. Unfinished or late tasks contribute zero. `primary_metrics_2` is the scenario temporal-value sum divided by `70`, then averaged equally over scenarios.

For communication, `generated_intents` counts every dynamically legal enqueue request, including queue-full drops and messages later lost. `delivered_intents` counts messages that arrive, and `timely_intents` counts delivered messages whose arrival boundary is no later than the corresponding task's `window_end`. A scenario with no generated intention has delivery-freshness quality zero; otherwise it is

```text
0.5*(delivered_intents/generated_intents)
+ 0.5*(timely_intents/delivered_intents),
```

with the second ratio zero when there is no delivered message. `auxiliary_metrics_1` is the equal scenario mean. Repeated authentic intentions remain separate denominator events.

For each `(receiver_hub,task)`, `preclaim_intents` counts at most one first timely arrived `claim` or `complete` intention when the receiver senses that task and the arrival boundary is strictly earlier than that hub's first claim attempt for it. A receiver that never attempts the task uses sentinel first attempt boundary `61`, so an otherwise qualifying arrival within boundaries `0,...,60` can represent a successfully suppressed claim. `duplicate_claims` counts legal attempts that lose same-hub simultaneous competition or that create an additional global commitment when another hub already has an active commitment, service ownership, or completion; these are performance events, not hidden-state legality failures. With no generated intention, a scenario's coordination efficiency is zero; otherwise it is

```text
0.6*(preclaim_intents/generated_intents)
+ 0.4*clip(1-duplicate_claims/18,0,1).
```

`auxiliary_metrics_2` is the equal scenario mean. The normalized task performance is

```text
(8*primary_metrics_1 + 7*primary_metrics_2
 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2) / 20.
```

#### Result record contract

`results.json` has exact root key `scenarios`, whose value is an array containing exactly one object for each canonical scenario. Every scenario object has exact keys `scenario` and `slots`. Its `slots` array contains exactly one record for every integer `t=0,...,59`. Each slot has exact keys `t`, `claims`, and `hub_actions`:

```json
{
  "scenarios": [
    {
      "scenario": "balanced_stream",
      "slots": [
        {
          "t": 1,
          "claims": [
            {"robot": 0, "task": null},
            {"robot": 1, "task": null}
          ],
          "hub_actions": [
            {
              "hub": 0,
              "enqueue": [
                {
                  "receiver": 1,
                  "robot": 0,
                  "task": "K00",
                  "status": "claim",
                  "event_boundary": 0
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

The deliberately abbreviated example illustrates nested field shape and is not a valid complete submission: all omitted scenarios, slots, robots, and hubs remain mandatory. An accepted claim event created at boundary `0` can first be enqueued at boundary `1`. Scenario, slot, claim, hub-action, and enqueue-array order is irrelevant. Every slot's `claims` contains exactly one action for each robot `0,...,5`; `robot` is a non-Boolean integer and `task` is `null` or a string from `K00,...,K17`. Every slot's `hub_actions` contains exactly one action for each hub `0,...,2`; `hub` is a non-Boolean integer and `enqueue` is an array. Within one hub action, each receiver may occur at most once.

Every enqueue object has exact keys `receiver`, `robot`, `task`, `status`, and `event_boundary`. `receiver` is a non-Boolean integer in `0,...,2`, `robot` is a non-Boolean integer in `0,...,5`, and `event_boundary` is a non-Boolean integer in `0,...,60`; `task` is a canonical task string, and `status` is `claim`, `cancel`, or `complete`. Dynamic edge, ownership, event-history, timing, and queue rules are checked during replay.

The canonical decision keys are the `5*60*6=1800` scenario-slot-robot claims and the `5*60*3=900` scenario-slot-hub actions. These decisions are the only task-specific result evidence. Do not submit task visibility, local knowledge, robot state, queues, message IDs or outcomes, assignments, travel, service, cancellations, returns, completions, or metrics as substitutes or duplicate fields; the grader derives all of them. JSON must be strict: duplicate object members, `NaN`, and infinities are invalid.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must be a thin launcher; model, method, and experiment code may use a reasonable multi-file organization. `run.py` must generate or overwrite root-level `results.json` from project code and the fixed conditions above. The grader deletes any old result before one official execution and evaluates only the newly generated file.

`report.md` must contain these six exact, nonempty Markdown sections: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the local-information model, reproducible allocation and communication procedure, experiments actually run, recorded findings, evidence-based interpretation, and limitations. Additional controlled-experiment evidence may be included in report tables or descriptively named project-generated text or JSON artifacts. Such evidence is for Judge review only and must not alter or replace canonical `results.json`.

## Auto Checks

- id: entry_point
  weight: 2
  desc: "Checks that run.py exists as a regular file and passes static Python compilation."
- id: official_run
  weight: 4
  desc: "Runs python run.py once within 600 seconds and requires a newly generated results.json."
- id: report_contract
  weight: 2
  desc: "Checks report.md for the six required nonempty research sections."
- id: result_json
  weight: 3
  desc: "Checks strict JSON parsing and the exact root and scenario-container structure."
- id: result_evidence
  weight: 4
  desc: "Checks exact slot, claim, hub-action, and intention-enqueue fields."
- id: canonical_coverage
  weight: 4
  desc: "Checks unique coverage of five scenarios, 60 slots, six robots, and three hubs."
- id: decision_legality
  weight: 4
  desc: "Checks local claim availability and knowledge plus intention routes and authentic observed events."
- id: replay_validity
  weight: 6
  desc: "Replays task release, local knowledge, FIFO messages, commitments, travel, service, and return."
- id: primary_metrics_1
  weight: 8
  desc: "Scores task value completed within fixed service windows."
- id: primary_metrics_2
  weight: 7
  desc: "Scores value-weighted completion timing within service windows."
- id: auxiliary_metrics_1
  weight: 3
  desc: "Scores intention delivery and arrival before task-window ends."
- id: auxiliary_metrics_2
  weight: 2
  desc: "Scores pre-claim intention arrival and suppression of duplicate claims."

## Judge Rubric

- id: problem_modeling
  weight: 8
  desc: "Correctness and precision of the online task, local information, message, travel, and occupancy model."
  levels:
    - {score: 0.0, desc: "Report: The formulation is absent or materially wrong. Implementation/evidence: No corresponding task-release, local-view, or communication-allocation process is present."}
    - {score: 0.4, desc: "Report: Only part of the windows, travel, hub information, claims, or return relation is defined and decisive timing is omitted. Implementation/evidence: State or event order materially differs from the report."}
    - {score: 0.7, desc: "Report: Task release, hub sensing, directed messages, commitments, delayed travel, windows, service, and return are accurately described. Implementation/evidence: The core model operates across all scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: Information splitting, task competition, hidden leg outcomes, message timing, service ownership, and long-term robot occupancy are causally and precisely connected. Implementation/evidence: The code faithfully realizes these relations with traceable condition-level evidence."}
- id: method_design
  weight: 15
  desc: "Quality of the unified local allocation and intention policy under windows and delayed occupancy."
  levels:
    - {score: 0.0, desc: "Report: No executable method is given, or a prohibited or information-violating method is used. Implementation/evidence: No legal unified procedure is present."}
    - {score: 0.4, desc: "Report: The method substantively uses only deadline, distance, or a fixed send period and omits information age, duplicate commitments, or return occupancy. Implementation/evidence: Decisive choices lack the stated state basis, use pre-filled scenario decisions, or differ from the report."}
    - {score: 0.7, desc: "Report: A legal, unified, reproducible online allocation and communication method addresses the principal trade-offs. Implementation/evidence: The same procedure operates across scenarios, with limited risk modeling or prediction depth."}
    - {score: 1.0, desc: "Report: A justified unified method jointly treats local task value and opportunity cost, information age, windows, travel risk, return occupancy, and intention queues. Implementation/evidence: Evolving permitted state drives contrasting decisions under information splitting and window pressure without scenario or task schedules."}
- id: experimental_coverage
  weight: 20
  desc: "Execution of controlled experiments on information flow, task pressure, travel outcomes, and completion."
  levels:
    - {score: 0.0, desc: "Report: No credible experiment is provided. Implementation/evidence: No checkable experimental execution is present."}
    - {score: 0.4, desc: "Report: Results are limited to canonical aggregates or isolated conditions and do not test communication-allocation coupling. Implementation/evidence: Actual records do not support the principal experimental claims."}
    - {score: 0.7, desc: "Report: All canonical scenarios and at least one controlled topology, arrival-load, window, travel-residual, or message-delay comparison are covered. Implementation/evidence: Both conditions are executed on comparable inputs with condition-level task, intention, occupancy, and completion records."}
    - {score: 1.0, desc: "Report: Complementary non-redundant experiments include an information-flow mechanism comparison and a window, conflict, or execution-risk sensitivity or failure boundary. Implementation/evidence: At least two meaningful controlled conditions have reconstructable views, messages, claims, occupancy, conflicts, and task metrics."}
- id: research_analysis
  weight: 12
  desc: "Evidence-based explanation of when limited intentions improve online delivery allocation."
  levels:
    - {score: 0.0, desc: "Report: Analysis is absent or contradicts the records. Implementation/evidence: The project cannot support the conclusions."}
    - {score: 0.4, desc: "Report: Discussion mainly repeats completion or message counts without information-age, duplicate-claim, and return-occupancy causality or material limits. Implementation/evidence: Records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: Actual records explain conditional trends in topology, windows, delayed travel, duplicate claims, and robot occupancy while distinguishing observation from inference and stating limitations. Implementation/evidence: Executed comparisons support those claims."}
    - {score: 1.0, desc: "Report: Controlled evidence explains when local information and marginal task value change allocation, quantifies topology, window, or risk sensitivity, tests a failure boundary, and limits the conclusion. Implementation/evidence: Reproducible condition-level records support the central explanation."}
