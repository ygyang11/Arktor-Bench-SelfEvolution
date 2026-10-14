---
id: T020_robot_fault_reporting_recovery
name: Event-Triggered Semantic Fault Recovery for Wireless Robotic Manipulation
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

A factory robot executes four ordered manipulation stages while a local semantic sensor observes the current object relation, pose deviation, path safety, and execution state. Drops, placement failures, pose shifts, and obstructions can stop progress. The robot cannot send its entire state directly to the edge server: it chooses a compact semantic report, whose size determines how long it occupies a bandwidth-limited, lossy uplink. The edge server can issue a recovery command only from reports that have arrived and completed processing. That command then competes for an intermittent, lossy downlink and may be stale or unusable when it reaches a busy robot.

The resulting closed loop couples semantic field selection, report age, two communication queues, edge-side diagnosis, recovery preconditions, action duration, and stage deadlines. Missing a decisive field may produce a wrong recovery, whereas repeatedly sending every field can congest the uplink and exhaust the report budget before later faults occur.

### Research Problem

Build a deterministic, non-learning simulator and a unified state-driven method for all five canonical scenarios. At each slot, the local reporting process must select zero or more semantic fields using only its current local observation and transmitter history. At a separate information boundary, the edge recovery process must select no command or one recovery action using only reports that have completed edge processing and its own command history. The two processes must not exchange information outside the specified wireless links.

Design the reporting and recovery logic jointly. It should account for field relevance and size, partial uplink service, queue occupancy, loss and expiry, report freshness, command transit, robot busy state reported in the past, compound-fault action ordering, recovery failure lockout, and remaining stage slack. Evaluate when compact event-triggered semantics improve timely recovery and stage completion, and when congestion, missing context, stale commands, or action preconditions limit that benefit. Deterministic rules, heuristics, planning, search, or optimization are allowed; the method and its parameters are your decisions.

### Experimental Setting and Constraints

#### Canonical scenarios, stages, and fault truth

The canonical scenario order and zero-based scenario indices are:

```text
0 pallet_drop
1 sorting_shift
2 blocked_pick
3 multi_fault
4 late_confirmation
```

Each scenario contains one robot, one local semantic sensor, one edge server, and four ordered stages `0,1,2,3`. Decision slots are `t=0,...,47`, each representing `0.5` seconds. Boundary `48` has no decision or communication service and performs only the terminal action-completion and stage-completion operations defined below.

The four stages require `(5,6,5,5)` units of nominal progress, have inclusive completion-boundary deadlines `(10,22,35,48)`, and have values `(1,1,2,2)`. Their relation labels are respectively `holding`, `holding`, the scenario's placement label, and `aligned`. The placement label is `on` in `pallet_drop` and `multi_fault`, and `inside` in the other scenarios. At the start of each stage, nominal progress is zero, the relation is satisfied, pose deviation is `0.0` metres, the path is clear, the alternate-path flag is false, no action is active, and the failure lock is false. A stage completes at a boundary exactly when its nominal progress has reached its requirement, every fault that occurred in that stage has been recovered, and no action is active. The next stage becomes active at that same boundary with the stated initial state. Stage 3 completion completes the task.

A fault row is `(fault_id, stage, trigger_progress, kind, magnitude, alternate_path)`. When its stage is active and the stage's nominal progress first equals or exceeds `trigger_progress`, that not-yet-occurred fault is injected at the fault-injection step of the current boundary. Rows with the same trigger condition are injected together. `-` means not applicable.

| scenario | placement label | fault rows |
|---|---|---|
| `pallet_drop` | `on` | `(F01,0,1,drop,-,-)`; `(F02,1,3,drop,-,-)`; `(F03,2,2,placement,-,-)`; `(F04,3,2,pose,0.20,-)` |
| `sorting_shift` | `inside` | `(F01,0,2,pose,0.25,-)`; `(F02,1,2,blockage,-,true)`; `(F03,2,1,placement,-,-)`; `(F04,3,2,pose,0.20,-)` |
| `blocked_pick` | `inside` | `(F01,0,1,drop,-,-)`; `(F02,0,1,blockage,-,true)`; `(F03,1,4,pose,0.25,-)`; `(F04,2,2,placement,-,-)`; `(F05,3,1,blockage,-,true)` |
| `multi_fault` | `on` | `(F01,0,1,drop,-,-)`; `(F02,1,2,drop,-,-)`; `(F03,1,2,blockage,-,true)`; `(F04,2,1,placement,-,-)`; `(F05,2,1,pose,0.25,-)`; `(F06,3,2,blockage,-,true)` |
| `late_confirmation` | `inside` | `(F01,0,3,pose,0.20,-)`; `(F02,1,4,blockage,-,true)`; `(F03,2,3,placement,-,-)`; `(F04,3,3,pose,0.25,-)` |

`drop` and `placement` make the current relation unsatisfied. `pose` sets pose deviation to the listed magnitude. `blockage` makes the path unclear and sets the alternate-path flag to the listed Boolean. The local sensor sees these effects immediately at that boundary. A fault's occurrence slot is this boundary index. Its recovery boundary is the first action-completion boundary at which a successful action clears it. Its inclusive timely-recovery deadline is `min(occurrence_slot + 8, 48)`. A fixed fault that never occurs because its stage is never reached is not removed from metric denominators and receives zero timely-recovery value.

The scenario and fault tables are public simulator truth, but they are not policy-visible truth. They may be used by the environment transition function to inject faults. Neither decision process may inspect a not-yet-injected row, a future fault, a future channel outcome, or a terminal result. The method must invoke its reporting and edge-command procedures on the filtered views described below; it must not turn the tables, a scenario name, a fault ID, a stage ID, or an absolute slot directly into a pre-filled decision schedule.

#### Local report generation and uplink

At every active-task decision boundary, the local view contains exactly the current stage and relation label; nominal progress; relation-satisfied value; pose deviation; path-clear and alternate-path values; active-action kind and completion boundary, or `null` values when idle; failure-lock value; the two queued reports with their generated fields, original sizes, and remaining bits; cumulative successfully-enqueued bits; whether earlier reports have left the local queue; and the history of its own prior local views and report decisions. It contains no report loss, expiry, processing, or edge receipt acknowledgement and no edge command decision or downlink outcome.

The local decision is a set of zero or more distinct fields from `{relation,pose,stage,safety}`. In `results.json` it is represented as a list; list order has no semantic meaning, and every permutation of the same set is equivalent. An empty list means no report request. At most one report is requested in a slot. A nonempty report contains an `8`-bit header and the values of its selected fields from that decision boundary:

| field | bits | fixed payload |
|---|---:|---|
| `relation` | 24 | current stage relation label and whether it is satisfied |
| `pose` | 16 | current pose deviation in metres |
| `stage` | 8 | stage ID, nominal progress, active-action state, and failure-lock value |
| `safety` | 8 | path-clear and alternate-path values |

The report ID is its generation slot. The uplink queue holds at most two whole reports and is first-in-first-out. A scenario has a budget of `480` successfully-enqueued bits. A request made while the queue is full is dropped, consumes no budget, and is not a legality violation. If queue space is available but the requested report size exceeds the remaining budget, the request is suppressed and is a legality violation. A list containing a duplicate field is suppressed and is a legality violation. An unknown field is a result-evidence failure and is likewise suppressed, but is not scored again as a legality defect. Only successful enqueues consume budget.

The per-slot uplink service capacity is selected by `t mod 8`:

| scenario | capacities in bits per slot for residues `0,...,7` |
|---|---|
| `pallet_drop` | `(48,32,0,24,40,32,0,48)` |
| `sorting_shift` | `(32,0,24,48,32,24,40,0)` |
| `blocked_pick` | `(24,32,0,24,48,0,32,40)` |
| `multi_fault` | `(40,24,0,32,24,48,0,32)` |
| `late_confirmation` | `(32,24,40,0,24,32,48,0)` |

A newly enqueued report may receive service in its generation slot. A slot serves only the queue head, by `min(capacity, head_remaining_bits)` bits. Capacity left after completing that report is unused. If a report generated at `g` completes service in slot `u`, it is lost exactly when `(3*scenario_index + g + u) mod 13` is in `{0,1}`. Otherwise it arrives at edge boundary `u+1`. An arrival whose age `(u+1)-g` is greater than `6` is expired and discarded. Every non-expired arrival requires one full processing slot: it enters the processed-report set before the edge decision at boundary `u+2`.

The edge retains all processed reports and each report's actual selected payload, generation slot, and processing boundary. For each field it also maintains the latest processed value, replacing an older value only when the new report has a later generation slot. Reports with equal generation slots cannot occur. At boundary `t`, define the fixed command evidence slot `g_cmd` as the maximum generation slot among all processed reports. It exists and is age-qualified exactly when at least one report has been processed and `t-g_cmd <= 10`.

#### Edge command generation, downlink, and recovery actions

At every active-task decision boundary, the edge view contains the current boundary; every processed report with its actual fields, values, generation slot, and processing boundary; its per-field latest knowledge; its current first-in-first-out command queue, whose entries expose command generation slot, action, and fixed evidence slot; and its own prior edge views and command decisions. It contains no current robot or fault truth except values carried in processed reports, no future report or channel outcome, and no acknowledgement of whether a served command was lost, discarded, started, or successful.

The edge decision is either `none` or one action in `{stop,regrasp,reposition,retry,bypass}`. A non-`none` request is legal exactly when `g_cmd` exists and is age-qualified; otherwise it is suppressed and is a legality violation. A legal command automatically stores `g_cmd` as its fixed evidence slot, so the candidate does not choose or submit a single report reference when its method may have fused several processed reports. The command ID is its generation slot. If a legal request finds the two-command queue full, it is dropped without a legality violation. Otherwise it is appended to the queue.

One queued command is served in slot `t` exactly when the queue is nonempty and `(scenario_index+t) mod 5 != 0`. A command generated at `c` and served at `v` is lost exactly when `(5*scenario_index+c+v) mod 17 = 0`; otherwise it arrives at robot boundary `v+1`. At arrival, discard the command if the task is complete, the robot has an active action, or its fixed evidence age `(v+1)-g_cmd` is greater than `12`. A non-discarded command immediately starts its action before the two decisions at that boundary. The edge receives no acknowledgement of any downlink result.

`stop` always succeeds, lasts one slot, clears the failure lock at completion, and does not recover a fixed fault. Every other action tests its success conditions at its start boundary. It succeeds only when the failure lock is false, at least one unrecovered fault of its corresponding kind is active in the current stage, and its additional condition is true. A successful action has its base duration and at completion clears all active current-stage faults of that kind and restores their state variable. A failed action lasts its base duration plus two slots, recovers nothing, and sets the failure lock at completion. While the lock is true, every non-`stop` action fails; `stop` must complete before another recovery attempt can succeed.

| action | base duration in slots | corresponding fault | additional success condition | successful state restoration |
|---|---:|---|---|---|
| `regrasp` | 2 | `drop` | path is clear | relation becomes satisfied |
| `reposition` | 2 | `pose` | pose deviation is in `(0,0.25]` and path is clear | pose deviation becomes `0.0` |
| `retry` | 1 | `placement` | path is clear | relation becomes satisfied |
| `bypass` | 3 | `blockage` | alternate path is available | path becomes clear and alternate-path flag becomes false |

An action beginning at boundary `b` with duration `d` is active on `[b,b+d)` and completes at boundary `b+d`. No command can start while it is active. A fault is recovered, and a stage may complete, at the completion boundary, not at action start. An action scheduled to complete after boundary `48` produces no success or failure consequence within the evaluated horizon.

#### Authoritative event order and record semantics

At each boundary `t=0,...,47`, the following order is authoritative:

1. Complete an action whose completion boundary is `t`, apply its success or failure consequence, and   assign this boundary as the recovery boundary of every fault it successfully clears.
2. If the active stage now satisfies its completion conditions, complete it at boundary `t` and   initialize the next stage; completing stage 3 completes the task.
3. If the task remains active, inject all not-yet-occurred rows for the active stage whose trigger   progress is at most the current nominal progress. Their effects are visible locally in this same   boundary.
4. Add reports whose processing completes at `t` to the edge processed-report set. Then receive uplink   reports arriving at `t`: discard lost or expired reports and schedule every other report to complete   processing at `t+1`.
5. Receive downlink commands arriving at `t`. Apply the discard rules above; if a command remains and   the robot is idle, start its action immediately.
6. If the task remains active, freeze two separate views after steps 1-5. Call the local reporting   procedure only on the local view and the edge recovery procedure only on the edge view. The two   submitted decisions are simultaneous with respect to this boundary and cannot use the current   decision of the other process.
7. If the task remains active, validate and enqueue the report request and command request in that   order, then perform this slot's uplink service and downlink service. Their outcomes cannot affect   either decision at this boundary.
8. If the task remains active, the robot has no active action, and the current stage has no unrecovered   fault, increase nominal progress by one at the slot end. Save all derived state and advance.

If step 2 completes the task, steps 4 and 5 still settle reports and commands already scheduled for that boundary; in particular, an arriving command is discarded by the task-complete rule. Steps 3 and 6-8 are then skipped. The same settlement-only rule applies at every later boundary through `47`.

At boundary `48`, perform only steps 1 and 2. Do not receive messages, inject faults, make decisions, serve links, or add nominal progress. A report or command scheduled to arrive at boundary `48` is not received. A task-complete slot still requires a canonical result record. Its values must satisfy the published result types and enums, and `report_fields` must still contain distinct fields. Both values are otherwise ignored by the simulator: they consume no resource, generate no message or action, and create no report-budget or command-information violation.

The submitted `report_fields` value is a local decision at step 6. The submitted `command_action` is the edge decision at step 6. All observations and processed reports used at those boundaries follow steps 1-5. Report payloads, IDs, sizes, queues, service, loss, arrival, edge knowledge, command IDs, action starts and outcomes, faults, recovery boundaries, stage states, terminal results, and metrics are post-decision or derived facts reconstructed by the grader; they are not submitted. This event order is the sole timing authority.

Illegal requests are suppressed only to let diagnostic replay continue; they still fail their stated legality atom. Queue-full drops and the deterministic environment discard rules are legal outcomes. There is no randomness. All integer fields reject Boolean values. Fixed-model floating comparisons use absolute tolerance `1e-9`.

#### Task metrics

All metrics are computed from the independent replay and lie in `[0,1]`; higher is better.

`primary_metrics_1` is timely stage completion. For each scenario, a stage earns its value from `(1,1,2,2)` exactly when its completion boundary is no later than its deadline from `(10,22,35,48)`. Divide the earned value by `6`, then average equally over the five scenarios.

For `primary_metrics_2`, fixed fault `f` has timely value zero if it never occurs, never recovers, or recovers after its deadline `D_f`. Otherwise, with occurrence slot `o_f` and recovery boundary `r_f`,

```text
T_f = clip(1 - (r_f-o_f)/(D_f-o_f), 0, 1).
```

Let `R_s` be the equal average of `T_f` over every fixed fault row of scenario `s`, including rows that never occur. Let `S_s=1` exactly when all four stages complete by boundary `48`, otherwise `0`. The scenario value is `0.7*R_s + 0.3*S_s`; average it equally over the five scenarios.

`auxiliary_metrics_1` measures useful semantic bits. Every bit of every successfully enqueued report, including its header, enters the denominator. A field's own fixed bit count enters the numerator only if its report is neither lost nor expired and completes edge processing, its generation slot is later than the edge's previous latest generation slot for that field, and its generation-boundary payload has the following relevance: `relation` sees an unrecovered `drop` or `placement`; `pose` sees an unrecovered `pose`; `safety` sees an unrecovered `blockage`; or `stage` sees any unrecovered fixed fault or a true failure lock. Headers, non-relevant fields, duplicate or older information, and fields in lost or expired reports never enter the numerator. For each scenario use zero if no bits were successfully enqueued, otherwise `clip(useful_field_bits/enqueued_bits,0,1)`, then average over scenarios.

For `auxiliary_metrics_2`, each non-`stop` action that actually starts is one sample. Its success value is one only if it completes no later than boundary `48` and successfully applies its recovery effect; otherwise it is zero. If it starts at boundary `b` and its command's fixed evidence slot is `g`, its freshness is `clip(1-(b-g)/12,0,1)`. Sample quality is `0.6*success + 0.4*freshness`. Average samples within a scenario, using zero for a scenario with no sample, then average over scenarios.

The normalized task performance is

```text
(8*primary_metrics_1 + 7*primary_metrics_2
 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2) / 20.
```

#### Result record contract

`results.json` has exact root key `scenarios`, whose value is an array. It contains exactly one object for each canonical scenario. Every scenario object has exact keys `scenario` and `slots`:

```json
{
  "scenarios": [
    {
      "scenario": "pallet_drop",
      "slots": [
        {
          "t": 0,
          "report_fields": [],
          "command_action": "none"
        }
      ]
    }
  ]
}
```

For each scenario, `slots` contains exactly one record for every integer `t=0,...,47`; `(scenario,t)` is the canonical key, giving `5*48=240` records. Scenario and slot array order is irrelevant. Every slot object has exactly the three keys shown above:

- `t` is the canonical integer slot and is not a Boolean.
- `report_fields` is a list of distinct strings from `{relation,pose,stage,safety}`. Its ordering is  irrelevant; an empty list means no report request.
- `command_action` is one of `none`, `stop`, `regrasp`, `reposition`, `retry`, or `bypass`.

These 240 minimal decision records are the only task-specific result evidence. Do not submit replayed observations, payloads, communication events, state traces, fault records, actions, stages, terminal summaries, or metric values as substitutes or duplicate evidence. The grader derives all such facts from the fixed model. JSON must be strict: duplicate object members, `NaN`, and infinities are invalid.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must be a thin launcher; model, method, and experiment code may use a reasonable multi-file organization. `run.py` must generate or overwrite root-level `results.json` from the project code and the fixed conditions above. The grader deletes any old result before one official execution and evaluates only the newly generated file.

`report.md` must contain these six exact, nonempty Markdown sections: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model and information boundaries, reproducible method, experiments actually run, recorded findings, evidence-based interpretation, and scope or limitations. Additional controlled-experiment evidence may be recorded in `report.md` tables or in descriptively named text or JSON artifacts generated by the project. Such artifacts are for Judge evidence only: they must not replace, wrap, or add fields to canonical `results.json`, and Auto Check does not treat them as canonical result evidence.

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
  desc: "Checks strict JSON parsing and the exact root and canonical scenario-container structure."
- id: result_evidence
  weight: 4
  desc: "Checks the slot containers, local report-field decisions, and edge command-action fields."
- id: canonical_coverage
  weight: 4
  desc: "Checks unique coverage of all five scenarios and all 48 decision slots per scenario."
- id: decision_legality
  weight: 4
  desc: "Checks field-set uniqueness, dynamic report budget, and whether nonempty commands have fresh processed evidence."
- id: replay_validity
  weight: 6
  desc: "Replays decision consumption, faults, stages, uplink processing, edge knowledge, downlink recovery, and terminal accounting."
- id: primary_metrics_1
  weight: 8
  desc: "Scores value-weighted stage completion at the fixed inclusive deadlines."
- id: primary_metrics_2
  weight: 7
  desc: "Scores timely recovery over all fixed faults together with complete-task success."
- id: auxiliary_metrics_1
  weight: 3
  desc: "Scores newly useful fault-semantic field bits relative to all successfully enqueued report bits."
- id: auxiliary_metrics_2
  weight: 2
  desc: "Scores successful non-stop recovery actions together with their fixed-evidence freshness."
## Judge Rubric

- id: problem_modeling
  weight: 8
  desc: "Correctness and precision of the semantic-reporting, communication, fault, recovery, and stage model."
  levels:
    - {score: 0.0, desc: "Report: The formulation is absent or materially wrong. Implementation/evidence: No corresponding staged fault, bidirectional communication, and recovery process is present."}
    - {score: 0.4, desc: "Report: Only part of the relation, queue, link, or action model is defined and decisive timing is omitted. Implementation/evidence: Simulated state or event order materially differs from the report."}
    - {score: 0.7, desc: "Report: Local and edge information boundaries, field sizes, bidirectional timing, stages, and action constraints are accurately described. Implementation/evidence: The core model is implemented across all scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: The causal boundaries among field choice, queue age, edge knowledge, recovery preconditions, and stage completion are specified precisely. Implementation/evidence: The implementation faithfully realizes them with traceable condition-level evidence and no material divergence."}
- id: method_design
  weight: 15
  desc: "Quality of the unified semantic reporting and edge recovery method under temporal and information constraints."
  levels:
    - {score: 0.0, desc: "Report: No executable method is given, or a prohibited or information-violating method is used. Implementation/evidence: No legal unified procedure is present."}
    - {score: 0.4, desc: "Report: The method, including any periodic-report or fixed-field design it uses, omits decisive queue, budget, freshness, or recovery-consequence reasoning. Implementation/evidence: The procedure is materially state-incomplete, relies on pre-filled schedules, or materially differs from the stated method."}
    - {score: 0.7, desc: "Report: A legal, unified, reproducible field-trigger and edge-recovery method addresses the main trade-offs. Implementation/evidence: The same procedure operates across scenarios and supports the principal choices, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: A justified unified method jointly treats fault-relevant fields, partial-service queues, report age, commands in transit, reported busy state, compound-fault order, and failure lockout. Implementation/evidence: Decisions follow evolving permitted state without scenario or slot decision tables and agree with the described method."}
- id: experimental_coverage
  weight: 20
  desc: "Execution of controlled experiments on semantic selection, communication conditions, and recovery behavior."
  levels:
    - {score: 0.0, desc: "Report: No credible experiment is provided. Implementation/evidence: No checkable experimental execution is present."}
    - {score: 0.4, desc: "Report: Results are limited to canonical aggregates or one isolated condition and do not test the semantic-recovery mechanism. Implementation/evidence: Actual records do not support the main experimental claims."}
    - {score: 0.7, desc: "Report: All canonical scenarios and at least one controlled comparison directly testing field selection, channel capacity, or recovery order are covered. Implementation/evidence: Both compared conditions are executed on comparable inputs with condition-level records."}
    - {score: 1.0, desc: "Report: Complementary non-redundant experiments include a mechanism comparison and a sensitivity or failure-boundary test that distinguish field size, links, retransmission, command scheduling, or action preconditions. Implementation/evidence: At least two meaningful controlled conditions have reconstructable condition-level records and metrics."}
- id: research_analysis
  weight: 12
  desc: "Evidence-based explanation of when semantic communication changes recovery outcomes and where it fails."
  levels:
    - {score: 0.0, desc: "Report: Analysis is absent or contradicts the records. Implementation/evidence: The project cannot support the conclusions."}
    - {score: 0.4, desc: "Report: Discussion mainly repeats completion or traffic values without timing and recovery causality or material limits. Implementation/evidence: Records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: Actual records are used to explain conditional trends in field value, report age, command freshness, wrong actions, and stage completion, while distinguishing observations from inference and stating limitations. Implementation/evidence: The executed comparisons support those claims."}
    - {score: 1.0, desc: "Report: Controlled evidence tests when fault semantics change recovery order, quantifies principal communication and action-delay sensitivities, and bounds the explanation with contrasting evidence or a failure boundary. Implementation/evidence: Reproducible condition records support the central causal claims."}
