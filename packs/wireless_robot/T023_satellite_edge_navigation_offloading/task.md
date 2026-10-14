---
id: T023_satellite_edge_navigation_offloading
name: Mobility-Aware Satellite and Edge Offloading for Multi-Robot Navigation
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

Four robots move through a bounded two-dimensional field toward four valuable targets while time-sensitive computing jobs must be processed. A job can run on its owner robot or be uploaded to a shared edge station or satellite. Motion changes edge and satellite distance and therefore upload time; uploaded jobs then compete for finite FIFO compute capacity. Navigation, target assignment, communication, computation, energy, and collision safety consequently form one stateful communication--robotics problem.

### Research Problem

Build a self-contained deterministic simulator and one non-training controller for all five canonical scenarios. At every slot, the controller jointly chooses one movement vector and one target request for each robot, and may submit at most one of that robot's jobs with a local, edge, or satellite placement. The same reusable procedure must derive all decisions from the public scenario inputs, the frozen joint state at the decision boundary, and prior history. It should trade timely target arrival against radio coverage, upload and FIFO delay, job deadlines, energy, and collision safety. The algorithm, planning depth, parameters, code organization, and additional experiments are otherwise your decisions.

### Experimental Setting and Constraints

#### Canonical Evaluation Units

There are five deterministic scenarios in the order shown below. Their zero-based row number is `scenario_index`. Each scenario has slots `t=0,...,59`, state boundaries `0,...,60`, four robots `0,...,3`, four targets `T0,...,T3`, one edge station `E`, and one satellite projection `S`. The field is the closed square `[0,20] x [0,20]`. Obstacles are closed axis-aligned rectangles.

| scenario | starts for robots `0..3` | targets `T0..T3` | obstacles | `E` | `S` |
|---|---|---|---|---|---|
| `open_service_area` | `(1,1),(1,19),(19,1),(19,19)` | `(4,4),(4,16),(16,4),(16,16)` | none | `(10,10)` | `(10,19)` |
| `satellite_only_corridor` | `(2,10),(5,10),(15,10),(18,10)` | `(3,3),(3,17),(17,3),(17,17)` | `[8,12]x[0,7]`, `[8,12]x[13,20]` | `(10,10)` | `(10,2)` |
| `edge_congestion` | `(2,2),(2,18),(18,2),(18,18)` | `(6,6),(6,14),(14,6),(14,14)` | `[9,11]x[4,16]` | `(10,10)` | `(19,10)` |
| `crossing_targets` | `(2,10),(10,2),(18,10),(10,18)` | `(16,4),(16,16),(4,16),(4,4)` | `[9,11]x[8,12]` | `(10,10)` | `(2,10)` |
| `late_jobs` | `(1,10),(6,1),(19,10),(14,19)` | `(3,15),(8,4),(17,6),(12,17)` | `[9,11]x[0,8]`, `[9,11]x[12,20]` | `(10,10)` | `(18,2)` |

The canonical action keys are all `5*60*4=1200` triples `(scenario,t,robot)`. A scenario ends after boundary `60`; unfinished targets and jobs receive no timely-completion value.

#### State, Actions, and Task Model

The target deadlines are `(34,38,42,46)` and target values are `(2,3,4,5)` in target-ID order. A target request is resolved before motion. All requests in a slot are simultaneous; if multiple previously unlocked robots request the same unlocked target, the lowest robot ID obtains it. A request for a target already locked by another robot is a legal no-op. A robot keeps its first accepted target lock and may subsequently submit `null` or repeat that target, but may not request a different target. Each target has at most one owner. The first boundary at which its owner is within Euclidean distance `0.6` is its arrival boundary, provided that owner's proposed motion in the preceding slot was accepted. The arrival is timely exactly when it is no later than the target deadline; later motion does not erase an arrival.

Every robot chooses one of these nine movement vectors per slot: `(0,0)`, `(0.5,0)`, `(-0.5,0)`, `(0,0.5)`, `(0,-0.5)`, and `(sx*0.5/sqrt(2),sy*0.5/sqrt(2))` for each `sx,sy` in `{-1,+1}`. An accepted vector, including `(0,0)`, costs `1+2*||move||_2^2` energy units. Each robot starts with `120` units.

There are twelve jobs `J00,...,J11` in every scenario. For job index `j=0,...,11`, its owner, release boundary, input size in bits, workload in cycles, deadline boundary, and value are

```text
owner    = j mod 4
release  = 3*floor(j/4) + scenario_index
bits     = 48000 + 8000*(j mod 3)
cycles   = 3000000 + 500000*(j mod 4)
deadline = release + 10 + (j mod 2)
value    = 1 + (j mod 4)
```

Each robot has an independent FIFO local-compute node with capacity two, counting the active job, and rate `0.4e6` cycles per slot. A locally admitted job costs its owner `5` energy units. The shared edge and satellite nodes each have FIFO capacity three, counting the active job, and rates `6e6` and `4e6` cycles per slot, respectively. A remote submission is in coverage exactly when the robot's post-motion distance is at most `7` from `E` for edge or at most `20` from the scenario's `S` for satellite. Its fixed upload rate, sampled at that post-motion boundary, is

```text
rate = clip(120000/(1+distance), 20000, 120000) bits/slot,
clip(x,a,b) = min(b,max(a,x)).
```

All admitted uploads advance concurrently and independently in every slot; later motion does not change their rates or cancel them. A valid remote admission costs `ceil(bits/48000)` energy units. Completed uploads enter the selected remote FIFO in ascending `(release,owner,job_id)` order when several arrive at the same boundary. Local capacity is tested at submission; remote capacity is tested when upload arrival is processed. A full queue rejects that job as a failed execution outcome. Each compute node serves at most one active job and applies one node-rate quantum per slot. Completion at the deadline boundary is timely. A job still uploading, queued, or active after its deadline becomes failed and is removed; every nonterminal job also becomes failed after boundary `60`.

#### Information Availability and Event Order

All scenario geometry, target data, node parameters, and all job attributes generated by the public formulas are known at boundary `0`. Thus a controller may use predictable lookahead over those fixed inputs. A job becomes eligible for submission only at its release boundary. There are no hidden random arrivals or random link outcomes. At boundary `t`, the centralized controller sees one frozen joint snapshot containing all four positions and energies, target locks, every released job's state, current uploads and FIFO occupancy, and all completion or failure events already processed at that boundary. It may use model-based predictions from public inputs and history, but it does not observe the outcome of its current slot's actions before choosing them.

The simulator processes every slot in this exact order:

1. Complete compute work scheduled for boundary `t`; process uploads arriving at `t` in the stated   order; mark any nonterminal job with `deadline < t` failed; then release jobs with `release=t`.
2. Freeze the joint pre-execution snapshot and obtain all four actions from the controller.
3. Resolve target requests. Resolve motion in robot-ID order. A proposed endpoint is checked against   the field, obstacles, already accepted lower-ID endpoints, and the unchanged current positions   of higher-ID robots. An unsafe proposal or a move that lacks energy is replaced by staying and   consumes no movement energy.
4. At the resulting post-motion positions, process at most one submitted job per robot. Admit valid   local submissions or valid covered remote uploads and charge their energy. Advance every remote   upload by one slot; newly completed uploads are processed at the next boundary.
5. Start an idle compute node from the head of its FIFO if possible, advance each active node by one   service quantum, and record the state at boundary `t+1`.

Boundary `60` performs step 1 only and accepts no action. Decision fields in `results.json` are the requests chosen from the frozen snapshot. Positions, locks, upload progress, FIFO state, energy, safety events, completion times, and metrics are execution-after quantities derived by the fixed model. Predictions are permitted; representing an execution-after quantity as already observed at the same decision boundary is not.

#### Legal Actions and Boundaries

An unsafe movement proposal and a movement or submission whose stated energy cost is unavailable are illegal for `decision_legality`; the deterministic replacement or failed-job outcome still lets replay continue for diagnosis. Initial robot positions are safe. The higher-ID-position check in motion resolution guarantees that replacement by staying cannot create a collision with an already accepted lower-ID endpoint. Endpoint and coverage comparisons include their stated boundaries.

A job may be submitted only by its owner, at a slot satisfying `release <= t < deadline`, while it is released, nonterminal, and not previously submitted. A null job requires null placement; a non-null job requires exactly one of `local`, `edge`, or `satellite`. A remote choice outside its post-motion coverage or without upload energy is illegal. Queue-full rejection after an otherwise valid submission is a legal execution outcome, not a decision violation, and the job cannot be resubmitted. A target-lock switch is illegal; a losing simultaneous request and a request for a target already owned by another robot are legal no-ops. Numeric movement comparisons use absolute tolerance `1e-6`.

#### Task Metrics

All metrics are independently derived from the action trace and fixed replay. Each metric is in `[0,1]`, higher is better, and scenario values are averaged equally over the five scenarios.

For scenario `s`, let `V_T=2+3+4+5=14`. `primary_metrics_1` is the mean of `sum(value_k * I[target k arrived by its deadline])/V_T`. Let `V_J` be the sum of the twelve public job values in a scenario. `primary_metrics_2` is the mean of `sum(value_j * I[job j completed by its deadline])/V_J`.

For `auxiliary_metrics_1`, its event set in a scenario is every valid edge or satellite upload admitted at submission. For event `o`, let `q_o` be its input bits, `a_o` its upload-completion boundary, `u_o` its submission slot, and `d_o` its deadline. Its upload-timeliness contribution is `clip(1-(a_o-u_o)/max(1,d_o-u_o),0,1)` if `a_o <= min(d_o,60)`, and `0` otherwise. `transfer_timeliness` is the `q_o`-weighted mean of these contributions. Let `b_o` be its service start boundary. Its queue-efficiency contribution is `clip(1-(b_o-a_o)/60,0,1)` when both `a_o` and `b_o` exist, and `0` otherwise; `queue_efficiency` is the arithmetic mean over the same events. Both terms are `0` when the event set is empty. The scenario score is `0.6*transfer_timeliness + 0.4*queue_efficiency`.

For `auxiliary_metrics_2`, `sent_offload_bits` is the sum of input bits of the same valid admitted remote submissions. `delivered_offload_bits` is the subset whose uploads reach their remote node no later than boundary `60`, irrespective of later compute admission or job timeliness. Remaining energy includes all energy charged for accepted movement, admitted local computation, and admitted remote upload. If `sent_offload_bits=0`, the scenario score is `0`; otherwise it is

```text
0.5*delivered_offload_bits/sent_offload_bits
+ 0.5*(1/4)*sum_r clip(remaining_energy_r/120,0,1).
```

The normalized task performance is `(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`.

#### Result Record Contract

`python run.py` must create or overwrite a UTF-8 `results.json` in the project root. Its root is an object with the sole key `scenarios`. The array contains exactly one object for each canonical scenario, with exact keys `scenario` and `slots`. `scenario` is the exact scenario string. `slots` contains exactly one object for every `t=0,...,59`, with exact keys `t` and `actions`; `t` is an integer, not a Boolean. `actions` contains exactly one object for every robot `0,...,3`.

Each action has exactly these keys:

```json
{"robot": 0, "move": [0.0, 0.0], "target_id": null, "job_id": null, "placement": null}
```

`robot` is an integer in `0,...,3`, not a Boolean. `move` is a two-element array of finite numbers matching one of the nine public vectors within absolute tolerance `1e-6`. `target_id` is null or one of `T0,...,T3`; `job_id` is null or one of `J00,...,J11`; `placement` is null, `local`, `edge`, or `satellite`. A null job has null placement, and a non-null job has non-null placement. Array order has no meaning; uniqueness is determined only by `(scenario,t,robot)`.

These 1200 action records are the complete and minimal independent-replay evidence. Do not submit positions, locks, uploads, queues, services, safety events, energy, completion times, or metrics as additional result fields: they have one authoritative value derived by the grader. JSON must be strict and must not contain duplicate object members, `NaN`, or infinities.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must generate or overwrite `results.json` without interactive input. The project may use additional source files. `report.md` must contain non-empty sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

Controlled comparisons are not part of the exact `results.json` contract or the Auto Checks. To support the `0.7` or `1.0` Judge levels, the `Experimental Setup` and `Results` sections must give one row for every named comparison condition. Each row must state the sole changed public parameter or decision regime, its value and unit, the inputs held fixed, the scenario set, and the four replayed metric values. The project source must contain the reusable function or command that generates every reported condition; any saved condition output must be identified in the same row. Such experiments may vary a public model parameter or decision regime, but must label the change and keep all other compared inputs fixed. The Judge inspects the report, source code, and existing project outputs but does not execute the project.

The four numbered metrics receive their stated values only when `official_run`, `result_json`, `result_evidence`, `canonical_coverage`, `decision_legality`, and `replay_validity` all receive full credit. If any prerequisite is not full, all four metric scores are zero.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official Python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the new results.json is strict JSON with the declared root and scenario containers", weight: 3}
- {id: result_evidence, desc: "every slot contains the exact typed movement, target, job, and placement decision fields", weight: 4}
- {id: canonical_coverage, desc: "all 1200 scenario-slot-robot action keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "target requests, motion, job submissions, coverage, and energy satisfy the public boundary rules", weight: 4}
- {id: replay_validity, desc: "the fixed model consistently replays motion, target locks, job lifecycles, uploads, FIFO service, and metric accounting", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed timely target value", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed deadline-qualified job value", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed remote upload timeliness and FIFO queue efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed remote delivery and remaining-energy efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the navigation, communication, computation, information, and deadline model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given, or the movement, target, link, queue, energy, or deadline model is materially wrong; Implementation/evidence: no executable state process or action evidence corresponds to the claimed model."}
    - {score: 0.4, desc: "Report: only part of the entities, information boundary, event order, constraints, or objectives is defined and a decisive relation is omitted; Implementation/evidence: the code or records implement only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: the public states, joint actions, known inputs, event order, target locks, coverage, FIFO service, deadlines, energy, safety, and objectives are accurately formulated; Implementation/evidence: the code and action trace realize the core model across all canonical scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely connects motion, coverage-dependent upload, shared computation, energy, safety, and target and job outcomes while distinguishing prediction from observation; Implementation/evidence: the implementation faithfully realizes those transitions and exposes traceable state evolution without material divergence."}
- id: method_design
  desc: "quality of the unified non-training navigation and offloading procedure"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method is specified, a prohibited training method is used, or the method violates the public action or information model; Implementation/evidence: no runnable unified procedure or supporting decision trace is present."}
    - {score: 0.4, desc: "Report: the method uses fixed target or placement rules and ignores a decisive coverage, queue, deadline, energy, or collision coupling; Implementation/evidence: the primary procedure relies on scenario- or slot-specific action tables or materially differs from its description."}
    - {score: 0.7, desc: "Report: a legal, reproducible state-driven method jointly addresses target choice, navigation, placement, queues, deadlines, energy, and safety with limited planning depth or robustness; Implementation/evidence: one reusable procedure generates decisions from inputs and evolving state across all scenarios, and the action trace supports its operation."}
    - {score: 1.0, desc: "Report: a justified unified method reasons over the temporal consequences of coverage-seeking motion, concurrent uploads, shared FIFO capacity, deadlines, energy, and conflicts; Implementation/evidence: the code consistently derives decisive choices from public inputs, current state, and history without pre-filled unit schedules, with records supporting the claimed mechanism."}
- id: experimental_coverage
  desc: "coverage and diagnostic value of navigation-offloading experiments"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose is provided; Implementation/evidence: the project contains no checkable canonical or controlled execution evidence."}
    - {score: 0.4, desc: "Report: experiments are incomplete or limited to canonical aggregate values and do not examine the movement--communication--compute relation; Implementation/evidence: code or condition-level outputs do not support the principal experimental claims."}
    - {score: 0.7, desc: "Report: all canonical scenarios are covered and one clearly identified controlled comparison varies one coverage, capacity, deadline, energy, or decision-regime condition while holding the other inputs fixed; Implementation/evidence: reusable code implements both conditions and reported condition-level decisions and metrics support the comparison."}
    - {score: 1.0, desc: "Report: at least two complementary non-redundant comparisons include a mechanism-focused condition and a sensitivity or failure-boundary condition, and quantify their target, job, communication, and resource consequences; Implementation/evidence: reusable code executes every condition and condition-level outputs distinguish the claims beyond canonical aggregates."}
- id: research_analysis
  desc: "evidence-grounded interpretation of mobility-aware offloading behavior and limits"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the recorded findings; Implementation/evidence: code and outputs cannot support the reported observations or conclusions."}
    - {score: 0.4, desc: "Report: the discussion mainly repeats values, asserts unsupported causes, or omits material limitations; Implementation/evidence: available records and controlled outputs do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: conclusions use actual records to explain important conditional trends among motion, coverage, upload or queue delay, deadlines, and energy, distinguish observation from inference, and state limitations; Implementation/evidence: canonical and controlled condition outputs support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies the central trade-offs, tests a causal explanation against a contrasting condition or failure boundary, and states scope and unresolved uncertainty; Implementation/evidence: reproducible condition-level decisions, state-derived outcomes, and metrics support every central conclusion."}
