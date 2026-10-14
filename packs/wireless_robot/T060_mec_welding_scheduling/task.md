---
id: T060_mec_welding_scheduling
name: Process-State-Aware MEC Scheduling for Collaborative Robotic Welding
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

Four robots prepare and execute welds at two shared workcells.  Before each weld, a robot must obtain a scene interpretation and a start-point localization; after welding it must obtain a verification result.  These computations may run on the robot, on one of two nearby MEC workers, or on a remote Cloud worker.  Remote placement can improve quality and shorten compute service, but it consumes a shared wireless service budget and encounters congestion, loss, propagation delay, queue contention, and a worker outage.  A late or low-quality perception result blocks the physical process, while poorly timed workcell requests create safety interventions or leave scarce welding capacity idle.

### Research Problem

Design a deterministic, non-training, state-driven scheduler for request placement, worker choice, priority, retry, start-result cache reuse, weld abandonment, and workcell requests.  One reusable procedure must operate across all five scenarios.  The research objective is to progress as many welds as possible before their deadlines while balancing result quality and freshness, local energy, wireless and remote-compute service, queue rejection, workcell flow, and safety.  Communication and compute decisions must change the robot process state rather than being evaluated as an isolated network benchmark.

### Experimental Setting and Constraints

#### Canonical units, weld activation and deadlines

Each canonical unit is one scenario in the table below.  Time has `H=240` action slots `t=0..239`, each lasting `50 ms`, and state boundaries `0..240`.  Robots are `R0..R3`; welds are `P0..P5`. Robot `Rr` processes its welds sequentially, and weld `Pp` uses workcell `W{(r+p) mod 2}`.  Its earliest availability is `a_(r,p)=base+10*p+offset_r`.  It activates at the first boundary `u_(r,p)>=a_(r,p)` at which the preceding weld is complete, failed, or abandoned.  Its inclusive hard deadline is `d_(r,p)=min(240,u_(r,p)+deadline_span)`.  Boundary `240` processes previously scheduled events and deadlines but accepts no new decisions.

Immediately before boundary-0 events, every worker and workcell is idle, all queues and event lists are empty, every cache is empty, all energy and service counters are zero, and each robot's current weld is `P0` in the not-yet-activated state.  Boundary 0 then applies the same activation and event rules as every other boundary.

| scenario | `base`; offsets `(R0,R1,R2,R3)`; deadline span | wireless condition | worker condition | local service multiplier | energy normalization budget |
|---|---|---|---|---|---:|
| `staggered_cells` | `0`; `(0,6,12,18)`; `68` | normal | all available | `1.0` | 760 |
| `simultaneous_preheat` | `0`; `(0,0,0,0)`; `60` | congested at slots `20..59`, normal otherwise | all available | `1.0` | 780 |
| `uplink_jitter` | `0`; `(0,3,6,9)`; `64` | jitter at slots `15..34,70..94,135..159`, normal otherwise | all available | `1.0` | 790 |
| `mec_worker_loss` | `0`; `(0,4,8,12)`; `66` | normal | `M1` unavailable at boundaries/slots `55..124` | `1.0` | 800 |
| `hardware_gap` | `0`; `(0,2,4,6)`; `62` | congested at slots `80..119`, normal otherwise | all available | `R0,R1:1.0`; `R2,R3:1.5` | 810 |

The energy values are fixed normalization budgets, not hidden validity gates.  Scenario schedules are environment truth.  A policy is called online and receives only the current condition described below; it must not query a future condition merely because the canonical simulator contains that schedule.

#### Perception requests and welding process

Every activated weld releases one `scene` request.  Adoption of a qualified `scene` result releases one `start` request.  Adoption of a qualified `start` result, or a qualified reuse of the robot's most recent adopted start result, makes the weld ready.  A workcell admission starts twelve consecutive, non-preemptible welding slots.  Their completion releases one `verify` request, and adoption of its qualified result completes the weld.  These are the 72 fixed request opportunities `4 robots * 6 welds * 3 types`.

| request type | input/output bits | inclusive request-deadline offset | minimum quality | Local/MEC/Cloud service units | Local/MEC/Cloud result quality | Local admission energy |
|---|---:|---:|---:|---|---|---:|
| `scene` | `2048/256` | 22 | 0.80 | `12/4/3` | `0.84/0.94/0.97` | 6 |
| `start` | `4096/384` | 26 | 0.90 | `18/6/4` | `0.91/0.96/0.99` | 9 |
| `verify` | `3072/256` | 20 | 0.86 | `15/5/3` | `0.88/0.95/0.98` | 7 |

A request released at boundary `s` has inclusive deadline `min(d_(r,p),s+offset)`.  A result is adopted automatically at its arrival boundary exactly when it is no later than that deadline and its path quality meets the threshold.  In `hardware_gap`, an `R2` or `R3` Local result loses `0.08` quality. Its Local service demand is `ceil(1.5*base_Local_units)`; all other demands use the table directly. An unadopted request expires immediately after its inclusive deadline boundary has processed arrivals.

For a released request the scheduler may wait or start an attempt before the request deadline.  An attempt selects `local`, `mec`, or `cloud`, a consistent worker, and integer priority `0..9`, where larger values are served first.  At most one attempt for a request may be unfinished.  Up to three attempts are allowed with consecutive `retry_index=0,1,2`; a new index may start no earlier than the boundary after the preceding attempt is lost, queue-rejected, made unavailable, or otherwise fails. Starting an attempt before release, after adoption/expiry, while one is unfinished, or with the wrong next retry index is a decision-legality error.  Queue rejection, a currently unavailable `M1`, link loss, and deadline expiry are valid modeled outcomes, not structural errors.  A returned result below its quality threshold is also a failed attempt and makes the next retry eligible at the following boundary.

Each robot retains only its latest adopted qualified `start` result.  At reuse boundary `t`, its age is `t-original_adoption_slot` and its reuse quality is `q'=q-0.005*age`.  Reuse is qualified only for age at most `40` and `q'>=0.90`.  Cache reuse is a weld-level fallback and remains available in `locate_wait` after the request's transport/computation opportunity has expired, but only a reuse no later than that request deadline contributes to deadline service.  A listed reuse when the corresponding weld is not in `locate_wait`, or when the cache is unqualified, is rejected and adds one safety event.  A robot may list one abandonment boundary for a weld; it succeeds only while that weld is current, unfinished, and not welding, and an invalid abandonment likewise adds one safety event.  At the weld hard deadline, after arrivals are processed, every still-unfinished weld fails and all of its pending work is cancelled; the next weld may then activate at that boundary.  Activation and hard-deadline termination repeat until stable, so a weld that first activates at boundary `240` immediately fails there.

At a boundary, a listed workcell request made by a weld that is not the robot's current `ready_weld` adds one safety event.  A request by a ready weld while its cell is busy is legal but cannot enter a contest.  When a cell is free, all ready listed requests are ranked by earlier ready boundary, then smaller remaining hard-deadline slack, then smaller robot ID.  The winner starts welding; every other qualified contender adds one workcell intervention.  A ready robot may legally wait without requesting.

#### Communication, queues, workers and budgets

Local workers are `L0..L3`, with `Lr` usable only by `Rr`; MEC workers are `M0,M1`; the Cloud worker is `C0`.  Each worker is single-threaded and non-preemptive.  Local and MEC workers provide one service unit per slot; `C0` provides one only in even-numbered slots.  The number of unfinished jobs admitted to `L0..L3,M0,M1,C0`, including any active job, may not exceed respectively `1,1,1,1,2,2,2`. Within each worker, an active job finishes without preemption; the next job is selected by decreasing priority, earlier request deadline, earlier admission slot, then smaller derived attempt ID.  Request expiry, weld termination, and the specified `M1` outage cancel affected work and are the only exceptions to non-preemption.

Derived IDs are public deterministic strings.  A request uses `Q_Rr_Pp_kind`; its retry attempt uses `A_Rr_Pp_kind_i`; that attempt's input and output packets use `I_A_Rr_Pp_kind_i` and `O_A_Rr_Pp_kind_i`.  Here `Rr`, `Pp`, `kind`, and `i` are the record's literal robot, weld, request type, and retry index.  Weld-completion events use `Rr/Pp`.  Every “smaller ID” comparison and same-class event tie uses ordinary lexicographic order on these strings; attempt-related events use the attempt ID.

A remote attempt first creates an input packet and, after compute completion, creates an output packet. The shared `uplink` and `downlink` each hold at most three queued packets and at most one belonging to one robot.  A packet or worker job that cannot enter its queue makes its attempt `queue_rejected` at the next boundary.  Simultaneous input packets created by attempt decisions seek admission by decreasing priority, earlier request deadline, then smaller derived attempt ID.  Each channel serves at most one packet per slot, selected by decreasing priority, earlier request deadline, earlier admission slot, then smaller packet ID.  The slot's base capacity is `1024` bits uplink and `512` bits downlink.  For a packet belonging to `Rr`, multiply it by:

- `1` in a normal slot;
- `0.5` in a congested slot;
- `0.5+0.5*((t+r) mod 3)/2` in a jitter slot;

and floor the resulting number of served bits.  The link state at the packet's final service slot `t` has level `normal=0`, `congested=1`, or `jitter=2`.  With request-kind index `scene=0,start=1,verify=2`, the packet succeeds exactly when `((17*r+7*t+kind) mod 100)/100 >= 0.04+0.12*level`.  Failure is known at boundary `t+1`. A successful MEC packet arrives after `1+((t+r) mod 3)` boundaries.  A Cloud packet adds `8+((3*t+r) mod 11)` to that delay.  Input and output packets use the same capacity, loss, and delay rules.  Only a successfully returned output can create a remote result.

Across one scenario, the channels stop serving after `32000` cumulative bits, and `M0,M1,C0` stop after `48` cumulative service units.  Partial service up to a remaining budget is allowed.  These are physical replay limits: excess queued work waits rather than making the submitted decision shape illegal.  Every Local attempt admitted to `Lr` consumes the table's Local energy immediately; failed or repeated Local attempts consume it again.  Other actions add no energy in this model.

In `mec_worker_loss`, `M1` becomes unavailable before boundary-55 decisions and remains unavailable through slot 124.  Every active or queued `M1` attempt fails as `worker_unavailable` at boundary 55; completed work is discarded and its next retry can start at boundary 56.  Input arriving at unavailable `M1` fails in the same way.  `M1` resumes at boundary 125.

#### Event order and information boundary

At each action-slot boundary `t=0..239`, apply worker failure or recovery; process attempt failures, compute completions, input arrivals, output arrivals, and weld completions scheduled by prior slots, in that class order and then by derived ID; an input arrival immediately seeks worker admission and a compute completion immediately seeks output-channel admission, while each qualified Local completion or output arrival is adopted as its event is processed; perform the resulting process transitions and request releases; expire requests whose inclusive deadline has just been processed and fail welds whose inclusive hard deadline has been reached; read the submitted decisions for this boundary; apply cache-reuse requests, then attempt starts, then abandonments; admit new jobs or packets subject to queue capacity; serve each channel and worker for this slot; resolve the submitted workcell requests at idle-cell contests and execute one slot of every admitted weld; schedule slot-end communication, compute and welding events for later boundaries; then update progress, wait, intervention, safety, energy and budget histories.  Boundary 240 executes only the event, adoption, expiry and weld-failure parts.  When the common budget leaves an ordering relevant, service order is uplink then downlink and `L0,L1,L2,L3,M0,M1,C0` for workers.

A robot policy may read its own current weld/process state, released request and result status, latest start cache, local hardware profile and worker queue, current link state/capacity, its own cumulative Local-admission energy, and workcell busy/admission feedback.  The remote scheduler may read requests and packets that have arrived remotely, current worker/channel queues and availability, current link state, and remaining public service budgets. The unified scheduler may combine these contemporaneous robot-side and remote-side views when choosing an attempt and its worker.  Future releases, link outcomes, failure/recovery, worker completions, and workcell admissions are not available before their event boundary.  The same policy may coordinate its own components through these allowed current observations, but may not use a scenario, robot, weld, request ID, or absolute slot as a lookup key for pre-filled placement, retry, cache, abandonment, or workcell schedules.

Decision fields in `results.json` are the requests made at their listed boundaries.  Request release, packet and queue state, worker progress, results, cache quality, process transitions, workcell outcome, safety, energy, budgets and metrics are post-decision facts reconstructed by the grader.  The exact event order above is authoritative.

#### Task metrics

Metrics are computed separately in each scenario and then averaged equally.  A weld counts as an on-time completion only if it reaches `complete` by its hard deadline.  Its stage progress is the sum of milestones actually completed before failure: `inspect=0.15`, `locate=0.25`, `weld=0.40`, and `verify=0.20`.  Average both quantities over all 24 welds and define `primary_metrics_1=0.65*on_time_weld_completion+0.35*stage_progress`.

Perception quality is the arithmetic mean over the 72 fixed request opportunities.  An adopted result contributes `quality*clip(1-request_age/24,0,1)` to perception quality, where `request_age=adoption_slot-release_slot`; an unactivated or unadopted opportunity contributes zero. Deadline service is the fraction of the 72 opportunities adopted by their request deadline.  With `S` replayed safety events, define `primary_metrics_2=0.5*perception_quality+0.3*deadline_service` `+0.2*clip(1-S/(4*240),0,1)`.

`productive_weld_slots` is the number of robot-slots that actually execute welding.  A `ready_wait_slot` occurs when a weld is ready and its workcell is free at contest time but the robot does not start welding.  If their sum is zero, `workcell_flow=0`; otherwise it is `productive_weld_slots/(productive_weld_slots+ready_wait_slots)`.  Let `I` be total workcell interventions, `E` be total Local-admission energy, and `B_E` the scenario energy budget in the table. Define `primary_metrics_3=0.45*workcell_flow+0.25*clip(1-I/(4*240),0,1)` `+0.30*clip(1-E/B_E,0,1)`.

For remote attempts, `remote_bits` is all actually served input and output bits.  `useful_remote_bits` is the served bits belonging to an attempt whose returned result is adopted by its request deadline. `offloaded_attempts` counts started MEC/Cloud attempts and `on_time_remote_results` counts those that produce such an adopted result.  `result_freshness` averages `clip(1-(adoption_slot-release_slot)/24,0,1)` over adopted remote results and is zero with no sample. Define `auxiliary_metrics_1=0.45*useful_remote_bits/max(1,remote_bits)` `+0.30*on_time_remote_results/max(1,offloaded_attempts)+0.25*result_freshness`.

Let `communication_bits` and `remote_compute_units` be the actual cumulative services, and let `Q` be the number of packet/job admission rejections.  Define `auxiliary_metrics_2=0.45*clip(1-communication_bits/32000,0,1)` `+0.35*clip(1-remote_compute_units/48,0,1)+0.20*clip(1-Q/72,0,1)`. All metric values are clipped to `[0,1]`.  The normalized task performance used for difficulty calibration is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3` `+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result contract

Write strict UTF-8 JSON whose root is exactly `{"scenarios":[...]}`.  Each scenario appears exactly once and has exactly `scenario,request_decisions,weld_decisions`.

`request_decisions` contains exactly 72 objects, one for every `(robot_id,weld_id,request_type)` in `R0..R3 * P0..P5 * scene,start,verify`.  Each object is exactly `{robot_id,weld_id,request_type,attempts,cache_reuse_slots}`.  `attempts` has at most three objects, each exactly `{retry_index,start_slot,path,worker_id,priority}`.  Attempt start slots are distinct integers in `0..239`; paths are `local,mec,cloud`; the worker must respectively be the robot's `Lr`, one of `M0,M1`, or `C0`; and priority is an integer in `0..9`.  Within a request, attempts are identified by a unique consecutive retry index starting at zero and are ordered semantically by their strictly increasing start slots.  `cache_reuse_slots` is empty for `scene` and `verify`; for `start` it is a strictly increasing, duplicate-free array of integer boundaries in `0..239`.

`weld_decisions` contains exactly 24 objects, one for every `(robot_id,weld_id)`.  Each is exactly `{robot_id,weld_id,workcell_request_slots,abandon_slot}`.  Workcell slots form a strictly increasing, duplicate-free integer array in `0..239`; `abandon_slot` is one such integer or `null`.

JSON integers exclude booleans.  IDs and enums are strings; missing optional values use JSON `null`. All objects reject undeclared fields.  Array order does not affect canonical coverage; the explicit keys and retry indices do.  All environment, queue, worker, packet, result, cache, process, workcell, energy, budget, and metric facts are grader-derived and must not be duplicated as required result fields.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain non-empty headings named exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Candidate-defined files may contain additional controlled experiments and condition-level evidence.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact scenario and decision containers", weight: 3}
- {id: result_evidence, desc: "request attempts, cache requests, workcell requests and abandonment are typed evidence", weight: 4}
- {id: canonical_coverage, desc: "all scenario, request-opportunity and weld keys are complete and unique", weight: 4}
- {id: decision_legality, desc: "retry order, path-worker relations, decision sequences and dynamic attempt eligibility are legal", weight: 4}
- {id: replay_validity, desc: "activation, deadlines, network, workers, process, workcells and budgets replay consistently", weight: 6}
- {id: primary_metrics_1, desc: "on-time weld completion and weighted process-stage progress", weight: 8}
- {id: primary_metrics_2, desc: "perception quality, deadline service and process safety", weight: 4}
- {id: primary_metrics_3, desc: "workcell flow, interventions and local-compute energy", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful remote-bit delivery, return success and freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "communication, remote-compute and queue efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the coupled perception-compute-network-welding formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent welding-process and remote-perception formulation; Implementation/evidence: no corresponding executable model or records."}
    - {score: 0.4, desc: "Report: a decisive request, deadline, quality, queue, worker, workcell or information relation is missing; Implementation/evidence: only a partial or materially divergent model is implemented."}
    - {score: 0.7, desc: "Report: the core requests, paths, workers, communication, quality, retries, workcells and deadlines are accurately related; Implementation/evidence: the model operates across every canonical scenario with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: event order, information boundaries, resource contention, process transitions, objectives and boundary cases are precise; Implementation/evidence: code and condition-level records faithfully expose these coupled effects without material divergence."}
- id: method_design
  desc: "quality of the unified state-aware placement, recovery and workcell scheduler"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal executable method or a prohibited method is proposed; Implementation/evidence: no reusable scheduling procedure supports it."}
    - {score: 0.4, desc: "Report: the design ignores a decisive temporal, quality or resource trade-off, or primarily presents unit-specific schedules; Implementation/evidence: decisions are partial, pre-filled or materially inconsistent with the report."}
    - {score: 0.7, desc: "Report: a legal, coherent and reproducible method addresses the principal placement, priority, retry, fallback and workcell decisions; Implementation/evidence: one state-driven procedure is used across all units, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified method treats the central process urgency, communication, compute, quality and physical-resource consequences; Implementation/evidence: evolving allowed state consistently drives decisive choices without unit-specific result tables or material divergence."}
- id: experimental_coverage
  desc: "coverage of process states, communication-compute contention and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: experiments are isolated, incomplete or limited to uninformative aggregates; Implementation/evidence: records do not test the central scheduling relation."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one controlled condition beyond canonical aggregates test a central mechanism or trade-off; Implementation/evidence: comparable conditions are actually executed and recorded."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check; Implementation/evidence: reproducible condition-level records and metrics distinguish both claims."}
- id: research_analysis
  desc: "quality of evidence-based process, latency, quality and resource analysis"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts results; Implementation/evidence: conclusions cannot be supported."}
    - {score: 0.4, desc: "Report: values are repeated without explaining mechanisms, trade-offs or limits; Implementation/evidence: records do not support the asserted explanation."}
    - {score: 0.7, desc: "Report: actual evidence explains important condition-dependent process, quality, queue, worker and workcell trends while separating observation, inference and limitations; Implementation/evidence: code and results support the comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence explains mechanisms, quantifies principal trade-offs or sensitivities, tests the explanation against contrasting evidence or a failure boundary, and limits scope; Implementation/evidence: reproducible condition-level results support the central conclusions."}
