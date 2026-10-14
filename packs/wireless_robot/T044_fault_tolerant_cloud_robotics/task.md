---
id: T044_fault_tolerant_cloud_robotics
name: Fault-Tolerant Replicated Cloud Services for Robotic Pick-and-Place
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

A mobile manipulator must scan, plan for, pick, and deliver released objects on a bounded factory grid. Scan and motion-plan computation is provided by three replicated edge-cloud services reached through independent, delayed links. A request may be sent to one or two replicas, and the first timely response with the current information and map versions advances the robot task. Extra replicas improve resilience to loss, queue contention, regional slowdown, and server preemption, but consume communication and cloud capacity. Meanwhile, obstacle-map changes invalidate old plans, deadlines continue to advance, and a robot carrying an object cannot safely move after its plan becomes stale. The resulting communication policy, service recovery, and robot route must therefore be designed as one closed loop.

### Research Problem

Design one deterministic, non-learning, state-driven procedure for all fifteen conditions. At each slot it may dispatch, cancel, or locally replace a cloud scan; it also selects one grid motion and one manipulation action. The procedure should adapt replication and retries to already received service reports and responses, maintain information and map versions, and schedule robot motion and pick-and-place work before object deadlines. It must trade completion value and cloud-stage reliability against replicated traffic, queue delay, local fallback, motion energy, and collisions. The same reusable procedure must operate from current observations and history rather than a table indexed by condition, variant, object, stage, or absolute slot.

### Experimental Setting and Constraints

The following subpoints are the complete public model and delivery contract. Any deterministic implementation satisfying them is valid; no private reference policy or hidden result representation is required.

#### Canonical evaluation units

The base conditions are `stable_replicas`, `spot_preemption`, `regional_slowdown`, `queue_contention`, and `recovery_overlap`. Each uses `variant v in {0,1,2}`, giving the fifteen canonical keys `<scenario>_v<variant>`. A condition has action slots `t=0..149` and boundaries `t=0..150`; each slot represents `1 s`.

The grid contains integer cells `x=0..11`, `y=0..9`. The robot starts at `(0,0)`, the placement cell is `(11,0)`, initial energy is `145`, and capacity is one carried object. The static blocked cells are `(4,2)`, `(4,3)`, `(4,4)`, `(4,6)`, `(4,7)`, `(7,1)`, `(7,2)`, `(7,7)`, and `(7,8)`.

| object | initial cell | value | base release slot | base deadline slot |
|---|---:|---:|---:|---:|
| `o1` | `(2,2)` | 1.00 | 0 | 43 |
| `o2` | `(5,1)` | 1.15 | 0 | 54 |
| `o3` | `(3,6)` | 1.10 | 10 | 72 |
| `o4` | `(6,4)` | 1.30 | 16 | 86 |
| `o5` | `(8,8)` | 1.00 | 22 | 105 |
| `o6` | `(1,8)` | 1.05 | 30 | 116 |
| `o7` | `(10,6)` | 1.35 | 38 | 132 |
| `o8` | `(9,2)` | 1.25 | 45 | 145 |

For variant `v`, every release is its base slot plus `2v`, every deadline is its base slot minus `2v`, and phase is `p=p_s+5v`. All intervals below are left-closed and right-open, and every listed interval endpoint and obstacle-event slot is shifted by `2v`.

| scenario | `p_s` | replica preemption | regional slowdown | background jobs | dynamic obstacle and base event slots |
|---|---:|---|---|---|---|
| `stable_replicas` | 1 | none | none | replica `j` adds one unit job when `(t+2j+p) mod 9=0` | add/remove `(5,5)` at `46,82` |
| `spot_preemption` | 4 | `R1:[30,38)`, `R3:[74,82)` | none | same as `stable_replicas` | add/remove `(6,5)` at `42,90` |
| `regional_slowdown` | 7 | none | `R2:[32,70)` | same as `stable_replicas` | add/remove `(8,3)` at `44,88` |
| `queue_contention` | 10 | none | none | `R1`: `(t+j+p) mod 3=0`; `R2`: `(t+2j+p) mod 4=0`; `R3`: `(t+3j+p) mod 5=0` | add/remove `(3,5)` at `52,94` |
| `recovery_overlap` | 13 | `R1:[36,44)`, `R2:[40,48)` | none | replica `j` adds one unit job when `(t+2j+p) mod 6=0` | add/remove `(6,6)` at `48,92` |

Here and below replica `Rj` has numeric index `j in {1,2,3}`. A dynamic obstacle set starts empty and `map_version=0`. The first event adds the listed cell and the second removes it; each event increments the map version by one. An addition does not retroactively cause a collision.

#### State, actions, and task model

Each released object follows `scan -> plan -> pick -> place`. Its information version starts at `0`. The first timely valid cloud scan response, or completion of its one permitted local scan, changes that version to `1`; it is never scanned again. A plan response is usable only for information version `1` and the current map version. A map event immediately invalidates every plan from an older map version, including a plan for a carried object. A subsequent current-version plan may restore motion permission.

A `scan` request carries `360 bit` and needs one service unit. A `plan` request carries `520 bit` and needs two consecutive service units. A dispatch selects one or two distinct replicas. It creates a request group with fixed ID `<condition_id>|<object_id>|<stage>|g<dispatch_slot>` and one attempt per selected replica, with attempt ID `<request_group_id>|rj`. These IDs and every lifecycle event are derived by replay; they are not submitted. At most one current-version request group for an object and stage may be active. A map change does not invalidate an active `scan` group; it remains the current scan group until it is accepted, times out, or is cancelled (terminal outcomes of individual attempts alone do not end the group). An old-version plan group stops blocking a current-version dispatch after a map change, although its attempts continue and can only yield a stale or failed response.

For a request dispatched at slot `g` to replica `j`, uplink latency is

`1+((5g+2j+p) mod 3)+s_j(g)`,

and uplink loss occurs when `(11g+5j+3p) mod 23 < 1+l_j(g)`. A response generated by service completion in slot `h` has downlink latency

`1+((3g+j+2p) mod 2)+s_j(g)`,

and is lost when `(7h+4j+5p) mod 29 < 1+l_j(h)`. Outside the shifted `regional_slowdown` interval, `s_j(x)=l_j(x)=0`; inside it, only `R2` has `s_2(x)=2` and `l_2(x)=2`. In the downlink-latency formula, `s_j` is evaluated at dispatch slot `g`; in the downlink-loss formula, `l_j` is evaluated at service-completion slot `h`.

Each replica can contain at most `12` total jobs, counting an unfinished in-service job and all waiting jobs. At a slot, new background jobs and robot attempts that have completed their uplink are ordered lexicographically by `b/<replica>/<slot>` and `r/<request_group_id>/<replica>` and admitted in that order. An excess background job is dropped; an excess robot attempt is rejected. FIFO order is then preserved. An available replica executes at most one service unit per slot. Once a two-unit plan starts, it occupies that slot and the next unless preempted. A response is generated in the final service slot and receives its fixed downlink outcome; later preemption does not revoke it. A downlink-lost attempt reaches its terminal outcome at its scheduled response-arrival boundary.

At a shifted preemption lower bound, the replica becomes unavailable, its background jobs are cleared, and queued or in-service robot attempts become preempted. At the shifted upper bound it recovers as an available empty replica. Every available replica reports at the end of every slot. That report arrives at the next boundary and contains `replica`, `reported_slot`, the literal availability value `up`, the current map version, and queue length including an unfinished service and all waiting jobs. An unavailable replica sends no report.

A request group dispatched at `g` accepts a response delivered at boundary `g+12`, but if no response has been accepted after delivery at that boundary, the group times out. Responses delivered after the group timeout or object deadline are late; a response with the wrong information version, or a plan response with the wrong map version, is stale; after one response from a group has been accepted, another valid response from that group is duplicate. Responses due at the same boundary are processed in attempt-ID lexicographic order. The first remaining timely, version-matching response is accepted. A cloud scan acceptance changes information version to `1`; a plan acceptance installs a plan for its response map version. For each `(object,stage)`, replay also retains the earliest accepted response and its delay for the cloud-stage metric.

A cancel action identifies the current `(object,stage)` group. It occurs before same-slot network admission and service, cancels every attempt in that group that has not begun service and has not generated a response, and makes the group inactive. Attempts already in service or awaiting a generated response continue. Uplink loss, queue rejection, preemption, cancellation, downlink loss, timeout, staleness, duplication, and acceptance are fixed terminal outcomes; a terminal outcome never changes.

The motion actions are `N`, `E`, `S`, `W`, and `wait`, changing the proposed cell by `(0,1)`, `(1,0)`, `(0,-1)`, `(-1,0)`, and `(0,0)`. Each directional action costs one energy unit whether it succeeds or collides. A proposed out-of-bounds, statically blocked, or currently dynamically blocked cell causes one collision for the slot and leaves the robot in place. `wait` costs no motion energy. A carried object may make a directional move only while it has a plan for the current map version.

`pick` and `place` each cost two energy units, require `motion=wait`, and require no simultaneous service action. A pick requires no currently carried object, the robot at the unique released object's initial cell, a completed scan, a current-map plan, and a nonexpired object. A place requires a carried object and robot position `(11,0)`. A dispatch or cancel may run with ordinary motion. Energy must be sufficient before the slot cost and may never become negative.

A local scan may be started once for an object only when the object is released and unscanned, the robot is at its cell, no current scan group is active, an earlier cloud scan group for it has timed out, and `t+2` is no later than the object deadline. Starting it consumes that object's one scan opportunity, so no cloud scan for the same object may be dispatched afterward. It costs four energy units at start and occupies its start slot and the following two slots. Motion must be `wait` and manipulation `none` throughout those three slots; dispatch or cancel for another object remains allowed during the latter two. The scan becomes available after the third occupied slot and does not replace planning.

Deadlines are inclusive: a response delivered, pick executed, or place executed in slot `t<=deadline` may count. At the start of a boundary with `t>deadline`, an unplaced object expires and its current groups become inactive. If it was carried, it is removed from the gripper so the robot can continue, while its earlier pick remains eligible for partial-completion credit. An expired object cannot be scanned, planned, picked, or placed. Existing attempts continue their lifecycle, but their responses are late.

#### Information availability and event order

At action boundary `t`, the decision procedure receives only: responses actually delivered at that boundary; for every released object, its cell, value, deadline, information/scan-ready state, current plan-map version (or no plan), expiration flag, and picked/placed status; robot position, energy, and carried object; current map version and static/dynamic blocked cells within Manhattan distance two; the latest successfully delivered status report for each replica, including `availability`, `reported_slot` (hence report age), `map_version`, and reported queue length; active request groups, remaining dispatch budget, local-scan state; and its own retained action and observation history.

The decision procedure does not receive the condition ID, variant, phase, future object releases, future preemptions or recovery, future dynamic-obstacle events, future network outcomes, future background jobs, current unreported replica internals, or future task outcomes. Although the fixed formulas are public so an outer simulator and grader can reproduce the experiment, a controller must not call, precompute, or reconstruct their current or future realizations. Only events already exposed through the listed observations may affect its decision.

Each slot uses this exact order:

1. Deliver due non-lost responses in attempt-ID order and the previous slot's status reports. Classify and apply each response.
2. Apply preemption starts, recovery ends, the dynamic-obstacle event, and object releases for this boundary. Expire unplaced objects only when `t>deadline`.
3. After delivery at boundary `g+12`, time out any still-unaccepted group dispatched at `g`; invalidate old map-version plans and update the observable state.
4. Expose the permitted information and read one service action, one motion, and one manipulation action.
5. Apply cancellation or create dispatch/local-scan state; admit due uplinks and background jobs; execute one service unit per available replica; generate responses and end-of-slot status reports.
6. Execute robot motion or manipulation, update local scan, energy, collisions, stage state, and object completion.
7. Retain the derived state for the next boundary.

Boundary `150` only delivers already generated responses and assigns `late` to attempts that still have no terminal outcome; it has no action or service. Decision fields are action-start submissions. Responses and reports are observations already visible at that boundary. Queues, versions, robot and object state, collisions, energy, lifecycle outcomes, and metrics are execution results derived by fixed replay.

#### Legal actions and boundaries

Every condition contains all `150` actions. A condition may dispatch at most `56` replica attempts, so a two-replica dispatch consumes two units. A dispatch is legal only for a released, nonexpired object whose requested stage is currently needed, with no current-version active group for that object and stage; `plan` additionally requires scan information version `1`. The selected replicas must be one or two distinct members of `{R1,R2,R3}`. A cancel must identify an active current group. A local scan must satisfy all fallback and three-slot conditions above.

During local-scan occupancy, motion and manipulation restrictions apply before any spatial execution. Pick, place, carried motion, stage/version dependencies, the one-object carrying limit, dispatch budget, and energy are hard legality conditions. An illegal action is not silently repaired: the relevant legality atom fails, while deterministic diagnostic replay substitutes no service action, `wait`, and `none` only for the invalid component so other failures remain inspectable. A legal directional action into a blocked or exterior cell is not a legality failure; it is executed as a collision event with the fixed energy cost. Fixed IDs, slot integers, strings, nulls, and arrays are exact. Numeric comparisons use absolute tolerance `1e-6`.

#### Task metrics

All metrics are computed per condition by independent replay and then averaged equally across the fifteen conditions. Define `clip(x,l,h)=min(h,max(l,x))`.

`primary_metrics_1` is deadline completion value. For each object, award `1` if it is placed by its deadline, `0.25` if it was picked by its deadline but not placed by the deadline, and `0` otherwise. Take the object-value-weighted mean.

`primary_metrics_2` uses the fixed sixteen `(object,scan|plan)` cloud stages. A stage success is `1` if it has at least one timely version-matching accepted cloud response and `0` otherwise. For a successful stage, timeliness is `clip(1-accepted_delay/12,0,1)`, where `accepted_delay` is response-arrival boundary minus dispatch slot for that stage's earliest accepted response; otherwise it is `0`. The condition score is `0.7*mean(success)+0.3*mean(timeliness)`.

`primary_metrics_3` is `0.75*clip(1-collision_slots/150,0,1)+0.25*clip(1-motion_energy/150,0,1)`. A slot contributes at most one collision; `motion_energy` counts every submitted legal directional action, including one that collides.

`auxiliary_metrics_1` is `0.6*useful_cloud_bits/max(sent_cloud_bits,1)+0.4*clip(1-dispatches/56,0,1)`. Each dispatched replica attempt contributes its stage payload to `sent_cloud_bits`. Only the attempt that produces the earliest accepted response for one of the sixteen cloud stages contributes its payload to `useful_cloud_bits`; when nothing is dispatched, the first term is `0`. `dispatches` counts replica attempts rather than request groups.

`auxiliary_metrics_2` is `0.6*clip(1-mean_queue_wait/12,0,1)+0.4*clip(1-fallback_count/8,0,1)`. For every robot attempt that begins service, queue wait is `service_start_slot-arrival_slot`; their mean is used, or `12` when none begins service. `fallback_count` is the number of local scans started.

All five values are finite and clipped to `[0,1]`. Normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Minimal result contract

`results.json` is strict finite JSON with exactly the root key `conditions`. Its array contains exactly one object for each canonical `condition_id`, in any order. Each condition object has exactly `condition_id` and `slots`. `slots` contains exactly one object for every integer `slot=0..149`, in any order.

Each slot object has exactly `slot`, `motion`, `service_action`, and `manipulation`. `slot` is an integer. `motion` is one of `N`, `E`, `S`, `W`, `wait`; `manipulation` is one of `none`, `pick`, `place`. `service_action` has exactly `kind`, `object_id`, `stage`, and `replicas`:

- For `kind=dispatch`, `object_id` is one of `o1..o8`, `stage` is `scan` or `plan`, and `replicas` is a one- or two-element array of distinct strings from `R1,R2,R3`; replica order has no semantic effect.
- For `kind=cancel`, `object_id` and `stage` identify the current group, and `replicas` is empty.
- For `kind=local_scan`, `object_id` is one of `o1..o8`, `stage=scan`, and `replicas` is empty.
- For `kind=none`, `object_id=null`, `stage=null`, and `replicas` is empty.

These actions are the only candidate evidence. Request IDs, network and queue events, responses, status reports, versions, robot state, object state, energy, collisions, summaries, and metrics must not be added to `results.json`; the grader derives them from the public model. Equivalent implementations receive the same result whenever their submitted actions have the same semantics.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain three readable ordinary files, not symbolic links: `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the information boundary, one reusable cloud-stage and robot-control policy, all canonical units, at least one controlled comparison or failure-boundary analysis, and supported limitations. Code and report must keep environment event generation outside the decision procedure and must not pass future releases, failures, queues, links, obstacles, or outcomes into a decision.

Evidence for Judge-only controlled comparisons may be supplied as condition-level tables in `report.md` or as additional readable text/JSON artifacts. These optional artifacts do not alter the exact `results.json` contract. For each claimed comparison, identify both executed conditions or decision regimes and give condition-level metrics or trajectory-derived quantities so both sides can be checked from code and records.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required non-empty headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON, the exact root structure, and exact condition and slot shapes", weight: 3}
- {id: result_evidence, desc: "condition and slot records contain the exact typed service, motion, and manipulation decisions", weight: 4}
- {id: canonical_coverage, desc: "all fifteen conditions and all 150 action slots occur exactly once", weight: 4}
- {id: decision_legality, desc: "stage, version, group, replication, manipulation, scan, dispatch-budget, and energy rules hold", weight: 4}
- {id: replay_validity, desc: "fixed replay derives consistent links, queues, failures, versions, robot execution, and task state", weight: 6}
- {id: primary_metrics_1, desc: "deadline-weighted pick-and-place completion value", weight: 8}
- {id: primary_metrics_2, desc: "timely accepted cloud-stage reliability", weight: 4}
- {id: primary_metrics_3, desc: "collision safety and motion-energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful replicated cloud traffic and dispatch efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "queue-wait and local-fallback efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the replicated cloud-service, versioned robot-stage, failure, and deadline model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent cloud-stage and robot-task formulation, or a materially wrong model; Implementation/evidence: no executable replicated service or pick-and-place process corresponds to the claims."}
    - {score: 0.4, desc: "Report: describes only replication or robot motion and omits a decisive first-response, version, queue, failure-recovery, or deadline relation; Implementation/evidence: state, information boundary, or event order materially departs from the public process."}
    - {score: 0.7, desc: "Report: accurately defines cloud stages, local information, links, queues, first response, replica failure/recovery, map versions, robot actions, and deadlines; Implementation/evidence: one reusable process implements the core model across all conditions with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely closes event order, information boundaries, service and network state, timeout and version invalidation, robot execution, energy, deadlines, and resource trade-offs; Implementation/evidence: all conditions faithfully realize those relations with reconstructible decisions and no future-event leakage or material divergence."}
- id: method_design
  desc: "quality of unified replication, retry, version, scheduling, and pick-and-place decisions"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training is used; Implementation/evidence: no runnable unified cloud and robot procedure is present."}
    - {score: 0.4, desc: "Report: relies on an object/request table, always uses one replication level, or ignores stage dependencies, status age, version invalidation, recovery, or carrying constraints; Implementation/evidence: decisions are pre-filled, use future environment events, or materially differ from the report."}
    - {score: 0.7, desc: "Report: gives a legal reproducible method for stage scheduling, adaptive dispatch or retry, current-version planning, navigation, and manipulation; Implementation/evidence: the same state-driven procedure covers all conditions without future-event access, with limited recovery or resource planning depth."}
    - {score: 1.0, desc: "Report: justifies replication value using observed status and urgency while jointly treating timeouts, queue age, failure recovery, version changes, deadlines, position, carrying, budget, and energy; Implementation/evidence: evolving permitted observations drive corresponding request and motion differences without condition, object, stage, or slot action tables."}
- id: experimental_coverage
  desc: "coverage of replication, preemption, network slowdown, queue contention, versions, and robot deadlines"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose; Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: only a stable replica or isolated object result is summarized and fault-tolerant task effects are not tested; Implementation/evidence: condition records cannot support the main claims."}
    - {score: 0.7, desc: "Report: covers every canonical condition and includes one explicitly identified comparable two-sided experiment on replication, preemption, queues, network, versions, or deadlines; Implementation/evidence: both sides execute with condition-level cloud and robot outcomes."}
    - {score: 1.0, desc: "Report: complementary non-redundant controlled experiments separately test replication behavior and a sensitivity or failure boundary, explaining interactions among recovery, contention, versions, and stage deadlines; Implementation/evidence: each side executes with reconstructible condition-level services, requests, trajectories, and metrics beyond aggregate scores."}
- id: research_analysis
  desc: "evidence-based interpretation of first responses, recovery, versions, completion, and resource trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records; Implementation/evidence: conclusions have no supporting execution evidence."}
    - {score: 0.4, desc: "Report: mainly repeats completion, latency, or dispatch counts, makes unsupported causal claims, and omits material limitations; Implementation/evidence: records do not support the asserted explanation."}
    - {score: 0.7, desc: "Report: uses actual results to explain replication, queue, recovery, version, deadline, navigation, and cost trends while separating observations, inference, and limitations; Implementation/evidence: executed comparisons support the explanations."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify interactions among replica selection, recovery timing, queues, map invalidation, stage deadlines, robot completion, and cloud cost, and states scope and unknowns; Implementation/evidence: condition-level records support the central conclusions."}
