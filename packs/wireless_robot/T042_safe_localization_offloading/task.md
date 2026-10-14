---
id: T042_safe_localization_offloading
name: Risk-Aware Localization Scheduling for Edge-Assisted Robot Navigation
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

A mobile robot follows a route through a structured workspace while its localization estimate becomes less reliable during motion. Compact localization measurements can be sent to an edge service, but the shared service queues packets, delays replies, and occasionally loses them. The robot must choose between a short branch with less clearance and a longer branch with more clearance, regulate its speed, and decide when to send measurements. A stale or missing correction can make a narrow or moving bottleneck unsafe, while excessive measurement traffic and braking can prevent timely completion. The experiment studies fifteen public route, obstacle, and link-phase units in one finite, replayable model.

### Research Problem

Design one deterministic, non-learning, state-driven controller for all units. At every action slot it chooses the currently legal route branch when the common route reaches its branch point, a discrete speed, and whether to send one localization measurement. The controller should jointly manage uncertainty growth, delayed corrections, queue contention, energy, route clearance, and the completion deadline. It may use only the state and feedback exposed at the current action boundary and the history retained by its own procedure. The same reusable process must generate every unit rather than a table of actions indexed by a scenario, variant, or slot.

### Experimental Setting and Constraints

#### Canonical evaluation units

The displayed subpoints below are the complete public model and delivery contract. Any deterministic state-driven implementation satisfying them is valid; no private policy, reference implementation, or internal evidence layout is required.

The five base scenarios are `open_turn`, `narrow_corridor`, `moving_bottleneck`, `handover_gap`, and `energy_limited`. Each has `variant=0,1,2`, producing fifteen keys `scenario=<name>|variant=<v>`. A unit has action slots `t=0..119` and state boundaries `b=0..120`; each slot is `0.1 s`. The robot starts at route progress `p_0=0`, with `speed_-1=0`, uncertainty `u_hat_0=0.10`, and energy `80` except `energy_limited`, which starts at `62`. The speed alphabet is `{0.0,0.1,...,1.0} m/s`.

| scenario | branch a: length / nominal clearance / turn interval | branch b: length / nominal clearance / turn interval | deadline | base phase | lower nominal-clearance branch |
|---|---|---|---:|---:|---|
| `open_turn` | `7.8 / 1.40 / [3.6,4.8]` | `6.9 / 1.00 / [3.2,4.3]` | `108` | `2` | `b` |
| `narrow_corridor` | `8.2 / 1.20 / none` | `7.0 / 0.62 / none` | `112` | `5` | `b` |
| `moving_bottleneck` | `8.3 / 1.10 / [4.0,5.0]` | `7.2 / 0.76 / [3.5,4.5]` | `112` | `8` | `b` |
| `handover_gap` | `7.7 / 1.00 / [4.2,5.1]` | `7.1 / 0.78 / [3.8,4.6]` | `108` | `11` | `b` |
| `energy_limited` | `6.8 / 0.86 / none` | `8.1 / 1.30 / [4.4,5.6]` | `110` | `14` | `a` |

The final column only identifies the branch with the smaller *nominal* clearance in that row. It does not add a time-varying obstacle. The only dynamic reduced-clearance rules are the two affected branches and intervals stated explicitly below (`moving_bottleneck/b` and `energy_limited/a`).

The branch table is public route input, not a schedule of actions. Before the branch point, the route is common and progress follows `p_{t+1}=min(2.0,p_t+0.1*speed_t)`. The common-route nominal clearance is the public canonical value `c_s`, equal to the row's branch-`a` nominal clearance (`1.40, 1.20, 1.10, 1.00, 0.86 m` in the table order); it is used while `branch="undecided"`, and branch-specific dynamic reductions are inactive before selection. At the first decision boundary with `p_t>=2.0`, the controller chooses `a` or `b`; before that boundary it must submit `branch="undecided"`. Once chosen, the branch cannot change. A branch with no turn interval has heading `0`. Otherwise its heading is `0` before the interval, increases linearly to `0.70 rad` across the inclusive interval, and remains `0.70 rad` after it. For a slot, `omega_t=wrap(heading(p_{t+1})-heading(p_t))/0.1`.

#### Route, motion, and visible clearance

At slot `t`, the submitted speed is `speed_t` and `p_{t+1}=min(L_r,p_t+0.1*speed_t)`, where `L_r` is the selected branch length. The speed change from the previous submitted speed is at most `0.1 m/s`. Once `p_t=L_r`, the controller must submit speed `0`. The robot's current clearance is visible before the decision: it is `c_s` while the branch is undecided, and after selection it equals the selected branch's nominal clearance minus `0.35 m` exactly when `((t+phase_{s,v}) mod 24)∈{10,11,12,13,14,15}` and the current progress lies in `[3.8,5.0]` for the affected `moving_bottleneck` branch or `[4.4,5.6]` for the affected `energy_limited` branch; otherwise it is the selected branch's nominal clearance. The affected-branch rule is inactive in the other scenarios. The variant phase is `phase_{s,v}=phase_s+3v`; `phase_s` is the base phase in the table. A slot consumes `0.5*speed_t^2` energy, and a sent measurement consumes one additional energy unit. Energy may never become negative.

#### Measurement service and link events

A measurement has `300 bit` and at most `36` measurements may be sent in a unit. A `send=true` decision creates one new attempt at the end of that action slot. It is admitted only if the pre-send FIFO queue has fewer than six packets, the send-count limit is not reached, and the current energy covers the send cost; otherwise the decision is illegal. Admitted packets enter a FIFO edge queue. After the action, the queue serves at most one head packet; its service slot is the slot `q` at which service actually starts. A service beginning during `handover_gap` slots `t∈[44+4v,56+4v)` is paused at the head until the interval ends. For a packet generated at `g`, its response is scheduled for boundary `a=q+1+((g+phase_{s,v}+4v) mod 3)`. The packet is lost when `(17g+5q+11v+phase_s) mod 19 < loss_s`, where `loss_s` is `1,2,3,4,2` for the five rows in order. A lost response produces loss feedback at the same scheduled boundary; a non-lost response produces an uncertainty correction. Boundary `120` reveals due responses and loss feedback, records terminal replay evidence, and has no action; only responses at boundaries greater than `120` are ignored for control and metrics. There is no candidate retry field: each `send=true` is one attempt, and all attempt IDs, loss samples, service times, and response events are derived by replay.

#### Information availability and uncertainty

At action start the controller receives the unit's declared scenario/variant context, deadline, current branch and progress, current heading, current clearance, visible `u_hat`, remaining energy, current queue length, its sent-count history, and the age of the latest valid correction. It receives all response or loss feedback revealed at the current boundary and its own prior decisions. It does not receive future service slots, future losses, future responses, future obstacle phases, or terminal state. The public route table and deterministic transition laws may be evaluated for a forecast, but a forecast is not an observed event and may not be reported as one; no hidden truth or pre-filled action table may be queried.

At the beginning of a unit `u_hat=0.10`. After a legal action computes `p_{t+1}`, first calculate `omega_t` and then grow uncertainty at the slot end as `u_hat^-_t=clip(u_hat_t+0.015*speed_t+0.01*abs(omega_t),0,1)`. A response generated at `g` stores `u_hat_g`. At boundary `a`, valid responses are processed in ascending `(arrival_boundary,measurement_id)` order. For response age `age=a-g`, its correction is `u_corr=clip(0.35*u_hat_g+0.025*age,0,1)` and the visible uncertainty becomes `min(current_u_hat,u_corr)`. Responses arriving after boundary `120` are not valid updates. A correction is useful for the communication metric only when it strictly reduces uncertainty; its leave-one-out usefulness is computed by the grader on the same submitted action sequence after removing only that measurement and its correction.

#### Event order and evidence semantics

Every nonterminal slot follows this order: (1) reveal responses and loss feedback due at boundary `t`; (2) apply valid corrections and update the visible age; (3) expose current branch, progress, heading, clearance, uncertainty, queue, energy, and history; (4) read and validate `branch`, `speed`, and `send`; (5) admit a measurement and append it to the FIFO queue; (6) start service for at most one queue head and schedule its response; (7) advance route progress, heading, and uncertainty; (8) record clearance risk, uncertainty excess, energy, and deadline state. Boundary `120` only reveals due responses and performs no decision. Result fields are action-start decisions; responses, queue/service events, uncertainty, route state, risk, energy, useful bits, and metrics are execution or replay evidence derived after the fixed steps. Equivalent implementations are accepted when they submit decisions with the same public semantics and therefore have the same fixed-model consequences.

#### Legal actions and termination

`branch` is exactly `undecided`, `a`, or `b`; a branch is not allowed before `p_t>=2.0`, and a chosen branch cannot change. `speed` is one of the eleven public values, finite, and obeys the `0.1` slew bound. `send` is a JSON Boolean. Sending is illegal when the pre-send queue has six packets, the unit has already sent `36` measurements, or energy would become negative. Any action after arrival must use speed `0`; actions through slot `119` are recorded even if the route is already complete. Comparisons use absolute tolerance `1e-6`. A safety risk or uncertainty excess is an execution result, not an extra hidden legality condition; the fixed model records it rather than rejecting the action.

#### Task metrics

Metrics are computed per unit from fixed replay and then averaged over fifteen units. `primary_metrics_1` is `0.7*arrival_quality+0.3*timeliness`: `arrival_quality=1` exactly when the route is reached by the deadline with no clearance-risk slot, otherwise `0.5*clip(final_progress/L_r,0,1)`; `timeliness=1` for on-time arrival, `0.5*clip(1-(arrival-deadline)/max(1,120-deadline),0,1)` for late arrival, and `0` if never reached. `primary_metrics_2` is `0.6*clip(1-risk_slots/120,0,1)+0.4*clip(1-uncertainty_excess/120,0,1)`, where `risk_slots` counts post-action clearances below `0.45 m` and `uncertainty_excess=sum_t max(0,u_hat_t-u_required_t)` with `u_required=clip((clearance_t-0.45)/1.0,0,1)`. `primary_metrics_3` is `0.6*route_efficiency+0.4*clip(1-motion_energy/60,0,1)`; on arrival `route_efficiency=L_min/max(L_r,travel_distance)`, otherwise it is zero, with `L_min=min(L_a,L_b)` and `travel_distance=sum_t 0.1*speed_t`.

If a controller never reaches the branch point, no branch is selected and the unit cannot arrive; in that case the non-arrival progress term uses the fixed common-route denominator `2.0 m`, and `route_efficiency` is `0`. Once a branch is selected, `L_r` is that branch's public length.

#### Communication metrics

`auxiliary_metrics_1` is `0.6*useful_bits/max(sent_bits,1)+0.4*useful_bits/(36*300)`, where useful bits are `300` per distinct response that strictly lowers uncertainty and is useful under the grader's leave-one-out test, otherwise zero. Leave-one-out usefulness keeps the baseline action sequence and all other packet service, loss, and response outcomes fixed, removing only that measurement's correction, and compares `risk_slots` and `uncertainty_excess` accumulated over action slots `t=0..119`. A response revealed exactly at boundary `120` is valid replay evidence and may lower the terminal uncertainty, but no action slot follows it; under this slot-based leave-one-out test it therefore cannot by itself increase either counterfactual and contributes zero `useful_bits`. Only arrivals at boundaries greater than `120` are ignored. `auxiliary_metrics_2` is the mean over slots of `clip(1-age_t/6,0,1)`, where `age_t` is the age immediately after responses revealed at the action-start boundary `t` have been processed, and is `6` before the first valid correction. All five metrics are finite in `[0,1]`, and normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Minimal result contract

`results.json` is strict finite JSON with exactly root key `scenarios`. The array contains exactly one record for every canonical `(scenario,variant)` pair, in any order. Each unit has exactly `scenario`, `variant`, and `slots`. `scenario` is a string in the public set and `variant` is a JSON integer in `0..2` (Boolean is not an integer). `slots` contains exactly one record for every integer `t=0..119`, in any order; each record has exactly `t`, `branch`, `speed`, and `send`. `t` is an integer, `branch` is one of the three public strings, `speed` is a finite JSON number, and `send` is Boolean. Measurements, service slots, responses, uncertainty states, energy, route progress, risks, and all metrics are not candidate fields; the grader derives them from the minimal decisions. `result_json` checks strict root/unit structure, `result_evidence` checks primitive fields and finite values, `canonical_coverage` checks unit and slot sets, `decision_legality` checks branch/speed/send relations, and `replay_validity` checks the fixed event and state process. No candidate report number is used as evidence.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain three readable ordinary files, not symbolic links: `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the information boundary, one reusable route/measurement/speed controller, all canonical units, at least one controlled comparison or failure-boundary analysis, and supported limitations. Code and report must distinguish observations and revealed corrections from public-model forecasts and must not claim future execution results as observed evidence. Official execution must be self-contained and must not depend on an external service or unpublished reference code.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required non-empty headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenarios and unit structure", weight: 3}
- {id: result_evidence, desc: "unit and slot records have the exact typed minimal decision fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen units and all action slots occur exactly once", weight: 4}
- {id: decision_legality, desc: "branch timing, speed slew, send budget, queue admission, and energy obey public rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives service, responses, uncertainty, route, safety, and mission events", weight: 6}
- {id: primary_metrics_1, desc: "deadline-qualified route completion and progress", weight: 8}
- {id: primary_metrics_2, desc: "clearance-risk and localization-uncertainty safety quality", weight: 4}
- {id: primary_metrics_3, desc: "route efficiency and motion-energy quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful admitted localization communication efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "freshness of valid localization corrections", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the localization, route, queue, uncertainty, safety, and deadline model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent communication-control localization model or a materially wrong model; Implementation/evidence: no executable route, measurement, and motion process corresponds to the claimed model."}
    - {score: 0.4, desc: "Report: describes only speed or measurement frequency and omits a decisive uncertainty, queue, route, information, or safety relation; Implementation/evidence: the executable state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: accurately defines units, branch and speed decisions, visible uncertainty and feedback, queue and loss process, deadlines, objectives, and safety constraints; Implementation/evidence: one reusable process implements the core model across all units with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely closes action-start information, response order, uncertainty growth/correction, branch and clearance coupling, energy, metrics, and boundary conditions; Implementation/evidence: all variants faithfully realize those relations with reconstructible decisions and no material divergence."}
- id: method_design
  desc: "quality of the unified state-driven route, measurement, uncertainty, and speed controller"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training is used; Implementation/evidence: no runnable unified controller or supporting decisions are present."}
    - {score: 0.4, desc: "Report: relies on fixed branch/speed/measurement schedules or ignores decisive queue, correction age, energy, clearance, or deadline consequences; Implementation/evidence: decisions are table-driven by unit or materially differ from the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning method jointly handling branch choice, discrete speed, measurement triggering, delayed correction, and safety; Implementation/evidence: the same causal procedure covers all units and records support the principal choices, with limited planning depth."}
    - {score: 1.0, desc: "Report: justifies state-dependent uncertainty forecasts, correction value, queue and budget use, branch clearance, deadline feasibility, and energy/safety trade-offs; Implementation/evidence: evolving action-start state drives decisive choices without scenario-specific action tables, replay-only truth, or forbidden future outcomes."}
- id: experimental_coverage
  desc: "coverage of route, link phase, queue, uncertainty, safety, and deadline conditions"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose; Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: only one branch or one communication condition is summarized and the communication-safety relation is not tested; Implementation/evidence: condition evidence cannot support the main claims."}
    - {score: 0.7, desc: "Report: covers every canonical unit and includes at least one explicitly identified two-sided comparison of branch, measurement, queue, obstacle, energy, or deadline behavior; Implementation/evidence: both conditions execute on comparable inputs with condition-level records."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments separately test a measurement/queue mechanism and a route, uncertainty, energy, or safety failure boundary; Implementation/evidence: each condition is actually executed with reconstructible metrics beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based interpretation of localization freshness, route, safety, energy, and communication trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records; Implementation/evidence: conclusions have no supporting execution evidence."}
    - {score: 0.4, desc: "Report: mainly repeats scores or makes unsupported causal claims and omits material limitations; Implementation/evidence: execution conditions do not support the claims."}
    - {score: 0.7, desc: "Report: uses actual results to explain correction age, queueing, route clearance, uncertainty, deadline, energy, and safety trends while distinguishing observations, inferences, and limitations; Implementation/evidence: executed comparisons support the explanations."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify interactions among correction timing, queue competition, branch clearance, speed, energy, risk, and completion, and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
