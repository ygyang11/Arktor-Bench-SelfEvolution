---
id: T043_event_triggered_robotic_fish
name: Event-Triggered Acoustic Feedback for Underactuated Robotic-Fish Heading Control
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

An underactuated robotic fish cruises through a bounded water channel while an external localization service can send occasional state measurements over a low-rate acoustic link. The fish has a fixed forward speed but its yaw-rate response has inertia, so an old or missing measurement can make a turning command accumulate cross-track error. Water-current disturbances vary during the mission, and the channel becomes narrower in one family of conditions. A controller that transmits at every opportunity can exhaust its energy and communication budget, while one that waits too long may miss a turn or leave the safe channel. The experiment models this closed loop with fifteen public conditions, delayed and lossy measurements, and a finite mission horizon.

### Research Problem

Design one deterministic, non-learning, state-driven event-triggered controller for all conditions. At every action slot it chooses a bounded yaw command and whether to request one external measurement. The controller may maintain its own predictor and history, but it may use only the local observation, public route information, already revealed measurement feedback, and its retained history at that boundary. It must jointly manage underactuated response, current disturbance, heading and cross-track tracking, measurement age, trigger spacing, energy, and channel safety. The same reusable procedure must generate every condition instead of a table indexed by condition, variant, segment, or absolute slot.

### Experimental Setting and Constraints

The displayed subpoints are the complete public model and delivery contract. Any deterministic implementation satisfying them is valid; no private policy, reference representation, or hidden replay evidence is required.

#### Canonical evaluation units

The base conditions are `single_turn`, `double_turn`, `current_bias`, `rate_drop`, and `narrow_channel`; each has `variant=0,1,2`, for fifteen canonical keys `<scenario>_v<variant>`. Each condition has action slots `t=0..179` and state boundaries `t=0..180`, with slot duration `0.1 s`. The initial true state is `(q,e,psi,r)=(0,0,0,0)`, where `q` is progress along the channel centerline in metres, `e` is signed cross-track error in metres, `psi` is heading in radians, and `r` is yaw rate in radians per second. Forward speed is `0.35 m/s`, the initial energy is `72`, the robot radius is `0.18 m`, and the control alphabet is the continuous interval `u_t∈[-1,1]`.

| scenario | reference headings `psi_i` (rad) | segment targets `q_i` (m) | deadlines | width `W` (m) | current amplitude / bias | base phase / loss threshold |
|---|---|---|---|---:|---|---|
| `single_turn` | `0, 0, 0.32, 0.58, 0.58` | `1.0, 2.0, 3.1, 4.2, 5.6` | `34,66,102,138,170` | `1.30` | `0.025 / 0` | `2 / 1` |
| `double_turn` | `0, 0.36, 0.68, 0.22, 0` | `1.0, 2.0, 3.1, 4.2, 5.6` | `34,68,106,142,172` | `1.24` | `0.030 / 0` | `5 / 2` |
| `current_bias` | `0, 0.24, 0.50, 0.24, 0` | `1.0, 2.1, 3.2, 4.4, 5.7` | `36,70,108,146,174` | `1.22` | `0.028 / 0.020` | `8 / 2` |
| `rate_drop` | `0, -0.30, -0.58, -0.30, 0` | `1.0, 2.0, 3.1, 4.2, 5.6` | `34,68,106,142,172` | `1.28` | `0.026 / 0` | `11 / 4` |
| `narrow_channel` | `0, 0.22, 0.46, 0.46, 0.12` | `1.0, 2.0, 3.0, 4.1, 5.5` | `34,66,102,138,170` | `0.92` | `0.024 / 0` | `14 / 3` |

All five segment indices are `i=0..4`. For variant `v`, offsets apply equation by equation: the parallel-current offset is `7v`, the nominal lateral-current offset is `phase_s+7v`, the yaw-disturbance offset is `phase_s` with no variant term, the response-delay offset is `phase_s+5v`, and the loss-test offset is `phase_s+7v`. No offset is applied twice. In `narrow_channel`, the width is additionally reduced by `0.06v m` during segments `i=2` and `i=3`. The active segment is the first target not yet reached; its reference heading is `psi_ref`. For the environment-truth terms below, define the integer seed `k=phase_s+17v` and the dimensionless stream `h(k,t)=(((37*(t+1)+53*k+11*(t+1)*k) mod 101)-50)/50`.

#### State, action, and motion model

Throughout the fixed model, `clip(x,l,h)=min(h,max(l,x))`, `wrap(x)=((x+pi) mod (2*pi))-pi` has range `[-pi,pi)`, and `sign(x)` is `-1`, `0`, or `1` for negative, zero, or positive `x`, respectively. At slot `t`, the fixed environment applies the submitted `u_t` after the communication event. With `u_-2=u_-1=0`, it computes

`r_{t+1}=clip(0.82*r_t+0.35*u_{t-2}+0.05*sin(2*pi*(t+phase_s)/27)+0.08*h(k+19,t), -1.2, 1.2)`;

`psi_{t+1}=wrap(psi_t+0.1*r_{t+1})`;

`q_{t+1}=clip(q_t+0.1*(0.35*cos(psi_t-psi_ref)+c_parallel), 0, q_4)`;

`e_{t+1}=e_t+0.1*(0.35*sin(psi_t-psi_ref)+c_lateral)`.

For the reproducible environment, `c_parallel=0.015*sin(2*pi*(t+7v)/36)` and `c_lateral=b_s+a_s*sin(2*pi*(t+phase_s+7v)/30)+0.08*h(k,t) m/s`, using the row's amplitude `a_s` and bias `b_s`. The `0.08*h(k,t)` lateral-current term and the `0.08*h(k+19,t)` yaw-rate term above are realized motion innovations: the outer simulator and grader evaluate them, but their current or future values are not passed to the controller. The action must satisfy `-1≤u_t≤1`; there is no unstated action-rate or quantization rule. Control energy for the slot is `u_t^2`, and sending an admitted measurement consumes one additional energy unit. Total energy must remain nonnegative. The robot's safe boundary is `abs(e_t)≤W_t/2-0.18`; a strict violation is a safety event. The fixed replay evaluates the resulting boundary `t+1`: its active segment, width, and reference heading are determined from `q_{t+1}`, and only the first not-yet-completed segment may complete. A segment completes at the first resulting boundary satisfying `q_t≥q_i`, `abs(wrap(psi_t-psi_i))≤0.25`, and the safe boundary; a completion after its deadline is late but still recorded. The final segment must be complete by boundary `180` for the mission to be complete.

#### Local information and measurement process

The controller sees at action start the local odometry `z_t=(q_t+b^q_t,e_t+b^e_t,wrap(psi_t+b^psi_t),r_t+b^r_t)`, public segment targets and headings, the latest revealed measurement feedback and its age, its remaining trigger and energy budget, and its own predictor history. Biases start at zero and after the action update as `b^q_{t+1}=clip(0.97b^q_t+0.001,-0.12,0.12)`, `b^e_{t+1}=clip(0.97b^e_t+0.002*sign(c_lateral)+0.008*h(k+43,t),-0.12,0.12)`, `b^psi_{t+1}=clip(0.96b^psi_t-0.003u_t+0.006*h(k+71,t),-0.20,0.20)`, and `b^r_{t+1}=0.90b^r_t+0.01u_t+0.008*h(k+89,t)`. The added terms, in the units of their respective bias states, are independent realized odometry-bias innovations evaluated only by the outer simulator and grader. Consequently, local odometry alone cannot exactly separate realized motion disturbances from sensor drift; a delivered true-state measurement can recalibrate that separation.

If `trigger=true`, one new `240 bit` measurement request is admitted at the action boundary. After that slot's motion is executed, its payload stores the true state at the resulting boundary `g+1`; the payload is available to the controller only when its response arrives. The candidate does not submit the payload. At most `36` triggers are allowed in one condition, triggers must be at least `3` slots apart, and at most `1` generated attempt may be in flight at any time; a new trigger therefore requires that no earlier attempt remain in flight at that action boundary. For a sample generated at action slot `g`, its response boundary is `g+1+((g+phase_s+5v) mod 4)`. It is lost exactly when `(13g+7v+phase_s) mod 17 < loss_s`, where `loss_s` is the last column's threshold. Each trigger is a new single attempt; no retry field exists, so `generated_slot` uniquely identifies every sample in a condition. A response with age `arrival-g≤8` that is not lost is eligible. At an arrival boundary, eligible responses are processed in increasing `generated_slot`; only one whose generated slot is strictly newer than the current active measurement is accepted, and older responses cannot roll the active state back. Responses after boundary `180` are ignored for updates and metrics. The active measurement age is `t-generated_slot`, clipped to `8`; before the first valid response it is `8`.

#### Information boundary and event order

At each boundary `t`, the fixed process first reveals and processes responses due at `t`, then exposes the local observations and already revealed feedback, reads `trigger_t` and `u_t`, checks trigger spacing, in-flight count, budget and energy, and admits any legal request. It then advances the dynamics and local biases, samples the resulting boundary `t+1` state for the admitted request, and records that state, the safety event, sequential segment completion, energy and age. Boundary `180` reveals responses but has no action. The outer simulator must evaluate all five `h` streams to generate the reproducible environment, but the reusable decision procedure must not receive, call, precompute, or reconstruct their realized values; their effect becomes available only through later local odometry and delivered feedback. The nominal sinusoidal current model may be used for causal forecasts. Future truth-stream values, future true states, future losses, future arrivals and terminal outcomes are not observations. Decisions represent action-start submissions; measurements and all state, age, safety, completion and metric values are execution evidence derived by the fixed replay. Equally valid implementations are accepted whenever their submitted decisions have the same public semantics and satisfy the same constraints within `1e-6` numerical tolerance. If a submitted trigger is illegal, it is not admitted and the deterministic replay continues with the control cost only; the legality gate still fails, so such continuation is diagnostic rather than a way to obtain performance credit.

#### Legal actions and termination

`trigger` is Boolean and `u` is finite in `[-1,1]`. A trigger is illegal when it violates the three-slot spacing, the `36` trigger budget, the one-attempt in-flight capacity, or the remaining energy after `u_t^2+1`; a non-trigger action still pays `u_t^2`. Invalid actions fail only the relevant legality check and are not silently repaired. All 180 action slots are required even after the final segment completes. Integer and Boolean fields are exact; finite numeric comparisons use absolute tolerance `1e-6`.

#### Task metrics

Independent replay computes each metric per condition and averages across all fifteen conditions. The RMSE metrics use exactly the 180 resulting states at boundaries `1..180` (the initial boundary `0` is an initial condition, not an action sample). Let `C_t=W_t/2-0.18` be the positive safe cross-track clearance at a resulting boundary. The heading normalization `0.68 rad` is the largest absolute public reference heading, and `e_t/C_t` normalizes cross-track error by the boundary's fixed geometric clearance. Then `primary_metrics_1` is `0.6*clip(1-rmse_heading/0.68,0,1)+0.4*clip(1-rmse(e_t/C_t),0,1)`, where each heading error uses the reference heading of the segment active at that resulting boundary. `primary_metrics_2` is `0.7*completed_segment_ratio+0.3*timeliness`, where the ratio is the fraction of five sequential segments with a completion boundary and each timely term is `1` by its deadline, `0.5*clip(1-(completion-deadline)/max(1,180-deadline),0,1)` when late, or `0` when incomplete. `primary_metrics_3` is `0.7*clip(1-unsafe_slots/180,0,1)+0.3*clip(1-control_energy/180,0,1)`. `auxiliary_metrics_1` is `0.6*clip(1-trigger_count/36,0,1)+0.4*delivery_fraction`, where `delivery_fraction` is the number of non-lost responses arriving by boundary `180` divided by `trigger_count`, or `0` when no trigger was sent. `auxiliary_metrics_2` is the mean over action boundaries `t=0..179` of `clip(1-age_t/8,0,1)` after the responses at boundary `t` have been processed. All metric values are finite and in `[0,1]`; normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Minimal result contract

`results.json` is strict finite JSON with exactly the root key `conditions`. The array contains exactly one object for each of the fifteen public `condition_id` values, in any order. Each object has exactly `condition_id` and `slots`; `condition_id` is a string from the public canonical set. `slots` contains exactly one object for every integer `slot=0..179`, in any order. Each slot object has exactly `slot`, `trigger`, and `u`; `slot` is an integer, `trigger` is Boolean, and `u` is a finite JSON number in `[-1,1]`. The candidate submits no measurement, state, age, segment, energy, safety, or metric fields. The grader derives those facts from the action records and the fixed model, so no single reference representation or redundant self-reported summary is required.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain three readable ordinary files, not symbolic links: `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the information boundary, one reusable event-trigger and heading controller, all canonical units, at least one controlled comparison or failure-boundary analysis, and supported limitations. Code and report must separate the outer environment that evaluates truth innovations from the decision procedure, distinguish observations and revealed measurements from nominal-model forecasts, and must not pass truth-stream samples or future execution results into a decision. Official execution must be self-contained and must not depend on an external service or unpublished reference code.

Evidence for Judge-only controlled comparisons may be supplied as condition-level tables in `report.md` or as additional readable text/JSON artifacts. Such optional artifacts do not alter the exact `results.json` contract. For each claimed comparison, identify both executed conditions or decision regimes and provide their condition-level metrics or trajectory-derived quantities so the two sides can be checked from the submitted code and records.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required non-empty headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON, the exact root structure, and exact condition-object shape", weight: 3}
- {id: result_evidence, desc: "condition and slot records have the exact typed minimal decision fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen conditions and all action slots occur exactly once", weight: 4}
- {id: decision_legality, desc: "trigger spacing, budget, in-flight, control, and energy obey public rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives dynamics, measurements, segment events, safety, and metrics", weight: 6}
- {id: primary_metrics_1, desc: "heading and cross-track tracking quality", weight: 8}
- {id: primary_metrics_2, desc: "segment completion and deadline quality", weight: 4}
- {id: primary_metrics_3, desc: "channel safety and control-energy quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "event-trigger communication efficiency and delivery", weight: 3}
- {id: auxiliary_metrics_2, desc: "freshness of valid external measurements", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the underactuated fish, measurement, disturbance, channel, and deadline model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent heading-control and event-communication model or a materially wrong model; Implementation/evidence: no executable fish dynamics, measurement process, or channel task corresponds to the claims."}
    - {score: 0.4, desc: "Report: describes only control or trigger rate and omits a decisive underactuation, measurement-age, truth-innovation, or safety relation; Implementation/evidence: the state, information boundary, or event order materially departs from the public process."}
    - {score: 0.7, desc: "Report: accurately defines the fish state, heading segments, local information, delayed measurements, truth innovations, trigger limits, disturbances, and channel constraints; Implementation/evidence: one reusable process implements the core model across all conditions and keeps truth generation outside the controller, with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely closes action-start information, delayed measurement handling, truth/observation separation, underactuated transition, current phases, safety boundaries, energy, deadlines, and metric semantics; Implementation/evidence: all conditions faithfully realize these relations with reconstructible decisions, no truth-stream leakage, and no material divergence."}
- id: method_design
  desc: "quality of the unified event-trigger, predictor, and underactuated heading controller"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training is used; Implementation/evidence: no runnable unified controller or supporting decisions are present."}
    - {score: 0.4, desc: "Report: relies on fixed trigger or turn tables, ignores measurement age, prediction drift, saturation, or energy, or uses condition-specific actions; Implementation/evidence: the procedure is table-driven, feeds realized truth-stream values into decisions, or materially differs from the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning method jointly handling event triggering, nominal prediction, delayed feedback, bounded control, and channel safety; Implementation/evidence: the same procedure covers all conditions without truth-stream access and records support its principal choices, with limited planning depth."}
    - {score: 1.0, desc: "Report: justifies state-dependent trigger value, prediction correction under unobserved innovations, heading-segment and disturbance handling, input saturation, energy and safety trade-offs; Implementation/evidence: evolving local state and revealed feedback drive decisive actions without truth-stream access or condition/slot action tables."}
- id: experimental_coverage
  desc: "coverage of turns, disturbances, communication degradation, and channel boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or purpose is provided; Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: only one heading or communication condition is summarized and the event-trigger mechanism is not tested; Implementation/evidence: condition records cannot support the main claims."}
    - {score: 0.7, desc: "Report: covers all canonical conditions and includes at least one explicitly identified controlled comparison of triggering, disturbance, turn, or channel behavior; Implementation/evidence: both comparable conditions execute with condition-level records."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments separately test communication-trigger behavior and a sensitivity or failure boundary involving turns, disturbance, saturation, or channel width; Implementation/evidence: each condition is executed with reconstructible metrics beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based interpretation of prediction, triggering, underactuation, safety, and tracking trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records; Implementation/evidence: conclusions have no supporting execution evidence."}
    - {score: 0.4, desc: "Report: mainly repeats tracking or trigger counts, makes unsupported causal claims, and omits material limits; Implementation/evidence: records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: uses actual results to explain prediction drift, delayed feedback, current, saturation, channel safety, and completion trends while distinguishing observations and limitations; Implementation/evidence: executed comparisons support the explanations."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify interactions among trigger timing, predictor correction, underactuated response, current, energy, safety, and completion, and states scope and unknowns; Implementation/evidence: condition-level records support the central conclusions."}
