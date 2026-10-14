---
id: T041_replicated_cloud_requests
name: Redundant Cloud-Feedback Routing for Deadline-Constrained Robot Tracking
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

A mobile robot tracks a moving target while cloud perception requests travel over three wireless interfaces to three edge replicas. Requests can be duplicated over distinct interface--replica paths, queued behind earlier traffic, lost, or completed after a tracking deadline. The robot has a local target observation at each request opportunity and may receive a cloud estimate later. It must use only information available at the current action boundary to choose communication actions and safe motion. The experiment compares fifteen public phase and failure variants in which latency, loss, interface outages, replica service, target motion, and obstacle geometry jointly affect feedback freshness, tracking, safety, and resource use.

### Research Problem

Design and evaluate one deterministic, non-learning, state-driven controller for the complete request stream. At each request opportunity it chooses an ordered set of at most two interface--replica paths and may start a bounded local fallback; it may retry each visibly failed selected path once on the next slot. At every slot it chooses a world-frame velocity target subject to the lagged motion and safety rules. The controller should balance timely first responses, response age, queue and uplink cost, fallback energy, target-tracking quality, and safe motion under correlated outages and service bursts. The same reusable procedure must generate all canonical units from current observations, revealed responses, queue state, prior decisions, and remaining resources. Scenario, variant, request index, and absolute slot may be used as declared context, but they must not index pre-filled actions or replay-only truth; future events must not be used.

### Experimental Setting and Constraints

#### Canonical evaluation units

The following displayed subpoints are the complete public model and delivery contract. Any deterministic state-driven implementation that satisfies these semantics is valid; no private policy, reference representation, or internal data layout is required. The five base scenarios are `balanced_links`, `handover_gap`, `correlated_outage`, `server_burst`, and `deadline_tight`. Each has variants `v=0,1,2`, giving the fifteen canonical keys `scenario=<name>|variant=<v>`. The workspace is `[0,14] x [0,10] m`. There is one holonomic robot, one moving target, interfaces `I0..I2`, replicas `C0..C2`, `H=90` action slots `t=0..89`, and state boundaries `b=0..90`. Each slot lasts `0.25 s`. Initial actual velocity and the previous submitted target velocity are both `(0,0) m/s`. Maximum target and actual speed is `0.8 m/s`, target-velocity slew is at most `0.05 m/s` per slot, safety clearance is `0.35 m`, and risk clearance is `0.60 m`.

| scenario | robot start | target start and velocity | obstacles | target deadline | link condition |
|---|---|---|---|---:|---|
| `balanced_links` | `(1,5)` | `(12,5)`, `(+0.02,0)` | circles `(6,3,0.8)`, `(6,7,0.8)` | `76` | baseline |
| `handover_gap` | `(1,2)` | `(12,8)`, `(0,+0.02)` | rectangle `[5,6]x[4,6]` | `80` | `I1` unavailable at `30+4v..38+4v` |
| `correlated_outage` | `(1,8)` | `(12,2)`, `(+0.01,+0.01)` | circle `(6,5,0.9)` | `82` | `I0` and `I1` unavailable at `42+4v..50+4v` |
| `server_burst` | `(1,5)` | `(12,5)`, `(0,+0.02)` | circles `(4,5,0.7)`, `(8,5,0.7)` | `78` | replica service takes four slots at `24+4v..35+4v` |
| `deadline_tight` | `(1,5)` | `(11,5)`, `(+0.02,0)` | rectangles `[5,6]x[0,3]`, `[5,6]x[7,9]` | `66` | baseline loss plus `0.10` |

All integer ranges written `a..b` in the table are inclusive. Interface coordinates are `I0=(2,1)`, `I1=(7,5)`, and `I2=(12,8)`. Replicas `C0..C2` are logical service endpoints; replica location is not part of any distance or delay. Variant `v` adds `v` to request phase, `4v` to the listed outage or burst phase, and `17v` to the loss-hash salt. Target truth at boundary `b` is `clip(target_start+b*velocity,workspace)` coordinatewise. The target starts, velocities, clipping rule, outage intervals, service intervals, and loss-hash formula are public deterministic model laws. A candidate may evaluate those laws to form a model-based forecast for planning, but a forecast is not an action-start observation or an execution event and must not be presented as revealed evidence. The policy must not query hidden simulator state or recover replay-only state through a private lookup table.

For a point `p`, workspace clearance is `min(p_x,14-p_x,p_y,10-p_y)`. Circle clearance is `||p-c||-r`. For rectangle `[x0,x1]x[y0,y1]`, outside clearance is `sqrt(max(x0-p_x,0,p_x-x1)^2+max(y0-p_y,0,p_y-y1)^2)`; a point inside or on the rectangle has clearance `-min(p_x-x0,x1-p_x,p_y-y0,y1-p_y)`. The point's fixed geometric clearance is the minimum workspace and obstacle clearance. The thresholds above are robot-center thresholds and already represent the footprint; no radius is added again. The listed static obstacle geometry is policy-visible.

#### State, requests, and local observations

Request opportunities are `g_v={v+3k | 0<=k<30}`. At `g`, the simulator generates request `scenario=<s>|variant=<v>|request=<k>` with request deadline `g+6` and a `400`-bit local observation. Let `d_o=clip(||p_g-target(g)||/3,0,1)`, and let line-of-sight occlusion `o` be `1` exactly when the closed segment from `p_g` to `target(g)` intersects a listed closed obstacle. Observation quality is `q=clip(1-0.12*d_o-0.15*o,0,1)`. With `theta=2*pi*((k+2v) mod 12)/12` and `e=1.20+0.80o+0.40d_o`, the finite world-frame local estimate is `clip(target(g)+e*(cos(theta),sin(theta)),workspace)` coordinatewise. At action start the policy receives the request ID, generated slot, deadline, estimate, and `q`; it does not receive target truth, `d_o`, or `o` as observations. The simulator may compute replay-only truth to produce the observation. The policy may combine the exposed values with forecasts from the public deterministic laws, but it must keep those forecasts distinct from observed execution facts.

#### Request, retry, and fallback decisions

At opportunity `g`, the policy submits an ordered list of zero, one, or two distinct `(interface_id,replica_id)` paths. Each selected path submits attempt `0` at `g`; the list contains the paths actually submitted, so there is no cancellation field. An attempt submitted at `s` reveals exactly one outcome at boundary `s+1`: `scheduled` when it was admitted and not lost, otherwise `lost`, `interface_down`, `budget_exhausted`, or `queue_rejected`. A path whose attempt-0 outcome is not `scheduled` may be retried once, on the same interface and replica, at exactly `g+1`; the retry is attempt `1` and is represented by one retry record. No scheduled path, unknown request, unselected path, or other send boundary may be retried.

A request chooses `fallback="none"` or `fallback="start"`. `start` begins at `g`, incurs `3` energy units immediately, completes at `g+2`, and then exposes a fallback estimate anchored to target truth at `g`. Its error magnitude is `0.35+0.04*2`, its direction is `2*pi*((k+v) mod 8)/8`, and its world position is clipped to the workspace coordinatewise. Local observations and completed fallback estimates remain available in policy history. The first timely cloud response is the canonical cloud result for its request; duplicate and late responses remain execution events but cannot replace it. A policy decision may reference only information exposed by its action boundary.

#### Network and replica transitions

Each admitted uplink attempt consumes `400` bits from a per-unit budget of `16800` bits. The fixed `200`-bit response is downlink traffic and does not consume this budget. At send boundary `s`, let `d` be robot-center distance to the interface and `d_n=clip(d/14,0,1)`. An admitted interface reservation is active on `[s,service_start_slot)` with capacity four, and its replica reservation is active on `[s,completion_slot)` with capacity six, including attempts later classified as lost. Retire reservations whose right endpoint equals the current boundary before taking its queue snapshot. `queue_wait` is the number of active reservations on the selected interface plus the number on the selected replica in that one pre-admission snapshot; it is not a sum of residual durations. Every attempt submitted at the same boundary uses this same `queue_wait` snapshot. Capacity and budget admission nevertheless process those attempts in ascending `(request_deadline,generated_slot,path_rank,interface_id,replica_id,attempt)` order, so earlier admitted attempts at that boundary consume capacity and budget for later ones. `path_rank` is the zero-based position in the request's ordered selected-path list and remains that path's rank on retry.

An attempt is rejected when its interface is down, its admission would exceed the uplink budget, or its interface or replica reservation capacity is already full at its turn. If more than one rejection condition holds, the revealed outcome uses priority `interface_down`, then `budget_exhausted`, then `queue_rejected`. `handover_penalty=1` exactly when the chosen interface is up at `s` but was down at `s-1`; otherwise it is zero. For an admitted attempt, `path_delay=1+queue_wait+floor(2*d/5)+handover_penalty`, `service_start_slot=s+path_delay`, and `completion_slot=service_start_slot+service_time`. Normal `service_time` is two slots. In `server_burst`, a service beginning in `24+4v..35+4v` takes four slots. The loss rate is `0.08+0.03*d_n+0.04*path_rank`, plus `0.10` in `deadline_tight`. The loss sample is the first sixteen hexadecimal SHA-256 characters interpreted as an unsigned big-endian integer and divided by `2^64`, using UTF-8 text `<scenario>|<variant>|<request_id>|<interface_id>|<replica_id>|a<attempt>|salt=<17v>`. This hash rule is public and may be evaluated as a model forecast, but its result is still an unobserved future event until the specified reveal boundary. A sample below the loss rate produces `lost` and no response; every other admitted attempt is `scheduled` and produces one response at completion.

At completion, let `phi=2*pi*((k+3i+5c+7a+v) mod 16)/16`, where `i`, `c`, and `a` are the integer suffixes of the interface, replica, and attempt identifiers. The full attempt ID is `<request_id>|<interface_id>|<replica_id>|a<attempt>`. The response estimate is `target(g)+0.08*(cos(phi),sin(phi))` in world coordinates. Responses completing at one boundary are processed by ascending full attempt ID. The first response for a request with `completion_slot<=min(request_deadline,90)` is accepted; all later or ineligible responses are `late_or_duplicate`. Only an accepted first response is control-usable, exactly at action boundaries `t` for which `0<=t-arrival_slot<=6`. Metric freshness instead uses request age `arrival_slot-generated_slot` as defined below. Attempts and responses due after boundary `90` are ignored.

#### Information availability and event order

At action start `t`, the policy receives exactly the current slot, scenario, variant, and scenario deadline; robot positions and executed velocities through boundary `t`; the prior submitted target velocity; the current request and local observation if generated at `t`; all attempt outcomes and responses revealed by `t`; current interface-up flags, current interface and replica reservation counts, and the service regime at `t`; the listed static obstacle geometry; prior decisions; accumulated fallback and motion energy; and used and remaining uplink bits. Target truth, `d_o`, occlusion, future target positions, future loss samples, future attempt outcomes, future response arrivals, future queue states, and terminal state are not action-start event inputs. Public deterministic environment laws may be evaluated to form forecasts, but those forecasts are not revealed events; the policy must not query simulator future outputs, hidden state, or retrieve pre-filled actions or replay-only truth by scenario, variant, request ID, or absolute slot.

Every nonterminal slot follows this fixed order:

1. Generate the scheduled request and local observation from boundary-`t` state.
2. Reveal attempt outcomes scheduled for `t`, then reveal due responses in ascending attempt-ID order and accept the first eligible response per request.
3. Retire interface and replica reservations and complete fallbacks whose right endpoint is `t`.
4. Expose the action-start information listed above.
5. Read the request, retry, fallback, and velocity decisions applicable at `t`, and validate their public fields and relationships.
6. Order submitted attempts, perform capacity and budget admission from the common pre-admission queue snapshot, sample loss, schedule each attempt outcome for `t+1`, and schedule every non-lost response.
7. Apply the lagged motion and fixed safety executor, advance the target to boundary `t+1`, and record tracking, risk, collision, queue, energy, and communication events.

Boundary `90` performs only the due-event reveal, acceptance, retirement, and fallback-completion parts of steps 2 and 3; it has no decision or motion. Request, retry, fallback, and target-velocity fields are candidate decisions at their stated action boundaries. Observations and revealed communication events are action-start information. Queue, robot, motion, mission, safety, and metric values are execution or replay evidence derived after the corresponding fixed steps. This event order is authoritative; equivalent internal implementations are accepted when they submit the same public decisions and therefore have the same fixed-model consequences.

#### Legal actions and execution boundaries

The submitted world-frame target velocity is `u_t`. It must contain finite numbers, satisfy `||u_t||<=0.8`, and satisfy `||u_t-u_(t-1)||<=0.05`, with `u_-1=(0,0)`. Actual velocity is `v_t=clip_vec(0.55*v_(t-1)+0.45*u_t,0.8)`, with `v_-1=(0,0)`, and proposed position is `p*=p_t+0.25*v_t`. If fixed geometric clearance of `p*` is below `0.35 m`, the executor records one collision event, keeps `p_(t+1)=p_t`, and sets executed velocity to zero. Otherwise it accepts `p_(t+1)=p*`. The slot records one risk event when the proposed clearance is below `0.60 m`, including a proposal rejected by the safety executor. Multiple simultaneous conflicts still count once. Motion energy is `0.5*||executed_velocity||^2` per slot.

Path rejection, loss, delay, response lateness, fallback completion, risk, and collision are fixed execution outcomes, not additional hidden action-legality conditions. The legality gate checks request and retry references and timing, path-count and path-distinctness rules, and target-velocity speed and slew. A policy may base motion on any exposed local observation, usable cloud response, completed fallback estimate, retained visible history, or braking behavior; it does not submit a control-source label.

#### Task metrics

Each metric is computed independently per unit from fixed replay and then averaged over the fifteen units. Let `initial_error` be robot-target distance at boundary `0`. `primary_metrics_1` is deadline-qualified tracking progress: `0.6*clip(1-rmse/initial_error,0,1)+0.4*clip(1-final_error/initial_error,0,1)`, where `rmse` uses tracking error at boundaries `0..scenario_deadline` and `final_error` is the error at that deadline boundary. `primary_metrics_2` is first-response reliability and freshness: `0.7*timely_first_response_ratio+0.3*freshness`. The ratio is the number of requests with a first accepted response divided by `30`; freshness is the mean of `clip(1-(arrival_slot-generated_slot)/6,0,1)` over those accepted first responses, or zero when there is none. This request-age formula is distinct from the control-usage age `t-arrival_slot`. `primary_metrics_3` is safe motion and control efficiency: `0.7*clip(1-risk_steps/90,0,1)+0.3*clip(1-motion_energy/(90*0.5*0.8^2),0,1)`.

`auxiliary_metrics_1` is useful uplink efficiency: `0.6*useful_request_bits/max(sent_request_bits,1)+0.4*clip(1-sent_request_bits/16800,0,1)`. Every admitted attempt, including a lost one, contributes `400` sent bits; each request's first accepted response contributes `400` useful bits once. `auxiliary_metrics_2` is queue and fallback efficiency: `0.5*clip(1-mean(queue_wait)/90,0,1)+0.5*clip(1-fallback_energy/100,0,1)`, where the mean includes admitted attempts only and is zero if none is admitted. Every metric lies in `[0,1]`. Normalized task performance is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result record contract

`results.json` is strict finite JSON with exactly root key `scenarios`. The array contains exactly one unit record for every canonical `(scenario,variant)` pair, in any order. Every unit has exactly `scenario`, `variant`, `requests`, `retries`, and `controls`. `scenario` is a string and `variant` is a JSON integer in `0..2`; Boolean values are not integers.

`requests` contains exactly one record for every `g_v`, in any order. Each has exactly string `request_id`, integer `generated_slot`, array `selected_paths`, and string `fallback`. Each path has exactly string `interface_id` in `I0..I2` and string `replica_id` in `C0..C2`; legality limits the array to zero, one, or two distinct path pairs. `fallback` is exactly `none` or `start`. `retries` is an array in any order; each record has exactly string `request_id`, string `interface_id` in `I0..I2`, string `replica_id` in `C0..C2`, and integer `send_slot`. Attempt `1` is implied. `controls` contains exactly one record for every integer `slot=0..89`, in any order. Each has exactly `slot` and `target_velocity`, where the latter is a length-two array of finite JSON numbers in world-frame `m/s`; Boolean values are not numbers.

Only these request, retry, fallback, and target-velocity decisions are submitted. The grader derives local observations, attempt outcomes, responses, queue snapshots, fallback completions, robot trajectory, motion and mission events, and every metric from the decisions and fixed model. None of those derived facts, hidden target truth, aggregate scores, `path_rank`, `n_path`, completion times, or arrival fields is a candidate result field. Absolute numeric tolerance is `1e-6`.

Validation responsibilities are distinct. `result_json` checks strict parsing, the exact root, unit container, and unit field structure. `result_evidence` checks the declared primitive types, enums, path objects, finite vectors, and exact request, retry, and control fields. `canonical_coverage` checks the fifteen unit keys, thirty generated-slot keys per unit, and ninety control-slot keys per unit for missing, duplicate, or unknown records. `decision_legality` checks the cross-record request IDs, path count and distinctness, retry timing and visible failed-attempt relation, and speed and slew. `replay_validity` checks that the fixed observation, network, response, fallback, motion, and mission process can be reconstructed consistently. Any representation satisfying these public records and event semantics is accepted.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain three readable ordinary files, not symbolic links: `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the causal information boundary, the unified routing, retry, fallback, and control method, all canonical units, controlled evidence, and supported limitations. Code and report must expose that policy decisions use the declared action-start information and distinguish revealed observations and responses from forecasts made with public model laws; forecasts must not be claimed as target truth or future execution evidence. Additional project files and report sections are allowed. Official execution must be self-contained and must not depend on an external service or unpublished reference code.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required non-empty headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenario and unit structure", weight: 3}
- {id: result_evidence, desc: "unit identifiers and request, retry, and control records have the exact typed minimal fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen units, request opportunities, and action slots occur exactly once", weight: 4}
- {id: decision_legality, desc: "paths, retries, target speed, and target slew obey action-boundary rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives observations, queues, responses, motion, and mission events", weight: 6}
- {id: primary_metrics_1, desc: "deadline-qualified target-tracking progress", weight: 8}
- {id: primary_metrics_2, desc: "timely first-response reliability and request freshness", weight: 4}
- {id: primary_metrics_3, desc: "clearance-risk and motion-energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful admitted uplink and redundancy efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "queue-wait and local-fallback resource efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the replicated cloud-request, information-timing, queue, response, tracking, and safety model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent cloud-feedback tracking formulation, or the model is materially wrong; Implementation/evidence: no executable request-to-motion process corresponds to the claimed model."}
    - {score: 0.4, desc: "Report: describes only routing or tracking and omits a decisive request identity, queue, response, information-boundary, or safety relation; Implementation/evidence: the executable state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: accurately defines the units, states, path and fallback decisions, revealed information, reservations, deadlines, objectives, and motion constraints; Implementation/evidence: one reusable process implements the core model across all units with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely closes event order, action-start information, first-response semantics, temporal and resource couplings, execution outcomes, and boundary conditions; Implementation/evidence: all variants faithfully realize those relations with reconstructible decisions and no material divergence."}
- id: method_design
  desc: "quality of the unified state-driven path, redundancy, retry, fallback, and motion-control method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method, or prohibited training is used; Implementation/evidence: no runnable unified controller or supporting decisions are present."}
    - {score: 0.4, desc: "Report: relies on pre-filled unit schedules, unconditional duplication, or a partial rule that ignores a decisive queue, loss, response-age, fallback, or safety consequence; Implementation/evidence: decisions are table-driven by unit or materially differ from the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning method that jointly handles path choice, bounded redundancy, retry or fallback, and motion from current state and history; Implementation/evidence: the same causal procedure covers all variants and its records support the principal choices."}
    - {score: 1.0, desc: "Report: justifies state-dependent reliability and queue estimates, redundancy marginal value, freshness, fallback cost, local safety, and uplink-motion trade-offs; Implementation/evidence: evolving action-start state drives all decisive choices without unit-specific action tables, replay-only truth, or forbidden future outcomes."}
- id: experimental_coverage
  desc: "coverage of canonical variants and controlled communication, queue, fallback, tracking, and execution conditions"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose; Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: only one network or motion condition is summarized and the central communication-to-tracking relation is not tested; Implementation/evidence: condition evidence cannot support the main claims."}
    - {score: 0.7, desc: "Report: covers every canonical unit and at least one explicitly identified two-sided comparison of paths, redundancy, queueing, outage, deadline, or fallback behavior; Implementation/evidence: both conditions execute on comparable inputs with attributable condition-level records."}
    - {score: 1.0, desc: "Report: complementary non-redundant controls separately test a path or redundancy mechanism and a queue, outage, deadline, fallback, or safety failure boundary; Implementation/evidence: each condition is actually executed with reconstructible records and metrics beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based interpretation of response timing, redundancy, resource, tracking, and safety trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the records; Implementation/evidence: the conclusions have no supporting execution evidence."}
    - {score: 0.4, desc: "Report: mainly repeats scores or makes unsupported causal claims and omits material limitations; Implementation/evidence: the recorded conditions do not support those claims."}
    - {score: 0.7, desc: "Report: uses actual results to explain response age, correlated failures, queueing, fallback, tracking, and safety trends while distinguishing observations, inferences, and limitations; Implementation/evidence: executed comparisons support those explanations."}
    - {score: 1.0, desc: "Report: uses controlled evidence and a failure boundary to quantify interactions among redundancy benefit, correlation, queue competition, freshness, resource cost, and robot outcome and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
