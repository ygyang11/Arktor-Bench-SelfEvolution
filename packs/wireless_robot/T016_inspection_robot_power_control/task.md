---
id: T016_inspection_robot_power_control
name: Communication-Reliable Power Control for Multi-Robot Substation Inspection
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

Three mobile robots inspect high-voltage equipment along fixed routes in a substation. At each checkpoint, a robot must deliver a time-sensitive inspection report to one wireless access point before it can continue normally. Motion changes the propagation distance and can place a robot behind equipment, while electromagnetic interference produces occasional severe RSSI drops. Two simultaneous reports also reduce each other's received power. A weak attempt may be lost or delayed beyond its checkpoint deadline; repeated high-power attempts consume battery and constrain later reports. The coordinator must therefore allocate shared transmission opportunities and discrete powers while route progress, report confirmations, link observations, retry history, and remaining energy evolve together.

### Research Problem

Build a deterministic, non-learning simulator and one unified online coordinator for all canonical scenes. At every decision boundary, jointly choose whether each robot sends its current report and, if so, its transmit power. The method may filter observed baseline RSSI, prioritize deadlines and checkpoint values, coordinate simultaneous transmissions, and use the public fixed model to plan energy or retries, but it must not read evaluator-only runtime state or treat an event as observed before its boundary. Study how power control and report scheduling affect timely verified inspection, route progress, energy efficiency, link margin, and power stability. The same state-driven procedure must handle open, shielded, interference-heavy, bottleneck, and long-return routes without scene-specific transmission schedules.

### Experimental Setting and Constraints

#### Canonical scenes, routes, and inspection values

Coordinates are `[x,y]` in metres in the closed region `[0,20] x [0,20]`; the access point is fixed at `a=[10,10]`. Scene index `s` is the zero-based index in the fixed order `open_yard`, `shielded_turns`, `interference_corridor`, `shared_bottleneck`, and `long_return`. Every scene has robots `0,1,2`, six ordered checkpoint IDs `0,...,5` per robot, decision slots `t=0,...,47`, and a terminal boundary at `t=48`. A route row lists checkpoint coordinates in ID order:

| scene | robot | checkpoints `0,...,5` |
|---|---:|---|
| `open_yard` | 0 | `[[8,8],[6,8],[4,8],[4,12],[6,14],[9,13]]` |
| `open_yard` | 1 | `[[12,8],[14,8],[16,8],[16,12],[14,14],[11,13]]` |
| `open_yard` | 2 | `[[10,12],[10,15],[7,16],[5,14],[7,11],[10,9]]` |
| `shielded_turns` | 0 | `[[7,7],[5,7],[3,9],[3,13],[6,16],[9,15]]` |
| `shielded_turns` | 1 | `[[13,7],[15,7],[17,9],[17,13],[14,16],[11,15]]` |
| `shielded_turns` | 2 | `[[10,13],[10,16],[7,18],[4,16],[5,12],[8,10]]` |
| `interference_corridor` | 0 | `[[4,8],[6,8],[8,8],[10,8],[12,8],[14,8]]` |
| `interference_corridor` | 1 | `[[4,10],[6,10],[8,10],[10,10],[12,10],[14,10]]` |
| `interference_corridor` | 2 | `[[4,12],[6,12],[8,12],[10,12],[12,12],[14,12]]` |
| `shared_bottleneck` | 0 | `[[3,7],[6,8],[9,9],[12,10],[15,11],[17,13]]` |
| `shared_bottleneck` | 1 | `[[3,10],[6,10],[9,10],[12,10],[15,10],[18,10]]` |
| `shared_bottleneck` | 2 | `[[3,13],[6,12],[9,11],[12,10],[15,9],[17,7]]` |
| `long_return` | 0 | `[[10,8],[14,6],[18,4],[18,10],[16,16],[10,18]]` |
| `long_return` | 1 | `[[8,10],[5,6],[2,3],[2,10],[4,17],[10,19]]` |
| `long_return` | 2 | `[[12,10],[15,7],[18,7],[19,13],[15,18],[9,17]]` |

Checkpoint values for IDs `0,...,5` are `[4,7,5,9,6,10]` for robot 0, `[6,5,8,6,10,7]` for robot 1, and `[5,8,6,10,7,9]` for robot 2. If checkpoint `k` is released at boundary `r`, its inclusive confirmation deadline is `r+W[s,k]`, where every window is positive:

| scene | `W[s,0],...,W[s,5]` |
|---|---|
| `open_yard` | `[2,2,2,2,2,2]` |
| `shielded_turns` | `[2,3,2,3,2,4]` |
| `interference_corridor` | `[2,2,3,2,3,3]` |
| `shared_bottleneck` | `[2,2,2,3,2,3]` |
| `long_return` | `[3,3,4,3,4,4]` |

At boundary 0, every robot is at checkpoint 0; that report is released at 0. A robot waiting for a report confirmation does not move. When its current report is confirmed or misses its deadline at a boundary, the robot immediately enters `departing` and remains still for the complete slot following that boundary. At the next boundary it enters `travel` and, at the end of every travel slot, moves in a straight line toward its next checkpoint by `min(1.5,d)` metres, where `d` is the remaining distance. Reaching a checkpoint at the end of slot `q` releases that report at boundary `q+1`. Resolving checkpoint 5 makes the route `complete` immediately; there is no later movement. Missing a report does not prevent travel to the next checkpoint, but that checkpoint remains unverified permanently.

#### Baseline RSSI and realized report outcomes

At decision boundary `t`, robot `i` receives a zero-power baseline observation

`b[s,i,t] = -44 - 12*log10(1+d[s,i,t]) - o[s,i,t] - h[s,i,t]` dBm,

where `d` is Euclidean distance from its current position to the access point. Obstruction loss `o` is the sum of the losses of every closed rectangle containing the current position; a boundary point is inside. A dash means no rectangle.

| scene | closed rectangles `[xmin,xmax] x [ymin,ymax] : loss dB` |
|---|---|
| `open_yard` | -- |
| `shielded_turns` | `[2,7.5]x[8.5,14.5]:4.5`; `[12.5,18]x[8.5,14.5]:4.5`; `[6,14]x[15,18]:3` |
| `interference_corridor` | `[7,13]x[7,13]:2` |
| `shared_bottleneck` | `[7.5,9.5]x[5.5,14.5]:3.5`; `[10.5,12.5]x[5.5,14.5]:3.5` |
| `long_return` | `[0,6]x[0,8]:5`; `[14,20]x[0,8]:5`; `[0,6]x[14,20]:6`; `[14,20]x[14,20]:6` |

The nonnegative realized disturbance is deterministic for replay but only its current effect through `b[s,i,t]` is a policy observation. Let `z=(11t+7i+5s) mod 31` and define `tau(z)` as 9 for `z=0`, 5 for `z` in `{1,2}`, 2 for `z` in `{3,4,5,6}`, and 0 otherwise. Then `h=beta[s]+tau(z)+burst`:

| scene | `beta[s]` | additional `burst` |
|---|---:|---|
| `open_yard` | 0 | 0 |
| `shielded_turns` | 1 | 3 when `(t+2i) mod 17` is 8 or 9, otherwise 0 |
| `interference_corridor` | 3 | 4 when `(floor(t/4)+i) mod 4=1`, otherwise 0 |
| `shared_bottleneck` | 1.5 | 3 when `15<=t<=29` and `(t+i) mod 3` is nonzero, otherwise 0 |
| `long_return` | 1.5 | 4 when `t>=24` and `(t+2i) mod 7` is 0 or 1, otherwise 0 |

Thus exact future disturbances can be calculated from public scene metadata and `t`; using that calculation for model-based planning is allowed. It is not a direct observation of a future event.

The coordinator chooses one joint action after observing all three current baselines. A sending robot uses `p` in `{2,6,10,14,18,22}` dBm. For the set `S` of current senders, its realized RSSI and margin are

`rssi_i = b[s,i,t] + p_i - 0.35*sum_{j in S, j!=i} p_j`,

`margin_i = rssi_i - (-58)` dB.

An attempt with negative margin is lost. An attempt with `0<=margin<6` arrives two boundaries later; one with `margin>=6` arrives one boundary later. Every attempt consumes energy regardless of loss, delay, a duplicate pending attempt, or a later deadline miss. Arrivals at the same boundary are handled by increasing generation slot and then robot ID. The first on-time arrival for the robot's still-current checkpoint confirms it; later duplicates and arrivals for an earlier report are ignored. Arrival at the inclusive deadline can confirm because delivery precedes deadline processing. An arrival after its deadline cannot confirm. Boundary 48 admits and processes arrivals and deadline events but has no decision or movement; arrivals after 48 have no evaluated effect.

#### Joint decisions, resource limits, and information boundary

Each robot starts with 20 energy units. A physically processed attempt at power `p` costs `1+0.015p^2` units. The action is illegal when that cost exceeds the robot's remaining energy. At most two robots may send in one slot. Each robot may send in at most three immediately consecutive slots; an idle slot resets this streak. If it sends in both `t-1` and `t`, the two powers must differ by at most 6 dBm. A gap removes this adjacent-slot slew relation. A robot may send only while its current checkpoint is in `awaiting_report`, after release and strictly before the deadline boundary; it always sends that current report. There is no report payload or checkpoint choice.

At the joint decision boundary, the coordinator may read the public scene name and index and `t`; for every robot, its ID, full fixed route, checkpoint values and deadline windows, current position and route state, current checkpoint ID, release slot and deadline when awaiting, remaining energy, current baseline RSSI, immediately preceding send streak and power, and the released/verified/missed history and confirmation slots. It may retain all past inputs and decisions. The simulator, but not the policy, stores raw current disturbance components, realized action outcomes, the pending-arrival schedule, and route states that have not yet occurred. The policy cannot read those evaluator-only runtime values or an arrival before its boundary. It may simulate or forecast consequences from every public fixed equation, including the deterministic disturbance law. Scene-, checkpoint-, robot-, or absolute-slot power and send tables remain prohibited. Robot ID may support one uniform priority rule, but not a robot-specific schedule.

Each decision slot has this authoritative order: admit report arrivals and apply the first eligible confirmation; expire any still-unconfirmed report at its deadline; apply completion or entry into the one-slot departure state, advance an earlier departure to travel, and release a checkpoint reached in the preceding slot; compute and expose current baseline observations and the joint policy view; choose the three actions together; check and debit resources; compute joint interference and schedule successful arrivals; move traveling robots; then record only the submitted decisions. Positions, route states, disturbances, baselines, attempts, margins, delays, arrivals, confirmations, deadlines, energy, and all metrics are fixed-model consequences reconstructed by the grader. Unified-policy provenance and online-information use are evaluated from code and report, not inferred from a legal decision sequence alone.

#### Result contract and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. The array contains exactly one record for each canonical scene; its order is irrelevant. A scene object has exactly `scenario` and `slots`. `slots` contains exactly one record for each integer `t` in `0,...,47`; its order is irrelevant. A slot object has exactly `t` and `actions`. `actions` contains exactly one record for each robot ID `0,1,2`; its order is irrelevant. An action object has exactly integer `robot`, Boolean `send`, and integer `power_dbm`. Booleans are not integers. When `send` is false, `power_dbm` is exactly 0; when it is true, `power_dbm` is one of the six stated levels. The 240 slot records and 720 robot-slot actions are the entire required replay evidence. No position, route-state, RSSI, attempt, energy, confirmation, inspection, aggregate, metric, or particular controller-state representation is required.

Invalid submissions receive no task-metric score, but replay uses fixed continuations for attributable diagnostics. Unknown records are ignored by replay after structural and canonical checks. Replay retains the first scene, slot, and robot record for each expected explicit ID. A missing or unrelatable robot-slot action becomes `send=false,power_dbm=0`, with its ID defect scored only by canonical coverage. An identified action with a malformed `send` or `power_dbm` field also becomes idle and fails only its evidence atom. A well-typed but invalid send-power pair fails only power selection and becomes idle. A well-formed request made without a sendable report or sufficient energy fails its owning legality rule, is ignored physically, and does not count toward concurrent capacity. Every other well-formed send is processed even when the resulting physical sender set exceeds capacity or its streak or slew is illegal.

Because an incomplete action can change same-slot interference and every robot's later state, the first missing, unrelatable, or locally malformed robot-slot action establishes a scene-wide causal cutoff. Concurrent-capacity legality is not inferred at that boundary or later. Report-availability, energy, streak, and slew legality for complete actions at the cutoff boundary remains attributable from the already fixed pre-decision state, but those state-dependent rules are not inferred at later boundaries. Power-selection legality remains local at every boundary. Duplicate records do not create uncertainty: the retained first explicit record controls replay and only canonical coverage scores the duplicate. An otherwise complete but illegal decision never establishes a cutoff. These continuations affect only partial diagnostics because every metric is gated by all validity checks.

`primary_metrics_1` is computed per scene as the sum of fixed values of all checkpoint reports confirmed no later than their deadlines, divided by the sum of all 18 checkpoint values; it is then averaged over the five scenes.

For every robot-checkpoint report released at `r`, confirmed at `c`, and verified by its deadline, define `e=clip(1-(c-r)/W[s,k],0,1)`; unreleased, missed, or unconfirmed reports have `e=0`. `primary_metrics_2` is the mean of `e` over the 18 reports in each scene and then over scenes.

For `auxiliary_metrics_1`, let `V_verified` be a robot's fixed value sum for reports verified in one scene, `V_total` its six-report value sum, and `E_used` its attempted-transmission energy. Its score is `V_verified/(V_verified+V_total*E_used/20)`; when the denominator is zero, it is zero. The metric is the mean over all 15 scene-robot pairs.

For each physically processed attempt, let `p_prev` be that robot's most recent earlier attempt power in the same scene, or 0 for its first attempt. Its attempt quality is `0.6*clip((margin+8)/16,0,1)+0.4*clip(1-|p-p_prev|/22,0,1)`. First average attempts within each scene, using zero for a scene with no attempt, then average the five scene values to obtain `auxiliary_metrics_2`.

Here `clip(x,0,1)=min(1,max(0,x))`. Higher is better for every metric. Normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. All values come from independent replay and are scored only when every validity check is full.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite root-level `results.json` from the project code and fixed conditions above. The evaluator deletes an old result before one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations. Controlled experiments used as Judge evidence may produce optional text or JSON artifacts outside `results.json`; for every claimed comparison, the report must identify the conditions, shared inputs, changed factor, artifact path, and condition-level outcomes.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "slot, send, and discrete-power decisions satisfy their local types", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "power, capacity, report state, energy, retry, and slew decisions are legal", weight: 4}
- {id: replay_validity, desc: "baseline, interference, arrival, route, energy, and decision processes replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed value-weighted timely report verification", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed checkpoint deadline quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed verified-value energy efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed margin and power-stability quality", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of route progress, RSSI, shared transmission, confirmation, energy, and deadlines"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent inspection-link model. Implementation/evidence: no corresponding executable states or decisions."}
    - {score: 0.4, desc: "Report: omits a decisive route, baseline-RSSI, interference, confirmation, energy, or deadline relation. Implementation/evidence: code realizes only a partial or materially divergent model."}
    - {score: 0.7, desc: "Report: accurately defines routes, report state, observed RSSI, joint interference, delays, capacity, energy, and deadlines. Implementation/evidence: code and decisions realize the core model across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains the event order, current-versus-future information, report lifecycle, route coupling, resource state, and reliability-efficiency trade-offs. Implementation/evidence: code faithfully realizes every central relation with traceable internal evidence and no material divergence."}
- id: method_design
  desc: "unified online power, concurrency, retry, deadline, value, and energy coordination"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable decision process."}
    - {score: 0.4, desc: "Report: fixed powers, periodic retries, or unit schedules dominate while deadlines, joint interference, or energy are ignored. Implementation/evidence: pre-filled scene, robot, checkpoint, or slot decisions dominate or code diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method uses current link, report, retry, and resource state to coordinate sends and powers. Implementation/evidence: one procedure operates across all scenes with limited risk treatment or planning depth."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly handles observed link risk, concurrent interference, pending consequences, checkpoint value and slack, retries, slew, and remaining energy. Implementation/evidence: evolving permitted state changes decisive choices without unit-specific schedules or material divergence."}
- id: experimental_coverage
  desc: "controlled evidence for power reliability, interference scheduling, deadlines, and energy"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or isolated power and success summaries do not test communication-inspection causality. Implementation/evidence: outputs cannot support the principal claims."}
    - {score: 0.7, desc: "Report: all scenes and one controlled power, interference, capacity, deadline, or energy comparison directly test a central trade-off. Implementation/evidence: both conditions execute with comparable condition-level attempt, confirmation, and resource evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check for reliable inspection. Implementation/evidence: reproducible condition-level route, attempt, margin, confirmation, energy, and metric outcomes support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of link margin, retries, energy, deadlines, and inspection progress"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats power, success, or energy values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important margin, retry, confirmation, energy, and deadline trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains the central mechanism, quantifies principal sensitivities, tests a contrasting or failure boundary, and states scope and unknowns. Implementation/evidence: reproducible condition-level comparisons support the central conclusions."}
