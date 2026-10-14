---
id: T015_delay_aware_collision_avoidance
name: Delay-Aware Distributed Safety Control for Multi-Robot Navigation
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

Four mobile robots carry time-sensitive items between fixed stations in a shared indoor transfer zone. Each robot knows its own state and goal, but it can estimate another robot only from state broadcasts that have already arrived. Messages are range-limited, delayed, and sometimes lost. A nominal goal controller is efficient in isolation but creates head-on, crossing, merging, and dense-swap conflicts. A collision can damage a robot or its payload, whereas indefinite stopping defeats the transfer task. Broadcasting more often can refresh distributed safety estimates, yet all robots share a finite communication budget. Each robot must therefore choose a bounded motion and whether to broadcast while delayed neighbor information, collision risk, goal progress, and communication use evolve together.

### Research Problem

Build a deterministic, non-learning simulator and one unified distributed policy for all canonical scenes. At every slot, each active robot must choose a two-dimensional action and a broadcast decision using only its own state and history, delivered neighbor messages, and the exposed shared resource counters. Design a prediction or safety-filtering method that remains close to the fixed nominal goal controller when possible, reacts conservatively to stale information when necessary, and allocates broadcasts where freshness can improve collision avoidance. Study the resulting safety, goal progress, control intervention, message age, and communication trade-offs without using current hidden neighbor states or future link outcomes.

### Experimental Setting and Constraints

#### Canonical scenes and dynamics

The closed workspace is `[0,6] x [0,6]`. It contains no static obstacles. Scene order, starts, and goals for robots `0,1,2,3` are fixed below. Coordinates are `[x,y]` in workspace units.

| scene | starts | goals |
|---|---|---|
| `opposing_goals` | `[[0.8,2.0],[5.2,2.0],[0.8,4.0],[5.2,4.0]]` | `[[5.2,2.0],[0.8,2.0],[5.2,4.0],[0.8,4.0]]` |
| `four_way_crossing` | `[[3.0,0.6],[5.4,3.0],[3.0,5.4],[0.6,3.0]]` | `[[3.0,5.4],[0.6,3.0],[3.0,0.6],[5.4,3.0]]` |
| `narrow_merge` | `[[0.6,2.0],[0.6,4.0],[5.4,2.5],[5.4,3.5]]` | `[[5.4,2.0],[5.4,4.0],[0.6,2.5],[0.6,3.5]]` |
| `offset_swap` | `[[0.8,1.0],[5.2,1.8],[0.8,4.2],[5.2,5.0]]` | `[[5.2,1.8],[0.8,1.0],[5.2,5.0],[0.8,4.2]]` |
| `dense_ring` | `[[3.0,1.9],[4.1,3.0],[3.0,4.1],[1.9,3.0]]` | `[[3.0,4.1],[1.9,3.0],[3.0,1.9],[4.1,3.0]]` |

Every scene has slots `t=0,...,59`. Position `p_i(t)` is the pre-move position at slot `t`. An active robot submits `u_i(t)=[ux,uy]`. Let `u_bound_i(t)` be that vector unchanged when its norm is at most `0.4`, and otherwise radially scaled to norm `0.4`. The environment forms `p_i(t)+u_bound_i(t)` and clips each coordinate independently to `[0,6]`; the difference between the clipped endpoint and `p_i(t)` is the executed action. The submitted-action legality tolerance is defined below. All robot endpoints are computed from the same pre-move state, so movement is simultaneous.

Let `g_i` be a robot's goal. Its nominal action at a boundary is zero after collision or goal completion. Otherwise, for `d=g_i-p_i(t)`, it is zero when `||d||_2 <= 1e-6` and is

`u_nom_i(t) = min(0.4, ||d||_2) * d / ||d||_2`.

A robot completes its goal when a collision-free slot endpoint first has goal distance at most `0.25 + 1e-6`. It remains completed and every later submitted action is executed as zero; a submitted norm above `1e-6` is illegal, while a tolerance-sized action is accepted but still executed as zero. A completed robot may still broadcast. Robot radius is `0.2`. After simultaneous movement, every unordered pair is tested at its endpoints. A distance strictly below `0.4 - 1e-6` is a collision; every involved robot becomes collided, every later submitted action is executed as zero, and every later broadcast must be `false`. As for completion, a collided robot's submitted action is legal only when its norm is at most `1e-6`. A non-colliding distance strictly below `0.8 - 1e-6` is a safety-margin alert. Collision status is permanent. An action norm at most `0.4 + 1e-6` satisfies the action bound, but every submitted action whose norm exceeds the physical limit `0.4` is radially clipped to norm `0.4` before endpoint clipping. Coordinate clipping to `[0,6]` is exact. These formulas are the complete tolerance convention; no additional rounding is applied.

#### Broadcasts, message delay, and neighbor history

The communication radius is `1.8`. A robot's Boolean broadcast decision is made at the pre-move boundary. One broadcast counts once against communication limits even if no receiver is in range or all resulting messages are lost. For every other robot whose pre-move separation is at most `1.8 + 1e-6`, the broadcast generates one directed message. With sender `i`, receiver `j`, scene index `s` in canonical order starting from zero, and generation slot `t`, that directed message is lost exactly when

`(5*t + 2*i + 3*j + s) mod 13` is in `{0,1}`.

Otherwise its delay is `1 + ((t+i+j+s) mod 3)` and it arrives at the start of slot `t+delay`. Arrivals at slot 60 or later have no evaluated effect. The message contains the sender ID, generation slot, sender pre-move position, and sender's previous executed action; the previous action is `[0,0]` before slot 0. Link eligibility and payload always use the generation boundary, not the later arrival boundary.

At each receiver, messages are kept separately by sender. After current arrivals are delivered, the available message for sender `j` is the delivered message with greatest generation slot, or `null` if none has arrived. Older messages that arrive later do not replace a newer one. Its age at boundary `t` is `t-generation_t`. A policy may retain all messages and its own past inputs, but an unavailable or not-yet-arrived message conveys no information.

Each scene permits at most `48` team broadcasts. Each individual robot may broadcast at most three times in every inclusive four-slot window `max(0,t-3),...,t`. Robots choose in numeric order only for resource accounting. At robot `i`'s boundary, let `B` be the processed broadcasts in all prior slots plus those already chosen by lower-numbered robots in slot `t`. The exposed remaining team budget is `max(0,48-B)`; the exposed preceding-three-slot count covers slots `max(0,t-3),...,t-1` for robot `i`; and the exposed earlier-current count is the number of those lower-numbered current-slot broadcasts. The latter broadcasts are already included in `B`. The policy may not read the lower-numbered robots' current motion decisions. A well-formed broadcast that exceeds either limit is still processed for diagnostic replay, but fails legality and closes every task-metric gate.

#### Information boundaries and slot order

The policy view for robot `i` contains only `t`; robot ID; own start, goal, position, previous executed action, the nominal action deterministically computed above, collision and completion status; the latest delivered message from each other robot with the payload and age defined above; remaining team broadcast budget; its own preceding-three-slot broadcast count; and earlier current-slot broadcast count. A policy may retain its own past inputs and decisions. The simulator additionally uses the scene identifier, other robots' true current positions, the undelivered message schedule, losses, and link outcomes to advance the environment, but those values are not policy inputs. In particular, a policy may not read another robot's current position or action, a message before arrival, current link eligibility or loss, or any future state or event. The same callable policy must use the exposed state and retained history in every scene; the scene name is not passed into it, and scene-, robot-, or absolute-slot action tables are prohibited.

Each slot has this authoritative order: deliver scheduled messages and retain the greatest-generation message per sender; compute nominal actions; in robot order expose robot `i`'s view and immediately choose its submitted action and broadcast before exposing robot `i+1`'s resource counters; determine directed link eligibility, loss, delay, and payload from the pre-move state and schedule successful messages; clip endpoints and execute all motions simultaneously; evaluate endpoint pair distances, collisions, alerts, and new goal completions; then record only the submitted decisions. Freshness is sampled after current message delivery and before decisions. Control intervention is sampled from the nominal and executed actions of that slot.

Submitted fields are decisions at the stated boundaries. Positions, nominal and executed actions, messages, neighbor histories, ages, boundary clipping, pair distances, alerts, collisions, completions, and all metrics are fixed-model consequences reconstructed by the grader. Online-information use and unified-policy provenance are evaluated from the implementation and report, not inferred from a particular legal decision sequence.

#### Result contract and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. The array contains the five scenes in canonical order. Each scene object has exactly `scenario` and `frames`. `frames` contains exactly 60 records in slot order. Each frame has exactly `t` and `robots`; `t` is an integer and `robots` contains exactly four records in robot order.

Each robot record has exactly `robot`, `action`, and `broadcast`. `robot` is an integer. `action` is a length-two array of JSON numbers that each convert to a finite IEEE-754 binary64 value; Booleans are rejected as numbers. `broadcast` is a JSON Boolean. No derived trace, prediction, message record, state trajectory, pair-distance table, event table, aggregate, metric, or particular internal state representation is required.

Invalid submissions receive no task-metric score, but replay uses fixed continuation rules for attributable diagnostics. Unknown records are ignored for replay. Across submitted scene rows, replay retains the first frame for each expected `(scenario,t)` pair and, within that frame, the first record for each expected robot ID. A missing or unrelatable robot-slot decision becomes `[0,0]` and `false`; these ID defects are scored only by canonical coverage. Stateless action and resource legality remain checkable throughout an incomplete scene. Collision- or completion-dependent legality after its first incomplete slot is checked only for a permanent state already established before that slot. A malformed action field becomes `[0,0]` without a second legality failure; a well-typed action whose norm exceeds `0.4` is radially clipped to norm `0.4`, and it fails action legality only when its norm also exceeds `0.4 + 1e-6`. Every well-typed action submitted after collision or completion is executed as `[0,0]`; it fails stationary-action legality only when its submitted norm exceeds `1e-6`. A malformed broadcast becomes `false` without a second legality failure. A `true` broadcast after collision is ignored and fails inactive-broadcast legality; other well-typed broadcasts are processed even when they violate a rolling or total budget. Workspace clipping is always applied after these rules. These continuations affect only partial diagnostics because every metric is gated by all validity checks.

`primary_metrics_1` is the fraction of the 20 scene-robot trajectories that never collide.

For `primary_metrics_2`, let `g[s,i]` be 1 when a robot completes its goal by slot 59 and 0 otherwise. For every scene-robot trajectory that collides, `q[s,i]=0`. Otherwise,

`q[s,i] = 0.6*g[s,i] + 0.4*mean_t clip(1-||u_exec[s,i,t]-u_nom[s,i,t]||_2/0.8,0,1)`.

The metric is the mean of `q[s,i]` over all 20 trajectories. The denominator `0.8` is the maximum Euclidean difference between two actions of norm at most `0.4`.

`auxiliary_metrics_1` is the scene mean of `clip(1-n_broadcast[s]/48,0,1)`, where a team broadcast is counted whether or not it produces a delivered message.

For `auxiliary_metrics_2`, at every post-delivery boundary and every ordered receiver-sender pair, the freshness score is zero when no message from that sender is available and otherwise `max(0,1-age/3)`. The metric averages this score over five scenes, 60 slots, and 12 ordered pairs.

Higher is better for every metric. Normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. Metrics are scored only when all validity checks pass.

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
- {id: result_evidence, desc: "frame, two-dimensional action, and broadcast decisions are typed", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "actions, stationary states, and rolling and total broadcasts are legal", weight: 4}
- {id: replay_validity, desc: "action, motion, message, history, collision, and goal processes replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed collision-free trajectory fraction", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed goal and minimally invasive safety quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed team-broadcast efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed delivered-neighbor-state freshness", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of delayed distributed state, simultaneous motion, collision safety, and goals"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent delayed-message safety model. Implementation/evidence: no corresponding executable states or decisions."}
    - {score: 0.4, desc: "Report: omits a decisive dynamics, message-age, collision, broadcast, or goal relation. Implementation/evidence: code implements only a partial or divergent model."}
    - {score: 0.7, desc: "Report: accurately defines the core local information, message process, bounded simultaneous motion, collision state, communication limits, and goals. Implementation/evidence: code and decisions realize them across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order, delayed-state uncertainty, safety margins, boundary motion, permanent outcomes, and communication-safety-efficiency trade-offs. Implementation/evidence: code faithfully realizes every central relation without material divergence."}
- id: method_design
  desc: "unified delayed-state prediction, safe motion, and broadcast allocation"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable procedure or decision evidence."}
    - {score: 0.4, desc: "Report: the method is partial or dominated by fixed decisions that ignore decisive stale-information, resource, or safety consequences. Implementation/evidence: pre-filled unit actions dominate or code materially diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method addresses neighbor estimation, safety intervention, goal progress, and broadcast use from permitted state and history. Implementation/evidence: one procedure runs across all scenes with limited prediction depth or robustness."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly treats the central temporal, informational, safety, and resource consequences of delayed distributed control. Implementation/evidence: evolving permitted state drives all decisive choices without scene-specific schedules or material divergence."}
- id: experimental_coverage
  desc: "controlled evidence for delay, freshness, collision avoidance, and intervention costs"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or isolated collision totals do not examine the delayed-information safety relation. Implementation/evidence: outputs do not support the principal claims."}
    - {score: 0.7, desc: "Report: all scenes and one controlled delay, broadcast, prediction, or safety comparison directly test a central trade-off. Implementation/evidence: both conditions execute with comparable condition-level evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check for delayed distributed safety. Implementation/evidence: reproducible condition-level message, action, collision, and goal outcomes support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of information age, safety, goal progress, and communication"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats collision, goal, or broadcast values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important age, prediction, intervention, safety, goal, and communication trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains the central mechanism, quantifies principal sensitivities, tests a contrasting or failure boundary, and states scope and unknowns. Implementation/evidence: reproducible comparisons support the central conclusions."}
