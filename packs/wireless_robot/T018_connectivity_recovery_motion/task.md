---
id: T018_connectivity_recovery_motion
name: Freshness-Weighted Connectivity Recovery in Obstacle-Constrained Robot Motion
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

Four ground robots must cross cluttered indoor regions and reach assigned destinations without losing safe separation. A geometric radio link exists only when two robots are close enough and an inflated obstacle does not block their line of sight. The link is useful for coordinated motion only while both endpoints hold sufficiently recent state estimates of one another. Each broadcast shares the sender's current state and its locally cached team state, but arrives only at the next decision boundary and consumes a limited team resource. A team may start split or separate while routing around an obstacle; it must then restore a fresh, distance-robust coordination graph while continuing toward its goals.

### Research Problem

Build a deterministic, non-learning simulator and a unified local-state-driven method for all canonical scenes. At each slot, choose one discrete motion direction and whether to broadcast for every robot. The method should use delivered cache versions and evolving geometry to balance goal progress, obstacle and robot clearance, coordination-margin maintenance, deadline-bounded recovery, and communication use. It may use deterministic planning, search, optimization, or heuristics, but it must not substitute scene-specific action schedules for a reusable procedure. Study when maintaining a compact formation is beneficial, when obstacle traversal requires temporary margin loss, and how broadcast timing affects recovery through state freshness.

### Experimental Setting and Constraints

#### Canonical scenes

The closed workspace is `[0,14] x [0,14]`. Coordinates are `[x,y]`; one coordinate unit is one metre. Robots are indexed `0,...,3`. A physical obstacle is a closed axis-aligned rectangle `[xmin,xmax] x [ymin,ymax]`. For robot motion and radio line of sight, every listed rectangle is expanded outward by `0.25` on all four sides. The five scenes, their starts and goals in robot-ID order, their physical obstacles, and their recovery limits are fixed below.

| scene | starts | goals | physical rectangles `[xmin,xmax,ymin,ymax]` | recovery limit |
|---|---|---|---|---:|
| `initial_split` | `[[1.4,5.0],[1.4,6.2],[4.4,5.0],[4.4,6.2]]` | `[[10.8,5.0],[10.8,6.2],[12.2,5.0],[12.2,6.2]]` | `[[6.2,7.4,0.0,4.0],[6.2,7.4,7.2,14.0]]` | 12 |
| `crossing_obstacles` | `[[1.2,3.2],[1.2,4.4],[2.4,3.2],[2.4,4.4]]` | `[[11.4,8.6],[11.4,9.8],[12.7,8.6],[12.7,9.8]]` | `[[4.0,5.0,0.0,6.2],[7.0,8.0,7.8,14.0],[10.0,11.0,0.0,6.2]]` | 8 |
| `narrow_passage` | `[[1.2,5.4],[1.2,6.6],[2.4,5.4],[2.4,6.6]]` | `[[11.4,5.4],[11.4,6.6],[12.6,5.4],[12.6,6.6]]` | `[[5.0,9.0,0.0,4.8],[5.0,9.0,7.2,14.0]]` | 8 |
| `temporary_separation` | `[[1.2,6.4],[1.2,7.6],[2.4,6.4],[2.4,7.6]]` | `[[11.4,6.4],[11.4,7.6],[12.6,6.4],[12.6,7.6]]` | `[[5.2,8.8,5.3,8.7]]` | 10 |
| `dense_cluster` | `[[2.0,5.0],[2.0,6.0],[3.0,5.0],[3.0,6.0]]` | `[[11.0,6.0],[11.0,5.0],[12.0,6.0],[12.0,5.0]]` | `[[5.3,6.4,4.0,5.2],[7.4,8.5,5.8,7.0]]` | 6 |

Every scene has decision slots `t=0,...,59`, each lasting `0.5` seconds. Boundary `b=t` is the pre-move boundary of slot `t`; boundary 60 processes the last broadcast arrivals and outcomes but has no decision. Goals, starts, rectangles, and the complete horizon are public to every robot. There are no random events and no early termination.

#### Motion, obstacles, and safety

Each robot chooses one of `{N,NE,E,SE,S,SW,W,NW,stay}`. Cardinal directions have displacement `0.35` along one axis; diagonal directions have displacement `0.35/sqrt(2)` along both named axes, and `stay` has zero displacement. Positive `y` is north and positive `x` is east. All four straight-line motions are executed simultaneously from `p_i(t)` to `p_i(t+1)`.

A requested segment is geometrically legal only when its endpoint lies in the closed workspace and the entire segment has empty intersection with every closed expanded rectangle; touching an expanded rectangle is an intersection. A geometrically illegal request is replaced by `stay` for diagnostic continuation and fails motion legality. For robots `i,j`, define their simultaneous swept separation as

`min_{a in [0,1]} ||(p_i(t)+a*(p_i(t+1)-p_i(t))) - (p_j(t)+a*(p_j(t+1)-p_j(t)))||_2`.

Every swept separation must be at least `0.8-1e-6`. A smaller value is a safety violation and fails legality; the already determined diagnostic motions are not changed. A robot first reaches its goal at a boundary whose centre-to-goal distance is at most `0.35+1e-6`. Reaching a goal is permanent for metric purposes, but it does not force the robot to stop.

#### Broadcasts, caches, and coordination graph

The physical radio radius is `2.4`. At a boundary, a physical undirected link exists exactly when the centre distance is at most `2.4+1e-6` and the closed segment between the centres has empty intersection with every closed expanded rectangle. Robot centres at legal positions cannot lie in such a rectangle.

At boundary 0, every robot has the mission-brief cache entry `(generation=0, position=start_k)` for every robot `k`. Robot `i` always refreshes its own cache entry to its exact boundary position with generation equal to the current boundary. A broadcast requested in slot `t` is generated after motion, after the sender refreshes its own entry to `(t+1,p_i(t+1))`. Its fixed payload is the sender's complete four-entry cache snapshot at that instant. It is scheduled for every physical neighbour at the post-move geometry and arrives at boundary `t+1`; a broadcast with no physical neighbour still consumes one broadcast. On arrival, a receiver replaces an entry only when the payload generation is strictly greater than its cached generation. All same-boundary messages are compared with the receiver's cache before that arrival batch, and the retained entry for each robot is the greatest generation; equal canonical generations represent the same state. Sender-ID order breaks any remaining equal-generation tie. An item received at a boundary may be used in that boundary's decision but could not have been used in the preceding slot.

Each robot may request at most three broadcasts in any four consecutive decision slots, including shorter prefix windows. A scene permits at most 120 team broadcasts. For diagnostic continuation, well-typed requests are processed in slot order and then robot-ID order; a request that would exceed either limit is suppressed and fails the corresponding legality rule. There is no packet loss, channel capacity limit, or unlisted delay.

At boundary `b`, let `g_i^j(b)` be the generation of robot `j` in robot `i`'s cache. A pair contributes to the weighted coordination graph only if it has a physical link. Its mutual state age, freshness factor, distance factor, and edge weight are

`age_ij(b) = b - min(g_i^j(b),g_j^i(b))`,

`fresh_ij(b) = max(0,1-age_ij(b)/5)`,

`range_ij(b) = max(0,1-(||p_i(b)-p_j(b)||_2/2.4)^2)`,

`w_ij(b) = fresh_ij(b) * range_ij(b)`.

All other weights are zero. Let `L(b)=D(b)-W(b)` be the symmetric weighted Laplacian and `lambda2(b)` its second-smallest eigenvalue. The coordination graph is topologically connected when `lambda2(b)>1e-9`; it meets the required coordination margin when `lambda2(b)>=0.30-1e-6`. Unlike a unit-weight connectivity test, this margin depends continuously on both separation and mutual state freshness.

A recovery window is a maximal consecutive run of boundaries in `0,...,60` at which the required margin is not met. Its start is the first boundary in that run. It succeeds only if the first later boundary meeting the margin occurs no more than the scene's recovery limit after its start; a run that continues through boundary 60 fails. A later deficient run opens a new window. This definition includes an initially deficient graph and a one-boundary deficiency.

#### Information and event order

At a decision boundary, robot `i` may use its ID; the current slot; its own exact position and goal; all public starts, goals, rectangles, constants, and limits; its four cache entries; current physical neighbour IDs discovered by the radio interface; its own past views and decisions; its preceding-three slot broadcast history; and the remaining team budget before current-slot requests. It may not use another robot's true current position except through a delivered cache entry, a message before arrival, another robot's current decision, a future physical link, or a future state. Robot ID may be used as a uniform priority or tie-break. The same callable procedure must handle all robots and scenes. Direct scene-, robot-, phase-, or absolute-slot-to-action tables are prohibited.

A joint implementation may first produce robot-local candidate actions from the permitted views and then resolve simultaneous segment conflicts using only those announced candidate segments, scalar priorities, public geometry, and robot IDs. The resolver must not inspect one robot's private cache to choose another robot's motion or broadcast.

Each slot follows this authoritative order: admit the broadcasts generated in the preceding slot and merge the arrival batch; refresh every robot's own boundary cache entry; form each permitted local view; choose the four direction and broadcast requests; replace out-of-domain or geometrically illegal directions by `stay` and suppress budget-exceeding broadcasts for diagnostic continuation; execute all motions simultaneously and test swept separation; determine the post-move physical graph; refresh each broadcaster's own payload entry, generate its fixed cache snapshot, and deliver it at boundary `t+1`; then compute the boundary-`t+1` weighted graph, goal events, and recovery state. Boundary 60 performs the last merge and derived-state computation but no action.

Submitted direction and broadcast fields are decisions at the stated boundary. Positions, physical links, messages, caches, edge weights, eigenvalues, safety events, goal events, recovery windows, and all metric values are execution consequences derived by the evaluator. The implementation and report, not a particular legal decision pattern, establish that the policy respected the stated information boundary.

#### Result contract, diagnostic continuation, and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. Object member names must be unique; `NaN`, infinities, and other non-JSON constants are invalid. The array contains exactly one record for each canonical scene; order is irrelevant. A scene record has exactly `scenario` and `slots`. `slots` contains exactly one record for every integer `t` in `0,...,59`; order is irrelevant. A slot record has exactly `t` and `actions`. `actions` contains exactly one record for every integer robot ID `0,...,3`; order is irrelevant. An action record has exactly integer `robot`, string `direction`, and Boolean `broadcast`. Booleans are not integers. The direction string must belong to the nine-value set above. These decisions are the complete required replay evidence. No derived state or aggregate field is required or accepted at these levels.

Unknown records are ignored by indexing and replay but count as coverage defects. At each level, the first record with an expected explicit ID is retained; later duplicates cannot replace it or provide missing descendants. A missing or unrelatable robot-slot decision becomes `stay,false` and fails only canonical coverage. A retained action whose container or field is malformed also becomes `stay,false` and fails its evidence atom. A well-typed but unknown direction becomes `stay` and fails direction legality. State-dependent legality is not inferred from or after the first incomplete or malformed slot in a scene, preventing a replacement from causing secondary safety diagnoses; field-domain and broadcast-count checks remain attributable. These continuations support diagnostics only: every task metric is zero unless all validity checks are full.

For `primary_metrics_1`, let `d0` be one robot's start-to-goal distance and `dmin` its smallest boundary-to-goal distance over boundaries `0,...,60`. Define `progress=clip((d0-dmin)/(d0-0.35),0,1)` and `reached=1` if the goal was reached and 0 otherwise. The robot score is `0.7*reached+0.3*progress`; the metric is its mean over 20 scene-robot trajectories.

For each scene, define `margin_quality` as the mean of `clip(lambda2(b)/0.30,0,1)` over post-move boundaries `b=1,...,60`. Define `recovery_quality` as the fraction of successful recovery windows when at least one exists, and as `margin_quality` otherwise. `primary_metrics_2` is the five-scene mean of `0.65*margin_quality+0.35*recovery_quality`.

For communication metrics, `sent` counts accepted team broadcasts. A sent broadcast is delivered when it has at least one recipient, while a recipient event is one broadcast-receiver pair. For every payload entry in every recipient event, its age is `arrival_boundary-generation`; its freshness value is `clip(1-age/5,0,1)`. Per scene, `delivery_fraction=delivered_broadcasts/max(1,sent)` and `payload_freshness` is the mean payload-entry freshness, or zero with no recipient event. `auxiliary_metrics_1` is the scene mean of `0.5*delivery_fraction+0.5*payload_freshness`.

A novel update is a payload entry whose generation is strictly newer than the recipient's pre-batch entry. Let `U` be the number of novel updates and `R` the number of recipient events in a scene. If `U=0`, that scene's synchronization-efficiency score is zero. Otherwise it is `0.7*U/(4*R)+0.3*clip(1-sent/120,0,1)`. `auxiliary_metrics_2` is the mean of this score over scenes.

Here `clip(x,0,1)=min(1,max(0,x))`. Higher is better for all metrics. Normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. All metric inputs are derived by fixed-model replay and are scored only when every validity check is full. The absolute numerical tolerance is `1e-6` unless a formula above states a strict threshold.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite the root-level `results.json` from project code and the fixed conditions above. The evaluator deletes an old result before the one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations. Controlled experiments used as Judge evidence may produce optional text or JSON artifacts outside `results.json`; for every claimed comparison, identify the conditions, shared inputs, changed factor, artifact path, and condition-level outcomes.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "slot, direction, and broadcast decision fields satisfy local types", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "directions, swept safety, obstacles, and broadcast budgets are legal", weight: 4}
- {id: replay_validity, desc: "motion, broadcasts, caches, weighted graphs, recovery, and goals replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed goal arrival and progress quality", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed connectivity-margin and recovery quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed broadcast delivery and payload freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed novel synchronization efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of obstacle motion, local caches, weighted connectivity, and recovery"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent motion-connectivity formulation or a materially wrong model. Implementation/evidence: no executable states or decisions correspond to the claimed model."}
    - {score: 0.4, desc: "Report: omits a decisive geometry, message timing, freshness-weight, recovery, safety, or goal relation. Implementation/evidence: code realizes only a partial or materially divergent model."}
    - {score: 0.7, desc: "Report: accurately defines the core simultaneous motion, local cache, physical link, weighted Laplacian, recovery, safety, and goal relations. Implementation/evidence: code and decisions realize them across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order, information boundaries, distance-freshness coupling, obstacle-induced separation, recovery deadlines, and objective trade-offs. Implementation/evidence: code faithfully realizes every central relation and exposes its operation without material divergence."}
- id: method_design
  desc: "unified local-state motion planning and freshness-aware broadcast control"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable procedure or supporting decisions."}
    - {score: 0.4, desc: "Report: a goal-only, fixed-formation, or fixed-broadcast rule ignores a decisive obstacle, recovery, freshness, or resource consequence. Implementation/evidence: pre-filled schedules dominate or code materially diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method coordinates obstacle-safe motion, local state estimates, connectivity risk, recovery, and broadcast use. Implementation/evidence: the same state-driven procedure runs across all scenes with limited lookahead or robustness."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly treats the central temporal, informational, geometric, and resource trade-offs. Implementation/evidence: permitted evolving local state drives motion and communication across contrasting conditions without scene-specific action tables or material divergence."}
- id: experimental_coverage
  desc: "controlled evidence for connectivity recovery, obstacle pressure, freshness, and communication"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or isolated goal and connectivity totals do not test the communication-motion mechanism. Implementation/evidence: outputs do not support the principal claims."}
    - {score: 0.7, desc: "Report: all canonical scenes and one controlled broadcast, freshness, margin, or recovery comparison directly test a central trade-off. Implementation/evidence: both conditions execute with comparable condition-level evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused communication comparison and a sensitivity or failure-boundary check under obstacle or recovery pressure. Implementation/evidence: reproducible condition-level trajectories, messages, graph margins, windows, and goals support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of freshness, geometric connectivity, recovery, and goal progress"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats goal or connectivity values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important freshness, distance-margin, recovery-delay, obstacle, and goal trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains when to maintain, temporarily lose, or recover coordination, quantifies principal sensitivities, tests a contrasting or failure boundary, and states scope and unknowns. Implementation/evidence: reproducible comparisons support the central conclusions."}
