---
id: T038_acoustic_object_map_exchange
name: Acoustic Object-Map Exchange for Robust Underwater Robot Mapping
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

Three autonomous underwater vehicles (AUVs) survey a bounded two-dimensional work area. Each AUV observes at most one nearby object per time slot. Objects include repeated shapes, so a local object observation is not by itself a reliable loop closure. The AUVs share one delayed acoustic channel and a verification service. A compact object-map summary can make observations from another AUV available, while a larger geometric verification packet can test a possible correspondence. Accepted loop closures alter the AUV pose drift used by later sampling decisions. The mission therefore couples movement, acoustic scheduling, object ambiguity, queueing, verification feedback, and sampling deadlines.

### Research Problem

Design and evaluate one deterministic, state-driven non-learning method for acoustic object-map exchange and underwater mission execution. Your method must decide when each AUV broadcasts its current local summary, which currently visible candidate edge to verify, whether to retransmit a packet after feedback, whether to accept the oldest completed verification feedback, and how each AUV moves. Study the trade-off between acoustic payload, queue delay, false loop closures, pose/map consistency, safety, and value collected before the public deadlines. The same decision procedure must handle all fifteen public scenario variants without indexing a precomputed action table by scenario, variant, robot, or absolute slot.

### Experimental Setting and Constraints

#### Canonical evaluation units

The work area is the integer grid `x=0..23`, `y=0..17`. There are `H=75` one-second action slots `0..74` and state boundaries `0..75`. The AUV IDs are `r0`, `r1`, and `r2`. The five base scenarios are `repeated_pylons`, `sparse_landmarks`, `depth_link_shift`, `crossing_surveys`, and `acoustic_burst`; each has `variant v` in `{0,1,2}`. The fifteen `(scenario,variant)` pairs are the canonical evaluation units. Variant `v` changes observation sigma from `0.06` to `0.06+0.06v`, shifts the public link-stress intervals by `6v`, and sets the packet hash salt to `19v`. Each unit starts from its table entry and ends at boundary `75`.

| scenario | starts `(r0,r1,r2)` | station | object coordinates `O0..O7` | targets `(object,value,deadline)` | drift |
|---|---|---|---|---|---:|
| `repeated_pylons` | `(1,2),(1,9),(1,15)` | `(0,9)` | `(4,3),(4,8),(4,13),(10,3),(10,8),(10,13),(18,5),(18,12)` | `(O0,1,20),(O2,1,26),(O3,2,38),(O5,2,48),(O6,3,62),(O7,3,70)` | `0.024` |
| `sparse_landmarks` | `(2,2),(2,15),(12,9)` | `(0,9)` | `(3,3),(6,14),(10,5),(13,13),(17,3),(19,10),(22,5),(22,15)` | `(O0,1,18),(O1,1,28),(O2,2,36),(O4,2,50),(O5,3,60),(O7,4,72)` | `0.028` |
| `depth_link_shift` | `(1,3),(1,14),(20,9)` | `(0,9)` | `(4,4),(4,13),(9,3),(9,14),(15,4),(15,13),(21,5),(21,12)` | `(O0,1,20),(O1,1,26),(O2,2,38),(O3,2,44),(O6,4,62),(O7,4,68)` | `0.026` |
| `crossing_surveys` | `(1,2),(1,15),(22,8)` | `(12,9)` | `(4,4),(4,13),(9,7),(9,11),(15,7),(15,11),(20,4),(20,13)` | `(O0,1,22),(O1,1,28),(O2,2,38),(O4,2,48),(O6,3,62),(O7,3,70)` | `0.022` |
| `acoustic_burst` | `(2,2),(2,15),(21,9)` | `(12,9)` | `(4,3),(4,14),(8,6),(8,11),(16,6),(16,11),(21,3),(21,14)` | `(O0,2,18),(O1,2,24),(O2,3,34),(O3,3,40),(O6,4,56),(O7,4,62)` | `0.025` |

#### State, actions, and task model

Object classes for `O0..O7` are `A,B,A,C,B,C,A,B`; dimensions are `A=(1.0,0.5)`, `B=(0.7,0.7)`, and `C=(1.2,0.4)`. At action start, the nearest object in Manhattan radius `4` is selected, ties by object ID; otherwise the observation is invisible. Observation ID is `obs=<scenario>|variant=<v>|robot=<i>|slot=<t>`. Every public observation has exactly `(obs_id,visible,class,size,g,d,sigma,q)`. When visible, `g=(dx,dy,w,h)`, `dx=x_object-x_robot-drift_x`, `dy=y_object-y_robot-drift_y`, `size=(w,h)`, `d=clip(ManhattanDistance/4,0,1)`, and `q=clip(1-0.04d-0.10sigma,0,1)`. When invisible, class, size, and `g` are null, `d=1`, and `q=0`. Sigma is `0.06+0.06v`, plus `0.04` for `depth_link_shift` at `t>=30+6v`. Canonical object identity is private environment truth and is not part of a public observation.

Each action contains `move`, `broadcast_metadata`, `validate_edge`, `retry`, and `accept_validation`. A summary contains exactly the latest `min(3,t+1)` local opportunities, including invisible ones, in chronological order. Its ID is `metadata|r<i>|generation=<t>`, deadline `t+8`, and payload `1`. A geometry packet has ID `verification|<edge_id>|r<i>|generation=<t>`, candidate-expiry deadline, and payload `6`. The channel admits whole packets by `(deadline,queue_age,robot_id,packet_type,edge_id_or_empty,attempt)`, where `metadata < verification`, `queue_age=t-generation`, capacity is `12` units per slot, and the unit budget is `600`. Admission consumes payload even if the link later fails. A request after its deadline is rejected; an admitted packet may arrive or finish service after its deadline but never after boundary `75`.

A packet sent by AUV `i` at position `p_i` has delay `1+floor(||p_i-station||_2/8)`. Let `d_n=clip(||p_i-station||_2/12,0,1)`. Link stress `z=1` for `depth_link_shift` at `t>=30+6v` and for `acoustic_burst` at `22+6v<=t<=33+6v`; otherwise `z=0`. For the `n`-th admitted packet in that slot, `p=clip(0.92-0.05d_n-0.08z-0.04(n-1),0.08,0.92)`. The link succeeds exactly when the first 64 bits of `SHA256(UTF-8("<scenario>|<variant>|<packet_id>|<attempt>|salt=<19v>"))`, divided by `2^64`, are below `p`.

On summary arrival, every not-previously-generated cross-robot observation pair with the same class and `similarity=clip(1-||g_a-g_b||_2/12,0,1)>=0.55` creates an edge. Its ID is `edge=<lexicographically smaller obs_id>--<larger obs_id>`, and it is visible from generation through `generation+10`, inclusive. Let `ell_a=(x_object_a-x_robot_a,y_object_a-y_robot_a)` and define `ell_b` likewise from private replay truth. Residual is `||((g_ax-g_bx),(g_ay-g_by))-(ell_a-ell_b)||_2+0.5|width_a-width_b|+0.5|height_a-height_b|`. The edge is consistent exactly when both observations refer to the same canonical object and residual is at most `0.35`.

Successful geometry packets enter a FIFO waiting queue of capacity `10`, excluding the active service. An idle service starts one waiting packet; a packet started at boundary `s` completes at `s+2`. Its sample passes exactly when the first-64-bit SHA-256 fraction for `<scenario>|<variant>|<edge_id>|validate|<attempt>|salt=<19v>` is below `0.95-0.05v`. Completion feedback exposes `(edge_id,residual,consistent,sample_pass)` to its owner. Accepting a consistent, sample-passing edge multiplies both endpoint drifts by `0.60`. Accepting an inconsistent, sample-passing edge adds `0.12` to each coordinate of both endpoint drifts. Other consumed feedback causes no drift update.

After movement and any correction, drift coordinate `k` receives increment `drift*(1+0.2v)*(2u-1)`, where `u` is the first-64-bit SHA-256 fraction for `<scenario>|<variant>|r<i>|t<t>|drift|<k>`. The first post-move visit to a sampling object records that boundary, AUV, and the post-correction, pre-increment drift norm; later visits do not replace it.

#### Information availability and event order

The starts, station, object coordinates/classes/sizes, and sampling target coordinates/values/deadlines in the table are public task inputs; the canonical object identity that generated a particular local observation remains private replay truth. At an action boundary, a policy may use the public table, its AUV's integer position, energy, current and prior public local observations, all successfully received public summaries, current unexpired edge IDs with similarity/generation/expiry, cumulative payload, current service/queue history, owner-visible failure feedback, owner-visible completion feedback, and past outcomes. It may not use the private identity behind an observation, unsent observations of another AUV, a candidate residual or consistency before completion feedback, or any future hash, link, arrival, queue, feedback, or terminal state.

Each nonterminal boundary executes in this order: (1) expire candidates; (2) deliver arrivals and generate candidates from summaries; (3) complete service due now, then start one waiting packet if idle; (4) generate observations from pre-acceptance drift and expose the permitted action-start state; (5) read all three actions and consume acceptance feedback; (6) form, order, admit, and sample packets; (7) execute safe movement and record first target visits from post-acceptance, pre-increment drift; (8) add drift increments for the next boundary. At boundary `75`, arrivals and a due completion close communication accounting, but there is no action, correction, movement, or sampling. Later events are discarded. Thus submitted fields are decision-boundary actions; observations and feedback are already visible inputs at that boundary; all packet, queue, graph, pose, and mission values are post-event evidence derived by replay.

#### Legal actions and boundaries

`move` is `stay`, `N`, `E`, `S`, or `W`. The safety executor processes `r0,r1,r2`. A non-stay destination is unsafe if it is outside the grid, already accepted for a lower-index AUV, or the action-start cell of a higher-index AUV; it becomes `stay` and counts once. An executed move costs `1` energy unit, an executed stay costs `0.2`, communication costs no energy, and each AUV starts with `60`. A transition below zero energy is illegal.

Each AUV may issue at most one new packet per slot. `validate_edge` is null or an edge visible at action start and may be newly requested only once. `retry` is null, `metadata`, or `verification`; at slot `t` it retransmits the oldest owned attempt-0 packet of that kind whose capacity, budget, link, or queue failure feedback was generated at `t-1` and whose original deadline is at least `t`. It preserves the packet ID, payload, and original deadline and uses attempt `1`. Missing that retry slot, reaching the deadline, or failing attempt `1` ends the packet. Retry and a new packet cannot coexist. Capacity, budget, link, and full-queue outcomes are legal fixed-model rejections, not illegal decisions. A new validation or retry after its deadline is illegal. `accept_validation` is null or Boolean; a Boolean consumes the owner's oldest completion feedback and is illegal if none exists. Feedback cannot be consumed twice.

#### Task metrics

Metrics are first calculated per canonical unit and then arithmetically averaged over all fifteen units. For `primary_metrics_1`, each target first visited no later than its deadline earns its value times `1` when drift error is at most `0.8`, otherwise times `0.5*clip(1-error/0.8,0,1)`; late or unvisited targets earn zero. Divide by that unit's total target value. For `primary_metrics_2`, `pose_MSE` is the mean squared drift norm over all `3*76` AUV-boundary samples. Each accepted correct graph edge contributes its residual, each accepted false edge contributes `16`, and `graph_residual` is their arithmetic mean; an empty graph also has residual `16`. The metric is `0.6*clip(1-pose_MSE/24^2,0,1)+0.4*clip(1-graph_residual/16,0,1)`.

`primary_metrics_3` is `0.6*clip(1-safety_events/(75*3),0,1)+0.4*clip(1-motion_cost/(75*3),0,1)`, where motion cost sums executed move/stay costs. `auxiliary_metrics_1` is `0.5*clip(1-accepted_units/600,0,1)+0.5*informative_units/max(1,accepted_units)`. Accepted units sum every channel-admitted attempt. Informative units add `6` when a consumed, sample-passing completion is correctly accepted if consistent or correctly rejected if inconsistent; summaries add none. `auxiliary_metrics_2` is `0.5*clip(1-queue_wait_slots/75,0,1)+0.5*clip(1-accepted_false_constraints/max(1,total_constraints),0,1)`. Queue wait is the sum of `service_start-arrival` for all geometry packets that start service; total constraints count consumed completion feedback, and accepted false constraints count consumed feedback accepted while inconsistent and sample-passing. All metrics are in `[0,1]`; normalized task performance is their `8,4,3,3,2` weighted mean.

#### Result record contract

`results.json` is strict finite JSON with exactly the root key `scenarios`. It contains exactly one record for every canonical unit, each with exactly `scenario` (string), `variant` (integer), and `actions` (array). Each actions array contains exactly one record for every `(slot,robot)`, hence `75*3=225` records. Every action has exactly: `slot` (integer `0..74`), `robot` (`r0`,`r1`,`r2`), `move` (the five-value enum), `broadcast_metadata` (Boolean), `validate_edge` (string or null), `retry` (null, `metadata`, or `verification`), and `accept_validation` (Boolean or null). Array order and JSON whitespace are immaterial. Observations, summaries, packets, candidates, graph state, mission state, and metrics are fixed-model derivatives and must not be submitted.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json` in that one execution. `report.md` must contain non-empty headings exactly named `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. It must explain the unified policy, all canonical units, controlled communication or verification evidence, and supported limitations. Additional project files and report sections are allowed, but external services, hidden files, and unpublished reference code are not.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains six non-empty required headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenario container", weight: 3}
- {id: result_evidence, desc: "scenario and action records contain the declared minimal decision fields", weight: 4}
- {id: canonical_coverage, desc: "all fifteen variants and slot-robot action keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "actions obey public visibility, retry, feedback, budget and energy rules", weight: 4}
- {id: replay_validity, desc: "fixed replay consistently derives observations, packets, service, drift, graph and sampling", weight: 6}
- {id: primary_metrics_1, desc: "deadline-qualified target sampling quality", weight: 8}
- {id: primary_metrics_2, desc: "pose and object-graph consistency quality", weight: 4}
- {id: primary_metrics_3, desc: "safety and motion efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "acoustic payload and informative-validation efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "verification queue and false-constraint efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the underwater observation, acoustic, verification, drift and sampling model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent coupled formulation; Implementation/evidence: no executable process corresponds to the task."}
    - {score: 0.4, desc: "Report: covers only motion or packets; Implementation/evidence: omits a decisive queue, drift or verification timing relation."}
    - {score: 0.7, desc: "Report: accurately states the variants, information boundary, packets, service and sampling; Implementation/evidence: one reusable process implements the main model with limited omissions."}
    - {score: 1.0, desc: "Report: precisely closes observation, packet, service, feedback, drift and sampling order; Implementation/evidence: all variants are faithfully realized with traceable decisions."}
- id: method_design
  desc: "quality of the unified acoustic exchange, verification, feedback and motion policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or uses prohibited training; Implementation/evidence: no legal unified policy."}
    - {score: 0.4, desc: "Report: uses a fixed object-pair or slot table or ignores queue and retry state; Implementation/evidence: actions violate information boundaries or the report."}
    - {score: 0.7, desc: "Report: gives a reproducible non-learning summary, verification, retry, acceptance and motion policy; Implementation/evidence: one program covers all variants and public timing."}
    - {score: 1.0, desc: "Report: justifies confidence, residual, queue age, budget, feedback and deadline decisions; Implementation/evidence: current state drives actions without a scenario or slot action table."}
- id: experimental_coverage
  desc: "coverage of acoustic mechanisms, variants, controls and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one condition or aggregate is shown; Implementation/evidence: no evidence that communication changes task outcomes."}
    - {score: 0.7, desc: "Report: all variants plus one controlled bandwidth, retry, verification, depth or drift comparison; Implementation/evidence: condition records support the comparison."}
    - {score: 1.0, desc: "Report: complementary single-factor controls cover mechanism and failure sensitivity; Implementation/evidence: every condition is executed and reconstructible."}
- id: research_analysis
  desc: "evidence-based interpretation of ambiguity, queues, feedback, drift and sampling trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradicts records; Implementation/evidence: conclusions are unsupported."}
    - {score: 0.4, desc: "Report: lists metrics without causal discussion or material limits; Implementation/evidence: condition evidence is insufficient."}
    - {score: 0.7, desc: "Report: explains communication, verification, drift, queue and sampling trends and states limitations; Implementation/evidence: replayed results support the claims."}
    - {score: 1.0, desc: "Report: quantifies interactions and failure boundaries and separates observation, inference and unknowns; Implementation/evidence: complementary condition evidence supports central claims."}
