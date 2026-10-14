---
id: T026_robotic_sensor_collection_codesign
name: Motion-Assisted Sensor Collection with Communication-Aware Obstacle Avoidance
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

A mobile collection robot moves through a bounded warehouse-like area and gathers data from four wireless sensors. The robot follows a coarse route toward a terminal point, but static obstacles can hide parts of the route. A sensor transmission has two coupled effects: its power and the robot-sensor distance determine collected data, and a sufficiently reliable observation window can deliver a short-lived report describing obstacle geometry near that sensor. Reports arrive after transmission and can reveal obstacles that are outside the robot's local view. Thus communication decisions affect both the data collected and the safety of later motion. The uplink is from each sensor to the robot; the robot's motion and power command takes effect in the selected slot, while a report becomes usable only at its stated arrival boundary.

### Research Problem

Build a self-contained, non-training simulation and a reusable state-driven policy for the closed loop. In every control slot choose a bounded nonholonomic motion command and one power level for each sensor. Balance total collection, the weakest sensor, transmit energy, route and deadline progress, and collision avoidance. The policy must use only observations available at that time: local obstacle visibility, previous link outcomes, arrived reports, pose and power history, and remaining time and energy. It must not use future link results, hidden obstacle geometry, or a table of actions indexed by scenario or absolute slot.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios: `open_collection`, `occluded_crossing`, `far_sensor`, `power_burst`, and `fairness_route`. The rectangular domain is `[0,20] x [0,12]`; the robot has radius `0.3`, and the safety clearance used for collision is `0.45`. Slots are `t=0,...,59`, each of length `Delta t=0.2 s`; state boundaries are `0,...,60`. The state is `(x,y,theta,v,omega,remaining_tx_energy,done)`, with the scenario table below supplying the initial pose, terminal point, deadline, four sensor coordinates, coarse route, obstacle rectangles, and local-view range.

| scenario | initial pose; goal/deadline | sensors | coarse route | obstacle rectangles; local view |
|---|---|---|---|---|
| `open_collection` | `(2,2,0)`; `(7.2,2.0)` / `60` | `(2.8,3.0),(4.1,1.1),(5.4,3.0),(6.7,1.1)` | `(2,2)->(4.6,2)->(7.2,2)` | none; `1.5` |
| `occluded_crossing` | `(2,3,0)`; `(7.0,3.0)` / `60` | `(2.8,4.0),(4.0,2.0),(5.2,4.0),(6.4,2.0)` | `(2,3)->(4.5,3)->(7,3)` | `[4.05,4.55]x[2.55,3.45]`; `1.2` |
| `far_sensor` | `(2,6,0)`; `(7.0,6.0)` / `58` | `(2.5,7.0),(3.8,5.0),(5.2,8.0),(6.6,5.0)` | `(2,6)->(4.5,6)->(7,6)` | `[4.5,5.1]x[6.2,7.4]`; `1.4` |
| `power_burst` | `(3,9,0)`; `(8.0,9.0)` / `56` | `(3.5,10.0),(4.8,8.0),(6.2,10.0),(7.5,8.0)` | `(3,9)->(5.5,9)->(8,9)` | `[5.2,5.8]x[8.55,9.45]`; `1.1` |
| `fairness_route` | `(10,3,0)`; `(15.0,3.0)` / `60` | `(10.6,4.0),(11.8,2.0),(13.6,5.0),(14.4,2.0)` | `(10,3)->(12.5,3)->(15,3)` | `[12.6,13.2]x[3.3,4.5]`; `1.3` |

#### State, action and task model

The robot update for a legal action is `x'=x+0.2*v*cos(theta)`, `y'=y+0.2*v*sin(theta)`, and `theta'=wrap(theta+0.2*omega)` in `[-pi,pi)`. Actions satisfy `0<=v<=0.5`, `abs(omega)<=1`, `abs(v-v_previous)<=0.2`, and `abs(omega-omega_previous)<=0.5`; the previous action at slot `-1` is zero. For successive transmission events, including events separated by zero-power slots, the Euclidean difference of the submitted `(v,omega)` pairs is at most `0.75`. A slot is a collision if its motion segment intersects any fixed true obstacle expanded by `0.45`, or if its endpoint is within `0.45` of any fixed true obstacle. This environment-truth collision test is independent of which obstacles the policy currently knows. An endpoint outside the domain is out of bounds. Either event sets `done`; reaching the goal does not set `done`. After `done`, the realized motion is `(v,omega)=(0,0)` and the pose remains fixed, although all 60 submitted decisions must remain legal. Submitted sensor powers continue to consume energy, collect data at the fixed terminal pose, and participate in report generation. The goal is reached at the first boundary within `0.35` of the goal. Route alert is true when the boundary pose is more than `1.2` from the route polyline.

The four sensor powers are chosen from `{0, 0.00025, 0.0005, 0.00075, 0.001}` W and must sum to at most `0.002` W per slot. The cumulative transmit-energy budget is `0.012` J. With `d=max(0.5, distance)` at the slot-start pose, `R=100000*log2(1 + 0.001*p/(1e-9*d^2))` bit/s, collected bits are `0.2*R`, and energy is `0.2*p`. A sensor is rate-qualified in a slot when `R>=700000` bit/s. Two consecutive qualified slots generate one versioned report at the next state boundary; the run must become unqualified before another version can be generated. A report's payload is the intersection of the sensor square `[sx-1,sx+1]x[sy-1,sy+1]` with each scene obstacle and the domain, encoded as sorted `[xmin,ymin,xmax,ymax]` rectangles. A report arriving at boundary `a` is valid through `min(60,a+3)`. Only a strictly newer version replaces a cached report; expiry removes it. The first report from a sensor has version `1`, and later reports increment that sensor's version by one.

#### Information availability and event order

At each boundary, process arrivals and expiry, fuse local and active external obstacle payloads, choose the action from currently available information, execute motion and link collection, check safety against the fixed true obstacles and update the goal state, generate any report at the slot end, then record the next boundary. The known local/external obstacle union is a policy input and determines the reported obstacle-distance snapshot; it does not replace environment truth in the collision test. The strategy cannot inspect obstacle truth outside local visibility, future reports, or future link outcomes. For pose `(x,y)` and closed rectangle `[xmin,xmax]x[ymin,ymax]`, define the point-to-rectangle distance as `sqrt(max(xmin-x,0,x-xmax)^2 + max(ymin-y,0,y-ymax)^2)`. Local visibility includes rectangles whose distance by this formula is no greater than the scenario's local-view range. External payloads are the active report rectangles; `min_distance` uses the same formula over the union of local and active external rectangles.

#### Legal actions and boundary conditions

Submitted values must obey the stated motion, power, energy, report-lifetime, terminal-state and domain boundaries. Collision and out-of-bounds flags are cumulative after first occurrence. Post-terminal submitted motion remains subject to the action and continuity bounds but cannot alter the realized pose; post-terminal powers retain the communication behavior defined above.

#### Task metrics

The primary metrics are: (1) the mean over scenarios of the indicator for reaching the goal by the scenario deadline without collision or out-of-bounds; and (2) the mean over scenarios of `0.6*clip(total_bits/D_total_ref,0,1) + 0.4*clip(min_k(bits_k/D_fair_ref),0,1)`. The references are relaxed fixed-model upper bounds obtained by enumerating legal per-slot power combinations at `d=0.5` and multiplying the best per-slot value by 60; retaining the per-slot power limit while relaxing the cumulative energy limit makes them valid upper bounds rather than achievable targets. Auxiliary metrics are (1) mean `clip(1-total_power/(60*0.002),0,1)`, where `total_power` is the sum over slots of each slot's four-sensor total power, and (2) mean of `0.6*arrival_quality+0.4*mean(report_freshness)`. Arrival quality uses the earliest fused report from any sensor and is `clip(1-arrival_state/60,0,1)` when that arrival is within boundaries `1,...,60`, otherwise zero. Freshness at state `s` for a sensor is `clip(1-(s-arrival_state)/4,0,1)` for its freshest nominally valid report, otherwise zero, and is averaged over all four sensors and states `1,...,60`. The four metric weights are `8,7,3,2`, and the normalized task performance is their weighted sum divided by `20`; all metric values are independently replayed and lie in `[0,1]`.

#### Result record contract

`results.json` must be a strict JSON object with only a `scenarios` array containing one record per canonical scenario. Each record has exactly `scenario`, `decisions`, `states`, `links`, `reports`, `obstacles`, and `collection`.

* `decisions` has 60 records ordered by increasing `t`, with `t`, `v`, `omega`, `powers`  (four power values), and `total_power`.
* `states` has 61 records ordered by increasing `state_t`, with `state_t`, finite  `x,y,theta,v,omega,remaining_tx_energy`,  `route_deviation`, booleans `done,collision,out_of_bounds,goal_reached`, and  `goal_arrival_state` (integer or null).
* `links` has 240 records keyed by `sensor,t`, ordered by increasing `t` then `sensor`, with  `distance`, `power`, `rate`, `bits`, and `threshold_met`. The recorded `distance` is the raw  Euclidean slot-start distance; only the rate formula replaces it by `d=max(0.5,distance)`.
* `reports` is a dynamic array ordered by increasing non-negative integer `report_id` (unique and  assigned in generation order). Each record has `report_id`, `sensor`, strictly increasing  `report_version` per sensor, `generated_slot`, `arrival_state`, `trigger_slots` (the two  qualifying slots), `region`, `obstacle_intersections`, `valid_until_state`, `fusion_status`,  and `used_states`. A generated report always has  `arrival_state=generated_slot+1` in `1,...,60`; `used_states` is the nominal inclusive integer  list from arrival through valid-until. Actual cache use at a boundary is identified by  `active_report_ids`; if a newer version replaces an older one, the older report can remain in  its nominal `used_states` interval without being active. `fusion_status` is a string and must  equal `fused`: under the fixed one-boundary delivery and strictly increasing per-sensor  versions, every generated report arrives within the horizon and replaces that sensor's older  cache entry. `region` is the clamped sensor square itself, encoded as `[xmin,ymin,xmax,ymax]`; each entry in `obstacle_intersections` uses the same order and is the non-empty rectangle intersection payload. All obstacle and sensor rectangles are closed; an intersection that only touches at a boundary is retained as a zero-width or zero-height rectangle. `trigger_slots` is the increasing chronological pair `[t-1,t]`.
* `obstacles` has 61 records ordered by increasing `state_t`, with lexicographically sorted  `local_visible` and sorted  `external_fused` entries `[sensor,version,rectangle]`, `active_report_ids`, `min_distance`,  `collision`, `out_of_bounds`, and `route_alert`. `min_distance` is the Euclidean distance from  the pose to the union of local and active external rectangles, or `20.0` when that union is  empty. This known-obstacle distance is not the true-obstacle collision test. The collision and  out-of-bounds values are cumulative flags. Rectangles in both lists use  `[xmin,ymin,xmax,ymax]`. External entries are sorted by sensor, version, then rectangle, and  `active_report_ids` is sorted in increasing numeric order.
* `collection` has four records ordered by increasing `sensor`, with `total_bits`,  `valid_report_count`, `first_valid_state`, `last_valid_state`, `fairness_component`, and  `deadline_met`. `valid_report_count` is the number of that sensor's reports whose final status  is `fused`; `first_valid_state` is their earliest `arrival_state`, `last_valid_state` is their  latest `valid_until_state`, and both are null when no report is fused. `deadline_met` is true  exactly when `first_valid_state` is non-null and no greater than that scenario's goal deadline.  `fairness_component` equals that sensor's independently replayed total bits.

All fixed keys, lengths, domains, ordering rules and numeric values are checked with absolute tolerance `1e-6`. Dynamic reports do not replace the fixed canonical coverage units.

The preceding root, record, field, ordering and tolerance definitions are the complete result record contract; no additional hidden fields or defaults are used.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root directory must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must create or overwrite `results.json`. `report.md` must contain the non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact root and scenario structures are present", weight: 3}
- {id: result_evidence, desc: "decision, state, link, report, obstacle and collection evidence is typed and complete", weight: 4}
- {id: canonical_coverage, desc: "all fixed scenario, slot, state and sensor keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "motion, power, energy and transmitted-command continuity constraints are legal", weight: 4}
- {id: replay_validity, desc: "independent motion, link, report, obstacle and collection replay agrees", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed safe deadline arrival quality", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed collection and fairness quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed transmit-power efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed arrival timing and report-freshness quality", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the motion, sensor link, report and obstacle-information model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent model; Implementation/evidence: no executable collection and safety process."}
    - {score: 0.4, desc: "Report: a decisive relation such as power, threshold reports or collision timing is omitted; Implementation/evidence: records cover only part of the model."}
    - {score: 0.7, desc: "Report: motion, power budget, path loss, report validity, information boundary and deadlines are accurate; Implementation/evidence: the core model runs for all units with only minor omissions."}
    - {score: 1.0, desc: "Report: event order and the interaction of distance, power, report age, obstacles, route and deadline are precise; Implementation/evidence: link, report, trajectory and safety records faithfully realize them."}
- id: method_design
  desc: "quality of the unified non-training motion and power policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or prohibited method; Implementation/evidence: no legal reusable procedure."}
    - {score: 0.4, desc: "Report: fixed route/power tables or local choices ignore report age, budget or sensor fairness; Implementation/evidence: scenario-specific actions or material divergence."}
    - {score: 0.7, desc: "Report: a legal reusable policy addresses motion, power, safety and collection trade-offs; Implementation/evidence: one procedure covers all scenarios with supporting records."}
    - {score: 1.0, desc: "Report: a justified reproducible policy jointly handles rate, report threshold, active obstacle knowledge, remaining energy, route and fairness; Implementation/evidence: decisions are state-driven without action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for communication-assisted safety and collection"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only aggregate collection or one condition is shown; Implementation/evidence: no evidence for the central communication relation."}
    - {score: 0.7, desc: "Report: all canonical scenes and at least one explicitly identified controlled power, report, fairness or route comparison beyond canonical aggregate contrasts with comparable inputs and records are covered; Implementation/evidence: condition-level records support it."}
    - {score: 1.0, desc: "Report: complementary controlled conditions include a mechanism comparison and a sensitivity or failure-boundary check; Implementation/evidence: trajectories, link events, reports and metrics distinguish the claims."}
- id: research_analysis
  desc: "evidence-grounded conclusions and limitations"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradictory analysis; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: values are repeated without causal explanation of distance, power, reports and safety, and limits are absent; Implementation/evidence: claims are unsupported."}
    - {score: 0.7, desc: "Report: replayed records support conditional trends and trade-offs, with observations separated from inference and material limitations stated; Implementation/evidence: evidence agrees."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies communication/safety/fairness interactions and failure boundaries with scope and unknowns; Implementation/evidence: condition-level records reproduce the analysis."}
