---
id: T036_keyframe_selection_slam
name: Bandwidth-Aware Keyframe Selection for Collaborative Robot Mapping
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

Three mobile robots survey an obstructed indoor site while a nearby edge server builds a shared landmark map. Each robot maintains a local pose estimate and observes only landmarks within its short sensing radius. Local tracking accumulates drift, while the server can reduce drift and close map loops when it receives useful keyframes from different robots. The wireless uplink is shared: a frame can wait in a local queue, be transmitted with one of three payload sizes, be lost and retried once, or be discarded when its age or a queue limit is reached. Delayed or redundant frames consume capacity without improving the map, while withholding every frame leaves the robots with drifting local estimates and can prevent them from reaching their goals by the required deadline.

At every action boundary, one unified controller chooses a legal motion and a retention/encoding decision for each robot, and may request at most one already queued keyframe for upload. Following the fixed event order below, the simulator first processes edge arrivals and service, then ranks and samples upload requests, applies motion safety, generates post-motion observations, and updates the queues and pose graph before exposing the resulting feedback at the next boundary. The research objective is to preserve deadline-qualified navigation and map consistency while controlling payload load, queue delay, and dropped frames.

### Research Problem

Build a deterministic collaborative-mapping simulator and a reusable state-driven method for all fifteen canonical scenario variants. Decide when a newly generated local keyframe is worth retaining, which public compression level to use, which queued keyframe to transmit, and how to move each robot while drift, loop-closure opportunities, shared bandwidth, edge service, retry windows, and navigation deadlines evolve. The method should use only the fixed task inputs, the robot's currently visible observations and history, received server corrections, and action-start queue/link state. It may use deterministic heuristics, planning, search, or optimization, but it must not use machine learning, training, or future hash samples.

### Experimental Setting and Constraints

#### Canonical Evaluation Units

There are five base scenarios, each with variants `v=0,1,2`. The canonical key is `(scenario, variant)` and the fifteen keys are the Cartesian product of:

| Scenario | Starts `(r0,r1,r2)`; goals `(r0,r1,r2)` | Closed rectangular obstacles | Landmarks | Deadline; drift coefficient |
|---|---|---|---|---|
| `loop_rich` | `(1,2),(1,15),(20,8)`; `(19,15),(19,2),(11,9)` | `[7,8]x[0,6]`; `[7,8]x[11,17]` | `(4,3),(4,14),(11,4),(11,14),(18,9)` | `56`; `0.018` |
| `drift_corridor` | `(2,2),(2,15),(19,2)`; `(19,15),(11,9),(19,8)` | `[9,12]x[6,7]`; `[9,12]x[10,11]` | `(5,4),(5,13),(15,4),(15,13)` | `64`; `0.026` |
| `bandwidth_burst` | `(1,1),(1,16),(20,1)`; `(20,16),(11,9),(20,8)` | `[6,7]x[5,12]`; `[15,16]x[5,12]` | `(4,2),(4,16),(11,3),(11,15),(19,9)` | `60`; `0.020`; capacity half on `24..35` |
| `heterogeneous_observation` | `(2,8),(10,2),(19,8)`; `(2,15),(11,9),(19,2)` | `[8,9]x[3,14]` | `(4,8),(11,2),(11,16),(18,8)` | `68`; `0.022`; noise multipliers `(1,1.5,0.7)` |
| `crossing_routes` | `(1,3),(1,14),(20,3)`; `(20,14),(11,9),(20,8)` | `[10,11]x[0,5]`; `[10,11]x[12,17]` | `(4,4),(4,13),(17,4),(17,13)` | `62`; `0.024` |

Coordinates are integer cells on a `22x18` grid, with `x=0..21` and `y=0..17`. The edge server is fixed at `(11,9)`. A cell is traversable when it is in the grid and outside every listed closed rectangle. Every scenario has action slots `t=0..71` and state boundaries `b=0..72`; boundary `0` contains the listed starts, energy `100` for each robot, zero drift, empty local queues, an empty server queue, and an empty map.

#### State, Actions, and Task Model

At action-start boundary `t`, robot `ri` has its true grid position, energy, two-dimensional local drift, a local queue of retained frame records, and the already revealed transmission/service history. The controller may read its own true grid position and energy, its current local pose estimate (the true position plus the currently accumulated local drift), landmarks within Euclidean radius `3`, received server corrections, and its own queue and revealed outcomes. A controller may use the public fixed geometry and formulas, but it may not use an unobserved landmark, an ungenerated frame, a future queue result, a future fusion correction, or a future hash sample.

Each action record contains exactly `move`, `encode`, and `send_keyframe`. `move` is one of `stay,N,E,S,W`; `encode` is one of `drop,sparse,balanced,full` and applies to the frame generated after this robot's movement; `send_keyframe` is null or the exact ID of one frame in this robot's action-start local queue. A non-null send may occur while the robot moves. A frame generated in slot `t` is not available for sending until boundary `t+1`. The deterministic frame ID is `scenario=<name>|variant=<v>|robot=<i>|slot=<t>`.

For robot `i`, landmark `j`, and slot `t`, let `d=clip(||p_{i,t}-L_j||_2/3,0,1)`. Only landmarks with distance at most `3` are visible. Let `m_i=1` except that `heterogeneous_observation` uses `(m_0,m_1,m_2)=(1,1.5,0.7)`, and let `sigma=m_i*(0.10+0.08v)`. Observation quality is `q=clip(1-0.12*sigma-0.04*d,0,1)`. For each visible landmark and axis `a` in `{x,y}`, define

```text
u = uint64(SHA256(UTF-8("<scenario>|<variant>|r<robot>|t<slot>|l<landmark>|a<axis>|obs"))[:16]) / 2^64
xi = 2*u - 1
```

The local observation is `L_j + drift + 0.25*sigma*(1+d)*(xi_x,xi_y)`. These samples are fixed model facts and are revealed only after the corresponding action. At the end of each slot, every robot creates one candidate frame opportunity containing its visible landmark observations and current local pose. An `encode=drop` action retains no frame. Otherwise the frame has the following fixed `(payload_units, retained_fraction, correction_gain)`:

| Encoding | Payload units | Retained fraction | Correction gain |
|---|---:|---:|---:|
| `sparse` | 1 | 0.50 | 0.18 |
| `balanced` | 2 | 0.75 | 0.30 |
| `full` | 4 | 1.00 | 0.46 |

For a non-drop frame with `m` visible landmarks, the fixed retained fraction keeps `k=max(1,ceil(retained_fraction*m))` landmarks, or all landmarks when `m=0`; the retained subset is the first `k` items sorted by descending observation quality and then ascending landmark ID. The frame stores only that subset for fusion, while its priority remains the sum of all visible landmark qualities plus `0.02` times the visible-landmark count. Each local queue holds at most `8` retained frames. At every action-start boundary, a frame whose age `boundary-generation_slot` is strictly greater than `12` is removed. When a new retained frame would exceed this capacity, the queue keeps the prefix sorted by `(generation_slot+12, -priority, generation_slot, frame_id)` and drops the rest. The local queue and its packet metadata are visible to its owner at the next boundary.

The shared uplink has `8` payload units per action slot. Requests are sorted by `(generation_slot, -priority, robot_id, frame_id, attempt)` and complete frames are admitted while capacity remains; a frame is never split. The `bandwidth_burst` scenario has capacity `4` rather than `8` on slots `24..35`. The edge server has a waiting FIFO capacity of `6` frames and consumes `2` payload units per action-start boundary. A successful transmission arrives at the next boundary; if the waiting FIFO is full at that boundary it is dropped. When multiple successful transmissions arrive at the same boundary, they are appended and checked for FIFO overflow in the admitted request order of their transmission slot, namely `(generation_slot, -priority, robot_id, frame_id, attempt)`. A first failed attempt may be requested only in the immediately following slot with the same frame ID and attempt `1`; if it is not requested, is not admitted, or fails again, it is dropped. A successful frame is removed from its local queue when its transmission succeeds. A request consumes `0.05` energy per admitted payload unit.

For a request, `d_link` is the Euclidean distance from the action-start robot position to `(11,9)` divided by the grid diagonal and clipped to `[0,1]`. Let `b=1` only for `bandwidth_burst` slots `24..35`, and let `n` be the number of requests admitted in that slot. Its probability is `p=clip(0.93-0.05*d_link-0.06*b-0.03*(n-1),0.10,0.93)`. For attempt `a`, the public transmission ID is `scenario=<name>|variant=<v>|robot=<i>|slot=<generation_slot>|attempt=<a>`, and the sample is `uint64(SHA256(UTF-8(tx_id+"|salt="+str(13v)))[:16])/2^64`; the attempt succeeds exactly when the sample is less than `p`. The sample is evaluated after the action and cannot be used to choose that action.

The server services the waiting FIFO at each action-start boundary after processing arrivals. If no frame is active, it starts the oldest waiting frame; it then consumes up to `2` remaining payload units. When a frame reaches zero remaining units, it is fused at that boundary. Fusion adds its observed landmarks to the map in landmark-ID order. For each landmark, the frame contributes a graph edge residual equal to the Euclidean norm of its observation minus the fixed landmark coordinate; if the landmark was already mapped, that residual is multiplied by `0.85`. The source robot's drift is multiplied by `(1-correction_gain)` when the frame completes. A fusion correction is visible to the source robot only at that boundary after service and before its next action.

After a proposed move, robots are resolved in order `r0,r1,r2`; a destination outside the grid, inside an obstacle, already accepted by a lower-order robot, or equal to a higher-order action-start position is replaced by stay and increments `safety_events` once only when the proposed move is a valid non-stay enum. A malformed move is a legality failure and does not add a safety event. A valid move costs `1`, a stay costs `0.2`, and an admitted payload unit costs `0.05`; energy may not become less than `-1e-6`. After post-motion observations and frame retention are generated, drift receives a deterministic per-axis increment at the end of the slot: `drift += c*(1+0.2v)*(2u-1)` for each axis, where `c` is the scenario drift coefficient and `u` is SHA-256 of `"<scenario>|<variant>|r<robot>|t<slot>|drift|<axis>"` using the same first-16-hex-digit over `2^64` rule. The goal and map state are fixed-model results, not submitted fields.

The fixed event order is: (1) at action-start, process arrivals and server service, apply completed fusion, remove over-age local frames, and expose current state; (2) read all three action records; (3) rank and admit complete transmission requests; (4) evaluate admitted link samples; (5) resolve safe movement and energy; (6) generate local observations and the encoded frame for the post-motion cell; (7) update local queues, schedule successful arrivals, retry/drop state, drift, graph and mission state; (8) record the next boundary. Decision fields describe candidates submitted at the action boundary; observations, probabilities, samples, queues, service, fusion, drift, graph, safety and metrics are derived replay facts.

#### Information Availability and Event Order

The controller may use the current robot-local state, current local queue metadata, previously revealed transmission outcomes, completed server corrections, fixed scenario geometry, public deadlines, capacities and formulas. It may not use future observations, future frame contents, future service outcomes, future positions, future queue retention, or future SHA-256 samples. The fixed event order above is authoritative. A frame or correction produced at boundary `t+1` is unavailable to the action submitted for slot `t` and becomes visible only at the stated next boundary.

#### Legal Actions and Boundaries

The only legal moves and encodings are the listed strings. A non-null `send_keyframe` must be a retained frame in the same robot's action-start queue; malformed or future frame IDs are a legality failure. A retry frame must be requested in the immediately following slot with the same ID and attempt `1`. A request that does not fit the remaining slot capacity is an environment admission outcome, but a malformed request, invalid reference, invalid encoding, or resulting energy below `-1e-6` fails `decision_legality`. Safety replacement of a valid but unsafe proposed move is a replay safety event, not a legality failure. At terminal boundary `72`, arrivals and service are processed and any still-open retry window is closed as dropped; no action is read at boundary `72`.

#### Task Metrics

All metrics are computed from fixed replay and clipped to `[0,1]`. For each robot, let `arrival` be the first boundary at which its true position equals its goal. `primary_metrics_1` uses quality `1` when `arrival` is at most the scenario deadline, `0.5*clip(1-(arrival-deadline)/max(1,72-deadline),0,1)` when later, and `0.5*clip(1-shortest_distance(final_position,goal)/max(1,shortest_distance(start,goal)),0,1)` when never reached; it averages robots and variants. `primary_metrics_2` is `0.6*clip(1-rmse_pose_error/22,0,1)+0.4*clip(1-map_residual/16,0,1)`, where pose RMSE averages drift norms over all `73*3` boundaries and map residual is the final mean graph residual, with empty graph `16`. `primary_metrics_3` is `0.6*clip(1-safety_events/(72*3),0,1)+0.4*clip(1-motion_cost/(72*3),0,1)`. `auxiliary_metrics_1` is `0.5*clip(1-total_transmitted_payload_units/864,0,1)+0.5*informative_payload_units/max(1,total_transmitted_payload_units)`, where informative units are payload units of fused frames whose addition strictly lowers at least one landmark's best residual. `auxiliary_metrics_2` is `0.5*clip(1-edge_busy_slots/72,0,1)+0.5*clip(1-dropped_or_expired_frames/max(1,total_retained_frames),0,1)`; `edge_busy_slots` counts action slots `0..71` after that slot's action-start arrivals and service when an active frame or waiting FIFO entry remains; terminal boundary `72` is not a slot. `total_retained_frames` counts every non-drop candidate frame generated after movement, including frames later removed by queue overflow. `dropped_or_expired_frames` counts frames removed by queue overflow, age expiry, retry failure/window closure, or edge FIFO overflow, and excludes fused frames and successful pending arrivals. When no frame is retained the second term is `1`. The normalized performance is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result Record Contract

`results.json` must be exactly `{"scenarios":[...]}`. There is one record for every `(scenario,variant)` key, with exact keys `scenario`, `variant`, and `actions`. Each `actions` array contains exactly `72*3=216` unique `(slot,robot)` records with exact keys `slot`, `robot`, `move`, `encode`, and `send_keyframe`. `slot` is an integer in `0..71`; `robot` is `r0`, `r1`, or `r2`; `move` is one of `stay,N,E,S,W`; `encode` is one of `drop,sparse,balanced,full`; and `send_keyframe` is null or the exact public frame ID of an action-start local-queue frame. Scenario and action array order is irrelevant. No positions, drift, observations, frames, packets, probabilities, samples, queues, service records, graph edges, mission summaries or metric values may be required from the candidate because the grader derives them from these decisions and fixed rules.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The implementation should use one reusable, deterministic state-driven procedure for every canonical scenario variant. It may use the current action-start state, fixed geometry and formulas, and already revealed history, but it must not use future hash samples, future observations, future queue outcomes, or pre-filled scenario/variant/slot action tables.

The project must contain `run.py`, `report.md`, and `results.json` in its candidate workspace root. `run.py` must create or overwrite `results.json` in one execution. `report.md` must contain non-empty headings exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Controlled experiments must be clearly separated from the canonical result and must not add extra records or fields to canonical `results.json`.

## Auto Checks

- {id: entry_point, desc: "root run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official execution finishes and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains six non-empty required headings", weight: 2}
- {id: result_json, desc: "results.json has strict JSON and the exact scenario container", weight: 3}
- {id: result_evidence, desc: "scenario and action records contain the declared decision fields and types", weight: 4}
- {id: canonical_coverage, desc: "all fifteen scenario variants and slot-robot keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "moves, encodings, action-start frame references, admission, retry timing and energy obey the public rules", weight: 4}
- {id: replay_validity, desc: "the fixed replay consistently derives observations, transmissions, service, fusion, navigation and metrics", weight: 6}
- {id: primary_metrics_1, desc: "deadline-qualified navigation quality", weight: 8}
- {id: primary_metrics_2, desc: "pose and map consistency quality", weight: 4}
- {id: primary_metrics_3, desc: "safety and motion efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "payload and informative-fusion efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "edge queue and retained-frame efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the collaborative keyframe, observation, communication, fusion, queue, navigation, and event-order model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent keyframe, map, communication, or navigation model; Implementation/evidence: no executable process corresponds to the claimed task."}
    - {score: 0.4, desc: "Report: describes only part of compression or navigation and omits a decisive drift, loop, queue, or service relation; Implementation/evidence: state or event order materially departs from the task."}
    - {score: 0.7, desc: "Report: accurately defines the fifteen variants, local observations, keyframes, compression, edge service, queues, deadlines, and boundaries; Implementation/evidence: one reusable procedure realizes the principal model with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely closes event order, information visibility, compression/fusion mapping, queue and navigation trade-offs, and assumptions; Implementation/evidence: code and minimal decisions faithfully realize all variants and principal derived facts."}
- id: method_design
  desc: "quality of the unified keyframe selection, encoding, upload scheduling, motion, retry, and navigation method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, a prohibited training method, or future-sample access; Implementation/evidence: no legal unified procedure."}
    - {score: 0.4, desc: "Report: relies on every-frame transmission, fixed sampling, or static compression and ignores drift, loops, queues, or deadlines; Implementation/evidence: decisions are not a reusable state-driven process or diverge materially from the report."}
    - {score: 0.7, desc: "Report: presents a legal, unified, reproducible keyframe, compression, upload, and motion method addressing principal temporal constraints; Implementation/evidence: the same procedure covers all variants, with limited joint-planning depth."}
    - {score: 1.0, desc: "Report: justifies a unified method jointly handling information gain, loop value, compression cost, queue age, fusion delay, retry, energy, and navigation deadlines; Implementation/evidence: decisive actions arise from evolving state without scenario, variant, or slot action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for selection, compression, bandwidth, observation, queue, and service mechanisms"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only a canonical aggregate or an uninformative single condition is given and selection/compression effects are not tested; Implementation/evidence: records cannot support the main claims."}
    - {score: 0.7, desc: "Report: all canonical variants are covered and at least one two-sided comparable bandwidth, compression, loop, queue, service, or observation comparison is executed; Implementation/evidence: both conditions retain reproducible decisions and outcomes."}
    - {score: 1.0, desc: "Report: at least two complementary single-factor comparisons test selection/compression mechanisms and a bandwidth, observation, queue, or service failure boundary; Implementation/evidence: condition-level artifacts support comparisons without contaminating canonical results."}
- id: research_analysis
  desc: "evidence-based explanation of information selection, compression, fusion, queue, navigation, and deadline interactions"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: code and outputs cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly lists payload or error values without causal explanation of selection, compression, fusion, queues, or navigation and without material limitations; Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: uses actual results to explain keyframe, compression, service delay, queue, and deadline trends, distinguishing observation, inference, and limitations; Implementation/evidence: records support the comparisons."}
    - {score: 1.0, desc: "Report: uses reproducible comparisons and a failure boundary to quantify interactions among information value, compression, loop closure, queue age, and navigation, while stating scope and unknowns; Implementation/evidence: condition-level evidence supports the central mechanism claims."}
