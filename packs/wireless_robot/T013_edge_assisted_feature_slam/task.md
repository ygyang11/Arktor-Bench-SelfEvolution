---
id: T013_edge_assisted_feature_slam
name: Communication-Aware Keyframe Selection for Edge-Assisted Multi-Robot SLAM
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

Three robots traverse an indoor area while maintaining local pose estimates from odometry. Their estimates drift over time. Each frame also exposes nearby visual feature tokens. A robot can upload a compact summary or a more expensive keyframe to an edge-cloud service, but transmissions share a limited frame capacity and total byte budget and may be lost or delayed. Serviced summaries produce modest edge tracking corrections. Serviced keyframes can form cross-robot loop constraints against a bounded cloud keyframe store and produce stronger corrections. Consequently, upload timing, retry, queue service, and keyframe retention jointly affect trajectory accuracy, cross-robot consistency, communication load, and correction freshness.

### Research Problem

Build a deterministic, non-learning simulator and one unified online policy for all canonical scenes. At each frame, the policy must select at most one queued packet for service, decide whether a serviced keyframe is retained and which stored keyframe is evicted when necessary, and select each robot's transmission action. The method must use only information available at that decision boundary. It should preserve accurate local trajectories and establish diverse, low-residual cross-robot loops while limiting transmitted bytes and stale corrections. The same reusable state-driven procedure must handle drift growth, feature novelty, lossy attempts, retry opportunities, queue age, and the finite cloud store without scene-specific schedules.

### Experimental Setting and Constraints

#### Canonical scenes, trajectories, and observations

The canonical scene order is `hall_loop`, `split_route`, `delayed_merge`, `sparse_landmarks`, and `drift_corridor`. Each scene contains robots `0,1,2` and frames `t=0,...,39`. Define

`tri(q)=10-|((q mod 20)-10)|`.

The continuous true pose in grid units is

`P[s,r,t]=[0.5+tri(t+x_phase[s,r]), 0.5+tri(t+y_phase[s,r])]`.

The fixed scene parameters are:

| scene | `x_phase` for robots 0,1,2 | `y_phase` for robots 0,1,2 | feature radius | drift scale |
|---|---|---|---:|---:|
| `hall_loop` | `[0,4,8]` | `[5,9,13]` | 3.20 | 0.85 |
| `split_route` | `[0,10,5]` | `[10,0,15]` | 3.00 | 1.00 |
| `delayed_merge` | `[0,7,14]` | `[14,7,0]` | 2.80 | 1.20 |
| `sparse_landmarks` | `[0,6,12]` | `[3,11,19]` | 2.35 | 1.10 |
| `drift_corridor` | `[0,2,4]` | `[10,12,14]` | 2.90 | 1.45 |

There are 16 landmarks. For `k=0,...,15`, token `fKK` uses a two-digit `KK` and landmark position

`Q[k]=[1.5+3*(k mod 4), 1.5+3*floor(k/4)]`.

Robot `r` observes token `fKK` in frame `t` exactly when the Euclidean distance from `P[s,r,t]` to `Q[k]` is no greater than the scene's feature radius. Token arrays are sorted by increasing token ID. True poses, unobserved tokens, future observations, and current or future link outcomes are environment information and are not policy inputs.

The local pose starts at `L[r,0]=P[s,r,0]`, and uncertainty starts at `U[r,0]=0.05`. For `t>0`, let `j` be the zero-based scene index and let `d_s` be the scene drift scale. The odometry increment is the true pose increment plus

`d_s*[0.032*(r-1)+0.006*(((t+r+j) mod 3)-1),` `     0.026*(1-r)+0.005*(((2*t+r+j) mod 3)-1)]`.

After feedback application as defined below, this increment is added to the local pose and `0.07*d_s` is added to uncertainty. Frame 0 has no odometry increment. All formulas use ordinary double-precision arithmetic.

#### Packets, attempts, and queue

A new `summary` or `keyframe` captures the sender's current local pose, uncertainty, and visible token array. Packet IDs reset in each scene and are assigned `p0,p1,...` in new-packet generation order, with robot decisions applied in robot order inside a frame. A summary with `n` tokens costs `8+6n` bytes; a keyframe costs `32+14n` bytes. A retry uses the same packet ID, payload, mode, and byte cost and does not create a new packet.

Every attempt counts toward all resource limits. Attempt bytes in one frame must total at most 120; each scene has a 2400-byte total; and every inclusive window `max(0,t-3),...,t` may contain at most three attempts. Each robot selects exactly one action from `skip`, `summary`, `keyframe`, or `retry` per frame. `retry_packet_id` is non-null exactly for `retry`. A packet is retryable by its original robot from the frame after its first attempt is known to be dropped until it is retried. Each packet may be retried at most once; a delivered, queued, serviced, late, or already retried packet is not retryable. At a decision boundary, `retryable_packet_ids` contains exactly the packets currently retryable by that robot and is ordered by increasing numeric packet-ID suffix.

The edge is at `[5.5,5.5]`. For an attempt by robot `r` in frame `t`, let `m` be the numeric packet ID, `h=0` for the initial attempt and `h=1` for its retry, and

`d=floor(|P_x[s,r,t]-5.5|+|P_y[s,r,t]-5.5|+1e-12)`.

The attempt is dropped when `d>8` or `(5t+3r+7j+m+2h+d) mod 13` belongs to `{0,1}`. Otherwise its delay is `1+((t+r+j+m+h+d) mod 3)`. Its arrival frame is `t+delay`; an arrival at 40 or later is late. Delivered packets enter an unbounded service queue at the beginning of their arrival frame. The queue is ordered by `(arrival_frame,numeric_packet_id)`. A service decision is `null` or one ID in this service-before queue. At most one packet is serviced per frame and each packet is serviced at most once.

#### Fixed edge-cloud update, loops, and retention

Servicing any packet creates exactly one feedback message for its sender. Feedback IDs reset in each scene and are assigned `f0,f1,...` in service order. For a packet generated in frame `u`, define its stored pose error as `e=L_packet-P[s,r,u]`.

A serviced summary creates feedback with gain `0.20` and arrival one frame after service. A serviced keyframe is compared, in numeric packet-ID order, with every keyframe already in the cloud store from a different robot. A pair is an accepted loop when the two token arrays share at least two tokens and

`rho=||(L_current-L_previous)-(P_current-P_previous)||_2 <= 0.5`.

The ordered loop record is `[current_packet_id,previous_packet_id]`. A keyframe creates feedback with gain `0.75` if at least one loop is accepted and gain `0.40` otherwise; it arrives two frames after service. Feedback with gain `g` adds `-g*e` to the sender's current local pose and multiplies current uncertainty by `1-g`. Feedback arriving at frame 40 or later has no evaluated effect. Feedback is applied in increasing numeric feedback-ID order. A robot's `applied_feedback_ids` contains exactly the feedback applied at the start of the current frame, not cumulative feedback history, and uses that same numeric-suffix order.

The loop comparisons use the pre-service cloud store whether or not the current keyframe will be retained. After comparisons and feedback creation, retention is processed. The store holds at most 12 keyframes. For a serviced keyframe, `retain_serviced_keyframe=false` requires `evict_packet_id=null`. If retention is true and the store has fewer than 12 entries, eviction must be null. If retention is true and the store already has 12 entries, the eviction ID must name exactly one currently retained keyframe; that keyframe is removed before the current keyframe is appended. A summary or null service requires retention false and eviction null. The retained array is ordered by retention time.

#### Information boundary and frame order

At a robot's transmission boundary, its rule may use its current local pose, uncertainty, currently visible token array, feedback applied to that robot, its own retryable packet IDs and history, the remaining shared byte budget, the number of attempts in the previous three frames, and earlier robot transmission choices in the current frame. It may not use another robot's unreported current observation. The cloud rule may use the service-before queue with each queued packet's ID, sender, generation frame, mode, bytes, tokens, stored local pose, and uncertainty; the retained keyframes with the same fields; prior cloud events; and resource history. It may not use true poses, future tokens, unarrived packets, or unseen current link results.

Each frame has this exact order: enqueue delivered attempts; apply arriving feedback; for `t>0` apply the current odometry increment and uncertainty increase; expose current robot observations, retryability, service-before queue, retained store, and resource state; jointly choose cloud and robot decisions; service the selected packet, determine loops and feedback, and update retention; process the three robot transmission actions in robot order and derive their link events; then record the frame. Current-frame transmissions cannot enter the queue or affect current-frame service.

Decision fields record choices at this shared boundary. Robot trace fields record information already visible at that boundary. Queue IDs are the service-before queue; retained IDs, accepted loop pairs, and the feedback ID are post-service evidence. This event order is authoritative. A unified method may use the frame number as state but must not map scenes, robots, regimes, or absolute frames to prefilled transmission, retry, service, retention, or eviction tables.

#### Result contract and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. The array contains the five scenes in canonical order. Each scene object has exactly `scenario`, `decisions`, `robot_trace`, and `cloud_trace`.

`decisions` contains exactly 40 records in frame order. Each has exactly `t`, `service_packet_id`, `retain_serviced_keyframe`, `evict_packet_id`, and `robots`. The two IDs are a packet-ID string or `null`; retention is Boolean. `robots` contains exactly three records in robot order. Each has exactly `robot` as an integer, `transmit` as one of the four action strings, and `retry_packet_id` as a packet ID or `null`, with the action-field relations defined above.

`robot_trace` contains exactly 120 records in `(t,robot)` order. Each has exactly `t`, `robot`, `local_pose`, `uncertainty`, `visible_tokens`, `applied_feedback_ids`, and `retryable_packet_ids`. `local_pose` is a two-number finite array in grid units; uncertainty is a finite nonnegative number. Token and ID fields are unique string arrays in the orders defined above.

`cloud_trace` contains exactly 40 records in frame order. Each has exactly `t`, `queue_packet_ids`, `retained_keyframe_ids`, `accepted_loop_pairs`, and `feedback_id`. The first two are unique packet-ID arrays in their defined orders. Each accepted pair is a two-element packet-ID array `[current,previous]`, ordered by previous numeric packet ID, and pairs are unique. The feedback ID is a string or `null`.

Packet payloads, attempts, link outcomes, byte totals, pose errors, loop residuals, feedback vectors, and metric summaries are derived by the grader and are not submitted as duplicate event tables. Integers reject Booleans. Numeric evidence is compared with absolute tolerance `1e-6`; discrete fields and arrays must match exactly.

`primary_metrics_1` evaluates frames `9,19,29,39`. In each scene, RMSE is the square root of the mean squared Euclidean local-pose error over these four frames and three robots. Its score is `clip(1-RMSE/2,0,1)`, averaged over scenes. The fixed localization-utility boundary is 2 grid units: an RMSE at or above that mission error radius has zero retained trajectory utility.

For `primary_metrics_2`, every accepted loop has quality `clip(1-rho/0.5,0,1)` and signature `(lower_robot,higher_robot,two lexicographically smallest shared tokens)`. Only the best quality for each signature in a scene is retained. The scene score is `clip(sum(best qualities)/4,0,1)`, averaged over scenes. The fixed fusion target is four units of diverse loop quality per scene; once their summed best-signature quality reaches four, the cloud graph is treated as sufficiently constrained and additional signatures add no metric utility.

`auxiliary_metrics_1` is the scene mean of `clip(1-attempt_bytes/2400,0,1)`. `auxiliary_metrics_2` is averaged over every scene, frame, and robot. If the robot has no applied feedback yet, its value is 0. Otherwise, let `u` be the generation frame of its most recently applied feedback's packet; the value is `max(0,1-(t-u)/6)`.

Higher is better for every metric. The normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. Metrics are scored only when all validity checks pass.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite root-level `results.json` from the project code and fixed conditions above. The evaluator deletes an old result before one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations.

Controlled experiments used as Judge evidence may be documented in `report.md` and supported by additional project-generated text or JSON artifacts outside `results.json`. Such artifacts are optional for Auto Checks and do not change the exact `results.json` contract. The report must identify the compared conditions, shared inputs, changed factor, artifact path, and condition-level outcomes for any comparison it claims.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "decision, local-pose, feature, queue, loop, and feedback evidence is typed", weight: 4}
- {id: canonical_coverage, desc: "all scene, decision-frame, robot-frame, and cloud-frame keys are covered", weight: 4}
- {id: decision_legality, desc: "transmissions, retries, service, retention, capacity, and budgets are legal", weight: 4}
- {id: replay_validity, desc: "odometry, packets, queue, loops, retention, and feedback replay matches", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed multi-robot trajectory accuracy", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed diverse cross-robot loop quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed feature-transmission byte efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed applied-feedback freshness", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of drifting local poses, feature uploads, cloud loops, and feedback correction"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent edge-assisted multi-robot localization model. Implementation/evidence: no corresponding executable states or records."}
    - {score: 0.4, desc: "Report: omits a decisive odometry, packet, queue, loop, retention, or feedback relation. Implementation/evidence: records implement only a partial or divergent model."}
    - {score: 0.7, desc: "Report: accurately defines the core local poses, feature packets, queue, cloud store, loops, corrections, and resource constraints. Implementation/evidence: code and records implement them across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order, information boundaries, drift, loop validation, delayed correction, and accuracy-resource consequences. Implementation/evidence: code and records faithfully expose every central coupling without material divergence."}
- id: method_design
  desc: "unified online transmission, retry, service, and keyframe-retention decisions"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable procedure or decision evidence."}
    - {score: 0.4, desc: "Report: the method is dominated by fixed schedules or isolated choices that ignore decisive drift, feature, queue, retry, or store consequences. Implementation/evidence: unit-specific decision tables dominate or code diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method addresses upload mode, retry, service, and retention using available state and history. Implementation/evidence: one procedure runs across all scenes with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly treats drift risk, feature novelty, loop value, packet age, shared resources, and store diversity. Implementation/evidence: evolving visible state drives all decisive choices without scene-specific schedules."}
- id: experimental_coverage
  desc: "controlled evidence for keyframe, bandwidth, delay, loop, and correction trade-offs"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or uninformative aggregates do not examine the upload-loop-correction relation. Implementation/evidence: outputs do not support the principal claims."}
    - {score: 0.7, desc: "Report: all scenes and one controlled upload, retry, service, retention, or bandwidth comparison test a central trade-off. Implementation/evidence: both conditions are executed with comparable condition-level evidence."}
    - {score: 1.0, desc: "Report: complementary mechanism and sensitivity or failure-boundary experiments distinguish keyframe value, congestion, loop diversity, and correction effects. Implementation/evidence: reproducible condition-level traces and metrics support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of drift, feature value, communication, and map consistency"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats accuracy or byte values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important drift, queue, loop, feedback, and resource trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies central sensitivities, tests a mechanism against contrasting or failure-boundary evidence, and states scope and unknowns. Implementation/evidence: reproducible comparisons support the central conclusions."}
