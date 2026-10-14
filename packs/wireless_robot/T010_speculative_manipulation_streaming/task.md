---
id: T010_speculative_manipulation_streaming
name: Adaptive Speculative Control Streaming under Cloud-Robot Latency
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

A mobile manipulator follows a two-dimensional end-effector reference while a cloud controller supplies short predictive action fragments. The edge cache can mask cloud latency, but contact disturbances and model drift can make a cached prediction unsafe. Clearing a mismatched cache and holding safely costs progress, while long fragments reduce waiting but increase stale-state risk.

At every 20 ms control slot the local policy validates the cache head, executes it or performs a zero-acceleration safety hold, and may request a new fragment. A request is generated from the state visible at that slot and returns after the scene latency. The task succeeds when all three reference stages are tracked by their deadlines without leaving the safety workspace.

### Research Problem

Build a deterministic, non-learning simulation and one unified online policy for speculative control streaming. The policy chooses request depth and timing, generates nominal actions, validates cached predictions, and recovers from mismatch. It should balance latency masking, stage completion, tracking error, communication requests, and safe holds.

Use only the public trajectory and model, current state, cache and request history, and arrived fragments. The same reusable procedure must operate in all five scenes. Do not use future disturbances, post-action state while selecting a request, scene-specific schedules, or pre-filled action/result tables.

### Experimental Setting and Constraints

#### Canonical scenes and reference

Scenes, in order, are free_motion, contact_transition, multi_stage, high_latency, and drift_recovery. Each has slots t=0,...,47, initial position (0.0,0.0), velocity (0.0,0.0), workspace 0<=x<=10, 0<=y<=8, and a componentwise velocity limit of 1.5. Delays and environment truth are:

| scene | delay | contact centre | radius |
|---|---:|---|---:|
| free_motion | 2 | (20,20) | 0 |
| contact_transition | 4 | (2.35,1.45) | 0.85 |
| multi_stage | 3 | (5.05,1.05) | 0.90 |
| high_latency | 6 | (2.35,1.45) | 0.85 |
| drift_recovery | 4 | (6.0,2.8) | 0.95 |

Each scene also has a fixed deterministic disturbance sequence used only by the environment and independent replay. Its slots and vectors are hidden from the policy until their slots occur. The three stages are fixed for every scene:

| scene | disturbance slots | additive vector applied at each listed slot |
|---|---|---|
| `free_motion` | `15,16` | `(0.06,-0.03)` |
| `contact_transition` | `17,18,19` | `(-0.10,0.05)` |
| `multi_stage` | `12,13,30` | `(0.08,0.08)` |
| `high_latency` | `8,9,10,27` | `(-0.08,0.06)` |
| `drift_recovery` | `22,23,24` | `(0.12,-0.10)` |

Slots not listed have zero disturbance. The table is public scene definition, but a policy may not use a listed disturbance before that slot occurs.

| stage | slots | start | end | deadline |
|---:|---|---|---|---:|
| 0 | 0..15 | (0.0,0.0) | (2.5,1.5) | 15 |
| 1 | 16..31 | (2.5,1.5) | (5.8,0.8) | 31 |
| 2 | 32..47 | (5.8,0.8) | (8.6,4.8) | 47 |

For stage bounds [a,b], the target at slot t is start + ((t-a+1)/(b-a+1))*(end-start).

#### Dynamics and cache rule

An action is an integer pair in {-1,0,1}^2. Let g=0.55 inside the contact radius and g=1 otherwise. The no-disturbance transition is v'=clip(0.85*v+0.18*g*a,-1.5,1.5) componentwise and p'=p+v'. The scene disturbance is then added to p'; clipping to the workspace is recorded as violation, as is any velocity component whose absolute value exceeds 1.5. Other slots are normal or contact according to the pre-action contact test.

The cache is FIFO. Returned requests are appended at slot start in request-ID order. The head has nominal position p_hat and velocity v_hat; its error is e=||p-p_hat||_2+0.4*||v-v_hat||_2. A head with e<=0.15 must be executed and popped. Otherwise the whole cache is cleared and the action is zero (safe_hold). An empty cache also requires safe_hold. A trace decision is cache or safe_hold.

The policy may set request_depth to 0 or an integer 2..8. Nonzero sends one request, with at least two slots between sends and at most 18 requests per scene. The request uses the pre-decision state and returns at send_t+delay; returns after slot 47 are late and never enter the cache. It contains exactly depth commands for consecutive slots beginning at its return. The first nominal state is the request state propagated with zero action for the delay; subsequent nominal states use the submitted actions and the no-disturbance transition. Command actions and nominal states are candidate decisions; replay checks their consistency.

Slot order is: append returns; record pre-decision state and head; choose decision/action/depth; validate or clear and execute; apply dynamics and disturbance; create a request from the pre-decision state; record trace and events. Future disturbances and post-action state are not available when choosing the request.

#### Results contract

The root is exactly an object with scenarios. The five scene records are ordered canonically and each has exactly scenario, trace, requests, cache_events, and segments.

Each trace has 48 ordered records with exactly: t, target, position, velocity, segment, cache_size, cache_head, cache_error, decision, action, request_depth, request_id, post_position, post_velocity, safety_state, segment_complete. Positions, velocities and targets are finite length-two arrays; cache_head is null or {request_id,command_index,apply_t,action,nominal_position,nominal_velocity}; cache_error is finite or null; actions have two integers in -1..1. `decision` is exactly `cache` or `safe_hold`; `safety_state` is exactly `normal`, `contact`, or `violation` as determined by the dynamics. `request_id` is null exactly when `request_depth` is zero; otherwise it is the request ID of the request sent in that slot, with IDs `q0`, `q1`, ... assigned contiguously in send order.

Each request has exactly request_id, send_t, base_position, base_velocity, depth, delay, arrival_t, status, and commands; status is delivered for arrival at most 47 and late otherwise. Each command has exactly command_index, apply_t, action, nominal_position, nominal_velocity. Requests are in ID and send order.

Each cache event has exactly t, event, request_id, command_index, cache_size, reason. Events are append, consume, clear, or safe_hold; reasons are respectively returned, validated, mismatch, or empty. A returned command produces one append event in command order, with the cache size after that command is appended. A consumed command produces one consume event with the cache size after the head is removed. A mismatch produces one clear event followed by the slot's safe_hold event; both have null request and command IDs and cache size zero. An empty cache produces one safe_hold event with null IDs and cache size zero. Events are recorded in this causal order within each slot. `segments` has exactly three ordered records with segment, start_t, end_t, target_end, finish_t, completed, and max_error; `max_error` is the maximum Euclidean distance from `post_position` to that slot's target over the segment. A stage finishes at its end slot when endpoint distance is at most 0.9 and it had no violation; otherwise finish_t is null. Numeric tolerance is 1e-6; all numbers are finite and no self-reported metrics or random seeds are accepted.

#### Metrics

With clip(x,0,1)=min(1,max(0,x)), independent replay computes:

- primary_metrics_1: completed scene-stage pairs divided by 15.
- primary_metrics_2: mean over scenes of clip(1-mse/4.0,0,1), with 48-slot squared post-action position error against the target at the same slot.
- auxiliary_metrics_1: mean over scenes of clip(1-n_requests/18,0,1).
- auxiliary_metrics_2: mean over scenes of clip(1-(n_safe_hold+n_mismatch)/48,0,1).

The normalized task performance is (8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20. Metrics are scored only when all validity checks pass.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root directory must contain run.py, report.md, and results.json. run.py is the only official entry point and must generate or overwrite results.json. report.md must contain these non-empty headings exactly: Problem Formulation, Method Design, Experimental Setup, Results, Research Analysis, and Conclusion and Limitations.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "trace, request, cache-event, and segment records provide declared evidence", weight: 4}
- {id: canonical_coverage, desc: "all scene, trace-slot, request, event, and segment keys are covered", weight: 4}
- {id: decision_legality, desc: "actions, cache decisions, request limits, and information-time rules obey constraints", weight: 4}
- {id: replay_validity, desc: "independent replay matches fragments, dynamics, cache events, safety, and stage outcomes", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed safe scene-stage completion fraction", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed normalized post-action tracking quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed request efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed safe-hold and mismatch cost", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of the latency, cache, dynamics, safety, and stage-tracking task"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent latency/cache/robot model. Implementation/evidence: no corresponding trace."}
    - {score: 0.4, desc: "Report: a decisive dynamics, delay, cache, or safety relation is missing. Implementation/evidence: records omit or contradict it."}
    - {score: 0.7, desc: "Report: state, action, delay, validation, disturbance, and stage objectives are correct. Implementation/evidence: the core model covers all scenes."}
    - {score: 1.0, desc: "Report: event order, nominal/real coupling, contact, mismatch recovery, and boundaries are precise. Implementation/evidence: records faithfully expose them."}
- id: method_design
  desc: "design of unified adaptive requests, nominal fragments, cache validation, and recovery"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal policy or prohibited training is used. Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: fixed depth or schedule ignores latency-safety trade-offs. Implementation/evidence: decisions rely on scene-specific tables or diverge from the report."}
    - {score: 0.7, desc: "Report: a legal reusable state/history-driven request and cache policy is given. Implementation/evidence: the same procedure runs across scenes with main recovery choices."}
    - {score: 1.0, desc: "Report: a justified adaptive policy coordinates depth, timing, error, remaining stages, and recovery. Implementation/evidence: decisive actions are unified and traceable without action tables."}
- id: experimental_coverage
  desc: "coverage of latency masking, disturbances, cache thresholds, and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only aggregate or one latency case is shown. Implementation/evidence: no mechanism-level evidence."}
    - {score: 0.7, desc: "Report: all scenes and at least one controlled latency, disturbance, cache, or validation comparison are executed. Implementation/evidence: comparable traces support it."}
    - {score: 1.0, desc: "Report: complementary controlled conditions and a failure-boundary check explain masking and safety cost. Implementation/evidence: condition-level records distinguish the mechanisms."}
- id: research_analysis
  desc: "evidence-based interpretation of delay, drift, cache, and safety trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts results. Implementation/evidence: records cannot support the claims."}
    - {score: 0.4, desc: "Report: scores are repeated without mechanism explanation or limitations. Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: replayed traces explain cache, delay, disturbance, and recovery trends, distinguish observation from inference, and state limitations. Implementation/evidence: results support the analysis."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies sensitivity and failure boundaries, explains long-fragment masking versus drift, and limits scope. Implementation/evidence: reproducible condition-level records support conclusions."}
