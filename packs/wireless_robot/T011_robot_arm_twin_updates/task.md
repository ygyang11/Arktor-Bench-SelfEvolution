---
id: T011_robot_arm_twin_updates
name: Phase-Aware Wireless Updates for Digital-Twin Robot-Arm Reconstruction
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

An industrial robot arm executes a finite pick-and-place program while an edge digital twin reconstructs its joint, gripper, and contact state. Wireless updates consume a byte budget and can be delayed or lost. Stale twin state can deny a safety authorization at a phase transition, making the physical arm hold position or eventually abort. Sending every field every slot improves reconstruction but wastes bandwidth; sparse, phase-aware updates can preserve the task while reducing communication.

### Research Problem

Build a deterministic, non-learning simulator and one reusable online policy that chooses when to send an update and which of joint_position, gripper, and contact_force fields to include. The policy must use current physical and twin state, visible phase, arrived acknowledgements, send history, and remaining budget. It must balance twin accuracy, update freshness, byte cost, and physical progress across all five scenes without future state or link outcomes.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenes in this order: `pick_place`, `pick_toss`, `push_pull`, `contact_transition`, and `pause_dominant`. Each has slots `t=0,...,39`. The physical program index starts at `0` and can advance to `39`. A state has joint position `q=(q1,q2)` in `[-1,1]^2`, velocity `v=(v1,v2)` in `[-0.25,0.25]^2`, binary gripper `g`, and force `f` in `[0,1]`.
#### State, actions and task model

The nominal state is public and is exactly the following deterministic function. Let `off = 0.01 * scene_index`, where scene indices follow the canonical order above. Start with `q1 = -0.65 + 0.03*k
+ off` and `q2 = -0.35 + 0.02*k - off`; additionally add `0.025*sin(k/3)` to `q1` for `push_pull`, and add `0.02*sin(k/2)` to `q2` for `pick_toss`. Clip each joint coordinate to `[-1,1]`. Set `v=(0,0)` at `k=0`; otherwise set `v_i=q_i(k)-q_i(k-1)`. Set `g=1` exactly for `10 <= k < 30`, and `g=0` otherwise. Set `f=0.7` for `25 <= k < 30`, `f=0.45` for `10 <= k < 15`, and `f=0.08` otherwise. The physical state at slot `t` is the nominal state at the current program index. The policy receives only the current physical state and current phase, not future nominal rows.

The phase is reach for indices 0..9, grasp for 10..14, transport for 15..24, contact for 25..29, and release for 30..39. Phase-specific authorization thresholds are (e_q_max,e_v_max,e_g_max,e_f_max): reach=(0.08,0.10,--,--), transport=(0.08,0.10,--,--), grasp=(0.05,0.08,0,0.12), contact=(0.05,0.08,0,0.08), and release=(0.06,0.08,0,0.12). A dash means that field is not required. Errors are normalized as e_q=mean_i min(1,abs(q_i-qhat_i)/2), e_v=mean_i min(1,abs(v_i-vhat_i)/0.5), e_g=0 or 1, and e_f=min(1,abs(f-fhat)).

#### Information availability and event order

At the beginning of slot `t`, initialize the twin once to the nominal state at `(scene, k=0)` and set every field age to `6`. At every later slot, first increment each age by one and clip it to `6`, then process delivered updates arriving in that slot. A delivered `joint_position` payload replaces the twin joint position and sets twin velocity to the component-wise clipped difference between the new twin joint position and its previous twin joint position, clipped to `[-0.25,0.25]`. A delivered `gripper` or `contact_force` payload replaces only that field. A field updated in the slot has age `0`; other fields keep their incremented age. No other prediction is applied.

Compute the four errors. For authorization, use the current phase except when the attempted `p -> p+1` advance crosses a phase boundary; in that case use the target phase's threshold row. If authorization passes and `program_step<39`, the physical index advances by one at slot end; if it fails before task abortion or completion, the arm holds and records one `safety_intervention`. Once `program_step=39` is reached, the task is complete and remaining fixed-horizon rows are padding: there is no further authorization attempt, progress, or intervention. An attempted crossing into `grasp`, `contact`, or `release` counts consecutive failures only for that one target phase. The counter is cleared after any successful advance, when the target phase changes, or at task termination. Three consecutive failures on the same target crossing set `task_aborted=true` at the end of the third failure slot. After abortion or completion, the remaining fixed-horizon records are padding: no authorization attempt, physical progress, or safety intervention is counted, while already scheduled delivered messages may still be processed for trace completeness. Padding rows are excluded from critical-phase compliance denominators.

#### Legal actions and resource bounds

An update is sent after the gate decision, using the current physical state and a non-empty sorted subset of the three fields. Field sizes are `joint_position=24`, `gripper=8`, and `contact_force=12` bytes. A send in slot `t` has `arrival_t=t+1`. The delivery acknowledgement is visible at the beginning of that same arrival slot, so `ack_t=arrival_t` for a delivered update. The fixed loss sets are:

| scene | loss send slots |
|---|---|
| `pick_place` | `{3, 11, 23}` |
| `pick_toss` | `{5, 12, 26}` |
| `push_pull` | `{2, 16, 28}` |
| `contact_transition` | `{8, 18, 27}` |
| `pause_dominant` | `{6, 7, 21, 32}` |

The policy does not know the current or future loss set before sending. A send in a listed slot is `dropped`, has `ack_t=null`, and does not refresh fields. A send with `arrival_t>39` is `late`, has `ack_t=null`, and does not refresh fields. Other sends are `delivered`. At most two consecutive slots may contain sends, total bytes per scene are at most `1760`, and at most one message is sent per slot. `send_t` is an integer in `0..39`; messages sent in slot `39` are late.

The exact slot order is: increment ages; process arriving updates and acknowledgements; apply the public twin update rule; compute errors and authorization; choose fields for this slot; execute the authorized physical transition or hold; resolve the deterministic link result; update resources and transition counters; record all evidence. The policy cannot inspect future nominal rows, future physical states, or current/future loss outcomes.

Record semantics are fixed. Decision fields (`physical_trace` state, `twin_trace` state, errors, phase, and authorization) describe the beginning-of-slot state after arrivals and before the current send. Observation/feedback fields are the delivered payloads, `status`, and `ack_t` resolved for that send. Post-event fields are the next physical index implied by the gate, updated resources, abort state, and the update's link result. A trace row's `program_step` is the pre-transition index; it does not report the end-of-slot index.

#### Task metrics

Metrics are calculated by independent replay. `primary_metrics_1` is the mean over scenes of `0.5*reconstruction_compliance + 0.5*physical_execution`, where compliance is the fraction of non-padding slots in entered critical phases passing authorization and physical execution is zero on abort, otherwise `0.7*(final_program_step/39)+0.3*clip(1-safety_interventions/40,0,1)`. `primary_metrics_2` is the mean over all 200 slots of `clip(1-(0.35e_q+0.25e_v+0.25e_g+0.15e_f),0,1)`. `auxiliary_metrics_1` is the mean of `clip(1-bytes/1760,0,1)` per scene. `auxiliary_metrics_2` is the mean of `clip(1-mean_field_age/6,0,1)` per scene. The normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result recording contract

`results.json` has exactly `{"scenarios":[...]}`. Each scene record has exactly `scenario`, `physical_trace`, `twin_trace`, `updates`, `phases`, and `task_outcome`. Both traces have exactly 40 ordered records. A physical record has `t`, `program_step`, `q`, `v`, `gripper`, `contact_force`, `phase`, `errors`, `gate_passed`, `safety_intervention`, and `task_aborted`; `program_step` and all state/error values are the pre-transition values defined above. A twin record has `t`, `q`, `v`, `gripper`, `contact_force`, and `ages` (a mapping for all three fields with integer ages in `0..6`). Each update has `message_id=m0,m1,...`, integer `send_t` in `0..39`, a non-empty sorted fields subset, `bytes`, `arrival_t=send_t+1`, `status` (`delivered`, `dropped`, or `late`), `ack_t` (the integer arrival slot for delivered, otherwise `null`), and a `payload` containing exactly the selected current physical field values from `send_t`. Updates are in send order. `phases` has exactly the five records in canonical phase order. Each record has the phase name, fixed `start_index` and `end_index`, `entered`, and `first_t`/`last_t`: these are the minimum/maximum trace slots whose `program_step` lies in that phase range, or `null` when `entered=false`; terminal padding rows are included if their recorded `program_step` lies in the range, but are excluded from critical-phase compliance. For a padding row after completion or abortion, `gate_passed` is exactly `false` and `safety_intervention` is exactly `0`. `task_outcome` has exactly `final_program_step`, `completed_fraction=final_program_step/39`, `safety_interventions`, `task_aborted`, and `completed`. All numeric values are finite and compared with absolute tolerance `1e-6`; no self-reported metric or random seed is accepted.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite UTF-8 `results.json` without interactive input. `report.md` must contain six substantive Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`. The report must describe the actual implementation and results, including the assumptions, the communication-control trade-off and the limitations supported by the experiments.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run creates a fresh results.json within the limit", weight: 4}
- {id: report_contract, desc: "report.md has the six required nonempty sections", weight: 2}
- {id: result_json, desc: "strict JSON root and scene containers match the contract", weight: 3}
- {id: result_evidence, desc: "physical, twin, update, phase, and outcome evidence is present", weight: 4}
- {id: canonical_coverage, desc: "all scenes and 40-slot trace keys are covered exactly once", weight: 4}
- {id: decision_legality, desc: "field, byte, spacing, acknowledgement, and information rules are legal", weight: 4}
- {id: replay_validity, desc: "independent replay matches state, updates, authorization, and outcomes", weight: 6}
- {id: primary_metrics_1, desc: "replayed critical-phase compliance and physical execution", weight: 8}
- {id: primary_metrics_2, desc: "replayed normalized twin-state quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "replayed byte efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "replayed field freshness", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of phase-aware physical/twin reconstruction and wireless consequences"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent entity-twin model. Implementation/evidence: no corresponding traces."}
    - {score: 0.4, desc: "Report: omits a decisive field, phase, link, or authorization relation. Implementation/evidence: records are partial or divergent."}
    - {score: 0.7, desc: "Report: accurately defines states, phases, updates, ages, errors, and safety authorization. Implementation/evidence: core model covers all scenes."}
    - {score: 1.0, desc: "Report: precisely explains phase-specific information value, ACK timing, stale-state effects, and physical consequences. Implementation/evidence: traces faithfully expose these relations."}
- id: method_design
  desc: "unified field-selection and update-timing method under budget"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or prohibited training. Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: fixed full updates or fixed fields ignore phase, age, ACK, or budget. Implementation/evidence: scene-specific schedules dominate."}
    - {score: 0.7, desc: "Report: legal reusable state-driven field and timing policy. Implementation/evidence: same procedure runs across scenes with limited adaptation."}
    - {score: 1.0, desc: "Report: justified adaptive policy coordinates phase, error margin, field value, ACK history, and bytes. Implementation/evidence: decisions are unified and traceable."}
- id: experimental_coverage
  desc: "coverage of phase, loss, bandwidth, and freshness trade-offs"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one aggregate condition is shown. Implementation/evidence: no mechanism-level comparison."}
    - {score: 0.7, desc: "Report: all scenes and one controlled phase, loss, budget, or threshold comparison. Implementation/evidence: comparable records support it."}
    - {score: 1.0, desc: "Report: complementary mechanism and failure-boundary conditions quantify field and timing trade-offs. Implementation/evidence: condition-level evidence distinguishes them."}
- id: research_analysis
  desc: "evidence-based interpretation of reconstruction and physical-task trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradictory analysis. Implementation/evidence: records cannot support claims."}
    - {score: 0.4, desc: "Report: repeats bytes or errors without phase mechanism or limitations. Implementation/evidence: insufficient evidence."}
    - {score: 0.7, desc: "Report: replayed traces explain phase, age, loss, and intervention trends and state limitations. Implementation/evidence: results support the analysis."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies freshness and budget sensitivities, tests a failure boundary, and limits scope. Implementation/evidence: reproducible records support conclusions."}
