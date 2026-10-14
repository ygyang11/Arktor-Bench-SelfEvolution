---
id: T002_edge_qos_task_placement
name: Closed-Loop QoS-Aware Perception Placement under Persistent Resource Pressure
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

An industrial work cell uses two cooperating robot nodes and a nearby edge node to run a camera-driven visual-servoing loop. Robot R1 acquires image frames and can execute perception locally. Robot R2 applies the pose or joint command produced by the planning and control stages. An edge node E can execute perception as the controller-selected offload destination, relieving R1 load. A placement decision therefore changes both the wireless path and the compute state of the closed loop.

When `edge` is selected, the effective data path is R1 → E over the one-way wireless link (with delay `n[k]`), and E's perception output is consumed by the planning/control stage that supplies R2's pose or joint command; under `local`, R1 supplies that stage without the R1 → E hop. The command takes effect for the cycle only when its perception result reaches planning before the deadline.

The loop is periodic. A perception result that reaches planning after the cycle deadline cannot be used for that cycle, so that cycle proceeds without a new perception update and is evaluated as a deadline miss when the latency limit is exceeded. Local execution avoids the offload path but consumes R1 capacity; edge execution can reduce R1 load while incurring network delay, edge contention and a migration cost when placement changes. Work that is not completed in one cycle remains as backlog and affects later cycles. The operating condition may change during a run, so a useful placement controller must account for both current QoS and the state inherited from earlier windows.

### Research Problem

Develop a self-contained simulation and a non-training placement controller for this closed-loop multi-robot task. At the beginning of each four-cycle window, choose one placement for perception: `local` on R1 or `edge` on E. The placement remains active for all four cycles in that window. The controller must manage the backlog and placement history carried from earlier windows while balancing deadline compliance, latency, resource safety and switching stability.

The research question is how a single placement procedure should respond to changing robot pressure, wireless delay and edge contention when a locally or remotely accumulated workload affects subsequent cycles. You may use analytical planning, finite-horizon search, optimization, thresholds, hysteresis or another deterministic method. The model, policy, numerical procedure and project organization are your decisions, subject to the fixed setting and result contract below.

Use one state-driven, reusable procedure across all canonical scenarios. Derive every window decision from the scenario inputs, current backlog, previous placement and relevant history; do not provide scenario-specific placement tables or branches keyed to scenario names and window indices.

### Experimental Setting and Constraints

#### Canonical evaluation units and operating conditions

There are five canonical scenarios. Each has 16 windows indexed by `k = 0,...,15`; every window contains four cycles indexed by `c = 0,...,3`, with global cycle `i = 4k+c`. The following window-aligned sequences are fixed inputs. `r[k]` is robot-side pressure in percentage points, `n[k]` is one-way wireless delay for edge placement in milliseconds, `e[k]` is edge-side pressure in percentage points, and `q[k]` is the total perception workload assigned during window `k`.

| scenario | `r[0..15]` | `n[0..15]` (ms) | `e[0..15]` | `q[0..15]` |
|---|---|---|---|---|
| `nominal` | `12` repeated 16 times | `2` repeated 16 times | `14` repeated 16 times | `1.86,2.48,1.705,2.79,2.17,2.635,1.55,2.945,2.015,2.48,1.705,2.79,2.325,1.86,2.635,2.17` |
| `robot_stress` | `12,12,12,12,58,58,58,58,28,28,28,28,50,50,50,50` | `2` repeated 16 times | `14` repeated 16 times | same `q` sequence |
| `network_fault` | `12,12,12,12,20,20,20,20,36,36,36,36,18,18,18,18` | `2,2,2,2,24,24,24,24,8,8,8,8,20,20,20,20` | `14` repeated 16 times | same `q` sequence |
| `edge_contention` | `12,12,12,12,22,22,22,22,18,18,18,18,34,34,34,34` | `2` repeated 16 times | `14,14,14,14,58,58,58,58,25,25,25,25,48,48,48,48` | same `q` sequence |
| `alternating_conditions` | `55,55,16,16` repeated four times | `2,2,17,17` repeated four times | `14` repeated 16 times | same `q` sequence |

All profiles are deterministic and available to the simulation. No random samples, external data or assets are used.

#### State, action and task model

At a window boundary, the persistent task state is the pair of local and edge backlogs together with the placement active in the preceding window. The controller's action is the window placement `m[k]`, which is held constant for that window; its legal values and the effect of changing it are specified under `Legal actions and boundary conditions` below.

Each canonical scenario is an independent episode. Immediately before window `0`, `b_R = b_E = 0` workload units and the preceding placement is `local`; no backlog, placement or history carries between scenario records. Within an episode, both backlog queues persist across all windows and cycles as specified below.

At the start of each cycle, the selected placement releases `q[k]/4` workload units to its queue. Let `b_R^-` and `b_E^-` be the local and edge backlog in workload units immediately after this release and before service. During that cycle both queues are serviced. Their service capacities, in workload units per cycle, are

```text
cap_R = max(0.25, 0.53 - 0.003*r[k])
cap_E = max(0.30, 0.72 - 0.004*e[k])
```

The selected queue contributes a waiting term `7*b^-` milliseconds, using the pre-service backlog, where `b^-` means `b_R^-` for `local` and `b_E^-` for `edge`. Latency for the cycle is

```text
L = 16 + 0.32*r[k] + 7*b_R^- + j[c] + migration_penalty       if m[k] = local
L = 20 + n[k] + 0.25*e[k] + 7*b_E^- + j[c] + migration_penalty if m[k] = edge
```

where `j = [0,2,1,3]` ms and `migration_penalty` is `12` only for `c=0` in a changed window. After latency is evaluated, service both queues and define the post-service backlogs `b_R^+ := max(0,b_R^- - cap_R)` and `b_E^+ := max(0,b_E^- - cap_E)`.

The CPU utilizations recorded for the cycle are

```text
R1 = 35 + r[k] + 12*b_R^+ + migration_overhead       if m[k] = local
E  = 28 + 0.10*e[k] + 6*b_E^+ + migration_overhead  if m[k] = local
R1 = 25 + 0.12*r[k] + 6*b_R^+ + migration_overhead  if m[k] = edge
E  = 38 + e[k] + 12*b_E^+ + migration_overhead       if m[k] = edge
```

Here `migration_overhead` is `6` only for `c=0` in a changed window.

#### Information availability and event order

Before each canonical scenario's independent episode, its complete fixed profile is disclosed, including all listed pressure, delay and workload values; future entries of these deterministic sequences are therefore available inputs rather than hidden observations. At each window boundary, the procedure may use the current profile values, the disclosed values for later windows, the exact backlogs produced by its earlier decisions, the preceding placement and any history it has recorded. It may not use post-service values from the current cycle before that cycle is processed, unlisted measurements, random samples or values reconstructed from the grader's output fields.

Within each cycle, the event order is: select the window placement, apply migration state, release the cycle workload, compute latency from the pre-service backlog, service both queues, compute CPU values from the post-service backlog, then proceed to the next cycle. The full backlog state is carried across windows. In the result records, window and cycle placement fields are the decisions made at the corresponding boundary, while latency, CPU, migration, deadline and safety fields are evidence observed or produced after the stated event sequence; the detailed sequence is authoritative.

#### Legal actions and boundary conditions

The initial placement immediately before window `0` is `local`. Together with the zero backlogs specified above, these initial conditions apply independently to every canonical scenario; no state is carried between scenario records. The only placement actions are `m[k] ∈ {local, edge}` at `k = 0,...,15`; a placement cannot change within a four-cycle window. A change from the preceding placement creates a migration workload of `0.65` workload units on the destination queue and adds `12` ms to, and `6` percentage points to each CPU utilization during, the first cycle of that window. A cycle meets its control deadline when `L <= 42` ms and is resource-safe when both CPU utilizations are at most `90` percent. There are no actions after window `15`; the episode ends after cycle `63`, and any residual backlog is not scored. All indices, finite numeric fields and replay comparisons use the declared ranges and an absolute tolerance of `1e-6`.

#### Task metrics

Independent replay over all five scenarios and 320 cycles computes:

* `deadline_compliance`: fraction of cycles with no deadline miss;
* `latency_quality`: mean of `clip((100-L)/85,0,1)` over all cycles;
* `resource_safety`: fraction of cycles marked resource-safe;
* `placement_stability`: `clip(1-n_switch/80,0,1)`, where `n_switch` counts changes from the preceding placement, including the initial `local` placement.

Here `clip(x,0,1)=min(1,max(0,x))`; higher values are better.

#### Result contract

`python run.py` must create or overwrite a UTF-8 `results.json` in the project root. Its root is an object with the sole key `scenarios`, whose array contains one record for each canonical scenario. Each record has exactly `scenario`, `windows` and `cycles`.

`windows` contains exactly 16 records `{ "window": integer, "placement": "local" | "edge" }` for `window=0,...,15`. `cycles` contains exactly 64 records for `cycle=0,...,63`. Each cycle record has `cycle`, `window`, `placement`, finite numeric `latency_ms`, `robot_cpu_pct`, `edge_cpu_pct` and `switch_penalty_ms`, and boolean `deadline_miss` and `resource_safe`. The cycle's `window` equals `cycle // 4`, and its placement equals its window record. Record order is not significant; every canonical key occurs exactly once. Numeric comparisons use an absolute tolerance of `1e-6`.

Cycle and window decisions are the input to independent replay. The recorded QoS fields are evidence; the grader derives backlog, migration effects, deadlines, resource outcomes and all metrics from the placement decisions and fixed model.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The root of your project must contain `run.py`, `report.md` and `results.json`. You may organize the implementation across multiple files; `run.py` is the only official entry point and should start the simulation without interactive input. `report.md` must contain six substantive Markdown sections with these exact headings: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`. The report must describe the actual model, placement method, experiments, recorded QoS results, resource-switching trade-off and limitations supported by the implementation.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "the newly generated results.json parses as strict JSON and matches the declared root and scenario structure", weight: 3}
- {id: result_evidence, desc: "each scenario provides typed window decisions and cycle-level QoS and resource evidence required by the result contract", weight: 4}
- {id: canonical_coverage, desc: "the five canonical scenarios and every declared window and cycle key occur exactly once", weight: 4}
- {id: decision_legality, desc: "window placements and cycle assignments satisfy the declared domain and residency constraints", weight: 4}
- {id: replay_validity, desc: "recorded latency, resource, migration and outcome fields match the independent backlog replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline compliance", weight: 9}
- {id: primary_metrics_2, desc: "independently replayed normalized latency quality", weight: 6}
- {id: auxiliary_metrics_1, desc: "independently replayed resource safety rate", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed placement stability", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the persistent-backlog placement, QoS, resource and closed-loop deadline model in the report and implementation"
  weight: 8
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation is given, or the stated placement, workload, backlog, QoS or deadline model is materially wrong; Implementation/evidence: no executable representation or recorded evidence corresponds to the claimed process."}
    - {score: 0.4, desc: "Report: only part of the placement, backlog, latency, resource, migration or closed-loop model is defined and a decisive relation is omitted; Implementation/evidence: the code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: the principal states, placement actions, information boundary, event order, profiles, constraints, objectives and persistent cross-window relations are accurately defined; Implementation/evidence: the code and cycles instantiate that core model across all canonical scenarios, with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely specifies workload release, queue service, migration, latency, resource, deadline and closed-loop trade-offs; Implementation/evidence: the code and records faithfully realize these relations and link placement decisions to outcomes without material divergence."}

- id: method_design
  desc: "quality and task relevance of the non-training policy for stateful placement decisions"
  weight: 15
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no executable placement method is specified, a prohibited training method is used, or the method conflicts with the fixed information, state transition or placement constraints; Implementation/evidence: no runnable procedure or supporting decision record is present."}
    - {score: 0.4, desc: "Report: the method is partial, treats windows independently, ignores a decisive backlog, pressure, delay or switching trade-off, or presents scenario- or window-specific schedules as the primary solution; Implementation/evidence: the procedure relies on pre-filled actions or materially differs from the report. A fixed schedule explicitly used as a comparison is assessed under experimental coverage."}
    - {score: 0.7, desc: "Report: a legal, reproducible and reusable state-driven policy derives each placement from current inputs, persistent backlog and placement history while addressing deadline, resource and switching trade-offs; Implementation/evidence: the same procedure generates decisions across all canonical scenarios and the records support its operation, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified and reproducible procedure accounts for queue accumulation and recovery, migration, changing pressure or delay, and downstream deadline and resource effects; Implementation/evidence: the implementation derives decisions from input and evolving state without scenario-specific action tables, with records supporting the decisive choices."}

- id: experimental_coverage
  desc: "coverage and diagnostic value of experiments for persistent placement and QoS trade-offs"
  weight: 20
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose is provided; Implementation/evidence: the code and records contain no checkable experimental execution."}
    - {score: 0.4, desc: "Report: the work shows only isolated windows, a single fixed placement or incomplete canonical coverage and does not examine persistent state or QoS-resource coupling; Implementation/evidence: the recorded outputs do not execute or support the principal experimental claims."}
    - {score: 0.7, desc: "Report: all five canonical scenarios are covered and at least one explicitly identified controlled comparison or condition-specific experiment beyond merely contrasting canonical aggregates directly tests a central mechanism or trade-off among backlog, delay, resource safety, deadlines and switching; Implementation/evidence: both sides or conditions are actually executed with comparable inputs and records that support the stated comparison."}
    - {score: 1.0, desc: "Report: complementary, non-redundant experiments cover at least two meaningful controlled state or QoS conditions or decision regimes, including a mechanism-focused comparison and a sensitivity or failure-boundary check; Implementation/evidence: they are executed with condition-level decisions, outcomes and metrics, with queue-state evidence reproducible from the decisions and fixed model, distinguishing the claims beyond aggregate canonical scores."}

- id: research_analysis
  desc: "evidence-grounded interpretation of stateful placement behavior and limitations"
  weight: 12
  artifacts: []
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the recorded cycles, or presents unimplemented experiments as facts; Implementation/evidence: the code and records cannot support the reported observations or conclusions."}
    - {score: 0.4, desc: "Report: the discussion mainly lists QoS values or makes unsupported placement claims and does not explain how backlog, migration or persistent state changes later outcomes or identify material limits; Implementation/evidence: the records do not support the asserted analysis."}
    - {score: 0.7, desc: "Report: conclusions are grounded in actual records, explain important condition-dependent placement, backlog and QoS trends, distinguish observation from inference and state material limitations; Implementation/evidence: the implementation, cycles and metrics support those observations and explanations."}
    - {score: 1.0, desc: "Report: the analysis uses controlled evidence to explain how queue state, migration, pressure or delay and placement shape outcomes, quantify the principal trade-offs or sensitivities, test the explanation against contrasting evidence or a failure boundary, and state scope and unknowns; Implementation/evidence: reproducible condition-level decisions and metrics, with queue-state trajectories reconstructible from the fixed model, support the central conclusions."}
