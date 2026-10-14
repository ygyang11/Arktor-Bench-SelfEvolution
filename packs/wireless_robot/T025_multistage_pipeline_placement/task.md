---
id: T025_multistage_pipeline_placement
name: DAG Placement Control for Deadline-Sensitive Edge-Robot Pipelines
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

Two mobile robots run a four-stage sensing--planning--control pipeline while sharing one edge server. Sensing and control execute on the corresponding robot; perception and planning may be kept local or sent to the edge. A placement therefore changes both computation queues and wireless transfers between stages. Workload bursts and link jitter can make a control result arrive after its period, so the robot continues with its previous command. The robots must navigate around fixed rectangular obstacles to their goals, making pipeline placement a closed-loop robotics problem.

### Research Problem

Design one deterministic, non-learning online procedure for all five canonical scenarios. At each four-cycle window, choose the local/edge placement of the perception and planning stages for each robot. At each cycle, choose the bounded non-holonomic command that is active for that cycle, using only information available at its start. The procedure should balance deadline-compliant control, goal completion, cross-node transfer freshness, queue contention, and placement changes. You may choose the controller, placement policy, prediction horizon, parameters, and controlled experiments, but the fixed model and result contract below cannot be changed.

### Experimental Setting and Constraints

#### Canonical Evaluation Units

There are five canonical scenarios. Each has two robots, a `20 x 12` metre workspace, `48` control cycles `t=0,...,47`, cycle duration `0.5` seconds, and windows `w=0,...,11` of four cycles. The table fixes starts `(x,y,theta)`, goals, inclusive goal deadlines, and closed rectangular obstacles. Angles are radians and distances are metres.

| scenario | starts (robot 0, robot 1) | goals (robot 0, robot 1) | deadlines | obstacles |
|---|---|---|---|---|
| `balanced_pipeline` | `(1,1,0),(1,9,0)` | `(18,10),(18,2)` | `42,42` | `[8,0,10,4]`, `[8,8,10,12]` |
| `robot_cpu_burst` | `(1,2,0),(1,10,0)` | `(18,10),(18,2)` | `44,44` | `[9,0,11,3]`, `[9,9,11,12]` |
| `edge_queue_burst` | `(1,1,0),(1,9,0)` | `(18,11),(18,1)` | `43,43` | `[7,4,13,6]` |
| `link_jitter` | `(1,2,0),(1,10,0)` | `(18,9),(18,3)` | `45,45` | `[6,0,8,5]`, `[6,7,8,12]` |
| `deadline_race` | `(1,1,0),(1,9,0)` | `(18,10),(18,2)` | `38,38` | `[8,0,10,4]`, `[8,8,10,12]` |

#### State, Actions, and Task Model

Each cycle releases one instance of the fixed DAG `Sensing -> Perception -> Planning -> Control` for each robot at time `t*0.5`. Sensing and Control are pinned to that robot's node. Perception and Planning independently use `local` (the robot node) or `edge`; their choice is constant for the four cycles in a window. Before window zero, both movable stages are local. The window-zero choice is still an explicit decision; selecting edge there incurs the migration cost below. A later change is compared with the immediately preceding window. Every stage of a pipeline instance retains the placement selected in that instance's release window, including a successor that becomes ready after a later window begins. Migration cost is likewise determined by the instance's release window, not its eventual service-start time.

The workload in work units is fixed by scene index `q=0,...,4` in the table's listed order, robot `r`, cycle `t`, and stage `s`:

```text
base = {Sensing:0.10, Perception:0.14, Planning:0.11, Control:0.09}[s]
work = base + 0.02*((q + 2*r + t) mod 3)
robot_cpu_burst and r=0 and 12<=t<=23: add 0.36 to Sensing and 0.18 to Control
edge_queue_burst and s in {Perception,Planning} and 8<=t<=19: add 0.32
deadline_race and t in {8,...,19} and s in {Perception,Planning}: add 0.14
```

Each robot node and the edge node has a non-preemptive FIFO service queue with capacity `2` waiting jobs. Service rates are `1.2` work units/second on a robot node and `2.4` on the edge. When a stage becomes ready, two jobs that precede it under the ordering below and are still waiting at its selected node cause that stage to be dropped; all of its descendants are dropped. Otherwise its service starts at the later of its ready time and node availability. Its ordinary service duration is `work/service_rate`. For the first instance of a movable stage in a window whose placement differs from the previous window, `0.18` seconds is added to that duration; this migration work is processed first and occupies the selected node. Stage jobs at a node are ordered by ready time, stage topological order, robot ID, and cycle ID. The model allows instances from different cycles to overlap.

When a predecessor and successor use different nodes, a `256`-bit transfer is required. At most `1,024` bits can start in one cycle. Transfers are ordered by ready time, robot ID, predecessor cycle, and successor stage order. Starting with the first slot boundary no earlier than predecessor completion, a transfer uses the first cycle whose bit budget is not full. It starts at `slot*0.5`, and its arrival is that time plus the link delay for `slot`. Its communication wait is arrival minus predecessor completion. Same-node dependencies have zero communication wait. A transfer arriving after the final time `48*0.5` is not delivered for the auxiliary metric, although the fixed stage timing remains deterministic.

The link state for a transfer starting in cycle `t` is `good` with delay `0.05` seconds or `jitter` with delay `0.20` seconds. It is `good` for `balanced_pipeline`; for `robot_cpu_burst` it is `good` when `t mod 6 < 4`; for `edge_queue_burst` it is `good` when `t < 12` or `t mod 8 < 3`; for `link_jitter` it is `jitter` when `t mod 10` is `3`, `4`, or `5`; and for `deadline_race` it is `jitter` for `10<=t<=21`. It is otherwise `jitter`. The public workload and link formulas are deterministic and may be evaluated for planning. The future execution consequences of those formulas—queue admission, completion times, transfer outcomes, and robot state—are not available before they occur. The current link state at a decision boundary is public; future link states may be computed from the public formula, but future transfer outcomes may not be used before arrival.

At cycle `t`, the command is `[v,omega]`, with `0<=v<=1.2` metres/second and `-1.0<=omega<=1.0` radians/second. A robot's command may differ from that robot's previous active command only when at least one Control stage of that same robot from an earlier cycle newly completes in the boundary interval `((t-1)*0.5,t*0.5]`. A completion at the left endpoint was already visible at the preceding decision and does not authorise another change; several completions in one interval authorise only the current boundary. A Control completion whose latency exceeds `0.8` seconds is late for `primary_metrics_1`, but it may still authorise a change only at the boundary where it is first exposed. If no completion is newly exposed, the command must equal the previous active command; an unused opportunity is not carried to a later boundary. The initial command is `[0,0]`, and no earlier Control stage exists at `t=0`. The command is held for the whole cycle and updates `(x,y,theta)` by `x'=x+0.5*v*cos(theta)`, `y'=y+0.5*v*sin(theta)`, and `theta'=wrap(theta+0.5*omega)`, where `wrap` returns an angle in `[-pi,pi)`.

#### Information Availability and Event Order

At a window boundary, the controller may use all completed-stage observations, prior delays and queue lengths, node utilisation, the current public link state, target progress, remaining deadline, and the deterministic workload and link formulas. At a cycle boundary it may use the state and all events completed by that boundary. It cannot use a future queue admission, completion, transfer outcome, robot state, or post-motion safety result as an observation before that event occurs. Model-based predictions computed only from public formulas, current state, and already observed history are permitted, including deterministic lookahead under the candidate's own proposed placements and commands even when its prediction later equals the realised outcome. Such a prediction is not an early observation; the procedure must not condition a decision on an execution record labelled as observed before that record's event time. Placement is frozen for the current window. The same reusable procedure must generate every scenario and may not map a scenario name, absolute cycle, stage, or robot ID directly to a pre-filled placement or command schedule.

At boundary `t`, the fixed model first exposes all events completed no later than `t*0.5`, including the latency of each newly exposed Control completion. If this is a window boundary, the controller selects the placement used by that window; it then records the active command for cycle `t` under the new-completion rule above. The current DAG instances are released at `t*0.5` under that placement, and service and cross-node transfer events proceed asynchronously in the fixed queues. The active commands are executed for the cycle, after which one simultaneous safety test restores every unsafe robot to its complete previous pose and records the resulting state. Events caused by the current release or motion are therefore not visible to either current decision. Placement and command records are boundary decisions; stage timing, queue/resource values, control freshness, positions, safety, goals, and metrics are execution-after evidence derived by the fixed replay.

#### Legal Actions and Boundaries

Every placement value is exactly `local` or `edge`. A placement change is illegal when the same movable stage changed in the immediately preceding window; the residence rule therefore requires one unchanged window between changes. Commands must be finite and within the stated bounds. A command change without a newly exposed earlier-cycle Control completion of that same robot in the current boundary interval is illegal; a completion cannot authorise changes at later boundaries. The fixed model, not the candidate, determines queue admission, service order, transfer capacity, migration cost, stage drop, control completion, and all safety consequences. A proposed centre outside `[0,20] x [0,12]`, inside a closed obstacle, or with simultaneous pair distance below `0.50` metres is unsafe. Both proposed poses are tested simultaneously; every unsafe robot restores its complete prior pose and no second test is performed. A first goal entry within radius `0.4` is a successful timely arrival only when it is no later than that robot's deadline and the robot had no earlier safety violation. The state after executing cycle `t` is boundary `b=t+1`; goal-deadline comparisons use this post-cycle boundary index.

#### Task Metrics

All metrics are independently replayed, higher is better, and lie in `[0,1]`. For each scenario, `primary_metrics_1` is the fraction of the `2*48` released pipeline cycles whose Control stage completes within `0.8` seconds of release; the five scenario fractions are averaged equally. `primary_metrics_2` is the mean over scenarios of the fraction of robots that first enter their goal by deadline without any earlier safety violation.

For `auxiliary_metrics_1`, each generated cross-node predecessor transfer contributes `256` bits. For each of the `2*12=24` robot-window pairs, collect delivered transfers whose predecessor belongs to a pipeline cycle released in that window. A pair with no delivered transfer has value zero. For each delivered transfer, let `wait_cycles` be the smallest non-negative integer `k` for which complete arrival is no later than predecessor completion plus `k*0.5` seconds. A covered pair has value `0.6 + 0.4*mean(clip(1-wait_cycles/48,0,1))` over its delivered transfers. The scenario value is the mean of these 24 pair values, and the five scenario values are averaged equally. This fixed robot-window coverage denominator prevents a small number of transfers from representing sustained communication.

For `auxiliary_metrics_2`, `migrations` counts movable-stage changes at window boundaries, with normalisation upper bound `24`, and `generated_cross_node_bits` counts all generated transfer bits. The scenario value is `0.6*clip(1-migrations/24,0,1) + 0.4*clip(1-generated_cross_node_bits/(48*1024),0,1)`, averaged over scenarios. This metric does not repeat the deadline metric.

The formal task performance is `(8*primary_metrics_1 + 7*primary_metrics_2 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2)/20`.

#### Result Record Contract

`results.json` is strict JSON with only root key `scenarios`. It contains exactly one object for each scenario, with exact keys `scenario`, `placements`, and `commands`.

- `placements` contains exactly one object for each `(robot,window)` pair, hence `2*12` records per  scenario. Each object has exact keys `robot`, `window`, `perception`, and `planning`; the two  placement values are `local` or `edge`. The parent scenario object supplies the scenario name.
- `commands` contains exactly one object for each `(robot,cycle)` pair, hence `2*48` records per  scenario. Each object has exact keys `robot`, `cycle`, and `command`, where `command` is a finite  two-number list `[v,omega]` in the stated domain. The parent scenario object supplies the  scenario name. It denotes the active command used for that cycle, not a self-reported metric or  state summary.

Canonical coverage requires exactly one parent scenario object for each named scenario. Within each parent object, the decision keys are `(scenario,robot,window)` and `(scenario,robot,cycle)`; array order has no meaning. These are the complete minimal decision records. Do not submit stage, queue, transfer, resource, completion, position, safety, goal, or metric fields: the grader derives them from the public model. Duplicate object members, non-finite numbers, Boolean integers, duplicate keys, and missing or unknown canonical records are invalid. All numeric comparisons use absolute tolerance `1e-6`.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must generate or overwrite `results.json`. The report must contain non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Controlled experiments are optional for the exact `results.json` contract and Auto Checks, but a strong research report is expected to use at least one controlled single-factor comparison to test the central queue, communication, migration, or control relation; complementary mechanism and failure-boundary comparisons provide stronger evidence. Each comparison must vary exactly one listed factor between its two conditions; all other task conditions must remain equal. The listed factors are a public workload, link delay, service capacity, migration cost, or controller parameter. Reported comparisons must use the same five canonical scenarios and compute all four metrics under the fixed replay definitions. Supporting code, report tables, or separately named artifacts may document them, but they must not add fields to or overwrite the canonical `results.json`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official Python run finishes within 600 seconds and creates a new results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required substantive sections", weight: 2}
- {id: result_json, desc: "strict JSON has the declared scenario and top-level decision containers", weight: 3}
- {id: result_evidence, desc: "placement and command decision records have the declared types and domains", weight: 4}
- {id: canonical_coverage, desc: "each canonical scenario container and all scenario-window and scenario-cycle decision keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "placement residence and newly observed Control command-update relations are legal", weight: 4}
- {id: replay_validity, desc: "the fixed model independently replays DAG timing, communication, resources, motion, deadlines, and safety", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed Control completion within the public deadline", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed safe goal completion by deadline", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed robot-window communication coverage and transfer freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed placement stability and cross-node bit cost", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the DAG, placement, queue, communication, deadline, and robot model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent pipeline or robot formulation; Implementation/evidence: no executable DAG or placement process."}
    - {score: 0.4, desc: "Report: only part of stages, predecessors, queues, transfers, or deadlines is defined; Implementation/evidence: records cover only a partial model or materially diverge."}
    - {score: 0.7, desc: "Report: stages, placement choices, queue/service rules, link timing, control deadline, and navigation constraints are accurately formulated; Implementation/evidence: one procedure realizes the core model across all scenarios."}
    - {score: 1.0, desc: "Report: event order and the causal chain from migration and shared queues through Control freshness to robot motion are precise; Implementation/evidence: the reusable code and report derive those relations from the submitted placement and command decisions."}
- id: method_design
  desc: "quality of the unified online placement and navigation procedure"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no runnable method, prohibited training, or information violation; Implementation/evidence: no legal unified procedure."}
    - {score: 0.4, desc: "Report: fixed placement, current-delay-only switching, or scene tables ignore shared queues and migration; Implementation/evidence: decisions are not state-driven or materially differ from the report."}
    - {score: 0.7, desc: "Report: a legal reusable window method addresses queue, deadline, residence, and navigation coupling; Implementation/evidence: the same procedure covers all scenarios with limited prediction depth."}
    - {score: 1.0, desc: "Report: a justified state-driven method jointly handles DAG readiness, edge contention, link timing, migration cost, deadline slack, and robot progress; Implementation/evidence: mapping and commands are consistently generated from evolving observations without scenario action tables."}
- id: experimental_coverage
  desc: "coverage and causal value of placement, workload, and link experiments"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable canonical execution."}
    - {score: 0.4, desc: "Report: only aggregate canonical values or one isolated condition is reported; Implementation/evidence: no condition-level evidence tests queue/deadline consequences."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one strict single-factor workload, link, capacity, migration, or controller comparison are covered; Implementation/evidence: both conditions retain replayable decisions and report-derived pipeline, resource, motion, and metric evidence."}
    - {score: 1.0, desc: "Report: complementary single-factor comparisons test workload/edge contention and a failure boundary such as link delay or migration cost; Implementation/evidence: reusable condition artifacts and code derive stage/resource traces and distinguish the mechanisms without requiring extra result fields."}
- id: research_analysis
  desc: "evidence-grounded analysis of placement, timing, and navigation trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records cannot support the conclusions."}
    - {score: 0.4, desc: "Report: discussion mainly repeats deadline or placement numbers and omits queue/migration limits; Implementation/evidence: evidence does not support causal claims."}
    - {score: 0.7, desc: "Report: replay results explain conditional workload, link, placement, and navigation trends, distinguish observation from inference, and state limitations; Implementation/evidence: canonical and controlled outputs support the analysis."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies DAG/queue/migration interactions and tests them against contrasting failure boundaries with scope and unknowns; Implementation/evidence: reproducible condition-level records support each central conclusion."}
