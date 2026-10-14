---
id: T033_intermittent_exploration_rendezvous
name: Intermittent Rendezvous Scheduling for Communication-Constrained Exploration
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

Four mobile robots explore a facility while normally operating beyond direct map-sharing range. Each robot builds a local cell map from nearby observations. Short control beacons let robots submit matching rendezvous proposals, but actual map summaries can be exchanged only when the proposed members physically gather during the accepted window. Meeting too early sacrifices exploration; meeting too late leaves valuable regions known to too few robots before their deadlines. Poorly chosen locations also cause travel, waiting, and safety interventions, while the limited summary capacity can waste a successful rendezvous on stale or redundant cells.

### Research Problem

Design one deterministic, state-driven method with a per-robot proposal role and a per-robot action role. In every stage, each robot may independently propose one member set, rendezvous point, and three-slot window from its available information. During every slot it selects an optional local frontier intent, one motion proposal, and up to four locally known cells to summarize if it belongs to a pending rendezvous. The same reusable procedures must handle all canonical scenarios without scenario-specific rendezvous, routes, summary lists, or action tables. The method should obtain weighted unique coverage and timely multi-robot knowledge of valuable regions while controlling travel, waiting, safety, energy, stale summaries, rendezvous failures, and the shared map budget.

### Experimental Setting and Constraints

#### Canonical evaluation units

The canonical scenarios, in scenario-index order, are `long_corridor`, `branching_frontiers`, `rendezvous_bottleneck`, `late_frontier`, and `range_shift`. Each uses grid coordinates `x in {0,...,23}`, `y in {0,...,17}`, robots `0,...,3`, and `H=96` one-second action slots `t=0,...,95`. State boundaries are `0,...,96`; slot `t` maps boundary `t` to boundary `t+1`. Stages `g=0,1,2` have slot intervals `[32g,32g+32)`. Boundary `96` contains the movement, observation, rendezvous, map exchange, and energy effects caused by slot `95`, but no later action.

Cell `(x,y)` has integer ID `c=24*y+x`. A closed obstacle rectangle includes every integer cell in its inclusive coordinate ranges. Every non-obstacle cell is traversable and explorable. Six valuable regions `F0,...,F5` are the `2x2` blocks with lower-left corners `(2,2)`, `(2,13)`, `(8,2)`, `(8,13)`, `(16,2)`, and `(16,13)`. Every cell outside these blocks has unit value `1`; the four cells of region `Ff` each have the scenario-specific region value below. All starts, region cells, and candidate rendezvous points are traversable.

| scenario | starts `(r0,r1,r2,r3)` | closed obstacle rectangles | `(value,deadline boundary)` for `F0..F5` | rendezvous points, IDs `0..2` | stage ranges |
|---|---|---|---|---|---|
| `long_corridor` | `(1,1),(1,16),(22,1),(22,16)` | `[11,12]x[0,6]`, `[11,12]x[11,17]` | `(1,28),(1,34),(2,44),(2,52),(3,72),(4,88)` | `(6,9),(12,9),(18,9)` | `(5,5,5)` |
| `branching_frontiers` | `(1,2),(1,15),(22,2),(22,15)` | `[5,8]x[7,8]`, `[15,18]x[9,10]` | `(2,24),(1,36),(3,48),(2,60),(4,76),(3,92)` | `(5,5),(12,9),(19,12)` | `(5,4,4)` |
| `rendezvous_bottleneck` | `(2,1),(2,16),(21,1),(21,16)` | `[10,13]x[6,11]` | `(1,26),(2,38),(2,50),(3,62),(4,78),(4,94)` | `(8,5),(12,5),(16,12)` | `(4,4,3)` |
| `late_frontier` | `(1,1),(1,16),(22,1),(22,16)` | `[6,7]x[3,14]`, `[18,19]x[4,13]` | `(3,20),(3,28),(2,44),(2,56),(4,72),(4,90)` | `(4,9),(12,3),(20,9)` | `(5,5,5)` |
| `range_shift` | `(2,2),(2,15),(21,2),(21,15)` | `[10,11]x[4,13]` | `(1,30),(2,42),(2,54),(3,66),(4,82),(3,94)` | `(6,9),(12,5),(18,9)` | `(6,4,3)` |

The grid, scenario identity, complete table, obstacles, starts, regions, deadlines, candidate points, and stage communication ranges are public from boundary `0`. They are fixed simulator inputs, not scenario-specific action prescriptions.

#### Local maps, motion, safety, and energy

At boundary `0`, robot `i` observes every traversable cell within Manhattan distance `2` of its start. Its local map associates each known cell with the most recent observation timestamp. After slot `t` movement, it observes all traversable cells within distance `2` of its executed position and assigns those cells timestamp `t+1`. Re-observation refreshes that robot's timestamp. Global unique coverage is the union of local-map keys and counts each cell once.

Each robot proposes one of `(0,0),(1,0),(-1,0),(0,1),(0,-1)`. Proposals are processed in robot-ID order. A destination outside the grid, inside an obstacle, equal to an already accepted lower-ID destination, or equal to a still-current higher-ID position is replaced by stay. This also replaces both sides of a pairwise swap. Each replaced nonzero proposal adds one safety event. An executed nonzero move costs `1` energy and an executed stay costs `0.2`.

Every robot starts with `100` energy. Adoption of a rendezvous plan costs each member `1` energy, and every admitted source-cell map unit costs its source `0.5`. A proposal or action is illegal if the costs produced under the fixed admission and safety rules make any energy smaller than `-1e-6`; balances in `[-1e-6,0]` are treated as zero solely for floating-point comparison.

#### Rendezvous proposals and adoption

In stage `g`, slots `32g,...,32g+3` form its proposal window. A low-rate control channel carries only the submitted proposal tuple; it does not carry maps or robot state. Each robot submits at most one proposal for that stage, at one action-start boundary in this window, or explicitly submits no proposal. A proposal contains a set of `2..4` distinct members including the proposer, candidate point ID `0..2`, and a three-slot rendezvous window starting at integer `w in {32g+4,...,32g+29}`. Member order has no meaning.

At boundary `32g+4`, before that boundary's action, a proposal tuple `(members,point_id,w)` is complete exactly when every listed member submitted that same tuple; their proposal slots may differ. If multiple tuples are complete, the environment selects by: more members first; then the smaller latest proposal slot among the tuple's members; then the lexicographically sorted member tuple, point ID, and window start. At most one plan is adopted per stage. With no complete tuple, that stage has no rendezvous. The adopted plan is disclosed to its members and expires at the stage end.

During slots `[w,w+3)`, the plan succeeds at the first post-motion boundary where every member is within Manhattan distance `2` of the selected point and the maximum pairwise member distance is at most that stage's public communication range. If this never occurs, the plan is cancelled after the last window slot. A pending member contributes one wait slot when, at action start during the window, it is already within distance `2` of the point and submits stay.

#### Map summaries and fusion

An action may select at most four distinct `summary_cells`. A nonempty selection is legal only for a member of the current pending plan during its three-slot window, and every selected cell must be present in that source's local map at action start. Selections produce no transmission unless the rendezvous first succeeds after that slot's movement. On that success, every selected source-cell pair from the successful slot is a broadcast map unit carrying the source's action-start timestamp for that cell.

Units are ordered by newer timestamp first, then smaller source ID, then smaller cell ID. The first eight, and at most the remaining part of the scenario-wide `20`-unit budget, are admitted. Dropped units consume no energy and do not update maps. Each admitted unit is sent to every other plan member. Recipients retain the maximum timestamp for each cell. A unit is useful if processing it strictly increases at least one recipient's timestamp; simultaneous units are processed in the fixed order. Admitted redundant units still consume budget and energy.

Region `Ff` has merge-completion boundary `m_f` equal to the earliest boundary at which at least three robots' current local maps each contain all four region cells. A single robot locally observing the region does not by itself complete this multi-robot information objective.

#### Information availability and event order

Robot `i` may use the public fixed inputs; its own current position and energy; its own action and proposal history; its current local map, including maps previously fused at successful rendezvous; and an adopted plan only when `i` is a member. At a past successful rendezvous it may also retain the then-observed member positions. It may not use another robot's live position or energy, an unmerged map, another robot's proposal state, current intent or summary selection, a future action, or a future rendezvous result. A proposal must be formed from the proposer's information at its declared proposal slot. The complete environment state is available only after all actions are submitted, for fixed safety execution, sensing, rendezvous, fusion, and replay.

For slot `t`, the authoritative order is: at a proposal-adoption boundary, normalize and select the current stage's complete plan and charge member plan energy; expose each robot's allowed action-start view; read all actions; validate frontier intents, moves, and summaries; apply the ID-ordered safety rule and charge move/stay energy; execute movement; observe at boundary `t+1`; test the pending rendezvous geometry; on its first success, order and admit action-start map units, charge source energy, and merge timestamps; update region completions, unique coverage, waits, safety, energy, and boundary histories; and cancel an unsuccessful plan after its final window slot.

Proposal and action fields are candidate requests made at their declared decision boundaries. Local maps, executed movement, plan adoption, rendezvous outcome, map-unit timestamps/admission/ usefulness, region completion, coverage, waits, safety, energy, and metrics are post-event facts derived by the grader. The grader checks whether submitted fields are consistent with the public model; the report and implementation remain the evidence that the policy actually respected its private information boundary.

#### Legal actions and boundary conditions

At action start, `intent_cell` is null or a local frontier: an unknown traversable cell with at least one known traversable four-neighbor in that robot's local map. An intent does not force a particular move. Move and summary legality use action-start state, so cells newly observed after the same slot cannot be summarized in that slot. A completed or cancelled plan accepts no later summaries. Capacity-dropped units and safety-replaced moves are legal environment outcomes.

For a no-proposal row, `proposal_slot`, `point_id`, and `window_start` are null and `members=[]`. JSON null and integer `-1` are equivalent only for those three fields and `intent_cell`. For a real proposal the three fields are nonnegative integers satisfying the stage ranges. Invalid proposal members, IDs, slots, windows, moves, frontier intents, summary arrays, unknown summary cells, timing, or insufficient energy under the declared `1e-6` tolerance close the validity gate.

#### Task metrics

Let `K_s` be the traversable cells in scenario `s`, `v_s(c)` the public cell value, and `C_s` the union of final local-map keys. Then

`primary_metrics_1=(1/5)*sum_s [sum_(c in C_s) v_s(c) / sum_(c in K_s) v_s(c)]`.

For region `f` with value `v_s,f`, deadline `d_s,f`, and merge-completion `m_s,f`, define quality `q_s,f=1` if `m_s,f<=d_s,f`, `q_s,f=0.5*clip(1-(m_s,f-d_s,f)/max(1,96-d_s,f),0,1)` if `d_s,f<m_s,f<=96`, and `0` if it never completes. Then

`primary_metrics_2=(1/5)*sum_s [sum_f v_s,f*q_s,f / sum_f v_s,f]`.

If `W_s`, `S_s`, and `E_s` are total wait slots, safety events, and energy used, define

`primary_metrics_3=(1/5)*sum_s [0.35*clip(1-W_s/(96*4),0,1)+0.35*clip(1-S_s/(96*4),0,1)+0.30*clip(1-E_s/(4*100),0,1)]`.

Let `A_s` be admitted source-cell units, `U_s` useful admitted units, and `D_s` the number of distinct `(source,stage)` pairs represented by useful units. There are `4*3=12` possible pairs. Set `e_s=0` when `A_s=0`; otherwise

`e_s=clip(D_s/12,0,1)*(0.6+0.4*clip(U_s/A_s,0,1))`.

Then `auxiliary_metrics_1=(1/5)*sum_s e_s`.

For freshness, consider the fixed 24 region cells. For each robot, region cell, and boundary `b=1,...,96`, let `q` be the greatest timestamp that robot has received for that cell through an admitted broadcast from another robot by boundary `b`. The sample is `clip(1-(b-q)/32,0,1)` when such `q` exists and `0` otherwise. Let `fresh_s` be the mean over the fixed `4*24*96` samples and `R_s` the number of stages with a successful rendezvous. Define

`auxiliary_metrics_2=(1/5)*sum_s [0.7*fresh_s+0.3*clip(R_s/3,0,1)]`.

All metrics are independently replayed and clipped to `[0,1]`. Normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Result record contract

`results.json` contains exactly `{"scenarios":[...]}` with one record per canonical scenario. Each record has exactly `scenario`, `proposals`, and `actions`. `proposals` has exactly 12 rows uniquely keyed by `(stage,robot)` for `stage in {0,1,2}`, `robot in {0,1,2,3}`. Every proposal row has exactly `stage,robot,proposal_slot,members,point_id,window_start`. `members` is a JSON array of integer robot IDs. The other fields are JSON integers or, for the four declared optional ID fields, null as allowed above.

`actions` has exactly `96*4` rows uniquely keyed by `(t,robot)`. Every action row has exactly `t,robot,dx,dy,intent_cell,summary_cells`. The first four fields are JSON integers, `intent_cell` is a JSON integer or null, and `summary_cells` is a JSON array of integers. Scenario, proposal, action, member, and summary-cell array order has no semantic meaning. All fields are required and all JSON numbers must be finite. No position, map, timestamp, executed move, energy, plan, rendezvous, exchange, region-completion, wait, safety, coverage, or metric summary is submitted; the grader derives these facts from the fixed model, so equivalent correct simulators are not rejected for using another redundant state representation.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain non-empty sections titled `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Non-canonical controlled experiments may be generated by executable project code and summarized in the report or supplementary files, but they must not be inserted into canonical `results.json`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact root and scenario containers are present", weight: 3}
- {id: result_evidence, desc: "minimal proposal and action fields are typed, finite, and auditable", weight: 4}
- {id: canonical_coverage, desc: "all scenarios, stage-robot proposals, and slot-robot actions occur exactly once", weight: 4}
- {id: decision_legality, desc: "proposals, moves, frontier intents, summaries, timing, and energy are legal", weight: 4}
- {id: replay_validity, desc: "motion, sensing, rendezvous, fusion, region, resource, and metric replay is complete", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed weighted unique exploration coverage", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed timely multi-robot region knowledge", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed waiting, safety, and energy quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed useful map-unit source-stage coverage and efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed received-map freshness and rendezvous success", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the local-map exploration, rendezvous, fusion, safety, and energy model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent local-map rendezvous problem is formulated; Implementation/evidence: no executable exploration and map-fusion process is present."}
    - {score: 0.4, desc: "Report: exploration or communication is partial and omits a decisive proposal, window, map-version, capacity, deadline, or information relation; Implementation/evidence: the implemented state or event order materially departs from the public model."}
    - {score: 0.7, desc: "Report: local sensing, proposals, rendezvous geometry, map units, fusion, motion, resources, and objectives are accurate; Implementation/evidence: the core model runs across all scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: event order, information boundaries, deadline knowledge, summary admission, and exploration-wait trade-offs are precise; Implementation/evidence: code and records faithfully realize every principal relation without material divergence."}
- id: method_design
  desc: "quality of the unified local proposal, rendezvous, frontier, motion, and map-summary policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method, a prohibited training method, or a policy using other robots' unavailable live state; Implementation/evidence: decisions lack a reusable procedure under the per-robot views."}
    - {score: 0.4, desc: "Report: fixed rendezvous timing or independent nearest-frontier motion ignores a decisive reachability, map-age, capacity, or waiting consequence; Implementation/evidence: decisions are prefilled, weakly state-driven, or materially inconsistent with the report."}
    - {score: 0.7, desc: "Report: one reproducible state-driven method handles proposals, window travel, frontier choice, summaries, and resources; Implementation/evidence: the same legal per-robot procedures generate every scenario with limited prediction or adaptation depth."}
    - {score: 1.0, desc: "Report: a justified method adapts members, point, timing, exploration, and summaries using reachability, marginal information, age, waiting, energy, and stage range; Implementation/evidence: evolving allowed state drives decisions without scenario, stage, cell, or slot action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for intermittent rendezvous and map-sharing effects"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution is present."}
    - {score: 0.4, desc: "Report: only canonical aggregates or one uninformative rendezvous setting is shown; Implementation/evidence: evidence does not test the central map-sharing mechanism."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled range, capacity, window, summary, or rendezvous-policy comparison are covered; Implementation/evidence: both conditions execute with comparable condition-level records."}
    - {score: 1.0, desc: "Report: complementary controlled experiments distinguish rendezvous timing or location from a range, capacity, freshness, or waiting failure boundary; Implementation/evidence: reproducible condition-level proposals, actions, exchanges, and metrics support both comparisons."}
- id: research_analysis
  desc: "evidence-grounded interpretation of coverage, region knowledge, waiting, freshness, and capacity"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records cannot support the conclusions."}
    - {score: 0.4, desc: "Report: values are listed without causal rendezvous, local-information, age, waiting, or capacity analysis or material limitations; Implementation/evidence: the asserted claims are unsupported."}
    - {score: 0.7, desc: "Report: actual records explain important scenario, window, exchange, deadline, and freshness trends and limitations; Implementation/evidence: observations agree with generated evidence."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies rendezvous-information mechanisms, tests the explanation at a contrasting condition or failure boundary, and states scope; Implementation/evidence: reproducible condition-level evidence supports the conclusions."}
