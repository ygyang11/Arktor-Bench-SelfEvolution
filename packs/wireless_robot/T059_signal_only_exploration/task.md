---
id: T059_signal_only_exploration
name: Signal-Only Decentralized Exploration with Implicit Robot Coordination
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

Three ground robots explore an unknown indoor grid without exchanging maps, routes, task labels, or coverage summaries.  Each robot can inspect only a small local neighborhood and may occasionally send a low-rate wireless pilot to one other robot.  The pilot reveals a delayed, noisy range and several possible bearings; reflections and missed packets make relative positions uncertain.  The robots must choose exploration frontiers, avoid repeatedly covering the same space, and decide when to wait or resume after a teammate becomes unavailable.  A policy that silently uses another robot's map or the future signal outcome would have information that the real robots do not have.

### Research Problem

Design a deterministic, non-training, state-driven procedure that generates every robot's motion, pilot requests, relative-position estimates, frontier choice, and standby decision.  The same procedure must run over all five environments.  Your method should make signal timing, multipath ambiguity, estimate age and uncertainty, local frontier structure, asynchronous waiting, and the fixed robot failure affect exploration coverage, duplicate observations, safety, and pilot efficiency.  Explain and measure the trade-off between spending scarce pilots to coordinate and moving to discover new cells.

### Experimental Setting and Constraints

#### Canonical units and environment

Each canonical unit is one named scenario.  The grid is `x=0..23,y=0..19` with one metre cells and slots `t=0..179`; state boundaries are `0..180`.  Robots are `R0,R1,R2` and start at the positions in the table.  A listed rectangle uses inclusive integer bounds.  A cell is traversable unless the table or the rules below make it blocked.

| scenario | starts `(R0,R1,R2)` | permanent blocked cells | signal condition | fixed failure |
|---|---|---|---|---|
| `homogeneous_clutter` | `(1,1),(22,1),(12,18)` | `(6..7,3..14)` except all cells with `y=8,9`; `(15..16,5..16)` except all cells with `y=11,12` | low noise | none |
| `slow_robot` | `(1,2),(22,2),(12,18)` | `(9..10,0..7)` and `(9..10,10..19)`; cells `(9,8),(10,8),(9,9),(10,9)` form the open channel | low noise | none |
| `signal_multipath` | `(2,2),(21,2),(12,17)` | `(7..9,5..14)` and `(15..17,5..14)` | high noise at `40..119`, low noise otherwise | none |
| `robot_failure` | `(1,1),(22,1),(12,18)` | `(10..13,4..15)` except `(10,4),(13,4),(10,15),(13,15)` | medium noise | `R2` is failed from slot `85` onward |
| `narrow_partition` | `(1,9),(22,9),(12,18)` | every cell outside `y=8..11`, `x=10..13`, and rooms `[0,3]x[8,11]`, `[20,23]x[8,11]`, `[0,3]x[16,19]`, `[20,23]x[0,3]` | wall occlusion, medium noise | none |

The exception clauses in the table are exact set differences, not informal openings.  A robot's local map starts completely unknown in every scenario.  At the beginning of each slot, a live robot observes every cell in Chebyshev radius `2` of its current position.  A true traversable cell observed for the first time by any robot is a team discovery; observations by other robots are repeats.

#### State, frontiers, actions and energy

Each local map stores `unknown`, `traversable`, or `blocked`.  In `slow_robot`, `R1` may execute a non-`stay` action only on even-numbered slots; on odd-numbered slots its request is treated as `stay`, and that forced conversion is not a safety event.  A frontier cell is a known traversable cell with at least one unknown four-neighbour that is inside the grid; positions outside the grid are not cells and cannot create a frontier.  Frontier cells are grouped into eight-connected components.  A component ID is `F{x}_{y}`, where `(x,y)` is its lexicographically smallest cell.  Its candidate viewpoints are the cell nearest the component's coordinate mean and the two component cells with greatest Manhattan distance from that mean; ties are resolved by `(x,y)`.  Select the nearest by minimum `(Manhattan distance to the mean,x,y)` and the two farthest by the first two entries sorted by `(-Manhattan distance to the mean,x,y)`, then remove duplicates (so a one-cell component has one candidate and a two-cell component at most two).  A selected viewpoint must be one of those candidates and must be reachable through known traversable cells.  A component is reachable exactly when at least one of its candidate viewpoints is reachable by such a path.  A planning boundary is every slot `0,5,...,175`.  At that boundary a robot chooses one component and viewpoint, or sets `standby=true` and chooses both fields as `null`.  A choice made at boundary `t` is the active planning mode for slots `t..min(t+4,179)`, and a new choice is made at the next planning boundary.  An active standby mode requires the robot to request `stay` in every slot of that interval.  A non-standby frontier and viewpoint record is an auditable boundary decision; the Auto Checks do not prescribe a particular path or require that the component retain the same ID after later observations.  The grader checks that boundary decision against the local map reconstructed solely from the robot's own observations; arrived pilots may affect estimates, HGrid and choice utility but never the local map.

At every slot each live robot requests exactly one action from `stay,N,S,E,W`, with displacements `stay=(0,0), N=(0,+1), S=(0,-1), E=(+1,0), W=(-1,0)`.  A request outside the grid or into a true blocked cell is replaced by `stay`.  Two-way swaps are rejected.  If several requests target one cell, the winner is the robot with the smaller number of currently unknown cells in its radius-two target neighbourhood, then smaller robot ID.  A move into a cell occupied by a robot that remains still is rejected; repeat this rule until stable.  Every requested non-stay action that does not execute is one safety event.  A failed robot must request `stay` and cannot ping or be selected as a neighbour.

Each executed move costs `1` energy unit and each endpoint of an executed pilot costs `0.2`.  Energy may be reported as an additional diagnostic, but there is no energy budget and energy is not a scored metric.

#### Signal-only pilots and relative estimates

Every slot has one pilot opportunity.  `robot_a` may be a live sender and `robot_b` a different live receiver, or both may be `null` for an idle opportunity.  At most `90` non-idle opportunities are allowed in a scenario.  The pilot is exactly `64` bits and carries no map, frontier, action, pose, coverage, failure, or energy field.

For a requested pair at slot `t`, freeze both robots' positions after observation and planning but before either robot's slot-`t` motion.  All range, bearing and loss calculations for that pilot use this frozen position pair.  Let `d` be their true Euclidean distance.  The integer-rounded segment is the following fixed DDA sequence: with endpoint coordinates `a,b`, let `L=max(|b_x-a_x|,|b_y-a_y|)` and, for `k=0..L`, append `(floor(a_x+(b_x-a_x)k/max(1,L)+0.5), floor(a_y+(b_y-a_y)k/max(1,L)+0.5))`; remove repeated cells while preserving first occurrence.  Thus every half-integer is rounded upward, not ties-to-even.  Let `n_block` be the number of distinct blocked cells in this sequence (endpoints included).  If `d>14` or `n_block>2`, the pilot is lost.  Otherwise it succeeds iff `((17*i+11*j+5*t) mod 100)/100 >= 0.04 + 0.08*n_block + 0.12*noise_level`, where `i,j` are the numeric robot indices and `noise_level` is `2` in `signal_multipath` slots `40..119`, `1` in every medium-noise scenario/slot, and `0` in every low-noise slot.  A successful result arrives at both endpoints at the beginning of slot `t+1`, except that an endpoint already failed at that boundary discards its arrival; the other live endpoint still receives it.  The pilot counts as on-time when its physical transmission succeeds and at least one endpoint remains live at arrival.  Its range is `d+0.08*((7*i+3*j+t) mod 5-2)` and its direct bearing at the sender is `atan2(dy,dx)+0.03*((5*i+7*j+t) mod 5-2)`, normalized to `[-pi,pi)`.  Each blocked cell intersected by the segment also creates one reflected bearing candidate obtained by mirroring the direct bearing in that cell's nearest grid-wall normal.  For each blocked cell, use the wall normal of the first retained sequence transition entering it: vertical for a transition whose x coordinate changes and horizontal otherwise; if both coordinates change, use vertical.  With the normalized direct angle `phi`, the reflected angle is `-phi` for the vertical case and `pi-phi` for the horizontal case, normalized again to `[-pi,pi)`.  Sort candidates by decreasing strength `1/(1+d)`, then angle, then blocked-cell coordinate; the direct candidate precedes a reflected candidate if their other sort keys are identical, and each blocked cell contributes at most one candidate.  For each sender candidate `phi`, the receiver candidate is exactly `normalize(phi+pi)` with the same strength and originating blocked-cell identity. Re-sort the receiver candidates using the same complete key and the same direct-before-reflected tie rule.  A lost pilot creates no usable estimate.

At each planning boundary, each ordered pair may submit a relative estimate based only on successful measurements whose arrival boundary is no later than that planning slot.  The estimate is a signed displacement `(neighbor_x-observer_x, neighbor_y-observer_y)` in the fixed grid axes, not an absolute pose.  When used for HGrid, its estimated absolute neighbour position is the observer's current position at that planning boundary plus this displacement.  A valid estimate has finite coordinates, a symmetric positive-definite covariance `[[cov_xx,cov_xy],[cov_xy,cov_yy]]`, `0<=age<=12`, and one or more source IDs.  Its source IDs are distinct IDs of the observer's arrived successful measurements and its age equals the planning slot minus the greatest source ping slot.  An invalid estimate has `valid=false`, all five estimate and covariance fields and `age` equal to `null`, and an empty source list.  An estimate older than twelve slots must be invalid.  No estimate may reference a future, lost, or other robot's measurement.  In `robot_failure`, every estimate with `R2` as observer or neighbour at planning slots `85..175` must be invalid.

#### HGrid, waiting, failure and event order

Throughout this task, `clip(z,0,1)=min(1,max(0,z))`.  The HGrid has the same boundary and six-by-five `4x4` cells.  At a planning boundary a live robot's own exact position contributes weight `1`; its valid arrived estimate of a live neighbour contributes `clip(1-tr(Sigma)/8,0,1)` to the HGrid cell containing its estimated neighbour position.  For a continuous estimated position `(p_x,p_y)` inside the environment, that cell has indices `(floor(p_x/4),floor(p_y/4))`; an estimated position outside `0<=p_x<24,0<=p_y<20` contributes zero weight.  These contributions are cumulative over planning boundaries and weights accumulated before a failure are not erased.  A failed robot and every estimate involving it add no new HGrid weight.  A cell is filled once its cumulative weight reaches `3`.  For a reachable frontier component, its estimated overlap is the mean over component cells of `clip(hgrid_weight[cell]/3,0,1)`, using the HGrid cell containing each frontier cell.  A robot is eligible for standby when at least `65%` of its HGrid cells are filled and it has no reachable frontier whose estimated overlap is below `0.5`; it is also eligible when it has no reachable frontier component. Standby still performs observation and pilots and may be left when the applicable condition no longer holds.  In `robot_failure`, fixed failure is applied before all slot-`85` events: `R2` no longer observes, moves, pings, receives pilots, updates HGrid, or contributes a relative estimate.  Its required planning records at slots `85..175` have `standby=true` and null frontier and viewpoint; this fixed-failure record is exempt from the ordinary HGrid standby-eligibility test.  Cells first discovered only by the failed robot are unavailable for final usable coverage until a surviving robot observes them.

The exact slot order is: apply fixed failures; process measurements arriving at the slot and discard estimates older than twelve; each live robot observes locally; at a planning boundary evaluate coverage termination from that boundary's observed state, record the six estimates, apply their and the live robot's HGrid contributions, then record the frontier choice and standby flag; choose the pilot pair and freeze its pre-motion position pair; resolve and execute simultaneous motion; generate the next-slot pilot result from the frozen pair; then update discovery, energy, overlap and history.  Termination and the first `95%`-coverage time are evaluated only at boundaries `0,5,...,175,180`; the standby condition at a boundary uses the preceding twelve complete slot decisions.  Boundary `180` processes already scheduled arrivals and expiry, evaluates termination, and performs no observation or decision.  Decision fields (`intended_actions`, planning choices, pilot pair and estimates) describe requests or state available at that boundary.  Trajectories, maps, measurements, HGrid values, coverage, failure status, energy and termination are post-execution facts reconstructed by the grader.

`t_stop` is an evaluation marker rather than an instruction to truncate a canonical unit.  All 180 action and pilot records are replayed, all planning records remain required, and observation and state updates continue through slot `179` even when the marker was reached earlier.

At a decision boundary, a robot may use its own current grid position, local map and frontiers, action and target history, HGrid, measurements received at that endpoint, relative estimates maintained by that robot, the current liveness flags, and the remaining pilot budget.  The one shared opportunity is assigned by an abstract pilot scheduler, which may use the current slot, liveness flags, remaining budget, past selected pairs, and whether their endpoint measurements arrived, but may not inspect any range or bearing value, robot map, action, position, HGrid, relative estimate, or coverage state.  Once a pilot is assigned, its range and bearing candidates are private to its two endpoints.  The fixed maps and all robot positions are environment truth used to generate observations, motion outcomes and signal events; a decision procedure must not query unobserved obstacle cells, another robot's state or history, team coverage, a true relative position, a future signal outcome, or any future event.

#### Task metrics

All task metrics are calculated independently from the replay.  Let `N` be the number of true traversable cells in a scenario and let `C_t` be the usable discovered-cell count after slot `t`, where a cell observed only by `R2` before its failure is unusable until a surviving robot revisits it.  Define `final_usable_coverage=C_179/N` and `coverage_auc=(1/180)*sum_{t=0}^{179}(C_t/N)`.  Then `primary_metrics_1=0.7*final_usable_coverage+0.3*coverage_auc`.

For each robot `r`, let `F_r` be the set of true traversable cells personally observed by `r` while it is live, and let `U=union_r F_r`.  If `U` is non-empty, define `overlap_suppression=clip(1-((sum_r |F_r|/|U|)-1)/2,0,1)`; otherwise it is `0`.  Over the boundary set `B={0,5,...,175,180}`, `t_95` is the first boundary whose usable coverage is at least `0.95*N`, or `180` if none exists.  `t_stop` is the first boundary that reaches the same coverage threshold or at which every still-live robot was in standby during each of the preceding twelve complete slots; it is `180` if neither condition occurs.  Define `termination_quality=final_usable_coverage*clip(1-|t_stop-t_95|/180,0,1)`.  In `robot_failure`, let `O` contain the cells observed by `R2` but by neither survivor through slot `84`; `failure_recovery` is the fraction of `O` observed by `R0` or `R1` during slots `85..179`, and is `1` if `O` is empty.  It is also `1` in the other scenarios.  Then `primary_metrics_2=0.5*overlap_suppression+0.3*termination_quality+0.2*failure_recovery`.

Let `D=C_179`, let `M` be the number of executed non-stay moves, and let `S` be the number of requested non-stay actions that do not execute.  Let `A` be the number of live robot-slot decisions whose active planning choice is not standby; failed `R2` contributes neither active nor standby slots from slot `85` onward, while the normalization remains `3*180`.  Define `primary_metrics_3=0.45*clip(D/max(1,M),0,1)+0.3*clip(1-S/(3*180),0,1)` `+0.25*final_usable_coverage*clip(1-A/(3*180),0,1)`.

A non-idle pilot is decision-used exactly when at least one of its two successful, grader-derived measurement IDs is cited by a valid relative estimate.  Let `P` be the number of non-idle pilots, `P_u` the number of decision-used pilots, and `P_o` the number that succeed and arrive at the next slot. Both fractions are `0` when `P=0`, and `auxiliary_metrics_1=0.65*(P_u/P)+0.35*(P_o/P)` otherwise.

For every valid relative estimate, let `e` be its Euclidean displacement error against the replayed relative position at that planning boundary, let `a` be its age, and let `NIS=error_vector^T*Sigma^{-1}*error_vector`.  Average `clip(1-e/20,0,1)` and `clip(1-a/12,0,1)` over valid estimates to obtain `relative_position_quality` and `estimate_freshness`.  Let `coverage95` be the fraction with `NIS<=5.991` and let `mean_NIS` be their mean.  Define `uncertainty_calibration=0.5*clip(1-|coverage95-0.95|/0.25,0,1)` `+0.5*clip(1-|mean_NIS-2|/4,0,1)` and `auxiliary_metrics_2=0.45*relative_position_quality+0.3*estimate_freshness` `+0.25*uncertainty_calibration`.  If there is no valid estimate, all three components and this metric are `0`.

Each metric is first computed separately in each scenario and then averaged equally over the five scenarios.  All five metrics are clipped to `[0,1]`.  The normalized task performance used for difficulty calibration is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3` `+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`.

#### Result contract

The project must write strict UTF-8 JSON with root exactly `{"scenarios":[...]}`.  Each of the five scenario objects has exactly `scenario,robots,planning_opportunities,ping_opportunities,relative_estimates`. `robots` has one object per `R0..R2`, each exactly `{robot_id,intended_actions}`; actions contain every slot `0..179` exactly once as `{slot,action}`.  `planning_opportunities` has `3*36=108` objects exactly `{robot_id,slot,selected_frontier_id,selected_viewpoint,standby}`, with `slot` in `0,5,...,175` and the viewpoint either `{x,y}` or `null`.  `ping_opportunities` has 180 objects exactly `{slot,robot_a,robot_b}`; an idle opportunity uses two nulls, and a non-idle pair uses distinct IDs. `relative_estimates` has 216 objects exactly `{observer,neighbor,planning_slot,estimate_x,estimate_y,cov_xx,cov_xy,cov_yy,age,valid,source_measurement_ids}`, one for every ordered pair and planning slot.  All IDs, slots, coordinates, enums and nullability are typed as stated; finite floats use absolute tolerance `1e-6`; all objects reject undeclared fields. Array order is ignored except where a `slot` or pair key is explicit.  Measurement records are derived only by the grader and are not submitted separately.  Their IDs have the form `M{ping_slot}_{receiver}_{other_robot}` and may be submitted only as entries of a valid estimate's `source_measurement_ids`. For `robot_failure`, the `R2` planning records at slots `85,90,...,175` must be standby with both target fields null, and every estimate involving `R2` at those planning slots must use the invalid/null form.  This is the only fixed-failure-specific decision requirement; all other post-execution failure and coverage facts are replay-derived.

For auxiliary scoring, a non-idle pilot opportunity is decision-used exactly when at least one of its two successful derived measurement IDs appears in `source_measurement_ids` of any valid estimate. The decision-used fraction and the fraction that successfully arrives at `t+1` both use all non-idle pilot opportunities as denominator; either fraction is zero when no pilot is requested.  This scoring property does not require a counterfactual trace or any additional result field.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain non-empty headings named exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. The report must state the policy inputs used by each decision component and how the information boundary is obeyed. Candidate-defined files may contain additional controlled experiments and condition-level evidence.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact scenario, robot, planning, ping and estimate containers", weight: 3}
- {id: result_evidence, desc: "action, planning, pilot and estimate records satisfy their local field contracts", weight: 4}
- {id: canonical_coverage, desc: "all scenario, action, planning, ping and ordered-pair keys are complete and unique", weight: 4}
- {id: decision_legality, desc: "standby and failure decisions, pilot participation and the shared budget are legal", weight: 4}
- {id: replay_validity, desc: "local observation, signal sources, estimate timing, frontiers and motion replay consistently", weight: 6}
- {id: primary_metrics_1, desc: "final usable coverage and time-integrated usable coverage", weight: 8}
- {id: primary_metrics_2, desc: "overlap suppression, asynchronous termination and failure recovery", weight: 4}
- {id: primary_metrics_3, desc: "discovery yield, safety and standby efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "decision-used source and on-time pilot efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "relative-position quality, estimate freshness and covariance calibration", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the signal-only exploration, relative-estimate and local-information model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent local exploration or signal-only formulation; Implementation/evidence: no executable model exists."}
    - {score: 0.4, desc: "Report: a decisive omission of local maps, delayed pilots, multipath, uncertainty, standby or failure timing; Implementation/evidence: state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: local maps, pilot requests, measurements, relative estimates, frontiers and motion are accurately related; Implementation/evidence: the core model runs across all scenarios."}
    - {score: 1.0, desc: "Report: event order, information boundary, multipath uncertainty, HGrid effects, waiting and failure consequences are precise; Implementation/evidence: code and records trace these relations without material divergence."}
- id: method_design
  desc: "quality of the unified signal-driven frontier, estimation and motion method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal method, a prohibited method or explicit map exchange; Implementation/evidence: no reusable procedure supports it."}
    - {score: 0.4, desc: "Report: the method ignores a decisive signal-estimation, temporal, overlap, standby or failure trade-off, or presents unit-specific schedules as its primary solution; Implementation/evidence: decisions are partial, not state-driven, or differ materially from the report."}
    - {score: 0.7, desc: "Report: a legal, coherent and reproducible method addresses the principal pilot, estimation, frontier, standby and motion decisions; Implementation/evidence: the same procedure uses allowed local state and history across all units, with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified method addresses the central temporal, informational and resource consequences without prescribing away meaningful trade-offs; Implementation/evidence: allowed evolving state and history consistently drive its decisive choices without unit-specific action tables or material divergence."}
- id: experimental_coverage
  desc: "coverage of signal coordination, uncertainty, waiting and failure boundaries"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution exists."}
    - {score: 0.4, desc: "Report: only one canonical aggregate or an unexecuted condition claim; Implementation/evidence: records cannot test the signal-to-exploration relation."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one controlled comparison beyond canonical aggregates test a central signal-to-exploration mechanism or trade-off; Implementation/evidence: both conditions execute on comparable inputs with records supporting the comparison."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check; Implementation/evidence: condition-level records and metrics reproducibly distinguish both claims beyond aggregate scores."}
- id: research_analysis
  desc: "quality of evidence-based signal and exploration analysis"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: conclusions cannot be supported."}
    - {score: 0.4, desc: "Report: coverage or pilot counts are listed without mechanisms, uncertainty or limits; Implementation/evidence: records do not support the assertions."}
    - {score: 0.7, desc: "Report: actual records explain signal age, multipath, frontier overlap, standby, failure and safety trends while separating evidence, inference and limitations; Implementation/evidence: code supports the comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence explains mechanisms, quantifies principal trade-offs or sensitivities, tests the explanation against contrasting evidence or a failure boundary, and limits scope; Implementation/evidence: reproducible condition-level records support the central conclusions."}
