---
id: T032_bandwidth_aware_robot_coalitions
name: Bandwidth-Aware Coalition Formation for Cooperative Robot Information Gathering
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

Six mobile robots gather staged information targets in a cluttered facility. Each target contributes value only on its first observation, while four dependency groups add bonuses only if one linked target is completed in every stage. Robots can exchange short intended-plan summaries only within their current coalition. A large or poorly chosen coalition can consume the shared wireless capacity, whereas fragmented coalitions may duplicate valuable visits or meet at the same narrow passage without timely intent information. Re-forming coalitions also consumes energy. Coalition choice, plan delivery, and motion therefore jointly determine unique information reward, deadline quality, conflict, and energy use.

### Research Problem

Design one deterministic, state-driven method with two related decision roles. At each stage boundary, a coalition coordinator selects a partition from the newly available stage snapshot. Within the stage, the same per-robot policy selects an intent target, optional use of a delivered plan, one motion proposal, and at most one new or retry plan transmission from that robot's published view. The method must handle all canonical scenarios without scenario-specific partitions, routes, send schedules, or result tables. It should preserve high-value dependency groups, deconflict target visits and paths, and keep plans useful while controlling bandwidth, partition switching, safety interventions, and energy.

### Experimental Setting and Constraints

#### Canonical evaluation units

The canonical scenarios, in scenario-index order `s=0,...,4`, are `overlapping_targets`, `corridor_conflict`, `nonlocal_dependency`, `value_shift`, and `bandwidth_saturation`. Each uses a grid with `x in {0,...,21}` and `y in {0,...,15}`, robots `0,...,5`, and `H=72` one-second action slots `t=0,...,71`. State boundaries are `0,...,72`; slot `t` maps boundary `t` to `t+1`. The stages are `g=0,1,2` with action-slot intervals `[0,24)`, `[24,48)`, and `[48,72)`. Boundary `72` contains the packet results, executed movement, energy charge, and target observations caused by slot `71`. It has no subsequent activation, partition, action decision, or movement.

The target positions are fixed as follows. In every JSON ID field, target `Ti` is represented by the integer `i`. Targets `T0,...,T3`, `T4,...,T7`, and `T8,...,T11` activate at boundaries `0`, `24`, and `48`, respectively, and have deadline states `24`, `48`, and `72`. Target `Ti` belongs to group `G(i mod 4)`. Thus `G0={T0,T4,T8}`, `G1={T1,T5,T9}`, `G2={T2,T6,T10}`, and `G3={T3,T7,T11}`.

| target | position | target | position | target | position |
|---|---:|---|---:|---|---:|
| `T0` | `(2,2)` | `T4` | `(10,2)` | `T8` | `(18,2)` |
| `T1` | `(2,13)` | `T5` | `(10,13)` | `T9` | `(18,13)` |
| `T2` | `(6,3)` | `T6` | `(14,3)` | `T10` | `(20,5)` |
| `T3` | `(6,12)` | `T7` | `(14,12)` | `T11` | `(20,10)` |

| scenario | robot starts `0..5` | closed obstacle rectangles | `value(T0..T11)` | `bonus(G0..G3)` | `ell_s` |
|---|---|---|---|---|---:|
| `overlapping_targets` | `(0,1),(0,4),(0,7),(0,9),(0,12),(0,14)` | `[8,9]x[4,11]` | `(2,2,3,3,3,3,4,4,4,4,5,5)` | `(5,5,6,6)` | `0.08` |
| `corridor_conflict` | `(1,1),(1,5),(1,10),(1,14),(8,1),(8,14)` | `[7,16]x[5,6]`, `[7,16]x[9,10]` | `(2,2,2,2,4,4,4,4,5,5,5,5)` | `(6,6,6,6)` | `0.10` |
| `nonlocal_dependency` | `(0,2),(0,13),(5,1),(5,14),(11,1),(11,14)` | `[9,10]x[3,12]` | `(1,1,3,3,2,2,5,5,3,3,6,6)` | `(8,8,5,5)` | `0.12` |
| `value_shift` | `(1,2),(1,13),(7,2),(7,13),(13,2),(13,13)` | `[4,5]x[5,10]`, `[15,16]x[5,10]` | `(4,1,4,1,1,5,1,5,6,2,6,2)` | `(5,7,5,7)` | `0.10` |
| `bandwidth_saturation` | `(0,1),(0,4),(0,7),(0,9),(0,12),(0,14)` | `[11,12]x[2,13]` | `(3,3,3,3,4,4,4,4,5,5,5,5)` | `(7,7,7,7)` | `0.18` |

These tables are public simulator inputs. From boundary `0`, both decision roles know the grid, scenario index, starts, obstacle map, the current scenario's `ell_s`, and the published packet outcome formula. The environment may hold the complete table, but the decision procedures must receive target specifications only after their stage activates. Future target values, positions, deadlines, and groups may not influence an earlier partition or robot action.

#### Targets, movement, safety, and energy

An active, incomplete target is first observed at the post-motion boundary of a slot when a robot's executed position has Manhattan distance at most `1` from it. Observations are processed in robot-ID order and then target-ID order. The first observation completes the target at boundary `t+1` and earns its value; another robot in range at that boundary, and later visits to the completed target, are repeats and add no value. A target is observable only during its stage, including the post-motion boundary of that stage's final slot. Completing all three members of a dependency group by their respective deadlines earns that group's bonus once.

Each robot proposes one of `(0,0),(1,0),(-1,0),(0,1),(0,-1)`. Proposals are processed in robot-ID order. A destination outside the grid, inside an active closed obstacle, equal to an already accepted lower-ID destination, or equal to a still-current higher-ID position is replaced by stay. This rule also replaces both sides of a pairwise swap. Each replaced nonzero proposal adds one safety event. A successfully executed nonzero move costs `2` energy and an executed stay costs `0.2`. Each admitted communication attempt costs its sender another `1`. Every robot starts with `120` energy, and the partition plus action for a slot is illegal if its actual switching, communication, and executed-motion costs would make any energy negative.

Each closed rectangle `[xmin,xmax]x[ymin,ymax]` contains every integer grid position satisfying both inclusive coordinate bounds. Obstacles do not change during a scenario.

#### Stage partitions and switching

At boundaries `0`, `24`, and `48`, the environment first activates and publishes the current four targets. It then gives the coalition coordinator a stage snapshot containing all six current positions and energies, all completed targets, the previous partition, switch history, and the remaining accepted-attempt budget. The coordinator submits the current stage's `coalitions`. Future-stage target specifications are absent from this snapshot.

A partition consists of nonempty, pairwise-disjoint coalitions whose union is exactly `{0,1,2,3,4,5}`; coalition size is `1..4`. Member order and coalition order have no meaning. The partition remains fixed for all 24 action slots of its stage. Stage `0` has no switch cost. At boundaries `24` and `48`, robot `i` incurs one coalition switch exactly when the set of its partners in the new partition differs from the preceding stage, and each such robot automatically pays `1` energy before that boundary's action. There can therefore be at most `12` switches per scenario.

#### Plan packets, delivery, and use

At most one communication attempt appears in each robot action. `comm` is `none`, `new`, or `retry`. A `new` action creates a scenario-unique nonnegative `packet_id`; its `recipient` must be a different member of the source's current coalition. The abstract plan summary contains the source's current active, incomplete `plan_target` and two predicted source move proposals for action slots `t+1` and `t+2`. The two predictions use the same five-move domain as motion. `plan_target` must equal that row's non-null `intent_target`. `retry` names only a previously admitted logical packet whose first attempt failed in slot `t-1`, and only its original source may retry it. The source and original recipient must remain in the same current coalition, and the generation and retry slots must belong to the same stage. A logical packet can be retried once. `none` creates no attempt.

All otherwise legal attempts in a slot are ordered by: retry before new, original generation slot, the source's current coalition as a sorted tuple, source ID, recipient ID, and packet ID. At most the first four and at most the remaining part of the `216`-attempt scenario budget are admitted. Capacity- or budget-dropped attempts consume no energy, produce no arrival, and cannot be retried; submitting them is not itself illegal. A dropped `new` attempt still consumes its scenario-unique packet ID, and a dropped `retry` exhausts the packet's only retry opportunity. Each admitted attempt costs one energy and produces success or failure feedback one boundary later.

For scenario index `s`, stage `g=floor(t/24)`, source `i`, recipient `j`, and attempt `a in {0,1}`, define

`u=((97*(t+1)+31*(i+1)+17*(j+1)+13*(g+1)+19*(a+1)+29*(s+1)) mod 1000)/1000`.

The attempt fails exactly when `u < ell_s`; otherwise its plan is delivered at boundary `t+1`. This outcome is independent of candidate-chosen packet IDs and plan content. A first failure opens one retry opportunity in the immediately following action slot; omission or retry failure discards the packet. A delivered plan may be used only by its recipient, only when its generation slot and use slot belong to the same stage and source and recipient belong to that stage's same coalition, and only in action slots `generation_slot+1` and `generation_slot+2`. A retry delivery can consequently be used only in the second slot. The optional `plan_ref` is the robot's explicit claim that one delivered valid packet was consulted for that action; null makes no such claim. The automatic replay checks only whether the cited packet was available under the public timing and information rules. The report and implementation remain the evidence for whether the policy genuinely consulted the cited plan.

At a stage boundary, a packet generated in the preceding stage may still produce its scheduled success or failure feedback, but any resulting delivery or retry opportunity immediately expires: it cannot be used or retried in the new stage. At boundary `72`, scheduled feedback is likewise processed for replay accounting, but no plan use or retry action remains in the horizon.

#### Information and event timing

Within a stage, robot `i` may use its own current position, energy, past actions, sent-packet feedback, and received-plan history; the current partition; current positions of members of its own coalition; all target specifications published so far and their public completion states; the public obstacle map; and currently valid plans addressed to `i`. It may not use another member's energy or private history, any other coalition's live position, energy, intent, plan, or packet state, an undelivered packet result, or a future-stage target. The coordinator's full stage-boundary snapshot is not continuously available to robot policies. Plan summaries provide no unstated state. The coordinator's only candidate output is the stage partition: it may not turn its full snapshot into a team-wide target assignment or precomputed robot actions. A candidate simulator may use global positions to apply the fixed safety rule only after all six actions have been independently formed from their allowed views; a pre-submission reservation pass using other coalitions' live positions is not an allowed robot-policy input.

For each slot, the authoritative order is: expire any retry opportunity omitted in its sole eligible preceding slot; process prior-slot packet results and expose a same-stage first-failure opportunity; at a stage boundary, immediately expire preceding-stage deliveries and retry opportunities, activate the four targets, apply the submitted partition, count switches, and charge switch energy; expose the coordinator or per-robot decision views; read all actions; validate intents, plan references, packet references, and move domains; mark a newly exposed retry opportunity discarded if its source does not submit that packet's retry in this slot; admit attempts and schedule next-boundary outcomes; apply the ID-ordered safety rule; validate and charge actual communication and motion/stay energy; execute movement; process target observations; and record boundary `t+1`. Submitted partition and action fields are decision-boundary requests. Executed positions, energy, packet outcomes, target completion, repeats, safety events, switches, dependency rewards, and metrics are post-event facts derived by the grader.

For metric replay, a logical packet has a coordination-effective citation when its recipient's earliest valid `plan_ref` citation satisfies all of the following in that action slot: the source's submitted move matches the advertised move for that offset; the recipient's `intent_target` is null or differs from the packet's `plan_target`; and the source's and recipient's submitted destinations are distinct and do not form a pairwise swap. These are observable consistency conditions, not a hidden counterfactual test of the candidate's internal state.

#### Legal actions and boundary conditions

For every row, `intent_target` is null or an active, incomplete target at action start. `plan_ref` is null or a delivered valid packet addressed to that robot. For `none`, `packet_id`, `recipient`, `plan_target`, and all four plan-move fields are null. For `new`, all those fields are non-null and meet the creation rules above. For `retry`, only `packet_id` is non-null; `recipient`, `plan_target`, and all four plan-move fields are null. JSON null and integer `-1` are equivalent for `intent_target`, `plan_ref`, `packet_id`, `recipient`, and `plan_target` under all rules. Predicted moves need only be in the move domain because future state may change; the actual source proposal is not forced to match. Invalid partitions, move domains, intents, plan references, packet IDs, recipients, summaries, retries, or insufficient energy close the validity gate. Published capacity and collision interventions are legal environment outcomes and affect metrics.

#### Task metrics

Let `C_s,r` indicate that target `r` was completed by its deadline in scenario `s`, and let `V_s^max=sum_r value_s,r + sum_g bonus_s,g`. Then

`primary_metrics_1=(1/5)*sum_s [sum_r value_s,r*C_s,r + sum_g bonus_s,g*prod_(r in Gg) C_s,r]/V_s^max`.

For target `r` in stage `g`, define `q_s,r=clip(1-(completion_state_s,r-24*g)/24,0,1)` if it completes by its deadline and `0` otherwise. Then `primary_metrics_2=(1/5)*sum_s [sum_r value_s,r*q_s,r / sum_r value_s,r]`.

For scenario `s`, let `M_s` be executed nonzero move steps, `E_s` total energy used, and `S_s` safety events. Define `h_s=0.5*clip(1-S_s/(72*6),0,1)+0.3*clip(1-M_s/(72*6),0,1)+0.2*clip(1-E_s/(6*120),0,1)`. Then `primary_metrics_3=(1/5)*sum_s h_s`.

Let `A_s` be admitted communication attempts, `U_s` coordination-effective logical packets, and `D_s` the number of distinct `(recipient,stage)` pairs represented by those packets. There are exactly `6*3=18` possible recipient-stage pairs. Set `e_s=0` when `A_s=0`; otherwise set `e_s=clip(D_s/18,0,1)*(0.6+0.4*clip(U_s/A_s,0,1))`. Each logical packet contributes at most once to `U_s`, and repeated packets for one recipient-stage pair do not increase `D_s`. Then `auxiliary_metrics_1=(1/5)*sum_s e_s`.

For each delivered logical packet and advertised offset `k in {1,2}` that remains in its generation stage and within the horizon, compare the advertised move with the source's submitted move at `generation_slot+k`. Let `P_s` be the number of distinct `(source,stage,k)` triples having at least one such match. There are exactly `6*3*2=36` possible triples. If `W_s` is the number of robots whose partner set changes at boundaries `24` and `48`, define `r_s=clip(P_s/36,0,1)*(0.7+0.3*clip(1-W_s/12,0,1))` and `auxiliary_metrics_2=(1/5)*sum_s r_s`. Thus silence, a single packet, or repeated packets over one already-covered robot-stage cell cannot nearly saturate either auxiliary metric.

All five metrics are independently replayed and clipped to `[0,1]`. Normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Result record contract

`results.json` contains exactly `{"scenarios": [...]}` with one record per canonical scenario. Each scenario record has exactly `scenario`, `partitions`, and `actions`. `partitions` contains exactly three rows, uniquely keyed by `stage in {0,1,2}`; each row has exactly `stage` and `coalitions`. `actions` contains exactly `72*6` rows, uniquely keyed by `(t,robot)`. Scenario, partition, coalition, member, and action array order has no semantic meaning.

Every action row has exactly `t,robot,dx,dy,intent_target,plan_ref,comm,packet_id,recipient, plan_target,plan_dx1,plan_dy1,plan_dx2,plan_dy2`. `t`, `robot`, `dx`, and `dy` are JSON integers; `comm` is a string; the five optional ID fields are JSON integers or null; and the four plan-move fields are JSON integers for `new` and JSON null for `none` or `retry`. Each `coalitions` value is an array of nonempty arrays of JSON integer robot IDs. The domains and conditional field relationships are those defined above. No position, energy, executed move, packet outcome, plan age, target completion, repeat, safety, switch, dependency, or metric summary is submitted. The grader derives those facts from the fixed model, so equivalent correct simulators are not rejected for choosing another redundant state representation. No field is optional and all JSON numbers must be finite.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain non-empty sections titled `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Non-canonical controlled experiments used for the report or Judge must be generated by executable project code and may be summarized in `report.md` or separate supplementary files. They must not be inserted into canonical `results.json`; supplementary files are not Auto Check inputs.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact root and scenario containers are present", weight: 3}
- {id: result_evidence, desc: "minimal partition and action fields are typed, finite, and auditable", weight: 4}
- {id: canonical_coverage, desc: "all scenarios, stages, and slot-robot action keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "partitions, moves, intents, plan references, packets, retries, and energy are legal", weight: 4}
- {id: replay_validity, desc: "switch, packet, safety, target, dependency, and metric replay is complete", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed unique target and dependency reward", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed stage response quality", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed safety, movement, and energy quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed coordination-effective plan coverage and efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed prediction coverage and coalition continuity", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the staged target, coalition, plan, bandwidth, safety, and energy model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent staged coalition-information model; Implementation/evidence: no executable target, partition, communication, and motion process."}
    - {score: 0.4, desc: "Report: coalition or target collection is partial and omits stage timing, plan delivery, capacity, switching, dependencies, or information boundaries; Implementation/evidence: only a partial process or a model with a material information departure is implemented."}
    - {score: 0.7, desc: "Report: activated targets, coordinator snapshots, per-robot views, partitions, plan packets, shared capacity, switching, motion, and reward are accurate; Implementation/evidence: the core model runs across all scenarios with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: event order and the causal coalition-bandwidth-dependency trade-off are precise, including boundary and failure behavior; Implementation/evidence: minimal decisions and code faithfully realize all relations without material divergence."}
- id: method_design
  desc: "quality of the unified coalition, plan exchange, target, and motion policy under the published decision views"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, a prohibited training method, or a policy that relies on unavailable future or cross-coalition live state; Implementation/evidence: this includes using the coordinator's full snapshot to assign team-wide robot intents, passing such a global assignment into robot actions, or rewriting proposed moves in a pre-submission reservation pass over other coalitions' positions. Global truth is allowed only for the fixed post-submission environment replay."}
    - {score: 0.4, desc: "Report: fixed or spatial grouping and independent greedy motion ignore a decisive dependency, congestion, prediction, or switch consequence; Implementation/evidence: decisions are prefilled, only weakly state-driven, or materially inconsistent with the report."}
    - {score: 0.7, desc: "Report: one reproducible state-driven method handles stage partitions, plan exchange, target choice, safe motion, and resource limits; Implementation/evidence: the coordinator and robot policy use their published views across every scenario with limited planning depth."}
    - {score: 1.0, desc: "Report: a justified method adapts coalitions and motion using dependency value, plan uncertainty, capacity congestion, spatial conflict, stage change, and switch cost; Implementation/evidence: evolving allowed state drives decisions without scenario, stage, target, or slot action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for coalition-bandwidth effects"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only canonical aggregates or one uninformative coalition condition are shown; Implementation/evidence: evidence does not test the coalition-bandwidth mechanism."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled capacity, dependency, stage, switch-cost, or coalition comparison are covered; Implementation/evidence: both conditions execute with comparable condition-level records."}
    - {score: 1.0, desc: "Report: complementary controlled experiments test dependency-aware grouping and a bandwidth, conflict, stage-change, or switching failure boundary; Implementation/evidence: reproducible condition-level partitions, actions, packet outcomes, and metrics support both comparisons."}
- id: research_analysis
  desc: "evidence-grounded interpretation of coalition value, plan traffic, target reward, conflict, and switching"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: values are listed without causal coalition, bandwidth, dependency, conflict, or switch analysis or material limitations; Implementation/evidence: claims are unsupported."}
    - {score: 0.7, desc: "Report: actual records explain important partition, plan, target, safety, and stage trends and limitations; Implementation/evidence: observations agree with generated evidence."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies coalition and bandwidth mechanisms, tests the explanation at a contrasting condition or failure boundary, and states scope; Implementation/evidence: reproducible condition-level evidence supports the conclusions."}
