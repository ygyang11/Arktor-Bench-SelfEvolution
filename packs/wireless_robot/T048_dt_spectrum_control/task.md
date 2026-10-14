---
id: T048_dt_spectrum_control
name: Digital-Twin-Aware Spectrum and Formation Control for Rescue Robots
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

Three point robots operate in a temporary rescue network. A leader moves toward a terminal location while two followers must maintain a formation and pass a collaboration gate. The leader can send short position-prediction packets over two directed D2D links, but a slow edge scheduler allocates a shared twelve-unit spectrum budget to the next ten-slot control window. Link gains, FIFO delay, packet loss, prediction age, obstacles, and control energy interact: a larger share can make a prediction arrive sooner, while an allocation chosen from the current link view may be poor in the next window. A follower without a valid recent prediction must stop.

### Research Problem

Build a self-contained deterministic simulator and design a non-training online procedure that, in every control slot, chooses the three robot commands and, at the even-slot prediction opportunities, whether to send a leader prediction on each link. At allocation boundaries choose the spectrum shares for the next window from the allowed current snapshot. The procedure should balance terminal progress, collaboration-gate completion, formation and clearance safety, smooth motion, communication load, prediction freshness, queue waiting, and energy.

Use one reusable state-driven procedure for all five canonical scenarios. The procedure may use the current slot, the public static map, and gains revealed for the current slot. It must not map a scenario, phase, task ID, absolute slot, gain table, or obstacle table directly to pre-filled allocations, commands, send slots, or trajectories, and it must not use future gains, packet results, positions, or actions. Deterministic rules, search, planning, optimization, and analytical approximations are allowed, but machine learning, deep learning, reinforcement learning, training, fine-tuning, and online learning are forbidden.

### Experimental Setting and Constraints

#### World, robots, and scenarios

The closed operating rectangle is `R=[0,20] x [0,14]` metres. There are `H=120` control slots, each of duration `delta=0.25 s`, numbered `t=0,...,119`, followed by terminal boundary `t=120`. Robots are `L`, `F1`, and `F2`. Their initial positions are respectively `(1.5,7.0)`, `(0.6,6.4)`, and `(0.6,7.6)`. Desired follower offsets from the leader are `o_F1=(-0.9,-0.6)` and `o_F2=(-0.9,0.6)` metres. Every listed obstacle is a closed axis-aligned rectangle. Let `clr(z)` be the minimum Euclidean distance from point `z` to the operating-region boundary and all obstacle rectangles. A valid robot position has `clr(z)>=0.4`; every pair of robot centres must be at least `0.8 m` apart.

The leader target is `g_L` in the table. Follower targets are `g_L+o_F1` and `g_L+o_F2`. A robot first entering its target radius `0.35 m` after an action has its arrival slot. The terminal target deadline is `d_goal=112`. A collaboration gate is evaluated after actions in slots `t in {20,...,d_gate}`. It succeeds at the first such slot for which the leader is within `0.35 m` of `c_s`, the mean follower formation error is at most `e_ref=0.55 m`, and every pair distance is at least `0.8 m`.

| scenario | leader target `g_L` | gate centre `c_s` | `d_gate` | static obstacles |
|---|---|---|---:|---|
| `formation_clearance` | `(18.0,7.0)` | `(10.0,7.0)` | 68 | `[7.6,8.4]x[0,5.1]`; `[7.6,8.4]x[8.9,14]`; `[13.2,14.0]x[0,3.5]`; `[13.2,14.0]x[10.5,14]` |
| `obstacle_detour` | `(18.0,11.2)` | `(14.0,11.2)` | 82 | `[6.5,12.8]x[4.1,9.9]`; `[14.4,15.1]x[0,7.0]` |
| `shared_bottleneck` | `(18.0,7.0)` | `(10.8,7.0)` | 72 | `[8.0,9.0]x[0,5.0]`; `[8.0,9.0]x[9.0,14]`; `[13.0,13.8]x[0,5.2]`; `[13.0,13.8]x[8.8,14]` |
| `asymmetric_links` | `(18.0,3.0)` | `(13.0,3.0)` | 86 | `[6.0,10.0]x[5.2,8.8]`; `[13.0,13.7]x[7.5,14]` |
| `spectrum_surge` | `(18.0,7.0)` | `(10.0,7.0)` | 76 | `[7.8,8.6]x[0,5.4]`; `[7.8,8.6]x[8.6,14]`; `[13.5,14.3]x[0,4.6]`; `[13.5,14.3]x[9.4,14]` |

#### Spectrum, packets, queues, and energy

There are exactly two directed links, `L_F1` and `L_F2`; no reverse data link exists. ACKs are local feedback events and consume no queue or spectrum. Windows are `W_k={10k,...,10k+9}` for `k=0,...,11`. In slot `t`, link gain is the entry at `k=floor(t/10)` in the table below. The current gains for the current slot are revealed at slot start. Future gains and future outcomes are not revealed. The two link queues are serviced independently. Each window has integer shares `q_F1,q_F2 in {0,...,12}` with `q_F1+q_F2<=12`. `W_0` is fixed at `(6,6)`. At slot `10k` for `k=0,...,10`, a submitted allocation affects only `W_(k+1)`, beginning at slot `10(k+1)`; no allocation is submitted at `k=11`. For link `l`,

`B_l(t)=floor(40*q_l*G_l[floor(t/10)])` bits.

| scenario | loss phase `p_s` | `G_F1[0:11]` | `G_F2[0:11]` |
|---|---:|---|---|
| `formation_clearance` | 1 | `0.90,0.75,0.65,1.05,0.55,0.95,0.70,1.10,0.60,0.85,0.75,1.00` | `0.70,1.00,0.60,0.75,1.10,0.55,0.95,0.65,1.05,0.70,0.85,0.60` |
| `obstacle_detour` | 5 | `0.85,0.55,0.70,0.60,1.05,0.75,0.60,1.10,0.65,0.90,0.55,0.80` | `0.65,1.05,0.85,0.70,0.55,0.95,1.10,0.65,0.80,0.60,0.90,0.75` |
| `shared_bottleneck` | 9 | `0.70,0.60,1.05,0.75,0.55,1.10,0.65,0.90,0.60,0.85,1.00,0.70` | `1.00,0.75,0.60,1.10,0.65,0.55,0.95,0.70,1.05,0.60,0.80,0.90` |
| `asymmetric_links` | 13 | `1.10,1.00,0.95,0.80,0.70,1.05,0.90,0.75,1.10,0.85,0.95,0.80` | `0.55,0.60,0.65,0.70,0.55,0.60,0.65,0.55,0.70,0.60,0.55,0.65` |
| `spectrum_surge` | 17 | `0.85,0.75,0.60,1.10,0.55,0.65,1.15,0.70,0.60,1.05,0.55,0.90` | `0.70,1.05,0.65,0.60,1.10,0.55,0.70,1.15,0.60,0.65,1.05,0.55` |

At every even slot `r in {0,2,...,118}` and on each link, exactly one prediction opportunity exists. A sent packet has exactly `960` bits and contains the leader position at source slot `r` and eight subsequent slot-start predictions, for nine points total. A packet's `source_slot` is the even opportunity slot. Packet IDs must be unique and generated by the reusable procedure from the link and source opportunity; they must not be reused. Each link FIFO has capacity `3840` bits. Choosing `send` creates a packet record. If the queue has insufficient remaining capacity, the scenario has already admitted 80 packets, or leader energy is insufficient, the packet is recorded as `admitted=false,outcome=dropped` and never enters a queue. Accepted packets enter after the decision and before service. Queue order is `(queue_enter_slot, packet_id)`; same-slot admissions are processed in `L_F1` then `L_F2` order, including the global 80-packet limit. A queue head keeps its residual bits between slots; after it completes, unused service in that slot continues on the next head of that link, and only unused service at the slot end is discarded.

A packet completed in slot `t` is lost exactly when

`(17*t + 5*p_s + 7*r_l) mod 37 == 0`,

where `r_F1=1` and `r_F2=2`; otherwise it arrives at the start of `t+1`. A lost packet is removed at the completion boundary and is not retried automatically. Its ACK is visible at `t+1`, with `arrival_slot=null` and `ack_slot=t+1`. A delivered packet has both arrival and ACK at `t+1`. Any accepted packet not completed by the end of slot 119, whether or not it received service, is `unfinished`. A packet completed in slot 119 may arrive and produce terminal feedback at boundary 120, but it cannot change control, allocation, or task metrics. Same-slot packets and all dynamic packet records are ordered by `(source_slot, link_rank, packet_id)` with `link_rank(L_F1)=0` and `link_rank(L_F2)=1`.

Every robot starts with energy 26. For a command velocity `v_i(t)`, with `v_i(-1)=(0,0)`, the motion cost is

`0.04*||v_i(t)||^2 + 0.02*||v_i(t)-v_i(t-1)||`.

Each accepted prediction packet additionally costs the leader `0.00025*960`. A command is limited to `||v_i(t)||<=1` and, except for the mandatory fallback below, to `||v_i(t)-v_i(t-1)||<=0.35`. Accepted packet energy is debited at admission before command energy is checked. If the selected command cost would then make that robot's energy negative, that command is illegal, is replayed as zero velocity, and does not debit command energy.

#### Information, prediction, and event order

All robots know the static map, their own target, the gate, desired offsets, and the current effective allocation. The leader additionally sees its own position, energy, previous command, both queue summaries, prior ACKs, and the two gains revealed for the current slot. Each follower sees only its own position, energy, previous command, its-link ACK history, and predictions already arrived on its link. A follower cannot read the leader's true position or velocity, a future trajectory, or the other follower's state. At allocation boundaries the edge scheduler sees the three current robot states, task progress, both queue summaries, prior ACKs, and both current gains. The current gains are visible decision inputs to the edge and leader and are independently checked in the submitted allocation evidence. The edge may derive current formation error from its three current position observations; that derived value and the other robots' current positions are not visible to an individual robot role. Future gains, future packet results, future actions, future positions, future formation error, and collision outcomes are hidden environment facts.

For a packet created at source slot `r`, its trajectory `P=(p_0,...,p_8)` must have `p_0=z_L(r)`, every point inside the region with `clr(p_j)>=0.4`, and `||p_(j+1)-p_j||<=0.25`. At the start of slot `t`, a follower may reference an arrived packet on its own link exactly when its source slot is `r` and `0<=t-r<=8`; it then uses `P[t-r]` as the leader prediction. Without a valid reference, the only legal follower command is the fixed safety fallback: a null prediction ID and zero executed velocity. This mandatory emergency stop is exempt from the `0.35` velocity-change limit, but its braking change is still included in the command-energy formula unless the command energy itself is unaffordable. A non-null unavailable, wrong-link, or expired prediction ID is illegal even when paired with zero velocity. Prediction error is `||P[t-r]-z_L(t)||`.

Each slot executes in this order:

1. Process packets arriving at boundary `t` and expose their ACKs.
2. Reveal the current gains and construct the permitted role snapshots.
3. At `t=10k,k<=10`, choose the allocation for the next window.
4. Choose the three robot commands.
5. If `t` is even, choose `send` or `wait` and a trajectory for each link.
6. Admit new packets in link order, debit accepted packet energy, and service both FIFO queues   with the current effective shares. A completed packet is removed before the next head is served.
7. Apply the loss rule to completed packets, then execute motion, collision, energy, target, and   gate updates.
8. Record the slot result and the next state.

For a command segment from `z_i(t)` to `z_i(t)+delta*v_i(t)`, if its minimum distance to the region boundary or any obstacle is below `0.4`, that robot is involved in a collision. If the minimum distance between any two commanded segments is below `0.8`, both robots are involved. All robots involved in at least one violation stay at their old positions; each involved robot gets one collision event in that slot, while its command energy is still charged. Non-involved robots move by `delta*v_i(t)`. Arrival and gate checks use the resulting positions. A collision count is therefore the number of robot-slot collision events, at most `3*H`.

Allocation, prediction-reference, and send fields below record the candidate choices at their decision boundary. Slot velocity fields record the effective command after the mandatory safety fallback and the energy-affordability rule; they are unchanged by a later collision. Allocation gain fields record observations visible at that boundary. State, packet, and queue fields are post-rule traces retained as implementation-audit evidence. The event order above is authoritative, and the grader independently replays those traces from the decision records and the fixed model.

#### Task metrics

Let `z_i(t)` be the independently replayed slot-start position and `f_s(t)=0.5*(||z_F1(t)-z_L(t)-o_F1||+||z_F2(t)-z_L(t)-o_F2||)`. Let `C_s={20,...,d_gate}`. For robot `i`, set `D_i=max(1,||z_i(0)-g_i||)`. Set `p_i=1` if its first arrival slot is at most `d_goal`; otherwise set `p_i=clip(1-||z_i(d_goal+1)-g_i||/D_i,0,1)`. Define `ind_s=0.4*p_L+0.3*p_F1+0.3*p_F2`, and let `D_gate` be the leader's initial Euclidean distance to `c_s`. Then

`gate_s=max_(t in C_s) clip(1-||z_L(t+1)-c_s||/D_gate,0,1) * clip(1-f_s(t+1)/0.55,0,1) * I[min_pair_distance(t+1)>=0.8]`,

and `task_s=0.55*ind_s+0.45*gate_s`.

- `primary_metrics_1` is the five-scenario mean of `task_s`.
- For `t in C_s`, let `safe_s` be the fraction of slots in which all three positions have  clearance at least `0.4` and every pair distance is at least `0.8`. Let `valid_s` be the fraction  of the `2*|C_s|` follower-slot samples whose referenced packet has arrived, has age at most eight,  and predicts the current leader position with error at most `0.75`. Let  `form_s=mean_(t in C_s) clip(1-f_s(t)/0.55,0,1)` and  `clear_s=mean_(i,t in {L,F1,F2}xC_s) clip((clr(z_i(t))-0.4)/0.6,0,1)`.  `primary_metrics_2` is the five-scenario mean of  `0.45*safe_s+0.25*valid_s+0.30*(0.5*form_s+0.5*clear_s)`.
- Let `collision_s` be the number of robot-slot collision events,  `variation_s=sum_(i,t=0..119)||v_i(t)-v_i(t-1)||`, and  `energy_s=sum_i(26-energy_i(120))`. Define  `safe_motion_s=clip(1-collision_s/(3H),0,1)`,  `smooth_s=clip(1-variation_s/(0.35*H*3),0,1)`, and  `energy_eff_s=clip(1-energy_s/78,0,1)`. `primary_metrics_3` is the five-scenario mean of  `task_s*(0.4*safe_motion_s+0.3*smooth_s+0.3*energy_eff_s)`.

For the auxiliary metrics, let `V_s` be the number of follower-slot samples counted by `valid_s`, `support_s=V_s/(2*|C_s|)`, `A_s` be the number of accepted prediction packets, and `U_s` be the set of accepted packets referenced by at least one such valid sample. Set `useful_s=|U_s|/A_s`, or zero when `A_s=0`.

- `auxiliary_metrics_1` is the five-scenario mean of  `support_s*(0.4+0.6*useful_s)`.
- For each packet in `U_s`, let `wait=service_start_slot-queue_enter_slot`, and let `age_s` be the  mean age over the valid follower-slot samples. Define  `fresh_s=support_s*clip(1-age_s/8,0,1)` and  `wait_s=support_s*clip(1-mean(wait)/H,0,1)`; both are zero if `U_s` is empty. At the start  `t=10k` of window `k`, let `a_lk=8` if link `l` has no arrived packet with source age at most  eight, and otherwise let it be the source age of the latest such packet. Let  `e_lk=||z_l(10k)-z_L(10k)-o_l||` and  `u_lk=1+I[10k in C_s]*(1+clip(a_lk/8,0,1)+clip(e_lk/0.55,0,1))`. If `q_lk` is the effective  share in `W_k`, define `q*_lk=12*u_lk/(u_F1k+u_F2k)` and  `h_l=sum_k u_lk*min(1,q_lk/q*_lk)/sum_k u_lk`. If `h_F1+h_F2=0`, set `balance_s=0`; otherwise  set  `balance_s=((h_F1+h_F2)/2)*((h_F1+h_F2)^2/(2*(h_F1^2+h_F2^2)))`.  `auxiliary_metrics_2` is the five-scenario mean of  `0.4*fresh_s+0.3*wait_s+0.3*support_s*balance_s`.

Every metric is clipped to `[0,1]` and higher is better. The normalized task performance is `(8*primary_metrics_1+4*primary_metrics_2+3*primary_metrics_3+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. All metric inputs come from independent replay, and finite comparisons use absolute tolerance `1e-6`.

#### `results.json` contract

The project root must generate a JSON object with exactly one key, `scenarios`. It contains five records in the scenario-table order. Each record has exactly `scenario`, `states`, `slots`, `allocation_decisions`, `prediction_opportunities`, `packets`, and `queue_states`.

`states` has 121 entries in `slot=0,...,120` order. Each entry has exactly `slot`, `leader`, `follower_1`, and `follower_2`. Each robot object has finite `x` in `[0,20]`, `y` in `[0,14]`, and `energy` in `[0,26]`. Entry `t` is the state at the start of slot `t`; entry 120 is terminal.

`slots` has 120 entries in `slot=0,...,119` order. Each has exactly `slot`, `leader_velocity`, `follower_1_velocity`, `follower_2_velocity`, `follower_1_prediction_id`, and `follower_2_prediction_id`. A velocity has finite `vx,vy` in `[-1,1]` and is the effective command defined above; a prediction ID is a string or `null`. A null follower prediction ID requires zero follower velocity, while a non-null ID must name a valid arrived packet on that follower's link.

`allocation_decisions` has 11 entries in `decision_window=0,...,10` order. Each has exactly `decision_window`, `decision_slot=10*decision_window`, `effective_window=decision_window+1`, integer `follower_1_units,follower_2_units` in `0..12`, and finite observed gains in `[0.55,1.15]`. The observed gains equal the current gains revealed at the decision slot. `W_0=(6,6)` is not duplicated as a decision record.

`prediction_opportunities` has 120 entries in `(slot,link)` order, with `slot` even from 0 through 118 and `link` in `L_F1,L_F2` order. Each has exactly `link`, `slot`, boolean `send`, `packet_id`, and `trajectory`. For `wait`, the last two are `null`; for `send`, the ID is a scenario-local unique nonempty string and the trajectory has nine finite `{x,y}` points satisfying the prediction constraints.

`packets` is a dynamic array. Every entry has exactly `packet_id`, `link`, `source_slot`, `bits`, `admitted`, `queue_enter_slot`, `service_start_slot`, `service_complete_slot`, `arrival_slot`, `ack_slot`, and `outcome`. Link is `L_F1` or `L_F2`; source slots are even in `0..118`; bits is 960; admitted is boolean; event slots are in `0..119` or `1..120` as applicable. A dropped packet has `admitted=false`, outcome `dropped`, and null queue/service events. An unfinished accepted packet has no completion, arrival, or ACK event. A lost packet has `ack_slot=service_complete_slot+1` and null arrival. A delivered packet has `arrival_slot=ack_slot=service_complete_slot+1`. Packets are created only by send opportunities and follow the stated stable order.

`queue_states` has 240 entries in `(slot,link)` order for every slot and both links. Each has exactly `slot`, `link`, integer `bits_start,bits_end` in `0..3840`, integer `bits_admitted` in `{0,960}`, and integer `bits_served` in `0..552`. `bits_start` is after boundary arrivals and before decisions, `bits_admitted` is accepted input in that slot, and `bits_served` is actual FIFO service. `bits_end=bits_start+bits_admitted-bits_served` and is nonnegative.

State slots, action slots, allocation decisions, prediction opportunities, and queue slots are immutable canonical units. Packets are dynamic evidence, and packet-ID uniqueness is scoped to one scenario. The required state, packet-outcome, and queue traces expose post-rule evidence for replay consistency and implementation/Judge audit; each is checked only in its declared evidence or replay ownership. The grader independently derives prediction errors, collisions, target and gate completion, energy-use totals, and all aggregate metrics, so candidates submit no separate fields for those summaries. Integers, IDs, enums, booleans, array lengths, and ordering use exact equality. Packet-energy admission and command-energy affordability use their stated exact comparisons; all other finite real comparisons use absolute tolerance `1e-6`.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite root-level `results.json` from the project code and fixed conditions above. The evaluator deletes an old result before one official run.

`report.md` must contain these non-empty Markdown headings exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, reusable method, actual experiment, evidence-based analysis, and limitations. Optional controlled-experiment text or JSON artifacts are allowed but must not replace or add fields to the canonical `results.json` contract.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run succeeds within the limit and creates a fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the declared root and scenario container structure", weight: 3}
- {id: result_evidence, desc: "state, command, allocation, prediction, packet, and queue evidence has the declared form", weight: 4}
- {id: canonical_coverage, desc: "all scenarios and immutable state, action, allocation, opportunity, and queue units have exact coverage", weight: 4}
- {id: decision_legality, desc: "shares, prediction trajectories, commands, energy, and prediction references are legal", weight: 4}
- {id: replay_validity, desc: "allocation timing, packet admission and service, outcomes, motion, collision, energy, and task events match replay", weight: 6}
- {id: primary_metrics_1, desc: "replayed deadline target and collaboration-gate task quality", weight: 8}
- {id: primary_metrics_2, desc: "replayed formation, clearance, and prediction validity quality", weight: 4}
- {id: primary_metrics_3, desc: "progress-gated safety, smoothness, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "effective prediction support relative to admitted communication", weight: 3}
- {id: auxiliary_metrics_2, desc: "effective prediction freshness, queue waiting, and demand-share balance", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the dynamic rescue, prediction, spectrum, formation, and control formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent formulation or a materially wrong process; Implementation/evidence: no executable communication and formation-control simulation."}
    - {score: 0.4, desc: "Report: only scheduling or control is described and prediction age, delayed window effect, queues, or information boundaries are missing; Implementation/evidence: state or event order is materially inconsistent."}
    - {score: 0.7, desc: "Report: slow allocation, fast control, prediction, directed links, formation, obstacles, and deadlines are accurately described; Implementation/evidence: all scenarios realize the main model with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: delayed allocation and prediction validity are precisely connected to the gate, clearance, formation, and terminal progress; Implementation/evidence: states, allocations, packets, trajectories, queues, and motion faithfully support the causal model."}
- id: method_design
  desc: "quality of the unified non-training scheduler, predictor, and formation-control method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, prohibited training, or hidden information; Implementation/evidence: no legal reusable procedure."}
    - {score: 0.4, desc: "Report: fixed shares, scenario/slot tables, non-local true state, or instantaneous-gain greed dominates; Implementation/evidence: decisions are not unified and state-driven or diverge from the description."}
    - {score: 0.7, desc: "Report: a legal, reusable, reproducible allocation, prediction, and safe-control method handles the main timing constraints; Implementation/evidence: one program spans scenarios but long-horizon resource or robustness trade-offs are limited."}
    - {score: 1.0, desc: "Report: a justified reusable method jointly treats urgency, prediction error and age, service opportunity, queue slack, energy, and formation safety; Implementation/evidence: contrasting visible link or collaboration states produce traceable shares, sends, and commands without scenario or time tables."}
- id: experimental_coverage
  desc: "coverage of canonical rescue conditions and controlled communication comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one allocation or control condition is summarized and communication is not tested as a cause of collaboration results; Implementation/evidence: condition records are incomplete."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one explicitly identified comparable controlled condition test prediction or allocation; Implementation/evidence: condition-level allocation, packet, queue, prediction, trajectory, and metric records are available."}
    - {score: 1.0, desc: "Report: non-redundant single-factor comparisons test prediction or spectrum allocation and a link-asymmetry, burst, or gate-deadline boundary; Implementation/evidence: condition-level records reconstruct mechanism and failure boundary."}
- id: research_analysis
  desc: "evidence-based analysis of spectrum, prediction age, queues, formation safety, and deadlines"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or materially contradictory analysis; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: lists delay or completion values without causal explanation or limitations; Implementation/evidence: records do not support the claims."}
    - {score: 0.7, desc: "Report: uses replayed results to explain allocation, gain, prediction, formation, and deadline trends, distinguishes observation from inference, and states limitations; Implementation/evidence: reported comparisons are supported."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies the interaction of shares, prediction age, backlog, collaboration safety, and progress, tests explanations against contrasting or boundary evidence, and limits conclusions; Implementation/evidence: reproducible condition-level evidence supports the central findings."}
