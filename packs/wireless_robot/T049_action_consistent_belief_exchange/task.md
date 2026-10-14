---
id: T049_action_consistent_belief_exchange
name: Action-Consistent Belief Exchange for Safe Multi-Robot Search
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

Two ground robots search a bounded indoor grid while holding private, noisy observations about targets and hazards. Each robot maintains its own belief and proposes a four-slot macro action for the pair. A finite bidirectional summary channel can reconcile beliefs, but delay, loss, queue capacity and a global packet budget make communication selective. If the two proposals do not form the same observable commitment, both robots execute a safe stay fallback. The central question is how belief-versioned summaries and action-consistent commitments change search value, safety and communication cost.

### Research Problem

Build a self-contained deterministic simulator and a non-training online procedure that, in every planning cycle, updates each robot's mutually exclusive target/hazard belief, ranks the 25 joint macro actions allowed by the public model, chooses proposals and commitments, and decides which private observations to send in each direction. The procedure should obtain useful target confirmation while limiting fallback, unsafe motion, packet attempts and stale information. Use one reusable state-driven procedure for all five canonical scenarios. It may use the current grid and shared simulator feedback, current observations, arrived summaries, ACK history, outgoing-queue state and current outgoing-link availability, but it must not use the other robot's current private state, the target-motion schedule, future observations, target/hazard truth, future loss bits, future actions or a scenario/cycle lookup table. Deterministic rules, search, analytical approximations and planning are allowed; machine learning and training are forbidden.

### Experimental Setting and Constraints

#### Canonical scenarios and world

There are five canonical scenarios in the order shown below. Each uses a `16 x 16` grid with integer coordinates `0<=x,y<16`, 96 primitive slots of `0.5 s`, and 24 planning cycles `k=0,...,23`; cycle `k` controls primitive slots `4k,...,4k+3`. The uncertain-cell set is `U={(x,y):2<=x<=13,2<=y<=12}`. Every cell in `U` has exactly one hidden state: `target`, `hazard`, or `empty`. Initial target and hazard priors are `0.25` and `0.15`. A target identity is its tabled initial cell, and its value is `w(x,y)=1+((3*x+y) mod 4)/3` at that initial cell.

| scenario | starts `(R1,R2)` | closed obstacles | target identities/initial cells | hazard cells | deadline | fixed change |
|---|---|---|---|---|---:|---|
| `crossing_hazard` | `(1,7),(14,7)` | `[6,9]x[0,5]`; `[6,9]x[10,15]` | `(4,6),(11,8),(5,9),(10,5)` | `(5,7),(10,7),(8,6),(7,9)` | 18 | none |
| `split_targets` | `(2,2),(13,13)` | `[5,7]x[5,10]`; `[9,11]x[5,10]` | `(3,4),(12,11),(4,12),(11,3)` | `(4,5),(11,10),(7,4),(8,11)` | 20 | none |
| `narrow_merge` | `(1,3),(1,12)` | `[4,11]x[0,4]`; `[4,11]x[11,15]` | `(3,8),(12,7),(7,6),(8,9)` | `(5,8),(10,7),(7,5),(8,10)` | 19 | none |
| `moving_target` | `(2,7),(13,8)` | `[5,8]x[0,4]`; `[8,11]x[11,15]` | `(4,6),(11,9),(6,10),(9,5)` | `(5,6),(10,9),(7,8),(8,7)` | 22 | target identity `(4,6)` moves at the start of cycles 6, 12 and 18 to `(3,6)`, `(3,7)` and `(4,7)` |
| `communication_blackout` | `(1,1),(14,14)` | `[5,10]x[4,6]`; `[5,10]x[9,11]` | `(3,3),(12,12),(3,12),(12,3)` | `(4,4),(11,11),(4,11),(11,4)` | 21 | neither direction starts service in cycles 7, 8 or 9 |

Cells outside the current target and listed hazard cells are `empty`. The obstacle rectangles are closed and occupy their integer grid cells. The moving target keeps identity and value `(4,6)` at all four locations; the old cell becomes empty before observations in each move cycle. Target and hazard truth, the tabled target-motion schedule and signal outcomes are public rules only for the candidate's environment simulator and the grader's replay; none is policy-visible state. The decision procedure receives no target schedule or truth interface. It is called only after the simulator has generated the observations and feedback visible at that decision boundary and must not inspect truth, branch on the published motion schedule, or invoke the signal generator for future information.

#### Deterministic observations and belief update

At the start of cycle `k`, robot `r` (`r=0` for `R1`, `r=1` for `R2`) observes up to the first three cells in lexicographic `(x,y)` order among cells in `U` with Manhattan distance at most two from its current position; if fewer than three such cells exist, it observes all of them. Each observation has two binary signals. With `s_index=0,...,4` in table order, define `u_T=(13*x+7*y+5*k+3*r+s_index) mod 20` and `u_H=(11*x+5*y+3*k+5*r+2*s_index) mod 20`. The target signal is one iff `u_T<17` when the current hidden cell is a target and iff `u_T<2` otherwise; the hazard signal is analogous using `u_H` and current hazard truth. These are deterministic realizations of `P(signal=1|state)=0.85` and `P(signal=1|not state)=0.10`. An observation ID is `obs-R<1-or-2>-<cycle>-<x>-<y>`, and `observed_slot=4*k`.

Each robot stores a mutually exclusive `(target,hazard,empty)` distribution for every cell. Start with `(0.25,0.15,0.60)`. For signals `(s_T,s_H)`, calculate `a_T=p_T L_T(s_T|target)L_H(s_H|not hazard)`, `a_H=p_H L_T(s_T|not target)L_H(s_H|hazard)`, and `a_E=p_E L_T(s_T|not target)L_H(s_H|not hazard)`, then normalize `a`. Replace every component by `max(0.01,component)` and normalize again. An observation is applied to a robot at most once. At a cycle boundary, apply newly arrived packets by `(source_cycle,direction_rank,packet_id)`, where `direction_rank=0` for `R1->R2` and `1` for `R2->R1`, and their listed observation-ID order, then apply new local observations in robot/cell order. Each application uses the receiver's then-current distribution. A robot's `information_version` starts at zero and increments once per newly applied observation, regardless of which cell it concerns.

#### Macro actions, protocol and execution

The primitive action alphabet is `stay,north,east,south,west`, with indices `0..4` and coordinate deltas `(0,0),(0,+1),(+1,0),(0,-1),(-1,0)`, respectively; a joint action ID is `5*index(R1)+index(R2)` in `0..24`. A macro action repeats its two primitive directions for the four slots of one cycle. At the decision boundary each robot submits a `proposal` (a joint-action ID or `null`), a `commitment_id`, and its exact current `information_version`. The shared-evidence version starts at zero and increments once for each distinct observation in a packet when that packet's arrival and ACK become visible. Both robots receive this same version. A null proposal requires a null commitment. A non-null proposal `a` at shared-evidence version `v` requires the exact string `ac:<a>:<v>`, with the decimal integers substituted without whitespace. The protocol sets `confirmed=true` exactly when both proposals are the same non-null ID and both submissions carry that same valid encoded commitment. A confirmed cycle executes that joint action; otherwise both robots execute `stay` as fallback. A current-cycle send cannot affect the predicate. After the predicate is evaluated, both submitted protocol rows and the resulting confirmed/fallback outcome become shared history visible at the next cycle. A robot may therefore react to the other robot's prior submission, but it cannot use the other robot's simultaneous current proposal. Each robot also reports a complete action-ID ranking and a possible-preferred set as method evidence. The grader checks only their published local relations and never compares either with a reference planner or requires the first-ranked action to equal the proposal.

For each primitive slot, first replace an energy-infeasible moving primitive by `stay`, then form both proposed next cells. An out-of-grid, obstacle or current true-hazard proposal blocks only its robot. A direct swap, or equal proposed cells, blocks both robots. Collision checks use the two simultaneous proposed cells; after blocked robots are reset, if exactly one intended move would enter the other robot's retained old cell, that mover is also blocked. Every blocked robot records one safety violation for the slot. A legal moving robot reaches its proposed cell. Every requested moving primitive with at least one unit of energy costs one unit even if safety blocking prevents motion; `stay` and an energy-infeasible request cost zero. Each robot starts with 96 motion-energy units. After each slot, a robot at a current true target cell with its own `p_T>=0.9` can create the first confirmation of that target identity. Confirmation is global per identity across both robots, not per robot; simultaneous candidates are resolved in `R1,R2` order. A global first confirmation in cycle `k<=deadline` earns deadline value; a later global first confirmation remains a trace fact but earns none. The terminal state is after slot 96.

#### Summaries, queues and event order

Each direction (`R1->R2` and `R2->R1`) has one send opportunity per cycle. A requested packet is 192 bits and contains one to four distinct own observation IDs not used in an earlier send request in that direction; each observation uses one of four 40-bit payload positions, unused positions are zero-padded, and the header is 32 bits. Packet IDs are unique, non-empty, scenario-local strings. Each direction has FIFO capacity three, counting a packet in service, and a scenario admits at most 32 requested packets. A request is `dropped` if either limit is already reached; otherwise it is admitted and costs `0.25` aggregate communication-energy units. Admission is the candidate's only packet-level control: all admitted-packet service and retransmission events follow the fixed model.

An idle direction that is not in blackout repeatedly removes queue heads whose next attempt cannot finish by expiry, then starts at most one attempt on the first remaining head during the service phase. An attempt starting in cycle `j` completes at the end of `j+1+((5*j+offset_s+r) mod 3)`, where `r=0` for `R1->R2`, `r=1` for `R2->R1`, and `offset_s=(0,1,2,0,1)` in scenario order. An observation expires at `min(23,observation_cycle+6)` and a packet's expiry is the minimum over its observations. Before starting an attempt, a head whose prospective completion would exceed its expiry becomes `expired`. On completion cycle `j_c`, loss occurs exactly when `(11*j_c+3*r+2*s_index+offset_s) mod 17=0`. A lost packet completed strictly before expiry is automatically appended to the queue tail with its next attempt number; a loss at expiry ends as `expired`. A successful packet is removed and its summary and ACK become visible at the start of cycle `j_c+1`. An arrival at cycle 24 is retained for packet accounting, but no post-terminal observation is applied and no post-terminal ACK changes a decision. At the terminal boundary, an admitted packet that has neither delivered nor expired is `unfinished`.

The delay and loss formulas are available to the simulator for event generation. The policy may use the fixed delay model and completed outcomes already revealed through queue state or ACK history, but it must not call or evaluate the loss formula for a future completion when choosing an action or send.

Each cycle uses this exact order: move the target if scheduled; expose packet arrivals and ACKs and increment the shared-evidence version for their newly common distinct observations; apply their previously unseen observations in the order above; generate and apply local observations; submit evaluations, proposals, commitments and sends; complete attempts due at this cycle's end and apply their loss/requeue/delivery outcome; start at most one new attempt on each now-idle eligible direction using only packets present before current admissions; admit current sends in the order `R1->R2` and then `R2->R1`; execute four primitive slots and update target confirmations, safety and motion energy. Thus new sends are never serviced in their source cycle, while a requeued loss may wait behind existing packets. The initial state is slot 0; each slot record is the post-slot state.

Decision fields represent values submitted at their decision boundary. Observation and feedback fields represent information already generated at that boundary. Replay checkpoints are post-cycle audit evidence and are never policy inputs. At every next-cycle boundary the simulator publishes to both robots the previous cycle's four post-slot joint positions, energies and per-robot blocking flags, the two protocol submissions and outcome, every prior request's admitted/dropped feedback, any global first target-confirmation event, the global admitted-packet count and remaining budget, and the current shared-evidence version. All robots also know the static grid, starts, priors, action alphabet, deadlines and current joint positions and energies. A robot additionally sees its own belief, its own current and past observations, newly arrived summaries and ACKs, its outgoing FIFO (including an in-service head) and current outgoing-link availability. It cannot see the other robot's current belief, private observations or queue, unarrived observations, hidden truth, target-motion schedule, future observations, future loss results or future actions. Candidate checkpoints do not determine replayed states or metrics; the grader independently derives those values and compares the submitted audit evidence with them.

#### Metrics

For scenario `s`, let `V_s` be the sum of initial-identity target values first confirmed in cycles no later than its deadline and `V_all,s` the sum over all four target identities; set `d_s=V_s/V_all,s`. At the end of cycle 23 let `pbar_s(c)` be the mean target probability of the two robots, `h(p)=-p log2(p)-(1-p)log2(1-p)`, and `H0=sum_(c in U) h(0.25)`. Define `R_s=sum_(c in U) max(0,h(0.25)-h(pbar_s(c)))` only for cells whose threshold classification `pbar_s(c)>=0.5` matches current cycle-23 target truth, and `e_s=clip(R_s/H0,0,1)`. `primary_metrics_1` is the five-scenario mean of `0.6*d_s+0.4*e_s`.

Let `confirmed_s` be the number of confirmed non-fallback cycles and `viol_s` the number of robot-slot safety violations. Set `cons_s=confirmed_s/24` and `safe_s=1-viol_s/192`. `primary_metrics_2` is the mean of `0.6*cons_s+0.4*safe_s`. Let `reqmove_s` be the number of energy-feasible requested moving primitives, `f_s` the number of fallback cycles and `euse_s=reqmove_s+0.25*admitted_packets_s`. `primary_metrics_3` is the mean of `0.35*(1-reqmove_s/192)+0.35*(1-f_s/24)+0.30*(1-euse_s/200)`, with every component clipped to `[0,1]`.

Communication usefulness is derived without candidate labels. A delivered observation is useful if the receiver had not applied it before arrival and, from its arrival cycle until before the next later cycle in which that receiver applies a newly delivered summary observation, either (a) a cycle is confirmed, non-fallback and its joint action has at least one non-`stay` component, or (b) that receiver first confirms a target identity by the deadline. Associate the earliest such event with the observation. Let `D_s` be distinct delivered observation IDs and `G_s` the useful subset. Let `B_s=192*admitted_packets_s` and `U_s=40*|G_s|`. `auxiliary_metrics_1` is the mean of `0.5*clip(U_s/B_s,0,1)+0.5*|G_s|/|D_s|`; its scenario value is zero if `B_s=0` or `D_s` is empty.

For every useful observation, associate its earliest delivering packet and the first qualifying cycle above. Let `Q_s` be the distinct associated packets and set `age_q` to the minimum qualifying-cycle minus arrival-cycle difference among that packet's useful observations. Define `fresh_s=mean_(q in Q_s) clip(1-age_q/12,0,1)`, `attempt_s=clip(1-max(0,total_attempts_s-admitted_packets_s)/admitted_packets_s,0,1)`, and define `recover_s` as follows. For each maximal consecutive fallback run `[k_0,k_1]`, consider the first confirmed non-fallback cycle, if any, in `k_1+1,...,min(23,k_1+12)`. The run is recovered exactly when at least one useful observation associated with that cycle was delivered at a cycle boundary `k>=k_0`; each run contributes at most once. `recover_s` is the recovered-run fraction and is zero when there is no fallback run. For target-based usefulness, the receiver itself must produce the global first confirmation of that identity. `auxiliary_metrics_2` is the mean of `0.4*fresh_s+0.3*attempt_s+0.3*recover_s`; its scenario value is zero when no packet is admitted or `Q_s` is empty. All ratios use exact replay counts and finite values are clipped to `[0,1]`.

#### Result contract

`results.json` is a JSON object with exactly `scenarios`, an array containing the five scenario records. Scenario array order is not significant; `scenario` is its unique canonical key. Each record has exactly `scenario`, `action_evaluations`, `communication_opportunities`, `protocol_submissions` and `replay_checkpoints`.

`action_evaluations` contains exactly one row for every `(robot,cycle)`. Each row has exactly `robot` (`R1` or `R2`), integer `cycle` in `0..23`, `ranked_joint_action_ids`, and `possible_preferred_ids`. `ranked_joint_action_ids` is a permutation of `0..24` in decreasing preference order. `possible_preferred_ids` is a nonempty, strictly increasing array of distinct joint-action IDs and contains the first ranked ID. Neither that first ID nor the possible set must equal the submitted proposal. No utilities, ranks, preferred booleans or other derived evaluation fields are required or accepted.

`communication_opportunities` contains exactly one row per `(cycle,direction)`. Each row has exactly `cycle`, `direction`, boolean `send`, `observation_ids`, `packet_id`, and string `trigger_reason`. A wait has an empty ID array, null packet ID and any string reason. A send has a unique non-empty packet ID, one to four distinct own observation IDs in lexicographic ID order, and a non-empty free-text `trigger_reason`. The IDs must already be visible to the sender by that cycle and must not occur in another requested packet in the same direction. A request may legally be dropped by the fixed capacity/budget rule.

`protocol_submissions` contains exactly one row per `(robot,cycle)`. Each has exactly `robot`, `cycle`, `proposal`, `commitment_id`, and `information_version`. `proposal` is a joint-action ID or null, `commitment_id` is a non-empty string or null, and the version equals the fixed-model version visible to that robot. The proposal/commitment relationship must use the encoding defined above.

`replay_checkpoints` contains exactly one row for every integer boundary `b=0,...,24`. Boundary 0 is the initial state. Boundary `b>0` is after all communication events, admissions and four primitive slots of cycle `b-1`, but before the target move, arrivals and observations of cycle `b`; boundary 24 is terminal and has no subsequent observation update. Each row has exactly `boundary`, `R1`, `R2`, `queue_R1_to_R2`, `queue_R2_to_R1`, `confirmed_target_ids`, `safety_violation_count`, `fallback_count`, `admitted_packet_count`, and `shared_evidence_version`. Each robot object has exactly integer `x,y` in `0..15`, finite `energy` in `[0,96]`, integer `information_version` in `0..200`, and finite `target_probability_sum,hazard_probability_sum` in `[0,132]`, the sums of that robot's 132 cell probabilities. Each queue is the packet-ID array in FIFO order and includes its in-service head. Confirmed target IDs are distinct two-integer arrays denoting tabled initial target cells for that scenario; their array order is not significant. `safety_violation_count`, `fallback_count`, `admitted_packet_count`, and `shared_evidence_version` are integers in `0..192`, `0..24`, `0..32`, and `0..128`, respectively. Checkpoint real values use absolute tolerance `1e-6`; all other checkpoint values are exact.

Scenario records, action evaluations, communication opportunities, protocol submissions and replay checkpoints may use any array order except where an array's order is itself defined above. The grader indexes them by their published canonical keys. JSON numbers must be finite; booleans are not integers; all fixed-model real comparisons use absolute tolerance `1e-6`. The grader derives every observation, belief, queue, packet event, protocol result, executed action, state transition and metric from the decisions, then uses checkpoints only as independent consistency evidence. It does not use checkpoint values to compute metrics or compare candidate-selected decisions and method evidence with a reference policy.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain the six non-empty Markdown sections `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The exact `results.json` contract above is reserved for the canonical evaluation and must not be extended with controlled-experiment fields. Controlled comparisons are optional for the validity gate but provide evidence for the higher Judge levels below. Such evidence may be represented in `report.md`, reproducible project code, and optionally additional text or JSON project artifacts. Any clear condition-level representation is accepted; no additional filename or schema is required.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "evaluation, request, protocol, and replay-checkpoint fields have the declared forms", weight: 4}
- {id: canonical_coverage, desc: "every scenario, decision record, and replay boundary occurs exactly once", weight: 4}
- {id: decision_legality, desc: "rankings, requests, encoded commitments, and visible-information relations are legal", weight: 4}
- {id: replay_validity, desc: "submitted checkpoints match independently replayed belief, queue, protocol, motion, safety, and target state", weight: 6}
- {id: primary_metrics_1, desc: "deadline target value and truth-aware belief entropy reduction", weight: 8}
- {id: primary_metrics_2, desc: "confirmed non-fallback action and safe-execution fractions", weight: 4}
- {id: primary_metrics_3, desc: "requested-motion, fallback, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "delivered summaries that support a later task-active event", weight: 3}
- {id: auxiliary_metrics_2, desc: "useful-summary freshness, retransmission burden, and fallback recovery", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the private-belief, summary, commitment, and safe-search formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent model; Implementation/evidence: no executable belief and protocol process."}
    - {score: 0.4, desc: "Report: belief versions, summary timing, or safety is omitted; Implementation/evidence: the state or event process is materially inconsistent."}
    - {score: 0.7, desc: "Report: local observations, belief updates, proposals, arrival, confirmation, and fallback are accurate; Implementation/evidence: all scenarios implement the core model."}
    - {score: 1.0, desc: "Report: private information and versioned exchange are precisely connected to search and safety; Implementation/evidence: model, information boundary, protocol, and generated records are fully traceable."}
- id: method_design
  desc: "quality of the unified non-training belief, action, commitment, and summary policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable or legal method; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: always/no communication or fixed triggers ignore budget, version, or fallback; Implementation/evidence: decisions are unit-specific or inconsistent."}
    - {score: 0.7, desc: "Report: a legal unified belief, action, commitment, and trigger method treats the main constraints; Implementation/evidence: the same procedure covers every unit with limited planning depth."}
    - {score: 1.0, desc: "Report: temporal information, resource, commitment, and safety consequences are jointly justified; Implementation/evidence: decisions respond coherently to evolving visible state without scenario or cycle answer tables."}
- id: experimental_coverage
  desc: "coverage of canonical searches and controlled inconsistent-belief communication comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one incomplete condition is given; Implementation/evidence: communication and inconsistent-belief claims lack evidence."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled observation, communication, macro-length, or blackout comparison are covered; Implementation/evidence: both conditions use comparable inputs and retain condition-level records."}
    - {score: 1.0, desc: "Report: complementary mechanism and failure-boundary comparisons isolate belief difference, delay or budget, triggers, and safety; Implementation/evidence: reproducible condition records support both comparisons."}
- id: research_analysis
  desc: "evidence-based analysis of belief disagreement, summaries, commitments, and search outcomes"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records cannot support the claims."}
    - {score: 0.4, desc: "Report: agreement or target values are repeated without causal explanation or limitations; Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: conditional observation, trigger, budget, delay, and safety trends and limits are explained; Implementation/evidence: actual records support the analysis."}
    - {score: 1.0, desc: "Report: interactions are quantified and explanations are tested against contrasting evidence or a failure boundary; Implementation/evidence: condition-level records support scoped conclusions."}
