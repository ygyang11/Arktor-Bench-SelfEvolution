---
id: T021_probabilistic_connectivity_control
name: Probabilistic Communication-Safe Control for Multi-Robot Navigation
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

Four mobile robots must reach separate goals in a bounded factory workspace while avoiding each other and fixed obstacles. Their required directed communication chain supports coordinated navigation, but link reception changes with robot separation, interference, and an unobserved fading residual. Each robot sees its own state exactly and receives other robots' states only through delayed, lossy broadcasts. Motion that separates the team may improve goal progress while degrading the required links; motion that preserves every conservative coordination link can instead stall at crossings and narrow passages.

The robots may temporarily lower the local control threshold of one designated optional outgoing coordination link each. This relaxation can make a locally feasible maneuver possible, but it neither improves the true channel nor relaxes the required chain or physical collision constraints. Broadcasts refresh neighbor positions but consume a shared finite packet budget, while every outgoing link is monitored independently in every slot. A useful controller must therefore couple safe motion, partial information, fading risk, communication timing, recovery from disconnection, goal deadlines, and limited coordination resources.

### Research Problem

Build the deterministic simulator below and design one unified, non-learning local control procedure for all five canonical scenarios. At each slot, each robot selects a two-dimensional velocity, whether to broadcast one state packet, and whether to start one optional-link threshold-relaxation episode. The procedure may use only that robot's current filtered view and its own history. The four robots' action tuples are simultaneous; a robot cannot inspect another robot's current action or any current post-motion channel outcome.

The method should make goal progress without violating physical safety or its locally checkable link thresholds, use broadcasts to control neighbor-state age, and reserve optional-link relaxations for states where their finite duration and launch limit are useful. Evaluate how interference, delayed neighbor information, probability margin, packet budget, and recovery deadlines alter the connectivity-safety-progress trade-off. Deterministic rules, heuristics, search, planning, or optimization are permitted; algorithm choice and parameters remain yours.

### Experimental Setting and Constraints

#### Canonical scenarios and geometry

There are five scenarios in this fixed order, with zero-based `scenario_index`:

```text
0 opposing_goals
1 fading_funnel
2 narrow_bridge
3 jam_burst
4 dense_targets
```

Each scenario has robots `0,1,2,3`, decision slots `t=0,...,59`, and boundaries `0,...,60`. One slot is `0.5` seconds. Positions are in the closed square `[0,12] x [0,12]` metres. A submitted velocity `v_i,t` is in metres per second and produces the endpoint `p_i,t+1 = p_i,t + 0.5*v_i,t`. Its Euclidean norm must not exceed `0.4`. Only endpoints are tested for the stated workspace, obstacle, and inter-robot constraints; segment intersection between endpoints is not an additional condition.

Obstacles are closed axis-aligned rectangles `(xmin,ymin,xmax,ymax)`. An endpoint on or inside a rectangle is unsafe. Simultaneous endpoints of two robots must be at least `0.75` metres apart. The following table fixes initial positions, goals, inclusive goal-arrival boundaries, and obstacles. Four tuples within one cell are ordered by robot ID. `none` means an empty obstacle set.

| scenario | initial positions | goals | deadlines | obstacles |
|---|---|---|---|---|
| `opposing_goals` | `(2,4.5);(4,4.5);(6,4.5);(8,4.5)` | `(9,7.5);(7,7.5);(5,7.5);(3,7.5)` | `52;48;48;52` | `(5.25,0,5.75,3.5);(5.25,8.5,5.75,12)` |
| `fading_funnel` | `(1.5,2);(1.5,4);(1.5,8);(1.5,10)` | `(10.5,4);(10.5,5);(10.5,7);(10.5,8)` | `58;58;58;58` | `(4,0,8,3.2);(4,8.8,8,12)` |
| `narrow_bridge` | `(2,3.5);(2,5);(2,7);(2,8.5)` | `(10,8.5);(10,7);(10,5);(10,3.5)` | `60;58;58;60` | `(4.5,0,7.5,4.2);(4.5,7.8,7.5,12)` |
| `jam_burst` | `(2,3);(2,5);(2,7);(2,9)` | `(10,3);(10,5);(10,7);(10,9)` | `56;56;56;56` | `(5,4.2,7,7.8)` |
| `dense_targets` | `(4.5,4.5);(5.5,4.5);(4.5,5.5);(5.5,5.5)` | `(7.5,7.5);(6.5,7.5);(7.5,6.5);(6.5,6.5)` | `36;34;34;36` | `none` |

A robot first reaches its goal at boundary `b` when `||p_i,b-goal_i||_2 <= 0.3`. Reaching by the listed boundary is timely. Physical safety is a separate hard legality requirement and is not counted again in the timely-arrival metric. The simulator continues through slot `59` after a goal is first reached; the method may hold position or continue moving legally.

#### Directed links, interference, and fixed channel truth

Every ordered pair `(i,j)` with `i != j` is a directed sender-to-receiver link, giving 12 links per slot. The six required links are `(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)`. The four optional coordination links are `(0,2),(1,3),(2,0),(3,1)`; `(0,3)` and `(3,0)` are monitored optional links without a local control threshold. Let `L=sqrt(12^2+12^2)` and let `d_i,j,t+1=clip(||p_i,t+1-p_j,t+1||_2/L,0,1)`.

Current interference `j_s,t` is visible to every robot at decision boundary `t`. It is fixed by:

| scenario | interference rule |
|---|---|
| `opposing_goals` | `0.15` for `t mod 20 < 10`, otherwise `0.25` |
| `fading_funnel` | `0.20` for `t<18`, `0.65` for `18<=t<=34`, otherwise `0.35` |
| `narrow_bridge` | `0.25` for `t<24`, `0.45` for `24<=t<=40`, otherwise `0.30` |
| `jam_burst` | `0.15` for `t<18`, `0.85` for `18<=t<=25`, `0.55` for `26<=t<=31`, otherwise `0.20` |
| `dense_targets` | `0.30` for even `t`, otherwise `0.40` |

The environment-only fading residual and reception event are deterministic:

```text
q_s,i,j,t = ((37*(s+1)+19*(i+1)+11*(j+1)+23*(t+1)) mod 101)/50 - 1
u_s,i,j,t = (((29*(s+1)+31*(i+1)+17*(j+1)+43*(t+1)) mod 997)+0.5)/997
```

Here `s=scenario_index`; `q` lies in `[-1,1]` and `u` lies in `(0,1)`. Both are public simulator truth but are unavailable to either policy before that slot's four decisions. They must not be used to select a velocity, broadcast, or relaxation. The previous link monitor value is initialized to `m_i,j,-1=1`. After motion in slot `t`, the true reception probability is

```text
omega_true_i,j,t = clip(0.90 + 0.04*m_i,j,t-1 - 0.34*d_i,j,t+1
                         - 0.35*j_s,t + 0.10*q_s,i,j,t, 0, 1).
```

The link-monitor event is `m_i,j,t=1` exactly when `u_s,i,j,t <= omega_true_i,j,t`, otherwise zero. It occurs for every directed link even if the sender did not broadcast. A link is `reliable` when `omega_true>=0.75`, `vulnerable` when `0.50<=omega_true<0.75`, and `disconnected` otherwise. These classifications do not depend on whether a packet was sent.

#### Local information and predicted link constraints

At boundary `0`, every robot has one initialized neighbor record for every other robot containing that neighbor's exact initial position, generation boundary `0`, and age `0`. At a later boundary `t`, robot `i` knows its own exact `p_i,t`, goal and deadline; all fixed obstacles; current `j_s,t`; the latest successfully received record from each other robot, with its generation boundary and age `t-generation`; its outgoing `m_i,j,t-1` and previous normalized true distance `d_i,j,t` from the link monitor; its active relaxation and launch count; the shared number of packets already sent; and its own prior views and decisions. It does not know another robot's current true position unless that position is the payload in its latest record. It cannot see current decisions, current `q/u`, post-motion probabilities, packet outcomes, future interference, or future channel truth. Scenario names and indices are simulator routing keys and are not policy inputs.

For a candidate velocity, robot `i` predicts its own endpoint and assumes each neighbor remains at the position in its latest received record. With record age `a`, predicted normalized distance `d_pred`, and current interference, its locally checkable probability is

```text
omega_hat = clip(0.96 + 0.03*m_prev - 0.30*d_pred
                  - 0.18*j_s,t - 0.012*min(a,10), 0, 1).
```

Every submitted velocity must give `omega_hat>=0.50` for every required outgoing link. For the one optional coordination link outgoing from each robot it must give `omega_hat>=effective_threshold`, where the normal threshold is `0.66` and the active-relaxation threshold defined below is `0.50`. The two other optional links have no local threshold. These are decision-time local constraints evaluated from exactly the filtered records above; grader legality never substitutes current hidden neighbor truth or `omega_true` for them. The constraints apply even when no packet is sent. Required-link thresholds cannot be relaxed.

#### Broadcasts and optional-link relaxation

Each robot's Boolean broadcast decision requests one `16`-byte packet. The four requests are simultaneous. A scenario has a shared budget of `120` sent packets; the decisions are legal exactly when the number requested in the slot does not exceed the remaining budget. Every legal request is sent after motion and carries the sender's exact endpoint `p_i,t+1` with generation boundary `t+1`. For each receiver `j`, that copy arrives at boundary `t+1` exactly when `m_i,j,t=1`; otherwise it is lost. Thus one sent packet creates three receiver outcomes, and a successful arrival can replace the receiver's prior sender record before decisions at the next boundary. The packet budget counts sender packets, not receiver copies. No packet is sent for an illegal over-budget batch, which remains a legality failure so replay diagnostics can continue.

A relaxation request is `null` or an object with exact keys `receiver` and `duration`. For robot `i`, `receiver` must identify robot `i`'s one optional coordination edge `(i,receiver)`, and `duration` must be integer `1`, `2`, or `3`. A legal request at slot `t` starts the derived episode `RL|i|receiver|t`; the episode is active for decisions in the inclusive slots `t,...,t+duration-1`, including the velocity selected with the launch request. It lowers only that optional edge's local predicted threshold from `0.66` to `0.50`. It does not change `omega_true`, link classification, packet delivery, physical safety, or required-edge recovery.

Each robot may launch at most two episodes per scenario and may have at most one active episode. A request is illegal if its sender already has an active episode, its edge is required, its receiver is invalid, its duration is invalid, or it exceeds the launch count. The episode expires before the decision at `t+duration`; a new request may then start. Because there is at most one active episode per robot, at most four episodes can be active in a slot. Episode IDs and active/expired status are fully derived and are not submitted.

#### Authoritative event order and recovery windows

For each slot `t=0,...,59`, this order is authoritative:

1. Expire relaxation episodes whose last active slot was `t-1`. Deliver successful packet copies sent   in slot `t-1`, update latest neighbor records, and expose the previous link-monitor outcomes and   distances. Boundary `0` instead uses the initialized records and `m_i,j,-1=1`, and initializes the   visible distance as `d_i,j,0=||p_i,0-p_j,0||_2/L`.
2. Reveal current interference `j_s,t` and freeze four separate local views. Each robot is called only   with its own view. The four velocity, broadcast, and relaxation decisions are simultaneous.
3. Validate field domains, speed, relaxation state, shared packet budget, and the local predicted link   thresholds. Start legal relaxation requests before evaluating the submitted velocity's optional   threshold. If one robot fails a field-domain, speed, relaxation-state, or local-link check, replace   only that robot's entire action by zero velocity, no broadcast, and no new relaxation to continue   diagnostic replay. If the otherwise valid simultaneous broadcast requests exceed the shared   remaining budget, suppress every broadcast in that slot but retain all valid velocities and newly   started relaxations. Either case still fails its corresponding validity check.
4. Apply the four simultaneous motions. Evaluate workspace, obstacle, and pairwise endpoint safety.   A safety violation is recorded and fails the physical-safety legality atom; positions remain at the   submitted endpoints so later diagnostics are deterministic.
5. Compute all 12 `omega_true`, link classifications, and link-monitor events from the post-motion   positions. Send the legal broadcast packets and schedule successful receiver copies for boundary   `t+1`. Update the sent-packet count.
6. Update required-link recovery and goal evidence, save all derived state, and advance. Nothing at   slot `t` after step 2 can affect a decision at the same boundary.

Let a required-link failure episode begin at the first slot in which at least one required link is `disconnected` after a slot with no disconnected required link. Slot `0` is compared with an all-nondisconnected initial condition. Its recovery window includes the start slot and the following four slots, capped at slot `59`. It succeeds and closes at the first slot in that window in which all six required links are `reliable`. If that never occurs, it fails at its last window slot. While a window is open, further disconnected links do not start another window. After a failed window, no new window starts until all required links have first become non-disconnected; a later transition back to disconnected then starts a new episode. All window IDs, boundaries, and outcomes are derived.

The submitted fields are decisions at step 2. Neighbor records, estimates, relaxation episodes, positions, true probabilities, link states, packet outcomes, safety events, recovery windows, goal arrivals, and metrics are execution or derived evidence reconstructed by the grader. This event order is the sole timing authority. The process has no randomness. All integers reject Boolean values; floating comparisons use absolute tolerance `1e-6`.

#### Task metrics

All four metrics are independently replayed, lie in `[0,1]`, and are higher-is-better.

`primary_metrics_1` is the equal mean over the 20 robot-scenario pairs of the timely-arrival indicator: first goal arrival is no later than that robot's fixed deadline. Physical safety is assessed only by the hard legality check and is not a second metric fact.

For each scenario, let `A` be the number of slots in which all six required links are `reliable`, and let `W` be its derived recovery windows. Define `recovery_fraction` as the successful-window fraction when `W` is nonempty, otherwise `A/60`. The scenario communication-recovery quality is `0.6*(A/60)+0.4*recovery_fraction`. `primary_metrics_2` is its equal mean over scenarios.

For a scenario with `B` sent broadcast packets, there are `3*B` receiver copies. Its delivery quality is zero if `B=0`, otherwise the number of successful copies divided by `3*B`. `auxiliary_metrics_1` is the equal mean of this quality over scenarios. It does not reuse the required-link reliable-slot objective.

For a scenario, let initial and final goal distances be `D_i,0` and `D_i,60`, and define

```text
progress_norm = clip(sum_i max(0,D_i,0-D_i,60) / (4*60*0.5*0.4), 0, 1).
smoothness = mean over all robots and slots of
             clip(1-||v_i,t-v_i,t-1||_2/0.8,0,1), with v_i,-1=(0,0).
packet_efficiency = clip(1-B/120,0,1).
```

The scenario action-efficiency value is `progress_norm*(0.6*smoothness+0.4*packet_efficiency)`. `auxiliary_metrics_2` is its equal mean over scenarios. The normalized task performance is

```text
(8*primary_metrics_1 + 7*primary_metrics_2
 + 3*auxiliary_metrics_1 + 2*auxiliary_metrics_2) / 20.
```

#### Result record contract

`results.json` has exact root key `scenarios`, whose value is an array containing exactly one object for each canonical scenario. Every scenario object has exact keys `scenario` and `slots`. Its `slots` array contains exactly one record for every integer `t=0,...,59`, and each slot has exact keys `t` and `actions`. Each `actions` array contains exactly one action for every robot `0,...,3`:

```json
{
  "scenarios": [
    {
      "scenario": "opposing_goals",
      "slots": [
        {
          "t": 0,
          "actions": [
            {
              "robot": 0,
              "velocity": [0.0, 0.0],
              "broadcast": false,
              "relax": null
            }
          ]
        }
      ]
    }
  ]
}
```

Scenario, slot, and action array order is irrelevant. `(scenario,t,robot)` is the canonical action key, giving `5*60*4=1200` minimal action records. `t` and `robot` are non-Boolean integers. `velocity` is a length-two list of finite JSON numbers. `broadcast` is a Boolean. `relax` is either `null` or an exact object whose `receiver` is a non-Boolean integer and whose `duration` is a non-Boolean integer; dynamic edge and episode restrictions are evaluated during replay.

These decisions are the only task-specific result evidence. Do not submit positions, local views, probabilities, links, messages, safety records, episodes, recovery windows, goals, or metrics as substitutes or duplicate fields. The grader derives all of them from the fixed model. JSON must be strict: duplicate object members, `NaN`, and infinities are invalid.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the sole official entry point and must be a thin launcher; model, method, and experiment code may use a reasonable multi-file organization. `run.py` must generate or overwrite root-level `results.json` from the project code and the fixed conditions above. The grader deletes any old result before one official execution and evaluates only the newly generated file.

`report.md` must contain these six exact, nonempty Markdown sections: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the local-information model, reproducible controller, experiments actually run, recorded findings, evidence-based interpretation, and limitations. Additional controlled-experiment evidence may be included in report tables or descriptively named project-generated text or JSON artifacts. Such evidence is for Judge review only and must not alter or replace canonical `results.json`.

## Auto Checks

- id: entry_point
  weight: 2
  desc: "Checks that run.py exists as a regular file and passes static Python compilation."
- id: official_run
  weight: 4
  desc: "Runs python run.py once within 600 seconds and requires a newly generated results.json."
- id: report_contract
  weight: 2
  desc: "Checks report.md for the six required nonempty research sections."
- id: result_json
  weight: 3
  desc: "Checks strict JSON parsing and the exact root and scenario-container structure."
- id: result_evidence
  weight: 4
  desc: "Checks exact slot and robot-action fields, finite velocities, broadcasts, and relaxation requests."
- id: canonical_coverage
  weight: 4
  desc: "Checks unique coverage of five scenarios, 60 slots, and four robot actions per slot."
- id: decision_legality
  weight: 4
  desc: "Checks speeds, physical safety, packet budget, relaxation state, and local predicted link thresholds."
- id: replay_validity
  weight: 6
  desc: "Replays motion, local records, probabilities, messages, relaxations, recovery windows, and goals."
- id: primary_metrics_1
  weight: 8
  desc: "Scores timely goal arrival over all robot-scenario pairs."
- id: primary_metrics_2
  weight: 7
  desc: "Scores required-link reliable time and successful disconnection recovery windows."
- id: auxiliary_metrics_1
  weight: 3
  desc: "Scores successful receiver copies relative to all broadcast receiver opportunities."
- id: auxiliary_metrics_2
  weight: 2
  desc: "Scores goal-distance progress with velocity smoothness and packet efficiency."

## Judge Rubric

- id: problem_modeling
  weight: 8
  desc: "Correctness and precision of the local motion, fading-link, safety, message, and recovery model."
  levels:
    - {score: 0.0, desc: "Report: The formulation is absent or materially wrong. Implementation/evidence: No corresponding local motion, probabilistic link, or safety process is present."}
    - {score: 0.4, desc: "Report: Only part of the probability, neighbor-information, collision, or goal relation is defined and decisive timing is omitted. Implementation/evidence: State or event order materially differs from the report."}
    - {score: 0.7, desc: "Report: Local records, probability estimates, true link events, thresholds, relaxation, collision, and goals are accurately described. Implementation/evidence: The core model operates across all scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: Position, interference, information age, local constraints, true fading, recovery windows, and goal progress are causally and precisely connected. Implementation/evidence: The code faithfully realizes these relations with traceable condition-level evidence."}
- id: method_design
  weight: 15
  desc: "Quality of the unified local risk controller under information, communication, and safety constraints."
  levels:
    - {score: 0.0, desc: "Report: No executable method is given, or a prohibited or information-violating method is used. Implementation/evidence: No legal unified procedure is present."}
    - {score: 0.4, desc: "Report: The method omits decisive neighbor-age, link-risk, collision, packet-budget, relaxation, or recovery consequences. Implementation/evidence: The procedure is materially state-incomplete, uses pre-filled scenario actions, or differs from the stated method."}
    - {score: 0.7, desc: "Report: A legal, unified, reproducible local risk, broadcast, and recovery method addresses the main trade-offs. Implementation/evidence: The same procedure operates across scenarios, with limited prediction depth or robustness."}
    - {score: 1.0, desc: "Report: A justified unified method jointly treats observable history, link margin, neighbor age, collision constraints, packet and relaxation budgets, recovery deadlines, and goal cost. Implementation/evidence: Evolving permitted state drives contrasting actions without scenario or slot action tables."}
- id: experimental_coverage
  weight: 20
  desc: "Execution of controlled experiments on fading risk, local information, communication, and navigation."
  levels:
    - {score: 0.0, desc: "Report: No credible experiment is provided. Implementation/evidence: No checkable experimental execution is present."}
    - {score: 0.4, desc: "Report: Results are limited to canonical aggregates or isolated conditions and do not test the connectivity-motion mechanism. Implementation/evidence: Actual records do not support the principal experimental claims."}
    - {score: 0.7, desc: "Report: All canonical scenarios and at least one controlled interference, threshold, relaxation-budget, message-delay, or goal-conflict comparison are covered. Implementation/evidence: Both conditions are executed on comparable inputs with condition-level link, trajectory, safety, and recovery records."}
    - {score: 1.0, desc: "Report: Complementary non-redundant experiments include a mechanism comparison and a sensitivity or recovery-failure boundary. Implementation/evidence: At least two meaningful controlled conditions have reconstructable probabilities, local states, actions, constraints, and goal metrics."}
- id: research_analysis
  weight: 12
  desc: "Evidence-based explanation of when probability-aware local control improves safe connected navigation."
  levels:
    - {score: 0.0, desc: "Report: Analysis is absent or contradicts the records. Implementation/evidence: The project cannot support the conclusions."}
    - {score: 0.4, desc: "Report: Discussion mainly repeats safety or probability values without local-estimate, relaxation, and motion causality or material limits. Implementation/evidence: Records do not support the asserted interpretation."}
    - {score: 0.7, desc: "Report: Actual records explain conditional trends in reliability, safety, relaxation, recovery, and goal progress while distinguishing observation from inference and stating limitations. Implementation/evidence: Executed comparisons support those claims."}
    - {score: 1.0, desc: "Report: Controlled evidence explains when probability margin, interference, and local information change safety-progress decisions, quantifies principal sensitivities, and bounds the result with contrasting evidence or a failure boundary. Implementation/evidence: Reproducible condition-level records support the central conclusions."}
 
