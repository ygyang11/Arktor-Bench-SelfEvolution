---
id: T053_submodular_connected_tracking
name: Communication-Aware Submodular Target Tracking with Epoch Connectivity
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

Three mobile robots observe moving targets on a grid. Each robot executes an eight-slot motion primitive, collects local binary detections, and may exchange observation summaries during two short wireless slots at the end of every epoch. A robot can therefore improve its own next decision from local observations while the team benefits only when a summary reaches another robot. The targets move between epochs, obstacles can split the map, endpoint links can disappear, and each robot has a finite observation queue. Purely independent target chasing can overlap observations or leave the endpoint graph disconnected; a fixed meeting point can waste sensing range and make summaries stale.

### Research Problem

Build a deterministic simulator and a non-training online method that chooses one motion primitive per robot and local FIFO-prefix transmission proposals. The method should maximize the fixed submodular coverage of active targets, maintain endpoint connectivity when useful, and deliver fresh observations for cross-robot fusion under packet loss, queue capacity, hop and packet budgets, and collision and energy costs. Use one reusable state-driven procedure for all canonical scenarios. The deterministic scenario tables and formulas below are public task inputs. A method may calculate or forward-simulate quantities determined solely by those inputs, including target positions, link gates, loss bits, and hash outcomes; this is model prediction rather than access to a hidden runtime record. Decisions must still be emitted in chronological order by one reusable state-driven procedure. A robot's local decision rule may use the public map and task templates, its current position, local queue and history, observations already received, current incident link availability, and the shared remaining packet budget. It must not directly read another robot's private queue or observation history, or import a future event record that is not derived from the public model and the submitted decisions. Deterministic planning, search, optimization, and analytical heuristics are allowed.

### Experimental Setting and Constraints

#### Canonical evaluation units and scenarios

The canonical scenarios are `clustered_targets`, `split_targets`, `narrow_bridge`, `crossing_tracks`, and `link_burst`. Each has 12 epochs. An epoch has ten one-second slots: offsets `0..7` execute a selected primitive and offsets `8,9` are exchange slots. Boundary 120 is terminal. The grid is `0<=x<20, 0<=y<16`; robots start at `(2,4)`, `(2,8)`, and `(2,12)`. Obstacles are closed integer rectangles: `split_targets` has `[7,8]x[4,11]`; `narrow_bridge` has `[8,11]x[0,6]` and `[8,11]x[9,15]`; `crossing_tracks` has `[9,10]x[0,5]` and `[9,10]x[10,15]`; the other scenarios have no obstacles.

#### State, actions, and task model

Primitive IDs are fixed eight-step delta sequences. IDs `0..8` are respectively `(0,0)`, `(1,0)`, `(-1,0)`, `(0,1)`, `(0,-1)`, `(1,1)`, `(1,-1)`, `(-1,1)`, and `(-1,-1)` repeated eight times. IDs `9..16` are the four repetitions of `(1,0)` then `(0,1)`, `(1,0)` then `(0,-1)`, `(-1,0)` then `(0,1)`, `(-1,0)` then `(0,-1)`, `(0,1)` then `(1,0)`, `(0,1)` then `(-1,0)`, `(0,-1)` then `(1,0)`, and `(0,-1)` then `(-1,0)`. Every intermediate cell must be inside the grid and outside obstacles. A nonzero move costs one energy and a stay costs `0.05`; each robot starts with 110 energy. Each robot proposes a next cell at every primitive step, including its unchanged current cell for a stay. If two or more proposed next cells are equal, all robots sharing that cell are involved and all their moves are rejected; thus a move into a stationary robot's cell rejects both robots. A simultaneous exchange of cells is also rejected for both robots. Each involved robot records a collision robot-slot while remaining at its previous cell.

#### State, targets, observations, and official coverage

Targets are `T00..T11`. Their scenario-specific initial positions are:

| scenario | positions in `T00..T11` order |
|---|---|
| `clustered_targets` | `(7,4),(8,5),(9,4),(10,5),(11,4),(12,5),(7,10),(8,11),(9,10),(10,11),(11,10),(12,11)` |
| `split_targets` | `(5,2),(5,4),(5,6),(5,8),(5,10),(5,12),(15,2),(15,4),(15,6),(15,8),(15,10),(15,12)` |
| `narrow_bridge` | `(5,3),(5,5),(5,7),(5,9),(5,11),(5,13),(14,3),(14,5),(14,7),(14,9),(14,11),(14,13)` |
| `crossing_tracks` | `(4,2),(6,3),(8,4),(10,5),(12,6),(14,7),(14,13),(12,12),(10,11),(8,10),(6,9),(4,8)` |
| `link_burst` | `(6,3),(7,4),(8,5),(9,6),(10,7),(11,8),(12,9),(13,10),(14,11),(15,12),(8,12),(12,3)` |

Target value is `1+((target_index+scenario_index) mod 4)`, where scenario indices follow the order above. All targets are active in all epochs. At the start of epoch `e`, target positions are obtained from the preceding position by applying the following displacement and retaining the old position if the candidate cell is invalid. The listed target starts may be obstacle cells: obstacles constrain robot cells and the five interior line-of-sight sample cells, not the target endpoint. A target motion candidate is valid only when it is inside the grid and outside obstacles; if invalid, the preceding target position is retained, even when that old position is an obstacle cell. For `clustered_targets`, only odd `e` moves toward x=9 when `e mod 4=1` and away when `e mod 4=3`. For `split_targets`, `dy=+1` when `(j+e) mod 2=0` and `dy=-1` otherwise. For `narrow_bridge`, residues `0,1` move toward x=8 and residues `2,3` move away. For `crossing_tracks`, targets `j<6` use directions `(+1,+1),(-1,+1),(-1,-1),(+1,-1)` cyclically and targets `j>=6` use the reverse sequence. For `link_burst`, every three epochs use direction index `(floor(e/3)+j) mod 8` in the ordered list `(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)`.

After each action, robot-target distance is measured. For line-of-sight, evaluate the five equally spaced interior points at fractions `1/6,2/6,...,5/6`; round each coordinate to the nearest integer, with exact half-integers rounded toward the lower integer. If any rounded interior cell is an obstacle, detection probability is zero; otherwise `p=max(0,1-distance/6)`. The binary outcome is one exactly when `(7919*(scenario_index+1)+104729*(epoch+1)+1009*(offset+1)+31*(target_index+1)+17*(robot_index+1)) mod 10000 < floor(10000*p)`. The outcome is revealed only to that robot at the end of the slot. The official epoch coverage is `F(e)=sum_j value_j*(1-product_(robot,offset)(1-p_robot,j,e,offset))/sum_j value_j`, using all positive-probability opportunities, not the candidate's private estimate.

#### Information availability and event order

At boundary `10*e` before offset 0, each robot commits its `primitive_id` for epoch `e`; all eight primitive steps are immutable through offsets `0..7`. At every slot boundary, process arrivals from the preceding slot, then expose current links and make the decisions allowed at that boundary. For offsets `0..7`, execute the committed primitive step and generate observations after movement. For offsets `8,9`, validate and arbitrate proposals, then schedule one-slot arrivals. The successful offset-9 packet of slot 119 arrives at terminal boundary 120; it may enter the destination FIFO but cannot start another decision or receive fusion credit.

The fixed scenario model is public and deterministic. A policy may evaluate its tables and formulas to form a model-based forecast, including future target positions, link gates, loss bits and hash outcomes; such a forecast is not a received runtime event. The private-state boundary concerns direct use of another robot's queue or observation history, or importing a future event record rather than deriving it from the public model and the submitted decisions. In the result contract, `epoch_plans` and `transmission_proposals` are decision records submitted at their commit/exchange boundaries; observations, arrivals, queues, motion, resources and metrics are execution results reconstructed by the grader after those decisions.

#### Legal actions and communication boundaries

Robots exchange only at offsets 8 and 9, using their positions after offset 7. A pair has a base link when its Euclidean distance in grid coordinates `(x,y)` is at most 5 and the five interior points at fractions `1/6,2/6,...,5/6` round to no obstacle cell (nearest integer, ties toward the lower integer). For pair index 0,1,2 in fixed order `R1-R2,R1-R3,R2-R3`, the link gate is one unless `(5*(scenario_index+1)+7*(epoch+1)+3*pair_index+2*(offset-8)) mod 11 = 0`. In `link_burst`, at `epoch mod 3=1, offset=9`, only pairs `R1-R2` and `R2-R3` use gate one; `R1-R3` is blocked. Availability is the conjunction of base link and gate. Only current incident availability is visible to a robot.

Each positive binary detection creates one observation token with ID `o_<robot>_T<target:02d>_e<epoch:02d>_r<offset>`, target value as its fusion value, and its generation slot. The token is placed at the detecting robot's FIFO, whose capacity is 12; admitting a new token drops the oldest token when necessary. A proposal may be submitted only at offset 8 or 9, names a distinct rover destination, and carries a nonempty FIFO prefix of one to four observation IDs. Its priority is an integer `0..9`. A selected proposal removes its prefix and sends it to the next boundary; a deterministic loss bit is one when `(3571*(scenario_index+1)+101*(epoch+1)+17*(offset-8)+13*pair_index+7) mod 19` belongs to the scenario loss set: `{0}` for `clustered_targets`, `{0,1}` for `split_targets` and `narrow_bridge`, `{0,1,2,3}` for `crossing_tracks`, and `{0,1,2}` for `link_burst`. A lost prefix is returned to the source FIFO; a delivered prefix is appended at the destination after removing duplicate IDs, dropping the oldest existing token first if the destination FIFO is full. Each observation has a hop count initialized to zero; a delivered packet increments the count and an observation at hop count two cannot be selected for another packet. Let `observation_slot` be the generation slot, `service_slot` the exchange slot in which a proposal is admitted, and `arrival_slot=service_slot+1` the boundary where its packet is processed. Delivery age is `arrival_slot-observation_slot`; the first delivery to another robot receives fusion credit exactly when this age is at most 20, except that terminal boundary 120 never grants fusion credit. Later or duplicate delivery has no credit. At most one proposal is selected per exchange slot because each selected proposal uses two distinct endpoints and there are three robots; if multiple valid proposals exist, choose the one in descending `priority` then Unicode `(src,dst,first_observation_id)` order. At most 24 packets are selected per scenario; later proposals remain well formed but are not selected.

#### Task metrics

All five metrics are averaged over scenarios. `primary_metrics_1` is the mean of official `F(e)` over all 60 scenario-epochs. `primary_metrics_2` is `0.55*connected_fraction+0.45*fused_value_fraction`, where an epoch is connected if the union of its two exchange-slot graphs connects all three robots, and `fused_value_fraction` is the target-value sum of unique observations first fused within 20 slots divided by the value sum of all generated observations. `primary_metrics_3` is coverage multiplied by `0.4*clip(1-moves/288,0,1)+0.3*clip(1-collisions/288,0,1)+0.3*clip(1-energy/330,0,1)`. `auxiliary_metrics_1` is `0.6*useful_bits/max(1,sent_bits)+0.4*fusion_packets/max(1,selected_packets)`, with 48 useful bits per first useful observation and 256 bits per selected packet. Finally, `auxiliary_metrics_2=0.4*fresh+0.3*redundancy+0.3*wait`, where fresh and wait are the means of `clip(1-(arrival_slot-observation_slot)/20,0,1)` and `clip(1-(service_slot-observation_slot)/20,0,1)` over fused observations, and redundancy is one minus duplicate delivered observations divided by selected packets. Empty sample terms are zero. All values are clipped to `[0,1]` and are computed only after every validity gate passes.

#### Result recording contract

`results.json` has exactly one root key `scenarios`, an array with exactly one record per scenario; scenario order is arbitrary. Each record has exactly `scenario`, `epoch_plans`, and `transmission_proposals`:

- `scenario` is one of the five canonical strings.
- `epoch_plans` has exactly 36 rows, each exactly `{robot,epoch,primitive_id}` with a rover string,  integer `epoch` in `0..11`, and integer `primitive_id` in `0..16`. Each `(robot,epoch)` occurs once.
- `transmission_proposals` is dynamic. Each row exactly `{slot,src,dst,observation_ids,priority}`;  `slot` is an exchange slot in `0..119`, `src` and `dst` are distinct rover strings,  `observation_ids` is a one-to-four list of distinct canonical IDs matching  `o_<robot>_T<target:02d>_e<epoch:02d>_r<offset>`, with robot `R1..R3`, target and epoch `00..11`,  and offset `0..7`; `priority` is an integer `0..9`. Each `(slot,src)` occurs at most once.  Absence means no proposal. Array order has no  meaning. The grader independently derives targets, observations, links, packets, queues, fusion,  collisions, energy, and all metrics; candidates do not submit derived state or metric tables.

JSON must be strict: duplicate keys, `NaN`, `Infinity`, and booleans used as integers are invalid.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain the six non-empty Markdown sections `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Controlled comparisons are optional for the validity gate but support higher Judge levels; any reproducible condition-level representation is accepted.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "epoch plans and transmission proposals have the declared forms", weight: 4}
- {id: canonical_coverage, desc: "all scenarios and robot-epoch plan keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "primitive IDs, plan keys, proposal keys, endpoints, and priorities are legal", weight: 4}
- {id: replay_validity, desc: "independent replay preserves motion, observations, links, FIFO, fusion, and resource invariants", weight: 6}
- {id: primary_metrics_1, desc: "independent submodular target coverage across scenario-epochs", weight: 8}
- {id: primary_metrics_2, desc: "endpoint connectivity and timely cross-robot fusion", weight: 4}
- {id: primary_metrics_3, desc: "coverage-conditioned motion, collision, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful observation bits and productive packet efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "fusion freshness, duplicate suppression, and queue waiting", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the submodular observation, endpoint connectivity, and fusion model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent target-observation or communication model; Implementation/evidence: no executable process."}
    - {score: 0.4, desc: "Report: target motion, primitive, local observation, endpoint link, queue, or fusion timing is decisively incomplete; Implementation/evidence: state or event order materially diverges."}
    - {score: 0.7, desc: "Report: local observations, fixed coverage, primitives, endpoint graph, FIFO summaries, losses, and next-epoch fusion are accurate; Implementation/evidence: all scenarios implement the core model with limited noncritical omissions."}
    - {score: 1.0, desc: "Report: event timing, marginal coverage, collision, endpoint connectivity, summary age, and belief/fusion consequences are precisely connected; Implementation/evidence: records faithfully realize the public model and information boundary."}
- id: method_design
  desc: "quality of the unified primitive and summary scheduling method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited method; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: fixed endpoint or independent greedy rules ignore marginal gain, links, queues, or age; Implementation/evidence: decisions are scenario-specific or inconsistent."}
    - {score: 0.7, desc: "Report: legal unified target, primitive, endpoint, and FIFO-summary method addresses the main epoch trade-offs; Implementation/evidence: one procedure covers all scenarios with limited long-horizon handling."}
    - {score: 1.0, desc: "Report: evolving local evidence drives marginal coverage, endpoint graph, queue age, loss, packet budget, collision, and energy choices; Implementation/evidence: no scenario/epoch answer tables and decisions match the method."}
- id: experimental_coverage
  desc: "coverage of target-distribution, endpoint, and communication comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only canonical aggregates are shown and endpoint connectivity or fusion is not tested; Implementation/evidence: condition evidence is insufficient."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled target-density, communication, or epoch comparison are executed with checkable records; Implementation/evidence: the comparison has condition-level outputs that can be traced to the reported claim."}
    - {score: 1.0, desc: "Report: complementary single-factor comparisons test overlap/endpoint mechanisms and a link, loss, age, or obstacle boundary; Implementation/evidence: condition-level records support both claims beyond aggregate scores and identify the tested boundary."}
- id: research_analysis
  desc: "evidence-based interpretation of marginal coverage and communication freshness"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: claims are unsupported."}
    - {score: 0.4, desc: "Report: values are repeated without explaining overlap, endpoint, age, or fusion consequences and limits; Implementation/evidence: records do not support the claims."}
    - {score: 0.7, desc: "Report: actual records explain target distribution, obstacle, link, queue, fusion, and energy trends while distinguishing observation from inference; Implementation/evidence: results support the analysis."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies marginal-gain, endpoint, packet-age, and failure-boundary interactions, tests alternatives, and states scope; Implementation/evidence: reproducible condition records support the conclusions."}
