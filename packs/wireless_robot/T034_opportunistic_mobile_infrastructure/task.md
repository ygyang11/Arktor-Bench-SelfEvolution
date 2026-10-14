---
id: T034_opportunistic_mobile_infrastructure
name: Opportunistic Mobile Infrastructure for Deadline-Coupled Robot Missions
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

Three task robots patrol a cluttered site while sending status units to a fixed mission station. Two mobile-infrastructure (MID) robots can reposition and relay those units. A transmission is a broadcast: the sender supplies an ordered list of eligible next-hop candidates, and the first candidate that successfully receives becomes the sole carrier. This opportunistic choice can preserve a route when the preferred link fails, but extra candidates consume the shared link-attempt capacity and simultaneous broadcasts reduce every link's success probability.

Communication has a direct mission consequence. Each task robot has an ordered sequence of goals, and a goal can be completed only while the robot occupies its goal cell, before its deadline, and after enough status units from that robot have reached the station. A robot that outruns its delivered status must wait or return. MID motion can improve delivery, but consumes energy and may leave another urgent flow unsupported. The system must therefore coordinate task motion, MID positioning, queue service, relay candidates, and retransmission under deadlines and congestion.

### Research Problem

Build a deterministic simulator and design one unified, state-driven method for all canonical scenarios. At every action boundary, the method must choose a grid move for each mobile node and may choose one queued status unit to broadcast with an ordered candidate-relay list. The method should use current queues, acknowledged deliveries, node state, flow urgency, link geometry, and relevant history to trade off timely delivery, ordered goal completion, coverage, MID travel, safety, path diversity, and communication load.

The method, approximations, search procedure, and parameters are your choices. The fixed scenarios are evaluation inputs, not permission to supply scenario-indexed paths, relay tables, packet schedules, or result rows. Additional controlled experiments may vary a mechanism or parameter, but the canonical result must use the model below.

### Experimental Setting and Constraints

#### Canonical Evaluation Units

There are five scenarios: `dispersing_patrol`, `crossing_flows`, `support_gap`, `priority_burst`, and `moving_endpoints`. Each uses the integer grid `x in {0,...,23}`, `y in {0,...,17}`, five mobile nodes `t0,t1,t2,m0,m1`, a fixed station `base`, and `H=84` action slots `t=0,...,83`. State boundaries are `b=0,...,84`; slot `t` starts at boundary `t` and ends at boundary `t+1`.

Obstacle rectangles are closed and include every integer cell in the stated Cartesian product. All other cells are traversable. Starts, station cells, and goal cells are traversable. The scenario table fixes the ordered task-robot starts, ordered MID starts, station, obstacles, communication radius `r`, ordered goals `(x,y,value,deadline)`, and flow `(unit count N, priority, flow deadline D)`:

| scenario | task starts `(t0,t1,t2)` | MID starts; station | closed obstacles | ordered goals for `t0`; `t1`; `t2` | flows `t0;t1;t2`; `r` |
|---|---|---|---|---|---|
| `dispersing_patrol` | `(2,2),(2,9),(2,16)` | `(11,2),(11,16)`; `(22,9)` | `[8,9]x[4,13]` | `(6,2,2,28),(10,4,2,42),(16,2,3,58)`; `(6,9,2,30),(12,9,3,48),(18,12,3,66)`; `(6,16,2,32),(14,16,3,54),(20,14,4,72)` | `(6,2,40);(6,2,48);(6,1,56)`; `6` |
| `crossing_flows` | `(2,3),(2,14),(20,3)` | `(10,3),(10,15)`; `(22,9)` | `[9,10]x[6,11]` | `(7,12,2,34),(12,6,3,50),(18,12,3,68)`; `(7,6,2,36),(12,12,3,52),(18,6,3,70)`; `(8,9,2,40),(14,3,3,58),(20,9,4,76)` | `(8,3,52);(8,2,60);(8,1,68)`; `5` |
| `support_gap` | `(3,2),(3,15),(19,9)` | `(4,9),(17,9)`; `(22,9)` | `[7,8]x[3,14]` | `(8,2,2,30),(12,4,3,46),(18,4,3,64)`; `(8,15,2,32),(12,14,3,50),(18,14,4,70)`; `(10,9,2,36),(14,9,3,56),(20,9,4,78)` | `(7,2,50);(7,2,60);(7,3,72)`; `5` |
| `priority_burst` | `(2,2),(2,16),(20,9)` | `(11,4),(11,14)`; `(22,9)` | `[10,11]x[5,13]` | `(6,3,2,26),(10,3,3,40),(16,3,4,58)`; `(6,15,2,28),(10,15,3,42),(16,15,4,60)`; `(12,9,5,34),(18,9,5,46),(21,9,5,62)` | `(10,1,46);(10,1,48);(14,3,54)`; `6` |
| `moving_endpoints` | `(2,5),(2,13),(20,9)` | `(10,5),(10,13)`; `(22,9)` | `[6,7]x[6,12]` | `(7,5,2,30),(13,5,3,48),(19,5,4,66)`; `(7,13,2,32),(13,13,3,50),(19,13,4,68)`; `(11,9,3,36),(16,9,4,54),(21,9,5,76)` | `(8,2,52);(8,2,58);(10,3,70)`; `4` |

Each source has units `t<i>:u<j>` for `j=0,...,N-1`. In all scenarios except `priority_burst`, unit `j` of every source is released at boundary `3j`. In `priority_burst`, `t0` and `t1` unit `j` is released at boundary `2j`, while `t2` unit `j` is released at boundary `18+j`. All released units remain eligible for late delivery through boundary `84` unless the queue rules drop them.

#### State, Actions, and Task Model

Every mobile node starts with energy `120`. An executed four-neighbor move costs `1.0` for a task robot and `1.5` for a MID; an executed stay costs `0.2`. A broadcast for which at least one candidate link is admitted costs its sender `0.5`, independent of the number of admitted candidates. Absolute tolerance `1e-6` is used for energy comparisons: a balance below `-1e-6` is illegal and a balance in `[-1e-6,0]` is treated as zero for comparisons.

Each task robot has a current goal, initially the first in its ordered list. For source flow size `N`, goal indices `k=0,1,2` require respectively `ceil(N/3)`, `ceil(2N/3)`, and `N` distinct units from that source to have reached `base`. At boundary `b`, the current goal completes if the robot occupies its goal cell, `b` is no later than the goal deadline, and its delivery requirement is met. Completion advances to the next goal. A missed current-goal deadline blocks the later goal prefix. Coverage is the set of traversable cells occupied by `t0,t1,t2` at any boundary, including boundary `0`. A traversable cell has coverage value `1`, except a goal cell has the largest goal value assigned to that cell in its scenario.

Every released unit has exactly one live carrier until it is delivered or dropped. Each mobile node has a queue capacity of `6`. At a boundary, existing queued units, successful incoming units, and newly released source units are ordered by `(flow deadline ascending, priority descending, release boundary ascending, source index ascending, unit index ascending)`; the first six are retained and the remainder are dropped. A sender may broadcast at most one queued unit per slot and list one to three distinct candidates. A unit may use at most four successful hops. At one carrier it receives at most two actual broadcast attempts: after two failures at that carrier it is dropped; a successful transfer resets this carrier-attempt count at the new carrier. Delivery to `base` is acknowledged at the arrival boundary.

Let `L_s(p)` be shortest four-neighbor path length from traversable cell `p` to the station in scenario `s`; it is fixed by the public grid. A listed relay candidate must differ from the sender, be at action-start Manhattan distance at most `r`, and have strictly smaller `L_s` than the sender. `base` has distance zero. A mobile relay is allowed only while the unit has fewer than three prior successful hops; `base` is allowed while it has fewer than four. The distance rule enforces progress at the current boundary, but moving nodes may later return a unit to a previous carrier; the four-hop limit bounds such cycling without imposing an unannounced no-revisit rule.

Each candidate entry consumes one shared link-attempt unit. At most six candidate entries are admitted in a slot and at most `504=6H` over a scenario. Candidate entries are ranked by the packet queue order above, then sender order `t0,t1,t2,m0,m1`, then their submitted list index; the first entries within the remaining capacity are admitted. A sender with no admitted candidate does not broadcast, spend energy, or consume one of its two carrier attempts.

For each admitted candidate from sender `i` to receiver `j`, use action-start positions and let `d=||p_i-p_j||_1/r`. Let `o=1` exactly when the shortest traversable four-neighbor distance between their cells is greater than their Manhattan distance, and `o=0` otherwise. If `n` senders have at least one admitted candidate in the slot, the success probability is

`p = clip(0.95 - 0.20d - 0.10o - 0.05(n-1), 0.05, 0.95)`.

For unit `q`, sender `i`, candidate `j`, prior successful-hop count `h`, and zero-based attempt count `a` at the current carrier, define

`u = int(SHA256(f"{scenario}|{slot}|{q}|{i}|{j}|{h}|{a}").hexdigest()[:16],16) / 2^64`.

The candidate succeeds exactly when `u<p`. Among admitted candidates that succeed, the first in submitted list order becomes the sole receiver. A successful mobile transfer removes the sender copy and arrives at the relay at boundary `t+1`; a successful `base` transfer completes the unit at that boundary. If no admitted candidate succeeds, the copy remains at the sender unless this was its second carrier attempt. The deterministic samples are replay facts revealed after the action; a compliant online policy must not branch on a future sample before the corresponding transmission.

#### Information Availability and Event Order

The unified decision procedure receives the complete action-start state produced at the preceding boundary: all mobile positions and energies, current queue contents and packet metadata, acknowledged deliveries, goal prefixes, prior moves, prior admitted attempts, and prior link outcomes. It also knows all fixed scenario geometry, goals, release rules, and formulas. It does not receive future link samples, future actions, future queue outcomes, or future task state. A method may maintain estimates or forecasts from visible history, but may not replace the online process with scenario-, slot-, flow-, or node-indexed pre-filled decisions.

Boundary `0` initializes positions, energy, coverage, empty queues, and then releases boundary-0 units. At every slot `t`, the authoritative order is:

1. expose the complete state at boundary `t` and read all five action records;
2. validate queued send requests and relay candidates using action-start state;
3. admit candidate entries, determine concurrency, and evaluate admitted links using action-start positions;
4. resolve and execute all moves, debit executed motion/stay and admitted-broadcast energy, and create boundary `t+1` positions;
5. at boundary `t+1`, apply the selected packet transfers or failures, increment successful hops or carrier attempts, process delivery ACKs and drops, release units whose release boundary is `t+1`, and apply the deterministic queue-retention order;
6. update coverage and ordered goal completion using boundary `t+1` positions and deliveries, then record histories.

Boundary `84` processes every consequence of slot `83` but has no subsequent decision. The action fields are requests made at the action boundary. Executed moves, safety replacements, admitted links, probability samples, packet paths, queue contents, deliveries, drops, energy, goals, coverage, and metrics are post-event facts derived by the grader and are not submitted.

#### Legal Actions and Boundaries

Each action move is exactly one of `(0,0),(1,0),(-1,0),(0,1),(0,-1)`. Proposed destinations are resolved in fixed node order `t0,t1,t2,m0,m1`. A move is replaced by stay if its destination is out of bounds, an obstacle, an already accepted lower-order destination, or the action-start position of a higher-order node. Thus all post-motion positions are distinct and pairwise swaps make both participants stay. Each replaced nonzero move is one safety event.

A null `send_unit` requires `relay_candidates=[]`. A non-null value must name a unit currently in that sender's action-start queue and requires one to three distinct eligible node IDs from `{t0,t1,t2,m0,m1,base}`. Candidate-list order is semantic; scenario-record and action-record order is not. A legal submission must keep every node's resulting energy at least `-1e-6`. Capacity overflow, failed links, retry exhaustion, and safety replacement are defined environment outcomes, not illegal decisions. Any malformed move, unavailable packet, ineligible candidate, or negative resulting energy is a decision-legality failure.

#### Task Metrics

All metrics are independently replayed, clipped to `[0,1]`, and averaged equally over the five scenarios.

- `primary_metrics_1` (8 points), deadline-weighted delivery quality. A unit arriving by its flow deadline gets quality `1`; a unit arriving later at boundary `b<=84` gets `0.5*clip(1-(b-D)/max(1,84-D),0,1)`; an undelivered unit gets `0`. Average by unit with its flow priority as weight.
- `primary_metrics_2` (4 points), communication-coupled mission quality. Let `goal_progress` be the value of the completed on-time ordered goal prefixes divided by total goal value. Let `coverage_progress` be visited coverage value divided by total traversable coverage value. The scenario value is `0.6*goal_progress+0.4*coverage_progress`.
- `primary_metrics_3` (3 points), safe support efficiency. The scenario value is `0.4*clip(1-safety_events/(84*5),0,1)+0.3*clip(1-mid_moves/(84*2),0,1)+0.3*clip(1-energy_used/(5*120),0,1)`, where energy use includes executed motion/stay and admitted broadcasts.
- `auxiliary_metrics_1` (3 points), effective route diversity. For each flow with at least two delivered units, form each delivered unit's ordered sequence of intermediate mobile relays, excluding its source and `base`. `diverse_flow_fraction` is the number of flows having at least two distinct such sequences divided by three. `mean_hops` is the mean successful-hop count of all delivered units, or `4` if none are delivered. The scenario value is `0.5*diverse_flow_fraction+0.5*clip(1-(mean_hops-1)/3,0,1)`.
- `auxiliary_metrics_2` (2 points), load and queue efficiency. If `A` is the number of admitted candidate entries, `Q` the number of distinct units dropped by queue overflow or retry exhaustion, and `R` the fixed number of released units, the scenario value is `0.5*clip(1-A/504,0,1)+0.5*clip(1-Q/R,0,1)`.

The normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`. No candidate-reported aggregate enters a metric.

#### Result Record Contract

`results.json` must have the exact root object `{"scenarios":[...]}` and exactly one record for each canonical scenario. Each scenario record has exactly `scenario` and `actions`. Its `actions` array contains exactly `84*5=420` records, uniquely keyed by `(slot,node)` for every slot and node. Each action object has exactly:

```json
{
  "slot": 0,
  "node": "t0",
  "dx": 0,
  "dy": 0,
  "send_unit": "t0:u0",
  "relay_candidates": ["m0", "m1"]
}
```

`slot` is an integer in `0..83`; `node` is one of `t0,t1,t2,m0,m1`; `dx,dy` are JSON integers defining the proposed move; `send_unit` is either JSON null or a canonical unit string; and `relay_candidates` is an array of node-ID strings. All object-key sets are exact. Scenario and action array order is irrelevant. Candidate-list order is the only array order with decision meaning. The contract contains only decisions: all fixed inputs and all execution, link, queue, packet, mission, and metric facts are grader-derived.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite root-level `results.json` from the project code and the fixed conditions above. You may use a reasonable multi-file organization for models, algorithms, and experiments.

`report.md` must contain these non-empty sections with headings written exactly as shown: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, experiments, actual results, evidence-based interpretation, and limitations. Claims about MID placement, opportunistic candidates, congestion, deadlines, and mission effects must be supported by executed condition-level evidence.

Supplementary controlled experiments may write clearly named files other than canonical `results.json`, or may be reproduced by project code and summarized in `report.md`. Auto Checks parse only canonical `results.json`; supplementary evidence must not add scenarios, fields, or records to that exact contract.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON and exact root and scenario containers are present", weight: 3}
- {id: result_evidence, desc: "minimal movement and send-request fields have exact auditable types", weight: 4}
- {id: canonical_coverage, desc: "all scenarios and every slot-node action occur exactly once", weight: 4}
- {id: decision_legality, desc: "moves, queued-unit requests, relay candidates, and resulting energy are legal", weight: 4}
- {id: replay_validity, desc: "motion, links, packets, queues, acknowledgments, goals, and coverage replay completely", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-weighted delivery quality", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed communication-gated goals and task coverage", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed safety, MID motion, and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed successful-path diversity and hop efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed link-attempt load and packet-drop efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the motion, opportunistic forwarding, queue, deadline, and mission-coupling model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent model of robot motion or packet delivery; Implementation/evidence: no executable state process represents the task."}
    - {score: 0.4, desc: "Report: MID placement or forwarding omits a decisive queue, deadline, goal-clearance, or event-order relation; Implementation/evidence: code or records materially depart from the stated process."}
    - {score: 0.7, desc: "Report: nodes, queues, links, ordered relays, deadlines, energy, and communication-gated goals are accurate; Implementation/evidence: the core model runs across all scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: action-start geometry, candidate diversity, congestion, ACK timing, queues, and delivery thresholds precisely explain the communication-mission coupling; Implementation/evidence: code and records faithfully realize every principal relation."}
- id: method_design
  desc: "quality of the unified task-motion, MID-placement, queue-service, and relay-selection method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or a prohibited method; Implementation/evidence: no runnable unified decision procedure."}
    - {score: 0.4, desc: "Report: a fixed midpoint, static relay table, isolated shortest paths, or single-flow priority ignores central temporal trade-offs; Implementation/evidence: decisions are pre-filled or materially inconsistent with the report."}
    - {score: 0.7, desc: "Report: a legal reproducible method handles task motion, MID positioning, queue service, and relays using evolving state; Implementation/evidence: the same procedure generates all units with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified unified method jointly treats flow urgency, support reachability, diversity, congestion, retry state, mission clearance, and energy; Implementation/evidence: evolving state drives decisions without scenario or slot action tables and matches the report."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for MID positioning, opportunistic diversity, congestion, and deadlines"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only canonical aggregates or one uninformative flow example; Implementation/evidence: records do not execute or support the principal claims."}
    - {score: 0.7, desc: "Report: every canonical scenario and one controlled MID-motion, candidate-count, priority, range, capacity, or retry comparison are covered; Implementation/evidence: both comparable conditions retain condition-level actions and outcomes."}
    - {score: 1.0, desc: "Report: complementary comparisons separate MID positioning or candidate diversity from a congestion, range, burst, deadline, or motion failure boundary; Implementation/evidence: at least two comparisons execute with reproducible condition-level packet, queue, mission, and metric evidence."}
- id: research_analysis
  desc: "evidence-grounded interpretation of support motion, route diversity, queues, delivery, and mission effects"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: results cannot support the claims."}
    - {score: 0.4, desc: "Report: delivery totals are repeated without explaining MID position, relay diversity, queues, mission effects, or limitations; Implementation/evidence: evidence is insufficient for the inference."}
    - {score: 0.7, desc: "Report: actual results explain important flow, geometry, congestion, deadline, and mission trends and state limitations; Implementation/evidence: executed records support the comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence and a failure boundary quantify how diversity, MID motion, congestion, and deadlines interact and limit scope; Implementation/evidence: reproducible condition-level evidence supports every central conclusion."}
