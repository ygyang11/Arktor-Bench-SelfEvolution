---
id: T017_topology_adaptive_operator_coordination
name: Online Topology Adaptation for Communication-Constrained Multi-Robot Exploration
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

Four mobile robots explore an unfamiliar facility for one operator. A robot learns a map node only by visiting it or receiving its summary. The operator learns discoveries only when robots form a short-range, directed parent topology and forward summaries hop by hop. Maintaining that topology makes its endpoints stationary and consumes energy, while changing a parent takes a full setup slot. During exploration, the operator releases region-priority and sustained-assistance requests. These requests must propagate down the same topology before a target robot can act on them. Exploration, information freshness, request service, topology adaptation, and energy use must therefore be coordinated online.

### Research Problem

Build a deterministic, non-learning simulator and a unified online method for all canonical scenes. At each decision boundary, choose every robot's movement destination, desired parent, and optional map summary to send upward. The method should decide when robots explore independently, form relay chains, change attachment, forward urgent discoveries, or maintain a service path. It may use deterministic search, planning, optimization, or heuristics, but it must respect what has been discovered and which requests have actually been released or delivered. Study the trade-off among timely operator-visible coverage, online request quality, map freshness, and communication/topology cost without scene-specific action schedules.

### Experimental Setting and Constraints

#### Canonical scenes and graph model

Each scene is an undirected graph on nodes `0,...,11`. Node 0 is the initial operator position and all four robots `0,...,3` also start there. Decision slots are `t=0,...,59`; boundary 60 processes arrivals from slot 59 but has no decision, service slot, or movement. One slot represents one second. A robot may stay or traverse one listed edge per slot. Multiple robots may occupy the same node; there is no hidden collision rule. At the end of a movement slot, a robot that enters a node receives its map summary at the next boundary. All robots and the operator initially know node 0 at boundary 0.

The table gives each scene's edge list and node values in node-ID order. Edges are unordered pairs. Values are deterministic environment truth used in simulation and scoring. A policy may read a node's value only after that robot has learned the node; publishing the table makes the evaluator model reproducible, not the unrevealed values observable to the online policy.

| scene | undirected edges | values for nodes `0,...,11` |
|---|---|---|
| `open_rooms` | `[[0,1],[0,2],[1,3],[1,4],[2,5],[2,6],[3,7],[4,7],[4,8],[5,8],[5,9],[6,9],[7,10],[8,10],[8,11],[9,11]]` | `[0,4,5,6,5,7,4,8,6,7,9,10]` |
| `branched_corridor` | `[[0,1],[1,2],[2,3],[3,4],[2,5],[5,6],[6,7],[3,8],[8,9],[1,10],[10,11]]` | `[0,3,4,5,8,6,9,10,7,11,5,12]` |
| `narrow_cave` | `[[0,1],[1,2],[2,3],[3,4],[4,5],[2,6],[6,7],[7,8],[4,9],[9,10],[10,11]]` | `[0,3,5,6,8,10,4,7,11,5,9,12]` |
| `request_burst` | `[[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[3,9],[4,9],[9,10],[10,11]]` | `[0,5,4,7,6,8,9,10,11,5,12,13]` |
| `operator_shift` | `[[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[3,7],[4,7],[5,8],[6,9],[7,10],[8,11],[9,11],[10,11]]` | `[0,4,6,5,8,7,9,10,6,11,12,14]` |

For the first four scenes, the operator remains at node 0. In `operator_shift`, it is at node 0 through boundary 19, at node 1 for boundaries 20 and 21, and at node 3 from boundary 22 through boundary 60. The current operator node becomes visible at its boundary; the future relocation is not a policy observation even though the fixed trajectory is public for reproducibility.

A robot's local node knowledge maps each known node to the earliest generation boundary carried by the summary it learned. Discovering node `v` after slot `t` creates generation time `t+1`; receiving a copy preserves the transmitted generation time. The environment also tracks the earliest discovery boundary of every node across the team. Repeated discoveries or summaries retain the earliest available time.

#### Parent topology, propagation, and energy

At boundary `t`, robot `i` declares a desired `parent_i(t)` equal to `null`, `-1` for the operator, or a different robot ID. All non-null parent chains must be acyclic and terminate at the operator. A robot may have at most one robot child, so its desired incident degree is at most two including its own parent; the operator may have at most two robot children. Every desired parent-child pair must be within radio range at the slot start: its endpoints occupy the same graph node or two nodes joined by a listed edge.

Let `stored_parent_i(t)` be the parent retained from the last accepted slot, initially `null`. A desired edge is in setup during slot `t` when `parent_i(t) != stored_parent_i(t)` and is inactive in that slot. It is active when the desired parent equals the stored parent, is non-null, and remains in radio range. At the end of an accepted slot, `stored_parent_i(t+1)=parent_i(t)`. Thus a new or changed attachment must be declared unchanged for one following slot before it carries data. Declaring `null` disconnects the old edge immediately. A robot incident to any desired edge, whether setup or active, must set `move_to` to its current node. A robot with no desired incident edge may stay or move to an adjacent node.

On one active edge during slot `t`, the child may offer at most one node summary through its `map_node` decision. A non-null offer is legal only when the child knows that node at boundary `t`; it arrives at the parent at boundary `t+1`, preserving its generation time. The parent may already know it, in which case the transmission still consumes energy but changes no state. A robot may forward a summary learned from another robot. If the active parent knows any released request that the child does not know, the oldest such request, ordered by `(release_slot,request_id)`, is automatically sent downward on the same edge and arrives at boundary `t+1`. Upward map and downward request transfer may occur simultaneously. Transfers use the knowledge snapshot at boundary `t`, so an item received at `t+1` cannot be forwarded in slot `t`. The operator knows each request at its release boundary.

Every robot begins with 36 energy units. In an accepted slot, robot `i` pays the sum of:

- `1.0` if it moves to a different node;
- `0.25` for each desired incident robot edge, and another `0.25` if its own parent is the operator;
- `0.5` if its desired parent differs from its stored parent, including a change to or from `null`;
- `0.1` if it submits a non-null `map_node`.

The second term charges each robot endpoint of a robot-robot edge; the operator has no energy account. All costs are checked before execution and may use an absolute tolerance of `1e-6`. A joint decision in which any robot would exceed its remaining energy fails the energy rule; for diagnostic continuation the whole joint decision is replaced by four robots staying, retaining their stored parents, and sending no summary. Costs from this replacement, not the rejected request, are applied.

#### Online requests and event order

Each scene has two region requests and one assistance request. Region targets are node sets. Assistance targets are robot IDs and require four consecutive service slots. A region request accepts map arrivals at its inclusive deadline boundary. Assistance may accrue only in decision slots strictly before its deadline.

| scene | `R0`: release, deadline, targets | `R1`: release, deadline, targets | `R2`: release, deadline, target robot |
|---|---|---|---|
| `open_rooms` | `8,25,[7,8,10]` | `20,43,[9,10,11]` | `28,48,2` |
| `branched_corridor` | `7,24,[4,6,7]` | `21,45,[9,11]` | `30,52,3` |
| `narrow_cave` | `8,26,[5,8]` | `23,47,[10,11]` | `31,54,1` |
| `request_burst` | `14,30,[7,9,10]` | `16,38,[8,10,11]` | `18,40,3` |
| `operator_shift` | `9,28,[6,7,8]` | `24,48,[10,11]` | `20,42,0` |

At a boundary, first admit map and request transfers scheduled in the preceding slot. Then release the current requests at the operator and update request state. Next expose the current policy view and choose the four actions jointly. Check field relations, movement, desired topology, map availability, and energy; determine setup and active edges; schedule upward map and automatic downward request transfers; execute movement by robots with no desired incident edge; create discoveries for the next boundary; accrue one assistance slot if its target already knew `R2` at the decision boundary and has an entirely active parent chain to the operator during this slot; debit energy; and record only the submitted decisions. At boundary 60, admit final transfers and update region request results, but do not accrue assistance or act.

The policy view may contain the scene name and index, `t`, current operator node, the public graph, each robot's current node, its locally known nodes with generation times and revealed values, its delivered request records, stored parent and whether that edge was active in the preceding slot, remaining energy, current radio-range neighbors, and all past inputs and decisions. It must not contain unrevealed node values, unreleased request contents, transfers that have not arrived, or future operator positions. In a joint implementation, each robot-local proposal must be formed only from that robot's private knowledge and requests plus the public current state. A joint constraint resolver may use those proposed movement or parent intents and scalar connection or resource demands to resolve shared topology, resource, or duplicate-target conflicts. It must not inspect one robot's underlying private node values, request contents, or map-offer payload to choose another robot's exploration target, map offer, or request response. Holding several local views in one process does not merge their simulated knowledge, and an item becomes private input to another robot only after a completed transfer. Robot ID may define one uniform priority or tie-break, not robot-specific trajectories. Scene, request, robot, or absolute-slot action tables are prohibited.

#### Result contract, invalid continuations, and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. Object member names must be unique; `NaN`, infinities, and other non-JSON constants are invalid. The array has exactly one record for each canonical scene; order is irrelevant. A scene object has exactly `scenario` and `slots`. `slots` has exactly one record for every integer `t` in `0,...,59`; order is irrelevant. A slot object has exactly `t` and `actions`. `actions` has exactly one record for each integer robot ID `0,...,3`; order is irrelevant. An action object has exactly integer `robot`, integer `move_to`, `parent`, and `map_node`. Booleans are not integers. For the local evidence format, `parent` and `map_node` are each JSON `null` or an integer. Their legal domains remain the ones defined above: `parent` is `-1` or another robot ID, and `map_node` is a node ID in `0,...,11`; `move_to` must be a node ID and then satisfy the movement rule. Thus a well-typed but out-of-domain integer is a complete decision whose illegality is handled below, not a malformed field. These 300 slots and 1200 actions are the complete required replay evidence. No position, knowledge, message, request, topology, energy, aggregate, metric, role, or controller-state field is required.

Invalid submissions receive no task-metric score, but replay uses deterministic continuations for attributable diagnostics. Unknown records are ignored by indexing and replay, while canonical coverage still counts their explicit IDs and the local evidence check still validates every submitted record's container and decision-field format. At every hierarchy level, replay retains the first scene, then the first slot within that retained scene, then the first robot action within that retained slot for each expected explicit ID; a later duplicate cannot supply missing descendants of the retained record. A missing or unrelatable action becomes `move_to=current_node`, `parent=null`, `map_node=null`. An identified action whose container or any decision field is malformed uses the same replacement and fails its corresponding evidence atom. Either case establishes a scene-wide causal cutoff because topology and knowledge couple all robots: from that boundary through the end of the scene, integer domain checks remain attributable, while adjacency, radio range, knowledge, activity, and energy checks that depend on reconstructed state are not inferred. A duplicate by itself does not establish a cutoff.

For a complete, well-typed stream, corrections are applied in the following fixed order. First, a `move_to` outside `0,...,11` or not equal or adjacent to the current node fails movement destination and becomes a stay; a `parent` outside `{-1,0,...,3}` or equal to the child fails parent selection and becomes `null`; and a non-null `map_node` outside `0,...,11` fails map availability and is ignored. Second, the parent graph formed by the corrected parents is checked. If it is cyclic, unrooted, over-degree, or has an edge out of radio range at the slot start, parent topology fails and all four parents become `null` for that slot. Third, movement by an endpoint of any remaining desired edge fails linked mobility and becomes a stay. Fourth, each remaining map offer is legal only if the child knows that node at the decision boundary and its own corrected parent edge is active; an illegal offer fails map availability and is ignored. Finally, energy is computed from these corrected decisions. These complete illegal decisions do not establish a causal cutoff.

If the corrected joint decision exceeds any robot's remaining energy, energy legality fails and the executed continuation for all four robots is: stay, use each robot's slot-start stored parent as its desired parent, and send no summary. This replacement has no parent-change charge; its desired-edge endpoint charges are still debited, even if that diagnostic debit makes remaining energy negative. No parent change from the rejected joint decision is counted. Otherwise, the corrected decision is accepted, its costs are debited, and `C` counts each robot whose executed parent differs from its slot-start stored parent. Every submitted fact has one format, coverage, legality, or replay owner, and every metric remains gated by all validity checks.

Let `g_v` be the team's earliest discovery boundary for node `v`, `a_v` its earliest arrival at the operator, and `q_v` its fixed value. Node 0 has value zero and does not affect denominators.

`primary_metrics_1` is, per scene, `sum_v q_v * I[a_v-g_v<=8] / sum_v q_v`, requiring both times to exist, then averaged over five scenes.

For region request `R`, its quality is the fixed value of target nodes known by the operator no later than the inclusive deadline divided by the total target value. For assistance `R2`, quality is `min(1,L/4)`, where `L` is the longest consecutive run of eligible service slots after the target learned the request and before its deadline. `primary_metrics_2` is `(quality_R0+2*quality_R1+2*quality_R2)/5` per scene, then averaged over scenes.

`auxiliary_metrics_1` is `sum_v q_v*clip(1-(a_v-g_v)/8,0,1)/sum_v q_v` per scene, treating absent times as zero contribution, then averaged over scenes.

For `auxiliary_metrics_2`, let `V_t` be the timely value used in `primary_metrics_1`, `V` the total node value, `E_c` the sum of desired-edge endpoint and non-null-summary energy actually charged to all robots, and `C` the number of accepted robot parent changes. Its scene score is `V_t/(V_t+V*(E_c/(4*36)+C/(4*60)))`, or zero when the denominator is zero; `4*36` is the total initial energy and `4*60` is the physical maximum number of robot-slot parent changes. Average the score over scenes.

Here `clip(x,0,1)=min(1,max(0,x))`. Higher is better for every metric. Normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. All metrics come from fixed-model replay and are scored only when every validity check is full.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite the root-level `results.json` from project code and the fixed conditions above. The evaluator deletes an old result before the one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations. Controlled experiments used as Judge evidence may produce optional text or JSON artifacts outside `results.json`; for every claimed comparison, identify the conditions, shared inputs, changed factor, artifact path, and condition-level outcomes.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "slot, movement, parent, and map-summary fields satisfy local types", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "movement, topology, summary, information, and energy decisions are legal", weight: 4}
- {id: replay_validity, desc: "movement, links, propagation, requests, energy, and decisions replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed timely operator-visible map value", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed online region and assistance quality", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed value-weighted map freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed communication and topology efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of exploration, local knowledge, topology, propagation, requests, and energy"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent exploration-communication model. Implementation/evidence: no corresponding executable states or decisions."}
    - {score: 0.4, desc: "Report: omits a decisive discovery, parent-link, propagation, request, movement, or energy relation. Implementation/evidence: code realizes only a partial or materially divergent model."}
    - {score: 0.7, desc: "Report: accurately defines graph exploration, local knowledge, link setup, bidirectional propagation, request service, and resources. Implementation/evidence: code and decisions realize the core model across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains event order, information boundaries, topology transitions, temporal couplings, objective trade-offs, and boundary cases. Implementation/evidence: code faithfully realizes every central relation with traceable internal evidence and no material divergence."}
- id: method_design
  desc: "unified online exploration, relay-topology, summary, request, and energy coordination"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable decision process."}
    - {score: 0.4, desc: "Report: fixed tours, periodic returns, or unit schedules dominate while freshness, setup, request, or energy consequences are ignored. Implementation/evidence: pre-filled scene, request, robot, or slot decisions dominate or code diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method uses revealed graph, knowledge age, topology, requests, and resources to coordinate actions. Implementation/evidence: one procedure operates across all scenes with limited lookahead or robustness."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly addresses frontier value, multi-hop freshness, link setup, region deadlines, sustained assistance, and remaining energy. Implementation/evidence: evolving permitted state changes decisive choices without unit-specific action tables or material divergence."}
- id: experimental_coverage
  desc: "controlled evidence for topology adaptation, map propagation, online requests, and resources"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or isolated coverage summaries do not test exploration-communication causality. Implementation/evidence: outputs cannot support the principal claims."}
    - {score: 0.7, desc: "Report: all scenes and one controlled topology, propagation, request, or energy comparison directly test a central trade-off. Implementation/evidence: both conditions execute with comparable condition-level map, link, request, and resource evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check. Implementation/evidence: reproducible condition-level discovery, transfer, topology, request, energy, and metric outcomes support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of freshness, topology transitions, requests, energy, and coverage"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats coverage or delay values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important discovery, freshness, topology, request, and energy trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence explains the central mechanism, quantifies principal trade-offs or sensitivities, tests contrasting evidence or a failure boundary, and states scope and unknowns. Implementation/evidence: reproducible condition-level comparisons support the central conclusions."}
