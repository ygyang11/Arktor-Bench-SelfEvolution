---
id: T014_5g_map_upload_control
name: Heartbeat-Guided Map Upload Control for Multi-Robot Exploration
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

Three robots explore a bounded indoor grid while building local maps. Each robot can continue moving without network service, but only map blocks uploaded through a variable 5G link can be merged at an edge server and returned as global situation summaries. Coverage holes, shared uplink capacity, and background load can discard map uploads or delay heartbeat feedback. Sending too aggressively wastes the finite budget and makes later heartbeat samples stale; waiting too long fills onboard caches and delays global coverage. Movement, map-block selection, upload priority, and cache eviction therefore jointly determine exploration progress, communication load, and global-map freshness.

### Research Problem

Build a deterministic, non-learning simulator and one unified online policy for all canonical scenes. At each slot, each robot must choose a legal local move and may upload one cached map block with a priority. After the move reveals new cells and possibly creates another block, it may discard cached blocks to respect storage. The policy must use delayed heartbeat samples, its local map, its latest received global summary, its own cache and history, and the remaining shared resources without using unobserved map truth or current and future link outcomes. Design the policy to form a broad and timely edge map while limiting attempted bytes, cache pressure, and stale global information under coverage loss and recovery.

### Experimental Setting and Constraints

#### Canonical scenes and grid maps

The canonical scene order is `steady_cell`, `coverage_gap`, `handover_route`, `congested_return`, and `branching_search`. Every scene uses grid coordinates `[x,y]` with integer `x,y` in `0,...,13`, robots `0,1,2`, and slots `t=0,...,47`. A robot may occupy any non-obstacle cell. The obstacle sets and robot starts are fixed as follows; range endpoints are inclusive and set subtraction applies before forming the coordinate pairs.

| scene | obstacle cells | starts for robots 0,1,2 |
|---|---|---|
| `steady_cell` | `{[4,y]: y in 0..10 minus {2,8}}` union `{[9,y]: y in 3..13 minus {5,11}}` | `[[1,1],[12,1],[1,12]]` |
| `coverage_gap` | `{[5,y]: y in 0..12 minus {3,10}}` union `{[8,y]: y in 1..13 minus {4,11}}` | `[[1,1],[12,2],[2,12]]` |
| `handover_route` | `{[x,5]: x in 0..11 minus {2,9}}` union `{[x,9]: x in 2..13 minus {4,11}}` | `[[1,1],[12,1],[6,12]]` |
| `congested_return` | `{[6,y]: y in 1..12 minus {3,10}}` union `{[x,7]: x in 1..12 minus {3,9}}` | `[[1,1],[12,1],[1,12]]` |
| `branching_search` | `{[6,y]: y in 2..13 minus {5,10}}` union `{[x,6]: x in 0..5 minus {2}}` union `{[x,6]: x in 7..13 minus {10}}` | `[[1,1],[12,1],[6,10]]` |

All other grid cells are free. Each listed free-space graph is connected. The environment stores the complete map, but a policy may use only cells already observed by that robot or received in a global summary. At a position `[x,y]`, the observation footprint is the in-grid center cell and its four Manhattan-distance-one neighbors. The local map records the free or obstacle status of every cell in that footprint.

Before slot 0, every robot observes its start footprint. The newly observed free cells form one initial map block per robot, generated at slot `-1`. In robot order these receive the first global block IDs. Thereafter, after a move in slot `t`, the free cells in the new footprint that were not previously in that robot's local map form one new block if the set is nonempty. Obstacle observations update the local map but are not included in a block. Block cells are lexicographically sorted by `(x,y)`. Block IDs reset in each scene and are `b0,b1,...` in generation order, with robots processed in numeric order within a slot. A block with `n` cells costs `16+8n` bytes; `1<=n<=5`.

#### Radio state, map uploads, and heartbeat feedback

Let `p=[x,y]` be a robot's position at the shared decision boundary. Its radio class `c` is computed from the scene and current slot:

- `steady_cell`: for Manhattan distance `d` from `[6,6]`, `c=2` if `d<=9`, otherwise `c=1`.
- `coverage_gap`: `c=0` when `5<=x<=8` and `2<=y<=11`; otherwise, for Manhattan distance `d` from  `[6,6]`, `c=2` if `d<=8`, otherwise `c=1`.
- `handover_route`: let `d` be the smaller Manhattan distance to `[1,6]` and `[12,6]`; first set  `c=2` for `d<=5`, `c=1` for `5<d<=8`, and `c=0` otherwise, then replace `c` by `max(0,c-1)` when  `x` is 6 or 7.
- `congested_return`: first use the `steady_cell` rule, then replace `c` by `max(0,c-1)` when  `t mod 12` is 7, 8, or 9.
- `branching_search`: for Manhattan distance `d` from `[6,1]`, `c=2` if `d<=6`, `c=1` if  `6<d<=10`, and `c=0` otherwise.

The shared map-uplink capacity `A[s,t]` in bytes is:

- 144 in `steady_cell`;
- 120 when `t mod 10<6` and 88 otherwise in `coverage_gap`;
- 96 when `floor(t/6) mod 2=0` and 128 otherwise in `handover_route`;
- 144, 112, 80, or 96 according as `floor((t mod 12)/3)` is 0, 1, 2, or 3 in  `congested_return`;
- 112 in `branching_search`.

At the pre-move boundary, each robot may attempt at most one block currently in its own cache. An attempt has an integer priority from 0 through 3, removes that block from the cache whether or not it succeeds, and counts its full bytes against the scene budget. Attempts are processed in robot order. An attempt succeeds exactly when its robot has `c>0` and its bytes do not exceed the remaining shared capacity after earlier successful attempts in the same slot. A success arrives at the edge in slot `t+3-c`; otherwise it is lost. An arrival at slot 48 or later has no evaluated effect. Each scene has a 2400-byte map-attempt budget, and each robot may make at most three attempts in every inclusive four-slot window `max(0,t-3),...,t`. Heartbeat bytes do not consume these map limits.

Every robot generates one mandatory heartbeat after all map attempts in every slot. With `B` equal to the total attempted map bytes in that slot, a heartbeat at radio class zero is lost. Otherwise its delay is

`3-c + I[B>0.75*A[s,t]] + I[B>A[s,t]]`,

where `I` is 1 when its condition is true and 0 otherwise. A delivered heartbeat reports its sample slot, radio class, shared capacity, attempted map bytes, and this delay. At a boundary, a robot sees only its delivered heartbeat with greatest sample slot, or `null` if none has arrived. Thus a heavy upload slot can delay later network feedback; a report never reveals a current or future radio state.

#### Edge service, global summaries, and caches

Successful map blocks enter an unbounded edge queue at the start of their arrival slot. Before each robot decision boundary, the edge orders queued blocks by decreasing submitted priority, then increasing generation slot, then increasing numeric block-ID suffix. It scans that order once with a 120-byte service allowance: a block that fits the remaining allowance is serviced and removed, while a block that does not fit remains queued and scanning continues. Serviced block cells are unioned into the edge global map.

After edge service in slot `t`, global-summary version `t` snapshots the merged free-cell set. For each robot independently, scheduling occurs after that robot's current move and uses its post-move position and radio class. The summary is lost for class zero and otherwise arrives at the start of slot `t+3-c`. An arrival at 48 or later has no evaluated effect. A robot retains the arrived summary with greatest version and may use its cells for planning; global cells do not replace the robot's locally observed obstacle status.

Each robot cache is an ordered collection of unsent map blocks with a 600-byte hard limit. Its initial block is cached before slot 0. A valid send removes its block before movement. After movement and new block generation, the policy may discard any set of blocks then present in that robot's cache; discard IDs are applied in increasing numeric suffix order. Sending or discarding never removes cells from the robot's local map. Cache bytes after these discards must not exceed 600. A block can therefore have exactly one terminal disposition: attempted once, discarded once, or still cached at slot 47.

#### Information boundaries and slot order

At a robot's pre-move boundary, its rule may read its position; locally observed free and obstacle cells; latest arrived heartbeat; latest arrived global summary; its cached blocks with IDs, generation slots, cells, and bytes; its own movement, upload, and discard history; remaining scene attempt bytes; and its own attempt count in the preceding three slots. Robots are processed in numeric order and may also see earlier robots' current attempt byte totals for shared-capacity planning. Each earlier total is exposed immediately after that robot submits its action, before the next robot is queried; the totals are attempted bytes rather than success outcomes, and all attempts are then processed in robot order. A robot may not read another robot's unreported local map or cache, unobserved obstacle truth, current radio class or capacity, future link outcomes, packets not yet at the edge, or summaries not yet delivered.

The pre-move action is one of `stay`, `north`, `east`, `south`, or `west`, with displacements `[0,0]`, `[0,1]`, `[1,0]`, `[0,-1]`, and `[-1,0]`. The destination must be an in-grid free cell already present in that robot's local map at the boundary. Robots may share cells and cross; there is no collision constraint. The upload choice is `null` or one block in that robot's boundary cache, and the priority is `null` exactly when the upload is `null`.

Each slot has this authoritative order: enqueue map arrivals; service and merge edge blocks; deliver heartbeats and global summaries; expose the pre-move state; choose the three pre-move actions in robot order; process all map attempts and remove attempted blocks; generate heartbeats; execute moves; update local observations and generate blocks in robot order; expose each robot's post-observation cache; choose and apply its discard IDs; create and schedule the current global summaries; then record the decisions. A block generated after movement cannot be attempted until the next slot.

Pre-move decision fields record only submitted movement and upload choices. Post-observation discard fields record choices after the current block, if any, has entered the cache. Heartbeats, map-block contents, radio outcomes, cache states, edge service, merged maps, global-summary deliveries, and all metric values are fixed-model consequences reconstructed by the grader. Online-information use and unified-policy provenance are evaluated from the implementation and report, not inferred from a particular legal decision sequence.

#### Result contract and metrics

`results.json` is strict JSON with exactly `{"scenarios":[...]}`. The array contains the five scenes in canonical order. Each scene object has exactly `scenario` and `frames`. `frames` contains exactly 48 records in slot order. Each frame has exactly `t` and `robots`; `t` is an integer and `robots` contains exactly three records in robot order.

Each robot record has exactly `robot`, `move`, `send_block_id`, `priority`, and `drop_block_ids`. `robot` is an integer; `move` is one of the five action strings; `send_block_id` is a block-ID string or `null`; `priority` is an integer from 0 through 3 or `null`, and it is non-null exactly when the send ID is non-null. `drop_block_ids` is a unique block-ID string array ordered by increasing numeric suffix. The field relationships, cache membership, movement boundary, resource limits, and block lifecycle are checked against the fixed replay. Integers reject Booleans. No derived trace, event table, aggregate, metric, or particular internal state representation is required.

Invalid submissions receive no task-metric score, but replay uses the following fixed continuation rules to keep diagnostics attributable. A non-string movement field is treated as `stay` without a second legality failure; a well-typed movement that is not a legal known-free move also leaves the robot in place and fails movement legality. A malformed upload ID or priority type is ignored without a second legality failure. A well-typed upload pair that violates the null/range relation or names a block outside the boundary cache is also ignored and fails its corresponding legality rule. A well-typed cached upload with a priority in `0,...,3` is processed even when it makes a rolling window or scene budget illegal. A malformed discard field applies no discards; for a well-typed ID array, cached IDs are applied and nonmembers are ignored even when uniqueness, order, or membership legality fails. No automatic eviction repairs a cache-limit violation. These rules affect only partial failure diagnostics because every task metric remains gated by all validity checks.

`primary_metrics_1` is, for each scene, the number of distinct free cells in the edge global map after slot 47 divided by the fixed number of free cells in that scene, averaged across scenes.

`primary_metrics_2` is the mean over all scenes and slots of the edge global-map coverage immediately after that slot's service: `|merged_cells[s,t]|/|free_cells[s]|`.

`auxiliary_metrics_1` is the scene mean of `clip(1-attempt_bytes/2400,0,1)`.

For `auxiliary_metrics_2`, after post-observation discards in each scene, slot, and robot, define the cache score as `clip(1-cache_bytes/600,0,1)`. At its pre-move boundary, define summary age as 4 if no summary has arrived, otherwise `min(4,t-v)` for latest arrived version `v`; the summary score is `max(0,1-age/4)`. The metric averages `(cache_score+summary_score)/2` over all scene-slot-robot objects.

Higher is better for every metric. The normalized task performance is `(8*primary_metrics_1+7*primary_metrics_2+3*auxiliary_metrics_1+2*auxiliary_metrics_2)/20`. Metrics are scored only when all validity checks pass.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and may launch a reasonably organized multi-file project. It must generate or overwrite root-level `results.json` from the project code and fixed conditions above. The evaluator deletes an old result before one official run.

`report.md` must contain these non-empty Markdown sections with headings written exactly: `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`. Explain the model, unified method, actual experiments and results, evidence-based analysis, and limitations. Controlled experiments used as Judge evidence may produce optional text or JSON artifacts outside `results.json`; for every claimed comparison, the report must identify the conditions, shared inputs, changed factor, artifact path, and condition-level outcomes.

## Auto Checks

- {id: entry_point, desc: "run.py exists as a regular file and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes and creates a fresh regular results.json", weight: 4}
- {id: report_contract, desc: "report.md contains the six required nonempty research sections", weight: 2}
- {id: result_json, desc: "results.json has the strict root and canonical scene containers", weight: 3}
- {id: result_evidence, desc: "frame, movement, upload-priority, and discard decisions are typed", weight: 4}
- {id: canonical_coverage, desc: "all scene, slot, and robot-slot decision keys are covered once", weight: 4}
- {id: decision_legality, desc: "moves, sends, priorities, windows, budgets, discards, and caches are legal", weight: 4}
- {id: replay_validity, desc: "map generation, radio, heartbeat, edge service, summaries, and block lifecycle replay", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed terminal edge-map coverage", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed time-domain global-map progress", weight: 7}
- {id: auxiliary_metrics_1, desc: "independently replayed map-attempt byte efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed cache and global-summary freshness", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "formulation of local exploration, heartbeat feedback, map buffering, and edge merging"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent map-upload exploration model. Implementation/evidence: no corresponding executable states or decisions."}
    - {score: 0.4, desc: "Report: omits a decisive local-map, heartbeat, cache, radio, edge-service, or summary relation. Implementation/evidence: code implements only a partial or divergent model."}
    - {score: 0.7, desc: "Report: accurately defines the core observations, map blocks, delayed feedback, cache and uplink constraints, edge merge, and exploration objective. Implementation/evidence: code and decisions realize them across all scenes with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: precisely explains both decision boundaries, event order, information limits, congestion feedback, map freshness, and exploration-resource consequences. Implementation/evidence: code faithfully exposes every central coupling without material divergence."}
- id: method_design
  desc: "unified online movement, heartbeat-guided upload, priority, and cache decisions"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited training method. Implementation/evidence: no reusable procedure or decision evidence."}
    - {score: 0.4, desc: "Report: the method is dominated by fixed rates, FIFO choices, or unit schedules that ignore decisive feedback, cache, value, or exploration consequences. Implementation/evidence: pre-filled actions dominate or code diverges from the report."}
    - {score: 0.7, desc: "Report: a legal reusable method addresses local motion, delayed-heartbeat throttling, block priority, and cache control using available state and history. Implementation/evidence: one procedure runs across all scenes with limited planning depth or robustness."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly treats the central temporal, informational, and shared-resource consequences, including delayed link feedback and the local-global exploration trade-off. Implementation/evidence: evolving permitted state drives all decisive choices without scene-specific schedules or material divergence."}
- id: experimental_coverage
  desc: "controlled evidence for coverage gaps, congestion, cache recovery, and map value"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment. Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: incomplete scenes or isolated byte totals do not examine the heartbeat-buffer-map relation. Implementation/evidence: outputs do not support the principal claims."}
    - {score: 0.7, desc: "Report: all scenes and one controlled capacity, heartbeat, upload, cache, or exploration comparison test a central trade-off. Implementation/evidence: both conditions execute with comparable condition-level evidence."}
    - {score: 1.0, desc: "Report: complementary non-redundant experiments include a mechanism-focused comparison and a sensitivity or failure-boundary check for central communication-exploration trade-offs. Implementation/evidence: reproducible condition-level queue, map, and task outcomes support both comparisons."}
- id: research_analysis
  desc: "evidence-based explanation of network feedback, cache pressure, and exploration progress"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts the records. Implementation/evidence: results cannot support the conclusions."}
    - {score: 0.4, desc: "Report: mostly repeats coverage, byte, or cache values without causal analysis or material limits. Implementation/evidence: evidence is insufficient for the claims."}
    - {score: 0.7, desc: "Report: recorded conditions support important coverage, heartbeat, queue, cache, summary, and exploration trends, with inference and limitations identified. Implementation/evidence: code and results support those observations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies central sensitivities, tests a congestion or coverage mechanism against contrasting or failure-boundary evidence, and states scope and unknowns. Implementation/evidence: reproducible comparisons support the central conclusions."}
