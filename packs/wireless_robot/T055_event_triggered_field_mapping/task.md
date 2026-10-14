---
id: T055_event_triggered_field_mapping
name: Event-Triggered Multi-Resolution Field Mapping for Mobile Robot Teams
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

Four mobile robots cooperatively map a changing scalar field. Each robot senses only its current cell and maintains a hierarchical map. Event-triggered summaries cross a position-dependent, bandwidth-limited wireless graph; delayed or missing summaries leave teammates with stale estimates. The team must preserve map quality in high-gradient regions while spending finite computation, memory, movement, energy and communication resources.

### Research Problem

Design a deterministic, non-training online procedure that chooses robot motion, hierarchical split/merge operations, event-triggered summaries, packet contents, link service and retransmission decisions for all five canonical scenarios. The procedure must use only each robot's local state, past measurements, arrived summaries, current link probes, queues and public fixed rules. It should make received information affect later sampling or map operations, balancing accuracy and uncertainty against trigger redundancy, queue delay, stale consensus, motion and energy.

### Experimental Setting and Constraints

#### Canonical evaluation units and scenario conditions

The grid is `x=0..19,y=0..15`, with four robots `R0,R1,R2,R3` and `H=120` one-second slots. Each slot every robot first measures its current cell, then chooses `stay,N,S,E,W`; an out-of-grid or obstacle-core move executes `stay`. The five scenarios are:

|scenario|initial `R0,R1,R2,R3`|obstacle cores|field `f(x,y,t)`|radius / interference|energy|
|---|---|---|---|---|---:|
|`smooth_field`|`(1,1),(18,1),(1,14),(18,14)`|none|`clip(0.15+0.018x+0.015y+0.08 sin((t+x)/18),0,1)`|9 / low|480|
|`sharp_frontier`|`(1,2),(1,13),(18,2),(18,13)`|`[9,9]x[0,6]`, `[9,9]x[9,15]`|`clip(0.15+0.65*1[x>=10]+0.08 sin((y+t)/10),0,1)`|8 / high at `40..59`|500|
|`range_gap`|`(2,2),(7,2),(12,13),(17,13)`|`[8,11]x[5,10]`|`clip(0.2+0.5 exp(-((x-15)^2+(y-4)^2)/30)+0.1 sin(t/12),0,1)`|6 / high at `30..69`|500|
|`moving_hotspot`|`(1,1),(18,1),(1,14),(18,14)`|`[7,12]x[6,9]`|`clip(0.1+0.8 exp(-((x-(3+(floor(t/10) mod 14)))^2+(y-(3+(floor(t/15) mod 10)))^2)/12),0,1)`|8 / medium at `50..79`|520|
|`bandwidth_burst`|`(2,8),(7,8),(12,8),(17,8)`|`[5,5]x[3,12]`, `[14,14]x[3,12]`|`clip(0.2+0.35*1[y>=8]+0.25 sin((x+t)/8),0,1)`|8 / high at `24..47,72..95`|490|

#### State, actions and task model

The field is environment truth. A measurement at `(x,y,t)` by robot index `i` is `z=clip(f(x,y,t)+0.04*(((17x+11y+5t+3i) mod 9)-4)/4,0,1)`. Future field and noise are not visible to a robot. A `5 x 4` grid of 4-by-4 root regions forms a quadtree of maximum depth two; leaf side lengths are 4, 2 and 1. Each robot starts with every root as an active leaf, mean `0.5`, precision `4`, and source version `-1`.

For an unambiguous public node identity, let `depth=0,1,2` for node sizes `4,2,1` and define `node_id=10000*depth+32*y0+x0`, where `(x0,y0)` is the node's lower-left cell. Root ids therefore use depth zero, and a split's four child ids are determined by their published geometry. A snapshot `source_versions` is an object with exactly the four keys `R0,R1,R2,R3` and integer values; use `-1` for a source not yet fused at that active node.

For a measured leaf, update `tau'=tau+100`, `eta'=tau*mu+100z`, `mu'=eta'/tau'`. Splitting a node creates its four children, inheriting mean, precision, local version, `updated_slot` and all source versions. Merging four sibling leaves uses the area-weighted mean and area-weighted variance `sum(area*(1/tau+(mu-mu_bar)^2))/sum(area)`, clipped below by `0.0025`; precision is its inverse. The merged parent's local version is the maximum child local version, each source version is the maximum child value for that source, and `updated_slot` is the merge slot. Fusion leaves the local version unchanged, updates the accepted source version and sets `updated_slot` to the arrival slot. An operation is `update`, `split`, `merge` or `fuse`. Costs are respectively 1, 4, 2 and 1; per-robot per-slot cost is at most 48, total cost at most 3600 per scenario, and active leaves per robot are at most 128. A non-stay move costs 1 energy, a measurement costs 0.1, computation costs 0.02 per cost unit, and transmission costs 0.001 energy per bit.

#### Information availability and communication event order

A summary has a 32-bit header and one to eight node entries of 64 bits each. A node entry contains `node_id,source_robot,version,delta_tau,delta_eta`; the packet payload is `32+64*n` bits. For line of sight, coordinate `(x,y)` is the center of the closed cell square `[x-0.5,x+0.5] x [y-0.5,y+0.5]`. The supercover of two endpoints is every grid cell whose closed square intersects the center-to-center segment; intersection only at an edge or corner still counts. Remove the two endpoint cells before testing obstacles. Directed links exist when endpoint distance is within the scenario radius and none of these remaining supercover cells is an obstacle core. Thus, for example, the segment from `(6,15)` to `(3,10)` includes `(5,12)` and is blocked by that obstacle cell.

#### Legal actions and boundary conditions

At low, medium and high interference, link capacity is respectively `544,288,0` bits per selected link per slot and completion delay is `0,1,2` slots. All links share one channel and at most two selected links per slot may have disjoint endpoints. Each sender FIFO holds 16 packets. A packet completion succeeds exactly when `((19*i+7*j+3*t) mod 100)/100 >= 0.05+0.15*level`, where `i,j` are endpoint indices and `level` is the current interference level. Failed service is retried at the head; first transmissions and retransmissions together may serve at most 48000 bits. A packet older than 16 slots expires before service. A packet is created only when the sender has an event trigger and is admitted only if the FIFO and bit budget allow it. Admission reserves no future bits: the bit-budget condition means only that cumulative bits already served are below 48000; a selected positive service amount must keep the cumulative total at or below 48000. Same-robot trigger slots must differ by at least two.

Each packet has exactly one receiver (the `receivers` array therefore has length one); this keeps the single `arrival_slot`, `status` and `fused_node_ids` fields minimal and fully replayable. A robot's initial energy is one quarter of the scenario energy budget, and `team_energy` is the sum of the four remaining balances. A packet completion that fails resets its partially transmitted payload to the FIFO head and increments `retransmissions`; a successful completion arrives after the public interference delay. More precisely, completion in slot `c` is processed at the start of slot `c+1+delay`; this arrival slot may be `120..122`, after the decision horizon. A queued or partially served packet remains in the FIFO; successful completion removes it. The queue row is the end-of-slot FIFO after expiry, admission and service. Its `served_packet_ids` contains the sender's head iff it received positive service, and `rejected_packet_ids` contains creation attempts not admitted that slot. If multiple packets share an arrival slot, process them by increasing completion slot (`last_service_slot`), with ties following their order in that completion slot's aligned `service_slots` arrays. This same order governs their fusion state changes.

A packet node is a retained local-update increment: `source_robot` equals the sender, `node_id` and `version` identify one of that sender's earlier `update` operations, `delta_tau=100`, and `delta_eta=100*measurement` for that update. This prevents invented or multiply counted information. At arrival, an entry newer than an active receiving leaf's recorded version for that source is projected by overlap: add `q*delta_tau,q*delta_eta`, where `q` is overlap area divided by the summary-node area. Every accepted packet entry has one `fuse` operation for every receiving leaf it changes, in packet node order and then ascending receiving `node_id`; multiple entries may therefore produce multiple operations for the same leaf. The summary node appears once in `fused_node_ids`; entries already known everywhere have no effect. This rule allows sender and receiver quadtrees to have different legal resolutions without privileging one reference partition.

Packet-node `(node_id,version)` pairs are unique within a packet. At the horizon, `status` is `rejected` for a non-admitted creation, `expired` for age removal, `arrived` after successful completion (including a scheduled post-horizon arrival), `in_service` only for a remaining partial payload, and otherwise `queued`. Expiry is checked only before service; after successful completion a packet is removed from the FIFO and arrives and fuses after its fixed delay even when arrival is more than 16 slots after creation. Rejected packets have null service/arrival slots; queued, expired and partial packets have null arrival. `first_service_slot` and `last_service_slot` are the first and last slots with positive service, or null if none. `fused_node_ids` contains each summary node that changed at least one receiver leaf exactly once, and is empty for a post-horizon arrival.

The public slot order is: process arrivals and fuse them; measure and update the local map; choose operations, motion, summaries and trigger/service proposals; serve selected links; execute motion; then update node ages, compute/memory, energy and mission state. The fixed snapshots are boundaries `9,19,...,119`; each robot decodes its active leaves over all 320 cells. A snapshot cell has the latest active estimate covering it, its precision, local node version and age since that leaf's last update, accepted fusion or merge. Snapshot label `s` is taken after slot `s` and corresponds to the same evolved map as state boundary `s+1`. Link probes and measurements use the pre-motion positions at state boundary `s`. The field gradient is `g(c,t)=max(|f(c,t)-f(n,t)|)` over in-grid four-neighbors, including obstacle-core cells because the analytic field remains defined there; `w(c)=(1+2g)/sum_c(1+2g)` and a cell is high-gradient when `g>=0.25`. Robots may use only local state, history, arrived summaries, current probes and queues. True field, future noise, future link outcomes, other robots' unsent state and the high-gradient truth are replay-only.

Within one robot and slot, `map_operations` are applied in ascending `operation_id`: arrival `fuse` evidence comes first, then the mandatory measurement `update`, then chosen split/merge operations. For `update`, `node_id` is the updated leaf, `parent_id=null`, `child_ids=[]`, and source fields identify the robot and resulting local version. Candidates assign nonnegative update versions; they must be strictly increasing for each `(source_robot,node_id)` stream, but may use either node-local counters or a robot-global counter. No consecutive numbering is required. For `split`, `node_id` is the parent and `child_ids` are its four children; for `merge`, `node_id` is the resulting parent and `child_ids` are the four removed children. For `fuse`, `node_id` is the changed receiving leaf and source fields identify the accepted entry. Inapplicable source fields are `source_robot=robot_id,source_version=-1`; inapplicable parent/children are `null` and `[]`. State row `0` is the initial boundary; row `t+1` is after slot `t`, and its `compute_used` is that robot's slot-`t` operation cost. Mission boundary `t+1` likewise includes all slot-`t` costs and counts cells measured in slots `0..t` as visited.

#### Task metrics

The primary metrics are independently replayed. `primary_metrics_1` averages `0.65*(1-weighted_MAE)+0.35*clip(1-mean((1/tau)/0.25),0,1)` over robot/snapshot pairs. `primary_metrics_2` averages `0.6*timely_team_coverage+0.4*quality_qualified_visit` over high-gradient snapshot cells; timely coverage requires a visit in the preceding ten slots and at least three robots with error <=0.15 and variance <=0.05 at snapshot, while qualified visit requires the visiting robot to meet those bounds after visiting. Empty sets score one. `primary_metrics_3` is `0.3*clip(1-compute/3600,0,1)+0.25*clip(1-memory_peak/512,0,1)+0.2*clip(1-motion/(4H),0,1)+0.25*clip(1-energy_used/energy_budget,0,1)`. `auxiliary_metrics_1` is `0.6*useful_bits/max(1,sent_bits)+0.4*on_time_packets/max(1,generated_packets)`; a successfully fused node is useful when removing its delta from the next snapshot lowers weighted MAE by at least 0.01 or newly satisfies a high-gradient coverage cell. `auxiliary_metrics_2` is `0.4*consensus_freshness+0.3*(1-redundant_triggers/max(1,triggers))+0.3*clip(1-mean_queue_wait/16,0,1)`. For snapshot `s`, the preceding ten slots are `max(0,s-9)..s`. Queue wait is admission to first positive service; an admitted packet never served waits through boundary `120`. When no triggers exist, the non-redundancy component is zero; when no packets are admitted, the queue component is one. The formal score is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`, clipped to `[0,1]`, after equal scenario averaging.

#### Result contract

The result contract is exact. The root is `{"scenarios":[...]}` with one record for each of the five scenario names; array order is irrelevant, each containing exactly `scenario,robots,observations,map_operations,map_snapshots,links,service_slots,packets,queues,mission_state`. Decision-bearing fields are `intended_actions`, chosen split/merge rows, packet creations and contents, and the aligned service arrays. `observations` and `links` are information available at their public pre-motion slot boundary. `executed_actions`, mandatory update/fuse rows, snapshots, packet terminal fields, queues, state traces and mission rows are post-event evidence recorded only after the events that determine them; a policy may not read those future or same-slot post-decision values when choosing its action. Grading uses each field according to this stated nature and never treats an execution-evidence field as an earlier observation. `robots` has four records with `robot_id,state_trace,intended_actions,executed_actions`; state has 121 rows `{slot,x,y,energy,active_leaf_count,compute_used}`, and actions have 120 `{slot,action}`. `observations` has 480 unique `(robot_id,slot)` rows `{robot_id,slot,x,y,measurement,leaf_id}`. `map_operations` is dynamic and has `{operation_id,robot_id,slot,kind,node_id,parent_id,child_ids,source_robot,source_version,cost}`; `map_snapshots` has 48 unique `(robot_id,snapshot_slot)` rows `{robot_id,snapshot_slot,cells,active_nodes}`, where cells has 320 `{x,y,mean,precision,version,age}` entries and active nodes have `{node_id,x0,y0,size,mean,precision,version,updated_slot,source_versions}`. `links` has 1440 unique `(sender,receiver,slot)` rows `{sender,receiver,slot,observed_available,observed_capacity}`. `service_slots` has 120 `{slot,selected_links,packet_ids,served_bits}` rows; arrays are aligned and at most two selected links have disjoint endpoints. `packets` is dynamic with `{packet_id,sender,receivers,created_slot,nodes,payload_bits,admitted_slot,first_service_slot,last_service_slot,arrival_slot,status,retransmissions,fused_node_ids}`; nodes have `{node_id,source_robot,version,delta_tau,delta_eta}`. `queues` has 480 `{robot_id,slot,packet_ids,served_packet_ids,rejected_packet_ids}` rows. `mission_state` has 121 boundary rows `{slot,team_compute,team_memory,team_energy,visited_cells,expired_packets}`. Every `selected_links` entry is exactly the ASCII string `sender->receiver`, for example `R0->R2`; array order is the candidate's service priority and is semantically meaningful. `operation_id` and `packet_id` are scenario-local nonnegative integers, and queue/service packet references use them. All indices, versions, counts and bits are integers; all other numeric values are finite; tolerance is `1e-6`; objects contain no undeclared fields. `__pycache__` is an allowed compiler artifact only.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md` and a freshly written `results.json`. `report.md` must contain non-empty headings `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis` and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "robot, map, observation, link, packet, queue and mission evidence are typed and finite", weight: 4}
- {id: canonical_coverage, desc: "all fixed scenario, robot, snapshot, cell, link, service, queue and mission keys occur exactly once", weight: 4}
- {id: decision_legality, desc: "actions, operations, triggers, link arbitration, FIFO and budgets are legal", weight: 4}
- {id: replay_validity, desc: "field, observation, map, fusion, packet, motion, age, resource and snapshot replay agree", weight: 6}
- {id: primary_metrics_1, desc: "weighted map accuracy and uncertainty quality from replay", weight: 8}
- {id: primary_metrics_2, desc: "timely high-gradient team coverage and qualified visits", weight: 4}
- {id: primary_metrics_3, desc: "compute, memory, motion and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful delivered-summary efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "consensus freshness, trigger non-redundancy and queue efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the event-triggered multi-resolution mapping model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent event-triggered mapping formulation is given, or the stated model is materially wrong; Implementation/evidence: no executable mapping process or corresponding records are present."}
    - {score: 0.4, desc: "Report: a decisive part of local sensing, hierarchical representation, communication, event timing or task relation is omitted; Implementation/evidence: the code or records cover only a partial model or materially depart from the report."}
    - {score: 0.7, desc: "Report: local sensing, hierarchical statistics, summaries, link variation and snapshots are accurately defined across the scenarios; Implementation/evidence: the code and results implement the core model across all canonical scenarios with only limited non-critical omissions."}
    - {score: 1.0, desc: "Report: the formulation precisely connects resolution, triggers, age, communication and high-gradient task quality, including assumptions and boundaries; Implementation/evidence: the code and records faithfully realize and trace these relations without material divergence."}
- id: method_design
  desc: "quality of the unified state-driven mapping method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method is specified, or the method is prohibited or incompatible with the task model; Implementation/evidence: no reusable procedure or supporting decision evidence is present."}
    - {score: 0.4, desc: "Report: fixed resolution, periodic broadcasts, prefilled routes or another decisive simplification dominates the proposed decisions; Implementation/evidence: the submitted procedure relies on unit-specific actions or materially differs from the described method."}
    - {score: 0.7, desc: "Report: a deterministic reusable method jointly handles sampling, hierarchy and communication while acknowledging limited long-horizon handling; Implementation/evidence: the same procedure runs across the canonical scenarios and records support its principal decisions."}
    - {score: 1.0, desc: "Report: a justified reproducible method jointly handles information value, resolution, trigger hysteresis, link age, queues, motion and resources; Implementation/evidence: the code applies it consistently and records support the decisive choices without scenario-specific action tables."}
- id: experimental_coverage
  desc: "coverage of field structure, communication and resource comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment or experimental purpose is provided; Implementation/evidence: the code and records contain no checkable execution."}
    - {score: 0.4, desc: "Report: the experiment is isolated or limited to one field or communication condition and does not examine the central relation; Implementation/evidence: the recorded outputs do not execute or support the principal claim."}
    - {score: 0.7, desc: "Report: all scenarios are covered and at least one explicitly controlled resolution, trigger, communication or budget comparison is identified; Implementation/evidence: both comparable conditions are actually executed with records supporting the comparison."}
    - {score: 1.0, desc: "Report: complementary controlled comparisons test field structure, resolution, trigger timing, outages, packet age and failure boundaries; Implementation/evidence: condition-level records and metrics distinguish the stated mechanisms beyond aggregate scenario scores."}
- id: research_analysis
  desc: "evidence-based interpretation of mapping and communication trade-offs"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or materially contradicts the recorded findings; Implementation/evidence: the code and results cannot support the reported observations or conclusions."}
    - {score: 0.4, desc: "Report: results are mainly listed without mechanism explanations or material limits; Implementation/evidence: the records do not support the analysis asserted in the report."}
    - {score: 0.7, desc: "Report: age, queues, field variation, sampling and resource trends are explained from records, with observations distinguished from inference and limitations stated; Implementation/evidence: the code and results support those observations and explanations."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies interactions and failure boundaries without causal overclaiming and states scope and unknowns; Implementation/evidence: reproducible condition-level records support the central conclusions."}
