---
id: T031_connected_task_service
name: Connected Online Service Allocation for Safe Networked Robot Teams
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

Four mobile robots serve online requests in a factory while remaining a connected radio team. Requests appear over time at separated work cells and occupy their assigned robot for several slots. Robots submit assignment proposals to a dispatcher through lossy direct or single-relay links. A locally attractive assignment can pull a robot out of communication range, block a narrow passage, or leave the remaining robots unable to move without disconnecting the graph. Delayed proposals, relay queues, service deadlines, and deadlock recovery therefore affect both completed work and safe motion.

### Research Problem

Design one deterministic, state-driven controller that generates one action for each robot at every slot. It must jointly select proposal costs, direct or relayed proposal delivery, connected motion, and releases after a detected deadlock. The same controller must handle all canonical scenarios from the requests currently released, dispatcher assignments, local history, packet feedback, and current neighbor observations. It should complete valuable requests promptly while controlling message load, stale assignments, relay occupancy, connectivity interventions, and energy use.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five scenarios: `staggered_requests`, `narrow_bridge`, `late_priority`, `moving_obstacles`, and `range_partition`. Each uses a `20 x 14` grid with cells `x in {0,...,19}, y in {0,...,13}`, robots `0..3`, one fixed proposal dispatcher, and `H=100` one-second action slots `t=0,...,99`. State boundaries are `0,...,100`; slot `t` maps boundary `t` to `t+1`. A release at slot `t` is visible at action-start boundary `t`. Boundary `100` processes arrivals and may form assignments, but has no activation, observation, decision, motion, or service.

The request templates below give `(x,y)`, required service slots, value, and absolute deadline state. For each scenario, request `Ri` retains its own service, value, and deadline but uses the position of template `P_i` from that scenario's permutation. Completion is on time exactly when `completion_state <= deadline_state`.

| request | template position | service slots | value | deadline state |
|---|---:|---:|---:|---:|
| `R0` | `(3,2)` | 4 | 4 | 34 |
| `R1` | `(6,11)` | 6 | 3 | 46 |
| `R2` | `(11,3)` | 5 | 5 | 52 |
| `R3` | `(15,10)` | 7 | 2 | 60 |
| `R4` | `(4,8)` | 4 | 6 | 68 |
| `R5` | `(8,12)` | 6 | 4 | 76 |
| `R6` | `(14,4)` | 5 | 7 | 84 |
| `R7` | `(17,11)` | 8 | 3 | 94 |

| scenario | robot starts `0..3`; dispatcher | closed obstacle rectangles | release slots `R0..R7` | position permutation `P0..P7` |
|---|---|---|---|---|
| `staggered_requests` | `(1,1),(3,1),(5,1),(7,1)`; `(0,1)` | `[8,9]x[3,10]` | `(0,8,16,24,32,40,48,56)` | `(0,1,2,3,4,5,6,7)` |
| `narrow_bridge` | `(1,6),(3,6),(5,6),(7,6)`; `(0,6)` | `[9,10]x[0,5]`, `[9,10]x[8,13]` | `(0,4,8,12,32,36,40,44)` | `(4,5,6,7,0,1,2,3)` |
| `late_priority` | `(6,3),(8,3),(10,3),(12,3)`; `(5,3)` | `[6,13]x[6,7]` | `(0,2,4,6,54,60,66,72)` | `(0,4,1,5,2,6,3,7)` |
| `moving_obstacles` | `(2,6),(4,6),(6,6),(8,6)`; `(0,6)` | `[7,8]x[3,5]`, initially `[11,12]x[8,10]` | `(0,10,20,30,40,50,60,70)` | `(1,2,3,4,5,6,7,0)` |
| `range_partition` | `(2,6),(4,6),(6,6),(8,6)`; `(0,6)` | `[9,10]x[5,8]` | `(0,6,12,18,30,36,42,48)` | `(0,2,4,6,1,3,5,7)` |

All tables are public simulator inputs. To preserve the online information boundary, the controller procedure invoked at slot `t` may receive only requests released by `t`; it may not use an unreleased request's location, service time, value, or deadline to choose an action. The full table may be held by the environment layer solely to reveal requests and replay outcomes.

#### Motion, obstacles, connectivity, service, and energy

The second rectangle in `moving_obstacles` attempts to move one cell right at the end of slots `39,43,47,51`. Its proposed new closed rectangle applies at the next boundary unless it contains a post-motion robot position; such an attempt is cancelled, and a later attempt starts from the actual rectangle position. The obstacle configuration at action-start boundary `t` determines links and legal destinations in slot `t`. Other rectangles are fixed. A segment intersects a rectangle when it touches its closed interior, edge, or corner.

Each action proposes one of `(0,0),(1,0),(-1,0),(0,1),(0,-1)`. Proposals are processed in robot-ID order. A destination outside the grid, in an active obstacle, equal to an already accepted lower-ID destination, or equal to a still-current higher-ID position is replaced by stay. This also rejects pairwise swaps. After all four provisional positions are formed, construct the undirected robot graph: an edge exists when Manhattan distance is at most `4` and the endpoint segment intersects no active closed obstacle. If that graph is disconnected, every provisionally accepted nonzero move is replaced by stay. Initial graphs are connected. Each nonzero proposal replaced by either stage adds one safety event. Successful nonzero motion costs `2` energy; stay costs `0.2`.

An assigned robot automatically starts service in a slot where its executed post-motion position is the request cell; that slot is service unit one. While service is active, its submitted move must be stay. Service advances once per action slot and completes at that slot's post-event boundary after the fixed number of units. It cannot be interrupted or released. A completed request and robot are then free; an assigned request that has not started may be released only through the deadlock rule below. Every robot starts with `120` energy.

#### Proposals, packets, relays, and assignment

Every unassigned request has a proposal epoch beginning at its release boundary, or at the boundary after a release. A `new` control packet snapshots its source robot, request, current epoch, candidate-chosen finite `proposal_cost in [0,1]`, and generation slot. The dispatcher keeps the arrived packet with the lexicographically largest `(generation_slot,packet_id)` for each `(robot,request,epoch)`; a later arrival carrying an older pair does not replace it. At each arrival boundary, requests are examined in increasing request-ID order. For the current request, the eligible set contains retained current-epoch proposals whose source currently has no assigned or serving request. The dispatcher assigns when either all four source IDs occur in that eligible set, or four action slots have elapsed since the epoch began and the set is nonempty. The unique winner minimizes `(proposal_cost,generation_slot,robot,packet_id)`. That robot becomes busy before the next request ID is examined at the same boundary. A retained proposal is ignored while its source is busy but may become eligible after that robot is free. The assignment is immediately public to all robots at that boundary. If the eligible set is empty at the four-slot timeout, its first later nonempty state triggers assignment. Assignment outcomes are derived and are not submitted.

The dispatcher is the final receiver. A packet may go directly from its source or use one named robot relay. A successful source-to-relay hop arrives at the next boundary and enters that relay's FIFO; it can first be forwarded in the next slot. Each relay FIFO has capacity `4`. Simultaneous arrivals are sorted by request deadline, epoch-start boundary, generation slot, source robot, and packet ID before remaining capacity is filled; overflow is dropped. A `forward` operation must name the FIFO head, that head must have no outstanding failed hop, and its relay robot must stay. A failed forwarding packet remains at the head during its retry opportunity and can then be named only by `retry`.

Each robot submits at most one hop-attempt per slot. The team admits at most four attempts per slot and `160` over a scenario. Otherwise legal simultaneous attempts are ordered by request deadline, epoch-start boundary, generation slot, source, and packet ID. Every admitted source, retry, or forwarding hop costs its transmitting robot `1` energy and counts one budget unit. A hop is eligible only when its endpoint Manhattan distance is at most `4`. Let `o=1` when the endpoint segment intersects an active obstacle and `0` otherwise, and let `n` be the number of attempts admitted in that slot. Its success probability is `p=clip(0.92-0.06*(d/4)-0.08*o-0.04*(n-1),0.1,0.92)`. Define `u=int(SHA256(f"{scenario}|{slot}|{source}|{proposal_request}|{epoch}|{hop}|{attempt}"). hexdigest()[:8],16)/0xffffffff`. The hop succeeds exactly when `u<=p` and arrives at the next boundary. Here `hop` is `0` for the source hop and `1` for forwarding; `attempt` is `0` for the first attempt and `1` for its retry. The sample key is independent of candidate-chosen `packet_id`. The deterministic rule is public and may be used in planning; the resulting success or failure becomes packet feedback only at the next action-start boundary.

Each hop may be retried once, only in the slot immediately after its failure feedback, by the current holder. Omitting that retry or failing again discards the packet at the following boundary. Source-hop retry belongs to the source; forwarding retry belongs to the relay, must preserve FIFO-head service, and requires the relay to stay. Arrivals after boundary `100` expire without effect. Before admission, a robot must have enough energy for its eventually executed move or stay plus the proposed hop; insufficient-energy actions are illegal. A valid action beyond the remaining team budget is also illegal. All admitted attempts use slot-start positions, before motion.

Assignment, completion, or a new epoch never purges a packet already created. Pending arrivals, failed hops, and relay-queued packets continue through the same retry and FIFO lifecycle and may therefore consume energy, budget, or queue capacity after their proposal becomes stale. Such a packet can still be retried or forwarded when otherwise legal, but a stale final arrival cannot enter the dispatcher's eligible proposal set. It remains one successfully delivered proposal packet for `auxiliary_metrics_2`. A retained dispatcher proposal from an older epoch is removed when the new epoch begins; packets physically still in flight or in a relay FIFO are not removed.

#### Information, deadlock, and event timing

At an action-start boundary, a robot may use its own current position, energy, assignment, service state, packet history, and past observations. The released request specifications and their current `released`, `assigned`, `serving`, or `completed` status, dispatcher assignments, and the most recent deadlock event are public team feedback. A robot also receives outcomes for packets it sourced, the ordered packet IDs and outcomes in its own relay FIFO, and the current positions of its direct graph neighbors. The current active obstacle rectangles, including whether a scheduled movement was cancelled at the preceding boundary, are public action-start feedback. Other robots' energy, non-neighbor positions, proposal costs, packet state, remaining paths, and private history are not available. Proposal packets carry no implicit neighbor-summary channel. Future actions and future dynamic-obstacle cancellation are unavailable before their event; scheduled obstacle-attempt slots and the deterministic packet rule above are public.

At a post-event boundary with at least one assigned but unstarted request, enumerate for each such assigned robot the four nonzero moves while all other robots stay. A progress move must have a free destination, avoid collision, leave the robot graph connected, and strictly reduce Manhattan distance to its assigned target. The boundary is locally blocked when no progress move exists for any such robot. Three consecutive action slots ending in locally blocked boundaries, with no service completion or release in those slots, produce one deadlock event at the third post-event boundary. Only the immediately following action slot accepts release proposals. Each proposal must name that robot's own assigned, unstarted request. If several are legal, exactly one is applied after service, minimizing `(request_value,request_id,robot_id)`; the others have no effect. The released request starts a new proposal epoch at the next boundary, and all old-epoch proposals become stale. After a deadlock event the blocked counter restarts whether or not a release is proposed. If the named request starts service earlier in that same slot, the release proposal has no effect because service is already non-interruptible; this does not make the action-start proposal malformed.

For each slot, the authoritative order is: discard packets whose immediately preceding retry opportunity was omitted, then process arrivals and relay queues; update dispatcher proposals and form assignments; activate this slot's requests; expose action-start information; choose all actions; validate releases and packet references, then admit and advance hops using slot-start endpoints; resolve collision and graph-connected motion; charge actual communication and movement/stay energy; execute movement; start, advance, and complete service; apply one selected release; update deadlock state; attempt the scheduled obstacle motion; record boundary `t+1`. Submitted action rows are requests at the action-start boundary. Packet outcomes, assignments, executed motion, obstacle state, graph, service, deadlock, energy, and metrics are post-event facts derived by the grader. Boundary `100` performs only its arrival, relay-queue, dispatcher-update, and assignment prefix, including slot-99 arrivals.

#### Legal actions and boundary conditions

`comm` is `none`, `new`, `retry`, or `forward`. For `none`, `packet_id`, `proposal_request`, `proposal_cost`, and `relay` are null and no hop is submitted. `new` simultaneously creates and submits a source hop: `packet_id` is a scenario-unique nonnegative integer; `proposal_request in {0,...,7}` is released, unassigned, and in its current epoch; `proposal_cost` is finite in `[0,1]`; the source has no assigned uncompleted request; and `relay` is null for direct delivery or a different robot ID. The selected source-hop endpoint must be within Manhattan distance `4`; obstruction affects probability but not range eligibility. For `retry` and `forward`, only `packet_id` is non-null; all proposal fields and `relay` are null. A retry must reference the accepted failed hop from exactly the preceding slot. A forward must reference that robot's FIFO head. For `release_request`, null means no release proposal; otherwise it is an integer request ID satisfying the one-slot deadlock rule. JSON `null` and integer `-1` are equivalent for `release_request`, `packet_id`, `proposal_request`, and `relay` under every rule. Malformed references, serving motion, energy, FIFO, retry, epoch, range, or budget violations close the validity gate. Geometry and connectivity conflicts are instead resolved by the published safety executor and affect performance.

#### Task metrics

Let `v_r` be request `r`'s value and `V=sum_r v_r=34`. For scenario `s`, an on-time completion has quality `q_s,r=1`, a late completion at state `c_s,r` has `q_s,r=0.5*clip(1-(c_s,r-deadline_r)/H,0,1)`, and an incomplete request has `q_s,r=0`. `primary_metrics_1=(1/5)*sum_s (sum_r v_r*q_s,r)/V`.

For a started request define `z_s,r=clip(1-(service_start_slot_s,r-release_slot_s,r)/H,0,1)` and set `z_s,r=0` when it never starts. `primary_metrics_2=(1/5)*sum_s (sum_r v_r*z_s,r)/V`.

For post-action boundaries `1..H`, let `connected_fraction` be the fraction with a connected robot graph. At a connected boundary, `link_margin` is the minimum `(4-d)/4` over all current graph edges; at a disconnected boundary it is `0`. For scenario `s`, define `g_s=0.4*connected_fraction_s + 0.2*(sum_{b=1}^H link_margin_s,b/H) + 0.4*clip(1-safety_events_s/(H*4),0,1)`. Then `primary_metrics_3=(1/5)*sum_s g_s`.

If scenario `s` admits no packet unit, set `e_s=0`; otherwise set `0.5*clip(1-accepted_packet_units/160,0,1) + 0.5*useful_assignment_units/accepted_packet_units`. A useful assignment unit is the successful final hop carrying the winning proposal for one assignment epoch; each assignment contributes at most one. This expression defines `e_s`; `auxiliary_metrics_1=(1/5)*sum_s e_s`.

For every unique proposal packet successfully delivered to the dispatcher by boundary `H`, define `age=arrival_state-generation_slot`. Packet freshness is the mean `clip(1-age/H,0,1)`. The first-attempt fraction is the fraction of those packets whose source hop and, when present, forwarding hop both succeeded with `attempt=0`. If no packet is delivered, both are zero. If `f_s=0.6*packet_freshness_s+0.4*first_attempt_fraction_s`, then `auxiliary_metrics_2=(1/5)*sum_s f_s`.

All metrics are independently replayed and clipped to `[0,1]`. Normalized task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20`.

#### Result record contract

`results.json` contains exactly `{"scenarios": [...]}` with one record per canonical scenario. Each scenario record has exactly `scenario` and `actions`. `actions` contains exactly `4*100` rows, uniquely keyed by `(t,robot)`; scenario and action array order has no semantic meaning. Every action has exactly `t,robot,dx,dy,release_request,comm,packet_id,proposal_request,proposal_cost,relay`. The integer IDs and moves use the domains above. `proposal_cost` is a finite JSON number only for `new`, and otherwise null. The four optional ID fields accept the equivalent null/`-1` sentinel. No robot state, request state, assignment, obstacle, graph, queue, packet outcome, service, deadlock, energy, or metric summary is submitted. The grader derives them from the fixed model, so equivalent correct simulators are not rejected for choosing another redundant state representation. No field is optional and all JSON numbers must be finite.

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
- {id: result_evidence, desc: "minimal action fields are typed, finite, and auditable for replay", weight: 4}
- {id: canonical_coverage, desc: "all scenarios and every slot-robot action key occur exactly once", weight: 4}
- {id: decision_legality, desc: "moves, releases, proposal epochs, relays, retries, FIFO, budgets, and energy are legal", weight: 4}
- {id: replay_validity, desc: "request, assignment, obstacle, graph, packet, deadlock, and service replay is complete", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed deadline-aware service completion quality", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed service response quality", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed connectivity and motion safety quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed useful assignment communication efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed proposal freshness and retry efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the online request, proposal, connectivity, service, and deadlock model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent online service model; Implementation/evidence: no executable request, communication, and connected-motion process."}
    - {score: 0.4, desc: "Report: assignment or motion is partial, omits request timing, packet delivery, service occupancy, connectivity, or deadlock, or incorrectly calls full-team geometry or private state reconstructable at action time; Implementation/evidence: only a partial process or a model with that information-boundary departure is implemented."}
    - {score: 0.7, desc: "Report: released requests, proposal epochs, per-robot information boundaries, relay delivery, connected motion, service, energy, and deadlock are accurate; Implementation/evidence: the core model, including the published observations, runs across all scenarios with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: event order and the causal assignment-connectivity-service trade-off are precise; Implementation/evidence: the minimal action trace and code faithfully realize proposal, graph, service, and recovery behavior."}
- id: method_design
  desc: "quality of the unified proposal, connected planning, relay, and recovery policy, including the actual state read before each robot submits its action"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method, a prohibited training method, or a policy that relies on unavailable cross-robot state; Implementation/evidence: inspect the actual values read before action submission, regardless of report claims. A global environment simulator and the published post-submission safety executor are allowed, but a policy that does not first form filtered per-robot views and instead iterates another robot's private energy, non-neighbor position, packet/queue/path history, or jointly enumerates team actions from such state has no reusable legal procedure under the published observations."}
    - {score: 0.4, desc: "Report: nearest-request assignment or static bridge behavior ignores waiting, packet state, graph margin, or deadlock; Implementation/evidence: actions are prefilled or materially inconsistent."}
    - {score: 0.7, desc: "Report: one reproducible state-driven policy handles proposals, connected paths, service, and recovery under the per-robot information boundary; Implementation/evidence: the same procedure operates from each robot's published view across every scenario with limited planning depth."}
    - {score: 1.0, desc: "Report: a justified policy couples value and waiting with proposal age, relay FIFO, graph margin, bridge roles, non-interruptible service, and recovery; Implementation/evidence: evolving state drives decisions without scenario or absolute-slot action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for assignment-connectivity effects"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only canonical aggregates or an uninformative condition are shown; Implementation/evidence: evidence does not test the assignment-connectivity mechanism."}
    - {score: 0.7, desc: "Report: all scenarios and one controlled capacity, arrival, radius, obstacle, or recovery comparison are covered; Implementation/evidence: both conditions are executed with comparable records."}
    - {score: 1.0, desc: "Report: complementary controlled experiments test proposal delivery and a bridge, deadline, or deadlock failure boundary; Implementation/evidence: condition-level action traces and metrics reproduce both comparisons."}
- id: research_analysis
  desc: "evidence-grounded interpretation of service, communication, connectivity, and recovery"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradictory; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: values are listed without causal assignment, graph, delay, or recovery analysis or material limitations; Implementation/evidence: claims are unsupported."}
    - {score: 0.7, desc: "Report: actual records explain important service, proposal, connectivity, and deadlock trends and limitations; Implementation/evidence: observations agree with generated evidence."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies coupled mechanisms, tests the explanation at a contrasting condition or failure boundary, and states scope; Implementation/evidence: reproducible condition-level records support the conclusions."}
