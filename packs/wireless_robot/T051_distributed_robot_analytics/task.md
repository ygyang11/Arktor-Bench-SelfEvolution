---
id: T051_distributed_robot_analytics
name: Communication-Aware Dependency Scheduling for Distributed Robot Analytics
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

Three mobile exploration robots share computation with one fixed edge node. Each robot must repeatedly capture, localize, plan, and dispatch its motion workflow. Entering a science zone also releases an optional collection, analysis, and storage workflow. Intermediate data must be physically held by the node that executes the next task, so remote computation requires fragmented store-and-forward transfer over intermittent contacts. Computation, transmission, and reception compete for each node's single activity resource. A poor placement can therefore miss a required deadline, strand a science product, or consume a contact that another dependency chain needs.

### Research Problem

Build a deterministic simulator and a non-training online method that jointly chooses rover movement, task starts and execution nodes, and local transmission proposals. The method should complete required dependency chains on time while using heterogeneous processors and time-varying contacts to store valuable science at the edge node. It must manage FIFO custody, partial transfers, losses, product age, node activity, and energy without a pre-filled task schedule. Use one reusable state-driven procedure for all five canonical scenarios. Each node may use only its current local snapshot and shared remaining attempt budget; it may not inspect another node's private activity or data, a future science release, future contact or loss, or a scenario/task/zone/slot answer table. Deterministic search, scheduling, planning, optimization, and analytical heuristics are allowed.

### Experimental Setting and Constraints

#### Canonical scenarios and spatial inputs

There are five canonical scenarios in the order below, with scenario index `k=0,...,4`. The integer grid is `0<=x<18`, `0<=y<14`. The fixed edge node `E` is at `(16,7)`. A listed closed obstacle rectangle contains every integer cell in its Cartesian bounds. Slots last one second; decisions and service occur at `t=0,...,119`, and boundary 120 is terminal.

| scenario | starts `(R1,R2,R3)` | obstacle rectangles | science-zone centers `Z00,...,Z05` in ID order |
|---|---|---|---|
| `relay_gap` | `(1,2),(1,7),(1,12)` | `[6,7]x[0,5]`; `[6,7]x[9,13]` | `(4,10),(11,3),(14,10),(9,7),(3,5),(13,5)` |
| `science_cluster` | `(2,3),(2,7),(2,11)` | `[7,10]x[5,8]`; `[12,13]x[0,4]` | `(5,6),(5,7),(5,8),(14,6),(14,7),(14,8)` |
| `compute_imbalance` | `(1,1),(1,6),(1,11)` | `[5,6]x[0,4]`; `[5,6]x[9,13]` | `(4,4),(10,10),(14,4),(9,7),(3,11),(15,11)` |
| `contact_outage` | `(2,2),(2,7),(2,12)` | `[8,9]x[0,5]`; `[8,9]x[9,13]` | `(4,6),(4,8),(13,6),(13,8),(10,3),(10,11)` |
| `deadline_chain` | `(1,4),(1,7),(1,10)` | `[6,8]x[4,9]`; `[12,13]x[0,3]` | `(4,3),(4,11),(10,2),(10,12),(15,5),(15,9)` |

A science zone comprises the center and the four cells at Manhattan distance one. Static grid, obstacle, zone, task-template, duration, energy, and required-release information is public to all nodes. Robots may occupy the same cell and may swap cells.

#### Dependency chains, products, and computation

At each boundary `r` in `{0,30,60,90}`, every rover releases one required chain. Its chain ID is `req_<robot>_<r:03d>`; for example, `req_R2_030`. Its four task IDs append, in order, `_capture`, `_localize`, `_plan`, and `_dispatch`. Thus the 12 required chains contribute 48 fixed task IDs. Their direct precedence is `capture -> localize -> plan -> dispatch`. A chain deadline is `d_req=r+17`, inclusive.

Every zone `Zzz` has one potential science chain `sci_Zzz` with task IDs ending `_collect`, `_analyze`, and `_store`, in that direct precedence order. A released zone has value `value_Z=2+((3*z+1) mod 5)`, where `z` is its zero-based numeric suffix, and deadline `d_sci=release_boundary+36`, inclusive. All 18 potential science task IDs are canonical even if their zones are never released.

The fixed duration in slots, start energy, and output size in 256-bit data units are:

| task type | `R1` duration/energy | `R2` duration/energy | `R3` duration/energy | `E` duration/energy | output units |
|---|---:|---:|---:|---:|---:|
| `capture` | 3/2 | 3/2 | 3/2 | unavailable | 4 |
| `localize` | 6/3 | 1/2 | 5/2 | 2/3 | 3 |
| `plan` | 7/4 | 2/3 | 6/3 | 3/4 | 2 |
| `dispatch` | 2/1 | 2/1 | 2/1 | unavailable | 1 |
| `collect` | 4/2 | 4/2 | 4/2 | unavailable | 5 |
| `analyze` | 8/4 | 6/3 | 7/3 | 3/4 | 3 |
| `store` | unavailable | unavailable | unavailable | 2/2 | 1 |

`capture` and `dispatch` may run only on their required chain's source rover. `collect` may run only on its zone's first-entry winner, and `store` only on `E`; every node may run every other task type shown as available. Each node is single-threaded and computation is non-preemptive. A task started in slot `t` occupies that node in slots `t,...,t+duration-1` and finishes at boundary `t+duration`. Its computation energy is charged at start. A start is semantically valid exactly when the task is released and has never started, the chosen node supports it, the node is idle, and every direct predecessor product is in that node's FIFO. Starting consumes those products. A rover must also afford the combined task and movement energy selected for the slot.

The output product ID is `p:<task_id>`. At completion, every nonterminal task output enters its execution node's pending-product table with `ready_boundary=t+duration`. Completed `dispatch` and `store` instead produce the required or science mission result and do not enter a FIFO. Products carry their producer task, original ready boundary, size, chain source or zone winner, release, deadline, and immediate successor metadata. Consequently a receiving node can identify and execute an eligible successor without learning unrelated private node state.

#### Science release and movement

For each unreleased zone, movement in slot `t` creates an entry only for a rover whose old cell was outside and end cell is inside. If several rovers enter it in the same slot, `R1,R2,R3` order chooses the sole winner. The zone releases at boundary `t+1`; the winner then observes its entry outcome and may start `collect`. Other nodes learn that chain instance only by receiving one of its products. A slot-119 entry is applied at terminal boundary 120, when no task can start. Staying inside, or entering an already released zone, creates no further release.

Each rover selects one action from `stay,north,east,south,west`, with deltas `(0,0),(0,+1),(+1,0),(0,-1),(-1,0)`. A non-`stay` destination must be in the grid and not an obstacle and costs one energy unit. `stay` costs `0.1`; when less energy remains it is still legal and costs all remaining energy. A non-`stay` move is legal only if the rover can pay it together with a same-slot task start, or with the maximum possible send cost of its own submitted proposal. Illegal moves close decision legality; for deterministic replay they become `stay` and spend `min(0.1,current_energy)`. Movement can occur while the rover computes, sends, or receives.

#### Contacts, FIFO custody, and fragmented handoff

The node order is `R1,R2,R3,E`. Enumerate the 12 directed links by this source order and, for each source, the remaining nodes in node order; call the zero-based link index `p`. For a pair of slot-start positions, require Euclidean distance at most 6. At the five internal sixth-points `j=1,...,5`, round each coordinate to its nearest integer, with an exact half rounded to the smaller integer. The physical path is clear exactly when none of those five cells is an obstacle. Base capacity is 4 data units when both conditions hold and zero otherwise.

Let `q=(7*t+3*p+5*k) mod 12`. The current interference factor is zero on the first residue set, one half on the second, and one otherwise:

| scenario | zero residues | half-capacity residues | additional outage |
|---|---|---|---|
| `relay_gap` | `{0}` | `{1,2}` | none |
| `science_cluster` | `{0,1}` | `{2,3}` | none |
| `compute_imbalance` | `{0}` | `{1,2,3}` | none |
| `contact_outage` | `{0,1}` | `{2,3,4}` | every link incident to `E` is zero in slots `35..49` and `75..89` |
| `deadline_chain` | `{0}` | `{1}` | none |

Actual capacity is base capacity times that factor and is therefore `0`, `2`, or `4`. At each slot, only the current incident directed capacities are revealed to each endpoint. The deterministic environment implements the complete formula, but a node policy may not query it for a future slot or for a link not incident to that node.

Each node has a FIFO of capacity 24 data units and an unbounded pending-product table ordered by `(ready_boundary,product_id)`. A complete product occupies its full published size in the FIFO even during a partial outgoing handoff. At each admission step, admit the pending head if its entire size fits; repeat until the pending head does not fit, then stop. A local task may consume any predecessor product in its FIFO, but only when it has no active or partial outgoing handoff session; transmission may use only the FIFO head.

The first selected transmission for a queued product opens a handoff session to that `dst`; all later attempts at that custody node must retain the same `dst`. In one selected slot, attempt the lowest offsets not yet successfully sent, up to the current link capacity, the remaining product units, and the remaining scenario budget of 160 unit-attempts. Offsets are integers `0,...,size-1`. For an attempted unit let `h=(11*t+7*p+5*offset+3*k) mod 17`. It is lost when `h` is in the scenario's loss set and otherwise arrives at the next boundary:

| scenario | loss residues |
|---|---|
| `relay_gap` | `{0,1}` |
| `science_cluster` | `{0}` |
| `compute_imbalance` | `{0,1}` |
| `contact_outage` | `{0,1,2,3}` |
| `deadline_chain` | `{0,1,2}` |

Lost offsets remain eligible for retry; duplicate arrivals do not add inventory. Once every offset has been sent successfully, the sender removes the product at slot end. The receiver reassembles the product when the last unit arrives and adds it to its pending table. The original ready boundary is unchanged across hops. A proposal can be selected only in slots `t<ready_boundary+40`; at every boundary `b>ready_boundary+40`, all FIFO, pending, in-flight-fragment, and session state for that product expires. A slot-119 success arrives and can reassemble or enter pending at boundary 120 but cannot support another action.

#### Proposals, arbitration, energy, and boundary order

Each node may submit at most one proposal in a slot. It names the submitting `src`, another `dst`, the exact current FIFO-head `product_id`, and integer priority `0..9`. It must use a currently positive incident capacity, obey an existing handoff destination, be within product age, and come from a source that is not already computing and does not start a task that slot. A rover source must be able to pay its move plus `0.05` per maximum unit that this proposal could attempt; only actually selected units are charged. `E` computation and transmission energy enters metrics but has no finite budget.

Valid proposals are sorted by greatest priority and then ascending `(src,dst,product_id)` Unicode code-point order. After valid task starts occupy their nodes, the arbiter greedily selects at most two proposals whose endpoints are not already used and whose source and destination are not computing. A source cannot observe a destination's private activity; therefore a proposal excluded only because the destination is computing or used by an earlier selected link is legal and simply not selected. Every selected unit consumes one of the 160 attempts whether delivered or lost. Unselected proposals do nothing.

At every nonterminal boundary `t=0,...,119`, use this order:

1. Remove product, pending, fragment, and session state older than the 40-slot limit.
2. Finish computations due at `t`; process successful units from slot `t-1`, reassembly, and products.
3. Apply required releases due at `t` and science releases and entry feedback from slot `t-1`.
4. Admit pending products by the fixed FIFO rule.
5. Compute and reveal current incident capacities from slot-start positions.
6. Each node submits its candidate task start and proposal; each rover also submits its movement.
7. Validate task starts and combined rover energy first, occupy compute nodes, validate proposals, and   apply the fixed arbitration.
8. Execute selected unit attempts and rover movements; then detect new science-zone entries.
9. Update local feedback, energy, handoff sessions, and remaining computation time for the next boundary.

At boundary 120 perform only steps 1--4, including slot-119 arrivals and science release, then freeze. Decision arrays contain the candidates submitted in step 6. The grader reconstructs all tasks, releases, computation, positions, contacts, loss, fragment handoffs, FIFO state, energy, completion, and metrics from the three decision arrays. Candidates do not submit derived state or metric tables.

#### Local information

A node's decision snapshot contains the public static grid, capability table, task templates, and required release schedule; its own activity, FIFO, pending products, handoff sessions, fragments, known releases, received product metadata, and earlier local task/proposal feedback; its current incident directed capacities; and the globally shared remaining unit-attempt count. A rover also sees only its own position, energy, movement feedback, and science entry/release feedback. `E` has no additional global view. A node cannot inspect another node's position, energy, activity, data, release knowledge, task decision, proposal, or movement. Current decisions cannot use simultaneous other-node decisions or any future release, capacity, loss, arrival, movement, or activity. The environment and arbiter use global state solely to execute the published transition.

#### Metrics

For scenario `s`, let `on_s` be required chains whose `dispatch` output is available by their inclusive deadline, and let `N_s` be required task nodes whose output is available by their chain deadline. `primary_metrics_1` is the five-scenario mean of `0.75*|on_s|/12+0.25*N_s/48`.

Let `R_s` be released science zones, `V_s` their total value, and `S_s` the value whose `store` output is available at `E` by its inclusive science deadline. The per-scenario value is zero when `R_s` is empty; otherwise it is `S_s/V_s`. `primary_metrics_2` is its five-scenario mean.

For each on-time required chain, its finish boundary is the `dispatch` output boundary. Define `lat_s=clip(mean((deadline-finish+1)/18),0,1)`, or zero if `on_s` is empty. Let `move_s` be the number of non-`stay` moves. Let `energy_s` be all energy spent by the three rovers plus `E` computation and transmission energy; its fixed upper bound is `3*180+120*4+160*0.05=1028`. Then `primary_metrics_3` is the five-scenario mean of `(|on_s|/12)*(0.4*lat_s+0.3*clip(1-move_s/360,0,1)+0.3*clip(1-energy_s/1028,0,1))`.

A migratable task is `localize`, `plan`, or `analyze`. For each such task in a chain whose terminal result is on time and whose output its direct successor actually consumes, let `tau_src` be its duration on the chain source rover and `tau_node` its actual duration; its realized gain is `max(0,tau_src-tau_node)` when executed away from the source, otherwise zero. `G_s` is the gain sum. For the same task sample, `G_max,s` sums each task's maximum positive `tau_src-tau_candidate` over its legal nodes. Let `U_s` be delivered fragment units of products eventually consumed in an on-time chain, and `A_s` all attempted units. `auxiliary_metrics_1` is the five-scenario mean of `0.5*G_s/max(1,G_max,s)+0.5*U_s/max(1,A_s)`; a term with no actual sample is zero.

Immediately after task-start occupation, a slot--directed-link is a service opportunity when capacity is positive, both endpoints are noncomputing, the source has a nonexpired FIFO head with an uncompleted offset, and its session is absent or names that destination. This set is defined even when the shared attempt budget is exhausted. Let `service_s` be the fraction of these opportunities in which the link is selected and delivers at least one previously unsent unit. For every delivered unit eventually consumed in an on-time chain, `custody_queue_enter` is the boundary when the product is created at its current custody node or fully reassembled there, before any pending-to-FIFO admission delay. Queue wait is `attempt_slot-custody_queue_enter` and data age is `consumer_start-arrival_boundary`. Define `wait_s=clip(1-mean(queue_wait)/40,0,1)` and `age_s=clip(1-mean(data_age)/40,0,1)`; each is zero without a sample. `auxiliary_metrics_2` is the five-scenario mean of `0.4*service_s+0.3*wait_s+0.3*age_s`. All metrics are clipped to `[0,1]` and computed only from independent replay.

#### Result contract

`results.json` is a JSON object with exactly `scenarios`, an array containing the five scenario records in any order. `scenario` is the unique canonical key. Each record has exactly `scenario`, `moves`, `task_starts`, and `transmission_proposals`.

`moves` has exactly one row for every `(slot,rover)` in `{0,...,119} x {R1,R2,R3}`. A row has exactly `slot`, `rover`, and `action`, using the integer/string domains and movement enumeration above. Row order is arbitrary.

`task_starts` is dynamic. Each row has exactly `slot`, `node`, and `task_id`; `slot` is an integer `0..119`, `node` is in `{R1,R2,R3,E}`, and `task_id` is one of the 66 fixed IDs. At most one row may use a `(slot,node)` key and at most one row may name a task ID, including invalid candidates. Absence of a node-slot row means no task start. Row order is arbitrary.

`transmission_proposals` is dynamic. Each row has exactly `slot`, `src`, `dst`, `product_id`, and `priority`; `slot` is an integer `0..119`, `src,dst` are different public nodes, `product_id` is `p:<task_id>` for a fixed nonterminal task, and `priority` is an integer `0..9`. At most one row may use a `(slot,src)` key. Absence means no proposal. Row order is arbitrary.

All three decision arrays are indexed by their declared public keys, so their row order is immaterial. JSON numbers must be finite, duplicate object keys are invalid, and booleans are not integers. The grader never compares a candidate schedule, route, move, or priority to a reference policy and does not require per-slot state, checkpoint, task, contact, transfer, queue, release, or metric tables.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain the six non-empty Markdown sections `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The exact `results.json` contract above is reserved for canonical evaluation. Controlled comparisons are optional for the validity gate but support higher Judge levels. Record them in `report.md`, reproducible project code, or optional additional text/JSON project artifacts; any clear condition-level representation is accepted, and no extra filename or schema is required.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run completes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root and scenario-record containers", weight: 3}
- {id: result_evidence, desc: "move, task-start, and proposal fields have the declared forms", weight: 4}
- {id: canonical_coverage, desc: "every scenario and rover-slot move occurs exactly once", weight: 4}
- {id: decision_legality, desc: "movement, task starts, FIFO proposals, sessions, activity, age, and energy are legal", weight: 4}
- {id: replay_validity, desc: "independent replay preserves compute, product, transfer, release, and mission invariants", weight: 6}
- {id: primary_metrics_1, desc: "on-time required-chain and required-task completion", weight: 8}
- {id: primary_metrics_2, desc: "on-time stored value among released science chains", weight: 4}
- {id: primary_metrics_3, desc: "required progress with latency, movement, and total-energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful remote-compute gain and consumed-transfer efficiency", weight: 3}
- {id: auxiliary_metrics_2, desc: "contact service, queue wait, and consumed-data age", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the dependency, heterogeneous compute, DTN custody, and science-release formulation"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent dependency-scheduling and transfer model; Implementation/evidence: no executable task or product process."}
    - {score: 0.4, desc: "Report: DAG, activity, release, contact, or deadline relations are decisively incomplete; Implementation/evidence: state or timing materially diverges."}
    - {score: 0.7, desc: "Report: task eligibility, heterogeneous execution, movement contacts, fragmented FIFO custody, releases, and deadlines are accurate; Implementation/evidence: all scenarios implement the core model with limited noncritical omissions."}
    - {score: 1.0, desc: "Report: release, dependency data, activity competition, DTN handoff, energy, and task returns are precisely connected; Implementation/evidence: local information, event order, and auditable state are faithfully realized without material divergence."}
- id: method_design
  desc: "quality of the unified online placement, scheduling, routing, and movement method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable legal method or a prohibited method is used; Implementation/evidence: no reusable procedure."}
    - {score: 0.4, desc: "Report: local-only or fixed-node placement omits decisive dependency, contact, queue, or deadline effects; Implementation/evidence: decisions are unit-specific or inconsistent."}
    - {score: 0.7, desc: "Report: a legal unified method coordinates task starts, placement, routing, and movement under the principal temporal constraints; Implementation/evidence: one procedure covers all units with limited horizon or science trade-off handling."}
    - {score: 1.0, desc: "Report: critical paths, node load, revealed contacts, FIFO age, science value, movement, energy, and deadlines are jointly justified; Implementation/evidence: evolving local state drives task and transfer choices without scenario or task answer tables."}
- id: experimental_coverage
  desc: "coverage of canonical schedules and controlled compute-sharing or contact comparisons"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one incomplete condition is shown and compute sharing is not tested; Implementation/evidence: condition evidence is insufficient."}
    - {score: 0.7, desc: "Report: all canonical scenarios and one controlled capability, contact, queue, value, or deadline comparison are covered; Implementation/evidence: comparable condition-level task, transfer, and outcome records are retained."}
    - {score: 1.0, desc: "Report: complementary experiments include one compute-sharing or routing mechanism comparison and one capability, contact, queue, value, or deadline sensitivity or failure boundary; Implementation/evidence: reproducible condition records support both beyond aggregate canonical scores."}
- id: research_analysis
  desc: "evidence-based analysis of dependency placement, intermittent transfer, and science return"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts records; Implementation/evidence: claims are unsupported."}
    - {score: 0.4, desc: "Report: completion or value counts are repeated without causal dependency, contact, or queue explanation and limits; Implementation/evidence: evidence is insufficient."}
    - {score: 0.7, desc: "Report: node load, releases, contacts, data age, value, deadlines, and limitations are interpreted from actual comparisons; Implementation/evidence: records support the stated trends."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies load sharing, DTN routing, dependency latency, and a failure boundary while testing alternatives and scope; Implementation/evidence: reconstructible condition records support the central conclusions."}
