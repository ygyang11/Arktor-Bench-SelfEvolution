---
id: T030_radio_aware_relay_exploration
name: Radio-Aware Relay Planning for Multi-Robot Factory Exploration
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

Four ground robots explore a fixed factory grid containing metal obstacles and weak-signal regions. The complete canonical table below, including obstacle and deterministic interference schedules, is public simulator input and may be used as a planning prior. Local sensing nevertheless determines which cells a source may put in a packet. Each robot can move, continue exploring, pause as a relay, or return toward the edge station. A direct path may be unavailable; a relay can forward a summary, but relay placement, queueing, interference, and energy use affect whether a high-value region is reported before its deadline.

### Research Problem

Design one deterministic, state-driven controller that emits one action for each robot at every slot. The controller may use the tuple of the four robots' current local observations and histories, arrived packet summaries, packet outcomes, queue feedback, and public protocol rules, but it may not use an unobserved cell in a packet. There is no implicit neighbour-summary channel: summaries arrive only through the explicitly modelled edge or relay hops. It must balance exploration coverage, reliable and fresh map delivery, report deadlines, safety, relay occupancy, and energy.

### Experimental Setting and Constraints

#### Canonical evaluation units

There are five canonical scenarios: `metal_corridor`, `split_weak_zone`, `relay_bottleneck`, `moving_interference`, and `coverage_deadline`. Each scenario uses an `18 x 18` grid with cells `x,y in {0,...,17}`, four robots numbered `0..3`, one edge station, and `H=80` one-second action slots. State boundaries are `state_t=0,...,80`; slot `t` updates boundary `t` to `t+1`. Each scenario has eight regions. A region whose lower-left cell is `(x,y)` contains `{(x,y),(x+1,y),(x,y+1),(x+1,y+1)}`. Every table column is public, including future deterministic interference; dynamic sensing, packet feedback, and queue outcomes still obey the timing rules below.

| scenario | starts; edge | closed obstacle rectangles | regions `(x,y)(value/deadline)` | interference |
|---|---|---|---|---|
| `metal_corridor` | `(1,2),(1,5),(1,12),(1,15)`; `(0,8)` | `[5,6]x[0,6]`, `[5,6]x[10,17]`, `[11,12]x[3,14]` | `(3,2)(1/24)`, `(3,13)(1/28)`, `(7,7)(2/38)`, `(8,14)(2/44)`, `(13,2)(3/52)`, `(13,7)(3/58)`, `(15,12)(4/68)`, `(15,15)(4/74)` | none |
| `split_weak_zone` | `(2,2),(2,15),(8,2),(8,15)`; `(0,8)` | `[8,9]x[4,13]` | `(4,3)(1/26)`, `(4,12)(1/30)`, `(10,2)(2/42)`, `(10,14)(2/46)`, `(13,4)(3/56)`, `(13,11)(3/60)`, `(15,7)(4/70)`, `(15,14)(4/76)` | endpoint `x>=10` adds `-8 dB` |
| `relay_bottleneck` | `(1,8),(4,2),(4,15),(8,8)`; `(0,8)` | `[7,11]x[0,6]`, `[7,11]x[10,17]` | `(5,4)(1/24)`, `(5,12)(1/28)`, `(9,7)(3/40)`, `(12,3)(2/50)`, `(12,13)(2/54)`, `(15,2)(4/66)`, `(15,8)(4/70)`, `(15,14)(4/74)` | none |
| `moving_interference` | `(2,2),(2,15),(15,2),(15,15)`; `(9,9)` | `[6,7]x[5,12]` | `(3,6)(1/26)`, `(3,13)(1/30)`, `(8,2)(2/40)`, `(8,14)(2/44)`, `(12,4)(3/54)`, `(12,11)(3/60)`, `(15,7)(4/70)`, `(15,14)(4/76)` | center `(4+(t mod 10),9)`, radius 2 |
| `coverage_deadline` | `(1,1),(1,16),(16,1),(16,16)`; `(9,9)` | `[4,5]x[4,13]`, `[12,13]x[4,13]` | `(2,7)(3/20)`, `(2,13)(3/24)`, `(6,2)(2/34)`, `(6,14)(2/38)`, `(10,2)(2/44)`, `(10,14)(2/48)`, `(14,7)(4/60)`, `(14,13)(4/66)` | slots `30..39` |

#### State, motion, observation, and link model

Robots occupy integer free cells and may propose one of the five moves `(0,0),(1,0),(-1,0), (0,1),(0,-1)`. Proposed moves are executed in increasing robot ID. Each proposal is checked against already accepted new positions and the still-current positions of higher-ID robots; pairwise swaps are rejected. A move outside the grid, into a closed obstacle rectangle, or violating the minimum Euclidean robot separation `1` is executed as stay and increments a safety event. A nonzero executed move costs `1` energy; every robot starts with `E_0=120` and may not spend below zero. The observation radius is Manhattan distance `2`. At every action start, each robot observes the obstacle/free class of every in-domain cell in that radius and obtains one radio sample at its current cell. The radio sample is the current-cell-to-edge RSSI formula below evaluated with `n=1`. Every cell observed by robot `r` at boundary `t` receives version `4*t+r+1` and observation boundary `t`; later observations replace an older local version. The edge keeps the largest version it has received for each cell. Versions are derived by replay and are not submitted in `results.json`. A robot may not put an unobserved cell in a packet. Future packet and queue outcomes are unavailable, whereas the deterministic table and formulas are public and predictable.

For a hop whose endpoints are integer cells `a,b`, let `d` be Manhattan distance and `o=1` when the segment intersects any closed obstacle rectangle, otherwise `0`. Let `n` be the number of hop-attempts accepted in that slot. The RSSI is `-35 - 4*d - 12*o - 8*h - 3*(n-1) - w`, in dBm. Here `w=8` only in `split_weak_zone` when an endpoint has `x>=10`; otherwise `w=0`. In `moving_interference`, `h=1` when an endpoint is within Manhattan distance `2` of `(4+(t mod 10),9)`; in `coverage_deadline`, `h=1` for `t=30..39`; in other cases `h=0`. A hop is eligible only when `d<=5` and RSSI is at least `-75` dBm. A successful hop arrives at the next boundary; a failed attempt is reported at that boundary and may be retried once in the next slot. The one-retry limit applies separately to each hop: a relayed packet may retry its source hop once and, after queueing, retry its forwarding hop once.

#### Information, packet, relay, and event timing

At boundary `t`, arrivals are fused and any newly complete region reports are formed first. The controller then uses the available local maps and prior packet/queue feedback to choose each robot's move, role, and one communication operation. A direct packet uses one hop from its source to the edge. A relay packet names one relay robot and uses source-to-relay followed by relay-to-edge; the relay must be in `relay` role and a forwarding hop requires it to stay. The relay has a FIFO of capacity `4`. At a boundary with simultaneous successful arrivals, the priority rule below is applied before filling available queue entries; arrivals beyond capacity are dropped. The shared channel admits at most four hop-attempt units per slot. Packets contain at most four distinct cell IDs; their versions and observation boundaries are the source's latest locally known values at generation and are derived by replay rather than submitted redundantly. A successful source-to-relay hop becomes eligible for relay forwarding at the next boundary. Simultaneous submitted hop attempts are ordered by the earliest deadline of a region containing any packet cell, then generation slot, source ID, and packet ID; a packet containing no region cell uses deadline `H`. The team may admit at most `120` hop-attempt units and each robot may generate at most `30` raw packets; relay forwarding and retries count only toward the team unit budget. A packet attempt uses one energy when accepted. Because four robots each submit at most one attempt, all otherwise legal attempts are admitted while team budget remains; a proposal beyond that budget is illegal. A failed accepted attempt may be retried once at the next slot. A packet arriving after boundary `80` is recorded as expired and has no effect. A failed first attempt remains pending only through its next-slot retry opportunity. If that retry is omitted, the packet is discarded after that slot; if the retry also fails, it is discarded at the following boundary. A discarded forwarding packet is removed from its relay FIFO, so later entries cannot overtake it during the retry opportunity.

Each robot action has communication operation `comm` in `none`, `new`, `retry`, or `forward`. `new` creates a nonnegative `packet_id` unique within that scenario and simultaneously submits its first hop; it supplies one to four locally known cells and an optional relay. `retry` references an accepted failed attempt from exactly the preceding slot. `forward` references the head of that robot's relay FIFO. `retry` and `forward` do not repeat cell contents because the referenced packet fixes them. `none` submits no hop. One robot submits at most one hop-attempt per slot. A packet with a relay is eligible only when that relay selected role `relay` in the source-hop slot; a relay FIFO entry remains queued until legally forwarded or the horizon ends.

The edge stores the newest received version per cell. Region `r` is reported exactly once at the first boundary where all four cells in its fixed set have been received with a version newer than zero. The report uses the arrival boundary and the observation boundary of each cell. The mission ends only after boundary `80`; late packets do not change reports. The event order is therefore: process arrivals, fuse, update relay queues, and form reports; form local observations; choose actions; admit and advance hop attempts; execute motion; update safety, deadlines, and energy; and record the post-event boundary. The new position is first sensed at the next action-start boundary; boundary `80` has no action and no new local sensing, but arrivals at boundary `80` are fused and may complete reports. The grader's derived state, queue, and mission records are post-event facts; submitted action row `t` is the request for slot `t`, and observations and packet outcomes are available to later decisions only at their action-start or arrival boundary.

#### Legal actions and boundary conditions

Moves, roles (`explore`, `relay`, `return`), packet cell references, relay IDs, retry attempts, queue capacity, shared channel capacity, energy, cell and region IDs, array lengths, strict JSON, finite numbers, and all index ranges are checked. A JSON `null` in optional packet or relay fields is the exact equivalent of integer `-1`/no packet; the grader normalizes both forms before applying every rule. A direct `new` packet uses `relay=null` (or `-1`). For `none`, both `packet_id` and `relay` are null and `cells=[]`; for `retry` or `forward`, `packet_id` references the existing packet, `relay=null`, and `cells=[]`. A candidate may include in a `new` packet only cells already present in that source's local map at the action-start boundary. Geometrically blocked or conflicting move proposals are resolved to stay by the public movement rule and count as safety events; malformed packet, relay, retry, forwarding, budget, or insufficient-energy actions are illegal and close the validity gate.

#### Task metrics

`primary_metrics_1`: a complete report by boundary `H` scores `1`; an incomplete region scores `0.5*received_required_cells/4`. Aggregate by region value within each scenario, then average scenarios. `primary_metrics_2`: an on-time report scores `clip(1-arrival_state/deadline,0,1)`, otherwise `0`, with value weighting. `primary_metrics_3` is the mean over scenarios of `0.5*effective_progress + 0.3*clip(1-safety_events/(H*4),0,1) + 0.2*clip(1-energy_used/(4E_0),0,1)`, where effective progress is the number of distinct free cells observed by any robot at an action-start boundary divided by the total number of free cells; every free cell has unit value. `auxiliary_metrics_1` averages `clip(1-age/8,0,1)` over the four cells of each completed report and then value-weights completed reports; a scenario with no complete report scores zero. `auxiliary_metrics_2` is the mean of `0.5*clip(1-accepted_packet_units/120,0,1)+0.5*useful_units/max(1,accepted_packet_units)` over scenarios; the efficiency term is zero when no hop is admitted. Only a successful final hop to the edge is useful, at most once, when one of the versions it installs at the edge is used by a region's first complete report. All five metric values and the normalized performance `(8*p1+4*p2+3*p3+3*a1+2*a2)/20` are independently replayed and clipped to `[0,1]`.

#### Result record contract

`results.json` is deliberately minimal. It contains only `scenarios`, one record per canonical scenario. Each scenario record has exactly `scenario` and `actions` and no other keys. `actions` contains exactly `4*80` rows, uniquely keyed by `(t,robot)`; array order has no semantic meaning. Every row has exactly `t,robot,dx,dy,role,comm,packet_id,cells,relay`. `t` and `robot` are integers in the canonical ranges; `dx,dy` encode one legal four-neighbour or stay proposal; `role` is `explore`, `relay`, or `return`; and `comm` follows the operation rules above. `cells` is a list of distinct integer cell IDs `cell=18*y+x`. JSON `null` and integer `-1` are equivalent only for `packet_id` and `relay`. No state, observation, link, queue, report, or metric summary is submitted: the grader derives these facts from the fixed model and the action sequence, so equivalent correct simulators are not rejected for choosing a different redundant record representation. All JSON numbers must be finite and no field is optional.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The workspace root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite `results.json`. `report.md` must contain non-empty sections titled `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

## Auto Checks

- {id: entry_point, desc: "the root run.py is present and statically executable", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates new results.json", weight: 4}
- {id: report_contract, desc: "report.md is non-empty and contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON, exact root and scenario structures are present", weight: 3}
- {id: result_evidence, desc: "the minimal action rows are complete, typed, uniquely keyed and auditable for deterministic replay", weight: 4}
- {id: canonical_coverage, desc: "all five scenarios and every (t,robot) action key occur exactly once", weight: 4}
- {id: decision_legality, desc: "motion and role domains, per-source observed packet cells, relay/retry/FIFO relations, budgets and energy are legal", weight: 4}
- {id: replay_validity, desc: "the action trace supports a complete deterministic replay of motion, radio, packets, queues, reports and metrics", weight: 6}
- {id: primary_metrics_1, desc: "independently replayed region report completeness by the horizon", weight: 8}
- {id: primary_metrics_2, desc: "independently replayed region report timing quality", weight: 4}
- {id: primary_metrics_3, desc: "independently replayed safe exploration and energy quality", weight: 3}
- {id: auxiliary_metrics_1, desc: "independently replayed completed-report freshness", weight: 3}
- {id: auxiliary_metrics_2, desc: "independently replayed useful communication and load efficiency", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "accuracy of the obstacle, RSSI, exploration, relay, packet, queue and report model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: no coherent model; Implementation/evidence: no executable exploration or communication process."}
    - {score: 0.4, desc: "Report: coverage or communication is described but local observations, relay, queue, or report timing are omitted; Implementation/evidence: only part of the process is recorded."}
    - {score: 0.7, desc: "Report: local obstacle/RSSI observations, relays, shared capacity, ages, energy, and deadlines are accurate; Implementation/evidence: all scenarios run with limited non-critical omissions."}
    - {score: 1.0, desc: "Report: radio quality and relay placement are causally connected to exploration and report delivery; Implementation/evidence: the minimal action trace is sufficient for the grader to reconstruct observations, links, hops, queues, and reports."}
- id: method_design
  desc: "quality of the unified non-training exploration, relay, and packet policy"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no executable method or prohibited training; Implementation/evidence: no reusable legal procedure."}
    - {score: 0.4, desc: "Report: fixed relay points, role sequences, or nearest-neighbour rules ignore interference, queues, and energy; Implementation/evidence: decisions are prefilled or materially inconsistent."}
    - {score: 0.7, desc: "Report: a legal reusable policy handles exploration, relay, and packet timing; Implementation/evidence: one procedure covers all scenarios with supporting records."}
    - {score: 1.0, desc: "Report: a justified reproducible policy handles value, RSSI margin, relay reachability, queue age, deadlines, safety, and energy; Implementation/evidence: actions are state-driven without scenario or absolute-slot action tables."}
- id: experimental_coverage
  desc: "coverage and controlled evidence for radio-aware relay effects"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment; Implementation/evidence: no checkable execution."}
    - {score: 0.4, desc: "Report: only one map or no relay/fixed relay condition is shown; Implementation/evidence: no evidence for radio effects."}
    - {score: 0.7, desc: "Report: all canonical scenarios and at least one relay, interference, capacity, deadline, or energy comparison are covered; Implementation/evidence: condition-level records support it."}
    - {score: 1.0, desc: "Report: controlled single-factor experiments quantify relay placement, radio quality, capacity/interference, value, and deadline failure boundaries; Implementation/evidence: records reproduce the claims."}
- id: research_analysis
  desc: "evidence-grounded conclusions and limitations"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: absent or contradictory analysis; Implementation/evidence: records cannot support conclusions."}
    - {score: 0.4, desc: "Report: coverage or connectivity values are listed without causal relay, age, queue, or exploration analysis or limitations; Implementation/evidence: claims are unsupported."}
    - {score: 0.7, desc: "Report: replayed records explain relay, interference, capacity, energy, and deadline trends while separating observation from inference; Implementation/evidence: evidence agrees."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies radio-map, relay, queue, and exploration-value interactions with explicit failure boundaries and scope; Implementation/evidence: condition-level evidence is reproducible."}
