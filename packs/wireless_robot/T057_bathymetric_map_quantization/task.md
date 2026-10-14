---
id: T057_bathymetric_map_quantization
name: Acoustic Bathymetric Map Quantization for Collaborative AUV Navigation
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

Three autonomous underwater vehicles survey a seabed while a gateway assembles their bathymetric submaps. Dead-reckoning bias accumulates during motion. Each AUV can quantize a sonar submap into depth centers, codeword indices and optional orientations, but all map uploads and returned pose corrections share one low-rate half-duplex acoustic channel. Omitting orientations saves acoustic bits while increasing gateway restoration work and may erase terrain structure needed for a cross-vehicle match. Upload delay, loss, gateway congestion and delayed corrections can therefore prevent accurately localized survey gates or a timely return.

### Research Problem

Design a deterministic, non-training, state-driven procedure that jointly chooses AUV motion, submap blocks, transmitted orientations, half-duplex uplink or downlink service, and the next gateway restoration job. One reusable procedure must operate across all canonical scenarios through the actor-specific information interfaces below. It should retain useful terrain structure and timely cross-AUV matches while balancing acoustic bits, restoration computation, dead-reckoning drift, ordered survey deadlines, safety, energy and return progress.

### Experimental Setting and Constraints

#### Canonical scenarios, terrain and mission

The grid has integer cells `x=0..23,y=0..17` at `1 m` spacing. AUVs are `A0,A1,A2`, the gateway is at `(12,0)`, and each scenario has `H=96` one-second decision slots `0..95` and state boundaries `0..96`. Forbidden rectangles are inclusive. The depth `z(x,y,t)` is in metres; an indicator is one when its condition is true and zero otherwise. The formulas are environment truth used for observation and replay, not a policy interface for an unobserved cell or future slot.

| scenario | initial `A0,A1,A2` | forbidden rectangle | depth `z(x,y,t)` | acoustic state | per-AUV energy reference `B` |
|---|---|---|---|---|---:|
| `flat_ridge` | `(1,2),(22,2),(1,15)` | none | `35+0.25*x+0.15*y+3*1[x>=12]` | normal | `115` |
| `steep_ravine` | `(1,1),(22,1),(1,16)` | `[10,13]x[7,10]` | `30+0.2*y+8*abs(x-12)/12` | normal | `120` |
| `orientation_shift` | `(2,2),(21,2),(2,15)` | `[7,9]x[7,10]` | `38+4*sin((x+y)/3)+4*cos((x-y+t/24)/4)` | congested at `32..55` | `122` |
| `acoustic_gap` | `(1,3),(22,3),(1,14)` | `[9,14]x[5,12]` | `32+0.18*x+0.22*y+3*1[y>=9]` | interrupted at `28..51` | `118` |
| `overlap_chain` | `(1,1),(22,1),(1,16)` | `[11,12]x[4,13]` | `36+3*sin(x/3)+3*cos(y/2)` | congested at `40..71` | `125` |

Each AUV has six ordered survey gates. The `A0` centers are listed below. For `A1`, replace every center `(x,y)` by `(23-x,y)`; for `A2`, replace it by `(x,17-y)`. Each gate is the Chebyshev-radius-one set about its center. The six inclusive boundary deadlines are respectively `20,34,48,62,76,88`.

| scenario | `A0` gate centers in order |
|---|---|
| `flat_ridge` | `(4,4),(8,8),(12,8),(16,12),(20,14),(22,16)` |
| `steep_ravine` | `(4,3),(8,6),(9,13),(15,13),(19,8),(22,15)` |
| `orientation_shift` | `(5,3),(6,12),(11,14),(15,10),(19,5),(22,14)` |
| `acoustic_gap` | `(5,4),(7,13),(12,14),(16,4),(20,8),(22,15)` |
| `overlap_chain` | `(4,5),(8,5),(10,14),(15,14),(19,9),(22,16)` |

At a post-motion boundary only the lowest-index incomplete gate can complete, at most one per AUV. It completes when that boundary does not exceed its deadline, the real position is in the gate, and the Euclidean error between real and estimated position is at most `2 m`. Early presence in a later gate has no effect. The return set is the Chebyshev-radius-one set around each AUV's own initial position and is tested at boundary `96`.

The initial estimate equals the real position. For AUV index `i`, every executed move adds its unit displacement and `0.04*((-1)^i,(-1)^(i+1))` to the estimate; `stay` adds neither. An on-time arrived correction is applied automatically before the slot decision. Requested actions have grid displacements `stay=(0,0)`, `N=(0,+1)`, `S=(0,-1)`, `E=(+1,0)`, and `W=(-1,0)`. First replace by `stay` every non-stay request whose target is outside the grid, forbidden, or satisfies `abs(z(target,t)-z(source,t))>4`. Next reject both directions of every two-AUV swap. Among remaining moves with the same target, keep only the smallest AUV ID. Repeatedly reject a move whose target is occupied by an AUV that now stays, until stable. All remaining moves execute simultaneously. Every requested non-stay action that does not execute is one safety event.

An executed move costs `1` energy. Every AUV sonar opportunity costs `0.1`, whether retained or not. Each served uplink bit costs its source AUV `0.0005`; each gateway compute unit costs `0.01` system energy. Downlink gateway transmit energy is outside the AUV budget. Team energy is the sum of these costs and uses the scenario normalization denominator `3*B`; `B` is not a hard feasibility cap.

#### Sonar blocks and fixed quantizer

Every AUV has a sonar opportunity at slots `7,15,...,87`, before that slot's motion. At pre-motion real position `(x,y)`, the submap origin is `(clip(x-3,0,16),clip(y-3,0,10))`; hence the window is always a complete `8 x 8` set of grid cells. It is divided into sixteen `2 x 2` blocks in row-major local order. Block `B(4*by+bx)` is written with two digits as `B00..B15`, for `bx,by in {0,1,2,3}`. Its vector order is local coordinates `(0,0),(1,0),(0,1),(1,1)`.

For a block vector `v`, define the ten-bit center `D=clip(floor(10*mean(v)+0.5),0,1023)` and decoded center `D/10 m`. All depths are nonnegative, so an exact half tie rounds upward. For `k=0..15` and vector index `j=0..3`, the dimensionless codeword is `c_k[j]=(2*((k>>j)&1)-1)/2`. Let `R(u,v)=(v,1-u)` and `M(u,v)=(1-u,v)`. Transform `T_o` applies `R` exactly `o mod 4` times, followed by `M` when `o>=4`. The permutation `P_o` moves the codeword value at input coordinate `(u,v)` to output coordinate `T_o(u,v)`. Reconstruction is `hat_v[j]=D/10+2*P_o(c_k)[j] m`, so its offset is `+1 m` or `-1 m`.

The fixed encoder minimizes `mean_j((v[j]-hat_v[j])^2)` over `(k,o)`. A tie within `1e-6` selects the smaller `k`, then the smaller `o`. At an opportunity, that AUV observes all sixteen block IDs, cell IDs, raw depth values and fixed `D,k,o`. These current observations are available to its block decision. It may submit no selected blocks and drop the submap, or select `4..16` distinct blocks; it may transmit the fixed direction for any subset of selected blocks. The packet has `48+14*n_block+3*n_orientation` bits. IDs are fixed as `S_<auv_id>_<slot>` for the submap and `U_<auv_id>_<slot>` for its uplink packet.

When a direction is sent, the gateway uses the encoded `o`. Otherwise it tests `o=0..7` and minimizes the sum of squared depth differences along common edges with already restored four-neighbour blocks of this same submap. Blocks restore in ascending ID order. A left or right common edge pairs equal local `y`; a lower or upper edge pairs equal local `x`. With no already restored neighbour choose `o=0`; ties within `1e-6` choose the smaller `o`. A block costs `2` gateway units plus `6` more when its direction was omitted. The gateway has `48` units per slot. Its capacity-eight system counts the current non-preemptive job plus waiting packets. Arrivals wait in `(arrival_slot,source_auv,submap_id)` order, but when no job is active the candidate may select any waiting submap. The job processes blocks in ID order across slots. An active job continues automatically and no new job may be selected until the next slot after it completes.

#### Acoustic transport, matching and corrections

Each AUV has a capacity-four uplink FIFO. Each AUV also has a capacity-six correction downlink FIFO at the gateway. All six directed links share one half-duplex channel; at most one FIFO head is requested per slot. For the selected AUV's pre-motion position, let `d` be Euclidean distance to the gateway and `depth_level=clip(floor((z(x,y,t)-30)/5),0,3)`. Acoustic `gap` is `0` when normal, `0.25` when congested and `1` when interrupted. The exact current quality is `q=clip(1-d/32-0.04*depth_level-0.15*gap,0,1)`. An interrupted link is unavailable with zero capacity. Otherwise its service capacity is the largest member of `{64,96,128}` no greater than `128*q`, or zero when no member qualifies. The same exact current probe is used for either direction.

A non-null channel request must name a direction with a nonempty, unexpired FIFO and an AUV whose current service capacity is positive. Service is the minimum of capacity, FIFO-head remaining bits, and, for uplink, the remaining team budget. Team cumulative uplink service is capped at `7200` bits; the final service is truncated to the remaining budget. A partial packet stays at the head. When a packet completes in slot `t`, it succeeds exactly when `((13*i+7*t+kind) mod 100)/100 >= 0.08+0.2*(1-q)`, where `i` is the AUV index and `kind=0` for uplink or `1` for downlink. Failure leaves the same packet at the head, resets its remaining bits to the full payload, and increments retries. Success removes it and schedules arrival at boundary `t+2+floor(4*(1-q))`.

An opportunity's new uplink packet is admitted after current-slot channel service and therefore is first serviceable next slot. A nonempty submap choice is invalid when that AUV already had four uplink packets at the decision boundary, even if current service later frees one. Uplink arrivals are offered to the gateway in `(arrival_slot,source_auv,submap_id)` order; overflow is discarded. Submap packets and gateway jobs may affect the map only at age at most 24 from the opportunity slot. Correction packets may be queued, served, arrive and apply only at age at most 24 from generation. At a boundary, arrivals are handled and then every state with age greater than 24 is removed.

Every two sonar opportunities from different AUVs are ordered by `(slot,auv_id,submap_id)`. They form a geometric match opportunity when their `8 x 8` windows overlap in at least eight cells and their opportunity slots differ by at most 16. Its ID is `M_<later_submap_id>__<earlier_submap_id>`, and its inclusive deadline is the later opportunity slot plus 24. Once both selected submaps have completed restoration, retain only geometric-overlap cells restored by both packets. At least eight such cells are required, and all following comparisons use this restored overlap. The depth RMSE must be at most `3 m`. For every horizontally or vertically adjacent pair of restored-overlap cells, take the sign of each submap's depth difference, using sign zero when absolute difference is at most `1e-6`; gradient agreement is the equal-sign fraction, or zero with no such pairs. Agreement must be at least `0.7`. A timely pair passing both tests is accepted once.

An accepted match creates one `96`-bit correction for the later ordered submap's source. Let the residual be the later-minus-earlier estimated position at capture minus the corresponding real position difference. The correction is `-0.5*residual`, with each component clipped to `[-3,3]`. Its ID is `C_<later_submap_id>__<earlier_submap_id>`. Corrections are appended in match-ID order to the target downlink FIFO after that slot's channel service; overflow is discarded. An on-time successful arrival is automatically applied once at the boundary.

#### Information boundary and event order

At the start of decision slot `t`, process scheduled arrivals and remove over-age uplink, downlink and gateway work; apply every correction that arrives on time. Next form the pre-motion sonar opportunity, if any, and reveal current exact link probes. The candidate then submits all local, channel and gateway decisions at one boundary. Replay first serves the selected channel FIFO, then continues the active gateway job or starts the selected waiting job and uses up to 48 units. A job completion evaluates newly available matches and generates corrections after channel service. Replay then filters and executes simultaneous motion. Finally it admits this slot's selected submap packets, updates dead reckoning, energy, ordered gates and the post-slot boundary state.

Boundary `96` processes only arrivals, expiration and automatic correction application that were already scheduled; there is no channel service, gateway computation, motion or sonar decision. Arrivals scheduled later remain in flight and cannot affect any metric.

An AUV-local actor may read the public static model and only its own estimate, next gate and completion index, cumulative own energy, previous requested and executed/rejected actions, its uplink FIFO length and each packet's ID, payload, remaining bits, creation slot, retry count and revealed outcomes, automatically applied corrections, current sonar blocks and fixed encodings, and its current exact uplink/downlink probe. It cannot read another AUV's real or estimated position, gate state, action feedback, unsent sonar content, or an unobserved/future terrain or transfer outcome.

The shared channel scheduler may read cumulative uplink bits, all six current probes and, for each FIFO, direction, AUV ID, queue length, and head packet ID, payload, remaining bits, creation slot, retry count and revealed outcome. It cannot read queued submap block content or correction vectors. The gateway scheduler may read arrived and restored submap source, window, selected blocks and encoded content, waiting and active job metadata, cumulative compute and completed match results, but not an unarrived submap. One implementation may coordinate these actors only through the stated interfaces. Real positions, terrain truth, safety filtering, future outcomes, trajectories, queues, restoration, matches, corrections and mission state are environment or post-decision evidence reconstructed by the grader.

An actor may simulate the public deterministic formulas conditionally from information already available to it and hypothetical future decisions. This is model-based prediction, not access to a future realized observation: the actor must not be supplied another actor's hidden state or a later terrain, queue, transfer, restoration, match, correction or motion outcome as current input.

Thus `auv_decisions`, `channel` and `gateway_submap_id` are requests submitted at the public decision boundary. Current sonar, estimates, probes, FIFO metadata and gateway metadata are observations or feedback already visible to their specified actor. Executed motion, service results, arrivals, restored blocks, matches, corrections, resource use and task completion are later evidence and are not result fields. The detailed event sequence above is authoritative.

#### Metrics

All five metrics are clipped to `[0,1]` per scenario and averaged equally over the five scenarios. `primary_metrics_1=0.6*survey_gate_completion+0.4*pose_quality`. The first term is completed gates divided by 18. `pose_quality=clip(1-pose_RMSE/6,0,1)`, where pose RMSE pools all three AUVs at boundaries `0..96`, exactly `3*97` Euclidean-error samples.

`primary_metrics_2=0.65*map_quality+0.35*loop_coverage`. Target cells are all non-forbidden grid cells. For each recovered cell use the record with the latest opportunity slot, breaking ties by submap ID then block ID; compare it with terrain truth at that record's opportunity slot. `map_quality=recovered_cell_fraction*clip(1-depth_RMSE/4,0,1)`, or zero with no recovered cells. `loop_coverage` is accepted geometric match opportunities divided by all geometric match opportunities, or zero when none exist.

`primary_metrics_3` is `0.45*return_fraction+0.35*clip(1-safety_events/(3*H),0,1)+0.2*clip(1-total_energy/(3*B),0,1)`. Return fraction is the fraction of AUVs in their own return set at boundary 96.

`auxiliary_metrics_1=0.6*useful_payload_bits/max(1,uplink_served_bits)+0.4*essential_orientation_yield`. A timely block is useful when it supplies a final map value or participates in an accepted match. Each useful block contributes its 14 base bits once, plus 3 bits if its direction was sent; a packet with at least one useful block also contributes its 48-bit header once. For each timely restored sent orientation, replay also reconstructs the direction that would have been inferred from the same already restored neighbours. The orientation is essential when using the sent direction reduces that block's four-value RMSE against its captured depths by at least `0.5 m`. Yield is essential count divided by all timely restored sent orientations, or zero with none. Served uplink bits include failed full attempts and retries.

`auxiliary_metrics_2=0.4*on_time_submap_fraction+0.35*correction_freshness+0.25*productive_service_fraction`. The on-time numerator is nonempty opportunity packets whose selected blocks all finish restoration by age 24; its fixed denominator is 33. Correction freshness averages `clip(1-(applied_slot-created_slot)/24,0,1)` over applied corrections, or is zero with none. Productive service is the arithmetic mean of two fractions: communication bits belonging to an uplink packet with a useful block or to an applied correction, divided by all served communication bits; and compute units spent on useful completed blocks, divided by all gateway compute units. A zero denominator gives zero for that subfraction.

Formal task performance is `(8*p1+4*p2+3*p3+3*a1+2*a2)/20` after every validity gate passes.

#### Result contract

`results.json` is strict JSON with root exactly `{"scenarios":[...]}`. There is exactly one object for each canonical scenario, in any order, with exact keys `scenario,slots`. Each scenario has exactly one slot object for every integer `0..95`, in any order, with exact keys `slot,auv_decisions,channel,gateway_submap_id`. Each `auv_decisions` array contains exactly one object for each `A0,A1,A2`, in any order, with exact keys `auv_id,action,selected_block_ids,orientation_block_ids`. `channel` has exact keys `direction,auv_id`.

`slot` is a non-boolean integer. IDs and actions are strings. `action` is one of `stay,N,S,E,W`. Both block fields are arrays of distinct strings interpreted as unordered sets and may contain only `B00..B15`. Outside sonar opportunities both are empty. At an opportunity, `selected_block_ids` is empty or has `4..16` members and `orientation_block_ids` is its subset. `channel.direction` is `uplink`, `downlink` or JSON `null`; `channel.auv_id` is an AUV ID or `null`, and the two are null together or non-null together. A non-null request must satisfy the current FIFO and positive-capacity rule. `gateway_submap_id` is `null` or a dynamic ID of the exact public form; it must satisfy the active/waiting job rule.

Every object rejects undeclared fields. Canonical keys are scenario, slot and AUV ID within a slot. Apart from the channel's paired direction/AUV selection, container order has no reference meaning. All submap, packet, match and correction IDs are generated by public events; candidates assign none. The grader accepts every permitted ordering and does not compare decisions with a reference policy. Observations, trajectories, encodings, links, served bits, packet states, queues, restored maps, matches, corrections, energy and task summaries are all derived and must not be added to this minimal result object. Replay uses absolute floating-point tolerance `1e-6`.

### Project and Report Requirements

You must not use machine learning, deep learning, reinforcement learning, model training, fine-tuning, or online learning.

The implementation must use a unified, reusable procedure to generate results from the task inputs, communication conditions, and relevant history across all canonical units. The project must solve each unit through this procedure rather than supplying pre-filled actions or result tables for individual units.

You have 60 minutes to complete this task. Your work will be stopped automatically when that limit is reached. We will run `python run.py` once. It must complete within 600 seconds.

The project root must contain `run.py`, `report.md`, and `results.json`. `run.py` is the only official entry point and must generate or overwrite root `results.json` from the project code and fixed conditions. `report.md` must contain non-empty headings named exactly `Problem Formulation`, `Method Design`, `Experimental Setup`, `Results`, `Research Analysis`, and `Conclusion and Limitations`.

The executed project and its report must include condition-level evidence from at least one actual controlled encoding or restoration comparison and one actual acoustic/gateway sensitivity or failure-boundary comparison with comparable fixed inputs. Evidence may be written in the report or in candidate-defined UTF-8 JSON/CSV artifacts and does not alter canonical `results.json`. The report must also identify each decision component's policy inputs and explain compliance with the actor information interfaces.

## Auto Checks

- {id: entry_point, desc: "run.py exists and passes static compilation", weight: 2}
- {id: official_run, desc: "one official run finishes within 600 seconds and creates fresh results.json", weight: 4}
- {id: report_contract, desc: "report.md contains all six required non-empty sections", weight: 2}
- {id: result_json, desc: "strict JSON has the exact root, scenario and slot containers", weight: 3}
- {id: result_evidence, desc: "slot, AUV, block-set, channel and gateway decision fields are exactly typed", weight: 4}
- {id: canonical_coverage, desc: "every scenario, slot and AUV key appears exactly once", weight: 4}
- {id: decision_legality, desc: "actions, opportunity sets, half-duplex nulls and dynamic ID forms are legal", weight: 4}
- {id: replay_validity, desc: "decision-boundary submap admission, selected FIFO service and gateway job requests are dynamically eligible", weight: 6}
- {id: primary_metrics_1, desc: "ordered localized survey completion and pose quality", weight: 8}
- {id: primary_metrics_2, desc: "gateway bathymetric map and timely cross-AUV loop quality", weight: 4}
- {id: primary_metrics_3, desc: "return, safety and energy efficiency", weight: 3}
- {id: auxiliary_metrics_1, desc: "useful quantized payload and essential orientation yield", weight: 3}
- {id: auxiliary_metrics_2, desc: "on-time restoration, correction freshness and productive service", weight: 2}

## Judge Rubric

- id: problem_modeling
  desc: "quality of the bathymetric quantization, acoustic transport and navigation model"
  weight: 8
  levels:
    - {score: 0.0, desc: "Report: the formulation is absent or materially wrong; Implementation/evidence: no executable underwater mapping and communication model exists."}
    - {score: 0.4, desc: "Report: encoding, direction restoration, half-duplex transport, gateway work, matching or correction timing has a decisive omission; Implementation/evidence: state or event order materially departs from the report."}
    - {score: 0.7, desc: "Report: block and direction choices, acoustic and compute queues, restoration, matching and navigation relations are accurate; Implementation/evidence: the core model runs across every scenario."}
    - {score: 1.0, desc: "Report: information timing and the quantization-transport-restoration-correction-task couplings and boundaries are precise; Implementation/evidence: code, grader-derived replay and any optional experiment artifacts faithfully realize every central relation with traceable state."}
- id: method_design
  desc: "quality of the unified state-driven quantization and scheduling method"
  weight: 15
  levels:
    - {score: 0.0, desc: "Report: no legal method is given or a prohibited method is used; Implementation/evidence: no runnable reusable procedure supports it."}
    - {score: 0.4, desc: "Report: fixed compression ratios, block tables or one FIFO rule dominate and a decisive coupling is ignored; Implementation/evidence: decisions are not state-driven or materially differ from the report."}
    - {score: 0.7, desc: "Report: one reproducible method uses state for blocks, directions, half-duplex service, restoration and navigation; Implementation/evidence: the same procedure runs across all units with limited long-horizon trade-offs."}
    - {score: 1.0, desc: "Report: the justified unified method jointly handles restoration value, orientation gain, acoustic and compute congestion, age, matching, drift and task slack; Implementation/evidence: evolving allowed state drives decisive choices without unit or slot action tables."}
- id: experimental_coverage
  desc: "coverage of orientation, transport, restoration and correction mechanisms"
  weight: 20
  levels:
    - {score: 0.0, desc: "Report: no credible experiment exists; Implementation/evidence: no checkable execution exists."}
    - {score: 0.4, desc: "Report: only canonical aggregates or unexecuted condition declarations are given; Implementation/evidence: records cannot test the central quantization-communication relation."}
    - {score: 0.7, desc: "Report: all scenarios and one executed comparable direction, capacity, loss, compute or drift condition are covered; Implementation/evidence: both sides use comparable inputs and condition-level records."}
    - {score: 1.0, desc: "Report: complementary single-factor experiments test orientation communication-compute trade-offs, correction timing and a failure boundary; Implementation/evidence: executed condition-level block, service, match, correction and task records reconstruct both comparisons."}
- id: research_analysis
  desc: "quality of evidence-based bathymetric map and navigation analysis"
  weight: 12
  levels:
    - {score: 0.0, desc: "Report: analysis is absent or contradicts results; Implementation/evidence: records cannot support the conclusions."}
    - {score: 0.4, desc: "Report: compression or error values are listed without a grounded mechanism or material limits; Implementation/evidence: the asserted explanation lacks support."}
    - {score: 0.7, desc: "Report: actual results explain terrain, orientation, waiting, correction and mission trends while separating evidence, inference and limits; Implementation/evidence: code and records support the comparisons."}
    - {score: 1.0, desc: "Report: controlled evidence quantifies restoration error, orientation budget, service timing and drift correction interactions, tests a failure boundary and limits scope; Implementation/evidence: reproducible condition records fully support the central claims."}
