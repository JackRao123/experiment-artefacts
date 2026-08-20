# E2 REVISED — legal overlap levers after the E2b verdict (borel, 2026-08-12 ~21:1x CDT)

STATUS: FINAL (~21:2x). L1 constraints resolved from source — L1 is LEGAL in
our regime and is the a2a-overlap lever we run. Run order in §3.

## 1. Verdict: E2b as specced in HANDOFF is DEAD (do not re-attempt)

Jack's directive is "overlap the compute and comms." HANDOFF scoped that onto
mcore's `overlap_moe_expert_parallel_comm` with "block+K recompute" as the
memory dial. Source (vendored mcore, submodule byte-identical to the box
clone; independently confirmed by gibbs on-box):

- The flag hard-asserts (transformer_config.py:2631-2642): recompute
  granularity != 'full', method None, num_layers None, no 'moe' in
  recompute_modules. Block+K needs granularity='full' + method='block' +
  num_layers=K → violates three asserts.
- 'block' is only honored under granularity='full' (transformer_block.py:624);
  selective + num_layers=K is separately rejected (transformer_config.py
  :1713-1718 — file corrected by serre's verification pass). There is NO
  per-layer recompute dial in mcore — selective wiring is per-module-type at
  model construction (attention.py:380, moe_layer.py:247), no layer index
  filter.
- The only legal memory knobs under the flag: selective modules minus 'moe'
  (usable on our MoE layers: core_attn, layernorm, moe_act). That is the E1
  family: measured OOM at 258.8 GiB with ONE in-flight microbatch (d1,
  stage 1). M=N holds min(M,PP)=2 in-flight → strictly worse. moe_act saves
  single-digit GiB (gibbs) — nowhere near the ~90 GiB gap.
- NOT the blocker: THD/variable-length shapes. The overlap machinery
  (combined_1f1b.py + model_chunk_schedule_plan.py) threads packed_seq_params
  per-microbatch and exchanges p2p shapes dynamically
  (p2p_communication.py:322). Memory is the only wall. Recorded so nobody
  re-litigates shapes.
- Bridge-side duplicate validator: comm_overlap.py:470-505 (fires before
  mcore's).

Conclusion for the morning report: making the full EP-a2a overlap viable at
131k requires upstream work (per-layer recompute support under the overlap
flag, or activation offload engineering) — a follow-up project, not a
tonight item.

## 2. Legal lever candidates (revised E2)

### L1 — `overlap_dispatch_backward_with_experts_wgrad` — LEGAL, our a2a lever
- Constraints (transformer_config.py:2700-2711, ALL of them): big overlap
  flag OFF ✓ (we can't run it anyway), TE ≥ 2.3.0 (verify on box:
  `server/.venv/bin/python -c "import transformer_engine as te;
  print(te.__version__)"` — expected ~2.1x, fine), delay_wgrad_compute OFF ✓
  (it's dead for us anyway — requires the big flag: :2690-2693).
  **NO recompute constraint of any kind** — composes with full recompute.
  No dispatcher/EP/VPP/dtype requirements. Implicit dependency:
  moe_grouped_gemm=True (TE GroupedLinear carries the deferral,
  transformer_engine.py:1974-1984) — trainer already forces it
  (megatron_config.py:53-54).
- Mechanism (moe_layer.py:496-497, :536-538, :427-435, :777-796): per MoE
  layer, expert weight-gradient GEMMs are deferred to a side CUDA stream
  that waits only on the expert dgrad-completion event — so they run
  CONCURRENTLY with the dispatch-backward all-to-all still in flight on the
  main stream. Pure intra-layer autograd + side-stream machinery: engages
  under plain non-interleaved 1F1B with M=N, no schedule involvement, and
  re-created correctly inside the full-recompute checkpointed backward
  (CheckpointFunction re-runs forward → shims rebuilt → same relative order).
- Plumbing: NOT reachable from trainer JSON today — trainer CommOverlapConfig
  (models/src/loops_models/control.py:149-153) is extra="forbid" with four
  fields, bridge's dataclass lacks the field too, and adding it to the
  trainer dataclass alone breaks the model_dump() splat at
  megatron_config.py:325-328. Cheapest correct hunk: new trainer config
  field consumed directly in _build_config (set
  provider.overlap_dispatch_backward_with_experts_wgrad = True), NOT routed
  through the bridge dataclass. Small, CI-testable — assigned to poincare.
- Honest EV: hides 1 of 9 a2a per layer (the dispatch-backward, once per MoE
  layer per backward) behind that layer's wgrad GEMMs, plus whatever
  straggler wait the a2a carries gets absorbed for free. ~+1.5-3% at
  d4/d16. Cheap: one hunk + one boot.
- Validation watch items (from source, not blockers): (a) DDP grad-hook
  lifecycle is split across TE/mcore (experts.py:891-892 TODO) — if the
  deferred wgrad ever failed to fire, loss would not train; (b) side-stream
  semantics inside reentrant checkpoint. Both are caught by the standard d2
  canary (loss band + grad-norm comparability) — treat grad-norm drift as a
  stop signal, not tolerance.

### L2 — E2a: VPP2 [18,20,20,20] + `overlap_p2p_comm` + FULL recompute
- Legal and memory-known (full recompute; 169-194 GiB measured). All code
  already on the branch: vpp config field (61c41d9e), per-chunk iterator fix
  (8c13ed31), M=N seam resolved in the 73c24b00 merge. M=N satisfies the
  interleaved group constraint (M=4 ≥ PP=2) that killed the first E2a
  attempt.
- What it buys: halves the pipeline fill/drain on the critical path
  (~(PP-1)×t_mb(stage) ≈ 7.3s → ~3.7s with half-size chunks) and
  `overlap_p2p_comm` (interleaved-only, schedules.py:2165) posts p2p
  eagerly.
- Honest EV: fill/drain is roughly constant per step while the step grows
  with M → ~+10% at d4 but only ~+3% at d16, where the headline lives.
  Worth one boot, not the main event.

### L3 — Diagnosis: traced d16 on the known-good M=N config
- BT_PROFILE_RANKS=0,8, config unchanged from the 1052 tok/s/GPU d16 run.
  Zero experiment risk; kineto tax at d4 was +1.5%.
- Why it may be the highest-value box run for the mission metric: the d8/d16
  under-predictions point at the CPU-blocked serialization class
  (4259 stream syncs + 5428 blocking host-device copies per step on the
  stage-1 rank, ~21s/step at d4 pre-M=N, hidden inside parks then — exposed
  now, and it GROWS with M). If the trace confirms it on the critical path,
  the recovered/rebuilt dispatcher host-sync caches (the Aug-9 work that was
  never landed — no branch, no PR) become the top +10%-class morning lever.
- Deliverable: rank-8 CPU-runahead analysis (maxwell's queued queries:
  GPU-empty gaps, loss-head wall, DSA-bwd host-sync fingerprint) at d16.

## 3. Recommended run order (box frees after parity legs B+C + save probe)

0. **L0 — JACK'S PROBE (direct request ~21:3x CDT, jumps the queue): selective
   granularity + VPP2 + overlap_moe_expert_parallel_comm=True, bounded
   memory ramp.** Jack's ask: "turn down recompute granularity, and then do
   this? see if it works." Assembly (all pieces exist): E1 selective config
   (configs/trainer_pp2cp8ep8_131k_selective.json) + vpp layout
   [18,20,20,20] (fields on 73c24b00 lineage) +
   comm_overlap.overlap_moe_expert_parallel_comm=true (JSON-reachable — it
   IS one of the four trainer CommOverlapConfig fields) + the
   moe_shared_expert_overlap clearing hunk (preserved patch on box, first
   hunk of pre_checkout_local_changes_0e0b65a6.patch — mcore asserts the
   shared-expert overlap OFF with the big flag). Pre-flight: torch ≥ 2.6
   (flag asserts it) + TE version. RAMP STARTS AT d2, not d1 — the
   interleaved schedule is illegal at M=1 (microbatch_group_size constraint,
   schedules.py:1141-1148; measured as E2a boot-2 failure). Abort > 255 GiB
   reserved; 10-min bound. Borel's prediction, stated for the record: OOM
   during warmup forward at d2 (~95% confidence) — E1 measured this memory
   class at 258.8/275 GiB with ONE microbatch in flight and no VPP; the
   saved-for-backward MoE tensors are schedule-independent. The honest 5%:
   the flag's combined-1f1b executor eagerly frees input storages and can
   release attention memory early (ep_overlap_early_attn_memory_release) —
   machinery E1 never ran. Possible clean early-exits that are also
   results: MTP+packing assert (fine_grained_callables.py:755-757) if
   mtp_num_layers is set; any validation assert → capture the exact text.
   If it FITS at d2: canary bars, then d4 A/B vs 918 — that would be the
   full prize and everything below reorders.
1. **L3 traced d16** — cheap, known config, feeds the morning report and
   ranks the next levers with data. The L1 hunk is being written in
   parallel (poincare) so it's reviewed and ready by the time L3 is done.
2. **L1** — one flag, one boot: d2 canary (grad-norm drift = STOP, see L1
   watch items) → d4 A/B vs 918 → d16 vs 1052. Pre-flight: TE version check
   on box.
3. **L2 (E2a)** — one boot, d4 + d16 A/B. Only if the night has box time
   left after L1; smallest EV of the three at d16.
One variable per boot; virgin-topology bounded probes (5-10 min); d1 ramp
only where memory model changes (L2: VPP raises in-flight chunk checkpoint
count modestly — d1 ramp first; L1/L3: no memory model change, straight to
d4/d16).
Canary bars unchanged: loss 12.2-12.4 band, grad norms comparable, peak
memory vs 275 GiB, traces pulled off the volatile dir immediately.

## 3b. Pre-registered decision rules for AFTER the L3 trace (Jack's
measure-first directive, ~23:1x CDT — data picks the lever, not vibes)

Decompose the traced d16 step into: pipeline fill/drain parks, a2a exposure
(comm time with no concurrent compute), CPU-blocked-on-critical-path (host
syncs/memcpys with GPU empty), optimizer+tail, other gaps. Then:
- CPU-blocked critical-path share ≥ 15% of step → the host-sync attack
  becomes the top post-L1 priority (rebuild the never-landed Aug-9
  dispatcher caches; scope the probs-a2a async fix). L1 still runs — it is
  cheap and independent.
- a2a exposure ≥ 20% with CPU-blocked < 15% → L1 first, then scope
  probs-a2a fusion/async as the follow-on.
- fill/drain ≥ 10% → L2 (VPP2) gains priority over the host-sync attack.
- Nothing dominant (all < 10%) → report the asymptote as structural;
  recommend regime-level changes for morning rather than more night boots.
Every lever ships only with a d2 canary + A/B vs 918 (d4) / 1052 (d16).

## 3c. Parallel track — box B (2×8 B300, seeker launched ~23:1x CDT)

Rationale: the remaining queue serialized ~4-5 h on one box; Jack directed
parallelization. Box B takes: (1) golden EP16/CP16 parity reference legs
(padded + unpadded boots) — poincare drives, hausdorff does box mechanics
after their sync-verify; (2) L4 below if time. Box A (w56lorq) keeps:
sync-verify → L3 → L0 → L1 → L2. One-seeker rule honored (no other seeker
active). Box B released as soon as its queue drains.

### L0b — THE BIG-WIN PROBE (added ~00:4x after the P2 source verdict):
selective recompute + overlap_moe_expert_parallel_comm +
`fine_grained_activation_offloading` at 131k. This is the legal,
upstream-supported path to the full a2a overlap: mcore's newer
module-level offload (saved_tensors_hooks based, async D2H/H2D streams
with backward prefetch) is explicitly CI-tested WITH the overlap flag +
selective recompute + PP2 + EP (unit test
test_fine_grained_activation_offloading.py:394; functional test at our
exact shape with NVTE_CPU_OFFLOAD_V1=1). Legacy `cpu_offloading` is
NO-GO (forbidden with recompute :1683 AND with PP>1 :1678 — do not spend
a probe on it). Requirements: poincare's plumbing hunk (ActivationOffload
config → provider fields), boot env NVTE_CPU_OFFLOAD_V1=1 (mandatory with
TE 2.16; bridge config.py:1971) + expandable_segments. Arm 1:
offload_modules [core_attn, attn_proj, expert_fc1, moe_act] (expert-side
modules are load-bearing — the 131k wall is ~90 GiB of MoE intermediates),
recompute_modules [layernorm, mla_up_proj, mlp, moe_act], VPP2, alltoall
dispatcher (NOT ncclep). d2 start, memory-abort, 10-min bounds. Ground
truth: the rank-0 "Activation Offload Summary (MB)" table. Risk ledger:
pad-to-131k constant shapes neutralize the two worst upstream risks
(pinned-pool growth keyed on (shape,dtype); warmup calibration frozen on
first call); LoRA+offload unvalidated upstream → canary bars load-bearing;
expert_fc1 offload forces bf16 raw-input saves (inflates before
deflating) and disengages the TE op-fuser (throughput delta unrelated to
PCIe); PCIe stall signature = compute-stream gap at expert-backward group
heads; back-off ladder = activation_offload_fraction down → drop moe_act.
Slots after parity legs + the cache-A/B decision point.

### L4 — outside-the-box feasibility probe (stretch):
the big overlap flag at CUSTOMER-REGIME seqlen. At 16k the selective
recompute it requires may FIT (the 258.8 GiB wall scales with sequence
length). A bounded d2@16k probe with the full E2b assembly would (a)
quantify the flag's real win where it is legal — hard data for the
upstream ask ("worth X% at 16k; per-layer recompute would unlock it at
131k"), (b) directly serve the customer regime (16k) from the Aug-9 sweep.
Only if box-B time remains after parity legs; d2 start, memory-abort,
10-min bound.

## 4. Status of the a2a overlap directive, in plain language (for Jack)

The specific Megatron feature that overlaps expert-communication with
compute cannot run at our sequence length: it demands we stop using full
activation recompute, and without full recompute the activations of a 131k
sequence do not fit in GPU memory (we measured 258.8 GiB against the 275 GiB
budget with even one microbatch in flight, and the schedule needs two). No
partial setting exists in the framework. What we CAN do tonight: overlap the
pipeline handoffs (L2), overlap one backward communication phase with
gradient compute if the smaller flag checks out (L1), and measure exactly
what the biggest remaining removable cost is at the 2M-token step size (L3).
The full expert-communication overlap needs an upstream framework change —
proposed as a follow-up work item, not attempted tonight.

Update after the source answer: the smaller flag (L1) checked out clean — it
overlaps one of the nine per-layer communication calls (the backward of the
token dispatch) with that same layer's expert weight-gradient math, costs no
extra memory, and works with our schedule and full recompute. Expected win
is small (one to three percent) but essentially free. We run the measurement
pass (L3) first, then this.
