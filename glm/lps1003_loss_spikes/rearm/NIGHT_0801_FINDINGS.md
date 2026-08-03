# LPS-1003 Issue 2 — overnight session findings (2026-08-01, autonomous)

> **2026-08-02 CORRECTION (ARMING_MECHANISM.md):** the fix and the
> F1/F2/S geometry stand, but the causal framing "prefill and launch lose
> mutual ordering on the default stream / inversion imposed at the
> driver-channel layer" is superseded: the prefill never ran on the
> default stream at all. `torch.cuda.ExternalStream(0)` returns a fresh
> torch POOL stream per call (stream_ptr=0 treated as absent), so the
> fills and the out-alloc were on a rotating private stream while the
> kernel launched on the legacy stream — never ordered, warm or cold.
> Manifestation is gated purely by legacy-stream depth at submission.
> No driver anomaly; NVIDIA driver report withdrawn.

> **2026-08-01 PM CORRECTION (see DETERMINISM_MECHANISM_0801.md):** the root
> cause and fix below stand, but two explanatory clauses are superseded:
> (1) "deterministic erase boundary = fill-kernel block schedule" — wrong;
> the boundary is PyTorch TensorIterator's 32-bit-indexing **launch split**:
> fill_ on the >INT32_MAX-**byte** scores buffer is TWO kernels split at
> exactly numel/2 = row 7957, first half first (source-verified on torch
> 2.11.0 + observed live via device spy, 20/20). The destroyer is the
> second fill launch (F2) executing after S. (2) "rank15's smaller fills
> win the race" — rank15's buffer is 5% UNDER the byte threshold → single
> fill launch → no F2 to reorder; ranks 0–14 are all over it (rank14 by
> 0.36%) and all fired. 0/2059 corpus events start below row 7957.
> SM-resource starvation is also excluded (measured 100% co-residency), so
> the F1→S→F2 full-serialization signature is imposed at the driver/channel
> layer — that framing goes in the NVIDIA report.

Status at write time: Part A narrowing — mechanism reframed with direct
evidence; stream-pin A/B experiment queued. Supersedes the "CuTe indexer
kernel skips work items" framing in HANDOFF.md §sure-set.

## The reframe (evidence-backed): the kernel is innocent

Full 2D diff-mask dumps (adjudicator v2.1, node-local
/root/lps1003_local/adjud2/, laptop mirror scratchpad adj2_data/) show, for
every whole-segment destroyer event:

- Written rows are bitwise-correct out to their EXACT causal limit
  (last_bad = q_causal_offset + local_idx, to the element).
- The lost region = global out rows [7957, 15913] — exactly total_q/2 —
  which is MID-THD-segment (seg8 spans 6536-8417), mid-work-item
  (token 1421 of seg8), mid-4-row-store-tile, not 128-aligned, not
  2MiB-page-aligned.
- Segments 9-13 produce nothing; segments 0-7 perfect; seg8 written for
  local rows [0,1421) only.

The ONLY mechanism that produces this shape: the kernel executed with
**cu_seqlens_q == cu_true.clamp(max=7957)** — first-half descriptor — while
q_causal_offsets and cu_seqlens_k were read TRUE (written rows bitwise
correct). num_m_blocks_cur/num_n_blocks guards then skip exactly what we
observe, including the mid-store-tile cut (per-token OOB guard in the THD
scalar-store path). Scheduler (S1), offsets-read-zero (S4-zero), and
page-level TLB store-loss are all REFUTED for the destroyer by this
geometry: none can cut at token 1421.

A clamped-at-half cu is what the main-attention THD-CP path builds every
layer (same 15-int32 size class) → caching-allocator block reuse makes the
predecessor bytes deterministic → 203/203 (old) + 1388 (tonight) identical
events.

So: **run1 consumes a stale/predecessor version of ONE small input tensor
(cu_seqlens_q); reruns read the settled truth.** The corruption window
correlates with expandable-segments remap activity
(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this trainer AND in
prod): executions whose output allocation forces cuMemMap read stale
metadata; warm-block executions don't. This explains conn=1 masking (queue
serialization restores visibility order), empty_cache arming, rep1 healing,
rank15 immunity (smallest shapes -> warm blocks), and the low-state flips
(other/partial stale reads; background all_differ events are n-block-aligned
interior bands, same family).

## Prime fix candidate: TVM-FFI env-stream launch mismatch

jackrao/dsa-topk-stream-pin (worktree wt-streampin, uncommitted, +dsatopk3)
documents the same defect class for the TOPK kernel: compiled kernels
dispatch on TVM-FFI's thread-local ENV STREAM, which is never synchronized
with torch's stream state → launches have NO ordering vs torch-stream
producers (our cu cumsum/cat) → data race; conn=1 collapses HW queues and
masks it. If the indexer-forward JIT launch also lands on the env stream
(despite _interface.py passing the resolved torch stream into the compiled
artifact), the SAME pin — tvm_ffi.use_torch_stream() around the launch — is
the real fix (+dsatopk4), and "unrelated" in HANDOFF.md was premature.

Discriminating experiment queued (adjudicator v2.5, lever-only, no reboot):
per call pre/post clone of cu_seqlens_q (innocent-copy stale capture),
memory_stats map-delta correlation, one-time torch-vs-env stream identity
log, and mode-file toggles {pin_stream, sync_before} for the A/B:
empty_cache cycles with pin ON should stop firing if the env-stream theory
is right; sync_before is the coarser fallback discriminator.

## Discriminating experiments (all lever-only, warm trainer job 76)

Executed in sequence; each with empty_cache A/B against the ctrl payload:

1. **Clone capture (v2.5/P2):** an innocent torch copy of cu_seqlens_q
   enqueued immediately before the kernel reads TRUE values on corrupt calls
   (pre_eq_true=1 with d12=183M whole-half -inf) and map-op deltas are ZERO
   during the corrupt exec. DRAM at the runtime VA is provably fresh.
2. **Poison test (v2.6):** pass a fresh clone while poisoning the original
   tensor's memory → poison never appears in output (15/16 ranks diff=0; the
   one diff = ambient corruption of the REFERENCE exec). Per-call pointer
   marshaling by the JIT executor is correct; baked-pointer theory dead.
3. **PTX audit:** all scalar loads are plain ld.global.b32 (no .nc), grid is
   a pure shape function, tvm-ffi env stream == torch stream == 0.
4. **cop='cv' volatile-load kernel patch (P5), hot-swapped via module reload
   + fresh compile cache:** parity-validated off-trainer (bitwise equal, 25
   volatile loads in PTX), then **4/4 treated reps STILL fully destroyed**
   (9.0-9.7 nats). Cache-hierarchy staleness REFUTED.

Since volatile loads still return predecessor bytes while a same-stream
torch copy returns fresh bytes at the same VA, the surviving mechanism is
**stale address translation**: under expandable_segments (cuMemUnmap/Map VA
recycling), the kernel's SM/unit translates the metadata VA to the OLD
physical page (whose content = the predecessor tensor, deterministically
TE-CP's clamped cu) for a window after remap, when
CUDA_DEVICE_MAX_CONNECTIONS is unset. conn=1 serializes the queues against
the driver's map/invalidate ops → masked. No user-level kernel or wrapper
code can fix translation coherence → fix surface = remove the substrate
(PYTORCH_CUDA_ALLOC_CONF expandable_segments:False) + NVIDIA escalation
(needs Jack's approval).

**armH8 A/B (running):** rebooted the SAME golden config with
expandable_segments:False, everything else identical (conn unset, pristine
kernel). Predictions if substrate-causal: boot rep0 CLEAN (vs 10/10
destroyed historically), empty_cache ladder 0/N (vs 14/14), background
~0/call. OOM risk at 262k without expandable segments is the open cost —
Part B step 5 (real training steps) must watch memory headroom.

## FINAL ROOT CAUSE + FIX (nailed ~10:00Z, verified through Part B)

The stale-metadata framing was a mirage: any mechanism erasing out rows
>= 7957 produces masks identical to "cu clamped at total/2" (the per-row
diff region is just the kernel's full causal write extent). The remaining
ladder — arena (persistent true-content metadata buffers): still 4/4
destroyed; armH8 (expandable_segments:False): boot rep0 still destroyed;
CU_STREAM_LEGACY launch handle: still 3/3 — then two decisive positives:

- fill->launch device-sync inside indexer_fwd: 0/5 clean, re-arm 2/2 on
  removal (clean ABA);
- own-stream arrangement (prefill + launch on a dedicated stream,
  event-chained to the caller): 0/3 clean, zero wall cost.

**Root cause: the indexer_fwd `-inf` output prefill loses ordering against
the tvm-ffi-dispatched CuTe kernel launch when the caller is on the default
stream and CUDA_DEVICE_MAX_CONNECTIONS is unset — the prefill lands after
the kernel's stores and erases the trailing rows** (deterministic erase
boundary = fill-kernel block schedule; second half of the packed row →
"second zigzag chunk destroyed"; rank15's smaller fills win the race;
conn=1 collapses the queues → masked; cold starts widen the window →
rep0). The kernel, scheduler, metadata, allocator, and TLB are ALL
innocent.

**Fix (+dsatopk4)**: run the prefill + launch on a dedicated per-device
stream, event-chained to the caller's stream
(server/patches/cudnn-frontend-1.26.0-dsa-indexer-launch-stream.patch).
First fix attempt had an ExternalStream(0) event-chain bug (hung the
trainer, full-tensor d12) — corrected to use the real current-stream
object. Off-trainer parity: bitwise-identical outputs.

## Part B verification (armH9: FRESH boot, PROD env expandable=True, conn UNSET, fixed wheel)

1. **Determinism**: empty_cache lever x10 with bitwise double-exec:
   0 destroyed, 0 low, **0/2896 self-disagreements** (baseline ~25% of
   treated calls). Treated walls 109-148s = full remap cost paid, clean.
2. **Treated == control rep1**: every treated rep at healed reference
   (3.856/3.625/4.425/2.828/3.454/3.984/3.852) within ±0.05.
3. **Origin A/B**: fresh prod-faithful boot + ORIGINAL Mudith docs 0-6
   payload: **rep0 = 3.89/3.63/4.37/2.84/3.50/4.05/3.88 ≈ rep1 ≈ rep2 ≈
   rep3** (historically destroyed 10/10 devbox + 6/6 prod).
4. **150-rep soak with FULL logprob dumps** (armH9_evidence/
   b4_h9_logprobs.jsonl.gz): 0 destroyed, 0 low-state flips (~8%/rep
   baseline), **0/23984 calls with -inf rows** (~1%/call baseline).
5. **Training stability**: 12 real steps (forward_backward + optim_step)
   over the exact prod batches: train_mean_nll 0.766 -> 0.464, smooth
   descent, grad norms 7e-4..1.9e-3, no step-0 spike, no sawtooth
   (armH9_evidence/b5.log, b5_train.jsonl).
6. **PR OPEN: https://github.com/basetenlabs/trainers/pull/875** —
   +dsatopk4 wheel, patch file, regression test, README, uv.lock (branch
   jackrao/dsa-indexer-launch-stream, worktree
   trainers-wt-dsa-launch-stream). NVIDIA-report decision left to Jack
   (the default-stream/tvm-ffi ordering hazard likely affects other
   CuTe-DSL dispatch sites; topk had the same family issue — +dsatopk3
   stream-pin branch).

## Devbox end state (~12:45Z)

- NO trainer running. Two driver-orphaned GPU contexts (node0 GPU6 267GB,
  node1 GPU7 135GB; no live pids, per-GPU reset unsupported) block the
  next boot — needs a node reboot. Left over from the W1 in-place
  wheel-swap crash (NCCL abort during 16-rank module reload; do reloads
  only via fresh boots from now on).
- The CPFS venv's cudnn tree = byte-exact +dsatopk4 wheel content (the
  Part B evidence itself ran on stream-fix + the inert, parity-proven
  cv-loads variant; a byte-exact-wheel origin re-probe is the one nicety
  not completed — CI's GPU test on the PR covers the artifact).
- Node-local /root/lps1003_local on both nodes holds all raw campaign
  logs/instruments; laptop mirrors: rearm/armH9_evidence/ (incl. 150-rep
  full logprob dumps + 12-step training record), rearm/fix_candidate/.

## Instrumentation post-mortems (cost several hours; avoid repeats)

- pycode exec() installs give wrappers NO closures (module-globals) →
  closure-walking recovery finds nothing → each adjudicator wrapped its
  predecessor: at worst megatron → v2.1(x3) → v2.0 → v1(x3) → h6(x2) → raw
  = 18 kernel execs/call. Diagnose with a chain probe (probe_chain.py)
  BEFORE trusting any wrapper-level instrument.
- harness6 installs a MetaPathFinder on the api module: ANY
  importlib.reload re-wraps immediately (this killed the v2.4 install; its
  bare assert produced harness7's empty "pycode:FAILED:" log line). v2.5
  removes the finder before reloading.
- v2.1-era "whole-seg on every call incl. heal reps" was an artifact of the
  18x chain's allocation churn — but a diagnostic one: it proved the stale
  read tracks REMAP ACTIVITY, not the empty_cache lever per se.
- The NLL flip direction (destroyed-UP vs low-DOWN) depends on exec
  multiplicity per call; both are corruption. Treated reps under light
  chains tend to go LOW; heavy chains went UP. Don't use NLL direction
  alone as the firing criterion for A/Bs; use the wrapper-level d12 +
  cu-capture instead.

## Artifacts

- Laptop: scratchpad adj2_data/ (2059 mask events + jsonls),
  mine_adjud.py, analyze_masks.py, adjudicator2{,4,5}.py,
  adjud2_campaign.py, v24_campaign.py, probe_chain.py.
- Devbox node-local: /root/lps1003_local/ (all of the above + campaign
  logs p1.log, campaign.log, v24_mode.json).
- Trainer job 76 still warm; chain state after v2.4 mishap = h6-only →
  after v2.5 install = v2.5-only over pristine raw.
