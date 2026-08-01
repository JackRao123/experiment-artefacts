# LPS-1003 parity night — running notebook (2026-07-31)

Working notes, newest at the bottom. Proof-grade claims marked **[PROVEN]**,
working hypotheses marked [HYP]. Full context: `README.md` in this dir.

## Session identifiers

- loops session `8w6k4y3`, run `4q9zxjw`, trainer deployment **`5wolkzw`**,
  image `trainer-cuda13-sm103-0e0b65a` (verified on pod)
- prod pods: `baseten-trainer-5wolkzw-multinode-0` (k8s node e02-sg-e1n4vn65z0g,
  OS hostname `b300-1-r3dsin4d-0014`), `-0-1` (e02-sg-e1n4vn65z0t)
- devbox `tj-3y0gjkq` = nodes `b300-1-4wprtzyj-0003` (= tj-3y0gjkq-1, Slurm
  rank0, serves :8001) + `b300-1-h673xc6t-0010` (= tj-3y0gjkq-0, leader)
- launcher keepalive log: this session's scratchpad `launch_parity.log`
- local data: `parity/runs/prod_5wolkzw/` (P0/P1 dumps + fingerprint.json),
  `parity/runs/devbox_arm/` (D0/D1 dumps + fingerprint_rank0.json)

## 07:00Z — plain parity arms complete

**[PROVEN] Prod fires, devbox doesn't, simultaneously, same image commit.**
- Prod boot window (probes start 06:50:44Z, ~40s after READY): destruction in
  **7/8 window reps + 5/8 "steady" reps** before healing; datum_mean up to 3.14;
  victims ALWAYS partition-tail runs (p0:4-6, p1:10-14, p2:18-22, p3:27-30).
  Window lasted ~8 min this boot (far hotter than earlier events). Prod now 5/5
  lifetime windows.
- Devbox same minutes, same payload, BT_SKIP_FULL_WARMUP=1: 14/14 clean
  (0.760-0.768). Devbox now 0/10 lifetime windows.

**[PROVEN] Event anatomy: whole-document corruption.** Destroyed datums show
+4..+11 nats excess in EVERY position decile of the supervised span (vs devbox
clean rep, identical payload). Not localized, not boundary-only. Mild victims
(2-3 nats) show patchy deciles. Consistent with "document attends garbage keys"
class mechanisms; rules out masking/boundary artifacts again.

**[PROVEN] Healed prod == devbox computationally (token level).**
- wobble floors: devbox rep-to-rep rms 0.023-0.026 nats, max ~4, >1nat 0.011-0.016%
- prod healed rep-to-rep (p1 r6 vs r7): rms 0.049, max 15.5, >1nat 0.028%
- cross prod-healed vs devbox: rms 0.041-0.043 — INSIDE prod's own floor;
  >1nat 0.020-0.025% — between the two floors. No systematic offset.
- per-datum deltas: only d06 (+0.10) and d28 (+0.17) exceed 0.05 — both are
  partition-TAIL datums ⇒ residual sub-threshold simmering on prod, not a
  computational difference.
- batch mean: prod healed 0.7735 vs devbox 0.7657 (delta +0.008).

**[PROVEN] Binary/runtime identity (fingerprint diff, prod pod vs devbox rank0):**
- SAME: GPU (L20D), driver 580.105.08, VBIOS, kernel 6.6.102-5.200.al8, and the
  OS hostnames show BOTH are `b300-1-*` pool machines. Node-pool hypothesis dead.
- SAME (sha256): libcudnn.so.9, libcublas/Lt.so.13, libnccl.so.2, libtorch*,
  60/73 mapped .so total. Python 3.12.3 identical build. All key wheels
  IDENTICAL versions: torch 2.11.0+cu130, cudnn-frontend **1.26.0+dsatopk1**,
  TE 2.16.0, triton 3.6.0, cudnn 9.19.0.56, cublas 13.6.0.2, transformers
  5.8.1, cutlass-dsl 4.5.2, megatron-core 0.19.0, numpy 1.26.4,
  nvshmem 3.4.5, nccl wheel 2.28.9. **The parked "prod image vs devbox venv
  CUDA libs" hypothesis is DEAD — prod loads the same pip wheels, sha-matched.**
- Prod uses torchrun --nnodes=2 --nproc_per_node=8 like devbox. allocator conf
  expandable_segments:True on both.
- DIFF (residual candidates):
  1. env: devbox-only `CUDA_DEVICE_MAX_CONNECTIONS=1`, `MEGATRON_SKIP_GLOO_GROUPS=1`,
     `NVTE_FRAMEWORK=pytorch`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `NCCL_DEBUG`,
     `OMP_NUM_THREADS=1`; prod-only `NVTE_CUDA_ARCHS=103a`. The old prod-mimic
     experiment never matched these. CUDA_DEVICE_MAX_CONNECTIONS is a
     concurrency-semantics change ⇒ plausible timing/reuse amplifier. [HYP-1]
  2. glibc family 13 .so differ (Ubuntu 24.04.4 image vs 24.04.1 devbox pod) —
     host-side only. [HYP-2, weak]
  3. libnvshmem_host.so.3 sizes differ wildly (160MB vs 41MB) though wheel
     version is 3.4.5 on both — unexplained, but same version string. [parked]
  4. Unmeasured state/timing: cgroup CPU limits, k8s pod vs slurm task,
     boot-time op sequence (platform init/weight-sync handshake). [HYP-3]

## In flight (07:10Z)

1. **Prod traced pods**: LWS patched (initContainer writes
   /probe/sitecustomize.py, PYTHONPATH=/probe:/app/src, BT_LTRACE=1,
   BT_LTRACE_DIR=/probe/out). Pods rolling; next boot's windows will be traced
   → `ltrace_analyze.py` gives the FIRST DIVERGENT LAYER (event vs healed reps,
   same payload). This is the layer-by-layer bisection.
2. **Devbox D2 prod-env-mimic**: boot with run_trainer_node_prodenv.sh (the 6
   devbox-only env vars removed, NVTE_CUDA_ARCHS=103a added) + tracer.
   8 window reps + 6 steady. If it fires → repro achieved, bisect env vars next.
3. Tracer overhead gate: devbox traced steady must stay in 0.760-0.771.

## 07:12Z — traced phase launched

- Prod pods recreated via LWS patch; **ltrace armed on all 16 ranks** (pod logs
  show "[ltrace rN] GPTModel.forward wrapped" for r0-r15; /probe/__pycache__
  confirms import). Driver2 in pod: p2t_window x8 at READY, p2t_steady x8.
  Traces land in /probe/out (emptyDir — SURVIVES container crash-restarts,
  DIES with pod deletion: pull before deleting pods!).
- Devbox D2: job 34 booting run_trainer_node_prodenv.sh (prod-exact env) +
  tracer; driver at runs/d2_prodenv_0731_070818. d2_penv_window x8 then steady x6.
- LWS template edits STICK (not reverted by the platform operator) — useful
  operational fact for future prod instrumentation.

## 07:20Z — POSITIONAL LAW [PROVEN on 13 event reps]

**A doc is destroyed only if it has tokens in the SECOND HALF of its packed
row; docs entirely inside the first half are never destroyed.** Checked against
exact token offsets of batch-0's greedy packing:
- p0: safe d00-d03 (d03 ends 0.411·row), destroyed d04-06 (d04 spans 0.411-0.647,
  crossing 0.5)
- p1: safe d07-09 (end 0.397), destroyed d10-14 (d10 spans 0.397-0.518, crossing
  half=0.5 of rowlen; NOTE absolute-131072 threshold FAILS for d10 — it is
  fractional-of-row, not absolute position)
- p2: safe d15-17 (end 0.387), destroyed d18-22 (d18 crosses)
- p3: safe d23-26 (d26 ends 0.4952!), destroyed d27-30 (d27 starts 0.4952)
- p4 (d31, single 50k-token row = 19% of cap): NEVER destroyed in any event
  ever → short rows exempt (large-row condition, cf. large_occupancy compile
  needs / chunk-count geometry).
- Within a destroyed tail-run the amplitude is FLAT (mean frac-increasing 0.538
  ≈ 0.5 over 20 runs) — it is an extent effect, not a gradient.
- p1 was hit in 13/13 event reps this boot (persistently simmering); others
  come and go per rep.

Interpretation: with CP16 zigzag THD sharding, the second half of the packed
row is exactly the set of positions living in each rank's SECOND zigzag chunk
(chunks 16..31 = chunk #2 of ranks 15..0). ⇒ corruption enters in per-rank
chunk-2 processing — the bottom-right causal-offsets attention/DSA-indexer
path (all prod top-k calls run the dense-with-q_causal_offsets branch,
bottom_right_key_start set). The tracer's 2048-token bins across each rank's
local [chunk1|chunk2] will show whether layer stats diverge ONLY in chunk-2
bins, and at which layer. Predictions:
  1. traced event: anomalous bins concentrate in the second half of each
     rank's local T (bins 8-15 of 16 for a full 16384-token local shard).
  2. any repro attempt must use rows where docs cross the half-row boundary
     (batch-0 qualifies).

## 07:21Z — **DEVBOX REPRO ACHIEVED (first fire in 11 attempts)**

`d2_penv_window` rep0 (boot d2_prodenv_0731_070818, job 34): datum_mean 1.4721,
**destroyed {4: 5.18, 5: 6.76, 6: 10.62}** — the exact prod signature
(partition-tail run, whole-doc destruction) on devbox hardware, first window
attempt under PROD-EXACT env. Devbox env-mimic changes vs the 0/10 stock boots:
- REMOVED: CUDA_DEVICE_MAX_CONNECTIONS=1, MEGATRON_SKIP_GLOO_GROUPS=1,
  NVTE_FRAMEWORK=pytorch, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600,
  NCCL_DEBUG=WARN, OMP_NUM_THREADS=1
- ADDED: NVTE_CUDA_ARCHS=103a
- ALSO ACTIVE (confound to eliminate): the ltrace tracer.

Traced prod window also fired (p2t rep0: 12 destroyed incl d28=18.27 nats,
prod 6/6 lifetime) → tracer does not suppress; event+healed traces now exist on
BOTH prod (/probe/out) and devbox (runs/d2_prodenv_0731_070818/ltrace).

**Attribution matrix queued (each ~20 min/boot):**
1. prodenv WITHOUT tracer → fires? (kills/keeps tracer-as-trigger)
2. stock devbox env MINUS CUDA_DEVICE_MAX_CONNECTIONS=1 only [prime suspect:
   device-connection concurrency semantics]
3. stock env MINUS MEGATRON_SKIP_GLOO_GROUPS only
4. stock env PLUS NVTE_CUDA_ARCHS=103a only
5. repeat full prodenv for rate stats

## 07:45Z — LAYER BISECTION RESULT [PROVEN, prod AND devbox events]

Analysis: `ltrace_first.py` (event op vs ≥2 healed ops of identical call
signature; z vs healed variance; rel-min 2% gate). Devbox event = d2_penv
rep0 (destroyed 4,5,6); prod event = p2t rep0 (12 destroyed).

1. **embedding: CLEAN (z≈0) in every event op on every rank, both machines.**
2. **First divergent hook: `decoder.layers.N.self_attention.core_attention`
   (class DSAttention), with N=1 in almost every rank/op** (one prod op
   fired at L0; scattered later-N for non-destroyed partitions). Everything
   upstream is bit-identical; L0 usually bit-identical too.
3. **Divergent bins = 3-7 of 8** (local T=16384 = [chunk1|chunk2] of 8192
   each) — i.e. the SECOND ZIGZAG CHUNK (+boundary bin 3) — activation-level
   confirmation of the positional law.
4. **The nature of the corruption: plausible-scale WRONG CONTENT, not
   garbage.** At the first divergent hook: nnf=0 (no NaN/Inf anywhere),
   absmax normal (~0.006), rms/mean shifted 4-10%. Chunk-1 bins are
   BYTE-IDENTICAL to healed reps (astronomical z = tiny healed variance).
   ⇒ output is still a convex combination of V rows — the attention is
   reading a WRONG KEY/VALUE SET for chunk-2 queries. NOT fp garbage, NOT
   uninitialized-looking values in the output itself.
5. Non-destroyed partitions in the same event rep show scattered low-z
   flicker at later layers (sub-threshold simmering).
6. Caveat: the tracer's `self_attention.indexer` pattern NEVER MATCHED —
   real path is `self_attention.core_attention.indexer` (DSAIndexer inside
   DSAttention), and DSAttention calls `indexer.forward_before_topk` (module
   forward not used; top-k runs via dsa_cudnn_kernels functions). So the
   indexer/topk CONTENT is uninstrumented; "indexer clean" is NOT yet
   established. GLM-5.2: 78 TransformerLayers, MLP dense x3 (L0-2),
   MoELayer x75, all 78 attn = GlmAbsorbedMLASelfAttention wrapping
   DSAttention.

**Remaining split inside DSAttention: (a) top-k SELECTS wrong (scores/topk
corrupted) vs (b) attention GATHER reads wrong memory for correct indices.**
Instrument v2 (next boots): per-call top-k row-sum hash at the
`_indexer_top_k_one_chunk` seam (probe2) + healed-vs-healed hash stability
control; if event hashes == healed hashes while output diverges ⇒ (b);
hashes differ on chunk-2 calls at L1 ⇒ (a).

## 07:47Z — attribution: TRACER EXONERATED, env alone triggers [PROVEN]

attrib#1 (prodenv, NO tracer) rep0: destroyed {4: 4.97, 5: 8.23, 6: 9.99} —
near-identical victim set to the traced run. **Devbox repro is 2/2 boots under
prod-exact env vs 0/10 under stock env.** The deterministic repro procedure:
1. boot GLM-5.2 golden leaf on the 2×8 B300 devbox with
   `run_trainer_node_prodenv.sh` env (BT_SKIP_FULL_WARMUP=1)
2. fire batch-0 /forward the moment /health answers
3. rep0 shows destroyed partition-0 tail datums (4,5,6) at 5-10 nats,
   healing by rep1, wobble/simmer stays elevated (rms 0.06-0.09 vs stock 0.024).

Devbox events (2/2) hit partition 0 at rep0 only = the FIRST big forward after
boot — freshest allocator state; prod events run hotter/longer (probably due
to platform init/weight-sync op mix in the window).

Chain running unattended (attrib_chain.sh): conn:2 → gloo:0 → arch:0 →
nvte:0 → penvconn:0, i.e. single-var sufficiency arms (stock minus one /
plus NVTE_CUDA_ARCHS), then the necessity arm (prodenv but conn=1 kept).
conn arm carries harness2 (ltrace + dsa rowsum digests) → if it fires, the
selection-vs-gather split lands on the same boot. rowsum_analyze.py ready.
Prod mitigation patch (add CUDA_DEVICE_MAX_CONNECTIONS=1 to LWS) staged in
scratchpad/lws_conn_patch.json — apply only after the chain names the var.

## 08:13Z — **ROOT-CAUSE VARIABLE NAMED: CUDA_DEVICE_MAX_CONNECTIONS** [PROVEN sufficiency]

conn arm (STOCK devbox env, ONLY `CUDA_DEVICE_MAX_CONNECTIONS=1` removed →
prod-default multi-connection): rep0 destroyed {4: 5.79, 5: 8.65, 6: 9.97}.
Cumulative:
- conn=1 present (stock): **0/10** windows fire
- conn unset: **3/3** windows fire (prodenv×2 + conn-only×1), same victim set
  {4,5,6} every time ⇒ the repro is effectively deterministic per boot.

Interpretation: CUDA_DEVICE_MAX_CONNECTIONS=1 serializes ALL stream launches
into one HW work queue; unset (default 8+) allows cross-stream concurrency.
⇒ the bug class is a MISSING STREAM/EVENT DEPENDENCY in the GLM-5.2 DSA/CP
forward (consumer kernel racing its producer on another stream), masked
whenever launches serialize. Chunk-2-only corruption fits a race on the
REMOTE-KV delivery path (chunk-2 queries attend keys gathered from other
ranks / other chunks; chunk-1 keys are local) — e.g. attention/indexer
consuming gathered KV before the comm/copy stream finishes. Explains: boot
window (cold = slow producer = wide race), healing, per-boot victim
randomness, devbox-vs-prod split (devbox scripts hardcode =1; prod launch.sh
does NOT set it), Nemotron-CP4-clean (non-DSA path), byte-identical-replay
heal (timing not data).

NOTE devbox stock scripts set =1 because Megatron *recommends* it (TP overlap
ordering); prod containers never did. Neither is "wrong" per CUDA semantics —
correct code must carry its own stream deps; =1 merely hides the missing one.

Next: (a) conn arm ran harness2 → pull rowsum digests → selection-vs-gather
verdict; (b) prod mitigation A/B: LWS env patch conn=1 → recycle pods →
expect windows to STOP firing (0/N boots); (c) chain continues gloo/arch/nvte
(expect clean) + penvconn necessity arm (expect clean).

## 08:30Z — corruption bracketed INSIDE DSAttention [PROVEN]; rowsum inconclusive

conn-arm event (harness2 active) analysis:
- **Indexer input projections bit-clean during the event**: at L1 (first
  divergent layer) `linear_wk`, `linear_wq_b`, `linear_weights_proj` outputs
  have z=0 vs healed (bit-identical) on every rank checked, while
  `core_attention` (DSAttention) output diverges in chunk-2 bins. At L2 the
  linears diverge — downstream propagation of L1's wrong output. ⇒ fault is
  strictly inside DSAttention: [cuDNN indexer scores → CuTe-DSL top-k → KV
  gather → sparse attention], consuming something cross-stream.
- **rowsum digest instrument INCONCLUSIVE BY DESIGN**: healed-vs-healed
  top-k selections churn massively (sha1 mismatch 91-97% of calls; |Δtotal|
  up to 2.2e11 between two CLEAN reps) — huge tie plateaus reshuffle
  below-cutoff candidates without changing attention output. Event churn is
  statistically similar. Selection-vs-gather remains OPEN; needs a
  weight-aware instrument (e.g. top-k selected SCORES above softmax cutoff,
  or per-row max-score), not index sums.
- Stream plumbing notes for tomorrow's code read: DSAttention CP is
  allgather-only (mcore gather_from_sequence_parallel_region + index_select
  reorder — usually stream-correct); the top-k kernel launches via TVM FFI
  "env stream" (`make_fake_stream(use_tvm_ffi_env_stream=True)`,
  indexer_top_k_decode_varlen.py:653,709); indexer fwd + sparse attention are
  cuDNN graph executions with their own handle/stream binding. The missing
  dependency is somewhere in this wheel↔torch stream boundary. GLM path:
  cudnn/deepseek_sparse_attention/* in the vendored wheel.

## 08:33Z — **PROD MITIGATION VALIDATED: conn=1 → 8/8 clean boot-window reps** [PROVEN]

LWS env patch added CUDA_DEVICE_MAX_CONNECTIONS=1 to trainer 5wolkzw; pods
recycled (fresh boot = the historically 6/6-firing window); probes:
p3c_window 8/8 CLEAN, datum_mean 0.7629-0.7661 (tight devbox-clean band, no
elevated simmer at all) — on the SAME deployment/nodes that fired 13/16
window reps earlier tonight. Bidirectional causality complete:

| config | boot windows fired |
|---|---|
| devbox stock (conn=1) | 0/10 |
| devbox conn unset (any variant) | 3/3 |
| devbox stock minus MEGATRON_SKIP_GLOO_GROUPS (gloo arm, conn=1 kept) | 0/1 boot, 0/12 reps clean |
| prod stock (conn unset) | 6/6 (13/16 reps destroyed) |
| prod + conn=1 | **0/2 boots, 0/16 window reps** (p3c 8/8 + p4c 8/8 clean, 0.761-0.767) |

⇒ **Immediate customer mitigation: set CUDA_DEVICE_MAX_CONNECTIONS=1 in the
trainer env (image/launch.sh or platform env injection) for GLM-5.2 DSA
configs.** Root FIX is still owed in the DSA stream plumbing (see 08:30Z
bracket): the bug is a missing stream dependency; conn=1 masks it by
serializing HW launch queues (likely with some perf cost on comm overlap —
NB Megatron itself recommends =1 for TP overlap correctness, so this is
aligning prod with Megatron's documented expectation, not a hack).

## 09:05Z — prod teardown state

- All prod data pulled (P0/P1 stock windows, p2t traced + ltrace_prod,
  p3c/p4c mitigation, fingerprint). Launcher keepalive killed.
- **No teardown endpoint exists** (DELETE /v1/loops/{sessions,deployments,runs}
  → 405). Trainer deployment `5wolkzw` remains up IDLE on 2 e02-sg B300
  nodes, with the LWS carrying my edits: tracer initContainer/PYTHONPATH
  (inert unless BT_LTRACE set — which IS set; harness only logs to /probe/out)
  and **CUDA_DEVICE_MAX_CONNECTIONS=1** (the mitigated config). → Jack:
  deprovision via internal tooling when done, or reuse
  (LOOPS_REUSE_FROM_SESSION_ID=8w6k4y3) for follow-up prod tests.
- Devbox: attribution chain finishing (nvte, penvconn); each arm stops its
  trainer; chain end leaves squeue empty (verify).

## 09:46Z — NIGHT COMPLETE. Final attribution matrix

All boots: GLM-5.2-FP8 golden leaf, 2×8 B300 devbox tj-3y0gjkq,
BT_SKIP_FULL_WARMUP=1, batch-0 probe at /health, destroyed = datum NLL > 2.0.

| arm (one boot each unless noted) | conn=1? | window result |
|---|---|---|
| stock ×10 (prior sessions + D0 tonight) | yes | 0/10 boots fired |
| d2 prodenv + tracer | no | **FIRED** rep0 {4,5,6} @ 5-10 nats |
| attrib prodenv no-tracer | no | **FIRED** rep0 {4,5,6} @ 5-10 nats |
| attrib conn-only (stock minus conn) | no | **FIRED** rep0 {4,5,6} @ 6-10 nats |
| attrib gloo (stock minus MEGATRON_SKIP_GLOO_GROUPS) | yes | clean 0/12 |
| attrib arch (stock + NVTE_CUDA_ARCHS=103a) | yes | clean 0/12 |
| attrib nvte (stock minus NVTE_FRAMEWORK) | yes | clean 0/12 |
| attrib penvconn (FULL prodenv + conn=1 restored) | yes | clean 0/12 |
| PROD stock boots ×6 (incl. traced) | no | 6/6 fired, 13/16 + 6/8 reps destroyed |
| PROD + conn=1 (LWS patch), 2 fresh boots | yes | **0/16 reps** (0.758-0.771) |

⇒ `CUDA_DEVICE_MAX_CONNECTIONS=1` is NECESSARY (penvconn clean) and
SUFFICIENT (conn-only fires) as the toggle, on devbox and prod alike.
Fire rate when absent: 9/9 windows across both machines. Suppression when
present: 0/13 boots / 0/68+ window reps.

End-state: devbox idle (squeue empty, GPUs 0); prod deployment 5wolkzw idle
with mitigated env (see 09:05Z); all raw data local under parity/runs/
(prod_5wolkzw incl. ltrace_prod + p2t/p3c/p4c; d2_prodenv incl. ltrace;
attrib_conn incl. ltrace+rowsum; attrib_archive/<5 arms>).

Open items for the day session:
1. Fix PR: export CUDA_DEVICE_MAX_CONNECTIONS=1 in server/scripts/launch.sh
   (repo sets it nowhere today; decide global vs GLM-scoped; Megatron already
   recommends =1 — check throughput impact on a step-timing A/B).
2. Root fix: find the missing stream dependency inside DSAttention
   (wheel↔torch boundary: TVM-FFI env-stream top-k launch
   indexer_top_k_decode_varlen.py:653/709, cudnn graph handle streams,
   allgathered-KV consumption). Repro is deterministic → bisect by forcing
   torch.cuda.synchronize() between DSAttention sub-stages.
3. Selection-vs-gather split: needs a weight-aware instrument (top-k selected
   scores above softmax cutoff), index digests are churn-blinded.
4. Flag: Mudith's other live trainers (zq8ykgw, v31gz13, 232l1xw, 2qj08pw)
   still run pre-#814 image 5a4ae4d (session reuse skips image resolution).
5. Deprovision 5wolkzw when done (no API DELETE exists; internal tooling).

## Notes / cautions

- 3rd in-process init_trainer_server rebuild DEADLOCKS the trainer — max 2.
- LWS patch: unknown whether the platform reverts template edits; verify pods
  carry PYTHONPATH after recreation.
- Monitor for the prod driver died with the old pod — re-arm after new pod up.
- Mudith's OTHER live trainers (zq8ykgw, v31gz13, 232l1xw, 2qj08pw) run the OLD
  pre-#814 image `trainer-cuda13-sm103-5a4ae4d` (likely session-reuse skipping
  image resolution) — flag to Jack: crash-prone + spiky, worth a nudge.
# ── DAY SESSION 2026-07-31 — root-fix hunt (Jack: FIX, not mitigation). CAMPAIGN STOPPED 22:22Z per Jack; docs consolidated. ──

Method: the overnight deterministic repro (prod-exact env, conn UNSET,
BT_SKIP_FULL_WARMUP=1, batch-0 /forward at /health) turned every hypothesis
into a ~25-min single-variable boot. 7 boots today; **conn-unset devbox boots
are now 10/10 lifetime rep0-fires** (3 overnight + 7 today), victims {4,5,6}
every rep0, instrumented arms occasionally hotter (multi-partition rep0 and
mid-window flickers — first devbox re-excitations ever seen: streamfix1 rep4
{22}, armF rep1 {22}, armG rep5 {4,5}, armDg rep2/3).

## Full arm matrix (each = one boot, 10 window reps, identical payload/venv)

| arm | single delta vs prod-exact env | rep0 | reps 1-9 |
|---|---|---|---|
| streamfix0 | BT_TOPK_STREAM_FIX=0 (control) | FIRED {4,5,6} 1.50 | clean |
| streamfix1 | top-k launch pinned to torch stream | FIRED {4,5,6,18-22} 2.83 | rep4 {22} |
| armC | no expandable_segments (unset ALLOC_CONF) | FIRED {4,5,6} 1.56 | clean |
| armDg | torch.cuda.synchronize() after EVERY seq-par gather | FIRED 13 docs/4 partitions 2.49 | rep2 {14}, rep3 {13,14,22} |
| armE | NCCL_MAX_NCHANNELS=1 | FIRED {4,5,6} 1.47 | clean |
| armF | CuTe radix top-k replaced by torch.topk | FIRED {4,5,6} 1.42 | rep1 {22} |
| armG | scorescan observer only | FIRED {4,5,6} 1.64 | rep5 {4,5} |

## What today PROVED (all [PROVEN], each by a dedicated boot)

1. **The overnight "missing stream/event dependency on the consumption path"
   interpretation is FALSIFIED in every specific form tested:**
   - Top-k TVM-FFI env-stream launch: genuinely unsynchronized BY
     CONSTRUCTION (only kernel in the wheel launched via TVMFFIEnvGetStream;
     env stream = thread-local, defaults NULL, never coupled to torch;
     IndexerTopK.execute drops its stream arg) — **but in this trainer every
     top-k call runs at torch_stream=0 == env_stream=0** (in-situ streamlog,
     16 ranks), and pinning it (streamfix1) does not stop the bug. Real
     LATENT defect (branch jackrao/dsa-topk-stream-pin, wheel +dsatopk3,
     regression tests pass) — NOT the LPS-1003 defect.
   - NCCL delivery of gathered K/KV: full device sync between every gather
     and its consumers (armDg, wrap verified on all ranks) cannot be raced —
     still fires. EXONERATED as data source.
   - Caching-allocator expandable_segments: fires without it (armC).
   - CuTe radix top-k kernel: fires with torch.topk selection (armF, 59
     armed-replacement lines). Kernel EXONERATED as corruption source; the
     SCORES it reads (or the attention read of the selected KV) are already
     wrong.
2. **DSAttention forward runs on torch's DEFAULT stream** (streamlog); the
   only non-default-stream GPU actors during a /forward are NCCL comm
   streams (+ MoE side streams from layer 3 on).
3. Standalone kernel-level probes cannot reproduce the failure (fat producers
   self-defeat by SM occupancy; naive index-set detectors false-positive on
   the tie-degenerate cutoff — compare selected VALUES, never index sets).
4. armG's online "-inf interior tile" scorescan is INVALID as designed (it
   counted the ratio-causal upper triangle as dropped tiles — per-row causal
   limits, not segment key-lengths, define the expected-finite region).

## Standing conclusion + remaining suspects

Corruption originates INSIDE the fused DSA score/attention kernel path —
wrong-but-plausible scores or attention reads produced under ambient GPU
concurrency, masked when CUDA_DEVICE_MAX_CONNECTIONS=1 serializes launch
queues (prod mitigation on 5wolkzw remains validated for boot windows).
Remaining suspects, in order:
1. **cuDNN CuTe indexer forward kernel** — warp-specialized persistent kernel,
   six warp roles independently walking a CLC dynamic tile-scheduler queue;
   a co-residency-perturbed schedule that skips a tile leaves its scores at
   the -inf prefill => silent plausible mis-selection; chunk-2 rows have the
   most tiles (positional law); #814 proved this kernel family fragile.
2. **FlashMLA sparse forward** (reads selected KV).

## Ready-to-run next steps (built, deployed, not run — campaign stopped)

- **arm H (harness6, BT_DSA_DOUBLE=indexer,flashmla)**: run each suspect
  kernel TWICE per call, bitwise-compare (causal -inf identical across runs;
  ANY diff = genuine nondeterminism) => names the defective kernel in one
  fired boot. Launch: `bash parity/devbox_streamfix2.sh H` on the leader.
- If indexer: cutlass-dsl ships StaticPersistentTileScheduler with the same
  method surface as the CLC one — wheel patch swapping the scheduler is the
  root-fix candidate (+ NVIDIA report). If FlashMLA: kernel patch/upstream.
- Then the final A/B per Jack's spec: with fix reps 0-9 in clean band
  (0.760-0.771); without fix rep0 fires. Harness: devbox_streamfix2.sh.

Ops notes: never edit driver scripts a running bash may still read
(version filenames / write-new-then-mv — cost 40 min + an orphaned trainer
today); pkill -f over ssh must use runtime-assembled patterns. Devbox left
idle (squeue empty), all 7 arms' data pulled to local parity/runs/.

# ── EVENING 2026-07-31 (UTC 08-01) — minimization probe: cp1 minimal config is CLEAN ──

One boot, Jack's directive: shrink f — no CP, synthetic data, 5 fwds.
Config tp1/pp8/ep2/**cp1** (dp2) @64k (pp8 legal on 78 layers via
`_glm52_dsa.py` uneven layout), prodenv conn UNSET + BT_SKIP_FULL_WARMUP=1
(the proven trigger), payload = 2×50k synthetic datums (palindrome +
random-256-tile), 5 window /forward reps.

**Result: 0/5 fired.** Max per-datum |ΔNLL| vs rep0 = 0.0056 (signature is
5–11). No rep0-outlier structure: >1-nat token divergence in ALL rep pairs
(15–165 tokens) at two fixed intra-chunk offsets (213/102 mod 256) =
ambient near-tie wobble; means drift monotonically (warm-up), reps 2–4
tightest.

**Reading: cp>1 is part of minimal f** — expected structurally (THD packed
path + zigzag bins are gated on cp>1; cp1 takes padded-BSH `_pack_batch`).
Next minimal candidate = **cp2** (smallest zigzag-preserving; single node
possible). Synthetic data remains untested under cp>1. Full writeup +
artifacts: `../min64k/README.md` (run min64k_0731_234442).

## 2026-08-01 minimization step 2 — cp16 golden leaf @ 32k: CLEAN 3/3

Same trigger (prodenv conn unset + skip-warmup + window reps), only
max_seq_len 262144→32768; payload = random-tile 30k + real bundle datum
b21 i13 (29,985 tok, prefix 29,682 → supervised tokens at the row tail).
Single-doc partitions. Worst per-datum |ΔNLL| 0.021, rep0 lowest not
highest; per-token wobble has no rep0 structure (random-tile hits all at
offset 175 mod 256, max 4.99; mudith all pairs confined to the completion
band). **CP + trigger ≠ f — prod-scale row geometry (64–131k multi-doc
rows, per-rank local ~8k) is the prime remaining ingredient.** Next: cp16 @
64k+ multi-doc synthetic. Details: `../min32k/README.md`.

## 2026-08-01 minimization step 3 — cp16@32k MULTI-DOC (3×10k) window: CLEAN; two side findings

No fire (3/3 window reps; tail doc stable 2.69-2.72, rep0 lowest). But:
(1) **/forward zero-fills logprobs at weight-0 positions** — masked-payload
dumps only measure supervised bands; min32k boot-1 "prompt quiet" and an
apparent boot-dependent churn explosion were artifacts of this. Uniform
weights (Jack's no-masking directive) are now standing probe policy.
(2) **Ambient per-token churn noise floor on real text**: 12-18% of
positions swing >1 nat (max ~19!) between identical clean forwards, ALL rep
pairs, independent of row position/packing/solo; synthetics are silent
(saturated logits); datum means stable ±0.04. Matches "top-k digests churn
90% between clean reps". Detectors must key on asymmetric rep0 MEAN shifts,
not token-level deltas. Details: INVESTIGATION.md 2026-08-01 (later);
artifacts min32k/x3x10k_0801_011540/.

## 2026-08-01 POSITIVE CONTROL — partition 1 alone FIRES; cutoff = chunk 17/32; day's negatives validated

Batch-0 docs 0-6 only, uniform loss (Jack's design), golden 262k config,
same trigger: rep0 destroys docs 4/5/6 (4.78/8.36/8.60 vs healed 3.4-4.0),
1/1 boot. Chunk profile: 0-12 clean, 13-16 flicker (both directions — reps
1-2 unstable too), 17-31 hard-destroyed (−2 to −6.2 mean). Hard onset ~row
135k vs midpoint 127.5k. Template ~2k prefix immune everywhere; healed-rep
churn 2× higher in second half. f is now ONE 254.5k row / 7 real docs /
single partition. Row-scale threshold 30k↔254.5k unbisected — next 64k/131k.
Details: `../ctrl/README.md`.

## 2026-08-01 soak — steady-state second-half instability, ~8%/rep, NOT window-only

146 identical /forwards (9.5s each) on the fired-control boot: docs 0-3
pinned ±0.06; docs 4-6 coherently visit a LOWER-NLL state ~8% of reps
(doc5 3.98 median / 2.16 min — 2-nat coherent swing over 34.6k tok), rate
flat over the run. Leading (unconfirmed) inversion: the rare low state =
correct computation; ~92% of prod-env steady forwards semi-degraded on row
tails — every step, gradients included. All prior steady-state stability
lore was conn=1. DECISIVE NEXT: conn=1 boot + same payload → its level
names the correct state. Exemplar dump pairs + full series:
ctrl/soak_0801_015956/. Note the detection threshold irony: at 30k rows
the same-family effect may exist below our 2-nat/mean radar.
