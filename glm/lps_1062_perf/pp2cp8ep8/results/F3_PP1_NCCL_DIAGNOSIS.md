# F3 — PP1/CP8/EP8 grad-sync NCCL crash: code-level diagnosis + discriminator spec

Author: serre (K3), 2026-08-13 early CDT. Lane: diagnosis only — no GPU runs
were launched for this document; every claim below is from the failure logs or
from reading the exact code that ran on the box (local vendored copy verified
line-identical to box `trainers_main @ 73c24b00` at every cited line).

## TL;DR

The crash is a **single ~390 MB bf16 all-reduce of the LoRA-adapter gradient
buffer over the 8-rank intra-node DP+CP group**, issued at the tail of the
seq=64 startup warmup. Code reading **exonerates every DDP-side scaling
hypothesis** (bucket count, coalesced-group size, frozen-param coverage): with
this trainer config there is exactly **one bucket and one collective**, and it
is structurally *identical* to the collective that passes at PP2 all night —
same group geometry, same transport, same code path, only ~2× larger.

The CUDA error itself is **`cudaErrorContained` (226)** — a *sticky* device
fault class whose listed causes are "invalid accesses of peer GPU memory over
nvlink" or "certain classes of hardware errors" (CUDA `driver_types.h`
:713-720). Sticky means: the all-reduce is almost certainly the **messenger**,
not the culprit — the first CUDA call after the real fault returns this error
wherever it happens to land. The diagnosis therefore hinges on one question the
existing logs cannot answer (NCCL_DEBUG was off): **did the DP+CP
communicator's own transport setup fail (NCCL's fault), or did the comm init
cleanly and inherit a fault from an earlier kernel (a compute/comm kernel's
fault)?** The discriminator ladder below settles that with one bounded boot,
then bisects with single-env-var follow-ups.

## 1. Evidence (both boots, same signature)

| | boot 1 (lost raw log, lines from NOTEBOOK 02:0x) | boot 2 (this log) |
|---|---|---|
| config | PP1/CP8/EP8/**DP2**, 2 nodes, 16 ranks | PP1/CP8/EP8/**DP1**, 1 node, 8 ranks |
| failing rank | rank6 (raise) + rank5 (watchdog) | rank6 (raise) |
| error | `RuntimeError: NCCL Error 1: unhandled cuda error` + watchdog `CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error` on PG `DATA_PARALLEL_GROUP_WITH_CP` | `RuntimeError: NCCL Error 1: unhandled cuda error` |
| stack | finalize_model_grads → finish_grad_sync → start_grad_sync → `_coalescing_manager` → `allreduce_coalesced` (param_and_grad_buffer.py:660, distributed_c10d.py:2757) | identical |
| when | warmup tail | warmup tail (boot 01:38:08, crash 01:58:58 UTC — ~21 min, mostly JIT compile) |

Boot-2 log: box `lps1062_pp2/logs/legB_pp1cp8ep8_dp1_1node_nccl_fail_20260813_0159.log`
(sha256 5bb62a48…), Mac copy at `logs/` same name. NCCL_DEBUG was **not** set —
zero transport lines; that gap is what the discriminator boot fixes.

Counter-evidence (hardware exoneration, borel's discriminator): the PP2/CP8
clean-init boot *after* these failures ran the same intra-node CP8 grad
reduction cleanly (NOTEBOOK 21:2x). Leader GPU0 had nonfatal RLW_RXPIPE Xid
137/145 ~5 min before boot 2 — noted, discriminated against, keep as background
(see H5).

## 2. What the failing collective actually is (code-verified)

Trainer DDP config (`server/src/trainers_server/dp_worker/backends/megatron_bridge/megatron_config.py`):

- `use_distributed_optimizer=False` (:282) → grad sync = **all-reduce**, never
  reduce-scatter (param_and_grad_buffer.py:677-684 branch).
- `ddp = DistributedDataParallelConfig()` (:297) = all defaults:
  `overlap_grad_reduce=False`, `nccl_ub=False`, `grad_reduce_in_fp32=False`,
  `bucket_size=None` (distributed_data_parallel_config.py:15-48,118).
- CP (THD) independently forces `overlap_grad_reduce` off (megatron_config.py
  :97-100).

Consequences, each verified in mcore source:

1. **One bucket per buffer.** `overlap_grad_reduce=False` → `bucket_size=None`
   (distributed_data_parallel.py:70-72) → `_compute_default_per_buffer_param_layout`
   puts all params in a single bucket (param_and_grad_buffer.py:931-941, the
   `bucket_size is not None` split never fires). The traceback's
   `finish_grad_sync → start_grad_sync` at :754 (not the :762 first-batch
   overlap path) independently confirms overlap is off.
2. **Buffers cover trainable params only — frozen base weights are NOT in the
   grad buffer.** DDP skips `not param.requires_grad` params
   (distributed_data_parallel.py:114-116); `group_params_for_buffers` asserts
   `param.requires_grad` (:864). LoRA freezes the base; GLM-5.2's target list
   (lora_targets.py:143-174) covers attention (q_down/q_up/kv_down/proj), the 3
   dense-MLP layers, shared experts, and the LM head — and **deliberately
   excludes routed experts**, so every adapter param is non-expert
   (`param.allreduce` defaults True, param_and_grad_buffer.py:870;
   tensor_parallel/layers.py:945/1295 set it False only for routed-expert
   weights). → **exactly one buffer** (bf16, non-expert), hence one bucket
   group (partition_buckets Case 2, :1593-1607), hence **one all-reduce** per
   grad sync.
3. **Size estimate** (from HF config: 78 layers, h=6144, q_lora_rank 2048,
   kv_lora_rank 512, 64 heads, qk 192+64, v 256, dense-inter 12288 ×3 layers,
   shared-expert inter 2048 ×75 layers, vocab 154880, LoRA r=32):
   attention 1.788 M/layer ×78 + dense MLP 1.573 M ×3 + shared experts 0.590 M
   ×75 + head 5.15 M ≈ **194 M params ≈ 388 MB bf16** at PP1. At PP2 the split
   is ~93 M / ~100 M per stage (≈ 190/200 MB). Both are routine NVLink
   all-reduce sizes; **size alone is not a plausible NCCL failure mode** — but
   it is the only in-collective difference (see §3).
4. **The DP+CP communicator is used for the FIRST TIME at the crash.** The
   trainer passes `data_parallel_random_init=False` (backend.py:325), so
   `broadcast_params` never runs (dist_utils.py:350-352) — no DP+CP collective
   at init. The warmup forward/backward exercises the EP (MoE all-to-all) and
   CP (attention) communicators, which are *different* NCCL communicators from
   DP+CP. Inside `finalize_model_grads`, `finish_grad_sync` (:502) runs before
   the embedding/router-bias reductions (:532, :538; both vacuous here — untied
   head at PP1, frozen router bias). NCCL establishes transport connections
   lazily at a communicator's first collective → **P2P/cuMem transport setup
   for the DP+CP comm happens exactly at the crashing call.**
5. **The error is `cudaErrorContained` (226).** The watchdog string "Invalid
   access of peer GPU memory over nvlink or a hardware error" is CUDA's own
   (present in libcuda.so.1/libcudart.so on the box; documented at
   driver_types.h:713-720): a device exception *contained* by the GPU, after
   which "any further CUDA work will return the same error." It is **sticky and
   asynchronous** — the first host call that checks errors (here: NCCL's launch
   path inside `allreduce_coalesced`) reports it, regardless of which kernel
   faulted. NCCL Error 1 = `ncclUnhandledCudaError` = NCCL caught a CUDA error;
   with no NCCL_DEBUG there are no transport lines to say where.
6. **NCCL defaults on this stack** (venv NCCL 2.28.9,
   `site-packages/nvidia/nccl/lib/libnccl.so.2`; image's system 2.25.1 is not
   what torch loads — `torch.cuda.nccl.version()` = (2,28,9)):
   cuMem allocation path is **auto-ON** (`NCCL_PARAM(CuMemEnable,
   "CUMEM_ENABLE", -2)`, cudawrap.cc:45-49); NVLS is **auto** (default 2,
   nvls.cc:152); cross-process intra-node P2P maps peer buffers via
   `ncclP2pImportShareableBuffer` (p2p.cc:349). Allocator:
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (launch.sh:53) — torch
   memory lives in cuMem VMM segments too.

## 3. PP1 vs PP2 — the complete delta list

Same box, same build (73c24b00 + TF32 patch), same warmup op (backend.py
:212-224: seq=64 forward+backward then grad sync), same NCCL env. PP2/CP8/DP1
passes; PP1/CP8 fails at DP2 *and* at 1-node DP1.

| dimension | PP2/CP8/EP8/DP1 (works) | PP1/CP8/EP8 (fails) |
|---|---|---|
| layers per rank | 38 / 40 (split across 2 nodes) | **78 (all on every rank)** |
| weights per rank | ~55 GB FP8 | **~109 GB FP8** |
| grad buffer (adapters) | ~190-200 MB | **~388 MB (2×)** |
| DP+CP grad group | 8 ranks, intra-node | 8 ranks, intra-node (DP1) — **identical geometry** |
| schedule in warmup | combined 1F1B (M=N) | `forward_backward_no_pipelining` |
| pipeline p2p comms | yes (PP send/recv) | none |
| a2a / CP comm kernels per rank before the sync | 35-40 layers' worth | **75 layers' worth (2×)** |
| JIT kernel coverage per rank | stage-local set | full-model set per rank (~17 min compile) |

Note for the "per-rank state" hypothesis (borel's): the **golden
EP16/CP16/PP1** reference is *also* PP1 (78 layers, ~109 GB/rank) and is
production-hardened — so PP1-scale per-rank state is not sufficient by itself;
the trigger must interact with tonight's box/build/group-geometry. But also
note the golden config has never run on w56lorq (it was staged for box B), so
"PP1 works elsewhere" is not evidence the box is innocent — only that the
config class is not inherently broken.

## 4. Ranked hypotheses + cheapest discriminators

### H1 — An earlier device kernel faulted; the grad all-reduce is the messenger
*(most likely)*

Mechanism: some kernel before the grad sync executes an invalid NVLink
peer-memory access (or hits a contained hardware fault); the sticky
`cudaErrorContained` is then returned by the first error-checking call —
NCCL's launch path inside `allreduce_coalesced`. Candidates for the faulting
kernel, in order: (a) an NCCL P2P kernel on the EP or CP communicators (MoE
all-to-all / DSA context-parallel comms — 75 layers' worth per rank at PP1, 2×
PP2); (b) a compute kernel with a latent OOB that only touches unmapped memory
under PP1's memory layout (109 GB of weights shifts every allocation); (c) a
genuine NVLink hardware fault (see H5). PP1-specificity comes from *count* and
*layout*, not from a unique op — consistent with "same op passes at PP2."

- **Discriminator (cheap, conclusive): `CUDA_LAUNCH_BLOCKING=1`** on the
  repro boot. Synchronous launches surface the error at the true faulting
  kernel with a Python stack instead of at the all-reduce. Caveats: slows the
  compile phase; can mask timing-dependent races — a *pass* under CLB is itself
  diagnostic (timing-sensitive fault), a *failure* names the kernel.
- **Companion (run in the same boot): `NCCL_DEBUG=INFO` +
  `NCCL_DEBUG_SUBSYS=INIT,P2P,NVLS,GRAPH`** with per-rank `NCCL_DEBUG_FILE`.
  If all communicators (including DP+CP) report clean init/channel setup and
  the fault appears only at the first grad-sync kernel → H1 confirmed, H2/H3
  dead.

### H2 — DP+CP communicator transport setup fails at first use (NCCL-side)

Mechanism: the DP+CP comm's first collective triggers lazy transport setup;
the cross-process cuMem shareable-buffer import (p2p.cc:349) or a related
mapping faults/fails under PP1's memory map (109 GB weights + torch
`expandable_segments` VMM allocations). At PP2 (55 GB) the same setup succeeds.
Raw capacity is *not* the issue (~170 GB free at PP1) — this would be a
VA-layout / handle-import / fragmentation-class bug, which is exactly the class
that expandable-segments + NCCL-cuMem stacks have historically traded in.

- **Discriminator 1: `NCCL_CUMEM_ENABLE=0`** — forces the legacy
  cudaMalloc/cudaIpc P2P path. Pass ⇒ cuMem mapping path implicated.
- **Discriminator 2: `NCCL_P2P_DISABLE=1`** — intra-node comms fall back to
  SHM. Pass ⇒ the NVLink P2P transport itself is implicated (and H1's NCCL-kernel
  sub-case weakens).
- **Discriminator 3: `PYTORCH_CUDA_ALLOC_CONF=` (empty — disable expandable
  segments)** — diagnosis-only (perf-relevant, do not ship); pass ⇒
  allocator-layout interaction.
- NCCL_DEBUG=INFO from the H1 boot already shows *where* setup stops
  (p2p.cc/nvls.cc WARN lines name the failing call) — in the best case H2 is
  confirmed without any toggle boot.

### H3 — NVLS (NVLink SHARP / multicast) engagement

Mechanism: NCCL 2.28 auto-engages NVLS for intra-node all-reduces on
NVSwitch systems when the fabric manager allows; multicast bind/map faults
present as NVLink peer-memory errors (cf. nvls.cc:285's own warning text). The
grad all-reduce is the largest *all-reduce* the DP+CP comm runs; EP/CP comms
in the warmup do all-to-all/p2p, not NVLS all-reduce — so an NVLS-specific
fault would first appear exactly here, at both PP1 and PP2… PP2 passing makes
this weaker, but PP1's 2× message size can cross an algorithm-selection
threshold, so it stays on the list.

- **Discriminator: `NCCL_NVLS_ENABLE=0`.** One env var; pass ⇒ NVLS
  implicated. NCCL_DEBUG=INFO also prints the chosen algo per channel.

### H4 — DDP bucketing / coalesced-group / frozen-param coverage — **EXONERATED BY CODE** (no boot needed)

- Bucket count: 1 (overlap off → bucket_size=None; §2.1). Coalesced group: a
  single tensor (§2.2). Frozen params: excluded from buffers (§2.2). borel's
  "bucket count/size explosion" and "coalesced group size" candidates cannot
  fire in this configuration — there is nothing to explode.

### H5 — Hardware (background, currently exonerated)

Leader GPU0 RLW_RXPIPE Xid 137/145 at 01:33:45 UTC (~5 min pre-boot-2).
`cudaErrorContained` cause (b) *is* "certain classes of hardware errors", so
this cannot be fully dismissed — but the PP2 clean-init boot ran the same
intra-node CP8 grad reduction on the same GPUs *after* the failures and was
clean (borel's discriminator). Keep: check `dmesg -T | grep -i xid` on both
nodes for the repro window whenever the discriminator boot runs; escalate to a
hardware ticket only if fresh Xids correlate with the crash.

## 5. Recommended bounded-boot spec (for whoever holds a freed box)

**Boot A — observe (no behavior change intended):**

```bash
cd /root/.cache/user_artifacts
source env.sh
export BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062_pp2/trainer_pp1cp8ep8_131k.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062_pp2/trainer_server.json
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,P2P,NVLS,GRAPH \
NCCL_DEBUG_FILE=/root/.cache/user_artifacts/lps1062_pp2/logs/f3_nccl_%h_%p.log \
  bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh --num-nodes 1
```

- Bound: 25 min from dispatch (cold compile was ~17-20 min; node 0's kernel
  cache is now warm from the failed boot, so likely faster). Failure expected
  at the warmup tail if it reproduces.
- **Snapshot before ANY relaunch** (clobbering rule):
  `cp .devbox_up/trainer_srun.log lps1062_pp2/logs/f3_bootA_$(date +%Y%m%d_%H%M).log`;
  the per-rank `f3_nccl_*.log` files are already clobber-safe.
- Read: (i) does the DP+CP comm's init complete (channels/transport up) before
  the fault? (ii) any p2p.cc/nvls.cc/cuMem WARN or error lines; (iii) which
  rank(s) report first; (iv) `dmesg -T | grep -i xid` on both nodes for the
  window.

**Decision rules:**

- Transport-setup failure lines → H2/H3. **Boot B**: add `NCCL_CUMEM_ENABLE=0`;
  if still failing, `NCCL_NVLS_ENABLE=0`; if still failing,
  `NCCL_P2P_DISABLE=1`. First passing toggle names the path.
- Clean comm inits, fault at the first grad-sync kernel → H1. **Boot C**:
  `CUDA_LAUNCH_BLOCKING=1` (no NCCL toggles) → the traceback names the true
  faulting kernel; that kernel's owner becomes the bug.
- Boot A *passes* → heisenbug / environment-sensitive; do not keep rebooting
  blindly — diff the passing boot's NCCL INFO lines against the failed DP2/DP1
  expectations and report back to borel for re-planning.

Each boot is the same PP1/CP8/EP8/DP1 1-node config; nothing else changes.
Do not iterate past ~3 boots without reporting — the parity claim is already
restructured around the golden EP16/CP16 reference, so F3 is a correctness
investigation, not a nightly blocker.

## 6. Exonerated / ruled-out summary

- DDP bucket count, coalesced-allreduce group size, frozen-param buffer
  coverage — code-exonerated (§2, H4).
- DP2-specific cross-node path — DP1 1-node reproduces (the whole point of
  boot 2).
- Ship NCCL env (IB/RoCE settings) — present on all boots; PP2 fine; intra-node
  DP1 boot doesn't touch IB.
- Raw GPU memory capacity at PP1 — ~112 GB used of 288 GB at the crash point.
- Hardware — see H5 (exonerated by PP2-after-failure, keep Xid watch).
- NCCL version mixup — torch loads the venv's NCCL 2.28.9, not the image's
  2.25.1 (verified `torch.cuda.nccl.version()` + the venv lib contains the
  error string).
