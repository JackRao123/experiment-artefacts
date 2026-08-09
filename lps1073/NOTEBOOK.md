# LPS-1073 investigation notebook

> **STATUS: DONE — root-caused, mechanism-proven, fixed at tip of `main`.**
>
> This investigation was originally tracked as **LPS-1063** ("save_state /
> reload correctness mismatch on MoE trainers"), which consolidated two
> reports that turned out to be **unrelated bugs**. LPS-1063 was split:
>
> - **LPS-1073 (this folder)** — the Nemotron Ultra nightly gate failure =
>   TE fused-attn forward **nondeterminism** under CP (tail-padded THD).
>   NOT a `save_state` bug; the save/reload compare was just the test that
>   surfaced it. Fix merged: Megatron-LM#25 → Megatron-Bridge#31 →
>   trainers#994.
> - **LPS-1074** — MoE `save_state` checkpoint truncation (DSv4 /
>   Qwen3.5-397B-A17B): the actual `save_state` bug. **Open.**
>
> Folder renamed `lps1063` → `lps1073` on 2026-08-09. Inline `LPS-1063`
> references below are historical (they refer to the pre-split ticket).

Running log for the Nemotron nightly-gate failure → TE fused-attn forward
nondeterminism under CP (tail-padded THD).
Ticket: https://linear.app/baseten/issue/LPS-1073
Newest entries at the bottom. Keep updating as work proceeds.

---

## 2026-08-07 — session 1 (feynman)

### Question being answered

Are the two reports in LPS-1063 the same bug?
1. **Nightly failure** (Jack): Nemotron-3-Ultra-550B ckpt-roundtrip gate, 587
   logprobs, max |Δ| 0.175 after save→load (tolerance 1e-4). Run 31073249926,
   trainers @4ec66ce3, 4 nodes × 8 B200, TP8/CP4/EP32/PP1, LoRA r16, LR 2e-4.
2. **Jerry's report**: `save_state` succeeds but MoE checkpoint ~7× too small,
   suspected missing expert-LoRA + optimizer state (DSv4, Qwen3.5-397B).
   Prime hypothesis H1: `adapter_key_filter` misses grouped-GEMM expert FQNs.

### Static analysis conclusions (code-verified)

- The gate (`examples/kl_alignment_gate.py:760` `_ckpt_roundtrip`) loads
  **in-place** into the live model (`execute_load_state`, no re-init), the
  **same PEFT filter runs on save AND load** (`checkpointing.py:2772` @57d5f291),
  and module-level load is **non-strict on PEFT resume** (`checkpointing.py:2873`).
  ⇒ Jerry-style truncation (missing keys) is structurally INVISIBLE to the
  in-process gate: dropped keys are never requested at load, live post-step
  values persist, fwd1 == fwd0, delta 0. Missing optimizer state can't affect
  a forward at all.
- Therefore the nightly's 0.175 delta must come from tensors that DID
  round-trip returning different values (or a non-weight reload side effect).
  **Verdict: likely different bugs (~70/30)**; residual shared-root scenarios
  are EP-topology ones (async-DCP finalize race H3, resharding).
- Nemotron caveat from ticket confirmed relevant: Nemotron routed experts are
  not LoRA-targeted, so H1 has nothing to drop on Nemotron anyway.

### Repro platform

- B300/ali capacity was 0 fittable nodes; boxes tj-qr4m7r3 (fermi/LPS-950) and
  tj-qzlr0o3 (banach/LPS-1062) both claimed by other agent sessions — hands off
  qzlr0o3. A node freed (`e02-sg-e1n4vn65z0e`) → provisioned **tj-3m9469q**
  (job 3m9469q, 1 node × 8 B300, project jrao123-ali) via devbox-up.
- **Key fidelity finding**: the entire save/load path is byte-identical
  between the failing nightly pins (trainers 4ec66ce3 / bridge 57d5f291 /
  M-LM a69b6b95) and the B300 branch `trainer-cuda13-sm103` (0e0b65a6 /
  0f26fc36 / d3932e75): controller save/load functions identical, bridge
  checkpointing.py + peft/ identical, M-LM diff = 1 line pyproject. So the
  B300 box's stock stack exercises the nightly's checkpoint logic.
  (Also: server/uv.lock differs between 4ec66ce3 and current main via #910
  cuDNN swap — do NOT reuse main venv for nightly-pin source on B200 attempts.)
- Weights: `models--baseten--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4-dequant-to-BF16`
  snapshot c1a0675d, pre-cached on ali team_artifacts (1.1T). hendrycks_math
  dataset also cached.

### Harness (staged at tj-3m9469q:/root/.cache/user_artifacts/nemotron-repro-1node/, local copy ~/nemotron-repro/1node/)

- Topology shrink: TP8/CP1/EP8 on 8 GPUs (nightly: TP8/CP4/EP32 on 32).
  max_seq_len kept 262144. LoRA r16, LR 2e-4, same datum construction as the
  nightly gate (587-token probe, 8-datum mixed group).
- Driver `repro_roundtrip_1node.py` sequence:
  fwd_pre → forward_backward+optim_step → fwd0 → fwd0b (determinism) →
  save P1 → load → fwd1 (gate) → load again → fwd2 (idempotence) → save P2.
  Three-way discriminator: fwd1==fwd0 clean | fwd1==fwd_pre adapters-reset
  (Jerry-compatible) | neither ⇒ value corruption.
- Server instrumentation: `bt_state_probe.py` + 2 hooks in a PYTHONPATH-
  shadowed COPY of trainers_server (shared checkout untouched): per-rank
  per-tensor float64 stats (sum/abssum/sq/min/max) for all params, buffers,
  optimizer state, RNG at save_pre and load_post → $RD/statedump/.
- Offline tools: dump_ckpt_metadata.py (DCP key inventory + shard sizes),
  diff_state_dumps.py (fingerprint diff).

### Operational gotchas hit (also in ~/.claude memory)

- **CPFS zero-fill**: files scp'd through the relay to qr4m7r3 read as
  all-NUL (correct size!) from tj-3m9469q → JSONDecodeError char 0 at boot.
  Fix: rewrite via `base64 | ssh 'base64 -d'` from/through the consumer box;
  verify with `tr -d "\0" | wc -c`, not ls.
- **1-node boxes have no srun** but generated start_trainer.sh calls it.
  Bypass: `SLURM_NODEID=0 nohup bash run_trainer_node_repro.sh > log &`.
- **pkill/pgrep self-match**: patterns like `dp_worker.main` match your own
  ssh shell/monitor — use `[d]p_worker` style. A pkill self-match killed a
  session mid-script once (trainer relaunch didn't run; caught and redone).
- devbox-up re-provision rebuilt shared trainers_main → examples/.venv
  vanished; rebuilt with `uv sync` in examples/.
- sm103-branch loops SDK requires `run_id` on TrainingClient for save_state
  (pass run_id="nemotron-roundtrip-repro-1node"); the nightly-pin SDK didn't.

### Results so far (aborted run 1, still valid data)

- **Forward is bit-deterministic** at TP8/EP8/CP1 on B300: fwd0 vs fwd0b
  max |Δ| = 0.0 over 697 logprobs. Any gate delta is real state change.
- Training step effect on probe: max |Δ| 3.37, mean 0.054, 585/697 nonzero.
- Run 1 aborted at save_state (missing run_id — fixed). Trainer restarted
  pristine because run 1 had already trained a step (policy_version 1) which
  would muddy the fwd_pre-at-init discriminator.

### Current state (as of this entry)

- Trainer rebooting pristine on tj-3m9469q (boot ~12 min), fixed driver
  staged (md5 216f14bc). Monitor armed. Next: run driver end-to-end, read
  verdict + fingerprint diffs, then dump_ckpt_metadata on P1/P2.

### Phase 1 result — local-path roundtrip at 1 node: CLEAN (bitwise)

Pristine run on tj-3m9469q, TP8/CP1/EP8, local checkpoint path
(weight_sync disabled), trainers @0e0b65a6:

- fwd0 vs fwd0b (determinism): max |Δ| = **0.0** / 697 logprobs
- fwd0 vs fwd_pre (training effect): max 2.208, mean 0.053, 584/697 nonzero
- **GATE fwd1 vs fwd0: max |Δ| = 0.0** — NOT reproduced at this topology
- fwd2 vs fwd1 (double-load idempotence): 0.0
- step counter restored correctly (1)
- Fingerprints rank0: 7476 params, 98 buffers, 864 opt tensors, 5 RNG — all
  IDENTICAL across save→load. Probe machinery validated.
- Checkpoint inventory (iter_0000001/): 2505 keys, 1.25 GB, P1 == P2 size.
  **Optimizer state IS present** — keys are `optimizer.state.exp_avg.<param>`
  etc. (my first bucketing misread them as "adapter" because param paths
  contain `.adapter.`; corrected before claiming). Adapters on
  `mixer.in_proj/out_proj` + `mlp.shared_experts.*` only — routed experts not
  LoRA-targeted, per ticket caveat.
- Evidence: `evidence-phase1-local-roundtrip/` in this folder (validated JSON).

### Discovery that reorders the suspects

`nightly-correctness.yml` **provisions a real prod Loops trainer** for the
gate ("Provisioning a Loops trainer + sampler on prod"). So the nightly's
save→load went through **S3 upload → bt:// resolution → download**, which my
phase-1 run never exercised. Remaining suspect axes, reordered:

1. **Remote leg** (upload content, bt:// version resolution, download) —
   Jerry's H4 + new H6 "resolution returns wrong artifact". Partially
   testable at 1 node with `weight_sync: {type: local, path: <CPFS dir>}`
   (save publishes a mirror; load reads the MIRROR, not save_dir).
2. CP > 1 (testable 1-node as TP4/CP2/EP8, memory permitting).
3. EP32 / multinode DCP sharding (needs ≥2 nodes; 4 for full fidelity).
4. B200-arch/CUDA12 numerics (not testable on this box).

### Phase 2 result — weight-sync mirror roundtrip: CLEAN (bitwise)

Same 1-node topology, `weight_sync: {type: local, path: CPFS mirror}`.
save published to `weight_sync_mirror/weights/lps1063-roundtrip-p1` and load
read THE MIRROR (path in driver log confirms). GATE max |Δ| = 0.0, determinism
0.0, idempotence 0.0. ⇒ publish_full_checkpoint content + mirror read are
exact at this scale. The prod-only legs left untested: real S3 backend,
bt:// version resolution, download staging.

Suspects remaining (in order): CP>1 → EP≥16/multinode DCP → prod S3/bt://
resolution → B200/CUDA12 arch numerics.

### Phase 3 — CP axis (TP4/CP2/EP8, weight_sync disabled)

- First boot FAILED at warmup forward: TE context-parallel fused-attn loader
  (`cp_p2p_fwd_fused_attn → fused_attn_fwd`) raises
  `Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13`.
  Cause: devbox IMAGE is CUDA 12.8 (`/usr/local/cuda`→cuda-12.8 on
  LD_LIBRARY_PATH) while the venv stack is cu130. Only the CP>1 fused-attn
  path performs the ambiguous cudart discovery — CP1 boots fine. Prod B300
  pods run the cu13 image, so prod never sees this.
- Workaround escalation (each retried, earlier ones insufficient):
  1. `LD_LIBRARY_PATH`/`LD_PRELOAD` → still failed.
  2. Purged all cuda entries from `/etc/ld.so.conf.d/` (000_cuda, 988_cuda-12,
     gds-12-8, nvidia) + `ldconfig` → cudart gone from linker cache → STILL
     failed. TE's probe evidently dlopens absolute paths.
  3. **Physically moved** `libcudart.so{,.12,.12.8.90}` from
     `/usr/local/cuda-12.8/targets/x86_64-linux/lib/` to
     `/root/ldso-backup/cudart12-files/` (node-local, reversible; verified no
     venv .so NEEDs cudart-12 — TE itself NEEDs .13). Attempt 4 in flight.
  All backups under `/root/ldso-backup/` on tj-3m9469q.

### Phase 3 result — CP axis (TP4/CP2/EP8, seq 32768): CLEAN (bitwise)

After the cudart fix + OOM fix (warmup profiles at max_seq_len; TP4 doubles
per-rank dense weights → 262144 OOMed at warmup MoE compute; reduced to
32768 — CP-axis test doesn't need the big budget, phase 1 already covered
seq 262144 at CP1): GATE max |Δ| = 0.0, determinism 0.0, idempotence 0.0.
**CP sharding alone does not break the round-trip.**

1-node conclusion: local path, mirror publish, EP8, CP2 all bitwise clean.
The bug requires EP32/multinode DCP, the prod S3/bt:// leg, B200 arch, or a
combination.

### Phase 4 — full-fidelity 4-node repro (in flight)

8 whole B300 nodes freed up (~20:15 UTC) → provisioning a 4-node devbox for
the EXACT nightly topology: TP8/CP4/EP32/PP1/ETP1/DP1, seq 262144, LoRA r16.
Remaining deltas vs nightly: B300+cu13 (vs B200+cu12) only.
To replicate on the new box: cudart-12 removal on ALL 4 nodes (srun fan-out),
4-node srun start scripts regenerated from new .devbox_up, examples venv
rebuild if provisioning wiped it. banach notified (pinned, immune).

### Phase 4 setup (tj-w7977d3, 4 nodes × 8 B300)

- Provisioned ~20:15 UTC as job w7977d3; devbox-up REUSED the shared
  trainers_main clone (venvs survived, no zero-fill on staged files).
- cudart-12 purged on all 4 nodes via srun fan-out (backups in
  /root/ldso-backup/ per node).
- 4-node srun launch scripts generated as run_trainer_node_repro4.sh /
  start_trainer_repro4.sh (separate names from the 1-node variants).
- Trainer dispatched TP8/CP4/EP32/PP1, seq 262144 — exact nightly topology.
- tj-3m9469q torn down (phases 1-3 complete, evidence harvested; full
  per-rank dumps persist on project CPFS user_artifacts).
- banach note: qzlr0o3 restarts every ~25 min through the night — expected
  churn, not a free box.

### Phase 4 RESULT — full nightly topology (TP8/CP4/EP32, 32×B300): TWO FINDINGS

**Finding 1 (headline): forward is NONDETERMINISTIC at the nightly topology.**
fwd0 vs fwd0b (back-to-back, NO save/load between) over the same fixed datum:
max |Δ| = **1.6188**, mean 0.0340, 585/697 logprobs differ. At 8 GPUs
(EP8, CP1 or CP2) the same comparison was bitwise 0.0 across three phases.
The nightly gate's failing delta (0.175 max, and 0.199 the night before)
sits WELL INSIDE this nondeterminism envelope — the ckpt-roundtrip gate at
tolerance 1e-4 cannot distinguish checkpoint corruption from ordinary
forward nondeterminism at this topology. The 0.175 "save/reload mismatch"
is very plausibly not a checkpoint bug at all.
Likely mechanism: MoE all-to-all / grouped-GEMM accumulation-order variation
across EP32 dispatch (token routing order differs run-to-run at 32 ranks).
Caveats: B300+cu13 stack (not the nightly's B200+cu12); single pair of
forwards (one sample, though 585/697 nonzero is unambiguous).

**Finding 2: load_state_with_optimizer deadlocked at 4-node** (client
ready_timeout 3600s expired; GPUs pinned at 100% busy-wait). py-spy stacks:
- rank 0 (asyncio_0 executor thread): execute_load_state →
  checkpoint_manager.load → megatron dist_checkpointing.load →
  `determine_global_metadata → all_gather_object` (validation.py:497) — waiting.
- peer rank: `_peer_loop → broadcast_object_list` (worker.py:241) — waiting
  for the NEXT OP, i.e. it never entered the load op. Mismatched collectives
  on one PG = permanent hang.
The prod nightly's load COMPLETED (its gate compared logprobs after load),
so this may be devbox-dispatch-specific — but the op-dispatch race it
implies (rank0 starts executing before peers receive the op broadcast?) is
worth eyes. Stacks + logs in `evidence-phase4-fullfidelity/`.

Save-side note: save_state at 4-node SUCCEEDED (P1 written, all 32 ranks
dumped save_pre fingerprints — on CPFS under statedump-4node/).

### Disposition

- tj-w7977d3 (4-node) torn down immediately after evidence pull (~15:28 PT).
  tj-3m9469q torn down earlier. No GPUs held.
- Evidence dirs in this folder: phase1 (local roundtrip clean),
  phase2 (mirror roundtrip clean), phase3 (CP2 clean),
  phase4 (nondeterminism + deadlock stacks).

### Implications for LPS-1063 (to discuss with Jerry / ticket update)

1. The nightly Nemotron gate failure is likely **forward nondeterminism at
   EP32-scale**, not save/reload corruption: the roundtrip is provably
   bitwise-clean at every 1-node topology tested, and the gate's delta is
   an order of magnitude below the measured nondeterminism at the real
   topology. → Recommend: make the gate compare **state fingerprints**
   (per-tensor, like bt_state_probe) instead of forward logprobs, or run
   the probe forward with deterministic MoE dispatch if/where supported.
2. This CONFIRMS the "different bugs" assessment vs Jerry's truncation
   report: nothing in these runs produced missing keys — Nemotron 1-node
   checkpoints carried adapters + optimizer.state.* + rng_state complete
   (2505 keys, 1.25GB, P1==P2). Jerry's DSv4/Qwen expert-LoRA truncation
   (H1) remains a separate, still-unreproduced report.
3. New independent lead: the 4-node load deadlock (finding 2) — possibly
   devbox-only, but the dispatch race shape deserves a look.

### Handoffs

- 2026-08-07 ~15:45 PT: code-level nondeterminism investigation handed to
  session **ramanujan** — determine where run-to-run forward variance enters
  at EP32/CP4 (MoE dispatch / grouped-GEMM / CP fused-attn / determinism
  flags), using glm/lps1003_loss_spikes as method prior art. Replies come
  back to session feynman.
- Still open for a future GPU session: complete a 4-node load (bypass the
  op-dispatch deadlock, e.g. boot-time load_checkpoint_dir) and diff
  save_pre vs load_post fingerprints — the decisive loading-bug test.

### Housekeeping / open ends

- fermi tore down qr4m7r3 (no longer needed). banach still on qzlr0o3.
- Backup of shared scripts made: user_artifacts/devbox_up.bak-fermi-qr4m7r3-20260807/.
- Box teardown when done: `truss train stop --remote baseten --job-id 3m9469q`.
- Evidence lands on box at $RD/evidence/ + $RD/statedump/ — copy to
  team_artifacts AND this folder before teardown.

---

## 2026-08-07 — session 2 (ramanujan): code-level root-cause of the EP32/CP4 forward nondeterminism

Handoff from feynman: determine WHERE run-to-run forward nondeterminism enters
at the nightly topology (TP8/CP4/EP32, 4×8 B300) and whether it is expected
accumulation-order noise or a real bug. Method: static audit of the exact pins
(trainers 0e0b65a6 / bridge 0f26fc36 / M-LM d3932e75, TE 2.16.0) via three
parallel code-audit passes (MoE dispatch/combine; controller config + probe
path; CP4 attention + Mamba CP). No GPU used.

### Config facts established (all file:line-verified)

- **The prod trainer runs with zero determinism guards.** No
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO` (TE default `1` = nondeterministic
  allowed, checked at M-LM `extensions/transformer_engine.py:1697-1703`), no
  `CUBLAS_WORKSPACE_CONFIG`, no `NCCL_ALGO`, `torch.use_deterministic_algorithms`
  off, `deterministic_mode=False`. launch.sh/Dockerfile set none of these
  (only alloc conf / NVTE_CUDA_ARCHS). The fork SHIPS the exact right recipe —
  `megatron/training/determinism.py:27-31` pins `NCCL_ALGO=Ring`,
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` —
  but it is reachable ONLY via Megatron argparse `--deterministic-mode`
  (`training/arguments.py:1512`); the trainers embedded launch builds provider
  configs directly and never runs it.
- Nemotron-3-Ultra nightly: `alltoall` dispatcher (golden config sets no
  `moe_dispatcher`; flex/DeepEP NOT used), `moe_grouped_gemm=True` →
  **TEGroupedMLP → cublasLt grouped GEMM** (legacy cutlass GroupedMLP doesn't
  exist in the fork), `moe_permute_fusion=True` → TE fused permute/unpermute
  (deterministic gather-based — the notorious `scatter_add_` unfused path
  `moe_utils.py:528` is dead here), router fp32 GEMM via `te_general_gemm`
  (`moe_utils.py:1269`), topk over 512 experts, aux loss 0, expert bias frozen
  3 ways for LoRA, **no fp8** (no amax state), no CUDA graphs, DP=1,
  attention `"auto"` silently promoted to **cuDNN fused** at CP>1 for hybrids
  (megatron_controller.py:415), CP attn comm `p2p`, Mamba CP = head-parallel
  a2a (`mamba_context_parallel.py`, no distributed scan, no state exchange).
- **`moe_shared_expert_overlap=True` is inherited from the NemotronH bridge**
  (nemotron_h_bridge.py:300) and the trainer only zeroes it on the flex branch
  (megatron_controller.py:350-351) — so on the alltoall path the shared-expert
  MLP runs on a **side CUDA stream** concurrently with the EP32 a2a and with
  the main-stream expert GEMMs (`shared_experts.py:192,226,260,321,362`,
  driven from `token_dispatcher.py:686,702-703,857-858`).
- RNG/dropout ruled out empirically: any RNG consumption would have broken
  bitwise equality at 1-node too (it didn't). `/forward` runs train-mode
  module under `torch.no_grad` (megatron_controller.py:2761); dropouts are 0.0.
- 922ebcbdb "clear dispatcher forward state" audited: memory-retention fix
  only; every cleared attr is unconditionally re-assigned next forward at
  EP32/16-local-experts. Not causal.
- Exonerated as deterministic: EP a2a + size-exchange allgather (movement,
  no reduction), D2H split-size side stream (correctly fenced with
  wait_stream + record_stream + event sync, token_dispatcher.py:929-959),
  TP allreduces in TE parallel CE (stable within a process), CP logprob
  stitch (pure allgather), TE CP ring order (fixed `cp_global_ranks`),
  Mamba CP a2a reorder (coverage asserted by packing.py:1036; also CP2 was
  bitwise clean in phase 3).

### The two mechanisms that fit "bitwise clean at 1-node, dirty at 4-node"

**H-A (expected-class): cublasLt split-K/atomics at EP32/CP4-specific GEMM
shapes.** With no CUBLAS_WORKSPACE_CONFIG and NVTE nondeterminism allowed,
cublasLt may pick split-K with `REDUCTION_SCHEME_INPLACE` (atomic, run-to-run
nondeterministic) — a shape-dependent choice. The nightly topology changes
exactly the shapes: grouped GEMM per-expert M is tiny at EP32 (16 local
experts vs 64 at EP8; different num_gemms/m_splits), and the fp32 router
gating GEMM's M shrinks 4× under CP4. **Amplifier**: router
`topk(k=22 of 512)` (`moe_utils.py:806-810`) — a 1-ULP wobble at the 22/23
boundary swaps an expert → O(1) logit change. This explains the observed
magnitude: fwd0-vs-fwd0b deltas (max 1.62, mean 0.034) are ~the size of the
training step's own effect (max 2.47, mean 0.042) — i.e. routing flips, not
linear jitter passthrough.

**H-B (real-bug-class): shared-expert overlap stream races.** Two concrete
hazards, both gated on the inherited `moe_shared_expert_overlap=True`:
  1. TE's **singleton cublasLt workspace** (`get_workspace()`, used via
     M-LM `extensions/transformer_engine.py:3302-3342`) is shared between
     side-stream shared-expert Linears and main-stream TEGroupedMLP GEMMs
     running concurrently — unsynchronized if any algo uses workspace.
  2. Side-stream TP all-gather into the unfenced global `"mpu"`
     `GlobalMemoryBuffer` (`shared_experts.py:237-243` vs
     `tensor_parallel/layers.py:526`; buffer has no stream tracking,
     `core/utils.py:693-719`) — likely cold with TE linears, but unfenced.
  Topology gating is the a2a duration: intra-node EP8 a2a ≈ μs (race never
  loses → 1-node bitwise clean); cross-node EP32 a2a ≈ ms (side-stream GEMMs
  genuinely overlap main-stream work). Latent extra: shared-expert output
  returned without `record_stream` (`shared_experts.py:354-363`) — accidentally
  covered on the alltoall path, uncovered on flex; report, don't chase.

### Verdict

The nightly gate delta is **not evidence of checkpoint corruption** (phases
1-3 already showed the roundtrip bitwise clean); the forward nondeterminism
is real and enters via (at least) H-A, with H-B a credible co-suspect that is
an actual bug (unintended flag inheritance + missing workspace fencing).
"Expected vs bug": H-A is expected-given-config — the stack is *configured*
nondeterministic and the fork's own determinism module documents the exact
guards that are not applied in prod. H-B, if it fires, is a correctness bug
affecting training numerics, not just probes.

### Discriminating experiments (cheap, in order — need a 4-node window)

1. Env kill test: `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
   CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring` (must be set before first
   TE/cuBLAS/NCCL use) → fwd0 vs fwd0b. Bitwise 0 ⇒ H-A confirmed.
2. `moe_shared_expert_overlap=False` (one line in `_configure_moe_provider`,
   mirroring the flex branch) → bitwise 0 ⇒ H-B confirmed (real bug; also the
   flag should arguably be off or explicit regardless).
3. Localize: dump per-layer `routing_map`/top_indices for both forwards.
   Routing differs ⇒ source upstream of dispatcher (router GEMM / attention);
   routing identical but outputs differ ⇒ TEGroupedMLP GEMMs or workspace race.

### Gate recommendation (unchanged, now with mechanism)

At this topology a logprob-comparison gate at tol 1e-4 measures GEMM algo
choice + routing chaos, not checkpoint integrity. Either (a) compare state
fingerprints (bt_state_probe style), or (b) run the gate's probe forwards
with the determinism env from `megatron/training/determinism.py` applied at
trainer boot (nightly-only env is fine).

### Finding 2 (4-node load deadlock) — not covered by this audit

py-spy showed rank0 inside dist_checkpointing `all_gather_object`
(validation.py:497) while a peer sat in `_peer_loop broadcast_object_list`
(worker.py:241) waiting for the NEXT op — i.e. the peer never entered the
load op: an op-dispatch ordering race in the devbox driver path, not
necessarily prod. Separate lead, untouched here.

---

## 2026-08-07 — session 3 (ramanujan, cont.): GPU discriminators for the EP32/CP4 nondeterminism

Jack's directive: 4×B300 devbox, (a) CONTROL — rerun the forward ~10× on the
identical datum at the full nightly topology to confirm the nondeterminism
still reproduces; (b) H-B DISCRIMINATOR — relaunch with
`moe_shared_expert_overlap=False`, 10 forwards; (c) if not fixed — env-trio
(H-A), then systematically shrink the repro (nodes / seq / iteration time)
and localize per-layer (LPS-1003 parity method as prior art).

Jack's prior: dismiss H-A ("algo nondeterminism can't be this large"),
favor H-B (race, rhymes with LPS-1003). On record: session-2's audit says
H-A CAN be this large here — router topk(22/512) amplifies 1-ULP GEMM wobble
into expert swaps → O(1) logit deltas, and the observed magnitude ≈ one
training step's effect is exactly the routing-flip signature. Running
overlap-off first per directive; env-trio stays queued.

### Setup

- No live jobs on ali (banach's qzlr0o3 gone) → devbox-up clobber-safe.
- Provisioning **4 nodes × 8 B300** via `devbox-up 4 b300 ali` → job
  **3yl01lq** (tj-3yl01lq). Same-cluster CPFS: trainers_main clone reused.
- New harness `~/nemotron-repro/4node/` (box: user_artifacts/nemotron-repro-4node/):
  - `repro_fwd_determinism.py` — fwd_pre → 1 train step → N=10 forwards on
    the identical datum0, NO save/load (sidesteps the phase-4 load deadlock).
    Reports vs-first deltas, full pairwise max, distinct-bitwise-outcome groups.
  - `apply_patch_overlap.py` — env-gated override in `_configure_moe_provider`
    (anchor verified at pin 0e0b65a6): `BT_MOE_SHARED_EXPERT_OVERLAP=0` →
    False; effective value printed to trainer log either way ([bt_probe] line)
    so every arm's config is verifiable from the srun log.
  - run-script insert: `BT_DETERMINISM_ENV=1` → exports
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, CUBLAS_WORKSPACE_CONFIG=:4096:8,
    NCCL_ALGO=Ring before server start (H-A arm, queued).
  - `purge_cudart12.sh` — phase-3 dual-cudart fix via srun fan-out, all nodes.
- Arms are separate trainer boots (overlap flag is construction-time).

### Box up + CORRECTION that kills H-B before it's tested

- Job **3yl01lq** RUNNING in ~6 min (node pool warm), full provision+verify
  in ~7 min. Harness staged, md5-verified from all 4 nodes (no zero-fill).
  cudart-12 purged on all 4 nodes (only libcudart_static.a remains — inert).
- Control trainer booted healthy in ~12 min (TP8/CP4/EP32, seq 262144).
- **The [bt_probe] line shows `moe_shared_expert_overlap = False (env=None)`
  in the STOCK control arm.** Root cause of the discrepancy vs session-2's
  audit: nemotron_h_bridge.py:300 does set True, but lines 304-305
  (`if hasattr(hf_config, "moe_shared_expert_overlap"): provider... =
  hf_config...`) honor the checkpoint's HF config — and this model's
  config.json (snapshot c1a0675d, the same one the nightly runs) carries
  `"moe_shared_expert_overlap": false`. Session-2 missed the override 4
  lines below the line it cited. **H-B (shared-expert side-stream race) is
  falsified by construction: the side stream never runs in the nightly
  config.** Jack's prior (race a la LPS-1003) loses its main candidate;
  H-A (cublasLt algo nondeterminism × topk(22/512) routing amplification)
  is now the primary hypothesis. Plan: control 10-fwd repro → env-trio arm.

### CONTROL RESULT — nondeterminism REPRODUCED, 10/10 distinct

Stock env, TP8/CP4/EP32, overlap natively False, fixed datum0 (697 logprobs),
one train step then 10 back-to-back forwards:
- **10 forwards → 10 DISTINCT bitwise outcomes** (no two identical).
- vs-fwd0 max |Δ| per forward: 2.24–3.50; full pairwise max **4.32** (pair
  fwd4/fwd5); means ~0.037; ~585/697 tokens differ every time.
- Training-step effect for scale: max 2.48, mean 0.045 → run-to-run forward
  noise is AS LARGE AS a whole optimizer step's effect. Routing-flip
  signature, consistent with phase 4 (max 1.62 over a single pair).
- Repro speed: each forward ~1s; a 10-fwd determinism check ≈ 30 s against a
  live trainer. The only slow part is trainer boot (~12 min). "Button-press"
  repro achieved: `repro_fwd_determinism.py --label X --n-fwd 10`.
- Evidence: `evidence-phase5-determinism-arms/fwd_determinism_control.json`.

Next: env-trio arm (BT_DETERMINISM_ENV=1 → NVTE_ALLOW_NONDETERMINISTIC_ALGO=0,
CUBLAS_WORKSPACE_CONFIG=:4096:8, NCCL_ALGO=Ring), same driver.

### ENV-TRIO RESULT — did NOT fix it. Both headline hypotheses now dead.

Trainer relaunched with the trio exported on all 4 nodes before server start
([run_script] echo ×4 confirms). Same driver, 10 forwards:
- **10/10 distinct bitwise outcomes**, pairwise max |Δ| **5.12**, means
  ~0.039 — statistically indistinguishable from control. Evidence:
  `fwd_determinism_envtrio.json`.
- H-B dead (overlap never on). H-A-as-env-testable dead (trio applied,
  no effect).
- **Caveat that keeps a cublasLt variant of H-A alive**:
  CUBLAS_WORKSPACE_CONFIG governs legacy cuBLAS handles, NOT cublasLt.
  TE grouped GEMM + te_general_gemm are cublasLt. A split-K
  atomic-reduction algo there is uncontrolled by any of the trio. So the
  GEMM-atomics mechanism is NOT excluded — only the "fork determinism.py
  recipe would have fixed prod" claim is.
- NVTE_ALLOW_NONDETERMINISTIC_ALGO mainly constrains fused-attn BACKWARD
  backend choice; forward attn was already deterministic-in-practice at
  1-node. No surprise it didn't move.

### Next: localize Y directly — per-layer parity probe (LPS-1003 method)

Plan: forward hooks registered from the (already-shadowed) controller at
model-build time, env-gated (BT_FWD_PROBE=1):
- every decoder layer output → float64 fingerprint (sum/abssum/max) per
  forward call, per rank;
- every MoE router output (top-k indices / routing_map) → sha256 + logit
  stats per layer per call.
Then 10 forwards → offline diff: FIRST divergent layer, and at that layer
whether ROUTING differs (wobble enters upstream: router GEMM / attention /
mamba) or routing is identical while values differ (expert GEMM / combine).
No megatron package shadowing needed — hooks attach from trainers_server.

### FWD-PROBE RESULT — single hot origin: FIRST ATTENTION LAYER, CP ranks 1&2, last SP shard

Probe arm (stock env + hooks on 111 layers + 49 routers per rank) still
nondeterministic (10/10 distinct, max 3.32) — hooks don't mask it.
Analysis (`fwdprobe_analysis_stock.json`, `fwdprobe_pairs_stock.json`):

- **All 45/45 call-pairs first diverge at the SAME module: `layers.7`, the
  first TransformerLayer (attention) in the hybrid stack — and only on
  ranks 15 and 23** (= TP rank 7, the LAST sequence-parallel shard, of CP
  ranks 1 and 2; not CP0/rank7, not CP3/rank31).
- At the origin the wobble is tiny (layer-output float64-sum spread
  ~1e-3..1e-2); routers at/before the origin are bitwise IDENTICAL. First
  router flip comes 1-3 layers later (logit sum spread ~1e-6-1e-7 at the
  22/512 boundary) and THEN deltas explode. Mamba layers (9,11,13, CP
  head-parallel a2a mixes full sequence) spread the wobble to all shards.
  → topk routing flips are the AMPLIFIER; the SOURCE is attention.
- Per-rank "first divergent layer" varies (7/9/12/13/15/18/21) only because
  each rank sees a different token shard — per-pair global analysis
  collapses it all to layers.7@{15,23}.
- Mechanism candidates, all inside the CP4 cuDNN fused-attn forward
  (`cp_comm_type=p2p`, backend auto→cuDNN for hybrids at CP>1):
  ring-step partial-output merge or split-KV accumulation for specific
  chunk positions (load-balanced THD chunking: CP1 holds chunks 1+6, CP2
  holds 2+5; the divergent SP shard = tail of the SECOND chunk each).
  Notably CP0 (chunks 0+7, incl. sequence tail+padding) does NOT diverge.

### Next two discriminators

1. Submodule probe inside layers.7 (+14 as control): hook
   core_attention / linear_qkv / linear_proj etc. → is the divergent tensor
   the core-attention output while qkv inputs are identical? (pins the
   kernel) + per-token row norms (which rows wobble).
2. `attention_backend: flash` arm (config-only change, LPS-1003 precedent
   trainer-config.flash.json): if flash fwd at CP4 is bitwise stable, cuDNN
   fused CP fwd is confirmed as sole source AND we have a workaround.

### SUBMODULE PROBE — KERNEL PINNED: cuDNN FusedAttention fwd, single row

Layer-7 submodule drill-down (hooks on all layers.7/.14 submodules with
per-token row norms; still 10/10 distinct overall, max 3.35):
- rank 15 exec order: `linear_qkv` (incl. LoRA adapter chain) output
  **bitwise SAME across all 10 calls** → attention INPUT identical.
- `core_attention.fused_attention [FusedAttention]` output [176,1024]:
  **DIFF, 10 distinct**. Everything after (linear_proj, adapters, layer
  output) inherits it. Divergence enters INSIDE the TE cuDNN fused-attn
  forward call.
- Row-level: across ALL 45 pairs, on BOTH rank 15 and rank 23, EXACTLY ONE
  row differs: **row 174 of 176** (= row 86 of the rank's second
  load-balanced CP chunk; different global tokens on CP1 vs CP2 — 614 vs
  526). Same structural site every call; only the value varies.
- Site profile: CP ranks 1,2 only (not 0,3), TP rank 7 only (q-heads 56-63,
  MQA kv-head 7), one Q row near the second chunk's tail. All TP ranks run
  the same kernel schedule on different head VALUES → structural
  schedule-anchored site + value-dependent tie/order sensitivity, or a
  split-KV/stat-merge race in the cuDNN THD kernel for the ring-step shapes
  seen by middle CP ranks. cuDNN-internal either way.
- Explains all topology gating: CP1/CP2-only ⇒ needs CP4 (phase-3 CP2 was
  clean); nothing to do with EP32, nodes, or NCCL. The "4-node" trigger was
  really the CP4 trigger (CP4×TP8 needs 32 GPUs for the 550B).
- Also explains why the env trio did nothing: NVTE_ALLOW_NONDETERMINISTIC_ALGO
  gates fused-attn BACKWARD backend choice; forward split-stat behavior is
  not covered.

Verdict shape: X (trigger) = cuDNN fused attention, THD + CP4 p2p, this
head/seq geometry, B300/cu13 (B200 nightly presumably same — delta 0.175
matches amplified wobble); Y (location) = TE FusedAttention forward;
mechanism = single-site run-to-run output variation at fixed input.

Next: (1) attention_backend=flash arm → workaround + backend confirmation;
(2) standalone TE 4-GPU CP4 repro script (no trainer, no 550B) for NVIDIA.

### FLASH ARM — no escape hatch: TE silently falls back to cuDNN

`attention_backend: flash` boots healthy and the controller log confirms
`AttnBackend.flash (set from trainer config)` — but the arm is STILL
nondeterministic (10/10 distinct, max 5.07). Probe re-run on the flash boot
shows why: the origin module that actually executed is STILL
`layers.7.self_attention.core_attention.fused_attention [FusedAttention]`
(45/45 pairs) — TE's runtime support filter rejects flash-attn for this
thd+CP-p2p+MQA geometry and silently falls back to cuDNN fused attention.
**There is no backend swap available for this geometry in this stack.**
(This flash boot's origin spanned all 16 TP ranks of CP1+CP2 — site profile
varies per boot; the kernel + CP1/CP2 signature is the invariant.)

### STANDALONE REPRO — button achieved. 4 GPUs, no model, ~2 min

`standalone_te_cp4_repro.py` (in evidence folder + $RD): pure TE
DotProductAttention, thd + padding_causal + CP p2p + MQA (8 q heads / 1 kv
head / d128, bf16), one 698-token seq padded to 704, random inputs, 20
forwards on identical inputs. torchrun --nproc-per-node=4 on ONE node:

- **REPRODUCES: CP ranks 0,1,2 give 3-7 distinct outputs in 20 iters; the
  wobble is ALWAYS exactly local row 174** (T_LOCAL-2, tail region of the
  rank's second load-balanced chunk). CP rank 3 always clean.
- Trainer-vs-standalone delta explained: standalone CP0's row 174 = global
  token 702 = PADDING (random-filled here, zeros in the trainer) — hence
  trainer showed CP1/CP2 only.
- **CP2 variant: wobbles too, just rarer** (1 hit in 3 seeds × 20 iters,
  row 349 = again second-chunk tail) — phase-3 "CP2 clean" was a small
  sample, not immunity. CP1 (no CP) not run (no CP path).
- **`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` does NOT fix it** (kernel-level
  confirmation; the knob governs bprop backend choice).
- Stack: TE 2.16.0, torch 2.11.0+cu130, cuDNN 9.19 (91900), B300 sm103.
- Evidence: `standalone_te_cp4.log`, script alongside.

### FINAL VERDICT (session 3)

- **Root cause of the LPS-1063 nightly gate failures: run-to-run
  nondeterminism in the cuDNN fused-attention FORWARD kernel under
  context parallelism (thd, p2p ring), hitting a single structural row in
  the tail of each rank's second load-balanced chunk.** MoE topk(22/512)
  routing flips amplify the 1-row wobble into ~0.03-0.05 mean /
  up-to-5-max logprob deltas within 2-3 layers. Checkpoint save/load was
  never at fault (phases 1-3 bitwise clean).
- Jack's H-B (shared-expert overlap race): falsified by construction —
  the model's config.json sets moe_shared_expert_overlap=false and the
  bridge honors it; the flag was never on. (Session-2 audit error:
  missed the hf_config override at nemotron_h_bridge.py:304-305.)
- H-A as originally framed (cublasLt GEMM atomics + env-trio fix):
  falsified — env trio has no effect; the source is the attention kernel.
  The topk-amplification half of H-A was correct.
- Trigger X minimized: 4 GPUs, 1 node, no weights, 2 minutes, ~always
  fires within 20 iters. Location Y minimized: one cuDNN kernel, one row.
  Mechanism: cuDNN-internal (tile/split scheduling race at the chunk-tail
  boundary); needs NVIDIA. Not fixable by any documented TE/torch knob we
  tested.

### Recommendations (for ticket + Jerry)

1. Nightly ckpt-roundtrip gate: STOP comparing logprobs at CP>1
   topologies; compare per-tensor state fingerprints (bt_state_probe
   pattern), or run the probe forward at CP1.
2. File TE/cuDNN bug with standalone_te_cp4_repro.py (TE 2.16.0 /
   cuDNN 9.19 / sm103; also expected on sm100 — the nightly's B200 showed
   the same amplified signature).
3. Prod awareness: ALL CP>1 Nemotron/hybrid training forwards carry this
   nondeterminism (training itself, not just probes). Magnitude at the
   loss level is small per-step but breaks any bitwise-reproducibility
   assumption (KL probes, importance-sampling ratios computed against
   stale logprobs, etc.).
4. LPS-1063 disposition: nightly failure root-caused (not a checkpoint
   bug); Jerry's DSv4/Qwen truncation report remains a separate,
   unreproduced issue.

### Housekeeping (session 3)

- Arms all on job 3yl01lq (4×8 B300). Evidence pulled to
  `evidence-phase5-determinism-arms/` (driver JSONs, probe analyses,
  standalone log + scripts). Full per-rank probe dumps persist on CPFS:
  user_artifacts/nemotron-repro-4node/fwdprobe-{4node-stock,layer7,flash}.
- Harness (incl. standalone repro + fwd-probe hooks + analyzers) also in
  ~/nemotron-repro/4node/ locally.
- Box tj-3yl01lq torn down at end of session (standalone repro needs only
  4 GPUs of any B300 node in future).

## 2026-08-08 — session 4 (feynman): mechanism to full evidence + fix (overnight)

Mission per HANDOFF-CUDNN-BUG.md: (A) validate repro on the nightly's own
stack (B200+cu12+trainers@4ec66ce3), (B) ring-step isolation → decide
kernel-internal vs TE-race vs correction-kernel, (C) version matrix
(delegated to gibbs), (D) fix + proof. Supervisor: ramanujan.

### Prep (offline, while B200 box provisions)

- B200 box attempt 1 (job wprr2p3, vul) DIED mid-provision (infra flake,
  RUNNING→FAILED). ramanujan re-provisioning; alias TBD.
- Granularity correction to session-3 verdict, agreed with ramanujan: the
  probe pinned the wobble to the FusedAttention MODULE output = the whole
  AttnFuncWithCPAndKVP2P orchestration — per-step cuDNN kernels PLUS TE's
  post-ring correction kernels (tex.thd_out_correction ×cp_size,
  tex.thd_second_half_lse_correction / lse-correction on two alternating
  CUDA streams). "cuDNN kernel bug" is NOT yet proven at kernel-call
  granularity. Phase B decides: input-side (p2p buffer race) vs
  kernel-output-side (cuDNN) vs correction-side (TE race).
- Built (in ~/nemotron-repro/4node/):
  - cp_probe_patch.py — env-gated monkeypatch of TE 2.16
    context_parallel.py: per-ring-step fingerprints of q/k/v AS CONSUMED,
    per-step out+lse, each lse-correction result, each thd_out_correction
    result, final out. Clones on the producing/consuming streams (no
    added cross-stream syncs), single hash sync per forward;
    BT_CP_PROBE_SAVE=1 also torch.saves full captures incl. replay args.
  - standalone_te_cp_repro_v2.py — v1 generalized: CP=world (incl. CP1),
    --n-iters/--seeds, probe integration.
  - replay_single_call.py — replays ONE captured ring-step
    cp_p2p_fwd_fused_attn call solo (1 GPU, no dist): wobbles solo ⇒
    minimal single-call NVIDIA repro; stable solo ⇒ needs concurrency ⇒
    TE-race territory.
  - setup_phaseA_nightly_venv.sh — worktree @4ec66ce3 + pinned-lock server
    venv build on the box (cu12 lane, CUDNN_PATH/pybind11/ninja gotchas).
- ramanujan's web-research digest (full text to land in
  evidence-phase6-mechanism/upstream_research.md): no exact upstream match
  (TE≤2.17.1, cuDNN 9.19→9.25 notes, cudnn-frontend 1.27, NeMo/Megatron
  trackers) — likely novel. Closest prior art: TE PR #3186 (main-only;
  seqlens-offset cross-thread race, interleaved layouts only — ours is
  thd_thd_thd; rule out empirically via TE-main arm or
  NVTE_FUSED_ATTN_DIRECT_SEQLENS on cuDNN≥9.24) and TE issue #2186
  (THD+CP second-chunk-tail NaN in the SAME code region, backward,
  fixed in cuDNN 9.18 — cite as region bug history). cuDNN matrix
  priorities: 9.25, 9.24, one 9.21.x (Blackwell SDPA fwd fixes 9.21-9.23;
  TE blacklists 9.23.0/1).
- Work split: gibbs (Kimi K3 session) takes C (version matrix, GPUs 4-7);
  feynman owns A+B+D (GPUs 0-3).

### Offline source analysis (TE v2.16 checkout) — a sharp mechanism candidate

Structural decode of the wobble site (local row T_LOCAL-2 = 174, CP4):
- seq 698, T_TOTAL 704: diagonal step cu_q = 698//4 = 174 real rows of 176
  → rows 174,175 are PADDING for diagonal/lower steps. Upper-triangle steps
  use half-q with cu_q = 698//8 = 87 real rows of 88 → half-row 86 = LAST
  REAL row, maps to local row 88+86 = 174. **The wobble row is exactly the
  row that is padding-in-diagonal ∩ last-real-in-upper.**
- Rank wobble pattern decoded: upper-triangle steps exist only for
  rank < cp_size-1 (steps i > rank). Standalone: ranks 0,1,2 wobble, rank 3
  NEVER — perfect correlation with "has ≥1 upper-triangle step".
- attention.cpp (fused_attn_fwd): te_O IS zeroed for THD+BF16
  (te_O.zero_()), but the softmax-stats aux tensor (LSE, [t,h,1] packed,
  padded rows included) is allocateSpace(..., init_to_zeros=FALSE) =
  at::empty = UNINITIALIZED; cuDNN workspace likewise at::empty.
- Candidate mechanism (fits intermittency + row-174-only + rank pattern +
  row-175-clean): cuDNN writes LSE only for real rows → padding entries =
  run-varying allocator garbage → thd_second_half_lse_correction merges
  diagonal-step garbage LSE[174] with upper-step REAL lse (log-sum-exp
  merge: garbage magnitude decides the result) → thd_out_correction scales
  the upper step's real out row 86 by exp(lse_step − merged_garbage) →
  nondeterministic out[174]. Row 175 stays clean because out_per_step
  padding rows are ZEROED (0 × garbage = 0). Rank 3 clean: no second-half
  merge at all.
- Wrinkle to check in data: trainer CP0 was clean (zero-padded q) — naive
  version of this mechanism predicts CP0 wobbles too. Elementwise probe
  dumps (captures mode) will show whether step-LSE padding entries really
  vary and how they propagate.
- If confirmed: **pure-Python TE-side fix is possible** (sanitize LSE
  padding entries to -inf after each step in context_parallel.py before
  the merge — PYTHONPATH-shadowable in the trainer). If instead per-step
  out REAL rows wobble at identical inputs → kernel-internal (workspace
  garbage / split-tile race) → cuDNN territory.
- Probe/analyzer upgraded accordingly: BT_CP_PROBE_SAVE captures + offline
  ELEMENTWISE cross-iter diffs (real-region vs padding-region attribution).

### Root cause sharpened (still offline; pre-GPU confirmation)

Reading the tex correction kernels (common/fused_attn/context_parallel.cu)
plus dot_product_attention.py:1519-1529 completes the picture:

1. thd_out_correction_kernel / thd_lse_kernel iterate PADDED token ranges
   (cu_seqlens_q_padded passed from python) and carry an explicit
   zero-guard `p_per_step==0 ? 0 : ...` — the design CONTRACT is: garbage
   LSE at padding rows is tolerable because out_per_step is zeroed there.
2. The contract breaks because cu_seqlens_q_per_step = cu_seqlens_q //
   cp_size (pad_between_seqs=False fast path) MISLABELS rows whenever
   real_len % (2*cp) != 0: uniform 174-of-176 per rank, while the actual
   padding (6 rows) lives only in rank0's second chunk. Consequences:
   a) NONDETERMINISM: row 174 = mislabeled-padding for diagonal/lower
      steps (LSE garbage, out zero) ∩ last-REAL for upper-triangle steps
      (LSE real, out NONZERO) → merge mixes garbage into the scale factor
      → exp(real − merge(garbage, real)) varies run to run.
   b) CORRECTNESS (silent, deterministic): boundary rows mis-attended;
      row 175 is a REAL token on ranks 1..cp-1 whose output is EXACT ZERO
      (never computed by any step: diag/lower see it as padding, upper's
      cu=87 skips half-row 87).
3. WHY the fast path engages: auto-detect (dpa.py:1521) compares
   cu_padded[:-1] vs cu[:-1] — the [:-1] deliberately ignores tail padding
   after the LAST sequence → single tail-padded seq ⇒ pad_between_seqs=
   False. Without CP that's harmless (cuDNN masks by real cu directly);
   under CP the tail padding wraps into the middle of rank0's local
   buffer and the //cp approximation poisons every rank's boundary.
4. TE ALREADY HAS the exact path: pad_between_seqs=True →
   get_cu_seqlens_on_cp_rank (true per-rank per-step counts; rank0 cu
   [0,170], ranks1-3 [0,176]) → LSE fully written where out is nonzero,
   zero-guard holds, AND correct attention for all real rows.

⇒ Fix candidate A (PRIMARY): engage pad_between_seqs=True for thd+CP with
  tail padding (public kwarg; upstream fix = make the auto-detect count
  tail padding when context_parallel). Fixes BOTH defects.
⇒ Fix candidate B (fallback): sanitize per-step LSE padding rows to -1e30
  pre-merge (apply_te_lse_fix.py) — kills nondeterminism only.

Proof plan (GPU, per ramanujan's standard):
- Correlation: probe elementwise — step-LSE padding rows vary at bitwise-
  identical inputs; everything upstream stable.
- Causality (poison arm, BT_CP_LSE_POISON in cp_probe_patch): stock →
  wobble; +1000 → DETERMINISTIC WRONG (row174 ≈ 0); -1e30 →
  DETERMINISTIC CORRECT. 3-arm table = mechanism proven.
- Corroborator: PYTORCH_NO_CUDA_MEMORY_CACHING=1 (fresh cudaMalloc pages)
  → expect quiet or near-quiet.
- Fix A arm: --pad-between-seqs (new flag in repro v2) ≥100 iters × CP{2,4}
  → 0 distinct AND CP4-reassembled == CP1 reference on real rows
  (compare_cp_outputs.py; also demonstrates correctness bug (b) in stock).
- Genus note for writeup: LPS-1003 rhyme — uninitialized aux buffer
  consumed by attention path. Jack's "race/uninit, not algo" instinct was
  directionally right at the mechanism level (wrong candidate H-B, right
  genus).

### Upstream status (offline check, laptop clone of NVIDIA/TransformerEngine)

- Releases v2.16.0, v2.17, v2.17.1: auto-detect unchanged (the buggy
  `torch.equal(cu_padded[:-1], cu[:-1])` tail-padding-blind comparison).
  → ALL released TE versions carry the defect.
- MAIN: auto-detect rewritten (identity check: same-object → False, else
  padded-supplied → True; touched around PR #3201) — our call pattern
  (distinct real/padded cu tensors) now lands on the exact path, so main
  incidentally BYPASSES the bug. The commit comment motivates the change
  as sync-free + CUDA-graph-stable — NOT as a fix for nondeterminism or
  the zero-row bug → upstream likely unaware; filing remains valuable
  (also: kernel-level hardening — the //cp fast path is still reachable
  via same-object padded cu, and the zero-guard contract remains fragile).
- Falsifiable prediction for gibbs's version matrix: TE-release ×
  ANY cuDNN (9.19..9.25) = FIRES; TE-main × any cuDNN = quiet.
  cuDNN version should be IRRELEVANT (mechanism is TE-side).
- Additional blast radius to verify on-box: softmax_lse (with merged
  garbage) is save_for_backward'd → boundary-row GRADIENTS in real
  training should inherit the poison (nondeterministic grads + wrong
  zero-row grads), beyond the forward-probe symptom the gate caught.

### GPU session (tj-w6xx15w, 1-node B300 ali, job w6xx15w)

Box notes: nvidia-smi name-masked "NVIDIA L20D cc8.9" — torch runtime says
cc=(10,3), 267 GiB/GPU = B300-class sm103 (authoritative). Stack = session-3
anchor exactly: torch 2.11.0+cu130 / TE 2.16.0 / cuDNN 91900. Gotchas hit:
dual-cudart (fixed by phase-3 purge, no srun on 1-node → ran inner block
directly). GPUs 0-3 mine; 4-7 gibbs (version matrix, launched 10:00Z).

**B0 (stock ×30 iters ×3 seeds): fires — ranks 0,1,2 wobble, always exactly
row 174; rank 3 clean. 3-7+ distinct/seed.** Same with probe on (hooks
don't mask). rc=1 by design.

**B0b probe ELEMENTWISE (within-seed, the correlation arm) — MECHANISM
CONFIRMED, cuDNN EXONERATED:**
- Every step's q/k/v/cu INPUTS and every step's OUT: bitwise stable
  (absent from cross-iter diffs entirely).
- Per-step LSE aux varies at EXACTLY the mislabeled-padding coordinates:
  step0 (diagonal, [176,8,1]) rows {174,175} = 16 elems; step2
  (upper-triangle, [88,8,1]) row {87} = 8 elems (ranks 0,1). Which steps
  show varying garbage is allocator-dependent (steps 1,3 buffers happened
  to be iteration-stable) — hash-level distinct counts 39/3/19/3 match.
- Propagation: lse2corr*.lse_post + corr*.lse_total_pre vary at columns
  174/175 (packed [8,176]) → corr{upper-steps}.out_post vary at exactly
  row [174] (~700-870 elems = that row's channels) → final.out row [174].
  Rank0 shows corr1,2,3 (upper steps 1-3); rank1 shows corr2,3 (upper
  steps 2,3) — 1:1 with the section schedule.
- Rank 3: SAME lse garbage (step lse + total lse vary at 174/175) but NO
  out/final variation — zero-guard holds (no upper-triangle steps). The
  "clean rank" was never clean in the LSE domain.
- Evidence: probe/cp_probe.rank*.jsonl + 360 capture .pt files on box;
  to be pulled into evidence-phase6-mechanism/.

B1 battery in flight: poison trio (BT_CP_LSE_POISON=+1000 / -1e30),
--pad-between-seqs fix arm (×100), CP1 ground truth + reassembly
comparator (correctness: predict stock CP4 row-175 real-token output ==
exact 0 on ranks 1-3, and row-174 == upper-contribution-only), no-cache
corroborator, backward pair (gradient poisoning, stock vs fix).

### ⚠ CORRECTION superseding session-3 FINAL VERDICT (ramanujan concurs)

Session 3 concluded "cuDNN fused-attention FORWARD kernel bug" — that
attribution was to the FusedAttention MODULE boundary and is WRONG at
kernel granularity. Session-4 element-level fingerprints show every value
the cuDNN kernel WRITES is bitwise deterministic; the nondeterminism is
introduced by TRANSFORMER ENGINE's CP orchestration (uninitialized
at::empty LSE aux rows at positions mislabeled by the pad_between_seqs=
False cu//cp fast path, merged by TE's own correction kernels). Any
upstream filing goes to NVIDIA/TransformerEngine, NOT cuDNN. The
session-3 standalone repro and topology analysis remain valid; only the
final attribution is superseded.

### B1 battery — CAUSALITY PROVEN, FIX PROVEN (fwd+bwd), CORRECTNESS BUG BROADER THAN PREDICTED

All arms CP4 ×3 seeds on tj-w6xx15w (logs/b1_*.log, dumps/ on box):
- POISON +1000 (BT_CP_LSE_POISON): DETERMINISTIC 30/30 iters, all ranks,
  all seeds — and out[global 614/526] = EXACT 0 (predicted: scale factor
  exp(real−1000)→0). Deterministic-WRONG as forecast.
- POISON -1e30: DETERMINISTIC — but row 174 still |d|=0.248 vs CP1:
  sanitize CANNOT restore the diagonal-step contribution the kernel never
  computed. (Refines the fix-B claim with proof: determinism-only.)
- Stock wobble + both poisons pinning the output ⇒ uninitialized LSE
  content is THE cause. (PYTORCH_NO_CUDA_MEMORY_CACHING still fires —
  fresh pages are also undefined content; neutral corroborator.)
- FIX A --pad-between-seqs: bitwise DETERMINISTIC 100/100 iters ×3 seeds.
- CP1: deterministic (CP required for the bug — phase-A question answered
  at mechanism level).
- BACKWARD stock (NVTE_ALLOW_NONDETERMINISTIC_ALGO=0): GRADIENT
  CORRUPTION, 4/4 RANKS incl. "clean" rank 3 — grads nondeterministic
  across MANY rows (list truncation ≥10 rows), not just 174: garbage LSE
  feeds the bwd kernels and the dKV ring SPREADS it cross-rank.
  Training math contamination is broad, the fwd 1-row symptom understates.
- BACKWARD --pad-between-seqs: outs AND grads bitwise deterministic
  20/20 ×3 seeds, all ranks.

CORRECTNESS (compare_cp_outputs vs CP1 ground truth, seed 16):
- STOCK: 607/698 real rows differ >1e-2, mean|d|=0.040, max 0.362. Root:
  mislabeled cu drops each rank's last-2 chunk KV rows as padding KEYS →
  6 REAL KEYS (globals 438,439,526,527,614,615) excluded from EVERYONE's
  attention. Plus query rows: 615/527/438/439 EXACT-ZERO output (real
  tokens!), 614/526 nondeterministic-and-wrong.
- FIX A: max|d|=0.0039, mean 0.0015, 0 rows >1e-2, boundary rows match
  ref norms — CP4 == CP1 at bf16-rounding level. ACCEPTANCE PASSED.
- Nightly-gate blindness explained: the deterministic mis-attention is
  identical on both sides of save-vs-reload compares and CANCELS; only
  the nondeterminism surfaced (0.175). The mis-attention ships silently
  in every CP>1 THD training forward with real%(2cp)≠0.

B200/cu12 end-to-end checkbox: OPEN, LOW-PRIORITY (morning follow-up; no
B200 capacity tonight — vul 2× fail, hyd qzll783 FAILED. Mechanism is
proven architecture-independent at element level).

In flight (b2): CP2 stock/fix ×100, perf timing stock-vs-fix ×200.

### B2 — CP2 validation + first perf reading

- CP2 stock ×100 iters: FIRES (rank0 seed17, row 349 = the CP2 boundary
  row; rare, matches session-3 sampling). CP2 padfix: deterministic
  300/300. Fix generalizes across CP sizes.
- Perf (crude, whole-process `time`, 200 iters ×1 seed): stock 33.4s vs
  padfix 58.8s → ~127ms/call extra — TOO BIG for the cu math; either boot
  noise or a real per-call host cost (plan-cache misses?). Proper
  in-process measurement (warmup-excluded, 704 + 32768 tokens) in flight
  (b3) before drawing rollout conclusions.
- Evidence pulled to laptop evidence-phase6-mechanism/ (probe jsonls,
  logs, dumps, captures-sample; full 360-capture set stays on CPFS).
- 4-node B300 provisioning in flight for the trainer bitwise arm
  (ramanujan green-light; fix-arm only, no save/load, patched TE via
  package COPY not in-place venv edit).

### B3 — clean perf (in-process, warmup-excluded, CP4)

- T=704:   stock 1.784 ms/iter → padfix 2.073 ms/iter (+0.29 ms)
- T=32768: stock 2.023 ms/iter → padfix 3.499 ms/iter (+1.48 ms)
- The earlier "127 ms/call" from whole-process `time` was boot noise.
  Real cost: ~0.3-1.5 ms/call HOST-side (both arms host-bound at these
  sizes — 32k tokens fwd = 2 ms on B300). At trainer scale (262k seq,
  tens-of-ms attention kernels, multi-second steps) this is noise;
  trainer arm will confirm per-fwd timing directly.

### 4-node logistics + filing draft

- 4-node attempt 1 (job 3m99r6q): rank-2 node never became SSH-reachable
  (job RUNNING, ssh proxy rejected through grace + 6 retries) → devbox-up
  FATAL at topology step. Job STOPPED promptly (was holding 32 B300).
  Attempt 2 in flight.
- Upstream filing drafted: evidence-phase6-mechanism/TE_ISSUE_DRAFT.md
  (mechanism, element-level evidence chain, poison causality, fix
  validation, main-accidental-fix note + 3 asks incl. regression test,
  workaround; [pending] markers for gibbs's remaining matrix cells).

### Trainer arm (plan B: 1-node Qwen3-0.6B TP2×CP4 on tj-w6xx15w)

- Wiring complete: patched_te shadow (TE package copy + padfix, sanity
  requires torch imported first — venv nvidia libs preload via torch, not
  TE RPATH; nvidia/torch sibling symlinks added), PYTHONPATH order
  patched_te:patched_src in run_trainer_node_det.sh, cpgate relaxation
  (BT_CP_ALLOW_GENERIC=1 allows generic dense provider at CP>1 with
  attention_backend=fused; controller otherwise hard-rejects by provider
  TYPE — ramanujan's flag confirmed) applied to patched_src.
- gibbs matrix: 12/12 wheel cells done — TE 2.16.0 AND 2.17.1 fire 3/3
  seeds on cuDNN 9.19/9.21.1/9.24/9.25 (row-174 signature throughout);
  DIRECT_SEQLENS arms no-ops (var absent in releases). TE-main
  discriminator cell pending (CPU-only build in flight, overlapping my
  trainer boots).
- Control boot attempt 1: died on missing BT_TRAINER_SERVER_CONFIG_PATH
  (now exported). Relaunch interrupted by ssh relay throttling (2×rc=255,
  the 4am relay again); recovery monitor armed — on reconnect: inspect
  for partial pkill/double-launch, clean-slate relaunch.
- 1-node launch bypass documented: no srun on 1-node boxes → run
  run_trainer_node_det.sh directly with SLURM_NODEID=0 exported; health
  via generated wait_trainer_det.sh (ad-hoc polls are hook-blocked).

### Trainer arm continued — relay-proof orchestration

- run_startup_warmup crash root-caused: CP>1 full-warmup (LPS-1003 window
  burner) builds docs down to remaining≥2048 → max_seq_len=1024 yields
  ZERO docs → pydantic too_short. Resolution: max_seq_len 1024→4096
  (ramanujan: keeps prod boot semantics, no BT_SKIP_FULL_WARMUP hatch in
  the evidence chain).
- ssh relay degraded to ~1 admitted connection/window from laptop
  (multi-minute rc=255 streaks; same genus as the 4-node worker
  registration failures — infra flag for the morning). Countermeasure:
  on-box orchestrator (orchestrate_qwen_arms.sh) runs the FULL sequence
  unattended — control boot (BT_TE_CP_TAILPAD_FIX=0) → wait_trainer_det →
  driver 10-fwd → padfix boot (=1) → wait → driver → teardown; single
  push via first admitted connection. STARTED 12:4xZ (pid 37114).
- Verified working in earlier boot logs: cpgate marker
  ("[bt_probe] BT_CP_ALLOW_GENERIC=1: allowing generic GPTModelProvider
  at CP>1 (attention_backend=fused)") — patched_src + patched_te shadow
  both engaged in the trainer process.

### TRAINER ARM VERDICT — FIX PROVEN IN THE FULL TRAINER STACK (Jack's bar)

Orchestrator run 12:05-12:13Z on tj-w6xx15w (Qwen3-0.6B TP2×CP4, 8 GPUs,
prod-like boot with 4096 warmup, fwd-probe hooks on, no save/load):
- CONTROL (BT_TE_CP_TAILPAD_FIX=0): **NONDETERMINISTIC — max|Δ|=0.0827
  over 10 back-to-back fwds on the identical datum.** Bug fires through
  trainers_server → megatron → TE on a generic DENSE model; magnitude ~40×
  smaller than the 550B (no MoE topk amplifier), as predicted.
- PADFIX (BT_TE_CP_TAILPAD_FIX=1): **BITWISE_DETERMINISTIC — 10/10
  identical logprob vectors.**
- Fix engagement proven from boot logs: '[bt_te_padfix] CP tail-padding
  fix ACTIVE: pad_between_seqs=True (exact per-rank cu path)' ×5 in the
  padfix boot, 0 occurrences in control.
- Evidence: evidence-phase6-mechanism/qwen_arm_evidence.tgz (driver
  verdict JSONs incl. full logprob vectors, boot+driver logs, fwdprobe
  dirs, orchestrator.status).

Proof obligations status: mechanism ✅ (element-level fingerprints),
causality ✅ (poison trio), fix standalone ✅ (100/100×3seeds×CP{2,4},
fwd+bwd), fix-in-trainer ✅ (this), correctness-vs-CP1 ✅ (max|d|=0.0039),
gradient blast radius ✅. Remaining: gibbs TE-main matrix cells; then
teardown. Morning: 550B/4-node formality, B200/cu12 checkbox, infra flags
(vul pool, ali multinode worker scheduling, ssh relay degradation).

- Bonus corroboration from the control arm's fwd-probe hooks: 9/45 fwd
  pairs diverge, origin histogram = 100% `layers.0 [TransformerLayer]` —
  the FIRST attention layer, the dense-model analogue of session-3's
  layers.7-first-attention signature. Same origin class, same
  intermittency, now demonstrated inside the full trainer on a dense
  model. (fwdprobe-qwen-control/ in the evidence tarball.)

## 2026-08-08 — session 4 addendum (gibbs): work item C version matrix COMPLETE

Platform note: B200/vul died 2x; matrix ran on tj-w6xx15w (1-node B300-class
sm103, ali), cu13 lane, torch 2.11.0+cu130, GPUs 4-7 (feynman had 0-3).
Harness: ~/nemotron-repro/4node/version_matrix/ (driver + per-cell loaded-cudnn
probe; scratch venvs, never the shared clone).

**Matrix result (18/18 cells, CP4, seeds 16/17/18 x 60 iters = 180 fwd/rank/cell):**

| TE \ cuDNN(cu13) | 9.19.0.56 | 9.21.1.3 | 9.24.0.43 | 9.25.0.15 |
|---|---|---|---|---|
| 2.16.0 | FIRES 3/3 | FIRES 3/3 | FIRES 3/3 | FIRES 3/3 |
| 2.17.1 | FIRES 3/3 | FIRES 3/3 | FIRES 3/3 | FIRES 3/3 |
| main (2.19.0.dev0+8260f49) | quiet 0/3 | quiet 0/3 | quiet 0/3 | quiet 0/3 |

- cuDNN is NOT the variable; the fix lives in TE main, bracketed (2.17.1, main].
  Corroborates feynman's TE-side pad_between_seqs root cause (session 4 above).
- Wobble row = 174 (T_LOCAL-2, 2nd-chunk tail) in 108/108 firing instances —
  session-3 signature invariant across cuDNN versions.
- Loaded libcudnn proven == requested wheel per cell (/proc/self/maps +
  cudnnGetVersion); no cell counted without that proof.
- NVTE_FUSED_ATTN_DIRECT_SEQLENS absent (py+binary) in 2.16.0, 2.17.1 AND main —
  those arm cells are inert duplicates; main's fix is unconditional, not env-gated.
- Build gotchas encoded in the driver for reuse: TE-torch sdist needs CUDNN_PATH
  (venv nvidia/cudnn); TE cu13 wheels need nvidia-cublas>=13.6 (torch cu130 pins
  13.1.0.3 -> undefined cublasLtGroupedMatrixLayout*); TE sanity check needs the
  transformer-engine metapackage; TE main build needs cuda-toolkit-13.0
  (/usr/local/cuda-13.0; /usr/local/cuda kept at 12.8, feynman's cudart12 exile
  intact), NVTE_WITH_NCCL_EP=0, NVTE_CUDA_ARCHS=100 (NOT 103 — arch-specific
  cutlass sources static-assert against generic compute_103; "100" expands to
  sm_100+100a+103a), ~23 min compile at -j32.
- Deliverables: evidence-phase6-mechanism/version_matrix.{json,md} +
  version_matrix_logs/ (18 cell logs). Working TE-main venv left on box at
  /root/.cache/user_artifacts/lps1063-matrix-gibbs/venvs/te_main.

### SESSION 4 CLOSING VERDICT (feynman; gibbs ran work item C; ramanujan supervised)

**LPS-1063 is root-caused, mechanism-proven, and FIXED — proof at every
link, no speculation:**

1. ROOT CAUSE: TransformerEngine ≤2.17.1, thd+CP(p2p) with a tail-padded
   sequence (real % (2·cp) ≠ 0 — the common case): the pad_between_seqs
   auto-detect deliberately ignores tail padding (cu[:-1] compare) → the
   cu//cp fast path mislabels each rank's chunk-boundary rows →
   (a) NONDETERMINISM: uninitialized (at::empty) per-step LSE rows merged
   into a real row's scale factor by TE's own correction kernels;
   (b) SILENT MIS-ATTENTION: 2(cp-1) real keys dropped from everyone's
   attention + boundary query rows zeroed/partial (607/698 rows off vs
   CP1 on the standalone geometry);
   (c) BACKWARD: gradients broadly nondeterministic on all CP ranks.
   cuDNN fully EXONERATED (every kernel-written value bitwise stable;
   18/18 version-matrix cells show cuDNN-independence; row 174 in
   108/108 firing instances).
2. CAUSALITY: poison trio — stock wobbles; LSE padding rows pinned to
   +1000 → deterministic-wrong (row 174 exactly 0, as predicted);
   -1e30 → deterministic. Uninit memory is THE cause.
3. FIX: pad_between_seqs=True (TE's existing exact path). Proven:
   standalone bitwise fwd+bwd 100/100 ×3 seeds ×CP{2,4}; CP4==CP1 at
   bf16 rounding (max|d|=0.0039); full trainer stack (Qwen3-0.6B TP2×CP4)
   control NONDETERMINISTIC (max|Δ|=0.0827) vs padfix
   BITWISE_DETERMINISTIC 10/10 with the '[bt_te_padfix] ACTIVE'
   engagement marker. Perf cost ~0.3-1.5 ms/call host-side (noise at
   trainer scale; trainer arms booted+drove indistinguishably).
   Prod patcher: apply_te_padfix.py (env BT_TE_CP_TAILPAD_FIX, default
   on). Upstream: TE main already (accidentally) fixed via detect
   rewrite; filing drafted (TE_ISSUE_DRAFT.md) asking for intentional
   semantics + regression test + LSE-init hardening.
4. NIGHTLY GATE: 0.175 was the nondeterminism (amplified by MoE topk on
   the 550B); the mis-attention cancels in save-vs-reload compares and
   ships silently in every CP>1 THD training forward. Gate recommendation
   unchanged (fingerprint-compare) PLUS adopt the TE fix in prod.

### Morning checkboxes (consolidated)

- [ ] 550B/4-node (TP8/CP4/EP32) trainer arm with the fix — FORMALITY
      (mechanism is model/arch-independent; blocked tonight by 0 fittable
      B300 nodes on ali; harness + patched_te ready on CPFS).
- [ ] B200/cu12 nightly-stack standalone checkbox (vul pool broken 2×,
      hyd job qzll783 FAILED after 60min DEPLOYING).
- [ ] INFRA FLAGS for Jack: (1) vul B200 provisioning broken (2×
      RUNNING-but-ssh-never-up); (2) ali multinode worker pods pending
      forever behind landed leaders when capacity short — devbox-up
      should preflight whole-node fit (0 fittable B300 tonight);
      (3) ssh relay degradation ~12:15-12:45Z (rc=255 streaks, ~1
      admitted connection/window from laptop; asymmetric per client).
- [ ] File TE_ISSUE_DRAFT.md upstream (NVIDIA/TransformerEngine).
- [ ] Prod rollout decision: apply_te_padfix.py to trainers (env-gated),
      + Jerry/gate follow-ups per session-3 recommendations.
- [ ] Teardown-time cleanup gotcha for the runbook: pkill patterns must
      include multiprocessing spawn_main children (orphans held ~618MiB
      CUDA context each after dp_worker pkill).

### Housekeeping (session 4 final)

- Box tj-w6xx15w torn down at session end (see below). CPFS artifacts
  persist: lps1063-phase6/ (probe captures full set),
  nemotron-repro-4node/ (harness incl. patched_te + qwen evidence),
  lps1063-matrix-gibbs/ (matrix venvs incl. working TE-main build).
- Laptop evidence: evidence-phase6-mechanism/ (probe jsonls, elementwise
  captures sample, dumps, b0-b3 logs, qwen_arm_evidence.tgz,
  version_matrix.{json,md} + logs, upstream_research.md, TE_ISSUE_DRAFT).

### Prod fix PR (session 3/ramanujan, post-wrap)

- basetenlabs/trainers#993 — `fix(server): force TE onto the exact
  cu_seqlens path for tail-padded THD under CP (LPS-1063)`.
  Runtime wrapper around DotProductAttention.forward applied at the top of
  run_rank (every rank, before first forward); forces pad_between_seqs=True
  only when thd + CP>1 + full-array cu divergence; kill switch
  BT_TE_CP_TAILPAD_FIX=0; deletable once a TE release ships main's rewritten
  auto-detect. 12 unit tests (decision logic, scoping, positional binding,
  kill switch, idempotence). make check green; branch
  jackrao/lps-1063-te-cp-tailpad-fix off main @8af05cf5.

### Gate run on the PR branch — image-pin gotcha + a valuable fresh control

- Dispatched nightly-correctness (nemotron-3-ultra, steps=2) on the PR
  branch: run 31272697069. Result: checkpoint_roundtrip FAILED at
  max |Δ| = 0.178 — but this did NOT exercise the fix: the gate client
  provisions a prod Loops trainer whose image comes from the PROD REGISTRY
  (published tags like baseten/trainers-server:*), not from the dispatched
  branch. Only examples/ ran from the branch.
- Silver lining, and it matters: this is a FRESH, same-day confirmation
  that the STOCK PROD stack (B200/cu12 lane) still fails at exactly the
  known magnitude (0.178 vs 0.175/0.199) — effectively closing the
  "B200/cu12 stock side" checkbox via prod itself. The fix side on prod
  requires the image: no SDK/gate image override exists
  (create_lora_training_client → control plane → registry).
- Branch trainer image build dispatched anyway (run 31274459239, tags
  <ref>-<short-sha> + -cu13) — usable for a registry publish or devbox
  image runs if wanted.
- Paths to gate-level fix validation: (a) merge PR → release-build →
  reconcile → tonight's nightly runs the fixed image automatically
  (recommended); (b) pre-merge publish of the branch image to the prod
  registry (prod-touching, Jack's call); (c) full-fidelity 4×B300 devbox
  gate-replica (the existing morning checkbox; needs capacity).

### Fix re-shaped as the call-site chain (Jack's call: fork version)

- #993 (runtime wrapper) CLOSED as superseded. Replacement chain, merge in
  order:
  1. basetenlabs/Megatron-LM#25 (branch jackrao/lps-1063-thd-cp-pad-between-seqs
     off trainers-main): TEDotProductAttention computes the tail-inclusive
     pad_between_seqs and passes it explicitly when thd + cp>1 + cu differs
     from cu_padded. 25 lines, one call site, both TE invocation branches.
  2. basetenlabs/Megatron-Bridge#31: 3rdparty/Megatron-LM gitlink bump.
  3. basetenlabs/trainers#994: server/vendor/megatron-bridge gitlink bump.
     Pure-Python via the editable vendored megatron-core path dep — no wheel
     rebuild, no lock change.
- Gitlinks reference PR-branch shas; retarget if upstream squash/rebase-merges.
- Post-merge validation: nightly Nemotron gate on prod picks up the release
  image automatically (today's stock-prod control: 0.178).
