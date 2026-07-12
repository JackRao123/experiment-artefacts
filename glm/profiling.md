# GLM-5.2 memory profiling — toward 131k trainer context

**TL;DR (overnight run 2026-07-08): 131 072-token LoRA SFT fwd-bwd RUNS on
TP1/PP1/EP32/CP32 at 98.2 GiB peak-alloc / 138.2 GiB hottest-GPU used (~40 GiB
headroom), ~56 s per fwd-bwd — via the new GLM-DSA contiguous-CP adapter
(0.03% loss parity vs CP1). The current golden config (TP1/PP16/EP2) walls just
past 65k; CP4-with-DP8 is WORSE than golden (EP dispatch scales with global
batch tokens). Two blockers noted for training (not memory): NaN grads on the
first synthetic optim step (all sweeps ran fb-only; Paras is fixing this), and
CP is CE-only v1.**

Profiling of `zai-org/GLM-5.2-FP8` LoRA SFT trainer memory vs sequence length on
devbox `qr9oevq` (**4×8 B200**, 179 GiB usable/GPU, slurm + IB). GLM-5.2 is
`GlmMoeDsaForCausalLM`: **78 layers** (3 dense + 75 MoE), **256 routed experts /
top-8** + 1 shared, MLA + DSA (dynamic sparse attention, top-2048 frozen lightning
indexer, IndexShare freq=4/offset=3), ~800B params (~60B active), FP8 block-wise
dequantized to bf16 on load. Attention-only LoRA r16 (experts + indexer frozen),
fused frozen-indexer path (`apply_dsa_kernel_fusion`, FlashMLA fwd + cuDNN bwd).

Trainer only (no sampler), SFT cross-entropy. Full activation recompute
(committed default: `recompute_granularity=full`, uniform, 1 layer), micro-batch 1,
**one S-token synthetic sequence per DP rank** per forward_backward
(`--num-datums = --microbatch-size = DP`), 2 steps per point, `--reset-peak` per
point. Stack: `jackrao/glm-131k-profiling` @ origin/main `2e19a1bc` + re-applied
memory hooks (revert of `65c53447`) extended to report **all-32-rank**
peak_alloc/max_reserved via an `all_gather_object` in `get_status`.

## Measurement channels

- **`peak_alloc` (all ranks)** — `torch.cuda.max_memory_allocated` per rank from
  `/status` (`rank{r}` keys), reset per point. The fit series.
- **`max_reserved` (all ranks)** — allocator pool high-water per rank from `/status`.
- **hottest-GPU `memory.used`** — 1 Hz `nvidia-smi` on all 4 nodes via
  `srun --overlap` from the leader (U1-rerun pattern), max-over-point per GPU,
  sliced by per-point epochs. The OOM-binding number (includes NCCL/cuBLAS/TE/
  FlashMLA workspaces + fragmentation the allocator never sees); also the only
  channel that captures pre-crash peaks on OOM rows.
- **fwd-bwd time** — warm (2nd) step wall time from `sft_driver.py` (fb + optim;
  step 1 includes autotune/compile). PP16 with one microbatch/rank is mostly
  pipeline bubble — config 1 times are not throughput numbers.

Driven with `sft_driver.py --source synthetic --seq-len S --steps 2` against the
long-lived HTTP trainer (torchrun fan-out from the laptop; scripts in
`experiment_artefacts/glm/scripts/`, configs in `glm/trainer-configs/`).
Grid: 8192, 16384, 32768, 65536, 98304, 131072 (ascending; fit-as-we-go).
Note `efb25fca` (PP forward-length trim) is on main, so unlike the Ultra sweeps a
single boot at `max_seq_len=131072` serves all points — verified at the first
point (8k peak must sit far below the 64k-validated level).

## Configs (world = 32)

| # | config | DP | datums/fb | notes |
|---|--------|----|-----------|-------|
| 1 | golden TP1/PP16/EP2/ETP1/CP1 | 2 | 2 | committed golden (validated S64K); `(78,16)` DSA pipeline layout |
| 2 | TP1/PP1/EP32/CP4 | 8 | 8 | CP stack (see below) |
| 3 | TP1/PP1/EP32/CP32 | 1 | 1 | CP stack |

CP stack (`jackrao/glm-131k-cp`): main + `jerry/dsv4-cp-next` (THD contiguous CP
data path, CP-aware loss/grad normalization, NCCL fixes) + megatron-bridge
`6827d86c` + ported glm5_bridge FP8-dequant/IndexShare delta + Megatron-LM
`trainers-main-next-dsv4` (contains upstream PR #5087) + **new GLM-DSA CP adapter**
(`jackrao/glm-dsa-cp`, commit `7b27c96b2`): contiguous-CP forward for
`DSAttentionFused` (autograd CP all-gather of latent KV, frozen-indexer K gather +
`q_causal_offsets` top-k via `csa_cp_utils`, THD flat-index lowering), contiguous
branch in the shared THD RoPE, `max_seqlen` plumb in `absorbed_mla`, and config
validation widened to `dsa`.

## Experiment G1 — golden config seqlen sweep (TP1/PP16/EP2)

Boot debugging trail (for the record; each cost one relaunch):
1. ssh fan-out hang left ranks 1–3 unlaunched (missing `< /dev/null` on nohup).
2. Missing `pybind11` in the fresh venv broke Megatron's `helpers_cpp` build.
3. First cross-node collective timed out until the `configure_remote.sh`
   NCCL/IB env block (NCCL_IB_HCA, GLOO_SOCKET_IFNAME, …) was added.
4. `python3.12-dev` was missing on WORKER nodes → torch-inductor JIT warmup
   died (`Python.h`); the surviving ranks busy-spin in `all_gather` looking
   exactly like a stalled load. Headers must be apt-installed on every node.
5. `base_model: "zai-org/GLM-5.2-FP8"` (HF id) hits the transformers
   `qk_rope_head_dim` collapse bug (192 vs 64 → kv_a_proj shape mismatch
   704≠576): the glm5_bridge workaround only reads the raw config.json when
   base_model is a LOCAL path. **GLM-5.2-FP8 must be launched with base_model
   = the local snapshot path** (matches `_is_glm52_local_path` support).
6. The FP8 blockwise dequant-on-load was a per-block Python loop: ~60 CPU-min
   per rank for the 800B checkpoint. Vectorized (bit-exact, verified) in
   `quantization_utils.dequantize_fp8_blockwise` → load ~15 min end-to-end.

**Finding (needs follow-up, out of scope tonight): NaN gradients on the first
synthetic optim step.** fwd loss is finite (13.04 @ 8k), but `optimizer.step()`
reports `grad_norm=NaN` and the applied update poisons the weights (subsequent
losses NaN; `/status` then 500s trying to JSON-render `last_loss=NaN`). The
Jun-2026 smoke validated finite loss/grad-norm with REAL tokens on PP1/EP16
(2×8); tonight's repro is synthetic ramp + PP16 on main@`2e19a1bc` — the delta
(data, PP16 layout, or a main regression) is unresolved. **Update 2026-07-08:
Paras owns this issue and is working on the fix.** **All sweep points
below therefore ran `--learning-rate 0`** (identical compute & memory, zero
update, weights stay clean); the per-step loss consistency replaces the
loss-decrease validity check.

One boot at `max_seq_len=131072` (the `efb25fca` trim engages: 8k activations sit
far below the 64k level ✓), 2 fb-only steps per point, `--reset-peak` per point,
one 8192‑…‑131072-token sequence per DP rank (DP=2). `peak_alloc`/`max_reserved`
are the max over all 32 ranks from `/status`; `hottest used` is the max-over-run
per GPU from the 1 Hz pollers, sliced per point.

| seq_len | peak_alloc max-rank (GiB) | rank0 (stage 0) | max_reserved max (GiB) | hottest GPU used (GiB) | used min/med | warm fb (s) | warm TPS/GPU | loss step1→2 | status |
|---|---|---|---|---|---|---|---|---|---|
| 8 192   | 84.14  | 39.03 | 84.89  | 89.87  | 44.5 / 50.2   | 14.2 | 36.1 | 13.037 → 13.033 | ok |
| 16 384  | 90.90  | 45.72 | 92.33  | 97.31  | 51.3 / 57.7   | 17.9 | 57.2 | 13.470 → 13.478 | ok |
| 32 768  | 108.48 | 58.94 | 112.28 | 117.25 | 65.9 / 72.6   | 27.5 | 74.5 | 13.350 → 13.351 | ok |
| 65 536  | 143.71 | 97.94 | 160.69 | 165.67 | 117.1 / 120.2 | 50.0 | 81.9 | 13.107 → 13.093 | ok |
| 98 304  | —      | —     | —      | **177.72** | 163.1 / 166.9 | — | — | — | **fb never completed (>30 min): allocator thrash at the ceiling** |
| 131 072 | —      | —     | —      | —      | — | — | — | — | not reachable (queued behind the wedged 98k op) |

TPS/GPU = global batch tokens per fwd-bwd (DP · S = 2·S here) ÷ warm-step wall
time ÷ 32 GPUs — same convention as the Ultra benchmark harness's "Main FB
TPS/GPU". fb-only steps (skip-optim; optim adds ~0.02 s at LoRA scale). Note
PP16 with one microbatch per DP rank is mostly pipeline bubble, so these are
profiling-shape numbers, not tuned-throughput numbers (grad-accum microbatches
would fill the pipeline and raise TPS substantially).

- **Fit (hottest-rank `peak_alloc`, 4 clean points): y = 1.076 GiB/1k-tok · (S/1024) + 74.5 GiB, R² = 0.9990.**
- **Fit (hottest-GPU `used`): y = 1.374 GiB/1k-tok · (S/1024) + 76.3 GiB, R² = 0.9946.**
- Projected walls on 179 GiB HBM: ~97k (alloc channel), **~76k (used channel — the
  binding one)**. Empirically: 65 536 fits (165.7 GiB hottest used, ~13 GiB
  headroom); at 98 304 every GPU climbs past 163 GiB, the hottest hits 177.7,
  and the forward-backward stops making progress (expandable-segments thrash)
  rather than crisply OOMing. **The wall is just past 65k — consistent with S64K
  being the validated golden seqlen. 131k does NOT fit TP1/PP16/EP2.**
- **The (78,16) pipeline layout is the binding imbalance**: the three 8-layer
  tail stages (ranks 26–31, node 3 gpu 2–7) run ~142–144 GiB alloc / ~165 GiB
  used at 65k while the twelve 4-layer middle stages sit at ~101/​~120. A more
  balanced layout (e.g. 6-layer tail stages) would buy roughly 20 GiB on the
  binding stages — worth trying, but alone it does not reach 131k on the alloc
  slope (needs ≈ 131k·1.08 + 55 ≈ 193 GiB).
- fb time (2 steps, one 131k... seq per DP rank): 14.2 s @8k → 50.0 s @65k —
  near-linear in S, no indexer-cost bend visible at this grid (top-2048 already
  saturated at ≥8k).

## CP smoke — debug GLM (8 layers, vocab 2048), 1 node, CP1 vs CP2

`smoke_cp.sh` on the S2 stack, seq 4096, one datum, fb-only, synthetic vocab
1800 (< debug vocab). Two adapter bugs found and fixed en route: the re-layered
GLM fused commit still used the pre-#5087 `build_flat_topk_idxs(seqlen_kv=...)`
signature (CP1 path had never run on the DSv4 branch), and the CP indexer's
rope table was sized by `max_seqlen_q` (max REAL datum length) while positions
come from the PADDED cu_seqlens layout (OOB gather → device assert; now sized
by the global padded row count + clamped).

| variant | loss step1/step2 | peak_alloc max (GiB) |
|---|---|---|
| CP1 (tp1/pp1/ep2) | 7.7833 / 7.7833 | 2.33 |
| CP2 (tp1/pp1/ep2, adapter) | 7.7856 / 7.7849 | 1.70 |

**Loss parity 0.03%** (bar ≲0.5%; jerry's DSv4 reference 0.15–0.32%), per-rank
peak shards 2.33 → 1.70 GiB (weights dominate the tiny model; the activation
share halves). Note: CP2 shows ~9e-4 step-to-step jitter with frozen weights
(some nondeterministic kernel in the CP path — the gathers or FlashMLA), CP1 is
bit-stable; harmless for profiling.

## Experiment G2 — TP1/PP1/EP32/CP4 (DP=8)

One boot, `max_seq_len=131072`, 8 datums per fb (one S-token sequence per DP
replica; each CP rank holds a contiguous S/4 slice). **Static weights: a
perfectly flat 78.0 GiB on all 32 ranks post-boot** — PP1/TP1/EP32 holds the
800B model comfortably (the ~120 GiB pre-estimate was pessimistic), and unlike
PP16 there is no stage imbalance.

| seq_len | peak_alloc max-rank (GiB) | max_reserved max | hottest GPU used | used min/med | warm fb (s) | warm TPS/GPU | loss step1→2 | status |
|---|---|---|---|---|---|---|---|---|
| 8 192  | 87.31  | 89.05  | 93.15  | 89.4 / 90.5   | 45.8   | 44.7 | 13.149 → 13.151 | ok |
| 16 384 | 96.41  | 101.79 | 106.51 | 97.9 / 100.1  | 77.5   | 52.9 | 13.581 → 13.577 | ok |
| 32 768 | 119.08 | 136.93 | 141.03 | 120.8 / 126.6 | 153.8  | 53.3 | 13.935 → 13.923 | ok |
| 65 536 | 155.51 | 174.24 | **178.35** | 176.6 / 178.1 | **1346** | **12.2** | 13.656 → 13.659 | ok, but at the wall (46 min/2 steps: allocator thrash) |
| 98 304+ | — | — | — | — | — | — | — | skipped: projected ≥190 GiB, doomed (sweep cut to save the night for CP32) |

TPS/GPU here uses DP·S = 8·S global tokens per fwd-bwd. The 8k–32k plateau
(~45–53) beats G3's small-seq numbers because DP=8 keeps all replicas busy;
the 65k collapse to 12.2 is the memory-ceiling thrash, not compute.

- **Fit (max-rank `peak_alloc`): y = 1.225 GiB/1k-tok · (S/1024) + 77.8 GiB, R² = 0.998.**
- **Fit (hottest used): y = 1.525 GiB/1k-tok · (S/1024) + 84.0 GiB, R² = 0.979.**
- **Verdict: CP4 with DP=8 does NOT reach 131k — its slope is WORSE than the
  golden config's (1.23 vs 1.08 MiB/tok alloc).** The per-rank attention
  activations do shard 4×, but each forward-backward pushes 8 concurrent
  sequences (DP=8) through the EP32 experts, so the dropless dispatch/expert
  buffers scale with 8·S tokens per fb and dominate. Wall ≈ 64k (used channel) /
  84k (alloc). The memory-relevant knob is tokens concurrently in flight per
  forward-backward through EP, not CP alone (and not tokens per optimizer
  step — grad accumulation over sequential fb microbatches is memory-flat). (A DP&lt;8 variant — cp4 with fewer concurrent sequences, or
  microbatching the DP replicas — would flatten this; untested tonight.)
- Loss finite and step-stable at every point (fb-only), matching CP1-vs-CP2
  smoke parity: the adapter's numbers are trustworthy.

## Experiment G3 — TP1/PP1/EP32/CP32 (DP=1)

One boot, `max_seq_len=131072`, ONE datum per fb (the single DP replica; each
CP rank holds a contiguous S/32 slice — 4 096 tokens/rank at 131k).

| seq_len | peak_alloc max-rank (GiB) | max_reserved max | hottest GPU used | used min/med | warm fb (s) | warm TPS/GPU | loss step1→2 | status |
|---|---|---|---|---|---|---|---|---|
| 8 192   | 79.21 | 79.85  | 86.25  | 85.0 / 85.7   | 15.2 | 16.8 | 12.895 → 12.891 | ok |
| 16 384  | 80.45 | 81.32  | 87.66  | 86.2 / 86.9   | 19.5 | 26.3 | 13.604 → 13.623 | ok |
| 32 768  | 83.01 | 86.06  | 92.46  | 89.4 / 90.3   | 25.0 | 41.0 | 13.764 → 13.754 | ok |
| 65 536  | 88.69 | 97.61  | 103.32 | 98.4 / 100.5  | 33.8 | 60.6 | 13.289 → 13.305 | ok |
| 98 304  | 93.29 | 112.42 | 118.77 | 112.2 / 114.7 | 43.9 | 70.0 | 13.422 → 13.396 | ok |
| **131 072** | **98.16** | **131.79** | **138.16** | 129.0 / 132.3 | **56.2** | **72.9** | 13.522 → 13.539 | **ok — 131k FITS with ~40 GiB headroom** |

TPS/GPU here uses DP·S = 1·S global tokens per fwd-bwd: with DP=1 only one
sequence is in flight per step, so small-seq TPS is low (16.8 @8k — 32-way
sharding of a 8k sequence underfills every rank) and climbs with S as the
per-rank slice fattens (72.9 @131k, where each CP rank holds 4 096 tokens).

- **Fit (max-rank `peak_alloc`): y = 0.159 GiB/1k-tok · (S/1024) + 78.0 GiB, R² = 0.9989.**
- **Fit (hottest used): y = 0.426 GiB/1k-tok · (S/1024) + 80.0 GiB, R² = 0.979.**
- Projected walls: ~650k tokens (alloc) / **~230k (used channel — binding)**.
  131k measured directly: 138.2 GiB hottest used, all 32 GPUs within a 9 GiB
  band (no hot-rank pathology). fb time stays sane: 56 s at 131k (vs CP4's
  46-minute thrash at 65k).
- The slope decomposes as ~0.16 MiB/tok allocator + ~0.27 MiB/tok non-torch
  (workspaces/dispatch, the used−alloc gap) — with DP=1, global tokens per step
  = S, so the EP32 dispatch grows 8× slower than G2's.

## Cross-config comparison & verdict

| config | slope alloc (MiB/tok) | intercept | slope used | wall (used) | TPS/GPU @65k | 131k |
|---|---|---|---|---|---|---|
| golden TP1/PP16/EP2 (DP2) | 1.076 | 74.5 | 1.374 | ~76k (65k ok, 98k wedges) | 81.9 | ✗ |
| TP1/PP1/EP32/CP4 (DP8) | 1.225 | 77.8 | 1.525 | ~64k (65k at the wall) | 12.2 (thrash) | ✗ |
| **TP1/PP1/EP32/CP32 (DP1)** | **0.159** | **78.0** | **0.426** | **~230k** | 60.6 | **✓ 98.2 / 138.2 GiB, 72.9 TPS/GPU measured** |

1. **131k on 32×B200 is real today via CP32** (memory-wise). The lever is NOT
   just CP sharding — it is keeping **tokens concurrently in flight per
   forward-backward through the EP32 experts low (DP=1)**. CP4/DP8 pays 8·S
   tokens in dispatch per fb and is strictly worse than golden despite 4-way
   activation sharding. DP=1 caps concurrency, NOT batch size: effective batch
   per optimizer step comes from grad accumulation over sequential fb
   microbatches (memory-flat; ~56 s per 131k sequence, so wall-clock per optim
   step is ~N·56 s). Corollary: an intermediate CP8/DP4 or CP16/DP2 would land
   between G2 and G3 and trades per-step wall-clock against memory; needs its
   own sweep if that matters.
2. **Golden (PP16/EP2) walls just past 65k** — consistent with S64K being the
   validated seq — and its (78,16) tail stages are 40 GiB hotter than the middle
   (a layout rebalance buys headroom but not 131k).
3. **Before CP32 becomes a golden config, the training-side gaps must close:**
   the NaN-grad-on-optim finding (repros on golden PP16 with synthetic data;
   unresolved whether data-, layout- or main-regression-caused — **Paras is
   fixing this**), CP CE-only v1
   limits (no per-datum logprobs / DPO / MTP; weight-sync under CP untested),
   and a real-data loss-parity + convergence run vs CP1.

## Reproduction

- Stacks: trainers branches `jackrao/glm-131k-profiling` (S1 baseline) and
  `jackrao/glm-131k-cp` (S2 = S1 + `jerry/dsv4-cp-next` + submodule bumps);
  Megatron-LM fork branch `jackrao/glm-dsa-cp`; bridge branch
  `jackrao/glm-cp-bridge`. Since pushed as the PR stack tracked in
  `productionise.md` (trainers#592 → Megatron-LM#7 + Bridge#12); devbox clones under
  `/root/.cache/user_artifacts/trainers_glm{,_cp}` on `qr9oevq`.
- Harness: `experiment_artefacts/glm/scripts/` (`trainer_ctl.sh` fan-out boot,
  `sweep_local.sh` leader-side sweeps, `smoke_cp.sh`, `srun_pollers.sh` 1 Hz
  nvidia-smi, `analyze.py` tables+fits). Raw data: `experiment_artefacts/glm/data/`
  (driver logs, per-point `/status` JSONs, poller CSVs).
- Ops gotchas hit tonight (each cost a relaunch — see boot trail in G1):
  `< /dev/null` on nohup-over-ssh, `pkill -f "[b]racket"` self-match, NCCL/IB
  env block, python3.12-dev on EVERY node, local-snapshot base_model for
  GLM-5.2-FP8, vectorized FP8 dequant, no curl on the box (use venv httpx).

## CP adapter scope & caveats

- Memory-representative fwd-bwd + CE loss; jerry's 1/cp grad normalization is in
  (0.15–0.32% loss parity on DSv4 at cp2–16). CP smoke gate for GLM: CP1-vs-CP2
  loss parity ≲0.5% on the debug checkpoint + per-rank activation sharding.
- CE-only v1: no per-datum logprobs under CP, no DPO, no MTP; weight-sync/export
  under CP untested.
- IndexShare under CP: holders store (topk, layout) per computing layer per rank.
