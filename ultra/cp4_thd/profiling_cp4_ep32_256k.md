# Nemotron-3-Ultra 550B — packed-THD CP4/EP32 on 4×8 B200

Branch: `ultra-thd-cp4` (trainers). Devbox: `tj-w64072q` (4 nodes × 8 B200,
Birch/Weka). Base weights: `nemotron3-ultra-550b-nvfp4-dequant-bf16`
(local BF16 dequant of the NVFP4 checkpoint). LoRA rank 16, alpha 32.

Target topology: **TP8 / PP1 / CP4 / EP32 / ETP1 / DP1** (32 ranks).
Production baseline: **TP8 / PP4 / CP1 / EP4 / ETP2 / DP1**.

All numbers below are fresh measurements from this work. Nothing is inherited
from the historical Ultra CP branch.

## Methodology

- Trainer boots via `devbox_up` lifecycle scripts on the shared
  `trainers_main` checkout of the PR branch; driven over the worker HTTP API
  by `scripts/sft_driver.py` (deterministic seeded payloads; identical bytes
  across boots for parity runs).
- lr=0 optimizer steps so parity runs never mutate weights; grad_norm is
  still computed and reported by the optimizer.
- Memory: `POST /memory_stats` (new all-rank CUDA-allocator op added in this
  PR; max/reserved peaks reset per profiling point) + all-32-GPU NVML
  `memory.used` sampled at 2 s by `scripts/nvml_poll.sh` (raw CSVs retained
  on shared devbox storage under `ultra_cp4/out/`).
- Profiling payloads: one document of exactly L tokens per point,
  L ∈ {8192, 32768, 65536, 131072, 196608, 262144}; two measured
  forward/backward + optim cycles per point (three at 256K), optimizer state
  materialized by the lr=0 step between them.

## Small-model hard gate (tiny 70M NemotronH hybrid)

Tiny hybrid (mamba + attention + moe blocks, real weight-loading through
NemotronHBridge) on one node, identical payloads:

| metric | TP4/CP1/EP2 (DP2) | TP4/CP2/EP2 | agreement |
|---|---|---|---|
| CE loss | 11.941921 | 11.941895 | 2.2e-6 rel |
| CE grad_norm (step 0) | 1.290149 | 1.290077 | 5.6e-5 rel |
| per-datum logprob sums | — | — | ≤ 2e-5 rel |
| logprob prefixes (8 vals/datum) | — | — | identical |
| IS RL mismatch_kl | 9.936015 | 9.935977 | 3.8e-6 rel |

The THD CP data path (zigzag shard, DP×CP reduction, logprob stitching, RL
fields) is numerically equivalent to the CP1 path at tiny scale, for both CE
and the token-level RL losses.

Caveat found and fixed while building the probe: transformers-native
NemotronH `save_pretrained` emits `backbone.embedding.weight`, while the
Ultra checkpoints (and NemotronHBridge's mapping) use
`backbone.embeddings.weight`. The bridge silently leaves unmapped weights
zero-initialized — the model forwards on a zero residual stream (CE =
ln(padded vocab) exactly) and every LoRA gradient is zero. The probe builder
renames the key; real Ultra/Super checkpoints already use the mapped name.

## 550B parity: production PP4 vs target CP4

Identical 5-document payload (4096/2560/6144/1024/3072 tokens, seed 4242),
fresh boot each side, lr=0, two repeated steps per boot (repeated steps are
bit-identical on both sides — the runs are deterministic within a boot).

| metric | TP8/PP4/CP1/EP4/ETP2 | TP8/PP1/CP4/EP32/ETP1 | delta |
|---|---|---|---|
| CE loss | 17.860410 | 17.856461 | 2.2e-4 rel |
| active loss tokens | 15266 | 15266 | exact |
| per-datum logprob sums | — | — | 7e-5 – 1.2e-3 rel |
| per-datum logprob shapes | 4096/2560/6144/1024/3072 | same | exact |
| grad_norm (step 0) | 30.027 | 24.379 | ratio 0.81 |
| warm forward/backward (~17k tok) | 4.5 s | 2.6 s | CP4 1.7× faster |

Reading:

- Forward/loss/logprob parity holds at the level expected for a 550B
  bf16 model across two different kernel/parallelism stacks (the pair changes
  attention backend flash→fused, EP4/ETP2→EP32/ETP1, PP4→CP4
  simultaneously; the tiny-scale CP-only comparison bounds the CP data-path
  contribution at ~1e-5).
- grad_norm is finite and same-order but differs ~19%. Attribution:
  - tiny-scale CP-only change (TP4/CP1 → TP4/CP2): 5.6e-5 rel — not the CP
    data path;
  - tiny-scale ETP-only change (ETP1 → ETP2, same payload): 2.2e-5 rel —
    not the expert-tensor re-shard;
  - same-topology, same-boot invocation noise at 550B (identical payload,
    fresh /forward_backward): ~1% (e.g. 186.2 vs 184.5 at 131k; 131.8 vs
    131.0 at 256k) — nondeterministic kernel reductions;
  - remaining co-varying factors are the attention backend (flash → TE
    fused, required for hybrid THD CP) and EP4 → EP32 reduction orders.
    LoRA-B gradients at adapter init are near-cancelling sums, so per-layer
    bf16 kernel differences (forward logprob deltas already reach ~1e-3)
    amplify in the gradient norm. The loss/logprob surfaces bound the
    forward divergence at ≤1.2e-3 per datum.
- This validates the shared SFT forward/shard/loss path. It does not by
  itself claim an Ultra RL end-to-end pass (the RL losses share the THD
  plumbing and are covered at tiny scale; a 550B CP4 IS-loss smoke returned
  finite loss/grad_norm/mismatch_kl with correctly shaped per-datum
  logprobs).

## Profiling: CP4/EP32, ascending sequence lengths

One L-token document per point; ascending order on one boot so each point's
allocator peaks are attributable (peaks reset between points via
`/memory_stats reset_peaks`). Warm = second (or later) step. LoRA rank 16,
lr=0 optimizer steps with Adam state materialized. Full rows in
`out/profile_cp4_ep32.csv`; raw NVML CSVs retained on the devbox under
`user_artifacts/ultra_cp4/out/`.

| seq len | warm step | TPS/GPU | max-rank max_allocated | max-rank reserved peak | status |
|---|---|---|---|---|---|
| 8192 | 3.1 s | 82 | 44.6 GiB | 48.2 GiB | pass |
| 32768 | 3.3 s | 311 | 52.5 GiB | 53.7 GiB | pass |
| 65536 | 5.7 s | 360 | 63.3 GiB | 66.1 GiB | pass |
| 131072 | 11.1 s | 368 | 84.6 GiB | 89.8 GiB | pass |
| 196608 | 16.9 s | 364 | 106.0 GiB | 113.9 GiB | pass |
| 262144 | 22.9 s | 358 | 127.4 GiB | 135.1 GiB | pass (3 full steps) |

- Loss is bit-identical across repeated steps at every point; grad norms
  finite throughout; no OOM at any point.
- 256K steady state (3 consecutive fb+optim cycles): 22.8/22.8/22.9 s,
  ~358 TPS/GPU, loss 16.950252 each step.
- All-32-GPU NVML during the 131K+256K rerun window (device view, includes
  retained allocator pool + NCCL buffers): hottest 145,417 MiB,
  median 113,345 MiB, min 98,631 MiB of 183,359 MiB — ~21% headroom on the
  hottest GPU at 256K.
- Activation memory scales ≈ linearly above 32K at ~0.35 GiB per 1K tokens
  (max-rank allocator view).

## Dependency decisions

- No bumps to Megatron-Bridge, Megatron-LM/MCore, Transformer Engine,
  mamba-ssm, causal-conv1d, or vLLM. The pinned stack already contains hybrid
  packed-THD CP support (`MambaContextParallel` + `PackedSeqParams.seq_idx`),
  and the sampler stack is untouched by trainer CP topology.
- Nemotron NVFP4 sampler patches unchanged.

## Limitations

- grad_norm parity is bounded at ~19% across the full topology change;
  per-component attribution beyond the tiny-scale CP isolation is best-effort
  (see ETP isolation).
- DPO and image inputs remain rejected under CP (unchanged policy).
- MTP layers are not exercised by the trainer loss path (no labels passed to
  the model; loss computed via the chunked LM head).
