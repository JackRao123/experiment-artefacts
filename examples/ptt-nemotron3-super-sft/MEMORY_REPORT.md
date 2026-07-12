# Nemotron-3-Super-120B-A12B — BF16 LoRA SFT activation-memory report

Goal: find a parallelism layout that fits **bf16 LoRA SFT at `max_seq_len = 131072`**
on **2 nodes / 16× B200** (178.35 GiB usable HBM per GPU) **without OOM**.

All numbers below are `torch.cuda.max_memory_allocated()` (peak *allocated*, not
reserved) read from the trainer's `/status` endpoint
(`gpu_max_memory_allocated`), captured by `sft_driver.py`. Peak is per worker
process since launch, so a fresh trainer per `max_seq_len` gives a clean
per-run peak.

Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` — hybrid
Mamba2 + Transformer MoE. LoRA rank 16, micro-batch 1.

> **CORRECTION (see `profiling.md`):** the numbers below were taken with
> `recompute_granularity="full"`, which we later proved is **INERT** for this
> model — NemotronH runs Megatron's `HybridStack`, which doesn't implement
> full/layer recompute (only `TransformerBlock` does), and `"full"` also
> disables the selective paths. So every peak in this report is effectively a
> **no-recompute** measurement. Switching to `recompute_granularity="selective"`
> with `recompute_modules=["core_attn","moe","moe_act"]` actually engages
> recompute for the attention+MoE layers and ~halves the per-token activation
> slope (2.57 → 1.17 MiB/token), moving the single-node cap from ~57k to ~126k.
> The PP=4 131k peak (100.79 GiB) should therefore drop substantially with
> selective recompute — re-profile PP=2/PP=4 + selective before finalizing the
> golden node count.

## Profiling runs (synthetic packed sequences, `--source synthetic`)

Layout: **TP=8, PP=1, CP=1, EP=8, ETP=1** (single node, 8× B200).

| seq_len | peak_alloc / GPU | activation Δ over baseline |
| ------- | ---------------- | -------------------------- |
| baseline (weights + framework, pre-forward) | ~30.57 GiB | — |
| 8 192   | 50.87 GiB        | ~20.3 GiB |
| 16 384  | 72.20 GiB        | ~41.6 GiB |

**Per-token activation slope** (the part that scales with sequence length):

```
(72.20 - 50.87) GiB / (16384 - 8192) tokens = 21.33 GiB / 8192 tokens
                                            ≈ 2.6 MiB / token / GPU
```

The fit is essentially linear (near-zero fixed activation intercept): even under
full recompute the dominant cost is the per-layer activation checkpoints that
must be retained across the sequence, and these are **not** tensor-sharded by
TP. The CP=2 sweep below shows they are not effectively sequence-sharded by CP
either — see "CP=2 sweep".

## Extrapolation to 131 072 tokens

At **TP=8, CP=1** (no sequence sharding):

```
30.57 GiB baseline + 131072 * 2.6 MiB ≈ 30.6 + 340 ≈ 371 GiB / GPU
```

That is **~2× the 183 GiB B200** → cannot fit at CP=1 by any micro-batch
reduction (micro-batch is already 1).

### Why TP can't absorb it

TP is **architecturally capped at 8** for this model: the Mamba2 mixer has
`ngroups = 8`, and Megatron asserts `ngroups % tp_size == 0`
(`AssertionError: ngroups must be evenly divisible by tp_size`). TP=16 fails at
construction. So the only way to get 16-way *sequence* sharding across the 16
GPUs is **CP=2** (TP=8 × CP=2 = 16). CP is therefore **required** here, not
optional — confirmed empirically, not just by the math above.

## CP=2 sweep (the important result)

Layout: **TP=8, PP=1, CP=2, EP=16, ETP=1** (2 nodes, world=16), cuDNN
FusedAttention (forced for CP>1, see "attention backend" below), full recompute,
micro-batch 1, synthetic packed sequences.

| seq_len | peak_alloc / GPU | activation Δ over baseline | result |
| ------- | ---------------- | -------------------------- | ------ |
| baseline (weights + framework, pre-forward) | ~17.12 GiB | — | — |
| 8 192   | 38.18 GiB        | ~21.06 GiB | ok |
| 16 384  | 60.08 GiB        | ~42.96 GiB | ok |
| 32 768  | 103.92 GiB       | ~86.80 GiB | ok |
| 65 536  | —                | — (extrapolates to ~188 GiB) | **OOM** (node-1 rank died) |
| 131 072 | —                | — (extrapolates to ~358 GiB) | **OOM** |

**Per-token activation slope at CP=2:**

```
(103.92 - 38.18) GiB / (32768 - 8192) tokens = 65.74 GiB / 24576 tokens
                                             ≈ 2.68 MiB / token / GPU
```

### The headline finding: CP is NOT sharding the per-token activation

The CP=2 slope (**2.68 MiB/token/GPU**) is essentially identical to the CP=1
slope (**2.6 MiB/token/GPU**). If CP=2 were sequence-sharding as intended, each
GPU would process only half the tokens and the per-GPU slope would be ~halved
(~1.3 MiB/token). It is not. So for this hybrid Mamba model, **CP=2 buys almost
no per-GPU activation reduction** — it just spreads the *same* per-token cost
across 2× the GPUs.

Decomposing the 8k point (CP=1 single node vs CP=2 two nodes) makes the cost
explicit:

| layout | per-GPU peak @8k | GPUs | **cluster total** | baseline total | activation total |
| ------ | ---------------- | ---- | ----------------- | -------------- | ---------------- |
| TP8, EP8, **CP1** (1 node)  | 50.87 GiB | 8  | **~407 GB** | ~245 GB | ~162 GB |
| TP8, EP16, **CP2** (2 nodes) | 38.18 GiB | 16 | **~611 GB** | ~274 GB | ~337 GB |

- Baseline (weights + CUDA ctx) total only rises **+29 GB**: CP=2 does replicate
  the non-expert weights across the 2 CP groups, but `EP 8→16` shards the expert
  weights (the bulk of a 120B MoE) twice as hard, nearly cancelling it. So the
  "second copy of weights" is real but a near-wash here.
- Activation total **doubles, +175 GB** (162 → 337). This is the dominant term
  and the real reason CP=2 costs more overall: the activations aren't sharded,
  so 16 GPUs each carry ~the full per-token cost.

### This predicts every OOM we saw

```
64k  @ CP2: 17.12 + 65536  * 2.68 MiB ≈ 188 GiB  > 178 usable → OOM   ✓ observed
131k @ CP2: 17.12 + 131072 * 2.68 MiB ≈ 358 GiB  >> 178       → OOM   ✓ observed
```

The earlier 131k notes (which guessed CP "halved" the CP=1 estimate to ~168 GiB)
were **wrong**: the ~168 GiB seen at the 131k OOM was just where allocation hit
the wall mid-build, not a steady state. The slope data shows no halving at all.

### Root cause: CP sequence-sharding is not wired into our data path

The ambiguity above ("CP not sharding" vs "CP sharding but 2× per-token cost")
is resolved by the code: **it is not sharding.** Our trainer builds its own
microbatches and bypasses Megatron's data loader, so the CP sequence-slice that
Megatron requires never runs:

- `packing.py` emits only **`bshd`** ((B, S) per datum) and contains no CP
  logic — `cu_seqlens` / `PackedSeqParams` appear only in a "future THD work"
  comment block.
- `_make_ce_forward_step` passes the full `(B, S)` `input_ids` straight into
  `model(...)` — no per-rank slicing, no packed-seq params.
- Megatron's CP contract requires the **data loader** to call
  `get_batch_on_this_cp_rank()` (`megatron/core/utils.py:1980`), which reshapes
  the sequence into `2*cp_size` chunks and `index_select`s this rank's two
  chunks down to `seq/(2*cp)`. Our controller calls `_forward_backward` directly
  on its own `bshd` microbatches, so this is **never invoked**.

Consequence: setting `context_parallel_size=2` builds the CP process groups and
a CP-capable attention kernel, but **every CP rank still receives the entire
sequence**. That is exactly why per-GPU activation didn't drop and total memory
doubled. It is also a **correctness risk** (cf. hybrid-CP skill pitfall #5:
unsharded/misconfigured CP silently produces wrong gradients) — CP>1 in this
codebase is not just memory-useless, its numerics are unverified.

### Conclusion: CP is the wrong lever; PP is the real one

Two compounding reasons CP cannot rescue 131k for this model here:

1. **Not wired** (above): no sequence slice in our data path → zero activation
   savings + suspect numerics.
2. **Architecturally weak even if wired**: Mamba2 layers are a recurrent scan
   over the sequence, which does not context-parallelize like pairwise
   attention. CP would shard the attention layers but not cleanly the SSM
   layers.

The lever that actually reduces this model's per-GPU activation is **pipeline
parallelism (PP)** — it shards *layers* across stages, so each GPU retains only
its stage's activation checkpoints, and it sidesteps the Mamba-CP problem
entirely. PP is already wired in this codebase (`get_forward_backward_func`,
`provider.pipeline_model_parallel_size`, fixed-shape handling in
`_forward_batch_seq_length`). Rough extrapolation from the CP=1 ~371 GiB/GPU @
131k estimate, scaling activation ∝ 1/PP at TP=8:

| layout | GPUs / nodes | est. peak/GPU @131k | fits 178 GiB? |
| ------ | ------------ | ------------------- | ------------- |
| TP8, PP1 | 8 / 1   | ~371 GiB | no |
| TP8, PP2 | 16 / 2  | ~190 GiB | borderline-OOM |
| TP8, PP4 | 32 / 4  | ~100 GiB | yes |

So a realistic, validated 131k for Nemotron-3-Super most likely needs **4 nodes
(TP=8, PP=4)**; 2 nodes (TP=8, PP=2) is right at the wall and probably still
OOMs once framework/CUDA overhead is added. This should be re-profiled with the
PP path rather than CP. (Numbers above are extrapolations, not measured.)

## VALIDATED: TP=8, PP=4, CP=1, EP=8 fits 131k (4 nodes / 32× B200)

Measured on a 4-node 8× B200 box, branch `jack-nemo3super-131k` with the PP>1
fixes (see below), full activation recompute, micro-batch 1, LoRA rank 16,
synthetic packed 131072-token sequence:

| layout | GPUs / nodes | measured peak_alloc / GPU @131k | result |
| ------ | ------------ | ------------------------------- | ------ |
| TP8, PP4, CP1, EP8 | 32 / 4 | **100.79 GiB** | **OK, no OOM** |

- baseline (weights, pre-forward): 8.34 GiB/GPU.
- step 1 loss 0.6673, step 2 loss 0.6543 (decreasing → training is real and the
  last-stage loss is correctly broadcast to all PP ranks).
- step time: 491 s on the first step (kernel/cuDNN autotune warmup), **11.3 s
  steady-state** thereafter.
- The measured 100.79 GiB lands almost exactly on the ~100 GiB PP=4
  extrapolation above, confirming the activation ∝ 1/PP model.
- 88 model layers → PP must divide 88: PP∈{1,2,4,8,11,...}. PP=4 (22 layers/
  stage) is the smallest clean split that fits; PP=3 is impossible without an
  uneven pipe-separator layer pattern.

### PP>1 fixes required (all in-repo, see `findings.md`)
1. `megatron_controller`: set `provider._pg_collection` before `get_model`
   (Mamba provider `is_pp_*_stage(self._pg_collection.pp)` NPEs otherwise).
2. `megatron_controller`: `is_pipeline_last_stage()` guards in CE + RL forward
   steps (non-last stages have no LM head / `output_layer`).
3. `megatron_controller`: `_broadcast_pipeline_loss_metrics` (loss only on last
   stage) + nonzero-rank publish guard in `write_lora_adapter`.
4. `_bridge_patches`: PP-safe LoRA adapter export; `main.py`:
   `set_device(LOCAL_RANK)`.
5. Deps: `pybind11` (Megatron compiles `helpers_cpp` at startup) + the venv
   `bin` on PATH so the helpers Makefile's `python3 -m pybind11` resolves to the
   venv. `megatron-bridge[ssm]` for mamba-ssm/causal-conv1d.

### Attention backend (resolved)

The model's hybrid provider hardcodes `AttnBackend.flash`, but on B200 the only
installed flash build is FlashAttention-4, and **TE 2.16 explicitly disables FA4
whenever CP>1** (`dot_product_attention/utils.py`: "Disabling FlashAttention 4 as
it does not support context parallelism yet"). With CP>1 the flash-only policy
leaves zero usable backends → `No dot product attention backend is available`.
Fix in `megatron_controller.py`: conditionally set `AttnBackend.fused` (CP-capable
cuDNN FusedAttention) when CP>1, and keep the native FA4 flash path at CP==1.
This is what the CP=2 sweep above ran on.

## Reproduce

Config: `examples/trainer-configs/nemotron3-super-b200-131k-pp4-lora.json`
(TP=8, PP=4, CP=1, EP=8, max_seq_len=131072). Launch one rank per node on a
4-node 8× B200 job (rank 0 is the rendezvous master). `BT_LEADER_ADDR` is
injected on Baseten multinode jobs; the weights load from a shared HF cache:

```bash
# on each node N in 0..3 (e.g. via tmux so it detaches):
export HF_HOME=/root/.cache/user_artifacts/huggingface HF_HUB_OFFLINE=1
NUM_NODES=4 NUM_GPUS=8 CONFIG=/root/trainer_config.json \
  bash examples/ptt-nemotron3-super-sft/run_trainer_node.sh <N>
```

If Megatron's `helpers_cpp` fails to compile with `pybind11/pybind11.h: No such
file or directory`, ensure `pybind11` is installed in the venv AND the venv
`bin` is on PATH (so the helpers Makefile's `python3 -m pybind11` resolves to
the venv), e.g. prepend `PATH=<server>/.venv/bin:$PATH`.

Drive a memory probe at a given sequence length (run from a box that can reach
rank-0's `:8000`):

```bash
python sft_driver.py --source synthetic --seq-len 131072 --steps 2 --microbatch-size 1
# real long-context SFT:
python sft_driver.py --source dataset \
  --dataset-name nvidia/ChatQA2-Long-SFT-data --dataset-config NarrativeQA_131072 \
  --seq-len 131072 --steps 2
```

`[RESULT] ... peak_alloc_max=<X>GiB` is the per-GPU peak; compare across
`--seq-len` values to recover the per-token slope.
