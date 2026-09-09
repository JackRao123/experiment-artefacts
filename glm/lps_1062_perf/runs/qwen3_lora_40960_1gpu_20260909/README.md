# Qwen3 LoRA SFT: 40,960 tokens, one B300

Run date: 2026-09-09. Devbox: `tj-32vj99q`. Trainer revision:
`3962e7af2a3f87ae6f471b2408ead5c4c1e63dc9`.

Both models use GPU 0 exclusively, TP=PP=CP=EP=1, BF16 base weights,
LoRA rank/alpha 32, full decoder-block activation recomputation, and Adam
learning rate 1e-5. Every step uses the same single 40,960-token synthetic
datum, seed `0xB300`, vocabulary 151,936. There are 40,959 loss tokens after
the causal-label shift. TPS counts the 40,960 input tokens, matching the driver.
The run is a performance measurement, not a quality evaluation.

Final protocol per model: one warmup, **10 unprofiled controls**, one
memory-profiled forward/backward + optimizer step, then one runtime-profiled
forward/backward + optimizer step. The final JSON names end in `-c10.json`.
Earlier 0.6B measurements are retained as preliminary records and do not
contribute to final statistics. Both final protocols start from fresh trainers.

## Files and reproduction

- `profile_driver.py`: Qwen-specific copy of the original driver; identical
  request/timing protocol, corrected vocabulary, node-local output directory,
  no GLM-only FLOP calculations.
- `mfu.py`: standalone FLOP estimator and benchmark statistics.
- `collect_inventory.py`, `inventories/`: complete tensor-header inventory and
  grouped shape/dtype census from the actual downloaded checkpoints.
- `runs/`: exact trainer configs, raw benchmark records, logs, and MFU results.
- `qwen3-06b.pt.trace.json`, `qwen3-30b-a3b.pt.trace.json`: final Kineto traces.
- `qwen3-06b.memory.rank0.pickle`, `qwen3-30b-a3b.memory.rank0.pickle`: final
  PyTorch CUDA allocator snapshots, usable in PyTorch memory_viz.
- `lifecycle/`: copies of the generated devbox startup/wait/stop scripts.

The shared filesystem rejected writes with ENOSPC. Lifecycle state, configs,
results and profiles therefore live node-locally on the devbox. The original
generated scripts were preserved: the copies change only their state/script
directory and allow `NUM_GPUS` to override the hardcoded eight processes.
Launch uses `NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0`. The 0.6B checkpoint loads
from team cache; the 30B checkpoint loads from node-local staging.

Driver command on a healthy trainer:

```bash
/root/.devbox-venvs/server/bin/python /root/qwen06-profile/profile_driver.py \
  --label qwen3-06b-40960-1gpu-c10 --seq-len 40960 --datums 1 \
  --num-gpus 1 --lora-rank 32 --control-repeats 10 \
  --memory-profile --runtime-profile
```

From this directory on the Mac:

```bash
python3 mfu.py --model 0.6b \
  --benchmark runs/qwen3-06b-40960-1gpu-c10.json \
  --output runs/qwen3-06b-mfu.json
python3 mfu.py --model 30b-a3b \
  --benchmark runs/qwen3-30b-a3b-40960-1gpu-c10.json \
  --output runs/qwen3-30b-a3b-mfu.json
```

Original `tools/mfu.py` and `tools/profile_driver.py` remain pristine. Their Git
blob hashes are `1eea8b85f6583e7281a584ac52ff062f4062a417` and
`209b8c574f53ea76aa3f4ceff232bbb4d4530393`, respectively.

## FLOP accounting

FMA counts as two FLOPs. Let H=hidden size, Q=query heads × head dimension,
K=KV heads × head dimension, I=MLP width, N=layers, V=vocabulary, S=sequence
length, and k=active experts (one for dense MLPs).

Forward FLOPs per input token:

- Attention projection GEMMs: `2 N H (2 Q + 2 K)`.
- SwiGLU GEMMs: `2 N k (3 H I)`.
- MoE router GEMM: `2 N H E`, where E is the total expert count.
- Output projection: `2 H V`, even when its weights are tied. Embedding lookup
  is not counted as a matrix multiply.
- Causal QK-transpose and probability-times-V: `2 N Q (S+1)`, from exactly
  `S(S+1)/2` causal pairs. GQA reduces K/V projections, not query attention work.
- Rank-r adapter on a fused linear with dimensions a→b: `2 r (a+b)`.
  This trainer uses fused QKV and fused gate/up adapters, not separate HF
  q/k/v and gate/up adapters. Routed adapter FLOPs scale with k even though
  their weights are shared across experts at EP=1. Router has no adapter.

Define B=base GEMM forward FLOPs, A=causal attention forward FLOPs, and
L=adapter forward FLOPs. Useful LoRA SFT FLOPs are **`2B + 3A + 3L`**:
frozen weights require forward + input-gradient, while attention and adapters
require full backward. Full-weight `6 × parameter_count` is inappropriate.
The tied 0.6B output head has no LoRA; the untied 30B head does.

The useful estimates at S=40,960 and r=32 are **16.581738496 GFLOP/token**
(0.6B) and **61.110296576 GFLOP/token** (30B-A3B). The estimator audits every
checkpoint tensor name and shape. Unique base + adapter counts match the
trainer's logged totals exactly: **613,482,496** and **30,567,327,744**.
The 0.6B checkpoint serializes both tied embedding/head tensors; its raw
on-disk count is therefore larger than its unique runtime parameter count.

Headline MFU = unprofiled forward/backward TPS/GPU × useful FLOPs/token /
**2.25e15 FLOP/s**, NVIDIA's dense BF16 B300 peak. `--peak-tflops` is explicit
and overrideable; 2.5 PFLOP/s from the original GLM helper is not used.
TPS is total control tokens divided by total control elapsed time, not an
arithmetic mean of per-step rates. Optimizer-inclusive TPS/MFU are also saved.
Runtime- and memory-profiled timings never enter the headline MFU.

The pass-model HFU adds one decoder-block forward for full checkpointing;
it does not incorrectly checkpoint the output head. The flash-model HFU
additionally estimates one score-matrix reconstruction in attention backward.
These are analytic execution estimates, **not hardware-counter measurements**.
Scalar operations (norms, RoPE, softmax, SwiGLU activation, routing/reduction,
cross entropy), Adam, communication, kernel tiling/padding and extra implementation
work are excluded. The last masked output token introduces a negligible
one-token head overcount. FLOPs are accurate for the stated dominant-GEMM
convention, not an exact instruction count. Packed multi-document inputs
require summing causal pairs per document; using their total length would
overestimate attention. Other MFU tools may use full-square attention rather
than causal pairs, so their values need not be directly comparable.

## Sources

- [Qwen3-0.6B config, pinned checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
- [Qwen3-30B-A3B config, pinned checkpoint](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)
- [HF Qwen3 modeling, v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
- [HF Qwen3 MoE modeling, v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
- [NVIDIA dense peak table](https://github.com/NVIDIA/exemplar-performance#peak-theoretical-throughput)
- Trainer source at the revision above: `lora_targets.py`, `backend.py`,
  `chunked_lm_head.py`, `models/src/loops_models/control.py`, and vendored
  `megatron/bridge/peft/lora.py`. The actual Megatron implementation governs
  adapter fusion/sharing, frozen weights and recomputation, beyond HF config.
- Reference notebook: `../../analysis/tensor_analysis.ipynb`. It provides the
  tensor census approach; it primarily analyzes memory, not full operation counts.
