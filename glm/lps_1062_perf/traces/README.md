# Trace capture

Use `../tools/profile_driver.py` when recording traces.

For every curated trace, record:

- `--seq-len`
- `--datums`
- `--num-gpus`
- `--lora-rank`
- `--control-repeats`

# Curated traces

## `glm52-b300-cp16-ep16-256k-rank0.pt.trace.json`

Rank-0 Kineto runtime trace from the original GLM-5.2 B300 256K baseline:

- Model: GLM-5.2, LoRA rank 32
- Hardware: 2 nodes × 8 B300 GPUs
- Topology: TP1 / PP1 / CP16 / EP16 / ETP1 / DP1
- Sequence length: 262,144
- SHA: `0e0b65a69b52767676d54f6402397ce9c4f5df3c`
- Captured: 2026-08-06
- Purpose: understand the communication-heavy CP16/EP16 execution profile,
  particularly NCCL and MoE communication costs