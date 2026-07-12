# NVFP4-vs-bf16 spike (Nemotron-3-Ultra, Option A)

Go/no-go harness for serving the Ultra base in **NVFP4** on the sampler while
the trainer stays **bf16** (Option A). It answers the one question that gates
the rest of the work:

> How far do the NVFP4 sampler's per-token logprobs drift from the bf16
> reference, and does the existing importance-sampling correction (`cispo`)
> stay healthy under that drift?

It also surfaces the highest-risk unknown for *this* adapter: two of its three
LoRA target groups (`mixer.in_proj`/`out_proj` and `mixer.shared_experts.*`)
land on **FP8** layers in the official NVFP4 checkpoint, and attention
(`q/k/v/o_proj`) lands on BF16. If unmerged LoRA can't be applied on the FP8
Mamba projections, `score.py generate` is where it fails — and that failure is
itself the finding.

## Checkpoints

| role | checkpoint | fits |
|------|-----------|------|
| NVFP4 base (sampler / behavior) | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | 1 B200 node |
| bf16 base (trainer ref / target) | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | multi-node |
| adapter | `~/Documents/nemo3ultra_rl_overnight/adapters/step-199` | — |

Official NVFP4 recipe (for context): routed experts NVFP4 (four-over-six),
shared experts + Mamba projections **FP8**, FP8 KV cache, attention/embeddings
BF16. NVIDIA's reference serve uses `--kv-cache-dtype fp8`,
`--mamba-ssm-cache-dtype float16`, `--mamba-backend flashinfer`.

> The adapter dir must be a PEFT-style dir vLLM can load (an
> `adapter_config.json` next to `adapter_model.safetensors`). `step-199/` only
> has the safetensors here — drop in the matching `adapter_config.json` before
> running, or point `--adapter` at the trainer's full export dir.

## Two stages (mirrors swift_compare / tinker_compare)

1. **`score.py`** — GPU, vLLM offline. One job per run so each lands on the
   right node set:
   - `generate` → NVFP4 base + adapter produce rollouts + per-token logprobs
     (the *behavior* policy).
   - `rescore` → bf16 base + same adapter teacher-force the *same* tokens (the
     *target* policy).
   - `boot` → init engine, dump KV/Mamba capacity, exit (headroom probe).
2. **`compare.py`** — pure CPU. Logprob drift, KL (k1/k3), importance-weight
   distribution, clip fraction, token-level ESS, and a coarse verdict.

## Run (5-node devbox)

`--tp`/`--pp` below are placeholders — set them to whatever your devbox
topology dictates (the NVFP4 box is single-node; bf16 spans nodes).

```bash
cd tests/comparison_studies/nvfp4_compare

# 1) behavior: NVFP4 sampler generates rollouts (single node)
python score.py generate \
  --model /path/to/nemotron3-ultra-nvfp4 \
  --adapter /path/to/step-199 \
  --prompts prompts.jsonl \
  --tp 8 --max-model-len 131072 --max-new-tokens 512 \
  --kv-cache-dtype fp8 \
  --out out/gen_nvfp4.json

# 2) target: bf16 base rescores the exact same tokens (multi-node)
python score.py rescore \
  --model /path/to/nemotron3-ultra-bf16 \
  --adapter /path/to/step-199 \
  --sequences out/gen_nvfp4.json \
  --tp 8 --pp 4 --max-model-len 131072 \
  --out out/rescore_bf16.json

# 3) divergence + ESS/clip verdict (no GPU)
python compare.py \
  --behavior out/gen_nvfp4.json \
  --target out/rescore_bf16.json \
  --clip-low 0.8 --clip-high 1.2 \
  --out out/report.json
```

### Recommended extra runs

- **Engine-mismatch floor**: also `rescore` with the **NVFP4** checkpoint and
  compare against the NVFP4 `generate`. generate-vs-rescore on the same model
  should be ~0 drift; any gap there is API/teacher-forcing artifact, not
  quantization, and tells you how much of the bf16-vs-NVFP4 number to discount.
- **bf16-vs-bf16**: if you can also `generate` from bf16, comparing bf16
  generate vs bf16 rescore isolates the pure engine mismatch that already
  exists in production — subtract it to read the *incremental* NVFP4 cost.
- **Headroom**: `boot` against NVFP4 and bf16 on the same node count; diff the
  reported KV/Mamba capacity to quantify the concurrency gain (the whole point
  of Option A). Cross-check vLLM's own `GPU KV cache size` / `Maximum
  concurrency` boot log lines — authoritative for hybrid layouts.

## Interpreting the verdict

- **HEALTHY**: clip fraction < 5%, ESS/N > 0.8, k3 KL < 0.02 → NVFP4 gap is
  small, `cispo` corrects it comfortably. Proceed to the serving build.
- **MARGINAL**: watch ESS/clip during a real short RL run before trusting.
- **COLLAPSE RISK**: heavy clipping / ESS/N well below 0.5 → NVFP4 drifted too
  far; fall back to FP8 base (2x savings, smaller gap) or revisit QLoRA.

The thresholds in `compare._verdict` are a first cut — recalibrate after the
engine-mismatch floor run, since some drift is the pre-existing
Megatron-vs-vLLM gap, not quantization.

## Validate the CPU math

```bash
python test_compare.py          # or: python -m pytest test_compare.py
```
