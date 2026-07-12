# Nemotron-3-Super: module inventory & recompute activation gap

Question: with selective recompute we only checkpoint *some* submodules, so we
save less activation than a hypothetical `granularity="full"` (every module).
What modules does the model actually have, how much per-token activation does
each hold, and how much are we leaving on the table vs. full recompute?

Source of truth: HF `config.json` for
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` + the measured slopes in
`profiling.md`. Slopes are MiB/token/GPU at **TP=8, PP=1, EP=8, SP on** (the
single-node profiling box). bf16, LoRA r16, micro-batch 1.

## 1. Module inventory (88 layers)

`hybrid_override_pattern` (88 chars) decodes to:

| char | module | count | what it is |
| ---- | ------ | ----- | ---------- |
| `M`  | Mamba2 mixer  | **40** | SSM/recurrent scan. `d_inner=8192` (expand 2), 128 heads × hd 64, `n_groups=8`, `d_state=128`, `conv_k=4`, `chunk=128` |
| `E`  | MoE FFN       | **40** | 512 routed experts, **top-22**, 1 shared expert, expert int 2688, shared int 5376, `relu2` |
| `*`  | Attention     | **8**  | GQA 32 q / 2 kv heads, hd 128, full RoPE |

(No `-`/dense layers exist, so the `mlp` recompute module is a genuine no-op,
and `mla_up_proj` is N/A — matches the `profiling.md` notes. Note `profiling.md`
said "44 E layers" in passing; the config says **40**.)

Plus embedding, final norm, and a `vocab=131072` LM head (handled separately /
chunked, not part of the per-token layer slope).

`n_groups=8` is the hard cap forcing **TP ≤ 8** (`ngroups % tp == 0`).

## 2. What selective recompute actually buys (measured)

| recompute setting | slope (MiB/tok/GPU) | Δ |
| ----------------- | ------------------- | - |
| none / `"full"` (inert on HybridStack) | **2.57** | — |
| selective `core_attn,moe,moe_act` | 1.174 | −1.40 |
| selective `+layernorm` (current golden) | **1.155** | −0.02 |

Selective removes **1.415 MiB/tok = 55%** of the per-token activation. From the
analytical per-module sizes below, essentially **all** of that saving is the
**MoE** layers (the top-22 expert expansion is the single biggest per-token
tensor); attention contributes ~0.016 MiB/tok (flash already stores almost
nothing), and `layernorm` 0.019 MiB/tok.

## 3. The floor we cannot touch, and the full-recompute target

The remaining **1.155 MiB/tok** floor is everything selective can't checkpoint.
A hypothetical `"full"` (checkpoint each whole layer, store only its input)
would, with sequence-parallel on, retain just the 88 layer-boundary residuals:

```
88 layers × (H/TP) × 2 B = 88 × 512 × 2 = 88 KiB/tok = 0.086 MiB/tok   (+ a 1-layer recompute transient)
```

So **what we fail to save vs full ≈ 1.155 − 0.086 ≈ 1.07 MiB/tok/GPU.**

That gap is dominated by the **40 Mamba2 layers**, which have **no recompute
path** in this Megatron pin (`HybridStack`/`MambaLayer` implement neither
full/layer nor selective checkpointing). Attributing the whole gap to Mamba ⇒
~27 KiB/tok/GPU per Mamba layer. Independent analytical estimate of a Mamba2
layer's saved tensors (in_proj z/x/B/C/dt + conv out + SSD chunk states in fp32
+ intra-chunk blocks, TP=8 sharded) ≈ **17–27 KiB/tok/layer**, i.e. 0.67–1.07
MiB/tok over 40 layers — corroborating that Mamba is essentially the entire
floor (the rest is residual/LN activations selective keeps but full would drop).

## 4. How much "context" this costs

Single node, 178 GiB, ~30 GiB weights ⇒ ~148 GiB activation budget:

| setting | slope MiB/tok | single-node context cap |
| ------- | ------------- | ----------------------- |
| no recompute | 2.57 | ~59K |
| **selective (today)** | **1.155** | **~131K** |
| full floor (residual only) | 0.086 | ~1.7M (unreachable; transient-bound) |
| full, conservative (~0.25 w/ transient) | 0.25 | ~600K |

At fixed **S=131072, PP=1**: selective holds **~148 GiB** activation/GPU vs a
full-recompute floor of **~11 GiB** — a **~137 GiB/GPU** gap. That gap is
exactly why 131K does **not** fit on one node and we fall back to **PP=4 /
4 nodes**. If Mamba-layer checkpointing existed upstream, 131K would very
plausibly fit on a single node.

## 5. Takeaway

- Selective recompute is doing its job: it kills the MoE activation (55% of the
  total), which is the only big thing it *can* reach.
- The ~1.07 MiB/tok we can't recover is the **40 Mamba2 layers**, blocked by a
  missing `HybridStack`/`MambaLayer` recompute implementation — a vendor patch,
  not a config change. That's the lever for getting 131K onto fewer nodes.
