# LAUNCH-TAX SCOPING — kernel-launch reduction on our stack (lebesgue, 2026-08-13)

Mac-side source scoping, fermi's optional idle task. Target: the **~4.4 s/step
generic per-kernel dispatch tax** — the residual class in kepler's
IDLE_WINDOW_DECOMPOSITION.md with no lever (10.41 s/step diffuse sub-1ms tax
total, 897k gaps, mean 11.6 µs; ~6 s is indexer/a2a/cat-adjacent and attacked
by those named levers; the 4.4 s is the generic remainder at 1.07M GPU
ops/step, d16/fe127, rank-0). Question: is there a ≤2d lever here, or is this
a roadmap item? No pre-registration (scoping, not measurement).

**TL;DR: roadmap item.** The only mechanism that attacks *diffuse* launch tax
is CUDA graph capture, and every capture scope mcore offers is disqualified on
this stack by data-dependent host reads (DSA indexer/layout, MoE alltoall
dispatch sizing, THD layout). The unlock chain is multi-day and runs through
the staged host-sync stack (B/F + A-v3 per IDLE_RESIDUAL_MAP) anyway. Details
and citations below.

## 1. mcore's CUDA-graph support surface (as pinned, 57efae08b)

Config surface (transformer_config.py:979-1050):

- `cuda_graph_impl`: `none` (default) | `local` | `transformer_engine` |
  `full_iteration`.
  - `local`: per-layer graphs via `CudaGraphManager` (module.py:157,
    `GraphableMegatronModule`; capture/replay machinery in cuda_graphs.py —
    static input/output/grad buffers throughout, the standard graph
    requirement: fixed shapes, fixed addresses).
  - `transformer_engine`: TE `make_graphed_callables()`, N graphs per layer
    (N = microbatches, for pipeline fwd-before-bwd); the training script must
    populate `layer.cuda_graphs` before the first step.
  - `full_iteration`: one graph for the whole fwd-bwd minus optimizer.
- `cuda_graph_modules`: per-layer capture scopes — `attn`, `mlp`, `moe`,
  `moe_router` (up to `MoELayer.router()`), `moe_preprocess`, `mamba`; empty =
  whole layer. mcore anticipates *partial* layer capture (router-boundary
  scopes), which matters in §3.
- Warmup: `cuda_graph_warmup_steps=3`; single-mempool option for
  full_iteration.

**Executor compatibility (the campaign's direction):** the combined-1F1B
executor's schedule-plan world is graph-*aware* but only via the TE path:
`_BackwardDWWrapper` (fine_grained_callables.py:421-470) exists explicitly for
"cuda graphed ep overlap", gating wgrad callables on whether a layer is
replaying a graph, with `GraphableMegatronModule`/`TransformerLayer` asserts.
There is no `local`-impl integration with the schedule-plan nodes, and nothing
in the trainer plumbs any of this: `cuda_graph_impl` does not exist in
`loops_models.control` / `megatron_config.py` — zero trainer surface today.

## 2. The disqualifiers, specific to us — hard wall vs engineering

| Feature | Where it bites | Verdict |
|---|---|---|
| **DSA indexer/layout host reads** | `dsa_layout.py:127,152,192` (`.item()` on cu_seqlens-derived values), `:158-161` (`nonzero` → data-dependent shapes), `dsa.py:242-244` (`.tolist()`/`.sum().item()`), `dsa_cudnn_kernels.py:2130` (`torch.nonzero(topk_length > 0)` in the bwd) — per layer, per microbatch; kepler measured 55,328 `aten::nonzero`/step | **Hard wall** for any region containing attention (i.e. every layer). Capture fails or replays wrong shapes. The elimination path is the staged host-sync stack (B/F + A-v3 per IDLE_RESIDUAL_MAP: FIX B's layout cache owns 98.9% of the calls / 89% of the leak, A-v3 the bwd compaction) — the graph unlock and the host-read elimination are the same project |
| **MoE dynamic routing** | alltoall dispatcher DtoH for `num_global_tokens_per_local_expert` before permutation (token_dispatcher.py:600-603); mission config sets **no** `moe_expert_capacity` → data-dependent expert shapes | **Hard wall as configured.** Engineering escape exists: capacity padding makes shapes static — but `moe_pad_experts_for_cuda_graph_inference` is inference-only (transformer_config.py:870), and training-side capacity padding changes routing semantics (dropping) → needs its own correctness A/B |
| **THD varlen** | Per-rank microbatch *shapes* are static only because the mission tail-pads to 131k; but `cu_seqlens` content is data-dependent and read on host per microbatch by the DSA layout (above) | **Hard wall** in practice — via the DSA reads, not the padding |
| **LoRA** | Adapter matmuls are pure tensor math (graphable); frozen embedding × executor needed the b37c01f2e grad-root fix (already landed on the W2 tree). No per-forward Python hooks in the trainer's LoRA path | **Not a wall** — neutral after b37c01f2e |
| **Combined-1F1B executor** | Graph integration is TE-impl-only (`_BackwardDWWrapper`), requires the training script to populate per-layer graphs (unplumbed here), and inherits every regional disqualifier above | **Engineering (plumbing) + inherits the hard walls** |

## 3. Partial options, honestly priced

- **Per-layer static-region capture** (`local` impl, or TE on subregions):
  the graphable residue per layer after excluding attention (DSA host reads)
  and MoE dispatch (DtoH + dynamic shapes) is small — norms, projections, the
  dense shared-expert MLP, maybe `moe_router`/`moe_preprocess` scope. At
  ~1,670 ops per layer-microbatch (1.07M / (40 layers × 16 mb)), the capturable
  fraction is a few percent of the tax — call it ≤0.5 s of the 4.4 s — for a
  multi-day plumbing + capture-debugging effort, and graph memory pools pin
  activation memory the 131k memory leg is already fighting over. **Fails the
  ≤2d bar on both cost and EV.**
- **torch.compile regions**: inductor graph-breaks on exactly the same host
  reads (`.item()`/`nonzero`) and on the opaque custom cuDNN DSA kernels
  (`FusedIndexerSparseAttnFunc` & co.); `reduce-overhead`'s cudagraphs inherit
  the walls. Region-compiling dense math (loss head, norms) is possible but
  those aren't the tax mass. **Not a lever for this class.**
- **CUDA work queues / launch batching**: no PyTorch-level mechanism exists
  (no batched-launch API; device-side graph launch and conditional graph nodes
  are CUDA-12.4+ features not exposed to eager PyTorch and still require
  static shapes). **Non-starter.**

## 4. Verdict: roadmap item, with the unlock chain named

There is **no ≤2d lever** on the 4.4 s generic launch tax. The EV math: even
kepler's optimistic "fusion/graphs 2–4 s" bundles the *named*-class fusion
(indexer/a2a/cat-adjacent, attacked by their own levers); the generic residue
addressable by graphs alone is smaller, and every graph scope is gated behind:

1. **B/F + A-v3 host-read elimination** (the staged host-sync stack per
   IDLE_RESIDUAL_MAP — FIX B's CP-layout cache owns 54,720 of the 55,328
   nonzero calls/step and ~89% of the leak; A-v3 owns the 608-call bwd
   compaction; the graph-relevant question is whether the cached/layout paths
   drive off device tensors with no per-step host read at replay),
2. **MoE static-shape dispatch** (capacity padding + its correctness A/B, or a
   sync-free sizing path),
3. **THD layout host-read removal** (device-side cu_seqlens handling).

Only after (1)–(3) does per-layer capture of attention/MoE regions become
available, at which point graphs could attack a real share of the then-
remaining diffuse tax. That is a final-report roadmap item, not a campaign
lever.

**Tension worth recording:** the campaign's own direction (the combined-1F1B
executor) adds host-side node orchestration per layer — more Python/launch
pressure, not less — while it hides the big comm mass. If the W2 trace shows
the diffuse tax *growing* flag-ON vs flag-OFF, that is the early signal that
launch tax becomes binding *after* the overlap prize lands — the right moment
to re-open this scoping. Recommend doppler capture the diffuse-tax line
(mean-gap × gap-count) in the W2 canary traces for exactly this comparison.

## 5. What the near-term attack actually is (not mine — kepler's list, restated)

The 10.41 s diffuse line decomposes as ~6 s named-adjacent (indexer/a2a/cat —
attacked by the staged host-sync stack per IDLE_RESIDUAL_MAP, plus the
dispatch-buffer and cat levers) + 4.4 s generic (this doc: roadmap). Plus seam
python 1–1.5 s (py-spy probe names it in one shot). Total realistic 8–14
s/step (6–10%) — the composition is host-side, and the host-side levers are
the named ones, not graphs.
