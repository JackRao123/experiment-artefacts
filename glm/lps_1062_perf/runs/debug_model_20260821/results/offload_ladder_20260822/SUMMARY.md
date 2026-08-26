# Activation-offload ladder

Date: 2026-08-22

## Objective

Measure process-lifetime peak GPU allocation, peak GPU reservation, and
forward/backward throughput while cumulatively enabling activation-offload
groups on the 0D2M GLM-5.2 B300 proxy. Recompute is disabled in every arm.

## Provenance

```text
trainers:        c226338ab13dfa709e4ac7c126ded016b513ea1b
Megatron-Bridge: cad4f085cde65ba5d884c32545c5e7a70fee1b97
Megatron-LM:     d7a72ec1fead20e0c32048836fb3374737cb952e
```

- Host: `tj-wdpok4w`, one B300 exposed as `cuda:0`.
- Snapshot: `/root/.cache/user_artifacts/glm52-debug-0d2m`.
- Shape: TP1/PP1/CP1/EP1, 8,192 tokens, one datum per window.
- Each arm used a fresh trainer process, one warmup window, and three measured
  windows with deterministic inputs.
- Weight sync and recompute were disabled. Offload used fraction 1.0, minimum
  tensor size 0, no inflight cap, NUMA binding, and the backward-order prefetch
  implementation above.
- `aggregate TPS` is 8,192 divided by the arithmetic mean forward/backward
  duration. `median TPS` is the median of the three individual measured-window
  rates and is the better summary here because several arms had slow outliers.
- Raw results are in `raw/`, trainer logs in `logs/`, and exact inputs and
  server config are in `../../configs/activation_offload_ladder_20260822/`.
  The exact benchmark driver, MFU helper, and runners are in `../../tools/`.

## Results

`incremental saved` is the peak-allocation decrease from the preceding row,
attributed to the newly added group on one steadily offloaded MoE layer.
`cumulative saved` is the total decrease from the no-offload baseline. The two
support columns also describe the newly added group, not the cumulative set.
The final row adds five groups at once, so its incremental and support cells
describe that bundle.

| newly added group(s) | stock MCore recognized it? | worked on GLM-5.2 before our changes? | incremental saved | cumulative saved | median TPS/GPU |
|---|---|---|---:|---:|---:|
| None: no recompute, no offload | N/A | N/A | 0 | 0 | 13,213 |
| `core_attn` | Yes | No | 1.151 GiB | 1.151 GiB | 12,464 |
| `moe_act` | Yes | Yes | 1.250 GiB | 2.402 GiB | 10,959 |
| `qkv_linear` | Yes | No | 0.169 GiB | 2.571 GiB | 8,152 |
| `attn_proj` | Yes | No | 0.250 GiB | 2.822 GiB | 11,980 |
| `moe_router` | No | No | 0.006 GiB | 2.828 GiB | 11,985 |
| `attn_norm`, `mlp_norm`, `moe_dispatcher`, `expert_fc1`, `shared_experts` | Mixed | Mixed | 0.330 GiB | 3.157 GiB | 9,644 |

<!--
Columns intentionally hidden from the rendered table, retained here for the
self-contained experiment record:

| arm | peak allocated | peak reserved | aggregate TPS/GPU | measured-window TPS/GPU |
|---|---:|---:|---:|---|
| No recompute, no offload | 65.155 GiB | 68.223 GiB | 13,026 | 12,519; 13,378; 13,213 |
| core_attn | 64.003 GiB | 67.189 GiB | 11,087 | 8,950; 12,720; 12,464 |
| + moe_act | 62.753 GiB | 67.365 GiB | 7,492 | 10,959; 12,175; 4,405 |
| + qkv_linear | 62.583 GiB | 68.398 GiB | 7,671 | 5,316; 12,454; 8,152 |
| + attn_proj | 62.333 GiB | 67.404 GiB | 11,832 | 11,180; 12,402; 11,980 |
| + moe_router | 62.327 GiB | 67.545 GiB | 11,797 | 11,241; 12,211; 11,985 |
| All supported groups | 61.997 GiB | 67.525 GiB | 9,270 | 7,560; 9,644; 11,406 |
-->

Exact values, including byte counters, losses, gradient norms, and timings, are
also flattened into `results.csv`.

## Interpretation

The table reports whole-process peak deltas. However, they effectively measure
one offloaded MoE layer: the proxy has two cloned MoE layers, and in steady
state MCore offloads the earlier layer while retaining the final layer's
same-name groups for immediate backward. Do not divide the deltas by two.

The `core_attn` result validates that interpretation directly. Its peak
allocated reduction is 1.151367664 GiB, exactly 1,179 MiB, matching the
one-offloaded-layer transfer census in the parent notebook.

Stock MCore already recognized `core_attn`, `moe_act`, `qkv_linear`, and
`attn_proj` as offload group names. Only `moe_act` was already wired end to end
on this GLM-5.2 path. AbsorbedMLA needed new `core_attn`, `qkv_linear`, and
`attn_proj` scopes. `moe_router` was entirely new: vocabulary, validation, and
the routing/preprocessing scope were added together. In the final bundle,
`attn_norm`, `mlp_norm`, and `expert_fc1` were stock names, while
`moe_dispatcher` and `shared_experts` were added by this work.

At this proxy shape:

- `core_attn` saves 1.151 GiB per offloaded MoE layer and costs 5.7% median TPS
  versus no recompute/no offload.
- `core_attn`, `moe_act`, `qkv_linear`, and `attn_proj` save 2.822 GiB per
  offloaded MoE layer and cost 9.3% median TPS.
- Adding `moe_router` changes peak allocated by only 0.006 GiB in this run.
- All groups save 3.157 GiB per offloaded MoE layer and cost 27.0% median TPS.
- Peak reserved is not monotonic and should not be used for group attribution;
  allocator caching and process history dominate it.

The losses agree across all arms to approximately 2e-6 or better, and gradient
norms agree to approximately 1e-8 or better. The throughput sample is only
three measured windows per arm and contains substantial variance, especially
for the `moe_act` and `qkv_linear` cumulative arms. Treat memory deltas as the
primary result and TPS differences as directional.

Production scaling is not a blind multiplication by model layer count. It must
account for which layers are actually offloaded, final-layer residency, dense
versus MoE layers, sequence length, and TP/CP/EP topology.

## Layer-depth follow-up

The selected four-group policy was subsequently tested at two, four, and six
MoE layers. Allocation savings were byte-exactly linear and correctness held,
but the throughput penalty increased materially at four and six layers. See
`../layer_scaling_20260822/SUMMARY.md` for the complete result.
