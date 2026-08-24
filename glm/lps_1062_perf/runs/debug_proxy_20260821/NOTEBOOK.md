# LPS-1062 single-GPU debug proxy notebook

Date: 2026-08-21

Status: complete for the 0D2M per-layer offload-coverage pass and cumulative
offload ladder. The trainer on `tj-wdpok4w` was stopped after the artifacts
below were copied locally.

## Objective

Create a fast, real-code GLM-5.2 proxy for activation recompute/offload work.
The proxy must use the production GLM5 Bridge, AbsorbedMLA, DSA, MoE, LoRA,
and trainer control path, but avoid each experiment requiring a 2-node,
78-layer, 131k-context boot.

The governing rule is: shrink configuration and tensor counts, not the
mechanism implementation. A prior no-model toy reproduced recompute-only and
offload-only independently but did not reproduce their real-stack interaction.

## Production anchor

- Model: GLM-5.2-FP8, 78 transformer layers (3 dense + 75 MoE).
- Target shape: 2 nodes x 8 B300, TP1/PP2/CP8/EP8, 131,072 tokens per datum.
- Baseline of record: 645.1 tok/s/GPU at d2 (262,144 tokens/step).
- Baseline plateaus: 152.5 GiB first stage, 162.6 GiB last stage.
- Full layer recompute costs approximately 35% of the production step.
- Goal: recompute cheap operations, offload large saved activations, retain the
  rest; target approximately +40% throughput.

## Code provenance

The live trainer resolves to this exact chain:

```text
trainers:        1764a816979cb1ddd879aecffda211dc1df17b24
Megatron-Bridge: f95828b6150897f2f827a1e919b42cfbab0d3e2e
Megatron-LM:     2c3d720cf54179e13206465b2474b4e672c85b0c
```

Entrypoint:

```text
server-megatron-bridge/.venv/bin/python
  -u -m trainers_server_main.main --backend megatron_bridge
```

The tracked trainers tree was clean at the pinned SHA during verification.
An untracked legacy `server/` directory existed but was not on the live import
path.

## Mechanism vocabulary and layer placement

| module | mechanism in current investigation | dense layer | MoE layer |
|---|---|---:|---:|
| `core_attn` | selective recompute | yes | yes |
| `layernorm` | selective recompute | yes | yes |
| `mla_up_proj` | selective recompute | yes | yes |
| `moe_act` | selective recompute | no | yes |
| `attn_proj` | activation offload | yes | yes |
| `expert_fc1` | activation offload | no | yes |

The dense/MoE distinction affects the feed-forward half. AbsorbedMLA attention
and layer norms exist in both layer types. `mlp` selective recompute would be
dense-specific, but it is not present in the current failure matrix.

## Full-scale failure matrix received from the production investigation

| config | recompute | offload | outcome |
|---|---|---|---|
| 3b | `core_attn` | `moe_act`, `attn_proj` | OOM, 258.40 GiB allocated |
| v1 | `core_attn`, `layernorm`, `moe_act`, `mla_up_proj` | `attn_proj`, `expert_fc1` | `_recompute` crash |
| v2 | `core_attn`, `layernorm`, `moe_act` | `attn_proj`, `expert_fc1` | `_recompute` crash |
| v3 | `core_attn`, `layernorm` | `attn_proj`, `expert_fc1` | OOM, 256.87 GiB allocated |

Under fixed offload `[attn_proj, expert_fc1]`, this isolates `moe_act`
recompute as necessary for the `_recompute` crash. It does not yet distinguish
whether the interaction requires `expert_fc1`, `attn_proj`, either offload
window, or merely any active offload machinery.

The intended minimal matrix is:

| recompute | offload |
|---|---|
| `moe_act` | none |
| `moe_act` | `attn_proj` |
| `moe_act` | `expert_fc1` |
| `moe_act` | `attn_proj`, `expert_fc1` |

## Proxy models

### 1 dense + 1 MoE

The first proxy preserved one dense-prefix layer and one MoE layer.

- Architecture config: `configs/glm52-debug-1d1m-config.json`.
- Trainer config: `configs/trainer_config_debug1d1m.json`.
- Remote snapshot: `/root/.cache/user_artifacts/glm52-debug-1d1m`.
- Snapshot size: approximately 24.3 GB.
- Runtime parameter count: 12,192,366,080.
- Context: 8,192.
- Topology: TP1/PP1/CP1/EP1.

This model was useful to prove the real trainer stack boots, but the dense layer
is not required for the suspected MoE-only `moe_act x expert_fc1` interaction.

### 0 dense + 1 MoE

The current proxy removes the dense layer entirely.

- Architecture config: `configs/glm52-debug-0d1m-config.json`.
- Full recompute config: `configs/trainer_config_debug0d1m_full.json`.
- No-recompute config: `configs/trainer_config_debug0d1m_none.json`.
- Remote snapshot: `/root/.cache/user_artifacts/glm52-debug-0d1m`.
- Snapshot size: 21.94 GiB.
- Context: 8,192.
- Topology: TP1/PP1/CP1/EP1.
- Routed experts: 256; top-k: 8; shared experts: 1.

Production layer 6 was used as the shape source because it is a MoE layer that
owns a DSA indexer. It was renumbered to layer 0. This avoids the missing-indexer
warnings seen when the 1D1M snapshot used production layer 3, an index-sharing
layer that does not own indexer weights.

### 0 dense + 2 MoE

The steady-state offload proxy duplicates the 0D1M MoE layer so the first
same-name group has another layer's compute before backward consumes it.

- Architecture config: `configs/glm52-debug-0d2m-config.json`.
- No-recompute/no-offload config: `configs/trainer_config_debug0d2m_none.json`.
- Maximal-offload config: `configs/trainer_config_debug0d2m_all_offload.json`.
- Remote snapshot: `/root/.cache/user_artifacts/glm52-debug-0d2m`.
- Snapshot size: 40.34 GiB of tensors, 41 GiB on disk.
- Layers: two MoE layers, both with full DSA indexers.
- Builder: `tools/build_debug_snapshot_0d2m.py`.

This is deliberately a mechanism proxy rather than a distinct pretrained
checkpoint: layer 1 starts as a clone of layer 0. The second layer exists to
exercise mcore's earlier-group offload policy.

## Checkpoint representation caveat

Both proxies use random bf16 weights. The production `quantization_config` was
removed because these snapshots do not contain E4M3 FP8 matrices or
`weight_scale_inv` block scales. Leaving the config in place would falsely
describe the snapshot and send the loader down the wrong path.

This preserves architecture and activation mechanisms but does not preserve
FP8 checkpoint storage or its weight-memory/startup characteristics. It is
appropriate for saved-activation correctness and memory deltas, not an exact
production model-state footprint.

Builders:

- `tools/build_debug_snapshot.py`
- `tools/build_debug_snapshot_0d1m.py`

## Devbox launch

Host: `ssh tj-wdpok4w` (one B300, reported as `NVIDIA L20D`).

```bash
source /root/.cache/user_artifacts/env.sh
D=/root/.cache/user_artifacts/.devbox_up

export BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/trainer_config_debug0d1m_none.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/trainer-server-config.json

bash "$D/start_trainer.sh"
bash "$D/wait_trainer_health.sh"
```

Stop with:

```bash
bash "$D/stop_trainer.sh"
```

The generated scripts on this box required migration repairs after trainers
split `server/` into `server-main/`, `server-interface/`, and
`server-megatron-bridge/`:

- Trainer venv: `server-megatron-bridge/.venv`.
- Launch cwd/script: `server-main/scripts/launch.sh`.
- Required CLI argument: `--backend megatron_bridge`.
- Health liveness process: `trainers_server_main.main`, not `dp_worker.main`.
- Venv target: `make megatron-bridge-venv CUDA_FLAVOR=cu13`.

The existing box scripts were overwritten with corrected versions. Laplace is
fixing the `devbox-up` source generator. Papercut: `pc_1e0438442eb4`.

## Startup timing

Observed launch-to-health times:

| model/config | condition | seconds |
|---|---|---:|
| 1D1M | first/cold launch | 158.1 |
| 1D1M | warm launch | 75 |
| 1D1M | instrumented restart | 103.9 |
| 0D1M full recompute | warm launch | 71 |
| 0D1M no recompute | warm launch | 70 |

Instrumented 1D1M restart decomposition:

| phase | interval | duration |
|---|---:|---:|
| Python/CUDA/Megatron imports and provider setup | 0-48.6s | 48.6s |
| Construct 12.19B-parameter model | 48.6-67.0s | 18.4s |
| Load 24.3 GB bf16 checkpoint | 67.0-75.4s | 8.4s |
| Offload/optimizer/checkpoint setup | 75.4-76.2s | 0.8s |
| Startup forward/backward warmup | 76.2-103.7s | 27.5s |
| Start HTTP server | 103.7-103.9s | 0.2s |

Import time varied substantially with filesystem/page-cache state. Warmup must
not simply be disabled for this investigation: the production recompute/offload
crash itself occurs during startup backward. Shrinking warmup workload is valid;
removing it would remove the target.

## Benchmark protocol

Driver: `lps1062/bench_driver2c.py` on the box.

- Synthetic encoded tokens, deterministic RNG seed `0xB300`.
- One 8,192-token datum per window.
- One warmup window, followed by measured repeats.
- `/forward_backward`, then `/optim_step`.
- One GPU.

The driver now records both `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` from `/status`. Both are process-lifetime
high-water marks; the counters are not reset around each request. The legacy
`peak_gpu_memory_bytes` field remains an alias for peak reserved memory.

## 1D1M full-recompute result

The trainer JSON omitted recompute, which resolves to the controller default:

```json
{
  "granularity": "full",
  "method": "uniform",
  "num_layers": 1
}
```

With two layers, uniform one-layer checkpoint groups fully recompute both
layers. Offload was disabled.

| window | forward/backward | FB throughput | optimizer |
|---|---:|---:|---:|
| warmup | 2.827s | 2,898 tok/s/GPU | 0.314s |
| measured | 0.842s | 9,727 tok/s/GPU | 0.006s |

- Peak reserved after startup: 26.74 GiB.
- Peak reserved after benchmark: 43.35 GiB.
- Result: `results/debug1d1m-8192-maxseq.json`.

The MFU/HFU values in this result are invalid because the driver FLOP formula
assumes the full 78-layer model. Timings and token throughput are valid.

## 0D1M recompute comparison

Both arms used the exact same snapshot, one warmup, three measured repeats,
8,192 tokens, and fresh trainer processes. Activation offload was disabled.

No recompute is encoded as selective recompute with an explicit empty module
list because the current trainer schema has no disabled recompute enum:

```json
{
  "granularity": "selective",
  "modules": []
}
```

### Results

| configuration | mean FB time | FB TPS/GPU | peak reserved | startup peak reserved |
|---|---:|---:|---:|---:|
| Full recompute (`full/uniform/1`) | 0.638594s | 12,828.18 | 42.45 GiB | 25.95 GiB |
| No recompute (`selective/[]`) | 0.598197s | 13,694.47 | 46.49 GiB | 25.98 GiB |
| Delta, no recompute vs full | -6.3% | +6.8% | +4.04 GiB (+9.5%) | approximately flat |

Per-window forward/backward times:

```text
full: 0.661528, 0.623647, 0.630606 seconds
none: 0.623871, 0.582060, 0.588661 seconds
```

Exact peak reserved values:

```text
full: 45,581,598,720 bytes
none: 49,920,606,208 bytes
```

Losses and gradient norms matched across arms. Results:

- `results/debug0d1m-full-8192.json`
- `results/debug0d1m-none-8192.json`

### Interpretation

The +6.8% no-recompute throughput result is real for this proxy but must not be
extrapolated to the 78-layer production model. Only one transformer layer is
recomputed, while the large 154,880-vocabulary LM head, embedding, loss, and
request overhead are not recomputed. In production, the fixed head/embedding
cost is amortized across 78 layers, so a roughly 35% full-recompute penalty
remains plausible.

This one-layer proxy is useful for:

- Correctness and crash reproduction.
- Verifying whether offload hooks engage.
- Per-layer activation-memory deltas.
- Fast recompute/offload configuration bisection.

It is not currently representative for:

- End-to-end production throughput ratios.
- PCIe/compute overlap at production scale.
- PP2 in-flight microbatch multiplication.
- CP8/EP8 communication behavior.

## 0D1M no-recompute, maximal-offload smoke

Goal: disable recompute completely and enable every activation-offload module
that the trainer accepts.

Config: `configs/trainer_config_debug0d1m_all_offload.json`.

```json
{
  "recompute": {
    "granularity": "selective",
    "modules": []
  },
  "activation_offload": {
    "enabled": true,
    "modules": [
      "attn_norm",
      "qkv_linear",
      "core_attn",
      "attn_proj",
      "mlp_norm",
      "expert_fc1",
      "moe_act"
    ],
    "min_tensor_size": 0,
    "fraction": 1.0,
    "max_inflight_offloads": null
  }
}
```

`fused_group_mlp` is the eighth mcore vocabulary entry, but the trainer rejects
it at config parse time because `use_transformer_engine_op_fuser` is not
exposed. It cannot be included in this maximal legal arm.

Required boot environment:

```bash
export NVTE_CPU_OFFLOAD_V1=1
```

The live process contained the variable before Transformer Engine import. The
trainer reported:

```text
activation-offload NVTE latch: env=1 latch=True -> OK
fine_grained_activation_offloading=True
```

The trainer reached health and completed an 8,192-token forward/backward plus
optimizer step. Loss and gradient norm matched the no-offload baseline:

```text
loss      11.951154468318887
grad_norm 0.000720879586879164
```

Result: `results/debug0d1m-alloffload-8192.json`.

### Commit telemetry

A second restart enabled:

```bash
export BT_OFFLOAD_VALVE_TELEMETRY=1
export BT_OFFLOAD_VALVE_TELEMETRY_EVERY=1
```

The cumulative counters reported one uncapped commit for each of these five
windows after startup warmup:

```text
attn_norm:  commits=1, max_pending=1
attn_proj:  commits=1, max_pending=1
mlp_norm:   commits=1, max_pending=1
expert_fc1: commits=1, max_pending=1
moe_act:    commits=1, max_pending=1
```

All had `drain_firings=0` and `events_drained=0`, as expected with no inflight
cap.

Configured modules with zero commits:

- `qkv_linear`
- `core_attn`

This confirms the known AbsorbedMLA wiring gap: those flags are accepted and
baked true but no corresponding offload window is entered. The maximal legal
seven-name configuration therefore has five wired offload groups.

Telemetry-enabled result:
`results/debug0d1m-alloffload-telemetry-8192.json`.

Valve telemetry materially perturbed this tiny proxy's timing (3,703 tok/s vs
13,480 without telemetry), so its throughput result must not be used. The
functional result and cumulative commit counts are valid.

The counters are cumulative and are not cleared by the manager's per-iteration
`reset()`. A transfer-level census subsequently showed that all five commits
belonged to forced startup warmup. Post-warmup benchmark calls transferred zero
bytes. With one layer and one microbatch, each group is the last group of its
name and backward will consume it immediately, so `should_bulk_offload()`
correctly keeps it on the GPU. This proxy validates the startup transfer path,
but it does not exercise steady-state offload policy without adding another
matching layer.

## Saved-for-backward activation census

An env-gated `saved_tensors_hooks` census recorded tensors retained by autograd
for backward. Exact parameter and buffer storages are identified from the model
and reported separately. The census calls every other saved tensor an
`activation`, but that bucket also includes derived weight tensors such as
temporary float32 casts; it means non-model-storage, not necessarily
input-dependent. Storage aliases are deduplicated for the unique-storage total.

At 8,192 tokens with no recompute and no offload, one measured call retained:

| kind | save events | logical bytes | unique storage bytes |
|---|---:|---:|---:|
| activations | 576 | 16,243,682,500 (15.128 GiB) | 16,334,847,172 (15.213 GiB) |
| parameters | 1,050 | 39,383,666,688 | 19,718,852,608 |

The largest non-model-storage saves were dominated by output/loss and derived
weight tensors rather than the one transformer layer:

| shape | dtype | bytes per event | count among largest events |
|---|---|---:|---:|
| `[154880, 6144]` | float32 | 3,806,330,880 | 2 |
| `[1, 4096, 154880]` | float32 | 2,537,553,920 | 1 |
| `[1, 4095, 154880]` | float32 | 2,536,934,400 | 1 |
| `[8192, 64, 576]` | bf16 | 603,979,776 | 1 |
| `[8192, 64, 512]` | bf16 | 536,870,912 | 1 |

Artifacts:

- `results/debug0d1m-none-census-8192.json`
- `results/saved_tensor_census_nooffload/`

## Actual offload transfer census

The transfer census wraps mcore's actual `bulk_offload_group()` D2H path and
`reload()` H2D path. It reports logical tensor bytes and deduplicated pinned
host-pool storage without changing transfer behavior.

Forced 8,192-token startup warmup produced balanced transfers:

| group | events each direction | D2H bytes | H2D bytes |
|---|---:|---:|---:|
| `attn_proj` | 2 | 268,959,744 | 268,959,744 |
| `mlp_norm` | 2 | 100,696,064 | 100,696,064 |
| `moe_act` | 5 | 1,074,003,968 | 1,074,003,968 |
| `expert_fc1` | 256 | 0 | 0 |
| total | 265 | 1,443,659,776 | 1,443,659,776 |

The total is 1,376.781 MiB each direction. The 256 `expert_fc1` events are
zero-sized tensors for experts receiving no routed tokens. The unique pinned
host pool was also 1,443,659,776 bytes. This transfer volume is 8.89% of the
no-offload call's logical non-model-storage saved bytes, but that percentage is
only a size comparison: census classification and offload eligibility are
different sets.

Both subsequent benchmark calls reported zero transfer events for the
single-layer immediate-backward reason above. Artifacts:

- `results/offload_tensor_census_alloffload/`
- `results/debug0d1m-alloffload-alloc-8192.json`

The process-lifetime peak counters were 46,678,991,360 allocated and
49,920,606,208 reserved bytes for the no-offload process, versus
46,678,925,824 allocated and 50,361,008,128 reserved bytes for the offload
process. Those numbers do not isolate a per-step memory delta: startup dominates
the allocation high-water mark, and the measured post-warmup call did not
offload.

## Two-MoE-layer steady-state census

The 0D2M proxy confirms the offload policy behavior directly. Recompute was
disabled in both arms.

### No-offload saved tensors

At 8,192 tokens, a measured no-offload call retained:

| kind | save events | logical bytes | unique storage bytes |
|---|---:|---:|---:|
| non-model-storage (`activation`) | 1,142 | 19,699,035,588 (18.346 GiB) | 19,830,504,900 (18.469 GiB) |
| exact parameter storage | 2,100 | 78,811,209,728 | 39,459,643,392 |

Adding the second MoE layer increased the non-model-storage logical total by
3,455,353,088 bytes (3.218 GiB) and unique storage by 3,495,657,728 bytes
(3.256 GiB). The raw JSON retains each tensor's shape, dtype, byte count,
storage identity, unpack count, and Python stack.

Artifacts:

- `results/debug0d2m-none-census-8192.json`
- `results/saved_tensor_census_0d2m_nooffload/`

### Initial maximal-legal transfers

Forced startup warmup offloaded both layers and moved 2,988,015,616 bytes
(2,849.594 MiB) each direction. Subsequent warmup and measured benchmark calls
each moved exactly 1,443,659,776 bytes (1,376.781 MiB) each direction:

| steady-state group | events each direction | D2H bytes | H2D bytes |
|---|---:|---:|---:|
| `attn_proj` | 2 | 268,959,744 | 268,959,744 |
| `mlp_norm` | 2 | 100,696,064 | 100,696,064 |
| `moe_act` | 5 | 1,074,003,968 | 1,074,003,968 |
| `expert_fc1` | 256 | 0 | 0 |
| total | 265 | 1,443,659,776 | 1,443,659,776 |

The two steady-state calls match byte-for-byte and equal one 0D1M forced
startup transfer. This confirms that mcore offloads the first layer's eligible
tensors and retains the final layer's same-name groups for immediate backward.
The steady-state transfer is 41.78% of the additional logical saved bytes from
adding the second layer, but this remains only a size comparison because the
saved-tensor and offload sets use different classification rules.

Artifacts:

- `results/debug0d2m-alloffload-census-8192.json`
- `results/offload_tensor_census_0d2m_alloffload/`

## Implemented per-layer coverage

The initial seven-name configuration exposed three implementation gaps:

- AbsorbedMLA did not enter `qkv_linear` or `core_attn` offload windows.
- Shared experts running on the overlap stream had no offload windows.
- MoE routing, dispatch, combine, and unpermute metadata had no windows.

The implementation now adds:

- AbsorbedMLA `qkv_linear` and `core_attn` scopes.
- A multi-output commit autograd node that waits for every output branch before
  allowing backward to cross the reload barrier.
- `shared_experts`, split internally into FC1, activation, and FC2 groups on the
  shared-expert CUDA stream.
- `moe_router` and `moe_dispatcher`, with dispatcher subgroups for dispatch,
  combine, and unpermute.
- Validation that rejects the new groups under unsupported CUDA graph and
  fused grouped-shared-expert configurations.

The transfer progression for one offloaded layer was:

| implementation | steady D2H | steady H2D |
|---|---:|---:|
| Original wired groups | 1,443,659,776 (1,376.781 MiB) | 1,443,659,776 |
| + AbsorbedMLA `core_attn` | 2,679,931,136 (2,555.781 MiB) | 2,679,931,136 |
| + AbsorbedMLA QKV | 2,975,170,816 (2,837.344 MiB) | 2,975,170,816 |
| + shared-expert FC1/FC2 | 3,110,437,120 (2,966.344 MiB) | 3,110,437,120 |
| + MoE router/preprocess | 3,249,242,368 (3,098.719 MiB) | 3,249,242,368 |
| + shared-expert activation | 3,349,905,664 (3,194.719 MiB) | 3,349,905,664 |
| + dispatcher metadata, final | 3,367,239,936 (3,211.250 MiB) | 3,367,239,936 |

Final steady-state one-way payload by group:

| group | MiB |
|---|---:|
| `core_attn` | 1,179.000 |
| `moe_act` | 1,024.250 |
| `qkv_linear` | 281.562 |
| `attn_proj` | 256.500 |
| `moe_router` | 132.375 |
| `shared_expert_fc1` | 96.500 |
| `mlp_norm` | 96.031 |
| `shared_expert_act` | 96.000 |
| `shared_expert_fc2` | 32.500 |
| `moe_unpermute` | 16.031 |
| `moe_dispatch` | 0.250 |
| `moe_combine` | 0.250 |
| total | 3,211.250 |

The final fresh-process smoke matched the no-offload loss exactly:

```text
loss, no offload: 11.951191666157978
loss, final:      11.951191666157978
grad norm, no offload: 0.0007247441681101918
grad norm, final:      0.0007247457397170365
```

Process-lifetime peak allocated fell from 69,959,224,320 bytes without offload
to 66,569,079,296 bytes with final offload, a 3.157 GiB reduction. Peak reserved
fell by 0.639 GiB. These fresh-process values are directionally useful, but the
counters still include startup and allocator-history effects.

### Residual census

The final combined run nested the offload saved-tensor hook inside the census
hook. Consequently, its saved-tensor JSON is deliberately a residual inventory,
not a complete inventory: saves inside any offload window are handled by the
inner hook and do not reach the outer census.

After the final dispatcher scopes, the residual had no transformer/MoE-layer
scaled non-model storage. All remaining 12,939,361,220 unique bytes were fixed
output-side state or final normalization:

- FP32 LM-head weight materializations.
- Cross-entropy logits/softmax state.
- Chunked-head and LoRA inputs.
- Final model normalization and loss-mask state.

Those fixed costs were intentionally excluded from this optimization pass.
The ratio of final D2H logical bytes to the added-layer logical saved bytes is
97.45%, but it must not be described as memory coverage: aliases, views,
oversized backing storage, repeated saves, and intentional final-layer
residency make the counters semantically different.

Final artifacts:

- `results/debug0d2m-final-v4-smoke-8192.json`
- `results/offload_tensor_census_0d2m_final_v4/`
- `results/saved_tensor_census_0d2m_final_v3/`
- `results/debug0d2m-final-v3-combined-8192.json`

## Cumulative offload ladder

On 2026-08-22, seven fresh-process arms measured the no-recompute/no-offload
baseline followed by cumulative activation-offload groups. Every arm used the
same 0D2M snapshot, deterministic 8,192-token input, one warmup window, and
three measured windows.

| arm | peak allocated | saved vs baseline | peak reserved | median TPS/GPU |
|---|---:|---:|---:|---:|
| No recompute, no offload | 65.155 GiB | 0 | 68.223 GiB | 13,213 |
| `core_attn` | 64.003 GiB | 1.151 GiB | 67.189 GiB | 12,464 |
| + `moe_act` | 62.753 GiB | 2.402 GiB | 67.365 GiB | 10,959 |
| + `qkv_linear` | 62.583 GiB | 2.571 GiB | 68.398 GiB | 8,152 |
| + `attn_proj` | 62.333 GiB | 2.822 GiB | 67.404 GiB | 11,980 |
| + `moe_router` | 62.327 GiB | 2.828 GiB | 67.545 GiB | 11,985 |
| All supported groups | 61.997 GiB | 3.157 GiB | 67.525 GiB | 9,644 |

These are whole-process peak deltas but effectively represent one offloaded
MoE layer. The two-layer proxy offloads the earlier layer in steady state and
retains the final layer's same-name groups for immediate backward. Therefore,
do not divide the savings by two. The `core_attn` peak reduction of 1.151 GiB
equals its independently measured one-layer payload of 1,179 MiB.

The four-group arm through `attn_proj` saved 2.822 GiB per offloaded MoE layer
at a 9.3% median-TPS cost versus baseline. Adding `moe_router` changed peak
allocated by only 0.006 GiB. All groups saved 3.157 GiB at a 27.0% median-TPS
cost. Peak reserved was non-monotonic because allocator caching and process
history dominate it. Loss and gradient norm matched across all arms; throughput
was noisy in the `moe_act` and `qkv_linear` arms and is directional rather than
precise.

The complete run record is in `results/offload_ladder_20260822/`, including
raw JSON outputs, exact byte counters, all timing windows, configs, runners,
and trainer logs.

## Four-group layer scaling

The selected policy (`core_attn`, `moe_act`, `qkv_linear`, `attn_proj`) was
rerun against fresh no-offload baselines at two, four, and six cloned MoE
layers. MCore offloads N-1 layers because it retains the final layer's
same-name groups for immediate backward.

| MoE layers | offloaded layers | allocation saved | saved per offloaded layer | no-offload median TPS/GPU | offload median TPS/GPU | TPS change |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 2.821595 GiB | 2.821595 GiB | 13,203 | 12,065 | -8.6% |
| 4 | 3 | 8.464784 GiB | 2.821595 GiB | 11,722 | 7,212 | -38.5% |
| 6 | 5 | 14.107974 GiB | 2.821595 GiB | 10,034 | 6,888 | -31.4% |

The memory delta matches `3,029,664,256 * (layers - 1)` bytes exactly at all
three depths. Loss deltas stay below 2.9e-6 and gradient-norm deltas below
2.6e-7. Functionality and memory scaling therefore hold through six layers.

Throughput does not preserve the two-layer relationship: deeper offload is
roughly 31-38% slower than its same-depth baseline. This blocks a confident
performance extrapolation to 78 layers even though the memory result is ideal.
The full record is in `results/layer_scaling_20260822/`.

## Overlap trace investigation (2026-08-22, session 2)

Full findings, artifacts, and the fix live in
`results/overlap_trace_20260822/SUMMARY.md`. Short version:

Kineto traces of one steady 8,192-token window (baseline 0.724 s vs
four-group offload 1.024 s, fresh processes, HTTP
`/runtime_profile/start|stop`) show the offload penalty is **entirely GPU
idle time** — main-stream busy is identical across arms (633 vs 634 ms,
identical per phase). PCIe bandwidth is healthy in both directions
(D2H 53.2 GiB/s, H2D 51.8 GiB/s, 8.031 GiB each way = 3 layers ×
2.68 GiB exactly), so the NUMA/bandwidth hypothesis is dead. Steady state
has **zero demand misses** (telemetry: all misses date from warmup) — the
stalls are prefetch hits that arrive too late.

Three separate idle mechanisms, phase by phase (traced window):

1. Forward +82 ms: the MoE dispatcher's host readback after
   `te_moe::chunk_sort_fwd` queues on the D2H copy engine behind the
   offload stream's GiB-scale transfers (its `cudaEventSynchronize` ends
   at the exact completion timestamp of the tiny readback stuck behind
   them).
2. Mid +139 ms: a roaming 80-260 ms host stall — single
   `cudaEventQuery`/`cudaEventSynchronize`/`aten::add` calls that block
   with zero CUDA activity from the blocked thread — lands ~once per
   window at a random phase. Owner is invisible to the trace (kineto only
   records registered threads); prime suspects are allocator work on an
   untraced thread. It is offload-specific and explains the campaign's
   window-to-window variance (including the old 3,225-TPS outlier).
   Mechanism still open; expandable_segments A/B queued.
3. Backward +79 ms: reloads were issued one per group-start backward node,
   so the first H2D was CPU-issued 245 ms into a 418 ms backward, and all
   8 GiB crammed into the last 172 ms (46% overlap) while consumers
   blocked on reload events.

Fix (patch `bt_offload_backward_prefetch.patch`, Megatron-LM
`fine_grained_activation_offload.py`, both knobs default-off):

- `BT_OFFLOAD_PREFETCH_DEPTH=K` — top up H2D to K groups in flight at
  every backward group boundary (bounded early re-residency ≈ K groups).
- `BT_OFFLOAD_H2D_UNCHAINED=1` — drop `h2d_stream.wait_stream(main)`
  before reloads (ordering-only edge; lifetime safety already carried by
  the group offload event, per-stream allocator pools, the pool reuse
  event, and `record_stream` at unpack).

Traced result with K=6 + unchained: **H2D overlap 46% → 97%, uncovered
83.4 → 4.3 ms, backward wall 411 → 336 ms (baseline-equal)**. Knobs-off
arm on the patched file reproduces legacy behavior (regression guard).
Loss parity across all arms within 3e-6 with grad-norm deltas ≤ 1e-7;
two window-index-matched pairs across runs were bitwise identical.

Deciding 5-arm × 10-window A/B (`results/overlap_trace_20260822/stat_ab/`,
median window fb):

| arm | median fb | stdev | max | median TPS | peak alloc |
|---|---:|---:|---:|---:|---:|
| baseline, ES on | 0.699 s | 0.045 | 0.839 | 11,721 | 108.52 GiB |
| offload legacy, ES on | 0.923 s | 0.143 | 1.301 | 8,871 | 100.05 GiB |
| offload K6+unchained, ES on | 0.905 s | 0.883 | 3.669 | 9,115 | 100.05 GiB |
| offload K6+unchained, ES off | 0.854 s | 0.070 | 0.996 | 9,587 | 100.06 GiB |
| baseline, ES off | 0.695 s | 0.148 | 1.163 | 11,787 | 108.53 GiB |

Headline: knobs + `expandable_segments:False` cut the four-layer offload
penalty from **−24.3% to −18.1%** median TPS, collapsed the variance
(stdev 0.143 → 0.070, worst window 1.301/3.669 → 0.996 s), and preserved
the full −8.46 GiB peak-allocation saving bit-for-bit.

Full-recompute reference arm (added later same day, 10 windows,
`04-full.json`): median 10,469 TPS (−10.7% vs no-recompute), peak alloc
95.97 GiB (−12.55 GiB). **At proxy scale full recompute beats offload on
both speed and memory** — expected to invert with depth (production full
recompute costs ~35%/step at 78 layers vs 10.7% here, where four layers
of recompute are cheap next to the fixed LM-head/loss cost), but every
future proxy claim should quote all three arms. The roaming host
stall is an **expandable_segments × offload interaction** (offload arms
only; baselines ES-insensitive); it must be root-caused or ES disabled on
offload ranks before production trials. The residual −18% is
trace-attributed to (a) forward D2H exposure intrinsic to the 8k proxy's
copy-to-compute ratio (~10-15%) and (b) MoE dispatcher host readbacks
queueing behind bulk D2H on the copy engine (~5%) — both shrink at the
131k production shape. Details: `results/overlap_trace_20260822/PATCHED_RESULTS.md`.

## Memory terminology

| metric | meaning |
|---|---|
| `memory_allocated()` | current live PyTorch tensor/allocation bytes |
| `max_memory_allocated()` | high-water live PyTorch allocation bytes |
| `memory_reserved()` | current CUDA segments held by PyTorch, including cache |
| `max_memory_reserved()` | high-water PyTorch reservation; benchmark metric |
| NVML process used | PyTorch plus CUDA context, workspaces, NCCL, extensions, etc. |
| NVML device used | all processes plus device/driver overhead |
| pinned host memory | page-locked CPU memory for offload; not GPU memory |

The benchmark records both allocator peaks but does not currently reset them
per window. Fresh processes make the earlier recompute A/B's peak-reserved
comparison fair, but process-lifetime peaks cannot isolate the transient memory
effect of a specific offload window.

## Current conclusions

1. A real GLM5/AbsorbedMLA/DSA/MoE trainer can boot in approximately 70-104s on
   one B300 instead of approximately 20 minutes on the full two-node model.
2. The dense layer is unnecessary for the suspected MoE-only failure.
3. Full recompute saves 4.04 GiB peak reserved on one 8k MoE layer at a 6.8%
   throughput cost in this end-to-end proxy.
4. `moe_act` recompute is the necessary module in the observed full-scale
   `_recompute` crash under offload `[attn_proj, expert_fc1]`.
5. No recompute plus maximal legal offload boots and completes startup warmup
   with matching loss/gradients. Three groups transfer nonzero bytes;
   `expert_fc1` only sees zero-sized expert tensors in this input, and
   `attn_norm` has no eligible tensor despite its committed window.
6. Startup moves 1,443,659,776 bytes each direction. Steady-state calls move
   zero bytes because one layer gives the offload policy no earlier same-name
   group worth hiding behind compute.
7. The 0D2M proxy supplies that missing overlap window. After implementing the
   missing layer scopes, steady state reliably moves 3,367,239,936 bytes each
   direction for the earlier layer.
8. The residual census contains no repeated transformer/MoE-layer activation
   outside a hook window. LM-head/loss and final-normalization costs were left
   alone because they do not scale with the 75-layer production stack.
9. The next useful correctness experiment is the four-arm `moe_act`
   recompute/offload interaction matrix using these completed hook windows.
