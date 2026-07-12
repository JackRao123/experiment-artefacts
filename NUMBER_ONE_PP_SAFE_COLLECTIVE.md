# PP-Safe LoRA Adapter Export (Nemotron-3-Ultra, PP=4)

## TL;DR

When the trainer runs with **pipeline parallelism > 1** (Nemotron-3-Ultra LoRA
SFT uses `TP=8, PP=4, EP=8`, world=32 on 4×8 B200), the **upstream
Megatron-Bridge LoRA adapter export path deadlocks**. The fix in
`baseten-weight-sync/baseten_weight_sync/_bridge_patches.py`
(`_pp_safe_materialize_adapter_weight` + `_pp_safe_megatron_to_hf_tensor`)
replaces the upstream "PP-broadcast-then-TP-gather" materialization with a
"TP-gather-on-owning-stage → CPU → object-gather-across-PP" materialization,
which completes cleanly.

This was verified empirically with an A/B test on the devbox (`w57o7m3`), same
loaded model, single boot, toggling only the materialization call:

| | WITH fix (pp-safe) | WITHOUT fix (upstream) |
|---|---|---|
| Outcome | completed | **hung / deadlocked** |
| Wall time | 23.8s (server 23.71s) | 600s client timeout, never finished |
| Adapter output | valid `adapter_model.safetensors` (179 MB) + `adapter_config.json` | no output dir created |
| Trainer after | healthy, kept serving | wedged; required `SIGKILL` + `scancel --signal=KILL` |

**Conclusion: the fix is load-bearing for sampler weight sync on Ultra, not a
performance nicety.**

---

## Background

### Where this code sits

`/save_weights_for_sampler` publishes the LoRA adapter (PEFT format) so an
inference sampler can hot-load it. The call chain:

```
POST /save_weights_for_sampler            (rank 0, dispatched to ALL ranks)
  -> MegatronBridgeController.execute_save_weights_for_sampler
    -> _write_lora_adapter
      -> _apply_bridge_patches()          # installs the monkeypatch (idempotent)
      -> model_bridge.stream_adapter_weights_megatron_to_hf(...)   # COLLECTIVE
      -> WeightSyncClient.publish_lora_adapter(...)                # rank 0 writes
```

`stream_adapter_weights_megatron_to_hf` is a **collective**: it runs on every
rank because the op is broadcast to all ranks by the dispatcher, and the
adapter tensors are sharded across TP/PP.

### Why PP makes this hard

With `PP=4`, the model is split into 4 pipeline stages. A given LoRA adapter
tensor (e.g. `decoder.layers.10.self_attention.linear_qkv` LoRA) physically
exists **only on the pipeline stage that owns layer 10**. The other 3 stages do
not hold that parameter. To produce one HF/PEFT tensor you must:

1. reconstruct the full tensor from its TP shards (TP gather, within the owning
   stage), and
2. make the result available where it gets written (rank 0).

The order in which you do the PP and TP communication, and which ranks
participate, is where upstream and the fix diverge.

---

## The two implementations

### Upstream (the path that hangs)

`MegatronPeftBridge.materialize_adapter_weights` (non-expert branch) calls
`mapping.megatron_to_hf(...)`, which for Column/Row/Replicated mappings does:

```text
broadcast_from_pp_rank(tensor)   # PP first: make the tensor exist on ALL pp stages
gather_from_tp_ranks(tensor)     # then TP gather on every stage
torch.cat(...)                   # reconstruct full tensor
```

`broadcast_from_pp_rank` (in
`server/vendor/megatron-bridge/src/megatron/bridge/models/conversion/param_mapping.py`)
is itself coordinated: it `all_gather_object`s a per-PP-rank "tensor spec",
picks the lowest owning PP rank as the source, allocates an empty tensor on
non-owning stages, and `torch.distributed.broadcast`s. On paper this looks
collective-safe (every rank participates, source is discovered, not assumed).

In practice, for the NemotronH hybrid at `PP=4`, this path **deadlocks** (see
results below). The exact upstream defect was not root-caused line-by-line, but
the symptom is a collective deadlock during adapter streaming — i.e. ranks stop
agreeing on the broadcast/gather sequence and block forever. Candidate causes
(not confirmed): the `cache_key=str(self.hf_param)` spec cache interacting with
the hybrid layer pattern, or per-tensor PP broadcasts being entered in an order
that differs across stages once the hybrid (Mamba/attention/MoE) layer mix is
involved.

### The fix (the path that works)

`_pp_safe_materialize_adapter_weight` → `_pp_safe_megatron_to_hf_tensor` invert
the order and avoid the PP broadcast of GPU tensors:

```text
on the OWNING pp stage only:
    gather_from_tp_ranks(tensor)   # TP gather
    torch.cat(...)                 # reconstruct full tensor
    .detach().cpu().clone()        # move to CPU
all pp stages:
    all_gather_object(cpu_tensor_or_None, group=pp_group)   # object gather
    return the one non-None entry
```

Only TP-rank-0 of the owning PP stage produces a materialized tensor; all other
ranks contribute `None` but still participate in the `all_gather_object` so the
collective is symmetric. No empty GPU tensor is allocated on non-owning stages,
and no GPU `broadcast` of the weight occurs.

Grouped routed-expert adapters (`requires_expert_splits=True`) still fall back
to the upstream `materialize_adapter_weights`. This is acceptable **only because
the Nemotron LoRA target set deliberately excludes routed experts** (see
`server/src/trainers_server/dp_worker/api/_lora_targets.py`: attention qkv/proj,
Mamba mixer in/out, and **shared** experts are targeted; routed
`mlp.experts.*` are not). If routed-expert LoRA is ever enabled, that fallback
would hit the same upstream path and would need its own PP-safe handling.

---

## Experiment

### Setup

- Devbox: `tj-trainer-w57o7m3-4x8xB200`, 4 nodes × 8× B200, `world_size=32`.
- Branch: `jack-nemo3ultra-256k` @ `8f2fa4c0`, submodules synced on all nodes.
- Config: Nemotron-3-Ultra, `TP=8 PP=4 EP=8 ETP=1`, `lora_rank=16`,
  `max_seq_len=16384`, `weight_sync: {"type":"local","path": <shared dir>}`.
  (seq len is irrelevant to export; LoRA adapter shapes are independent of it.)
- Weights loaded from the on-box HF cache (`HF_HOME`, `HF_HUB_OFFLINE=1`).

### Method (clean isolation of the single variable)

Instead of two boots, the bridge file was temporarily instrumented on all 4
nodes (md5-verified identical) so a **single boot** could export both ways:

1. A runtime toggle file gated the materialization call:
   - toggle **absent** → use `_pp_safe_materialize_adapter_weight` (WITH fix)
   - toggle **present** → use upstream `materialize_adapter_weights` (WITHOUT fix)
   Everything else in the streamer (task building, per-expert formatting, etc.)
   was byte-for-byte identical between the two runs. This isolates exactly the
   one line under test, rather than comparing against a fully-unpatched bridge.
2. Per-rank instrumentation: at export start each rank did
   `reset_peak_memory_stats()`; at export end each rank wrote
   `max_memory_allocated()` to a per-rank file. (Reaching the end-of-export
   write is itself a signal that the rank's export completed.)
3. WITH fix was run **first** (known-good baseline), WITHOUT fix **second** (in
   case it wedged the trainer — which it did).
4. Export was driven via the real async-op endpoint (`POST
   /save_weights_for_sampler` → poll `/operations/{id}`), client-timed, 600s cap.

All instrumentation was reverted afterward and the original file restored on all
4 nodes (md5 back to the pre-experiment value); temp scripts deleted.

---

## Results

### WITH fix (pp-safe ON)

- Client wall time: **23.77s**; server dispatcher duration: **23.71s**.
- Result: `{"version": 0, "path": ".../wsync/sampler_weights/ab_ppsafe"}`.
- Output: `adapter_model.safetensors` (**179,957,064 bytes**) +
  `adapter_config.json` (694 bytes).
- All 32 ranks logged `save_weights_for_sampler done`; rank-0 logged
  `write_lora_adapter done`.

Per-rank export peak `max_memory_allocated` (bytes), post-reset:

```
rank 0  = 35,293,349,376     <- TP0 of PP stage 0 (does the gather)
rank 8  = 35,024,913,920     <- TP0 of PP stage 1
rank 16 = 35,024,913,920     <- TP0 of PP stage 2
rank 24 = 38,431,846,912     <- TP0 of PP stage 3
all other 28 ranks = 2,560   <- contribute None, ~zero extra allocation
```

Interpretation: only the **4 TP-rank-0 ranks** (one per pipeline stage) do real
work; the value there is dominated by the already-resident model shard (peak is
reported relative to current allocation), and the export itself adds little. The
other **28 ranks allocate ~nothing** during export. This is the intended,
minimal-footprint behavior.

### WITHOUT fix (pp-safe OFF / upstream)

- Client: **CLIENT_TIMEOUT/HANG after 600.45s** — never completed.
- Server log: op submitted (`202 Accepted`), then an unbroken stream of `408
  Request Timeout` on `GET /operations/{id}` for the full 600s. The op never
  transitioned to `done` or `error`.
- Per-rank peak files: **none written** for the `nopp` run — meaning **no rank
  ever reached the end of the export**. Every rank was blocked mid-stream inside
  the collective. This is a **true deadlock**, not slowness.
- Output: **no adapter directory created**.
- Process state: torchrun workers stayed **alive** (state `Sl`/`Ssl`) but
  wedged; the Slurm job then sat in `CG` (completing) and required
  `scancel --signal=KILL` + `pkill -9` to reap. A deadlocked NCCL collective
  does not tear down on plain `SIGTERM`.

### Side-by-side

| Metric | WITH fix | WITHOUT fix |
|---|---|---|
| Completes? | yes | no (deadlock) |
| Time | 23.8s | ∞ (killed at 600s) |
| Adapter written | 179 MB safetensors + config | none |
| Ranks finishing export | 32/32 | 0/32 |
| Extra GPU mem on non-owning ranks | ~2.5 KB | n/a (never finished) |
| Recovery | none needed | force-kill whole job |

Note: without-fix **memory** numbers are unobtainable because the run never
completed; the decisive difference is *completes vs. hangs*, which dominates any
memory consideration.

---

## What this changes about the earlier assessment

An earlier static read of the vendored Megatron-Bridge suggested the upstream
path looked "collective-order safe" (it discovers the source PP rank via
`all_gather_object` before broadcasting), so the fix might be only defensive or
a memory optimization. **The A/B disproves that.** Upstream hard-hangs on
NemotronH `PP=4` LoRA export. Keep the fix.

### Recommended follow-ups

1. **Keep the fix** — it is required for `/save_weights_for_sampler` on any
   PP>1 NemotronH (Ultra) deployment.
2. **Tighten the code comment** in `_bridge_patches.py`. The current wording
   ("can enter different collectives on different PP stages") is a hypothesis;
   replace with the reproduced fact: *"Upstream `materialize_adapter_weights`
   (PP-broadcast-then-TP-gather) deadlocks for NemotronH at PP>1 LoRA export
   (reproduced on 4×8 B200, Ultra TP8/PP4/EP8: upstream hangs >600s, pp-safe
   completes in ~24s)."*
3. **Add a guard/regression signal.** A multinode export is expensive to test in
   CI, but consider at least a unit/integration assertion that the PP-safe path
   is the one installed for PP>1, so a future refactor can't silently revert to
   the hanging path.
4. **Routed-expert LoRA is still unsafe.** `requires_expert_splits=True` falls
   back to upstream. Fine today (routed experts are excluded from LoRA targets),
   but document it as a known limitation; enabling routed-expert LoRA on Ultra
   would reintroduce the deadlock and needs a PP-safe grouped-expert path.

---

## Reproduction notes

- The earlier SFT smoke test used `weight_sync: disabled`, which **never invokes
  this code path**, so it could not have caught this. Exercising the deadlock
  requires `weight_sync` enabled (`local` is sufficient) + `lora_rank` set +
  `PP>1`, then calling `/save_weights_for_sampler`.
- `/b10/workspace` is **node-local** on these devboxes, so any code edit for an
  experiment like this must be applied to **all** nodes (verified via md5),
  not just rank 0.
- A deadlocked run leaves the Slurm allocation in `CG`; use
  `scancel --signal=KILL <job>` and `pkill -9 -f trainers_server.dp_worker.main`
  to fully clear it before the next boot.
