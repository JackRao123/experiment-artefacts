# FSDP + CP8/EP1 debug-model result

The 1d2m proxy successfully runs with fully sharded BF16 parameters, CP8/EP1,
LoRA rank/alpha32, and full one-layer recompute at 131072 tokens.

| Measurement | Result |
|---|---:|
| Five unprofiled FB controls, mean ± sample SD | 0.6640 ± 0.0094 s |
| Five-control TPS/GPU | 24,674 |
| Additional 20-control FB mean | 0.6621 s |
| Additional 20-control TPS/GPU | 24,744 |
| Peak allocated memory | 53.394 GiB |
| Peak reserved memory | 65.125 GiB |

Three full warmups precede the five controls. One memory snapshot and eight
Kineto traces are local under cp8ep1/result. Sizes and SHA256 hashes passed
validation; the memory snapshot has 7769 history events. Do not combine these
controls with the separate matched-initialization numerical experiment.

FSDP is slower than unsharded EP1 here (27,556 TPS/GPU in its 20-control set).
It removes parameter replication, not all communication. The rank0 trace has
four expert-weight all-gather calls totaling 72 GiB of logical output tensors
and 115.1 ms summed GPU kernel duration. Non-expert parameter all-gathers add
6.64 GiB logical output and 24.3 ms summed duration. These overlap work and
include waiting, so they cannot be added directly to request wall time.
Ordinary CP communication remains. No expert dispatch/combine is introduced.

The small proxy's peak-memory saving is only 4.87 GiB: unsharded compute
buffers and activations remain even when persistent parameter state is
sharded. Full-model capacity must be measured, not inferred from a naive
eightfold reduction of the proxy's peak.

## Correctness and fixes

- The loader initializes on meta, wraps with MFSDP V1, then uses Bridge's
  existing DTensor-aware HF importer. It imported all 1074 mapped parameters.
- Backend DDP-only type checks require accepting the concrete FSDP V1 class.
  The exported FullyShardedDataParallel name is a factory, not a type.
- Core unwrap_model omitted V1's inner MegatronFSDP wrapper. That is fixed in
  Core commit 27f9fdc9d57c20c84ac298279242dae6c54a99e4 and PR #76.
- A separate matched-adapter CP8EP1 comparison accounts for every shard of
  all 38 adapter tensors. With identical BF16-representable A and zero B,
  LR=0, five controls give loss difference 0.0000139 and gradient-norm
  difference 0.0724% between DDP and FSDP. See the sibling
  glm53_1d2m_131k_fsdp_parity_20260909/comparison.json. This is a loss/norm
  check, not a full gradient-vector or save/load validation.
- Multi-adapter switching and checkpoint save/load are not validated or
  advertised. The FSDP wiring remains an explicit run-local experiment.
- GC observation is Python-only to avoid re-entering FX from a GC callback.
  No tests were added or modified, and original profiling/MFU tools remain
  unchanged.

Actual routing is very uneven: at one control, the shared-index MoE block has
63–73 empty experts and another 85–97 experts with 1–15 rows per rank.
Mean rows/expert is still 512. Isolated balanced GEMM probes are therefore
diagnostic shape comparisons, not replays of this exact routing distribution.
