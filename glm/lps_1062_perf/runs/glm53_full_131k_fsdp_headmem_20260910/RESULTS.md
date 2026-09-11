# Full GLM-5.3 CP4: lower-memory FP32 head

COMPLETE. Eight HGX B300 GPUs, BF16 FSDP CP4EP1/DP2, two 131072-token datums,
LoRA rank/alpha32, full one-layer recompute, persistent buffers and no parameter
lookahead. Opt-in functional FP32 head fusion and 2048-token head chunks.
Three warmups, five unprofiled controls, one memory profile and eight Kineto traces.

| Metric | Result |
|---|---:|
| FB mean ± sample SD | 21.5274 ± 0.3600 s |
| FB range | 21.1567–21.9681 s |
| TPS/GPU | 1522.15 |
| Peak allocated | 252.642 GiB |
| Peak reserved | 258.383 GiB |
| Allocation retries / OOMs over 40 rank-controls | 0 / 0 |
| Device frees / allocator-wide stream syncs | 0 / 0 |
| Device allocations over 40 rank-controls | 12 |

The rank0 memory-profile window mapped 0.527 GiB and unmapped nothing. Small
new allocations remain; the large cyclic mapping/unmapping and retry behavior
seen in the earlier runs is absent from the measured controls.

Compared with CP4 without the head-memory changes, TPS/GPU improved from
1240.06 to 1522.15 and live peak fell from 256.81 to 252.64 GiB. That earlier
case had sixteen allocation retries and is retained as diagnostic evidence.

The earlier CP8 prefetched persistent-buffer recipe measured 1398.64 TPS/GPU,
11.714 seconds for one datum, and 248.10 GiB live peak. This CP4 recipe is 8.8%
more efficient per GPU in these five-control measurements, but takes longer
per request because it processes two datums. This is a recipe comparison,
not an isolated CP-only ablation: lookahead and head settings differ.

CP2 without lookahead failed a 6-GiB expert-output allocation before reaching
the LM head. Its live-state OOM snapshots are in the noprefetch sibling run.
It was not rerun with head fusion because that failure occurs earlier in the
model. CP4 is the lowest CP successfully measured in this investigation.

## Numerical and artifact checks

The functional head fusion and gradient split passed bitwise output and
hidden/LoRA-gradient checks at representative BF16/FP16 shapes, including the
full 4096x6144x154880 head probe. The residual split explicitly preserves
intermediate precision casts. Head chunking remains FP32 and does not shorten
the model sequences; floating-point accumulation order can differ with chunks.

validate_results.py verified all artifact sizes/hashes, the protocol and the
40 rank-controls. Runtime/memory files are under cp4ep1/result; exact source
fingerprints are in cp4ep1/source-pins.json. No tests or original profiling/MFU
tools changed. This remains a draft, experimental single-adapter FSDP path;
save/load and multi-adapter behavior were not validated.
