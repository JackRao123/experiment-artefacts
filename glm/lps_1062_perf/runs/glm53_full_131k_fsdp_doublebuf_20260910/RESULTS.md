# Full GLM-5.3 FSDP: persistent-buffer result

CP8EP1 is complete. CP4EP1 with lookahead OOMed; a separate no-lookahead
variant is being checked. BF16 expert storage, eight HGX
B300 GPUs, LoRA rank/alpha32, 131072 tokens per datum, full one-layer recompute.
Three warmups, five unprofiled controls, separate memory and all-rank runtime
captures. Standard TE expert GEMMs; no full-iteration CUDA graphs.

| Full-model CP8EP1 | Dynamic buffers | Persistent max-pool buffers |
|---|---:|---:|
| FB mean ± sample SD | 84.200 ± 4.501 s | 11.714 ± 0.244 s |
| TPS/GPU | 194.58 | 1398.64 |
| Peak allocated GiB | 239.132 | 248.103 |
| Peak reserved GiB | 262.795 | 252.785 |
| Rank0 memory-profile segment-map GiB | 93.346 | 0.117 |
| Rank0 memory-profile segment-unmap GiB | 92.193 | 0 |

The full-step speedup is 7.19x. Persistent buffers use about 9 GiB more live
memory but about 10 GiB less reserved memory, and avoid the large mapping /
unmapping cycle. The runtime pool policy is the material training change;
startup changes are separately recorded and do not alter the intended BF16
training arithmetic.

Across all 40 rank-controls in the persistent-buffer run: zero allocation
retries, zero OOMs, zero device frees, and zero allocator-wide stream
synchronizations. There were ten device allocations in total, so this is not
a claim of literally zero allocations. The separate memory-profile step
mapped one additional 120-MiB region and unmapped nothing.

## Communication is still present

EP1 has no expert dispatch/combine. FSDP still gathers weights: the rank0
runtime trace has 150 expert-weight all-gathers with 2700 GiB of logical
output tensors, totaling 4.450 seconds of GPU kernel duration. Non-expert
parameter all-gathers total 1.544 seconds, CP all-gather 0.429 seconds and CP
reduce-scatter 0.169 seconds. These sums overlap computation and include peer
waits/profiling overhead; they are not additive exposed communication costs.

## Startup and correctness

The explicit >50%-memory GC/cache-flush loop and repeated root-parameter
counting were fixed in Core. The loader now uses bounded reader caching,
indexed exact lookup and GPU load-time dequantization, scoped to the immutable
HF snapshot. Sampled real weights were bitwise identical to the original
loader, and original methods were restored after the scope.

Full-model checkpoint import dropped from 1108–1176 seconds per rank to
approximately 198–201 seconds. Each rank fetched 117060 tensors with 153 shard
opens. No per-forward weight dequantization was introduced: storage stays BF16.

The preceding matched-adapter debug run compared DDP and FSDP loss/gradient
norms; it was not a full gradient-vector or save/load validation. This remains
an experimental, single-adapter path. No tests or original profiling/MFU tools
were changed.

Artifacts: cp8ep1/result/benchmark.json, summary.json, timings/, runtime/ and
memory/. source-pins.json records exact code/dependency fingerprints.
validate_results.py checks protocol, hashes, snapshots and allocator counters.
The CP4 attempt uses two real datums for its two DP replicas and also contains
the empty-DP phantom-partition alignment fix; its source pins are separate.
That attempt failed in FP32 LM-head logit addition on the first max-length
warmup: 2.36 GiB requested, 1.16 GiB free, 260.75 GiB live. It produced no valid
controls or normal profiles. See cp4ep1/FAILED.md. This is a capacity boundary
for this lookahead strategy, not a universal claim about all CP4 configurations.
