# Final results

Both models: one B300 GPU, 40,960 input tokens per step, LoRA rank/alpha 32,
BF16, full decoder-block recompute. Each final run starts fresh and contains
one warmup, **10 unprofiled controls**, one memory-profiled step and one
runtime-profiled step. Final trainer step is 13.

| Statistic | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---:|---:|
| Unprofiled forward/backward TPS/GPU | 39,797.8 | 9,501.5 |
| Estimated useful-FLOP MFU | 29.33% | 25.81% |
| Forward/backward mean ± sample SD | 1.0292 ± 0.0045 s | 4.3109 ± 0.0172 s |
| Forward/backward timing CV | 0.437% | 0.399% |
| Mean optimizer request elapsed | 0.0477 s | 0.0414 s |
| TPS including optimizer requests | 38,036.6 | 9,411.1 |
| MFU including optimizer requests | 28.03% | 25.56% |
| Peak allocated during controls | 15.454 GiB | 79.467 GiB |
| Peak reserved during controls | 15.502 GiB | 79.742 GiB |
| Useful LoRA SFT GFLOP/input token | 16.5817 | 61.1103 |
| Runtime-profiled forward/backward | 1.0387 s | 4.4772 s |
| Control loss, first → last | 13.3781 → 13.0467 | 12.8247 → 12.7270 |
| Control gradient norm, first → last | 5.8335 → 4.0040 | 1.8176 → 1.5283 |

TPS uses total tokens / total elapsed across controls. MFU uses the causal
attention, frozen-base LoRA accounting documented in [README.md](README.md)
and NVIDIA's dense BF16 peak of 2.25 PFLOP/s/GPU. It excludes recomputation
from useful FLOPs. It is an analytic estimate, not a measured hardware counter.
The profile windows are excluded from headline timing. Client elapsed includes
HTTP/dispatch overhead; optimizer timings contain occasional request latency.
Synthetic repeated-token-data losses are not model-quality results.

## Local captures

| Model | Kineto runtime trace | CUDA allocator memory snapshot |
|---|---|---|
| Qwen3-0.6B | [qwen3-06b.pt.trace.json](qwen3-06b.pt.trace.json) | [qwen3-06b.memory.rank0.pickle](qwen3-06b.memory.rank0.pickle) |
| Qwen3-30B-A3B | [qwen3-30b-a3b.pt.trace.json](qwen3-30b-a3b.pt.trace.json) | [qwen3-30b-a3b.memory.rank0.pickle](qwen3-30b-a3b.memory.rank0.pickle) |

Kineto traces contain forward/backward plus optimizer for one step. Memory
history covers its own separate step, with a 5,000,000-event limit. These are
PyTorch allocator snapshots, not nsys captures or a record of non-PyTorch GPU
allocations. Raw records and calculated statistics are under `runs/`.
`artifact_manifest.json` records file sizes, SHA-256 hashes and structural
validation of both capture types.

Qwen3-0.6B remains running on devbox GPU 0 at port 8001, step 13; 30B was
stopped via the lifecycle script before the fresh 0.6B run. Stop the remaining
trainer with `bash /root/qwen06-profile/.devbox_up/stop_trainer.sh` on the devbox.
All deliverables have been copied to the Mac. The original GLM tools were
not edited; all custom scripts and data are inside this run directory.
