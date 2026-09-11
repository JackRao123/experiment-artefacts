# Matched LoRA initialization and grouped-MM prototype

All runs use CP8EP1, BF16, 1d2m, 131072 tokens, full recompute, and identical
constant BF16-representable LoRA A / zero B with LR=0. Three warmups and five
controls. These are numerical/debug populations, not the primary performance
results with the normal adapter initialization.

| Backend | Mean loss | Mean gradient norm | HTTP FB TPS/GPU | Backend FB+optim s |
|---|---:|---:|---:|---:|
| DDP / standard TE | 13.55346396 | 0.17985440 | 25,952 | 0.53127 |
| MFSDP / standard TE | 13.55345004 | 0.17998463 | 25,218 | 0.56234 |
| MFSDP / zero-copy grouped-MM | 13.55345958 | 0.17997800 | 25,367 | 0.55790 |

All 38 adapter tensor shapes/values match and every FSDP shard is accounted
for. DDP versus MFSDP has loss difference 0.0000139 and gradient-norm relative
difference 0.0724%. MFSDP standard versus grouped-MM has loss difference
0.00000954 and norm difference 0.00368%. The check passes tolerances of
0.001 absolute loss and 0.5% relative norm. It does not compare full gradient
vectors and is not a substitute for production save/load or multi-adapter tests.

The FSDP layout audit reports contiguous expert storage for every captured
projection. The prototype creates an as_strided view without copying weights,
uses grouped_mm for forward/input gradients, and rebuilds the view from current
unsharded parameters during backward to avoid retaining a stale FSDP buffer.
It is limited to frozen BF16, TP1, full-recompute expert projections.

The prototype's approximately 0.6% HTTP / 0.8% backend improvement is too small
and noisy to call an end-to-end win. Standard TE is retained for the full-model
FSDP capacity run. No tests were added or changed. The prototype remains an
experiment-only script, not a production default.
