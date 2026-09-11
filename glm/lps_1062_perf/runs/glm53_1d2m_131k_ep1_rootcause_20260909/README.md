# GLM-5.3 EP1 root cause and FSDP investigation

IN PROGRESS. Sequence length 131072. Preserve the previously agreed 1d2m
proxy (dense/indexer, MoE/shared index, MoE/indexer) for comparison, rather
than changing to the earlier proposed 2d2m midway through the experiment.

Starting point: BF16 expert storage, exact previous trainer source
c9a723bf431621ca05580726f4acd01ec618326a, Core 3b893e39ead7113d897e86105505f6fd3e49a9f1.
First change: remove the identity source-rank/expert chunk transpose when
EP=ETP=1. Ordinary token-to-expert permutation and its inverse remain.
No Transformer Engine edits: removing redundant work also avoids its
incomplete chunk-sort autotune key on this singleton path.

Protocol: three complete warmups, five unprofiled controls, one memory
snapshot, one all-rank Kineto capture. Additional same-count stability
controls for all topologies. Headline timing excludes profiling and
optimizer; CUDA-event sublayer instrumentation is retained in all cases.
Original tools/profile_driver.py and tools/mfu.py remain pristine.
No tests are added or modified; numerical probes are experiment artifacts.

Remote host tj-32vj99q; run /root/glm53-131k-ep1-rootcause-20260909.
Use run-local copies of devbox-up-generated start/wait/stop scripts.
Do not shut down the devbox without preserving node-local source/model files.

Next: account for complete-step and per-sublayer CP/EP costs, then attempt
parameter-sharded FSDP+CP8/EP1 on the proxy before the full model. An EP1
communication saving is not assumed to imply a particular TPS improvement.
