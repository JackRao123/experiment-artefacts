# PR #829 description update (draft — production validation section pending)

## Summary

Pin only the GLM-5.2 B300 trainer config to
`baseten/trainers-server:trainer-cuda13-sm103-0e0b65a`. The shared B300 image
pin remains unchanged for every other model.

## Why (updated with causal evidence)

The previous GLM B300 image, `trainer-cuda13-sm103-5a4ae4d`, ships pristine
`nvidia-cudnn-frontend==1.26.0`, whose CuTe-DSL indexer top-k kernel has the
OOB-lane candidate-flood defect fixed by #814 (`1.26.0+dsatopk1`, carried in
`98cd395`; `0e0b65a` is a config-only repin on top).

Incident `rwn24dw` (Loops run `owp7glw`, 2026-07-28 21:38:37Z and 22:00:36Z)
is this defect class, established by a controlled dev-box A/B (below) plus a
fleet fingerprint: the same customer org hit >=4 identical crashes in ~26h
across >=3 deployments (`dq48213` 07-27 22:02, `4w79o03` 07-28 00:35,
`rwn24dw` x2), on different nodes and GPUs, all async IMAs surfacing via NCCL
watchdog / RoPE sync during GLM-5.2 LoRA long-sequence forward_backward. Host
dmesg on two of those nodes shows Xid 13 "Out Of Range Address" + "Multiple
Warp Errors" + Xid 43 with byte-identical ESR words (0x1f81fb60/0x1174) — the
SM/shared-memory out-of-range signature of the top-k candidate flood (the #814
analysis measured 26 concurrent invalid __shared__ writes). The alternative
backward index-value faults produce Xid 31 MMU faults instead (measured
empirically on B300 via the #821 regression cases), excluding them for this
incident class.

## Dev-box exact A/B (2026-07-29, devbox qexzp23 = 2x8 B300 on ali; one node,
b300-1-ana8db87-0004, is itself an incident node)

Environment: trainers @ `5a4ae4d1d`, `make server-venv` (torch 2.11.0+cu130,
CUDA 13.0, driver 580.105.08); pristine top-k kernel md5 `b74f18f1...` ==
incident image. Frozen artifact `topk_call_41.pt` sha256
`8aaa196939c70d9ce36879d25b1c0a88a011692d090b28a3c65785ba206668f9`; synthetic
artifact `server/scripts/repro_cudnn_dsa_indexer_topk_oob.py` (no customer
data). Arms = one patch each via PYTHONPATH shadow (tree md5s logged).

| arm (single patch)        | frozen artifact | synthetic | result |
|---------------------------|-----------------|-----------|--------|
| pristine 1.26.0           | 3 runs (1 cold-JIT) | 2 runs | 5/5 IMA |
| #396 TMEM WAR (bwd)       | 2               | 1         | 3/3 IMA |
| #439 empty-topk-row (bwd) | 2               | 1         | 3/3 IMA |
| index-bounds hardening    | 2               | 1         | 3/3 IMA |
| **#814 top-k OOB (dsatopk1)** | **10 (2 cold-JIT)** | **1** | **11/11 PASS**, outputs structurally valid (indices in-bounds) |

Only the #814 patch flips the outcome; the three #821 backward patches
individually do not (they remain valid defense-in-depth for kernel-level
fault modes proven separately: oob_hi/oob_neg/empty_row all IMA pristine,
pass on dsatopk2 per #821).

Faithful workload replays on the exact incident topology (CP16/EP16, 2x8
B300, seed 1234, LoRA r32/a32, historical op shapes and pipelining) ran 16
cycles clean on the unpatched baseline, and 35k+ instrumented top-k calls
show synthetic token data cannot reach the fp16-collapse flood precondition
(extreme real score -337 vs the -65504 threshold) — consistent with the
incident's data-dependence and non-deterministic identical-bytes retries;
exact customer payloads are not recoverable (documented in the audit dir).

## Production validation

(existing section retained: release build, owp2dzq/q0pe7r1 two-node B300
validation with sampler, dsatopk1 wheel verified in rank 0, 84,173-token
regression run)

**Fixed-image frozen-workload validation (2026-07-29): PENDING — will append
project/job IDs, pod image digest, in-pod wheel verification, frozen replay +
reduced-reproducer results.**

## Audit trail

`experiment_artefacts/glm/dsa_topk_ima/b300_rwn24dw/` (RUNBOOK.md, ab/results.tsv,
env manifests, probe logs, crash forensics).
