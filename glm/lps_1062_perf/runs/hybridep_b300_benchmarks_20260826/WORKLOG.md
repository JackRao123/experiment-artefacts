# GLM-5.2 main vs HybridEP B300 benchmarks

## Scope

- Devbox: `tj-w5y89m3`, 2 nodes x 8 B300 GPUs (`NVIDIA L20D` label).
- Main SHA: `061947aca8bc00540ee7ad9c7c3c8d7f0df68e20`.
- HybridEP PR 1150 head: `7210af32bd4945e0f7772c9196ca069e35cd1450`.
- Driver: `tools/profile_driver.py` from the experiment root, SHA-256
  `8b8097eec8e9c1901b3287c10448f77305ba72728c927d1c9d02a912b6c9e51f`.
- MFU helper: `tools/mfu.py`, SHA-256
  `4b009fffda3f6100c97ef03acd4864260221748ccda9b4b516ec37ac5383c43b`.
- Every driver invocation uses `--control-repeats 3 --runtime-profile` and
  does not use `--memory-profile`.

## Cases

| case | revision | model | topology | seq len | datums | GPUs |
|---|---|---|---|---:|---:|---:|
| 1 | main | local 0D1M debug proxy | TP1/PP1/CP16/EP16/ETP1/DP1 | 131072 | 1 | 16 |
| 2 | main | local 0D1M debug proxy | TP1/PP1/CP8/EP8/ETP1/DP1 | 131072 | 1 | 8 |
| 3 | main | `zai-org/GLM-5.2-FP8` | TP1/PP2/CP8/EP8/ETP1/DP1 | 131072 | 4 | 16 |
| 4 | PR 1150 | `zai-org/GLM-5.2-FP8` | TP1/PP2/CP8/EP8/ETP1/DP1 | 131072 | 4 | 16 |

Case 4 keeps all case 3 fields and adds only PR 1150's golden HybridEP
selection: flex dispatcher, HybridEP backend, and 16 dispatcher SMs.

## Log

20260826 13:41 PDT Confirmed both Slurm nodes are idle and each exposes eight
275040 MiB `NVIDIA L20D` devices, the devbox-up B300 identifier.

20260826 13:41 PDT Confirmed the devbox worktree is at the requested main SHA.
Fetched PR 1150 head and verified its merge base is exactly that SHA.

20260826 13:41 PDT Validated requested mesh sizes using
`PP * max(TP * CP, EP * ETP)`: 16, 8, and 16 GPUs for cases 1, 2, and 3/4,
respectively. Sequence length 131072 is divisible by `2 * CP` in every case.

20260826 13:45 PDT Case 1 launch attempt 1 failed during configuration parsing,
before model initialization. Current `TrainerControllerConfig` rejects the
obsolete `activation_offload` object copied from an older debug experiment.
Stopped the generated trainer job; all 16 GPUs returned to idle. Preserved the
full launch log as `case1_startup.log`. Removed only that unsupported field;
the requested shape and all effective settings are unchanged.

20260826 13:52 PDT Case 1 completed at exact trainer SHA
`061947aca8bc00540ee7ad9c7c3c8d7f0df68e20`. Driver arguments:
`--label main-061947ac-debug0d1m-131k-cp16-ep16-d1 --seq-len 131072
--datums 1 --num-gpus 16 --lora-rank 32 --control-repeats 3
--runtime-profile`. No memory profile. Control throughput was 5610.5, 8567.6,
and 8594.7 tok/s/GPU; arithmetic headline 7293.8 tok/s/GPU, mean FB 1.1231 s,
mean optimizer 0.0731 s. Runtime-profile FB was 0.9625 s. Driver elapsed time
was 18.1681 s. Loss and grad norm remained finite. The first control retained
a cold-start penalty, so the headline has high variance. Driver-reported MFU
66.4% and HFU 96.4% are not valid for this debug proxy because `mfu.py`
hardcodes full 78-layer GLM-5.2 FLOPs. Result: `case1_result.json`; trace:
`case1_runtime.pt.trace.json`; final trainer log: `case1_trainer.log`.

20260826 13:57 PDT Case 2 completed at exact trainer SHA
`061947aca8bc00540ee7ad9c7c3c8d7f0df68e20`. Driver arguments:
`--label main-061947ac-debug0d1m-131k-cp8-ep8-d1 --seq-len 131072
--datums 1 --num-gpus 8 --lora-rank 32 --control-repeats 3
--runtime-profile`. No memory profile. Trainer startup took 135 s. Control
throughput was 10183.1, 11341.7, and 12187.9 tok/s/GPU; arithmetic headline
11176.5 tok/s/GPU, mean FB 1.4659 s, mean optimizer 0.0368 s.
Runtime-profile FB was 1.3496 s. Driver elapsed time was 20.5517 s. Loss and
grad norm remained finite. Driver-reported MFU 101.7% and HFU 147.7% are
invalid for this debug proxy because `mfu.py` hardcodes full 78-layer GLM-5.2
FLOPs. Result: `case2_result.json`; trace: `case2_runtime.pt.trace.json`;
trainer log: `case2_trainer.log`.

20260826 14:12 PDT Case 3 completed at exact trainer SHA
`061947aca8bc00540ee7ad9c7c3c8d7f0df68e20`. Driver arguments:
`--label main-061947ac-glm52-131k-d4 --seq-len 131072 --datums 4
--num-gpus 16 --lora-rank 32 --control-repeats 3 --runtime-profile`. No
memory profile. Trainer startup took approximately 485 s. Control throughput
was 799.1, 819.8, and 820.8 tok/s/GPU; headline 813.1 tok/s/GPU, mean FB
40.2979 s, mean optimizer 0.0886 s, MFU 7.40%, HFU 10.74%.
Runtime-profile FB was 39.9707 s. Driver elapsed time was 266.0727 s (4m 26s).
Loss and grad norm remained finite. Result: `case3_result.json`; trace:
`case3_runtime.pt.trace.json`; trainer log: `case3_trainer.log`.

20260826 14:21 PDT Case 4 exact-config startup attempt 1 failed before health
during the HybridEP JIT warmup. DeepEP invoked `/bin/nvcc`; the reusable
devbox environment did not inherit PR 1150's Dockerfile settings
`CUDA_HOME=/usr/local/cuda`, `CUDA_PATH=/usr/local/cuda`, and CUDA `PATH`.
The driver did not run, so this attempt produced no driver JSON or runtime
trace. Stopped the trainer and preserved `case4_startup.log`. Verified
`/usr/local/cuda/bin/nvcc` exists on both nodes. Papercut:
`pc_3bbba585fa7c`. Relaunching the unchanged topology and trainer config with
the PR's required CUDA environment exported is an environment repair, not a
configuration change.

20260826 14:35 PDT Case 4 completed at exact PR head SHA
`7210af32bd4945e0f7772c9196ca069e35cd1450`. Driver arguments:
`--label pr1150-7210af32-hybridep-glm52-131k-d4 --seq-len 131072
--datums 4 --num-gpus 16 --lora-rank 32 --control-repeats 3
--runtime-profile`. No memory profile. Successful trainer startup took
approximately 295 s. Control throughput was 888.6, 898.1, and 906.2
tok/s/GPU; headline 897.6 tok/s/GPU, mean FB 36.5072 s, mean optimizer
0.0468 s, MFU 8.17%, HFU 11.86%. Runtime-profile FB was 36.3038 s. Driver
elapsed time was 289.1144 s (4m 49s). Loss and grad norm remained finite.
Result: `case4_result.json`; trace: `case4_runtime.pt.trace.json`; successful
trainer log: `case4_trainer.log`.

## Comparison

| case | valid tok/s/GPU | mean control FB | MFU | HFU | driver elapsed |
|---|---:|---:|---:|---:|---:|
| 1 main debug CP16/EP16 | 7293.8 | 1.1231 s | N/A | N/A | 18.17 s |
| 2 main debug CP8/EP8 | 11176.5 | 1.4659 s | N/A | N/A | 20.55 s |
| 3 main full model | 813.1 | 40.2979 s | 7.40% | 10.74% | 266.07 s |
| 4 PR 1150 HybridEP full model | 897.6 | 36.5072 s | 8.17% | 11.86% | 289.11 s |

HybridEP improved full-model per-GPU throughput by 10.4% and reduced mean
control FB time by 9.4% relative to main. Total driver elapsed was longer for
case 4 because its warmup took 124.0 s versus 84.6 s on main; the three
steady-state controls were consistently faster.

The debug-model MFU/HFU fields in the raw case 1/2 driver JSONs are deliberately
reported as N/A here. The driver executed correctly, but `mfu.py` models the
full 78-layer GLM-5.2 architecture and therefore cannot produce meaningful
utilization percentages for the one-layer proxy.

20260826 14:36 PDT Final verification complete. All four driver result JSONs
parse successfully. Local runtime trace sizes exactly match the sizes returned
by the driver: case 1 8075927 bytes, case 2 6234978 bytes, case 3 735950130
bytes, case 4 700203252 bytes. Recorded result and trace hashes in
`SHA256SUMS`. Stopped the generated trainer allocation; Slurm has no queued or
running jobs and neither node reports a GPU compute process. The remote
worktree remains detached at exact PR head
`7210af32bd4945e0f7772c9196ca069e35cd1450`.
