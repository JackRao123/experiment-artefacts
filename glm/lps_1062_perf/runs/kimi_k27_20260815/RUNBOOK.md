# Kimi-K2.7-Code B300 profile run — 2026-08-15 (jack/tao)

Goal: run `tools/profile_driver_new.py` (the "new" profiling driver) against a
Kimi-K2.7-Code trainer in its golden B300 shape, producing a kineto trace +
steady-state timing directly comparable to the GLM-5.2 LPS-1062 anchors, then
trace-compare the two models.

## Comparison point

GLM-5.2 anchor (round-3, `A131-131k-d4-r3anchor-318g61w`): 2x8 B300 (ali),
TP1/PP1/EP16/CP16, 131k x d4 = 524,288 tok/step, LoRA r32:
**~702-707 tok/s/GPU steady (46.4-46.7 s/step), MFU 6.1%, peak ~201 GiB/GPU.**

Kimi-K2.7-Code golden B300 (trainer_configs.py): TP8/PP1/EP16/ETP1/CP1, flash,
trust_remote_code, 256k-capable. Same box shape (2x8 B300 ali), same driver,
same 131k x d4 operating point, LoRA r32.

Arch deltas that matter for the trace read (HF configs, 2026-08-15):

| | GLM-5.2 | Kimi-K2.7-Code |
|---|---|---|
| layers | 78 (3 dense) | 61 (1 dense) |
| hidden | 6144 | 7168 |
| experts | 256 top-8 (+1 shared) | 384 top-8 (+1 shared) |
| attn | MLA + **DSA** (top-2048, length-indep) | MLA **full causal** (L/2) |
| vocab | 154880 | 163840 |
| fwd GF/token @131k | 102.7 | 227.3 (attn 163.7 of it) |
| useful GF/token @131k (LoRA) | 227.6 | 618.4 |
| quant | FP8 blockwise -> bf16 on load | int4 pack-quant experts -> bf16 on load |
| LoRA targets | q_down/q_up/kv_down/o + dense+shared MLP + head | q_down/kv_down/o + head |

## Box

- job `w7982d3` (devbox-up 16 b300 ali, 2026-08-15 ~14:00 PDT)
- leader: `ssh tj-w7982d3`

## Launch env (mirrors GLM ship env minus the GLM-only TF32 head patch)

```
NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1
NCCL_NCHANNELS_PER_NET_PEER=8
BT_SKIP_WARMUP unset (default 1-datum boot warmup)
```

NOT set: BT_TF32_LM_HEAD (GLM branch-only patch, absent on main);
BT_PROFILE_RANKS (branch-only; main hardcodes rank_set={0} -> rank-0 trace,
same as the canonical GLM traces).

## Kit (this folder)

- `kit/trainer-config.json` — Kimi golden B300 shape, max_seq_len 131072
- `kit/trainer-server-config.json`
- `kit/mfu.py` — Kimi arch constants (driver imports `mfu` by module name)
- `kit/profile_driver_kimi.py` — driver copy, VOCAB_SIZE=163840

## Driver invocation (on leader, kit as cwd)

```
python3 profile_driver_kimi.py --label kimi27-131k-d4 --seq-len 131072 \
  --datums 4 --num-gpus 16 --lora-rank 32 --control-repeats 2
```

Outputs: `/root/.cache/user_artifacts/lps1062_bench/kimi27-131k-d4.json`;
trace: node-0 `/tmp/checkpoints/profiles/torch_trace/*.pt.trace.json`
(copy to shared FS + pull Mac-side immediately; node-local /tmp is ephemeral).

## RESULT (done 2026-08-15, box w7982d3)

- **641 tok/s/GPU** (controls 638/645, step 51.1s), mfu3x 15.9%, hfu 21.7%,
  peak 170 GiB. Traced step 52.8s (+3.3% kineto).
- vs GLM-5.2 r3anchor (~705 tok/s/GPU): **GLM ~10% faster; Kimi ~2.6x MFU.**
- Full trace comparison + abnormal-area analysis: `COMPARE.md`.
- Gotcha hit: devbox-up died at step 8 (curl 92 on the /root/trainers clone);
  resumed per the devbox-up-cli memory (devbox_resume.py). Also: the trainer
  HTTP :8001 landed on **tj-w7982d3-1** this time — rank 0 follows *Slurm
  hostname order*, not the k8s pod index; the ssh alias order can be inverted.
- Tree: shared trainers_main flipped 73c24b00 (campaign) -> clean main @
  29b59564 for this run; campaign local mods archived to
  `user_artifacts/kimi_kit/*localmods.patch` on the box.
