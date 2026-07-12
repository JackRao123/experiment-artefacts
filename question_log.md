# Question log (answer in the morning)

Working autonomously overnight on the 131k Nemotron-3-Super PP=4 run on the new
4-node box `tj-trainer-w6llv5w`. Below are decisions where I made a reasonable
assumption rather than blocking. Please confirm/correct.

## Decisions made (assumptions)

1. **Parallelism layout: TP=8, PP=4, CP=1, EP=8, ETP=1** (world=32, DP=1).
   - PP=4 because the model has 88 layers; 88 is divisible by 4 (22 layers/stage)
     but NOT by 3, so PP=3 on 24 GPUs is impossible without an uneven pipe-split
     pattern. 4 nodes makes PP=4 clean.
   - EP=8 because each PP stage has TP×DP = 8 GPUs (EP can't exceed that).
   - Assumption: you're OK with EP=8 (not 16); EP=16 isn't possible at PP=4/DP=1.

2. **Attention backend:** kept my conditional override (cuDNN `fused` only when
   CP>1; native FA4 `flash` at CP=1). Confirmed FA4 (`flash_attn_4-4.0.0b11`) IS
   installed on this box, so flash works at CP=1. Did NOT take nemo3ultra's
   unconditional `AttnBackend.auto` override.

3. **Weights:** downloaded `NVIDIA-Nemotron-3-Super-120B-A12B-BF16` ONCE into the
   shared `/root/.cache/user_artifacts/huggingface` (confirmed shared across all
   4 nodes). Reused the HF token from the old box's `super_env.sh`.

4. **HF cache at launch:** will set `HF_HOME=/root/.cache/user_artifacts/huggingface`
   in the trainer launch env so all ranks load weights from the shared cache
   (this box injects no HF_HOME by default).

5. **Repo:** all 4 nodes cloned at branch `jack-nemo3super-131k` (commit with the
   PP>1 fixes), `/root/trainers` symlinked, `uv sync --extra worker --extra dev`
   done. `mamba-ssm` prebuilt-wheel fetch hit intermittent DNS failures on 2
   nodes; fixed with `MAMBA_FORCE_BUILD=TRUE` (source build).

6. **lora_rank=16, micro_batch=1, full activation recompute** — same as the
   profiling configs. Assumption: rank 16 is fine for the golden config.

## OUTCOME (validated)

**131k LoRA SFT works at TP=8/PP=4/CP=1/EP=8 on 4 nodes, peak 100.79 GiB/GPU,
no OOM, loss decreasing.** Golden row added, fixes committed, **PR opened:
https://github.com/basetenlabs/trainers/pull/437** (into `jack-nemo3super`).

All assumptions in the section above were validated by the live run. EP=8 was
required (not a free choice) — PP=4/DP=1 gives 8 GPUs/stage, so EP can't exceed 8.

## Open questions / follow-ups for you

1. **Real-dataset run: NOT NEEDED (resolved 2026-06-19).** You confirmed
   synthetic data is sufficient — no `ChatQA2`/`NarrativeQA` run required. The
   synthetic 131k path exercises the identical activation-memory, PP pipeline,
   and gradient path, so the config is fully validated for "works and doesn't
   OOM." (The `sft_driver.py --source dataset` path remains in the repo as
   tooling but is unused/unverified; the earlier hang was a data-loader issue,
   not a trainer/config issue.)

2. **Current live box:** `w57o7m3` (healthy, GPU-bind-gated). Trainer idle and
   ready. Kill with `tmux kill-session -t trainer` on each node to reclaim GPUs.

3. **EP=8 vs EP=16:** EP=16 is impossible at PP=4/DP=1 (only 8 GPUs/stage). If
   you want EP=16 you'd need DP=2 (8 nodes) or a different PP. EP=8 fit fine.
