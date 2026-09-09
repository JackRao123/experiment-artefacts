# GLM-5.3 main: 262,144 tokens, CP8/EP8, B300

Requested 2026-09-09: profile tip of trainers main with one warmup, five
unprofiled controls, one memory-profiled step and one runtime-profiled step;
retrieve artifacts to the Mac and open the runtime capture in Perfetto.

Main was resolved from GitHub at run setup to
`33d19a3542c3d67553e9dc9d30381d2bdf6db31b`.
Bridge pin: `6646e5f7e4440a740338f1c12946ef7be4cde427`.
Megatron-Core pin: `494c1d5725959c31b72b1b65e1c515339caf3466`.

Devbox `tj-32vj99q`: one node, 8 HGX B300 GPUs (reported as NVIDIA L20D),
1,100 W max power each. The prior Qwen trainer was stopped with its lifecycle
script before launch. GLM uses all eight GPUs, TP1/PP1/CP8/EP8/ETP1/DP1.

Model snapshot:
`/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6`.
All 141 shards were present and nonempty at preflight.

Configuration follows the earlier GLM 262k run: native-FP8 routed experts,
LoRA rank/alpha 32, HybridEP, flash attention, full uniform one-layer recompute.
See the exact `trainer-config.json`.

The new detached checkout is `/root/glm53-main-262k-20260909/trainers`.
It was cloned from the existing golden repository, with a Git bundle carrying
the main delta. Pinned submodules were initialized recursively from existing
local clones. Existing user edits in the prior checkout were preserved.

Dependencies and submodule pins did not change between the previously loaded
main and this main. The existing CUDA 13 environment at
`/root/.devbox-venvs/server` is reused via a symlink. PYTHONPATH points the
trainer, models, weight-sync, Bridge and Core at the new checkout. Preflight
verified all four trainer/model package origins resolve to the new checkout.
No source changes were applied to main; the only untracked entry is `.venv`.

Copies of devbox-up-generated lifecycle scripts are under `lifecycle/` locally
and `/root/glm53-main-262k-20260909/.devbox_up/` remotely. Their paths are
relocated to node-local storage, as the shared filesystem rejected writes
earlier in this session. The node runner adds the explicit PYTHONPATH.
The original lifecycle scripts and golden checkout are untouched.

The run-local driver is copied from `../../tools/profile_driver.py`; its
output directory is the only functional change. It reuses the same synthetic
datum across all windows (seed 0xB300, 154,880-token vocabulary).
The copied `mfu.py` retains the existing GLM estimate and HGX B300 2.25e15 peak.
Its output remains an analytic GLM estimate, not a new GLM-5.3 FLOP audit.

Command on the healthy devbox:

```bash
/root/.devbox-venvs/server/bin/python -u \
  /root/glm53-main-262k-20260909/profile_driver.py \
  --label glm53-main-262k-cp8ep8-c5 --seq-len 262144 --datums 1 \
  --num-gpus 8 --lora-rank 32 --control-repeats 5 \
  --memory-profile --runtime-profile
```

Lifecycle commands on the devbox:

```bash
bash /root/glm53-main-262k-20260909/.devbox_up/wait_trainer_health.sh
bash /root/glm53-main-262k-20260909/.devbox_up/stop_trainer.sh
```
