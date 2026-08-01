# LPS-1003 follow-up items (rewritten 2026-07-31 evening after the day-session
# root-fix hunt; supersedes the morning version)

State: trigger + mask PROVEN (conn unset fires 10/10 devbox boots + 6/6 prod;
conn=1 suppresses 0/13+ boots). Root DEFECT narrowed to two kernels but NOT
yet named; NO root fix shipped. Jack's direction: ship a FIX, not the
mitigation. Evidence chain: parity/NOTEBOOK.md (esp. "DAY SESSION
2026-07-31" arm matrix); corrected status: HANDOFF.md top block.

1. **Root fix (the objective).** Next step is built and ready: arm H
   double-exec discriminator (`bash parity/devbox_streamfix2.sh H` on
   tj-3y0gjkq leader; harness6 runs the cuDNN indexer fwd and FlashMLA twice
   per call, bitwise compare — the self-disagreeing kernel during a fired
   rep0 is the defect). Then:
   - if indexer fwd: patch the wheel to swap the CLC dynamic persistent tile
     scheduler for cutlass-dsl's StaticPersistentTileScheduler (same method
     surface; NOTEBOOK has pointers) + NVIDIA report;
   - if FlashMLA: kernel patch / upstream (deepseek-ai/FlashMLA nv_dev).
   Then the final A/B per Jack's spec: with fix reps 0-9 clean band
   (0.760-0.771), without fix rep0 fires (harness: devbox_streamfix2.sh).
2. **Latent-defect PR (independent of LPS-1003):** top-k TVM-FFI env-stream
   launch pin — branch `jackrao/dsa-topk-stream-pin`, vendored wheel
   `+dsatopk3` (+patch file, README, 2 regression tests, uv.lock; +dsatopk2
   is PR #821's). Proven latent-by-construction, proven NOT this bug — word
   the PR accordingly.
3. **Prod protection decision (Jack):** conn=1 is validated on 5wolkzw boot
   windows (0/16 reps vs 13/16 before) and remains live there via LWS patch.
   NOT shipped fleet-wide per Jack's fix-not-mitigation call. If customer
   exposure appears before the root fix lands, this is the known-good lever
   (steady-state efficacy inferred, not measured — boot-window data only).
4. **Mudith's live trainers on pre-#814 image** (zq8ykgw, v31gz13, 232l1xw,
   2qj08pw = trainer-cuda13-sm103-5a4ae4d, crash-prone; session-reuse skips
   image resolution). Nudge to recycle onto 0e0b65a.
5. **Deprovision trainer 5wolkzw** (idle, mitigated env, tracer initContainer
   in LWS; no DELETE API — internal tooling). Session 8w6k4y3 reusable for
   prod re-tests until then.
6. Unfiled from before (still valid): init_trainer_server 3rd-in-process
   REBUILD deadlock (reproduced on prod 2026-07-30); Issue-3 crash-restart
   desync is LPS-1013.
7. **Devbox-up env parity:** run_trainer_node.sh hand-copied exports forked
   devbox behavior from prod (see parity/ENV_PARITY.md). Short-term: prodenv
   script + check_env_parity.sh (on devbox CPFS). Long-term: make devbox-up
   derive the trainer env from the image's own launch path; fold into the
   devbox-up repo/skill.
