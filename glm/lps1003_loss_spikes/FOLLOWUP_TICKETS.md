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

## Added 2026-08-02 late PM (post ARMING_MECHANISM.md — arming question resolved)

8. **NVIDIA report reframed (supersedes the driver-report item):** no
   driver bug exists; report withdrawn. Two reports to send instead
   (Jack's call on send/no-send, inputs = rearm/ARMING_MECHANISM.md +
   rearm/arming/):
   - **cudnn-frontend / DSA wheel team:** `utils/runtime.py::
     torch_stream_context` (and the inline copy in `sdpa/bwd/api.py:420`)
     builds `torch.cuda.ExternalStream(int(handle))`; for handle 0 torch
     returns a rotating POOL stream, so every "run on caller's stream"
     torch op actually runs on an unordered private stream. 60+ call
     sites (indexer fwd/bwd, score_recompute, sdpa bwd) carry the
     landmine; indexer-forward's -inf prefill was merely the site with
     catastrophic blast radius. Most other sites are currently no-ops on
     our path (contiguous inputs, TMA-aligned shapes) — latent, not safe.
   - **PyTorch upstream (optional):** `torch.cuda.ExternalStream(0)`
     silently treats stream_ptr=0 as "absent" and returns a pool stream;
     should raise or alias the default stream.
9. **PR #875 narrative touch-up before merge:** the patch-header comment
   and test docstring say the default-stream arrangement "loses inter-op
   ordering with CUDA_DEVICE_MAX_CONNECTIONS unset". Mechanically the fix
   is right and stays; wording should say the old arrangement put the
   prefill on a torch pool stream via ExternalStream(0) (never ordered),
   and conn is a masking variable, not part of the defect.
10. **Wheel-glue audit:** decide whether to patch `torch_stream_context`
    itself in our vendored wheel (guard: `if int(current_stream) == 0:
    yield; return`) to defuse the other 60+ sites until NVIDIA ships a
    fixed wheel. The indexer-bwd sites run under `backend_stream` —
    verify what that resolves to on our path.

## Added 2026-08-03 (post PR-review cycle — see PR875_REVIEW_0803.md)

Status updates to the items above:

- **Item 1 (root fix): DONE.** Wheel revised to `+dsatopk5` in review
  (try/except → fail loudly; behavior-neutral for first-party callers).
  Jack's final A/B spec satisfied in CI-runnable form: minimal single-GPU
  repro found (prefill > 2^31 bytes → two fill launches → second races;
  2.25 GiB 6/6 unfixed vs 2.0 GiB 0/6), productionised as
  test_indexer_fwd_bitwise_stable_across_cold_starts — fix removed FAILS
  at rep0, +dsatopk5 15/15. PR rebased onto main, /test pr-tests green.
- **Item 2 (topk stream-pin PR): fold into item 11** — don't ship
  +dsatopk3 as a separate wheel patch.
- **Item 3 (prod protection) — framing corrected per Jack:** never say
  "we deployed conn=1". 5wolkzw was the investigation's prod TEST
  deployment only (conn=1 via LWS patch to validate the lever; 0 replicas
  since). conn=1 remains the known-good emergency lever, unshipped.
- **Item 4 (live trainers) — still open, target updated:** 08-03 check:
  232ln0w / 4w5yx73 / v31gz13 / zq8y7pw all on 5a4ae4d (47 commits
  pre-fix), no conn=1 → exposed. Recycle target is now an image from the
  trainer-cuda13-sm103 tip (3ef3b1952 carries +dsatopk5), NOT 0e0b65a.
- **Item 9 (patch narrative rewording): STILL OPEN** — the +dsatopk5
  rebuild kept the module-level comment's "loses inter-op ordering with
  CUDA_DEVICE_MAX_CONNECTIONS unset" framing; next wheel touch should
  reword to the ExternalStream(0)-pool-stream mechanism (conn is a masking
  variable, not the defect).

- **Item 8, PyTorch-upstream bullet: MOOT — already reported and fixed
  upstream.** pytorch/pytorch#182960 (also hit independently by Triton,
  triton-lang/triton#10186), fixed by PR #183258 merged 2026-05-20:
  `ExternalStream(0)` now aliases the NULL/default stream, behaviorally
  identical to `get_stream_from_external(0)` and `default_stream()`.
  Verified in-tag: v2.11.0/v2.12.0/v2.12.1 carry the truthiness path;
  FIXED in v2.13.0, released 2026-07-08 (Stream.cpp Py_None-vs-int
  parsing). Our stack (torch 2.11) stays exposed until a torch bump.
  Implications: (a) no report to file; (b) the wheel-team
  report gets stronger — cite #183258 as upstream's own admission the API
  was broken, and hand them `torch.cuda.get_stream_from_external(0)` as
  the drop-in remediation correct on ALL torch versions; (c) note for the
  wheel team that on torch ≥2.13 their handle-0 sites silently change
  behavior (pool stream → legacy default stream) with no wheel change —
  another reason to fix explicitly rather than inherit torch semantics.

11. **Caller-side dedicated-stream helper + family audit (adopts
    pstefa1707's review suggestion; complements/absorbs items 2 & 10):**
    caller-side is PROVEN viable on the pristine wheel (2.25 GiB
    double-exec: default-stream control 6/6; ambient side-stream 0/6;
    explicit stream arg 0/6 — only handle 0 is poisoned). One
    dedicated-stream wrapper around the six `_cudnn_dsa.*` call sites in
    dsa_cudnn_kernels.py (indexer fwd/bwd, dense bwd, top-k, sparse/dense
    score_recompute) defuses the whole glue family without more wheel
    patches; verify each site with the double-exec probe at prod geometry
    (probe scripts: devbox /root/lps1003_review/). Mind allocator
    stream-tagging if the wrap is ambient (multi-GiB outputs get tagged to
    the side stream → downstream consumers need lifetime discipline).

## Added 2026-08-03 late PM (cudnn-frontend develop-pin migration)

12. **Items 8/10/11 MOOT — upstream fixed it themselves.** cudnn-frontend
    PR #354 (merged into develop 2026-07-09, three weeks before our
    root-cause) switches torch_stream_context to get_stream_from_external —
    the exact drop-in we planned to report. trainers PR #910 pins develop @
    74785165 (1.27.0+git7478516), deletes both server/patches, and retires
    +dsatopk5. Validated on tj-q8x5ky3 (/root/cfe_migration/): A/B pristine
    12/12 fire @ 4GiB out vs develop 0/12; 100-rep identical-fwd soak clean
    kernel+module; parity failure set identical to +dsatopk5 control.
    Item 9 (patch narrative rewording) also moot — patches deleted.
