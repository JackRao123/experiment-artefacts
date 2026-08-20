# SELF-REVIEW — jackrao/lps-1062-pp2-packing @ 8b0ef108 (+ b8d868ff) vs origin/main

lebesgue, 2026-08-13 ~00:0x CDT. Findings only — no fixes (read-only lane).
Scope: the 4 commits on `jackrao/lps-1062-pp2-packing` (merge-base
`df831501`: `36c3c8f4` packing, `b34b7aed` fixtures, `d1c939c3` M=N,
`8b0ef108` seam docs) plus the L1 plumbing commit `b8d868ff` on
`jackrao/lps-1062-pp2cp8ep8`.

## Mac harness run (deliverable 3, recorded)

Command (from the artefacts tools dir, worktree HEAD `8b0ef108`):

```
~/Documents/trainers/server/.venv/bin/python tools/run_server_tests_mac.py \
  <wt-pp2-packing>/server/tests/unit/dp_worker/api/test_cp_thd_slicing.py \
  <wt-pp2-packing>/server/tests/unit/dp_worker/api/test_cp_thd_dispatch.py \
  <wt-pp2-packing>/server/tests/unit/dp_worker/api/test_ce_loss.py \
  <wt-pp2-packing>/server/tests/unit/dp_worker/api/test_router_replay_dp_consensus.py
```

Result: **142 passed / 5 failed / 40 deselected (`not gpu`), 16.3s.**
Full log: `/var/folders/1m/bllgmvfs6t7czgc4w3l_h7f00000gn/T/opencode/harness_run.log`.

The 5 failures are all in `test_router_replay_dp_consensus.py` and all fail
*before any test logic*, on any commit, because the Mac has no CUDA and the
harness's import stubs don't cross process boundaries:

- `test_two_rank_gloo_replay_consensus[consensus]` / `[peer_failure]` —
  torch.multiprocessing.spawn children die in bootstrap (the documented
  spawn-stub artifact).
- `test_r3_rejects_losses_that_never_arm[cross_entropy]` / `[dpo]` and
  `test_missing_routes_warns_on_every_batch_not_once` — the runner touches
  `torch.cuda.current_device()` before the guard under test fires. Verified
  the CUDA call precedes the guard on **both** origin/main (`:546`) and HEAD
  (`:600`): identical-on-base by construction, not a branch regression.
  (Doc imprecision: the harness docstring says "5 spawn-based" — actually 2
  spawn + 3 CUDA-touch. Same conclusion, wrong mechanism label; artefacts
  tool, not PR code.)

Green for PR purposes. Still open (volta's flagged gap, not re-run tonight):
on-box `make test-server` on the M=N head.

## Findings

**F1 (cosmetic, docs) — stale phrasing in `_thd_partition_pad_to_length`
docstring** (`training_runner.py`, added in `d1c939c3`): says "every
per-partition schedule call" presents the activation shape — that is the
pre-M=N framing. Under M=N there is one schedule call per op; the uniformity
applies per-microbatch within that call. Mechanism and gate are correct;
only the wording predates the restructure.

**F2 (minor, footgun) — `BT_PACK_PAD_TO_MAX` truthiness**
(`training_runner.py`): `os.environ.get("BT_PACK_PAD_TO_MAX")` treats any
non-empty string as ON — including `BT_PACK_PAD_TO_MAX=0`, which an operator
would reasonably expect to mean OFF. Deliberate env gate (parity A/B
vehicle; flagged in PR §7), but the parse convention should be a reviewer
decision. Also noted: env is read per call, but effectively
process-lifetime (relaunch to change) — the docstring says so.

**F3 (landing-order hygiene) — `8b0ef108` seam comment names an unmerged
branch** ("gibbs's VPP branch") at the M=N call site. Deliberate
merge-conflict marker; coordinated with weierstrass's infra PR (which
carries the VPP iterator fix `8c13ed31` over the same lines). Whichever PR
lands second should reconcile the comment to describe merged reality.

**F4 (history note) — `36c3c8f4` commit message describes the pre-M=N
convention** ("one schedule call per THD partition with num_microbatches=1")
as present tense. Accurate when written (Aug-10); superseded by `d1c939c3`
on the same branch. Only matters for commit-by-commit readers.

**F5 (scope note) — `b8d868ff` is not on this branch.** Verified it
cherry-picks clean (`git apply --check` against `8b0ef108` passes). It is
default-off plumbing with no on-box A/B yet (L1 boot queued tonight).
Include-as-fifth-commit vs stacked follow-up is Jack's call; the PR draft
(§3 last block) presents both.

## Checks with negative results (nothing to flag)

- No `print(`/`breakpoint`/`pdb`/`TODO`/`FIXME`/`XXX`/`HACK`/`DEBUG`
  anywhere in the branch diff.
- Both worktrees clean; `jackrao/lps-1062-pp2-packing` has no unpushed
  commits (== origin).
- Empty-DP-slice and `forward_only` gating in the new finalize path matches
  the old post-loop semantics exactly (incl. the `finalize is not None`
  guard).
- `micro_batch_size=microbatches[0].input_ids.shape[0]` takes partition 0 as
  representative — verified homogeneous by construction (every THD row is
  `(1, L)`); same value the old per-partition calls passed.
- New last-stage length guard converts the old cryptic unpack failure into
  an explicit `ValueError` — deliberate; the updated test
  (`test_run_forward_backward_requires_last_stage_metric`) encodes it.
- All test-file changes are explicable: dispatch suite rewritten to the M=N
  contract, 7 new `pad_to_length_*` cases, and 4 one-line fixture mirrors of
  the cached `_pipeline_parallel_world_size` (`b34b7aed`).
- `b8d868ff` diff reviewed: pop-before-splat with no default (fail-loud),
  provider write lands on the object the model reads, trainer validator
  mirrors mcore's exclusions, no bridge-side overwrite path. Matches
  doppler's on-record APPROVE (NOTEBOOK 21:2x).
