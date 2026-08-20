# SELF-REVIEW — diff pass on both lineages vs origin/main

weierstrass, 2026-08-13 night. Read-only pass; no code changed.
Lineages reviewed:

- **bringup** `jackrao/lps-1062-pp2cp8ep8` @ b8d868ff vs origin/main
  (11 commits, 17 files, +836/−187) — includes volta's M=N group and
  poincare's L1 hunk, which are lebesgue's lane to shape; findings about
  them here are coordination notes, not a substitute for lebesgue's
  self-review (SELF_REVIEW_MN_PACKING.md).
- **export** `jackrao/lps-1062-pp2-export` @ db5d1826 vs origin/main
  (4 commits, 4 files, +353/−51).

Method: full `git log` + per-commit `git show` read of every commit in my
lane; call-site and consumer greps for every new symbol/env var; debug
leftover scan (`TODO`/`FIXME`/`XXX`/`breakpoint`/`pdb`/stray `print(`) on
both full diffs — **clean**, none added by either lineage (the export
test's `print` of the key/shape table is intentional diagnostics, matching
the existing `test_pp_adapter_export.py` pattern).

## Findings — bringup lineage

1. **[coordination, not a bug] The VPP-iterator/M=N seam is real and
   marked.** 8c13ed31's two call sites in `training_runner.py` are exactly
   the lines the M=N merge (73c24b00) rewrote; 8b0ef108's comment marks
   the seam. Merged form verified in-tree: both call sites route through
   `_schedule_data_iterator(microbatches, len(self._stack.model_list))`.
   Whichever PR lands second reconciles there. Flagged to lebesgue; they
   acked and carry the landing-order note in their draft.
2. **[scope] b8d868ff (poincare's `overlap_dispatch_backward_with_experts_
   wgrad` plumbing) sits on the pushed branch tip but has never run on
   hardware** (its first boot is queued in E2_REVISED_SPEC.md; doppler's
   fresh-eyes review approved it and its 14/14 Mac-harness tests pass).
   Any PR cut from the branch tip silently includes it. Morning PRs must
   be cut at explicit commits, and whoever packages it (lebesgue) must
   label it unvalidated-on-hardware.
3. **[nit, follow-up] `virtual_pipeline_parallel_size > 1` with
   `pipeline_parallel_size == 1` is a silent no-op** (61c41d9e): the
   layout lookup misses and the provider field stays unset, so the user
   gets a non-interleaved schedule with no signal. House rule 3 argues
   for a parse-time validator (vpp>1 requires pp>1). Not blocking —
   tonight's validated configs all have PP2 — but worth a follow-up line
   in INFRA-B or an issue.
4. **[nit] `BT_PROFILE_RANKS` edge cases** (ad39a97d): garbage entries
   (`"0,x"`) raise `ValueError` at boot — fail-loud, acceptable. But a
   blank-after-filter value (`","`) yields an *empty* rank set, which
   silently traces nothing. Minor; a one-line guard would make it loud.
5. **[verified] 21d0c578 call-site sweep**: `_validate_thd_context_
   parallelism` has exactly one production caller
   (`_apply_thd_cp_provider_overrides`, updated) and four test call sites
   (all updated). Non-DSA rejection path preserved and re-pinned by tests.
   The guard docstring update Jack required is in the commit.
6. **[verified] 760021be**: `os` already imported in `controller.py`; env
   read happens per `optim_step` call (cheap dict lookup, no caching
   concern); default path byte-identical; gloo/CPU testable, tests pass in
   CI shape.
7. **[verified] ad39a97d wiring**: `rank_set` is consumed at
   `profile_manager.py:300` (`recording = rank in rank_set`) — the env
   actually reaches the tracing decision; default `{0}` unchanged.
8. **[branch hygiene]** Branch is 2 behind origin/main (f51fae2e DSv4
   repin #1013, 05f3fafc CI permissions #1012) — no file overlap with the
   package; routine rebase at cut time.

## Findings — export lineage

1. **[deliberate divergence — the one reviewers must understand]**
   8a4ae08b rewrites `_validate_thd_context_parallelism` (raise→warning,
   G1) and `_apply_thd_cp_provider_overrides` (generic-GPT passthrough,
   G2), and rewrites the same `test_cp_thd_dispatch.py` cases that
   21d0c578 rewrites differently on the bringup branch. **Merging both
   branches' `megatron_config.py` is a conflict and a semantic clash.**
   Resolution is decided, not open: the bringup DSA-only exemption is the
   mainline version; the export branch's relaxations are test-branch-only.
   Both PR drafts say so in their bodies.
2. **[evidence-not-test] L2b is not a committed test.** The 1.3% worst
   per-tensor std-deviation number comes from an on-box comparison whose
   artifacts are archived (`export_test/evidence/ws_ref_pp1cp1/`,
   `ws_test_pp2cp2/`, `workers_*_l2b.log`). This matches the design
   ("report, don't gate", EXPORT_TEST.md §6), but the PR draft cites it as
   evidence paths, not as a runnable check. If Jack wants it runnable,
   that is a small follow-up (non-gated statistical test).
3. **[verified] db5d1826 toggle mechanics**: strict `== "1"` parse;
   default preserves `async_save=True`; `async_ckpt_use_cpu_shm=True`
   coexisting with `async_save=False` is not just source-plausible but
   verified by execution at both scales (small-model probe green;
   production-scale 70.8s/537MB save completed). `os` and `logger`
   already present in `megatron_config.py` on this branch.
4. **[verified] L2a skip is load-bearing, not lazy**: the skip reason
   names the finding and the re-enable condition; the mkdir fix (boot
   dirs pre-created before `mb_cluster`) is correct against the harness's
   expectations.
5. **[nit] G1 warning text** says "under validation (LPS-1062)" — right
   for a test branch; if any hunk were ever cherry-picked mainline the
   wording would need to change. Moot under the test-branch-only decision;
   noted so a future cherry-pick doesn't carry it verbatim.
6. **[verified] `server/tests/helpers.py`** change is additive-only
   (`attention_backend` kwarg, default `None`, omitted from the config
   blob when unset) — no existing caller affected.

## Cross-lineage

- No file is modified by both lineages except `megatron_config.py` and
  `test_cp_thd_dispatch.py` — both covered by finding export-1.
- Neither lineage touches the other's evidence claims: the export verdict
  (Qwen3, dense) does not cover GLM-5.2's MoE/EP export branches, and the
  infra PR's validation evidence (GLM-5.2 training) does not cover export.
  The drafts cite them as complementary, not overlapping.
